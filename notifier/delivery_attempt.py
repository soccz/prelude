"""Durable, append-only send intents; an unresolved intent never authorizes retry.

Writers must hold the sender's snapshot lock. Readers take no lock and perform
no writes. This is conservative at-most-once *attempt authorization*, not an
exactly-once Telegram guarantee: a crash before HTTP may still require manual
reconciliation. Local checksums are not signatures or protection from deletion
of the entire evidence tree. Do not deploy the first journal mid-send-window.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import os
from pathlib import Path
import re
import stat
import uuid

from notifier.delivery_receipt import (
    RECEIPT_INTEGRITY_ACTIVATION_DATE, _LIVE_SEND_WINDOWS, _validate_document,
    read_delivery_receipt, receipt_path,
)
from notifier.telegram import telegram_error_is_ambiguous
from ops.artifact_provenance import canonical_json_bytes, strict_json_object_bytes

INTENT_SCHEMA = "recommend_delivery_attempt.intent.v1"
RESULT_SCHEMA = "recommend_delivery_attempt.result.v1"
_FILE = re.compile(r"([0-9]{6})\.(intent|result)\.json")
_SHA = re.compile(r"[0-9a-f]{64}")
_ID = re.compile(r"[0-9a-f]{32}")
_INTENT_FIELDS = {"schema", "sequence", "attempt_id", "identity", "attempted_at",
                  "message_sha256", "previous_result_sha256", "previous_receipt_sha256",
                  "previous_receipt", "integrity_sha256"}
_RESULT_FIELDS = {"schema", "sequence", "attempt_id", "intent_sha256", "receipt_sha256",
                  "receipt", "recorded_at", "integrity_sha256"}


class DeliveryAttemptError(RuntimeError):
    """Unsafe or unresolved delivery evidence: do not call the transport."""


def _require(condition, reason):
    if not condition:
        raise DeliveryAttemptError(reason)


def _sha(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _seal(value):
    return {**value, "integrity_sha256": _sha(value)}


def _time(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    _require(isinstance(value, datetime) and value.tzinfo is not None
             and value.utcoffset() is not None, "attempt clock must be timezone-aware")
    return value.astimezone(timezone.utc)


def _identity(snapshot):
    return {"snapshot_id": snapshot.get("snapshot_id"), "snapshot_path": snapshot.get("snapshot_path"),
            "asof": snapshot.get("asof"), "slot": snapshot.get("slot"),
            "model_id": (snapshot.get("model") or {}).get("id"),
            "ranking": (snapshot.get("model") or {}).get("ranking", "R1")}


def _attempt_time(value, snapshot, now):
    stamp = _time(value)
    local = stamp.astimezone(timezone(timedelta(hours=9)))
    start, end = _LIVE_SEND_WINDOWS[snapshot["slot"]]
    _require(local.date().isoformat() == snapshot["asof"]
             and start <= local.timetz().replace(tzinfo=None) < end,
             "attempt timestamp outside its decision date/slot window")
    _require(stamp <= now, "future attempt timestamp")
    completed = snapshot.get("decision_completed_at")
    if completed is not None:
        _require(stamp >= _time(completed), "attempt predates snapshot completion")
    return stamp


def _bare(receipt):
    return None if receipt is None else {k: v for k, v in receipt.items() if k != "receipt_path"}


def _resolution(receipt):
    if receipt["delivery_ok"]:
        return "delivered"
    if receipt.get("telegram_messages") or telegram_error_is_ambiguous(receipt.get("error")):
        return "ambiguous"
    return "clear_failure"


def _checked_receipt(receipt, snapshot, path):
    receipt = _bare(receipt)
    if receipt is not None:
        _validate_document(receipt, snapshot, path)
    return receipt


def _matches(receipt, intent):
    if receipt is None or _sha(receipt) == intent["previous_receipt_sha256"]:
        return False
    if receipt["attempted_at"] != intent["attempted_at"]:
        return False
    if datetime.fromisoformat(receipt["asof"]).date() >= RECEIPT_INTEGRITY_ACTIVATION_DATE:
        if receipt.get("message_sha256") != intent["message_sha256"]:
            return False
    return True


class DeliveryAttemptJournal:
    """Descriptor-bound journal; caller serializes the whole send transaction."""

    def __init__(self, snapshot, *, receipt_root=None, create=False):
        self.snapshot = snapshot
        self.receipt_root = receipt_root
        self.receipt = receipt_path(snapshot, root=receipt_root)
        self.path = self.receipt.parent / f"{self.receipt.stem}.attempts"
        self.chain = []
        self.fd = None
        # Resolve only ancestors above the receipt root, never a root/date/leaf
        # symlink. The project's intentional /home/.../22tb alias remains valid.
        root = Path(os.path.abspath(self.receipt.parent.parent))
        anchor = root.parent
        missing = [root.name, self.receipt.parent.name, self.path.name]
        while not anchor.exists() and not anchor.is_symlink():
            missing.insert(0, anchor.name)
            anchor = anchor.parent
        anchor = anchor.resolve(strict=True)
        try:
            fd = os.open(anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            self.chain.append((anchor, fd))
            for part in missing:
                child = anchor / part
                if create:
                    try:
                        os.mkdir(part, 0o700, dir_fd=fd)
                    except FileExistsError:
                        pass
                    # Sync every parent entry, including pre-existing entries
                    # another concurrent creator may not have synced yet.
                    os.fsync(fd)
                try:
                    child_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                except FileNotFoundError:
                    if not create:
                        self.verify()
                        return
                    raise
                self.chain.append((child, child_fd))
                anchor, fd = child, child_fd
            self.fd = fd
            self.verify()
        except BaseException:
            self.close()
            raise

    def close(self):
        while self.chain:
            _, fd = self.chain.pop()
            os.close(fd)
        self.fd = None

    def verify(self):
        for path, fd in self.chain:
            observed, held = path.lstat(), os.fstat(fd)
            _require(stat.S_ISDIR(observed.st_mode)
                     and (observed.st_dev, observed.st_ino) == (held.st_dev, held.st_ino),
                     "attempt directory changed or became a symlink")

    def _read(self):
        if self.fd is None:
            self.verify()
            return {}, []
        self.verify()
        names = sorted(os.listdir(self.fd))
        documents, evidence = {}, []
        for name in names:
            match = _FILE.fullmatch(name)
            _require(match is not None and int(match[1]) > 0, "unexpected attempt journal file")
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            fd = os.open(name, flags, dir_fd=self.fd)
            try:
                before = os.fstat(fd)
                _require(stat.S_ISREG(before.st_mode) and before.st_uid == os.geteuid(), "unsafe attempt evidence file")
                chunks = []
                while chunk := os.read(fd, 65536):
                    chunks.append(chunk)
                raw = b"".join(chunks)
                after = os.fstat(fd)
                current = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
                def token(s):
                    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns
                _require(token(before) == token(after) == token(current), "attempt evidence changed while reading")
                document = strict_json_object_bytes(raw, source=self.path / name)
                fields = _INTENT_FIELDS if match[2] == "intent" else _RESULT_FIELDS
                _require(set(document) == fields, "attempt evidence fields mismatch")
                expected = INTENT_SCHEMA if match[2] == "intent" else RESULT_SCHEMA
                _require(document["schema"] == expected and type(document["sequence"]) is int
                         and document["sequence"] == int(match[1]), "attempt schema/sequence mismatch")
                _require(isinstance(document["attempt_id"], str) and _ID.fullmatch(document["attempt_id"]), "invalid attempt ID")
                _require(document["integrity_sha256"] == _sha({k: v for k, v in document.items() if k != "integrity_sha256"}),
                         "attempt evidence checksum mismatch")
                documents[int(match[1]), match[2]] = document
                evidence.append({"path": str(self.path / name), "sha256": hashlib.sha256(raw).hexdigest()})
            finally:
                os.close(fd)
        _require(sorted(os.listdir(self.fd)) == names, "attempt journal changed while reading")
        self.verify()
        return documents, evidence

    def inspect(self, *, now):
        now = _time(now)
        documents, evidence = self._read()
        current = _checked_receipt(read_delivery_receipt(self.snapshot, root=self.receipt_root), self.snapshot, self.receipt)
        sequences = sorted({seq for seq, _ in documents})
        result = {"state": "absent", "resolution": None, "latest_attempt_id": None,
                  "attempted_at": None, "attempt_count": len(sequences),
                  "journal_sha256": _sha(evidence), "reason": "no attempt journal evidence",
                  "evidence_files": evidence}
        _require(sequences == list(range(1, len(sequences)+1)), "attempt sequence gap")
        previous = None
        seen_ids = set()
        for sequence in sequences:
            intent = documents.get((sequence, "intent"))
            terminal = documents.get((sequence, "result"))
            _require(intent is not None, "orphan attempt result")
            _require(intent["identity"] == _identity(self.snapshot), "attempt snapshot identity mismatch")
            _require(intent["attempt_id"] not in seen_ids, "duplicate attempt ID")
            seen_ids.add(intent["attempt_id"])
            attempt_time = _attempt_time(intent["attempted_at"], self.snapshot, now)
            _require(attempt_time <= now and attempt_time.isoformat() == intent["attempted_at"], "future/noncanonical attempt time")
            _require(isinstance(intent["message_sha256"], str) and _SHA.fullmatch(intent["message_sha256"]), "invalid attempt message checksum")
            prior = _checked_receipt(intent["previous_receipt"], self.snapshot, self.receipt)
            _require(intent["previous_receipt_sha256"] == (None if prior is None else _sha(prior)), "previous receipt checksum mismatch")
            _require(intent["previous_result_sha256"] == (None if previous is None else previous["integrity_sha256"]), "previous result chain mismatch")
            if prior is not None:
                _require(_resolution(prior) == "clear_failure", "attempt followed a delivered/ambiguous receipt")
                # Exact clock ties reject retry; never fabricate a later time.
                _require(attempt_time > _time(prior["recorded_at"]), "retry clock must advance beyond previous receipt")
            if previous is not None:
                _require(prior == previous["receipt"], "retry baseline differs from previous result")
                _require(attempt_time >= _time(previous["recorded_at"]), "retry clock precedes previous result")
            if terminal is not None:
                _require(terminal["attempt_id"] == intent["attempt_id"]
                         and terminal["intent_sha256"] == intent["integrity_sha256"], "result/intent binding mismatch")
                receipt = _checked_receipt(terminal["receipt"], self.snapshot, self.receipt)
                _require(receipt is not None and terminal["receipt_sha256"] == _sha(receipt)
                         and _matches(receipt, intent), "result receipt does not resolve this attempt")
                _require(attempt_time <= _time(receipt["recorded_at"]) <= _time(terminal["recorded_at"]) <= now,
                         "result clock chronology mismatch")
            if sequence != sequences[-1]:
                _require(terminal is not None, "attempt follows unresolved intent")
            previous = terminal
        if sequences:
            intent = documents[sequences[-1], "intent"]
            terminal = documents.get((sequences[-1], "result"))
            result.update(latest_attempt_id=intent["attempt_id"], attempted_at=intent["attempted_at"])
            if terminal is not None:
                _require(current == terminal["receipt"], "current receipt differs from completed journal")
            if _matches(current, intent):
                _require(_time(current["recorded_at"]) <= now, "receipt completion is future-dated")
                result.update(state="resolved", resolution=_resolution(current), reason="matching receipt resolves latest intent")
            else:
                # An old clear-failure receipt can legitimately remain after a
                # new attempt crashed, but it must not resolve that new intent.
                _require(current == intent["previous_receipt"], "unrelated current receipt for pending intent")
                result.update(state="pending", reason="delivery_uncertain: latest intent has no matching receipt")
        self.verify()
        return result

    def _publish(self, name, document):
        _require(self.fd is not None, "attempt directory is absent")
        self.verify()
        temporary = f".{name}.{uuid.uuid4().hex}.tmp"
        fd = None
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self.fd)
            content = canonical_json_bytes(document) + b"\n"
            with os.fdopen(fd, "wb") as handle:
                fd = None
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, name, src_dir_fd=self.fd, dst_dir_fd=self.fd, follow_symlinks=False)
            os.fsync(self.fd)
            self.verify()
        finally:
            if fd is not None:
                os.close(fd)
            try:
                os.unlink(temporary, dir_fd=self.fd)
                os.fsync(self.fd)
            except FileNotFoundError:
                pass

    def begin(self, message, *, now, receipt):
        _require(isinstance(message, str) and bool(message), "attempt message must be nonempty text")
        now = _time(now)
        _attempt_time(now, self.snapshot, now)
        state = self.inspect(now=now)
        _require(state["state"] != "pending", state["reason"])
        current = _checked_receipt(read_delivery_receipt(self.snapshot, root=self.receipt_root), self.snapshot, self.receipt)
        _require(current == _bare(receipt), "receipt changed before attempt authorization")
        if current is not None:
            _require(_resolution(current) == "clear_failure", "delivered/ambiguous receipt forbids retry")
            _require(now > _time(current["recorded_at"]), "retry clock must advance beyond previous receipt")
        documents, _ = self._read()
        count = state["attempt_count"]
        previous = documents.get((count, "result"))
        if count and previous is None:
            # A receipt may have been durably saved just before result writing
            # crashed. Archive only that verified matching clear failure.
            self.complete(documents[count, "intent"], receipt=current, now=now)
            documents, _ = self._read()
            previous = documents[count, "result"]
        if previous is not None:
            _require(now >= _time(previous["recorded_at"]), "retry clock precedes previous result")
        sequence = count + 1
        _require(sequence <= 999999, "attempt sequence exhausted")
        intent = _seal({"schema": INTENT_SCHEMA, "sequence": sequence, "attempt_id": uuid.uuid4().hex,
                        "identity": _identity(self.snapshot), "attempted_at": now.isoformat(),
                        "message_sha256": hashlib.sha256(message.encode()).hexdigest(),
                        "previous_result_sha256": None if previous is None else previous["integrity_sha256"],
                        "previous_receipt_sha256": None if current is None else _sha(current),
                        "previous_receipt": current})
        self._publish(f"{sequence:06d}.intent.json", intent)
        return intent

    def complete(self, intent, *, receipt, now):
        now = _time(now)
        documents, _ = self._read()
        sequence = intent["sequence"]
        _require(documents.get((sequence, "intent")) == intent
                 and sequence == max(seq for seq, _ in documents), "completion is not for latest saved intent")
        receipt = _checked_receipt(receipt, self.snapshot, self.receipt)
        current = _checked_receipt(read_delivery_receipt(self.snapshot, root=self.receipt_root), self.snapshot, self.receipt)
        _require(receipt == current and _matches(receipt, intent), "receipt does not resolve latest attempt")
        _require(_time(intent["attempted_at"]) <= _time(receipt["recorded_at"]) <= now, "completion clock chronology mismatch")
        existing = documents.get((sequence, "result"))
        if existing is not None:
            _require(existing["receipt"] == receipt, "attempt result cannot be overwritten")
            return existing
        result = _seal({"schema": RESULT_SCHEMA, "sequence": sequence, "attempt_id": intent["attempt_id"],
                        "intent_sha256": intent["integrity_sha256"], "receipt_sha256": _sha(receipt),
                        "receipt": receipt, "recorded_at": now.isoformat()})
        self._publish(f"{sequence:06d}.result.json", result)
        return result


@contextmanager
def delivery_attempt_journal(snapshot, *, receipt_root=None):
    """Create durable parents; hold their descriptors through the whole send."""
    journal = DeliveryAttemptJournal(snapshot, receipt_root=receipt_root, create=True)
    try:
        yield journal
    finally:
        journal.close()


def inspect_delivery_attempt(snapshot, *, receipt_root=None, now):
    """Read-only status API. Invalid evidence is explicit, never absent/success."""
    journal = None
    try:
        journal = DeliveryAttemptJournal(snapshot, receipt_root=receipt_root, create=False)
        return journal.inspect(now=now)
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        return {"state": "invalid", "resolution": None, "latest_attempt_id": None,
                "attempted_at": None, "attempt_count": None, "journal_sha256": None,
                "reason": f"{type(exc).__name__}: {exc}", "evidence_files": []}
    finally:
        if journal is not None:
            journal.close()
