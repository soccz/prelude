"""Immutable terminal GO/KILL verdict for the retired pump-v2 radar.

The daily scorecard is mutable reporting evidence.  This module deliberately
keeps the one-way operational decision in a separate, content-addressed file:

* before 2026-09-01, a negative cumulative mean records an immediate KILL;
* on/after 2026-09-01, all frozen criteria record GO, otherwise KILL;
* once recorded, the verdict cannot be replaced or revised;
* malformed, tampered, future-dated, or missing-at-deadline state fails closed
  at real Telegram boundaries.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, cast
from zoneinfo import ZoneInfo

from ops.artifact_provenance import (
    ArtifactValidationError,
    strict_json_object,
)
from ops.file_lock import FileLockError, file_lock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RADAR_TERMINAL_STATE = PROJECT_ROOT / "output" / "radar_terminal_verdict.json"
RADAR_TERMINAL_ANCHOR = (
    PROJECT_ROOT / "data" / "radar_terminal_verdict.anchor.json"
)
RADAR_VERDICT_SCHEMA = "radar_terminal_verdict.v1"
JUDGMENT_DAY = date(2026, 9, 1)
KST = ZoneInfo("Asia/Seoul")
MAX_SEND_CLOCK_SKEW = timedelta(minutes=5)
# 2026-08-05에 실제로 봉인된 pump-v2 terminal KILL. 이 시각 이후에는
# state+anchor가 둘 다 사라져도 "아직 미결"로 되돌아갈 수 없다.
PUMP_V2_RETIRED_AT = datetime(
    2026,
    8,
    5,
    1,
    30,
    43,
    423095,
    tzinfo=timezone.utc,
)
PUMP_V2_TERMINAL_EFFECTIVE_ASOF = date(2026, 8, 5)
PUMP_V2_TERMINAL_VERDICT_ID = "radar-verdict-cae133d0a18f1b030273715f"

CRITERIA_KEYS = (
    "n>=200",
    "mean>0",
    "CI_0_제외",
    "2레짐_or_t>=2",
)
_TOP_LEVEL_KEYS = {
    "schema",
    "verdict_id",
    "integrity_sha256",
    "verdict",
    "status",
    "effective_asof",
    "judgment_day",
    "criteria",
    "criteria_met",
    "closed_n",
    "mean_net_pct",
    "ci95",
    "per_day_t",
    "regimes",
    "source_status",
    "source_error",
    "reason",
    "recorded_at",
}


class RadarVerdictError(RuntimeError):
    """The pump-v2 terminal verdict is missing, corrupt, or contradictory."""


class RadarTerminalKill(RadarVerdictError):
    """A valid, effective immutable KILL stopped pump-v2 as designed."""


def _canonical_json(value: Mapping[str, object]) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise RadarVerdictError("radar verdict is not canonical JSON") from exc


def _verdict_core(payload: Mapping[str, object]) -> dict[str, object]:
    return {
        key: payload[key]
        for key in sorted(
            _TOP_LEVEL_KEYS
            - {"verdict_id", "integrity_sha256", "recorded_at"}
        )
    }


def _verdict_id(payload: Mapping[str, object]) -> str:
    digest = hashlib.sha256(_canonical_json(_verdict_core(payload))).hexdigest()
    return f"radar-verdict-{digest[:24]}"


def _integrity_sha256(payload: Mapping[str, object]) -> str:
    body = {
        key: payload[key]
        for key in sorted(_TOP_LEVEL_KEYS - {"integrity_sha256"})
    }
    return hashlib.sha256(_canonical_json(body)).hexdigest()


def _finite_optional(
    value: object,
    *,
    field: str,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise RadarVerdictError(f"radar verdict {field} must be numeric")
    try:
        parsed = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise RadarVerdictError(
            f"radar verdict {field} must be numeric"
        ) from exc
    if not math.isfinite(parsed):
        raise RadarVerdictError(f"radar verdict {field} must be finite")
    return parsed


def _parse_date(value: object, *, field: str) -> date:
    if not isinstance(value, str):
        raise RadarVerdictError(f"radar verdict {field} must be an ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise RadarVerdictError(
            f"radar verdict {field} must be an ISO date"
        ) from exc


def _parse_recorded_at(value: object) -> datetime:
    if not isinstance(value, str):
        raise RadarVerdictError("radar verdict recorded_at must be ISO datetime")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise RadarVerdictError(
            "radar verdict recorded_at must be ISO datetime"
        ) from exc
    if parsed.tzinfo is None:
        raise RadarVerdictError("radar verdict recorded_at must be aware")
    return parsed


def _validated_criteria(payload: Mapping[str, object]) -> dict[str, bool]:
    raw = payload.get("criteria")
    if not isinstance(raw, dict) or set(raw) != set(CRITERIA_KEYS):
        raise RadarVerdictError("radar verdict criteria identity is invalid")
    if any(not isinstance(raw[key], bool) for key in CRITERIA_KEYS):
        raise RadarVerdictError("radar verdict criteria must be booleans")
    return {key: raw[key] for key in CRITERIA_KEYS}


def _validate_payload(payload: Mapping[str, object]) -> dict[str, Any]:
    if set(payload) != _TOP_LEVEL_KEYS:
        raise RadarVerdictError(
            "radar verdict fields are missing or unsupported"
        )
    if payload.get("schema") != RADAR_VERDICT_SCHEMA:
        raise RadarVerdictError("unsupported radar verdict schema")
    if payload.get("judgment_day") != JUDGMENT_DAY.isoformat():
        raise RadarVerdictError("radar verdict judgment_day mismatch")

    effective_asof = _parse_date(
        payload.get("effective_asof"),
        field="effective_asof",
    )
    recorded_at = _parse_recorded_at(payload.get("recorded_at"))
    if recorded_at.astimezone(KST).date() < effective_asof:
        raise RadarVerdictError(
            "radar verdict recorded_at predates effective_asof"
        )
    criteria = _validated_criteria(payload)

    closed_n = payload.get("closed_n")
    if (
        isinstance(closed_n, bool)
        or not isinstance(closed_n, int)
        or closed_n < 0
    ):
        raise RadarVerdictError("radar verdict closed_n is invalid")
    mean = _finite_optional(payload.get("mean_net_pct"), field="mean_net_pct")
    per_day_t = _finite_optional(payload.get("per_day_t"), field="per_day_t")
    ci_raw = payload.get("ci95")
    if ci_raw is None:
        ci: list[float] | None = None
    elif isinstance(ci_raw, list) and len(ci_raw) == 2:
        lower = _finite_optional(ci_raw[0], field="ci95[0]")
        upper = _finite_optional(ci_raw[1], field="ci95[1]")
        if lower is None or upper is None or lower > upper:
            raise RadarVerdictError("radar verdict ci95 is invalid")
        ci = [lower, upper]
    else:
        raise RadarVerdictError("radar verdict ci95 is invalid")
    regimes = payload.get("regimes")
    if (
        not isinstance(regimes, list)
        or any(not isinstance(value, str) for value in regimes)
        or regimes != sorted(set(regimes))
    ):
        raise RadarVerdictError("radar verdict regimes are invalid")

    recomputed = {
        "n>=200": closed_n >= 200,
        "mean>0": mean is not None and mean > 0,
        "CI_0_제외": ci is not None and ci[0] > 0,
        "2레짐_or_t>=2": (
            len(regimes) >= 2
            or (per_day_t is not None and per_day_t >= 2)
        ),
    }
    if criteria != recomputed:
        raise RadarVerdictError("radar verdict criteria/metrics mismatch")
    if payload.get("criteria_met") != sum(criteria.values()):
        raise RadarVerdictError("radar verdict criteria_met mismatch")

    verdict = payload.get("verdict")
    status = payload.get("status")
    reason = payload.get("reason")
    all_met = all(criteria.values())
    if effective_asof < JUDGMENT_DAY:
        expected = (
            "kill",
            "early_kill",
            "cumulative_mean_net_below_zero_before_deadline",
        )
        if mean is None or mean >= 0:
            raise RadarVerdictError(
                "pre-deadline terminal verdict requires negative mean"
            )
    elif all_met:
        if effective_asof != JUDGMENT_DAY:
            raise RadarVerdictError(
                "judgment verdict must be effective on frozen deadline"
            )
        expected = (
            "go",
            "judgment_go",
            "all_frozen_criteria_met_at_deadline",
        )
    else:
        if effective_asof != JUDGMENT_DAY:
            raise RadarVerdictError(
                "judgment verdict must be effective on frozen deadline"
            )
        expected = (
            "kill",
            "judgment_kill",
            "one_or_more_frozen_criteria_not_met_at_deadline",
        )
    if (verdict, status, reason) != expected:
        raise RadarVerdictError("radar verdict terminal semantics mismatch")

    source_status = payload.get("source_status")
    source_error = payload.get("source_error")
    if not isinstance(source_status, str) or not source_status:
        raise RadarVerdictError("radar verdict source_status is invalid")
    if source_error is not None and not isinstance(source_error, str):
        raise RadarVerdictError("radar verdict source_error is invalid")
    if payload.get("verdict_id") != _verdict_id(payload):
        raise RadarVerdictError("radar verdict checksum mismatch")
    if payload.get("integrity_sha256") != _integrity_sha256(payload):
        raise RadarVerdictError("radar verdict document integrity mismatch")
    return dict(payload)


def terminal_candidate(
    scorecard: Mapping[str, object],
    *,
    asof: date,
    recorded_at: datetime | None = None,
) -> dict[str, object] | None:
    """Build a terminal record only when the frozen rule has resolved."""
    criteria_raw = scorecard.get("criteria")
    if not isinstance(criteria_raw, dict):
        raise RadarVerdictError("scorecard criteria are missing")
    criteria = _validated_criteria(scorecard)
    closed_n = scorecard.get("closed_n")
    if (
        isinstance(closed_n, bool)
        or not isinstance(closed_n, int)
        or closed_n < 0
    ):
        raise RadarVerdictError("scorecard closed_n is invalid")
    metric_values = scorecard.get("terminal_metric_values")
    if metric_values is None:
        metric_values = {
            "mean_net_pct": scorecard.get("mean_net_pct"),
            "ci95": scorecard.get("ci95"),
            "per_day_t": scorecard.get("per_day_t"),
        }
    if not isinstance(metric_values, dict):
        raise RadarVerdictError("scorecard terminal_metric_values is invalid")
    mean = _finite_optional(
        metric_values.get("mean_net_pct"),
        field="scorecard.mean_net_pct",
    )
    if asof < JUDGMENT_DAY and not (
        scorecard.get("early_kill_breached") is True
        and mean is not None
        and mean < 0
    ):
        return None

    all_met = all(criteria.values())
    if asof < JUDGMENT_DAY:
        verdict = "kill"
        status = "early_kill"
        reason = "cumulative_mean_net_below_zero_before_deadline"
        effective_asof = asof
    elif all_met:
        verdict = "go"
        status = "judgment_go"
        reason = "all_frozen_criteria_met_at_deadline"
        effective_asof = JUDGMENT_DAY
    else:
        verdict = "kill"
        status = "judgment_kill"
        reason = "one_or_more_frozen_criteria_not_met_at_deadline"
        effective_asof = JUDGMENT_DAY

    now = recorded_at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise RadarVerdictError("terminal candidate recorded_at must be aware")
    if now.astimezone(KST).date() != asof:
        raise RadarVerdictError(
            "terminal candidate recorded_at must match evaluation asof"
        )
    payload: dict[str, object] = {
        "schema": RADAR_VERDICT_SCHEMA,
        "verdict_id": "",
        "integrity_sha256": "",
        "verdict": verdict,
        "status": status,
        "effective_asof": effective_asof.isoformat(),
        "judgment_day": JUDGMENT_DAY.isoformat(),
        "criteria": criteria,
        "criteria_met": sum(criteria.values()),
        "closed_n": closed_n,
        "mean_net_pct": mean,
        "ci95": metric_values.get("ci95"),
        "per_day_t": metric_values.get("per_day_t"),
        "regimes": scorecard.get("regimes", []),
        "source_status": str(scorecard.get("status", "unknown")),
        "source_error": scorecard.get("error"),
        "reason": reason,
        "recorded_at": now.astimezone(timezone.utc).isoformat(),
    }
    payload["verdict_id"] = _verdict_id(payload)
    payload["integrity_sha256"] = _integrity_sha256(payload)
    return _validate_payload(payload)


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    try:
        with file_lock(lock_path):
            yield
    except FileLockError as exc:
        raise RadarVerdictError(
            f"radar terminal verdict lock is unsafe: {lock_path}"
        ) from exc


def _read_validated(path: Path) -> dict[str, Any]:
    try:
        value = strict_json_object(path)
    except ArtifactValidationError as exc:
        raise RadarVerdictError(
            f"radar terminal verdict is unreadable: {path}"
        ) from exc
    return _validate_payload(value)


def terminal_anchor_path(
    path: str | Path = RADAR_TERMINAL_STATE,
) -> Path:
    """Resolve the independent mirror that remembers a deleted output state."""
    resolved = Path(path).resolve(strict=False)
    if resolved == RADAR_TERMINAL_STATE.resolve(strict=False):
        return RADAR_TERMINAL_ANCHOR
    return Path(path).with_name(f"{Path(path).name}.anchor")


def _directory_entry_exists(path: Path) -> bool:
    """Distinguish true absence from dangling links and unreadable entries."""
    return _directory_entry_token(path) is not None


def _stat_token(value: os.stat_result) -> tuple[int, ...]:
    """Capture replacement/mutation metadata while ignoring access time."""
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _inode(value: os.stat_result) -> tuple[int, int]:
    return int(value.st_dev), int(value.st_ino)


def _directory_entry_token(path: Path) -> tuple[int, ...] | None:
    try:
        value = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RadarVerdictError(
            f"cannot inspect radar terminal verdict entry: {path}"
        ) from exc
    return _stat_token(value)


def _stat_at(
    directory_fd: int,
    name: str,
    *,
    path: Path,
) -> os.stat_result | None:
    try:
        return os.stat(
            name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RadarVerdictError(
            f"cannot inspect radar terminal verdict entry: {path}"
        ) from exc


@contextmanager
def _verified_directory(path: Path) -> Iterator[int]:
    """Open one real directory and keep path operations bound to its inode."""
    if not hasattr(os, "O_NOFOLLOW"):
        raise RadarVerdictError(
            "O_NOFOLLOW is required for safe radar verdict storage"
        )
    try:
        before = path.lstat()
    except OSError as exc:
        raise RadarVerdictError(
            f"cannot inspect radar verdict directory: {path}"
        ) from exc
    if not stat.S_ISDIR(before.st_mode):
        raise RadarVerdictError(
            f"radar verdict parent must be a real directory: {path}"
        )
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        directory_fd = os.open(path, flags)
    except OSError as exc:
        raise RadarVerdictError(
            f"cannot safely open radar verdict directory: {path}"
        ) from exc
    try:
        if _inode(os.fstat(directory_fd)) != _inode(before):
            raise RadarVerdictError(
                f"radar verdict directory changed while opening: {path}"
            )
        yield directory_fd
    finally:
        os.close(directory_fd)


def _load_pair_unlocked(path: Path) -> dict[str, Any] | None:
    anchor = terminal_anchor_path(path)
    # Path.exists() follows links and reports a dangling symlink as absent.
    # Treat any directory entry as present so a replaced terminal KILL pair
    # cannot silently become an uninitialized pre-deadline state.
    tokens_before = (
        _directory_entry_token(path),
        _directory_entry_token(anchor),
    )
    state_exists = tokens_before[0] is not None
    anchor_exists = tokens_before[1] is not None
    if not state_exists and not anchor_exists:
        return None
    if state_exists != anchor_exists:
        raise RadarVerdictError(
            "radar terminal verdict state/anchor pair is incomplete"
        )
    state = _read_validated(path)
    anchor_state = _read_validated(anchor)
    tokens_after = (
        _directory_entry_token(path),
        _directory_entry_token(anchor),
    )
    if tokens_before != tokens_after:
        raise RadarVerdictError(
            "radar terminal verdict pair changed while reading"
        )
    if state != anchor_state:
        raise RadarVerdictError(
            "radar terminal verdict state/anchor pair mismatch"
        )
    return state


def load_terminal_verdict(
    path: str | Path = RADAR_TERMINAL_STATE,
) -> dict[str, Any] | None:
    resolved = Path(path)
    with _exclusive_lock(resolved):
        return _load_pair_unlocked(resolved)


def _atomic_write(
    path: Path,
    payload: Mapping[str, object],
    *,
    replace_from: Path | None = None,
) -> None:
    """Durably publish a verdict, replacing only preserved exact evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = (
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    with _verified_directory(path.parent) as directory_fd:
        existing = _stat_at(directory_fd, path.name, path=path)
        preserved: os.stat_result | None = None
        if replace_from is not None:
            if replace_from.parent != path.parent:
                raise RadarVerdictError(
                    "radar verdict replacement evidence must be a sibling"
                )
            preserved = _stat_at(
                directory_fd,
                replace_from.name,
                path=replace_from,
            )
            if (
                existing is None
                or preserved is None
                or not stat.S_ISREG(existing.st_mode)
                or _inode(existing) != _inode(preserved)
                or _stat_token(existing) != _stat_token(preserved)
            ):
                raise RadarVerdictError(
                    "radar verdict replacement evidence changed: "
                    f"{path}"
                )
        elif existing is not None:
            kind = (
                "existing"
                if stat.S_ISREG(existing.st_mode)
                else "unsafe"
            )
            raise RadarVerdictError(
                f"radar verdict atomic write refuses {kind} entry: {path}"
            )

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        fd = -1
        temp_created = False
        try:
            fd = os.open(
                temp_name,
                flags,
                0o600,
                dir_fd=directory_fd,
            )
            temp_created = True
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                fd = -1
                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
                written = os.fstat(handle.fileno())

            temp_path = path.with_name(temp_name)
            named_temp = _stat_at(
                directory_fd,
                temp_name,
                path=temp_path,
            )
            if (
                named_temp is None
                or not stat.S_ISREG(named_temp.st_mode)
                or named_temp.st_nlink != 1
                or _stat_token(named_temp) != _stat_token(written)
            ):
                raise RadarVerdictError(
                    f"radar verdict temporary file changed: {temp_path}"
                )

            current = _stat_at(directory_fd, path.name, path=path)
            if replace_from is None:
                if current is not None:
                    raise RadarVerdictError(
                        "radar verdict target appeared during atomic write: "
                        f"{path}"
                    )
                try:
                    # A hard-link publication is atomic and, unlike replace(),
                    # cannot overwrite a verdict that appears after the check.
                    os.link(
                        temp_name,
                        path.name,
                        src_dir_fd=directory_fd,
                        dst_dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                except FileExistsError as exc:
                    raise RadarVerdictError(
                        "radar verdict target appeared during atomic write: "
                        f"{path}"
                    ) from exc
                os.unlink(temp_name, dir_fd=directory_fd)
            else:
                preserved_now = _stat_at(
                    directory_fd,
                    replace_from.name,
                    path=replace_from,
                )
                if (
                    current is None
                    or preserved is None
                    or preserved_now is None
                    or _inode(current) != _inode(preserved)
                    or _inode(preserved_now) != _inode(preserved)
                    or _stat_token(current) != _stat_token(preserved_now)
                    or _stat_token(preserved_now) != _stat_token(preserved)
                ):
                    raise RadarVerdictError(
                        "radar verdict replacement target changed: "
                        f"{path}"
                    )
                # Replacement is safe here because the exact displaced inode
                # remains durably reachable through the preserved sibling.
                os.replace(
                    temp_name,
                    path.name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                )
            temp_created = False
            os.fsync(directory_fd)
        finally:
            if fd >= 0:
                os.close(fd)
            if temp_created:
                try:
                    os.unlink(temp_name, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass


def _hash_fd(fd: int) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: os.read(fd, 1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _quarantine_digest(
    directory_fd: int,
    path: Path,
    expected: os.stat_result,
) -> str:
    """Hash one stable named entry without following a symlink."""
    if stat.S_ISLNK(expected.st_mode):
        try:
            first = os.readlink(path.name, dir_fd=directory_fd)
            second = os.readlink(path.name, dir_fd=directory_fd)
        except OSError as exc:
            raise RadarVerdictError(
                f"radar verdict entry changed while hashing: {path}"
            ) from exc
        current = _stat_at(directory_fd, path.name, path=path)
        if (
            current is None
            or _stat_token(current) != _stat_token(expected)
            or first != second
        ):
            raise RadarVerdictError(
                f"radar verdict entry changed while hashing: {path}"
            )
        content = b"symlink\0" + os.fsencode(first)
        return hashlib.sha256(content).hexdigest()[:16]

    if not stat.S_ISREG(expected.st_mode):
        raise RadarVerdictError(
            f"cannot safely quarantine non-file radar entry: {path}"
        )

    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        fd = os.open(path.name, flags, dir_fd=directory_fd)
    except OSError as exc:
        current = _stat_at(directory_fd, path.name, path=path)
        if current is None or _stat_token(current) != _stat_token(expected):
            raise RadarVerdictError(
                f"radar verdict entry changed while hashing: {path}"
            ) from exc
        return "unreadable"
    try:
        opened = os.fstat(fd)
        if _inode(opened) != _inode(expected):
            raise RadarVerdictError(
                f"radar verdict entry changed while hashing: {path}"
            )
        digest = _hash_fd(fd)
        os.lseek(fd, 0, os.SEEK_SET)
        verification = _hash_fd(fd)
        opened_after = os.fstat(fd)
    finally:
        os.close(fd)
    named_after = _stat_at(directory_fd, path.name, path=path)
    if (
        digest != verification
        or _stat_token(opened) != _stat_token(opened_after)
        or named_after is None
        or _stat_token(named_after) != _stat_token(opened_after)
        or _stat_token(expected) != _stat_token(opened)
    ):
        raise RadarVerdictError(
            f"radar verdict entry changed while hashing: {path}"
        )
    return digest[:16]


def _quarantine(
    path: Path,
    *,
    remove: bool = True,
) -> Path | None:
    """Preserve exact evidence and optionally remove its original name."""
    with _verified_directory(path.parent) as directory_fd:
        before = _stat_at(directory_fd, path.name, path=path)
        if before is None:
            return None
        digest = _quarantine_digest(directory_fd, path, before)
        base = path.with_name(f"{path.name}.corrupt-{digest}")
        target = base
        index = 1
        linked = False
        committed = False
        source_removed = False
        try:
            while True:
                try:
                    os.link(
                        path.name,
                        target.name,
                        src_dir_fd=directory_fd,
                        dst_dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                    linked = True
                    break
                except FileExistsError:
                    target = path.with_name(f"{base.name}-{index}")
                    index += 1
                except OSError as exc:
                    raise RadarVerdictError(
                        f"cannot quarantine radar terminal verdict: {path}"
                    ) from exc

            linked_entry = _stat_at(
                directory_fd,
                target.name,
                path=target,
            )
            current = _stat_at(
                directory_fd,
                path.name,
                path=path,
            )
            if (
                linked_entry is None
                or current is None
                or _inode(linked_entry) != _inode(before)
                or _inode(current) != _inode(before)
            ):
                raise RadarVerdictError(
                    f"radar verdict entry changed during quarantine: {path}"
                )
            if _quarantine_digest(
                directory_fd,
                target,
                linked_entry,
            ) != digest:
                raise RadarVerdictError(
                    f"radar verdict entry changed during quarantine: {path}"
                )
            current = _stat_at(
                directory_fd,
                path.name,
                path=path,
            )
            if current is None or _inode(current) != _inode(before):
                raise RadarVerdictError(
                    f"radar verdict entry changed during quarantine: {path}"
                )
            if remove:
                os.unlink(path.name, dir_fd=directory_fd)
                source_removed = True
            os.fsync(directory_fd)
            committed = True
            return target
        finally:
            if linked and not committed and not source_removed:
                linked_entry = _stat_at(
                    directory_fd,
                    target.name,
                    path=target,
                )
                if (
                    linked_entry is not None
                    and _inode(linked_entry) == _inode(before)
                ):
                    try:
                        os.unlink(target.name, dir_fd=directory_fd)
                    except FileNotFoundError:
                        pass


def _try_read_validated(path: Path) -> dict[str, Any] | None:
    if not _directory_entry_exists(path):
        return None
    try:
        return _read_validated(path)
    except RadarVerdictError:
        return None


def recover_terminal_verdict(
    *,
    path: str | Path = RADAR_TERMINAL_STATE,
    forced_kill: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Repair one valid mirror or replace irreconcilable evidence with KILL.

    Corrupt/mismatched files are moved aside, never silently overwritten.
    ``forced_kill`` is required when neither mirror is trustworthy and must
    itself be a fully validated KILL verdict.
    """
    resolved = Path(path)
    anchor = terminal_anchor_path(resolved)
    validated_forced = (
        _validate_payload(forced_kill)
        if forced_kill is not None
        else None
    )
    if (
        validated_forced is not None
        and validated_forced["verdict"] != "kill"
    ):
        raise RadarVerdictError("terminal recovery may only force KILL")

    with _exclusive_lock(resolved):
        tokens_before = (
            _directory_entry_token(resolved),
            _directory_entry_token(anchor),
        )
        state = _try_read_validated(resolved)
        anchor_state = _try_read_validated(anchor)
        tokens_after = (
            _directory_entry_token(resolved),
            _directory_entry_token(anchor),
        )
        if tokens_before != tokens_after:
            raise RadarVerdictError(
                "terminal verdict mirrors changed during recovery"
            )
        recovered: dict[str, Any]
        if state is not None and anchor_state is not None:
            if state == anchor_state:
                return state
            if validated_forced is None:
                raise RadarVerdictError(
                    "terminal verdict mirrors are irreconcilable"
                )
            state_evidence = _quarantine(resolved, remove=False)
            anchor_evidence = _quarantine(anchor, remove=False)
            if state_evidence is None or anchor_evidence is None:
                raise RadarVerdictError(
                    "terminal recovery evidence disappeared"
                )
            _atomic_write(
                resolved,
                validated_forced,
                replace_from=state_evidence,
            )
            _atomic_write(
                anchor,
                validated_forced,
                replace_from=anchor_evidence,
            )
            recovered = validated_forced
        elif state is not None:
            if _directory_entry_exists(anchor):
                _quarantine(anchor)
            _atomic_write(anchor, state)
            recovered = state
        elif anchor_state is not None:
            if _directory_entry_exists(resolved):
                _quarantine(resolved)
            _atomic_write(resolved, anchor_state)
            recovered = anchor_state
        else:
            if validated_forced is None:
                raise RadarVerdictError(
                    "terminal verdict mirrors are irreconcilable"
                )
            _quarantine(resolved)
            _quarantine(anchor)
            _atomic_write(resolved, validated_forced)
            _atomic_write(anchor, validated_forced)
            recovered = validated_forced

        verified = _load_pair_unlocked(resolved)
        if verified is None or verified != recovered:
            raise RadarVerdictError(
                "terminal verdict mirrors changed during recovery"
            )
        return verified


def record_terminal_verdict(
    candidate: Mapping[str, object],
    *,
    path: str | Path = RADAR_TERMINAL_STATE,
) -> dict[str, Any]:
    """Create the terminal verdict once; every later mutation is rejected."""
    resolved = Path(path)
    validated = _validate_payload(candidate)
    with _exclusive_lock(resolved):
        existing = _load_pair_unlocked(resolved)
        if existing is not None:
            if existing["verdict_id"] != validated["verdict_id"]:
                raise RadarVerdictError(
                    "radar terminal verdict is immutable and already resolved"
                )
            return existing
        _atomic_write(resolved, validated)
        _atomic_write(terminal_anchor_path(resolved), validated)
    return validated


def verdict_status_for(
    asof: date,
    *,
    path: str | Path = RADAR_TERMINAL_STATE,
) -> dict[str, object]:
    """Return the terminal state applicable to a reporting cutoff."""
    state = load_terminal_verdict(path)
    if state is None:
        if asof >= JUDGMENT_DAY:
            raise RadarVerdictError(
                "radar terminal verdict missing at/after judgment deadline"
            )
        return {
            "status": "pending",
            "verdict": None,
            "judgment_day": JUDGMENT_DAY.isoformat(),
            "days_to_judgment": (JUDGMENT_DAY - asof).days,
            "effective": False,
        }
    effective_asof = date.fromisoformat(state["effective_asof"])
    if asof < effective_asof:
        return {
            "status": "pending",
            "verdict": None,
            "judgment_day": JUDGMENT_DAY.isoformat(),
            "days_to_judgment": (JUDGMENT_DAY - asof).days,
            "effective": False,
        }
    return {
        "status": state["status"],
        "verdict": state["verdict"],
        "verdict_id": state["verdict_id"],
        "effective_asof": state["effective_asof"],
        "judgment_day": state["judgment_day"],
        "reason": state["reason"],
        "effective": True,
    }


def assert_radar_send_allowed(
    *,
    path: str | Path = RADAR_TERMINAL_STATE,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Fail closed before any canonical radar's real Telegram side effect."""
    with radar_send_guard(path=path, now=now) as state:
        return state


def assert_pump_v2_runtime_allowed(
    *,
    path: str | Path = RADAR_TERMINAL_STATE,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Apply the permanent, exact pump-v2 retirement contract."""
    with radar_send_guard(
        path=path,
        now=now,
        enforce_pump_v2_retirement=True,
    ) as state:
        return state


@contextmanager
def radar_send_guard(
    *,
    path: str | Path = RADAR_TERMINAL_STATE,
    now: datetime | None = None,
    clock: Callable[[], datetime] | None = None,
    boundary_check: Callable[[datetime], None] | None = None,
    enforce_pump_v2_retirement: bool = False,
) -> Iterator[dict[str, Any] | None]:
    """Linearize terminal resolution and one real Telegram API attempt.

    The caller must keep the API call inside this context.  A concurrent
    terminal writer then resolves either wholly before the attempt (and blocks
    it) or wholly after the attempt; there is no check/write TOCTOU interval.
    """
    if now is not None and clock is not None:
        raise RadarVerdictError("radar send guard accepts now or clock, not both")
    resolved = Path(path)
    with _exclusive_lock(resolved):
        observed = now or (clock() if clock is not None else datetime.now(KST))
        if observed.tzinfo is None:
            raise RadarVerdictError("radar send clock must be timezone-aware")
        observed_day = observed.astimezone(KST).date()
        state = _load_pair_unlocked(resolved)
        retirement_active = (
            enforce_pump_v2_retirement
            and observed.astimezone(timezone.utc) >= PUMP_V2_RETIRED_AT
        )
        if retirement_active:
            if state is None:
                raise RadarVerdictError(
                    "pump-v2 terminal KILL pair missing after retirement"
                )
            expected_identity = {
                "verdict": "kill",
                "status": "early_kill",
                "verdict_id": PUMP_V2_TERMINAL_VERDICT_ID,
                "effective_asof": (
                    PUMP_V2_TERMINAL_EFFECTIVE_ASOF.isoformat()
                ),
                "recorded_at": PUMP_V2_RETIRED_AT.isoformat(),
            }
            if any(
                state.get(key) != value
                for key, value in expected_identity.items()
            ):
                raise RadarVerdictError(
                    "pump-v2 terminal KILL identity mismatch after retirement"
                )
        if state is None:
            if observed_day >= JUDGMENT_DAY:
                raise RadarVerdictError(
                    "radar send blocked: terminal verdict missing after deadline"
                )
            if boundary_check is not None:
                boundary_check(observed)
            yield None
            return
        effective_asof = date.fromisoformat(state["effective_asof"])
        recorded_at = _parse_recorded_at(state["recorded_at"])
        if recorded_at > observed.astimezone(timezone.utc) + MAX_SEND_CLOCK_SKEW:
            raise RadarVerdictError(
                "radar send blocked: terminal verdict was recorded in future"
            )
        if effective_asof > observed_day:
            raise RadarVerdictError(
                "radar send blocked: terminal verdict is future-dated"
            )
        if state["verdict"] != "go":
            raise RadarTerminalKill(
                "radar send blocked by immutable KILL verdict "
                f"{state['verdict_id']}"
            )
        if boundary_check is not None:
            boundary_check(observed)
        yield state
