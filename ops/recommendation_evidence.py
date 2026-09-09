"""Read-only, cross-bound R1 evidence for offline diagnostics, never live inference.

The existing loaders validate each document. This module also proves that the
snapshot, server receipt and modern path labels describe the same decision.
Missing evidence is not a zero-return observation or proof of non-delivery.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from ledger.path_quality import next_bar_boundary
from notifier.delivery_receipt import (
    RECEIPT_INTEGRITY_ACTIVATION_DATE,
    _LIVE_SEND_WINDOWS,
    read_delivery_receipt,
    receipt_path,
)
from ops.artifact_provenance import file_set_identity
from signals.recommend_score_labels import (
    FORWARD_PROVENANCE_COHORT,
    ROUND_TRIP_COST,
    _CANDIDATE_KEYS,
    load_label_artifact,
    path_window,
)
from signals.recommend_snapshot import SNAPSHOT_SCHEMA_VERSION, load_snapshot

ROOT = Path(__file__).resolve().parent.parent
KST = timezone(timedelta(hours=9))


class EvidenceError(ValueError):
    """Corrupt or contradictory evidence: do not silently drop this sample."""


class EvidenceUnavailable(EvidenceError):
    """Known unavailable/non-forward evidence: report the exclusion explicitly."""


def aware_time(value: str | datetime, field: str = "now") -> datetime:
    try:
        result = datetime.fromisoformat(value) if isinstance(value, str) else value
        if not isinstance(result, datetime) or result.utcoffset() is None:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"{field}: timezone-aware datetime required") from exc
    return result


def discover_r1_snapshots(
    snapshot_root: str | Path,
    *,
    start_date: date,
    end_date: date,
    slots: tuple[str, ...] = ("open", "preopen"),
) -> list[Path]:
    """List canonical identities only; limited/research copies are not inputs."""
    if type(start_date) is not date or type(end_date) is not date:
        raise EvidenceError("start_date/end_date must be calendar dates")
    if start_date > end_date:
        raise EvidenceError("start_date is after end_date")
    if not slots or len(set(slots)) != len(slots) or set(slots) - {"open", "preopen"}:
        raise EvidenceError("slots must be unique open/preopen values")
    root = Path(snapshot_root)
    if not root.is_dir():
        raise EvidenceUnavailable("snapshot_root_missing")
    result = []
    for directory in sorted(root.iterdir()):
        try:
            day = date.fromisoformat(directory.name)
        except ValueError:
            continue
        if directory.name != day.isoformat() or not start_date <= day <= end_date:
            continue
        if directory.is_symlink():
            raise EvidenceError(f"snapshot_date_directory_is_symlink: {directory}")
        if not directory.is_dir():
            continue
        for slot in slots:
            path = directory / f"{slot}_r1.json"
            if path.exists() or path.is_symlink():
                result.append(path)
    return result


def _identity_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise EvidenceError(reason)


def load_recommendation_evidence(
    snapshot_path: str | Path,
    *,
    label_root: str | Path,
    receipt_root: str | Path,
    now: datetime,
) -> dict:
    """Return frozen document copies and hashes, without collecting or writing.

Only modern, successfully delivered R1 decisions are eligible. Their entire
universe is returned; callers must use ``snapshot.top3`` to identify alerts.
"""
    observed_at = aware_time(now)
    path = Path(snapshot_path)
    if path.is_symlink():
        raise EvidenceError("snapshot_is_symlink")
    if not path.exists():
        raise EvidenceUnavailable("snapshot_missing")
    # Normalize the intentional project-root alias, not document contents.
    path = path.resolve()
    try:
        day = date.fromisoformat(path.parent.name)
        _require(path.parent.name == day.isoformat(), "noncanonical_snapshot_date")
        slot = path.name.removesuffix("_r1.json")
        _require(
            path.name == f"{slot}_r1.json" and slot in {"open", "preopen"},
            "noncanonical_snapshot_name",
        )
        label_path = Path(label_root).absolute() / day.isoformat() / path.name
        expected_receipt = Path(receipt_root).absolute() / day.isoformat() / path.name
        paths = {"snapshot": path, "label": label_path, "receipt": expected_receipt}
        before = file_set_identity(paths, root=ROOT)
        snapshot = load_snapshot(path, asof=day.isoformat(), slot=slot, ranking="R1")
        if snapshot["snapshot_schema"] != SNAPSHOT_SCHEMA_VERSION:
            raise EvidenceUnavailable("legacy_snapshot")
        _require(snapshot.get("ranking") == "R1", "snapshot_ranking_mismatch")
        _require(
            snapshot["model"]["id"] == f"recommend_r1_{slot}",
            "unapproved_r1_model_identity",
        )
        _require(
            snapshot.get("rank_basis") == "R1_riskreward(de-corr head)"
            and snapshot["rule"]["version"] == "r1_riskreward_v1"
            and snapshot.get("score_schema_version") == "recommend_score.v2",
            "unapproved_r1_ranking",
        )
        if not before["label"]["exists"]:
            raise EvidenceUnavailable("label_missing")
        if not before["receipt"]["exists"]:
            raise EvidenceUnavailable("receipt_missing_delivery_unknown")
        receipt = read_delivery_receipt(snapshot, root=Path(receipt_root).absolute())
        if receipt is None:
            raise EvidenceError("receipt_disappeared_during_read")
        if day < RECEIPT_INTEGRITY_ACTIVATION_DATE:
            raise EvidenceUnavailable("legacy_receipt_without_server_evidence")
        if not receipt["delivery_ok"]:
            raise EvidenceUnavailable("receipt_not_confirmed_success")
        _require(bool(receipt.get("telegram_messages")), "server_evidence_missing")
        _require(
            receipt_path(snapshot, root=Path(receipt_root).absolute()) == expected_receipt,
            "receipt_path_mismatch",
        )
        label = load_label_artifact(label_path)
        if "round_trip_cost_fraction" not in label or "label_code" not in label:
            raise EvidenceUnavailable("legacy_label_contract")
        if label["artifact_status"] != "complete":
            raise EvidenceUnavailable(f"label_status={label['artifact_status']}")
        if (
            label.get("provenance_cohort") != FORWARD_PROVENANCE_COHORT
            or label.get("forward_eligible") is not True
        ):
            raise EvidenceUnavailable("not_forward_observed")

        expected_fields = {
            "asof": day.isoformat(), "slot": slot, "ranking": "R1",
            "snapshot_id": snapshot["snapshot_id"],
            "snapshot_payload_sha256": snapshot["payload_sha256"],
            "feature_asof": snapshot["feature_asof"],
            "snapshot_model": snapshot["model"], "snapshot_rule": snapshot["rule"],
            "snapshot_code": snapshot["code"], "snapshot_data": snapshot["data"],
            "delivery_ok": True, "execution_time_basis": "delivery_sent_at",
        }
        for key, value in expected_fields.items():
            _require(label.get(key) == value, f"label_snapshot_mismatch:{key}")
        _require(_identity_path(label["snapshot_path"]) == path, "label_snapshot_path_mismatch")
        _require(
            _identity_path(label["receipt_path"]) == expected_receipt.resolve(),
            "label_receipt_path_mismatch",
        )
        _require(
            math.isclose(label["round_trip_cost_fraction"], ROUND_TRIP_COST, abs_tol=1e-12),
            "cost_contract_mismatch",
        )

        sent_at = aware_time(receipt["sent_at"], "sent_at")
        decision_started = aware_time(snapshot["decision_started_at"])
        decision_completed = aware_time(snapshot["decision_completed_at"])
        attempted_at = aware_time(receipt["attempted_at"])
        live_start, live_end = (
            datetime.combine(day, wall, tzinfo=KST)
            for wall in _LIVE_SEND_WINDOWS[slot]
        )
        _require(
            live_start <= decision_started <= decision_completed <= attempted_at < live_end,
            "decision_delivery_chronology_or_slot_mismatch",
        )
        start, _ = path_window(day.isoformat())
        expected_start = max(start, next_bar_boundary(sent_at))
        execution_start = aware_time(label["execution_start_at"], "execution_start_at")
        _require(aware_time(label["execution_at"]) == sent_at, "execution_receipt_time_mismatch")
        _require(execution_start == expected_start, "execution_start_mismatch")
        _require(aware_time(label["path_window_start"]) == execution_start, "path_start_mismatch")
        end_at = aware_time(label["path_window_end"], "path_window_end")
        _require(end_at == execution_start + timedelta(days=1), "horizon_not_24h")
        labeled_at = aware_time(label["labeled_at"], "labeled_at")
        _require(labeled_at >= end_at, "label_written_before_outcome_mature")
        for key in ("decision_started_at", "decision_completed_at", "created_at"):
            _require(aware_time(snapshot[key], key) <= observed_at, f"future_snapshot:{key}")
        for key in ("attempted_at", "recorded_at", "sent_at"):
            _require(aware_time(receipt[key], key) <= observed_at, f"future_receipt:{key}")
        if max(end_at, labeled_at) > observed_at:
            raise EvidenceUnavailable("label_not_available_as_of_now")

        candidates = {(row["coin"], row["rank"]): row for row in snapshot["universe"]}
        rows = {(row["coin"], row["rank"]): row for row in label["rows"]}
        _require(len(candidates) == len(snapshot["universe"]), "duplicate_snapshot_candidate")
        _require(len(rows) == len(label["rows"]), "duplicate_label_candidate")
        _require(candidates.keys() == rows.keys(), "label_universe_identity_mismatch")
        for identity, candidate in candidates.items():
            row = rows[identity]
            for key in _CANDIDATE_KEYS:
                _require(row[key] == candidate[key], f"candidate_mismatch:{identity[0]}:{key}")
        after = file_set_identity(paths, root=ROOT)
        _require(before == after, "evidence_changed_during_read")
        return {
            "snapshot": snapshot, "label": label, "receipt": receipt,
            "manifest": {
                "files": before,
                "snapshot_id": snapshot["snapshot_id"],
                "snapshot_payload_sha256": snapshot["payload_sha256"],
                "label_payload_sha256": label["label_payload_sha256"],
                "label_available_at": max(end_at, labeled_at).isoformat(),
            },
        }
    except EvidenceError:
        raise
    except Exception as exc:
        raise EvidenceError(f"evidence_invalid:{type(exc).__name__}:{exc}") from exc
