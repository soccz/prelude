"""Delivery compatibility for independent heads, without changing frozen research.

The frozen reader rejects inconsistent top-k probabilities. This reader records
the exact same raw candidates with an explicit quality marker and hides that
family in the UI. It never recalibrates, clips, reorders or relaxes any numeric,
provenance, timing, ranking or receipt check. Strict research readers stay strict.
"""

from __future__ import annotations

import copy
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from ops.artifact_provenance import ArtifactValidationError, strict_json_object
from ops.file_lock import file_lock
from signals import recommend_snapshot as native

DEFAULT_SNAPSHOT_ROOT = native.DEFAULT_SNAPSHOT_ROOT
SNAPSHOT_SCHEMA_VERSION = native.SNAPSHOT_SCHEMA_VERSION
SnapshotError = native.SnapshotError
snapshot_path = native.snapshot_path
QUALITY_SCHEMA = "recommend_delivery_probability_quality.v1"


def inconsistent_families(row: dict) -> list[str]:
    families = []
    up = [row.get(k) for k in ("p_up5", "p_up10", "p_up20")]
    down = [row.get(k) for k in ("p_dn5", "p_dn10")]
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in up):
        if not up[0] >= up[1] >= up[2]:
            families.append("upside")
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in down):
        if down[0] < down[1]:
            families.append("downside")
    return families


def probability_quality(document: dict) -> dict:
    return {
        "schema": QUALITY_SCHEMA,
        "policy": "raw_preserved_inconsistent_family_hidden",
        "candidates": {
            row["coin"]: families
            for row in document["universe"]
            if (families := inconsistent_families(row))
        },
    }


def _validate_marked(document: dict, path: Path) -> None:
    if (
        document.get("ranking") != "R1"
        or document.get("snapshot_schema") != SNAPSHOT_SCHEMA_VERSION
    ):
        raise SnapshotError("delivery probability policy requires a modern R1 snapshot")
    # Reuse the entire frozen validator, including ranges, completeness,
    # leakage, RR arithmetic and time/provenance. Its existing record-only
    # branch already supports non-nested independent heads. This temporary
    # validation view is never persisted or returned as a different ranking.
    view = copy.deepcopy(document)
    view["ranking"] = view["request"]["ranking"] = view["model"]["ranking"] = "R2"
    native._validate_snapshot_contract(view, path)
    expected = probability_quality(document)
    if document.get("probability_quality") != expected:
        raise SnapshotError(
            "delivery probability quality marker does not match raw estimates"
        )
    top_coins = {row["coin"] for row in document["top3"]}
    if not top_coins.intersection(expected["candidates"]):
        raise SnapshotError("delivery compatibility marker is unnecessary")
    if (
        document.get("model", {}).get("ranking") != "R1"
        or document.get("request", {}).get("ranking") != "R1"
    ):
        raise SnapshotError("delivery snapshot original ranking identity mismatch")
    policy_source = document.get("delivery_policy_source_sha256")
    if (
        not isinstance(policy_source, str)
        or len(policy_source) != 64
        or any(c not in "0123456789abcdef" for c in policy_source)
    ):
        raise SnapshotError("delivery policy source identity missing")


def load_snapshot(
    path, *, asof=None, slot=None, ranking=None, limit_markets=None, model_id=None
):
    p = Path(path)
    try:
        document = strict_json_object(p)
    except ArtifactValidationError as exc:
        raise SnapshotError(f"delivery snapshot read failed: {p}") from exc
    if "probability_quality" not in document:
        return native.load_snapshot(
            p,
            asof=asof,
            slot=slot,
            ranking=ranking,
            limit_markets=limit_markets,
            model_id=model_id,
        )
    digest = native._document_digest(document)
    if (
        document.get("payload_sha256") != digest
        or document.get("snapshot_id") != f"recommend-{digest[:20]}"
    ):
        raise SnapshotError("delivery snapshot checksum mismatch")
    _validate_marked(document, p)
    request = document["request"]
    for key, expected in (
        ("asof", asof),
        ("slot", slot),
        ("ranking", ranking.upper() if ranking else None),
        ("limit_markets", limit_markets),
    ):
        if (expected is not None or key == "limit_markets") and request.get(
            key
        ) != expected:
            raise SnapshotError(f"delivery snapshot identity mismatch: {key}")
    if document["asof"] != request["asof"] or document["slot"] != request["slot"]:
        raise SnapshotError("delivery snapshot top-level/request mismatch")
    if model_id is not None and document["model"]["id"] != model_id:
        raise SnapshotError("delivery snapshot model identity mismatch")
    result = dict(document)
    try:
        result["snapshot_path"] = str(p.relative_to(native._ROOT))
    except ValueError:
        result["snapshot_path"] = str(p)
    return result


def _get_or_create_recommend_snapshot(
    asof,
    *,
    slot="open",
    ranking="R1",
    limit_markets=None,
    model_id=None,
    root=None,
    scorer=None,
):
    if slot == "auto":
        for candidate_slot in ("open", "preopen"):
            target = snapshot_path(
                asof, candidate_slot, ranking, limit_markets, root=root
            )
            if target.exists():
                return load_snapshot(
                    target,
                    asof=asof,
                    slot=candidate_slot,
                    ranking=ranking,
                    limit_markets=limit_markets,
                    model_id=model_id,
                )
    elif snapshot_path(asof, slot, ranking, limit_markets, root=root).exists():
        return load_snapshot(
            snapshot_path(asof, slot, ranking, limit_markets, root=root),
            asof=asof,
            slot=slot,
            ranking=ranking,
            limit_markets=limit_markets,
            model_id=model_id,
        )
    captured = {}

    def capture(*args, **kwargs):
        source = scorer
        if source is None:
            from signals.recommend import score_candidates

            source = score_candidates
        captured["code"] = native._git_code_metadata()
        captured["data"] = native._data_metadata()
        captured["policy"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        captured["environment"] = native._environment_metadata()
        captured["started"] = datetime.now(timezone.utc).isoformat()
        result = source(*args, **kwargs)
        captured["completed"] = datetime.now(timezone.utc).isoformat()
        captured["result"] = copy.deepcopy(result)
        if (
            captured["code"] != native._git_code_metadata()
            or captured["data"] != native._data_metadata()
            or captured["policy"]
            != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        ):
            raise SnapshotError("delivery inputs or source changed during inference")
        return result

    try:
        return native.get_or_create_recommend_snapshot(
            asof,
            slot=slot,
            ranking=ranking,
            limit_markets=limit_markets,
            model_id=model_id,
            root=root,
            scorer=capture,
        )
    except SnapshotError as exc:
        if slot in {"open", "preopen"}:
            cached = snapshot_path(asof, slot, ranking, limit_markets, root=root)
            if cached.exists():
                return load_snapshot(
                    cached,
                    asof=asof,
                    slot=slot,
                    ranking=ranking,
                    limit_markets=limit_markets,
                    model_id=model_id,
                )
        if (
            ranking.upper() != "R1"
            or "probability nesting violated in top-k:" not in str(exc)
            or "result" not in captured
        ):
            raise
    raw = captured["result"]
    resolved_slot = raw["slot"]
    target = snapshot_path(asof, resolved_slot, ranking, limit_markets, root=root)
    with native._exclusive_lock(target):
        if target.exists():
            return load_snapshot(
                target,
                asof=asof,
                slot=resolved_slot,
                ranking=ranking,
                limit_markets=limit_markets,
                model_id=model_id,
            )
        document = native._build_document(
            raw,
            ranking=ranking,
            limit_markets=limit_markets,
            model_id=model_id,
            created_at=captured["completed"],
            decision_started_at=captured["started"],
            decision_completed_at=captured["completed"],
            code_metadata=captured["code"],
            data_metadata=captured["data"],
            environment_metadata=captured["environment"],
        )
        document["probability_quality"] = probability_quality(document)
        document["delivery_policy_source_sha256"] = captured["policy"]
        digest = native._document_digest(document)
        document.update(payload_sha256=digest, snapshot_id=f"recommend-{digest[:20]}")
        _validate_marked(document, target)
        native._atomic_write_json(target, document)
    return load_snapshot(
        target,
        asof=asof,
        slot=resolved_slot,
        ranking=ranking,
        limit_markets=limit_markets,
        model_id=model_id,
    )


def get_or_create_recommend_snapshot(
    asof,
    *,
    slot="open",
    ranking="R1",
    limit_markets=None,
    model_id=None,
    root=None,
    scorer=None,
):
    # Keep the frozen creation attempt AND its marked fallback in one delivery
    # transaction. Another sender/ledger process must not refit in that gap.
    anchor = snapshot_path(asof, "open", ranking, limit_markets, root=root)
    anchor.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(anchor.parent / ("." + anchor.name + ".delivery.lock")):
        return _get_or_create_recommend_snapshot(
            asof,
            slot=slot,
            ranking=ranking,
            limit_markets=limit_markets,
            model_id=model_id,
            root=root,
            scorer=scorer,
        )
