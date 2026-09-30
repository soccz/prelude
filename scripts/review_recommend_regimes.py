"""Read-only canonical R1 failure/context review; optional NEW report only.

No collector, retraining, trial publication, Telegram or operational state is
invoked. Slot and source versions stay separate. Unavailable evidence is listed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from ops.artifact_provenance import (  # noqa: E402
    canonical_json_bytes, file_identity, sha256_bytes,
)
from ops.recommendation_evidence import (  # noqa: E402
    EvidenceError, EvidenceUnavailable, discover_r1_snapshots, load_recommendation_evidence,
)
from scripts.build_recommend_training_data import write_new_report  # noqa: E402
from signals.recommend_experiment_eval import METRICS, _metric_rows, _validate_outcomes  # noqa: E402
from signals.recommend_regime_replay import (  # noqa: E402
    CONFIG, aware, context_from_snapshot, failure_summary, version_from_snapshot,
)
from signals.recommend_training_data import GENERATOR_SOURCES as EVIDENCE_SOURCES  # noqa: E402

SOURCES = sorted(set(EVIDENCE_SOURCES) | {
    "scripts/review_recommend_regimes.py", "signals/recommend_regime_replay.py",
    "signals/recommend_experiment_eval.py",
})


def _sources():
    return [file_identity(ROOT / path, root=ROOT) for path in SOURCES]


def _record(evidence):
    snapshot, label = evidence["snapshot"], evidence["label"]
    context = context_from_snapshot(snapshot)  # Freeze predictors before outcomes.
    selected = {row["coin"] for row in snapshot["top3"]}
    if len(selected) != 3:
        raise EvidenceUnavailable("not_exactly_three_delivered_picks")
    frame = pd.DataFrame(label["rows"])
    _validate_outcomes(frame)
    picks = frame.index[frame["coin"].isin(selected)].tolist()
    if len(picks) != 3 or not frame.loc[picks, "label_status"].eq("labeled").all():
        raise EvidenceUnavailable("selected_outcome_unavailable")
    universe = frame.index[frame["label_status"].eq("labeled")].tolist()
    return {
        "date": snapshot["asof"], "slot": snapshot["slot"],
        "snapshot_id": snapshot["snapshot_id"], "version": version_from_snapshot(snapshot),
        "context": context, "labeled_universe_size": len(universe),
        "halted_universe_size": len(frame) - len(universe),
        "outcomes": {name: dict(zip(METRICS, _metric_rows(frame, indices).mean(axis=0).tolist(), strict=True))
                     for name, indices in (("control", picks), ("universe", universe))},
    }


def build_review(*, start_date, end_date, slots=("open", "preopen"), now=None,
                 snapshot_root=None, label_root=None, receipt_root=None, n_boot=1000):
    observed = aware(now or datetime.now(timezone.utc))
    snapshot_root = snapshot_root or ROOT / "output/recommend_snapshots"
    label_root = label_root or ROOT / "output/recommend_score_labels"
    receipt_root = receipt_root or ROOT / "output/recommend_receipts"
    sources = _sources()
    paths = discover_r1_snapshots(snapshot_root, start_date=start_date, end_date=end_date, slots=slots)
    records, excluded, manifests = [], [], []
    seen = set()
    for path in paths:
        try:
            evidence = load_recommendation_evidence(
                path, label_root=label_root, receipt_root=receipt_root, now=observed,
            )
            manifests.append(evidence["manifest"])
            record = _record(evidence)
        except EvidenceUnavailable as exc:
            excluded.append({"snapshot_path": str(path), "reason": str(exc)})
            continue
        identity = (record["date"], record["slot"])
        if identity in seen:
            raise EvidenceError("duplicate failure-review date/slot")
        seen.add(identity)
        records.append(record)
    diagnostics = failure_summary(records, n_boot=n_boot)
    inputs = {}
    for manifest in manifests:
        for item in manifest["files"].values():
            if item["path"] in inputs and item != inputs[item["path"]]:
                raise EvidenceError("mixed failure-review input generations")
            inputs[item["path"]] = item
    for item in inputs.values():
        path = Path(item["path"])
        if file_identity(path if path.is_absolute() else ROOT / path, root=ROOT) != item:
            raise EvidenceError("failure-review inputs changed during evaluation")
    if sources != _sources():
        raise EvidenceError("failure-review sources changed during evaluation")
    discovered = {(path.parent.name, path.name.removesuffix("_r1.json")) for path in paths}
    requested = [(start_date + timedelta(days=i)).isoformat()
                 for i in range((end_date - start_date).days + 1)]
    report = {
        "schema": "recommend_regime_failures.v1", "generated_at": observed.isoformat(),
        "config": dict(CONFIG), "deployable": False, "automatic_promotion": False,
        "model_fitted": False, "is_untouched_holdout": False,
        "requested": {"start_date": start_date.isoformat(), "end_date": end_date.isoformat(), "slots": slots},
        "summary": {
            "snapshots_discovered": len(paths), "snapshots_included": len(records),
            "dates_by_slot": dict(Counter(row["slot"] for row in records)),
            "context_counts": dict(Counter(row["context"]["state"] or "unknown" for row in records)),
            "source_versions": len({row["version"] for row in records if row["version"] is not None}),
        },
        "missing_snapshots": [{"date": day, "slot": slot} for day in requested for slot in slots
                              if (day, slot) not in discovered],
        "excluded": excluded, "diagnostics": diagnostics, "daily": records,
        "generator_sources": sources, "input_manifests": manifests,
        "cost_basis": "existing_24h_net_labels_already_deducted_0.0015_not_portfolio_returns",
    }
    return {**report, "payload_sha256": sha256_bytes(canonical_json_bytes(report))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True, type=date.fromisoformat)
    parser.add_argument("--end-date", required=True, type=date.fromisoformat)
    parser.add_argument("--slot", choices=("both", "open", "preopen"), default="both")
    parser.add_argument("--output", type=Path, help="optional NEW JSON inside this project; never overwrite")
    args = parser.parse_args(argv)
    try:
        report = build_review(start_date=args.start_date, end_date=args.end_date,
                              slots=("open", "preopen") if args.slot == "both" else (args.slot,))
        if args.output is not None:
            write_new_report(args.output, report)
        print(json.dumps({"summary": report["summary"], "excluded": report["excluded"],
                          "output": str(args.output) if args.output else None}, ensure_ascii=False, allow_nan=False))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"regime review failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
