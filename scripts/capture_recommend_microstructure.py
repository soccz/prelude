#!/usr/bin/env python3
"""Independent public capture -> causal feature -> write-once shadow trial.

Never sends recommendations, trains a model, changes a snapshot or places an
order. A failure here is reported to the independent collector service only.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import collector_upbit_microstructure as collector  # noqa: E402
from ops.artifact_provenance import file_identity  # noqa: E402
from signals.recommend_microstructure import read_recommend_microstructure  # noqa: E402
from signals.recommend_microstructure_trial import (  # noqa: E402
    evaluate_microstructure_trials,
    record_microstructure_trial,
    write_new_trial_report as write_feature_once,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRIAL_ROOT = ROOT / "output" / "recommend_microstructure_trials"
SHORTLIST_START = date(2026, 9, 10)


def record_trade_shortlist(*args, **kwargs):
    # Lazy import: a broken new trial must not prevent the original score.
    from signals.recommend_trade_shortlist_trial import record_trade_shortlist_trial

    return record_trade_shortlist_trial(*args, **kwargs)


def evaluate_trade_shortlist(*args, **kwargs):
    from signals.recommend_trade_shortlist_eval import evaluate_trade_shortlist_trials

    return evaluate_trade_shortlist_trials(*args, **kwargs)


def capture_storage_preflight(
    output_root: Path, trial_root: Path, payload_budget: int
) -> list[dict]:
    """Admission check only: never deletes evidence or reserves filesystem space.

    Initial allowance: two streams, 4x payload for envelopes/working files, plus
    1 GiB spare. Revisit with regular-session measurements; this is not a quota,
    a proven worst-case expansion bound, or protection from concurrent writers.
    """
    if (
        isinstance(payload_budget, bool)
        or not isinstance(payload_budget, int)
        or payload_budget <= 0
    ):
        raise ValueError("payload budget must be a positive integer")
    required = 1024**3 + 4 * 2 * payload_budget
    observations = []
    for root in (output_root, trial_root):
        directory = Path(root).absolute()
        while not directory.exists() and not directory.is_symlink():
            directory = directory.parent
        if not directory.is_dir():
            raise ValueError(f"capture storage parent must be a directory: {directory}")
        usage = os.statvfs(directory)
        available, unit = usage.f_bavail, usage.f_frsize
        if (
            isinstance(available, bool)
            or not isinstance(available, int)
            or isinstance(unit, bool)
            or not isinstance(unit, int)
            or unit <= 0
        ):
            raise RuntimeError(f"invalid filesystem space response: {directory}")
        available_bytes = max(0, available) * unit
        if available_bytes < required:
            raise RuntimeError(
                f"insufficient capture storage: {directory}: "
                f"available={available_bytes}, required={required} bytes"
            )
        observations.append(
            {
                "checked_directory": str(directory),
                "available_bytes": available_bytes,
                "required_bytes": required,
            }
        )
    return observations


def run_session(
    config: collector.CaptureConfig,
    *,
    trial_root: Path = DEFAULT_TRIAL_ROOT,
    shortlist_root: Path | None = None,
) -> dict:
    if config.cutoff_slot != "open":
        raise ValueError("scheduled feature session requires the real open snapshot")
    storage = capture_storage_preflight(
        config.output_root, trial_root, config.max_raw_bytes_per_channel
    )
    asof = config.asof or datetime.now(ZoneInfo("Asia/Seoul")).date()
    shortlist_due = asof >= SHORTLIST_START
    shortlist_root = (
        shortlist_root or Path(trial_root).parent / "recommend_trade_shortlist_trials"
    )
    research_errors = []
    if shortlist_due:
        try:
            storage.extend(
                capture_storage_preflight(
                    config.output_root, shortlist_root, config.max_raw_bytes_per_channel
                )
            )
        except (OSError, ValueError, RuntimeError) as exc:
            research_errors.append(f"shortlist_storage:{type(exc).__name__}:{exc}")
    capture = collector.run_capture(config)
    report = {
        "scope": "public_record_only_no_live_change",
        "storage_preflight": storage,
        "capture_manifest": str(capture.manifest_path),
        "capture_complete": capture.complete,
        "feature_evidence_valid": False,
        "feature_path": None,
        "trial": None,
        "evaluation_path": None,
        "effect_status": "not_evaluated",
        "shortlist_due": shortlist_due,
        "shortlist_trial": None,
        "shortlist_evaluation_path": None,
        "research_errors": research_errors,
    }
    source = capture.manifest.get("cutoff_source") or {}
    snapshot_path = Path(source["path"]) if source.get("path") else None
    if snapshot_path is None or not snapshot_path.is_file():
        report["unavailable_reason"] = "original_snapshot_not_available"
        return report
    manifest_before = file_identity(capture.manifest_path, root=ROOT)
    feature = read_recommend_microstructure(snapshot_path, capture.manifest_path)
    if file_identity(capture.manifest_path, root=ROOT) != manifest_before:
        raise ValueError("capture manifest changed during feature generation")
    feature_path = capture.manifest_path.parent / "recommend_features.json"
    write_feature_once(feature_path, feature)
    report["feature_path"] = str(feature_path)
    report["feature_evidence_valid"] = feature["feature_evidence_valid"] is True
    if not report["feature_evidence_valid"]:
        report["unavailable_reason"] = "feature_evidence_invalid"
        return report
    report["trial"] = record_microstructure_trial(
        snapshot_path, feature_path, output_root=trial_root
    )
    if report["trial"].get("status") != "committed":
        report["unavailable_reason"] = "trial_publication_unconfirmed"
        return report
    # The original prediction is already durable. Publish the new experiment
    # BEFORE cumulative evaluation so history cannot delay its entry deadline.
    # New failures are loud but do not suppress the original evaluation.
    if shortlist_due and not research_errors:
        try:
            report["shortlist_trial"] = record_trade_shortlist(
                snapshot_path, feature_path, output_root=shortlist_root
            )
            if report["shortlist_trial"].get("status") != "committed":
                report["research_errors"].append("shortlist_publication_unconfirmed")
        except Exception as exc:
            report["research_errors"].append(
                f"shortlist_publication:{type(exc).__name__}:{exc}"
            )
    # Evaluate only AFTER today's score is durably recorded. A later read error
    # must not delete that prediction or move its score-persisted timestamp.
    try:
        evaluation = evaluate_microstructure_trials(trial_root)
        evaluation_path = capture.manifest_path.parent / "trial_evaluation.json"
        write_feature_once(evaluation_path, evaluation)
        report["evaluation_path"] = str(evaluation_path)
    except Exception as exc:
        if not shortlist_due:
            raise  # Preserve the original pre-launch caller contract.
        report["research_errors"].append(
            f"original_evaluation:{type(exc).__name__}:{exc}"
        )
    if (
        shortlist_due
        and (report.get("shortlist_trial") or {}).get("status") == "committed"
    ):
        try:
            shortlist_evaluation = evaluate_trade_shortlist(shortlist_root)
            shortlist_evaluation_path = (
                capture.manifest_path.parent / "trade_shortlist_evaluation.json"
            )
            write_feature_once(shortlist_evaluation_path, shortlist_evaluation)
            report["shortlist_evaluation_path"] = str(shortlist_evaluation_path)
        except Exception as exc:
            report["research_errors"].append(
                f"shortlist_evaluation:{type(exc).__name__}:{exc}"
            )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asof", type=date.fromisoformat)
    parser.add_argument(
        "--output-root", type=Path, default=collector.DEFAULT_OUTPUT_ROOT
    )
    parser.add_argument(
        "--snapshot-root", type=Path, default=collector.DEFAULT_SNAPSHOT_ROOT
    )
    parser.add_argument("--trial-root", type=Path, default=DEFAULT_TRIAL_ROOT)
    parser.add_argument("--shortlist-root", type=Path)
    parser.add_argument(
        "--orderbook-depth", type=int, choices=(1, 5, 15, 30), default=1
    )
    parser.add_argument("--max-wait-seconds", type=float, default=2400)
    parser.add_argument("--required-warmup-seconds", type=float, default=600)
    parser.add_argument(
        "--max-raw-bytes-per-channel",
        type=int,
        default=collector.DEFAULT_MAX_RAW_BYTES_PER_CHANNEL,
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    try:
        capture_storage_preflight(
            args.output_root, args.trial_root, args.max_raw_bytes_per_channel
        )
        asof = args.asof or datetime.now(ZoneInfo("Asia/Seoul")).date()
        if asof >= SHORTLIST_START:
            try:
                capture_storage_preflight(
                    args.output_root,
                    args.shortlist_root
                    or args.trial_root.parent / "recommend_trade_shortlist_trials",
                    args.max_raw_bytes_per_channel,
                )
            except (OSError, ValueError, RuntimeError) as exc:
                # run_session repeats and reports this failure while preserving
                # the original capture/trial on its independently checked disk.
                logging.warning("new shortlist storage unavailable: %s", exc)
        markets, source, fetch_started, observed = collector._resolve_markets(None)
        config = collector.CaptureConfig(
            markets=markets,
            output_root=args.output_root,
            snapshot_root=args.snapshot_root,
            cutoff_slot="open",
            asof=args.asof,
            orderbook_depth=args.orderbook_depth,
            max_wait_seconds=args.max_wait_seconds,
            required_warmup_seconds=args.required_warmup_seconds,
            max_raw_bytes_per_channel=args.max_raw_bytes_per_channel,
            universe_source=source,
            universe_fetch_started_at_ns=fetch_started,
            universe_observed_at_ns=observed,
        )
        report = run_session(
            config, trial_root=args.trial_root, shortlist_root=args.shortlist_root
        )
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        logging.error("microstructure session failed: %s: %s", type(exc).__name__, exc)
        return 2
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, sort_keys=True))
    complete = (
        report["capture_complete"]
        and report["feature_evidence_valid"]
        and (report.get("trial") or {}).get("status") == "committed"
        and report.get("evaluation_path") is not None
        and not report.get("research_errors")
        and (
            not report.get("shortlist_due")
            or (
                (report.get("shortlist_trial") or {}).get("status") == "committed"
                and report.get("shortlist_evaluation_path") is not None
            )
        )
    )
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
