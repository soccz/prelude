"""Read-only execution sensitivity for stored L1 and R1 picks, never a reselector."""

from datetime import timedelta

import numpy as np
import pandas as pd

from data.database import connect_readonly
from ops.recommendation_evidence import EvidenceUnavailable
from scripts import evaluate_recommend_entry_delay as delay
from signals.recommend_book_readiness import CONFIG
from signals.recommend_experiment_eval import METRICS, _require, _validate_outcomes


def aggregate(paths, selection):
    """Independently reconstruct every cached mean/difference from pick outcomes."""
    coins = set(selection["control"]) | set(selection["challenger"])
    _require(set(paths) == coins, "execution coin set mismatch")
    result = {}
    for minutes in CONFIG["delays_minutes"]:
        key = str(minutes)
        for coin in coins:
            _require(
                set(paths[coin]) == {str(n) for n in CONFIG["delays_minutes"]},
                "execution delay set mismatch",
            )
            item = paths[coin][key]
            _require(item["label_status"] == "labeled", "incomplete execution path")
            raw = item["raw_input"]
            _require(
                raw["market"] == coin and len(raw["raw_input_sha256"]) == 64,
                "invalid execution path identity",
            )
            _validate_outcomes(
                pd.DataFrame(
                    [{"coin": coin, "label_status": "labeled", **item["outcomes"]}]
                )
            )
        arms = {}
        for arm in ("control", "challenger"):
            picks = [delay._metrics(paths[c][key]["outcomes"]) for c in selection[arm]]
            _require(len(picks) == 3, "three execution picks required")
            arms[arm] = {m: float(np.mean([p[m] for p in picks])) for m in METRICS}
        arms["challenger_minus_control"] = {
            m: arms["challenger"][m] - arms["control"][m] for m in METRICS
        }
        result[key] = arms
    return result


def evaluate(score, evidence, *, db_path, now):
    snapshot, label = evidence["snapshot"], evidence["label"]
    start = delay._aware(score["entry_at"])
    _require(start == delay._aware(label["execution_start_at"]), "entry mismatch")
    if delay._aware(now) < start + timedelta(
        days=1, minutes=max(CONFIG["delays_minutes"])
    ):
        raise EvidenceUnavailable("delayed_24h_window_not_mature")
    candidates = {r["coin"]: r for r in snapshot["universe"]}
    canonical = {r["coin"]: r for r in label["rows"]}
    selection = score["selection"]
    coins = sorted(set(selection["control"]) | set(selection["challenger"]))
    paths = {}
    with connect_readonly(db_path) as conn:
        conn.execute("BEGIN")
        for coin in coins:
            paths[coin] = {}
            for minutes in CONFIG["delays_minutes"]:
                at = start + timedelta(minutes=minutes)
                assessment = delay.assess_15m_window(coin, at, connection=conn)
                outcome = delay._label_candidate(
                    candidates[coin],
                    assessment,
                    snapshot_id=snapshot["snapshot_id"],
                    snapshot_hash=snapshot["payload_sha256"],
                    execution={"execution_start_at": at.isoformat()},
                )
                if outcome["label_status"] != "labeled":
                    raise EvidenceUnavailable("execution_path_incomplete")
                if minutes == 0:
                    _require(
                        not delay._parity_differences(outcome, canonical[coin]),
                        "canonical zero-delay parity failed",
                    )
                paths[coin][str(minutes)] = {
                    "label_status": "labeled",
                    "outcomes": {k: outcome[k] for k in delay.PARITY_FIELDS},
                    "raw_input": delay._path_manifest(conn, coin, at),
                }
        conn.rollback()
    return {
        "status": "evaluated",
        "paths": paths,
        "metrics": aggregate(paths, selection),
        "access": "read_only_single_transaction",
        "horizon": "each_start_plus_24h",
        "actual_execution": False,
    }
