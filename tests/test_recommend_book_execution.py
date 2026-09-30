from copy import deepcopy
import hashlib

import pytest

from ops import recommend_book_execution as execution
from test_recommend_entry_delay import make_case, no_network as no_network  # noqa: F401


def paths_from_canonical(score, evidence):
    rows = {r["coin"]: r for r in evidence["label"]["rows"]}
    return {
        coin: {
            str(delay): {
                "label_status": "labeled",
                "outcomes": {k: rows[coin][k] for k in execution.delay.PARITY_FIELDS},
                "raw_input": {"market": coin, "raw_input_sha256": "a" * 64},
            }
            for delay in (0, 15, 30)
        }
        for coin in set(score["selection"]["control"])
        | set(score["selection"]["challenger"])
    }


def prepare(tmp_path, monkeypatch):
    options, evidence, db, start = make_case(tmp_path, monkeypatch)
    evidence["snapshot"]["universe"] = evidence["snapshot"]["top3"]
    coins = [r["coin"] for r in evidence["snapshot"]["top3"]]
    return (
        {
            "entry_at": start.isoformat(),
            "selection": {"control": coins, "challenger": list(reversed(coins))},
        },
        evidence,
        db,
        options["now"],
    )


def test_real_sqlite_no_write_zero_parity_and_fresh_24h_delays(tmp_path, monkeypatch):
    score, evidence, db, now = prepare(tmp_path, monkeypatch)
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    result = execution.evaluate(score, evidence, db_path=db, now=now)
    assert result["status"] == "evaluated" and result["actual_execution"] is False
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before
    for delay in ("0", "15", "30"):
        assert all(
            v == pytest.approx(0)
            for v in result["metrics"][delay]["challenger_minus_control"].values()
        )
    assert (
        result["metrics"]["30"]["control"]["up10"]
        > result["metrics"]["0"]["control"]["up10"]
    )
    assert result["metrics"] == execution.aggregate(result["paths"], score["selection"])


def test_zero_parity_failure_and_premature_window_block(tmp_path, monkeypatch):
    score, evidence, db, now = prepare(tmp_path, monkeypatch)
    bad = deepcopy(evidence)
    bad["label"]["rows"][0]["eod_return_net"] += 0.1
    with pytest.raises(ValueError, match="parity"):
        execution.evaluate(score, bad, db_path=db, now=now)
    with pytest.raises(execution.EvidenceUnavailable, match="not_mature"):
        execution.evaluate(
            score, evidence, db_path=db, now=execution.delay._aware(score["entry_at"])
        )


def test_invalid_or_missing_cached_execution_outcomes_block(tmp_path, monkeypatch):
    score, evidence, db, now = prepare(tmp_path, monkeypatch)
    result = execution.evaluate(score, evidence, db_path=db, now=now)
    paths = result["paths"]
    paths["KRW-T1"]["0"]["outcomes"]["up10"] = "yes"
    with pytest.raises(ValueError):
        execution.aggregate(paths, score["selection"])
