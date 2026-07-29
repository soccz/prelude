from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import signals.pump_detector_v1 as pump_detector_v1
from data.market_universe import SIGNAL_EXCLUDED_KRW_MARKETS
from signals.model_registry import get_model
from signals.pump_detector_v1 import apply_pump_rules


def _set_canonical_live_paths(
    mod,
    monkeypatch,
    *,
    day: str,
    ledger: Path,
    decisions: Path,
) -> None:
    monkeypatch.setattr(mod, "_today_kst", lambda: day)
    monkeypatch.setattr(
        mod,
        "_now_kst",
        lambda: datetime.fromisoformat(f"{day}T09:15:00").replace(
            tzinfo=mod.KST
        ),
    )
    monkeypatch.setattr(mod, "PUMP_HUNTER_LEDGER", str(ledger))
    monkeypatch.setattr(mod, "PUMP_V1_DECISION_ROOT", str(decisions))


def _upbit_daily_frame(
    *,
    daily_growth: float,
    quote_volume: float,
) -> pd.DataFrame:
    periods = 30
    timestamps = pd.date_range(
        end="2026-07-26 09:00:00",
        periods=periods,
        freq="D",
    )
    close = 100.0 * np.power(1.0 + daily_growth, np.arange(periods))
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": quote_volume / close,
            "quote_volume": np.full(periods, quote_volume),
        }
    )


def test_btc_regime_propagates_candle_load_failure(monkeypatch):
    def fail_load(_db_path, _market):
        raise OSError("BTC candle database unavailable")

    monkeypatch.setattr(pump_detector_v1, "load_candles", fail_load)

    with pytest.raises(OSError, match="database unavailable"):
        pump_detector_v1._btc_regime_for_feature_date(
            "unavailable.db",
            pd.Timestamp("2026-07-25 09:00:00"),
        )


def test_btc_regime_is_unknown_when_candle_query_is_empty(monkeypatch):
    monkeypatch.setattr(
        pump_detector_v1,
        "load_candles",
        lambda _db_path, _market: pd.DataFrame(),
    )

    assert (
        pump_detector_v1._btc_regime_for_feature_date(
            "empty.db",
            pd.Timestamp("2026-07-25 09:00:00"),
        )
        == "unknown"
    )


def test_btc_regime_is_unknown_without_feature_date_candle(monkeypatch):
    btc = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-07-24 09:00:00")],
            "close": [100.0],
        }
    )
    monkeypatch.setattr(
        pump_detector_v1,
        "load_candles",
        lambda _db_path, _market: btc.copy(),
    )

    assert (
        pump_detector_v1._btc_regime_for_feature_date(
            "insufficient.db",
            pd.Timestamp("2026-07-25 09:00:00"),
        )
        == "unknown"
    )


@pytest.mark.parametrize(
    "excluded_market",
    sorted(SIGNAL_EXCLUDED_KRW_MARKETS),
)
def test_excluded_market_extremes_cannot_change_v1_liquidity_or_signal_ranks(
    monkeypatch,
    excluded_market,
):
    frames = {
        "KRW-A": _upbit_daily_frame(
            daily_growth=0.05,
            quote_volume=300.0,
        ),
        "KRW-B": _upbit_daily_frame(
            daily_growth=0.03,
            quote_volume=200.0,
        ),
        "KRW-C": _upbit_daily_frame(
            daily_growth=0.01,
            quote_volume=100.0,
        ),
        # Both liquidity and momentum are deliberately dominant. If this row
        # reaches either rank, it displaces KRW-B and changes KRW-A's pct rank.
        excluded_market: _upbit_daily_frame(
            daily_growth=0.90,
            quote_volume=1e15,
        ),
    }
    market_sets = iter(
        [
            ["KRW-A", "KRW-B", "KRW-C"],
            [excluded_market, "KRW-A", "KRW-B", "KRW-C"],
        ]
    )
    monkeypatch.setattr(
        pump_detector_v1,
        "list_markets",
        lambda _db: next(market_sets),
    )
    monkeypatch.setattr(
        pump_detector_v1,
        "load_candles",
        lambda _db, market: frames[market].copy(),
    )
    monkeypatch.setattr(
        pump_detector_v1,
        "_btc_regime_for_feature_date",
        lambda *_args: "bull_quiet",
    )

    baseline = pump_detector_v1.build_feature_frame(
        "2026-07-26",
        db_path="unused.db",
        top_universe=2,
        limit_markets=2,
    )
    attacked = pump_detector_v1.build_feature_frame(
        "2026-07-26",
        db_path="unused.db",
        top_universe=2,
        limit_markets=2,
    )

    columns = [
        "market",
        "liq_rank_daily",
        "roc_7d_rank",
        "atr_pct_14_rank",
    ]
    pd.testing.assert_frame_equal(
        baseline[columns].reset_index(drop=True),
        attacked[columns].reset_index(drop=True),
    )
    assert baseline["market"].tolist() == ["KRW-A", "KRW-B"]
    assert excluded_market not in set(attacked["market"])
    assert apply_pump_rules(baseline)["market"].tolist() == ["KRW-A"]
    assert apply_pump_rules(attacked)["market"].tolist() == ["KRW-A"]


def test_v1_excludes_market_with_gap_in_required_daily_history(
    monkeypatch,
):
    frame = _upbit_daily_frame(
        daily_growth=0.05,
        quote_volume=300.0,
    )
    frame = frame[
        frame["timestamp"] != pd.Timestamp("2026-07-20 09:00:00")
    ].copy()
    monkeypatch.setattr(
        pump_detector_v1,
        "list_markets",
        lambda _db: ["KRW-A"],
    )
    monkeypatch.setattr(
        pump_detector_v1,
        "load_candles",
        lambda _db, _market: frame.copy(),
    )
    monkeypatch.setattr(
        pump_detector_v1,
        "_btc_regime_for_feature_date",
        lambda *_args: "bull_quiet",
    )

    result = pump_detector_v1.build_feature_frame(
        pd.Timestamp("2026-07-26 09:05:00", tz="Asia/Seoul"),
        db_path="unused.db",
    )

    assert result.empty


def test_pump_rule_detector_fires_on_mined_roc_and_atr_rule():
    frame = pd.DataFrame([
        {
            "market": "KRW-A",
            "roc_7d_rank": 0.86,
            "atr_pct_14_rank": 0.50,
            "atr_pct_14": 0.05,
            "log_return_1d": 0.02,
        },
        {
            "market": "KRW-B",
            "roc_7d_rank": 0.90,
            "atr_pct_14_rank": 0.90,
            "atr_pct_14": 0.08,
            "log_return_1d": 0.05,
        },
        {
            "market": "KRW-C",
            "roc_7d_rank": 0.70,
            "atr_pct_14_rank": 0.95,
            "atr_pct_14": 0.09,
            "log_return_1d": 0.01,
        },
    ])

    out = apply_pump_rules(frame, max_candidates=20)

    assert out["market"].tolist() == ["KRW-B", "KRW-A"]
    assert out.loc[out["market"] == "KRW-A", "pump20_rule"].item()
    assert out.loc[out["market"] == "KRW-B", "pump15_rule"].item()
    assert "KRW-C" not in set(out["market"])


def test_model_registry_has_pump_hunter_as_challenger_only():
    spec = get_model("pump_hunter")

    assert spec.ledger_path == "output/shadow_ledger_pump_hunter.csv"
    assert spec.slots == ["open"]
    assert spec.challenger_only is True
    assert spec.predict_ref == "signals.pump_detector_v1:score_pump_candidates"


def test_pump_forward_activation_boundary_is_shared_with_close_gate():
    import scripts.pump_detector_today as v1_runner
    import scripts.pump_detector_v2_today as v2_runner
    from ops.close_input_gate import CLOSE_EVIDENCE_ACTIVATION_DATE

    assert (
        v1_runner.FORWARD_EVIDENCE_ACTIVATION_DATE
        == CLOSE_EVIDENCE_ACTIVATION_DATE
    )
    assert (
        v2_runner.FORWARD_EVIDENCE_ACTIVATION_DATE
        == CLOSE_EVIDENCE_ACTIVATION_DATE
    )


def test_pump_detector_ledger_append_is_idempotent(tmp_path, monkeypatch):
    import scripts.pump_detector_today as mod

    def fake_score(*args, **kwargs):
        return {
            "asof": "2026-06-04",
            "feature_date": "2026-06-03",
            "model_id": "pump_hunter",
            "rule_version": "pump_detector_v1",
            "top_universe": 100,
            "universe_n": 100,
            "n_candidates": 1,
            "rules": {"pump20": "rule-20", "pump15": "rule-15"},
            "candidates": [
                {
                    "market": "KRW-PUMP",
                    "rank": 1,
                    "score": 0.93,
                    "estimated_pump20_prob": 0.064,
                    "dump_risk_flag": False,
                    "btc_regime": "bull_quiet",
                    "entry_open": 100.0,
                    "rule_id": "roc7_rank_pump20",
                    "liq_rank_daily": 12,
                    "roc_7d": 55.0,
                    "roc_7d_rank": 0.91,
                    "atr_pct_14": 0.08,
                    "log_return_1d": 0.04,
                    "pump20_rule": True,
                    "pump15_rule": False,
                    "estimated_pump15_prob": None,
                    "overheated_flag": False,
                }
            ],
        }

    monkeypatch.setattr(mod, "score_pump_candidates", fake_score)
    ledger = tmp_path / "shadow_ledger_pump_hunter.csv"
    decisions = tmp_path / "pump_v1_decisions"
    _set_canonical_live_paths(
        mod,
        monkeypatch,
        day="2026-06-04",
        ledger=ledger,
        decisions=decisions,
    )

    mod.append_today(
        "2026-06-04",
        ledger_path=str(ledger),
        decision_root=decisions,
    )
    mod.append_today(
        "2026-06-04",
        ledger_path=str(ledger),
        decision_root=decisions,
    )

    df = pd.read_csv(ledger)
    assert len(df) == 1
    assert df.iloc[0]["coin"] == "KRW-PUMP"
    assert df.iloc[0]["model_id"] == "pump_hunter"
    assert df.iloc[0]["status"] == "open"
    decision_path = decisions / "2026-06-04.json"
    payload = json.loads(decision_path.read_text(encoding="utf-8"))
    assert df.iloc[0]["snapshot_id"] == payload["decision_id"]
    assert Path(df.iloc[0]["snapshot_path"]) == decision_path

    df.loc[0, "score"] = 0.01
    df.to_csv(ledger, index=False)
    with pytest.raises(RuntimeError, match="immutable decision row conflict"):
        mod.append_today(
            "2026-06-04",
            ledger_path=str(ledger),
            decision_root=decisions,
        )


def test_pump_v1_persists_healthy_zero_pick_decision_without_empty_ledger(
    tmp_path,
    monkeypatch,
):
    import scripts.pump_detector_today as mod

    monkeypatch.setattr(
        mod,
        "score_pump_candidates",
        lambda *_args, **_kwargs: {
            "asof": "2026-06-04",
            "feature_date": "2026-06-03",
            "model_id": "pump_hunter",
            "rule_version": "pump_detector_v1",
            "top_universe": 100,
            "universe_n": 100,
            "n_candidates": 0,
            "rules": {"pump20": "rule-20", "pump15": "rule-15"},
            "candidates": [],
        },
    )
    ledger = tmp_path / "shadow_ledger_pump_hunter.csv"
    decisions = tmp_path / "pump_v1_decisions"
    _set_canonical_live_paths(
        mod,
        monkeypatch,
        day="2026-06-04",
        ledger=ledger,
        decisions=decisions,
    )

    result = mod.append_today(
        "2026-06-04",
        ledger_path=str(ledger),
        decision_root=decisions,
    )

    assert result["n_candidates"] == 0
    assert not ledger.exists()
    payload = json.loads(
        (decisions / "2026-06-04.json").read_text(encoding="utf-8")
    )
    assert payload["schema"] == mod.PUMP_V1_DECISION_SCHEMA
    assert payload["decision"]["candidates"] == []
    assert (
        payload["decision"]["execution_provenance"]["evidence_class"]
        == "canonical_forward"
    )


def test_pump_v1_normalizes_scorer_date_objects_to_strict_json():
    import scripts.pump_detector_today as mod

    validated = mod._validate_decision_result(
        {
            "asof": "2026-07-26",
            "feature_date": date(2026, 7, 25),
            "model_id": "pump_hunter",
            "rule_version": "pump_detector_v1",
            "top_universe": 100,
            "universe_n": 100,
            "n_candidates": 0,
            "rules": {"pump20": "rule-20", "pump15": "rule-15"},
            "candidates": [],
        }
    )

    assert validated["feature_date"] == "2026-07-25"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("estimated_pump20_prob", True, "finite number"),
        ("estimated_pump20_prob", float("nan"), "finite number"),
        ("estimated_pump20_prob", 1.01, r"must be in \[0, 1\]"),
        ("estimated_pump15_prob", -0.01, r"must be in \[0, 1\]"),
        ("dump_risk_flag", 1, "must be boolean"),
        ("overheated_flag", "false", "must be boolean"),
    ],
)
def test_pump_v1_rejects_invalid_optional_candidate_contract(
    field,
    value,
    message,
):
    import scripts.pump_detector_today as mod

    candidate = {
        "market": "KRW-TEST",
        "rank": 1,
        "score": 0.9,
        "entry_open": 100.0,
        "roc_7d_rank": 0.9,
        "btc_regime": "bull_quiet",
        "rule_id": "rule-20",
        field: value,
    }
    decision = {
        "asof": "2026-07-26",
        "feature_date": "2026-07-25",
        "model_id": "pump_hunter",
        "rule_version": "pump_detector_v1",
        "top_universe": 100,
        "universe_n": 100,
        "n_candidates": 1,
        "rules": {"pump20": "rule-20", "pump15": "rule-15"},
        "candidates": [candidate],
    }

    with pytest.raises(ValueError, match=message):
        mod._validate_decision_result(decision)


def test_pump_v1_decision_is_immutable_and_unhealthy_zero_pick_is_rejected(
    tmp_path,
):
    import scripts.pump_detector_today as mod

    decision = {
        "asof": "2026-06-04",
        "feature_date": "2026-06-03",
        "model_id": "pump_hunter",
        "rule_version": "pump_detector_v1",
        "top_universe": 100,
        "universe_n": 100,
        "n_candidates": 0,
        "rules": {"pump20": "rule-20", "pump15": "rule-15"},
        "candidates": [],
    }
    decisions = tmp_path / "decisions"
    path = mod.persist_decision(decision, decision_root=decisions)
    before = path.read_bytes()
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["schema"] == mod.PUMP_V1_LEGACY_DECISION_SCHEMA
    assert mod._validate_decision_document(
        persisted,
        decision,
        path,
    ) == decision

    changed = dict(decision)
    changed["rules"] = {"pump20": "changed", "pump15": "rule-15"}
    with pytest.raises(RuntimeError, match="different pump v1 decision"):
        mod.persist_decision(changed, decision_root=decisions)
    assert path.read_bytes() == before

    unhealthy = dict(decision)
    unhealthy["asof"] = "2026-06-05"
    unhealthy["feature_date"] = "2026-06-04"
    unhealthy["universe_n"] = 0
    with pytest.raises(ValueError, match="universe_n must be positive"):
        mod.persist_decision(unhealthy, decision_root=decisions)
    assert not (decisions / "2026-06-05.json").exists()


@pytest.mark.parametrize(
    "corruption",
    ["duplicate_key", "nan"],
)
def test_pump_v1_existing_manifest_uses_strict_json(
    tmp_path,
    corruption,
):
    import scripts.pump_detector_today as mod

    decision = {
        "asof": "2026-06-04",
        "feature_date": "2026-06-03",
        "model_id": "pump_hunter",
        "rule_version": "pump_detector_v1",
        "top_universe": 100,
        "universe_n": 100,
        "n_candidates": 0,
        "rules": {"pump20": "rule-20", "pump15": "rule-15"},
        "candidates": [],
    }
    path = mod.persist_decision(decision, decision_root=tmp_path)
    raw = path.read_text(encoding="utf-8")
    if corruption == "duplicate_key":
        raw = raw.replace(
            f'"schema": "{mod.PUMP_V1_LEGACY_DECISION_SCHEMA}"',
            (
                f'"schema": "{mod.PUMP_V1_LEGACY_DECISION_SCHEMA}", '
                f'"schema": "{mod.PUMP_V1_LEGACY_DECISION_SCHEMA}"'
            ),
            1,
        )
    else:
        raw = raw.replace('"universe_n": 100', '"universe_n": NaN', 1)
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(RuntimeError, match="decision read failed"):
        mod.persist_decision(decision, decision_root=tmp_path)


def test_pump_v1_live_write_rejects_noncanonical_request_before_scoring(
    tmp_path,
    monkeypatch,
):
    import scripts.pump_detector_today as mod

    calls: list[str] = []
    monkeypatch.setattr(
        mod,
        "_now_kst",
        lambda: datetime(2026, 7, 27, 9, 15, tzinfo=mod.KST),
    )
    monkeypatch.setattr(
        mod,
        "score_pump_candidates",
        lambda *_args, **_kwargs: calls.append("score"),
    )

    with pytest.raises(RuntimeError, match="stale pump v1 live run"):
        mod.append_today(
            "2026-07-26",
            ledger_path=str(tmp_path / "ledger.csv"),
            decision_root=tmp_path / "decisions",
        )

    assert calls == []
    assert list(tmp_path.iterdir()) == []


def test_pump_v1_live_write_rejects_development_knobs(
    tmp_path,
    monkeypatch,
):
    import scripts.pump_detector_today as mod

    ledger = tmp_path / "ledger.csv"
    decisions = tmp_path / "decisions"
    _set_canonical_live_paths(
        mod,
        monkeypatch,
        day="2026-07-27",
        ledger=ledger,
        decisions=decisions,
    )
    calls: list[str] = []
    monkeypatch.setattr(
        mod,
        "score_pump_candidates",
        lambda *_args, **_kwargs: calls.append("score"),
    )

    with pytest.raises(RuntimeError, match="limit_markets"):
        mod.append_today(
            "2026-07-27",
            ledger_path=str(ledger),
            decision_root=decisions,
            limit_markets=3,
        )

    assert calls == []
    assert list(tmp_path.iterdir()) == []


def test_pump_v1_dry_run_allows_replay_knobs_without_filesystem_writes(
    tmp_path,
    monkeypatch,
):
    import scripts.pump_detector_today as mod

    monkeypatch.setattr(
        mod,
        "score_pump_candidates",
        lambda *_args, **_kwargs: {
            "asof": "2026-01-02",
            "feature_date": "2026-01-01",
            "model_id": "pump_hunter",
            "rule_version": "pump_detector_v1",
            "top_universe": 7,
            "universe_n": 7,
            "n_candidates": 1,
            "rules": {"pump20": "rule-20", "pump15": "rule-15"},
            "candidates": [
                {
                    "market": "KRW-TEST",
                    "rank": 1,
                    "score": 0.9,
                    "entry_open": 100.0,
                    "roc_7d_rank": 0.9,
                    "btc_regime": "bull_quiet",
                    "rule_id": "rule-20",
                }
            ],
        },
    )

    result = mod.append_today(
        "2026-01-02",
        dry_run=True,
        ledger_path=str(tmp_path / "custom" / "ledger.csv"),
        decision_root=tmp_path / "custom" / "decisions",
        top_universe=7,
        max_candidates=2,
        limit_markets=10,
    )

    assert result["n_candidates"] == 1
    assert list(tmp_path.iterdir()) == []


def test_pump_v1_post_activation_legacy_manifest_is_not_forward_valid(
    tmp_path,
):
    import scripts.pump_detector_today as mod

    decision = {
        "asof": "2026-07-27",
        "feature_date": "2026-07-26",
        "model_id": "pump_hunter",
        "rule_version": "pump_detector_v1",
        "top_universe": 100,
        "universe_n": 100,
        "n_candidates": 0,
        "rules": {"pump20": "rule-20", "pump15": "rule-15"},
        "candidates": [],
    }

    with pytest.raises(RuntimeError, match="legacy decision.*not forward-valid"):
        mod.persist_decision(decision, decision_root=tmp_path)

    assert [
        path
        for path in tmp_path.iterdir()
        if not path.name.endswith(".lock")
    ] == []


def test_pump_v1_forward_manifest_seals_outer_recorded_at(
    tmp_path,
    monkeypatch,
):
    import scripts.pump_detector_today as mod

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = datetime(
                2026,
                7,
                27,
                0,
                5,
                tzinfo=timezone.utc,
            )
            return value if tz is None else value.astimezone(tz)

    decision = {
        "asof": "2026-07-27",
        "feature_date": "2026-07-26",
        "model_id": "pump_hunter",
        "rule_version": "pump_detector_v1",
        "top_universe": 100,
        "universe_n": 100,
        "n_candidates": 0,
        "rules": {"pump20": "rule-20", "pump15": "rule-15"},
        "candidates": [],
    }
    decision = mod._with_forward_provenance(decision)
    monkeypatch.setattr(mod, "datetime", FixedDatetime)

    path = mod.persist_decision(decision, decision_root=tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert len(payload["integrity_sha256"]) == 64
    assert mod.manifest_digest_matches(
        payload,
        digest_key="integrity_sha256",
    )
    unsigned = dict(payload)
    unsigned.pop("integrity_sha256")
    with pytest.raises(RuntimeError, match="outer schema mismatch"):
        mod._validate_decision_document(unsigned, decision, path)

    payload["recorded_at"] = "2026-07-27T00:06:00+00:00"
    with pytest.raises(RuntimeError, match="outer integrity mismatch"):
        mod._validate_decision_document(payload, decision, path)
