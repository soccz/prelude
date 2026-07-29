from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import scripts.health_check as health_check
from ledger.risk import evaluate_risk
from ops.drift_detector import evaluate_drift
from scripts.health_check import (
    check_db_freshness,
    check_drift_state,
    check_log_age,
    check_risk_state,
    check_universe_coverage,
    db_checks_for_channel,
    log_names_for_channel,
)


def test_log_age_rejects_symlink_and_future_mtime(tmp_path):
    target = tmp_path / "target.log"
    target.write_text("ok\n", encoding="utf-8")
    link = tmp_path / "cron.log"
    link.symlink_to(target)

    link_ok, link_message = check_log_age(str(link))

    assert not link_ok
    assert "regular non-symlink" in link_message

    linked_parent = tmp_path / "linked-output"
    linked_parent.symlink_to(tmp_path, target_is_directory=True)
    parent_ok, parent_message = check_log_age(
        str(linked_parent / target.name)
    )

    assert not parent_ok
    assert "parent must be a real directory" in parent_message

    future = datetime.now(timezone.utc) + timedelta(hours=1)
    os.utime(target, (future.timestamp(), future.timestamp()))
    future_ok, future_message = check_log_age(str(target))

    assert not future_ok
    assert "future-dated" in future_message


def _write_health_state(tmp_path, name, payload):
    output = tmp_path / "output"
    output.mkdir(exist_ok=True)
    (output / name).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _valid_risk_state(**overrides):
    payload = {
        "is_active": True,
        "silenced_until": None,
        "trigger_reason": None,
        "last_daily_pnl_pct": None,
        "current_mdd_pct": None,
    }
    payload.update(overrides)
    return payload


def _valid_drift_state(**overrides):
    payload = {
        "state": "OK",
        "triggers": [],
        "last_check": "2026-07-26T00:00:00",
        "details": {},
    }
    payload.update(overrides)
    return payload


def test_health_check_preopen_uses_15m_gate():
    checks = dict(db_checks_for_channel("preopen"))

    assert "data/upbit_d1.db" in checks
    assert "data/upbit_15m.db" in checks
    assert "data/upbit_4h.db" not in checks
    assert checks["data/upbit_15m.db"] == 2


def test_health_check_distribution_uses_4h_gate():
    checks = dict(db_checks_for_channel("distribution"))

    assert "data/upbit_d1.db" in checks
    assert "data/upbit_4h.db" in checks
    assert "data/upbit_15m.db" not in checks
    assert checks["data/upbit_4h.db"] == 8


def test_health_check_recommend_uses_d1_only_gate():
    checks = dict(db_checks_for_channel("recommend"))

    assert checks == {"data/upbit_d1.db": 30}


def test_health_check_preopen_recommend_uses_prior_d1_only_gate():
    checks = dict(db_checks_for_channel("recommend-preopen"))

    assert checks == {"data/upbit_d1.db": 48}


def test_health_check_log_names_are_channel_specific():
    assert log_names_for_channel("preopen", "20260525") == [
        "output/cron_preopen_20260525.log"
    ]
    assert log_names_for_channel("distribution", "20260525") == [
        "output/cron_dist_20260525.log"
    ]
    assert log_names_for_channel("recommend", "20260525") == [
        "output/cron_dist_20260525.log"
    ]
    assert log_names_for_channel("recommend-preopen", "20260525") == [
        "output/cron_preopen_20260525.log"
    ]


def test_risk_state_rejects_duplicate_keys_fail_closed(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    (output / "risk_state.json").write_text(
        '{"is_active": true, "is_active": false}',
        encoding="utf-8",
    )

    ok, message = check_risk_state()

    assert ok is False
    assert "invalid artifact" in message
    assert "duplicate JSON object key" in message


def test_missing_health_states_are_explicit_bootstrap(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    risk_ok, risk_message = check_risk_state()
    drift_ok, drift_message = check_drift_state()

    assert risk_ok is True
    assert drift_ok is True
    assert "BOOTSTRAP_UNINITIALIZED" in risk_message
    assert "BOOTSTRAP_UNINITIALIZED" in drift_message
    assert "evaluator unwired" in risk_message
    assert "evaluator unwired" in drift_message


def test_runtime_health_does_not_report_unwired_risk_or_drift(
    monkeypatch,
    capsys,
):
    """Legacy validators must not masquerade as live production monitors."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["health_check.py", "--channel", "recommend", "--no-telegram"],
    )
    monkeypatch.setattr(
        health_check,
        "check_universe_coverage",
        lambda *_args, **_kwargs: (True, "coverage ok"),
    )
    monkeypatch.setattr(
        health_check,
        "log_names_for_channel",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        health_check,
        "check_risk_state",
        lambda: pytest.fail("unwired risk validator called by runtime health"),
    )
    monkeypatch.setattr(
        health_check,
        "check_drift_state",
        lambda: pytest.fail("unwired drift validator called by runtime health"),
    )

    with pytest.raises(SystemExit) as exc_info:
        health_check.main()

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "risk" not in output
    assert "drift" not in output
    assert "ALL OK" in output


def test_valid_risk_states_follow_active_and_silenced_semantics(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    _write_health_state(tmp_path, "risk_state.json", _valid_risk_state())

    active_ok, active_message = check_risk_state()

    assert active_ok is True
    assert active_message == "risk: ACTIVE"

    _write_health_state(
        tmp_path,
        "risk_state.json",
        _valid_risk_state(
            is_active=False,
            silenced_until="2026-07-27",
            trigger_reason="DAILY_LOSS_LIMIT (-3.0%)",
            last_daily_pnl_pct=-0.03,
        ),
    )

    silenced_ok, silenced_message = check_risk_state()

    assert silenced_ok is False
    assert silenced_message == (
        "risk: SILENCED until 2026-07-27 "
        "(DAILY_LOSS_LIMIT (-3.0%))"
    )

    _write_health_state(
        tmp_path,
        "risk_state.json",
        _valid_risk_state(
            is_active=False,
            silenced_until="2026-08-02",
            trigger_reason="MDD_LIMIT (-15.0%)",
            current_mdd_pct=-0.15,
        ),
    )

    mdd_ok, mdd_message = check_risk_state()

    assert mdd_ok is False
    assert mdd_message == (
        "risk: SILENCED until 2026-08-02 (MDD_LIMIT (-15.0%))"
    )


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({}, "missing="),
        (
            {
                **_valid_risk_state(),
                "is_actve": True,
            },
            "unknown=",
        ),
        (
            _valid_risk_state(is_active=1),
            "is_active must be boolean",
        ),
        (
            _valid_risk_state(is_active=None),
            "is_active must be boolean",
        ),
        (
            _valid_risk_state(last_daily_pnl_pct=True),
            "finite JSON number or null",
        ),
        (
            _valid_risk_state(last_daily_pnl_pct=10**400),
            "finite JSON number or null",
        ),
        (
            _valid_risk_state(trigger_reason=7),
            "trigger_reason must be string or null",
        ),
        (
            _valid_risk_state(silenced_until=20260727),
            "must be YYYY-MM-DD",
        ),
        (
            _valid_risk_state(silenced_until="2026-7-7"),
            "must be YYYY-MM-DD",
        ),
        (
            _valid_risk_state(
                is_active=False,
                silenced_until="2026-07-27 ",
                trigger_reason="DAILY_LOSS_LIMIT (-3.0%)",
                last_daily_pnl_pct=-0.03,
            ),
            "must be YYYY-MM-DD",
        ),
        (
            _valid_risk_state(
                is_active=False,
                trigger_reason="DAILY_LOSS_LIMIT (-3.0%)",
                last_daily_pnl_pct=-0.03,
            ),
            "requires silenced_until",
        ),
        (
            _valid_risk_state(
                is_active=False,
                silenced_until="2026-07-27",
            ),
            "requires trigger_reason",
        ),
        (
            _valid_risk_state(
                is_active=False,
                silenced_until="2026-07-27",
                trigger_reason="MANUAL",
            ),
            "unknown trigger enum",
        ),
        (
            _valid_risk_state(
                is_active=False,
                silenced_until="2026-07-27",
                trigger_reason="DAILY_LOSS_LIMIT (-0.0%)",
            ),
            "matching negative metric",
        ),
        (
            _valid_risk_state(
                is_active=False,
                silenced_until="2026-07-27",
                trigger_reason="MDD_LIMIT (-15.0%) ",
                current_mdd_pct=-0.15,
            ),
            "does not match its metric",
        ),
        (
            _valid_risk_state(
                trigger_reason="DAILY_LOSS_LIMIT (-3.0%)",
                last_daily_pnl_pct=-0.03,
            ),
            "cannot retain silence metadata",
        ),
    ],
)
def test_risk_state_schema_and_semantics_fail_closed(
    tmp_path,
    monkeypatch,
    payload,
    expected,
):
    monkeypatch.chdir(tmp_path)
    _write_health_state(tmp_path, "risk_state.json", payload)

    ok, message = check_risk_state()

    assert ok is False
    assert "invalid artifact" in message
    assert expected in message


def test_drift_state_rejects_nonfinite_number_fail_closed(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    (output / "drift_state.json").write_text(
        '{"state": "OK", "score": NaN}',
        encoding="utf-8",
    )

    ok, message = check_drift_state()

    assert ok is False
    assert "invalid artifact" in message
    assert "non-standard JSON numeric constant" in message


def test_risk_state_rejects_finite_syntax_that_overflows_float(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    (output / "risk_state.json").write_text(
        (
            '{"is_active":true,"silenced_until":null,'
            '"trigger_reason":null,"last_daily_pnl_pct":1e400,'
            '"current_mdd_pct":null}'
        ),
        encoding="utf-8",
    )

    ok, message = check_risk_state()

    assert ok is False
    assert "invalid artifact" in message
    assert "finite JSON number or null" in message


def test_valid_drift_states_follow_trigger_semantics(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    _write_health_state(tmp_path, "drift_state.json", _valid_drift_state())

    ok_ok, ok_message = check_drift_state()

    assert ok_ok is True
    assert ok_message == "drift: OK"

    _write_health_state(
        tmp_path,
        "drift_state.json",
        _valid_drift_state(
            state="WARN",
            triggers=["HIT_RATE_DROP", "DIST_SHIFT"],
            details={
                "hit_7d": 0.2,
                "hit_30d": 0.4,
                "dist_ks_pvalue": 0.005,
            },
        ),
    )

    warn_ok, warn_message = check_drift_state()

    assert warn_ok is False
    assert warn_message == (
        "drift: WARN (['HIT_RATE_DROP', 'DIST_SHIFT'])"
    )

    _write_health_state(
        tmp_path,
        "drift_state.json",
        _valid_drift_state(
            state="FREEZE",
            triggers=["SIGN_FLIP"],
            details={"ic_24h": -0.1, "ic_7d_ma": 0.2},
        ),
    )

    freeze_ok, freeze_message = check_drift_state()

    assert freeze_ok is False
    assert freeze_message == "drift: FREEZE (['SIGN_FLIP'])"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({}, "missing="),
        (
            {
                **_valid_drift_state(),
                "states": "OK",
            },
            "unknown=",
        ),
        (
            _valid_drift_state(state="OK "),
            "must be one of OK, WARN, FREEZE",
        ),
        (
            _valid_drift_state(state=None),
            "must be one of OK, WARN, FREEZE",
        ),
        (
            _valid_drift_state(triggers=None),
            "must be a JSON array",
        ),
        (
            _valid_drift_state(triggers=["DIST_SHIFT "]),
            "unknown trigger enum",
        ),
        (
            _valid_drift_state(triggers=[True]),
            "unknown trigger enum",
        ),
        (
            _valid_drift_state(last_check=None),
            "canonical ISO-8601 datetime",
        ),
        (
            _valid_drift_state(last_check="2026-07-26T00:00:00 "),
            "canonical ISO-8601 datetime",
        ),
        (
            _valid_drift_state(last_check="20260726T000000"),
            "canonical ISO-8601 datetime",
        ),
        (
            _valid_drift_state(details={"hit_7d": True}),
            "finite JSON number or null",
        ),
        (
            _valid_drift_state(details={"hit_rate": 0.2}),
            "unknown keys",
        ),
        (
            _valid_drift_state(details=None),
            "must be a JSON object",
        ),
        (
            _valid_drift_state(details={"dist_ks_pvalue": None}),
            "values cannot be null",
        ),
        (
            _valid_drift_state(
                details={"ic_24h": -1.1, "ic_7d_ma": 0.2},
            ),
            "must be in [-1, 1]",
        ),
        (
            _valid_drift_state(
                details={"hit_7d": 0.2, "hit_30d": 1.1},
            ),
            "must be in [0, 1]",
        ),
        (
            _valid_drift_state(state="WARN"),
            "inconsistent with triggers",
        ),
        (
            _valid_drift_state(
                state="WARN",
                triggers=["DIST_SHIFT", "DIST_SHIFT"],
                details={"dist_ks_pvalue": 0.001},
            ),
            "duplicate values",
        ),
        (
            _valid_drift_state(
                state="WARN",
                triggers=["DIST_SHIFT"],
            ),
            "is missing details",
        ),
        (
            _valid_drift_state(
                details={"hit_7d": 0.2},
            ),
            "requires hit_7d and hit_30d together",
        ),
    ],
)
def test_drift_state_schema_and_semantics_fail_closed(
    tmp_path,
    monkeypatch,
    payload,
    expected,
):
    monkeypatch.chdir(tmp_path)
    _write_health_state(tmp_path, "drift_state.json", payload)

    ok, message = check_drift_state()

    assert ok is False
    assert "invalid artifact" in message
    assert expected in message


def test_health_states_reject_non_object_json_fail_closed(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    (output / "risk_state.json").write_text("[]", encoding="utf-8")
    (output / "drift_state.json").write_text("null", encoding="utf-8")

    risk_ok, risk_message = check_risk_state()
    drift_ok, drift_message = check_drift_state()

    assert risk_ok is False
    assert drift_ok is False
    assert "JSON artifact must be a JSON object" in risk_message
    assert "JSON artifact must be a JSON object" in drift_message


def test_dangling_state_symlink_is_not_treated_as_bootstrap(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    (output / "risk_state.json").symlink_to(tmp_path / "missing.json")

    ok, message = check_risk_state()

    assert ok is False
    assert "invalid artifact" in message


def test_health_state_schema_accepts_native_generator_outputs(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "output"
    risk_path = output / "risk_state.json"
    drift_path = output / "drift_state.json"

    evaluate_risk(
        daily_pnl_pct=0.0,
        current_mdd_pct=0.0,
        today=datetime(2026, 7, 26),
        state_path=risk_path,
    )
    evaluate_drift(
        pd.Series(dtype=float),
        pd.Series(dtype=float),
        asof=datetime(2026, 7, 26),
        state_path=drift_path,
    )

    risk_ok, risk_message = check_risk_state()
    drift_ok, drift_message = check_drift_state()

    assert risk_ok is True
    assert risk_message == "risk: ACTIVE"
    assert drift_ok is True
    assert drift_message == "drift: OK"


def _stub_signal_inputs(
    monkeypatch,
    *,
    quote_values,
    history_counts,
    ranges_by_db,
    exact_by_db_and_timestamp,
):
    rank_calls: list[tuple[str, str]] = []
    exact_calls: list[tuple[str, str]] = []

    def read_rank(db_path, rank_at):
        rank_calls.append((db_path, health_check._timestamp_text(rank_at)))
        return dict(quote_values), dict(history_counts)

    def read_exact(db_path, required_at):
        timestamp = health_check._timestamp_text(required_at)
        exact_calls.append((db_path, timestamp))
        return set(exact_by_db_and_timestamp[(db_path, timestamp)])

    monkeypatch.setattr(health_check, "_d1_rank_inputs_readonly", read_rank)
    monkeypatch.setattr(
        health_check,
        "_markets_at_timestamp_readonly",
        read_exact,
    )
    monkeypatch.setattr(
        health_check,
        "market_timestamp_ranges_readonly",
        lambda db_path: ranges_by_db[db_path],
    )
    return rank_calls, exact_calls


def test_pit_top_rank_preserves_min_rank_ties_at_boundary():
    expected, lower, insufficient, invalid = (
        health_check._rank_signal_universe(
            {"KRW-A", "KRW-B", "KRW-C", "KRW-D"},
            {"KRW-A": 300, "KRW-B": 200, "KRW-C": 200, "KRW-D": 100},
            {"KRW-A": 100, "KRW-B": 100, "KRW-C": 100, "KRW-D": 100},
            top_n=2,
        )
    )

    assert expected == {"KRW-A", "KRW-B", "KRW-C"}
    assert lower == {"KRW-D"}
    assert insufficient == set()
    assert invalid == set()


def test_pit_rank_excludes_69_prior_bars_before_top_n_displacement():
    expected, lower, insufficient, invalid = (
        health_check._rank_signal_universe(
            {"KRW-COLD", "KRW-READY", "KRW-LOWER"},
            {
                "KRW-COLD": 1_000,
                "KRW-READY": 900,
                "KRW-LOWER": 800,
            },
            {
                "KRW-COLD": 69,
                "KRW-READY": 70,
                "KRW-LOWER": 70,
            },
            top_n=1,
        )
    )

    assert expected == {"KRW-READY"}
    assert lower == {"KRW-LOWER"}
    assert insufficient == {"KRW-COLD"}
    assert invalid == set()


def test_open_freshness_uses_pit_top_set_and_reports_non_candidates(monkeypatch):
    now = datetime(2026, 7, 26, 9, 5)
    d1 = "data/upbit_d1.db"
    rank_calls, exact_calls = _stub_signal_inputs(
        monkeypatch,
        quote_values={"KRW-A": 300, "KRW-B": 200, "KRW-C": 100},
        history_counts={"KRW-A": 100, "KRW-B": 100, "KRW-C": 100, "KRW-NEW": 1},
        ranges_by_db={
            d1: {
                "KRW-A": (datetime(2025, 1, 1, 9), datetime(2026, 7, 26, 9)),
                "KRW-B": (datetime(2025, 1, 1, 9), datetime(2026, 7, 26, 9)),
                "KRW-C": (datetime(2025, 1, 1, 9), datetime(2026, 7, 26, 9)),
                "KRW-OLD": (datetime(2020, 1, 1, 9), datetime(2025, 1, 1, 9)),
            }
        },
        exact_by_db_and_timestamp={
            (d1, "2026-07-26 09:00:00"): {"KRW-A", "KRW-B", "KRW-C"},
        },
    )

    ok, message = check_universe_coverage(
        [(d1, 30)],
        live_markets=[
            "KRW-A",
            "KRW-B",
            "KRW-C",
            "KRW-NEW",
            "KRW-USDT",
            "KRW-IP",
        ],
        now=now,
        channel="recommend",
        top_n=2,
    )

    assert ok is True
    assert rank_calls == [(d1, "2026-07-25 09:00:00")]
    assert exact_calls == [(d1, "2026-07-26 09:00:00")]
    assert "pit_top=2/2" in message
    assert "lower_ranked_excluded=1[KRW-C]" in message
    assert "insufficient_history_excluded=1[KRW-NEW]" in message
    assert "signal_market_excluded=2[KRW-IP,KRW-USDT]" in message
    assert "raw_confirmed_no_current_excluded=1[KRW-NEW]" in message
    assert "expected_current_start=2026-07-26 09:00:00" in message


def test_open_freshness_fails_when_pit_candidate_exact_row_is_missing(
    monkeypatch,
):
    # 업스트림엔 봉이 있는데 DB에 없다 = 진짜 수집 갭 → 종전대로 fail.
    monkeypatch.setattr(
        health_check,
        "_upstream_candle_exists",
        lambda market, required_at, db_name: True,
    )
    now = datetime(2026, 7, 26, 9, 5)
    d1 = "data/upbit_d1.db"
    _stub_signal_inputs(
        monkeypatch,
        quote_values={"KRW-A": 300, "KRW-B": 200},
        history_counts={"KRW-A": 100, "KRW-B": 100},
        ranges_by_db={
            d1: {
                "KRW-A": (datetime(2025, 1, 1, 9), datetime(2026, 7, 26, 9)),
                "KRW-B": (datetime(2025, 1, 1, 9), datetime(2026, 7, 25, 9)),
            }
        },
        exact_by_db_and_timestamp={
            (d1, "2026-07-26 09:00:00"): {"KRW-A"},
        },
    )

    ok, message = check_universe_coverage(
        [(d1, 30)],
        live_markets=["KRW-A", "KRW-B"],
        now=now,
        channel="recommend",
        top_n=2,
        min_coverage_ratio=0.5,
    )

    assert ok is False
    assert "exact 1/2" in message
    assert "missing=1[KRW-B]" in message


def test_open_freshness_rejects_future_d1_candidate_timestamp(monkeypatch):
    now = datetime(2026, 7, 26, 9, 5)
    d1 = "data/upbit_d1.db"
    _stub_signal_inputs(
        monkeypatch,
        quote_values={"KRW-BTC": 300},
        history_counts={"KRW-BTC": 100},
        ranges_by_db={
            d1: {
                "KRW-BTC": (
                    datetime(2025, 1, 1, 9),
                    datetime(2026, 7, 26, 9, 0, 1),
                ),
            }
        },
        exact_by_db_and_timestamp={
            (d1, "2026-07-26 09:00:00"): {"KRW-BTC"},
        },
    )

    ok, message = check_universe_coverage(
        [(d1, 30)],
        live_markets=["KRW-BTC"],
        now=now,
        channel="recommend",
    )

    assert ok is False
    assert "future=1[KRW-BTC]" in message


def test_aware_utc_now_is_converted_to_kst_scope(monkeypatch):
    now_utc = datetime(2026, 7, 26, 0, 5, tzinfo=timezone.utc)
    d1 = "data/upbit_d1.db"
    rank_calls, exact_calls = _stub_signal_inputs(
        monkeypatch,
        quote_values={"KRW-BTC": 300},
        history_counts={"KRW-BTC": 100},
        ranges_by_db={
            d1: {
                "KRW-BTC": (
                    datetime(2025, 1, 1, 9),
                    datetime(2026, 7, 26, 9),
                )
            }
        },
        exact_by_db_and_timestamp={
            (d1, "2026-07-26 09:00:00"): {"KRW-BTC"},
        },
    )

    ok, _ = check_universe_coverage(
        [(d1, 30)],
        live_markets=["KRW-BTC"],
        now=now_utc,
        channel="recommend",
    )

    assert ok is True
    assert rank_calls == [(d1, "2026-07-25 09:00:00")]
    assert exact_calls == [(d1, "2026-07-26 09:00:00")]


def test_preopen_uses_d2_rank_yesterday_d1_and_last_closed_15m(
    monkeypatch,
):
    now = datetime(2026, 7, 26, 8, 50)
    d1 = "data/upbit_d1.db"
    m15 = "data/upbit_15m.db"
    rank_calls, exact_calls = _stub_signal_inputs(
        monkeypatch,
        quote_values={"KRW-BTC": 300},
        history_counts={"KRW-BTC": 100},
        ranges_by_db={
            d1: {
                "KRW-BTC": (
                    datetime(2025, 1, 1, 9),
                    datetime(2026, 7, 25, 9),
                )
            },
            m15: {
                "KRW-BTC": (
                    datetime(2026, 7, 23, 9),
                    datetime(2026, 7, 26, 8, 45),
                )
            },
        },
        exact_by_db_and_timestamp={
            (d1, "2026-07-25 09:00:00"): {"KRW-BTC"},
            (m15, "2026-07-26 08:30:00"): {"KRW-BTC"},
        },
    )

    ok, message = check_universe_coverage(
        [(d1, 48), (m15, 2)],
        live_markets=["KRW-BTC"],
        now=now,
        channel="preopen",
    )

    assert ok is True
    assert rank_calls == [(d1, "2026-07-24 09:00:00")]
    assert exact_calls == [
        (d1, "2026-07-25 09:00:00"),
        (m15, "2026-07-26 08:30:00"),
    ]
    assert "expected_last_closed_start=2026-07-25 09:00:00" in message
    assert "expected_last_closed_start=2026-07-26 08:30:00" in message


def test_recommend_preopen_uses_d2_rank_and_yesterday_d1_only(
    monkeypatch,
):
    now = datetime(2026, 7, 26, 8, 50)
    d1 = "data/upbit_d1.db"
    rank_calls, exact_calls = _stub_signal_inputs(
        monkeypatch,
        quote_values={"KRW-BTC": 300},
        history_counts={"KRW-BTC": 100},
        ranges_by_db={
            d1: {
                "KRW-BTC": (
                    datetime(2025, 1, 1, 9),
                    datetime(2026, 7, 25, 9),
                )
            },
        },
        exact_by_db_and_timestamp={
            (d1, "2026-07-25 09:00:00"): {"KRW-BTC"},
        },
    )

    ok, message = check_universe_coverage(
        [(d1, 48)],
        live_markets=["KRW-BTC"],
        now=now,
        channel="recommend-preopen",
    )

    assert ok is True
    assert rank_calls == [(d1, "2026-07-24 09:00:00")]
    assert exact_calls == [(d1, "2026-07-25 09:00:00")]
    assert "scope=recommend-preopen" in message
    assert "expected_last_closed_start=2026-07-25 09:00:00" in message


def test_distribution_requires_last_closed_4h_start(monkeypatch):
    now = datetime(2026, 7, 26, 9, 5)
    d1 = "data/upbit_d1.db"
    h4 = "data/upbit_4h.db"
    _, exact_calls = _stub_signal_inputs(
        monkeypatch,
        quote_values={"KRW-BTC": 300},
        history_counts={"KRW-BTC": 100},
        ranges_by_db={
            d1: {
                "KRW-BTC": (
                    datetime(2025, 1, 1, 9),
                    datetime(2026, 7, 26, 9),
                )
            },
            h4: {
                "KRW-BTC": (
                    datetime(2026, 7, 20, 1),
                    datetime(2026, 7, 26, 9),
                )
            },
        },
        exact_by_db_and_timestamp={
            (d1, "2026-07-26 09:00:00"): {"KRW-BTC"},
            (h4, "2026-07-26 05:00:00"): {"KRW-BTC"},
        },
    )

    ok, message = check_universe_coverage(
        [(d1, 30), (h4, 8)],
        live_markets=["KRW-BTC"],
        now=now,
        channel="distribution",
    )

    assert ok is True
    assert exact_calls[-1] == (h4, "2026-07-26 05:00:00")
    assert "expected_last_closed_start=2026-07-26 05:00:00" in message


def test_universe_coverage_fails_closed_when_live_api_fails(monkeypatch):
    def fail_live_markets(*, include_stablecoins_for_audit=False):
        assert include_stablecoins_for_audit is True
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(health_check, "get_krw_markets", fail_live_markets)

    ok, message = check_universe_coverage([("data/upbit_d1.db", 30)])

    assert ok is False
    assert "ticker fetch failed" in message


def test_signal_pit_freshness_missing_db_is_read_only(tmp_path):
    missing = tmp_path / "missing" / "upbit_d1.db"

    ok, message = check_universe_coverage(
        [(str(missing), 30)],
        live_markets=["KRW-BTC"],
        now=datetime(2026, 7, 26, 9, 5),
        channel="recommend",
    )

    assert ok is False
    assert "D1 PIT rank read failed" in message
    assert not missing.exists()
    assert not missing.parent.exists()


def test_single_btc_freshness_rejects_future_timestamp(monkeypatch):
    now = datetime(2026, 7, 26, 9, 5)
    monkeypatch.setattr(
        health_check,
        "market_timestamp_ranges_readonly",
        lambda _db_path: {
            "KRW-BTC": (
                datetime(2024, 1, 1),
                now + timedelta(hours=1),
            )
        },
    )

    ok, message = check_db_freshness(
        "data/upbit_d1.db",
        now=now,
    )

    assert ok is False
    assert "future timestamp" in message


def test_single_freshness_missing_db_is_read_only(tmp_path):
    missing = tmp_path / "missing" / "upbit_d1.db"

    ok, message = check_db_freshness(str(missing))

    assert ok is False
    assert "read-only freshness check failed" in message
    assert not missing.exists()
    assert not missing.parent.exists()


def test_missing_candidate_with_upstream_absent_is_no_trade_tolerated(
    monkeypatch,
):
    # 업스트림에도 봉이 없다 = 그 경계에 거래가 없던 얇은 코인 — 구조적 관용.
    # 코인 1개의 무거래가 발송 전체를 죽이던 2026-07-29 가용성 회귀 방지.
    probed = []

    def absent(market, required_at, db_name):
        probed.append((market, db_name))
        return False

    monkeypatch.setattr(health_check, "_upstream_candle_exists", absent)
    now = datetime(2026, 7, 26, 9, 5)
    d1 = "data/upbit_d1.db"
    _stub_signal_inputs(
        monkeypatch,
        quote_values={"KRW-A": 300, "KRW-B": 200},
        history_counts={"KRW-A": 100, "KRW-B": 100},
        ranges_by_db={
            d1: {
                "KRW-A": (datetime(2025, 1, 1, 9), datetime(2026, 7, 26, 9)),
                "KRW-B": (datetime(2025, 1, 1, 9), datetime(2026, 7, 25, 9)),
            }
        },
        exact_by_db_and_timestamp={
            (d1, "2026-07-26 09:00:00"): {"KRW-A"},
        },
    )

    ok, message = check_universe_coverage(
        [(d1, 30)],
        live_markets=["KRW-A", "KRW-B"],
        now=now,
        channel="recommend",
        top_n=2,
    )

    assert ok is True
    assert probed == [("KRW-B", "upbit_d1.db")]
    assert "no_trade_confirmed=1[KRW-B]" in message
    assert "missing=0[none]" in message
    assert "exact 1/1" in message


def test_missing_candidate_with_unconfirmable_upstream_fails_closed(
    monkeypatch,
):
    def unconfirmable(market, required_at, db_name):
        raise RuntimeError("probe failed")

    monkeypatch.setattr(
        health_check,
        "_upstream_candle_exists",
        unconfirmable,
    )
    now = datetime(2026, 7, 26, 9, 5)
    d1 = "data/upbit_d1.db"
    _stub_signal_inputs(
        monkeypatch,
        quote_values={"KRW-A": 300, "KRW-B": 200},
        history_counts={"KRW-A": 100, "KRW-B": 100},
        ranges_by_db={
            d1: {
                "KRW-A": (datetime(2025, 1, 1, 9), datetime(2026, 7, 26, 9)),
                "KRW-B": (datetime(2025, 1, 1, 9), datetime(2026, 7, 25, 9)),
            }
        },
        exact_by_db_and_timestamp={
            (d1, "2026-07-26 09:00:00"): {"KRW-A"},
        },
    )

    ok, message = check_universe_coverage(
        [(d1, 30)],
        live_markets=["KRW-A", "KRW-B"],
        now=now,
        channel="recommend",
        top_n=2,
    )

    assert ok is False
    assert "missing=1[KRW-B]" in message


def test_mass_missing_skips_probes_and_fails(monkeypatch):
    # 대량 결측(>5)은 계통 장애 — 업스트림 확인 없이 전량 fail-closed.
    def must_not_probe(market, required_at, db_name):
        raise AssertionError("probe must not run for mass missing")

    monkeypatch.setattr(
        health_check,
        "_upstream_candle_exists",
        must_not_probe,
    )
    now = datetime(2026, 7, 26, 9, 5)
    d1 = "data/upbit_d1.db"
    markets = [f"KRW-M{i}" for i in range(8)]
    _stub_signal_inputs(
        monkeypatch,
        quote_values={m: 300 - i for i, m in enumerate(markets)},
        history_counts={m: 100 for m in markets},
        ranges_by_db={
            d1: {
                m: (datetime(2025, 1, 1, 9), datetime(2026, 7, 25, 9))
                for m in markets
            }
        },
        exact_by_db_and_timestamp={
            (d1, "2026-07-26 09:00:00"): set(),
        },
    )

    ok, message = check_universe_coverage(
        [(d1, 30)],
        live_markets=markets,
        now=now,
        channel="recommend",
        top_n=8,
    )

    assert ok is False
    assert "missing=8[" in message
