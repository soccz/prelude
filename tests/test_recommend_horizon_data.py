"""Hermetic D1 reconstruction tests: no real DB, fits, writers or notifications."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from copy import deepcopy
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

from ops.artifact_provenance import file_set_identity
from signals import recommend_horizon_data as horizon


@pytest.fixture
def candles():
    rows = []
    for market_index, market in enumerate(("KRW-A", "KRW-B", "KRW-C", "KRW-USDT")):
        for index, stamp in enumerate(pd.date_range("2026-01-01 09:00", periods=100)):
            price = 100.0 + index + market_index
            rows.append((market, stamp, price, price * 1.12, price * .92,
                         price * (1 + .01 * np.sin(index)),
                         10.0 + index, 10000.0 + index * 50 + market_index))
    return pd.DataFrame(rows, columns=horizon.RAW_COLUMNS)


ASOF = "2026-04-12T09:01:00+09:00"


def frozen_for(raw, references=("2026-04-01", "2026-04-06")):
    panel = horizon._raw_panel(raw, build_asof=ASOF)
    records = []
    for reference in references:
        day = (pd.Timestamp(reference) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        group = panel.loc[panel.date.astype(str).eq(reference) & panel.f_qv_rank.le(100)]
        for rank, row in enumerate(group.to_dict("records"), 1):
            records.append({
                **{name: round(float(row[name]), 8) if pd.notna(row[name]) else None
                   for name in horizon.ALLOWED_FEATURES},
                "date": day, "slot": "preopen", "coin": row["market"], "rank": rank,
                "snapshot_id": f"sid-{day}", "decision_started_at": f"{day}T08:50:00+09:00",
                "was_delivered": rank <= 3, "up10": True, "label_status": "labeled",
            })
    return pd.DataFrame(records)


def test_shapes_contract_and_input_immutability(candles):
    frozen = frozen_for(candles)
    raw_before, frozen_before = candles.copy(deep=True), frozen.copy(deep=True)
    result = horizon.build_horizon_frames(candles, frozen, build_asof=ASOF)
    pd.testing.assert_frame_equal(candles, raw_before)
    pd.testing.assert_frame_equal(frozen, frozen_before)
    training = result["training_frame"]
    assert list(training.columns) == list(horizon.TRAIN_COLUMNS)
    assert len(training) == 60 and set(training.coin) == {"KRW-A", "KRW-B", "KRW-C"}
    assert training.feature_date.max() == "2026-03-31"
    assert training.feature_date.min() == "2026-03-12"
    assert all(training[name].str.endswith("+00:00").all()
               for name in ("input_end_at", "B_target_end_at", "C_target_end_at"))
    assert result["data_audit"]["historical_ingestion_proven"] is False
    assert result["data_audit"]["historical_fullprecision_training_proven"] is False
    assert "up10" not in training and "was_delivered" not in training
    assert result["prediction_frame"].up10.all()
    assert result["decision_metadata"]["sid-2026-04-07"] == {
        "date": "2026-04-07", "feature_row_date": "2026-04-06",
        "cutoff_exclusive": "2026-04-01", "decision_started_at": "2026-04-06T23:50:00+00:00"}


def test_input_shuffle_is_equivalent(candles):
    frozen = frozen_for(candles)
    expected = horizon.build_horizon_frames(candles, frozen, build_asof=ASOF)
    actual = horizon.build_horizon_frames(candles.sample(frac=1, random_state=7),
                                          frozen.sample(frac=1, random_state=8), build_asof=ASOF)
    for name in ("training_frame", "prediction_frame"):
        pd.testing.assert_frame_equal(expected[name], actual[name])
    assert expected["decision_metadata"] == actual["decision_metadata"]
    assert expected["data_audit"] == actual["data_audit"]


def test_calendar_gap_does_not_shift_to_the_next_observed_day(candles):
    gap = (candles.market == "KRW-A") & (candles.timestamp == pd.Timestamp("2026-03-18 09:00"))
    candles = candles.loc[~gap].copy()
    result = horizon.build_horizon_frames(candles, frozen_for(candles), build_asof=ASOF)
    training = result["training_frame"].set_index(["feature_date", "coin"])
    missing = training.loc[("2026-03-17", "KRW-A")]
    assert not missing.C_target_present and pd.isna(missing.C_high_ret) and missing.C_target_end_at is None
    # shift(1) consumes the preceding observed bar; its end is not invented as t.
    after_gap = training.loc[("2026-03-19", "KRW-A")]
    assert after_gap.input_end_at == "2026-03-18T00:00:00+00:00"
    assert result["data_audit"]["common_training_by_decision"][0]["missing_exact_next_calendar_targets"] == [
        {"feature_date": "2026-03-17", "coin": "KRW-A"}]


def test_terminal_target_missing_is_preserved_and_common_excluded(candles):
    candles = candles.loc[~(candles.market.eq("KRW-B") & candles.timestamp.gt("2026-03-20 09:00"))]
    result = horizon.build_horizon_frames(candles, frozen_for(candles), build_asof=ASOF)
    training = result["training_frame"]
    terminal = training.loc[training.coin.eq("KRW-B") & training.feature_date.eq("2026-03-20")]
    assert len(terminal) == 1 and not terminal.C_target_present.iloc[0]
    for decision in result["decision_metadata"].values():
        assert not horizon.common_training_mask(training, decision).loc[terminal.index].any()


def test_original_fraction_and_epsilon_definitions(candles):
    result = horizon.build_horizon_frames(candles, frozen_for(candles), build_asof=ASOF)
    row = result["training_frame"].iloc[0]
    raw = candles.loc[candles.market.eq(row.coin)].set_index("timestamp")
    current = raw.loc[pd.Timestamp(row.feature_date) + pd.Timedelta(hours=9)]
    following = raw.loc[pd.Timestamp(row.feature_date) + pd.Timedelta(days=1, hours=9)]
    assert row.B_high_ret == current.high / (current.open + 1e-12) - 1
    assert row.B_low_ret == current.low / (current.open + 1e-12) - 1
    assert row.C_high_ret == following.high / (following.open + 1e-12) - 1
    assert row.C_low_ret == following.low / (following.open + 1e-12) - 1
    assert row.B_target_end_at == "2026-03-13T00:00:00+00:00"
    assert row.C_target_end_at == "2026-03-14T00:00:00+00:00"


def test_exactly_70_rows_excluded_and_71_rows_yields_one(candles):
    a = candles.loc[candles.market.eq("KRW-A")].iloc[:71]
    assert len(horizon._raw_panel(a, build_asof=ASOF)) == 1
    with pytest.raises(horizon.HorizonDataError, match="70"):
        horizon._raw_panel(a.iloc[:70], build_asof=ASOF)


@pytest.mark.parametrize("field,value", [("high", np.nan), ("open", 0), ("low", -1),
                                          ("close", np.inf), ("volume", -1), ("open", True)])
def test_invalid_ohlc_fails_closed(candles, field, value):
    candles[field] = candles[field].astype(object)
    candles.loc[0, field] = value
    with pytest.raises((horizon.HorizonDataError, ValueError, TypeError)):
        horizon._raw_panel(candles, build_asof=ASOF)


def test_raw_duplicate_inconsistent_boundary_and_future_fail_closed(candles):
    for mutation, match in (("duplicate", "duplicate"), ("boundary", "09:00"),
                             ("inconsistent", "OHLC"), ("future", "incomplete/future")):
        changed = candles.copy(deep=True)
        if mutation == "duplicate":
            changed = pd.concat([changed, changed.iloc[:1]])
        elif mutation == "boundary":
            changed.loc[0, "timestamp"] += pd.Timedelta(minutes=1)
        elif mutation == "inconsistent":
            changed.loc[0, "high"] = 1
        elif mutation == "future":
            changed.loc[0, "timestamp"] += pd.Timedelta(days=1000)
        with pytest.raises(horizon.HorizonDataError, match=match):
            horizon._raw_panel(changed, build_asof=ASOF)


def test_future_prices_cannot_change_past_features(candles):
    changed = candles.copy(deep=True)
    mask = changed.timestamp.ge("2026-04-07 09:00")
    changed.loc[mask, ["open", "high", "low", "close"]] *= 10
    a = horizon._raw_panel(candles, build_asof=ASOF)
    b = horizon._raw_panel(changed, build_asof=ASOF)
    past = a.date.astype(str).le("2026-04-07")
    pd.testing.assert_frame_equal(a.loc[past, list(horizon.ALLOWED_FEATURES)],
                                  b.loc[past, list(horizon.ALLOWED_FEATURES)])


def test_null_quote_volume_uses_original_fallback(candles):
    candles.loc[80, "quote_volume"] = np.nan
    explicit = candles.copy()
    explicit.loc[80, "quote_volume"] = explicit.loc[80, "volume"] * explicit.loc[80, "close"]
    pd.testing.assert_frame_equal(horizon._raw_panel(candles, build_asof=ASOF),
                                  horizon._raw_panel(explicit, build_asof=ASOF))


def test_nan_and_null_feature_match_without_imputation(candles, monkeypatch):
    panel = horizon._raw_panel(candles, build_asof=ASOF)
    panel["f_rsi_14"] = np.nan
    frozen = frozen_for(candles)
    frozen["f_rsi_14"] = None
    monkeypatch.setattr(horizon, "_raw_panel", lambda *_args, **_kwargs: panel.copy(deep=True))
    result = horizon.build_horizon_frames(candles, frozen, build_asof=ASOF)
    assert result["training_frame"].f_rsi_14.isna().all()
    assert result["prediction_frame"].f_rsi_14.isna().all()


@pytest.mark.parametrize("mutation", ["feature", "boolean", "nan", "rank", "duplicate", "universe",
                                       "slot", "late", "naive", "metadata", "sid"])
def test_snapshot_parity_and_chronology_fail_closed(candles, mutation):
    frozen = frozen_for(candles)
    if mutation == "feature":
        frozen.loc[0, "f_ret_1d"] += .01
    elif mutation == "boolean":
        frozen["f_ret_1d"] = frozen.f_ret_1d.astype(object)
        frozen.loc[0, "f_ret_1d"] = True
    elif mutation == "nan":
        frozen.loc[0, "f_ret_1d"] = np.nan
    elif mutation == "rank":
        frozen.loc[0, "rank"] = 99
    elif mutation == "duplicate":
        frozen = pd.concat([frozen, frozen.iloc[:1]])
    elif mutation == "universe":
        frozen.loc[0, "coin"] = "KRW-WRONG"
    elif mutation == "slot":
        frozen.loc[0, "slot"] = "open"
    elif mutation in {"late", "naive", "metadata"}:
        first = frozen.date.eq(frozen.date.iloc[0])
        value = frozen.decision_started_at.iloc[0]
        if mutation == "late":
            frozen.loc[first, "decision_started_at"] = value.replace("08:50", "09:00")
        elif mutation == "naive":
            frozen.loc[first, "decision_started_at"] = value.replace("+09:00", "")
        else:
            frozen.loc[0, "decision_started_at"] = value.replace("08:50", "08:51")
    else:
        frozen["snapshot_id"] = "duplicate-id"
    with pytest.raises(horizon.HorizonDataError):
        horizon.build_horizon_frames(candles, frozen, build_asof=ASOF)


def test_common_training_maturity_is_strict(candles):
    result = horizon.build_horizon_frames(candles, frozen_for(candles), build_asof=ASOF)
    training = result["training_frame"].iloc[:1].copy()
    decision = {"cutoff_exclusive": "2026-04-01", "decision_started_at": "2026-03-14T00:00:00+00:00"}
    assert not horizon.common_training_mask(training, decision).any()  # C end equals decision.
    decision["decision_started_at"] = "2026-03-14T00:00:01+00:00"
    assert horizon.common_training_mask(training, decision).all()
    training.loc[training.index[0], "input_end_at"] = decision["decision_started_at"]
    assert not horizon.common_training_mask(training, decision).any()


def test_logical_hash_is_order_and_value_sensitive():
    rows = [("KRW-A", "2026-01-01 09:00:00", 1.0, 2.0, .5, 1.0, 10.0, None),
            ("KRW-B", "2026-01-01 09:00:00", 1.0, 2.0, .5, 1.0, 10.0, 10.0)]
    expected = horizon.logical_rows_identity(rows)
    assert expected["rows"] == 2 and len(expected["sha256"]) == 64
    assert horizon.logical_rows_identity(iter(rows)) == expected
    assert horizon.logical_rows_identity(rows[::-1]) != expected


@pytest.fixture
def pinned(tmp_path, monkeypatch):
    monkeypatch.setattr(horizon, "ROOT", tmp_path)
    source = tmp_path / "source.py"
    source.write_text("# synthetic source\n")
    readiness = tmp_path / "readiness.json"
    readiness.write_text("{}")
    sources = file_set_identity({"source.py": source}, root=tmp_path)
    evidence = file_set_identity({"readiness_report": readiness}, root=tmp_path)
    db = tmp_path / "data/upbit_d1.db"
    db.parent.mkdir()
    rows = [("KRW-A", "2026-01-01 09:00:00", 100., 110., 90., 101., 10., 1000.)]
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE candles (market TEXT,timestamp TEXT,open REAL,high REAL,low REAL,close REAL,volume REAL,quote_volume REAL)")
        connection.executemany("INSERT INTO candles VALUES (?,?,?,?,?,?,?,?)", rows)
    logical = horizon.logical_rows_identity(rows)
    dates = pd.date_range("2026-07-28", periods=35).strftime("%Y-%m-%d").tolist()
    parity = [{"date": day, "snapshot_id": f"sid-{day}", "candidate_rows": 100} for day in dates]
    checks = [{"date": day, "base_top100_feature_rows": index + 1, "common_training_rows": index}
              for index, day in enumerate(dates)]
    audit = {"sql": {"raw_ohlc_query": horizon.RAW_SQL}, "database": {"logical_rows": logical},
             "sources": {"files": sources}, "frozen_evidence": {"identity_root": str(tmp_path), "files": evidence},
             "parity": {"by_date": parity}, "first_last_common_training_checks": [
                 {"asof": check["date"], **{k: v for k, v in check.items() if k != "date"}}
                 for check in (checks[0], checks[-1])], "d1_vs_existing_preopen_canonical_labels": {"test": True}}
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(audit))
    monkeypatch.setitem(horizon.DATA_CONFIG, "reconstruction_audit_sha256", hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setitem(horizon.DATA_CONFIG, "raw_sql_sha256", logical["sha256"])
    frame = pd.DataFrame({"slot": ["preopen"] * 3500})
    monkeypatch.setattr(horizon, "load_experiment_data", lambda path: {
        "frame": frame, "provenance": {"files": evidence}})
    result = {"training_frame": pd.DataFrame(), "prediction_frame": frame,
              "decision_metadata": {}, "data_audit": {"parity": parity, "common_training_by_decision": checks}}
    monkeypatch.setattr(horizon, "build_horizon_frames", lambda *a, **kw: deepcopy(result))
    return {"root": tmp_path, "source": source, "readiness": readiness, "audit": path,
            "database": db, "result": result}


def test_prepare_rebinds_and_read_only_sql_preserves_bytes(pinned):
    before = {path: path.read_bytes() for path in pinned["root"].rglob("*") if path.is_file()}
    result = horizon.prepare_horizon_data(pinned["readiness"], pinned["audit"])
    assert result["provenance"]["source_evidence_before_after_identical"] is True
    assert result["provenance"]["database_mode"] == "ro/query_only/BEGIN/temp_store=MEMORY"
    assert before == {path: path.read_bytes() for path in pinned["root"].rglob("*") if path.is_file()}


@pytest.mark.parametrize("target", ["source", "readiness", "audit", "database"])
def test_prepare_rejects_preexisting_tampering(pinned, target):
    if target == "database":
        with sqlite3.connect(pinned[target]) as connection:
            connection.execute("UPDATE candles SET high=120")
    else:
        pinned[target].write_text("modified")
    with pytest.raises(horizon.HorizonDataError):
        horizon.prepare_horizon_data(pinned["readiness"], pinned["audit"])


@pytest.mark.parametrize("target", ["source", "readiness", "audit"])
def test_prepare_rejects_input_change_during_compute(pinned, monkeypatch, target):
    def changed(*args, **kwargs):
        pinned[target].write_text("changed during compute")
        return deepcopy(pinned["result"])
    monkeypatch.setattr(horizon, "build_horizon_frames", changed)
    with pytest.raises(horizon.HorizonDataError):
        horizon.prepare_horizon_data(pinned["readiness"], pinned["audit"])


def test_sql_reader_is_really_query_only(pinned, monkeypatch):
    original = horizon.connect_readonly
    @contextmanager
    def guarded(path):
        with original(path) as connection:
            with pytest.raises(sqlite3.OperationalError):
                connection.execute("DELETE FROM candles")
            connection.rollback()  # Failed DML still opens sqlite's implicit transaction.
            yield connection
    monkeypatch.setattr(horizon, "connect_readonly", guarded)
    horizon.prepare_horizon_data(pinned["readiness"], pinned["audit"])
