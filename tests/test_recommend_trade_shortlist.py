"""Synthetic-only tests of the separate, non-publishing shortlist prototype."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sqlite3

import pytest
import requests

import test_recommend_microstructure as fixtures
import signals.recommend_microstructure as native
import signals.recommend_microstructure_trial as old_trial
import signals.recommend_snapshot as snapshots
import signals.recommend_trade_shortlist as shortlist


@pytest.fixture(autouse=True)
def no_live_operations(monkeypatch):
    monkeypatch.setenv("PRELUDE_FORBID_TELEGRAM", "1")

    def forbidden(*args, **kwargs):
        pytest.fail("offline planner attempted I/O, training, publication or live work")

    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    for name in ("_data_metadata", "_git_code_metadata", "_environment_metadata"):
        monkeypatch.setattr(snapshots, name, forbidden)
    for name in ("_load_inputs", "record_microstructure_trial", "evaluate_microstructure_trials"):
        monkeypatch.setattr(old_trial, name, forbidden)


def _bind(snapshot, feature=None):
    snapshot["top3"] = copy.deepcopy(sorted(snapshot["universe"], key=lambda r: r["rank"])[:3])
    digest = snapshots._document_digest(snapshot)
    snapshot.update(payload_sha256=digest, snapshot_id=f"recommend-{digest[:20]}")
    if feature is not None:
        feature["snapshot"].update(payload_sha256=digest, snapshot_id=snapshot["snapshot_id"])


def _inputs(values=None, *, missing=()):
    values = values or {rank: rank / 100 for rank in range(1, 101)}
    trades = []
    for rank, coin in enumerate(fixtures.MARKETS, 1):
        if rank in missing:
            continue
        value = values.get(rank, 0.0)
        for side, volume in (("BID", 1 + value), ("ASK", 1 - value)):
            if volume:
                trades.append(fixtures._trade(coin=coin, side=side, volume=volume, trade_id=len(trades)+1))
    snapshot, manifest, records = fixtures._sample(trades)
    for row in snapshot["universe"]:
        row["p_up10"] = (101 - row["rank"]) / 1000
        row["rr_ratio"] = row["p_up10"] / row["p_dn5"]
        row["feature_values"].update(f_atr_pct_14=row["rank"]/1000, f_log_qv=20-row["rank"]/100)
    _bind(snapshot)
    manifest["cutoff_source"].update(snapshot_id=snapshot["snapshot_id"], payload_sha256=snapshot["payload_sha256"])
    feature = native.compute_trade_imbalance(
        snapshot, manifest, trade_records=records["trade"], orderbook_records=records["orderbook"]
    )
    return snapshot, feature


def test_non_tied_ranks_eight_through_ten_replace_all_three_without_rank_eleven():
    snapshot, feature = _inputs()
    before = copy.deepcopy((snapshot, feature))
    result = shortlist.plan_trade_shortlist(snapshot, feature)
    coins = fixtures.MARKETS
    assert snapshot["universe"][2]["rr_ratio"] != snapshot["universe"][3]["rr_ratio"]
    assert result["challenger_top3"] == [coins[9], coins[8], coins[7]]
    assert result["changed_picks"] == 3 and result["is_no_op"] is False
    assert coins[10] not in result["challenger_top3"]  # higher flow, but outside the frozen pool
    assert result["control_ranking"] == coins and result["control_top3"] == coins[:3]
    assert result["challenger_ranking"][10:] == coins[10:]
    assert sorted(result["challenger_ranking"]) == sorted(coins)
    assert (snapshot, feature) == before
    assert result["effect_status"] == "not_evaluated" and result["prospective_prediction"] is False
    assert result["deployable"] is False
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("value", [-1.0, 0.0, 1.0])
def test_all_equal_including_real_zero_is_noop(value):
    result = shortlist.plan_trade_shortlist(*_inputs({rank: value for rank in range(1, 101)}))
    assert result["status"] == "planned" and result["changed_picks"] == 0
    assert result["is_no_op"] is True and result["challenger_ranking"] == result["control_ranking"]
    assert result["coverage"]["shortlist_available_features"] == 10


def test_negative_values_are_not_vetoed():
    result = shortlist.plan_trade_shortlist(*_inputs({rank: -1 + rank/200 for rank in range(1, 101)}))
    assert result["status"] == "planned" and result["challenger_top3"] == fixtures.MARKETS[7:10][::-1]


@pytest.mark.parametrize("rank", [1, 3, 4, 10])
def test_missing_any_required_name_does_not_shrink_refill_or_impute(rank):
    result = shortlist.plan_trade_shortlist(*_inputs(missing=(rank,)))
    assert result["status"] == "unavailable" and result["challenger_ranking"] is None
    assert result["changed_picks"] is None and result["is_no_op"] is None
    assert result["required_missing_features"] == [fixtures.MARKETS[rank-1]]
    assert len(result["shortlist"]) == 10 and result["control_top3"] == fixtures.MARKETS[:3]


def test_outside_ninety_no_trade_values_do_not_block_valid_shortlist():
    result = shortlist.plan_trade_shortlist(*_inputs(missing=range(11, 101)))
    assert result["status"] == "planned" and result["coverage"]["feature_available_rows"] == 10
    assert result["coverage"]["feature_coverage"] == .1
    assert result["required_missing_features"] == []


def test_invalid_capture_is_unavailable_even_when_old_selection_would_be_noop():
    snapshot, feature = _inputs({rank: 0. for rank in range(1, 101)})
    feature.update(feature_evidence_valid=False, quality_reasons=["capture_not_complete"],
                   feature_available_rows=0, feature_coverage=0.)
    for row in feature["rows"]:
        row.update({native.FEATURE: None, "feature_status": "unavailable"})
    feature["coverage"].update(feature_available_rows=0, available_coins=[])
    result = shortlist.plan_trade_shortlist(snapshot, feature)
    assert result["reason"] == "feature_evidence_invalid" and result["is_no_op"] is None


def test_single_tiny_one_sided_trade_is_explicitly_diagnosed_not_silently_filtered():
    snapshot, feature = _inputs()
    row = feature["rows"][5]
    row.update({native.FEATURE: 1., "bid_notional": .0001, "ask_notional": 0., "observed_trade_count": 1})
    result = shortlist.plan_trade_shortlist(snapshot, feature)
    coin = fixtures.MARKETS[5]
    assert result["challenger_top3"][0] == coin
    assert result["diagnostics"]["selected_single_trade_coins"] == [coin]
    assert result["diagnostics"]["selected_saturated_coins"] == [coin]
    assert result["candidate_inputs"][5]["total_notional"] == .0001


def test_row_order_and_outcomes_do_not_change_selection():
    snapshot, feature = _inputs()
    expected = shortlist.plan_trade_shortlist(snapshot, feature)
    snapshot["universe"].reverse()
    feature["rows"].reverse()
    _bind(snapshot, feature)
    shuffled = shortlist.plan_trade_shortlist(snapshot, feature)
    assert {k:v for k,v in shuffled.items() if k != "snapshot"} == {
        k:v for k,v in expected.items() if k != "snapshot"}
    for row in snapshot["universe"]:
        row.update(up10=row["rank"] < 4, dn5=row["rank"] > 3, eod_return_net=100-row["rank"],
                   label_status="halted_no_observations")
    _bind(snapshot, feature)
    with_outcomes = shortlist.plan_trade_shortlist(snapshot, feature)
    assert {k:v for k,v in with_outcomes.items() if k != "snapshot"} == {
        k:v for k,v in expected.items() if k != "snapshot"}


def test_optional_atr_liquidity_context_cannot_change_selection():
    snapshot, feature = _inputs()
    expected = shortlist.plan_trade_shortlist(snapshot, feature)["challenger_top3"]
    for row in snapshot["universe"]:
        row["feature_values"] = {}
    _bind(snapshot, feature)
    result = shortlist.plan_trade_shortlist(snapshot, feature)
    assert result["challenger_top3"] == expected
    assert all(row["stored_f_atr_pct_14"] is None for row in result["candidate_inputs"])


def test_planning_itself_opens_no_files_or_database(monkeypatch):
    snapshot, feature = _inputs()

    def forbidden(*args, **kwargs):
        pytest.fail("pure planner attempted filesystem or DB I/O")

    with monkeypatch.context() as patch:
        patch.setattr("builtins.open", forbidden)
        patch.setattr(os, "open", forbidden)
        patch.setattr(Path, "open", forbidden)
        patch.setattr(sqlite3, "connect", forbidden)
        result = shortlist.plan_trade_shortlist(snapshot, feature)
    assert result["status"] == "planned"


@pytest.mark.parametrize("damage", ["nan", "inf", "bool", "count_bool", "negative_count", "duplicate_coin",
                                   "duplicate_rank", "candidate_99", "duplicate_feature", "future_window",
                                   "wrong_identity", "quality_contradiction", "wrong_ratio", "wrong_top3"])
def test_malformed_or_contradictory_inputs_fail_closed(damage):
    snapshot, feature = _inputs()
    if damage in {"nan", "inf", "bool"}:
        feature["rows"][0][native.FEATURE] = {"nan": float("nan"), "inf": float("inf"), "bool": True}[damage]
    elif damage == "count_bool":
        feature["rows"][0]["observed_trade_count"] = True
    elif damage == "negative_count":
        feature["rows"][0]["observed_trade_count"] = -1
    elif damage == "duplicate_coin":
        snapshot["universe"][99]["coin"] = snapshot["universe"][98]["coin"]
        _bind(snapshot, feature)
    elif damage == "duplicate_rank":
        snapshot["universe"][99]["rank"] = 99
        _bind(snapshot, feature)
    elif damage == "candidate_99":
        snapshot["universe"].pop()
        _bind(snapshot, feature)
    elif damage == "duplicate_feature":
        feature["rows"][99] = copy.deepcopy(feature["rows"][98])
    elif damage == "future_window":
        feature["window"]["end_exclusive_at_ns"] += 1
    elif damage == "wrong_identity":
        feature["snapshot"]["snapshot_id"] = "other"
    elif damage == "quality_contradiction":
        feature["quality_reasons"] = ["incomplete"]
    elif damage == "wrong_ratio":
        feature["rows"][0][native.FEATURE] = .9
    else:
        snapshot["top3"] = snapshot["universe"][1:4]
    with pytest.raises(shortlist.TradeShortlistError):
        shortlist.plan_trade_shortlist(snapshot, feature)


def test_returned_config_is_not_shared_mutable_state():
    snapshot, feature = _inputs()
    result = shortlist.plan_trade_shortlist(snapshot, feature)
    result["config"]["selection_order"].clear()
    assert shortlist.plan_trade_shortlist(snapshot, feature)["config"] == shortlist.SHORTLIST_CONFIG
    assert shortlist.SHORTLIST_CONFIG["selection_order"]
