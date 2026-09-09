"""Pure, hermetic slot diagnostics; no scoring, training, DB, or transport."""
from __future__ import annotations

import json
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ops.artifact_provenance import canonical_json_bytes, file_set_identity
from signals import recommend_slot_diagnostics as slots


def _case(days=7):
    rows, metadata = [], {}
    for offset in range(days):
        day = date(2026, 9, 1) + timedelta(days=offset)
        for slot in ("preopen", "open"):
            sid = f"snapshot-{day}-{slot}"
            is_open = slot == "open"
            coins = ["B", "C", "D", "A"] if is_open else ["A", "B", "C", "D"]
            sent = pd.Timestamp(f"{day}T{'09:08:26' if is_open else '08:53:12'}+09:00")
            entry = pd.Timestamp(f"{day}T{'09:15' if is_open else '09:00'}:00+09:00")
            end = entry + pd.Timedelta(days=1)
            available = pd.Timestamp(day + timedelta(days=1), tz="Asia/Seoul") + pd.Timedelta(hours=11)
            feature = day if is_open else day - timedelta(days=1)
            cutoff = feature - timedelta(days=5)
            meta = {
                "snapshot_id": sid, "date": str(day), "slot": slot,
                "model_id": f"recommend_r1_{slot}", "model_fit_mode": "ephemeral_daily_fit",
                "score_source_sha256": "a" * 64,
                "feature_row_date": str(feature), "nominal_shift1_input_bar_date": str(feature - timedelta(days=1)),
                "actual_latest_input_timestamp": None, "actual_input_freshness": "unknown_per_coin_timestamp_not_recorded",
                "db_max_timestamp_raw": f"{feature} 09:00:00", "db_manifest_id": "b" * 64,
                "training_cutoff_exclusive": str(cutoff), "training_base_start": "2025-01-01",
                "training_base_end": str(cutoff - timedelta(days=1)), "training_base_rows": 10000,
                "training_base_dates": 100, "embargo_days": 5,
                "actual_head_train_end": None, "actual_head_train_rows": None,
                "decision_started_at": (sent - pd.Timedelta(seconds=15)).isoformat(),
                "decision_completed_at": (sent - pd.Timedelta(seconds=.8)).isoformat(),
                "attempted_at": (sent - pd.Timedelta(seconds=.5)).isoformat(),
                "sent_at": sent.isoformat(), "recorded_at": (sent + pd.Timedelta(seconds=.6)).isoformat(),
                "execution_start_at": entry.isoformat(), "outcome_end_at": end.isoformat(),
                "label_created_at": available.isoformat(), "label_available_at": available.isoformat(),
                "execution_time_basis": "delivery_sent_at", "round_trip_cost_fraction": .0015,
            }
            for rank, coin in enumerate(coins, 1):
                positive = (ord(coin) + offset + int(is_open)) % 3 == 0
                downside = (ord(coin) + offset + int(is_open)) % 4 == 0
                rows.append({
                    "date": str(day), "slot": slot, "snapshot_id": sid, "coin": coin, "rank": rank,
                    "p_up10": .9 - rank / 10, "p_dn5": .2, "p_dn10": .1,
                    "exp_downside": -.03, "rr_ratio": (.9 - rank / 10) / .2,
                    "was_delivered": rank <= 3, "label_status": "labeled", "label_available": True,
                    "decision_started_at": meta["decision_started_at"], "outcome_end_at": meta["outcome_end_at"],
                    "label_available_at": meta["label_available_at"], "up10": positive, "dn5": downside,
                    "mfe": .12 if positive else .04, "mae": -.08 if downside else -.02,
                    "eod_return_net": (ord(coin) - 66 + offset / 5 + int(is_open)) / 100 - .0015,
                    "tp5_sl3_return_net": -.0315 if downside else .0485,
                })
            group = [row for row in rows if row["snapshot_id"] == sid]
            meta.update(universe_n=4, universe_identity_sha256=slots._digest(slots._identities(group)),
                        top3=slots._identities(group[:3]))
            metadata[sid] = meta
    frame = pd.DataFrame(rows)
    for name in ("up10", "dn5"):
        frame[name] = frame[name].astype(object)
    return frame, metadata


@pytest.fixture
def case():
    return _case()


def _run(case, **kwargs):
    return slots.analyze_slots(*case, n_boot=80, seed=42, **kwargs)


def _resync(frame, metadata, sid):
    group = frame.loc[frame.snapshot_id.eq(sid)]
    metadata[sid]["universe_n"] = len(group)
    metadata[sid]["universe_identity_sha256"] = slots._digest(slots._identities(group.to_dict("records")))
    metadata[sid]["top3"] = slots._identities(group.loc[group.was_delivered].to_dict("records"))


def test_pqr_use_fixed_picks_common_dates_and_exact_daily_additivity(case):
    frame, _ = case
    report = _run(case)
    assert report["cohort"]["dates"] == 7 and report["cohort"]["pick_rows_per_leg"] == 21
    assert report["coverage"]["excluded_dates"] == 0
    assert report["status"] == "historical_diagnostic"
    # A:1->4, B:2->1, C:3->2, D:4->3 => MAE=1.5, not RMS=sqrt(3).
    assert report["common_candidate_diagnostics"][0]["mean_absolute_rank_change"] == 1.5
    for row in report["daily"]:
        assert row["P"] == row["Q"] == ["A", "B", "C"]
        assert row["R"] == ["B", "C", "D"]
        assert row["timing"]["entry_gap_seconds"] == 900
        assert row["timing"]["feature_row_gap_days"] == row["timing"]["training_base_end_gap_days"] == 1
        expected = {}
        for leg, slot in (("P", "preopen"), ("Q", "open"), ("R", "open")):
            chosen = frame.loc[frame.date.eq(row["date"]) & frame.slot.eq(slot) & frame.coin.isin(row[leg])]
            expected[leg] = chosen.eod_return_net.mean()
            assert row["metrics"][leg]["eod_return_net"] == pytest.approx(expected[leg])
        assert row["metrics"]["window"]["eod_return_net"] == pytest.approx(expected["Q"] - expected["P"])
        for metric in slots.METRICS:
            assert row["metrics"]["window"][metric] + row["metrics"]["selection"][metric] == pytest.approx(row["metrics"]["total"][metric])
    assert len(report["leave_one_date_out"]) == 7
    assert report["summary"]["ci_status"] == "available"
    assert report["deployable"] is report["is_untouched_holdout"] is False
    json.dumps(report, allow_nan=False)


def test_frame_metadata_unchanged_and_shuffle_invariant(case):
    frame, metadata = case
    before, meta_before = frame.copy(deep=True), deepcopy(metadata)
    expected = _run(case)
    shuffled = frame.sample(frac=1, random_state=8)
    actual = slots.analyze_slots(shuffled, dict(reversed(list(metadata.items()))), n_boot=80, seed=42)
    assert actual == expected
    pd.testing.assert_frame_equal(frame, before)
    assert metadata == meta_before


def test_missing_open_slot_excludes_whole_date_not_zero(case):
    frame, metadata = case
    sid = "snapshot-2026-09-01-open"
    frame = frame.loc[~frame.snapshot_id.eq(sid)].copy()
    metadata.pop(sid)
    report = _run((frame, metadata))
    assert report["cohort"]["dates"] == 6
    assert report["coverage"]["excluded"] == [{"date": "2026-09-01", "reason": "missing_slot", "missing_slots": ["open"], "coins": []}]


def test_preopen_pick_absent_from_original_open_universe_is_a_legal_exclusion(case):
    frame, metadata = case
    sid = "snapshot-2026-09-01-open"
    frame.loc[frame.snapshot_id.eq(sid) & frame.coin.eq("A"), "coin"] = "X"
    _resync(frame, metadata, sid)
    report = _run(case)
    assert report["cohort"]["dates"] == 6
    assert report["coverage"]["excluded"][0]["reason"] == "preopen_pick_missing_from_open_universe"
    assert report["coverage"]["excluded"][0]["coins"] == ["A"]
    assert all("2026-09-01" != row["date"] for row in report["daily"])


def _halt(frame, mask):
    frame.loc[mask, "label_status"] = "halted_no_observations"
    frame.loc[mask, "label_available"] = False
    frame.loc[mask, ["up10", "dn5", "mfe", "mae", "eod_return_net", "tp5_sl3_return_net"]] = np.nan


@pytest.mark.parametrize("slot,coin", [("preopen", "A"), ("open", "A"), ("open", "D")])
def test_unlabeled_selected_leg_excludes_all_legs_without_reselection(case, slot, coin):
    frame, _ = case
    _halt(frame, frame.date.eq("2026-09-01") & frame.slot.eq(slot) & frame.coin.eq(coin))
    report = _run(case)
    assert report["cohort"]["dates"] == 6 and report["cohort"]["pick_rows_per_leg"] == 18
    exclusion = report["coverage"]["excluded"][0]
    assert exclusion["reason"] == "selected_label_unavailable" and exclusion["coins"] == [coin]
    assert all(row["P"] == ["A", "B", "C"] for row in report["daily"])


def test_future_open_pick_never_requires_preopen_outcome(case):
    frame, _ = case
    # D is selected only by open. Its old-window label must not be consulted.
    _halt(frame, frame.slot.eq("preopen") & frame.coin.eq("D"))
    report = _run(case)
    assert report["cohort"]["dates"] == 7 and report["coverage"]["excluded_dates"] == 0


@pytest.mark.parametrize("kind", ["missing_metadata", "extra_metadata", "time_end", "time_decision", "label_available",
                                  "metadata_identity", "nominal_date", "fake_freshness", "training_end", "entry",
                                  "delivery_before_decision", "label_before_end"])
def test_metadata_corruption_is_not_a_routine_exclusion(case, kind):
    frame, metadata = case
    sid = "snapshot-2026-09-01-open"
    meta = metadata[sid]
    if kind == "missing_metadata":
        metadata.pop(sid)
    elif kind == "extra_metadata":
        metadata["other"] = deepcopy(meta)
    elif kind in {"time_end", "time_decision", "label_available"}:
        field = {"time_end": "outcome_end_at", "time_decision": "decision_started_at", "label_available": "label_available_at"}[kind]
        frame.loc[frame.snapshot_id.eq(sid), field] = "2026-09-01T00:00:00+00:00"
    elif kind == "metadata_identity":
        meta["snapshot_id"] = "other"
    elif kind == "nominal_date":
        meta["feature_row_date"] = "2026-08-31"
    elif kind == "fake_freshness":
        meta["actual_latest_input_timestamp"] = "2026-09-01T00:00:00+00:00"
    elif kind == "training_end":
        meta["training_base_end"] = meta["training_cutoff_exclusive"]
    elif kind == "entry":
        meta["execution_start_at"] = "2026-09-01T09:00:00+09:00"
    elif kind == "delivery_before_decision":
        meta["sent_at"] = "2026-09-01T09:07:00+09:00"
    else:
        meta["label_created_at"] = "2026-09-01T10:00:00+09:00"
    with pytest.raises(ValueError):
        _run(case)


@pytest.mark.parametrize("kind", ["binary_contradiction", "halted_has_outcome", "availability_contradiction",
                                  "missing_candidate", "duplicate_candidate", "wrong_top3", "wrong_parity"])
def test_frame_corruption_hard_fails_even_on_nonselected_rows(case, kind):
    frame, _ = case
    index = frame.index[(frame.date == "2026-09-01") & (frame.slot == "preopen") & (frame.coin == "D")][0]
    if kind == "binary_contradiction":
        frame.at[index, "up10"] = not frame.at[index, "up10"]
    elif kind == "halted_has_outcome":
        frame.at[index, "label_status"] = "halted_no_observations"
        frame.at[index, "label_available"] = False
    elif kind == "availability_contradiction":
        frame.at[index, "label_available"] = False
    elif kind == "missing_candidate":
        frame.drop(index, inplace=True)
    elif kind == "duplicate_candidate":
        frame.loc[len(frame)] = frame.loc[index]
    elif kind == "wrong_top3":
        frame.at[index, "was_delivered"] = True
    else:
        frame.at[index, "p_up10"] = .99
    with pytest.raises(ValueError):
        _run(case)


def test_all_dates_legally_unavailable_produce_no_means_or_fake_ci(case):
    frame, _ = case
    _halt(frame, frame.coin.eq("A"))
    report = _run(case)
    assert report["cohort"]["dates"] == 0
    assert report["status"] == "no_common_evaluable_dates"
    assert report["coverage"]["excluded_dates"] == 7
    assert report["summary"]["means"] is None and report["summary"]["bootstrap"] == {}
    assert report["leave_one_date_out"] == [] and report["daily"] == []
    json.dumps(report, allow_nan=False)


def test_small_sample_has_means_without_ci():
    report = _run(_case(days=2))
    assert report["summary"]["ci_status"] == "insufficient_dates"
    assert report["summary"]["means"] is not None
    assert report["summary"]["bootstrap"] == {}


def test_identical_outcomes_produce_exact_zero_components(case):
    frame, _ = case
    frame["up10"], frame["dn5"] = False, False
    frame["mfe"], frame["mae"] = .03, -.02
    frame["eod_return_net"], frame["tp5_sl3_return_net"] = .0085, .0085
    report = _run(case)
    for part in ("window", "selection", "total"):
        assert set(report["summary"]["means"][part].values()) == {0.0}
        for bootstrap in report["summary"]["bootstrap"].values():
            assert all(interval == [0.0, 0.0] for interval in bootstrap["ci95"][part].values())


def test_joint_bootstrap_matches_independent_resampling_and_nonwrapping_blocks():
    legs = np.random.default_rng(8).normal(size=(7, 3, len(slots.METRICS)))
    values = np.stack((legs[:, 0], legs[:, 1], legs[:, 2], legs[:, 1] - legs[:, 0],
                       legs[:, 2] - legs[:, 1], legs[:, 2] - legs[:, 0]), axis=1)
    result = slots._joint_summary(values, 80, 42)
    rng = np.random.default_rng(42)
    iid = rng.integers(0, 7, size=(80, 7))
    starts = rng.integers(0, 5, size=(80, 3))
    blocks = (starts[:, :, None] + np.arange(3)).reshape(80, -1)[:, :7]
    assert blocks.min() >= 0 and blocks.max() <= 6
    for name, indices in zip(slots.SLOT_CONFIG["bootstrap"], (iid, blocks), strict=True):
        samples = values[indices].mean(axis=1)
        assert np.allclose(samples[:, 3] + samples[:, 4], samples[:, 5], atol=1e-12)
        actual = result["bootstrap"][name]
        assert actual["joint_indices_sha256"] == slots._digest(indices.tolist())
        low, high = np.quantile(samples, [.025, .975], axis=0)
        for i, part in enumerate(slots.PARTS):
            for j, metric in enumerate(slots.METRICS):
                assert actual["ci95"][part][metric] == pytest.approx([low[i, j], high[i, j]])
        assert actual["replicate_additivity_max_error"] < 1e-12
        metric = slots.METRICS[0]
        assert actual["ci95"]["window"][metric][0] + actual["ci95"]["selection"][metric][0] != pytest.approx(actual["ci95"]["total"][metric][0])


def test_additive_contract_failure_is_rejected():
    values = np.zeros((6, 6, len(slots.METRICS)))
    values[0, 3, 0] = 1
    with pytest.raises(slots.SlotDiagnosticError, match="selection"):
        slots._joint_summary(values, 80, 42)


@pytest.mark.parametrize("kwargs", [{"n_boot": True}, {"n_boot": 0}, {"seed": -1}, {"seed": 1.5}])
def test_invalid_bootstrap_arguments(case, kwargs):
    with pytest.raises(ValueError):
        slots.analyze_slots(*case, **kwargs)


@pytest.fixture
def stored(tmp_path, monkeypatch):
    frame, metas = _case(days=1)
    records, manifests, files = {}, [], {}
    for sid, meta in metas.items():
        paths = {role: tmp_path / role / meta["date"] / f"{meta['slot']}_r1.json"
                 for role in ("snapshot", "label", "receipt")}
        for role, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
            files[f"{sid}:{role}"] = path
        universe = frame.loc[frame.snapshot_id.eq(sid), ["coin", "rank"]].to_dict("records")
        snapshot = {
            "snapshot_id": sid, "asof": meta["date"], "slot": meta["slot"],
            "feature_asof": meta["feature_row_date"],
            "model": {"id": meta["model_id"], "fit_mode": meta["model_fit_mode"]},
            "code": {"score_source_sha256": meta["score_source_sha256"]},
            "data": {"max_timestamp": meta["db_max_timestamp_raw"], "manifest_id": meta["db_manifest_id"]},
            "training": {**{name: meta[f"training_base_{name}"] for name in ("start", "end", "rows", "dates")},
                         "cutoff_exclusive": meta["training_cutoff_exclusive"], "embargo_days": 5},
            "decision_started_at": meta["decision_started_at"], "decision_completed_at": meta["decision_completed_at"],
            "universe": universe, "top3": universe[:3],
        }
        label = {"execution_start_at": meta["execution_start_at"], "path_window_end": meta["outcome_end_at"],
                 "labeled_at": meta["label_created_at"], "execution_time_basis": "delivery_sent_at", "round_trip_cost_fraction": .0015}
        receipt = {key: meta[key] for key in ("attempted_at", "sent_at", "recorded_at")}
        manifest = {"snapshot_id": sid, "files": file_set_identity(paths, root=tmp_path),
                    "label_available_at": meta["label_available_at"]}
        records[paths["snapshot"]] = {"snapshot": snapshot, "label": label, "receipt": receipt, "manifest": manifest}
        manifests.append(deepcopy(manifest))
    dataset = {"frame": frame, "as_of": "2026-09-03T00:00:00+00:00", "provenance": {
        "root": str(tmp_path), "files": file_set_identity(files, root=tmp_path), "input_manifests": manifests,
    }}
    calls = []

    def load(path, **kwargs):
        calls.append((path, kwargs))
        return deepcopy(records[path])

    monkeypatch.setattr(slots, "load_recommendation_evidence", load)
    return dataset, files, calls, records


def test_metadata_loader_rebinds_only_frozen_files_preserves_unknowns_and_inputs(stored):
    dataset, paths, calls, _ = stored
    before_frame = dataset["frame"].copy(deep=True)
    before_provenance = deepcopy(dataset["provenance"])
    before_files = {key: path.read_bytes() for key, path in paths.items()}
    metadata = slots.load_slot_metadata(dataset)
    assert len(metadata) == len(calls) == 2
    for sid, meta in metadata.items():
        assert meta["actual_latest_input_timestamp"] is meta["actual_head_train_end"] is meta["actual_head_train_rows"] is None
        assert "base train" in meta["training_summary_scope"]
        assert meta["universe_n"] == 4 and meta["top3"][0]["rank"] == 1
        assert "global" in meta["db_timestamp_scope"]
        assert meta["snapshot_id"] == sid
    assert all(kwargs["now"] == pd.Timestamp(dataset["as_of"]) for _, kwargs in calls)
    assert slots.analyze_slots(dataset["frame"], metadata, n_boot=10)["cohort"]["dates"] == 1
    pd.testing.assert_frame_equal(before_frame, dataset["frame"])
    assert before_provenance == dataset["provenance"]
    assert before_files == {key: path.read_bytes() for key, path in paths.items()}


def test_metadata_loader_changed_file_blocks_before_rebind(stored):
    dataset, paths, calls, _ = stored
    next(iter(paths.values())).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="before"):
        slots.load_slot_metadata(dataset)
    assert calls == []


def test_metadata_loader_change_during_read_fails(stored, monkeypatch):
    dataset, paths, _, _ = stored
    original = slots.load_recommendation_evidence

    def changed(*args, **kwargs):
        evidence = original(*args, **kwargs)
        next(iter(paths.values())).write_text("changed while reading", encoding="utf-8")
        return evidence

    monkeypatch.setattr(slots, "load_recommendation_evidence", changed)
    with pytest.raises(ValueError, match="during"):
        slots.load_slot_metadata(dataset)


def test_metadata_loader_manifest_mismatch_is_fatal(stored):
    dataset, _, _, records = stored
    next(iter(records.values()))["manifest"]["label_available_at"] = "2026-09-04T00:00:00+00:00"
    with pytest.raises(ValueError, match="manifest mismatch"):
        slots.load_slot_metadata(dataset)


def test_metadata_loader_ignores_unrecorded_new_files(stored):
    dataset, paths, calls, _ = stored
    extra = Path(dataset["provenance"]["root"]) / "snapshot/2026-09-02/open_r1.json"
    extra.parent.mkdir()
    extra.write_text("invalid unrelated new file", encoding="utf-8")
    result = slots.load_slot_metadata(dataset)
    assert len(result) == 2 and len(calls) == 2
    assert extra not in paths.values()


def test_result_config_and_metadata_are_not_mutable_aliases(case):
    report = _run(case)
    report["config"]["legs"]["P"] = "changed"
    report["decision_metadata"][0]["top3"][0]["coin"] = "changed"
    assert slots.SLOT_CONFIG["legs"]["P"] != "changed"
    assert all(meta["top3"][0]["coin"] != "changed" for meta in case[1].values())
    canonical_json_bytes(_run(case))
