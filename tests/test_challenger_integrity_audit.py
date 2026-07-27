from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import scripts.downside_veto_challenger_v1 as downside
import scripts.first_passage_head_challenger_v1 as fp
import scripts.safeup_head_challenger_v1 as safeup
import scripts.semivol_joint_challenger_v1 as semivol
import scripts.upside_head_challenger_v1 as upside


def _make_15m_db(path: Path, date: str = "2024-01-01") -> None:
    start = pd.Timestamp(date) + pd.Timedelta(hours=9)
    timestamps = pd.date_range(
        start=start,
        end=start + pd.Timedelta(days=1, minutes=15),
        freq="15min",
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE candles (
                market TEXT,
                timestamp TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL
            )
            """
        )
        connection.executemany(
            "INSERT INTO candles VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    safeup.BENCHMARK_MARKET,
                    timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                )
                for timestamp in timestamps
            ],
        )
        connection.execute(
            "INSERT INTO candles VALUES (?, ?, ?, ?, ?, ?)",
            (
                "KRW-TEST",
                (start - pd.Timedelta(minutes=15)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                10.0,
                10.0,
                10.0,
                10.0,
            ),
        )


def _valid_oos_frame(date: str = "2024-01-01") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [date] * 100,
            "market": [f"KRW-T{i:03d}" for i in range(100)],
            "fold": [0] * 100,
            "p_lab_up_10": np.linspace(0.01, 0.50, 100),
            "p_lab_dn_05": np.linspace(0.02, 0.51, 100),
            "p_lab_dn_10": np.linspace(0.01, 0.20, 100),
            "exp_downside": np.linspace(-0.2, -0.01, 100),
            "up_high_ret": [0.20] * 100,
            "down_low_ret": [-0.10] * 100,
            "eod_ret": [0.05] * 100,
            "f_qv_rank": np.arange(1, 101),
        }
    )


def _valid_path_panel(date: str = "2024-01-01") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market": [f"KRW-T{i:03d}" for i in range(100)],
            "date": [date] * 100,
            "history_prior_bars": [100] * 100,
            "f_qv_rank": np.arange(1, 101),
            "path_complete": [True] * 100,
            "benchmark_complete": [True] * 100,
            "path_quality": ["complete"] * 100,
            "path_raw_bars": [96] * 100,
            "path_flat_filled_bars": [0] * 100,
            "path_benchmark_bars": [96] * 100,
            fp.FIXED_TARGET: [0, 1] * 50,
            fp.ATR_TARGET: [0, 1] * 50,
            "path_up10": [0, 1] * 50,
            "path_dn5": [1, 0] * 50,
            "path_bracket_net": [-0.0315, 0.0485] * 50,
            "path_eod_net": [-0.0015] * 100,
        }
    )


def test_downside_oos_rejects_nonfinite_and_rank_corruption(
    tmp_path: Path,
) -> None:
    frame = _valid_oos_frame()
    path = tmp_path / "oos.csv"
    frame.to_csv(path, index=False)
    loaded = downside._read_oos(path)
    assert len(loaded) == 100

    frame.loc[0, "p_lab_up_10"] = np.inf
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="nonfinite"):
        downside._read_oos(path)

    frame = _valid_oos_frame()
    frame.loc[0, "f_qv_rank"] = 2
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="permutation"):
        downside._read_oos(path)


def test_downside_oos_rejects_incomplete_label_date(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _valid_oos_frame("2024-01-02")
    path = tmp_path / "oos.csv"
    frame.to_csv(path, index=False)
    monkeypatch.setattr(
        downside, "_completed_label_cutoff", lambda: pd.Timestamp("2024-01-01").date()
    )
    with pytest.raises(ValueError, match="not-yet-completed"):
        downside._read_oos(path)


def test_downside_path_window_starts_at_first_executable_bar() -> None:
    start, grid = downside._window(pd.Timestamp("2024-01-01").date())

    assert start == pd.Timestamp("2024-01-01 09:15:00")
    assert len(grid) == 96
    assert grid[0] == start
    assert grid[-1] == pd.Timestamp("2024-01-02 09:00:00")


def test_zero_observation_target_is_never_flat_filled_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "m15.db"
    _make_15m_db(db)
    pair = pd.DataFrame(
        {"market": ["KRW-TEST"], "date": [pd.Timestamp("2024-01-01").date()]}
    )

    d1_path = downside._bulk_paths(pair, db)[
        ("KRW-TEST", pd.Timestamp("2024-01-01").date())
    ]
    assert not d1_path.complete
    assert d1_path.quality == "target_no_observations"

    execution_path = safeup._bulk_execution_paths(pair, db)[0][
        ("KRW-TEST", pd.Timestamp("2024-01-01").date())
    ]
    assert not execution_path.complete
    assert execution_path.quality == "target_no_observations"

    monkeypatch.setattr(upside, "UPBIT_15M_DB", db)
    cached = upside.build_path_cache(pair)[0][
        ("KRW-TEST", pd.Timestamp("2024-01-01").date())
    ]
    assert not cached["path_complete"]
    assert cached["path_quality"] == "target_no_observations"


def test_string_false_cache_value_remains_false() -> None:
    frame = pd.DataFrame({"path_complete": ["False", "true"]})
    parsed = fp._parse_bool_column(frame, "path_complete")
    assert parsed.tolist() == [False, True]


def test_labeled_panel_enforces_exact_96_bar_contract() -> None:
    panel = _valid_path_panel()
    validated = fp._validate_labeled_panel(panel)
    assert len(validated) == 100

    panel.loc[0, "path_raw_bars"] = 0
    panel.loc[0, "path_flat_filled_bars"] = 96
    with pytest.raises(RuntimeError, match="exact 96-bar"):
        fp._validate_labeled_panel(panel)


def test_first_passage_same_bar_is_downside_first_and_cost_once() -> None:
    bars = ((100.0, 111.0, 94.0, 100.0),) + (
        (100.0, 100.0, 100.0, 100.0),
    ) * 95
    label, outcome = fp._first_passage(
        bars, up_barrier=0.10, down_barrier=0.05
    )
    assert label == 0
    assert outcome == "down_first_same_bar"

    flat = safeup.BulkPath(
        bars=((100.0, 100.0, 100.0, 100.0),) * 96,
        complete=True,
        quality="complete",
        raw_bars=96,
        flat_filled_bars=0,
        benchmark_bars=96,
    )
    labels = fp._path_labels(flat, 0.02)
    assert labels["path_eod_net"] == pytest.approx(-fp.ROUND_TRIP_COST)
    assert labels["path_bracket_net"] == pytest.approx(-fp.ROUND_TRIP_COST)


def test_baseline_alignment_fails_on_nonfinite_or_missing(
    tmp_path: Path,
) -> None:
    date = pd.Timestamp("2024-01-01").date()
    predictions = pd.DataFrame(
        {
            "split_schedule_sha256": ["schedule"] * 100,
            "scope": ["discovery_oof"] * 100,
            "fold": [0] * 100,
            "date": [date] * 100,
            "market": [f"KRW-T{i:03d}" for i in range(100)],
        }
    )
    baseline_discovery = pd.DataFrame(
        {
            "split_schedule_sha256": ["schedule"] * 100,
            "scope": ["discovery_oof"] * 100,
            "fold": [0] * 100,
            "date": [date] * 100,
            "market": predictions["market"],
            "score_R1_repaired": [1.0] * 100,
            "score_safeup_head": [0.5] * 100,
        }
    )
    baseline_holdout = baseline_discovery.copy()
    baseline_holdout["scope"] = "locked_holdout"
    baseline_holdout["fold"] = -1
    baseline = pd.concat(
        [baseline_discovery, baseline_holdout],
        ignore_index=True,
    )
    path = tmp_path / "baseline.csv"
    baseline.loc[0, "score_R1_repaired"] = np.inf
    baseline.to_csv(path, index=False)
    with pytest.raises(RuntimeError, match="nonfinite"):
        fp.attach_reproducible_baselines(predictions, path)

    baseline = baseline.iloc[1:].copy()
    baseline["score_R1_repaired"] = 1.0
    baseline.to_csv(path, index=False)
    with pytest.raises(RuntimeError, match="exactly cover"):
        fp.attach_reproducible_baselines(predictions, path)

    wrong_schedule = pd.concat(
        [baseline_discovery, baseline_holdout],
        ignore_index=True,
    )
    wrong_schedule["split_schedule_sha256"] = "other-schedule"
    wrong_schedule.to_csv(path, index=False)
    with pytest.raises(RuntimeError, match="split-schedule hash mismatch"):
        fp.attach_reproducible_baselines(predictions, path)


def test_expanding_splits_have_exact_embargo_and_no_overlap() -> None:
    dates = [pd.Timestamp("2024-01-01").date() + pd.Timedelta(days=i) for i in range(300)]
    for splits in (
        safeup._expanding_splits(
            dates,
            n_folds=5,
            embargo=5,
            warmup_fraction=0.35,
            minimum_warmup=90,
        ),
        upside._outer_splits(dates, n_folds=5, embargo=5),
        upside._inner_splits(dates, n_folds=3, embargo=5),
    ):
        seen: set = set()
        all_test_dates: set = set()
        ordered = sorted(set(dates))
        positions = {value: index for index, value in enumerate(ordered)}
        for train_dates, test_dates in splits:
            assert set(train_dates).isdisjoint(test_dates)
            assert seen.isdisjoint(test_dates)
            assert (
                positions[min(test_dates)] - positions[max(train_dates)] - 1
                == 5
            )
            seen.update(test_dates)
            all_test_dates.update(test_dates)
        test_positions = sorted(
            positions[value] for value in all_test_dates
        )
        assert test_positions == list(
            range(test_positions[0], test_positions[-1] + 1)
        )


def test_safeup_oof_dates_cover_nested_first_passage_oof_cohort() -> None:
    dates = [
        pd.Timestamp("2023-08-10").date() + pd.Timedelta(days=i)
        for i in range(1_081)
    ]
    safeup_discovery = dates[:-safeup.LOCKED_HOLDOUT_DATES]
    safeup_splits = safeup._outer_splits(safeup_discovery)
    safeup_oof_dates = {
        date
        for _, test_dates in safeup_splits
        for date in test_dates
    }

    benchmark_complete = dates[-445:]
    first_passage_discovery = benchmark_complete[
        :-fp.LOCKED_COMMON_DATES
    ]
    first_passage_splits = fp._expanding_splits(
        first_passage_discovery,
        n_folds=fp.OUTER_FOLDS,
        minimum_warmup=90,
    )
    first_passage_oof_dates = {
        date
        for _, test_dates in first_passage_splits
        for date in test_dates
    }

    assert len(safeup_splits) == safeup.OUTER_FOLDS
    assert len(first_passage_splits) == fp.OUTER_FOLDS
    assert first_passage_oof_dates <= safeup_oof_dates


def test_safeup_prediction_scope_contract_rejects_internal_oof_gap() -> None:
    dates = list(
        pd.date_range("2025-01-01", periods=10, freq="D").date
    )
    rows = [
        {
            "scope": (
                "discovery_oof" if date < dates[5] else "locked_holdout"
            ),
            "date": date,
            "market": f"KRW-T{index:03d}",
        }
        for date in dates
        for index in range(100)
    ]
    predictions = pd.DataFrame(rows)
    holdout_dates = dates[5:]
    locked = {
        "n_dates": len(holdout_dates),
        "start": str(holdout_dates[0]),
        "end": str(holdout_dates[-1]),
        "dates_sha256": hashlib.sha256(
            "\n".join(map(str, holdout_dates)).encode()
        ).hexdigest(),
    }
    audit = safeup._prediction_scope_contract(predictions, locked)
    assert audit["prediction_dates_contiguous"]

    with_gap = predictions[predictions["date"] != dates[2]]
    with pytest.raises(RuntimeError, match="internal date gaps"):
        safeup._prediction_scope_contract(with_gap, locked)


def test_common_schedule_keeps_exact_180_benchmark_dates() -> None:
    calendar = list(
        pd.date_range("2025-09-01", periods=300, freq="D").date
    )
    unavailable = {calendar[-75], calendar[-149]}
    benchmark_complete = [
        date for date in calendar if date not in unavailable
    ]
    schedule, splits, holdout_train, holdout = (
        safeup.build_common_benchmark_schedule(benchmark_complete)
    )
    assert len(holdout) == 180
    assert list(holdout) == benchmark_complete[-180:]
    assert schedule["locked_holdout_dates"] == list(map(str, holdout))
    assert schedule["historical_holdout_contaminated"]
    assert len(splits) == safeup.OUTER_FOLDS
    assert schedule["folds"][-1]["train_end"] == str(
        holdout_train[-1]
    )
    assert schedule["folds"][-1]["embargo_dates"] == list(
        map(str, benchmark_complete[:-180][-5:])
    )
    assert (
        safeup._verify_schedule_contract(schedule)
        == schedule["split_schedule_sha256"]
    )
    tampered = json.loads(json.dumps(schedule))
    tampered["folds"][-1]["train_end"] = "1900-01-01"
    with pytest.raises(RuntimeError, match="hash mismatch"):
        safeup._verify_schedule_contract(tampered)
    self_consistent_but_invalid = json.loads(json.dumps(schedule))
    self_consistent_but_invalid["locked_holdout_dates"] = (
        self_consistent_but_invalid["locked_holdout_dates"][1:]
    )
    self_consistent_but_invalid["locked_holdout_dates_sha256"] = (
        safeup._dates_sha256(
            self_consistent_but_invalid["locked_holdout_dates"]
        )
    )
    self_consistent_but_invalid.pop("split_schedule_sha256")
    self_consistent_but_invalid["split_schedule_sha256"] = hashlib.sha256(
        json.dumps(
            self_consistent_but_invalid,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    with pytest.raises(RuntimeError, match="final-180 partition"):
        safeup._verify_schedule_contract(self_consistent_but_invalid)


def test_schedule_aligned_baseline_uses_common_scope_and_cutoffs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_dates = list(
        pd.date_range("2025-01-01", periods=320, freq="D").date
    )
    benchmark_dates = full_dates[-300:]
    panel = pd.DataFrame(
        [
            {"date": date, "market": f"KRW-T{index:03d}"}
            for date in full_dates
            for index in range(100)
        ]
    )
    captured = []

    def fake_predict(train, test, features, *, scope, fold):
        captured.append(
            {
                "scope": scope,
                "fold": fold,
                "train_end": max(train["date"]),
                "train_dates": sorted(train["date"].unique()),
                "test_dates": sorted(test["date"].unique()),
            }
        )
        result = test[["date", "market"]].copy()
        result["scope"] = scope
        result["fold"] = fold
        result["score_R1_repaired"] = 1.0
        result["score_safeup_head"] = 0.5
        return result, [{"scope": scope, "fold": fold}]

    monkeypatch.setattr(safeup, "_predict_split", fake_predict)
    predictions, schedule, _ = safeup.run_common_schedule_baseline(
        panel,
        [],
        benchmark_dates,
    )
    assert len(captured) == safeup.OUTER_FOLDS + 1
    assert captured[-1]["scope"] == "locked_holdout"
    assert captured[-1]["fold"] == -1
    assert captured[-1]["train_end"] == pd.Timestamp(
        schedule["folds"][-1]["train_end"]
    ).date()
    assert captured[-1]["train_dates"] == [
        pd.Timestamp(value).date()
        for value in schedule["folds"][-1]["train_dates"]
    ]
    assert captured[-1]["test_dates"] == [
        pd.Timestamp(value).date()
        for value in schedule["locked_holdout_dates"]
    ]
    for captured_fold, contract in zip(captured, schedule["folds"]):
        assert captured_fold["train_dates"] == [
            pd.Timestamp(value).date()
            for value in contract["train_dates"]
        ]
        assert captured_fold["test_dates"] == [
            pd.Timestamp(value).date()
            for value in contract["test_dates"]
        ]
    assert set(predictions["split_schedule_sha256"]) == {
        schedule["split_schedule_sha256"]
    }
    assert not predictions.duplicated(
        [
            "split_schedule_sha256",
            "scope",
            "fold",
            "date",
            "market",
        ]
    ).any()


def test_safeup_holdout_embargo_uses_eligible_dates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = [
        pd.Timestamp("2024-01-01").date() + pd.Timedelta(days=i * 2)
        for i in range(20)
    ]
    panel = pd.DataFrame(
        {"date": dates, "market": ["KRW-TEST"] * len(dates)}
    )
    captured: dict = {}

    def fake_predict(train, test, features, *, scope, fold):
        captured["train"] = list(train["date"])
        return test.copy(), [{"scope": scope, "fold": fold}]

    monkeypatch.setattr(safeup, "_predict_split", fake_predict)
    safeup.run_locked_holdout(panel, np.asarray(dates[-2:]), [])
    assert captured["train"][-1] == dates[-8]
    assert set(dates[-7:-2]).isdisjoint(captured["train"])


def test_semivol_lineage_rejects_tampered_path_panel(
    tmp_path: Path,
) -> None:
    panel_path = tmp_path / "panel.csv.gz"
    panel_path.write_bytes(b"panel")
    d1_db = tmp_path / "d1.db"
    d1_db.write_bytes(b"d1")
    meta_path = tmp_path / "meta.json"
    metadata = {
        "signature": {
            "schema": fp.PATH_CACHE_SCHEMA,
            "d1_db": fp._file_signature(d1_db),
            "safeup_script_sha256": semivol._sha256(Path(safeup.__file__)),
            "code_lineage": fp._code_lineage(),
            "completed_label_cutoff": str(
                safeup._completed_label_cutoff().date()
            ),
        },
        "path_meta": {},
        "cache_sha256": hashlib.sha256(b"panel").hexdigest(),
    }
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    semivol._validate_path_panel_lineage(
        panel_path=panel_path,
        meta_path=meta_path,
        d1_db=d1_db,
    )
    panel_path.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="bytes"):
        semivol._validate_path_panel_lineage(
            panel_path=panel_path,
            meta_path=meta_path,
            d1_db=d1_db,
        )


def test_semivol_feature_source_must_be_no_later_than_d_minus_1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    date = pd.Timestamp("2024-01-02").date()
    panel = pd.DataFrame({"market": ["KRW-TEST"], "date": [date]})

    def same_day_feature(market: str, d1_db: Path) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "market": [market],
                "date": [date],
                "semivol_source_date": [date],
                "downside_semivol_7": [0.1],
                "downside_semivol_21": [0.1],
                "upside_semivol_21": [0.1],
                "semivol_asym_21": [0.0],
            }
        )

    monkeypatch.setattr(semivol, "_market_semivol", same_day_feature)
    with pytest.raises(RuntimeError, match="later than D-1"):
        semivol.attach_semivol_features(panel, tmp_path / "unused.db")


def test_core_reference_nonfinite_fails_closed(tmp_path: Path) -> None:
    date = pd.Timestamp("2024-01-01").date()
    predictions = pd.DataFrame(
        {
            "split_schedule_sha256": ["schedule"],
            "scope": ["locked_holdout"],
            "fold": [-1],
            "date": [date],
            "market": ["KRW-BTC"],
            f"raw_{semivol.CORE_POLICY}": [0.1],
            f"p_{semivol.CORE_POLICY}": [0.1],
        }
    )
    reference = pd.DataFrame(
        {
            "split_schedule_sha256": ["schedule"],
            "scope": ["locked_holdout"],
            "fold": [-1],
            "date": [date],
            "market": ["KRW-BTC"],
            "raw_fp_fixed_head": [np.inf],
            "p_fp_fixed_head": [0.1],
        }
    )
    path = tmp_path / "reference.csv"
    reference.to_csv(path, index=False)
    with pytest.raises(RuntimeError, match="nonfinite"):
        semivol._core_reference_audit(predictions, path)


def test_upside_common_path_dates_are_global_across_policies() -> None:
    d1 = pd.Timestamp("2024-01-01").date()
    d2 = pd.Timestamp("2024-01-02").date()
    picks: dict[str, pd.DataFrame] = {}
    cache: dict[tuple[str, object], dict] = {}
    for policy in ("a", "b", "c"):
        rows = []
        for date in (d1, d2):
            for index in range(3):
                market = f"KRW-{policy.upper()}{index}"
                rows.append({"date": date, "market": market})
                cache[(market, date)] = {"path_complete": True}
        picks[policy] = pd.DataFrame(rows)
    cache[("KRW-B0", d2)] = {"path_complete": False}
    assert upside._common_complete_path_dates(picks, cache) == {d1}


def test_upside_evaluation_rejects_misaligned_variant_cohort() -> None:
    date = pd.Timestamp("2024-01-01").date()
    predictions = pd.DataFrame(
        {
            "variant": ["cls_core", "rank_core"],
            "date": [date, date],
            "market": ["KRW-BTC", "KRW-ETH"],
            "score_raw": [0.1, 0.2],
            "p_cal": [0.1, 0.2],
        }
    )
    with pytest.raises(RuntimeError, match="not aligned"):
        upside.evaluate(predictions, include_paths=False)


def test_manifest_validation_rejects_missing_or_tampered_artifacts(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.csv"
    artifact.write_text("value\n1\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "test_manifest",
                "files": {
                    "artifact": {
                        "path": str(artifact),
                        "bytes": artifact.stat().st_size,
                        "sha256": fp._sha256(artifact),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    fp._verify_manifest(
        manifest_path=manifest,
        schema="test_manifest",
        expected={"artifact": artifact},
    )
    artifact.write_text("value\n2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum"):
        fp._verify_manifest(
            manifest_path=manifest,
            schema="test_manifest",
            expected={"artifact": artifact},
        )

    with pytest.raises(RuntimeError, match="missing"):
        fp._verify_manifest(
            manifest_path=tmp_path / "absent.json",
            schema="test_manifest",
            expected={"artifact": artifact},
        )


def test_first_passage_existing_validator_requires_manifest(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="manifest is missing"):
        fp.validate_existing_artifacts(
            output_prefix=tmp_path / "first_passage",
            d1_db=tmp_path / "d1.db",
            m15_db=tmp_path / "m15.db",
            baseline_predictions=tmp_path / "baseline.csv",
        )


def test_safeup_dependency_requires_generation_manifest(
    tmp_path: Path,
) -> None:
    predictions = tmp_path / "safeup_fp_schedule_predictions.csv.gz"
    predictions.write_bytes(b"not-a-valid-generation")
    d1_db = tmp_path / "d1.db"
    m15_db = tmp_path / "m15.db"
    d1_db.write_bytes(b"d1")
    m15_db.write_bytes(b"m15")
    with pytest.raises(RuntimeError, match="manifest is missing"):
        fp._validate_safeup_baseline_lineage(
            baseline_predictions=predictions,
            d1_db=d1_db,
            m15_db=m15_db,
        )


def test_standalone_validators_require_generation_manifest(
    tmp_path: Path,
) -> None:
    oos = tmp_path / "oos.csv"
    d1 = tmp_path / "d1.db"
    binance = tmp_path / "binance.db"
    m15 = tmp_path / "m15.db"
    for path in (oos, d1, binance, m15):
        path.write_bytes(b"input")
    with pytest.raises(RuntimeError, match="manifest is missing"):
        downside.validate_existing_artifacts(
            output_prefix=tmp_path / "downside",
            input_oos=oos,
            m15_db=m15,
        )
    with pytest.raises(RuntimeError, match="manifest is missing"):
        upside.validate_existing_artifacts(
            output_prefix=tmp_path / "upside",
            upbit_d1_db=d1,
            binance_d1_db=binance,
            upbit_15m_db=m15,
        )


@pytest.mark.parametrize("module", [downside, fp, safeup, upside])
def test_generation_publish_rolls_back_every_original(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module,
) -> None:
    target_a = tmp_path / "a.txt"
    target_b = tmp_path / "b.txt"
    target_a.write_text("old-a", encoding="utf-8")
    target_b.write_text("old-b", encoding="utf-8")
    stage = tmp_path / "stage"
    stage.mkdir()
    staged_a = stage / "a.txt"
    staged_b = stage / "b.txt"
    staged_a.write_text("new-a", encoding="utf-8")
    staged_b.write_text("new-b", encoding="utf-8")

    real_replace = module.os.replace
    failed = False

    def fail_second_publish(source, target):
        nonlocal failed
        if Path(source) == staged_b and not failed:
            failed = True
            raise OSError("injected publish failure")
        return real_replace(source, target)

    monkeypatch.setattr(module.os, "replace", fail_second_publish)
    with pytest.raises(OSError, match="injected"):
        module._publish_staged_files(
            {staged_a: target_a, staged_b: target_b}
        )
    assert target_a.read_text(encoding="utf-8") == "old-a"
    assert target_b.read_text(encoding="utf-8") == "old-b"


def test_generation_publish_preserves_recovery_backup_if_rollback_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_a = tmp_path / "a.txt"
    target_b = tmp_path / "b.txt"
    target_a.write_text("old-a", encoding="utf-8")
    target_b.write_text("old-b", encoding="utf-8")
    stage = tmp_path / "stage"
    stage.mkdir()
    staged_a = stage / "a.txt"
    staged_b = stage / "b.txt"
    staged_a.write_text("new-a", encoding="utf-8")
    staged_b.write_text("new-b", encoding="utf-8")
    real_replace = safeup.os.replace

    def fail_publish_and_rollback(source, target):
        source_path = Path(source)
        target_path = Path(target)
        if source_path == staged_b:
            raise OSError("injected publish failure")
        if (
            source_path.parent.name.startswith(".challenger-backup.")
            and target_path == target_a
        ):
            raise OSError("injected rollback failure")
        return real_replace(source, target)

    monkeypatch.setattr(safeup.os, "replace", fail_publish_and_rollback)
    with pytest.raises(RuntimeError, match="recovery backups preserved"):
        safeup._publish_staged_files(
            {staged_a: target_a, staged_b: target_b}
        )
    backup_directories = list(tmp_path.glob(".challenger-backup.*"))
    assert len(backup_directories) == 1
    assert any(
        path.read_text(encoding="utf-8") == "old-a"
        for path in backup_directories[0].iterdir()
    )


def test_first_passage_cache_is_restored_after_later_failure(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache.csv.gz"
    metadata = tmp_path / "cache.json"
    cache.write_bytes(b"old-cache")
    metadata.write_bytes(b"old-meta")
    with pytest.raises(RuntimeError, match="later failure"):
        with fp._preserve_existing_files_on_failure([cache, metadata]):
            cache.write_bytes(b"new-cache")
            metadata.write_bytes(b"new-meta")
            raise RuntimeError("later failure")
    assert cache.read_bytes() == b"old-cache"
    assert metadata.read_bytes() == b"old-meta"


def test_first_passage_validation_does_not_enter_cache_restore_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = SimpleNamespace(validate_existing=True)
    calls = []
    monkeypatch.setattr(fp, "parse_args", lambda: args)
    monkeypatch.setattr(fp, "_run", lambda value: calls.append(value))

    def forbidden_context(_targets):
        raise AssertionError("read-only validation entered cache backup")

    monkeypatch.setattr(
        fp,
        "_preserve_existing_files_on_failure",
        forbidden_context,
    )
    fp.main()
    assert calls == [args]
