from __future__ import annotations

import json
import argparse
import copy
from pathlib import Path

import pandas as pd
import pytest

import scripts.evaluate_recommend_score_labels as evaluator
from signals.recommend_score_labels import (
    FORWARD_PROVENANCE_COHORT,
    LABEL_SCHEMA_VERSION,
    SCHEDULED_REPLAY_PROVENANCE_COHORT,
    _artifact_digest,
    label_recommend_snapshot,
    load_label_artifact,
)
from ledger.path_quality import PathAssessment
from signals.recommend_snapshot import get_or_create_recommend_snapshot


def _row(date_index: int, rank: int, *, features: bool = True) -> dict:
    up10 = rank <= 5
    dn5 = rank % 5 == 0
    row = {
        "coin": f"KRW-T{rank:02d}",
        "rank": rank,
        "score": 1.0 - rank / 100.0,
        "p_up10": 0.8 if up10 else 0.1,
        "p_dn5": 0.7 if dn5 else 0.1,
        "up10": up10,
        "dn5": dn5,
        "mfe": 0.12 if up10 else 0.03,
        "mae": -0.06 if dn5 else -0.01,
        "eod_return": 0.02 if up10 else -0.01,
        "tp5_sl3_first_passage": "sl_first" if dn5 else (
            "tp_first" if up10 else "neither"
        ),
        "tp5_sl3_return_net": -0.0315 if dn5 else (
            0.0485 if up10 else -0.0115
        ),
        "label_status": "labeled",
        "path_complete": True,
        "path_quality": "complete",
        "delivery_ok": date_index % 2 == 0,
        "execution_at": f"2026-07-{date_index + 1:02d}T09:10:00+09:00",
    }
    row["feature_values"] = (
        {
            "f_log_qv": 10.0 + rank * 0.1,
            "f_atr_pct_14": 0.01 + (rank % 3) * 0.02,
        }
        if features
        else {}
    )
    return row


def _write_artifact(
    root: Path,
    date: str,
    *,
    n_rows: int = 20,
    features: bool = True,
    status: str = "complete",
    with_net: bool = False,
    with_eod_net: bool = False,
    provenance: str = FORWARD_PROVENANCE_COHORT,
    explicit_provenance: bool = True,
    reverse_up_head: bool = False,
) -> Path:
    rows = [_row(int(date[-2:]) - 1, rank, features=features)
            for rank in range(1, n_rows + 1)]
    basis = (
        "scheduled_slot_fallback_snapshot_outside_window"
        if provenance == SCHEDULED_REPLAY_PROVENANCE_COHORT
        else "snapshot_created_at_no_receipt"
    )
    for row in rows:
        row["execution_time_basis"] = basis
        if explicit_provenance:
            row["provenance_cohort"] = provenance
            row["forward_eligible"] = provenance == FORWARD_PROVENANCE_COHORT
        if reverse_up_head:
            row["p_up10"] = 1.0 - row["p_up10"]
    if with_net:
        for row in rows:
            row["net_return"] = row["eod_return"] - 0.0015
    if with_eod_net:
        for row in rows:
            row["eod_return_net"] = row["eod_return"] - 0.0015
    document = {
        "schema": LABEL_SCHEMA_VERSION,
        "artifact_status": status,
        "return_unit": "fraction",
        "asof": date,
        "slot": "open",
        "ranking": "R1",
        "feature_asof": date,
        "snapshot_id": f"snapshot-{date}",
        "snapshot_payload_sha256": f"hash-{date}",
        "snapshot_path": f"/snapshots/{date}/open_r1.json",
        "path_window_start": f"{date}T09:00:00+09:00",
        "path_window_end": f"{date}T09:00:00+09:00",
        "labeled_at": f"{date}T01:00:00+00:00",
        "execution_time_basis": basis,
        "summary": {
            "snapshot_universe_n": n_rows,
            "rows": n_rows,
            "labeled": n_rows if status == "complete" else 0,
            "incomplete": 0 if status == "complete" else n_rows,
            "flat_filled": 0,
        },
        "rows": rows,
    }
    if explicit_provenance:
        document["provenance_cohort"] = provenance
        document["forward_eligible"] = provenance == FORWARD_PROVENANCE_COHORT
    document["label_payload_sha256"] = _artifact_digest(document)
    path = root / date / "open_r1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return path


def test_full_audit_metrics_baselines_and_cluster_ci(tmp_path):
    root = tmp_path / "labels"
    for day in range(1, 9):
        _write_artifact(root, f"2026-07-{day:02d}")
    _write_artifact(root, "2026-07-09", status="partial")
    output = tmp_path / "evaluation.json"

    report = evaluator.evaluate_label_root(
        root,
        output_path=output,
        top_ns=(3,),
        n_boot=200,
        min_rows=30,
        min_days=5,
    )

    assert report["artifacts"]["complete_used"] == 8
    assert any(
        item["reason"] == "artifact_status=partial"
        for item in report["artifacts"]["skipped"]
    )
    channel = report["channels"]["open:R1"]
    assert channel["return_basis"]["field"] == "eod_return"
    assert channel["return_basis"]["is_net"] is False
    assert channel["return_basis"]["cost_adjustment"] == "none_gross_diagnostic"

    all_scores = channel["cohorts"]["all_scores"]
    assert all_scores["heads"]["p_up10"]["status"] == "ok"
    assert all_scores["heads"]["p_up10"]["auc"]["value"] == 1.0
    assert all_scores["heads"]["p_up10"]["auc"]["ci95"] is not None
    assert all_scores["heads"]["p_up10"]["brier"]["value"] < 0.05
    assert all_scores["heads"]["p_up10"]["calibration"]["bins"]

    delivered = channel["cohorts"]["delivered"]
    assert delivered["selection"] == "delivery_ok_true_and_rank_le_3"
    assert delivered["n_rows"] == 12
    assert delivered["heads"]["p_up10"]["status"] == "insufficient"
    assert delivered["heads"]["p_up10"]["auc"] is None

    top3 = channel["top_n_vs_full_universe"]["3"]
    up_lift = top3["metrics"]["up10_rate"]["difference_selected_minus_baseline"]
    dn_delta = top3["metrics"]["dn5_rate"]["difference_selected_minus_baseline"]
    safe_up_lift = top3["metrics"]["safe_up10_rate"][
        "difference_selected_minus_baseline"
    ]
    tp_lift = top3["metrics"]["tp_first_rate"][
        "difference_selected_minus_baseline"
    ]
    sl_delta = top3["metrics"]["sl_first_rate"][
        "difference_selected_minus_baseline"
    ]
    assert up_lift["value"] > 0
    assert up_lift["ci95"] is not None
    assert dn_delta["value"] < 0
    assert safe_up_lift["value"] > 0
    assert tp_lift["value"] > 0
    assert sl_delta["value"] < 0
    assert all_scores["outcomes"]["safe_up10_rate"] is not None
    assert all_scores["outcomes"]["first_passage_net_mean"] is not None

    liquidity = channel["liquidity_matched_baseline"]["3"]
    assert liquidity["status"] == "ok"
    assert liquidity["feature"] == "f_log_qv"
    assert liquidity["matching"].startswith("absolute_distance")
    assert liquidity["matched_pairs"] == 24

    volatility = channel["within_volatility_band_lift"]["3"]
    assert volatility["status"] == "ok"
    assert volatility["feature"] == "f_atr_pct_14"
    assert volatility["metrics"]["up10_rate"][
        "difference_selected_minus_baseline"
    ]["ci95"] is not None

    raw = output.read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert json.loads(raw)["report_payload_sha256"] == report["report_payload_sha256"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_boot": 0},
        {"n_boot": evaluator.MAX_BOOTSTRAPS + 1},
        {"seed": -1},
        {"min_rows": 0},
        {"min_days": 0},
        {"top_ns": (0, 3)},
    ],
)
def test_invalid_evaluation_configuration_fails_before_writing(
    tmp_path,
    kwargs,
):
    output = tmp_path / "must-not-exist.json"

    with pytest.raises(ValueError):
        evaluator.evaluate_label_root(
            tmp_path / "labels",
            output_path=output,
            **kwargs,
        )

    assert not output.exists()


@pytest.mark.parametrize("value", ["", "3,nope", "0,3", "-1"])
def test_invalid_top_n_cli_value_is_rejected(value):
    with pytest.raises(argparse.ArgumentTypeError):
        evaluator._parse_top_ns(value)


def test_evaluation_excludes_artifacts_after_explicit_completed_cutoff(
    tmp_path,
):
    root = tmp_path / "labels"
    _write_artifact(root, "2026-07-01")
    future = _write_artifact(root, "2026-07-02")

    report = evaluator.evaluate_label_root(
        root,
        output_path=tmp_path / "evaluation.json",
        top_ns=(3,),
        n_boot=20,
        min_rows=1,
        min_days=1,
        through_date="2026-07-01",
    )

    assert report["artifacts"]["complete_used"] == 1
    assert report["methodology"]["completed_through_date_kst"] == "2026-07-01"
    assert {
        (item["path"], item["reason"])
        for item in report["artifacts"]["skipped"]
    } == {
        (
            str(future),
            "artifact_after_completed_cutoff=2026-07-01",
        )
    }


def test_invalid_evaluation_cutoff_fails_before_writing(tmp_path):
    output = tmp_path / "must-not-exist.json"

    with pytest.raises(ValueError, match="through_date"):
        evaluator.evaluate_label_root(
            tmp_path / "labels",
            output_path=output,
            through_date="not-a-date",
        )

    assert not output.exists()


def test_complete_net_return_is_used_without_second_cost_deduction(tmp_path):
    root = tmp_path / "labels"
    for day in range(1, 6):
        _write_artifact(root, f"2026-07-{day:02d}", with_net=True)

    report = evaluator.evaluate_label_root(
        root,
        output_path=tmp_path / "net.json",
        top_ns=(3,),
        n_boot=100,
        min_rows=20,
        min_days=5,
    )
    channel = report["channels"]["open:R1"]
    basis = channel["return_basis"]
    assert basis == {
        "field": "net_return",
        "is_net": True,
        "cost_adjustment": "none_already_net",
        "n_rows": 100,
        "reason": None,
    }
    selected = channel["top_n_vs_full_universe"]["3"]["metrics"]["return_mean"][
        "selected_day_equal"
    ]
    assert abs(selected - 0.0185) < 1e-12

    eod_net_root = tmp_path / "eod_net_labels"
    for day in range(1, 6):
        _write_artifact(
            eod_net_root, f"2026-07-{day:02d}", with_eod_net=True
        )
    eod_net_report = evaluator.evaluate_label_root(
        eod_net_root,
        output_path=tmp_path / "eod_net.json",
        top_ns=(3,),
        n_boot=100,
        min_rows=20,
        min_days=5,
    )
    eod_net_basis = eod_net_report["channels"]["open:R1"]["return_basis"]
    assert eod_net_basis["field"] == "eod_return_net"
    assert eod_net_basis["cost_adjustment"] == "none_already_net"


def test_missing_features_and_small_sample_are_explicitly_null(tmp_path):
    root = tmp_path / "labels"
    for day in range(1, 3):
        _write_artifact(
            root, f"2026-07-{day:02d}", n_rows=10, features=False
        )

    report = evaluator.evaluate_label_root(
        root,
        output_path=tmp_path / "small.json",
        top_ns=(3,),
        n_boot=50,
        min_rows=30,
        min_days=5,
    )
    channel = report["channels"]["open:R1"]
    head = channel["cohorts"]["all_scores"]["heads"]["p_up10"]
    assert head["status"] == "insufficient"
    assert head["auc"] is None
    assert "n_rows<30" in head["reason"]
    assert "n_days<5" in head["reason"]

    liquidity = channel["liquidity_matched_baseline"]["3"]
    assert liquidity["status"] == "unavailable"
    assert liquidity["metrics"] is None
    assert liquidity["reason"].startswith("liquidity_feature_unavailable")

    volatility = channel["within_volatility_band_lift"]["3"]
    assert volatility["status"] == "unavailable"
    assert volatility["metrics"] is None
    assert volatility["reason"].startswith("atr_feature_unavailable")

    inference = channel["top_n_vs_full_universe"]["3"]["metrics"]["up10_rate"][
        "difference_selected_minus_baseline"
    ]
    assert inference["ci95"] is None
    assert inference["reason"] == "n_days<5"


def test_evaluation_is_deterministic_except_generation_time(tmp_path):
    root = tmp_path / "labels"
    for day in range(1, 7):
        _write_artifact(root, f"2026-07-{day:02d}")

    first = evaluator.evaluate_label_root(
        root,
        output_path=tmp_path / "first.json",
        top_ns=(3,),
        n_boot=100,
        min_rows=20,
        min_days=5,
    )
    second = evaluator.evaluate_label_root(
        root,
        output_path=tmp_path / "second.json",
        top_ns=(3,),
        n_boot=100,
        min_rows=20,
        min_days=5,
    )
    assert first["report_payload_sha256"] == second["report_payload_sha256"]
    assert first["channels"] == second["channels"]


def test_scheduled_replay_is_excluded_from_forward_and_reported_separately(tmp_path):
    root = tmp_path / "labels"
    for day in range(1, 6):
        _write_artifact(root, f"2026-07-{day:02d}")
    _write_artifact(
        root,
        "2026-07-06",
        provenance=SCHEDULED_REPLAY_PROVENANCE_COHORT,
        reverse_up_head=True,
    )
    # Additive provenance가 생기기 전 artifact도 exact execution basis로 안전하게 분류한다.
    _write_artifact(
        root,
        "2026-07-07",
        provenance=SCHEDULED_REPLAY_PROVENANCE_COHORT,
        explicit_provenance=False,
        reverse_up_head=True,
    )

    report = evaluator.evaluate_label_root(
        root,
        output_path=tmp_path / "provenance.json",
        top_ns=(3,),
        n_boot=100,
        min_rows=20,
        min_days=5,
    )

    channel = report["channels"]["open:R1"]
    assert channel["input_n_rows"] == 140
    assert channel["n_rows"] == 100
    assert channel["n_dates"] == 5
    assert channel["cohorts"]["all_scores"]["n_rows"] == 100
    assert channel["cohorts"]["all_scores"]["heads"]["p_up10"]["auc"]["value"] == 1.0
    assert channel["cohorts"]["delivered"]["n_rows"] == 9

    counts = channel["provenance"]["counts"]
    assert counts[FORWARD_PROVENANCE_COHORT]["n_rows"] == 100
    assert counts[SCHEDULED_REPLAY_PROVENANCE_COHORT]["n_rows"] == 40
    assert counts[SCHEDULED_REPLAY_PROVENANCE_COHORT][
        "included_in_default_forward_statistics"
    ] is False
    replay = channel["excluded_provenance_cohorts"][
        SCHEDULED_REPLAY_PROVENANCE_COHORT
    ]
    assert replay["n_rows"] == 40
    assert replay["n_dates"] == 2
    assert replay["cohorts"]["all_scores"]["n_rows"] == 40
    assert replay["cohorts"]["delivered"]["n_rows"] == 3

    assert report["artifacts"]["provenance_artifacts"] == {
        FORWARD_PROVENANCE_COHORT: 5,
        SCHEDULED_REPLAY_PROVENANCE_COHORT: 2,
    }
    inferred = next(
        item for item in report["artifacts"]["used"]
        if item["asof"] == "2026-07-07"
    )
    assert inferred["provenance_source"] == "inferred_from_execution_time_basis"


def test_path_quality_summary_and_complete_only_cohort(tmp_path):
    # flat_filled 행은 경로 의존 지표가 0 쪽으로 축소 편향되므로,
    # 비중을 상시 노출하고 complete-only 코호트를 별도 보고한다.
    root = tmp_path / "labels"
    for day in range(1, 9):
        path = _write_artifact(root, f"2026-07-{day:02d}")
        document = json.loads(path.read_text(encoding="utf-8"))
        document.pop("label_payload_sha256")
        for row in document["rows"]:
            if row["rank"] > 15:
                row["path_quality"] = "flat_filled"
        document["label_payload_sha256"] = _artifact_digest(document)
        path.write_text(
            json.dumps(document, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
    output = tmp_path / "evaluation.json"

    report = evaluator.evaluate_label_root(
        root,
        output_path=output,
        top_ns=(3,),
        n_boot=200,
        min_rows=30,
        min_days=5,
    )

    channel = report["channels"]["open:R1"]
    summary = channel["path_quality"]
    assert summary["status"] == "available"
    assert summary["counts"] == {"complete": 120, "flat_filled": 40}
    assert summary["flat_filled_share"] == 0.25
    assert summary["flat_filled_bars_mean"] is None
    assert "biased toward zero" in summary["bias_note"]

    complete_only = channel["cohorts"]["complete_path_only"]
    assert complete_only["selection"] == "path_quality_complete_only"
    assert complete_only["n_rows"] == 120
    assert complete_only["heads"]["p_up10"]["status"] == "ok"
    assert (
        channel["cohorts"]["all_scores"]["n_rows"]
        == complete_only["n_rows"] + 40
    )


def _channel_rows(*, days: int = 1, n_rows: int = 20) -> list[dict]:
    return [
        {**_row(day, rank), "_date": f"2026-07-{day + 1:02d}",
         "_provenance_cohort": FORWARD_PROVENANCE_COHORT}
        for day in range(days) for rank in range(1, n_rows + 1)
    ]


def _halt(row: dict) -> None:
    row["label_status"] = "halted_no_observations"
    row["path_quality"] = "halted_no_observations"
    row["path_complete"] = False
    for field in (
        "up10", "dn5", "mfe", "mae", "eod_return", "eod_return_net",
        "net_return", "tp5_sl3_first_passage", "tp5_sl3_return_net",
    ):
        if field in row:
            row[field] = None


def _evaluate_rows(rows: list[dict], *, min_days: int = 1) -> dict:
    return evaluator._evaluate_channel(
        rows, top_ns=(3,), n_boot=50, seed=42, min_rows=5, min_days=min_days
    )


FAMILIES = (
    "top_n_vs_full_universe", "liquidity_matched_baseline",
    "within_volatility_band_lift",
)


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_halted_top_pick_is_never_replaced_with_rank_four(rank):
    rows = _channel_rows()
    _halt(rows[rank - 1])
    channel = _evaluate_rows(rows)
    for family in FAMILIES:
        result = channel[family]["3"]
        assert result["status"] == "unavailable"
        assert result["metrics"] is None
        audit = result["selection_audit"][0]
        assert [item["rank"] for item in audit["selected"]] == [1, 2, 3]
        assert audit["unobserved_selected"] == [{
            "coin": f"KRW-T{rank:02d}",
            "label_status": "halted_no_observations",
        }]
        assert "selected_outcome_unavailable" in audit["evaluation_reasons"]
    assert channel["n_rows"] == 19
    assert channel["recorded_n_rows"] == 20
    delivered = channel["cohorts"]["delivered"]
    assert delivered["n_rows"] == 2
    assert delivered["outcome_coverage"]["expected_rows"] == 3
    assert delivered["outcome_coverage"]["observed_rows"] == 2
    assert delivered["outcome_coverage"]["unobserved_rows"] == 1
    assert delivered["outcome_coverage"]["unobserved_fraction"] == pytest.approx(1 / 3)


def test_halted_nearest_control_is_not_rematched():
    rows = _channel_rows()
    _halt(rows[3])  # Rank 4 is the frozen nearest control for rank 1.
    result = _evaluate_rows(rows)["liquidity_matched_baseline"]["3"]
    audit = result["selection_audit"][0]
    assert audit["baseline_groups"] == [["KRW-T04", "KRW-T05", "KRW-T06"]]
    assert result["status"] == "unavailable"
    assert audit["evaluation_reasons"] == ["baseline_outcome_unavailable"]
    assert audit["unobserved_baseline"][0]["coin"] == "KRW-T04"
    assert result["matched_pairs"] == 0


def test_halted_unmatched_control_blocks_only_comparisons_that_need_it():
    rows = _channel_rows()
    _halt(rows[-1])
    channel = _evaluate_rows(rows)
    assert channel["liquidity_matched_baseline"]["3"]["status"] == "ok"
    assert channel["top_n_vs_full_universe"]["3"]["status"] == "unavailable"
    volatility = channel["within_volatility_band_lift"]["3"]
    audit = volatility["selection_audit"][0]
    assert "KRW-T20" in audit["baseline_groups"][1]
    assert audit["evaluation_reasons"] == ["baseline_outcome_unavailable"]
    assert volatility["status"] == "unavailable"


def test_all_halted_date_retains_raw_denominators_and_empty_observations():
    rows = _channel_rows()
    for row in rows:
        _halt(row)
    channel = _evaluate_rows(rows)
    assert channel["recorded_n_rows"] == 20
    assert channel["recorded_n_dates"] == 1
    assert channel["n_rows"] == channel["n_dates"] == 0
    assert channel["outcome_coverage"]["unobserved_fraction"] == 1.0
    assert channel["return_basis"]["field"] is None
    assert channel["cohorts"]["all_scores"]["heads"]["p_up10"]["status"] == "insufficient"
    for family in FAMILIES:
        result = channel[family]["3"]
        assert result["coverage"]["recorded_dates"] == 1
        assert result["coverage"]["planned_dates"] == 1
        assert result["coverage"]["comparable_dates"] == 0
        assert result["n_days"] == 0
        assert result["metrics"] is None


def test_plans_are_outcome_blind_shuffle_stable_and_input_is_immutable():
    rows = _channel_rows(days=3)
    original = copy.deepcopy(rows)
    plan = evaluator._build_comparison_plans(pd.DataFrame(rows), (3, 5))
    changed = copy.deepcopy(rows)
    for row in changed:
        _halt(row)
        row["up10"] = False
        row["dn5"] = True
        row["eod_return"] = -0.99
    changed.reverse()
    assert evaluator._build_comparison_plans(pd.DataFrame(changed), (3, 5)) == plan
    first = _evaluate_rows(rows)
    assert rows == original
    assert _evaluate_rows(list(reversed(rows))) == first


@pytest.mark.parametrize("missing_rank", [1, 4])
@pytest.mark.parametrize("family", FAMILIES)
def test_missing_metric_never_averages_only_remaining_members(family, missing_rank):
    rows = _channel_rows()
    rows[missing_rank - 1]["tp5_sl3_return_net"] = None
    result = _evaluate_rows(rows)[family]["3"]
    metric = result["metrics"]["first_passage_net_mean"]
    assert metric["selected_day_equal"] is None
    assert metric["baseline_day_equal"] is None
    assert metric["coverage"]["comparable_dates"] == 0
    assert metric["coverage"]["unavailable_dates"] == 1
    assert result["metrics"]["up10_rate"]["selected_day_equal"] == 1.0
    assert result["selection_audit"][0]["evaluation_status"] == "comparable"
    assert "first_passage_net_mean" in result["selection_audit"][0]["unavailable_metrics"]


def test_metric_missing_date_is_paired_excluded_not_cross_date_averaged():
    rows = _channel_rows(days=2)
    rows[0]["tp5_sl3_return_net"] = None
    for row in rows[20:]:
        row["tp5_sl3_return_net"] = 0.0123
    result = _evaluate_rows(rows, min_days=2)["top_n_vs_full_universe"]["3"]
    metric = result["metrics"]["first_passage_net_mean"]
    assert metric["selected_day_equal"] == pytest.approx(0.0123)
    assert metric["baseline_day_equal"] == pytest.approx(0.0123)
    assert metric["coverage"]["comparable_dates"] == 1
    delta = metric["difference_selected_minus_baseline"]
    assert delta["ci95"] is None
    assert delta["n_days"] == 1


def test_halt_does_not_downgrade_observed_net_basis_to_gross():
    rows = _channel_rows()
    for row in rows:
        row["eod_return_net"] = row["eod_return"] - 0.0015
    _halt(rows[-1])
    channel = _evaluate_rows(rows)
    assert channel["return_basis"]["field"] == "eod_return_net"
    assert channel["return_basis"]["cost_adjustment"] == "none_already_net"
    metric = channel["liquidity_matched_baseline"]["3"]["metrics"]["return_mean"]
    assert metric["selected_day_equal"] == pytest.approx(0.0185)


def test_delivery_fields_absent_are_unknown_not_zero_expected_picks():
    rows = _channel_rows()
    for row in rows:
        row.pop("delivery_ok")
    delivered = _evaluate_rows(rows)["cohorts"]["delivered"]
    assert delivered["status"] == "unavailable"
    assert delivered["outcome_coverage"]["expected_rows"] is None
    assert delivered["outcome_coverage"]["unobserved_fraction"] is None


def test_version_diagnostics_do_not_filter_or_create_approval():
    rows = _channel_rows(days=2)
    for index, row in enumerate(rows):
        row["_model_identity"] = {
            "model_id": "r1", "model_version": "v1" if index < 20 else "v2",
            "rule_version": "r1-rule", "score_source_sha256": "a" * 64,
        }
    channel = _evaluate_rows(rows)
    versions = channel["model_versions"]
    assert versions["mixed_recorded_identities"] is True
    assert versions["has_unknown_identity_fields"] is False
    assert [group["recorded_rows"] for group in versions["groups"]] == [20, 20]
    assert channel["n_rows"] == 40
    assert "not a fresh-holdout or promotion" in versions["note"]


def test_normal_labeled_comparisons_match_fixed_manual_groups():
    rows = _channel_rows()
    channel = _evaluate_rows(rows)
    expected_groups = {
        FAMILIES[0]: [list(range(1, 21))],
        FAMILIES[1]: [[4, 5, 6]],
        FAMILIES[2]: [[4, 7, 10, 13, 16, 19], [5, 8, 11, 14, 17, 20], [6, 9, 12, 15, 18]],
    }
    def mean(values):
        return sum(values) / len(values)
    for family, groups in expected_groups.items():
        result = channel[family]["3"]
        assert result["selection_audit"][0]["baseline_groups"] == [
            [f"KRW-T{rank:02d}" for rank in group] for group in groups
        ]
        expected = mean([mean([float(rows[rank - 1]["up10"]) for rank in group]) for group in groups])
        assert result["metrics"]["up10_rate"]["baseline_day_equal"] == pytest.approx(expected)
        assert result["metrics"]["up10_rate"]["selected_day_equal"] == 1.0


def _modern_halted_artifact(tmp_path: Path, *, all_halted: bool) -> Path:
    """Bound synthetic snapshot/label; injected scorers and paths, no DB or HTTP."""
    def scorer(asof, *, slot, ranking, limit_markets):
        candidates = []
        for rank in range(1, 5):
            candidates.append({
                "coin": f"KRW-T{rank:02d}", "rank": rank, "score": 0.5,
                "pump_prob": 0.02, "pump_prob_pct": "2.0%", "rr_ratio": 1.0,
                "p_up5": 0.3, "p_up10": 0.1, "p_up20": 0.02,
                "p_dn5": 0.1, "p_dn10": 0.03, "exp_downside": -0.02,
                "dump_risk_flag": False, "entry_open": 100.0,
                "sl": -0.03, "tp": 0.05, "btc_regime": "neutral",
                "feature_values": {"f_log_qv": 10.0 + rank / 10},
            })
        return {
            "asof": asof, "slot": slot, "ranking": ranking, "feature_date": asof,
            "btc_regime": "neutral", "universe_n": 4,
            "calibration_source": "bucket_score_pump20",
            "rank_basis": "R1_riskreward(de-corr head)", "n_history_dates": 100,
            "score_schema_version": "recommend_score.v2",
            "rule_version": "r1_riskreward_v1", "model_random_seed": 42,
            "feature_columns": ["f_log_qv"],
            "training": {
                "start": "2025-01-01", "end": "2026-07-18",
                "cutoff_exclusive": "2026-07-19", "embargo_days": 5,
                "rows": 1000, "dates": 100,
            },
            "universe": candidates, "top3": candidates[:3],
        }
    snapshot = get_or_create_recommend_snapshot(
        "2026-07-24", slot="open", root=tmp_path / "snapshots", scorer=scorer
    )
    def assessor(coin, start_at, **_kwargs):
        halted = all_halted or coin == "KRW-T01"
        return PathAssessment(
            bars=[] if halted else [(100.0, 101.0, 99.0, 100.0)] * 96,
            timestamps=tuple(pd.date_range(pd.Timestamp(start_at).tz_localize(None), periods=96, freq="15min")),
            path_complete=not halted,
            path_quality="target_no_observations" if halted else "complete",
            raw_bars=0 if halted else 96, expected_bars=96,
            flat_filled_bars=0, benchmark_bars=96,
        )
    result = label_recommend_snapshot(
        snapshot["snapshot_path"], output_root=tmp_path / "labels",
        receipt_root=tmp_path / "receipts", db_path=tmp_path / "never-created.db",
        now="2026-07-26 10:00:00", assessor=assessor, halt_prober=lambda _coin: True,
    )
    assert not (tmp_path / "never-created.db").exists()
    path = Path(result["artifact_path"])
    load_label_artifact(path)
    return path


@pytest.mark.parametrize("all_halted", [False, True])
def test_loader_retains_validated_modern_halt_rows_and_all_halt_artifact(tmp_path, all_halted):
    path = _modern_halted_artifact(tmp_path, all_halted=all_halted)
    before = path.read_bytes()
    channels, audit = evaluator._load_complete_rows(
        tmp_path / "labels", through_date=evaluator.calendar_date(2026, 7, 25)
    )
    assert len(channels[("open", "R1")]) == 4
    assert audit["complete_used"] == 1
    assert audit["used"][0]["recorded_rows"] == 4
    assert audit["used"][0]["rows"] == (0 if all_halted else 3)
    assert audit["used"][0]["unobserved_rows"] == (4 if all_halted else 1)
    report = evaluator.evaluate_label_root(
        tmp_path / "labels", output_path=tmp_path / "new-report.json",
        top_ns=(3,), n_boot=50, min_rows=5, min_days=1,
        through_date="2026-07-25",
    )
    assert report["schema"] == "recommend_score_label_evaluation.v3"
    assert report["channels"]["open:R1"]["input_recorded_n_rows"] == 4
    assert path.read_bytes() == before
