from __future__ import annotations

import json
import argparse
from pathlib import Path

import pytest

import scripts.evaluate_recommend_score_labels as evaluator
from signals.recommend_score_labels import (
    FORWARD_PROVENANCE_COHORT,
    LABEL_SCHEMA_VERSION,
    SCHEDULED_REPLAY_PROVENANCE_COHORT,
    _artifact_digest,
)


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
