from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from signals.recommend_score_alignment import align_scores, evaluate_alignment, transform_predictions
from test_recommend_experiment_eval import _predictions


def frame():
    data = _predictions(slots=("open", "preopen"))
    data["C_status"] = "ok"
    return data


def test_known_mapping_and_ties_preserve_mean_not_arbitrary_coin_order():
    assert align_scores([.3, .1, .2], [.8, .2, .5]).tolist() == [.8, .2, .5]
    result = align_scores([.5, .1, .5, .8], [1., 0., .2, .4])
    np.testing.assert_allclose(result, [.3, 0., .3, 1.])
    assert result.mean() == pytest.approx(.4)
    assert align_scores([.5] * 4, [0., .2, .4, 1.]).tolist() == [.4] * 4


@pytest.mark.parametrize("values", [[0., 1.], [.2757] * 100, [0., 0., .2, .2, .8, 1.], [.09999999999999998]])
def test_self_alignment_is_bit_exact(values):
    np.testing.assert_array_equal(align_scores(values, values), values)


def test_mean_order_and_permutation_properties():
    rng = np.random.default_rng(42)
    for size in (1, 3, 100):
        for _ in range(20):
            raw = np.round(rng.random(size), 1)
            reference = rng.random(size)
            result = align_scores(raw, reference)
            assert result.mean() == pytest.approx(reference.mean(), abs=1e-15)
            assert np.all(np.diff(result[np.argsort(raw)]) >= 0)
            for value in set(raw):
                assert len(set(result[raw == value])) == 1
            order = rng.permutation(size)
            np.testing.assert_array_equal(align_scores(raw[order], reference[order]), result[order])


@pytest.mark.parametrize("bad", [[], [None], [float("nan")], [float("inf")], [True], ["0.5"], [-.1], [1.1], [[.5]]])
def test_invalid_score_inputs_do_not_silently_impute(bad):
    with pytest.raises(ValueError):
        align_scores(bad, [.2])
    with pytest.raises(ValueError):
        align_scores([.2], bad)


def test_population_size_must_match():
    with pytest.raises(ValueError, match="sizes"):
        align_scores([.1, .2], [.3])


def test_frame_transform_preserves_source_and_does_not_read_outcomes():
    original = frame()
    before = original.copy(deep=True)
    transformed, audit = transform_predictions(original)
    pd.testing.assert_frame_equal(original, before)
    pd.testing.assert_series_equal(transformed.p_up10_B, original.p_up10_C, check_names=False)
    for field in ("p_dn5", "p_dn10", "rank", "exp_downside", "mfe", "mae", "eod_return_net"):
        pd.testing.assert_series_equal(transformed[field], original[field])
    assert len(audit) == 12
    assert all(a['reference_mean'] == pytest.approx(a['aligned_mean']) for a in audit)
    changed = original.drop(columns=["up10", "dn5", "mfe", "mae", "eod_return_net", "tp5_sl3_return_net", "label_status"])
    alternative, second_audit = transform_predictions(changed)
    pd.testing.assert_series_equal(transformed.p_up10_C, alternative.p_up10_C)
    assert audit == second_audit


def test_future_dates_other_slot_and_row_order_cannot_change_current_mapping():
    original = frame()
    transformed, _ = transform_predictions(original)
    modified = original.copy()
    modified.loc[modified.date.ne("2026-09-01") | modified.slot.ne("open"), "p_up10_C"] = .1
    alternative, _ = transform_predictions(modified.sample(frac=1, random_state=42))
    left = transformed[(transformed.date == "2026-09-01") & (transformed.slot == "open")].sort_values("coin")
    right = alternative[(alternative.date == "2026-09-01") & (alternative.slot == "open")].sort_values("coin")
    np.testing.assert_array_equal(left.p_up10_C, right.p_up10_C)


def test_unavailable_and_halted_candidates_remain_in_population():
    original = frame()
    unavailable = original.fold.eq(2) & original.slot.eq("open")
    original.loc[unavailable, "p_up10_C"] = np.nan
    original.loc[unavailable, "C_status"] = "unavailable"
    original.loc[original.coin.eq("KRW-C5"), "label_status"] = "halted_no_observations"
    result, audit = transform_predictions(original)
    assert len(result) == len(original)
    assert result.loc[unavailable, ["p_up10_B", "p_up10_C"]].isna().all().all()
    assert sum(a["status"] == "unavailable" for a in audit) == 3
    assert result.loc[~unavailable, "p_up10_C"].notna().all()


@pytest.mark.parametrize("mutation", ["duplicate", "missing_rank", "mixed_snapshot", "mixed_status", "nan_ok", "finite_unavailable"])
def test_broken_candidate_contract_blocks_transform(mutation):
    original = frame()
    if mutation == "duplicate":
        original = pd.concat([original, original.iloc[:1]])
    elif mutation == "missing_rank":
        original = original.drop(index=0)
    elif mutation == "mixed_snapshot":
        original.at[0, "snapshot_id"] = "other"
    elif mutation == "mixed_status":
        original.at[0, "C_status"] = "unavailable"
    elif mutation == "nan_ok":
        original.at[0, "p_up10_C"] = np.nan
    else:
        original["C_status"] = "unavailable"
    with pytest.raises(ValueError):
        transform_predictions(original)


def test_ablation_metrics_are_net_once_and_explicit_about_arm_meanings():
    original = frame()
    transformed, _ = transform_predictions(original)
    report = evaluate_alignment(transformed, n_boot=30)
    assert report["arm_mapping"] == {"A": "original_R1", "B": "original_upper_C", "C": "aligned_upper_D"}
    assert "monotonic_B" not in report["methodology"]
    for slot, summary in report["slots"].items():
        assert "probability_diagnostics" not in summary
        assert "brier" not in summary["score_diagnostics"]["C"]["selected_top3"]["up10"]
        records = [r for r in report['daily'] if r['slot'] == slot]
        for metric in ("eod_return_net", "up10", "dn5"):
            expected = np.mean([r['arms']['C'][metric] - r['arms']['B'][metric] for r in records])
            assert summary['aligned_D_minus_original_C']['mean'][metric] == pytest.approx(expected)
        selected = transformed[(transformed.slot == slot) & transformed.was_delivered]
        assert summary['absolute']['A']['mean']['eod_return_net'] == pytest.approx(selected.eod_return_net.mean())


def test_label_change_cannot_change_selected_coins_or_matching_pools():
    transformed, _ = transform_predictions(frame())
    original = evaluate_alignment(transformed, n_boot=10)
    changed = deepcopy(transformed)
    changed.loc[0, 'up10'], changed.loc[0, 'mfe'] = 0., .02
    changed.loc[0, 'dn5'], changed.loc[0, 'mae'] = 0., -.01
    changed.loc[0, ['eod_return_net', 'tp5_sl3_return_net']] = -.0115
    alternative = evaluate_alignment(changed, n_boot=10)
    assert original['selection_audit'] == alternative['selection_audit']
