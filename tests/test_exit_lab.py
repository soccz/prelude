"""exit_lab 검증 — (1) simulate_path 와의 동등성 고정, (2) 변형별 semantics.

동등성 테스트가 깨지면 ledger/exit_lab.walk_path 와
scripts/recommender_downside_exit_v1.simulate_path 중 한쪽이 바뀐 것 —
둘을 다시 sync 하라 (같은 봉 SL 우선 보수 규약 공유).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ledger.exit_lab import (  # noqa: E402
    EXIT_LAB_COLS,
    EXIT_VARIANTS,
    evaluate_exit_variants,
    walk_path,
)


def _bar(o, h, l, c):
    return (float(o), float(h), float(l), float(c))


# 100 진입 기준 시나리오들: (bars, 설명)
SCENARIOS = [
    ([_bar(100, 101, 99, 100.5), _bar(100.5, 106, 100, 105)], "2번째 봉 TP5 도달"),
    ([_bar(100, 100.5, 96.5, 97)], "첫 봉 SL3 터치"),
    ([_bar(100, 104, 96.9, 103), _bar(103, 112, 102, 110)], "동시구간: SL 우선 보수"),
    ([_bar(100, 101, 99.5, 100.2), _bar(100.2, 102, 99.8, 101)], "아무것도 안 닿음 → EOD"),
    ([_bar(100, 111, 95, 108)], "한 봉에서 SL·TP10 모두 — SL 먼저"),
    ([_bar(100, 102, 98, 101), _bar(101, 109, 100, 102), _bar(102, 115, 101, 96)], "3봉 경로"),
]


def test_walk_path_equivalence_with_simulate_path():
    """walk_path ≡ simulate_path(trail=None) — 모든 시나리오 × 파라미터 격자."""
    from scripts.recommender_downside_exit_v1 import simulate_path

    grid = [(0.03, 0.05), (None, 0.05), (0.03, None), (None, None), (0.05, 0.10)]
    for bars, desc in SCENARIOS:
        for sl, tp in grid:
            g1, o1 = walk_path(bars, sl, tp)
            g2, o2 = simulate_path(bars, sl, tp, None)
            assert o1 == o2, f"{desc} sl={sl} tp={tp}: outcome {o1} != {o2}"
            if np.isfinite(g1) or np.isfinite(g2):
                assert g1 == pytest.approx(g2), f"{desc} sl={sl} tp={tp}: gross {g1} != {g2}"


def test_variants_diverge_on_dump_after_pump():
    """펌프 후 덤프 경로: TP5/SL3 는 TP, eod 는 덤프 손실 — 변형이 실제로 갈라지는지."""
    bars = [_bar(100, 103, 99, 102), _bar(102, 107, 101, 106), _bar(106, 108, 90, 92)]
    out = evaluate_exit_variants(bars, round_trip_cost=0.0015)
    assert out is not None
    # tp5_nosl: 2번째 봉 high 107 ≥ 105 → TP (+5% - 0.15%)
    assert out["exit_tp5_nosl_reason"] == "TP"
    assert out["exit_tp5_nosl_pct"] == pytest.approx((0.05 - 0.0015) * 100, abs=1e-6)
    # tp10_nosl: max high 108 < 110 → EOD (92/100 - 1 = -8% - 0.15%)
    assert out["exit_tp10_nosl_reason"] == "EOD"
    assert out["exit_tp10_nosl_pct"] == pytest.approx((-0.08 - 0.0015) * 100, abs=1e-6)
    # eod 동일 경로
    assert out["exit_eod_pct"] == pytest.approx((-0.08 - 0.0015) * 100, abs=1e-6)
    # path 극값
    assert out["path_max_pct"] == pytest.approx(8.0, abs=1e-6)
    assert out["path_min_pct"] == pytest.approx(-10.0, abs=1e-6)


def test_variants_capture_late_pump_that_sl_cuts():
    """핵심 가설 재현: 먼저 -3% 살짝 찍고 오후에 +12% 펌프.
    TP5/SL3 는 -3% 손절, TP10/noSL 은 +10% 익절 — 잣대 차이가 보여야 함."""
    bars = [
        _bar(100, 100.5, 96.8, 97.5),    # 아침: low 96.8 → SL3 터치
        _bar(97.5, 104, 97, 103),
        _bar(103, 112.5, 102, 111),       # 오후: +12.5% 펌프
    ]
    out = evaluate_exit_variants(bars, round_trip_cost=0.0015)
    assert out is not None
    assert out["exit_tp10_nosl_reason"] == "TP"
    assert out["exit_tp10_nosl_pct"] == pytest.approx((0.10 - 0.0015) * 100, abs=1e-6)
    # 운영 기본 (TP5/SL3) 은 walk_path 로 직접 확인 — SL 에 잘림
    gross, outcome = walk_path(bars, 0.03, 0.05)
    assert outcome == "sl" and gross == pytest.approx(-0.03)


def test_evaluate_returns_none_on_bad_input():
    assert evaluate_exit_variants([]) is None
    assert evaluate_exit_variants([_bar(0, 0, 0, 0)]) is None


def test_exit_lab_cols_match_evaluate_keys():
    """EXIT_LAB_COLS 와 evaluate 산출 key 가 1:1 — ledger 스키마 drift 방지."""
    bars = [_bar(100, 101, 99, 100.5)]
    out = evaluate_exit_variants(bars)
    assert out is not None
    assert set(out.keys()) == set(EXIT_LAB_COLS)
    assert set(EXIT_VARIANTS.keys()) == {
        "tp10_nosl", "tp5_nosl", "eod", "tp5_sl2", "tp10_sl5", "tp8_sl3",
    }


def test_bracket_variants_asymmetry():
    """비대칭 bracket semantics: 같은 경로에서 SL 폭이 결과를 가른다.
    경로: -2.5% 터치 후 +9% 펌프 — SL2 는 -2% 절단, SL3/SL5 는 생존."""
    bars = [
        _bar(100, 100.8, 97.5, 98),      # low 97.5 → SL2 (-2%) 만 터치
        _bar(98, 105, 97.6, 104),
        _bar(104, 109.2, 103, 108),       # high 109.2 → TP8 도달, TP10 미달
    ]
    out = evaluate_exit_variants(bars, round_trip_cost=0.0015)
    assert out is not None
    assert out["exit_tp5_sl2_reason"] == "SL"
    assert out["exit_tp5_sl2_pct"] == pytest.approx((-0.02 - 0.0015) * 100, abs=1e-6)
    assert out["exit_tp8_sl3_reason"] == "TP"
    assert out["exit_tp8_sl3_pct"] == pytest.approx((0.08 - 0.0015) * 100, abs=1e-6)
    # tp10_sl5: SL5 미터치 (-2.5% 가 최저), TP10 미도달 → EOD (108/100-1 = +8%)
    assert out["exit_tp10_sl5_reason"] == "EOD"
    assert out["exit_tp10_sl5_pct"] == pytest.approx((0.08 - 0.0015) * 100, abs=1e-6)
