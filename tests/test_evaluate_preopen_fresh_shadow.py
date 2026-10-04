"""scripts/evaluate_preopen_fresh_shadow.py — 평가 산술 (합성 데이터만)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.evaluate_preopen_fresh_shadow as ev
from data.database import init_db

ENTRY = pd.Timestamp("2026-10-05 09:00:00")


def _bars(prices: list[tuple[float, float, float]], start=ENTRY) -> pd.DataFrame:
    """(open, high, low) 목록 → 15분봉 DataFrame."""
    return pd.DataFrame({
        "timestamp": [start + i * ev.BAR for i in range(len(prices))],
        "open": [p[0] for p in prices],
        "high": [p[1] for p in prices],
        "low": [p[2] for p in prices],
    })


def test_path_outcome_thresholds_inside_96_bar_window():
    prices = [(100.0, 101.0, 99.0)] * 96
    prices[10] = (100.0, 110.0, 99.0)     # +10% 정확히 도달
    prices[20] = (100.0, 101.0, 95.0)     # −5% 정확히 도달
    prices += [(100.0, 150.0, 50.0)]      # 97번째 봉 — 창 밖
    out = ev.path_outcome(_bars(prices), ENTRY)
    assert out["hit10"] == 1.0 and out["hit5"] == 1.0 and out["dn5"] == 1.0
    assert out["n_bars"] == 96.0
    assert out["max_up"] == pytest.approx(0.10) and out["max_dn"] == pytest.approx(-0.05)


def test_path_outcome_misses_and_missing_entry_bar():
    flat = ev.path_outcome(_bars([(100.0, 104.9, 95.1)] * 96), ENTRY)
    assert (flat["hit10"], flat["hit5"], flat["dn5"]) == (0.0, 0.0, 0.0)
    late = _bars([(100.0, 120.0, 90.0)] * 10, start=ENTRY + ev.BAR)  # 09:00 봉 없음
    assert ev.path_outcome(late, ENTRY) is None
    assert ev.path_outcome(pd.DataFrame(), ENTRY) is None


def test_overlap_summary_arithmetic():
    a1 = {"top3": ["A", "B", "C"], "ranks": {f"X{i}": i for i in range(1, 11)}}
    b1 = {"top3": ["A", "B", "C"], "ranks": {"A": 1, "B": 2, "C": 3}}
    a2 = {"top3": ["A", "D", "E"], "ranks": {}}
    b2 = {"top3": ["A", "B", "C"], "ranks": {"A": 1, "D": 12}}
    out = ev.overlap_summary([(a1, b1), (a2, b2)])
    assert out["n_days"] == 2
    assert out["mean_overlap_of_3"] == 2.0          # (3 + 1) / 2
    assert out["share_3of3"] == 0.5 and out["share_0of3"] == 0.0
    assert out["overlap_dist"] == {"0": 0, "1": 1, "2": 0, "3": 1}
    # a 의 Top3 가 b 에서 받은 순위: 1,2,3,1,12 (+E 없음)
    assert out["a_top3_rank_in_b_median"] == 2.0
    assert out["a_top3_in_b_top10"] == 0.8 and out["a_top3_missing_in_b"] == 1
    assert ev.overlap_summary([]) == {"n_days": 0}


def test_rank_spearman_perfect_and_reversed():
    ra = {f"C{i}": i for i in range(20)}
    assert ev.rank_spearman(ra, dict(ra)) == pytest.approx(1.0)
    assert ev.rank_spearman(ra, {c: -r for c, r in ra.items()}) == pytest.approx(-1.0)
    assert np.isnan(ev.rank_spearman({"A": 1}, {"A": 1}))


def test_paired_bootstrap_constant_and_mixed_difference():
    const = ev.paired_bootstrap(np.array([0.5, 0.5, 0.5]), np.array([0.2, 0.2, 0.2]))
    assert const == {"n_days": 3, "mean_diff": 0.3, "ci95": [0.3, 0.3]}
    mixed = ev.paired_bootstrap(np.array([1.0, 0.0] * 20), np.zeros(40))
    assert mixed["mean_diff"] == 0.5
    assert mixed["ci95"][0] < 0.5 < mixed["ci95"][1]
    assert ev.paired_bootstrap(np.array([]), np.array([]))["n_days"] == 0


def _write_snapshot(root: Path, day: str, name: str, coins: list[str]) -> None:
    path = root / day / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "top3": [{"coin": c, "rank": i + 1} for i, c in enumerate(coins)],
        "universe": [{"coin": c, "rank": i + 1} for i, c in enumerate(coins)],
    }))


def _write_shadow(root: Path, day: str, coins: list[str], status: str = "ok",
                  rank_basis: str = "R1_riskreward(de-corr head)",
                  health_gate: str | None = "passed") -> None:
    root.mkdir(parents=True, exist_ok=True)
    cron = {"ok": True} if health_gate is None else {"ok": True, "health_gate": health_gate}
    (root / f"{day}.json").write_text(json.dumps({
        "asof": day, "status": status, "freshness": {"cron_log": cron},
        "result": {"rank_basis": rank_basis,
                   "top3": [{"coin": c, "rank": i + 1} for i, c in enumerate(coins)],
                   "candidates": [[c, i + 1, 1.0, 0.2, 0.2] for i, c in enumerate(coins)]}
        if status in ("ok", "degraded") else None,
    }))


def _m15_db(path: Path, days: list[str], paths: dict[str, tuple[float, float]]) -> Path:
    """coin → (그날 최고가, 최저가) — 진입 100 기준 96봉 경로. 마지막 날 다음 09:00 봉까지."""
    init_db(path)
    con = sqlite3.connect(path)
    with con:
        for day in days:
            start = pd.Timestamp(day) + pd.Timedelta(hours=9)
            for coin, (hi, lo) in paths.items():
                for i in range(96):
                    ts = start + i * ev.BAR
                    h, low = (hi, lo) if i == 50 else (100.5, 99.5)
                    con.execute("INSERT INTO candles VALUES (?, ?, 100, ?, ?, 100, 1, 1)",
                                (coin, str(ts), h, low))
        con.execute("INSERT INTO candles VALUES ('KRW-ZZZ', ?, 1, 1, 1, 1, 1, 1)",
                    (str(pd.Timestamp(days[-1]) + pd.Timedelta(days=1, hours=9)),))
    con.close()
    return path


def test_evaluate_end_to_end_on_synthetic_days(tmp_path):
    shadow_root, snaps = tmp_path / "shadow", tmp_path / "snaps"
    days = ["2026-10-05", "2026-10-06"]
    for day in days:
        _write_shadow(shadow_root, day, ["KRW-UP", "KRW-FLAT", "KRW-DN"])
        _write_snapshot(snaps, day, "open_r1.json", ["KRW-UP", "KRW-FLAT", "KRW-X"])
        _write_snapshot(snaps, day, "preopen_r1.json", ["KRW-DN", "KRW-X", "KRW-FLAT"])
    _write_shadow(shadow_root, "2026-10-07", [], status="stale_input")  # ok 아님 → 제외
    # 대체 정렬(fallback) — degraded 든, degraded 도입 전 ok 기록이든 제외.
    _write_shadow(shadow_root, "2026-10-08", ["KRW-UP"], status="degraded",
                  rank_basis="score_rank(fallback)")
    _write_shadow(shadow_root, "2026-10-09", ["KRW-UP"], rank_basis="score_rank(fallback)")
    # R1 preopen D1 health gate 실패·미확인·필드 없음 — ok 기록이어도 제외.
    _write_shadow(shadow_root, "2026-10-10", ["KRW-UP"], health_gate="failed")
    _write_shadow(shadow_root, "2026-10-11", ["KRW-UP"], health_gate="unknown")
    _write_shadow(shadow_root, "2026-10-12", ["KRW-UP"], health_gate=None)
    db = _m15_db(tmp_path / "m15.db", days, {
        "KRW-UP": (111.0, 99.0), "KRW-FLAT": (101.0, 99.0),
        "KRW-DN": (101.0, 94.0), "KRW-X": (106.0, 99.0)})

    out = ev.evaluate(shadow_root, snaps, db)

    assert out["shadow_days_ok"] == 2
    assert set(ev.load_shadow(shadow_root)) == set(days)
    assert out["shadow_excluded"] == {
        "n_days": 6,
        "by_reason": {"health_gate=failed": 1, "health_gate=missing": 1,
                      "health_gate=unknown": 1, "non-R1 rank_basis": 1,
                      "status=degraded": 1, "status=stale_input": 1},
        "days": {"2026-10-07": "status=stale_input", "2026-10-08": "status=degraded",
                 "2026-10-09": "non-R1 rank_basis", "2026-10-10": "health_gate=failed",
                 "2026-10-11": "health_gate=unknown", "2026-10-12": "health_gate=missing"},
    }
    # 비용 미반영(gross) 진단치임을 보고서에 명시.
    assert out["cost_basis"] == "gross" and "비용 미반영(gross)" in out["note"]
    assert out["definition"]["costs"].startswith("none (gross")
    sv = out["overlap"]["shadow_vs_open0905"]
    assert sv["n_days"] == 2 and sv["mean_overlap_of_3"] == 2.0
    oc = out["outcomes"]
    assert oc["n_days_complete"] == 2 and oc["incomplete_days"] == []
    # shadow: UP(+10,+5), FLAT(-), DN(−5) → hit10 1/3, hit5 1/3, dn5 1/3
    assert oc["arms"]["shadow"] == {"n_days": 2, "hit10": 0.3333, "hit5": 0.3333,
                                    "dn5": 0.3333}
    # open: UP, FLAT, X(+5) → hit10 1/3, hit5 2/3, dn5 0
    assert oc["arms"]["open0905"] == {"n_days": 2, "hit10": 0.3333, "hit5": 0.6667,
                                      "dn5": 0.0}
    diff = oc["diff"]["shadow-open0905"]
    assert diff["hit5"]["mean_diff"] == pytest.approx(-0.3333, abs=1e-4)
    assert diff["dn5"]["ci95"] == [pytest.approx(0.3333, abs=1e-4)] * 2
    assert diff["hit10"]["n_days"] == 2


def test_evaluate_marks_days_without_full_96_bar_path_incomplete(tmp_path):
    shadow_root, snaps = tmp_path / "shadow", tmp_path / "snaps"
    _write_shadow(shadow_root, "2026-10-05", ["KRW-UP"])
    db = tmp_path / "m15.db"
    init_db(db)  # 빈 15m DB
    out = ev.evaluate(shadow_root, snaps, db)
    assert out["outcomes"]["incomplete_days"] == ["2026-10-05"]
    assert out["outcomes"]["n_days_complete"] == 0
    assert ev.evaluate(shadow_root, snaps, tmp_path / "missing.db")["outcomes"] == {
        "skipped": "15m DB unavailable"}
