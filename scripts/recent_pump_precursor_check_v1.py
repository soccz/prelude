"""Recent-pump precursor coverage check (Angle A 보강).

이번주 실제 급등 코인들이 '급등 전날(D-1) feature' 기준으로 top-lift 신호의
깃발을 미리 세웠는지 확인. leak-free 보장: D 의 펌프를 D-1 feature 의 그날
cross-section 분위로만 평가(라벨 D 는 결과 표기용).

사용:
    python scripts/recent_pump_precursor_check_v1.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# build_panel/add_cross_sectional 재사용 (self-contained import, prelude 내부)
from scripts.univariate_precursor_lift_v1 import build_panel, add_cross_sectional

# 이번주 급등 케이스 (market, pump_date) — pump_date = 급등 발생 day D
CASES = [
    ("KRW-ID",   "2026-05-29"), ("KRW-ID",   "2026-05-30"),
    ("KRW-GMT",  "2026-05-30"), ("KRW-VTHO", "2026-05-30"),
    ("KRW-META", "2026-05-30"), ("KRW-ERA",  "2026-05-30"),
    ("KRW-PROVE","2026-05-30"), ("KRW-ALT",  "2026-05-30"),
    ("KRW-PRL",  "2026-05-30"), ("KRW-XLM",  "2026-05-28"),
    ("KRW-XLM",  "2026-05-29"), ("KRW-WLD",  "2026-05-30"),
]

# top OOS-lift precursors (전수 스크린 결과) — D-1 feature, 그날 cross-section 분위로 평가
# (feature, direction, decile_threshold). high=상위10%, low=하위10%
PRECURSORS = [
    ("f_qv_surge_30d", "high", 0.90),
    ("f_bounce_off_7d_low", "high", 0.90),
    ("f_ret_3d", "high", 0.90),
    ("f_ret_7d", "high", 0.90),
    ("f_qv_surge_7d", "high", 0.90),
    ("f_rv_7d", "high", 0.90),
    ("f_atr_pct_14", "high", 0.90),
    ("f_range_contraction_7d", "low", 0.10),  # dir=low = range expansion
    ("f_qv_rank", "low", 0.10),               # 거래대금 상위(rank 작음)
]


def main():
    panel = build_panel(None)
    panel = add_cross_sectional(panel)
    panel["date"] = pd.to_datetime(panel["timestamp"]).dt.date.astype(str)

    # 각 feature 의 그날 cross-section 분위 (D-1 feature → D panel 내 rank, leak 아님)
    for feat, _, _ in PRECURSORS:
        panel[f"{feat}__xs"] = panel.groupby("date")[feat].rank(pct=True)

    print("=== 이번주 급등 코인의 급등 전날(D-1) 선행신호 깃발 여부 ===")
    print("(분위는 그날 살아있던 전체 코인 cross-section 기준, 입력 feature 는 D-1 값)\n")
    hit_counts = []
    for market, pdate in CASES:
        row = panel[(panel["market"] == market) & (panel["date"] == pdate)]
        if len(row) == 0:
            print(f"{market:11s} {pdate}: panel 에 없음(데이터 부재)")
            continue
        r = row.iloc[0]
        intraday = r["intraday_high_ret"]
        flags = []
        n_hit = 0
        for feat, direction, thr in PRECURSORS:
            q = r.get(f"{feat}__xs", np.nan)
            if pd.isna(q):
                flags.append(f"{feat.replace('f_',''):>18}=NA")
                continue
            fired = (q >= thr) if direction == "high" else (q <= thr)
            n_hit += int(fired)
            mark = "FLAG" if fired else "  . "
            flags.append(f"{feat.replace('f_',''):>18}[{mark} q={q:.2f}]")
        hit_counts.append(n_hit)
        print(f"{market:11s} {pdate} (high/open={intraday:+.1%}) — fired {n_hit}/{len(PRECURSORS)}")
        for f in flags:
            print(f"      {f}")
        print()
    if hit_counts:
        print(f"평균 발화 신호 수: {np.mean(hit_counts):.1f}/{len(PRECURSORS)} "
              f"(케이스 {len(hit_counts)}개)")
        print(f"1개 이상 발화: {sum(1 for x in hit_counts if x>=1)}/{len(hit_counts)} 케이스")
        print(f"3개 이상 발화: {sum(1 for x in hit_counts if x>=3)}/{len(hit_counts)} 케이스")


if __name__ == "__main__":
    main()
