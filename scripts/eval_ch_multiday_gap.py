#!/usr/bin/env python3
"""Track B3 gap-risk audit: sim 의 SL/trail 체결가정이 일봉 갭에서 낙관인지 검증.

핵심 의심: sim_bracket 은 lo<=sl_px 면 정확히 -sl 로 청산, sim_trail 은 trail_px 로 청산.
하지만 현실에서 다음봉이 stop 아래로 '갭다운 open' 하면 체결가는 open(더 나쁨)이다.
=> bracket/trail 의 worst 가 -SL band 에 정확히 capped 되는 건 낙관 편향일 수 있다.
이걸 일봉 데이터로 직접 측정: stop 트리거된 봉의 open 이 stop_px 보다 아래인 빈도/심도.
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
D1_DB = str(ROOT / "data" / "upbit_d1.db")


def main():
    conn = sqlite3.connect(D1_DB)
    df = pd.read_sql("SELECT market, timestamp, open, high, low, close FROM candles", conn)
    conn.close()
    df["ts"] = pd.to_datetime(df["timestamp"])
    df["bar_date"] = df["ts"].dt.date
    df = df.sort_values(["market", "ts"]).reset_index(drop=True)
    groups = {m: g.reset_index(drop=True) for m, g in df.groupby("market")}

    picks = pd.read_csv(OUT / "r2_challenger_picks_v1.csv")
    picks = picks[picks.policy == "R1_ratio"].copy()
    picks["entry_date"] = pd.to_datetime(picks["date"]).dt.date

    # Replicate trail dd=0.08, N=5: when trail triggers, compare assumed fill (trail_px)
    # vs realistic fill (that bar's OPEN if it gapped below trail_px).
    DD, N = 0.08, 5
    assumed, realistic, gap_cases = [], [], 0
    n_trig = 0
    for _, r in picks.iterrows():
        g = groups.get(r["market"])
        if g is None:
            continue
        sub = g[g["bar_date"] >= r["entry_date"]].iloc[:N]
        if len(sub) == 0 or sub.iloc[0]["bar_date"] != r["entry_date"]:
            continue
        entry_open = float(sub.iloc[0]["open"])
        if entry_open <= 0:
            continue
        bars = list(zip(sub["high"].astype(float), sub["low"].astype(float),
                        sub["close"].astype(float), sub["open"].astype(float)))
        run_high = entry_open
        triggered = False
        for (hi, lo, cl, op) in bars:
            trail_px = run_high * (1.0 - DD)
            if lo <= trail_px:
                # sim assumes fill at trail_px; realistic = min(trail_px, this bar open)
                fill_sim = trail_px
                fill_real = min(trail_px, op)  # if gapped open below stop, fill at open
                assumed.append(fill_sim / entry_open - 1.0)
                realistic.append(fill_real / entry_open - 1.0)
                if op < trail_px:
                    gap_cases += 1
                triggered = True
                n_trig += 1
                break
            run_high = max(run_high, hi)
        if not triggered:
            ret = bars[-1][2] / entry_open - 1.0
            assumed.append(ret)
            realistic.append(ret)

    assumed = np.array(assumed) - 0.0015
    realistic = np.array(realistic) - 0.0015
    print("=== trail dd0.08 N5: sim 체결가정 vs 갭반영 현실 체결 ===")
    print(f"  trades={len(assumed)}  trail triggered={n_trig}  "
          f"그 중 갭다운(open<trail_px)={gap_cases} ({gap_cases/max(n_trig,1):.1%})")
    print(f"  net_mean  sim={assumed.mean():+.6f}  realistic={realistic.mean():+.6f}  "
          f"Δ={realistic.mean()-assumed.mean():+.6f}")
    print(f"  worst     sim={assumed.min():+.6f}  realistic={realistic.min():+.6f}")
    print(f"  deep_loss(<=-5%) sim={(assumed<=-0.05).mean():.4f}  realistic={(realistic<=-0.05).mean():.4f}")
    print("  => 결론: 갭반영 시 net 이 더 나빠지면 sim 은 trail 변형에 '낙관' 편향")

    # Same for bracket SL=0.05 N5
    print("\n=== bracket tp0.08 sl0.05 N5: SL 체결가정 vs 갭반영 ===")
    TP, SL = 0.08, 0.05
    a2, r2 = [], []
    g2 = 0
    n_sl = 0
    for _, r in picks.iterrows():
        g = groups.get(r["market"])
        if g is None:
            continue
        sub = g[g["bar_date"] >= r["entry_date"]].iloc[:N]
        if len(sub) == 0 or sub.iloc[0]["bar_date"] != r["entry_date"]:
            continue
        entry_open = float(sub.iloc[0]["open"])
        if entry_open <= 0:
            continue
        tp_px, sl_px = entry_open * (1 + TP), entry_open * (1 - SL)
        bars = list(zip(sub["high"].astype(float), sub["low"].astype(float),
                        sub["close"].astype(float), sub["open"].astype(float)))
        done = False
        for (hi, lo, cl, op) in bars:
            if lo <= sl_px:
                fill_real = min(sl_px, op)
                a2.append(-SL)
                r2.append(fill_real / entry_open - 1.0)
                if op < sl_px:
                    g2 += 1
                n_sl += 1
                done = True
                break
            if hi >= tp_px:
                a2.append(TP)
                r2.append(TP)  # TP gap-up = better, keep conservative (assume TP)
                done = True
                break
        if not done:
            ret = bars[-1][2] / entry_open - 1.0
            a2.append(ret)
            r2.append(ret)
    a2 = np.array(a2) - 0.0015
    r2 = np.array(r2) - 0.0015
    print(f"  trades={len(a2)} SL triggered={n_sl} 그 중 갭다운(open<sl_px)={g2} ({g2/max(n_sl,1):.1%})")
    print(f"  net_mean sim={a2.mean():+.6f} realistic={r2.mean():+.6f} Δ={r2.mean()-a2.mean():+.6f}")
    print(f"  worst    sim={a2.min():+.6f} realistic={r2.min():+.6f}")
    print(f"  deep_loss sim={(a2<=-0.05).mean():.4f} realistic={(r2<=-0.05).mean():.4f}")


if __name__ == "__main__":
    main()
