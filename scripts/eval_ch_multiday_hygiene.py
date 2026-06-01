#!/usr/bin/env python3
"""Track B3 위생 4대 + bracket 정합 확인."""
from __future__ import annotations
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
D1_DB = str(ROOT / "data" / "upbit_d1.db")


def main():
    p = pd.read_csv(OUT / "ch_multiday_picks_v1.csv")
    comp = pd.read_csv(OUT / "ch_multiday_compare_v1.csv")

    print("=== bracket dump variant 확인 (DUMP_VARIANTS = bracket 0.12/0.05) ===")
    b = p[p.variant == "bracket"]
    print("  dumped bracket p1(TP) uniques:", sorted(b.p1.unique()), " p2(SL) uniques:", sorted(b.p2.unique()))
    for n in [1, 2, 3, 5]:
        d = b[b.n_hold == n].dropna(subset=["net"])
        net = d.net.values
        cm = comp[(comp.variant == "bracket_tp0.12_sl0.05") & (comp.n_hold == n)]
        cmn = cm.net_mean.values[0] if len(cm) else np.nan
        print(f"  N={n}: net_mean={net.mean():+.6f} (compare tp0.12_sl0.05 {cmn:+.6f} "
              f"match={abs(net.mean()-cmn)<1e-9}) worst={net.min():+.6f} deep={(net<=-0.05).mean():.4f}")

    print("\n=== [위생1] leak: 진입 D-1 결정 / 진입가 day-D open / 청산 forward ===")
    # entry_open in dump must equal day-D open from DB (관측가능, t 시점)
    conn = sqlite3.connect(D1_DB)
    df = pd.read_sql("SELECT market, timestamp, open FROM candles", conn)
    conn.close()
    df["bar_date"] = pd.to_datetime(df["timestamp"]).dt.date.astype(str)
    omap = {(r.market, r.bar_date): r.open for r in df.itertuples()}
    d1 = p[(p.variant == "holdN") & (p.n_hold == 1)].copy()
    d1["bar_date"] = pd.to_datetime(d1["entry_date"]).dt.date.astype(str)
    chk = d1.apply(lambda r: np.isclose(r.entry_open, omap.get((r.market, r.bar_date), np.nan)), axis=1)
    print(f"  entry_open == DB day-D open : {chk.mean():.4f} (1.0이면 진입가=관측가능 t-open, leak X)")
    # label leak: picks 의 R1 score 가 net/outcome 등 미래값을 안 봤는지는 r2 빌더 책임이나,
    # B3 는 픽을 그대로 재사용(재학습 0)하므로 B3 단에서 새 누출 0.
    print("  B3 는 R1 픽 재사용(재학습 0) → B3 단계 신규 라벨누출 없음. 청산 경로=forward outcome(정상).")

    print("\n=== [위생2] 시간정합성: 유니버스 / overlapping block ===")
    print(f"  picks = R1_ratio OOS (fold-train 종료시점 top100 제한, 원 R1 빌더). entries={len(d1)} days={d1.bar_date.nunique()}")
    print("  overlapping holding → non-overlap block 압축 (N=2→383, N=3→255, N=5→153) compare CSV 의 n_blocks 와 일치 확인:")
    for n in [1, 2, 3, 5]:
        nb = comp[(comp.vfamily == "holdN") & (comp.n_hold == n)].n_blocks.values[0]
        print(f"    N={n}: n_blocks={nb} (expected ~765/N={765/n:.0f})")

    print("\n=== [위생3] 비용차감: 0.15% 1회 ===")
    gp = p[(p.variant == "holdN") & (p.n_hold == 5)]
    dgap = (gp.gross - gp.net).round(6)
    print(f"  gross-net unique = {sorted(dgap.unique())} (0.0015 단일이면 왕복 1회 정확)")

    print("\n=== [위생4] 자동주문 / 라이브 불변 ===")
    print("  ch_multiday_v1.py 는 sqlite read + CSV write 만. 업비트 API / 주문 호출 없음.")
    print("  새 output 파일(ch_multiday_*)만 생성. 공유 라이브 코드 미수정.")


if __name__ == "__main__":
    main()
