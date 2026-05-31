"""Case study: reconstruct pre-pump state of this week's pumps using D-1 only.

Angle C — "왜 이 코인이 급등했나" 를 급등 전 데이터로 역분석.

규율 (절대):
  - 급등 이벤트 = day D 일봉 high/open-1 >= TH (0.20 main, 0.15 보조)
  - precursor 는 D-1 및 그 이전 일봉(+선택 15m/4h, D-1 09:00 이전 마감 분봉)만 사용
  - day D 의 high/low/close 절대 미사용
  - 동적 유니버스 rank 도 D-1 기준 trailing quote_volume 으로 계산

출력:
  - 각 케이스 코인의 first-pump-day D 직전 5일 일봉 상태 + precursor feature 표
  - 전체 패널에서 동일 precursor 의 base rate / lift (OOS lift 는 별도 스크립트)
"""
from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd

D1 = "data/upbit_d1.db"

EPS = 1e-12

# first pump day (급등 에피소드의 첫날 — precursor 가 잡아야 하는 날)
CASES = {
    "ID":    "2026-05-29",
    "GMT":   "2026-05-23",
    "VTHO":  "2026-05-30",
    "META":  "2026-05-22",
    "ERA":   "2026-05-25",
    "PROVE": "2026-05-21",
    "ALT":   "2026-05-22",
    "PRL":   "2026-05-27",
    "XLM":   "2026-05-28",
    "WLD":   "2026-05-26",
}


def load_d1():
    c = sqlite3.connect(D1)
    df = pd.read_sql_query(
        "SELECT market,timestamp,open,high,low,close,volume,quote_volume FROM candles",
        c, parse_dates=["timestamp"])
    c.close()
    df = df.sort_values(["market", "timestamp"]).reset_index(drop=True)
    df["date"] = df["timestamp"].dt.date
    return df


def per_coin_features(g: pd.DataFrame) -> pd.DataFrame:
    """모든 피처는 당일 close 까지 사용해 계산하되, 사용할 때는 D-1 행을 참조한다.
    (D-1 행의 피처 = D-1 close/high/low 까지만 쓰므로 leak 없음.)"""
    g = g.sort_values("timestamp").reset_index(drop=True)
    o, h, l, cl = g["open"], g["high"], g["low"], g["close"]
    qv, vol = g["quote_volume"], g["volume"]

    g["ret_1d"] = cl / cl.shift(1) - 1
    g["co_1d"] = cl / o - 1
    g["ho_1d"] = h / o - 1
    for N in (3, 5, 7, 14, 21):
        g[f"ret_{N}d"] = cl / cl.shift(N) - 1
    lr = np.log(cl / cl.shift(1) + EPS)
    g["vol_7d"] = lr.rolling(7, min_periods=4).std()
    g["vol_21d"] = lr.rolling(21, min_periods=10).std()
    # range_contraction: 최근 7일 일중 range / 21일 일중 range (작을수록 수축)
    daily_range = (h - l) / (cl.shift(1) + EPS)
    g["range_7d"] = daily_range.rolling(7, min_periods=4).mean()
    g["range_21d"] = daily_range.rolling(21, min_periods=10).mean()
    g["range_contraction"] = g["range_7d"] / (g["range_21d"] + EPS)
    # 거래대금 추세
    g["qv_ma5"] = qv.rolling(5, min_periods=3).mean()
    g["qv_ma20"] = qv.rolling(20, min_periods=10).mean()
    g["qv_ratio_1_5"] = qv / (g["qv_ma5"] + EPS)          # 어제 vol vs 최근5d 평균
    g["qv_ratio_5_20"] = g["qv_ma5"] / (g["qv_ma20"] + EPS)  # 최근5d vs 20d (누적 선행 증가)
    g["qv_ratio_1_20"] = qv / (g["qv_ma20"] + EPS)
    # 거래량 추세 가속 (3일 연속 증가 카운트)
    g["qv_up"] = (qv > qv.shift(1)).astype(int)
    g["qv_up_streak"] = g["qv_up"].rolling(3, min_periods=3).sum()
    # 신고가 근접: 어제 close 가 N일 최고가의 몇 %
    g["high_20d"] = h.rolling(20, min_periods=10).max()
    g["close_vs_high20"] = cl / (g["high_20d"] + EPS)
    # 직전 dump: 최근 5일 max close 에서 어제 close 낙폭
    g["max_close_5d"] = cl.rolling(5, min_periods=3).max()
    g["drawdown_5d"] = cl / (g["max_close_5d"] + EPS) - 1
    return g


def add_universe_rank(df: pd.DataFrame) -> pd.DataFrame:
    """각 (date) 별로 trailing qv_ma5 로 cross-sectional rank.
    rank 은 그 날의 qv_ma5(=당일까지 5일평균) 기준. precursor 로 쓸 땐 D-1 행 참조."""
    df = df.copy()
    df["uni_rank"] = df.groupby("date")["qv_ma5"].rank(ascending=False, method="min")
    return df


def main():
    df = load_d1()
    df = df.groupby("market", group_keys=False).apply(per_coin_features)
    df = add_universe_rank(df)

    btc = df[df["market"] == "KRW-BTC"][["date", "ret_1d", "ret_7d", "vol_21d"]].rename(
        columns={"ret_1d": "btc_ret_1d", "ret_7d": "btc_ret_7d", "vol_21d": "btc_vol_21d"})
    df = df.merge(btc, on="date", how="left")

    fcols = ["ret_1d", "ret_3d", "ret_5d", "ret_7d", "ret_14d", "ho_1d", "co_1d",
             "vol_7d", "vol_21d", "range_contraction", "qv_ratio_1_5", "qv_ratio_5_20",
             "qv_ratio_1_20", "qv_up_streak", "close_vs_high20", "drawdown_5d",
             "uni_rank", "btc_ret_1d", "btc_ret_7d"]

    print("=" * 100)
    print("CASE STUDY — pre-pump state (D-1 행 = 급등 전날까지의 데이터만)")
    print("=" * 100)
    rows = []
    for tok, dstr in CASES.items():
        m = "KRW-" + tok
        D = pd.Timestamp(dstr).date()
        g = df[df["market"] == m].sort_values("date").reset_index(drop=True)
        idxD = g.index[g["date"] == D]
        if len(idxD) == 0:
            print(tok, "no pump-day row"); continue
        iD = idxD[0]
        if iD < 1:
            print(tok, "no D-1"); continue
        dm1 = g.iloc[iD - 1]   # D-1 행 = precursor 평가 시점
        rec = {"tok": tok, "D": dstr}
        for fc in fcols:
            rec[fc] = dm1[fc]
        rows.append(rec)

    res = pd.DataFrame(rows).set_index("tok")
    pd.set_option("display.width", 200, "display.max_columns", 40)
    # 핵심 컬럼만 보기 좋게
    show = ["D", "uni_rank", "qv_ratio_1_5", "qv_ratio_5_20", "qv_ratio_1_20",
            "qv_up_streak", "ret_5d", "ret_7d", "range_contraction", "drawdown_5d",
            "close_vs_high20", "vol_7d", "vol_21d", "btc_ret_1d"]
    print(res[show].round(3).to_string())
    res.to_csv("output/case_study_precursor_state.csv")
    print("\nsaved output/case_study_precursor_state.csv")

    # ====== base rate / lift on full panel ======
    # 라벨: next-day high/open-1 >= TH  (D 행 기준, shift -1 로 D 의 ho_1d 를 D-1 에 붙임)
    print("\n" + "=" * 100)
    print("PANEL LIFT — precursor(D-1) vs next-day pump(D) high/open>=0.20 (& 0.15)")
    print("=" * 100)
    df = df.sort_values(["market", "date"])
    df["next_ho"] = df.groupby("market")["ho_1d"].shift(-1)
    df["y20"] = (df["next_ho"] >= 0.20).astype(int)
    df["y15"] = (df["next_ho"] >= 0.15).astype(int)
    # 시간정합: 2024+ 만, BTC 제외, qv 충분
    panel = df[(df["date"] >= pd.Timestamp("2024-01-01").date()) &
               (df["market"] != "KRW-BTC")].copy()
    panel = panel.dropna(subset=["next_ho", "qv_ratio_5_20", "uni_rank"])
    n = len(panel)
    base20 = panel["y20"].mean()
    base15 = panel["y15"].mean()
    print(f"panel rows={n}  base rate y20={base20:.4%}  y15={base15:.4%}")

    def lift(mask, name):
        sub = panel[mask]
        if len(sub) < 30:
            print(f"  {name:42s} n={len(sub):5d} (too few)"); return
        r20 = sub["y20"].mean(); r15 = sub["y15"].mean()
        print(f"  {name:42s} n={len(sub):6d}  y20={r20:.3%} (lift {r20/base20:4.2f}x)  "
              f"y15={r15:.3%} (lift {r15/base15:4.2f}x)")

    print("\n-- single precursors --")
    lift(panel["qv_ratio_5_20"] >= 1.5, "qv_ma5/qv_ma20 >= 1.5 (누적 선행증가)")
    lift(panel["qv_ratio_5_20"] >= 2.0, "qv_ma5/qv_ma20 >= 2.0")
    lift(panel["qv_ratio_1_5"] >= 2.0, "어제qv/5d평균 >= 2.0 (직전 스파이크)")
    lift(panel["qv_ratio_1_5"] >= 3.0, "어제qv/5d평균 >= 3.0")
    lift(panel["qv_ratio_1_20"] >= 3.0, "어제qv/20d평균 >= 3.0")
    lift(panel["qv_up_streak"] >= 3, "거래대금 3일연속 증가")
    lift(panel["close_vs_high20"] >= 0.97, "어제close >= 20d고가*0.97 (신고가근접)")
    lift(panel["range_contraction"] <= 0.7, "range_contraction <= 0.7 (수축)")
    lift(panel["drawdown_5d"] <= -0.15, "최근5d고점 대비 -15% 이상 dump")
    lift(panel["ret_7d"] >= 0.20, "7일 모멘텀 >= +20%")
    lift(panel["uni_rank"] <= 50, "유니버스 rank <= 50 (대형)")
    lift(panel["uni_rank"] > 100, "유니버스 rank > 100 (밖)")

    print("\n-- 2-way intersections --")
    lift((panel["qv_ratio_5_20"] >= 1.5) & (panel["qv_ratio_1_5"] >= 2.0),
         "누적증가1.5 & 직전스파이크2.0")
    lift((panel["qv_ratio_1_5"] >= 2.0) & (panel["close_vs_high20"] >= 0.95),
         "직전스파이크2.0 & 신고가근접0.95")
    lift((panel["qv_ratio_1_5"] >= 2.0) & (panel["ret_7d"] >= 0.10),
         "직전스파이크2.0 & 7d모멘텀+10%")
    lift((panel["qv_ratio_1_5"] >= 2.0) & (panel["drawdown_5d"] <= -0.10),
         "직전스파이크2.0 & 직전dump-10%")
    lift((panel["qv_ratio_1_20"] >= 2.0) & (panel["uni_rank"] <= 100),
         "어제qv/20d>=2.0 & rank<=100")

    panel.to_parquet("output/case_study_panel.parquet")
    print("\nsaved output/case_study_panel.parquet")


if __name__ == "__main__":
    main()
