"""market_breadth_discovery_v1.py

각도 1 — 시장 레벨 "언제 펌프 풍부한 장인가".

목표: 날짜별 펌프 breadth 시계열을 만들고, 펌프-풍부 날 vs 펌프-희박 날을
가르는 D-1 이전 시장레벨 선행조건을 찾는다. layer = "market".

=== LEAK 규율 (이 프로젝트 same-day leak 2번 전적) ===
- 펌프 라벨(타겟, day D): high_D / open_D - 1 >= THRESH. 이건 미래(오늘 일봉).
- breadth_D = day D 에 펌프 친 코인 수 / 그날 상장(거래된) 코인 수. (타겟)
- 모든 선행조건(market feature)은 day D-1 종가까지의 데이터로만 계산.
  -> breadth 시계열을 만든 뒤, 모든 market feature 를 .shift(1) 해서 D row 에 D-1 값을 붙인다.
  -> 즉 "D row 의 feature = D-1 까지로 계산된 값" 을 보장.
- BTC regime 도 D-1 까지로 (shift(1)).
- OOS: 시간순 walk-forward + embargo. lift 가 OOS 에서 살아야 의미.

거래비용/청산은 이 스크립트 범위 아님(market layer = breadth 예측). net PnL 은
coin-layer 에서. 여기서는 breadth 예측의 lift/AUC/base-rate-by-regime 만.
"""
import argparse
import sqlite3
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DB = "data/upbit_d1.db"
EPS = 1e-12

# 펌프 임계 (placeholder — 데이터가 더 좋은 값 주면 변경)
PUMP_THRESH_MAIN = 0.20
PUMP_THRESH_AUX = 0.15

# BTC regime 상수 (signals/features.py 와 동일)
BTC_MA_WINDOW = 200
BTC_RV_WINDOW = 30
BTC_INTENSITY_LOOKBACK = 252
INTENSITY_THRESHOLD = 0.5

# 거래 활성 필터 (placeholder): 그날 quote_volume 이 양수여야 "상장/거래중"으로 카운트
MIN_QV = 0.0


def load_candles() -> pd.DataFrame:
    con = sqlite3.connect(DB)
    df = pd.read_sql(
        "SELECT market, timestamp, open, high, low, close, volume, quote_volume "
        "FROM candles ORDER BY market, timestamp",
        con,
    )
    con.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.normalize()
    return df


# ---------------------------------------------------------------------------
# 1) coin-level 펌프 플래그 (day D, 타겟) — high_D/open_D - 1
# ---------------------------------------------------------------------------
def build_coin_pump_flags(df: pd.DataFrame) -> pd.DataFrame:
    g = df.copy()
    # intraday 펌프: 그날 시가 대비 그날 고가
    g["intraday_ret"] = g["high"] / (g["open"] + EPS) - 1.0
    g["is_pump20"] = (g["intraday_ret"] >= PUMP_THRESH_MAIN).astype(int)
    g["is_pump15"] = (g["intraday_ret"] >= PUMP_THRESH_AUX).astype(int)
    # 거래중 플래그
    g["active"] = ((g["quote_volume"].fillna(0) > MIN_QV) & (g["volume"] > 0)).astype(int)
    return g


# ---------------------------------------------------------------------------
# 2) breadth 시계열 (날짜별, 타겟) + market-level raw aggregates (당일 — feature 만들기 전)
# ---------------------------------------------------------------------------
def build_daily_panel(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, sub in g.groupby("date"):
        active = sub[sub["active"] == 1]
        n_active = len(active)
        if n_active == 0:
            continue
        n_pump20 = int(active["is_pump20"].sum())
        n_pump15 = int(active["is_pump15"].sum())
        # 당일 시장 raw aggregate (D 값 — feature 로 쓸 땐 shift 해서 D-1 으로 사용)
        # 종가/시가 일간 수익률로 advancers 비율
        day_ret = active["close"] / (active["open"] + EPS) - 1.0
        rows.append(
            {
                "date": date,
                "n_active": n_active,
                "n_pump20": n_pump20,
                "n_pump15": n_pump15,
                "breadth20": n_pump20 / n_active,
                "breadth15": n_pump15 / n_active,
                # 당일 raw (shift 대상)
                "adv_ratio_raw": float((day_ret > 0).mean()),
                "median_day_ret_raw": float(day_ret.median()),
                "total_qv_raw": float(active["quote_volume"].fillna(0).sum()),
            }
        )
    panel = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return panel


# ---------------------------------------------------------------------------
# 3) BTC regime + BTC dynamics (D-1 까지)
# ---------------------------------------------------------------------------
def build_btc_features(df: pd.DataFrame) -> pd.DataFrame:
    btc = df[df["market"] == "KRW-BTC"].sort_values("date").copy()
    if btc.empty:
        raise RuntimeError("KRW-BTC not found")
    close = btc["close"]
    log_ret = np.log(close / close.shift(1) + EPS)

    out = pd.DataFrame({"date": btc["date"].values})
    out["btc_ret_1d"] = (close / close.shift(1) - 1).values
    out["btc_ret_3d"] = (close / close.shift(3) - 1).values
    out["btc_ret_7d"] = (close / close.shift(7) - 1).values
    out["btc_ret_14d"] = (close / close.shift(14) - 1).values
    ma = close.rolling(BTC_MA_WINDOW).mean()
    out["btc_ma_distance"] = ((close - ma) / (ma + EPS)).values
    rv30 = log_ret.rolling(BTC_RV_WINDOW).std()
    out["btc_rv_30d"] = rv30.values
    out["btc_intensity"] = rv30.rolling(BTC_INTENSITY_LOOKBACK).rank(pct=True).values

    bull = out["btc_ma_distance"] > 0
    volatile = out["btc_intensity"] > INTENSITY_THRESHOLD

    def _regime(b, v):
        if pd.isna(b) or pd.isna(v):
            return "unknown"
        if b and not v:
            return "bull_quiet"
        if b and v:
            return "bull_volatile"
        if not b and not v:
            return "bear_quiet"
        return "bear_volatile"

    out["btc_regime"] = [_regime(b, v) for b, v in zip(bull, volatile)]
    return out


# ---------------------------------------------------------------------------
# 4) market-level features (모두 D-1 까지) — breadth/qv/adv 추세 + clustering
# ---------------------------------------------------------------------------
def build_market_features(panel: pd.DataFrame, btc: pd.DataFrame) -> pd.DataFrame:
    p = panel.merge(btc, on="date", how="left").sort_values("date").reset_index(drop=True)

    # --- 펌프 clustering: 어제/최근 펌프 수가 오늘을 예측? ---
    # breadth lag & 추세. .shift(1) 로 모든 raw 시계열을 D-1 까지로.
    for col in ["breadth20", "breadth15", "n_pump20", "n_pump15",
                "adv_ratio_raw", "median_day_ret_raw", "total_qv_raw"]:
        p[f"{col}_lag1"] = p[col].shift(1)

    # 최근 N일 평균 펌프 수 / breadth (모두 shift(1) 로 D-1 이전 윈도우만)
    for N in (3, 7, 14):
        p[f"breadth20_ma{N}"] = p["breadth20"].shift(1).rolling(N).mean()
        p[f"breadth15_ma{N}"] = p["breadth15"].shift(1).rolling(N).mean()
        p[f"n_pump20_ma{N}"] = p["n_pump20"].shift(1).rolling(N).mean()
        p[f"adv_ratio_ma{N}"] = p["adv_ratio_raw"].shift(1).rolling(N).mean()

    # 거래대금 총합 추세: 최근 7d / 최근 30d 비율 (D-1 까지)
    qv_lag = p["total_qv_raw"].shift(1)
    p["qv_trend_7_30"] = qv_lag.rolling(7).mean() / (qv_lag.rolling(30).mean() + EPS)
    p["qv_zscore_30"] = (
        (qv_lag - qv_lag.rolling(30).mean()) / (qv_lag.rolling(30).std() + EPS)
    )

    # breadth 모멘텀: 최근 3d 평균 vs 최근 14d 평균
    b20_lag = p["breadth20"].shift(1)
    p["breadth20_mom_3_14"] = b20_lag.rolling(3).mean() - b20_lag.rolling(14).mean()

    # alt decoupling: BTC 횡보(낮은 |ret|) + alt advancers 높음
    # btc_ret_* 는 이미 D 종가까지 = 당일. shift(1) 해서 D-1 까지로.
    for col in ["btc_ret_1d", "btc_ret_3d", "btc_ret_7d", "btc_ret_14d",
                "btc_ma_distance", "btc_rv_30d", "btc_intensity"]:
        p[f"{col}_d1"] = p[col].shift(1)
    p["btc_regime_d1"] = p["btc_regime"].shift(1)

    # decoupling 지표: alt advancers(D-1) 높은데 BTC |ret_1d|(D-1) 작음
    p["decouple_d1"] = p["adv_ratio_raw_lag1"] - p["btc_ret_1d_d1"].abs() * 5.0

    return p


# ---------------------------------------------------------------------------
# 5) 분석: regime 별 base rate + feature lift (OOS walk-forward)
# ---------------------------------------------------------------------------
def label_pump_rich(p: pd.DataFrame, breadth_col: str, top_q: float) -> pd.Series:
    """펌프-풍부 날 = breadth 상위 top_q 분위(전체 히스토리 기준 라벨; 진단용).
    NOTE: 이 임계 자체는 타겟 정의(미래 breadth)라 leak 아님. feature 만 D-1."""
    thresh = p[breadth_col].quantile(1 - top_q)
    return (p[breadth_col] >= thresh).astype(int)


def feature_lift_oos(p: pd.DataFrame, feat: str, target: str,
                     n_folds: int, embargo: int, top_q: float):
    """시간순 walk-forward. fold val 에서 feature 상위 top_q vs 전체 base rate.
    feature 는 high-dir 가정(클수록 펌프-풍부 예상) — 양방향 모두 체크."""
    d = p[["date", feat, target]].dropna().sort_values("date").reset_index(drop=True)
    if len(d) < 200:
        return None
    n = len(d)
    fold_size = n // (n_folds + 1)
    res = []
    for k in range(1, n_folds + 1):
        tr_end = fold_size * k
        val_start = tr_end + embargo
        val_end = val_start + fold_size
        if val_start >= n:
            break
        val = d.iloc[val_start:min(val_end, n)]
        tr = d.iloc[:tr_end]
        if len(val) < 30 or len(tr) < 50:
            continue
        base = val[target].mean()
        # threshold 는 train 분위 (val 누수 X)
        for direction in ("high", "low"):
            if direction == "high":
                thr = tr[feat].quantile(1 - top_q)
                sel = val[val[feat] >= thr]
            else:
                thr = tr[feat].quantile(top_q)
                sel = val[val[feat] <= thr]
            if len(sel) < 10:
                continue
            prec = sel[target].mean()
            lift = prec / (base + EPS)
            res.append({"fold": k, "direction": direction,
                        "val_base": base, "sel_prec": prec, "lift": lift,
                        "sel_n": len(sel), "val_n": len(val)})
    if not res:
        return None
    rdf = pd.DataFrame(res)
    agg = []
    for direction, sub in rdf.groupby("direction"):
        # n-weighted prec
        w = sub["sel_n"]
        wprec = (sub["sel_prec"] * w).sum() / (w.sum() + EPS)
        wbase = (sub["val_base"] * sub["val_n"]).sum() / (sub["val_n"].sum() + EPS)
        agg.append({
            "feature": feat, "direction": direction,
            "oos_base": wbase, "oos_sel_prec": wprec,
            "oos_lift": wprec / (wbase + EPS),
            "oos_sel_n": int(w.sum()), "n_folds_used": sub["fold"].nunique(),
            "lift_min_fold": sub["lift"].min(), "lift_std": sub["lift"].std(),
        })
    return pd.DataFrame(agg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-q", type=float, default=0.20,
                    help="펌프-풍부 = breadth 상위 분위 (초기값 0.20)")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--embargo", type=int, default=3)
    ap.add_argument("--breadth", default="breadth20")
    ap.add_argument("--out-prefix", default="output/market_breadth")
    args = ap.parse_args()

    print("[1/5] load candles...")
    df = load_candles()
    print("[2/5] coin pump flags...")
    g = build_coin_pump_flags(df)
    print("[3/5] daily breadth panel...")
    panel = build_daily_panel(g)
    print(f"   days={len(panel)} range={panel['date'].min().date()}..{panel['date'].max().date()}")
    print("[4/5] btc + market features (D-1)...")
    btc = build_btc_features(df)
    p = build_market_features(panel, btc)

    # 타겟: 펌프-풍부 날 (breadth 상위 top_q)
    p["pump_rich"] = label_pump_rich(p, args.breadth, args.top_q)

    # warmup 제거: btc_ma_distance_d1 / qv windows 충분히 채워진 뒤부터
    valid = p.dropna(subset=["btc_ma_distance_d1", "qv_trend_7_30",
                             "breadth20_ma14"]).copy()
    print(f"   valid days (warmup 제거 후)={len(valid)}")

    # breadth 시계열 + regime base rate 저장
    panel_out = p[["date", "n_active", "n_pump20", "n_pump15",
                   "breadth20", "breadth15", "btc_regime_d1", "pump_rich",
                   "adv_ratio_raw", "total_qv_raw"]].copy()
    panel_out.to_csv(f"{args.out_prefix}_panel_v1.csv", index=False)

    # --- regime 별 base rate ---
    print("\n=== regime 별 펌프 base rate (D-1 regime 기준) ===")
    reg_rows = []
    for reg, sub in valid.groupby("btc_regime_d1"):
        reg_rows.append({
            "regime_d1": reg, "n_days": len(sub),
            "mean_breadth20": sub["breadth20"].mean(),
            "mean_breadth15": sub["breadth15"].mean(),
            "mean_n_pump20": sub["n_pump20"].mean(),
            "mean_n_pump15": sub["n_pump15"].mean(),
            "p_any_pump20": (sub["n_pump20"] > 0).mean(),
            "pump_rich_rate": sub["pump_rich"].mean(),
        })
    reg_df = pd.DataFrame(reg_rows).sort_values("mean_breadth20", ascending=False)
    print(reg_df.to_string(index=False))
    reg_df.to_csv(f"{args.out_prefix}_regime_baserate_v1.csv", index=False)

    # --- feature lift (OOS) ---
    print("\n=== market feature OOS lift (pump_rich 예측) ===")
    feats = [
        "breadth20_lag1", "breadth15_lag1", "n_pump20_lag1",
        "breadth20_ma3", "breadth20_ma7", "breadth20_ma14",
        "n_pump20_ma3", "n_pump20_ma7",
        "adv_ratio_raw_lag1", "adv_ratio_ma3", "adv_ratio_ma7",
        "qv_trend_7_30", "qv_zscore_30",
        "breadth20_mom_3_14",
        "btc_ret_1d_d1", "btc_ret_3d_d1", "btc_ret_7d_d1", "btc_ret_14d_d1",
        "btc_ma_distance_d1", "btc_rv_30d_d1", "btc_intensity_d1",
        "decouple_d1", "median_day_ret_raw_lag1",
    ]
    all_lift = []
    n_tried = 0
    for f in feats:
        if f not in valid.columns:
            continue
        r = feature_lift_oos(valid, f, "pump_rich", args.n_folds,
                             args.embargo, args.top_q)
        n_tried += 2  # high+low direction
        if r is not None:
            all_lift.append(r)
    lift_df = pd.concat(all_lift, ignore_index=True)
    lift_df = lift_df.sort_values("oos_lift", ascending=False)
    print(lift_df.to_string(index=False))
    lift_df.to_csv(f"{args.out_prefix}_feature_lift_v1.csv", index=False)
    print(f"\n[selection] 시도 조합 수 = {n_tried} (feature {len(feats)} x 2 direction)")
    print(f"[done] outputs: {args.out_prefix}_panel_v1.csv / _regime_baserate_v1.csv / _feature_lift_v1.csv")


if __name__ == "__main__":
    main()
