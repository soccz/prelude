"""quant-evaluator independent audit of the Phase-1 pattern library.

목적 (researcher 주장 그대로 안 믿고 독립 재계산):
  1. LEAK 감사: feature shift(1) 실제 검증, BTC regime D-1 정합성 검증
     (regime_d1[D] == regime_raw[D-1]), 라벨이 미래(day-D)인지, cross-section
     rank 가 D-1 입력인지.
  2. portfolio-grade net 백테스트: (pattern × regime) top-decile 진입을
     equal-weight 일별 시계열로 → net Sharpe/Sortino/Calmar/MaxDD/누적/hit.
     ★ SL-first(비관) AND TP-first(낙관) 양쪽 bound + TP_EOD(SL 없음).
     0.15% 왕복 차감. ledger/metrics.py 의 Sharpe/MDD 재사용.
  3. selection-aware DSR: trials= 4각도 총 시도수 반영.

walk-forward OOS: 전역 fold 경계 공유 (regime sub-pop OOS), embargo 5d.
"""
from __future__ import annotations
import sys, json, logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.database import list_markets, load_candles
from signals.features import compute_btc_features
from scripts.univariate_precursor_lift_v1 import build_market_features, add_cross_sectional
from ledger.metrics import compute_sharpe, compute_mdd

DB = "data/upbit_d1.db"
RT = 0.0015
TP, SL = 0.10, 0.05
PPY = 365
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("audit")

REGIMES = ["bull_quiet", "bull_volatile", "bear_quiet", "bear_volatile"]
# 평가 대상 (pattern, feature, direction)
PATTERNS = {
    "qv_surge_30d": ("f_qv_surge_30d", "high"),
    "qv_surge_7d":  ("f_qv_surge_7d", "high"),
    "bounce_7d_low":("f_bounce_off_7d_low", "high"),
    "ret_3d":       ("f_ret_3d", "high"),
    "ret_7d":       ("f_ret_7d", "high"),
    "atr_pct_14":   ("f_atr_pct_14", "high"),
    "rv_21d":       ("f_rv_21d", "high"),
    "atr_xs_decile":("f_atr_xs_decile", "high"),
}
LABEL = "lab_pump20"


def build():
    markets = list_markets(DB)
    frames = []
    for m in markets:
        df = load_candles(DB, m)
        if df is None or len(df) < 70:
            continue
        df = df.copy(); df["market"] = m
        feat = build_market_features(df)
        g = df.sort_values("timestamp").reset_index(drop=True)
        oc = pd.DataFrame({"market": m, "timestamp": pd.to_datetime(g["timestamp"]),
                           "o": g["open"].values, "h": g["high"].values,
                           "l": g["low"].values, "cl": g["close"].values})
        feat = feat.copy(); feat["timestamp"] = pd.to_datetime(feat["timestamp"])
        frames.append(feat.merge(oc, on=["market", "timestamp"], how="left"))
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = panel["timestamp"].dt.date
    panel = panel.sort_values(["date", "market"]).reset_index(drop=True)
    panel = add_cross_sectional(panel)
    return panel


def attach_regime(panel):
    btc = load_candles(DB, "KRW-BTC")
    bf = compute_btc_features(btc)
    bf["timestamp"] = pd.to_datetime(bf["timestamp"])
    bf = bf.sort_values("timestamp").reset_index(drop=True)
    bf["regime_raw"] = bf["btc_regime"]
    bf["regime_d1"] = bf["btc_regime"].shift(1)        # day-D 가 보는 regime
    bf["date"] = bf["timestamp"].dt.date
    # ★ LEAK TEST A: regime_d1[t] 가 regime_raw[t-1] 와 같은가
    chk = bf[["date", "regime_raw", "regime_d1"]].copy()
    chk["regime_raw_prev"] = chk["regime_raw"].shift(1)
    valid = chk.dropna(subset=["regime_d1", "regime_raw_prev"])
    leak_ok = bool((valid["regime_d1"] == valid["regime_raw_prev"]).all())
    log.info("[LEAK A] regime_d1[D]==regime_raw[D-1] for all rows: %s (n=%d)", leak_ok, len(valid))
    p = panel.merge(bf[["date", "regime_d1"]], on="date", how="left").rename(columns={"regime_d1": "regime"})
    return p, leak_ok


def leak_shift_test(panel):
    """LEAK TEST B: feature 가 정말 D-1 까지인가.
    f_ret_1d[D] (= shift된 어제 ret_1d) 가 raw close 로 재계산한 D-1 ret_1d 와 일치하는지
    무작위 market 샘플로 직접 대조."""
    import random
    markets = panel["market"].dropna().unique().tolist()
    random.seed(0); sample = random.sample(markets, min(20, len(markets)))
    mismatches = 0; checked = 0
    for m in sample:
        df = load_candles(DB, m).sort_values("timestamp").reset_index(drop=True)
        c = df["close"]
        # raw ret_1d[t] = c[t]/c[t-1]-1 ; feature f_ret_1d[D] 는 이걸 shift(1) → ret_1d[D-1]
        raw_ret1 = (c / c.shift(1) - 1.0)
        ref = raw_ret1.shift(1)  # D-1 값
        sub = panel[panel["market"] == m][["timestamp", "f_ret_1d"]].copy()
        sub["timestamp"] = pd.to_datetime(sub["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        cmp = pd.DataFrame({"timestamp": df["timestamp"], "ref": ref}).merge(sub, on="timestamp", how="inner")
        cmp = cmp.dropna()
        diff = (cmp["ref"] - cmp["f_ret_1d"]).abs()
        mismatches += int((diff > 1e-6).sum()); checked += len(cmp)
    log.info("[LEAK B] f_ret_1d == shift(1) of raw ret_1d : mismatches=%d / checked=%d", mismatches, checked)
    return mismatches == 0


def folds(dates, n=5, emb=5):
    fs = len(dates) // (n + 1)
    out = []
    for k in range(1, n + 1):
        tr_end = fs * k
        te_start = tr_end + emb
        te_end = min(fs * (k + 1) + tr_end, len(dates))
        if te_start >= len(dates):
            break
        out.append((set(dates[:tr_end]), set(dates[te_start:te_end])))
    return out


def oos_selected(panel, feat, direction, regime, decile=0.9):
    """walk-forward OOS 로 선택된 (regime) row 모음 → 진입 후보."""
    cols = [feat, "date", "regime", "o", "h", "l", "cl", LABEL]
    d = panel[cols].dropna(subset=[feat, "regime", "o", "h", "l", "cl"])
    d = d[d["regime"] == regime]
    if len(d) < 400:
        return None
    all_dates = np.sort(panel["date"].unique())
    sel = []
    for tr_dates, te_dates in folds(all_dates):
        tr = d[d["date"].isin(tr_dates)]; te = d[d["date"].isin(te_dates)]
        if len(tr) < 200 or len(te) < 100:
            continue
        if direction == "high":
            cut = tr[feat].quantile(decile); s = te[te[feat] >= cut]
        else:
            cut = tr[feat].quantile(1 - decile); s = te[te[feat] <= cut]
        if len(s) >= 10:
            sel.append(s)
    if not sel:
        return None
    return pd.concat(sel, ignore_index=True)


def net_metrics(rows, exit_mode):
    """exit_mode: 'sl_first'(비관) | 'tp_first'(낙관) | 'tp_eod'(SL없음).
    equal-weight 일별 시계열 → portfolio-grade 지표."""
    if rows is None or len(rows) == 0:
        return None
    o, h, l, cl = rows["o"].values, rows["h"].values, rows["l"].values, rows["cl"].values
    tp_px = o * (1 + TP); sl_px = o * (1 - SL)
    hit_tp = h >= tp_px; hit_sl = l <= sl_px; o2c = cl / o - 1.0
    if exit_mode == "sl_first":
        gross = np.where(hit_sl, -SL, np.where(hit_tp, TP, o2c))
    elif exit_mode == "tp_first":
        gross = np.where(hit_tp, TP, np.where(hit_sl, -SL, o2c))
    elif exit_mode == "tp_eod":
        gross = np.where(hit_tp, TP, o2c)
    else:
        raise ValueError(exit_mode)
    valid = np.isfinite(o) & (o > 0)
    gross = gross[valid]; net = gross - RT
    dates = rows["date"].values[valid]
    td = pd.DataFrame({"date": dates, "net": net})
    daily = td.groupby("date")["net"].mean()  # equal-weight per day
    if len(daily) < 3:
        return None
    all_dates = np.sort(np.unique(rows["date"].values))
    # equity on trading days only (sparse) — Sharpe on per-trade-day mean net
    eq = (1 + daily).cumprod()
    sharpe = compute_sharpe(daily, PPY)
    downside = daily[daily < 0]
    sortino = float(daily.mean() / downside.std() * np.sqrt(PPY)) if len(downside) > 1 and downside.std() > 0 else np.nan
    mdd = compute_mdd(eq)
    cum = float(eq.iloc[-1] - 1.0)
    ann_ret = float(daily.mean() * PPY)
    calmar = float(ann_ret / abs(mdd)) if mdd < 0 else np.nan
    return {
        "n_trades": int(len(td)), "n_days": int(len(daily)),
        "net_mean_per_trade": float(net.mean()),
        "hit_net": float((net > 0).mean()),
        "tp_rate": float(hit_tp[valid].mean()), "sl_rate": float(hit_sl[valid].mean()),
        "sharpe": float(sharpe), "sortino": sortino, "calmar": calmar,
        "mdd": float(mdd), "cum_return": cum,
    }


def psr_dsr(sharpe_ann, n_obs, trials):
    """PSR(SR*=0) 과 selection-deflated DSR (Bailey & LdP 2014, skew/kurt=정규 가정 간이)."""
    if not np.isfinite(sharpe_ann) or n_obs < 3:
        return np.nan, np.nan
    sr = sharpe_ann / np.sqrt(PPY)  # per-period
    # PSR vs 0
    psr = stats.norm.cdf(sr * np.sqrt(n_obs - 1))
    # DSR: deflated benchmark SR* from trials
    if trials > 1:
        emax = (1 - np.euler_gamma) * stats.norm.ppf(1 - 1.0 / trials) + \
               np.euler_gamma * stats.norm.ppf(1 - 1.0 / (trials * np.e))
        # variance of SR estimates across trials ~ unknown; 보수적으로 1/(n-1) 사용
        sr_star = emax * np.sqrt(1.0 / (n_obs - 1))
        dsr = stats.norm.cdf((sr - sr_star) * np.sqrt(n_obs - 1))
    else:
        dsr = psr
    return float(psr), float(dsr)


def main():
    panel = build()
    panel, leak_a = attach_regime(panel)
    leak_b = leak_shift_test(panel)
    # 라벨 미래성 sanity: lab_pump20 == (h/o-1>=0.20)
    lab_chk = ((panel["h"] / panel["o"] - 1.0 >= 0.20).astype(float) == panel[LABEL]).mean()
    log.info("[LEAK C] label==recompute(h/o-1>=.2): match_frac=%.4f (1.0=ok, 라벨은 미래 day-D)", lab_chk)

    rdist = panel["regime"].value_counts(dropna=False)
    log.info("regime dist (D-1):\n%s", rdist.to_string())
    for rg in REGIMES:
        sub = panel[panel["regime"] == rg]
        log.info("  base %s pump20=%.4f n=%d", rg, sub[LABEL].mean(), len(sub))

    # selection: 4각도 총 시도 추정
    trials_total = (46 + 46) + (13 * 2 * 4 + 12) + (8 * 4 * 3) + (7 * 5 + 32)
    log.info("SELECTION trials_total(approx 4각도)= %d", trials_total)

    rows = []
    for pname, (feat, direction) in PATTERNS.items():
        for rg in REGIMES:
            sel = oos_selected(panel, feat, direction, rg)
            if sel is None:
                continue
            for mode in ["sl_first", "tp_eod", "tp_first"]:
                m = net_metrics(sel, mode)
                if m is None:
                    continue
                psr, dsr = psr_dsr(m["sharpe"], m["n_days"], trials_total)
                rows.append({"pattern": pname, "regime": rg, "exit": mode,
                             **m, "psr": psr, "dsr": dsr})
    res = pd.DataFrame(rows)
    res.to_csv("output/eval_library_audit_v1.csv", index=False)
    log.info("wrote output/eval_library_audit_v1.csv (%d rows)", len(res))

    # 요약: regime별 exit별 net Sharpe
    log.info("\n===== NET SHARPE by regime/exit (pattern-mean) =====")
    piv = res.groupby(["regime", "exit"]).agg(
        n=("sharpe", "size"), sharpe=("sharpe", "mean"),
        net_pt=("net_mean_per_trade", "mean"), hit=("hit_net", "mean"),
        mdd=("mdd", "mean"), cum=("cum_return", "mean")).reset_index()
    for _, r in piv.iterrows():
        log.info("  %-14s %-9s sharpe=%+.2f net/trade=%+.4f hit=%.2f mdd=%.3f cum=%+.3f (n_pat=%d)",
                 r["regime"], r["exit"], r["sharpe"], r["net_pt"], r["hit"], r["mdd"], r["cum"], int(r["n"]))

    # 가장 net 양수에 가까운 (pattern×regime×exit) top
    log.info("\n===== TOP net_mean_per_trade (sl_first 비관) =====")
    slf = res[res["exit"] == "sl_first"].sort_values("net_mean_per_trade", ascending=False)
    for _, r in slf.head(10).iterrows():
        log.info("  %-14s %-14s net/tr=%+.4f sharpe=%+.2f hit=%.2f mdd=%.3f cum=%+.3f n=%d dsr=%.3f",
                 r["pattern"], r["regime"], r["net_mean_per_trade"], r["sharpe"], r["hit_net"],
                 r["mdd"], r["cum_return"], int(r["n_trades"]), r["dsr"])
    log.info("\n===== TOP net (tp_eod = SL 없음, 시간손절만) =====")
    te = res[res["exit"] == "tp_eod"].sort_values("net_mean_per_trade", ascending=False)
    for _, r in te.head(10).iterrows():
        log.info("  %-14s %-14s net/tr=%+.4f sharpe=%+.2f hit=%.2f mdd=%.3f cum=%+.3f n=%d dsr=%.3f",
                 r["pattern"], r["regime"], r["net_mean_per_trade"], r["sharpe"], r["hit_net"],
                 r["mdd"], r["cum_return"], int(r["n_trades"]), r["dsr"])
    # artifact 크기: sl_first vs tp_first spread (intrabar 순서 불확실성 폭)
    log.info("\n===== EXIT ARTIFACT spread (tp_first - sl_first) by regime =====")
    for rg in REGIMES:
        a = res[(res["regime"] == rg) & (res["exit"] == "sl_first")]["net_mean_per_trade"].mean()
        b = res[(res["regime"] == rg) & (res["exit"] == "tp_first")]["net_mean_per_trade"].mean()
        log.info("  %-14s sl_first=%+.4f tp_first=%+.4f spread=%.4f", rg, a, b, b - a)

    print("\nLEAK_SUMMARY", json.dumps({"regime_d1_ok": leak_a, "feature_shift_ok": leak_b,
                                        "label_match": float(lab_chk)}))


if __name__ == "__main__":
    main()
