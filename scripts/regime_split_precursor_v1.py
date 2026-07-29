"""Regime-split coin precursor library (Angle 2).

목표: 확립된 코인 선행패턴(qv_surge_30d, bounce_off_7d_low, short-momentum
ret_3d/7d, ATR/RV)을 BTC regime 4종(bull_quiet/bull_volatile/bear_quiet/
bear_volatile)으로 분할해 각 regime에서 OOS lift / net PnL이 어떻게 다른지
전체 히스토리로 측정. "이 regime에선 이 패턴 set이 산다"를 만든다.

핵심 질문:
  - 같은 패턴이 bull_volatile vs bear_quiet에서 작동이 다른가?
  - 어떤 패턴이 어떤 regime 전용인가?
  - regime별 새 패턴(shallow tree) 탐색.

★ LEAK 방어 (이 프로젝트 same-day leak 2번 전적):
  - precursor feature: market별 시계열에서 .shift(1) 후 값만 사용 → day D row는
    D-1까지 정보. (univariate_precursor_lift_v1.py와 동일 빌더 재사용.)
  - 라벨 pump20/pump15: day D의 open/high만 (타겟, 미래).
  - ★ BTC regime: features.compute_btc_features는 day t 값으로 regime을 계산하므로,
    그대로 join하면 day-D close가 regime에 들어간다 = leak. 따라서 BTC regime을
    timestamp 기준 join 후 **market 무관 시계열로 1일 shift** → day D는 D-1 regime을 본다.
    (D-1 regime이 "오늘 어떤 패턴 set을 쓸지" 결정하는 게 운영과도 정합.)
  - net PnL: 진입가 = open_D (regime/feature 다 D-1까지로 결정 후 시가 진입),
    TP/SL은 day D high/low로 판정(미래, 청산 시점), 왕복 0.15% 차감.

검증:
  - regime별 시간순 walk-forward OOS lift (embargo 5d). 전역 fold 경계를 공유해
    각 regime sub-population에서 OOS precision 측정 → regime 간 비교 공정.
  - net PnL: regime×feature top-decile 진입 시 평균 net 수익 + win rate.
  - selection bias: 시도 조합 수 = features × labels × regimes 기록.

사용:
    python scripts/regime_split_precursor_v1.py
    python scripts/regime_split_precursor_v1.py --limit-markets 40   # 개발
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.database import list_markets, load_candles
from data.market_universe import is_excluded_signal_market
from signals.features import compute_btc_features
# 재사용: 동일한 leak-free feature 빌더
from scripts.univariate_precursor_lift_v1 import (
    build_market_features,
    add_cross_sectional,
    completed_upbit_d1_mask,
    expanding_purged_date_folds,
)

DB_PATH = "data/upbit_d1.db"
OUT_DIR = Path("output")
EPS = 1e-12
ROUND_TRIP_COST = 0.0015  # 왕복 0.15% (ledger/config.py와 일치)
TP_PCT = 0.10             # +10% 익절 (placeholder, ledger 기본)
SL_PCT = 0.05             # -5% 손절

log = logging.getLogger("regime_split")

REGIMES = ["bull_quiet", "bull_volatile", "bear_quiet", "bear_volatile"]

# 확립된 top 선행 (직전 발굴 univariate_precursor_lift_v1.csv 기준)
ESTABLISHED = [
    "f_qv_surge_30d",       # OOS lift 3.0~3.7
    "f_qv_surge_7d",
    "f_bounce_off_7d_low",  # 3.2~3.7
    "f_ret_3d",             # short momentum ~3
    "f_ret_7d",
    "f_roc_7d",
    "f_ret_14d",
    "f_atr_pct_14",         # ATR regime 2.2~2.6
    "f_rv_7d",
    "f_rv_21d",
    "f_atr_xs_decile",
    "f_log_qv",             # 유동성 baseline (top50 base rate)
    "f_qv_rank_pct",
]
LABELS = ["lab_pump20", "lab_pump15"]


# ============================================================================
# 1. Panel build (D-1 features) + BTC regime (D-1) join
# ============================================================================
def build_panel(limit_markets: int | None) -> pd.DataFrame:
    markets = [
        market
        for market in list_markets(DB_PATH)
        if not is_excluded_signal_market(str(market))
    ]
    if limit_markets:
        markets = markets[:limit_markets]
    log.info("loading %d markets", len(markets))
    frames = []
    for i, m in enumerate(markets):
        df = load_candles(DB_PATH, m)
        if df is None or len(df) < 70:
            continue
        df = df.copy()
        df["market"] = m
        frames.append(build_market_features(df))
        if (i + 1) % 50 == 0:
            log.info("  %d/%d markets", i + 1, len(markets))
    panel = pd.concat(frames, ignore_index=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel["date"] = panel["timestamp"].dt.date
    panel = panel.sort_values(["date", "market"]).reset_index(drop=True)
    return panel


def attach_btc_regime(panel: pd.DataFrame) -> pd.DataFrame:
    """BTC regime을 D-1 시점으로 panel에 부착 (leak 방어 shift).

    features.compute_btc_features는 day t의 close로 regime을 계산 → 그대로 join하면
    same-day leak. timestamp별 regime 시계열을 만들고 1일 shift해서 day D에는
    D-1 regime이 붙도록 한다.
    """
    btc = load_candles(DB_PATH, "KRW-BTC")
    btc_feat = compute_btc_features(btc)  # timestamp + btc_* (regime은 day t값)
    btc_feat = btc_feat[["timestamp", "btc_regime", "btc_ma_distance",
                          "btc_intensity_d"]].copy()
    btc_feat["timestamp"] = pd.to_datetime(btc_feat["timestamp"])
    btc_feat = btc_feat.sort_values("timestamp").reset_index(drop=True)

    # ★ leak 방어: regime/btc feature를 1일 shift → day D row가 D-1 regime을 본다
    for c in ["btc_regime", "btc_ma_distance", "btc_intensity_d"]:
        btc_feat[f"{c}_dm1"] = btc_feat[c].shift(1)
    btc_dm1 = btc_feat[["timestamp", "btc_regime_dm1",
                        "btc_ma_distance_dm1", "btc_intensity_d_dm1"]]
    p = panel.merge(btc_dm1, on="timestamp", how="left")
    p = p.rename(columns={"btc_regime_dm1": "regime"})
    return p


# ============================================================================
# 2. Regime-shared walk-forward folds → per-regime OOS lift
# ============================================================================
def regime_walk_forward(panel: pd.DataFrame, feat: str, label: str,
                        regime: str, n_folds: int = 5,
                        embargo_days: int = 5) -> dict:
    """전역 fold 경계를 공유하되 train/test 둘 다 해당 regime row로 필터.

    train(해당 regime)에서 top/bot decile cutoff를 정하고, test(해당 regime)에서
    precision 측정. base rate도 test(해당 regime) 기준 → regime 내 lift.
    """
    base_cols = [feat, label, "date", "regime"]
    d = panel[base_cols].dropna(subset=[feat, label, "regime"])
    d = d[d["regime"] == regime]
    if len(d) < 400:
        return _empty_wf()

    all_dates = np.sort(panel["date"].unique())
    oos_precs, oos_ns, base_rates, sel_dirs = [], [], [], []

    for train_dates, test_dates in expanding_purged_date_folds(
        all_dates, n_folds, embargo_days
    ):
        tr = d[d["date"].isin(train_dates)]
        te = d[d["date"].isin(test_dates)]
        if len(tr) < 200 or len(te) < 100:
            continue
        tr_top_cut = tr[feat].quantile(0.9)
        tr_bot_cut = tr[feat].quantile(0.1)
        top_rate = tr[tr[feat] >= tr_top_cut][label].mean()
        bot_rate = tr[tr[feat] <= tr_bot_cut][label].mean()
        if (np.isnan(top_rate) and np.isnan(bot_rate)):
            continue
        if (top_rate if not np.isnan(top_rate) else -1) >= \
           (bot_rate if not np.isnan(bot_rate) else -1):
            sel = te[te[feat] >= tr_top_cut]
            sel_dirs.append("high")
        else:
            sel = te[te[feat] <= tr_bot_cut]
            sel_dirs.append("low")
        if len(sel) < 20:
            continue
        oos_precs.append(sel[label].mean())
        oos_ns.append(len(sel))
        base_rates.append(te[label].mean())

    if not oos_precs:
        return _empty_wf()
    w = np.array(oos_ns, dtype=float)
    oos_prec = float(np.average(oos_precs, weights=w))
    base = float(np.average(base_rates, weights=w))
    dir_mode = max(set(sel_dirs), key=sel_dirs.count) if sel_dirs else "na"
    return {
        "oos_prec": oos_prec,
        "oos_base": base,
        "oos_lift": oos_prec / base if base > 0 else np.nan,
        "oos_n": int(w.sum()),
        "n_folds_used": len(oos_precs),
        "signal_dir": dir_mode,
    }


def _empty_wf() -> dict:
    return {"oos_prec": np.nan, "oos_base": np.nan, "oos_lift": np.nan,
            "oos_n": 0, "n_folds_used": 0, "signal_dir": "na"}


def _net_pnl_on_rows(rows: pd.DataFrame) -> dict:
    """진입 row 집합 → net PnL 지표 (진입가 open_D, TP/SL/EOD, 왕복 0.15% 차감).

    rows: open_D/high_D/low_D/close_D/date 보유. day-D OHLC는 청산 시점(미래)이라 OK.
    """
    trades = []
    for _, r in rows.iterrows():
        o, hi, lo, cl = r["open_D"], r["high_D"], r["low_D"], r["close_D"]
        if not np.isfinite(o) or o <= 0:
            continue
        tp_px = o * (1 + TP_PCT)
        sl_px = o * (1 - SL_PCT)
        if lo <= sl_px:           # SL 우선 (보수, 같은 봉)
            gross, outcome = -SL_PCT, "sl"
        elif hi >= tp_px:
            gross, outcome = TP_PCT, "tp"
        else:
            gross, outcome = cl / o - 1.0, "eod"
        trades.append((r["date"], gross - ROUND_TRIP_COST, outcome))
    if not trades:
        return {"net_mean": np.nan, "net_sharpe": np.nan, "n_trades": 0,
                "win_rate": np.nan, "tp_rate": np.nan, "sl_rate": np.nan}
    td = pd.DataFrame(trades, columns=["date", "net", "outcome"])
    daily = td.groupby("date")["net"].mean()
    mu, sigma = daily.mean(), daily.std()
    sharpe = float(mu / sigma * np.sqrt(365)) if sigma and sigma > 0 else np.nan
    return {
        "net_mean": float(td["net"].mean()),
        "net_sharpe": sharpe,
        "n_trades": int(len(td)),
        "win_rate": float((td["net"] > 0).mean()),
        "tp_rate": float((td["outcome"] == "tp").mean()),
        "sl_rate": float((td["outcome"] == "sl").mean()),
    }


# ============================================================================
# 3. Net PnL — regime×feature top-decile 진입 시뮬 (walk-forward OOS)
# ============================================================================
def regime_net_pnl(panel: pd.DataFrame, feat: str, label: str, regime: str,
                   direction: str, n_folds: int = 5,
                   embargo_days: int = 5) -> dict:
    """OOS 진입 시뮬: train regime에서 cutoff → test regime selected에 진입.

    진입가 = open_D (regime/feature 모두 D-1결정). 청산:
      - day D high가 open*(1+TP) 도달 → +TP
      - day D low가 open*(1-SL) 도달 → -SL  (둘 다면 SL 먼저 = 보수)
      - 아니면 close_D로 청산 (eod return)
    모든 trade에 왕복 0.15% 차감. 거래일별 평균 net으로 Sharpe 근사.
    """
    cols = [feat, "date", "regime", "open_D", "high_D", "low_D", "close_D"]
    d = panel[cols].dropna(subset=[feat, "regime", "open_D", "high_D",
                                   "low_D", "close_D"])
    d = d[d["regime"] == regime]
    if len(d) < 400:
        return {"net_mean": np.nan, "net_sharpe": np.nan, "n_trades": 0,
                "win_rate": np.nan, "tp_rate": np.nan, "sl_rate": np.nan}

    all_dates = np.sort(panel["date"].unique())
    sel_frames = []  # OOS selected rows across folds
    for train_dates, test_dates in expanding_purged_date_folds(
        all_dates, n_folds, embargo_days
    ):
        tr = d[d["date"].isin(train_dates)]
        te = d[d["date"].isin(test_dates)]
        if len(tr) < 200 or len(te) < 100:
            continue
        if direction == "high":
            cut = tr[feat].quantile(0.9)
            sel_frames.append(te[te[feat] >= cut])
        else:
            cut = tr[feat].quantile(0.1)
            sel_frames.append(te[te[feat] <= cut])
    if not sel_frames:
        return {"net_mean": np.nan, "net_sharpe": np.nan, "n_trades": 0,
                "win_rate": np.nan, "tp_rate": np.nan, "sl_rate": np.nan}
    return _net_pnl_on_rows(pd.concat(sel_frames, ignore_index=True))


# ============================================================================
# 4. Per-regime setup discovery (shallow tree, train-only)
# ============================================================================
def discover_regime_setups(panel: pd.DataFrame, regime: str, label: str,
                           feats: list[str], n_folds: int = 5,
                           embargo_days: int = 5, max_depth: int = 3) -> list[dict]:
    """regime별 shallow decision tree로 새 패턴 탐색.

    train(regime)에서 tree 학습 → leaf rule 추출 → test(regime)에서 OOS precision/lift.
    selection: tree는 train-only로 학습, leaf 평가는 OOS.
    """
    try:
        from sklearn.tree import DecisionTreeClassifier, _tree
    except Exception:
        return []
    ohlc = ["open_D", "high_D", "low_D", "close_D"]
    has_ohlc = all(c in panel.columns for c in ohlc)
    cols = feats + [label, "date", "regime"] + (ohlc if has_ohlc else [])
    d = panel[cols].dropna(subset=[label, "regime"])
    d = d[d["regime"] == regime]
    if len(d) < 600:
        return []

    all_dates = np.sort(panel["date"].unique())
    folds = expanding_purged_date_folds(all_dates, n_folds, embargo_days)
    # 마지막 fold 사용 (가장 큰 train, OOS test) — discovery는 1 fold OOS로 충분
    if not folds:
        # fold가 부족하면 70/30 시간분할
        cut = int(len(all_dates) * 0.7)
        train_dates = set(all_dates[:cut])
        test_dates = set(all_dates[cut + embargo_days:])
    else:
        train_dates, test_dates = folds[-1]
    tr = d[d["date"].isin(train_dates)]
    te = d[d["date"].isin(test_dates)]
    if len(tr) < 400 or len(te) < 150:
        return []

    Xtr = tr[feats].fillna(tr[feats].median())
    ytr = tr[label].values
    if ytr.sum() < 15:
        return []
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=max(80, int(len(tr) * 0.005)),
        class_weight="balanced", random_state=0,
    )
    clf.fit(Xtr, ytr)

    tree = clf.tree_
    feat_names = feats
    base_te = te[label].mean()
    results = []

    def recurse(node, conds):
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            name = feat_names[tree.feature[node]]
            thr = tree.threshold[node]
            recurse(tree.children_left[node],
                    conds + [(name, "<=", thr)])
            recurse(tree.children_right[node],
                    conds + [(name, ">", thr)])
        else:
            if not conds:
                return
            # ★ class_weight='balanced'면 tree.value는 가중 카운트라 raw rate가 아님.
            #   leaf 규칙을 train/test 원본 row에 직접 적용해 raw 카운트/precision 계산.
            tr_mask = pd.Series(True, index=tr.index)
            te_mask = pd.Series(True, index=te.index)
            for (nm, op, th) in conds:
                tr_col = tr[nm].fillna(tr[nm].median())
                te_col = te[nm].fillna(te[nm].median())
                tr_mask &= (tr_col <= th) if op == "<=" else (tr_col > th)
                te_mask &= (te_col <= th) if op == "<=" else (te_col > th)
            tr_sub = tr[tr_mask]
            te_sub = te[te_mask]
            n_train = len(tr_sub)
            pos_train = tr_sub[label].mean() if n_train > 0 else 0.0
            base_tr = tr[label].mean()
            # train에서 base 대비 lift 1.3x 이상 + 충분한 표본
            if n_train < 80 or pos_train <= base_tr * 1.3:
                return
            if len(te_sub) < 30:
                return
            oos_prec = te_sub[label].mean()
            rule = " AND ".join(
                f"{nm} {op} {th:+.4g}" for (nm, op, th) in conds)
            rec = {
                "regime": regime, "label": label, "rule": rule,
                "n_train": int(n_train), "pos_train": float(pos_train),
                "train_lift": float(pos_train / base_tr) if base_tr > 0 else np.nan,
                "n_oos": int(len(te_sub)), "oos_prec": float(oos_prec),
                "oos_base": float(base_te),
                "oos_lift": float(oos_prec / base_te) if base_te > 0 else np.nan,
            }
            # net PnL (OOS leaf 진입 시뮬, TP/SL/EOD, 왕복 0.15% 차감)
            if has_ohlc:
                rec.update(_net_pnl_on_rows(te_sub))
            results.append(rec)

    recurse(0, [])
    return results


# ============================================================================
# main
# ============================================================================
def _configure_cli_logging() -> None:
    """Configure stdout logging only for direct CLI execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    _configure_cli_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-markets", type=int, default=None)
    ap.add_argument("--universe-top", type=int, default=None,
                    help="D-1 qv top-N 유니버스 제한")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    panel = build_panel(args.limit_markets)
    panel = add_cross_sectional(panel)
    panel = attach_btc_regime(panel)

    # net PnL용 원본 OHLC (day D, 청산 시점) — feature와 분리해 명시 보관
    # build_market_features는 OHLC를 안 남기므로 다시 붙인다 (timestamp+market)
    raw = []
    markets = panel["market"].unique()
    for m in markets:
        df = load_candles(DB_PATH, m)
        if df is None or len(df) == 0:
            continue
        df = df[["timestamp", "open", "high", "low", "close"]].copy()
        df["market"] = m
        raw.append(df)
    rawdf = pd.concat(raw, ignore_index=True)
    rawdf["timestamp"] = pd.to_datetime(rawdf["timestamp"])
    rawdf = rawdf[
        completed_upbit_d1_mask(rawdf["timestamp"])
    ].copy()
    rawdf = rawdf.rename(columns={"open": "open_D", "high": "high_D",
                                  "low": "low_D", "close": "close_D"})
    panel = panel.merge(rawdf, on=["market", "timestamp"], how="left")

    log.info("panel rows=%d markets=%d dates=%d",
             len(panel), panel["market"].nunique(), panel["date"].nunique())

    # regime 분포 (D-1 기준)
    rdist = panel["regime"].value_counts(dropna=False)
    log.info("regime dist (coin-day, D-1 regime):\n%s", rdist.to_string())
    for lab in LABELS:
        log.info("global base %s = %.4f", lab, panel[lab].mean())
        for rg in REGIMES:
            sub = panel[panel["regime"] == rg]
            log.info("  regime=%-14s base=%.4f n=%d", rg, sub[lab].mean(), len(sub))

    n_tries = len(ESTABLISHED) * len(LABELS) * len(REGIMES)
    log.info("SELECTION BIAS: established=%d × labels=%d × regimes=%d = %d "
             "regime-lift tries (+ per-regime tree discovery)",
             len(ESTABLISHED), len(LABELS), len(REGIMES), n_tries)

    # ---- 4a. regime-split lift + net PnL ----
    rows = []
    for lab in LABELS:
        for rg in REGIMES:
            for feat in ESTABLISHED:
                if feat not in panel.columns:
                    continue
                wf = regime_walk_forward(panel, feat, lab, rg)
                if wf["oos_n"] == 0:
                    continue
                pnl = regime_net_pnl(panel, feat, lab, rg, wf["signal_dir"])
                row = {"feature": feat, "label": lab, "regime": rg}
                row.update(wf)
                row.update(pnl)
                rows.append(row)

    lift_df = pd.DataFrame(rows)
    lift_df = lift_df.sort_values(["label", "regime", "oos_lift"],
                                  ascending=[True, True, False])
    lift_path = OUT_DIR / "regime_split_precursor_lift_v1.csv"
    lift_df.to_csv(lift_path, index=False)
    log.info("wrote %s (%d rows)", lift_path, len(lift_df))

    # ---- 4b. per-regime setup discovery ----
    disc_feats = [f for f in ESTABLISHED if f in panel.columns]
    disc_rows = []
    for lab in LABELS:
        for rg in REGIMES:
            res = discover_regime_setups(panel, rg, lab, disc_feats)
            disc_rows.extend(res)
    disc_df = pd.DataFrame(disc_rows)
    if len(disc_df):
        disc_df = disc_df.sort_values(["label", "regime", "oos_lift"],
                                      ascending=[True, True, False])
        disc_path = OUT_DIR / "regime_split_setup_discovery_v1.csv"
        disc_df.to_csv(disc_path, index=False)
        log.info("wrote %s (%d rows)", disc_path, len(disc_df))

    # ---- 콘솔 요약 ----
    log.info("\n========== REGIME-SPLIT OOS LIFT (label=lab_pump20, oos_n>=80) ==========")
    sub = lift_df[(lift_df["label"] == "lab_pump20") & (lift_df["oos_n"] >= 80)]
    for rg in REGIMES:
        ss = sub[sub["regime"] == rg].sort_values("oos_lift", ascending=False)
        log.info("--- regime=%s ---", rg)
        for _, r in ss.head(8).iterrows():
            log.info("  %-22s dir=%-4s OOS_lift=%.2f prec=%.3f base=%.3f n=%d | "
                     "net_mean=%+.4f sharpe=%s win=%.2f tp=%.2f sl=%.2f nt=%d",
                     r["feature"], r["signal_dir"], r["oos_lift"], r["oos_prec"],
                     r["oos_base"], int(r["oos_n"]),
                     r["net_mean"],
                     f"{r['net_sharpe']:+.2f}" if pd.notna(r["net_sharpe"]) else "  nan",
                     r["win_rate"], r["tp_rate"], r["sl_rate"], int(r["n_trades"]))

    if len(disc_df):
        log.info("\n========== PER-REGIME DISCOVERED SETUPS (top by OOS lift) ==========")
        for rg in REGIMES:
            ss = disc_df[(disc_df["regime"] == rg) &
                         (disc_df["label"] == "lab_pump20")].head(3)
            for _, r in ss.iterrows():
                nm_pnl = (f"net={r['net_mean']:+.4f} win={r['win_rate']:.2f} "
                          f"tp={r['tp_rate']:.2f} sl={r['sl_rate']:.2f} nt={int(r['n_trades'])}"
                          if "net_mean" in r and pd.notna(r.get("net_mean")) else "net=na")
                log.info("  [%s] lift=%.2f prec=%.3f base=%.3f n_oos=%d | %s :: %s",
                         rg, r["oos_lift"], r["oos_prec"], r["oos_base"],
                         int(r["n_oos"]), nm_pnl, r["rule"])


if __name__ == "__main__":
    main()
