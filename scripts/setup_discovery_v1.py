"""Setup Discovery v1 — shallow tree per head → interpretable leaf rules.

목적 (사용자 setup library 방향):
  multi-head model 의 score 가 "왜 높은지" 를 사람이 볼 수 있는 setup 으로 변환.
  각 head 별로 shallow decision tree 학습, leaf 가 setup 후보.
  WF OOF 로 leaf 의 lift / fail / support 검증.

heads (사용자 확정):
  h2_hit_3_4h   — 4h 안 +3% hit (실사용 핵심)
  h5_tail_20    — +20% rare tail
  h6_hit_5_24h  — 24h 안 +5% hit (사용자가 말한 "최소 몇% 오를 후보")

method:
  - sklearn DecisionTreeClassifier(max_depth=3, min_samples_leaf=500, class_weight='balanced')
  - leaves: depth=3 → 최대 8 leaf
  - per leaf: support_val, prec_val, lift_val, fail_eod_val
  - selection bias 방어: train 통계 + val 통계 둘 다 표시
  - cross-fold robust setup: 3+ fold 에서 lift > 1.5 + support 충분

universe 비교: all vs top100

산출물:
  output/setup_discovery_v1.csv   — per (head, fold, leaf) 결과
  stdout — top setups + rule 출력

사용:
    python scripts/setup_discovery_v1.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.database import list_markets, load_candles
from data.market_universe import signal_eligible_markets
from signals.features import assemble_training_panel, compute_btc_features
from signals.labels_distribution import HEADS, compute_distribution_labels, MAX_BARS
from signals.models.xgb_phase1 import EXCLUDE_COLS
from signals.validate import PurgedWalkForward


LEAK_COLS = {"net_under_tp", "max_return", "label", "label_tail",
             "next_open", "next_high", "next_low", "next_close",
             "next_max_return", "next_eod_return", "next_max_dd"}

# 우선순위 head (사용자 확정)
TARGET_HEADS = ["h2_hit_3_4h", "h5_tail_20", "h6_hit_5_24h"]

TREE_PARAMS = dict(
    max_depth=3,
    min_samples_leaf=500,
    class_weight="balanced",
    random_state=42,
)


def feature_cols_from(df, label_cols):
    EXTRA_DROP = {"quote_volume_d1", "date_only", "liq_rank_daily",
                   "highs", "lows", "n_bars", "btc_regime_4h", "label_date",
                   "eod_ret_next"}
    cols = []
    for c in df.columns:
        if c in EXCLUDE_COLS or c in label_cols or c in HEADS:
            continue
        if c in LEAK_COLS or c in EXTRA_DROP:
            continue
        if c.startswith("next_"):
            continue
        dt = df[c].dtype
        if pd.api.types.is_object_dtype(dt) or "datetime" in str(dt):
            continue
        cols.append(c)
    return cols


def build_4h_panel_for_labels(candles_4h, btc_regime_map):
    rows = []
    for market, df in candles_4h.items():
        if df is None or len(df) < 1:
            continue
        df = df.sort_values("timestamp").copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["bar_date"] = (df["timestamp"] - pd.Timedelta(hours=9)).dt.date
        for date, g in df.groupby("bar_date", sort=False):
            g2 = g.sort_values("timestamp")
            n = min(len(g2), MAX_BARS)
            if n < 1:
                continue
            opens = g2["open"].values.astype(float)
            closes = g2["close"].values.astype(float)
            highs = g2["high"].values.astype(float)[:MAX_BARS]
            lows = g2["low"].values.astype(float)[:MAX_BARS]
            highs_p = np.full(MAX_BARS, np.nan)
            lows_p = np.full(MAX_BARS, np.nan)
            highs_p[:n] = highs[:n]
            lows_p[:n] = lows[:n]
            rows.append({
                "market": market, "date_only": date,
                "open_4h": float(opens[0]), "close_4h": float(closes[-1]),
                "highs": highs_p, "lows": lows_p,
                "btc_regime_4h": btc_regime_map.get(date, "unknown"),
                "n_bars": n,
                "eod_ret_4h": float(closes[-1]) / float(opens[0]) - 1 if opens[0] > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def extract_leaf_rules(tree: DecisionTreeClassifier, feature_names: list) -> dict:
    """returns dict: leaf_node_id -> list of (feature, op, threshold)"""
    rules = {}
    t = tree.tree_

    def recurse(node, conds):
        if t.children_left[node] == -1 and t.children_right[node] == -1:
            rules[node] = list(conds)
            return
        feat = feature_names[t.feature[node]]
        thr = float(t.threshold[node])
        recurse(t.children_left[node], conds + [(feat, "<=", thr)])
        recurse(t.children_right[node], conds + [(feat, ">", thr)])

    recurse(0, [])
    return rules


def format_rule(conds: list) -> str:
    return " AND ".join(f"{f} {op} {thr:+.3f}" for f, op, thr in conds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upbit-d1", default="data/upbit_d1.db")
    parser.add_argument("--upbit-4h", default="data/upbit_4h.db")
    parser.add_argument("--binance-d1", default="data/binance_d1.db")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--embargo", type=int, default=10)
    parser.add_argument("--holdout", type=int, default=180)
    parser.add_argument("--out-csv", default="output/setup_discovery_v1.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("setup")

    print(f"=== Setup Discovery v1 — shallow tree (depth={TREE_PARAMS['max_depth']}) per head ===\n")
    print(f"Heads: {TARGET_HEADS}\n")

    # ===== load data (same path as sweep) =====
    log.info("loading panel + labels...")
    krw = signal_eligible_markets(list_markets(args.upbit_d1))
    candles_d1 = {m: load_candles(args.upbit_d1, m) for m in krw}
    if Path(args.binance_d1).exists():
        for m in list_markets(args.binance_d1):
            candles_d1[m] = load_candles(args.binance_d1, m)
    candles_d1 = {k: v for k, v in candles_d1.items() if v is not None and len(v) > 30}
    btc_d1 = load_candles(args.upbit_d1, "KRW-BTC")
    panel = assemble_training_panel(candles_d1, btc_d1, normalize=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values(["market", "timestamp"]).reset_index(drop=True)
    panel["date_only"] = panel["timestamp"].dt.date
    panel["quote_volume_d1"] = panel.get("quote_volume", np.nan)

    krw_4h = [
        m
        for m in signal_eligible_markets(list_markets(args.upbit_4h))
        if m.startswith("KRW-")
    ]
    candles_4h = {m: load_candles(args.upbit_4h, m) for m in krw_4h}
    candles_4h = {k: v for k, v in candles_4h.items() if v is not None and len(v) > 0}
    btc_feat = compute_btc_features(btc_d1.copy())
    btc_feat["date_only"] = pd.to_datetime(btc_feat["timestamp"]).dt.date
    btc_regime_map = dict(zip(btc_feat["date_only"], btc_feat["btc_regime"]))
    panel_4h = build_4h_panel_for_labels(candles_4h, btc_regime_map)
    label_df = compute_distribution_labels(panel_4h)
    label_df["market"] = panel_4h["market"].values
    label_df["label_date"] = panel_4h["date_only"].values
    label_df["eod_ret_next"] = panel_4h["eod_ret_4h"].values
    # leak fix
    label_df["date_only"] = (pd.to_datetime(label_df["label_date"]) - pd.Timedelta(days=1)).dt.date

    full = panel.merge(label_df.drop(columns=["label_date"]),
                        on=["market", "date_only"], how="inner")
    full["liq_rank_daily"] = full.groupby("date_only")["quote_volume_d1"].rank(
        method="dense", ascending=False, na_option="bottom"
    )
    feature_cols = feature_cols_from(full, list(HEADS.keys()) + ["eod_ret_next"])
    log.info(f"  joined: {full.shape}, features: {len(feature_cols)}")

    # ===== WF + setup discovery =====
    splitter = PurgedWalkForward(args.n_folds, args.embargo, args.holdout)
    fold_data = []
    for fold, (train_dates, val_dates) in enumerate(splitter.split(full["timestamp"]), 1):
        train_p = full[full["timestamp"].isin(train_dates)]
        val_p = full[full["timestamp"].isin(val_dates)].copy()
        if len(train_p) < 100 or len(val_p) < 50:
            continue
        log.info(f"Fold {fold}: train {len(train_p):,} / val {len(val_p):,}")
        fold_data.append((fold, train_p, val_p))

    rows = []
    for head in TARGET_HEADS:
        log.info(f"\n--- head: {head} ---")
        for fold, train_p, val_p in fold_data:
            df_tr = train_p[train_p[head].notna()].copy()
            df_va = val_p[val_p[head].notna()].copy()
            if len(df_tr) < 500:
                continue
            X_tr = df_tr[feature_cols].astype(float).values
            y_tr = df_tr[head].astype(int).values
            if y_tr.sum() < 50 or (y_tr == 0).sum() < 50:
                continue
            X_va = df_va[feature_cols].astype(float).values
            y_va = df_va[head].astype(int).values
            base_va = float(y_va.mean())

            # NaN handling: trees can't accept NaN. Use median imputation per col (from train).
            col_medians = np.nanmedian(X_tr, axis=0)
            X_tr_imp = np.where(np.isnan(X_tr), col_medians, X_tr)
            X_va_imp = np.where(np.isnan(X_va), col_medians, X_va)

            tree = DecisionTreeClassifier(**TREE_PARAMS)
            tree.fit(X_tr_imp, y_tr)
            rules = extract_leaf_rules(tree, feature_cols)

            leaf_tr = tree.apply(X_tr_imp)
            leaf_va = tree.apply(X_va_imp)

            for leaf_id, conds in rules.items():
                m_tr = leaf_tr == leaf_id
                m_va = leaf_va == leaf_id
                support_tr = int(m_tr.sum())
                support_va = int(m_va.sum())
                if support_tr < 50 or support_va < 30:
                    continue
                pos_tr = float(y_tr[m_tr].mean())
                pos_va = float(y_va[m_va].mean())
                lift_va = pos_va / base_va if base_va > 0 else np.nan
                # fail eod (val, label==0)
                eod_va = df_va["eod_ret_next"].values
                fails_va = m_va & (y_va == 0)
                fail_eod = float(np.nanmean(eod_va[fails_va]) * 100) if fails_va.any() else np.nan
                # universe (top100) within this leaf
                liq_rank = df_va["liq_rank_daily"].values
                m_va_top100 = m_va & (liq_rank <= 100)
                support_va_top100 = int(m_va_top100.sum())
                pos_va_top100 = float(y_va[m_va_top100].mean()) if m_va_top100.any() else np.nan
                lift_va_top100 = pos_va_top100 / base_va if base_va > 0 and not np.isnan(pos_va_top100) else np.nan

                rows.append({
                    "head": head, "fold": fold, "leaf": int(leaf_id),
                    "rule": format_rule(conds),
                    "depth": len(conds),
                    "support_train": support_tr,
                    "support_val": support_va,
                    "support_val_top100": support_va_top100,
                    "pos_train_pct": pos_tr * 100,
                    "pos_val_pct": pos_va * 100,
                    "pos_val_top100_pct": pos_va_top100 * 100 if not np.isnan(pos_va_top100) else np.nan,
                    "base_val_pct": base_va * 100,
                    "lift_val": lift_va,
                    "lift_val_top100": lift_va_top100,
                    "fail_eod_val_pct": fail_eod,
                })

    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    log.info(f"saved {args.out_csv} ({len(df)} setup rows)")

    # =========================================================================
    # 출력
    # =========================================================================
    show_cols = ["head", "fold", "leaf", "depth", "support_val", "pos_train_pct",
                 "pos_val_pct", "lift_val", "fail_eod_val_pct", "rule"]

    def show(title, df_sel, max_rows=15):
        print(f"\n=== {title} ===")
        if len(df_sel) == 0:
            print("(no rows)")
            return
        with pd.option_context("display.max_colwidth", 100):
            print(df_sel[show_cols].head(max_rows).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # 1. Per head: top leaves by val lift (support filter)
    print(f"\n{'='*140}")
    print("1. Per-head best leaves (val lift, support_val ≥ 100)")
    for head in TARGET_HEADS:
        sub = df[(df["head"] == head) & (df["support_val"] >= 100)].sort_values("lift_val", ascending=False)
        show(f"{head} — top 10 leaves", sub.head(10))

    # 2. Cross-fold robust: same head, leaf with positive lift in 3+ folds (similar rule via depth+lift bucket)
    print(f"\n{'='*140}")
    print("2. Cross-fold robust setups (head 별, lift_val ≥ 1.5 in ≥3 folds)")
    for head in TARGET_HEADS:
        sub = df[(df["head"] == head) & (df["lift_val"] >= 1.5) & (df["support_val"] >= 100)]
        # group by similar rule (rough: same root feature)
        sub = sub.copy()
        sub["root_feat"] = sub["rule"].str.split().str[0]
        agg = sub.groupby("root_feat").agg(
            n_folds=("fold", "nunique"),
            mean_lift=("lift_val", "mean"),
            mean_pos_pct=("pos_val_pct", "mean"),
            mean_fail_eod=("fail_eod_val_pct", "mean"),
            sum_support=("support_val", "sum"),
        ).reset_index()
        agg = agg[agg["n_folds"] >= 3].sort_values("mean_lift", ascending=False)
        print(f"\n  -- {head} --")
        if len(agg) == 0:
            print("  (no robust setup — 3+ fold 에서 lift>1.5 인 root feature 없음)")
        else:
            print(agg.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # 3. Top 후보 setup 정리 — high lift + support sufficient
    print(f"\n{'='*140}")
    show("3. Top setups (모든 head, lift ≥ 2.0, support_val ≥ 200)",
         df[(df["lift_val"] >= 2.0) & (df["support_val"] >= 200)
            ].sort_values("lift_val", ascending=False).head(20))

    # 4. h5 deep dive (rare event, sparse leaves expected)
    print(f"\n{'='*140}")
    show("4. h5_tail_20 — 모든 leaf (rare event 진단용)",
         df[df["head"] == "h5_tail_20"].sort_values("lift_val", ascending=False).head(15))

    print(f"\n전체 결과: {len(df)} rows → {args.out_csv}")


if __name__ == "__main__":
    main()
