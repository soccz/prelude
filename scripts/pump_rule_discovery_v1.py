"""Pump Rule Discovery v1 — daily >=20%/>=15% 급등 라벨에 대한 해석 가능한 선행 룰 채굴.

각도 B (rule_mining): setup_discovery_v1.py 방식(shallow decision tree leaf mining,
per-fold walk-forward)을 **사용자 #1 타겟인 일봉 급등 이벤트**에 적용.

타겟 (미래 = day D):
  pump20 = (high_D / open_D - 1) >= 0.20    (주 라벨)
  pump15 = (high_D / open_D - 1) >= 0.15    (보조)

선행조건 (반드시 D-1 및 이전 데이터로만):
  - assemble_training_panel 의 일봉 피처(모두 close/high/low rolling, next_* 없음)
  - 4h-scale range_contraction 피처 (upbit_4h.db, D-1 21:00 KST 마지막 바까지)
  → 라벨을 +1일 shift 하여 feature_date = label_date - 1day 로 join (distribution engine 과 동일 leak-fix)

leak 방어:
  - 라벨은 high_D/open_D (day D). 피처행 날짜 t 에 label(t+1) 을 붙임 → feature_date = D-1
  - same-day(D) high/low/close 절대 미사용. next_* / LEAK_COLS 전부 제외
  - 유니버스(거래대금) rank 도 feature_date(D-1) 기준으로만 계산
  - per-fold: train-only 로 tree fit + leaf rule 추출, val 에서 lift 검증

overfit 거름:
  - train_pos_pct vs val_pos_pct gap < ~10pp 인 leaf 만 robust 후보
  - 3+ fold 에서 같은 root feature 로 lift>임계 반복되면 cross-fold robust

산출물:
  output/pump_rule_discovery_v1.csv  — per (target, scale, fold, leaf)
  stdout — robust 룰 + 발화빈도 + precision/recall 트레이드오프

사용:
    python scripts/pump_rule_discovery_v1.py --target pump20 --max-depth 3
    python scripts/pump_rule_discovery_v1.py --target pump15 --use-4h
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
from data.market_universe import is_excluded_signal_market
from signals.features import assemble_training_panel
from signals.validate import PurgedWalkForward
from scripts.univariate_precursor_lift_v1 import completed_upbit_d1_mask


# 학습 제외 컬럼 (leak + 메타). next_* 는 별도 prefix 필터.
LEAK_COLS = {
    "net_under_tp", "max_return", "label", "label_tail",
    "next_open", "next_high", "next_low", "next_close",
    "next_max_return", "next_eod_return", "next_max_dd",
}
META_COLS = {
    "market", "timestamp", "open", "high", "low", "close", "volume",
    "quote_volume", "quote_volume_d1", "date_only", "label_date",
    "liq_rank_daily", "btc_regime", "btc_regime_id",
    # 타겟/보조 라벨 (절대 피처로 X)
    "pump20", "pump15", "pump_max_return", "eod_ret_D",
}


def eligible_upbit_markets(db_path: str) -> list[str]:
    """Upbit KRW research targets under the canonical live universe."""
    return [
        market
        for market in list_markets(db_path)
        if str(market).startswith("KRW-")
        and not is_excluded_signal_market(str(market))
    ]


def validate_upbit_target_panel(panel: pd.DataFrame) -> None:
    """Fail closed if a target panel escapes the canonical Upbit KRW universe."""
    invalid_targets = sorted(
        {
            str(market)
            for market in panel["market"].dropna().unique()
            if not str(market).startswith("KRW-")
            or is_excluded_signal_market(str(market))
        }
    )
    if invalid_targets:
        raise RuntimeError(
            "pump-rule target panel escaped canonical Upbit universe: "
            f"{invalid_targets}"
        )


def feature_cols_from(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in META_COLS or c in LEAK_COLS:
            continue
        if c.startswith("next_") or c.startswith("pump"):
            continue
        dt = df[c].dtype
        if dt == object or "datetime" in str(dt):
            continue
        cols.append(c)
    return cols


# ============================================================================
# 4h-scale range_contraction 피처 (D-1 마지막 바까지)
# ============================================================================
def build_4h_features(candles_4h: dict, lookback_days: tuple = (3, 7, 14)) -> pd.DataFrame:
    """각 (market, KST date) 에 대해, 그 날 21:00 KST 까지의 4h 바 기반 range_contraction.

    feature_date = KST date. 이 피처는 그 날의 일봉 마감(다음날 09:00) 직전까지 정보만 사용.
    라벨 shift 단계에서 feature_date = D-1 로 매핑되므로 leak 없음.
    """
    rows = []
    for market, df in candles_4h.items():
        if df is None or len(df) < 60:
            continue
        d = df.sort_values("timestamp").copy()
        d["timestamp"] = pd.to_datetime(d["timestamp"])
        # KST 일자: 09:00 시작 → -9h 후 날짜
        d["kst_date"] = (d["timestamp"] - pd.Timedelta(hours=9)).dt.date
        high = d["high"].values.astype(float)
        low = d["low"].values.astype(float)
        close = d["close"].values.astype(float)
        n = len(d)
        # 6 bars per day (4h). range_contraction over last K days vs prior K days.
        # 일별 마지막 인덱스(=그 KST 날의 마지막 바)에서 계산
        out_by_date = {}
        # rolling on bar index
        for K in lookback_days:
            w = K * 6
            for i in range(n):
                if i < 2 * w:
                    continue
                recent_hi = high[i - w + 1:i + 1].max()
                recent_lo = low[i - w + 1:i + 1].min()
                prev_hi = high[i - 2 * w + 1:i - w + 1].max()
                prev_lo = low[i - 2 * w + 1:i - w + 1].min()
                c = close[i]
                if c <= 0:
                    continue
                recent_range = (recent_hi - recent_lo) / c
                prev_range = (prev_hi - prev_lo) / c
                rc = prev_range / (recent_range + 1e-12)  # >1 = 최근이 더 조용 (수축)
                kd = d["kst_date"].values[i]
                out_by_date.setdefault(kd, {})[f"rc4h_{K}d"] = rc
                # 4h 변동성/거래량 비율도 (보조)
                if K == 7:
                    rng = (high[i - w + 1:i + 1] - low[i - w + 1:i + 1]) / c
                    out_by_date[kd]["rc4h_atr_pct"] = float(np.mean(rng))
        for kd, feats in out_by_date.items():
            row = {"market": market, "date_only": kd}
            row.update(feats)
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    # 같은 (market,date) 의 마지막(=가장 최신 바) 값만: groupby last
    out = pd.DataFrame(rows)
    out = out.groupby(["market", "date_only"], as_index=False).last()
    return out


# ============================================================================
# leaf rule 추출
# ============================================================================
def extract_leaf_rules(tree: DecisionTreeClassifier, feature_names: list) -> dict:
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


def root_feature(rule: str) -> str:
    return rule.split()[0] if rule else "?"


# ============================================================================
# main
# ============================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upbit-d1", default="data/upbit_d1.db")
    p.add_argument("--upbit-4h", default="data/upbit_4h.db")
    p.add_argument(
        "--binance-d1",
        default="data/binance_d1.db",
        help=(
            "legacy compatibility only; Binance rows are not valid Upbit "
            "targets and are not loaded"
        ),
    )
    p.add_argument("--target", default="pump20", choices=["pump20", "pump15"])
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--min-leaf", type=int, default=300)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--embargo", type=int, default=10)
    p.add_argument("--holdout", type=int, default=180)
    p.add_argument("--use-4h", action="store_true", help="4h range_contraction 피처 추가")
    p.add_argument("--top-universe", type=int, default=100, help="유니버스 거래대금 상위 N")
    p.add_argument("--out-csv", default="output/pump_rule_discovery_v1.csv")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("pumprule")

    print(f"=== Pump Rule Discovery v1 — target={args.target}, depth<={args.max_depth}, "
          f"min_leaf={args.min_leaf}, 4h={'on' if args.use_4h else 'off'} ===\n")

    # ---- load daily panel ----
    log.info("loading daily panel + features...")
    krw = eligible_upbit_markets(args.upbit_d1)
    candles_d1 = {m: load_candles(args.upbit_d1, m) for m in krw}
    candles_d1 = {k: v for k, v in candles_d1.items() if v is not None and len(v) > 30}
    btc_d1 = load_candles(args.upbit_d1, "KRW-BTC")
    panel = assemble_training_panel(candles_d1, btc_d1, normalize=True)
    validate_upbit_target_panel(panel)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values(["market", "timestamp"]).reset_index(drop=True)
    panel["date_only"] = panel["timestamp"].dt.date
    panel["quote_volume_d1"] = panel.get("quote_volume", np.nan)

    # ---- 4h features (optional) ----
    if args.use_4h and Path(args.upbit_4h).exists():
        log.info("building 4h range_contraction features...")
        krw_4h = eligible_upbit_markets(args.upbit_4h)
        candles_4h = {m: load_candles(args.upbit_4h, m) for m in krw_4h}
        f4h = build_4h_features(candles_4h)
        if len(f4h):
            panel = panel.merge(f4h, on=["market", "date_only"], how="left")
            log.info(f"  merged 4h features: {[c for c in f4h.columns if c.startswith('rc4h')]}")

    # ---- daily pump label on RAW candles (target = day D high/open) ----
    # 라벨은 같은 행(=day D 일봉)에서 계산하지만, join 단계에서 +1 shift 하여
    # feature_date = D-1 로 매핑한다. 여기서는 라벨 테이블만 만든다.
    lab_rows = []
    for market, df in candles_d1.items():
        if df is None or len(df) < 2:
            continue
        d = df.sort_values("timestamp").copy()
        d["timestamp"] = pd.to_datetime(d["timestamp"])
        op = d["open"].values.astype(float)
        hi = d["high"].values.astype(float)
        cl = d["close"].values.astype(float)
        mr = np.where(op > 0, hi / op - 1.0, np.nan)
        eod = np.where(op > 0, cl / op - 1.0, np.nan)
        dd = pd.DataFrame({
            "market": market,
            "label_date": d["timestamp"].dt.date.values,
            "pump_max_return": mr,
            "eod_ret_D": eod,
        })
        dd = dd.loc[
            completed_upbit_d1_mask(d["timestamp"]).to_numpy()
        ].copy()
        lab_rows.append(dd)
    label_df = pd.concat(lab_rows, ignore_index=True)
    label_df["pump20"] = (label_df["pump_max_return"] >= 0.20).astype(int)
    label_df["pump15"] = (label_df["pump_max_return"] >= 0.15).astype(int)
    # *** leak-fix: feature_date = label_date - 1 day ***
    label_df["date_only"] = (pd.to_datetime(label_df["label_date"]) - pd.Timedelta(days=1)).dt.date

    full = panel.merge(
        label_df[["market", "date_only", "pump20", "pump15", "pump_max_return", "eod_ret_D"]],
        on=["market", "date_only"], how="inner",
    )
    full = full.dropna(subset=[args.target]).reset_index(drop=True)

    # 유니버스 rank — feature_date(D-1) 기준
    full["liq_rank_daily"] = full.groupby("date_only")["quote_volume_d1"].rank(
        method="dense", ascending=False, na_option="bottom"
    )

    feat_cols = feature_cols_from(full)
    base_all = float(full[args.target].mean())
    log.info(f"  joined: {full.shape}, features: {len(feat_cols)}, "
             f"base rate({args.target}) = {base_all*100:.2f}%")
    print(f"시도 feature 수: {len(feat_cols)}  (그 중 4h: "
          f"{len([c for c in feat_cols if c.startswith('rc4h')])})")

    # ---- WF leaf mining ----
    splitter = PurgedWalkForward(args.n_folds, args.embargo, args.holdout)
    fold_data = []
    for fold, (tr_dates, va_dates) in enumerate(splitter.split(full["timestamp"]), 1):
        tr = full[full["timestamp"].isin(tr_dates)]
        va = full[full["timestamp"].isin(va_dates)].copy()
        if len(tr) < 500 or len(va) < 200:
            continue
        fold_data.append((fold, tr, va, tr_dates, va_dates))

    rows = []
    n_trees = 0
    for fold, tr, va, tr_dates, va_dates in fold_data:
        y_tr = tr[args.target].astype(int).values
        if y_tr.sum() < 30:
            continue
        X_tr = tr[feat_cols].astype(float).values
        X_va = va[feat_cols].astype(float).values
        y_va = va[args.target].astype(int).values
        base_tr = float(y_tr.mean())
        base_va = float(y_va.mean())
        n_pos_va_total = int(y_va.sum())

        col_med = np.nanmedian(X_tr, axis=0)
        col_med = np.where(np.isnan(col_med), 0.0, col_med)
        X_tr_i = np.where(np.isnan(X_tr), col_med, X_tr)
        X_va_i = np.where(np.isnan(X_va), col_med, X_va)

        tree = DecisionTreeClassifier(
            max_depth=args.max_depth, min_samples_leaf=args.min_leaf,
            class_weight="balanced", random_state=42,
        )
        tree.fit(X_tr_i, y_tr)
        n_trees += 1
        rules = extract_leaf_rules(tree, feat_cols)
        leaf_tr = tree.apply(X_tr_i)
        leaf_va = tree.apply(X_va_i)

        liq = va["liq_rank_daily"].values
        eod_va = va["eod_ret_D"].values

        for leaf_id, conds in rules.items():
            m_tr = leaf_tr == leaf_id
            m_va = leaf_va == leaf_id
            sup_tr = int(m_tr.sum())
            sup_va = int(m_va.sum())
            if sup_tr < args.min_leaf or sup_va < 50:
                continue
            pos_tr = float(y_tr[m_tr].mean())
            pos_va = float(y_va[m_va].mean())
            lift_va = pos_va / base_va if base_va > 0 else np.nan
            gap = (pos_tr - pos_va) * 100
            # recall (이 leaf 가 val 의 전체 pump 중 몇 % 잡나)
            recall_va = float((m_va & (y_va == 1)).sum()) / n_pos_va_total if n_pos_va_total else np.nan
            # 발화빈도: val 행 중 이 leaf 깃발 비율
            fire_rate = sup_va / len(va)
            # top-universe 안에서
            m_top = m_va & (liq <= args.top_universe)
            sup_top = int(m_top.sum())
            pos_top = float(y_va[m_top].mean()) if m_top.any() else np.nan
            lift_top = pos_top / base_va if (base_va > 0 and not np.isnan(pos_top)) else np.nan
            # fail 시 eod (잡았는데 pump 안 난 경우 손익 방향)
            fails = m_va & (y_va == 0)
            fail_eod = float(np.nanmean(eod_va[fails]) * 100) if fails.any() else np.nan

            rows.append({
                "target": args.target, "fold": fold, "leaf": int(leaf_id),
                "depth": len(conds), "rule": format_rule(conds),
                "root_feat": root_feature(format_rule(conds)),
                "support_train": sup_tr, "support_val": sup_va,
                "pos_train_pct": pos_tr * 100, "pos_val_pct": pos_va * 100,
                "base_val_pct": base_va * 100, "lift_val": lift_va,
                "gap_train_val_pp": gap, "recall_val": recall_va * 100,
                "fire_rate_val_pct": fire_rate * 100,
                "support_val_top": sup_top, "pos_val_top_pct": pos_top * 100 if not np.isnan(pos_top) else np.nan,
                "lift_val_top": lift_top, "fail_eod_val_pct": fail_eod,
            })

    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    # append mode if file exists and different target — keep both targets
    if Path(args.out_csv).exists():
        try:
            old = pd.read_csv(args.out_csv)
            old = old[old["target"] != args.target]
            df_save = pd.concat([old, df], ignore_index=True)
        except Exception:
            df_save = df
    else:
        df_save = df
    df_save.to_csv(args.out_csv, index=False)
    log.info(f"saved {args.out_csv} ({len(df)} new rows for {args.target}, {len(df_save)} total)")
    print(f"\n학습한 tree 수(folds): {n_trees}  | leaf rows: {len(df)}")

    # ================= 출력 =================
    if len(df) == 0:
        print("(no leaves passed support filter)")
        return

    show_cols = ["fold", "depth", "support_val", "pos_train_pct", "pos_val_pct",
                 "lift_val", "gap_train_val_pp", "recall_val", "fire_rate_val_pct",
                 "fail_eod_val_pct", "rule"]

    def show(title, sel, n=15):
        print(f"\n=== {title} ===")
        if len(sel) == 0:
            print("(none)")
            return
        with pd.option_context("display.max_colwidth", 90, "display.width", 220):
            print(sel[show_cols].head(n).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print(f"\n{'='*150}")
    print(f"1. Top leaves by val lift (support_val>=50, gap<10pp overfit 필터)")
    robust = df[(df["support_val"] >= 50) & (df["gap_train_val_pp"] < 10) & (df["lift_val"] > 1.0)]
    show("robust (gap<10pp) — lift 순", robust.sort_values("lift_val", ascending=False))

    print(f"\n{'='*150}")
    print(f"2. Cross-fold robust root features (lift_val>=1.5 in >=3 folds)")
    rob2 = df[(df["lift_val"] >= 1.5) & (df["support_val"] >= 50)]
    agg = rob2.groupby("root_feat").agg(
        n_folds=("fold", "nunique"), mean_lift=("lift_val", "mean"),
        mean_pos_pct=("pos_val_pct", "mean"), mean_gap=("gap_train_val_pp", "mean"),
        mean_recall=("recall_val", "mean"), mean_fire=("fire_rate_val_pct", "mean"),
        mean_fail_eod=("fail_eod_val_pct", "mean"), sum_support=("support_val", "sum"),
    ).reset_index()
    agg = agg[agg["n_folds"] >= 3].sort_values("mean_lift", ascending=False)
    if len(agg):
        print(agg.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    else:
        print("(no root feature with lift>=1.5 in >=3 folds)")

    print(f"\n{'='*150}")
    print(f"3. High-recall leaves (recall_val>=10%, lift_val>1.2) — S01~S03 희소 발화 개선 후보")
    hr = df[(df["recall_val"] >= 10) & (df["lift_val"] > 1.2) & (df["gap_train_val_pp"] < 12)]
    show("high recall + lift", hr.sort_values("recall_val", ascending=False))

    print(f"\n{'='*150}")
    print(f"4. range_contraction 룰 (root_feat 에 range_contraction 또는 rc4h)")
    rc = df[df["root_feat"].str.contains("range_contraction|rc4h", na=False)]
    show("range_contraction 관련 leaf", rc.sort_values("lift_val", ascending=False), n=20)
    # 추가: range_contraction 피처가 어느 leaf 든 등장하는지
    rc_any = df[df["rule"].str.contains("range_contraction|rc4h", na=False)]
    print(f"\n  range_contraction/rc4h 가 룰 조건에 등장한 leaf: {len(rc_any)} / {len(df)}")

    print(f"\n전체 결과 → {args.out_csv}")


if __name__ == "__main__":
    main()
