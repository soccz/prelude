"""C4 — R1 챔피언 엔진 *내부* 하이퍼파라미터 sweep (후처리 아님, 엔진 자체 튜닝).

배경:
  이전 7실험은 R1 의 *출력 픽* 을 후처리(랭킹/청산/필터/발사/보유)해서 천장에 닿았다.
  C4 는 R1 엔진 *내부* 하이퍼파라미터를 sweep 해서 챔피언 자체를 개선한다.
  R1 현 구성(재현 대상): rank/XGB head(P(>=10%)/P(<=-5%)) + rr_ratio 정렬 +
  per-fold OOF bucket calibration + 정적 top100 D-1 유니버스 + 15m SL/TP/EOD net.

  ★ 이 파일은 공유 라이브 파일(signals/recommend.py 등) 을 건드리지 않는다.
    엔진 로직을 scripts/downside_head_riskreward_v1.py / r2_challenger_compare_v1.py
    에서 *복제* 해 새 파일에서 sweep 한다 (self-contained, 채택결정 X).

sweep 축 (사전고정 격자 — selection deflate 기록용. one-axis-at-a-time off baseline):
  A) 피처셋   : baseline(24) + add(qv_ma7_vs_ma30, rv_14d, range_contraction_7d,
                bb_width_pctile_60, log_qv 가중대용) / drop ablation.
  B) head hp  : n_estimators / max_depth / learning_rate / min_child_weight / reg_lambda 소격자.
  C) universe : top50 / top100 / top150.
  D) calib bk : 8 / 10 / 15.
  E) RR_EPS   : 1e-2 / 1e-3 / 1e-4 (R1 downside floor 민감도) + tie-break on/off.

비교(모두 동일 OOS·동일 15m 청산경로·동일 비용):
  net_mean · Sharpe · Sortino · MaxDD · hit · %SL · deep-loss(noSL) · precision@3(pump20).
  baseline = 현 R1 구성. 각 셀 Δ 표.

★ LEAK 방어 (same-day leak 2번 전적 — 양보 X):
  - feature = build_market_features 의 .shift(1) (D-1), LEAK_COLS/next_* 제외.
  - 라벨 = day-D open 대비 high/low (미래 타겟) — train 에서만 fit.
  - calibration = per-fold train OOF bucket (train-only). test fold 적용만.
  - purged walk-forward + embargo 5. 유니버스 = D-1 qv rank.
  - 15m 경로는 진입일 D in-trade outcome (진입 결정은 D-1 까지) → leak 아님.
  - "너무 좋으면 leak/overfit" 자가알람: OOS 개선이 walk-forward 내 fold 전반에서
    일관적인지(fold-level dispersion) 확인. in-sample 선택편향 경계.

사용:
    python scripts/cc_r1_core_opt_v1.py
    python scripts/cc_r1_core_opt_v1.py --limit-markets 80   # 개발(빠름)
    python scripts/cc_r1_core_opt_v1.py --axes A,C           # 일부 축만
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# R1 과 동일한 leak-free 빌더 (recommend.py 가 의존하는 바로 그 모듈) — 복제 아닌 재사용.
from scripts.downside_head_riskreward_v1 import (  # noqa: E402
    build_panel,
    add_cross_sectional,
    attach_btc_regime,
    UP_THRESH,
    DN_THRESH,
    LEAK_COLS,
    _oof_bucket_calib,
    _apply_calib,
)
from scripts.recommendation_scorer_v1 import PRECURSOR_FEATURES  # noqa: E402
from scripts.recommender_downside_exit_v1 import simulate_path  # noqa: E402

D1_DB = str(_ROOT / "data" / "upbit_d1.db")
M15_DB = str(_ROOT / "data" / "upbit_15m.db")
OUT = _ROOT / "output"

ROUND_TRIP_COST = 0.0015
HARD_SL = 0.03
TP = 0.05
DEEP_LOSS = -0.05

UP_ANCHOR = "lab_up_10"
DN_ANCHOR = "lab_dn_05"
TOP_K = 3
N_FOLDS = 6
EMBARGO = 5

# ---- baseline (현 R1) 구성 ----
BASE = dict(
    feats="baseline",
    n_estimators=180, max_depth=4, learning_rate=0.05,
    min_child_weight=5, reg_lambda=1.5,
    universe=100, cal_buckets=10, rr_eps=1e-3, tie_break=True,
)

# 라벨 (R1 head 가 쓰는 2개만 학습하면 정렬 가능: up_10, dn_05. dn_10 은 tie-break.)
LABEL_COLS = ["lab_up_10", "lab_dn_05", "lab_dn_10"]

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("cc_r1_opt")


# ============================================================================
# feature set 후보 (add/drop ablation). build_market_features 가 만든 f_* 중에서.
# ============================================================================
def feature_set(name: str, panel_cols) -> list[str]:
    base = [c for c in PRECURSOR_FEATURES if c in panel_cols and c not in LEAK_COLS]
    if name == "baseline":
        return base
    # 후보 추가 feature (panel 에 이미 존재 — build_market_features 산출)
    adds = {
        "add_rv14": ["f_rv_14d"],          # 없으면 무시(아래서 필터)
        "add_range_contraction": ["f_range_contraction_7d"],
        "add_bb_squeeze": ["f_bb_width_pctile_60", "f_bb_width"],
        "add_dist_high": ["f_dist_from_20d_high", "f_dist_from_60d_high", "f_near_20d_high"],
        "add_qvspike": ["f_qv_spiked_yday"],
        "add_all_extra": ["f_range_contraction_7d", "f_bb_width_pctile_60",
                          "f_dist_from_20d_high", "f_dist_from_60d_high",
                          "f_qv_spiked_yday"],
    }
    drops = {
        "drop_xs_decile": [c for c in base if c.endswith("_xs_decile") or c == "f_qv_rank_pct"],
        "drop_rsi": ["f_rsi_14"],
        "drop_streak": ["f_up_streak"],
        "drop_momentum_long": ["f_ret_14d", "f_ret_7d"],
    }
    if name in adds:
        extra = [c for c in adds[name] if c in panel_cols]
        return base + extra
    if name in drops:
        rm = set(drops[name])
        return [c for c in base if c not in rm]
    raise ValueError(f"unknown feature_set {name}")


# ============================================================================
# walk-forward heads — cfg(hp/feats/cal_buckets) parametrized 복제판.
#   downside_head_riskreward_v1.walk_forward_heads 의 fold 분할/calib 규율을 그대로
#   따르되 XGB hp 와 feature/calib_buckets 를 cfg 로 받는다 (sweep 가능).
# ============================================================================
def _xgb_fit_predict(Xtr, ytr, Xte, cfg):
    import xgboost as xgb
    pos = ytr.sum()
    if pos < 12 or len(np.unique(ytr)) < 2:
        return None, None
    spw = float((len(ytr) - pos) / max(pos, 1))
    m = xgb.XGBClassifier(
        n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"], subsample=0.8, colsample_bytree=0.8,
        min_child_weight=cfg["min_child_weight"], reg_lambda=cfg["reg_lambda"],
        scale_pos_weight=spw, n_jobs=4, eval_metric="logloss",
        tree_method="hist", random_state=42,
    )
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1], m


def walk_forward_heads_cfg(panel, feats, label_cols, cfg, n_folds=N_FOLDS, embargo=EMBARGO):
    """expanding purged WF. fold 분할/embargo/OOF-calib 는 R1 빌더와 동일 규율."""
    dates = np.sort(panel["date"].unique())
    n = len(dates)
    start = int(n * 0.35)
    edges_f = np.linspace(start, n, n_folds + 1).astype(int)
    out = []
    for k in range(n_folds):
        tr_end = edges_f[k]
        te_start = tr_end + embargo
        te_end = edges_f[k + 1]
        if te_start >= te_end:
            continue
        tr_d = set(dates[:tr_end]); te_d = set(dates[te_start:te_end])
        tr = panel[panel["date"].isin(tr_d)].copy()
        te = panel[panel["date"].isin(te_d)].copy()
        if len(tr) < 1000 or len(te) < 100:
            continue
        Xtr = tr[feats].replace([np.inf, -np.inf], np.nan)
        med = Xtr.median(); Xtr = Xtr.fillna(med).values
        Xte = te[feats].replace([np.inf, -np.inf], np.nan).fillna(med).values
        res = te.copy(); res["fold"] = k
        for lc in label_cols:
            y = tr[lc].values
            raw_te, m = _xgb_fit_predict(Xtr, y, Xte, cfg)
            if raw_te is None:
                res[f"p_{lc}"] = float(np.nanmean(y)); continue
            raw_tr = m.predict_proba(Xtr)[:, 1]
            ed, hm, base = _oof_bucket_calib(raw_tr, y, cfg["cal_buckets"])
            res[f"p_{lc}"] = _apply_calib(raw_te, ed, hm, base)
        # exp_downside (조건부 하방, tie-break 용) — train-only bucket map.
        ytr_raw, mtmp = _xgb_fit_predict(Xtr, tr["lab_dn_05"].values, Xtr, cfg)
        if ytr_raw is not None:
            tdf = pd.DataFrame({"s": ytr_raw, "lr": tr["down_low_ret"].values}).dropna()
            try:
                tdf["bk"] = pd.qcut(tdf["s"].rank(method="first"), cfg["cal_buckets"],
                                    labels=False, duplicates="drop")
                gg = tdf.groupby("bk").agg(hi=("s", "max"), mean_lr=("lr", "mean"))
                edd = gg["hi"].values; mmap = gg["mean_lr"].to_dict()
                raw_dn_te = mtmp.predict_proba(Xte)[:, 1]
                idx = np.clip(np.searchsorted(edd, raw_dn_te, side="left"), 0, len(edd) - 1)
                res["exp_downside"] = [mmap.get(int(b), float(tdf["lr"].mean())) for b in idx]
            except ValueError:
                res["exp_downside"] = float(tr["down_low_ret"].mean())
        else:
            res["exp_downside"] = np.nan
        out.append(res)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


# ============================================================================
# 15m 경로 net (정렬키/hp 와 무관 → (market,date) 별 1회 캐시).
# ============================================================================
def load_paths(pairs):
    conn = sqlite3.connect(M15_DB)
    paths = {}
    for _, r in pairs.iterrows():
        m, dt = r["market"], pd.Timestamp(r["date"])
        start = dt.strftime("%Y-%m-%d 09:00:00")
        end = (dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d 09:00:00")
        rows = conn.execute(
            "SELECT open,high,low,close FROM candles WHERE market=? AND "
            "timestamp>=? AND timestamp<? ORDER BY timestamp", (m, start, end)
        ).fetchall()
        if rows:
            paths[(m, dt.date())] = rows
    conn.close()
    return paths


def realize_net(bars):
    gross, outcome = simulate_path(bars, HARD_SL, TP, None)
    eod_gross, _ = simulate_path(bars, None, None, None)
    if not np.isfinite(gross):
        return np.nan, "nodata", np.nan
    eod_net = (eod_gross - ROUND_TRIP_COST) if np.isfinite(eod_gross) else np.nan
    return gross - ROUND_TRIP_COST, outcome, eod_net


# ============================================================================
# 정렬 + net 지표
# ============================================================================
def topk_picks(oos, score_col, k, tie_break):
    if tie_break:
        s = oos.sort_values(
            ["date", score_col, "p_lab_dn_10", "p_lab_up_10", "exp_downside"],
            ascending=[True, False, True, False, False])
    else:
        s = oos.sort_values(["date", score_col], ascending=[True, False])
    return s.groupby("date").head(k).copy()


def net_metrics(trades):
    d = trades.dropna(subset=["net"]).copy()
    n = len(d)
    if n == 0:
        return {}
    net = d["net"].values
    daily = d.groupby("date")["net"].mean().sort_index()
    eq = (1 + daily).cumprod(); peak = eq.cummax()
    mdd = float(((eq - peak) / peak).min())
    cum = float(eq.iloc[-1] - 1.0)
    mu = float(net.mean()); sd = float(daily.std())
    dstd = float(daily[daily < 0].std()) if (daily < 0).any() else np.nan
    k5 = max(1, int(np.ceil(0.05 * n)))
    cvar95 = float(np.sort(net)[:k5].mean())
    oc = d["outcome"]
    eod = d["eod_net"].dropna().values if "eod_net" in d.columns else np.array([])
    # fold-level net_mean dispersion (overfit/일관성 진단 — in-sample 선택편향 경계)
    fold_means = d.groupby("fold")["net"].mean() if "fold" in d.columns else pd.Series(dtype=float)
    return dict(
        n=int(n), n_days=int(d["date"].nunique()),
        pct_sl=float((oc == "sl").mean()),
        deep_loss_noSL=float((eod <= DEEP_LOSS).mean()) if len(eod) else np.nan,
        net_mean=mu, net_median=float(np.median(net)),
        hit=float((net > 0).mean()),
        precision_pump20=float(d["pump20_hit"].dropna().mean())
        if d["pump20_hit"].notna().any() else np.nan,
        cvar95=cvar95, worst=float(net.min()), mdd=mdd, cum=cum,
        sharpe=float(mu / sd * np.sqrt(365)) if sd and sd > 0 else np.nan,
        sortino=float(mu / dstd * np.sqrt(365)) if dstd and dstd > 0 else np.nan,
        pct_tp=float((oc == "tp").mean()), pct_eod=float((oc == "eod").mean()),
        fold_net_min=float(fold_means.min()) if len(fold_means) else np.nan,
        fold_net_pos_frac=float((fold_means > 0).mean()) if len(fold_means) else np.nan,
    )


# ============================================================================
# 한 cfg 셀 평가
# ============================================================================
def eval_cell(panel_uni, cfg, bars_map):
    """panel_uni = 이미 universe 적용된 panel. → WF heads → top-3 net 지표."""
    feats = feature_set(cfg["feats"], panel_uni.columns)
    oos = walk_forward_heads_cfg(panel_uni, feats, LABEL_COLS, cfg)
    if oos.empty:
        return None, 0
    up = f"p_{UP_ANCHOR}"; dn = f"p_{DN_ANCHOR}"
    oos = oos.dropna(subset=[up, dn]).copy()
    # 15m 경로 net 부착 (캐시에서)
    keys = list(zip(oos["market"], oos["date"]))
    in_map = [k in bars_map for k in keys]
    oos = oos[in_map].reset_index(drop=True)
    if oos.empty:
        return None, 0
    nets, outs, eods, p20 = [], [], [], []
    for _, r in oos.iterrows():
        net, oc, eod_net = realize_net(bars_map[(r["market"], r["date"])])
        nets.append(net); outs.append(oc); eods.append(eod_net)
        p20.append(1 if (r["up_high_ret"] >= 0.20) else 0)
    oos["net"] = nets; oos["outcome"] = outs; oos["eod_net"] = eods
    oos["pump20_hit"] = p20
    oos = oos.dropna(subset=["net"]).reset_index(drop=True)
    # R1 정렬키 = p_up10 / max(p_dn5, rr_eps)
    oos["R1"] = oos[up] / np.maximum(oos[dn], cfg["rr_eps"])
    picks = topk_picks(oos, "R1", TOP_K, cfg["tie_break"])
    m = net_metrics(picks)
    return m, int(oos["date"].nunique())


# ============================================================================
# sweep 격자 생성 (one-axis-at-a-time off baseline)
# ============================================================================
def build_grid(axes):
    cells = [("baseline", "-", dict(BASE))]
    if "A" in axes:   # feature set
        for fs in ["add_range_contraction", "add_bb_squeeze", "add_dist_high",
                   "add_qvspike", "add_all_extra", "drop_xs_decile", "drop_rsi",
                   "drop_streak", "drop_momentum_long"]:
            c = dict(BASE); c["feats"] = fs
            cells.append(("A_feats", fs, c))
    if "B" in axes:   # head hp
        for key, vals in [
            ("n_estimators", [120, 300]),
            ("max_depth", [3, 5, 6]),
            ("learning_rate", [0.03, 0.10]),
            ("min_child_weight", [3, 10, 20]),
            ("reg_lambda", [0.5, 3.0, 5.0]),
        ]:
            for v in vals:
                c = dict(BASE); c[key] = v
                cells.append(("B_hp", f"{key}={v}", c))
    if "C" in axes:   # universe
        for u in [50, 150]:   # 100 = baseline
            c = dict(BASE); c["universe"] = u
            cells.append(("C_universe", f"top{u}", c))
    if "D" in axes:   # calib buckets
        for b in [8, 15]:     # 10 = baseline
            c = dict(BASE); c["cal_buckets"] = b
            cells.append(("D_calib", f"bk{b}", c))
    if "E" in axes:   # RR_EPS + tie-break
        for e in [1e-2, 1e-4]:   # 1e-3 = baseline
            c = dict(BASE); c["rr_eps"] = e
            cells.append(("E_rreps", f"eps={e:g}", c))
        c = dict(BASE); c["tie_break"] = False
        cells.append(("E_rreps", "no_tiebreak", c))
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-markets", type=int, default=None)
    ap.add_argument("--axes", type=str, default="A,B,C,D,E")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    axes = set(args.axes.split(","))

    # --- panel 1회 빌드 (top150 까지 보관 — universe sweep 위해 maxN 으로 cap) ---
    panel = build_panel(args.limit_markets)
    panel = add_cross_sectional(panel)
    panel = attach_btc_regime(panel)
    panel_max = panel[panel["f_qv_rank"] <= 150].copy()
    panel_max["in_universe"] = True
    log.info("panel(top150 max) rows=%d markets=%d dates=%d",
             len(panel_max), panel_max["market"].nunique(), panel_max["date"].nunique())

    # --- 15m 경로 1회 로드 (top150 union 의 모든 후보 — 정렬키/hp 무관 캐시) ---
    #   ★ 충실성 핵심: WF head 는 *전체 date 범위* 에서 fit 한다 (panel 을 m15_start 로
    #   truncate 하면 fold 경계가 이동해 R2 challenger 와 OOS 가 어긋남). m15_start
    #   필터는 net 실현 시점에만 적용 — 15m 데이터 있는 (market,date) 만 bars_map 에 들어가므로
    #   eval_cell 의 'bars_map 키만 유지' 가 자동으로 OOS 를 15m 윈도로 한정한다.
    conn = sqlite3.connect(M15_DB)
    m15_min = conn.execute("SELECT MIN(timestamp) FROM candles").fetchone()[0]
    conn.close()
    m15_start = pd.Timestamp(m15_min).date()
    pairs = panel_max[panel_max["date"] >= m15_start][["market", "date"]].drop_duplicates()
    log.info("loading 15m paths for %d (market,date) pairs (>= %s)...", len(pairs), m15_start)
    bars_map = load_paths(pairs)
    log.info("15m paths loaded: %d", len(bars_map))

    cells = build_grid(axes)
    n_combos = len(cells)
    log.info("SELECTION: %d cells (axes=%s, one-axis-at-a-time off baseline)",
             n_combos, sorted(axes))

    # universe 별 panel slice 캐시
    uni_cache = {}
    def get_uni(u):
        if u not in uni_cache:
            uni_cache[u] = panel_max[panel_max["f_qv_rank"] <= u].copy()
        return uni_cache[u]

    rows = []
    for i, (axis, param, cfg) in enumerate(cells):
        pu = get_uni(cfg["universe"])
        m, ndays = eval_cell(pu, cfg, bars_map)
        if m is None:
            log.warning("[%d/%d] %s %s — no OOS, skip", i + 1, n_combos, axis, param)
            continue
        m.update(axis=axis, param=param, feats=cfg["feats"],
                 n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
                 learning_rate=cfg["learning_rate"],
                 min_child_weight=cfg["min_child_weight"],
                 reg_lambda=cfg["reg_lambda"], universe=cfg["universe"],
                 cal_buckets=cfg["cal_buckets"], rr_eps=cfg["rr_eps"],
                 tie_break=cfg["tie_break"])
        rows.append(m)
        log.info("[%d/%d] %-12s %-18s net=%+.5f Sharpe=%+.2f hit=%.3f "
                 "%%SL=%.3f deepNoSL=%.3f prec20=%.4f foldPos=%.2f n=%d",
                 i + 1, n_combos, axis, str(param), m["net_mean"],
                 m["sharpe"] if pd.notna(m["sharpe"]) else float("nan"),
                 m["hit"], m["pct_sl"], m["deep_loss_noSL"]
                 if pd.notna(m["deep_loss_noSL"]) else float("nan"),
                 m["precision_pump20"] if pd.notna(m["precision_pump20"]) else float("nan"),
                 m["fold_net_pos_frac"] if pd.notna(m["fold_net_pos_frac"]) else float("nan"),
                 m["n"])

    res = pd.DataFrame(rows)
    lead = ["axis", "param", "feats", "n_estimators", "max_depth", "learning_rate",
            "min_child_weight", "reg_lambda", "universe", "cal_buckets", "rr_eps",
            "tie_break", "n", "n_days", "net_mean", "net_median", "sharpe", "sortino",
            "mdd", "cum", "hit", "pct_sl", "deep_loss_noSL", "precision_pump20",
            "cvar95", "worst", "pct_tp", "pct_eod", "fold_net_min", "fold_net_pos_frac"]
    res = res[[c for c in lead if c in res.columns]]

    # baseline 대비 Δ 부착
    if (res["axis"] == "baseline").any():
        b = res[res["axis"] == "baseline"].iloc[0]
        for col in ["net_mean", "sharpe", "sortino", "mdd", "hit", "pct_sl",
                    "deep_loss_noSL", "precision_pump20", "cvar95"]:
            res[f"d_{col}"] = res[col] - b[col]

    res = res.sort_values(["axis", "net_mean"], ascending=[True, False])
    res_path = OUT / "cc_r1_core_opt_compare_v1.csv"
    res.to_csv(res_path, index=False)

    cov = dict(
        oos_dates=int(panel_max["date"].nunique()),
        oos_window=[str(panel_max["date"].min()), str(panel_max["date"].max())],
        m15_start=str(m15_start),
        n_folds=N_FOLDS, embargo=EMBARGO, top_k=TOP_K,
        n_cells=n_combos, axes=sorted(axes),
        baseline=BASE, label_cols=LABEL_COLS,
        exit_path="15m SL-3%/TP+5%/EOD net 0.15%",
    )
    (OUT / "cc_r1_core_opt_coverage_v1.json").write_text(json.dumps(cov, indent=2, default=str))
    log.info("DONE. wrote %s + cc_r1_core_opt_coverage_v1.json", res_path)


if __name__ == "__main__":
    main()
