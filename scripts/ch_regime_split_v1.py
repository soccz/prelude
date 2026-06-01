"""A2 challenger — regime-split ranking + regime abstain vs R1 champion (OOS net 비교).

목적 (signal-researcher 트랙 B3):
  현 챔피언 R1(risk-reward 비율 p_up10/max(p_dn5,eps))에 도전하는 challenger A2 =
  **BTC regime별 분리 행동 + 기권(abstain)**. 프로젝트 발견: volatile regime = hit
  최고·net 최악. regime마다 진입 dynamics가 다르다 → regime별로 다르게 행동하면
  net 개선되나? 사용자 선호: 하락최소화 > 상승, ~-5% 손실 수용.

  R1   = p_up10 / max(p_dn5, eps)             (챔피언, byte-identical 정렬키)
  A2a  = regime-conditional 정렬키:            regime별로 R1 vs downside-penalized
         (p_up10 - lam_r*p_dn5) 중 prior-fold train OOS에서 더 나은 키를 선택.
  A2b  = regime abstain:                       net 최악 regime(들)에서 추천 0 (발사 안 함).
         어느 regime을 끄는지는 prior-fold train OOS net이 음수인 regime → train-only 결정.
  A2c  = a+b 조합.

★ 왜 이 스크립트가 r2_challenger_compare_v1.py와 다른가:
  - R2는 전역 단일 정렬키(λ 고정). A2a는 regime별로 다른 정렬키/λ를 쓴다.
  - A2b/c는 기권을 도입한다 — 추천 0인 날을 만들어 그 날 손실을 회피.
  - panel·head·calibration·15m청산은 R2와 완전 동일 substrate를 재사용 (정렬/기권 효과 isolate).

★ degeneracy 가드 (오케스트레이터 요구):
  - "덜 거래해서 net 올리기"가 아님을 분리 증명: traded 픽 net/precision 개선 + 기권의
    가치(기권한 날 baseline net이 실제로 음수였나)를 따로 측정.
  - 픽수·커버리지·기권일수·기권일 baseline net을 모두 보고.

★ LEAK 방어 (same-day leak 2번 전적 — 양보 X):
  - head feature = build_market_features의 .shift(1) (D-1까지), LEAK_COLS/next_* 제외.
  - regime = attach_btc_regime의 D-1 shift (같은날 BTC 종가 X). leak-free.
  - per-fold OOF bucket calibration (train-only). test fold는 적용만.
  - purged WF + embargo 5d. fold train 종료 시점 기준.
  - ★ A2a regime별 정렬키 선택 / A2b 끌 regime 선택 = oos[fold < f] (이전 fold OOS)에서만 결정.
    test fold(fold == f)는 그 결정을 적용만 (in-sample regime tuning leak 방어).
  - 청산 15m 경로는 진입일 D in-trade outcome → leak 아님.

self-contained: prelude 내부 모듈만 import (gan_t/xsec_alpha/fin import 0).

사용:
    python scripts/ch_regime_split_v1.py                  # 캐시 있으면 재사용
    python scripts/ch_regime_split_v1.py --rebuild-panel  # panel 재생성 (XGB+15m, 느림)
    python scripts/ch_regime_split_v1.py --limit-markets 80  # 개발
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

# R1/R2와 동일한 leak-free head 파이프 (recommend.py가 import하는 바로 그 빌더).
from scripts.downside_head_riskreward_v1 import (  # noqa: E402
    build_panel,
    add_cross_sectional,
    attach_btc_regime,
    walk_forward_heads,
    _feats,
    UP_THRESH,
    DN_THRESH,
)
# 라이브 ledger와 동일한 15m SL/TP/EOD 경로청산.
from scripts.recommender_downside_exit_v1 import simulate_path  # noqa: E402

D1_DB = str(_ROOT / "data" / "upbit_d1.db")
M15_DB = str(_ROOT / "data" / "upbit_15m.db")
OUT = _ROOT / "output"
PANEL_CACHE = OUT / "ch_regime_split_panel_v1.parquet"

ROUND_TRIP_COST = 0.0015     # 왕복 0.15%
HARD_SL = 0.03               # -3% 손절 (라이브와 동일)
TP = 0.05                    # +5% 익절
DEEP_LOSS = -0.05            # 사용자 기준 '나쁜 사건'

UP_ANCHOR = "lab_up_10"
DN_ANCHOR = "lab_dn_05"
RR_EPS = 1e-3
TOP_K = 3
UNIVERSE_TOP_N = 100
N_FOLDS = 6
EMBARGO = 5

REGIMES = ["bull_quiet", "bull_volatile", "bear_quiet", "bear_volatile"]
# A2a 후보 정렬키: R1 ratio + downside-penalized λ grid (regime별로 train에서 선택).
LAMBDA_GRID = [1.0, 2.0, 3.0]

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("ch_regime_split")


# ============================================================================
# 1. Full OOS candidate panel build (R2와 동일 substrate) — 캐시
# ============================================================================
def load_paths(pairs: pd.DataFrame) -> dict:
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


def realize_net(bars: list):
    gross, outcome = simulate_path(bars, HARD_SL, TP, None)
    eod_gross, _ = simulate_path(bars, None, None, None)   # no-SL/no-TP EOD
    if not np.isfinite(gross):
        return np.nan, "nodata", np.nan
    eod_net = (eod_gross - ROUND_TRIP_COST) if np.isfinite(eod_gross) else np.nan
    return gross - ROUND_TRIP_COST, outcome, eod_net


def build_full_oos_panel(limit_markets) -> pd.DataFrame:
    """전체 OOS 후보 패널 (모든 row, top-3 아님). 정렬키 산출 + 15m net 부착.

    R2 스크립트와 동일 빌더·유니버스·fold·청산경로 → A2는 이 위에서 선택/기권만 바꾼다.
    """
    panel = build_panel(limit_markets)
    panel = add_cross_sectional(panel)
    panel = attach_btc_regime(panel)
    panel = panel[panel["f_qv_rank"] <= UNIVERSE_TOP_N].copy()
    panel["in_universe"] = True
    feats = _feats(panel)
    log.info("panel(top%d) rows=%d markets=%d dates=%d feats=%d",
             UNIVERSE_TOP_N, len(panel), panel["market"].nunique(),
             panel["date"].nunique(), len(feats))

    up_labels = [f"lab_{n}" for n in UP_THRESH]
    dn_labels = [f"lab_{n}" for n in DN_THRESH]
    label_cols = up_labels + dn_labels

    log.info("=== walk-forward heads (R1과 동일 빌더) ===")
    oos = walk_forward_heads(panel, feats, label_cols, N_FOLDS, EMBARGO)
    if oos.empty:
        log.error("no OOS folds — abort"); sys.exit(1)
    up = f"p_{UP_ANCHOR}"; dn = f"p_{DN_ANCHOR}"
    oos = oos.dropna(subset=[up, dn]).copy()

    conn = sqlite3.connect(M15_DB)
    m15_min = conn.execute("SELECT MIN(timestamp) FROM candles").fetchone()[0]
    conn.close()
    m15_start = pd.Timestamp(m15_min).date()
    oos = oos[oos["date"] >= m15_start].reset_index(drop=True)

    pairs = oos[["market", "date"]].drop_duplicates()
    bars_map = load_paths(pairs)
    oos = oos[oos.apply(lambda r: (r["market"], r["date"]) in bars_map,
                        axis=1)].reset_index(drop=True)
    nets, outs, eods, p20 = [], [], [], []
    for _, r in oos.iterrows():
        net, oc, eod_net = realize_net(bars_map[(r["market"], r["date"])])
        nets.append(net); outs.append(oc); eods.append(eod_net)
        p20.append(1 if (r["up_high_ret"] >= 0.20) else 0)
    oos["net"] = nets
    oos["outcome"] = outs
    oos["eod_net"] = eods
    oos["pump20_hit"] = p20
    oos = oos.dropna(subset=["net"]).reset_index(drop=True)

    oos["R1"] = oos[up] / np.maximum(oos[dn], RR_EPS)
    for lam in LAMBDA_GRID:
        oos[f"pen_lam{lam}"] = oos[up] - lam * oos[dn]
    keep = ["market", "date", "fold", "regime",
            up, dn, "p_lab_dn_10", "p_lab_up_10", "exp_downside",
            "R1"] + [f"pen_lam{lam}" for lam in LAMBDA_GRID] + \
           ["net", "eod_net", "outcome", "pump20_hit", "up_high_ret", "down_low_ret"]
    # ★ up=p_lab_up_10 이 anchor와 명시 둘 다 들어가 중복 → 순서 보존 dedup.
    seen, dedup = set(), []
    for c in keep:
        if c in oos.columns and c not in seen:
            seen.add(c); dedup.append(c)
    return oos[dedup].copy()


def get_panel(args) -> pd.DataFrame:
    if PANEL_CACHE.exists() and not args.rebuild_panel:
        log.info("loading cached panel %s", PANEL_CACHE)
        p = pd.read_parquet(PANEL_CACHE)
        p["date"] = pd.to_datetime(p["date"]).dt.date
        return p
    log.info("building full OOS panel (XGB heads + 15m net) — 느림 ...")
    p = build_full_oos_panel(args.limit_markets)
    p.to_parquet(PANEL_CACHE, index=False)
    log.info("cached panel %s rows=%d", PANEL_CACHE, len(p))
    return p


# ============================================================================
# 2. 하방-우선 net 지표 (R2 net_metrics와 동일 정의 — 공정 비교)
# ============================================================================
def net_metrics(picks: pd.DataFrame) -> dict:
    d = picks.dropna(subset=["net"]).copy()
    n = len(d)
    if n == 0:
        return {"n": 0}
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
    return dict(
        n=int(n), n_days=int(d["date"].nunique()),
        pct_sl=float((oc == "sl").mean()),                              # ★ 1순위
        deep_loss_freq_noSL=float((eod <= DEEP_LOSS).mean()) if len(eod) else np.nan,
        net_mean=mu,                                                     # ★ 2순위
        hit=float((net > 0).mean()),                                     # ★ 3순위
        precision_pump20=float(d["pump20_hit"].dropna().mean())
        if d["pump20_hit"].notna().any() else np.nan,
        deep_loss_freq=float((net <= DEEP_LOSS).mean()),
        cvar95=cvar95, worst=float(net.min()), mdd=mdd, cum=cum,
        sharpe=float(mu / sd * np.sqrt(365)) if sd and sd > 0 else np.nan,
        sortino=float(mu / dstd * np.sqrt(365)) if dstd and dstd > 0 else np.nan,
        pct_tp=float((oc == "tp").mean()), pct_eod=float((oc == "eod").mean()),
    )


# ============================================================================
# 3. 픽 선택 — R1 baseline / A2a regime-split ranking / A2b abstain / A2c
# ============================================================================
def topk_by_key(day_df: pd.DataFrame, score_col: str, k: int) -> pd.DataFrame:
    """R1/R2 tie-break과 동일: deep-dump↓ → p_up10↑ → exp_downside↑."""
    s = day_df.sort_values(
        [score_col, "p_lab_dn_10", "p_lab_up_10", "exp_downside"],
        ascending=[False, True, False, False])
    return s.head(k)


def picks_global_key(oos: pd.DataFrame, score_col: str, k: int) -> pd.DataFrame:
    """전역 단일 정렬키 top-k (R1 baseline 용)."""
    return oos.groupby("date", group_keys=False).apply(
        lambda g: topk_by_key(g, score_col, k))


def _regime_train_decision(train_oos: pd.DataFrame, k: int):
    """train fold OOS만으로:
      (a) regime별 best 정렬키 (R1 vs pen_lam*) — 하방-우선 (%SL↓ → no-SL deep↓ → -net↑)
      (b) regime별 R1 train net_mean (기권 판단용)
    leak 방어: 이 결정은 오직 train_oos에서 나온다. test fold엔 적용만.
    """
    keys = ["R1"] + [c for c in train_oos.columns if c.startswith("pen_lam")]
    best_key, regime_net = {}, {}
    for rg in REGIMES:
        sub = train_oos[train_oos["regime"] == rg]
        if len(sub) < 60:
            best_key[rg] = "R1"   # 표본 부족 → 챔피언 기본
            regime_net[rg] = np.nan
            continue
        best, best_rank = "R1", None
        for key in keys:
            pk = sub.groupby("date", group_keys=False).apply(
                lambda g: topk_by_key(g, key, k))
            m = net_metrics(pk)
            if m.get("n", 0) < 30:
                continue
            # 하방-우선 정렬: (%SL, no-SL deep, -net_mean) 작을수록 좋음
            rank = (m.get("pct_sl", 1.0),
                    m.get("deep_loss_freq_noSL", 1.0)
                    if pd.notna(m.get("deep_loss_freq_noSL")) else 1.0,
                    -m.get("net_mean", -1.0))
            if best_rank is None or rank < best_rank:
                best_rank, best = rank, key
        best_key[rg] = best
        pk_r1 = sub.groupby("date", group_keys=False).apply(
            lambda g: topk_by_key(g, "R1", k))
        regime_net[rg] = net_metrics(pk_r1).get("net_mean", np.nan)
    return best_key, regime_net


def build_a2_picks(oos: pd.DataFrame, k: int, mode: str, abstain_thresh: float):
    """fold별로 train OOS(이전 fold)에서 regime 결정 → test fold(현 fold)에 적용.

    mode: 'a' (regime-split ranking), 'b' (regime abstain), 'c' (둘 다).
    abstain_thresh: train regime net_mean이 이 값보다 낮으면 그 regime 기권 (b/c).
    반환: (picks_df, abstained_rows_df, decisions list)
    """
    folds = sorted(oos["fold"].unique())
    all_picks, abstained, decisions = [], [], []
    for f in folds:
        train = oos[oos["fold"] < f]       # ★ 이전 fold OOS만 (leak 방어)
        test = oos[oos["fold"] == f]
        if len(test) == 0:
            continue
        if len(train) < 200:
            # 첫 fold(이전 train 없음) → 챔피언 R1 기본, 기권 없음
            pk = test.groupby("date", group_keys=False).apply(
                lambda g: topk_by_key(g, "R1", k))
            all_picks.append(pk)
            decisions.append({"fold": int(f), "note": "no_prior_train_R1"})
            continue
        best_key, regime_net = _regime_train_decision(train, k)
        abstain_set = set()
        if mode in ("b", "c"):
            for rg, nm in regime_net.items():
                if pd.notna(nm) and nm < abstain_thresh:
                    abstain_set.add(rg)
        for rg in test["regime"].dropna().unique():
            sub = test[test["regime"] == rg]
            if rg in abstain_set:
                # 기권 — 이 날들 픽 0. baseline(R1 top-k)은 abstained로 따로 기록.
                base_pk = sub.groupby("date", group_keys=False).apply(
                    lambda g: topk_by_key(g, "R1", k))
                abstained.append(base_pk.assign(abstain_regime=rg))
                continue
            key = best_key.get(rg, "R1") if mode in ("a", "c") else "R1"
            pk = sub.groupby("date", group_keys=False).apply(
                lambda g: topk_by_key(g, key, k))
            all_picks.append(pk)
        nanr = test[test["regime"].isna()]
        if len(nanr):
            pk = nanr.groupby("date", group_keys=False).apply(
                lambda g: topk_by_key(g, "R1", k))
            all_picks.append(pk)
        decisions.append({"fold": int(f), "best_key": best_key,
                          "regime_net_train": {k2: (None if pd.isna(v) else round(float(v), 5))
                                               for k2, v in regime_net.items()},
                          "abstain": sorted(abstain_set)})
    picks = pd.concat(all_picks, ignore_index=True) if all_picks else pd.DataFrame()
    abst = pd.concat(abstained, ignore_index=True) if abstained else pd.DataFrame()
    return picks, abst, decisions


# ============================================================================
# 4. regime별 + 전체 비교
# ============================================================================
def per_regime_metrics(picks: pd.DataFrame, tag: str) -> list:
    rows = []
    for rg in REGIMES:
        sub = picks[picks["regime"] == rg] if len(picks) else picks
        m = net_metrics(sub)
        m.update(policy=tag, regime=rg)
        rows.append(m)
    m = net_metrics(picks)
    m.update(policy=tag, regime="ALL")
    rows.append(m)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-markets", type=int, default=None)
    ap.add_argument("--rebuild-panel", action="store_true")
    ap.add_argument("--abstain-thresh", type=float, default=0.0,
                    help="train regime net_mean이 이 값 미만이면 기권 (b/c). 0=net음수 regime")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    oos = get_panel(args)
    log.info("OOS panel rows=%d dates=%d window %s..%s folds=%s",
             len(oos), oos["date"].nunique(), oos["date"].min(),
             oos["date"].max(), sorted(oos["fold"].unique()))

    # --- R1 baseline (전역 단일 정렬키, 기권 없음) ---
    r1_picks = picks_global_key(oos, "R1", TOP_K)
    log.info("R1 baseline picks=%d", len(r1_picks))

    # --- A2 variants ---
    a2a_picks, _, a2a_dec = build_a2_picks(oos, TOP_K, "a", args.abstain_thresh)
    a2b_picks, a2b_abst, a2b_dec = build_a2_picks(oos, TOP_K, "b", args.abstain_thresh)
    a2c_picks, a2c_abst, a2c_dec = build_a2_picks(oos, TOP_K, "c", args.abstain_thresh)

    # --- 비교 표 (regime별 + ALL) ---
    rows = []
    rows += per_regime_metrics(r1_picks, "R1_baseline")
    rows += per_regime_metrics(a2a_picks, "A2a_regime_rank")
    rows += per_regime_metrics(a2b_picks, "A2b_abstain")
    rows += per_regime_metrics(a2c_picks, "A2c_both")
    res = pd.DataFrame(rows)
    lead = ["policy", "regime", "n", "n_days", "pct_sl", "deep_loss_freq_noSL",
            "net_mean", "hit", "precision_pump20", "cvar95", "worst",
            "mdd", "cum", "sharpe", "sortino", "pct_tp", "pct_eod"]
    res = res[[c for c in lead if c in res.columns]]
    res_path = OUT / "ch_regime_split_compare_v1.csv"
    res.to_csv(res_path, index=False)

    # --- 기권 가치: 기권한 날들 baseline(R1) net이 실제로 음수였나? (degeneracy 가드) ---
    abstain_value = {}
    for tag, abst in [("A2b", a2b_abst), ("A2c", a2c_abst)]:
        if len(abst):
            m = net_metrics(abst)
            abstain_value[tag] = {
                "n_abstained_picks": int(m["n"]),
                "n_abstained_days": int(m["n_days"]),
                "abstained_baseline_net_mean": round(m["net_mean"], 5),
                "abstained_baseline_pct_sl": round(m["pct_sl"], 4),
                "abstained_baseline_deep_noSL": (None if pd.isna(m["deep_loss_freq_noSL"])
                                                 else round(m["deep_loss_freq_noSL"], 4)),
                "abstained_baseline_cum": round(m["cum"], 4),
                "abstain_regimes": sorted(set(abst["abstain_regime"])),
            }
        else:
            abstain_value[tag] = {"n_abstained_picks": 0}

    # --- 커버리지 + fold 결정 dump ---
    r1_all = net_metrics(r1_picks)
    cov = dict(
        oos_rows=int(len(oos)), oos_dates=int(oos["date"].nunique()),
        oos_window=[str(oos["date"].min()), str(oos["date"].max())],
        universe="static_top100", n_folds=N_FOLDS, embargo=EMBARGO,
        lambda_grid=LAMBDA_GRID, abstain_thresh=args.abstain_thresh,
        exit_path="15m SL-3%/TP+5%/EOD net 0.15%",
        n_ranking_keys_per_regime=1 + len(LAMBDA_GRID),
        regime_dist={k: int(v) for k, v in oos["regime"].value_counts().items()},
        coverage={
            "R1_baseline_picks": int(len(r1_picks)),
            "R1_baseline_days": int(r1_all["n_days"]),
            "A2a_picks": int(len(a2a_picks)),
            "A2b_picks": int(len(a2b_picks)),
            "A2c_picks": int(len(a2c_picks)),
        },
        abstain_value=abstain_value,
        fold_decisions={"A2a": a2a_dec, "A2c": a2c_dec},
    )
    (OUT / "ch_regime_split_coverage_v1.json").write_text(json.dumps(cov, indent=2, default=str))

    # 픽 dump (감사용)
    dump_cols = ["date", "market", "regime", "fold", f"p_{UP_ANCHOR}",
                 f"p_{DN_ANCHOR}", "R1", "net", "eod_net", "outcome", "pump20_hit"]
    dump_cols = [c for c in dump_cols if c in oos.columns]
    pd.concat([
        r1_picks[dump_cols].assign(policy="R1_baseline"),
        a2a_picks[dump_cols].assign(policy="A2a"),
        a2b_picks[dump_cols].assign(policy="A2b"),
        a2c_picks[dump_cols].assign(policy="A2c"),
    ], ignore_index=True).to_csv(OUT / "ch_regime_split_picks_v1.csv", index=False)

    # ---- 콘솔 요약 ----
    log.info("\n===== A2 regime-split: regime별 + ALL (OOS net 0.15%, 15m SL/TP/EOD) =====")
    log.info("  policy            regime          n   days  %%SL  deepNoSL net_mean hit prec20  cum    Sharpe")
    for _, r in res.iterrows():
        if r["n"] == 0:
            continue
        log.info("  %-17s %-14s %4d %4d  %.3f  %s  %+.4f %.2f  %s %+.3f  %s",
                 r["policy"], r["regime"], int(r["n"]), int(r["n_days"]),
                 r["pct_sl"],
                 f"{r['deep_loss_freq_noSL']:.3f}" if pd.notna(r["deep_loss_freq_noSL"]) else " nan ",
                 r["net_mean"], r["hit"],
                 f"{r['precision_pump20']:.3f}" if pd.notna(r["precision_pump20"]) else " nan ",
                 r["cum"], f"{r['sharpe']:+.2f}" if pd.notna(r["sharpe"]) else "nan")

    log.info("\n===== 기권 가치 (degeneracy 가드: 기권한 날 baseline net이 음수였나) =====")
    for tag, v in abstain_value.items():
        if v.get("n_abstained_picks", 0) == 0:
            log.info("  %s: 기권 0", tag); continue
        log.info("  %s: 기권픽 %d (%d일, regime=%s) -> baseline net_mean=%+.4f "
                 "(%%SL=%.3f deepNoSL=%s cum=%+.3f)",
                 tag, v["n_abstained_picks"], v["n_abstained_days"],
                 v["abstain_regimes"], v["abstained_baseline_net_mean"],
                 v["abstained_baseline_pct_sl"], v["abstained_baseline_deep_noSL"],
                 v["abstained_baseline_cum"])

    log.info("DONE. wrote %s + ch_regime_split_picks_v1.csv + coverage json", res_path)


if __name__ == "__main__":
    main()
