"""Track C1 — sustained-gain LABEL challenger vs R1 champion (증거 생성).

사용자 비전 (Track C1):
  이전 7실험(R2 랭킹·exit·A1~A4·멀티데이)은 전부 R1 의 **같은 진입집합**을
  후처리(재랭킹/청산변경)만 했다 → R1 천장(net 음수)을 못 넘었다(검증됨:
  output/r2_challenger_compare_v1.csv 의 모든 정렬키 net_mean<0, Sharpe -2~-5).

  C1 은 그걸 반복하지 않는다. **타겟 라벨 자체를 바꿔** 새 진입을 만든다.
  R1 타겟 = '일중 고가 high/open-1 >= +20%(pump20)' / 헤드 anchor = P(high/open>=+10%).
    → 스파이크 찍고 종가에 덤프하는 코인도 잡힘(high 만 보니까).
  C1 가설: 타겟을 **종가 기준 실현이익**(close/open-1)이나 **익일 보유수익**으로
    바꾸면 헤드가 "실제로 버는(종가까지 들고 가도 양수)" 코인을 고르도록 학습 →
    같은 유니버스·같은 청산경로라도 **다른 진입집합** → net 이 양수로 갈 가능성.

  ★ 핵심: high(스파이크) 가 아니라 close(실현) 를 타겟으로 하면 TP-before-SL /
    pump-then-dump 함정(EDA hit != ledger Sharpe, 프로젝트 hard lesson)을 학습
    단계에서 직접 회피하려는 시도. 검증 대상.

후보 라벨 (전부 day-D 타겟, 미래, 학습에만 — feature 에 안 섞임):
  sus3   = (close/open - 1) >= +0.03     # 당일 종가가 진입가 대비 +3% 이상 마감
  sus5   = (close/open - 1) >= +0.05     # +5% 이상 마감
  sus_net = (close/open - 1) >= +0.0065  # 종가 마감이 왕복비용(0.15%)+여유 넘김 (net 양수 지향)
  fwd1   = (next_close/close - 1) >= +0.03  # 익일(D+1) 종가-대-종가 보유수익 +3% 이상

  → R1 anchor(p_lab_up_10)와 동일 인프라로 헤드 학습. C1 픽 = 해당 라벨 확률 top-3.
    (risk-reward 변형은 보조: C1 확률에서 R1 의 p_dn5 패널티를 빼는 rr 도 같이 표기.)

스코어러/헤드/검증 = R1 인프라 그대로 재사용 (정당 비교):
  - build_panel / add_cross_sectional / attach_btc_regime (downside_head_riskreward_v1)
  - walk_forward_heads (per-fold OOF bucket calib, train-only, XGB random_state=42)
  - _feats (PRECURSOR_FEATURES - LEAK_COLS), 유니버스 = 정적 top100 (D-1 f_qv_rank)
  - purged WF 6-fold, embargo=5
  - 청산/net = R1 과 동일 15m SL-3%/TP+5%/EOD net 0.15% (simulate_path)
  - 비교 baseline = R1 (p_lab_up_10/max(p_lab_dn_05,eps)), 동일 OOS 행에서 정렬키만 교체.

★ LEAK 방어 (same-day leak 2번 전적 — 양보 X):
  - feature = build_market_features .shift(1) (D-1 까지). LEAK_COLS 제외.
  - sustained 라벨(sus3/sus5/sus_net/fwd1) = day-D(또는 D+1) close 기반 (타겟, 미래).
    학습 feature 에 절대 안 섞임 → SUS_LEAK_COLS 로 _feats 단계 제외 재확인.
  - fwd1 은 next_close = g["close"].shift(-1) 로 만들고, 마지막 행은 NaN (미래 없음) →
    학습/평가에서 dropna. shift(-1) 은 라벨에만, feature 에는 절대 안 들어감.
  - per-fold OOF bucket calibration (train-only). test fold 적용만.
  - 청산 15m 경로는 진입일 D in-trade outcome → 진입결정(D-1 feature)과 시점분리, leak X.
  - "너무 좋으면 leak" 자가알람: net_mean 이 R1 대비 비현실적으로 좋으면(예 +5%/일)
    sustained 라벨이 feature 로 새는지 재검증 (SUS_LEAK_COLS, fwd1 shift 방향).

self-contained: prelude 내부 모듈만 import (gan_t/xsec_alpha/fin import 0).
채택결정 X (evaluator 몫). 라벨 정의 변경은 사용자 컨펌(§4) — 본 스크립트는 증거만.

사용:
    python scripts/cc_sustained_label_v1.py
    python scripts/cc_sustained_label_v1.py --limit-markets 80   # 개발
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

from data.database import list_markets, load_candles  # noqa: E402
from data.market_universe import is_excluded_signal_market  # noqa: E402
from scripts.univariate_precursor_lift_v1 import (  # noqa: E402
    build_market_features,
    add_cross_sectional,
    completed_upbit_d1_mask,
)
from scripts.regime_split_precursor_v1 import attach_btc_regime  # noqa: E402
# R1 과 동일한 leak-free 헤드 학습/calibration 파이프 (정당 비교의 핵심).
from scripts.downside_head_riskreward_v1 import (  # noqa: E402
    walk_forward_heads,
    _feats,
    EPS,
)
# 라이브 R1 ledger 와 동일한 15m SL/TP/EOD 경로청산.
from scripts.recommender_downside_exit_v1 import (  # noqa: E402
    load_paths as load_execution_paths,
    simulate_path,
)

D1_DB = str(_ROOT / "data" / "upbit_d1.db")
M15_DB = str(_ROOT / "data" / "upbit_15m.db")
OUT = _ROOT / "output"

ROUND_TRIP_COST = 0.0015     # 왕복 0.15%
HARD_SL = 0.03               # -3% 손절 (R1 동일)
TP = 0.05                    # +5% 익절 (R1 동일)
DEEP_LOSS = -0.05            # 사용자 기준: net 이보다 깊은 손실 = 나쁜 사건

# R1 anchor (정당 비교 baseline & 하방 헤드 — recommend.py 와 동일 정렬키).
R1_UP = "lab_up_10"          # P(high/open >= +10%) — R1 분자
R1_DN = "lab_dn_05"          # P(low/open <= -5%)    — R1 분모
RR_EPS = 1e-3

# ---- C1 새 sustained-gain 라벨 (전부 day-D/D+1 타겟, 미래) ----
# close/open-1 임계 (당일 종가 실현). sus_net = 왕복비용 0.15% 를 넘기는 최소 마감.
SUS_THRESH = {"sus3": 0.03, "sus5": 0.05, "sus_net": ROUND_TRIP_COST + 0.005}
FWD_THRESH = {"fwd1": 0.03}   # 익일 close-to-close 보유수익
SUS_LABELS = list(SUS_THRESH) + list(FWD_THRESH)

# 학습 feature 에서 절대 제외 (C1 라벨 + R1 라벨 + 미래 outcome).
SUS_LEAK_COLS = set(SUS_LABELS) | {f"lab_{n}" for n in SUS_LABELS} | {
    "eod_open_ret", "next_close_ret",
}

UNIVERSE_TOP_N = 100
N_FOLDS = 6
EMBARGO = 5
TOP_K = 3

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("cc_sustained")


# ============================================================================
# 1. Panel — R1 과 동일한 leak-free D-1 feature + R1 라벨 + C1 sustained 라벨
#    (downside_head_riskreward_v1.build_panel 을 그대로 따르되 sustained 라벨 추가)
# ============================================================================
# R1 build_panel 의 outcome 라벨 (정당 비교 위해 동일 정의 재현)
_UP_THRESH = {"up_05": 0.05, "up_10": 0.10, "up_15": 0.15, "up_20": 0.20}
_DN_THRESH = {"dn_03": -0.03, "dn_05": -0.05, "dn_10": -0.10}


def _add_all_labels(
    g: pd.DataFrame,
    asof_kst: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """day-D open 대비 R1 outcome 라벨 + C1 sustained 라벨 부착.
    feature 는 build_market_features 에서 이미 shift(1) 됨 → 여기 라벨은 same-row
    (day-D) open/high/low/close 로 만든다 (타겟=미래, 시점분리).
    fwd1 만 next_close=close.shift(-1) (D+1, 미래) → 마지막 행 NaN.
    """
    o = g["open"].values
    h = g["high"].values
    l = g["low"].values
    c = g["close"].values
    g = g.copy()
    g["up_high_ret"] = h / (o + EPS) - 1.0
    g["down_low_ret"] = l / (o + EPS) - 1.0
    g["eod_ret"] = c / (o + EPS) - 1.0           # = close/open - 1 (당일 실현, R1 비교용)
    g["eod_open_ret"] = g["eod_ret"]             # C1 sus 라벨 origin (진단 alias)
    # R1 라벨 (정당 비교 baseline 정렬키 입력)
    for name, thr in _UP_THRESH.items():
        g[f"lab_{name}"] = (g["up_high_ret"] >= thr).astype(float)
    for name, thr in _DN_THRESH.items():
        g[f"lab_{name}"] = (g["down_low_ret"] <= thr).astype(float)
    # C1 sustained 라벨: 당일 종가 마감 기준 (close/open - 1)
    for name, thr in SUS_THRESH.items():
        g[f"lab_{name}"] = (g["eod_ret"] >= thr).astype(float)
    # C1 fwd1: 익일 close-to-close 보유수익 (next_close 는 라벨 전용 shift(-1)).
    next_close = g["close"].shift(-1)
    g["next_close_ret"] = next_close / (g["close"] + EPS) - 1.0
    for name, thr in FWD_THRESH.items():
        lab = (g["next_close_ret"] >= thr).astype(float)
        # D+1 close가 완전히 확정된 경우에만 라벨로 인정한다.
        next_complete = completed_upbit_d1_mask(
            g["timestamp"], asof_kst
        ).shift(-1, fill_value=False)
        lab[~next_complete] = np.nan
        g[f"lab_{name}"] = lab
    complete = completed_upbit_d1_mask(g["timestamp"], asof_kst)
    same_day_targets = (
        ["up_high_ret", "down_low_ret", "eod_ret", "eod_open_ret"]
        + [f"lab_{name}" for name in _UP_THRESH]
        + [f"lab_{name}" for name in _DN_THRESH]
        + [f"lab_{name}" for name in SUS_THRESH]
    )
    g.loc[~complete, same_day_targets] = np.nan
    next_complete = complete.shift(-1, fill_value=False)
    g.loc[~next_complete, "next_close_ret"] = np.nan
    return g


def build_panel_c1(limit_markets):
    markets = [
        market
        for market in list_markets(D1_DB)
        if not is_excluded_signal_market(str(market))
    ]
    if limit_markets:
        markets = markets[:limit_markets]
    log.info("loading %d markets", len(markets))
    out_label_cols = (
        ["up_high_ret", "down_low_ret", "eod_ret", "next_close_ret"]
        + [f"lab_{n}" for n in _UP_THRESH]
        + [f"lab_{n}" for n in _DN_THRESH]
        + [f"lab_{n}" for n in SUS_LABELS]
    )
    frames = []
    for i, m in enumerate(markets):
        df = load_candles(D1_DB, m)
        if df is None or len(df) < 70:
            continue
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["market"] = m
        feat = build_market_features(df)        # D-1 feature
        oc = _add_all_labels(df)[["timestamp"] + out_label_cols]
        feat = feat.merge(oc, on="timestamp", how="left")
        frames.append(feat)
        if (i + 1) % 60 == 0:
            log.info("  %d/%d", i + 1, len(markets))
    panel = pd.concat(frames, ignore_index=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel["date"] = panel["timestamp"].dt.date
    panel = panel.sort_values(["date", "market"]).reset_index(drop=True)
    return panel


# ============================================================================
# 2. 15m 경로 net (R1 r2_challenger 와 byte-identical 경로)
# ============================================================================
def load_paths(pairs: pd.DataFrame) -> dict:
    return load_execution_paths(pairs, db_path=M15_DB)


def realize_net(bars: list):
    """R1 과 동일: SL-3%/TP+5%/EOD 경로 net + no-SL EOD net (정렬키 본질 하방)."""
    gross, outcome = simulate_path(bars, HARD_SL, TP, None)
    eod_gross, _ = simulate_path(bars, None, None, None)
    if not np.isfinite(gross):
        return np.nan, "nodata", np.nan
    eod_net = (eod_gross - ROUND_TRIP_COST) if np.isfinite(eod_gross) else np.nan
    return gross - ROUND_TRIP_COST, outcome, eod_net


# ============================================================================
# 3. net 지표 (R1 net_metrics 와 동일 정의)
# ============================================================================
def net_metrics(trades: pd.DataFrame, self_label: str | None) -> dict:
    d = trades.dropna(subset=["net"]).copy()
    n = len(d)
    if n == 0:
        return {}
    net = d["net"].values
    daily = d.groupby("date")["net"].mean().sort_index()
    eq = (1 + daily).cumprod()
    peak = eq.cummax()
    mdd = float(((eq - peak) / peak).min())
    cum = float(eq.iloc[-1] - 1.0)
    trade_mu = float(net.mean())
    daily_mu = float(daily.mean())
    sd = float(daily.std())
    dstd = float(daily[daily < 0].std()) if (daily < 0).any() else np.nan
    k5 = max(1, int(np.ceil(0.05 * n)))
    cvar95 = float(np.sort(net)[:k5].mean())
    oc = d["outcome"]
    eod = d["eod_net"].dropna().values if "eod_net" in d.columns else np.array([])
    # self-label precision: 픽이 자기 sustained 라벨을 실제 달성했나 (close 기반).
    prec_self = np.nan
    if self_label and self_label in d.columns:
        sv = d[self_label].dropna()
        prec_self = float(sv.mean()) if len(sv) else np.nan
    return dict(
        n=int(n),
        n_days=int(d["date"].nunique()),
        pct_sl=float((oc == "sl").mean()),
        deep_loss_freq_noSL=float((eod <= DEEP_LOSS).mean()) if len(eod) else np.nan,
        net_mean=trade_mu,
        net_median=float(np.median(net)),
        hit=float((net > 0).mean()),
        precision_self=prec_self,                                      # 자기 라벨 적중
        precision_pump20=float(d["pump20_hit"].dropna().mean())
        if d["pump20_hit"].notna().any() else np.nan,                  # R1 pump20 적중
        cvar95=cvar95,
        worst=float(net.min()),
        mdd=mdd,
        cum=cum,
        sharpe=float(daily_mu / sd * np.sqrt(365)) if sd and sd > 0 else np.nan,
        sortino=float(daily_mu / dstd * np.sqrt(365)) if dstd and dstd > 0 else np.nan,
        pct_tp=float((oc == "tp").mean()),
        pct_eod=float((oc == "eod").mean()),
    )


def topk_picks(oos: pd.DataFrame, score_col: str, k: int) -> pd.DataFrame:
    """각 날 score_col 내림차순 top-k. tie-break = 하방-우선(R1 과 동일 정신):
    p_lab_dn_10 ↑작게 → p_lab_up_10 ↑크게 → exp_downside ↑(덜 깊은 하방)."""
    cols = ["date", score_col]
    asc = [True, False]
    for tb, a in [("p_lab_dn_10", True), ("p_lab_up_10", False),
                  ("exp_downside", False)]:
        if tb in oos.columns and tb != score_col:
            cols.append(tb); asc.append(a)
    s = oos.sort_values(cols, ascending=asc)
    return s.groupby("date").head(k).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-markets", type=int, default=None)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    # --- 1) leak-free panel + R1 라벨 + C1 sustained 라벨 ---
    panel = build_panel_c1(args.limit_markets)
    panel = add_cross_sectional(panel)
    panel = attach_btc_regime(panel)
    # 라이브 R1 유니버스 = 정적 top100 (D-1 f_qv_rank), recommend.py 와 동일.
    panel = panel[panel["f_qv_rank"] <= UNIVERSE_TOP_N].copy()
    panel["in_universe"] = True
    feats = [f for f in _feats(panel) if f not in SUS_LEAK_COLS]   # C1 라벨 제외 재확인
    log.info("panel(top%d universe) rows=%d markets=%d dates=%d feats=%d",
             UNIVERSE_TOP_N, len(panel), panel["market"].nunique(),
             panel["date"].nunique(), len(feats))

    # ★ leak 자가검증 1: sustained 라벨이 feature 목록에 없는가
    bad = [f for f in feats if f in SUS_LEAK_COLS or f.startswith("lab_")
           or f in ("eod_ret", "next_close_ret", "up_high_ret", "down_low_ret")]
    if bad:
        log.error("LEAK GUARD FAIL: label/outcome cols in feats: %s", bad)
        sys.exit(1)
    log.info("LEAK GUARD ok: 0 label/outcome cols in %d feats", len(feats))

    # base rate (유니버스 내 전체, 임계별)
    log.info("=== base rates (universe, day-D) ===")
    br = {}
    for lc in [R1_UP, R1_DN] + [f"lab_{n}" for n in SUS_LABELS]:
        br[lc] = float(panel[lc].mean())
        log.info("  %-12s base=%.4f", lc, br[lc])

    # --- 2) purged WF heads — R1 anchor(up10/dn05/dn10) + C1 sustained 라벨 동시 학습 ---
    #     (동일 빌더·동일 calib. C1 라벨은 타겟만 다름.)
    # R1 anchor 헤드(lab_up_10/lab_dn_05/lab_dn_10) + C1 sustained 라벨 동시 학습.
    # walk_forward_heads 는 실제 컬럼명(lab_*)을 받고 p_{lc} 를 만든다.
    label_cols = ["lab_up_10", "lab_dn_05", "lab_dn_10"] + [f"lab_{n}" for n in SUS_LABELS]
    log.info("=== walk-forward heads (R1 동일 빌더; R1 anchor + C1 sustained 라벨) ===")
    log.info("    label_cols=%s", label_cols)
    oos = walk_forward_heads(panel, feats, label_cols, N_FOLDS, EMBARGO)
    if oos.empty:
        log.error("no OOS folds — abort"); sys.exit(1)
    log.info("OOS rows=%d dates %s..%s folds=%s",
             len(oos), oos["date"].min(), oos["date"].max(),
             sorted(oos["fold"].unique()))

    up = f"p_{R1_UP}"; dn = f"p_{R1_DN}"   # walk_forward_heads -> p_{lc} (lc=lab_up_10)
    oos = oos.dropna(subset=[up, dn]).copy()

    # --- 3) 15m 경로 커버리지 (R1 과 동일 window) ---
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
    log.info("OOS with 15m net path: rows=%d dates=%d (outcome dist: %s)",
             len(oos), oos["date"].nunique(), oos["outcome"].value_counts().to_dict())

    # --- 4) 정렬키 산출 ---
    #   R1 baseline = p_up10 / max(p_dn5, eps)  (동일 OOS 행).
    #   C1 라벨별 = p_lab_{sus} (확률 정렬) + rr 변형(C1 확률 - p_dn5, 하방패널티 보조).
    oos["R1"] = oos[up] / np.maximum(oos[dn], RR_EPS)

    rank_keys = [("R1_baseline", "R1", "p_up10/p_dn5", None)]
    for lc in SUS_LABELS:
        pc = f"p_lab_{lc}"
        if pc not in oos.columns:
            log.warning("missing head prob %s — skip", pc); continue
        oos[f"rr_{lc}"] = oos[pc] - oos[dn]      # C1 확률에서 R1 하방확률 빼기(보조 변형)
        rank_keys.append((f"C1_{lc}", pc, f"P({lc})", f"lab_{lc}"))
        rank_keys.append((f"C1_{lc}_rr", f"rr_{lc}", f"P({lc})-p_dn5", f"lab_{lc}"))

    n_combos = len(rank_keys)
    log.info("SELECTION: 정렬키 %d개 (R1 baseline + C1 라벨 %d개 x {prob,rr}), "
             "동일 OOS·유니버스·청산경로", n_combos, len(SUS_LABELS))

    # --- 5) 정렬키별 top-3 net 지표 ---
    rows = []
    for policy, col, param, self_lab in rank_keys:
        picks = topk_picks(oos, col, TOP_K)
        m = net_metrics(picks, self_lab)
        m.update(policy=policy, ranking_key=col, param=param, K=TOP_K)
        rows.append(m)
    res = pd.DataFrame(rows)
    lead = ["policy", "ranking_key", "param", "K", "n", "n_days",
            "net_mean", "net_median", "sharpe", "sortino", "mdd", "cum",
            "hit", "deep_loss_freq_noSL", "pct_sl", "cvar95", "worst",
            "precision_self", "precision_pump20", "pct_tp", "pct_eod"]
    res = res[[c for c in lead if c in res.columns]]
    res_path = OUT / "cc_sustained_label_compare_v1.csv"
    res.to_csv(res_path, index=False)

    # --- 6) picks dump (감사용): R1 + 각 C1 라벨 prob top-3 side-by-side ---
    dump_frames = []
    dump_base = ["date", "market", "regime", "fold", up, dn, "p_lab_dn_10",
                 "exp_downside", "net", "outcome", "pump20_hit",
                 "eod_ret", "up_high_ret", "down_low_ret", "next_close_ret"]
    for policy, col, param, self_lab in rank_keys:
        picks = topk_picks(oos, col, TOP_K)
        cols = [c for c in dump_base if c in picks.columns]
        extra = [c for c in [col, self_lab] if c and c in picks.columns and c not in cols]
        dump_frames.append(picks[cols + extra].assign(policy=policy, sort_col=col))
    pd.concat(dump_frames, ignore_index=True).to_csv(
        OUT / "cc_sustained_label_picks_v1.csv", index=False)

    # --- 7) coverage / leak self-check json ---
    r1_net = res[res["policy"] == "R1_baseline"]["net_mean"].iloc[0]
    c1_probs = res[res["policy"].str.startswith("C1_") & ~res["policy"].str.endswith("_rr")]
    best_c1 = c1_probs.sort_values("net_mean", ascending=False).iloc[0] if len(c1_probs) else None
    cov = dict(
        oos_rows=int(len(oos)), oos_dates=int(oos["date"].nunique()),
        oos_window=[str(oos["date"].min()), str(oos["date"].max())],
        universe="static_top100_D-1", n_folds=N_FOLDS, embargo=EMBARGO,
        n_ranking_keys=int(n_combos), top_k=TOP_K,
        sus_thresh=SUS_THRESH, fwd_thresh=FWD_THRESH,
        exit_path="15m SL-3%/TP+5%/EOD net 0.15%",
        base_rates=br,
        r1_baseline_net_mean=float(r1_net),
        best_c1_label=str(best_c1["policy"]) if best_c1 is not None else None,
        best_c1_net_mean=float(best_c1["net_mean"]) if best_c1 is not None else None,
        leak_self_alarm=bool(best_c1 is not None and best_c1["net_mean"] > 0.03),
    )
    (OUT / "cc_sustained_label_coverage_v1.json").write_text(json.dumps(cov, indent=2))

    # ---- 콘솔 요약 ----
    log.info("\n===== C1 sustained-label vs R1 (top-3, OOS net 0.15% 차감, 15m SL/TP/EOD) =====")
    log.info("  %-16s %-13s n    days  net_mean  net_med  Sharpe Sortino  MaxDD    cum    hit  deepNoSL %%SL  precSelf prec20",
             "policy", "param")
    for _, r in res.iterrows():
        log.info("  %-16s %-13s %4d %4d  %+.5f %+.5f  %s %s  %+.3f %+.3f  %.2f  %.3f  %.3f  %s  %s",
                 r["policy"], str(r["param"]), int(r["n"]), int(r["n_days"]),
                 r["net_mean"], r["net_median"],
                 f"{r['sharpe']:+.2f}" if pd.notna(r["sharpe"]) else " nan ",
                 f"{r['sortino']:+.2f}" if pd.notna(r["sortino"]) else " nan ",
                 r["mdd"], r["cum"], r["hit"],
                 r["deep_loss_freq_noSL"] if pd.notna(r["deep_loss_freq_noSL"]) else np.nan,
                 r["pct_sl"],
                 f"{r['precision_self']:.3f}" if pd.notna(r["precision_self"]) else " nan ",
                 f"{r['precision_pump20']:.3f}" if pd.notna(r["precision_pump20"]) else " nan ")

    log.info("\n===== ★ net 양수/R1 우위 라벨 =====")
    log.info("  R1 baseline net_mean = %+.5f", r1_net)
    pos = res[(res["net_mean"] > 0)]
    beat = res[(res["net_mean"] > r1_net) & (res["policy"] != "R1_baseline")]
    if len(pos):
        for _, r in pos.iterrows():
            log.info("  [NET POSITIVE] %-16s net_mean=%+.5f Sharpe=%s hit=%.2f",
                     r["policy"], r["net_mean"],
                     f"{r['sharpe']:+.2f}" if pd.notna(r["sharpe"]) else "nan", r["hit"])
    else:
        log.info("  net 양수 라벨 없음 (모두 net_mean <= 0).")
    if len(beat):
        log.info("  R1 우위(net_mean > R1): %s",
                 ", ".join(f"{r['policy']}({r['net_mean']:+.5f})"
                           for _, r in beat.iterrows()))
    else:
        log.info("  R1 우위 라벨 없음.")
    if cov["leak_self_alarm"]:
        log.warning("  ★ LEAK SELF-ALARM: best C1 net_mean=%+.5f > +3%%/day — sustained "
                    "라벨 feature 누수/ fwd1 shift 방향 재검증 필요!", cov["best_c1_net_mean"])
    log.info("DONE. wrote %s + cc_sustained_label_picks_v1.csv + cc_sustained_label_coverage_v1.json",
             res_path)


if __name__ == "__main__":
    main()
