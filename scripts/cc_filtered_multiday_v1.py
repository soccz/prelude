#!/usr/bin/env python3
"""Challenger C3 — 하방-선별 진입(A1 sustainability filter) + 멀티데이 홀딩 (증거 생성).

[배경 — 두 선행 실험의 합성]
  (a) Track B3 멀티데이(ch_multiday_v1): R1 *unfiltered* top-3 진입 위에선 보유를 늘리면
      손실꼬리가 먼저·더 깊이 산다(deep-loss 0.135→0.329, worst -0.34→-0.63) → REJECT.
      메타-발견: R1 진입집합 *내부*에선 수익꼬리·손실꼬리 분리 불가.
  (b) A1 sustainability(ch_sustainability_v1): D-1 dump head 로 깊은-하방 후보를 진입에서
      배제 → deep-loss(noSL) 절반(0.135→0.060), %SL 0.456→0.326. 단 net 음수 유지·상방도 깎임.

[C3 가설]
  "수익꼬리·손실꼬리 분리불가"는 *현 R1 진입집합 내부*에서만 성립한다. **하방품질로 사전선별한
  부분집합 위에서의 멀티데이**가 분리 가능성이 남은 유일한 조합 — 저-하방 픽만 골라 N일 보유하면
  손실꼬리 폭증이 억제되어, B3 에서 죽었던 net 이 양수/개선될 수 있나?

[★ 핵심 질문]
  하방선별(A1 필터)이 멀티데이의 손실꼬리 폭증을 막아 net 이 양수/개선되는
  (필터강도, N, 청산) 조합이 *갭반영 보수청산 후에도* 존재하는가? 베타(상승장 보유프리미엄)와
  진짜 엣지를 excess-over-market 으로 분리, overlapping 표본은 block 보정.

[설계]
  진입집합(4종, 같은 OOS·같은 유니버스 top100):
    - R1_unfiltered : R1 top-K (B3 가 깐 baseline). filter 강도 = none.
    - C3_filt_q0.6  : A1 dump_B q=0.60 re-selection (★A1 best-downside, 가장 공격적 배제)
    - C3_filt_q0.7  : A1 dump_B q=0.70
    - C3_filt_q0.8  : A1 dump_B q=0.80 (보수적 배제)
  (dump_B = "장중 +5% 찍고 종가 -2% 미만" = 펌프-후-덤프. A1 에서 dump_B 가 dump_A 보다 일관 우세.)

  멀티데이 청산(일봉 d1 경로), N∈{1,2,3,5}:
    - holdN      : N일 후 종가.
    - bracket    : N일 내 +TP/-SL 먼저. TP∈{0.08,0.12}, SL∈{0.05}. (사용자 ~-5% 수용 → SL 5%)
    - trail      : N일 running-high 대비 -DD 트레일. DD∈{0.08}.
  각 청산을 **두 체결모델**로:
    - opt(낙관)  : 레벨 정확체결(B3 와 동일 가정). 동봉 SL·TP 동시→SL 우선.
    - gap(보수)  : 갭 반영 — 봉 open 이 이미 레벨을 지나쳤으면(갭) open 가격에 체결(SL/trail 은
                   더 나쁜 가격, TP 는 더 좋은 가격이지만 보수적으로 TP 갭업은 레벨가 유지).
                   B3 evaluator 가 trail 갭다운 21.7% 낙관편향 지적 → gap 모델이 그 보수 청산.

[★ 4대 위생 — 양보 X]
  - look-ahead: 입력 feature ≤ D-1(build_market_features shift(1)); dump 라벨 day-D(미래)는
    head 학습 타겟이지 입력 아님. head/cutoff 모두 train-only fit(test fold 적용만), embargo=5.
    진입가 = day-D open(관측가능). 청산경로 day-D~D+N-1 = forward outcome.
  - 유니버스 시간정합: top100 = D-1 qv_rank(f_universe_qv=qv.shift(1)). A1 와 동일 유니버스.
  - 비용: 왕복 0.15% 1회(멀티데이도 1 거래). 전부 net.
  - 자동주문 X / 공유 라이브 파일 미편집. NEW 파일만.

[degeneracy/베타]
  - "덜 거래" vs traded 개선: 모든 정책 항상 top-K 채움(거래량 동일) → 픽 *교체* 효과만 분리.
    추가로 filter 가 픽을 줄이지 않음(coverage 보고).
  - 베타: 같은 N일 시장바스켓(top100 EW) 보유수익 → excess = picks_net - mkt. excess<0 이면
    베타조차 못 이김. BTC 윈도우 수익도 부착.
  - overlapping: N일 홀딩 인접 진입 겹침 → non-overlapping block(진입일 N간격 묶음)으로
    Sharpe/Sortino 재계산. 명목 Sharpe 는 낙관편향 → block 우선.

사용:
    python scripts/cc_filtered_multiday_v1.py                 # full
    python scripts/cc_filtered_multiday_v1.py --limit-markets 60  # smoke
    python scripts/cc_filtered_multiday_v1.py --use-cache     # 캐시된 OOS 재사용(빠른 재실행)
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

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.market_universe import is_excluded_signal_market  # noqa: E402
# A1/R1 과 동일한 leak-free head 파이프 재사용 (read-only import; 공유 라이브 파일 아님 —
# 연구 스크립트). 코드는 복제하지 않고 동일 빌더를 호출만 한다(A1 baseline byte-일치 보장).
from scripts.downside_head_riskreward_v1 import (  # noqa: E402
    build_panel, add_cross_sectional, walk_forward_heads, _feats,
    UP_THRESH, DN_THRESH,
)
from scripts.regime_split_precursor_v1 import attach_btc_regime  # noqa: E402
from scripts.r2_challenger_compare_v1 import (  # noqa: E402
    UP_ANCHOR, DN_ANCHOR, RR_EPS, TOP_K, UNIVERSE_TOP_N, N_FOLDS, EMBARGO,
)

D1_DB = str(_ROOT / "data" / "upbit_d1.db")
OUT = _ROOT / "output"
EPS = 1e-12

ROUND_TRIP_COST = 0.0015
DEEP_LOSS = -0.05
ANN = 365

# --- C3 격자 (placeholder, CLAUDE.md §2.5) ---
HOLD_GRID = [1, 2, 3, 5]
TP_GRID = [0.08, 0.12]
SL_GRID = [0.05]
TRAIL_GRID = [0.08]
# 진입 필터(A1 dump_B re-selection cutoff 분위). 낮을수록 공격적 배제.
FILTER_Q = [0.60, 0.70, 0.80]
DUMP_NAME = "dump_B"                       # A1 에서 일관 우세 라벨
DUMP_LABELS = {"dump_B": dict(up=0.05, eod=-0.02)}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("cc_c3")


def _reject_contaminated_oos(frame: pd.DataFrame) -> None:
    excluded = sorted(
        {
            str(market)
            for market in frame["market"].dropna().unique()
            if is_excluded_signal_market(str(market))
        }
    )
    if excluded:
        raise RuntimeError(
            "cached C3 OOS contains excluded signal markets "
            f"{excluded}; rerun without --use-cache"
        )
    required_outcomes = ["up_high_ret", "down_low_ret", "eod_ret", "lab_dump_B"]
    missing = [column for column in required_outcomes if column not in frame]
    if missing or frame[required_outcomes].isna().any(axis=None):
        raise RuntimeError(
            "cached C3 OOS contains incomplete outcome labels; "
            "rerun without --use-cache"
        )


# ============================================================================
# 1. OOS 빌드 (A1 동일 빌더) — 캐시 가능
# ============================================================================
def add_dump_labels(panel):
    cols = []
    for name, sp in DUMP_LABELS.items():
        lab = f"lab_{name}"
        complete = panel[["up_high_ret", "eod_ret"]].notna().all(axis=1)
        panel[lab] = (
            (panel["up_high_ret"] >= sp["up"])
            & (panel["eod_ret"] < sp["eod"])
        ).astype(float)
        panel.loc[~complete, lab] = np.nan
        cols.append(lab)
    return cols


def build_oos(limit_markets):
    panel = build_panel(limit_markets)            # up_high_ret/down_low_ret/eod_ret 부착
    panel = add_cross_sectional(panel)
    panel = attach_btc_regime(panel)
    panel = panel[panel["f_qv_rank"] <= UNIVERSE_TOP_N].copy()
    panel["in_universe"] = True
    feats = _feats(panel)
    up_labels = [f"lab_{n}" for n in UP_THRESH]
    dn_labels = [f"lab_{n}" for n in DN_THRESH]
    dump_cols = add_dump_labels(panel)
    label_cols = up_labels + dn_labels + dump_cols
    log.info("panel(top%d) rows=%d markets=%d dates=%d feats=%d dump_base=%s",
             UNIVERSE_TOP_N, len(panel), panel["market"].nunique(),
             panel["date"].nunique(), len(feats),
             {c: round(float(panel[c].mean()), 4) for c in dump_cols})
    oos = walk_forward_heads(panel, feats, label_cols, N_FOLDS, EMBARGO)
    up = f"p_{UP_ANCHOR}"; dn = f"p_{DN_ANCHOR}"
    oos = oos.dropna(
        subset=[up, dn, "up_high_ret", "down_low_ret", "eod_ret", *dump_cols]
    ).copy()
    return oos


# ============================================================================
# 2. R1 정렬 + A1 re-selection (ch_sustainability_v1 와 동일 정신)
# ============================================================================
def r1_order(oos):
    up = f"p_{UP_ANCHOR}"; dn = f"p_{DN_ANCHOR}"
    d = oos.copy()
    d["R1"] = d[up] / np.maximum(d[dn], RR_EPS)
    d = d.sort_values(["date", "R1", "p_lab_dn_10", up, "exp_downside"],
                      ascending=[True, False, True, False, False])
    d["r1_rank"] = d.groupby("date").cumcount() + 1
    return d


def per_fold_cutoff(d_ordered, dump_col, q):
    """fold k cutoff = fold<k(train OOS) p_dumped 의 q 분위. fold0=inf(강등 안 함). OOF/leak X."""
    cutoffs = {}
    for k in sorted(d_ordered["fold"].unique()):
        prior = d_ordered[d_ordered["fold"] < k]
        if len(prior) < 200 or prior[dump_col].notna().sum() < 50:
            cutoffs[k] = np.inf
        else:
            cutoffs[k] = float(prior[dump_col].quantile(q))
    return cutoffs


def select_entries(d_ordered, dump_col, q, k):
    """A1 re-selection: p_dumped>cutoff 강등 후 다음 R1 후보로 top-k 채움. q=None → R1 그대로."""
    d = d_ordered.copy()
    if q is None:
        sel = d[d["r1_rank"] <= k].copy()
        sel["a1_substituted"] = False
        sel["a1_filled_from_demoted"] = False
        return sel
    cutoffs = per_fold_cutoff(d, dump_col, q)
    d["_cut"] = d["fold"].map(cutoffs)
    d["_dumpprone"] = (d[dump_col] > d["_cut"]).fillna(False)
    frames = []
    for dt, g in d.groupby("date"):
        g = g.sort_values("r1_rank")
        r1_set = set(g[g["r1_rank"] <= k]["market"])
        passed = g[~g["_dumpprone"]]
        rejected = g[g["_dumpprone"]]
        need = max(0, k - len(passed.head(k)))
        chosen = pd.concat([passed.head(k), rejected.head(need)])
        chosen = chosen.sort_values("r1_rank").head(k).copy()
        chosen["a1_filled_from_demoted"] = chosen["_dumpprone"].values
        chosen["a1_substituted"] = ~chosen["market"].isin(r1_set)
        frames.append(chosen)
    return pd.concat(frames, ignore_index=True)


# ============================================================================
# 3. 일봉 경로 로딩 + 멀티데이 청산 (B3 와 동일 + 갭반영 보수 모델)
# ============================================================================
def load_d1():
    conn = sqlite3.connect(D1_DB)
    df = pd.read_sql("SELECT market, timestamp, open, high, low, close, quote_volume "
                     "FROM candles", conn)
    conn.close()
    df = df[
        ~df["market"]
        .astype(str)
        .map(is_excluded_signal_market)
    ].copy()
    df["ts"] = pd.to_datetime(df["timestamp"])
    df["bar_date"] = df["ts"].dt.date
    df = df.sort_values(["market", "ts"]).reset_index(drop=True)
    return df


def build_basket(d1):
    df = d1.copy()
    df["qv_rank"] = df.groupby("bar_date")["quote_volume"].rank(ascending=False, method="first")
    top = df[df["qv_rank"] <= UNIVERSE_TOP_N].copy()
    top["day_ret"] = top["close"] / top["open"] - 1.0
    return top.groupby("bar_date")["day_ret"].mean().rename("mkt_day_ret")


def window_return(basket, start_date, n):
    fut = basket[basket.index >= start_date].iloc[:n]
    return float((1.0 + fut.values).prod() - 1.0) if len(fut) else np.nan


def build_market_arrays(d1):
    """market 별 (dates list, ohlc ndarray[N,4], date→pos dict). hot-path 벡터 슬라이싱용."""
    arrays = {}
    for m, g in d1.groupby("market"):
        g = g.sort_values("ts")
        dates = list(g["bar_date"].values)
        ohlc = g[["open", "high", "low", "close"]].to_numpy(dtype=float)
        pos = {d: i for i, d in enumerate(dates)}
        arrays[m] = (dates, ohlc, pos)
    return arrays


def get_hold_bars(arr, entry_date, n):
    """entry_date(정확 매칭) 부터 n 개 일봉. 반환 (entry_open, ohlc_slice ndarray[<=n,4])."""
    dates, ohlc, pos = arr
    i = pos.get(entry_date)
    if i is None:
        return None
    sl = ohlc[i:i + n]
    if len(sl) == 0:
        return None
    entry_open = float(sl[0, 0])
    if entry_open <= 0:
        return None
    return entry_open, sl


def sim_holdN(entry_open, bars):
    return float(bars[-1, 3]) / entry_open - 1.0, "holdN", len(bars)


def sim_bracket(entry_open, bars, tp, sl, gap_aware):
    """N일 내 +tp/-sl 먼저. 동봉 동시→SL(보수).
    gap_aware: 봉 open 이 이미 sl 아래로 갭다운했으면 그 open 가격(더 나쁨)에 체결.
               TP 는 갭업해도 레벨가 유지(보수 — TP 갭업 이득은 안 줌)."""
    tp_px = entry_open * (1.0 + tp)
    sl_px = entry_open * (1.0 - sl)
    for i in range(len(bars)):
        op, hi, lo = bars[i, 0], bars[i, 1], bars[i, 2]
        if gap_aware and op <= sl_px:                 # 갭다운으로 SL 통과 → open 체결(보수)
            return float(op) / entry_open - 1.0, "sl_gap", i + 1
        if lo <= sl_px:
            return -sl, "sl", i + 1                    # 레벨가 체결(낙관) — gap 모델은 위에서 처리됨
        if hi >= tp_px:
            return tp, "tp", i + 1
    return float(bars[-1, 3]) / entry_open - 1.0, "eod", len(bars)


def sim_trail(entry_open, bars, dd, gap_aware):
    """running-high 대비 -dd 트레일. gap_aware: 봉 open 이 trail_px 아래로 갭다운하면
    open 가격(더 나쁨)에 체결. 미발동 시 마지막 종가."""
    run_high = entry_open
    for i in range(len(bars)):
        op, hi, lo = bars[i, 0], bars[i, 1], bars[i, 2]
        trail_px = run_high * (1.0 - dd)
        if gap_aware and op <= trail_px:
            return float(op) / entry_open - 1.0, "trail_gap", i + 1
        if lo <= trail_px:
            return trail_px / entry_open - 1.0, "trail", i + 1
        if hi > run_high:
            run_high = float(hi)
    return float(bars[-1, 3]) / entry_open - 1.0, "trail_eod", len(bars)


# ============================================================================
# 4. 지표 (net 차감) — B3 와 동일(block 보정 포함)
# ============================================================================
def metrics(trades, n_hold):
    d = trades.dropna(subset=["net"]).copy()
    if len(d) == 0:
        return {}
    net = d["net"].values
    n = len(net)
    mu = float(net.mean())
    daily = d.groupby("entry_date")["net"].mean().sort_index()
    eq = (1 + daily).cumprod()
    peak = eq.cummax()
    mdd = float(((eq - peak) / peak).min())
    cum = float(eq.iloc[-1] - 1.0)
    sd = float(daily.std())
    dstd = float(daily[daily < 0].std()) if (daily < 0).any() else np.nan
    sharpe_nom = float(mu / sd * np.sqrt(ANN)) if sd and sd > 0 else np.nan
    sortino_nom = float(mu / dstd * np.sqrt(ANN)) if dstd and dstd > 0 else np.nan
    # non-overlapping block
    days_sorted = list(daily.index)
    block_id = {dt: i // n_hold for i, dt in enumerate(days_sorted)}
    db = d.copy()
    db["block"] = db["entry_date"].map(block_id)
    block_ret = db.groupby("block")["net"].mean()
    n_blocks = int(block_ret.shape[0])
    bsd = float(block_ret.std())
    bpy = ANN / n_hold
    sharpe_block = float(block_ret.mean() / bsd) * np.sqrt(bpy) if bsd and bsd > 0 else np.nan
    bneg = block_ret[block_ret < 0]
    bdsd = float(bneg.std()) if len(bneg) else np.nan
    sortino_block = float(block_ret.mean() / bdsd) * np.sqrt(bpy) if bdsd and bdsd > 0 else np.nan
    oc = d["outcome"]
    k5 = max(1, int(np.ceil(0.05 * n)))
    cvar95 = float(np.sort(net)[:k5].mean())
    excess = d["excess_mkt"].dropna().values
    sl_like = oc.isin(["sl", "sl_gap"])
    return dict(
        n=n, n_entry_days=int(d["entry_date"].nunique()), n_blocks=n_blocks,
        avg_hold_days=float(d["hold_days"].mean()),
        net_mean=mu, net_median=float(np.median(net)),
        hit=float((net > 0).mean()),
        deep_loss_freq=float((net <= DEEP_LOSS).mean()),
        worst=float(net.min()), cvar95=cvar95, mdd=mdd, cum=cum,
        sharpe_block=sharpe_block, sortino_block=sortino_block,
        sharpe_nom=sharpe_nom, sortino_nom=sortino_nom,
        pct_tp=float((oc == "tp").mean()), pct_sl=float(sl_like.mean()),
        pct_gap_exit=float(oc.isin(["sl_gap", "trail_gap"]).mean()),
        excess_mkt_mean=float(np.mean(excess)) if len(excess) else np.nan,
        excess_mkt_pos_frac=float((excess > 0).mean()) if len(excess) else np.nan,
        btc_window_mean=float(d["btc_window_ret"].mean()) if "btc_window_ret" in d else np.nan,
    )


# ============================================================================
# main
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-markets", type=int, default=None)
    ap.add_argument("--use-cache", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    cache = OUT / "cc_filtered_multiday_oos_v1.parquet"

    # --- 1) OOS (A1 동일 빌더) ---
    if args.use_cache and cache.exists():
        oos = pd.read_parquet(cache)
        _reject_contaminated_oos(oos)
        oos["date"] = pd.to_datetime(oos["date"]).dt.date
        log.info("loaded cached OOS rows=%d", len(oos))
    else:
        oos = build_oos(args.limit_markets)
        if oos.empty:
            log.error("empty OOS — abort"); sys.exit(1)
        oos.to_parquet(cache, index=False)
    log.info("OOS rows=%d dates %s..%s folds=%s",
             len(oos), oos["date"].min(), oos["date"].max(),
             sorted(oos["fold"].unique()))

    ordered = r1_order(oos)
    dump_col = f"p_lab_{DUMP_NAME}"
    if dump_col not in ordered.columns:
        log.error("missing %s — abort", dump_col); sys.exit(1)

    # --- 2) 진입집합 4종 ---
    entry_sets = {"R1_unfiltered": None}
    for q in FILTER_Q:
        entry_sets[f"C3_filt_q{q}"] = q
    picks_by_set = {}
    for name, q in entry_sets.items():
        sel = select_entries(ordered, dump_col, q, TOP_K)
        sel["entry_date"] = pd.to_datetime(sel["date"]).dt.date
        picks_by_set[name] = sel
        log.info("entry set %-14s n=%d days=%d sub=%.2f median_qv=%.0f frac_top10qv=%.3f "
                 "pump5_rate=%.3f mean_up=%.4f mean_dn=%.4f",
                 name, len(sel), sel["entry_date"].nunique(),
                 float(sel.get("a1_substituted", pd.Series([0])).mean()),
                 float(sel["f_qv_rank"].median()),
                 float((sel["f_qv_rank"] <= 10).mean()),
                 float((sel["up_high_ret"] >= 0.05).mean()),
                 float(sel["up_high_ret"].mean()), float(sel["down_low_ret"].mean()))

    # --- 3) 일봉 경로 ---
    d1 = load_d1()
    basket = build_basket(d1)
    arrays = build_market_arrays(d1)
    btc = d1[d1["market"] == "KRW-BTC"].sort_values("ts")
    btc_dr = (pd.Series((btc["close"] / btc["open"] - 1.0).values,
                        index=btc["bar_date"].values).sort_index()
              if len(btc) else None)

    # --- 4) 청산변형 (gap_aware on/off 별도 변형) ---
    variants = [("holdN", None, None)]
    for tp in TP_GRID:
        for sl in SL_GRID:
            variants.append(("bracket", tp, sl))
    for dd in TRAIL_GRID:
        variants.append(("trail", dd, None))
    # holdN 은 갭개념 없음(종가청산) → gap 모델 1개. bracket/trail 은 opt/gap 둘 다.

    rows = []
    trade_rows = []
    DUMP_KEY = ("bracket", 0.12, 0.05)   # trade-level 덤프할 대표 청산

    n_combos = 0
    for set_name, picks in picks_by_set.items():
        for n_hold in HOLD_GRID:
            for (vname, p1, p2) in variants:
                gap_modes = [False] if vname == "holdN" else [False, True]
                for gap_aware in gap_modes:
                    n_combos += 1
                    recs = []
                    mk_arr = picks["market"].values
                    ed_arr = picks["entry_date"].values
                    for ri in range(len(picks)):
                        m, ed = mk_arr[ri], ed_arr[ri]
                        arr = arrays.get(m)
                        if arr is None:
                            continue
                        hb = get_hold_bars(arr, ed, n_hold)
                        if hb is None:
                            continue
                        entry_open, bars = hb
                        if vname == "holdN":
                            gross, oc, used = sim_holdN(entry_open, bars)
                        elif vname == "bracket":
                            gross, oc, used = sim_bracket(entry_open, bars, p1, p2, gap_aware)
                        else:
                            gross, oc, used = sim_trail(entry_open, bars, p1, gap_aware)
                        net = gross - ROUND_TRIP_COST
                        mkt = window_return(basket, ed, n_hold)
                        btc_w = (float((1.0 + btc_dr[btc_dr.index >= ed].iloc[:n_hold]).prod() - 1.0)
                                 if btc_dr is not None and len(btc_dr[btc_dr.index >= ed]) else np.nan)
                        rec = dict(set_name=set_name, entry_date=ed, market=m, n_hold=n_hold,
                                   variant=vname, p1=p1, p2=p2, gap_aware=gap_aware,
                                   entry_open=entry_open, net=net, gross=gross,
                                   outcome=oc, hold_days=used,
                                   mkt_window_ret=mkt, btc_window_ret=btc_w,
                                   excess_mkt=(net - mkt) if pd.notna(mkt) else np.nan)
                        recs.append(rec)
                        if (vname, p1, p2) == DUMP_KEY and gap_aware:
                            trade_rows.append(rec)
                    tr = pd.DataFrame(recs)
                    mm = metrics(tr, n_hold)
                    if not mm:
                        continue
                    exitname = (vname if vname == "holdN"
                                else f"{vname}_tp{p1}_sl{p2}" if vname == "bracket"
                                else f"{vname}_dd{p1}")
                    fill = "gap" if gap_aware else "opt"
                    mm.update(set_name=set_name, filter_q=(entry_sets[set_name] or 0.0),
                              n_hold=n_hold, exit=exitname, fill=fill, vfamily=vname)
                    rows.append(mm)
                    log.info("%-14s N=%d %-18s %s n=%d blk=%d hold=%.1f net=%+.4f hit=%.2f "
                             "deep=%.3f worst=%+.3f Sh_blk=%s exMkt=%s gapEx=%.2f",
                             set_name, n_hold, exitname, fill, mm["n"], mm["n_blocks"],
                             mm["avg_hold_days"], mm["net_mean"], mm["hit"],
                             mm["deep_loss_freq"], mm["worst"],
                             f"{mm['sharpe_block']:+.2f}" if pd.notna(mm["sharpe_block"]) else "nan",
                             f"{mm['excess_mkt_mean']:+.4f}" if pd.notna(mm["excess_mkt_mean"]) else "nan",
                             mm["pct_gap_exit"])

    res = pd.DataFrame(rows)
    lead = ["set_name", "filter_q", "n_hold", "exit", "fill", "vfamily",
            "n", "n_blocks", "n_entry_days", "avg_hold_days",
            "net_mean", "net_median", "hit", "deep_loss_freq", "worst", "cvar95",
            "mdd", "cum", "sharpe_block", "sortino_block", "sharpe_nom", "sortino_nom",
            "pct_tp", "pct_sl", "pct_gap_exit",
            "excess_mkt_mean", "excess_mkt_pos_frac", "btc_window_mean"]
    res = res[[c for c in lead if c in res.columns]]
    res = res.sort_values(["set_name", "n_hold", "exit", "fill"]).reset_index(drop=True)
    res.to_csv(OUT / "cc_filtered_multiday_compare_v1.csv", index=False)
    pd.DataFrame(trade_rows).to_csv(OUT / "cc_filtered_multiday_picks_v1.csv", index=False)

    # coverage / meta
    cov = dict(
        oos_rows=int(len(oos)), oos_dates=int(oos["date"].nunique()),
        oos_window=[str(oos["date"].min()), str(oos["date"].max())],
        universe="static_top100", n_folds=N_FOLDS, embargo=EMBARGO,
        dump_label=DUMP_NAME, filter_q_grid=FILTER_Q, hold_grid=HOLD_GRID,
        tp_grid=TP_GRID, sl_grid=SL_GRID, trail_grid=TRAIL_GRID,
        n_combos_total=int(n_combos),
        cost_round_trip=ROUND_TRIP_COST, deep_loss_thresh=DEEP_LOSS,
        entry_set_coverage={name: dict(
            n_picks=int(len(p)), n_days=int(p["entry_date"].nunique()),
            frac_substituted=float(p.get("a1_substituted", pd.Series([0])).mean()))
            for name, p in picks_by_set.items()},
    )
    (OUT / "cc_filtered_multiday_coverage_v1.json").write_text(json.dumps(cov, indent=2))

    # ---- 요약: net>0 + 베이스라인 대비 ----
    log.info("\n===== net_mean>0 조합 (gap 우선 보고) =====")
    pos = res[res["net_mean"] > 0].sort_values("net_mean", ascending=False)
    if len(pos):
        for _, r in pos.head(20).iterrows():
            log.info("  %-14s N=%d %-18s %s net=%+.4f Sh_blk=%s deep=%.3f worst=%+.3f exMkt=%s",
                     r["set_name"], int(r["n_hold"]), r["exit"], r["fill"], r["net_mean"],
                     f"{r['sharpe_block']:+.2f}" if pd.notna(r["sharpe_block"]) else "nan",
                     r["deep_loss_freq"], r["worst"],
                     f"{r['excess_mkt_mean']:+.4f}" if pd.notna(r["excess_mkt_mean"]) else "nan")
    else:
        log.info("  (없음 — 어떤 (필터,N,청산,체결) 조합도 net_mean>0 아님)")

    # R1 N=1 baseline 대비 best C3
    base = res[(res["set_name"] == "R1_unfiltered") & (res["n_hold"] == 1)]
    log.info("\n===== R1_unfiltered N=1 baseline (참조) =====")
    for _, r in base.iterrows():
        log.info("  %-18s %s net=%+.4f deep=%.3f worst=%+.3f Sh_blk=%s",
                 r["exit"], r["fill"], r["net_mean"], r["deep_loss_freq"], r["worst"],
                 f"{r['sharpe_block']:+.2f}" if pd.notna(r["sharpe_block"]) else "nan")
    log.info("DONE. wrote cc_filtered_multiday_{compare,picks,coverage}_v1.*  (combos=%d)", n_combos)


if __name__ == "__main__":
    main()
