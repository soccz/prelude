#!/usr/bin/env python3
"""ch_multiday_v1 — 멀티데이 홀딩 청산 sim (Track B3).

목표(메타-발견 우회 축):
  R1 진입(1일 보유 + 하드 -3%SL/+5%TP/EOD)에서는 수익꼬리·손실꼬리가 같은 진입에
  묶여 분리 불가 → net 흑자전환 0 으로 6실험이 수렴했다. 천장을 우회하는 유일한
  구조적 축 = **보유기간을 늘려(N일) 수익꼬리에 시간을 주는 것**. 1일 SL-floor 가
  winner 를 조기 절단하는 구조를 바꾼다.

가설:
  R1 top-3 진입을 N일(2/3/5) 보유하면, +5%TP 에 일찍 막히던 winner 가 더 크게 달려
  net 이 양수로 갈 수 있다. 하방도 늘 수 있으니(더 오래 노출) net·하방 trade-off 를
  정직하게 측정. 사용자 선호: 하락위험 최소화 > 상승, 거래당 ~-5% 수용.

진입 집합(고정 — 새로 모델 학습 X, leak 우회):
  output/r2_challenger_picks_v1.csv 의 policy=='R1_ratio' 행 = R1 daily top-3 픽.
  (765일 OOS, 1531 entries). 진입가 = day-D 일봉 open(timestamp '{date} 09:00:00').

청산 변형(작은 격자, 모두 일봉 d1 경로):
  (a) holdN     : N일 보유 후 마지막날 종가 청산.
  (b) bracket   : N일 내 +TP%/-SL% 먼저 닿으면 그날 청산, 아니면 N일 종가.
                  TP∈{0.08,0.12,0.20}, SL∈{0.05,0.08}.  (1일 baseline 의 TP5/SL3 대비
                  더 넓은 격자 — 수익꼬리 신장이 목적이므로 TP 를 멀리, SL 도 사용자 수용 ~5%)
  (c) trail     : N일 running-high 대비 -DD% 트레일링 스탑(첫날 진입가 기준 시작),
                  미발동 시 N일 종가.  DD∈{0.08,0.12}.

★ leak (4/4 양보불가):
  - 진입 결정은 D-1 까지(R1 픽 그대로 — 재학습 안 함). 진입가 day-D open 은 관측가능.
  - 청산 경로 = day-D open ~ D+N-1 (미래=outcome, 정상 forward).
  - 같은 봉 SL·TP 동시 도달 → SL 먼저(보수). intrabar 경로 모르므로 high/low 둘 다 닿으면
    나쁜 쪽(SL) 우선.
  - 비용 왕복 0.15% 1회(멀티데이도 1 거래).
  - 유니버스 시간정합성: 픽 자체가 fold train 종료시점 top100 으로 이미 제한된 OOS.

베타 분리:
  같은 N일 윈도우의 BTC buy&hold 수익(net), 그리고 top100 equal-weight 시장바스켓
  수익을 각 trade 에 부착 → excess(picks - market) 보고. net 개선이 진짜 엣지인지
  단순 보유프리미엄/베타인지 분리.

overlapping 표본:
  N일 홀딩은 D, D+1 픽이 시간상 겹침 → 유효표본 < 명목표본. 거래일-블록(non-overlapping
  진입일 그룹) 단위 분산/Sharpe 도 보고하고 유효표본 감소를 명시.

출력:
  output/ch_multiday_compare_v1.csv  (변형×N 별 지표)
  output/ch_multiday_picks_v1.csv    (trade-level: entry/exit/net/hold_days/outcome/excess)
"""
from __future__ import annotations

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

D1_DB = str(_ROOT / "data" / "upbit_d1.db")
OUT = _ROOT / "output"
PICKS = OUT / "r2_challenger_picks_v1.csv"

ROUND_TRIP_COST = 0.0015     # 왕복 0.15% (1 거래 1회)
DEEP_LOSS = -0.05            # 사용자 기준 '나쁜 사건' 임계
ANN = 365                    # 코인 연중무휴

HOLD_GRID = [1, 2, 3, 5]     # N=1 은 holdN 1일종가 = 베이스라인 참조(15m baseline 과 별개)
TP_GRID = [0.08, 0.12, 0.20]
SL_GRID = [0.05, 0.08]
TRAIL_GRID = [0.08, 0.12]
UNIVERSE_TOP_N = 100         # 시장바스켓 베타 = 일별 top100(quote_volume) equal-weight

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("ch_multiday")


def _excluded_signal_markets(frame: pd.DataFrame) -> list[str]:
    return sorted(
        {
            str(market)
            for market in frame["market"].dropna().unique()
            if is_excluded_signal_market(str(market))
        }
    )


def _reject_contaminated_picks(picks: pd.DataFrame) -> None:
    excluded = _excluded_signal_markets(picks)
    if excluded:
        raise RuntimeError(
            "R2 picks contain excluded signal markets "
            f"{excluded}; regenerate r2_challenger_picks_v1.csv from the "
            "canonical research universe"
        )


# ============================================================================
# 1. 일봉 경로 로딩
# ============================================================================
def load_d1() -> pd.DataFrame:
    """모든 일봉을 (market, bar_date) 인덱스로. bar_date = timestamp 의 날짜(09:00 KST 봉)."""
    conn = sqlite3.connect(D1_DB)
    df = pd.read_sql(
        "SELECT market, timestamp, open, high, low, close, quote_volume FROM candles",
        conn)
    conn.close()
    df = df[
        ~df["market"]
        .astype(str)
        .map(is_excluded_signal_market)
    ].copy()
    df["ts"] = pd.to_datetime(df["timestamp"])
    df["bar_date"] = df["ts"].dt.date
    df = df.sort_values(["market", "ts"]).reset_index(drop=True)
    # market 별로 거래일 시퀀스 위치(연속 N일 슬라이싱용). 결측일은 자연스럽게 건너뜀.
    df["seq"] = df.groupby("market").cumcount()
    return df


def build_market_basket(d1: pd.DataFrame) -> pd.DataFrame:
    """각 bar_date 의 top100(quote_volume) equal-weight 당일 수익률(close/open-1) 및
    open->open N일 보유용 close 시퀀스. 베타 proxy.

    주의: 시간정합성 — 유니버스 랭크는 '그 날' quote_volume 기준(라이브와 동일하게
    당일 거래대금으로 top100). 진입은 그 날 open, 측정은 forward 라 leak 아님.
    """
    df = d1.copy()
    df["qv_rank"] = df.groupby("bar_date")["quote_volume"].rank(
        ascending=False, method="first")
    top = df[df["qv_rank"] <= UNIVERSE_TOP_N].copy()
    top["day_ret"] = top["close"] / top["open"] - 1.0
    basket = top.groupby("bar_date")["day_ret"].mean().rename("mkt_day_ret")
    return basket


def market_window_return(seq_df: pd.DataFrame, basket: pd.Series,
                         start_date, n: int) -> float:
    """진입일 start_date 부터 n 거래일(시장바스켓) 누적 보유수익(beta proxy).
    bar_date 기준 start_date 이상에서 최대 n 일 합성. 비용 미차감(엣지 비교는 net-net 으로
    별도; 여기선 베타 '총수익' proxy 로 그대로 사용해 excess 의 부호 의미를 단순화)."""
    fut = basket[basket.index >= start_date].iloc[:n]
    if len(fut) == 0:
        return np.nan
    return float((1.0 + fut.values).prod() - 1.0)


# ============================================================================
# 2. 멀티데이 청산 sim (3 변형)
# ============================================================================
def get_hold_bars(g: pd.DataFrame, entry_date, n: int):
    """market group g(시간순)에서 entry_date(포함) 부터 n 개 일봉. 진입가 = 첫 봉 open.
    반환: (entry_open, [(high,low,close), ...] 길이<=n) 또는 None."""
    sub = g[g["bar_date"] >= entry_date].iloc[:n]
    if len(sub) == 0 or sub.iloc[0]["bar_date"] != entry_date:
        return None
    entry_open = float(sub.iloc[0]["open"])
    if entry_open <= 0:
        return None
    bars = list(zip(sub["high"].astype(float), sub["low"].astype(float),
                    sub["close"].astype(float)))
    return entry_open, bars, len(sub)


def sim_holdN(entry_open, bars):
    """N일 보유 후 마지막 봉 종가 청산. gross = lastClose/entry - 1."""
    last_close = bars[-1][2]
    return last_close / entry_open - 1.0, "holdN", len(bars)


def sim_bracket(entry_open, bars, tp, sl):
    """N일 내 +tp / -sl 먼저 닿으면 그날 청산(동봉 동시 → SL 먼저, 보수).
    미발동 시 마지막 봉 종가. day-1(진입일)도 intrabar 평가 포함(진입 후 같은 날 변동)."""
    tp_px = entry_open * (1.0 + tp)
    sl_px = entry_open * (1.0 - sl)
    for i, (hi, lo, cl) in enumerate(bars):
        hit_sl = lo <= sl_px
        hit_tp = hi >= tp_px
        if hit_sl:                       # 보수: 동봉 SL·TP 동시 → SL
            return -sl, "sl", i + 1
        if hit_tp:
            return tp, "tp", i + 1
    return bars[-1][2] / entry_open - 1.0, "eod", len(bars)


def sim_trail(entry_open, bars, dd):
    """running-high(진입가 포함) 대비 -dd 트레일링 스탑. 발동 봉에서 트레일가로 청산
    (보수: 그 봉 low 가 트레일가 이하로 내려갔으니 트레일가 체결로 본다 — 슬리피지는
    ROUND_TRIP_COST 에 일부 반영). 미발동 시 마지막 봉 종가."""
    run_high = entry_open
    for i, (hi, lo, cl) in enumerate(bars):
        trail_px = run_high * (1.0 - dd)
        if lo <= trail_px:
            return trail_px / entry_open - 1.0, "trail", i + 1
        run_high = max(run_high, hi)     # 봉 마감 후 고점 갱신(보수: 청산판정 먼저)
    return bars[-1][2] / entry_open - 1.0, "trail_eod", len(bars)


# ============================================================================
# 3. 지표 (net 차감 후) — 하방-우선 + net 개선 동시 평가
# ============================================================================
def metrics(trades: pd.DataFrame, n_hold: int) -> dict:
    d = trades.dropna(subset=["net"]).copy()
    if len(d) == 0:
        return {}
    net = d["net"].values
    n = len(net)
    mu = float(net.mean())

    # ---- 명목(overlapping) 일별 집계: 진입일 단위 평균 net ----
    daily = d.groupby("entry_date")["net"].mean().sort_index()
    eq = (1 + daily).cumprod()
    peak = eq.cummax()
    mdd = float(((eq - peak) / peak).min())
    cum = float(eq.iloc[-1] - 1.0)
    sd = float(daily.std())
    dstd = float(daily[daily < 0].std()) if (daily < 0).any() else np.nan
    # 명목 Sharpe(진입일 단위, overlapping → 낙관 편향 가능 → 아래 block 으로 보정)
    sharpe_nom = float(mu / sd * np.sqrt(ANN)) if sd and sd > 0 else np.nan
    sortino_nom = float(mu / dstd * np.sqrt(ANN)) if dstd and dstd > 0 else np.nan

    # ---- non-overlapping block: 진입일을 n_hold 간격으로 묶어 유효표본 추정 ----
    #   같은 블록(겹치는 holding) 내 진입은 시간상 종속 → 블록 1개로 압축.
    days_sorted = list(daily.index)
    block_id = {dt: i // n_hold for i, dt in enumerate(days_sorted)}
    db = d.copy()
    db["block"] = db["entry_date"].map(block_id)
    block_ret = db.groupby("block")["net"].mean()
    n_blocks = int(block_ret.shape[0])
    bsd = float(block_ret.std())
    # 블록 수익률을 '보유일=n_hold' 1회로 보고 연환산: 연간 블록 수 ~ ANN / n_hold
    blocks_per_year = ANN / n_hold
    sharpe_block = (float(block_ret.mean() / bsd) * np.sqrt(blocks_per_year)
                    if bsd and bsd > 0 else np.nan)
    bneg = block_ret[block_ret < 0]
    bdsd = float(bneg.std()) if len(bneg) else np.nan
    sortino_block = (float(block_ret.mean() / bdsd) * np.sqrt(blocks_per_year)
                     if bdsd and bdsd > 0 else np.nan)

    oc = d["outcome"]
    k5 = max(1, int(np.ceil(0.05 * n)))
    cvar95 = float(np.sort(net)[:k5].mean())
    excess = d["excess_mkt"].dropna().values
    return dict(
        n=n,
        n_entry_days=int(d["entry_date"].nunique()),
        n_blocks=n_blocks,                       # 유효표본 ~ 이 값
        avg_hold_days=float(d["hold_days"].mean()),
        net_mean=mu,                             # ★ net 개선 핵심
        net_median=float(np.median(net)),
        hit=float((net > 0).mean()),
        deep_loss_freq=float((net <= DEEP_LOSS).mean()),   # 사용자 1순위 하방
        worst=float(net.min()),
        cvar95=cvar95,
        mdd=mdd,
        cum=cum,
        sharpe_nom=sharpe_nom,                   # overlapping(낙관 편향 주의)
        sortino_nom=sortino_nom,
        sharpe_block=sharpe_block,               # ★ 유효표본 보정 Sharpe
        sortino_block=sortino_block,
        pct_tp=float((oc == "tp").mean()) if (oc == "tp").any() else 0.0,
        pct_sl=float((oc == "sl").mean()) if (oc == "sl").any() else 0.0,
        excess_mkt_mean=float(np.mean(excess)) if len(excess) else np.nan,
        excess_mkt_pos_frac=float((excess > 0).mean()) if len(excess) else np.nan,
    )


# ============================================================================
# 4. main
# ============================================================================
def main():
    OUT.mkdir(exist_ok=True)
    picks = pd.read_csv(PICKS)
    _reject_contaminated_picks(picks)
    picks = picks[picks["policy"] == "R1_ratio"].copy()
    picks["entry_date"] = pd.to_datetime(picks["date"]).dt.date
    log.info("R1 entry set: %d entries, %d days, %d markets",
             len(picks), picks["entry_date"].nunique(), picks["market"].nunique())

    d1 = load_d1()
    basket = build_market_basket(d1)
    log.info("d1 loaded: %d rows %d markets; basket days=%d",
             len(d1), d1["market"].nunique(), len(basket))
    groups = {m: g.reset_index(drop=True) for m, g in d1.groupby("market")}

    # BTC 베타 proxy: BTC open->... N일 보유 수익
    btc = groups.get("KRW-BTC")
    btc_basket = (btc.set_index("bar_date")["close"] / btc.set_index("bar_date")["open"] - 1.0
                  if btc is not None else None)

    variants = []
    variants.append(("holdN", None, None))
    for tp in TP_GRID:
        for sl in SL_GRID:
            variants.append(("bracket", tp, sl))
    for dd in TRAIL_GRID:
        variants.append(("trail", dd, None))

    rows = []          # metrics
    trade_rows = []     # trade-level dump (대표 변형만 덤프해 파일 비대화 방지)
    DUMP_VARIANTS = {("holdN", None, None), ("bracket", 0.12, 0.05), ("trail", 0.08, None)}

    for n_hold in HOLD_GRID:
        for (vname, p1, p2) in variants:
            recs = []
            for _, r in picks.iterrows():
                m, ed = r["market"], r["entry_date"]
                g = groups.get(m)
                if g is None:
                    continue
                hb = get_hold_bars(g, ed, n_hold)
                if hb is None:
                    continue
                entry_open, bars, got = hb
                if vname == "holdN":
                    gross, oc, used = sim_holdN(entry_open, bars)
                elif vname == "bracket":
                    gross, oc, used = sim_bracket(entry_open, bars, p1, p2)
                else:
                    gross, oc, used = sim_trail(entry_open, bars, p1)
                net = gross - ROUND_TRIP_COST
                # 베타: 같은 윈도우 시장바스켓 보유수익(총수익 proxy), excess = picks_net - mkt
                mkt = market_window_return(g, basket, ed, n_hold)
                btc_w = (float((1.0 + btc_basket[btc_basket.index >= ed].iloc[:n_hold]).prod() - 1.0)
                         if btc_basket is not None and
                         len(btc_basket[btc_basket.index >= ed]) else np.nan)
                rec = dict(entry_date=ed, market=m, n_hold=n_hold,
                           variant=vname, p1=p1, p2=p2,
                           entry_open=entry_open, net=net, gross=gross,
                           outcome=oc, hold_days=used,
                           mkt_window_ret=mkt, btc_window_ret=btc_w,
                           excess_mkt=(net - mkt) if pd.notna(mkt) else np.nan)
                recs.append(rec)
                if (vname, p1, p2) in DUMP_VARIANTS:
                    trade_rows.append(rec)
            tr = pd.DataFrame(recs)
            m = metrics(tr, n_hold)
            if not m:
                continue
            label = (vname if vname == "holdN"
                     else f"{vname}_tp{p1}_sl{p2}" if vname == "bracket"
                     else f"{vname}_dd{p1}")
            m.update(variant=label, n_hold=n_hold, vfamily=vname,
                     tp=p1 if vname == "bracket" else np.nan,
                     sl=p2 if vname == "bracket" else np.nan,
                     trail_dd=p1 if vname == "trail" else np.nan)
            rows.append(m)
            log.info("N=%d %-22s n=%d blk=%d hold=%.1f net=%+.4f hit=%.2f "
                     "deep=%.3f Sh_blk=%s cum=%+.3f exMkt=%s",
                     n_hold, label, m["n"], m["n_blocks"], m["avg_hold_days"],
                     m["net_mean"], m["hit"], m["deep_loss_freq"],
                     f"{m['sharpe_block']:+.2f}" if pd.notna(m["sharpe_block"]) else "nan",
                     m["cum"],
                     f"{m['excess_mkt_mean']:+.4f}" if pd.notna(m["excess_mkt_mean"]) else "nan")

    res = pd.DataFrame(rows)
    lead = ["n_hold", "variant", "vfamily", "n", "n_blocks", "n_entry_days",
            "avg_hold_days", "net_mean", "net_median", "hit", "deep_loss_freq",
            "worst", "cvar95", "mdd", "cum",
            "sharpe_block", "sortino_block", "sharpe_nom", "sortino_nom",
            "pct_tp", "pct_sl", "excess_mkt_mean", "excess_mkt_pos_frac",
            "tp", "sl", "trail_dd"]
    res = res[[c for c in lead if c in res.columns]]
    res = res.sort_values(["vfamily", "n_hold", "variant"]).reset_index(drop=True)
    res.to_csv(OUT / "ch_multiday_compare_v1.csv", index=False)
    pd.DataFrame(trade_rows).to_csv(OUT / "ch_multiday_picks_v1.csv", index=False)
    log.info("wrote %s (%d rows) and ch_multiday_picks_v1.csv (%d trades)",
             OUT / "ch_multiday_compare_v1.csv", len(res), len(trade_rows))

    # ---- 요약: net 유의 양수/개선 조합 강조 ----
    log.info("\n===== net 개선 후보 (net_mean>0 정렬) =====")
    pos = res[res["net_mean"] > 0].sort_values("net_mean", ascending=False)
    if len(pos):
        for _, r in pos.head(12).iterrows():
            log.info("  N=%d %-22s net=%+.4f Sh_blk=%s deep=%.3f exMkt=%s n_blk=%d",
                     int(r["n_hold"]), r["variant"], r["net_mean"],
                     f"{r['sharpe_block']:+.2f}" if pd.notna(r["sharpe_block"]) else "nan",
                     r["deep_loss_freq"],
                     f"{r['excess_mkt_mean']:+.4f}" if pd.notna(r["excess_mkt_mean"]) else "nan",
                     int(r["n_blocks"]))
    else:
        log.info("  (없음 — 어떤 N·청산 조합도 net_mean>0 아님)")


if __name__ == "__main__":
    main()
