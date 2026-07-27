"""Exit-discipline challenger vs R1 champion exit (baseline) — OOS net, FIXED R1 entries.

가설 (track B2):
  R1 챔피언의 **똑같은 top-3 진입을 고정**하고, 고정 -3%SL/+5%TP/EOD 청산 대신
  더 똑똑한 청산 규칙만 바꾸면 net·하방이 개선된다. 진입을 안 바꾸므로 (R2 처럼)
  "안 움직이는 대형주로 도망가는" degeneracy 가 구조적으로 불가능 — 순수 EXIT 효과만 격리.

진입 집합 (FIXED, 재사용):
  output/r2_challenger_picks_v1.csv 의 policy=R1_ratio (1531 picks, 765 일, top-3/day, OOS).
  진입 = day-D open = [09:00 UTC D, 09:00 UTC D+1) 윈도의 첫 15m 봉 open
  (= d1 day-open 과 byte-aligned; 실측 확인). 진입 결정은 D-1 까지 정보 → leak 아님.

청산 룰 sweep (모두 시간순 walk, 각 봉의 청산 판단은 그 봉 시점까지 정보만):
  baseline : hard_sl 0.03 / tp 0.05 / EOD              (현 챔피언 청산 — 기준선)
  trail    : hard_sl 0.03 floor + TP-arm trailing       (+arm% 도달 후 고점-trail% 청산; winner run)
  time     : hard_sl 0.03 / tp 0.05 + first-1h time-stop (1h 내 TP 미달이면 봉종가 청산; dump 꼬리 컷)
  vol      : SL/TP 를 D-1 ATR% 로 스케일 (calm 넓게, volatile 좁게)
  regime   : regime 별 (bear_volatile 타이트·빠름, bull 느슨)

LEAK 방어 (양보불가):
  - 입력 피처/진입 = R1 (D-1 까지, 이미 leak-free). 진입 집합 그대로 재사용.
  - 청산 결정 causal: 트레일링 = 현재까지 running_high (미래 고점 X), arm 도 현재까지 도달여부.
    time-stop = 경과 봉 수. vol = 진입 시점(D-1) ATR. 같은 봉 SL·TP 동시 → SL 먼저(보수).
  - day-D 경로는 in-trade outcome 이므로 leak 아님. 미래 봉 미리보기 0 (시간 오름차순 walk).
  - causal-exit 자가검증: assert_causal() 가 각 룰을 prefix-only 로 재계산해 동일성 확인.

거래비용: 왕복 0.15% 차감 (net = gross - 0.0015). gross/IS 금지.

사용:
    python scripts/exit_challenger_compare_v1.py
"""
from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.database import list_markets, load_candles
from data.market_universe import signal_eligible_markets
from scripts.univariate_precursor_lift_v1 import build_market_features, add_cross_sectional

M15_DB = "data/upbit_15m.db"
D1_DB = "data/upbit_d1.db"
R1_PICKS = "output/r2_challenger_picks_v1.csv"
OUT_COMPARE = "output/exit_challenger_compare_v1.csv"
OUT_TRADES = "output/exit_challenger_trades_v1.csv"

ROUND_TRIP_COST = 0.0015   # 왕복 0.15%
BARS_PER_HOUR = 4          # 15m
EPS = 1e-12

# 챔피언 청산 (불변 — baseline)
CHAMP_SL = 0.03
CHAMP_TP = 0.05

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("exit_chal")


# ============================================================================
# 1. 15m 경로 로드 (R1 진입과 동일 윈도)
# ============================================================================
def load_paths(pairs: pd.DataFrame) -> dict:
    """[(market,date)] → 15m (open,high,low,close) 리스트, 시간 오름차순.
    윈도 = [09:00 UTC D, 09:00 UTC D+1) (= R1 simulate_path 와 동일)."""
    conn = sqlite3.connect(M15_DB)
    paths = {}
    for _, r in pairs.drop_duplicates(["market", "date"]).iterrows():
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


# ============================================================================
# 2. D-1 ATR% (vol 룰 스케일용 — leak-free, shift(1))
# ============================================================================
def build_atr_lookup() -> dict:
    """(market, date) → f_atr_pct_14 (D-1 ATR%) + f_atr_xs_decile (그날 횡단 ATR rank).
    build_market_features 의 shift(1) raw → leak 아님."""
    markets = signal_eligible_markets(list_markets(D1_DB))
    frames = []
    for m in markets:
        df = load_candles(D1_DB, m)
        if df is None or len(df) < 70:
            continue
        df = df.copy(); df["market"] = m
        frames.append(build_market_features(df))
    panel = pd.concat(frames, ignore_index=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel["date"] = panel["timestamp"].dt.date
    panel = add_cross_sectional(panel)
    lut = {}
    for _, r in panel[["market", "date", "f_atr_pct_14", "f_atr_xs_decile"]].iterrows():
        lut[(r["market"], r["date"])] = (r["f_atr_pct_14"], r["f_atr_xs_decile"])
    return lut


# ============================================================================
# 3. 청산 시뮬 — 시간순 walk, 각 봉의 판단은 그 봉 시점까지 정보만 (causal)
# ============================================================================
def simulate(bars, *, hard_sl=None, tp=None, trail_arm=None, trail=None,
             time_stop_h=None):
    """15m 봉 시간순 walk → (gross, outcome, hold_h, low_excursion).

    우선순위 (같은 봉 동시 → 보수=SL 먼저):
      1) hard SL : low <= entry*(1-hard_sl)
      2) trail SL: arm (high>=entry*(1+trail_arm)) 발동 후 low <= running_high*(1-trail)
      3) TP      : high >= entry*(1+tp)
      4) time-stop: 진입 후 time_stop_h 경과 봉 종가 청산
    아무것도 안 걸리면 EOD = 마지막 봉 close.

    low_excursion = 진입 대비 그날 최저 도달 (deep-loss 사건 노출 측정; gross 청산과 별개).
    트레일링 arm·running_high·경과봉 모두 현재까지 정보만 → 미래 봉 미리보기 0.
    """
    if not bars:
        return np.nan, "nodata", np.nan, np.nan
    entry = bars[0][0]
    if not np.isfinite(entry) or entry <= 0:
        return np.nan, "nodata", np.nan, np.nan
    sl_px = entry * (1 - hard_sl) if hard_sl is not None else None
    tp_px = entry * (1 + tp) if tp is not None else None
    arm_px = entry * (1 + trail_arm) if (trail is not None and trail_arm is not None) else None
    ts_bars = int(round(time_stop_h * BARS_PER_HOUR)) if time_stop_h is not None else None
    running_high = entry
    running_low = entry
    armed = (trail is not None and trail_arm is None)  # arm 없으면 즉시 활성
    for i, (o, h, l, c) in enumerate(bars):
        running_low = min(running_low, l)              # excursion (진단)
        if h > running_high:
            running_high = h
        if arm_px is not None and not armed and h >= arm_px:
            armed = True
        trail_px = running_high * (1 - trail) if (trail is not None and armed) else None
        # 1) hard SL
        if sl_px is not None and l <= sl_px:
            return -hard_sl, "sl", (i + 1) / BARS_PER_HOUR, running_low / entry - 1.0
        # 2) trailing SL
        if trail_px is not None and l <= trail_px:
            return trail_px / entry - 1.0, "trail", (i + 1) / BARS_PER_HOUR, running_low / entry - 1.0
        # 3) TP
        if tp_px is not None and h >= tp_px:
            return tp, "tp", (i + 1) / BARS_PER_HOUR, running_low / entry - 1.0
        # 4) time-stop
        if ts_bars is not None and (i + 1) >= ts_bars:
            return c / entry - 1.0, "timestop", (i + 1) / BARS_PER_HOUR, running_low / entry - 1.0
    last_c = bars[-1][3]
    return last_c / entry - 1.0, "eod", len(bars) / BARS_PER_HOUR, running_low / entry - 1.0


# ============================================================================
# 4. 청산 룰 정의 (placeholder 그리드, §2.5)
# ============================================================================
def rule_params(rule, regime, atr_pct, atr_decile):
    """룰별 simulate() kwargs. atr_pct/atr_decile = D-1 (leak-free)."""
    if rule == "baseline":
        return dict(hard_sl=CHAMP_SL, tp=CHAMP_TP)
    if rule == "trail":
        # 하드 -3% floor 유지 + +3% 도달 후 고점대비 -2% 트레일 (winner run, 이익보존)
        return dict(hard_sl=CHAMP_SL, trail_arm=0.03, trail=0.02, tp=None)
    if rule == "time":
        # 챔피언 SL/TP + 첫 1h 안에 TP 미달이면 봉종가 청산 (pump-then-dump 꼬리 컷)
        return dict(hard_sl=CHAMP_SL, tp=CHAMP_TP, time_stop_h=1.0)
    if rule == "vol":
        # SL/TP 를 D-1 ATR% 로 스케일: SL=clip(2*ATR,1.5%,5%), TP=clip(4*ATR,3%,10%)
        a = atr_pct if (atr_pct is not None and np.isfinite(atr_pct)) else 0.025
        sl = float(np.clip(2.0 * a, 0.015, 0.05))
        tp = float(np.clip(4.0 * a, 0.03, 0.10))
        return dict(hard_sl=sl, tp=tp)
    if rule == "vol_tight":
        # 하방-우선 vol 변형: SL 은 챔피언 3% 를 절대 넘기지 않음(deep-loss 노출 차단),
        # volatile 코인에선 더 타이트(SL=clip(2*ATR,1.5%,3%)). TP 만 ATR 로 넓힘.
        a = atr_pct if (atr_pct is not None and np.isfinite(atr_pct)) else 0.025
        sl = float(np.clip(2.0 * a, 0.015, CHAMP_SL))   # cap = 챔피언 3% (never wider)
        tp = float(np.clip(4.0 * a, 0.03, 0.10))
        return dict(hard_sl=sl, tp=tp)
    if rule == "regime":
        # bear_volatile: 빠르고 타이트 (SL 2% + 1h time-stop). bear_quiet: 중간.
        # bull_*: 느슨 (SL 4% / TP 8%, EOD 까지).
        if regime == "bear_volatile":
            return dict(hard_sl=0.02, tp=CHAMP_TP, time_stop_h=1.0)
        if regime == "bear_quiet":
            return dict(hard_sl=CHAMP_SL, tp=CHAMP_TP, time_stop_h=2.0)
        # bull_quiet / bull_volatile
        return dict(hard_sl=0.04, tp=0.08)
    raise ValueError(rule)


RULES = ["baseline", "trail", "time", "vol", "vol_tight", "regime"]


# ============================================================================
# 5. 지표 (net 0.15% 차감, day-equal-weight)
# ============================================================================
def metrics(trades: pd.DataFrame, label: str) -> dict:
    t = trades.dropna(subset=["net"])
    n = len(t)
    if n == 0:
        return dict(rule=label, n=0)
    net = t["net"].values
    daily = t.groupby("date")["net"].mean().sort_index()
    mu, sig = daily.mean(), daily.std()
    sharpe = float(mu / sig * np.sqrt(365)) if sig and sig > 0 else np.nan
    dn_std = float(np.sqrt(np.mean(np.minimum(daily.values, 0.0) ** 2)))
    sortino = float(mu / dn_std * np.sqrt(365)) if dn_std > 0 else np.nan
    eq = (1 + daily).cumprod(); peak = eq.cummax()
    mdd = float(((eq - peak) / peak).min())
    cum = float(eq.iloc[-1] - 1.0)
    k5 = max(1, int(np.ceil(0.05 * n)))
    cvar95 = float(np.sort(net)[:k5].mean())
    oc = t["outcome"]
    le = t["low_excursion"].values    # 진입대비 그날 최저 도달
    return dict(
        rule=label, n=int(n), days=int(t["date"].nunique()),
        net_mean=float(net.mean()),
        sharpe=sharpe, sortino=sortino, mdd=mdd, cum=cum,
        hit=float((net > 0).mean()),
        cvar95=cvar95, worst=float(net.min()),
        # deep-loss: (a) 실현 net 기준  (b) 장중저가 도달 기준 (SL floor 무관 = 진짜 노출)
        p_net_le_m5=float((net <= -0.05).mean()),
        p_net_le_m10=float((net <= -0.10).mean()),
        p_low_le_m5=float((le <= -0.05).mean()),
        p_low_le_m10=float((le <= -0.10).mean()),
        pct_sl=float((oc == "sl").mean()),
        pct_trail=float((oc == "trail").mean()),
        pct_tp=float((oc == "tp").mean()),
        pct_timestop=float((oc == "timestop").mean()),
        pct_eod=float((oc == "eod").mean()),
        avg_hold_h=float(t["hold_h"].mean()),
    )


# ============================================================================
# 6. causal-exit 자가검증 (leak 라인 확인)
# ============================================================================
def assert_causal(paths: dict, atr_lut: dict, picks: pd.DataFrame, n_check=200):
    """무작위 picks 에 대해, 각 룰을 '봉을 한 개씩 늘려가며(prefix-only)' 재생성한
    청산 시점이 전체 경로로 한 번에 돌린 결과와 동일한지 확인.
    어느 룰이든 미래 봉을 미리 봤다면(look-ahead) prefix 결과가 달라진다."""
    rng = np.random.RandomState(0)
    keys = list(paths.keys())
    idx = rng.choice(len(keys), size=min(n_check, len(keys)), replace=False)
    pick_lut = {(r["market"], pd.Timestamp(r["date"]).date()): r["regime"]
                for _, r in picks.iterrows()}
    fails = 0
    for j in idx:
        k = keys[j]
        bars = paths[k]
        if not bars:
            continue
        regime = pick_lut.get(k, "bull_quiet")
        atr_pct, atr_dec = atr_lut.get(k, (None, None))
        for rule in RULES:
            kw = rule_params(rule, regime, atr_pct, atr_dec)
            g_full, oc_full, _, _ = simulate(bars, **kw)
            # prefix-replay: 청산이 발생한 시점까지의 bars 만으로 같은 결정이 나야 함
            g_pref, oc_pref = None, None
            for L in range(1, len(bars) + 1):
                gg, occ, _, _ = simulate(bars[:L], **kw)
                if occ in ("sl", "trail", "tp", "timestop"):
                    g_pref, oc_pref = gg, occ
                    break
            if g_pref is None:   # 끝까지 안 걸림 → EOD 는 마지막 봉 필요 (정상)
                g_pref, oc_pref = simulate(bars, **kw)[:2]
            if oc_pref != oc_full or not np.isclose(g_pref, g_full, atol=1e-9, equal_nan=True):
                fails += 1
    return fails


# ============================================================================
# 7. main
# ============================================================================
def main():
    log.info("R1 진입 픽 로드: %s", R1_PICKS)
    df = pd.read_csv(R1_PICKS)
    r1 = df[df["policy"] == "R1_ratio"].copy()
    r1["date"] = pd.to_datetime(r1["date"]).dt.date
    log.info("R1 진입: %d picks, %d days, %s..%s",
             len(r1), r1["date"].nunique(), min(r1["date"]), max(r1["date"]))

    log.info("15m 경로 로드...")
    paths = load_paths(r1.rename(columns={"market": "market", "date": "date"}))
    log.info("  경로 확보 %d / %d (market,date)", len(paths),
             r1.drop_duplicates(["market", "date"]).shape[0])

    log.info("D-1 ATR lookup 빌드 (vol 룰)...")
    atr_lut = build_atr_lookup()
    log.info("  atr lut %d rows", len(atr_lut))

    # causal-exit 자가검증 (leak 라인)
    log.info("causal-exit 자가검증 (prefix-replay)...")
    fails = assert_causal(paths, atr_lut, r1, n_check=300)
    log.info("  prefix-replay mismatch = %d (0 이어야 leak-free)", fails)

    # 룰별 trade 시뮬
    all_trades = []
    for rule in RULES:
        recs = []
        for _, r in r1.iterrows():
            k = (r["market"], r["date"])
            bars = paths.get(k)
            if not bars:
                continue
            atr_pct, atr_dec = atr_lut.get(k, (None, None))
            kw = rule_params(rule, r["regime"], atr_pct, atr_dec)
            gross, oc, hold, low_exc = simulate(bars, **kw)
            if not np.isfinite(gross):
                continue
            net = gross - ROUND_TRIP_COST
            recs.append(dict(rule=rule, date=r["date"], market=r["market"],
                             regime=r["regime"], gross=gross, net=net, outcome=oc,
                             hold_h=hold, low_excursion=low_exc))
        all_trades.append(pd.DataFrame(recs))

    trades_df = pd.concat(all_trades, ignore_index=True)
    trades_df.to_csv(OUT_TRADES, index=False)
    log.info("trades dump -> %s (%d rows)", OUT_TRADES, len(trades_df))

    rows = [metrics(trades_df[trades_df["rule"] == rule], rule) for rule in RULES]
    comp = pd.DataFrame(rows)
    # baseline 대비 Δ
    base = comp[comp["rule"] == "baseline"].iloc[0]
    for col in ["net_mean", "sharpe", "sortino", "mdd", "cum", "hit",
                "p_net_le_m5", "p_low_le_m5", "p_low_le_m10", "pct_sl"]:
        comp[f"d_{col}"] = comp[col] - base[col]
    comp["leak_prefix_mismatch"] = fails
    comp.to_csv(OUT_COMPARE, index=False)
    log.info("compare -> %s", OUT_COMPARE)

    show = ["rule", "n", "net_mean", "sharpe", "sortino", "mdd", "cum", "hit",
            "p_net_le_m5", "p_low_le_m5", "p_low_le_m10", "pct_sl", "pct_tp",
            "pct_trail", "pct_timestop", "avg_hold_h"]
    pd.set_option("display.width", 200, "display.max_columns", 40)
    print(comp[show].to_string(index=False))
    print("\nΔ vs baseline:")
    print(comp[["rule", "d_net_mean", "d_sharpe", "d_sortino", "d_mdd",
                "d_cum", "d_p_low_le_m5", "d_p_low_le_m10", "d_pct_sl"]].to_string(index=False))
    print(f"\ncausal-exit prefix-replay mismatch = {fails} (must be 0)")


if __name__ == "__main__":
    main()
