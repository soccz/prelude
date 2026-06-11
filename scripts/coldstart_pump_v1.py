"""Cold-start pump v1 — 장중 초기 신호(첫 15m bars)로 cold-start 펌프를 당일 중에 포착.

배경 (이미 확립):
  - pump_rule_discovery_v1: D-1 일봉 모멘텀 룰(roc_7d_rank>0.85)은 펌프의 ~1/7만 사전 포착.
    6/7은 cold-start (D-1 모멘텀 없이 당일 갑자기 터짐).
  - bear_quiet: 펌프는 보통 오후~밤(15~20h KST)에 완성. 일찍 자르면 손해.
  - 11실험 천장: D-1 일봉 진입 net 은 음수 천장. 진입 집합 근본 변경 필요.

연구 질문:
  장중 초기 신호(첫 15m~60m 거래량/가격 미세동학)로 cold-start 펌프를 당일 중 포착 가능한가?

결정 시점(decision time): 09:30 / 10:00 / 11:00 KST (09:00 이후 bars 2/4/8개 사용).
라벨: 결정시점 가격(= 결정시점 직전 bar close) 대비 그날 잔여시간 max(high) 가 +10%/+15%/+20% 도달.
유니버스: D-1 거래대금 top 100~150 (시간정합성 — D-1 기준).

== Leak 방어 (양보 X — same-day leak 2번 전적) ==
  1) 피처는 결정시점 strictly 이전 bars(09:00 ~ 결정시점-1bar)만.
     라벨은 결정시점 이후(결정시점 bar 부터 그날 마지막 bar)의 high. 같은 bar 양쪽 사용 금지.
  2) 유니버스 rank 는 D-1 기준.
  3) net = 0.15% 왕복 차감 (+ 보수 0.5% 병기).
  4) Purged WF(날짜 fold, embargo>=5d), coin-day dedup, base rate 대비 lift.
  5) 표본 시간 집중 여부 명시.

사용:
    python scripts/coldstart_pump_v1.py --mode probe          # 데이터/timestamp 의미 확인
    python scripts/coldstart_pump_v1.py --mode run --decision 10:00 --tp 0.15
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 장시간 실행 시 stdout block-buffering 회피 (리다이렉트 시에도 진행상황 보이게)
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

D15_DB = "data/upbit_15m.db"
D1_DB = "data/upbit_d1.db"


# ============================================================================
# read-only loaders (database.py 의 init_db write side-effect 회피 + self-contained)
# ============================================================================
def _ro_conn(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True)


def load_all_15m(db_path: str) -> pd.DataFrame:
    with _ro_conn(db_path) as c:
        df = pd.read_sql_query(
            "SELECT market, timestamp, open, high, low, close, volume, quote_volume FROM candles",
            c, parse_dates=["timestamp"],
        )
    return df


def load_all_d1(db_path: str) -> pd.DataFrame:
    with _ro_conn(db_path) as c:
        df = pd.read_sql_query(
            "SELECT market, timestamp, open, high, low, close, volume, quote_volume FROM candles",
            c, parse_dates=["timestamp"],
        )
    return df


# ============================================================================
# PROBE — timestamp 의미 / coverage 확정 (leak 방어 전제)
# ============================================================================
def run_probe():
    print("=" * 70)
    print("PROBE: 15m DB timestamp 의미 + coverage")
    print("=" * 70)
    df = load_all_15m(D15_DB)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"\n15m rows: {len(df):,}  markets: {df['market'].nunique()}")
    print(f"range: {df['timestamp'].min()} -> {df['timestamp'].max()}")

    # time-of-day grid: bar 가 어떤 분에 찍히나
    tod = df["timestamp"].dt.strftime("%H:%M").value_counts().sort_index()
    print(f"\ndistinct time-of-day: {len(tod)} (15m grid 이면 96)")
    # 09:00 부근 + 자정 부근만 표시
    near = [t for t in tod.index if t <= "01:00" or ("08:30" <= t <= "11:30")]
    print("counts near 00:00 & 09:00 (bar 시작 시각 추정):")
    for t in near:
        print(f"  {t}: {tod[t]:,}")

    # 한 코인의 특정 날 bars (KRW-BTC, 최근 full day)
    btc = df[df["market"] == "KRW-BTC"].sort_values("timestamp")
    if len(btc):
        last_full = (btc["timestamp"].max().normalize() - pd.Timedelta(days=1))
        day = btc[(btc["timestamp"] >= last_full) & (btc["timestamp"] < last_full + pd.Timedelta(days=1))]
        print(f"\nKRW-BTC bars on {last_full.date()}  (n={len(day)}):")
        print("  first 6:", list(day["timestamp"].dt.strftime("%H:%M").head(6)))
        print("  last 6 :", list(day["timestamp"].dt.strftime("%H:%M").tail(6)))

    # d1 대조: 같은 코인 같은 날 d1 open/high vs 15m 첫 bar open
    d1 = load_all_d1(D1_DB)
    d1["timestamp"] = pd.to_datetime(d1["timestamp"])
    print(f"\nd1 range: {d1['timestamp'].min()} -> {d1['timestamp'].max()}  markets: {d1['market'].nunique()}")
    # d1 timestamp 의 time-of-day
    print("d1 time-of-day:", sorted(d1["timestamp"].dt.strftime("%H:%M").unique())[:5])

    # *** 핵심 정합성 테스트 ***
    # 가설 A: 15m timestamp 가 KST 이고 bar 시작이면, KST 09:00:00 bar 의 open == d1 그 날 open.
    # 가설 B: 15m timestamp 가 UTC 이면 d1 일봉(09:00 KST=00:00 UTC) 시작은 15m 00:00 bar.
    # d1 의 한 날 (timestamp=YYYY-MM-DD 09:00) open 을 잡고, 15m 에서 어떤 bar 의 open 과 같은지 찾는다.
    test_mk = "KRW-BTC"
    d1b = d1[d1["market"] == test_mk].sort_values("timestamp")
    # 최근 5 일봉
    for _, r in d1b.tail(5).iterrows():
        dts = r["timestamp"]
        d1_open, d1_high = r["open"], r["high"]
        # 그 일봉 날짜(=timestamp 의 date) 의 15m bars
        # 후보 1: 같은 date 09:00 KST 시작 (15m ts 가 KST 가정)
        win = btc[(btc["timestamp"] >= dts.normalize()) &
                  (btc["timestamp"] < dts.normalize() + pd.Timedelta(days=1))].sort_values("timestamp")
        match_open = win[np.isclose(win["open"], d1_open, rtol=1e-4)]
        # 그 날 15m high max
        day_high = win["high"].max() if len(win) else np.nan
        print(f"\n  d1 {dts}  open={d1_open:.0f} high={d1_high:.0f}")
        if len(win):
            print(f"    15m bars in [{dts.normalize().date()} 00:00, +1d): n={len(win)}, "
                  f"first ts={win['timestamp'].iloc[0].strftime('%H:%M')}, "
                  f"first open={win['open'].iloc[0]:.0f}, day_high(15m)={day_high:.0f}")
        if len(match_open):
            print(f"    >>> open match at 15m ts: {match_open['timestamp'].iloc[0]}")
        else:
            print(f"    >>> NO open match in same calendar-date window")

    # coin-day coverage: bars per coin-day 분포 (완전한 날 = 96 bars 기대 if KST-aligned full day)
    df["kst_date"] = df["timestamp"].dt.date  # placeholder; probe 후 확정
    bpd = df.groupby(["market", "kst_date"]).size()
    print(f"\nbars-per-(market,calendar-date) 분포: "
          f"median={bpd.median():.0f} p10={bpd.quantile(.1):.0f} p90={bpd.quantile(.9):.0f} max={bpd.max()}")


# ============================================================================
# trade_day 정의: KST 09:00 시작 일봉. timestamp(KST) - 9h 의 date = trade_day.
#   → 09:00 ~ 다음날 08:45 가 같은 trade_day 로 묶임 (d1 일봉 D 와 정합).
# ============================================================================
def add_trade_day(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["trade_day"] = (df["timestamp"] - pd.Timedelta(hours=9)).dt.normalize()
    # minutes since 09:00 (decision-time slicing 용): 0 == 09:00 bar
    df["min_since_open"] = ((df["timestamp"] - pd.Timedelta(hours=9)) -
                            df["trade_day"]).dt.total_seconds() / 60.0
    return df


def run_probe2():
    """trade_day(09:00 시작) 정의 검증 + pump base rate (decision-time entry 기준).
    leak-free 라벨 = decision bar(포함) 부터 trade_day 끝까지 max(high) vs decision 가격.
    """
    print("=" * 70)
    print("PROBE2: trade_day 정의 검증 + cold-start pump base rate")
    print("=" * 70)
    df = load_all_15m(D15_DB)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = add_trade_day(df)

    # 1) trade_day 당 bar 수: 온전한 날이면 96 (09:00~다음날 08:45)
    bpd = df.groupby(["market", "trade_day"]).size()
    print(f"\nbars-per-(market,trade_day): median={bpd.median():.0f} "
          f"p10={bpd.quantile(.1):.0f} p25={bpd.quantile(.25):.0f} p90={bpd.quantile(.9):.0f}")
    print(f"  trade_day 수(완전 96): {(bpd==96).mean()*100:.1f}%, >=80 bar: {(bpd>=80).mean()*100:.1f}%")

    # 2) d1 정합: trade_day 의 첫 bar(min_since_open=0, =09:00) open == d1 open?
    d1 = load_all_d1(D1_DB)
    d1["timestamp"] = pd.to_datetime(d1["timestamp"])
    d1["trade_day"] = d1["timestamp"].dt.normalize()  # d1 timestamp 가 곧 09:00 KST 시작
    first_bar = df[df["min_since_open"] == 0][["market", "trade_day", "open"]].rename(columns={"open": "open15"})
    chk = d1.merge(first_bar, on=["market", "trade_day"], how="inner")
    chk = chk[(chk["open"] > 0) & (chk["open15"] > 0)]
    chk["rel_err"] = (chk["open15"] / chk["open"] - 1.0).abs()
    print(f"\nd1.open vs 15m 09:00-bar.open  (n={len(chk):,}): "
          f"match<0.1%={ (chk['rel_err']<0.001).mean()*100:.1f}%, "
          f"median rel_err={chk['rel_err'].median()*1e4:.2f}bp, p95={chk['rel_err'].quantile(.95)*1e4:.1f}bp")

    # 3) cold-start pump base rate by decision time & TP.
    #    decision_min ∈ {30,60,120} → 09:30/10:00/11:00. price@decision = decision bar open.
    #    label = max(high over bars with min_since_open >= decision_min) / price - 1 >= TP.
    print("\ncold-start pump base rate (decision-time entry, residual-day max high):")
    print(f"{'dec':>6} {'TP':>5} {'n_coinday':>10} {'pump%':>7} {'note'}")
    g = df.sort_values(["market", "trade_day", "min_since_open"])
    for dec in (30, 60, 120):
        # decision bar = first bar at min_since_open == dec
        at_dec = g[g["min_since_open"] == dec][["market", "trade_day", "open"]].rename(columns={"open": "p_dec"})
        # residual high: bars with min_since_open >= dec
        resid = g[g["min_since_open"] >= dec].groupby(["market", "trade_day"])["high"].max().rename("resid_high").reset_index()
        m = at_dec.merge(resid, on=["market", "trade_day"], how="inner")
        m = m[m["p_dec"] > 0]
        m["mr"] = m["resid_high"] / m["p_dec"] - 1.0
        for tp in (0.10, 0.15, 0.20):
            rate = (m["mr"] >= tp).mean() * 100
            print(f"{dec:>6} {tp:>5.2f} {len(m):>10,} {rate:>7.3f}  {'09:'+str(dec//60).zfill(2) if dec>=60 else '09:'+str(dec)}")

    # 4) 시간 분산: pump 가 특정 월에 몰리나 (TP=0.15, dec=60)
    at_dec = g[g["min_since_open"] == 60][["market", "trade_day", "open"]].rename(columns={"open": "p_dec"})
    resid = g[g["min_since_open"] >= 60].groupby(["market", "trade_day"])["high"].max().rename("resid_high").reset_index()
    m = at_dec.merge(resid, on=["market", "trade_day"], how="inner")
    m = m[m["p_dec"] > 0]
    m["mr"] = m["resid_high"] / m["p_dec"] - 1.0
    m["pump15"] = (m["mr"] >= 0.15).astype(int)
    m["ym"] = m["trade_day"].dt.strftime("%Y-%m")
    by_m = m.groupby("ym")["pump15"].agg(["sum", "size"])
    by_m["rate%"] = by_m["sum"] / by_m["size"] * 100
    print(f"\npump15 @09:30(dec60) by month (마지막 8개월):")
    print(by_m.tail(8).to_string())


def run_probe3():
    """d1 유니버스 가용성 + cold-start 필터(roc_7d rank) 분포."""
    print("=" * 70)
    print("PROBE3: d1 universe + cold-start roc_7d filter")
    print("=" * 70)
    d1 = load_all_d1(D1_DB)
    d1["timestamp"] = pd.to_datetime(d1["timestamp"])
    print(f"d1 rows {len(d1):,} markets {d1.market.nunique()}")
    print(f"quote_volume null frac: {d1.quote_volume.isna().mean():.4f}")
    d1["d"] = d1.timestamp.dt.normalize()
    cpd = d1.groupby("d").market.nunique()
    print(f"coins-per-day d1: median={int(cpd.median())} min={int(cpd.min())} last={int(cpd.iloc[-1])}")
    print(f"d1 dates >= 2023-05-10: {(d1.d >= '2023-05-10').sum():,} rows")

    # roc_7d (KRW only), cross-sectional rank per day
    d1 = d1[d1.market.str.startswith("KRW-")].sort_values(["market", "timestamp"])
    d1["roc_7d"] = d1.groupby("market")["close"].pct_change(7)
    d1["roc7_rank"] = d1.groupby("d")["roc_7d"].rank(pct=True)
    # universe rank by quote_volume per day
    d1["liq_rank"] = d1.groupby("d")["quote_volume"].rank(method="dense", ascending=False, na_option="bottom")
    sub = d1[(d1.d >= "2023-06-01") & d1.liq_rank.le(120) & d1.roc7_rank.notna()]
    print(f"\ntop120 universe coin-days (d1, roc7 available): {len(sub):,}")
    print(f"  cold-start (roc7_rank < 0.70) frac in top120: {(sub.roc7_rank < 0.70).mean()*100:.1f}%")
    print(f"  roc7_rank quantiles: p25={sub.roc7_rank.quantile(.25):.2f} p50={sub.roc7_rank.quantile(.5):.2f} p75={sub.roc7_rank.quantile(.75):.2f}")


# ============================================================================
# D-1 일봉 컨텍스트 (leak-free: 모두 전일 종가까지)
# ============================================================================
def build_d1_context(d1: pd.DataFrame) -> pd.DataFrame:
    """각 (market, trade_day) 에 대해 전일(D-1) 일봉 기반 컨텍스트.
    반환 키 trade_day = D (오늘). 값은 전부 D-1 까지의 정보 → shift(1).
    """
    d1 = d1[d1.market.str.startswith("KRW-")].copy()
    d1["timestamp"] = pd.to_datetime(d1["timestamp"])
    d1 = d1.sort_values(["market", "timestamp"])
    g = d1.groupby("market", group_keys=False)
    # D-1 까지 계산되는 raw 피처들 (today row 에서 보면 자기 자신 포함 — shift 로 D-1 로 민다)
    d1["roc_7d"] = g["close"].pct_change(7)
    d1["roc_3d"] = g["close"].pct_change(3)
    d1["log_ret_1d"] = np.log(d1["close"] / g["close"].shift(1) + 1e-12)
    tr = (d1["high"] - d1["low"]) / (d1["close"] + 1e-12)
    d1["atr_pct_14"] = tr.groupby(d1["market"]).transform(lambda s: s.rolling(14, min_periods=5).mean())
    d1["ret_5d"] = g["close"].pct_change(5)
    d1["d1_close"] = d1["close"]
    d1["d1_qv"] = d1["quote_volume"]

    # *** shift(1): today(D) row 에 D-1 값 부착 (look-ahead 방어) ***
    shift_cols = ["roc_7d", "roc_3d", "log_ret_1d", "atr_pct_14", "ret_5d", "d1_close", "d1_qv"]
    for c in shift_cols:
        d1[c + "_lag"] = g[c].shift(1)
    d1["trade_day"] = d1["timestamp"].dt.normalize()  # = today D

    out = d1[["market", "trade_day"] + [c + "_lag" for c in shift_cols]].copy()
    out = out.rename(columns={c + "_lag": c for c in shift_cols})
    # cross-sectional rank per trade_day (전일 값 기준이므로 leak-free)
    out["roc7_rank"] = out.groupby("trade_day")["roc_7d"].rank(pct=True)
    out["roc3_rank"] = out.groupby("trade_day")["roc_3d"].rank(pct=True)
    out["liq_rank"] = out.groupby("trade_day")["d1_qv"].rank(method="dense", ascending=False, na_option="bottom")
    out["atr_rank"] = out.groupby("trade_day")["atr_pct_14"].rank(pct=True)
    return out


# ============================================================================
# Intraday 피처 (결정시점 strictly 이전 bars: min_since_open < dec_min)
# + decision-time 라벨/exit (min_since_open >= dec_min)
# ============================================================================
def build_intraday(df15: pd.DataFrame, dec_min: int, hist_window: int = 20) -> pd.DataFrame:
    """각 (market, trade_day) 에 대해 결정시점(=09:00 + dec_min) 기준 피처/라벨.

    피처 (모두 min_since_open < dec_min bars; 즉 결정 bar 직전까지):
      - cum_ret: (decision 직전 bar close) / (09:00 bar open) - 1   [당일 첫구간 수익률]
      - cum_qv: 09:00 ~ 직전 bar 누적 quote_volume
      - qv_surge: cum_qv / (자기 과거 hist_window trade_day 의 같은 dec_min 창 cum_qv 평균)
      - up_bar_frac: 양봉 비율, range_pct: 평균 (high-low)/open, body_pos: (close-low)/(high-low) 평균
      - max_bar_ret: 단일 bar 최대 수익률, n_pre: 사용된 bar 수
    decision_price = 결정 bar(min_since_open==dec_min) open  (진입가).
    라벨/exit (min_since_open >= dec_min):
      - resid_high, resid_low, eod_close (마지막 bar close)
    leak: 피처와 라벨은 dec_min 경계로 strictly 분리 (decision bar 는 라벨쪽; open 만 진입가로).
    """
    d = df15.sort_values(["market", "trade_day", "min_since_open"])
    pre = d[d["min_since_open"] < dec_min]
    post = d[d["min_since_open"] >= dec_min]

    # --- pre-decision 집계 ---
    def agg_pre(grp):
        op0 = grp["open"].iloc[0]            # 09:00 bar open (= day open)
        last_close = grp["close"].iloc[-1]   # decision 직전 bar close
        hi = grp["high"].values; lo = grp["low"].values
        cl = grp["close"].values; op = grp["open"].values
        n = len(grp)
        cum_qv = grp["quote_volume"].sum()
        up_frac = float(np.mean(cl > op)) if n else np.nan
        rng = np.mean((hi - lo) / (op + 1e-12)) if n else np.nan
        denom = (hi - lo + 1e-12)
        body_pos = float(np.mean((cl - lo) / denom)) if n else np.nan
        bar_ret = cl / (op + 1e-12) - 1.0
        return pd.Series({
            "day_open": op0,
            "pre_close": last_close,
            "cum_ret": last_close / (op0 + 1e-12) - 1.0,
            "cum_qv": cum_qv,
            "up_bar_frac": up_frac,
            "range_pct": rng,
            "body_pos": body_pos,
            "max_bar_ret": float(np.nanmax(bar_ret)) if n else np.nan,
            "pre_high": float(np.nanmax(hi)) if n else np.nan,
            "n_pre": n,
        })
    pre_agg = pre.groupby(["market", "trade_day"]).apply(agg_pre).reset_index()

    # --- decision bar open (진입가) ---
    dec_bar = d[d["min_since_open"] == dec_min][["market", "trade_day", "open"]].rename(columns={"open": "p_dec"})

    # --- post-decision (라벨/exit) ---
    post_agg = post.groupby(["market", "trade_day"]).agg(
        resid_high=("high", "max"),
        resid_low=("low", "min"),
        n_post=("close", "size"),
    ).reset_index()
    # EOD close = trade_day 의 마지막 bar close (post 안)
    eod = post.sort_values("min_since_open").groupby(["market", "trade_day"])["close"].last().rename("eod_close").reset_index()

    m = pre_agg.merge(dec_bar, on=["market", "trade_day"], how="inner")
    m = m.merge(post_agg, on=["market", "trade_day"], how="inner")
    m = m.merge(eod, on=["market", "trade_day"], how="inner")
    m = m[(m["p_dec"] > 0) & (m["day_open"] > 0)]

    # qv_surge: 자기 과거 hist_window trade_day 의 같은-창 cum_qv 평균 대비 (leak-free, 과거만)
    m = m.sort_values(["market", "trade_day"])
    m["hist_cum_qv"] = (m.groupby("market")["cum_qv"]
                        .transform(lambda s: s.shift(1).rolling(hist_window, min_periods=5).mean()))
    m["qv_surge"] = m["cum_qv"] / (m["hist_cum_qv"] + 1e-9)
    # decision_price 대비 위치들
    m["pos_vs_dayopen"] = m["p_dec"] / (m["day_open"] + 1e-12) - 1.0
    # pre_high breakout: p_dec 가 pre 구간 고가를 넘었나 (돌파 proxy)
    m["pdec_vs_prehigh"] = m["p_dec"] / (m["pre_high"] + 1e-12) - 1.0
    return m


# ============================================================================
# Purged Walk-Forward (날짜 fold, embargo)
# ============================================================================
def wf_folds(days: np.ndarray, n_folds: int, embargo: int, holdout: int):
    """days = 정렬된 unique trade_day(datetime64). val fold 양쪽 embargo 만큼 train 제외."""
    days = np.sort(np.unique(days))
    if holdout > 0:
        days = days[:-holdout] if holdout < len(days) else days
    n = len(days)
    fold_sz = n // (n_folds + 1)
    folds = []
    for k in range(1, n_folds + 1):
        va_lo = k * fold_sz
        va_hi = (k + 1) * fold_sz if k < n_folds else n
        va_days = days[va_lo:va_hi]
        emb = pd.Timedelta(days=embargo)
        va_start, va_end = va_days[0], va_days[-1]
        tr_mask = (days < va_start - emb) | (days > va_end + emb)
        tr_days = days[tr_mask]
        folds.append((tr_days, va_days))
    return folds


from sklearn.tree import DecisionTreeClassifier

# intraday 피처 (decision-time, leak-free pre-decision)
INTRADAY_FEATS = [
    "cum_ret", "qv_surge", "up_bar_frac", "range_pct", "body_pos",
    "max_bar_ret", "pos_vs_dayopen", "pdec_vs_prehigh", "n_pre",
]
# D-1 일봉 컨텍스트 (전일까지, leak-free)
D1_FEATS = ["roc7_rank", "roc3_rank", "atr_rank", "atr_pct_14", "log_ret_1d", "ret_5d"]


def extract_leaf_rules(tree, names):
    rules, t = {}, tree.tree_
    def rec(node, conds):
        if t.children_left[node] == -1:
            rules[node] = list(conds); return
        f = names[t.feature[node]]; thr = float(t.threshold[node])
        rec(t.children_left[node], conds + [(f, "<=", thr)])
        rec(t.children_right[node], conds + [(f, ">", thr)])
    rec(0, [])
    return rules


def fmt_rule(conds):
    return " AND ".join(f"{f} {op} {thr:+.3f}" for f, op, thr in conds)


def net_pnl(df_sel: pd.DataFrame, tp: float, cost: float) -> dict:
    """decision_price 진입, residual-day exit. TP touch → +tp, else EOD close.
    보수: TP 도달해도 그 가격에 못 팔 수 있으나 여기선 TP 가격 체결 가정(낙관).
    하방: SL 없음(noSL) — 잔여시간 끝까지 보유, EOD close 로 청산.
    return per-trade net return list aggregates.
    """
    p = df_sel["p_dec"].values
    rh = df_sel["resid_high"].values
    ec = df_sel["eod_close"].values
    tp_hit = (rh / p - 1.0) >= tp
    gross = np.where(tp_hit, tp, ec / p - 1.0)
    net = gross - cost  # 왕복 거래비용
    return {
        "n": len(net), "tp_hit_rate": float(tp_hit.mean()) if len(net) else np.nan,
        "mean_net": float(np.mean(net)) if len(net) else np.nan,
        "median_net": float(np.median(net)) if len(net) else np.nan,
        "sum_net": float(np.sum(net)) if len(net) else np.nan,
        "win_rate": float(np.mean(net > 0)) if len(net) else np.nan,
        "p10_net": float(np.percentile(net, 10)) if len(net) else np.nan,
    }


def run_main(args):
    dec_map = {"09:30": 30, "10:00": 60, "11:00": 120}
    dec_min = dec_map[args.decision]
    print("=" * 70)
    print(f"COLD-START PUMP v1 — decision={args.decision} (dec_min={dec_min}), "
          f"TP={args.tp}, universe top{args.top_universe}, cold-start roc7<{args.cs_cut}")
    print("=" * 70)

    # ---- load ----
    print("loading 15m + d1 ...")
    df15 = load_all_15m(D15_DB)
    df15["timestamp"] = pd.to_datetime(df15["timestamp"])
    df15 = df15[df15.market.str.startswith("KRW-")]
    df15 = add_trade_day(df15)
    d1 = load_all_d1(D1_DB)
    d1ctx = build_d1_context(d1)
    print(f"  15m KRW rows {len(df15):,}, d1 ctx rows {len(d1ctx):,}")

    # ---- intraday panel ----
    print("building intraday features (this is the slow part) ...")
    intra = build_intraday(df15, dec_min, hist_window=args.hist_window)
    print(f"  intraday coin-days: {len(intra):,}")

    # ---- merge d1 context, filter universe + cold-start ----
    panel = intra.merge(d1ctx, on=["market", "trade_day"], how="inner")
    panel = panel[panel["liq_rank"] <= args.top_universe]
    panel = panel.dropna(subset=["roc7_rank"])
    # cold-start: 전일 모멘텀 약한 것만 (모멘텀 룰 사각지대)
    cs = panel[panel["roc7_rank"] < args.cs_cut].copy()
    # minimum pre bars (decision time 이전 bar 가 충분해야 — 데이터 결손 방어)
    min_pre = max(1, dec_min // 15 - 1)  # 09:30→1, 10:00→3, 11:00→7
    cs = cs[cs["n_pre"] >= min_pre]
    # label
    cs["mr"] = cs["resid_high"] / cs["p_dec"] - 1.0
    cs["y"] = (cs["mr"] >= args.tp).astype(int)
    cs = cs.dropna(subset=INTRADAY_FEATS + D1_FEATS, how="all")
    base = float(cs["y"].mean())
    print(f"  cold-start panel: {len(cs):,} coin-days, base pump rate(TP{args.tp}) = {base*100:.3f}%")
    print(f"  time span: {cs['trade_day'].min().date()} -> {cs['trade_day'].max().date()}, "
          f"pos count = {int(cs['y'].sum())}")

    feats = INTRADAY_FEATS + D1_FEATS
    # ---- WF leaf mining ----
    folds = wf_folds(cs["trade_day"].values, args.n_folds, args.embargo, args.holdout)
    cost = args.cost
    leaf_rows, oos_rows = [], []
    n_trees = 0
    for fi, (tr_days, va_days) in enumerate(folds, 1):
        tr = cs[cs["trade_day"].isin(tr_days)]
        va = cs[cs["trade_day"].isin(va_days)].copy()
        if len(tr) < 400 or len(va) < 150 or tr["y"].sum() < 20:
            continue
        Xtr = tr[feats].astype(float).values
        Xva = va[feats].astype(float).values
        ytr = tr["y"].values; yva = va["y"].values
        med = np.nanmedian(Xtr, axis=0); med = np.where(np.isnan(med), 0.0, med)
        Xtr = np.where(np.isnan(Xtr), med, Xtr)
        Xva = np.where(np.isnan(Xva), med, Xva)
        base_va = float(yva.mean()); npos_va = int(yva.sum())
        tree = DecisionTreeClassifier(max_depth=args.max_depth, min_samples_leaf=args.min_leaf,
                                      class_weight="balanced", random_state=42)
        tree.fit(Xtr, ytr); n_trees += 1
        rules = extract_leaf_rules(tree, feats)
        lt, lv = tree.apply(Xtr), tree.apply(Xva)
        for lid, conds in rules.items():
            mtr, mva = lt == lid, lv == lid
            if mtr.sum() < args.min_leaf or mva.sum() < 30:
                continue
            pos_tr, pos_va = float(ytr[mtr].mean()), float(yva[mva].mean())
            lift = pos_va / base_va if base_va > 0 else np.nan
            recall = float((mva & (yva == 1)).sum()) / npos_va if npos_va else np.nan
            sub_va = va[mva]
            net = net_pnl(sub_va, args.tp, cost)
            net_cons = net_pnl(sub_va, args.tp, 0.005)  # 보수 0.5%
            leaf_rows.append({
                "decision": args.decision, "tp": args.tp, "fold": fi, "leaf": int(lid),
                "rule": fmt_rule(conds), "root": fmt_rule(conds).split()[0],
                "sup_tr": int(mtr.sum()), "sup_va": int(mva.sum()),
                "pos_tr_pct": pos_tr * 100, "pos_va_pct": pos_va * 100,
                "base_va_pct": base_va * 100, "lift_va": lift,
                "gap_pp": (pos_tr - pos_va) * 100, "recall_va_pct": recall * 100,
                "fire_pct": mva.sum() / len(va) * 100,
                "net_mean_pct": net["mean_net"] * 100, "net_median_pct": net["median_net"] * 100,
                "net_win_pct": net["win_rate"] * 100, "net_p10_pct": net["p10_net"] * 100,
                "net_cons_mean_pct": net_cons["mean_net"] * 100,
                "tp_hit_pct": net["tp_hit_rate"] * 100,
            })
        # whole-fold OOS baseline (rule-free, all cold-start picks) net
        netall = net_pnl(va, args.tp, cost)
        oos_rows.append({"decision": args.decision, "tp": args.tp, "fold": fi,
                         "n_va": len(va), "base_va_pct": base_va * 100,
                         "allpick_net_mean_pct": netall["mean_net"] * 100,
                         "allpick_tp_hit_pct": netall["tp_hit_rate"] * 100})

    ldf = pd.DataFrame(leaf_rows)
    odf = pd.DataFrame(oos_rows)
    Path("output").mkdir(exist_ok=True)
    tag = f"{args.decision.replace(':','')}_tp{int(args.tp*100)}"
    leaf_path = f"output/coldstart_pump_v1_leaves_{tag}.csv"
    oos_path = f"output/coldstart_pump_v1_oos_{tag}.csv"
    ldf.to_csv(leaf_path, index=False)
    odf.to_csv(oos_path, index=False)
    print(f"\ntrees={n_trees}, leaf rows={len(ldf)} -> {leaf_path}")
    print(f"whole-fold OOS -> {oos_path}")

    if len(ldf) == 0:
        print("(no leaves passed support filter)")
        return

    # ---- cross-fold robust report ----
    print("\n=== whole-fold OOS (rule-free cold-start universe, net) ===")
    print(odf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== robust leaves: lift_va>=2, gap<12pp, recall>=8% (cross-fold) ===")
    rob = ldf[(ldf.lift_va >= 2.0) & (ldf.gap_pp < 12) & (ldf.recall_va_pct >= 8)]
    show = ["fold", "sup_va", "pos_va_pct", "lift_va", "gap_pp", "recall_va_pct",
            "fire_pct", "tp_hit_pct", "net_mean_pct", "net_cons_mean_pct", "net_win_pct", "rule"]
    if len(rob):
        with pd.option_context("display.max_colwidth", 95, "display.width", 240):
            print(rob.sort_values("lift_va", ascending=False)[show].head(20).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    else:
        print("(none)")

    print("\n=== cross-fold robust roots (lift_va>=1.8 in >=3 folds) ===")
    r2 = ldf[(ldf.lift_va >= 1.8) & (ldf.sup_va >= 30)]
    agg = r2.groupby("root").agg(n_folds=("fold", "nunique"), mean_lift=("lift_va", "mean"),
                                 mean_pos=("pos_va_pct", "mean"), mean_recall=("recall_va_pct", "mean"),
                                 mean_net=("net_mean_pct", "mean"), mean_net_cons=("net_cons_mean_pct", "mean"),
                                 mean_gap=("gap_pp", "mean"), sum_sup=("sup_va", "sum")).reset_index()
    agg = agg[agg.n_folds >= 3].sort_values("mean_lift", ascending=False)
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.2f}") if len(agg) else "(no root in >=3 folds)")

    # ---- best single rule net summary ----
    print("\n=== top-5 leaves by net_mean (any lift, sup_va>=30) ===")
    topnet = ldf[ldf.sup_va >= 30].sort_values("net_mean_pct", ascending=False).head(5)
    with pd.option_context("display.max_colwidth", 95, "display.width", 240):
        print(topnet[show].to_string(index=False, float_format=lambda x: f"{x:.2f}"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="probe", choices=["probe", "probe2", "probe3", "run"])
    p.add_argument("--decision", default="10:00", choices=["09:30", "10:00", "11:00"])
    p.add_argument("--tp", type=float, default=0.15)
    p.add_argument("--top-universe", type=int, default=120)
    p.add_argument("--cs-cut", type=float, default=0.70, help="cold-start: roc7_rank < cs_cut")
    p.add_argument("--hist-window", type=int, default=20, help="qv_surge 분모 과거 trade_day 수")
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--min-leaf", type=int, default=200)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--embargo", type=int, default=7)
    p.add_argument("--holdout", type=int, default=120)
    p.add_argument("--cost", type=float, default=0.0015, help="왕복 거래비용 (기본 0.15%)")
    args = p.parse_args()
    if args.mode == "probe":
        run_probe()
    elif args.mode == "probe2":
        run_probe2()
    elif args.mode == "probe3":
        run_probe3()
    else:
        run_main(args)


if __name__ == "__main__":
    main()
