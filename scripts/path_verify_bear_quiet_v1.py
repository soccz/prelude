"""bear_quiet 펌프 후보 청산 lever 의 15m 경로 검증 (what-hit-first 해소).

배경:
  일봉 백테스트가 청산 가정에 따라 sl_first(비관, Sharpe 음수) ~ tp_first(낙관, 양수)
  로 갈렸다. 일봉 OHLC 는 intrabar 경로(SL/TP 누가 먼저 닿았나)를 모른다.
  이 스크립트는 동일한 leak-free 진입 선택을 재현한 뒤, day-D 의 15m 경로를
  시간순으로 걸어 실제 선후를 판정한다.

진입 선택 (leak-free, 기존 regime_split_precursor_v1 과 동일 규율):
  - 모든 precursor feature 는 market 별 .shift(1) → day-D row 는 D-1 까지만 본다.
  - BTC regime 도 1일 shift → day-D 는 D-1 regime(bear_quiet) 을 본다.
  - bear_quiet = (btc_ma_distance<0) & (btc_intensity_d<=0.5), D-1 기준.
  - cross-section top-decile: purged walk-forward(5 fold, embargo 5d),
    train fold 의 quantile(0.9) cutoff 로 test fold 선택. (selection = train-only)
  - 진입 = day-D open(09:00 KST).

경로 시뮬 (in-trade, leak 아님 — 단 미래 봉 미리보기 금지, 시간순으로만):
  - day-D 경로 = 15m 봉 [D 09:00, D+1 09:00) (~96봉). 진입가 = 첫 봉(09:00) open.
  - 봉을 시간순으로 walk. 각 봉에서 low<=SL_level 이면 SL 터치, high>=TP_level 이면 TP 터치.
  - 한 봉 안에서 둘 다 터치(15m 보다 미세한 순서 불명) = ambiguous-bar:
    보수적으로 SL-first 가정하되 그 건수를 별도 보고(진실은 알 수 없음을 정직히).
  - 거래비용 왕복 0.15% 차감. spot-only.

⚠️ 이건 검증 백테스트. 라벨/모델/유니버스/사이징 변경 X.
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
from data.market_universe import signal_eligible_markets
from signals.features import compute_btc_features

D1_DB = "data/upbit_d1.db"
M15_DB = "data/upbit_15m.db"
OUT_DIR = Path("output")
EPS = 1e-12
COST = 0.0015  # 왕복 거래비용

TP = 0.10   # +10%
SL = 0.05   # -5%

PRECURSORS = [
    ("f_qv_surge_7d", "high"),        # 1순위
    ("f_qv_surge_30d", "high"),
    ("f_bounce_off_7d_low", "high"),
    ("f_atr_xs_decile", "high"),
]

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("path_verify")


# ---------------------------------------------------------------------------
# 1. leak-free panel (D-1 features) — univariate_precursor 빌더 재사용
# ---------------------------------------------------------------------------
def build_market_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.sort_values("timestamp").reset_index(drop=True).copy()
    o, h, l, c = g["open"], g["high"], g["low"], g["close"]
    v = g["volume"]
    qv = g["quote_volume"].fillna(v * c)
    raw = pd.DataFrame(index=g.index)
    log_ret = np.log(c / c.shift(1) + EPS)
    for N in (1, 3, 7, 14):
        raw[f"ret_{N}d"] = c / c.shift(N) - 1.0
    qv_ma7 = qv.rolling(7).mean()
    qv_ma30 = qv.rolling(30).mean()
    raw["qv_surge_7d"] = qv / (qv_ma7 + EPS)
    raw["qv_surge_30d"] = qv / (qv_ma30 + EPS)
    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()],
                   axis=1).max(axis=1)
    raw["atr_pct_14"] = tr.rolling(14, min_periods=7).mean() / (c + EPS)
    raw["bounce_off_7d_low"] = c / (l.rolling(7).min() + EPS) - 1.0
    # ★ shift(1) → D-1 까지만
    shifted = raw.shift(1)
    shifted.columns = [f"f_{x}" for x in shifted.columns]
    out = pd.concat([g[["market", "timestamp"]], shifted], axis=1)
    out["f_universe_qv"] = qv.shift(1)
    # day-D OHLC (청산 시점, 타겟 — feature 아님)
    out["open_D"] = o
    out["high_D"] = h
    out["low_D"] = l
    out["close_D"] = c
    return out


def build_panel(limit_markets):
    markets = signal_eligible_markets(list_markets(D1_DB))
    if limit_markets:
        markets = markets[:limit_markets]
    log.info("loading %d markets (d1)", len(markets))
    frames = []
    for i, m in enumerate(markets):
        df = load_candles(D1_DB, m)
        if df is None or len(df) < 70:
            continue
        df = df.copy()
        df["market"] = m
        frames.append(build_market_features(df))
    panel = pd.concat(frames, ignore_index=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel["date"] = panel["timestamp"].dt.normalize()
    panel = panel.sort_values(["date", "market"]).reset_index(drop=True)
    return panel


def add_cross_sectional(panel):
    p = panel.copy()
    p["f_atr_xs_decile"] = p.groupby("date")["f_atr_pct_14"].rank(pct=True)
    return p


def attach_btc_regime(panel):
    btc = load_candles(D1_DB, "KRW-BTC")
    bf = compute_btc_features(btc)[["timestamp", "btc_regime",
                                    "btc_ma_distance", "btc_intensity_d"]].copy()
    bf["timestamp"] = pd.to_datetime(bf["timestamp"])
    bf = bf.sort_values("timestamp").reset_index(drop=True)
    for col in ["btc_regime", "btc_ma_distance", "btc_intensity_d"]:
        bf[f"{col}_dm1"] = bf[col].shift(1)
    bf = bf[["timestamp", "btc_regime_dm1"]]
    p = panel.merge(bf, on="timestamp", how="left")
    p = p.rename(columns={"btc_regime_dm1": "regime"})
    return p


# ---------------------------------------------------------------------------
# 2. leak-free OOS 진입 선택 (purged WF, train-only cutoff)
# ---------------------------------------------------------------------------
def select_entries(panel, feat, direction, n_folds=5, embargo_days=5):
    cols = [feat, "date", "regime", "market", "open_D", "high_D", "low_D", "close_D"]
    d = panel[cols].dropna(subset=[feat, "regime", "open_D", "high_D",
                                   "low_D", "close_D"])
    d = d[d["regime"] == "bear_quiet"]
    all_dates = np.sort(panel["date"].unique())
    fold_size = len(all_dates) // (n_folds + 1)
    sel = []
    for k in range(1, n_folds + 1):
        train_end = fold_size * k
        test_start = train_end + embargo_days
        test_end = min(fold_size * (k + 1) + train_end, len(all_dates))
        if test_start >= len(all_dates):
            break
        train_dates = set(all_dates[:train_end])
        test_dates = set(all_dates[test_start:test_end])
        tr = d[d["date"].isin(train_dates)]
        te = d[d["date"].isin(test_dates)]
        if len(tr) < 50 or len(te) < 20:
            continue
        if direction == "high":
            cut = tr[feat].quantile(0.9)
            picks = te[te[feat] >= cut].copy()
        else:
            cut = tr[feat].quantile(0.1)
            picks = te[te[feat] <= cut].copy()
        picks["fold"] = k
        sel.append(picks)
    if not sel:
        return pd.DataFrame()
    return pd.concat(sel, ignore_index=True)


# ---------------------------------------------------------------------------
# 3. 15m 경로 walk — what-hit-first + 다중 청산 정책
# ---------------------------------------------------------------------------
def simulate_path(bars: pd.DataFrame):
    """bars: day-D 15m 봉(시간순). 반환 = 경로 판정 dict.

    bars columns: timestamp, open, high, low, close. 진입가 = 첫 봉 open.
    시간순 walk. SL_level/TP_level 터치 시각·선후 기록. 한 봉 내 둘 다 터치는
    ambiguous-bar (SL-first 보수 가정, 별도 카운트).
    """
    if len(bars) == 0:
        return None
    entry = bars.iloc[0]["open"]
    if entry <= 0 or pd.isna(entry):
        return None
    tp_lvl = entry * (1 + TP)
    sl_lvl = entry * (1 - SL)
    n = len(bars)
    high = bars["high"].values
    low = bars["low"].values
    close = bars["close"].values

    tp_bar = None
    sl_bar = None
    ambiguous_bar = None  # 같은 봉 안에서 둘 다 터치한 첫 봉
    for i in range(n):
        hit_tp = high[i] >= tp_lvl
        hit_sl = low[i] <= sl_lvl
        if hit_tp and hit_sl and ambiguous_bar is None and tp_bar is None and sl_bar is None:
            ambiguous_bar = i
        if hit_tp and tp_bar is None:
            tp_bar = i
        if hit_sl and sl_bar is None:
            sl_bar = i
        if tp_bar is not None or sl_bar is not None:
            # 첫 터치(둘 중 빠른 봉)에서 무엇이 먼저인지 확정되면 break
            if tp_bar is not None and sl_bar is not None:
                break
            # 한쪽만 터치된 봉이면 그 봉이 first
            break

    # what-hit-first 판정
    intraday_max = high.max() / entry - 1.0
    intraday_min = low.min() / entry - 1.0
    eod_ret = close[-1] / entry - 1.0
    touched_tp_anytime = (high >= tp_lvl).any()
    touched_sl_anytime = (low <= sl_lvl).any()

    if tp_bar is not None and sl_bar is not None:
        if tp_bar < sl_bar:
            first = "tp"
        elif sl_bar < tp_bar:
            first = "sl"
        else:
            first = "ambiguous"  # 같은 봉
    elif tp_bar is not None:
        first = "tp"
    elif sl_bar is not None:
        first = "sl"
    else:
        first = "none"

    return {
        "entry": entry, "n_bars": n,
        "tp_bar": tp_bar, "sl_bar": sl_bar, "ambiguous_bar": ambiguous_bar,
        "first": first,
        "touched_tp": bool(touched_tp_anytime),
        "touched_sl": bool(touched_sl_anytime),
        "intraday_max": intraday_max, "intraday_min": intraday_min,
        "eod_ret": eod_ret,
        "high": high, "low": low, "close": close,
        "tp_lvl": tp_lvl, "sl_lvl": sl_lvl,
    }


def exit_under_policy(p, policy):
    """경로 판정 p + 정책 → net return(왕복비용 차감). 봉수 4시간=16봉."""
    if p is None:
        return np.nan
    entry = p["entry"]
    high, low, close = p["high"], p["low"], p["close"]
    tp_lvl, sl_lvl = p["tp_lvl"], p["sl_lvl"]
    n = len(close)
    H4 = 16  # 4h = 16 * 15m

    if policy == "hard_sl_path":
        # 경로상 먼저 닿은 것으로 청산. ambiguous-bar = SL-first(보수).
        if p["first"] == "tp":
            g = TP
        elif p["first"] in ("sl", "ambiguous"):
            g = -SL
        else:
            g = p["eod_ret"]
        return g - COST

    if policy == "tp_eod_noSL":
        # SL 없음. TP 닿으면 +TP, 아니면 EOD close.
        if p["touched_tp"]:
            return TP - COST
        return p["eod_ret"] - COST

    if policy == "timestop_4h":
        # 4h 내 TP/SL 경로청산, 미발생 시 16번째 봉 close 로 강제청산.
        for i in range(min(H4, n)):
            hit_tp = high[i] >= tp_lvl
            hit_sl = low[i] <= sl_lvl
            if hit_tp and hit_sl:
                return -SL - COST   # ambiguous-bar 보수
            if hit_tp:
                return TP - COST
            if hit_sl:
                return -SL - COST
        idx = min(H4, n) - 1
        return close[idx] / entry - 1.0 - COST

    if policy == "eod_tpcap":
        # SL 없음, TP cap. 닿으면 +TP, 아니면 EOD (downside 무제한).
        if p["touched_tp"]:
            return TP - COST
        return p["eod_ret"] - COST

    if policy == "trailing_3pct":
        # +3% 트레일링 스탑(고점 대비). 진입 후 peak 추적, peak*(1-trail) 이탈 시 청산.
        # +TP 도달 시 +TP. 시간순 walk.
        trail = 0.03
        peak = entry
        for i in range(n):
            # 봉 내 high 로 peak 갱신 → 그 다음 low 로 트레일 체크(보수: 같은 봉 내 high 먼저)
            if high[i] >= tp_lvl:
                return TP - COST
            peak = max(peak, high[i])
            stop = peak * (1 - trail)
            if low[i] <= stop and peak > entry * (1 + trail):
                # 트레일 발동(이익 구간에서만). 청산가 = stop.
                return stop / entry - 1.0 - COST
            # 하드 손절도 같이(트레일이 손실 구간 보호 못하므로 -SL 바닥)
            if low[i] <= sl_lvl:
                return -SL - COST
        return close[-1] / entry - 1.0 - COST

    raise ValueError(policy)


def load_day_path(market, date_D):
    """day-D 경로 = 15m 봉 [D 09:00, D+1 09:00). 시간순."""
    start = pd.Timestamp(date_D).normalize() + pd.Timedelta(hours=9)
    end = start + pd.Timedelta(days=1)
    bars = load_candles(M15_DB, market, since=start, until=end)
    if bars is None or len(bars) == 0:
        return None
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    # 첫 봉이 09:00 인지 확인(아니면 진입 정합 깨짐 → drop)
    if bars.iloc[0]["timestamp"] != start:
        return None
    return bars


def net_metrics(returns):
    r = np.asarray([x for x in returns if not pd.isna(x)], dtype=float)
    if len(r) == 0:
        return dict(n=0, mean=np.nan, sharpe=np.nan, hit=np.nan,
                    cum=np.nan, mdd=np.nan)
    mean = r.mean()
    sigma = r.std()
    sharpe = mean / sigma * np.sqrt(365) if sigma > 0 else 0.0
    hit = (r > 0).mean()
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(eq)
    mdd = ((eq - peak) / peak).min()
    cum = eq[-1] - 1.0
    return dict(n=len(r), mean=mean, sharpe=sharpe, hit=hit, cum=cum, mdd=mdd)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-markets", type=int, default=None)
    args = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)

    # 15m 커버리지 윈도우
    m15_min = load_candles(M15_DB, "KRW-BTC").iloc[0]["timestamp"]
    import sqlite3
    con = sqlite3.connect(M15_DB)
    g_min, g_max = con.execute("SELECT MIN(timestamp), MAX(timestamp) FROM candles").fetchone()
    con.close()
    log.info("15m coverage: %s .. %s", g_min, g_max)
    cov_start = pd.to_datetime(g_min).normalize()
    cov_end = pd.to_datetime(g_max).normalize()

    panel = build_panel(args.limit_markets)
    panel = add_cross_sectional(panel)
    panel = attach_btc_regime(panel)

    bq = panel[panel["regime"] == "bear_quiet"]
    bq_dates_all = pd.to_datetime(bq["date"].unique())
    bq_in_cov = bq_dates_all[(bq_dates_all >= cov_start) & (bq_dates_all <= cov_end)]
    log.info("bear_quiet days: full=%d, in 15m coverage=%d",
             len(bq_dates_all), len(bq_in_cov))

    # 진입 선택은 full history 로(WF fold 경계 정합) 한 뒤, 15m 커버리지 내 trade만 경로검증
    rng = np.random.default_rng(42)
    all_rows = []
    summaries = {}
    for feat, direction in PRECURSORS:
        sel = select_entries(panel, feat, direction)
        if len(sel) == 0:
            log.warning("no OOS selection for %s", feat)
            continue
        sel["date"] = pd.to_datetime(sel["date"])
        sel_cov = sel[(sel["date"] >= cov_start) & (sel["date"] <= cov_end)].copy()
        log.info("[%s] OOS selected total=%d, in 15m coverage=%d",
                 feat, len(sel), len(sel_cov))

        recs = []
        for _, r in sel_cov.iterrows():
            bars = load_day_path(r["market"], r["date"])
            if bars is None or len(bars) < 8:
                continue
            p = simulate_path(bars)
            if p is None:
                continue
            rec = dict(feature=feat, market=r["market"],
                       date=r["date"].strftime("%Y-%m-%d"),
                       first=p["first"], touched_tp=p["touched_tp"],
                       touched_sl=p["touched_sl"],
                       both_touched=bool(p["touched_tp"] and p["touched_sl"]),
                       tp_bar=p["tp_bar"], sl_bar=p["sl_bar"],
                       ambiguous_bar=p["ambiguous_bar"],
                       intraday_max=p["intraday_max"],
                       intraday_min=p["intraday_min"],
                       eod_ret=p["eod_ret"], n_bars=p["n_bars"])
            for pol in ["hard_sl_path", "tp_eod_noSL", "timestop_4h",
                        "eod_tpcap", "trailing_3pct"]:
                rec[f"net_{pol}"] = exit_under_policy(p, pol)
            recs.append(rec)
        if not recs:
            log.warning("no path trades for %s", feat)
            continue
        df = pd.DataFrame(recs)
        all_rows.append(df)
        summaries[feat] = df

    if not all_rows:
        log.error("no path trades at all — aborting")
        return
    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv(OUT_DIR / "path_verify_bear_quiet_trades.csv", index=False)
    log.info("wrote %d path trades", len(full))

    # ---- baseline: 동일 bear_quiet 날 무작위 코인 (qv_surge_7d 후보와 같은 날짜수 매칭) ----
    base_recs = []
    bq_cov_panel = panel[(panel["regime"] == "bear_quiet")].copy()
    bq_cov_panel["date"] = pd.to_datetime(bq_cov_panel["date"])
    bq_cov_panel = bq_cov_panel[(bq_cov_panel["date"] >= cov_start) &
                                (bq_cov_panel["date"] <= cov_end)]
    # 후보 진입이 발생한 (날짜) 별로 같은 수의 무작위 코인 진입
    cand7 = summaries.get("f_qv_surge_7d")
    if cand7 is not None:
        per_date = cand7.groupby("date").size().to_dict()
        for dt, k in per_date.items():
            pool = bq_cov_panel[bq_cov_panel["date"] == pd.Timestamp(dt)]
            pool = pool.dropna(subset=["open_D"])
            if len(pool) == 0:
                continue
            take = min(k, len(pool))
            picks = pool.sample(take, random_state=int(rng.integers(0, 1e9)))
            for _, r in picks.iterrows():
                bars = load_day_path(r["market"], r["date"])
                if bars is None or len(bars) < 8:
                    continue
                p = simulate_path(bars)
                if p is None:
                    continue
                rec = dict(market=r["market"], date=dt, first=p["first"],
                           both_touched=bool(p["touched_tp"] and p["touched_sl"]),
                           intraday_max=p["intraday_max"],
                           intraday_min=p["intraday_min"], eod_ret=p["eod_ret"])
                for pol in ["hard_sl_path", "tp_eod_noSL", "timestop_4h"]:
                    rec[f"net_{pol}"] = exit_under_policy(p, pol)
                base_recs.append(rec)
    base = pd.DataFrame(base_recs)
    if len(base):
        base.to_csv(OUT_DIR / "path_verify_bear_quiet_baseline.csv", index=False)

    # ---- 리포트 ----
    print("\n" + "=" * 78)
    print("WHAT-HIT-FIRST 분해 (per feature)")
    print("=" * 78)
    for feat in summaries:
        df = summaries[feat]
        n = len(df)
        both = df["both_touched"].sum()
        amb_pct = both / n * 100
        sub = df[df["both_touched"]]
        if len(sub):
            sl_first = (sub["first"] == "sl").sum()
            tp_first = (sub["first"] == "tp").sum()
            amb = (sub["first"] == "ambiguous").sum()
            print(f"\n[{feat}] n={n}  both_touched(ambiguous)={both} ({amb_pct:.1f}%)")
            print(f"   그중: TP_first={tp_first} ({tp_first/len(sub)*100:.1f}%)  "
                  f"SL_first={sl_first} ({sl_first/len(sub)*100:.1f}%)  "
                  f"same-bar-ambiguous={amb} ({amb/len(sub)*100:.1f}%)")
            tpf = sub[sub["first"] == "tp"]
            slf = sub[sub["first"] == "sl"]
            if len(tpf):
                print(f"   TP-first 고점도달 봉(중앙)={tpf['tp_bar'].median():.0f}봉 "
                      f"(~{tpf['tp_bar'].median()*15/60:.1f}h)")
            if len(slf):
                print(f"   SL-first -5%도달 봉(중앙)={slf['sl_bar'].median():.0f}봉 "
                      f"(~{slf['sl_bar'].median()*15/60:.1f}h)")
        else:
            print(f"\n[{feat}] n={n}  both_touched=0")
        # 전체 first 분포
        fc = df["first"].value_counts().to_dict()
        print(f"   전체 first 분포: {fc}")

    print("\n" + "=" * 78)
    print("청산 정책별 net (per feature, 0.15% 차감)")
    print("=" * 78)
    pol_rows = []
    for feat in summaries:
        df = summaries[feat]
        for pol in ["hard_sl_path", "tp_eod_noSL", "timestop_4h",
                    "eod_tpcap", "trailing_3pct"]:
            m = net_metrics(df[f"net_{pol}"])
            pol_rows.append(dict(feature=feat, policy=pol, **m))
            print(f"[{feat}] {pol:16s} n={m['n']:4d} mean={m['mean']*100:+.3f}% "
                  f"Sharpe={m['sharpe']:+.2f} hit={m['hit']*100:.1f}% "
                  f"cum={m['cum']*100:+.1f}% MDD={m['mdd']*100:.1f}%")
    pd.DataFrame(pol_rows).to_csv(OUT_DIR / "path_verify_bear_quiet_policy.csv", index=False)

    if len(base):
        print("\n" + "=" * 78)
        print("BASELINE (동일 bear_quiet 날 무작위 코인)")
        print("=" * 78)
        n = len(base)
        both = base["both_touched"].sum()
        print(f"baseline n={n} both_touched={both} ({both/n*100:.1f}%)")
        sub = base[base["both_touched"]]
        if len(sub):
            print(f"   그중 TP_first={ (sub['first']=='tp').sum()} "
                  f"SL_first={(sub['first']=='sl').sum()}")
        for pol in ["hard_sl_path", "tp_eod_noSL", "timestop_4h"]:
            m = net_metrics(base[f"net_{pol}"])
            print(f"baseline {pol:16s} n={m['n']} mean={m['mean']*100:+.3f}% "
                  f"Sharpe={m['sharpe']:+.2f} hit={m['hit']*100:.1f}%")

    log.info("done")


if __name__ == "__main__":
    main()
