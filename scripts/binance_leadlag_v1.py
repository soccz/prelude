"""Binance USDT D-1 lead-lag → Upbit KRW pump (day D) incremental-lift probe.

연구 질문
---------
Binance USDT 시장의 D-1 신호 (수익률, 거래량 surge, 거래소간 수익률 차)가
Upbit KRW 펌프 (day D high/open) 예측에 Upbit-only 베이스라인 대비 *추가 lift*를
주는가? (한 번도 신호로 안 써본 유일한 새 정보원)

이 스크립트는 self-contained 이다 (gan_t/xsec_alpha/fin import 없음). prelude 안의
data.database / data.collector_* 만 쓴다. 모델/라벨/피처 변경 없음 — 기존 pump_hunter
라벨(high_D/open_D ≥ thr)과 베이스라인 룰(roc_7d_rank > 0.85)을 그대로 재사용하고
Binance D-1 피처의 *증분*만 측정한다.

위생 (양보 X — CLAUDE.md §2.5)
------------------------------
1. Look-ahead leak 방어:
   - 모든 피처는 strictly ≤ D-1 (Upbit feature_date = D-1, Binance bar date = D-1).
   - 라벨은 day D (high_D/open_D) — feature_date = label_date - 1day 로 join (기존 leak-fix).
   - LEAK_COLS / next_* 는 애초에 만들지 않는다 (라벨만 별도 테이블).
2. 유니버스 시간정합: 유니버스 랭크는 feature_date(D-1) 거래대금 기준.
3. 거래비용: net 평가 시 왕복 0.15% (기본) + 보수 tier 0.5% 차감.

경계 정렬 (Boundary alignment) — 이 연구의 핵심 가정
----------------------------------------------------
- Upbit KRW 일봉: timestamp = KST-naive, 값 'YYYY-MM-DD 09:00:00' = 그 봉 시작.
  실제 커버 구간 = [그날 09:00 KST, 다음날 09:00 KST] = [그날 00:00 UTC, 다음날 00:00 UTC].
- Binance USDT 일봉: timestamp = UTC-naive, 값 'YYYY-MM-DD 00:00:00' = 그 봉 시작.
  실제 커버 구간 = [그날 00:00 UTC, 다음날 00:00 UTC].
- => Binance 날짜 D-1 봉 ≡ Upbit 날짜 D-1 봉 (DATE part 동일) = 동일 실제 UTC 일.
  Binance D-1 봉은 D-1 24:00 UTC = D 09:00 KST 에 완전 마감 → Upbit day D open 직전.
  따라서 Binance D-1 피처는 strictly ≤ Upbit D open. JOIN KEY = date part.
- 출처: SIGNAL.md §1.3 ("업비트 일봉 마감 = UTC 00:00 = KST 09:00") +
  data/collector_binance_d1.py (UTC ms → tz-naive UTC) + data/database.py 주석.
- `--inspect` / 본 실행은 이 가정을 실데이터로 재검증한다 (verify_boundary). 실패 시 abort.

사용
----
    # 0) (선택) Binance DB 갱신 — 39일 stale 이면 먼저
    python -m data.collector_binance_d1 --all --days 60

    # 1) 경계/매핑 점검만 (빠름)
    python scripts/binance_leadlag_v1.py --inspect

    # 2) 전체 증분 lift 평가 (pump20 라벨, top120 유니버스, embargo 10d)
    python scripts/binance_leadlag_v1.py --target pump20 --top-universe 120 \
        --embargo 10 --n-folds 5

산출물
------
    output/binance_leadlag_v1_boundary.csv      # 경계 검증 (BTC 정렬 corr 등)
    output/binance_leadlag_v1_mapping.csv       # KRW↔USDT 매핑 + 커버리지
    output/binance_leadlag_v1_oof_picks.csv     # OOF pick (rule_fire) 행 + 라벨
    output/binance_leadlag_v1_lift.csv          # fold별/전체 증분 lift 표
    output/binance_leadlag_v1_net.csv           # net (TP/SL) 시뮬 결과
"""
from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.database import list_markets, load_candles  # noqa: E402
from signals.validate import PurgedWalkForward  # noqa: E402

print = partial(print, flush=True)  # noqa: A001  (재현 로그 — 항상 flush)

ROOT = Path(__file__).resolve().parent.parent
UPBIT_DB = str(ROOT / "data" / "upbit_d1.db")
BINANCE_DB = str(ROOT / "data" / "binance_d1.db")
OUT_DIR = ROOT / "output"

# ---- 기존 pump_hunter 계약 (signals/pump_detector_v1.py 와 일치) ----
FEATURE_LOOKBACK_MIN = 30
BASELINE_ROC7_RANK_MIN = 0.85       # pump20 베이스라인 룰
COST_ROUNDTRIP = 0.0015             # net 기본 (수수료 0.1% + 슬리피지 0.05%)
COST_CONSERVATIVE = 0.005           # 보수 tier (왕복 0.5%)
TP_PCT = 0.05
SL_PCT = -0.03

# 스테이블/wrapped — Binance 매핑 제외 (가격이 ~1 USD, 펌프 무의미)
STABLE_WRAPPED = {
    "USDT", "USDC", "BUSD", "DAI", "FDUSD", "TUSD", "USDP", "PYUSD", "USDS",
    "USDD", "GUSD", "EUR", "EURT", "WBTC", "WETH", "STETH", "WBETH",
}


# ============================================================================
# 심볼 매핑
# ============================================================================
def krw_to_binance(market: str) -> str | None:
    """'KRW-XXX' → 'BINANCE-XXXUSDT'. 스테이블/wrapped 는 None."""
    if not market.startswith("KRW-"):
        return None
    base = market[len("KRW-"):]
    if base in STABLE_WRAPPED:
        return None
    return f"BINANCE-{base}USDT"


# ============================================================================
# 경계 정렬 검증 (실데이터)
# ============================================================================
def verify_boundary(upbit_db: str, binance_db: str) -> pd.DataFrame:
    """
    BTC 로 Upbit/Binance 일봉 경계가 같은 실제 UTC 일을 가리키는지 검증.

    검증 항목:
      1. Upbit timestamp time-of-day == 09:00:00, Binance == 00:00:00.
      2. 같은 DATE part 에 대해 두 거래소의 *일간 수익률*이 강하게 상관 (≥ 0.7 기대).
         (가격 레벨은 KRW vs USD 라 다르지만, log-return 은 거의 같아야 — 같은 자산 같은 봉.)
      3. shift 검정: Upbit ret(date) 와 Binance ret(date) 의 corr 가
         Upbit ret(date) 와 Binance ret(date-1) 또는 (date+1) 의 corr 보다 높아야
         (정렬이 맞으면 lag0 가 최대). 안 맞으면 abort 신호.

    return: 한 줄짜리 진단 DataFrame. corr_lag0 < corr_lag±1 이면 'MISALIGNED'.
    """
    ub = load_candles(upbit_db, "KRW-BTC")
    bn = load_candles(binance_db, "BINANCE-BTCUSDT")
    rows = []
    if ub is None or ub.empty or bn is None or bn.empty:
        return pd.DataFrame([{
            "check": "btc_load", "status": "FAIL",
            "detail": "KRW-BTC or BINANCE-BTCUSDT empty",
        }])

    ub = ub.copy(); bn = bn.copy()
    ub["timestamp"] = pd.to_datetime(ub["timestamp"])
    bn["timestamp"] = pd.to_datetime(bn["timestamp"])

    # 1) time-of-day
    ub_tod = ub["timestamp"].dt.strftime("%H:%M:%S").mode().iloc[0]
    bn_tod = bn["timestamp"].dt.strftime("%H:%M:%S").mode().iloc[0]
    rows.append({"check": "upbit_time_of_day", "status": "INFO", "detail": ub_tod})
    rows.append({"check": "binance_time_of_day", "status": "INFO", "detail": bn_tod})

    # 2,3) date-part return alignment
    ub["d"] = ub["timestamp"].dt.date
    bn["d"] = bn["timestamp"].dt.date
    ub["ret"] = np.log(ub["close"] / ub["close"].shift(1))
    bn["ret"] = np.log(bn["close"] / bn["close"].shift(1))
    m = pd.merge(
        ub[["d", "ret"]].rename(columns={"ret": "ret_ub"}),
        bn[["d", "ret"]].rename(columns={"ret": "ret_bn"}),
        on="d", how="inner",
    ).dropna()
    if len(m) < 30:
        rows.append({"check": "overlap", "status": "FAIL",
                     "detail": f"only {len(m)} overlapping days"})
        return pd.DataFrame(rows)

    corr0 = m["ret_ub"].corr(m["ret_bn"])
    # lag ±1 (Binance 날짜를 ±1 밀어서 join)
    bn_shift = bn[["d", "ret"]].copy()
    bn_shift["d_plus1"] = bn_shift["d"] + pd.Timedelta(days=1)
    bn_shift["d_minus1"] = bn_shift["d"] - pd.Timedelta(days=1)
    m_p1 = pd.merge(ub[["d", "ret"]], bn_shift[["d_plus1", "ret"]].rename(
        columns={"d_plus1": "d", "ret": "ret_bn"}), on="d", how="inner").dropna()
    m_m1 = pd.merge(ub[["d", "ret"]], bn_shift[["d_minus1", "ret"]].rename(
        columns={"d_minus1": "d", "ret": "ret_bn"}), on="d", how="inner").dropna()
    corr_p1 = m_p1["ret"].corr(m_p1["ret_bn"]) if len(m_p1) > 30 else np.nan
    corr_m1 = m_m1["ret"].corr(m_m1["ret_bn"]) if len(m_m1) > 30 else np.nan

    aligned = (corr0 >= 0.6) and (corr0 >= (corr_p1 if not np.isnan(corr_p1) else -1)) \
        and (corr0 >= (corr_m1 if not np.isnan(corr_m1) else -1))
    rows.append({"check": "btc_ret_corr_lag0", "status": "INFO", "detail": f"{corr0:.4f}"})
    rows.append({"check": "btc_ret_corr_lag+1", "status": "INFO", "detail": f"{corr_p1:.4f}"})
    rows.append({"check": "btc_ret_corr_lag-1", "status": "INFO", "detail": f"{corr_m1:.4f}"})
    rows.append({"check": "n_overlap_days", "status": "INFO", "detail": str(len(m))})
    rows.append({"check": "ALIGNMENT", "status": "PASS" if aligned else "MISALIGNED",
                 "detail": f"date-part join valid (lag0 corr {corr0:.3f} is max)"
                 if aligned else
                 f"lag0 corr {corr0:.3f} NOT max — DO NOT trust join"})
    return pd.DataFrame(rows)


# ============================================================================
# Upbit 베이스라인 피처 + 라벨
# ============================================================================
def _upbit_market_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("timestamp").copy()
    close = d["close"].astype(float)
    high = d["high"].astype(float)
    low = d["low"].astype(float)
    prev = close.shift(1)
    d["u_log_return_1d"] = np.log(close / prev)
    d["u_roc_7d"] = close.pct_change(7) * 100.0
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    d["u_atr_pct_14"] = tr.rolling(14).mean() / close.replace(0, np.nan)
    # 라벨 (day D = 이 행)
    op = d["open"].astype(float)
    d["pump_max_return"] = np.where(op > 0, high / op - 1.0, np.nan)
    d["eod_ret_D"] = np.where(op > 0, close / op - 1.0, np.nan)
    return d


def build_upbit_panel(upbit_db: str) -> pd.DataFrame:
    """
    Upbit KRW 일봉 → (feature_date 기준) 베이스라인 피처 + (label_date=feature_date+1) 라벨.

    반환 행 = 하나의 (market, feature_date). 그 행은:
      - 피처: feature_date 까지의 Upbit 베이스라인 (u_*) — strictly ≤ D-1
      - 라벨/진입: day D (= feature_date + 1) 의 open/high/close
      - quote_volume_d1: feature_date 거래대금 (유니버스 랭크용)
    leak-fix: 라벨을 별도 계산 후 feature_date = label_date - 1day 로 self-join.
    """
    markets = [m for m in list_markets(upbit_db) if str(m).startswith("KRW-")]
    feat_rows = []
    for market in markets:
        raw = load_candles(upbit_db, market)
        if raw is None or len(raw) < FEATURE_LOOKBACK_MIN + 2:
            continue
        d = _upbit_market_features(raw)
        d["market"] = market  # load_candles 는 market 컬럼을 반환하지 않음
        d["timestamp"] = pd.to_datetime(d["timestamp"])
        d["feature_date"] = d["timestamp"].dt.date
        # entry/label from NEXT row (day D)
        d["label_date"] = (d["timestamp"] + pd.Timedelta(days=1)).dt.date
        nxt = d[["feature_date", "open", "high", "close", "pump_max_return", "eod_ret_D"]].copy()
        nxt = nxt.rename(columns={
            "feature_date": "label_date_key", "open": "entry_open_D",
            "high": "high_D", "close": "close_D",
            "pump_max_return": "pump_max_return_D", "eod_ret_D": "eod_ret_D2",
        })
        # feature row at t join label row at t+1 (=day D)
        cur = d[["market", "timestamp", "feature_date",
                 "u_log_return_1d", "u_roc_7d", "u_atr_pct_14",
                 "quote_volume", "close"]].copy()
        cur["label_date"] = (cur["timestamp"] + pd.Timedelta(days=1)).dt.date
        merged = cur.merge(nxt, left_on=["label_date"], right_on=["label_date_key"], how="inner")
        # nxt 는 feature_date 키가 아니라 같은 market 안에서만 — market 단위 처리라 OK,
        # 하지만 안전하게 market 별로 처리하므로 cross-market join 위험 없음.
        feat_rows.append(merged)
    if not feat_rows:
        return pd.DataFrame()
    out = pd.concat(feat_rows, ignore_index=True)
    out = out.replace([np.inf, -np.inf], np.nan)
    out["pump_max_return"] = out["pump_max_return_D"]
    out["eod_ret_D"] = out["eod_ret_D2"]
    out["quote_volume_d1"] = out["quote_volume"]
    out = out.dropna(subset=["pump_max_return", "entry_open_D", "u_roc_7d"]).reset_index(drop=True)
    return out


# ============================================================================
# Binance D-1 피처
# ============================================================================
def build_binance_features(binance_db: str, needed_bn_markets: set[str]) -> pd.DataFrame:
    """
    Binance D-1 일봉 → 거래소간 lead-lag 피처 (전부 ≤ D-1).

      b_ret_1d  : Binance D-1 일간 log return
      b_ret_7d  : Binance 7일 수익률
      b_vol_surge : D-1 quote_volume / 자기 20d 평균 (>1 = 거래대금 급증)
      (Upbit 대비 상대는 join 후 계산: b_ret_1d - u_log_return_1d 등)

    join key = (binance_market, bar_date). bar_date(Binance, UTC) == feature_date(Upbit, D-1).
    """
    rows = []
    for bm in sorted(needed_bn_markets):
        raw = load_candles(binance_db, bm)
        if raw is None or len(raw) < 25:
            continue
        d = raw.sort_values("timestamp").copy()
        d["timestamp"] = pd.to_datetime(d["timestamp"])
        close = d["close"].astype(float)
        qv = d["quote_volume"].astype(float)
        d["b_ret_1d"] = np.log(close / close.shift(1))
        d["b_ret_7d"] = close.pct_change(7)
        d["b_vol_surge"] = qv / qv.rolling(20, min_periods=10).mean()
        d["bn_market"] = bm
        # bar_date(UTC) == Upbit feature_date(D-1)  (경계 정렬: date part 동일)
        d["feature_date"] = d["timestamp"].dt.date
        rows.append(d[["bn_market", "feature_date", "b_ret_1d", "b_ret_7d", "b_vol_surge"]])
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out.replace([np.inf, -np.inf], np.nan)


# ============================================================================
# 베이스라인 룰 + 증분 lift
# ============================================================================
def add_universe_and_rank(panel: pd.DataFrame, top_universe: int) -> pd.DataFrame:
    """feature_date 기준 유니버스 top-N + roc_7d_rank (기존 pump_hunter 와 동일 랭크)."""
    out = panel.copy()
    out["liq_rank_daily"] = out.groupby("feature_date")["quote_volume_d1"].rank(
        method="dense", ascending=False, na_option="bottom")
    out = out[out["liq_rank_daily"] <= top_universe].copy()
    out["roc_7d_rank"] = out.groupby("feature_date")["u_roc_7d"].rank(pct=True)
    # Binance 상대 피처 (join 후) — 거래소간 수익률 차 = 김프 변동 proxy
    out["bn_minus_upbit_ret_1d"] = out["b_ret_1d"] - out["u_log_return_1d"]
    return out.reset_index(drop=True)


def lift_table(df: pd.DataFrame, target: str, mask_name: str, mask: pd.Series) -> dict:
    """mask 가 잡은 행의 base rate / lift / recall / n."""
    n = int(mask.sum())
    base = float(df[target].mean()) if len(df) else np.nan
    if n == 0:
        return {"rule": mask_name, "n_fire": 0, "hit": np.nan, "base": base,
                "lift": np.nan, "recall": 0.0, "n_pos_total": int(df[target].sum())}
    sub = df[mask]
    hit = float(sub[target].mean())
    n_pos_total = int(df[target].sum())
    recall = float(sub[target].sum()) / n_pos_total if n_pos_total > 0 else np.nan
    return {"rule": mask_name, "n_fire": n, "hit": hit, "base": base,
            "lift": hit / base if base > 0 else np.nan,
            "recall": recall, "n_pos_total": n_pos_total}


def net_sim(df: pd.DataFrame, mask: pd.Series, cost: float) -> dict:
    """
    rule 이 잡은 행을 day D open 진입 → TP+5%/SL-3% 보수 모델.
    경로(high/low intraday)는 일봉만으론 순서 불명 → 보수적으로:
      - high_D/open-1 >= TP 이고 low 가 SL 안 쳤으면 TP (낙관) 대신,
      - **보수**: SL 우선 가정 불가하니 day-EOD 와 TP/SL 중 worst-case 가 아니라
        실현가능 추정 = min(TP 도달, EOD) with SL floor.
    일봉 한계상 정확한 TP/SL 순서를 모름 → 두 가지 다 보고:
      gross_eod  : close_D/open_D - 1 (단순 종가청산)
      gross_tpsl : TP 도달시 +5%, 아니면 EOD, 단 EOD < SL 이면 SL (-3%) floor
    net = gross - cost.
    """
    if mask.sum() == 0:
        return {"n": 0}
    sub = df[mask].copy()
    op = sub["entry_open_D"].astype(float)
    hi = sub["high_D"].astype(float)
    cl = sub["close_D"].astype(float)
    eod = cl / op - 1.0
    tp_hit = (hi / op - 1.0) >= TP_PCT
    gross_tpsl = np.where(tp_hit, TP_PCT, np.maximum(eod, SL_PCT))
    out = {}
    for name, g in [("eod", eod.values), ("tpsl", np.asarray(gross_tpsl))]:
        net = g - cost
        out[f"gross_{name}_mean"] = float(np.nanmean(g))
        out[f"net_{name}_mean"] = float(np.nanmean(net))
        out[f"net_{name}_median"] = float(np.nanmedian(net))
        wins = net[net > 0]
        out[f"net_{name}_winrate"] = float(len(wins) / len(net)) if len(net) else np.nan
    out["n"] = int(mask.sum())
    return out


def run(args) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP 1 — 경계 정렬 검증 (BTC, 실데이터)")
    print("=" * 70)
    bdf = verify_boundary(args.upbit_db, args.binance_db)
    print(bdf.to_string(index=False))
    bdf.to_csv(OUT_DIR / "binance_leadlag_v1_boundary.csv", index=False)
    align = bdf[bdf["check"] == "ALIGNMENT"]
    if not align.empty and align.iloc[0]["status"] == "MISALIGNED":
        print("\n[ABORT] 경계 정렬 실패 — Binance D-1 join 을 신뢰할 수 없음. 중단.")
        sys.exit(2)
    if args.inspect_only:
        # 매핑 커버리지만 추가로 보고
        _mapping_report(args)
        print("\n[--inspect] 경계 + 매핑만 점검 완료.")
        return

    print("\n" + "=" * 70)
    print("STEP 2 — Upbit 베이스라인 panel (피처 ≤ D-1, 라벨 day D)")
    print("=" * 70)
    up = build_upbit_panel(args.upbit_db)
    if up.empty:
        print("[ABORT] Upbit panel 비어있음."); sys.exit(2)
    up["timestamp"] = pd.to_datetime(up["timestamp"])
    print(f"Upbit panel rows={len(up):,}  markets={up['market'].nunique()}  "
          f"date range {up['feature_date'].min()} ~ {up['feature_date'].max()}")

    print("\n" + "=" * 70)
    print("STEP 3 — 심볼 매핑 + Binance D-1 피처 join")
    print("=" * 70)
    up["bn_market"] = up["market"].map(krw_to_binance)
    n_mappable = up.loc[up["bn_market"].notna(), "market"].nunique()
    n_total = up["market"].nunique()
    needed = set(up["bn_market"].dropna().unique())
    bn = build_binance_features(args.binance_db, needed)
    if bn.empty:
        print("[ABORT] Binance 피처 비어있음 (DB stale/empty?)."); sys.exit(2)
    bn_have = set(bn["bn_market"].unique())
    merged = up.merge(bn, on=["bn_market", "feature_date"], how="left")
    # join 성공 = Binance 피처가 실제로 붙은 행
    has_bn = merged["b_ret_1d"].notna()
    map_rows = []
    for m in sorted(up["market"].unique()):
        bm = krw_to_binance(m)
        sub = merged[merged["market"] == m]
        map_rows.append({
            "krw_market": m, "binance_market": bm,
            "mappable": bm is not None,
            "binance_db_has": bm in bn_have if bm else False,
            "n_rows": len(sub),
            "n_rows_with_binance": int(sub["b_ret_1d"].notna().sum()),
            "coverage": float(sub["b_ret_1d"].notna().mean()) if len(sub) else 0.0,
        })
    mapdf = pd.DataFrame(map_rows)
    mapdf.to_csv(OUT_DIR / "binance_leadlag_v1_mapping.csv", index=False)
    row_cov = float(has_bn.mean())
    print(f"매핑 가능 코인: {n_mappable}/{n_total} ({n_mappable/n_total*100:.1f}%)  "
          f"(스테이블/wrapped 제외)")
    print(f"Binance DB 보유: {len(bn_have & needed)}/{len(needed)} 매핑심볼")
    print(f"행 단위 Binance join 커버리지: {has_bn.sum():,}/{len(merged):,} "
          f"({row_cov*100:.1f}%)  ← Binance DB stale 면 이 값이 낮음")
    print(f"Binance 피처 날짜 max: {bn['feature_date'].max()}  "
          f"(Upbit feature_date max: {up['feature_date'].max()})")

    print("\n" + "=" * 70)
    print("STEP 4 — 유니버스 top-N + 베이스라인 룰 + 증분 lift (Purged WF, OOF)")
    print("=" * 70)
    # Binance 피처 있는 구간으로 scope (stale 이면 자동으로 최근 fold 가 빠짐 — 정직)
    merged = merged[merged["feature_date"] <= bn["feature_date"].max()].copy()
    panel = add_universe_and_rank(merged, args.top_universe)
    panel = panel.dropna(subset=["pump_max_return"]).reset_index(drop=True)
    panel[args.target + "_bin"] = (panel["pump_max_return"] >= (
        0.20 if args.target == "pump20" else 0.15)).astype(int)
    target = args.target + "_bin"
    print(f"유니버스 panel rows={len(panel):,}  base rate({args.target})="
          f"{panel[target].mean()*100:.2f}%")

    # ---- Purged WF: 각 fold val(OOF)에서 룰 비교 ----
    splitter = PurgedWalkForward(n_folds=args.n_folds, embargo_days=args.embargo,
                                 holdout_days=args.holdout)
    fold_lifts = []
    oof_pick_frames = []
    for fold, (tr_dates, va_dates) in enumerate(splitter.split(panel["timestamp"]), 1):
        va = panel[panel["timestamp"].isin(va_dates)].copy()
        if len(va) < 200:
            continue
        # 베이스라인 룰: roc_7d_rank > 0.85
        base_mask = va["roc_7d_rank"] > BASELINE_ROC7_RANK_MIN
        # 증분 후보 1: 베이스라인 AND Binance D-1 momentum (b_ret_1d > 0)
        #   가설: Binance 에서 이미 D-1 에 위로 움직인 코인이 Upbit 펌프로 이어짐
        bn_avail = va["b_ret_1d"].notna()
        base_and_bn_up = base_mask & bn_avail & (va["b_ret_1d"] > 0)
        # 증분 후보 2: 베이스라인 AND Binance vol surge (b_vol_surge > 1.5)
        base_and_bn_volsurge = base_mask & bn_avail & (va["b_vol_surge"] > 1.5)
        # 증분 후보 3: 베이스라인 AND Upbit가 Binance보다 약했다 (catch-up 가설)
        #   bn_minus_upbit_ret_1d > 0 = Binance가 더 올랐다 = Upbit catch-up 여지
        base_and_catchup = base_mask & bn_avail & (va["bn_minus_upbit_ret_1d"] > 0)
        # Binance-only (베이스라인 없이) — 단독 정보가치 참고용
        bn_only = bn_avail & (va["b_vol_surge"] > 2.0) & (va["b_ret_1d"] > 0.03)

        for nm, mk in [
            ("baseline_roc7", base_mask),
            ("base_AND_bn_up", base_and_bn_up),
            ("base_AND_bn_volsurge", base_and_bn_volsurge),
            ("base_AND_catchup", base_and_catchup),
            ("bn_only_surge+mom", bn_only),
        ]:
            lt = lift_table(va, target, nm, mk)
            lt["fold"] = fold
            lt["n_val"] = len(va)
            lt["bn_coverage_val"] = float(bn_avail.mean())
            fold_lifts.append(lt)

        # OOF picks (베이스라인 fire 행 + Binance 피처 — evaluator 가 재검증)
        picks = va[base_mask].copy()
        picks["fold"] = fold
        oof_pick_frames.append(picks[[
            "market", "feature_date", "timestamp", "roc_7d_rank", "u_roc_7d",
            "u_atr_pct_14", "b_ret_1d", "b_ret_7d", "b_vol_surge",
            "bn_minus_upbit_ret_1d", "pump_max_return", target,
            "entry_open_D", "high_D", "close_D",
        ]])

    if not fold_lifts:
        print("[ABORT] WF fold 가 유효하지 않음 (val < 200)."); sys.exit(2)
    lift_df = pd.DataFrame(fold_lifts)
    lift_df.to_csv(OUT_DIR / "binance_leadlag_v1_lift.csv", index=False)

    # ---- pooled OOF (전체 fold 합쳐서) ----
    oof = pd.concat(oof_pick_frames, ignore_index=True) if oof_pick_frames else pd.DataFrame()
    if not oof.empty:
        oof.to_csv(OUT_DIR / "binance_leadlag_v1_oof_picks.csv", index=False)

    print("\n--- fold별 증분 lift (OOF) ---")
    show = lift_df[["fold", "rule", "n_fire", "hit", "base", "lift", "recall",
                    "bn_coverage_val"]].copy()
    for c in ["hit", "base", "lift", "recall", "bn_coverage_val"]:
        show[c] = show[c].astype(float).round(4)
    print(show.to_string(index=False))

    # ---- rule별 fold 평균 요약 ----
    print("\n--- rule별 fold 평균 ---")
    summ = []
    for nm, g in lift_df.groupby("rule"):
        valid = g.dropna(subset=["lift"])
        summ.append({
            "rule": nm,
            "n_folds": int(len(g)),
            "n_folds_fired": int((g["n_fire"] > 0).sum()),
            "mean_lift": float(valid["lift"].mean()) if len(valid) else np.nan,
            "min_lift": float(valid["lift"].min()) if len(valid) else np.nan,
            "mean_hit": float(valid["hit"].mean()) if len(valid) else np.nan,
            "total_fire": int(g["n_fire"].sum()),
            "mean_recall": float(g["recall"].mean()),
        })
    summ_df = pd.DataFrame(summ).sort_values("mean_lift", ascending=False)
    print(summ_df.to_string(index=False))

    # ---- 증분 판정 (베이스라인 대비) ----
    base_ml = summ_df.loc[summ_df["rule"] == "baseline_roc7", "mean_lift"]
    base_ml = float(base_ml.iloc[0]) if len(base_ml) and not base_ml.isna().all() else np.nan
    print(f"\n베이스라인 mean_lift = {base_ml:.3f}")
    print("증분 룰의 베이스라인 대비 lift 배수 (≥1.3x + fold 일관 이어야 후보):")
    for _, r in summ_df.iterrows():
        if r["rule"] == "baseline_roc7":
            continue
        ratio = r["mean_lift"] / base_ml if base_ml and base_ml > 0 else np.nan
        fold_consistent = r["n_folds_fired"] >= max(2, int(0.6 * r["n_folds"]))
        verdict = "후보가능" if (not np.isnan(ratio) and ratio >= 1.3 and fold_consistent) else "약함"
        print(f"  {r['rule']:<26} incr_lift_ratio={ratio:.3f}  "
              f"folds_fired={r['n_folds_fired']}/{r['n_folds']}  -> {verdict}")

    # ---- net 비교 (베이스라인 vs 최고 증분 룰) ----
    print("\n" + "=" * 70)
    print("STEP 5 — net 시뮬 (TP5/SL3, 비용 0.15% + 보수 0.5%)")
    print("=" * 70)
    if not oof.empty:
        net_rows = []
        for nm, mk in [
            ("baseline_roc7", pd.Series(True, index=oof.index)),
            ("base_AND_bn_up", oof["b_ret_1d"] > 0),
            ("base_AND_bn_volsurge", oof["b_vol_surge"] > 1.5),
            ("base_AND_catchup", oof["bn_minus_upbit_ret_1d"] > 0),
        ]:
            for cost, ctag in [(COST_ROUNDTRIP, "0.15%"), (COST_CONSERVATIVE, "0.50%")]:
                ns = net_sim(oof, mk.fillna(False), cost)
                ns["rule"] = nm
                ns["cost"] = ctag
                net_rows.append(ns)
        net_df = pd.DataFrame(net_rows)
        net_df.to_csv(OUT_DIR / "binance_leadlag_v1_net.csv", index=False)
        cols = ["rule", "cost", "n", "net_tpsl_mean", "net_tpsl_winrate",
                "net_eod_mean"]
        cols = [c for c in cols if c in net_df.columns]
        print(net_df[cols].round(4).to_string(index=False))

    print("\n[DONE] 산출물: output/binance_leadlag_v1_*.csv")


def _mapping_report(args) -> None:
    """--inspect 전용: panel 없이 매핑 가능 코인 % 만 빠르게."""
    krw = [m for m in list_markets(args.upbit_db) if str(m).startswith("KRW-")]
    bn_have = set(list_markets(args.binance_db))
    rows = []
    n_map = 0
    n_have = 0
    for m in krw:
        bm = krw_to_binance(m)
        have = bm in bn_have if bm else False
        if bm:
            n_map += 1
        if have:
            n_have += 1
        rows.append({"krw_market": m, "binance_market": bm,
                     "mappable": bm is not None, "binance_db_has": have})
    mdf = pd.DataFrame(rows)
    mdf.to_csv(OUT_DIR / "binance_leadlag_v1_mapping.csv", index=False)
    print(f"\nKRW markets={len(krw)}  mappable={n_map} ({n_map/len(krw)*100:.1f}%)  "
          f"in_binance_db={n_have} ({n_have/len(krw)*100:.1f}%)")
    # Binance DB 최신성
    from data.database import latest_timestamp
    bt = latest_timestamp(args.binance_db, "BINANCE-BTCUSDT")
    ut = latest_timestamp(args.upbit_db, "KRW-BTC")
    print(f"Binance BTC latest bar: {bt}   Upbit BTC latest bar: {ut}")


def main():
    p = argparse.ArgumentParser(description="Binance D-1 → Upbit pump 증분 lift 연구")
    p.add_argument("--upbit-db", default=UPBIT_DB)
    p.add_argument("--binance-db", default=BINANCE_DB)
    p.add_argument("--target", default="pump20", choices=["pump20", "pump15"])
    p.add_argument("--top-universe", type=int, default=120,
                   help="유니버스 거래대금 상위 N (feature_date 기준)")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--embargo", type=int, default=10, help="WF embargo (days, ≥5 위생)")
    p.add_argument("--holdout", type=int, default=180)
    p.add_argument("--inspect", dest="inspect_only", action="store_true",
                   help="경계 정렬 + 매핑 커버리지만 점검 (빠름)")
    args = p.parse_args()

    if args.embargo < 5:
        print(f"[WARN] embargo {args.embargo} < 5 — 위생 권고 위반. 5 로 올림.")
        args.embargo = 5
    run(args)


if __name__ == "__main__":
    main()
