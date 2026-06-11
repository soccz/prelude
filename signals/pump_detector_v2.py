"""PUMP hunter rule detector v2 — Upbit 모멘텀 + Binance 거래량 surge 융합.

검증 체인 (2026-06-11, scripts/binance_leadlag_v1.py + evaluator 감사 + 15m 재계산):
- 룰: roc_7d_rank > 0.85 (Upbit, D-1 횡단 rank) AND b_vol_surge > 1.5
  (Binance 같은 자산 D-1 거래대금 / 자기 20d 평균)
- 최근 7개월 순수 OOS: pump20 hit 8.1% (113/1390) vs baseline 5.6% vs
  base rate ~1.4% — lift ~6x, 5/5 fold 일관 (전기간 lift 4.34x)
- ★ 정직 고지: 자동매매 룰 (TP5/SL3) 기준 net 은 음수 (-0.36%, 최근 OOS).
  이 detector 는 radar — hit 엣지로 후보를 띄우고 exit 은 사용자 판단.

Leak contract (v1 과 동일):
- 결정일 D 는 D-1 이전 피처만 사용. Binance D-1 일봉은 00:00 UTC = KST 09:00
  마감 — Upbit D open 직전 완전 확정 (경계 검증: BTC ret corr lag0 0.962).
- entry_open 은 D 09:00 일봉 open (paper/shadow 평가용).
- 이 모듈은 텔레그램/주문/원장 기록 없음 (runner 가 담당).

Binance 데이터 의존: data/binance_d1.db 가 feature_date 까지 있어야 함.
stale 이면 후보 0 + meta 에 사유 기록 (조용히 잘못된 신호 내지 않음).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.database import load_candles
from signals.pump_detector_v1 import (
    UNIVERSE_TOP_N,
    build_feature_frame,
)

BINANCE_DB = str(Path(__file__).resolve().parent.parent / "data" / "binance_d1.db")

# 검증된 룰 임계 (placeholder §2.5 — scripts/binance_leadlag_v1.py 의 고정 임계)
ROC7_RANK_MIN = 0.85
BN_VOL_SURGE_MIN = 1.5
BN_VOL_LOOKBACK = 20
MAX_CANDIDATES = 5          # 텔레그램 radar 용 — 최근 OOS 평균 fire ~9/일 중 상위만

# 정직 표기용 OOS 실측 (최근 7개월, 2025-11 ~ 2026-06)
OOS_HIT_PCT = 8.1           # pump20 hit (113/1390)
OOS_BASELINE_HIT_PCT = 5.6  # roc 룰 단독
OOS_BASE_RATE_PCT = 1.4     # universe 전체
OOS_NET_TP5SL3_PCT = -0.36  # 자동 룰 기준 net (음수 — radar 고지용)

SL_PCT = -0.03
TP_PCT = 0.05

# 스테이블/래핑 등 Binance USDT 페어 매핑 제외 (binance_leadlag_v1 과 동일)
_UNMAPPABLE = {"KRW-USDT", "KRW-USDC", "KRW-DAI", "KRW-TUSD", "KRW-WBTC", "KRW-WETH"}


def krw_to_binance(market: str) -> str | None:
    if market in _UNMAPPABLE:
        return None
    base = str(market).replace("KRW-", "")
    return f"BINANCE-{base}USDT"


def binance_volsurge_for_date(feature_date, needed_bn: set[str],
                              binance_db: str = BINANCE_DB) -> tuple[pd.DataFrame, str]:
    """feature_date 의 Binance b_vol_surge. (df, status) — status 는 freshness 진단."""
    f_ts = pd.Timestamp(feature_date)
    rows = []
    latest_seen = None
    for bm in sorted(needed_bn):
        raw = load_candles(binance_db, bm)
        if raw is None or len(raw) < BN_VOL_LOOKBACK + 2:
            continue
        d = raw.sort_values("timestamp").copy()
        d["timestamp"] = pd.to_datetime(d["timestamp"])
        if latest_seen is None or d["timestamp"].max() > latest_seen:
            latest_seen = d["timestamp"].max()
        d = d[d["timestamp"] <= f_ts]
        if len(d) < BN_VOL_LOOKBACK + 1:
            continue
        last = d.iloc[-1]
        if pd.Timestamp(last["timestamp"]).normalize() != f_ts.normalize():
            continue  # feature_date 봉 없음 (stale/미상장)
        qv = d["quote_volume"].astype(float)
        base = qv.iloc[-(BN_VOL_LOOKBACK + 1):-1].mean()
        if not np.isfinite(base) or base <= 0:
            continue
        rows.append({
            "bn_market": bm,
            "b_vol_surge": float(qv.iloc[-1] / base),
            "b_ret_1d": float(np.log(d["close"].iloc[-1] / d["close"].iloc[-2]))
            if len(d) >= 2 and d["close"].iloc[-2] > 0 else np.nan,
        })
    out = pd.DataFrame(rows)
    if latest_seen is None:
        return out, "binance_db_empty"
    if latest_seen.normalize() < f_ts.normalize():
        return out, f"binance_stale (latest={latest_seen.date()}, need={f_ts.date()})"
    return out, "ok"


def score_pump_v2_candidates(asof_date, *, db_path: str | None = None,
                             binance_db: str = BINANCE_DB,
                             top_universe: int = UNIVERSE_TOP_N,
                             max_candidates: int = MAX_CANDIDATES,
                             limit_markets: int | None = None) -> dict:
    """결정일 D 의 v2 후보 (record-only scoring). 텔레그램/원장 기록 없음."""
    asof = pd.Timestamp(asof_date).normalize()
    kwargs = {"top_universe": top_universe, "limit_markets": limit_markets}
    if db_path:
        kwargs["db_path"] = db_path
    frame = build_feature_frame(asof, **kwargs)

    meta = {
        "asof": str(asof.date()),
        "model_id": "pump_hunter_v2",
        "rule_version": "pump_detector_v2",
        "rule": f"roc_7d_rank > {ROC7_RANK_MIN} AND b_vol_surge > {BN_VOL_SURGE_MIN}",
        "universe_n": int(len(frame)),
        "binance_status": "n/a",
        "n_candidates": 0,
        "candidates": [],
        "oos": {
            "hit_pct": OOS_HIT_PCT,
            "baseline_hit_pct": OOS_BASELINE_HIT_PCT,
            "base_rate_pct": OOS_BASE_RATE_PCT,
            "net_tp5sl3_pct": OOS_NET_TP5SL3_PCT,
        },
    }
    if frame.empty:
        meta["binance_status"] = "upbit_frame_empty"
        return meta
    meta["feature_date"] = str(frame["feature_date"].iloc[0])
    meta["btc_regime"] = str(frame["btc_regime"].iloc[0])

    frame = frame.copy()
    frame["bn_market"] = frame["market"].map(krw_to_binance)
    base_mask = frame["roc_7d_rank"] > ROC7_RANK_MIN
    needed = set(frame.loc[base_mask, "bn_market"].dropna())
    bn, status = binance_volsurge_for_date(meta["feature_date"], needed, binance_db)
    meta["binance_status"] = status
    if bn.empty:
        return meta

    cand = frame[base_mask].merge(bn, on="bn_market", how="inner")
    cand = cand[cand["b_vol_surge"] > BN_VOL_SURGE_MIN].copy()
    if cand.empty:
        return meta

    # 정렬: surge 강도 × 모멘텀 rank (단순·해석가능)
    cand["score"] = cand["b_vol_surge"].clip(upper=10.0) / 10.0 * 0.6 + cand["roc_7d_rank"] * 0.4
    cand = cand.sort_values(["score", "b_vol_surge"], ascending=False).head(max_candidates)
    cand["rank"] = np.arange(1, len(cand) + 1)

    meta["n_candidates"] = int(len(cand))
    meta["candidates"] = [
        {
            "market": r["market"],
            "rank": int(r["rank"]),
            "score": round(float(r["score"]), 4),
            "entry_open": float(r["entry_open"]),
            "roc_7d": float(r["roc_7d"]),
            "roc_7d_rank": round(float(r["roc_7d_rank"]), 4),
            "atr_pct_14": float(r["atr_pct_14"]),
            "log_return_1d": float(r["log_return_1d"]),
            "b_vol_surge": round(float(r["b_vol_surge"]), 3),
            "b_ret_1d": round(float(r["b_ret_1d"]), 5) if pd.notna(r["b_ret_1d"]) else None,
            "liq_rank_daily": float(r["liq_rank_daily"]),
            "btc_regime": r["btc_regime"],
            "rule_id": "roc7_rank+bn_volsurge",
        }
        for _, r in cand.iterrows()
    ]
    return meta
