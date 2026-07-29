"""PUMP hunter rule detector v1.

This module turns the pump rule discovery result into a daily, record-only
detector. It is intentionally a rule detector, not a promoted trading model.

Leak contract:
- decision date D uses features from D-1 and earlier only
- entry_open is read from the D 09:00 daily candle for paper/shadow evaluation
- no Telegram, no order API, no ledger writes in this signal module
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data.database import list_markets, load_candles
from data.market_universe import is_excluded_signal_market
from signals.features import compute_btc_features

DB_PATH = str(Path(__file__).resolve().parent.parent / "data" / "upbit_d1.db")

FEATURE_LOOKBACK_MIN = 22
UNIVERSE_TOP_N = 100
MAX_CANDIDATES = 20

# Rules mined by scripts/pump_rule_discovery_v1.py and summarized in
# _workspace/angleB_signal-researcher_pump_rule_mining.md.
ROC7_RANK_PUMP20_MIN = 0.85
ROC7_RANK_PUMP15_MIN = 0.854
ATR_PCT14_PUMP15_MIN = 0.070
LOG_RETURN_1D_PUMP15_MAX = 0.117

# Honest coarse probabilities for display/filtering only. They are discovery
# priors, not a calibrated production model.
PUMP20_BASE_RATE = 0.019
PUMP20_RULE_MEAN_LIFT = 3.35
PUMP15_BASE_RATE = 0.035
PUMP15_RULE_MEAN_LIFT = 2.10
EST_PUMP20_PROB = PUMP20_BASE_RATE * PUMP20_RULE_MEAN_LIFT
EST_PUMP15_PROB = PUMP15_BASE_RATE * PUMP15_RULE_MEAN_LIFT

SL_PCT = -0.03
TP_PCT = 0.05


@dataclass(frozen=True)
class PumpDetectorConfig:
    upbit_d1: str = DB_PATH
    top_universe: int = UNIVERSE_TOP_N
    max_candidates: int = MAX_CANDIDATES


def _asof_ts(asof_date) -> pd.Timestamp:
    timestamp = pd.Timestamp(asof_date)
    if pd.isna(timestamp):
        raise ValueError("asof_date must be a valid timestamp")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("Asia/Seoul").tz_localize(None)
    return timestamp.normalize() + pd.Timedelta(hours=9)


def _feature_ts(asof_date) -> pd.Timestamp:
    return _asof_ts(asof_date) - pd.Timedelta(days=1)


def _compute_market_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("timestamp").copy()
    close = d["close"].astype(float)
    high = d["high"].astype(float)
    low = d["low"].astype(float)
    prev_close = close.shift(1)

    d["log_return_1d"] = np.log(close / prev_close)
    d["roc_7d"] = close.pct_change(7) * 100.0

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    d["atr_pct_14"] = tr.rolling(14).mean() / close.replace(0, np.nan)
    return d


def _btc_regime_for_feature_date(db_path: str, feature_ts: pd.Timestamp) -> str:
    # DB/파일/SQLite 장애를 정상 데이터값인 "unknown" 으로 강등하지 않는다.
    # load_candles 예외는 일일 runner까지 전파해 nonzero 실패·운영 경보로
    # 이어져야 한다. "unknown" 은 조회가 정상 완료됐지만 해당 시점의
    # BTC 데이터가 실제로 없는 경우에만 사용한다.
    btc = load_candles(db_path, "KRW-BTC")
    if btc is None or btc.empty:
        return "unknown"
    btc = btc.copy()
    btc["timestamp"] = pd.to_datetime(btc["timestamp"])
    btc = btc[btc["timestamp"] <= feature_ts]
    if btc.empty:
        return "unknown"
    feats = compute_btc_features(btc)
    row = feats[feats["timestamp"] == feature_ts]
    if row.empty:
        return "unknown"
    return str(row.iloc[-1].get("btc_regime", "unknown"))


def build_feature_frame(asof_date, *, db_path: str = DB_PATH,
                        top_universe: int = UNIVERSE_TOP_N,
                        limit_markets: int | None = None) -> pd.DataFrame:
    """Build D-1 feature rows plus D entry open for all eligible KRW markets."""
    if isinstance(top_universe, bool) or top_universe <= 0:
        raise ValueError("top_universe must be positive")
    if (
        limit_markets is not None
        and (isinstance(limit_markets, bool) or limit_markets <= 0)
    ):
        raise ValueError("limit_markets must be positive when provided")
    decision_ts = _asof_ts(asof_date)
    feature_ts = _feature_ts(asof_date)

    markets = [
        str(market)
        for market in list_markets(db_path)
        if (
            str(market).startswith("KRW-")
            and not is_excluded_signal_market(str(market))
        )
    ]
    if limit_markets is not None:
        markets = markets[:limit_markets]

    btc_regime = _btc_regime_for_feature_date(db_path, feature_ts)
    rows: list[dict] = []
    for market in markets:
        raw = load_candles(db_path, market)
        if raw is None or len(raw) < FEATURE_LOOKBACK_MIN:
            continue
        raw = raw.copy()
        raw["timestamp"] = pd.to_datetime(raw["timestamp"])
        raw = raw[raw["timestamp"] <= decision_ts]
        if len(raw) < FEATURE_LOOKBACK_MIN:
            continue

        entry_row = raw[raw["timestamp"] == decision_ts]
        if len(entry_row) != 1:
            continue

        hist = raw[raw["timestamp"] <= feature_ts]
        if len(hist) < FEATURE_LOOKBACK_MIN:
            continue
        recent_history = hist.tail(FEATURE_LOOKBACK_MIN)
        expected_timestamps = pd.date_range(
            end=feature_ts,
            periods=FEATURE_LOOKBACK_MIN,
            freq="D",
        )
        if not pd.DatetimeIndex(recent_history["timestamp"]).equals(
            expected_timestamps
        ):
            continue
        feat = _compute_market_features(hist)
        feature_row = feat[feat["timestamp"] == feature_ts]
        if len(feature_row) != 1:
            continue
        fr = feature_row.iloc[-1]
        er = entry_row.iloc[-1]
        quote_volume = fr.get("quote_volume", np.nan)
        if pd.isna(quote_volume):
            quote_volume = float(fr["close"]) * float(fr["volume"])
        numeric_values = np.asarray(
            [
                er["open"],
                quote_volume,
                fr["log_return_1d"],
                fr["roc_7d"],
                fr["atr_pct_14"],
            ],
            dtype=float,
        )
        if (
            not np.isfinite(numeric_values).all()
            or numeric_values[0] <= 0
            or numeric_values[1] <= 0
            or numeric_values[4] < 0
        ):
            continue

        rows.append({
            "market": market,
            "date": decision_ts.normalize().date(),
            "feature_date": feature_ts.normalize().date(),
            "entry_open": float(er["open"]),
            "quote_volume_d1": float(quote_volume),
            "log_return_1d": float(fr["log_return_1d"]),
            "roc_7d": float(fr["roc_7d"]),
            "atr_pct_14": float(fr["atr_pct_14"]),
            "btc_regime": btc_regime,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.replace([np.inf, -np.inf], np.nan)
    out["liq_rank_daily"] = out["quote_volume_d1"].rank(
        method="dense", ascending=False, na_option="bottom"
    )
    out = out[out["liq_rank_daily"] <= top_universe].copy()
    if out.empty:
        return out

    # Discovery used rank-normalized roc_7d. For production we rank over the
    # tradable KRW universe at feature_date.
    out["roc_7d_rank"] = out["roc_7d"].rank(pct=True)
    out["atr_pct_14_rank"] = out["atr_pct_14"].rank(pct=True)
    return out.reset_index(drop=True)


def apply_pump_rules(frame: pd.DataFrame,
                     max_candidates: int = MAX_CANDIDATES) -> pd.DataFrame:
    """Apply mined PUMP rules and return ranked candidate rows."""
    if isinstance(max_candidates, bool) or max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    out["pump20_rule"] = out["roc_7d_rank"] > ROC7_RANK_PUMP20_MIN
    out["pump15_rule"] = (
        (out["atr_pct_14"] > ATR_PCT14_PUMP15_MIN)
        & (out["roc_7d_rank"] > ROC7_RANK_PUMP15_MIN)
        & (out["log_return_1d"] <= LOG_RETURN_1D_PUMP15_MAX)
    )
    out["rule_fire"] = out["pump20_rule"] | out["pump15_rule"]
    out = out[out["rule_fire"]].copy()
    if out.empty:
        return out

    out["estimated_pump20_prob"] = EST_PUMP20_PROB
    out["estimated_pump15_prob"] = np.where(
        out["pump15_rule"], EST_PUMP15_PROB, np.nan
    )
    out["overheated_flag"] = out["log_return_1d"] > LOG_RETURN_1D_PUMP15_MAX
    out["dump_risk_flag"] = out["overheated_flag"]
    out["rule_id"] = np.where(
        out["pump15_rule"],
        "roc7_rank_pump20+pump15_atr_momo",
        "roc7_rank_pump20",
    )
    out["score"] = np.clip(
        0.85 * out["roc_7d_rank"].fillna(0.0)
        + 0.10 * out["atr_pct_14_rank"].fillna(0.5)
        + 0.05 * out["pump15_rule"].astype(float)
        - 0.03 * out["overheated_flag"].astype(float),
        0.0,
        1.0,
    )
    out = out.sort_values(
        ["score", "roc_7d_rank", "atr_pct_14"],
        ascending=[False, False, False],
    ).head(max_candidates)
    out["rank"] = np.arange(1, len(out) + 1)
    return out.reset_index(drop=True)


def score_pump_candidates(asof_date, *, db_path: str = DB_PATH,
                          top_universe: int = UNIVERSE_TOP_N,
                          max_candidates: int = MAX_CANDIDATES,
                          limit_markets: int | None = None) -> dict:
    """Return record-only PUMP hunter candidates for decision date D."""
    asof = pd.Timestamp(asof_date).normalize()
    frame = build_feature_frame(
        asof,
        db_path=db_path,
        top_universe=top_universe,
        limit_markets=limit_markets,
    )
    candidates = apply_pump_rules(frame, max_candidates=max_candidates)
    return {
        "asof": str(asof.date()),
        "feature_date": str((_feature_ts(asof)).normalize().date()),
        "model_id": "pump_hunter",
        "rule_version": "pump_detector_v1",
        "top_universe": int(top_universe),
        "universe_n": int(len(frame)),
        "n_candidates": int(len(candidates)),
        "rules": {
            "pump20": (
                f"roc_7d_rank > {ROC7_RANK_PUMP20_MIN:.3f}; "
                f"mean_lift={PUMP20_RULE_MEAN_LIFT:.2f}"
            ),
            "pump15": (
                f"atr_pct_14 > {ATR_PCT14_PUMP15_MIN:.3f} and "
                f"roc_7d_rank > {ROC7_RANK_PUMP15_MIN:.3f} and "
                f"log_return_1d <= {LOG_RETURN_1D_PUMP15_MAX:.3f}; "
                f"mean_lift={PUMP15_RULE_MEAN_LIFT:.2f}"
            ),
        },
        "candidates": candidates.to_dict(orient="records"),
    }
