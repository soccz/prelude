"""일일 추론 — 매일 KST 08:30 cron 호출용.

설계 (SIGNAL.md §6):
  - 어제까지 데이터로 오늘 일봉 분포 예측
  - 출력: { 코인, P(≥+5/10/15/20%), 기대 max, CI, BTC regime }
  - 추천 universe: 업비트 KRW only (예측 시 KRW-* prefix filter)
  - 가능하면 calibration 적용 (isotonic)
  - output/predictions_YYYYMMDD.csv

사용:
    python -m signals.predict --date 2026-05-03 --dry-run
    python -m signals.predict                                 # today
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.database import latest_timestamp, list_markets, load_candles
from signals.calibration import IsotonicCalibrator
from signals.features import assemble_training_panel
from signals.labels import (
    DEFAULT_BIN_CENTERS,
    DEFAULT_BINS,
    approx_ci_from_bins,
    cumulative_probs,
    expected_max_return,
)
from signals.models.xgb_phase1 import EXCLUDE_COLS, XGBPhase1

logger = logging.getLogger("predict")

# ============================================================================
# 추천 universe filter
# ============================================================================
def is_tradable_market(market: str) -> bool:
    """추천 가능 universe = 업비트 KRW only (실거래 가능)."""
    return market.startswith("KRW-")


# ============================================================================
# 일일 추론
# ============================================================================
@dataclass
class Prediction:
    coin: str
    btc_regime: str
    bin_probs: list           # raw bin probs (6,)
    p_ge_5: float
    p_ge_10: float
    p_ge_15: float
    p_ge_20: float
    expected_max: float
    ci_low: float
    ci_high: float


def predict_today(
    model: XGBPhase1,
    upbit_db: str | Path = "data/upbit_d1.db",
    binance_db: Optional[str | Path] = "data/binance_d1.db",
    btc_market: str = "KRW-BTC",
    asof: Optional[pd.Timestamp] = None,
    calibrator: Optional[IsotonicCalibrator] = None,
    only_tradable: bool = True,
) -> pd.DataFrame:
    """
    매일 KST 08:30 cron 추론.

    asof: 추론 기준 시점 (None=now). 어제 마감된 일봉이 마지막 입력.
    only_tradable: True 면 KRW-* 만 추천 candidate (학습은 BINANCE 도 사용 가능).
    """
    if asof is None:
        asof = pd.Timestamp.now()

    # 1. 데이터 로드
    krw_markets = list_markets(upbit_db)
    candles = {m: load_candles(upbit_db, m, until=asof) for m in krw_markets}
    candles = {k: v for k, v in candles.items() if len(v) >= 30}

    if binance_db and Path(binance_db).exists():
        bn_markets = list_markets(binance_db)
        for m in bn_markets:
            d = load_candles(binance_db, m, until=asof)
            if len(d) >= 30:
                candles[m] = d

    btc = load_candles(upbit_db, btc_market, until=asof)
    logger.info(f"loaded {len(candles)} markets, BTC {len(btc)} rows")

    # 2. Panel
    panel = assemble_training_panel(candles, btc, normalize=True)
    if len(panel) == 0:
        return pd.DataFrame()

    # 3. 추론 시점 = 각 코인의 가장 최근 row
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    latest = panel.sort_values("timestamp").groupby("market").tail(1)

    if only_tradable:
        latest = latest[latest["market"].apply(is_tradable_market)]
    if len(latest) == 0:
        return pd.DataFrame()

    # 4. 피처 추출
    if model.feature_names is None:
        feature_cols = [c for c in latest.columns if c not in EXCLUDE_COLS]
    else:
        feature_cols = model.feature_names

    X = latest[feature_cols].astype(float).values

    # 5. 예측
    bin_probs = model.predict_proba(X)
    cum = cumulative_probs(bin_probs)
    expected = expected_max_return(bin_probs)
    ci_low, ci_high = approx_ci_from_bins(bin_probs)

    # 6. (옵션) calibration 적용
    if calibrator is not None:
        cum_calibrated = calibrator.transform(bin_probs)
        cum = cum_calibrated

    # 7. 출력 DataFrame
    out = pd.DataFrame({
        "coin": latest["market"].values,
        "btc_regime": latest.get("btc_regime", pd.Series(["unknown"] * len(latest))).values,
        "bin_probs": [list(map(float, p)) for p in bin_probs],
        "p_ge_5": cum["p_ge_5"],
        "p_ge_10": cum["p_ge_10"],
        "p_ge_15": cum["p_ge_15"],
        "p_ge_20": cum["p_ge_20"],
        "expected_max": expected,
        "ci_low": ci_low,
        "ci_high": ci_high,
    })

    out = out.sort_values("p_ge_10", ascending=False).reset_index(drop=True)
    return out


# ============================================================================
# Top-K 필터 (LEDGER 가 받음)
# ============================================================================
def top_k(predictions: pd.DataFrame, k: int = 3, min_p_ge_5: float = 0.5) -> pd.DataFrame:
    """top-K 추천 (P(≥5%) >= 임계 + P(≥10%) 정렬 기준)."""
    candidates = predictions[predictions["p_ge_5"] >= min_p_ge_5]
    return candidates.head(k)


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="signals/models/ckpt/phase1_smoke.json")
    parser.add_argument("--config", default="signals/models/configs/phase1_smoke.json")
    parser.add_argument("--upbit-db", default="data/upbit_d1.db")
    parser.add_argument("--binance-db", default="data/binance_d1.db")
    parser.add_argument("--asof", type=str, help="기준 시점 (YYYY-MM-DD), default=now")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="저장 X, 출력만")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # 모델 로드
    if not Path(args.ckpt).exists():
        print(f"ERROR: model checkpoint not found: {args.ckpt}")
        print("    먼저 학습: python -m signals.models.xgb_phase1")
        sys.exit(1)

    print(f"loading model: {args.ckpt}")
    model = XGBPhase1.load(args.ckpt, args.config)

    asof = pd.Timestamp(args.asof) if args.asof else None

    print(f"predicting (asof={asof or 'now'})...")
    preds = predict_today(model, upbit_db=args.upbit_db, binance_db=args.binance_db, asof=asof)

    if len(preds) == 0:
        print("(no predictions)")
        return

    print(f"\n=== Top {args.top} ===")
    show_cols = ["coin", "btc_regime", "p_ge_5", "p_ge_10", "p_ge_15", "p_ge_20",
                 "expected_max", "ci_low", "ci_high"]
    print(preds[show_cols].head(args.top).to_string(index=False, float_format=lambda x: f"{x:6.3f}"))

    if not args.dry_run:
        date_str = (asof or pd.Timestamp.now()).strftime("%Y%m%d")
        out_path = Path(f"output/predictions_{date_str}.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        preds.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
