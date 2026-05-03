"""Preflight 게이트 — 추론 전 데이터 / 모델 / calibration 위생 체크.

OPS §2 — 통과 못 하면 안전 모드 (가상 진입 X, 텔레그램에는 'preflight 실패' 알림).

체크 항목 (모든 임계값 placeholder, CLAUDE.md §2.5):
  - freshness: 어제 일봉 timestamp 가 어제 KST 09:00 마감 ± 30분 안인가
  - NaN 비율: 피처 NaN ≤ 20% per-coin
  - universe churn: 오늘 universe vs 어제 universe 30% 이상 안 바뀌었는가
  - 모델 ckpt 존재
  - calibration 30일 이내 갱신
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.database import latest_timestamp, list_markets, load_candles


# ============================================================================
# 임계값 (placeholder)
# ============================================================================
FRESHNESS_MAX_LAG_MINUTES = 30
NAN_THRESHOLD_PCT = 0.20
UNIVERSE_CHURN_THRESHOLD = 0.30
CALIBRATION_MAX_AGE_DAYS = 30


# ============================================================================
# 결과 dataclass
# ============================================================================
@dataclass
class PreflightResult:
    passed: bool
    failures: list           # ['freshness', 'nan_high', ...]
    details: dict

    def __bool__(self):
        return self.passed


# ============================================================================
# 개별 체크
# ============================================================================
def check_freshness(
    db_path: str | Path,
    btc_market: str = "KRW-BTC",
    max_lag_minutes: int = FRESHNESS_MAX_LAG_MINUTES,
    asof: Optional[pd.Timestamp] = None,
) -> tuple[bool, dict]:
    """어제 일봉이 KST 09:00 마감 후 max_lag_minutes 이내 갱신됐나."""
    if asof is None:
        asof = pd.Timestamp.now()

    latest = latest_timestamp(db_path, btc_market)
    if latest is None:
        return False, {"reason": "no_data", "market": btc_market}

    # 어제 KST 09:00 = asof.normalize() - 0 if asof.hour < 9 else +0
    yesterday_close = asof.normalize().replace(hour=9)
    if asof.hour < 9:
        yesterday_close -= pd.Timedelta(days=1)

    # latest 가 yesterday_close (= 어제 KST 09:00 시작 = 그제 일봉 마감) 이상이어야
    # 사실 latest 는 어제 일봉의 timestamp = 어제 KST 09:00:00
    expected_latest = yesterday_close
    lag = (asof - latest).total_seconds() / 60

    return abs(lag - 24*60) <= max_lag_minutes or latest >= expected_latest, {
        "latest": str(latest),
        "expected_latest": str(expected_latest),
        "lag_minutes": float(lag),
        "max_lag_minutes": max_lag_minutes,
    }


def check_nan_ratio(
    panel: pd.DataFrame,
    threshold: float = NAN_THRESHOLD_PCT,
    feature_cols: Optional[list] = None,
) -> tuple[bool, dict]:
    """panel 의 피처 NaN 비율 체크."""
    if len(panel) == 0:
        return False, {"reason": "empty_panel"}

    if feature_cols is None:
        feature_cols = [
            c for c in panel.columns
            if c.startswith(("return_", "vol_", "range_", "btc_", "rsi", "macd", "bb_"))
        ]
    feature_cols = [c for c in feature_cols if c in panel.columns]

    if not feature_cols:
        return True, {"reason": "no_feature_cols_to_check"}

    nan_ratio = panel[feature_cols].isna().mean().mean()
    return nan_ratio <= threshold, {
        "nan_ratio": float(nan_ratio),
        "threshold": threshold,
        "n_features_checked": len(feature_cols),
    }


def check_universe_churn(
    today_universe: list[str],
    yesterday_universe: list[str],
    threshold: float = UNIVERSE_CHURN_THRESHOLD,
) -> tuple[bool, dict]:
    """오늘 vs 어제 universe 변경률."""
    if not yesterday_universe:
        return True, {"reason": "first_run"}
    today_set = set(today_universe)
    yest_set = set(yesterday_universe)
    union = today_set | yest_set
    if not union:
        return False, {"reason": "empty_universe"}
    diff = today_set.symmetric_difference(yest_set)
    churn = len(diff) / len(union)
    return churn <= threshold, {
        "churn_ratio": float(churn),
        "threshold": threshold,
        "today_n": len(today_set),
        "yesterday_n": len(yest_set),
        "added": list(today_set - yest_set)[:5],
        "removed": list(yest_set - today_set)[:5],
    }


def check_model_exists(ckpt_path: str | Path) -> tuple[bool, dict]:
    """모델 ckpt 파일 존재."""
    p = Path(ckpt_path)
    return p.exists(), {"path": str(p), "exists": p.exists()}


def check_calibration_age(
    cal_path: str | Path,
    max_age_days: int = CALIBRATION_MAX_AGE_DAYS,
    asof: Optional[datetime] = None,
) -> tuple[bool, dict]:
    """calibration json 파일이 max_age_days 이내 갱신."""
    asof = asof or datetime.now()
    p = Path(cal_path)
    if not p.exists():
        return False, {"reason": "no_calibration_file", "path": str(p)}
    mtime = datetime.fromtimestamp(p.stat().st_mtime)
    age_days = (asof - mtime).days
    return age_days <= max_age_days, {
        "calibration_path": str(p),
        "age_days": age_days,
        "max_age_days": max_age_days,
    }


# ============================================================================
# 종합 preflight
# ============================================================================
def run_preflight(
    db_path: str | Path = "data/upbit_d1.db",
    panel: Optional[pd.DataFrame] = None,
    today_universe: Optional[list[str]] = None,
    yesterday_universe: Optional[list[str]] = None,
    ckpt_path: Optional[str | Path] = None,
    cal_path: Optional[str | Path] = None,
    asof: Optional[pd.Timestamp] = None,
) -> PreflightResult:
    """
    종합 preflight 체크 (OPS §2.1).
    """
    failures = []
    details = {}

    # 1. freshness
    ok, info = check_freshness(db_path, asof=asof)
    details["freshness"] = info
    if not ok:
        failures.append("freshness")

    # 2. NaN
    if panel is not None:
        ok, info = check_nan_ratio(panel)
        details["nan"] = info
        if not ok:
            failures.append("nan_high")

    # 3. universe churn
    if today_universe is not None and yesterday_universe is not None:
        ok, info = check_universe_churn(today_universe, yesterday_universe)
        details["universe_churn"] = info
        if not ok:
            failures.append("universe_churn")

    # 4. ckpt
    if ckpt_path:
        ok, info = check_model_exists(ckpt_path)
        details["ckpt"] = info
        if not ok:
            failures.append("ckpt_missing")

    # 5. calibration
    if cal_path:
        ok, info = check_calibration_age(cal_path, asof=asof)
        details["calibration"] = info
        if not ok:
            failures.append("calibration_stale")

    return PreflightResult(
        passed=(len(failures) == 0),
        failures=failures,
        details=details,
    )
