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

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.database import load_candles
from data.market_universe import is_excluded_signal_market
from signals.pump_detector_v1 import (
    UNIVERSE_TOP_N,
    build_feature_frame,
)

log = logging.getLogger(__name__)

BINANCE_DB = str(Path(__file__).resolve().parent.parent / "data" / "binance_d1.db")
# 상시 상장 앵커: 이 심볼의 f_day 봉 + 연속 lookback 이력이 없으면 DB 자체가
# 유실/재구축/수집장애 상태다 — 구조적 제외 판정(0 rows = 미상장 등)의 전제.
BN_ANCHOR = "BINANCE-BTCUSDT"
# 개별 심볼만 장기 stale = 상폐/거래정지 추정 임계 (초기값; 앵커 신선이 전제.
# 수집기는 심볼 실패 시 run exit code 가 실패로 승격돼 그날 시끄럽게 알려지므로
# grace 를 넘긴 개별 stale 은 상장 이탈 신호로 본다)
DELIST_GRACE_DAYS = 7

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

_BINANCE_FEATURE_COLUMNS = ["bn_market", "b_vol_surge", "b_ret_1d"]


def krw_to_binance(market: str) -> str | None:
    market = str(market)
    if (
        is_excluded_signal_market(market)
        or re.fullmatch(r"KRW-[A-Z0-9]+", market) is None
    ):
        return None
    base = market.removeprefix("KRW-")
    return f"BINANCE-{base}USDT"


def binance_volsurge_for_date(feature_date, needed_bn: set[str],
                              binance_db: str = BINANCE_DB) -> tuple[pd.DataFrame, str]:
    """필요한 모든 Binance symbol의 D-1 vol-surge를 원자적으로 만든다.

    symbol 하나라도 feature_date candle 또는 직전 연속 20일 history가 없으면
    부분 frame을 반환하지 않는다. inner join으로 누락 자산만 조용히 제거되는 선택편향을
    막기 위한 fail-closed 계약이다.

    단, 구조적 미상장은 수집 장애가 아니다 (2026-07-28 정정): 일일 수집기가
    live active USDT 상장 목록 기준 --all 로 돌므로 DB 0 rows = Binance 미상장
    (업비트 단독 상장), 그리고 전체 이력이 lookback 창 이후에 시작하면 신규
    상장이다. 이 둘은 제외 목록으로 분리해 나머지 ready 심볼로 진행한다 —
    07-26 강화가 이 구분 없이 원자 fail-closed 를 적용해 roc 상위에 업비트
    단독 코인 1개만 섞여도 후보 생산이 영구 0이 되던 회귀의 수리. 기존 상장
    심볼의 stale/gap/read_error 는 종전대로 전량 fail-closed.
    """
    feature_timestamp = pd.Timestamp(feature_date)
    if pd.isna(feature_timestamp):
        raise ValueError("feature_date must be a valid timestamp")
    if feature_timestamp.tzinfo is not None:
        feature_timestamp = feature_timestamp.tz_convert(
            "Asia/Seoul"
        ).tz_localize(None)
    f_day = feature_timestamp.normalize()
    empty = pd.DataFrame(columns=_BINANCE_FEATURE_COLUMNS)
    empty.attrs["binance_excluded"] = []
    if not needed_bn:
        return empty, "ok"

    lookback_start = f_day - pd.Timedelta(days=BN_VOL_LOOKBACK)

    # 앵커 프로브 — DB 유실 후 빈 스키마 자동생성, --days 3 증분만으로 재구축된
    # DB 등에서 전 심볼이 not_listed/newly_listed 로 오분류되어 "ok 무추천일"로
    # 위장되는 fail-open 을 차단한다. 앵커가 불완전하면 무조건 fail-closed.
    try:
        anchor_raw = load_candles(binance_db, BN_ANCHOR)
    except Exception as exc:
        return empty, f"binance_anchor_unreadable[{type(exc).__name__}]"
    anchor_ok = False
    if anchor_raw is not None and not anchor_raw.empty:
        anchor = anchor_raw.sort_values("timestamp").copy()
        anchor["day"] = pd.to_datetime(anchor["timestamp"]).dt.normalize()
        anchor_through = anchor[anchor["day"] <= f_day]
        anchor_history = anchor_through[
            anchor_through["day"] < f_day
        ].tail(BN_VOL_LOOKBACK)
        anchor_ok = (
            int((anchor_through["day"] == f_day).sum()) == 1
            and len(anchor_history) == BN_VOL_LOOKBACK
            and pd.DatetimeIndex(anchor_history["day"]).equals(
                pd.date_range(
                    lookback_start,
                    f_day - pd.Timedelta(days=1),
                    freq="D",
                )
            )
        )
    if not anchor_ok:
        return empty, (
            f"binance_anchor_incomplete (need {BN_ANCHOR} candle at "
            f"{f_day.date()} + {BN_VOL_LOOKBACK}d continuous history — "
            "DB loss/rebuild/collector outage suspected)"
        )

    rows: list[dict] = []
    issues: list[str] = []
    excluded: list[str] = []
    for bm in sorted(needed_bn):
        try:
            raw = load_candles(binance_db, bm)
        except Exception as exc:
            issues.append(f"{bm}:read_error[{type(exc).__name__}]")
            continue
        if raw is None or raw.empty:
            excluded.append(f"{bm}:not_listed")
            continue

        d = raw.sort_values("timestamp").copy()
        d["timestamp"] = pd.to_datetime(d["timestamp"])
        d["day"] = d["timestamp"].dt.normalize()
        through_feature = d[d["day"] <= f_day]
        feature_rows = through_feature[through_feature["day"] == f_day]
        if len(feature_rows) != 1:
            first_day = d["day"].min()
            if first_day > f_day:
                # 이력 전체가 f_day 이후 시작 = 그 시점엔 Binance 미상장.
                excluded.append(
                    f"{bm}:not_listed_asof[first={first_day.date()}]"
                )
                continue
            last_day = (
                through_feature["day"].max()
                if not through_feature.empty
                else None
            )
            if (
                last_day is not None
                and last_day
                < f_day - pd.Timedelta(days=DELIST_GRACE_DAYS)
            ):
                # 앵커 신선이 위에서 보증된 상태에서 개별 심볼만 장기 stale
                # = 상폐/거래정지 (예: ARDRUSDT 2026-07-10 정지) — 구조적 제외.
                excluded.append(
                    f"{bm}:delisted_asof[last={last_day.date()}]"
                )
                continue
            latest = (
                through_feature["day"].max().date()
                if not through_feature.empty
                else "none"
            )
            issues.append(
                f"{bm}:feature_date_missing[latest={latest},need={f_day.date()}]"
            )
            continue

        history = through_feature[through_feature["day"] < f_day].tail(
            BN_VOL_LOOKBACK
        )
        if len(history) != BN_VOL_LOOKBACK:
            first_day = d["day"].min()
            if first_day > lookback_start:
                excluded.append(
                    f"{bm}:newly_listed[first={first_day.date()}]"
                )
                continue
            issues.append(
                f"{bm}:history_short[{len(history)}/{BN_VOL_LOOKBACK}]"
            )
            continue

        expected_days = pd.date_range(
            f_day - pd.Timedelta(days=BN_VOL_LOOKBACK),
            f_day - pd.Timedelta(days=1),
            freq="D",
        )
        actual_days = pd.DatetimeIndex(history["day"])
        if not actual_days.equals(expected_days):
            issues.append(f"{bm}:history_gap")
            continue

        last = feature_rows.iloc[0]
        baseline_qv = pd.to_numeric(
            history["quote_volume"], errors="coerce"
        ).to_numpy(dtype=float)
        current_qv = float(
            pd.to_numeric(
                pd.Series([last["quote_volume"]]), errors="coerce"
            ).iloc[0]
        )
        base = float(np.mean(baseline_qv))
        if (
            not np.isfinite(baseline_qv).all()
            or not np.isfinite(current_qv)
            or not np.isfinite(base)
            or base <= 0
            or current_qv < 0
            or (baseline_qv < 0).any()
        ):
            issues.append(f"{bm}:invalid_quote_volume")
            continue

        close_pair = pd.to_numeric(
            pd.Series([history["close"].iloc[-1], last["close"]]),
            errors="coerce",
        ).to_numpy(dtype=float)
        previous_close, current_close = close_pair
        rows.append({
            "bn_market": bm,
            "b_vol_surge": float(current_qv / base),
            "b_ret_1d": (
                float(np.log(current_close / previous_close))
                if (
                    np.isfinite(current_close)
                    and np.isfinite(previous_close)
                    and current_close > 0
                    and previous_close > 0
                )
                else np.nan
            ),
        })

    if issues:
        shown = ",".join(issues[:8])
        if len(issues) > 8:
            shown += f",+{len(issues) - 8}"
        if excluded:
            shown += f"; excluded={len(excluded)}"
        frame = empty.copy()
        frame.attrs["binance_excluded"] = sorted(excluded)
        frame.attrs["binance_ready_n"] = len(rows)
        return (
            frame,
            f"binance_partial (ready={len(rows)}/{len(needed_bn)}; {shown})",
        )
    if not rows:
        # 전량 구조적 제외를 "ok 무추천일"로 위장하지 않는다 — 룰이 정상적으로
        # 안 걸린 날(healthy zero-pick)과 판정 자체가 불가능했던 날은 다르다.
        frame = empty.copy()
        frame.attrs["binance_excluded"] = sorted(excluded)
        frame.attrs["binance_ready_n"] = 0
        return (
            frame,
            f"binance_no_ready (excluded={len(excluded)}/{len(needed_bn)})",
        )
    if excluded:
        # status 는 소비자 계약("ok" 정확 일치: 후보 검증·알림 문구·stale 판정)
        # 을 지키기 위해 그대로 두고, 구조적 제외는 decision meta(caller)와
        # 로그로 감사 가시화한다.
        log.warning(
            "binance volsurge structural exclusions %d/%d (proceeding with "
            "ready=%d): %s",
            len(excluded),
            len(needed_bn),
            len(rows),
            ",".join(excluded[:8])
            + (f",+{len(excluded) - 8}" if len(excluded) > 8 else ""),
        )
    frame = pd.DataFrame(rows, columns=_BINANCE_FEATURE_COLUMNS)
    frame.attrs["binance_excluded"] = sorted(excluded)
    frame.attrs["binance_ready_n"] = len(rows)
    return frame, "ok"


def score_pump_v2_candidates(asof_date, *, db_path: str | None = None,
                             binance_db: str = BINANCE_DB,
                             top_universe: int = UNIVERSE_TOP_N,
                             max_candidates: int = MAX_CANDIDATES,
                             limit_markets: int | None = None) -> dict:
    """결정일 D 의 v2 후보 (record-only scoring). 텔레그램/원장 기록 없음."""
    if top_universe <= 0:
        raise ValueError("top_universe must be positive")
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    if limit_markets is not None and limit_markets <= 0:
        raise ValueError("limit_markets must be positive when provided")
    asof = pd.Timestamp(asof_date).normalize()
    kwargs: dict[str, Any] = {
        "top_universe": top_universe,
        "limit_markets": limit_markets,
    }
    if db_path:
        kwargs["db_path"] = db_path
    frame = build_feature_frame(asof, **kwargs)
    if not frame.empty:
        if "market" not in frame.columns:
            raise ValueError("v2 Upbit feature frame is missing market identity")
        excluded_rows = sorted({
            str(market)
            for market in frame["market"]
            if is_excluded_signal_market(str(market))
        })
        if excluded_rows:
            raise RuntimeError(
                "v2 upstream universe contains excluded signal market(s): "
                + ",".join(excluded_rows)
            )

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
    # 구조적 제외를 decision manifest 에 박제 — ready=15/15 였던 날과 8/15 였던
    # 날을 사후 구분할 수 있어야 한다 (로그는 로테이트되지만 manifest 는 불변).
    meta["binance_needed_n"] = int(len(needed))
    meta["binance_ready_n"] = int(bn.attrs.get("binance_ready_n", len(bn)))
    meta["binance_excluded"] = list(bn.attrs.get("binance_excluded", []))
    if status != "ok" or bn.empty:
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
