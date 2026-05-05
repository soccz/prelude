"""가상 포지션 진입 / 익절 / 손절 / 청산 시뮬 (LEDGER §3 — 옵션 3).

설계:
  - 09:00 KST 시가 매수 (entry)
  - 4h 봉 단위 시뮬 — 매 봉 high/low 체크
  - high ≥ entry × (1 + TP_pct) → 즉시 익절 (그 봉 시점)
  - low ≤ entry × (1 - SL_pct) → 즉시 손절 (그 봉 시점)
  - 같은 4h 봉에 둘 다 트리거 → SL 우선 (보수, config.SL_PRIORITY)
  - 둘 다 안 도달 → max_hold 후 다음 09:00 종가 청산
  - 거래비용 왕복 차감

핵심 함수:
  - simulate_position(market, entry_date, db_4h_path, tp_pct, sl_pct, max_hold_hours)
  - simulate_batch(positions, db_4h_path) → 모든 가상 포지션 결과
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.database import load_candles
from ledger.config import (
    MAX_HOLD_HOURS,
    ROUND_TRIP_COST_PCT,
    SL_PCT,
    SL_PRIORITY,
    TP_PCT,
)


# ============================================================================
# 단일 포지션 시뮬
# ============================================================================
@dataclass
class PositionResult:
    market: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    exit_type: str            # 'tp' / 'sl' / 'eod'
    hold_hours: float
    gross_return_pct: float
    cost_pct: float
    net_return_pct: float


def _next_kst_9am(dt: pd.Timestamp) -> pd.Timestamp:
    """다음 KST 09:00 시각 (24h 후)."""
    return dt + pd.Timedelta(hours=24)


def simulate_d1_simple(
    open_price: float,
    high_price: float,
    low_price: float,
    close_price: float,
    tp_pct: float = TP_PCT,
    sl_pct: float = SL_PCT,
    cost_pct: float = ROUND_TRIP_COST_PCT,
    sl_priority: bool = SL_PRIORITY,
) -> dict:
    """
    빠른 일봉 단순 시뮬 (4h 봉 안 씀, backtest 대량 시뮬용).

    근사:
      - entry = open
      - TP_hit if high >= entry × (1+tp_pct)
      - SL_hit if low  <= entry × (1-sl_pct)
      - 둘 다 시 → SL 우선 (보수)
      - 둘 다 안 도달 → close 청산
      - hold 시간 정보 없음 (eod 가정)

    return: { exit_type, gross_return_pct, net_return_pct }
    """
    if open_price <= 0:
        return {"exit_type": "skip", "gross_return_pct": 0.0, "net_return_pct": 0.0}

    tp_price = open_price * (1 + tp_pct)
    sl_price = open_price * (1 - sl_pct)
    tp_hit = high_price >= tp_price
    sl_hit = low_price <= sl_price

    if tp_hit and sl_hit:
        if sl_priority:
            exit_type = "sl"
            exit_price = sl_price
        else:
            exit_type = "tp"
            exit_price = tp_price
    elif tp_hit:
        exit_type = "tp"
        exit_price = tp_price
    elif sl_hit:
        exit_type = "sl"
        exit_price = sl_price
    else:
        exit_type = "eod"
        exit_price = close_price

    gross = (exit_price - open_price) / open_price
    return {
        "exit_type": exit_type,
        "gross_return_pct": gross,
        "net_return_pct": gross - cost_pct,
    }


def simulate_position(
    market: str,
    entry_date: pd.Timestamp,
    db_4h_path: str | Path = "data/upbit_4h.db",
    db_d1_path: str | Path = "data/upbit_d1.db",
    tp_pct: float = TP_PCT,
    sl_pct: float = SL_PCT,
    max_hold_hours: int = MAX_HOLD_HOURS,
    cost_pct: float = ROUND_TRIP_COST_PCT,
    sl_priority: bool = SL_PRIORITY,
) -> Optional[PositionResult]:
    """
    한 코인 한 일봉 동안 가상 진입/청산 시뮬.

    entry_date: KST 09:00 (그 일봉 시작 시각)
    end_date  : entry_date + 24h
    """
    entry_date = pd.to_datetime(entry_date)
    end_date = entry_date + pd.Timedelta(hours=max_hold_hours)

    # entry_price = 일봉 시가 (entry_date 의 d1 봉)
    df_d1 = load_candles(db_d1_path, market, since=entry_date, until=entry_date + pd.Timedelta(days=1))
    if len(df_d1) == 0:
        return None
    entry_row = df_d1[df_d1["timestamp"] == entry_date]
    if len(entry_row) == 0:
        # 정확한 KST 09:00 매칭 안 되면 가장 가까운 첫 봉
        entry_row = df_d1.iloc[[0]]
    entry_price = float(entry_row["open"].iloc[0])

    if entry_price <= 0:
        return None

    # 4h 봉 시뮬 (entry_date ~ end_date)
    df_4h = load_candles(db_4h_path, market, since=entry_date, until=end_date + pd.Timedelta(hours=1))
    if len(df_4h) == 0:
        # 4h 데이터 없으면 일봉 hold (보수: 종가 청산)
        if len(df_d1) > 0:
            close_price = float(df_d1["close"].iloc[0])
            gross = (close_price - entry_price) / entry_price
            return PositionResult(
                market=market, entry_date=entry_date, entry_price=entry_price,
                exit_date=entry_date + pd.Timedelta(hours=24), exit_price=close_price,
                exit_type="eod", hold_hours=24.0,
                gross_return_pct=gross, cost_pct=cost_pct, net_return_pct=gross - cost_pct,
            )
        return None

    tp_price = entry_price * (1 + tp_pct)
    sl_price = entry_price * (1 - sl_pct)

    # 매 봉 체크
    for _, row in df_4h.iterrows():
        bar_time = pd.to_datetime(row["timestamp"])
        if bar_time < entry_date or bar_time > end_date:
            continue

        h = float(row["high"])
        l = float(row["low"])

        tp_hit = h >= tp_price
        sl_hit = l <= sl_price

        if tp_hit and sl_hit:
            # 같은 봉에 둘 다 → SL 우선 (보수)
            if sl_priority:
                exit_type = "sl"
                exit_price = sl_price
            else:
                exit_type = "tp"
                exit_price = tp_price
        elif tp_hit:
            exit_type = "tp"
            exit_price = tp_price
        elif sl_hit:
            exit_type = "sl"
            exit_price = sl_price
        else:
            continue

        hold_hours = (bar_time - entry_date).total_seconds() / 3600
        gross = (exit_price - entry_price) / entry_price
        return PositionResult(
            market=market, entry_date=entry_date, entry_price=entry_price,
            exit_date=bar_time, exit_price=exit_price,
            exit_type=exit_type, hold_hours=hold_hours,
            gross_return_pct=gross, cost_pct=cost_pct, net_return_pct=gross - cost_pct,
        )

    # TP/SL 둘 다 안 도달 → 다음 09:00 종가 청산
    next_d1 = load_candles(
        db_d1_path, market, since=end_date - pd.Timedelta(hours=1), until=end_date + pd.Timedelta(days=1)
    )
    if len(next_d1) == 0:
        # 데이터 부족 → 마지막 4h 봉 종가
        last_4h = df_4h.iloc[-1]
        exit_price = float(last_4h["close"])
        exit_date = pd.to_datetime(last_4h["timestamp"])
    else:
        # 그 일봉 종가 = 다음 봉 시가 (어차피 같음 — d1 봉의 close)
        exit_row = next_d1.iloc[0]
        exit_price = float(exit_row["close"])
        exit_date = pd.to_datetime(exit_row["timestamp"])

    gross = (exit_price - entry_price) / entry_price
    return PositionResult(
        market=market, entry_date=entry_date, entry_price=entry_price,
        exit_date=exit_date, exit_price=exit_price,
        exit_type="eod", hold_hours=float(max_hold_hours),
        gross_return_pct=gross, cost_pct=cost_pct, net_return_pct=gross - cost_pct,
    )


# ============================================================================
# Batch (recommendations DataFrame → results DataFrame)
# ============================================================================
def simulate_batch(
    recommendations: pd.DataFrame,
    db_4h_path: str | Path = "data/upbit_4h.db",
    db_d1_path: str | Path = "data/upbit_d1.db",
    tp_pct: float = TP_PCT,
    sl_pct: float = SL_PCT,
) -> pd.DataFrame:
    """
    recommendations: { coin, entry_date, position_size_pct, ... }
    return: 모든 포지션 결과 DataFrame
    """
    results = []
    for _, row in recommendations.iterrows():
        market = row["coin"]
        entry_date = pd.to_datetime(row["entry_date"])
        try:
            res = simulate_position(market, entry_date, db_4h_path, db_d1_path, tp_pct, sl_pct)
            if res is None:
                continue
            d = {
                "coin": res.market,
                "entry_date": res.entry_date,
                "entry_price": res.entry_price,
                "exit_date": res.exit_date,
                "exit_price": res.exit_price,
                "exit_type": res.exit_type,
                "hold_hours": res.hold_hours,
                "gross_return_pct": res.gross_return_pct,
                "cost_pct": res.cost_pct,
                "net_return_pct": res.net_return_pct,
                "position_size_pct": row.get("position_size_pct", 0),
            }
            # 시그널 메타 보존
            for k in ["p_ge_5", "p_ge_10", "p_ge_15", "p_ge_20", "expected_max", "btc_regime"]:
                if k in row:
                    d[f"signal_{k}"] = row[k]
            results.append(d)
        except Exception as e:
            print(f"  simulate {market} {entry_date}: {e}")

    return pd.DataFrame(results)


# ============================================================================
# 직접 실행 — smoke test
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--coin", default="KRW-BTC")
    parser.add_argument("--date", default=None, help="entry KST 09:00 (YYYY-MM-DD)")
    parser.add_argument("--tp", type=float, default=TP_PCT)
    parser.add_argument("--sl", type=float, default=SL_PCT)
    args = parser.parse_args()

    # 기본: 일주일 전
    entry_dt = (
        pd.Timestamp(args.date)
        if args.date
        else pd.Timestamp.now().normalize() - pd.Timedelta(days=7)
    )
    entry_dt = entry_dt.replace(hour=9)

    print(f"Simulating {args.coin} entry {entry_dt} TP={args.tp:.0%} SL={args.sl:.0%}")
    res = simulate_position(args.coin, entry_dt, tp_pct=args.tp, sl_pct=args.sl)
    if res is None:
        print("(no result)")
    else:
        for k, v in res.__dict__.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}" + (" pct" if k.endswith("pct") else ""))
            else:
                print(f"  {k}: {v}")
