"""업비트 KRW 15m 봉 수집기 (Phase C Pre-open Trigger 진짜 08:30 snapshot 검증 용).

설계: collector_1h_upbit.py 와 동일 패턴, interval='minute15'.
용도: 08:15-08:30 closed bar 사용 → 사용자 "08:30 trigger" 가설 직접 검증.

Adapted from: data/collector_1h_upbit.py
Changes:
  - interval=minute15
  - DB: data/upbit_15m.db
  - 페이지 수: 1h 138페이지 → 15m ~525페이지/코인 (3년)
  - 백필 ~4-5시간 예상 (long-running background)

사용:
    python -m data.collector_15m_upbit --coin KRW-BTC --days 1095
    python -m data.collector_15m_upbit --all --days 1095
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyupbit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.database import (  # noqa: E402
    init_db,
    latest_timestamp,
    list_markets,
    oldest_timestamp,
    save_candles,
    stats,
)
from data.collector_d1 import get_krw_markets, get_top_markets


DB_PATH = Path(__file__).resolve().parent / "upbit_15m.db"
INTERVAL = "minute15"
PAGE_SIZE = 200
RETRY_MAX = 3
RETRY_BACKOFF = 1.5
SLEEP_BETWEEN_PAGES = 0.12   # 4h: 0.15. 15m 가 페이지 4배 → 약간 줄임
SLEEP_BETWEEN_MARKETS = 0.25

logger = logging.getLogger("collector_15m")


def _fetch_page(market: str, to: datetime, count: int = PAGE_SIZE) -> pd.DataFrame | None:
    last_err = None
    for attempt in range(1, RETRY_MAX + 1):
        try:
            df = pyupbit.get_ohlcv(
                market, interval=INTERVAL, to=to.strftime("%Y%m%d %H%M%S"), count=count
            )
            if df is None or len(df) == 0:
                return None
            return df
        except Exception as e:
            last_err = e
            sleep_for = RETRY_BACKOFF ** attempt
            logger.warning(f"fetch {market} attempt {attempt}/{RETRY_MAX} fail: {e}; sleep {sleep_for:.1f}s")
            time.sleep(sleep_for)
    logger.error(f"fetch {market} all retries failed: {last_err}")
    return None


def collect_market(market: str, days: int = 365 * 3, db_path: Path = DB_PATH) -> int:
    """단일 마켓 15m 백필. 15m 봉 = 일 96봉, 3년 = ~105k봉 = ~525페이지."""
    init_db(db_path)
    target_oldest = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
    saved = 0

    # incremental update
    latest = latest_timestamp(db_path, market)
    if latest is not None:
        df_recent = _fetch_page(market, datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=1))
        if df_recent is not None and len(df_recent) > 0:
            n = save_candles(db_path, df_recent, market)
            saved += n

    oldest = oldest_timestamp(db_path, market)
    if oldest is None:
        cur_to = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=1)
    else:
        cur_to = oldest

    if cur_to <= target_oldest:
        logger.info(f"{market}: 15m already covers down to {oldest}")
        return saved

    pages = 0
    while cur_to > target_oldest:
        df = _fetch_page(market, cur_to)
        if df is None or len(df) == 0:
            break

        n = save_candles(db_path, df, market)
        saved += n
        pages += 1

        new_oldest = df.index.min().to_pydatetime().replace(tzinfo=None)
        if new_oldest >= cur_to:
            logger.info(f"{market}: 15m reached listing start at {new_oldest}")
            break
        cur_to = new_oldest
        time.sleep(SLEEP_BETWEEN_PAGES)

    logger.info(f"{market}: 15m backfill done, {saved} rows ({pages} pages)")
    return saved


def collect_all(markets: list[str], days: int = 365 * 3, db_path: Path = DB_PATH) -> dict:
    results = {}
    for i, m in enumerate(markets, 1):
        logger.info(f"[{i}/{len(markets)}] 15m {m}")
        try:
            n = collect_market(m, days=days, db_path=db_path)
            results[m] = n
        except Exception as e:
            logger.error(f"{m}: 15m FAIL: {e}")
            results[m] = -1
        time.sleep(SLEEP_BETWEEN_MARKETS)
    return results


def main():
    parser = argparse.ArgumentParser(description="Upbit KRW 15m 봉 수집기")
    parser.add_argument("--coin", type=str, help="단일 코인")
    parser.add_argument("--top", type=int, help="24h 거래대금 top N")
    parser.add_argument("--all", action="store_true", help="KRW 전체 마켓")
    parser.add_argument("--days", type=int, default=365 * 3)
    parser.add_argument("--db", type=str, default=str(DB_PATH))
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    db_path = Path(args.db)

    if args.coin:
        n = collect_market(args.coin, days=args.days, db_path=db_path)
        print(f"Saved {n} rows for {args.coin} (15m)")
    elif args.all:
        markets = get_krw_markets()
        print(f"전체 KRW 15m 백필 {len(markets)}개 시작 (days={args.days})")
        results = collect_all(markets, days=args.days, db_path=db_path)
        ok = sum(1 for v in results.values() if v >= 0)
        fail = sum(1 for v in results.values() if v < 0)
        total = sum(v for v in results.values() if v >= 0)
        print(f"\n=== 15m Done: OK {ok} / FAIL {fail} / Total {total} rows ===")
    elif args.top:
        markets = get_top_markets(args.top)
        results = collect_all(markets, days=args.days, db_path=db_path)
        print(f"15m Done: {len(results)} markets")
    else:
        parser.print_help()
        return

    print("\n=== 15m DB stats ===")
    print(stats(db_path).to_string(index=False))


if __name__ == "__main__":
    main()
