"""업비트 KRW 4 시간 봉 수집기.

설계: collector_d1.py 와 동일 패턴, interval='minute240' (= 4h).
용도: 일봉 안의 장중 max(high) / max drawdown 정확 측정 (LEDGER §3 익절/손절 시뮬).

Adapted from: data/collector_d1.py
Changes:
  - interval=240 (4h)
  - DB: data/upbit_4h.db
  - 일봉 6 페이지 vs 4h 33 페이지 — 더 오래 걸림

사용:
    python -m data.collector_4h --coin KRW-BTC --days 365
    python -m data.collector_4h --all --days 1095
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
from data.collector_d1 import get_krw_markets, get_top_markets  # 마켓 목록 재사용

# ============================================================================
# 설정
# ============================================================================
DB_PATH = Path(__file__).resolve().parent / "upbit_4h.db"
INTERVAL = "minute240"  # 4시간 봉
PAGE_SIZE = 200
RETRY_MAX = 3
RETRY_BACKOFF = 1.5
SLEEP_BETWEEN_PAGES = 0.15
SLEEP_BETWEEN_MARKETS = 0.3

logger = logging.getLogger("collector_4h")


# ============================================================================
# 백필
# ============================================================================
def _fetch_page(market: str, to: datetime, count: int = PAGE_SIZE) -> pd.DataFrame | None:
    """한 페이지 (200 봉 = 33일치) 수집."""
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
    """단일 마켓 4h 백필. 4h 봉 = 일 6봉, 3년 = ~6500봉 = 33페이지."""
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

    # 과거 backfill
    oldest = oldest_timestamp(db_path, market)
    if oldest is None:
        cur_to = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=1)
    else:
        cur_to = oldest

    if cur_to <= target_oldest:
        logger.info(f"{market}: 4h already covers down to {oldest}")
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
            logger.info(f"{market}: 4h reached listing start at {new_oldest}")
            break
        cur_to = new_oldest
        time.sleep(SLEEP_BETWEEN_PAGES)

    logger.info(f"{market}: 4h backfill done, {saved} rows ({pages} pages)")
    return saved


def collect_all(markets: list[str], days: int = 365 * 3, db_path: Path = DB_PATH) -> dict:
    results = {}
    for i, m in enumerate(markets, 1):
        logger.info(f"[{i}/{len(markets)}] 4h {m}")
        try:
            n = collect_market(m, days=days, db_path=db_path)
            results[m] = n
        except Exception as e:
            logger.error(f"{m}: 4h FAIL: {e}")
            results[m] = -1
        time.sleep(SLEEP_BETWEEN_MARKETS)
    return results


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Upbit KRW 4h 봉 수집기")
    parser.add_argument("--coin", type=str, help="단일 코인")
    parser.add_argument("--top", type=int, help="24h 거래대금 top N")
    parser.add_argument("--all", action="store_true", help="KRW 전체 마켓")
    parser.add_argument("--days", type=int, default=365 * 3, help="백필 일수")
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
        print(f"Saved {n} rows for {args.coin} (4h)")
    elif args.all:
        markets = get_krw_markets()
        print(f"전체 KRW 4h 백필 {len(markets)}개 시작 (days={args.days})")
        results = collect_all(markets, days=args.days, db_path=db_path)
        ok = sum(1 for v in results.values() if v >= 0)
        fail = sum(1 for v in results.values() if v < 0)
        total = sum(v for v in results.values() if v >= 0)
        print(f"\n=== 4h Done: OK {ok} / FAIL {fail} / Total {total} rows ===")
    elif args.top:
        markets = get_top_markets(args.top)
        results = collect_all(markets, days=args.days, db_path=db_path)
        print(f"4h Done: {len(results)} markets")
    else:
        parser.print_help()
        return

    print("\n=== 4h DB stats ===")
    print(stats(db_path).to_string(index=False))


if __name__ == "__main__":
    main()
