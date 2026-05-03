"""업비트 KRW 일봉 수집기.

설계 (SIGNAL.md §1):
  - 업비트 KRW 마켓 전체 (필요 시 top N 만)
  - 일봉 (KST 09:00 마감 = UTC 00:00)
  - pyupbit 사용 (REST API wrapper)
  - 백필: oldest_timestamp 기준 거꾸로 페이지네이션 (200 봉씩)
  - upsert (중복 안전)
  - retry + backoff

Adapted from: gan_t/data/collector.py
Changes:
  - 일봉 only (gan_t 는 1h 봉)
  - pyupbit 라이브러리 사용 (gan_t 는 raw requests)
  - 단일 함수 진입점 (collect_market / collect_all)
  - top N 동적 선정 (24h 거래대금)
  - quote_volume (KRW 거래대금) 정확히 저장 — pyupbit 의 'value' 컬럼

사용:
    python -m data.collector_d1 --coin KRW-BTC --days 365     # 단일 코인
    python -m data.collector_d1 --top 100 --days 365          # top 100 코인
    python -m data.collector_d1 --all --days 1095             # 전체 KRW 마켓 3년치
    python -m data.collector_d1 --update                      # 모든 기존 코인 incremental 갱신
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

# 프로젝트 root 를 path 에 추가 (직접 실행 시)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.database import (  # noqa: E402
    init_db,
    latest_timestamp,
    list_markets,
    oldest_timestamp,
    save_candles,
    stats,
)

# ============================================================================
# 설정
# ============================================================================
DB_PATH = Path(__file__).resolve().parent / "upbit_d1.db"
INTERVAL = "day"
PAGE_SIZE = 200       # 업비트 candles API 최대 200
RETRY_MAX = 3
RETRY_BACKOFF = 1.5    # exponential
SLEEP_BETWEEN_PAGES = 0.15  # API rate limit (10/s 안전)
SLEEP_BETWEEN_MARKETS = 0.3

logger = logging.getLogger("collector_d1")


# ============================================================================
# 마켓 목록
# ============================================================================
def get_krw_markets() -> list[str]:
    """업비트 KRW 마켓 전체 (스테이블 / 레버리지 토큰 제외)."""
    EXCLUDE = {
        "KRW-USDT",  # 스테이블
        "KRW-USDC",
    }
    markets = pyupbit.get_tickers(fiat="KRW")
    return sorted(m for m in markets if m not in EXCLUDE)


def get_top_markets(top_n: int) -> list[str]:
    """24h 거래대금 기준 top N."""
    markets = get_krw_markets()
    # 24h 거래대금: 어제 일봉 1개씩 받아 quote_volume 기준 정렬
    snaps = []
    for m in markets:
        try:
            df = pyupbit.get_ohlcv(m, interval=INTERVAL, count=2)
            if df is None or len(df) == 0:
                continue
            # df.value 가 KRW 거래대금
            v = float(df["value"].iloc[-1])
            snaps.append((m, v))
            time.sleep(0.05)  # rate limit
        except Exception as e:
            logger.warning(f"top_markets snapshot fail {m}: {e}")
            continue
    snaps.sort(key=lambda x: -x[1])
    chosen = [m for m, _ in snaps[:top_n]]
    logger.info(f"Selected top {len(chosen)} markets by 24h quote_volume")
    return chosen


# ============================================================================
# 백필
# ============================================================================
def _fetch_page(market: str, to: datetime, count: int = PAGE_SIZE) -> pd.DataFrame | None:
    """한 페이지 (200 봉) 수집. retry + backoff."""
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
    """
    단일 마켓 백필 (현재 → 과거 days 일).

    이미 DB 에 있으면 oldest_timestamp 기준 더 과거만 추가 백필.
    현재가 가장 최근 (latest_timestamp) 부터 어제까지는 incremental update.

    return: 새로 저장된 행 수
    """
    init_db(db_path)
    target_oldest = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
    saved = 0

    # 1. incremental update (latest 부터 현재까지)
    latest = latest_timestamp(db_path, market)
    if latest is None:
        # 빈 DB → 미래 1일 (안전 마진) 부터 거꾸로
        to = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=1)
        oldest = None
    else:
        oldest = oldest_timestamp(db_path, market)
        # 오늘 일봉이 아직 안 마감됐을 수도 있으니 latest 부터 최근까지 다시 가져와 upsert
        df_recent = _fetch_page(market, datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=1))
        if df_recent is not None and len(df_recent) > 0:
            n = save_candles(db_path, df_recent, market)
            saved += n
            logger.info(f"{market}: incremental upsert {n} rows (latest was {latest})")

    # 2. 과거 backfill (oldest 부터 target_oldest 까지)
    if oldest is None:
        cur_to = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=1)
    else:
        cur_to = oldest

    if cur_to <= target_oldest:
        logger.info(f"{market}: already covers down to {oldest}, target={target_oldest}. skip backfill")
        return saved

    pages = 0
    while cur_to > target_oldest:
        df = _fetch_page(market, cur_to)
        if df is None or len(df) == 0:
            logger.info(f"{market}: no more data before {cur_to}")
            break

        n = save_candles(db_path, df, market)
        saved += n
        pages += 1

        new_oldest = df.index.min().to_pydatetime().replace(tzinfo=None)
        logger.debug(f"{market}: page {pages} → {n} rows, oldest now {new_oldest}")

        if new_oldest >= cur_to:
            # 더 이상 과거 없음 (상장 시작점)
            logger.info(f"{market}: reached listing start at {new_oldest}")
            break
        cur_to = new_oldest
        time.sleep(SLEEP_BETWEEN_PAGES)

    logger.info(f"{market}: backfill done, total saved {saved} rows ({pages} pages)")
    return saved


def collect_all(markets: list[str], days: int = 365 * 3, db_path: Path = DB_PATH) -> dict:
    """여러 마켓 backfill. 결과 요약 dict 반환."""
    results = {}
    for i, m in enumerate(markets, 1):
        logger.info(f"[{i}/{len(markets)}] {m}")
        try:
            n = collect_market(m, days=days, db_path=db_path)
            results[m] = n
        except Exception as e:
            logger.error(f"{m}: collect FAIL: {e}")
            results[m] = -1
        time.sleep(SLEEP_BETWEEN_MARKETS)
    return results


def update_existing(db_path: Path = DB_PATH) -> dict:
    """DB 에 이미 있는 모든 market incremental update (어제 일봉 추가)."""
    markets = list_markets(db_path)
    if not markets:
        logger.warning("DB 에 마켓 없음. --top N 또는 --coin 으로 먼저 백필.")
        return {}
    logger.info(f"Updating {len(markets)} existing markets")
    return collect_all(markets, days=7, db_path=db_path)  # 최근 7일만 (incremental)


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Upbit KRW 일봉 수집기")
    parser.add_argument("--coin", type=str, help="단일 코인 (예: KRW-BTC)")
    parser.add_argument("--top", type=int, help="24h 거래대금 top N")
    parser.add_argument("--all", action="store_true", help="KRW 전체 마켓 (top N 제한 X)")
    parser.add_argument("--days", type=int, default=365 * 3, help="백필 일수 (기본 3년)")
    parser.add_argument("--update", action="store_true", help="기존 모든 코인 incremental update")
    parser.add_argument("--db", type=str, default=str(DB_PATH), help="DB 경로")
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
        print(f"Saved {n} rows for {args.coin}")
    elif args.all:
        markets = get_krw_markets()
        print(f"전체 KRW 마켓 {len(markets)}개 백필 시작 (days={args.days})")
        results = collect_all(markets, days=args.days, db_path=db_path)
        ok = sum(1 for v in results.values() if v >= 0)
        fail = sum(1 for v in results.values() if v < 0)
        total = sum(v for v in results.values() if v >= 0)
        print(f"\n=== Done: OK {ok} / FAIL {fail} / Total {total} rows ===")
    elif args.top:
        markets = get_top_markets(args.top)
        results = collect_all(markets, days=args.days, db_path=db_path)
        print(f"\n=== Done: {len(results)} markets ===")
        ok = sum(1 for v in results.values() if v >= 0)
        fail = sum(1 for v in results.values() if v < 0)
        total = sum(v for v in results.values() if v >= 0)
        print(f"OK {ok} / FAIL {fail} / Total saved {total} rows")
    elif args.update:
        results = update_existing(db_path=db_path)
        print(f"Updated {len(results)} markets")
    else:
        parser.print_help()
        return

    print("\n=== DB stats ===")
    print(stats(db_path).to_string(index=False))


if __name__ == "__main__":
    main()
