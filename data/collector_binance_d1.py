"""바이낸스 USDT 일봉 수집기 — 학습 panel 다양성 확보용.

★ 2026-07-25부터 distribution runner가 매일 증분 갱신한다. 일부 심볼이라도
  최종 재시도에 실패하면 CLI가 nonzero를 반환해 후속 단계가 실패를 감지한다.

설계: collector_binance.py (1h) 와 동일 패턴, timeframe='1d'.
용도: 학습 panel 에 바이낸스 USDT 전체 코인 일봉 추가 (업비트 KRW 와 함께 학습)
      → 추론은 KRW only, 학습은 KRW + BINANCE 둘 다.

DB key: 'BINANCE-XXXUSDT' (collector_binance.py 와 동일 prefix)
DB 파일: data/binance_d1.db (1h 와 분리)

데이터 양: 432 코인 × 1095일 = ~470K rows (1h 의 ~10%, 빠름)

사용:
    python -m data.collector_binance_d1 --coin BTC --days 365
    python -m data.collector_binance_d1 --all --days 1095
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.database import (  # noqa: E402
    init_db,
    latest_timestamp,
    oldest_timestamp,
    save_candles,
    stats,
)
from data.collector_binance import get_binance_usdt_symbols, get_exchange  # 재사용

# ============================================================================
# 설정
# ============================================================================
DB_PATH = Path(__file__).resolve().parent / "binance_d1.db"
TIMEFRAME = "1d"
PAGE_SIZE = 1000
RETRY_MAX = 3
RETRY_BACKOFF = 1.5
SLEEP_BETWEEN_PAGES = 0.1
SLEEP_BETWEEN_MARKETS = 0.15

logger = logging.getLogger("collector_binance_d1")


class FetchPageError(RuntimeError):
    """Binance 일봉 페이지를 재시도 후에도 가져오지 못한 경우."""


def _utc_naive_to_epoch_ms(value: datetime) -> int:
    """timezone-naive Binance candle 시각을 host TZ와 무관한 UTC ms로 변환."""
    aware = (
        value.replace(tzinfo=timezone.utc)
        if value.tzinfo is None
        else value.astimezone(timezone.utc)
    )
    return int(aware.timestamp() * 1000)


# ============================================================================
# 백필
# ============================================================================
def _fetch_page(symbol_pair: str, since_ms: int, limit: int = PAGE_SIZE) -> list:
    ex = get_exchange()
    last_err = None
    for attempt in range(1, RETRY_MAX + 1):
        try:
            data = ex.fetch_ohlcv(symbol_pair, timeframe=TIMEFRAME, since=since_ms, limit=limit)
            if data is None:
                raise RuntimeError("fetch_ohlcv returned None")
            if len(data) == 0:
                raise RuntimeError("fetch_ohlcv returned an empty page")
            return data
        except Exception as e:
            last_err = e
            if attempt < RETRY_MAX:
                sleep_for = RETRY_BACKOFF ** attempt
                logger.warning(
                    "binance_d1 %s attempt %d fail: %s; sleep %.1fs",
                    symbol_pair,
                    attempt,
                    e,
                    sleep_for,
                )
                time.sleep(sleep_for)
    raise FetchPageError(
        f"binance_d1 {symbol_pair} failed after {RETRY_MAX} attempts: {last_err}"
    ) from last_err


def collect_market(base_symbol: str, days: int = 365 * 3, db_path: Path = DB_PATH) -> int:
    if days <= 0:
        raise ValueError("days must be positive")
    init_db(db_path)
    pair = f"{base_symbol}/USDT"
    market_key = f"BINANCE-{base_symbol}USDT"
    saved = 0

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    target_since = (now - timedelta(days=days)).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )

    latest = latest_timestamp(db_path, market_key)
    oldest = (
        oldest_timestamp(db_path, market_key)
        if latest is not None
        else None
    )
    if latest is not None and oldest is not None and oldest <= target_since:
        since_dt = latest - timedelta(days=2)
    else:
        # A prior short collection may contain recent rows while still missing
        # the requested historical prefix.  Resume from the requested boundary
        # instead of mistaking "market exists" for complete coverage.
        since_dt = target_since

    since_ms = _utc_naive_to_epoch_ms(since_dt)
    pages = 0
    now_ms = _utc_naive_to_epoch_ms(now)

    while since_ms < now_ms:
        data = _fetch_page(pair, since_ms)
        if len(data) == 0:
            raise FetchPageError(
                f"binance_d1 {pair} returned an empty page"
            )

        try:
            page_timestamps = [int(row[0]) for row in data]
        except (IndexError, TypeError, ValueError) as exc:
            raise FetchPageError(
                f"binance_d1 {pair} returned malformed candle rows"
            ) from exc
        if page_timestamps != sorted(set(page_timestamps)):
            raise FetchPageError(
                f"binance_d1 {pair} timestamps are not strictly increasing"
            )
        last_ts_ms = page_timestamps[-1]
        if last_ts_ms < since_ms:
            raise FetchPageError(
                f"binance_d1 {pair} page made no forward progress: "
                f"last={last_ts_ms} since={since_ms}"
            )

        df = pd.DataFrame(data, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.tz_convert(None)
        df["quote_volume"] = df["close"] * df["volume"]  # 근사 USD 거래대금
        df = df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]]

        n = save_candles(db_path, df, market_key)
        saved += n
        pages += 1

        since_ms = last_ts_ms + 24 * 60 * 60 * 1000  # +1d

        if since_ms < now_ms:
            time.sleep(SLEEP_BETWEEN_PAGES)

    logger.info(f"{pair}: d1 backfill done, {saved} rows ({pages} pages)")
    return saved


def collect_all(symbols: list[str], days: int = 365 * 3, db_path: Path = DB_PATH) -> dict:
    results = {}
    for i, sym in enumerate(symbols, 1):
        logger.info(f"[{i}/{len(symbols)}] binance_d1 {sym}")
        try:
            n = collect_market(sym, days=days, db_path=db_path)
            results[sym] = n
        except Exception as e:
            logger.error(f"binance_d1 {sym} FAIL: {e}")
            results[sym] = -1
        time.sleep(SLEEP_BETWEEN_MARKETS)
    return results


# ============================================================================
# CLI
# ============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="Binance USDT 1d 수집기")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--coin", type=str, help="단일 BASE")
    mode.add_argument("--all", action="store_true", help="USDT spot 전체")
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

    exit_code = 0
    if args.coin:
        n = collect_market(args.coin, days=args.days, db_path=db_path)
        print(f"Saved {n} rows for binance {args.coin}/USDT (1d)")
    elif args.all:
        symbols = sorted(get_binance_usdt_symbols())
        # 스테이블 / wrapped 제외
        EXCLUDE = {"USDC", "BUSD", "DAI", "FDUSD", "TUSD", "USDP", "PYUSD", "USDS"}
        symbols = [s for s in symbols if s not in EXCLUDE]
        print(f"바이낸스 USDT 전체 d1 백필 {len(symbols)}개 시작 (days={args.days})")
        results = collect_all(symbols, days=args.days, db_path=db_path)
        ok = sum(1 for v in results.values() if v >= 0)
        fail = sum(1 for v in results.values() if v < 0)
        total = sum(v for v in results.values() if v >= 0)
        print(f"\n=== Binance d1 Done: OK {ok} / FAIL {fail} / Total {total} rows ===")
        exit_code = 1 if fail else 0
    print("\n=== Binance d1 DB stats ===")
    print(stats(db_path).head(20).to_string(index=False))
    s = stats(db_path)
    print(f"\nTotal markets: {len(s)}, Total rows: {s['rows'].sum():,}" if len(s) > 0 else "")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
