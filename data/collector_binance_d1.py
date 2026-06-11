"""바이낸스 USDT 일봉 수집기 — 학습 panel 다양성 확보용.

★ ARCHIVED-ISH (2026-06-11) — 일일 cron 미등록 (주간 retrain_run.sh 에서만 호출).
  binance_d1.db 는 retrain 주기 외엔 정지 상태가 정상. backup_db.sh 가 7일+
  unchanged DB 를 자동 skip 하므로 매일 백업 낭비 없음.

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

import ccxt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.database import (  # noqa: E402
    init_db,
    latest_timestamp,
    list_markets,
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


# ============================================================================
# 백필
# ============================================================================
def _fetch_page(symbol_pair: str, since_ms: int, limit: int = PAGE_SIZE) -> list | None:
    ex = get_exchange()
    last_err = None
    for attempt in range(1, RETRY_MAX + 1):
        try:
            data = ex.fetch_ohlcv(symbol_pair, timeframe=TIMEFRAME, since=since_ms, limit=limit)
            return data
        except Exception as e:
            last_err = e
            sleep_for = RETRY_BACKOFF ** attempt
            logger.warning(f"binance_d1 {symbol_pair} attempt {attempt} fail: {e}; sleep {sleep_for:.1f}s")
            time.sleep(sleep_for)
    logger.error(f"binance_d1 {symbol_pair} all retries failed: {last_err}")
    return None


def collect_market(base_symbol: str, days: int = 365 * 3, db_path: Path = DB_PATH) -> int:
    init_db(db_path)
    pair = f"{base_symbol}/USDT"
    market_key = f"BINANCE-{base_symbol}USDT"
    saved = 0

    target_since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)

    latest = latest_timestamp(db_path, market_key)
    if latest is not None and latest > target_since:
        since_dt = latest - timedelta(days=2)
    else:
        since_dt = target_since

    since_ms = int(since_dt.timestamp() * 1000)
    pages = 0
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    while since_ms < now_ms:
        data = _fetch_page(pair, since_ms)
        if not data:
            break

        df = pd.DataFrame(data, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.tz_convert(None)
        df["quote_volume"] = df["close"] * df["volume"]  # 근사 USD 거래대금
        df = df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]]

        n = save_candles(db_path, df, market_key)
        saved += n
        pages += 1

        last_ts_ms = data[-1][0]
        if last_ts_ms <= since_ms:
            break
        since_ms = last_ts_ms + 24 * 60 * 60 * 1000  # +1d

        if len(data) < PAGE_SIZE:
            break

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
def main():
    parser = argparse.ArgumentParser(description="Binance USDT 1d 수집기")
    parser.add_argument("--coin", type=str, help="단일 BASE")
    parser.add_argument("--all", action="store_true", help="USDT spot 전체")
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
    else:
        parser.print_help()
        return

    print("\n=== Binance d1 DB stats ===")
    print(stats(db_path).head(20).to_string(index=False))
    s = stats(db_path)
    print(f"\nTotal markets: {len(s)}, Total rows: {s['rows'].sum():,}" if len(s) > 0 else "")


if __name__ == "__main__":
    main()
