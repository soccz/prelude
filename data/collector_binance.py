"""바이낸스 USDT 1 시간 봉 수집기.

용도:
  - 김프 (업비트 KRW vs 바이낸스 USDT 환율 보정 후 가격 차)
  - binance_lead (글로벌 가격 선행 신호 — fin paper §17 검증된 강력 시그널)
  - 업비트 KRW 코인과 매칭되는 USDT pair 만 수집

라이브러리: ccxt (다 거래소 통합 SDK)
  - binance.fetch_ohlcv("BTC/USDT", timeframe="1h", since=..., limit=1000)
  - 1000 봉/페이지 (업비트 200 보다 많음)

사용:
    python -m data.collector_binance --coin BTC --days 365      # 단일
    python -m data.collector_binance --upbit-match --days 1095   # 업비트 KRW 와 매칭되는 모두
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

# ============================================================================
# 설정
# ============================================================================
DB_PATH = Path(__file__).resolve().parent / "binance_1h.db"
TIMEFRAME = "1h"
PAGE_SIZE = 1000          # binance klines 1000 봉/page
RETRY_MAX = 3
RETRY_BACKOFF = 1.5
SLEEP_BETWEEN_PAGES = 0.1   # binance rate 1200 weights/min → 20 req/s 까지 가능, 보수
SLEEP_BETWEEN_MARKETS = 0.2

logger = logging.getLogger("collector_binance")

_exchange = None


def get_exchange():
    """싱글톤 ccxt binance 인스턴스."""
    global _exchange
    if _exchange is None:
        _exchange = ccxt.binance({
            'enableRateLimit': True,  # ccxt 자동 rate limit
            'options': {'defaultType': 'spot'},
        })
    return _exchange


# ============================================================================
# 마켓 목록
# ============================================================================
def get_binance_usdt_symbols() -> set[str]:
    """바이낸스 USDT spot 마켓 전체 (BASE 만 — 예: 'BTC', 'ETH')."""
    ex = get_exchange()
    ex.load_markets()
    bases = set()
    for symbol, m in ex.markets.items():
        if m.get('quote') == 'USDT' and m.get('spot') and m.get('active'):
            bases.add(m.get('base'))
    return bases


def get_upbit_match_symbols() -> list[str]:
    """업비트 KRW 와 바이낸스 USDT 둘 다 상장된 코인 (BASE)."""
    EXCLUDE = {"USDT", "USDC"}  # 스테이블
    krw_markets = pyupbit.get_tickers(fiat="KRW")
    krw_bases = {m.split("-")[1] for m in krw_markets} - EXCLUDE
    binance_bases = get_binance_usdt_symbols()
    matched = sorted(krw_bases & binance_bases)
    logger.info(f"업비트 KRW {len(krw_bases)} + 바이낸스 USDT {len(binance_bases)} → 교집합 {len(matched)}")
    return matched


# ============================================================================
# 백필
# ============================================================================
def _fetch_page(symbol_pair: str, since_ms: int, limit: int = PAGE_SIZE) -> list | None:
    """한 페이지 (1000 봉) 수집. ccxt 자동 rate limit + retry."""
    ex = get_exchange()
    last_err = None
    for attempt in range(1, RETRY_MAX + 1):
        try:
            data = ex.fetch_ohlcv(symbol_pair, timeframe=TIMEFRAME, since=since_ms, limit=limit)
            return data
        except Exception as e:
            last_err = e
            sleep_for = RETRY_BACKOFF ** attempt
            logger.warning(f"binance {symbol_pair} attempt {attempt} fail: {e}; sleep {sleep_for:.1f}s")
            time.sleep(sleep_for)
    logger.error(f"binance {symbol_pair} all retries failed: {last_err}")
    return None


def collect_market(base_symbol: str, days: int = 365 * 3, db_path: Path = DB_PATH) -> int:
    """단일 코인 1h 백필 (since → now 방향, ccxt 패턴)."""
    init_db(db_path)
    pair = f"{base_symbol}/USDT"
    market_key = f"BINANCE-{base_symbol}USDT"  # DB 저장 키 (업비트 KRW-XXX 와 구분)
    saved = 0

    # since 결정
    target_since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)

    latest = latest_timestamp(db_path, market_key)
    if latest is not None and latest > target_since:
        # incremental: latest 부터 now 까지
        since_dt = latest - timedelta(hours=2)  # 약간 overlap (UPSERT)
    else:
        since_dt = target_since

    since_ms = int(since_dt.timestamp() * 1000)
    pages = 0
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    while since_ms < now_ms:
        data = _fetch_page(pair, since_ms)
        if not data:
            break

        # ccxt format: [timestamp_ms, open, high, low, close, volume]
        df = pd.DataFrame(data, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.tz_convert(None)
        df["quote_volume"] = df["close"] * df["volume"]  # 근사 quote_volume (USD)
        df = df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]]

        n = save_candles(db_path, df, market_key)
        saved += n
        pages += 1

        # 다음 페이지 since: 마지막 timestamp + 1 timeframe
        last_ts_ms = data[-1][0]
        if last_ts_ms <= since_ms:
            # 더 이상 진행 안 됨
            break
        since_ms = last_ts_ms + 60 * 60 * 1000  # +1h

        if len(data) < PAGE_SIZE:
            # 마지막 페이지 (덜 받음)
            break

        time.sleep(SLEEP_BETWEEN_PAGES)

    logger.info(f"{pair}: 1h backfill done, {saved} rows ({pages} pages)")
    return saved


def collect_all(symbols: list[str], days: int = 365 * 3, db_path: Path = DB_PATH) -> dict:
    results = {}
    for i, sym in enumerate(symbols, 1):
        logger.info(f"[{i}/{len(symbols)}] binance {sym}")
        try:
            n = collect_market(sym, days=days, db_path=db_path)
            results[sym] = n
        except Exception as e:
            logger.error(f"binance {sym} FAIL: {e}")
            results[sym] = -1
        time.sleep(SLEEP_BETWEEN_MARKETS)
    return results


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Binance USDT 1h 수집기")
    parser.add_argument("--coin", type=str, help="단일 BASE (예: BTC)")
    parser.add_argument("--upbit-match", action="store_true",
                        help="업비트 KRW 와 매칭되는 모든 코인")
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
        print(f"Saved {n} rows for binance {args.coin}/USDT")
    elif args.upbit_match:
        symbols = get_upbit_match_symbols()
        print(f"업비트 매칭 바이낸스 {len(symbols)}개 백필 시작 (days={args.days})")
        results = collect_all(symbols, days=args.days, db_path=db_path)
        ok = sum(1 for v in results.values() if v >= 0)
        fail = sum(1 for v in results.values() if v < 0)
        total = sum(v for v in results.values() if v >= 0)
        print(f"\n=== Binance Done: OK {ok} / FAIL {fail} / Total {total} rows ===")
    else:
        parser.print_help()
        return

    print("\n=== Binance DB stats ===")
    print(stats(db_path).to_string(index=False))


if __name__ == "__main__":
    main()
