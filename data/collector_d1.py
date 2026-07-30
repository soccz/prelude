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
    python -m data.collector_d1 --refresh-current-boundary    # 현재 09:00 결측만 재수집
"""
from __future__ import annotations

import argparse
import logging
import math
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyupbit
import requests  # type: ignore[import-untyped]

# 프로젝트 root 를 path 에 추가 (직접 실행 시)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.database import (  # noqa: E402
    connect_readonly,
    init_db,
    latest_timestamp,
    market_timestamp_ranges_readonly,
    oldest_timestamp,
    save_candles,
    stats,
)
from data.market_universe import (  # noqa: E402
    is_excluded_stablecoin_market,
    stablecoin_exclusions,
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
SLEEP_BETWEEN_TICKER_FALLBACKS = 0.12
SLEEP_BETWEEN_TOP_MARKETS = 0.12
# 초기값 64: 07-30 관측 결손 44개에 여유를 두되, 광역 API/DB 장애 때
# live universe 전체를 순회해 발송 창을 소진하지 않는다. missing_before와
# 복구 소요시간을 축적해 조정하며, 초과는 최종 health gate가 fail-closed한다.
MAX_CURRENT_BOUNDARY_REFRESH_MARKETS = 64
KST = timezone(timedelta(hours=9))

logger = logging.getLogger("collector_d1")
_KRW_MARKET_RE = re.compile(r"KRW-[A-Z0-9]+")


# ============================================================================
# 업데이트 결과
# ============================================================================
@dataclass(frozen=True)
class UniverseCoverage:
    """라이브 KRW 유니버스 대비 D1 DB 커버리지 스냅샷."""

    live_count: int
    db_before_count: int
    db_after_count: int
    covered_before_count: int
    covered_after_count: int
    new_markets: tuple[str, ...]
    backfill_markets: tuple[str, ...]
    inactive_db_markets: tuple[str, ...]
    missing_after: tuple[str, ...]
    failed_markets: tuple[str, ...]

    @property
    def ratio_before(self) -> float:
        return self.covered_before_count / self.live_count if self.live_count else 0.0

    @property
    def ratio_after(self) -> float:
        return self.covered_after_count / self.live_count if self.live_count else 0.0


class UpdateResults(dict[str, int]):
    """기존 ``dict[market, rows]`` 계약을 유지하면서 coverage 를 함께 반환."""

    coverage: UniverseCoverage

    def __init__(self, results: dict[str, int], coverage: UniverseCoverage):
        super().__init__(results)
        self.coverage = coverage


@dataclass(frozen=True)
class CurrentBoundaryRefresh:
    """현재 KST D1 경계 결측 live market만 재수집한 결과."""

    boundary: datetime
    live_count: int
    missing_before: tuple[str, ...]
    unresolved_after: tuple[str, ...]
    results: dict[str, int]
    failed_markets: tuple[str, ...]


class FetchPageError(RuntimeError):
    """API 재시도 전패를 정상적인 상장 시작/빈 페이지와 구분한다."""


class _TickerBatchNotFound(RuntimeError):
    """A mixed ticker request contains at least one not-yet-active pair."""


def _now_kst_naive() -> datetime:
    """pyupbit의 timezone-naive KST candle timestamp와 같은 wall-clock."""
    return datetime.now(KST).replace(tzinfo=None)


def current_d1_boundary(now: datetime | None = None) -> datetime:
    """현재 시각에 유효한 KST 09:00 D1 candle 시작 경계.

    09:00 정각부터는 당일 09:00, 그 전에는 전일 09:00이다. aware 입력은
    KST로 변환하고 naive 입력은 기존 수집기 계약대로 KST wall-clock이다.
    """
    wall = _wall_clock_kst(now or _now_kst_naive())
    boundary = wall.replace(hour=9, minute=0, second=0, microsecond=0)
    if wall < boundary:
        boundary -= timedelta(days=1)
    return boundary


def _markets_at_d1_boundary_readonly(
    db_path: Path,
    boundary: datetime,
) -> set[str]:
    """정확한 D1 boundary 행을 가진 market identity를 읽기 전용 조회."""
    timestamp = boundary.strftime("%Y-%m-%d %H:%M:%S")
    with connect_readonly(db_path) as conn:
        rows = conn.execute(
            "SELECT market FROM candles WHERE timestamp = ?",
            (timestamp,),
        ).fetchall()
    return {str(row[0]) for row in rows}


# ============================================================================
# 마켓 목록
# ============================================================================
def get_krw_markets(
    *,
    include_stablecoins_for_audit: bool = False,
) -> list[str]:
    """현재 ticker snapshot이 있는 업비트 KRW 마켓.

    ``market/all``에는 거래 개시 전 공지된 pair도 먼저 나타날 수 있다. 그런
    pair는 candle/ticker가 아직 HTTP 200 ``[]``이므로 현재 평가 유니버스에서
    제외하고, 첫 체결 snapshot이 생긴 다음 실행부터 자동 포함한다.

    ``include_stablecoins_for_audit``는 health provenance가 실제 제외 identity를
    보존하기 위한 진단 전용 옵션이다. 수집·시그널 경로는 기본값을 사용하므로
    stablecoin은 계속 중앙 정책에 따라 제외된다.
    """
    markets = None
    last_error: Exception | None = None
    for attempt in range(1, RETRY_MAX + 1):
        try:
            markets = pyupbit.get_tickers(fiat="KRW")
            if markets is None:
                raise RuntimeError("pyupbit.get_tickers returned None")
            break
        except Exception as exc:
            last_error = exc
            if attempt < RETRY_MAX:
                time.sleep(RETRY_BACKOFF ** attempt)
    if markets is None:
        raise RuntimeError(
            "Upbit live KRW market list request failed after retries: "
            f"{last_error}"
        ) from last_error
    if not isinstance(markets, list):
        raise RuntimeError(
            f"Upbit live KRW market list has invalid type: "
            f"{type(markets).__name__}"
        )
    if not markets:
        raise RuntimeError("Upbit live KRW market list is empty")
    if any(
        not isinstance(market, str)
        or _KRW_MARKET_RE.fullmatch(market) is None
        for market in markets
    ):
        raise RuntimeError("Upbit live KRW market list contains invalid identity")
    if len(set(markets)) != len(markets):
        raise RuntimeError("Upbit live KRW market list contains duplicates")
    excluded_stables = stablecoin_exclusions(markets)
    if excluded_stables:
        logger.info(
            "%s %d stablecoin KRW market(s): %s",
            (
                "Retained for universe audit"
                if include_stablecoins_for_audit
                else "Excluded"
            ),
            len(excluded_stables),
            ", ".join(excluded_stables),
        )
    eligible = sorted(
        market
        for market in markets
        if (
            include_stablecoins_for_audit
            or not is_excluded_stablecoin_market(market)
        )
    )
    if not eligible:
        raise RuntimeError("Upbit eligible KRW market list is empty")
    active = _markets_with_ticker_snapshot(eligible)
    if include_stablecoins_for_audit:
        logger.info(
            "Active KRW markets for universe audit: %d "
            "(stablecoin_identities_retained=%d)",
            len(active),
            len(stablecoin_exclusions(active)),
        )
    else:
        logger.info(
            "Active signal-eligible KRW markets: %d (stablecoin_excluded=%d)",
            len(active),
            len(excluded_stables),
        )
    return active


def _request_ticker_batch(batch: list[str]) -> list[dict]:
    payload: list[dict] | None = None
    last_error: Exception | None = None
    last_status: int | None = None
    for attempt in range(1, RETRY_MAX + 1):
        response = None
        try:
            response = requests.get(
                "https://api.upbit.com/v1/ticker",
                params={"markets": ",".join(batch)},
                headers={"accept": "application/json"},
                timeout=5,
            )
            response.raise_for_status()
            raw_payload = response.json()
            if not isinstance(raw_payload, list):
                raise ValueError("ticker response is not a list")
            payload = raw_payload
            break
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            error_response = getattr(exc, "response", None) or response
            last_status = getattr(error_response, "status_code", None)
            if attempt < RETRY_MAX:
                time.sleep(RETRY_BACKOFF ** attempt)
    if payload is not None:
        return payload
    if last_status == 404:
        raise _TickerBatchNotFound(
            f"Upbit ticker batch contains an inactive pair: {batch}"
        ) from last_error
    raise RuntimeError(
        f"Upbit ticker snapshot failed after retries: {last_error}"
    )


def _active_tickers_for_batch(batch: list[str]) -> set[str]:
    try:
        payload = _request_ticker_batch(batch)
    except _TickerBatchNotFound:
        # Some Upbit deployments reject a mixed ticker query when one listed
        # pair has no first trade yet instead of returning a partial/empty
        # HTTP-200 list. Split only that explicit 404 contract; transport/429/
        # 5xx failures remain fatal rather than silently shrinking universe.
        if len(batch) == 1:
            return set()
        midpoint = len(batch) // 2
        left = _active_tickers_for_batch(batch[:midpoint])
        time.sleep(SLEEP_BETWEEN_TICKER_FALLBACKS)
        right = _active_tickers_for_batch(batch[midpoint:])
        return left | right

    batch_set = set(batch)
    active: set[str] = set()
    for item in payload:
        if not isinstance(item, dict):
            raise RuntimeError("Upbit ticker snapshot contains a non-object")
        market = item.get("market")
        if market not in batch_set or market in active:
            raise RuntimeError(
                f"Upbit ticker snapshot identity violation: {market!r}"
            )
        try:
            trade_price = float(item["trade_price"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Upbit ticker snapshot invalid price: {market!r}"
            ) from exc
        if not math.isfinite(trade_price) or trade_price <= 0:
            raise RuntimeError(
                f"Upbit ticker snapshot non-positive price: {market!r}"
            )
        active.add(str(market))

    missing = [market for market in batch if market not in active]
    if not missing:
        return active
    if len(batch) == 1:
        # Only an explicit HTTP-200 empty singleton (or the 404 path above)
        # proves that a listed pair has no current trade snapshot.  A partial
        # multi-market response alone is not enough evidence to shrink the
        # universe because it is indistinguishable from a truncated response.
        return set()
    for market in missing:
        time.sleep(SLEEP_BETWEEN_TICKER_FALLBACKS)
        active.update(_active_tickers_for_batch([market]))
    return active


def _markets_with_ticker_snapshot(markets: list[str]) -> list[str]:
    active: set[str] = set()
    requested = set(markets)
    for offset in range(0, len(markets), 100):
        active.update(
            _active_tickers_for_batch(markets[offset:offset + 100])
        )
    if markets and not active:
        raise RuntimeError("Upbit active KRW ticker snapshot is empty")
    pending = sorted(requested - active)
    if pending:
        logger.info(
            "Excluded %d listed pair(s) without a trade snapshot: %s",
            len(pending),
            ", ".join(pending),
        )
    return sorted(active)


def get_top_markets(top_n: int, *, now: datetime | None = None) -> list[str]:
    """마지막 완결 D1 candle 거래대금 기준 top N.

    Upbit D1 candle은 KST 09:00에 시작한다. 09:00 이후 API의 마지막 row는
    당일 진행 중 candle이므로 ``candle_start + 24h <= now``인 row만 사용한다.
    """
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    snapshot_now = _wall_clock_kst(now or _now_kst_naive())
    markets = get_krw_markets()
    snaps = []
    for m in markets:
        df = _fetch_page(
            m,
            snapshot_now + timedelta(days=1),
            count=3,
        )
        # candle endpoint는 초당 10회 그룹 제한이다. 성공/empty 어느 쪽이든
        # 다음 market 요청 전에 간격을 둬 신규상장 다수도 burst가 되지 않게 한다.
        time.sleep(SLEEP_BETWEEN_TOP_MARKETS)
        if df is None or len(df) == 0:
            logger.info("%s: no candle yet; excluded from completed-D1 rank", m)
            continue

        candle_starts = pd.DatetimeIndex(pd.to_datetime(df.index))
        if candle_starts.tz is not None:
            candle_starts = candle_starts.tz_convert(KST).tz_localize(None)
        completed = df.loc[
            (candle_starts + pd.Timedelta(days=1)) <= pd.Timestamp(snapshot_now)
        ]
        if completed.empty:
            logger.info(
                "%s: no completed D1 candle at %s KST; excluded from rank",
                m,
                snapshot_now,
            )
            continue
        # pyupbit ``value`` = KRW 거래대금.
        v = float(completed["value"].iloc[-1])
        if not math.isfinite(v) or v < 0:
            raise FetchPageError(f"{m}: invalid completed quote_volume={v!r}")
        snaps.append((m, v))
    snaps.sort(key=lambda x: -x[1])
    chosen = [m for m, _ in snaps[:top_n]]
    if not chosen:
        raise FetchPageError(
            f"no eligible completed-D1 market at {snapshot_now} KST"
        )
    if len(chosen) != min(top_n, len(snaps)):
        raise FetchPageError(
            f"top-market snapshot incomplete: {len(chosen)}/{min(top_n, len(snaps))}"
        )
    logger.info(f"Selected top {len(chosen)} markets by 24h quote_volume")
    return chosen


def _wall_clock_kst(value: datetime) -> datetime:
    """aware 입력은 KST로 변환하고 naive 입력은 KST wall-clock으로 해석."""
    if value.tzinfo is None:
        return value
    return value.astimezone(KST).replace(tzinfo=None)


def _kst_wall_to_utc_naive(value: datetime) -> datetime:
    """Upbit REST ``to``용 KST wall-clock을 timezone-naive UTC로 변환."""
    kst_wall = _wall_clock_kst(value)
    return (
        kst_wall.replace(tzinfo=KST)
        .astimezone(timezone.utc)
        .replace(tzinfo=None)
    )


def _is_confirmed_empty_page(
    market: str,
    to: datetime,
    count: int,
    interval: str,
) -> bool:
    """Disambiguate pyupbit's ``None`` between an empty page and an error.

    ``pyupbit.get_ohlcv`` catches every exception and returns ``None``. It also
    hits that path when Upbit returns HTTP 200 with ``[]`` for a listed pair
    whose trading hasn't started yet. A direct, bounded REST probe treats only
    that exact response as a legitimate empty page; all transport, HTTP,
    decoding, or non-empty parse failures remain retryable errors.
    """
    candle_path = {
        "day": "days",
        "minute15": "minutes/15",
        "minute240": "minutes/240",
    }.get(interval)
    if candle_path is None:
        raise ValueError(f"unsupported Upbit candle interval: {interval}")
    api_to = _kst_wall_to_utc_naive(to)
    try:
        response = requests.get(
            f"https://api.upbit.com/v1/candles/{candle_path}",
            params={
                "market": market,
                "to": api_to.strftime("%Y-%m-%d %H:%M:%S"),
                "count": count,
            },
            headers={"accept": "application/json"},
            timeout=5,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return False
    return isinstance(payload, list) and not payload


# ============================================================================
# 백필
# ============================================================================
def _fetch_page(market: str, to: datetime, count: int = PAGE_SIZE) -> pd.DataFrame | None:
    """한 페이지 (200 봉) 수집. retry + backoff."""
    last_err = None
    api_to = _kst_wall_to_utc_naive(to)
    for attempt in range(1, RETRY_MAX + 1):
        try:
            df = pyupbit.get_ohlcv(
                market,
                interval=INTERVAL,
                to=api_to.strftime("%Y%m%d %H%M%S"),
                count=count,
            )
            # pyupbit는 HTTP/JSON 예외를 내부에서 삼키고 None으로 반환한다.
            # None을 상장 시작점으로 취급하면 부분 수집이 정상 종료로 위장한다.
            if df is None:
                if _is_confirmed_empty_page(
                    market,
                    to,
                    count,
                    INTERVAL,
                ):
                    return None
                raise RuntimeError("pyupbit.get_ohlcv returned None")
            if len(df) == 0:
                if _is_confirmed_empty_page(
                    market,
                    to,
                    count,
                    INTERVAL,
                ):
                    return None
                raise RuntimeError("pyupbit.get_ohlcv returned unconfirmed empty")
            return df
        except Exception as e:
            last_err = e
            if attempt < RETRY_MAX:
                sleep_for = RETRY_BACKOFF ** attempt
                logger.warning(f"fetch {market} attempt {attempt}/{RETRY_MAX} fail: {e}; sleep {sleep_for:.1f}s")
                time.sleep(sleep_for)
    logger.error(f"fetch {market} all retries failed: {last_err}")
    raise FetchPageError(f"{market}: all retries failed: {last_err}")


def collect_market(market: str, days: int = 365 * 3, db_path: Path = DB_PATH) -> int:
    """
    단일 마켓 백필 (현재 → 과거 days 일).

    이미 DB 에 있으면 oldest_timestamp 기준 더 과거만 추가 백필.
    현재가 가장 최근 (latest_timestamp) 부터 어제까지는 incremental update.

    return: 새로 저장된 행 수
    """
    if days <= 0:
        raise ValueError("days must be positive")
    init_db(db_path)
    now = _now_kst_naive()
    target_oldest = now - timedelta(days=days)
    saved = 0

    # 1. incremental update (latest 부터 현재까지)
    latest = latest_timestamp(db_path, market)
    if latest is None:
        oldest = None
    else:
        oldest = oldest_timestamp(db_path, market)
        # 오늘 일봉이 아직 안 마감됐을 수도 있으니 latest 부터 최근까지 다시 가져와 upsert
        df_recent = _fetch_page(market, now + timedelta(days=1))
        if df_recent is None or len(df_recent) == 0:
            raise FetchPageError(
                f"{market}: empty recent D1 page during live update"
            )
        n = save_candles(db_path, df_recent, market)
        saved += n
        logger.info(f"{market}: incremental upsert {n} rows (latest was {latest})")

    # 2. 과거 backfill (oldest 부터 target_oldest 까지)
    if oldest is None:
        cur_to = now + timedelta(days=1)
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


def refresh_current_d1_boundary(
    db_path: Path = DB_PATH,
    *,
    now: datetime | None = None,
    live_markets: list[str] | None = None,
) -> CurrentBoundaryRefresh:
    """현재 09:00 경계가 없는 live market만 한 번 bounded 재수집한다.

    재수집 후에도 경계가 없는 market은 여기서 실패로 단정하지 않는다. 얇은
    종목의 구조적 무거래인지 실제 DB gap인지는 곧이어 실행되는 health gate가
    업스트림을 재확인해 판정한다. 이 함수의 실패는 fetch 자체의 실패뿐이다.
    """
    init_db(db_path)
    boundary = current_d1_boundary(now)
    live = sorted(
        set(get_krw_markets() if live_markets is None else live_markets)
    )
    exact = _markets_at_d1_boundary_readonly(db_path, boundary)
    missing = sorted(set(live) - exact)
    logger.info(
        "D1 current-boundary targeted refresh: boundary=%s "
        "live=%d missing=%d%s",
        boundary,
        len(live),
        len(missing),
        f" [{', '.join(missing)}]" if missing else "",
    )
    targets = missing
    if len(missing) > MAX_CURRENT_BOUNDARY_REFRESH_MARKETS:
        logger.warning(
            "D1 current-boundary gap is too broad for targeted refresh: "
            "missing=%d cap=%d; defer to fail-closed health gate",
            len(missing),
            MAX_CURRENT_BOUNDARY_REFRESH_MARKETS,
        )
        targets = []
    results = (
        collect_all(targets, days=1, db_path=db_path)
        if targets
        else {}
    )
    failed = tuple(sorted(
        market for market, rows in results.items() if rows < 0
    ))
    exact_after = (
        _markets_at_d1_boundary_readonly(db_path, boundary)
        if missing
        else exact
    )
    unresolved = tuple(sorted(set(live) - exact_after))
    logger.info(
        "D1 current-boundary targeted refresh complete: "
        "attempted=%d failed=%d unresolved=%d%s",
        len(results),
        len(failed),
        len(unresolved),
        f" [{', '.join(unresolved)}]" if unresolved else "",
    )
    return CurrentBoundaryRefresh(
        boundary=boundary,
        live_count=len(live),
        missing_before=tuple(missing),
        unresolved_after=unresolved,
        results=results,
        failed_markets=failed,
    )


def update_existing(
    db_path: Path = DB_PATH,
    new_market_days: int = 365 * 3,
) -> UpdateResults:
    """라이브 유니버스를 동기화한 뒤 기존 market 을 incremental update.

    ``--update`` 가 DB 에 이미 있는 market 만 순회하면 신규 상장이 영구 누락된다.
    따라서 현재 Upbit KRW live ticker 와 DB 를 먼저 비교하고, 신규 market 을
    ``new_market_days`` 만큼 백필한 다음 현재 live인 기존 DB market만 갱신한다.
    상폐/비활성 market 의 과거 데이터는 survivorship 방어를 위해 삭제하지 않되,
    매일 실패가 확정된 API fetch를 반복하지 않는다.

    반환값은 기존 ``dict[market, rows]`` 호환 ``UpdateResults`` 이며,
    ``result.coverage`` 에 전/후 유니버스 커버리지를 명시적으로 담는다.
    """
    if new_market_days <= 0:
        raise ValueError("new_market_days must be positive")

    init_db(db_path)
    target_oldest = pd.Timestamp(
        _now_kst_naive() - timedelta(days=new_market_days)
    )
    ranges_before = market_timestamp_ranges_readonly(db_path)
    db_before = set(ranges_before)
    live_markets = set(get_krw_markets())
    new_markets = sorted(live_markets - db_before)
    incomplete_existing = sorted(
        market
        for market in live_markets & db_before
        if ranges_before[market][0] > target_oldest
    )
    backfill_markets = sorted(set(new_markets) | set(incomplete_existing))
    inactive_db_markets = sorted(db_before - live_markets)
    covered_before = live_markets & db_before

    logger.info(
        "D1 universe before update: live=%d db=%d covered=%d/%d (%.2f%%) "
        "new=%d backfill_required=%d inactive_retained=%d",
        len(live_markets),
        len(db_before),
        len(covered_before),
        len(live_markets),
        100.0 * len(covered_before) / len(live_markets),
        len(new_markets),
        len(backfill_markets),
        len(inactive_db_markets),
    )

    results: dict[str, int] = {}

    # 신규 또는 목표 oldest에 못 미친 부분 백필 market을 먼저 보충한다. 신규 수집이
    # 몇 page 저장한 뒤 실패해 DB에 존재하게 되어도 다음 --update에서 다시 이 분기로
    # 들어온다.
    if backfill_markets:
        logger.info(
            "Backfilling %d live markets (days=%d): %s",
            len(backfill_markets),
            new_market_days,
            ", ".join(backfill_markets),
        )
        results.update(
            collect_all(backfill_markets, days=new_market_days, db_path=db_path)
        )

    existing_markets = sorted((db_before & live_markets) - set(backfill_markets))
    if existing_markets:
        logger.info("Updating %d existing live DB markets", len(existing_markets))
        results.update(
            collect_all(existing_markets, days=7, db_path=db_path)
        )

    ranges_after = market_timestamp_ranges_readonly(db_path)
    db_after = set(ranges_after)
    covered_after = live_markets & db_after
    missing_after = sorted(live_markets - db_after)
    failed_markets = sorted(m for m, rows in results.items() if rows < 0)

    coverage = UniverseCoverage(
        live_count=len(live_markets),
        db_before_count=len(db_before),
        db_after_count=len(db_after),
        covered_before_count=len(covered_before),
        covered_after_count=len(covered_after),
        new_markets=tuple(new_markets),
        backfill_markets=tuple(backfill_markets),
        inactive_db_markets=tuple(inactive_db_markets),
        missing_after=tuple(missing_after),
        failed_markets=tuple(failed_markets),
    )
    logger.info(
        "D1 universe after update: covered=%d/%d (%.2f%%) missing=%s failed=%s",
        coverage.covered_after_count,
        coverage.live_count,
        100.0 * coverage.ratio_after,
        ", ".join(coverage.missing_after) or "none",
        ", ".join(coverage.failed_markets) or "none",
    )
    return UpdateResults(results, coverage)


# ============================================================================
# CLI
# ============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="Upbit KRW 일봉 수집기")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--coin", type=str, help="단일 코인 (예: KRW-BTC)")
    mode.add_argument("--top", type=int, help="완결 D1 거래대금 top N")
    mode.add_argument("--all", action="store_true", help="KRW 전체 마켓 (top N 제한 X)")
    mode.add_argument(
        "--update",
        action="store_true",
        help="라이브 KRW 신규/부분 마켓 백필 + 기존 DB 마켓 incremental update",
    )
    mode.add_argument(
        "--refresh-current-boundary",
        action="store_true",
        help="현재 KST 09:00 D1 행이 없는 live market만 1회 재수집",
    )
    parser.add_argument("--days", type=int, default=365 * 3, help="백필 일수 (기본 3년)")
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
        failed = n < 0
    elif args.all:
        markets = get_krw_markets()
        print(f"전체 KRW 마켓 {len(markets)}개 백필 시작 (days={args.days})")
        results = collect_all(markets, days=args.days, db_path=db_path)
        ok = sum(1 for v in results.values() if v >= 0)
        fail = sum(1 for v in results.values() if v < 0)
        total = sum(v for v in results.values() if v >= 0)
        print(f"\n=== Done: OK {ok} / FAIL {fail} / Total {total} rows ===")
        failed = fail > 0
    elif args.top is not None:
        markets = get_top_markets(args.top)
        results = collect_all(markets, days=args.days, db_path=db_path)
        print(f"\n=== Done: {len(results)} markets ===")
        ok = sum(1 for v in results.values() if v >= 0)
        fail = sum(1 for v in results.values() if v < 0)
        total = sum(v for v in results.values() if v >= 0)
        print(f"OK {ok} / FAIL {fail} / Total saved {total} rows")
        failed = fail > 0
    elif args.update:
        results = update_existing(db_path=db_path, new_market_days=args.days)
        print(f"Updated {len(results)} markets")
        coverage = results.coverage
        print(
            "Universe coverage: "
            f"{coverage.covered_before_count}/{coverage.live_count} "
            f"({coverage.ratio_before:.2%}) -> "
            f"{coverage.covered_after_count}/{coverage.live_count} "
            f"({coverage.ratio_after:.2%}); "
            f"new={len(coverage.new_markets)} "
            f"missing={len(coverage.missing_after)} "
            f"inactive_retained={len(coverage.inactive_db_markets)}"
        )
        if coverage.missing_after:
            print(f"Missing live markets: {', '.join(coverage.missing_after)}")
        if coverage.failed_markets:
            print(f"Failed markets: {', '.join(coverage.failed_markets)}")
        failed = bool(coverage.missing_after or coverage.failed_markets)
    elif args.refresh_current_boundary:
        refresh = refresh_current_d1_boundary(db_path=db_path)
        print(
            "Current-boundary targeted refresh: "
            f"boundary={refresh.boundary} "
            f"live={refresh.live_count} "
            f"missing_before={len(refresh.missing_before)} "
            f"attempted={len(refresh.results)} "
            f"failed={len(refresh.failed_markets)} "
            f"unresolved={len(refresh.unresolved_after)}"
        )
        if refresh.missing_before:
            print(
                "Targeted markets: "
                f"{', '.join(refresh.missing_before)}"
            )
        if refresh.failed_markets:
            print(f"Failed markets: {', '.join(refresh.failed_markets)}")
        if refresh.unresolved_after:
            print(
                "Unresolved markets (final health gate decides): "
                f"{', '.join(refresh.unresolved_after)}"
            )
        return 1 if refresh.failed_markets else 0
    print("\n=== DB stats ===")
    print(stats(db_path).to_string(index=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
