"""업비트 공개 API — 거래지원 종료 예정·유의 종목 조회 (알림 표시 전용).

R1 추천 메시지에 '🚫 거래지원 종료 예정' / '⚠️유의 종목' 표시를 붙이기 위한
조회 모듈. 추천 종목 선택·snapshot·receipt·원장에는 관여하지 않는다(표시만).

구성 (테스트 가능하도록 순수 파싱과 네트워크를 분리):
  - parse_market_warnings(payload)          : market/all?isDetails=true → 유의 심볼
  - parse_delisting_title(title, listed_at) : 공지 제목 → (심볼들, 종료 시각) | 날짜 미상 | 철회
  - parse_delisting_schedule(notices)       : 공지 목록 → ({심볼: 종료 시각}, {심볼: 날짜 미상 표시 기한})
  - parse_delisting_notices(notices)        : 위의 날짜 있는 부분만 {심볼: 종료 시각}
  - fetch_market_status(...)                : 네트워크 조회 (실패는 예외)
  - lookup_market_status(...)               : fail-open 래퍼 — 절대 예외를 올리지 않는다

★ fail-open: 조회 실패·타임아웃·파싱 오류는 발송을 막지 않는다. 결과의
  ``available=False`` 로만 표시되고, 메시지 끝에 '확인 불가' 한 줄이 붙는다.
★ 공개 GET 만 사용 (API key·주문 없음). 테스트 프로세스는
  PRELUDE_FORBID_MARKET_STATUS_FETCH=1 (conftest) 로 네트워크를 원천 차단한다.
"""
from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable

KST = timezone(timedelta(hours=9))

MARKET_ALL_URL = "https://api.upbit.com/v1/market/all"
ANNOUNCEMENTS_URL = "https://api-manager.upbit.com/api/v1/announcements"
FORBID_FETCH_ENV = "PRELUDE_FORBID_MARKET_STATUS_FETCH"

# 초기값(placeholder) — 발송 창(08:45~09:00 / 09:00~09:21)을 잠식하지 않게 짧게.
CONNECT_TIMEOUT_S = 2.0
READ_TIMEOUT_S = 3.0
TOTAL_BUDGET_S = 8.0        # 조회 전체 상한 (초과 시 남은 페이지 생략 → 확인 불가)
ANNOUNCEMENT_PAGES = 2      # 거래 공지 1페이지 ≈ 5주. 종료 공지는 보통 ~1달 전 게시
ANNOUNCEMENT_PER_PAGE = 20
ANNOUNCEMENT_LOOKBACK_DAYS = 45  # 이보다 오래된 공지만 남으면 다음 페이지 생략
# 2페이지를 다 읽고도 가장 오래된 일반 공지가 이보다 최근이면 종료 공지(보통
# 종료 ~30일 전 게시)를 놓쳤을 수 있다 → available=False ('확인 불가' 줄).
ANNOUNCEMENT_MIN_COVERAGE_DAYS = 31
UNDATED_VALID_DAYS = 45     # 날짜를 못 읽은 종료 공지의 표시 기한 (게시일 기준)
YEAR_ROLLBACK_DAYS = 300    # 종료 시각이 게시일보다 이만큼 넘게 미래면 전년도로 해석

_DELIST_KEYWORD = "거래지원 종료"
# 철회 판정은 '거래지원 종료' 와 붙어 나오는 패턴만 — 제목 끝에 덧붙은 부속
# 공지('- 비(BBB) 입금 취소', '(헤미(HEMI) 거래지원 취소 안내)')는 철회가 아니다.
_CANCEL_KEYWORDS = ("종료 결정 철회", "종료 철회", "종료 취소", "결정 취소")
# 'MM/DD HH:MM' — 연장·변경 공지는 여러 개일 수 있어 마지막(최신 일정)을 쓴다.
_SCHEDULE_RE = re.compile(r"(\d{1,2})/(\d{1,2})\s+(\d{1,2}):(\d{2})")
_WITHDRAW_KEYWORDS = ("출금", "입금")
# 'Name(SYM)' 의 SYM. 숫자 시작 심볼(0G, 1INCH)도 허용.
_SYMBOL_RE = re.compile(r"\(([A-Z0-9][A-Z0-9.\-]{0,19})\)")
_MARKET_ONLY_RE = re.compile(r"(KRW|BTC|USDT)(?:\s*,\s*(?:KRW|BTC|USDT))*\s*마켓")


class MarketStatusError(RuntimeError):
    """조회·파싱 실패 (fail-open 래퍼가 흡수한다)."""


@dataclass(frozen=True)
class DelistingNotice:
    symbols: tuple[str, ...]
    ends_at: datetime | None      # None = 철회/취소 공지 또는 날짜 미상
    cancelled: bool = False
    undated: bool = False         # 종료 공지인데 제목에서 종료 시각을 못 읽음


@dataclass(frozen=True)
class MarketStatus:
    """Top3 표시용 조회 결과. available=False 면 일부/전체 정보 없음."""

    available: bool
    warnings: frozenset[str] = frozenset()          # 유의 종목 심볼 (KRW 마켓)
    delistings: dict[str, datetime] = field(default_factory=dict)  # 심볼 → 종료 시각
    # 날짜 미상 종료 공지: 심볼 → 표시 기한 (게시일 + UNDATED_VALID_DAYS)
    undated_delistings: dict[str, datetime] = field(default_factory=dict)
    errors: tuple[str, ...] = ()

    @classmethod
    def unavailable(cls, reason: str) -> "MarketStatus":
        return cls(available=False, errors=(reason,))

    def badge(self, symbol: str, now: datetime) -> str | None:
        """종목 1개의 표시 문구. 종료 예정이 유의보다 우선."""
        ends_at = self.delistings.get(symbol)
        if ends_at is not None and ends_at > now:
            return f"🚫 거래지원 종료 예정 {ends_at:%m/%d} — 매수 비권장"
        undated_until = self.undated_delistings.get(symbol)
        if undated_until is not None and undated_until > now:
            return "🚫 거래지원 종료 예정(일정은 공지 확인)"
        if symbol in self.warnings:
            return "⚠️유의 종목"
        return None


# ==========================================================================
# 순수 파싱
# ==========================================================================
def parse_market_warnings(payload: Any) -> frozenset[str]:
    """market/all?isDetails=true 응답 → KRW 마켓 유의(warning=True) 심볼 집합."""
    if not isinstance(payload, list):
        raise MarketStatusError("market/all payload must be a list")
    out: set[str] = set()
    for row in payload:
        if not isinstance(row, dict):
            raise MarketStatusError("market/all row must be an object")
        market = row.get("market")
        if not isinstance(market, str) or not market.startswith("KRW-"):
            continue
        event = row.get("market_event")
        warning = event.get("warning") if isinstance(event, dict) else None
        # 구 스키마(market_warning='CAUTION') 도 허용.
        legacy = row.get("market_warning")
        if warning is True or legacy == "CAUTION":
            out.add(market[len("KRW-"):])
    return frozenset(out)


def _parse_listed_at(listed_at: Any) -> datetime:
    if not isinstance(listed_at, str):
        raise MarketStatusError("announcement listed_at missing")
    try:
        value = datetime.fromisoformat(listed_at)
    except ValueError as exc:
        raise MarketStatusError(f"bad listed_at: {listed_at!r}") from exc
    if value.tzinfo is None:
        value = value.replace(tzinfo=KST)
    return value.astimezone(KST)


def _end_schedules(title: str) -> list[tuple[str, ...]]:
    """제목의 'MM/DD HH:MM' 중 종료 시각으로 쓸 수 있는 것만.

    제외: 바로 뒤에 '~' 가 붙은 시작 시각('(09/01 09:00 ~)'), 그리고 앞쪽에서
    가장 가까운 문맥 키워드가 '거래지원 종료' 가 아니라 '출금'/'입금' 인 시각
    ('(출금 지원 종료: 11/12 15:00)')."""
    out: list[tuple[str, ...]] = []
    for match in _SCHEDULE_RE.finditer(title):
        if title[match.end():].lstrip().startswith("~"):
            continue
        prefix = title[:match.start()]
        withdraw_at = max(prefix.rfind(word) for word in _WITHDRAW_KEYWORDS)
        if withdraw_at > prefix.rfind(_DELIST_KEYWORD):
            continue
        out.append(match.groups())
    return out


def parse_delisting_title(title: str, listed_at: datetime) -> DelistingNotice | None:
    """공지 제목 1개 → 거래지원 종료 공지면 DelistingNotice, 아니면 None.

    예) '레이븐코인(RVN) 거래지원 종료 안내 (10/12 15:00)'
        'A(AAA), B(BBB) 거래지원 종료 안내 (10/19 15:00)'
        '...(RVN) 거래지원 종료 안내 (10/12 15:00) (일정 연장 안내: 10/20 15:00)'
        '...(XXX) 거래지원 종료 결정 철회 안내'  → cancelled
        '썬더코어(TT) 거래지원 종료 안내 (거래지원 종료 일정 변경 안내)' → undated
    연도는 제목에 없으므로 게시일(listed_at) 연도로 두고, 게시일보다 30일 이상
    과거가 되면 다음 해로 넘긴다(12월 공지 → 1월 종료). 반대로 게시일보다
    300일 넘게 미래면 전년도로 되돌린다(1월 공지 → 12월 종료).
    종료 시각을 못 읽으면(일정 변경 공지 등) undated=True — 표시는 '일정은 공지 확인'.
    특정 마켓만 종료(예: 'BTC 마켓 거래지원 종료')이고 KRW 가 없으면 None.
    """
    if not isinstance(title, str) or _DELIST_KEYWORD not in title:
        return None
    head = title.split(_DELIST_KEYWORD, 1)[0]
    symbols = tuple(dict.fromkeys(_SYMBOL_RE.findall(head)))
    if not symbols:
        return None
    market_only = _MARKET_ONLY_RE.search(title)
    if market_only and "KRW" not in market_only.group(0):
        return None
    if any(word in title for word in _CANCEL_KEYWORDS):
        return DelistingNotice(symbols=symbols, ends_at=None, cancelled=True)
    schedules = _end_schedules(title)
    if not schedules:
        return DelistingNotice(symbols=symbols, ends_at=None, undated=True)
    month, day, hour, minute = (int(x) for x in schedules[-1])
    try:
        ends_at = datetime(listed_at.year, month, day, hour, minute, tzinfo=KST)
    except ValueError as exc:
        raise MarketStatusError(f"bad delisting schedule in title: {title!r}") from exc
    if ends_at < listed_at - timedelta(days=30):
        ends_at = ends_at.replace(year=ends_at.year + 1)
    elif ends_at > listed_at + timedelta(days=YEAR_ROLLBACK_DAYS):
        ends_at = ends_at.replace(year=ends_at.year - 1)
    return DelistingNotice(symbols=symbols, ends_at=ends_at)


def parse_delisting_schedule(
    notices: Iterable[Any],
) -> tuple[dict[str, datetime], dict[str, datetime]]:
    """공지 목록 → ({심볼: 종료 시각}, {심볼: 날짜 미상 표시 기한}).

    심볼별 가장 최근 공지가 결정한다 (최근 공지가 철회면 종료 예정에서 제외,
    최근 공지가 날짜 미상이면 이전 날짜 대신 '일정은 공지 확인' 으로 표시)."""
    parsed: list[tuple[datetime, DelistingNotice]] = []
    for notice in notices:
        if not isinstance(notice, dict):
            raise MarketStatusError("announcement row must be an object")
        listed_at = _parse_listed_at(notice.get("listed_at"))
        item = parse_delisting_title(str(notice.get("title", "")), listed_at)
        if item is not None:
            parsed.append((listed_at, item))
    parsed.sort(key=lambda pair: pair[0], reverse=True)
    dated: dict[str, datetime] = {}
    undated: dict[str, datetime] = {}
    decided: set[str] = set()
    for listed_at, item in parsed:
        for symbol in item.symbols:
            if symbol in decided:
                continue
            decided.add(symbol)
            if item.cancelled:
                continue
            if item.undated:
                undated[symbol] = listed_at + timedelta(days=UNDATED_VALID_DAYS)
            elif item.ends_at is not None:
                dated[symbol] = item.ends_at
    return dated, undated


def parse_delisting_notices(notices: Iterable[Any]) -> dict[str, datetime]:
    """공지 목록 → {심볼: 종료 시각} (날짜 있는 종료 예정만)."""
    return parse_delisting_schedule(notices)[0]


def _announcement_rows(payload: Any) -> tuple[list[Any], list[Any], int | None]:
    """→ (고정 공지, 일반 공지, total_pages). 페이지 범위 판정은 일반 공지로만."""
    if not isinstance(payload, dict) or payload.get("success") is False:
        raise MarketStatusError("announcement payload not successful")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise MarketStatusError("announcement payload missing data")
    parts: dict[str, list[Any]] = {}
    for key in ("fixed_notices", "notices"):
        value = data.get(key) or []
        if not isinstance(value, list):
            raise MarketStatusError(f"announcement {key} must be a list")
        parts[key] = value
    total_pages = data.get("total_pages")
    return (parts["fixed_notices"], parts["notices"],
            total_pages if isinstance(total_pages, int) else None)


# ==========================================================================
# 네트워크
# ==========================================================================
def _default_get_json(url: str, params: dict, timeout: tuple[float, float]) -> Any:
    import requests  # 지연 import — 순수 파싱만 쓰는 쪽에 의존성 강제 X

    response = requests.get(
        url, params=params, timeout=timeout, headers={"Accept": "application/json"}
    )
    response.raise_for_status()
    return response.json()


def fetch_market_status(
    *,
    now: datetime,
    get_json: Callable[[str, dict, tuple[float, float]], Any] | None = None,
    pages: int = ANNOUNCEMENT_PAGES,
    budget_s: float = TOTAL_BUDGET_S,
    monotonic: Callable[[], float] = time.monotonic,
) -> MarketStatus:
    """market/all(1회) + 거래 공지(최대 ``pages`` 페이지) 조회.

    한쪽만 실패하면 얻은 정보는 살리고 available=False 로 돌려준다.
    양쪽 모두 실패하면 MarketStatusError.
    """
    if os.environ.get(FORBID_FETCH_ENV):
        raise MarketStatusError(f"{FORBID_FETCH_ENV} set — network lookup disabled")
    getter = get_json or _default_get_json
    started = monotonic()
    errors: list[str] = []

    def remaining_timeout() -> tuple[float, float]:
        left = budget_s - (monotonic() - started)
        if left <= 0.5:
            raise MarketStatusError("market status lookup budget exhausted")
        return (min(CONNECT_TIMEOUT_S, left), min(READ_TIMEOUT_S, left))

    warnings: frozenset[str] | None = None
    try:
        payload = getter(MARKET_ALL_URL, {"isDetails": "true"}, remaining_timeout())
        warnings = parse_market_warnings(payload)
    except Exception as exc:  # noqa: BLE001 — fail-open 경계
        errors.append(f"market_all: {type(exc).__name__}: {exc}")

    rows: list[Any] = []
    notices_ok = True
    read_all_pages = False
    oldest_general: datetime | None = None
    cutoff = now - timedelta(days=ANNOUNCEMENT_LOOKBACK_DAYS)
    for page in range(1, max(1, pages) + 1):
        try:
            payload = getter(
                ANNOUNCEMENTS_URL,
                {"os": "web", "page": page, "per_page": ANNOUNCEMENT_PER_PAGE,
                 "category": "trade"},
                remaining_timeout(),
            )
            fixed_rows, general_rows, total_pages = _announcement_rows(payload)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"announcements p{page}: {type(exc).__name__}: {exc}")
            notices_ok = False
            break
        rows.extend(fixed_rows)
        rows.extend(general_rows)
        if total_pages is not None and page >= total_pages:
            read_all_pages = True
            break
        # 오래된 고정 공지가 범위를 넓게 보이게 하지 않도록 일반 공지로만 판정.
        try:
            oldest = min(_parse_listed_at(r.get("listed_at")) for r in general_rows
                         if isinstance(r, dict))
        except (MarketStatusError, ValueError):
            break  # 빈 페이지·형식 이상 → 다음 페이지 의미 없음 (파싱 단계가 판정)
        oldest_general = oldest if oldest_general is None else min(oldest_general, oldest)
        if oldest < cutoff:
            break

    if notices_ok and not read_all_pages:
        min_coverage = now - timedelta(days=ANNOUNCEMENT_MIN_COVERAGE_DAYS)
        if oldest_general is None or oldest_general > min_coverage:
            since = f"{oldest_general:%m/%d}" if oldest_general else "unknown"
            errors.append(
                f"announcements cover only since {since} "
                f"(< {ANNOUNCEMENT_MIN_COVERAGE_DAYS}d) — older delisting notices may be missed")
            notices_ok = False

    delistings: dict[str, datetime] = {}
    undated: dict[str, datetime] = {}
    try:
        delistings, undated = parse_delisting_schedule(rows)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"announcements parse: {type(exc).__name__}: {exc}")
        notices_ok = False

    if warnings is None and not notices_ok:
        raise MarketStatusError("; ".join(errors))
    return MarketStatus(
        available=warnings is not None and notices_ok,
        warnings=warnings or frozenset(),
        delistings=delistings,
        undated_delistings=undated,
        errors=tuple(errors),
    )


def lookup_market_status(
    *,
    now: datetime | None = None,
    hard_timeout_s: float = TOTAL_BUDGET_S + 2.0,
    **kwargs: Any,
) -> MarketStatus:
    """fail-open 래퍼 — 어떤 예외도 올리지 않는다 (발송 경로 보호).

    requests timeout 은 DNS 해석(getaddrinfo)을 덮지 못하므로, 조회를 daemon
    thread 에서 돌리고 ``hard_timeout_s`` 가 지나면 기다리지 않고 '확인 불가'로
    진행한다 (남은 thread 는 프로세스 종료와 함께 사라진다)."""
    box: dict[str, MarketStatus] = {}

    def run() -> None:
        try:
            box["status"] = fetch_market_status(now=now or datetime.now(KST), **kwargs)
        except BaseException as exc:  # noqa: BLE001 — thread 밖으로 새지 않게
            box["status"] = MarketStatus.unavailable(f"{type(exc).__name__}: {exc}")

    try:
        worker = threading.Thread(target=run, name="upbit-market-status", daemon=True)
        worker.start()
        worker.join(hard_timeout_s)
    except Exception as exc:  # noqa: BLE001
        return MarketStatus.unavailable(f"{type(exc).__name__}: {exc}")
    status = box.get("status")
    if status is None:
        return MarketStatus.unavailable(f"lookup exceeded {hard_timeout_s:.0f}s")
    return status
