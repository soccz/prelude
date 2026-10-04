"""거래지원 종료 예정·유의 종목 조회 + 2026-10-04 알림 문구 변경 테스트.

네트워크는 get_json 주입으로만 흉내 낸다 (conftest 가
PRELUDE_FORBID_MARKET_STATUS_FETCH=1 로 실조회를 원천 차단).
"""
from __future__ import annotations

import copy
import time
from datetime import datetime, timedelta, timezone

import pytest

import data.upbit_market_status as status_mod
import scripts.recommend_send as sender
from data.upbit_market_status import (
    MarketStatus,
    MarketStatusError,
    parse_delisting_notices,
    parse_delisting_schedule,
    parse_delisting_title,
    parse_market_warnings,
)

KST = timezone(timedelta(hours=9))
NOW = datetime(2026, 10, 4, 8, 50, tzinfo=KST)


def _listed(text: str) -> datetime:
    return datetime.fromisoformat(text)


# ==========================================================================
# 공지 제목 파싱
# ==========================================================================
@pytest.mark.parametrize(
    "title, listed_at, symbols, ends_at",
    [
        ("레이븐코인(RVN) 거래지원 종료 안내 (10/12 15:00)",
         "2026-09-10T18:30:01+09:00", ("RVN",), datetime(2026, 10, 12, 15, 0, tzinfo=KST)),
        ("아이콘(ICX) 거래지원 종료 안내 (10/19 15:00)",
         "2026-09-18T16:30:00+09:00", ("ICX",), datetime(2026, 10, 19, 15, 0, tzinfo=KST)),
        ("신세틱스(SNX) 거래지원 종료 안내 (9/28 15:00)",
         "2026-08-28T16:00:00+09:00", ("SNX",), datetime(2026, 9, 28, 15, 0, tzinfo=KST)),
        ("에이(AAA), 비비(BBB), 제로지(0G) 거래지원 종료 안내 (10/20 15:00)",
         "2026-09-20T16:00:00+09:00", ("AAA", "BBB", "0G"),
         datetime(2026, 10, 20, 15, 0, tzinfo=KST)),
        # 연장·변경 공지: 마지막(최신) 일정이 이긴다.
        ("레이븐코인(RVN) 거래지원 종료 안내 (10/12 15:00) (일정 연장 안내: 10/26 15:00)",
         "2026-10-05T10:00:00+09:00", ("RVN",), datetime(2026, 10, 26, 15, 0, tzinfo=KST)),
        # 12월 게시 → 1월 종료는 다음 해.
        ("코인(CCC) 거래지원 종료 안내 (1/15 15:00)",
         "2026-12-20T16:00:00+09:00", ("CCC",), datetime(2027, 1, 15, 15, 0, tzinfo=KST)),
        # KRW 를 포함한 마켓 한정 종료는 KRW 영향 → 포함.
        ("코인(DDD) KRW, BTC 마켓 거래지원 종료 안내 (10/30 15:00)",
         "2026-10-01T16:00:00+09:00", ("DDD",), datetime(2026, 10, 30, 15, 0, tzinfo=KST)),
    ],
)
def test_delisting_title_parses_symbols_and_schedule(title, listed_at, symbols, ends_at):
    notice = parse_delisting_title(title, _listed(listed_at))
    assert notice is not None and not notice.cancelled
    assert notice.symbols == symbols
    assert notice.ends_at == ends_at


@pytest.mark.parametrize(
    "title",
    [
        "블라스트(BLAST) 거래 유의 종목 지정 안내",
        "돌핀(POD) 신규 거래지원 안내 (KRW, BTC, USDT 마켓)",
        "만트라(MANTRA) 거래 유의 종목 지정 기간 연장 안내",
        "코인(EEE) BTC 마켓 거래지원 종료 안내 (10/30 15:00)",  # KRW 무관
        "코인(FFF) USDT, BTC 마켓 거래지원 종료 안내 (10/30 15:00)",
        "거래지원 종료 안내 (10/30 15:00)",  # 심볼 없음
        "헤미(HEMI) 거래지원 취소 안내",  # 신규 상장 취소 — 종료 공지 아님
    ],
)
def test_non_krw_or_non_delisting_titles_are_ignored(title):
    assert parse_delisting_title(title, NOW) is None


# --- 2026-10-04 리뷰 반영: 날짜 미상·시작 시각·출금 문맥·철회 키워드·연도 대칭 ---
@pytest.mark.parametrize(
    "title",
    [
        # 실제 공지 (업비트는 일정 변경 시 같은 글의 제목·listed_at 을 바꾼다)
        "썬더코어(TT) 거래지원 종료 안내 (거래지원 종료 일정 변경 안내)",
        # '~' 가 붙은 시각은 출금 중단 '시작' 시각 — 종료 시각으로 쓰지 않는다.
        "ThunderCore 네트워크 종료에 따른 썬더코어(TT) 출금 중단(종료) 및 "
        "거래지원 종료 예정일 변경 안내 (09/01 09:00 ~)",
        "코인(GGG) 거래지원 종료 안내",  # 일정 없음
    ],
)
def test_delisting_title_without_readable_end_is_undated(title):
    notice = parse_delisting_title(title, _listed("2026-08-28T10:00:00+09:00"))
    assert notice is not None
    assert notice.undated and not notice.cancelled and notice.ends_at is None
    assert notice.symbols in {("TT",), ("GGG",)}


@pytest.mark.parametrize(
    "title, ends_at",
    [
        # 출금 문맥의 시각은 종료 시각이 아니다.
        ("에이(AAA) 거래지원 종료 안내 (10/12 15:00) (출금 지원 종료: 11/12 15:00)",
         datetime(2026, 10, 12, 15, 0, tzinfo=KST)),
        # '출금 중단 및 거래지원 종료' — 가장 가까운 문맥이 거래지원 종료면 그대로 쓴다.
        ("에이(AAA) 출금 중단 및 거래지원 종료 안내 (10/12 15:00)",
         datetime(2026, 10, 12, 15, 0, tzinfo=KST)),
        # 범위 표기: '~' 앞 시작 시각은 버리고 뒤 시각을 쓴다.
        ("에이(AAA) 거래지원 종료 안내 (10/01 09:00 ~ 10/12 15:00)",
         datetime(2026, 10, 12, 15, 0, tzinfo=KST)),
    ],
)
def test_start_and_withdrawal_times_are_not_end_times(title, ends_at):
    notice = parse_delisting_title(title, _listed("2026-09-20T10:00:00+09:00"))
    assert notice is not None and not notice.undated
    assert notice.symbols == ("AAA",) and notice.ends_at == ends_at


@pytest.mark.parametrize(
    "title",
    [
        "에이(AAA) 거래지원 종료 안내 (10/12 15:00) - 비(BBB) 입금 취소",
        "에이(AAA) 거래지원 종료 안내 (10/12 15:00) (헤미(HEMI) 거래지원 취소 안내)",
    ],
)
def test_appended_unrelated_cancel_word_does_not_cancel_delisting(title):
    notice = parse_delisting_title(title, _listed("2026-09-20T10:00:00+09:00"))
    assert notice is not None and not notice.cancelled
    assert notice.symbols == ("AAA",)
    assert notice.ends_at == datetime(2026, 10, 12, 15, 0, tzinfo=KST)


@pytest.mark.parametrize(
    "title",
    [
        "레이븐코인(RVN) 거래지원 종료 결정 철회 안내",
        "레이븐코인(RVN) 거래지원 종료 철회 안내",
        "레이븐코인(RVN) 거래지원 종료 취소 안내",
        "레이븐코인(RVN) 거래지원 종료 결정 취소 안내",
    ],
)
def test_cancellation_keywords_attached_to_delisting(title):
    notice = parse_delisting_title(title, NOW)
    assert notice is not None and notice.cancelled and notice.symbols == ("RVN",)


def test_year_inference_is_symmetric():
    # 1월 게시 + 12/31 → 전년도 12/31 (1년 가까이 미래로 해석하지 않는다).
    notice = parse_delisting_title("코인(CCC) 거래지원 종료 안내 (12/31 15:00)",
                                   _listed("2027-01-03T10:00:00+09:00"))
    assert notice is not None
    assert notice.ends_at == datetime(2026, 12, 31, 15, 0, tzinfo=KST)


def test_undated_newest_notice_overrides_older_date_for_45_days():
    notices = [
        {"listed_at": "2026-08-01T16:00:00+09:00",
         "title": "썬더코어(TT) 거래지원 종료 안내 (09/01 15:00)"},
        {"listed_at": "2026-09-25T16:00:00+09:00",
         "title": "썬더코어(TT) 거래지원 종료 안내 (거래지원 종료 일정 변경 안내)"},
        {"listed_at": "2026-09-18T16:30:00+09:00",
         "title": "아이콘(ICX) 거래지원 종료 안내 (10/19 15:00)"},
    ]
    dated, undated = parse_delisting_schedule(notices)
    assert dated == {"ICX": datetime(2026, 10, 19, 15, 0, tzinfo=KST)}
    assert undated == {"TT": datetime(2026, 11, 9, 16, 0, tzinfo=KST)}
    assert parse_delisting_notices(notices) == dated
    status = MarketStatus(available=True, warnings=frozenset({"TT"}),
                          delistings=dated, undated_delistings=undated)
    assert status.badge("TT", NOW) == "🚫 거래지원 종료 예정(일정은 공지 확인)"
    # 45일이 지나면 날짜 미상 표시는 내리고 유의 표시만 남는다.
    assert status.badge("TT", datetime(2026, 11, 9, 16, 0, tzinfo=KST)) == "⚠️유의 종목"


def test_newer_dated_notice_replaces_older_undated():
    notices = [
        {"listed_at": "2026-09-01T16:00:00+09:00",
         "title": "썬더코어(TT) 거래지원 종료 안내 (거래지원 종료 일정 변경 안내)"},
        {"listed_at": "2026-09-20T16:00:00+09:00",
         "title": "썬더코어(TT) 거래지원 종료 안내 (10/20 15:00)"},
    ]
    dated, undated = parse_delisting_schedule(notices)
    assert dated == {"TT": datetime(2026, 10, 20, 15, 0, tzinfo=KST)} and undated == {}


def test_cancellation_notice_overrides_older_delisting():
    notices = [
        {"listed_at": "2026-09-10T18:30:01+09:00",
         "title": "레이븐코인(RVN) 거래지원 종료 안내 (10/12 15:00)"},
        {"listed_at": "2026-09-18T16:30:00+09:00",
         "title": "아이콘(ICX) 거래지원 종료 안내 (10/19 15:00)"},
        {"listed_at": "2026-09-20T10:00:00+09:00",
         "title": "레이븐코인(RVN) 거래지원 종료 결정 철회 안내"},
    ]
    assert parse_delisting_notices(notices) == {
        "ICX": datetime(2026, 10, 19, 15, 0, tzinfo=KST)
    }


def test_newest_notice_wins_regardless_of_input_order():
    notices = [
        {"listed_at": "2026-10-05T10:00:00+09:00",
         "title": "레이븐코인(RVN) 거래지원 종료 안내 (10/12 15:00) (일정 연장 안내: 10/26 15:00)"},
        {"listed_at": "2026-09-10T18:30:01+09:00",
         "title": "레이븐코인(RVN) 거래지원 종료 안내 (10/12 15:00)"},
    ]
    assert parse_delisting_notices(notices)["RVN"] == datetime(2026, 10, 26, 15, 0, tzinfo=KST)


def test_malformed_notice_raises_for_fail_open_handling():
    with pytest.raises(MarketStatusError):
        parse_delisting_notices([{"title": "x(X) 거래지원 종료 안내 (10/12 15:00)"}])
    with pytest.raises(MarketStatusError):
        parse_delisting_title("x(XYZ) 거래지원 종료 안내 (13/45 15:00)", NOW)


# ==========================================================================
# market_event 파싱
# ==========================================================================
def test_market_event_warning_only_krw_symbols():
    payload = [
        {"market": "KRW-RVN", "market_event": {"warning": True, "caution": {}}},
        {"market": "BTC-ZIL", "market_event": {"warning": True, "caution": {}}},
        {"market": "KRW-BTC", "market_event": {"warning": False,
                                               "caution": {"PRICE_FLUCTUATIONS": True}}},
        {"market": "KRW-OLD", "market_warning": "CAUTION"},
        {"market": "KRW-NONE"},
    ]
    assert parse_market_warnings(payload) == frozenset({"RVN", "OLD"})


@pytest.mark.parametrize("payload", [{}, None, "x", [1]])
def test_market_event_rejects_malformed_payload(payload):
    with pytest.raises(MarketStatusError):
        parse_market_warnings(payload)


# ==========================================================================
# 네트워크 함수 (주입 getter) + fail-open
# ==========================================================================
MARKET_ALL = [
    {"market": "KRW-RVN", "market_event": {"warning": True}},
    {"market": "KRW-ICX", "market_event": {"warning": True}},
    {"market": "KRW-BLAST", "market_event": {"warning": True}},
    {"market": "KRW-BAT", "market_event": {"warning": False}},
]
PAGE1 = {
    "success": True,
    "data": {
        "total_pages": 40,
        "fixed_notices": [],
        "notices": [
            {"listed_at": "2026-10-03T01:30:00+09:00",
             "title": "블라스트(BLAST) 거래 유의 종목 지정 안내"},
            {"listed_at": "2026-09-18T16:30:00+09:00",
             "title": "아이콘(ICX) 거래지원 종료 안내 (10/19 15:00)"},
            {"listed_at": "2026-09-10T18:30:01+09:00",
             "title": "레이븐코인(RVN) 거래지원 종료 안내 (10/12 15:00)"},
            {"listed_at": "2026-08-28T16:00:00+09:00",
             "title": "신세틱스(SNX) 거래지원 종료 안내 (9/28 15:00)"},
        ],
    },
}

PAGE2 = {
    "success": True,
    "data": {"total_pages": 40, "notices": [
        {"listed_at": "2026-08-10T16:00:00+09:00",
         "title": "코인(OLDX) 거래지원 종료 안내 (9/10 15:00)"},
    ]},
}


def _getter(responses, calls):
    def get_json(url, params, timeout):
        calls.append((url, dict(params), timeout))
        connect, read = timeout
        assert connect <= 2.0 and read <= 3.0
        value = responses[(url, params.get("page"))]
        if isinstance(value, Exception):
            raise value
        return value
    return get_json


@pytest.fixture
def allow_fetch(monkeypatch):
    monkeypatch.delenv(status_mod.FORBID_FETCH_ENV, raising=False)


def test_fetch_combines_warning_and_delisting(allow_fetch):
    calls = []
    responses = {
        (status_mod.MARKET_ALL_URL, None): MARKET_ALL,
        (status_mod.ANNOUNCEMENTS_URL, 1): PAGE1,
        (status_mod.ANNOUNCEMENTS_URL, 2): PAGE2,
    }
    status = status_mod.fetch_market_status(now=NOW, get_json=_getter(responses, calls))
    assert status.available
    assert status.warnings == frozenset({"RVN", "ICX", "BLAST"})
    assert status.delistings["RVN"] == datetime(2026, 10, 12, 15, 0, tzinfo=KST)
    # 1페이지 가장 오래된 공지(8/28)가 45일 lookback(8/20) 안 → 2페이지까지, 그 이상은 X.
    assert [c[1].get("page") for c in calls] == [None, 1, 2]
    assert status.badge("RVN", NOW) == "🚫 거래지원 종료 예정 10/12 — 매수 비권장"
    assert status.badge("ICX", NOW) == "🚫 거래지원 종료 예정 10/19 — 매수 비권장"
    assert status.badge("BLAST", NOW) == "⚠️유의 종목"
    assert status.badge("BAT", NOW) is None
    # 종료 시각이 지난 공지는 표시하지 않는다 (SNX 9/28).
    assert status.badge("SNX", NOW) is None
    assert status.badge("RVN", datetime(2026, 10, 12, 15, 0, tzinfo=KST)) == "⚠️유의 종목"


def test_fetch_stops_paging_at_lookback_or_total_pages(allow_fetch):
    calls = []
    page = copy.deepcopy(PAGE1)
    page["data"]["notices"][-1]["listed_at"] = "2026-08-01T00:00:00+09:00"
    responses = {(status_mod.MARKET_ALL_URL, None): MARKET_ALL,
                 (status_mod.ANNOUNCEMENTS_URL, 1): page}
    status = status_mod.fetch_market_status(now=NOW, get_json=_getter(responses, calls))
    assert status.available and len(calls) == 2


def _recent_page(total_pages: int, days: list[int], fixed_listed: str | None = None) -> dict:
    """NOW 기준 days 일 전 게시 일반 공지 + (선택) 오래된 고정 공지."""
    notices = [{"listed_at": (NOW - timedelta(days=d)).isoformat(),
                "title": f"코인(N{d}) 신규 거래지원 안내"} for d in days]
    fixed = ([{"listed_at": fixed_listed, "title": "고정(FIX) 이용 안내"}]
             if fixed_listed else [])
    return {"success": True, "data": {"total_pages": total_pages,
                                      "fixed_notices": fixed, "notices": notices}}


def test_old_fixed_notice_does_not_stop_paging(allow_fetch):
    calls = []
    responses = {
        (status_mod.MARKET_ALL_URL, None): MARKET_ALL,
        (status_mod.ANNOUNCEMENTS_URL, 1): _recent_page(40, [1, 10, 20],
                                                        "2024-01-01T00:00:00+09:00"),
        (status_mod.ANNOUNCEMENTS_URL, 2): _recent_page(40, [25, 50]),
    }
    status = status_mod.fetch_market_status(now=NOW, get_json=_getter(responses, calls))
    assert [c[1].get("page") for c in calls] == [None, 1, 2]
    assert status.available


def test_short_announcement_coverage_marks_unavailable(allow_fetch):
    # 2페이지를 다 읽어도 가장 오래된 일반 공지가 31일보다 최근 → 확인 불가.
    calls = []
    responses = {
        (status_mod.MARKET_ALL_URL, None): MARKET_ALL,
        (status_mod.ANNOUNCEMENTS_URL, 1): _recent_page(40, [1, 5, 10],
                                                        "2024-01-01T00:00:00+09:00"),
        (status_mod.ANNOUNCEMENTS_URL, 2): _recent_page(40, [15, 20, 30]),
    }
    status = status_mod.fetch_market_status(now=NOW, get_json=_getter(responses, calls))
    assert [c[1].get("page") for c in calls] == [None, 1, 2]
    assert not status.available
    assert any("cover only since" in e for e in status.errors)
    assert status.warnings  # 얻은 유의 정보는 살린다


def test_reading_every_page_is_full_coverage(allow_fetch):
    calls = []
    responses = {
        (status_mod.MARKET_ALL_URL, None): MARKET_ALL,
        (status_mod.ANNOUNCEMENTS_URL, 1): _recent_page(1, [1, 5]),
    }
    status = status_mod.fetch_market_status(now=NOW, get_json=_getter(responses, calls))
    assert status.available and len(calls) == 2


def test_partial_failure_keeps_found_info_but_marks_unavailable(allow_fetch):
    calls = []
    responses = {(status_mod.MARKET_ALL_URL, None): TimeoutError("read timeout"),
                 (status_mod.ANNOUNCEMENTS_URL, 1): PAGE1}
    status = status_mod.fetch_market_status(now=NOW, get_json=_getter(responses, calls),
                                            pages=1)
    assert not status.available
    assert "RVN" in status.delistings and not status.warnings


def test_lookup_is_fail_open_for_every_error(allow_fetch):
    def boom(*_args, **_kwargs):
        raise ConnectionError("dns")
    status = status_mod.lookup_market_status(now=NOW, get_json=boom)
    assert isinstance(status, MarketStatus) and not status.available


def test_lookup_hard_timeout_does_not_wait_for_hung_lookup(allow_fetch):
    def hang(*_args, **_kwargs):
        time.sleep(1.5)
    started = time.monotonic()
    status = status_mod.lookup_market_status(now=NOW, get_json=hang, hard_timeout_s=0.3,
                                             budget_s=1.0)
    assert time.monotonic() - started < 1.2
    assert not status.available and "exceeded" in status.errors[0]


def test_budget_exhaustion_skips_remaining_requests(allow_fetch):
    calls = []
    ticks = iter([0.0, 0.0, 7.8, 7.8, 7.8])
    responses = {(status_mod.MARKET_ALL_URL, None): MARKET_ALL,
                 (status_mod.ANNOUNCEMENTS_URL, 1): PAGE1}
    status = status_mod.fetch_market_status(
        now=NOW, get_json=_getter(responses, calls), monotonic=lambda: next(ticks))
    assert len(calls) == 1 and not status.available


def test_tests_never_touch_real_network_by_default():
    status = status_mod.lookup_market_status(now=NOW)
    assert not status.available and status_mod.FORBID_FETCH_ENV in status.errors[0]


# ==========================================================================
# 메시지 포맷
# ==========================================================================
def _radar(slot="open", coins=("RVN", "BAT", "ICX"), flags=(True, False, True)):
    return {
        "asof": "2026-10-03",
        "slot": slot,
        "btc_regime": "bull_volatile",
        "universe_n": 100,
        "top3": [
            {"coin": f"KRW-{coin}", "rank": rank, "entry_open": 3.46,
             "p_up5": .40, "p_up10": .28, "p_up20": .06, "p_dn5": .27,
             "p_dn10": .08, "exp_downside": -.04, "dump_risk_flag": flag}
            for rank, (coin, flag) in enumerate(zip(coins, flags), start=1)
        ],
    }


STATUS = MarketStatus(
    available=True,
    warnings=frozenset({"RVN", "ICX", "BAT"}),
    delistings={"RVN": datetime(2026, 10, 12, 15, 0, tzinfo=KST),
                "ICX": datetime(2026, 10, 19, 15, 0, tzinfo=KST)},
)


def _utf16_len(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def test_preopen_notice_only_on_preopen():
    pre = sender.format_radar(_radar("preopen"), "preopen")
    opn = sender.format_radar(_radar("open"), "open")
    assert sender.PREOPEN_NOTICE in pre
    assert sender.PREOPEN_NOTICE == (
        "ℹ️ 08:50 목록은 어제 데이터로 뽑아서 어제 09:05 추천과 대부분 같습니다"
        " · 오늘 추천은 09:05에 옵니다")
    # 헤더 바로 아래 한 줄 (BTC 줄 다음).
    lines = pre.splitlines()
    assert lines[lines.index(sender.PREOPEN_NOTICE) - 1].startswith("BTC:")
    assert sender.PREOPEN_NOTICE not in opn


def test_preopen_notice_also_on_empty_candidates():
    res = _radar("preopen")
    res["top3"] = []
    message = sender.format_radar(res, "preopen", market_status=STATUS)
    assert sender.PREOPEN_NOTICE in message
    assert sender.MARKET_STATUS_UNAVAILABLE not in message


def test_volatility_tag_and_legend_replace_dump_risk_wording():
    message = sender.format_radar(_radar(), "open")
    assert "dump_risk" not in message
    assert "사이즈" not in message and "축소" not in message
    lines = message.splitlines()
    assert next(line for line in lines if line.startswith("#1 RVN")).endswith(" ↕️변동큼")
    assert not next(line for line in lines if line.startswith("#2 BAT")).endswith("↕️변동큼")
    assert "• ↕️변동큼 = 오를 때도 빠질 때도 크게 움직이는 종목" in lines


def test_delisting_and_warning_badges_follow_their_coin_without_reordering():
    res = _radar()
    before = copy.deepcopy(res)
    message = sender.format_radar(res, "open", market_status=STATUS, now=NOW)
    assert res == before  # snapshot dict 불변
    lines = message.splitlines()
    i1 = lines.index(next(line for line in lines if line.startswith("#1 RVN")))
    i2 = lines.index(next(line for line in lines if line.startswith("#2 BAT")))
    i3 = lines.index(next(line for line in lines if line.startswith("#3 ICX")))
    assert i1 < i2 < i3
    assert lines[i1 + 1] == "   🚫 거래지원 종료 예정 10/12 — 매수 비권장"
    assert lines[i2 + 1] == "   ⚠️유의 종목"
    assert lines[i3 + 1] == "   🚫 거래지원 종료 예정 10/19 — 매수 비권장"
    assert sender.MARKET_STATUS_UNAVAILABLE not in message


def test_passed_delisting_is_not_shown():
    after = datetime(2026, 10, 20, 9, 5, tzinfo=KST)
    message = sender.format_radar(_radar(), "open", market_status=STATUS, now=after)
    assert "거래지원 종료 예정" not in message
    assert message.count("⚠️유의 종목") == 3


def test_unavailable_status_appends_single_fail_open_line():
    status = MarketStatus.unavailable("ConnectionError: x")
    message = sender.format_radar(_radar(), "open", market_status=status, now=NOW)
    assert message.splitlines()[-1] == "ℹ️ 유의 종목 정보 확인 불가 — 업비트 앱에서 확인"
    assert message.count("확인 불가") == 1
    assert "#1 RVN" in message and "#3 ICX" in message


def test_badge_exception_is_fail_open():
    class Broken(MarketStatus):
        def badge(self, symbol, now):
            raise RuntimeError("bad")
    message = sender.format_radar(_radar(), "open",
                                  market_status=Broken(available=True), now=NOW)
    assert message.splitlines()[-1] == sender.MARKET_STATUS_UNAVAILABLE


def test_no_lookup_means_no_status_lines():
    message = sender.format_radar(_radar(), "open")
    assert "확인 불가" not in message and "거래지원 종료" not in message


@pytest.mark.parametrize("slot", ["open", "preopen"])
def test_worst_case_message_fits_one_telegram_message(slot):
    res = _radar(slot, coins=("ABCDEFGHIJ", "KLMNOPQRST", "UVWXYZABCD"), flags=(True,) * 3)
    status = MarketStatus(
        available=False,
        warnings=frozenset(),
        delistings={c: datetime(2026, 12, 31, 15, 0, tzinfo=KST)
                    for c in ("ABCDEFGHIJ", "KLMNOPQRST", "UVWXYZABCD")},
    )
    message = sender.format_radar(res, slot, dry_run=True,
                                  champion_id="recommend_r1_open", is_fallback=True,
                                  market_status=status, now=NOW)
    assert _utf16_len(message) < 4096


# ==========================================================================
# 발송 경로: 조회 예외가 발송을 막지 않는다
# ==========================================================================
def test_send_path_survives_lookup_crash(monkeypatch):
    res = _radar("open")
    res["model"] = {"id": "recommend_r1_open"}
    sent = []
    spec = type("Spec", (), {"id": "recommend_r1_open",
                             "predict_ref": "signals.recommend:score_candidates"})()
    monkeypatch.setattr(sender, "resolve_champion", lambda *_a, **_k: (spec, False, "t"))
    monkeypatch.setattr(sender, "call_predict", lambda *_a, **_k: res)
    monkeypatch.setattr(sender, "maybe_notify_champion_change", lambda *_a, **_k: None)

    def crash(**_kwargs):
        raise RuntimeError("lookup exploded")

    monkeypatch.setattr(sender, "lookup_market_status", crash)
    monkeypatch.setattr(sender, "send_telegram",
                        lambda message, **_k: sent.append(message) or True)
    res.update(calibration_source="t")
    assert sender.send_recommendation("2026-10-03", "open", dry_run=True)
    assert len(sent) == 1
    assert sent[0].splitlines()[-1] == sender.MARKET_STATUS_UNAVAILABLE


def test_send_path_skips_lookup_without_candidates(monkeypatch):
    res = _radar("open")
    res["top3"] = []
    res.update(model={"id": "recommend_r1_open"}, calibration_source="t")
    spec = type("Spec", (), {"id": "recommend_r1_open",
                             "predict_ref": "signals.recommend:score_candidates"})()
    monkeypatch.setattr(sender, "resolve_champion", lambda *_a, **_k: (spec, False, "t"))
    monkeypatch.setattr(sender, "call_predict", lambda *_a, **_k: res)
    monkeypatch.setattr(sender, "maybe_notify_champion_change", lambda *_a, **_k: None)
    monkeypatch.setattr(sender, "lookup_market_status",
                        lambda **_k: pytest.fail("no lookup without candidates"))
    monkeypatch.setattr(sender, "send_telegram", lambda message, **_k: True)
    assert sender.send_recommendation("2026-10-03", "open", dry_run=True)


class _RecordingLog:
    def __init__(self):
        self.records: list[tuple[str, str]] = []

    def info(self, fmt, *args):
        self.records.append(("info", fmt % args))

    def warning(self, fmt, *args):
        self.records.append(("warning", fmt % args))


@pytest.mark.parametrize("available", [True, False])
def test_lookup_logs_elapsed_ms(monkeypatch, available):
    # 발송 지연(L1 진입봉 영향) 관찰용 — 조회 소요 시간(ms)을 항상 info 로 남긴다.
    recorder = _RecordingLog()
    monkeypatch.setattr(sender, "log", recorder)
    result = STATUS if available else MarketStatus.unavailable("down")
    monkeypatch.setattr(sender, "lookup_market_status", lambda **_k: result)
    assert sender._lookup_market_status() is result
    timing = [m for level, m in recorder.records
              if level == "info" and m.startswith("market status lookup took ")]
    assert len(timing) == 1 and timing[0].endswith(f"ms (available={available})")
