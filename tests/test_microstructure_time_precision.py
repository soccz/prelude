"""Pure clock-conversion regressions; no raw evidence, DB or network writes."""
from __future__ import annotations

import calendar
from datetime import datetime, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo

import pytest

from data.upbit_microstructure import datetime_to_ns, ms_to_iso, ns_to_iso, parse_utc_datetime
from signals.recommend_microstructure import _ns as canonical_ns


def _calendar_ns(value):
    utc = value.astimezone(timezone.utc)
    return calendar.timegm(utc.utctimetuple()) * 1_000_000_000 + utc.microsecond * 1_000


def test_real_20260909_cutoff_preserves_all_original_microseconds():
    text = "2026-09-09T00:08:44.491612+00:00"
    # Observed immutable manifest: 1788912524491611904 (-96 ns). Its ISO
    # field remained correct, so the rounded display hid the integer mismatch.
    expected = 1_788_912_524_491_612_000
    value = parse_utc_datetime(text)
    assert datetime_to_ns(value) == expected == canonical_ns(text)


@pytest.mark.parametrize("microsecond", [0, 1, 2, 7, 99, 123456, 491612, 861916, 999998, 999999])
@pytest.mark.parametrize("offset_minutes", [0, 540, -240, 345, 765])
def test_microseconds_and_fixed_offset_timezones_are_exact(microsecond, offset_minutes):
    value = datetime(2026, 9, 9, 9, 8, 44, microsecond,
                     tzinfo=timezone(timedelta(minutes=offset_minutes)))
    assert datetime_to_ns(value) == _calendar_ns(value)
    assert datetime_to_ns(value) % 1_000 == 0


@pytest.mark.parametrize("text", [
    "1969-12-31T23:59:59.999999+00:00",
    "1969-12-31T23:59:59.000001+00:00",
    "1900-01-02T03:04:05.123457+00:00",
    "1969-12-31T23:59:59.999999+09:00",
    "1970-01-01T00:00:00+00:00",
    "2038-01-19T03:14:07.999999+00:00",
    "2100-01-01T00:00:00.123457+00:00",
    "0001-01-01T00:00:00.000001+00:00",
    "9999-12-31T23:59:59.999999+00:00",
])
def test_epoch_sign_and_full_datetime_range_without_float(text):
    value = datetime.fromisoformat(text)
    assert datetime_to_ns(value) == _calendar_ns(value)


def test_dst_fold_is_a_real_one_hour_difference():
    first = datetime(2026, 11, 1, 1, 30, 0, 123457,
                     tzinfo=ZoneInfo("America/New_York"), fold=0)
    second = first.replace(fold=1)
    assert datetime_to_ns(first) == _calendar_ns(first)
    assert datetime_to_ns(second) == _calendar_ns(second)
    assert datetime_to_ns(second) - datetime_to_ns(first) == 3_600_000_000_000


def test_microsecond_progress_is_monotonic_with_no_quantization():
    start = datetime(2026, 9, 9, 0, 8, 44, tzinfo=timezone.utc)
    base = _calendar_ns(start)
    for microsecond in range(0, 1_000_000, 997):
        assert datetime_to_ns(start.replace(microsecond=microsecond)) == base + microsecond * 1_000


class _NoOffset(tzinfo):
    def utcoffset(self, value):
        return None


@pytest.mark.parametrize("value", [datetime(2026, 9, 9), datetime(2026, 9, 9, tzinfo=_NoOffset())])
def test_naive_datetime_is_rejected_without_local_timezone_assumption(value):
    with pytest.raises(ValueError, match="timezone"):
        datetime_to_ns(value)


def test_timestamp_float_api_is_not_used():
    class IntegerOnlyDatetime(datetime):
        def timestamp(self):
            raise AssertionError("float timestamp path must not be used")

    value = IntegerOnlyDatetime(2026, 9, 9, 0, 8, 44, 491612, tzinfo=timezone.utc)
    assert datetime_to_ns(value) == 1_788_912_524_491_612_000


def test_display_helpers_keep_existing_nullable_microsecond_format():
    # These raw-envelope display helpers are not the authoritative cutoff.
    # Do not change their historical rounding/serialization in this repair.
    assert ns_to_iso(None) is None and ms_to_iso(None) is None
    assert ns_to_iso(1_788_912_524_491_612_000) == "2026-09-09T00:08:44.491612+00:00"
    assert ms_to_iso(1_788_912_524_491) == "2026-09-09T00:08:44.491000+00:00"
    assert parse_utc_datetime("2026-09-09T09:08:44.491612+09:00") == parse_utc_datetime(
        "2026-09-09T00:08:44.491612Z"
    )
    with pytest.raises(ValueError, match="timezone"):
        parse_utc_datetime("2026-09-09T00:08:44.491612")
