"""scripts/record_market_depth.py — 기록 전용 호가 30단 스냅샷 (네트워크 mock)."""
from __future__ import annotations

import gzip
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import scripts.record_market_depth as rec

KST = timezone(timedelta(hours=9))
NOW = datetime(2026, 10, 5, 8, 52, 0, tzinfo=KST)


def _market_all(n_krw: int) -> str:
    rows = [{"market": f"KRW-C{i:03d}", "market_event": {"warning": False}}
            for i in range(n_krw)]
    rows += [{"market": "BTC-C000"}, {"market": "USDT-C001"}]
    return json.dumps(rows)


def _book(market: str) -> dict:
    return {"market": market, "timestamp": 1, "total_ask_size": 1.0,
            "total_bid_size": 1.0, "level": 0,
            "orderbook_units": [{"ask_price": 2, "bid_price": 1, "ask_size": 1,
                                 "bid_size": 1}] * rec.ORDERBOOK_DEPTH}


class FakeHttp:
    def __init__(self, n_krw=250, statuses=None):
        self.n_krw = n_krw
        self.statuses = list(statuses or [])
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url, params, timeout):
        self.calls.append((url, dict(params)))
        assert timeout == rec.TIMEOUT
        if url == rec.MARKET_ALL_URL:
            assert params == {"isDetails": "true"}
            return 200, _market_all(self.n_krw)
        assert url == rec.ORDERBOOK_URL and params["count"] == 30
        status = self.statuses.pop(0) if self.statuses else 200
        if isinstance(status, Exception):
            raise status
        if status != 200:
            return status, '{"error":{"name":"too_many_requests"}}'
        return 200, json.dumps([_book(m) for m in params["markets"].split(",")])


def _run(tmp_path: Path, http) -> tuple[int, Path]:
    code = rec.run(tmp_path, http_get=http, clock=lambda: NOW, sleep=lambda _s: None)
    return code, tmp_path / "2026-10-05" / "0852.json.gz"


def test_complete_snapshot_is_gzip_json_with_hashes(tmp_path):
    http = FakeHttp(n_krw=250)
    code, path = _run(tmp_path, http)
    assert code == 0 and path.exists()
    record = json.loads(gzip.decompress(path.read_bytes()))
    assert record["schema"] == rec.SCHEMA
    assert record["status"] == "complete"
    assert (record["asof"], record["slot_hhmm"]) == ("2026-10-05", "0852")
    assert len(record["krw_markets"]) == 250
    # 1 market/all + ceil(250/100)=3 orderbook 호출
    assert len(http.calls) == 4
    assert [len(e["markets"]) for e in record["orderbook"]] == [100, 100, 50]
    assert sum(e["n_books"] for e in record["orderbook"]) == 250
    for entry in [record["market_all"], *record["orderbook"]]:
        assert entry["fetched_at"] and entry["http_status"] == 200
        assert entry["body_sha256"] == hashlib.sha256(entry["body"].encode()).hexdigest()
    books = json.loads(record["orderbook"][0]["body"])
    assert len(books[0]["orderbook_units"]) == 30
    log = (tmp_path / "record.log").read_text()
    digest = hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest()
    assert f"complete {path} sha256={digest}" in log and "books=250" in log


def test_write_once_never_overwrites_and_skips_network(tmp_path):
    first = FakeHttp(n_krw=5)
    assert _run(tmp_path, first)[0] == 0
    path = tmp_path / "2026-10-05" / "0852.json.gz"
    before = path.read_bytes()
    second = FakeHttp(n_krw=7)
    code, _ = _run(tmp_path, second)
    assert code == 0 and second.calls == []
    assert path.read_bytes() == before
    assert "skip exists" in (tmp_path / "record.log").read_text()


def test_write_once_loses_race_without_overwrite(tmp_path):
    record = rec.collect(http_get=FakeHttp(n_krw=3), clock=lambda: NOW,
                         sleep=lambda _s: None)
    assert rec.write_once(record, tmp_path) is not None
    path = tmp_path / "2026-10-05" / "0852.json.gz"
    before = path.read_bytes()
    record["status"] = "tampered"
    assert rec.write_once(record, tmp_path) is None
    assert path.read_bytes() == before
    assert not list(path.parent.glob(".*tmp*"))


def test_429_is_recorded_without_retry_and_stops_remaining_requests(tmp_path):
    http = FakeHttp(n_krw=250, statuses=[200, 429])
    code, path = _run(tmp_path, http)
    assert code == 1
    record = json.loads(gzip.decompress(path.read_bytes()))
    assert record["status"] == "partial"
    assert [e["http_status"] for e in record["orderbook"]] == [200, 429]
    assert len(http.calls) == 3  # market/all + 200 + 429, 세 번째 묶음·재시도 없음
    assert any("429" in err for err in record["errors"])


@pytest.mark.parametrize("status", [418, 500])
def test_any_http_error_stops_remaining_requests(tmp_path, status):
    # 418(IP 차단)·5xx 도 429 와 같이 남은 묶음을 요청하지 않는다.
    http = FakeHttp(n_krw=250, statuses=[status, 200, 200])
    code, path = _run(tmp_path, http)
    assert code == 1
    record = json.loads(gzip.decompress(path.read_bytes()))
    assert record["status"] == "failed"
    assert [e["http_status"] for e in record["orderbook"]] == [status]
    assert len(http.calls) == 2  # market/all + 첫 묶음, 나머지 묶음·재시도 없음
    assert any(f"HTTP {status} — remaining" in err for err in record["errors"])


def test_transport_failure_is_recorded_not_raised(tmp_path):
    http = FakeHttp(n_krw=50, statuses=[TimeoutError("read timeout")])
    code, path = _run(tmp_path, http)
    assert code == 1
    record = json.loads(gzip.decompress(path.read_bytes()))
    assert record["status"] == "failed"
    assert "TimeoutError" in record["orderbook"][0]["error"]


def test_market_all_failure_writes_failed_record(tmp_path):
    def down(url, params, timeout):
        raise ConnectionError("dns")
    code, path = _run(tmp_path, down)
    assert code == 1
    record = json.loads(gzip.decompress(path.read_bytes()))
    assert record["status"] == "failed" and record["orderbook"] == []
    assert "ConnectionError" in record["market_all"]["error"]


def test_recorder_has_no_db_or_telegram_or_score_dependencies():
    source = Path(rec.__file__).read_text()
    for forbidden in ("sqlite3", "notifier", "telegram.", "signals.", "ledger.",
                      "data.database", "send_telegram"):
        assert forbidden not in source
