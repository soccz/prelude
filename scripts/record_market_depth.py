#!/usr/bin/env python3
"""업비트 KRW 전 종목 호가 30단 + market/all 원문 — 기록 전용(record-only) 스냅샷.

매일 08:52 / 09:07 KST (deploy/prelude-depth-record.timer) 에 1회씩 실행해
output/depth_snapshots/YYYY-MM-DD/<HHMM>.json.gz 에 한 번만 쓴다.

★ 기록 전용 — 추천·snapshot·receipt·원장·DB·텔레그램 어디에도 쓰지 않는다.
  실패해도 운영 파이프라인에 영향 없음 (systemd OnFailure 경보도 붙이지 않음).
★ 공개 GET 만 (API key·주문 없음). 호출 수 = market/all 1 + orderbook ⌈N/100⌉.
★ write-once: 같은 경로가 이미 있으면 덮어쓰지 않고 종료 (os.link 원자 생성).
★ HTTP ≥400(429 rate limit·418 IP 차단 포함) 은 재시도하지 않고 실패로 기록,
  남은 호가 요청도 중단.

파일 내용(gzip JSON):
  schema, asof, slot_hhmm, started_at, completed_at, status(complete|partial|failed),
  market_all  : {url, requested_at, fetched_at, http_status, body_sha256, body, error}
  orderbook[] : 같은 형식 + markets (요청 묶음)
body 는 응답 원문 텍스트 그대로(재파싱 가능), body_sha256 은 그 UTF-8 바이트 해시.

수동 실행 (기본 out-root = output/depth_snapshots):
    python scripts/record_market_depth.py --out-root /home/soccz/22tb/tmp/depth_test
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_ROOT = ROOT / "output" / "depth_snapshots"
KST = timezone(timedelta(hours=9))

SCHEMA = "upbit_depth_snapshot.v1"
MARKET_ALL_URL = "https://api.upbit.com/v1/market/all"
ORDERBOOK_URL = "https://api.upbit.com/v1/orderbook"
# 초기값(placeholder) — 관측 후 조정.
ORDERBOOK_DEPTH = 30        # REST count=30 → 30단 (2026-10-04 실측 확인)
MARKETS_PER_REQUEST = 100   # URL 길이·요청 수 균형
REQUEST_GAP_S = 0.2         # orderbook 그룹 초당 한도(≈10/s) 대비 여유
TIMEOUT = (2.0, 5.0)        # (connect, read)

# http_get(url, params, timeout) -> (status_code, text). 예외 = 전송 실패.
HttpGet = Callable[[str, dict, tuple[float, float]], tuple[int, str]]


def _default_http_get(url: str, params: dict, timeout: tuple[float, float]) -> tuple[int, str]:
    import requests

    response = requests.get(
        url, params=params, timeout=timeout, headers={"Accept": "application/json"}
    )
    return response.status_code, response.text


def _iso(value: datetime) -> str:
    return value.astimezone(KST).isoformat(timespec="milliseconds")


def _fetch(http_get: HttpGet, url: str, params: dict,
           clock: Callable[[], datetime]) -> dict[str, Any]:
    """한 번의 GET 결과를 기록용 dict 로. 예외는 error 필드로 흡수."""
    entry: dict[str, Any] = {
        "url": url,
        "params": params,
        "requested_at": _iso(clock()),
        "fetched_at": None,
        "http_status": None,
        "body_sha256": None,
        "body": None,
        "error": None,
    }
    try:
        status, text = http_get(url, params, TIMEOUT)
    except Exception as exc:  # noqa: BLE001 — 기록 전용, 실패도 기록
        entry["error"] = f"{type(exc).__name__}: {exc}"
        return entry
    entry["fetched_at"] = _iso(clock())
    entry["http_status"] = int(status)
    entry["body"] = text
    entry["body_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if status != 200:
        entry["error"] = f"HTTP {status}"
    return entry


def _krw_markets(market_all_body: str) -> list[str]:
    payload = json.loads(market_all_body)
    if not isinstance(payload, list):
        raise ValueError("market/all payload must be a list")
    return sorted(
        row["market"] for row in payload
        if isinstance(row, dict) and str(row.get("market", "")).startswith("KRW-")
    )


def collect(
    *,
    http_get: HttpGet = _default_http_get,
    clock: Callable[[], datetime] = lambda: datetime.now(KST),
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """네트워크 조회만 (파일 쓰기 없음). 결과 record dict 반환."""
    started = clock()
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "asof": started.astimezone(KST).strftime("%Y-%m-%d"),
        "slot_hhmm": started.astimezone(KST).strftime("%H%M"),
        "started_at": _iso(started),
        "completed_at": None,
        "status": "failed",
        "orderbook_depth": ORDERBOOK_DEPTH,
        "krw_markets": [],
        "market_all": None,
        "orderbook": [],
        "errors": [],
    }
    market_all = _fetch(http_get, MARKET_ALL_URL, {"isDetails": "true"}, clock)
    record["market_all"] = market_all
    markets: list[str] = []
    if market_all["error"] is None:
        try:
            markets = _krw_markets(market_all["body"])
        except (ValueError, KeyError, TypeError) as exc:
            record["errors"].append(f"market_all parse: {type(exc).__name__}: {exc}")
    else:
        record["errors"].append(f"market_all: {market_all['error']}")
    record["krw_markets"] = markets

    books_ok = bool(markets)
    for index in range(0, len(markets), MARKETS_PER_REQUEST):
        if index:
            sleep(REQUEST_GAP_S)
        chunk = markets[index:index + MARKETS_PER_REQUEST]
        entry = _fetch(
            http_get, ORDERBOOK_URL,
            {"markets": ",".join(chunk), "count": ORDERBOOK_DEPTH}, clock,
        )
        entry["markets"] = chunk
        entry["n_books"] = 0
        if entry["error"] is None:
            try:
                books = json.loads(entry["body"])
                if not isinstance(books, list):
                    raise ValueError("orderbook payload must be a list")
                entry["n_books"] = len(books)
            except ValueError as exc:
                entry["error"] = f"parse: {exc}"
        record["orderbook"].append(entry)
        if entry["error"] is not None:
            books_ok = False
            record["errors"].append(f"orderbook[{index // MARKETS_PER_REQUEST}]: {entry['error']}")
            status = entry["http_status"]
            if status is not None and status >= 400:
                # 재시도 없음 — 남은 묶음도 요청하지 않는다. 429 뒤에 계속 요청하면
                # 418(IP 차단)으로 올라가 같은 IP 의 캔들 수집까지 막힐 수 있다.
                record["errors"].append(
                    f"HTTP {status} — remaining orderbook requests skipped")
                break

    if market_all["error"] is None and books_ok and not record["errors"]:
        record["status"] = "complete"
    elif record["orderbook"] and any(e["error"] is None for e in record["orderbook"]):
        record["status"] = "partial"
    record["completed_at"] = _iso(clock())
    return record


def write_once(record: dict[str, Any], out_root: Path) -> tuple[Path, str] | None:
    """gzip JSON 을 원자적으로 1회 생성. 이미 있으면 None (덮어쓰지 않음).
    반환 = (경로, 비압축 JSON sha256)."""
    day_dir = out_root / record["asof"]
    day_dir.mkdir(parents=True, exist_ok=True)
    final = day_dir / f"{record['slot_hhmm']}.json.gz"
    if final.exists():
        return None
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    blob = gzip.compress(payload, mtime=0)
    tmp = day_dir / f".{final.name}.tmp-{os.getpid()}"
    try:
        with open(tmp, "xb") as fh:
            fh.write(blob)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(tmp, final)  # 원자적 no-overwrite 생성
        except FileExistsError:
            return None
        dir_fd = os.open(day_dir, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        tmp.unlink(missing_ok=True)
    return final, digest


def _log(out_root: Path, line: str) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(KST).isoformat(timespec="seconds")
    with open(out_root / "record.log", "a", encoding="utf-8") as fh:
        fh.write(f"{stamp} {line}\n")
    print(line)


def run(
    out_root: Path,
    *,
    http_get: HttpGet = _default_http_get,
    clock: Callable[[], datetime] = lambda: datetime.now(KST),
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """0 = complete 저장 또는 이미 존재(skip) / 1 = partial·failed 저장 / 2 = 내부 오류."""
    now = clock().astimezone(KST)
    target = out_root / now.strftime("%Y-%m-%d") / f"{now.strftime('%H%M')}.json.gz"
    if target.exists():
        # 같은 분에 재실행 → 호출 없이 종료 (write-once).
        _log(out_root, f"skip exists {target}")
        return 0
    try:
        record = collect(http_get=http_get, clock=clock, sleep=sleep)
        written = write_once(record, out_root)
    except Exception as exc:  # noqa: BLE001 — 기록 전용, 운영 영향 없음
        _log(out_root, f"error {type(exc).__name__}: {exc}")
        return 2
    if written is None:
        _log(out_root, f"skip exists {target} (concurrent writer)")
        return 0
    path, digest = written
    n_books = sum(int(e.get("n_books") or 0) for e in record["orderbook"])
    errors = "; ".join(record["errors"]) or "-"
    _log(out_root, (
        f"{record['status']} {path} sha256={digest} "
        f"krw_markets={len(record['krw_markets'])} books={n_books} "
        f"calls={1 + len(record['orderbook'])} errors={errors}"
    ))
    return 0 if record["status"] == "complete" else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="업비트 KRW 호가 30단 기록 전용 스냅샷")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT,
                    help="기본 output/depth_snapshots")
    args = ap.parse_args()
    sys.exit(run(args.out_root))


if __name__ == "__main__":
    main()
