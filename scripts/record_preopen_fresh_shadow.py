#!/usr/bin/env python3
"""08:50 그림자 기록 — 진행 중 어제(D-1) 일봉을 반영한 R1 open-방식 점수 (record-only).

목적: 지금 08:50 알림은 어제 마감 데이터로 뽑아 사실상 어제 09:05 Top3 와 같다.
08:50 preopen 단계의 ``collector_d1 --update`` (08:53경 완료) 가 진행 중인 D-1
일봉을 거의 끝까지 채워 넣으므로, 그 일봉을 반영해 R1 을 **open 슬롯 방식**
(D 자리 더미 행 → D 행 feature = shift(1) = 진행 중 D-1) 으로 계산하면 같은 날
09:05 R1 에 가까운 목록을 08:5x 에 얻을 수 있다. 이 스크립트는 그 목록을 매일
1회 기록만 해 두고, 4~8주 뒤 scripts/evaluate_preopen_fresh_shadow.py 로 같은 날
09:05 R1 과 비교한다 (연구 재현: tmp/prelude_research/fresh-preopen, 2026-10-04).

★ 기록 전용 — 텔레그램·원장·recommend_snapshots·receipts·라이브 DB 쓰기 없음.
  출력은 output/preopen_fresh_shadow/YYYY-MM-DD.json (write-once) 와 record.log 뿐.
★ 봉인 R1 소스(signals/recommend.py 등 9개)는 수정하지 않는다. 라이브
  data/upbit_d1.db 를 read-only 로 열어 sqlite backup API 로 임시 복사하고,
  복사본에만 D 더미 행을 넣은 뒤 런타임에서만 DB_PATH 를 복사본으로 지정한다.
★ 신선도 가드: 오늘 cron_preopen 로그에서 D1 갱신 완료(08:45 이후, 요약 줄
  'missing=none failed=none', 다음 단계 헤더 '[2/6]' 까지)를 확인하고, 복사본에서
  D-1 행 존재·D 행 부재를 확인한다. 하나라도 못 하면 계산 없이 status=stale_input.
★ R1 preopen D1 health gate ('[2/6]' 단계) 결과는 기록만 한다 (계산·status 무관):
  freshness.cron_log.health_gate = 'failed' (블록 안에 'R1 preopen D1 health gate
  failed' critical 줄) / 'passed' (critical 줄 없이 다음 단계 헤더 '[3/6]') /
  'unknown' (아직 판단 불가 — 점수 계산 뒤 한 번 더 읽는다). 그날 라이브 08:50 은
  gate 실패면 발송되지 않으므로, 평가 스크립트는 health_gate=='passed' 만 쓴다.
★ 마감 가드: asof 08:59:30 KST 까지 끝나지 않으면 중단하고 status=failed
  (09:05 운영과 경합 방지). systemd 정지(SIGTERM) 도 failed 로 기록한다.
  ShadowAbort 는 BaseException 하위라 봉인 scorer 안의 'except Exception' 에
  삼켜지지 않고, 혹시 삼켜져도 신호 플래그가 남아 있으면 failed 로 기록한다.
  service TimeoutStartSec(420s) 는 이 마감보다 확실히 늦다 — 스크립트 마감이 먼저.
★ rank_basis 가 'R1_riskreward' 로 시작하지 않으면(대체 정렬 fallback)
  status=degraded — 평가 스크립트는 status=ok 만 쓴다.
★ 임시 DB 복사본은 라이브 output 이 아니라 DEFAULT_WORK_ROOT
  (/home/soccz/22tb/tmp/prelude_shadow_work) 아래에 만들고, 시작 시 1시간 넘은
  잔여 작업 디렉토리(.work-*, SIGKILL 잔여물)를 지운다 (수동 실행이 1시간을 넘지
  않는다는 전제).

종료 코드: 0 = ok 저장 또는 이미 존재(skip) / 1 = stale_input·degraded·failed 저장
/ 2 = 내부 오류.

수동 실행 (출력 경로만 바꿔 시험):
    python scripts/record_preopen_fresh_shadow.py \\
        --out-root /home/soccz/22tb/tmp/prelude_impl/shadow_trial \\
        --work-root /home/soccz/22tb/tmp/prelude_impl/shadow_work
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import sqlite3
import sys
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from datetime import time as dtime
from pathlib import Path
from typing import Any, Callable, Iterator

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUT_ROOT = ROOT / "output" / "preopen_fresh_shadow"
DEFAULT_DB = ROOT / "data" / "upbit_d1.db"
DEFAULT_LOG_DIR = ROOT / "output"
# 임시 DB 복사본(약 37MB) 작업 공간 — 라이브 output/ 밖 (전역 규칙: /tmp 금지 → 22tb/tmp).
DEFAULT_WORK_ROOT = Path("/home/soccz/22tb/tmp/prelude_shadow_work")
WORK_PREFIX = ".work-"
# 이보다 오래된 .work-* 는 시작 시 정리 (SIGKILL 잔여물). 서비스 최대 수명은
# TimeoutStartSec 420s + TimeoutStopSec 15s — 수동 실행도 1시간을 넘지 않는다는 전제.
STALE_WORK_AGE_S = 3600.0
KST = timezone(timedelta(hours=9))

SCHEMA = "preopen_fresh_shadow.v1"
# 초기값(placeholder) — 관측 후 조정.
FRESH_AFTER = dtime(8, 45)        # D1 갱신 완료 시각 하한 (그 이후 체결까지 반영)
WAIT_D1_UNTIL = dtime(8, 58)      # D1 단계가 아직 진행 중이면 이 시각까지만 기다림
WAIT_POLL_S = 15.0
# 이 시각까지 못 끝내면 failed (09:05 운영과 경합 방지). 08:56 timer + service
# TimeoutStartSec=420 (→ 09:03 SIGTERM) 보다 확실히 앞서야 한다 (계약 테스트로 고정).
DEADLINE = dtime(8, 59, 30)
MIN_DM1_COVERAGE = 0.95           # D-2 행이 있던 종목 중 D-1 행이 있어야 하는 비율
BTC_MARKET = "KRW-BTC"
R1_RANK_BASIS_PREFIX = "R1_riskreward"   # 이 접두가 아니면 대체 정렬 → degraded

log = logging.getLogger("preopen_fresh_shadow")


class ShadowAbort(BaseException):
    """마감 초과·정지 신호 — 계산을 중단하고 failed 로 기록.

    BaseException 하위: 봉인 scorer(signals/recommend.py) 안의 ``except Exception``
    (RR·A1 head 학습 감쌈) 에 잡혀 사라지지 않게 한다. 그래도 bare except 등으로
    삼켜질 수 있으므로 신호 핸들러는 플래그도 세운다 (build_record 가 확인).
    """


# --------------------------------------------------------------------------
# 1. 신선도 가드 — 오늘 cron_preopen 로그
# --------------------------------------------------------------------------
_HEADER_RE = re.compile(
    r"^=== prelude pre-open trigger KST (\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) ===$")
_D1_DONE_RE = re.compile(
    r"^(\d{2}:\d{2}:\d{2}) \[INFO\] D1 universe after update: (.*)$")
_UPDATED_RE = re.compile(r"^Updated (\d+) markets$")
_D1_CRITICAL = "[critical] preopen D1 universe update failed"
# collector_d1 요약 줄 꼬리 — 갱신 누락·실패 종목이 없을 때만 'none'.
_D1_CLEAN_RE = re.compile(r"\bmissing=none failed=none$")
# daily_run_preopen.sh 다음 단계 헤더. D1 critical 줄은 항상 이 헤더 앞에 쓰인다.
_NEXT_STEP = "[2/6] "
# '[2/6]' R1 preopen D1 health gate 실패 줄 — 항상 '[3/6]' 헤더 앞에 쓰인다.
_HEALTH_CRITICAL = "[critical] R1 preopen D1 health gate failed"
_HEALTH_DONE_STEP = "[3/6] "
HEALTH_GATE_VALUES = ("passed", "failed", "unknown")


def cron_log_path(log_dir: Path, asof: date) -> Path:
    return log_dir / f"cron_preopen_{asof.strftime('%Y%m%d')}.log"


def check_cron_log(log_path: Path, asof: date) -> dict[str, Any]:
    """오늘 preopen 실행 블록(마지막 헤더 이후)에서 D1 갱신 완료를 찾는다.

    ok=True 조건: 헤더 날짜 == asof, 'D1 universe after update' 시각 >= 08:45 이고
    그 요약이 'missing=none failed=none', 'Updated N markets' 존재, 다음 단계 헤더
    '[2/6]' 존재(= collector 종료 뒤 bash 의 critical 줄 기록 기회가 지나감), D1
    critical 실패 줄 없음. pending=True 는 블록은 있는데 D1 단계가 아직 안 끝난
    상태(기다릴 가치가 있음).

    health_gate (기록 전용, ok 판정과 무관): D1 갱신이 끝난 블록에서 'R1 preopen D1
    health gate failed' 줄이 있으면 'failed', 없이 '[3/6]' 헤더가 나왔으면 'passed',
    그 밖(gate 진행 중·D1 단계 실패로 gate 생략 등)은 'unknown'.
    """
    out: dict[str, Any] = {
        "path": str(log_path), "ok": False, "pending": False, "reason": None,
        "run_started": None, "d1_completed": None, "d1_summary": None,
        "updated_markets": None, "next_step_seen": False, "health_gate": "unknown",
    }
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        out["reason"] = "cron_preopen log missing"
        return out
    except OSError as exc:
        out["reason"] = f"cron_preopen log unreadable: {type(exc).__name__}"
        return out
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        m = _HEADER_RE.match(line.strip())
        if m and m.group(1) == asof.isoformat():
            start = i
            out["run_started"] = m.group(2)
    if start is None:
        out["reason"] = f"no pre-open run header for {asof}"
        return out
    block = lines[start + 1:]
    if any(_D1_CRITICAL in line for line in block):
        out["reason"] = "D1 update failed in today's pre-open run"
        return out
    out["health_gate"] = _health_gate_state(block)
    for line in block:
        m = _D1_DONE_RE.match(line.strip())
        if m:
            out["d1_completed"] = m.group(1)
            out["d1_summary"] = m.group(2)
        m = _UPDATED_RE.match(line.strip())
        if m:
            out["updated_markets"] = int(m.group(1))
        if line.startswith(_NEXT_STEP):
            out["next_step_seen"] = True
    if out["d1_summary"] is not None and not _D1_CLEAN_RE.search(out["d1_summary"].strip()):
        out["reason"] = f"D1 update incomplete: {out['d1_summary'].strip()}"
        return out
    if (out["d1_completed"] is None or out["updated_markets"] is None
            or not out["next_step_seen"]):
        out["pending"] = True
        out["reason"] = "D1 update not completed yet in today's pre-open run"
        return out
    done = dtime.fromisoformat(out["d1_completed"])
    if done < FRESH_AFTER:
        out["reason"] = f"D1 update completed at {out['d1_completed']} (< {FRESH_AFTER:%H:%M})"
        return out
    out["ok"] = True
    return out


def _health_gate_state(block: list[str]) -> str:
    """'[2/6]' R1 preopen D1 health gate 결과 — passed / failed / unknown.

    daily_run_preopen.sh 는 gate 가 끝난 뒤에야 '[3/6]' 를 쓰고, 실패면 그 전에
    critical 줄을 쓴다. 그래서 '[3/6]' 가 보이고 critical 줄이 없으면 통과다.
    호출 전제: D1 단계 critical 줄이 없는 블록 (있으면 gate 자체가 생략된다).
    """
    if any(_HEALTH_CRITICAL in line for line in block):
        return "failed"
    if any(line.startswith(_HEALTH_DONE_STEP) for line in block):
        return "passed"
    return "unknown"


def wait_for_cron_log(
    log_path: Path, asof: date, *,
    clock: Callable[[], datetime], sleep: Callable[[float], None],
) -> dict[str, Any]:
    """D1 단계가 진행 중(pending)이면 WAIT_D1_UNTIL 까지 짧게 기다린다."""
    limit = datetime.combine(asof, WAIT_D1_UNTIL, tzinfo=KST)
    while True:
        result = check_cron_log(log_path, asof)
        if result["ok"] or not result["pending"]:
            return result
        now = clock().astimezone(KST)
        if now >= limit:
            return result
        sleep(min(WAIT_POLL_S, max(0.0, (limit - now).total_seconds())))


# --------------------------------------------------------------------------
# 2. DB 준비 — read-only 원본 → backup 복사본 → 검사 → D 더미 행
# --------------------------------------------------------------------------
def _bar_ts(day: date) -> str:
    """업비트 KRW 일봉 시작 시각 (DB 표기: 'YYYY-MM-DD 09:00:00')."""
    return f"{day.isoformat()} 09:00:00"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cleanup_stale_work(work_root: Path, now_ts: float) -> list[str]:
    """work_root 아래 STALE_WORK_AGE_S(1시간) 보다 오래된 .work-* 디렉토리를 지운다.

    SIGKILL(TimeoutStopSec 초과) 이면 TemporaryDirectory 정리가 안 돼 DB 복사본이
    남는다. 기준 1시간 = 서비스 최대 수명(420s + 15s)보다 충분히 길고, 다음날 08:56
    실행이 전날 잔여물을 반드시 지운다. 전제: 같은 work_root 를 쓰는 수동 실행이
    1시간을 넘지 않는다 (넘으면 그 실행의 복사본이 다른 실행에 지워질 수 있다).
    1시간 미만 항목·심볼릭 링크·일반 파일은 건드리지 않는다. 실패는 기록 계산을
    막지 않는다.
    """
    removed: list[str] = []
    try:
        entries = list(work_root.iterdir())
    except OSError:
        return removed
    for entry in entries:
        if not entry.name.startswith(WORK_PREFIX):
            continue
        try:
            st = entry.lstat()
        except OSError:
            continue
        if entry.is_symlink() or not entry.is_dir():
            continue
        if now_ts - st.st_mtime <= STALE_WORK_AGE_S:
            continue
        shutil.rmtree(entry, ignore_errors=True)
        if not entry.exists():
            removed.append(entry.name)
    return removed


def backup_copy(src: Path, dst: Path) -> None:
    """원본을 read-only 로 열어 sqlite backup API 로 dst 에 복사 (원본 무변경)."""
    from data.database import connect_readonly

    if dst.exists():
        raise FileExistsError(f"backup destination already exists: {dst}")
    with connect_readonly(src) as source:
        target = sqlite3.connect(dst)
        try:
            source.backup(target)
        finally:
            target.close()


def inspect_copy(db: Path, asof: date) -> dict[str, Any]:
    """복사본에서 D-1 진행 중 일봉 존재와 D 행 부재를 확인한다."""
    d1 = _bar_ts(asof - timedelta(days=1))
    d2 = _bar_ts(asof - timedelta(days=2))
    d0 = _bar_ts(asof)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        dm1 = {r[0] for r in con.execute(
            "SELECT market FROM candles WHERE timestamp = ?", (d1,))}
        dm2 = {r[0] for r in con.execute(
            "SELECT market FROM candles WHERE timestamp = ?", (d2,))}
        n_future = con.execute(
            "SELECT COUNT(*) FROM candles WHERE timestamp >= ?", (d0,)).fetchone()[0]
        n_rows = con.execute("SELECT COUNT(*) FROM candles").fetchone()[0]
    finally:
        con.close()
    covered = len(dm1 & dm2)
    coverage = covered / len(dm2) if dm2 else 0.0
    out: dict[str, Any] = {
        "rows": int(n_rows),
        "dm1_bar": d1,
        "n_markets_dm1": len(dm1),
        "n_markets_dm2": len(dm2),
        "dm1_coverage_of_dm2": round(coverage, 4),
        "missing_dm1_examples": sorted(dm2 - dm1)[:10],
        "n_rows_at_or_after_asof": int(n_future),
        "btc_has_dm1": BTC_MARKET in dm1,
        "ok": False,
        "reason": None,
    }
    if n_future:
        out["reason"] = f"{n_future} row(s) at/after {d0} already present (post-open DB)"
    elif BTC_MARKET not in dm1:
        out["reason"] = f"{BTC_MARKET} has no {d1} bar"
    elif coverage < MIN_DM1_COVERAGE:
        out["reason"] = (f"D-1 bar coverage {coverage:.3f} < {MIN_DM1_COVERAGE} "
                         f"({covered}/{len(dm2)})")
    else:
        out["ok"] = True
    return out


def add_dummy_asof_rows(db: Path, asof: date) -> int:
    """D-1 행이 있는 종목마다 D 09:00 더미 행을 넣는다 (복사본 전용).

    더미 = D-1 진행 중 종가 고정·거래량 0. open 슬롯은 D 행의 feature 를
    shift(1) = D-1 로 만들므로 더미 값 자체는 feature 에 들어가지 않는다.
    (D 행 라벨은 train cutoff 밖이라 학습에도 안 섞인다.)
    """
    d1 = _bar_ts(asof - timedelta(days=1))
    d0 = _bar_ts(asof)
    con = sqlite3.connect(db)
    try:
        with con:
            cur = con.execute(
                "INSERT INTO candles (market, timestamp, open, high, low, close, "
                "volume, quote_volume) "
                "SELECT market, ?, close, close, close, close, 0.0, 0.0 "
                "FROM candles WHERE timestamp = ?",
                (d0, d1),
            )
            inserted = cur.rowcount
    finally:
        con.close()
    return int(inserted)


# --------------------------------------------------------------------------
# 3. 점수 — 봉인 score_candidates 를 복사본 DB 로 (런타임 경로 지정만)
# --------------------------------------------------------------------------
@contextlib.contextmanager
def _scorer_db(db: Path) -> Iterator[Any]:
    import scripts.regime_split_precursor_v1 as rsp
    import signals.recommend as rec

    saved = (rec.DB_PATH, rsp.DB_PATH)
    rec.DB_PATH = str(db)
    rsp.DB_PATH = str(db)
    try:
        yield rec
    finally:
        rec.DB_PATH, rsp.DB_PATH = saved


def score_copy(db: Path, asof: date) -> dict:
    with _scorer_db(db) as rec:
        return rec.score_candidates(asof.isoformat(), slot="open", ranking="R1")


def score_source_sha256() -> str | None:
    try:
        from signals.recommend_snapshot import _source_sha256
        return _source_sha256()
    except Exception as exc:  # noqa: BLE001 — 기록 전용
        log.warning("score_source_sha256 unavailable: %s", exc)
        return None


def _top_item(item: dict) -> dict:
    keys = ("coin", "rank", "rr_ratio", "p_up5", "p_up10", "p_up20",
            "p_dn5", "p_dn10", "exp_downside", "dump_risk_flag")
    return {k: item.get(k) for k in keys}


def summarize_scores(res: dict) -> dict:
    universe = res.get("universe") or []
    return {
        "feature_date": res.get("feature_date"),
        "slot": res.get("slot"),
        "ranking": res.get("ranking"),
        "rank_basis": res.get("rank_basis"),
        "rule_version": res.get("rule_version"),
        "btc_regime": res.get("btc_regime"),
        "universe_n": res.get("universe_n"),
        "training": res.get("training"),
        "top3": [_top_item(u) for u in (res.get("top3") or [])],
        # 후보 순위 전체: [coin, rank, rr_ratio, p_up10, p_dn5]
        "candidates": [
            [u.get("coin"), u.get("rank"), u.get("rr_ratio"), u.get("p_up10"), u.get("p_dn5")]
            for u in universe
        ],
    }


# --------------------------------------------------------------------------
# 4. 기록 — write-once
# --------------------------------------------------------------------------
def record_path(out_root: Path, asof: date) -> Path:
    return out_root / f"{asof.isoformat()}.json"


def write_once(record: dict, out_root: Path) -> Path | None:
    """JSON 을 원자적으로 1회 생성. 이미 있으면 None (덮어쓰지 않음)."""
    out_root.mkdir(parents=True, exist_ok=True)
    final = record_path(out_root, date.fromisoformat(record["asof"]))
    if final.exists():
        return None
    payload = (json.dumps(record, ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode()
    tmp = out_root / f".{final.name}.tmp-{os.getpid()}"
    try:
        with open(tmp, "xb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(tmp, final)  # 원자적 no-overwrite 생성
        except FileExistsError:
            return None
        dir_fd = os.open(out_root, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        tmp.unlink(missing_ok=True)
    return final


def _log_line(out_root: Path, line: str) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(KST).isoformat(timespec="seconds")
    with open(out_root / "record.log", "a", encoding="utf-8") as fh:
        fh.write(f"{stamp} {line}\n")
    print(line)


def _iso(value: datetime) -> str:
    return value.astimezone(KST).isoformat(timespec="milliseconds")


@contextlib.contextmanager
def _abort_guards(
    deadline: datetime, clock: Callable[[], datetime], abort: dict[str, Any],
) -> Iterator[None]:
    """마감(SIGALRM)·정지(SIGTERM) 시 플래그를 세우고 ShadowAbort 를 던진다.

    메인 스레드 전용. 첫 신호의 사유만 ``abort["reason"]`` 에 남긴다 — scorer 가
    예외를 삼켜도 호출자가 이 플래그로 failed 를 판정한다.

    원복 경쟁 방어: 빠져나올 때 먼저 ``closing`` 을 세워 핸들러가 더는 던지지 않고
    플래그만 남기게 하고, 원복 자체는 중첩 finally 로 항상 실행한다. 원복 동안은
    SIGALRM·SIGTERM 을 막아 두고(pthread_sigmask), 그 사이 도착해 대기 중인 신호는
    sigtimedwait 로 소비해 플래그에 반영한다 — 원복된 기본 핸들러(SIG_DFL)로 늦게
    배달돼 기록 없이 죽는 일이 없다. 마스크는 진입 전 상태로 되돌린다.
    """
    sigs = {signal.SIGALRM, signal.SIGTERM}
    reasons = {signal.SIGALRM: f"deadline {deadline:%H:%M:%S} KST reached",
               signal.SIGTERM: "terminated (SIGTERM)"}
    closing = False

    def _note(signum: int) -> None:
        if abort.get("reason") is None:
            abort["reason"] = reasons[signum]

    def _handler(signum, frame):  # noqa: ARG001
        _note(signum)
        if not closing:
            raise ShadowAbort(abort["reason"])

    def _restore() -> None:
        old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, sigs)
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_alarm)
            signal.signal(signal.SIGTERM, old_term)
            for signum in sorted(sigs & signal.sigpending()):
                if signal.sigtimedwait({signum}, 0) is not None:
                    _note(signum)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)

    remaining = (deadline - clock()).total_seconds()
    old_alarm = signal.signal(signal.SIGALRM, _handler)
    old_term = signal.signal(signal.SIGTERM, _handler)
    signal.setitimer(signal.ITIMER_REAL, max(remaining, 0.001))
    try:
        yield
    finally:
        try:
            closing = True
        finally:
            _restore()


def _base_record(asof: date, *, db: Path, started: datetime) -> dict[str, Any]:
    """status=failed 로 시작하는 기록 골격 (성공 경로에서 채워 나간다)."""
    return {
        "schema": SCHEMA,
        "asof": asof.isoformat(),
        "created_at": None,
        "started_at": _iso(started),
        "deadline": _iso(datetime.combine(asof, DEADLINE, tzinfo=KST)),
        "status": "failed",
        "reason": None,
        "method": ("R1 open-slot on temp DB copy: in-progress D-1 bar as of the "
                   "08:50 pre-open D1 update + dummy D 09:00 rows"),
        "record_only": True,
        "source_db": str(db),
        "freshness": {"cron_log": None, "db_copy": None, "source_db_mtime": None},
        "copy_db_sha256": None,
        "prepared_db_sha256": None,
        "dummy_rows_inserted": None,
        "score_source_sha256": None,
        "result": None,
        "timing": {"score_s": None, "elapsed_s": None},
    }


def build_record(
    asof: date, *, db: Path, log_dir: Path,
    clock: Callable[[], datetime], sleep: Callable[[float], None],
    scorer: Callable[[Path, date], dict] = score_copy,
    work_root: Path,
) -> dict[str, Any]:
    """신선도 확인 → 복사 → 더미 → 점수. 파일 쓰기 없음 (임시 복사본 제외)."""
    started = clock()
    deadline = datetime.combine(asof, DEADLINE, tzinfo=KST)
    abort: dict[str, Any] = {"reason": None}
    record = _base_record(asof, db=db, started=started)
    t0 = time.monotonic()
    try:
        if started.astimezone(KST) >= deadline:
            record["reason"] = f"started after deadline {DEADLINE:%H:%M:%S} KST"
            return record
        with _abort_guards(deadline, clock, abort):
            cron = wait_for_cron_log(cron_log_path(log_dir, asof), asof,
                                     clock=clock, sleep=sleep)
            record["freshness"]["cron_log"] = cron
            try:
                mtime = datetime.fromtimestamp(db.stat().st_mtime, KST)
                record["freshness"]["source_db_mtime"] = _iso(mtime)
            except OSError:
                pass
            if not cron["ok"]:
                record["status"] = "stale_input"
                record["reason"] = cron["reason"]
                return record
            work_root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=WORK_PREFIX, dir=work_root) as tmp:
                copy = Path(tmp) / "upbit_d1_copy.db"
                backup_copy(db, copy)
                record["copy_db_sha256"] = sha256_file(copy)
                check = inspect_copy(copy, asof)
                record["freshness"]["db_copy"] = check
                if not check["ok"]:
                    record["status"] = "stale_input"
                    record["reason"] = check["reason"]
                    return record
                record["dummy_rows_inserted"] = add_dummy_asof_rows(copy, asof)
                record["prepared_db_sha256"] = sha256_file(copy)
                record["score_source_sha256"] = score_source_sha256()
                ts = time.monotonic()
                res = scorer(copy, asof)
                record["timing"]["score_s"] = round(time.monotonic() - ts, 2)
                # scorer 가 ShadowAbort 를 삼켰어도 신호 플래그가 서 있으면 failed.
                if abort["reason"] is not None:
                    raise ShadowAbort(abort["reason"])
                if cron["health_gate"] == "unknown":
                    # 08:56 에 gate 가 진행 중이었으면 계산 뒤 한 번 더 읽는다 (기록 전용).
                    cron["health_gate"] = check_cron_log(
                        cron_log_path(log_dir, asof), asof)["health_gate"]
                    cron["health_gate_rechecked"] = True
                record["result"] = summarize_scores(res)
        if clock().astimezone(KST) >= deadline:
            record["result"] = None
            record["reason"] = f"finished after deadline {DEADLINE:%H:%M:%S} KST"
            return record
        rank_basis = str(record["result"].get("rank_basis") or "")
        if not rank_basis.startswith(R1_RANK_BASIS_PREFIX):
            # 대체 정렬(fallback) Top3 — R1 그림자로 평가하면 안 된다. 진단용으로 남김.
            record["status"] = "degraded"
            record["reason"] = f"non-R1 rank_basis: {rank_basis or 'missing'}"
            return record
        record["status"] = "ok"
    except ShadowAbort as exc:
        record["result"] = None
        record["reason"] = str(exc)
    except Exception as exc:  # noqa: BLE001 — 기록 전용, 실패도 기록
        record["result"] = None
        record["reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        if abort["reason"] is not None:
            # 어느 경로로 빠져나왔든 정지·마감 신호를 받았으면 failed 로 고정.
            record["status"] = "failed"
            record["result"] = None
            record["reason"] = abort["reason"]
        record["timing"]["elapsed_s"] = round(time.monotonic() - t0, 2)
        record["created_at"] = _iso(clock())
    return record


def run(
    *, asof: date | None = None, out_root: Path = DEFAULT_OUT_ROOT,
    db: Path = DEFAULT_DB, log_dir: Path = DEFAULT_LOG_DIR,
    clock: Callable[[], datetime] = lambda: datetime.now(KST),
    sleep: Callable[[float], None] = time.sleep,
    scorer: Callable[[Path, date], dict] = score_copy,
    work_root: Path = DEFAULT_WORK_ROOT,
) -> int:
    """0 = ok 저장 또는 이미 존재(skip) / 1 = stale_input·degraded·failed 저장 / 2 = 내부 오류."""
    asof = asof or clock().astimezone(KST).date()
    target = record_path(out_root, asof)
    if target.exists():
        _log_line(out_root, f"skip exists {target}")
        return 0
    removed = cleanup_stale_work(work_root, clock().timestamp())
    if removed:
        _log_line(out_root, f"cleanup stale work dirs under {work_root}: {','.join(removed)}")
    try:
        try:
            record = build_record(asof, db=db, log_dir=log_dir, clock=clock, sleep=sleep,
                                  scorer=scorer, work_root=work_root)
        except ShadowAbort as exc:
            # 방어: 가드 원복 경계에서 새어 나온 중단 — 결과 없이 failed 로 남긴다.
            record = _base_record(asof, db=db, started=clock())
            record["reason"] = str(exc) or "aborted"
            record["created_at"] = _iso(clock())
        written = write_once(record, out_root)
    except Exception as exc:  # noqa: BLE001 — 기록 전용, 운영 영향 없음
        _log_line(out_root, f"error {type(exc).__name__}: {exc}")
        return 2
    if written is None:
        _log_line(out_root, f"skip exists {target} (concurrent writer)")
        return 0
    top3 = ",".join(t["coin"] for t in (record["result"] or {}).get("top3", [])) or "-"
    _log_line(out_root, (
        f"{record['status']} {written} top3={top3} "
        f"score_s={record['timing']['score_s']} elapsed_s={record['timing']['elapsed_s']} "
        f"reason={record['reason'] or '-'}"
    ))
    return 0 if record["status"] == "ok" else 1


def main() -> None:
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="08:50 fresh D-1 R1 그림자 기록 (record-only)")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT,
                    help="기본 output/preopen_fresh_shadow")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB,
                    help="원본 D1 DB (read-only 로만 연다)")
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR,
                    help="cron_preopen_YYYYMMDD.log 위치 (기본 output)")
    ap.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT,
                    help="임시 DB 복사본 작업 공간 (기본 /home/soccz/22tb/tmp/prelude_shadow_work)")
    args = ap.parse_args()
    sys.exit(run(out_root=args.out_root, db=args.db, log_dir=args.log_dir,
                 work_root=args.work_root))


if __name__ == "__main__":
    main()
