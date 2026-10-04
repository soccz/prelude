"""scripts/record_preopen_fresh_shadow.py — 08:50 fresh D-1 R1 그림자 기록 (record-only).

임시 소형 sqlite 로 DB 준비·신선도 가드·write-once·마감 가드·status 기록을 본다.
score_candidates 는 mock (실제 학습 없음).
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import signal
import sqlite3
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

import scripts.record_preopen_fresh_shadow as shadow
from data.database import init_db

KST = shadow.KST
ASOF = date(2026, 10, 5)
AT_0856 = datetime(2026, 10, 5, 8, 56, 0, tzinfo=KST)
MARKETS = ("KRW-BTC", "KRW-AAA", "KRW-BBB", "KRW-CCC")


def _make_db(path: Path, *, days: int = 5, markets=MARKETS, dm1_missing=(),
             future=False) -> Path:
    """asof 직전 ``days`` 일 일봉 (마지막 = D-1 진행 중 일봉)."""
    init_db(path)
    con = sqlite3.connect(path)
    with con:
        for back in range(days, 0, -1):
            day = ASOF - timedelta(days=back)
            for i, market in enumerate(markets):
                if back == 1 and market in dm1_missing:
                    continue
                price = 100.0 + i + back
                con.execute(
                    "INSERT INTO candles VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (market, f"{day} 09:00:00", price, price + 2, price - 2,
                     price + 1, 10.0, 1000.0))
        if future:
            con.execute("INSERT INTO candles VALUES (?, ?, 1, 1, 1, 1, 1, 1)",
                        ("KRW-AAA", f"{ASOF} 09:00:00"))
    con.close()
    return path


def _cron_log(log_dir: Path, *, d1_done: str | None = "08:53:01", header_date: date = ASOF,
              critical: bool = False, extra_block: str | None = None,
              summary_tail: str = "missing=none failed=none",
              next_step: bool = True, health: str | None = None) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"=== prelude pre-open trigger KST {header_date} 08:50:24 ===",
             "[1/6] data update — d1",
             "08:50:25 [INFO] Active signal-eligible KRW markets: 286"]
    if critical:
        lines.append("  [critical] preopen D1 universe update failed (exit=1) — 가능한 후속 단계는 계속")
    if d1_done:
        # 실제 형식 (output/cron_preopen_2026100{1,2,3}.log 확인, 2026-10-04).
        lines += [f"{d1_done} [INFO] D1 universe after update: covered=286/286 (100.00%) "
                  f"{summary_tail}",
                  "Updated 286 markets",
                  "Universe coverage: 285/286 (99.65%) -> 286/286 (100.00%); new=1 "
                  "missing=0 inactive_retained=10",
                  "=== DB stats ==="]
        if next_step:
            lines.append("[2/6] health_check gate (recommend-preopen: d1 only)")
            lines += _health_lines(health)
    if extra_block:
        lines += ["", extra_block]
    path = shadow.cron_log_path(log_dir, ASOF)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _health_lines(health: str | None) -> list[str]:
    """'[2/6]' 뒤 health gate 출력 (실제 형식: cron_preopen_20261003.log, 2026-10-04)."""
    if health is None:  # gate 진행 중
        return [f"=== prelude health {ASOF}T08:53:02.138868+09:00 channel=recommend-preopen ==="]
    if health == "passed":
        return [f"=== prelude health {ASOF}T08:53:02.138868+09:00 channel=recommend-preopen ===",
                "  OK: universe coverage: scope=recommend-preopen pit_top=100/100",
                "",
                "✅ ALL OK",
                "[3/6] recommend_send + recommend_today (R1 preopen 전용 ledger)"]
    assert health == "failed"
    return [f"=== prelude health {ASOF}T08:53:02.138868+09:00 channel=recommend-preopen ===",
            "  CRITICAL: universe coverage: upbit_d1.db=exact 97/100 (97.00%)",
            "  [critical] R1 preopen D1 health gate failed (exit=1) — 가능한 후속 단계는 계속",
            "  R1 preopen send·record skipped (stale D1)",
            "[3/6] recommend_send + recommend_today (R1 preopen 전용 ledger)"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeScorer:
    def __init__(self, action=None, rank_basis="R1_riskreward(de-corr head)"):
        self.calls: list[tuple[Path, date]] = []
        self.action = action
        self.rank_basis = rank_basis

    def __call__(self, db: Path, asof: date) -> dict:
        self.calls.append((db, asof))
        if self.action:
            self.action()
        universe = [{"coin": f"KRW-C{i:02d}", "rank": i + 1, "rr_ratio": 1.0 - i / 100,
                     "p_up5": 0.4, "p_up10": 0.2, "p_up20": 0.05, "p_dn5": 0.2,
                     "p_dn10": 0.05, "exp_downside": -0.04, "dump_risk_flag": False,
                     "entry_open": 1.0} for i in range(100)]
        return {"asof": str(asof), "slot": "open", "feature_date": str(asof),
                "ranking": "R1", "rank_basis": self.rank_basis,
                "rule_version": "r1", "btc_regime": "bull", "universe_n": 100,
                "training": {"rows": 1}, "universe": universe, "top3": universe[:3]}


class SwallowingScorer(FakeScorer):
    """봉인 signals/recommend.py 처럼 학습 구간을 넓은 except 로 감싸고,
    실패하면 대체 정렬(score_rank fallback) 결과를 돌려주는 scorer."""

    def __init__(self, during, catch=Exception):
        super().__init__(rank_basis="score_rank(fallback)")
        self.during = during
        self.catch = catch
        self.swallowed = False

    def __call__(self, db: Path, asof: date) -> dict:
        try:
            self.during()
            time.sleep(0.2)  # 신호 핸들러가 이 try 안에서 돈다
        except self.catch:  # noqa: BLE001 — 봉인 scorer 의 'except Exception' 재현
            self.swallowed = True
        return super().__call__(db, asof)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(shadow, "score_source_sha256", lambda: "a" * 64)
    db = _make_db(tmp_path / "live" / "upbit_d1.db")
    return {"db": db, "log_dir": tmp_path / "logs", "out": tmp_path / "out",
            "work": tmp_path / "work"}


def _run(env, *, clock=lambda: AT_0856, scorer=None, sleep=lambda _s: None):
    scorer = scorer or FakeScorer()
    code = shadow.run(asof=ASOF, out_root=env["out"], db=env["db"],
                      log_dir=env["log_dir"], clock=clock, sleep=sleep, scorer=scorer,
                      work_root=env["work"])
    path = env["out"] / f"{ASOF}.json"
    record = json.loads(path.read_text()) if path.exists() else None
    return code, record, scorer


# ==========================================================================
# 신선도 가드 — cron_preopen 로그
# ==========================================================================
def test_cron_log_ok_when_d1_update_finished_after_0845(tmp_path):
    path = _cron_log(tmp_path)
    result = shadow.check_cron_log(path, ASOF)
    assert result["ok"] and not result["pending"]
    assert (result["run_started"], result["d1_completed"], result["updated_markets"]) == (
        "08:50:24", "08:53:01", 286)


@pytest.mark.parametrize("kwargs, pending, reason", [
    ({"d1_done": None}, True, "not completed"),
    ({"d1_done": "08:40:00"}, False, "< 08:45"),
    ({"critical": True}, False, "D1 update failed"),
    ({"header_date": date(2026, 10, 4)}, False, "no pre-open run header"),
    # 요약 줄에 누락·실패 종목 — collector exit 1 (critical 줄이 아직 안 써졌어도).
    ({"summary_tail": "missing=KRW-AAA failed=none", "next_step": False}, False,
     "D1 update incomplete"),
    ({"summary_tail": "missing=none failed=KRW-BBB, KRW-CCC"}, False,
     "D1 update incomplete"),
    # 'Updated N' 은 보였지만 다음 단계 헤더 전 — bash 가 critical 줄을 쓸 수 있는 구간.
    ({"next_step": False}, True, "not completed"),
])
def test_cron_log_not_fresh(tmp_path, kwargs, pending, reason):
    path = _cron_log(tmp_path, **kwargs)
    result = shadow.check_cron_log(path, ASOF)
    assert not result["ok"] and result["pending"] is pending
    assert reason in result["reason"]


@pytest.mark.parametrize("health, expected", [
    ("passed", "passed"), ("failed", "failed"), (None, "unknown")])
def test_cron_log_records_r1_health_gate_without_changing_ok(tmp_path, health, expected):
    """health gate 는 기록 전용 — D1 갱신이 끝났으면 gate 결과와 무관하게 ok."""
    result = shadow.check_cron_log(_cron_log(tmp_path, health=health), ASOF)
    assert result["ok"] and result["health_gate"] == expected


def test_cron_log_health_gate_unknown_when_d1_step_failed_or_pending(tmp_path):
    # D1 단계 실패면 bash 가 gate 를 건너뛰고 '[3/6]' 만 쓴다 — passed 로 읽으면 안 된다.
    crit = shadow.check_cron_log(_cron_log(tmp_path, critical=True, health="passed"), ASOF)
    assert not crit["ok"] and crit["health_gate"] == "unknown"
    pending = shadow.check_cron_log(_cron_log(tmp_path, next_step=False), ASOF)
    assert pending["pending"] and pending["health_gate"] == "unknown"
    # 앞 실행 블록의 결과는 마지막(재)실행 블록에 이어지지 않는다.
    rerun = _cron_log(tmp_path, health="failed",
                      extra_block=f"=== prelude pre-open trigger KST {ASOF} 08:54:00 ===")
    assert shadow.check_cron_log(rerun, ASOF)["health_gate"] == "unknown"


def test_cron_log_missing_and_latest_run_block_decides(tmp_path):
    assert shadow.check_cron_log(tmp_path / "nope.log", ASOF)["reason"] == \
        "cron_preopen log missing"
    # 앞 실행은 완료됐어도 마지막(재)실행 블록이 진행 중이면 pending.
    path = _cron_log(tmp_path, extra_block=f"=== prelude pre-open trigger KST {ASOF} 08:54:00 ===")
    result = shadow.check_cron_log(path, ASOF)
    assert not result["ok"] and result["pending"] and result["run_started"] == "08:54:00"


def test_wait_for_cron_log_polls_until_d1_done(tmp_path):
    path = _cron_log(tmp_path, d1_done=None)
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        _cron_log(tmp_path)  # 기다리는 동안 D1 단계 완료

    result = shadow.wait_for_cron_log(path, ASOF, clock=lambda: AT_0856, sleep=sleep)
    assert result["ok"] and sleeps == [shadow.WAIT_POLL_S]


def test_cron_log_race_updated_line_then_critical_is_never_ok(tmp_path):
    """'Updated N markets' 직후·critical 줄 직전에 읽어도 ok 가 아니다."""
    lines = [f"=== prelude pre-open trigger KST {ASOF} 08:50:24 ===",
             "[1/6] data update — d1",
             "08:53:01 [INFO] D1 universe after update: covered=285/286 (99.65%) "
             "missing=none failed=none",
             "Updated 286 markets"]
    path = shadow.cron_log_path(tmp_path, ASOF)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    first = shadow.check_cron_log(path, ASOF)
    assert not first["ok"] and first["pending"] and not first["next_step_seen"]
    lines += ["  [critical] preopen D1 universe update failed (exit=1) — 가능한 후속 단계는 계속",
              "  R1 preopen send·record skipped (partial/failed D1 update)",
              "[2/6] health_check gate (recommend-preopen: d1 only)"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    second = shadow.check_cron_log(path, ASOF)
    assert not second["ok"] and not second["pending"]
    assert "D1 update failed" in second["reason"]


def test_cron_log_accepts_real_log_excerpt_format(tmp_path):
    """실제 cron_preopen 로그 발췌(2026-10-03, 줄 사이 다른 출력 포함)."""
    text = "\n".join([
        f"=== prelude pre-open trigger KST {ASOF} 08:50:24 ===",
        "[1/6] data update — d1",
        "08:50:25 [INFO] Active signal-eligible KRW markets: 286",
        "08:53:01 [INFO] D1 universe after update: covered=286/286 (100.00%) "
        "missing=none failed=none",
        "Updated 286 markets",
        "Universe coverage: 285/286 (99.65%) -> 286/286 (100.00%); new=1 missing=0 "
        "inactive_retained=10",
        "",
        "=== DB stats ===",
        "KRW-BTC: 3000 rows",
        "[2/6] health_check gate (recommend-preopen: d1 only)",
        "=== prelude health 2026-10-05T08:53:02.138868+09:00 channel=recommend-preopen ===",
        "  OK: output/cron_preopen_20261005.log: 0.0h ago",
        "",
        "✅ ALL OK",
        "[3/6] recommend_send + recommend_today (R1 preopen 전용 ledger)",
        "2026-10-05 08:53:02,964 [INFO] slot=preopen champion=recommend_r1_preopen",
        "[4/6] data update — 15m all 1 day (record-only legacy)",
        "  [critical] legacy preopen health gate failed (exit=1) — 가능한 후속 단계는 계속",
    ]) + "\n"
    path = shadow.cron_log_path(tmp_path, ASOF)
    path.write_text(text, encoding="utf-8")
    result = shadow.check_cron_log(path, ASOF)
    assert result["ok"] and result["next_step_seen"] and result["updated_markets"] == 286
    # legacy(15m) gate 실패 줄은 R1 health gate 가 아니다.
    assert result["health_gate"] == "passed"


def test_guard_strings_match_daily_run_preopen_and_collector_sources():
    """신선도 가드가 기대는 문자열이 실제 생산 코드에 그대로 있다 (형식 변경 감지).

    순서도 고정: D1 critical 줄은 '[2/6]' 헤더보다 먼저 쓰인다.
    """
    root = Path(shadow.__file__).resolve().parent.parent
    sh = (root / "scripts" / "daily_run_preopen.sh").read_text(encoding="utf-8")
    assert "=== prelude pre-open trigger KST" in sh
    i_d1 = sh.index('echo "[1/6] data update — d1"')
    i_crit = sh.index('record_critical_failure "$rc" "preopen D1 universe update"')
    i_next = sh.index(f'echo "{shadow._NEXT_STEP}')
    assert i_d1 < i_crit < i_next
    assert "[critical] $step failed" in sh  # → '[critical] preopen D1 universe update failed'
    # R1 health gate 실패 줄('[critical] R1 preopen D1 health gate failed')은 '[2/6]' 와
    # '[3/6]' 헤더 사이에서만 쓰인다 — '[3/6]' 가 보이면 gate 결과가 확정.
    assert shadow._HEALTH_CRITICAL == "[critical] R1 preopen D1 health gate failed"
    i_gate = sh.index('record_critical_failure "$?" "R1 preopen D1 health gate"')
    i_done = sh.index(f'echo "{shadow._HEALTH_DONE_STEP}')
    assert i_next < i_gate < i_done
    collector = (root / "data" / "collector_d1.py").read_text(encoding="utf-8")
    assert '"D1 universe after update: covered=%d/%d (%.2f%%) missing=%s failed=%s"' \
        in collector
    assert 'or "none"' in collector and 'print(f"Updated {len(results)} markets")' in collector


def test_wait_for_cron_log_gives_up_at_wait_limit(tmp_path):
    path = _cron_log(tmp_path, d1_done=None)
    now = [AT_0856]

    def sleep(seconds):
        now[0] += timedelta(seconds=seconds)

    result = shadow.wait_for_cron_log(path, ASOF, clock=lambda: now[0], sleep=sleep)
    assert result["pending"] and not result["ok"]
    assert now[0] == datetime.combine(ASOF, shadow.WAIT_D1_UNTIL, tzinfo=KST)


# ==========================================================================
# DB 준비
# ==========================================================================
def test_backup_copy_leaves_source_untouched(tmp_path):
    src = _make_db(tmp_path / "src.db")
    before = _sha(src)
    dst = tmp_path / "copy.db"
    shadow.backup_copy(src, dst)
    assert _sha(src) == before
    con = sqlite3.connect(dst)
    assert con.execute("SELECT COUNT(*) FROM candles").fetchone()[0] == 5 * len(MARKETS)
    con.close()
    with pytest.raises(FileExistsError):
        shadow.backup_copy(src, dst)


def test_inspect_copy_accepts_fresh_dm1(tmp_path):
    result = shadow.inspect_copy(_make_db(tmp_path / "a.db"), ASOF)
    assert result["ok"] and result["n_markets_dm1"] == 4 and result["btc_has_dm1"]
    assert result["dm1_coverage_of_dm2"] == 1.0


@pytest.mark.parametrize("kwargs, reason", [
    ({"future": True}, "post-open DB"),
    ({"dm1_missing": ("KRW-BTC",)}, "KRW-BTC has no"),
    ({"dm1_missing": ("KRW-AAA",)}, "coverage"),
])
def test_inspect_copy_rejects_stale_or_post_open(tmp_path, kwargs, reason):
    result = shadow.inspect_copy(_make_db(tmp_path / "a.db", **kwargs), ASOF)
    assert not result["ok"] and reason in result["reason"]


def test_dummy_rows_copy_dm1_close_with_zero_volume(tmp_path):
    db = _make_db(tmp_path / "a.db", dm1_missing=("KRW-CCC",))
    assert shadow.add_dummy_asof_rows(db, ASOF) == 3
    con = sqlite3.connect(db)
    rows = con.execute(
        "SELECT d.market, d.open, d.high, d.low, d.close, d.volume, d.quote_volume, p.close "
        "FROM candles d JOIN candles p ON p.market = d.market AND p.timestamp = ? "
        "WHERE d.timestamp = ?", (f"{ASOF - timedelta(days=1)} 09:00:00",
                                  f"{ASOF} 09:00:00")).fetchall()
    con.close()
    assert sorted(r[0] for r in rows) == ["KRW-AAA", "KRW-BBB", "KRW-BTC"]
    for _m, o, h, lo, c, vol, qv, prev_close in rows:
        assert o == h == lo == c == prev_close and vol == 0.0 and qv == 0.0


def test_score_copy_points_scorer_at_copy_then_restores(monkeypatch, tmp_path):
    import scripts.regime_split_precursor_v1 as rsp
    import signals.recommend as rec

    before = (rec.DB_PATH, rsp.DB_PATH)
    seen = {}

    def fake_score(asof, *, slot, ranking):
        seen.update(asof=asof, slot=slot, ranking=ranking,
                    paths=(rec.DB_PATH, rsp.DB_PATH))
        return {"top3": []}

    monkeypatch.setattr(rec, "score_candidates", fake_score)
    copy = tmp_path / "copy.db"
    assert shadow.score_copy(copy, ASOF) == {"top3": []}
    assert seen == {"asof": "2026-10-05", "slot": "open", "ranking": "R1",
                    "paths": (str(copy), str(copy))}
    assert (rec.DB_PATH, rsp.DB_PATH) == before


# ==========================================================================
# run — status 기록·write-once·마감
# ==========================================================================
def test_ok_record_has_inputs_hashes_top3_and_candidates(env):
    _cron_log(env["log_dir"])
    live_before = _sha(env["db"])
    code, record, scorer = _run(env)

    assert code == 0 and record["status"] == "ok" and record["reason"] is None
    assert _sha(env["db"]) == live_before  # 원본 무변경
    (copy_path, asof), = scorer.calls
    assert asof == ASOF and not copy_path.exists()  # 임시 복사본은 지워진다
    # 복사본은 라이브 output 이 아니라 작업 공간(work_root) 아래에 만들어졌다.
    assert copy_path.parent.parent == env["work"]
    assert copy_path.parent.name.startswith(shadow.WORK_PREFIX)
    assert list(env["work"].iterdir()) == []
    # backup 복사본 해시 (sqlite 헤더 차이로 원본 파일 해시와는 다를 수 있다).
    assert len(record["copy_db_sha256"]) == 64
    assert record["prepared_db_sha256"] != record["copy_db_sha256"]
    assert record["dummy_rows_inserted"] == len(MARKETS)
    assert record["score_source_sha256"] == "a" * 64
    assert record["freshness"]["cron_log"]["ok"] and record["freshness"]["db_copy"]["ok"]
    result = record["result"]
    assert [t["coin"] for t in result["top3"]] == ["KRW-C00", "KRW-C01", "KRW-C02"]
    assert set(result["top3"][0]) >= {"coin", "rank", "rr_ratio", "p_up10", "p_dn5"}
    assert len(result["candidates"]) == 100 and result["candidates"][0][:2] == ["KRW-C00", 1]
    assert record["timing"]["score_s"] is not None and record["record_only"] is True
    assert sorted(p.name for p in env["out"].iterdir()) == [f"{ASOF}.json", "record.log"]


@pytest.mark.parametrize("health", ["passed", "failed"])
def test_health_gate_is_recorded_but_does_not_change_status(env, health):
    _cron_log(env["log_dir"], health=health)
    code, record, scorer = _run(env)
    assert code == 0 and record["status"] == "ok" and len(scorer.calls) == 1
    cron = record["freshness"]["cron_log"]
    assert cron["health_gate"] == health and "health_gate_rechecked" not in cron
    assert record["result"]["top3"]


def test_health_gate_unknown_at_start_is_rechecked_after_scoring(env):
    _cron_log(env["log_dir"])  # gate 진행 중
    scorer = FakeScorer(action=lambda: _cron_log(env["log_dir"], health="failed"))
    code, record, _ = _run(env, scorer=scorer)
    assert code == 0 and record["status"] == "ok"
    cron = record["freshness"]["cron_log"]
    assert cron["health_gate"] == "failed" and cron["health_gate_rechecked"] is True


def test_health_gate_stays_unknown_if_still_running_after_scoring(env):
    _cron_log(env["log_dir"])
    _code, record, _ = _run(env)
    cron = record["freshness"]["cron_log"]
    assert record["status"] == "ok"
    assert cron["health_gate"] == "unknown" and cron["health_gate_rechecked"] is True


def test_stale_log_records_stale_input_without_scoring(env):
    code, record, scorer = _run(env)  # 로그 없음
    assert code == 1 and record["status"] == "stale_input"
    assert record["reason"] == "cron_preopen log missing"
    assert scorer.calls == [] and record["copy_db_sha256"] is None


def test_post_open_db_records_stale_input_without_scoring(env, tmp_path):
    _cron_log(env["log_dir"])
    env["db"] = _make_db(tmp_path / "post" / "upbit_d1.db", future=True)
    code, record, scorer = _run(env)
    assert code == 1 and record["status"] == "stale_input"
    assert "post-open DB" in record["reason"] and scorer.calls == []
    assert record["copy_db_sha256"] is not None


def test_write_once_skips_second_run_without_scoring(env):
    _cron_log(env["log_dir"])
    code, _record, scorer = _run(env)
    path = env["out"] / f"{ASOF}.json"
    before = path.read_bytes()
    code2, _record2, scorer2 = _run(env)
    assert (code, code2) == (0, 0)
    assert len(scorer.calls) == 1 and scorer2.calls == []
    assert path.read_bytes() == before
    assert "skip exists" in (env["out"] / "record.log").read_text()


def test_write_once_never_overwrites_existing_record(tmp_path):
    record = {"asof": str(ASOF), "status": "ok"}
    assert shadow.write_once(record, tmp_path) == tmp_path / f"{ASOF}.json"
    assert shadow.write_once({"asof": str(ASOF), "status": "failed"}, tmp_path) is None
    assert json.loads((tmp_path / f"{ASOF}.json").read_text())["status"] == "ok"
    assert not list(tmp_path.glob(".*tmp*"))


def test_started_after_deadline_records_failed_without_reading_inputs(env):
    _cron_log(env["log_dir"])
    late = datetime.combine(ASOF, shadow.DEADLINE, tzinfo=KST)
    code, record, scorer = _run(env, clock=lambda: late)
    assert code == 1 and record["status"] == "failed"
    assert "started after deadline" in record["reason"] and scorer.calls == []


def test_deadline_alarm_aborts_long_scoring_as_failed(env):
    _cron_log(env["log_dir"])
    deadline = datetime.combine(ASOF, shadow.DEADLINE, tzinfo=KST)
    t0 = time.monotonic()

    def clock():  # 마감 0.3초 전부터 실제 시간처럼 흐름
        return deadline - timedelta(seconds=0.3) + timedelta(seconds=time.monotonic() - t0)

    scorer = FakeScorer(action=lambda: time.sleep(5))
    started = time.monotonic()
    code, record, _ = _run(env, clock=clock, scorer=scorer)
    assert time.monotonic() - started < 3
    assert code == 1 and record["status"] == "failed" and record["result"] is None
    assert "deadline 08:59:30 KST reached" in record["reason"]
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    assert not [p for p in env["out"].iterdir() if p.name.startswith(".work-")]
    assert list(env["work"].iterdir()) == []


def test_sigterm_during_scoring_records_failed(env):
    _cron_log(env["log_dir"])
    handler_before = signal.getsignal(signal.SIGTERM)
    scorer = FakeScorer(action=lambda: os.kill(os.getpid(), signal.SIGTERM))
    code, record, _ = _run(env, scorer=scorer)
    assert code == 1 and record["status"] == "failed"
    assert record["reason"] == "terminated (SIGTERM)"
    assert signal.getsignal(signal.SIGTERM) is handler_before  # 핸들러 원복


def test_shadow_abort_is_not_an_exception_subclass():
    # 봉인 scorer 의 'except Exception' 에 잡히지 않아야 한다.
    assert issubclass(shadow.ShadowAbort, BaseException)
    assert not issubclass(shadow.ShadowAbort, Exception)


@pytest.mark.parametrize("catch", [Exception, BaseException])
def test_sigterm_inside_broad_except_scorer_still_records_failed(env, catch):
    """봉인 scorer 의 넓은 except 안에서 SIGTERM — fallback 결과로 ok 가 되면 안 된다.

    Exception: ShadowAbort(BaseException) 가 빠져나와 failed.
    BaseException: 삼켜져도 신호 플래그로 failed.
    """
    _cron_log(env["log_dir"])
    handler_before = signal.getsignal(signal.SIGTERM)
    scorer = SwallowingScorer(lambda: os.kill(os.getpid(), signal.SIGTERM), catch=catch)
    code, record, _ = _run(env, scorer=scorer)
    assert scorer.swallowed is (catch is BaseException)
    assert code == 1 and record["status"] == "failed" and record["result"] is None
    assert record["reason"] == "terminated (SIGTERM)"
    assert signal.getsignal(signal.SIGTERM) is handler_before
    assert list(env["work"].iterdir()) == []


@pytest.mark.parametrize("catch", [Exception, BaseException])
def test_deadline_alarm_inside_broad_except_scorer_still_records_failed(env, catch):
    _cron_log(env["log_dir"])

    def deadline_fires_inside_scorer():
        # 실제 마감 타이머(_abort_guards 가 건 ITIMER_REAL)를 scorer 안에서 곧 울리게
        # 당긴다 — 준비 단계 소요와 무관하게 신호가 scorer 의 except 안에서 돈다.
        signal.setitimer(signal.ITIMER_REAL, 0.05)
        time.sleep(2)

    scorer = SwallowingScorer(deadline_fires_inside_scorer, catch=catch)
    started = time.monotonic()
    code, record, _ = _run(env, scorer=scorer)
    assert time.monotonic() - started < 1.5
    assert scorer.swallowed is (catch is BaseException)
    assert code == 1 and record["status"] == "failed" and record["result"] is None
    assert record["reason"] == "deadline 08:59:30 KST reached"
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def _guards(abort):
    deadline = datetime.now(KST) + timedelta(seconds=30)
    return shadow._abort_guards(deadline, lambda: datetime.now(KST), abort)


@pytest.mark.parametrize("signum, reason", [
    (signal.SIGTERM, "terminated (SIGTERM)"),
    (signal.SIGALRM, "deadline"),
])
def test_guard_exit_drains_signal_pending_during_restore(signum, reason):
    """원복 경계에서 도착한 신호 — 대기 중이면 원복 중 소비돼 플래그만 남는다.

    결정적 재현: 메인 스레드에서 신호를 막아 둔 채 그 스레드로 보내(pending) 가드를
    빠져나온다. 원복 뒤 기본 핸들러(SIG_DFL)로 늦게 배달되면 프로세스가 죽는다.
    """
    import threading

    sigs = {signal.SIGALRM, signal.SIGTERM}
    term_before = signal.getsignal(signal.SIGTERM)
    alarm_before = signal.getsignal(signal.SIGALRM)
    abort = {"reason": None}
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, sigs)
    try:
        with _guards(abort):
            signal.pthread_kill(threading.get_ident(), signum)
            assert signum in signal.sigpending()
        # 가드가 소비했다 — 막힌 채로 남은 신호 없음, 마스크는 진입 전(막힘) 그대로.
        assert not (sigs & signal.sigpending())
        assert sigs <= signal.pthread_sigmask(signal.SIG_BLOCK, set())
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
    assert reason in abort["reason"]
    assert signal.getsignal(signal.SIGTERM) is term_before
    assert signal.getsignal(signal.SIGALRM) is alarm_before
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_guard_handler_does_not_raise_once_closing():
    """원복 중(closing) 도착한 신호는 던지지 않고 플래그만 — 원복이 끝까지 실행된다."""
    term_before = signal.getsignal(signal.SIGTERM)
    abort = {"reason": None}
    real_setitimer = signal.setitimer
    fired = []

    def setitimer_then_signal(which, seconds, *rest):
        # 원복 첫 단계(itimer 해제) 직후 SIGTERM — 핸들러가 아직 설치된 시점.
        real_setitimer(which, seconds, *rest)
        if seconds == 0 and not fired:
            fired.append(True)
            handler = signal.getsignal(signal.SIGTERM)
            handler(signal.SIGTERM, None)  # closing 이면 ShadowAbort 없이 반환

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(shadow.signal, "setitimer", setitimer_then_signal)
        with _guards(abort):
            pass
    assert fired and abort["reason"] == "terminated (SIGTERM)"
    assert signal.getsignal(signal.SIGTERM) is term_before
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_run_records_failed_when_shadow_abort_escapes_build_record(env, monkeypatch):
    def leak(*_a, **_k):
        raise shadow.ShadowAbort("terminated (SIGTERM)")

    monkeypatch.setattr(shadow, "build_record", leak)
    code, record, _ = _run(env)
    assert code == 1 and record["status"] == "failed" and record["result"] is None
    assert record["reason"] == "terminated (SIGTERM)" and record["schema"] == shadow.SCHEMA
    assert record["created_at"] is not None
    assert "failed" in (env["out"] / "record.log").read_text()


def test_fallback_rank_basis_records_degraded_not_ok(env):
    _cron_log(env["log_dir"])
    code, record, _ = _run(env, scorer=FakeScorer(rank_basis="score_rank(fallback)"))
    assert code == 1 and record["status"] == "degraded"
    assert record["reason"] == "non-R1 rank_basis: score_rank(fallback)"
    # 진단용으로 결과는 남기되, 평가 스크립트는 status=ok 만 쓴다.
    assert record["result"]["rank_basis"] == "score_rank(fallback)"
    assert "degraded" in (env["out"] / "record.log").read_text()


@pytest.mark.parametrize("basis", ["A1_sustain(R1 fallback — dump head 실패)", None])
def test_other_non_r1_rank_basis_is_degraded(env, basis):
    _cron_log(env["log_dir"])
    _code, record, _ = _run(env, scorer=FakeScorer(rank_basis=basis))
    assert record["status"] == "degraded"


def test_stale_work_dirs_older_than_an_hour_are_removed_at_start(env):
    _cron_log(env["log_dir"])
    work = env["work"]
    now = AT_0856.timestamp()

    def make(name, age_s, *, is_dir=True):
        path = work / name
        if is_dir:
            path.mkdir(parents=True)
            (path / "upbit_d1_copy.db").write_bytes(b"x" * 10)
        else:
            work.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
        os.utime(path, (now - age_s, now - age_s))
        return path

    assert shadow.STALE_WORK_AGE_S == 3600.0
    old = make(".work-old", 2 * 3600)
    # 어제 08:56:30 실행의 SIGKILL 잔여물을 오늘 08:56:05 에 본다 (age 86375s) — 지운다.
    yesterday = make(".work-yesterday", 86375)
    fresh = make(".work-fresh", 1800)          # 다른 실행이 쓰는 중일 수 있음 — 유지
    other = make("keep-me", 3 * 86400)         # 접두가 다름 — 유지
    old_file = make(".work-file", 3 * 86400, is_dir=False)  # 디렉토리 아님 — 유지
    code, record, _ = _run(env)
    assert code == 0 and record["status"] == "ok"
    assert not old.exists() and not yesterday.exists()
    assert fresh.exists() and other.exists() and old_file.exists()
    assert "cleanup stale work dirs" in (env["out"] / "record.log").read_text()
    assert ".work-old" in (env["out"] / "record.log").read_text()


def test_default_work_root_is_outside_live_output_and_root_tmp():
    work = shadow.DEFAULT_WORK_ROOT
    assert work == Path("/home/soccz/22tb/tmp/prelude_shadow_work")
    assert shadow.ROOT / "output" not in work.parents
    assert not str(work).startswith("/tmp")


def test_scorer_exception_records_failed(env):
    _cron_log(env["log_dir"])

    def boom():
        raise RuntimeError("panel empty")

    code, record, _ = _run(env, scorer=FakeScorer(action=boom))
    assert code == 1 and record["status"] == "failed"
    assert record["reason"] == "RuntimeError: panel empty"


def test_recorder_has_no_telegram_ledger_or_snapshot_writer_imports():
    tree = ast.parse(Path(shadow.__file__).read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    assert not any(m.startswith(("notifier", "ledger", "ops.")) for m in modules)
    # 봉인 R1 소스는 import 만 (score 경로 + 버전 해시용) — 수정·쓰기 없음.
    assert {"signals.recommend", "scripts.regime_split_precursor_v1",
            "signals.recommend_snapshot", "data.database"} <= modules
    source = Path(shadow.__file__).read_text(encoding="utf-8")
    assert "send_telegram" not in source and "get_or_create_recommend_snapshot" not in source
