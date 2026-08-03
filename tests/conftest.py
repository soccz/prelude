"""테스트 전역 안전 가드.

픽스처가 실제 셸 스크립트(backup_db.sh 등)를 subprocess 로 실행하고
subprocess 는 os.environ 을 상속하므로, 테스트 프로세스 전체에서 실제
텔레그램 발송을 구조적으로 차단한다 (notifier/telegram.py 의 kill-switch).

2026-07-28 사고 기록: 이 가드가 없던 시절, backup 실패 시나리오 픽스처가
venv 의 어디서나-import 가능한 실제 notifier + 실제 .env 토큰으로
전체 스위트 1회당 실경보 수 통을 사용자 텔레그램에 발사했다.

발송 경로 자체를 검증하는 모듈(requests 를 mock)은 자체 autouse fixture 로
이 변수를 해제한다 — subprocess 를 쓰지 않는 in-process 테스트에 한한다.

상속 환경의 빈 문자열·임의 값도 신뢰하지 않는다. ``setdefault`` 는
``PRELUDE_FORBID_TELEGRAM=""`` 을 그대로 보존해 notifier 의 truthy
검사를 우회하므로, pytest 시작마다 반드시 안전값으로 덮어쓴다.
"""
import os
import sqlite3
import sys
from pathlib import Path

import pytest

os.environ["PRELUDE_FORBID_TELEGRAM"] = "1"

# 실 운영 DB 디렉토리 — 테스트가 여기 접근하면 hermeticity 위반이다.
_PRODUCTION_DATA_ROOT = str(
    (Path(__file__).resolve().parent.parent / "data").resolve()
) + os.sep
_REAL_SQLITE_CONNECT = sqlite3.connect


@pytest.fixture(scope="session")
def _hermetic_d1_db(tmp_path_factory):
    """snapshot D1 provenance 실측용 최소 실제 DB.

    로더가 exists=True·rows>0·markets>0 를 요구(fail-closed)하므로 부재
    파일로는 대체할 수 없다 — 1행짜리 실제 sqlite 를 세션당 1회 만든다.
    """
    path = tmp_path_factory.mktemp("hermetic") / "upbit_d1.db"
    connection = _REAL_SQLITE_CONNECT(path)
    try:
        connection.execute(
            "CREATE TABLE candles ("
            "market TEXT, timestamp TEXT, open REAL, high REAL, "
            "low REAL, close REAL, volume REAL)"
        )
        connection.execute(
            "INSERT INTO candles VALUES "
            "('KRW-BTC', '2026-07-01 09:00:00', 1.0, 1.0, 1.0, 1.0, 1.0)"
        )
        connection.commit()
    finally:
        connection.close()
    return path


@pytest.fixture(autouse=True)
def _forbid_production_db_access(monkeypatch, _hermetic_d1_db):
    """테스트 hermeticity 가드 — 실 운영 DB(data/*.db) 접근 즉시 실패.

    2026-08-03 실사고: 부팅 폭풍에서 selftest 스위트가 라이브 수집기와
    동시에 돌았고, 기본 인자(M15_DB_PATH)로 실 DB 를 읽던 테스트 1건이
    manifest 안정성 검증(파일이 캡처 중 변경되면 거부)에 걸려 실패 →
    OnFailure 경보. 평시엔 통과해서 존재조차 몰랐던 탈출구다. 실 DB 를
    읽는 테스트는 '동시성에 따라 결과가 달라지는 테스트'이므로 클래스
    자체를 금지한다 (in-process 한정; subprocess 는 인자 계약으로 방어).
    """

    # snapshot provenance 실측 대상도 tmp 로 재지향 — manifest 의 canonical
    # "path" 문자열 계약은 프로덕션 코드가 유지한다. (collection 단계에서
    # 모든 테스트 모듈 import 가 끝나므로 sys.modules 검사로 충분하고,
    # 무관한 경량 테스트에 무거운 import 를 강제하지 않는다.)
    snapshot_module = sys.modules.get("signals.recommend_snapshot")
    if snapshot_module is not None:
        monkeypatch.setattr(
            snapshot_module, "_DATA_DB_PATH", _hermetic_d1_db
        )

    def guarded_connect(database, *args, **kwargs):
        raw = str(database)
        if raw.startswith("file:"):
            raw = raw[5:].split("?", 1)[0]
        if raw and raw != ":memory:":
            try:
                resolved = str(Path(raw).resolve())
            except (OSError, ValueError):
                resolved = ""
            if resolved.startswith(_PRODUCTION_DATA_ROOT):
                raise AssertionError(
                    "hermeticity violation: 테스트가 실 운영 DB 를 열었다 — "
                    f"{resolved} (tmp_path fixture 로 격리할 것)"
                )
        return _REAL_SQLITE_CONNECT(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", guarded_connect)
