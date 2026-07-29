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

os.environ["PRELUDE_FORBID_TELEGRAM"] = "1"
