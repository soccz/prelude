"""import 부작용 금지 불변식 — root logger 는 import 만으로 구성되지 않는다.

07-28/29 이틀 연속 라이브 장애의 근본 원인: 일일 CLI 스크립트가 모듈
레벨에서 ``logging.basicConfig(stdout)`` 을 실행해, 이들을 import 하는
소비자(close gate NUL 프로토콜, heartbeat 'ok' 프로브)의 기계 파싱
stdout 이 오염됐다.  이 테스트는 라이브 import 그래프 전체를 새
프로세스에서 import 한 뒤 root logger 에 핸들러가 하나도 없어야 함을
강제한다 — 새 모듈이 같은 실수를 다시 들여오면 여기서 즉시 실패한다.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LIVE_IMPORT_GRAPH = (
    "ops.close_input_gate",
    "ops.policy_competition",
    "ops.v2_provenance",
    "ops.champion_selector",
    "scripts.pump_detector_today",
    "scripts.pump_detector_v2_today",
    "scripts.recommend_today",
    "scripts.recommend_send",
    "scripts.close_recommend_ledger",
    "scripts.v2_scoreboard",
    "signals.recommend_snapshot",
    "signals.recommend_score_labels",
    "notifier.telegram",
    "notifier.delivery_receipt",
)


def test_live_import_graph_leaves_root_logger_untouched():
    probe = (
        "import logging, sys\n"
        + "\n".join(f"import {module}" for module in LIVE_IMPORT_GRAPH)
        + "\n"
        "handlers = logging.getLogger().handlers\n"
        "if handlers:\n"
        "    print('root logger configured at import: %r' % handlers,\n"
        "          file=sys.stderr)\n"
        "    sys.exit(1)\n"
        "sys.exit(0)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        cwd=PROJECT_ROOT,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
