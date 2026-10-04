from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from scripts.systemd_failure_alert import build_message


def test_failure_message_names_unit_and_time():
    now = datetime(2026, 7, 25, 9, 10, tzinfo=ZoneInfo("Asia/Seoul"))

    msg = build_message("prelude-distribution.service", now=now)

    assert "prelude-distribution.service" in msg
    assert "2026-07-25 09:10:00 KST" in msg
    assert "journalctl -u prelude-distribution.service" in msg


def test_failure_message_converts_aware_clock_to_kst():
    msg = build_message(
        "prelude-close.service",
        now=datetime(2026, 7, 25, 0, 10, tzinfo=timezone.utc),
    )

    assert "2026-07-25 09:10:00 KST" in msg


def test_failure_message_rejects_naive_clock():
    with pytest.raises(ValueError, match="timezone-aware"):
        build_message(
            "prelude-close.service",
            now=datetime(2026, 7, 25, 9, 10),
        )


def test_every_operational_service_has_onfailure():
    services = [
        p for p in Path("deploy").glob("prelude-*.service")
        if "failure-alert@" not in p.name
        # 기록 전용(record-only) 유닛은 의도적으로 경보 없음 — 별도 테스트
        # (tests/test_systemd_deploy_contract.py::
        #  test_depth_record_units_are_record_only_without_failure_alert,
        #  test_preopen_shadow_units_are_record_only_and_bounded) 가 OnFailure 부재를 단언.
        and p.name not in {"prelude-depth-record.service", "prelude-preopen-shadow.service"}
    ]

    assert {service.name for service in services} == {
        "prelude-backup.service",
        "prelude-close.service",
        "prelude-distribution.service",
        "prelude-heartbeat.service",
        "prelude-microstructure.service",
        "prelude-preopen-close.service",
        "prelude-preopen.service",
        "prelude-publish-dashboard.service",
        "prelude-selftest.service",
    }
    for service in services:
        text = service.read_text()
        assert "OnFailure=prelude-failure-alert@%n.service" in text, service


def test_installer_copies_failure_template():
    text = Path("deploy/install_systemd.sh").read_text()

    assert "prelude-failure-alert@.service" in text


def test_installer_rejects_active_prelude_cron_before_enabling_timers():
    text = Path("deploy/install_systemd.sh").read_text()

    guard = text.index("ERROR: active prelude cron found")
    transaction = text.index("TRANSACTION_ACTIVE=1")
    activation = text.index('for unit in "${AFFECTED_TIMERS[@]}"; do', transaction)
    first_enable = text.index('"$SYSTEMCTL_BIN" enable "$unit"', activation)
    assert "reject_legacy_cron" in text
    assert guard < first_enable
