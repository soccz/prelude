from __future__ import annotations

from scripts.health_check import db_checks_for_channel, log_names_for_channel


def test_health_check_preopen_uses_15m_gate():
    checks = dict(db_checks_for_channel("preopen"))

    assert "data/upbit_d1.db" in checks
    assert "data/upbit_15m.db" in checks
    assert "data/upbit_4h.db" not in checks
    assert checks["data/upbit_15m.db"] == 2


def test_health_check_distribution_uses_4h_gate():
    checks = dict(db_checks_for_channel("distribution"))

    assert "data/upbit_d1.db" in checks
    assert "data/upbit_4h.db" in checks
    assert "data/upbit_15m.db" not in checks
    assert checks["data/upbit_4h.db"] == 8


def test_health_check_log_names_are_channel_specific():
    assert log_names_for_channel("preopen", "20260525") == [
        "output/cron_preopen_20260525.log"
    ]
    assert log_names_for_channel("distribution", "20260525") == [
        "output/cron_dist_20260525.log"
    ]
