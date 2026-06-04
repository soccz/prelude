from __future__ import annotations

from pathlib import Path


def test_distribution_daily_script_keeps_distribution_record_only():
    text = Path("scripts/daily_run_distribution.sh").read_text()

    predict_block = text.split("python scripts/predict_today_distribution.py", 1)[1]
    predict_block = predict_block.split("echo \"[4/5] recommend_send", 1)[0]

    assert "--send-telegram" not in predict_block
    assert "--send-silence-telegram" not in predict_block
    assert "python scripts/recommend_send.py --slot open" in text
    assert "python scripts/pump_detector_today.py" in text


def test_preopen_daily_script_keeps_preopen_predict_record_only():
    text = Path("scripts/daily_run_preopen.sh").read_text()
    predict_block = text.split("python scripts/predict_preopen_trigger.py", 1)[1]
    predict_block = predict_block.split("echo \"[4/4] recommend_send", 1)[0]

    assert "--no-telegram" in predict_block
    assert "python scripts/recommend_send.py --slot preopen" in text


def test_distribution_close_script_closes_pump_hunter_ledger():
    text = Path("scripts/daily_close_distribution.sh").read_text()

    assert "output/shadow_ledger_pump_hunter.csv" in text
    assert "python -m ops.policy_competition" in text
