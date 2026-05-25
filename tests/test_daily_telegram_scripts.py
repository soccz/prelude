from __future__ import annotations

from pathlib import Path


def test_distribution_daily_script_sends_silence_telegram():
    text = Path("scripts/daily_run_distribution.sh").read_text()

    assert "--send-telegram" in text
    assert "--send-silence-telegram" in text


def test_preopen_daily_script_sends_silence_telegram():
    text = Path("scripts/daily_run_preopen.sh").read_text()
    predict_block = text.split("python scripts/predict_preopen_trigger.py", 1)[1]

    assert "--no-telegram" not in predict_block
    assert "--send-silence-telegram" in text
