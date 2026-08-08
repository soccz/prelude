from __future__ import annotations

from pathlib import Path


def test_distribution_daily_script_keeps_distribution_record_only():
    text = Path("scripts/daily_run_distribution.sh").read_text()

    predict_block = text.split("python scripts/predict_today_distribution.py", 1)[1]
    predict_block = predict_block.split("echo \"[7/10]", 1)[0]

    assert "--send-telegram" not in predict_block
    assert "--send-silence-telegram" not in predict_block
    assert "python scripts/recommend_send.py --slot open" in text
    assert "python scripts/pump_detector_today.py" in text
    health_helper = text.split("run_health_with_d1_reconcile() {", 1)[1]
    health_helper = health_helper.split("\n}\n", 1)[0]
    assert health_helper.count("python scripts/health_check.py") == 2
    assert '--channel "$channel" --no-telegram' in health_helper
    assert "-m data.collector_d1 \\\n        --refresh-current-boundary" in (
        health_helper
    )
    assert text.index("python -m data.collector_d1 --update") < text.index(
        "run_health_with_d1_reconcile recommend"
    )
    assert text.index(
        "run_health_with_d1_reconcile recommend"
    ) < text.index("python scripts/recommend_send.py --slot open")
    assert text.index("python scripts/recommend_send.py --slot open") < text.index(
        "python -m data.collector_4h --all"
    )
    assert text.index("python -m data.collector_4h --all") < text.index(
        "run_health_with_d1_reconcile distribution"
    )
    assert text.index("run_health_with_d1_reconcile distribution") < text.index(
        "python scripts/predict_today_distribution.py"
    )


def test_preopen_daily_script_keeps_preopen_predict_record_only():
    text = Path("scripts/daily_run_preopen.sh").read_text()
    predict_block = text.split("python scripts/predict_preopen_trigger.py", 1)[1]
    predict_block = predict_block.split("echo \"[done]", 1)[0]

    assert "--no-telegram" in predict_block
    assert "python scripts/recommend_send.py --slot preopen" in text
    assert text.index(
        "python scripts/health_check.py \\\n"
        "        --channel recommend-preopen"
    ) < text.index("python scripts/recommend_send.py --slot preopen")
    assert text.index("python scripts/recommend_send.py --slot preopen") < text.index(
        "python -m data.collector_15m_upbit --all"
    )
    assert text.index(
        "python scripts/health_check.py \\\n"
        "        --channel preopen"
    ) < text.index("python scripts/predict_preopen_trigger.py")


def test_distribution_close_script_closes_pump_hunter_ledger():
    text = Path("scripts/daily_close_distribution.sh").read_text()

    assert "output/shadow_ledger_pump_hunter.csv" in text
    assert "python -m ops.policy_competition" in text
