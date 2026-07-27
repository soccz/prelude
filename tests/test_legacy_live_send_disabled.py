from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("script", "args", "slot"),
    [
        ("scripts/predict_preopen_trigger.py", [], "preopen"),
        (
            "scripts/predict_preopen_trigger.py",
            ["--allow-late-run"],
            "preopen",
        ),
        (
            "scripts/predict_today_distribution.py",
            ["--send-telegram"],
            "open",
        ),
        ("scripts/predict_today.py", ["--send-telegram"], "open"),
        ("scripts/predict_today_legacy.py", [], "open"),
    ],
)
def test_legacy_manual_live_send_paths_fail_before_output_mutation(
    tmp_path: Path,
    script: str,
    args: list[str],
    slot: str,
) -> None:
    out_dir = tmp_path / "must-not-exist"
    command = [sys.executable, script, *args]
    if script != "scripts/predict_today_legacy.py":
        command.extend(["--out-dir", str(out_dir)])
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "live Telegram sending is disabled" in result.stderr
    assert (
        f"python scripts/recommend_send.py --slot {slot}"
        in result.stderr
    )
    assert not out_dir.exists()


@pytest.mark.parametrize(
    ("script", "required_fragment"),
    [
        ("scripts/predict_preopen_trigger.py", "--no-telegram"),
        ("scripts/predict_today_distribution.py", "recommend_send.py"),
        ("scripts/predict_today.py", "recommend_send.py"),
        ("scripts/predict_today_legacy.py", "--dry-run"),
    ],
)
def test_legacy_runner_help_keeps_record_only_instruction(
    script: str,
    required_fragment: str,
) -> None:
    result = subprocess.run(
        [sys.executable, script, "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert required_fragment in result.stdout
