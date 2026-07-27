#!/usr/bin/env python3
"""Send a concise Telegram alert when a prelude systemd unit fails."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from notifier.telegram import send_telegram  # noqa: E402


def build_message(unit: str, *, now: datetime | None = None) -> str:
    ts = now or datetime.now(ZoneInfo("Asia/Seoul"))
    if ts.tzinfo is None:
        raise ValueError("systemd failure alert clock must be timezone-aware")
    ts = ts.astimezone(ZoneInfo("Asia/Seoul"))
    return (
        "⚠️ prelude systemd failure\n"
        f"- unit: {unit}\n"
        f"- time: {ts:%Y-%m-%d %H:%M:%S KST}\n"
        f"- check: journalctl -u {unit} -n 100"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", required=True)
    args = parser.parse_args()
    return 0 if send_telegram(build_message(args.unit)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
