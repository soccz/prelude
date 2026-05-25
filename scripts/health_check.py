"""Daily health check — cron 정상 작동 + DB 신선도 + risk state 점검.

매일 KST 09:30 또는 별도 cron.
문제 발견 시 텔레그램 alert.
"""
from __future__ import annotations

import json
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.database import latest_timestamp
from notifier.telegram import send_telegram


def check_db_freshness(db_path: str, market: str = "KRW-BTC", max_lag_hours: int = 30) -> tuple[bool, str]:
    latest = latest_timestamp(db_path, market)
    if latest is None:
        return False, f"DB {db_path}: no data for {market}"
    lag = datetime.now() - latest.to_pydatetime()
    if lag.total_seconds() / 3600 > max_lag_hours:
        return False, f"DB {db_path}: stale by {lag}"
    return True, f"DB {db_path}: latest {latest} (lag {lag.total_seconds()/3600:.1f}h)"


def db_checks_for_channel(channel: str) -> list[tuple[str, int]]:
    """Return required DB freshness checks for the operating channel."""
    if channel == "preopen":
        return [
            ("data/upbit_d1.db", 48),
            ("data/upbit_15m.db", 2),
        ]
    if channel == "distribution":
        return [
            ("data/upbit_d1.db", 30),
            ("data/upbit_4h.db", 8),
        ]
    if channel == "all":
        return [
            ("data/upbit_d1.db", 30),
            ("data/upbit_4h.db", 8),
            ("data/upbit_15m.db", 2),
        ]
    raise ValueError(f"unknown channel: {channel}")


def log_names_for_channel(channel: str, today: str) -> list[str]:
    if channel == "preopen":
        return [f"output/cron_preopen_{today}.log"]
    if channel == "distribution":
        return [f"output/cron_dist_{today}.log"]
    if channel == "all":
        return [
            f"output/cron_preopen_{today}.log",
            f"output/cron_dist_{today}.log",
        ]
    raise ValueError(f"unknown channel: {channel}")


def check_log_age(log_path: str, max_age_hours: int = 26) -> tuple[bool, str]:
    p = Path(log_path)
    if not p.exists():
        return False, f"{log_path}: missing"
    mtime = datetime.fromtimestamp(p.stat().st_mtime)
    age = datetime.now() - mtime
    if age.total_seconds() / 3600 > max_age_hours:
        return False, f"{log_path}: stale by {age}"
    return True, f"{log_path}: {age.total_seconds()/3600:.1f}h ago"


def check_risk_state() -> tuple[bool, str]:
    p = Path("output/risk_state.json")
    if not p.exists():
        return True, "risk_state: not initialized"
    with open(p) as f:
        s = json.load(f)
    if s.get("is_active", True):
        return True, f"risk: ACTIVE"
    return False, f"risk: SILENCED until {s.get('silenced_until')} ({s.get('trigger_reason')})"


def check_drift_state() -> tuple[bool, str]:
    p = Path("output/drift_state.json")
    if not p.exists():
        return True, "drift_state: not initialized"
    with open(p) as f:
        s = json.load(f)
    state = s.get("state", "OK")
    if state in ("WARN", "FREEZE"):
        return False, f"drift: {state} ({s.get('triggers')})"
    return True, f"drift: {state}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", default="distribution",
                        choices=["distribution", "preopen", "all"],
                        help="Operating channel to validate")
    parser.add_argument("--no-telegram", action="store_true",
                        help="Do not send Telegram on failure; exit nonzero only")
    args = parser.parse_args()

    print(f"=== prelude health {datetime.now()} channel={args.channel} ===")
    issues = []

    # DB freshness
    for db, max_lag in db_checks_for_channel(args.channel):
        ok, msg = check_db_freshness(db, max_lag_hours=max_lag)
        print(f"  {'OK' if ok else 'FAIL'}: {msg}")
        if not ok:
            issues.append(msg)

    # Log age
    today = datetime.now().strftime("%Y%m%d")
    for log_name in log_names_for_channel(args.channel, today):
        if Path(log_name).exists():
            ok, msg = check_log_age(log_name)
            print(f"  {'OK' if ok else 'FAIL'}: {msg}")
            if not ok:
                issues.append(msg)

    # Risk
    ok, msg = check_risk_state()
    print(f"  {'OK' if ok else 'WARN'}: {msg}")
    if not ok:
        issues.append(msg)

    # Drift
    ok, msg = check_drift_state()
    print(f"  {'OK' if ok else 'WARN'}: {msg}")
    if not ok:
        issues.append(msg)

    if issues:
        msg = "⚠️ prelude health issues:\n" + "\n".join(f"  • {i}" for i in issues)
        if not args.no_telegram:
            send_telegram(msg)
        sys.exit(1)
    else:
        print("\n✅ ALL OK")
        sys.exit(0)


if __name__ == "__main__":
    main()
