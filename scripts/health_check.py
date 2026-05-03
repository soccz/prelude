"""Daily health check — cron 정상 작동 + DB 신선도 + risk state 점검.

매일 KST 09:30 또는 별도 cron.
문제 발견 시 텔레그램 alert.
"""
from __future__ import annotations

import json
import sys
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
    print(f"=== prelude health {datetime.now()} ===")
    issues = []

    # DB freshness
    for db in ["data/upbit_d1.db", "data/upbit_4h.db"]:
        ok, msg = check_db_freshness(db)
        print(f"  {'OK' if ok else 'FAIL'}: {msg}")
        if not ok:
            issues.append(msg)

    # Log age
    today = datetime.now().strftime("%Y%m%d")
    for log_name in [f"output/cron_daily_{today}.log"]:
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
        send_telegram(msg)
        sys.exit(1)
    else:
        print("\n✅ ALL OK")
        sys.exit(0)


if __name__ == "__main__":
    main()
