"""ledger.csv ↔ 어제 텔레그램 메시지 일관성 검증.

xsec_alpha 의 verify_telegram.py 패턴 (이전에 LONG 부호 버그로 모든 행 반대 표시된 적).

매일 KST 09:30 cron — post_open_run 의 일부로 자동 실행.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notifier.telegram import send_telegram


def verify_signs(ledger_df: pd.DataFrame) -> tuple[bool, list]:
    """net_return 부호와 exit_type 일관성."""
    issues = []
    for _, row in ledger_df.iterrows():
        nr = row.get("net_return_pct", 0)
        ex = row.get("exit_type", "")
        if ex == "tp" and nr <= 0:
            issues.append(f"{row['coin']}: TP with non-positive return {nr}")
        elif ex == "sl" and nr >= 0:
            issues.append(f"{row['coin']}: SL with non-negative return {nr}")
    return len(issues) == 0, issues


def verify_magnitude(ledger_df: pd.DataFrame, tp_pct: float = 0.10, sl_pct: float = 0.05) -> tuple[bool, list]:
    """TP / SL 도달 시 net_return 크기가 합리적인가."""
    issues = []
    for _, row in ledger_df.iterrows():
        nr = row.get("net_return_pct", 0)
        ex = row.get("exit_type", "")
        if ex == "tp" and abs(nr - tp_pct + 0.002) > 0.02:  # cost ~0.2%
            issues.append(f"{row['coin']}: TP magnitude {nr:+.4f} (expected ~{tp_pct - 0.002:+.4f})")
        elif ex == "sl" and abs(nr - (-sl_pct) + 0.002) > 0.02:
            issues.append(f"{row['coin']}: SL magnitude {nr:+.4f} (expected ~{-sl_pct - 0.002:+.4f})")
    return len(issues) == 0, issues


def main():
    ledger_path = Path("output/ledger.csv")
    if not ledger_path.exists():
        print("no ledger yet — skip verify")
        sys.exit(0)

    df = pd.read_csv(ledger_path)
    if len(df) == 0:
        print("ledger empty")
        sys.exit(0)

    df["entry_date"] = pd.to_datetime(df["entry_date"])
    yesterday = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    yesterday_df = df[df["entry_date"].dt.normalize() == yesterday]

    if len(yesterday_df) == 0:
        print(f"no positions for {yesterday.date()}")
        sys.exit(0)

    print(f"=== verify {yesterday.date()} ({len(yesterday_df)} positions) ===")

    ok_signs, sign_issues = verify_signs(yesterday_df)
    ok_mag, mag_issues = verify_magnitude(yesterday_df)

    all_issues = sign_issues + mag_issues
    if all_issues:
        msg = (
            f"⚠️ prelude verify_telegram FAIL ({yesterday.date()})\n"
            + "\n".join(f"  • {i}" for i in all_issues[:10])
        )
        print(msg)
        send_telegram(msg)
        sys.exit(1)
    else:
        print("✅ ALL OK")
        sys.exit(0)


if __name__ == "__main__":
    main()
