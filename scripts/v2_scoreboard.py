#!/usr/bin/env python3
"""pump_hunter_v2 시한부 판정(2026-09-01) 일일 자동 채점.

사전등록 블록(PHASES.md, 커밋 1d46a9c) 기준을 그대로 기계화:
  - 승격 기준: closed n>=200 AND mean net>0 AND 95% CI 0 제외 AND (2레짐 관측 OR per-day t>=2)
  - 조기 KILL: closed 누적 mean net < 0 전환 시 즉시
비용: 왕복 0.15% 차감 (CLAUDE.md §2.4, 커밋 1d46a9c 산정과 동일).

출력: 한 줄 요약(stdout) + output/v2_scoreboard.json.
종료코드: 0=정상 / 21=조기킬 조건 발동(heartbeat가 텔레그램 알림).
cron 아님 — heartbeat.sh가 매일 호출.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "output" / "shadow_ledger_pump_hunter_v2.csv"
OUT = ROOT / "output" / "v2_scoreboard.json"
COST_PCT = 0.15
JUDGMENT_DAY = date(2026, 9, 1)


def main() -> int:
    rows = [r for r in csv.DictReader(open(LEDGER)) if r.get("status") == "closed" and r.get("realized_pct")]
    nets = [float(r["realized_pct"]) - COST_PCT for r in rows]
    n = len(nets)
    if n < 2:
        print(f"[v2-scoreboard] closed n={n} — 표본 부족, 채점 보류")
        return 0

    mean = statistics.mean(nets)
    sd = statistics.stdev(nets)
    se = sd / math.sqrt(n)
    ci = (mean - 1.96 * se, mean + 1.96 * se)

    # per-day (클러스터) t
    by_day = defaultdict(list)
    for r, v in zip(rows, nets):
        by_day[r["date"]].append(v)
    daily = [statistics.mean(v) for v in by_day.values()]
    k = len(daily)
    t_daily = (statistics.mean(daily) / (statistics.stdev(daily) / math.sqrt(k))) if k >= 2 and statistics.stdev(daily) > 0 else 0.0

    regimes = sorted({r.get("btc_regime", "?") for r in rows})

    crit = {
        "n>=200": n >= 200,
        "mean>0": mean > 0,
        "CI_0_제외": ci[0] > 0,
        "2레짐_or_t>=2": len(regimes) >= 2 or t_daily >= 2,
    }
    early_kill = mean < 0
    days_left = (JUDGMENT_DAY - date.today()).days

    payload = {
        "asof": date.today().isoformat(),
        "closed_n": n, "mean_net_pct": round(mean, 4),
        "ci95": [round(ci[0], 4), round(ci[1], 4)],
        "per_day_t": round(t_daily, 3), "trade_days": k,
        "regimes": regimes, "criteria": crit,
        "criteria_met": sum(crit.values()),
        "early_kill_breached": early_kill,
        "days_to_judgment": days_left,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1))

    line = (f"[v2-scoreboard] D-{days_left} n={n} mean_net={mean:+.3f}% "
            f"CI[{ci[0]:+.3f},{ci[1]:+.3f}] t_day={t_daily:.2f} 기준충족 {sum(crit.values())}/4")
    if early_kill:
        print(line + " ⛔ 조기 KILL 조건 발동(누적 mean<0) — 사전등록상 radar 전체 KILL")
        return 21
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
