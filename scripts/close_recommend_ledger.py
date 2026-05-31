"""SHADOW recommend ledger 청산 — 'open' row 의 day-D 경로 실현 채우기 (기록만).

흐름:
  1. output/shadow_ledger_recommend.csv 의 status='open' 행 로드
  2. 청산 가능 조건: date <= asof - 1day (day-D 경로 [09:00 D, 09:00 D+1) 완전 마감)
  3. 15m 경로(data/upbit_15m.db)로 -3% SL / +5% TP 경로청산 시뮬:
       recommender_downside_exit_v1.simulate_path(bars, hard_sl=0.03, tp=0.05, trail=None)
       → exit_reason ∈ {SL, TP, EOD}, gross_return
     net realized = gross - 왕복 0.15% (ops-steward §0: 거래비용 항상 차감)
  4. 일봉 pump20_hit = (day-D high/open - 1) >= 0.20 (upbit_d1.db)
  5. exit_price/exit_reason/realized_pct/pump20_hit/closed_at 채움 + status='closed'

★★★ SHADOW(기록만): 텔레그램 발송·cron·업비트 주문/API key 전부 없음.
    청산은 과거 봉 데이터로 가상 평가만 한다 (실거래 X).

★ LEAK 방어: 청산은 진입일 D 의 경로(미래 봉)를 쓰지만, 이건 in-trade outcome
  으로 leak 이 아니다 (진입 결정은 D-1 까지로만 — signals.recommend 가 보장).
  15m 봉은 시간 오름차순으로만 순회, 같은 봉 SL·TP 동시면 SL 먼저(보수, simulate_path).

사용:
    python scripts/close_recommend_ledger.py                 # 어제까지 open close
    python scripts/close_recommend_ledger.py --asof 2026-06-02
    python scripts/close_recommend_ledger.py --dry-run       # 저장 X, 미리보기
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# -3% SL / +5% TP 경로청산 로직 재사용 (self-contained: prelude 내부 모듈만).
from scripts.recommender_downside_exit_v1 import simulate_path  # noqa: E402

from scripts.recommend_today import (  # noqa: E402
    RECOMMEND_LEDGER_COLS,
    SHADOW_RECOMMEND_LEDGER,
)

M15_DB = "data/upbit_15m.db"
D1_DB = "data/upbit_d1.db"
ROUND_TRIP_COST = 0.0015     # 왕복 0.15% (수수료 0.1% + 슬리피지 0.05%)
PUMP20_THRESH = 0.20         # 일봉 라벨: (high_D/open_D - 1) >= 0.20

# 청산 플랜 절대값 (사용자 확정 — 변경 금지). ledger 에 음수 sl_pct/양수 tp_pct 로
# 저장되지만 simulate_path 는 양수 magnitude 를 받으므로 abs 로 정규화해서 넘긴다.
HARD_SL = 0.03               # -3% 손절
TP = 0.05                    # +5% 익절

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("close_recommend")


def _load_15m_path(coin: str, date: pd.Timestamp) -> list:
    """[09:00 D, 09:00 D+1) 의 15m 봉 (open,high,low,close) 리스트. 시간 오름차순.
    (recommender_downside_exit_v1.load_paths 와 동일 윈도 — 단건 조회.)"""
    start = pd.Timestamp(date).strftime("%Y-%m-%d 09:00:00")
    end = (pd.Timestamp(date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d 09:00:00")
    conn = sqlite3.connect(M15_DB)
    rows = conn.execute(
        "SELECT open,high,low,close FROM candles WHERE market=? AND "
        "timestamp>=? AND timestamp<? ORDER BY timestamp", (coin, start, end)
    ).fetchall()
    conn.close()
    return rows


def _daily_pump20(coin: str, date: pd.Timestamp) -> dict:
    """day-D 일봉 open/high → pump20_hit. timestamp = 'YYYY-MM-DD 09:00:00'."""
    ts = pd.Timestamp(date).strftime("%Y-%m-%d 09:00:00")
    conn = sqlite3.connect(D1_DB)
    row = conn.execute(
        "SELECT open, high FROM candles WHERE market=? AND timestamp=?", (coin, ts)
    ).fetchone()
    conn.close()
    if row is None:
        return {"status": "no_d1"}
    o, h = float(row[0]), float(row[1])
    if o <= 0:
        return {"status": "bad_open"}
    return {"status": "ok", "pump20_hit": int((h / o - 1.0) >= PUMP20_THRESH)}


def close_recommend_ledger(ledger_path: str, asof: pd.Timestamp,
                           dry_run: bool, log: logging.Logger) -> None:
    p = Path(ledger_path)
    if not p.exists():
        log.error("ledger missing: %s (recommend_today.py 먼저 실행)", p)
        sys.exit(1)

    ledger = pd.read_csv(p)
    for c in RECOMMEND_LEDGER_COLS:
        if c not in ledger.columns:
            ledger[c] = pd.NA
    ledger["status"] = ledger["status"].fillna("").astype(str)
    # close-fill 컬럼은 read_csv 가 빈 값으로 float64 추론 → 문자열(SL/TP/iso) 대입 시
    # FutureWarning. object 로 캐스트해서 in-place 대입을 안전하게 한다.
    for c in ["exit_reason", "closed_at", "exit_price", "realized_pct", "pump20_hit"]:
        ledger[c] = ledger[c].astype(object)
    log.info("ledger: %d rows total", len(ledger))

    cutoff_date = (asof - pd.Timedelta(days=1)).date()
    open_mask = ledger["status"].isin(["open", "no_data"])
    n_open = int(open_mask.sum())
    log.info("  open: %d, closing date <= %s", n_open, cutoff_date)
    if n_open == 0:
        log.info("nothing to close")
        return

    n_closed = 0
    n_no_data = 0
    closed_rows = []
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for idx, r in ledger[open_mask].iterrows():
        entry_date = pd.to_datetime(r["date"])
        if entry_date.date() > cutoff_date:
            continue  # day-D 경로 아직 마감 전

        coin = r["coin"]
        bars = _load_15m_path(coin, entry_date)
        if not bars:
            ledger.at[idx, "status"] = "no_data"
            n_no_data += 1
            continue

        gross, outcome = simulate_path(bars, HARD_SL, TP, None)
        if not np.isfinite(gross):
            ledger.at[idx, "status"] = "no_data"
            n_no_data += 1
            continue

        # simulate_path outcome: 'sl'|'tp'|'eod'|'trail'(여기선 trail None). → SL/TP/EOD.
        reason = {"sl": "SL", "tp": "TP", "eod": "EOD"}.get(outcome, outcome.upper())
        entry = float(bars[0][0])
        # exit_price 복원: SL=entry*(1-0.03), TP=entry*(1+0.05), EOD=마지막봉 close.
        if outcome == "sl":
            exit_price = entry * (1 - HARD_SL)
        elif outcome == "tp":
            exit_price = entry * (1 + TP)
        else:
            exit_price = float(bars[-1][3])
        net = gross - ROUND_TRIP_COST   # 거래비용 차감 후 net

        d1 = _daily_pump20(coin, entry_date)
        pump20 = d1.get("pump20_hit", pd.NA)

        ledger.at[idx, "exit_price"] = round(exit_price, 8)
        ledger.at[idx, "exit_reason"] = reason
        ledger.at[idx, "realized_pct"] = round(net * 100, 4)
        ledger.at[idx, "pump20_hit"] = pump20
        ledger.at[idx, "closed_at"] = now_iso
        ledger.at[idx, "status"] = "closed"
        n_closed += 1
        closed_rows.append({
            "date": r["date"], "coin": coin, "rank": r["rank"],
            "exit_reason": reason, "realized_pct_net": net * 100,
            "pump20_hit": pump20,
        })

    log.info("  closed: %d, no_data: %d", n_closed, n_no_data)

    if not dry_run:
        ordered = [c for c in RECOMMEND_LEDGER_COLS if c in ledger.columns]
        extras = [c for c in ledger.columns if c not in ordered]
        ledger = ledger[ordered + extras]
        ledger.to_csv(p, index=False)
        log.info("saved %s", p)

    if closed_rows:
        df = pd.DataFrame(closed_rows)
        print("\n=== closed recommend rows (net, 0.15%% cost 차감) ===")
        print(df.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
        print("\n=== exit reason dist ===")
        print(df["exit_reason"].value_counts().to_string())
        print(f"\n=== net realized: mean {df['realized_pct_net'].mean():+.2f}%, "
              f"sum {df['realized_pct_net'].sum():+.2f}% (n={len(df)}) ===")
        hits = df["pump20_hit"].dropna()
        if len(hits):
            print(f"=== pump20 hit: {hits.mean()*100:.1f}% ({int(hits.sum())}/{len(hits)}) ===")


def main():
    ap = argparse.ArgumentParser(description="SHADOW recommend ledger 청산 (기록만)")
    ap.add_argument("--ledger", type=str, default=SHADOW_RECOMMEND_LEDGER)
    ap.add_argument("--asof", type=str, default=None,
                    help="기준 시점 (default=now); open 중 date < asof-1day 만 close")
    ap.add_argument("--dry-run", action="store_true", help="저장 X, 미리보기만")
    args = ap.parse_args()

    asof = pd.Timestamp(args.asof) if args.asof else pd.Timestamp.now()
    close_recommend_ledger(args.ledger, asof, args.dry_run, log)


if __name__ == "__main__":
    main()
