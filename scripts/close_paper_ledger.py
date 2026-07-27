"""Paper ledger close-out — entered rows 의 다음날 realized 채우기.

흐름:
  1. output/paper_ledger.csv 의 status="entered" 행 로드
  2. 각 행의 (coin, date) 에서 next_date = date + 1day 의 4h bars 조회
  3. next-day 4h bars 로 realized 계산:
     - next_max_return_pct
     - next_close_return_pct
     - hit_h2 (4h 안 +3% high)
     - hit_h6 (24h 안 +5% high)
     - hit_h5 (24h 안 +20% high)
  4. status → "closed" + 저장

운영:
  - 매일 KST 09:30 cron 으로 돌림 (어제 일봉 100% 마감 후)
  - entered 인 채로 남는 row = 다음날 데이터 없음 (delisted, 휴장, 미래)
  - close 시 paper ledger 의 calibration 검증 가능

사용:
    python scripts/close_paper_ledger.py
    python scripts/close_paper_ledger.py --asof 2025-11-07  # 시뮬
    python scripts/close_paper_ledger.py --dry-run          # 결과만 출력
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ledger.csv_store import atomic_write_csv, ledger_lock
from ledger.path_quality import PATH_QUALITY_COLS, assess_4h_path
from scripts.predict_today_distribution import PAPER_LEDGER_COLS


MAX_BARS = 6  # 24h / 4h
DEFAULT_SHADOW_LEDGER = "output/shadow_ledger_distribution.csv"


def compute_realized(coin: str, target_date: pd.Timestamp,
                     candles_4h_db: str = "data/upbit_4h.db") -> dict:
    """coin 의 target_date 6개 4h bars로 realized 계산.

    target_date = paper_ledger.date = 예측 대상 day (= label day after leak fix).
    target_date 의 09:00 KST = first 4h bar (timestamp - 9h = target_date 00:00)
    """
    assessment = assess_4h_path(coin, target_date, db_path=candles_4h_db)
    metadata = {
        **assessment.metadata(),
        "path_start_at": (
            assessment.timestamps[0].isoformat()
            if assessment.timestamps else None
        ),
        "path_used_bars": len(assessment.bars),
    }
    if not assessment.path_complete:
        return {
            "status": "no_data",
            "notes": assessment.path_quality,
            **metadata,
        }

    bars = np.asarray(assessment.bars, dtype=float)
    if bars.shape != (MAX_BARS, 4) or not np.isfinite(bars).all():
        return {
            "status": "no_data",
            "notes": "assessment_contract_invalid",
            **metadata,
            "path_complete": False,
            "path_quality": "assessment_contract_invalid",
        }
    open_t = float(bars[0, 0])
    close_t = float(bars[-1, 3])
    if open_t <= 0:
        return {
            "status": "no_data",
            "notes": "bad_open",
            **metadata,
            "path_complete": False,
            "path_quality": "bad_open",
        }

    highs = bars[:, 1]
    lows = bars[:, 2]
    high_max = float(highs.max())
    low_min = float(lows.min())

    next_max_return = high_max / open_t - 1
    next_close_return = close_t / open_t - 1
    next_min_return = low_min / open_t - 1

    # head hits (head 정의와 동일)
    # h2: 4h 안 (== first bar) high >= open*1.03
    n_bars_h2 = max(1, 4 // 4)  # 1
    n_bars_h6 = max(1, 24 // 4)  # 6
    n_bars_h5 = 6  # 24h

    hit_h2 = bool((highs[:n_bars_h2] >= open_t * 1.03).any())
    hit_h6 = bool((highs[:n_bars_h6] >= open_t * 1.05).any())
    hit_h5 = bool((highs[:n_bars_h5] >= open_t * 1.20).any())

    return {
        "status": "closed",
        **metadata,
        "next_open": open_t,
        "next_high": high_max,
        "next_low": low_min,
        "next_close": close_t,
        "next_max_return_pct": next_max_return * 100,
        "next_min_return_pct": next_min_return * 100,
        "next_close_return_pct": next_close_return * 100,
        "hit_h2": int(hit_h2),
        "hit_h6": int(hit_h6),
        "hit_h5": int(hit_h5),
    }


def close_ledger_file(
    ledger_path: str,
    candles_4h_db: str,
    asof: pd.Timestamp,
    dry_run: bool,
    log: logging.Logger,
    *,
    required: bool = True,
) -> None:
    p = Path(ledger_path)
    with ledger_lock(p):
        _close_ledger_file_locked(
            p,
            candles_4h_db,
            asof,
            dry_run,
            log,
            required=required,
        )


def _close_ledger_file_locked(
    p: Path,
    candles_4h_db: str,
    asof: pd.Timestamp,
    dry_run: bool,
    log: logging.Logger,
    *,
    required: bool,
) -> None:
    if not p.exists():
        msg = f"ledger missing: {p}"
        if required:
            log.error(msg)
            sys.exit(1)
        log.info(msg)
        return

    ledger = pd.read_csv(p)
    # Ensure all expected columns exist (older ledgers may miss new realized cols).
    for col in PAPER_LEDGER_COLS:
        if col not in ledger.columns:
            ledger[col] = np.nan
    for col in PATH_QUALITY_COLS:
        if col not in ledger.columns:
            ledger[col] = pd.NA
        ledger[col] = ledger[col].astype(object)
    ledger["status"] = ledger["status"].fillna("").astype(str)
    ledger["notes"] = ledger["notes"].fillna("").astype(str)
    log.info(f"ledger: {len(ledger)} rows total")

    cutoff_date = (asof - pd.Timedelta(days=1)).date()

    # no_data rows are retried because a failed/late 4h update should not
    # permanently prevent close-out on the next successful run.
    entered_mask = ledger["status"].isin(["entered", "no_data"])
    n_entered = int(entered_mask.sum())
    log.info(f"  entered: {n_entered}, closing date <= {cutoff_date}")

    if n_entered == 0:
        log.info("nothing to close")
        return

    n_closed = 0
    n_no_data = 0
    rows_to_update = []
    for idx, r in ledger[entered_mask].iterrows():
        entry_date = pd.to_datetime(r["date"]).date()
        # paper_ledger.date 은 예측 대상 day (= label day after leak fix)
        target_date = pd.Timestamp(entry_date)
        # 닫으려면 target_date 의 4h bars 가 모두 마감되어야 함
        # = target_date < (asof - 1day)
        if target_date.date() > cutoff_date:
            continue

        result = compute_realized(r["coin"], target_date, candles_4h_db)
        if result["status"] == "no_data":
            for key in PATH_QUALITY_COLS:
                if key in result:
                    ledger.at[idx, key] = result[key]
            ledger.at[idx, "status"] = "no_data"
            ledger.at[idx, "notes"] = result.get(
                "notes", f"no 4h data for {target_date.date()}"
            )
            n_no_data += 1
            continue
        if result["status"] != "closed":
            continue

        for k, v in result.items():
            if k == "status":
                continue
            ledger.at[idx, k] = v
        ledger.at[idx, "status"] = "closed"
        n_closed += 1
        rows_to_update.append({
            "date": r["date"], "coin": r["coin"], "setups": r["setup_ids"],
            "max_ret_pct": result["next_max_return_pct"],
            "close_ret_pct": result["next_close_return_pct"],
            "h2": result["hit_h2"], "h6": result["hit_h6"], "h5": result["hit_h5"],
        })

    log.info(f"  closed: {n_closed}, no_data: {n_no_data}")

    if not dry_run:
        # reorder to canonical schema (extra cols at the end are preserved)
        ordered = [c for c in PAPER_LEDGER_COLS if c in ledger.columns]
        extras = [c for c in ledger.columns if c not in ordered]
        ledger = ledger[ordered + extras]
        atomic_write_csv(ledger, p)
        log.info(f"saved {p}")

    # show closed rows summary
    if rows_to_update:
        print(f"\n=== Closed rows ({len(rows_to_update)}) ===")
        df = pd.DataFrame(rows_to_update)
        print(df.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

        # Per-head hit rate (closed only)
        print("\n=== Realized hit rate (closed alerts) ===")
        for h in ["h2", "h6", "h5"]:
            n = len(df)
            h_pct = float(df[h].mean()) * 100
            print(f"  hit_{h}: {h_pct:.1f}% ({int(df[h].sum())}/{n})")

        # Per-setup hit rate (intersection)
        print("\n=== Per-setup hit rate ===")
        setups_seen = set()
        for s in df["setups"].fillna(""):
            for s_id in s.split("+"):
                if s_id:
                    setups_seen.add(s_id)
        for s_id in sorted(setups_seen):
            mask = df["setups"].str.contains(s_id, na=False)
            n = int(mask.sum())
            if n == 0:
                continue
            sub = df[mask]
            h2_pct = float(sub["h2"].mean()) * 100
            h6_pct = float(sub["h6"].mean()) * 100
            h5_pct = float(sub["h5"].mean()) * 100
            print(f"  {s_id} (n={n}): hit_h2 {h2_pct:.1f}%, hit_h6 {h6_pct:.1f}%, hit_h5 {h5_pct:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-ledger", default="output/paper_ledger.csv")
    parser.add_argument("--shadow-ledger", default=DEFAULT_SHADOW_LEDGER)
    parser.add_argument("--skip-shadow-ledger", action="store_true")
    parser.add_argument("--upbit-4h", default="data/upbit_4h.db")
    parser.add_argument("--asof", type=str, help="기준 시점 (default=now); entered 중 date < asof - 1day 만 close")
    parser.add_argument("--dry-run", action="store_true", help="저장 X, 변경 미리보기만")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("close")

    asof = (
        pd.Timestamp(args.asof)
        if args.asof
        else pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None)
    )
    close_ledger_file(args.paper_ledger, args.upbit_4h, asof, args.dry_run, log, required=True)

    if not args.skip_shadow_ledger and Path(args.shadow_ledger) != Path(args.paper_ledger):
        close_ledger_file(args.shadow_ledger, args.upbit_4h, asof, args.dry_run, log, required=False)


if __name__ == "__main__":
    main()
