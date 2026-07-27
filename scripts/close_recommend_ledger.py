"""SHADOW recommend ledger 청산 — 'open' row 의 day-D 경로 실현 채우기 (기록만).

흐름:
  1. output/shadow_ledger_recommend.csv 의 status='open' 행 로드
  2. 청산 가능 조건: date <= asof - 1day (day-D 경로 [09:00 D, 09:00 D+1) 완전 마감)
  3. KRW-BTC 기준으로 DB horizon/수집 완결성을 확인한 뒤, 대상 종목의
     무체결 15m gap만 직전 close로 flat-fill한다. 불완전 경로는 청산 보류.
  4. delivery_ok=False 행은 성과 표본에서 제외한다. sent_at가 있으면 KST 기준
     엄격히 다음 15분봉부터 실행 가능한 경로만 사용한다.
  5. 15m 경로(data/upbit_15m.db)로 -3% SL / +5% TP 경로청산 시뮬:
       recommender_downside_exit_v1.simulate_path(bars, hard_sl=0.03, tp=0.05, trail=None)
       → exit_reason ∈ {SL, TP, EOD}, gross_return
     net realized = gross - 왕복 0.15% (ops-steward §0: 거래비용 항상 차감)
  6. 일봉 pump20_hit = (day-D high/open - 1) >= 0.20 (upbit_d1.db)
  7. exit_price/exit_reason/realized_pct/pump20_hit/closed_at 채움 + status='closed'

★★★ SHADOW(기록만): 텔레그램 발송·cron·업비트 주문/API key 전부 없음.
    청산은 과거 봉 데이터로 가상 평가만 한다 (실거래 X).

★ LEAK 방어: 청산은 진입일 D 의 경로(미래 봉)를 쓰지만, 이건 in-trade outcome
  으로 leak 이 아니다 (진입 결정은 D-1 까지로만 — signals.recommend 가 보장).
  15m 봉은 시간 오름차순으로만 순회, 같은 봉 SL·TP 동시면 SL 먼저(보수, simulate_path).

사용:
    python scripts/close_recommend_ledger.py --decision-date 2026-06-01
    python scripts/close_recommend_ledger.py --decision-date 2026-06-01 \
        --asof 2026-06-02
    python scripts/close_recommend_ledger.py --decision-date 2026-06-01 \
        --dry-run                                            # 저장 X
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# -3% SL / +5% TP 경로청산 로직 재사용 (self-contained: prelude 내부 모듈만).
from scripts.recommender_downside_exit_v1 import simulate_path  # noqa: E402

from data.database import connect_readonly  # noqa: E402
from ledger.config import ROUND_TRIP_COST_PCT  # noqa: E402
from ledger.csv_store import atomic_write_csv, ledger_lock  # noqa: E402
from ledger.exit_lab import EXIT_LAB_COLS, evaluate_exit_variants  # noqa: E402
from ledger.path_quality import (  # noqa: E402
    EXECUTION_QUALITY_COLS,
    PathAssessment,
    assess_15m_path,
    assess_15m_window,
    next_bar_boundary,
)
from ops.close_input_gate import (  # noqa: E402
    COHORTS,
    CloseInputError,
    validate_close_input,
)
from scripts.recommend_today import (  # noqa: E402
    RECOMMEND_LEDGER_COLS,
    SHADOW_RECOMMEND_LEDGER,
)

M15_DB = "data/upbit_15m.db"
D1_DB = "data/upbit_d1.db"
ROUND_TRIP_COST = ROUND_TRIP_COST_PCT  # 왕복 0.15% — ledger/config.py 단일 출처
PUMP20_THRESH = 0.20         # 일봉 라벨: (high_D/open_D - 1) >= 0.20
RECOMMEND_OUTCOME_AUDIT_COLS = [
    "execution_entry_open",
    "post_send_pump20_hit",
    "pump20_label_basis",
]
FORWARD_EXECUTION_TIME_ACTIVATION_DATE = date(2026, 7, 27)

# 청산 플랜 절대값 (사용자 확정 — 변경 금지). ledger 에 음수 sl_pct/양수 tp_pct 로
# 저장되지만 simulate_path 는 양수 magnitude 를 받으므로 abs 로 정규화해서 넘긴다.
HARD_SL = 0.03               # -3% 손절
TP = 0.05                    # +5% 익절

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("close_recommend")


def _load_15m_path(
    coin: str,
    date: pd.Timestamp,
    *,
    start_at: pd.Timestamp | None = None,
) -> PathAssessment:
    """실제 진입시각부터 24시간 또는 legacy day-D 경로를 판정한다."""
    if start_at is not None:
        return assess_15m_window(coin, start_at, db_path=M15_DB)
    return assess_15m_path(coin, date, db_path=M15_DB)


def _delivery_state(value: object) -> bool | None:
    """CSV bool/string/number delivery 값을 strict tri-state로 정규화."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if numeric in {0.0, 1.0}:
            return bool(numeric)
        raise ValueError(f"invalid numeric delivery_ok: {value!r}")
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "sent"}:
        return True
    if text in {"false", "0", "no", "n", "failed"}:
        return False
    raise ValueError(f"invalid delivery_ok: {value!r}")


def _sent_at_kst_naive(value: object) -> pd.Timestamp | None:
    """UTC/offset ISO sent_at를 Upbit DB의 timezone-naive KST timestamp로 변환."""
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    sent = pd.Timestamp(value)
    if sent.tzinfo is None:
        raise ValueError("sent_at must be timezone-aware")
    return sent.tz_convert("Asia/Seoul").tz_localize(None)


def _decision_completed_at_kst_naive(value: object) -> pd.Timestamp | None:
    """Immutable scorer/decision completion time in DB-naive KST."""
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    completed = pd.Timestamp(value)
    if completed.tzinfo is None:
        raise ValueError("decision_completed_at must be timezone-aware")
    return completed.tz_convert("Asia/Seoul").tz_localize(None)


def _daily_pump20(coin: str, date: pd.Timestamp) -> dict:
    """day-D 일봉 open/high → pump20_hit. timestamp = 'YYYY-MM-DD 09:00:00'."""
    ts = pd.Timestamp(date).strftime("%Y-%m-%d 09:00:00")
    with connect_readonly(D1_DB) as conn:
        row = conn.execute(
            "SELECT open, high FROM candles WHERE market=? AND timestamp=?",
            (coin, ts),
        ).fetchone()
    if row is None:
        return {"status": "no_d1"}
    o, h = float(row[0]), float(row[1])
    if o <= 0:
        return {"status": "bad_open"}
    return {"status": "ok", "pump20_hit": int((h / o - 1.0) >= PUMP20_THRESH)}


def close_recommend_ledger(
    ledger_path: str,
    asof: pd.Timestamp,
    dry_run: bool,
    log: logging.Logger,
    *,
    decision_date: pd.Timestamp,
    cohort: str | None = None,
    expected_mode: str | None = None,
    output_root: str | Path | None = None,
) -> None:
    p = Path(ledger_path)
    # Hold the same lock used by appenders for the entire read/assessment/write
    # transaction.  This intentionally favors correctness over close latency.
    with ledger_lock(p):
        if cohort is not None:
            spec = COHORTS.get(cohort)
            if spec is None:
                raise CloseInputError(f"unsupported close cohort: {cohort}")
            if expected_mode not in {
                "close",
                "skip-zero-pick",
                "skip-legacy-unverifiable",
            }:
                raise CloseInputError(
                    f"invalid expected close mode: {expected_mode!r}"
                )
            root = Path(output_root) if output_root is not None else p.parent
            expected_ledger = root / spec.ledger_name
            if p.resolve(strict=False) != expected_ledger.resolve(strict=False):
                raise CloseInputError(
                    f"{cohort} ledger path does not match canonical path"
                )
            actual_mode = validate_close_input(
                asof=str(decision_date.date()),
                cohort=cohort,
                output_root=root,
            )
            if actual_mode != expected_mode:
                raise CloseInputError(
                    f"{cohort} close mode changed under ledger lock: "
                    f"{expected_mode!r} -> {actual_mode!r}"
                )
            if actual_mode != "close":
                log.info(
                    "validated no-op under ledger lock: %s (%s)",
                    actual_mode,
                    decision_date.date(),
                )
                return
        _close_recommend_ledger_locked(
            p,
            asof,
            dry_run,
            log,
            decision_date=decision_date,
        )


def _close_recommend_ledger_locked(
    p: Path,
    asof: pd.Timestamp,
    dry_run: bool,
    log: logging.Logger,
    *,
    decision_date: pd.Timestamp,
) -> None:
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
    # exit lab 변형 컬럼 (TP10/noSL 등) — 같은 경로의 병렬 가상 평가 (record-only).
    for c in EXIT_LAB_COLS:
        if c not in ledger.columns:
            ledger[c] = pd.NA
        ledger[c] = ledger[c].astype(object)
    # 경로 품질 감사 컬럼. 과거 CLOSED row는 빈 값 그대로 보존하고 새 청산에만 채운다.
    for c in EXECUTION_QUALITY_COLS:
        if c not in ledger.columns:
            ledger[c] = pd.NA
        ledger[c] = ledger[c].astype(object)
    for c in RECOMMEND_OUTCOME_AUDIT_COLS:
        if c not in ledger.columns:
            ledger[c] = pd.NA
        ledger[c] = ledger[c].astype(object)
    log.info("ledger: %d rows total", len(ledger))

    cutoff_date = (asof - pd.Timedelta(days=1)).date()
    target_timestamp = pd.Timestamp(decision_date)
    if target_timestamp.tzinfo is not None:
        raise ValueError("decision_date must be timezone-naive")
    if target_timestamp != target_timestamp.normalize():
        raise ValueError("decision_date must not include a time")
    target_date = target_timestamp.date()
    if target_date > cutoff_date:
        raise ValueError(
            f"decision_date {target_date} is newer than close cutoff "
            f"{cutoff_date}"
        )
    open_mask = (
        ledger["status"].isin(["open", "no_data"])
        & (ledger["date"].astype(str) == str(target_date))
    )
    n_open = int(open_mask.sum())
    log.info(
        "  eligible: %d, exact decision date=%s, cutoff=%s",
        n_open,
        target_date,
        cutoff_date,
    )
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
        try:
            delivery_state = _delivery_state(r.get("delivery_ok"))
            sent_at = _sent_at_kst_naive(r.get("sent_at"))
            decision_completed_at = _decision_completed_at_kst_naive(
                r.get("decision_completed_at")
            )
            if delivery_state is True and sent_at is None:
                raise ValueError("successful delivery requires sent_at")
            if (
                sent_at is not None
                and decision_completed_at is not None
                and sent_at < decision_completed_at
                and sent_at.floor("s") != decision_completed_at.floor("s")
            ):
                raise ValueError(
                    "delivery cannot precede immutable decision completion"
                )
            if (
                entry_date.date()
                >= FORWARD_EXECUTION_TIME_ACTIVATION_DATE
                and decision_completed_at is None
            ):
                raise ValueError(
                    "strict-forward row requires decision_completed_at"
                )
        except (TypeError, ValueError):
            ledger.at[idx, "status"] = "no_data"
            ledger.at[idx, "path_complete"] = False
            ledger.at[idx, "path_quality"] = "invalid_delivery_metadata"
            n_no_data += 1
            continue
        if delivery_state is False:
            ledger.at[idx, "status"] = "not_delivered"
            ledger.at[idx, "path_complete"] = False
            ledger.at[idx, "path_quality"] = "delivery_failed"
            n_no_data += 1
            continue

        execution_at = (
            max(sent_at, decision_completed_at)
            if sent_at is not None and decision_completed_at is not None
            else (
                sent_at
                if sent_at is not None
                else decision_completed_at
            )
        )
        path_start_at = (
            next_bar_boundary(execution_at)
            if execution_at is not None
            else None
        )
        assessment = _load_15m_path(
            coin,
            entry_date,
            start_at=path_start_at,
        )
        for key, value in assessment.metadata().items():
            ledger.at[idx, key] = value
        if not assessment.path_complete:
            ledger.at[idx, "status"] = "no_data"
            n_no_data += 1
            continue

        bars = assessment.bars
        has_sent_at = sent_at is not None
        has_decision_at = decision_completed_at is not None
        if path_start_at is None and assessment.timestamps:
            path_start_at = assessment.timestamps[0]
        ledger.at[idx, "path_start_at"] = (
            path_start_at.isoformat() if path_start_at is not None else pd.NA
        )
        ledger.at[idx, "entry_observable_at"] = (
            path_start_at.isoformat() if path_start_at is not None else pd.NA
        )
        ledger.at[idx, "entry_price_source"] = (
            "15m_open_at_or_after_delivery"
            if has_sent_at
            else (
                "15m_open_at_or_after_decision"
                if has_decision_at
                else "day_open_09:00_legacy_no_receipt"
            )
        )
        ledger.at[idx, "path_used_bars"] = len(bars)
        if not bars:
            ledger.at[idx, "status"] = "no_data"
            ledger.at[idx, "path_complete"] = False
            ledger.at[idx, "path_quality"] = "no_executable_path"
            n_no_data += 1
            continue

        try:
            hard_sl = abs(float(r.get("sl_pct")))
            take_profit = float(r.get("tp_pct"))
        except (TypeError, ValueError):
            hard_sl = take_profit = float("nan")
        if (
            not np.isfinite(hard_sl)
            or not np.isfinite(take_profit)
            or hard_sl <= 0
            or take_profit <= 0
        ):
            ledger.at[idx, "status"] = "no_data"
            ledger.at[idx, "path_complete"] = False
            ledger.at[idx, "path_quality"] = "invalid_exit_contract"
            n_no_data += 1
            continue

        gross, outcome = simulate_path(bars, hard_sl, take_profit, None)
        if not np.isfinite(gross):
            ledger.at[idx, "status"] = "no_data"
            ledger.at[idx, "path_complete"] = False
            ledger.at[idx, "path_quality"] = "invalid_simulation_path"
            n_no_data += 1
            continue

        # simulate_path outcome: 'sl'|'tp'|'eod'|'trail'(여기선 trail None). → SL/TP/EOD.
        reason = {"sl": "SL", "tp": "TP", "eod": "EOD"}.get(outcome, outcome.upper())
        entry = float(bars[0][0])
        # Signal-time ``entry_open`` is immutable snapshot evidence.  The
        # executable post-delivery entry belongs in a separate outcome column
        # so a close cannot erase or rewrite the decision input.
        ledger.at[idx, "execution_entry_open"] = round(entry, 8)
        # exit_price 복원: SL=entry*(1-0.03), TP=entry*(1+0.05), EOD=마지막봉 close.
        if outcome == "sl":
            exit_price = entry * (1 - hard_sl)
        elif outcome == "tp":
            exit_price = entry * (1 + take_profit)
        else:
            exit_price = float(bars[-1][3])
        net = gross - ROUND_TRIP_COST   # 거래비용 차감 후 net

        d1 = _daily_pump20(coin, entry_date)
        pump20 = d1.get("pump20_hit", pd.NA)
        post_send_pump20 = int(
            max(float(bar[1]) for bar in bars) / entry - 1.0 >= PUMP20_THRESH
        )

        ledger.at[idx, "exit_price"] = round(exit_price, 8)
        ledger.at[idx, "exit_reason"] = reason
        ledger.at[idx, "realized_pct"] = round(net * 100, 4)
        ledger.at[idx, "pump20_hit"] = pump20
        # 기존 pump20_hit은 동결 비교용 full-day label로 보존한다. 실제 사용자가
        # 받을 수 있었던 상방은 sent_at 이후 경로로 별도 기록해 정의 충돌을 드러낸다.
        ledger.at[idx, "post_send_pump20_hit"] = post_send_pump20
        ledger.at[idx, "pump20_label_basis"] = (
            "legacy_full_day_D1_and_post_send_15m_both_recorded"
        )
        ledger.at[idx, "closed_at"] = now_iso
        ledger.at[idx, "status"] = "closed"
        # exit lab — 같은 bars 로 TP10/noSL 등 변형 병렬 기록 (record-only).
        lab = evaluate_exit_variants(bars, round_trip_cost=ROUND_TRIP_COST)
        if lab is not None:
            for k, v in lab.items():
                ledger.at[idx, k] = v
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
        atomic_write_csv(ledger, p)
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
                    help="기준 시점 (default=now); decision-date 완결성 cutoff")
    ap.add_argument(
        "--decision-date",
        required=True,
        help="증거 gate를 통과한 정확한 추천일 YYYY-MM-DD",
    )
    ap.add_argument(
        "--cohort",
        choices=sorted(COHORTS),
        help="revalidate this canonical evidence cohort under the ledger lock",
    )
    ap.add_argument(
        "--expected-mode",
        choices=("close", "skip-zero-pick", "skip-legacy-unverifiable"),
        help="mode emitted by the outer plan; must still match under lock",
    )
    ap.add_argument("--dry-run", action="store_true", help="저장 X, 미리보기만")
    args = ap.parse_args()

    asof = (
        pd.Timestamp(args.asof)
        if args.asof
        else pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None)
    )
    try:
        decision_date = pd.Timestamp(args.decision_date)
        if (
            args.decision_date != str(decision_date.date())
            or decision_date != decision_date.normalize()
        ):
            raise ValueError
    except (TypeError, ValueError):
        ap.error("--decision-date must be a canonical YYYY-MM-DD")
    if (args.cohort is None) != (args.expected_mode is None):
        ap.error("--cohort and --expected-mode must be supplied together")
    try:
        close_recommend_ledger(
            args.ledger,
            asof,
            args.dry_run,
            log,
            decision_date=decision_date,
            cohort=args.cohort,
            expected_mode=args.expected_mode,
        )
    except CloseInputError as exc:
        log.error("close evidence revalidation failed: %s", exc)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
