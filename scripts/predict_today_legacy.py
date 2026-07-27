"""매일 KST 09:05 메인 entry — 추론 + 가상 ledger + 텔레그램 알림.

흐름 (OPS §0):
  1. preflight 게이트
  2. 데이터 로드 + panel 빌드
  3. predict (분포 + 기대 max + CI)
  4. 가상 사이징 (top-K)
  5. 어제 추천의 청산 시뮬 → 가상 ledger 갱신
  6. 텔레그램 알림 (포맷: 분포 + 어제 결과 + 누적 + 정확도)
  7. predictions_YYYYMMDD.csv 저장

사용:
    python scripts/predict_today_legacy.py --dry-run
    python scripts/predict_today_legacy.py --dry-run --asof 2026-05-03

실발송은 이 legacy runner에서 금지한다. canonical sender:
    python scripts/recommend_send.py --slot open
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ledger.config import K_POSITIONS, LEDGER_CSV
from ledger.metrics import compute_summary
from ledger.sizing import size_positions
from ledger.tracker import simulate_batch
from notifier.format import (
    format_daily_alert,
    format_preflight_fail,
    format_silence_alert,
)
from notifier.telegram import send_telegram
from ops.live_send_gate import reject_legacy_live_send
from ops.preflight import run_preflight
from signals.models.xgb_phase1 import XGBPhase1
from signals.predict import predict_today


def _latest_ckpt_path() -> tuple[Path, Path] | None:
    ckpt_dir = Path("signals/models/ckpt")
    cfg_dir = Path("signals/models/configs")
    if not ckpt_dir.exists():
        return None
    candidates = sorted(ckpt_dir.glob("phase1_*.json"), reverse=True)
    if not candidates:
        return None
    ckpt = candidates[0]
    cfg = cfg_dir / ckpt.name
    return ckpt, cfg


def _yesterday_recs_path(asof: datetime) -> Path:
    yesterday = pd.Timestamp(asof).normalize() - pd.Timedelta(days=1)
    return Path(f"output/predictions_{yesterday.strftime('%Y%m%d')}.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None, help="모델 ckpt (default: 최신)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--upbit-db", default="data/upbit_d1.db")
    parser.add_argument("--upbit-4h-db", default="data/upbit_4h.db")
    parser.add_argument("--binance-db", default="data/binance_d1.db")
    parser.add_argument("--btc-market", default="KRW-BTC")
    parser.add_argument("--asof", type=str, help="기준 시점 (YYYY-MM-DD), default=now")
    parser.add_argument("--top-k", type=int, default=K_POSITIONS)
    parser.add_argument("--dry-run", action="store_true", help="텔레그램 발송 X")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()
    reject_legacy_live_send(
        parser,
        requested=not args.dry_run,
        slot="open",
        scoring_command="python scripts/predict_today_legacy.py --dry-run",
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("predict_today")

    asof = (
        pd.Timestamp(args.asof)
        if args.asof
        else pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None)
    )

    # ============================================================
    # 1. 모델 로드
    # ============================================================
    if args.ckpt:
        ckpt = Path(args.ckpt)
        cfg = Path(args.config) if args.config else None
    else:
        latest = _latest_ckpt_path()
        if latest is None:
            print("ERROR: 모델 ckpt 없음")
            print("  먼저 학습: python scripts/train_phase1_full.py")
            sys.exit(1)
        ckpt, cfg = latest
        logger.info(f"  using latest ckpt: {ckpt.name}")

    model = XGBPhase1.load(ckpt, cfg)

    # ============================================================
    # 2. Preflight
    # ============================================================
    if not args.skip_preflight:
        # 간단 freshness 만 (panel 은 아래서 빌드)
        pre = run_preflight(
            db_path=args.upbit_db,
            ckpt_path=ckpt,
            cal_path=f"output/calibration_{ckpt.stem.split('_')[-1]}.json",
            asof=asof,
        )
        if not pre.passed:
            logger.warning(f"preflight FAIL: {pre.failures}")
            msg = format_preflight_fail(pre.failures, pre.details, asof=asof)
            send_telegram(msg, dry_run=args.dry_run)
            sys.exit(2)

    # ============================================================
    # 3. 추론
    # ============================================================
    logger.info("predicting...")
    preds = predict_today(
        model,
        upbit_db=args.upbit_db,
        binance_db=args.binance_db,
        btc_market=args.btc_market,
        asof=asof,
    )
    if len(preds) == 0:
        logger.warning("no predictions")
        msg = format_silence_alert("시그널 없음", asof=asof)
        send_telegram(msg, dry_run=args.dry_run)
        sys.exit(0)

    btc_regime = preds.iloc[0]["btc_regime"] if len(preds) > 0 else "unknown"
    logger.info(f"  predictions: {len(preds)} coins, BTC regime: {btc_regime}")

    # ============================================================
    # 4. 가상 사이징 (top-K + regime gate)
    # ============================================================
    sized = size_positions(preds, btc_regime=btc_regime, k=args.top_k)
    logger.info(f"  sized: {len(sized)} positions")

    # ============================================================
    # 5. 어제 결과 (가상 ledger)
    # ============================================================
    ledger_y = pd.DataFrame()
    ledger_summary = None
    yesterday_path = _yesterday_recs_path(asof)
    if yesterday_path.exists():
        try:
            yesterday_recs = pd.read_csv(yesterday_path)
            yesterday_recs["entry_date"] = pd.to_datetime(asof.normalize().replace(hour=9))
            # 어제 추천 → 오늘 진입 시뮬 (사실은 어제→오늘 진입이지만 단순화)
            ledger_y = simulate_batch(
                yesterday_recs.head(args.top_k),
                db_4h_path=args.upbit_4h_db,
                db_d1_path=args.upbit_db,
            )
            # 누적 ledger.csv 갱신
            if len(ledger_y) > 0 and not args.dry_run:
                ledger_path = Path(LEDGER_CSV)
                ledger_path.parent.mkdir(parents=True, exist_ok=True)
                if ledger_path.exists():
                    existing = pd.read_csv(ledger_path)
                    combined = pd.concat([existing, ledger_y], ignore_index=True)
                    combined = combined.drop_duplicates(subset=["coin", "entry_date"])
                else:
                    combined = ledger_y
                combined.to_csv(ledger_path, index=False)
                ledger_summary = compute_summary(combined)
            elif len(ledger_y) > 0:
                ledger_summary = compute_summary(ledger_y)
        except Exception as e:
            logger.warning(f"yesterday ledger read fail: {e}")

    # ============================================================
    # 6. 정확도 (옵션)
    # ============================================================
    cal_text = None
    # (calibration_report 는 학습 시 저장됨 — 여기선 표시만)

    # ============================================================
    # 7. 텔레그램 알림
    # ============================================================
    universe_size = len(preds)
    msg = format_daily_alert(
        predictions=sized if len(sized) > 0 else preds,
        ledger_yesterday=ledger_y,
        ledger_summary=ledger_summary,
        calibration_text=cal_text,
        asof=asof,
        btc_regime=btc_regime,
        universe_size=universe_size,
        top_k=args.top_k,
    )
    print(msg)
    print()

    sent = send_telegram(msg, dry_run=args.dry_run)
    if sent and not args.dry_run:
        logger.info("telegram sent OK")

    # ============================================================
    # 8. predictions_YYYYMMDD.csv 저장
    # ============================================================
    if not args.dry_run:
        date_str = asof.strftime("%Y%m%d")
        out_path = Path(f"output/predictions_{date_str}.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        preds.to_csv(out_path, index=False)
        logger.info(f"saved: {out_path}")


if __name__ == "__main__":
    main()
