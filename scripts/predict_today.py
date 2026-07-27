"""Detector v1 운영 entry — KST 09:05 cron.

흐름:
  1. detector_v1 + threshold artifact 로드 (output/detector_threshold.json)
  2. 어제까지 데이터로 panel 빌드 (asof = 오늘)
  3. score → regime gate → threshold gate → cap
  4. 결과:
     - output/detector_log_YYYYMMDD.json   (전체 진단 + alerts)
     - output/predictions_YYYYMMDD.csv     (KRW score 분포)
  5. (옵션) telegram 발송 — Stage 1 dry-run 에서는 stdout 만

운영 원칙 (사용자 확정):
  - threshold = artifact 고정값 (라이브 quantile X)
  - silence-heavy: 알림 0~3건/주 예상

사용:
    python scripts/predict_today.py                # Stage 1 dry-run (default)
    python scripts/predict_today.py --asof 2026-05-03  # 시뮬레이션

실발송은 이 legacy/manual runner에서 금지한다. canonical sender:
    python scripts/recommend_send.py --slot open
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.database import list_markets, load_candles
from data.market_universe import signal_eligible_markets
from notifier.format import format_detector_beta
from ops.live_send_gate import reject_legacy_live_send
from signals.detector import DetectorV1
from signals.features import assemble_training_panel


def build_panel_for_asof(upbit_db: str, binance_db: str, asof: pd.Timestamp) -> pd.DataFrame:
    """asof 시점까지의 데이터로 panel 생성, 각 마켓의 마지막 row 반환."""
    krw = signal_eligible_markets(list_markets(upbit_db))
    candles = {m: load_candles(upbit_db, m, until=asof) for m in krw}
    if Path(binance_db).exists():
        for m in list_markets(binance_db):
            d = load_candles(binance_db, m, until=asof)
            if len(d) >= 30:
                candles[m] = d
    candles = {k: v for k, v in candles.items() if len(v) >= 30}
    btc = load_candles(upbit_db, "KRW-BTC", until=asof)

    panel = assemble_training_panel(candles, btc, normalize=True)
    if len(panel) == 0:
        return panel
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])

    # 운영 안전장치: in-progress 일봉 제거 (predict_today_distribution.py 와 동일 fix)
    # pyupbit 가 today's in-progress 도 줌. fully closed 만 사용.
    asof_ts = pd.Timestamp(asof) if not isinstance(asof, pd.Timestamp) else asof
    closed_mask = panel["timestamp"] + pd.Timedelta(hours=24) <= asof_ts
    panel = panel[closed_mask]
    if len(panel) == 0:
        return panel

    # 각 코인의 가장 최근 row = 어제 마감 일봉 (fully closed)
    latest = panel.sort_values("timestamp").groupby("market").tail(1).reset_index(drop=True)
    return latest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upbit-db", default="data/upbit_d1.db")
    parser.add_argument("--binance-db", default="data/binance_d1.db")
    parser.add_argument("--config", default="output/detector_threshold.json")
    parser.add_argument("--asof", type=str, help="기준 시점 YYYY-MM-DD (default=now)")
    parser.add_argument("--send-telegram", action="store_true",
                        help="disabled; use recommend_send.py --slot open")
    parser.add_argument("--out-dir", default="output")
    args = parser.parse_args()
    reject_legacy_live_send(
        parser,
        requested=args.send_telegram,
        slot="open",
        scoring_command="python scripts/predict_today.py",
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("predict_today")

    asof = (
        pd.Timestamp(args.asof)
        if args.asof
        else pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None)
    )
    date_str = asof.strftime("%Y%m%d")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dry_run = not args.send_telegram

    log.info(f"asof={asof.date()}, dry_run={dry_run}")

    # 1) detector 로드
    detector = DetectorV1.load(args.config)
    log.info(f"detector v1: threshold={detector.config.threshold:.4f}, "
             f"regimes={detector.config.regimes}, cap={detector.config.cap}")

    # 2) panel
    log.info("building panel...")
    panel = build_panel_for_asof(args.upbit_db, args.binance_db, asof)
    if len(panel) == 0:
        log.error("panel empty")
        sys.exit(1)
    log.info(f"panel: {len(panel)} markets")

    # 3) detect + diagnose
    diagnose = detector.diagnose(panel)
    alerts = detector.detect(panel)
    log.info(f"alerts: {len(alerts)} | diagnose: {diagnose}")

    # 4) BTC regime (latest BTC row from panel)
    btc_row = panel[panel["market"] == "KRW-BTC"]
    btc_regime = str(btc_row["btc_regime"].iloc[0]) if len(btc_row) else "unknown"

    # 5) save predictions CSV (KRW universe scores)
    krw = panel[panel["market"].str.startswith("KRW-")].copy()
    krw["score"] = detector.score(krw)
    krw["passes_regime"] = krw["btc_regime"].isin(detector.config.regimes)
    krw["passes_threshold"] = krw["score"] >= detector.config.threshold
    pred_path = out_dir / f"predictions_{date_str}.csv"
    krw[["market", "timestamp", "btc_regime", "score",
         "passes_regime", "passes_threshold"]].sort_values(
        "score", ascending=False).to_csv(pred_path, index=False)
    log.info(f"saved {pred_path}")

    # 6) save detector log JSON
    log_payload = {
        "version": detector.config.version,
        "asof": asof.isoformat(),
        "btc_regime": btc_regime,
        "universe_size": int((panel["market"].str.startswith("KRW-")).sum()),
        "threshold": detector.config.threshold,
        "regimes_active": detector.config.regimes,
        "cap": detector.config.cap,
        "diagnose": {k: (list(v.items()) if isinstance(v, dict) else v)
                     for k, v in diagnose.items()},
        "n_alerts": int(len(alerts)),
        "alerts": [
            {"coin": r["coin"], "score": float(r["score"]),
             "alert_rank": int(r["alert_rank"]),
             "btc_regime": str(r["btc_regime"])}
            for _, r in alerts.iterrows()
        ],
        "dry_run": dry_run,
    }
    log_path = out_dir / f"detector_log_{date_str}.json"
    with open(log_path, "w") as f:
        json.dump(log_payload, f, indent=2, ensure_ascii=False)
    log.info(f"saved {log_path}")

    # 7) telegram 메시지 (dry-run 이면 stdout 만)
    msg = format_detector_beta(
        alerts=alerts,
        threshold=detector.config.threshold,
        btc_regime=btc_regime,
        universe_size=log_payload["universe_size"],
        diagnose=diagnose,
        asof=asof.to_pydatetime(),
        dry_run=dry_run,
    )
    print()
    print(msg)
    print()

    if not dry_run:
        from notifier.telegram import send_telegram
        try:
            send_telegram(msg)
            log.info("telegram sent")
        except Exception as e:
            log.error(f"telegram fail: {e}")


if __name__ == "__main__":
    main()
