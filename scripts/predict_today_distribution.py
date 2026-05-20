"""Distribution beta 운영 entry — KST 09:05 cron (사용자 신규 방향).

흐름:
  1. dist_engine_v1 artifact 로드
  2. 어제까지 데이터로 panel 빌드 (asof = 오늘)
  3. multi-head score + setup 매칭
  4. universe filter (top100 default) + top-K alert (composite ranking)
  5. 출력:
     - output/distribution_log_YYYYMMDD.json   (전체 진단 + alerts)
     - output/paper_ledger.csv                  (entry rows 누적, 사용자 ledger schema)
     - stdout (Stage 1 dry-run) | telegram (Stage 2)

Paper ledger schema (사용자 정의):
  date, coin, setup_ids, btc_regime,
  p_h2_3pct_4h, p_h6_5pct_24h, p_h5_20pct_tail,
  fail_eod_est, evidence_*,
  actual_high_return, actual_close_return, hit_h2, hit_h6, hit_h5,
  notes, status

운영 원칙:
  - 자동매매 X. paper ledger 는 reference only.
  - distribution beta 는 detector_v1 (rare tail) 과 병렬 운영.
  - 매수 추천 X. 분포 + 위험 표시.

사용:
    python scripts/predict_today_distribution.py                  # Stage 1 dry-run
    python scripts/predict_today_distribution.py --send-telegram  # Stage 2
    python scripts/predict_today_distribution.py --asof 2025-11-06
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.database import list_markets, load_candles
from notifier.format import format_distribution_beta
from signals.distribution_engine import DistributionEngine
from signals.features import assemble_training_panel


PAPER_LEDGER_COLS = [
    "date", "coin", "setup_ids", "btc_regime", "btc_context",
    "p_h2_3pct_4h", "p_h6_5pct_24h", "p_h5_20pct_tail",
    "p_h1_stable_4h", "p_h7_mid_7_24h",
    # evidence
    "log_return_1d_pct", "atr_pct_14_pct", "return_5d_rank", "return_7d_rank",
    "vol_5d_rank", "roc_3d_rank",
    # ranking
    "composite_score", "alert_rank",
    # realized OHLC + return (closed by close_paper_ledger.py later)
    "next_open", "next_high", "next_low", "next_close",
    "next_max_return_pct", "next_min_return_pct", "next_close_return_pct",
    "hit_h2", "hit_h5", "hit_h6",
    "status",  # entered / closed
    "notes",
]


def build_panel_for_asof(upbit_db: str, binance_db: str, asof: pd.Timestamp) -> pd.DataFrame:
    krw = list_markets(upbit_db)
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

    # === 운영 안전장치: in-progress 일봉 제거 ===
    # pyupbit 는 today's in-progress 일봉도 줌 (timestamp = today 09:00, 일부만 채워짐).
    # 이걸 그대로 쓰면 panel.latest = today (partial) → label_date = tomorrow (X+1) 예측
    # → paper_ledger.date = asof.date() (today) 와 mismatch.
    # → close-out 시 wrong day's 4h bars 봄.
    # Fix: candle ENDS (timestamp + 24h) ≤ asof 인 행만 사용 (= fully closed only).
    asof_ts = pd.Timestamp(asof) if not isinstance(asof, pd.Timestamp) else asof
    closed_mask = panel["timestamp"] + pd.Timedelta(hours=24) <= asof_ts
    panel = panel[closed_mask]
    if len(panel) == 0:
        return panel

    latest = panel.sort_values("timestamp").groupby("market").tail(1).reset_index(drop=True)
    latest["date_only"] = latest["timestamp"].dt.date
    latest["quote_volume_d1"] = latest.get("quote_volume", np.nan)
    # daily liquidity rank
    latest["liq_rank_daily"] = np.nan
    krw_mask = latest["market"].str.startswith("KRW-")
    latest.loc[krw_mask, "liq_rank_daily"] = latest.loc[krw_mask, "quote_volume_d1"].rank(
        method="dense", ascending=False, na_option="bottom"
    )
    return latest


def append_to_paper_ledger(alerts: pd.DataFrame, asof: pd.Timestamp,
                            ledger_path: str) -> int:
    """alert rows → paper_ledger.csv 추가. 중복 방지 (date+coin)."""
    if len(alerts) == 0:
        return 0
    rows = []
    for rank, (_, r) in enumerate(alerts.iterrows(), 1):
        rows.append({
            "date": asof.strftime("%Y-%m-%d"),
            "coin": r["market"],
            "setup_ids": "+".join(r.get("primary_setups", [])),
            "btc_regime": r.get("btc_regime", "unknown"),
            "btc_context": r.get("btc_context", "—"),
            "p_h2_3pct_4h": float(r.get("p_h2_hit_3_4h", 0)),
            "p_h6_5pct_24h": float(r.get("p_h6_hit_5_24h", 0)),
            "p_h5_20pct_tail": float(r.get("p_h5_tail_20", 0)),
            "p_h1_stable_4h": float(r.get("p_h1_stable_3_4h", 0)),
            "p_h7_mid_7_24h": float(r.get("p_h7_mid_7_24h", 0)),
            "log_return_1d_pct": float(r.get("log_return_1d", 0)) * 100,
            "atr_pct_14_pct": float(r.get("atr_pct_14", 0)) * 100,
            "return_5d_rank": float(r.get("return_5d", 0)),
            "return_7d_rank": float(r.get("return_7d", 0)),
            "vol_5d_rank": float(r.get("vol_5d", 0)),
            "roc_3d_rank": float(r.get("roc_3d", 0)),
            "composite_score": float(r.get("composite", 0)),
            "alert_rank": rank,
            "next_open": np.nan,
            "next_high": np.nan,
            "next_low": np.nan,
            "next_close": np.nan,
            "next_max_return_pct": np.nan,
            "next_min_return_pct": np.nan,
            "next_close_return_pct": np.nan,
            "hit_h2": np.nan,
            "hit_h5": np.nan,
            "hit_h6": np.nan,
            "status": "entered",
            "notes": "",
        })

    new_df = pd.DataFrame(rows)[PAPER_LEDGER_COLS]
    p = Path(ledger_path)

    if p.exists():
        existing = pd.read_csv(p)
        if "date" in existing.columns and asof.strftime("%Y-%m-%d") in set(existing["date"].astype(str)):
            # A paper-ledger date is one decision snapshot. Re-running the
            # script later the same day must not inflate that day's top-K.
            return 0
        # 중복 방지: 같은 (date, coin) 은 덮어쓰지 X (entered 상태 유지)
        exist_keys = set(zip(existing["date"], existing["coin"]))
        mask_new = ~new_df.apply(lambda r: (r["date"], r["coin"]) in exist_keys, axis=1)
        to_append = new_df[mask_new]
        if len(to_append) > 0:
            combined = pd.concat([existing, to_append], ignore_index=True)
            combined.to_csv(p, index=False)
            return len(to_append)
        return 0
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        new_df.to_csv(p, index=False)
        return len(new_df)


def is_decision_window(asof: pd.Timestamp) -> bool:
    """Paper entries are valid only near the intended KST decision time.

    This protects Stage 1 from systemd Persistent catch-up or manual late runs
    creating paper entries hours after the 09:00 decision point.
    """
    return (asof.hour == 8 and asof.minute >= 50) or (asof.hour == 9 and asof.minute <= 20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upbit-db", default="data/upbit_d1.db")
    parser.add_argument("--binance-db", default="data/binance_d1.db")
    parser.add_argument("--ckpt-dir", default="signals/models/ckpt/dist_engine_v1")
    parser.add_argument("--asof", type=str, help="기준 시점 YYYY-MM-DD (default=now)")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--universe", default="top100", choices=["top50", "top100", "all"])
    parser.add_argument("--send-telegram", action="store_true")
    parser.add_argument("--allow-late-ledger", action="store_true",
                        help="Allow paper ledger append outside the 08:50-09:20 decision window")
    parser.add_argument("--out-dir", default="output")
    parser.add_argument("--paper-ledger", default="output/paper_ledger.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("predict_dist")

    asof = pd.Timestamp(args.asof) if args.asof else pd.Timestamp.now()
    date_str = asof.strftime("%Y%m%d")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dry_run = not args.send_telegram

    log.info(f"asof={asof.date()}, dry_run={dry_run}, universe={args.universe}")

    # 1) Engine
    engine = DistributionEngine.load(args.ckpt_dir)

    # 2) Panel
    log.info("building panel...")
    panel = build_panel_for_asof(args.upbit_db, args.binance_db, asof)
    if len(panel) == 0:
        log.error("panel empty")
        sys.exit(1)
    log.info(f"panel: {len(panel)} markets")

    # 3) Score + setup + assemble
    scored = engine.score_panel(panel)
    diagnose = engine.diagnose(scored, universe=args.universe)
    alerts = engine.assemble_alerts(scored, top_k=args.top_k, universe=args.universe)
    log.info(f"alerts: {len(alerts)} | diagnose: setup_fire={diagnose.get('setup_fire_counts')}")

    # 4) BTC regime
    btc_row = panel[panel["market"] == "KRW-BTC"]
    btc_regime = str(btc_row["btc_regime"].iloc[0]) if len(btc_row) else "unknown"

    # 4a) Bear regime silence — detector_v1 패턴 일관성 (2026-05-20 추가).
    # 5/8~5/20 라이브 결과 89% alert 이 bear regime 에서 발사되어 학습 분포 밖.
    # cum PnL -52% 확인 후 학습 가정 (BTC bull conditional) 과 운영 align.
    if btc_regime.startswith("bear"):
        log.warning(f"bear regime ({btc_regime}) — alerts silence (학습 분포 밖)")
        alerts = alerts.iloc[0:0]  # empty

    # 5) Save distribution log JSON
    log_payload = {
        "asof": asof.isoformat(),
        "btc_regime": btc_regime,
        "universe": args.universe,
        "universe_size": int(diagnose.get("n_universe_filtered", 0)),
        "diagnose": diagnose,
        "n_alerts": int(len(alerts)),
        "alerts": [
            {
                "coin": r["market"],
                "setups": r.get("primary_setups", []),
                "btc_context": r.get("btc_context", "—"),
                "p_h2": float(r.get("p_h2_hit_3_4h", 0)),
                "p_h5": float(r.get("p_h5_tail_20", 0)),
                "p_h6": float(r.get("p_h6_hit_5_24h", 0)),
                "composite": float(r.get("composite", 0)),
            }
            for _, r in alerts.iterrows()
        ],
        "dry_run": dry_run,
    }
    log_path = out_dir / f"distribution_log_{date_str}.json"
    with open(log_path, "w") as f:
        json.dump(log_payload, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"saved {log_path}")

    # 6) Append to paper ledger
    if args.allow_late_ledger or is_decision_window(asof):
        n_new = append_to_paper_ledger(alerts, asof, args.paper_ledger)
        log.info(f"paper_ledger: {n_new} new rows → {args.paper_ledger}")
    else:
        log.warning(
            "paper_ledger: skipped append outside decision window "
            f"(asof={asof.strftime('%H:%M')}; use --allow-late-ledger only for intentional backfill)"
        )

    # 7) Format + (optional) telegram
    # 진단/Setup library 블록은 텔레그램에서 빼고 stdout/log 로만.
    # log_payload (distribution_log JSON) 에 동일 정보 보존됨.
    msg = format_distribution_beta(
        alerts=alerts,
        btc_regime=btc_regime,
        universe_label=args.universe,
        universe_size=diagnose.get("n_universe_filtered", 0),
        asof=asof,
        dry_run=dry_run,
    )
    sc = diagnose.get("setup_fire_counts", {})
    log.info(
        f"diag — universe {diagnose.get('n_universe_filtered', 0)} | "
        f"setup fire S01:{sc.get('S01', 0)} S02:{sc.get('S02', 0)} "
        f"S03:{sc.get('S03', 0)} S04:{sc.get('S04', 0)} | "
        f"any primary {diagnose.get('n_with_any_primary_setup', 0)}"
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
