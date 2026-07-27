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
    python scripts/predict_today_distribution.py --asof 2025-11-06

실발송은 이 legacy/manual runner에서 금지한다. canonical sender:
    python scripts/recommend_send.py --slot open
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.database import list_markets, load_candles
from data.market_universe import signal_eligible_markets
from ledger.csv_store import atomic_write_csv, atomic_write_json, ledger_lock
from ledger.shadow import append_shadow_ledger
from notifier.format import format_distribution_beta
from ops.decision_policy import active_only, apply_distribution_policy, decision_counts
from ops.decision_policy import POLICY_ID, POLICY_SUMMARY, POLICY_VERSION
from ops.live_send_gate import reject_legacy_live_send
from ops.recommendation_quality import (
    DEFAULT_META_MODEL_DIR,
    META_POLICY_ID,
    META_POLICY_VERSION,
    apply_recommendation_quality,
    load_channel_history,
    load_trained_meta_model,
)
from signals.bucket_calibration import BucketCalibrator
from signals.distribution_engine import DistributionEngine
from signals.features import assemble_training_panel


SHADOW_LEDGER_PATH = "output/shadow_ledger_distribution.csv"

PAPER_LEDGER_COLS = [
    "date", "coin", "setup_ids", "btc_regime", "btc_context",
    "decision", "idea_id", "setup_quality", "decision_reason",
    "policy_id", "policy_version",
    "base_decision", "meta_policy_id", "meta_policy_version",
    "meta_decision", "meta_reason",
    "trained_model_id", "trained_model_version", "trained_model_status",
    "trained_model_p_win", "trained_model_threshold",
    "confidence_score", "confidence_tier",
    "evidence_key", "evidence_n_closed", "evidence_net_pnl_sum_pct",
    "evidence_avg_net_pnl_pct", "evidence_tp5_hit_rate_pct", "evidence_win_rate_pct",
    "calibrated_hit_pct", "calibrated_bucket", "calibration_source", "expected_edge_pct",
    "source_rank",
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
            "decision": r.get("decision", "ACTIVE"),
            "idea_id": r.get("idea_id", ""),
            "setup_quality": r.get("setup_quality", ""),
            "decision_reason": r.get("decision_reason", ""),
            "policy_id": r.get("policy_id", POLICY_ID),
            "policy_version": r.get("policy_version", POLICY_VERSION),
            "base_decision": r.get("base_decision", r.get("decision", "")),
            "meta_policy_id": r.get("meta_policy_id", META_POLICY_ID),
            "meta_policy_version": r.get("meta_policy_version", META_POLICY_VERSION),
            "meta_decision": r.get("meta_decision", r.get("decision", "")),
            "meta_reason": r.get("meta_reason", ""),
            "trained_model_id": r.get("trained_model_id", ""),
            "trained_model_version": r.get("trained_model_version", ""),
            "trained_model_status": r.get("trained_model_status", ""),
            "trained_model_p_win": float(r.get("trained_model_p_win", np.nan)),
            "trained_model_threshold": float(r.get("trained_model_threshold", np.nan)),
            "confidence_score": float(r.get("confidence_score", np.nan)),
            "confidence_tier": r.get("confidence_tier", ""),
            "evidence_key": r.get("evidence_key", ""),
            "evidence_n_closed": float(r.get("evidence_n_closed", np.nan)),
            "evidence_net_pnl_sum_pct": float(r.get("evidence_net_pnl_sum_pct", np.nan)),
            "evidence_avg_net_pnl_pct": float(r.get("evidence_avg_net_pnl_pct", np.nan)),
            "evidence_tp5_hit_rate_pct": float(r.get("evidence_tp5_hit_rate_pct", np.nan)),
            "evidence_win_rate_pct": float(r.get("evidence_win_rate_pct", np.nan)),
            "calibrated_hit_pct": float(r.get("calibrated_hit_pct", np.nan)),
            "calibrated_bucket": float(r.get("calibrated_bucket", np.nan)),
            "calibration_source": r.get("calibration_source", ""),
            "expected_edge_pct": float(r.get("expected_edge_pct", np.nan)),
            "source_rank": float(r.get("source_rank", rank)),
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
    with ledger_lock(p):
        if p.exists():
            existing = pd.read_csv(p)
            if (
                "date" in existing.columns
                and asof.strftime("%Y-%m-%d")
                in set(existing["date"].astype(str))
            ):
                # A paper-ledger date is one decision snapshot. Re-running the
                # script later the same day must not inflate that day's top-K.
                return 0
            # 중복 방지: 같은 (date, coin) 은 덮어쓰지 X (entered 상태 유지)
            exist_keys = set(zip(existing["date"], existing["coin"]))
            mask_new = ~new_df.apply(
                lambda r: (r["date"], r["coin"]) in exist_keys,
                axis=1,
            )
            to_append = new_df[mask_new]
            if len(to_append) > 0:
                combined = pd.concat([existing, to_append], ignore_index=True)
                atomic_write_csv(combined, p)
                return len(to_append)
            return 0

        atomic_write_csv(new_df, p)
        return len(new_df)


def is_decision_window(asof: pd.Timestamp) -> bool:
    """Paper entries are valid only near the intended KST decision time.

    This protects Stage 1 from systemd Persistent catch-up or manual late runs
    creating paper entries hours after the 09:00 decision point.
    """
    return (asof.hour == 8 and asof.minute >= 50) or (asof.hour == 9 and asof.minute <= 20)


def should_send_telegram(dry_run: bool, alerts: pd.DataFrame, send_silence: bool = False) -> bool:
    """Telegram is for actionable ACTIVE recommendations by default."""
    return (not dry_run) and (len(alerts) > 0 or send_silence)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upbit-db", default="data/upbit_d1.db")
    parser.add_argument("--binance-db", default="data/binance_d1.db")
    parser.add_argument("--ckpt-dir", default="signals/models/ckpt/dist_engine_v1")
    parser.add_argument("--asof", type=str, help="기준 시점 YYYY-MM-DD (default=now)")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--universe", default="top100", choices=["top50", "top100", "all"])
    parser.add_argument(
        "--send-telegram",
        action="store_true",
        help="disabled; use recommend_send.py --slot open",
    )
    parser.add_argument("--send-silence-telegram", action="store_true",
                        help="Send a Telegram silence/empty message when no ACTIVE recommendation exists")
    parser.add_argument("--disable-meta-filter", action="store_true",
                        help="Disable historical recommendation-quality demotion")
    parser.add_argument("--allow-late-ledger", action="store_true",
                        help="Allow paper ledger append outside the 08:50-09:20 decision window")
    parser.add_argument("--out-dir", default="output")
    parser.add_argument("--paper-ledger", default="output/paper_ledger.csv")
    parser.add_argument("--shadow-ledger", default=SHADOW_LEDGER_PATH)
    parser.add_argument("--meta-model-dir", default=DEFAULT_META_MODEL_DIR)
    args = parser.parse_args()
    reject_legacy_live_send(
        parser,
        requested=args.send_telegram,
        slot="open",
        scoring_command="python scripts/predict_today_distribution.py",
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("predict_dist")

    asof = (
        pd.Timestamp(args.asof)
        if args.asof
        else pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None)
    )
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
    candidates = engine.assemble_alerts(scored, top_k=args.top_k, universe=args.universe)
    log.info(f"candidates: {len(candidates)} | diagnose: setup_fire={diagnose.get('setup_fire_counts')}")

    # 4) BTC regime
    btc_row = panel[panel["market"] == "KRW-BTC"]
    btc_regime = str(btc_row["btc_regime"].iloc[0]) if len(btc_row) else "unknown"

    # 4a) Decision policy — model 후보를 ACTIVE/WATCH_ONLY/SILENCE 로 분리.
    # Bear hard-silence 대신 shadow ledger 에 후보를 남겨 아이디어를 검증한다.
    calibrator = BucketCalibrator.load(args.out_dir)
    decisions = apply_distribution_policy(candidates, btc_regime=btc_regime, calibrator=calibrator)
    history = load_channel_history(args.paper_ledger, args.shadow_ledger, "distribution")
    trained_meta = load_trained_meta_model(
        args.meta_model_dir,
        enabled=not args.disable_meta_filter,
    )
    decisions = apply_recommendation_quality(
        decisions,
        history,
        asof,
        "distribution",
        enabled=not args.disable_meta_filter,
        trained_model=trained_meta,
    )
    counts = decision_counts(decisions)
    alerts = active_only(decisions)
    log.info(f"decision policy: {counts} → active alerts: {len(alerts)}")
    if len(decisions) and "confidence_tier" in decisions.columns:
        meta_counts = decisions["confidence_tier"].fillna("").astype(str).value_counts().to_dict()
        log.info(f"recommendation quality: {meta_counts}")

    # 5) Save distribution log JSON
    log_payload = {
        "asof": asof.isoformat(),
        "btc_regime": btc_regime,
        "universe": args.universe,
        "universe_size": int(diagnose.get("n_universe_filtered", 0)),
        "diagnose": diagnose,
        "n_candidates": int(len(decisions)),
        "policy": {
            "policy_id": POLICY_ID,
            "policy_version": POLICY_VERSION,
            "summary": POLICY_SUMMARY,
            "meta_policy_id": META_POLICY_ID,
            "meta_policy_version": META_POLICY_VERSION,
            "meta_filter_enabled": not args.disable_meta_filter,
            "trained_meta_model": trained_meta.get("meta", {}) if trained_meta else {"available": False},
        },
        "decision_counts": counts,
        "n_alerts": int(len(alerts)),
        "alerts": [
            {
                "coin": r["market"],
                "setups": r.get("primary_setups", []),
                "decision": r.get("decision", ""),
                "base_decision": r.get("base_decision", ""),
                "idea_id": r.get("idea_id", ""),
                "setup_quality": r.get("setup_quality", ""),
                "decision_reason": r.get("decision_reason", ""),
                "meta_reason": r.get("meta_reason", ""),
                "trained_model_status": r.get("trained_model_status", ""),
                "trained_model_p_win": float(r.get("trained_model_p_win", np.nan)),
                "trained_model_threshold": float(r.get("trained_model_threshold", np.nan)),
                "confidence_score": float(r.get("confidence_score", np.nan)),
                "confidence_tier": r.get("confidence_tier", ""),
                "evidence_key": r.get("evidence_key", ""),
                "evidence_n_closed": float(r.get("evidence_n_closed", np.nan)),
                "evidence_avg_net_pnl_pct": float(r.get("evidence_avg_net_pnl_pct", np.nan)),
                "evidence_tp5_hit_rate_pct": float(r.get("evidence_tp5_hit_rate_pct", np.nan)),
                "calibrated_hit_pct": float(r.get("calibrated_hit_pct", np.nan)),
                "calibrated_bucket": float(r.get("calibrated_bucket", np.nan)),
                "calibration_source": r.get("calibration_source", ""),
                "expected_edge_pct": float(r.get("expected_edge_pct", np.nan)),
                "source_rank": int(r.get("source_rank", 0)),
                "policy_id": r.get("policy_id", POLICY_ID),
                "policy_version": r.get("policy_version", POLICY_VERSION),
                "btc_context": r.get("btc_context", "—"),
                "p_h2": float(r.get("p_h2_hit_3_4h", 0)),
                "p_h5": float(r.get("p_h5_tail_20", 0)),
                "p_h6": float(r.get("p_h6_hit_5_24h", 0)),
                "composite": float(r.get("composite", 0)),
                "entry_price_proxy": float(r.get("close", np.nan)),
            }
            for _, r in alerts.iterrows()
        ],
        "candidates": [
            {
                "coin": r["market"],
                "setups": r.get("primary_setups", []),
                "decision": r.get("decision", ""),
                "base_decision": r.get("base_decision", ""),
                "blocked_reason": r.get("blocked_reason", ""),
                "decision_reason": r.get("decision_reason", ""),
                "meta_reason": r.get("meta_reason", ""),
                "trained_model_status": r.get("trained_model_status", ""),
                "trained_model_p_win": float(r.get("trained_model_p_win", np.nan)),
                "trained_model_threshold": float(r.get("trained_model_threshold", np.nan)),
                "idea_id": r.get("idea_id", ""),
                "setup_quality": r.get("setup_quality", ""),
                "source_rank": int(r.get("source_rank", 0)),
                "confidence_score": float(r.get("confidence_score", np.nan)),
                "confidence_tier": r.get("confidence_tier", ""),
                "evidence_key": r.get("evidence_key", ""),
                "evidence_n_closed": float(r.get("evidence_n_closed", np.nan)),
                "evidence_net_pnl_sum_pct": float(r.get("evidence_net_pnl_sum_pct", np.nan)),
                "evidence_avg_net_pnl_pct": float(r.get("evidence_avg_net_pnl_pct", np.nan)),
                "evidence_tp5_hit_rate_pct": float(r.get("evidence_tp5_hit_rate_pct", np.nan)),
                "evidence_win_rate_pct": float(r.get("evidence_win_rate_pct", np.nan)),
                "calibrated_hit_pct": float(r.get("calibrated_hit_pct", np.nan)),
                "calibrated_bucket": float(r.get("calibrated_bucket", np.nan)),
                "calibration_source": r.get("calibration_source", ""),
                "expected_edge_pct": float(r.get("expected_edge_pct", np.nan)),
                "policy_id": r.get("policy_id", POLICY_ID),
                "policy_version": r.get("policy_version", POLICY_VERSION),
                "btc_context": r.get("btc_context", "—"),
                "p_h2": float(r.get("p_h2_hit_3_4h", 0)),
                "p_h5": float(r.get("p_h5_tail_20", 0)),
                "p_h6": float(r.get("p_h6_hit_5_24h", 0)),
                "composite": float(r.get("composite", 0)),
                "entry_price_proxy": float(r.get("close", np.nan)),
            }
            for _, r in decisions.iterrows()
        ],
        "dry_run": dry_run,
    }
    log_path = out_dir / f"distribution_log_{date_str}.json"
    atomic_write_json(log_payload, log_path)
    log.info(f"saved {log_path}")

    # 6) Append to paper ledger
    if args.allow_late_ledger or is_decision_window(asof):
        n_shadow = append_shadow_ledger(decisions, asof, args.shadow_ledger, "distribution")
        log.info(f"shadow_ledger: {n_shadow} new rows → {args.shadow_ledger}")
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

    send_active_message = should_send_telegram(dry_run, alerts, args.send_silence_telegram)
    if not dry_run and len(alerts) == 0 and not args.send_silence_telegram:
        log.info("telegram skipped: no ACTIVE recommendation; details kept in logs/dashboard")

    if send_active_message:
        from notifier.telegram import send_telegram
        try:
            if send_telegram(msg):
                log.info("telegram sent")
            else:
                log.error("telegram not sent")
        except Exception as e:
            log.error(f"telegram fail: {e}")


if __name__ == "__main__":
    main()
