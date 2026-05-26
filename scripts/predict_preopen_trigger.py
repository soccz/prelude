"""Pre-open Trigger 운영 entry — KST 08:55 cron.

타겟: 09:00 직후 첫 15m/30m/1h 펌프 후보.

흐름:
  1. preopen_v1 artifacts 로드 (6 heads)
  2. 15m precursor (08:30 snapshot) 계산 + daily panel 은 universe/display 용으로만 사용
  3. 6 heads score
  4. composite ranking → top-K alert
  5. telegram (Stage 2 mode) + paper_ledger_preopen.csv

운영 원칙:
  - 08:55 = 09:00 boundary 5분 전.
  - model input 은 08:30 까지의 15m precursor 만 사용한다.
  - daily panel 은 top100 universe/display 용도이며 model feature 에 들어가지 않는다.
  - 15m precursor 는 08:15-08:30 closed bar (full closed) 사용.
  - paper_ledger_preopen.csv 분리 (distribution beta 와 별도).
  - Same-day close-out 가능 (09:30 에 09:00 첫 15m bar 사용).

사용:
    python scripts/predict_preopen_trigger.py                  # default Stage 2 (telegram ON)
    python scripts/predict_preopen_trigger.py --no-telegram    # dry-run
    python scripts/predict_preopen_trigger.py --asof 2026-05-06 08:55
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
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.database import list_markets, load_candles
from ledger.shadow import append_shadow_ledger
from notifier.format import format_preopen_beta
from notifier.telegram import send_telegram
from ops.decision_policy import PREOPEN_DEMOTED, active_only, apply_preopen_policy, decision_counts
from ops.decision_policy import POLICY_ID, POLICY_SUMMARY, POLICY_VERSION
from ops.recommendation_quality import (
    DEFAULT_META_MODEL_DIR,
    META_POLICY_ID,
    META_POLICY_VERSION,
    apply_recommendation_quality,
    load_channel_history,
    load_trained_meta_model,
)
from signals.features import assemble_training_panel
from signals.labels_preopen import PREOPEN_HEADS
from signals.precursors import build_15m_precursor


CKPT_DIR = "signals/models/ckpt/preopen_v1"
PAPER_LEDGER_PATH = "output/paper_ledger_preopen.csv"
SHADOW_LEDGER_PATH = "output/shadow_ledger_preopen.csv"

PAPER_LEDGER_COLS = [
    "date", "coin", "btc_regime",
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
    "entry_price_proxy",
    "p_first15_3pct", "p_first15_5pct",
    "p_first30_3pct", "p_first30_5pct",
    "p_first1h_3pct", "p_first1h_5pct",
    "log_return_1d_pct", "atr_pct_14_pct",
    "vol5_rank", "return5_rank",
    "composite_score", "alert_rank",
    # realized OHLC + returns
    "first_open",
    "first_15m_high", "first_30m_high", "first_1h_high",
    "first_15m_low", "first_30m_low", "first_1h_low",
    "first_1h_close",
    "first_1h_max_return_pct", "first_1h_min_return_pct", "first_1h_close_return_pct",
    "hit_first15_3pct", "hit_first15_5pct",
    "hit_first30_3pct", "hit_first30_5pct",
    "hit_first1h_3pct", "hit_first1h_5pct",
    "status", "notes",
]


def load_models(ckpt_dir: str = CKPT_DIR):
    p = Path(ckpt_dir)
    with open(p / "meta.json") as f:
        meta = json.load(f)
    models = {}
    for head_id in PREOPEN_HEADS:
        mp = p / f"{head_id}.json"
        if mp.exists():
            m = xgb.XGBClassifier()
            m.load_model(str(mp))
            models[head_id] = m
    return models, meta


def is_preopen_window(asof: pd.Timestamp) -> bool:
    """Valid paper/telegram window for the 08:55 KST decision.

    Protects against manual late runs and systemd catch-up creating entries
    hours after the pre-open decision point.
    """
    # window: 08:45 ~ 09:10 — collector 점진적 slowdown 마진 (2026-05 관측).
    # 08:50 timer fire → collector 5~7분 → predict 가 09:00~09:05 도달 가능.
    # paper ledger 무결성 위해 09:10 까지만 허용 (manual run 차단).
    if asof.hour == 8 and asof.minute >= 45:
        return True
    if asof.hour == 9 and asof.minute <= 10:
        return True
    return False


def should_send_telegram(send_tg: bool, alerts: pd.DataFrame, send_silence: bool = False) -> bool:
    """Telegram is for actionable ACTIVE pre-open recommendations by default."""
    return send_tg and (len(alerts) > 0 or send_silence)


def build_panel_for_asof(upbit_d1: str, upbit_15m: str, binance_d1: str,
                          asof: pd.Timestamp) -> tuple:
    """Build daily panel + 15m precursor at 08:55 timing.

    Daily: 어제 candle 은 almost closed (5 min from close). 사용 (partial 5min OK).
    15m: 08:30 snapshot (07:00-08:30 bars, fully closed).
    """
    # Daily — load all up to asof (including in-progress yesterday)
    krw = list_markets(upbit_d1)
    candles_d1 = {m: load_candles(upbit_d1, m, until=asof) for m in krw}
    if Path(binance_d1).exists():
        for m in list_markets(binance_d1):
            d = load_candles(binance_d1, m, until=asof)
            if d is not None and len(d) >= 30:
                candles_d1[m] = d
    candles_d1 = {k: v for k, v in candles_d1.items() if v is not None and len(v) >= 30}
    btc = load_candles(upbit_d1, "KRW-BTC", until=asof)

    panel = assemble_training_panel(candles_d1, btc, normalize=True)
    if len(panel) == 0:
        return panel, pd.DataFrame()
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])

    # For pre-open at 08:55, latest daily candle = "almost closed" (5 min from 09:00 boundary).
    # Allow candles that close within 30 min from asof — treat as essentially closed.
    asof_ts = pd.Timestamp(asof) if not isinstance(asof, pd.Timestamp) else asof
    closed_mask = panel["timestamp"] + pd.Timedelta(hours=24) <= asof_ts + pd.Timedelta(minutes=30)
    panel = panel[closed_mask]
    if len(panel) == 0:
        return panel, pd.DataFrame()

    latest = panel.sort_values("timestamp").groupby("market").tail(1).reset_index(drop=True)
    latest["date_only"] = latest["timestamp"].dt.date
    latest["quote_volume_d1"] = latest.get("quote_volume", np.nan)
    krw_mask = latest["market"].str.startswith("KRW-")
    latest["liq_rank_daily"] = np.nan
    latest.loc[krw_mask, "liq_rank_daily"] = latest.loc[krw_mask, "quote_volume_d1"].rank(
        method="dense", ascending=False, na_option="bottom"
    )

    # 15m precursor (08:30 snapshot) — load only recent rows.
    # Full-history precursor recomputation takes several minutes and makes an
    # 08:55 alert arrive too late. Three days provide enough prior 15m bars for
    # the 24h rolling features used by build_15m_precursor.
    krw_15m = [m for m in list_markets(upbit_15m) if m.startswith("KRW-")]
    since_15m = asof_ts - pd.Timedelta(days=3)
    candles_15m = {m: load_candles(upbit_15m, m, since=since_15m, until=asof) for m in krw_15m}
    candles_15m = {k: v for k, v in candles_15m.items() if v is not None and len(v) > 100}
    btc_15m = candles_15m.get("KRW-BTC", pd.DataFrame())
    precursor_df = build_15m_precursor(candles_15m, btc_15m)

    # Filter precursor to most recent date_only per market
    if len(precursor_df) > 0:
        precursor_df = (precursor_df.sort_values("date_only")
                                     .groupby("market").tail(1).reset_index(drop=True))

    return latest, precursor_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upbit-d1", default="data/upbit_d1.db")
    parser.add_argument("--upbit-15m", default="data/upbit_15m.db")
    parser.add_argument("--binance-d1", default="data/binance_d1.db")
    parser.add_argument("--asof", type=str, help="기준 시점 'YYYY-MM-DD HH:MM' (default=now)")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--universe", default="top100", choices=["top50", "top100", "all"])
    parser.add_argument("--no-telegram", action="store_true",
                        help="default: telegram ON. specify to disable.")
    parser.add_argument("--send-silence-telegram", action="store_true",
                        help="Send a Telegram silence/empty message when no ACTIVE recommendation exists")
    parser.add_argument("--disable-meta-filter", action="store_true",
                        help="Disable historical recommendation-quality demotion")
    parser.add_argument("--allow-late-run", action="store_true",
                        help="Allow telegram/ledger outside 08:45-08:59 for intentional manual tests.")
    parser.add_argument("--out-dir", default="output")
    parser.add_argument("--paper-ledger", default=PAPER_LEDGER_PATH)
    parser.add_argument("--shadow-ledger", default=SHADOW_LEDGER_PATH)
    parser.add_argument("--meta-model-dir", default=DEFAULT_META_MODEL_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("preopen")

    asof = pd.Timestamp(args.asof) if args.asof else pd.Timestamp.now()
    date_str = asof.strftime("%Y%m%d")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    send_tg = not args.no_telegram
    in_window = is_preopen_window(asof)
    if not in_window and not args.allow_late_run:
        log.warning(
            "outside pre-open decision window (%s); telegram+ledger append disabled "
            "(use --allow-late-run only for intentional tests)",
            asof.strftime("%H:%M"),
        )
        send_tg = False

    log.info(f"asof={asof}, telegram={send_tg}, universe={args.universe}")

    # Load models
    models, meta = load_models()
    feature_cols = meta["feature_cols"]
    log.info(f"loaded {len(models)} heads, {len(feature_cols)} features")

    # Build panel + precursor
    log.info("building panel + 15m precursor (08:30 snapshot)...")
    panel, precursor_df = build_panel_for_asof(args.upbit_d1, args.upbit_15m, args.binance_d1, asof)
    if len(panel) == 0 or len(precursor_df) == 0:
        log.error("empty panel or precursor")
        sys.exit(1)

    # Join
    full = panel.merge(precursor_df, on=["market", "date_only"], how="inner")
    log.info(f"joined: {full.shape}")

    # Score all heads
    krw = full[full["market"].str.startswith("KRW-")].copy()
    if len(krw) == 0:
        log.error("no KRW markets after merge")
        sys.exit(1)

    # Entry price proxy: latest daily candle close (= 08:55 spot, daily candle 5min from close)
    if "close" not in krw.columns:
        krw["close"] = np.nan

    # Universe filter
    if args.universe.startswith("top"):
        n = int(args.universe.replace("top", ""))
        krw = krw[krw["liq_rank_daily"] <= n]
    log.info(f"universe filtered: {len(krw)}")

    if len(krw) == 0:
        log.warning("empty universe after filter")
        sys.exit(0)

    X = krw[feature_cols].astype(float).values
    for head_id, m in models.items():
        krw[f"p_{head_id}"] = m.predict_proba(X)[:, 1]

    # Composite: weighted by lift × hit value
    # first15_3pct (hit value 3, lift 8) → weight 24
    # first15_5pct (hit value 5, lift 9) → weight 45
    # first30_3pct (3, 7) → 21
    # first30_5pct (5, 8) → 40
    # first1h_3pct (3, 6) → 18
    # first1h_5pct (5, 7) → 35
    # → simple sum of probabilities (each contributes proportional value)
    krw["composite"] = (
        krw["p_first15_3pct"] * 0.5 + krw["p_first15_5pct"] * 1.0 +
        krw["p_first30_3pct"] * 0.4 + krw["p_first30_5pct"] * 0.8 +
        krw["p_first1h_3pct"] * 0.3 + krw["p_first1h_5pct"] * 0.6
    )
    candidates = krw.sort_values("composite", ascending=False).head(args.top_k).copy()
    candidates["alert_rank"] = range(1, len(candidates) + 1)
    log.info(f"candidates: {len(candidates)}")

    # BTC regime
    btc_row = panel[panel["market"] == "KRW-BTC"]
    btc_regime = str(btc_row["btc_regime"].iloc[0]) if len(btc_row) else "unknown"

    # Decision policy — preopen 은 live edge 가 약해 기본 WATCH_ONLY 중심.
    decisions = apply_preopen_policy(candidates, btc_regime=btc_regime)
    history = load_channel_history(args.paper_ledger, args.shadow_ledger, "preopen")
    trained_meta = load_trained_meta_model(args.meta_model_dir)
    decisions = apply_recommendation_quality(
        decisions,
        history,
        asof,
        "preopen",
        enabled=not args.disable_meta_filter,
        trained_model=trained_meta,
    )
    counts = decision_counts(decisions)
    alerts = active_only(decisions)
    log.info(f"decision policy: {counts} → active alerts: {len(alerts)}")
    if len(decisions) and "confidence_tier" in decisions.columns:
        meta_counts = decisions["confidence_tier"].fillna("").astype(str).value_counts().to_dict()
        log.info(f"recommendation quality: {meta_counts}")

    # Save log JSON
    log_path = out_dir / f"preopen_log_{date_str}.json"
    log_payload = {
        "asof": asof.isoformat(),
        "btc_regime": btc_regime,
        "universe": args.universe,
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
            {"coin": r["market"],
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
             "p_first15_3pct": float(r["p_first15_3pct"]),
             "p_first15_5pct": float(r["p_first15_5pct"]),
             "p_first1h_3pct": float(r["p_first1h_3pct"]),
             "p_first1h_5pct": float(r.get("p_first1h_5pct", 0)),
             "composite": float(r["composite"]),
             "entry_price_proxy": float(r.get("close", np.nan))}
            for _, r in alerts.iterrows()
        ],
        "candidates": [
            {"coin": r["market"],
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
             "p_first15_3pct": float(r.get("p_first15_3pct", 0)),
             "p_first15_5pct": float(r.get("p_first15_5pct", 0)),
             "p_first1h_3pct": float(r.get("p_first1h_3pct", 0)),
             "p_first1h_5pct": float(r.get("p_first1h_5pct", 0)),
             "composite": float(r["composite"]),
             "entry_price_proxy": float(r.get("close", np.nan))}
            for _, r in decisions.iterrows()
        ],
    }
    with open(log_path, "w") as f:
        json.dump(log_payload, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"saved {log_path}")

    # Append to paper_ledger_preopen.csv (date = today, predicted day = today)
    rows = []
    for _, r in alerts.iterrows():
        rows.append({
            "date": asof.strftime("%Y-%m-%d"),
            "coin": r["market"],
            "btc_regime": r.get("btc_regime", "unknown"),
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
            "source_rank": float(r.get("source_rank", r["alert_rank"])),
            "entry_price_proxy": float(r.get("close", np.nan)),
            "p_first15_3pct": float(r["p_first15_3pct"]),
            "p_first15_5pct": float(r["p_first15_5pct"]),
            "p_first30_3pct": float(r["p_first30_3pct"]),
            "p_first30_5pct": float(r["p_first30_5pct"]),
            "p_first1h_3pct": float(r["p_first1h_3pct"]),
            "p_first1h_5pct": float(r["p_first1h_5pct"]),
            "log_return_1d_pct": float(r.get("log_return_1d", 0)) * 100,
            "atr_pct_14_pct": float(r.get("atr_pct_14", 0)) * 100,
            "vol5_rank": float(r.get("vol_5d", 0)),
            "return5_rank": float(r.get("return_5d", 0)),
            "composite_score": float(r["composite"]),
            "alert_rank": int(r["alert_rank"]),
            "first_open": np.nan,
            "first_15m_high": np.nan, "first_30m_high": np.nan, "first_1h_high": np.nan,
            "first_15m_low": np.nan, "first_30m_low": np.nan, "first_1h_low": np.nan,
            "first_1h_close": np.nan,
            "first_1h_max_return_pct": np.nan,
            "first_1h_min_return_pct": np.nan,
            "first_1h_close_return_pct": np.nan,
            "hit_first15_3pct": np.nan, "hit_first15_5pct": np.nan,
            "hit_first30_3pct": np.nan, "hit_first30_5pct": np.nan,
            "hit_first1h_3pct": np.nan, "hit_first1h_5pct": np.nan,
            "status": "entered", "notes": "",
        })
    if in_window or args.allow_late_run:
        n_shadow = append_shadow_ledger(decisions, asof, args.shadow_ledger, "preopen")
        log.info(f"shadow_ledger: {n_shadow} new rows → {args.shadow_ledger}")
        if rows:
            new_df = pd.DataFrame(rows)[PAPER_LEDGER_COLS]
            p = Path(args.paper_ledger)
            if p.exists():
                existing = pd.read_csv(p)
                keys = set(zip(existing["date"], existing["coin"]))
                mask = ~new_df.apply(lambda r: (r["date"], r["coin"]) in keys, axis=1)
                to_append = new_df[mask]
                if len(to_append) > 0:
                    pd.concat([existing, to_append], ignore_index=True).to_csv(p, index=False)
                    log.info(f"appended {len(to_append)} rows to {p}")
                else:
                    log.info(f"paper_ledger: 0 new rows → {p}")
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                new_df.to_csv(p, index=False)
                log.info(f"created {p} with {len(new_df)} rows")
        else:
            log.info(f"paper_ledger: 0 active rows → {args.paper_ledger}")
    else:
        log.warning("paper/shadow ledger append skipped outside pre-open window")

    # Build telegram message — formatter 통일 (notifier/format.py)
    msg = format_preopen_beta(
        alerts=alerts,
        btc_regime=btc_regime,
        universe_label=args.universe,
        asof=asof,
        dry_run=not send_tg,
        demoted=PREOPEN_DEMOTED,
    )
    print(msg)

    # DEMOTED 면 silence flag 없어도 매일 한 통 보냄 (사용자 가시성 유지).
    send_active_message = should_send_telegram(send_tg, alerts, args.send_silence_telegram or PREOPEN_DEMOTED)
    if send_tg and len(alerts) == 0 and not args.send_silence_telegram and not PREOPEN_DEMOTED:
        log.info("telegram skipped: no ACTIVE recommendation; details kept in logs/dashboard")

    if send_active_message:
        try:
            if send_telegram(msg):
                log.info("telegram sent")
            else:
                log.error("telegram not sent")
        except Exception as e:
            log.error(f"telegram fail: {e}")


if __name__ == "__main__":
    main()
