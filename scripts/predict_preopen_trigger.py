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
from notifier.format import format_preopen_beta
from notifier.telegram import send_telegram
from signals.features import assemble_training_panel
from signals.labels_preopen import PREOPEN_HEADS
from signals.precursors import build_15m_precursor


CKPT_DIR = "signals/models/ckpt/preopen_v1"
PAPER_LEDGER_PATH = "output/paper_ledger_preopen.csv"

PAPER_LEDGER_COLS = [
    "date", "coin", "btc_regime",
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
    return asof.hour == 8 and 45 <= asof.minute <= 59


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
    parser.add_argument("--allow-late-run", action="store_true",
                        help="Allow telegram/ledger outside 08:45-08:59 for intentional manual tests.")
    parser.add_argument("--out-dir", default="output")
    parser.add_argument("--paper-ledger", default=PAPER_LEDGER_PATH)
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
    alerts = krw.sort_values("composite", ascending=False).head(args.top_k).copy()
    alerts["alert_rank"] = range(1, len(alerts) + 1)
    log.info(f"alerts: {len(alerts)}")

    # BTC regime
    btc_row = panel[panel["market"] == "KRW-BTC"]
    btc_regime = str(btc_row["btc_regime"].iloc[0]) if len(btc_row) else "unknown"

    # Save log JSON
    log_path = out_dir / f"preopen_log_{date_str}.json"
    log_payload = {
        "asof": asof.isoformat(),
        "btc_regime": btc_regime,
        "universe": args.universe,
        "n_alerts": int(len(alerts)),
        "alerts": [
            {"coin": r["market"],
             "p_first15_3pct": float(r["p_first15_3pct"]),
             "p_first15_5pct": float(r["p_first15_5pct"]),
             "p_first1h_3pct": float(r["p_first1h_3pct"]),
             "composite": float(r["composite"])}
            for _, r in alerts.iterrows()
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
            p.parent.mkdir(parents=True, exist_ok=True)
            new_df.to_csv(p, index=False)
            log.info(f"created {p} with {len(new_df)} rows")
    else:
        log.warning("paper ledger append skipped outside pre-open window")

    # Build telegram message — formatter 통일 (notifier/format.py)
    msg = format_preopen_beta(
        alerts=alerts,
        btc_regime=btc_regime,
        universe_label=args.universe,
        asof=asof,
        dry_run=not send_tg,
    )
    print(msg)

    if send_tg:
        try:
            send_telegram(msg)
            log.info("telegram sent")
        except Exception as e:
            log.error(f"telegram fail: {e}")


if __name__ == "__main__":
    main()
