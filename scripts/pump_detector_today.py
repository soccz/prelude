"""Record-only daily PUMP hunter ledger append.

This script writes candidates from signals.pump_detector_v1 to a dedicated
shadow ledger. It sends no Telegram messages and places no orders.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.recommend_today import RECOMMEND_LEDGER_COLS  # noqa: E402
from signals.pump_detector_v1 import (  # noqa: E402
    EST_PUMP20_PROB,
    MAX_CANDIDATES,
    SL_PCT,
    TP_PCT,
    UNIVERSE_TOP_N,
    score_pump_candidates,
)

PUMP_HUNTER_LEDGER = "output/shadow_ledger_pump_hunter.csv"

EXTRA_LEDGER_COLS = [
    "model_id",
    "rule_version",
    "rule_id",
    "feature_date",
    "liq_rank_daily",
    "roc_7d",
    "roc_7d_rank",
    "atr_pct_14",
    "log_return_1d",
    "pump20_rule",
    "pump15_rule",
    "estimated_pump15_prob",
    "overheated_flag",
]
PUMP_LEDGER_COLS = RECOMMEND_LEDGER_COLS + EXTRA_LEDGER_COLS

KST = timezone(timedelta(hours=9))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("pump_detector_today")


def _today_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d")


def append_today(asof: str, *, dry_run: bool = False,
                 ledger_path: str = PUMP_HUNTER_LEDGER,
                 top_universe: int = UNIVERSE_TOP_N,
                 max_candidates: int = MAX_CANDIDATES,
                 limit_markets: int | None = None) -> dict:
    res = score_pump_candidates(
        asof,
        top_universe=top_universe,
        max_candidates=max_candidates,
        limit_markets=limit_markets,
    )
    log.info(
        "asof=%s feature_date=%s universe_n=%d candidates=%d",
        res["asof"],
        res["feature_date"],
        res["universe_n"],
        res["n_candidates"],
    )

    candidates = res.get("candidates", [])
    if not candidates:
        log.warning("no PUMP hunter candidates — nothing to record")
        return res

    rows = []
    for item in candidates:
        pump_prob = float(item.get("estimated_pump20_prob", EST_PUMP20_PROB))
        rows.append({
            "date": res["asof"],
            "coin": item["market"],
            "rank": int(item["rank"]),
            "score": float(item["score"]),
            "pump_prob": pump_prob,
            "pump_prob_pct": f"{pump_prob * 100:.1f}%",
            "dump_risk_flag": bool(item.get("dump_risk_flag", False)),
            "btc_regime": item.get("btc_regime", "unknown"),
            "entry_open": item.get("entry_open"),
            "sl_pct": SL_PCT,
            "tp_pct": TP_PCT,
            "status": "open",
            "calibration_source": "pump_rule_discovery_v1_prior",
            "p_up5": pd.NA,
            "p_up10": pd.NA,
            "p_up20": pump_prob,
            "p_dn5": pd.NA,
            "p_dn10": pd.NA,
            "exp_downside": pd.NA,
            "rr_ratio": pd.NA,
            "exit_price": pd.NA,
            "exit_reason": pd.NA,
            "realized_pct": pd.NA,
            "pump20_hit": pd.NA,
            "closed_at": pd.NA,
            "model_id": res["model_id"],
            "rule_version": res["rule_version"],
            "rule_id": item.get("rule_id"),
            "feature_date": res["feature_date"],
            "liq_rank_daily": item.get("liq_rank_daily"),
            "roc_7d": item.get("roc_7d"),
            "roc_7d_rank": item.get("roc_7d_rank"),
            "atr_pct_14": item.get("atr_pct_14"),
            "log_return_1d": item.get("log_return_1d"),
            "pump20_rule": bool(item.get("pump20_rule", False)),
            "pump15_rule": bool(item.get("pump15_rule", False)),
            "estimated_pump15_prob": item.get("estimated_pump15_prob"),
            "overheated_flag": bool(item.get("overheated_flag", False)),
        })

    new = pd.DataFrame(rows)
    for col in PUMP_LEDGER_COLS:
        if col not in new.columns:
            new[col] = pd.NA
    new = new[PUMP_LEDGER_COLS]

    p = Path(ledger_path)
    if p.exists():
        existing = pd.read_csv(p)
        for col in PUMP_LEDGER_COLS:
            if col not in existing.columns:
                existing[col] = pd.NA
        ordered = [c for c in PUMP_LEDGER_COLS if c in existing.columns]
        extras = [c for c in existing.columns if c not in ordered]
        existing = existing[ordered + extras]
        already = existing[existing["date"].astype(str) == res["asof"]]
        if len(already) > 0:
            log.warning(
                "asof=%s already has %d rows in %s — skip append (idempotent)",
                res["asof"],
                len(already),
                p,
            )
            _print_rows(already)
            return res
        combined = pd.concat([existing, new], ignore_index=True)
    else:
        combined = new

    if dry_run:
        log.info("[dry-run] would append %d rows to %s", len(new), p)
        _print_rows(new)
        return res

    p.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(p, index=False)
    log.info("appended %d rows -> %s (total %d)", len(new), p, len(combined))
    _print_rows(new)
    return res


def _print_rows(df: pd.DataFrame) -> None:
    cols = [
        "date",
        "coin",
        "rank",
        "score",
        "pump_prob_pct",
        "rule_id",
        "roc_7d_rank",
        "atr_pct_14",
        "log_return_1d",
        "status",
    ]
    cols = [c for c in cols if c in df.columns]
    print("\n=== PUMP hunter candidates (SHADOW, record-only) ===")
    print(df[cols].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="PUMP hunter daily record-only ledger append")
    ap.add_argument("--asof", type=str, default=None, help="YYYY-MM-DD (default=today KST)")
    ap.add_argument("--ledger", type=str, default=PUMP_HUNTER_LEDGER)
    ap.add_argument("--top-universe", type=int, default=UNIVERSE_TOP_N)
    ap.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    ap.add_argument("--limit-markets", type=int, default=None, help="development only")
    ap.add_argument("--dry-run", action="store_true", help="do not write ledger")
    args = ap.parse_args()

    append_today(
        args.asof or _today_kst(),
        dry_run=args.dry_run,
        ledger_path=args.ledger,
        top_universe=args.top_universe,
        max_candidates=args.max_candidates,
        limit_markets=args.limit_markets,
    )


if __name__ == "__main__":
    main()
