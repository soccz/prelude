"""PUMP hunter v2 daily runner — shadow ledger append + 텔레그램 radar 발사.

사용자 컨펌 (2026-06-11): "텔레그램으로 오게 제대로 완성" — v2 radar 메시지 추가.

발사 정책:
- 후보 ≥ 1 → 🎯 radar 메시지 (hit 엣지 + 자동 net 음수 정직 고지 포함)
- 후보 0 + binance stale → ⚠️ 데이터 stale 1줄 (운영 알림)
- 후보 0 + 정상 → 텔레그램 X (로그만 — 매일 5번째 빈 메시지 방지)

원장: output/shadow_ledger_pump_hunter_v2.csv (close_recommend_ledger 가
TP5/SL3 + exit lab 7 잣대 자동 청산 — record-only, 자동 주문 없음).
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notifier.telegram import send_telegram  # noqa: E402
from scripts.recommend_today import RECOMMEND_LEDGER_COLS  # noqa: E402
from signals.pump_detector_v2 import (  # noqa: E402
    MAX_CANDIDATES,
    OOS_BASE_RATE_PCT,
    OOS_HIT_PCT,
    OOS_NET_TP5SL3_PCT,
    SL_PCT,
    TP_PCT,
    score_pump_v2_candidates,
)

PUMP_V2_LEDGER = "output/shadow_ledger_pump_hunter_v2.csv"

EXTRA_LEDGER_COLS = [
    "model_id", "rule_version", "rule_id", "feature_date",
    "liq_rank_daily", "roc_7d", "roc_7d_rank", "atr_pct_14", "log_return_1d",
    "b_vol_surge", "b_ret_1d",
]
PUMP_V2_LEDGER_COLS = RECOMMEND_LEDGER_COLS + EXTRA_LEDGER_COLS

KST = timezone(timedelta(hours=9))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("pump_detector_v2_today")

BTC_KR = {
    "bull_quiet": "🟢 강세 안정", "bull_volatile": "🟡 강세 변동",
    "bear_quiet": "🔴 약세 안정", "bear_volatile": "🔴 약세 변동",
}


def _today_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d")


def _fmt_price(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    if v >= 1000:
        return f"{v:,.0f}원"
    if v >= 1:
        return f"{v:.1f}원"
    return f"{v:.4f}원"


def build_message(res: dict, dry_run: bool = False) -> str:
    date_str = res["asof"]
    header = f"🎯 PUMP 레이더 v2 {date_str} (KST 09:05)"
    if dry_run:
        header += "  [DRY-RUN]"
    lines = [header]
    regime = BTC_KR.get(str(res.get("btc_regime", "")), str(res.get("btc_regime", "—")))
    lines.append(f"BTC: {regime} | rule: 7d모멘텀 상위 + Binance 거래량 surge")
    lines.append("")

    cands = res.get("candidates", [])
    if not cands:
        if str(res.get("binance_status", "ok")) != "ok":
            lines.append(f"⚠️ binance 데이터 문제 — {res.get('binance_status')}")
        else:
            lines.append("━━━ 오늘 rule fire 없음 ━━━")
        return "\n".join(lines)

    lines.append(f"━━━ 급등 후보 {len(cands)}건 (검증 hit {OOS_HIT_PCT}% ≈ base {OOS_BASE_RATE_PCT}% 의 6배) ━━━")
    lines.append("")
    for c in cands:
        coin = str(c["market"]).replace("KRW-", "")
        lines.append(
            f"#{c['rank']} {coin}  기준가 ≈ {_fmt_price(c['entry_open'])}"
        )
        lines.append(
            f"   ▸ Binance surge {c['b_vol_surge']:.1f}× | 7d모멘텀 rank {c['roc_7d_rank']:.2f}"
            f" | 7d {c['roc_7d']:+.1f}%"
        )
    lines.append("")
    lines.append("━━━ 사용 ━━━")
    lines.append(f"• 검증 (최근 7개월 OOS): +20% 도달 {OOS_HIT_PCT}% — 약 12건 중 1건")
    lines.append(f"• ★ 자동 룰 (TP5/SL3) net 은 {OOS_NET_TP5SL3_PCT}% 음수 — 진입·청산은 본인 판단")
    lines.append("• 하락 위험 동반 — 사이즈 작게, 자동 주문 없음 (shadow 기록만)")
    return "\n".join(lines)


def append_ledger(res: dict, ledger_path: str, dry_run: bool) -> None:
    cands = res.get("candidates", [])
    if not cands:
        log.info("no v2 candidates — ledger skip")
        return
    rows = []
    for c in cands:
        rows.append({
            "date": res["asof"], "coin": c["market"], "rank": c["rank"],
            "score": c["score"], "pump_prob": OOS_HIT_PCT / 100.0,
            "pump_prob_pct": f"{OOS_HIT_PCT:.1f}%",
            "dump_risk_flag": False, "btc_regime": c.get("btc_regime", "unknown"),
            "entry_open": c["entry_open"], "sl_pct": SL_PCT, "tp_pct": TP_PCT,
            "status": "open", "calibration_source": "binance_leadlag_v1_oos",
            "p_up20": OOS_HIT_PCT / 100.0,
            "model_id": res["model_id"], "rule_version": res["rule_version"],
            "rule_id": c["rule_id"], "feature_date": res.get("feature_date"),
            "liq_rank_daily": c["liq_rank_daily"], "roc_7d": c["roc_7d"],
            "roc_7d_rank": c["roc_7d_rank"], "atr_pct_14": c["atr_pct_14"],
            "log_return_1d": c["log_return_1d"],
            "b_vol_surge": c["b_vol_surge"], "b_ret_1d": c.get("b_ret_1d"),
        })
    new = pd.DataFrame(rows)
    for col in PUMP_V2_LEDGER_COLS:
        if col not in new.columns:
            new[col] = pd.NA
    new = new[PUMP_V2_LEDGER_COLS]

    p = Path(ledger_path)
    if p.exists():
        existing = pd.read_csv(p)
        for col in PUMP_V2_LEDGER_COLS:
            if col not in existing.columns:
                existing[col] = pd.NA
        ordered = [c for c in PUMP_V2_LEDGER_COLS if c in existing.columns]
        extras = [c for c in existing.columns if c not in ordered]
        existing = existing[ordered + extras]
        if (existing["date"].astype(str) == res["asof"]).any():
            log.warning("asof=%s already in %s — skip append (idempotent)", res["asof"], p)
            return
        combined = pd.concat([existing, new], ignore_index=True)
    else:
        combined = new
    if dry_run:
        log.info("[dry-run] would append %d rows to %s", len(new), p)
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(p, index=False)
    log.info("appended %d rows -> %s (total %d)", len(new), p, len(combined))


def main() -> None:
    ap = argparse.ArgumentParser(description="PUMP hunter v2 daily — shadow + telegram radar")
    ap.add_argument("--asof", type=str, default=None)
    ap.add_argument("--ledger", type=str, default=PUMP_V2_LEDGER)
    ap.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    ap.add_argument("--send-telegram", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="ledger 기록 X")
    args = ap.parse_args()

    asof = args.asof or _today_kst()
    res = score_pump_v2_candidates(asof, max_candidates=args.max_candidates)
    log.info("asof=%s universe=%s binance=%s candidates=%d",
             res["asof"], res.get("universe_n"), res.get("binance_status"),
             res.get("n_candidates", 0))

    append_ledger(res, args.ledger, args.dry_run)

    msg = build_message(res, dry_run=not args.send_telegram)
    print(msg)

    has_cands = res.get("n_candidates", 0) > 0
    stale = str(res.get("binance_status", "ok")) != "ok"
    if args.send_telegram and (has_cands or stale):
        try:
            if send_telegram(msg):
                log.info("telegram sent")
            else:
                log.error("telegram not sent")
        except Exception as e:  # noqa: BLE001
            log.error("telegram fail: %s", e)
    elif args.send_telegram:
        log.info("telegram skipped: 후보 0 + binance ok (무소음 정책)")


if __name__ == "__main__":
    main()
