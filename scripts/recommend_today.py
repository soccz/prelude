"""SHADOW 채널 일일 추천 기록 스크립트 (기록만 — 발송/주문/cron 전부 X).

흐름:
  1. signals.recommend.score_candidates(today) 호출 → leak-free rank-mean top-3
  2. top-3 를 output/shadow_ledger_recommend.csv 에 append (status='open')
  3. 청산 realized 는 다음날 scripts/close_recommend_ledger.py 가 채운다 (status='closed')

★★★ 이 스크립트는 SHADOW(기록만) 채널이다 (CLAUDE.md §3.1, ops-steward §0):
    - 텔레그램 발송 코드 없음.
    - cron/systemd timer 미등록 (수동/추후 사용자 sudo 후 등록).
    - 업비트 자동주문·API key 절대 없음. 진입가(entry_open)는 DB 의 그날 open(09:00)
      을 시그널 모듈이 부착한 플랜값일 뿐 — 실제 주문 X.
    - 사이징/실제 청산 시뮬은 ledger 책임 (여기서는 sl/tp 플랜값만 기록).

진입 = day-D open(09:00 KST). 청산 플랜 = -3% 손절 / +5% 익절 (shadow 가상평가용).
유니버스/라벨/사이징은 사용자 확정값 그대로 (signals.recommend 상수).

거래비용: realized PnL 차감은 청산 스크립트(close_recommend_ledger.py)에서 net 처리.

사용:
    python scripts/recommend_today.py                 # 오늘(KST) 기록
    python scripts/recommend_today.py --asof 2026-05-31
    python scripts/recommend_today.py --dry-run       # 기록 X, 출력만
    python scripts/recommend_today.py --limit-markets 120  # 개발용
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signals.recommend import score_candidates  # noqa: E402

# SHADOW recommend ledger — 기존 shadow_ledger_distribution.csv 컨벤션 참고.
# date/coin/rank/score/pump_prob/dump_risk_flag/btc_regime/entry_open/sl_pct/tp_pct/status
# + 청산 후 채움(close_recommend_ledger.py): exit_price/exit_reason/realized_pct/pump20_hit/closed_at
SHADOW_RECOMMEND_LEDGER = "output/shadow_ledger_recommend.csv"

RECOMMEND_LEDGER_COLS = [
    "date",            # 추천일 D (= 진입일, KST 일봉 09:00)
    "coin",            # KRW-XXX
    "rank",            # 1..3
    "score",           # rank-mean score (0~1)
    "pump_prob",       # calibrated 급등확률 (bucket historical-hit)
    "pump_prob_pct",   # 정직 표기 문자열 (예: "5.5%")
    "dump_risk_flag",  # ⚠️ hi-risk (상위 1/3) bool
    "btc_regime",      # D-1 regime
    "entry_open",      # 진입가 = day-D open(09:00)
    "sl_pct",          # 청산 플랜 손절 = -0.03
    "tp_pct",          # 청산 플랜 익절 = +0.05
    "status",          # 'open' → close 후 'closed'
    "calibration_source",  # bucket_score_pump20 | base_rate
    # --- 평가용 분포/하방 확률 (SHADOW→ADOPT calibration 정직성·rr 정렬 감사용) ---
    "p_up5",           # P(고가 ≥+5%)  calibrated
    "p_up10",          # P(고가 ≥+10%) calibrated (R1 분자)
    "p_up20",          # P(고가 ≥+20%) calibrated
    "p_dn5",           # P(저가 ≤-5%)  calibrated (R1 분모, ~-5% 손실 anchor)
    "p_dn10",          # P(저가 ≤-10%) deep-dump
    "exp_downside",    # E[하방] (음수 low excursion 기대값)
    "rr_ratio",        # R1 정렬키 = p_up10 / max(p_dn5, eps)
    # --- 청산 후 채움 (close_recommend_ledger.py) ---
    "exit_price",      # 청산가 (15m 경로)
    "exit_reason",     # SL | TP | EOD
    "realized_pct",    # net 실현 수익률 % (왕복 0.15% 차감 후)
    "pump20_hit",      # 일봉 라벨: (high_D/open_D - 1) >= 0.20 적중 여부 (0/1)
    "closed_at",       # 청산 기록 시각 (UTC iso)
]

KST = timezone(timedelta(hours=9))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("recommend_today")


def _today_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d")


def append_today(asof: str, *, dry_run: bool = False,
                 limit_markets: int | None = None,
                 ledger_path: str = SHADOW_RECOMMEND_LEDGER) -> dict:
    """asof 기준 top-3 를 shadow recommend ledger 에 append (status='open')."""
    res = score_candidates(asof, limit_markets=limit_markets)
    log.info("asof=%s btc_regime=%s universe_n=%d calib=%s n_hist_dates=%d top=%d",
             res["asof"], res["btc_regime"], res["universe_n"],
             res["calibration_source"], res["n_history_dates"], len(res["top3"]))

    if not res["top3"]:
        log.warning("top3 empty — nothing to record (universe 0?)")
        return res

    rows = []
    for item in res["top3"]:
        rows.append({
            "date": res["asof"],
            "coin": item["coin"],
            "rank": item["rank"],
            "score": item["score"],
            "pump_prob": item["pump_prob"],
            "pump_prob_pct": item["pump_prob_pct"],
            "dump_risk_flag": bool(item["dump_risk_flag"]),
            "btc_regime": item["btc_regime"],
            "entry_open": item["entry_open"],
            "sl_pct": item["sl"],     # -0.03
            "tp_pct": item["tp"],     # +0.05
            "status": "open",
            "calibration_source": res["calibration_source"],
            "p_up5": item.get("p_up5"),
            "p_up10": item.get("p_up10"),
            "p_up20": item.get("p_up20"),
            "p_dn5": item.get("p_dn5"),
            "p_dn10": item.get("p_dn10"),
            "exp_downside": item.get("exp_downside"),
            "rr_ratio": item.get("rr_ratio"),
            "exit_price": pd.NA,
            "exit_reason": pd.NA,
            "realized_pct": pd.NA,
            "pump20_hit": pd.NA,
            "closed_at": pd.NA,
        })
    new = pd.DataFrame(rows)[RECOMMEND_LEDGER_COLS]

    p = Path(ledger_path)
    if p.exists():
        existing = pd.read_csv(p)
        for c in RECOMMEND_LEDGER_COLS:
            if c not in existing.columns:
                existing[c] = pd.NA
        existing = existing[RECOMMEND_LEDGER_COLS]
        # 같은 날 이미 기록돼 있으면 중복 append 방지 (idempotent).
        already = existing[(existing["date"].astype(str) == res["asof"])]
        if len(already) > 0:
            log.warning("asof=%s already has %d rows in %s — skip append (idempotent)",
                        res["asof"], len(already), p)
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
    log.info("appended %d rows → %s (total %d)", len(new), p, len(combined))
    _print_rows(new)
    return res


def _print_rows(df: pd.DataFrame) -> None:
    cols = ["date", "coin", "rank", "score", "pump_prob", "pump_prob_pct",
            "dump_risk_flag", "btc_regime", "entry_open", "sl_pct", "tp_pct", "status"]
    cols = [c for c in cols if c in df.columns]
    print("\n=== recorded top-3 (SHADOW, record-only) ===")
    print(df[cols].to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description="SHADOW 일일 추천 기록 (발송/주문/cron X)")
    ap.add_argument("--asof", type=str, default=None, help="YYYY-MM-DD (default=오늘 KST)")
    ap.add_argument("--limit-markets", type=int, default=None, help="개발용 마켓 제한")
    ap.add_argument("--dry-run", action="store_true", help="기록 X, 출력만")
    ap.add_argument("--ledger", type=str, default=SHADOW_RECOMMEND_LEDGER)
    args = ap.parse_args()

    asof = args.asof or _today_kst()
    append_today(asof, dry_run=args.dry_run,
                 limit_markets=args.limit_markets, ledger_path=args.ledger)


if __name__ == "__main__":
    main()
