#!/usr/bin/env python
"""self_impact_decay_v1 — 영상 효율시장 아이러니: 발사(ACTIVE)가 alpha 를 자가소멸시키나?

Veritasium "The Trillion Dollar Equation": Bachelier 효율시장 — 예측은 가격에 반영돼 사라진다.
prelude 가 텔레그램으로 발사(ACTIVE)한 코인은 사용자(+잠재 군중) 매수로 진입가가 밀려 alpha 를
스스로 죽일 수 있다. WATCH_ONLY 는 같은 모델이 본 후보지만 발사 안 함(침묵). 두 그룹의 forward
realized 차 = self-impact + selection.

★★ 구조적 한계 (현재 데이터): shadow_ledger_distribution.csv = 30일(2026-05~06), ACTIVE 20(전부
A_TRIPLE) vs WATCH_ONLY A_TRIPLE **5건**, 전 구간 **bear regime** → 인과 추정 불가(INSUFFICIENT).
selection bias: ACTIVE 는 더 강한 후보로 선택됨(confidence_score↑) → naive ACTIVE-WATCH 차는
self-impact(음의 방향: 발사가 진입가 밀어 forward↓)와 selection(양의 방향: 발사가 더 좋은 픽)이 혼재.
n=5 로는 분리 불가.

이 스크립트는 **재실행 가능한 측정 하네스 + 현재 raw read + 누적 게이트**다. 라이브 누적 시 자동 성숙.
게이트: ACTIVE n≥50 AND WATCH_ONLY(A_TRIPLE) n≥30 이전엔 INSUFFICIENT_SAMPLE — 판정 보류.
forward 는 candles 에서 직접 계산(ledger status 무관). 진입=발사일 09:00 open, EOD=같은 봉 close.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("self_impact")

LEDGER = Path("output/shadow_ledger_distribution.csv")
D1_DB = "data/upbit_d1.db"
OUT_DIR = Path("output")
COST = 0.0015
MIN_ACTIVE = 50          # 누적 게이트
MIN_WATCH_ATRIPLE = 30


def fetch_forward(coin: str, date: str, conn) -> dict:
    """발사일 일봉(09:00 open) → EOD net / max. candles 에서 직접(ledger status 무관)."""
    row = conn.execute(
        "SELECT open, high, low, close FROM candles "
        "WHERE market=? AND substr(timestamp,1,10)=? LIMIT 1", (coin, date)).fetchone()
    if not row or not row[0] or row[0] <= 0:
        return {}
    o, h, l, c = row
    return dict(fwd_eod_net=c / o - 1 - COST, fwd_max=h / o - 1, fwd_min=l / o - 1)


def grp_stats(d: pd.DataFrame) -> dict:
    if not len(d):
        return dict(n=0)
    v = d["fwd_eod_net"].dropna().values
    return dict(n=int(len(v)), fwd_eod_net=float(np.mean(v)) if len(v) else np.nan,
                hit=float(np.mean(v > 0)) if len(v) else np.nan,
                fwd_max=float(d["fwd_max"].mean()), fwd_min=float(d["fwd_min"].mean()),
                conf=float(d["confidence_score"].mean()) if "confidence_score" in d else np.nan)


def main():
    led = pd.read_csv(LEDGER)
    conn = sqlite3.connect(D1_DB)
    fwd = led.apply(lambda r: fetch_forward(r["coin"], str(r["date"])[:10], conn), axis=1)
    conn.close()
    led = pd.concat([led, pd.DataFrame(list(fwd))], axis=1)
    cov = led["fwd_eod_net"].notna()
    log.info("ledger %d행, forward 조인 성공 %d (%.0f%%), date %s~%s, regime=%s",
             len(led), int(cov.sum()), 100 * cov.mean(), led.date.min(), led.date.max(),
             led.btc_regime.value_counts().to_dict())

    led = led[cov].copy()
    rows = []
    log.info("\n===== decision 별 forward (전체) =====")
    for dec in ["ACTIVE", "WATCH_ONLY", "SILENCE"]:
        s = grp_stats(led[led.decision == dec])
        if s["n"]:
            log.info("  %-11s n=%-4d fwd_eod_net=%+.4f hit=%.3f fwd_max=%+.4f conf=%.1f",
                     dec, s["n"], s.get("fwd_eod_net", np.nan), s.get("hit", np.nan),
                     s.get("fwd_max", np.nan), s.get("conf", np.nan))
        rows.append(dict(strat="ALL", decision=dec, **s))

    # A_TRIPLE 만 (ACTIVE 가 사는 유일 stratum) — 매칭 비교
    log.info("\n===== A_TRIPLE 내 (ACTIVE vs WATCH 매칭 — 같은 setup_quality) =====")
    at = led[led.setup_quality == "A_TRIPLE"]
    act = grp_stats(at[at.decision == "ACTIVE"])
    wat = grp_stats(at[at.decision == "WATCH_ONLY"])
    for dec, s in [("ACTIVE", act), ("WATCH_ONLY", wat)]:
        log.info("  %-11s n=%-4d fwd_eod_net=%+.4f hit=%.3f conf=%.1f",
                 dec, s["n"], s.get("fwd_eod_net", np.nan), s.get("hit", np.nan), s.get("conf", np.nan))
        rows.append(dict(strat="A_TRIPLE", decision=dec, **s))

    naive_att = (act.get("fwd_eod_net", np.nan) - wat.get("fwd_eod_net", np.nan)
                 if act["n"] and wat["n"] else np.nan)
    conf_gap = (act.get("conf", np.nan) - wat.get("conf", np.nan)
                if act["n"] and wat["n"] else np.nan)

    # ---------- 누적 게이트 ----------
    sufficient = act["n"] >= MIN_ACTIVE and wat["n"] >= MIN_WATCH_ATRIPLE
    bull_n = int((led.btc_regime.astype(str).str.startswith("bull")).sum())
    log.info("\n===== 판정 =====")
    log.info("  naive ATT(ACTIVE-WATCH, A_TRIPLE) = %s  | conf_gap = %s (양수=ACTIVE 가 더 강한 후보=selection)",
             f"{naive_att:+.4f}" if not np.isnan(naive_att) else "n/a",
             f"{conf_gap:+.1f}" if not np.isnan(conf_gap) else "n/a")
    log.info("  표본: ACTIVE n=%d (게이트 %d), WATCH(A_TRIPLE) n=%d (게이트 %d), bull regime n=%d",
             act["n"], MIN_ACTIVE, wat["n"], MIN_WATCH_ATRIPLE, bull_n)
    if not sufficient:
        log.info("  → **INSUFFICIENT_SAMPLE** — 판정 보류. naive ATT 은 self-impact(음)와 selection(양)이")
        log.info("     혼재돼 해석 불가(n=%d WATCH). 전 구간 bear(bull %d) — 일반화 불가.", wat["n"], bull_n)
        log.info("     누적 필요: ACTIVE %d→%d, WATCH A_TRIPLE %d→%d (~2-3개월 라이브). RD-around-cut 은 더 큰 n 필요.",
                 act["n"], MIN_ACTIVE, wat["n"], MIN_WATCH_ATRIPLE)
        verdict = "INSUFFICIENT_SAMPLE"
    else:
        verdict = "READY — selection 통제(conf 매칭/RD) 후 self-impact 추정 가능"
        log.info("  → %s", verdict)

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "self_impact_decay_v1.csv", index=False)
    coverage = {
        "ledger": str(LEDGER), "n_rows": int(len(led)),
        "date_range": [str(led.date.min()), str(led.date.max())],
        "regime_dist": {k: int(v) for k, v in led.btc_regime.value_counts().items()},
        "decision_dist": {k: int(v) for k, v in led.decision.value_counts().items()},
        "naive_att_a_triple": None if np.isnan(naive_att) else float(naive_att),
        "conf_gap_a_triple": None if np.isnan(conf_gap) else float(conf_gap),
        "active_n": act["n"], "watch_atriple_n": wat["n"], "bull_n": bull_n,
        "gate": {"min_active": MIN_ACTIVE, "min_watch_atriple": MIN_WATCH_ATRIPLE,
                 "sufficient": bool(sufficient)},
        "verdict": verdict,
        "caveats": [
            "selection bias: ACTIVE 는 더 강한 후보로 선택(conf_gap>0) → naive ATT 은 self-impact(음)와 혼재",
            "전 구간 bear regime — bull 일반화 불가",
            "intraday 슬립 미기록 → mechanical(진입가 밀림) vs crowding 분리 불가",
            "forward=발사일 일봉 EOD(09:00 open→close). 실제 사용자 진입 타이밍/사이징 미반영",
        ],
        "next_step": "라이브 누적 → 게이트 충족 시 (1)conf-매칭 또는 (2)RD around hit≥cut 으로 selection 통제 후 ATT 재추정.",
    }
    (OUT_DIR / "self_impact_decay_coverage_v1.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2, default=float))
    log.info("\nwrote output/self_impact_decay_v1.csv + self_impact_decay_coverage_v1.json")


if __name__ == "__main__":
    main()
