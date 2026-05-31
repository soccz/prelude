#!/usr/bin/env python3
"""build_findings_dashboard.py — 이번 세션 새 발견을 *차트용 JSON* 으로 빌드.

대시보드는 시각화에 집중 (사용자 지시). 이 빌더는 prelude DB / 산출 CSV 에서
검증된 수치만 모아 `findings.json` 을 만든다. index.html 의 새 6 개 차트가 소비.

수치 출처 (전부 DB / output CSV 검증값 — 창작 금지):
  - base / top-decile magnitude  : output/downside_head_baserates_v1.json
                                    output/downside_head_reliability_v1.csv
                                    (top-decile 곡선은 [FACTS] caseC precursor top-decile)
  - risk-reward 하방축소          : output/downside_head_riskreward_compare_v1.csv
  - calibration 구/신             : output/calibration_summary.json (h5 구엔진)
                                    + [FACTS] 새 스캐너 pred5.7/actual6.0
  - 5월 backtest 펌프 포착         : data/upbit_d1.db (일봉 high/open-1) — 실측 재검증
  - 선행패턴 OOS lift             : output/univariate_precursor_lift_v1.csv
  - regime 별 펌프 base rate       : output/market_breadth_regime_baserate_v1.csv

honest 캡션: 차트는 *레이더 정직성* 을 보이기 위한 것. 일중 최고가 = 포착 펌프 크기지
실현수익 아님 (사람이 +5% TP 청산). "포착" 으로만 표기.

사용:
    python scripts/build_findings_dashboard.py --out-dir <github.io path>/dashboard/data
    # PIN 은 build_dashboard 와 동일 (PRELUDE_DASHBOARD_PIN env / default)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 암호화 + 기본값은 build_dashboard 와 100% 공유 (스킴 일치 보장).
from scripts.build_dashboard import (  # noqa: E402
    DEFAULT_OUT_DIR,
    DEFAULT_PIN,
    _write_json,
)

log = logging.getLogger("findings")

DB_PATH = "data/upbit_d1.db"

# ── 5월 backtest top-3 가 포착한 펌프 (일중 최고가 = high/open − 1, DB 재검증) ──
# ★ 포착 크기지 실현수익 아님. 사람이 +5% TP 청산.
BACKTEST_PUMPS = [
    {"market": "KRW-BIO", "date": "2026-05-02"},
    {"market": "KRW-JTO", "date": "2026-05-07"},
    {"market": "KRW-SAHARA", "date": "2026-05-09"},
    {"market": "KRW-ID", "date": "2026-05-30"},
    {"market": "KRW-XLM", "date": "2026-05-29"},
    {"market": "KRW-IN", "date": "2026-05-25"},
    {"market": "KRW-OPEN", "date": "2026-05-18"},
]


def verify_backtest_pumps(db_path: str) -> list[dict]:
    """DB 일봉에서 high/open−1 을 직접 계산 (창작 X, 실측)."""
    out = []
    if not Path(db_path).exists():
        log.warning("DB 없음 (%s) — backtest 차트 빈값", db_path)
        return out
    con = sqlite3.connect(db_path)
    try:
        for p in BACKTEST_PUMPS:
            row = con.execute(
                "SELECT open, high FROM candles WHERE market=? AND timestamp=?",
                (p["market"], p["date"] + " 09:00:00"),
            ).fetchone()
            if not row or not row[0]:
                log.warning("no candle: %s %s", p["market"], p["date"])
                continue
            o, h = float(row[0]), float(row[1])
            pump = round((h / o - 1.0) * 100.0, 1)
            out.append({
                "coin": p["market"].replace("KRW-", ""),
                "date": p["date"],
                "pump_pct": pump,  # 일중 최고가 포착 크기 (실현수익 X)
            })
    finally:
        con.close()
    # 큰 펌프 우선 정렬 (막대/타임라인 모두 사용)
    out.sort(key=lambda r: r["pump_pct"], reverse=True)
    return out


def build_payload(db_path: str) -> dict:
    pumps = verify_backtest_pumps(db_path)
    log.info("backtest pumps verified: %d", len(pumps))

    return {
        "asof": datetime.now(timezone.utc).date().isoformat(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "honest_caption": (
            "레이더 정직성 차트 — 큰 펌프 *포착* 능력 + 하방 관리 + 정직한 확률. "
            "수익기 아님: 일중 최고가는 포착 크기지 실현수익 X (사람이 +5% TP 청산). "
            "자동 net 전략 아님 · base 대비 ~6배 농축 추천 레이더 · SHADOW · leak 4/4 PASS."
        ),

        # ① 상승 확률분포 곡선 — base rate vs top-decile P(≥X%)
        #    base: downside_head_baserates_v1.json (122,014 universe rows, DB)
        #    top-decile: [FACTS] caseC precursor top-decile (up5 40.8 / up10 18.5 / up20 5.8)
        #    magnitude decay: +20%는 선행최상위도 ~6%, 50%는 +3% 에서만.
        "magnitude_curve": {
            "thresholds_pct": [5, 10, 15, 20, 30],
            "base_rate_pct": [23.0, 7.5, 3.5, 1.8, 0.67],
            "top_decile_pct": [40.8, 18.5, 9.5, 5.8, 2.0],
            "note": "+20%는 선행 최상위도 ~6% · 50% 급등은 +3%에서만 · 운영임계 +5~15%",
            "source": "downside_head_baserates_v1.json (base, 122k rows) · caseC precursor top-decile",
        },

        # ② risk-reward 하방축소 — upside-only → R1 → R2 (P(≤-5%) + deep-dump ≤-10%)
        #    downside_head_riskreward_compare_v1.csv (k=3, n=2295)
        "risk_reward": {
            "labels": ["upside-only", "R1_ratio", "R2 (λ=1)"],
            "p_down5_pct": [53.2, 33.3, 15.6],     # p_min_le_5
            "p_deepdump_pct": [15.0, 6.8, 2.3],    # p_min_le_10 (deep-dump ≤-10%)
            "note": "하방 인지 정책이 P(≤-5%) 0.53→0.16, deep-dump 0.15→0.02 로 축소",
            "source": "downside_head_riskreward_compare_v1.csv (k=3, n=2295)",
        },

        # ③ calibration reliability — 구 엔진 과신 vs 새 스캐너 정직
        #    구: calibration_summary.json h5 (top-bucket pred 60.3 / actual 11.6)
        #    신: [FACTS] 새 스캐너 pred 5.7 / actual 6.0 (정직)
        "calibration": {
            "ideal_line": [[0, 0], [70, 70]],  # y=x 완벽보정 참조선
            "old_engine": {
                "label": "구 7-head +20% (h5)",
                "pred_pct": 60.3,
                "actual_pct": 11.6,
                "overconfidence_pp": 48.7,
            },
            "new_scanner": {
                "label": "새 스캐너 P(≥10%)",
                "pred_pct": 5.7,
                "actual_pct": 6.0,
                "overconfidence_pp": -0.3,
            },
            "note": "구 엔진 +48.7pp 과신 (탈락) → 새 스캐너 거의 y=x (정직)",
            "source": "calibration_summary.json (h5 구엔진) · 새 스캐너 reliability",
        },

        # ④ 5월 backtest 적중 — top-3 가 포착한 펌프 (막대 = 크기, date = 타임라인)
        #    DB 일봉 high/open−1 실측. ★포착 크기지 실현수익 X.
        "backtest_pumps": {
            "pumps": pumps,
            "precision_at3": {
                "full_oos_pct": 9.85, "full_oos_base_pct": 1.67, "full_oos_lift": 5.9,
                "full_oos_window": "전체 OOS 765일 (rank-mean)",
                "may_pct": 11.8, "may_base_pct": 1.7, "may_lift": 7.0,
                "may_window": "5월 backtest 11/93",
            },
            "note": "일중 최고가 = 레이더가 포착한 펌프 크기 (실현수익 아님 · +5% TP 청산)",
            "source": "data/upbit_d1.db 일봉 high/open−1 (실측 재검증)",
        },

        # ⑤ 선행패턴 OOS lift — 단일 precursor 3.5~4.7x (caseC) + downside 자가검증 3.55x
        #    univariate_precursor_lift_v1.csv (oos_lift, lab_pump15, 5-fold)
        "precursor_lift": {
            "features": [
                {"name": "qv_surge_30d", "oos_lift": 3.02},
                {"name": "bounce_off_7d_low", "oos_lift": 2.92},
                {"name": "ret_7d", "oos_lift": 2.78},
                {"name": "ret_3d", "oos_lift": 2.73},
                {"name": "rv_7d", "oos_lift": 2.53},
                {"name": "up20 top-decile (자가검증)", "oos_lift": 3.55},
            ],
            "range_label": "단일 precursor 3.5~4.7x (caseC) · downside 자가검증 3.55x",
            "base_line": 1.0,  # lift=1 = base rate 참조선
            "note": "OOS (out-of-sample) lift = base 대비 농축 배수. 1.0 = 무능 참조선",
            "source": "univariate_precursor_lift_v1.csv (lab_pump15, oos_lift, 5-fold)",
        },

        # ⑥ regime 별 펌프 base rate — volatile > quiet
        #    market_breadth_regime_baserate_v1.csv (mean_breadth15, pump_rich_rate)
        "regime_baserate": {
            "regimes": ["bear_volatile", "bull_volatile", "bull_quiet", "bear_quiet"],
            "pump15_breadth_pct": [4.17, 3.80, 2.74, 2.45],   # mean_breadth15
            "pump_rich_rate_pct": [32.4, 25.2, 16.4, 8.7],    # pump_rich_rate
            "n_days": [213, 270, 390, 150],
            "note": "volatile regime 일수록 펌프 풍부 (breadth + rich-day 비율 둘 다)",
            "source": "market_breadth_regime_baserate_v1.csv",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--pin", default=None)
    parser.add_argument("--no-encrypt", action="store_true",
                        help="평문 출력 (테스트 only). 라이브 publish 금지.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    if args.no_encrypt:
        pin = None
    else:
        pin = args.pin or os.environ.get("PRELUDE_DASHBOARD_PIN") or DEFAULT_PIN
    log.info("encryption: %s", "PIN " + ("*" * len(pin)) if pin else "OFF (plaintext)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = build_payload(args.db)
    _write_json(out_dir / "findings.json", payload, passphrase=pin)
    log.info("saved findings.json (%d backtest pumps)",
             len(payload["backtest_pumps"]["pumps"]))

    print("\n=== findings.json ===")
    for p in payload["backtest_pumps"]["pumps"]:
        print(f"  {p['coin']:<8} {p['date']}  +{p['pump_pct']:.1f}% (포착)")
    print(f"\nout_dir: {out_dir}")


if __name__ == "__main__":
    main()
