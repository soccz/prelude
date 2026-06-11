"""Policy evolution timeline — 정책/시스템 변경 history.

build_dashboard 가 import 해서 summary.json 에 policy_evolution 을 채운다.
새 정책/fix 추가 시 EVENTS 에 한 줄 추가하고 commit. 날짜·라벨은 사람이 관리,
이벤트 직후의 누적 PnL 등은 paper_ledger 데이터로 자동 계산해서 stale 위험 제거.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import pandas as pd


@dataclass
class PolicyEvent:
    date: str          # YYYY-MM-DD
    label: str         # 짧은 식별자 (e.g. "라이브 시작", "bear silence")
    headline: str      # 한 줄 요약 (e.g. "cum +8% → -52% 감지")
    category: str      # "diagnose" | "fix" | "policy" | "milestone"
    description: Optional[str] = None  # (옵션) 본문에 표시할 긴 설명


# 외부에서 사용 — 여기 한 줄 추가하면 dashboard 자동 반영.
EVENTS: list[PolicyEvent] = [
    PolicyEvent(
        date="2026-05-08",
        label="라이브 paper trading 시작",
        headline="paper ledger 첫 row — 라이브 검증 진입",
        category="milestone",
        description="모델 그대로 두고 실제 시장 데이터로 검증 시작. 백테스트 기대치와 라이브 결과 격차 측정.",
    ),
    PolicyEvent(
        date="2026-05-13",
        label="timer 사고 fix",
        headline="systemd preopen 08:55→08:50 + window 08:45-09:10",
        category="fix",
        description="decision window 진입 못한 사고 (collector 늦어 09:00+ 도달). timer 5분 당기고 window 확장.",
    ),
    PolicyEvent(
        date="2026-05-20",
        label="bear regime silence",
        headline="학습 분포 (bull) 밖에서 알림 강제 묵음",
        category="fix",
        description="라이브 5/8~5/19 동안 알림 89% 가 bear regime 에서 발사. 학습 분포 밖이라 EV 음수 — 강제 침묵.",
    ),
    PolicyEvent(
        date="2026-05-25",
        label="decision policy layer 도입",
        headline="ACTIVE / WATCH_ONLY / SILENCE 3-tier + shadow ledger",
        category="policy",
        description="단순 bear silence → setup_quality + regime 결합 정책. 묵음된 후보는 shadow ledger 로 검증 누적, meta-label 학습 + policy gate (자동 promotion 차단).",
    ),
    PolicyEvent(
        date="2026-05-26",
        label="운영 안전 + 대시보드 polish",
        headline="DB backup + heartbeat + 대시보드 portfolio polish",
        category="milestone",
        description="silent fail 방지 (heartbeat 10:30 KST), DB atomic backup (04:00 KST). 대시보드는 hardcoded narrative 제거, 데이터 기반 자동 렌더로 전환.",
    ),
    PolicyEvent(
        date="2026-06-04",
        label="PUMP hunter SHADOW + policy_competition",
        headline="아이디어 확장 — model × send-policy 25 조합 record-only 감사 + PUMP detector",
        category="policy",
        description="기존 R1/R2/A1/distribution_engine 만으로는 +20% 급등 capture 0%~17% 천장. pump_rule_discovery 의 D-1 leak-free 룰 (roc_7d > 0.85 등) 을 별도 detector 로 배선해 매일 15 watchlist 를 shadow_ledger 에 기록. challenger_only=True 라 Telegram/ACTIVE 승격은 코드 레벨 차단. policy_competition 표가 누적 후 pump_hunter vs 기존 모델을 자동 비교. + publish_dashboard 에 preflight guard (충돌 site repo 차단).",
    ),
    PolicyEvent(
        date="2026-06-11",
        label="exit lab + 🎯 PUMP v2 radar",
        headline="청산 잣대 7개 병렬 기록 + Binance volsurge 융합 radar 텔레그램 (사용자 컨펌)",
        category="policy",
        description="exit lab: 같은 15m 경로에 청산 변형 7개 (noSL 군·비대칭 bracket) 병렬 가상 평가 — 1차 결과 SL 이 하방을 지킴 (noSL 군 전부 악화), tp5_sl2 가 개선 후보. 연구 3종 (cold-start 장중·day-quality gate REJECT, binance lead-lag 는 evaluator 가 가짜 +1.24% 적발 후 15m 재검증). 살아남은 hit 엣지 (roc7+volsurge, 최근 7개월 OOS hit 8.1% ≈ base 6배) 를 🎯 PUMP v2 radar 로 배선 — 자동 net 음수 정직 고지 포함, champion 승격은 차단 유지.",
    ),
]


CATEGORY_COLOR = {
    "milestone": "#16a34a",   # green
    "diagnose":  "#dc2626",   # red
    "fix":       "#b45309",   # amber
    "policy":    "#2563eb",   # blue
}
CATEGORY_GLYPH = {
    "milestone": "✓",
    "diagnose":  "!",
    "fix":       "×",
    "policy":    "↻",
}


def _cum_pnl_since(df: pd.DataFrame, since: str) -> Optional[float]:
    if df is None or df.empty or "date" not in df.columns or "virtual_pnl_pct" not in df.columns:
        return None
    s = df[df["date"] >= since]
    if s.empty:
        return None
    return float(s["virtual_pnl_pct"].sum())


def _window_cum_pnl(df: pd.DataFrame, start: str, end: str) -> Optional[float]:
    if df is None or df.empty or "date" not in df.columns or "virtual_pnl_pct" not in df.columns:
        return None
    s = df[(df["date"] >= start) & (df["date"] <= end)]
    if s.empty:
        return None
    return float(s["virtual_pnl_pct"].sum())


def build_policy_evolution(
    paper_df: pd.DataFrame | None = None,
    preopen_df: pd.DataFrame | None = None,
    asof: Optional[str] = None,
) -> dict:
    """Render 가능한 policy_evolution payload 를 빌드.

    events: 각 event 에 dyn metrics (이벤트 직후 누적 PnL 등) 추가.
    live_window: 라이브 paper 첫 N 일 요약 (hardcoded 음수 PnL 대신).
    """
    asof = asof or datetime.utcnow().strftime("%Y-%m-%d")

    events_out = []
    for ev in EVENTS:
        item = {
            "date": ev.date,
            "label": ev.label,
            "headline": ev.headline,
            "category": ev.category,
            "color": CATEGORY_COLOR.get(ev.category, "#a8a29e"),
            "glyph": CATEGORY_GLYPH.get(ev.category, "•"),
            "description": ev.description or "",
            "is_future": ev.date > asof,
        }
        # 이벤트 직후 누적 PnL (해당 날짜부터 다음 event 직전까지)
        events_out.append(item)

    # 다음 이벤트 직전까지 window 누적 PnL
    for i, item in enumerate(events_out):
        next_date = events_out[i + 1]["date"] if i + 1 < len(events_out) else asof
        item["dist_cum_pnl_pct"] = _window_cum_pnl(paper_df, item["date"], next_date)
        item["preopen_cum_pnl_pct"] = _window_cum_pnl(preopen_df, item["date"], next_date)
        item["window_end"] = next_date

    # 라이브 구간 요약 — milestone event ("라이브 paper trading 시작") 부터.
    # ledger 첫 row 는 백테스트 백필일 수 있어서 EVENTS 의 milestone 기준이 더 정직.
    live_summary = None
    if paper_df is not None and not paper_df.empty and "date" in paper_df.columns:
        live_event = next((e for e in EVENTS if e.category == "milestone"), None)
        live_start = live_event.date if live_event else str(paper_df["date"].min())
        last_date = str(paper_df["date"].max())
        if live_start > last_date:
            live_start = str(paper_df["date"].min())
        live_dist = paper_df[paper_df["date"] >= live_start]
        live_pre = preopen_df[preopen_df["date"] >= live_start] if preopen_df is not None and not preopen_df.empty and "date" in preopen_df.columns else None
        live_summary = {
            "first_date": live_start,
            "last_date": last_date,
            "n_days": (pd.to_datetime(last_date) - pd.to_datetime(live_start)).days + 1,
            "dist_cum_pnl_pct": _cum_pnl_since(paper_df, live_start),
            "preopen_cum_pnl_pct": _cum_pnl_since(preopen_df, live_start) if preopen_df is not None else None,
            "n_alerts_dist": int(len(live_dist)),
            "n_alerts_preopen": int(len(live_pre)) if live_pre is not None else 0,
        }

    return {
        "asof": asof,
        "events": events_out,
        "live_summary": live_summary,
    }
