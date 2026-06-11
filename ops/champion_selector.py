"""champion_selector — slot 별 챔피언 모델을 forward CLOSED ledger 로 자동 선정 (unattended).

완전 자동(방치 운영, 사람 컨펌 게이트 없음 — 사용자 확정). 매 호출 시:
  1. signals.model_registry.MODELS 의 각 모델에 대해 forward CLOSED ledger 행을
     rolling N(=30) 거래일 윈도로 읽어 **하방-우선 점수**를 계산.
  2. 게이트: forward closed n>=MIN_CLOSED + challenger_only=False 인 모델만 챔피언 후보.
     게이트 통과 모델이 없으면 → 기본 챔피언 = 백테스트-최선(R1) 으로 fallback (항상 SHADOW 발송).
  3. 히스테리시스: 현 챔피언 교체는 챌린저가 하방-우선 점수에서 챔피언을 *마진 이상* 그리고
     *연속 K(=5) 거래일* 이겨야 (flip-flop 방지).
  4. slot(preopen/open) 별 챔피언 따로 선정 (해당 slot 가능 모델 중).
  5. output/champion_state.json 작성 + 교체 이력 append + 모든 근거 로그(decision_policy 스타일).

★ 하방-우선 점수 (사용자 메모리: 하락위험 최소화 > 상승):
    1순위 P(realized ≤ DEEP_LOSS=-5%) 빈도 ↓  (깊은 손실 빈도 최소화)
    2순위 net mean (왕복 0.15% 차감) ↑
    3순위 hit rate ↑
  → (deep_loss_freq, -net_mean, -hit) lexicographic. 작을수록 좋은 모델.

★ LEAK / 시간정합성 (셀렉터는 미래를 안 본다):
  - CLOSED 행만 평가 = 이미 day-D 경로가 끝나 실현된 **과거**. 진행 중(open) 행은 제외.
  - realized 는 ledger 가 이미 채운 값(close_recommend_ledger / paper_ledger 파이프).
    셀렉터는 그 과거 실현만 집계 — 미래 데이터 접근 0.
  - rolling 윈도는 ledger date_col 내림차순 최근 N 거래일 (asof 미래는 ledger 에 없음).
  - cost_already_deducted=False 인 ledger(gross)는 셀렉터가 왕복 0.15% 차감 후 평가.

self-contained: gan_t/xsec_alpha/fin import 0 (signals.model_registry 만 import).

사용:
    python -m ops.champion_selector                 # champion_state.json 생성/갱신
    python -m ops.champion_selector --asof 2026-06-01 --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ledger.config import ROUND_TRIP_COST_PP  # noqa: E402
from signals.model_registry import (  # noqa: E402
    MODELS, ModelSpec, all_slots, models_for_slot, fallback_model,
)

log = logging.getLogger("champion_selector")

# --------------------------------------------------------------------------
# 튜닝 가능 상수 (placeholder — CLAUDE.md §2.5). 데이터 누적 후 조정.
# --------------------------------------------------------------------------
STATE_PATH = _ROOT / "output" / "champion_state.json"
ROLLING_N = 30               # rolling 윈도 = 최근 30 거래일(추천일 기준) CLOSED 행
MIN_CLOSED = 30              # 게이트: forward closed n>=30 만 챔피언 후보
DEEP_LOSS = -5.0             # 깊은 손실 임계 (%, realized <= -5% = 사용자 손실 수용 anchor)
ROUND_TRIP_COST_PCT = ROUND_TRIP_COST_PP   # 왕복 비용 (%p) — ledger/config.py 단일 출처

# 히스테리시스 (flip-flop 방지): 챌린저가 챔피언을 아래 마진 이상 + 연속 K일 이겨야 교체.
HYST_K = 5                   # 연속 K 거래일 우위
MARGIN_DEEP_LOSS_PP = 5.0    # P(≤-5%) 를 5%p 이상 낮추거나
MARGIN_NET_PP = 0.3          # net mean 을 0.3%p 이상 높여야 (둘 중 하나면 충분)


# ==========================================================================
# 1. 모델별 하방-우선 메트릭 (forward CLOSED, rolling N).
# ==========================================================================
@dataclass
class ModelMetric:
    model_id: str
    n_closed: int                 # rolling 윈도 내 CLOSED 행 수 (메트릭 표본)
    n_days: int                   # rolling 윈도 내 고유 거래일 수 (= 독립 forward 관측)
    deep_loss_freq: float         # P(realized <= -5%) — 1순위(↓)
    net_mean_pct: float           # net 평균 수익률 % — 2순위(↑)
    hit_rate: float               # 적중률 (없으면 nan) — 3순위(↑)
    gate_pass: bool               # n_days>=MIN_CLOSED + not challenger_only
    last_date: str | None         # 가장 최근 추천일 (윈도 끝)
    reason: str                   # 게이트/스킵 사유

    def score_key(self):
        """lexicographic 정렬키 (작을수록 좋음): deep_loss↓, net↑, hit↑."""
        hit = self.hit_rate if np.isfinite(self.hit_rate) else 0.0
        return (self.deep_loss_freq, -self.net_mean_pct, -hit)


def _load_closed(spec: ModelSpec, asof: pd.Timestamp) -> pd.DataFrame:
    """모델 ledger 에서 CLOSED + date <= asof + 최근 ROLLING_N 거래일 행 반환.
    ★ CLOSED 만 = 이미 실현된 과거 (미래 안 봄). date <= asof 로 미래 행도 제외(이중 안전)."""
    path = spec.abs_ledger_path()
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    m = spec.metric
    needed = [m.status_col, m.date_col, m.realized_pct_col]
    if any(c not in df.columns for c in needed):
        return pd.DataFrame()
    df = df[df[m.status_col].astype(str) == m.closed_value].copy()
    if df.empty:
        return df
    df[m.date_col] = pd.to_datetime(df[m.date_col], errors="coerce")
    df = df[df[m.date_col].notna() & (df[m.date_col].dt.normalize() <= asof)]
    if df.empty:
        return df
    # 최근 ROLLING_N 거래일(고유 date)만.
    recent_dates = sorted(df[m.date_col].dt.normalize().unique())[-ROLLING_N:]
    df = df[df[m.date_col].dt.normalize().isin(recent_dates)]
    return df


def compute_metric(spec: ModelSpec, asof: pd.Timestamp) -> ModelMetric:
    df = _load_closed(spec, asof)
    m = spec.metric
    if df.empty:
        return ModelMetric(spec.id, 0, 0, np.nan, np.nan, np.nan, False, None,
                           "no closed rows (forward 표본 0)")
    valid = df[pd.to_numeric(df[m.realized_pct_col], errors="coerce").notna()].copy()
    realized = pd.to_numeric(valid[m.realized_pct_col], errors="coerce")
    n = len(realized)
    if n == 0:
        return ModelMetric(spec.id, 0, 0, np.nan, np.nan, np.nan, False, None,
                           "closed rows 있으나 realized 전부 NaN")
    # gross ledger 면 왕복 비용 차감 후 net 으로 평가 (위생: 비용 항상 차감).
    if not m.cost_already_deducted:
        realized = realized - ROUND_TRIP_COST_PCT
    # ★ 게이트는 고유 거래일 수(n_days) 로 — 같은 날 다중 픽은 상관(같은 BTC regime)이라
    #   행 수로 세면 backfill 한 블록이 게이트를 통과시킨다. 독립 forward 관측 = 거래일.
    n_days = int(valid[m.date_col].dt.normalize().nunique())
    deep_loss_freq = float((realized <= DEEP_LOSS).mean())
    net_mean = float(realized.mean())
    hit = np.nan
    if m.hit_col and m.hit_col in valid.columns:
        h = pd.to_numeric(valid[m.hit_col], errors="coerce").dropna()
        hit = float(h.mean()) if len(h) else np.nan
    last_date = str(valid[m.date_col].max().date())
    gate = (n_days >= MIN_CLOSED) and (not spec.challenger_only)
    if spec.challenger_only:
        reason = f"challenger_only (승격 금지). n_days={n_days} (rows={n})"
    elif n_days < MIN_CLOSED:
        reason = f"게이트 미달: n_days={n_days} < {MIN_CLOSED} (rows={n})"
    else:
        reason = f"게이트 통과: n_days={n_days} >= {MIN_CLOSED} (rows={n})"
    return ModelMetric(spec.id, n, n_days, deep_loss_freq, net_mean, hit, gate,
                       last_date, reason)


# ==========================================================================
# 2. 히스테리시스 — 현 챔피언 교체 판정 (연속 K일 + 마진).
# ==========================================================================
def _beats_by_margin(challenger: ModelMetric, champ: ModelMetric) -> bool:
    """챌린저가 챔피언을 마진 이상 이기는가 (P(≤-5%) 5%p↓ 또는 net 0.3%p↑)."""
    dl_better = (champ.deep_loss_freq - challenger.deep_loss_freq) * 100.0 >= MARGIN_DEEP_LOSS_PP
    net_better = (challenger.net_mean_pct - champ.net_mean_pct) >= MARGIN_NET_PP
    return bool(dl_better or net_better)


def _update_streak(state: dict, slot: str, current_champ: str | None,
                   best_challenger_id: str | None, beats: bool,
                   challenger_last_date: str | None = None) -> int:
    """slot 의 연속-우위 streak 업데이트.

    ★ '거래일' 기준 (flip-flop·수동재실행 방어): 같은 챌린저가 마진 이상 이기되,
       **새 거래일(last_date 전진)이 추가됐을 때만** +1. 같은 last_date 로 재실행하거나
       주말/수집실패로 새 CLOSED 거래일이 없는 날엔 streak 증가 X (stale 관측으로
       K일 약속을 채우지 못하게). 챌린저가 바뀌거나 못 이기면 리셋.
    state['streaks'][slot] = {'challenger_id':..., 'count':int, 'last_date':str|None}."""
    streaks = state.setdefault("streaks", {})
    cur = streaks.get(slot, {"challenger_id": None, "count": 0, "last_date": None})
    if best_challenger_id is not None and beats and best_challenger_id != current_champ:
        if cur.get("challenger_id") == best_challenger_id:
            # 같은 챌린저: 새 거래일(ISO date 문자열은 사전식==시간순)일 때만 카운트 증가.
            prev_ld = cur.get("last_date")
            advanced = challenger_last_date is not None and (
                prev_ld is None or str(challenger_last_date) > str(prev_ld))
            cur = {
                "challenger_id": best_challenger_id,
                "count": int(cur["count"]) + (1 if advanced else 0),
                "last_date": str(challenger_last_date) if advanced else prev_ld,
            }
        else:
            # 새 챌린저 → 첫 거래일 관측.
            cur = {"challenger_id": best_challenger_id, "count": 1,
                   "last_date": str(challenger_last_date) if challenger_last_date else None}
    else:
        cur = {"challenger_id": None, "count": 0, "last_date": None}
    streaks[slot] = cur
    return cur["count"]


# ==========================================================================
# 3. slot 챔피언 선정.
# ==========================================================================
def select_for_slot(slot: str, metrics: dict[str, ModelMetric], state: dict,
                    asof: pd.Timestamp) -> dict:
    """해당 slot 의 챔피언을 선정하고 근거 dict 반환 (champion_state.json 의 slot entry)."""
    specs = models_for_slot(slot)
    cand_metrics = [metrics[s.id] for s in specs if s.id in metrics]
    gate_passers = [mm for mm in cand_metrics if mm.gate_pass]

    prev = state.get("slots", {}).get(slot, {})
    prev_champ = prev.get("champion_id")

    # --- 게이트 통과 모델 없음 → 백테스트-최선 fallback (항상 SHADOW 발송) ---
    if not gate_passers:
        fb = fallback_model(slot)
        fb_id = fb.id if fb else None
        mm = metrics.get(fb_id) if fb_id else None
        reason = (f"게이트 통과 모델 없음 (forward closed n<{MIN_CLOSED} 또는 challenger_only) "
                  f"→ 기본 챔피언=백테스트-최선({fb_id}), SHADOW 발송")
        # streak 리셋 (교체 후보 없음).
        _update_streak(state, slot, prev_champ, None, False)
        return _slot_entry(slot, fb_id, prev, mm, reason, asof, fallback=True)

    # --- 게이트 통과자 중 하방-우선 점수 최선 = 챌린저 후보 ---
    gate_passers.sort(key=lambda mm: mm.score_key())
    best = gate_passers[0]

    # 현 챔피언이 여전히 게이트 통과 & 후보면 그 metric, 아니면 None.
    champ_mm = metrics.get(prev_champ) if prev_champ else None
    champ_in_slot = prev_champ in {s.id for s in specs} if prev_champ else False
    champ_eligible = bool(champ_mm and champ_mm.gate_pass and champ_in_slot)

    if not champ_eligible:
        # 현 챔피언이 없거나 더 이상 자격 없음 → 즉시 최선으로 (히스테리시스 면제: 초기 설정).
        _update_streak(state, slot, prev_champ, None, False)
        reason = (f"챔피언 미설정/자격상실 → 게이트 통과 최선({best.model_id}) 즉시 선정 "
                  f"[deep_loss={best.deep_loss_freq:.3f} net={best.net_mean_pct:+.2f}% n={best.n_closed}]")
        return _slot_entry(slot, best.model_id, prev, best, reason, asof, fallback=False)

    # 현 챔피언 자격 유지 → 히스테리시스로 교체 판정.
    # ★ best.last_date 를 넘겨 streak 이 '거래일' 단위로만 증가하게 (stale 재실행 방어).
    beats = _beats_by_margin(best, champ_mm) and best.model_id != prev_champ
    streak = _update_streak(state, slot, prev_champ, best.model_id, beats,
                            challenger_last_date=best.last_date)

    if beats and streak >= HYST_K:
        reason = (f"교체: 챌린저({best.model_id}) 가 챔피언({prev_champ}) 을 마진 이상 + "
                  f"연속 {streak}>= {HYST_K}일 우위 "
                  f"[Δdeep_loss={(champ_mm.deep_loss_freq-best.deep_loss_freq)*100:+.1f}pp "
                  f"Δnet={best.net_mean_pct-champ_mm.net_mean_pct:+.2f}pp]")
        # 교체 확정 → streak 리셋.
        state.setdefault("streaks", {})[slot] = {"challenger_id": None, "count": 0}
        return _slot_entry(slot, best.model_id, prev, best, reason, asof, fallback=False)

    # 챔피언 유지.
    if best.model_id == prev_champ:
        reason = (f"유지: 현 챔피언({prev_champ}) 이 여전히 하방-우선 최선 "
                  f"[deep_loss={champ_mm.deep_loss_freq:.3f} net={champ_mm.net_mean_pct:+.2f}% n={champ_mm.n_closed}]")
    elif beats:
        reason = (f"유지(히스테리시스): 챌린저({best.model_id}) 우위지만 연속 {streak}/{HYST_K}일 "
                  f"미충족 → flip-flop 방지")
    else:
        reason = (f"유지: 챌린저({best.model_id}) 가 마진(P≤-5% {MARGIN_DEEP_LOSS_PP}pp↓ "
                  f"또는 net {MARGIN_NET_PP}pp↑) 미충족")
    return _slot_entry(slot, prev_champ, prev, champ_mm, reason, asof, fallback=False)


def _slot_entry(slot: str, champ_id: str | None, prev: dict, mm: ModelMetric | None,
                reason: str, asof: pd.Timestamp, fallback: bool) -> dict:
    prev_champ = prev.get("champion_id")
    since = prev.get("since") if (prev_champ == champ_id and prev.get("since")) else str(asof.date())
    metric = None
    if mm is not None:
        metric = {
            "n_closed": mm.n_closed,
            "n_days": mm.n_days,
            "deep_loss_freq": None if np.isnan(mm.deep_loss_freq) else round(mm.deep_loss_freq, 4),
            "net_mean_pct": None if np.isnan(mm.net_mean_pct) else round(mm.net_mean_pct, 4),
            "hit_rate": None if (mm.hit_rate is None or np.isnan(mm.hit_rate)) else round(mm.hit_rate, 4),
            "last_date": mm.last_date,
        }
    return {
        "champion_id": champ_id,
        "since": since,
        "is_fallback": fallback,
        "metric": metric,
        "reason": reason,
    }


# ==========================================================================
# 4. 전체 실행.
# ==========================================================================
def run(asof: pd.Timestamp, dry_run: bool) -> dict:
    log.info("champion_selector asof=%s (rolling N=%d, gate n>=%d, hyst K=%d)",
             asof.date(), ROLLING_N, MIN_CLOSED, HYST_K)
    state = _load_state()

    # 모든 모델 metric 계산 (로그 = 감사추적).
    metrics: dict[str, ModelMetric] = {}
    for spec in MODELS:
        mm = compute_metric(spec, asof)
        metrics[spec.id] = mm
        log.info("  [%-26s] rows=%-3d days=%-3d deep_loss=%s net=%s hit=%s gate=%s | %s",
                 spec.id, mm.n_closed, mm.n_days,
                 "  nan" if np.isnan(mm.deep_loss_freq) else f"{mm.deep_loss_freq:.3f}",
                 "   nan" if np.isnan(mm.net_mean_pct) else f"{mm.net_mean_pct:+.2f}%",
                 " nan" if (mm.hit_rate is None or np.isnan(mm.hit_rate)) else f"{mm.hit_rate:.2f}",
                 mm.gate_pass, mm.reason)

    # slot 별 챔피언 선정.
    slot_entries = {}
    history_events = []
    for slot in all_slots():
        prev_champ = state.get("slots", {}).get(slot, {}).get("champion_id")
        entry = select_for_slot(slot, metrics, state, asof)
        slot_entries[slot] = entry
        log.info("  >> slot=%-7s champion=%-26s fallback=%s | %s",
                 slot, str(entry["champion_id"]), entry["is_fallback"], entry["reason"])
        if prev_champ != entry["champion_id"]:
            history_events.append({
                "asof": str(asof.date()), "slot": slot,
                "from": prev_champ, "to": entry["champion_id"],
                "reason": entry["reason"],
            })

    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new_state = {
        "asof": str(asof.date()),
        "updated_at": now_iso,
        "config": {
            "rolling_n": ROLLING_N, "min_closed": MIN_CLOSED, "deep_loss_pct": DEEP_LOSS,
            "hyst_k": HYST_K, "margin_deep_loss_pp": MARGIN_DEEP_LOSS_PP,
            "margin_net_pp": MARGIN_NET_PP, "round_trip_cost_pct": ROUND_TRIP_COST_PCT,
        },
        "slots": slot_entries,
        "streaks": state.get("streaks", {}),
        "history": state.get("history", []) + history_events,
    }

    if dry_run:
        log.info("dry-run: champion_state.json 저장 안 함")
    else:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_PATH, "w") as f:
            json.dump(new_state, f, indent=2, ensure_ascii=False)
        log.info("saved %s", STATE_PATH)
    return new_state


def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            with open(STATE_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            log.warning("champion_state.json 손상 → 초기화")
    return {"slots": {}, "streaks": {}, "history": []}


# ==========================================================================
# ops-steward 가 읽을 헬퍼 (champion_state 읽는 표준 인터페이스).
# ==========================================================================
def get_champion(slot: str, state_path: Path | str = STATE_PATH) -> dict | None:
    """slot 의 현 챔피언 entry 를 반환 (없으면 None).
    반환 = {'champion_id','since','is_fallback','metric','reason'}.
    ops-steward send dispatcher 가 champion_id → model_registry.get_model 로 predict_ref 획득."""
    p = Path(state_path)
    if not p.exists():
        return None
    with open(p) as f:
        st = json.load(f)
    return st.get("slots", {}).get(slot)


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)])
    ap = argparse.ArgumentParser(description="champion_selector (unattended, downside-first)")
    ap.add_argument("--asof", type=str, default=None, help="YYYY-MM-DD (default=today)")
    ap.add_argument("--dry-run", action="store_true", help="champion_state.json 저장 X")
    args = ap.parse_args()
    asof = pd.Timestamp(args.asof).normalize() if args.asof else pd.Timestamp.now().normalize()
    state = run(asof, args.dry_run)
    print(json.dumps(state, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
