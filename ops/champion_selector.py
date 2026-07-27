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
    1순위 P(완결 no-SL path 저점 ≤ DEEP_LOSS=-5%) 빈도 ↓
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
import copy
import json
import logging
import stat
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ledger.config import ROUND_TRIP_COST_PP  # noqa: E402
from ops.artifact_provenance import (  # noqa: E402
    ArtifactValidationError,
    ArtifactSourceChangedError,
    atomic_write_bytes,
    atomic_write_json,
    file_identity,
    file_set_identity,
    payload_digest,
    sha256_bytes,
    strict_json_object_bytes,
)
from ops.file_lock import FileLockError, file_lock  # noqa: E402
from signals.model_registry import (  # noqa: E402
    MODELS,
    ModelSpec,
    all_slots,
    fallback_model,
    get_model,
    models_for_slot,
)

log = logging.getLogger("champion_selector")

# --------------------------------------------------------------------------
# 튜닝 가능 상수 (placeholder — CLAUDE.md §2.5). 데이터 누적 후 조정.
# --------------------------------------------------------------------------
STATE_PATH = _ROOT / "output" / "champion_state.json"
STATE_SCHEMA_VERSION = "champion_state.v1"
MAX_STATE_AGE_DAYS = 1
ROLLING_N = 30               # rolling 윈도 = 최근 30 거래일(추천일 기준) CLOSED 행
MIN_CLOSED = 30              # 게이트: forward closed n>=30 만 챔피언 후보
MIN_NET_MEAN_PCT = 0.0       # 검증 챔피언은 비용 차감 forward 평균이 반드시 양수
DEEP_LOSS = -5.0             # 깊은 손실 임계 (%, realized <= -5% = 사용자 손실 수용 anchor)
ROUND_TRIP_COST_PCT = ROUND_TRIP_COST_PP   # 왕복 비용 (%p) — ledger/config.py 단일 출처

# 히스테리시스 (flip-flop 방지): 챌린저가 챔피언을 아래 마진 이상 + 연속 K일 이겨야 교체.
HYST_K = 5                   # 연속 K 거래일 우위
MARGIN_DEEP_LOSS_PP = 5.0    # P(≤-5%) 를 5%p 이상 낮추거나
MARGIN_NET_PP = 0.3          # net mean 을 0.3%p 이상 높여야 (둘 중 하나면 충분)


class ChampionStateError(RuntimeError):
    """An existing state cannot be authenticated or safely migrated."""


@dataclass(frozen=True)
class LoadedChampionState:
    state: dict
    legacy_bytes: bytes | None = None


@dataclass(frozen=True)
class ValidatedChampionStateArtifact:
    """One checksummed state payload and the exact bytes it was decoded from."""

    payload: dict
    identity: dict


# ==========================================================================
# 1. 모델별 하방-우선 메트릭 (forward CLOSED, rolling N).
# ==========================================================================
@dataclass
class ModelMetric:
    model_id: str
    n_closed: int                 # rolling 윈도 내 CLOSED 행 수 (메트릭 표본)
    n_days: int                   # rolling 윈도 내 고유 거래일 수 (= 독립 forward 관측)
    n_downside: int               # 완결된 no-SL path 저점 표본 수
    n_downside_days: int          # 완결된 no-SL path 고유 거래일 수
    deep_loss_freq: float         # P(complete no-SL path min <= -5%) — 1순위(↓)
    net_mean_pct: float           # net 평균 수익률 % — 2순위(↑)
    hit_rate: float               # 적중률 (없으면 nan) — 3순위(↑)
    gate_pass: bool               # 표본/경로 충분 + 비용차감 net>0 + 승격 가능
    last_date: str | None         # 가장 최근 추천일 (윈도 끝)
    reason: str                   # 게이트/스킵 사유

    def score_key(self):
        """lexicographic 정렬키 (작을수록 좋음): deep_loss↓, net↑, hit↑."""
        hit = self.hit_rate if np.isfinite(self.hit_rate) else 0.0
        return (self.deep_loss_freq, -self.net_mean_pct, -hit)


def _load_closed(spec: ModelSpec, asof: pd.Timestamp) -> pd.DataFrame:
    """모델 ledger 에서 CLOSED + date < asof + 최근 ROLLING_N 거래일 행 반환.

    추천일 D의 경로는 다음 KST open 뒤에야 완결된다. 따라서 CLOSED라고
    표시됐더라도 asof 당일 행은 그 asof 의 선택 근거가 될 수 없다.
    """
    path = spec.abs_ledger_path()
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    m = spec.metric
    needed = [m.status_col, m.date_col, m.realized_pct_col]
    if any(c not in df.columns for c in needed):
        return pd.DataFrame()
    cutoff = pd.Timestamp(asof).normalize()
    df = df[df[m.status_col].astype(str) == m.closed_value].copy()
    if df.empty:
        return df
    df[m.date_col] = pd.to_datetime(df[m.date_col], errors="coerce")
    df = df[
        df[m.date_col].notna()
        & (df[m.date_col].dt.normalize() < cutoff)
    ]
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
        return ModelMetric(
            spec.id,
            0,
            0,
            0,
            0,
            np.nan,
            np.nan,
            np.nan,
            False,
            None,
            "no closed rows (forward 표본 0)",
        )
    realized_numeric = pd.to_numeric(
        df[m.realized_pct_col],
        errors="coerce",
    )
    realized_finite = realized_numeric.notna() & np.isfinite(realized_numeric)
    valid = df[realized_finite].copy()
    realized = realized_numeric.loc[valid.index].astype(float)
    n = len(realized)
    if n == 0:
        return ModelMetric(
            spec.id,
            0,
            0,
            0,
            0,
            np.nan,
            np.nan,
            np.nan,
            False,
            None,
            "closed rows 있으나 realized 전부 NaN",
        )
    # gross ledger 면 왕복 비용 차감 후 net 으로 평가 (위생: 비용 항상 차감).
    if not m.cost_already_deducted:
        realized = realized - ROUND_TRIP_COST_PCT
    # ★ 게이트는 고유 거래일 수(n_days) 로 — 같은 날 다중 픽은 상관(같은 BTC regime)이라
    #   행 수로 세면 backfill 한 블록이 게이트를 통과시킨다. 독립 forward 관측 = 거래일.
    n_days = int(valid[m.date_col].dt.normalize().nunique())
    downside = pd.Series(dtype=float)
    downside_valid = valid.iloc[0:0].copy()
    if (
        m.downside_pct_col
        and m.path_complete_col
        and m.downside_pct_col in valid.columns
        and m.path_complete_col in valid.columns
    ):
        complete = valid[m.path_complete_col].map(
            lambda value: (
                bool(value)
                if isinstance(value, (bool, np.bool_))
                else str(value).strip().lower() in {"true", "1"}
            )
        )
        downside_numeric = pd.to_numeric(
            valid[m.downside_pct_col],
            errors="coerce",
        )
        downside_valid = valid[
            complete
            & downside_numeric.notna()
            & np.isfinite(downside_numeric)
        ].copy()
        downside = downside_numeric.loc[downside_valid.index]
    n_downside = int(len(downside))
    n_downside_days = int(
        downside_valid[m.date_col].dt.normalize().nunique()
    )
    deep_loss_freq = (
        float((downside <= DEEP_LOSS).mean())
        if n_downside
        else np.nan
    )
    net_mean = float(realized.mean())
    hit = np.nan
    if m.hit_col and m.hit_col in valid.columns:
        hit_numeric = pd.to_numeric(valid[m.hit_col], errors="coerce")
        h = hit_numeric[
            hit_numeric.notna()
            & np.isfinite(hit_numeric)
            & hit_numeric.isin({0, 1})
        ]
        hit = float(h.mean()) if len(h) else np.nan
    last_date = str(valid[m.date_col].max().date())
    gate = (
        n_days >= MIN_CLOSED
        and n_downside_days >= MIN_CLOSED
        and net_mean > MIN_NET_MEAN_PCT
        and not spec.challenger_only
    )
    if spec.challenger_only:
        reason = (
            f"challenger_only (승격 금지). net_days={n_days}, "
            f"path_days={n_downside_days} (rows={n})"
        )
    elif n_days < MIN_CLOSED or n_downside_days < MIN_CLOSED:
        reason = (
            f"게이트 미달: net_days={n_days}, "
            f"complete_path_days={n_downside_days}; 둘 다 >= {MIN_CLOSED} 필요 "
            f"(rows={n}, path_rows={n_downside})"
        )
    elif net_mean <= MIN_NET_MEAN_PCT:
        reason = (
            f"게이트 미달: 비용차감 forward net_mean={net_mean:+.4f}% "
            f"<= {MIN_NET_MEAN_PCT:+.4f}% (양수 절대 결과 필요; "
            f"rows={n}, days={n_days})"
        )
    else:
        reason = (
            f"게이트 통과: net_days={n_days}, "
            f"complete_path_days={n_downside_days} >= {MIN_CLOSED}, "
            f"net_mean={net_mean:+.4f}% > {MIN_NET_MEAN_PCT:+.4f}% "
            f"(rows={n}, path_rows={n_downside})"
        )
    return ModelMetric(
        spec.id,
        n,
        n_days,
        n_downside,
        n_downside_days,
        deep_loss_freq,
        net_mean,
        hit,
        gate,
        last_date,
        reason,
    )


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
        reason = (
            "게이트 통과 모델 없음 (표본/완결경로 부족, 비용차감 "
            f"net_mean<={MIN_NET_MEAN_PCT:+.4f}%, 또는 challenger_only) "
            f"→ 기본 챔피언=백테스트-최선({fb_id}), SHADOW 발송"
        )
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
    if champ_mm is None:
        raise ChampionStateError(
            "champion eligibility invariant lost its metric"
        )
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
            "n_downside": mm.n_downside,
            "n_downside_days": mm.n_downside_days,
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
def _state_digest(state: dict) -> str:
    return payload_digest(state)


def _expected_config() -> dict:
    return {
        "rolling_n": ROLLING_N,
        "min_closed": MIN_CLOSED,
        "min_net_mean_pct": MIN_NET_MEAN_PCT,
        "net_gate_basis": "round_trip_cost_deducted_forward_mean_gt_threshold",
        "deep_loss_pct": DEEP_LOSS,
        "deep_loss_basis": "complete_no_sl_path_min",
        "hyst_k": HYST_K,
        "margin_deep_loss_pp": MARGIN_DEEP_LOSS_PP,
        "margin_net_pp": MARGIN_NET_PP,
        "round_trip_cost_pct": ROUND_TRIP_COST_PCT,
    }


def _legacy_config_without_path_or_net_gate() -> dict:
    """The one checksummed state contract eligible for automatic migration."""
    return {
        "rolling_n": ROLLING_N,
        "min_closed": MIN_CLOSED,
        "deep_loss_pct": DEEP_LOSS,
        "hyst_k": HYST_K,
        "margin_deep_loss_pp": MARGIN_DEEP_LOSS_PP,
        "margin_net_pp": MARGIN_NET_PP,
        "round_trip_cost_pct": ROUND_TRIP_COST_PCT,
    }


def _validate_slot_entry_shape(
    slot: str,
    entry: object,
    *,
    state_asof: pd.Timestamp,
    allow_fallback_slot_rewrite: bool = False,
) -> tuple[dict, dict | None]:
    if not isinstance(entry, dict):
        raise ValueError(f"champion state slot entry invalid: {slot}")
    champion_id = entry.get("champion_id")
    if not isinstance(champion_id, str) or not champion_id:
        raise ValueError(f"champion state champion_id invalid: {slot}")
    try:
        spec = get_model(champion_id)
    except KeyError as exc:
        raise ValueError(
            f"champion state model is not registered: {champion_id}"
        ) from exc
    is_fallback = entry.get("is_fallback")
    if not isinstance(is_fallback, bool):
        raise ValueError(f"champion state fallback flag invalid: {slot}")
    rewrite = None
    if slot not in spec.slots:
        fallback = fallback_model(slot)
        if (
            not allow_fallback_slot_rewrite
            or not is_fallback
            or fallback is None
        ):
            raise ValueError(
                f"champion state model/slot mismatch: {champion_id}/{slot}"
            )
        rewrite = {
            "slot": slot,
            "from": champion_id,
            "to": fallback.id,
            "reason": "legacy cross-slot fallback identity corrected",
        }
    if not isinstance(entry.get("reason"), str):
        raise ValueError(f"champion state reason invalid: {slot}")
    try:
        since = pd.Timestamp(entry["since"]).normalize()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"champion state since invalid: {slot}"
        ) from exc
    if since > state_asof:
        raise ValueError(f"champion state since is after asof: {slot}")
    if entry.get("metric") is not None and not isinstance(
        entry.get("metric"), dict
    ):
        raise ValueError(f"champion state metric invalid: {slot}")
    return entry, rewrite


def _validate_state(
    state: object,
    *,
    expected_asof: str | pd.Timestamp | None = None,
) -> dict:
    if not isinstance(state, dict):
        raise ValueError("champion state must be an object")
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ValueError("champion state schema mismatch")
    checksum = state.get("payload_sha256")
    if (
        not isinstance(checksum, str)
        or len(checksum) != 64
        or checksum != _state_digest(state)
    ):
        raise ValueError("champion state checksum mismatch")
    try:
        state_asof = pd.Timestamp(state["asof"]).normalize()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("champion state asof invalid") from exc
    if expected_asof is not None:
        expected = pd.Timestamp(expected_asof).normalize()
        age_days = int((expected - state_asof).days)
        if age_days < 0 or age_days > MAX_STATE_AGE_DAYS:
            raise ValueError(
                f"champion state asof outside allowed decision lag: "
                f"state={state_asof.date()} decision={expected.date()} "
                f"age_days={age_days}"
            )
    if state.get("config") != _expected_config():
        raise ValueError("champion state config mismatch")
    if not isinstance(state.get("streaks"), dict) or not isinstance(
        state.get("history"), list
    ):
        raise ValueError("champion state history/streaks invalid")
    slots = state.get("slots")
    if not isinstance(slots, dict) or set(slots) != set(all_slots()):
        raise ValueError("champion state slots incomplete")
    for slot, entry in slots.items():
        _validate_slot_entry_shape(
            slot,
            entry,
            state_asof=state_asof,
        )
    migrations = state.get("migrations", [])
    if not isinstance(migrations, list) or any(
        not isinstance(item, dict) for item in migrations
    ):
        raise ValueError("champion state migrations invalid")
    input_manifest = state.get("input_manifest")
    if input_manifest is not None:
        if (
            not isinstance(input_manifest, dict)
            or input_manifest.get("schema") != "champion_selector_input.v1"
            or input_manifest.get("asof") != str(state_asof.date())
            or not isinstance(input_manifest.get("files"), dict)
        ):
            raise ValueError("champion state input manifest invalid")
    return state


def _migrate_legacy_state(
    state: object,
    *,
    target_asof: pd.Timestamp,
) -> dict:
    """Migrate only the authenticated pre-path/pre-net-gate v1 contract.

    Metrics and hysteresis streaks were computed under a different objective,
    so they are cleared. Historical champion-change events remain byte-for-byte
    represented in ``history`` and migration details live in a separate list;
    this avoids emitting a false operational champion-change notice.
    """
    if not isinstance(state, dict):
        raise ValueError("legacy champion state must be an object")
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ValueError("legacy champion state schema mismatch")
    checksum = state.get("payload_sha256")
    if (
        not isinstance(checksum, str)
        or len(checksum) != 64
        or checksum != _state_digest(state)
    ):
        raise ValueError("legacy champion state checksum mismatch")
    if state.get("config") != _legacy_config_without_path_or_net_gate():
        raise ValueError("legacy champion state contract is not migratable")
    try:
        legacy_asof = pd.Timestamp(state["asof"]).normalize()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("legacy champion state asof invalid") from exc
    target = pd.Timestamp(target_asof).normalize()
    if legacy_asof > target:
        raise ValueError("legacy champion state is future-dated")
    if not isinstance(state.get("streaks"), dict) or not isinstance(
        state.get("history"), list
    ):
        raise ValueError("legacy champion state history/streaks invalid")
    slots = state.get("slots")
    if not isinstance(slots, dict) or set(slots) != set(all_slots()):
        raise ValueError("legacy champion state slots incomplete")

    migrated_slots: dict[str, dict] = {}
    rewrites = []
    for slot, raw_entry in slots.items():
        entry, rewrite = _validate_slot_entry_shape(
            slot,
            raw_entry,
            state_asof=legacy_asof,
            allow_fallback_slot_rewrite=True,
        )
        migrated_entry = copy.deepcopy(entry)
        migrated_entry["metric"] = None
        if rewrite is not None:
            migrated_entry["champion_id"] = rewrite["to"]
            migrated_entry["since"] = str(target.date())
            migrated_entry["is_fallback"] = True
            migrated_entry["reason"] = rewrite["reason"]
            rewrites.append(rewrite)
        migrated_slots[slot] = migrated_entry

    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    migrations = copy.deepcopy(state.get("migrations", []))
    migrations.append({
        "schema": "champion_state_migration.v1",
        "migrated_at": now_iso,
        "asof": str(target.date()),
        "legacy_asof": str(legacy_asof.date()),
        "from_payload_sha256": checksum,
        "reason": (
            "adopt complete no-SL path downside basis and positive "
            "cost-deducted forward-net absolute gate"
        ),
        "history_entries_preserved": len(state["history"]),
        "slot_rewrites": rewrites,
    })
    migrated = {
        "schema_version": STATE_SCHEMA_VERSION,
        "asof": str(target.date()),
        "updated_at": now_iso,
        "config": _expected_config(),
        "slots": migrated_slots,
        # Old streaks compared models under an obsolete objective.
        "streaks": {},
        "history": copy.deepcopy(state["history"]),
        "migrations": migrations,
    }
    migrated["payload_sha256"] = _state_digest(migrated)
    return _validate_state(migrated)


@contextmanager
def _state_lock(path: Path | None = None, *, shared: bool = False):
    path = path or STATE_PATH
    lock_path = path.with_name(f".{path.name}.lock")
    try:
        with file_lock(lock_path, shared=shared):
            yield
    except FileLockError as exc:
        raise ChampionStateError(
            f"champion state lock cannot be opened safely: {lock_path}"
        ) from exc


def _atomic_write_state(path: Path, state: dict) -> None:
    atomic_write_json(path, state)


def _preserve_legacy_state(path: Path, raw: bytes) -> Path:
    """Create a content-addressed, never-overwritten copy before migration."""
    digest = sha256_bytes(raw)
    backup = path.with_name(
        f"{path.stem}.legacy.{digest}{path.suffix}"
    )
    try:
        existing = backup.lstat()
    except FileNotFoundError:
        existing = None
    except OSError as exc:
        raise ChampionStateError(
            f"legacy champion backup cannot be inspected: {backup}"
        ) from exc
    if existing is not None:
        if (
            not stat.S_ISREG(existing.st_mode)
            or existing.st_nlink != 1
        ):
            raise ChampionStateError(
                f"legacy champion backup is unsafe: {backup}"
            )
        if _read_stable_state_bytes(backup) != raw:
            raise ChampionStateError(
                f"legacy champion backup hash collision: {backup}"
            )
        return backup

    try:
        atomic_write_bytes(backup, raw)
        created = backup.lstat()
        if (
            not stat.S_ISREG(created.st_mode)
            or created.st_nlink != 1
            or _read_stable_state_bytes(backup) != raw
        ):
            raise ChampionStateError(
                f"legacy champion backup publication failed: {backup}"
            )
    except ChampionStateError:
        raise
    except (OSError, ArtifactSourceChangedError) as exc:
        raise ChampionStateError(
            f"legacy champion backup cannot be created safely: {backup}"
        ) from exc
    return backup


def run(asof: pd.Timestamp, dry_run: bool) -> dict:
    """Serialize the state read/decision/write transaction across processes."""
    with _state_lock():
        return _run_locked(asof, dry_run)


def _run_locked(asof: pd.Timestamp, dry_run: bool) -> dict:
    log.info("champion_selector asof=%s (rolling N=%d, gate n>=%d, hyst K=%d)",
             asof.date(), ROLLING_N, MIN_CLOSED, HYST_K)
    loaded = _load_state(asof=asof)
    state = loaded.state
    source_paths = {
        f"ledger:{spec.id}": spec.abs_ledger_path()
        for spec in MODELS
    }
    source_paths.update(
        {
            "source:champion_selector": Path(__file__).resolve(),
            "source:model_registry": _ROOT / "signals/model_registry.py",
            "source:ledger_config": _ROOT / "ledger/config.py",
        }
    )
    try:
        inputs_before = file_set_identity(source_paths, root=_ROOT)
    except (OSError, ArtifactSourceChangedError) as exc:
        raise ChampionStateError(
            "champion selector inputs are not stable regular files"
        ) from exc

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
    try:
        inputs_after = file_set_identity(source_paths, root=_ROOT)
    except (OSError, ArtifactSourceChangedError) as exc:
        raise ChampionStateError(
            "champion selector inputs changed during metric computation"
        ) from exc
    if inputs_before != inputs_after:
        raise ChampionStateError(
            "champion selector inputs changed during metric computation"
        )

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
        "schema_version": STATE_SCHEMA_VERSION,
        "asof": str(asof.date()),
        "updated_at": now_iso,
        "config": _expected_config(),
        "slots": slot_entries,
        "streaks": state.get("streaks", {}),
        "history": state.get("history", []) + history_events,
        "input_manifest": {
            "schema": "champion_selector_input.v1",
            "asof": str(asof.date()),
            "files": inputs_after,
        },
    }
    if state.get("migrations"):
        new_state["migrations"] = state["migrations"]
    new_state["payload_sha256"] = _state_digest(new_state)

    if dry_run:
        log.info("dry-run: champion_state.json 저장 안 함")
    else:
        if loaded.legacy_bytes is not None:
            preserved = _preserve_legacy_state(
                STATE_PATH,
                loaded.legacy_bytes,
            )
            log.warning("preserved legacy champion state: %s", preserved)
        _atomic_write_state(STATE_PATH, new_state)
        log.info("saved %s", STATE_PATH)
    return new_state


def _decode_state(raw: bytes, path: Path) -> dict:
    """Decode strict JSON so malformed state can never trigger a fallback."""
    try:
        return strict_json_object_bytes(raw, source=path)
    except ArtifactValidationError as exc:
        raise ChampionStateError(
            "existing champion state is unreadable; refusing overwrite/use: "
            f"{path}"
        ) from exc


def _path_entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ChampionStateError(
            f"champion state cannot be inspected: {path}"
        ) from exc
    return True


def _read_stable_state_bytes(path: Path) -> bytes:
    try:
        before = file_identity(path, root=path.parent)
        raw = path.read_bytes()
        after = file_identity(path, root=path.parent)
    except (OSError, ArtifactSourceChangedError) as exc:
        raise ChampionStateError(
            f"champion state read failed: {path}"
        ) from exc
    if (
        not before.get("exists")
        or before != after
        or before.get("sha256") != sha256_bytes(raw)
    ):
        raise ChampionStateError(
            f"champion state changed while reading: {path}"
        )
    return raw


def _load_state(
    *,
    asof: pd.Timestamp,
    state_path: Path | None = None,
) -> LoadedChampionState:
    path = state_path or STATE_PATH
    if not _path_entry_exists(path):
        return LoadedChampionState(
            {"slots": {}, "streaks": {}, "history": []}
        )
    raw = _read_stable_state_bytes(path)
    state = _decode_state(raw, path)

    try:
        current = _validate_state(state)
        state_asof = pd.Timestamp(current["asof"]).normalize()
        if state_asof > pd.Timestamp(asof).normalize():
            raise ValueError("champion state is future-dated for selector run")
        return LoadedChampionState(current)
    except ValueError as current_error:
        try:
            migrated = _migrate_legacy_state(
                state,
                target_asof=asof,
            )
        except ValueError as migration_error:
            raise ChampionStateError(
                "existing champion state is invalid and not an authenticated "
                f"migratable legacy contract; refusing overwrite: {path}; "
                f"current={current_error}; legacy={migration_error}"
            ) from migration_error
        log.warning(
            "authenticated legacy champion state will be migrated; "
            "history=%d",
            len(migrated["history"]),
        )
        return LoadedChampionState(migrated, legacy_bytes=raw)


# ==========================================================================
# ops-steward 가 읽을 헬퍼 (champion_state 읽는 표준 인터페이스).
# ==========================================================================
def load_champion_state_artifact(
    state_path: Path | str = STATE_PATH,
    *,
    expected_asof: str | pd.Timestamp | None = None,
) -> ValidatedChampionStateArtifact | None:
    """Strictly decode, authenticate, and as-of validate one state snapshot."""
    path = Path(state_path)
    with _state_lock(path, shared=True):
        if not _path_entry_exists(path):
            return None
        raw = _read_stable_state_bytes(path)
        state = _decode_state(raw, path)
        try:
            validated = _validate_state(
                state,
                expected_asof=expected_asof,
            )
        except (TypeError, ValueError) as exc:
            log.error("champion state read failed: %s: %s", path, exc)
            raise ChampionStateError(
                f"champion state validation failed: {path}: {exc}"
            ) from exc
        return ValidatedChampionStateArtifact(
            payload=validated,
            identity={
                "path": str(path),
                "sha256": sha256_bytes(raw),
                "size": len(raw),
            },
        )


def get_champion(
    slot: str,
    state_path: Path | str = STATE_PATH,
    *,
    expected_asof: str | pd.Timestamp | None = None,
) -> dict | None:
    """slot 의 현 챔피언 entry 를 반환 (없으면 None).
    반환 = {'champion_id','since','is_fallback','metric','reason'}.
    ops-steward send dispatcher 가 champion_id → model_registry.get_model 로 predict_ref 획득."""
    artifact = load_champion_state_artifact(
        state_path,
        expected_asof=expected_asof,
    )
    if artifact is None:
        return None
    slots = artifact.payload["slots"]
    entry = slots.get(slot)
    return entry if isinstance(entry, dict) else None


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)])
    ap = argparse.ArgumentParser(description="champion_selector (unattended, downside-first)")
    ap.add_argument("--asof", type=str, default=None, help="YYYY-MM-DD (default=today)")
    ap.add_argument("--dry-run", action="store_true", help="champion_state.json 저장 X")
    args = ap.parse_args()
    asof = (
        pd.Timestamp(args.asof).normalize()
        if args.asof
        else pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None).normalize()
    )
    state = run(asof, args.dry_run)
    print(json.dumps(state, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
