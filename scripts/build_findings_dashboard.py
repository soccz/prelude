#!/usr/bin/env python3
"""build_findings_dashboard.py — 이번 세션 새 발견을 *차트용 JSON* 으로 빌드.

대시보드는 시각화에 집중 (사용자 지시). 이 빌더는 prelude DB / 산출 CSV 에서
검증된 수치만 모아 `findings.json` 을 만든다. index.html 의 새 6 개 차트가 소비.

수치 출처 (전부 DB / output CSV 검증값 — 창작 금지):
  - base / head top-bucket       : output/downside_head_baserates_v1.json
                                    output/downside_head_reliability_v1.csv
  - risk-reward 하방축소          : output/downside_head_riskreward_compare_v1.csv
  - calibration 구/신             : output/calibration_summary.json (h5 구엔진)
                                    + output/recommendation_scorer_calibration_v1.csv
  - historical OOS precision     : output/recommendation_scorer_oos_metrics_v1.csv
  - 5월 backtest 펌프 포착         : data/upbit_d1.db (일봉 high/open-1) — 실측 재검증
  - 선행패턴 OOS lift             : output/univariate_precursor_lift_v1.csv
  - regime 별 펌프 base rate       : output/market_breadth_regime_baserate_v1.csv

honest 캡션: 차트는 *레이더 정직성* 을 보이기 위한 것. 일중 최고가 = 포착 펌프 크기지
실현수익 아님 (사람이 +5% TP 청산). "포착" 으로만 표기.

사용:
    python scripts/build_findings_dashboard.py --out-dir <github.io path>/dashboard/data
    # passphrase는 build_dashboard와 동일(--pin/PRELUDE_DASHBOARD_PIN, 기본값 없음)
"""
from __future__ import annotations

import argparse
import io
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 암호화 + 기본값은 build_dashboard 와 100% 공유 (스킴 일치 보장).
from scripts.build_dashboard import (  # noqa: E402
    DEFAULT_OUT_DIR,
    _read_stable_artifact_bytes,
    _write_json,
    resolve_dashboard_generation_id,
    resolve_dashboard_passphrase,
)
from data.database import connect_readonly  # noqa: E402
from ledger.portfolio_metrics import normalize_kst_date  # noqa: E402
from ops.artifact_provenance import (  # noqa: E402
    ArtifactValidationError,
    sha256_bytes,
    strict_json_object_bytes,
)
from ops.policy_competition import (  # noqa: E402
    PolicyArtifactError,
    load_policy_artifact,
)

log = logging.getLogger("findings")

DB_PATH = "data/upbit_d1.db"
POLICY_COMPETITION_JSON = "output/policy_competition_summary.json"
POLICY_COMPETITION_DB = "data/policy_competition.db"
BASE_RATES_JSON = "output/downside_head_baserates_v1.json"
HEAD_RELIABILITY_CSV = "output/downside_head_reliability_v1.csv"
RISK_REWARD_CSV = "output/downside_head_riskreward_compare_v1.csv"
LEGACY_CALIBRATION_JSON = "output/calibration_summary.json"
SCORER_CALIBRATION_CSV = "output/recommendation_scorer_calibration_v1.csv"
SCORER_METRICS_CSV = "output/recommendation_scorer_oos_metrics_v1.csv"
PRECURSOR_LIFT_CSV = "output/univariate_precursor_lift_v1.csv"
REGIME_BASERATE_CSV = "output/market_breadth_regime_baserate_v1.csv"

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


class FindingsDataError(RuntimeError):
    """A displayed research metric lacks current, parseable source evidence."""


def _artifact_identity(path: Path, raw: bytes) -> dict:
    return {
        "path": str(path),
        "sha256": sha256_bytes(raw),
        "size": len(raw),
    }


def _read_json_artifact(path_value: str) -> tuple[dict, dict]:
    path = Path(path_value)
    try:
        raw = _read_stable_artifact_bytes(path)
        if raw is None:
            raise FileNotFoundError(path)
        payload = strict_json_object_bytes(raw, source=path)
    except (OSError, ArtifactValidationError) as exc:
        raise FindingsDataError(f"invalid findings JSON source: {path}") from exc
    return payload, _artifact_identity(path, raw)


def _read_csv_artifact(
    path_value: str,
    *,
    required: set[str],
) -> tuple[pd.DataFrame, dict]:
    path = Path(path_value)
    try:
        raw = _read_stable_artifact_bytes(path)
        if raw is None:
            raise FileNotFoundError(path)
        frame = pd.read_csv(io.BytesIO(raw))
    except (
        OSError,
        ArtifactValidationError,
        pd.errors.ParserError,
        UnicodeError,
    ) as exc:
        raise FindingsDataError(f"invalid findings CSV source: {path}") from exc
    missing = sorted(required - set(frame.columns))
    if missing or frame.empty:
        raise FindingsDataError(
            f"findings CSV schema/data invalid: {path}; missing={missing}"
        )
    return frame, _artifact_identity(path, raw)


def _one_row(frame: pd.DataFrame, mask: pd.Series, *, source: str) -> pd.Series:
    selected = frame.loc[mask]
    if len(selected) != 1:
        raise FindingsDataError(
            f"expected exactly one findings row for {source}, got {len(selected)}"
        )
    return selected.iloc[0]


def _pct(value: object, *, source: str, scale: float = 100.0) -> float:
    number = _finite_number(value, math.nan)
    if not math.isfinite(number):
        raise FindingsDataError(f"non-finite findings value: {source}")
    return round(number * scale, 4)


def _finite_number(value, default: float) -> float:
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _nonnegative_int(value) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        return None
    return int(number)


def _required_nonnegative_int(value, *, source: str) -> int:
    parsed = _nonnegative_int(value)
    if parsed is None:
        raise FindingsDataError(
            f"findings count must be a non-negative integer: {source}"
        )
    return parsed


def build_champion_leaderboard(*, asof=None) -> dict:
    """champion/challenger 리더보드 — 각 등록 모델의 forward 성과 + 현 champion 표시.

    ★ 단일 정본 재사용: ops.champion_selector.compute_metric 가 셀렉터가 쓰는 바로 그
      하방-우선 metric(P(≤-5%)·net mean·hit·n_days, 왕복 0.15% 차감)을 산출한다. 여기서
      재계산하지 않는다 (대시보드 ↔ 셀렉터 숫자 불일치 방지). champion 마킹은
      get_champion(slot). 데이터 없으면(forward 0행) graceful — 빈 metric.

    leak 무관: compute_metric 은 CLOSED(실현된 과거) 행만 본다 (champion_selector docstring)."""
    from ops.champion_selector import (
        STATE_PATH,
        ChampionStateError,
        compute_metric,
        load_champion_state_artifact,
    )
    from signals.model_registry import MODELS

    cutoff = normalize_kst_date(asof)
    try:
        artifact = load_champion_state_artifact(
            STATE_PATH,
            expected_asof=cutoff,
        )
        if artifact is None:
            raise ChampionStateError("champion state is absent")
        state = artifact.payload
        state_asof = normalize_kst_date(state.get("asof"))
        state_identity = artifact.identity
    except (ChampionStateError, TypeError, ValueError) as exc:
        raise FindingsDataError(
            f"champion state is missing or invalid: {STATE_PATH}"
        ) from exc

    # _validate_state 가 schema/checksum/config/등록 모델/slot/as-of 를 검증한다.
    # 대시보드 cutoff 뒤의 metric 이 state 안에 섞인 경우도 표시 전에 차단한다.
    champ_by_slot: dict[str, str] = {}
    for slot in ("open", "preopen"):
        entry = state["slots"][slot]
        metric = entry.get("metric")
        last_date = metric.get("last_date") if isinstance(metric, dict) else None
        if last_date is not None:
            try:
                if normalize_kst_date(last_date) > cutoff:
                    raise FindingsDataError(
                        f"champion metric is future-dated: {slot}/{last_date}"
                    )
            except ValueError as exc:
                raise FindingsDataError(
                    f"champion metric last_date is invalid: {slot}/{last_date}"
                ) from exc
        champ_by_slot[slot] = entry["champion_id"]

    rows = []
    for spec in MODELS:
        mm = compute_metric(spec, cutoff)
        # 이 모델이 어느 slot 의 현 champion 인가 (open 우선 표기).
        champ_slots = [s for s in spec.slots if champ_by_slot.get(s) == spec.id]
        rows.append({
            "model_id": spec.id,
            "name": spec.name,
            "slots": list(spec.slots),
            "is_champion_slots": champ_slots,           # [] 면 챔피언 아님
            "challenger_only": bool(spec.challenger_only),
            "n_days": int(mm.n_days),                    # 독립 forward 관측(거래일)
            "n_closed": int(mm.n_closed),
            "deep_loss_freq_pct": (None if (mm.deep_loss_freq != mm.deep_loss_freq)
                                   else round(mm.deep_loss_freq * 100.0, 1)),  # P(≤-5%)
            "net_mean_pct": (None if (mm.net_mean_pct != mm.net_mean_pct)
                             else round(mm.net_mean_pct, 2)),                  # net (비용차감)
            "hit_rate_pct": (None if (mm.hit_rate is None or mm.hit_rate != mm.hit_rate)
                             else round(mm.hit_rate * 100.0, 1)),
            "gate_pass": bool(mm.gate_pass),
            "last_date": mm.last_date,
            "reason": mm.reason,
        })
    return {
        "asof": str(cutoff.date()),
        "champion_state_asof": (
            str(state_asof.date()) if state_asof is not None else None
        ),
        "champion_state_usable": True,
        "champion_state_identity": state_identity,
        "current_champion": champ_by_slot,
        "deep_loss_threshold_pct": -5.0,
        "rows": rows,
        "note": ("하방-우선 자동선정(unattended): 1순위 P(≤-5%)↓ · 2순위 net(왕복0.15%차감)↑ · "
                 "3순위 hit↑. forward CLOSED 거래일 n≥30 게이트 통과 + 챌린저 아님 모델만 "
                 "챔피언 후보. 미달이면 백테스트-최선(R1) SHADOW fallback."),
        "source": "ops.champion_selector.compute_metric (셀렉터 정본 재사용) · champion_state.json",
    }


def build_policy_competition_panel(
    summary_path: str = POLICY_COMPETITION_JSON,
    db_path: str = POLICY_COMPETITION_DB,
    *,
    asof=None,
) -> dict:
    """Model + send-policy competition panel for dashboard findings.

    This consumes the record-only artifact produced after close-out. It does
    not promote or demote anything; it makes the send/no-send policy measurable.
    """
    path = Path(summary_path)
    cutoff = normalize_kst_date(asof)
    try:
        raw_payload = _read_stable_artifact_bytes(path)
        if raw_payload is None:
            raise FileNotFoundError(path)
        payload = load_policy_artifact(
            path,
            csv_path=path.with_suffix(".csv"),
            db_path=Path(db_path),
            asof=cutoff,
            require_current=True,
        )
        confirmed_payload = _read_stable_artifact_bytes(path)
        if confirmed_payload != raw_payload:
            raise FindingsDataError(
                "policy competition summary changed during findings build"
            )
    except (OSError, ArtifactValidationError, PolicyArtifactError) as exc:
        raise FindingsDataError(
            "policy competition artifact triplet is missing, stale, or "
            f"inconsistent: {summary_path}"
        ) from exc
    # ``load_policy_artifact`` already validates schema, dates, config, every
    # row field, digest, exact CSV bytes, and the matching SQLite snapshot.
    artifact_asof = normalize_kst_date(payload["asof"])
    raw_rows = payload["rows"]
    rows = [row for row in raw_rows if row["n_closed"] > 0]
    rows.sort(
        key=lambda r: (
            _finite_number(r.get("pump20_recall_pct"), -1),
            _finite_number(r.get("net_mean_pct"), -999),
            r["n_closed"],
        ),
        reverse=True,
    )

    def _top(metric: str) -> dict | None:
        valid = [
            r for r in rows
            if math.isfinite(_finite_number(r.get(metric), math.nan))
        ]
        if not valid:
            return None
        return max(
            valid,
            key=lambda r: (
                _finite_number(r.get(metric), -999),
                _finite_number(r.get("net_mean_pct"), -999),
                r["n_closed"],
            ),
        )

    best_pump = _top("pump20_recall_pct")
    best_net = _top("net_mean_pct")
    db_info = {
        "path": db_path,
        "exists": True,
        "latest_asof": str(artifact_asof.date()),
        "latest_row_count": len(raw_rows),
        "best_pump_participant": (
            best_pump["participant_id"] if best_pump else None
        ),
        "best_net_participant": (
            best_net["participant_id"] if best_net else None
        ),
    }

    r1_like = [
        r for r in rows
        if str(r.get("source_id")) in {
            "recommend_r1_open",
            "recommend_r2_open",
            "recommend_r1_sustain_open",
            "R1/R2/A1",
        }
    ]
    r1_captured = sum(
        _nonnegative_int(r.get("pump20_captured")) or 0 for r in r1_like
    )

    return {
        "asof": str(artifact_asof.date()),
        "generated_at_utc": payload.get("generated_at_utc"),
        "artifact_identity": _artifact_identity(path, raw_payload),
        "config": payload.get("config", {}),
        "best_pump_recall": best_pump,
        "best_net_mean": best_net,
        "rows": rows[:16],
        "database": db_info,
        "diagnosis": {
            "r1_r2_a1_pump20_captured_total": int(r1_captured),
            "message": (
                "R1/R2/A1 계열은 현 forward 표본에서 +20% 급등 recall 이 낮다. "
                "실전 알림 전환 전, PUMP/WATCH 후보 생성기와 send/no-send policy 를 "
                "분리해 경쟁시켜야 한다."
            ),
        },
        "note": (
            "Record-only model + send-policy competition. CLOSED forward rows only; "
            "pump20_recall denominator is all KRW +20% daily high/open events for the same dates."
        ),
        "source": (
            "output/policy_competition_summary.json + "
            "output/policy_competition_summary.csv + "
            "data/policy_competition.db"
        ),
    }


def build_magnitude_curve(*, asof=None) -> dict:
    cutoff = normalize_kst_date(asof)
    base, base_identity = _read_json_artifact(BASE_RATES_JSON)
    reliability, reliability_identity = _read_csv_artifact(
        HEAD_RELIABILITY_CSV,
        required={"label", "bucket", "n", "pred_mean", "actual"},
    )
    try:
        source_asof = normalize_kst_date(base["asof"])
    except (KeyError, ValueError) as exc:
        raise FindingsDataError("base-rate artifact lacks a valid asof") from exc
    if source_asof > cutoff:
        raise FindingsDataError("base-rate artifact is future-dated")
    base_rates = base.get("base_rates")
    if not isinstance(base_rates, dict):
        raise FindingsDataError("base-rate artifact lacks base_rates")
    labels = (
        (5, "lab_up_05"),
        (10, "lab_up_10"),
        (15, "lab_up_15"),
        (20, "lab_up_20"),
    )
    base_values = []
    top_values = []
    top_counts = []
    for threshold, label in labels:
        base_values.append(
            _pct(
                base_rates.get(label),
                source=f"{BASE_RATES_JSON}:{label}",
            )
        )
        rows = reliability[reliability["label"].astype(str) == label]
        if rows.empty:
            raise FindingsDataError(f"reliability label missing: {label}")
        bucket = pd.to_numeric(rows["bucket"], errors="raise").max()
        row = _one_row(
            rows,
            pd.to_numeric(rows["bucket"], errors="raise") == bucket,
            source=f"{HEAD_RELIABILITY_CSV}:{label}:top_bucket",
        )
        top_values.append(
            _pct(row["actual"], source=f"{label}.top.actual")
        )
        top_counts.append(
            _required_nonnegative_int(row["n"], source=f"{label}.n")
        )
    return {
        "thresholds_pct": [item[0] for item in labels],
        "base_rate_pct": base_values,
        # Backward-compatible dashboard key. Values are now derived from the
        # head reliability top bucket, not a hand-entered case-study series.
        "top_decile_pct": top_values,
        "top_bucket_actual_pct": top_values,
        "top_bucket_n": top_counts,
        "artifact_asof": str(source_asof.date()),
        "note": (
            "전체 base rate와 각 독립 head의 최고 예측 bucket 실제 적중률. "
            "동일 후보군의 누적 생존곡선이 아니므로 threshold 간 단조성 증거로 "
            "사용하지 않는다."
        ),
        "sources": [base_identity, reliability_identity],
    }


def build_risk_reward_panel() -> dict:
    frame, identity = _read_csv_artifact(
        RISK_REWARD_CSV,
        required={
            "policy",
            "k",
            "param",
            "n",
            "p_min_le_5",
            "p_min_le_10",
        },
    )
    k = pd.to_numeric(frame["k"], errors="coerce")
    policies = (
        ("upside-only", "upside_only", "-"),
        ("R1_ratio", "R1_ratio", "-"),
        ("R2 (λ=1)", "R2_penalized", "lam=1.0"),
    )
    rows = [
        _one_row(
            frame,
            (frame["policy"].astype(str) == policy)
            & (frame["param"].astype(str) == param)
            & (k == 3),
            source=f"{policy}/{param}/K3",
        )
        for _, policy, param in policies
    ]
    return {
        "labels": [label for label, _, _ in policies],
        "p_down5_pct": [
            _pct(row["p_min_le_5"], source=f"{label}.p_min_le_5")
            for row, (label, _, _) in zip(rows, policies, strict=True)
        ],
        "p_deepdump_pct": [
            _pct(row["p_min_le_10"], source=f"{label}.p_min_le_10")
            for row, (label, _, _) in zip(rows, policies, strict=True)
        ],
        "n": [
            _required_nonnegative_int(
                row["n"],
                source=f"{label}.n",
            )
            for row, (label, _, _) in zip(rows, policies, strict=True)
        ],
        "note": "동일 artifact의 K=3 정책 비교이며 실현수익 보장은 아니다.",
        "sources": [identity],
    }


def build_calibration_panel() -> dict:
    legacy, legacy_identity = _read_json_artifact(LEGACY_CALIBRATION_JSON)
    calibration, calibration_identity = _read_csv_artifact(
        SCORER_CALIBRATION_CSV,
        required={"b", "pred_prob", "actual_hit", "n", "score", "label"},
    )
    legacy_h5 = legacy.get("h5")
    if not isinstance(legacy_h5, dict):
        raise FindingsDataError("legacy calibration h5 payload missing")
    current = calibration[
        (calibration["score"].astype(str) == "cal_composite")
        & (calibration["label"].astype(str) == "lab_pump20")
    ]
    if current.empty:
        raise FindingsDataError("current scorer calibration rows missing")
    max_bucket = pd.to_numeric(current["b"], errors="raise").max()
    row = _one_row(
        current,
        pd.to_numeric(current["b"], errors="raise") == max_bucket,
        source="cal_composite/lab_pump20/top_bucket",
    )
    old_pred = _finite_number(
        legacy_h5.get("top_bucket_mean_pred_pct"),
        math.nan,
    )
    old_actual = _finite_number(
        legacy_h5.get("top_bucket_actual_hit_pct"),
        math.nan,
    )
    if not math.isfinite(old_pred) or not math.isfinite(old_actual):
        raise FindingsDataError("legacy calibration values are invalid")
    new_pred = _pct(row["pred_prob"], source="current.pred_prob")
    new_actual = _pct(row["actual_hit"], source="current.actual_hit")
    research_scorer = {
        "label": "research cal_composite P(≥20%) 최고 bucket",
        "pred_pct": new_pred,
        "actual_pct": new_actual,
        "overconfidence_pp": round(new_pred - new_actual, 4),
        "n": _required_nonnegative_int(
            row["n"],
            source="current.top_bucket.n",
        ),
    }
    return {
        "ideal_line": [[0, 0], [100, 100]],
        "old_engine": {
            "label": "구 7-head +20% (h5)",
            "pred_pct": round(old_pred, 4),
            "actual_pct": round(old_actual, 4),
            "overconfidence_pp": round(old_pred - old_actual, 4),
            "n": _required_nonnegative_int(
                legacy_h5.get("top_bucket_n"),
                source="legacy_h5.top_bucket_n",
            ),
        },
        "research_scorer": research_scorer,
        # Existing dashboard renderer consumes this key. It is deliberately
        # labeled research-only and must not be read as live R1 calibration.
        "new_scanner": research_scorer,
        "note": (
            "둘 다 저장된 historical OOS/reliability artifact 비교다. "
            "현재 live R1의 strict post-contract forward calibration 증거는 0건이므로 "
            "live 확률 정직성으로 일반화하지 않는다."
        ),
        "sources": [legacy_identity, calibration_identity],
    }


def build_precision_at3_panel() -> dict:
    frame, identity = _read_csv_artifact(
        SCORER_METRICS_CSV,
        required={
            "score",
            "label",
            "regime",
            "K",
            "n_days",
            "n_picks",
            "precision_at_k",
            "base_rate",
            "lift",
        },
    )
    row = _one_row(
        frame,
        (frame["score"].astype(str) == "cal_composite")
        & (frame["label"].astype(str) == "lab_pump20")
        & (frame["regime"].astype(str) == "ALL")
        & (pd.to_numeric(frame["K"], errors="coerce") == 3),
        source="cal_composite/lab_pump20/ALL/K3",
    )
    lift = _finite_number(row["lift"], math.nan)
    if not math.isfinite(lift):
        raise FindingsDataError("precision-at-3 lift is non-finite")
    n_days = _required_nonnegative_int(row["n_days"], source="precision.n_days")
    return {
        "full_oos_pct": _pct(row["precision_at_k"], source="precision_at_k"),
        "full_oos_base_pct": _pct(row["base_rate"], source="base_rate"),
        "full_oos_lift": round(lift, 4),
        "full_oos_window": f"historical OOS {n_days}일",
        "n_picks": _required_nonnegative_int(
            row["n_picks"],
            source="precision.n_picks",
        ),
        "scope": "research scorer artifact; current live R1 strict forward proof 아님",
        "sources": [identity],
    }


def build_precursor_lift_panel() -> dict:
    frame, identity = _read_csv_artifact(
        PRECURSOR_LIFT_CSV,
        required={
            "feature",
            "label",
            "oos_lift",
            "oos_n",
            "n_folds_used",
        },
    )
    wanted = (
        "f_qv_surge_30d",
        "f_bounce_off_7d_low",
        "f_ret_7d",
        "f_ret_3d",
        "f_rv_7d",
    )
    rows = []
    for feature in wanted:
        row = _one_row(
            frame,
            (frame["feature"].astype(str) == feature)
            & (frame["label"].astype(str) == "lab_pump15"),
            source=f"{feature}/lab_pump15",
        )
        oos_lift = _finite_number(row["oos_lift"], math.nan)
        if not math.isfinite(oos_lift):
            raise FindingsDataError(
                f"precursor OOS lift is non-finite: {feature}"
            )
        rows.append(
            {
                "name": feature.removeprefix("f_"),
                "oos_lift": round(oos_lift, 4),
                "oos_n": _required_nonnegative_int(
                    row["oos_n"],
                    source=f"{feature}.oos_n",
                ),
                "n_folds_used": _required_nonnegative_int(
                    row["n_folds_used"],
                    source=f"{feature}.n_folds_used",
                ),
            }
        )
    return {
        "features": rows,
        "base_line": 1.0,
        "note": (
            "단일 feature historical OOS lift. 조합 전략 수익률이나 현재 live "
            "forward 성과가 아니다."
        ),
        "sources": [identity],
    }


def build_regime_baserate_panel() -> dict:
    frame, identity = _read_csv_artifact(
        REGIME_BASERATE_CSV,
        required={
            "regime_d1",
            "n_days",
            "mean_breadth15",
            "pump_rich_rate",
        },
    )
    ordered = frame.sort_values("mean_breadth15", ascending=False)
    return {
        "regimes": ordered["regime_d1"].astype(str).tolist(),
        "pump15_breadth_pct": [
            _pct(value, source="mean_breadth15")
            for value in ordered["mean_breadth15"]
        ],
        "pump_rich_rate_pct": [
            _pct(value, source="pump_rich_rate")
            for value in ordered["pump_rich_rate"]
        ],
        "n_days": [
            _required_nonnegative_int(value, source="regime.n_days")
            for value in ordered["n_days"]
        ],
        "note": "저장된 regime별 historical base-rate 집계.",
        "sources": [identity],
    }


def verify_backtest_pumps(db_path: str, *, asof=None) -> list[dict]:
    """DB 일봉에서 high/open−1 을 직접 계산 (창작 X, 실측)."""
    cutoff = normalize_kst_date(asof)
    out: list[dict] = []
    if not Path(db_path).exists():
        log.warning("DB 없음 (%s) — backtest 차트 빈값", db_path)
        return out
    with connect_readonly(db_path) as con:
        for p in BACKTEST_PUMPS:
            try:
                pump_date = normalize_kst_date(p["date"])
            except (KeyError, ValueError):
                log.warning("invalid backtest pump identity: %r", p)
                continue
            if pump_date > cutoff:
                log.warning(
                    "future backtest pump excluded: %s %s > %s",
                    p.get("market"),
                    p.get("date"),
                    cutoff.date(),
                )
                continue
            row = con.execute(
                "SELECT open, high FROM candles WHERE market=? AND timestamp=?",
                (p["market"], p["date"] + " 09:00:00"),
            ).fetchone()
            if not row or not row[0]:
                log.warning("no candle: %s %s", p["market"], p["date"])
                continue
            try:
                o, h = float(row[0]), float(row[1])
            except (TypeError, ValueError):
                log.warning("invalid candle numeric: %s %s", p["market"], p["date"])
                continue
            if (
                not math.isfinite(o)
                or not math.isfinite(h)
                or o <= 0
                or h < o
            ):
                log.warning("invalid candle OHLC: %s %s", p["market"], p["date"])
                continue
            pump = round((h / o - 1.0) * 100.0, 1)
            out.append({
                "coin": p["market"].replace("KRW-", ""),
                "date": p["date"],
                "pump_pct": pump,  # 일중 최고가 포착 크기 (실현수익 X)
            })
    # 큰 펌프 우선 정렬 (막대/타임라인 모두 사용)
    out.sort(key=lambda r: float(r["pump_pct"]), reverse=True)
    return out


def build_payload(
    db_path: str,
    *,
    asof=None,
    allow_missing_core: bool = False,
) -> dict:
    cutoff = normalize_kst_date(asof)
    pumps = verify_backtest_pumps(db_path, asof=cutoff)
    log.info("backtest pumps verified: %d", len(pumps))
    magnitude_curve = build_magnitude_curve(asof=cutoff)
    risk_reward = build_risk_reward_panel()
    calibration = build_calibration_panel()
    precision_at3 = build_precision_at3_panel()
    precursor_lift = build_precursor_lift_panel()
    regime_baserate = build_regime_baserate_panel()

    try:
        leaderboard = build_champion_leaderboard(asof=cutoff)
        log.info("champion leaderboard: %d models, champion(open)=%s preopen=%s",
                 len(leaderboard["rows"]),
                 leaderboard["current_champion"].get("open"),
                 leaderboard["current_champion"].get("preopen"))
    except Exception as exc:
        if not allow_missing_core:
            raise FindingsDataError(
                "canonical findings build requires valid champion state"
            ) from exc
        log.warning("preview only: champion leaderboard unavailable: %s", exc)
        leaderboard = None

    try:
        policy_competition = build_policy_competition_panel(asof=cutoff)
        if policy_competition["asof"] != str(cutoff.date()):
            raise FindingsDataError(
                "policy competition snapshot is stale for findings cutoff: "
                f"{policy_competition['asof']} != {cutoff.date()}"
            )
        log.info("policy competition panel: %s",
                 policy_competition["asof"])
    except Exception as exc:
        if not allow_missing_core:
            raise FindingsDataError(
                "canonical findings build requires current policy competition"
            ) from exc
        log.warning("preview only: policy competition unavailable: %s", exc)
        policy_competition = None

    return {
        "asof": str(cutoff.date()),
        "asof_timezone": "Asia/Seoul",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "champion_leaderboard": leaderboard,
        "policy_competition": policy_competition,
        "honest_caption": (
            "Historical research diagnostics — 큰 펌프 *포착* 능력과 하방 비교. "
            "수익기 아님: 일중 최고가는 포착 크기지 실현수익 X (사람이 +5% TP 청산). "
            "자동 net 전략 아님 · 현재 live R1 strict forward/calibration 증거 0건."
        ),

        # ① 상승 확률분포 — 저장 artifact에서 동적으로 읽은 base/head 최고 bucket.
        "magnitude_curve": magnitude_curve,

        # ② risk-reward 하방축소 — artifact의 K=3 동일 조건 비교.
        "risk_reward": risk_reward,

        # ③ calibration reliability — 두 저장 artifact를 현재 hash와 함께 표시.
        "calibration": calibration,

        # ④ 5월 backtest 적중 — top-3 가 포착한 펌프 (막대 = 크기, date = 타임라인)
        #    DB 일봉 high/open−1 실측. ★포착 크기지 실현수익 X.
        "backtest_pumps": {
            "pumps": pumps,
            "precision_at3": precision_at3,
            "note": "일중 최고가 = 레이더가 포착한 펌프 크기 (실현수익 아님 · +5% TP 청산)",
            "source": "data/upbit_d1.db 일봉 high/open−1 (실측 재검증)",
        },

        # ⑤ 선행패턴 historical OOS lift — CSV 원값.
        "precursor_lift": precursor_lift,

        # ⑥ regime 별 historical 펌프 base rate — CSV 원값.
        "regime_baserate": regime_baserate,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--asof", help="inclusive KST cutoff (default=today KST)")
    parser.add_argument("--pin", default=None)
    parser.add_argument("--no-encrypt", action="store_true",
                        help="평문 출력 (테스트 only). 라이브 publish 금지.")
    parser.add_argument(
        "--allow-missing-core",
        action="store_true",
        help=(
            "preview only: champion/policy 핵심 입력 오류를 빈 패널로 허용 "
            "(--no-encrypt 필수, publish 금지)"
        ),
    )
    args = parser.parse_args()

    if args.allow_missing_core and not args.no_encrypt:
        parser.error("--allow-missing-core requires --no-encrypt (preview only)")

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    if args.no_encrypt:
        pin = None
    else:
        try:
            pin = resolve_dashboard_passphrase(args.pin)
        except ValueError as exc:
            parser.error(str(exc))
    try:
        generation_id = resolve_dashboard_generation_id()
    except ValueError as exc:
        parser.error(str(exc))
    log.info("encryption: %s", "PIN " + ("*" * len(pin)) if pin else "OFF (plaintext)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = build_payload(
        args.db,
        asof=args.asof,
        allow_missing_core=args.allow_missing_core,
    )
    payload["dashboard_generation_id"] = generation_id
    _write_json(out_dir / "findings.json", payload, passphrase=pin)
    log.info("saved findings.json (%d backtest pumps)",
             len(payload["backtest_pumps"]["pumps"]))

    print("\n=== findings.json ===")
    for p in payload["backtest_pumps"]["pumps"]:
        print(f"  {p['coin']:<8} {p['date']}  +{p['pump_pct']:.1f}% (포착)")
    pc = payload.get("policy_competition") or {}
    best = pc.get("best_pump_recall") or {}
    if best:
        print(
            "  policy competition best pump recall: "
            f"{best.get('participant_id')} "
            f"{best.get('pump20_captured')}/{best.get('pump20_actual')}"
        )
    print(f"\nout_dir: {out_dir}")


if __name__ == "__main__":
    main()
