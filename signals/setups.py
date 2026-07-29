"""Setup Library v1 — Setup Discovery v1 결과 기반 룰 정의.

setup discovery 결과 (output/setup_discovery_v1.csv) 에서 robust + interpretable
한 leaf 들을 named setup 으로 인코딩.

각 setup 은:
  detect_fn(row) -> bool   # 패널 행 1개에 룰 적용
  past_stats               # 과거 검증 통계 (alert 표시용)

S04 는 setup 이 아니라 BTC context (regime) — get_btc_context() 사용.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

log = logging.getLogger(__name__)


# ============================================================================
# Setup definitions
# ============================================================================
SETUP_LIBRARY: dict[str, dict] = {
    "S01": {
        "name": "high-vol momentum",
        "desc": "ATR 높음 + 전일 상승 + 3d ROC 상위 (h2/h6 공통)",
        "heads_supported": ["h2_hit_3_4h", "h6_hit_5_24h"],
        "rule_text": "atr_pct_14 > 0.06 AND log_return_1d > 0.04 AND roc_3d (rank) > 0.93",
        "detect_fn": lambda r: (
            r.get("atr_pct_14", 0) > 0.06
            and r.get("log_return_1d", 0) > 0.04
            and r.get("roc_3d", 0) > 0.93
        ),
        "past_stats": {
            "h2_lift": 4.09, "h2_prec_pct": 47.5, "h2_n": 552, "h2_fail_eod_pct": -2.5,
            "h6_lift": 2.63, "h6_prec_pct": 48.6, "h6_n": 479, "h6_fail_eod_pct": -4.2,
        },
    },
    "S02": {
        "name": "strong yesterday move",
        "desc": "전일 강한 상승 (log_return_1d > 5%) — h5 핵심 robust setup",
        "heads_supported": ["h5_tail_20"],
        "rule_text": "log_return_1d > 0.048 (depth=1, fold 1: lift 4.07)",
        "detect_fn": lambda r: r.get("log_return_1d", 0) > 0.048,
        "past_stats": {
            "h5_lift": 4.07, "h5_prec_pct": 6.5, "h5_n": 1408, "h5_fail_eod_pct": -0.65,
        },
    },
    "S03": {
        "name": "vol expansion + 5d momentum",
        "desc": "5d 변동성 상위 + 5/7d return 상위 + 전일 상승 (h5 보조)",
        "heads_supported": ["h5_tail_20"],
        "rule_text": "vol_5d (rank) > 0.83 AND return_7d (rank) > 0.91 AND log_return_1d > 0.059",
        "detect_fn": lambda r: (
            r.get("vol_5d", 0) > 0.83
            and r.get("return_7d", 0) > 0.91
            and r.get("log_return_1d", 0) > 0.059
        ),
        "past_stats": {
            "h5_lift": 6.81, "h5_prec_pct": 8.05, "h5_n": 298, "h5_fail_eod_pct": -1.89,
        },
    },
    "S04": {
        # BTC context — not a setup, but tracked
        "name": "BTC bull context",
        "desc": "BTC regime ∈ {bull_quiet, bull_volatile}",
        "heads_supported": [],
        "rule_text": "btc_regime in {bull_quiet, bull_volatile}",
        "detect_fn": lambda r: r.get("btc_regime", "") in {"bull_quiet", "bull_volatile"},
        "past_stats": {"context_only": True},
    },
}


def _detect_setups(
    row: Any,
    setup_library: Mapping[str, Mapping[str, Any]],
    error_counts: dict[str, int] | None = None,
    first_errors: dict[str, Exception] | None = None,
) -> list[str]:
    fired = []
    for setup_id, sdef in setup_library.items():
        try:
            if sdef["detect_fn"](row):
                fired.append(setup_id)
        except Exception as exc:
            # 단일 행 API의 기존 관용 계약은 유지한다. 배치 실행자는 로컬
            # 카운터를 넘겨 일부 오류를 진단하고 전량 오류를 fail-loud 한다.
            if error_counts is not None:
                error_counts[setup_id] += 1
            if first_errors is not None:
                first_errors.setdefault(setup_id, exc)
            continue
    return fired


def detect_setups(row: Any) -> list[str]:
    """Apply all setup rules to one row, preserving the legacy permissive API."""
    return _detect_setups(row, SETUP_LIBRARY)


def evaluate_setup_rows(
    rows: Iterable[Any],
    *,
    setup_library: Mapping[str, Mapping[str, Any]] | None = None,
    log_partial_errors: bool = False,
) -> tuple[list[list[str]], dict[str, int]]:
    """Evaluate one execution batch and return matches plus local error counts.

    A rule that raises for only part of the batch remains non-fatal so healthy
    rows keep their existing setup matches. A rule that raises for every
    evaluated row is a broken rule/schema contract, not a valid zero-fire
    signal, and therefore fails the execution.
    """
    library = SETUP_LIBRARY if setup_library is None else setup_library
    error_counts = {setup_id: 0 for setup_id in library}
    first_errors: dict[str, Exception] = {}
    matched = [
        _detect_setups(row, library, error_counts, first_errors)
        for row in rows
    ]
    row_count = len(matched)

    failed_all = [
        setup_id
        for setup_id, count in error_counts.items()
        if row_count > 0 and count == row_count
    ]
    if failed_all:
        details = "; ".join(
            (
                f"{setup_id}={error_counts[setup_id]}/{row_count} "
                f"({type(first_errors[setup_id]).__name__}: "
                f"{first_errors[setup_id]})"
            )
            for setup_id in failed_all
        )
        raise RuntimeError(
            "setup rule failed for every evaluated row: " + details
        )

    if log_partial_errors:
        for setup_id, count in error_counts.items():
            if count == 0:
                continue
            exc = first_errors[setup_id]
            log.warning(
                "setup rule %s failed for %d/%d rows (%s: %s)",
                setup_id,
                count,
                row_count,
                type(exc).__name__,
                exc,
            )

    return matched, error_counts


def get_setup_summary(setup_ids: list[str]) -> str:
    """Format setup ids as readable string for alerts."""
    if not setup_ids:
        return "(no setup)"
    parts = []
    for s_id in setup_ids:
        if s_id in SETUP_LIBRARY:
            parts.append(f"{s_id} {SETUP_LIBRARY[s_id]['name']}")
    return ", ".join(parts)
