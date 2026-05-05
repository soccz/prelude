"""Setup filter — daily 09:05 시스템용.

검증 결과 (2026-05-03):
  - quiet+contraction = anti-signal on daily (전체 절반 hit rate)
  - momentum-continuation = +2.7x alpha on daily (≥10% hit rate 21% vs 8%)

→ **daily 시스템은 momentum-continuation 사용** (xsec_alpha 의 quiet→pump 는 12h/1h 시간봉 단위 alpha).
→ quiet-contraction 은 폐기 X — **4h/1h intraday alert 트랙 보류** (별도 시스템).

설계:
  - momentum_candidate (메인, daily 09:05): vol↑ + return_7d↑ + volume↑ + BTC ≠ bear_volatile
  - quiet_candidate (보류, intraday 트랙): 기존 SetupFilterConfig 유지 — 4h/1h 봉에서 재시도
  - 모델은 candidate 안에서만 ranking (random within candidate vs model)
  - threshold: train-only quantile 또는 cross-sectional rank per-day (전기간 global X)

config (placeholder, CLAUDE.md §2.5 — 모두 sweep 가능):
  - vol_pct_max:        14d vol cross-sectional rank ≤ X (조용)
  - range_contraction_min:  range 7d 압축 임계
  - require_squeeze:    squeeze_on 강제 / 완화
  - volume_revival_min: volume_ratio_3d ≥ X (살짝 살아남)
  - roc_3d_min:         roc_3d ≥ X (반전 신호)
  - return_7d_max:      return_7d ≤ X (이미 오른 거 제외)
  - btc_regimes:        허용 regime 셋
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


# ============================================================================
# Filter config
# ============================================================================
@dataclass
class SetupFilterConfig:
    """
    임계는 모두 cross-sectional rank (0~1) 기준 — features.py 의
    cross_sectional_normalize 후 panel 사용 가정.

    의미:
      - vol_pct_max=0.30: vol 하위 30% (조용한 코인)
      - range_contraction_min=0.70: range_contraction rank 상위 30% (압축 강함)
      - volume_revival_min=0.50: volume rank 중간 이상 (살짝 살아남)
      - roc_3d_min=0.50: roc rank 중간 이상 (반전 시작)
      - return_7d_max=0.80: 7d return rank 상위 20% 제외 (이미 오른 거 X)
    """
    # 조용 (low vol)
    vol_pct_max: float = 0.30
    vol_col: str = "vol_14d"

    # 압축 (range contraction rank 상위)
    range_contraction_min: float = 0.70
    range_contraction_col: str = "range_contraction_7d"
    range_contraction_alt_col: str = "range_contraction_14d"
    range_contraction_alt_min: float = 0.80

    # squeeze
    require_squeeze: bool = True
    relax_squeeze_with_alt: bool = True

    # 압력 (volume revival + ROC reversal)
    volume_revival_min: float = 0.50
    volume_revival_col: str = "volume_ratio_3d"
    roc_3d_min: float = 0.50
    roc_3d_col: str = "roc_3d"

    # 너무 이미 오른 거 X
    return_7d_max: float = 0.80
    return_7d_col: str = "return_7d"

    # BTC regime gate
    btc_regimes: tuple = ("bull_quiet", "bull_volatile", "bear_quiet")
    btc_regime_col: str = "btc_regime"


DEFAULT_CONFIG = SetupFilterConfig()


# ============================================================================
# Momentum candidate (daily 09:05 시스템 메인)
# ============================================================================
@dataclass
class MomentumCandidateConfig:
    """
    Daily next-day pump 의 momentum-continuation candidate.

    전체 panel 검증 (510K rows, 274 days, 2025-11~2026-05):
      momentum candidate (n=21K):
        평균 max_return 7.47% (전체 4.06%, 1.84x)
        ≥10% 비율 21.1% (전체 7.86%, 2.7x)
        ≥15% 비율 11.7% (전체 3.54%, 3.3x)

    임계 = cross-sectional rank top X (train-only quantile, 매일 갱신).
    """
    vol_pct_min: float = 0.70           # 14d vol cross-section top 30%
    vol_col: str = "vol_14d"

    return_7d_min: float = 0.70         # 7d return cross-section top 30%
    return_7d_col: str = "return_7d"

    volume_revival_min: float = 0.70    # 3d volume cross-section top 30%
    volume_revival_col: str = "volume_ratio_3d"

    btc_regimes_excluded: tuple = ("bear_volatile",)
    btc_regime_col: str = "btc_regime"


DEFAULT_MOMENTUM_CONFIG = MomentumCandidateConfig()


def is_momentum_candidate(row: pd.Series, config: MomentumCandidateConfig = DEFAULT_MOMENTUM_CONFIG) -> bool:
    """한 row 가 momentum candidate 인가."""
    if pd.isna(row.get(config.vol_col)) or row[config.vol_col] < config.vol_pct_min:
        return False
    if pd.isna(row.get(config.return_7d_col)) or row[config.return_7d_col] < config.return_7d_min:
        return False
    if pd.isna(row.get(config.volume_revival_col)) or row[config.volume_revival_col] < config.volume_revival_min:
        return False
    regime = row.get(config.btc_regime_col, "unknown")
    if regime in config.btc_regimes_excluded:
        return False
    return True


def filter_momentum_panel(
    panel: pd.DataFrame,
    config: MomentumCandidateConfig = DEFAULT_MOMENTUM_CONFIG,
) -> pd.DataFrame:
    """Panel → momentum candidate 만."""
    mask = panel.apply(lambda r: is_momentum_candidate(r, config), axis=1)
    return panel[mask].copy()


# ============================================================================
# 단일 row 필터
# ============================================================================
def is_setup_candidate(row: pd.Series, config: SetupFilterConfig = DEFAULT_CONFIG, use_alt: bool = False) -> bool:
    """
    한 행이 candidate setup 인가.

    use_alt: True 면 squeeze 대신 range_contraction_alt 임계 사용 (sparse 완화).
    """
    # vol cross-sectional rank — 이미 normalize 된 값 (0~1)
    if pd.isna(row.get(config.vol_col)):
        return False
    if row[config.vol_col] > config.vol_pct_max:
        return False

    # 압축 (rank 상위 = 압축 강함)
    rc7 = row.get(config.range_contraction_col)
    if pd.isna(rc7):
        return False

    if use_alt:
        rc14 = row.get(config.range_contraction_alt_col)
        if pd.isna(rc14) or rc14 < config.range_contraction_alt_min:
            return False
    else:
        if rc7 < config.range_contraction_min:
            return False
        if config.require_squeeze:
            sq = row.get("squeeze_on", 0)
            if sq != 1:
                return False

    # 압력
    vol_rev = row.get(config.volume_revival_col)
    if pd.isna(vol_rev) or vol_rev < config.volume_revival_min:
        return False

    roc = row.get(config.roc_3d_col)
    if pd.isna(roc) or roc < config.roc_3d_min:
        return False

    # 이미 오른 거 제외 (rank top 20% 제외)
    r7 = row.get(config.return_7d_col)
    if pd.isna(r7):
        return False
    if r7 > config.return_7d_max:
        return False

    # BTC regime gate
    regime = row.get(config.btc_regime_col, "unknown")
    if regime not in config.btc_regimes:
        return False

    return True


# ============================================================================
# Panel 단위 filter
# ============================================================================
def filter_setup_panel(
    panel: pd.DataFrame,
    config: SetupFilterConfig = DEFAULT_CONFIG,
    auto_relax: bool = True,
    min_candidates_per_day: int = 1,
) -> pd.DataFrame:
    """
    Panel → candidate setup 만 남김.

    auto_relax: 그 날의 candidate 가 min_candidates_per_day 미만이면
                squeeze 완화 (range_contraction_14d 상위로 대체).
    return: candidate panel (column 동일, sparse 행)
    """
    out = panel.copy()
    # 1차 strict filter
    mask = out.apply(lambda r: is_setup_candidate(r, config, use_alt=False), axis=1)
    candidates = out[mask].copy()

    if not auto_relax:
        return candidates

    # 일별 candidate 수 체크 → 부족 일 → relaxed filter 추가
    if "timestamp" in out.columns:
        cand_per_day = candidates.groupby("timestamp").size()
        sparse_days = cand_per_day[cand_per_day < min_candidates_per_day].index

        if len(sparse_days) > 0:
            # sparse day 의 row 에 대해 alt filter 적용
            sparse_panel = out[out["timestamp"].isin(sparse_days)]
            mask_alt = sparse_panel.apply(
                lambda r: is_setup_candidate(r, config, use_alt=True), axis=1
            )
            relaxed = sparse_panel[mask_alt]
            candidates = pd.concat([candidates, relaxed]).drop_duplicates(
                subset=["market", "timestamp"]
            )
    return candidates


# ============================================================================
# Filter 진단 (시뮬 후 사용)
# ============================================================================
def filter_summary(panel: pd.DataFrame, candidates: pd.DataFrame) -> dict:
    total = len(panel)
    n_cand = len(candidates)
    cand_per_day = candidates.groupby("timestamp").size() if "timestamp" in candidates.columns else pd.Series()

    return {
        "total_rows": total,
        "candidate_rows": n_cand,
        "filter_pass_rate": n_cand / total if total > 0 else 0,
        "candidates_per_day": {
            "mean": float(cand_per_day.mean()) if len(cand_per_day) > 0 else 0,
            "median": float(cand_per_day.median()) if len(cand_per_day) > 0 else 0,
            "min": int(cand_per_day.min()) if len(cand_per_day) > 0 else 0,
            "max": int(cand_per_day.max()) if len(cand_per_day) > 0 else 0,
            "n_zero_days": int((cand_per_day == 0).sum()),
            "n_days": len(cand_per_day),
        },
    }


# ============================================================================
# 직접 실행 — filter 분포 분석 (EDA)
# ============================================================================
if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data.database import list_markets, load_candles
    from signals.features import assemble_training_panel

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/upbit_d1.db")
    parser.add_argument("--btc-market", default="KRW-BTC")
    parser.add_argument("--limit", type=int, default=100, help="샘플 코인 갯수")
    args = parser.parse_args()

    print(f"loading {args.limit} markets...")
    markets = list_markets(args.db)[: args.limit]
    candles = {m: load_candles(args.db, m) for m in markets}
    candles = {k: v for k, v in candles.items() if len(v) > 30}
    btc = load_candles(args.db, args.btc_market)

    print("building panel...")
    panel = assemble_training_panel(candles, btc, normalize=True)
    panel = panel.dropna(subset=["label"]).copy()
    print(f"  panel: {panel.shape}")

    # default filter
    print("\n=== default filter (strict) ===")
    cands = filter_setup_panel(panel, auto_relax=False)
    s = filter_summary(panel, cands)
    print(f"  pass rate: {s['filter_pass_rate']*100:.2f}% ({s['candidate_rows']:,}/{s['total_rows']:,})")
    print(f"  candidates per day: mean {s['candidates_per_day']['mean']:.1f}, "
          f"median {s['candidates_per_day']['median']:.0f}, "
          f"min {s['candidates_per_day']['min']}, max {s['candidates_per_day']['max']}, "
          f"zero days {s['candidates_per_day']['n_zero_days']}/{s['candidates_per_day']['n_days']}")

    # auto-relax
    print("\n=== with auto-relax ===")
    cands = filter_setup_panel(panel, auto_relax=True, min_candidates_per_day=1)
    s = filter_summary(panel, cands)
    print(f"  pass rate: {s['filter_pass_rate']*100:.2f}% ({s['candidate_rows']:,}/{s['total_rows']:,})")
    print(f"  candidates per day: mean {s['candidates_per_day']['mean']:.1f}, "
          f"median {s['candidates_per_day']['median']:.0f}, "
          f"min {s['candidates_per_day']['min']}, max {s['candidates_per_day']['max']}")

    # Label 분포 (candidate 만)
    if "label" in cands.columns:
        print("\n=== candidate label 분포 ===")
        print(cands["label"].value_counts().sort_index().to_string())
        print(f"\n  candidate 평균 max_return: {cands['max_return'].mean():.4f}")
        # 비교: 전체 평균
        print(f"  전체 panel 평균 max_return: {panel['max_return'].mean():.4f}")
        # ≥10% 비율
        print(f"  candidate 중 ≥10% 비율: {(cands['max_return'] >= 0.10).mean()*100:.2f}%")
        print(f"  전체 중   ≥10% 비율: {(panel['max_return'] >= 0.10).mean()*100:.2f}%")
