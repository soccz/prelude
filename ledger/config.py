"""가상 ledger 설정 — 모든 숫자 placeholder (CLAUDE.md §2.5).

데이터 보고 조정. NOTES.md 에 사용자 본인 매매 기준 따로 적기.
"""
from __future__ import annotations

# ============================================================================
# 가상 자본 — 사용자 조정 가능
# ============================================================================
INITIAL_CAPITAL_KRW = 10_000_000   # 1,000 만원

# ============================================================================
# 포지션 사이징 (LEDGER §2)
# ============================================================================
SIZING_RULE = "equal"               # 'equal' / 'prob_weighted' / 'kelly_quarter'
K_POSITIONS = 3                     # top-K (placeholder)
MAX_POSITION_SIZE_PCT = 0.05        # per-position 5%
MAX_TOTAL_EXPOSURE_PCT = 0.60       # max 60%

# BTC regime 별 exposure cut (placeholder)
REGIME_EXPOSURE_CAP = {
    "bull_quiet": 0.60,
    "bull_volatile": 0.40,
    "bear_quiet": 0.30,
    "bear_volatile": 0.00,           # 침묵
    "unknown": 0.30,
}

# ============================================================================
# 진입 / 익절 / 손절 / 청산 (LEDGER §3)
# ============================================================================
TP_PCT = 0.10                       # +10% 익절
SL_PCT = 0.05                       # -5% 손절
MAX_HOLD_HOURS = 24                 # 24h 후 강제 청산

# 같은 봉에 TP, SL 둘 다 트리거 시: SL 먼저 (보수)
SL_PRIORITY = True

# ============================================================================
# 거래비용 (placeholder) — ★ 단일 출처. ops/scripts 는 여기서 import 만 한다.
#   decimal 단위는 *_PCT (0.0015), percentage-point 단위는 *_PP (0.15) 로 구분
#   (2026-06-11 이전 ops 2곳이 같은 이름을 %p 단위로 재정의하던 footgun 제거).
# ============================================================================
FEE_PCT = 0.0005                     # 업비트 일반 0.05%
SLIPPAGE_PCT = 0.00025               # 편도 0.025% → 왕복 슬리피지 0.05%
# 왕복 = 2 × (FEE + SLIPPAGE) = 0.15%  (decimal 0.0015)
ROUND_TRIP_COST_PCT = 2 * (FEE_PCT + SLIPPAGE_PCT)
ROUND_TRIP_COST_PP = ROUND_TRIP_COST_PCT * 100.0   # %p 단위 (0.15)

# 보수 시나리오 — 펌프/저유동 코인의 현실 슬리피지 (편도 0.2%, placeholder §2.5).
# 표준 가정 (편도 0.025%) 은 변동성 폭발 구간에서 낙관적 → 평가 보고에 보수 net
# 병기용 두 번째 tier. 운영 ledger 차감은 표준 tier 유지.
CONSERVATIVE_SLIPPAGE_PCT = 0.002
ROUND_TRIP_COST_CONSERVATIVE_PCT = 2 * (FEE_PCT + CONSERVATIVE_SLIPPAGE_PCT)  # 0.005
ROUND_TRIP_COST_CONSERVATIVE_PP = ROUND_TRIP_COST_CONSERVATIVE_PCT * 100.0    # 0.5

# ============================================================================
# Kill switch (LEDGER §8)
# ============================================================================
DAILY_LOSS_LIMIT_PCT = 0.03         # 일일 -3% → 다음 날 침묵
MDD_LIMIT_PCT = 0.15                # MDD -15% → 7일 cool-down
COOLDOWN_DAYS = 7

# ============================================================================
# 시그널 게이트 (OPS §3.4 침묵 조건)
# ============================================================================
MIN_P_GE_5_FOR_RECOMMEND = 0.5      # P(≥5%) ≥ 50% 인 코인 0개 면 침묵

# ============================================================================
# 출력 path
# ============================================================================
LEDGER_CSV = "output/ledger.csv"
SUMMARY_JSON = "output/ledger_summary.json"
