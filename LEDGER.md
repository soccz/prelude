# LEDGER.md — 가상 reference ledger

> **현재 운영: detector_v1 detector beta. ledger 는 "추천 그대로 따라갔으면" reference 만.**
> 시스템 = 텔레그램 알림. 실제 매매는 사용자 직접. ledger 는 시스템 정직성 검증용 reference.
> 자동매매 0. 백테스트 EV(+7.4% 평균, 2024 -0.89%)는 운영 보장 X — Stage 1/2 라이브 데이터로 재확인.

**detector beta 단계 (현재) ledger 역할**:
- alerts → 가상 reference ledger 자동 누적 (TP=+20%, EOD close, 왕복 0.15%)
- 사용자 실제 매매 NOTES 와 비교 (시스템 가상 vs 사용자 실제)
- backtest 와 라이브 reference 의 EV 차이 추적 (백테스트 → 라이브 drift 감지)
- **시스템 사이징/익절 룰은 사용자에게 강제 X** — 사용자 본인 룰

아래 §1~§9 는 **TP/SL 옵션 3** 기반 가상 ledger 인프라 (legacy + reference) — 자동매매 X 원칙 유지.

---

## 0. 한 줄 결론

```
SIGNAL → top-K 코인 + 분포 (P(≥5%/10%/15%/20%)) + 기대 max + CI
   ↓
가상 사이징 (균등 1/K 또는 prob_weighted)
   ↓
시가 진입 (KST 09:00, 다음날)
   ↓
익절 / 손절 hold (옵션 3)
  - +TP_pct 도달 시 즉시 익절
  - -SL_pct 도달 시 즉시 손절
  - 둘 다 안 도달 → 다음 09:00 종가 청산
   ↓
거래비용 차감 (왕복 0.15%)
   ↓
output/ledger.csv 자동 누적 + 일일 텔레그램 리포트
   ↓
사용자 실제 매매 (NOTES.md 손글) 와 비교
```

---

## 1. 모드

### Mode A — 알림만 (기본)
시스템 = 텔레그램 알림. 가상 ledger X. 사용자가 알림 보고 본인 매매 + NOTES.

### Mode B — 가상 ledger 자동 추적 (메인)
시스템 = 알림 + **가상 포지션 자동 진입 / 익절·손절 / 청산 / 성과 추적**. 실거래 X.

### Mode C — 실거래 자동 (영원히 사용자 명시 전 X)
업비트 API 자동 주문. **이 프로젝트에선 영원히 안 함** (CLAUDE.md §3.1).

**기본값**: Mode A + B 동시.

---

## 2. 가상 사이징 (ledger/sizing.py)

### 2.1 옵션
| 룰 | 설명 | 장단점 |
|---|---|---|
| **equal** | top-K 균등 1/K | 단순, 안정 |
| **prob_weighted** | P(≥10%) 비례 | 분포 정보 활용 |
| **expected_weighted** | 기대 max 비례 | expected 가 부정확하면 위험 |
| **kelly_quarter** | quarter-Kelly fraction | 이론적, 추정 오차 민감 |

**Phase 1 시작**: `equal` (단순). Phase 2 에서 `prob_weighted` 비교.

### 2.2 K (포지션 수)
- **K = 3 은 초기값** (CLAUDE.md §2.5). EDA K sweep:
  - K = 1 / 2 / 3 / 5 가상 ledger 백테스트 → net Sharpe / MDD / turnover
  - 시그널 적게 나오면 K 작아도 OK
- 시그널 많아도 K cap (집중 위험) — cap 도 placeholder

### 2.3 max position size
- per-position cap **5% 는 초기값** (CLAUDE.md §2.5). 가상 ledger 4 주 후 분포 보고 조정.

### 2.4 total exposure
- max total exposure **60% 는 초기값** (CLAUDE.md §2.5)
- BTC regime 따라 동적 조정 — **각 regime exposure 도 모두 placeholder**:
  - bull_quiet → 초기 60%
  - bull_volatile → 초기 40%
  - bear_quiet → 초기 30%
  - bear_volatile → 초기 0% (침묵)
- 데이터 보고 조정 (예: bear_volatile 도 가상 결과 좋으면 0 → 10%)

---

## 3. 진입 / 익절 / 손절 / 청산 (ledger/tracker.py)

### 3.1 옵션 3 — TP/SL hold

```python
def simulate_position(market, entry_time, entry_price, tp_pct, sl_pct, max_hold_hours=24):
    """
    옵션 3: 익절 / 손절 / hold cap.
    
    - entry_time = KST 09:00 (그날 일봉 시가)
    - entry_price = open_today
    - 4h candle 데이터로 시뮬:
      - 매 봉 high / low 체크
      - high >= entry_price * (1 + tp_pct) → 익절 (그 시점)
      - low  <= entry_price * (1 - sl_pct) → 손절 (그 시점)
    - 둘 다 안 도달 → max_hold (24h) 후 다음 09:00 종가 청산
    
    return: { exit_time, exit_price, exit_type ('tp'/'sl'/'eod'), gross_return, hold_hours }
    """
```

### 3.2 우선순위 (같은 봉에 TP, SL 둘 다 트리거 시)
- 4h 봉 단위 시뮬 → 정확도 한계 있음
- **보수적 가정**: 같은 4h 봉에 high 가 TP 도달 + low 가 SL 도달이면 → **SL 먼저 적용 (보수)**
  - 이유: 실거래에서 SL 이 먼저 trigger 되는 경우 많음 (sudden dip)
  - 옵션: 1h 봉 데이터 추가하면 정확도 ↑ (Phase 2 옵션)

### 3.3 TP_pct / SL_pct 도 placeholder

**초기값 (CLAUDE.md §2.5)**:
- `TP_pct = 0.10` (10% 익절)
- `SL_pct = 0.05` (5% 손절)

**가상 ledger 백테스트 sweep**:
- TP {0.05, 0.08, 0.10, 0.12, 0.15} × SL {0.03, 0.05, 0.07, 0.10}
- 각 조합:
  - net Sharpe (왕복 0.15% 차감)
  - hit rate (TP 도달 비율)
  - 평균 hold 시간
  - MDD
- 최적 조합 채택 (또는 사용자 본인 매매 룰 결정 시 참고)

`scripts/tp_sl_sweep.py` 매주 자동 실행 → 최적 TP/SL 추천.

### 3.4 max_hold (24h) cap
24h 후 강제 청산. 이것도 placeholder (12h / 36h 도 비교 가능).

---

## 4. 거래비용 모델

- **사용자 본인 등급 확인** (업비트 일반 0.05%, VIP 더 낮음)
- 슬리피지 **실제 측정 필요** — 사용자 NOTES 의 시가 vs 체결가 차이 통계
- **초기값 왕복 0.15%** (수수료 0.1% + 슬리피지 0.05%, CLAUDE.md §2.5)
- TP/SL trigger 시에도 동일 비용 차감 (시장가 주문 가정)
- 거래비용 차감 자체는 양보 X (위생) — but 수치는 데이터 기반 조정

---

## 5. ledger 데이터 (output/ledger.csv)

스키마:
```csv
date,coin,btc_regime,
  signal_p_ge_5,signal_p_ge_10,signal_p_ge_15,signal_p_ge_20,
  signal_expected_max,signal_ci_low,signal_ci_high,
  position_size_pct,
  entry_price,entry_time,
  exit_type,exit_price,exit_time,hold_hours,
  gross_return_pct,cost_pct,net_return_pct,
  realized_pnl_krw,cumulative_pnl_krw,equity_krw
```

각 행 = 한 가상 포지션 한 사이클. 매일 KST 09:30 (어제 추천 → 오늘 진입 시뮬 후) 자동 갱신.

**가상 자본 시작**: 1,000 만원 (사용자 조정 가능, `ledger/config.py`)

---

## 6. 성과 메트릭 (ledger/metrics.py)

### 6.1 핵심 (매일 텔레그램 리포트)
- **누적 net PnL** (원 + %)
- **이번 주 net Sharpe**
- **TP 적중률** (전체 / per coin / per BTC regime)
- **SL 적중률**
- **평균 hold 시간** (몇 시간 안에 청산?)
- **현재 MDD**
- **현재 capital deployment** (가상 노출)

### 6.2 주간 / 월간 리포트 (scripts/ledger_summary.py)
```
=== Week 18 (2026-05-04 ~ 05-10) ===
- 추천 코인: 12 (15 알림 중 3 침묵 - bear_volatile)
- net PnL: +47.3 만원 (+4.7%)
- net Sharpe: 2.34
- TP 적중: 8/12 (67%) — 평균 hold 4.2h
- SL 적중: 2/12 (17%) — 평균 hold 1.8h
- EOD 청산: 2/12 (17%) — 평균 +1.3%
- MDD: -3.2%
- Best: KAITO TP +10% in 3h
- Worst: NMR SL -5% in 2h

시스템 정확도:
  ≥+5% 예측 70% → 실제 75% ✓
  ≥+10% 예측 45% → 실제 50% ✓
```

### 6.3 학술 메트릭 (사후 진단)
- IC (Spearman, P(≥10%) vs 실제 max) → `output/ic_history.json`
- ICIR

**우선순위**: 핵심 트레이딩 메트릭으로 결정. 학술 옆 표기만 (CLAUDE.md §2.3).

---

## 7. 사용자 실제 매매와 비교

### 7.1 NOTES.md 형식 (사용자 손글)
사용자가 매일 NOTES.md 에 적는 자유 형식 — `scripts/ledger_vs_user.py` 가 파싱.

### 7.2 비교 스크립트
주 1 회 NOTES 파싱 → 사용자 실제 ledger 추출 → 시스템 가상 ledger 와 비교:

```
=== 시스템 가상 vs 사용자 실제 (지난 주) ===
- 시스템 가상 net: +4.7% (TP +10% / SL -5%)
- 사용자 실제 net: +6.1% (사용자 룰 다름)
- 일치 진입: 8/12 추천
- 사용자 익절 라인: 평균 +8% (시스템보다 보수적)
- 사용자 손절 라인: 평균 -3% (시스템보다 보수적)
- 사용자 skip 한 시스템 추천: 4 개
  - 그 중 사후 적중: 2 개 (사용자 보수적 → 기회 손실)
  - 그 중 사후 실패: 2 개 (사용자 옳음)
```

이 비교가 사용자 trading skill 측정. 시간 지나면 패턴 발견.

---

## 8. kill switch / 리스크 한도 (ledger/risk.py)

### 8.1 일일 한도
- **일일 가상 손실 -3% 는 초기값** (CLAUDE.md §2.5)
- 발동 시 다음 날 자동 침묵 (알림 X, 가상 진입 X)
- 사용자 텔레그램 경고
- 사용자 명시적 "재개" 명령 전까지 침묵

### 8.2 MDD 한도
- **누적 MDD -15% 는 초기값** (CLAUDE.md §2.5)
- 발동 시 7 일 cool-down (가상 watch-only)

### 8.3 drift 발동 시
- ops/drift_detector.py 가 sign flip / 정확도 drop 감지 → 자동 watch-only

### 8.4 사용자 직접 매매에는 강제 X
이 한도는 **가상 ledger 한정**. 사용자 본인은 본인 판단.

---

## 9. ledger ↔ 텔레그램 일관성 검증

xsec_alpha 의 verify_telegram.py 패턴: 텔레그램이 ledger 와 부호 / 크기 일치하는지 자동 검증.

`scripts/verify_telegram.py`:
- 매일 KST 09:30 (가상 진입 후) 자동
- 어제 텔레그램 vs 오늘 ledger 비교
- 부호 / 크기 / 코인 일치
- 불일치 → 텔레그램 alert

---

## 10. 책임 경계

| 이건 LEDGER 의 일 | 이건 LEDGER 의 일 X |
|---|---|
| 가상 사이징 / K / max exposure | 시그널 분포 → SIGNAL |
| 가상 진입 / 익절 / 손절 / 청산 시뮬 | 라벨 정의 → SIGNAL |
| TP / SL sweep | 모델 학습 → SIGNAL |
| 거래비용 모델 | 텔레그램 메시지 포맷 → OPS |
| 가상 ledger 누적 | cron 스케줄 → OPS |
| 성과 메트릭 (PnL, Sharpe, TP/SL hit) | drift 감지 → OPS |
| kill switch (가상 한정) | calibration → SIGNAL |
| 사용자 매매와 비교 (NOTES 파싱) | 실거래 자동 (영원히 X) |

---

## 11. 핵심 파일 인덱스

| 파일 | 역할 |
|---|---|
| `ledger/config.py` | 가상 자본, K, max exposure, TP/SL 초기값 |
| `ledger/sizing.py` | equal / prob_weighted / kelly |
| `ledger/tracker.py` | 진입 / 익절 / 손절 / 청산 시뮬 (4h 봉 기반) |
| `ledger/risk.py` | kill switch / MDD 한도 |
| `ledger/metrics.py` | Sharpe / MDD / TP-SL hit rate |
| `output/ledger.csv` | 누적 가상 ledger |
| `scripts/ledger_summary.py` | 주간 / 월간 리포트 |
| `scripts/ledger_vs_user.py` | 시스템 vs 사용자 비교 |
| `scripts/tp_sl_sweep.py` | TP/SL 최적 조합 sweep |
| `scripts/verify_telegram.py` | ledger ↔ telegram 일관성 |
