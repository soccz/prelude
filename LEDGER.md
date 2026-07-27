# LEDGER.md — 가상 reference ledger

> **현재 운영: R1 preopen/open과 pump v2의 분리된 reference ledger.**
> 시스템은 추천·기록만 하고 실제 매매는 사용자가 직접 한다.

**현재 운영 원장**:
- R1 open: `output/shadow_ledger_recommend.csv`
- R1 preopen: `output/shadow_ledger_recommend_preopen.csv`
  (첫 compliant preopen run 전에는 파일이 없을 수 있음)
- R2/A1/pump v1/v2: 각 모델의 별도 `output/shadow_ledger_*.csv`
- 성공 delivery receipt 이후 다음 실행 가능한 15분봉부터 96봉을 평가하고
  TP5/SL3/EOD·왕복 0.15% 비용을 기록
- 같은 날 추천은 equal-weight, 무추천일은 cash 0%로 집계

통합 `output/ledger.csv`는 현재 생성되지 않는다. 아래 §0L~§9는 원래 Phase 1
TP/SL 통합 ledger 설계이며, 일부 모듈만 남은 **legacy·미배선 문서**다.

---

## 0L. Legacy 통합 ledger 설계

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

`scripts/tp_sl_sweep.py` 자동 실행은 과거 계획이며 현재 스크립트도 scheduler도 없다.

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

## 5. Legacy ledger 데이터 (`output/ledger.csv`, 현재 미생성)

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

설계상 각 행은 한 가상 포지션 한 사이클이며 KST 09:30 갱신 예정이었지만,
현재 운영은 상단의 채널별 원장을 사용한다.

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

### 6.2 주간 / 월간 리포트 (legacy 계획, 스크립트 미구현)
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
- IC (Spearman, P(≥10%) vs 실제 max) → `output/ic_history.json` (legacy 계획, 현재 미생성)
- ICIR

**우선순위**: 핵심 트레이딩 메트릭으로 결정. 학술 옆 표기만 (CLAUDE.md §2.3).

---

## 7. 사용자 실제 매매와 비교 (legacy 계획)

### 7.1 NOTES.md 형식 (사용자 손글)
사용자가 매일 NOTES.md에 적는 방식은 유지하지만 `scripts/ledger_vs_user.py`는
현재 구현되지 않았다.

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

## 8. kill switch / 리스크 한도 (legacy·미배선, ledger/risk.py)

### 8.1 일일 한도
- **일일 가상 손실 -3% 는 초기값** (CLAUDE.md §2.5)
- 발동 시 다음 날 자동 침묵 (알림 X, 가상 진입 X)
- 사용자 텔레그램 경고
- 사용자 명시적 "재개" 명령 전까지 침묵

### 8.2 MDD 한도
- **누적 MDD -15% 는 초기값** (CLAUDE.md §2.5)
- 발동 시 7 일 cool-down (가상 watch-only)

### 8.3 drift 발동 시
- `ops/drift_detector.py` 구현은 남아 있지만 production evaluator 호출자는 없어
  자동 watch-only로 배선되지 않았다

### 8.4 사용자 직접 매매에는 강제 X
이 한도는 **가상 ledger 한정**. 사용자 본인은 본인 판단.

---

## 9. ledger ↔ 텔레그램 일관성 검증 (legacy·미등록)

xsec_alpha 의 verify_telegram.py 패턴: 텔레그램이 ledger 와 부호 / 크기 일치하는지 자동 검증.

`scripts/verify_telegram.py`는 legacy `output/ledger.csv`용 수동 검사이며 현재
systemd timer에서 호출되지 않는다. 현재 R1/pump 경로는 snapshot/decision,
delivery receipt, 전용 ledger identity를 쓰기·청산 단계에서 fail closed로 검증한다.

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
| `output/shadow_ledger_recommend.csv` | 현재 R1 open 전용 원장 |
| `output/shadow_ledger_recommend_preopen.csv` | 현재 R1 preopen 전용 원장 (첫 실행 전 미생성 가능) |
| `output/shadow_ledger_pump_hunter_v2.csv` | 현재 pump v2 원장 |
| `scripts/close_recommend_ledger.py` | receipt 이후 15m 경로 청산 + `skip-no-decision` 검증·마커 기록 |
| `ledger/path_quality.py` | 15m 경로 완전성 판정 (`assess_15m_window` 단건 = `assess_15m_windows` 날짜별 벌크 로더, 동등성 테스트 고정) |
| `output/close_no_decision/{cohort}/{asof}.json` | 발송 파이프가 아예 죽어 증거·원장 행이 전무했던 날의 감사 마커 (커버리지 분모 보정용, 백업 포함) |
| `ledger/portfolio_metrics.py` | day-equal/cash-day 포트폴리오 지표 |
| `ledger/exit_lab.py` | record-only 청산 변형 |
| `ops/champion_selector.py` | 채널별 forward champion 판정 |
| `ops/policy_competition.py` | 모델·정책·청산 근거 비교 |
| `ledger/config.py` | legacy 가상 자본·사이징·TP/SL 상수 |
| `ledger/sizing.py` | legacy sizing (운영 미배선) |
| `ledger/tracker.py` | legacy 4h TP/SL 시뮬 |
| `ledger/risk.py` | legacy kill switch (운영 미배선) |
| `ledger/metrics.py` | legacy 통합 ledger 지표 |
| `scripts/verify_telegram.py` | legacy 수동 검사 |
