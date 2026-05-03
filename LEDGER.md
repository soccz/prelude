# LEDGER.md — 가상 포트폴리오 추적

> 시스템은 **알림 + 가상 ledger 만**. 실제 매매는 사용자 직접. 이 문서는 "**시스템이 추천대로 가상 자본을 굴렸으면 어떻게 됐을까**" 자동 추적부.

---

## 0. 한 줄 결론

```
SIGNAL → top-K 코인 + 확률 + tier
   ↓
가상 사이징 (균등 1/K 또는 σ-tier 기반)
   ↓
시가 진입 (KST 09:00) → 종가 청산 (다음 KST 09:00)
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
시스템 = 텔레그램 알림 보내기만. 가상 ledger 도 X. 사용자가 알림 보고 본인 결정 + 본인 매매 + NOTES 에 기록.

### Mode B — 가상 ledger 자동 추적 (메인)
시스템 = 알림 + **가상 포지션 자동 진입 / 청산 / 성과 추적**. 실거래 X (사용자 매매는 별개로 NOTES). 시스템 가상 vs 사용자 실제 둘 다 비교 가능.

### Mode C — 실거래 자동 (영원히 사용자 명시 전 X)
업비트 API 자동 주문. **이 프로젝트에선 영원히 안 함** (사용자 결정 전까지). CLAUDE.md §3.1 절대 금지.

**기본값**: Mode A + B 동시 — 알림 받고 사용자 직접 매매, 시스템은 가상 ledger 로 자체 추적.

---

## 2. 가상 사이징 (ledger/sizing.py)

### 2.1 옵션
| 룰 | 설명 | 장단점 |
|---|---|---|
| **equal** | top-K 균등 1/K | 단순, 안정. 신호 강도 무시 |
| **sigma_tier** | 🔥 3% / ✅ 2% / ▫ 1% (xsec_alpha 패턴) | 신호 강도 반영. tier cutoff 임의성 |
| **score_weighted** | 확률 점수 비례 | 연속적. 점수 차이 작으면 의미 적음 |
| **kelly_quarter** | quarter-Kelly fraction | 이론적. 추정 오차 민감 |

**Phase 1 시작**: `equal` (가장 단순). Phase 2 에서 `sigma_tier` 비교.

### 2.2 K (포지션 수)
- **K = 3 은 초기값** (CLAUDE.md §2.5). EDA 에서 K sweep 비교:
  - K = 1 / 2 / 3 / 5 가상 ledger 백테스트 → net Sharpe / MDD / turnover 비교
  - 시그널 적게 나오면 K 작아도 OK
- 시그널 많아도 K cap (집중 위험) — cap 도 placeholder (초기 5)

### 2.3 max position size
- per-position cap **5% 는 초기값** (CLAUDE.md §2.5). 가상 ledger 4 주 후 분포 보고 조정:
  - 한 코인이 자본의 X% 차지했을 때 MDD 영향 측정
  - 보수적 시작 후 결과 좋으면 7~10% 까지 완화 가능

### 2.4 total exposure
- max total exposure **60% 는 초기값** (CLAUDE.md §2.5)
- BTC regime 따라 동적 조정 — **각 regime 의 exposure 도 모두 placeholder**:
  - bull_quiet → 초기 60%
  - bull_volatile → 초기 40%
  - bear_quiet → 초기 30%
  - bear_volatile → 초기 0% (침묵)
- 실제 가상 ledger 결과 보고 조정:
  - 각 regime 별 평균 PnL / hit rate 분석 → 좋은 regime 은 exposure ↑, 나쁜 regime 은 ↓ 또는 0
  - 데이터가 "bear_volatile 도 일부 펌프 잡힘" 보여주면 0 → 10% 완화 OK

---

## 3. 진입 / 청산 (ledger/tracker.py)

### 3.1 단순 룰 (Phase 1)
- **진입**: 추천 발생 다음 봉 KST 09:00 시가
- **청산**: 진입 후 24h (다음 KST 09:00 시가) — hold 1 일
- 단순함의 끝. 라벨이 1 일 펌프 분류라 자연스러움

### 3.2 거래비용 모델
- 업비트 KRW 현물 수수료: **사용자 본인 등급 확인** (일반 0.05%, VIP 더 낮음, 결제수단 따라 다름)
- 슬리피지: **실제 측정 필요** — 사용자가 NOTES.md 에 실거래 시 시가 vs 체결가 차이 기록 → 통계 분석
- **초기값: 왕복 0.15%** (수수료 0.1% + 슬리피지 0.05% 보수, CLAUDE.md §2.5)
- Phase 1 라이브 시작 후 사용자 실제 매매 데이터 수집 → 실제 슬리피지 분포로 모델 갱신
- 거래비용 차감 자체는 양보 X (CLAUDE.md §2.5 위생) — but **수치는 데이터 기반 조정**

### 3.3 손절 / 익절 (Phase 2 옵션)
Phase 1 에선 단순 hold 1 일. Phase 2 에서 비교:
- 익절: +15% 도달 시 즉시
- 손절: -5% 도달 시 즉시
- 둘 다 도달 안 하면 24h 후 자동 청산

비교 후 net Sharpe / MDD 더 좋은 룰 채택.

### 3.4 부분 진입 / 청산 (Phase 3 옵션)
Phase 1/2 안 함. 단순 all-in / all-out.

---

## 4. ledger 데이터 (output/ledger.csv)

스키마:
```csv
date,coin,signal_score,signal_sigma,signal_tier,btc_regime,
  position_size_pct,entry_price,entry_time,
  exit_price,exit_time,gross_return_pct,cost_pct,net_return_pct,
  realized_pnl_krw,cumulative_pnl_krw,equity_krw
```

각 행 = 한 가상 포지션 한 사이클. 매일 KST 09:00 직후 (어제 추천 → 오늘 진입 → 내일 청산 시점) 자동 갱신.

**가상 자본 시작**: 1,000 만원 (사용자 조정 가능, `ledger/config.py`)

---

## 5. 성과 메트릭 (ledger/metrics.py)

### 5.1 핵심 (매일 텔레그램 리포트)
- **누적 net PnL** (원 + %)
- **win rate** (라벨 = 1 적중률 기반)
- **이번 주 net Sharpe**
- **현재 MDD** (peak 대비)
- **현재 capital deployment** (전 자본 중 몇 % 가상 포지션)

### 5.2 주간 / 월간 리포트 (scripts/ledger_summary.py)
```
=== Week 18 (2026-05-04 ~ 05-10) ===
- 추천 코인 수: 12 (15 알림 중 3 침묵)
- net PnL: +47.3 만원 (+4.7%)
- net Sharpe (annualized): 2.34
- hit rate: 67% (8/12)
- MDD: -3.2%
- Best: KAITO +18.4%
- Worst: NMR -4.1%
- BTC regime 분포: bull_quiet 4d / bull_volatile 2d / bear_quiet 1d
```

### 5.3 학술 메트릭 (사후 진단만)
- IC (일별 cross-sectional Spearman) → `output/ic_history.json`
- ICIR
- CRPS / PI_80 coverage (Phase 2 hybrid 모델 시)

**우선순위**: 핵심 트레이딩 메트릭으로 결정. 학술은 옆에 표기만 (CLAUDE.md §2.3).

---

## 6. 사용자 실제 매매와 비교

### 6.1 NOTES.md 형식 (사용자 손글)
사용자가 매일 NOTES.md 에 적는 형식 (자유):

```markdown
## 2026-05-03 KST 08:30 알림
- 시스템 추천: KAITO 78%, ETH 65%, SOL 62%
- 내가 진입: KAITO 30 만원 (시스템과 동일), SOL 안 들어감 (BTC 추세 의심)
- 진입 시각: 09:05
- 청산: KAITO 다음날 08:50 종가 직전 +12.3%
- 비고: SOL 도 +8% 갔음. 너무 보수적이었나?

## 2026-05-04 KST 08:30 알림
...
```

### 6.2 비교 스크립트 (scripts/ledger_vs_user.py)
주 1 회 실행. NOTES.md 파싱 → 사용자 실제 매매 ledger 추출 → 시스템 가상 ledger 와 비교:

```
=== 시스템 가상 vs 사용자 실제 (지난 주) ===
- 시스템 가상 net: +4.7%
- 사용자 실제 net: +6.1%
- 일치 진입: 8/12 추천 (67%)
- 사용자 skip 한 시스템 추천: 4 개
  - 그 중 사후 적중: 2 개 (사용자 보수적 → 기회 손실)
  - 그 중 사후 실패: 2 개 (사용자 옳음)
- 사용자 추가 진입 (시스템 추천 외): 1 개 → +3.2%
```

이 비교가 사용자 trading skill 도 측정해줌. 시간 지나면 패턴 (예: "사용자가 BTC volatile 때 보수적 → 옳음" 같은) 발견 가능.

---

## 7. kill switch / 리스크 한도 (ledger/risk.py)

### 7.1 일일 한도
- **일일 가상 손실 -3% 는 초기값** (CLAUDE.md §2.5)
- 발동 시 다음 날 자동 침묵 (알림 X, 가상 진입 X)
- 사용자 텔레그램 경고 + 사유 명시
- 사용자가 명시적으로 "재개" 명령 전까지 침묵 유지
- **조정 process**: Phase 1 라이브 4 주 후 일일 가상 PnL 분포 분석 → -3% 가 너무 자주 발동하면 -5%, 안 발동하면 -2% 로 조정

### 7.2 MDD 한도
- **누적 MDD -15% 는 초기값** (CLAUDE.md §2.5)
- 발동 시 7 일 cool-down (가상 ledger 자동 watch-only, 알림은 계속하되 가상 진입 X)
- 7 일 후 자동 재개
- **조정 process**: 가상 ledger MDD 분포 + 사용자 본인 위험 선호 보고 조정 (-10% / -20% 도 가능)

### 7.3 drift 발동 시
- ops/drift_detector.py 가 sign flip / 50% drop 감지 → 자동 watch-only 전환
- 사용자 컨펌 + 다음 retrain 트리거 후 재개

### 7.4 사용자 직접 매매에는 강제 X
이 한도는 **가상 ledger 한정**. 사용자 본인 매매는 본인 판단 (NOTES 에 본인이 한도 적기).

---

## 8. ledger ↔ 텔레그램 일관성 검증 (xsec_alpha 교훈)

xsec_alpha 의 verify_telegram.py 패턴: 텔레그램이 ledger 와 부호 / 크기 일치하는지 자동 검증. (이전에 LONG 부호 버그로 모든 행 반대 표시된 적 있음)

`scripts/verify_telegram.py` (prelude 안에 새로):
- 매일 KST 09:30 (가상 진입 후) 자동 실행
- ledger.csv 의 어제 행 vs 오늘 텔레그램 메시지 비교
- 부호 / 크기 / 코인 일치 확인
- 불일치 발견 시 텔레그램 alert

---

## 9. 책임 경계 (다른 MD 와의 분리)

| 이건 LEDGER 의 일 | 이건 LEDGER 의 일 X |
|---|---|
| 가상 사이징 / K / max exposure | 시그널 확률 → SIGNAL |
| 가상 진입 / 청산 룰 | 라벨 정의 → SIGNAL |
| 거래비용 모델 | 텔레그램 메시지 포맷 → OPS |
| 가상 ledger 누적 | cron 스케줄 → OPS |
| 성과 메트릭 (PnL, Sharpe) | drift 감지 → OPS |
| kill switch (가상 한정) | 모델 재학습 → SIGNAL |
| 사용자 매매와의 비교 (NOTES 파싱) | 실거래 자동 주문 (영원히 X) |

---

## 10. 핵심 파일 인덱스

| 파일 | 역할 |
|---|---|
| `ledger/config.py` | 가상 자본, K, max exposure 등 상수 |
| `ledger/sizing.py` | equal / sigma_tier / kelly 사이징 |
| `ledger/tracker.py` | 진입 / 청산 자동 |
| `ledger/risk.py` | kill switch / MDD 한도 |
| `ledger/metrics.py` | Sharpe, MDD, hit rate 계산 |
| `output/ledger.csv` | 누적 가상 ledger |
| `scripts/ledger_summary.py` | 주간 / 월간 리포트 |
| `scripts/ledger_vs_user.py` | 시스템 vs 사용자 비교 |
| `scripts/verify_telegram.py` | ledger ↔ 텔레그램 일관성 검증 |
