# PHASES.md — 단계별 액션 + 체크리스트

> Phase 별 구체 작업. **체크박스로 진행 추적**. 매 세션 시작 시 이 문서 head 60 줄 보고 어디까지 왔는지 파악 (CLAUDE.md §0).

---

## 현재 상태 (요약)

- **현재 Phase**: Phase 0 — 설계 문서 작성
- **마지막 갱신**: 2026-05-03
- **다음 액션**: 8 개 MD 작성 마무리 → Phase 1 데이터 수집 시작

진행률 한눈에:
- [x] Phase 0 — 설계 문서 (8 개 MD)
- [ ] Phase 1 — XGBoost baseline + KST 08:30 알림 + paper trading
- [ ] Phase 2 — hybrid 모델 / σ-tier 사이징 / window signature
- [ ] Phase 3 (옵션) — APF motif / prototype bank / 학술 트랙

---

## Phase 0 — 설계 문서 (현재)

**목적**: 코드 짜기 전 8 개 MD 로 모든 결정 명문화. 합의된 설계 위에서 구현.

### 액션
- [x] 폴더 위치 확정 (`/home/soccz/22tb/prelude`)
- [x] 폴더 스켈레톤 (data/signals/ledger/ops/notifier/scripts/notebooks/output/tests/deploy)
- [x] `.gitignore` + `requirements.txt`
- [x] `.claude/settings.local.json` (권한 / 환경)
- [x] CLAUDE.md (작업 규칙)
- [x] README.md (안내판)
- [x] SIGNAL.md (시그널 생성)
- [x] LEDGER.md (가상 ledger)
- [x] OPS.md (운영 인프라)
- [x] ASSETS.md (외부 자산 매핑)
- [x] PHASES.md (이 문서)
- [ ] NOTES.md placeholder
- [ ] git init + 첫 커밋
- [ ] Phase 1 시작 컨펌

### Exit criteria → Phase 1
- 8 개 MD 사용자 검토 완료
- 사용자가 Phase 1 시작 명시 OK

---

## Phase 1 — XGBoost baseline + KST 08:30 알림 (1-2 주)

**목적**: 가장 단순한 작동 시스템 완성. 매일 KST 08:30 텔레그램 알림 받기. 가상 ledger 자동 누적. 실거래는 사용자 직접.

### 1.1 데이터 (data/)
- [ ] `data/collector_d1.py` — 업비트 KRW 일봉 수집 (pyupbit)
  - 출처: `gan_t/data/collector.py`
  - 변경: 일봉 단일, KRW only, 3 년 백필
- [ ] `data/collector_4h.py` — 4h 보조 (장중 max drawdown 용)
- [ ] `data/collector_binance.py` — 바이낸스 1h (김프 / lead-lag 용)
- [ ] `data/upbit_d1.db` 백필 (3 년치, top 200 코인)
- [ ] `data/database.py` — sqlite 헬퍼 (load / save / latest_timestamp)
- [ ] 첫 EDA 노트북: `notebooks/01_data_eda.ipynb`
  - 라벨 분포 (X=8% / Y=3% 시작값 비율)
  - 코인별 데이터 길이 / 결측 분포
  - BTC regime 분포 (4-state)

### 1.2 라벨 + EDA (signals/)
- [ ] `signals/labels.py::today_pump_label`
- [ ] `notebooks/02_label_sweep.ipynb`
  - X ∈ {5, 8, 10, 15} × Y ∈ {2, 3, 5} 격자
  - 라벨 비율 분포 + sweep 결과 보고
  - **사용자 컨펌**: X / Y 최종 결정

### 1.3 피처 (signals/features.py)
- [ ] alt multi-lookback {3, 5, 7, 14, 21} 일 (return / vol / range_contraction)
- [ ] BTC regime (return_Nd, ma_distance, intensity, 4-state)
- [ ] 기술지표 (RSI, MACD, ADX, BB, squeeze, ROC)
- [ ] 크로스섹션 (rank_return_5d, breadth_ratio, top_n_return)
- [ ] cross-market (kimchi, binance_lead) — 보조
- [ ] cross-sectional rank norm (코인별 피처) + rolling z (BTC 피처)
- [ ] **`fillna(0)` 절대 X** (gan_t known gap)

### 1.4 모델 (signals/models/)
- [ ] `signals/models/xgb_phase1.py` — XGBoost binary
  - 출처: `gan_t/training/pump_trainer.py` (4-class → binary 단순화)
  - Optuna 50 trial (objective: binary cross-entropy + sample_weight balanced)
- [ ] `notebooks/03_xgb_baseline.ipynb`
  - 첫 학습 + SHAP 피처 중요도
  - lookback 별 기여 분석

### 1.5 검증 (signals/validate.py + scripts/backtest_wf.py)
- [ ] `signals/validate.py::PurgedWalkForward` (5-fold + 10d embargo)
- [ ] `scripts/backtest_wf.py` — 전체 WF 실행 + 리포트
- [ ] 트레이딩 메트릭: net Sharpe (왕복 0.15% 차감), Max DD, hit rate, 누적 PnL
- [ ] 진단 메트릭 (사후): IC, ICIR, CRPS — 옆에 표기만

### 1.6 Calibration (signals/calibration.py)
- [ ] `signals/calibration.py::SigmaBucketCalibration` — fit / predict / save
- [ ] `output/calibration_sigma.json` 첫 생성

### 1.7 가상 ledger (ledger/)
- [ ] `ledger/config.py` — 가상 자본 1,000 만, K=3, max position 5%, max exposure 60%
- [ ] `ledger/sizing.py::equal_weight` — 1/K 균등 (Phase 1 단순)
- [ ] `ledger/tracker.py` — 시가 진입 / 종가 청산, 거래비용 0.15% 차감
- [ ] `ledger/risk.py` — 일일 -3% / MDD -15% kill switch
- [ ] `ledger/metrics.py` — Sharpe / MDD / hit rate
- [ ] `output/ledger.csv` 자동 누적

### 1.8 운영 (ops/ + notifier/ + scripts/)
- [ ] `ops/preflight.py` — freshness / NaN / churn 체크
- [ ] `ops/run_lock.py` — cron 중복 방지
- [ ] `ops/drift_detector.py` — sign flip / 50% drop 감지
- [ ] `notifier/telegram.py` — 텔레그램 봇 클래스
- [ ] `notifier/format.py` — 메시지 포맷 (OPS §3.1)
- [ ] **별도 텔레그램 봇 발급 + 채팅 ID** (gan_t 와 분리)
  - **사용자 컨펌**: 봇 토큰 / 채팅 ID 환경변수 셋업 (`.env`, .gitignore)
- [ ] `scripts/predict_today.py` — 수동 dry-run
- [ ] `scripts/daily_run.sh` — KST 08:30 cron entry
- [ ] `scripts/post_open_run.sh` — KST 09:30 청산 + verify
- [ ] `deploy/crontab.txt` — cron 등록 명령 + README
- [ ] `scripts/health_check.py` — 일일 헬스
- [ ] `scripts/verify_telegram.py` — ledger ↔ telegram 일관성

### 1.9 첫 알림 발사
- [ ] dry-run 1 일 (수동 실행, 텔레그램 발송 X)
- [ ] dry-run 결과 + 알림 포맷 사용자 검토
- [ ] **사용자 컨펌**: 라이브 cron 등록
- [ ] **D-Day**: 첫 KST 08:30 라이브 알림
- [ ] 매일 KST 09:30 ledger 자동 갱신 확인

### Exit criteria → Phase 2
- [ ] **라이브 기간 충분** (초기 2 주, 데이터 안정성 보고 조정 — CLAUDE.md §2.5)
- [ ] 가상 net Sharpe **양수 + 의미 있는 수준** (cutoff 도 데이터 기반: 음수면 모델 재설계, 0~0.5 면 Phase 2 갈지 모델 강화 갈지 고민, ≥ 0.5 면 자연스럽게 Phase 2)
- [ ] 일관성 검증 통과 (verify_telegram 모든 날 OK)
- [ ] 사용자가 Phase 2 명시 OK

### Phase 1 lessons (작성: Phase 1 끝나는 시점)
*(여기에 Phase 1 진행하면서 배운 점, 실패, 의외의 발견 기록)*

---

## Phase 2 — Hybrid 모델 + 사이징 강화 (2-4 주)

**목적**: Phase 1 baseline 대비 의미 있는 net 개선 (DM test). σ-tier 기반 사이징 비교.

**Phase 1 결과가 충분히 좋으면 Phase 2 유보** (단순 유지). CLAUDE.md §2.3 — 학술적 정교화로 결과 더 나빠지는 경우 많음.

### 2.1 Hybrid 모델 (Phase 1 의미 있을 때만)
- [ ] `signals/models/hybrid_phase2.py` — Transformer + TCN + Gate + FiLM + CVAE
  - 출처: `gan_t/models/hybrid_model.py` + `AETHER_IDEA.md`
  - 변경: 일봉 호라이즌, residual return 입력 옵션
- [ ] `notebooks/04_hybrid_train.ipynb`
- [ ] DM test: Hybrid vs XGBoost baseline (net Sharpe 차이 유의?)
- [ ] **사용자 컨펌**: Hybrid 채택 vs Phase 1 유지

### 2.2 σ-tier 사이징 비교
- [ ] `ledger/sizing.py::sigma_tier_weight` — 🔥 3% / ✅ 2% / ▫ 1%
- [ ] backtest 비교: equal vs sigma_tier (net Sharpe / MDD)
- [ ] **사용자 컨펌**: 사이징 룰 변경 또는 유지

### 2.3 Window signature (선택)
- [ ] `signals/models/window_signature.py` — 윈도우 → GRU → 16d signature
  - 출처: `fin/paper/economic_time/window_signature_model.py`
- [ ] hybrid 모델에 conditioning token 추가
- [ ] DM test: with vs without signature

### 2.4 Multi-day continuation 점수 (사용자 1 차 메시지의 보조 아이디어)
- [ ] 메인 라벨은 그대로 (오늘 1 일), but 별도 회귀로 N=3 일 지속 점수 출력
- [ ] 알림에 보조 점수로 표시: "오늘오를 78% / 3 일지속 65%"
- [ ] **사용자 컨펌**: 알림 포맷 변경

### 2.5 손절 / 익절 룰 비교 (가상 ledger)
- [ ] `ledger/tracker.py` 에 옵션 추가
- [ ] 비교: hold 1d (현재) vs +15% 익절 / -5% 손절
- [ ] backtest net Sharpe / MDD / 평균 hold 시간

### Exit criteria → Phase 3
- [ ] Phase 2 라이브 4 주 완료
- [ ] Hybrid 채택 시: Phase 1 대비 DM test 유의 (p < 0.05)
- [ ] σ-tier 채택 시: net Sharpe 개선
- [ ] 사용자 명시적 Phase 3 OK

### Phase 2 lessons
*(작성: Phase 2 끝나는 시점)*

---

## Phase 3 (옵션) — APF motif + prototype bank + 학술 (선택)

**옵션**. 트레이딩 결과 Phase 1/2 만으로 충분하면 Phase 3 안 해도 됨. 학술 트랙 관심 있으면 진행.

### 3.1 APF motif 진단 (Phase 2 hybrid 위)
- [ ] `signals/models/apf_diagnostic.py` — attention map → motif 분류 (stripe/block/spike/diagonal)
  - 출처: `fin/Attention Pattern Fields/src/apf/`
- [ ] 알림에 motif 표시: "추천 근거: spike motif 89% (이벤트 탐지)"

### 3.2 Prototype bank + DTW
- [ ] `signals/prototype_bank.py` — 과거 stable_pump 성공 윈도우 DB
  - 출처: `gan_t/data/success_patterns.npy` + AETHER prototype bank 컨셉
- [ ] 일일 추론 시 DTW 매칭: "과거 패턴 #7 유사도 0.81 (당시 +12.3%)"
- [ ] 알림에 추가

### 3.3 학술 논문화 (선택)
- [ ] APF 페이퍼 의 금융 도메인 첫 적용 — TMLR 보강 또는 별도 짧은 논문
- [ ] BTC × Upbit version of F=14.335 finding
- [ ] **사용자 명시 시에만 진행** — 트레이딩 결과 우선 (CLAUDE.md §2.3)

### 3.4 Cycle-PE / linear attention 실험 (장기)
- [ ] hybrid 모델 인코더에 Cycle-aware PE 추가 (AETHER §5)
- [ ] linear attention 변형 (fin Ch.08 +49% finding)
- [ ] DM test 비교

### Exit criteria → 안정 운영
- [ ] Phase 3 추가 모듈 중 net 결과 개선되는 것만 채택
- [ ] 결과 안 좋으면 Phase 2 로 롤백
- [ ] 안정 운영 모드 진입

### Phase 3 lessons
*(작성)*

---

## 비상 / 롤백 절차

각 Phase 진행 중 다음 발생 시:

| 상황 | 액션 |
|---|---|
| 가상 net PnL 4 주 누적 음수 | 텔레그램 watch-only 전환 + 사용자 컨펌 후 모델 재설계 |
| drift detector FREEZE 발동 | 즉시 가상 진입 X + retrain 강제 트리거 |
| verify_telegram 부호 / 크기 불일치 | 즉시 알림 발사 정지 + 디버깅 + 재개 컨펌 |
| 데이터 24h 이상 stale | preflight 자동 watch-only |
| 사용자 NOTES 에 "시스템 신뢰 X" 기록 | 다음 세션 시작 시 Claude 가 발견 → 사용자 컨펌 |

---

## 작업 흐름 (매 Phase 공통)

1. 이 문서 head 60 줄로 어디까지 왔는지 확인
2. 다음 미체크 액션 1 개 in_progress
3. TodoWrite 로 세션 내 작업 추적
4. 액션 완료 시 즉시 [x] 체크 + 한 줄 lessons 기록 (선택)
5. 큰 결정 (모델 변경, 라벨 X/Y, 알림 포맷) 은 사용자 컨펌
6. Phase exit criteria 충족 시 사용자 컨펌 후 다음 Phase 진입

---

## 변경 이력 (Phase 단위)

| 날짜 | Phase | 변경 / 결정 |
|---|---|---|
| 2026-05-03 | Phase 0 | 폴더 + 8 개 MD 설계 시작 |
| 2026-05-03 | Phase 0 | 모든 숫자 placeholder 명시 + CLAUDE.md §2.5 신설 (데이터가 결정) |
| | | |
