# PHASES.md — 단계별 액션 + 체크리스트

> Phase 별 구체 작업. **체크박스로 진행 추적**. 매 세션 시작 시 이 문서 head 60 줄 보고 어디까지 왔는지 파악 (CLAUDE.md §0).

---

## 현재 상태 (요약, 2026-05-03)

- **현재 운영**: detector_v1 Stage 1 dry-run (cron 등록 직전)
- **백테스트 채택**: C3 (bull_all p99.95 cap2), EV +7.40% / 2024 -0.89% / 3 active fold 양수
- **아카이브**: Phase 0/1 (6-class 분포 + 일반 펌프 detector) — legacy 보존, 미운영

진행 트랙:
- [x] Phase 0 — 8 MD 설계 완료
- [x] Phase 1 — 데이터 수집 / 6-class 모델 / 인프라 (legacy)
- [x] Phase X — leak 발견 → detector 재정의 → C3 채택 → detector_v1 artifact (이번 세션)
- [x] **Phase X+2** — dashboard publish 파이프 (paper_ledger → soccz.github.io 정적 회고) 2026-05-07
- [ ] **Stage 1** (cron dry-run, telegram off) — 사용자 cron 등록 후 1~2주
- [ ] **Stage 2** (telegram beta, 자동매매 X) — Stage 1 결과 보고 결정
- [ ] **Stage 3** (NOTES 기반 threshold/tier 조정) — Stage 2 후
- [ ] **Phase X+1** — Distribution head (multi-target) + label space discovery (사용자 신규 방향, §"향후 방향" 참조)
- [ ] [Research] Downside guard / 4h confirmation tier (병렬)
- [ ] [Later] MTF features / regime split / Optuna

### Phase X+2 — Dashboard publish 파이프 (2026-05-07)

**의도**: 텔레그램은 오늘 판단용. github.io 의 dashboard 는 회고용 — "어제 샀으면 어땠나, 시스템이 잘 맞추고 있나, 누적이 어떤 흐름인가" 본인 모니터링.

**구조**:
- 데이터 보강: `paper_ledger.csv` 두 개에 OHLC + min_return_pct 컬럼 추가. close 스크립트가 누락 컬럼 자동 보강 + canonical 순서로 reorder. historical 28+24 row backfill 완료.
- 빌더: `scripts/build_dashboard.py` → JSON 3종 (summary/history/accuracy) 산출. 가상 PnL 룰은 텔레그램 가이드와 동일 (5% TP / EOD close, 비용 0.15% 차감, equal weight).
- 정적 페이지: `soccz.github.io/projects/prelude/dashboard/index.html` (chart.js CDN, vanilla JS) — KPI 카드 + 누적 PnL 곡선 + rolling hit rate + 정렬/필터 가능 알림 표.
- 자동 publish: `scripts/publish_dashboard.sh` 가 build → site repo add/commit/push. 실패 시 텔레그램 alert.
- systemd: `prelude-publish-dashboard.{service,timer}` (KST 10:10, close cron 두 개 끝나고 5분 여유).

**라이브 첫 결과 (28 closed dist + 24 closed preopen)**: 누적 가상 PnL 둘 다 음수 (dist -12.97%, preopen -13.47%). avg_max +6.82% / avg_min -5.94% (dist) — 변동성은 크지만 5% TP 룰 + 비용으로 누적은 깎임. 라이브 paper 데이터 더 쌓이면서 calibration 트랙 (사용자 NOTES + dashboard) 으로 룰 조정.

**주의**: 첫 site repo commit + push 는 사용자 수동 (라이브 반영 confirmation). 그 후부터 자동.

---

### Algorithm audit update (2026-05-05)

- v1 dry-run 유지. **v2 multi-scale swap 보류** — 1주 live paper 결과 확인 후 결정.
- Common-period ablation: head 별 best scale 이 다름.
  - h2 즉발 +3%: daily+1h / daily+15m 둘 다 강함
  - h5 +20% tail: daily+1h+15m 가 우위
  - h6 +5% 24h: daily+1h 정도면 충분
- Baseline showdown: distribution_beta 가 TP3/TP5 에서 setup/momentum baseline 을 이김.
- 다만 edge modest: TP5 path-aware Sharpe diff vs setup_momentum +0.18, bootstrap CI 가 0 근처를 걸침.
- MDD 큼: full-size TP5 MDD 약 -55%; 운영 해석은 1/4~1/8 fractional sizing 기준.
- SL 룰: 4h SL-first 와 15m path 양쪽 모두 음수. 자동 SL 룰은 운영 채택 X, 사용자 수동 판단.
- 09:05 timer audit: 전체 시장 첫 15m hit 비중은 9~12% 수준이지만, distribution alerts 는 +3% hit 의 33%, +5% hit 의 25% 가 첫 15m candle 에 발생. 09:05 는 데이터 위생상 유지하되, 실제 즉발 진입용 08:55 pre-open trigger 는 별도 모델/검증 트랙으로 분리.
- Pre-open first15 model audit: 08:55 as-of 를 엄격히 맞춰 `D-2 closed daily + D-1 08:30 precursor` 로 검증. `preopen_15m` 단독이 first15_t3 top1% precision 38.2% (base 5.1%, lift 7.6), first15_t5 top1% precision 22.4% (base 2.8%, lift 8.1). 08:55 전용 모델은 연구/운영 후보로 충분히 정당화됨. 단 v1 09:05 distribution timer 는 유지.
- Pre-open code audit: live 모델을 15m precursor-only 19 features 로 재빌드해 daily partial mismatch 제거. late manual run guard 추가(08:45~08:59 밖에서는 telegram/ledger skip), raw score 문구로 표시, 15m recent-window 로 predict runtime 4m48s → 21s. close-out 은 15m DB update wrapper 필요(`scripts/daily_close_preopen.sh`).
- Survivorship bias 는 여전히 미처리 caveat.

---

## Phase X — Detector 재정의 (2026-05-03 완료)

**lessons (이번 세션 핵심)**:
- 일반 6-class softprob 모델 → ledger 음수 → "task 정의 자체가 잘못" 진단
- 재정의: ≥20% tail pump detector (rare event, silence-heavy)
- BTC bull regime conditional + TP20-only execution
- Sweep 90 조합 (regime × threshold × cap) → bull_quiet × p99.95 sweet spot
- Fold stability 검증 v1 (regime-internal threshold leak) → v2 (train direct, overfit zero-trade) → v3 (train-OOF, **C3 통과**)
- v3 발견:
  - C1/C2 (bull_quiet 단독) sparse → 3/5 fold 침묵 → fail
  - **C3 (bull_all p99.95 cap2)** 3/4 active 양수, 2024 -0.89% — 채택
  - C8-C10 rank fallback EV 음수 → 폐기 (silence-heavy 가 옳다는 증거)
- artifact: threshold 0.8815 (full panel OOF p99.95, KRW 136,924 samples) 고정
- 운영 원칙: threshold 라이브 quantile 재계산 금지, bear regime silence, cap 2

---

## 향후 방향 — Distribution head + label space discovery (사용자 2026-05-03 제안)

**문제 의식**: detector_v1 은 "≥20% tail" 한 점만 본 것. 사용자가 원하는 건 매매 판단에 필요한 **조건부 확률 분포 + 다중 head**.

**제안 구조**:
```
Distribution head (multiple binary XGBoost):
- upside heads:    P(high ≥ +3/+5/+7/+10/+15/+20%)
- close heads:     P(close ≥ +0/+3/+5%)
- downside heads:  P(low ≤ -2/-3/-5%)
- path heads:      P(hit +X before drawdown -Y)
- expected:        E[max_return], E[close_return], E[max_drawdown]

알림 출력 = 분포 테이블 (코인당), 단일 score X
```

**Label space discovery (선결)**:
- profit_target × min_close_hold × max_pre_hit_dd × max_post_hit_giveback × time_to_hit × btc_regime sweep
- 평가: base_rate, lift@top, avg fail EOD, worst5%, active_days, hit time
- 4 조건 동시 만족: 너무 sparse X, model lift > random, fail 손실 작음, 사용자 대응 가능
- detector_v1 = 분포의 오른쪽 꼬리 head 1개로 자연스럽게 흡수됨

**우선순위**: Stage 1 dry-run + Downside guard / 4h confirmation 보다 **뒤** (detector_v1 안정 후 트랙 분리). 단 stable_v1 prototype 은 detector_v1 운영과 병렬 research 가능.

---

## Phase 0 — 설계 문서 (legacy)

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

## Phase 1 — XGBoost baseline + KST 09:05 알림 (1-2 주)

**목적**: 가장 단순한 작동 시스템 완성. 매일 KST 09:05 텔레그램 알림 받기. 가상 ledger 자동 누적. 실거래는 사용자 직접.

### 1.1 데이터 (data/)
- [ ] `data/collector_d1.py` — 업비트 KRW 일봉 수집 (pyupbit)
  - 출처: `gan_t/data/collector.py`
  - 변경: 일봉 단일, KRW only, 3 년 백필
- [ ] `data/collector_4h.py` — 4h 보조 (장중 max drawdown 용)
- [ ] `data/collector_binance.py` — 바이낸스 1h (김프 / lead-lag 용)
- [ ] `data/upbit_d1.db` 백필 (3 년치, top 200 코인)
- [ ] `data/database.py` — sqlite 헬퍼 (load / save / latest_timestamp)
- [ ] 첫 EDA 노트북: `notebooks/01_data_eda.ipynb`
  - 코인별 데이터 길이 / 결측 분포
  - BTC regime 분포 (4-state)
  - 일봉 max(high)/open 분포 (multi-class 라벨 후보 bin 검증용)

### 1.2 라벨 + EDA (signals/) — Multi-class 분포
- [ ] `signals/labels.py::today_pump_label` (multi-class, max(high)/open 기반, SIGNAL §2.1)
- [ ] `notebooks/02_label_distribution.ipynb`
  - bin 경계 후보 (0/5/10/15/20%) 별 라벨 분포
  - 각 bin 비율 5~30% 가 학습에 좋음 — sparse / dense 면 cutoff 조정
  - 4h 봉 데이터로 max(high) 정확 측정
  - **사용자 컨펌**: bin 경계 최종 결정

### 1.3 피처 (signals/features.py)
- [ ] alt multi-lookback {3, 5, 7, 14, 21} 일 (return / vol / range_contraction)
- [ ] BTC regime (return_Nd, ma_distance, intensity, 4-state)
- [ ] 기술지표 (RSI, MACD, ADX, BB, squeeze, ROC)
- [ ] 크로스섹션 (rank_return_5d, breadth_ratio, top_n_return)
- [ ] cross-market (kimchi, binance_lead) — 보조
- [ ] cross-sectional rank norm (코인별 피처) + rolling z (BTC 피처)
- [ ] **`fillna(0)` 절대 X** (gan_t known gap)

### 1.4 모델 (signals/models/) — Multi-class softprob
- [ ] `signals/models/xgb_phase1.py` — XGBoost multi-class (objective='multi:softprob', num_class=6)
  - 출처: `gan_t/training/pump_trainer.py` (4-class 패턴 차용, 6-class 로 확장)
  - Optuna 50 trial (objective: mlogloss + per-bin macro F1)
- [ ] `notebooks/03_xgb_baseline.ipynb`
  - 첫 학습 + SHAP 피처 중요도
  - lookback 별 기여 분석
  - bin 별 적중률 (per-class accuracy)

### 1.5 검증 (signals/validate.py + scripts/backtest_wf.py)
- [ ] `signals/validate.py::PurgedWalkForward` (5-fold + 10d embargo)
- [ ] `scripts/backtest_wf.py` — 전체 WF 실행
- [ ] **트레이딩** 메트릭: net Sharpe (옵션 3 익절/손절 시뮬, 왕복 0.15%), Max DD, 누적 PnL
- [ ] **정확도** 메트릭 (사용자 핵심 요구): Brier score, Reliability diagram, Quantile coverage, per-bin accuracy
- [ ] 진단 (학술 사후): IC, ICIR — 옆 표기만

### 1.6 Calibration (signals/calibration.py)
- [ ] `signals/calibration.py::ReliabilityCalibration` — multi-class 보정
- [ ] `output/reliability_curves.json` (각 cutoff: P(≥5%), P(≥10%), ...) 첫 생성
- [ ] `output/brier_history.json` (Brier score 누적)

### 1.7 가상 ledger (ledger/) — TP/SL 시뮬
- [ ] `ledger/config.py` — 가상 자본 1,000 만, K=3, max position 5%, TP=0.10, SL=0.05 (placeholder)
- [ ] `ledger/sizing.py::equal_weight` — 1/K 균등 (Phase 1 단순)
- [ ] `ledger/tracker.py` — 옵션 3: 시가 진입 → TP/SL 또는 24h 종가 청산 (4h 봉 시뮬)
- [ ] `ledger/risk.py` — 일일 -3% / MDD -15% kill switch
- [ ] `ledger/metrics.py` — Sharpe / MDD / TP-SL hit rate / 평균 hold
- [ ] `output/ledger.csv` 자동 누적
- [ ] `scripts/tp_sl_sweep.py` — TP × SL 격자 백테스트 → 최적 조합 추천

### 1.8 운영 (ops/ + notifier/ + scripts/)
- [ ] `ops/preflight.py` — freshness / NaN / churn 체크
- [ ] `ops/run_lock.py` — cron 중복 방지
- [ ] `ops/drift_detector.py` — sign flip / 50% drop 감지
- [ ] `notifier/telegram.py` — 텔레그램 봇 클래스
- [ ] `notifier/format.py` — 메시지 포맷 (OPS §3.1)
- [ ] **별도 텔레그램 봇 발급 + 채팅 ID** (gan_t 와 분리)
  - **사용자 컨펌**: 봇 토큰 / 채팅 ID 환경변수 셋업 (`.env`, .gitignore)
- [ ] `scripts/predict_today.py` — 수동 dry-run
- [ ] `scripts/daily_run.sh` — KST 09:05 cron entry
- [ ] `scripts/post_open_run.sh` — KST 09:30 청산 + verify
- [ ] `deploy/crontab.txt` — cron 등록 명령 + README
- [ ] `scripts/health_check.py` — 일일 헬스
- [ ] `scripts/verify_telegram.py` — ledger ↔ telegram 일관성

### 1.9 첫 알림 발사
- [ ] dry-run 1 일 (수동 실행, 텔레그램 발송 X)
- [ ] dry-run 결과 + 알림 포맷 사용자 검토
- [ ] **사용자 컨펌**: 라이브 cron 등록
- [ ] **D-Day**: 첫 KST 09:05 라이브 알림
- [ ] 매일 KST 09:30 ledger 자동 갱신 확인

### Exit criteria → Phase 2
- [ ] **라이브 기간 충분** (초기 2 주, 데이터 안정성 보고 조정 — CLAUDE.md §2.5)
- [ ] 가상 net Sharpe **양수 + 의미 있는 수준** (음수면 모델 재설계, 0~0.5 면 Phase 2 갈지 고민, ≥ 0.5 면 자연스럽게 Phase 2)
- [ ] **시스템 정확도 검증**: Reliability 가 대각선 ± 10pp 이내 (예: 예측 50% → 실제 40~60%)
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
| 2026-05-03 | Phase 0 | 라벨 binary → multi-class (max(high)/open 분포), ledger 단순 hold → TP/SL 옵션 3 |
| 2026-05-03 | Phase 0 | `today_pump` 폴더명 → `prelude` 통일 + GitHub `soccz/prelude` push |
| 2026-05-03 | Phase 0 | `data/database.py` + `data/collector_d1.py` 작성, KRW-BTC smoke test PASS |
| 2026-05-03 | Phase 1.0 | 4 collectors 백필 완료 (KRW d1 252, KRW 4h 252, BINANCE 1h 185, BINANCE d1 427) |
| 2026-05-03 | Phase 1.1 | **leak 발견** — features[t] (close[t]) → label[t] (high[t]/open[t]) 동시점 사용 |
| 2026-05-03 | Phase 1.1 | label_panel **market별 shift(-1)** 수정 — leak 제거 |
| 2026-05-03 | Phase 1.1 | leak 후 accuracy 67%→30% (random 17% 대비 1.83x — 약한 신호만) |
| 2026-05-03 | Phase 1.2 | 알림 시간 **08:30 → 09:05** (어제 일봉 100% 마감 후 leak-free) |
| 2026-05-03 | Phase 1.2 | Pattern sweep — 7 family WF ledger backtest 모두 음수 Sharpe |
| 2026-05-03 | Phase 1.2 | EDA hit rate ≠ Sharpe — momentum hit 21% but Sharpe -5.2 (SL 46% 함정) |
| 2026-05-03 | Phase 1.3 | Execution sweep — 15 룰 × 3 family → **TP15_only 만 Sharpe +0.13** |
| 2026-05-03 | Phase 1.3 | Filter alpha X — baseline_full random 이 모든 family 이김 |
| 2026-05-03 | Phase 1.3 | 진단: 일봉 long 대부분 손해, 드문 +15% 꼬리 펌프만 알파 가능성 |
| 2026-05-03 | Phase 1.4 | 핵심 검증: TP15_only execution + binary 모델 vs random (진행 중) |
| | | |
