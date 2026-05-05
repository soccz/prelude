# ASSETS.md — 외부 자산 참조 매핑

> 다른 폴더에서 **import 안 함**. 코드는 prelude 안에 self-contained 새로 짠다 (CLAUDE.md §2.1). 이 문서는 어디서 어떤 아이디어 / 패턴을 참조했는지 출처만 기록.

> **운영 컨텍스트 (2026-05-03)**:
> 현재 메인 = `detector_v1` (binary tail ≥20%). 6-class softprob (gan_t pump_classifier 차용 패턴) 은 **legacy** — `signals/predict.py` + `scripts/predict_today_legacy.py` 에 보존.
> 사용자 제안 신규 방향 (distribution head + label discovery, PHASES "향후 방향" §) 은 6-class 분포 모델 패턴을 **다중 binary head 형태로 재사용** 가능.

---

## 0. 원칙

- **코드 import X**: `from gan_t.*` `from xsec_alpha.*` `from fin.*` 절대 X
- **참조 OK**: 코드 읽기 / 패턴 차용 / 알고리즘 학습
- **새로 짜기**: prelude 안에 새 파일로, 이 프로젝트에 맞게 단순화
- **출처 기록 필수**: 모든 차용은 이 문서에 + 차용한 파일 docstring 첫 줄에 출처 코멘트

---

## 1. 자산 출처별 정리

### 1.1 gan_t (`/home/soccz/22tb/main/gan_t/`)

크립토 펌프 예측 + KST 08:00 morning cron 시스템. 시간봉 기준이고 작동 안 하는 부분 많지만 인프라 패턴은 좋음.

| gan_t 위치 | 핵심 아이디어 | prelude 구현 |
|---|---|---|
| `data/preprocessor.py::create_pump_features` | volume_spike_score, squeeze_on, roc — 펌프 직전 피처 | `signals/features.py::pump_microstructure_features` (일봉 스케일로) |
| `data/preprocessor.py::create_pump_labels` | 미래 N 시간 max high 기반 펌프 라벨 (10/15/20% multi-class) | `signals/labels.py::today_pump_label` (일봉 + drawdown 추가, binary) |
| `training/pump_trainer.py` | XGBoost 4-class + Optuna 튜닝 (n_estimators=782, max_depth=10) | `signals/models/xgb_phase1.py` (binary 로 단순화) |
| `models/hybrid_model.py` | Transformer + TCN + Gate + FiLM + GAN/CVAE 하이브리드 | `signals/models/hybrid_phase2.py` (Phase 2 — 일봉, residual return) |
| `utils/scheduled_modes.py::run_morning_report_mode` | KST 08:00 morning cron 흐름 (preflight → predict → telegram) | `ops/preflight.py` + `scripts/daily_run.sh` (KST 09:05, 일봉) |
| `utils/telegram_bot.py::send_morning_report` | 텔레그램 메시지 포맷 + signal alert dedup | `notifier/telegram.py` + `notifier/format.py` (prelude 톤) |
| `utils/freshness.py` | 데이터 신선도 게이트 + exit code 2 fallback | `ops/preflight.py::check_freshness` |
| `utils/run_lock.py` | cron 중복 실행 방지 lock | `ops/run_lock.py` (그대로 차용 가능, single-file) |
| `utils/watchdog.py` | timeout watchdog | `ops/watchdog.py` |
| `data/success_patterns.npy` | 과거 성공 패턴 DB (DTW 매칭용) | `signals/prototype_bank.py` (Phase 3, 일봉 stable_pump 성공 사례 DB) |

**경고 (gan_t 의 known gap)**: prelude 에서 반복하지 말 것:
- `fillna(0)` → RSI=0, vol=0 같은 불가능 값 만듦 (gan_t known gap #2)
- `pct_change` 단순 사용 → 캔들 갭 미감지 (gan_t known gap #4)
- 1h horizon AC=0.025 (노이즈) 에서 학습 → prelude 는 일봉이라 다르지만 horizon 짧으면 동일 함정
- `train ⊃ test` (xsec_alpha 1단계 교훈) → Purged WF 필수

---

### 1.2 xsec_alpha (`/home/soccz/22tb/main/gan_t/xsec_alpha/`)

크립토 크로스섹션 랭킹 + 운영 게이트 풀세트. **시스템적으로 가장 발전한 단일 자산**.

| xsec_alpha 위치 | 핵심 아이디어 | prelude 구현 |
|---|---|---|
| `data/features.py::compute_unified_factors` | 10 개 검증 팩터 (range_contraction_12h, reversal_1h/4h, volatility_inv_24h, kimchi_inv, binance_lead_1h, order_flow_bear, dow_bull, hour_vol) | `signals/features.py` (일봉 스케일 변환 — _12h → _Nd) |
| `utils/magnitude.py::predict_one` + σ-bucket | sigma = score/batch_std → bucket 별 hit_rate / mean_signed_return / CI 출력 | `signals/calibration.py::SigmaBucketCalibration` |
| `output/calibration_sigma.json` | calibration 영구 저장 포맷 | `output/calibration_sigma.json` (prelude 안에 새로) |
| `utils/preflight.py` | freshness ≤ 90min, NaN ≤ 20%, universe churn ≤ 30% | `ops/preflight.py` (일봉 → 임계 조정) |
| `utils/ic_gate.py` | OK/WARN/FREEZE/LIQUIDATE 상태머신 | `ops/ic_gate.py` (사후 진단으로만, 게이트 X) |
| `utils/drift_detector.py` | per-factor IC 24h vs 7d MA, SIGN_FLIP / 50% drop 감지 | `ops/drift_detector.py` |
| `utils/enrich.py` | 두 모델 consensus ⚡ + 원장 기반 per-coin 신뢰도 ⭐⚠ | `notifier/format.py::add_confidence_marks` |
| `scripts/retrain_pipeline.py` | 주 1 회 학습 + promotion gate (new_IC ≥ old_IC - 0.015) | `ops/retrain_pipeline.py` (net Sharpe 기준으로 변경) |
| `scripts/health_snapshot.py` | 세션 킥오프 헬스 스냅샷 | `scripts/health_check.py` |
| `scripts/verify_telegram.py` | ledger ↔ telegram 부호 일관성 regression test | `scripts/verify_telegram.py` (prelude 안에 새로) |
| `output/recommendation_ledger.csv` | 추천-실현 자동 누적 ledger | `output/ledger.csv` (가상 포지션 기반) |
| `deploy/install_f1.sh` + systemd timer | 운영 자동화 deploy 스크립트 | `deploy/install.sh` + systemd timer |

**xsec_alpha 의 결정적 발견 (prelude 에 흡수)**:
- `range_contraction_12h` IC = +0.179 (4 분기 안정) → prelude 의 `range_contraction_Nd` 메인 피처
- 단순 모멘텀 IC 전 호라이즌 음수 → "조용해진 뒤 터질 코인" 컨셉
- σ-bucket calibration 의 효과 (per-coin 확률 + CI)
- F1 통합 아키텍처: 한 피처셋 → 두 모델 (다른 horizon) → 92% 방향 합의
- bull-only 학습 → survivorship bias → 전 regime 학습으로 변경 (prelude 도 동일 적용)

**xsec_alpha 의 실패 (prelude 에서 피할 것)**:
- Long-only WF Sharpe -0.78 → prelude 는 binary 펌프 분류라 롱 진입 시 더 보수적
- 텔레그램 부호 버그 (verify_telegram.py 가 검증) → prelude 도 자동 검증 (OPS §8)

---

### 1.3 fin / Attention Pattern Fields (`/home/soccz/22tb/fin/`)

학술 트랙. **prelude 우선순위 낮음** (논문 형식 적용은 결과 나쁘게 만들 위험 — CLAUDE.md §2.3). 사후 진단 / Phase 3 옵션으로만.

| fin 위치 | 핵심 아이디어 | prelude 구현 (Phase 3 옵션) |
|---|---|---|
| `Attention Pattern Fields/src/apf/` | motif (stripe/block/spike/diagonal) intervention pipeline | `signals/models/apf_diagnostic.py` (Phase 3, attention 진단만) |
| `paper/economic_time/window_signature_model.py` | 윈도우 → GRU → 16d signature → conditioning token | `signals/models/window_signature.py` (Phase 2 옵션) |
| `paper/method_paper_writing/17_*` finding | F=14.335, p=6.09e-07 — market_return × intensity interaction | `signals/features.py::btc_alt_interaction` (이미 multi-lookback 으로 자연스럽게 학습) |
| `AETHER_IDEA.md::Cycle-aware PE` | btc_ma_distance + btc_regime_rv 두 신호로 토큰별 BTC 위치 주입 | `signals/features.py::btc_cycle_features` (피처로 단순화) |
| `AETHER_IDEA.md::prototype bank + DTW` | 과거 성공 윈도우 DB → DTW 거리 매칭 | `signals/prototype_bank.py` (Phase 3, gan_t success_patterns 와 합침) |
| `AETHER_IDEA.md::FiLM regime + gate diagnostic` | 4-state regime affine + Trend/Pattern gate 변수 노출 | `signals/models/hybrid_phase2.py` 안 (Phase 2) |
| `_archive/structural_filtering/` | 3-stage protocol (seed-paired + exact-carry + pooled stack) — 403 hypothesis × 143k experiment 검증 | `ops/structural_filter.py` (사후 검증으로만, 게이트 X) |
| `paper_idea_ko.md::Purged WF + 10d embargo` | look-ahead 방어 표준 | `signals/validate.py::PurgedWalkForward` (이건 양보 X — CLAUDE.md §2.3 위생) |
| `paper_drafts/paper_3_TTPA.md` | test-time positional adaptation — 추론 시 시간 좌표 적응 | (Phase 3 학술 옵션, 우선 X) |
| `economic-time-research-guide/08_linear_attention.md` | softmax → ELU+1 교체 +49% IC | (Phase 3 학술 옵션, hybrid 모델 인코더 변형) |

**경고 (fin 트랙의 함정 — 사용자 직접 경험)**:
- 논문 형식 strict 적용 → 트레이딩 결과 나빠짐 (feedback_academic_vs_trading)
- 이 자산들은 사후 진단 / 옵션 / 학술 트랙용. **사전 lever 로 X**

---

### 1.4 그 외

| 출처 | 용도 |
|---|---|
| `/home/soccz/22tb/main/gan_t/.env` | **MAE 텔레그램 봇** 토큰 / chat_id — prelude/.env 에 동일 값 복사 (gan_t/xsec_alpha 와 봇 공유, 메시지 prefix `🌅 prelude` 로 구분) |
| `/home/soccz/22tb/main/wqbrain/CLAUDE.md` | IQC 알파 실패 4 유형 (LOW_FITNESS, SELF_CORRELATION, LOW_SHARPE, SUB_UNIVERSE) — prelude 도 동일 패턴 디버깅 시 참조 |
| `/home/soccz/22tb/fin/ICQ_legacy/ICQ9/04_crowding_contrarian.md` | "조용해진 뒤 터질 코인" 알파 컨셉 — range_contraction 과 일치 |
| `/home/soccz/22tb/fin/ICQ_legacy/ICQ8/01_microstructure.md` | 일봉 안 마이크로 시그널 — Phase 2 추가 피처 후보 |
| `/home/soccz/22tb/fin/ICQ_legacy/ICQ9/02_multi_timeframe.md` | 일봉 + 주봉 결합 — Phase 2 lookback 확장 |
| pyupbit, ccxt | 외부 lib (실제 import OK — 외부 패키지) |
| python-telegram-bot | 외부 lib |
| xgboost, lightgbm, optuna | 외부 lib |

---

## 2. 차용 시 작업 흐름

### 2.1 새 모듈 추가 시
```
1. ASSETS.md 에서 비슷한 자산 있는지 검색
2. 있으면 그 위치 코드 Read 로 읽고 이해
3. prelude 안에 새 파일 작성:
   - 이 프로젝트 맥락 (일봉, 한국시간, 보수적 펌프) 에 맞게 단순화
   - 첫 줄 docstring 에 출처 명시:
     """ Adapted from: gan_t/data/preprocessor.py::create_pump_features
         Changes: 일봉 스케일, drawdown 조건 추가, fillna(0) 제거 """
4. ASSETS.md §1 의 해당 표 행 갱신 (prelude 구현 컬럼)
```

### 2.2 자산 발견 시 (이 표에 없는 거)
1. 짧게 해당 위치 핵심 읽기
2. 이 표에 새 행 추가 (출처 + 아이디어 + 어떻게 prelude 에서 쓸지)
3. 차용 시 §2.1 흐름

---

## 3. 자산 status 표 (현재)

| 자산 | 상태 | 마지막 확인 |
|---|---|---|
| gan_t | 작동 (시간봉 morning cron 작동, 추천 0건 상태) | 2026-05-03 |
| xsec_alpha | 작동 (live paper trading) | 2026-05-03 |
| fin (multi paper tracks) | 학술 작업 중 (P1 ProTran-TFA / P3 paper_1 / Attention Pattern Fields TMLR) | 2026-05-03 |
| Attention Pattern Fields | TMLR 제출 직전, 7,441 줄 코드 | 2026-04-30 |

자산 위치 / 코드 변경 시 이 표 갱신.

---

## 4. 절대 원칙 재명시

- **prelude 안 코드는 prelude 모듈만 import**
- **외부 폴더 코드 수정 금지** (CLAUDE.md §3.1)
- **참조는 자유, 차용 시 출처 명시 필수**
- **출처 자산이 변경되어도 prelude 는 영향 없음** (self-contained 의 가치)
