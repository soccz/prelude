# Challenger quant-evaluator v1 — 1차 독립 감사 작업 메모

> 상태: **WORKING / NOT FINAL**  
> 감사일: 2026-07-25 KST  
> 최종 판정서 예약 경로:
> `_workspace/challenger_quant_evaluator_verdict_v1.md`  
> 보류 사유: first-passage 연구 산출물이 아직 전달되지 않았다. 이 문서는 현재
> 세 후보(downside veto, upside head, safe-up head)와 공통 forward
> score-label 인프라에 대한 1차 감사만 고정한다.

## 1. 먼저 내리는 1차 결론

| 연구 후보 | 독립 상태 | 사용자 목적 기준 핵심 이유 | 운영 연결 |
|---|---|---|---|
| downside veto v1 | **REJECT** | 정확한 09:15 재산출에서 하방은 줄지만 net edge가 없고 상승·safe-up을 같이 잃음. 원 연구의 09:00 수치는 운영 표준 비교에 사용 불가 | 없음 |
| upside `cls_bvol` | **REJECT** | holdout net 음수, core 대비 개선 CI 미확정, dn5 68.91%. fresh Binance D-1가 발송 시점에 없음 | 없음 |
| direct `safeup_head` | **REJECT** | safe-up은 늘지만 dn5도 15.07pp(R1 repaired 대비) 증가. 사용자의 “낮은 하락 + 높은 상승” 목적에 부적합 | 없음 |
| post-hoc safe-up Pareto | **REJECT** | holdout을 본 뒤 만든 규칙이고 net -0.2790%/픽, safe-up 10.85% | 없음 |

세 후보는 모두 연구 스크립트와 격리 artifact일 뿐 현재 R1 발송·원장·스케줄러에
연결되지 않았다. 따라서 이번 결과가 오늘 추천 품질을 바꾼 것은 없다.

## 2. 독립 재계산 방법과 고정 입력

연구자의 summary/보고서 결론을 신뢰하지 않고 row artifact에서 날짜별 Top-3
평균을 다시 만들고, 별도 seed `20260725`와 20,000회 날짜-cluster bootstrap으로
paired CI를 재산출했다.

재계산기:

- `_workspace/recalc_challenger_quant_audit_v1.py`
- SHA256:
  `5fbecf270b189598976879a865c904e427fa8514e392d4de6137dbe325c27b2c`
- 파일 쓰기 없음. 세 row artifact와 D1/15m SQLite를 read-only로 읽는다.

핵심 row artifact:

| artifact | SHA256 |
|---|---|
| `output/downside_veto_challenger_v1_picks.csv` | `83e959749fac0d428ded5719a62f80024276647f7a40801e371267c76f396407` |
| `output/upside_head_challenger_v1_predictions.csv.gz` | `1312459b4133cd9d4e41dd1efb3bc5dc04a687c44eee2fcd712f3a1334020ef4` |
| `output/safeup_head_challenger_v1_picks.csv.gz` | `ed9c61e1c9c17767a0eb2ecc0372b87e7488d9c2bc201ea6650cd9fc621317f3` |

검증 명령:

```bash
python -u _workspace/recalc_challenger_quant_audit_v1.py
python -m pytest -q \
  tests/test_path_quality.py \
  tests/test_close_paper_path_quality.py \
  tests/test_recommend_score_labels.py \
  tests/test_evaluate_recommend_score_labels.py \
  tests/test_recommend_snapshot.py
```

결과: 재계산 완료, 표적 테스트 **27 passed in 7.66s**.

## 3. Downside veto v1 — 원 15m 수치는 폐기, 정책은 REJECT

### 3.1 치명적 감사 결함 3건

#### A. 실행 경로가 09:15가 아니라 09:00이다

`scripts/downside_veto_challenger_v1.py:280-283`은 `D 09:00`부터 96봉을
만든다. 실제 open R1은 보통 09:10 전후 도착하므로 현재 표준은
`[D 09:15, D+1 09:15)`이다.

따라서 기존 `m15_common_complete_locked_180` 수치는 “15m를 썼다”는 점만으로
운영 실행 성과가 되지 않는다. **운영 표준 비교·향후 보고서 baseline으로 재사용
금지**한다.

#### B. point-in-time 70봉 gate가 없다

입력 `cc_filtered_multiday_oos_v1.parquet`은
`scripts/cc_filtered_multiday_v1.py:124-139`에서 구형
`downside_head_riskreward_v1.build_panel`을 쓴다. 그 빌더는 종목의 최종 전체
수명이 70봉 이상인지 한 번만 검사하고
(`scripts/downside_head_riskreward_v1.py:125-154`), 역사적 각 날짜에 prior
70봉이 있었는지는 검사하지 않는다.

D1 원천 timestamp와 직접 대조한 결과:

| 구간 | 후보 행 | prior < 70 | 영향 날짜 |
|---|---:|---:|---:|
| 전체 OOS | 76,500 | 7,164 (9.36%) | 752 / 765 |
| locked 180일 | 18,000 | 1,225 (6.81%) | 167 / 180 |

locked 선택 픽에도 직접 들어갔다.

| 정책 | prior < 70 픽 / 540 | 영향 날짜 | 최소 prior |
|---|---:|---:|---:|
| R1 baseline | 38 | 34 | 2 |
| top-third veto | 22 | 19 | 7 |
| absolute veto | 39 | 35 | 2 |
| lexicographic | 42 | 42 | 26 |

즉 coverage의 “static PIT Top-100” 주장은 사실이 아니다. 더구나 invalid 종목을
제거한 뒤 다음 유동성 순위 종목을 복원할 row도 artifact에 없으므로 사후 필터만으로
정상 Top-100을 완전히 재구축할 수 없다.

#### C. 저장된 p_up/p_dn은 진짜 OOF calibration이 아니다

구형 `walk_forward_heads`는 outer-train 모델의 **train in-sample**
`predict_proba(Xtr)`를 만들고
(`scripts/downside_head_riskreward_v1.py:247-257`), 그것을
`_oof_bucket_calib`에 넣는다. 함수명·주석과 달리 inner OOF score가 아니다.

따라서 downside 연구는 outer test 예측 자체는 과거→미래지만, 그 확률 수준과
R1 비율에 들어가는 calibration은 optimistic한 in-sample mapping이다. 이 defect는
이번 veto 연구 안에서 고쳐지지 않았다.

### 3.2 같은 픽을 정확한 09:15 경로로 다시 계산한 결과

선택은 그대로 두고 경로만 canonical `assess_15m_window`와 동일한
`[09:15,+24h)`로 교체했다. 178 공통 날짜, 534픽/정책이며 canonical complete
path 10건을 byte/metadata 대조했다.

| 정책 | net/픽 | SL-first | dn5 | up10 | safe-up10 |
|---|---:|---:|---:|---:|---:|
| R1 baseline | -0.2298% | 51.50% | 36.52% | 16.29% | 14.98% |
| top-third veto | -0.2207% | 46.25% | 27.90% | 14.04% | 12.73% |
| absolute 50% veto | -0.2148% | 51.31% | 36.52% | 16.10% | 14.79% |
| lexicographic risk-first | -0.2311% | 17.23% | 5.62% | 0.94% | 0.94% |

top-third veto − R1, 날짜 paired:

| metric | delta | 독립 CI95 |
|---|---:|---:|
| net | +0.0091pp | [-0.2632, +0.2816]pp |
| SL-first | -5.24pp | [-9.18, -1.31]pp |
| dn5 | -8.61pp | [-12.36, -4.87]pp |
| up10 | -2.25pp | [-4.68, +0.19]pp |
| safe-up10 | -2.25pp | [-4.68, +0.19]pp |

해석:

- 하방 감소 방향은 재현된다.
- 그러나 net 개선은 사실상 0이고 CI가 넓게 0을 지난다.
- 사용자에게 중요한 상승/safe-up 표본을 같이 잃는다.
- absolute veto는 거의 발동하지 않아 baseline과 동일하다.
- lexicographic는 하방과 함께 상방을 거의 전부 제거한다.

기존 09:00 경로에서는 top-third의 net delta가 +0.0670pp였으나, 실행 가능한
09:15로 옮기면 +0.0091pp로 소멸한다. 이 민감도 자체가 원 수치를 폐기해야 하는
근거다.

### 3.3 holdout·trial 해석

- 180일은 스크립트 내부에서 마지막 구간으로 잘랐을 뿐 프로젝트 전체의 virgin
  holdout이 아니다.
- q=1/3 방향은 앞선 pseudo-veto 관찰의 영향을 받았다.
- coverage도 R2 4개, R3 gate 3개, A1 8개 등 관련 선행 변형 최소 15개를
  인정한다.
- 그러므로 “새로운 sealed confirmatory holdout”이 아니라 역사적 민감도
  확인으로만 취급한다.

최종 상태: **REJECT**. 코드 결함을 고쳐 재실험할 우선순위도 낮다. 정확한 09:15
감사에서도 사용자 목적에 맞는 net+safe 동시 개선이 없기 때문이다.

## 4. Upside head `cls_bvol` — 좋은 전체 AUC가 좋은 추천은 아니다

### 4.1 위생상 살아 있는 부분

- D-1 feature, outer expanding WF, 5일 embargo, inner expanding OOF isotonic은
  코드상 구현되어 있다.
- locked holdout에는 discovery에서 선택한 `cls_bvol`과 `cls_core`만 적합한다
  (`scripts/upside_head_challenger_v1.py:671-730`).
- 저장 artifact의 모든 variant/date는 정확히 100행이었다.
- 09:15 96봉, BTC complete grid, target-only flat-fill, 같은 봉 SL-first, 비용
  0.15% 1회 차감을 독립 path rebuild로 재현했다.

### 4.2 남은 위생 결함

#### A. 70 “prior bars” 계약 off-by-one

`history_n = cumcount()+1`, `history_n >= 70`
(`scripts/upside_head_challenger_v1.py:233-246`)이라 row D 이전 완료봉은 최소
69개다. 감사 계약의 “D 이전 완료 70봉”보다 한 봉 느슨하다.

locked core 18,000행 중 prior=69가 20행/20일, `cls_bvol` Top-3에는 3행/3일
들어갔다. 결론을 바꿀 규모는 아니지만 coverage의 “minimum prior 70” 표현은
정확하지 않다.

#### B. 선택 feature가 09:10에 존재하지 않는다

`cls_bvol`의 D-1 Binance UTC 일봉은 KST 09:00에 막 닫힌다. 현재 runner는
R1을 line 70에서 발송하고 fresh Binance 수집은 line 145에서 한다
(`scripts/daily_run_distribution.sh:67-80,142-146`).

따라서 연구 artifact에서 사용한 fresh `b_vol_surge`는 현 운영 시점에 이용
불가능하다. ops 순서를 바꾸거나 하루 더 lag하지 않는 한 deployable PIT feature가
아니다.

#### C. 보조 검증이 충분히 강하지 않다

- exact bracket random baseline은 seed 42 단 한 번이다. 1,000회 random 분포가
  아니다.
- `macro_within_vol_auc`는 전체 기간을 5개 band로 묶은 뒤 AUC 평균을 낸다
  (`scripts/upside_head_challenger_v1.py:795-800`). same-day within-band
  control이 아니다.
- matched-vol base에는 선택 픽 자체도 포함되어 lift가 약간 수축된다
  (`scripts/upside_head_challenger_v1.py:855-863`).
- 이 약점들은 후보가 이미 REJECT이므로 승격 오류를 만들지는 않는다.

### 4.3 독립 locked path 재계산

178 complete 공통 날짜, 534픽/정책:

| 정책 | net/픽 | SL-first | dn5 | up10 | safe-up10 |
|---|---:|---:|---:|---:|---:|
| selected `cls_bvol` | -0.4103% | 65.36% | 68.91% | 30.71% | 16.48% |
| `cls_core` | -0.3315% | 64.04% | 68.35% | 30.52% | 15.92% |

`cls_bvol` − core, 날짜 paired:

| metric | delta | 독립 CI95 |
|---|---:|---:|
| net | -0.0788pp | [-0.2506, +0.0921]pp |
| safe-up10 | +0.56pp | [-1.12, +2.43]pp |
| dn5 | +0.56pp | [-1.50, +2.62]pp |
| up10 | +0.19pp | [-1.69, +2.06]pp |
| SL-first | +1.31pp | [-0.94, +3.56]pp |

전체 18,000행의 raw AUC는 `cls_bvol=0.7450`, `core=0.7438`로 높지만 Top-3
추천 개선은 없다. 높은 AUC와 높은 up10은 high-volatility pick을 고르는 대가로
dn5 68.91%를 만든다. 사용자 목적에는 명백한 실패다.

holdout 2026-01-26~2026-07-24 역시 프로젝트의 기존 연구·live 관측과 겹치므로
전역 virgin holdout으로 간주하지 않는다. 다만 이 후보는 내부 holdout에서도
실패했으므로 상태는 변하지 않는다.

최종 상태: **REJECT**.

## 5. Direct safe-up head — 확률 head는 살아도 선택 정책은 실패

### 5.1 위생상 가장 깨끗한 연구

- row D 이전 완료봉을 실제로 `history_prior_bars >= 70`으로 먼저 자른 뒤
  cross-sectional universe를 만든다
  (`scripts/safeup_head_challenger_v1.py:147-201`).
- 24 D-1 feature 계약, outer expanding WF, 5일 embargo, train-only medians,
  진짜 inner expanding OOF isotonic이 구현되어 있다
  (`scripts/safeup_head_challenger_v1.py:258-407,475-500`).
- primary `safeup_head`는 holdout 전에 고정했다.
- 09:15 96봉 경로, BTC 완결성, target-only flat-fill, 비용 1회, 날짜 paired
  bootstrap을 row artifact에서 재현했다.
- `safeup_pareto_rank`가 holdout 사후 규칙이라는 사실도 artifact에 명시했다.

### 5.2 목표 정렬의 구조적 한계

학습 label은 day-open 09:00 기준
`up10 AND NOT dn5`다
(`scripts/safeup_head_challenger_v1.py:203-207`). 실제 사용자는 09:10 알림 뒤
09:15에 진입한다. 따라서 primary label 자체는 09:00~09:15 구간과 09:00 open을
포함하는 비실행 목표이며, 실제 판단은 09:15 secondary path에 의존해야 한다.

또 binary safe-up 확률만 최대화하면 “safe-up이 아닌 실패” 중 큰 하방에 별도
패널티를 주지 않는다. high-volatility 종목은 safe-up chance와 dn5 chance를
동시에 높일 수 있다. 이번 결과가 정확히 그 실패를 보인다.

### 5.3 독립 locked path 재계산

6정책 공통 172일, 516픽/정책:

| 정책 | net/픽 | SL-first | dn5 | up10 | safe-up10 |
|---|---:|---:|---:|---:|---:|
| `safeup_head` | -0.0797% | 59.11% | 54.07% | 23.45% | 17.05% |
| `R1_repaired` | -0.1805% | 46.90% | 39.15% | 16.86% | 11.43% |
| `R1_frozen_pattern` | -0.1000% | 47.87% | 30.81% | 14.92% | 11.82% |
| `up10_control` | -0.3586% | 64.53% | 67.44% | 30.81% | 16.28% |
| monkey seed42 | -0.4336% | 43.60% | 24.61% | 8.14% | 7.17% |
| post-hoc Pareto | -0.2790% | 45.54% | 24.81% | 12.79% | 10.85% |

정책별 complete 날짜만 pairwise 교집합으로 다시 계산한 `safeup_head −
R1_repaired` (177일):

| metric | delta | 독립 CI95 |
|---|---:|---:|
| safe-up10 | +5.65pp | [+2.07, +9.23]pp |
| up10 | +6.78pp | [+2.82, +10.73]pp |
| dn5 | **+15.07pp** | **[+10.92, +19.40]pp** |
| SL-first | +12.43pp | [+7.16, +17.70]pp |
| net | +0.0805pp | [-0.2843, +0.4474]pp |

즉 “상승할 가능성”만 보면 개선이 있지만 “하락할 가능성이 낮으면서”라는 필수
조건을 크게 위반한다. net 개선도 증명되지 않았다.

path가 존재하는 discovery fold는 3·4뿐이며 `safeup_head` net은 각각
-0.6509%, -0.3557%, locked는 -0.0797%다. 사용 가능한 세 시간 블록 모두
음수라 fold 안정성도 0/3이다.

post-hoc Pareto는 하방을 낮췄지만:

- holdout 결과를 본 뒤 만든 규칙이라 confirmatory evidence가 아니다.
- safe-up 10.85%와 net -0.2790%로 사용자 목적을 만족하지 못한다.

최종 상태: primary와 post-hoc 모두 **REJECT**.

## 6. 공통 경로·forward score-label 인프라 1차 감사

### 6.1 통과한 계약

`ledger/path_quality.py`, `signals/recommend_score_labels.py`,
`scripts/evaluate_recommend_score_labels.py`를 별도로 읽고 표적 테스트를 돌렸다.

- arbitrary aligned `start_at`부터 정확히 96개 15m grid
- KRW-BTC 96 timestamp + 다음 boundary 관측으로 마지막 봉 마감 확인
- benchmark gap은 collection failure
- target-only no-trade gap만 prior close로 flat-fill
- 첫 target gap에 prior close가 없거나 OHLC가 비정상이면 incomplete
- delivery `sent_at`을 15분 grid로 ceil: 09:10 → 09:15
- 실행 시점부터 새 24시간 경로를 사용하므로 09:00 소급 적중 없음
- 같은 봉 TP/SL 동시 도달은 SL-first
- TP/SL/EOD와 eod return 모두 0.15% 비용 1회 차감
- evaluator는 완전한 net field가 있으면 비용을 다시 빼지 않음
- scheduled replay와 observed forward를 분리
- snapshot/label payload hash, partial 재시도, complete idempotency

### 6.2 주의점

- delivery 실패/no-receipt snapshot도 실제 목표일에 생성됐으면
  `forward_observed` all-score cohort에 들어간다. 이는 모델 score 관측에는
  타당하지만 **사용자가 실제로 받은 추천 성과는 반드시 `delivered` cohort만**
  별도로 봐야 한다.
- evaluator는 fixed TP5/SL3 first-passage만 읽는다. 고정 barrier의 변동성
  교락을 제거할 ATR-normalized barrier 결과는 아직 없다.
- complete artifact만 기본 평가에 쓰므로 universe 동일성은 지키지만, partial
  날짜가 발생할 때 제외 패턴을 coverage와 함께 계속 감시해야 한다.
- full-universe score 기록이 실제 성능 증거가 되려면 날짜 수가 쌓여야 한다.

### 6.3 현재 실제 표본은 0

2026-07-25 감사 시점:

- `output/recommend_snapshots`: 실제 snapshot 0개
- `output/recommend_score_labels`: label artifact 0개
- `output/recommend_score_label_evaluation.json`:
  `found=0`, `complete_used=0`, channel 0개

따라서 forward score-label 인프라는 코드·테스트 기준으로는 준비됐지만 **실제
forward 추천 품질에 대한 증거는 아직 하나도 제공하지 않는다**. “인프라 완성”과
“모델 검증 완료”를 섞으면 안 된다.

## 7. 아직 남은 최종 감사

first-passage 연구 산출물이 도착하면 다음을 수행한 뒤에만 최종
`challenger_quant_evaluator_verdict_v1.md`를 만든다.

1. row artifact에서 fixed/ATR barrier 결과 전량 재산출
2. 날짜 paired CI와 공통 날짜 교집합 재검산
3. 09:15 96봉, BTC completeness, flat-fill, same-bar SL-first 확인
4. 비용 0.15%가 정확히 1회인지 대조
5. D-1/PIT history/universe/embargo/inner OOF 확인
6. candidate 선택과 holdout unlock 순서, post-hoc contamination, 전체 trial 수 확인
7. within-day volatility band에서 ATR label이 fixed barrier의 변동성 재포장을
   실제로 줄였는지 확인
8. 기존 세 후보와 함께 최종 SHADOW/REJECT 표 작성

현재까지 승격 가능한 후보는 **0개**다.
