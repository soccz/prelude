# 연구·검증 노트: upside_head_challenger_v1

작성: 2026-07-25 · production R1/라벨/알림/ledger 미수정 · 자동주문 없음

## 판정

## VERDICT: REJECT — 상방 head 고정라벨 challenger 6종

`P(day-D high/open >= +10%)` 자체의 전 유니버스 판별력은 살아 있다. 그러나 사용자가
실제로 실행 가능한 09:15 이후 Top 3에서는 **하방 동반 폭증과 비용 차감 손실**을 해결하지
못했다. discovery에서 선택된 `cls_bvol`은 잠금 180일 holdout에서:

- 실행 후 `up10=30.71%`, `safe_up10=(up10 & not dn5)=16.48%`
- 동시에 `dn5=68.91%`, SL-first `65.36%`
- TP5-before-SL3 `33.90%`
- 비용 0.15% 차감 net `-0.4103%/픽`, 누적 `-54.46%`, MDD `-53.69%`

기존 `cls_core` 대비 safe-up10 개선은 `+0.56%p`, paired day-bootstrap
CI95 `[-1.12,+2.43]%p`; net은 오히려 `-0.0788%p/픽`,
CI95 `[-0.2485,+0.0904]%p`였다. 둘 다 0을 포함한다. **상방 전체 AUC가 좋다는 이유만으로
SHADOW 판정을 주지 않는다.**

## 무엇을 고정하고 무엇을 시도했나

라벨 정의는 변경하지 않았다.

```text
y_up10(D) = 1[high_D / open_D - 1 >= 0.10]
```

사전 고정한 모델 후보는 6개이며 하이퍼파라미터 sweep은 하지 않았다.

| 후보 | 차이 |
|---|---|
| `cls_core` | 기존 24피처 XGB binary classifier, raw score 정렬 |
| `cls_bvol` | core + Binance D-1 `b_vol_surge` + missing flag |
| `cls_bvol_volbalanced` | bvol 모델을 D-1 ATR 5분위 안에서 class-balanced 학습 |
| `rank_core` | 일별 cross-section XGB pairwise ranker |
| `rank_bvol` | ranker + Binance D-1 volume surge |
| `rank_novol_bvol` | ranker에서 ATR/RV 수준 피처 제거 + bvol |

추가로 현 결함을 재현하는 `legacy_core_in_sample_bucket`을 참고선으로만 계산했다. 이 행은
별도 모델 fit 수에 넣지 않았고 채택 대상도 아니다.

## 시간정합성·누수·보정 감사

- 입력: `build_market_features`가 market별로 하루 shift한 `D-1` 이하 피처만 사용.
- 타겟: day-D open/high로만 생성. `next_*`, `lab_*`, outcome 열은 feature 0개.
- 유니버스: 각 날짜 `D-1 quote_volume top100`.
- 신규상장: **그 시점까지 history 70개 이상인 행만 먼저 남긴 뒤** 횡단 순위를 계산.
  총 `17,871`행을 제거했고 실제 최솟값은 `history_n=70`.
- discovery: expanding WF 5-fold, train/test 사이 embargo 5일.
- 확률 보정: 각 outer-train 내부의 expanding OOF 3-fold 예측만으로 isotonic을 fit하고
  outer-test에 적용. raw 정렬 성능과 calibrated probability를 분리했다.
- 보정 단조성: 모든 outer fold/후보에서 raw score 증가에 따라 isotonic 출력 비감소 확인.
- 최종 잠금: 최근 180일 `2026-01-26~2026-07-24`를 후보 선택 전에 SHA256
  `3d16c191...68dbd6c`로 잠그고, discovery 선택 후 `cls_bvol`과 `cls_core`만 한 번 평가.
- 비용: 15분 경로 gross return에서 왕복 `0.0015`를 정확히 한 번 차감.
- survivorship: DB에 보존된 inactive market도 포함; 학습 가능 257 market.

## 실행시각 경로

09:10 전후 알림을 09:00 open에 체결했다고 가정하지 않았다. 다음 실행 가능 15분 grid인
`[D 09:15, D+1 09:15)` 96개 봉을 썼다.

- `ledger.path_quality.assess_15m_window` 계약과 배치 구현을 12쌍씩 교차검증.
- KRW-BTC 96-grid와 다음 boundary가 완결된 날만 사용.
- 대상 코인에만 없는 무체결 봉은 직전 close로 flat-fill.
- 같은 봉에서 SL/TP 동시 도달 시 보수적으로 SL 우선.
- locked holdout 경로 완결률 `98.54%`.
- 정책별 Top 3 세 픽이 모두 완결된 날짜만 paired 평가.

## 핵심 결과

### 1. 전체 유니버스에서는 상방 label 판별력이 살아 있음

잠금 holdout:

| 후보 | raw AUC | calibrated AUC | 일별 macro AUC | within-vol AUC | Brier | score↔ATR Spearman |
|---|---:|---:|---:|---:|---:|---:|
| `cls_core` | 0.7438 | 0.7427 | 0.7470 | **0.7369** | 0.07965 | 0.616 |
| `cls_bvol` | **0.7450** | **0.7442** | **0.7494** | 0.7359 | 0.07957 | 0.618 |

`b_vol_surge`의 AUC 증가는 약 `+0.0012`로 작다. within-vol AUC도 core보다 좋아지지
않았다. 따라서 “상방 head가 모든 곳에서 죽었다”가 아니라, **전체 후보 분류는 가능하지만
실제 Top 3의 low-downside 목적과 맞지 않는다**가 정확한 결론이다.

### 2. Top 3 실행 성과는 실패

잠금 180일, 세 픽 모두 경로 완결인 날짜 기준:

| 정책 | 완결일/픽 | up10 | safe-up10 | dn5 | TP5-before-SL3 | SL-first | net/픽 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cls_bvol` | 178 / 534 | 30.71% | 16.48% | **68.91%** | 33.90% | 65.36% | **-0.4103%** |
| `cls_core` | 178 / 534 | 30.52% | 15.92% | 68.35% | 34.64% | 64.04% | -0.3315% |
| ATR Top 3 | 175 / 525 | 21.90% | **17.33%** | 41.33% | 32.00% | 54.10% | **-0.1582%** |
| random seed42 | 173 / 519 | 8.48% | 7.51% | 24.86% | 18.69% | 43.93% | -0.4429% |

`cls_bvol`은 무작위보다 up10은 많이 잡지만 dn5도 `+43.35%p` 늘린다. ATR Top 3보다
up10은 높지만 safe-up10은 오히려 낮고 net도 더 나쁘다. 즉 fixed +10% barrier가
고변동성 정렬로 재포장되는 교락이 실전 Top 3에서 여전히 크다.

### 3. 잠금 holdout paired day-bootstrap

| 비교 | 지표 | Δ challenger-baseline | CI95 | 판정 |
|---|---|---:|---:|---|
| vs `cls_core` (178일/각 534픽) | safe-up10 | +0.56%p | [-1.12,+2.43] | 무차이 |
|  | net/픽 | -0.0788%p | [-0.2485,+0.0904] | 무차이·점추정 악화 |
|  | TP5-before-SL3 | -0.75%p | [-2.81,+1.50] | 무차이 |
| vs random (173일/각 519픽) | safe-up10 | +9.25%p | [+5.20,+13.29] | 상승포착은 유효 |
|  | dn5 | **+43.35%p** | [+37.57,+49.13] | 하방 대폭 악화 |
|  | net/픽 | +0.0656%p | [-0.3940,+0.5015] | 양전·우위 입증 실패 |
| vs ATR Top 3 (175일/각 525픽) | safe-up10 | -0.57%p | [-4.95,+3.81] | 우위 없음 |
|  | net/픽 | -0.2356%p | [-0.7149,+0.2257] | 우위 없음 |

bootstrap은 날짜를 cluster 단위로 2,000회 재표집했고 seed=42다.

## Binance D-1 실가용성 결함

`b_vol_surge(D-1)` 봉은 UTC 00:00 = KST 09:00에 막 마감된다. 그런데 현재
`daily_run_distribution.sh`에서:

- R1 발송: line 70, step 3
- Binance D1 refresh: line 145, step 10

즉 `cls_bvol`이 선택됐더라도 **09:10 R1 발송 시점에 막 닫힌 D-1 Binance 봉이 현재
운영 순서상 아직 DB에 없다.** collector를 R1 앞으로 옮기는 운영 변경 없이는
point-in-time 배포 불가다. 이번 결과는 net도 음수이므로 운영 순서를 바꿀 근거도 없다.

## 해석과 다음 방향

1. true-inner-OOF 보정은 기존 in-sample bucket 결함을 고쳤지만 Top 3 손실을 구하지 못했다.
2. pairwise ranker와 volatility-balanced 학습도 fixed up10 목적의 하방 동반을 제거하지 못했다.
3. `b_vol_surge`는 full-universe AUC를 거의 늘리지 않았고, 실행 net·safe-up 우위도 CI에서
   사라졌다.
4. 따라서 현 라벨을 그대로 둔 “상방 head만 더 잘 fit”하는 축은 배포 가치가 없다.
5. 다음 연구가 필요하다면 학습 목표 자체를 실행시각 first-passage/safe-up으로 바꾸거나,
   하방 제약을 objective에 직접 넣어야 한다. 이는 라벨/architecture 변경이므로 사용자 승인
   사안이며 이 연구에서는 실행하지 않았다.

## 산출물

- `scripts/upside_head_challenger_v1.py`
- `output/upside_head_challenger_v1_predictions.csv.gz`
- `output/upside_head_challenger_v1_metrics.csv`
- `output/upside_head_challenger_v1_calibration.csv`
- `output/upside_head_challenger_v1_paired_bootstrap.csv`
- `output/upside_head_challenger_v1_coverage.json`

production 파일 변경: 없음.
