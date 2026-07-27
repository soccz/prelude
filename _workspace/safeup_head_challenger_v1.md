# safeup head challenger v1 — 독립 연구·적대 검증

## 결론

### VERDICT: REJECT — `safeup_head`

`safe_up10 = (당일 고가/시가-1 ≥ +10%) AND (당일 저가/시가-1 > -5%)`를
하나의 이진 라벨로 직접 학습하는 방법은 **안전한 상승 적중률은 높였지만 하락
가능성을 낮추지 못했다.** 운영·알림·레지스트리에 연결하지 않는다.

- 잠금 180일 D1 Top 3: safe_up10 `17.59%`, up10 `25.74%`, dn5 `57.41%`
- 실제 추천 실행 렌즈 공통 172일 `[09:15, 익일 09:15)`:
  safe_up10 `17.05%`, up10 `23.45%`, dn5 `54.07%`,
  TP5-first `36.43%`, SL3-first `59.11%`
- 비용 0.15% 1회 차감 net `-0.0797%/픽`,
  날짜 bootstrap CI95 `[-0.4406%, +0.2816%]`
- day-equal 누적 `-16.94%`, Sharpe `-0.64`, Sortino `-0.95`,
  MaxDD `-30.83%`
- discovery 09:15 경로 net은 `-0.5119%/픽`,
  CI95 `[-0.7824%, -0.2430%]`로 명확히 음수였다.

핵심 원인은 라벨 구조다. `safe_up10=0` 안에 “조용하고 안전하지만 +10%가
아닌 날”과 “크게 하락한 날”이 함께 들어간다. 모델은 희소한 +10% 사건을
잡기 위해 여전히 고변동 종목을 상단에 놓았고, 그 결과 상승과 하락이 함께
증가했다. 잠금 구간 safe_up10 within-ATR lift는 `1.735배`로 단순 변동성
재포장만은 아니지만, dn5는 같은 ATR 조성 기대치보다 `+22.12%p` 높았다.

### VERDICT: REJECT — `safeup_pareto_rank` (post-hoc)

`당일 pct-rank(raw_safe_up10) - pct-rank(raw_dn5)`는 원래 잠금 홀드아웃
결과를 본 뒤 추가된 **post-hoc 1회 trial**이다. 따라서 좋아도 ADOPT할 수
없고 최대 forward-only SHADOW 후보였으나, 실제 경로 결과가 음수이고
CI도 불확실해서 현재 형태 자체를 REJECT한다.

- 잠금 180일 D1 Top 3: safe_up10 `12.04%`, up10 `13.70%`, dn5 `22.59%`
- 실제 09:15 공통 172일: safe_up10 `10.85%`, up10 `12.79%`,
  dn5 `24.81%`, TP5-first `23.26%`, SL3-first `45.54%`
- net `-0.2790%/픽`, CI95 `[-0.5855%, +0.0306%]`
- day-equal 누적 `-40.42%`, Sharpe `-2.55`, Sortino `-3.43`,
  MaxDD `-44.65%`

결함 방식 R1 대조군과 비교하면 D1 dn5는 `-9.30%p`
CI95 `[-13.18, -5.43]%p`, 09:15 dn5는 `-6.01%p`
CI95 `[-9.88, -2.13]%p`로 낮아졌다. 그러나:

- D1 safe_up10 차이 `-0.78%p`, CI95 `[-3.88, +2.33]%p`
- D1 up10 차이 `-2.52%p`, CI95 `[-5.81, +0.78]%p`
- 09:15 safe_up10 차이 `-0.97%p`, CI95 `[-3.88, +1.94]%p`
- 09:15 up10 차이 `-2.13%p`, CI95 `[-5.23, +0.97]%p`
- SL3-first 차이 `-2.33%p`, CI95 `[-6.78, +2.13]%p`
- net 차이 `-0.1790%p/픽`, CI95 `[-0.4855, +0.1333]%p`

즉 하방은 줄였지만 안전한 상승은 늘리지 못했고, 비용 차감 결과는 더
나빠졌다. 사용자의 “하락 낮고 상승 높은” 두 축을 동시에 개선하지 못한다.

## 무엇을 돌렸나

### 고정 모델 trial

하이퍼파라미터 sweep 없이 기존 24개 D-1 피처로 세 헤드만 같은 XGBoost
구조에 적합했다.

1. `safe_up10`
2. 기존 `up10` control
3. `dn5`

그 점수로 다음 Top 3 정책을 동일 후보군에서 비교했다.

1. `safeup_head`
2. `up10_control`
3. `R1_repaired`: true-inner-OOF isotonic `p_up10 / p_dn5`
4. `R1_frozen_pattern`: outer-train in-sample·비단조 bucket을 재현한 결함 대조군
5. `monkey_seed42`
6. `safeup_pareto_rank` — 잠금 결과 확인 뒤 요청된 post-hoc 1회

`R1_frozen_pattern`은 기존 결함의 알고리즘적 재현 대조군이며, 과거 실제
텔레그램 발송 종목을 byte-identical하게 재생한 것이라고 주장하지 않는다.

## 잠금 홀드아웃 비교

아래 D1 수치는 전체 잠금 180일, 경로 수치는 여섯 정책 모두 Top 3 경로가
완결된 동일 172일만 사용했다.

| 정책 | D1 safe | D1 up10 | D1 dn5 | 09:15 safe | 09:15 up10 | 09:15 dn5 | TP5 | SL3 | net/픽 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| safeup_head | 17.59% | 25.74% | 57.41% | 17.05% | 23.45% | 54.07% | 36.43% | 59.11% | -0.0797% |
| up10_control | 13.70% | 32.41% | 73.33% | 16.28% | 30.81% | 67.44% | 34.30% | 64.53% | -0.3586% |
| R1_repaired | 11.85% | 18.89% | 41.48% | 11.43% | 16.86% | 39.15% | 25.97% | 46.90% | -0.1805% |
| R1_frozen_pattern | 12.59% | 15.93% | 32.04% | 11.82% | 14.92% | 30.81% | 28.49% | 47.87% | -0.1000% |
| monkey_seed42 | 6.85% | 9.44% | 27.96% | 7.17% | 8.14% | 24.61% | 18.60% | 43.60% | -0.4336% |
| safeup_pareto_rank* | 12.04% | 13.70% | 22.59% | 10.85% | 12.79% | 24.81% | 23.26% | 45.54% | -0.2790% |

\* post-hoc, promotion-ineligible.

`safeup_head`와 `up10_control`의 공통 172일 paired 비교:

- D1 safe_up10 `+3.68%p`, CI95 `[+0.39, +6.98]%p`
- D1 up10 `-6.59%p`, CI95 `[-10.85, -2.33]%p`
- D1 dn5 `-15.31%p`, CI95 `[-19.19, -11.43]%p`
- 09:15 safe_up10 `+0.78%p`, CI95 `[-2.52, +4.07]%p`
- 09:15 up10 `-7.36%p`, CI95 `[-11.43, -3.49]%p`
- 09:15 dn5 `-13.37%p`, CI95 `[-17.64, -9.30]%p`
- SL3-first `-5.43%p`, CI95 `[-10.08, -0.78]%p`
- net `+0.2789%p/픽`, CI95 `[-0.0758, +0.6312]%p`

기존 up10-only보다 덜 위험해졌지만 상승 포착도 유의하게 줄었고, net 개선은
0을 배제하지 못했다. 더 중요한 것은 R1 대조군보다 dn5·SL이 훨씬 높다는
점이다.

## 헤드 자체의 판별력

잠금 18,000 coin-day에서 true-inner-OOF isotonic 기준:

| 헤드 | raw AUC | calibrated AUC | Brier | 실현 base | 평균 예측 |
|---|---:|---:|---:|---:|---:|
| safe_up10 | 0.694 | 0.694 | 0.0668 | 7.45% | 6.70% |
| up10 | 0.743 | 0.742 | 0.0799 | 9.45% | 7.76% |
| dn5 | 0.738 | 0.738 | 0.1665 | 26.57% | 26.69% |

따라서 “상방 정보가 전혀 없다”는 결론은 아니다. 기존 운영 상방 헤드의
AUC 0.477과 달리, 제대로 된 outer WF + true inner OOF 구조에서는 up10
판별력이 살아났다. 문제는 up10 상단이 고변동·고하방에 몰리는 것이며,
단일 safe 이진 라벨도 이를 해결하지 못했다.

## 위생·재현 감사

- feature boundary: 모든 24개 피처 `≤ D-1`
- point-in-time history: 모든 평가 row에 prior D1 bars `≥70`
- 평가 후보군: discovery 643일 + holdout 180일 모두 정확히 100종목
- 유니버스: 해당 날짜 D-1 거래대금 Top 100; 정책 여섯 개가 동일 frame 공유
- outer: expanding 5-fold WF, embargo 5일
- calibration: 각 outer-train 내부 expanding 3-fold true OOF isotonic
- holdout: `2026-01-26..2026-07-24`, 180일,
  SHA256 `3d16c1918fceb46f45bebbc825c9c407804b5f944f23e9ff15f44e88668dbd6c`
- 09:15 경로: `ledger.path_quality.assess_15m_window`와 20건 대조,
  그중 완결 경로 10건의 96개 OHLC까지 완전 일치
- 경로 규칙: KRW-BTC 96봉+다음 경계 마감 증거, 대상만 누락 시 flat-fill,
  same-bar SL-first
- 비용: 왕복 0.15% 정확히 1회
- 자동 주문·운영 배선·알림 변경: 없음

### 제한

- 원래 D1 라벨은 09:00 시가 기준이라 09:10 전달 전 움직임을 포함한다.
  최종 판정은 이를 보완한 09:15 실제 실행 렌즈에 더 무게를 뒀다.
- 전체 잠금 180일 중 여섯 정책 모두 완결된 실행 경로는 172일이다.
- discovery 경로 공통 완결은 643일 중 242일이며 초반 세 fold에는 공통
  완결 날짜가 없다. discovery D1은 전 기간 사용했지만 실행 net의 장기
  안정성 주장은 할 수 없다.
- forward observed 표본은 없다. 백테스트만으로 ADOPT하지 않는다.
- Pareto 정책은 잠금 결과를 본 뒤 추가됐으므로 이 180일은 확인 표본이
  아니다.

## 다음 연구 방향

이번 결과가 막은 것은 “safe_up10 단일 이진 라벨이면 자동으로 저하방이
된다”는 가정이다. 다음 시도는 새 홀드아웃 또는 forward에서 다음처럼
명시적으로 두 제약을 분리해야 한다.

1. 상방 head는 이번 true-OOF up10 구조를 유지한다.
2. dn5 head에 절대 위험 budget/gate를 두되, gate는 과거 잠금 결과로
   재튜닝하지 않는다.
3. 채택 기준은 `safe_up10 증가 AND dn5/SL 감소 AND 09:15 net CI 하한 개선`
   을 동시에 요구한다.
4. post-hoc Pareto 식의 재활용은 금지하고, 새 forward 점수만 축적한다.

## 산출물

- `scripts/safeup_head_challenger_v1.py`
- `output/safeup_head_challenger_v1_predictions.csv.gz`
- `output/safeup_head_challenger_v1_picks.csv.gz`
- `output/safeup_head_challenger_v1_summary.csv`
- `output/safeup_head_challenger_v1_folds.csv`
- `output/safeup_head_challenger_v1_heads.csv`
- `output/safeup_head_challenger_v1_paired.csv`
- `output/safeup_head_challenger_v1_coverage.json`

