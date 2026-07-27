# ADDITIONAL_IDEAS.md — 하방은 낮고 상방은 높은 종목을 찾기 위한 추가 아이디어

> 작성 기준일: 2026-07-25
> 상태: **검증 반영 실행안** — `_workspace/ideas_review_orchestrator_v1.md`의 실측 결과를 반영했다.
> 운영 경계: 측정·재현·실패 전파는 즉시 수리하되, R1 순위·라벨·알림 문구와 09-01
> 사전등록 판정은 소급 변경하지 않는다. 새 모델 후보는 승인·검증 전 SHADOW에만 둔다.
> 최종 사용자: 텔레그램 알림을 참고해 직접 판단하고 직접 매매하는 사용자.

---

## 0. 한 줄 결론

prelude의 다음 발전 단계는 **알림 이후 결과를 재현 가능하게 전 유니버스에 기록한 뒤,
살아 있는 하방 head는 수준 보정하고 죽어 있는 상방 head를 다시 세우는 것**이다. 검증된
veto-then-rank 뼈대는 유지하되 지금 가장 큰 병목은 새 하방 veto가 아니라 상방 순위다.

현재 시스템은 연구·수집·shadow 운영 뼈대는 좋다. 그러나 R1 확률 보정, preopen 성과 귀속,
알림 시점과 원장 진입 시점, 성과 집계 방식이 아직 이 목표를 정확히 증명하지 못한다.

### 0.1 실측 검증으로 정정된 핵심

- `p_dn5`는 실현 하방 대비 AUC `0.721`로 판별력이 살아 있다. 다만 실제 빈도를 약
  `1.49배` 과소 추정하므로 새 head보다 OOF 수준 보정이 우선이다.
- `p_up10`은 AUC `0.477`, `corr(rr_ratio, realized)=0.041`로 현재 상방 정렬에 쓸 수 없다.
- R1은 `-0.624%/픽`, 같은 top100 몽키 baseline은 `-0.424%/픽`이었다. 현재 순위는
  무작위보다 낫다고 주장할 수 없다.
- 사후 pseudo-veto는 하방을 줄였지만 `n=162` 사후분석이고 기존 veto 계열이 여러 번
  실패했다. 즉시 운영 개입 근거가 아니라 paired SHADOW 사전등록 후보일 뿐이다.
- 고정 퍼센트 first-passage 키는 변동성 십분위에 `3.7~7.7배` 단조로 묶인 반면 net EV는
  평평했다. 향후에는 고정 barrier와 ATR barrier를 함께 만들고 `within-vol-band lift`로만
  채택한다.
- 가장 큰 즉시 lever는 매일 top3만 남기는 대신 동일 inference의 전 유니버스 약 100개
  score·feature를 저장하는 것이다. 월 표본이 약 90픽에서 약 3,000행으로 늘어난다.

### 0.2 2026-07-25 구현 현황

Track 1의 코드·데이터 수리는 완료했다. 활성 R1 정렬식·라벨·모델 구조·알림 문구는 바꾸지
않았고 실거래 주문도 추가하지 않았다.

| 항목 | 상태 | 확인 결과 |
|---|---|---|
| D1/4h/15m 신규상장·실패 전파 | 완료 | 실제 live KRW `269/269`를 세 DB 모두 확보, DB 간 유니버스 차이 0 |
| 실행시각·경로 완결성 | 완료 | `sent_at` 이후 첫 실행 가능 15분봉부터 새 96봉(24시간) 사용, 마지막 봉의 다음 경계 확인, 대상만 빈 봉이면 flat-fill |
| preopen/open 귀속 | 완료 | 슬롯별 snapshot·receipt·전용 원장 사용, receipt 누락·손상은 active 원장 기록 거부 |
| 단일 inference·전 유니버스 기록 | 완료 | top3와 같은 snapshot에 약 100개 score 및 행당 24개 feature 저장 |
| 확률·순위 사후 평가 | 완료 | `p_up10/p_dn5` AUC·Brier·calibration과 safe-up·dn3/dn5·first-passage·top-N·유동성/within-vol baseline 자동 산출 |
| forward/replay 분리 | 완료 | 목표일 밖 재생은 `scheduled_replay`로 기록하고 기본 forward 통계에서 제외 |
| 공통 성과 집계 | 완료 | day equal-weight, 무거래일 cash, 복리, 초기자본 MDD, `√365`, 날짜-cluster CI |
| 비용·운영 실패 | 완료 | v2 이중차감 제거, Telegram/수집/청산/평가 실패 nonzero 전파, 성공 receipt 동시성 보호 |
| systemd 실패 알림 | 소스 완료·설치 대기 | 7개 service의 `OnFailure`와 cron 중복 차단 준비. 실제 `/etc` 반영은 sudo 1회 필요 |

검증은 전체 `177 passed`, 임시 end-to-end `snapshot 100행 → 24h label 100행 → evaluator`
완료로 확인했다. 재생 100행은 실제로 forward `n=0`에서 제외됐고, 63개 무체결 gap은
저유동 종목을 버리지 않고 flat 경로로 복원됐다.

중요한 한계는 그대로다. 이 작업은 **측정을 믿을 수 있게 만든 것**이지 AUC `0.477`인
상방 head를 고친 것이 아니다. 현재 R1은 계속 baseline/SHADOW 수준으로 해석해야 하며,
“하락 가능성이 낮고 상승 가능성이 높은 추천”의 모델 개선은 새로 쌓이는 forward 표본과
09-01 이후 승인된 OOF·상방 head 재구축으로 판정한다.

### 0.3 남은 challenger 실행 결과

사용자가 2026-07-25에 남은 연구를 계속 진행하라고 명시해, 기존 동결 블록의 문구는
수정하지 않은 채 별도 historical challenger로 실행했다. 따라서 아래 결과는 코드 결함을
찾고 후보를 폐기하는 데는 쓸 수 있지만, 깨끗한 사전등록·virgin holdout 증거는 아니다.

| 후보 | 실제 09:15 경로 결과 | 판정 |
|---|---|---|
| downside top-third veto | R1 대비 `dn5 -8.61%p`지만 safe-up `-2.25%p`, net `+0.009%p` CI95 `[-0.263,+0.282]` | REJECT |
| Binance 상방 head | safe-up `16.48%`, `dn5 68.91%`, net `-0.410%/픽`; fresh Binance D-1도 R1 발송 뒤에 수집 | REJECT |
| direct safe-up head | R1 대비 safe-up `+5.65%p`와 함께 `dn5 +15.07%p`; net 차이 CI가 0 포함 | REJECT |
| fixed first-passage head | R1 대비 safe-FP `+5.89%p`지만 `dn5 +20.12%p`; net 차이 CI95 `[-0.253,+0.522]%p` | REJECT |
| core + downside semivol 4개 | core 대비 safe-FP 변화 0, `dn5 -2.17%p`도 CI가 0 포함, net 악화·AUC `0.705→0.698` | REJECT |
| lowest-ATR 진단 baseline | `dn5 4.88%`로 낮지만 `up10 0.61%`, safe-FP `0.41%`, net `-0.360%/픽` | 저하방만 얻고 상방을 잃는 퇴행 |

독립 재계산까지 거친 결론은 **채택 0, SHADOW 0**이다. 활성 R1·텔레그램·원장은
변경하지 않았다. 현재 D-1 피처는 상방 사건을 구분하는 정보는 일부 갖고 있지만, 그
상승과 함께 커지는 고변동·하방을 분리하는 정보가 부족하다. 임계값과 비율식을 더
사후조정하면 같은 기간을 반복해서 보는 데이터 마이닝만 늘어나므로 여기서 historical
탐색을 닫는다.

실제 `output/recommend_snapshots`와 `output/recommend_score_labels` forward 표본은
아직 0개다. 다음 정상 스케줄부터 약 100개 전 유니버스 score가 매일 쌓이고, 각 행은
실제 실행 가능 시각부터 24시간이 완전히 지난 뒤 라벨된다. 다음 승격 판단의 우선순위는
추가 backtest 식 찾기가 아니라 이 새 forward 표본에서 `safe-up 증가 + dn5 비악화 +
비용차감 net 비악화`를 동시에 확인하는 것이다.

---

## 1. 시스템 목표 재정의

### 1.1 목표

매일 알림 시점 `t`에 실제로 거래 가능한 업비트 KRW 유니버스에서 다음 조건을 함께 만족할
가능성이 높은 종목을 찾는다.

1. 알림 이후 큰 하락을 먼저 겪을 가능성이 낮다.
2. 알림 이후 의미 있는 상승에 도달할 가능성이 높다.
3. 상승하더라도 그전에 감내하기 어려운 하락을 거치는 종목은 낮게 평가한다.
4. 사용자가 직접 차트·뉴스·호가를 보고 최종 매매 판단을 할 수 있도록 근거를 제공한다.

이를 모델 출력으로 표현하면 다음이 핵심이다.

- `P(MAE <= -3%)`, `P(MAE <= -5%)`, `P(MAE <= -10%)`
- `P(MFE >= +5%)`, `P(MFE >= +10%)`, `P(MFE >= +20%)`
- `P(+5% before -3%)`, `P(+10% before -3%)`
- 예상 MAE와 MFE
- 상승 또는 하락 임계값까지 걸린 시간
- 과거 유사 표본 수와 확률 신뢰도

위의 `3%`, `5%`, `10%`, `20%`는 **현재 비교를 위한 초기값**이다. 실제 데이터와 사용자
체감에 따라 조정하며 절대값으로 고정하지 않는다. 라벨 정의 변경은 사용자 확인 후 진행한다.

### 1.2 비목표

- 실거래 자동 주문
- 사용자 대신 매수·매도 최종 결정
- 한 가지 고정 TP/SL을 모든 상황에 강제
- 논문용 지표를 통과하기 위한 모델 최적화
- gross 수익만 좋아 보이는 전략 채택

### 1.3 결과 우선순위

사용자는 2026-07-25에 직접 매매할 종목의 **추천 품질을 최우선**으로 명시했다. 따라서 다음
순서를 기본 거버넌스로 사용하되, 기존 v2 09-01 사전등록 판정은 동결된 구지표를 그대로
유지하고 새 지표를 병기한다.

1. 하락 선도달 회피율과 최대하락폭
2. 상승 선도달률과 최대상승폭
3. 같은 날 비교 유니버스 대비 top-K lift
4. 확률 calibration과 표본 신뢰도
5. 비용 차감 후 reference PnL·Sharpe·MDD
6. 사용자 실제 선택 결과와 시스템 추천의 차이

PnL은 버리지 않는다. 추천 품질이 실제 금전 결과로도 연결되는지 확인하는 최종 sanity check로
사용한다.

---

## 2. 양보할 수 없는 위생

다음 네 가지는 결과가 좋아 보여도 예외를 두지 않는다.

1. **Look-ahead 방어**: 입력은 의사결정 시점에 실제로 알 수 있던 값만 사용한다.
2. **유니버스 시간정합성**: 학습 fold 종료 시점에 존재했던 종목만 사용한다.
3. **거래비용 차감**: reference PnL에는 수수료와 슬리피지를 항상 포함한다.
4. **실거래 자동 주문 금지**: 사용자 명시 전에는 주문 API와 업비트 API key를 사용하지 않는다.

수치 임계값, 모델 종류, lookback, top-K, regime 구간 등은 모두 데이터 기반으로 바꿀 수 있다.

---

## 3. 현재 상태 진단

### 3.1 잘된 부분

- 데이터·시그널·원장·운영·알림의 책임이 대체로 분리되어 있다.
- R1은 상승 확률, 하락 확률, 예상 하방, risk-reward를 동시에 계산한다.
- 주요 일봉 피처와 BTC regime은 D-1 기준으로 shift된다.
- 학습 cutoff에 embargo가 존재한다.
- R2와 A1은 각각 하방 페널티와 pump-after-dump 위험을 별도 challenger로 다룬다.
- v2는 Binance 거래량 surge를 추가 정보로 사용하되 challenger 잠금을 유지한다.
- 비용이 원장에 명시되고, 동일 15분봉에서 TP·SL 동시 도달 시 SL 우선으로 처리한다.
- 미승인 후보를 shadow ledger에 축적하고 운영 모델과 분리한다.
- 실거래 주문 코드는 없다.

### 3.2 발견 당시 핵심 결함

아래 표는 Track 1 착수 전 진단이다. 현재 수리 상태는 §0.2를 기준으로 본다.

| 우선순위 | 결함 | 추천 품질에 미치는 영향 |
|---|---|---|
| P0 | 09:05 R1 알림이 실제로는 약 09:10에 도착하지만 원장은 09:00 open부터 평가 | 사용자가 알림을 받기 전 움직임이 성과에 포함됨 |
| P0 | 08:50 R1 전용 ledger·closer가 없음 | 실제 preopen 추천의 갭·하락·상승 성과를 측정할 수 없음 |
| P0 | v2 scoreboard가 이미 net인 수익에서 비용 0.15%p를 다시 차감 | 조기 KILL과 판단 수치 왜곡 |
| P0 | 불완전한 15분·4시간 경로도 CLOSED 처리 가능 | 잘못된 TP/SL·MAE·MFE가 영구 저장될 수 있음 |
| P0 | D1 신규 상장 18종 누락 | 신규 종목이 전체 D1 기반 레이더에서 조용히 제외됨 |
| P1 | R1의 OOF calibration이 실제로는 학습 데이터 재예측 | 하락·상승 확률 과신과 순위 왜곡 가능 |
| P1 | 모델별 exit·hit 정의가 다른데 동일 champion 표에서 비교 | 신호와 청산 정책의 효과가 섞임 |
| P1 | 일별 추천 수익을 가중치 없이 합산 | 추천 수가 많은 날이 높은 레버리지처럼 계산됨 |
| P1 | Telegram·close·backup 실패가 exit 0으로 숨겨지는 경로 존재 | “정상 운영”과 “실제 전달 성공”을 구분하기 어려움 |
| P2 | 모델·데이터·코드 snapshot manifest 부족 | 같은 추천을 정확히 재현하기 어려움 |

### 3.3 2026-07-25 기준 forward 참고치

아래 평균은 기존 CLOSED 원장의 비용 반영 수치다. 날짜 단위로 묶은 신뢰구간을 함께 본다.

| 모델 | 거래 / 거래일 | 거래당 평균 | 날짜-cluster 95% CI | 해석 |
|---|---:|---:|---:|---|
| R1 | 162 / 54 | -0.624% | [-1.223%, -0.002%] | 현재 부정적 |
| R2 | 162 / 54 | -0.551% | [-1.112%, +0.015%] | R1보다 약간 낫지만 증거 부족 |
| A1 | 159 / 53 | -0.679% | [-1.226%, -0.102%] | 현재 부정적 |
| pump v1 | 765 / 51 | -0.379% | [-0.719%, -0.053%] | 현재 부정적 |
| pump v2 | 202 / 44 | +0.219% | [-0.438%, +0.631%] | 양수지만 미확정 |

이 표는 현재 원장 결함을 모두 해결한 최종 판단표가 아니다. 특히 R1 open 진입 시각과 R1
preopen 귀속 문제를 먼저 바로잡아야 한다.

---

## 4. 목표 의사결정 구조

```text
시점별 데이터 snapshot
        ↓
유니버스·봉 완결성·freshness 검증
        ↓
슬롯당 단 한 번의 재현 가능한 inference snapshot
        ↓
1단계: downside veto
        ↓
2단계: upside ranking
        ↓
신뢰도·표본·regime 표시
        ↓
텔레그램 레이더 — 사용자가 직접 판단
        ↓
실제 발송 시각 이후 path ledger
        ↓
MAE/MFE · 상승/하락 선도달 · reference net PnL 평가
```

### 4.1 1단계 — Downside veto

하방 판별력은 이미 있으므로 먼저 `p_dn5`의 수준을 진짜 expanding OOF로 보정한다. 운영
제외·강등은 하지 않고, 다음 위험 축의 SHADOW veto가 같은 날 R1보다 실제 하방을 줄이는지
paired 비교한다.

- `P(-3% first)`, `P(-5%)`, `P(-10%)`
- 예상 MAE와 하위 분위 MAE
- pump-after-dump 확률
- 과열·유동성
- BTC 급락 regime 민감도
- 신규 상장·데이터 부족으로 인한 불확실성

하방 확률이 calibration되지 않은 상태에서는 단순 cutoff를 운영에 사용하지 않는다.
crowding·dump 계열 재탕과 현재 확보되지 않은 spread/호가 데이터는 새 피처 목록에서 제외한다.

### 4.2 2단계 — Upside ranking

Downside veto를 통과한 종목만 다음 기준으로 정렬한다.

- `P(+5% before -3%)`
- `P(+10% before -3%)`
- `P(MFE >= +5/+10/+20%)`
- 예상 MFE
- 상승까지 걸리는 시간
- 동일 regime·유동성 구간의 historical lift

### 4.3 권장 출력과 정렬 연구

초기 출력은 하나의 utility나 비율로 압축하지 않고 세 축을 그대로 보존한다.

1. 보정된 하방 위험
2. 변동성 band 안에서의 상방 lift
3. 표본·calibration 불확실성

`p_up / p_down` 비율, utility식, lexicographic 조기 채택은 과거 변형의 재탕이거나 이번
실측에서 전제가 확인되지 않았다. 향후 후보로 계산할 수는 있어도 독립적인 forward 우위를
보이기 전에는 운영 정렬키로 쓰지 않는다.

---

## 5. 올바른 평가 체계

### 5.1 추천 한 건의 필수 시각

각 추천에는 다음을 기록한다.

- `decision_started_at`
- `decision_completed_at`
- `sent_at`
- `delivery_ok`
- `entry_observable_at`
- `entry_price`
- `entry_price_source`
- `feature_asof`
- `data_snapshot_id`
- `model_id`, `model_hash`, `rule_version`

성과 경로는 `entry_observable_at` 이후부터만 계산한다. 08:50 preopen과 09:05 open은 절대 같은
원장으로 섞지 않는다.

### 5.2 추천 한 건의 필수 결과

- `mae_pct`, `mfe_pct`
- `time_to_mae`, `time_to_mfe`
- `first_hit ∈ {up, down, none}`
- `up5_before_dn3`
- `up10_before_dn3`
- `hit_up5`, `hit_up10`, `hit_up20`
- `hit_dn3`, `hit_dn5`, `hit_dn10`
- reference exit별 net PnL
- 경로 완결성 상태

### 5.3 일별·모델별 집계

- 같은 날 top-K는 equal-weight 또는 명시된 고정 exposure로 집계한다.
- 추천이 없는 날은 일수익률 0으로 포함한다.
- 복리 equity curve와 초기 자본을 포함한 MDD를 사용한다.
- crypto 일별 Sharpe는 기본 `√365`를 사용한다.
- 신뢰구간은 거래 단위 IID가 아니라 날짜 단위 또는 moving/block bootstrap으로 계산한다.
- 모델 비교는 같은 날짜·같은 유니버스·같은 exit·같은 비용에서 수행한다.
- signal ranking과 exit policy 성과를 별도로 보고한다.

### 5.4 핵심 KPI

#### 1순위 — 하방

- top-K `-3% first` 비율
- top-K `-5%`, `-10%` 도달률
- 평균·중앙값·하위 분위 MAE
- 유동성 매칭 baseline 대비 하방 개선폭

#### 2순위 — 상방

- `+5% before -3%`
- `+10% before -3%`
- `+5/+10/+20%` 도달률
- 평균·중앙값·상위 분위 MFE
- 상승까지 걸린 시간

#### 3순위 — 순위 품질

- top-1/top-3/top-5 lift
- 같은 날 전체 universe 대비 percentile
- regime·유동성별 안정성
- probability reliability와 Brier score

#### 4순위 — reference trading result

- 비용 차감 평균 net PnL
- 일별 Sharpe
- 복리 MDD
- hit rate, profit factor
- 비용·슬리피지 민감도

---

## 6. 단계별 개선 로드맵

기간과 표본 수는 초기 예상이며 데이터 수집 속도와 분산에 따라 조정한다.

실행은 모라토리엄 경계에 따라 세 트랙으로 나눈다.

- **Track 1 — 즉시**: 비용·경로·유니버스·실패 전파·단일 snapshot·전 유니버스 기록·공통
  집계를 수리한다. 이는 판정 무결성 복구라 09-01 동결과 충돌하지 않는다.
- **Track 2 — 사용자 결정 반영**: 추천 품질을 최우선 거버넌스로 두되, 이미 동결된 v2
  판정에는 구지표를 유지하고 신지표를 병기한다. preopen은 전용 원장으로 측정한다.
- **Track 3 — post-09-01 GO 시**: 진짜 OOF → 상방 head → paired veto → 고정/ATR
  first-passage 순서로 진행한다. 그 전에는 연구 결과를 운영 정렬에 넣지 않는다.

### Phase A — 측정 정상화

목표: 현재 모델이 실제로 무엇을 잘하고 못하는지 믿을 수 있게 만든다.

#### A1. 실행 가능한 원장

- 09:05 이후 실제 관측 가능한 가격부터 path를 시작한다.
- 실제 Telegram 발송 결과와 시각을 저장한다.
- R1 preopen 전용 writer와 closer를 분리한다.
- 중복 실행에 안전한 `(date, slot, model_id, rank, coin)` 키를 사용한다.

#### A2. 경로 완결성

- 15분봉은 기대 timestamp·시작·종료·개수·중복을 검증한다.
- 4시간봉도 동일하게 완전한 horizon만 CLOSED 처리한다.
- 업비트의 무체결 봉 부재는 직전 close의 flat 경로로 복원한다.
- 수집 장애는 같은 시각 KRW-BTC/reference coverage로 구분한다.
- reference도 비면 `INCOMPLETE`로 보류한다. 단순 `96봉 미만` complete-only 필터는
  저유동 종목을 계통 제거하므로 사용하지 않는다.

#### A3. 성과 계산 통합

- v2 비용 이중 차감을 제거한다.
- 모든 모델을 `signal × exit_policy × exposure × cost_model` 스키마로 정규화한다.
- 일별 equal-weight, 복리, 초기 equity, `√365`, 날짜-cluster CI를 공통 함수로 만든다.
- champion 비교에서 서로 다른 hit·exit 정의를 섞지 않는다.

#### A4. 데이터 유니버스

- 업비트 live KRW ticker와 각 DB의 ticker를 매일 비교한다.
- 신규 상장을 자동 backfill하고 최소 history 부족 상태를 명시한다.
- 전체 universe coverage와 candle 연속성을 health gate에 추가한다.

#### A5. 운영 성공 의미

- Telegram credential 누락은 성공이 아니라 명시적 실패로 처리한다.
- critical 단계와 optional 연구 단계를 구분한다.
- critical 실패는 systemd exit code에 반영한다.
- heartbeat가 실제 당일 발송·원장 기록·청산·게시·백업 receipt를 확인하게 한다.

#### Phase A 종료 조건

- open과 preopen 추천이 각각 독립 원장에 귀속된다.
- 알림 전 가격 움직임이 성과에 포함되지 않는다.
- 비용이 정확히 한 번만 차감된다.
- 불완전 경로가 CLOSED로 확정되지 않는다.
- 같은 입력으로 대시보드와 evaluator가 동일한 핵심 수치를 낸다.

### Phase B — 확률과 순위 신뢰도 (09-01 이후 GO일 때)

목표: R1의 상방·하방 확률을 실제 의사결정에 사용할 수 있게 만든다.

#### B1. 진짜 OOF calibration

- 시간순 expanding OOF 예측으로 calibration 데이터를 만든다.
- RR head, expected downside map, A1 dump head, cutoff를 OOF 값으로만 적합한다.
- as-of 추론용 모델만 마지막에 전체 train으로 다시 학습한다.
- raw probability와 calibrated probability를 모두 저장한다.

#### B2. 단일 inference snapshot — Track 1 구현 완료, forward 축적 중

- 슬롯당 `score_candidates`를 한 번만 실행한다.
- Telegram, R1/R2/A1 원장과 evaluator가 같은 snapshot을 사용한다.
- snapshot에 데이터 범위·feature schema·commit·환경·모델 hash를 남긴다.

#### B3. 공정한 baseline

각 거래일 같은 유니버스에서 다음과 비교한다.

- 유동성 상위 무작위
- 같은 날 유동성-matched 무작위
- top100 몽키 baseline
- 단순 모멘텀
- 단순 저변동성
- 현재 R1
- downside-first R2
- A1 dump guard
- v2 보조 조건

#### B4. 확률 품질 판정

- 하방 확률 reliability curve
- 상방 확률 reliability curve
- Brier score
- calibration slope/intercept
- regime별 calibration drift
- top-K lift와 coverage

#### Phase B 종료 조건

- OOF와 live 계산 경로가 재현 가능하다.
- 하방 확률이 실제 빈도와 일관된 방향으로 움직인다.
- downside veto가 baseline보다 하방을 줄이면서 상방을 전부 제거하지 않는다.
- 특정 한 regime·한 달에만 결과가 집중되지 않는다.

### Phase C — 엣지 개선 (09-01 이후, 상방 head 우선)

목표: 측정이 정상화된 뒤 실제 후보 선택 능력을 높인다.

#### C1. 상방 head 재구축

1. 전 유니버스 expanding OOF로 `p_up5/p_up10/p_up20`을 다시 만든다.
2. 이미 승인된 Binance `b_vol_surge`와 현재 D-1 feature를 함께 비교한다.
3. 고정 barrier와 ATR barrier를 병행하고 변동성 band 안에서 lift를 본다.
4. 같은 날 몽키·유동성-matched baseline보다 못하면 폐기한다.
5. 상방 head가 살아난 뒤에만 하방 veto와 결합한다.

#### C2. 하방 피처 후보

- ATR과 downside semivolatility
- BTC regime 민감도
- 현재 저장 중인 D-1 feature의 결측·불확실성

crowding, dump 계열, 변동성 수축, 1분/5분 ignition, spread 없는 spread proxy는 이미
판정됐거나 데이터가 없어 이번 신규 축에서 제외한다.

새 피처는 기여 검증 후 사용할 수 있지만, 라벨이나 모델 architecture 변경은 사용자 확인을 받는다.

#### C3. 상방 피처 후보

- D-1 상대 모멘텀
- 거래대금·거래량 surge
- 승인된 Binance `b_vol_surge`
- 시장 breadth와 동조 상승
- downside semivolatility와 결합한 비대칭 변동성

#### C4. Regime 분리

- BTC bull/bear와 quiet/volatile을 단순 보고용이 아니라 calibration 안정성 측정에 사용한다.
- 표본이 부족하면 regime별 모델을 따로 만들지 않고 공통 모델 + regime feature를 우선한다.
- 특정 regime 결과가 시간 decay와 겹치는지 반드시 확인한다.

#### C5. 기존 모델 권고

| 모델 | 권고 연구 상태 | 이유 |
|---|---|---|
| R1 | 기준 baseline / 검증 지속 | 목표 구조와 가장 직접적으로 일치하지만 현재 forward 부정적 |
| R2 | downside challenger | 하방 페널티 방향은 맞지만 개선 증거 부족 |
| A1 | dump guard challenger | 목적은 적합하지만 현재 결과 부정적 |
| pump v1 | 신규 연구 중단 또는 보관 | 현재 forward 부정적 |
| pump v2 | 보조 radar / SHADOW 유지 | 상방 탐지 가능성은 있으나 저위험 후보라는 증거는 부족 |
| distribution | 확률 진단용 | exit·hit 정의를 맞춘 뒤 비교 |
| legacy detector/6-class | archive 후보 | 실제 운영 경로와 불일치 |

promotion·demotion·알림 변경은 이 문서만으로 수행하지 않는다.

### Phase D — 사용자 의사결정 지원

목표: 모델 점수보다 사용자가 실제로 판단하기 쉬운 정보를 제공한다.

알림 포맷 변경은 사용자 확인 후 다음 형태를 검토한다.

```text
#1 KRW-XXX
하락 위험: 낮음/중간/높음
  -3% 먼저: xx% | -5% 도달: xx%
상승 기회:
  +5% 먼저: xx% | +10% 먼저: xx%
예상 경로:
  MAE 중앙값 -x.x% | MFE 중앙값 +x.x%
근거:
  상대모멘텀 · 거래량 surge · dump risk 낮음
신뢰도:
  유사 표본 n=xxx | regime=... | calibration=...
```

숫자만 많아지는 것을 피하기 위해 실제 알림에는 핵심 3~5개만 보이고, 상세 정보는 dashboard에
둔다.

사용자가 `NOTES.md`에 기록한 실제 선택과 결과는 다음 질문에만 사용한다.

- 사용자가 고른 종목이 시스템 top-K 중 무엇이었는가?
- 사용자가 거른 종목은 어떤 위험 신호를 보였는가?
- 시스템 점수보다 사용자 판단이 추가한 가치가 있었는가?

사용자 실제 거래 기록을 모델 학습에 자동 반영하지 않는다.

---

## 7. 채택 판단

### ADOPT 후보

- 4대 위생을 모두 통과한다.
- 실제 알림 시점 이후 경로에서 하방이 baseline보다 줄어든다.
- 상방 선도달률이 함께 유지되거나 개선된다.
- 비용 차감 reference 결과가 실용적이다.
- 여러 시기·regime에서 방향이 유지된다.
- live paper 결과가 backtest 방향과 크게 충돌하지 않는다.

### SHADOW

- 방향은 좋지만 표본이 부족하다.
- 특정 regime에서만 좋다.
- 신뢰구간이 넓다.
- 사후 발견된 filter·exit라 새 forward 검증이 필요하다.
- 사용자의 저하방 목표에는 맞지만 상방 희생 정도가 아직 불명확하다.

### REJECT 또는 보관

- look-ahead나 유니버스 누수가 있다.
- 비용 차감 후 장점이 사라진다.
- 하방은 줄지만 상방을 거의 전부 제거한다.
- 평균은 좋아도 소수 날짜·코인에 결과가 집중된다.
- 같은 조건의 단순 baseline보다 개선되지 않는다.

다중검정, PSR/DSR, IC 등은 옆에 진단값으로 표시하되, 실용적인 forward 결과를 자동으로
폐기하는 사전 게이트로 사용하지 않는다.

---

## 8. 구현 대상 지도

검증 후 확정된 구현 대상은 다음과 같다.

| 책임 | 주요 대상 |
|---|---|
| 신규 상장·coverage | `data/collector_d1.py`, `scripts/health_check.py` |
| 단일 inference snapshot | `signals/recommend.py`, `scripts/recommend_send.py`, `scripts/recommend_today.py` |
| OOF calibration | `signals/recommend.py`, calibration 모듈, 관련 연구 스크립트 |
| preopen 전용 원장 | `signals/model_registry.py`, preopen runner, 전용 writer/closer |
| 실행 가능 entry/path | `scripts/close_recommend_ledger.py`, 15분봉 + delivery receipt |
| 공통 성과 엔진 | `ledger/metrics.py`, `scripts/build_dashboard.py`, `scripts/idea_validation_report.py` |
| champion 공정 비교 | `ops/champion_selector.py`, `ops/policy_competition.py` |
| 전달·실패 감시 | `notifier/telegram.py`, daily shell, `scripts/heartbeat.sh` |
| 재현 manifest | model artifact 저장부, registry, output metadata |
| 전 유니버스 사후 라벨·평가 | `signals/recommend_score_labels.py`, `scripts/label_recommend_snapshots.py`, `scripts/evaluate_recommend_score_labels.py` |

시그널 코드에서 Telegram을 보내거나 ledger에서 모델을 불러오는 식으로 책임을 섞지 않는다.

---

## 9. 필수 테스트

### 데이터

- 신규 KRW 종목 자동 발견·backfill
- D1/4h/15m 유니버스 coverage 차이 감지
- 기대 timestamp 누락·중복·미완성 봉 거부
- Upbit KST와 Binance session date 정렬

### 시그널

- feature가 의사결정 시점 이후 값을 사용하지 않음
- expanding OOF prediction이 각 행보다 과거 학습만 사용
- snapshot 재실행 시 동일 결과
- preopen과 open feature date가 명시적으로 다름

### 원장

- 발송 전 가격 움직임 제외
- `+5% before -3%`와 반대 순서 판정
- 같은 봉 TP·SL 동시 도달 시 보수적 처리
- 비용 정확히 한 번 차감
- 불완전 path는 CLOSED 금지
- 무체결 누락은 flat-fill, reference 동시 누락은 INCOMPLETE
- 중복 rerun 멱등성

### 성과

- 같은 날 N개 추천의 exposure 고정
- 첫날 손실도 MDD에 포함
- no-trade day 포함
- 날짜-cluster bootstrap
- signal과 exit policy 분리
- dashboard와 evaluator 핵심 수치 일치

### 운영

- Telegram credential 누락·API 실패가 nonzero로 전파
- 발송 receipt 누락 시 heartbeat 경고
- backup partial failure 감지
- 당일 원장·청산·게시 freshness 확인

---

## 10. 하지 말아야 할 것

- 현재 평가 결함을 둔 채 새 모델 복잡도부터 올리기
- calibration되지 않은 작은 `p_down`으로 risk-reward ratio를 과대평가하기
- 08:50과 09:05 추천 성과를 같은 원장으로 합치기
- 알림 전 09:00 움직임을 09:10 추천 성과로 포함하기
- 서로 다른 exit·hit 정의를 한 champion 순위에서 비교하기
- post-hoc로 좋아 보인 subgroup을 바로 운영 승격하기
- gross 결과로 모델을 홍보하기
- 사용자 확인 없이 라벨·architecture·알림 포맷·sizing·promotion을 변경하기
- 실거래 주문 코드를 추가하기

---

## 11. 최종 우선순위

1. 알림 시점 이후 executable ledger
2. R1 preopen 전용 원장
3. 비용·MDD·일별 exposure·Sharpe 계산 정상화
4. 완전한 candle path 검증
5. 신규 상장 자동 편입
6. 슬롯당 단일 inference snapshot + 전 유니버스 score/feature 축적
7. delivery receipt·heartbeat·backup 신뢰도 강화
8. 09-01 이후 진짜 시간순 OOF calibration
9. 상방 head 재구축 + 몽키·유동성-matched baseline
10. paired SHADOW 하방 veto
11. 고정/ATR first-passage 병행과 within-vol-band 판정
12. 사용자 실제 선택과 시스템 추천 비교

이 순서를 지키면 “모델이 복잡해졌다”가 아니라 **사용자가 받을 만한 종목의 품질이 실제로
좋아졌는지**를 단계마다 답할 수 있다.
