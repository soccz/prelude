# Challenger quant-evaluator v1 — 최종 독립 판정

> 상태: **FINAL**
> 감사 기준일: 2026-07-25 KST
> 사용자 목적: 자동매매가 아니라, 직접 매매 판단에 쓸 **하락 가능성은 낮고
> 상승 가능성은 높은 추천 종목**을 받는 것
> 판정 단위: 연구 성능과 실제 운영 연결을 분리해 판정

## 1. 최종 결론

**채택 0개, SHADOW 0개, REJECT 전부**다.

| 후보 | 최종 상태 | 사용자 목적 기준 결론 | 운영 연결 |
|---|---|---|---|
| downside top-third veto | **REJECT** | 09:15 재계산에서 dn5는 줄지만 상승·safe-up도 줄고 net 개선은 사실상 0 | 없음 |
| upside `cls_bvol` | **REJECT** | up10과 AUC는 높지만 dn5 68.91%, net -0.4103%/픽. fresh Binance 피처도 발송 때 없음 | 없음 |
| direct `safeup_head` | **REJECT** | safe-up +5.65pp와 함께 dn5도 +15.07pp 증가 | 없음 |
| post-hoc safe-up Pareto | **REJECT** | holdout을 본 뒤 만든 규칙이며 net -0.2790%/픽, safe-up도 10.85%로 하락 | 없음 |
| first-passage fixed head | **REJECT** | safe-up은 늘지만 dn5가 R1보다 +20.12pp 증가. net 개선 CI도 0을 통과 | 없음 |
| first-passage ATR head | **REJECT** | 변동성 재포장은 줄였지만 discovery에서 dn5 +11.52pp, net -0.2265pp 악화 | 없음 |
| semivol joint head | **REJECT** | core 대비 safe/up 개선 0, dn5 감소는 불확실하고 net·AUC는 악화. R1 대비 dn5 +17.55pp | 없음 |

따라서 **이번 연구로 사용자가 새로 받는 추천은 없다.** 모든 후보는 격리된
연구 스크립트와 artifact이며, 기존 R1 발송·원장·스케줄러에는 연결되지 않았다.
후보 중 하나를 붙이는 것은 추천 품질 개선이 아니라 검증 실패 모델의 배포가 된다.

현재 남은 가치 있는 결과는 다음 두 가지뿐이다.

1. 실제 알림 도착 뒤인 09:15부터 평가하는 공통 경로·score-label 인프라
2. “상승 확률 하나를 높이면 하방도 같이 커진다”는 실패 원인의 정량 확인

다만 forward snapshot과 label 실표본은 아직 0개이므로, 인프라 준비를 모델 검증
완료로 해석하면 안 된다.

## 2. 독립 감사 방법

연구자가 만든 summary나 판정 문자열을 그대로 사용하지 않았다.

- 저장된 row artifact에서 정책별 Top-3와 날짜별 평균을 다시 구성
- 별도 seed `20260725`, 20,000회 날짜-cluster bootstrap으로 paired CI 재계산
- `[D 09:15, D+1 09:15)` 96개 봉, KRW-BTC 기준 경로 완결성 재검사
- target 종목의 무체결 봉만 직전 종가로 flat-fill
- 같은 봉에서 상·하단을 모두 건드리면 하방 우선
- 왕복 비용 0.15%가 정확히 한 번만 차감되는지 행 단위 대조
- D 이전 완료봉 수, Top-100, outer/inner 시간 순서, embargo, holdout unlock 순서 확인
- 고정 장벽 결과가 단순 변동성 순위인지 date×volatility-band 안에서도 재검사
- semivol 4개를 D1 DB에서 독립 재구성하고 원 artifact와 전 행 대조
- 연구 후보가 실제 발송 경로에 import·호출되는지 별도 검색

독립 재계산기:

- `_workspace/recalc_challenger_quant_audit_v1.py`
- `_workspace/recalc_first_passage_quant_audit_v1.py`
- `_workspace/recalc_semivol_quant_audit_v1.py`

세 스크립트는 입력을 읽기만 하며 production 파일을 수정하지 않는다.

## 3. Downside veto — 원 수치는 운영 비교 불가, 수정판도 REJECT

### 3.1 원 연구의 세 가지 핵심 결함

1. `scripts/downside_veto_challenger_v1.py:280-283`의 경로는 09:15가 아니라
   D 09:00부터 시작한다. 알림 도착 전 가격 움직임을 포함하므로 원 보고서의
   “m15” 수치는 **운영 표준 비교에 사용할 수 없다.**
2. 입력 OOS panel은 각 날짜의 `prior >= 70`을 보장하지 않는다. locked
   18,000행 중 1,225행(6.81%), 180일 중 167일에 prior 70 미만 후보가 있었다.
   R1 Top-3에도 38/540픽, 34일, 최소 prior 2인 행이 들어갔다.
3. 구형 확률 보정은 outer-train의 in-sample prediction을 bucket에 넣는다.
   이름과 달리 true inner OOF calibration이 아니다.

또한 q=1/3은 앞선 pseudo-veto 관찰의 영향을 받았고, coverage가 인정하는 관련
선행 정책 변형만 최소 15개다. 이 180일을 프로젝트 전체의 virgin holdout으로
볼 수 없다.

### 3.2 같은 픽의 정확한 09:15 재평가

공통 178일, 정책당 534픽:

| 정책 | net/픽 | SL-first | dn5 | up10 | safe-up10 |
|---|---:|---:|---:|---:|---:|
| R1 baseline | -0.2298% | 51.50% | 36.52% | 16.29% | 14.98% |
| top-third veto | -0.2207% | 46.25% | 27.90% | 14.04% | 12.73% |
| absolute veto | -0.2148% | 51.31% | 36.52% | 16.10% | 14.79% |
| lexicographic | -0.2311% | 17.23% | 5.62% | 0.94% | 0.94% |

top-third − R1의 독립 날짜 paired 결과:

| 지표 | 차이 | CI95 |
|---|---:|---:|
| net | +0.0091pp | [-0.2632, +0.2816]pp |
| SL-first | -5.24pp | [-9.18, -1.31]pp |
| dn5 | -8.61pp | [-12.36, -4.87]pp |
| up10 | -2.25pp | [-4.68, +0.19]pp |
| safe-up10 | -2.25pp | [-4.68, +0.19]pp |

하방 감소는 재현되지만 net edge는 없고 원하는 상승 후보도 같이 제거한다.
원 09:00 경로의 net 개선 +0.0670pp가 실행 가능한 09:15에서는 +0.0091pp로
소멸한다. **REJECT**다.

## 4. Upside `cls_bvol` — AUC는 살아도 추천 정책은 실패

외부 expanding WF, 5일 embargo, true inner OOF isotonic, 09:15 경로, 비용
1회는 구현됐다. 그러나 다음 문제가 남는다.

- `cumcount()+1 >= 70`이라 실제 D 이전 완료봉은 최소 69개다. locked Top-3에
  prior 69인 픽 3개가 포함됐다.
- 선택 피처 `b_vol_surge`에 필요한 fresh Binance D-1 봉은 09:00에 닫히지만,
  현재 runner는 R1 발송 뒤에 Binance를 갱신한다. 현재 09:10 운영에서는
  point-in-time으로 사용할 수 없다.
- discovery에서 6개 변형을 비교했다. holdout도 프로젝트 전체의 virgin
  holdout이 아니다.
- random 비교는 seed 42 한 묶음이며 random 분포 전체가 아니다.

locked 공통 178일, 정책당 534픽:

| 정책 | net/픽 | SL-first | dn5 | up10 | safe-up10 |
|---|---:|---:|---:|---:|---:|
| `cls_bvol` | -0.4103% | 65.36% | 68.91% | 30.71% | 16.48% |
| `cls_core` | -0.3315% | 64.04% | 68.35% | 30.52% | 15.92% |

`cls_bvol − cls_core`:

| 지표 | 차이 | CI95 |
|---|---:|---:|
| net | -0.0788pp | [-0.2506, +0.0921]pp |
| safe-up10 | +0.56pp | [-1.12, +2.43]pp |
| dn5 | +0.56pp | [-1.50, +2.62]pp |

전체 row AUC `0.7450`은 높지만 Top-3 개선으로 연결되지 않는다. 높은 상승률의
대가로 하방 노출이 68.91%가 되어 사용자 목적과 반대다. **REJECT**다.

## 5. Direct safe-up head — safe 확률만 최대화한 구조적 실패

이 연구는 prior 70봉, PIT Top-100, D-1 feature, true inner OOF, 09:15 경로와
비용 계약이 당시 세 후보 중 가장 깨끗하다. 하지만 학습 label은 09:00 day-open
기준이고 실제 진입은 09:15다. 더 중요한 문제는 binary safe-up 확률 하나가
safe-up이 아닌 실패 중 큰 하방을 별도 처벌하지 않는다는 점이다.

pairwise complete 177일의 `safeup_head − R1_repaired`:

| 지표 | 차이 | CI95 |
|---|---:|---:|
| safe-up10 | +5.65pp | [+2.07, +9.23]pp |
| up10 | +6.78pp | [+2.82, +10.73]pp |
| dn5 | **+15.07pp** | **[+10.92, +19.40]pp** |
| SL-first | +12.43pp | [+7.16, +17.70]pp |
| net | +0.0805pp | [-0.2843, +0.4474]pp |

상승 가능성은 늘었지만 하락 가능성도 더 크게 늘었다. net 개선은 확인되지
않았다. 공통 172일에서 자체 net도 -0.0797%/픽이다.

holdout을 본 뒤 만든 `safeup_pareto_rank`는 confirmatory evidence가 될 수
없으며, net -0.2790%/픽과 safe-up 10.85%로 결과 자체도 실패했다.
둘 다 **REJECT**다.

## 6. First-passage head — 측정은 가장 정확하지만 정책은 REJECT

### 6.1 위생·재현성 계약

최종 고정본에서 다음을 독립 확인했다.

- path panel 44,900행, 449일
- discovery와 locked prediction 모두 날짜마다 정확히 Top-100
- 모든 저장 정책이 날짜당 정확히 Top-3
- `history_prior_bars` 최소 70, 위반 0
- 고정 장벽과 저장 first-passage label 불일치 0
- ATR 장벽과 저장 label 불일치 0
- 모든 저장 pick의 path complete
- 비용 1회 대조 최대 오차 0
- holdout 180일 hash 일치
- 모델 seed 42, XGBoost 단일 thread, SHA256 동률·random 정렬
- deterministic gzip을 포함한 build→cache 재실행의 핵심 artifact byte hash 일치

고정 목표는 09:15 진입 후 `+10%`가 `-5%`보다 먼저 도달하는지이며, 같은 봉은
하방 우선이다. ATR 목표는 D-1 ATR14를 사용해 `up=2×ATR`, `down=1×ATR`로
고정했다. outer 5-fold, inner 3-fold OOF isotonic, embargo 5일이 적용됐다.

따라서 이 연구의 실패는 앞선 downside 연구처럼 시간창 오류 때문이 아니라,
**정확히 측정해도 선택 정책이 사용자 목적을 만족하지 못했기 때문**이다.

### 6.2 Discovery에서 두 head 모두 탈락

공통 136일:

| 정책 | safe +10/-5 | dn5 | up10 | SL-first | net/픽 |
|---|---:|---:|---:|---:|---:|
| repaired R1 | 13.48% | 26.23% | 14.71% | 45.34% | -0.1161% |
| fixed first-passage head | 19.85% | 53.43% | 22.06% | 62.75% | -0.4308% |
| ATR first-passage head | 15.20% | 37.75% | 17.89% | 51.47% | -0.3427% |

fixed head − R1:

- safe +6.37pp, CI95 `[+1.72, +11.27]`
- dn5 **+27.21pp**, CI95 `[+21.81, +32.60]`
- net -0.3147pp, CI95 `[-0.7783, +0.1426]`
- fixed head 자체 net의 날짜 CI95 `[-0.8464%, -0.0272%]`

ATR head − R1:

- ATR 자체 label +3.43pp, CI95 `[+0.24, +6.86]`
- 고정 safe +1.72pp, CI95 `[-2.45, +5.88]`
- dn5 **+11.52pp**, CI95 `[+5.88, +17.16]`
- net -0.2265pp, CI95 `[-0.6254, +0.1677]`

두 후보 모두 discovery eligibility를 통과하지 못했다. 스크립트는 이 경우
ex-ante primary인 fixed head를 fallback으로 holdout에 평가했을 뿐, discovery
승자를 찾은 것이 아니다. ATR head는 discovery 실패 후 holdout에 적합하지
않았다.

### 6.3 Locked holdout도 low-downside 조건 실패

holdout은 180개 KRW-BTC-complete 날짜로 고정됐고, 모든 정책의 path가 완전한
공통 164일에서 비교했다.

| 정책 | safe +10/-5 | dn5 | up10 | SL-first | net/픽 |
|---|---:|---:|---:|---:|---:|
| fixed first-passage head | 20.73% | **59.55%** | 25.81% | 58.94% | -0.0154% |
| repaired R1 | 14.84% | 39.43% | 17.89% | 46.75% | -0.1504% |
| direct safeup head | 19.72% | 54.07% | 23.78% | 59.35% | -0.1017% |
| monkey seed42 | 7.52% | 23.78% | 8.13% | 46.54% | -0.3546% |
| liquidity matched | 9.76% | 36.18% | 11.59% | 50.00% | -0.4841% |

fixed head − repaired R1:

| 지표 | 차이 | 독립 CI95 |
|---|---:|---:|
| safe +10/-5 | +5.89pp | [+1.42, +10.16]pp |
| up10 | +7.93pp | [+3.25, +12.60]pp |
| dn5 | **+20.12pp** | **[+14.43, +26.02]pp** |
| SL-first | +12.20pp | [+6.91, +17.68]pp |
| net | +0.1350pp | [-0.2553, +0.5300]pp |

상승·safe 확률은 실제로 높아졌지만 하방 위험이 더 크게 증가했다. fixed head의
절대 net도 -0.0154%/픽이고 날짜 CI95는 `[-0.3910%, +0.3733%]`다. “하락
가능성이 낮으면서 상승 가능성이 높은 종목”이라는 필수 조건을 통과하지 못한다.

### 6.4 변동성 교락 분석

고정 +10/-5 label의 holdout base rate는 최저→최고 변동성 band에서
2.91%→12.87%로 4.43배 증가한다.

- fixed head score와 ATR의 Spearman 상관: `+0.629`
- fixed head Top-3 중 최고 변동성 band 비율: `69.31%`
- pooled AUC: `0.705`
- date×vol-band 안의 row-weighted AUC: `0.648`

즉 fixed head가 변동성만 복사한 것은 아니고 band 안에서도 일부 판별력은 있다.
그러나 같은 날짜·같은 변동성 band 기대치와 비교해도:

| 지표 | fixed head 실제 | 같은 날·같은 vol band 기대 |
|---|---:|---:|
| safe | 20.73% | 11.00% |
| dn5 | **59.55%** | 33.14% |
| net/픽 | -0.0154% | -0.3491% |

safe lift와 함께 큰 하방 초과도 동시에 선택한다.

ATR label은 변동성 band의 단조 증가를 제거했고 ATR head Top-3도 band별
약 19~21%로 균형적이었다. own-label date×band AUC도 `0.716`으로 살아 있다.
그러나 최고 band의 ATR-label base rate가 오히려 최저 band의 0.65배로
역전됐고, 실제 고정 사용자 목표에서는 dn5 증가와 net 악화를 해결하지 못했다.
과학적으로 흥미로운 label인 것과 추천 정책으로 쓸 수 있는 것은 별개다.

### 6.5 Lowest-ATR Top-3 진단 — 저하방만 얻고 상방을 잃는 퇴행

부모 감사 요청에 따라 locked Top-100에서 D-1 ATR14가 가장 낮은 3개를 고르는
단순 진단을 추가했다. 이는 새 후보·채택 trial이 아니며, 저변동 선택이 사용자
목적의 대안이 되는지만 확인한다. 시장명 오름차순으로 동률을 고정했다.

기존 6정책과 lowest-ATR의 Top-3가 모두 path-complete인 공통 164일,
정책당 492픽:

| 정책 | safe +10/-5 | up10 | dn5 | SL-first | net/픽 |
|---|---:|---:|---:|---:|---:|
| fixed first-passage head | 20.73% | 25.81% | 59.55% | 58.94% | -0.0154% |
| repaired R1 | 14.84% | 17.89% | 39.43% | 46.75% | -0.1504% |
| direct safeup head | 19.72% | 23.78% | 54.07% | 59.35% | -0.1017% |
| highest-ATR Top-3 | 19.51% | 22.15% | 40.85% | 54.47% | -0.1599% |
| liquidity matched | 9.76% | 11.59% | 36.18% | 50.00% | -0.4841% |
| monkey seed42 | 7.52% | 8.13% | 23.78% | 46.54% | -0.3546% |
| **lowest-ATR Top-3 진단** | **0.41%** | **0.61%** | **4.88%** | **16.26%** | **-0.3604%** |

lowest-ATR − R1의 날짜 paired 결과:

- dn5 -34.55pp, CI95 `[-39.43, -29.67]`
- SL-first -30.49pp, CI95 `[-35.98, -25.20]`
- safe **-14.43pp**, CI95 `[-17.68, -11.18]`
- up10 **-17.28pp**, CI95 `[-20.73, -14.02]`
- net -0.2100pp, CI95 `[-0.5425, +0.1239]`
- lowest-ATR 자체 net CI95 `[-0.5666%, -0.1608%]`

저변동만 고르면 하방은 크게 줄지만 +10% 상승 후보가 사실상 사라지고 net은
유의하게 음수다. 즉 반대 극단도 해답이 아니며, **저하방만 얻고 상방을 잃는
명확한 퇴행**이다. 필요한 것은 고변동/저변동 극단 선택이 아니라 같은 위험
구간 안에서 상승과 하방을 함께 분리하는 조건부 정책이다.

### 6.6 Holdout·trial 해석

- discovery에서 명시적으로 비교한 head는 fixed와 ATR 두 개다.
- neither eligible 상태에서 fixed primary fallback만 holdout에 적합했다.
- 2026-01-24~2026-07-24 구간은 관련 연구가 이미 본 기간과 겹친다.
- coverage도 `virgin_or_clean_preregistered=false`를 명시한다.
- monkey와 liquidity-matched baseline은 각각 deterministic 한 번의 추출이지,
  다수 random draw 분포가 아니다.

따라서 통과했더라도 최대 SHADOW였고, 실제로는 downside와 net gate가 모두
실패했다. 최종 상태는 **REJECT**다.

## 7. Downside semivol joint head — 신규 축도 분리력 미확인

### 7.1 고정 비교와 독립 위생 감사

실행된 모델 비교는 하나다.

- core: 기존 first-passage 24개 D-1 피처
- augmented: core + `downside_semivol_7`,
  `downside_semivol_21`, `upside_semivol_21`,
  `semivol_asym_21`

일별 close return의 음수·양수 부분 제곱 평균을 각각 7일/21일 rolling한 뒤
정확히 한 행 shift했다. 타깃·모델·outer 5-fold·inner 3-fold OOF isotonic·
embargo 5일·09:15 path·비용은 first-passage core와 동일하다.

독립 재계산 결과:

- 저장 prediction 32,200행 전체에서 D1 DB로 4개 피처를 재구성한 최대 오차 0
- source date 불일치 0, source lag는 전 행 정확히 1일
- prior 70 미만 0, 날짜별 prediction 정확히 100개
- 정책/날짜별 정확히 Top-3, 저장 pick의 incomplete path 0
- fixed first-passage label mismatch 0, 비용 1회 최대 오차 0
- outer/inner 시간순서 위반 0, inner fold 수 위반 0, holdout embargo 위반 0
- 기존 core reference 32,200행의 raw·probability 최대 차이 0
- manifest에 기록된 7개 artifact hash 전부 일치

따라서 core 재현 실패나 PIT 누수 때문에 결과가 달라진 것은 아니다.

다만 “실행 trial 1개”는 맞아도 새 축 하나를 단일 피처로 검증한 것은 아니다.
7일·21일 downside, 21일 upside, 그 차이까지 **4개 묶음**으로 한 번에 넣었다.
코드상 다른 window·공식 sweep 흔적은 없지만, 어느 피처가 기여했는지는 이
설계로 식별할 수 없다. 더구나 이 묶음은 관련 holdout을 이미 본 뒤 설계됐다.

### 7.2 Locked 169일 결과

5개 비교 정책의 Top-3가 모두 path-complete인 169일, 정책당 507픽:

| 정책 | safe +10/-5 | up10 | dn5 | SL-first | net/픽 |
|---|---:|---:|---:|---:|---:|
| semivol joint | 21.30% | 26.43% | 57.00% | 58.19% | -0.0126% |
| core FP | 21.30% | 26.43% | 59.17% | 58.58% | +0.0181% |
| repaired R1 | 14.99% | 18.15% | 39.45% | 46.55% | -0.1318% |
| lowest-ATR | 0.39% | 0.39% | 4.54% | 15.78% | -0.3516% |
| liquidity matched | 9.86% | 11.44% | 34.71% | 50.30% | -0.4176% |

semivol joint − core FP, 독립 20,000회 날짜 paired:

| 지표 | 차이 | 독립 CI95 |
|---|---:|---:|
| safe | 0.00pp | [-2.76, +2.76]pp |
| up10 | 0.00pp | [-2.76, +2.76]pp |
| dn5 | -2.17pp | [-5.52, +0.99]pp |
| SL-first | -0.39pp | [-3.55, +2.56]pp |
| net | -0.0307pp | [-0.2697, +0.2081]pp |

full-universe AUC도 core `0.7052`에서 semivol `0.6981`로 낮아졌다.
semivol 자체 net -0.0126%/픽의 날짜 CI95는
`[-0.3713%, +0.3524%]`다.

169일 평균 Top-3 교집합은 2.08/3개이고 완전히 동일한 날짜는 23.67%뿐이라
실제로 종목은 바꿨다. 그럼에도 safe/up 결과는 소수점 전 범위에서 동일했고,
하방 감소는 불확실하며 net은 소폭 악화됐다.

semivol joint − repaired R1:

- safe +6.31pp, CI95 `[+1.78, +10.85]`
- up10 +8.28pp, CI95 `[+3.35, +13.21]`
- dn5 **+17.55pp**, CI95 **`[+12.03, +23.27]`**
- SL-first +11.64pp, CI95 `[+6.11, +17.16]`
- net +0.1192pp, CI95 `[-0.2758, +0.5088]`

safe는 늘지만 사용자가 피하려는 하방도 명확히 증가한다. safe, downside
non-worse, net non-adverse 중 두 gate가 실패하므로 **REJECT**다.

### 7.3 증거 경계와 운영 상태

2026-01-24~2026-07-24의 180일은 first-passage와 관련 후보들이 이미 본
구간이다. 따라서 통과했어도 최대 `FORWARD_SHADOW_CANDIDATE`였고 과거 결과로
승격할 수 없었다. 실제로는 gate를 실패했으므로 forward candidate로도 남기지
않는다.

production 경로에서 semivol 연구를 import·호출하는 곳은 없으며 SHADOW,
텔레그램, 원장 변경도 없다.

## 8. 공통 forward 인프라와 실제 운영 상태

`ledger/path_quality.py`, `signals/recommend_score_labels.py`,
`scripts/evaluate_recommend_score_labels.py`의 계약과 테스트는 통과했다.

- delivery `sent_at`을 다음 15분 grid로 올림: 09:10 → 09:15
- 96봉과 다음 boundary까지 KRW-BTC로 수집 완결성 확인
- benchmark gap과 종목 무체결 gap 분리
- 종목 무체결 gap만 flat-fill
- 같은 봉 TP/SL은 SL-first
- 비용 0.15% 1회
- observed forward와 scheduled replay provenance 분리
- payload hash, partial 재시도, complete idempotency

그러나 2026-07-25 현재:

- `output/recommend_snapshots/`: 디렉터리/실 artifact 0
- `output/recommend_score_labels/`: 디렉터리/실 artifact 0
- 평가 JSON: `found=0`, `complete_used=0`, channel 0

또한 production 경로 검색에서 감사한 challenger 이름은 각 연구 스크립트 안에서만
발견됐다. deploy, notifier, ops, 실제 recommend 경로에서 호출되지 않는다.

상태를 명확히 나누면:

| 구분 | 개수 |
|---|---:|
| production 채택 후보 | 0 |
| SHADOW 연결 후보 | 0 |
| REJECT 후보/정책 | 전부 |
| 실제 forward snapshot | 0 |
| 실제 forward label artifact | 0 |

## 9. 사용자에게 권하는 다음 진행

1. **현재 challenger는 하나도 발송에 붙이지 않는다.**
2. full-universe score snapshot과 delivery receipt를 매일 실제로 쌓는다.
3. 모델 품질은 all-score cohort와 사용자가 실제 받은 `delivered` cohort를
   분리해 본다.
4. 다음 후보는 단일 safe/up score가 아니라, 먼저 calibrated downside
   non-worse를 강제하고 통과한 종목 안에서 상승을 정렬하는 두 축 구조로 제한한다.
5. 새 historical 변형을 계속 늘리기보다 fresh forward 날짜의 within-day paired
   결과를 기다린다.
6. 향후 SHADOW 조건은 최소한 다음 세 조건을 동시에 요구한다.
   - safe 개선 CI95 하단 > 0
   - dn5 차이 CI95 상단 <= 0
   - 비용 차감 net 차이 CI95 하단 >= 0

현재 데이터가 주는 가장 솔직한 답은 “좋아 보이는 상승 head를 붙이는 것”이
아니라, **실제 발송시각 기준 전 유니버스 기록을 먼저 축적하고 fresh forward에서
하방 비열등을 증명할 때까지 기존 추천을 건드리지 않는 것**이다.

## 10. 검증 명령과 결과

```bash
python _workspace/recalc_challenger_quant_audit_v1.py
python _workspace/recalc_first_passage_quant_audit_v1.py
python _workspace/recalc_semivol_quant_audit_v1.py
python -m pytest -q \
  tests/test_path_quality.py \
  tests/test_close_paper_path_quality.py \
  tests/test_recommend_score_labels.py \
  tests/test_evaluate_recommend_score_labels.py \
  tests/test_recommend_snapshot.py
PYTHONDONTWRITEBYTECODE=1 python -m py_compile \
  scripts/first_passage_head_challenger_v1.py \
  scripts/semivol_joint_challenger_v1.py \
  _workspace/recalc_first_passage_quant_audit_v1.py \
  _workspace/recalc_semivol_quant_audit_v1.py \
  _workspace/recalc_challenger_quant_audit_v1.py
```

결과:

- 세 독립 row-level 재계산 완료
- 표적 테스트 **28 passed**
- 다섯 감사/연구 스크립트 compile 통과
- first-passage label mismatch 0
- 비용 1회 대조 최대 오차 0
- Top-100/Top-3/prior-history/holdout hash 계약 통과
- semivol 4개 D1 재구성 오차 0, source-date·시간순서 위반 0
- semivol manifest의 7개 artifact hash 전부 일치

## 11. 감사 대상 SHA256

| 파일 | SHA256 |
|---|---|
| `scripts/downside_veto_challenger_v1.py` | `1587d4d68a49b776f7508495a68df23801563b33a2516d467b46608d36eece87` |
| `output/downside_veto_challenger_v1_picks.csv` | `83e959749fac0d428ded5719a62f80024276647f7a40801e371267c76f396407` |
| `scripts/upside_head_challenger_v1.py` | `e5852b96dc806c595dd710f6c165ede88bc8f48f8ee6afdad554c3f344a470b9` |
| `output/upside_head_challenger_v1_predictions.csv.gz` | `1312459b4133cd9d4e41dd1efb3bc5dc04a687c44eee2fcd712f3a1334020ef4` |
| `scripts/safeup_head_challenger_v1.py` | `8ab8d1ca39de5a64d55de58d48b931addba39bc886c62360d79fadf595c0152a` |
| `output/safeup_head_challenger_v1_picks.csv.gz` | `ed9c61e1c9c17767a0eb2ecc0372b87e7488d9c2bc201ea6650cd9fc621317f3` |
| `scripts/first_passage_head_challenger_v1.py` | `d441cbc4c147b8ad956e511f73b3740f839fa4a7cb25b65ad117e559131d06f9` |
| `output/first_passage_head_challenger_v1_predictions.csv.gz` | `e8f9343bc18b60ee101d86cf8e95422426226b4ac5c4971a4fb39328563cda59` |
| `output/first_passage_head_challenger_v1_picks.csv.gz` | `6a218ca2d2bc60645bb6c00931ded6998daac2353e91117cd1d5b58fc1ebb54a` |
| `output/first_passage_head_challenger_v1_coverage.json` | `9d5c6672602cbf815a3b3011613164e0d3a8e6c00225892a7aff26000f8a01c1` |
| `scripts/semivol_joint_challenger_v1.py` | `dee56e4511ae7a00ad7df5a7ed4909eda6cf2bcf9f3aedc6dd06c5eb9e881643` |
| `output/semivol_joint_challenger_v1_manifest.json` | `72e9c900717a18da5f2bb283c63f22fdcd1cd4f509f2427304c30458491bd872` |
| `output/semivol_joint_challenger_v1_predictions.csv.gz` | `7e03de31dd8b7390e68e1b71ef447641709de2bd04f41a14989cd60965f38009` |
| `output/semivol_joint_challenger_v1_picks.csv.gz` | `84150b1da5169e936cd5c79a4624e84fd225bff9fa570bf319706e7ff492b7d0` |
| `output/semivol_joint_challenger_v1_coverage.json` | `edf4580f63075b0ab399647b3c0ed5c44a381df122126299dc93cd643ec35315` |
| `_workspace/recalc_challenger_quant_audit_v1.py` | `5fbecf270b189598976879a865c904e427fa8514e392d4de6137dbe325c27b2c` |
| `_workspace/recalc_first_passage_quant_audit_v1.py` | `117d210b10c7793210f5f93a1654954aa7bc7c6217bbbdde1bc3a3fa377d473a` |
| `_workspace/recalc_semivol_quant_audit_v1.py` | `155c108098c778a6e1e9f3af1e756c0bda952256cec23473dc07d5292542dda0` |
