# SIGNAL.md — 시그널 생성 (라벨 / 피처 / 모델 / 추론)

> **현재 메인 = R1 risk-reward recommender (`signals/recommend.py`).**
> detector_v1과 6-class 분포 모델은 legacy로 보존한다.
> 책임: 라벨 / 피처 / 모델 / 추론. 사이징·청산·알림은 LEDGER / OPS.

---

## 0. 한 줄 결론 (현재 운영, R1)

```
입력  : D-1까지의 KRW 일봉 피처 + PIT 거래대금 Top100
점수  : 8-feature rank mean + 독립 up/down head
정렬  : p_up10 / max(p_dn5, eps) 내림차순 → top 3
출력  : 슬롯별 단일 immutable snapshot
평가  : delivery receipt 이후 다음 15분봉부터 새 96봉
운영  : KST 08:50 preopen / 09:05 open R1
```

**운영 안전장치**:
- 현재 확률과 RR는 정렬용 score이며 strict calibrated probability로 해석하지 않음
- snapshot→delivery receipt→전용 ledger identity가 일치하지 않으면 fail closed
- 활성 R1 정렬·라벨·모델·표시는 사용자 승인 없이 변경하지 않음
- 08:50 preopen은 D-2 일봉 피처라 전날09:05 목록과 대부분 같다(§1.3, 27차)

---

## 0.0 실사용 검증만 남기는 준비 (26차, 2026-09-30)

25차의 L1 선택식은 그대로다. `signals/recommend_book_readiness.py`는 모델이 아니라 별도
진입 전 기록에 대한 **검토 기준**이다. 수정 구간 시작10-02, 최초30완결일 이후도 관찰을 계속한다.
같은 과거20일로 조건을 다시 최적화하거나 새로운 선택 후보를 추가하지 않았다.
최초10-01은 원 수집 완료보다 기록기의 대기 종료가 빨라 누락됐다. 실패와 최초 동결본을
보존한 뒤 준비 대기만 늘린 config v2로 구분했다. 선택·성과 기준 변경이나 소급 기록은 없다.

- 자료가 없으면 `waiting`, 표본/일관성이 부족하면 `continue_observing`.
- 초기 검토30일·실제 교체10일에 도달해도 하방 감소·상방/안전상승 유지·net 개선과 후보 net
  양수가 함께 충족되지 않으면 `do_not_adopt`(현재 관측 표본의 비채택, 영구 KILL 아님).
- 매칭 무작위 대비net·전후 구간·하루 제거·국면별 방향·0/15/30분 진입 민감도까지 양호하면
  `review_candidate`. **사용자 승인·forward/WF 검토를 대체하지 않으며 자동 승격은 없다.**
- 국면당5일 이상2국면, 관측일80%, 추가 후보 비용15bp 등은 검토용 초기값이다. 표본 수의
  충분성/검정력/실제 슬리피지를 보장하지 않는다. 변경 시에는 새 버전/구간으로 구분한다.
- 날짜 동일가중·관측 날짜 block3 CI와 가상 basket을 같이 보고한다. CI가0을 포함한다는 이유
  하나만으로 폐기하지 않는다. 같은 셀 R1과의 paired 차이·기존시즌선택 실험은 따로 유지한다.

`ops.recommend_book_execution`은 기존 진입 지연 도구의 경로/라벨/0분 parity 코드를 재사용한다.
두 방식의 모든 원선택이 모든 지연에서 완결된 같은 날짜만 비교하며, 각 진입부터 새24h다.
원0.15% 비용은 유지하고 **후보에만 추가15bp** 비용이 생긴 경우에도 net 차이·후보net이
양수인지 별도 민감도로 본다. 실제 거래·최적 진입법·5분 정확도·alert TTL 검증이 아니다.

진입 전 저장·정본 결과와 결합·지연 계산·캐시·판정·공개-safe 화면을 합성 데이터로 검증했고,
과거09-29의 실제100후보/SQLite 경로도 읽기 전용 연결 검산했다. 그 과거 하루는새 forward
표본으로 넣지 않는다. 현재 실추천 R1의 순위·상방/하방 head·라벨·학습은 변경하지 않았다.

## 0.1 고정 상방 후보의 새 날짜 검증 (offline, 2026-09-30)

`scripts/extend_recommend_upper.py`는09-07 A/B/C 비교를 수정하지 않고09-08~29로 연장한다.
A는 실제 R1, B는 과거 저장 확률의 수준 보정, C는 기존24피처/고정 XGB/inner expanding OOF 보정이다.
설계·입력·구 보고서·원 소스·실행기의 해시를 학습 전후 확인한다. 수동 오프라인 경로이며
모델 저장·자동 학습·배포·운영 순위 변경은 없다. 과거와 새 구간 성과를 섞지 않는다.

입력58일/11,295 labeled 행, 새 구간44 date/slot 모두 예측. 원래 공통 매칭 계약에 따라
open21일·preopen18일을 평가했다(누락5개 상세는 PHASES17차). open 결과는:

| 안 | +10% 경험 | −5% 경험 | 24h 말 net 평균 |
|---|---:|---:|---:|
| A 기존 R1 |20.63%|33.33%|+0.912%|
| B 수준 보정 |19.05%|25.40%|+1.088%|
| C 상방 재학습 |3.17%|11.11%|−0.487%|

C는 하방을 줄였지만 상승 기회도 잃었다. 전체 후보 상방 AUC는 A0.690/C0.688로 비슷한데,
C의 최종 선택은 원순위 평균45.1위·낮은 하방 score·낮은 ATR 쪽으로 이동했다. floor 발동0건이다.
이는 **상방 모델 판별력만 개선/유지해도 최종 RR Top3가 좋아지는 것은 아님**을 보여 준다.
인과적으로 특정 원인을 분리한 실험은 아니며 C의 영구 무효 판정도 아니다.

B의 EOD 차이+0.176%p는 block3 CI[−0.237,+0.694]%p이고09-27을 빼면−0.035%p다.
따라서 양 후보 모두 **운영 교체 미채택**이다. 작은 보정의 지속성·상승 손실과 최종 선택 결합을
후속 원인 분리 대상으로 삼았다. 아래18/19차는 설정을 고정한 별도 선별 가설이며,
같은22일에서 임계값을 반복 튜닝하거나 현재 시즌 시험 후보에 추가하지 않는다.
새 자동 학습/사전 기록 연결은 별도 승인 범위다. 이미 집계 결과를 본 historical extension이며
깨끗한 holdout/새 forward 성공/실계좌 PnL이 아니다. 상세 수치·민감도·검산은 PHASES17차 참조.

### 상방 모델과 최종 선택의 결합 검증 (18/19차, 같은 날 후속)

방향 제시에서 멈추지 않고 **새 선별 가설 두 개를 구현·실행·독립 재검산**했다.
두 안 모두 저장된 A/C 점수만 사용하며 모델 재학습, 실사용 순위 변경, 자동 시험 연결은 없다.

- **D 척도 대응:** `signals/recommend_score_alignment.py`는 C의 상승 순서에
  당일 A 점수 수준을 대응하고 기존 RR로 고른다. 동률은 대응 A 점수 평균, 자기 대응은 항등이다.
  변환값은 확률이 아닌 정렬 점수다.09-08~29 공통 open21일에서 기존 R1 대비 상승 경험이
  20.63%→7.94%, 하방33.33%→22.22%, 가상 EOD net+0.912%→−0.730%였다.
  원 C보다 상승을 일부 되찾았지만 실제 목적을 달성하지 못해 운영 교체 미채택이다.
- **E 원 점수 예산 조합:** `signals/recommend_budget_selection.py`는 전체100개 중
  A 상승 score 합은 원Top3 이상, 하방 score 합은 원Top3 이하인3개 조합에서 C 상승 합을
  최대화한다.83 snapshot의13,421,100개 조합을 완전 탐색·별도 재검산했다.
  E 공통 open22일에서는 기존 R1 대비 상승22.73%→21.21%, 하방31.82%→37.88%,
  가상 EOD net+1.320%→+0.726%였다. **점수 제약 준수는 실제 저위험 보장이 아니다.**
  전 기간 최종 선택이 원순위9위 이내여서 Top10 밖 기회도 활용하지 못했다. 운영 교체 미채택이다.

E와 D는 매칭 셀의 지원 여부로 공통 날짜가 다르다. A 기준값을 섞거나 유리한 slot만 고르지 않는다.
두 이전/이후 구간을 따로 보고 paired 날짜 CI·변동성/유동성 매칭·날짜 제거 민감도를 함께 남겼다.
가상 픽 net에는 왕복0.15% 비용이 한 번 반영됐다. 개발 자료 재사용이지 미관측 검증은 아니다.

추가 진단에서는 기존 체결Top10 200행 중 단일체결이0개라 단일체결 필터를 처방하지 않았다.
전체100개에는 사후 safe-up 후보가 있어도 Top10에0개인 날이 open22일 중6일이었다.
이는 좁은 후보군의 한계를 보여 주지만 미래를 알아야 계산하는 상한은 모델 성과가 아니다.
다음 가설은 이 실패 근거를 존중해야 하며, 같은 점수 재배열/예산 완화의 반복을 개선으로 포장하지 않는다.
현재는 기존 R1과 승인된 시즌별 사전 기록 시험을 유지한다. 재현 CLI·고정 JSON·정확한 분모와
검증 범위는 **PHASES18/19차**에 기록했다.

### 저장된 호가의 추가 정보 점검 (21차, offline)

`signals/recommend_spread_diagnostics.py`·`scripts/review_recommend_spread.py`는 추천 직전
스프레드 한 축을 **전체100후보의 거래대금/ATR 사분위 안**에서 비교한다. 결과를 읽기 전에
좁은 쪽/넓은 쪽의 겹치지 않는 쌍을 고정하며, cutoff 학습이나 새로운 Top3 선정은 없다.

09-10~29 open20일/2,000행/927쌍에서 좁은 쪽은 하방 경험이1.791%p 적었지만
safe-up도1.117%p 적었다. EOD net 차이+0.037%p의 날짜 block CI는[−0.863,+0.564]%p,
앞/뒤10일 차이+0.203/−0.129%p로 방향이 바뀌었다. **추가 선별 근거 부족으로 운영 필터 미채택**이다.
R1 Top3 대비 성능이 아니며 기존 비용0.15% 가정의 현실성이나 실거래 체결비용 검증도 아니다.
호가·체결 원본/소스/영수증/라벨의210파일 해시를 묶고, 신규 JSON만 출력한다.
이미 본20일의 관찰 진단이지 독립 holdout/자동 학습/새 전향 시험이 아니다. 상세는 PHASES21차.

### L1 대기 잔량 정보 점검 (23차, offline)

`signals/recommend_book_pressure.py`·`scripts/review_recommend_book_pressure.py`는 기존
체결대금 불균형과 별개인 **최우선 매수/매도 호가대금의 불균형**을 조사한다. depth1 raw에서
event/receive 시각 모두 추천 결정 시작 전인 마지막 수신 상태를 복원한다.19자리 ns는 정수로
유지하며 같은 거래소 timestamp의 후속 잔량 변경도 보존한다. 전체 잔량 total_*는 쓰지 않는다.
잔량은 취소 가능한 대기 주문이지 체결 매수·확정 수요·전체 호가 깊이가 아니다.

기존21차의 동결 input design을 읽어 별도 design에 소스/원본 해시를 결합하고, 기존 자료를
덮어쓰지 않는 수동 CLI로만 실행한다. 전체100후보의 ATR/거래대금 사분위 안에서 높은 쪽−낮은 쪽을
비교하며 결과를 보고 방향·cutoff·창 길이를 고르지 않는다. 기존 추천/피처/자동 수집/시험 경로는
변하지 않는다.20일927쌍의 높은 쪽−낮은 쪽은 up10+1.527%p, dn5−1.857%p,
EOD net+0.180%p였으나 앞/뒤10일 net−0.320/+0.681%p와 CI[−0.485,+0.984]%p로
**운영 미채택·연구 단서 유지**다. R1 Top3 우위로 읽지 않는다. 설계·실측·검산 정본은 PHASES23차다.

```bash
venv/bin/python -B scripts/review_recommend_book_pressure.py \
  --design _workspace/recommend_book_pressure_design_20260930_v2.json
```

### L1 단서의 Top3 추가 기여 (24차, offline)

`signals/recommend_book_top3.py`·`scripts/compare_recommend_book_top3.py`는23차 관찰 단서를
**실제3종목 선택**으로 검증하는 별도 수동 경로다. 원100후보의 거래대금·ATR 사분위16칸에서
원 R1 Top3가 고른 칸별 개수를 지키며, 같은 칸의 높은 L1 pressure 순으로 비중복 선택한다.
동률은 원 Top3 유지→원순위 순이다. 전체100개에서 선택하므로 Top10 밖도 가능하다.
라벨을 보기 전에 선택하며 결측 종목을 지운 뒤 다시 고르지 않는다. 기존 순위/모델과 연결하지 않는다.

고정20일·60픽에서49픽 교체,40픽이 원Top10 밖이었다. R1/L1의 dn5는28.333/21.667%,
up10은23.333/21.667%, safe-up은21.667/21.667%,24h 말 net는+1.586/+1.959%다.
같은 칸·개수 무작위 선택의 정확한 기대 평균(net+1.088%)보다 높지만,
뒤10일에는 R1 대비 net−0.197%p·up10−6.667%p다. 전체 net 차이 CI[−1.916,+2.315]%p와
하루 제거 시 부호 반전도 남았다. **후속 검증 후보 유지, 운영 교체 보류**이며 상방 개선 입증이 아니다.

입력/기존 endpoint/소스217파일을 고정하고 원 R1 선택·일별 지표의 native 일치도 확인한다.
원본 추출을 다시 하지 않고23차 검증된 endpoint를 사용한다. 신규 JSON만 출력하며 기존 보고서는
덮어쓰지 않는다. 같은 개발 자료에서 정한 규칙이지 독립 holdout/새 forward/WF 통과가 아니다.
가상 바스켓은 연속 비중첩24h가 검증된 경우만 계산하고, 실계좌/장중 낙폭과 구분한다. 상세는 PHASES24차다.

```bash
venv/bin/python -B scripts/compare_recommend_book_top3.py \
  --design _workspace/recommend_book_top3_design_20260930_v2.json
```

### 고정 L1의 새 날짜 검증 (25차, post-label replay)

`signals/recommend_book_validation.py`는24차 선택식을 그대로 사용해10-01~30 open을
개발20일과 분리한다.30일은 첫 관찰 창이지 채택에 충분하다고 정한 표본 수가 아니다.
규칙/소스/현재 R1 버전은 시작 전에 write-once design에 결합한다. 발송 성공·라벨 완료의
메타데이터만 선확인하고, 성과값 없이 원100개에서 선택한 뒤 canonical 라벨값을 조합한다.
기존 성공 발송 영수증에 따른 진입 시각과 완결24h·비용0.15% 라벨을 사용한다.
원 R1/새 선택/같은 칸 무작위 기대값/원100평균과 같은 날짜 차이6지표를 계산한다.

**고정 규칙의 새 날짜 사후 재생**이다. 새 L1 선택 자체를 진입 전에 저장하는 시스템은 아니므로
prospective pick 기록은0이며 국면 사전 기록과 합치지 않는다. 실제 체결·독립 WF 통과·실알림
전환이 아니다. 버전/원본 변경은 차단하고, 결측 종목을 버린 재선택 없이 날짜 제외로 남긴다.
관측 날짜 동일가중·block3 CI를 사용하고5일 미만 CI는 null이다. 날짜 공백은0%로 메우지 않으며
공백/중첩이 있으면 가상 바스켓 수치를 내지 않는다.0일에는 성과도 null이다.
동결 파일을 덮어쓰거나 종료 뒤 날짜를 늘려 유리한 구간을 찾지 않는다. 후속 설계는 별도 판단이다.
09-30 실제 연결 검사는 과거1일/100종목/36지표 일치였지만 새 구간 표본으로 세지 않았다.
운영·장애 경계는 OPS, 검증 영수증은 PHASES25차다.

## 0.2 시장 상태별 실패 진단·전환 재생 (offline, 2026-09-30)

`signals/recommend_regime_replay.py`는 R1을 학습/교체하는 모듈이 아니라,
같은 날 **R1 고정 / 최근 완료 성과 기반 선택 / 같은 상태의 완료 성과 기반 선택**을
비교하는 순수 계산 모듈이다. 상세 초기 기준과 변경 경계는 PHASES 15차에 기록했다.

- 상태는 frozen universe의 상승종목 비율과 거래대금7일/30일 비율로4칸을 정한다.
  입력 누락 시 상태를 미상으로 두며, 미래 DB 재계산·사후 분위수 fitting은 하지 않는다.
- 동일 모델/slot/score-source/규칙/score-schema에서 가상 결정 시각 이전에 완결·기록된
  최근10일(관측일 기준), 최소5일만 사용한다. 같은 상태 정책은 상태까지 일치해야 한다.
- Top10의 과거 net EOD가 높으면서 dn5·up10·MAE가 모두 악화되지 않았을 때만 가상 선택한다.
  그렇지 않으면 R1이다. 현재/미래 결과를 바꿔도 과거 선택은 변하지 않는 회귀 테스트를 둔다.
- Top10 기록은 R1 알림보다 늦을 수 있다. 가상 결정은 양쪽 점수 저장 완료 후이며,
  공통 진입봉보다 빨라야 한다. **실제 알림 시각의 전환 실증이나 미리 저장된 정책 기록은 아니다.**
- 기존 post-label review에 자동 재생 결과를 추가했다.09-10~29의20일 재생에서는 두 방식 모두
  Top10 선택0회. 최근 정책은 준비 부족6일/조건 미달14일, 상태 정책은15일/5일이었다.
  효과 표본0이므로 수익 차이0을 “안전성/동등성 입증”으로 해석하지 않는다.

`scripts/review_recommend_regimes.py`는 더 넓은 R1 자료의 실패 조건을 조사하는 수동 CLI다.
canonical 성공 receipt와24h label을 재검증하고, 원 Top3와 당시 전체 유니버스의 결과를
slot·소스버전·상태별로 나누어 같은 날짜 가중치로 비교한다. 유니버스 평균은 유동성 매칭 baseline이
아니며 상관관계가 실패 원인/필터 효능을 입증하지 않는다. 무라벨/무발송/3픽 미만을0수익으로 채우지 않는다.
기본 실행은 읽기 전용, `--output`은 프로젝트 내 **새 JSON에만** 저장하며 덮어쓰지 않는다.

```bash
PRELUDE_FORBID_TELEGRAM=1 PYTHONDONTWRITEBYTECODE=1 venv/bin/python -B scripts/review_recommend_regimes.py --start-date 2026-07-27 --end-date 2026-09-29 --slot both
```

이전 결과를 이미 본 상태에서 추가한2개 선택 가설이다. 현재 결론은 자동 전환/새 모델 채택이 아니라
**기존 R1 유지 + 상태별 대안 우위의 추가 검증**이다. 다른 상태에서 이기는 대안이 실제로 관찰되고,
결정 전 정책 저장을 포함한 별도 forward 검증까지 확보하기 전에는 운영 전환을 주장하지 않는다.

**09-30 후속 준비(16차):** 위3개 정책을10-01 open부터 실제 결과 전에 저장하는 경로를
`ops/recommend_regime_forward.py`에 추가했다. 실제 계산·내구 저장 시각이 검증된 발송 후
다음15분봉보다 빨라야 평가에 포함된다. 선택은 아침 native 평가의 과거 이력만 사용하며,
다음날에는 저장된 선택을 그대로 채점한다. 상세 clock/부분 저장/누락 감시는 OPS와 PHASES 16차 참조.
과거 재생(`adaptive_replay`)과 사전 기록(`forward_evaluation`)은 별도 결과다.
이는 **시즌별 선택 가설의 자동 기록 시험**이며, 실사용 R1 전환·추가 학습·성능 우위 입증이 아니다.

## 0.3 체결 정보 추가 검증 (record-only, 2026-09-07)

기존 일봉 R1을 변경하지 않고 `signals/recommend_microstructure.py`가
계산 시작 직전300초의 `(매수 체결대금 − 매도 체결대금) / 전체 체결대금` 하나를 기록한다.
업비트 공개 trade의 BID/ASK는 체결 방향이며, 이 값은 **순자금 유입액이 아니다**.
정의 출처: [업비트 공식 trade 문서](https://docs.upbit.com/kr/reference/websocket-trade).

event_at과 received_at이 모두 `[decision_started_at−300초, decision_started_at)`에 있어야 한다.
동일 체결 ID 중복은 이 시각 필터 뒤에 제거한다. 무체결은 값0이 아니라 null이고,
수집 장애·재접속·호가 초기상태 부족은 전체 증거 부적격이다. 후보100개와 원래 rank는 보존한다.

최초 가설 `r1_boundary_trade_imbalance_v1`은 저장 `rr_ratio/p_up10/p_dn5/p_dn10/exp_downside`
5값이 모두 정확히 같은 연속 블록이 rank3/4 경계를 지날 때만 체결 불균형으로 동점을 푼다.
블록 밖은 그대로이고, 동점 기회가 없으면 기존 추천과 동일한 no-op 날로 비교에 포함한다.
필수 피처가 하나라도 null이면 임의 중립값/차순위로 대체하지 않는다. 이는 저장값 동점이지
반올림 전 생산 점수 또는 실제 변동성·유동성까지 같다는 주장은 아니다.

과거 snapshot37일의 순위만 확인한 결과 동점 기회는7일이었다(성과는 사용하지 않음).
이것은 연구 기회의 크기이지 상승/하방 개선의 증거가 아니다. 새 원본·사전 저장된 별도 순위와
성숙한 기존 라벨을 같은 날짜로 묶어 검증한다. 비용은 기존 net 결과에 다시 차감하지 않는다.
새 자동 학습·실제 추천 변경·승격은 없으며, 입력·시각·저장·평가 계약은 OPS §1.3을 따른다.

## 0.4 다음 선별안 — 상위10개 내 체결 순위 (09-08 시제품 → 09-10 별도 기록 시험)

`signals/recommend_trade_shortlist.py::plan_trade_shortlist(snapshot, feature_document)`는
기존 R1 상위10개를 먼저 고정하고,10개 모두 유효한300초 체결 불균형이 있을 때 그 값 내림차순,
동점은 원래 순위로 정렬해3개를 고른다.11위 이하는 들어오지 않는다.10개는 원100개 중 상위10%라는
**단일 초기설정**이며, 과거 성과로 여러 범위를 비교해 고른 값이 아니다. 과거 연구 결과를 알고 설계했다는
사실까지 지우거나 이를 미관측 자료로 발굴한 가설이라고 부르지 않는다.

- 기존8차의 **동점 시험은 그대로**이며 이 모듈 자체는 여전히 파일 I/O 없는 순수 계획 함수다.
  09-09 추가한 별도 publisher가09-10부터 자동 비교용 score/commit을 저장한다. 학습·실제 추천에는 연결하지 않는다.
- 원100개와 원 Top3를 보존한다. 상위10개 중 null이 하나라도 있으면 계획 전체 unavailable,
  글로벌 수집 품질이 유효할 때만 나머지90개 무체결 null을 허용한다. 실제0은 중립값으로 정렬한다.
- 입력은 신뢰된 native snapshot/feature 문서라는 좁은 계약이다. 기존 순수 검증기를 재사용하지만,
  파일 graph/raw/소스해시 재조회 또는 진입 전 내구 저장 증명까지 수행하는 함수는 아니다.
- 거래 한 건의 imbalance=+1도 앞설 수 있다. 체결 건수·대금·포화값과 저장된 ATR/거래대금 피처를
  진단으로 출력하지만 임의 임계값/새 필터를 끼우지 않는다. **상위10 제한이 하방 안전장치는 아니다.**
- 구조 확인용 현재 snapshot37개는 모두 원100개·상위10 구조를 갖췄다. 이는 실제 교체율/수집 적격률이나
  성과의 증거가 아니다. 이번 규칙의 정규 체결 입력과 사전 저장 비교 표본은 아직 없다.

효과 검증은 양쪽이 동일하게 적격인 새 날짜의 실제 원 Top3와 비교한다. 누락일을 숨기거나
결과가 빠진 종목을 차순위로 채우지 않는다. 상승 도달·하락 도달·최대 역행·동일 canonical 비용 차감 결과와
변동성/유동성 편중을 함께 확인한 뒤에만 실제 추천 연결을 별도로 판단한다.
독립 검토 PASS는 **알고리즘 계약의 구현 가능성**에 한정하며 성능 개선·배포 승인과 다르다.

### 별도 기록·평가 계약 (2026-09-09 연결)

`signals/recommend_trade_shortlist_trial.py`는 별도ID `r1_top10_trade_imbalance_v1`로
native raw/입력/소스 해시를 검증하고 score/commit을 신규 저장한다. 설계 고정일09-09,
prospective 시작일09-10이며 원 동점 시험의09-08 달력은 바꾸지 않는다.
미세체결 실험군은 추가1개/총2개이며 프로젝트 전체 과거 시도가2개라는 뜻은 아니다.
이미 존재하는 점수나 불확실 commit은 덮어쓰거나 재구성하지 않는다.

`signals/recommend_trade_shortlist_eval.py`는 canonical receipt/기존24h 라벨에 연결한다.
점수 파일·디렉터리 fsync 뒤 관측한 시간이 canonical 진입보다 **엄격히 빨라야** 비교 대상이다.
양쪽 Top3를 먼저 고정하고 선택 종목 결과가 빠지면 날짜 전체를 주 비교에서 제외한다.
정상 no-op은 포함하며, 전체100개 기준선은100개 모두 완결한 별도 보조 코호트만 사용한다.
날짜 동일가중 paired bootstrap과 기존 순비용 결과를 사용한다. 상방/하방 도달·역행·net 결과를 함께 보며
변동성·유동성 문맥은 진단일 뿐 보정된 인과 효과나 자동 승격 gate가 아니다.
한 건 체결의+1 포화·누락률·변경률도 숨기지 않는다. 소표본 효과는 탐색적이며 보장된 신뢰도라고 해석하지 않는다.

09-09 실제 첫 수집은96ns 시각 변환 오류로 피처 생성에 실패했다. 다음 실행 코드는 수리했지만
당일 원본을 고쳐 시험 점수를 소급 생성하지 않았다. 신규 정규 적격 표본은 아직0이고 추천 향상은 미검증이다.
추가 모델 학습·라벨 변경·실제 추천 승격은 이 연결에 포함하지 않는다. 운영 배선과 설치 경계는 OPS §1.3 참조.

## 0.5 detector_v1 (legacy/archive — Phase X-2-D 채택)

### 정의
- target: `next_max_return ≥ 0.20` (next-day high / next-day open - 1)
- model: XGBoost binary (objective=binary:logistic, n_estimators=400, depth=6, balanced sample_weight)
- 검증: 5-fold purged WF + per-fold inner OOF threshold (overfit 보정)
- 채택 후보: **C3** (regimes=bull_quiet+bull_volatile, threshold=OOF p99.95, cap=2)
  - 3/4 active fold 양수 EV (+7.40% 평균), 2024 EV -0.89% (resilient), no_trade 1/5
  - rank-based fallback (cap만, threshold 없음) 은 모두 EV 음수 → 폐기

### artifact
- `signals/models/ckpt/detector_v1.json` — 모델 (full panel 학습)
- `signals/models/ckpt/detector_v1_meta.json` — feature_cols, params, train range
- `output/detector_threshold.json` — operational rule + threshold + framing

### LEAK_COLS (학습 시 제외)
```
{net_under_tp, max_return, label, label_tail,
 next_open, next_high, next_low, next_close,
 next_max_return, next_eod_return, next_max_dd}
```
+ `next_*` prefix 전체

---

## 0.9 6-class 분포 모델 (legacy / 보조)

아래 §1~§9 는 **6-class softprob 분포 모델 (Phase 1)** 설계 — **현재 운영 X**, 보존.
재가동 시 `signals/predict.py` (= legacy multi-class entry, `scripts/predict_today_legacy.py` 가 호출).

언제 다시 볼지:
- detector_v1 의 보조 시그널로 분포 정보가 필요할 때
- Phase 2 hybrid 모델로 진화 시 분포 학습이 다시 메인이 될 가능성
- calibration / reliability diagram 컨셉 재사용

---

## 1. (legacy) 한 줄 결론

```
입력  : 어제까지 KRW 코인 일봉 + BTC 일봉 (multi-lookback 흐름)
출력  : { 코인,
          P(max ≥ 5%), P(max ≥ 10%), P(max ≥ 15%), P(max ≥ 20%),
          기대 max(high)/open,
          95% CI,
          BTC regime }
관계자: SIGNAL → (분포 + 메타) → LEDGER → 텔레그램 알림 (OPS)
```

---

## 1. 데이터 (data/)

### 1.1 소스
- **업비트 KRW 일봉** (메인): `pyupbit` 라이브러리
- **업비트 KRW 4h 봉** (보조): 일봉 안 장중 max(high) 정확히 측정
- **바이낸스 USDT 1h 봉** (legacy 보조): 김프 / binance lead-lag
- **바이낸스 USDT 일봉** (pump v2): D-1 volume-surge
- **historical 3 년치** 백필 (2023~ 현재). 알트는 상장 기준 가능한 만큼

### 1.2 저장
- `data/upbit_d1.db` (sqlite) — 일봉 OHLCV + 거래대금
- `data/upbit_4h.db` — 4h 봉
- `data/binance_1h.db` — 바이낸스 1h
- 스키마:
  ```sql
  CREATE TABLE candles (
    market TEXT,
    timestamp DATETIME,  -- KST naive (KST 09:00 시작 봉 = '...09:00:00')
    open REAL, high REAL, low REAL, close REAL,
    volume REAL,
    quote_volume REAL,   -- KRW 거래대금 (24h universe 선정용)
    PRIMARY KEY (market, timestamp)
  );
  ```

### 1.3 시간 처리 (KST 기준)
- 업비트 일봉 마감 = **UTC 00:00 = KST 09:00**
- DB 저장은 **KST naive** (pyupbit 기본). timestamp = KST 09:00:00 = 그 봉 시작
- 추론 시 "오늘 일봉" = KST 09:00 시작 ~ 다음 KST 09:00 마감
- 추론 시점 KST 09:05 = 어제 일봉 100% 마감 후 5 분
- 추론 시점 KST 08:50(preopen) = 어제 일봉이 아직 진행 중이라 마감된 D-2 일봉 피처를 쓴다. 그래서
  전날09:05와 Top3 평균2.75/3이 겹치고(60일) 같은 날09:05와는0.08/3이다(10-04 연구, 알림에 안내 문구).
  진행 중 D-1 일봉을 쓴 ‘새08:50’은 기록 전용 그림자로만 남긴다(PHASES27차, OPS §1.6).
- 바이낸스 DB는 UTC-naive로 저장한다. timezone-aware 변환 후 업비트 KST session과
  명시적으로 정렬하며 host timezone이나 naive `+9h`에 의존하지 않는다

### 1.4 유니버스
- **업비트 KRW 24h 거래대금 top N**
- **N = 100 은 초기값** (CLAUDE.md §2.5). top 50 / 200 도 비교 대상
- 유니버스 선정은 **fold train 종료 시점 기준** ← 양보 X (look-ahead 위생)
- 상폐 코인도 학습 데이터 포함 (survivorship bias 방어, 양보 X)

---

## 2. 라벨 (signals/labels.py)

### 2.1 정의 — Multi-class (분포 학습)

```python
def today_pump_label(open_today, high_max_today, bins=(0.0, 0.05, 0.10, 0.15, 0.20)):
    """
    오늘 일봉의 max(high) / open - 1 을 multi-class 로 분류.

    bin 0: max_return < 0%       (음봉, 한 번도 시가 위로 안 감)
    bin 1: 0%   ≤ max < 5%
    bin 2: 5%   ≤ max < 10%
    bin 3: 10%  ≤ max < 15%
    bin 4: 15%  ≤ max < 20%
    bin 5: 20%  ≤ max
    """
    max_return = high_max_today / open_today - 1
    for i, b in enumerate(bins):
        if max_return < b:
            return i  # bin 0 ~ 4
    return len(bins)   # bin 5 (≥20%)
```

**핵심**:
- 타겟 = **`max(high)`** (장중 한 번이라도 도달한 최고가) — not 종가
- 즉 "**기회 있었나**" 분류 (사용자 선택 옵션 4)
- 안정성 (장중 낙폭) 은 **라벨에서 X** — 가상 ledger 의 익절/손절 시뮬에서 처리 (LEDGER §3)
- max(high) 는 **4h 봉 데이터** 로 정확히 측정 (일봉만 보면 high 한 개 값)

### 2.2 bin 경계도 placeholder

`bins = (0.0, 0.05, 0.10, 0.15, 0.20)` 은 초기값 (CLAUDE.md §2.5).

EDA 에서 분포 보고 조정:
- 라벨 비율 보고: 각 bin 에 약 5~30% 씩 들어가는 게 학습 좋음
- bin 너무 sparse (예: bin 5 가 < 1%) → bin 합치기 또는 cutoff 변경
- bin 너무 흔함 (예: bin 1 이 > 50%) → 더 세분화

`scripts/label_distribution.py`로 bin 분포를 분석한다는 legacy 계획이며,
현재 스크립트는 미구현이다. 실제 탐색 코드는 `scripts/label_space_discovery_v2.py`다.

### 2.3 cumulative 확률 (출력용)

bin 별 확률 → cumulative 변환 (사용자 알림용):
```
P(max ≥ 5%)  = P(bin ≥ 2) = sum(p_2, p_3, p_4, p_5)
P(max ≥ 10%) = P(bin ≥ 3) = sum(p_3, p_4, p_5)
P(max ≥ 15%) = P(bin ≥ 4) = sum(p_4, p_5)
P(max ≥ 20%) = P(bin ≥ 5) = p_5
```

이 cumulative 가 알림에 표시 — 사용자가 익절 라인 결정 근거.

---

## 3. 피처 (signals/features.py)

### 3.1 알트 multi-lookback 피처
각 코인의 어제까지 데이터 기준.

**lookback 격자 {3, 5, 7, 14, 21} 일 은 초기값** (CLAUDE.md §2.5). EDA SHAP 보고 조정.

```
[가격/수익률]
  return_1d, return_3d, return_5d, return_7d, return_14d, return_21d

[변동성]
  vol_3d, vol_7d, vol_14d, vol_21d
  vol_inv_7d  (= 1/vol)  — "조용한 코인" 지표

[고저폭 압축]  ← xsec_alpha 검증된 강력 피처 (IC +0.179)
  range_contraction_3d, _7d, _14d
  (= 최근 N일 high-low 범위가 그 이전 N일 대비 얼마나 좁아졌는가)

[거래량]
  volume_ratio_3d, _7d   (현재 / N일 평균)
  volume_spike_score     (= ROC 표준화)
  volume_breadth_d       (시장 전체 거래량 활성화)

[기술지표]
  rsi_14, macd, macdhist, adx, bb_position
  squeeze_on            (볼린저 밴드 압축 플래그)
  roc_3d, roc_7d

[크로스섹션]
  rank_return_5d        (시장 내 5d 수익률 순위 백분위)
  breadth_ratio         (양수 5d 수익률 코인 비율)
  top_n_return_5d       (상위 5 개 평균 5d 수익률)
```

### 3.2 BTC regime 피처 (전 코인 공통)

```
[BTC 추세]
  btc_return_1d, _3d, _5d, _7d, _14d, _21d
  btc_ma_distance       (= (BTC_close - BTC_MA200d) / BTC_MA200d)

[BTC 변동성]
  btc_rv_30d            (BTC 30 일 실현변동성)
  btc_intensity_d       (= rv_30d 의 252 일 분위 0~1)

[BTC regime 4-state]
  btc_regime: bull_quiet / bull_volatile / bear_quiet / bear_volatile
    bull = ma_distance > 0
    volatile = intensity > 0.5
```

**MA200 / RV30 / intensity cutoff 0.5 는 초기값** (CLAUDE.md §2.5). 데이터로 더 좋은 조합 발견 시 변경 OK.

### 3.3 cross-market 피처 (보조)

```
kimchi_premium       (업비트 KRW vs 바이낸스 USDT 환율 보정)
kimchi_zscore_7d     (김프 7일 zscore)
binance_lead_1h      (바이낸스 1h 수익률 — 글로벌 선행)
```

### 3.4 정규화 원칙
- **per-day cross-sectional rank normalization**: 코인별 피처 (return, vol, range_contraction 등)
- **rolling z-score (시간축)**: BTC regime 피처 (전 코인 공통값)
- **ffill only**: `fillna(0)` 절대 X (gan_t known gap #2 — RSI=0 같은 불가능 값)

---

## 4. 모델 (signals/models/)

### 4.1 Phase 1 — XGBoost multi-class softprob

**왜 multi-class softprob?**:
- 라벨이 6-class → softmax 출력으로 각 bin 확률 자연스럽게
- gan_t pump_classifier 가 4-class softprob — 패턴 검증됨
- multi-class → cumulative 확률 → 분포 (사용자 핵심 요구)
- 빠른 학습 / SHAP 해석

**구조**:
```python
xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=6,                  # bin 갯수, EDA 후 조정
    eval_metric='mlogloss',
    n_estimators=...,             # Optuna 튜닝
    max_depth=...,
    sample_weight='balanced',     # sparse bin 대응
    ...
)
```

Optuna 튜닝 (max 50 trial):
- objective: mlogloss + per-bin macro F1
- HPO 는 IC/CRPS 가 아니라 **mlogloss + Brier** 위주 (분포 학습)

**파일**:
- `signals/models/xgb_phase1.py`
- `signals/models/ckpt/phase1_<date>.json`
- `signals/models/configs/phase1.json`

### 4.2 출력 변환 (raw probs → user-facing distribution)

```python
def predict_distribution(model, features, bins=(0.0, 0.05, 0.10, 0.15, 0.20)):
    """XGBoost multi-class → cumulative + 기대값 + CI"""
    p = model.predict_proba(features)  # (n_coins, 6)
    
    # cumulative
    cum = {
        'p_ge_5':  p[:, 2:].sum(axis=1),   # bin 2~5
        'p_ge_10': p[:, 3:].sum(axis=1),
        'p_ge_15': p[:, 4:].sum(axis=1),
        'p_ge_20': p[:, 5:].sum(axis=1),
    }
    
    # 기대값 (bin 중간값 × 확률)
    bin_centers = np.array([-0.025, 0.025, 0.075, 0.125, 0.175, 0.25])
    expected_max = (p * bin_centers).sum(axis=1)
    
    # 95% CI — bin 분포 기반 quantile (Phase 1 근사)
    ci_low, ci_high = approx_ci_from_bins(p, bin_centers, alpha=0.05)
    
    return cum, expected_max, ci_low, ci_high
```

### 4.3 Phase 2 — Hybrid (Transformer + TCN + FiLM + CVAE)

CVAE decoder = 진짜 conditional 분포 (sample 200 회 → quantile 직접). multi-class XGBoost 보다 정확한 CI.

ASSETS.md 의 gan_t/models/hybrid_model.py 구조 참조 → today_pump 안에 새로:

```
[입력 (B, T=21일, F)]
  ↓
[Transformer Encoder]   ← 글로벌 / 장기 의존성
  ↓
[Attention-guided TCN]  ← 로컬 / 모티프
  ↓
[Gated Fusion]          ← 진단 변수 (gate=Trend or Pattern 지배)
  ↓
[FiLM Regime Conditioning]  ← BTC 4-state
  ↓
[CVAE Decoder]          ← max(high)/open 의 conditional 분포
  ↓
[출력: 분포 + epistemic + aleatoric]
```

**언제 도입?**: Phase 1 multi-class 정확도 (Brier / reliability) 충분하면 Phase 2 유보. 부족하면 도입 + DM test.

**파일**: `signals/models/hybrid_phase2.py` (Phase 2 계획, 현재 미구현)

### 4.4 Phase 3 — APF motif 진단 (옵션, 학술)

ASSETS.md 의 fin/Attention Pattern Fields 참조. Phase 1/2 안정 후 사용자 컨펌.

---

## 5. Calibration (signals/calibration.py)

### 5.1 Reliability diagram (multi-class 보정)

각 cumulative 확률의 실제 적중률:

```python
def reliability_diagram(predictions, actuals, threshold=0.10):
    """예: P(max ≥ 10%) bucket 별 실제 적중률"""
    pred_probs = predictions['p_ge_10']
    actual_hit = actuals['max_return'] >= threshold
    
    # 확률 bucket (0~10%, 10~20%, ..., 90~100%) 별 actual rate
    buckets = np.linspace(0, 1, 11)
    reliability = []
    for low, high in zip(buckets[:-1], buckets[1:]):
        mask = (pred_probs >= low) & (pred_probs < high)
        if mask.sum() > 0:
            reliability.append({
                'pred_avg': pred_probs[mask].mean(),
                'actual_rate': actual_hit[mask].mean(),
                'n': mask.sum()
            })
    return pd.DataFrame(reliability)
```

**목표**: pred_avg ≈ actual_rate (대각선). 벗어나면 isotonic regression 으로 calibrate.
**저장 계획**: `output/reliability_curves.json` (각 cutoff 별, 현재 미생성)

### 5.2 Quantile coverage

기대 max + CI 의 실제 커버리지:
```python
def quantile_coverage(predictions, actuals, alpha=0.05):
    in_ci = (actuals['max_return'] >= predictions['ci_low']) & \
            (actuals['max_return'] <= predictions['ci_high'])
    return in_ci.mean(), 1 - alpha  # (실제, 목표)
```
**목표**: 실제 ≈ 목표 (95%). 너무 낮으면 CI 좁음 (과신), 너무 높으면 CI 넓음 (둔감).

### 5.3 Brier score

multi-class 보정 metric:
```python
brier = np.mean(np.sum((pred_probs - actual_one_hot) ** 2, axis=1))
```
낮을수록 정확.

### 5.4 정확도 알림 (사용자 신뢰 핵심)

매주 calibration 리포트:
```
이번 주 시스템 정확도:
  ≥+5% 예측 70% → 실제 68%  ✓ 정확
  ≥+10% 예측 45% → 실제 42% ✓ 정확
  ≥+15% 예측 20% → 실제 17% ⚠ 살짝 과신
  95% CI 커버리지: 93% (목표 95%) ⚠ 살짝 좁음
  Brier: 0.18 (낮을수록 좋음)
```

---

## 6. 추론 (signals/predict.py)

```python
def predict_today():
    """매일 KST 09:05 cron"""
    universe = get_top100_by_quote_volume(asof=now_kst())
    btc_features = compute_btc_regime_features(asof=yesterday_d1())

    predictions = []
    for coin in universe:
        alt_features = compute_alt_features(coin, asof=yesterday_d1())
        features = concat(alt_features, btc_features)
        
        # multi-class softprob
        bin_probs = model.predict_proba(features)  # shape (6,)
        
        cum_probs = compute_cumulative(bin_probs)
        expected = compute_expected(bin_probs)
        ci_low, ci_high = compute_ci(bin_probs)
        
        predictions.append({
            'coin': coin,
            'bin_probs': bin_probs,
            'p_ge_5': cum_probs['p_ge_5'],
            'p_ge_10': cum_probs['p_ge_10'],
            'p_ge_15': cum_probs['p_ge_15'],
            'p_ge_20': cum_probs['p_ge_20'],
            'expected_max': expected,
            'ci_low': ci_low, 'ci_high': ci_high,
            'btc_regime': btc_features['regime'],
        })

    # 정렬: P(≥10%) 또는 expected 기준
    return sorted(predictions, key=lambda x: -x['p_ge_10'])
```

출력 → `output/predictions_YYYYMMDD.csv`. LEDGER + OPS 가 받음.

---

## 7. 검증 (현재 forward + legacy `scripts/backtest_wf_ledger.py`)

### 7.1 Purged Walk-Forward (legacy 6-class)
- 5-fold WF + 10일 embargo (보수)
- 최종 holdout: 마지막 6 개월 절대 락

### 7.2 지표 (legacy 6-class 설계)

**필수 (트레이딩)**:
- net Sharpe (옵션 3 익절/손절 시뮬, 왕복 0.15% 차감)
- Max DD
- 누적 PnL
- TP/SL sweep 결과

**필수 (정확도 — 사용자 핵심 요구)**:
- **Reliability** (각 cutoff)
- **Brier score** (multi-class)
- **Quantile coverage** (CI 95% 커버)
- **per-bin accuracy**

**진단 (학술, 사후만)**:
- IC (Spearman, P(≥10%) vs 실제 max_return)
- ICIR

### 7.3 forward 검증
백테스트만 보고 결정 X. 슬롯당 한 번 생성한 R1 snapshot의 전 유니버스 score와
피처를 `output/recommend_snapshots/`에 저장하고, Telegram delivery receipt의
`sent_at`을 다음 15분 경계로 올린 시각부터 정확히 96봉을 평가한다. 예를 들어
09:10 발송이면 `[D 09:15, D+1 09:15)`가 라벨 창이다.

- `signals/recommend_score_labels.py`: up5/10/20, dn3/5/10,
  TP5/SL3 선도달, MFE/MAE, 비용 차감 EOD를 행별 기록
- `scripts/evaluate_recommend_score_labels.py`: AUC/Brier/calibration,
  up10이면서 dn5가 아닌 safe-up10, day-equal top-N, 유동성-matched,
  within-volatility baseline과 날짜-cluster CI 산출
- 대상 종목만 빈 15분봉은 무체결 flat 경로로 복원하고, KRW-BTC 기준 경로가
  불완전하면 해당 artifact는 partial로 보류
- 목표일 밖에서 만든 과거 재생은 `scheduled_replay`로 분리해 기본 forward
  통계에서 제외
- 실제 사용자가 받은 추천은 all-score가 아니라 `delivery_ok=True` cohort로
  별도 확인

2026-07-26 open R1/R2/A1 snapshot은 각각 PIT Top100 100행으로 생성됐고 R1
delivery receipt도 검증됐다. 다만 새 계약으로 성숙한 complete label은 아직 0개이며,
당일 preopen은 이전 설치본의 15m gate 실패로 snapshot이 없다. 따라서 최소
**2주 live paper**는 초기값일 뿐이며, 날짜 수와 CI 폭이 충분해질 때까지 활성 모델
승격 근거로 사용하지 않는다.

`recommendation_quality_meta_label_v1`의 과거 metadata는 자체적으로
`deployable=true`를 선언하지만, content-addressed model·feature schema digest·명시적
승인 digest가 없는 legacy bundle이다. 엄격한 loader 판정은 `LEGACY_UNBOUND`이고
pickle을 실행하지 않으며, 현재 추천을 강등하지도 않는다. 이 결과는 historical
diagnostic/model card일 뿐 운영 승격 증거가 아니다.

### 7.4 확률 정합성 제한

현재 R1의 독립 binary head와 일부 bucket calibration은 확률을 서로 독립적으로
만들기 때문에 수학적으로 필요한 포함관계가 항상 성립하지 않는다.

- 2026-07-26 09:05에 고정한 R1/R2/A1 snapshot은 각 100행 중 36행에서
  `p_up20 > p_up10`이었다. 이후 현재 DB·코드로 같은 R1 cutoff를 재계산한
  100행에서는 37행이었고, 두 경우 모두 `p_up10 > p_up5`와
  `p_dn10 > p_dn5` 위반은 0행이었다. 과거 snapshot은 이 차이를 소급 반영해
  덮어쓰지 않는다.
- OOS 76,731행 중 포함관계 위반은 442행(0.576%)이었다.
- 기존 R1 forward 165행/55일에서는 `p_up10` 평균 20.49% 대 실현 13.94%,
  `p_dn5` 평균 22.98% 대 실현 33.94%로 RR가 낙관적이었다.

따라서 현재 확률과 RR는 정렬용 score이지 calibrated trading probability로 해석하지
않는다. 모델 변경 승인 후에는 inner-OOF calibration을 우선하고, rank anchor를
유지해야 하면 `p_up5=max(p_up5,p_up10)`,
`p_up20=min(p_up20,p_up10)`, `p_dn10=min(p_dn10,p_dn5)`의 단조 projection을
versioned shadow로 비교한다.

포함관계 검사의 운영 규칙 (2026-07-28 변경): 전 유니버스 fail-closed는 첫 실운영
(2026-07-27 09:05)에서 rank 77 후보 하나로 R1 발송·라벨 축적 전체를 죽였다
(36/100 위반은 위 문단의 알려진 모델 성질이므로 이 하드체크는 만족된 적 없는
불변식이었다). 이에 **실제 발송 경로인 R1 top-k만 하드 fail-closed를 유지하고,
R1 유니버스 꼬리와 발송되지 않는 R2/A1 challenger 전체는 집계 진단 경고로
강등**했다(snapshot 검증·label artifact 검증 동일 계약, quant-reviewer
적대검증 2회전 통과). 2026-08-12에는 R2의 첫 top-k 위반(KRW-DOGE, rank 3)이
record-only 배치 전체를 실패시킨 운영 결함도 이 경계로 수정했다. 위반율은
snapshot에 저장된 확률 벡터에서 사후 재계산 가능하다. 근본 해결인 calibration
재구축·단조 projection은 여전히 사용자 승인 대기다. 또한 현재 R1 formatter에는
과거 문구 `둘 다 검증된 calibrated`가 남아
있어 위 증거와 모순된다. 현 활성 정렬·표시·알림은 임의 변경하지 않으며, 이
문구 수정과 versioned projection은 사용자 승인 후 적용한다.

---

## 8. 재학습 (legacy·미등록, signals/retrain.py)

현재 `scripts/retrain_run.sh`와 `signals/retrain.py`는 systemd/cron에 등록되지 않은
legacy 수동 경로다. 구현 gate도 아래 설계와 완전히 일치하지 않으므로 사용자 승인과
gate 재검증 전에는 활성 R1 학습·배포 경로로 사용하지 않는다.

### 8.1 cadence
- 주 1 회 (일요일 KST 06:00)는 과거 설계이며 현재 scheduler에는 미등록
- **cadence 는 초기값** (CLAUDE.md §2.5)

### 8.2 promotion gate
신 모델 채택 (셋 다 통과):
- new net Sharpe ≥ old - **delta_sharpe** (degradation 허용)
- new Brier ≤ old + **delta_brier** (정확도 큰 손실 X)
- new reliability max deviation ≤ old + **delta_reliability**

**초기값**: `delta_sharpe = 0.1`, `delta_brier = 0.02`, `delta_reliability = 0.05` (CLAUDE.md §2.5).

### 8.3 실패 시
- 후보 폐기, 이전 모델 유지
- `output/retrain_history.json` 기록
- 3 회 연속 실패 → 텔레그램 경고 + 사용자 컨펌

---

## 9. drift 감지
OPS.md 참조. 시그널 정확도 + 분포 안정성 감시.

---

## 10. 책임 경계

| 이건 SIGNAL 의 일 | 이건 SIGNAL 의 일 X |
|---|---|
| 라벨 정의 (multi-class bin) | 가상 사이징 → LEDGER |
| 피처 계산 | 익절 / 손절 시뮬 → LEDGER |
| 모델 학습 / 추론 (분포 출력) | 알림 전송 → OPS |
| Calibration | cron 스케줄 → OPS |
| 백테스트 정확도 metric | 텔레그램 포맷 → OPS |

---

## 11. 핵심 파일 인덱스

| 파일 | 역할 |
|---|---|
| `data/collector_d1.py` | 업비트 일봉 수집 (이미 작성) |
| `data/collector_4h.py` | 4h 보조 (장중 max(high) 정확히) |
| `data/collector_binance.py` | 바이낸스 1h |
| `data/collector_binance_d1.py` | pump v2용 바이낸스 일봉 |
| `data/database.py` | sqlite 헬퍼 (이미 작성) |
| `signals/recommend.py` | 현재 R1 score와 risk-reward 정렬 |
| `signals/recommend_snapshot.py` | 슬롯별 immutable score snapshot |
| `signals/recommend_score_labels.py` | receipt 이후 96봉 forward label |
| `scripts/recommend_send.py` | 현재 R1 Telegram 발송 |
| `scripts/recommend_today.py` | R1/R2/A1 전용 원장 기록 |
| `scripts/label_recommend_snapshots.py` | snapshot 전 유니버스 label 생성 |
| `scripts/evaluate_recommend_score_labels.py` | 전 유니버스 forward 평가 |
| `signals/labels.py` | today_pump_label (multi-class) |
| `signals/features.py` | alt + BTC + cross-market 피처 |
| `signals/models/xgb_phase1.py` | legacy Phase 1 multi-class softprob |
| `signals/models/hybrid_phase2.py` | Phase 2 계획 (미구현) |
| `signals/calibration.py` | reliability / Brier / coverage |
| `signals/predict.py` | legacy 일일 추론 (분포 출력) |
| `signals/retrain.py` | legacy 수동 재학습 (scheduler 미등록) |
| `signals/validate.py` | Purged WF |
| `scripts/backtest_wf_ledger.py` | legacy Phase 1 WF 백테스트 |
| `scripts/label_space_discovery_v2.py` | legacy label-space 분석 |
| `scripts/predict_today.py` | legacy detector 수동 추론 (기본 dry-run) |
