# R1 챌린저 독립 적대검증 설계 v1

작성일: 2026-07-25

역할: `quant-evaluator` 독립 감사 설계

범위: 코드·데이터 읽기와 검증 설계만 수행. production 코드, 핵심 문서, `output/`은 수정하지 않음.

## 0. 결론부터

상방-head 후보와 하방-veto 후보는 바로 운영에 넣으면 안 된다. 현재 기준에서 둘 다 가능한 최고 판정은 **SHADOW**다. 이유는 두 가지다.

1. 2026-07-25 현재 새 snapshot/label evaluator의 실제 `forward_observed` 표본은 0일이다.
2. 기존 R1 연구 하네스에는 공정 비교 전에 반드시 바로잡아야 할 차이가 있다.
   - `signals/recommend.py:363`의 `raw_tr = m.predict_proba(Xtr)`는 **train in-sample prediction**이다. 함수·주석의 “OOF calibration” 표현과 다르다.
   - 과거 연구용 `downside_head_riskreward_v1.build_panel()`은 현재 시점 전체 데이터에서 70봉 이상인 종목의 과거 초기행까지 남긴다. 실제 일별 운영은 해당 시점에 70봉이 안 되면 그 종목을 제외한다.
   - 초기 연구 일부는 live static top100이 아니라 `top100 OR qv_surge`를 썼고, 15분봉이 한 개만 있어도 경로가 있다고 처리했다.
   - live head는 09:00 open 기준 고가/저가 라벨로 학습하지만 사용자가 행동 가능한 시각은 실제 전송 후다. 09:05 슬롯 실발송은 과거 로그상 약 09:10이어서 대개 첫 실행 가능 봉이 09:15다.
   - 현재 full-universe labeler는 open 슬롯에서 `[09:15, 다음날 09:00)`의 95개 봉을 사용한다. “실행 후 24시간”이라고 부르기에는 15분 짧다.

따라서 후보 판정은 아래 계약으로 **R1도 함께 다시 점수화**해야 한다. 후보만 더 정교한 라벨·OOF를 쓰고 과거 R1 CSV와 비교하면 공정하지 않다.

---

## 1. 현재 R1의 실제 동작과 감사 판정

| 항목 | 실제 코드 | 감사 판정 |
|---|---|---|
| 입력 시점 | `build_market_features()`가 모든 raw 지표를 market별 `shift(1)` | PASS: day D 행은 D-1 이하 정보 |
| BTC regime | regime/BTC 피처를 다시 1일 shift | PASS |
| 유니버스 | live는 `f_qv_rank <= 100`, `f_qv_rank`의 원천은 D-1 quote volume | 조건부 PASS: historical replay의 70봉 자격을 point-in-time으로 재현해야 함 |
| head 피처 | 24개 `PRECURSOR_FEATURES`, label/`next_*` 제외 | PASS 후보. 전역 scaling/imputation이 생기지 않는지 후보별 재확인 |
| head 모델 | XGBoost 180 trees, depth 4, seed 42. 호출 때마다 즉석 fit | 사실 확인. artifact가 아니라 snapshot+manifest가 재현 앵커 |
| 학습 cutoff | `date < feature_date - 5일` | PASS. 후보/R1 양쪽 동일 cutoff 필요 |
| 상방 라벨 | day-D `high/open - 1 >= 10%` | 시점 누수는 없으나 사용자 실행가와 target mismatch |
| 하방 라벨 | day-D `low/open - 1 <= -5%` | 시점 누수는 없으나 운영 SL은 -3%, 실행가도 전송 후 가격 |
| 확률 보정 | final train model의 train 자체 prediction을 10 bucket으로 매핑 | **FAIL: 진짜 OOF 아님** |
| R1 정렬 | `p_up10 / max(p_dn5, 0.001)`, 이후 `p_dn10↑`, `p_up10↓`, `exp_downside↓` tie-break | 사실 확인 |
| 라이브 진입평가 | receipt `sent_at`을 다음 15분 경계로 ceil | PASS. 과거 09:00 open 평가는 live와 별도 cohort로만 허용 |
| 비용 | TP/SL/EOD gross에서 왕복 0.15% 한 번 차감 | PASS 목표. 각 후보 결과에서 단 한 번인지 독립 재계산 |
| 같은 봉 TP/SL | 둘 다 닿으면 SL 우선 | PASS: 보수적. 동시 터치 비율도 별도 보고 |

### 1.1 same-day leak 반증 체크

다음 중 하나라도 발견되면 성능과 무관하게 후보를 **REJECT**한다.

- day D `high`, `low`, `close`, `quote_volume` 또는 이들의 당일 cross-section을 day D 추천 입력으로 사용
- feature/label merge 후 전체 panel 중앙값·표준화·quantile을 계산
- outer test의 score, label, base rate를 threshold·veto cutoff·calibrator 선택에 사용
- NaN인 미성숙 미래 라벨을 `(NaN >= threshold) == False` 방식으로 0 취급
- Binance 00:00 UTC 일봉처럼 09:00 KST에 막 마감한 데이터를 실제 수집 완료 전에 사용
- open과 preopen을 한 모델/한 성과표로 섞음

특히 `b_vol_surge`를 쓰는 상방 후보는 운영 가용성을 별도 증명해야 한다. 현재 스케줄은 R1 open 발송이 끝난 뒤 `[10/11]`에서 Binance D1을 갱신한다. 그러므로 현 스케줄 그대로라면:

- 08:50 preopen: 마지막 완결 Binance 일봉은 반드시 한 단계 더 lag된 값이어야 한다.
- 09:05 open: 새 09:00 마감봉은 아직 R1 snapshot 생성 전에 수집되지 않는다.
- 새 마감봉을 쓸 후보는 “과거 DB에 있으니 사용 가능했다”로 처리하면 **availability leak**이다. 발송 전 수집·freshness gate가 실제로 끝나는 운영 변경을 승인받거나, 한 일 더 lag해야 한다.

---

## 2. 공정 비교의 고정 계약

### 2.1 비교 단위

- `preopen`과 `open`은 완전히 별개 실험이다.
- 같은 슬롯 안에서 R1, 상방 후보, 하방-veto 후보가 **같은 날짜·같은 유니버스·같은 실행가·같은 경로**를 공유해야 한다.
- 어떤 후보가 3개를 못 고른 날은 현금 0%를 포함해 그 자체로 평가한다. 선택된 거래만 모아 평균을 높이면 안 된다.
- 주지표는 날짜별 동일가중 수익률이다. 하루 3픽이면 먼저 그날 평균을 만들고, 코인 시장의 무휴 거래 특성에 맞춰 무추천일을 cash 0%로 두고 `sqrt(365)`를 쓴다.

### 2.2 실행 시각과 라벨

후보 학습과 평가는 아래 두 세트를 모두 만들되, **채택 주지표는 실행 가능 가격 기준**이다.

| 슬롯 | 실행가 | 고정 경로 |
|---|---|---|
| preopen | 09:00 첫 15분봉 open | `[D 09:00, D+1 09:00)` 96봉 |
| open | 실제 `sent_at`의 다음 15분 경계 open. historical replay는 09:15로 고정 | `[D 09:15, D+1 09:15)` 96봉 |

현재 labeler의 open 경로는 09:15부터 다음날 08:45까지 95봉이다. 후보 최종 판정 전에 96봉의 true 24h 경로로 R1과 후보를 함께 재산출해야 한다. 기존 95봉 결과는 `target_day_close` 보조 지표로만 병기한다.

고정 출력:

- `MFE`, `MAE`
- `up5`, `up10`, `up20`
- `dn3`, `dn5`, `dn10`
- `TP+5% before SL-3%`
- `SL-3% before TP+5%`
- neither이면 24h 마지막 close 수익
- TP/SL 동시 터치면 `SL first`
- gross와 `gross - 0.0015` net을 둘 다 저장하되 판정은 net
- 같은 봉 동시 터치율 및 TP-first 민감도는 보조표로 보고

현재 head의 `up10(open 09:00 기준)`과 실제 사용자의 `up10(09:15 실행 기준)`은 다른 target이다. 상방 후보가 진짜 개선인지 보려면 후자를 직접 학습하거나, 전자로 학습했더라도 후자에서 성능이 유지되어야 한다.

### 2.3 경로 완결성

- 기준 BTC grid가 완결되고 DB horizon이 마지막 봉 마감까지 덮여야 한다.
- BTC는 있는데 대상 코인 봉만 없으면 “무거래”로 보고 직전 close flat-fill한다.
- BTC grid가 비면 수집 장애이므로 해당 날짜 전체를 defer한다.
- 단순 `len(bars) > 0` 또는 `len(bars) == 96` 필터는 금지한다. 전자는 부분 경로를 통과시키고, 후자는 저유동 종목을 체계적으로 지운다.
- 경로 품질(`complete`, `flat_filled`, incomplete reason)별 결과를 병기한다.

### 2.4 모델 학습·OOF

외부 test를 한 번만 보는 nested expanding walk-forward를 사용한다.

1. **Outer fold**: 시간순 expanding train → 5일 embargo → 미래 test. 최소 6 folds.
2. **Inner expanding folds**: outer-train 내부에서만 raw OOF prediction 생성.
3. calibrator의 bucket edge와 hit map은 inner-OOF prediction/label로만 적합.
4. outer-train 전체로 final head를 fit하고 outer-test raw score를 산출한 뒤, 3의 calibrator만 적용.
5. feature 선택, XGB parameter, veto threshold, score 조합식은 outer-train 안에서만 고정.
6. 후보를 고른 뒤 별도의 exact-deployment replay를 수행한다. 이 replay는 live처럼 날짜별 또는 고정 주간 cadence로 expanding refit하며, R1도 똑같은 cadence로 다시 계산한다.

`predict_proba(Xtr)`를 “OOF”라고 부르면 즉시 감사 FAIL이다. 진짜 OOF row에는 반드시 그 row의 날짜보다 과거 자료로만 학습한 `inner_fold_id`가 있어야 한다.

### 2.5 point-in-time 유니버스

각 날짜 D의 후보 집합은 다음 조건을 모두 만족해야 한다.

- D-1 quote volume 기준 상위 100
- 그 날짜 시점에 실제 존재하고 거래 가능했던 KRW market
- 그 날짜까지 확보된 D1 history가 최소 70봉
- 상장 전 행 없음
- 보존된 상폐 데이터는 상폐 전까지 포함
- cross-sectional rank는 그 날짜에 자격 있는 집합에서만 계산

현재 전체 DB를 한 번에 만든 research panel은 “나중에 70봉을 채운 종목”의 초기행을 남길 수 있다. challenger와 R1을 매 날짜 as-of로 재구축하거나, `history_count_asof >= 70`을 명시적으로 적용해야 한다.

---

## 3. 15분 first-passage 데이터의 실제 범위

2026-07-25 현재 `data/upbit_15m.db` 직접 조회:

- raw 범위: `2023-05-03 18:45:00` ~ `2026-07-25 21:45:00`
- 8,760,395 rows, 271 markets, timestamp가 걸친 달력일 1,180일
- KRW-BTC 09:00 기준 96봉 완결 trading-day:
  - 최초 `2023-05-04`
  - 최종 `2026-07-24`
  - 1,162일 완결 / 18일 불완전
- 현재 production static-top100 panel과 BTC 완결일을 교차하면:
  - 116,170 coin-days / 1,162일
  - 일평균 99.97개
- 기존 R2 OOS artifact의 표본은 38,682 universe rows / 765일 (`2024-04-03`~`2026-06-01`)
- 선행 fixed-barrier 교락 감사가 실제 사용한 표본은 45,557 coin-days

이 숫자들은 서로 다른 모집단이다. 116,170은 잠재 substrate이고, 38,682/45,557은 기존 fold·기간·필터가 적용된 연구 표본이다. 새 후보 보고서는 다음 단계별 N을 모두 적어야 한다.

`raw universe → history eligible → outer OOS → BTC path complete → target flat-filled/complete → valid label → selected top3`

과거 38,682/45,557을 새 후보의 최종 N으로 복사하면 안 된다. true 24h open 경로, point-in-time 70봉, 최신 상장/상폐 상태를 적용한 뒤 다시 계산한다.

---

## 4. fixed-% barrier의 변동성 재포장 방지

선행 감사에서 45,557 coin-days 기준 고정 barrier 관련 정렬키는 D-1 변동성 십분위와 단조였고, 저변동~고변동 사이 발생률이 3.7~7.7배 벌어졌다. 반면 비용 차감 net EV는 변동성 구간에 거의 평평했다. 따라서 pooled AUC나 `P(up10)` lift만 높으면 채택할 수 없다.

각 후보는 아래를 모두 제출한다.

1. 후보 score와 `f_atr_pct_14`, `f_rv_7d`의 Spearman 상관.
2. top3의 ATR decile 분포 vs 전체 top100.
3. ATR decile별 `up10`, `dn5`, TP-first, SL-first, net.
4. 같은 날짜·같은 ATR band 안에서 후보 top3와 non-top control의 paired lift.
5. fixed-% 라벨과 ATR-normalized 라벨을 병행:
   - 경제적 주지표: 실제 `TP+5%/SL-3%`
   - 메커니즘 진단: D-1 ATR 배수 상·하 barrier
6. “변동성이 높은 것을 고른 결과”를 제거한 뒤에도 후보 순위가 상방 또는 net을 개선하는지 확인.

최소 조건은 다음과 같다.

- top3 개선 방향이 ATR low/mid/high 3개 band 중 최소 2개에서 같아야 한다.
- within-volatility paired lift가 pooled 결과와 같은 방향이어야 한다.
- pooled 성능이 좋아도 within-vol lift가 0 또는 역전이면 **SHADOW 불가, REJECT**.
- ATR-normalized target만 좋아지고 실제 +5%/-3% net이 안 좋아지면 연구상 진단일 뿐 운영 후보가 아니다.

---

## 5. 반드시 붙일 baseline panel

모든 baseline은 같은 날짜, 같은 top100, 같은 15분 경로, 같은 비용을 쓴다.

1. **Frozen R1**: 현재 in-sample bucket calibration까지 그대로 재현한 champion.
2. **R1-calibration-repaired**: 동일 feature/XGB/ratio지만 inner-OOF calibration만 수리한 control.
3. **Upside-only**: repaired `p_up10` 내림차순 top3.
4. **Downside-only**: repaired `p_dn5` 오름차순 top3. “안 움직이는 대형주 선택” 퇴행 확인용.
5. **Monkey EW**: 그날 top100 전체 동일가중 평균. random top3의 기대값.
6. **Monkey top3 distribution**: 날짜별 3개 비복원 무작위 추출을 seed 고정해 최소 1,000회 반복. R1/후보가 random 분포의 어느 percentile인지 보고.
7. **Liquidity-matched**: 각 top3 픽마다 D-1 `f_log_qv` 최근접 non-top control을 중복 없이 매칭.
8. **Volatility-matched**: 같은 날 ATR band의 non-top control.
9. **Liquidity+volatility matched**: 표본이 허용하면 두 축 동시 nearest/stratified matching.

`top100 전체 매수`와 `random top3`는 평균의 기대값은 같지만 변동성과 DD가 다르므로 둘 다 필요하다. baseline이 후보 top3를 포함하는 경우와 제외하는 경우도 명시한다.

---

## 6. 공통 판정 지표

### 6.1 사용자 목적에 직접 대응하는 1차 지표

- `P(SL-3 first)` — 낮을수록 좋음
- `P(TP+5 first)` — 높을수록 좋음
- `TP-first / max(SL-first, eps)` — 진단용, 단독 최적화 금지
- `MFE`, `MAE`, `CVaR95`
- 비용 차감 `mean net/day`, 누적 복리, Sharpe/Sortino `sqrt(365)`, Max DD
- `up10` hit와 `dn5` hit

trade-row pooled 평균 외에 **day-equal**을 반드시 주지표로 쓴다. 불확실성은 날짜 전체를 resample하는 date-cluster bootstrap으로 계산한다.

### 6.2 head 품질

- AUC, Brier, calibration bucket별 predicted vs actual
- p_up10/p_dn5 top·middle·bottom 삼분위 실제율
- top3 precision과 full-universe base rate
- slot별, regime별, ATR band별, 유동성 band별
- 확률 수준 보정과 순위 판별력을 구분

### 6.3 selection 기록

- 시도한 feature set, label, hyperparameter, veto threshold, score식의 총 trials
- 최종 후보가 outer-test를 본 뒤 수정됐는지
- PSR/DSR/MinTRL은 실제 trials 수로 사후 표기

DSR/Holm이 나쁘다는 이유만으로 net+forward가 좋은 후보를 죽이지는 않는다. 반대로 leak·시간정합성·비용 FAIL은 성능과 무관하게 REJECT한다.

---

## 7. 후보별 적대검증 체크리스트

### 7.1 상방-head 후보

- [ ] target이 `open 09:00`인지 `실행 후 09:15`인지 명시
- [ ] user-actionable 09:15 target에서 p_up10 AUC와 top3 lift 재검증
- [ ] inner expanding OOF calibration 사용
- [ ] R1-calibration-repaired와 비교해 “보정 수리 효과”와 “새 feature/head 효과” 분리
- [ ] `b_vol_surge` 등 외부 피처는 슬롯별 실제 가용시각 증명
- [ ] coverage가 낮은 Binance 매핑 종목을 삭제해 universe를 쉽게 만들지 않음
- [ ] missing external feature는 train-only imputation 또는 명시적 missing indicator
- [ ] high-vol coin만 고르는지 within-ATR 비교
- [ ] p_up10뿐 아니라 TP5-before-SL3와 비용 차감 net 개선
- [ ] p_dn5/SL-first 비열화 한도 준수
- [ ] 동일 fold에서 frozen R1과 repaired R1 score를 함께 생성
- [ ] outer fold 6개 중 개선 부호와 최악 fold 보고

### 7.2 하방-veto 후보

- [ ] veto 입력은 해당 날짜 이전에 계산 가능한 score만 사용
- [ ] veto cutoff는 outer-train 또는 inner-OOF에서만 결정
- [ ] 매일 후보 부족 시 다음 순위로 채울지, 현금으로 둘지 사전에 고정
- [ ] veto가 제거한 종목과 대체한 종목을 날짜별 paired row로 보존
- [ ] `p_dn5`가 아니라 운영상 핵심인 `SL-3 first` 감소도 측정
- [ ] 고위험 제거가 단순 저변동 BTC/DOGE/TRX 선택으로 퇴행하지 않는지 검사
- [ ] 상방 유지율 `candidate TP-first / R1 TP-first` 보고
- [ ] up10, MFE, TP-first, net 중 무엇을 희생했는지 숨기지 않음
- [ ] dump flag/A1/R2/R3 등 기존 시도와 pick overlap 및 신규성 보고
- [ ] cutoff grid 전체 trials 기록
- [ ] same-day paired bootstrap으로 `R1 - veto` 차이 CI 계산
- [ ] leave-one-month-out에서 하방 개선 부호 유지

---

## 8. 최소 판정 기준

아래 숫자는 첫 challenger gate의 운영 초기값이다. 데이터가 명백히 다른 값을 지지하면 이유와 함께 조정할 수 있지만, 결과를 본 뒤 조용히 바꾸면 안 된다.

### 8.1 즉시 REJECT

- 4대 위생 중 하나라도 FAIL: look-ahead/availability leak, point-in-time universe FAIL, 비용 미차감·이중차감, 자동주문 추가
- true OOF가 아닌 train prediction을 OOF라고 사용
- 기존 R1과 다른 날짜·유니버스·경로로 비교
- within-volatility 비교에서 lift가 0 이하 또는 pooled 방향과 역전
- candidate net과 추천 품질이 R1 및 monkey에 모두 뒤짐
- 개선이 한 outer fold 또는 한 regime에만 존재
- TP-first/SL-first 동시봉 처리에 따라 결론이 뒤집히며 보수적 SL-first에서는 개선 없음

### 8.2 Offline SHADOW 진입 최소선

공통:

- 위생 4/4 PASS
- path-complete outer OOS 최소 360일, top3 유효 row 최소 900
- R1 baseline exact-reproduction self-check PASS
- 비용 차감 day-equal net의 `candidate - R1` point estimate > 0
- 6개 outer fold 중 최소 4개에서 net 차이 부호가 양수
- monkey EW와 liquidity-matched baseline보다 주지표가 양호
- within-volatility lift가 3 band 중 최소 2개에서 같은 방향
- date-cluster CI가 아직 0을 포함해도 SHADOW는 허용하되, 반대편 꼬리가 크면 REJECT

상방-head 추가 조건:

- full-universe p_up10 AUC point estimate `>= 0.52`
- p_up10 AUC의 date-cluster CI 하한이 최소 0.50 부근이며 R1 0.477보다 개선
- `TP-first`가 R1보다 최소 +3%p 또는 top3 `up10`이 최소 +20% 상대 개선
- `SL-first` 악화는 +2%p 이내
- Brier/calibration error가 repaired R1보다 악화되지 않음

하방-veto 추가 조건:

- `SL-first`가 R1보다 최소 -3%p
- 상방 유지율(`TP-first candidate / TP-first R1`) 최소 80%
- 비용 차감 net이 R1보다 악화되지 않음
- leave-one-month-out의 하방 개선 부호가 최소 75% 구간에서 유지

이 조건을 통과해도 **ACTIVE/ADOPT가 아니라 record-only SHADOW**다.

### 8.3 Forward ADOPT 신청 최소선

- 같은 snapshot에서 R1과 challenger를 동시에 점수화한 `forward_observed` 90일 이상
- 유효 paired top3 최소 240 rows, 예정 universe score coverage 90% 이상
- 실제 delivery 성공 R1 cohort와 동일한 실행 시각·경로
- date-cluster CI95 기준:
  - 비용 차감 day-equal `candidate - R1 > 0`의 하한 > 0
  - 상방 후보: `TP-first` 차이 하한 > 0, `SL-first` 비열화 +2%p 이내
  - 하방 veto: `SL-first` 감소 하한이 최소 3%p, 상방 유지율 80% 이상
- candidate 자체의 비용 차감 day-equal mean net > 0
- monkey top3 분포와 liquidity/volatility matched baseline보다 우위
- 월별 부호가 최근 3개 완결월 중 최소 2개에서 동일
- calibration drift가 허용 범위이고 fallback/head failure 일자가 없음
- 사용자 승인 전까지 발송 순위와 알림 문구 변경 금지

forward 90일 전에는 아무리 backtest가 좋아도 판정 상한은 SHADOW다.

---

## 9. 두 연구 후보가 넘겨야 할 최소 산출물

각 연구자는 production 파일을 건드리지 않고 `_workspace`에 다음을 남겨야 한다.

1. 후보 한 줄 정의와 시도 trials 수
2. feature availability 표
3. outer/inner fold 날짜와 embargo 표
4. 날짜×coin OOS score:
   - R1 frozen
   - R1 repaired
   - candidate
   - raw probability, calibrated probability, rank, veto 여부
5. 경로 row:
   - execution_at/start, entry, 96-bar completeness
   - TP/SL first passage, same-bar flag, MFE/MAE, gross/net
6. baseline panel 결과
7. ATR/liquidity/regime 층화
8. fold별·월별 paired 차이
9. 재현 manifest: DB hash/최대 timestamp, 코드 SHA, seed, feature list, model params
10. 위 §7 체크리스트 자체 판정

독립 evaluator는 연구 노트의 요약 숫자를 복사하지 않고 row artifact에서 다시 계산한다.

---

## 10. 최종 판정 카드 템플릿

```text
## VERDICT: SHADOW | REJECT | ADOPT_REQUEST — {후보명}
- 비교 슬롯/실행: preopen 09:00 | open receipt→next15m
- 표본: OOS days / top3 rows / full-universe rows / path coverage
- 1차: Δnet/day / ΔTP-first / ΔSL-first / ΔMFE / ΔMAE
- head: p_up10 AUC·Brier·calibration / p_dn5 AUC·Brier·calibration
- baselines: frozen R1 / repaired R1 / monkey / liquidity / volatility
- 층화: folds positive / months positive / regimes / ATR bands
- 비용: 0.15% once PASS|FAIL
- same-bar: SL-first primary, ambiguity rate, sensitivity
- 위생: leak / availability / universe / cost / auto-order
- selection: trials=N, outer-test 재사용 여부
- forward: observed days=N 또는 없음
- 판정 근거:
- 사용자 승인 필요:
```

## 11. 현재 시점의 감사 상태

- 기존 R1의 D-1 feature shift 자체는 살아 있다.
- p_dn5는 기존 forward 소표본에서 판별력이 있었지만 수준 보정과 실행가 target 정합이 필요하다.
- p_up10은 기존 delivered-pick 표본에서 AUC 0.477이어서 상방 후보가 해결해야 할 핵심이다.
- 전 유니버스 snapshot 기반 새 forward 표본은 아직 0일이다.
- 따라서 두 연구 후보의 당장 가능한 결과는:
  - 위생 또는 matched lift FAIL → REJECT
  - offline 기준 통과 → SHADOW
  - 실제 90일 paired forward까지 통과하고 사용자 승인 → ADOPT_REQUEST
