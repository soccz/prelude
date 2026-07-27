# 연구 노트: Angle C — 이번주 급등 케이스스터디 (precursor 역분석)

## 가설
"급등 1~2일 전, 그 코인의 일봉 거래대금(quote_volume)이 자기 20일 평균 대비 비정상적으로 선행 증가한다 — 즉 펌프는 cold 가 아니라 hidden accumulation 이 거래대금에 먼저 새어나온다." (단, 이 가설은 절반만 맞다 — 아래 Group B 반증)

## 무엇을 돌렸나
- 10개 급등 코인의 first-pump-day D 를 식별, D-1 및 그 이전 일봉으로 precursor 상태 재구성 (`scripts/case_study_precursor_v1.py`).
- 전체 패널 (2024+, 252코인 제외 BTC, n=141,255 코인-일) base rate / lift.
- purged walk-forward (5-fold, 5일 embargo, train-before-only) OOS lift (`scripts/case_study_wf_lift_v1.py`).
- Group B(cold) 코인은 15m DB 로 D 09:00 이전 12h 미세동학까지 확인.
- 산출: output/case_study_precursor_state.csv, output/case_study_panel.parquet.

## leak·시간정합성 방어
- feature = D-1 까지 close/high/low/qv 만. 라벨 = D 의 ho_1d 를 shift(-1)로 D-1 행에 부착.
- LEAK CHECK 통과: XLM D-1(05-27) 행의 next_ho=0.3223 == D(05-28) 행 ho_1d=0.3223 (정확히 일치, off-by-one 없음).
- 동적 유니버스 rank 도 D-1 까지 trailing qv_ma20 로만 산출.
- WF: fold test 시작 5일 전까지만 train, embargo 적용. in-sample 아닌 pooled OOS lift 보고.

## 시도 조합 수 (selection deflate 용)
- 단일 precursor 11종 + 2-way intersection 5종 + recent_attn 1종 = 약 17개 룰 평가.
- 임계(1.5/2.0/3.0 등)는 1차 lift 보고 hand-pick → quant-evaluator 가 deflate 필요.
- 케이스 코인/날짜는 사용자 제공(발견 아님), 패턴 임계만 데이터에서 선택.

## 1차 결과 (모두 OOS, base y20=1.70%)
| precursor | OOS n | OOS hit | OOS lift | 이번주 커버 |
|---|---|---|---|---|
| 어제qv/5d평균 >= 3.0 | 3,774 | 7.95% | 4.47x | PROVE,ALT,XLM |
| 어제qv/20d평균 >= 3.0 | 6,363 | 7.07% | 3.97x | PROVE,ALT,XLM |
| 7일 모멘텀 >= +20% | 5,249 | 7.13% | 4.00x | WLD |
| spike2 & 7d모멘텀+10% | 2,782 | 8.30% | 4.67x | XLM |
| cumvol1.5 & spike2 | 3,379 | 8.20% | 4.61x | PROVE,ALT |
| recent_attn(10일내 펌프이력) | 26,248 | 3.55% | 2.06x | META |

### 두 가지 family 발견 (핵심)
- **Group A — warm/volume-precursor 있음 (catchable):** PROVE(D-1 qv 60x baseline), ALT(37억), XLM(rank19, 441억), WLD(slow run-up, ret_7d +37%). 일봉 거래대금/모멘텀 precursor 가 D-1 에 깃발. → 10개 중 4~5개.
- **Group B — cold detonation (NOT catchable):** ID, GMT, VTHO, ERA. D-1 거래대금 flat/tiny(VTHO 1억, ERA 1.5억), 유니버스 rank 160~232(밖 깊숙이). 15m 로 D 09:00 이전 12h 봐도 ramp 없음(ID last45min 0.2억, ERA 0.01억, range<4%). 가격/거래량 microstructure 로 사전 예측 불가 = leak-free 음성 결과. (외생 catalyst가 09:00 이후 타격으로 추정.)
- **echo pump:** META(D-5 199억 펌프 후 D-1 cold)는 recent_attn 으로만 약하게(2.06x) 잡힘. PRL 은 어느 룰도 미발화(post-dump -17% drawdown_5d 후 발화 — drawdown 단독 lift 2.0x 이나 PRL D-1엔 미달).

### 사용자 직관 검증
- 동적 유니버스로 Group B 잡기: **반증.** D-1 에 유니버스(top100) 진입 조짐 없음(rank 160~232). 거래대금이 펌프 *당일* 09:00 이후에야 폭증.
- range_contraction(조용→폭발): **반증 재확인.** lift 0.73x (역효과). 수축은 펌프 선행조건 아님.
- low-cap 가설: rank>100 lift 0.83x(낮음), rank<=50 lift 1.39x. 대형이 base rate 높음 — RESEARCH.md §5 와 일관.

## evaluator 검증 요청
1. EDA hit ≠ Sharpe: 위 lift 는 high/open>=20% "터치" 기준. TP-before-SL/실현손익 경로 미반영. ledger Sharpe 로 재검증 필요(과거 momentum hit 21%인데 Sharpe -5.2 함정).
2. 17개 룰 hand-pick 임계 → DSR/deflate 로 selection bias 차감 요청.
3. Group A recall 상한 = 약 40~50% (Group B 구조적 미검출). recall 낮음은 모델 결함이 아니라 시장 구조 — 이 점 명시.
4. recent_attn n=26k 로 크지만 lift 2.06x 약함 → noise 가능성 점검.
