# 연구 노트: recommendation_scorer_v1 (일일 급등확률 추천 레이더)

## 가설
검증된 leak-free 선행패턴(qv_surge, bounce_off_7d_low, short-momentum, ATR/RV)을
하나의 모델로 결합하면, 단일 패턴 lift(3.0~3.7x)보다 **cross-sectional top-K 추천**의
precision이 더 안정적으로 base rate를 상회한다. "변동성·거래대금 급증·단기 모멘텀이
동시에 켜진 코인이 다음날 +20% 갈 후보"라는 가설을 daily top-K 형태로 운영 가능하게 만든다.

## 무엇을 돌렸나
- 코드: `scripts/recommendation_scorer_v1.py` (self-contained, prelude 안에 신규).
  feature 빌더는 검증된 `univariate_precursor_lift_v1.build_market_features` +
  `add_cross_sectional` + `regime_split_precursor_v1.attach_btc_regime` 재사용.
- 라벨: pump20 = (high_D/open_D - 1) >= 0.20 (메인). base rate full 1.73% / universe 1.82%.
- 24개 D-1 feature (검증된 선행 set) → 3 스코어러 비교:
  (a) composite = 9개 cross-sectional rank 가중합(OOS lift 비례 weight, 학습 X),
  (b) logit(class_weight=balanced), (c) XGBoost(depth4, scale_pos_weight).
- calibration: **bucket-based historical hit** — train raw score를 20-bin qcut →
  bin별 실제 hit rate → test raw score를 train bin에 매핑(rare-event raw prob 과신 방지).
- 검증: purged walk-forward, expanding train(min 35%) + embargo 5d, 6 fold.
  OOS rows=76,731, OOS 기간 2024-04-02 ~ 2026-05-31 (765 거래일).
- 동적 유니버스: D-1 qv top100 OR qv_surge_7d>=3x (둘 다 D-1 정보).

## leak·시간정합성 방어 (§1 체크리스트)
- 모든 feature는 build_market_features에서 market별 .shift(1) → day D row는 D-1까지만 봄.
- 라벨 pump20는 day D의 open/high만(타겟). feature에 same-day OHLC 안 섞임.
- BTC regime은 attach_btc_regime에서 무관 시계열 1일 shift → day D는 D-1 regime.
- 모델/calibration bin은 **expanding train(과거)에서만 fit**, test fold(미래)엔 적용만.
  calibration bin도 train에서 fit → OOS에서 historical-hit 매핑(val/test fit 금지).
- 유니버스 컷도 D-1(f_qv_rank, f_qv_surge_7d shift됨).
- 자가검증: precision@1=8.1%(98% 같은 수치 X), calibration 단조증가 → leak 의심신호 없음.

## 시도 조합 수 (selection deflate용)
- 스코어러 3 (composite/logit/xgb) × 라벨 메인 1(pump20) × calibration 1(bucket).
- feature 24개는 직전 univariate sweep(34 feat × 3 lab = 102 try)에서 검증된 dir=high
  상위를 hand-pick(데이터 본 뒤). composite weight도 그 OOS lift에서 도출.
- best scorer는 ALL-regime K=3 precision으로 1회 선택(cal_xgb). → evaluator가 deflate.

## 1차 결과 (OOS, calibrated cal_xgb = best)
| K | precision@K | lift | recall |
|---|---|---|---|
| 1 | 8.1% | 3.99x | 4.0% |
| 2 | 8.4% | 4.12x | 8.2% |
| 3 | 7.5% | 3.71x | 11.1% |
| 5 | 7.4% | 3.63x | 18.1% |

- 세 스코어러 거의 동률(logit/composite도 lift 3.5~4.0). XGB가 미세 우위.
- regime별 prec@3: bull_quiet 7.4%/lift4.64, bull_volatile 7.8%/3.61,
  bear_quiet 6.1%/3.72, bear_volatile 8.3%/2.94. (bear_volatile은 prec 높지만
  base rate도 높아 lift 낮음 — 펌프 풍부 regime.)
- 최근 14거래일(2026-05-18~31) top-3 실측: 4/14일 ≥1 hit, hit_precision 9.5%
  (OOS 추정 7.5%와 정합). top-5는 6/14일, 8.6%.
- net PnL은 본 작업에서 미산출(사용자 명시: 추천 품질 1순위, PnL 사후 참고).

## 주의: calibration ceiling (evaluator 검토 요청 지점 #1)
- 최상위 bin pred=0.120 vs actual=0.079 — 최고점에서 mild 과신. raw XGB score가
  top에서 saturate → 같은 calibrated prob(예 14.3%)이 여러 코인에 동시 부여(tie).
- 운영 표시는 "급등확률 ~14%" 같은 bucket 값으로(절대 90% 같은 raw 금지). 이 프로젝트
  +20% tail 90%→실제 11.6% 전적상, 14% 표기도 상한이라 명시 필요.

## recall gap (evaluator 검토 요청 지점 #2)
- 이번 라벨(pump20=+20%)에서 동적 유니버스(top100 OR surge3x)는 static top100 대비
  recall 72.3%→72.7%(+0.4pp)뿐. 직전 워크플로 +7.9pp는 더 tight base(top50)거나
  ≥20% 정의 차이로 추정. 본 측정: top50 base에선 surge2.0 gate가 +4.8pp로 유효.
- 유니버스밖 펌프 27.3%(837건)는 대부분 신규상장/무이력(qv_rank=-1: 0G+488%,
  MOVE+815%, VTHO+748%, MINA+697%, ZK+957%) = cold-start 구조적 한계(cold-pump와 정합).
- 즉 추천 레이더의 recall 상한 ≈ 72%(top100), 나머지는 일봉 D-1 정보로 비예측.

## 오늘(2026-05-31) top 후보 (cal_xgb calibrated)
IN/VTHO/WLD/ZK/POKT/XLM/ID 모두 ~14.3%(상위 bin, 동률), 다음 tier NXPC/VET/BLAST ~7.3%.
※ 14.3%는 calibration 상한값(과신 가능) — 실제 다음날 적중은 이 bin 역사적 7.9% 수준으로 해석.

## evaluator 검증 요청
1. calibration 최상위 bin 과신(0.120 vs 0.079) — deflated calibration / isotonic 권고?
2. 동적 유니버스 recall +0.4pp(이번 라벨) — 직전 +7.9pp와의 차이 원인(라벨/base 정의) 감사.
3. 3 스코어러 동률 + best=cal_xgb 1회 선택 → selection deflate, XGB vs logit 차이 유의성.
4. 최근 14일 표본(n=14일)은 너무 작음 — OOS 765일 추정이 본체. forward 축적 권고.
5. precision@K는 hit-rate 지표 — 사용자 명시상 PnL은 사후지만, evaluator는 실현손익 경로
   모델링(TP-before-SL 함정) 별도 확인 권장.

## 산출물 경로
- output/recommendation_scorer_oos_metrics_v1.csv (스코어러×K×regime precision/recall/lift)
- output/recommendation_scorer_recent_picks_v1.csv (최근 14일 top-K + 실측 펌프)
- output/recommendation_scorer_today_v1.csv (오늘 top 후보 + 급등확률)
- output/recommendation_scorer_calibration_v1.csv (bucket calibration)
- output/rec_scorer_full_run.log (전체 로그)
