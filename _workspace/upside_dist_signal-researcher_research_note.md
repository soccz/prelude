# Research Note — Upside Distribution Head v1 (component=upside_dist)

asof 2026-05-31 · signal-researcher · 채택 신청 아님 (quant-evaluator 판정 대상)

## 가설 (한 줄)
검증된 leak-free 선행 rank-mean score 를 per-threshold bucket-calibration 하면, 코인×일별로
정직한(과신 없는) 상승 확률분포 P(다음날 high/open≥{3,5,7,10,15,20,30}%) + 하락리스크
P(low/open≤-{5,10}%) 를 산출할 수 있고, 이것이 기존 7-head engine 의 raw 확률보다 calibrated 다.

## 무엇을 돌렸나
- `scripts/build_upside_dist_head_v1.py` (신규). 검증된 8-feat rank-mean score 재사용(새 hand-pick 없음).
- 9 calibration head (UP 7 + DN 2) — purged WF 5-fold, embargo 5d, score-bucket(10) historical-hit.
- 산출: base_rates / calibration reliability / vs-engine Brier / today scan.

## leak·시간정합성 방어
- feature 전부 build_market_features 의 market 별 .shift(1) (D-1 까지). 라벨=day-D open/high/low(미래).
- calibration bucket map = 각 fold train(과거)에서만 적합 → test(미래) 적용 (OOF). embargo 6일 gap, train/test overlap 0 (검증 완료).
- 유니버스 = D-1 qv top100. today scan: asof row 의 라벨은 절대 소비 안 함(train cutoff=asof-5d).
- 거래비용은 이 단계 비적용(확률 산출만). ledger 경로 평가 시 0.15% 차감은 quant-evaluator.

## 시도 조합 수 (selection)
9 임계 head × 동일 score/feature/universe(재사용). 새 feature 탐색/weight tuning 없음 → selection bias 최소.

## 1차 결과
- base rate (universe top100): up3=40.0% up5=23.2% up7=14.5% up10=7.5% up15=3.5% up20=1.8% up30=0.7%.
  top-decile lift 1.42→3.45 (임계 클수록 lift↑ 하지만 절대 P 급감). "20% 고정"은 데이터상 top-decile에서도 5.8%뿐 → 분포로 열어두는 게 맞다.
- OOF reliability mean|pred-actual|: up20=0.40pp up15=0.77pp up10=1.40pp up7=2.48pp up5=3.68pp up3=5.23pp (전부 약한 under-pred=보수적).
- vs 7-head engine: engine top-bucket +20% pred 60.3% vs actual 11.6% (+48.7pp 과신, calibration_summary.json). rank-mean OOF Brier(0.0198) < engine in-sample Brier(0.0305) — engine 이 in-sample 유리한데도 더 나쁨.
- TODAY(D-1 regime=bear_quiet): top-score 코인 P_up_5=42.5% 인데 P_dn_5=48.1% — 선행 score 는 '변동성'을 사서 상·하방 동반 상승, 상방-only 분리 못함. bear_quiet 에선 다운사이드 우위.

## quant-evaluator 가 검증해야 할 지점
1. engine 비교의 engine 쪽은 final-model in-sample(optimistic) — OOF 재학습으로 공정 비교 확정 필요.
2. ds_adj_index(P_up5 - λ·P_dn10, λ=1 placeholder) 의 λ 를 ledger 하방경로(15m)로 검증.
3. up3/up5/dn5 의 under-pred 잔차(actual>pred 3~5pp) 가 fold 별 안정적인지(regime drift?).
4. EDA-hit≠Sharpe: P_up 높음이 net PnL 로 이어지나 — TP-before-SL 함정 재확인(이 head 는 확률만, 경로손익 아님).
