# 연구 노트 B — Downside head + risk-reward 랭킹 (component=downside_rank)

## 가설
"오를 확률이 높은 코인" 만으로 고르면 그 코인은 보통 변동성·과열이 커서 **하방도 같이 크다**
(upside-only top-K 의 P(다음날 min≤-5%)=0.53). D-1 선행 feature 로 **상승 분포 + 하락 리스크**를
임계별 calibrated 확률로 같이 추정하면, cross-section 에서 "상승확률 유지 + 하락확률 낮은" 코인을
가려낼 수 있다. = 사용자 refined 비전 (하락 최소화 우선, ~-5% 손실 수용).

## 무엇을 돌렸나
`scripts/downside_head_riskreward_v1.py` (self-contained, prelude 내부 모듈만 재사용:
build_market_features / add_cross_sectional / attach_btc_regime / PRECURSOR_FEATURES).
- 패널: 239→ 유니버스(D-1 qv top100 OR qv_surge_7d≥3) 76,731 OOS rows, 2024-04~2026-05.
- 라벨 (day-D open 진입 기준, 분포로 열어둠): UP {+5,+10,+15,+20%} = high/open, DOWN
  {-3,-5,-10%} = min(low)/open, close<0 = close/open. 8 head.
- 모델: head 별 XGBoost (depth4, scale_pos_weight) + **임계별 per-fold train-OOF
  bucket(10) calibration** → raw prob 출력 X, calibrated bucket hist-hit 만.
- expanding purged WF 6-fold, embargo 5d.
- 기대/중앙 하방: train 의 raw dn_05 score → down_low_ret 조건부 bucket 평균/중앙 (leak-free 회귀 head).
- risk-reward 랭킹 3종 vs upside-only baseline 비교 (K=3,5):
    R1 ratio = P(+10)/P(-5),  R2 penalized = P(+10) − λ·P(-5) (λ∈{0.5,1,2,3}),
    R3 gate = P(-5) 상위분위 제외 후 P(+10) 정렬.
- 오늘(2026-05-31) scan: train(< asof-5d) fit → 유니버스 100코인 추론.

## leak·시간정합성 방어 (§1 체크리스트 통과)
- feature 24개 전부 `f_` prefix (build_market_features 의 .shift(1) = D-1). 코드로 검증:
  LEAK_COLS/next_*/lab_*/up_high_ret/down_low_ret/eod_ret 가 feature 행렬에 **0개**.
- 라벨 = day-D open 대비 high/low/close (미래 타겟) — 학습 feature 와 시점분리.
- 모델·calibration·기대하방 회귀 = expanding train(과거 fold)에서만 fit, test fold 적용만.
  calibration 은 train OOF bucket (val/test quantile 사용 X). 8개 임계 동일 파이프.
- 오늘 scan = train < asof-embargo, D-1 feature 만 추론 (라벨/미래봉 안 봄).
- 거래비용 0.15% 왕복 차감 (실현 net 지표). day-equal-weight 집계.
- **"너무 좋으면 leak 의심" 자가검증 통과**: OOS top-decile lift = up10 2.67x, up20 3.55x,
  dn05 2.12x, dn10 3.63x — leak 이면 나올 ~10x/near-100% 가 아님. reliability 8개 head 전부
  monotonic, 극단부 보수적(pred≥actual). → leak 신호 없음.

## 시도 조합 수 (selection deflate 용)
- head 8개 (사용자 지정 임계, hand-pick 아님 — {-3,-5,-10}/close + {+5,+10,+15,+20} 격자).
- risk-reward 조합 = 1(R1) + 4(R2 λ) + 3(R3 qcut) = 8개, K 2개 = **16 셀**. baseline 2.
- 모델 hyperparam 1세트 (sweep 안 함). feature set = 기존 검증 24개 재사용 (새 search X).
- best 결합(R2 λ=1, K=3) 은 16셀 중 데이터로 선택 → quant-evaluator 가 deflate 필요.

## 1차 결과 (net, OOS)
임계별 OOS base rate (유니버스): P+5=.232 P+10=.076 P+15=.035 **P+20=.018** |
P-3=.474 P-5=.253 P-10=.051 Pclose<0=.535.  → "+20% 다음날" 은 1.8% (정직: rare).

risk-reward 가 하방을 실제로 낮추나 (K=3, upside-only → R2 λ=1):
- p(min≤-5%)  0.532 → **0.156**     (deep 회피)
- p(min≤-10%) 0.150 → **0.023**
- deep_dump   0.280 → **0.072**
- mean min_ret -0.062 → **-0.028**,  CVaR95 -0.166 → -0.078
- 실현 pump10  0.205 → 0.071        (상승확률은 ~1/3 로 양보됨 — tradeoff)
- EOD net cum  -1.000 → **+1.095**, Sortino -4.57 → **+1.61**  (16셀 중 유일하게 net 양수)
R1 ratio 는 상승 유지가 더 좋음(pump10 0.136, p(min≤-5%)=0.333) — λ=1 보다 덜 공격적 절충.

오늘(05-31, BTC bear_quiet) 대비:
- upside-only top: POKT/INJ/VTHO — P+10=.279 **이지만 P-5=.670, E[min]=-.076** (고위험).
- R1/R2 top: **ZK, NXPC** — P+10=.279 유지하면서 P-5≈.281, P-10≈.02~.03, E[min]≈-.039.
  → 같은 상승확률에 하방 절반. risk-reward 가 surface 하려던 바로 그 코인.
- 더 안전하게 가면 PEPE/BTC/SHIB (P-5=.025, E[min]=-.016) 로 수렴 (상승도 같이 낮아짐).

## 산출물
- `output/downside_head_today_v1.csv` — 오늘 100코인 × 8 calibrated prob + E/med 하방 + rr 랭크.
- `output/downside_head_riskreward_compare_v1.csv` — 16셀 vs baseline 하방/상승 비교 (K=3,5).
- `output/downside_head_reliability_v1.csv` — 8 head × 5bin OOS pred vs actual.
- `output/downside_head_oos_picks_v1.csv` — OOS row-level (감사용).
- `output/downside_head_baserates_v1.json` — 임계별 base rate.

## quant-evaluator 검증 요청 (의심 지점)
1. **selection bias**: R2 λ=1, K=3 이 16셀 중 net 양수 유일 → cherry-pick 위험. λ·K·임계 동시
   deflate (PSR/DSR or bootstrap CI). OOS cum +1.095 가 특정 fold(2025 bull) 집중인지 fold-stability 확인.
2. **EOD net ≠ ledger**: 이 비교는 top-K **무필터 long EOD** (SL/TP 없음) 의 분포 대조용이지
   tradeable 전략 아님. 사용자 -5% 하드SL 정책 하 실현손익은 downside_aware_recommender_v1(15m 경로)
   과 합쳐 재평가 필요. hit≠Sharpe 함정.
3. **lab_close_neg head 약함**: top bin pred .71 vs actual .59, bot .34 vs .51 — 50% 로 압축(near-coinflip).
   이 head 는 보조표시만, 랭킹에 쓰지 말 것 권고.
4. **today scan n_fold-less**: 오늘 추론은 단일 train fit (WF 아님) → 오늘 숫자 자체는 OOS 검증 아님.
   OOS 신뢰는 compare/reliability 표가 근거.
5. dn05 조건부 기대하방 회귀가 bucket 평균이라 coarse(10단계). 분위회귀로 정밀화 가능(후속).
