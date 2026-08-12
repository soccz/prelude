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

---

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
