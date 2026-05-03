# SIGNAL.md — 시그널 생성 (라벨 / 피처 / 모델 / 추론)

> "**오늘 일봉 (KST 09:00 시작) 의 장중 max(high) 가 시가 대비 어디까지 올라갈지의 확률 분포**" 를 만드는 모든 책임. 출력은 코인별 **multi-class 확률 분포** + 기대값 + CI. 포지션 사이징 / 익절 손절 / 알림 전송은 LEDGER / OPS 의 일.

---

## 0. 한 줄 결론

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
- **바이낸스 USDT 1h 봉** (보조): 김프 / binance lead-lag
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
- 추론 시점 KST 08:30 = 어제 일봉 100% 마감 후 30 분
- 바이낸스 (UTC) join 시 +9 시간 변환 1 줄

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

`scripts/label_distribution.py` 로 bin 분포 분석 → 사용자 컨펌 후 최종.

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

**파일**: `signals/models/hybrid_phase2.py`

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
**저장**: `output/reliability_curves.json` (각 cutoff 별)

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
    """매일 KST 08:30 cron"""
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

## 7. 검증 (signals/validate.py + scripts/backtest_wf.py)

### 7.1 Purged Walk-Forward
- 5-fold WF + 10일 embargo (보수)
- 최종 holdout: 마지막 6 개월 절대 락

### 7.2 지표 (트레이딩 + 정확도 둘 다)

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
백테스트만 보고 결정 X. 최소 **2 주 live paper** 후 net + 정확도 확인.

---

## 8. 재학습 (signals/retrain.py)

### 8.1 cadence
- 주 1 회 (일요일 KST 06:00) — `ops/retrain_pipeline.py`
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
| `data/database.py` | sqlite 헬퍼 (이미 작성) |
| `signals/labels.py` | today_pump_label (multi-class) |
| `signals/features.py` | alt + BTC + cross-market 피처 |
| `signals/models/xgb_phase1.py` | Phase 1 multi-class softprob |
| `signals/models/hybrid_phase2.py` | Phase 2 (나중) |
| `signals/calibration.py` | reliability / Brier / coverage |
| `signals/predict.py` | 일일 추론 (분포 출력) |
| `signals/retrain.py` | 주간 재학습 + promotion |
| `signals/validate.py` | Purged WF |
| `scripts/backtest_wf.py` | 전체 백테스트 |
| `scripts/label_distribution.py` | bin 분포 분석 (EDA) |
| `scripts/predict_today.py` | 수동 추론 (dry-run) |
