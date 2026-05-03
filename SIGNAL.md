# SIGNAL.md — 시그널 생성 (라벨 / 피처 / 모델 / 추론)

> "**어떤 코인이 오늘 안정적으로 X% 이상 오를 확률**" 을 만드는 모든 책임. 출력은 코인별 확률 + 신뢰도. 포지션 사이징 / 자본 배분 / 알림 전송은 LEDGER / OPS 의 일.

---

## 0. 한 줄 결론

```
입력  : 어제까지 KRW 코인 일봉 + BTC 일봉 (multi-lookback 흐름)
출력  : { 코인, 오늘오를확률, 기대수익%, σ-tier, BTC regime, calibration CI }
관계자: SIGNAL → (확률 + 메타) → LEDGER → 텔레그램 알림 (OPS)
```

---

## 1. 데이터 (data/)

### 1.1 소스
- **업비트 KRW 일봉** (메인): `pyupbit` 라이브러리 사용
- **업비트 KRW 4h 봉** (보조): 일봉 안 장중 max drawdown 계산용
- **바이낸스 USDT 1h 봉** (보조): 김프 / binance lead-lag 피처용
- 가능하면 **historical 3 년치** 백필 (2023~ 현재). 알트는 상장일 기준 가능한 만큼

### 1.2 저장
- `data/upbit_d1.db` (sqlite) — 일봉 OHLCV + 거래대금
- `data/upbit_4h.db` — 4h 봉
- `data/binance_1h.db` — 바이낸스 1h
- 스키마 단순:
  ```sql
  CREATE TABLE candles (
    market TEXT,        -- 'KRW-BTC'
    timestamp DATETIME, -- KST 09:00 시작 봉이면 그 시각 (UTC 기준 저장)
    open REAL, high REAL, low REAL, close REAL,
    volume REAL,        -- 코인 단위
    quote_volume REAL,  -- KRW 단위 (24h 거래대금 계산용)
    PRIMARY KEY (market, timestamp)
  );
  ```

### 1.3 시간 처리 (KST 09:00 기준)
- 업비트 일봉 마감 = **UTC 00:00 = KST 09:00**
- DB 저장은 UTC 로 (혼동 방지)
- 추론 시 "오늘 일봉" = KST 09:00 시작 ~ 다음 KST 09:00 마감 봉
- 추론 시점 KST 08:30 = UTC 23:30 = 어제 일봉 100% 마감 후 30 분 (안전)

### 1.4 유니버스
- **업비트 KRW 24h 거래대금 top N** 으로 제한 (이하 = 저유동성 노이즈)
- **N = 100 은 초기값** (CLAUDE.md §2.5). top 50 / 200 도 백테스트 비교 대상. EDA 에서 거래대금 분포 / IC 안정성 보고 결정
- 유니버스 선정은 **fold train 종료 시점 (또는 추론 시점) 기준** 스냅샷 ← 이건 양보 X (look-ahead 위생)
- 상폐 코인도 학습 데이터에는 남김 (survivorship bias 방어, 양보 X)

---

## 2. 라벨 (signals/labels.py)

### 2.1 정의 (단순함의 끝)

```python
def today_pump_label(open_today, low_today, close_today, X=0.08, Y=0.03):
    """
    오늘 KST 09:00 시작 일봉 기준:
      조건 1: close / open - 1 >= X       (보수적으로 올랐다)
      조건 2: low / open - 1 >= -Y        (장중에 안 무너졌다)
    둘 다 만족하면 1, 아니면 0
    """
    return int(
        (close_today / open_today - 1 >= X)
        and (low_today / open_today - 1 >= -Y)
    )
```

- `X` = 종가 상승 임계 (0.08 = 8%)
- `Y` = 장중 낙폭 한도 (0.03 = 3%)
- 5~10% 는 일상이라 `X=8%` 부터 시작 (보수)
- "안 떨어진다" = 장중 저가가 시가 대비 -3% 이내

### 2.2 X / Y placeholder, 첫 주 EDA 후 결정
초기 (8%, 3%) 는 placeholder. 첫 주 데이터 본 후:
- 라벨 비율이 너무 흔하면 (예: ≥ 30%) → X 올리기
- 너무 희소하면 (< 1%) → X 내리기
- 보통 5~15% 정도가 학습에 좋음

### 2.3 라벨 sweep (선택적)

```bash
python scripts/label_sweep.py --x-grid 0.05,0.08,0.10,0.15 --y-grid 0.02,0.03,0.05
```

각 (X, Y) 조합으로 라벨 만들고:
- 라벨 비율 분포
- 백테스트 net Sharpe / hit rate
- 로 보고 사용자 결정

**경고**: 학술 표준 (structural_filter 등) 사후 검증으로 보고만, sweep 결과 채택은 net PnL 기준. (CLAUDE.md §2.3)

---

## 3. 피처 (signals/features.py)

### 3.1 알트 multi-lookback 피처
각 코인의 어제까지 데이터 기준.

**lookback 격자 {3, 5, 7, 14, 21} 일 은 초기값** (CLAUDE.md §2.5). EDA 에서 SHAP 기여도 보고 조정:
- 기여도 낮은 lookback 제거 (예: 21d 가 항상 무시되면 빼기)
- 더 긴 (30/60d) / 더 짧은 (1/2d) lookback 추가 검토
- 30 일 이상은 데이터 길이 / 코인 상장 기간 제약 확인 후

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
  volume_spike_score     (= ROC 의 표준화)
  volume_breadth_d       (시장 전체 거래량 활성화)

[기술지표]
  rsi_14, macd, macdhist, adx, bb_position
  squeeze_on            (볼린저 밴드 압축 플래그)
  roc_3d, roc_7d

[크로스섹션]
  rank_return_5d        (오늘 시점 시장 내 5d 수익률 순위 백분위)
  breadth_ratio         (양수 5d 수익률 코인 비율)
  top_n_return_5d       (상위 5 개 평균 5d 수익률 = "지금 뭐가 펌핑")
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

**MA200 / RV30 / intensity cutoff 0.5 는 초기값** (AETHER paper_idea_ko §2.2 차용, 학술적 사전 등록 패턴). 데이터 보고 조정:
- MA {120, 200, 252} × RV {20, 30, 60} 격자 백테스트 비교
- intensity cutoff 도 {0.3, 0.5, 0.7} 비교
- 더 좋은 net Sharpe 조합 발견 시 변경 (학술 사전 등록 무시 OK — CLAUDE.md §2.5)
```

### 3.3 cross-market 피처 (보조)

```
kimchi_premium       (업비트 KRW vs 바이낸스 USDT 환율 보정 후 가격 차)
kimchi_zscore_7d     (김프의 7 일 zscore)
binance_lead_1h      (바이낸스 1h 수익률 — 글로벌 선행)
```

### 3.4 정규화 원칙
- **per-day cross-sectional rank normalization**: 코인별 피처에만 적용 (return, vol, range_contraction 등). 같은 날 100 개 코인 사이에서 0~1 순위
- **rolling z-score (시간축)**: BTC regime 피처 (전 코인 공통값이라 cross-sectional 의미 X)
- **ffill only**: 결측은 forward fill, **`fillna(0)` 절대 X** (gan_t 의 known gap #2 — RSI=0 같은 불가능 값 만듦)

---

## 4. 모델 (signals/models/)

### 4.1 Phase 1 — XGBoost concat baseline

**왜 XGBoost?**:
- fin paper 검증: concat_a (단순 입력 결합) > FiLM / learned PE / tau-RoPE
- 빠른 학습 / 빠른 추론 / SHAP 해석 가능
- 일봉 데이터셋 작은 편 (코인당 수백~천 행) — 딥러닝 오버킬

**구조**:
- 입력: 위 §3 피처 다 concat (per-day cross-section 안에서 한 코인)
- 타겟: §2 의 binary label (오늘 펌프 / 아님)
- objective: `binary:logistic`
- sample weight: balanced (positive label 희소)
- Optuna 튜닝 (max 50 trial, IC + binary cross-entropy)

**파일**:
- `signals/models/xgb_phase1.py` — 학습 / 추론 클래스
- `signals/models/ckpt/phase1_<date>.json` — 모델 가중치
- `signals/models/configs/phase1.json` — 하이퍼파라미터

### 4.2 Phase 2 — Hybrid (Transformer + TCN + FiLM + CVAE)

ASSETS.md 의 gan_t/models/hybrid_model.py 구조를 today_pump 안에 새로 (self-contained):

```
[입력 (B, T=21일, F)]
  ↓
[Transformer Encoder]   ← 글로벌 / 장기 의존성
  ↓
[Attention-guided TCN]  ← 로컬 / 모멘티프
  ↓
[Gated Fusion]          ← 진단 변수 (gate=Trend or Pattern 지배)
  ↓
[FiLM Regime Conditioning]  ← BTC 4-state
  ↓
[CVAE Decoder]          ← 예측 분포 (epistemic + aleatoric 분리)
  ↓
[출력 (확률 + 분포)]
```

**언제 도입?**:
- Phase 1 의 net Sharpe / hit rate 가 충분히 높으면 → Phase 2 유보 (단순 모델 유지)
- Phase 1 결과 부족하면 Phase 2 도입 + DM test 로 의미 있는 개선 입증

**파일**: `signals/models/hybrid_phase2.py`

### 4.3 Phase 3 — APF motif 진단 (옵션, 학술 트랙)

ASSETS.md 의 fin/Attention Pattern Fields 코드를 참조해서 today_pump 안에 새로:

- 매일 추론 시 attention map 의 motif (stripe / block / spike / diagonal) 분류
- "이 추천은 spike motif 기반" 식의 진단을 텔레그램 알림에 추가
- Phase 2 hybrid 모델에 부착

**언제?**: Phase 1 / 2 안정 후 + 사용자가 "왜 추천했는지" 더 알고 싶을 때

---

## 5. Calibration (signals/calibration.py)

### 5.1 σ-bucket calibration
xsec_alpha 의 magnitude.py 패턴을 today_pump 안에 새로.

학습 / holdout 의 모든 예측을 모은 뒤:

```python
sigma = score / batch_std
# bucket: sigma 가 |0.5|, |1.0|, |1.5|, |2.0|, |2.5|, |3.0| 구간
# 각 bucket 별로:
#   - hit_rate (실제 라벨=1 비율)
#   - mean_signed_return_pct (sign(score) × realized return 평균)
#   - mean_abs_return_pct
#   - std

# 추론 시 새 코인의 sigma 계산 → 해당 bucket 의 hit_rate 가 그 코인의 calibrated 확률
```

저장: `output/calibration_sigma.json`

### 5.2 σ-tier 출력
calibration 기반 4-tier:
- 🔥 (가장 강함)
- ✅ (강함)
- ▫ (중간)
- · (무시)

**tier cutoff (sigma 2.0 / 1.5 / 1.0) 는 초기값** (CLAUDE.md §2.5). 실제 calibration 결과 hit_rate 기준으로 조정:
- 🔥 = hit_rate ≥ 0.70 인 sigma 구간 (예: 1.8 이상이면)
- ✅ = hit_rate ≥ 0.55 인 구간
- ▫ = hit_rate ≥ 0.50 인 구간
- · = hit_rate < 0.50 → 알림 X
- 즉 cutoff 는 sigma 절대값이 아니라 **calibration bucket 의 hit_rate 가 결정**

### 5.3 Per-coin CI
코인별 실현 변동성 × 1.96 으로 95% CI 생성. tier 가 같아도 코인마다 위험폭이 달라야.

---

## 6. 추론 (signals/predict.py)

```python
def predict_today():
    """매일 KST 08:30 cron 호출"""
    universe = get_top100_by_quote_volume(asof=now_kst())
    btc_features = compute_btc_regime_features(asof=yesterday_d1())

    predictions = []
    for coin in universe:
        alt_features = compute_alt_features(coin, asof=yesterday_d1())
        features = concat(alt_features, btc_features)
        score = model.predict_proba(features)
        sigma = (score - calibration.batch_mean) / calibration.batch_std
        tier = calibration.tier(sigma)
        ci = compute_ci(coin, score)
        predictions.append({
            'coin': coin,
            'score': score,
            'sigma': sigma,
            'tier': tier,
            'expected_pct': calibration.mean_signed_return(sigma),
            'ci_low': ci[0], 'ci_high': ci[1],
            'btc_regime': btc_features['regime'],
        })

    return sorted(predictions, key=lambda x: -x['score'])
```

출력 → `output/predictions_YYYYMMDD.csv`. LEDGER + OPS 가 받음.

---

## 7. 검증 (signals/validate.py + scripts/backtest_wf.py)

### 7.1 Purged Walk-Forward (데이터 누수 방어)
- 5-fold WF
- embargo: 10 일 (h=1, but 라벨 N-day 안정성 조건 위해 보수)
- 최종 holdout: 마지막 6 개월 절대 락 (HPO / ablation 중 열람 X)

### 7.2 지표 (트레이딩 결과 우선)

**필수 (트레이딩)**:
- net Sharpe (왕복 수수료 0.15% 차감)
- Max Drawdown
- hit rate (라벨 = 1 적중률)
- 누적 PnL (가상 ledger 기반)
- turnover (일일 회전률)

**진단 (학술)**:
- IC (Spearman, 일별 cross-sectional)
- ICIR
- CRPS
- PI_80 coverage

**우선순위**: 필수 결과로 결정. 진단은 옆에 표기만.

### 7.3 forward 검증
백테스트만 보고 결정 X. 최소 **2 주 live paper trading** (Mode B) 후 net 결과 확인.

---

## 8. 재학습 (signals/retrain.py)

### 8.1 cadence
- 주 1 회 (일요일 KST 06:00) — `ops/retrain_pipeline.py`
- 새 모델 후보 학습 → 신/구 holdout IC + net Sharpe 비교

### 8.2 promotion gate
신 모델 채택 조건 (둘 다 통과):
- new net Sharpe ≥ old net Sharpe - **delta_sharpe** (degradation 허용)
- new hit rate ≥ old hit rate - **delta_hit**

**초기값**: `delta_sharpe = 0.1`, `delta_hit = 0.02` (CLAUDE.md §2.5). Phase 1 라이브 4 주 후 retrain_history 보고 조정:
- old 모델 자체가 노이즈 큰 시기였으면 delta 크게 (관용적)
- old 모델 안정이면 delta 작게 (엄격)

학술적 IC 게이트 (xsec_alpha 처럼 IC ≥ 0.04) 는 **사후 진단으로만** — 채택 결정에 안 씀 (CLAUDE.md §2.3).

### 8.3 실패 시
- 후보 폐기, 이전 모델 유지
- `output/retrain_history.json` 에 기록
- 3 회 연속 실패 → 텔레그램 경고 + 사용자 컨펌 후 hard reset

---

## 9. drift 감지 (ops/drift_detector.py)
OPS.md 참조. 시그널 품질 감시.

---

## 10. 책임 경계 (다른 MD 와의 분리)

| 이건 SIGNAL 의 일 | 이건 SIGNAL 의 일 X |
|---|---|
| 라벨 정의 | 가상 자본 배분 → LEDGER |
| 피처 계산 | 알림 전송 → OPS |
| 모델 학습 / 추론 | cron 스케줄링 → OPS |
| Calibration | 텔레그램 포맷 → OPS |
| 백테스트 metric | 가상 포지션 진입 / 청산 → LEDGER |
| 재학습 promotion gate | 실거래 → 사용자 직접 (시스템 X) |

---

## 11. 핵심 파일 인덱스

| 파일 | 역할 |
|---|---|
| `data/collector_d1.py` | 업비트 일봉 수집 |
| `data/collector_4h.py` | 4h 보조 |
| `data/collector_binance.py` | 바이낸스 1h |
| `signals/labels.py` | today_pump_label |
| `signals/features.py` | alt + BTC + cross-market 피처 |
| `signals/models/xgb_phase1.py` | Phase 1 모델 |
| `signals/models/hybrid_phase2.py` | Phase 2 (나중) |
| `signals/calibration.py` | σ-bucket |
| `signals/predict.py` | 일일 추론 |
| `signals/retrain.py` | 주간 재학습 + promotion |
| `signals/validate.py` | Purged WF |
| `scripts/backtest_wf.py` | 전체 백테스트 |
| `scripts/label_sweep.py` | 라벨 X/Y sweep |
| `scripts/predict_today.py` | 수동 추론 (dry-run) |
