# RESEARCH.md — Crypto Distribution Engine 개발 narrative

> Personal trading radar 이자 quant research portfolio. 매일 KST 09:05 에 업비트 KRW 알트의 상승 setup 분포를 알리는 시스템.
> 이 문서는 **무엇을 만들었나** 가 아니라 **어떻게 발견하고 수정하며 진화시켰나** 의 기록.

**프로젝트**: `/home/soccz/22tb/prelude/` · 작업 기간: 2026-05-03 ~ (진행 중)

---

## 0. 한 줄 요약

```
"오를 코인 찍기" 가 아니라
과거 데이터에서 반복되는 상승 전 조건을 찾아서
오늘 그 조건에 가까운 코인을 확률 분포 + 위험 같이 보여주는
detect-and-defer 시스템 (자동매매 X, 사용자 본인 판단)
```

핵심 산출물:
- detector_v1: rare +20% tail radar (silence-heavy)
- distribution_engine_v1: 7-head probability + setup library + bucket-calibrated alert
- paper_ledger: 5,080 backfill rows + 자동 close-out + calibration loop

---

## 1. Initial design — leak 때문에 가짜 성능

### 1.1 1차 모델 (Phase 1 multi-class)
- 6-class softprob (next-day high/open ratio bucket)
- accuracy 67.4% — random 17%, 4× lift 같아 보였음

### 1.2 발견한 leak
panel row at day t 의 features 가 same-day high/low/close 를 포함.
label 은 `today_high / today_open` — **same-day**.
모델이 사실상 종가 보고 그날 high 추정. 가짜 성능.

### 1.3 수정
```python
# market 별 shift(-1)
panel["next_open"] = g["open"].shift(-1)
panel["next_high"] = g["high"].shift(-1)
panel["label"] = compute_label(panel["next_*"])
```
재검증: accuracy 67% → **30%** (random 1.83×). 작은 신호만 남음.

**Lesson**: pump 분류 task 에서 가장 쉬운 함정. 검증 시 features와 label의 시점 분리 명시 확인 필수.

---

## 2. Leak-free general classifier 는 약했다

leak fix 후 ledger backtest 모두 음수 Sharpe. EDA hit rate 와 ledger Sharpe 의 mismatch 발견:
- momentum hit 21% but Sharpe -5.2 (SL 46% 함정)
- → "맞히는 방향" ≠ "수익 나는 방향". TP-before-SL 게임.

7개 family × WF backtest 모두 fail. **task definition 자체가 잘못** 진단.

---

## 3. Target 재정의 — Detector + Distribution

### 3.1 "오를 코인" → "특정 패턴이 발화한 코인"
"평균을 맞히는 model" 폐기. 대신:
- **Detector**: rare event (+20% tail) 의 매우 강한 신호만 silent 하게 알림
- **Distribution**: 사용자가 mental model 로 가능한 라벨 공간 (`+3%/+5%/+10%/+20%` × `4h/24h` × `종가 유지/no`) 의 확률 분포 출력

사용자 결정 (취약): 단일 target 강요 X. **사용자가 시점 보고 본인 cutoff 결정**.

---

## 4. Detector v1 — silence-heavy rare radar

### 4.1 라벨
```python
label_tail = (next_day_max(high) >= 1.20 * next_day_open).astype(int)
# positive rate ≈ 1.89% over all KRW
```

### 4.2 검증 단계 (3 round)

| Round | 설계 | 결과 |
|---|---|---|
| v1 | regime-internal threshold quantile | leak (val quantile 사용) |
| v2 | train direct quantile (per fold) | overfit zero-trade (val 도달 X) |
| **v3** | **train OOF quantile (per fold inner CV)** | **C3 통과** |

**v2 발견**: XGBoost overfit 으로 train p99.95 = 0.982 인데 val p99.95 = 0.935. train threshold 가 val 에 거의 도달 못 함 (1/55,065).

**v3 fix**: outer fold 의 train 안에서 inner 3-fold CV 로 OOF score 생성, 그 quantile 을 threshold 로. realistic distribution 에서 cutoff 추출.

### 4.3 채택: C3 (bull_all p99.95 cap2)
- 3/4 active fold 양수 EV (+7.40%)
- 2024 EV -0.89% (resilient)
- no_trade 1/5
- rank-based fallback (cap만, no threshold) 모두 음수 → **silence-heavy 가 옳다는 증거**

### 4.4 운영 artifact
```
signals/models/ckpt/detector_v1.json     XGBoost binary (full panel)
output/detector_threshold.json            threshold 0.8815 (OOF p99.95) + regime + cap
```
운영 원칙: **threshold 라이브에서 quantile 재계산 금지**. artifact 고정값 사용.

---

## 5. Label Space Discovery — actionable space 가 어디인가

### 5.1 sweep
- T (target): {3, 5, 7, 10, 15, 20}%
- C (close hold): {0, 2, 3, 5}%
- DD (max pre-hit DD): {2, 3, 5}%
- H (time horizon): {4, 8, 12, 24}h
- 288 조합 × 4 regime × 3 liquidity = 4,608 평가

### 5.2 핵심 발견 (사용자 직관 반증 포함)

1. **종가 유지 (C ≥ 3%) 가 base rate 40% 깎음** — 윗꼬리 후 무너짐이 진짜 문제
2. **worst5 fail EOD ≈ -7% 가 자연 바닥** — 모든 combo 일관. -5% hard reject 는 데이터상 불가능
3. **hit_time 양극화** — 4h (즉발) 또는 24h (느린 펌프). 8/12h 거의 없음
4. **❗ liquidity 사용자 직관 반증** — "low-cap 이 펌프 끌어올린다" 가설 틀림. **top50 대형 코인이 base rate 1.5~3× 높음** (vol activation 효과). detector 도 universe top50 으로 좁히면 lift ↑
5. **❗ bull_quiet vs bull_volatile 거의 동일** — bull_volatile 가 살짝 base 높고 fail 도 덜 무너짐. "bull_volatile = nasty regime" 가설 부분 반증
6. **❗ bear_volatile 에서도 lift 양수** — base hit_h6 51.7%. detector 의 bear silence 정책 재검토 가치

→ data-discovered insights, fixed prior 위에 쌓이지 않음.

---

## 6. Distribution Engine v1 — 7 head + setup library

### 6.1 7 head (사용자 hand-picked, label discovery 결과 기반)
```
h1 즉발 안정       (T=3, C=2, DD=2, H=4)
h2 즉발 hit         (T=3, hit-only, H=4)
h3 종가 +3% 유지   (T=3, C=3, DD=3, H=24)   — 주의: 약함
h4 종가 +5% 유지   (T=5, C=3, DD=3, H=24)   — 주의: 약함
h5 +20% rare tail  (T=20, hit-only, H=24)   — = detector_v1
h6 +5% hit          (T=5, hit-only, H=24)
h7 +7% mid-strong  (T=7, C=0, DD=3, H=24)
```

### 6.2 또 leak — 같은 함정 두 번
첫 학습: h6 prec 100%, h3 prec 98%, h5 prec 100%. **너무 좋아 보임 → 의심**.
원인: panel row at t 의 features (= t일 daily close 포함) → label (= t일 4h bars). same-day.
동일 leak fix 패턴 적용:
```python
# label_df 의 label_date 를 feature_date - 1day 로 매핑하여 join
label_df["date_only"] = (pd.to_datetime(label_df["label_date"]) - pd.Timedelta(days=1)).dt.date
```
재학습 후: prec 100% → 60% (h6), 98% → 16% (h3). 진실한 lift 확인.

### 6.3 leak-fix 후 진짜 결과 (sweep, head × regime × universe × topK)

| Head | base | best lift @ topK | comment |
|---|---|---|---|
| h2 즉발 +3% hit | 16% | 3.43× @ top0.1% | **strong & frequent** |
| h5 +20% tail | 1.6% | **6.50× @ top0.5%** | rare, **top0.5 > top0.1** (극단부 noisy) |
| h6 +5% hit | 22% | 2.65× @ top0.1% | strong |
| h1 즉발 안정 | 6.5% | 2.45× | moderate |
| h7 +7% mid | 9.3% | 2.00× | borderline |
| h3 종가 +3% 유지 | 15% | 1.06× ❌ | **daily features 한계** |
| h4 종가 +5% 유지 | 12% | 1.89× | weak |

**Insight**: daily features 는 "어느 정도 위로 튈지" 는 잡지만 **"종가까지 유지될지" 는 못 잡음**. h3/h4 살리려면 4h/1h features 추가 필요.

**Insight**: detector_v1 의 p99.95 (top 0.05%) threshold 가 너무 빡셈. h5 lift 가 top0.1% 보다 top0.5% 에서 더 안정적.

---

## 7. Setup Discovery — interpretable rule library

shallow decision tree (depth=3, min_samples_leaf=500) per head, WF 5-fold leaf mining.

### 7.1 발견된 robust setups

| Setup | Rule | Heads | Past lift |
|---|---|---|---|
| **S01** high-vol momentum | ATR > 6.4% AND log_ret_1d > 4.4% AND roc_3d rank > 0.93 | h2, h6 | h2 4.09×, h6 2.63× |
| **S02** strong yesterday | log_return_1d > 4.8% (depth=1) | h5 | 4.07× |
| **S03** vol expansion 5d | vol_5d rank > 0.83 AND return_7d rank > 0.91 AND log_ret_1d > 5.9% | h5 | 6.81× |
| S04 BTC bull context | regime ∈ {bull_quiet, bull_volatile} | — (context) | — |

### 7.2 ❗ 사용자 가설 반증 발견

**ATR (변동성) 이 universal first-split** — 모든 head, 모든 fold 의 root. "변동성 높은 코인 = 펌프 후보" 가 단일 가장 강한 feature.

**Momentum continuation > Range contraction** — 사용자/xsec_alpha 의 핵심 가설 (조용해진 후 폭발 = `range_contraction`) 이 shallow tree 에서 거의 안 나옴. 일봉 horizon 에서는 momentum continuation 이 dominant. range_contraction 은 4h/1h scale 에서 더 잘 작동할 가능성 (Phase X+1 research 동기).

### 7.3 train vs val gap (overfit check)
모든 best leaf 에서 train-val gap < 10pp. shallow tree + min_samples_leaf 효과. h5 의 depth=1 leaf "log_ret_1d > 0.048" 은 train 6.78% / val 6.53% — 거의 perfect generalization.

---

## 8. Calibration overconfidence 발견 → bucket calibration 으로 수정

### 8.1 첫 운영 알림에서 의심
`p_h5 = 89.9%` 출력. 사용자 즉시 경고: "rare-event 에서 90% 는 거의 항상 과신".

### 8.2 검증 (5,080 OOF backfill rows 누적 후)

| Head | top decile mean predicted | top decile actual | error |
|---|---|---|---|
| h2 (+3% in 4h) | 90.8% | **58.1%** | +32.7pp |
| h5 (+20% tail) | 60.3% | **11.6%** | **+48.7pp** |
| h6 (+5% in 24h) | 88.2% | **58.1%** | +30.2pp |

추가: lower decile 은 under-confident (h2 bucket 0: pred 19.9% / actual 31.1%). 전형 sigmoid overconfidence.

### 8.3 수정 — bucket-based historical hit
isotonic 도 가능했지만 **rare event 에서 isotonic 이 표본 적은 극단부에서 흔들림**. bucket 표시 채택:
```
Before:  +20% tail: 89.9%       (raw, misleading)
After:   +20% tail: 11.6% hist hit (decile 10/10, n=508, rare)
```

알림 정직성 확보. 사용자가 "11.6% = 8건 중 1건 정도" 라는 보수적 해석 가능.

---

## 9. Paper Ledger + Live Evaluation Loop

### 9.1 Schema (사용자 정의)
```
date, coin, setup_ids, btc_regime, btc_context,
p_h2_3pct_4h, p_h6_5pct_24h, p_h5_20pct_tail, ..., 
log_return_1d_pct, atr_pct_14_pct, return_5d_rank, ...,
composite_score, alert_rank,
next_max_return_pct, next_close_return_pct, hit_h2, hit_h5, hit_h6,
status, notes
```

### 9.2 두 채널
- `output/paper_ledger.csv` — operational (매일 새 alert)
- `output/paper_ledger_backfill.csv` — WF OOF 시뮬 5,080 rows (calibration 신뢰성 확보)

### 9.3 Close-out
매일 KST 09:30 cron 으로 `close_paper_ledger.py` 가 다음날 4h bars 로 실현 채움 + status="closed". calibration 갱신 가능.

### 9.4 Date convention bug 발견 (post-hoc)
초기에 close_paper_ledger 가 `target_date = entry_date + 1` 로 잘못 닫음. MINA 11-06 alert 가 11-07 데이터로 평가됨 (max +28.81%, 실제 11-06 +23.81%). 발견 후 수정 + 8 rows 재close.

**Lesson**: data flow 의 시간 의미 (entry_date = predicted day vs reference day) 를 코드 layer 마다 명시. timezone / date convention 은 bug 의 단골.

---

## 10. 4-axis project structure

| 축 | 산출물 |
|---|---|
| **Research** | label/setup discovery, sweep CSVs, calibration tables |
| **Production detector** | detector_v1 + distribution_engine_v1 + setup library |
| **Paper ledger** | live + backfill ledger, close-out automation |
| **Writeup** | 이 문서 (RESEARCH.md), MD 8개 (README/SIGNAL/...) |

---

## 11. 현재 위치 + 다음 트랙

### 11.1 Stage
- ✅ Stage 0: backtest (모든 검증 완료)
- 🟡 **Stage 1: cron dry-run** (telegram off, 라이브 분포 1~2주 관찰) ← 진행 중
- ⏳ Stage 2: telegram beta (사용자 컨펌 후)
- ⏳ Stage 3: NOTES 기반 threshold/tier 조정

### 11.2 Research backlog
1. **Sustain head 재시도** — 4h bar features 추가 → h3/h4 살릴지 검증
2. **Range contraction v2** — depth 4 + subgroup mining 으로 사용자 가설 진지하게 재검토
3. **Cross-head intersection** — multi-head 동시 high score coin 패턴
4. **bear_volatile reactivation 검증** — sweep 결과상 좋았음, dry-run 데이터 본 후 결정
5. **S03 priority weighting** — backfill 에서 S03 hit_h2 55.8% > S01 52.8% > S02 44.9%, composite weighting 반영 가치

### 11.3 시스템 한계 명시
- Daily features 만으로는 sustain (close hold) 예측 불가
- raw probability 자체는 calibration 무시 시 misleading
- Selection bias: 288 → 4608 sweep, 5 head hand-pick — out-of-sample 으로 1주 dry-run 검증 후 신뢰
- 자동매매 X. 사용자 본인 매매 판단. 시스템은 paper trading reference only.

---

## 12. Reproducibility

전체 파이프라인 재실행:
```bash
# 1. Detector v1 artifact rebuild
python scripts/build_detector_v1.py

# 2. Distribution engine v1 rebuild
python scripts/build_distribution_engine_v1.py

# 3. Setup discovery (interpretable rules)
python scripts/setup_discovery_v1.py

# 4. Backfill paper ledger (5,080 OOF rows)
python scripts/backfill_paper_ledger.py --top-k 10 --universe top100

# 5. Calibration tables
python scripts/calibration_paper_ledger.py --paper-ledger output/paper_ledger_backfill.csv

# 6. Daily run (Stage 1 dry-run, telegram off)
bash scripts/daily_run_distribution.sh

# 7. Next-day close-out
python scripts/close_paper_ledger.py
```

---

## 13. Lessons summary

1. **Same-day leak 는 가장 흔한 함정** — 두 번 발견 (detector, distribution engine). 시점 분리 명시 필수.
2. **EDA hit rate ≠ ledger Sharpe** — 방향 맞히기와 돈 벌기는 다름. TP-before-SL 게임 모델링 필요.
3. **rare event 에서 raw probability 는 misleading** — bucket-based hist hit 가 운영적으로 더 안전.
4. **사용자 직관도 검증 대상** — range_contraction (조용함→폭발), low-cap pump, bear_volatile silence 모두 데이터로 부분/완전 반증.
5. **Selection bias 방어** — train-only label discovery, OOF threshold, hand-pick after data, dry-run validation.
6. **Code convention 일관성** — date 의미 (entry vs predicted vs reference) 는 layer 마다 명시.
7. **Silence-heavy detector + verbose distribution engine 병렬 운영** — 두 system 이 다른 use case.
8. **Setup library 는 "왜" 를 보여줌** — black-box 확률 + interpretable rule 결합이 사용자 신뢰 + research narrative 양쪽 강함.

---

*문서 작성: 2026-05-04. 진행 중인 시스템의 snapshot.*
