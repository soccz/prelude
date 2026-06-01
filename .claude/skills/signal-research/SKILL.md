---
name: signal-research
description: "prelude 시그널/모델 연구 절차. 라벨 설계, 피처 엔지니어링, 모델 학습(XGBoost detector / distribution 7-head / setup library), purged walk-forward 백테스트, sweep, label space discovery, setup discovery 를 수행할 때 사용. look-ahead leak 방어 체크리스트와 이 프로젝트가 데여본 함정(same-day leak 2번, EDA-hit≠Sharpe, rare-event 과신, 사용자직관 반증)을 강제한다. 모델/라벨/피처/백테스트/sweep 관련 작업이면 — 캐주얼하게 '이 코인 신호 찾아줘' 라고만 해도 — 이 스킬을 따른다. 후속: 모델 재학습/피처 추가/sweep 재실행/이전 백테스트 개선 요청 시에도 사용. 세션 첫 트리거면 prelude-quant 오케스트레이터가 세션 의례를 먼저 돌린 뒤 이 스킬로 라우팅하는 것이 기본."
---

# signal-research — leak 없이 엣지를 찾는 절차

prelude 의 시그널 연구는 **"오를 코인 찍기"가 아니라 과거에 반복되는 상승 전 조건을 leak 없이 찾아 확률+위험으로 보여주기**다. 목표는 100% 정확도가 아니라 재현 가능한 엣지 + (가능하면) 설명 가능한 가설.

## 0. 시작 전 — 무엇을 만드나
- **detector_v1**: rare +20% tail binary radar (silence-heavy). artifact 고정(`detector_v1.json`, threshold 0.8815).
- **distribution_engine_v1**: 7-head 확률(T×C×DD×H 라벨 공간) + setup library + bucket calibration.
- 새 연구: 위 두 개 개선(예: 4h feature 로 sustain head 살리기, range_contraction v2, cross-head intersection)이거나 새 라벨/모델.
- **legacy 6-class 분포 모델**: 현재 운영 X (보존). entry = `signals/predict.py` (+ `scripts/predict_today_legacy.py`). 재가동 요청 시 — same-day leak 2번 전적상 §1 leak 체크리스트(특히 distribution day-shift join, train OOF quantile) 재검증 필수, 채택은 signal-researcher → quant-evaluator(ADOPT/SHADOW/REJECT) 파이프로.

## 1. Leak 방어 체크리스트 (1번 규칙 — 어기면 전부 무효)

이 프로젝트는 **same-day leak 을 2번 겪었다**(detector, distribution engine). 둘 다 "성능이 너무 좋아서" 의심하다 발견. 학습/백테스트 전 반드시:

- [ ] **시점 분리**: 입력 피처는 t-1(어제)까지만, 타겟은 t(오늘) 이후. panel row at t 의 피처에 same-day high/low/close 가 섞이지 않았는지 확인.
- [ ] **shift(-1)**: market 별로 `next_open/high/low/close = g[...].shift(-1)`, 라벨은 next_* 로 계산.
- [ ] **LEAK_COLS 제외** (학습 feature 에서): `net_under_tp, max_return, label, label_tail, next_open, next_high, next_low, next_close, next_max_return, next_eod_return, next_max_dd` + `next_*` prefix 전체.
- [ ] **distribution day-shift join**: `label_df["date_only"] = (label_date - 1day).dt.date` 로 feature_date 와 매핑. (이게 틀려서 prec 100%→60% leak 이었다.)
- [ ] **threshold 는 train OOF quantile** — val quantile 사용 X(v1 leak), train direct quantile 도 X(v2 overfit zero-trade). outer fold train 안에서 inner CV 로 OOF score → 그 quantile. 운영에선 라이브 재계산 금지, artifact 고정값.
- [ ] **"성능이 너무 좋으면 leak 의심"** — prec 98%+, accuracy 67% 같은 건 거의 항상 leak. 채택 신청 전 자가검증.

## 2. 검증 방법 (양보 X 위생)
- **Purged Walk-Forward + embargo** — 데이터 누수 방어. fold train 종료 시점 기준으로 유니버스 시간정합성 유지.
- **거래비용 차감** — 백테스트는 0.15% 왕복(수수료 0.1% + 슬리피지 0.05%) 차감 후 net 으로 본다.
- **survivorship bias 방어** — 상폐 코인도 데이터 포함.

## 3. 이 프로젝트가 데여본 함정 (반복 금지)

| 함정 | 교훈 | 대응 |
|------|------|------|
| same-day leak (2번) | 성능 좋으면 leak 의심 | §1 체크리스트 |
| EDA hit rate ≠ ledger Sharpe | momentum hit 21% but Sharpe -5.2 (TP-before-SL) | hit rate 만으로 채택 신청 X. 실현손익 경로 모델링 → evaluator 에 넘김 |
| rare event raw probability 과신 | +20% tail 90% pred vs 11.6% actual | bucket-based hist hit 으로 표시 |
| 사용자 직관 미검증 | range_contraction·low-cap pump·bear silence 부분/완전 반증 | fixed prior 위에 쌓지 말고 데이터로 검증. 반증도 발견으로 기록 |
| date convention 혼동 | entry_date vs reference_date 혼동으로 MINA 오평가 | 코드 layer 마다 date 의미 명시 |
| selection bias | 288→4608 sweep, 5 head hand-pick | train-only discovery, OOF threshold, hand-pick after data. **시도 조합 수 기록** |

## 4. 발견된 robust setup (참고 — interpretable rule library)
- **ATR(변동성)이 universal first-split** — 모든 head/fold 의 root. "변동성 높은 코인=펌프 후보"가 단일 최강 feature.
- Momentum continuation > range contraction (일봉 horizon). range_contraction 은 4h/1h scale 가능성.
- S01 high-vol momentum / S02 strong yesterday / S03 vol expansion 5d (lift 표는 RESEARCH.md §7).
- top50 대형 코인이 base rate 1.5~3× 높음(사용자 low-cap 가설 반증).

## 5. self-contained 원칙
gan_t / xsec_alpha / fin / APF 에서 **import 금지**. 비슷한 코드 발견 시 읽고 이해 → prelude 안에 새로 작성 → 출처는 ASSETS.md 에 기록.

## 6. 출력 — 연구 노트
작업 끝에 연구 노트 1건을 남긴다(quant-evaluator 의 입력):
```
## 연구 노트: {모델/실험명}
- 가설: {왜 이게 오를 신호인가 한 줄}
- 무엇을 돌렸나: {라벨/피처/모델/검증 방식}
- leak·시간정합성 방어: {§1 체크리스트 어떻게 통과했는지}
- 시도 조합 수: {sweep N개 / hand-pick 여부} ← selection deflate 용
- 1차 결과: {hit/lift/EV, net 여부}
- evaluator 검증 요청: {특히 의심되는 지점 — leak 위치, n 작은 곳 등}
```

## 7. 사용자 컨펌 / 자유 경계
- **컨펌 필요**: 라벨 정의(X/Y/N) 변경, 모델 architecture 변경, 새 모델 배포.
- **Claude 자유**: 새 피처 추가(기여 검증 후 commit), 백테스트/sweep/discovery 스크립트 작성, notebooks 실험.

## 8. 재현 파이프라인 (참고)
```bash
python scripts/build_detector_v1.py            # detector artifact
python scripts/build_distribution_engine_v1.py # 7-head
python scripts/setup_discovery_v1.py           # interpretable rules
python scripts/backfill_paper_ledger.py --top-k 10 --universe top100
python scripts/calibration_paper_ledger.py --paper-ledger output/paper_ledger_backfill.csv
```
sweep/discovery 스크립트는 `scripts/*_sweep_v1.py`, `scripts/label_space_discovery_v*.py`, `scripts/fold_stability_v*.py` 참고.
