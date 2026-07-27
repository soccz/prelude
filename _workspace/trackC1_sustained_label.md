# 연구 노트: Track C1 — sustained-gain LABEL challenger vs R1

## 가설
R1 타겟 = 일중 고가 `high/open-1 ≥ +20%`(pump20) / 헤드 anchor = P(high/open ≥ +10%).
→ 스파이크 찍고 종가에 덤프하는 코인도 잡힘(high 만 보니까). 이전 7실험(R2 랭킹·exit·A1~A4·멀티데이)은
전부 **R1 의 같은 진입집합을 후처리**만 해서 net 음수 천장을 못 넘음(검증됨).

**C1 가설**: 타겟 라벨을 **종가 기준 실현이익**(close/open-1)이나 **익일 보유수익**으로 바꾸면
헤드가 "실제로 버는(종가까지 들고 가도 양수)" 코인을 고르도록 학습 → **다른 진입집합** → net 양수 가능.
핵심은 high(스파이크)가 아니라 close(실현)를 타겟으로 해서 TP-before-SL / pump-then-dump
함정을 학습 단계에서 직접 회피하려는 시도.

## 무엇을 돌렸나
- **새 라벨 4종(전부 day-D/D+1 타겟, 미래, 학습에만)**:
  - `sus3` = close/open-1 ≥ +0.03 (base rate 17.8%)
  - `sus5` = close/open-1 ≥ +0.05 (9.3%)
  - `sus_net` = close/open-1 ≥ +0.0065 (왕복비용 0.15%+여유 넘김, net 양수 지향; 39.1%)
  - `fwd1` = next_close/close-1 ≥ +0.03 (익일 close-to-close 보유수익; 17.6%)
- **스코어러/헤드/검증 = R1 인프라 그대로 재사용** (정당 비교): `build_market_features`(.shift(1) D-1),
  `add_cross_sectional`, `attach_btc_regime`, `walk_forward_heads`(per-fold OOF bucket calib,
  XGB `random_state=42`), `_feats`(PRECURSOR_FEATURES − LEAK_COLS). **타겟만 sustained 로 교체.**
- 유니버스 = 정적 top100 (D-1 `f_qv_rank`), purged WF 6-fold, embargo=5.
- 청산/net = **R1 과 동일** 15m SL-3%/TP+5%/EOD net 0.15% (`simulate_path`, r2_challenger 와 동일 경로).
- C1 픽 = 해당 라벨 확률 top-3 (+ 보조 변형 `P(sus)-p_dn5`). baseline = R1 `p_up10/max(p_dn5,eps)`,
  동일 OOS 행에서 정렬키만 교체.
- 산출: `output/cc_sustained_label_compare_v1.csv` / `_picks_v1.csv` / `_coverage_v1.json`,
  스크립트 `scripts/cc_sustained_label_v1.py`.

## leak·시간정합성 방어 (§1 체크리스트)
- **시점 분리**: feature 는 build_market_features 의 `.shift(1)` (D-1 까지). sustained 라벨은
  day-D(또는 D+1) close 기반 (타겟, 미래) — 학습 feature 에 안 섞임. `SUS_LEAK_COLS` 로 `_feats`
  단계 제외 재확인 + 런타임 **LEAK GUARD** assert(라벨/outcome 컬럼이 feats 에 0개 — 통과).
- **fwd1 shift 방향**: `next_close = close.shift(-1)` 는 **라벨에만** 적용, 마지막 행 NaN → dropna.
  shift(-1) 이 feature 로 들어가지 않음을 LEAK GUARD 가 확인.
- **calibration**: per-fold OOF bucket (train-only). test fold 적용만.
- **유니버스 시간정합성**: top100 = D-1 `f_qv_rank` (shift 된 quote volume). fold train 종료 기준 유지.
- **거래비용**: 0.15% 왕복 차감 후 net.
- **"너무 좋으면 leak" 자가알람**: best C1 net_mean > +3%/day 이면 경보 → `leak_self_alarm=false`.
  추가 정합성 증거: 본 스크립트의 R1 baseline net_mean = **-0.002843** 이 기존
  `output/r2_challenger_compare_v1.csv` R1 행(-0.002843, Sharpe -2.50)과 **byte-close 일치**
  → 파이프라인이 R1 비교 셋업을 충실 재현(leak 유입 없음의 강한 방증).

## 시도 조합 수 (selection deflate 용)
- 정렬키 **9개** = R1 baseline 1 + C1 라벨 4개 × {prob, rr 변형} 8. (hand-pick 없음; 라벨/임계는
  사전 정의, 데이터 보고 추가 cherry-pick 안 함.)
- 라벨 임계 placeholder(sus3/5, sus_net=비용+0.5%, fwd1=+3%)는 §2.5 데이터 기반 조정 대상.

## 1차 결과 (OOS 765일·net 0.15% 차감, 15m SL/TP/EOD)
| policy | net_mean | Sharpe | Sortino | MaxDD | hit | deep-loss(noSL) | %SL | prec_self | prec_pump20 |
|---|---|---|---|---|---|---|---|---|---|
| **R1_baseline** | **-0.00284** | -2.50 | -4.49 | -0.872 | 0.39 | 0.135 | 0.456 | — | 0.037 |
| C1_sus3 | -0.00329 | -2.80 | -5.12 | -0.899 | 0.39 | 0.157 | 0.487 | 0.201 | 0.032 |
| C1_sus3_rr | -0.00332 | -2.88 | -5.43 | -0.899 | 0.40 | 0.086 | 0.384 | 0.146 | 0.011 |
| C1_sus5 | -0.00350 | -3.02 | -5.44 | -0.914 | 0.38 | 0.157 | 0.497 | 0.119 | 0.039 |
| C1_sus5_rr | -0.00283 | -2.54 | -4.89 | -0.870 | 0.41 | 0.063 | 0.300 | 0.056 | 0.007 |
| **C1_sus_net** | **-0.00173** | **-1.44** | -2.74 | **-0.797** | 0.42 | 0.086 | 0.398 | 0.391 | 0.018 |
| C1_sus_net_rr | -0.00328 | -2.79 | -5.43 | -0.903 | 0.42 | 0.070 | 0.357 | 0.369 | 0.010 |
| C1_fwd1 | -0.00294 | -2.50 | -4.59 | -0.896 | 0.39 | 0.161 | 0.484 | 0.184 | 0.030 |
| C1_fwd1_rr | -0.00262 | -2.22 | -4.34 | -0.859 | 0.41 | 0.081 | 0.366 | 0.131 | 0.010 |

**★ net 양수 라벨: 없음.** 9개 정렬키 전부 net_mean ≤ 0, Sharpe 음수.
**R1 우위(net_mean > R1)는 3개**: `C1_sus5_rr`(-0.00283, R1과 사실상 동률), `C1_sus_net`(-0.00173),
`C1_fwd1_rr`(-0.00262). 그러나 전부 음수 → 흑자 전환 실패. **R1 천장이 라벨 교체로도 안 깨짐.**

**최선 후보 = `C1_sus_net`** (단, ADOPT 아님): net_mean -0.00173 (R1 대비 +0.00111 개선),
Sharpe -1.44(R1 -2.50보다 개선), MaxDD -0.797(R1 -0.872보다 얕음), deep-loss(noSL) 8.6%(R1 13.5%),
hit 0.42(R1 0.39). 즉 **하방·드로다운은 R1보다 일관되게 나음, 그러나 비용 차감 후 양수 미달.**

**구조적 진단 (왜 실패했나)**:
- C1 은 의도대로 **진짜 새 진입을 만듦** — `C1_sus_net` 픽의 **R1 과 겹침 30.4%, 새 진입 69.6%**
  (sus3 64.9%, sus5 59.5%, fwd1 68.1% new). 이전 7실험과 달리 R1 집합 후처리가 아님. 그럼에도 음수.
- `prec_self`(픽이 자기 close-기반 라벨을 실제 달성)가 sus_net 0.391 ≈ base rate 0.391 → **헤드가
  base rate 위로 못 끌어올림**. close 타겟은 high 타겟보다 신호가 약함(스파이크 변동성 feature
  ATR 의 예측력이 close 마감에는 덜 먹힘). 즉 "실현이익 코인"을 D-1 feature 로 고르는 엣지가
  이 feature set 으로는 약하다.
- 커버리지: 픽/일 = 2.00 (15m 경로 있는 행만; top100×765일 중 양 끝/상폐 일부 결손). R1 과 동일 기준.

## evaluator 가 볼 것 (3줄)
1. **R1 baseline 정합성**: 본 스크립트 R1 net_mean=-0.002843 가 기존 r2_challenger 행과 일치하는지
   (일치 → 비교 셋업 충실 재현·leak 없음의 방증). 불일치면 파이프 차이부터 의심.
2. **leak 재감사**: sus_net 의 prec_self≈base rate, leak_self_alarm=false, new-entry 69.6% →
   "성능이 너무 좋아서"가 아니라 **너무 평범해서** 의심 적음. 그래도 `next_close` shift(-1) 이
   라벨 전용인지, fwd1 마지막행 NaN 처리/dropna 가 fold 경계에서 미래누수 없는지 확인 요청.
3. **C1_sus_net 의 하방 우위(net 음수지만 MaxDD/-deep-loss/Sharpe 가 R1보다 일관 개선) 가
   SHADOW record-only 가치가 있나, 아니면 net 음수이므로 REJECT 인가** 판정. 사용자 선호(하방
   최소화>상승)상 net 미달이어도 "덜 깨지는" 후보의 의미를 deflate(9 정렬키 선택) 감안해 평가.

## 파일 경로
- 스크립트: `/mnt/20t/prelude/scripts/cc_sustained_label_v1.py`
- 비교표: `/mnt/20t/prelude/output/cc_sustained_label_compare_v1.csv`
- 픽 dump: `/mnt/20t/prelude/output/cc_sustained_label_picks_v1.csv`
- coverage/leak self-check: `/mnt/20t/prelude/output/cc_sustained_label_coverage_v1.json`
- baseline 대조: `/mnt/20t/prelude/output/r2_challenger_compare_v1.csv` (R1 행)
