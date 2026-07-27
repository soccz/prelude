# 연구 노트: A2 (regime-split ranking + regime abstain) 챌린저 vs R1 챔피언 — track B3

## 가설
프로젝트 발견 = **volatile regime 은 hit 최고·net 최악** (regime마다 진입 dynamics 가 다름).
→ BTC regime(bull_quiet/bull_volatile/bear_quiet/bear_volatile, D-1)별로 다르게 행동하면 net/하방
개선. 두 lever: (a) **regime-conditional 정렬키** (regime별 R1 ratio vs downside-penalized 중 선택),
(b) **regime abstain** (net 최악 regime 에서 추천 0 → 그 날 손실 회피). 사용자: 하락최소화 > 상승,
~-5% 손실 수용.

## 무엇을 돌렸나
- **스크립트**: `scripts/ch_regime_split_v1.py` (신규). R1/R2 와 **동일 substrate** 재사용 —
  `downside_head_riskreward_v1.{build_panel, add_cross_sectional, attach_btc_regime, walk_forward_heads,
  _feats}` (per-fold OOF bucket calib, top100 universe, 6 fold WF + embargo 5d) + 라이브와 동일한
  15m SL-3%/TP+5%/EOD 경로청산 (`recommender_downside_exit_v1.simulate_path`) net 0.15% 차감.
  → panel·head·calibration·청산경로 완전 공유, **정렬/기권만** 바꿔 효과 isolate.
- **OOS**: 38,682 후보 coin-day / 765일 / 2024-04-03..2026-06-01 (R2 coverage 와 byte-동일 = 동일 패널).
  R1 baseline picks = 1531 (R2 스크립트와 동일 재현 — 정합성 확인).
- **3 변형**:
  - **A2a (정렬만)**: regime별 best 정렬키를 prior-fold train OOS 에서 선택 (R1 / pen_lam{1,2,3}).
    픽수 R1 과 동일(1531) — 순수 랭킹 효과.
  - **A2b (기권만)**: prior-fold train regime net_mean < 0 인 regime 을 그 fold 에서 기권(R1 정렬 유지).
  - **A2c (둘다)**: a + b.
- 산출: `output/ch_regime_split_compare_v1.csv` (regime×policy), `_picks_v1.csv` (감사 dump),
  `_coverage_v1.json` (fold 결정·기권가치), `_panel_v1.parquet` (캐시 — 재실행 빠름).

## leak·시간정합성 방어 (§1 — 양보불가)
- **입력 ≤ t-1**: head feature = `build_market_features` 의 `.shift(1)` (D-1). `_feats` 가 `LEAK_COLS`
  (up_high_ret/down_low_ret/eod_ret/lab_* 전부) 제외 — 실측 재확인 (R2 노트와 동일 빌더).
- **regime D-1**: `attach_btc_regime` 가 btc_regime 을 timestamp 시계열에서 1일 shift → day-D row 는
  D-1 regime 을 본다 (같은날 BTC 종가 X). 운영 정합(D-1 regime 이 "오늘 어떤 키 쓸지" 결정).
- **★ A2 regime 결정 = prior-fold train OOS 에서만**: `oos[fold < f]` 로 regime별 정렬키/기권 결정,
  test fold(`fold == f`)는 적용만. fold 0/1 은 이전 train 부족 → R1 기본 (no_prior_train_R1).
  → in-sample regime tuning leak 방어. fold_decisions(coverage)에 fold별 선택 기록.
- **calibration train-only**, **purged WF + embargo 5d** (R1 과 공유), 청산 15m = 진입일 D in-trade (leak X).

## 시도 조합 수 (selection deflate 용)
- A2a 정렬키 = regime별 **4개** (R1 + pen_lam{1,2,3}) × 4 regime, fold마다 train 에서 재선택.
- A2b/c 기권 = regime별 on/off (threshold = train net<0, 1개 룰). abstain-thresh=0 (placeholder).
- **hand-pick 아님** — 모든 regime별 선택은 prior-fold train OOS 의 하방-우선 규칙(%SL↓→deep↓→net↑)이
  결정. 단 abstain threshold(0) 와 λ grid 는 데이터 본 뒤 placeholder → evaluator deflate 대상.

## 1차 결과 (OOS, net 0.15% 차감, 15m SL/TP/EOD)

### 전체(ALL) — R1 대비 Δ
| 정책 | 픽수 | days | %SL ↓ | deepNoSL ↓ | net_mean ↑ | hit | prec@3(pump20) | cum | Sharpe | Sortino |
|---|---|---|---|---|---|---|---|---|---|---|
| **R1 (챔피언)** | 1531 | 765 | 0.456 | 0.135 | -0.00284 | 0.39 | 0.037 | -0.864 | -2.50 | -4.49 |
| **A2a (정렬만)** | 1531 | 765 | **0.321** | **0.078** | -0.00239 | 0.42 | 0.018 | -0.824 | -2.23 | -4.01 |
| **A2b (기권만)** | 614 | 392 | 0.376 | 0.109 | -0.00216 | 0.42 | 0.028 | -0.427 | -1.92 | -3.46 |
| **A2c (둘다)** | 614 | 392 | **0.301** | **0.073** | **-0.00160** | 0.43 | 0.023 | **-0.356** | **-1.43** | **-2.64** |

### regime별 — A2a(정렬만, 픽수 유지) Δ vs R1
| regime | n | Δnet | Δ%SL | ΔdeepNoSL | Δprec20 | A2a net_mean | A2a cum |
|---|---|---|---|---|---|---|---|
| bull_quiet | 666 | +0.0002 | -0.074 | -0.032 | -0.011 | -0.0023 | -0.545 |
| bull_volatile | 157 | **-0.0009** | -0.057 | -0.032 | +0.000 | -0.0035 | **-0.370 (악화)** |
| **bear_quiet** | 208 | **+0.0017** | **-0.221** | -0.106 | -0.014 | **-0.0008** | **+0.040 (양수!)** |
| bear_volatile | 500 | +0.0007 | -0.204 | -0.076 | -0.036 | -0.0029 | -0.411 |

## ★ net 양수/개선 regime 있나
- **bear_quiet (A2a) = 유일하게 net-양수** (cum **+0.040**, net_mean -0.0008, %SL 0.428→0.207). fold 5 에서
  prior-train bear_quiet net 이 +0.00289 (유일한 양수) → pen_lam3.0 선택. A2a 픽이 R1 과 184쌍 다름
  (랩핑된 R1 아님, 진짜 랭킹 효과). **단 n=208 / 100일 — small-sample, bootstrap CI 필수.**
- bear_volatile(타깃): net-양수 못 됨이지만 %SL 0.552→0.348, deepNoSL 0.162→0.086 으로 **하방 대폭 개선**
  (cum -0.482→-0.411). 가설("volatile=net최악")은 맞고, A2 가 그 하방을 누름.
- bull_volatile 은 A2a 에서 **악화**(cum -0.342→-0.370) — penalty 가 이 regime 엔 역효과. (regime별로
  효과 갈림 = 가설의 "regime마다 dynamics 다름"을 역으로 확인.)

## ★ degeneracy 판정 (정식 fold 버전 — 기권이 거래량 감소뿐인가)
**분해 (모두 동일 OOS·청산):**
- A2a (정렬만, 픽수 1531 유지): net_mean -0.00284 → **-0.00239** (Δ+0.00045). **픽수 안 줄이고** 하방·net
  개선 = **degeneracy 아님 (순수 랭킹 엣지)**. %SL -0.135, deepNoSL -0.057 은 픽수 변화 0 에서 나온 것.
- A2b (기권만): 픽수 1531 → 614 (**-60%**), net_mean -0.00216. **기권가치 검증**: 기권한 917픽의 R1
  baseline net_mean = **-0.00330** vs 거래한 614픽 = -0.00216 → **기권한 날이 실제로 더 나빴다
  (기권 real value, 무작위 거래량 감소 아님)**. 단 abstain regime = bull_quiet/bull_volatile/bear_volatile
  (거의 전부) — bear_quiet 만 남기는 매우 공격적 "거의 안 쏨" 자세.
- A2c: net_mean -0.00160, cum -0.356, Sharpe -1.43 (4정책 중 최고) — 단 614픽.

**판정**: A2b/c 의 cum 큰 개선(-0.864→-0.356)은 **상당부분 거래량 감소(60% 컷)에서 옴** — cum 은 픽수에
민감. 그러나 (1) 기권한 날이 baseline 에서 진짜 더 나빴고(-0.0033 vs -0.0022), (2) **A2a 가 픽수 0 변화로도
net/하방을 개선**하므로 "순수 degeneracy(안 움직이는 것 골라 도망)"는 아니다. **핵심 가치 = A2a 의 랭킹
엣지 + bear_quiet 양수 전환**이고, A2b 기권은 보조(나쁜 regime-날 회피). **단 모든 정책이 net-음수** —
A2 는 돈을 벌게 하지 않고 **손실을 덜 깊게** 한다(프로젝트 메모리와 일치).

## ★ R2/B2 와의 관계 (반복 축 아님 — 새 발견)
- B2(exit-challenger) 노트 line 66: "regime 은 reject — bear_volatile 개선 미미(Δnet +6e-05)". 그건
  regime-conditional **exit** 였다. A2 는 regime-conditional **ranking+abstain** — bear_volatile Δnet
  **+0.0007** (B2 의 10배), 그리고 **bear_quiet net 양수 전환**(B2 엔 없던 결과). 다른 lever, 다른 결론.
- R2(전역 단일 penalty)의 degeneracy 우려(BTC/DOGE/TRX 저변동 대형주로 도망)는 A2a 에 부분 잔존
  가능 — fold 4/5 에서 bull regime 에 pen_lam3.0 선택됨. 단 A2a 가 픽수 유지로 net 개선한 점은 R2 와 차별.

## 내가 만든 파일
- **신규**: `scripts/ch_regime_split_v1.py`.
- **산출**: `output/ch_regime_split_compare_v1.csv`, `_picks_v1.csv`, `_coverage_v1.json`, `_panel_v1.parquet`.
- **공유 라이브 파일 미터치** (recommend.py/model_registry.py/daily_*.sh). 운영 배선은 아래 diff 제안만.

## recommend.py 배선 diff 제안 (ADOPT 시 — 편집 X, 제안만)
`score_candidates(..., ranking="A2")` 추가 안:
- 이미 `btc_regime = _mode_regime(today)` 계산됨(line 603). `ranking=="A2"` 분기에서 그 regime →
  `REGIME_KEY_MAP`(예 {bear_quiet: pen_lam3.0, ...}) 로 정렬키 선택 + `REGIME_ABSTAIN` set 이면 빈 추천.
- **artifact 고정값으로** (라이브 재선택 금지) — 즉 OOS fold 결정의 다수결을 상수화. R1 경로 byte-identical
  보존(ranking 기본 "R1"). model_registry 에 `recommend_a2_open` challenger_only spec + 자기 ledger.
- 채택 전 evaluator 판정 필요 — 본 노트는 증거 생성만.

## evaluator 가 봐야 할 것 (3줄)
1. **bear_quiet 양수(+0.040 cum)의 robustness**: n=208/100일 small-sample. bootstrap CI95(net_mean Δ)
   가 0 을 넘는지, fold 5 단일 결정(pen_lam3.0, train net +0.00289)에 의존하는지 — 1-fold 운 가능성.
2. **A2b/c degeneracy 비중**: cum 개선의 몇 %가 거래량 -60% 에서 오는지 vs 엣지인지. 픽수 정규화(픽당
   net, 또는 동일 픽수 R1 부분집합과 비교)로 분리. 기권날 baseline -0.0033 < 거래날 -0.0022 는 기권가치
   증거이나, "거의 안 쏨"이 레이더 취지(매일 알림)와 충돌 — 운영 가치 판정 필요.
3. **leak/selection 감사**: regime D-1 shift + prior-fold-only 결정이 진짜 leak-free 인지 적대 검증.
   selection deflate: regime별 4키×4regime×fold 재선택 + abstain threshold(0, 데이터 본 뒤) → trials 부풀림.
   모든 정책 net-음수 — A2 는 "하방 lever"로만 프레임(net-흑자 주장 아님).
