# 연구 노트: R2 (downside-penalized) 챌린저 vs R1 챔피언 — OOS net 비교

## 가설
하방을 **선형 패널티**(`p_up10 - λ·p_dn5`)로 더 강하게 누르면, R1(비율 `p_up10/max(p_dn5,eps)`)이
상단에 남기는 stop-out·deep-dump 픽을 배제해 **당일 하방위험이 낮아진다** (사용자: 하락최소화 > 상승,
~-5% 손실 수용). R1·R2 는 **동일 de-corr head 확률의 결정론적 재정렬**이라 head/calibration/panel 을
완전히 공유 — 정렬키만 다르다(랭킹 효과 isolate).

## 무엇을 돌렸나
- **스크립트**: `scripts/r2_challenger_compare_v1.py` (신규). R1 과 동일한 leak-free head 빌더
  (`downside_head_riskreward_v1.walk_forward_heads`, `_feats`, per-fold OOF bucket calib)를 재사용해
  purged WF(6 fold, embargo 5d) OOS 를 만들고, **라이브 R1 유니버스와 동일하게 static top100** 으로 제한.
- **realized path = 라이브 ledger 와 동일**: 15m 봉 SL -3% / TP +5% / EOD 경로청산
  (`recommender_downside_exit_v1.simulate_path`) **net 왕복 0.15% 차감**. (기존
  `downside_head_riskreward_compare_v1.csv` 는 realized=EOD 종가였음 — 라이브와 다름 → 새로 돌림.)
- **정렬키 5개 비교**: R1 + R2 λ∈{0.5,1,2,3}. 동일 OOS 행(매 정책 n=1531, 765일)에서 매일 top-3.
- OOS 윈도: **2024-04-03 .. 2026-06-01 (765 거래일, top-3 = 1531 coin-day)**. regime 분포:
  bull_quiet 17156 / bear_volatile 14856 / bear_quiet 5446 / bull_volatile 1224 (universe rows).

## leak·시간정합성 방어 (§1 체크리스트 — 양보불가)
- **입력 ≤ t-1**: head feature = `build_market_features` 의 `.shift(1)` (D-1 까지). `_feats` 가
  `PRECURSOR_FEATURES` 에서 `LEAK_COLS`(intraday_high_ret/down_low_ret/eod_ret/lab_* 등)·next_* 제외.
  **재확인 실측**: `PRECURSOR_FEATURES ∩ LEAK_COLS = []` (0개).
- **라벨 ≥ t**: up/down 라벨 = day-D open 대비 high/low (미래). train 에서만 fit, test fold 는 적용만.
- **calibration train-only**: per-fold OOF bucket historical-hit (raw prob 과신 금지).
- **purged WF + embargo 5d**: train 종료 시점 기준 유니버스 시간정합성.
- **청산 15m 경로**: 진입일 D 의 in-trade outcome → leak 아님(진입 결정은 D-1 까지). 15m 봉 시간
  오름차순 walk, 같은 봉 SL·TP 동시면 SL 먼저(보수).
- **R2 새 leak 유입 = 구조적 0**: R2 는 R1 과 **동일 head 확률의 결정론적 재정렬**. 실측 검증:
  모든 정책의 n=1531 동일(같은 OOS 행), worst=-0.0315 동일(하드 SL+비용 floor) → 픽만 바뀜.

## 시도 조합 수 (selection deflate 용)
- 정렬키 **5개**(R1 + R2 λ 4개). hand-pick 아님 — λ grid 는 랜딩(λ=1) 포함 placeholder.
- best λ 는 **데이터 본 뒤** 하방-우선 규칙으로 선정 → evaluator 가 deflate 시 5-trial 기준.
- 유니버스/fold/embargo/head 는 R1 과 공유(추가 자유도 0).

## 1차 결과 (OOS, net 0.15% 차감, 라이브 15m SL/TP/EOD 경로)

★ **핵심 함정 회피**: 하드 SL -3% 가 손실을 -3.15% 로 floor → `P(realized net ≤ -5%)` 는 **모든
정렬키에서 ~0** 이라 구분 불가(표의 deep_loss_freq 열 전부 0.0, worst 전부 -0.0315). 따라서 사용자의
하방-우선 선호는 (a) **%SL stop-out 빈도**(라이브 경로의 '나쁜 사건')와 (b) **no-SL EOD deep-loss**
(SL 끄면 그 픽이 얼마나 빠졌나 = 정렬키 본질 하방)로 측정. 둘 다 R1 vs R2 를 강하게 구분한다.

| 정렬키 | param | n | days | %SL ↓ | no-SL deep<br>(net≤-5%) ↓ | net_mean ↑ | hit | prec@3<br>(pump20) | CVaR95 | MaxDD | cum | Sharpe |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **R1_ratio (챔피언)** | - | 1531 | 765 | **0.456** | **0.135** | -0.0028 | 0.39 | 0.037 | -0.0315 | -0.872 | -0.864 | -2.50 |
| R2 penalized | λ=0.5 | 1531 | 765 | 0.483 | 0.149 | -0.0033 | 0.38 | 0.040 | -0.0315 | -0.898 | -0.893 | -2.91 |
| **R2 penalized** | **λ=1.0** | 1531 | 765 | **0.272** | **0.058** | -0.0023 | 0.42 | 0.017 | -0.0315 | -0.823 | -0.814 | -2.14 |
| R2 penalized | λ=2.0 | 1531 | 765 | 0.193 | 0.027 | -0.0022 | 0.43 | 0.005 | -0.0315 | -0.812 | -0.802 | -2.18 |
| **R2 penalized** | **λ=3.0** | 1531 | 765 | **0.185** | **0.026** | -0.0020 | 0.43 | 0.003 | -0.0315 | -0.795 | -0.784 | **-2.01** |

### best λ 선정 (하방-우선: %SL ↓ → no-SL deep-loss ↓ → net_mean ↑)
- **하방 최소화 절대 우선이면 λ=3.0**: %SL 0.185 (R1 0.456 대비 **-0.270**), no-SL deep 0.026 (**-0.108**),
  net_mean -0.0020 (R1 -0.0028 보다 **덜 손실, +0.0008**), Sharpe -2.01 (R1 -2.50 보다 개선). 단 상방 포기 큼
  (prec@3 0.037→0.003, %TP 0.219→0.065).
- **λ=1.0 = 랜딩값 + 균형점**: %SL 0.272 (하방 **반감**), no-SL deep 0.135→0.058 (**반감 이상**), hit 0.39→0.42,
  net_mean -0.0028→-0.0023. 상방 일부 유지(prec@3 0.017). 랜딩의 "R1 0.33 → R2(λ=1) 0.16 하방 반토막"
  주장(그건 intraday-touch 기준)을 **net 경로에서도 재현**(no-SL deep 0.135→0.058).
- **권고**: λ=1.0 을 `R2_LAMBDA` 기본값으로 채택(균형) — 하방 반감 + net/hit 유지. λ=3 은 "상방 거의 포기,
  하방 극소화" 극단 옵션. **둘 다 placeholder**(§2.5) — evaluator 판정 + forward 누적으로 갱신.

### ★ 정직한 caveat (EDA-hit ≠ 돈벌기)
**모든 정렬키가 net-negative** (cum -78%~-89%, Sharpe 음수). R2 는 **돈을 벌게 하지 않는다 — 하방을 줄인다**.
이는 프로젝트 메모리("진입엣지 진짜지만 자동 net 미확립; lever 는 exit/하방 규율")와 일치. R2 의 가치는
"net 흑자 전환"이 아니라 **같은 음수 영역에서 손실을 덜 깊게**(stop-out/deep-dump 반감, net_mean·MaxDD·
Sharpe 개선)다. 채택 판단은 이 프레임에서.

### 관찰 (evaluator 참고)
- R1 ratio 는 `p_up10=0` 인 저변동 코인(예: 조용한 날 BTC)을 ratio=0 으로 상단 잔류시키는 degenerate
  케이스가 있다(`r2_challenger_picks_v1.csv`). R2 penalty 는 0-upside·고-downside 코인에 강한 음수 점수를
  줘 밀어냄 → R2 하방 개선의 일부 출처.

## 내가 만든/바꾼 파일·함수
- **신규**: `scripts/r2_challenger_compare_v1.py` (OOS R1 vs R2 net 비교 빌더).
- **신규 산출물**: `output/r2_challenger_compare_v1.csv` (위 표), `output/r2_challenger_picks_v1.csv`
  (R1·R2 best-λ same-date side-by-side 픽 dump), `output/r2_challenger_coverage_v1.json`.
- **`signals/recommend.py`**: `score_candidates(..., ranking="R1")` 인자 추가. `R2_LAMBDA=1.0` 상수 추가.
  ranking="R1"(기본)=현행 정렬·tie-break **byte-identical 보존**, "R2"=`p_up10-λ·p_dn5` 재정렬
  (tie-break 동일 하방-우선). `rr_ratio`/`rr_pen` 은 출력 보존.
- **`scripts/recommend_today.py`**: `append_today(..., ranking="R1")` + `--ranking {R1,R2}` CLI +
  `SHADOW_RECOMMEND_LEDGER_R2="output/shadow_ledger_recommend_r2.csv"`. R2 는 자기 ledger 로 record-only.
- **`signals/model_registry.py`**: `recommend_r2_open` ModelSpec 추가 (slots=["open"], 자기 ledger,
  predict_ref=score_candidates(ranking='R2'), challenger_only=True, R1 과 동일 스키마).
- close 경로 무변경: `close_recommend_ledger.py --ledger output/shadow_ledger_recommend_r2.csv` 로 R2 청산
  가능(SL/TP 경로 ranking-agnostic).

## R1 byte-identical 확인 (실측)
`score_candidates('2026-05-30', slot='open')` before(git HEAD)/after 비교:
- **top3 dict 완전 동일 (top3 equal: True, full dict equal: True)**. 코인 NEWT/BARD/BIO 동일.
- R2 는 rank-3 만 교체: BIO(p_dn5=0.1895) → AAVE(p_dn5=0.0266) — 고-하방 픽을 저-하방 픽으로 강등.
→ **챔피언 발송 경로 불변 보장** (daily_run/텔레그램/champion_state 미터치).

## evaluator 가 봐야 할 것 (3줄)
1. **하방 지표 선택의 정당성**: 하드 SL -3% 때문에 P(net≤-5%)·worst·CVaR95 가 모든 정책에서 floor 돼
   무차별 → 나는 %SL stop-out + no-SL EOD deep-loss 로 하방을 측정했다. 이 대체지표가 사용자 선호
   (deep-loss freq ↓)를 정직하게 대변하는지, 더 나은 하방 metric(예: 갭다운 빈도)이 있는지 판정.
2. **net-negative 환경에서의 채택 논리**: R1·R2 모두 cum 음수. R2 의 "손실 덜 깊게"가 SHADOW 채택
   근거로 충분한지, λ=1 vs λ=3 중 무엇이 사용자 risk profile 에 맞는지, bootstrap CI95(하방 Δ)로
   765일 차이가 유의한지.
3. **leak/selection 감사**: R2=R1 head 의 결정론적 재정렬(새 leak 0)이라는 주장 적대 검증 + 5-trial
   selection deflate. 특히 best-λ 가 데이터 본 뒤 선정됐음(in-sample λ 선택 bias) 명시 — forward 로 재확인 필요.
