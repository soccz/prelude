<!-- 출처: prelude-next-research-design 워크플로우 (11 에이전트: 2 정찰 + 4 후보설계 + 4 적대감사 + 종합), 2026-06-25. -->
<!-- 4 후보 감사: P4HS DEAD_END(kill-test 실행 net CI<0) · SWING-DD DEAD_END(C1+C3 재포장) · LVG-XS WEAK(현물 long-only 환금벽) · PRPC WEAK→선정(entry-timing 미시도, mech4/nov4). -->

# prelude 다음 연구 — (b) 데이터/타임프레임 전환

## 0. 왜 (b)인가 + 선정 기준

**(a) post-processing은 천장에 닿았다 (메모리 15+ 실험).** 랭킹·청산·필터·발사·보유기간·옵션식 exit·사이징 — 09:00 고정진입 *위에서* 후처리만 바꾼 모든 실험이 net 0/음수. C4가 엔진 내부 30셀 전부 음수로 이를 못박았고, exit lever는 "소진 확정"으로 메모리에 기록됨. 따라서 남은 진짜 lever는 **데이터/타임프레임/진입 메커니즘 자체의 전환**뿐이다.

**선정 기준 = net-positive 가능성 × 견고검증 가능성.**
- net-positive 가능성: 죽은 15실험의 공통 실패구조(TP-before-SL · 스파이크-후-덤프 · 변동성-다운시프트 degeneracy)를 *메커니즘으로* 피하는가.
- 견고검증 가능성: 4h/d1 처럼 3.2년+ 조밀 데이터로 purged-WF가 가능한가 (15m alt-starved 회피).

---

## 1. 후보 4종 비교표

| 후보 (축) | 메커니즘 신빙성 | 데이터 충분 | 재포장? | 감사 판정 |
|---|---|---|---|---|
| **P4HS** 4h 확정돌파 진입 (entry-timing→4h close) | 2 | 5 | No | **DEAD_END** — kill-test 직접 실행, net CI95 [-0.0110,-0.0076] 0 완전 밑돔. 두 fix(first-bar SL 0.456→0.161)는 작동했으나 돌파-후-반전 구조 천장 불변 |
| **SWING-DD** 멀티데이 종가추세 + max-DD 게이트 (label reshaping) | 2 | 4 | **Yes** (C1+C3) | **DEAD_END** — pct_sl 낮춰도 net 부호 안 바뀜이 C1·C3에서 두 번 반증. excess<0 (알파 부재). DD게이트=손실분포 reshaping뿐 |
| **LVG-XS** 저변동 grind-up 시장중립 EXCESS (universe+net정의 전환) | 3 | 4 | No | **WEAK** — cross-sectional EXCESS 엣지는 진짜(repro 확인). 그러나 현물 long-only로 2026 bear에서 9게이트 전부 net 음수 = 환금 불가가 구조적 벽 |
| **★PRPC** 펌프-후 reclaim 확인진입 (entry-timing→observe-then-confirm) | **4** | 4 | No | **WEAK** — entry-timing 축 진짜 미시도. kill-test 미실행, "두 보강 후 실행"이 감사 요구. net-positive 경로 아직 살아있음 |

**선정 = PRPC.** 이유:
- DEAD_END 2개(P4HS·SWING-DD)는 제외 — 재litigate 금지.
- LVG-XS vs PRPC 중 LVG는 감사가 "엣지는 이미 확인됨, **추가 universe/label 튜닝 금지**"라 명시했고, net-positive 경로(현물 시장중립 숏)가 §3.1 자동주문/현물 제약상 막혀 있음. 환금 벽은 메커니즘이 아니라 자산 구조 문제 → prelude 범위에서 뚫기 어려움.
- PRPC는 mechanism_credible=4·novelty=4로 4후보 중 최강이고, entry-timing(관측-후-확인)이라는 lever를 **아직 한 번도 테스트 안 함**. 결정적 1차 kill-test가 싸고 명확.

---

## 2. ★선정 프로그램 — PRPC (Post-Pump Reclaim-Confirmation)

**이름:** PRPC — 펌프 후 consolidation을 거친 코인의 4h 재돌파 확인진입 스윙 레이더.

**가설:** "펌프 *자체*"가 아니라 "펌프 *이후* 과열이 풀리고(눌림·변동성수축·higher-low) 재돌파를 4h봉으로 *확인한* 코인"이 net-positive다. autopsy 사실(펌프 코인은 진짜 활성 P(touch+5%)=0.56이나, 고점이 장초반 t=0.06에 찍히고 EOD −3.4%로 흘러내려 *당일진입*이 손실원)을 역이용 — spike-then-revert 손실구간을 *건너뛰고*, "과열이 풀렸는가"를 4h로 확인 후에만 진입.

**왜 세 실패구조를 피하나:**
1. **TP-before-SL:** 진입을 펌프일(변동성 최고점)이 아니라 변동성 수축한 consolidation 재돌파봉에 두고, SL을 consolidation low(=관측된 higher-low) 바로 아래에 적응적으로 묶음. 임의 고정 −3% 조기절단이 아니라 가격구조가 깨질 때만 손절.
2. **스파이크-후-덤프:** 진입 *전에* 덤프 여부를 데이터로 본다. consolidation 통과 = "펌프 후 종가가 안정됨", 재돌파 close = "덤프 안 하고 되돌림" 확인. 덤프 코인은 게이트 미통과로 자동 배제.
3. **변동성-다운시프트 degeneracy:** 1차는 학습 없는 deterministic structure-rule. 저변동 대형주는 펌프 트리거 자체를 못 켜서 모집단에 안 들어옴. degeneracy는 학습 셀렉션의 병리인데 발생 경로 자체가 없음.

**라벨/진입/청산/TF/유니버스/데이터:**
- **TF:** 탐지·라벨·진입·청산 전부 **4h** (1.06M행, 3.2년, fresh, alt-starved 아님). 펌프 관측만 d1 high/open.
- **라벨(멀티데이, 현 7-head에 전무한 H≥2d):** 펌프관측일 D = `d1 high/open−1 ≥ T_pump`(초기 +12%). consolidation 윈도우 W = D+1..D+k bars(4h, k 초기 6~18). 진입후보 e = consolidation high를 4h close가 상향돌파한 첫 봉의 close. `y(e)=1 iff` e 진입 후 12개 4h봉(2일) 내 path에서 TP_swing(+8%)이 SL_swing(=consol_low − 0.5·ATR_4h) *보다 먼저* 도달. same-bar 동시 → SL 우선(보수). `label_panel_path_aware`를 multi-day로 일반화(e..e+12 bars 슬라이스).
- **진입(deterministic):** ① 펌프관측 → watch 등재(예측 아님). ② consolidation 게이트 3조건: (i) 과열해소 `max(4h_close in W)/pump_high−1 ≤ −0.03`, (ii) `range_contraction_4h ≥ 1.0`, (iii) higher-low. ③ 재돌파 확인 = 4h close가 consol high 상향돌파한 *그 봉의 close*에 진입(체결가는 보수적으로 *다음* 4h open). 동시 다수면 펌프강도 상위 K=3.
- **청산:** 4h-path bracket — TP_swing +8% / SL_swing(적응적, consol_low−0.5·ATR_4h) / 12봉 만기 4h-close. same-bar → SL 우선. **trail/ladder 등 옵션식 exit 일절 없음**(메모리: exit lever 소진). 왕복 0.15% 항상 차감.
- **유니버스:** 업비트 KRW, 4h DB 265마켓 중 D-1 종료 거래대금 top-100(fold train 종료시점 재산정). 펌프 트리거로 자연히 좁혀짐. 상폐코인 포함.

**leak 방어:** ① 펌프관측일 D는 KST 09:00 마감 후 확정, W·e 모두 D 종료 *이후* bar → look-ahead 없음. ② consol high/low·ATR·higher-low는 진입봉 e *이전*에 닫힌 봉만(e−1까지 컷). ③ 재돌파 동일봉 close-on-close OK이나 체결은 다음 4h open(동일봉 high 누수 차단). ④ path 시뮬은 market·day groupby로 코인간 shift 누수 차단. ⑤ 유니버스 fold 시점 재산정(survivorship). ⑥ T_pump/k/TP/SL 격자는 **train fold에서만** 튜닝, test fold 고정(C4의 +1.59σ noise 함정 차단). embargo ≥ 라벨 horizon(2일 = 12 4h봉).

---

## 3. 기각실험과의 차별 (재litigate 방지)

| 기각 | PRPC가 정확히 무엇이 다른가 |
|---|---|
| **C1** (sustained-close 라벨) | C1은 R1 펌프 유니버스 + R1 15m 청산경로를 그대로 두고 *라벨만* close로 교체 → base rate 못 넘음. PRPC는 진입 *시점 정의* 자체를 바꿈(펌프일이 아니라 재돌파 확인봉). 라벨이 4h-path multi-day(전례 0). |
| **B3/C3 multiday** | R1 펌프-예측 진입집합을 *고정 재사용*하고 보유만 N일 연장 → "보유 길수록 net 단조악화"(후처리). PRPC는 보유연장이 아니라 *진입을 늦춰* 손실경로(스파이크) 자체를 건너뜀. 진입집합이 펌프-예측이 아니라 펌프-관측-후-생존. |
| **cold-start 15m** | 15m alt-starved(2025-05 이후 sparse)로 OOS 사망. PRPC는 4h(3.2년 조밀)로 forward·검증 전부 수행 → noise 비노출. |
| **C2 dip-buy** | 과매도 진입이라 R1과 상관 +0.21~0.48(같은 베타). PRPC는 펌프-관측 모집단이라 *생성과정이 다름* → 상관구조 다를 수 있음(kill-test가 검증). |
| **pattern_sweep_v1** (직접 대조군) | `simulate_d1_simple(next_open,...)`로 consolidation 필터 후 **same-day 진입** → net 음수. PRPC는 같은 consolidation 필터 위에서 진입을 *재돌파 확인봉으로 늦춤*. **이 net delta가 양수인지가 진짜 새 축의 판가름** (감사 요구사항). |

---

## 4. 첫 kill-test (싸고 결정적, 학습 없음)

**구현:** `scripts/prpc_kill_test_v1.py`. 4h DB 벡터화 + `label_panel_path_aware` multi-day 일반화. 하루 내 실행 가능.

**2단 게이트 (하나라도 실패 = 즉사):**

**[게이트 0 — 표본]** 전 3.2년에서 PRPC armed→재돌파 트레이드 수 n과 **fold별 분해**를 먼저 센다.
- 전체 `n < 60`(연<20) → 즉시 DEAD(표본부족).
- 전체 n≥60이어도 **fold당 <15** → foldPos 검정 무의미, T_pump/k 완화 금지(완화하면 pattern_sweep fade 모집단으로 회귀) → DEAD.

**[게이트 1 — net + 대조]** n 통과 시 4-fold purged WF(embargo=5일=라벨horizon+여유)로 트레이드별 net(왕복 0.15% 차감, gap-aware 체결) 산출:
1. **net mean > 0** (R1 천장 −0.00284를 부호로 넘음). ≤0 → 15실험 천장 재확인, DEAD.
2. **block-bootstrap(블록=날짜) net CI95 하한 > 0** (less-negative 아닌 유의 부호전환).
3. **per-fold foldPos > 0.5** (4fold 중 3+ 양수, C4의 foldPos=1/6 운 함정 차단).
4. **★대조 게이트(감사 요구):** *같은 consolidation 필터 + same-day 진입*(pattern_sweep 방식) baseline 대비 PRPC(observe-then-confirm)의 **net delta block-bootstrap CI95 하한 > 0**. 이게 음수면 "관측-후-진입"이 새 축이 아니라 pattern_sweep 재현 → DEAD.
5. base rate 체크: armed-재돌파 hit가 R1 무조건진입 hit(5.6%)보다 within-date permutation p<0.05로 유의하게 높은가.

5개 전부 통과 못하면 즉사. 1차는 deterministic이라 모델 학습 불필요.

---

## 5. 단계 로드맵

1. **kill-test (이번)** — 위 2단 게이트. `_workspace/prpc_kill_test.md`에 n분해·net·CI·대조delta 보고. 실패 시 메모리에 negative result 보존("observe-then-confirm 진입도 net CI95<0" 또는 "표본부족").
2. **(통과 시) 본 백테스트** — T_pump/k/TP/SL을 train fold에서만 튜닝하는 격자(≤6셀, Holm/DSR trials 보고). XGBoost 헤드를 armed 모집단 위에 얹어 p(y=1) top-subset만 발사(selection-bias 주의 — full-pick net이 이미 양수여야 헤드 추가).
3. **평가** — quant-evaluator로 leak 4대(look-ahead/시간정합/비용/자동주문) 감사 + deflated 지표(PSR/DSR) + per-regime(bull/bear) 분해 + excess_mkt_mean(picks-market) 부호 확인.
4. **SHADOW** — backtest-only이므로 ADOPT 아님. SHADOW ledger로 4h forward 표본(특히 다음 bear 국면) 축적. forward net이 백테스트와 일치할 때만 PROMOTE 후보.

---

## 6. 정직한 리스크 (또 죽는다면)

1. **표본부족 사망(가장 가능성 높음):** T_pump+12% + consolidation 3조건 + 재돌파를 다 통과하는 코인이 너무 희소해 n<60. 게이트 완화하면 spike-then-dump 모집단으로 새어들어가 net이 R1로 회귀 → "깨끗한 구조 vs 충분한 표본" trade-off에 둘 다 못 잡음.
2. **재돌파가 또 펌프다:** consolidation 후 재돌파 자체가 2차 스파이크라 진입 직후 다시 장초반고점→되돌림 반복 → TP-before-SL이 4h 스케일에서 재현(autopsy의 spike-then-revert가 timeframe-invariant). pattern_sweep_v1이 이미 부분 반증(consolidation 필터 pump도 spike_fade_rate 0.5~0.7, net −0.005~−0.011) — **대조 게이트(4-4)가 이 리스크를 직접 잡는다.**
3. **EDA-hit≠Sharpe 함정 재발:** precursor lift(top-decile 2~3x)는 hit율이지 net이 아님. base rate 게이트(4-5)는 보조 진단일 뿐, **net+대조 게이트(4-1~4)가 주 판정**이어야 함.
4. **C2식 직교 실패:** armed 모집단이 결국 R1과 일별 net 상관 높아 같은 BTC 베타 하락에 동조 → 현물 숏 불가로 헤지 안 됨.

가장 깔끔한 kill: 게이트 0(표본) 또는 게이트 1-4(대조 delta ≤0). 후자면 "entry-timing을 observe-then-confirm으로 늦춰도 안 열림 = 천장은 시장구조(돌파-후-반전)이지 진입타이밍이 아님"을 4후보 통틀어 마지막 entry-timing lever의 negative result로 확정 보존.

---

**관련 파일(절대경로):**
- 대조 baseline: `/mnt/20t/prelude/scripts/pattern_sweep_v1.py` (same-day 진입, consolidation 필터 net 음수 — PRPC 대조군)
- baseline 증거: `/mnt/20t/prelude/output/failure_mode_pattern_regime_v1.csv`, `/mnt/20t/prelude/output/univariate_precursor_lift_v1.csv`
- 일반화 대상 커널: `/mnt/20t/prelude/signals/labels.py::label_panel_path_aware` (d1→d1 → e..e+12 4h-bar 슬라이스로 일반화)
- 구현 예정: `/mnt/20t/prelude/scripts/prpc_kill_test_v1.py`, 보고 `/mnt/20t/prelude/_workspace/prpc_kill_test.md`