## VERDICT: SHADOW — crowding_decay_v1 (군중쏠림 강등)

- 가설: D-1 까지 이미 많이 오른/쏠린 픽일수록 forward honest net↓·deep_loss↑ (Newton 과확장/늦은 칼받기). 진입 DOWNRANK 후보.
- net 성과 (비용 0.15% 차감, EOD honest): HIGH net -0.0109 / LOW +0.0017 / net차 -0.0126, day-cluster bootstrap CI95[-0.0193,-0.0060] 0제외 (재현 일치). deep차 +0.158 CI[+0.126,+0.190] 0제외.
- forward/observed: **15m 정직 경로는 2025-05~2025-11 (181일, 1227픽, 24.5% 커버) 만 존재.** 그 구간에서 효과 붕괴 (아래). live paper 표본 없음.
- 위생 감사 (4대 양보불가):
  - leak: **PASS** — 입력 D-1 (return_7d/roc_3d/return_5d = close/close.shift(N)-1, per-day cross-sectional rank, 미래 close 미사용). 타겟 = label_date(D)의 open→close (build_4h_panel eod_ret_4h). day-shift join 확인: backfill L164 `date_only = label_date - 1day` → 피처(D-1) merge 라벨(D). OOF = PurgedWalkForward(5fold, embargo10, holdout180) 의 val 점수. next_close 명칭은 오해소지지만 실제는 D 당일 EOD (leak 아님).
  - 시간정합성: **PASS** — universe top100 liq_rank_daily per-day, fold train 종료 기준 OOF.
  - 비용차감: **PASS** — 왕복 0.15% 차감, gross 미사용. 15m bracket 도 -COST.
  - 자동주문 부재: **PASS** — record-only, DOWNRANK 배선은 recommendation_quality stratum (사용자 컨펌 사안).
- selection: 단일 가설, crowd_index 고정(3프록시 평균, cherry-pick 없음), trials 낮음 → deflate 불필요.

### ★ 15m TP5/SL3 cross-check (적대 핵심) — 효과가 운영에서 살아남지 않음

| policy | LOW net | HIGH net | net차 | bootstrap CI95 | deep차 | perm_p |
|---|---|---|---|---|---|---|
| EOD (full 5010, 2023-25) | +0.0017 | -0.0110 | **-0.0126** | [-0.019,-0.006] 0제외 | **+0.158** | **0.002** |
| EOD (covered subset 1227, 2025-05~) | -0.0017 | -0.0110 | -0.0094 | [-0.022,**+0.003**] 0포함 | +0.122 | **0.70** |
| **TP5/SL3 bracket** (covered) | -0.0000 | -0.0033 | **-0.0033** | [-0.009,**+0.002**] 0포함 | **+0.000** | 0.52 |
| TP5/SL5 (covered) | +0.0011 | -0.0026 | -0.0037 | [-0.010,+0.003] 0포함 | +0.105 | 0.48 |
| TP10/SL3 (covered) | +0.0005 | -0.0063 | -0.0069 | [-0.015,+0.001] 0포함 | +0.000 | 0.19 |

- 15m sanity: eod15 vs backfill EOD corr=0.9984, MAE=0.0003 → 같은 픽·경로 확실.
- **deep차(+0.158, 가장 강한 주장)는 운영 -3% SL 이 전부 절단 → deep=0.000, 효과 소멸.** = "HIGH-crowd 의 deep-loss" 는 EOD 까지 들고 있을 때만 발생, 운영 bracket 에선 SL 이 먼저 잡음.
- **net차도 -0.0033 으로 축소, CI 0 포함.** 운영 청산에선 유의하지 않음.

### ★ 기간 분해 (왜 full 은 유의한데 운영은 아닌가)

| 기간 | n | net차 | within-date perm_p |
|---|---|---|---|
| FULL 2023-09~2025-11 | 5010 | -0.0125 | 0.002 |
| **PRE 2023-09~2025-05 (15m 데이터 無)** | 3766 | -0.0133 | **0.0005** |
| **RECENT 2025-05~ (15m 검증가능)** | 1244 | -0.0099 | **0.666** |

- **효과 전체가 PRE-2025-05 구간에 집중 (p=0.0005). 그 구간은 alt 15m 데이터가 없어 intraday 검증 불가.** 15m DB 는 BTC만 2023~, alt 는 2025-05-03~.
- 15m 으로 검증 가능한 최근 구간에서는 EOD 효과 자체가 **within-date noise (p=0.67)** — bracket 적용 전에 이미.

### 보조 적대 결과
- vol 재포장: corr(crowd_index, vol_5d_rank)=+0.10 (직접 재포장 아님), 단 **vol_5d_rank 단독으로도 거의 동일 효과 (net차 -0.0117, p=0.001)**. crowd 가설과 "고변동=나쁨" 가설이 관측상 거의 구분 안 됨. atr_pct 와는 -0.29 (단순 ATR 아님).
- 분위 민감도: 효과가 top quintile(q5 net -0.0137) 에 집중, q1-q4 평탄(+0.002/+0.000/-0.003/-0.002). 매끈한 gradient 아닌 "극단 top 20%" tail.
- DOWNRANK 경제성: 발사regime 픽의 **33% (1397/4255) 침묵** → net -0.0038 → -0.0000 (여전히 breakeven, 못 범). 알림 1/3 손실 대비 net 0 도달일 뿐, 양수 PnL 아님.

### 판정 근거 (SHADOW — ADOPT 아님, REJECT 도 아님)
1. **위생 4대 전부 PASS** — leak 없음. day-quality(permutation noise REJECT)·binance(일봉 낙관근사 환상)와 달리 EOD net차는 일봉 정직 청산이고 full-sample 에서 진짜 (p=0.002, 코인+날 둘다).
2. **그러나 운영 청산(TP5/SL3)에서 효과 미확립** — 강한 주장(deep차)은 SL 이 절단해 소멸, net차는 CI 0 포함. = binance 와 같은 함정의 변형: full-sample 일봉 강세 ↔ 15m 정직/최근 OOS 약화.
3. **§2.3 존중** — full EOD 결과·downside-first(deep-loss reshaping) 정합이 있어 학술적으로 REJECT 하지 않는다. 다만 forward 부재 + 최근 OOS noise + bracket 소멸 → **ADOPT 상한 = SHADOW.**
4. **메모리 정합** — radar-not-strategy (전 cell net≤0, 못 범 재확인). user-downside-first: deep-loss reshaping 매력적이나 운영 SL 이 이미 그 일을 함 (중복).

### SHADOW 다음 단계 (배선 권고)
- recommendation_quality 에 **evidence stratum (record-only)** 으로 crowd_q=high 픽에 `crowd_overext=1` 플래그만 부착. **DOWNRANK 침묵 배선 보류** — 침묵 대신 메시지에 "과확장 D-1 (deep-loss 주의)" 1줄 고지 + shadow ledger 에 crowd_q 컬럼 추가.
- **게이트(ADOPT 승격 조건):** 2025-05~ 이후 alt 15m 누적이 쌓이는 대로 TP5/SL3 bracket 에서 HIGH-LOW net차 bootstrap CI 가 0 제외 + within-date perm p<0.05 를 **순수 forward 구간**에서 재확인. 그때까지 침묵으로 알림 1/3 버리지 말 것.
- 재현: `scripts/crowding_decay_15m_crosscheck.py`, `output/crowding_decay_15m_crosscheck.csv`.
