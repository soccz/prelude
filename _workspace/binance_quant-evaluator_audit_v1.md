# Adversarial audit — Binance USDT D-1 lead-lag → Upbit pump (v1)

평가자: quant-evaluator · 2026-06-11 · 대상 run: 22:47 (`output/binance_leadlag_v1_*`, `binance_leadlag_run.log`)

> **주의:** 이 세션은 python/awk 실행이 harness 권한에서 전면 차단되어(researcher 가 겪은 sandbox 와 동일),
> 독립 재실행 스크립트(`_workspace/binance_audit_recalc.py`, 작성 완료·미실행)를 돌리지 못했다. 대신
> **이미 산출된 artifact 를 직접 다시 읽고**(run.log·net.csv·lift.csv·boundary.csv·oof_picks.csv 60+행 표본),
> 그 숫자를 만든 **method 코드를 라인 단위로 감사**해 판정한다. net 의 핵심 결함은 코드+표본만으로 결정적으로 입증됨.

---

## VERDICT: **REJECT** (live ADOPT 불가) — `base_AND_bn_volsurge`

핵심: **양수 net 은 거래비용이 아니라 낙관적 TP/SL 일봉 근사가 만든 환상.** 정직한 종가청산 net 은 음수.
위생(leak)은 깨끗하나, 4대 양보불가 중 **거래비용 차감 후 결과의 정직성**에서 method 가 실패했다(낙관 편향).

---

## 의심 포인트별 판정

### 1. leak / 경계 정렬 — **PASS** (이건 진짜 깨끗하다)
- boundary.csv: `btc_ret_corr_lag0 = 0.9617` vs `lag+1 = -0.0299`, `lag-1 = -0.0740` (n=1095일).
  lag0 가 압도적 최대 → Binance D-1 봉(00:00 UTC 마감 = 09:00 KST = Upbit D open 직전)을
  date-part 로 join 하는 게 정확함. **이건 강한 증거다 — 가짜 lift 의 흔한 원인(잘못된 날짜 join)을 배제.**
- 코드 감사: `build_upbit_panel` 의 라벨 join 은 `label_date = feature_date + 1day`, 라벨(high_D/close_D)은
  `nxt` 테이블에서만, 피처 `cur` 에는 day-D high/low/close 안 섞임. LEAK_COLS/next_* 미생성. market 내부 self-join.
- oof_picks 표본 직접 대조: `entry_open_D`/`high_D`/`close_D` 가 명백히 **다음날(day D) 일봉**이고 feature 행
  타임스탬프(D-1)와 1일 차이. 같은 날 high 가 피처로 새는 정황 **없음**.
- → same-day leak 2번 데인 이 프로젝트 기준으로도 **leak PASS**. (signal-researcher 의 leak 설계는 정확)

### 6. net 시뮬 정직성 — **FAIL** ← 판정의 핵심
- net.csv: volsurge@0.15% → `net_tpsl_mean = +0.0124`(주장된 +1.24%), `net_winrate = 0.5654`.
  **그러나 같은 행 `net_eod_mean = -0.0005`(종가청산), @0.50% 에선 `net_eod_mean = -0.0040`.**
- `net_sim` 코드(scripts L317-347)의 `gross_tpsl` 정의가 이중 낙관:
  ```
  tp_hit = (high_D/open_D - 1) >= 0.05
  gross_tpsl = np.where(tp_hit, 0.05, np.maximum(eod, -0.03))   # low_D 를 아예 안 본다
  ```
  - (a) **TP 낙관**: 장중 high 가 +5% 한 번만 닿으면 종가가 어디든 +5% 확정. 종가 폭락 무시.
  - (b) **SL 낙관**: TP 미달 시 `max(eod, -3%)` — 종가가 -3% 보다 나쁘면 -3% 로 **truncate**.
    게다가 **intraday low_D 를 한 번도 안 봄** → low 가 -10% 찍고 종가 -1% 면 -1% 로 기록(SL -3% 도 안 침).
  - 즉 high 는 "닿기만 하면 익절", low 는 "종가까지 버티면 손실 절단" — 방향이 반대로 낙관.
- **표본으로 입증**(oof_picks 직접 읽음, volsurge 행):
  - AAVE 2023-10-25 vs=1.98: open 115950 → high 120000(+3.5%, TP미달) → close 109150(**EOD -5.86%**).
    tpsl 기록 = max(-5.86%, -3%) = **-3%**. 실제 종가 -5.86% → **+2.86pp 낙관**.
  - AAVE 2023-12-28 vs=2.69: open 159100 → high 164550(+3.4%) → close 151050(**-5.06%**) → 기록 -3%. +2.06pp 낙관.
  - AAVE 2023-12-15 vs=3.92: open 149550 → close 143500(**-4.0%**) → 기록 -3%. +1pp 낙관.
  이런 "종가가 -3% 보다 나쁜데 -3% 로 기록"되는 행이 반복적. low_D 까지 보면 편향은 더 커진다.
- **정직한 단일 숫자(종가청산) net_eod 가 음수**라는 게 결론적. day-bar 로 bracket 을 정직하게 시뮬하려면
  최소 (sl_hit & tp_hit → SL 우선 가정) + low_D 사용이 필요한데 그건 미구현. 15m 경로 시뮬도 없음.
- → 이 프로젝트의 12 실험 전부 음수였던 패턴과 일관: **종가청산 기준으론 여전히 음수.** 양수는 bracket 낙관의 산물.

### 2. fold별 net 일관성 — **부분검증 (재실행 불가, 그러나 net 자체가 무의미)**
- run.log 에 per-fold net 미출력(pooled 만). 독립 per-fold net 재계산 스크립트는 작성했으나 실행 차단.
- lift 는 fold 별 출력됨: volsurge lift 5/5 fold 양수(min 3.03). **lift(hit-rate) 는 일관**되나
  §3 함정(EDA-hit ≠ Sharpe)대로 **hit 일관 ≠ net 일관**. net 의 분모가 낙관 bracket 이라 fold 분해 의미 약화.
- 의심 포인트(한 bull fold 가 끌어올림)는 **종가청산 net 자체가 음수**라 moot — 어느 fold 도 진짜 양수 보장 없음.

### 3. regime split (live=bear_quiet) — **FAIL (live 적용 불가)**
- **결정적 구조 결함**: oof_picks 의 최종 fold val 종료일 = **2025-11-04**.
  PurgedWalkForward holdout=180d + panel scope ≤ Binance max(2026-05-03) 때문에
  **2025-11-04 ~ 현재(2026-06-11)의 ~7개월이 OOF 평가에서 통째로 빠짐.**
- 즉 **현재 라이브 regime(bear_quiet, 2026)에서의 net 은 0 표본.** 백테스트→라이브 비전 사례 2회 있는
  프로젝트에서 "평가 안 된 regime 에 라이브 투입"은 정확히 금지 패턴.

### 4. 시간 집중 — **PASS (baseline 기준), 단 candidate net 은 §6 로 무의미**
- baseline 픽 월별 분포(grep): 2023-07 ~ 2025-11, 월 ~480-558개로 **고르게 분산**(특정 몇 달 몰림 없음).
- candidate(volsurge) 월별 net 집중은 재계산 차단으로 미확인이나, net 자체가 환상이라 우선순위 낮음.

### 5. 선택 편향 deflate — **약한 우려 (researcher 신고는 정직)**
- 시도 조합: 룰 5개(baseline+4 증분) hand-pick, 임계 1세트(b_ret>0 / surge>1.5 / diff>0), sweep 없음, target 2.
  → trials 작아 deflate 부담 작다는 researcher §5 주장은 맞음. surge 1.5 cutoff 가 sweep 안 된 직관값인 건 +.
- 단 per-trade net 이 너무 작아(+0.012 gross, eod 음수) DSR/PSR 로 양수 방어 불가. annualize 해야만 양수로 보이는데
  그건 낙관 bracket + 4픽/일 가정 위에서만 성립.

### 7. 거래 가능성 / freshness — **FAIL (live)**
- Binance DB feature max = **2026-05-03** (오늘 2026-06-11, **39일 stale**). 신호가 **오늘 계산 불가**.
- volsurge fire ~ baseline 의 1/3.5 (14580→4128, pooled). 일일 fire 빈도는 재계산 차단으로 정확값 미산출.
- 운영화하려면 daily binance 수집 파이프 신설 필요 = 신호 신선도가 미구축 인프라에 의존.

### 8. bn_only > baseline 교차검증 — **검증 불가 (artifact 누락) → 단독 주장 신뢰 보류**
- `bn_only_surge+mom` lift 4.92x 주장이지만 **oof_picks.csv 에는 baseline-fire 행만 있음**(bn_only 픽 미포함).
  → bn_only 의 **net 을 산출된 artifact 로 재검증할 방법이 없다.** lift(hit)만 있고 net 없음 = §3 함정 직격.
- "Upbit 모멘텀 없이 Binance 단독이 더 낫다"는 cold-start(장중) REJECT 와 표면상 충돌. 단독 채택은 절대 불가.

---

## 위생 감사 (4대 양보불가)
- **leak: PASS** — lag0 0.962 ≫ lag±1, label_date=feature+1, day-D high/low 피처 미혼입, LEAK_COLS 없음.
- **시간정합성: PASS** — 유니버스 랭크·roc_7d_rank 모두 feature_date(D-1) 기준. WF embargo 10d.
- **비용차감: 코드상 PASS(0.15%+0.5% 둘 다 차감) / 그러나 net 의 GROSS 정의가 낙관 → 실질 FAIL** ←핵심.
  비용은 정직히 뺐지만, 비용 빼기 *전* gross(bracket) 가 이미 부풀려져 net 정직성 실패.
- **자동주문 부재: PASS** — 연구/평가 스크립트, 주문 코드 없음.

## selection: trials = 5룰 × 2target (hand-pick, sweep 無). deflate 부담 작음(researcher 정직). 단 양수 방어 못 함.

---

## 판정 근거 (net 결과 우선, 학술표준은 표기)
이 프로젝트 원칙은 **net Sharpe/MaxDD/누적PnL 우선**이다. 그 1순위 지표가:
- **정직한 종가청산 net = 음수**(@0.15% -0.05%, @0.50% -0.40%). 양수 +1.24% 는 **low 를 안 보고 high 닿으면 익절·
  손실 -3% 절단**하는 낙관 bracket 의 산물 — 비용 문제가 아니라 **gross 추정의 환상**.
- "성능 좋으면 먼저 leak 의심" → leak 은 깨끗했다(드물게 PASS). **그러나 두 번째 환상원(낙관 fill)에 걸렸다.**
- 라이브 regime(bear_quiet)·최근 7개월 **OOS 표본 0**, Binance DB **39일 stale 로 오늘 계산 불가**.
→ **REJECT.** (leak 화려해서가 아니라, net 정직성·forward 표본·freshness 3중 실패.)

## REJECT 이지만 — 살릴 경로 (researcher 에게 돌려보냄)
**lift(hit-rate) 신호는 진짜일 가능성**이 있다(5/5 fold, leak-clean, 경계 정렬 확정). 죽일 건 net 추정 방식·운영성이지 가설이 아니다.
1. **net 을 정직하게 다시**: (a) day-bar 면 `low_D` 사용 + `(tp_hit & sl_hit) → SL 우선` 보수 가정,
   (b) 가능하면 **15m 경로 시뮬**(이미 exit_lab 인프라 있음 — 같은 15m 경로에 bracket 평가). 종가청산 net 도 항상 병기.
   → 정직 net 이 양수로 살아남으면 그때 재평가 요청.
2. **holdout/최근구간 평가**: Binance DB 갱신(collector_binance_d1 --all --days 60) 후 2025-11~2026 OOS 재실행.
   특히 **bear_quiet regime 표본**을 만들어야 라이브 후보 자격.
3. **bn_only 는 net artifact 부터**: oof_picks 에 bn_only 픽도 emit 해서 net 재검증 가능하게. 그 전엔 단독 주장 신뢰 X.
4. **SHADOW 조차 보류**: 보통 "backtest 좋고 forward 없음 → SHADOW" 인데, 여기선 *정직 backtest net 이 이미 음수*라
   shadow ledger 배선도 시기상조. 위 1·2 로 정직 net 양수를 먼저 만들고 → 그때 SHADOW(record-only) 권고.

---

## 재계산 스크립트 (작성 완료, 실행은 비-sandbox 세션에서)
`_workspace/binance_audit_recalc.py` — 권한 풀리면 1줄 실행 시 위 의심 포인트 전부 숫자로 산출:
per-fold net / regime split net / 낙관 bracket vs 보수(SL우선,low_D) vs 종가 net / bootstrap CI95 / 시간집중 / annualized Sharpe.
```
/mnt/20t/prelude/venv/bin/python -u /mnt/20t/prelude/_workspace/binance_audit_recalc.py
```
(low_D 를 raw DB 에서 다시 끌어와 보수 SL-우선 모델로 정직 net 을 산출 — 현재는 코드+표본 추론으로 음수 확정.)
