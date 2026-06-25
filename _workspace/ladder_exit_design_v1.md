<!-- 출처: prelude-delta-ladder-design 워크플로우 (8 에이전트: 3 설계변형 + self-impact → 적대감사 → 종합), 2026-06-25. -->
<!-- 구현 정정 노트: 아래 §6 게이트의 floor축(L5 vs L3) 방향은 downside_metrics 실측 정의와 대조 시 재해석 필요 —
     floor=3 단일 청산은 worst가 -3.15%로 캡되어 p_loss_lt_5=0, floor=5는 floor 체결이 -5.15%<-5%라 p_loss_lt_5↑.
     즉 "wider floor가 deep-loss를 늘린다"가 정상. 따라서 사다리의 1순위 lever = convexity(L3 vs A, 같은 floor)에서
     net_mean·hit·MaxDD·중간 손실분포 재분배로 측정. floor축(L5 vs L3)은 whipsaw vs worst-case 알려진 트레이드오프.
     구현은 4-cell 전 지표를 산출하고, 게이트 판정은 실측 숫자로 이 정정을 반영해 내린다. -->

# 델타-사다리 청산 — 최종 설계 (구현 직전)

> **LOCKED.** 세 변형(A_cost_faithful·B_downside_matched·C_dist_triggered) + 4건 적대감사 + 코드베이스 직접 재검증을 반영해 하나로 수렴. C의 분포-조건화(sustain_ratio=p_h4/p_h6)는 **삭제** — 배선 ckpt `dist_engine_v1`이 full-panel final fit(meta.json `built_at=2026-05-03`, 단일 163,284-sample fit, fold 없음)이라 OOF 픽 날짜(2024-04~2026-06)가 사실상 100% 학습셋 이전 → 조건화 채널 전체가 in-sample leak. A의 cost-faithful 골격 + B의 floor/convexity 분리 ablation을 병합한 무조건 사다리만 락한다.

---

## 0. 한 줄 + 판정축

**한 줄:** 진입가 대비 고정 arm(+2/+4/+6%)에서 1/3씩 부분익절하고 hard floor에서 잔여 전량 청산하는, walk_path 규약과 비트-정합한 비용충실 사다리 청산. 분포-조건화 없음(leak), 자유도 최소(arm/fraction 하드코딩).

**판정축 (왜 이게 옳은 lever인가):** net 양전이 목적이 **아니다**. 동일 OOF 픽셋은 이미 96-cell(`recommender_downside_exit_v1.csv`, 5885 coin-days)에서 전수 sweep됐고 **전 cell net-negative**(best SL0.08/TP0.05 net_mean=-0.14%/trade). 픽셋 자체가 모든 청산정책에서 구조적 손실이라 사다리는 net을 구제 못 한다 — **하방 재분배만** 가능. 따라서 판정축 = 메모리(`user-downside-first`, `radar-not-strategy`)대로 **"deep_loss freq↓·CVaR95↓를 net 비악화로 달성하는 하방 lever인가"**. ADOPT돼도 record-only — champion TP5/SL3 라이브 교체는 사용자 컨펌 필수(§3.2).

**★감사가 발견한 결정적 제약 (게이트 설계의 뿌리):** `downside_metrics`에서 `p_loss_lt_5`·`cvar95`는 **floor 깊이가 단독 결정**한다. floor=3%면 모든 손실이 -3.15% net으로 캡 → `p_loss_lt_5`가 arm 구성과 무관하게 **기계적으로 0.0**, `cvar95`가 **기계적으로 -0.0315**(96-cell·sustainability 8-cell 전 SL3 셀에서 실증 확인). **귀결:** floor 고정 하에서 "ΔP(loss<-5%)·ΔCVaR95"는 사다리 arm 차이를 **구분 못 하는 동어반복**이다. → 하방-lever는 **floor가 변하는 축**(B5 vs B3)에서만 측정 가능. convexity(arm 분할) 효과는 **net_mean·hit·상방분포(h_gt_+5)·MaxDD**에만 나타난다. 두 효과를 **반드시 분리**한다.

---

## 1. 메커니즘 (확정 파라미터)

**확정 파라미터 (placeholder §2.5, sweep 안 함 — primary 1셀 고정):**
- `arms = (0.02, 0.04, 0.06)` — 평균 arm ≈ +4%, 챔피언 단일 +5%와 근사 매칭
- `fractions = (1/3, 1/3, 1/3)` — 균등, 잔여 0 (tail_frac 미사용: arm 모두 닿으면 전량 청산)
- `floor ∈ {0.03, 0.05}` — 0.03=챔피언 매칭, 0.05=사용자 -5% anchor

**같은봉·멀티arm 규약 (walk_path의 SL-먼저 보수 정신 계승):**
1. **floor 먼저:** 같은 봉에서 `low ≤ floor_px`면 그 봉의 **잔여 전량을 -floor로 청산하고 종료** — 같은 봉 high가 arm을 위로 찍었어도 arm 무시(룩어헤드/낙관 차단). **단, 이전 봉에서 확정된 tranche는 보존**(floor가 소급 취소 X).
2. **멀티arm:** 한 봉 high가 여러 arm 동시 교차 시 **낮은 arm부터 순차 전부 체결**, 각 tranche는 **자기 arm의 정확 비율**로 잠금(봉 high 아님).
3. **EOD:** 미청산 잔여 = 마지막 봉 close.

**fill 규칙 (walk_path 비트 미러):**
- arm 체결가 = `entry*(1+arm_i)` 정확 (봉 high 부풀림 금지)
- floor 잔여 = `entry*(1-floor)` → 수익률 `-floor`
- EOD 잔여 = `bars[-1][3]/entry - 1`
- entry = `bars[0][0]` (첫 봉 09:00 KST open)

→ 단일-arm 환원(`arms=(tp,), fractions=(1.0,), floor=sl`)이 `walk_path(bars, sl, tp)`와 **gross 비트동일**(§8 회귀가 강제).

---

## 2. 비용모델 (정직한 공식)

**핵심 사실 (감사 PASS — 이 설계의 가장 강한 축):** 업비트 수수료는 **명목금액 비례** → 매도를 N tranche로 쪼개도 매도 수수료 합 = full 1회와 동일. **'tranche당 0.15% 고정'은 잘못된 비관, 명시적 기각.**

```
fee_total  = FEE_PCT × (1[매수] + Σ_i frac_i)          = FEE_PCT × 2     = 0.0010  (분할 무관, Σfrac=1)
slip_base  = SLIPPAGE_PCT × (1[매수] + Σ_i frac_i)     = SLIPPAGE_PCT × 2 = 0.0005  (분할 무관)
slip_extra = max(0, n_sell_orders - 1) × extra_slip_per_tranche_pct      (분할 페널티, 노브)
net = gross_weighted - fee_total - slip_base - slip_extra
```

- `extra_slip_per_tranche_pct = 0` → 왕복비용 = `2×(FEE+SLIP) = 0.0015` = **챔피언 TP5/SL3와 비트동일**(config: FEE_PCT=0.0005, SLIPPAGE_PCT=0.00025, ROUND_TRIP_COST_PCT=0.0015). **비용 핸디캡 0 = 공정비교 baseline.**
- 단일-arm 환원 시 `n_sell_orders=1` → `slip_extra=0` → 챔피언과 비용 동일(공정비교 보장).
- **슬리피지 민감도 노브 (robustness 축, selection 축 아님):** `extra_slip_per_tranche_pct ∈ {0, 0.0002, 0.0005}` (0/2bp/5bp). 추가로 `CONSERVATIVE_SLIPPAGE_PCT=0.002` tier(편도 0.2%)로 보수 net 병기.
- ⚠️ 감사 보완: base 식에서 fee는 **항상** `2×FEE_PCT`, slip만 `n_sell` 의존으로 **분리 검증**(§8 비용회귀가 강제). 호가창 깊이 데이터 없음 → 선형 노브는 가정, coverage json에 명시.

---

## 3. 새 함수 시그니처 + 의사코드 (exit_lab.py 정합)

기존 `walk_path`/`EXIT_VARIANTS`/`evaluate_exit_variants`는 **불변**. 아래만 추가.

```python
@dataclass(frozen=True)
class LadderSpec:
    arms: tuple          # 오름차순 양수 비율, 예 (0.02,0.04,0.06)
    fractions: tuple     # arm별 청산 비중, len==len(arms), sum==1.0
    floor: Optional[float]  # hard SL magnitude(양수), None=무
    label: str = ""

def walk_ladder_path(
    bars: list,
    arms: tuple = (0.02, 0.04, 0.06),
    fractions: tuple = (1/3, 1/3, 1/3),
    floor: Optional[float] = 0.05,
) -> tuple[float, str, int]:
    """15m 봉 시간순 walk → (gross_weighted_return, outcome, n_sell_orders).
    진입가=bars[0][0]. arm 체결=entry*(1+arm) 정확, floor=entry*(1-floor), EOD=마지막 close.
    outcome ∈ {'nodata','floor','tp_full','partial_eod','eod'}.
    n_sell_orders = 체결된 매도 주문 수(비용 분해용).
    """
    if not bars: return np.nan, "nodata", 0
    entry = bars[0][0]
    if not np.isfinite(entry) or entry <= 0: return np.nan, "nodata", 0
    assert len(arms) == len(fractions)
    assert all(arms[i] < arms[i+1] for i in range(len(arms)-1)), "arms 오름차순"
    assert abs(sum(fractions) - 1.0) < 1e-9, "fractions 합 1.0"
    floor_px = entry*(1-floor) if floor is not None else None
    arm_px = [entry*(1+a) for a in arms]
    remaining_idx = list(range(len(arms)))
    realized = 0.0; n_sell = 0; any_armed = False
    for (o, h, l, c) in bars:
        # 1) floor 먼저(보수): 잔여 전량 -floor, 이전 봉 확정분 보존
        if floor_px is not None and l <= floor_px:
            for i in remaining_idx: realized += fractions[i]*(-floor)
            n_sell += 1; remaining_idx.clear()
            return realized, ("floor" if not any_armed else "partial_floor"), n_sell
        # 2) 멀티arm: 낮은 arm부터 정확 비율 체결
        for i in remaining_idx[:]:
            if h >= arm_px[i]:
                realized += fractions[i]*arms[i]
                remaining_idx.remove(i); n_sell += 1; any_armed = True
    # 3) EOD: 잔여 = 마지막 close
    eod_ret = bars[-1][3]/entry - 1.0
    for i in remaining_idx: realized += fractions[i]*eod_ret
    if remaining_idx: n_sell += 1
    reason = "tp_full" if not remaining_idx else ("partial_eod" if any_armed else "eod")
    return realized, reason, n_sell

def evaluate_ladder_variant(
    bars: list, arms=(0.02,0.04,0.06), fractions=(1/3,1/3,1/3), floor=0.05,
    fee_pct=FEE_PCT, base_slip_pct=SLIPPAGE_PCT, extra_slip_per_tranche_pct=0.0,
) -> Optional[dict]:
    """net_pct(가중) + n_sell + path 극값 + 비용분해. bars 무효면 None."""
    g, reason, n_sell = walk_ladder_path(bars, arms, fractions, floor)
    if not np.isfinite(g): return None
    fee_total  = fee_pct * 2.0
    slip_base  = base_slip_pct * 2.0
    slip_extra = max(0, n_sell - 1) * extra_slip_per_tranche_pct
    net = g - fee_total - slip_base - slip_extra
    return {"net_pct": round(net*100,4), "gross_pct": round(g*100,4),
            "reason": reason, "n_sell": n_sell,
            "cost_pct": round((fee_total+slip_base+slip_extra)*100,4)}
```

---

## 4. leak 방어 체크리스트 (감사 통과 항목)

- [x] **봉 시간 오름차순 1-pass, 미래 봉 미열람** (walk_path와 동일 루프)
- [x] **arm/floor 체결가 = entry 기준 정확 비율** → 봉 내부 경로(고저 순서) 가정 불필요, 봉 극값으로 부풀림 X
- [x] **같은 봉 floor-먼저 보수** → 봉 내부 하방 우선 가정(낙관적 룩어헤드 차단)
- [x] **픽 = `collect_oof_picks(panel, 'lab_pump20')` purged WF OOF만** (train fold 픽 제외 — in-sample 청산 selection 차단). `build_market_features`가 `raw.shift(1)`로 D-1화, 유니버스 컷·BTC regime도 `.shift(1)` (감사 직접 검증). 라벨 `lab_pump20=(high_D/open_D-1)≥0.20`은 진짜 day-D 타겟
- [x] **청산 파라미터는 전 trade 동일 적용** — 정책이 trade outcome 미열람
- [x] **feature D-1, 진입=D open, 경로=[09:00 D, 09:00 D+1)** — 입력 t-1 / 타겟 t 이후 분리 (KRW-BTC 2026-03-15 15m 첫봉 open=d1 open 일치로 실측 검증됨)
- [x] **비용 항상 차감, gross 단독 보고 금지**
- [x] **❌삭제: sustain_ratio 분포-조건화** — `dist_engine_v1` full-panel ckpt leak. 무조건 사다리만 진행

---

## 5. 통제비교 설계 (floor 효과 / convexity 효과 분리)

같은 OOF 픽셋(`collect_oof_picks`, K=3 메인/5 보조), 같은 `bars_map`(`load_paths`), 같은 비용규약. 4-cell ablation:

| cell | 정의 | 격리 목적 |
|---|---|---|
| **A (챔피언)** | `walk_path(floor=0.03, 단일 arm@+5%)` ≡ TP5/SL3 | baseline |
| **L3** | `walk_ladder(arms=(2,4,6), floor=0.03)` | floor=3% 고정 |
| **L5** | `walk_ladder(arms=(2,4,6), floor=0.05)` | floor=5% |
| **S5** | `walk_path(floor=0.05, 단일 arm@+5%)` = TP5/SL5 | 단일+floor5 baseline |

**효과 분리 (★감사 핵심 수정):**
- **convexity 효과 (arm 분할):** `L3 − A` (floor 고정) AND `L5 − S5` (floor 고정). floor가 묶이므로 `p_loss_lt_5`·`cvar95`는 양쪽 **동일**(동어반복) → 이 비교는 **net_mean·hit·h_gt_+5·MaxDD에서만** 측정.
- **floor 효과 (하방 lever):** `L5 − L3` (사다리 고정) AND `S5 − A` (단일 고정). floor가 -3→-5로 움직이므로 `p_loss_lt_5`·`cvar95`가 **실제로 변하는 유일한 비교** → 여기서만 하방-lever 판정.

`downside_metrics(trades)` 양쪽 동일 적용. 1순위 = `p_loss_lt_5`·`cvar95`·`worst`·`mdd`(floor 축), 부수 = `net_mean`·`hit`·`h_gt_+5`(convexity 축).

---

## 6. 검증 게이트 (PurgedWF + 정량 + deflation)

**검증기:** `signals/validate.py:PurgedWalkForward(n_folds=5, embargo_days=10)`. **단 embargo 정직성:** `collect_oof_picks`→`make_folds`는 `EMBARGO=5`(`recall_universe_recommender_v1.py:85`)로 픽 생성. **픽 재생성 안 함**(챔피언과 동일 픽셋 유지가 paired 비교의 핵심) → 평가도 **EMBARGO=5 fold 기준**으로 집계하고 `embargo=5`를 coverage json에 **정직히 표기**(10d 명목 표기 금지 — 감사 지적). paired 비교라 픽단계 5d는 챔피언과 상쇄.

**게이트 (K=3, extra_slip=0 primary 기준):**

*floor 축 (L5 vs L3, 하방-lever — 진짜 판정):*
- **G1:** `ΔP(loss<-5%) ≤ -0.02` AND `Δworst ≥ 0` (단일 최악 trade 악화 없음)
- **G2:** `ΔCVaR95 ≥ +0.003` OR `ΔMaxDD ≥ +0.01`
- **G5:** 5 fold 중 ≥4에서 `ΔP(loss<-5%) < 0` (부호 안정)

*convexity 축 (L3 vs A, net 비악화 — 부분청산이 상방 안 죽이나):*
- **G3:** `Δnet_mean ≥ -0.0010` AND `Δh_gt_+5 ≥ -0.02` (꼬리 절단 과하지 않음)

*강건성:*
- **G4:** extra_slip∈{0,2bp,5bp} + CONSERVATIVE tier 전 구간에서 G1·G3 유지

**판정:**
- **ADOPT** = G1∧G2∧G3∧G4∧G5 (record-only, 라이브 교체는 사용자 컨펌)
- **SHADOW** = G1∧G3만 (꼬리 개선 약함 or 노브 취약 or 백테스트만)
- **REJECT** = G1 위반 OR G3 위반 OR conservative net이 챔피언 대비 -0.0010 초과 악화

⚠️ **판정카드 필수 명시:** "이 픽셋은 음수 net이라 ADOPT는 '덜 잃되 여전히 잃음'(radar-not-strategy)을 의미. 포트폴리오 가치 직결 X."

**trials & deflation (감사 핵심 수정):** primary cell 1개 고정이라 selection bias 구조적으로 낮으나, **family-wise 모집단은 12 cell이 아니라 동일 픽셋 누적 마이닝**. DSR `trials` 산정: 본 사다리(floor 2 × K 2 × slip 3 = 12) **+ prior 96-cell**(`recommender_downside_exit`) **+ prior 8-cell**(`ch_sustainability`) = **trials ≥ 116**. `eval_deflate_cummech_v1.py`의 psr/dsr 재사용, day-block bootstrap(B=2000, day-equal-weight, Holm step-down) 병기 — **§2.3대로 사후 보고용, ADOPT 게이트로 강제 X**(다중검정으로 하방 lever 죽이지 않음). primary 1셀만 판정, '12개 중 best 보고' 금지.

---

## 7. 구현계획

| 단계 | 파일 | 작업 |
|---|---|---|
| 1 | `ledger/exit_lab.py` | `LadderSpec`·`walk_ladder_path`·`evaluate_ladder_variant` 추가. 기존 불변. `from ledger.config import FEE_PCT, SLIPPAGE_PCT` import 추가 |
| 2 | `tests/test_exit_lab.py` | 단일-arm 환원 회귀 + 엣지케이스(§8). `pytest tests/test_exit_lab.py -q` 통과까지 |
| 3 | `scripts/ladder_exit_compare_v1.py` (신규) | `collect_oof_picks`·`load_paths`·`downside_metrics` **import만**(recommender 수정 X). A/L3/L5/S5 4-cell paired + slip sweep → `output/ladder_exit_compare_v1.csv` + `ladder_exit_coverage_v1.json`(매칭 trade 수·fold별 n·embargo=5·비용가정 명시) |
| 4 | 실행·집계 | DSR trials≥116 deflation 병기. 게이트 판정 → quant-evaluator 핸드오프 |
| 5 | `PHASES.md` | 판정 기록(negative result 보존) |

`ledger/config.py` 변경 없음(import만). 순서 엄수: **테스트 통과 후에만 비교 실행**.

---

## 8. 동등성/회귀 테스트 (단일-arm 환원)

**핵심 회귀:** `SCENARIOS × grid`에서 `walk_ladder_path(arms=(tp,), fractions=(1.0,), floor=sl)`의 `gross`가 `walk_path(bars, sl, tp)`와 **`pytest.approx` 일치**. **outcome은 동치집합 매핑으로 완화**(감사 지적: gross 동일해도 reason 갈리는 케이스 존재 — P&L 누수 아님): `{tp_full→tp, floor→sl, eod/partial_eod→eod}`.

**엣지케이스:**
1. 한 봉 high=+7% → fills=[.02,.04,.06], gross=0.04, reason=`tp_full`, **봉 high +7 일괄청산 아님 확인**
2. 같은 봉 floor+arm 동시(low<-5%,high>+2%) → reason=`floor`, gross=-floor, **arm 무시**
3. +2/+4 체결 후 EOD close +3% → gross=(.02+.04+.03)/3, reason=`partial_eod`
4. 무arm·무floor → 전부 EOD, gross=eod_ret = walk_path eod 동일
5. **비용 동등성:** extra_slip=0 → `cost_pct == ROUND_TRIP_COST_PCT*100`(분할이 수수료 안 늘림 고정)
6. **비용 분리:** fee_total은 항상 `2×FEE_PCT`, slip만 n_sell 의존 assert
7. `fractions` 합≠1 OR arms 비오름차순 → AssertionError
8. bad input([], entry≤0, malformed bar close∉[low,high]) → None/방어

---

## 9. self-impact 병행 진단 (요약 스펙)

**별도 트랙** (사다리와 독립, 동시 진행). **판정 아니라 표본축적 게이트:**
- 데이터: `output/shadow_ledger_distribution.csv` (ACTIVE 20 / WATCH_ONLY 70 / SILENCE 164, 30일). estimand = ATT(ACTIVE forward realized − 매칭 WATCH), 1순위 결과변수 = `next_min_return_pct`(하방, downside-first).
- 방법: same-date×setup_quality×btc_regime CEM(sanity) + A_TRIPLE within-quality 회귀(주력) + RD around policy cut(`hit≥52/55`). 추론 = wild-cluster bootstrap(cluster=date, 999). selection은 `decision_policy.py`로 완전관측.
- **게이트:** A_TRIPLE same-day 매칭쌍 ≥30 AND ACTIVE n≥50 **이전엔 `INSUFFICIENT_SAMPLE` 보류**(현재 쌍~5, ACTIVE 20 → 미달, 2~3개월 라이브 추가 필요). 충족 후에만 confirm/reject.
- 파일: `scripts/self_impact_decay_v1.py`·`output/self_impact_decay_v1.{csv,json}`·`tests/test_self_impact_decay.py` (전부 신규, self-contained). `decision_policy.py`/`validate.py`/`close_paper_ledger.py`는 참조만.
- 한계: C_PRIMARY·preopen ACTIVE 0건(대조 부재), bull 표본 0(일반화 불가), intraday 슬립 미기록(mechanical/crowding 완전분리 불가).

---

## 10. 미해결 리스크 & kill-gate

| # | 리스크 | kill-gate / 대응 |
|---|---|---|
| R1 | **픽셋 음수 net 구조적** (96-cell 전부 net<0). 사다리는 하방 재분배만 | ADOPT = record-only, 라이브 교체 X. "덜 잃음"으로만 해석 |
| R2 | **convexity 자기모순 가능성:** arm(2/4/6%) 조기 부분청산이 late-pump 볼록꼬리(h_gt_+5≈8~16%)를 **절단**할 수 있음 | G3에 `Δh_gt_+5 ≥ -0.02` 추가. 절단형이면 SHADOW 강등. arm을 분포로 조정하면 trial++ → deflation 재계산 |
| R3 | **15m 봉 내부 체결순서 미관측** — floor-먼저가 사다리를 비관 편향 | 게이트 통과 시 강한 신호(비관에도 통과). 기각 시 일부는 이 탓 가능 명시 |
| R4 | **슬리피지 명목비례 가정** — 저유동 펌프코인 비선형 충격. 호가창 데이터 없음 | extra_slip 노브 + CONSERVATIVE tier. **conservative net이 챔피언 대비 -0.0010 초과 악화 → 즉시 REJECT** |
| R5 | **floor 축 검정력** — late-pump 표본 적으면 G2 약함. 15m DB ~6개월 | day-block bootstrap CI 폭 보고. 좁으면 SHADOW |
| R6 | **deflation trials 누적** — 동일 픽셋 116+ 마이닝 | DSR trials≥116 보수 산정, 사후 보고. primary 1셀만 판정 |
| R7 | **forward 부재** — 백테스트 OOF replay만 | ADOPT 상한 = SHADOW까지. shadow ledger로 forward 사다리 net/상방 표본 축적 후 최종 |

**최종 kill-gate:** ① 단일-arm 환원 회귀 fail → 구현 중단(semantics drift=leak). ② conservative 비용에서 net 0.1%p 초과 악화 → REJECT. ③ G1(floor 축 deep-loss↓)·G3(convexity net 비악화) 동시 충족 못 하면 negative result로 PHASES 기록 후 종료.

**구현 파일 요약 (절대경로):**
- `/mnt/20t/prelude/ledger/exit_lab.py` (수정: LadderSpec·walk_ladder_path·evaluate_ladder_variant 추가)
- `/mnt/20t/prelude/tests/test_exit_lab.py` (수정: 환원 회귀 + 엣지)
- `/mnt/20t/prelude/scripts/ladder_exit_compare_v1.py` (신규)
- `/mnt/20t/prelude/output/ladder_exit_compare_v1.csv`, `/mnt/20t/prelude/output/ladder_exit_coverage_v1.json` (산출)
- `/mnt/20t/prelude/PHASES.md` (판정 기록)
- self-impact: `/mnt/20t/prelude/scripts/self_impact_decay_v1.py` 외 (별도 트랙, 신규)