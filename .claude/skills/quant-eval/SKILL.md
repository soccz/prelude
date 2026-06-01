---
name: quant-eval
description: "prelude 포트폴리오급 평가 + 위생 감사 절차. 시그널/모델/정책 후보를 net 거래비용 차감 성과지표와 deflated 지표(PSR/DSR/MinTRL), bootstrap CI, walk-forward 무결성, calibration 정직성으로 평가하고 look-ahead leak·selection bias·시간정합성을 적대적으로 감사해 ADOPT/SHADOW/REJECT 판정을 낸다. 백테스트 결과 검증, 모델/정책 채택 여부 판단, tear sheet 작성, 평가지표 신뢰도, leak 점검, '이거 실제 포트폴리오로 통하나' 질문 시 이 스킬을 따른다. 후속: 재평가/판정 갱신/지표 추가 요청 시에도 사용. 세션 첫 트리거면 prelude-quant 오케스트레이터가 세션 의례를 먼저 돌린 뒤 이 스킬로 라우팅하는 것이 기본."
---

# quant-eval — 포트폴리오로 통할 신뢰도를 계산하고, leak 을 잡는다

이 프로젝트의 1순위 목표는 **실제 AI 퀀트 포트폴리오로 내놓을 수준의 신뢰도·평가지표**다. 이 스킬은 그 신뢰도를 *어떻게 계산하고 어떻게 적대적으로 검증하는가*의 절차다. 핵심 자세: **researcher 주장을 믿지 않고 코드·데이터를 직접 다시 읽어 재계산한다.**

## 0. 판정 우선순위 (절대 혼동 X — CLAUDE.md §2.3)

```
1순위: net Sharpe / Max DD / hit rate / 누적 PnL  (거래비용 차감 후) — 판정의 근거
2순위: forward(live paper / shadow) 표본          — 백테스트만으로 ADOPT 안 함
3순위 (사후 표기만): PSR / DSR / MinTRL / Holm / IC / CRPS / 사전등록
양보 불가 (FAIL 시 net 결과 무관 REJECT): leak / 시간정합성 / 비용차감 / 자동주문 부재
```

**학술표준은 사전 게이트가 아니다.** 사용자가 여러 프로그램에서 본 함정: 다중검정 보정으로 신호 다 죽이기, IC 우선 최적화로 PnL 손해, 사전등록 때문에 발견 못 채택. 그러니 DSR 음수·Holm 탈락이어도 **net 결과+forward 가 좋으면 REJECT 가 아니라 "주의 표기 + SHADOW"**. 학술 지표는 신뢰도 *포장*이지 *사형선고*가 아니다.

**단 4대 위생은 예외**: leak 의심되면 net 결과가 화려해도 REJECT(환상이므로).

## 1. Portfolio-grade 지표 묶음 (net 기준)

거래비용 0.15% 왕복 차감 후 계산. 상세 공식·출처는 [references/metrics.md](references/metrics.md).

**1차(판정 근거):** Sharpe(ann) · Sortino · Calmar · Max DD · hit rate · 누적/평균 PnL
**리스크:** Volatility(ann) · VaR/CVaR(95) · Tail Ratio · Ulcer Index · Recovery Factor · max W/L streak
**deflated(사후 표기):** PSR · DSR(trials=시도 조합 수) · MinTRL · bootstrap CI95(블록 부트스트랩)
**상대(사후 표기):** vs BTC HODL — IR · Beta · Tracking Error
**층화:** regime(bull_quiet/volatile, bear) × setup × score 버킷별 분해 — 한 regime 에만 의존하는지 확인

n(closed trade) 이 작으면 지표를 신뢰하지 말 것 — 표본 수 항상 명시. 작으면 SHADOW.

## 2. Walk-forward 무결성 체크
- purged WF + embargo 가 실제로 적용됐는지(train/val 경계에서 시점 겹침 없는지) 코드로 확인.
- threshold/quantile 가 **train OOF** 에서 나왔는지(val 사용 = leak v1, train direct = overfit v2).
- 유니버스가 fold train 종료 시점 기준으로 잘렸는지(미래 상장 코인 포함 X).

## 3. 적대적 위생 감사 (반증 시도 — 4대 양보불가)
"성능이 좋으면 먼저 leak 부터 의심." 이 프로젝트는 same-day leak 2번 이력.
- **leak**: 입력 피처에 same-day high/low/close 섞였나? `next_*`·LEAK_COLS 가 feature 에서 빠졌나? distribution 의 `label_date = feature_date - 1day` join 이 맞나? prec/accuracy 가 비현실적으로 높지 않나?
- **시간정합성**: 유니버스·정규화 통계가 미래를 안 봤나?
- **비용차감**: gross 로 보고됐나(환상)? net 인가?
- **자동주문 부재**: 업비트 API key·자동 주문 코드가 추가되지 않았나?
하나라도 FAIL → REJECT(net 결과 무관).

## 4. selection bias deflate
researcher 가 시도한 조합 수(예: 288→4608 sweep, 7 head 중 5 hand-pick)를 받아 **DSR/PSR 의 trials(N) 인자**에 넣는다. "많이 던져서 하나 맞은 것"과 "robust edge" 구분. trials 가 크면 deflate 된 지표를 표기하되, **net+forward 가 좋으면 그 자체로 REJECT 하지 않는다**(§0). 주의: `build_dashboard.py` 자동 DSR 은 `n_trials=50` default 이므로(`n_trials_assumed` 필드), 실제 N 으로 `compute_psr_dsr(..., n_trials=N)` 재호출한 값을 판정에 쓴다 — dashboard DSR 을 그대로 trials=N 으로 옮기지 말 것.

## 5. calibration 정직성
rare event 에서 raw probability 는 과신(이 프로젝트: +20% tail pred 60.3% vs actual 11.6%, gap +48.7pp). top decile predicted vs actual 을 측정하고, 알림이 **bucket-based hist hit** 으로 표기됐는지 확인. isotonic 은 rare 극단부에서 흔들리므로 bucket 표기 선호.

## 6. forward / observed 우선
backtest replay 와 observed 가 크게 다르면 observed 우선(예: preopen observed -40.8% vs replay active 0건 → DEMOTE 가 옳았음). backtest 좋고 forward 없음 → 최대 SHADOW.

## 7. 판정 카드 (정확히 따른다 — ops-steward 입력)
```
## VERDICT: ADOPT | SHADOW | REJECT — {후보명}
- 가설: {한 줄}
- net 성과(0.15% 차감): Sharpe X / Sortino X / Calmar X / MaxDD X% / hit X% / 누적 PnL X% (n=N)
- deflated(사후 표기): PSR X / DSR X(trials=N) / MinTRL X / CI95 [lo,hi]
- forward/observed: {표본 결과 또는 "없음→SHADOW"}
- calibration: top decile pred X% vs actual X%(gap), bucket 표기 Y/N
- 위생(양보불가): leak P/F / 시간정합성 P/F / 비용차감 P/F / 자동주문부재 P/F
- selection: 시도 N, deflate 반영
- 판정 근거: {net 우선, 학술표준은 표기. 위생 FAIL 이면 무조건 REJECT}
```

**판정 기준:**
- **ADOPT**: 위생 4/4 PASS + net 결과 양호 + forward/observed 양호(또는 충분한 표본).
- **SHADOW**: 위생 PASS + net 유망하나 forward 표본 부족 or n 작음 or selection 우려 → shadow ledger 로 표본 더.
- **REJECT**: 위생 1개라도 FAIL, 또는 net+forward 모두 음수/무엣지.

## 8. 도구
기존 검증/지표 스크립트 재사용: `scripts/bootstrap_edge_v1.py`, `scripts/calibration_paper_ledger.py`, `scripts/baseline_showdown_v1.py`, `scripts/model_vs_random_v*.py`, `ops/recommendation_quality.py`, `ops/policy_gate.py`(`evaluate_policy_gate`, GATE_ID=policy_gate_v1 — replay/observed → PROMOTE_PAPER/COLLECT/DEMOTE 산출). 없는 지표는 prelude 안에 새로 작성(self-contained). dashboard tear sheet 는 `scripts/build_dashboard.py` 가 22+ metric 산출 — 공식 일치 참고.
