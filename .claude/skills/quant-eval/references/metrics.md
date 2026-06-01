# 평가지표 공식 + 출처 (quant-eval reference)

> quant-eval 이 portfolio-grade 지표를 계산/표기할 때 참조. 모든 지표는 **net(거래비용 0.15% 왕복 차감) 일별 수익률 시계열** 기준. dashboard tear sheet(`scripts/build_dashboard.py`)가 이미 22+ metric 을 산출하므로 공식 일치를 우선 확인하고, 없는 것만 새로 계산한다.
>
> **연율화 인자 √N — 코드베이스에 두 컨벤션 공존**: dashboard tear sheet(`build_dashboard.py`)·`idea_validation_report.py` 는 거래일 기준 **√252**. 그 외 crypto 일봉 백테스트·ledger 정본(`ledger/metrics.py` periods_per_year=365, `bootstrap_edge_v1.py`, `baseline_showdown_v1.py`, `model_vs_random_v*.py`, `backtest_wf_ledger.py`)은 365일 무휴거래 기준 **√365**. 같은 시계열도 연율 Sharpe 가 √(365/252)≈**1.20배** 차이난다. **판정 지표는 정본 ledger 와 일치하는 √365 권장**(어느 N 을 썼는지 판정카드에 명시). 아래 표의 √N 은 이 컨벤션을 따른다.

## 목차
1. 1차 성과 지표 (판정 근거)
2. 리스크 지표
3. Deflated / 통계 지표 (사후 표기)
4. 상대 지표 (vs BTC HODL)
5. 출처

---

## 1. 1차 성과 지표 (판정 근거)

| 지표 | 공식 | 비고 |
|------|------|------|
| Sharpe(ann) | mean(r)/std(r) × √N | rf≈0 가정. 일별 net return r |
| Sortino | mean(r)/downside_std(r) × √N | downside_std = 음수 수익만의 std. 상방 변동 패널티 X |
| Calmar | CAGR / |Max DD| | DD 대비 수익 |
| Max DD | min over t of (equity_t / cummax(equity) − 1) | 최대 낙폭 |
| hit rate | #(trade pnl > 0) / #trade | 방향이 아니라 **실현 PnL** 기준 |
| 누적/평균 PnL | Σ net trade pnl / mean | equal weight, 5% TP / EOD close 룰(텔레그램 가이드와 동일) |

**주의(이 프로젝트 함정):** hit rate(방향 맞히기) ≠ Sharpe(돈 벌기). momentum hit 21% 인데 Sharpe -5.2 였던 건 TP-before-SL 게임 때문. 항상 실현손익 경로 기준.

## 2. 리스크 지표

| 지표 | 공식/정의 |
|------|----------|
| Volatility(ann) | std(r) × √N |
| VaR(95) | r 분포의 5% quantile |
| CVaR(95) | VaR 이하 평균 (Rockafellar & Uryasev 2000) |
| Tail Ratio | |95% quantile| / |5% quantile| |
| Ulcer Index | √mean(drawdown_t²) (Martin 1987) — DD 의 깊이+지속 |
| Recovery Factor | 누적수익 / |Max DD| |
| Common Sense Ratio | Tail Ratio × Profit Factor |
| max W/L streak | 최장 연속 승/패 |

## 3. Deflated / 통계 지표 (사후 표기 — 사전 게이트 X)

**이 절의 지표는 신뢰도 *포장*이다. net+forward 가 좋으면 이 지표가 나빠도 REJECT 하지 않는다(CLAUDE.md §2.3).**

- **PSR (Probabilistic Sharpe Ratio)** — 관측 Sharpe 가 기준 Sharpe(보통 0)보다 클 확률. skew/kurtosis 보정:
  `PSR = Φ( (SR − SR*)·√(n−1) / √(1 − skew·SR + (kurt−1)/4·SR²) )`
- **DSR (Deflated Sharpe Ratio)** — PSR 인데 SR* 를 **다중시행(trials N)** 으로 부풀린 기대 최대 Sharpe 로 설정. **trials = signal-researcher 가 시도한 조합 수**(sweep N, hand-pick 포함). selection bias 보정. 단 `build_dashboard.py` 의 `compute_psr_dsr` 자동 호출은 **`n_trials=50` default(placeholder)**로 출력하므로(`n_trials_assumed` 필드 확인), deflate 시 evaluator 가 실제 N(288/4608 등)으로 `compute_psr_dsr(..., n_trials=N)` 재호출한 값을 판정카드에 쓴다 — dashboard DSR 을 그대로 trials=N 으로 옮기지 말 것.
- **MinTRL (Minimum Track Record Length)** — 주어진 신뢰수준에서 SR>SR* 를 주장하는 데 필요한 최소 표본 길이. 현재 n 이 MinTRL 미만이면 "표본 부족" → SHADOW 신호.
  → 출처: Bailey & López de Prado (2014).
- **bootstrap CI95** — 부트스트랩으로 Sharpe diff / avg PnL 의 CI95(블록 부트스트랩이면 시계열 자기상관 보존). CI95 low 가 양수면 강한 신호. **실재 적용처는 `ops/policy_gate.py` PROMOTE_PAPER 게이트**(replay active 의 avg_pnl CI low>0 조건, L80). 주의: C3(bull_all p99.95 cap2) 채택 근거는 bootstrap CI 가 아니라 **fold-level 양수 EV**(RESEARCH.md §4.3: 3/4 active fold +7.40%, 2024 −0.89% resilient, no_trade 1/5). `scripts/bootstrap_edge_v1.py` 는 distribution_beta vs setup_momentum 의 Sharpe-diff / fractional sizing 진단용(현 산출물 CI95=[−0.049, +0.416], P(diff≤0)=0.071)이며 C3 검증용이 아니다.

## 4. 상대 지표 (vs BTC HODL — 사후 표기)
- **Beta** = cov(r, r_btc)/var(r_btc)
- **IR (Information Ratio)** = mean(r − r_btc)/std(r − r_btc) × √N
- **Tracking Error** = std(r − r_btc) × √N
benchmark = BTC buy-and-hold. 알트 레이더가 단순 BTC 보유 대비 초과수익이 있는지.

## 5. 출처
Sharpe (1966) · Sortino & Price (1994) · Young (Calmar, 1991) · Martin (Ulcer, 1987) · Rockafellar & Uryasev (CVaR, 2000) · Treynor & Black (IR, 1973) · **Bailey & López de Prado (PSR/DSR/MinTRL, 2014)** · Efron (bootstrap, 1979) · pyfolio · quantstats. → 출처는 dashboard References 섹션(`soccz.github.io/projects/prelude/dashboard/index.html`, 11개)과 동일 계열이며, 여기 §5 는 metrics.md 가 다루는 지표 출처만 나열(dashboard 의 PBO — Bailey, Borwein, López de Prado & Zhu 2014 — 는 제외).
