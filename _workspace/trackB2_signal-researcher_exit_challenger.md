# 연구 노트: exit-discipline 챌린저 (R1 진입 고정, 청산만 sweep) — track B2

## 가설
R1 챔피언의 **똑같은 top-3 진입을 고정**하고, 고정 -3%SL/+5%TP/EOD 청산 대신 더 똑똑한
**청산 규칙만** 바꾸면 net·하방이 개선된다. 진입을 안 바꾸므로 R2(랭킹 변형)가 빠졌던
degeneracy("안 움직이는 대형주로 도망")가 **구조적으로 불가능** — 순수 EXIT 효과만 격리.
배경: 프로젝트 핵심 진단 = 진입엣지 real(lift 3-6x)인데 net breakeven~음수 = pump-then-dump.
진짜 lever 는 entry 게이트가 아니라 EXIT/하방 규율.

## 무엇을 돌렸나
- **스크립트**: `scripts/exit_challenger_compare_v1.py` (신규).
- **진입(FIXED, 재사용)**: `output/r2_challenger_picks_v1.csv` 의 policy=R1_ratio
  (1531 picks / 765일 OOS / top-3/day, 2024-04-03..2026-06-01). 진입 = day-D open =
  `[09:00 UTC D, 09:00 UTC D+1)` 윈도 첫 15m 봉 open. **실측 확인**: 09:00 UTC 15m open ==
  d1 day-open(예 BTC 141322000) byte-aligned. baseline 룰이 R1 기존 net(-0.00284/Sharpe -2.09/
  cum -0.864)을 **재현** → 진입·경로 정합성 검증됨.
- **청산 룰 6개 sweep**(같은 진입에 15m 경로로 realized net 재계산, 왕복 0.15% 차감):
  baseline(-3%/+5%/EOD), trail(+3% arm 후 고점-2% 트레일+3% floor), time(첫 1h time-stop),
  vol(SL=clip(2·ATR,1.5%,5%)/TP=clip(4·ATR,3%,10%)), vol_tight(vol 이되 SL cap=3%),
  regime(bear_volatile 타이트+1h, bear_quiet 2h, bull 느슨).
- **지표**: net_mean·Sharpe·Sortino·MaxDD·cum·hit·CVaR95 + deep-loss 2종
  (실현 net≤-5%/-10% **및** 장중저가≤-5%/-10% — SL floor 무관 진짜 노출).
- 산출: `output/exit_challenger_compare_v1.csv`(룰별+Δvs baseline), `output/exit_challenger_trades_v1.csv`(9186 trade dump).

## leak·시간정합성 방어 (§1 — 양보불가, 이 축 특히 주의)
- **입력 ≤ t-1 / 진입 leak-free**: 진입 집합 = R1(이미 D-1 feature·purged WF·OOF calib).
- **청산 결정 causal**: 트레일링=현재까지 running_high(미래 고점 X)+arm 도 현재까지 도달여부,
  time-stop=경과 봉 수, vol=진입 시점 **D-1 ATR**(`f_atr_pct_14` shift(1), 진입과 같은 정보집합).
  같은 봉 SL·TP 동시 → SL 먼저(보수). 시간 오름차순 walk → 미래 봉 미리보기 0.
- **causal-exit 자가검증(leak 라인)**: `assert_causal()` 가 300개 랜덤 픽에 대해 각 룰을
  **봉을 한 개씩 늘려가며(prefix-only)** 재생성한 청산 시점이 전체 경로 결과와 동일한지 확인.
  **prefix-replay mismatch = 0** (어느 룰이든 미래 봉을 봤다면 prefix 결과가 달라짐).
- **비용 항상 차감**(0.15%), gross/IS 금지. day-D 경로는 in-trade outcome(leak 아님).

## 시도 조합 수 (selection deflate 용)
- 청산 룰 **6개**(baseline + 5 변형). hand-pick 아님 — vol_tight 만 1차 결과(vol deep-loss 폭증)
  본 뒤 하방-우선 변형으로 추가(데이터 본 뒤 추가 = bias 명시). 진입/fold/embargo/head 는 R1 공유(추가 자유도 0).
- 룰 내부 파라미터(arm 3%, trail 2%, time 1h, ATR×2/×4, clip bound)는 §2.5 placeholder.

## 1차 결과 (OOS 765일, net 0.15% 차감, 15m 경로)

| 룰 | n | net_mean | Sharpe | Sortino | MaxDD | cum | hit | net≤-5% | **장중저가≤-5%** | 장중≤-10% | %SL | %TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **baseline (R1 청산)** | 1531 | -0.00284 | -2.09 | -2.84 | -0.872 | -0.864 | 0.39 | 0.000 | **0.020** | 0.0013 | 0.456 | 0.219 |
| trail | 1531 | -0.00300 | -2.41 | -3.31 | -0.896 | -0.890 | **0.46** | 0.000 | 0.017 | 0.0013 | 0.402 | 0(trail .32) |
| time (1h) | 1531 | -0.00293 | -4.41 | -5.32 | -0.863 | -0.858 | 0.34 | 0.000 | **0.015** | 0.0007 | 0.106 | 0.065 |
| **vol (ATR scale)** | 1531 | **-0.00260** | **-1.47** | **-2.04** | -0.884 | -0.872 | 0.40 | **0.286** | **0.285** | 0.0033 | 0.288 | 0.107 |
| vol_tight (SL cap 3%) | 1531 | -0.00315 | -1.93 | -2.79 | -0.895 | -0.889 | 0.34 | 0.000 | 0.024 | 0.0013 | 0.501 | 0.088 |
| regime | 1531 | -0.00311 | -2.48 | -3.38 | -0.901 | -0.893 | 0.36 | 0.000 | 0.031 | 0.0007 | 0.306 | 0.101 |

### net 이 양수로 가는 룰이 있나?
**전체 OOS 에선 없다 — 모든 룰 cum 음수(-43%~-89%), net_mean 음수.** exit tweak 만으로 net 흑자
전환은 안 된다(프로젝트 메모리와 일치: "진입엣지 real 이나 자동 net 미확립"). 단 **부분적으로**:
- **vol 이 유일하게 net_mean·Sharpe·Sortino 개선**(net -0.00284→-0.00260, Sharpe -2.09→-1.47,
  Sortino -2.84→-2.04). 그리고 **bear_quiet regime 에선 net-positive**(cum +0.051, Sharpe +0.61).
- ★ **그러나 하방-우선 위반**: vol 의 net 개선은 calm 코인 SL 을 5%까지 **넓혀서** 나온 것.
  장중저가≤-5% 노출이 0.020→**0.285**(14배), 실현 net≤-5%도 0→0.286. 사용자(하락최소화>상승)에겐
  나쁜 trade-off. **vol_tight**(SL cap=3%)로 deep-loss 를 baseline 수준(0.024/net≤-5%=0)으로 막으면
  net 이익이 사라짐(net -0.00315, baseline 보다 나쁨) → **vol 의 net 엣지 = 전적으로 "더 깊은 손실 수용"의 대가**.

### best exit 룰 선정 (net 1순위, 동률이면 하방)
- **순수 net/Sharpe 기준이면 vol**(유일한 개선). 하지만 deep-loss 폭증 = 하방-우선 사용자와 충돌.
- **하방-우선 기준이면 baseline(현 챔피언)이 사실상 최선** — net 개선 룰(vol)은 하방을 악화시키고,
  하방을 지키는 룰(vol_tight/time/regime)은 net 을 개선 못 함. time 은 deep-low 를 약간(0.020→0.015)
  줄이나 Sharpe 가 -4.41 로 붕괴(1h 컷이 winner 못 키우고 chop 손실만 실현).
- **regime 은 reject**: 타깃이던 bear_volatile 개선 미미(Δnet +6e-05), bear_quiet/bull_volatile 악화.
- **결론**: **이 고정 진입 위에서 청산 tweak 만으로는 "net 개선 + 하방 유지"를 동시 만족하는 룰 없음.**
  net lever(vol)와 downside lever(baseline)가 상충. 진짜 net 개선은 진입 quality(R1 의 pump-then-dump
  꼬리) 자체를 손봐야 할 가능성 — exit 만으론 한계.

## challenger 배선 (코드만, 활성화 X)
- **신규**: `signals/exit_rules.py` — `simulate_exit(...)`(파라미터화 청산) +
  `rule_params(exit_rule, regime, atr_pct)` + `simulate_exit_by_rule(...)` + `DEFAULT_EXIT_RULE='vol'`.
  self-contained, 챔피언 close 모듈 import 안 함.
- **`signals/model_registry.py`**: `recommend_r1_exitsmart_open` ModelSpec 추가.
  진입=`score_candidates`(ranking 기본=R1, **R1 과 byte-identical**), 자기 ledger
  `output/shadow_ledger_recommend_exitsmart.csv`, **challenger_only=True**(승격 금지).
- **챔피언 close 불변 확인(실측)**: `git status` 에서 `close_recommend_ledger.py` /
  `recommender_downside_exit_v1.py`(simulate_path -3%/+5%) / `recommend.py` **무변경**.
  exit_rules 모듈은 별도 파일 — 챔피언 -3%/+5%/EOD 경로 완전 보존. daily_run/텔레그램/champion_state 미터치.
- **모듈==백테스트 동일성 검증**: exit_rules.simulate_exit_by_rule('vol') 이 backtest trades dump 와 5/5 net 일치.

## evaluator 가 봐야 할 것 (3줄)
1. **net vs 하방 상충의 채택 논리**: vol 이 유일하게 net/Sharpe/Sortino 개선하나 장중 deep-loss
   노출 14배(0.020→0.285) — 하방-우선 사용자엔 독. 이 trade-off 에서 SHADOW 채택 가치가 있는지
   (bear_quiet net-positive 부분만 쓸지), bootstrap CI95(Δnet, Δdeep-loss)로 765일 차이가 유의한지.
2. **baseline 이 사실상 최선이라는 negative 결과**: "고정 진입 위 exit tweak 만으론 net+하방 동시개선 불가"가
   진짜인지(=net lever 는 진입 quality 에 있음) 적대 검증. time 룰 Sharpe -4.41 붕괴가 1h 컷의 구조적
   문제인지 파라미터(1h) 문제인지.
3. **causal-exit/leak 감사**: prefix-replay mismatch=0(미래 봉 미리보기 0) 주장 적대 검증 + vol 의
   D-1 ATR 이 진입과 같은 정보집합(leak 아님)인지 + 6-rule selection deflate(vol_tight 는 데이터 본 뒤 추가).
