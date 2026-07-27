# 연구 노트: C2 dip-buy (oversold-bounce) challenger vs R1 champion

생성: 2026-06-01 · signal-researcher · 채택결정 X (quant-evaluator 판정 대상)

## 가설
모멘텀(R1)이 죽는 구간 — 깊은 하락 후 RSI 낮고 7d-low 에서 *반등이 막 시작*된 코인 — 에서
mean-reversion 반등 엣지가 존재할 것이고, 진입 성격이 R1(상승 따라가기)과 직교라 net 분포·
R1 상관이 달라 분산 포트 가치가 있을 것이다.

## 무엇을 돌렸나
- 스크립트: `scripts/cc_dipbuy_v1.py` (self-contained, prelude 내부 import only).
- 유니버스: R1 과 동일 top100(D-1) → **그 위에 과매도 게이트로 subset 필터** (R2 와의 핵심 차이:
  R2 는 같은 행 재정렬이라 R1 천장 공유, C2 는 entry 유니버스 자체를 oversold 로 교체).
- 과매도 게이트(전부 D-1 feature): `f_rsi_14 <= cut` × `f_ret_7d <= cut` × `f_bounce_off_7d_low ∈ [0.01, 0.12]`
  (bounce band = 바닥은 아니되 반등 끝도 아닌 *반전 초입* — 떨어지는 칼 회피 의도).
- head 모델: R1 과 동일 빌더 재사용(`downside_head_riskreward_v1.walk_forward_heads`, XGB, random_state=42,
  per-fold OOF bucket calibration). subset 안에서 rr-ratio 정렬 / bounce확률(p_up_05) 정렬 둘 다.
- 검증: purged WF 6-fold, embargo=5, OOS 2024-04-03~2026-06-01 (765일, R1 과 동일 fold).
- 청산/평가: **R1 과 100% 동일** — 15m SL -3% / TP +5% / EOD, net 왕복 0.15% 차감 (`simulate_path`).
- 산출: `output/cc_dipbuy_compare_v1.csv` / `_corr_v1.csv` / `_regime_v1.csv` / `_picks_v1.csv` / `_coverage_v1.json`.

## leak·시간정합성 방어 (§1 체크리스트)
- 과매도 게이트 입력 = `build_market_features` 의 `.shift(1)` feature (D-1 까지). LEAK_COLS/next_* 제외.
- 라벨(up/down/bounce) = day-D open 대비 high/low/close (미래, 타겟). feature 와 시점분리.
- head 학습/calibration = expanding train(과거 fold)에서만 fit, test fold 적용만. OOF bucket calib.
- 게이트는 D-1 feature 의 결정론적 필터 → 새 leak 유입 구조적 0.
- 15m 경로청산 = 진입일 D in-trade outcome (진입결정은 D-1 까지) → leak 아님.
- ★ R1_champion 행(n=1531, net_mean -0.00284, hit 0.391, Sharpe -2.50)이
  `r2_challenger_compare_v1.csv` 의 R1_ratio 행과 **byte-identical** → 챔피언을 동일 fold 로 재현,
  새 leak 미유입 확인. "성능 너무 좋으면 leak" 규칙: C2 는 오히려 R1 보다 *나쁘므로* leak 의심 불필요.

## 시도 조합 수 (selection deflate 용)
- 게이트 격자: RSI {40,35,30} × ret7d {-0.08,-0.12,-0.18} = 9 subset × 정렬 2종(rr/bounce) = **18 C2 변형**.
- + R1 baseline 1 = 총 19 정렬키 비교 (coverage.json `n_combos_tried`).
- bounce band [0.01,0.12] 은 hand-set (사전, 데이터 보기 전 — 반전 초입 정의). sweep 안 함.
- ⚠ 18 변형 중 best 를 뽑는 것 = selection. 그런데 **18개 전부 net 음수**라 deflate 무의미(아래).

## 1차 결과 (OOS net 0.15% 차감, top-3, day-equal-weight)

| | n | %SL | noSL_deep | net_mean | hit | Sharpe | Sortino | MaxDD |
|---|---|---|---|---|---|---|---|---|
| **R1 champion** | 1531 | 0.456 | 0.135 | **-0.00284** | 0.391 | **-2.50** | -4.49 | -0.872 |
| C2 best(net) rsi30/ret7-8/rr | 869 | 0.513 | 0.161 | **-0.00252** | 0.387 | -1.82 | -3.99 | -0.785 |
| C2 worst rsi40/ret7-18/bounce | 571 | 0.595 | 0.198 | -0.00495 | 0.349 | -3.33 | -8.33 | -0.772 |

- **net 양수? 아니오.** R1·C2 18변형 **전부 net_mean 음수** (-0.0025 ~ -0.0049). 이 OOS 기간엔
  진입엣지가 net 흑자로 실현되지 않음(라이브 ledger 가 보던 것과 일관 — radar지 strategy 아님).
- **R1 우위? 부분적 X.** C2 best(net) -0.00252 가 R1 -0.00284 보다 *명목상 덜 음수*지만,
  표본이 R1 1531 vs C2 869 로 다르고 둘 다 음수 → "C2 가 net 으로 R1 을 이긴다"고 말할 수 없음.
  Sharpe 도 C2 best -1.82 가 R1 -2.50 보다 덜 음수지만 둘 다 강한 음수.
- **저상관? 아니오(분산 가치 약함).** C2 변형의 R1 일별net 상관 = **+0.21 ~ +0.48** (전부 양의 중상관).
  가장 낮은 게 rsi35/ret7-18 ~ +0.21. dip-buy 가 R1 과 *직교일* 거란 가설은 **반증** —
  과매도 코인도 결국 같은 시장 하락일에 같이 깨져서(특히 bear) net 이 동조. 진입신호는 반대지만
  실현 손익경로는 시장베타에 동조 → 음의/무상관 hedge 못 됨.
- **deep-loss 폭증? 예 (사용자 하방선 우려 현실화).** "떨어지는 칼" 검사:
  - %SL(라이브 -3% stop-out 빈도): R1 0.456 → C2 0.51~0.60 으로 **상승**. 게이트 깊게(ret7<=-0.18)
    갈수록 0.59 까지. 과매도 코인이 다음날 더 자주 -3% 손절선을 친다.
  - noSL_deep (SL 끄면 EOD net<=-5% 빈도 = 본질 하방): R1 0.135 → C2 0.155~0.211. 깊은 게이트일수록
    0.21 까지 = **본질적으로 더 깊이 빠짐**. 하드 SL 가 net deep_loss_freq 는 0 으로 floor 하지만,
    SL 가리기 전 실제로는 R1 보다 하방이 나쁨 → **dip-buy = 떨어지는 칼 가설 데이터로 확인**.

### regime 분해 (best C2 rsi30/ret7-8/rr vs R1)
| regime | R1 net | C2 net | R1 noSL_deep | C2 noSL_deep | 비고 |
|---|---|---|---|---|---|
| bear_quiet | -0.00250 | -0.00171 | 0.154 | 0.099 | C2 가 R1 보다 덜 음수+하방 양호(둘 다 음수) |
| bull_quiet | -0.00247 | -0.00150 | 0.120 | 0.184 | C2 net 덜 음수지만 하방은 더 나쁨 |
| bear_volatile | -0.00357 | -0.00423 | 0.162 | 0.149 | **C2 가 R1 보다 더 나쁨** (변동 약세 = 칼) |
| bull_volatile | -0.00259 | +0.00527 | 0.083 | 0.296 | C2 양수지만 **n=27 (11일)** — 표본 무의미 |
- 유일한 양수 net regime(bull_volatile)은 n=27 로 신뢰불가. bear_volatile(시장 약세 변동)에서 C2 가
  R1 보다 더 깨짐 = dip-buy 의 최대 약점 구간. "C2 가 R1 을 보완하는 regime" 은 신뢰 표본에선 없음.

## evaluator 가 볼 것 (3줄)
1. **net 음수·R1 저상관 실패가 결론 (가설 반증)**: 18 변형 전부 net<0, R1 상관 +0.21~+0.48(직교 아님),
   deep-loss(특히 noSL_deep, %SL) 가 R1 보다 폭증 → SHADOW도 과한 REJECT 후보. 본질 하방이 R1 보다 나쁨.
2. **표본/selection 함정 확인 요청**: best C2 는 18-way selection 산물이고 R1(n=1531) 대비 n 작음(869).
   bull_volatile 양수는 n=27 noise. C2 best net -0.00252 vs R1 -0.00284 의 "근소 우위"가 표본차/우연
   범위인지(부트스트랩 CI95, PSR/DSR with trials=18) 적대 검증 요청.
3. **leak 자가검증 통과**: R1 행이 r2 baseline 과 byte-identical(동일 fold 재현), 게이트는 D-1 결정론 필터,
   라벨 day-D, train-only calib. 성능이 R1 보다 *나쁘므로* leak 의심 불필요 — 그래도 시점분리 재확인 요청.

## 파일 경로 (전부 신규, 공유 라이브 파일 미편집)
- `/mnt/20t/prelude/scripts/cc_dipbuy_v1.py`
- `/mnt/20t/prelude/output/cc_dipbuy_compare_v1.csv`
- `/mnt/20t/prelude/output/cc_dipbuy_corr_v1.csv`
- `/mnt/20t/prelude/output/cc_dipbuy_regime_v1.csv`
- `/mnt/20t/prelude/output/cc_dipbuy_picks_v1.csv`
- `/mnt/20t/prelude/output/cc_dipbuy_coverage_v1.json`
