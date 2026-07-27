# 연구 노트: bear_quiet 펌프 후보 — 15m 경로순서 청산 백테스트 (A)

## 가설
bear_quiet regime(BTC D-1 ma_distance(200d)<0 & rv30 252d-intensity<=0.5) 날, D-1
cross-section top-decile 선행패턴(qv_surge_7d 1순위 + qv_surge_30d/bounce_off_7d_low/
atr_xs_decile) 코인을 day-D open 진입하면, 일봉에서 미상이던 intrabar 청산 경로를
15m으로 풀면 net 양(+)의 엣지가 특정 청산정책에서 살아남는다.

## 무엇을 돌렸나
- scripts/bear_quiet_path_exit_v1.py (신규, self-contained)
- d1 panel = build_market_features + add_cross_sectional (기존 leak-free 빌더 재사용),
  bear_quiet flag = attach_bear_quiet(BTC regime D-1 shift)
- 진입 = bear_quiet test fold에서 train-fold quantile(0.90) top-decile (OOF cutoff)
- 경로 = 15m DB [09:00 D, 09:00 D+1) ~96봉을 시간순 walk
  - 15m 봉 aggregate가 d1 캔들과 정확히 일치 검증(open/high/low/close 동일, 96봉) → grid 정렬 OK
- 청산 그리드: SL{none,-3,-5,-8%} × TP{none,+5,+8,+10,+15%} × time-stop{none,2/4/8h} ×
  trailing{none,-5/-8/-10%} = 320정책 × 후보4 = 1280 (policy×pattern)
- 산출: output/bear_quiet_path_exit_v1.csv(1280행), _what_hit_first_v1.csv, _coverage_v1.csv

## leak·시간정합성 방어
- 진입결정(코인/날): feature는 build_market_features shift(1)로 D-1까지, regime은
  attach_bear_quiet에서 btc_regime.shift(1) → day D는 D-1 regime을 본다. top-decile
  cutoff는 train fold quantile(OOF) — test fold에 누수 X.
- 경로: in-trade outcome(leak 아님). simulate_path는 15m 봉을 i 오름차순으로만 순회,
  미래 봉 미리 안 봄. 같은 15m 봉 TP&SL 동시 → SL 먼저(보수 유지).
- 거래비용 0.15% 왕복 차감(net=gross-0.0015). spot-only(공매도 X).

## 시도 조합 수 (selection deflate용)
1280 (policy 320 × candidate 4). hand-pick 없음 — 그리드 전수, 후보4 사전지정.
candidate cutoff quantile 0.90 고정(sweep 안 함).

## ★★ 핵심 결과 (deduplicated coin-day, 0.15% 차감, periods/yr=365)

### what-hit-first (TP+10 vs SL-5 둘다 닿은 날, 15m 경로순) — sl_first 비관 정당한가?
| 후보 | both | tp_first | sl_first | sl_first_share |
|---|---|---|---|---|
| qv_surge_7d | 105 | 48 | 57 | 0.54 |
| qv_surge_30d | 129 | 69 | 60 | **0.47** |
| bounce_off_7d_low | 144 | 69 | 75 | 0.52 |
| atr_xs_decile | 87 | 42 | 45 | 0.52 |
→ 일봉 백테스트의 "SL이 항상 먼저"(sl_first 100% 가정)는 **과도하게 비관적**. 실제론
  대략 반반(0.47~0.54). 일봉 sl_first bound가 음 Sharpe였던 건 이 비관가정 탓.

### headline 정책 (DEDUP, net/trade·Sharpe·hit·cum·MaxDD·n)
| 후보 | 정책 | net/trade | Sharpe | hit | cum | MaxDD | n |
|---|---|---|---|---|---|---|---|
| qv_surge_7d(1순위) | EOD | -0.02% | -0.85 | .43 | -9.2% | -23.9% | 1239 |
| qv_surge_7d | **TP+10/noSL/EOD** | **+0.29%** | **+0.52** | .45 | +2.1% | -18.7% | 1239 |
| qv_surge_7d | SL-5/TP+10 | +0.07% | -1.92 | .42 | -13.0% | -24.5% | 1239 |
| qv_surge_7d | ts4h/TP10 | -0.13% | -4.47 | .39 | -16.5% | -22.2% | 1239 |
| qv_surge_30d | TP+10/noSL | +0.16% | +2.00 | .43 | +13.1% | -10.8% | 1255 |
| qv_surge_30d | TP+5/noSL | +0.06% | +1.00 | .49 | +4.6% | -13.2% | 1255 |
| bounce_off_7d_low | TP+5/noSL | +0.06% | +2.86 | .51 | +19.5% | -9.0% | 1143 |
| bounce_off_7d_low | TP+10/noSL | -0.03% | +2.00 | .43 | +16.5% | -10.0% | 1143 |
| bounce_off_7d_low | EOD | -0.26% | -0.03 | .41 | -3.5% | -20.6% | 1143 |
| atr_xs_decile | TP+5/noSL | +0.23% | +2.47 | .52 | +12.0% | -9.2% | 1224 |
| atr_xs_decile | EOD | -0.25% | -2.00 | .42 | -14.2% | -21.5% | 1224 |

### 정책 lever 결론
1. **하드 SL은 모든 후보에서 net을 악화시킨다.** qv_surge_7d: SL 없으면 Sharpe
   +0.52, SL-5 붙이면 -1.92. SL이 펌프 직전 wiggle에 털려 EOD 회복분을 못 먹음.
   → 일봉 "하드 SL-5" 비관 시나리오가 음 Sharpe였던 진짜 원인 = SL 자체가 손해(경로상
   SL이 먼저 닿는 비율이 절반 이하인데도 -5% 확정청산이 EOD 평균회복을 깎음).
2. **time-stop(2/4/8h)도 악화.** atr_xs ts2h Sharpe -2.78, qv7 ts4h -4.47. 펌프는
   오후~밤(15~20h hold)에 완성 → 일찍 자르면 손해.
3. **trailing도 net 깎음**(qv7 trail-10 Sharpe -1.22). 변동성 큰 펌프 후보라 trailing이
   잡음에 털림.
4. **EOD-only(무TP)는 전 후보 음(-).** TP cap이 필수 — 펌프는 종가까지 못 버티고 되돌림.
5. **승급 lever = TP cap + 무SL + EOD fallback.** 후보별 최적 TP: qv7=+10%, qv30=+10%,
   bounce/atr=+5%.

## ★ evaluator 검증 요청 (이게 PORTFOLIO_GRADE 막을 수 있는 지점)
1. **표본 시간집중 = 최대 약점.** 15m DB 시작 2023-05-03이지만, 경로검증 가능한
   bear_quiet 진입은 **2026-01~05 단일 블록에 집중**(bounce/atr는 54일 전부 2026년,
   qv7/qv30은 56일이 2024-07~2026-05에 분산되나 대부분 2024-09/10 + 2026-01~05 두
   pocket). **독립 OOS fold가 사실상 1개**(최근 강세 펌프장). Sharpe +2~2.8은 단일
   regime block 성과 — out-of-block 재현성 미검증.
2. **fold-overlap n 부풀림 주의.** output/bear_quiet_path_exit_v1.csv의 n≈3700은
   walk-forward test 윈도우 겹침으로 같은 coin-day를 최대 3× 중복. **정직한 n은
   dedup coin-day ≈1140~1255**(위 표는 dedup). CSV의 Sharpe/cum은 같은 블록 3중계상
   → 3개 독립관측 아님.
3. **고Sharpe-저mean illusion.** bounce_off/qv30 TP+5는 net/trade +0.06%(왕복비용
   0.15% 겨우 상회), Sharpe 2~2.9는 tight-TP가 일변동성을 압축한 결과. 회전율 높고
   마진 얇음 — 슬리피지/체결 가정 민감. qv_surge_7d(1순위)만 net/trade +0.29%로 두꺼움
   (단 Sharpe +0.52, MDD -18.7%로 약함).
4. **1순위 qv_surge_7d 검증 우선** — 일봉서 유일 5-fold 전부 양(+)이었던 후보. 경로상
   여전히 marginal 양(net +0.29%/trade, Sharpe +0.52). MDD -18.7% 큼.
5. 15m 봉 내부는 여전히 ambiguous(보수 SL-먼저). but TP cap 정책은 SL이 없어 봉내
   ambiguity에 거의 무관(TP 도달여부만, 보수 영향 작음).

## 코드/산출물 경로
- scripts/bear_quiet_path_exit_v1.py
- output/bear_quiet_path_exit_v1.csv (1280 policy×pattern)
- output/bear_quiet_path_what_hit_first_v1.csv
- output/bear_quiet_path_coverage_v1.csv
