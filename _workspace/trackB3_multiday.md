# 연구 노트: Track B3 — 멀티데이 홀딩 (R1 진입 N일 보유)

## 가설
R1 top-3 진입(1일 보유 + 하드 -3%SL/+5%TP/EOD)에서 winner 가 +5%TP 에 조기 절단되는
구조 때문에 6실험이 net 흑자전환 0 으로 수렴했다. **보유기간을 N일(2/3/5)로 늘려
수익꼬리에 시간을 주면** net 이 양수로 갈 수 있다는 우회 가설. 단 더 오래 노출 →
하방도 늘 수 있으니 net·하방 trade-off 를 정직하게 측정. (사용자: 하락위험 최소화 >
상승, 거래당 ~-5% 수용.)

## 무엇을 돌렸나
- 진입 집합: `output/r2_challenger_picks_v1.csv` 의 `policy=='R1_ratio'` = R1 daily top-3
  (1531 entries / 765 OOS days / 224 markets, 2024-04-03~2026-06-01). **재학습 안 함**(픽 고정).
- 청산 sim(일봉 d1, `scripts/ch_multiday_v1.py`): 진입가 = day-D 일봉 open. 3 변형 × N∈{1,2,3,5}:
  - (a) holdN: N일 후 종가
  - (b) bracket: N일 내 +TP/-SL 먼저 도달(동봉 동시→SL, 보수). TP∈{8,12,20%}, SL∈{5,8%}
  - (c) trail: N일 running-high 대비 -DD 트레일. DD∈{8,12%}
  - 총 변형 9개 × N 4개 = **36 조합**(작은 격자, hand-pick 후행 X).
- 비용: 왕복 0.15% 1회(멀티데이도 1 거래). 지표: net_mean·median·hit·deep_loss(<=-5%)·
  worst·CVaR95·MaxDD·cum·**block-Sharpe/Sortino**(유효표본 보정)·nominal-Sharpe·excess-over-market.
- 진단(`scripts/ch_multiday_diag_v1.py`): degeneracy(top-winner 제거), 베타분리, baseline 정합.

## leak·시간정합성 방어 (4/4)
- **시점 분리**: 진입 결정은 D-1 까지(R1 모델 피처 t-1, 픽 그대로 재사용 — 새 학습 0).
  진입가 day-D open 은 관측가능. 청산 경로 day-D~D+N-1 = 미래 outcome(정상 forward).
- **보수적 동봉 처리**: 같은 일봉에서 SL·TP high/low 둘 다 닿으면 나쁜 쪽(SL) 먼저.
  intrabar 경로를 모르므로 낙관 금지.
- **비용 차감**: 0.15% 1회, 전부 net 보고(gross 환상 X).
- **유니버스 시간정합성**: 픽 자체가 fold-train 종료시점 top100 으로 제한된 OOS(원 R1 빌더).
- **overlapping 표본 처리**: N일 홀딩은 인접 진입일이 시간상 겹침 → 명목 1531 trade 의
  유효표본 << 명목. 진입일을 N 간격으로 묶은 **non-overlapping block** 단위로 Sharpe/
  Sortino 재계산(N=2→383블록, N=3→255, N=5→153). 명목 Sharpe 는 낙관 편향이라 block 우선.
- **"성능 너무 좋으면 leak" 자가검증**: 결과가 좋기는커녕 전부 net 음수 → leak 알람 OFF.

## 시도 조합 수 (selection deflate 용)
36 조합(9 청산변형 × 4 N), 격자 사전고정·hand-pick 후행 없음. → evaluator 가 36 trials 로 deflate.

## 1차 결과 (OOS, net 0.15% 차감)

**핵심: 36 조합 중 net_mean>0 은 단 1개(N=5 trail_dd0.08, +0.0007)이고 그것마저 가짜다.**

| 변형 | N=1 | N=2 | N=3 | N=5 |
|---|---|---|---|---|
| holdN net_mean | -0.0035 | -0.0083 | -0.0083 | -0.0116 |
| holdN deep_loss(<=-5%) | 0.135 | 0.229 | 0.270 | **0.329** |
| holdN worst | -0.338 | -0.472 | -0.604 | **-0.630** |
| best-Sharpe(block) 셀 | trail0.12 -1.49 | trail0.08 -0.86 | trail0.08 -0.38 | **trail0.08 +0.17** |

- **net 은 N 이 커질수록 더 나빠진다**(holdN: -0.0035→-0.0116). 수익꼬리 신장 효과(<TP 막힘
  풀림)보다 손실꼬리 확대가 우세 — 1일 SL-floor 를 풀자 winner 가 더 달리기 전에 **loser 가
  더 깊이 빠진다**(worst -0.34→-0.63, deep-loss 0.135→0.329). 1일 구조의 양꼬리 동시절단을
  푸는 순간, 사용자가 가장 싫어하는 하방이 먼저 커진다.
- block-Sharpe 는 N 커질수록 "덜 나쁨"으로 수렴하지만 **부호는 여전히 음수**(N=5 trail0.08 만
  +0.17, 사실상 0). 이는 net 개선이 아니라 변동성 확대에 따른 비율 효과 + 표본 153 블록의 noise.

**유일 양수 셀(N=5 trail_dd0.08, net +0.0007)의 정직 해부 — 채택 불가:**
- median **-0.0165**, frac>0 **0.402** → 절반 이상 손실. 양수 mean 은 소수 극단 winner
  (top5 = +47.6/51.2/53.7/56.3/56.4%) 가 끌어올린 것.
- **top 1% 제거 시 net = -0.0035** 로 즉시 음수 → degeneracy(broad 엣지 X).
- **excess-over-market = -0.0069, picks 가 시장 이기는 빈도 41.9%**. 같은 N일 윈도우 시장
  바스켓(top100 EW)은 +0.0076 올랐는데 picks 는 그보다 못함 → 이 +0.0007 은 **상승장 보유
  프리미엄(베타)의 잔여**이며, picks 는 그 베타조차 underperform. 진짜 알파 아님.

**베타 vs 진짜엣지 분리(전 셀 공통):**
- 모든 셀에서 **excess-over-market < 0**(holdN N=1 -0.0049 → N=5 -0.0192). 즉 R1 picks 를
  N일 보유하는 것은 **같은 기간 시장 평균보다 항상 나쁘다**. net 의 N-의존 변화는 시장 베타
  변화를 따라갈 뿐, picks 고유의 멀티데이 알파는 음수.
- BTC 윈도우 보유수익은 ~0 → BTC 자체 베타는 중립. picks 의 손실은 alt 쪽 깊은 하방에서 옴.

## 결론 (증거 생성까지 — 채택 판정 X)
멀티데이 홀딩 축은 **net 흑자전환에 실패**. 1일 천장을 우회하려 보유기간을 늘리면
**수익꼬리보다 손실꼬리가 먼저·더 커져** net 이 악화되고 deep-loss 가 사용자 수용선을 넘는다
(N=5 deep-loss 33%). 메타-발견의 "양꼬리 동시절단" 진단과 일관 — SL-floor 를 풀면 절단되던
손실꼬리가 그대로 살아난다. 유일 양수 셀은 degeneracy + 마이너스 excess 로 가짜.
→ 이 축은 **R1 picks 위에서는 dead end**. (단 진입 집합을 바꾸면(예: 하방품질로 사전선별한
픽) 다를 수 있음 — 별도 축.)

## evaluator 가 볼 것 (3줄)
1. **베타 잔여 함정**: 유일 양수 셀 net+0.0007 의 출처가 (i) 상승장 보유프리미엄 잔여이고
   (ii) excess-over-market -0.0069/win-rate 41.9% 로 시장에 지는지 — 내 excess 계산(시장바스켓은
   비용 미차감 총수익 proxy)이 베타를 과소/과대 차감하지 않는지 검토.
2. **유효표본/deflate**: overlapping holding 의 block 압축(N=5→153블록)이 충분한지, 36-trial
   selection 과 합쳐 block-Sharpe +0.17 이 PSR/DSR 상 0과 구분 불가인지(거의 확실히 그러함).
3. **동봉 SL-우선 보수성 + 일봉 갭**: intrabar 미관측으로 SL 우선/트레일 트레일가 체결 가정이
   결과를 하방으로 과·과소 편향시키는지(특히 trail 변형), 일봉 갭다운(worst -0.63)이 현실
   슬리피지로 더 나쁠 수 있는지.

## 파일
- 코드: `/mnt/20t/prelude/scripts/ch_multiday_v1.py`, `/mnt/20t/prelude/scripts/ch_multiday_diag_v1.py`
- 결과: `/mnt/20t/prelude/output/ch_multiday_compare_v1.csv` (36행), `/mnt/20t/prelude/output/ch_multiday_picks_v1.csv` (trade-level, 대표 3변형 덤프)
