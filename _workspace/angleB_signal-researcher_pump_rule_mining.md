# 각도 B — 일봉 급등(>=20%/>=15%) 선행 룰 채굴 (rule_mining)

## 가설
"왜 이 코인이 급등했나"를 D-1 이전 데이터로 역분석 → 해석 가능한 선행 룰. 기존 S01~S03 보다
recall 을 높이는 룰을 찾는다. 사용자 가설(range_contraction=조용해진 후 폭발)을 일봉/4h 에서 재검토.

## 라벨 (미래 = day D)
- pump20 = (high_D/open_D - 1) >= 0.20  (주), base rate ~1.9% coin-day
- pump15 >= 0.15  (보조), base rate ~3.6%

## leak 방어 (양보 X — 이 프로젝트 same-day leak 2번 전적)
- 라벨은 day D 의 high/open. **피처행 날짜 t 에 label(t+1) 을 붙여 feature_date = D-1** (distribution engine 과 동일 leak-fix).
- same-day(D) high/low/close 절대 미사용. next_* / LEAK_COLS / pump* 전부 학습에서 제외.
- 유니버스 거래대금 rank 도 feature_date(D-1) 기준.
- 4h range_contraction 은 그 KST 날 마지막 바(21:00)까지로만 계산 → +1 shift 후 D-1.
- per-fold: train-only 로 tree fit + leaf rule 추출, val 에서 lift 검증 (PurgedWalkForward 5-fold, embargo 10d, holdout 180d).

## 시도 조합 수 (selection bias 기록)
- feature 수: 일봉 46개 (+ 4h 4개 = 50). depth<=3, min_leaf=300, class_weight=balanced.
- target 2개 (pump20/pump15) × 4h on/off × 5 fold = fold당 1 tree → 총 20 tree, leaf rows 80.
- hand-pick 은 OOS(val) lift + gap<10pp 본 뒤. threshold 는 tree 가 데이터에서 자동 선택.
- 스크립트: scripts/pump_rule_discovery_v1.py / 결과: output/pump_rule_discovery_v1.csv

## 1차 결과

### 지배적 선행조건 = high-ATR + 7일 모멘텀 (continuation), range_contraction 아님
- pump20 cross-fold robust root: **roc_7d (rank) > 0.85** — 3 fold 에서 mean_lift 3.35, recall 11.8%, fire 4.7%.
- pump15 high-recall leaf: **atr_pct_14(raw) > 0.070 AND roc_7d(rank) > 0.854 AND log_return_1d(raw) <= 0.117**
  → val recall 21.3%, lift 2.1, fire 10.0%, gap +1.2pp (overfit 아님). fail 시 eod -1.6%.
- pump20 high-recall leaf: **log_return_1d <= 0.048 AND atr_pct_14 > 0.045 AND roc_7d > 0.800**
  → recall 25.3%, lift 2.0, fire 12.6%, gap -1.9pp. fail eod -0.25% (잡았는데 안 오를 때 손실 작음).

피처 정규화 주의: atr_pct_14, log_return_1d 는 **raw**; roc_7d/return_5d/vol_inv_7d/range_contraction_14d 는 **cross-sectional rank(0~1)**.

### range_contraction 가설 — 일봉/4h 둘 다 반증
- 4h rc4h_3/7/14d 피처(coverage 96~100%, median~1.05, p90~2.2) 추가했으나 **tree 가 한 번도 선택 안 함** (pump15 trees 에 rc4h 0회 등장).
- 일봉 range_contraction_14d 가 등장한 leaf 에서도, **lift 높은 가지는 rank<=0.526 (= 최근이 더 넓음/수축 아님)**. 수축(>0.526) 가지는 lift 낮음.
- → 사용자 "조용→폭발" 가설은 >=20%/>=15% 일봉 급등에 대해 **데이터로 반증**. 오히려 "이미 변동성 큰 + 모멘텀 살아있는" 코인이 더 오른다 (RESEARCH §7.2 와 일관, 4h scale 에서도 안 살아남).

### recall 천장 진단 (이번주 실제 급등 D-1 사전점검 — leak-free)
7개 실제 급등 케이스에 D-1 피처로 룰 적용:
- **사전 깃발 세움(1/7): KRW-WLD 05-30 (+31%)** — p15+p20 둘 다 발화. atr 0.125, roc7 rank 0.93, lr1 0.033, liq_rank 6. 이게 이번주 룰이 미리 잡았을 케이스.
- 놓침(6/7):
  - ID 05-30(+38.7%): lr1=0.345 (전일 +34%) → "미소진" 필터(<=0.117/<=0.048)에 걸려 제외. day-after 연속펌프 패턴(룰이 의도적으로 배제).
  - VTHO/ERA/ID 05-29: roc_7d rank 0.25~0.66 (모멘텀 cold) → 모멘텀-continuation 룰이 구조적으로 못 잡음 (cold-start 펌프).
  - XLM(+32%): roc rank 0.97 높지만 atr 0.047 < 0.070 변동성 게이트 미달.
- **결론**: 모멘텀-continuation 룰은 "이미 움직이는" 펌프(WLD/ID 05-30 타입)는 잡지만 **cold-start 펌프(ID 05-29, VTHO)는 구조적으로 미스**. 기존 setup recall 천장과 같은 한계. cold-start 는 다른 각도(거래량 미세동학/breakout 등) 필요.

## quant-evaluator 가 검증해야 할 지점
1. **leak 재감사**: feature_date=label_date-1 shift 가 모든 경로(일봉+4h merge)에서 D-1 보장되는지. atr_pct_14/log_return_1d 가 raw 라 D 가격이 새지 않았는지.
2. **EDA hit ≠ Sharpe 함정**: recall 21~25%, lift 2.0 는 방향성. fail 시 eod -0.25~-1.6% 는 작아 보이나, 실제 진입가(09:00 open) 대비 TP(+15/20%) 도달 전 SL 경로를 모델링해야. ledger Sharpe 로 환산 필요.
3. **fire rate 10~13%**: 유니버스 100 중 매일 10~13개 깃발 → S01~S03(05-29 13개) 대비 recall 개선이나, 너무 많이 울리면 selection 후 top-K 필요. lift_val_top(top100 내) 는 lift_val 보다 낮음 — 대형주에선 lift 약화 확인 요망.
4. **selection bias deflate**: leaf 80개 중 hand-pick. fold당 1 tree라 다중검정 폭은 작으나, gap<10pp 필터 통과한 것만 robust 주장.
5. **cold-start 미스**: 이 룰셋 단독으로는 유니버스밖/cold 펌프 못 잡음. 포트폴리오 내러티브에서 "momentum-continuation entry" 로 한정.
