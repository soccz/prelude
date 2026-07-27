# 연구 노트: C4 — R1 챔피언 엔진 *내부* 하이퍼파라미터 sweep

작성: signal-researcher · 2026-06-01 · 핸드오프 → quant-evaluator

## 가설
"R1 챔피언이 net-음수인 이유가 *출력 픽 후처리*(이전 7실험)가 아니라 **엔진 내부 구성**
(피처셋·XGB head hp·유니버스 크기·calibration buckets·RR_EPS) 의 미세조정 여지라면,
엔진 자체를 sweep 해 net 또는 precision 또는 하방을 유의하게 개선하는 셀이 존재할 것이다."

## 무엇을 돌렸나
- 새 파일 `scripts/cc_r1_core_opt_v1.py` — 공유 라이브 파일(`signals/recommend.py`)
  **편집·READ 0**. 엔진 로직은 `downside_head_riskreward_v1.py`(leak-free 빌더·WF·OOF calib)
  + `r2_challenger_compare_v1.py`(15m 청산경로) 에서 *재사용/복제* 해 sweep 하네스 구성.
- baseline = 현 R1 (rank/XGB head P(>=10%)·P(<=-5%) + rr_ratio 정렬 + per-fold OOF
  bucket calib + 정적 top100 D-1 유니버스 + 15m SL-3%/TP+5%/EOD net 0.15%).
- **sweep 30 cells, one-axis-at-a-time off baseline (사전고정 격자)**:
  - A 피처셋(9): add(range_contraction/bb_squeeze/dist_high/qvspike/all_extra) +
    drop(xs_decile/rsi/streak/momentum_long) ablation.
  - B head hp(13): n_estimators{120,300}/max_depth{3,5,6}/lr{0.03,0.10}/
    min_child_weight{3,10,20}/reg_lambda{0.5,3,5}.
  - C universe(2): top50 / top150 (100=baseline).
  - D calib buckets(2): 8 / 15 (10=baseline).
  - E RR_EPS(3): 1e-2 / 1e-4 + tie-break off.
- 비교: 동일 OOS(1531 trade·702 날·765 OOS일 풀)·동일 15m 청산·동일 0.15% 비용에서
  net_mean·Sharpe·Sortino·MaxDD·hit·%SL·deep-loss(noSL)·precision@3(pump20)
  + **fold-level net 일관성**(fold_net_pos_frac / fold_net_min, overfit 진단).

## ★ 엔진 충실성 self-check (먼저 검증) — PASS
baseline 셀이 `output/r2_challenger_picks_v1.csv` 의 R1_ratio 와 **bit-재현**:
net_mean -0.00284 / n 1531 / hit 0.391 / %SL 0.456 / deepNoSL 0.135 / prec20 0.0366 /
Sharpe -2.50 — R2 challenger compare CSV 와 전 지표 일치.
→ 1차 시도에서 panel 을 m15_start 로 truncate 해 fold 경계가 밀려 net -0.00222·702일로
어긋났음(충실성 FAIL). **수정**: WF head 는 전체 date 범위에서 fit, m15_start 필터는
net 실현 시점에만(bars_map 키만 유지) 적용 → R2 와 정합. 이 수정 후 baseline 일치 확인.

## leak·시간정합성 방어 (§1 체크리스트)
- feature = `build_market_features` 의 market 별 `.shift(1)` (D-1 까지). LEAK_COLS/`next_*`
  제외(`_feats`/`feature_set` 에서 필터). 라벨 = day-D open 대비 high/low (미래 타겟) —
  train 에서만 fit, 학습 feature 에 안 섞임.
- calibration = **per-fold train OOF bucket** (`_oof_bucket_calib`, train-only). test fold 적용만.
- purged walk-forward 6-fold + **embargo 5**. 유니버스 = D-1 qv rank (`f_qv_rank`).
- 15m 경로는 진입일 D in-trade outcome (진입 결정은 D-1 까지) → leak 아님.
- 유니버스 sweep(top50/100/150) 은 동일 `f_qv_rank`(전체 panel 1회 계산) 에서 cap 만 변경 —
  top100 direct vs via150 row-identical 확인(시간정합성 유지).
- "너무 좋으면 leak" 자가알람: **발동 안 함** — 모든 셀이 음수(아래). 오히려 overfit 경계가 쟁점.

## 시도 조합 수 (selection deflate 용)
- **30 cells** (baseline 1 + A9 + B13 + C2 + D2 + E3). one-axis-at-a-time (full grid 아님).
- hand-pick 없음. 격자 사전고정. RR_EPS 2값·tie-break은 데이터 본 뒤 추가 X.
- **DSR/deflate 대상**: best 셀의 Δnet 우위가 30-cell 다중시행의 산물인지 evaluator deflate 필요.

## 1차 결과 (net 거래비용 차감, top-3, 15m SL/TP/EOD)
baseline: net -0.00284 / Sharpe -2.50 / Sortino -4.49 / MaxDD -0.872 / hit 0.391 /
%SL 0.456 / deepNoSL 0.135 / prec20 0.0366 / **foldPos 1/6 · foldMin -0.0051**.

축별 best (net 기준) Δ표:
| axis | best 셀 | net | Δnet(bps) | Sharpe | %SL | deepNoSL | prec20 | foldPos |
|------|---------|-----|-----------|--------|-----|----------|--------|---------|
| A 피처셋 | drop_streak | -0.00218 | +6.6 | -1.92 | 0.443 | 0.127 | 0.0392 | 1/6 |
| B head hp | max_depth=6 | -0.00227 | +5.7 | -1.98 | 0.449 | 0.129 | 0.0320 | 1/6 |
| C universe | top50 | -0.00227 | +5.7 | -1.96 | 0.481↑ | 0.170↑ | 0.0503 | 1/6 |
| D calib | bk15 | -0.00241 | +4.3 | -2.09 | 0.452 | 0.146↑ | 0.0333 | 1/6 |
| E RR_EPS | eps=1e-2/1e-4 | -0.00284 | +0.0 | -2.50 | 0.456 | 0.135 | 0.0366 | 1/6 |

**핵심 음성 결과 (유의 개선 셀 없음)**:
- **net>0 / Sharpe>0 / foldPos>0.5 인 셀 = 0개** (30/30 전부 음수·Sharpe~-2).
- 최대 net 개선 = +6.6 bps/trade (drop_streak), 그러나 30-cell net 분포 std=0.00042 기준
  baseline 대비 **+1.59σ** — 30회 시행에서 순수 noise 의 기대 최댓값(~2σ) 이내. 선택편향 산물 의심.
- 모든 best 셀이 **foldPos 1/6 유지**(reg_lambda=0.5 만 2/6, 그래도 net -0.00245 음수,
  foldMin -0.0054 로 최악 fold 더 깊음). 개선이 fold 전반에 일관되지 않음 → 한 fold 운.
- **C universe**: top50 은 net 약간↑지만 %SL 0.481·deepNoSL 0.170 으로 **하방 악화**
  (사용자 하방-우선 위배). top150 은 %SL 0.438·deepNoSL 0.112 로 하방↓지만 net -0.00320·
  Sharpe -2.84 로 net 악화 → net↔하방 trade-off, 둘 다 만족하는 점 없음.
- **E RR_EPS**: top-3 픽에서 floor 미발동 → baseline 과 완전 동일(insensitive). tie-break OFF
  는 net -0.00315 로 악화 → 현 하방-우선 tie-break 은 (미미하게) 도움.
- **A 피처셋**: 후보 추가(range_contraction/bb_squeeze/dist_high/qvspike) 전부 net 비개선
  또는 악화. drop_streak/drop_rsi 가 미세 net↑ — 노이즈 feature 제거 효과지 새 엣지 아님.
  (range_contraction 추가가 net 안 올림 → "조용→폭발" 일봉가설 또 반증, RESEARCH §3 일관.)

산출:
- `output/cc_r1_core_opt_compare_v1.csv` (30 cells × 지표 + baseline Δ)
- `output/cc_r1_core_opt_coverage_v1.json` (OOS 윈도·fold·cell 수·baseline cfg)
- `output/cc_r1_core_opt_run_v1.log` (per-cell 로그)
- `scripts/cc_r1_core_opt_v1.py` (sweep 하네스, self-contained)

## 결론 (채택신청 X — evaluator 판정 대상)
**엔진 내부 하이퍼파라미터 sweep 로는 R1 의 net-음수를 못 뒤집는다.** 30셀 중 net>0·
Sharpe>0·fold-일관 셀 0개. 개선처럼 보이는 셀은 +1.59σ(noise 대) 의 한-fold 운 →
DSR deflate 시 소멸 예상. R1 음수의 원인은 진입엣지가 아니라(엔진은 충실 재현됨) **일봉
진입 자체의 손익경로**(TP-before-SL·%SL 0.456) 에 있음 — 이전 결론(MEMORY: lever는
exit/하방 규율) 과 일관. 엔진 튜닝은 막다른 길로 보임.

## evaluator 가 볼 것 (3줄)
1. **DSR/deflate**: drop_streak 의 Δ+6.6bps(+1.59σ/30셀) 가 다중시행 deflate 후 살아남나 —
   bootstrap CI95(net_mean) 가 baseline 과 겹치는지(거의 확실히 겹침) 확정해줘.
2. **fold 일관성**: 전 셀 foldPos 1/6 — 개선 셀의 우위가 어느 fold/regime 단일사건인지 확인
   (한 fold 운이면 REJECT 근거). fold_net_min/fold_net_pos_frac 컬럼 사용.
3. **충실성 재감사**: baseline 셀이 `r2_challenger_picks_v1.csv` R1_ratio 와 bit-일치(주장)
   — 독립 재현 + leak(D-1 입력·day-D 라벨·train-only calib·embargo5) 위생 한 번 더 적대검토.
