# 연구 노트: Challenger A1 — sustainability-filter (진입-quality)

- **작성**: signal-researcher, 2026-06-01
- **상태**: 증거 생성 완료. 채택 결정 X (quant-evaluator ADOPT/SHADOW/REJECT 대기).
- **산출 파일**:
  - 스크립트: `scripts/ch_sustainability_v1.py` (NEW)
  - 비교표: `output/ch_sustainability_compare_v1.csv` (R1 baseline + A1 8조합)
  - 커버리지/메타: `output/ch_sustainability_coverage_v1.json`
  - 픽 dump(감사): `output/ch_sustainability_picks_v1.csv` (R1 vs best-downside A1 side-by-side)
  - 실행 로그: `output/ch_sustainability_run_v1.log`

---

## 가설
R1 top-3 의 진입 엣지는 real(lift 3-6x)이지만 net breakeven~음수 = pump-then-dump.
**D-1 시점 신호로 "지속 펌프 vs 펌프-후-덤프"를 가를 수 있다면**, dump-prone 픽을 강등하고
다음 R1 후보로 교체해 진입 quality(=net·하방)를 살릴 수 있다.

## 무엇을 돌렸나
- **R1 baseline 충실 재현**: `r2_challenger_compare_v1.py` 와 **동일 빌더·동일 유니버스
  (정적 top100, D-1 qv_rank)·동일 head(`walk_forward_heads`, 6-fold purged WF + embargo5,
  per-fold OOF bucket calibration)·동일 15m SL-3%/TP+5%/EOD 청산경로(net 0.15% 차감)**.
  → R1 baseline 지표가 기존 `r2_challenger_compare_v1.csv` 의 R1_ratio row 와 **정확히 일치**
  (net_mean −0.0028, %SL 0.456, deepNoSL 0.135, n=1531, 765일). = 자가 leak 검증 PASS.
- **A1 sustainability head**: day-D "dumped" 라벨 2종을 R1 head 와 **동일 D-1 피처**로 학습.
  - `lab_dump_A` = (high/open−1 ≥ +5%) AND (close/open−1 < 0)      base 0.042
  - `lab_dump_B` = (high/open−1 ≥ +5%) AND (close/open−1 < −2%)    base 0.026
  - 피처 = `_feats(panel)` (24개, 전부 `f_` prefix D-1, LEAK_COLS/next_* 제외). dump_risk
    게이지 구성요소(f_ret_7d 과열·f_log_qv board-top)와 regime decile 이 이미 포함됨.
- **A1 re-selection**: 각 날 R1 순위대로 보되 `p_dumped > cutoff` 픽을 강등→다음 R1 후보로 교체.
  cutoff = **fold<k 의 train OOS p_dumped 의 q 분위 (OOF, train-only)**. fold0 은 prior train
  없어 강등 안 함(R1 그대로). 항상 top-3 채움(=거래량 동일).
- **sweep**: dump 라벨 2 × cutoff 분위 q∈{0.6,0.7,0.8,0.9} = **8 조합** (deflate 기록용).

## leak·시간정합성 방어 (§1 체크리스트)
- **시점 분리**: 입력 피처 ≤ D-1 (`build_market_features` shift(1)); dump 라벨은 day-D
  open/high/close(미래) → 학습 입력에 안 섞임. 확인: `lab_dump_A/B` ∉ PRECURSOR_FEATURES,
  PRECURSOR_FEATURES 전부 `f_` prefix.
- **train-only fit**: head(XGB)·OOF bucket calibration·cutoff 분위 모두 train fold 안에서만
  산출, test fold 는 적용만. embargo=5.
- **유니버스 시간정합**: top100 = D-1 qv_rank (`f_universe_qv=qv.shift(1)`).
- **비용**: 왕복 0.15% 차감 net (모든 지표).
- **"너무 좋으면 leak 의심"**: A1 net 은 여전히 **음수** (과신 신호 없음). R1 baseline 이
  기존 baseline 과 byte-일치 → A1 파이프가 leak 했다면 R1 재현이 어긋났을 것.
- 자동주문 X / 공유 라이브 파일(recommend.py/model_registry.py/daily_*.sh) 미편집.

## 시도 조합 수 (selection deflate 용)
- A1 sweep 8 조합 (dump 라벨 2 × cutoff 분위 4). best 는 **하방-우선**(%SL↓→deepNoSL↓→net↑)으로
  자동선정 = `dump_B q=0.6`. cutoff 는 OOF 지만 q grid 자체가 search → evaluator deflate 필요.

## 1차 결과 (OOS 765일, top-3, net 0.15% 차감, 15m 경로)

| policy | sub率 | %SL | deepNoSL | **net_mean** | hit | prec@3(p20) | P(min≤−5%) | cum | top10qv | medQVrank |
|---|---|---|---|---|---|---|---|---|---|---|
| **R1 baseline** | 0.00 | 0.456 | 0.135 | **−0.00284** | 0.39 | 0.0366 | 0.298 | −0.864 | 0.302 | 28 |
| A1 dump_A q0.6 | 0.42 | 0.396 | 0.089 | −0.00277 | 0.41 | 0.0183 | 0.218 | −0.858 | 0.292 | 45 |
| A1 dump_A q0.8 | 0.27 | 0.417 | 0.107 | −0.00241 | 0.40 | 0.0229 | 0.248 | −0.830 | 0.287 | 36 |
| **A1 dump_B q0.6** ★best-down | 0.64 | **0.326** | **0.060** | **−0.00155** | 0.43 | 0.0118 | **0.152** | −0.735 | 0.294 | 51 |
| A1 dump_B q0.7 ★best-net | 0.62 | 0.331 | 0.065 | **−0.00153** | 0.43 | 0.0144 | 0.159 | −0.733 | 0.294 | 50 |
| A1 dump_B q0.9 | 0.33 | 0.385 | 0.091 | −0.00268 | 0.40 | 0.0281 | 0.215 | −0.851 | 0.301 | 36 |

**Δ(best-downside A1=dump_B q0.6 − R1):** Δnet_mean **+0.0013** · Δ%SL **−0.130** · ΔdeepNoSL
**−0.074** · ΔP(min≤−5%) **−0.146** · Δprec@3 **−0.025** · Δ픽수 **+0** · Δcoverage **+0.000**.

### net 양수/개선되나?
- **개선되나 양수 아님.** 모든 A1 조합 net_mean 음수 유지 (best −0.00155 vs R1 −0.00284).
  하방(% SL·deepNoSL·P(min≤−5%))은 **크게·일관되게 개선**되고 net 도 +0.0013 개선되지만
  breakeven 못 넘김. 사용자 선호(하방최소화 우선)에는 부합, "돈 버는 시스템"엔 미달.

### degeneracy 판정 (덜 거래 vs 진짜 픽 개선)
- **"덜 거래" degeneracy 아님**: n_picks 1531·coverage 1.00 **모든 정책 동일** (A1 은 항상 top-3
  채움). 거래량 줄여 net 올린 게 아니다.
- **"안 움직이는 대형주" R2 degeneracy 아님 (반대 방향)**: A1 의 frac_top10_qv 0.302→0.294
  (감소), median_qv_rank 28→51 (**더 작은 코인** 쪽으로 이동). R2 가 대형주로 쏠려 하방을
  낮춘 것과 정반대 — A1 은 대형주 회피가 아니라 *덜 과열된* 후보로 교체.
- **단, "꼬리 동시 절단" 한계 (핵심)**: A1 픽은 하방만 줄인 게 아니라 **상방도 같이 줄였다**.
  - 픽 평균 up_high_ret 0.0471→0.0307, 펌프5%율 0.281→0.163, prec@3(pump20) 0.037→0.012.
  - 픽 평균 down_low_ret −0.0396→−0.0282.
  - = sustainability head 가 "지속 펌프"를 골라낸 게 아니라 **변동성 자체가 낮은 후보**로
    교체. 수익꼬리·손실꼬리가 같은 진입에 묶여있다는 기존 교훈(exit-규율 REJECT 사유)이
    **진입 단계에서도 재확인**. 하방 개선의 대가로 레이더 본질(펌프 포착)을 깎음.
- **하방 개선 ∝ 교체율**: dump_B(강한 라벨)·낮은 q(공격적 cutoff)일수록 sub率↑·하방↓.
  순수 효과(=같은 펌프력 유지하며 덤프만 회피)가 아니라 교체로 펌프력을 내준 trade-off.

## evaluator 가 볼 것 (3줄)
1. **net 음수 + trade-off**: net +0.0013 개선·하방 큰 폭 개선이지만 **여전히 net 음수**이고
   prec@3 0.037→0.012·펌프5%율 0.281→0.163 으로 상방을 같이 깎았다. "하방 개선이
   상방 희생의 부산물인가, 진짜 dump 회피인가"를 bootstrap CI(Δnet_mean·Δ%SL) + 같은-펌프력
   매칭으로 가를 것. (deepNoSL=SL 끈 본질 하방이 0.135→0.060 인 게 가장 신뢰도 높은 신호.)
2. **selection deflate**: 8조합 sweep + 하방-우선 best 자동선정. cutoff 는 OOF 지만 q grid·
   라벨 선택은 search → DSR/Holm 류로 deflate. best-down(q0.6) vs best-net(q0.7) 거의 동일 →
   q 에 robust 한 편이나 라벨(A vs B) 차이는 큼(B 가 일관 우세) — 라벨 정의가 lever.
3. **leak 재감사 포인트**: dump 라벨이 day-D close 를 쓰므로 (a) `_feats` 에 close 계열 same-day
   누출 없는지, (b) per-fold cutoff 가 fold<k train OOS 만 쓰는지(fold0 강등안함 처리) 재확인.
   R1 baseline byte-일치가 1차 방어선이나 dump head 경로는 독립 검증 권장. n 충분(38,682 OOS,
   765일) → small-n 함정은 없음.

## ops 핸드오프 (diff 제안만 — 라이브 파일 미편집)
A1 이 ADOPT/SHADOW 되면 `signals/recommend.py` 의 R1 정렬 뒤 re-selection 레이어로 배선:
- `score_candidates` 에 `ranking="A1"` 분기 추가 (R1 정렬 후 dump head p_dumped 로 강등→교체).
- dump head 빌더는 `walk_forward_heads` + day-D dump 라벨(`ch_sustainability_v1.add_dump_labels`)
  재사용. cutoff 는 artifact 고정값(라이브 재계산 금지) — 기존 dump_risk 게이지 자리에 흡수 가능.
- **사용자 컨펌 필요**: dump 라벨 정의(X/Y) + re-selection 이 라벨/architecture 변경에 해당.
- 단 현재 증거상 net 음수 → SHADOW(record-only) 가 적정선. forward 30거래일로 하방 개선 재현 확인 권장.
