# 연구 노트: Binance USDT D-1 lead-lag → Upbit KRW pump (day D) 증분 lift

작성: signal-researcher · 2026-06-11 · 상태: **실행 완료 · 판정 = 피처 후보 (evaluator 감사 대기)**
실행: 사용자 메인 루프. `build_upbit_panel` 의 `d["market"]=market` 1줄 누락 fix 후 풀런 성공
(`load_candles` 가 market 컬럼 미반환 → 추가). 로그: `output/binance_leadlag_run.log`.

---

## 0. 한 줄 결론

**Binance D-1 거래대금 surge(>1.5×)를 베이스라인 룰(roc_7d_rank>0.85)에 AND 로 얹으면,
pump20 적중 lift 가 2.66→4.34 (증분 1.63x) 로 5/5 fold 전부에서 오르고, net(TP5/SL3)도
+0.78%→+1.24%(보수 0.5% 비용에서도 +0.43%→+0.89%)로 개선된다.** lift 만이 아니라 net 까지
같은 방향이라 hit≠Sharpe 함정을 일단 통과. 단 **(a) Binance DB 2026-05-03까지(최근 39일 미검증),
(b) 윈도는 bull 포함인데 현 라이브는 bear_quiet, (c) per-fold net 일관성 미출력, (d) 룰 4개
hand-pick** 한계로 **자체 채택 안 함**. 판정: **피처 후보 (pump_hunter rule v2 후보) →
quant-evaluator 감사 대기.**

---

## 1. 가설

> **Binance USDT 에서 D-1 에 거래대금이 튀거나(surge) 위로 움직인 코인은, 같은 자산의 Upbit KRW
> 일봉이 day D 에 펌프(high_D/open_D ≥ 20%)할 확률이 Upbit-only 베이스라인보다 높다.**
> Binance 단독 성능이 아니라 **베이스라인 대비 증분(lift 더해지는지)** 이 질문.

세부 4룰(전부 ≤ D-1, hand-pick): bn_up(b_ret_1d>0) · **bn_volsurge(b_vol_surge>1.5)** ·
catchup(bn−upbit ret 차>0) · bn_only(surge>2.0 & ret>0.03, 베이스라인 없이 참고용).

---

## 2. 경계 정렬 — 실데이터 검증 PASS

`output/binance_leadlag_v1_boundary.csv`:
- upbit time-of-day = **09:00:00**, binance = **00:00:00** (예상대로).
- BTC 일간 log-return corr: **lag0 = 0.9617**, lag+1 = −0.0299, lag−1 = −0.0740 (overlap 1095일).
- **ALIGNMENT = PASS** — lag0 가 압도적 최대 → date-part join 정상. lag±1 이 0 근처라
  "하루 밀린 join" 사고 가능성 사실상 배제.

→ Binance 날짜 D-1 봉 ≡ Upbit 날짜 D-1 봉(동일 실제 UTC 일), D-1 봉은 D-1 24:00 UTC =
D 09:00 KST 완전 마감 = Upbit D open 직전 → **strictly ≤ D-1, leak 없음** (실데이터 확인).

---

## 3. 매핑 커버리지

`--inspect` + `output/binance_leadlag_v1_mapping.csv`:
- KRW↔USDT 매핑 가능: **251/252 (99.6%)** (스테이블/wrapped 만 제외).
- Binance DB 실제 보유: **184/252 (73.0%)** — 나머지 ~27% 는 Upbit 상장이지만 Binance USDT
  미상장/미수집(소형·국내전용 코인). 이들은 Binance 피처 NaN → `bn_avail` 게이트로 자연 제외.
- 행 단위 Binance join 커버리지: fold별 **0.70~0.78** (lift.csv `bn_coverage_val`). 즉 베이스라인
  fire 행의 ~73% 만 Binance 신호를 가짐 → 증분 룰은 그 부분집합에서만 작동(recall 자연 감소).

---

## 4. 무엇을 돌렸나

- 라벨: 기존 `pump20 = high_D/open_D−1 ≥ 0.20` 그대로(변경 없음, 컨펌사항 미터치).
- 베이스라인: `roc_7d_rank>0.85`(signals/pump_detector_v1.py 와 동일 랭크·임계).
- 유니버스: feature_date(D-1) 거래대금 **top-120**. 시간정합 — 랭크 D-1 기준.
- 피처(≤D-1): b_ret_1d, b_ret_7d, b_vol_surge(D-1 거래대금/20d평균), bn_minus_upbit_ret_1d.
- 검증: `signals.validate.PurgedWalkForward` (n_folds=5, **embargo=10d**, holdout 180d).
  각 fold val=**OOF** 에서만 룰 비교(룰은 고정식이라 fit 없음, 평가만 OOF 한정).
- net: day D open 진입, TP+5%/SL−3%. 일봉 한계상 intraday 순서 불명 → eod/tpsl 병기,
  비용 0.15%+0.50% 2-tier 차감.

---

## 5. leak·시간정합 방어 (실행 후 재확인)

- [x] 피처 행 = feature_date(D-1)까지; 진입/라벨 = `t+1`(=day D) market-내부 self-join.
- [x] same-day leak 방지: 피처 함수가 만든 feature-row 라벨은 `cur` 에 미포함 — 라벨은 day D 만.
- [x] LEAK_COLS/next_*: 미생성. 유니버스 D-1 랭크. survivorship: 과거 행 포함(누락일만 drop).
- [x] Binance 경계: §2 실데이터 PASS(lag0 0.96). 거래비용 차감 net 으로만 판정.

---

## 6. 1차 결과 (실측)

### 6.1 증분 lift (`output/binance_leadlag_v1_lift.csv`, base rate pump20 ≈ 1.2~2.1%/fold)

| rule | mean_lift | min_lift | folds_fired | 증분 vs baseline | 비고 |
|------|-----------|----------|-------------|------------------|------|
| baseline_roc7 | 2.661 | 2.140 | 5/5 | — (기준) | recall 0.32~0.54 |
| **base_AND_bn_volsurge** | **4.338** | **3.031** | **5/5** | **1.63x** | recall 0.11~0.26 |
| base_AND_bn_up | 3.510 | 2.282 | 5/5 | 1.32x | recall 0.13~0.30 |
| base_AND_catchup | 2.461 | 1.996 | 5/5 | 0.93x → **기각** | net 도 baseline 이하 |
| bn_only_surge+mom | 4.922 | 3.402 | 5/5 | 1.85x | **recall 0.09~0.20 (낮음), net 미시뮬** |

**핵심: base_AND_bn_volsurge 는 5 fold 모두에서 baseline 보다 lift 높음**
(per-fold lift: 5.02/5.43/3.60/3.03/4.60 vs baseline 3.49/2.59/2.14/2.42/2.66 →
per-fold 증분 1.44/2.10/1.68/1.25/1.73x, 전부 >1.2). fold 일관성 확보.

### 6.2 net (`output/binance_leadlag_v1_net.csv`, TP5/SL3, tpsl 모델)

| rule | cost | net_tpsl_mean | winrate | n |
|------|------|---------------|---------|---|
| baseline_roc7 | 0.15% | +0.776% | 51.3% | 14580 |
| baseline_roc7 | 0.50% | +0.426% | 48.6% | 14580 |
| **base_AND_bn_volsurge** | **0.15%** | **+1.239%** | **56.5%** | 4128 |
| **base_AND_bn_volsurge** | **0.50%** | **+0.889%** | **54.6%** | 4128 |
| base_AND_bn_up | 0.15% | +0.895% | 52.6% | 6332 |
| base_AND_catchup | 0.50% | +0.395% | 48.1% | 5367 (baseline 이하) |

→ bn_volsurge 는 **두 비용 tier 모두에서 baseline net 을 능가**(+0.46%p @0.15, +0.46%p @0.50),
winrate 도 51→56%. lift 와 net 이 같은 방향 = hit≠Sharpe 함정 일단 통과. n=4128 로 표본도 충분.
(eod 종가청산 모델은 모든 룰이 net 음수 — TP5/SL3 규율이 net 을 양으로 만드는 구조. 이건
exit lab 의 "lever=exit 규율" 진단과 일관.)

---

## 7. 한계 (주의해서 다룰 점 — 채택 막는 4가지)

1. **(a) 데이터 stale**: Binance DB = 2026-05-03 까지. **최근 39일 전혀 미검증.** 위 결과는
   2023~2026-05-03 윈도. collector_binance_d1 갱신(`--all --days 60`) 후 재실행이 정공법.
   갱신 시 최근 fold 가 추가되면 증분이 유지되는지 재확인 필요.
2. **(b) regime 미스매치**: 이 윈도는 **bull 구간 포함**. 현 라이브는 **bear_quiet**. 이 프로젝트는
   **백테스트→라이브 비전이 사례 2회**(bear_quiet 연구 가설 반증, coldstart REJECT) 전적이 있음.
   bull 에서 Binance surge→Upbit pump 가 강했어도 bear_quiet 에서 같으리란 보장 없음.
   **regime-split(bear_quiet 서브샘플) lift/net 을 evaluator 가 별도 확인해야** 함 — 미출력.
3. **(c) per-fold net 일관성 미출력**: net 은 pooled OOF 1개 숫자만. fold별 net 분산/최악 fold 는
   안 봄. lift 는 5/5 일관이지만 **net 의 fold 일관성은 모름** — 한 fold 가 net 을 끌어올렸을
   가능성 배제 못 함. 스크립트 확장(fold별 net_sim) 필요.
4. **(d) hand-pick 4룰**: 임계(surge>1.5, ret>0 등)는 sweep 없이 직관 1세트. 4룰 중 best 를
   사후 고른 셈 → **selection bias**. evaluator 가 4룰 × (사실상 임계 자유도)로 deflate 해야.
   채택 진행 시 임계를 OOF-sweep 으로 재선택 권고(현재는 deflate 부담 작은 "직관 1샷"이지만 best
   고른 행위 자체는 기록).

기타: bn_only 가 lift 최고(4.92)지만 recall 0.1~0.2 로 거의 안 잡고 net 미시뮬 — 헤드라인 아님,
별도 angle(베이스라인 무관 Binance 독립 신호)로만 기록.

---

## 8. 시도 조합 수 (selection deflate 용)

- 증분 룰 후보 **4개** + bn_only 참고 1 = best 1개(bn_volsurge) 사후 선택.
- 임계 **hand-pick 1세트**(sweep 안 함). target 1(pump20) × universe 1(120) × embargo 1(10).
- → evaluator: "4룰 중 best + 임계 자유도" 기준 deflate. PSR/DSR 시 trials 최소 4(룰) 반영,
  임계 튜닝까지 보수적으로 보면 더 큼.

---

## 9. quant-evaluator 감사 요청 (특히 의심 지점)

1. **regime-split**: bear_quiet 서브샘플에서 bn_volsurge 의 lift·net 이 유지되는가? (한계 b —
   가장 중요. bull artifact 면 라이브 무용.) 필요 시 panel 에 btc_regime join 해 재집계.
2. **per-fold net**: net 의 fold별 분산/최악 fold. 한 fold pump 폭발이 pooled +1.24% 를 만든 건
   아닌지(한계 c). `oof_picks.csv` 에 fold·pump_max_return 있으니 재현 가능.
3. **selection deflate**: §8 의 4룰+임계 자유도 반영(한계 d).
4. **stale**: 최근 39일 미검증(한계 a) — 갱신 후 재현 전엔 forward 신뢰도 보류.
5. **bn_volsurge vs baseline 교집합 성격**: surge 가 거는 4128행이 baseline 14580 의 단순 고-lift
   부분집합인지(=Binance 가 정보 추가 없이 그냥 더 센 momentum 만 고른 건지) 점검. roc_7d_rank
   분포가 bn_volsurge 행 vs baseline 행에서 유의하게 다른지 `oof_picks.csv` 로 확인 요청.

---

## 10. 판정

**피처 후보 (pump_hunter rule v2 후보) — quant-evaluator ADOPT/SHADOW/REJECT 감사 대기.**

근거: 증분 lift 1.63x(5/5 fold) + net 두 비용 tier 모두 baseline 능가 → 기준(증분 ≥1.3x +
fold 일관 + net 악화 없음) 충족. **단 자체 채택 안 함** — 한계 (a)stale (b)bull-only
(c)net fold 미검증 (d)hand-pick 때문에 evaluator 의 regime-split·per-fold net·selection deflate
감사를 통과해야 라이브 배선(ops-steward) 자격. SHADOW 로 먼저 기록 권장.

재현:
```bash
cd /home/soccz/22tb/prelude
venv/bin/python -m data.collector_binance_d1 --all --days 60        # stale 갱신
venv/bin/python scripts/binance_leadlag_v1.py --inspect             # 정렬 PASS 재확인
venv/bin/python scripts/binance_leadlag_v1.py --target pump20 --top-universe 120 --embargo 10
```
산출: `output/binance_leadlag_v1_{boundary,mapping,oof_picks,lift,net}.csv`,
로그 `output/binance_leadlag_run.log`.
