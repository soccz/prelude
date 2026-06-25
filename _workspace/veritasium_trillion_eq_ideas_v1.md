# prelude × Veritasium "The Trillion Dollar Equation" — 신규 아이디어 종합 (v1)

> 출처 영상: Veritasium "The Trillion Dollar Equation" 한국어판 (youtu.be/99BHnu64pu8, 2026-06-22).
> 방법: 멀티에이전트 워크플로우 40 에이전트 (6 정찰 + 8시드×3 아이디어 = 24 생성 + 24 적대평가(quant-evaluator) + 교차비평 + 종합).
> 종합 원칙: **radar not strategy. lever는 진입신호 추가가 아니라 exit·하방 규율.**
> 랭킹 = 복합점수(novelty+feasibility+downside_fit+edge_potential), 동점은 exit/하방 레버 우선.
> 영상 6개 개념(HMM 숨은상태 / Newton 과확장함정 / Bachelier 확산·공정가 / 옵션 볼록payoff / Thorp 가변사이징·동적헤지 / alpha decay)을 prelude 검증자산 위에 얹되, REJECT/SHADOW 트랙과 차별성을 적대 감사.

---

## TL;DR (복합점수 순, 상위 7)

1. **합성 델타-사다리 청산 (16)** — `walk_path`를 단일청산→부분청산 누적으로 일반화. 상방에서만 1/3씩 익절·SL은 절대 안 넓힘. exit_lab의 진짜 한계('1트레이드=1청산점')를 깨는 새 lever. **PROTOTYPE.** 핵심 위험: 멀티청산 비용(0.45~0.6pp drag).
2. **발견 후 군중쏠림 정찰 Crowding Decay (14)** — 코인×쏠림(연속랭크≥3일+roc_rank 상위) 버킷으로 '이미 늦은 펌프'를 ACTIVE→WATCH 강등. day-quality(날짜 침묵, REJECT)와 축이 다른 미시 필터. **PROTOTYPE.**
3. **코인별 확산경로 MC 브래킷 (13)** — D-1 변동성으로 코인별 first-passage TP/SL 차등. exit_lab REJECT 근본원인('고정 브래킷 무차별→14x deep-loss')을 정면으로 친다. **PROTOTYPE.**
4. **CSS 볼록성 점수 (13)** — recommend.py de-corr head 확률을 분포 우편향으로 재해석한 새 정렬키. 새 학습 0, feasibility 최상. **PROTOTYPE.** 단 2차차분 항 70% 클립 → 실질은 비율 reweight.
5. **분포 공정가 게이트 (12)** — head별 EV=q·T−(1−q)·fail_loss−cost>0일 때만 발사. h6 일반화. **PROTOTYPE.** 누수 2개(calibration in-sample·binary-fill 환상) 닫아야.
6. **모델분포 vs 확산분포 괴리 게이지 (12)** — 모델 head 확률 − 순수 확산 베이스라인 = '변동성 재포장 vs 진짜 정보' 진단. **PROTOTYPE.**
7. **calibration drift 센티넬 (12)** — head별 공정가 gap의 시간적 decay 감시 → down-weight. **PROTOTYPE.** forward 표본(152행/25일) 부족이 발목.

> 과확장 침묵 게이트(13)는 점수 높지만 **REDUNDANT** — A1/downside_aware cut=0.667 row가 이미 그 실험. 점수만으로 추천하지 않음.

---

## 영상개념 → prelude 아이디어 매핑

| 영상 개념 | 아이디어 | 핵심 연결 파일 | 판정 |
|---|---|---|---|
| HMM 숨은상태 (Simons/Baum) | HMM posterior 게이트 | `features.py:compute_btc_features`, `detector.py` regime gate | REDUNDANT |
| HMM transition | Regime-전이 dip-buy 경보 | `bear_quiet_path_verify_v1.py`, `downside_head_riskreward_v1.py` | PROTOTYPE |
| HMM 상태혼합 + 옵션 | HMM 분포가격 손실하한 사이징 | `regime_hmm.py`(신규), `sizing.py:kelly_quarter` | PROTOTYPE |
| Newton 과확장 함정 | 과확장 침묵 게이트 | `cc_dipbuy_v1.py:oversold_mask`, `recommend.py:_add_dump_risk` | REDUNDANT |
| Newton + 옵션 | 합성 옵션 페이오프 청산 | `exit_lab.py:walk_path`, `recommender_downside_exit_v1.py` | REDUNDANT |
| Newton + Bachelier 확산 | 확산-폭주 분리 분포 가드 | `downside_head_riskreward_v1.py`, `univariate_precursor_lift_v1.py` | REDUNDANT |
| Bachelier 공정가=calibration | 분포 공정가 EV 게이트 | `distribution_engine.py:assemble_alerts`, `decision_policy.py:_edge_proxy_pct` | PROTOTYPE |
| Bachelier 확산폭(√t) | 확산폭 vs 합성 브래킷 | `labels.py:approx_ci_from_bins`, `exit_lab.py:walk_path` | PROTOTYPE |
| Bachelier 확산 first-passage | 코인별 MC 브래킷 | `exit_lab.py:walk_path`, `features.py`(atr/vol) | **PROTOTYPE** |
| Bachelier first-passage 게이트 | 확산 하방게이지 진입 재정렬 | `recommend.py:score_candidates` | REDUNDANT |
| Bachelier 모델분포 vs 랜덤워크 | 괴리 게이지(calibration arbitrage) | `distribution_engine.py:score_panel`, `bucket_calibration.py` | PROTOTYPE |
| 옵션 볼록payoff + Thorp | 합성 델타-사다리 청산 | `exit_lab.py:walk_path/EXIT_VARIANTS` | **PROTOTYPE (top)** |
| 옵션 볼록 + 분포 | CSS 볼록성 점수 | `recommend.py:_fit_rr_head`, `downside_head_riskreward_v1.py` | **PROTOTYPE** |
| 옵션 볼록 + 분포 7-head | DistConvexity | `distribution_engine.py`, `labels_distribution.py:HEADS` | PROTOTYPE |
| 옵션 볼록 실현 | 실현-볼록성 게이트 RCG | `exit_lab.py:evaluate_exit_variants` | PROTOTYPE |
| Thorp 조건부 SL깊이 | 분포-조건부 동적 손절 | `distribution_engine.py`, `exit_lab.py:walk_path` | PROTOTYPE |
| Thorp 가변 Kelly | 분포기반 분수켈리 사이징 | `sizing.py:kelly_quarter` | REJECT |
| Thorp + 옵션 합성 | 합성 옵션 payoff 사이징 | `sizing.py`, `exit_lab.py:EXIT_VARIANTS` | REJECT |
| Thorp 자본곡선 보호 | 켈리 자본곡선 가드 | `risk.py:evaluate_risk`, `metrics.py:compute_mdd` | REJECT |
| Bachelier 확산-콘 | 확산-콘 추적바닥 | `exit_rules.py:simulate_exit` | REDUNDANT |
| alpha decay (효율시장) | 엣지 반감기 추적기 | `drift_detector.py`, `paper_ledger_backfill.csv` | PROTOTYPE |
| alpha decay (군중쏠림) | 발견 후 군중쏠림 정찰 | `recommendation_quality.py:_evidence_maps` | **PROTOTYPE** |
| alpha decay (정책부패) | 정책 변형 부패 감시 | `policy_competition.py:run` | PROTOTYPE |

---

## 우선 채택 추천 (상위 PROTOTYPE)

### 1. 합성 델타-사다리 청산 (복합 16, downside_fit 5)
- **무엇:** 진입 100% 보유에서 running_high가 +2/+4/+6% arm을 넘을 때마다 보유분 1/3씩 부분청산(이익 잠금), hard floor −5%는 항상 잔여 전량에 적용. exit_lab 7번째 record-only 변형으로 추가.
- **왜:** exit_rules 6변형 REJECT의 핵심은 '1트레이드=1청산점' + vol룰이 SL을 넓혀 14x deep-loss. 델타-사다리는 SL을 **절대 안 넓히고** 상방에서만 연속 축소 → 그 실패모드를 정면 회피. 부분청산은 단일 SL보다 하방분산을 구조적으로 줄여 net이 음수여도 deep-loss 빈도↓.
- **재사용:** `ledger/exit_lab.py:walk_path/EXIT_VARIANTS`, `recommender_downside_exit_v1.py:simulate_path/load_paths`, `tests/test_exit_lab.py`(동등성 게이트).
- **하방적합:** −5% floor가 worst_trades(−30~−47% noSL 참사)를 −5.15%로 전부 캡. 사용자 ~−5% 수용 anchor와 정확히 정합.
- **leak 방어:** arm 판정은 running_high로만, 같은 봉 SL-first, 시간 오름차순 walk. **체결가는 봉 high가 아니라 arm 터치가(arm_px)로 고정 필수**(아니면 룩어헤드). test_exit_lab byte-동등 회귀로 prefix-only 고정.
- **첫 실험:** recommender_downside_exit_v1 OOF pick set(K=3, lab_pump20, purged WF 5-fold+embargo 5d)에 delta_ladder 추가. fold별 net·deep_loss·CVaR95 bootstrap CI95. **게이트: deep_loss freq ≤ 챔피언 AND net Δ CI95 하한 ≥ 0 인 fold ≥ 4/5.**
- **핵심 위험:** 멀티청산 비용(3-tranche 0.45pp/full+floor 0.6pp)이 챔피언 0.15pp 대비 +0.3~0.45pp drag → 이미 음수인 net을 더 민다. 볼록꼬리(늦은 펌프 잔여 라이드)가 이 비용을 넘는지가 유일한 미검증 자유도.

### 2. 발견 후 군중쏠림 정찰 Crowding Decay (복합 14)
- **무엇:** D-1 쏠림 프록시(roc_7d_rank·volume_spike·연속랭크≥3일)로 '과열-소진' 버킷을 evidence stratum 추가, 그 버킷 과거 CLOSED net<0·deep_loss↑면 ACTIVE→WATCH DOWNRANK.
- **왜:** Newton 함정(군중 몰린 늦은 펌프=칼받기)의 코드화. day-quality(REJECT)는 '날 전체'를 죽였고 permutation noise였지만, 이건 '코인×쏠림' 미시 필터 + 침묵이 아니라 강등이라 best day를 안 끔.
- **재사용:** `recommendation_quality.py:apply_recommendation_quality/_evidence_maps`, `features.py:compute_alt_features`, backfill 5080행(754일).
- **leak 방어:** **honest net 필수 — next_close 또는 exit_lab 15m sim. max-bracket(high≥+5%→+5% 확정) 절대 금지**(binance lead-lag를 죽인 lognormal-truncation +2.86pp 환상 재현 위험). 버킷 n<8이면 판정 보류.
- **첫 실험:** backfill 과열/비과열 2버킷 라벨링 → honest net·deep_loss를 purged WF + bootstrap CI95. **게이트: 동일 setup_quality·regime 통제하 비과열 버킷이 net·deep_loss 둘 다 유의 개선이면 SHADOW, 양쪽 동등 음수면 REJECT.**
- **핵심 위험:** 거의 모든 버킷 net이 음수라 '과열 버킷 net<0'은 trivially 참 → 진짜 게이트는 비과열 differential 실재 여부.

### 3. 코인별 확산경로 MC 브래킷 (복합 13)
- **무엇:** 코인 D-1 변동성으로 캘리브레이션한 점프-확산 SDE로 96-step 경로 N=5000 합성 → 브래킷별 P(SL-first)·E[net]·P(loss<−5%) 추정 → 코인×regime별 '하방하한 제약 하 net 최대' 브래킷 추천(record-only 병기).
- **왜:** exit_lab REJECT 근본원인('고정 브래킷 무차별')을 정면으로 침. MC P(SL-first) vs 실현 0.52~0.54 reliability라는 자체 반증 장치 내장.
- **재사용:** `exit_lab.py:walk_path`, `recommender_downside_exit_v1.py:load_paths/simulate_path`, `bear_quiet_path_what_hit_first_v1.csv`.
- **leak 방어:** 합성경로는 미래 봉 0개 봄 → 구조적 look-ahead 불가. **단 '순환 검증' 주의 — 캘리브레이션 구간≠검증 구간 fold 분리 필수. selection-bias deflate(코인×regime×브래킷 trials) 의무.**
- **첫 실험:** 1 fold bear_quiet 고변동 코인 5~10개, SDE 적합→합성→EXIT_VARIANTS별 P(SL-first) 예측 vs 실제 15m 경로 실현치 1장 표 비교. **MC가 실현 SL-first를 ±0.05 내로 맞추는지부터 — 못 맞추면 generator가 틀린 것이니 즉시 중단.**
- **핵심 위험:** net lever=entry quality로 이미 확정. 합성 generator는 실현 분포에 없는 net을 못 만듦. 현실적 천장은 net flip이 아니라 코인별 deep-loss 꼬리 추정 도구. features.py에 skew/kurt/jump 피처 부재 → 신규 작성 필요.

### 4. CSS 볼록성 점수 (복합 13, feasibility 5)
- **무엇:** de-corr head P(up5/10/20)·P(dn5/10)·exp_downside를 이산 CDF로 보고 CSS = up_convexity/(downside_cost+eps). score_candidates에 ranking='CSS' 모드 추가, 정렬키만 교체(새 학습 0).
- **왜:** R1 ratio(p_up10/p_dn5)는 *한 점*의 risk-reward, CSS는 분포 *전체 형태*(꼬리두께 vs 하한). 비율형이라 상방0→분자0으로 저변동 대형주 자동 강등(R2 degeneracy 회피).
- **재사용:** `recommend.py:score_candidates/_fit_rr_head`, `downside_head_riskreward_v1.py:_oof_bucket_calib`.
- **첫 실험:** 동일 6-fold OOF에서 CSS top-3 vs R1 top-3를 simulate_path(TP5/SL3/EOD, 0.15% 차감). Δdeep_loss·Δnet CI95 + CSS↔rr_ratio rank-corr + 2차차분 항 clip률 보고.
- **핵심 위험:** 2차차분 convexity 항 70.7% 클립 → 실질은 (p_up20/p_up5)/(p_dn5+2p_dn10) 비율 재배열. deep-loss는 TP5/SL3에서 이미 ~0이라 ≥3pp↓ 합격선 거의 닿을 수 없음.

### 5. 분포 공정가 EV 게이트 (복합 12)
- **무엇:** 7-head를 calibrated q로 환산, EV_head = q·T − (1−q)·fail_loss − cost. coin 발사 = max_head EV>0, EV≤0이면 SILENCE.
- **왜:** Bachelier 공정가=calibration 정신. 일부(decision_policy._edge_proxy_pct h6)가 이미 배선 → 신규성은 7-head 일반화+per-coin argmax.
- **leak 방어 (치명 2개):** (1) 현 calibration_h*.csv는 5080행 in-sample qcut → **purged-WF train-fold-only 적합·test freeze artifact 먼저.** (2) **EV 상방항은 q·T가 아니라 exit_lab 실현 payoff 분포**(binary-fill 환상 회피).
- **핵심 위험:** 게이트는 발사만 거를 뿐 dump-after-pump lever를 안 건드림 → 기껏 거래수↓+deep_loss↓. honest 숫자로 h6만 +0.66%p 살고 h5는 −2.25%p 음수.

---

## 강력한 조합 (cross-critic)

1. **코인별 변동성 사다리 청산** = MC 브래킷 × 확산-폭주 분리 × CSS. 같은 substrate(D-1 변동성 atr_pct_14/vol_7d + de-corr head) 공유. 분포가드가 정상확산/폭주 분리, MC가 코인별 first-passage TP/SL, CSS가 상방볼록 코인만 와이드 TP. exit·하방·코인차등을 한 번에. **→ 가장 강한 통합 lever 후보.**
2. **분포로 트리거되는 부분청산 사다리** = 델타-사다리 × sustain_ratio(p_h4/p_h6). sustain 낮은 코인 +3% 절반 익절, 높은 코인 보유 연장.
3. **3층 alpha-decay 운영 가드** = 엣지 반감기 × 정책부패 × calibration drift. '여러 층 동시에 식는가'로 판정 → 오탐 감소. policy_competition.db + calibration_h*.csv 이미 적재. 진입신호가 아니라 '언제 발사를 보수화/중단할지'(하방의 시간축).
4. **HMM posterior 연속 노출 스케일러** = HMM 게이트 × 전이경보 × 켈리 자본곡선 가드. Newton+Thorp+Simons를 한 사슬로. 가장 영상-충실하나 HMM 적합비용 높아 단독은 redundant, 결합으로만 정당화.
5. **과열/쏠림 진입 디스카운트** = 과확장 게이트 × 군중쏠림. **단 사이징 캡(노출 축소)으로 연결돼야 의미.**

---

## ★ Critic이 지목한 빈 교차점 (아무도 안 다룬, 가치 높음 — v2 후보)

1. **갈튼보드 × 포트폴리오 PnL 분포** — 아무 아이디어도 K=3 동시보유 바스켓의 결합 손익분포(코인간 상관)를 MC로 안 그림. exit_lab/metrics는 per-trade 단위. 영상 핵심(개별은 랜덤이어도 집합은 분포). K=3 결합 손익분포로 **바스켓 VaR/하방**을 사이징에 환류 → downside-first 사용자에게 가장 직접적인 하방 lever인데 비어있음.
2. **효율시장 × 자기발사 임팩트(self-impact)** — 모든 decay 아이디어는 시장 decay만 봄. 텔레그램 발사 코인이 사용자(+군중) 매수로 진입가 밀리는 alpha 자가소멸 미측정. **shadow_ledger ACTIVE(발사) vs WATCH_ONLY(미발사) 동일조건 forward realized 차이**로 추정 가능.
3. **옵션 theta × 보유시간 비용** — 합성옵션 아이디어는 payoff 공간만 보고 시간축(보유할수록 녹는 theta) 무시. 분포head 시간프로파일(h2 4h vs h6 24h)로 **코인별 최적보유시간**을 푸는 theta-청산 미제안.
4. **Black-Scholes 델타중립 × 바스켓 베타 헤지** — 델타를 단일코인 청산에만 쓰고, BTC 동반하락 방어(상승장 진입 후 BTC 급락)는 비어있음. spot-only라 숏 불가하지만 '노출 축소=합성 부분헤지'로 BTC regime posterior에 델타 연동.
5. **Medallion 약신호 앙상블** — REJECT/SHADOW 신호(binance hit 8.1%, day-quality clustering 0.52, A1 dump head)를 단독 net으로만 판정해 버림. 직교 약신호 결합 alpha 미측정. Simons 철학은 '단독 약신호 다수 결합'.

---

## 보류/중복/기각 (시간 낭비 방지)

**REJECT (Thorp 사이징 3종 — 음수 엔진에 사이징은 엣지를 못 만듦):**
- **분수켈리 사이징**: 654/765일(93.7%) 음수켈리→0클립, 자본 0.1%만 투입. MDD 개선은 순수 de-leverage(fractional_sizing_v1.csv 이미 증명). 사이징은 음수 net을 양수로 못 뒤집음.
- **합성 옵션 payoff 사이징**: paper_ledger에 exit_*/path_* 컬럼 전무('이미 기록중' 주장 거짓). 하방 leg(next_min≤−5% 57%)가 상방(next_max≥5% 41%)보다 자주 먼저 침 → E[payoff] 구조적 음수.
- **켈리 자본곡선 가드**: position_size_pct/net_return_pct 컬럼 없음. MDD 축소=노출 0 수렴(98% 일수 multiplier<0.05)=사실상 침묵. day-quality 연속판.

**REDUNDANT (이미 측정·판정된 트랙 재포장):**
- **과확장 침묵 게이트**: downside_aware_compare_v1.csv cut=0.667 row가 정확히 그 실험(SHADOW 2026-06-01). 핵심 thesis 피처(신고가근접 AUC 0.485 역부호·연속상승 0.521)가 가장 약함. **→ 신규 트랙 금지, A1 SHADOW→ADOPT 승격 재평가만.**
- **합성 옵션 페이오프 청산**: 볼록 trailing은 exit_challenger_compare_v1(Δnet −0.000159) REJECT, 비대칭 bracket은 forward 누적중.
- **확산-폭주 분리 가드**: R3_gate가 이미 deep_dump 0.280→0.145 줄였으나 net 음수, R2_penalized에 strictly dominated.
- **확산-콘 추적바닥**: i=0에서 √(i/96)=0이라 floor=running_high≈entry(보호 0). vol룰과 동일 실패축. 48개 정책 이미 sweep, 전부 net 음수.
- **확산 하방게이지 진입 재정렬**: recommend.py rr_ratio가 이미 라이브 챔피언. bear_quiet sl_first_share 0.46~0.54(동전던지기)로 '고ATR=구조적 SL-first' 전제 자체가 거짓.
- **HMM posterior 게이트**: ch_regime_split이 이미 동일 metric으로 하드 발사/기권 sweep, net 음수. 경계날(intensity 0.45~0.55) 8.1%에만 차별화 갇힘.

---

## 즉시 실행 가능한 첫 실험 3개

### 실험 A — 델타-사다리 청산 (가장 강한 lever, 깨끗한 1회 반증)
```
대상: recommender_downside_exit_v1 OOF pick set (K=3, lab_pump20)
절차:
 1. signals/validate.py:PurgedWalkForward(5-fold + embargo 5d)
 2. exit_lab에 delta_ladder 변형 추가: arm +2/+4/+6% thirds,
    floor −5%, fill=arm_px(봉 high 아님!), 청산횟수만큼 0.0015 차감
 3. simulate_path 인프라로 fold별 net·deep_loss·CVaR95 산출
 4. 챔피언 TP5/SL3와 bootstrap CI95 비교
게이트: deep_loss freq ≤ 챔피언 AND net Δ CI95 하한 ≥ 0 인 fold ≥ 4/5
        (미달 시 REJECT — 볼록꼬리 가설 반증으로 종결)
검증: test_exit_lab.py에 단일청산=walk_path byte-동등 회귀 추가
```

### 실험 B — 군중쏠림 differential (honest net 필수)
```
대상: paper_ledger_backfill.csv (5080 OOF, 754일)
절차:
 1. 코인별 D-1 연속랭크(composite_score 상위 + roc/return rank ≥3일 지속)로
    과열/비과열 2버킷 라벨링
 2. honest net 계산 — next_close 기반 OR exit_lab 15m sim
    ★ max-bracket(high≥+5%→+5% 확정) 절대 금지 (lognormal 환상)
 3. 동일 setup_quality·regime 통제하 purged WF + bootstrap CI95
게이트: 비과열 버킷이 net CI·deep_loss_freq 둘 다 과열 대비 유의 개선
        → SHADOW 승격 / 양쪽 동등 음수 → REJECT
```

### 실험 C — MC 브래킷 generator sanity (순환검증 회피)
```
대상: 1 fold bear_quiet 고변동 코인 5~10개
절차:
 1. D-1 vol로 GBM + heavy-tail SDE 적합 → N=5000 합성경로 96-step
 2. EXIT_VARIANTS 6종별 P(SL-first) 예측
 3. 같은 fold test의 실제 15m 경로(load_paths)에 walk_path 적용 →
    실현 SL-first 및 net(0.15% 차감)
 4. reliability·net 일치도 1장 표 (대조군 고정 TP5/SL3)
게이트: MC가 실현 SL-first를 ±0.05 내로 맞추는지 먼저 확인
        못 맞추면 generator가 틀린 것 → 즉시 중단
        (캘리브레이션 구간 ≠ 검증 구간 fold 분리 필수)
```

> 세 실험 모두 신규 데이터 수집 0, 기존 검증된 causal simulator(walk_path/simulate_path) 재사용, purged WF + bootstrap CI 강제.
> 공통 철학: **net 양전을 노리지 말 것 — radar not strategy. 측정 목표는 deep_loss freq↓·CVaR95↓를 net 비악화로 달성하는가(하방 lever).** 셋 다 명확한 kill-gate 내장 → negative result도 보존.
