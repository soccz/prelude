# Day-Quality Gate v1 — 연구 노트 (signal-researcher)

**날짜**: 2026-06-11 · **판정**: **REJECT** (현 형태) + 재고 조건(pump20 richness axis) 명시
**스크립트**: `scripts/day_quality_gate_v1.py` (재현 가능, `python -c 'runpy.run_path(...)'` 로 실행)
**산출**: `output/day_quality_gate_v1_{daily,rules,wf,replay}.csv`, `_summary.json`

---

## 1. 가설
- **연구 질문**: "어떤 코인"이 아니라 "어떤 날" — 펌프가 없는 죽은 날을 D-1 정보만으로 식별해
  그날 알림 0개(침묵)로 가는 gate 가 시스템 픽의 net 을 개선하는가?
- **메커니즘 가설**: 펌프는 시간적으로 클러스터링한다 — D-1 시장 전체의 pump 수가 D 의 pump 풍부도를 예측한다.

## 2. 방법 (무엇을 돌렸나)
- **시장-레벨 라벨**: 일별, D-1 거래대금 top-100 유니버스(기존 `recommend.py` convention 그대로) 내
  `n_pump10 = #{high_D/open_D-1 ≥ 0.10}`, `n_pump20 (≥0.20)`. **1233 일** (2023-01-26~2026-06-11).
- **피처 (전부 ≤ D-1)**: n_pump10 의 lag1/lag2/3d/5d 평균(클러스터링), n_pump20 lag, breadth(양봉비율),
  횡단면 수익 분산, 시장 총거래대금 vs 20d, BTC ret/RV/MA20. 핵심: **일별 라벨 시계열을 만든 뒤
  `.shift()` 로 lag feature 생성** → row D 는 D-1 까지만 본다.
- **룰 sweep**: 단순 1~2 피처 임계 27 조합 (l1≥τ, 3d≥τ, breadth≥τ, l1&breadth, btc_rv≥τ).
- **검증**: pump-rich day = n_pump10 ≥ K(=median=6, base rate 0.52). Purged WF 5-fold, embargo 5일,
  per-fold OOF threshold.
- **★ Replay**: gate(`n_pump10_l1≥6`, **threshold 는 pre-2025-11 데이터에서만 선택** → replay 기간과 분리)
  를 5개 ledger 의 closed 픽에 `date` 로 부착, gate-ON vs ALL vs OFF net 비교 (거래비용 0.15% 차감).
  exit 변형 4종(realized=TP5/SL3, TP10_noSL, TP5_noSL, EOD). paper_ledger 는 path 극값으로 TP5/SL3 합성
  (SL-먼저 비관 가정 — downside-first 와 일치).

## 3. leak / 시간정합성 방어
- 피처: 일별 라벨 시계열 → `shift(1)+` 로만 lag. row D 에 D 의 high/open 안 들어감.
- 유니버스: D-1 `quote_volume` top-100 (qv 도 `shift(1)` 후 랭크). same-day volume 미사용.
- 라벨(n_pump_D)은 day-D high/open = D 09:00 의사결정 시점엔 미래 (정상 셋업).
- replay leak 차단: gate threshold 를 **replay 기간(2025-11+) 밖**에서만 선택. 또한 룰이 1-파라미터
  임계라 overfit 자유도 거의 0.
- 시도 조합: **27** (evaluator 가 deflate 할 수 있게 기록).

## 4. 1차 결과

### (a) "죽은 날" 전제가 pump10 에선 거짓
| | ==0 | ≤1 | ≤2 | median | autocorr(D vs D-1) |
|---|---|---|---|---|---|
| pump10 | **2.8%** | 9.7% | 19% | 6 | 0.39 |
| pump20 | **32.2%** | 57% | 75% | 1 | **0.52** |

→ 시장은 pump10 기준 거의 안 죽는다(90% 날이 ≥2 펌프). "펌프 없는 날" 은 pump20 에서만 의미.

### (b) 클러스터링 시그널은 진짜 + WF 안정 — 단, gate 로는 약함
- 룰 `n_pump10_l1≥6`: precision 0.75 vs base 0.52 → **lift 1.44**.
- WF per-fold test_lift: fold1 1.51 / fold2 1.61 / fold3 1.22 / fold4 1.06 — **한 번도 역전 안 함** (안정).
- BUT autocorr 0.39 → 분산 ~15% 만 설명. 좋은날/죽은날 깔끔히 못 가른다.

### (c) ★ Replay — gate-ON net 개선 (거래비용 차감, realized=TP5/SL3 가 라이브 exit)
| ledger | n_dates(ON) | silence% | net_mean ALL | net_mean ON | **improve** |
|---|---|---|---|---|---|
| distribution | 25(20) | 20% | -1.785 | -1.746 | **+0.04** |
| recommend_R1 | 10(7) | 30% | -1.289 | -1.253 | +0.04 |
| recommend_R2 | 10(7) | 30% | -1.334 | -1.356 | **-0.02** |
| recommend_sustain | 9(6) | 33% | -1.007 | -0.829 | +0.18 |
| pump_hunter | 7(4) | 43% | -1.257 | -1.457 | **-0.20** |

- **realized(TP5/SL3)에선 개선 ~0 또는 음수.** 모든 ledger net 여전히 음수(gate-ON 도).
- 더 커 보이는 개선(예: distribution TP5_noSL +0.31, sustain +0.39)은 전부 **noSL 변형** —
  라이브 미사용 + Phase X+6 가 이미 deep-loss 로 더 출혈한다고 결론낸 exit. gate 가 "도와주는" 게
  하필 더 지는 exit.
- **pump_hunter(105 픽, 최대 표본)는 모든 exit 에서 gate-ON 이 더 나쁨** — 가설과 정반대.

### (d) 개선은 무작위 날짜 선택과 구분 불가 (핵심 반증)
- distribution(유일하게 날짜 충분) **permutation test**: gate-ON improve +0.040 vs
  null(같은 날짜수 무작위) mean -0.005 / std 0.123 → **p = 0.426**. 신호 0.
- pump_hunter per-date: 유일한 호황일(2026-06-07 +1.80%)이 **gate-OFF (침묵 대상)**,
  gate-ON 4일 중 3일이 손실. gate 가 그 좋은 날을 죽였을 것.

## 5. 정직한 판정: **REJECT** (현 형태)
이유 3가지:
1. **전제 반증** — pump10 죽은 날 2.8% 뿐. gate 가 막을 "출혈 날" 이 통계적으로 거의 없음.
2. **replay 개선이 noise** — 라이브 exit(TP5/SL3)에서 개선 ~0, permutation p=0.43, 최대 표본 ledger 는
   오히려 악화. net 은 gate-ON 도 전부 음수 → 12-실험 천장과 일치(gate 가 음수를 양수로 못 바꿈).
3. **표본 기근** — replay distinct date 7~25개(gate-ON 4~20). 며칠 차이는 sampling error.
   "좋은 날만 1개" 를 검증하기엔 라이브 표본이 절대 부족.

→ ops 배선(policy_competition send-policy) **하지 않음**. cron/텔레그램 미배선(스펙 준수).

## 6. 재고 조건 (버리지 말 것 — evaluator/사용자 판단용)
- **pump20 richness axis 가 진짜 후보**: dead-day 32% (의미 있는 침묵 여지) + autocorr **0.52**
  (clustering 더 강함). 단 현재 ledger 들은 pump10/pump20-prob 픽이라 richness gate 와 결이 다르고,
  무엇보다 표본이 없다. **live paper 로 pump20 richness gate-ON/OFF 양쪽을 수개월 기록**해야 결판.
- gate 를 "픽 제거"가 아니라 **사이징 0↔1**(richness 낮은 날 size down)로 쓰면 downside-first 와 맞을 수
  있으나, 이는 ledger 사이징 룰 변경 = 사용자 권한. 데이터부터.
- 만약 다시 본다면: 라벨=pump20, K=분위 기반(예 q75), replay 는 표본 누적 후 + per-fold replay
  분리(gate train fold 에 ledger 날짜 배제)로 leak 완전 차단.

## 7. quant-evaluator 가 검증할 지점
1. **permutation p=0.43** 재현 — 내 결론(개선=noise)의 핵심 근거. 다른 ledger 도 권장(표본 허락 시).
2. **noSL 변형 개선의 함정** — "improve_mean 양수" 가 라이브 미사용 exit 에서만 크다는 점이
   채택 착시 아닌지. realized(TP5/SL3)만 보면 판정이 바뀌는가.
3. **selection bias**: 27 룰 조합 시도 → lift 1.44 의 deflate. WF lift 안정성(역전 0)이
   deflate 후에도 살아남는지.
4. **표본 기근 경고가 충분한지** — 7~25 date replay 로 SHADOW 신청했다면 거부했어야 할 사안.
   내 REJECT 가 과한지/적정한지.
5. (위생) gate threshold 를 pre-2025-11 에서만 골랐지만 daily 라벨 자체는 full-history.
   richness K(=median) 가 replay 기간 포함해 계산됨 → 미세 leak 여부 점검(영향 낮다고 봄: K 는 라벨 정의,
   gate 결정엔 미사용).
