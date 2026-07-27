# 연구 노트: cold-start pump v1 — 장중 초기 신호(첫 15m bars)로 cold-start 펌프 당일 포착

작성: signal-researcher / 2026-06-11
스크립트: `scripts/coldstart_pump_v1.py` (self-contained, 재현 가능)
결과 CSV: `output/coldstart_pump_v1_{leaves,oos}_{1000_tp15, 0930_tp10}.csv`
로그: `output/coldstart_run_{1000_tp15, 0930_tp10}.log`

---

## 가설
**D-1 일봉 모멘텀이 없어(roc_7d rank < 0.7) 사전 포착 불가능한 cold-start 펌프(전체 펌프의 6/7)를, 당일 09:00 개장 직후 첫 몇 개 15m bar 의 거래대금 surge / 가격 미세동학으로 당일 중에 포착할 수 있는가?**

이전 연구(`pump_rule_discovery_v1`)에서 D-1 모멘텀 룰은 펌프의 ~1/7 만 잡고, 6/7 은 D-1 모멘텀 없이 당일 갑자기 터지는 cold-start 였다. 11실험 천장(D-1 일봉 진입 net 음수)의 진단은 "진입 집합 근본 변경 필요". 이 연구는 진입 집합을 **D-1 일봉 → 당일 장중 초기(intraday)** 로 바꿔 본다.

---

## 방법

### 데이터 (coverage 확정)
- `data/upbit_15m.db`: KRW 262 markets, 7.80M rows, 2023-05-03 ~ 2026-06-11. 96개 15분 격자(00:00~23:45) 온전.
- `data/upbit_d1.db`: 252 markets, quote_volume 100% 가용, coins/day median 125.
- **timestamp 의미 확정(probe)**: 15m timestamp 는 벽시계 KST, bar **시작** 시각. d1 일봉(09:00 KST 시작) open == 15m **09:00 bar** open 이 n=85,474 에서 **100% 매칭(median rel_err 0.00bp)**. → `trade_day = (ts-9h).date()` 로 09:00~다음날 08:45 를 같은 거래일로 묶음 = d1 일봉 D 와 완전 정합.

### 결정 시점 / 라벨 (decision-time entry)
- 결정 시점: 09:30(dec_min=30, pre-bar=09:00 1개) / 10:00(dec_min=60, pre-bars=09:00~09:45 3개). [11:00 은 base rate↓로 미실행 — 아래 판정 참조]
- 진입가 `p_dec` = 결정 bar(min_since_open==dec_min)의 **open**.
- 라벨 y = (resid_high / p_dec − 1 ≥ TP). resid_high = 결정 bar 포함 그날 잔여시간 max(high). TP ∈ {0.10(09:30), 0.15(10:00)}.
- exit (net): TP 도달 시 +TP 체결(낙관), 미도달 시 EOD(거래일 마지막 bar) close 청산. **SL 없음(noSL)** — bear_quiet 교훈(일찍 자르면 손해) 반영.

### 피처
- **Intraday (결정시점 strictly 이전 bars = min_since_open < dec_min)**: cum_ret(09:00 open 대비), qv_surge(누적 거래대금 / 자기 과거 20거래일 같은-창 평균), up_bar_frac, range_pct, body_pos(체결강도 proxy), max_bar_ret, pos_vs_dayopen, pdec_vs_prehigh(돌파 proxy), n_pre.
- **D-1 일봉 컨텍스트(전일까지, shift(1))**: roc7_rank, roc3_rank, atr_rank, atr_pct_14, log_ret_1d, ret_5d.
- 총 15 feature. depth≤3, min_leaf=200, class_weight=balanced decision-tree leaf mining (pump_rule_discovery 방식).

### 유니버스 / cold-start 필터
- D-1 거래대금 top120(liq_rank ≤ 120, **전일 기준**).
- cold-start: roc7_rank < 0.70 (top120 의 66.7% = 이전 연구 "6/7" 와 일치). 모멘텀 룰 사각지대만 남김.
- 패널: 30,593(tp15) / 30,605(tp10) coin-days, 2023-05-14 ~ 2026-06-11.

### Leak 방어 (양보 X — same-day leak 2번 전적, 이번에 통과)
1. **시점 분리(핵심)**: 피처는 min_since_open < dec_min bars 만. 라벨/exit 는 min_since_open ≥ dec_min. 결정 bar 는 라벨 쪽이고 그 **open 만** 진입가로 씀(개장 첫 체결가 = 결정 순간 관측 가능). 같은 bar 의 high 는 라벨에만, open 만 진입가에 — 양쪽에 같은 정보를 쓰지 않음.
2. **D-1 컨텍스트 shift(1)**: roc/atr/ret_5d/log_ret_1d 전부 전일 종가까지로 밀어 부착. cross-sectional rank 도 전일 값 기준.
3. **qv_surge 분모**: 자기 과거 20거래일 같은-창 cum_qv 평균(shift(1).rolling), 전부 과거.
4. **유니버스 rank**: 전일(D-1) 거래대금.
5. **Purged WF**: 날짜 fold, embargo 7일, holdout 120일. 각 (market, trade_day) = 1 row 이므로 coin-day dedup 자동.
6. **시간 분산 확인**: pump15 월별 2~5%, 특정 한 달 몰림 없음(2025-11 5.1% ~ 2026-03 2.1%).
7. **성능 sanity**: 결과가 약함(아래) → leak 의심 없음. (강했다면 leak 재감사 대상이었음.)

### 시도 조합 수 (selection bias 기록)
- decision × TP 조합: **2개 실행**(10:00/tp15, 09:30/tp10). 11:00·tp0.20 미실행.
- 조합당 WF fold 3개(3-5) × 1 tree = 6 tree. leaf rows 21 + 20 = 41.
- hand-pick 은 OOS(val) lift + gap + net 본 뒤. tree threshold 는 데이터 자동 선택.
- min_pre 결손 필터(09:30→1, 10:00→3 bar 이상)로 데이터 빈 날 제외.

---

## 결과 (fold 별 — net 은 0.15% 표준 + 0.5% 보수 병기)

### whole-fold OOS (rule-free, cold-start 유니버스 무차별 진입 net)
| 조합 | fold | n_va | base% | allpick net% | net(보수 시사) |
|---|---|---|---|---|---|
| 10:00/tp15 | 3 | 4931 | 1.18 | **−0.18** | 더 음수 |
| 10:00/tp15 | 4 | 8332 | 2.02 | **−0.08** | |
| 10:00/tp15 | 5 | 8446 | 3.82 | **−0.36** | |
| 09:30/tp10 | 3 | 4932 | 4.46 | **−0.25** | |
| 09:30/tp10 | 4 | 8331 | 5.22 | **−0.13** | |
| 09:30/tp10 | 5 | 8451 | 8.99 | **−0.41** | |

→ **cold-start 유니버스 무차별 진입은 net 전부 음수**. 더 이른 진입(09:30)+낮은 TP(0.10)이 오히려 더 손실(잔여시간이 길어 noSL 보유 중 drift down 비용↑).

### cross-fold robust root (lift_va ≥ 1.8 in ≥3 folds)
- **두 조합 모두 (none).** fold 간 일관되게 작동하는 root feature 가 0개.

### 최선 leaf 들 (net 기준 상위) — 전부 단일 fold, 작은 n, intraday 아님
| 조합 | fold | rule | n | lift | gap_pp | recall% | net% | net보수% |
|---|---|---|---|---|---|---|---|---|
| 10:00/tp15 | 5 | `atr_pct_14≤0.097 AND max_bar_ret>0.025` | 223 | 4.81 | **−11.4** | 12.7 | +1.71 | +1.36 |
| 10:00/tp15 | 5 | `atr_pct_14>0.097 AND ret_5d≤−0.189` | 427 | 2.82 | +4.4 | 14.2 | +0.45 | +0.10 |
| 09:30/tp10 | 3 | `range_pct≤0.010 AND atr_rank>0.536 AND ret_5d≤−0.159` | 76 | 2.06 | +5.0 | 3.2 | +1.89 | +1.54 |
| 09:30/tp10 | 5 | `ret_5d≤−0.183 AND ret_5d>−0.254` | 559 | 2.49 | −4.2 | 16.4 | +0.90 | +0.55 |
| 09:30/tp10 | 4 | `range_pct>0.011 AND log_ret_1d>−0.118 AND pos_vs_dayopen≤0.028` | 1595 | 2.10 | −1.3 | 40.2 | +0.34 | −0.01 |

관찰:
- net-positive 쪽으로 나온 룰의 root 는 **거의 전부 D-1 일봉(ret_5d 깊은 음수, atr_rank/atr_pct_14)** — 즉 "전일~5일 급락한 high-vol 코인의 반등(mean-reversion / falling-knife bounce)"이지 장중 ignition 이 아님.
- **intraday 미세동학(qv_surge, body_pos, up_bar_frac, max_bar_ret)** 은 net-positive robust leaf 의 root 로 거의 등장 안 함. max_bar_ret 가 일부 등장하나(10:00 fold5) gap −11.4pp = val 행운(overfit).
- high-recall leaf(recall 25~40%, 예: 09:30 `...pos_vs_dayopen≤0.028` recall 40.2%)는 net ≤ 0. **EDA-hit ≠ Sharpe 함정의 전형**: 펌프 방향은 일부 맞히나 진입하면 비펌프 drift + noSL EOD 보유로 손실.

---

## 정직한 판정: **REJECT (이 형태로는 SHADOW 불가)**

판정 기준(OOS lift ≥ 2x + fold 일관 + 시간 분산 + net 양수 → SHADOW)에 대조:
- lift ≥ 2x: 일부 leaf 만족하나 **fold 일관 = 실패**(cross-fold robust root 0개, 양수 net 은 전부 fold5 한 곳에 집중).
- net 양수: **whole-fold OOS net 전부 음수**. 양수 leaf 는 단일 fold·작은 n·gap 음수(overfit)·보수비용에서 소멸하는 fragile.
- 시간 분산: 패널 전체는 분산되나 **신호 자체가 특정 fold(5)에 집중** = 표본 시간 집중의 나쁜 형태.

핵심 결론(가설 반증): **"장중 첫 15m~60m 거래량/가격 미세동학으로 cold-start 펌프를 잡는다"는 가설은 이 구현/유니버스/horizon 에서 데이터로 반증.** intraday surge 피처는 leak-free net edge 를 내지 못했다. 그나마의 미약한 신호는 intraday 가 아니라 "전일 급락 후 반등"이라는 **다른 현상**이며, 이마저 fold 일관성·net 양수를 못 넘었다. 이는 이전 두 연구(D-1 모멘텀 룰 1/7 천장, bear_quiet)와 11실험 천장 진단과 **일관**된 음성 결과다.

### 어느 조건이면 재고 가능한가 (다음 각도)
1. **exit 규율 분리 연구**: 음수 net 의 주범은 진입 신호가 아니라 **noSL + EOD 보유** 일 수 있음. 같은 cold-start 유니버스에서 (a) intraday trailing-stop, (b) TP 도달 즉시 청산 비중↑, (c) 09:00 이후 N분 내 미발화 시 즉시 손절 → exit 만 바꿔 net 천장 재측정. (메모리: lever 는 exit/하방 규율이라는 사용자 방향과 일치.)
2. **더 짧은 horizon / 더 미세한 bar**: 15m 는 ignition 포착에 거칠 수 있음. 1m/3m bar(있으면)로 첫 5~15분 체결 급증(틱 단위 거래량 폭발, 호가 불균형 proxy)을 재시도. cold-start 는 분 단위 현상일 가능성.
3. **돌파(breakout) 전용 라벨**: "잔여 max high ≥ TP"가 아니라 "09:00~결정시점 고가를 W% 돌파 후 유지" 같은 모멘텀-확인 라벨로 mean-reversion 노이즈 분리.
4. **유니버스 확대 검토**: top120 밖(소형주)에서 cold-start ignition 빈도↑ 가능하나, 슬리피지·체결 가능성 악화 → net 더 불리할 위험. 별도 비용 모델 필요.
5. **intersection(교집합) 신호**: intraday surge × D-1 변동성(atr_rank 상위) 교집합이 단독보다 나은지 — 단 현재 tree 가 이미 교호작용을 탐색했고 일관 신호 없음. 우선순위 낮음.

판정을 뒤집으려면 **whole-fold OOS net 이 ≥2 fold 에서 양수 + cross-fold robust root ≥1** 이 최소 조건. 현재는 둘 다 미달.

---

## quant-evaluator 가 검증해야 할 지점
1. **leak 재감사(형식)**: 결과가 약해서 leak 가능성 낮음. 단 형식 점검은 — (a) 결정 bar open 만 진입가, high 는 라벨로 간 분리가 코드(`build_intraday` pre/post min_since_open 경계)에서 누락 없는지, (b) qv_surge 의 shift(1).rolling 이 당일 cum_qv 를 분모에 안 넣는지, (c) D-1 컨텍스트 shift(1) 이 모든 6개 컬럼에 적용됐는지.
2. **음성 결과의 신뢰성**: whole-fold OOS net 음수가 진입 신호 부재 때문인지 vs exit(noSL+EOD) 페널티 때문인지 분해 요청. allpick_tp_hit% (1.2~9.0%) 는 base rate 와 동일(무차별이므로 당연) — TP 도달군의 평균 net 과 미도달군 EOD drift 를 따로 보면 exit 페널티 크기 추정 가능.
3. **selection deflate**: 양수 leaf 5개는 41 leaf 중 hand-pick, 전부 fold5 또는 작은 n. 단일 fold 집중 + gap 음수는 deflate 시 거의 소멸 예상 — DSR/시도수 보정 후 0 에 수렴하는지 확인.
4. **placeholder 재고**: cs_cut 0.70, top120, hist_window 20, embargo 7, holdout 120 은 전부 초기값. 음성 결론이 이 값들에 민감한지(특히 cs_cut, exit 규칙)는 위 "재고 조건 #1"에서 exit 재설계와 함께 재측정 권고.
