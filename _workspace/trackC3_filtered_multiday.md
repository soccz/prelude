# 연구 노트: Track C3 — 하방-선별 진입(A1 필터) + 멀티데이 홀딩

- **작성**: signal-researcher, 2026-06-01
- **상태**: 증거 생성 완료. 채택 결정 X (quant-evaluator ADOPT/SHADOW/REJECT 대기).
- **산출 파일**:
  - 코드: `/mnt/20t/prelude/scripts/cc_filtered_multiday_v1.py` (NEW, self-contained)
  - 비교표: `/mnt/20t/prelude/output/cc_filtered_multiday_compare_v1.csv` (112 조합)
  - 커버리지/메타: `/mnt/20t/prelude/output/cc_filtered_multiday_coverage_v1.json`
  - 픽 dump(감사): `/mnt/20t/prelude/output/cc_filtered_multiday_picks_v1.csv` (bracket gap 변형 trade-level)
  - 캐시 OOS: `/mnt/20t/prelude/output/cc_filtered_multiday_oos_v1.parquet` (재실행 `--use-cache`)
  - 실행 로그: `/mnt/20t/prelude/output/cc_filtered_multiday_run_v1.log`

---

## 가설 (두 선행 실험의 합성)
- **B3 멀티데이**(REJECT): R1 *unfiltered* top-3 진입 위에선 보유를 늘릴수록 손실꼬리가 먼저·더
  깊이 산다. "수익꼬리·손실꼬리 분리불가"는 *현 R1 진입집합 내부*에서만 성립.
- **A1 sustainability**(SHADOW): D-1 dump head 로 깊은-하방 후보를 진입에서 배제 → deep-loss 절반.
- **C3 가설**: 하방품질로 *사전선별한 부분집합* 위에서의 멀티데이가 분리 가능성이 남은 유일한 조합.
  저-하방 픽만 N일 보유하면 손실꼬리 폭증이 억제되어 net 이 양수/개선될 수 있나?
- **★핵심 질문**: 하방선별이 멀티데이의 손실꼬리 폭증을 막아 net 이 양수/개선되는 (필터,N,청산)
  조합이 *갭반영 보수청산 후에도* 존재하는가?

## 무엇을 돌렸나
- **진입집합 4종** (같은 OOS 765일, 같은 정적 top100 유니버스, n_picks=2295 전부 동일):
  - `R1_unfiltered` = R1 top-3 (B3 baseline 재현)
  - `C3_filt_q{0.6,0.7,0.8}` = A1 dump_B re-selection (p_dump>cutoff 강등→다음 R1 후보로 교체).
    cutoff = fold<k(train OOS) p_dump 의 q 분위 (OOF, train-only). q 낮을수록 공격적 배제.
    교체율 sub: q0.6=0.69, q0.7=0.61, q0.8=0.48 (coverage 1.00 — 항상 top-3 채움).
- **멀티데이 청산** (일봉 d1 경로), N∈{1,2,3,5} × {holdN, bracket(TP{8,12}/SL5), trail(DD8)}
  × 체결모델 {opt=레벨정확체결, gap=갭반영 보수}. holdN 은 종가청산이라 gap 개념 없음.
  → 총 **112 조합** (4 진입집합 × N × 청산 × 체결).
- **체결모델 gap(보수)** = B3 evaluator 의 trail 갭다운 21.7% 낙관편향 지적 반영: 봉 open 이 이미
  SL/trail 레벨 아래로 갭다운했으면 그 open 가격(더 나쁨)에 체결. TP 갭업 이득은 안 줌(보수).
- 비용 0.15% 1회. 지표: net_mean·median·hit·deep_loss(≤−5%)·worst·CVaR95·MaxDD·cum·
  block-Sharpe/Sortino(overlapping 보정)·nominal-Sharpe·excess-over-market·avg_hold·gap체결율.

## leak·시간정합성 방어 (4/4 — 양보 X)
- **시점 분리**: 입력 feature ≤ D-1 (build_market_features shift(1)). dump 라벨은 day-D
  open/high/close(미래) = head 학습 타겟이지 입력 아님(`lab_dump_B` ∉ feats, feats 전부 `f_` prefix).
  head/cutoff 모두 **train-only fit**(test fold 적용만), embargo=5, fold0 강등 안 함.
  진입가 = day-D open(관측가능), 청산경로 day-D~D+N-1 = forward outcome.
- **자가검증**: `R1_unfiltered` 가 B3(ch_multiday_v1) R1_ratio 결과를 충실 재현
  (holdN N=1 −0.0015·N=5 −0.0114, deep 0.160→0.368, worst −0.338→−0.630 = B3 노트와 일치).
  파이프가 leak 했다면 이 재현이 어긋났을 것 → leak 알람 OFF. **net 음수 = 과신 신호 없음.**
- **유니버스 시간정합**: top100 = D-1 qv_rank. A1 와 동일.
- **overlapping**: N일 홀딩 인접 진입 겹침 → non-overlapping block(진입일 N간격)으로 Sharpe/Sortino
  재계산(N=2→383블록, N=5→153). 명목 Sharpe 낙관편향이라 block 우선.
- 자동주문 X / 공유 라이브 파일 미편집(NEW 파일만).

## 시도 조합 수 (selection deflate 용)
**112 조합** (진입집합 4 × N 4 × 청산 ~4 × 체결 1~2). 격자 사전고정, hand-pick 후행 없음.
+ A1 dump 라벨/cutoff 선택은 선행 실험에서 이미 결정(dump_B 우세). → evaluator 가 112 trials
(+ A1 8 trials 상속)로 deflate.

## 1차 결과 (OOS 765일, n_picks=2295, net 0.15% 차감, 일봉 경로)

### ★ 핵심 질문 답: **아니오 — 갭반영 후 net>0 조합 0개.**
- **전체 112 조합 중 net>0 은 단 3개, 전부 `trail opt`(낙관 체결):**
  R1_unfiltered N=5 +0.0011, R1_unfiltered N=3 +0.0003, C3_filt_q0.8 N=5 +0.0002.
- **gap-aware(현실) 체결 48조합 중 net>0 = 0개.** 위 trail opt 양수는 갭반영하면 즉시 붕괴:
  C3_filt_q0.8 N=5 trail opt +0.0002 → gap −0.0054 (gap체결율 14%). 즉 그 양수는 갭다운을
  레벨가로 체결한다고 가정한 **낙관 잔여**일 뿐, 현실에서 사라진다 (B3 evaluator 지적 재확인).
- **net>0 3개 모두 excess_mkt<0** (−0.0032~−0.0059) → 베타(상승장 보유프리미엄)조차 못 이김.
  net>0 AND excess>0 인 조합 = **0개**.

### 필터는 손실꼬리를 줄였다 (A1 효과 멀티데이에서도 재확인) — 단 net 은 못 살림
holdN(청산 confound 없는 순수 보유) 기준, 같은 N 에서 C3 가 R1 대비 deep-loss·worst 일관 개선:
| N | deep R1 → C3_q0.6 | worst R1 → C3_q0.6 | net R1 → C3_q0.6 |
|---|---|---|---|
| 1 | 0.160 → **0.097** | −0.338 → −0.301 | −0.0015 → −0.0020 |
| 2 | 0.256 → **0.170** | −0.472 → −0.339 | −0.0044 → −0.0013 |
| 3 | 0.308 → **0.236** | −0.604 → **−0.291** | −0.0064 → −0.0038 |
| 5 | 0.368 → **0.313** | −0.630 → **−0.400** | −0.0114 → −0.0066 |
- 필터는 멀티데이의 손실꼬리 폭증을 **부분적으로** 눌렀다(특히 worst: N=3 −0.604→−0.291,
  N=5 −0.630→−0.400 = "더 오래 보유해도 깊은 갭다운이 덜 산다"). 사용자 하방선호엔 부합.
- **그러나 net 부호는 안 바뀜.** 필터가 net 을 R1 대비 개선(N=2/3/5)하지만 양수 전환 못 함.
  N 커질수록 양 진입집합 모두 net 악화 — 필터는 악화 *속도*만 늦출 뿐 방향을 못 뒤집는다.

### 베타 분리 + block-bootstrap (덜 거래 X, 진짜 교체 효과)
- coverage 1.00·n_picks 2295 전부 동일 → "덜 거래" degeneracy 아님. 순수 픽 *교체* 효과.
- **필터 net 개선은 통계적으로 0과 구분 불가.** 현실(bracket gap) 셀 기준 block-bootstrap
  Δnet(C3_q0.6 − R1): N=2 +0.0009 CI95[−0.0016,+0.0036]·N=3 +0.0006 [−0.0024,+0.0036]·
  N=5 +0.0006 [−0.0028,+0.0040] — 전부 CI 가 0 포함, P(Δ<0) 0.23~0.36. 방향성만 있고 유의 X.
- **모든 셀 excess_mkt<0.** R1 picks 든 필터 picks 든 N일 보유는 같은 기간 시장 평균보다 못함.
  net 의 N-의존 변화는 베타 추종일 뿐, picks 고유 멀티데이 알파는 음수.

## 결론 (증거 생성까지 — 채택 판정 X)
**C3 가설(하방선별이 멀티데이의 손실꼬리를 막아 net 양수전환) 은 falsified.** 필터는 손실꼬리를
*부분적으로* 눌렀지만(deep-loss·worst 일관 개선) net 부호를 못 뒤집었고, 갭반영하면 낙관 잔여
net>0 마저 0개로 사라진다. excess<0 으로 베타조차 못 이김. = B3 의 "멀티데이 dead end" 진단이
*하방선별 부분집합 위에서도* 유지된다. 단 의미 있는 부산물: **A1 필터의 하방 억제 효과는 멀티데이
홀딩에서도 작동**(특히 worst tail), net 이 아니라 *하방규율* lever 로서의 A1 가치를 재확인
(메모리 "lever 는 exit/하방규율" 과 일관). → 멀티데이 축은 진입집합을 바꿔도 net 흑자 불가.

## evaluator 가 볼 것 (3줄)
1. **갭 낙관 잔여 검증**: net>0 3개 전부 `trail opt` 이고 gap 모델에서 −0.005 대로 붕괴(gap체결율
   10~14%). 내 gap 모델(봉 open≤레벨 → open 체결, TP 갭업 이득 미부여)이 *충분히* 보수적인지,
   아니면 일중 추가 슬리피지(트레일 발동봉 체결가)로 더 나빠야 하는지. 어느 쪽이든 결론(net>0 없음)은 불변.
2. **필터 net 개선이 0과 구분되나**: block-bootstrap Δnet(C3−R1) +0.0006~+0.0009 CI95 전부 0
   포함(P(Δ<0)≈0.23~0.36). 112-trial(+A1 상속) deflate 시 사실상 0. deep-loss/worst 개선은
   신뢰도 높음(holdN noSL-경로상 일관) — 이게 A1 하방 lever 의 진짜 신호. net 은 아님.
3. **excess-over-market**: net 의 N-의존 변화가 베타인지 확인(모든 셀 excess<0, net>0 셀도 excess<0).
   시장바스켓을 비용 미차감 총수익 proxy 로 썼으니(보수적으로 picks 에 불리) 베타 과대차감 여부 점검.
   결론: picks 고유 멀티데이 알파는 음수 — 진짜 엣지 아님.
