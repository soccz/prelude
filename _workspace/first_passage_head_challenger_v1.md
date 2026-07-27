# First-passage 상방 헤드 챌린저 v1

작성일: 2026-07-25

상태: **연구 완료 / REJECT / 프로덕션 변경 없음**

## 0. 사용자가 받는 것

이번 작업의 결과물은 자동매매 코드나 새 텔레그램 추천이 아니다. 실제
09:10 전후 알림 뒤의 15분봉 경로를 직접 목표로 학습한 상방 헤드 2개를
동일 조건에서 검증한 **별도 연구 판정서와 재현 가능한 산출물**이다.

- 고정 장벽 헤드: `+10%`가 `-5%`보다 먼저 오는지 직접 예측
- ATR 장벽 헤드: `+2×D-1 ATR14`가 `-1×D-1 ATR14`보다 먼저 오는지 예측
- 비교 대상: repaired R1, 기존 safeup head, deterministic monkey,
  ATR Top3, 유동성 매칭 무작위
- 결과: 두 신규 헤드 모두 discovery 진입 조건을 통과하지 못했고,
  사전 고정 primary인 fixed head의 holdout 판정도 **REJECT**
- 따라서 현재 추천 시스템·알림·원장에는 아무것도 적용하지 않았다.

핵심 해석은 간단하다. fixed first-passage 헤드는 상승 후보를 찾는 판별력은
살아 있지만, 고변동 종목을 강하게 선호하여 사용자가 가장 싫어하는 하방도
동시에 크게 늘린다. 단독 추천 랭커로는 사용할 수 없다.

## 1. 잠근 연구 설계

### 1.1 데이터와 시점

- D1 DB의 전체 271개 마켓을 먼저 읽고, 각 날짜마다 **그 날짜 이전 완료
  일봉이 70개 이상인 마켓만** PIT 적격 처리했다.
- 적격 집합 안에서 D-1 quote volume 순위 Top100을 구성했다.
- 최종 path panel은 2025-05-02~2026-07-24, 449일 × 100개 =
  44,900행이다.
- 입력은 24개 D-1 피처뿐이며 타깃 이후 정보는 입력하지 않았다.
- 실행 경로는 `[D 09:15, D+1 09:15)`의 15분봉 96개다.
- KRW-BTC의 정확한 96-grid와 다음 경계 종결을 먼저 확인했다.
- 기준 코인 경로는 완전한데 대상 코인의 무체결 봉만 빠진 경우에는
  직전 종가 OHLC flat bar를 넣었다. 신규 상장으로 시작 가격 자체가 없는
  경로는 불완전으로 제외했다.
- 같은 봉에서 위·아래 장벽이 모두 닿으면 보수적으로 하방 선도달이다.
- TP5/SL3 경로도 같은 봉에서 SL을 먼저 처리했다.
- 모든 net 수익률에는 왕복 비용 0.15%를 정확히 한 번 차감했다.

경로 품질은 complete 28,741행, flat-filled 15,233행,
benchmark-gap 500행, 신규상장 horizon 불완전 426행이다. 타깃 완전 경로는
43,974행이며, bulk 구현과 canonical `assess_15m_window`를 20쌍
(완전 경로 10쌍 포함) 대조해 일치시켰다.

### 1.2 모델과 검증

- trial은 정확히 2개이며 hyperparameter sweep은 하지 않았다.
- 두 헤드는 동일한 XGBoost 사양을 사용했다.
  `180 trees / depth 4 / lr .05 / subsample .8 / colsample .8 /
  min_child_weight 5 / lambda 1.5 / seed 42`
- discovery는 5개 expanding outer fold와 각 fold 내부 3개 expanding
  OOF fold를 사용했다.
- outer와 inner 모두 5-date embargo를 적용했다.
- isotonic calibration은 outer train 안의 진짜 inner OOF 예측에만 맞췄다.
- 마지막 KRW-BTC-complete 180일
  (2026-01-24~2026-07-24)은 variant 선택 전 봉인했다.
- 다만 이 기간은 관련 연구가 이미 본 적 있는 기간이므로 virgin holdout이나
  깨끗한 사전등록 증거로 주장하지 않는다.

baseline artifact의 자체 WF embargo 때문에 discovery 5일
(2025-09-14~2025-09-18)이 whole-date로 빠졌다. 부분 날짜를 채우거나
점수를 보간하지 않고 날짜 전체를 제외했다.

- discovery 실제 prediction: 142일, 공통 Top3 완전 경로: 136일
- locked holdout prediction: 180일, 공통 Top3 완전 경로: 164일
- holdout baseline whole-date 누락: 0일

## 2. Discovery 판정

아래 delta는 신규 헤드 Top3 minus repaired R1 Top3이며 날짜 단위 paired
bootstrap 5,000회 결과다.

| 헤드 | safe FP delta | `dn5` delta | TP5/SL3 net delta | discovery 적격 |
|---|---:|---:|---:|---|
| fixed FP | +6.37%p | **+27.21%p** | -31.47bp, CI95 [-78.75, +14.62]bp | 아니오 |
| ATR FP | +1.72%p | **+11.52%p** | -22.65bp, CI95 [-60.18, +16.00]bp | 아니오 |

사전 선택 규칙은 다음 세 조건을 모두 요구했다.

1. safe first-passage point delta > 0
2. `dn5` point delta <= 0
3. net delta CI95 상단 >= 0

두 헤드 모두 두 번째 조건에서 명확히 실패했다. 따라서 discovery만 봐도
채택 후보는 0개였다. holdout에는 결과를 본 뒤 ATR/fixed 중 유리한 쪽을
고르지 않고, 사전에 primary로 정한 fixed head 하나만 diagnostic 목적으로
열었다.

## 3. Locked holdout 핵심 결과

모든 정책이 Top3 완전 경로를 가진 공통 164일, 492픽 기준이다.

| 정책 | safe FP | `up10` | `dn5` | SL-first | TP5/SL3 net/픽 |
|---|---:|---:|---:|---:|---:|
| **fixed FP** | **20.73%** | **25.81%** | **59.55%** | **58.94%** | -0.0154% |
| repaired R1 | 14.84% | 17.89% | 39.43% | 46.75% | -0.1504% |
| 기존 safeup | 19.72% | 23.78% | 54.07% | 59.35% | -0.1017% |
| ATR Top3 | 19.51% | 22.15% | 40.85% | 54.47% | -0.1599% |
| 유동성 매칭 | 9.76% | 11.59% | 36.18% | 50.00% | -0.4841% |
| monkey seed42 | 7.52% | 8.13% | 23.78% | 46.54% | -0.3546% |

fixed FP와 repaired R1의 날짜-paired 차이는 다음과 같다.

- safe FP: **+5.89%p**, CI95 **[+1.42, +10.16]%p**
- `up10`: +7.93%p, CI95 [+3.25, +12.60]%p
- `dn5`: **+20.12%p**, CI95 **[+14.23, +25.81]%p**
- SL-first: **+12.20%p**, CI95 **[+6.71, +17.68]%p**
- TP5/SL3 net: +13.50bp/픽, CI95 **[-25.32, +52.24]bp**

즉 safe FP 개선은 통계적으로 보이지만 하방 악화도 더 크고 명확하다. net
point estimate는 R1보다 낫지만 불확실성 하단이 음수다. 채택 게이트 세 개
중 safe-FP만 통과했고, downside non-worse와 net uncertainty non-adverse는
실패했다.

**최종 판정: REJECT. SHADOW에도 넣지 않는다.**

## 4. “변동성 재포장” 진단

fixed head의 full-universe 판별력 자체는 살아 있다.

| 구간 | raw AUC | 일별 macro AUC | vol-band 내부 AUC | score↔ATR Spearman |
|---|---:|---:|---:|---:|
| discovery | 0.698 | 0.714 | 0.665 | +0.502 |
| locked holdout | **0.705** | **0.716** | **0.681** | **+0.629** |

holdout에서 vol-band 안에서도 AUC 0.681이고, fixed Top3의 같은
날짜·vol-band 대비 safe-FP lift는 1.884배다. 따라서 fixed head가 **오직
변동성 순위를 복사한 것**은 아니다. 다만 score와 ATR의 상관이 +0.629이고
Top3 `dn5`가 59.55%까지 올라가므로, 실제 선택 꼬리에서는 강한 고변동
편향이 안전성 목표를 압도한다.

ATR head는 discovery에서 자기 ATR-label AUC 0.726,
vol-band 내부 AUC 0.725로 단순 vol 복사와는 다르다. 그러나 fixed
safe-FP에 대한 AUC는 0.571이고, Top3 `dn5`도 R1보다 +11.52%p 높아
실전 목적에는 실패했다.

결론은 “first-passage 학습이 무의미하다”가 아니다. **상방 사건
판별력은 생겼지만, 단일 확률로 낮은 하방과 높은 상방을 동시에 만족시키지
못했다**는 뜻이다.

## 5. ATR 장벽 sanity check

ATR 장벽은 holdout을 보지 않고 행별 D-1 ATR14에 고정했다.

`up = 2 × f_atr_pct_14`, `down = 1 × f_atr_pct_14`

어떤 cap, train quantile anchor, 최적화 배수도 사용하지 않았다.

| 구간 | up min / median / max | down min / median / max | up>50% = down>25% |
|---|---|---|---:|
| discovery | 2.43% / 14.14% / 80.52% | 1.21% / 7.07% / 40.26% | 105/26,136 = 0.40% |
| holdout | 2.64% / 14.95% / 82.81% | 1.32% / 7.47% / 41.41% | 128/17,838 = 0.72% |

비양수·비유한 장벽은 0건이지만 일부 극단 코인에서는 24시간 안에
`+80%/-40%`급 장벽을 요구한다. 이번 연구에서는 사후 cap을 만들지 않았다.
이는 trial을 늘리거나 holdout에 맞추는 것을 막기 위한 의도적 결정이다.

## 6. Fold 안정성

fixed head의 discovery 5개 fold safe FP는 R1 대비 4개 fold에서 높았지만,
`dn5`는 **5개 fold 전부** 높았다.

| fold | fixed safe FP | R1 safe FP | fixed `dn5` | R1 `dn5` | fixed net | R1 net |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 14.94% | 12.64% | 47.13% | 28.74% | -0.528% | -0.521% |
| 1 | 25.00% | 15.28% | 50.00% | 18.06% | -0.781% | -0.085% |
| 2 | 19.05% | 9.52% | 57.14% | 29.76% | -0.257% | -0.268% |
| 3 | 14.29% | 15.48% | 60.71% | 35.71% | -0.670% | -0.327% |
| 4 | 27.16% | 14.81% | 51.85% | 17.28% | +0.053% | +0.667% |

상방 개선은 어느 정도 반복되지만 하방 악화가 더 일관적이다. “한 fold의
우연한 손실”로 설명할 수 없다.

## 7. 위생·재현성 검증

### 7.1 계약 검증

- 최종 labeled panel `history_prior_bars` 최솟값: 70
- panel history 위반: 0/44,900
- discovery 전 정책 pick history 위반: 0/1,278
- holdout 전 정책 pick history 위반: 0/3,240
- 최종 공통 경로 output pick history 위반: 0/4,176
- prediction은 scope/date마다 정확히 Top100, 중복 0
- pick은 scope/date/policy마다 정확히 Top3, 중복 0
- 최종 pick의 불완전 경로: 0
- fixed same-bar 상하 동시 도달 6건은 모두 downside-first/label 0
- SL net은 정확히 -3.15%, TP net은 정확히 +4.85%로 비용 0.15%가
  한 번만 차감됨을 확인

### 7.2 재현성에서 발견하고 수리한 것

초기 build-memory 실행과 cache-read 실행에서 반올림 결과는 같아도 일부
CSV SHA가 달랐다. 확정된 원인은 cache float 왕복과 AUC 행 순서였으며,
멀티스레드와 gzip 시간 헤더도 잠재적 재현성 위험으로 함께 제거했다.

1. path cache는 `%.17g`와 `float_precision="round_trip"`으로
   build-memory/cache-read 입력을 일치시키고, gzip `mtime=0`으로 고정
2. AUC 출력의 Python `set` 순회를 명시적 tuple 순서로 고정
3. 재사용 helper의 XGBoost `n_jobs=4`도 연구 파일 안의 동일 사양
   `n_jobs=1` 학습기로 고정

그 뒤 새 cache를 만든 실행과 같은 cache를 읽은 다음 실행에서 아래 9개
산출물의 **압축 파일 자체를 포함한 byte SHA256이 전부 일치**했다.

| 산출물 | SHA256 |
|---|---|
| summary | `6d96b12a82c1f46081c12921145755b72b5d47ac2986914b8598f82d9ec8fd20` |
| auc | `0a59ecc5227e0c04297e553d7388dc7a118c40554fdee3f8b31f6a72c3f5f277` |
| paired | `8e9db3f28d63c2d99be8d1ed97a620d3e9f258585f8e27a5d1818775b2da7993` |
| folds | `c4ce86d27d0fdb713b5d9dace2613d3ff5a66a1480631db74e524cac2262d32e` |
| coverage | `9d5c6672602cbf815a3b3011613164e0d3a8e6c00225892a7aff26000f8a01c1` |
| predictions.gz | `e8f9343bc18b60ee101d86cf8e95422426226b4ac5c4971a4fb39328563cda59` |
| picks.gz | `6a218ca2d2bc60645bb6c00931ded6998daac2353e91117cd1d5b58fc1ebb54a` |
| path panel cache.gz | `df65fc4b6bfebd8728050ca3c4f1afbe7d6c160780796b84d76f3d4493f15599` |
| path cache meta | `ebe4f29e107b4bcfa65ac98ca9971a780e01f424adbb748daea9de09a7a5b4d9` |

검증 명령:

```bash
python -m py_compile scripts/first_passage_head_challenger_v1.py
python -m ruff check scripts/first_passage_head_challenger_v1.py
git diff --check -- scripts/first_passage_head_challenger_v1.py
python scripts/first_passage_head_challenger_v1.py
python scripts/first_passage_head_challenger_v1.py
sha256sum output/first_passage_head_challenger_v1_*
```

## 8. 발전 방향

### 지금의 결정

- fixed FP 단독 랭커: **폐기**
- ATR FP 단독 랭커: **폐기**
- 추천/알림/원장 반영: **하지 않음**
- 성능이 좋아 보이는 holdout net point estimate만 골라 채택: **하지 않음**

### 남길 가치가 있는 부분

fixed head의 holdout raw AUC 0.705와 safe-FP +5.89%p는 상방 헤드가 완전히
죽은 것은 아니라는 증거다. 버릴 것은 “상방 확률 하나로 안전 추천까지
해결한다”는 사용법이지, 학습된 상방 정보 전체가 아니다.

다음 단일 우선순위는 별도 sweep이 아니라 이미 판별력이 확인된 하방
확률을 먼저 veto하고, 통과 집합 안에서 fixed FP를 정렬하는
**veto-then-rank SHADOW 연구**다. 단, 이번 결과를 보고 임계값을 holdout에
맞추면 또 누수가 되므로:

1. 임계값은 discovery/forward만으로 정한다.
2. 같은 날짜의 후보를 paired 비교한다.
3. `dn5` non-worse를 가장 먼저 요구한다.
4. safe FP 개선과 net CI 하단 non-adverse를 동시에 요구한다.
5. 전 유니버스 일일 score snapshot을 쌓아 새 forward에서 확인한다.

이 후속도 사용자 승인과 현재 연구 거버넌스에 맞춰 별도 SHADOW로 해야
하며, 이번 REJECT 모델을 즉시 추천에 섞는 근거는 아니다.

## 9. 파일

- 실행 코드: `scripts/first_passage_head_challenger_v1.py`
- 상세 coverage/gate: `output/first_passage_head_challenger_v1_coverage.json`
- 정책 요약: `output/first_passage_head_challenger_v1_summary.csv`
- AUC/vol 진단: `output/first_passage_head_challenger_v1_auc.csv`
- 날짜-paired CI: `output/first_passage_head_challenger_v1_paired.csv`
- fold 안정성: `output/first_passage_head_challenger_v1_folds.csv`
- 전 유니버스 예측: `output/first_passage_head_challenger_v1_predictions.csv.gz`
- 공통 완전 경로 Top3: `output/first_passage_head_challenger_v1_picks.csv.gz`
- 경로 캐시: `output/first_passage_head_challenger_v1_path_panel.csv.gz`
