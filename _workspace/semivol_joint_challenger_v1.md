# Downside semivolatility joint head — 고정 1회 진단

## 결론

**REJECT. 운영·알림·SHADOW에 반영하지 않는다.**

하방/상방 semivolatility 4개를 직접 first-passage head에 추가하면 기존
24-feature head보다 `dn5`는 소폭 낮아졌지만 신뢰구간이 0을 포함했고,
safe-FP·up10은 전혀 늘지 않았으며 net은 오히려 소폭 낮아졌다. R1과
비교하면 safe-FP는 높아지지만 `dn5`와 SL-first도 함께 크게 증가한다.
즉 이번 4개 피처는 **“상승은 유지하면서 하방만 분리”하는 축이 되지
못했다.**

최종 180일은 관련 연구가 이미 본 오염된 진단 구간이므로, 통과했어도
최대 판정은 `FORWARD_SHADOW_CANDIDATE`였다.

## 잠근 비교

- 타깃: 실제 발송 후 `[D 09:15, D+1 09:15)` 96개 15분봉에서
  `+10%`가 `-5%`보다 먼저 도달. 같은 봉 동시 도달은 하방 우선.
- 유니버스: 매일 D-1 거래대금 PIT Top100, `history_prior_bars >= 70`.
- core: 기존 first-passage XGB 24 features.
- augmented: core + 아래 4개만 추가.
  - `downside_semivol_7 = sqrt(mean(min(ret,0)^2), 7)`
  - `downside_semivol_21 = sqrt(mean(min(ret,0)^2), 21)`
  - `upside_semivol_21 = sqrt(mean(max(ret,0)^2), 21)`
  - `semivol_asym_21 = upside_semivol_21 - downside_semivol_21`
- `ret = close_t / close_(t-1) - 1`; rolling 계산 뒤 정확히 1일 shift.
- 모델/창/공식 조정 없음. XGB seed 42, `n_jobs=1`, outer expanding
  5-fold, inner true OOF isotonic 3-fold, embargo 5일.
- 비용: TP +5% / SL -3% bracket 수익에서 왕복 0.15%를 1회 차감.
- 비교 정책: augmented, core, R1 repaired, lowest-ATR,
  liquidity-matched. 모든 정책 Top3 경로가 완결된 동일 날짜만 사용.

## PIT·재현성 감사

- path panel: 44,900행, 449일, 257 markets.
- semivol source date 위반: **0건**; 모든 44,900행의 source lag가
  정확히 1일.
- 피처 결측/비유한값: **0건**.
- 평가 prediction: 32,200행, 322일, 날짜별 정확히 100개.
- 기존 24-feature first-passage 산출과 재현 대조:
  32,200행 raw/probability 최대 절대차 **0**.
- holdout: 마지막 KRW-BTC 완결 180일
  (`2026-01-24`~`2026-07-24`), 5개 정책 공통 경로완결 **169일**.
- 두 번의 실행 산출물 SHA-256이 전부 동일했다
  (`gzip mtime=0`).

## Locked 169일 결과

| 정책 | safe-FP | up10 | dn5 | SL-first | TP5/SL3 net/픽 | within-vol lift |
|---|---:|---:|---:|---:|---:|---:|
| semivol joint | 21.30% | 26.43% | 57.00% | 58.19% | -0.0126% | 1.923 |
| core FP | 21.30% | 26.43% | 59.17% | 58.58% | +0.0181% | 1.935 |
| R1 repaired | 14.99% | 18.15% | 39.45% | 46.55% | -0.1318% | 1.763 |
| lowest-ATR | 0.39% | 0.39% | 4.54% | 15.78% | -0.3516% | 0.207 |
| liquidity-matched | 9.86% | 11.44% | 34.71% | 50.30% | -0.4176% | 1.159 |

Augmented minus core:

- safe-FP `0.00%p`, up10 `0.00%p`.
- dn5 `-2.17%p`, paired date CI95 `[-5.33, +1.18]%p`.
- net `-0.0307%p/픽`, CI95 `[-0.2698, +0.2096]%p`.
- full-universe AUC `0.7052 → 0.6981`, within-vol AUC
  `0.6806 → 0.6709`.

## 사전 고정 gate 대 R1

| 조건 | 관측 | 판정 |
|---|---:|---|
| dn5 delta CI95 upper `<= 0` | delta `+17.55%p`, CI `[+12.03,+23.27]` | FAIL |
| safe-FP delta CI95 lower `> 0` | delta `+6.31%p`, CI `[+1.78,+11.05]` | PASS |
| net delta CI95 lower `>= 0` | delta `+0.119%p`, CI `[-0.265,+0.506]` | FAIL |

3개 중 2개 실패이므로 최종 **REJECT**다. lowest-ATR는 하방을 매우
낮출 수 있지만 상승 가능성도 거의 0으로 만든다. 반대로 direct
first-passage 계열은 상승 포착과 함께 고변동·하방을 고른다. 이번
semivol 4개는 그 둘을 분리하지 못했다.

## 산출물

- 코드: `scripts/semivol_joint_challenger_v1.py`
- 전체 계약·gate: `output/semivol_joint_challenger_v1_coverage.json`
- 재현 manifest:
  `output/semivol_joint_challenger_v1_manifest.json`
- prediction/picks:
  `output/semivol_joint_challenger_v1_{predictions,picks}.csv.gz`
- 결과:
  `output/semivol_joint_challenger_v1_{summary,paired,auc,folds}.csv`

주요 SHA-256:

- manifest:
  `72e9c900717a18da5f2bb283c63f22fdcd1cd4f509f2427304c30458491bd872`
- predictions:
  `7e03de31dd8b7390e68e1b71ef447641709de2bd04f41a14989cd60965f38009`
- picks:
  `84150b1da5169e936cd5c79a4624e84fd225bff9fa570bf319706e7ff492b7d0`

## 실행·검증 명령

```bash
venv/bin/python -m py_compile scripts/semivol_joint_challenger_v1.py
ruff check scripts/semivol_joint_challenger_v1.py
venv/bin/python scripts/semivol_joint_challenger_v1.py --help
venv/bin/python scripts/semivol_joint_challenger_v1.py
venv/bin/python scripts/semivol_joint_challenger_v1.py
sha256sum output/semivol_joint_challenger_v1*
git diff --check -- scripts/semivol_joint_challenger_v1.py
```
