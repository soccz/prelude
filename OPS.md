# OPS.md — 매일 자동 운영 인프라

> 매일 KST 08:30 정확히 알림 발사. freshness 보장. 텔레그램 포맷. drift 감지. 주 1 회 재학습. 운영 인프라 모두.

---

## 0. 한 줄 결론

```
KST 08:30 cron 발사
   ↓
preflight (데이터 fresh? NaN? 유니버스 안정?)
   ↓
SIGNAL → predict → predictions_YYYYMMDD.csv
   ↓
LEDGER → 어제 가상 포지션 청산 + 오늘 진입 ledger 기록
   ↓
notifier → 텔레그램 메시지 (오늘 추천 + 어제 결과)
   ↓
post-hoc (drift_detector / ic_gate / verify_telegram)
```

매일 약 5 분 안에 끝남.

---

## 1. cron / systemd 스케줄 (deploy/)

### 1.1 메인 스케줄 (KST 시간 기준)

| 시각 (KST) | UTC | task | 스크립트 |
|---|---|---|---|
| **08:30 매일** | 23:30 | 일일 추론 + 알림 | `scripts/daily_run.sh` |
| **09:30 매일** | 00:30 | 어제 가상 포지션 청산 + ledger 갱신 + verify | `scripts/post_open_run.sh` |
| **00:00 매일** | 15:00 | 어제 holdout IC + drift 측정 | `scripts/measure_run.sh` |
| **일 06:00 주간** | 토 21:00 | 주간 재학습 + promotion gate | `scripts/retrain_run.sh` |
| **매시 정각** | — | 데이터 수집 (1h 봉 + 일봉 갱신) | `scripts/collect_run.sh` |

### 1.2 cron 표현 (deploy/crontab.txt)
```
# CRON_TZ 가능하면 사용 (시스템이 지원하면)
CRON_TZ=Asia/Seoul

# 데이터 수집 (매시 정각)
0 * * * * cd /home/soccz/22tb/today_pump && bash scripts/collect_run.sh >> output/cron_collect.log 2>&1

# 일일 추론 + 알림 (KST 08:30)
30 8 * * * cd /home/soccz/22tb/today_pump && bash scripts/daily_run.sh >> output/cron_daily.log 2>&1

# 어제 ledger 청산 + 검증 (KST 09:30, 어제 일봉 100% 마감 후)
30 9 * * * cd /home/soccz/22tb/today_pump && bash scripts/post_open_run.sh >> output/cron_post_open.log 2>&1

# IC + drift 측정 (KST 00:00)
0 0 * * * cd /home/soccz/22tb/today_pump && bash scripts/measure_run.sh >> output/cron_measure.log 2>&1

# 주간 재학습 (일요일 KST 06:00)
0 6 * * 0 cd /home/soccz/22tb/today_pump && bash scripts/retrain_run.sh >> output/cron_retrain.log 2>&1
```

### 1.3 systemd 대안 (deploy/systemd/)
cron 대신 systemd timer 사용 시 `td-pump-daily.timer` / `td-pump-measure.timer` / `td-pump-retrain.timer`. cron 문제 시 fallback.

---

## 2. preflight (ops/preflight.py)

추론 전 게이트. 통과 못 하면 안전 모드 (가상 진입 X, 텔레그램에는 "preflight 실패" 알림).

### 2.1 체크 항목
**모든 임계값은 초기값** (CLAUDE.md §2.5). false positive (preflight 너무 자주 fail) / false negative (실제 데이터 문제 못 잡음) 비율 보고 조정.

- **freshness**: 어제 일봉 timestamp 가 어제 KST 09:00 마감 ± 30 분 안인가? (초기 30 분, 업비트 API 안정성 보고 조정)
- **NaN 비율**: 피처 NaN 비율 ≤ 20% per-coin? (초기 20%, 실제 NaN 분포 보고 조정)
- **universe churn**: 오늘 universe vs 어제 universe 30% 이상 안 바뀌었는가? (초기 30%, top N 안정성 보고 조정)
- **모델 가중치 존재**: `signals/models/ckpt/<latest>.json` 있는가?
- **calibration 존재**: `output/calibration_sigma.json` 30 일 이내 갱신됐는가? (초기 30 일, retrain cadence 와 연동)

### 2.2 실패 시
- 텔레그램: "⚠️ preflight 실패 — <사유>"
- 가상 ledger: 그 날 진입 X (skip)
- 다음 날 자동 재시도

---

## 3. 텔레그램 알림 포맷 (notifier/telegram.py)

### 3.1 매일 KST 08:30 메인 알림

```
🌅 today_pump 2026-05-03 (KST 08:30)
BTC regime: bull_quiet | universe: 100 코인

━━━ 오늘 추천 (top 3) ━━━
🔥 KAITO  78%↑ (+9.2%) [+2.5,+14.8] ⭐⭐
✅ ETH    65%↑ (+6.1%) [+1.2, +9.8] ⭐
✅ SOL    62%↑ (+5.8%) [+0.8,+10.5] ⭐⭐

━━━ 어제 결과 (5/3 진입 → 5/4 청산) ━━━
✓ KAITO  +12.3% (예측 +9.2%) 🎯
✓ ETH     +5.4% (예측 +6.1%) 🎯
✗ SOL     -1.2% (예측 +5.8%) ❌

가상 ledger: 누적 +47.3 만원 (+4.7%) | MDD -3.2%
hit rate (이번 주): 67%

전체 성과: scripts/ledger_summary.py
```

### 3.2 출력 필드 정의
| 필드 | 의미 |
|---|---|
| 🔥/✅/▫/· | σ-tier (SIGNAL §5.2) |
| %↑ | 오늘 펌프 확률 (calibration 기반) |
| (+X.X%) | 기대 수익 (calibration mean_signed_return) |
| [a, b] | 95% CI (코인별 변동성 × 1.96) |
| ⭐⭐ | 과거 안정도 (이 코인 ledger hit rate ≥ 60% 면 ⭐⭐, 70% 면 ⭐⭐⭐) |
| 🎯/❌ | 어제 적중 / 실패 |

### 3.3 톤 원칙
- "**오늘**" 표현 (사용자 멘탈 모델 정확히)
- 미래형 "내일 오를" 절대 X
- 거짓 자신감 금지: 확률은 calibration 기반 정직 표시 (78% 가 78% 의미)
- 실패는 명시: ❌ 표시 빠지면 안 됨

### 3.4 침묵 조건
다음 중 하나면 알림 발사 X (또는 "오늘 침묵" 만):
- BTC regime = bear_volatile
- preflight 실패
- 일일 손실 한도 발동 (LEDGER §7.1)
- σ-tier ≥ ✅ 코인 0 개

침묵해도 매일 짧은 "today_pump alive" ping 은 보냄 (cron 죽었는지 사용자가 알 수 있게).

---

## 4. drift_detector (ops/drift_detector.py)

### 4.1 측정
매일 KST 00:00 (어제 ledger 청산 후) 자동 실행:

- **per-feature IC** 24h vs 7d MA 비교
- **모델 prediction 분포** 24h vs 7d (KS test)
- **hit rate** 7d vs 30d (chi-squared)

### 4.2 alarm 트리거
**모든 cutoff 는 초기값** (CLAUDE.md §2.5). drift_state.json 누적 후 false positive 비율 보고 조정.

- **SIGN_FLIP**: 7d MA IC > 0 였는데 24h IC < 0 (또는 반대) — cutoff 명확
- **HALF_DROP**: 24h IC < 7d MA IC × **drop_ratio** (초기 0.5, 너무 자주 발동하면 0.4 / 너무 안 발동하면 0.6)
- **HIT_RATE_DROP**: 7d hit rate < 30d hit rate - **hit_drop_threshold** (초기 0.15)
- 윈도우 (7d / 30d) 도 초기값 — drift 빈도 보고 조정

### 4.3 alarm 발동 시
- 텔레그램 경고: "⚠️ drift detected — <사유>"
- `output/drift_state.json` 갱신: `state: "WARN"` 또는 `"FREEZE"`
- WARN: 알림 계속하되 가상 사이즈 50% cut
- FREEZE: 가상 진입 X + 다음 retrain 강제 트리거
- 사용자 컨펌 후 정상 복귀

---

## 5. ic_gate (ops/ic_gate.py)

### 5.1 역할
**진단용 (사후 보고)**. 게이트로 신호 막지 않음 (CLAUDE.md §2.3).

### 5.2 측정
주 1 회 (재학습 직전) 실행:
- 지난 7d IC, 30d IC, ICIR
- `output/ic_history.json` 누적
- 텔레그램 주간 리포트에 옆에 표기

### 5.3 promotion 결정에 사용 X
재학습 promotion gate 는 net Sharpe / hit rate 기준 (LEDGER §8.2). IC 는 옆에 참고만.

---

## 6. retrain_pipeline (ops/retrain_pipeline.py)

### 6.1 흐름
**Cadence (주 1 회 / 일요일 KST 06:00) 는 초기값** (CLAUDE.md §2.5). 데이터 보고 조정:
- drift 빈도 자주 → cadence 짧게 (예: 일 1 회 light retrain)
- 결과 안정 → cadence 길게 (예: 월 1 회)
- promotion gate 통과율 추적 → 항상 fail 이면 모델 재설계 신호

일요일 KST 06:00 자동:

```
1. preflight (데이터 / 디스크)
2. 후보 모델 학습 (signals/models/xgb_phase1.py + 최신 데이터)
3. holdout 백테스트 (지난 7d, ledger 기반)
4. promotion gate (SIGNAL §8.2 — net Sharpe / hit rate 기준, IC 는 사후만)
   - new net Sharpe ≥ old - 0.1
   - new hit rate ≥ old - 0.02
5. 통과: ckpt 갱신 + calibration 재생성 + 텔레그램 보고
   실패: 후보 폐기 + 이전 모델 유지 + 로그
6. output/retrain_history.json 누적
```

### 6.2 사용자 컨펌
- 자동 retrain 은 OK
- but **3 회 연속 fail** 시 텔레그램 경고 + 사용자 명시적 hard reset 명령 전까지 새 모델 시도 X

---

## 7. 데이터 수집 (data/collector_*.py + ops/collect.py)

### 7.1 cadence
- 일봉 (메인): 매일 KST 09:00 직후 (UTC 00:00 직후) — 어제 일봉 마감 즉시 갱신
- 4h 봉: 매 4 시간 (KST 09 / 13 / 17 / 21 / 01 / 05)
- 바이낸스 1h: 매 시간 정각

### 7.2 freshness 보장
- 매시 cron `collect_run.sh` 가 last_timestamp 확인 → 부족하면 backfill
- preflight 가 "어제 마감 ± 30 분" 안인지 확인

### 7.3 retry / 안정성
- 업비트 / 바이낸스 API 일시적 fail → exponential backoff retry
- 24h 이상 fail 시 텔레그램 경고

---

## 8. 일관성 검증 (scripts/verify_telegram.py)

LEDGER §8 참조. 매일 KST 09:30 cron 자동:
- 어제 텔레그램 메시지 vs 오늘 ledger.csv 비교
- 부호 / 크기 / 코인 일치
- 불일치 → 텔레그램 alert + 운영 일시 정지

---

## 9. 로그 / 헬스 체크 (output/cron_*.log)

### 9.1 로그 파일
- `output/cron_daily.log` — 매일 추론 로그
- `output/cron_post_open.log` — ledger 청산
- `output/cron_measure.log` — IC / drift
- `output/cron_retrain.log` — 주간 재학습
- `output/cron_collect.log` — 데이터 수집

매일 자동 회전 (1 주 보관, 그 후 압축).

### 9.2 헬스 체크
- `scripts/health_check.py` — 매일 KST 09:30 실행
- 어제 cron 다 돌았는지 + log 에 ERROR 없는지 + DB freshness 확인
- 문제 시 텔레그램 alert

---

## 10. 책임 경계

| 이건 OPS 의 일 | 이건 OPS 의 일 X |
|---|---|
| cron 스케줄 | 라벨 / 모델 → SIGNAL |
| preflight (데이터 위생) | 가상 사이징 → LEDGER |
| 텔레그램 메시지 포맷 / 전송 | 가상 진입 청산 → LEDGER (호출만) |
| drift 감지 | 사용자 매매 일지 → NOTES |
| 주간 재학습 트리거 (모델 학습 자체 X) | 모델 architecture → SIGNAL |
| 일관성 검증 자동화 | 백테스트 결과 분석 → SIGNAL |

---

## 11. 핵심 파일 인덱스

| 파일 | 역할 |
|---|---|
| `deploy/crontab.txt` | cron 정의 |
| `deploy/systemd/*.{service,timer}` | systemd fallback |
| `scripts/daily_run.sh` | KST 08:30 메인 |
| `scripts/post_open_run.sh` | KST 09:30 청산 / 검증 |
| `scripts/measure_run.sh` | KST 00:00 IC / drift |
| `scripts/retrain_run.sh` | 일 06:00 재학습 |
| `scripts/collect_run.sh` | 매시 데이터 수집 |
| `scripts/health_check.py` | 일일 헬스 체크 |
| `ops/preflight.py` | 추론 전 게이트 |
| `ops/drift_detector.py` | drift 감지 |
| `ops/ic_gate.py` | IC 사후 보고 |
| `ops/retrain_pipeline.py` | 주간 재학습 자동 |
| `notifier/telegram.py` | 텔레그램 봇 |
| `notifier/format.py` | 메시지 포맷터 |
| `output/cron_*.log` | 운영 로그 |
| `output/drift_state.json` | drift 상태 |
| `output/retrain_history.json` | 재학습 이력 |
