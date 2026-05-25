# OPS.md — 매일 자동 운영 인프라

> **현재 운영: pre-open + distribution policy-gated beta (2026-05-25).**
> 텔레그램은 매일 보낸다. ACTIVE 가 있으면 추천, 없으면 짧은 침묵/상태 메시지.
> WATCH_ONLY/SILENCE 후보와 수수료 차감 검증 결과는 shadow/paper ledger,
> log JSON, dashboard 에 남긴다.

---

## 0. 한 줄 결론 (현재 운영)

```
KST 08:50 pre-open timer
   ↓
scripts/daily_run_preopen.sh
   ├─ d1 + 15m update
   ├─ health_check --channel preopen
   └─ predict_preopen_trigger.py
       ├─ decision_policy → ACTIVE / WATCH_ONLY / SILENCE
       ├─ recommendation_quality meta-filter → weak historical groups downranked
       ├─ ACTIVE만 paper_ledger_preopen
       ├─ Telegram daily: ACTIVE 추천 또는 침묵/상태
       └─ 전체 후보 shadow_ledger_preopen + preopen_log JSON

KST 09:05 distribution timer
   ↓
scripts/daily_run_distribution.sh
   ├─ d1 + 4h update
   ├─ health_check --channel distribution
   └─ predict_today_distribution.py
       ├─ decision_policy → ACTIVE / WATCH_ONLY / SILENCE
       ├─ recommendation_quality meta-filter → weak historical groups downranked
       ├─ ACTIVE만 paper_ledger
       ├─ Telegram daily: ACTIVE 추천 또는 침묵/상태
       └─ 전체 후보 shadow_ledger_distribution + distribution_log JSON

KST 09:30 / 10:05 close timers
   ↓
distribution/preopen paper + shadow ledger close
   ↓
idea_validation_summary.{csv,json} + idea_validation_report.html
   ↓
KST 10:10 dashboard publish
```

**Stage 진행**:
- Stage 1: shadow/paper live validation — 텔레그램은 ACTIVE만
- Stage 2: policy gate 가 충분한 live shadow 표본을 확보하면 promotion/demotion 조정
- Stage 3: NOTES 기반 사용자 실제 매매와 system 추천 비교

**원칙 (양보 X)**:
- 자동 실거래 주문 없음. 추천과 기록만 수행
- 텔레그램은 의미 있는 ACTIVE 추천만. 침묵/관찰 후보는 dashboard·ledger 기록
- look-ahead 방어: 입력은 as-of 이전 데이터만
- 거래비용 0.15% round-trip 차감 후 평가
- policy 숫자는 placeholder. live net PnL/Max DD/hit rate 로 조정

---

## 1. cron / systemd 스케줄 (deploy/)

### 1.1 메인 스케줄 (KST 시간 기준)

| 시각 (KST) | UTC | task | 스크립트 |
|---|---|---|---|
| **08:50 매일** | 23:50 전일 | pre-open 추론 + 매일 Telegram | `scripts/daily_run_preopen.sh` |
| **09:05 매일** | 00:05 | distribution 추론 + 매일 Telegram | `scripts/daily_run_distribution.sh` |
| **09:30 매일** | 00:30 | distribution paper/shadow ledger 청산 | `scripts/daily_close_distribution.sh` |
| **10:05 매일** | 01:05 | pre-open paper/shadow ledger 청산 | `scripts/daily_close_preopen.sh` |
| **10:10 매일** | 01:10 | dashboard JSON 빌드 + publish | `scripts/publish_dashboard.sh` |
| **일 06:00 주간** | 토 21:00 | 주간 재학습 + promotion gate | `scripts/retrain_run.sh` |

### 1.2 cron 표현 (deploy/crontab.txt)
```
# CRON_TZ 가능하면 사용 (시스템이 지원하면)
CRON_TZ=Asia/Seoul

# pre-open 추론 + 매일 Telegram (KST 08:50)
50 8 * * * cd /home/soccz/22tb/prelude && bash scripts/daily_run_preopen.sh >> output/cron_preopen.log 2>&1

# distribution 추론 + 매일 Telegram (KST 09:05)
5 9 * * * cd /home/soccz/22tb/prelude && bash scripts/daily_run_distribution.sh >> output/cron_dist.log 2>&1

# distribution ledger close (KST 09:30)
30 9 * * * cd /home/soccz/22tb/prelude && bash scripts/daily_close_distribution.sh >> output/cron_close.log 2>&1

# pre-open ledger close (KST 10:05)
5 10 * * * cd /home/soccz/22tb/prelude && bash scripts/daily_close_preopen.sh >> output/cron_preopen_close.log 2>&1

# dashboard publish (KST 10:10)
10 10 * * * cd /home/soccz/22tb/prelude && bash scripts/publish_dashboard.sh >> output/cron_publish.log 2>&1

# 주간 재학습 (일요일 KST 06:00)
0 6 * * 0 cd /home/soccz/22tb/prelude && bash scripts/retrain_run.sh >> output/cron_retrain.log 2>&1
```

### 1.3 systemd 대안 (deploy/)
cron 대신 systemd timer 사용 시 `deploy/prelude-*.service` / `deploy/prelude-*.timer`.
설치는 `sudo bash deploy/install_systemd.sh`. signal timer 는 timing-critical 이라
catch-up 실행을 막기 위해 `Persistent=false`.

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

## 3. 텔레그램 알림 포맷 (notifier/format.py)

### 3.1 현재 운영 포맷

텔레그램은 ACTIVE 추천만 발송한다. 모델 raw score 는 텔레그램에서 제거하고,
사용자가 바로 판단할 수 있는 policy/edge 중심으로 표시한다.

**distribution 예시**:
```
⚡ distribution 2026-05-25 (KST 09:05)
BTC: 🟢 강세 안정 | universe: top100 (100)

━━━ 상승 setup 후보 1건 ━━━

🔥 AAA  진입가 ≈ 1,234원  [B_S03 | rank #2]
   ▸ edge +2.31%p | 검증 hit 62.4% | setup S02+S03 / S04
   ▸ policy: S03-quality setup with positive h6 edge proxy

━━━ 사용 ━━━
• 09:00 직후 또는 첫 4h 안 진입
• 5% 오르면 즉시 매도, 자동 실거래 주문 없음
• 텔레그램은 ACTIVE만 발송, WATCH/SILENCE는 dashboard·ledger에 기록
```

**pre-open 예시**:
```
⚡ pre-open trigger 2026-05-25 (KST 08:55)
BTC: 🟢 강세 안정 | universe: top100

━━━ 09:00 직후 펌프 후보 1건 ━━━

🔥 BBB  진입가 ≈ 987.6원  [PREOPEN | rank #1]
   ▸ edge +1.20%p | 1h +5 signal 46.0%
   ▸ policy: bull regime with strong first1h_5 and composite
```

### 3.2 출력 필드 정의
| 필드 | 의미 |
|---|---|
| ACTIVE | 텔레그램 + paper ledger 에 들어가는 실제 추천 후보 |
| WATCH_ONLY | 텔레그램 미발송. shadow ledger 에 기록해서 정책 실험 |
| SILENCE | 텔레그램 미발송. risk-off 또는 setup 부재 |
| edge | 거래비용 차감 전후를 반영한 단순 EV proxy. 최종 성과 판단은 live ledger 기준 |
| 검증 hit / signal | bucket calibration 이 있으면 검증 hit, 없으면 raw fallback signal 로 표기 |
| policy | 왜 ACTIVE 로 승격됐는지 또는 왜 관찰 후보인지 설명 |

### 3.3 톤 원칙
- 텔레그램은 매일 상태를 보낸다. ACTIVE가 있으면 추천, 없으면 침묵/상태
- 자동 실거래 주문 없음. 사용자가 직접 판단
- raw 확률처럼 오해될 수 있는 모델 점수는 알림에서 제거
- 운영 스크립트는 `--send-silence-telegram` 을 켜서 cron 생존 여부도 확인

### 3.4 legacy 포맷
`format_detector_beta`, `format_daily_alert` 는 보존하지만 현재 메인 운영 X.

### 3.5 텔레그램 봇 — 기존 MAE 봇 공유

**별도 봇 발급 X**. gan_t / xsec_alpha 가 이미 쓰는 **MAE 봇** 그대로 사용 (사용자 본인 봇).

설정 (`prelude/.env`, `.gitignore` 처리됨):
```
TELEGRAM_BOT_TOKEN=...   # gan_t/.env 의 값
TELEGRAM_CHAT_ID=...     # 동일
```

`notifier/telegram.py` 가 자동 `load_dotenv()` — 추가 export 불필요.

메시지 구분: prelude 메시지는 prefix `🌅 prelude` — gan_t / xsec_alpha 메시지와 구분 가능.

**연결 테스트**:
```bash
python -c "from notifier.telegram import send_telegram; send_telegram('test')"
```

CLI 기본값은 ACTIVE 가 없으면 발송하지 않는다. 운영 스크립트는 매일 확인을
위해 `--send-silence-telegram` 을 명시적으로 붙인다.

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
- `output/cron_preopen*.log` — pre-open 추론 / close 로그
- `output/cron_dist*.log` — distribution 추론 로그
- `output/cron_close*.log` — distribution ledger 청산
- `output/cron_preopen_close*.log` — pre-open ledger 청산
- `output/cron_publish.log` — dashboard publish
- `output/cron_retrain.log` — 주간 재학습

매일 자동 회전 (1 주 보관, 그 후 압축).

### 9.2 헬스 체크
- `scripts/health_check.py --channel preopen` — d1 + 15m freshness 확인
- `scripts/health_check.py --channel distribution` — d1 + 4h freshness 확인
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
| `deploy/prelude-*.{service,timer}` | systemd timers |
| `scripts/daily_run_preopen.sh` | KST 08:50 pre-open |
| `scripts/daily_run_distribution.sh` | KST 09:05 distribution |
| `scripts/daily_close_distribution.sh` | KST 09:30 distribution close |
| `scripts/daily_close_preopen.sh` | KST 10:05 pre-open close |
| `scripts/publish_dashboard.sh` | KST 10:10 dashboard publish |
| `scripts/train_recommendation_meta.py` | closed ledger 기반 meta-label 학습 |
| `scripts/retrain_run.sh` | 일 06:00 재학습 |
| `scripts/health_check.py` | 일일 헬스 체크 |
| `ops/preflight.py` | 추론 전 게이트 |
| `ops/decision_policy.py` | ACTIVE / WATCH_ONLY / SILENCE 정책 |
| `ops/recommendation_quality.py` | historical evidence 기반 추천 confidence / demotion |
| `ops/policy_gate.py` | live/replay policy promotion 판단 |
| `ops/drift_detector.py` | drift 감지 |
| `ops/ic_gate.py` | IC 사후 보고 |
| `ops/retrain_pipeline.py` | 주간 재학습 자동 |
| `notifier/telegram.py` | 텔레그램 봇 |
| `notifier/format.py` | 메시지 포맷터 |
| `output/cron_*.log` | 운영 로그 |
| `output/drift_state.json` | drift 상태 |
| `output/retrain_history.json` | 재학습 이력 |
