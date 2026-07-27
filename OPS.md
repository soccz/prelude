# OPS.md — 매일 자동 운영 인프라

> **현재 운영: R1 preopen/open + pump v2 radar (2026-07-26).**
> 자동 주문은 없고 사용자가 알림을 보고 직접 판단한다. R2/A1·legacy
> distribution/preopen·pump v1은 record-only다. 현재 성과 해석은
> `radar-not-strategy`이며 forward 증거가 쌓이기 전 승격하지 않는다.

---

## 0. 한 줄 결론 (현재 운영)

```
KST 08:50 pre-open timer
   ↓
scripts/daily_run_preopen.sh
   ├─ D1 update
   ├─ recommend-preopen gate: D-2 PIT Top100 + D-1 exact D1
   ├─ immutable R1 snapshot → Telegram receipt → 전용 ledger
   └─ 그 뒤 15m update + legacy preopen record-only
      (15m 실패가 D1-only R1 발송을 막지 않음)

KST 09:05 distribution timer
   ↓
scripts/daily_run_distribution.sh
   ├─ D1 update + recommend gate: D-1 PIT Top100 + 당일 exact D1
   ├─ immutable R1 snapshot → Telegram receipt → 전용 ledger
   ├─ 4h update + exact closed-boundary gate → legacy distribution record-only
   ├─ R2 / A1 / pump v1 record-only
   └─ Binance D1 update → pump v2 decision/receipt/shadow ledger

KST 09:30 / 10:05 close timers
   ↓
canonical decision/snapshot + receipt + ledger identity gate
   ↓
distribution/preopen paper + shadow ledger close + R1 24h label/evaluator
   ↓
idea_validation_summary.{csv,json} + idea_validation_report.html
   ↓
KST 10:10 dashboard publish → KST 10:30 heartbeat
```

**원칙 (양보 X)**:
- 자동 실거래 주문 없음. 추천과 기록만 수행
- look-ahead 방어: preopen은 D-1까지, open은 해당 09:00에 관측 가능한 입력만 사용
- PIT 거래대금 Top100만 요구하고 KRW stablecoin 5종
  (`USD1/USDC/USDE/USDS/USDT`)은 공통 정책으로 제외
- 정확한 candle 경계·snapshot checksum·source identity가 하나라도 어긋나면 fail closed
- 거래비용 0.15% round-trip 차감 후 평가
- 모델·정렬·라벨·알림 문구·사이징 변경은 사용자 승인 후

---

## 1. systemd 스케줄 (deploy/)

### 1.1 메인 스케줄 (KST 시간 기준)

| 시각 (KST) | UTC | task | 스크립트 |
|---|---|---|---|
| **04:00 매일** | 19:00 전일 | versioned SQLite·verdict/anchor backup | `scripts/backup_db.sh` |
| **08:50 매일** | 23:50 전일 | R1 preopen 발송 + legacy record-only | `scripts/daily_run_preopen.sh` |
| **09:05 매일** | 00:05 | R1 open·pump v2 발송 + challenger 기록 | `scripts/daily_run_distribution.sh` |
| **09:30 매일** | 00:30 | distribution paper/shadow ledger 청산 | `scripts/daily_close_distribution.sh` |
| **10:05 매일** | 01:05 | pre-open 청산 + 전일 R1 24h label/evaluator | `scripts/daily_close_preopen.sh` |
| **10:10 매일** | 01:10 | dashboard JSON 빌드 + publish | `scripts/publish_dashboard.sh` |
| **10:30 매일** | 01:30 | evidence·publish·ledger·DB heartbeat | `scripts/heartbeat.sh` |

표의 시각은 nominal calendar다. `RandomizedDelaySec` 때문에 backup은 최대 120초,
preopen/preopen-close는 최대 30초, 나머지는 최대 60초 뒤 시작할 수 있다.

### 1.2 단일 scheduler 계약

운영 scheduler는 `deploy/prelude-*.service`와 7개 timer뿐이다.
`deploy/crontab.txt`는 과거 참고 자료이며 활성화하면 안 된다. 설치기는 전체 사용자
cron과 설치/등록된 prelude timer를 읽고, 중복 작업이 있으면 자동 수정하지 않고
fail closed한다.

```bash
# 먼저 .env에 PRELUDE_DASHBOARD_PIN을 추가
sudo bash deploy/install_systemd.sh --check-only
sudo bash deploy/install_systemd.sh
```

08:50·09:05 signal timer는 늦은 catch-up 발송을 막기 위해 `Persistent=false`,
나머지는 `Persistent=true`다. 7개 workload service는
`OnFailure=prelude-failure-alert@%n.service`로 실패를 알리고, stage wrapper가
후속 publish를 선행 stage 성공 증거와 연결한다.

**2026-07-26 반영 상태:** 저장소 unit 문법·KST calendar·테스트는 통과했지만
`/etc/systemd/system`의 15개 설치본은 모두 이전 버전이거나 누락이다. sudo password가
필요해 이 세션에서는 설치하지 못했다. 위 두 명령을 통과시키기 전에는 저장소 수정이
내일 timer에 적용됐다고 간주하면 안 된다.

### 1.4 R1 snapshot과 forward 평가

- preopen/open 슬롯은 같은 날 같은 슬롯의 모델을 두 번 학습하지 않고 단일 snapshot을
  Telegram·추천 원장·전 유니버스 score 기록이 함께 사용한다.
- 성공/실패와 `sent_at`은 별도 delivery receipt로 저장한다. active 원장은 성공 receipt가
  없거나 손상됐으면 기록을 거부한다.
- 다음 날 10:05 close runner가 전일 snapshot을 라벨한다. `sent_at` 다음 실행 가능한
  15분봉부터 새 96봉을 사용하므로 09:10 발송의 평가 창은
  `[D 09:15, D+1 09:15)`다.
- KRW-BTC 기준 경로가 불완전하면 partial로 보류하고, 대상 코인만 거래가 없던 봉은 직전
  close로 flat-fill한다. 평가기는 complete artifact만 기본 forward 통계에 사용한다.
- 과거 재생은 `scheduled_replay`, 실제 목표일 생성은 `forward_observed`로 분리한다.
  사용자가 실제로 받은 성과는 그중 `delivery_ok=True` cohort를 따로 본다.
- close 게이트 모드는 4종: `close`(정상 청산) / `skip-zero-pick`(검증된 무추천일) /
  `skip-legacy-unverifiable`(계약 이전) / `skip-no-decision`(발송 파이프 자체가 죽어
  snapshot·receipt·원장 행이 전부 없는 날 — 2026-07-28 신설). skip-no-decision은
  plan과 락 하 재검증이 같은 술어(`is_no_decision_day`)를 쓰고, 검증 시
  `output/close_no_decision/{cohort}/{asof}.json` 감사 마커를 남긴다(백업 포함).
  원장 행(상태 무관)이나 receipt가 하나라도 남아 있으면 조용한 skip이 아니라
  무결성 실패로 fail-closed 한다. 이 마커 일수는 커버리지 분모 보정에 쓴다
  (무추천일과 무결정일은 다르다 — MNAR 방지).

### 1.5 pump v2 evidence와 terminal 판정

- 새 decision→receipt→ledger 계약 활성일은 `2026-07-27`이다. 그 이전 205개
  CLOSED 행은 frozen scorecard 입력 5개 필드의 SHA-256
  `ac01ddde…94451`로 고정해 과거 수치 변조를 차단한다.
- 2026-07-27 이후 행은 canonical decision과 성공 receipt, 후보 identity가 모두
  일치해야 CLOSED 성과나 recall 근거로 사용한다. 2026-07-26 zero-pick receipt는
  검증하되 `legacy-unverifiable`로 분리한다.
- 조기 KILL 또는 2026-09-01 GO/KILL은 별도 immutable terminal state와 anchor에
  기록한다. 상태 누락·손상·미래시각·불일치 시 발송은 fail closed한다.
- 로컬 관리자 권한으로 state와 anchor를 함께 바꾸는 위협까지 방어하려면 추후
  HMAC 비밀키 또는 외부 WORM 저장소가 필요하다.

---

## 2. 현재 데이터 gate (`scripts/health_check.py`)

현재 daily runner는 tolerance 기반 legacy preflight가 아니라 PIT Top100의 정확한
candle boundary를 검사한다. 실패하면 해당 시그널 생성·발송·원장 기록을 건너뛰고
service가 nonzero를 반환한다.

### 2.1 체크 항목

- `recommend`: D-1 quote-volume PIT Top100 + 당일 09:00 D1 exact
- `recommend-preopen`: D-2 PIT Top100 + 전일 09:00 D1 exact
- `distribution`: recommend 조건 + 마지막 closed 4h exact
- `preopen`: recommend-preopen 조건 + 마지막 closed 15m exact
- 명시 stablecoin 5종과 insufficient-history/lower-ranked 종목은 감사 가능한 사유로 제외
- risk/drift state는 strict parse하되 evaluator 미배선 상태는
  `BOOTSTRAP_UNINITIALIZED`로 명시

### 2.2 실패 시
- daily runner가 해당 채널 작업을 skip하고 nonzero를 전파
- systemd `OnFailure`가 운영 실패 알림을 담당
- `ops/preflight.py`는 `scripts/predict_today_legacy.py`에서만 쓰는 legacy helper

---

## 3. 텔레그램 알림 포맷 (notifier/format.py)

### 3.1 현재 운영 포맷

현재 실발송 진입점은 R1의 `scripts/recommend_send.py`와 pump v2의
`scripts/pump_detector_v2_today.py`뿐이다. 아래 distribution/pre-open 예시는
record-only legacy formatter의 보존 문서이며 현재 R1 메시지 계약이 아니다.
알림 문구 자체는 사용자 승인 없이 변경하지 않는다.

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

### 3.2 출력 필드 정의 (legacy distribution/pre-open)
| 필드 | 의미 |
|---|---|
| ACTIVE | 텔레그램 + paper ledger 에 들어가는 실제 추천 후보 |
| WATCH_ONLY | 텔레그램 미발송. shadow ledger 에 기록해서 정책 실험 |
| SILENCE | 텔레그램 미발송. risk-off 또는 setup 부재 |
| edge | 거래비용 차감 전후를 반영한 단순 EV proxy. 최종 성과 판단은 live ledger 기준 |
| 검증 hit / signal | bucket calibration 이 있으면 검증 hit, 없으면 raw fallback signal 로 표기 |
| policy | 왜 ACTIVE 로 승격됐는지 또는 왜 관찰 후보인지 설명 |

### 3.3 톤 원칙 (legacy formatter)
- 텔레그램은 매일 상태를 보낸다. ACTIVE가 있으면 추천, 없으면 침묵/상태
- 자동 실거래 주문 없음. 사용자가 직접 판단
- raw 확률처럼 오해될 수 있는 모델 점수는 알림에서 제거
- legacy CLI에는 `--send-silence-telegram` 옵션이 있지만 현재 daily runner는
  이 옵션을 사용하지 않는다

### 3.4 legacy 포맷
`format_detector_beta`, `format_daily_alert` 는 보존하지만 현재 메인 운영 X.

### 3.5 텔레그램 봇 — 기존 MAE 봇 공유

**별도 봇 발급 X**. gan_t / xsec_alpha 가 이미 쓰는 **MAE 봇** 그대로 사용 (사용자 본인 봇).

설정 (`prelude/.env`, `.gitignore` 처리됨):
```
TELEGRAM_BOT_TOKEN=...   # gan_t/.env 의 값
TELEGRAM_CHAT_ID=...     # 동일
PRELUDE_DASHBOARD_PIN=... # installer/publish 필수
```

운영 셸은 `deploy/load_runtime_env.sh`의 strict parser로 `.env`를 검증한 뒤
허용된 세 키만 환경변수로 내보낸다. `.env`를 셸 코드로 `source`하거나
`notifier/telegram.py`가 암묵적으로 읽지 않는다.

legacy formatter의 prefix는 `🌅 prelude`다. 현재 R1/pump v2 메시지 계약은 위
실발송 진입점이 소유하며 사용자 승인 없이 바꾸지 않는다.

**연결 테스트**:
```bash
bash -c 'source deploy/load_runtime_env.sh &&
  load_prelude_runtime_env .env venv/bin/python &&
  venv/bin/python -c "from notifier.telegram import send_telegram(\"test\")"'
```

legacy CLI 기본값은 ACTIVE가 없으면 발송하지 않으며 현재 daily runner도
`--send-silence-telegram`을 붙이지 않는다.

---

## 4. drift_detector (legacy·미배선, ops/drift_detector.py)

### 4.1 측정
아래는 과거 설계다. 현재 production evaluator 호출자와 timer가 없어 자동 실행되지 않는다.

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

## 5. ic_gate (legacy 계획, 모듈 미구현)

### 5.1 역할
**진단용 (사후 보고)**. 게이트로 신호 막지 않음 (CLAUDE.md §2.3).

### 5.2 측정
주 1회 실행은 과거 계획이며 현재 `ops/ic_gate.py`와 scheduler가 없다:
- 지난 7d IC, 30d IC, ICIR
- `output/ic_history.json` 누적
- 텔레그램 주간 리포트에 옆에 표기

### 5.3 promotion 결정에 사용 X
재학습 promotion gate 는 net Sharpe / hit rate 기준 (LEDGER §8.2). IC 는 옆에 참고만.

---

## 6. retrain pipeline (legacy·미등록)

`scripts/retrain_run.sh`와 `signals/retrain.py`는 남아 있지만 7개 systemd timer나
활성 cron에는 등록되지 않았다. 구현 promotion gate도 아래 설계와 완전히 일치하지
않으므로 사용자 승인과 재검증 전에는 실행·배포하지 않는다.

### 6.1 흐름
**Cadence (주 1 회 / 일요일 KST 06:00) 는 초기값** (CLAUDE.md §2.5). 데이터 보고 조정:
- drift 빈도 자주 → cadence 짧게 (예: 일 1 회 light retrain)
- 결과 안정 → cadence 길게 (예: 월 1 회)
- promotion gate 통과율 추적 → 항상 fail 이면 모델 재설계 신호

일요일 KST 06:00은 과거 설계이며 현재 자동 실행되지 않는다:

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

## 7. 데이터 수집 (daily runner + `data/collector_*.py`)

### 7.1 cadence
- 업비트 D1: 08:50 preopen, 09:05 distribution과 close runner에서 갱신
- 업비트 4h: 09:05 distribution과 09:30 close에서 갱신
- 업비트 15m: 08:50 legacy preopen 및 09:30/10:05 close에서 갱신
- 바이낸스 D1: 09:05 pump v2 직전에 갱신
- Binance/Upbit 1h와 매시간 collector는 현재 scheduler 미등록

### 7.2 freshness 보장
- 각 runner가 필요한 수집 성공 여부와 exact PIT boundary를 함께 검사
- 낮은 순위 신규 ticker 하나가 전체 Top100 signal을 막지 않되, Top100 누락은 fail closed

### 7.3 retry / 안정성
- 업비트 / 바이낸스 API 일시적 fail → collector retry/backoff
- 최종 실패는 runner nonzero와 systemd `OnFailure`로 전파

---

## 8. 일관성 검증 (현재 inline provenance)

현재 R1/pump 경로는 snapshot/decision→delivery receipt→전용 ledger identity를
쓰기와 청산 단계에서 검증하고 불일치 시 fail closed한다.
`scripts/verify_telegram.py`는 legacy `output/ledger.csv`용 수동 검사이며 timer에서
호출되지 않는다.

---

## 9. 로그 / 헬스 체크

### 9.1 로그 파일
- `output/cron_preopen*.log` — pre-open 추론 / close 로그
- `output/cron_dist*.log` — distribution 추론 로그
- `output/cron_close*.log` — distribution ledger 청산
- `output/cron_preopen_close*.log` — pre-open ledger 청산
- `output/cron_publish.log` — dashboard publish
- `output/cron_heartbeat.log` — evidence·publish·ledger·DB heartbeat

systemd stdout/stderr는 journal에 남고 runner는 위 날짜별/고정 로그도 쓴다.
저장소 안에는 1주 후 압축하는 log rotation 구현이 없으므로 `journalctl`과
`output/cron_*`을 함께 확인한다.

### 9.2 헬스 체크
- `scripts/health_check.py --channel recommend-preopen` — R1 preopen D1 exact
- `scripts/health_check.py --channel recommend` — R1 open D1 exact
- `scripts/health_check.py --channel preopen` — legacy preopen D1 + 15m exact
- `scripts/health_check.py --channel distribution` — legacy distribution D1 + 4h exact
- 문제 시 텔레그램 alert

---

## 10. 책임 경계

| 이건 OPS 의 일 | 이건 OPS 의 일 X |
|---|---|
| systemd timer / daily runner | 라벨 / 모델 → SIGNAL |
| 현재 health·freshness·provenance gate | 가상 사이징 → LEDGER |
| 텔레그램 메시지 포맷 / 전송 | 가상 진입 청산 → LEDGER (호출만) |
| stage 실패 전파·일관성 검증 | 사용자 매매 일지 → NOTES |
| legacy drift/retrain 설계 보존 (운영 미배선) | 모델 architecture → SIGNAL |
| backup·publish·heartbeat | 백테스트 결과 분석 → SIGNAL |

---

## 11. 핵심 파일 인덱스

| 파일 | 역할 |
|---|---|
| `deploy/crontab.txt` | 비활성 과거 cron 참고 자료 |
| `deploy/prelude-*.{service,timer}` | systemd timers |
| `deploy/install_systemd.sh` | 단일 scheduler preflight/install |
| `deploy/load_runtime_env.sh` | strict `.env` parser/export |
| `deploy/run_pipeline_stage.sh` | close/publish/heartbeat stage evidence |
| `deploy/pipeline_marker.py` | stage marker 검증 |
| `scripts/daily_run_preopen.sh` | KST 08:50 pre-open |
| `scripts/daily_run_distribution.sh` | KST 09:05 distribution |
| `scripts/daily_close_distribution.sh` | KST 09:30 distribution close |
| `scripts/daily_close_preopen.sh` | KST 10:05 pre-open close |
| `scripts/publish_dashboard.sh` | KST 10:10 dashboard publish |
| `scripts/train_recommendation_meta.py` | closed ledger 기반 meta-label 학습 |
| `scripts/retrain_run.sh` | legacy 수동 재학습 (scheduler 미등록) |
| `scripts/health_check.py` | 일일 헬스 체크 |
| `scripts/heartbeat.sh` | KST 10:30 운영 heartbeat |
| `ops/preflight.py` | legacy 추론 전 게이트 (`predict_today_legacy.py`만 호출) |
| `ops/decision_policy.py` | ACTIVE / WATCH_ONLY / SILENCE 정책 |
| `ops/recommendation_quality.py` | historical evidence 기반 추천 confidence / demotion |
| `ops/policy_gate.py` | live/replay policy promotion 판단 |
| `ops/drift_detector.py` | legacy drift 계산 (production evaluator 미배선) |
| `notifier/telegram.py` | 텔레그램 봇 |
| `notifier/format.py` | 메시지 포맷터 |
| `output/cron_*.log` | 운영 로그 |
| `output/drift_state.json` | legacy drift evaluator 실행 시 생성 (현재 bootstrap 미생성) |
| `output/retrain_history.json` | legacy retrain 실행 시 생성 (현재 미생성) |
