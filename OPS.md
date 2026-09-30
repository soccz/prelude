# OPS.md — 매일 자동 운영 인프라

> **현재 운영: R1 preopen/open. pump-v2는 2026-08-05 KILL로 종료됐다.**
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
   │  └─ 첫 실패면 current-boundary 결손만 gate별 1회 재수집 후 동일 gate 재검증
   ├─ immutable R1 snapshot → Telegram receipt → 전용 ledger
   ├─ 4h update + exact closed-boundary gate → legacy distribution record-only
   ├─ R2 / A1 / pump v1 record-only
   └─ pump-v2 terminal KILL 검증 → 정상 no-op
      (신규 decision/receipt/ledger 없음, terminal 감사 marker/log만)

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
| **07:30 매일** | 22:30 전일 | full pytest selftest (Telegram kill-switch) | `venv/bin/python -m pytest` |
| **08:45 매일** | 23:45 전일 | 공개 체결·호가 → 인과 피처 → 별도 비교 실험 기록 | `scripts/capture_recommend_microstructure.py` |
| **08:50 매일** | 23:50 전일 | R1 preopen 발송 + legacy record-only | `scripts/daily_run_preopen.sh` |
| **09:05 매일** | 00:05 | R1 open 발송 + challenger 기록; pump-v2 terminal no-op | `scripts/daily_run_distribution.sh` |
| **09:30 매일** | 00:30 | distribution paper/shadow ledger 청산 | `scripts/daily_close_distribution.sh` |
| **10:05 매일** | 01:05 | pre-open 청산 + 전일 R1 24h label/evaluator | `scripts/daily_close_preopen.sh` |
| **10:10 매일** | 01:10 | dashboard JSON 빌드 + publish | `scripts/publish_dashboard.sh` |
| **10:30 매일** | 01:30 | evidence·publish·ledger·DB heartbeat | `scripts/heartbeat.sh` |

표의 시각은 nominal calendar다. `RandomizedDelaySec` 때문에 backup은 최대 120초,
preopen/preopen-close는 최대 30초, microstructure는 random delay 없이 AccuracySec=5초,
나머지는 최대 60초 뒤 시작할 수 있다.

### 1.2 단일 scheduler 계약

지원 scheduler는 `deploy/prelude-*.service`와 9개 timer다(기존8개 + 독립 microstructure).
새9번째 timer의 실제 설치 결과는 아래 §1.3과 PHASES 8차의 설치 검증을 따른다.
`deploy/crontab.txt`는 과거 참고 자료이며 활성화하면 안 된다. 설치기는 전체 사용자
cron과 설치/등록된 prelude timer를 읽고, 중복 작업이 있으면 자동 수정하지 않고
fail closed한다.

```bash
# 먼저 .env에 PRELUDE_DASHBOARD_PIN을 추가
sudo bash deploy/install_systemd.sh --check-only
sudo bash deploy/install_systemd.sh
# 기존8개 정의/상태는 보존하고 새 공개 수집기만 추가할 때
sudo bash deploy/install_systemd.sh --add-microstructure
```

08:45 수집 및 08:50·09:05 signal timer는 늦은 catch-up을 막기 위해 `Persistent=false`,
나머지는 `Persistent=true`다. 9개 service는
`OnFailure=prelude-failure-alert@%n.service`로 실패를 알리고, stage wrapper가
후속 publish를 선행 stage 성공 증거와 연결한다.

**09-09 설치 완료 확인:** selftest unit은 추천 서비스의 `Before`와 close 서비스의
`After`를 제거했다. 07:30 전수검사·실패 경보·timer는 유지하며 초기 Nice19/CPU50%/메모리4GiB,
낮은 CPU·I/O weight와 수치 라이브러리 단일 스레드로 제한한다. CPUQuota50%는 서버 전체의50%가 아니라
논리 CPU0.5개분이다. 이는 자원 경합이 전혀 없다는 보장이 아니다.
처음 AI의 `sudo -n` 시도는 비밀번호 필요로 실패했지만, 이후 사용자가 서버에서
`--update-selftest`를 적용했다. 09-09 후속 읽기 검사에서 설치19개 파일의 저장소 일치와
loaded selftest의 순서 의존 제거·CPUQuota500ms/1s·메모리4GiB·Nice19를 확인했다.
현재 추가 설치는 필요 없다. 다시 점검할 때는 다음 읽기 전용 명령을 사용한다.

```bash
sudo bash /home/soccz/22tb/prelude/deploy/install_systemd.sh --check-only
```

`--update-selftest`는 이미 설치된 service 하나만 교체하며 나머지18개 설치파일과9개 timer 상태를 보존한다.
실행 전/재로드 후 selftest가 inactive 또는 failed인지 확인하며, 실행 중이면 거부/복구하고 중단시키지 않는다.
두 상태 관측은 외부 관리자와의 원자적 실행 배제가 아니다. 다른 설치 정의 불일치도 묵인하지 않는다.
설치 전에는 지연 부팅 시 예전 `Before` 때문에 추천이 selftest 종료를 기다릴 수 있었다.
이 순서 연결은 추천 service를 영구 차단하는 배포 gate는 아니다. 코드 push 자체는 별도 pre-push
Ruff(변경 Python)+전수 pytest gate가 정상 `git push` 경로에서 차단한다.
이것은 로컬 안전장치라 Git 자체의 의도적 `--no-verify`까지 막지는 못한다.
원격에서 절대 강제하려면 별도 CI required check와 branch protection이 필요하며,
현재 저장소에는 그 원격 정책이 구성되어 있지 않다.

**09-10 selftest 후속 보완:** 아침 실제 실행은2,913 PASS/2 FAIL이며 두 실패는
spawn 준비와 작업 시간의 혼합 제한(join30초, barrier10초)에서 발생했다. CPU50%는
그대로 유지하고, 해당 동시성 검사는 전체 자식 준비에 초기90초를 별도로 부여한다.
모든 자식이 import·격리를 마쳐야 동시에 시작하며, 실제 작업은 기존30초/10초의
그룹 공통 기한으로 검사한다. 시작/작업 실패 시 terminate→join→kill→join으로 정리한다.
자식 수·spawn 방식·결과 단언·전체 pytest 실행은 유지한다. 준비 지연 주입으로 취약한
제한 결합을 확인했다. `systemd-run --user` 임시 작업은 이 서버에서 CPU 제한이 실효 적용되지 않아
대체 증거로 쓰지 않았다. 이후 사용자가 실제 system service를 실행해 **3,000 PASS/878.74초**를
확인했다.14:47 KST 읽기 검증에서도 CPUQuota500ms/1s 유지,14:27:21→14:42:03 정상exit0,
native `ops.selftest_status`의 `passed`/exit0을 확인했다. 이번 운영 검증은 완료 상태다.

10:30 heartbeat는 `ops.selftest_status`로 실제 systemd 최신 실행도 조회한다.
**KST 오늘 시작·종료한 정상 exit0**만 통과하며 실패·미실행·전일 성공·실행 중·조회 오류는
기존 경고 묶음에 포함한다. 내부8초/외부15초 제한으로 후속 점검을 무한히 막지 않는다.
나중에 당일 재실행이 성공하면 최신 상태는 정상으로 돌아오지만 이전 실패 경보·journal은 남는다.
이는 마지막 실행 상태 검사이지 현재 코드 리비전 인증이나 추천 성능 평가가 아니다.
서비스/타이머 정의 변경이 없으므로 **설치·daemon-reload는 필요 없다**.

```bash
# 읽기 전용: 실패/미완료/조회 불능은 exit1
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -B -m ops.selftest_status --format text
# 수정본의 실제 CPU50% 환경 전수검증이 필요할 때, 기존 selftest가 실행 중이 아닌지 확인 후
sudo systemctl start prelude-selftest.service
journalctl -u prelude-selftest.service -n 80 --no-pager
```

코드 수정만으로 failed 상태를 지우거나 성공으로 바꾸지 않는다.09-10에는 실제 후속 성공이
확인돼 정상으로 돌아왔으며 추가 수동 검사는 필요 없다. 이후07:30 예약 전수검사는 계속 실행된다.

**2026-07-29 반영 상태:** 8개 timer를 포함한 저장소 unit과
`/etc/systemd/system` 설치본을 동기화했다. 설치 후에도 위 `--check-only`를
통과해야 적용 완료로 간주한다.

### 1.3 공개 체결 정보의 독립 검증 경로 (2026-09-07)

#### 공개 대시보드의 현재 상태 계약 (2026-09-09)

`ops/dashboard_current.py`는 기존 암호화5파일 중 `summary.current_system`
(`prelude_dashboard_current.v1`)만 추가한다. 기존 `channels.preopen`은 역사적 구모델 원장이며
현재 R1 장전 알림과 합치지 않는다. R1 두 슬롯의 발송 증거와 연구 기록 상태를 별도 표시한다.

- 텔레그램 서버 수락과 사용자의 열람/실제 매매는 다르다. 미관측 후보 수는 null이며0이 아니다.
- 두 연구의 기존 읽기 전용 probe를 각30초 제한으로 호출한다. 실패·불일치·이전 예정일 누락은
  상태로 표시하고 R1 전달 카드를 숨기지 않는다. 원본 예외 문자열/경로/메시지ID는 내보내지 않는다.
- 이 projection은 누적 outcome 검증기가 아니다. 기록 완료/무교체/선정 변경을 효과 입증으로
  표시하지 않으며 비교 표본 수·수익률을 추정하지 않는다. 과거 asof에는 오늘 연구 상태를 붙이지 않는다.
- 개인 `NOTES.md`는 더 이상 게시 생성기에서 읽지 않는다. 기존 암호화 과거 원장 형식은 유지한다.
  이전에 게시된 Git 이력까지 삭제하는 작업이나 사용자의 PIN 변경은 포함하지 않는다.
- `validate_dashboard_assets.py`가 현재 상태의 필드 allowlist·시각·자료형을 검사한다.
  과거 암호화 세대는 호환하며 새 화면은 현재 상태 필드가 없으면 미제공이라고 표시한다.
- 상속 `PRELUDE_PUBLISH_LOCK_FD` 검증 실패는 즉시 종료한다. 인증된5파일·동일세대·출처 검사를
  통과한 임시 clone의 data만 게시하며 공유 Pages 작업 폴더를 덮어쓰지 않는다.

**09-30 공개 화면 후속:** 별도 `researchCheckpoint`는09-30에 검증한 Top10 공통20일,
선별안D/E 미채택, 시즌별 사전 기록 준비를 날짜 고정 개발 요약으로 표시한다.
이는 일일 자동 효과 집계가 아니며 이후 표본 수를 자동 갱신하지 않는다. 기존 운영 카드·
암호화5파일 스키마·PIN·복호화 JavaScript는 유지하고 자세한 개발일지와 연결한다.
일일 상태와 연구 효능을 혼동하지 않는 표시 변경이며 실알림/실험 정책 변경은 없다.

**09-10 공개 준비 중 보안 발견:** 과거 공개 문서에 현재 대시보드 PIN과 같은 리터럴이 있었다.
현재 문서와 테스트 예시의 같은 값은 제거했으나 Git 이력 노출은 남는다. 이를 안내한 뒤
사용자가 제거 가능한 본문만 정리하고 기존 PIN은 유지하라고 결정했다. runtime PIN·암호화5파일·
예약 작업·Git 이력은 변경하지 않았다. 코드·소개 게시 및 공개 HTML2개/암호화5파일의 일치와
인증 검증까지 완료했다. 기밀성 회복 조치가 아니며 공개 결과는 PHASES13차 참조.

**09-08 16:01 당시 확인:** 사용자 설치 완료.9개 timer loaded/enabled/active,
설치파일19개와 저장소 SHA 일치. 새 timer는 **09-09 08:45 KST** 첫 실행을 기다린다.
수집 service는 아직 실행 전이며, 설치 성공이 첫 수집/실험 성공을 뜻하지 않는다.
별도 root `--check-only` 재실행은 에이전트의 sudo 인증 제한으로 못 했으나,
사용자 설치기의 cron/중복·설치 후 검사 완료 로그와 직접 설치 상태 검증을 확인했다.

`prelude-microstructure.timer`는 R1 발송의 선행 의존성이나 pipeline lock을 사용하지 않는다.
공개 trade/orderbook만 구독하고 키·주문·추천 전송·자동 재학습을 사용하지 않는다.
오늘 실제 R1 snapshot이 만들어지면 수집을 종료하고, 그 snapshot의 **계산 시작 전 300초**로
event/received 양 시각을 다시 자른다. native capture의 종료 기준은 기존대로 decision_completed_at이다.

- 최초 목표일은2026-09-08이었으나 설치 전 오전 창이 지나 수집하지 못했다.
  09-08 오후 기준 설치 후 첫 정규 수집은 **2026-09-09 08:45 KST**다.
  실험 달력은 원래09-08 시작을 유지해 누락을 숨기지 않는다. 과거 수신시각을 복원하지 않는다.
- 초기 자원 예산: depth1, warmup600초, 최대 대기2400초, 채널별 수락 payload512MiB,
  CPU50%·메모리2GiB·oneshot 최대2700초·종료90초. payload 한도는 실제 파일/누적 디스크 용량 한도가 아니다.
- **09-08 수집 전 저장공간 보호 추가:** 예약 wrapper는 공개 시장 조회 전과 수집 시작 직전에
  raw/feature 경로와 별도 trial 경로의 가장 가까운 기존 디렉터리 파일시스템을 각각 검사한다.
  일반 사용자 가용량 `f_bavail × f_frsize`가 초기 `1GiB + 4 × 2채널 × payload한도`보다 작거나
  조회에 실패하면 새 연구 수집만 nonzero로 중단한다(기본 **5GiB**). 경로를 만들거나 증거를 삭제하지 않는다.
  계수4·여유1GiB는 정규 실측 후 조정할 운영 초기값이지 압축/메타데이터 최대치의 증명이 아니다.
  이는 수집 **진입 전 검사**이며 실행 중 다른 프로세스의 공간 소비·디스크 quota·누적 보존 정책을 대신하지 않는다.
  실제 추천 선행 의존성은 추가하지 않으며 unit 변경/재설치도 없다.
- 원본·manifest·`recommend_features.json`·`trial_evaluation.json`은 각 capture 디렉터리에 보존한다.
  별도 `output/recommend_microstructure_trials/<date>/r1_boundary_trade_imbalance_v1/`에
  원 Top3/시험 Top3와 `score.json`→`commit.json`을 덮어쓰기 없이 저장한다.
  이 별도 디렉터리도 기존 정규 evidence 백업 목록에 포함한다(실제 첫 백업은 정규 실행 후 확인).
- 원본은 순차 검증해 전체 호가를 메모리에 쌓지 않는다. 무체결은 null, 장애는 unavailable이며
  후보100개를 그대로 유지한다. canary 성공은 실험 적격 또는 추천 효과의 증거가 아니다.
- score 파일·디렉터리의 내구 저장 후 관측한 시각이 기존 receipt의 canonical entry보다 빨라야
  prospective 비교에 들어간다. 늦음·미확인 저장·미성숙 결과·수집 누락 날짜를 구별한다.
- 기존8개 byte/inode·enabled/active는 add-only 설치가 보존한다. 정의 불일치를 자동 수리하지 않으며,
  동일 활성 timer를 재시작하지 않는다. 실제 설치에서도 기존8개의 활성 진입시각은 유지됐다.
  다만 daemon-reload 전후 다음 실행의 초 단위 지연값은 달라졌으므로 시각 완전동일을 보장하지 않는다.
  전체 `--check-only`는 의도적으로 비활성인 timer도 실패로 보고한다.

읽기 전용 비교 확인(과거 보고서 덮어쓰기 없음):

```bash
PRELUDE_FORBID_TELEGRAM=1 venv/bin/python -B scripts/evaluate_recommend_microstructure_trial.py
systemctl status prelude-microstructure.timer --no-pager
```

이 경로는 새 정보가 유효한지를 검증하는 **record-only 실험**이다. 기존 R1 모델·순위·발송은 그대로다.
NTP yes는 정밀 시각 오차의 증명은 아니며, 같은 호스트의 OnFailure는 외부 독립 감시를 대신하지 못한다.
기존 selftest의 지연 부팅 순서 의존은 위의 한정 설치로 제거했다. close/publish의 연구 상태 결합은 신규 수집기와 별개다.

#### 첫 정규 실행 수리·Top10 별도 시험 연결 (2026-09-09)

09-09 수집은08:45:02 시작해282시장의335,706행을 저장했으나,09:09:02 피처 생성에서 실패했다.
원인은 `int(datetime.timestamp()*1e9)`가 원본 snapshot 완료시각을96ns 작게 기록한 것이다.
`data/upbit_microstructure.py::datetime_to_ns`를 UTC 정수 timedelta 산식으로 수리했다.
실패한 manifest/raw/snapshot은 그대로이며 오늘을 성공으로 소급 바꾸지 않는다.
같은 날 selftest의1실패도 수신 스레드 시작 직후 테스트가 종료 신호를 보내는 경쟁 조건으로 재현했다.
테스트가 실제 종료 이벤트를 기다리도록 수리했고 production collector의 종료 동작은 바꾸지 않았다.

**09-10부터 기존 설치 service가 읽는 wrapper**는 기존 동점 score/commit → 별도Top10 score/commit →
각 누적평가 순서로 실행한다. 새 시험은 `r1_top10_trade_imbalance_v1`, 저장 위치는
`output/recommend_trade_shortlist_trials/<date>/<trial_id>/score.json,commit.json`이다.
기존 동점 시험의09-08 시작일과 누락일은 보존하고 새 시험만09-10 시작이다. 기존 R1 발송·순위·라벨은 바뀌지 않는다.

- 새 시험 저장공간/발행 오류는 기존 동점 발행·평가를 막지 않지만 최종 nonzero로 경보한다.
  기존 발행 자체가 불확실하면 새 시험을 대신 성공시켜 정상으로 처리하지 않는다.
  양쪽 점수부터 내구 저장하므로 누적 과거 평가가 새 점수의 진입 전 저장을 늦추지 않는다.
- 새 평가 파일은 capture 디렉터리의 `trade_shortlist_evaluation.json`이다. 과거 보고서는 덮어쓰지 않는다.
  기존 raw 전체 해시 검증을 재사용하며 중복 해시는 남아 있다. 실제 신규 publisher의 정규 실행 시간은 아직 미측정이다.
  매일 누적평가 비용은 자료와 함께 증가할 수 있으며 기존 service 실행 상한 안에서 관찰해야 한다.
- 새 trial 디렉터리도 기존 evidence 백업 목록에 포함했다. fixture archive 회귀는 실제 첫 백업 성공과 다르다.
- `ops.recommend_trade_shortlist_status`가 새 namespace를 별도로 확인한다.09-10 전 not_started,
  당일10시 전 waiting과 직전 기한이 지난 날짜를 확인하며,10:30 heartbeat가 경고한다.
  현재 하루의 native raw/소스 검증, score/commit과 평가의 당일 선정·시각 연결을 확인한다.
  기존 경량 probe와 달리 원본 해시를 읽으므로 별도30초+종료5초로 제한한다. 한도 초과도 조용한 성공이 아닌 경고다.
  evaluation JSON은4MiB 초기 상한이다. 성과/라벨/진입 전 적격은 이 운영 probe의 검사 대상이 아니다.
- native reader는 현재 generator 해시 일치도 요구한다. 동결 소스가 바뀌면 과거 파일을 새 코드로 정당화하지 않고
  검증 실패로 남긴다. 추후 변경은 별도 버전/시험으로 설계해야 한다.
- 09-09 13:49 읽기 확인: 기존 두 슬롯은 Telegram 서버수락 증거 정상, 기존 수집은 evidence_invalid,
  새 시험은 not_started였다. 실패 원본4개 SHA도 최초 감사와 같았다. 실제 신규 적격 표본/추천 우위는 아직0/미검증이다.

읽기 전용 확인(실제 주문·발송·학습·출력 생성 없음):

```bash
PRELUDE_FORBID_TELEGRAM=1 PYTHONDONTWRITEBYTECODE=1 venv/bin/python -B -m ops.recommend_trade_shortlist_status --format text
PRELUDE_FORBID_TELEGRAM=1 PYTHONDONTWRITEBYTECODE=1 venv/bin/python -B scripts/evaluate_recommend_trade_shortlist_trial.py
```

#### 결과 마감 후 별도 실험 평가 (2026-09-30)

아침의 immutable 평가에는 전일24시간 결과가 아직 없을 수 있다. 기존 preopen-close가
원장·라벨·score 평가를 마친 뒤 두 native trial을 다시 평가하고
`output/recommend_trial_review.json`만 원자적으로 갱신한다. 원본 capture 평가/score/commit은 보존한다.
성능 비교, 실제 교체 날짜/픽 수, 교체로 들어온/빠진 종목 결과, 당시 BTC 국면별 결과를 포함한다.
실제 추천에는 국면 전환 규칙을 적용하지 않는다. 과거 성과의 아침 가용성은 라벨 기록 시각까지 검사한다.
09-30의15차부터 `adaptive_replay`에 고정R1/최근 성과 선택/같은 시장 상태 선택의
과거 정보 전용 가상 재생을 포함한다. 결정 시각은 양쪽 점수 저장 이후·진입봉 이전이고,
미완결 전일 결과는 과거 이력에 넣지 않는다. 초기10관측일/최소5일·하방/상방 보호 기준은
SIGNAL §0.2와 PHASES 15차에 명시했다. 일별 선택/유지 사유·이력 날짜·전환 횟수·paired CI를
기록하지만 **실제 전환·자동 승격·새 학습은 없다**. 재생 자체는 결정 전 정책 기록이 아니며,
16차에서 준비한 별도 사전 기록 시험은 아래를 따른다.
원본20일 재생에서 두 정책의 실질 교체는0회였으므로0차이 CI를 효과 검증으로 해석하지 않는다.

heartbeat의 별도20초 probe는 보고서 무결성·소스 버전·당일 갱신·직전 마감일 상태만 검사한다.
오늘 결과 대기는 정상, 어제 결과 누락은 경고이며, 수익 부진은 운영 실패가 아니다.
초기 유예10:25는 close 소요시간으로 조정할 운영값이다. raw 과거 전체 재검증은 refresh만 수행한다.
refresh는 기존300초 연구 제한/실패 전파를 따른다. 추가 timer/설치/실발송/모델 변경은 없다.
09-30의20일 실제 원본을 재검증한 첫 실행은211.23초/exit0, 최종 재실행은81.86초/exit0이었다. 누적 해시 비용은 날짜와 함께
증가하므로 일일 소요시간을 관찰하고, 상한에 가까워지면 무결성을 보존하는 증분 평가를 별도 설계한다.
시간초과를 정상으로 숨기거나, 검증을 생략하거나, 상한을 자동으로 늘리지는 않는다.
15차 전환 재생 포함 최종 갱신은67.12초/exit0, 읽기 probe는0.58초/exit0으로 확인했다.
이는 로컬 직접 실행 검증이며 다음 정규 예약 회차까지 이미 완료됐다는 뜻은 아니다.

```bash
# 읽기 전용 최신 상태와 효과 요약
PRELUDE_FORBID_TELEGRAM=1 PYTHONDONTWRITEBYTECODE=1 venv/bin/python -B -m ops.recommend_trial_review --format text
# 수동 재평가: 기존 증거는 읽기만, 전용 파생 보고서만 갱신
PRELUDE_FORBID_TELEGRAM=1 PYTHONDONTWRITEBYTECODE=1 venv/bin/python -B -m ops.recommend_trial_review --refresh
```

09-29 selftest에서 확인한 close-plan coproc 경합도 두 close와 env loader에서 수정한다.
부모 소유 FD와 자식 종료코드를 유지하고 미종결 NUL/불완전 sentinel은 거부한다.
환경 변수는 전체 허용 키 검사가 끝나야 export되며 `.env`를 셸 코드로 실행하지 않는다.

#### 시장 상태별 정책 사전 기록 (09-30 준비, 10-01 open 시작)

**09-30 추가: 웹에서 누적 진행 상황 확인.** 기존 dashboard build가
`ops.dashboard_research`를20초 제한으로 호출해 검증된 post-label review 집계를
`summary.research_progress`에 넣는다. 두 native 시험의 비교일/실제 교체/제외·대기와
상승·하락 경험률/기존net/paired 날짜 block CI, 국면 사전 기록의 적격·마감일/교체 수를
PIN 뒤 `#researchProgressSection`에 표시한다. 기록 시점과 결과 마감 시점을 구분한다.
실시간 상태가 아니라 게시 시점 확인본이며 국면 정책별 성과/자동 채택 패널은 아니다.
오래된/없는/시간초과 보고서는0건 대신 확인 불가,0비교일 성과와5일 미만 CI는 미제공이다.
연구 실패가 기존 실알림 상태 카드를 숨기지 않으며 새로운 수집/원장/일정/설치는 없다.
직접 읽기 점검은 `venv/bin/python -B -m ops.dashboard_research --probe --now <timezone포함시각>`이다.

09-30 기존 백업 추가 읽기 감사에서는 실제 archive2,136항목 구조/전체SHA,
최근 immutable24파일의 현재 원본 일치, 일봉·15분봉 DB SHA/integrity를 확인했다.
새 체결 raw와 두 시험도 보관돼 있었다. 아직 생성 전인10-01 국면 기록의 백업 성공,
실제 복원 훈련 또는 별도 물리 장치 사본을 확인한 것은 아니다. 상세는 PHASES22차다.

기존 capture가 두 시험의 점수와 native 아침 평가를 보존한 뒤
`ops.recommend_regime_forward.record_forward`를 호출한다. 결과는
`output/recommend_regime_forward/YYYY-MM-DD/r1_regime_forward_v1/{score,commit}.json`에
단 한 번 저장한다. 원래 R1 고정·최근 성과 선택·같은 상태 성과 선택의 종목/사유/이력을 함께 남긴다.
별도 timer·설치·실발송은 없고 기존 추천 경로에 새 정책을 적용하지 않는다.
새 기록 디렉터리는 Git 제외이며 기존 evidence 백업에 원본 capture·두 시험과 함께 포함한다.
이는 같은 서버의 기존 백업 확장이며 별도 물리 장치 백업을 추가한 것이 아니다.

- 실제 계획/score fsync 후 관찰 시각을 사용한다. 검증된 성공 receipt 뒤 다음15분봉보다
  늦거나 같으면 `late`이며 정상 성과 표본에 넣지 않는다. 오전 평가가 늦어지면 소급하지 않는다.
- score만 남은 중단은 `uncertain`; 재실행해도 commit을 보충하거나 기존 score를 바꾸지 않는다.
  구조적으로 후보가 없었던 `unavailable`은 기록 정상/효과 미관측이며, 누락·손상과 구별한다.
- 다음날 기존 post-label review의 `forward_evaluation`이 **당시 저장된 선택**만 채점한다.
  미완결·누락은0수익으로 채우지 않고, 적격일/실제 교체일·픽 수를 분리한다.
  raw를 매 정책일마다 중복 검사하지 않고 새 native 평가와 합쳐 중복 제거한 입력 집합을 검증한다.
- heartbeat는30초 제한으로 원본 아침 평가/score/receipt/라벨과 선택의 결합을 읽기 검사한다.
  **원본 raw 내용 전체 검사는 아니다.** 전체 검증은 발행과 post-label native 평가에서 수행한다.
  09:20 KST부터 당일 기록, 이전에는 전일을 기대한다. 최초10-01 09:20 전에는 시작 전 정상이다.
  오늘 시작 전/늦은 기록/원본 결손을 혼동하지 않으며, 이상은 기존 heartbeat 경고 묶음에 전달한다.
  결과 리뷰의 운영 경고는 최신 마감일 기준이다. 과거 결손은 날짜별 제외 이력에 계속 남기되,
  복구할 수 없는 하루 때문에 이후 정상일에도 매일 실패로 판정하지 않는다. 소급 보충은 금지한다.
- 누적 raw 평가의 장기 소요시간은 계속 관찰해야 한다.300초 연구 제한 초과를 정상으로 숨기거나
  자동으로 제한을 늘리지 않는다. 실제 첫 정규 발행·다음날 평가와 추천 우위는 아직 미검증이다.

09-30 최종 실제 보고서 갱신은68.59초/exit0이었다. 원본 두 시험의20일 비교는 유지됐고
새 정책 사전 기록/완결 비교는0건이다. 읽기 probe는 프로세스 전체 약0.39초/exit0이었다.
정책 자체의 가벼운 연결 검사 범위는 raw 전체 검증과 구분하며, raw 변조는 실제 성과 평가에서 거부한다.

```bash
# 읽기 전용: 정책 누락·연결 상태만 검사, 파일 생성/복구/학습/발송 없음
PRELUDE_FORBID_TELEGRAM=1 PYTHONDONTWRITEBYTECODE=1 venv/bin/python -B -m ops.recommend_regime_forward --format text
```

#### 수집 미실행·중간 중단 감시 (2026-09-08 추가)

기존 `OnFailure`는 서비스가 실행하다 실패한 경우를 알리지만, timer가 아예 실행하지 않은 날에는
실패 이벤트가 없다. 기존10:30 heartbeat에 `ops.recommend_microstructure_status`를 연결해 이 공백을 보완했다.
새 timer·sudo 설치·알림 포맷 변경은 없다. 정상은 로그만, 이상은 기존 heartbeat 경고 묶음에 한 항목으로 포함한다.
수집 실패 직후 경보와 나중 heartbeat의 미해결 상태 경고가 모두 올 수 있다. 상세 heartbeat 경보 전달 성공 시
기존 generic OnFailure 중복은 억제하고, 전달 실패 시에만 fallback을 사용한다.

- 운영 최초 기대일은 **09-09**, 연구 표본 달력은 **09-08**을 유지한다. 09-08 누락을 지우지 않는다.
- 완료 기한은 초기10:00 KST:08:45+실행45분+종료90초에 여유를 더했다. 실제 경보는 **heartbeat10:30 이후**다.
  전수 selftest/시스템 부하 등에 따라 heartbeat 자체가 지연될 수 있으며10:30 정각 전달을 보장하지 않는다.
  기한 전에는 오늘은waiting으로 두고 가장 최근 기한이 지난 전일도 검사해 늦은 부팅 오탐/미탐을 줄인다.
  무제한 과거 누락 탐색이 아니며, 연구 평가기의 누적 달력 진단은 별도로 유지한다.
- 당일 수집 디렉터리0개=미실행 증거 부재,2개 이상=모호성 경고다. 최신 성공본만 임의로 선택하지 않는다.
  manifest 미완결, 원본 snapshot 부재, 피처 품질 부적격, score/commit 누락, 평가 파일 누락·변조를 구분한다.
- 정상 무교체는 `complete_noop`, 시험 순위 변경은 `complete_changed`다. 실제 R1은 어느 경우에도 바뀌지 않는다.
  수집은 정상이나 필요한 동점 종목에 관측 체결이 없어 시험할 수 없으면 `complete_unavailable`로 구분한다.
  이는 조용히 남기는 자료 가용성 진단이지 유효 시험 표본 또는 정상 무교체가 아니다.
- 작은 JSON은 초기4MiB/파일 상한으로 읽고 snapshot/manifest/feature/score/commit/평가의 연결·체크섬·시각 순서를
  검사한다. 원본 gzip은 존재·일반 파일 여부·크기만 확인하며 내용은 열지 않는다. 파일 그래프 구조도 검사하지만
  현재 generator 소스와 원본 내용 전체 해시는 재계산하지 않는다. 읽는 중 변경·symlink/FIFO·중복 JSON key는 거부한다.
  점검은 외부 timeout30초+종료5초로 제한한다. 이 상한들도 실제 파일 증가·실행 시간 관측 후 조정할 운영 초기값이다.
- **통과 의미는 제한적이다:** 메타데이터 수준의 저장 경로 정상이다. 동일 크기 원본 내용 손상·소스 변경·진입 전 저장 적격·
  추천 성능을 증명하지 않는다. 후자는 기존 정본 연구 평가기가 담당한다. 경고가 나도 자동 재수집/복원/commit 재작성은 없다.

직접 확인(읽기 전용, 로그/원장/Telegram에 쓰지 않음):

```bash
PRELUDE_FORBID_TELEGRAM=1 PYTHONDONTWRITEBYTECODE=1 venv/bin/python -B -m ops.recommend_microstructure_status --format text
systemctl status prelude-microstructure.timer --no-pager
journalctl -u prelude-microstructure.service -n 50 --no-pager
```

CLI exit0은 정상/대기/운영 시작 전, exit1은 주의 필요, exit2는 호출·점검 설정 오류다.
`--now`는 장애 시각 재현용이며 과거 산출물을 만들지 않는다. 기한 전에도 전일 장애가 있으면 exit1이다.
미실행이면 timer와 서비스 로그를 먼저 확인하고, 현재 시각으로 오전 원본을 재수집해 과거 증거를 대체하지 않는다.
09-08 실제 읽기 실행은 `not_started`, 첫 timer 예약은09-09 08:45였다.09-09 실제 실패와 수리는 위 후속 항목을 따른다.

#### 기존 실제 백업 읽기 검증 (2026-09-08)

04:04 백업 `ledgers_20260908_040402_2529188.tar.gz`의 manifest/checksum/SHA 일치,
중복 없는1081개 member,09-07 open snapshot/receipt의 JSON 파싱·현재 immutable 파일과 바이트 일치를 확인했다.
archive SHA는 `c8949540fc5c216910576daa4bf003132b4be89daeac59b8989606dd104ed396`이며 원본을 변경하지 않았다.
같은 날짜 일봉·15분봉 SQLite 보관본도 checksum 및 읽기 전용 `PRAGMA integrity_check=ok`를 확인했다.
이는 전체 복원 훈련이나 새 정규 trial의 백업 성공을 뜻하지 않는다.
해당 archive의 microstructure 원본 경로는 빈 디렉터리뿐이고 trial은0개다. 새 실제 자료의 보관은 생성 후 확인한다.

**물리 장애 대비 미완료:** 현재 `backup/prelude_db`와 prelude 원본은 동일 파일시스템이다.
파일 손상/논리 실수의 복구점은 있지만 같은 디스크 고장을 버티는 독립 사본은 아니다.
다른 물리 장치/외부 저장소의 대상·용량·접근 권한이 정해져야 별도 복제를 설계할 수 있다.
외부 계정 생성·전송·원본 복원 덮어쓰기는 수행하지 않는다.

#### 핵심 결과 우선·연구 정지 시간 제한 (2026-09-08)

- `daily_close_distribution.sh`: 기존 수집 뒤 R1 open 전용 원장을 legacy paper 원장보다 먼저 갱신한다.
- `daily_close_preopen.sh`: 기존15분봉 수집 → R1 preopen 원장 → 전 유니버스 라벨 → legacy 원장 → 연구 평가 순서다.
  원장/라벨의 정의·입력·청산 인자·검증 gate는 바꾸지 않는다.
- 두 wrapper의 연구 명령(policy/meta/idea/HTML 및 preopen score 평가)은 개별 **300초** 뒤 TERM,
  추가10초 뒤 KILL로 제한한다. 초기120초는09-08 실제 score 평가의 약110.6초와 너무 가까워 반려했다.
  300초도 관측값의 약2.7배를 둔 운영 초기값이며 자료 증가·부하에 따라 재검토한다.
- 실패124/137 및 기존 실패는 계속 nonzero·critical log·OnFailure 경로로 전파한다.
  먼저 발생한 핵심 오류를 나중의 연구 오류로 덮지 않고 가능한 후속 작업은 계속한다.
  idea 보고서 실패 때 HTML을 만들지 않으며, 기존 pipeline success marker와 publish 차단도 유지한다.

이는 **핵심 결과 선처리와 연구 명령별 정지 시간 제한**이다. 연구 실패와 dashboard 게시의 완전 분리,
전체 pipeline lock 제거, 지연 부팅 selftest 의존 제거는 아니다. 핵심 수집/원장/라벨 및 legacy 작업에
새 시간제한을 적용한 것도 아니다. 따라서 모든 종류의 지연을 제거했다는 주장은 하지 않는다.
unit·일정·발송 포맷은 불변이고 기존 service가 다음 실행에서 갱신된 shell을 읽는다. 추가 설치는 없다.

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
- **2026-09-07 평가기 v3:** TopN·유동성 매칭·ATR band는 결과를 보지 않고 저장된 원후보로 먼저
  고정한다. 선정 종목이나 필요한 대조군이 `halted_no_observations`이면 그 날짜의 해당 비교를
  unavailable로 남기며 다음 순위·다른 대조군으로 대체하지 않는다. 지표 한 필드만 결측이면
  그 지표의 날짜 쌍만 제외하고 일부 종목 평균으로 메우지 않는다.
  기록된 후보/관측 결과/전달된 후보의 결과 미관측 coverage를 구분한다. 기존 확률 진단은 labeled-only,
  complete-path-only는 진단용이며 선정 필터가 아니다. 과거 보고서는 소급 덮어쓰지 않는다.
- close 게이트 기본 모드는 `close`(정상 청산) / `skip-zero-pick`(검증된 무추천일) /
  `skip-legacy-unverifiable`(계약 이전) / `skip-no-decision`(발송 파이프 자체가 죽어
  snapshot·receipt·원장 행이 전부 없는 날 — 2026-07-28 신설). skip-no-decision은
  plan과 락 하 재검증이 같은 술어(`is_no_decision_day`)를 쓰고, 검증 시
  `output/close_no_decision/{cohort}/{asof}.json` 감사 마커를 남긴다(백업 포함).
  원장 행(상태 무관)이나 receipt가 하나라도 남아 있으면 조용한 skip이 아니라
  무결성 실패로 fail-closed 한다. 이 마커 일수는 커버리지 분모 보정에 쓴다
  (무추천일과 무결정일은 다르다 — MNAR 방지).
- v2 KILL 전환에는 별도 `skip-terminal-kill`을 쓴다. 검증된 terminal state+anchor,
  `decision_date > effective_asof`, decision/receipt/원장 행 전무가 모두 참일 때만 허용하며
  pipeline death 분모에 넣지 않는다. 과거 공용 KILL이 R1을 막은 2026-08-06~08에는
  `skip-policy-blocked`로만 기록한다(`forward_valid=false`); 08-09부터 R1 receipt 계약은 다시 엄격하다.

#### 전달 불확실성과 안전한 상태 확인 (2026-09-07)

- 추천 및 보조 champion 변경 알림은 동일 send lock 안에서 **API 호출 전 발송 시도 intent를 fsync**한다.
  저장 위치는 receipt 옆 `<receipt.stem>.attempts/000001.intent.json`이며,
  검증된 receipt를 `000001.result.json`에 연결한다. 기존 receipt 형식과 성공 이력은 유지한다.
  이 디렉터리는 기존 `output/recommend_receipts` 재귀 백업 범위에 포함된다.
- API 수락 뒤 receipt 기록 전 종료하면 미완료 intent가 다음 실행을 막는다. 오래된 실패 receipt는
  최신 intent의 실패 증거가 아니다. 새 성공 receipt가 일치하면 result 파일 기록 직전 종료였어도
  중복 발송하지 않는다. 명확한 실패만 다음 시도를 허용하며 부분 수락·응답 불명은 자동 재시도하지 않는다.
- 이는 **정확히 한 번 전달 보장이 아니다**. intent 저장 직후 API 전에 죽어 실제로는 못 보냈어도
  보류될 수 있다. 임의 TTL 해제·파일 삭제 후 재발송은 하지 않는다. 장애 시 intent/receipt/로그와
  실제 채팅을 보존·대조하여 판단해야 한다. 슬롯 마감 후 당일 메시지를 소급 발송하지 않는다.
- `report_recommendation_status.py`는 snapshot/receipt/journal을 **읽기만** 한다. 미완료 시도는
  `delivery_uncertain`, 손상·불일치는 `invalid_evidence`; 과거 실패를 최신 미발송으로 오인하지 않는다.
  receipt를 읽기 전후 journal 증거를 재대조하여 조회 중 새 시도가 기록되면 확정 상태 대신
  확인 필요로 표시한다. 조회 중 증거가 손상된 경우도 실패 상태로 단정하지 않고 검증 실패로 남긴다.
  exit 0은 로컬 증거상 주의 상태 없음, 1은 확인 필요, 2는 잘못된 CLI 인자다.

```bash
PYTHONDONTWRITEBYTECODE=1 PRELUDE_FORBID_TELEGRAM=1 venv/bin/python -B scripts/report_recommendation_status.py --format text
```

- 사용자 승인에 따라 알림 확률은 **검증 중 추정치**, 09:00 가격은 **현재 체결가격이 아닌 참고가격**으로
  정정한다. 기존 종목·순위·확률값·TP/SL은 변경하지 않는다. 이미 보낸 메시지를 수정·재발송하지 않는다.
- 09-10 승인된 표시 보완: `0 < p < 0.01`은 `<1%`, 실제0은 `0%`, 1% 이상은 기존 정수
  표시를 유지한다. NaN/무한대/범위 밖/잘못된 값은 `—`다. 예전0% 메시지의 receipt/journal은
  원본 해시로 계속 검증하며 새 formatter로 재발송하지 않는다. 수치·순위·모델은 바꾸지 않는다.
- 새 `ModelSpec`의 기본값은 `challenger_only=True`다. 기존 7개 명시 등록 설정은 그대로다.
  기본 잠금이 별도 승인·모델 버전별 forward 검증 체계 전체를 대체하지는 않는다.
- 이번 변경은 저장소 Python 코드이며 systemd unit 설치/재시작은 필요 없다. 다음 정규 실행이 새 코드를
  읽지만, 실제 다음 발송 성공 여부는 그 실행의 새 intent/receipt로 확인해야 한다.
  이 조회는 독립 외부 감시가 아니므로 서버·스케줄러·네트워크 전체 장애까지 보장하지 않는다.

### 1.5 pump v2 evidence와 terminal 판정

- 새 decision→receipt→ledger 계약 활성일은 `2026-07-27`이다. 그 이전 205개
  CLOSED 행은 frozen scorecard 입력 5개 필드의 SHA-256
  `ac01ddde…94451`로 고정해 과거 수치 변조를 차단한다.
- 2026-07-27 이후 행은 canonical decision과 성공 receipt, 후보 identity가 모두
  일치해야 CLOSED 성과나 recall 근거로 사용한다. 2026-07-26 zero-pick receipt는
  검증하되 `legacy-unverifiable`로 분리한다.
- 조기 KILL 또는 2026-09-01 GO/KILL은 별도 immutable terminal state와 anchor에
  기록한다. 이 terminal은 pump-v2 전용이다. 유효 KILL은 runner의 정상 no-op이며,
  상태 누락·손상·미래시각·불일치만 fail closed한다. R1 발송과는 독립이다.
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
- risk/drift 평가기는 production runner에 미배선인 legacy 코드다. 검증 helper는
  향후 배선 검토용으로 보존하지만, production health 결과에는 포함하지 않는다
  (미실행 평가기를 `OK` 감시처럼 표시하지 않음).

### 2.2 실패 시
- 09:05 runner의 각 exact gate 첫 실패는 현재 09:00 경계가 없는 live market만
  최대 64개(초기값) 1회 재수집하고 동일 gate를 다시 평가한다. `recommend`와
  `distribution`은 검사 사이에 신규 거래가 시작될 수 있어 상한을 분리하며,
  gate별 두 번째 실패·광역 결손은 관용하지 않는다
- daily runner가 해당 채널 작업을 skip하고 nonzero를 전파
- systemd `OnFailure`가 운영 실패 알림을 담당
- `ops/preflight.py`는 `scripts/predict_today_legacy.py`에서만 쓰는 legacy helper

---

## 3. 텔레그램 알림 포맷 (notifier/format.py)

### 3.1 현재 운영 포맷

현재 실발송 진입점은 R1의 `scripts/recommend_send.py`다. 종료된 pump-v2의
`scripts/pump_detector_v2_today.py`는 immutable KILL 확인 후 전송 전에 끝난다. 아래 distribution/pre-open 예시는
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

legacy formatter의 prefix는 `🌅 prelude`다. 현재 R1 메시지 계약은 위
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

`scripts/retrain_run.sh`와 `signals/retrain.py`는 남아 있지만 지원 systemd timer나
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
- 바이낸스 D1: pump-v2 KILL 이후 신규 radar 입력 갱신에는 사용하지 않음(보존 데이터)
- Binance/Upbit 1h와 매시간 collector는 현재 scheduler 미등록

### 7.2 freshness 보장
- 각 runner가 필요한 수집 성공 여부와 exact PIT boundary를 함께 검사
- 낮은 순위 신규 ticker 하나가 전체 Top100 signal을 막지 않되, Top100 누락은 fail closed

### 7.3 retry / 안정성
- 업비트 / 바이낸스 API 일시적 fail → collector retry/backoff
- 09:05 D1 exact 결손은 2초(초기값) 후 결손 identity만 gate별 1회 targeted
  refresh한다. 전체 상한은 `recommend` 1회 + `distribution` 1회이며, 각 gate의
  재검증 실패와 64개 초과 광역 결손은 즉시 fail-closed
- 최종 실패는 runner nonzero와 systemd `OnFailure`로 전파

---

## 8. 일관성 검증 (현재 inline provenance)

현재 활성 R1 경로는 snapshot→delivery receipt→전용 ledger identity를 쓰기와
청산 단계에서 검증하고 불일치 시 fail closed한다. 기능 은퇴한 pump-v2의
decision→receipt→ledger 사슬도 과거 증거 검증과 터미널 no-op 판정에 보존한다.
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
- 10:30 heartbeat는 당일 evidence backup manifest가 묶은 archive/checksum
  SHA-256을 검증한다. 지연 부팅 catch-up에서는 manifest 미생성 상태만 bounded
  polling하고, 이미 존재하는 manifest/파일 손상·symlink는 즉시 실패한다.
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
