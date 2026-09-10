# PHASES.md — 단계별 액션 + 체크리스트

> Phase 별 구체 작업. **체크박스로 진행 추적**. 매 세션 시작 시 이 문서 head 60 줄 보고 어디까지 왔는지 파악 (CLAUDE.md §0).

---

## 실사용 강화 13차 — 일일 점검 실패·확률 표시·종합 감시 보완 (2026-09-10)

오늘 Telegram 기반 읽기 감사에서 추천2회·원장6행은 정상이나 selftest2건 실패를 확인했다.
사용자가 세 가지 보완(테스트 제한시간,0% 오해,heartbeat 누락)에 `오케이 진행해`로 승인했다.
모델·추천 순위·확률값·라벨·연구 동결 자료·과거 영수증·운영 output8개 dirty 변경은 건드리지 않는다.

### 설계·완료 경계

1. CPU50%·타이머·전수검사·결과 단언을 유지한다. spawn/import 준비와 실제 동시쓰기 기한을
   분리하고, 부분 시작 실패/준비 실패/작업 정체에서도 자식을 유한 시간 안에 정리한다.
2. `0<p<1%`만 `<1%`로 표시한다. 실제0/기존1% 이상 표시는 유지하고 잘못된 값은 `—`로 둔다.
   기존 성공 영수증과 발송 저널은 그대로이며, 새 포맷 때문에 재발송하거나 점수를 다시 계산하지 않는다.
3. heartbeat는 systemd selftest 최신 실행을 읽어 KST 오늘 시작·종료한 exit0만 정상으로 인정한다.
   실패/전일 성공/미실행/진행 중/조회 오류를 경고하고 기존 추천·다른 점검의 실행을 막지 않는다.
4. 표적 회귀·시간 지연/실패 주입·독립 검토·전수 pytest·Ruff/셸 검사 후 보고한다.
   sudo 비밀번호 없이 실제 서비스 재검증은 불가하며 로컬 PASS와 운영 제한 환경 PASS를 구분한다.

- [x] 실제 오늘 실패 journal과 설치 CPUQuota500ms/1s 재확인. 제한 환경 성공으로 위장하지 않음.
- [x] 테스트 준비90초와 작업30초/10초 분리, 공동 출발/공통 종료기한, 실패 cleanup·자식 격리.
- [x] 확률 표시26개 회귀와 과거 영수증/저널 byte 보존·중복발송 방지 확인.
- [x] selftest 읽기 검사 및 heartbeat 연결. 실제 오늘 failed를 exit1로 검출.
- [x] 최종 전수3,000 PASS/421.80초, 독립 검토 차단급 결함 없음. Ruff·셸 검사 통과.
- [x] 수정본의 실제 CPU50% service 전수통과 확인 — 사용자 수동 실행3,000 PASS/878.74초, 종료exit0 및 native 상태passed 독립 확인.

### 실측·검증 기록

- 오늘 자동 테스트는2,913 PASS/2 FAIL/1059.91초. 실패는 6개 spawn의 join30초와4개 spawn의
  barrier10초로, 수치 결과 불일치 단언 전에 발생했다. CPU 사용496.88초/벽시간약1063초.
- `systemd-run --user` 임시 unit으로2개 PASS/3.44초였지만 CPU7.99초/벽시간5.26초여서50% 제한
  재현 증거가 아니다. system transient 재현은 `sudo -n ...`가 비밀번호 필요로 거부됐다.
- 시간 주입에서40초 준비가 옛10초 barrier에 실패하는 경로를 확인하고, 새 준비/작업 분리에서는
  통과함을 검증했다. 준비90초 초과·작업정체·자식 비정상종료·부분 spawn실패·terminate 무시도 검사한다.
- 준비 예산90초는 초기값이며 실제 서비스의 cold-start와 부하를 관측해 조정한다.
  자동 재시도·skip·xfail·worker 축소·fork 전환·CPU 제한 해제는 없다.
- 표적: 확률/영수증/상태/Telegram192 PASS, heartbeat 확장194 PASS 및 최종83 PASS,
  worker harness/CSV 원장/v2 전달70 PASS. 마지막 전수 결과는 아래 최종 검증에 기록한다.
- `ops.selftest_status`는8초 bounded query,heartbeat hook은15초(+TERM 후5초)다.
  최신 당일 정상 재실행만 상태를 회복시킨다. reset-failed나 기존 실패 지우기는 하지 않는다.
- 미래 R1 알림에만 `<1%`가 반영된다. 이미 보낸 메시지·원장·snapshot·receipt는 바꾸지 않는다.
  GitHub/Pages 게시·systemd 설치·실발송·모델 재학습은 이번 변경에 포함하지 않았다.

### 최종 검증·인계

```bash
PRELUDE_FORBID_TELEGRAM=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/home/soccz/22tb/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 venv/bin/python -B -m pytest -q -p no:cacheprovider
# 3000 passed in 421.80s (85개 신규, 실제 CPU50% 서비스 실행 결과가 아님)
ruff check --no-cache scripts/recommend_send.py ops/selftest_status.py tests/process_helpers.py tests/test_process_helpers.py tests/test_recommend_send_format.py tests/test_selftest_status.py tests/test_heartbeat_selftest.py tests/test_heartbeat_microstructure.py tests/test_ledger_csv_atomicity.py tests/test_pump_detector_v2_delivery.py
bash -n scripts/heartbeat.sh
```

작업 파일은 `scripts/recommend_send.py`, `scripts/heartbeat.sh`, `ops/selftest_status.py`,
`tests/process_helpers.py` 및 관련 회귀7파일, `OPS.md`/`PHASES.md`/`README.md`다.
새 helper2파일 최종 Black 정렬은 전후AST 동등으로 기능 불변을 확인했다.
정렬 후 관련12개 테스트 파일354 PASS/29.00초, Ruff·Black6파일·셸문법·diff-check 모두 통과했다.
14:25 KST 최종 읽기 검사에서도 오늘 두 R1 발송의 원래 서버수락 상태와 시각이 유지됐다.
기존 운영 output8개와 비공개 연구자료는 그대로 보존하고 commit/push하지 않는다.
초기 인계 때 실제 service 재검증은 sudo 비밀번호를 처리할 수 없어 남겨 두었다.
이후 사용자가 `sudo systemctl start prelude-selftest.service`를 실행했고,14:42:01 journal에
**3,000 PASS/878.74초(14분38초)**가 기록됐다.14:47 KST 독립 읽기 검사에서 다음을 확인했다.

- 실제 시작14:27:21→종료14:42:03, `Result=success`, `ExecMainCode=1`, `ExecMainStatus=0`,
  `ActiveState=inactive`/`SubState=dead`. oneshot의 정상 종료이며 계속 실행 중이어야 하는 서비스가 아니다.
- `CPUQuotaPerSecUSec=500ms` 유지, CPU사용385.959초. 실제 제한을 풀어 얻은 통과가 아니다.
- `venv/bin/python -B -m ops.selftest_status --format json`: exit0, `state=passed`,
  `attention_required=false`, 당일 시작/종료시각 일치. 실패 지우기나 재발송 없이 최신 성공으로 회복했다.
- journal 끝부분을 `--since '2026-09-10 14:40:00' --until '2026-09-10 14:43:00'`로 직접 조회해
  사용자 제공 로그와 전수통과가 일치함도 확인했다.

**이번 세 가지 수정의 구현·로컬 검증·실제 service 검증은 완료됐다. 추가 설치/수동 검사는 필요 없다.**
다음 정규 실행부터 새 표시·heartbeat 감시를 사용한다. 한 번의 전수통과가 미래 무장애나 추천 성능을
보장하지 않으므로 매일 검사는 계속한다.

### 후속 공개 준비·PIN 유지 결정 (2026-09-10)

사용자의 후속 `마무리 다 하고 말해` 요청으로 GitHub·개발일지 Pages 공개를 준비했다.
코드14파일만 stage했고 기존 운영 output8개와 비공개 자료는 제외했다. 별도 Pages clone
`_workspace/pages_release_20260910`의 `projects/prelude/index.html`에 실패→수리→실제 서버
재검증 흐름을 추가했다. 기존 실패 기록은 유지하며 추천 성능 개선과 운영 정상화를 구분했다.

- 독립 검토: staged allowlist14개 일치, 본문과 새 HTML의 실제 runtime 비밀값 직접일치0건.
- HTML·내부 링크32개·중복 ID·inline JS·diff 검사 통과. PC1440/모바일390 화면을 확인했고
  가로 넘침·미처리 JS 예외가 없었다. 대시보드 검사는 가짜 상태만 사용했다.
  브라우저는 HTTP 확인 본문을 격리 렌더링했고 외부 글꼴 대신 로컬 대체 글꼴을 사용했다.
- 현재 공개 암호화5파일은09-10 동일 생성 세대·native 인증/출처 검증을 통과하고 공개본과
  SHA가 일치했다. 이는 데이터 형식/일관성 검사이지 아래 노출된 키의 기밀성 보장이 아니다.
- 공개 전 검사에서 과거 공개 문서의 PIN 리터럴이 현재 runtime 값과 같음을 발견했다.
  현재 PHASES 본문의 리터럴은 제거했으나 과거 Git 기록에도 있어 삭제만으로 비밀이 되지 않는다.
- **사용자 결정:** 노출 사실과 과거 Git 기록의 한계를 안내한 뒤 사용자가 현재 문서에서
  지울 수 있는 값만 지우고 PIN은 유지하라고 명시했다. 이 범위로 공개 준비를 재개한다.
  `.env`·PIN·암호화5파일·과거 Git 이력은 변경하지 않는다. 기밀성이 회복됐다고 주장하지 않는다.
  값 자체는 로그·채팅·새 문서에 기록하지 않는다.
- 추가 current-tree 감사에서 테스트 fixture2곳도 같은 문자열을 쓰고 있어 별도의 합성4자리
  값으로 교체했다. `test_dashboard_passphrase_accepts_four_digit_pin`의4자리 허용 검증은
  그대로다. 실제 `.env` 전후 byte 동일, 공개 범위395파일의 runtime 비밀값 일치0건을 확인했다.

### 공개 반영 결과 (2026-09-10)

- 코드 `192f182`:14파일을 정상 push했다. clean-worktree pre-push에서 변경 Python10파일
  Ruff와 전수 **3,000 PASS/465.88초**를 통과했다. 로컬 dirty 산출물에 의존하지 않는 커밋 검사다.
- 소개 `4b82349`: `projects/prelude/index.html`1파일만 정상 push했다.
  [Pages run34444170422](https://github.com/soccz/soccz.github.io/actions/runs/34444170422)가
  **15:12:32 KST success**로 끝났다. 사이트 공유 worktree·다른 프로젝트는 수정하지 않았다.
- `_workspace/release_checks_20260910/verify_published.py`로 공개 소개·대시보드 HTML2개와
  암호화 JSON5개의HTTP200·SHA 일치,09-10 동일 생성 세대, native 인증/출처 검증을 통과했다.
  기존 PIN과5파일은 재생성/재암호화하지 않았다. 개인 NOTES 제외·자동 주문/승격 없음도 유지한다.
- 테스트 fixture 후속 정리는 해당 파일 **9 PASS/1.85초**, Ruff 통과 후 이 공개 결과 기록과
  함께 별도 커밋으로 반영한다. 후속 커밋도 기존 clean-worktree 전수 pre-push를 생략하지 않는다.
- 15:16 KST native selftest 읽기 검사도 `passed`/`attention_required=false`/exit0였다.
  실제 서비스 재시작·추론·재발송·재학습·새 systemd 설치는 이번 공개 작업에서 실행하지 않았다.
- **운영 수리와 공개 반영을 마무리했으며 추가 사용자 설치는 없다.** 과거 Git 기록은 사용자
  결정대로 유지하고, 추천 우위 검증은 기존 forward 기록 절차를 계속 따른다.

---

## 실사용 강화 12차 — 개발일지·현재 대시보드·GitHub 공개 반영 (2026-09-09)

사용자가 공개 소개를 아이디어→가설→시도→검증→성공/실패→다음 질문이 이어지는 개발일지로
유지하라고 명시하고 `ㅇㅇ 좋아 가자`로 구현·검증·GitHub/Pages 반영을 승인했다.
실제 추천·동결 trial/라벨·모델 소스와 오늘 실패 원본은 변경하지 않는다.

### 구현 전 설계·완료 조건

1. 신규 trial/운영 상태의 Git 제외 누락을 보완하고 공개 코드·테스트·문서만 선별한다.
   기존 dirty 변경은 이전 단계의 사용자 작업으로 보존한다. 원시자료·NOTES·키·개인 발송 영수증은 공개하지 않는다.
2. publish의 상속 잠금 검증 실패 분기를 RED→수리→회귀 검증한다. 평문 게시나 잠금 우회는 허용하지 않는다.
3. 기존 암호화5파일 계약을 유지하면서 summary 안에 버전 있는 현재 상태 projection을 추가한다.
   R1 preopen/open 전달과 Top10 record-only를 분리하고, 원본 보고서를 그대로 내보내지 않는다.
   새 연구의 대기/실패는 표시하되 기존 추천을 성공으로 위장하거나 미계산 성과를0으로 채우지 않는다.
4. Pages는 prelude 안의 제외된 별도 clone에서 작업한다. 기존 소개 이력은 보존하고9월의 후속 사고/검증/
   체결 정보 가설로 연결한다. 과거 R2/v2 성과를 현재 R1 우위로 표현하지 않는다. UI는 기존 스타일을 재사용한다.
5. 코드·데이터 계약·페이지를 각각 독립 검토한다. targeted/full pytest, Ruff, 셸/HTML/JS 검사,
   데스크톱·모바일 실화면/링크/대기·결측·오류 상태를 확인한다. 사용자의 PIN 값은 출력·저장하지 않는다.
6. 코드 repo는 선택한 파일만 commit/push하고 기존 pre-push의 커밋 단위 검사를 통과한다.
   Pages는 호환 가능한 화면을 먼저 배포하고 검증된 암호화 데이터를 게시한다. 원격 Actions/HTTP/세대 일치까지 확인한다.
   사이트 공유 작업 폴더·타 프로젝트·Git 이력은 덮어쓰지 않는다. 권한이 필요한 단계는 도구 승인 절차를 따른다.

- [x] 공개 범위·잠금 실패 처리 수리 및 표적 검증. 잘못된 상속 FD4종 RED→16 PASS.
- [x] 현재 상태 데이터 계약·생성기·검증기·대시보드 연결. 개인 NOTES 생성 경로 제거.
- [x] 개발일지 후속 이야기·GitHub 설명·설치 완료 문서 정정. 원격 About/홈 설명까지 확인.
- [x] 독립 검토·전수/실화면 검증. **2,915 PASS/436.77초**; 최종 UI CSS 후 양화면 재검사 PASS.
- [x] 선별 코드 push·Pages/암호화 데이터 게시·원격 확인. 공유 Pages 작업 폴더 수정 없음.

### 독립 검토·수리와 공개 경계

- 기존 상속 잠금 오류4종이 후속 Python/Git/게시까지 진행하던 RED를 확인하고 즉시 종료로 수리했다.
  관련16 PASS는 잘못된FD에서 빌드·clone·실패알림·원격변경이 없는지도 확인한다.
- `summary.current_system`은 실제 R1 두슬롯/두 연구의 운영 관측만 allowlist로 투영한다.
  연구 probe가 outcomes를 검사하지 않으므로 누적 표본 수·수익·우위 판정을 여기서 새로 만들지 않는다.
  projection 추가에 native 동결 generator/score/eval 소스 수정은 필요하지 않았다.
- 생성자와 다른 검토자가 슬롯 하나의 잘못된 응답이 전체 표시를 막는 문제와 상태/주의/종료코드 모순을
  발견했다. 새 모의33건 RED 후 슬롯별 완전 검증→대입, 전날 주의 집계, asof/초 단위 시각 계약을 수리했다.
  새59개 포함 관련170 PASS/43.17초, native 연구 상태90 PASS, 공개 대상 Python Ruff PASS.
- 실제 읽기 생성 결과는09-09 R1 preopen/open 각각 `delivered_candidates`, 동점 연구 `evidence_invalid`,
  Top10 `not_started`였다. 동결12파일과 오늘 manifest/raw2개/open snapshot의 SHA가 시작 시점과 같다.
- 소개는 과거Journey·Failures Wall 원문과 기존앵커16개를 보존했다. 신규9월 장은 모델비교미채택→새정보→
  첫정규96ns실패→09-10기록시험으로 연결한다. 두 검토자가 cutoff와 점수 내구저장 증명 범위를 재교정했다.
  과거 R2/~6배 수치를 현재 R1 성능으로 올리지 않으며, 대시보드의 기존 수익 분모는 장후 R1 원장으로 명시한다.
- HTML구조/중복ID/내부링크45개/inlineJS 문법, 대시보드 가짜payload26검사와 PIN/crypto 원본 byte동등을 확인했다.
  로컬 Chrome에서는 HTTP로 확인한 HTML·원래CSS/JS·원래Chart.js를 격리문서에서 렌더했다.
  이 호스트의 headless 직접HTTP/외부폰트 로딩 정지 때문에 설치 한글 fallback폰트를 사용했다.
  데스크톱1440/모바일390의 현재·결측·불확실·과거fallback, 가로넘침 없음/uncaught0을 확인했다.
  실제 PIN 입력/개인 데이터 복호화 screenshot은 없다. 원격 배포의 HTTP/Actions 검사는 별도 완료 조건이다.
- 공개 allowlist는 코드·테스트·핵심문서107파일이다. tracked 운영 output8개와 비공개 실험 중간자료는
  disk에 보존하고 stage에서 제외한다. 기존 Pages 공유 폴더는 수정하지 않는다.
- 최종 전수2,915 PASS/436.77초, 공개Python Ruff/셸문법/Pages diff-check PASS.
  전체선별 stage 후 처음 드러난 기존 미추적 `signals/recommend_calibration_eval.py`와 해당 테스트의
  EOF 여분 빈줄2건은 형식 경고로 남긴다. 완료된 연구의 출처 소스를 공개 작업에서 재작성하지 않는다.
  데스크톱 current 카드는2열, 모바일1열로 조정해 긴 상태의 읽기 밀도를 낮췄고 신규앵커에80px 여백을 뒀다.
  원격 반영은 별도 선별 commit과 기존 정상 pre-push 게이트로 검증하며 결과를 다음 기록에 남긴다.

검증 명령: `PRELUDE_FORBID_TELEGRAM=1 venv/bin/python -B -m pytest -q -x -p no:cacheprovider`
(TMPDIR은허용임시경로, BLAS/OMP단일스레드, bytecode미작성); 선택Python `ruff check`,
`bash -n scripts/publish_dashboard.sh`, Pages inline script `node --check`, 격리Chrome CDP 가짜상태검사.

### 실제 게시 결과 (2026-09-09)

- 코드 `38fa217`:107파일 선별 commit/push 완료. 정상 pre-push가 **비공개 자료 없는 별도 HEAD**에서
  변경Python93개 Ruff와 전수 **2,915 PASS/431.07초**를 다시 확인했다. gate 우회/force push는 없다.
- Pages 화면 `01e131bd1`:홈 prelude설명/공개 소개/대시보드 HTML3파일만 반영했다.
  [정적 배포 성공](https://github.com/soccz/soccz.github.io/actions/runs/34316875080)을 확인했고,
  실제 HTTP의3 HTML SHA가 검증한 clone과 정확히 일치했다.
- 기존 publisher를 `PRELUDE_FORBID_TELEGRAM=1`로 수동1회 실행해 새 암호화 데이터를 게시했다.
  `9b57aeee4`,14:58:53 완료;[데이터 배포 성공](https://github.com/soccz/soccz.github.io/actions/runs/34317070317).
  이번 게시 준비가 갱신한 policy/idea 운영 산출물은 코드 commit에서 제외했다.
- 원격 암호화5파일을 다시 받아 HMAC·스키마·현재 출처·동일세대 검증과 HTTP바이트 일치를 확인했다.
  세대 `167a3618-4009-4886-85d3-43fa83d5b380`,운영 확인시각09-09 14:58:29 KST.
  R1 장전/장후 전달 확인, 동점시험 증거불일치, Top10 시작전을 그대로 표시한다. NOTES 제외도 검증했다.
  PIN은 출력/게시하지 않았고 개인 가상 원장의 평문이나 비공개 중간자료를 public source에 추가하지 않았다.
- GitHub About은 현재 R1/발송기반평가/별도시험/자동주문없음으로 갱신했고 homepage는 소개 URL로 연결했다.
  공유 Pages worktree는 후속 읽기에서도 clean이었다. 검증용 로컬 HTTP서버·임시Chrome만 종료했다.

**완료 경계:** 코드·공개 이야기·대시보드 반영 완료. 새 추천우위/09-10 첫정규수집 성공을 뜻하지 않는다.
대시보드 데이터는 기존10:10 예약 게시가 다음날 갱신하며, 소개 개발일지는 실제 개발·판정이 바뀔 때
그 근거와 함께 후속 장으로 갱신한다. 이번 작업으로 새 학습/실알림/주문/라벨 규칙을 바꾸지 않았다.

## 실사용 강화 11차 — 첫 정규 실행 장애 수리·추천 경로 격리 (2026-09-09)

사용자 `쭉쯕 가라고 왜 멈춰 계속`에 따라 후속 구현을 계속한다. 설계 중 실제 날짜가09-09로 넘어가
첫 정규 실행 로그가 생겼으므로, 새 Top10 시험 연결보다 **실제 발생한 장애의 근본 수리**를 먼저 한다.
09-08의 테스트 PASS를09-09 실제 운영 성공으로 대체하지 않는다.

### 수정 전 설계·우선순위

1. 실제09-09 수집은08:45:02 시작,282종목·capture complete=True였으나09:09:02 피처 생성이
   `capture cutoff is not original decision completion`으로 실패했다. datetime→ns의 부동소수점 경로와
   정본의 정수 시각 경로를 독립 비교한다. 원본manifest/raw/snapshot은 그대로 보존하고 다음 실행 코드만 수리한다.
2. 정규07:30 selftest는 `test_payload_budget_exact_boundary_and_rejected_frame[2-orderbook]`1실패,
   나머지2530통과였다. 즉시join의 경쟁 조건이 테스트 전제인지 실제worker결함인지 재현 후 분리해 최소 수리한다.
   재시도로 실패를 숨기거나 검사를 제외하지 않는다.
3. selftest의 close→selftest→추천 순서 연결을 제거하고 전수검사·07:30 timer·실패 경보는 보존한다.
   운영DB 접근 방어는 이미 있으므로 낮은 CPU/I/O 우선순위·명시 자원상한을 사용한다.
   단순sleep 이연은 대기 중 재부팅 시 검사가 사라질 수 있어 채택하지 않는다.
   unit 변경은 별도 최소 설치 모드/복구 테스트를 거친다. `/etc` 실제 설치는 sudo 권한이 확인될 때만 수행한다.
4. 연구/평가 담당이 제안한 Top10 별도prospective시험은 기존tie기록 후, 누적평가 전에 저장하는 방향이다.
   원래선별기·tie실험·라벨·실제추천은 보존한다. 오늘오전이 이미 지났으므로 새시험의최초기대일은
   09-10보다 빠를 수 없다. 실제 장애 수리 및 실행 시간 여유 검증 후 연결 여부를 판단한다.

### 후속 단일 시험 연결의 구현 전 결정

실제 원본335,706행 감사는9.36초, 원100종목 모두300초 창에 체결이 있었고 순서/체크섬 오류는0이었다.
새단일Top10시험을 별도ID `r1_top10_trade_imbalance_v1`/root `output/recommend_trade_shortlist_trials`로
연결한다. 설계 고정일09-09·첫 기대일09-10이며 미세체결 실험군의추가1개/총2개다(프로젝트과거시도전체가2개인것은아님).
규칙K10/피처/결측처리/소액거래포화진단은 기존오프라인planner그대로 유지한다. 새모델/라벨/실제추천변경은없다.

- 기존tie score/commit → 새Top10 score/commit → 기존/신규 누적평가 순서로 배선한다.
  신규저장소 오류·신규시험 오류가 기존tie예측 저장/평가를 막지 않으며 오류는 최종nonzero로 전달한다.
  기존tie발행이 실패/불확실하면 신규시험을 대신 성공시켜 전체를정상화하지 않는다.
- native `_load_inputs`의 검증된 전체원본/소스확인과 원자저장을 재사용한다. raw중복해시 최적화용 새검증캐시
  프레임워크는 지금추가하지 않는다. 전체행감사9.36초 및오늘원snapshot완료09:08:44→진입봉09:15의여유는
  실행가능성의근거지만 신규publisher속도측정은아니다. 누적평가는항상두score저장뒤로둔다.
- 주효과는동일날짜원Top3대새Top3, 정상무교체포함/결측종목대체금지/원정본순비용일회차감이다.
  전체100코인baseline은공통완결보조코호트이며 변동성·유동성보정된우위라고과장하지않는다.
  기존라벨의canonical진입보다실제내구저장이엄격히빨라야prospective이며늦은점수는제외한다.
- 새namespace도백업목록과읽기전용누락감시에포함한다. 기존10차감시만으로새시험까지본다고주장하지않는다.
  아직실제적격예측/효과표본은0이다. 오늘실패원본을고치거나새시험의과거점수를소급작성하지않는다.

- [x] 정규 수집 시각 장애 RED→수리→독립 검산, 원본 증거 보존.
- [x] selftest 간헐 실패 재현과 최소 수리.
- [x] 추천 순서 격리·한정 설치/복구 검증. 당시 저장소만 완료; 이후 사용자 설치와12차 읽기 확인으로 `/etc` 적용 확인.
- [x] 후속 추천 시험 연결 설계의 검증 가능 범위 이행. 정규 시작09-10, 추천 우위는 미검증.
- [x] 관련/전수 검증·현 운영 읽기 점검, OPS/SIGNAL 후 README 마지막 갱신. **2,852 PASS/412.34초**.

### 구현·반증 결과와 실제 확인

- **수집 장애:** `datetime_to_ns`의 float 곱셈을 UTC 정수 timedelta 산식으로 대체했다.
  정답1788912524491612000ns, 원manifest1788912524491611904ns로 **−96ns**를 독립 재현했다.
  큰 정수는 jq 출력에서 반올림될 수 있어 Python JSON의 정수로 검산했다.
  정밀도66개를 포함한 관련216 PASS/13.39초, 전체335,706행의 checksum/순서/채널/행수 오류0.
  300초 창에서 원100종목 모두 체결 관측. decision_started cutoff 뒤 수신된769행은 제외하며
  이를 수집 종료 뒤 행이라고 부르지 않는다. 오류 구간96ns 안의 원시행은0이지만 원본을 수리하지 않았다.
- **selftest 경쟁 조건:** 즉시join이 stop_event를 세우기 때문에 수신 전 테스트 종료가 가능했다.
  지연된 수신 스레드로 기존4경우 모두 RED를 재현하고 테스트에서 실제 종료 이벤트를 기다리게 했다.
  production ChannelWorker/run_capture는 변경하지 않았다. collector70 PASS/10.71초,
  경계8경우×20회 **160/160 PASS**. 재시도·테스트 제외로 실패를 숨기지 않았다.
- **추천 순서 격리:** selftest Before/close After 제거, 낮은 CPU/I/O 우선순위와 초기 자원상한,
  전수검사/07:30timer/OnFailure는 유지했다. `--update-selftest`는 한 서비스만 교체한다.
  실행 중 상태 거부·reload 후 재확인·실패 rollback·타18파일/9timer 보존·신규 runtime6개 안전성까지
  최종 관련 **172 PASS/27.42초**. 상태 조회 전후의 관측은 외부 실행과 원자적으로 배타적이지 않다.
  실제 `sudo -n ... --update-selftest`는 **password is required, exit1**로 아무것도 설치하지 못했다.
  13:50 저장소 unitSHA26ec6326…와 설치본b1e14de9… 차이 재확인. 관리자 수동 적용 명령은 OPS에 남겼다.
- **단일 후속 시험:** publisher39·평가기39개 모의검사, 각 native 관련 묶음204/240 PASS.
  생성자와 다른 검토자가 실제publisher→receipt writer(fake transport)→canonical label writer→strict loader→eval을
  연결한 **독립11개 포함181 PASS/28.31초**, 추가 critical0. 소급 저장/선택 결과 대체/이중비용은 없다.
- **통합 실패 격리:** 신규32개 포함 wrapper67 PASS/4.41초. launch 전후·한국 날짜 경계,
  oldcommit→newcommit→old/new평가 순서, 새 저장공간/ImportError/불확실 저장/평가 실패를 확인했다.
  독립 반증에서 내일날짜로 기존 테스트1개가 잘못 실패함을 발견해 과거 경로는 `--asof09-08`로 고정했다.
  수정 후 wrapper 시계를09-10 07:30KST로 바꾼 격리 재실행도 **67 PASS/4.40초**였다(시스템 시계 변경 없음).
  **재발 방지:** 달력으로 새 경로를 켜는 테스트는 실제 오늘 날짜에 기대지 않고 launch 전/후 날짜를 각각 고정한다.
- **새 기록 감시·백업:** 독립 native통합44 PASS/14.83초. 평가 JSON을 재seal해도 당일선정/시각/현재소스
  모순을 차단하며 동일 크기 raw 변조도 새 probe에서 거부한다. 정상 no-op/변경/자료부족과 오류를 구분한다.
  기존 경량 probe까지 raw검사를 추가한 것은 아니다. heartbeat/backup 격리 회귀 **144 PASS/67.21초**.
- **실제13:49 읽기 점검:** R1 preopen08:53:24/open09:08:45 서버수락 증거 정상, 새 발송시도 없음.
  원 수집 probe는 exit1 `evidence_invalid`, 새 trial probe는 exit0 `not_started`였다.
  이전 실패가 여전히 표시되는 것은 원본 증거를 보존한 결과이며 수정 코드의 다음 실행 실패라는 뜻이 아니다.
- **최종 검증:** 동결 소스에서 전수 **2,852 PASS/412.34초(6분52초), exit0**. 10차2,531개보다321개 증가했다.
  변경 관련 Python21파일 Ruff PASS, heartbeat/backup/installer `bash -n` PASS, capture CLI `--help` PASS.
  13:57 실제 새 평가 CLI도 exit0 `not_started`, paired_dates0, automatic_promotionfalse였다.
  설치파일 읽기 대조18개일치/오직selftest1개상이. 독립 OPS/SIGNAL 설명 대조도 필수 정정0이었다.
  이후 변경은 완료 결과 문서뿐이며 최종 `git diff --check`로 확인한다. git commit/push는 하지 않는다.

최종 검증 명령(실제 수행):

```bash
TMPDIR=/home/soccz/22tb/tmp PYTHONDONTWRITEBYTECODE=1 PRELUDE_FORBID_TELEGRAM=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 venv/bin/python -B -m pytest -q -x -p no:cacheprovider tests
ruff check data/upbit_microstructure.py signals/recommend_trade_shortlist_trial.py signals/recommend_trade_shortlist_eval.py ops/recommend_trade_shortlist_status.py scripts/capture_recommend_microstructure.py scripts/evaluate_recommend_trade_shortlist_trial.py tests/test_microstructure_time_precision.py tests/test_collector_upbit_microstructure.py tests/test_selftest_isolation.py tests/test_selftest_update_install.py tests/test_selftest_update_independent.py tests/test_recommend_trade_shortlist_trial.py tests/test_recommend_trade_shortlist_eval.py tests/test_recommend_trade_shortlist_status.py tests/test_trade_shortlist_independent.py tests/test_capture_recommend_microstructure.py tests/test_capture_trade_shortlist.py tests/test_heartbeat_microstructure.py tests/test_heartbeat_trade_shortlist.py tests/test_shell_failure_propagation.py tests/test_systemd_deploy_contract.py
bash -n scripts/heartbeat.sh scripts/backup_db.sh deploy/install_systemd.sh
PRELUDE_FORBID_TELEGRAM=1 PYTHONDONTWRITEBYTECODE=1 venv/bin/python -B scripts/capture_recommend_microstructure.py --help
PRELUDE_FORBID_TELEGRAM=1 PYTHONDONTWRITEBYTECODE=1 venv/bin/python -B scripts/report_recommendation_status.py
PRELUDE_FORBID_TELEGRAM=1 PYTHONDONTWRITEBYTECODE=1 venv/bin/python -B -m ops.recommend_microstructure_status --format text
PRELUDE_FORBID_TELEGRAM=1 PYTHONDONTWRITEBYTECODE=1 venv/bin/python -B -m ops.recommend_trade_shortlist_status --format text
PRELUDE_FORBID_TELEGRAM=1 PYTHONDONTWRITEBYTECODE=1 venv/bin/python -B scripts/evaluate_recommend_trade_shortlist_trial.py
```

원수집 probe의 exit1은 위에 명시한 실제 실패 증거이며 통과로 바꾸지 않았다.

원본 불변 SHA(13:49 재검산, 처음 독립 감사와 일치):

| 증거 | SHA-256 |
|---|---|
| 09-09 capture manifest | `aacadeb95e86de0dab31421ec0a961898aeb6cfbd5837e0941d6a03d31164fdf` |
| trade raw | `6683c7c4bb4a4447d6b7111e14ced022c2ffe8a671758a4ebd5354673101bc94` |
| orderbook raw | `3ac113c6e7cc86c1a301e8e46c0bf7c8feac0ee9f327566d30648dfebe84e7e3` |
| open R1 snapshot | `aef008f058d174d95b64d705bbde254926d5c237b735739282cc99825f52db6b` |

### 완료 경계·다음 관측

지금 수정 가능한 결함 수리와 별도 단일 시험의 예약 경로 연결은 구현했다. 목표인 저하방·고상방
추천 우위는 아직 결론을 낼 표본이 없으며, 자동 학습/실제 순위 승격/주문은 하지 않는다.
09-10 정규 실행에서 cutoff 정확성→양쪽 score/commit의 진입 전 저장→평가/heartbeat→다음 백업을
실제 증거로 확인해야 한다. 이 검사는 서버의 기존 스케줄이 실행하며 에이전트 재방문 예약은 만들지 않았다.
현재소스 고정 검증·raw 중복해시 비용·누적평가 성장·작은 거래1건의 포화는 후속 관찰 대상이다.
새 시험의100종목 기준선은 유동성/변동성 매칭된 우위가 아니며, 반복 확인의 탐색적 CI를 채택 보장으로 쓰지 않는다.
외부 감시/다른 물리 저장소 백업은 여전히 대상과 접근 권한이 필요하다. 그 목적지를 임의로 정하거나 전송하지 않았다.

이번 파일: 정밀도 helper/collector 테스트, selftest unit·installer·관련 테스트, 신규 Top10 publisher/eval/CLI,
capture wrapper·신규 status·heartbeat·backup·각 테스트, 설계JSON과 PHASES/OPS/SIGNAL/마지막README.
기존 native feature/tie trial/순수Top10 planner/경량probe와 actual R1 send/model registry는 이번 단계에서 바꾸지 않았다.
NOTES·DB·원장·실패capture·snapshot·receipt·과거 시험/보고서를 수정하지 않는다.

## 실사용 강화 10차 — 수집 미실행 감시 공백 수리 (2026-09-08)

사용자 요청: 개선안을 설계·추론·검증하고 객관적으로 확인한 범위까지 마무리한다.
이번 완료 목표는 기존10:30 heartbeat가 새 수집기의 **미실행/중간 실패/불완전 저장**을 놓치지 않게 하는 것이다.
기존 R1·동점 실험·오프라인 Top10 선별기·라벨·연구 달력·원본 증거는 변경하지 않는다.

### 수정 전 설계와 완료 기준

1. `ops/recommend_microstructure_status.py`에 읽기 전용 일일 상태 확인을 추가한다.
   운영상 최초 기대일09-09와 연구 시작일09-08을 분리한다. 당일10:00 KST 전은 대기:
   08:45 예약+2700초 실행+90초 종료에 여유를 둔 운영 초기값으로 실제 지연 관측 후 조정한다.
2. 당일 원본/manifest → 원본 R1 snapshot → feature → score/commit → 평가 파일을 확인한다.
   미실행, 미완결, 품질 부적격, 발행 불확실, 정상 무교체를 구별한다. 파일이 있다고 성공으로 판정하지 않는다.
   작은 문서/연결의 무결성과 원본 파일 존재·크기를 검사하되 대용량 원본 재해시/성과 재평가는 하지 않는다.
   그러므로 운영 점검 통과를 연구 적격·추천 우위의 증명으로 표현하지 않는다.
3. `scripts/heartbeat.sh`에30초+종료5초 상한의 점검을 연결하고 기존 WARN/경보 전달 방식을 재사용한다.
   새 timer·자동 복구·재수집·발송 형식 변경·sudo 설치는 없다.
4. 모의 정상/장애/시각 경계·CLI·셸 통합 테스트, 독립 반증, Ruff/셸 문법, 관련 회귀를 통과한 범위만 완료한다.
   실제 읽기 전용 실행으로09-08은 설치 전 기대 제외임을 확인한다. 다음 정규 실행 성공은 미리 주장하지 않는다.

- [x] 설계 반증과 구현. 기존 heartbeat12행 연결, 읽기 전용 probe와 신규 테스트3개 파일 추가.
- [x] 장애 주입·독립 검토·회귀 및 실제 읽기 확인. 최종 **2,531 PASS/354.29초**.
- [x] OPS/PHASES 기록 후 README 마지막 갱신. 이번 운영 보완 완료, 첫 정규 실행/추천 우위는 미검증.

### 객관화 과정과 검증 경계

- 독립 설계 반증으로 처음 설계에 **당일 중복 capture 경고**, **score/commit은 있어도 평가 파일 누락 분리**,
  **정상 무교체와 관측 결측 unavailable 분리**를 반영했다. 성공본을 임의 선택하거나 결측을0으로 대체하지 않는다.
- 늦은 부팅 시 오늘waiting만 표시하면 전날 미실행을 놓치므로, 기한 전에는 직전 기한이 지난 날짜도 검사한다.
  실제 관측시각은 그대로 유지하고 판정 대상 날짜만 바꾼다. 전일 문제는 오늘waiting이어도 exit1이다.
  임의의 전체 과거를 스캔하는 기능은 아니며 누적 연구 달력 검사는 기존 평가기에 남긴다.
- 작은 문서 간 SHA 연결뿐 아니라 native 입력 그래프의 경로/항목수/형식과
  `snapshot 완료 ≤ capture 종료 ≤ score 계획 ≤ durable 관측 ≤ 평가 생성 ≤ 현재`를 검사한다.
  독립 테스트는 위조 문서를 수정한 뒤 **후속 checksum까지 모두 다시 맞춘 상태**에서 의미상 모순 거부를 확인했다.
- 실제 형식의 모의 산출물을 생성한 신규 단위46개와 셸 통합17개 **63 PASS/26.25초**.
  독립 고장주입11개 **11 PASS/5.34초**, 원래46개와 함께 **57 PASS/18.21초**.
  기존 feature/trial/capture/shell/systemd 관련 **389 PASS/96.11초**. 신규 Python4개 Ruff 및 heartbeat 셸 문법 PASS.
  최종 동결 소스에서 전수 **2,531 PASS/354.29초(5분54초), exit0**. 이전2,457개에 신규74개가 추가됐다.
  이후 수정은 PHASES/README 기록뿐이며 `git diff --check`로 마지막 확인한다.
- 독립 운영 감사도 PASS: 실행코드/OPS 설명/상태 및 exit/경보 경로가 일치했고 설치19파일 SHA가 저장소와 모두 같았다.
  설치파일·timer·서비스를 변경/재시작하지 않았다. 수동 heartbeat·Telegram 발송·공개 수집도 실행하지 않았다.
- 09-08 17:10 실제 읽기 확인: 새 CLI exit0 `not_started`, 운영 시작09-09·연구 시작09-08을 각각 표시.
  microstructure timer는active/waiting, LastTrigger 없음, 다음09-09 08:45였다.
  17:13 기존 추천 상태도 두 슬롯 snapshot/receipt/intent 정상, `attention_required=false`였다.
  전달 확인은 Telegram 서버 수락 증거이며 사용자 읽음 확인이 아니다.
- 기존8개 핵심 소스 해시(feature/trial/Top10 planner/capture wrapper/close2개/send/model registry)는 이전 단계와 동일했다.
  새 probe SHA `8e0be20d80537878af59e6589c341e1c38bed18862ca3d5eb837c85415090b6a`,
  heartbeat SHA `990ca7413ccead6fb356c107f04ce33982e9446c19db03657990c57aae468edf`,
  독립 테스트 SHA `1f38494fb9bf72a109280705efdb2fb2f256b7aa427f6e86b6c48848f6c1bc77`.

검증 명령(실제 수행, 모든 pytest는 격리된 임시 자료/발송 금지 환경):

```bash
TMPDIR=/home/soccz/22tb/tmp PYTHONDONTWRITEBYTECODE=1 PRELUDE_FORBID_TELEGRAM=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 venv/bin/python -B -m pytest -q -x -p no:cacheprovider tests/test_recommend_microstructure_status.py tests/test_heartbeat_microstructure.py
TMPDIR=/home/soccz/22tb/tmp PYTHONDONTWRITEBYTECODE=1 PRELUDE_FORBID_TELEGRAM=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 venv/bin/python -B -m pytest -q -x -p no:cacheprovider tests
ruff check ops/recommend_microstructure_status.py tests/test_recommend_microstructure_status.py tests/test_heartbeat_microstructure.py tests/test_microstructure_status_independent.py
bash -n scripts/heartbeat.sh
PRELUDE_FORBID_TELEGRAM=1 PYTHONDONTWRITEBYTECODE=1 venv/bin/python -B -m ops.recommend_microstructure_status
PRELUDE_FORBID_TELEGRAM=1 PYTHONDONTWRITEBYTECODE=1 venv/bin/python -B scripts/report_recommendation_status.py
```

### 완료로 부르지 않는 것과 인계

목표인 저하방·고상방 추천 우위는 이번 운영 점검으로 증명할 수 없다. 첫 정규 feature/시험/백업 성공은
09-09 이후 실제 자료로 확인해야 하며, Top10 시제품은 여전히 오프라인이고 자동 학습·배포·실제 순위 교체는 없다.
이 점검은 **메타데이터 저장 상태**만 본다. 원본 내용/현재 generator 해시 전체·canonical 진입 전 적격·수익성은
미검사라고 결과에 명시한다. 동일 크기 raw 내용 변조가 통과하는 한계도 독립 테스트에 남겼다.

다음 안전한 작업은 실제 예약 실행 후 새 CLI/정본 평가기로 수집·저장·가용성·백업을 읽기 확인하는 것이다.
별도 자동 방문/주간 검토 예약을 만들지는 않았다. 서버 자체가 꺼진 경우의 외부 감시와 다른 물리 장치 백업은
대상/접근 권한이 필요하다. 기존 selftest의 `Before`에 따른 늦은 부팅 지연은 별도 unit 설계·설치 검증 대상으로 남긴다.
이번 변경 파일: `ops/recommend_microstructure_status.py`, `scripts/heartbeat.sh`, 신규 테스트3개,
`PHASES.md`, `OPS.md`, 마지막 `README.md`. 기존 연구 산출물·NOTES·모델·원장·DB는 수정하지 않는다.

## 실사용 강화 9차 — 다음 선별안 사전 검증·운영 잔여 보완 (2026-09-08)

사용자 `그치 당연하지 쭉쭉 가자`는 다음 선별 방식의 설계·사전 검증과 운영 취약점 보완을
계속하라는 요청이다. 기존 R1·8차 단일 동점 실험·사전 저장 증거는 보존한다.
새 architecture/라벨·자동 학습·실제 알림 순위 교체·외부 서비스 가입·systemd 재설치는 포함하지 않는다.

### 수정 전 작업 계획과 완료 기준

1. 연구 담당은 새 체결 정보를 쓰는 **단일 비동점 Top3 선별안**의 정확한 선정 규칙과 필요 자료를 제안한다.
   독립 평가 담당이 기존 실패 실험 재탕·시점 누수·거래 한 건의 imbalance 극단값·유동성/ATR 교락을 반증한다.
   설계가 성립할 때만 별도 오프라인 순수 planner와 모의/기존 snapshot의 결과 비열람 검사를 구현한다.
   새 정규 자료가 없는 상태에서 수익 우위·학습 완료·운영 채택을 주장하지 않는다.
2. 운영 담당은 close/publish 연구 오류가 핵심 결과 갱신에 끼치는 영향을 재현하고 최소 범위로 분리한다.
   연구 실패를 정상 처리하거나 오래된 연구 결과를 최신으로 게시하는 처방은 금지한다.
   대상 shell 경로와 고장주입 테스트를 먼저 합의한 후 수정한다. 알림 문구/정책은 유지한다.
3. 메인은 수집 wrapper의 **수집 전 디스크 여유 검사**를 설계·시험하고 기존 실제 백업을 읽기 검증한다.
   초기 여유량은 원본 payload 한도에 여유 계수를 더한 보수적 운영 초기값이며 실제 디스크 quota가 아니다.
   증거 자동 삭제나 원본 DB 복원 덮어쓰기는 하지 않는다.
4. 표적 pytest·독립 검산·Ruff/셸 문법·관련 회귀 및 필요 시 전수 테스트를 통과한 범위만 완료로 기록한다.
   문서는 PHASES/OPS/SIGNAL에 기록하고 README를 마지막에 갱신한다. 새 MD는 추가하지 않는다.

- [x] 단일 다음 선별안 제안 → 독립 사전 검증 → 허용 범위 구현/검사.
- [x] 운영 결합 재현 → 최소 수리 → 실패·정상 경로 회귀.
- [x] 수집 전 저장공간 검사 및 실제 기존 백업 읽기 검증.
- [x] 통합 검증·잔여 경계와 다음 단계 기록. 최종 **2,457 PASS/431.47초**.

### 이번에 실제 만든 것과 반증 결과

**선별기:** `signals/recommend_trade_shortlist.py::plan_trade_shortlist`는 원 R1 상위10개를
먼저 고정하고300초 체결 불균형으로3개를 고르는 **오프라인 프로토타입**이다. 동일 점수 조건이 없으므로
8~10위도3개 모두 들어올 수 있으나11위 이하는 불가하다. 원100개·Top3는 보존한다.
상위10개 중 관측 결측이 하나라도 있으면 날짜 전체 unavailable, 나머지90개 무체결 null은 글로벌 품질
유효 시 허용한다. 거래량/변동성 필터·가중치 sweep·자동 학습·새 실시간 시험 배선은 없다.
10은100의 상위10%라는 단일 초기값이며 결과로 최적화한 값이 아니다.

- 새 코드30개 테스트 PASS. 최종 소스의 기존 feature/trial 연결까지 독립 **165 PASS/17.30초**.
  독립 평가자는 산식 직접 재정렬, outcome 주입 불변, 파일 I/O 차단에서도 순수 실행됨을 확인했다.
- **소액 거래 한 건의+1 포화**가 큰 양방향 거래보다 앞설 수 있음을 테스트로 명시했다.
  그 결함을 숨기는 임의 보정 대신 체결 건수·대금·포화·저장 ATR/유동성을 진단으로 드러낸다.
  따라서 구조 검증 PASS는 추천 우위나 하방 보호의 PASS가 아니다.
- 새 정보는 기존 동점 실험과 같은 피처이며, 이번 차이는 비동점 후보까지의 선정 범위다.
  기존 veto/utility 계열과 구조적 독창성을 주장하지 않으며 새 일봉 피처·실패한 학습 축을 재시도하지 않는다.
- 원본 snapshot37일(07-28~09-08) 모두100후보/상위10 구조가 있었다. 과거 동점 기회는7일.
  이 구조 검사에는 성과/라벨/DB를 읽지 않았다. **실제 적격 feature0개이므로 새 교체율·성과는 미측정**이다.
- 입력 계약은 trusted native dict이며 파일 graph/raw/소스 해시 재확인은 호출자의 책임이다.
  새 publisher·score/commit·미래 평가기·스케줄은 구현하지 않았다. 과거에 만든 오프라인 순위는 forward 증거가 아니다.
- 설계/연구 노트: `_workspace/recommend_trade_shortlist_design_20260908_v1.json`,
  SHA `c1224c742444df3f2c2c6c5a8f93ecb7bbbe41720c6ab88a32b5bc89c95e7ac9`.

**운영:** 두 `daily_close_*.sh`에서 R1 원장을 legacy 원장보다 먼저 처리하고,
preopen 전체 후보 라벨도 연구 평가보다 앞에 둔다. 연구 명령만 개별300초/TERM 후10초KILL로 제한한다.
기존 실패코드·알림·success marker/publish 차단은 보존한다. report 실패 시 HTML 재생성도 계속 금지한다.

- 초기120초 설계는 모의 테스트를 통과했지만 메인의 실제 로그 검토에서 반려했다.
  `cron_preopen_close_20260908.log` 평가 첫 로그10:10:41.806 → 다음 명령 첫 로그10:12:32.434로
  정상 평가가 약110.6초였다. 초기값을300초(관측값 약2.7배)로 보강하고 전용22개 테스트 재검증 PASS.
  이는 전체 실행시간의 정밀 프로파일/미래 부하 보증은 아니며 자료 증가에 따라 다시 확인해야 한다.
- 최초120초 버전의 관련 회귀247 PASS와 최종300초 버전의22 PASS를 혼동하지 않는다.
  아래 전수 결과는 최종300초 소스를 대상으로 한다.
- 실제 GNU timeout으로 TERM 정상 처리124·TERM 무시137·첫 핵심 오류 유지·후속 작업 계속을 검증했다.
  실제 pipeline runner를 격리 fixture에서 통과시켜 연구 실패 때 완료 marker/게시가 차단됨도 확인했다.
- 완전한 연구/게시 분리나 전체 lock·selftest 지연 부팅 의존 제거는 아니다. 실제 일일 스크립트를
  수동 재실행하지 않았고 핵심 수집/legacy/원장/라벨에 새 제한을 적용하지 않았다.

**저장공간·백업:** 수집 wrapper에 raw와 trial 두 경로의 일반 사용자 가용공간을 검사하는 작은 함수를 추가했다.
시장 조회 전 및 직접 session의 수집 직전에 확인하고 부족/조회 실패는 새 연구 수집만 중단한다.
초기5GiB=`1GiB + 4 × 2 × 512MiB`는 보수적 진입 검사이며 예약·quota·실행 중 공간 보장이 아니다.

- 테스트를 먼저 추가해 기능 부재 실패를 확인한 뒤 구현했다.33 PASS 후 독립 검토가 추가한
  두 volume 불일치·시장조회 후 공간감소 반증도 영구 회귀에 넣어 **35개**로 보강했다.
- 두 독립 담당자의 코드 검토 PASS. 현재 가용20,141,994,115,072 bytes로 초기5GiB 기준을 통과했다.
  경로 생성·증거 삭제·다른 디스크 쓰기는 없다.
- 오늘 실제 archive1081개 member의 중복 부재/manifest/checksum/SHA 일치,
 09-07 open snapshot/receipt의 복원 가능한 JSON 내용과 현재 immutable 파일 바이트 일치를 확인했다.
  archive SHA `c8949540fc5c216910576daa4bf003132b4be89daeac59b8989606dd104ed396`.
  일봉(35,639,296 bytes)과15분봉(1,718,255,616 bytes) 실제 SQLite 보관본도 checksum 및
  `sqlite3 -readonly ... 'PRAGMA integrity_check;'`에서 모두 `ok`였다. 원본/백업을 덮어쓰지 않았다.
- **백업과 원본은 동일 파일시스템**임을 확인했다. 논리 복구점이지 디스크 고장을 버티는 독립 사본이 아니다.
  새 정규 수집 전 archive에는 microstructure 빈 디렉터리만 있고 실제 trial은0개다.
  전체 복원 훈련·새 trial 백업 성공·외부 저장소 확보를 완료했다고 하지 않는다.

### 보존 확인과 인계 경계

- 최종300초 shell·저장공간35개·새 선별기30개를 포함한 전수 **2,457 passed in431.47s**.
  변경 Python5개 Ruff, 두 shell/installer/backup 셸 문법, diff-check 모두 PASS.
  설계/구현 독립 검산과 `verify`의 실패 재현→수정→회귀 절차가120초 제한 정정과 저장공간 경계 보강에 반영됐다.
- 기존7차의 발송 intent/추천 전송/정본 평가/상태 조회/모델 레지스트리5개 SHA 불변,
 8차 feature/trial2개 SHA 불변. 기존 연구 산출물·NOTES·사용자 dirty 변경을 보존했다.
- 설치unit19개의 저장소/실제 설치파일 바이트 일치. 새 timer는09-09 08:45 대기 상태 그대로다.
  service 시작/재시작·systemd 변경·실제 알림 발송·새 학습·커밋/푸시는 하지 않았다.
- 16:33 읽기 전용 상태 CLI에서09-08 preopen/open 모두 snapshot/receipt/intent 유효,
  각3종목 server_accepted·attention_required=false를 재확인했다. 사용자 읽음이나 다음 실행 성공은 보장하지 않는다.
- 코드 변경 범위: 새 shortlist 모듈/테스트/설계 JSON, 기존 capture wrapper/테스트,
  두 daily close shell/새 회귀 테스트, PHASES/OPS/SIGNAL/README.
- 다음 실증은 첫 정규 capture/feature/기존 동점 trial과 보완된 close 실행이다.
  비동점 프로토타입의 정규 자료 진단·독립 효과 검증·실시간 기록/운영 채택은 별도 후속 단계이며
  이번 기계적 구현 PASS가 그 승인을 대신하지 않는다. 독립 백업/외부 감시는 대상·접근 권한 결정이 필요하다.

주요 재현 명령(테스트 환경: `TMPDIR=/home/soccz/22tb/tmp`, `PYTHONDONTWRITEBYTECODE=1`,
`PRELUDE_FORBID_TELEGRAM=1`, OMP/OPENBLAS/MKL thread=1):

```bash
venv/bin/python -B -m pytest -q -x -p no:cacheprovider tests
venv/bin/python -B -m pytest -q -x -p no:cacheprovider tests/test_capture_recommend_microstructure.py tests/test_close_research_isolation.py tests/test_recommend_trade_shortlist.py
ruff check --no-cache scripts/capture_recommend_microstructure.py signals/recommend_trade_shortlist.py tests/test_capture_recommend_microstructure.py tests/test_close_research_isolation.py tests/test_recommend_trade_shortlist.py
bash -n scripts/daily_close_distribution.sh scripts/daily_close_preopen.sh deploy/install_systemd.sh scripts/backup_db.sh
venv/bin/python -B scripts/report_recommendation_status.py
```

---

## 실사용 강화 8차 — 새 정보 수집·인과 피처·운영 격리 (2026-09-07)

사용자 `설계를 구체화하고 검증하고 검증까지 한번더 객관화 후 시스템 완성해 … 알아서 쭉쭉 가`
요청에 따라 설명에서 멈추지 않고, 기존 R1을 보존한 새 정보 연구 경로를 구현·검증한다.
`prelude-quant` 생성/검증 분리와 `verify` 회귀 절차를 적용한다.
기존7차 및 frozen 연구 증거·NOTES·사용자 dirty 변경은 보존한다. 새 모델 배포·라벨 변경·자동주문은 하지 않는다.

### 구현 전 고정한 설계와 완료 경계

1. 기존 공개 trade/orderbook collector와 raw 포맷을 재사용한다. 공개 시세만 읽고 API key/private 주문을 쓰지 않는다.
   작업 시작 시 실원본은0개였다. 수집을 실제 시작한 것과 과거·미래 성능이 검증된 것은 구별한다.
2. 새 순수 builder는 `[decision_started_at−300초, decision_started_at)`에 event/received 시각이 모두 들어오는
   REALTIME 체결만 사용한다. 기존 collector의 `decision_completed_at` raw 종료 계약은 보존한다.
   300초는 이번 단일 초기설정이며 결과를 본 뒤 여러 창을 골라 최선값으로 보고하지 않는다.
3. 첫 가설은 `trade_notional_imbalance_300s=(BID대금−ASK대금)/(BID대금+ASK대금)` 하나다.
   호가는 구독 준비·spread 품질 진단만 한다. 완결 수집의 무거래는 거래량0/imbalance=null,
   수집 누락은 unavailable로 구별한다. 원 snapshot 후보·순위·전달 Top3를 교체하지 않는다.
4. manifest/raw/snapshot identity·해시·ingress·중복체결·시계·warmup·연결/큐손실을 검산한다.
   v1은 불완전 capture를 사후 재해석해 구제하지 않는다. canary/명시시각 수집은 wire 검사이며 학습 준비로 승격하지 않는다.
5. 독립 collector service/timer를 설계한다(초기08:45, 늦은 catch-up/자동재시작 없음, depth1, 시간·자원예산).
   기존8개 스케줄 및 R1 pipeline lock/marker와 분리한다. 설치는 코드·unit·롤백 테스트 후 권한 승인 경로로 한다.
6. 운영 실패 격리는 실제 결합 지점을 확인한 뒤 수리한다. 단순 연구 오류 무시는 금지하며
   핵심 완료와 연구 unavailable을 분리해 stale 연구 수치가 최신으로 게시되지 않도록 해야 한다.
   별도 외부 감시 주소/계정은 현재 없으므로, 같은 서버 감시를 독립 외부 감시라고 표현하지 않는다.
7. 표적 고장주입→독립 코드/산식 재검산→전체 회귀→공개 시세 bounded canary→설치본 확인을 순서대로 수행한다.
   실험 효과 판정은 새 날짜 자료와 기존 비용/라벨을 연결한 동일 날짜 비교에서만 가능하다.

### 소유와 검증

- 메인: collector 실행경계·자원예산, 문서, 실접속 canary, 통합검증·권한 작업.
- 시그널 담당: 새 `signals/recommend_microstructure.py`와 해당 테스트(순수 feature/readiness, 학습/발송 없음).
- 평가 담당: 별도 인과성·평가계약 검토, 생성기 결과의 독립 검산; 새 모델 우위로 포장하지 않음.
- 운영 담당: 독립 수집기 unit/installer와 필요한 운영 격리 설계·회귀. 저장소 밖 설치는 메인만 수행.

- [x] 이전 상태·원본0개·기존 collector/운영 의존 감사.
- [x] 300초 단일가설/이중시각/원분모 보존 설계에 연구·평가 독립 검토 동의.
- [x] 구현 및 고장주입·독립 검증. 전수 재검증 **2,383 PASS /361.33초**.
- [x] 실제 공개 시세 수집 및 독립 원본 검산.
- [x] 저장소의 자동 수집·별도 순위 저장·후속 비교 배선과 설치기 검증.
- [x] 실제 시스템 설치: 사용자09-08 15:59 설치 완료, 9개 timer 및19개 설치파일 독립 검증 PASS.
- [ ] 첫 정규 수집·별도 시험 순위 기록: **09-09 08:45 KST 예정**, 설치 성공과 실행 성공은 구분한다.

### 실제 반증·보완과 기록 경로

- 체결 피처를 기존 위험점수의 정확동점 블록에서만 사용하는 단일 record-only 가설을 고정했다.
  과거 open snapshot37일(각100후보) 중 rank3/4 경계 동점7일(18.92%), 최대교체7/111픽이다.
  라벨/성과는 이 빈도 점검에 사용하지 않았다. 작은 구분력 실험이지 상방 헤드 재학습의 대체는 아니다.
- raw 전체를 list로 펼치는 초기 builder를 독립 검토에서 반려했다. gzip을 한 번씩 끝까지 검증하고,
  호가는 시장별 마지막 상태, 체결은300초 이내 ID·집계만 유지하도록 수정했다.
- 원래 writer의 부모 symlink 선검증 누락, 평가기의 잘못된 receipt 기본 폴더,
  uncertain score를 성공 처리할 수 있던 래퍼를 수리했다. 파일/디렉터리 내구 저장을 공통 writer로 합쳤다.
- 평가 입력의 manifest/raw 근거 제거와 valid flag/실패 사유 모순이 통과하는 초기 구현은
  독립 모의 공격으로 재현한 뒤 차단했다. 정상8파일 graph는 통과하고 공격5종은 거부한다.
- 최초 정규 관측일을 **2026-09-08**로 고정했다. 기록 자체가 없는 날짜도 달력 분모에 드러내며,
  저장된 no-op 날짜는 주 paired 비교에 포함한다. 자료 부족·late·미확인 저장을 개선 성공으로 바꾸지 않는다.
- 새 경로: 독립 timer → `scripts/capture_recommend_microstructure.py` → 원본/manifest →
  `signals/recommend_microstructure.py` → 새 `recommend_features.json` →
  `signals/recommend_microstructure_trial.py`의 write-once score/commit → 성숙 라벨의 누적 비교.
  현재 R1 snapshot/Top3/receipt/라벨/원장·자동 주문 부재는 불변이다.

### 실제 공개 시세 확인 (wire 검증만)

2026-09-07 **21:56:09~21:56:29 KST**, 공개 KRW281시장·depth1·20초·채널별16MiB 예산으로
API 키/실거래/Telegram 없이 접속했다. 자료는 다음 새 디렉터리에만 생성했다.

`data/microstructure/canary/2026/09/07/upbit-20260907-125609-7331454f/`

| 항목 | 체결 | 호가 |
|---|---:|---:|
| 실원본 이벤트 | 425 | 5,098 |
| 수락 payload bytes | 193,108 | 1,397,840 |
| gzip bytes | 78,993 | 884,453 |
| 재접속 / drop / parse error | 0 / 0 / 0 | 0 / 0 / 0 |

시작/종료 NTP yes, 호가 initial snapshot281개·realtime4,817개,
manifest SHA `8eb4a5f8c6a7a99f3f2a153ca5f166962966c234fb41ca19a86248aebffd97b1`.
독립 담당자가 원본 SHA·바이트·시각·순서와 builder의 **100행/null100/feature_evidence_valid=False**를
재검산했다. canary의 native complete=True는 이 새 연구 계약의 적격 판정이 아니다.
저녁 수집 목록에는 아침 원후보 KRW-BONK(rank51)가 없어 누락 gate도 발동했다.
폴더 증거만으로 그 원인을 단정하거나 후보를 다른 코인으로 바꾸지 않았다.

실파일5523행의 순차 검증0.482초, 프로세스 peak RSS 약105.5MiB를 측정했다.
같은 저녁 유량을23분으로 단순 확대하면 feature 처리CPU 약33초지만,
이는 개장 유량·CPUQuota50%의 소요시간·실제23분 지속 운용·최대 체결 ID 메모리의 보증이 아니다.
이전7차의 '원본0개'는 이번 접속 이전 상태였으며, 현재 **연구 적격 정규 날짜는 여전히0일**이다.

### 09-08 실제 운영 확인과 시간 경계 정정

작업 중 날짜가09-08 오후로 넘어가 목표했던08:45 수집 창이 지났다.
그때까지 새unit은 설치되지 않았으므로09-08을 정상 수집일 또는 no-op일로 만들지 않는다.
실험 시작 달력09-08은 유지하고 missing으로 드러내며, 설치 후 첫 정규 실행은09-09 08:45다.

독립 읽기 검증: **09-08 preopen08:53:05 / open09:08:50 KST**, 각3종목,
Telegram server_accepted·1회 시도·journal resolved·snapshot/receipt 유효였다.
사용자 읽음은 확인하지 않는다. 두 일일 스크립트도exit0이며 추천은 정상 유지됐다.
07:30 selftest는 '서비스8개' 및 옛 bulk-enable 문자열을 가정한 정적 테스트2건 때문에 실패했다
(2 failed /2,381 passed, 07:36 종료). R1 발송은 막지 않았고 두 테스트는 새9개·선택 활성화 계약으로 수리했다.

### 최종 검증·설치 경계 (2026-09-08)

- 구현자와 별개로 시그널/평가/운영 담당자가 검토했으며 발견한 경계 결함을 수정·재검산했다.
- 전수1회차: 2 failed /2,378 passed, 335.86초. 새 서비스 수 및 변경된 설치 방식에 관한
  두 정적 테스트를 고친 뒤 **전수2회차2,383 passed /361.33초**.
- 신규 feature66개, trial69개, wrapper14개, installer101개, 기존 OnFailure6개 표적 검증 PASS.
  Ruff·셸 문법·diff-check, 실제 빈 실험 저장소 CLI, 공개 시세 원본 독립 검산도 수행했다.
- 기존 백업 목록의 원본 `data/microstructure/upbit`에 더해 새
  `output/recommend_microstructure_trials`도 포함했다. 원본만 있고 score/commit을 잃는 복구 결손을 막는다.
  전수검증 뒤의 이1행 보강은 백업 관련 **21 PASS /56.15초**로 추가 검증했다.
  격리 fixture에서 동일 archive의 raw/score/commit 바이트·manifest SHA·원본 불변을 확인했으며
  실제 운영 백업/외부 저장소 쓰기는 실행하지 않았다.
- 새 진단: `_workspace/recommend_microstructure_trial_evaluation_20260908_v1.json`,
  SHA `e0dabb4252dfa27c6283d0f06c024e85fdeaa845c724a05721a3486729067150`.
  실제 상태는 expected1 /missing1 /paired0 /effect_status=not_evaluated /deployable=false다.
- 최초 자동 설치 시도(`sudo -n bash deploy/install_systemd.sh --add-microstructure`)는
  `sudo: a password is required`에서 중단됐다. 당시 두 새unit은 `LoadState=not-found`였으며,
  이 실패 전후 기존8timer 상태·다음 발화시각과 설치파일17개의 SHA·inode·권한이 모두 동일했다.
  이후 사용자 직접 설치와 아래 읽기 전용 검증으로 설치 대기는 해소됐다.
  에이전트는 실제 발송/재전송·자동 주문·자동 학습·수집기 수동 기동을 수행하지 않았다.

재현한 주요 명령(테스트 공통 환경: `TMPDIR=/home/soccz/22tb/tmp`,
`PYTHONDONTWRITEBYTECODE=1`, `PRELUDE_FORBID_TELEGRAM=1`, OMP/OPENBLAS/MKL thread=1):

```bash
venv/bin/python -B -m pytest -q -x -p no:cacheprovider tests
venv/bin/python -B -m pytest -q -x -p no:cacheprovider tests/test_shell_failure_propagation.py -k backup
venv/bin/python -B scripts/evaluate_recommend_microstructure_trial.py
bash -n deploy/install_systemd.sh scripts/backup_db.sh
```

### 사용자 설치 후 확인 — 2026-09-08 16:01 KST

사용자가 아래 설치 명령을 완료했다. 신규 timer 활성 진입시각은09-08 **15:59:14 KST**다.

```bash
sudo bash /home/soccz/22tb/prelude/deploy/install_systemd.sh --add-microstructure
```

- `systemctl show`로9개 timer의 loaded/enabled/active와 `/etc/systemd/system` FragmentPath를 확인했다.
  새 수집기는 active(waiting), 다음 실행 **09-09 08:45:00 KST**, LastTrigger 없음이다.
  service는 loaded/static/inactive이며 실제 실행 이력이 없다. Result=success만으로 수집 성공이라고 하지 않는다.
- 설치파일19개(service9+timer9+failure template1)의 SHA가 저장소와 모두 일치했다.
  기존17개 설치파일 inode·권한도 설치 전과 동일하고, 기존8timer ActiveEnterTimestamp는
  모두08-03 10:58:13으로 유지돼 이번 설치에서 재시작되지 않았음을 확인했다.
- 기존8timer의 다음 실행 **초 단위 값**은 설치 전과 달라졌다. nominal calendar·정의는 동일하며,
  변경 폭은 각각 RandomizedDelaySec 범위 안이다. daemon-reload 이후 지연값 재계산으로 추정하며,
  '다음 발화시각까지 완전히 동일'이라고 보고하지 않는다.
- 독립 운영 담당자가 새 service의 wrapper/인자, CPU50%·메모리2GiB·45분/90초 제한,
  자동재시작 없음·Telegram 금지·기존 R1 의존성 부재를 다시 확인했다.
- 사용자 설치 로그는 전체 cron/중복 timer 검사와 설치 후 검증 통과를 기록했다.
  에이전트의 별도 `sudo -n ... --check-only` 재실행은 sudo 인증에 막혀 수행하지 못했다.
  이 제한은 설치 실패가 아니며, 실제 설치 파일·등록 상태는 위 직접 조회로 확인했다.
- `snapd.service:23 RestartMode` 경고는 로컬 systemd249가 별도 vendor unit 옵션을 인식하지 못한 것이다.
  새 prelude 유닛에는 해당 옵션이 없고 설치는 완료됐다. 관련 호스트 설정을 수정하지 않았다.

**다음 확인:**09-09 정규 실행 후 capture manifest·feature·score/commit·비교 보고서를 확인한다.
지금 추가 설치나 수동 재실행은 필요하지 않다. 첫 실행과 추천 개선 효과는 아직 미검증이다.

아직 증명되지 않은 것: 정규 아침 장시간 수집·첫 durable challenger 기록·추천 개선 효과.
기존 selftest 지연 부팅 의존, close/publish 연구 상태 결합, 외부 독립 감시는 이번 신규 수집 경로와 별개다.
이는 시스템 전체 무결점/수익성 보장이 아니라, 새 정보를 안전하게 검증하기 위한 8차 구현의 인계 상태다.

---

## 실사용 강화 7차 — 전달·측정·신규 후보 기본 잠금 (2026-09-07)

사용자 `설계를 구체화하고 검증하고 검증까지 한번더 객관화 후 시스템 완성해`에 따라,
기존 운영을 재개발하지 않고 읽기 전용 3영역 감사에서 확인한 결함을 수리한다.
Telegram 표현 정정은 사용자 `ㅇㅇ 그건 정정해`로 별도 승인됐다.

### 구현 전 고정한 완료 기준

1. **중복 발송의 crash 창 차단:** `scripts/recommend_send.py`의 추천/보조 champion 알림 모두
   API 전에 durable intent를 남긴다. API 수락 후 receipt 저장 전 종료는 미확인 상태로 남고 자동 재발송하지 않는다.
   과거 실패 receipt와 새 미완료 시도를 구별하고, 성공을 추정하거나 일정 시간이 지났다고 자동 해제하지 않는다.
   기존 receipt 계약은 보존한다. 이는 정확히 한 번 전달 보장이 아니라, 불확실할 때 중복보다 보류를 택하는 정책이다.
2. **표현과 실제 의미 일치:** 확률은 검증 중 추정치, 09:00 open은 현재 체결가격이 아닌 참고가격으로 표기한다.
   종목·순위·수치·TP/SL·실거래 자동주문 부재는 불변이다. 과거 메시지/receipt는 수정하지 않는다.
3. **성과를 유리하게 바꾸는 사후 대체 금지:** 일상 `evaluate_recommend_score_labels.py`에서 원후보로
   TopN·최근접 유동성 대조군·ATR band를 먼저 고정한다. 선정/필수 대조군의 결과 미관측은 날짜별 비교 불가로
   표시하고 다음 순위로 교체하지 않는다. 지표별 일부 결측도 부분 종목 평균으로 대체하지 않는다.
   labeled-only 확률 진단·legacy gross·기존 net 비용 기준은 유지하며 기록/관측/전달 결과 coverage를 분리한다.
4. **새 등록 모델은 기본적으로 연구용:** `ModelSpec.challenger_only` 기본값을 잠금으로 바꾸되,
   현재 명시 등록 모델들의 설정과 결과는 동일하게 유지한다. 이 수리를 완전한 신규 모델 승격 승인체계로 과장하지 않는다.
5. **독립 재검증 후 종료:** 고장 주입·동시 실행·실제 프로세스 종료·과거 실패 receipt 잔존·입력/출력 변조·
   Top3/control halt·all-halt·shuffle·비용/legacy 회귀, 전수 tests, Ruff, 셸 문법 검사를 수행한다.
   실자료 검증은 원본을 읽고 새 검증 산출물만 만든다. 실제 발송·기존 보고서 덮어쓰기·서비스 재시작은 하지 않는다.

### 파일 소유와 검증 계획

- 운영 담당: 새 `notifier/delivery_attempt.py`, `scripts/recommend_send.py`, 해당 journal/send 테스트.
- 평가 담당: `scripts/evaluate_recommend_score_labels.py`, 해당 평가 테스트.
- 메인: `ops/recommendation_status.py`의 journal 미확인 상태 반영, `signals/model_registry.py`의 기본 잠금,
  해당 테스트 및 본 문서/OPS/README 정리.
- 독립 검토 담당: 구현 전 두 설계의 반증 검토, 구현 후 저자와 별개로 경계·실자료 수치 확인.
- 기존 source-pinned snapshot/receipt/Telegram/label/evidence·오프라인 실험 소스와 산출물은 보존한다.

### 진행 상태

- [x] 운영/평가/새 정보 가용성 3영역 감사. 기존 전달·청산·백업·OnFailure는 재사용.
- [x] 사용자 Telegram 문구 정정 승인.
- [x] 세부 설계 독립 재검토 PASS 후 구현. 운영·평가·상태/레지스트리 저자 분리 검토 완료.
- [x] 표적/고장 주입·전수 검증, 실자료 독립 검산. 최종 **2,165 PASS /322.37초**.
- [x] 운영 동작/한계/미완료 경계와 재현 명령 기록. 운영 수리 완료, 다음 실발송·추천 엣지 증명과 구분.

구현 후 반증 검토에서 intent의 슬롯/미래시각 검사, 합법적인 상위 경로 alias와 증거 symlink의 구분,
FIFO를 열 때 무기한 대기하지 않는 `O_NONBLOCK`, 잘못된 "09:00 가상평가 기산" 문구를 추가 확인·수리했다.
원래 확률의 고가/저가 의미는 보존하고 전문용어 대신 검증 중 추정치로 표현한다.
오늘 장전/장후 snapshot을 메모리에서만 렌더링하여 후보 각3개·원본 바이트 불변과 정정 문구를 확인했다.
등록 7개 `ModelSpec` 값도 변경 전후 정확히 동일하다(JSON sort_keys 직렬화 SHA-256:
`301a8a7e4d7aa7d03a44aff49af7c05d9bf5c11711a450026076d193496c61f7`).

### 실자료 판정과 보존

신규 보고서: `_workspace/recommend_score_label_evaluation_20260907_v3.json`
(file SHA `ac44f400ec990343619611197c84c0cc6e66a2b072058e69b07652400cba5ce4`).
2026-09-06까지143개 라벨 아티팩트만 읽었으며 원본/기존 운영 보고서는 변경하지 않았다.

| R1 슬롯 | 기록/관측 후보 행 | 전달/관측 픽 | Top3 전체·ATR 비교 | Top3 유동성 비교 |
|---|---:|---:|---:|---:|
| open | 3,600 / 3,600 | 99 / 99 | 36 / 36일 | 36 / 36일 |
| preopen | 3,800 / 3,797 | 105 / 105 | 36 / 38일 | 38 / 38일 |

미관측은 preopen 08-04 AERGO(rank19)·AQT(rank20), 09-01 TT(rank62)다.
현재 자료의 실제 전달 Top3에는 결과 미관측이 없었다. 실제 수치상 이번 수리는 **대조군 부분 평균 제거와
분모 공개**이며, Top3 대체 오류는 고장 주입으로 방어를 증명했다. 유동성 매칭은 이3개를 사용하지 않아
38일이 유지된다. 서로 다른 가용 날짜의 비교를 같은 표본의 결과로 해석하면 안 된다.
독립 검토자는 evaluator helper를 호출하지 않고 stdlib로143개 checksum, 전체4채널14,300/14,297행,
1,287개 날짜×TopN×비교군 선정·coverage를 다시 계산해 일치를 확인했다.

### 검증 명령·경계

- 표적: 전달/snapshot/radar198개, 평가기38개, 상태/registry72개 PASS; 정책 비교32개도 PASS.
- 1차 전체 회귀: **2,163 PASS /318.65초**. 이후 읽기 동시성 최소재현에서
  `inspect(absent) → 새 pending 게시 → old failed receipt 읽기`가 `not_delivered`를 내는 결함을 확인했다.
  receipt 읽기 전후 journal 증거 재대조로 수리하고 pending/손상 회귀2개를 추가했다.
  수정 후 상태/journal/registry109개 PASS, 최종 전수 **2,165 PASS /322.37초**.
  마지막 변경도 별도 운영 검토자의 코드 검토와 회귀2개 재실행 PASS를 받았다.
  조회는 여전히 무쓰기·무락이며, 진행 중 증거가 바뀌면 확정 판단을 보류한다.
  전수 검사 전후 운영 소스5개 및 기존 운영 출력8개 SHA 동일; Ruff12파일·셸5파일 문법 검사 재통과.

```bash
TMPDIR=/home/soccz/22tb/tmp PYTHONDONTWRITEBYTECODE=1 PRELUDE_FORBID_TELEGRAM=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 venv/bin/python -B -m pytest -q -x -p no:cacheprovider tests/
```

- Ruff: 변경 Python12개 파일 `ruff check` PASS. 활성 runner/heartbeat/backup 셸5개 `bash -n` PASS.
- 기존 horizon/calibration generator manifest16개 및 horizon data provenance230개 항목 checksum PASS
  (항목 간 중복 포함, 운영 DB 연결 없이 저장 파일만 대조).
- 실제 조회: `PYTHONDONTWRITEBYTECODE=1 PRELUDE_FORBID_TELEGRAM=1 venv/bin/python -B scripts/report_recommendation_status.py --format text`.
  09-07 preopen08:53:09/open09:08:07 서버 수락 증거 확인, 각3후보. 이는 수정 전 오늘 발송 기록의
  호환 검증이며 **새 journal을 쓴 다음 정규 발송의 실증은 아직 아니다**.
- 현재 champion state는 기본 잠금 변경 뒤에도 `_validate_state(expected_asof="2026-09-08")`를 통과했다.
  상태/모델을 재생성하지 않은 읽기 검사이며 다음날 시장 데이터 freshness나 실제 전송 성공 보장은 아니다.
- 구현/독립 리뷰 완료와 추천 수익성 인증을 구별한다. 신규 학습·실제 전송·라벨/추천 순위 변경·
  systemd 설치/재시작·기존 output 재생성은 하지 않았다.

### 이번 완료와 혼동하지 않을 것

- 6차 보정안은 계속 미채택이며 새 모델 학습/배포는 하지 않는다.
- b_vol_surge·breadth 등은 이미 비교·기각 이력이 있어 신규 아이디어로 재탕하지 않는다.
  실제 체결/호가 원본은 현재 `data/microstructure`에0개다. 수집 코드가 있다는 이유만으로 연구자료가 있다고 하지 않는다.
  향후 새 정보 연구는 open 결정 시작 시각 이전의 event/received 이중 시각을 갖춘 원본 확보가 먼저다.
- 서버 전체 장애를 감지할 독립 외부 감시, 연구 장애와 core publish의 대규모 분리, 새 모델의 승인/forward
  lifecycle은 이번 수리만으로 완성됐다고 주장하지 않는다. 별도 범위/실운영 증거가 필요한 경계로 남긴다.

---

## 실사용 강화 6차 — 보정용 점수 생성 방식 분리 비교 (2026-09-07)

**결론: 이번 고정 보정안은 개선 증거 부족으로 미채택, 운영 R1 유지.**
35일·335 fits 실행과 두 갈래 독립 검산을 완료했다. 시간순 보정은 확률 오차를 조금 줄였지만,
사용자가 원하는 저하방·고상방 Top3의 동시 개선을 입증하지 못했다. 효과가 없다는 확정이나
기존 R1의 수익성/안전성 인증은 아니며, 이번 후보를 배포할 근거가 없다는 의사결정이다.

사용자 `너가 그럼 앞으로 방향성을 설계하고 확실한 결론이 나올 때까지 쭉 진행해`에 따라,
직전 제안한 **보정 단독 오프라인 비교의 설계·구현·학습·독립 검산**을 진행한다.
`prelude-quant` 연구/평가/운영 교차검토와 `verify` 검증을 적용한다. 운영 추천·알림·원장·스케줄·배포는 변경하지 않는다.

### 결과를 보기 전에 고정한 설계

| 군 | 역할 | 최종 예측 모델 | 보정에 쓰는 자료 |
|---|---|---|---|
| A | 실제 발송 참조 | 원 snapshot 그대로 | 원본 보존 |
| F | 기존 전체 보정 | 5차 B의 기존 날짜 모델 재현 | 전체 학습행의 fitted score / 기존 10구간 |
| R | 최근 기간 효과 대조 | F와 동일 raw 예측 | 최근 180달력일의 fitted score / 기존 10구간 |
| O | 시간순 보정 후보 | F와 동일 raw 예측 | R과 정확히 같은 행·라벨의 과거-only OOT score / 같은 10구간 |

- 주효과 **O−R**, 기간 제한 효과 **R−F**, 실용 총효과 **O−F**를 동일 코호트에서 보고한다.
  A와의 비교는 실사용 참조이지 보정만의 인과효과가 아니다.
- 학습 target은 기존 B same-row D1 그대로. 상방 3개/하방 2개 확률 보정만 비교하고,
  기대 하방은 F의 기존 값을 F/R/O 모두에 고정한다. 장전 타깃 시차 문제를 함께 수리했다고 주장하지 않는다.
- outer는 기존 장전 35일·3,500후보/원 결과. F를 5개 head×35일=175 fits로 재현하고,
  원 B raw·보정확률·기대 하방·학습행/X/y/median/가중치/부스터 hash exact parity를 요구한다.
  불일치를 tolerance 확대나 새 대조군으로 치환하지 않는다.
- 보정 창은 `[outer cutoff−180일, outer cutoff)`; 합집합 2026-01-23~08-30의 220달력일이다.
  Jan23에서 시작한 7달력일 블록 32개로 내부 expanding OOT score를 만든다(최대160 fits).
  내부 모델은 주 단위 동결, outer 최종 모델은 매일 재학습. **전체 최대335 fits, 설정 한 가지**.
- 내부 학습 cutoff는 각 블록 시작일−5일 exclusive. 입력·B/C 타깃 종료는 블록 시작 09:00 이전이어야 한다.
  내부 median·class weight는 해당 내부 과거 train만 사용한다. 최종 모델의 통계를 가져오지 않는다.
  모델 생성 모의시각과 각 미래 행 피처/예측 가용시각을 분리한다. 뒤 날짜 피처를 블록 시작에 알았다고 기록하지 않는다.
- R/O 보정 행·순서·B 라벨은 exact 동일. OOT 점수 누락 시 표본을 임의 축소하지 않고 해당 날짜 두 군을 unavailable로 처리한다.
  초기 가용성 guard: 150행·5일 이상, 각 head 양성/음성 각각12개 이상. 내부 train은365일 이상.
  미달·상수 fallback은 숨기지 않으며 성과를 보고 guard를 완화하지 않는다.
- 위 숫자는 비용/표본을 고려한 이번 고정 초기설정이지 최적값이 아니다. O−R에는 과거 모델의 학습량·최대6일의
  모델 나이·점수 분포 차이도 포함된다. resub 과적합 제거 하나의 순수 인과효과나 OOF 일반론으로 확대하지 않는다.
- 4군 공통 baseline-complete 주코호트와 selected-only 보조코호트를 사전 구분한다.
  라벨을 보기 전에 Top3·유동성/변동성 매칭을 고정하고 pool<3을 사후 확대하지 않는다.
  기존 6지표/날짜 동일가중/paired IID 및 block3 CI(4,000회·seed42)/LODO/종목 기여합/5관측일 진단을 사용한다.
- 확률 오차만 줄어서는 채택하지 않는다. 상방 기회·하방 발생·MAE·두 가상 순수익을 함께 비교한다.
  악화한 설정은 기각, 균형 있게 좋아도 오프라인 유망까지만, 손익교환/넓은 CI는 증거 부족으로 남긴다.
  결과에 맞춘 즉석 sweep 없이 한 번의 고정 비교를 마친다.

### 실행·검증 상태

- [x] 기존 5차 bundle/report의 payload·source manifests 재확인 PASS, 원본 보존.
- [x] 연구·독립 평가·운영 담당의 설계 조건부 PASS를 반영해 위 비교군·시점·중단 기준 확정.
- [x] 새 오프라인 kernel/runner/evaluator/CLI 4개와 테스트 4개 구현. 신규 127개 표적 테스트 PASS,
  Ruff/diff-check PASS. 연구↔평가 독립 코드 리뷰에서 치명적 결함 없음.
  합성 실제 XGBoost의 기존 B exact parity와 kernel→runner 통합/입력 불변/외부 I/O 차단을 검증했다.
- [x] 고정 설계 JSON 기록 후 335 fits 실행, 공통 표본·F 재현·결과 독립 검산 PASS.
- [x] 전체 테스트 및 결론/한계/후속 방향 기록.

고정 설계: `_workspace/recommend_calibration_design_20260907_v1.json`, SHA256
`0841488f1f85b6b11cc68edc15607b7b0a26842038bca60caf4c919a2329881b`.
실자료 입력 preflight PASS: 학습 123,977행 / 후보 3,500행 / 35일 / 원본53열, 원 SHA·패키지 버전·기록된 출처 전후 일치.
전수 `tests/`: **2,084 passed in 285.39s** (Telegram 금지 및 DB 격리 guard 유지).
실행: **2026-09-07 17:43:54~17:59:58 KST**, 내부32구간×5 + 최종35일×5 = **335 fits**.
전 날짜 F exact parity PASS, R/O unavailable 0일. OOT 캐시22,000행·최근 fitted 캐시629,909행을
무손실로 보존해 보정맵과 적용 확률을 별도로 재계산할 수 있게 했다.

결과: `_workspace/recommend_calibration_comparison_20260907_v1.json`, whole-file SHA256
`4157b337e303bf82ca5da114181f5b5fd6e0d2a78acb9ecd6d5b92f6adaca493`.

독립 검증 두 갈래 모두 PASS이며, 각 보고서에 담은 검산 코드 자체도 별도 namespace에서 재실행해 일치했다.

- 선정/성과: 평가 helpers 재사용 없이 원 예측에서4군 순위·매칭·코호트·6지표·6대조·CI·LODO·
  종목기여·확률 진단을 재계산. **15,681개 수치 일치, 최대 오차1.11e−16**, 원본53열×3,500행 exact.
  `_workspace/recommend_calibration_result_audit_20260907_v1.json`, whole-file SHA256
  `2282fdbf3ea24204ff2983a3ad2a5c12b19981ea2bab560b9768334de2544ade`.
- 학습/보정 증거: 새 kernel/runner/보정 helpers 재사용 없이 inner32·outer35의 cutoff/키/median/X/y/가중치/시점,
  F↔기존B **38,500셀**, R/O **350보정맵·35,000확률** exact 재계산. 234개 고유 출처 파일 전후 SHA 일치.
  `_workspace/recommend_calibration_reconstruction_audit_20260907_v1.json`, whole-file SHA256
  `298694c757ad97f7047afd63842ad8f5725941b91a0ad829f466c44621baac86`.
- 검산은 추가 학습0회다. inner raw XGB 예측·부스터 자체를 또 생성한 것은 아니며, 전체역사 F rawtrain
  캐시가 없어 그 보정맵 학습 자체는 기존B 증거와의 동등성으로 확인했다. R/O 맵은 실제 캐시에서 독립 재계산했다.
  입력 달력상 성숙도 검증은 과거 DB 실제 입고시각까지 증명하지 않는다.

### 같은 날짜·같은 분모로 비교한 결과

주코호트 **33일·군별99픽**, 전체 후보3,300행이다. 원 라벨이 없는 08-04(AERGO/AQT)와
09-01(TT)만 제외했다. 보조코호트는35일·군별105픽이다. 5차의32일과 분모가 다르므로
기존 B가 정확히 재현돼도 그때의 집계 수치와 직접 비교하지 않는다.

| 주코호트 지표 | 실제 발송 A | 기존 전체 보정 F | 최근 보정 R | 시간순 보정 O |
|---|---:|---:|---:|---:|
| 24h 내 +10% 도달 | 13/99 = 13.13% | 9/99 = 9.09% | 8/99 = 8.08% | 9/99 = 9.09% |
| 24h 내 −5% 도달 | 21/99 = 21.21% | 18/99 = 18.18% | 19/99 = 19.19% | 20/99 = 20.20% |
| +10% 도달 AND 전 경로 −5% 미도달 | 11/99 = 11.11% | 7/99 = 7.07% | 7/99 = 7.07% | 7/99 = 7.07% |
| TP +5% / SL −3% 가상 순수익·픽 평균 | +0.232% | +0.222% | +0.053% | +0.244% |
| 24h 후 가상 순수익·픽 평균 | +0.497% | +0.329% | +0.005% | −0.022% |
| 평균 최대 불리 변동(MAE) | −3.385% | −3.134% | −3.312% | −3.324% |

가상 수익은 원 canonical 비용 차감값이며 재차감하지 않았다. 사용자 실제 매매·포트폴리오
복리수익·Sharpe/Max DD가 아니다. whole-path safe-up은 선도달 성공 또는 실제 익절과 다르다.

- **보정용 점수 방식 O−R:** TP/SL 가상 순수익 `+0.1904%p/픽`, IID CI95
  `[-0.2131, +0.5898]%p`, 3관측일 block CI95 `[-0.1125, +0.5051]%p`.
  +10%도달도1건 늘었지만 −5%도달도1건 늘었고, safe-up은 같았다. EOD `−0.0270%p`,
  MAE `−0.0117%p`. **6지표 모두 두 CI가0을 포함한다.** TP/SL LODO는33/33 양수
  (`+0.1130~+0.2797%p`)여도 위험·상방의 동시 우위를 입증하는 것은 아니다.
- **기존 방식 대비 총효과 O−F:** TP/SL 차이는 `+0.0212%p/픽`에 그쳤고 block CI95
  `[-0.5036, +0.6065]%p`. +10%도달은9건으로 같지만 −5%도달은18→20건,
  EOD `−0.3514%p`, MAE `−0.1904%p`. TP/SL LODO는 양수24회/음수9회로 방향도 고정되지 않았다.
- **기간 제한 R−F:** TP/SL `−0.1692%p`, EOD `−0.3244%p`; 두 CI 모두0 포함.
  최근 자료만 사용하면 좋아진다는 근거도 없다. O−R의 양수만 골라 기존 F보다 우월하다고 하지 않는다.
- **35일 보조에서도 채택 근거 없음:** O−F TP/SL `−0.0562%p`, EOD `−0.5237%p`,
  +10%도달11→10건, −5%도달21→23건. O−R TP/SL은 `+0.1592%p`지만 block CI95
  `[-0.1640, +0.4697]%p`. 더 유리한 주/보조 표본을 골라 결론을 바꾸지 않았다.
- 전체3,300후보의 **확률 오차는 개선**: up10 Brier F `0.097663`→R `0.097520`→O `0.096846`,
  dn5 F `0.155745`→R `0.156546`→O `0.155240`. 그러나 up10 AUC는 F `0.649537`→O `0.647576`이다.
  확률 숫자를 다듬는 것과 좋은 Top3를 고르는 것은 다른 문제였다. 각 군의 선택 종목이 다른
  selected-only AUC/Brier를 동일 표본 보정 개선으로 오해하지 않는다.
- O의 유동성/변동성 매칭군 대비 TP/SL lift는 `+0.3691%p`이나 dn5도 `+0.8321%p`,
  EOD는 `−0.0841%p`다. 이 보조 진단을 기존 F·실제 A를 이겼다는 증거로 대체하지 않는다.

### 결론을 바탕으로 정한 앞으로의 방향

1. **운영 유지:** F/R/O 중 어느 것도 새 champion으로 만들지 않는다. Telegram 문구·기존 순위·
   원장·주문·자동 재학습·systemd/cron을 변경하지 않았다. 기존 R1이 검증된 안전 수익 모델이라는 뜻도 아니다.
2. **같은 자료로 보정만 계속 바꾸는 탐색은 종료:** 이번 O를 살리려고 bucket 수·180일 창·
   embargo·RR 식을 사후 조정하지 않는다. 시간순 보정 일반론의 실패로 확대하지도 않는다.
3. **연구 우선순위는 상방 선별 정보와 Top3 선정력:** 새 정보/피처를 제안할 때는 현재 보유·시점
   증명 가능 여부부터 조사하고, 한 번에 한 요소만 비교한다. 기존 veto/utility·판정 완료 피처의 재탕을 피한다.
   전체 후보 AUC와 선택된3종목의 성과를 분리하고, 같은 날짜·변동성·유동성에서 상방 기회를 늘리면서
   하방을 악화시키지 않는지 확인한다. 이것은 후속 설계 원칙이며 새 모델의 성공 보장이 아니다.
4. **새 날짜의 독립 증거를 확보:** 기존 전 유니버스 snapshot/receipt/canonical 결과 기록을 유지하고,
   반복 관찰한 이번35일은 개발자료로 봉인한다. 다음 후보의 입력·설정·선정·평가기간·중단 기준은 새 결과를
   보기 전에 고정한다. 신뢰구간이 마음에 들 때까지 같은 과거 자료를 늘려 읽는 방식은 쓰지 않는다.
5. **운영 연결은 별도 단계:** 새로운 architecture/라벨/자동 재학습·실시간 후보 점수 기록은 별도 설계·승인 사안이다.
   이번 후보는 그 단계로 넘길 근거가 없어 shadow 스케줄도 추가하지 않는다. 향후 후보가 오프라인에서 유망한 경우에만
   결과 전 score 기록→새 forward→동일 날짜 위험/상방/순수익 비교→사용자 승인 순으로 진행한다.

### 검증·재현과 연속 작업 메모

```bash
TMPDIR=/home/soccz/22tb/tmp PYTHONDONTWRITEBYTECODE=1 PRELUDE_FORBID_TELEGRAM=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 venv/bin/python -B scripts/compare_recommend_calibration.py --bundle _workspace/recommend_horizon_bundle_20260907_v1.json --reference _workspace/recommend_horizon_comparison_20260907_v1.json --design _workspace/recommend_calibration_design_20260907_v1.json --output _workspace/recommend_calibration_comparison_20260907_replay_NEW.json
TMPDIR=/home/soccz/22tb/tmp PYTHONDONTWRITEBYTECODE=1 PRELUDE_FORBID_TELEGRAM=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 venv/bin/python -B -m pytest -q -x -p no:cacheprovider tests/
ruff check --no-cache scripts/compare_recommend_calibration.py signals/recommend_calibration_model.py signals/recommend_calibration_experiment.py signals/recommend_calibration_eval.py tests/test_compare_recommend_calibration.py tests/test_recommend_calibration_model.py tests/test_recommend_calibration_experiment.py tests/test_recommend_calibration_eval.py
git diff --check
```

- 이전 설계·bundle·report·source는 보존하고 재실행 결과는 새 이름만 허용한다. 위 첫 명령은335 fits를
  다시 수행하는 재현용 설명이지 자동 반복 예약이 아니다.
- 이 차수 변경: 신규 오프라인 Python 8개, 고정 설계/비교/독립 감사 JSON, 본 PHASES 절.
  테스트/전후 SHA/호출경로 감사에서는 운영 반영이 없음을 확인했으며, 기존 dirty output을 새 변경으로 귀속하지 않는다.
- 현재 승인된 보정 단독 비교는 설계→구현→학습→검산→미채택 결론까지 완료했다.
  다음은 새 독립 정보·자료 범위를 정하는 후속 설계이며, 같은35일 튜닝이나 운영 승격이 아니다.

반복 관찰한 35일은 새 holdout/forward가 아니다. 과거 DB 입고시각도 여전히 미확인이다.
새 실시간 점수 기록이나 모델 배포는 이 실행에 포함하지 않는다. 이번 판단이 불확실하면 불확실하다는 결론으로 멈춘다.

---

## 실사용 강화 5차 — 승인된 장전 날짜 정합화 대조 실행 (2026-09-07)

- **최종 결론: 실행·독립 검산 완료, 이번 고정 C는 채택 후보에서 기각·미배포.**
  원하는 저하방·고상방 추천 개선을 입증하지 못했고, TP/SL 가상 순수익은 대조 B 대비 주코호트 `−0.956%p/픽`.
  날짜 정합화 전체의 불가능성 증명은 아니다. 기존 R1 추천은 유지한다.
- **승인 범위:** 사용자 `ㅇㅇ 당연 가야지`는 직전 질문의 오프라인 비교용 타깃 날짜 변경·재학습 승인이다.
  운영 모델·추천·알림 포맷·원장·스케줄·자동 배포·주문은 바꾸지 않는다.
- **사전 검토:** 연구/운영/독립 평가 담당자가 동시 검토했다. 동일 공통 X·median·B 기준 class weight,
  기존 resubstitution 보정 유지, exact-calendar B/C 타깃, 날짜별 재학습 35회·총 최대 350 fits 조건이다.
  35일은 반복 관찰한 과거 자료이며 `calendar_maturity_only; historical_ingestion_unknown`이다.
- **주 비교:** C−B, baseline-common 주코호트(최대 33일)와 selected-only 보조(최대 35일)를 미리 구분한다.
  실제 분모는 halt/매칭 pool<3/학습불가로 더 줄 수 있다. 더 좋은 코호트를 사후 주결과로 바꾸지 않는다.
  날짜 paired CI·3관측일 block CI·LODO·종목 기여합을 병기한다. 5관측일 그룹은 OOF가 아닌 진단이다.
- [x] 신규 데이터/모델/평가/CLI 및 테스트 8개 Python 파일 구현. 표적 **138 passed**, Ruff/diff-check PASS.
  실제 합성 XGBoost에서 기존 함수 대비 B의 5개 확률·기대 하방 exact parity 및 dn5 재사용 동일성 PASS.
  타입 변경·복소수·시점/누락·사후 선정·파일 변조·입력 변이·덮어쓰기 차단 회귀 포함.
- [x] 고정 설계 `_workspace/recommend_horizon_design_20260907_v1.json` 기록.
- [x] 공통 자료를 새 bundle로 고정하고 Parquet/base64 왕복 정밀도·자료형 동일성 검증.
  `_workspace/recommend_horizon_bundle_20260907_v1.json`, whole-file SHA256
  `473f21b7efc0d196f07a060bf1192ea8c09078e4071159c44a07cf5429389225`.
  학습 단계는 이 파일만 읽으며 실 DB를 다시 조회하지 않는다. 기존 감사/입력/소스 SHA는 전후 재검한다.
- [x] 실제 bundle을 별도 구현으로 직접 JSON/base64/Parquet 복원하여 35개 날짜 공통 학습행을 독립 재계산,
  원본 snapshot 35개/후보 3,500개/피처 84,000값/실제 추천 표식/저장 점수를 전수 대조했다.
  generator 9 + data provenance 230 identity entries 전후 일치(중복 경로 포함이므로 고유 파일 수와 다름).
  독립 재현 스크립트와 결과는 `_workspace/recommend_horizon_bundle_audit_20260907_v1.json`에 보존,
  SHA256 `8607b357ab735d09e1bc2a5be7a7d53867eb35877a6015b5a9545f6e3b6b5527`.
- [x] 실제 bundle 재인코딩 byte SHA 동일, canonical 3,500행 평가 스키마 및 A Top3 35일 일치 smoke PASS.
  smoke의 B/C는 명시 unavailable이며 성과 실험이 아니다.
- [x] 전수 `tests/`: **1,957 passed in 265.15s** (`PRELUDE_FORBID_TELEGRAM=1`, DB hermetic guard 유지).
- [x] 고정 35일 비교 학습·평가 실행, 원 예측에서 독립 수치 검산.
- [x] **이번 고정 설정은 채택 후보에서 기각**으로 정리. 결과에 따른 즉석 sweep·운영 승격 없음.

### 실행 결과 — 독립 재계산 PASS

실행은 2026-09-07 16:23:28~16:38:05 KST, **35일·350 fits 모두 성공**했다.
결과: `_workspace/recommend_horizon_comparison_20260907_v1.json`, whole-file SHA256
`f271f3ffcb633334185b6ced62f3f05f761f834fccfe594d430cebb83358b654`.

독립 평가자가 평가 helpers를 재사용하지 않고 순위·매칭·코호트·6개 지표·paired IID/block CI·LODO·종목 기여·
확률 진단을 재계산했다. **9,107개 수치 일치, 최대 오차 1.11e−16**, 원 후보 53열×3,500행 exact 일치.
`_workspace/recommend_horizon_result_audit_20260907_v1.json`에 독립 Python 재현 코드와 결과를 보존했다.
whole-file SHA256 `589a6875db014c4f3424d1f3105d1f68ff3f1483f94a8ea5dd572c8bceba2a0a`.
저장된 embedded script 자체도 별도 namespace에서 재실행해 같은 결과를 확인했다.

다른 담당자의 별도 검산에서도 공통 학습 mask/keys/median/X hash, **350개 타깃 y hash·클래스수·공유 가중치**,
보정 확률 35,000셀·기대 하방 7,000셀 및 229개 고유 출처 파일 전후 해시가 일치했다.
부스터 UBJ 해시는 보존/형식만 확인했으며 실제 과거 모델을 또 학습해 복원했다고 주장하지 않는다.

주코호트는 **32일·각 군 96픽**, 비교 가능한 전체 후보는 3,200행이다.
08-04(AERGO/AQT halt), 09-01(TT halt), 08-10(C의 DEEP 매칭 pool=2)를 사전 규칙대로 제외했다.
후보 자체나 이미 정한 Top3는 삭제/대체하지 않았다. selected-only 보조는 **35일·각 105픽**이며 제외일 0.

| 주코호트 지표 | 실제 저장 추천 A | 기존 날짜 대조 B | 날짜 정합 후보 C |
|---|---:|---:|---:|
| 24h 내 +10% 도달 | 12/96 = 12.50% | 9/96 = 9.38% | 11/96 = 11.46% |
| 24h 내 −5% 도달 | 21/96 = 21.88% | 18/96 = 18.75% | 22/96 = 22.92% |
| +10% 도달 AND 전 경로 −5% 미도달 | 10/96 = 10.42% | 7/96 = 7.29% | 10/96 = 10.42% |
| TP +5% / SL −3% 가상 순수익·픽 평균 | +0.161% | +0.167% | **−0.789%** |
| 24h 후 가상 순수익·픽 평균 | +0.438% | +0.282% | −0.067% |
| 평균 최대 불리 변동(MAE) | −3.474% | −3.201% | −3.819% |

수익은 기존 canonical 결과의 비용 차감값을 그대로 쓰며 재차감하지 않았다. 실제 사용자 매매나 포트폴리오
수익/Sharpe/Max DD가 아니다. A는 실제 발송 참조이고, 타깃 시점의 직접 대조는 **C−B**이다.
whole-path safe-up은 first-passage 성공이나 실제 이익 실현과 다르다.

- **C−B 가상 TP/SL 순수익 차이 −0.956%p/픽.** 날짜 paired bootstrap 95% CI
  `[-1.783, -0.159]%p`, 3관측일 block CI `[-1.600, -0.111]%p`.
  하루씩 빼도 **32/32회 악화**(`−1.120~−0.802%p`), 고정 5관측일 진단 묶음도 7개 중 5개 악화.
- C−B MAE 차이 `−0.618%p`, IID CI `[-1.230, -0.035]%p`, block CI `[-1.101, -0.103]%p`.
  주코호트에서 C는 불리 변동이 더 컸다.
- 상방 도달률은 B보다 `+2.083%p`이나 9건→11건의 작은 차이이고 CI는 0을 포함한다.
  하방 도달률은 `+4.167%p`(18건→22건)이며 역시 CI는 0을 포함한다.
  **전체 지표가 모두 통계적으로 악화했다거나 타깃 정합화 일반이 반증됐다고 해석하지 않는다.**
- 35일 보조코호트에서도 C−B TP/SL `−1.014%p`, block CI `[-1.715, -0.104]%p`,
  하루씩 빼도 35/35 악화. +10%는 양쪽 11/105로 같고 −5%는 B 21/105→C 26/105.
  보조 결과로 주코호트를 교체하지 않았다.
- 유동성·변동성 매칭군 대비 C의 TP/SL lift도 `−0.689%p`. 전체 후보의 Brier는 C에서 조금 개선됐지만
  (up10 B 0.098424→C 0.097796, dn5 B 0.159028→C 0.156813), **확률 오차의 개선이 좋은 Top3를 보장하지 않았다.**
  이 진단은 성과를 대신하는 채택 기준으로 쓰지 않는다.

### 다음 방향과 경계

1. 이번 고정 C를 추천 개선안으로 승격하지 않는다. B도 A를 유의하게 이겼다는 근거가 없어 교체하지 않는다.
   날짜 정합화 자체의 논리적 타당성과 실제 추천 성과는 별개였으며, 운영은 기존 R1을 유지한다.
2. 같은 35일에서 tree/임계/비율식을 바꿔 C를 구제하는 즉석 탐색은 하지 않는다.
   시간대·타깃·보정·정렬을 동시에 바꾸거나 죽은 veto/utility 피처를 새 실험처럼 재탕하지 않는다.
3. 다음 **설계 후보**는 기존 날짜/모델을 고정한 진짜 expanding OOF 확률 보정의 단독 비교다.
   이는 현재 resubstitution 보정의 신뢰도 결함을 다른 변화와 분리해 확인하려는 것이지, 수익 개선을 보장하는 처방은 아니다.
   피처의 정확한 시점·타깃 종료 이전 학습 차단과 공통 날짜 비교를 유지하며, 상방/하방/순수익을 함께 본다.
   현재 승인은 날짜 정합화 오프라인 비교 한정이므로 **이 새 연구나 자동 학습·운영 연결을 실행한 것은 아니다.**
4. 새 후보가 향후 오프라인에서 유망해져도 이후 도래하는 슬롯의 결과 전 점수 기록·새 forward가 필요하다.
   반복 사용한 35일을 새 holdout/forward로 부르지 않으며, 현재 결과로 승격 시점이나 개선 확정을 약속하지 않는다.

### 검증·재현 명령과 연속 작업 메모

```bash
TMPDIR=/home/soccz/22tb/tmp PYTHONDONTWRITEBYTECODE=1 PRELUDE_FORBID_TELEGRAM=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 venv/bin/python -B scripts/compare_recommend_horizon.py run --bundle _workspace/recommend_horizon_bundle_20260907_v1.json --bundle-sha256 473f21b7efc0d196f07a060bf1192ea8c09078e4071159c44a07cf5429389225 --design _workspace/recommend_horizon_design_20260907_v1.json --output _workspace/recommend_horizon_comparison_20260907_replay_NEW.json
TMPDIR=/home/soccz/22tb/tmp PYTHONDONTWRITEBYTECODE=1 PRELUDE_FORBID_TELEGRAM=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 venv/bin/python -B -m pytest -q -p no:cacheprovider tests/
ruff check --no-cache scripts/compare_recommend_horizon.py signals/recommend_horizon_data.py signals/recommend_horizon_model.py signals/recommend_horizon_eval.py tests/test_compare_recommend_horizon.py tests/test_recommend_horizon_data.py tests/test_recommend_horizon_model.py tests/test_recommend_horizon_eval.py
git diff --check
```

- 재실행은 새로운 결과 파일명만 사용하며 원 설계·bundle·비교 결과를 덮어쓰지 않는다.
- Pandas 3 Parquet 왕복은 object/str/null뿐 아니라 **열 Index dtype**도 따로 보존·검사한다.
  자료형을 무시한 equality나 CSV/기본 소수점 JSON 변환으로 label 경계·순위 동률을 바꾸지 않는다.
- 이 차수 변경: 신규 오프라인 Python 8개, 고정 JSON artifact 및 본 PHASES 절. 기존 운영 소스·알림·스케줄은 무수정.
  **현재 승인 범위는 실행·검산·기각 기록까지 완료됐다.** 후속은 위 OOF 단독 비교의 별도 설계·승인 검토이며,
  새 연구·운영 변경을 자동으로 시작하는 것이 아니다.

새 artifact는 기존 자료를 덮어쓰지 않는다. 코드 변경은 새 오프라인 모듈에 한정하며 이전 source-pinned 증거는 보존한다.

---

## 실사용 강화 4차 — 장전 타깃 정합화 방향·자료 준비도 (2026-09-07)

사용자: "너가 그럼 앞으로 방향성을 설계하고 확실한 결론이 나올 때까지 쭉 진행해".
`prelude-quant` 연구·독립 평가 분리와 `verify` 절차를 적용한다.
현재 요청으로 설계·자료 감사를 진행했고, **사용자 `ㅇㅇ 당연 가야지`로 비교용 타깃 날짜 정합화·오프라인 재학습을 승인했다.**
운영 모델·알림·원장·스케줄·자동 주문은 변경하지 않는다.

### 설계에서 먼저 기각한 지름길

- 이전 2차 C는 이미 "장전 저장 입력 → 추천 이후 24h up10"을 학습했다.
  이를 다시 A와 비교해 타깃 정합화의 새 증거라고 부르면 피처/학습기간/모델 교락과 재탕이 된다.
- 전일 open 라벨은 09:15 시작이므로 09:00 D1 대조 타깃의 대용이 아니다.
  날짜 누락을 건너뛴 `shift(-1)`·nearest-date join도 허용하지 않는다.
- `recommend._build_panel`은 내부 기존 DB helper가 `init_db()`를 호출하므로
  읽기 전용 조사에서 직접 실행하지 않는다. `recommend_today --dry-run`도 즉석 학습을 하므로
  무학습 상태 조회/새 후보 점수 기록용으로 사용하지 않는다.

### 방향 — 타깃 시점 효과를 다른 변화와 분리

| 비교군 | 역할 | 입력·학습 원칙 |
|---|---|---|
| A: 저장된 실제 R1 | 실사용 참조 | 당시 Top3·점수·사후 canonical 경로를 그대로 보존 |
| B: 기존 타깃 재학습 | 직접 대조군 | 공통 원천 패널의 row t 입력 → t 09:00~t+1 09:00 |
| C: 정합 타깃 재학습 | 후보군 | B와 같은 입력·학습행 → 정확한 t+1 09:00~t+2 09:00 |

- 주 비교는 **C−B**. C−A는 실용 참조이지 날짜 정합화 하나의 효과가 아니다.
- 상방만 바꿨다고 하방 head까지 수리됐다고 말하지 않는다. 이번 후속 설계는 R1의
  up5/up10/up20/dn5/dn10·기대 하방의 **시간창을 함께 이동하는 한 가지 가설**로 좁힌다.
  +5/+10/+20·−5/−10 기준, 24h 길이, Top3 및 기존 정렬/비용은 바꾸지 않는다.
- 같은 D1 원천 행·피처 allowlist·공통 결측처리·모델 파라미터·보정 절차·후보군·분할을 사용한다.
  next-calendar 라벨을 찾지 못한 행은 양쪽 공통 학습 표본에서만 가용성 사유와 함께 구분하고,
  추론 후보는 사후 결과 가용성으로 삭제하지 않는다.
- 학습 cutoff는 기존 reference−5일 exclusive를 기본으로 유지하고 두 타깃 종료·가용시각도
  검증 의사결정 이전인지 별도 확인한다. 5일은 운영에서 가져온 초기값이며 이번 결과에 맞춰 탐색하지 않는다.
- 현 helper는 양성 비율에 따라 `scale_pos_weight`를 자동 변경한다. 엄밀한 동일 파라미터 대조에서는
  의사결정 날짜/head별 B의 학습 라벨에서 정한 가중치를 B/C 공통으로 쓰는 안이 독립 검토를 통과했다.
  같은 X의 median은 한 번만 적합한다. OOF 보정 개선까지 동시에 도입하지 않는다.
  현행 fitted-score bucket 보정의 과적합 한계는 그대로 명시한다.
- 운영과 같은 재적합 간격을 비교하려면 **35개 장전 날짜별 동일 cutoff로 각각 재학습**해야 한다.
  B/C 두 군·5개 binary head는 최대 350 fits이며, dn5를 기대 하방에 재사용하는 경우에도
  동일 예측임을 먼저 확인해야 한다. 5일 단위 모델 동결을 적용하고 "운영과 같은 학습 주기"라고 하지 않는다.
- 모델 기본값은 현행 180 trees/depth 4/lr 0.05/subsample 0.8/colsample 0.8/
  min child weight 5/lambda 1.5/seed 42를 가져와 양 군에 공통 고정한다.
  이들은 비교용 초기값이지 최적값이 아니며, 결과를 보고 grid를 넓히지 않는다.
  C는 B 라벨 비율에 고정한 class weight를 쓰므로 "C의 최적 설정" 실험이라고 하지 않는다.
- 추론은 저장된 8자리 24피처를 두 군에 동일 사용하고, 학습은 한 번 고정한 재구성 full-precision X를 쓴다.
  과거 DB 입고시각·당시 full-precision 학습행은 복원됐다고 주장하지 않는다.
  두 군의 현재 재구성 패널 내 대조이지 과거 운영의 정확한 재생은 아니다.
- **기존 평가기를 통째로 재사용하지 않는다.** `recommend_experiment_eval._rank`는 arm별 상방만 받고
  하방/깊은 하방/기대 하방은 공통 원본 열을 쓴다. all-head 비교에는 arm별 up10/dn5/dn10/
  기대 하방으로 선정 계획을 먼저 확정하는 경로가 필요하다. 기존 지표·bootstrap 순수 연산은 재사용 가능하나,
  가짜 상방값에 비율을 끼워 넣으면 확률 진단이 오염되므로 금지한다.

### 결론과 중단 기준

1. 누수·시점·비용·출처 위생 실패: 성과와 무관하게 결과 무효. 결함을 수리하고 다시 검증한다.
2. 고정 후보가 같은 대조군보다 상승 기회·net에서 일관되게 악화: **해당 설정 반증**, 즉석 sweep 중단.
3. 상승 유지·하방 감소·net 방향이 함께 유망: **오프라인 유망**까지. 새 forward 없이 운영 채택하지 않는다.
4. 지표 간 손익교환·넓은 불확실성·표본 부족: **증거 부족**. 0 포함 CI는 무효과 증명이 아니다.
5. 미래 검증: 승인·동결한 한 후보를 이후 도래하는 장전 슬롯에서 결과 전 기록한다.
   반복 관찰한 기존 35일이나 뒤늦게 계산한 점수를 새 forward로 바꾸어 부르지 않는다.

"몇 일 후 반드시 개선 확정"은 약속하지 않는다. 의미 있는 개선량·허용 위험폭과 실제 관찰 변동에 따라
추가 기간을 정하며, 관측치가 안 좋다는 이유로 평가 목표/기간/후보를 몰래 바꾸지 않는다.

### 구현·검증 진행

- [x] 연구자·평가자·운영 담당자의 읽기 전용 설계 검토.
- [x] 원 snapshot 전체 후보 기준 전 달력일 타깃 가용성 감사 설계 PASS.
  설계: `_workspace/recommend_horizon_coverage_design_20260907_v1.json`.
- [x] `scripts/audit_recommend_horizon_data.py` 구현: 원 후보 보존, 정확 창 검사,
  누락 사유·기존 fold별 양 타깃 가용시각·전일 open 입력 대체의 저장값 동일성 진단.
  새 타깃 export·학습·새 class-rate/성과 분석·DB 호출은 하지 않는다.
  원 readiness의 기존 class-count는 출처 요약으로만 보존한다.
- [x] 독립 코드 검토·회귀 테스트·실자료 실행·별도 숫자 검산.
- [x] 읽기 전용 D1 재구성의 저장 피처/유니버스 일치와 역사 재현 한계를 증거로 기록.
- [ ] 사용자 승인 후에만 실행용 비교 설정을 확정하고 B/C 학습을 시작한다.

### frozen 기록 가용성 — 실측·독립 검산 완료

결과: `_workspace/recommend_horizon_coverage_20260907_v1.json`.

- 원 장전 후보 **3,500행·35일**을 모두 보존했다. 현재 결과 가용은 3,497행, halt는 3행이다.
- 전 달력일 장전 타깃은 2,594행에서 가용. 전일 snapshot 부재 300행,
  전일 유니버스에 해당 종목 부재 606행이다. 양쪽 결과가 모두 있는 행은 **2,593행·32일**.
  100개 후보가 모두 연결된 날짜는 **0일**이다. 이 교집합을 원천 전체 학습 표본처럼 사용하지 않는다.
- 기존 readiness fold의 실제 최초 검증 의사결정 이전에 양 타깃이 모두 가용한 학습행은
  498/908/1,314/1,724/2,119행(6/11/16/21/26일)이다. 이는 학습 가용성 검사일 뿐,
  이 짧은 fold나 분할을 새 B/C 학습 설계로 채택했다는 뜻이 아니다.
- 전일 open에 같은 종목이 있던 경우는 2,886행, 저장 24피처가 전부 같은 경우는 400행뿐이다.
  전일 open 입력을 현 preopen 입력 대신 복사하면 안 된다. 차이의 원인은 이번에 확정하지 않았다.
- 독립 평가자가 원 snapshot/label/receipt 201파일에서 3,500행의 모든 진단 필드·35일 요약·
  5개 fold와 학습 identity hash를 별도로 재계산해 **오차 0**을 확인했다.
  보고서 payload checksum 및 연결된 generator/input **226개 identity**의 현재 SHA256도 일치했다.
- 신규 35개 테스트와 기존 슬롯 42개를 합쳐 **77 passed in 7.77s**, 신규 두 Python 파일 Ruff PASS.
  최종 전수 회귀는 **1,819 passed in 255.09s**. `git diff --check`도 통과했다.

### D1 재구성 — 교집합 자료의 한계를 우회할 원천 확인

증거: `_workspace/recommend_horizon_reconstruction_audit_20260907_v1.json`.
파일 SHA256: `04df21246a91ba43933cc2f00d78af1aed3069ac3d726a6818b9cdf0e132c1ec`.
이 JSON에는 별도 최상위 payload checksum이 없으며, 위 전체 파일 SHA를 감사 앵커로 사용한다.

- 단일 SQLite 읽기 transaction에서 DB 전체 **205,178행·290마켓**과 재구성에 실제 사용한
  09-07 이전 **204,896행**을 구분했다. 조회 OHLC 논리 해시는
  `37fbe422c30cc3e807c9054c66772c83ea556a43e912038331dd6a68c0d409f0`.
- 기존 시장 제외 규칙·70개 prior bar·동일 순수 피처 계산 후 **184,747행·262마켓**.
  장전 **35/35일의 Top100**, **3,500/3,500 후보**, **84,000/84,000 피처 셀**이
  저장 규칙 `round(value,8)` 기준으로 일치했다. 원래 full-precision 일치의 증명은 아니다.
- 첫/마지막 평가일의 기존 cutoff로 잘라 정확한 다음 달력일 타깃까지 연결되는 공통 학습 후보는
  **119,976행/1,204일 → 123,974행/1,244일**이다. 다음 달력일이 없는 terminal 행 1/3개는
  공통 학습 후보에서 제외 사유를 남겼다. 이 숫자는 두 끝점의 자료 준비도이며 학습 결과가 아니다.
- 소스 17개·고정 증거 211개의 전후/저장 전후 해시를 확인했다. exact SQL, 정렬·필터·함수·
  as-of·라이브러리 버전·실제 실행 Python 문자열·스크립트 해시를 JSON에 남겼다.
- 첫 출처 검사는 readiness 경로의 별칭 절대경로/상대경로 표시 차이로 중단됐다.
  바이트 해시·크기는 같았고, loader가 제공한 `provenance.root`를 사용해 재식별한 뒤 통과했다.
  **재발 방지:** 출처 identity 재검사는 임의 alias root로 재표시하지 말고 해당 manifest root를 사용한다.
- D1과 canonical 15m 결과가 모두 있는 **3,497행**에서 up10 1건·dn5 2건이 달랐고
  entry open은 7행 달랐다. 라벨 차이는 07-28 SAHARA/08-20 PLUME의 −5% 경계,
  08-22 DOT의 +10% 경계에서 분모 EPS/부동소수 연산 차이로 발생했다.
  **시간창 정합화가 원천 봉·수치 정의의 완전 동일성을 뜻하지 않는다.**
  후속 B/C 학습은 동일한 기존 D1 수치 정의, 평가는 동일한 원 canonical 15m 정의를 사용하고
  이 계약 차이를 명시해야 한다. 이번에 원 라벨을 재작성하거나 경계를 임의 반올림하지 않았다.
  up5/up20/dn10·기대 하방까지 canonical과 같은지 확인한 것은 아니다. 이 3개 경계 행이나
  진입가격이 다른 7행을 사후에 조용히 제외하거나 임의 tolerance로 뒤집지 않는다.
- 독립 평가자가 내장 스크립트 해시/compile·소스 17개/증거 211개를 확인하고,
  별도 읽기 전용 SQL로 **204,896행·논리 해시·전체 DB 통계·경계 3건·진입가격 7건 차이**를
  정확히 재현했다. 이 독립 검산에서 전체 피처 재계산이나 모델 학습을 실행한 것은 아니다.
- 달력상 결과 종료는 확인할 수 있지만 과거 DB 입고시각·당시 학습행/가중치·역사 정책은 미복원이다.
  현재 재구성 자료의 통제된 과거 비교는 가능하나, 정확한 과거 운영 재현/미래 성과 입증으로 부르지 않는다.

### 실제 미래 비교에 필요한 최소 보존 계약

- 현재 새 후보용 무학습·미발송 점수 기록 CLI는 없다. 기존 R1 snapshot wrapper는 모델 식별자·
  소스·DB 귀속이 R1에 고정돼 있어 custom callback만 넣어 후보 모델 기록기로 재사용하면 안 된다.
- 후보 확정 후에는 모델 파일/hash(보정 포함)·학습 행/입력/설정 hash·라벨 가용시각·피처 순서를 보존한다.
  같은 날짜의 원 R1 snapshot을 무학습으로 읽고 후보 점수·전체 순위·Top3를 별도 새 파일에 남긴다.
- 입력 API에는 허용 피처만 전달하며 결과 라벨은 나중에 별도 파일로 연결한다.
  예측 시작/완료/저장 시각이 평가창 시작 전이어야 하며, 놓친 날은 과거 재생으로 메우지 않는다.
- R1 Telegram receipt는 후보 점수 hash를 담지 않으므로 후보의 사전 기록 증명으로 빌려 쓰지 않는다.
  로컬 시계의 정확성·독립적 타임스탬프 증명은 별도 운영 검증 대상이다.
- 승인·동결 후 도래하는 새 날짜만 forward다. 새 후보 자동 학습/스케줄/알림/승격은 별도 승인 영역이다.

### 연속 작업 메모·승인 경계

- 이번 변경은 감사 CLI/테스트 Python 2파일, horizon 설계·가용성 결과·재구성 증거 JSON 3파일과
  이 PHASES 기록이다. 새 MD는 없으며 기존 dirty 산출물·`image.png`는 손대지 않았다.
- 설계·자료 준비도와 감사 도구 검증을 완료했다. **B/C 모델은 아직 학습하지 않았고 추천 개선 판정도 없다.**
- 사용자에게 "운영 유지, 비교용 장전 모델의 학습 대상 날짜 정합화·재학습 승인"을 별도로 질문했다.
  이후 사용자 `ㅇㅇ 당연 가야지` 응답으로 해당 오프라인 범위만 승인됐다. 현재 immutable 공통 학습자료 export와
  날짜별 모든 head의 B/C 동일 조건 학습·arm별 평가기를 구현·검증 중이다. 운영 승격은 승인 범위 밖이다.
- 이후 `진행`은 이 세 horizon JSON과 본 절을 이어서 사용한다. 과거 C 결과·교집합 2,593행을
  새 target-only 실험처럼 재사용하지 않으며, 필요한 승인·새 forward 관찰을 대체하지 않는다.

재실행 명령(기존 결과 덮어쓰기를 피하려고 `--output` 생략):

```bash
PYTHONDONTWRITEBYTECODE=1 PRELUDE_FORBID_TELEGRAM=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 venv/bin/python -B scripts/audit_recommend_horizon_data.py --input _workspace/recommend_training_readiness_20260907_v3.json --design _workspace/recommend_horizon_coverage_design_20260907_v1.json
TMPDIR=/home/soccz/22tb/tmp PYTHONDONTWRITEBYTECODE=1 PRELUDE_FORBID_TELEGRAM=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 venv/bin/python -B -m pytest -q -p no:cacheprovider tests/
ruff check --no-cache scripts/audit_recommend_horizon_data.py tests/test_audit_recommend_horizon_data.py
git diff --check
```

---

## 실사용 강화 3차 — 선정·시간대 분해 (2026-09-07)

사용자 "ㅇㅇ 가야지" 승인으로 **오프라인 진단 구현·실측**을 진행한다.
`prelude-quant`의 연구/독립 평가 분리로 아래 설계를 검토했으며,
**OFFLINE DIAGNOSTIC DESIGN PASS**는 추천 개선·승격 PASS가 아니다.

- [x] 기존 readiness v3의 6,700후보·67 snapshot만 재결합한다. 새 학습·수집·운영 변경 없음.
  전체 35일의 관찰 이력이며, 이전 outer 비교와 겹치므로 독립 검증 표본으로 합산하지 않는다.
- [x] 선정 진단: 실제 Top3 vs 전체 후보/동일 날짜·슬롯의 유동성·ATR 공동 사분위 매칭 기대값.
  기존 6지표·날짜 동일 가중·paired IID/3관측일 block CI·하루 제거 민감도를 사용한다.
  후보 선정·매칭을 먼저 확정한 후 결과 누락을 검사한다. 미지원 매칭 구간은 넓히지 않는다.
- [x] 순위 구조: 상방/저하방 백분위·RR floor·동점·3↔4 경계·고정 순위대
  (1–3/4–10/11–30/31+)만 진단한다. 로그 기여는 동일 snapshot의 mean(log) 중심으로
  가법성을 확인하며, 상방 확률 0이 있으면 임의 floor나 행 제외 없이 해당 로그 분해를 미지원 처리한다.
- [x] 시간대 3항 분해: P=preopen 실제 Top3의 preopen 평가창,
  Q=같은 preopen Top3의 open 평가창, R=open 실제 Top3의 open 평가창.
  **R−P=(Q−P)+(R−Q)**를 동일 날짜·각 3픽 가중으로 계산한다.
  미래 open 추천을 과거 preopen 창에 되돌리는 역방향 셀은 계산하지 않는다.
- [x] 독립 검토 조건 반영: 세 항의 일별·평균·공동 bootstrap 가법성 확인;
  선정 진단과 시간대 진단의 cohort/분모/제외 사유는 각각 기록한다.
  창의 시작과 끝이 함께 이동하므로 순수 진입 지연/입력 신선도의 인과효과로 해석하지 않는다.
- [x] 증거 한계: snapshot.training은 공통 base train 통계이지 결측 제거 후 head 표본 통계가 아니다.
  명목 입력일과 DB 전역 최신 시각을 코인별 실제 입력 시각으로 바꾸어 말하지 않는다.
- [x] 신규 선정 진단·시간대 진단·CLI 및 대응 회귀 테스트 구현.
- [x] 고정 설계로 실자료 실행 → 별도 평가자의 원본 독립 검산 → 결과 기록.
- [x] 후속 가설을 preopen 학습 타깃의 실사용 평가창 정합화로 좁혔다. 즉석 임계값 탐색/운영 배포 없음.

### 본 진단 결과와 반증

설계: `_workspace/recommend_selection_design_20260907_v1.json`.
결과: `_workspace/recommend_selection_diagnostics_20260907_v1.json`.
신규 소스는 `signals/recommend_selection_diagnostics.py`, `signals/recommend_slot_diagnostics.py`,
`scripts/diagnose_recommend_selection.py`와 대응 테스트 3파일이다. 기존 운영 소스는 수정하지 않았다.

| 슬롯·공통 성과 표본 | 실제 Top3 안전상승 | 매칭 기대 안전상승 | 실제 Top3 하락 | 매칭 기대 하락 |
|---|---:|---:|---:|---:|
| open 32일·96픽 | 17.71% (17/96) | 10.82% | 19.79% (19/96) | 20.67% |
| preopen 33일·99픽 | 11.11% (11/99) | 8.21% | 21.21% (21/99) | 19.66% |

- 안전상승은 기존 24h MFE≥10%이면서 MAE>−5%, 하락은 MAE≤−5%다.
  매칭은 같은 날짜·슬롯의 유동성·ATR 공동 사분위 내 기대 무작위 평균이다.
  preopen 08-04/09-01은 비추천 후보의 결과 부재 때문에 전체 후보와 공통 비교에서 제외했다.
- open 매칭 대비 안전상승 +6.89%p, 하락 −0.88%p; preopen은 +2.90%p, +1.55%p다.
  **두 슬롯 모두 해당 날짜 블록 CI95가 0을 포함한다.** 상승 후보를 더 잘 고르는 관측 신호를
  하방 감소까지 입증한 것으로 과장하지 않는다. 특히 preopen은 하락 차이 방향도 불리하다.
- 실제 Top3의 TP5/SL3 net 평균은 open −0.033%/preopen +0.232%,
  24h 보유 net 평균은 open +1.246%/preopen +0.497%다. 서로 다른 가상 청산 기준이며,
  매칭 대비 두 net 차이의 블록 CI95도 모두 0을 포함한다. 사용자 실제 수익이 아니다.
- RR floor 발동 **0/6,700행**. Top3의 평균 상방 백분위는 open 75.53/preopen 75.60,
  저하방 백분위는 60.17/59.62다. mean(log) 중심 기여도도 상방 1.03/1.05,
  저하방 0.28/0.27로, "하방 분모가 거의 0이라 순위가 폭발했다"는 설명을 지지하지 않는다.
- 각 Top3 행에 상방 저장 확률이 같은 다른 후보가 존재한 경우는 **201/201행**,
  재계산 RR이 같은 다른 후보가 존재한 경우는 **102/201행**이다.
  3↔4 경계 결정은 RR 33회/깊은 하방 확률 20회/원래 순위 14회다.
  마지막 14회는 **반올림 저장값에서의 동점**이며 실제 full-precision 운영 순서가 임의였다고
  확정하지 않는다. 기존 head가 raw 확률을 bucket 보정값으로 바꾸는 코드는 확인했다
  (`signals/recommend.py::_fit_rr_head`). 절편 보정만으로는 이미 같아진 값의 구분을 복원할 수 없다.
- snapshot만으로 시간대를 연결한 3항 분해는 **8일·각 24픽**만 가용했다.
  35일 중 open 증거 부재 3일, 양 슬롯이 있는 32일 중 preopen 종목의 open 유니버스 부재 24일.
  가용일이 07-31/08-12~16/08-20/08-22에 치우쳐 이 결과로 전체 시간대 우열을 판단하지 않는다.
- 독립 평가자가 분석 함수를 재호출하지 않고 원본에서 선정·매칭·순위대·로그·6지표·CI·LODO·
  P/Q/R·공동 재표집 인덱스까지 검산했다. 최대 차이 5.55e-17, 생성 근거 17개·입력 근거 211개
  SHA256 현재 일치. 신규 111개 포함 **전수 1,750 passed in 223.20s**.
  신규 6 Python 파일 `ruff check --no-cache`, `git diff --check` 통과.

### 실측 후 설계 보완 — 후보군 부재와 가격 경로 부재를 구분

8일만 남는 한계를 발견해, 이미 검증된 `_workspace/recommend_entry_delay_20260907_v3.json`을
추가 고정 입력으로 사용하는 **별도 보충 진단**을 설계했다. 새 데이터 수집/학습/원본 라벨 변경은 없다.
이 보고서는 당시 실제 Top3 201건의 +0/+15/+30분 각각 새 24h 경로를 보존한다.

- [x] 보충 설계 독립 검토 PASS: P/R의 +0분을 원 canonical과 직접 대조하고,
  Q는 open 평가창의 시작·종료와 **정확히 일치하는 셀**로만 정한다. 성과로 지연 시간을 고르지 않는다.
- [x] 원 67 snapshot의 실제 Top3와 지연 보고서 201행의 ID·날짜·슬롯·종목·순위가 일대일인지 확인하고,
  Q가 open 라벨에도 존재하는 모든 겹침은 직접 대조한다. 불일치는 보충 진단 전체를 차단한다.
- [x] 기존 지연 보고서의 all-delay 가용 플래그로 필터하지 않는다. 필요한 P/Q/R 셀만 확인한다.
  open 유니버스에 없는 Q 종목을 가짜 후보/점수로 삽입하지 않는다. 본 진단과 원본은 그대로 보존한다.
- [x] 보충 결과는 검증된 **저장 재생 결과의 재사용**이다. raw OHLC 본문은 없으므로 행 SHA를
  원 DB 재검산 증명으로 주장하지 않는다. 출처별 생성 시각과 이력 재사용 한계를 명시한다.
- [x] 별도 보충 도구·회귀 테스트 → 실제 실행·독립 검산 후 결과 기록.

보충 소스: `scripts/complete_recommend_slot_diagnostics.py`와 대응 테스트 1파일.
설계: `_workspace/recommend_slot_supplement_design_20260907_v1.json`.
결과: `_workspace/recommend_slot_supplement_20260907_v1.json`.
본 진단 소스·설정·결과는 수정하지 않고 보충 결과를 별도로 보존했다.

- 실제 Top3 지연 0 경로 **201/201**을 원 canonical 17필드·시작/종료 시각과 직접 대조했다.
  Q와 기존 open 라벨이 겹치는 **56/56**도 일치했고, 없던 Q 40건을 기존 지연 경로로 보완했다.
  Q 96건 모두 성과가 아닌 정확한 창 일치로 +15분 셀이 선택됐다.
- **32일·각 leg 96픽**으로 확장됐다. 제외는 open 증거가 없는 07-29/07-30/08-18의 3일뿐이다.
  원래 가용했던 8일의 종목과 모든 일별 지표는 그대로 일치한다.

| 동일 32일 비교 | 안전상승 | −5% 하락 | TP5/SL3 net 평균 | 24h 보유 net 평균 |
|---|---:|---:|---:|---:|
| P: preopen 실제 종목·원래 창 | 11.46% | 23.96% | +0.453% | +0.154% |
| Q: 같은 preopen 종목·open 창 | 10.42% | 23.96% | +0.386% | +0.244% |
| R: open 실제 종목·open 창 | 17.71% | 19.79% | −0.033% | +1.246% |

- R−P의 안전상승 +6.25%p·하락 −4.17%p는 관측상 유리하지만 TP5/SL3 net은 −0.487%p로 불리하다.
  24h net +1.092%p는 창 변화 +0.090%p와 같은 open 창의 선정 차이 +1.002%p로 분해된다.
  **모든 창/선정/전체 차이의 6개 지표에서 IID·블록 CI95가 0을 포함한다.**
  "15분 기다리면 개선" 또는 "open이 확실히 우월"로 해석하지 않는다.
- 8일과 32일의 일부 차이 방향이 바뀌었다. 이는 후보군 교집합으로 한정한 8일을 일반화하면
  안 된다는 실측 근거이며, 새 선정 규칙의 개선 결과가 아니다.
- 독립 평가자가 보충 도구/분석 함수를 쓰지 않고 먼저 계산한 32×6×6 일별 행렬과
  평균·IID/블록 CI·32개 하루 제거 결과가 **오차 0**으로 일치했다. 공동 재표집 인덱스 해시도 일치.
  생성·연결 근거 248개 identity의 현재 SHA256 일치, 보충 테스트 34개와 Ruff 통과.

### 후속 가설 우선순위 — 아직 운영 수정 아님

추가 코드 추적에서 **preopen의 학습 타깃과 실사용 평가창이 한 일봉 어긋남**을 확인했다.
연속 일봉을 가정하면 피처는 row t에 `raw.shift(1)`을 붙이고,
R1 head는 동일 row t의 high/open−1·low/open−1로 학습한다.
그런데 preopen은 D일 알림에서 reference row D−1을 사용하고, 실제 평가는 D일 09:00부터다.

- 학습 대응: D−2 입력 → D−1 09:00~D 09:00의 결과.
- 실사용 요구: D−2 입력 → D 09:00~D+1 09:00의 결과.
- 코드 근거: `scripts/univariate_precursor_lift_v1.py:230`의 shift,
  `signals/recommend.py:243`의 동일 날짜 OHLC 결합, `_rr_outcome_labels`의 동일 행 타깃,
  `_fit_rr_head`의 학습 y, preopen `feat_date`와 `today_all` 선택.
  무체결로 행이 빠지면 실제 직전 관측일은 명목 D−2보다 더 오래될 수 있다.
- 이는 코드상 시점 관계의 불일치다. **이것이 성과 차이의 원인인지, 정합화하면 개선되는지는 별도 검증**이다.
  저장된 상방 동점의 순서 복원은 보조 가설로 남기고, 우선 가설은
  **"preopen 전용 학습 타깃을 실제 사용하는 창과 맞추면 상승 기회 유지·하방 축소가 함께 가능한가"**로 둔다.
- 후속 비교 설계에서는 입력 가용시각·유니버스·피처·모델 설정을 통제하고, 타깃 종료와 검증 시작의
  시간 누수를 차단해야 한다. 현재 이력은 설계용이며 이후 새 날짜 검증 없이 운영 개선을 확정하지 않는다.
  라벨 정의/모델 학습·운영 전환은 사용자 컨펌 경계이므로 **이번 작업에서 변경하거나 실행하지 않았다.**
- 시그널 담당자와 독립 평가자가 이 시점 관계를 각각 확인하고, 다음 비교 설계의 우선순위에 동의했다.
  `shift(1)`을 제거하거나 아직 완성되지 않은 미래 일봉을 입력에 넣는 방식은 금지한다.
  다음 날 타깃은 실제 달력상 평가창으로 연결해야 하며, 무체결 누락 행을 무시한 단순 `shift(-1)`로
  대체하지 않는다. 학습 타깃 종료가 검증 의사결정 시각보다 앞선다는 조건도 별도로 검사해야 한다.

### 최종 검증·재실행·연속 작업 메모

- 이번 신규 테스트 145개(선정 37/시간대 42/CLI 32/보충 34)를 포함해
  **최종 전수 1,784 passed in 251.57s**. 신규 Python 8파일 Ruff와 `git diff --check` 통과.
- 본 진단의 67 snapshot·교집합 8일 분석과 보충의 32일 분석은 서로 겹치는 진단·범위 보완이다.
  별도 독립 실험이나 새 forward 검증으로 합산하지 않는다.
- 이번 변경: 신규 Python 8파일, 설계/결과 JSON 4파일, 이 PHASES 기록.
  기존 dirty 산출물·`image.png`는 보존했다. 새 MD, 운영 코드/알림/모델/라벨/DB/원장/스케줄 변경 없음.
- 다음 작업은 **preopen 학습 타깃–실사용 평가창 정합화 비교 설계**다.
  설계 검증과 사용자 승인 없이 라벨 정의를 바꾸거나 새 학습·운영 전환을 실행하지 않는다.

실제 실행·검증 명령(동일 결과 파일 덮어쓰기 방지를 위해 아래 재실행 예시는 `--output`을 생략):

```bash
PYTHONDONTWRITEBYTECODE=1 PRELUDE_FORBID_TELEGRAM=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 venv/bin/python -B scripts/diagnose_recommend_selection.py --input _workspace/recommend_training_readiness_20260907_v3.json --design _workspace/recommend_selection_design_20260907_v1.json
PYTHONDONTWRITEBYTECODE=1 PRELUDE_FORBID_TELEGRAM=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 venv/bin/python -B scripts/complete_recommend_slot_diagnostics.py --diagnostics _workspace/recommend_selection_diagnostics_20260907_v1.json --delays _workspace/recommend_entry_delay_20260907_v3.json --design _workspace/recommend_slot_supplement_design_20260907_v1.json
TMPDIR=/home/soccz/22tb/tmp PYTHONDONTWRITEBYTECODE=1 PRELUDE_FORBID_TELEGRAM=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 venv/bin/python -B -m pytest -q -p no:cacheprovider tests/
ruff check --no-cache scripts/diagnose_recommend_selection.py scripts/complete_recommend_slot_diagnostics.py signals/recommend_selection_diagnostics.py signals/recommend_slot_diagnostics.py tests/test_diagnose_recommend_selection.py tests/test_complete_recommend_slot_diagnostics.py tests/test_recommend_selection_diagnostics.py tests/test_recommend_slot_diagnostics.py
git diff --check
```


---

## 실사용 강화 2차 — 상방 비교 설계·반증 (2026-09-07)

사용자: "설계를 계속 객관화 해서 검증하면서 반영해". **오프라인 비교 실험 진행**으로 해석하며,
운영 R1·라벨·알림·스케줄·자동 주문·모델 배포는 변경하지 않는다.

- [x] 독립 설계 검토 **OFFLINE DESIGN PASS**. 이는 모델 우수성/승격 PASS가 아니다.
- [x] 설계 반영: 미래 halt 3행을 후보군에서 미리 빼지 않고 snapshot 전체 6,700행을 복원.
  학습에는 결과가 가용한 6,697행만 사용. 선택 후 결과 누락은 공통 날짜·슬롯 비교에서 차단한다.
- [x] 저장 확률 반올림 영향 실측: 67개 snapshot의 무변경 재정렬 Top3 순서 **67/67 일치**.
  전체 순위는 9개에서 달라 full-precision 복원으로 주장하지 않는다. 실험마다 Top3 일치 게이트 유지.
- [x] 보정 설계 축소: slope=1 logit 절편만 적합해 보정이 상방 순서를 뒤집지 못하게 함.
  B=기존 저장 확률의 과거 forward-score 재보정, C=24피처·기존 post-send up10의 얕은 XGB
  + inner expanding OOF 보정. A=실제 원본 추천. 하방 확률·R1 정렬식은 유지.
- [x] 35일 전체와 outer 검증 구간을 분리: 검증은 최대 open 19일/preopen 20일.
  과거에 이미 관찰한 이력이며 미래 미사용 검증이 아님. 기존 head와 C의 피처·학습기간·목표창이
  달라 "모델 구조만 바꾼 효과"로 해석하지 않는다.
- 사전 고정 설계: `_workspace/recommend_upper_design_20260907_v1.json`.
  입력 파일 해시·모델 파라미터를 학습 전에 확인. 단일 설정이며 결과를 본 뒤 sweep하지 않는다.
- [x] 원본 재결합 로더·B/C 학습기·동일 날짜 평가기·실행 CLI 구현 및 회귀 검증.
- [x] 고정 설정 실자료 실행 → 독립 재계산 → 결과와 한계 기록.
  설정은 한 종류만 평가했다. 최초 실행 후 저장 정밀도만 수정해 같은 설정으로 재실행했으며,
  두 결과의 configuration/evaluation/fold_audit/parity는 완전히 같다.

### 실측 판정 — 검증 도구는 반영, 후보 모델은 운영 미반영

**A 유지. B는 일관된 개선 근거 부족으로 보류, C는 이번 고정 설정의 운영 대체안으로 기각.**
설계·구현 검증 통과와 추천 품질 개선은 별개다. C가 하락을 덜 겪는 종목을 골랐다는 사실을
상승도 더 잘 찾았다는 뜻으로 해석하지 않는다. 모든 상방 재학습 방식이 실패한다는 판정도 아니다.

- 검증 대상은 4개 outer fold의 20고유일/39개 날짜·슬롯이다. 09-01 preopen은
  KRW-TT 결과가 없어 전 유니버스 baseline을 계산할 증거가 부족했다.
  이 날짜·슬롯을 모든 arm에서 공통 제외해 **open 19일·57픽, preopen 19일·57픽**으로 비교했다.
  B/C는 8개 fold·슬롯 모두 예측 가능했고, 실패한 모델을 A로 대체한 경우는 없다.
- 아래 비율은 선택된 Top3의 사후 관측 빈도이지 사용자에게 보장할 확률이 아니다.
  안전상승 = 기존 평가 시작 후 24시간 동안 MFE ≥ +10%이면서 MAE > −5%.
  하락 = 같은 창에서 MAE ≤ −5%. 안전상승은 TP 선도달과 다른, 전체 경로 진단 지표다.
  수익은 기존 비용 0.15%가 이미 차감된 가상 경로 수익이며 추가로 차감하지 않았다.

| 슬롯·안 | 안전상승 | 하락 | TP5/SL3 net 평균/픽 | 24h 보유 net 평균/픽 |
|---|---:|---:|---:|---:|
| open A 현재 추천 | 24.56% (14/57) | 19.30% (11/57) | +0.367% | +2.930% |
| open B 확률 보정 | 21.05% (12/57) | 21.05% (12/57) | +0.482% | +2.237% |
| open C 새 상방 head | 10.53% (6/57) | 10.53% (6/57) | +0.385% | +1.446% |
| preopen A 현재 추천 | 10.53% (6/57) | 21.05% (12/57) | +0.612% | +0.885% |
| preopen B 확률 보정 | 12.28% (7/57) | 19.30% (11/57) | +0.819% | +0.737% |
| preopen C 새 상방 head | 7.02% (4/57) | 5.26% (3/57) | +0.786% | +1.089% |

- C open은 4개 fold 중 3개에서 상승·24h net 모두 A보다 낮았다.
  하루씩 제외한 19번 비교에서도 상승·24h net 차이가 모두 음수였다.
  안전상승 차이 −14.04%p의 3일 이동블록 bootstrap CI95는 [−29.82, −1.75]%p다.
- C preopen은 4개 fold 모두 하락 빈도가 줄었지만 상승 빈도도 전체 비교에서 줄었다.
  24h net +0.204%p 차이의 CI95는 0을 포함한다. B preopen의 안전상승·하락 개선은
  각각 단 1건 차이여서 우수성으로 확정하지 않는다.
- 동일 날짜의 유동성·ATR 구간을 맞춘 무작위 기대 baseline 대비 C 안전상승 차이는
  open −2.29%p/preopen −0.11%p. C의 매칭 baseline 대비 TP/24h net 차이는
  두 슬롯 모두 CI95가 0을 포함해, 비슷한 변동성·유동성 안에서 상방 우위를 입증하지 못했다.
- 통계는 날짜 동일 가중·날짜 paired CI·3관측일 이동블록 CI·fold·하루/종목 기여도를 병기한다.
  3관측일은 연속 달력 3일을 보장하지 않는다. 후보 수천 행을 독립 표본으로 세지 않으며,
  비추천 후보의 결과는 과거 재생 평가다. 실제 매매 PnL·포트폴리오 Sharpe·새 미래 검증이 아니다.

### 이전 진단의 적용 범위 수정

현재 입력 전체(실제 포함 07-28~09-06, 35고유일)의 기존 저장 확률과 canonical label을
별도로 재계산했다. 아래는 위 outer 비교 구간과 다른 **전체 이력 pooled 진단**이다.

| 슬롯·모집단 | 행 수 | 기존 p_up10 AUC | 기존 p_dn5 AUC |
|---|---:|---:|---:|
| open 전 후보 | 3,200 | 0.718 | 0.764 |
| open 실제 Top3 | 96 | 0.624 | 0.599 |
| preopen 결과 가용 후보 | 3,497 | 0.664 | 0.713 |
| preopen 실제 Top3 | 105 | 0.560 | 0.576 |

과거 기록의 "상방 AUC 0.477, 죽은 head"를 오늘의 상태로 재사용하지 않는다.
시기·표본·평가 계약이 다르므로 위 차이를 모델 개선의 인과 증거로 주장하지도 않는다.
전체 후보에는 상방 판별력이 관찰되지만 실제 Top3, 특히 preopen에서는 더 약하다.
다음 설계는 head 교체를 전제로 삼지 말고 **현재 점수의 Top3 선정 손실과 슬롯별 차이**를 먼저
분해한다. Top3의 작은 표본·선택 편향 때문에 AUC만으로 정렬 결함을 확정할 수는 없다.

### 구현·검증·연속 작업 메모

- 추가 소스: `signals/recommend_experiment_data.py`(원본 재결합),
  `signals/recommend_upper_experiment.py`(고정 비교 학습),
  `signals/recommend_experiment_eval.py`(공통 표본 평가),
  `scripts/compare_recommend_upper.py`(검증·실행 CLI), 대응 테스트 4파일.
- 반영한 안전장치: 후보 선택 후 결과 확인, 날짜·라벨 가용시각 기준 누수 차단,
  진짜 inner OOF, 미학습/예측 불가와 손상 구분, 학습 전/직렬화 후 입력·소스 해시 재확인,
  원본·기존 보고서 덮어쓰기 방지. 모델 객체는 메모리 내 실험에만 사용하고 아티팩트는 저장하지 않는다.
- JSON 기본 10자리 반올림이 −5% 경계 부근 값을 바꾸는 사례를 발견해 부동소수 원정밀도와
  명시적 null을 보존하도록 수정했다. v1은 감사용으로 남기고
  **`_workspace/recommend_upper_comparison_20260907_v2.json`을 최신 결과로 사용**한다.
- 독립 검산은 모델/평가 함수를 재호출하지 않고 원본 snapshot·receipt·label 201파일과
  저장 예측에서 Top3·baseline·6개 지표·CI·fold·하루 제거·AUC/Brier를 재계산했다.
  최종 차이는 최대 1.11e-16, 경계 판정 불일치 0건, 비용 이중 차감 0건.
  현재 생성 근거 16파일과 입력 근거 211파일의 SHA256도 일치했다.
- **최종 전수 pytest 1,639 passed in 237.64s**, 이번 신규 123개 포함.
  신규 Python 8파일 `ruff check --no-cache`, `git diff --check`도 통과했다.
- 운영 R1·알림 문구·라벨·원장·DB·systemd/cron·실제 Telegram 전송은 변경하지 않았다.
  기존 `output/` dirty 변경과 `image.png`는 그대로 보존했다. 별도 새 MD는 만들지 않았다.
- 다음의 제한된 작업: 동일 날짜·변동성·유동성 조건에서 실제 Top3와 전체 후보의 차이 및
  open/preopen 불일치를 진단해 하나의 반증 가능한 개선 가설을 정한다.
  이번 결과에 맞춰 즉석 sweep하지 않는다. 후속 후보는 설정 고정 후 비교하고 새 날짜에서
  재확인하며, 라벨/구조/알림/자동 학습·배포 변경은 별도 승인 경계를 유지한다.

```bash
# 고정 설계 재실행: 기본 stdout 출력, 기존 보고서 덮어쓰기 없음.
PYTHONDONTWRITEBYTECODE=1 PRELUDE_FORBID_TELEGRAM=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 venv/bin/python -B scripts/compare_recommend_upper.py --input _workspace/recommend_training_readiness_20260907_v3.json --design _workspace/recommend_upper_design_20260907_v1.json
TMPDIR=/home/soccz/22tb/tmp PYTHONDONTWRITEBYTECODE=1 PRELUDE_FORBID_TELEGRAM=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 venv/bin/python -B -m pytest -q -p no:cacheprovider tests/
jq -r '.generator_files | to_entries[] | "\(.value.sha256)  \(.value.path)"' _workspace/recommend_upper_comparison_20260907_v2.json | sha256sum -c
jq -r '.data_provenance.files | to_entries[] | "\(.value.sha256)  \(.value.path)"' _workspace/recommend_upper_comparison_20260907_v2.json | sha256sum -c --quiet
```

---

## 실사용 강화 1차 — 설계 검증 후 오프라인 구현 (2026-09-07)

**현재 판정: 측정·자료 준비 도구의 설계 PASS. 추천 품질 개선·모델 승격 PASS가 아니다.**
사용자 요청은 "먼저 설계하고, 설계를 검증하고, 검증되면 진행"이다.
운영·경로평가·학습자료를 나눠 설계한 뒤 독립 검토를 거쳐 아래 범위만 구현했다.
이 절의 실측은 현재 폴더의 저장 증거 기준이다. 아래 2026-08-08 요약은 과거 기록이며,
당시 표본 수·설치 상태·AUC를 오늘의 상태로 재사용하지 않는다.

### 설계와 허용 범위

| 도구 | 사용자에게 필요한 답 | 검증 조건 |
|---|---|---|
| A. 추천 상태 확인 | 추천이 만들어졌는가, 서버가 받았는가, 후보가 몇 개인가 | snapshot·receipt·후보 수를 분리. receipt 없음은 미발송 확정이 아니라 확인 불가 |
| B. 늦은 확인의 영향 | 같은 추천을 늦게 보았을 때 결과가 달라지는가 | 기존 기준 시점 대비 +0/+15/+30분, 각각 새 24시간. 실제 발송 Top3의 같은 날짜·종목을 비교 |
| C. 후속 학습 자료 준비 | 기존 기록으로 누수 없이 비교 학습할 준비가 되었는가 | 저장된 24피처만 사용. 결과·순위·발송 여부는 입력에서 분리. 날짜 단위로 양 슬롯 함께 분할 |

- **수정하지 않은 것:** R1 정렬·확률·모델·라벨, 기존 알림 문구, 원장·DB,
  자동 학습·배포·주문, systemd/cron 및 `/etc` 설치본.
- **실행 방식:** 새 CLI를 사람이 호출하는 오프라인 도구. 전송·수집·재학습 없이 읽기만 한다.
  `--output`을 명시한 경우에만 새 보고서 파일을 만들며 기존 파일은 덮어쓰지 않는다.
- **공통 증거 계약:** 현재 R1 snapshot·서버 receipt·현대 경로 label의 ID/해시/후보/비용/시간을
  교차 검증한다. 누락·레거시·아직 미성숙한 결과는 이유를 남기고 제외하며,
  손상·모순은 분석을 차단한다. 파일 변경 전후와 분석 코드의 해시를 남긴다.
- **시간 안전성:** 슬롯 내 `decision_started <= decision_completed <= attempted_at`.
  학습에는 `max(경로 종료, 현재 label 생성시각) < 검증일의 첫 추천 계산 시작`인 날짜만 포함한다.
  사후 재생성된 label의 과거 가용시각을 추정해 앞당기지 않는다.
- **경로 안전성:** SQLite 읽기 전용의 동일 transaction에서 OHLC와 경로 완결성 근거를 읽는다.
  B의 +0분 결과가 기존 canonical label과 다르면 전체 통계를 차단한다.
  무체결 flat 처리와 장애 구분은 기존 path-quality 계약을 재사용한다.
- **해석 경계:** 피처 행 날짜는 실제 마지막 입력봉 시각과 다르다. 코인별 실제 시각은
  저장 증거가 없어 "확인 불가"로 표시한다. 서버 수락은 사용자의 읽음 확인이 아니다.
  후보 0개를 "품질 기준 탈락"으로 바꿔 말하지 않는다.

### 구현·실측 결과

- A: `ops/recommendation_status.py`, `scripts/report_recommendation_status.py`.
  09-07 저장 receipt 기준 preopen **08:53:09**, open **09:08:07 KST**, 각각 후보 3개.
  결과: `_workspace/recommendation_status_20260907_v1.txt`.
- B: `scripts/evaluate_recommend_entry_delay.py`.
  07-26~09-06 요청 구간에서 open **32일·96픽**, preopen **35일·105픽**.
  **201픽 모두 +0분 canonical 일치**. 실제 포함 자료는 07-27 이후다.
  legacy snapshot 1개·receipt 미확인 6개를 명시적으로 제외했다.
  결과: `_workspace/recommend_entry_delay_20260907_v3.json`.
  지연은 **canonical 평가 시작에서 추가한 시간**이지 문자 수신 후 정확한 분 단위 체결이 아니다.
  주요 지연 차이의 신뢰구간은 0을 포함해 추천 유효시간·최적 진입시각을 정할 근거는 부족하다.
- C: `signals/recommend_training_data.py`, `scripts/build_recommend_training_data.py`.
  정상 발송 증거 67 snapshot의 전 유니버스 **6,697행·35고유일**을 분리했다.
  open 3,200행/32일, preopen 3,497행/35일. 실제 Top3 표식은 201행.
  무관측 정지 3행은 음성 라벨로 만들지 않고 제외했다. 24피처 결측 0개.
  ret3d/roc3d와 ret7d/roc7d는 각각 전 행에서 동일해 독립 정보 4개가 아니다.
  진단 초기값 train 10일/validation 5일 기준 5분할 중 4개가 자료 분할 가능,
  첫 분할은 시간 누수 제거 후 train 9일이어서 부족으로 표시된다.
  **6,697개의 독립 검증이 아니며, 이미 관찰한 35일을 미사용 holdout으로 주장하지 않는다.**
  `model_fitted=false`, `deployable=false`, `promotion_status=NOT_EVALUATED`.
  최신 결과: `_workspace/recommend_training_readiness_20260907_v3.json`.
- 공통 모듈: `ops/recommendation_evidence.py`. 독립 구현 검토에서 발견한
  생성/발송 시간 역전·구 score schema 허용·손상 snapshot의 누락 분류 가림을 수정하고 회귀 테스트했다.
- 실제 JSON 검토에서 C 비용 메타데이터의 중복 단위 변환을 발견해 수정했다.
  현재 표기는 decimal **0.0015 = 0.15%**, 결과 수익은 기존 net label 복사로 추가 차감하지 않는다.
  최초 C 보고서는 `_workspace/recommend_training_readiness_20260907_v1_superseded_cost_metadata_bug.json`으로
  구분해 보존했다. 비용을 고친 v2 이후 출력 경로 보호를 추가한 **v3를 최신본으로 사용**한다.
  기존 운영 수익·원장에는 영향이 없었다.
- 마지막 독립 검토에서 새 보고서가 아직 없는 원본 파일명을 선점할 수 있음을 발견했다.
  기본 snapshot/label/receipt 폴더와 CLI 사용자 지정 입력 폴더를 출력 금지 대상으로 추가했고,
  세 도구 모두 해당 원본 경로 오염 방지 회귀 테스트 및 독립 재검토를 통과했다.

### 검증과 재실행

- [x] 구현 전 독립 설계 검토: 오프라인 범위 PASS, 모델 배포·알림 변경은 범위 밖.
- [x] 신규 테스트: 증거 연결/손상/시간 역전, 발송 불확실성, +0분 일치 차단,
  지연별 동일 표본, 날짜 누수 방어, 무학습·읽기 전용, 덮어쓰기 방지, 비용 단위 계약.
  최종 신규 4개 테스트 파일 **114 passed**.
- [x] 현재 폴더의 실제 자료로 A/B/C 실행 및 새 결과물 확인.
- [x] 최종 소스 전수 pytest **1516 passed in 222.10s** (신규 114개 포함).
  신규 Python 10파일 `ruff check --no-cache`와 `git diff --check` 통과.
  B/C 최신 보고서의 코드 해시를 `jq -r ... | sha256sum -c`로 현재 파일과 대조해
  각각 11개/9개 모두 일치 확인. 실제 결과의 비용·행수·일수·미학습 상태도 `jq -e`로 확인.

```bash
# 아래 명령은 프로젝트 루트에서 실행. 재실행은 기본 stdout만 사용해 기존 보고서를 보존한다.
PYTHONDONTWRITEBYTECODE=1 PRELUDE_FORBID_TELEGRAM=1 venv/bin/python -B scripts/report_recommendation_status.py --asof 2026-09-07 --format text
PYTHONDONTWRITEBYTECODE=1 PRELUDE_FORBID_TELEGRAM=1 venv/bin/python -B scripts/evaluate_recommend_entry_delay.py --start 2026-07-26 --end 2026-09-06
PYTHONDONTWRITEBYTECODE=1 PRELUDE_FORBID_TELEGRAM=1 venv/bin/python -B scripts/build_recommend_training_data.py --start-date 2026-07-26 --end-date 2026-09-06
TMPDIR=/home/soccz/22tb/tmp PYTHONDONTWRITEBYTECODE=1 PRELUDE_FORBID_TELEGRAM=1 venv/bin/python -B -m pytest -q -p no:cacheprovider tests/
```

### 다음 단계와 멈춤 조건

- **다음 설계 대상:** 현재 저장 피처·기존 라벨로 상방 head 비교 학습과 진짜 expanding OOF
  calibration. 양 슬롯의 시간 차이, 변동성·유동성 조건별 성과, 하방 악화 여부와
  비용 차감 결과를 함께 비교하고, 이후 새로 쌓이는 날짜에서 재확인한다.
- **이번 완료선:** 위 세 도구와 검증 가능한 입력 준비까지. 자동 retrain/모델 구조/라벨/알림
  변경은 별도 사용자 승인 전 진행하지 않는다. 데이터 준비 통과를 추천 안전성 보장으로 바꾸지 않는다.
- **연속 작업 메모:** 기존 `output/` dirty 변경과 `image.png`는 이 작업 이전 상태로 보존.
  이번 소스는 새 Python 10파일(공통 모듈·3도구·4테스트), 문서는 이 절만 추가했다.
  systemd 설치·외부 네트워크·실제 Telegram 전송은 수행하지 않았다.

---

## 현재 상태 (요약, 2026-08-08)

- **현재 운영 알림**: R1 preopen/open 발송. pump-v2는 2026-08-05 조기 KILL로 종료·동결. 사용자가 직접 최종 매매하며 자동 주문은 없음
- **현재 판정**: radar-not-strategy. 새 snapshot 계약 이전 historical forward에서 R1 상방 head AUC 0.477, 비용 차감 성과도 음수라 우수 추천기로 주장할 수 없음
- **검증 방향**: 알림 이후 경로의 저하방·고상방 추천 품질과 비용 차감 결과를 전 유니버스 forward로 함께 측정
- **Track 1 + 적대 감사 완료**: 단일 snapshot·receipt·실행시각 경로·원자적 ledger·PIT universe·exact freshness·source provenance·terminal verdict·versioned backup 구현. 현재 signal-eligible 266개(명시 stablecoin 5종 제외), open PIT Top100 D1/4h `100/100`
- **실검증**: 최종 전수검증 진행 중. SQLite 7개 `quick_check=ok`, R1/R2/A1 open snapshot 각 100행과 R1 delivery receipt 검증. 새 계약의 complete label은 아직 0개
- **남은 운영 조치**: 저장소 systemd 소스는 검증됐지만 `/etc` 설치본 15개가 stale/missing이고 `.env`의 `PRELUDE_DASHBOARD_PIN`도 설치 전제다. sudo preflight·재설치 전에는 내일 운영 반영으로 간주하지 않음
- **확률 주의**: 독립 head의 포함관계 위반과 R1 RR 낙관 편향을 확인. 활성 모델·정렬·표시는 변경하지 않았고 calibration 재구축은 사용자 승인 후
- **변경 경계**: 활성 R1 정렬·라벨·모델 구조·알림 문구와 아래 09-01 동결 판정은 변경하지 않음. 상방 head 재구축은 승인 및 모라토리엄 종료 후
- **AI quant 포트폴리오 강화**: historical recommendation-quality meta-filter + model card + idea attribution scorecard 추가
- **학습 결과**: `recommendation_quality_meta_label_v1` 학습 완료. holdout 는 손실 축소/정밀도 개선 신호가 있으나 selected n=1 이라 자동 배포는 보류(shadow scoring)
- **아카이브**: detector_v1 / legacy 6-class 모델은 보존, 현재 메인 운영 문맥에서는 후순위

진행 트랙:
- [x] Phase 0 — 8 MD 설계 완료
- [x] Phase 1 — 데이터 수집 / 6-class 모델 / 인프라 (legacy)
- [x] Phase X — leak 발견 → detector 재정의 → C3 채택 → detector_v1 artifact (historical)
- [x] **Phase X+2** — dashboard publish 파이프 (paper_ledger → soccz.github.io 정적 회고) 2026-05-07
- [x] **Phase X+3** — 운영 안전 (DB 백업 + heartbeat 모니터링) 2026-05-26
- [x] **Stage 1 구조 전환** — shadow/paper ledger + ACTIVE-only Telegram + idea validation dashboard
- [x] **Stage 1+ AI quant layer** — recommendation-quality meta-filter + model card/report
- [ ] **Stage 2 live paper 축적** — systemd timer 운영 후 live shadow 표본 확보
- [ ] **Stage 3** (NOTES 기반 사용자 실제 매매 vs system 추천 비교)
- [x] **Phase X+1 초안** — Distribution head (multi-target) + pre-open trigger 운영 후보
- [x] **Phase X+4** — 정책 채택 (distribution PROMOTE_PAPER + preopen DEMOTE) 2026-05-26 사용자 컨펌
- [x] **Phase X+5** — PUMP hunter rule detector SHADOW 배선 2026-06-04
- [x] **Phase X+6** — exit lab (멀티 청산 잣대) + 운영 강화 (ledger 백업·비용 단일화·CSV 검증) 2026-06-11
- [x] **Track 1 측정 무결성** — 단일 snapshot/receipt, 전 유니버스 label/evaluator, path·비용·집계·수집·실패 전파 수리 2026-07-25
- [x] [Research] Cold-start 장중 펌프 (15m 미세동학) — **REJECT** 2026-06-11 (OOS allpick net 전부 음수, robust root 0; _workspace/coldstart_signal-researcher_pump_v1.md)
- [x] [Research] Day-quality gate (죽은 날 침묵) — **REJECT** 2026-06-11 (죽은날 2.8%뿐, replay 개선 permutation p=0.43 noise; 클러스터링 corr 0.52 는 진짜 — pump20 axis 재고 조건)
- [x] [Research] Binance lead-lag (volsurge) — **lift 진짜·배선 보류** 2026-06-11. researcher lift 4.34x (5/5 fold) → evaluator 가 "+1.24% net" 을 일봉 낙관 근사 환상으로 REJECT → 15m 정직 경로 재계산: 전기간 -0.11% (CI 0 포함), **최근 7개월 순수 OOS -0.357% (CI 0 제외), bear_quiet -0.19%**. hit 8.1% vs baseline 5.6% — 증분 피처로는 유효. 다음 주간 retrain 시 b_vol_surge 피처 후보 (모델 변경 = 사용자 컨펌 사안). 독립 룰 배선은 net 양수 확인 전 보류.
- [x] [Research] 델타-사다리 청산 (Veritasium "Trillion Dollar Equation" → Thorp/BSM dynamic 델타헤지 합성) — **SHADOW** 2026-06-25. 4-cell ablation(A=챔피언TP5/SL3, L3/L5=사다리 floor3/5, S5=단일TP5/SL5; 같은 OOF 픽셋 K=3 n=3732, 420일, purged WF embargo5). **핵심: floor 매칭 L5 vs S5 — deep-loss 빈도 0.491→0.303 (-18.7pp, 38%↓) · net +0.0001(동등) · hit +6.4pp, 대신 mean_win -1.7pp(상방 절단)**. = "평균 승자 희생 ↔ deep-loss 빈도↓" 하방-우선 reshaping(downside-first 정합). 비용 robust(conservative+extra5bp 에서도 net차 -0.0005). **ADOPT 아님** — 챔피언 -3% floor 가 절대 하방은 더 타이트(deep-loss 0), 전 cell net 음수(radar-not-strategy 재확인). per-fold 4/4 음수·forward 부재 → ADOPT 상한 SHADOW. quant-reviewer 적대검증 PASS(leak/metric/cost/재현성). 구현: `ledger/exit_lab.py`(walk_ladder_path 외), `scripts/ladder_exit_compare_v1.py`, `tests/test_exit_lab.py`(단일-arm 환원 동등성 회귀 14/14). 설계: `_workspace/ladder_exit_design_v1.md`. 결과: `output/ladder_exit_{compare,perfold,coverage}_v1.*`. 발상: `_workspace/veritasium_trillion_eq_ideas_v1.md`.
- [x] [Research] 군중쏠림 강등 (Veritasium Newton 과확장 함정 → 이미 오른 코인=칼받기) — **SHADOW** 2026-06-25. crowd_index=mean(return_7d/roc_3d/return_5d rank, D-1), honest EOD net(next_close, max-bracket 금지), 5010픽/753일. **full-sample 강함**: HIGH-LOW net차 -0.0126 CI[-0.019,-0.006] 0제외 · deep +0.158 · within-date perm **p=0.002**(day-quality p=0.43 와 달리 통과) · bull_quiet 유의 · score 3밴드 전부 separate(corr(crowd,score)=+0.18=모델 과열 편향). **그러나 quant-evaluator 적대검증에서 운영 생존성 미확립**: (1) deep차(+0.158 최강 주장)는 운영 -3% SL 이 전부 절단 → bracket deep차 **0.000**·net차 -0.0033 CI 0포함, (2) 효과가 **PRE-2025-05(alt 15m 無 구간)에 집중**, 15m 존재 최근 OOS 에선 EOD 효과 자체가 noise(perm **p=0.67**). leak 4대 PASS(binance 낙관근사·day-quality permutation noise 와 다른, 처음부터 정직 EOD). → **ADOPT 아님**(forward 부재 + bracket 소멸 + 최근 OOS noise). DOWNRANK **침묵 배선 보류**(33% 침묵해도 net 0=여전히 못 범). 권고: recommendation_quality record-only flag `crowd_overext=1` + shadow ledger crowd_q 컬럼만, alt 15m 누적 후 forward bracket HIGH-LOW CI 0제외+perm p<0.05 재확인 시 ADOPT 승격. 구현: `scripts/crowding_decay_v1.py`(+`crowding_decay_15m_crosscheck.py`), 판정: `_workspace/crowding_decay_quant-evaluator_verdict.md`, 결과: `output/crowding_decay_{v1,coverage_v1,15m_crosscheck}.*`.
- [x] [Research] Self-impact decay (발사 ACTIVE vs WATCH forward 차 — 영상 효율시장 자가소멸) — **INSUFFICIENT_SAMPLE** 2026-06-25. 하네스(`scripts/self_impact_decay_v1.py`, candles forward 조인) 빌드 완료·재실행 가능. 현재: 30일·전구간 bear(bull 0)·ACTIVE 20(A_TRIPLE) vs WATCH A_TRIPLE **5**. naive ATT(A_TRIPLE)=-0.0341 이나 conf_gap=+47.1 → self-impact(음)와 selection(양) 혼재 분리불가. 힌트: ACTIVE fwd_max +9.9%(레이더는 펌프 포착) vs EOD -5.85%(튀고 덤프) — self-impact 시그니처 후보. 게이트: ACTIVE n≥50 & WATCH A_TRIPLE n≥30(~2-3개월 누적) 후 conf-매칭/RD 로 selection 통제 재추정. `output/self_impact_decay_{v1,coverage_v1}.*`.
- [x] [Research] Exit-timing autopsy (언제 파나 — self_impact 의 +9.9%max→EOD덤프 정조준) — **결론: exit 는 이미 frontier, 변경 불요** 2026-06-25. 장중 autopsy(K=3, 3732픽 420일): 고점 max p50 +5.9%/p90 +21%, **t_max 중앙 0.06(고점은 장 초반에 찍힘)**, EOD p50 −3.4%(이후 하루종일 흘러내림), touched+5% 후 P(EOD>0)=0.54(스파이크 안 유지=코인플립). 청산 토너먼트 13정책: hold_eod −2.14%(최악) → champ TP5/SL3 −0.47%(deep 0) → bestnet TP5/SL8 −0.17%(deep 0.32, −5% 초과로 사용자 성향 위배). **champion 을 net↑·deep≤ 로 동시개선하는 정책 없음** — 사다리(246/123)·부분익절(half5)·trail·TP2~4 전부 못 이김(tp4_sl3 −0.0044 = +0.03pp 노이즈). → **downside-first frontier 에서 champion TP5/SL3 이미 효율적, exit lever 소진. 잃는 건 exit 아니라 진입 엣지.** 사용자 수동매매 규칙: 고점은 장 초반, +5% 익절, EOD 까지 들지 말 것(hold_eod −2.1%). 구현 `scripts/exit_timing_autopsy_v1.py`, `output/exit_timing_autopsy_{v1,coverage_v1}.*`.
- [x] [Research] PRPC 펌프-후 reclaim 확인진입 (entry-timing 마지막 미답축, (b) 데이터/TF 전환) — **DEAD** 2026-06-25. 설계 워크플로(11에이전트) 4후보 중 선정: P4HS(4h돌파) DEAD·SWING-DD(C1+C3 재포장) DEAD·LVG-XS(저변동 시장중립 excess) WEAK·PRPC 선정. kill-test(`scripts/prpc_kill_test_v1.py`, 4h 3.2년, d1 펌프≥12%→과열해소·변동성수축·higher-low consolidation→4h 재돌파, purged 4-fold block-bootstrap): armed 2616/reclaim 825(표본 충분). **B 재돌파진입 net +0.0013 CI[−0.007,+0.009] 0포함·foldPos 2/4 → DEAD. A 즉시눌림목진입(전 armed clean) net −0.0061 CI[−0.011,−0.0002] 유의 음수.** 초기 대조서 즉시진입 +4.4% 는 reclaim-subset 조건부 leak(미래상승 조건)이었고 clean 전수서 소멸. → **entry-timing(관측-후-확인/눌림목)도 exhausted. pump-then-revert 는 timeframe-invariant(4h=일봉=15m), 천장은 진입타이밍 아닌 시장구조.** ★(b)에서 발굴된 유일 실제 엣지 = **LVG-XS cross-sectional excess(진짜·repro 확인, but 현물 long-only 환금불가 — 숏/futures or 불장 필요, prelude 범위 밖)**. 설계 `_workspace/next_research_PRPC_design_v1.md`.
- [x] [Research] Downside guard / 4h confirmation tier — 2026-07-25 historical challenger 5축 전 후보 REJECT, 채택·SHADOW·운영 연결 0
- [ ] [Later] MTF features / regime split / Optuna

### Phase X+7 — PUMP hunter v2 (Binance volsurge) 🎯 radar 텔레그램 (2026-06-11 사용자 컨펌)

**의도**: "텔레그램까지 제대로 완성" (사용자 명시 요청). 검증 체인 (researcher → evaluator
적대감사 → 15m 정직 재계산 → 최근 7개월 순수 OOS) 에서 살아남은 가장 강한 hit 엣지를
radar 로 배선 — 정직 고지 (자동 net 음수) 포함.

- **룰**: `roc_7d_rank > 0.85 (Upbit D-1) AND b_vol_surge > 1.5 (Binance D-1)`.
  최근 7개월 OOS hit 8.1% (113/1390) vs baseline 5.6% vs base 1.4% — ~6x. 5/5 fold.
  자동 룰 (TP5/SL3) net -0.36% (음수) — 메시지에 정직 고지, exit 은 사용자 판단.
- `signals/pump_detector_v2.py` — v1 frame 재사용 + Binance volsurge join (D-1 경계 검증:
  BTC ret corr lag0 0.962). binance stale 시 후보 0 + 사유 (조용한 오신호 방지).
- `scripts/pump_detector_v2_today.py` — shadow ledger (`output/shadow_ledger_pump_hunter_v2.csv`)
  + 🎯 텔레그램. 발사 정책: 후보 ≥1 만 발사 / stale 시 경고 1줄 / 후보 0 정상 = 무소음.
- systemd가 호출하는 `daily_run_distribution` [10/11] binance --days 3 refresh
  (메인 알림 뒤라 무영향) + [11/11] v2 발사. close: v2 ledger 청산 (exit lab 7 잣대 자동).
- registry: `pump_hunter_v2` challenger_only=True — champion 승격 차단 유지 (radar 와 별개).
  policy_competition 자동 편입 (7 모델). heartbeat schema 검증에 v2 ledger 추가.
- 첫 발사: 2026-06-11 23:14 테스트 1통 (KAT surge 10.6× 외 4건) + ledger 5 rows.

### Phase X+6 — exit lab + 운영 강화 (2026-06-11)

**의도**: "lever 는 exit/하방 규율" 진단을 가정이 아니라 forward 데이터로 결판 + P0 운영 구멍 fix.

**exit lab (핵심)**:
- `ledger/exit_lab.py` — 같은 15m 경로에 청산 변형 3개 (TP10/noSL, TP5/noSL, EOD) + path 극값을
  병렬 가상 평가. close 시 자동 기록 (`scripts/close_recommend_ledger.py` 배선). record-only.
- `scripts/backfill_exit_lab.py` — 기존 closed 192 rows 소급. **첫 결과 (forward 6/4~6/10)**:
  bear_quiet 연구 가설 (TP10/noSL 최적) 과 반대 — pump_hunter 에서 TP5/SL3 -1.26% vs
  TP10/noSL -3.72% (deep loss 0→37). 라이브 진입 집합에선 SL-3% 가 하방을 지키는 중.
  표본 1주 (시간 집중) 라 판정은 계속 누적 후 — 매일 자동 기록이 결판.
- `ops/policy_competition.py` 에 `exit_lab` 섹션 — 모델 × 변형 비교 + 보수 비용 (편도 0.2%) net 병기.
- dashboard ⑨ 에 exit lab 표 + ⑦ 에 champion gate 진행률 bar (n_days/30, D-카운트다운).

**운영 강화 (P0/P1)**:
- `backup_db.sh` — ledger CSV·champion_state tar 백업 추가 (gitignore 라 git 에도 없던 유실 위험 fix)
  + 7일+ unchanged DB 자동 skip (stale binance/1h 매일 중복 백업 제거).
- 비용 상수 단일화 — `ledger/config.py` 가 유일 출처 (decimal `*_PCT` / %p `*_PP` 구분).
  ops 2곳의 동명·다른단위 (0.15 %p) 재정의 제거. 보수 tier (왕복 0.5%) 신설.
- `heartbeat.sh` — ledger CSV pandas parse + 필수 컬럼 스키마 검증 (행수만 보던 구멍 fix).
- 아카이브 명시 — collector_1h (37일 stale·미사용), collector_binance_d1
  (pump v2 일일 증분 + legacy 수동 retrain 경로),
  ledger/sizing.py·risk.py (radar 철학상 의도된 미배선) 헤더 주석.

### Phase X+5 — PUMP hunter rule detector SHADOW 배선 (2026-06-04)

**의도**: pump_rule_discovery_v1 에서 찾은 급등 선행 rule 을 실제 daily record-only detector 로 배선.

- `signals/pump_detector_v1.py` — D-1 `roc_7d_rank`, `atr_pct_14`, `log_return_1d` rule 적용.
- `scripts/pump_detector_today.py` — `output/shadow_ledger_pump_hunter.csv` 에 max 20 watchlist 기록.
- `signals/model_registry.py` — `pump_hunter` 추가. `challenger_only=True`, Telegram/ACTIVE 승격 금지.
- `scripts/daily_run_distribution.sh` / `scripts/daily_close_distribution.sh` — 매일 기록 + 기존 15m close path 로 CLOSED 전환.
- `ops.policy_competition` 이 CLOSED forward rows 누적 후 기존 모델들과 pump20 recall / net / downside 를 자동 비교.

**게이트**: 새 detector 는 SHADOW only. 별도 Telegram 채널/ACTIVE 통합은 forward 표본 + evaluator 판정 + 사용자 컨펌 전까지 금지.

### Phase X+4 — 정책 채택 (2026-05-26 사용자 컨펌)

**의도**: policy_gate replay 결과를 사용자가 검토 후 실제 운영 정책으로 채택.

- **distribution PROMOTE_PAPER**: 새 decision policy (`setup_quality_policy_v1`) 가 replay 에서
  observed -45.0% vs replay +71.7% (Δ +116.8%, 32 closed) 였고 late split / bootstrap CI95 low 모두 양수.
  이미 5/25 부터 코드 상 적용 중이라 변경 없음. PHASES 에 사용자 채택 사실 기록.
- **preopen DEMOTE → WATCH_ONLY (전 채널)**: observed -40.8% over 88 alerts, replay active 0건.
  `ops/decision_policy.py` 에 `PREOPEN_DEMOTED=True` flag 추가, `apply_preopen_policy` 의 ACTIVE
  분기를 WATCH_ONLY 로 강등 (bear_volatile 만 SILENCE 유지). POLICY_VERSION → `2026-05-26.1`.
  이 historical 결정은 2026-07-18 사용자 지시의 R1 preopen/open 재발송으로 superseded.
- **텔레그램**: 사용자 의도로 매일 2 통 유지. preopen 은 매일 "DEMOTED (shadow only)" 한 줄 알림
  (정책 사실 가시성). distribution 은 ACTIVE/침묵 메시지 변동.
- **heartbeat**: preopen paper_ledger 빈 거 정상 처리. shadow_ledger_preopen 검사로 대체.
- **결과 검증**: shadow_ledger_preopen 누적 후 추후 재평가. 재활성 시 `PREOPEN_DEMOTED=False`
  한 줄 변경 + version bump.

### Phase X+3 — 운영 안전 (2026-05-26)

**의도**: 20 루프 진단으로 발견된 운영 약점 P0 두 개 fix.
- DB 백업 0 → sqlite 깨지면 1년치 데이터 손실 (re-collect 며칠 + 상폐 코인 영구 손실)
- 모니터링 silent fail → collector / predict / disk full / lock 다 silent

**구조**:
- `scripts/backup_db.sh` — sqlite `.backup` (atomic, lock 없이 안전) + `PRAGMA integrity_check` + 14일 보관. `/home/soccz/22tb/backup/prelude_db/` 위치.
- `scripts/heartbeat.sh` — paper_ledger 어제 row 0 (7일 연속 0 시 alert) + DB integrity + disk 90% + publish.log 최근 fail 검사. 이상 시 텔레그램 alert, 정상 시 silent.
- systemd timer 2 추가:
  - `prelude-backup.timer` — 매일 04:00 KST (cron 안 도는 새벽)
  - `prelude-heartbeat.timer` — 매일 10:30 KST (publish 후 20분, 모든 cron 확인)
- `deploy/install_systemd.sh` 에 등록 (5 → 7 timer)

**검증**: 6/6 DB backup (총 2.9GB) + integrity 다 통과. heartbeat smoke OK.

**사용자 sudo 1번**: `sudo bash deploy/install_systemd.sh` — 2 신규 timer 등록.

---

### Phase X+2 — Dashboard publish 파이프 (2026-05-07)

**의도**: 텔레그램은 오늘 판단용. github.io 의 dashboard 는 회고용 — "어제 샀으면 어땠나, 시스템이 잘 맞추고 있나, 누적이 어떤 흐름인가" 본인 모니터링.

**구조**:
- 데이터 보강: `paper_ledger.csv` 두 개에 OHLC + min_return_pct 컬럼 추가. close 스크립트가 누락 컬럼 자동 보강 + canonical 순서로 reorder. historical 28+24 row backfill 완료.
- 빌더: `scripts/build_dashboard.py` → JSON 3종 (summary/history/accuracy) 산출. 가상 PnL 룰은 텔레그램 가이드와 동일 (5% TP / EOD close, 비용 0.15% 차감, equal weight).
- 정적 페이지: `soccz.github.io/projects/prelude/dashboard/index.html` (chart.js CDN, vanilla JS) — KPI 카드 + 누적 PnL 곡선 + rolling hit rate + 정렬/필터 가능 알림 표.
- 자동 publish: `scripts/publish_dashboard.sh` 가 build → site repo add/commit/push. 실패 시 텔레그램 alert.
- systemd: `prelude-publish-dashboard.{service,timer}` (KST 10:10, close cron 두 개 끝나고 5분 여유).

**라이브 첫 결과 (28 closed dist + 24 closed preopen)**: 누적 가상 PnL 둘 다 음수 (dist -12.97%, preopen -13.47%). avg_max +6.82% / avg_min -5.94% (dist) — 변동성은 크지만 5% TP 룰 + 비용으로 누적은 깎임. 라이브 paper 데이터 더 쌓이면서 calibration 트랙 (사용자 NOTES + dashboard) 으로 룰 조정.

**Tear sheet 강화 (2026-05-07 추가)**: pyfolio / quantstats / Bailey & Lopez de Prado (2014) 표준까지 cover. 추가된 metric (총 22+) — Volatility / Skew / Kurt / VaR / CVaR / Tail Ratio / Recovery Factor / Ulcer Index / Common Sense Ratio / W-L streak / **PSR / DSR / MinTRL** / Information Ratio / Beta / Tracking Error vs BTC HODL / Top 5 Drawdowns / Underwater plot / Rolling Sharpe (30d ann) / Monthly returns heatmap / Best & Worst trades / Stratification (regime/setup/score) / Score×PnL scatter / CSV download. PIN 기반 PBKDF2+AES 암호화 + papers viewer 와 동일 패턴.

**Methodology 출처 1:1**: Sharpe (1966) / Sortino & Price (1994) / Young 1991 / Martin 1987 / Rockafellar & Uryasev 2000 / Treynor & Black 1973 / Bailey & Lopez de Prado 2014 / Efron 1979 / pyfolio / quantstats — chip sub + about-card + References 3중 표기.

**메인 보고서와 중복 제거**: dashboard About 섹션이 이전에 메인 페이지의 #architecture / #distribution / #preopen 과 중복. 이 부분은 anchor link 5개로 reframe (Architecture / Distribution Engine v1 / Pre-open Trigger v1 / 6 Experiments / Failures Wall). dashboard 만의 가치 = EXECUTION RULE + METRIC SOURCES + 라이브 통계 한계 5 + References.

**주의**: 첫 site repo commit + push 는 사용자 수동 (라이브 반영 confirmation). 그 후부터 자동.

---

### Algorithm audit update (2026-05-05)

- v1 dry-run 유지. **v2 multi-scale swap 보류** — 1주 live paper 결과 확인 후 결정.
- Common-period ablation: head 별 best scale 이 다름.
  - h2 즉발 +3%: daily+1h / daily+15m 둘 다 강함
  - h5 +20% tail: daily+1h+15m 가 우위
  - h6 +5% 24h: daily+1h 정도면 충분
- Baseline showdown: distribution_beta 가 TP3/TP5 에서 setup/momentum baseline 을 이김.
- 다만 edge modest: TP5 path-aware Sharpe diff vs setup_momentum +0.18, bootstrap CI 가 0 근처를 걸침.
- MDD 큼: full-size TP5 MDD 약 -55%; 운영 해석은 1/4~1/8 fractional sizing 기준.
- SL 룰: 4h SL-first 와 15m path 양쪽 모두 음수. 자동 SL 룰은 운영 채택 X, 사용자 수동 판단.
- 09:05 timer audit: 전체 시장 첫 15m hit 비중은 9~12% 수준이지만, distribution alerts 는 +3% hit 의 33%, +5% hit 의 25% 가 첫 15m candle 에 발생. 09:05 는 데이터 위생상 유지하되, 실제 즉발 진입용 08:55 pre-open trigger 는 별도 모델/검증 트랙으로 분리.
- Pre-open first15 model audit: 08:55 as-of 를 엄격히 맞춰 `D-2 closed daily + D-1 08:30 precursor` 로 검증. `preopen_15m` 단독이 first15_t3 top1% precision 38.2% (base 5.1%, lift 7.6), first15_t5 top1% precision 22.4% (base 2.8%, lift 8.1). 08:55 전용 모델은 연구/운영 후보로 충분히 정당화됨. 단 v1 09:05 distribution timer 는 유지.
- Pre-open code audit: live 모델을 15m precursor-only 19 features 로 재빌드해 daily partial mismatch 제거. late manual run guard 추가(08:45~08:59 밖에서는 telegram/ledger skip), 이후 ACTIVE-only policy/edge 포맷으로 raw score 노출 제거. 15m recent-window 로 predict runtime 4m48s → 21s. close-out 은 15m DB update wrapper 사용(`scripts/daily_close_preopen.sh`).
- Survivorship bias 는 여전히 미처리 caveat.

---

## Phase X — Detector 재정의 (2026-05-03 완료)

**lessons (이번 세션 핵심)**:
- 일반 6-class softprob 모델 → ledger 음수 → "task 정의 자체가 잘못" 진단
- 재정의: ≥20% tail pump detector (rare event, silence-heavy)
- BTC bull regime conditional + TP20-only execution
- Sweep 90 조합 (regime × threshold × cap) → bull_quiet × p99.95 sweet spot
- Fold stability 검증 v1 (regime-internal threshold leak) → v2 (train direct, overfit zero-trade) → v3 (train-OOF, **C3 통과**)
- v3 발견:
  - C1/C2 (bull_quiet 단독) sparse → 3/5 fold 침묵 → fail
  - **C3 (bull_all p99.95 cap2)** 3/4 active 양수, 2024 -0.89% — 채택
  - C8-C10 rank fallback EV 음수 → 폐기 (silence-heavy 가 옳다는 증거)
- artifact: threshold 0.8815 (full panel OOF p99.95, KRW 136,924 samples) 고정
- 운영 원칙: threshold 라이브 quantile 재계산 금지, bear regime silence, cap 2

---

## 향후 방향 — Distribution head + label space discovery (사용자 2026-05-03 제안)

**문제 의식**: detector_v1 은 "≥20% tail" 한 점만 본 것. 사용자가 원하는 건 매매 판단에 필요한 **조건부 확률 분포 + 다중 head**.

**제안 구조**:
```
Distribution head (multiple binary XGBoost):
- upside heads:    P(high ≥ +3/+5/+7/+10/+15/+20%)
- close heads:     P(close ≥ +0/+3/+5%)
- downside heads:  P(low ≤ -2/-3/-5%)
- path heads:      P(hit +X before drawdown -Y)
- expected:        E[max_return], E[close_return], E[max_drawdown]

알림 출력 = 분포 테이블 (코인당), 단일 score X
```

**Label space discovery (선결)**:
- profit_target × min_close_hold × max_pre_hit_dd × max_post_hit_giveback × time_to_hit × btc_regime sweep
- 평가: base_rate, lift@top, avg fail EOD, worst5%, active_days, hit time
- 4 조건 동시 만족: 너무 sparse X, model lift > random, fail 손실 작음, 사용자 대응 가능
- detector_v1 = 분포의 오른쪽 꼬리 head 1개로 자연스럽게 흡수됨

**우선순위**: Stage 1 dry-run + Downside guard / 4h confirmation 보다 **뒤** (detector_v1 안정 후 트랙 분리). 단 stable_v1 prototype 은 detector_v1 운영과 병렬 research 가능.

---

아래 Phase 0~3 체크박스는 당시 설계·진행 이력이다. 현재 작업 우선순위와 미완료 판정은
문서 상단 `현재 상태`, 09-01 동결 판정 블록, 최신 집행 노트를 기준으로 하며 legacy
미체크 항목을 자동 작업 큐로 해석하지 않는다.

## Phase 0 — 설계 문서 (legacy)

**목적**: 코드 짜기 전 8 개 MD 로 모든 결정 명문화. 합의된 설계 위에서 구현.

### 액션
- [x] 폴더 위치 확정 (`/home/soccz/22tb/prelude`)
- [x] 폴더 스켈레톤 (data/signals/ledger/ops/notifier/scripts/notebooks/output/tests/deploy)
- [x] `.gitignore` + `requirements.txt`
- [x] `.claude/settings.local.json` (권한 / 환경)
- [x] CLAUDE.md (작업 규칙)
- [x] README.md (안내판)
- [x] SIGNAL.md (시그널 생성)
- [x] LEDGER.md (가상 ledger)
- [x] OPS.md (운영 인프라)
- [x] ASSETS.md (외부 자산 매핑)
- [x] PHASES.md (이 문서)
- [ ] NOTES.md placeholder
- [ ] git init + 첫 커밋
- [ ] Phase 1 시작 컨펌

### Exit criteria → Phase 1
- 8 개 MD 사용자 검토 완료
- 사용자가 Phase 1 시작 명시 OK

---

## Phase 1 — XGBoost baseline + KST 09:05 알림 (1-2 주)

**목적**: 가장 단순한 작동 시스템 완성. 매일 KST 09:05 텔레그램 알림 받기. 가상 ledger 자동 누적. 실거래는 사용자 직접.

### 1.1 데이터 (data/)
- [ ] `data/collector_d1.py` — 업비트 KRW 일봉 수집 (pyupbit)
  - 출처: `gan_t/data/collector.py`
  - 변경: 일봉 단일, KRW only, 3 년 백필
- [ ] `data/collector_4h.py` — 4h 보조 (장중 max drawdown 용)
- [ ] `data/collector_binance.py` — 바이낸스 1h (김프 / lead-lag 용)
- [ ] `data/upbit_d1.db` 백필 (3 년치, top 200 코인)
- [ ] `data/database.py` — sqlite 헬퍼 (load / save / latest_timestamp)
- [ ] 첫 EDA 노트북: `notebooks/01_data_eda.ipynb`
  - 코인별 데이터 길이 / 결측 분포
  - BTC regime 분포 (4-state)
  - 일봉 max(high)/open 분포 (multi-class 라벨 후보 bin 검증용)

### 1.2 라벨 + EDA (signals/) — Multi-class 분포
- [ ] `signals/labels.py::today_pump_label` (multi-class, max(high)/open 기반, SIGNAL §2.1)
- [ ] `notebooks/02_label_distribution.ipynb`
  - bin 경계 후보 (0/5/10/15/20%) 별 라벨 분포
  - 각 bin 비율 5~30% 가 학습에 좋음 — sparse / dense 면 cutoff 조정
  - 4h 봉 데이터로 max(high) 정확 측정
  - **사용자 컨펌**: bin 경계 최종 결정

### 1.3 피처 (signals/features.py)
- [ ] alt multi-lookback {3, 5, 7, 14, 21} 일 (return / vol / range_contraction)
- [ ] BTC regime (return_Nd, ma_distance, intensity, 4-state)
- [ ] 기술지표 (RSI, MACD, ADX, BB, squeeze, ROC)
- [ ] 크로스섹션 (rank_return_5d, breadth_ratio, top_n_return)
- [ ] cross-market (kimchi, binance_lead) — 보조
- [ ] cross-sectional rank norm (코인별 피처) + rolling z (BTC 피처)
- [ ] **`fillna(0)` 절대 X** (gan_t known gap)

### 1.4 모델 (signals/models/) — Multi-class softprob
- [ ] `signals/models/xgb_phase1.py` — XGBoost multi-class (objective='multi:softprob', num_class=6)
  - 출처: `gan_t/training/pump_trainer.py` (4-class 패턴 차용, 6-class 로 확장)
  - Optuna 50 trial (objective: mlogloss + per-bin macro F1)
- [ ] `notebooks/03_xgb_baseline.ipynb`
  - 첫 학습 + SHAP 피처 중요도
  - lookback 별 기여 분석
  - bin 별 적중률 (per-class accuracy)

### 1.5 검증 (signals/validate.py + scripts/backtest_wf.py)
- [ ] `signals/validate.py::PurgedWalkForward` (5-fold + 10d embargo)
- [ ] `scripts/backtest_wf.py` — 전체 WF 실행
- [ ] **트레이딩** 메트릭: net Sharpe (옵션 3 익절/손절 시뮬, 왕복 0.15%), Max DD, 누적 PnL
- [ ] **정확도** 메트릭 (사용자 핵심 요구): Brier score, Reliability diagram, Quantile coverage, per-bin accuracy
- [ ] 진단 (학술 사후): IC, ICIR — 옆 표기만

### 1.6 Calibration (signals/calibration.py)
- [ ] `signals/calibration.py::ReliabilityCalibration` — multi-class 보정
- [ ] `output/reliability_curves.json` (각 cutoff: P(≥5%), P(≥10%), ...) 첫 생성
- [ ] `output/brier_history.json` (Brier score 누적)

### 1.7 가상 ledger (ledger/) — TP/SL 시뮬
- [ ] `ledger/config.py` — 가상 자본 1,000 만, K=3, max position 5%, TP=0.10, SL=0.05 (placeholder)
- [ ] `ledger/sizing.py::equal_weight` — 1/K 균등 (Phase 1 단순)
- [ ] `ledger/tracker.py` — 옵션 3: 시가 진입 → TP/SL 또는 24h 종가 청산 (4h 봉 시뮬)
- [ ] `ledger/risk.py` — 일일 -3% / MDD -15% kill switch
- [ ] `ledger/metrics.py` — Sharpe / MDD / TP-SL hit rate / 평균 hold
- [ ] `output/ledger.csv` 자동 누적
- [ ] `scripts/tp_sl_sweep.py` — TP × SL 격자 백테스트 → 최적 조합 추천

### 1.8 운영 (ops/ + notifier/ + scripts/)
- [ ] `ops/preflight.py` — freshness / NaN / churn 체크
- [ ] `ops/run_lock.py` — cron 중복 방지
- [ ] `ops/drift_detector.py` — sign flip / 50% drop 감지
- [ ] `notifier/telegram.py` — 텔레그램 봇 클래스
- [ ] `notifier/format.py` — 메시지 포맷 (OPS §3.1)
- [ ] **별도 텔레그램 봇 발급 + 채팅 ID** (gan_t 와 분리)
  - **사용자 컨펌**: 봇 토큰 / 채팅 ID 환경변수 셋업 (`.env`, .gitignore)
- [x] `scripts/predict_today_distribution.py` / `scripts/predict_preopen_trigger.py` — 수동 dry-run + 운영 후보
- [x] `scripts/daily_run_distribution.sh` / `scripts/daily_run_preopen.sh` — cron/systemd entry
- [x] `scripts/daily_close_distribution.sh` / `scripts/daily_close_preopen.sh` — paper/shadow ledger close
- [ ] `deploy/crontab.txt` — cron 등록 명령 + README
- [ ] `scripts/health_check.py` — 일일 헬스
- [ ] `scripts/verify_telegram.py` — ledger ↔ telegram 일관성

### 1.9 첫 알림 발사
- [ ] dry-run 1 일 (수동 실행, 텔레그램 발송 X)
- [ ] dry-run 결과 + 알림 포맷 사용자 검토
- [ ] **사용자 컨펌**: 라이브 cron 등록
- [ ] **D-Day**: 첫 KST 09:05 라이브 알림
- [ ] 매일 KST 09:30 ledger 자동 갱신 확인

### Exit criteria → Phase 2
- [ ] **라이브 기간 충분** (초기 2 주, 데이터 안정성 보고 조정 — CLAUDE.md §2.5)
- [ ] 가상 net Sharpe **양수 + 의미 있는 수준** (음수면 모델 재설계, 0~0.5 면 Phase 2 갈지 고민, ≥ 0.5 면 자연스럽게 Phase 2)
- [ ] **시스템 정확도 검증**: Reliability 가 대각선 ± 10pp 이내 (예: 예측 50% → 실제 40~60%)
- [ ] 일관성 검증 통과 (verify_telegram 모든 날 OK)
- [ ] 사용자가 Phase 2 명시 OK

### Phase 1 lessons (작성: Phase 1 끝나는 시점)
*(여기에 Phase 1 진행하면서 배운 점, 실패, 의외의 발견 기록)*

---

## Phase 2 — Hybrid 모델 + 사이징 강화 (2-4 주)

**목적**: Phase 1 baseline 대비 의미 있는 net 개선 (DM test). σ-tier 기반 사이징 비교.

**Phase 1 결과가 충분히 좋으면 Phase 2 유보** (단순 유지). CLAUDE.md §2.3 — 학술적 정교화로 결과 더 나빠지는 경우 많음.

### 2.1 Hybrid 모델 (Phase 1 의미 있을 때만)
- [ ] `signals/models/hybrid_phase2.py` — Transformer + TCN + Gate + FiLM + CVAE
  - 출처: `gan_t/models/hybrid_model.py` + `AETHER_IDEA.md`
  - 변경: 일봉 호라이즌, residual return 입력 옵션
- [ ] `notebooks/04_hybrid_train.ipynb`
- [ ] DM test: Hybrid vs XGBoost baseline (net Sharpe 차이 유의?)
- [ ] **사용자 컨펌**: Hybrid 채택 vs Phase 1 유지

### 2.2 σ-tier 사이징 비교
- [ ] `ledger/sizing.py::sigma_tier_weight` — 🔥 3% / ✅ 2% / ▫ 1%
- [ ] backtest 비교: equal vs sigma_tier (net Sharpe / MDD)
- [ ] **사용자 컨펌**: 사이징 룰 변경 또는 유지

### 2.3 Window signature (선택)
- [ ] `signals/models/window_signature.py` — 윈도우 → GRU → 16d signature
  - 출처: `fin/paper/economic_time/window_signature_model.py`
- [ ] hybrid 모델에 conditioning token 추가
- [ ] DM test: with vs without signature

### 2.4 Multi-day continuation 점수 (사용자 1 차 메시지의 보조 아이디어)
- [ ] 메인 라벨은 그대로 (오늘 1 일), but 별도 회귀로 N=3 일 지속 점수 출력
- [ ] 알림에 보조 점수로 표시: "오늘오를 78% / 3 일지속 65%"
- [ ] **사용자 컨펌**: 알림 포맷 변경

### 2.5 손절 / 익절 룰 비교 (가상 ledger)
- [ ] `ledger/tracker.py` 에 옵션 추가
- [ ] 비교: hold 1d (현재) vs +15% 익절 / -5% 손절
- [ ] backtest net Sharpe / MDD / 평균 hold 시간

### Exit criteria → Phase 3
- [ ] Phase 2 라이브 4 주 완료
- [ ] Hybrid 채택 시: Phase 1 대비 DM test 유의 (p < 0.05)
- [ ] σ-tier 채택 시: net Sharpe 개선
- [ ] 사용자 명시적 Phase 3 OK

### Phase 2 lessons
*(작성: Phase 2 끝나는 시점)*

---

## Phase 3 (옵션) — APF motif + prototype bank + 학술 (선택)

**옵션**. 트레이딩 결과 Phase 1/2 만으로 충분하면 Phase 3 안 해도 됨. 학술 트랙 관심 있으면 진행.

### 3.1 APF motif 진단 (Phase 2 hybrid 위)
- [ ] `signals/models/apf_diagnostic.py` — attention map → motif 분류 (stripe/block/spike/diagonal)
  - 출처: `fin/Attention Pattern Fields/src/apf/`
- [ ] 알림에 motif 표시: "추천 근거: spike motif 89% (이벤트 탐지)"

### 3.2 Prototype bank + DTW
- [ ] `signals/prototype_bank.py` — 과거 stable_pump 성공 윈도우 DB
  - 출처: `gan_t/data/success_patterns.npy` + AETHER prototype bank 컨셉
- [ ] 일일 추론 시 DTW 매칭: "과거 패턴 #7 유사도 0.81 (당시 +12.3%)"
- [ ] 알림에 추가

### 3.3 학술 논문화 (선택)
- [ ] APF 페이퍼 의 금융 도메인 첫 적용 — TMLR 보강 또는 별도 짧은 논문
- [ ] BTC × Upbit version of F=14.335 finding
- [ ] **사용자 명시 시에만 진행** — 트레이딩 결과 우선 (CLAUDE.md §2.3)

### 3.4 Cycle-PE / linear attention 실험 (장기)
- [ ] hybrid 모델 인코더에 Cycle-aware PE 추가 (AETHER §5)
- [ ] linear attention 변형 (fin Ch.08 +49% finding)
- [ ] DM test 비교

### Exit criteria → 안정 운영
- [ ] Phase 3 추가 모듈 중 net 결과 개선되는 것만 채택
- [ ] 결과 안 좋으면 Phase 2 로 롤백
- [ ] 안정 운영 모드 진입

### Phase 3 lessons
*(작성)*

---

## 비상 / 롤백 절차 (legacy 설계)

아래 표의 drift/retrain/verify 경로는 현재 scheduler에 배선되지 않은 과거 설계다.
현재 자동 중단 기준과 fail-closed 동작은 OPS.md의 실제 systemd/health 계약을 따른다.

각 Phase 진행 중 다음 발생 시:

| 상황 | 액션 |
|---|---|
| 가상 net PnL 4 주 누적 음수 | 텔레그램 watch-only 전환 + 사용자 컨펌 후 모델 재설계 |
| drift detector FREEZE 발동 | 즉시 가상 진입 X + retrain 강제 트리거 |
| verify_telegram 부호 / 크기 불일치 | 즉시 알림 발사 정지 + 디버깅 + 재개 컨펌 |
| 데이터 24h 이상 stale | preflight 자동 watch-only |
| 사용자 NOTES 에 "시스템 신뢰 X" 기록 | 다음 세션 시작 시 Claude 가 발견 → 사용자 컨펌 |

---

## 작업 흐름 (매 Phase 공통)

1. 이 문서 head 60 줄로 어디까지 왔는지 확인
2. 상단 현재 상태·동결 판정·최신 집행 노트에서 승인된 다음 액션 1개를 in_progress
3. TodoWrite 로 세션 내 작업 추적
4. 액션 완료 시 즉시 [x] 체크 + 한 줄 lessons 기록 (선택)
5. 큰 결정 (모델 변경, 라벨 X/Y, 알림 포맷) 은 사용자 컨펌
6. Phase exit criteria 충족 시 사용자 컨펌 후 다음 Phase 진입

---

## 변경 이력 (Phase 단위)

| 날짜 | Phase | 변경 / 결정 |
|---|---|---|
| 2026-05-03 | Phase 0 | 폴더 + 8 개 MD 설계 시작 |
| 2026-05-03 | Phase 0 | 모든 숫자 placeholder 명시 + CLAUDE.md §2.5 신설 (데이터가 결정) |
| 2026-05-03 | Phase 0 | 라벨 binary → multi-class (max(high)/open 분포), ledger 단순 hold → TP/SL 옵션 3 |
| 2026-05-03 | Phase 0 | `today_pump` 폴더명 → `prelude` 통일 + GitHub `soccz/prelude` push |
| 2026-05-03 | Phase 0 | `data/database.py` + `data/collector_d1.py` 작성, KRW-BTC smoke test PASS |
| 2026-05-03 | Phase 1.0 | 4 collectors 백필 완료 (KRW d1 252, KRW 4h 252, BINANCE 1h 185, BINANCE d1 427) |
| 2026-05-03 | Phase 1.1 | **leak 발견** — features[t] (close[t]) → label[t] (high[t]/open[t]) 동시점 사용 |
| 2026-05-03 | Phase 1.1 | label_panel **market별 shift(-1)** 수정 — leak 제거 |
| 2026-05-03 | Phase 1.1 | leak 후 accuracy 67%→30% (random 17% 대비 1.83x — 약한 신호만) |
| 2026-05-03 | Phase 1.2 | 알림 시간 **08:30 → 09:05** (어제 일봉 100% 마감 후 leak-free) |
| 2026-05-03 | Phase 1.2 | Pattern sweep — 7 family WF ledger backtest 모두 음수 Sharpe |
| 2026-05-03 | Phase 1.2 | EDA hit rate ≠ Sharpe — momentum hit 21% but Sharpe -5.2 (SL 46% 함정) |
| 2026-05-03 | Phase 1.3 | Execution sweep — 15 룰 × 3 family → **TP15_only 만 Sharpe +0.13** |
| 2026-05-03 | Phase 1.3 | Filter alpha X — baseline_full random 이 모든 family 이김 |
| 2026-05-03 | Phase 1.3 | 진단: 일봉 long 대부분 손해, 드문 +15% 꼬리 펌프만 알파 가능성 |
| 2026-05-03 | Phase 1.4 | 핵심 검증: TP15_only execution + binary 모델 vs random (진행 중) |
| 2026-07-08 | 사전등록 | GO/KILL 패널 비준 → 아래 시한부 판정 블록 박제 (DECISIONS.md #2) |
| | | |

---

## 🔒 사전등록 판정 블록 (2026-07-08 비준, 수정 금지 — 결과 보기 전 동결)

**프로젝트 상태: radar-not-strategy (전 정책 net 음수). "유지"가 아니라 시한부 실험으로 전환.**

- **판정일: 2026-09-01.** UNDECIDED 불허 — 그날 무조건 GO 또는 KILL.
- **v2 승격 기준 (전부 충족해야 champion 승격 + radar GO):**
  - closed 누적 n ≥ 200
  - per-trade mean net > 0 AND 95% CI가 0 제외
  - 2개 이상 레짐에서 관측 OR per-day t ≥ 2
  - n < 200이면 신호 생산률 자체가 판정 근거 → 그래도 KILL (UNDECIDED 불허)
- **미달 시:** v2 KILL + self_impact 재추정(ACTIVE n≥50 & WATCH n≥30)도 ADOPT 불가면 **radar 전체 KILL**(타이머 정지·아카이브).
- **조기 KILL:** 09-01 이전이라도 v2 closed 누적 net 평균이 0 미만 전환 시 즉시 radar 전체 KILL.
- **최소관심 모드(즉시 적용):** ACTIVE 외 텔레그램 음소거 · recommend·distribution record-only 강등 · **09-01까지 신규 연구 모라토리엄**(착수 시 관심누수 증거 = KILL).
- **LVG-XS futures:** 기각(4후보 감사서 WEAK·repro 아티팩트 0). 09-01 안건에서 repro 스크립트 재실행 아티팩트 제시 못 하면 영구 기각.
- **데드맨스위치:** 이 블록이 커밋된 시점부터 유효. 09-01에 사람이 판정 안 해도 위 기준으로 자동 KILL.

### 집행 노트 (블록 밖 — 동결 블록 무수정)

| 날짜 | 집행 | 근거 |
|------|------|------|
| 2026-07-11 | 블록 git 커밋 1d46a9c (07-08 작성분, 커밋 지각 = 형식요건 수리, 자동 킬 미발동 판정) | 커밋 메시지에 판정문·v2 실측(n=133, net +0.353%, 조기KILL 미충족) 기록 |
| 2026-07-11 | 최소관심 모드 실집행: R1 recommend_send 2건(preopen·open) `--dry-run` 강등 — 텔레그램 발송 X, R1 ledger 기록은 유지 | 블록 "recommend record-only 강등" 조항 |
| 2026-07-11 | **해석 명문화**: pump v2 radar 텔레그램(distribution [9/9])은 음소거 대상 아님 — v2는 09-01 판정 대상 실험 그 자체이며, 블록의 "ACTIVE 외 음소거"에서 v2가 유일한 판정 대상 채널. R2·A1·PUMP-rule 등 challenger는 종전대로 record-only(champion_selector 영구 차단) | 블록 취지 = 죽은 정책 관심누수 차단, 실험 관찰 채널 유지 |
| 2026-07-11 | heartbeat 이상시 알림·backup·dashboard publish는 운영 채널로 유지 | 판정과 무관한 위생 |
| 2026-07-18 | 사용자 지시로 R1 preopen/open 텔레그램 발송 재개. 원장·판정 기준은 그대로 유지 | 추천은 사용자가 직접 판단·매매하며 자동 주문 없음 |
| 2026-07-25 | Track 1 측정 무결성 수리 완료: 단일 inference snapshot·delivery receipt·전 유니버스 score/label/evaluator·실행 가능 시각부터 새 96봉·신규상장/수집 실패 차단·공통 day-equal 지표. 실데이터 D1/4h/15m `269/269`, 전체 테스트 `177 passed` | 판정 결함 수리이며 동결 블록·활성 R1 정렬/라벨/모델/알림 문구는 무수정. systemd unit 소스는 완료, `/etc` 재설치는 sudo 대기 |
| 2026-07-25 | 사용자 명시 요청으로 별도 historical challenger 5축(downside veto·upside·safe-up·first-passage·downside semivol)을 끝까지 실행하고 독립 재검산. 전 후보 REJECT, 채택 0·SHADOW 0·운영 연결 0 | 동결 블록 본문은 무수정. 다만 모라토리엄 기간 연구이자 이미 본 마지막 180일을 사용했으므로 깨끗한 사전등록/virgin holdout 증거로 사용할 수 없음. 추가 사후 식 탐색은 중단하고 새 forward 전 유니버스 표본을 우선 축적 |
| 2026-07-26 | 전 변경 코드 적대 감사·보강: stablecoin 5종 provenance, D1/4h/15m exact PIT gate, preopen R1과 legacy 15m 장애 격리, immutable snapshot/receipt/ledger, v2 205행 legacy digest + 07-27 strict 계약, GO/KILL state+anchor, close/publish/heartbeat/backup 실패 전파. 최종 전수검증 진행 중, SQLite 7개 `quick_check=ok` | 동결 판정·활성 R1 정렬/라벨/모델/알림 문구는 무수정. 실제 `/etc` systemd 설치는 sudo password와 `.env`의 `PRELUDE_DASHBOARD_PIN` 준비 대기 |
| 2026-07-28 | 07-27 발송·close 3중 장애 수리: ① snapshot 확률 단조성 하드체크가 모델이 만족한 적 없는 불변식(36/100 위반)으로 R1 발송 전면 사망 → top-k(발송분)만 하드 유지, 유니버스 꼬리는 진단 경고 강등(라벨 경로도 계약 통일) ② 07-26 이음매(구스키마 원장+신형 증거)를 close_input_gate가 거부 → `decision_completed_at` 1컬럼 allowlist 관용(부재·공란만, 값 불일치는 여전히 하드) 후 07-26 R1 3건 청산(MEW TP+4.85/KERNEL SL-3.15/ZBT SL-3.15) ③ 증거·행 전무한 no-decision 일자가 close 백로그 전체를 오염 → `skip-no-decision` 모드+감사 마커(`output/close_no_decision/`) 신설. quant-reviewer 적대 리뷰 2회전(CRITICAL 3+N1 해소) 후 "신뢰 가능" 판정, 전체 테스트 1262 passed | 07-12 이후 무커밋 작업 408건을 8개 논리 커밋으로 선행 박제. 후속 과제: top-k 단조성 마진 1랭크(open_r2 min 위반 rank 4), 라벨 flat_filled 67/100의 평가 필터 부재, no-decision 일수 커버리지 분모 표기 |
| 2026-07-28 | 완성 라운드(4축 병렬 진단 → 수리): ① **v2 후보 사멸 회귀 수리** — 07-26 강화의 binance 원자 fail-closed가 업비트 단독상장/신규상장/상폐 심볼과 수집장애를 구분하지 않아 후보 생산이 영구 0(소급: 신계약 기준 사실상 전일 사멸, 07-27까지 verified 증거 0). 구조적 제외(not_listed·newly_listed·not_listed_asof·delisted_asof[grace 7d 초기값])를 분리하되, **BTCUSDT 앵커 프로브**(f_day 봉+20d 연속 이력)를 "ok"의 전제로 요구해 DB 유실/재구축이 "정상 무추천일"로 위장되는 fail-open 차단 + 전량 제외는 `binance_no_ready`로 시끄럽게. 제외 내역은 decision manifest(`binance_ready_n/needed_n/excluded`)에 박제. 실검증: 07-26 5건·07-27 5건 후보 재생산 ② **heartbeat 거짓 FAIL 수리** — Telegram이 앞뒤 개행 strip해 에코하는데 신설 완전일치 검증이 ambiguous 오분류(07-27 알림은 실제 전달됐음). 서버 trim 허용+MSG trailing 개행 제거 ③ **flat_filled 평가 코호트** — path_quality 요약(양방향 편향 주석: flat-fill=0축소 / complete-only=유동성 낙관, 병기 강제)+`complete_path_only` 코호트 추가(기존 수치 무변경) ④ heartbeat에 no-decision 마커 관측(일수 기준)+`binance` 관련 문서 현행화 ⑤ **재생성 큐 소멸 판정** — "벌크로더 이전 산출=오염" 전제가 SHA-256 동일성 증명(682bc20)으로 반증, A그룹 14종은 위생수정 이후 코드 산출+manifest 봉인(21/21 SHA 일치), B그룹 7종은 재생성=6월 판정 증거 덮어쓰기라 금지 ⑥ meta 트레이너 영구 0샘플(ordered 증거 컬럼 미기록)은 의도된 fail-closed 확인 — 존치/은퇴는 DECISIONS #13 등재(데드라인 08-11, 기본값 현상유지) ⑦ signals/ ruff 클린(F401 13+E741 2) | 적대 리뷰 2회전(C1 feature_date_missing 잔존 회귀 소급 10/53일·C2 DB장애 fail-open 신설 → 전부 수리 후 재검증). 단조성 top-k 완화는 종전 문서의 "사용자 결정 대기안"을 가동 불능 해소를 위해 채택한 것 — 원복 가능(SIGNAL §7.4). v2 산술 주의: 수리해도 GO n≥200은 잔여일×캡5로 도달 불가(상한<200), 동결 조항상 09-01 KILL 경로는 불변 — 단 이제 증거는 실제로 쌓인다 |
| 2026-07-28 | **첫 실전 아침 라이브 검증**: R1 예고 08:52·확정 09:08 발송 성공(delivery_ok·원장 3행), failure-alert 3발 정상 발사(09:14/09:36/10:10 — 침묵 제거 실전 증명). 라이브 결함 3종 발화→당일 수리: ① seam 경고가 NUL 레코드 stdout 오염 → close 3채널 전멸(게이트 로거 stderr 고정) ② v2 healthy 경로 첫 가동에서 scorer date-only feature_date 잠복 폭발 → 오늘 v2 발송 유실(09:00 타임스탬프 수정, 내일부터 정상) ③ pump-v1 07-26 계약이전 잔존 행이 매일 close 독 → legacy-unverifiable 봉인. 수리 후 close 게이트 R1/R2/A1/v1 전부 정상(v1 07-27 15행 청산), preopen-close green, 대시보드 수동 발행(PIN, 내일부터 자동) | v2 07-27 unhealthy decision 은 1회성(내일 스코프 밖 자동 소멸). 후속: flaky 테스트 2건(microstructure 타이밍·scoreboard 순서, 각각 단독 통과) |
| 2026-07-28 | 테스트발 실발송 사고 봉쇄: pytest 픽스처의 유도 백업 실패가 venv 전역 import + 실 .env 토큰 경유로 스위트 1회당 실경보 4통을 발사(총 수십 통, 사용자 스크린샷 확정) → `PRELUDE_FORBID_TELEGRAM` kill-switch(notifier 최상단 + tests/conftest 전역, subprocess 상속)로 구조 봉쇄. 실물 증명: 동일 픽스처가 '❌ delivery failed'로 종결. 테스트 1,276 | mock 검증 모듈 4곳만 delenv 해제. 공개 산출물 3축 엄격 감사 후속: 대시보드 뷰어 R1 채널 배선·판정 배너, 페이지 구 디바이더/CSS 충돌 제거, README 뱃지 인코딩·심링크 실파일화 |
| 2026-07-29 | 둘째 실전 아침 — 잔여 결함 2계통 수리: ① **무거래 top100 코인 1개(AKT)가 발송 전체 차단** — coverage 게이트가 top100 내 결측을 무조건 fail(관용장치는 top100 밖 전용) → 결측 ≤5개는 업스트림 REST 확인으로 이분: 업스트림에도 없음=무거래 구조적 관용(`no_trade_confirmed=`), 있음=진짜 갭 fail 유지, 확인불가/대량은 전량 fail-closed. 오늘 확정·v2 발송은 유실(예고는 발송됨, 내일 no-decision 자동 처리) ② **snapshot/label 진단 경고가 root logger→stdout 으로 새어** close NUL 재오염 + heartbeat 도배 → 모듈 2종 stderr 고정+전파차단, gate main() root-stdout 방어, v2 침묵 스킵도 로그화. 복구 재실행: close·preopen-close·publish 전부 ExecMainStatus=0, AKT 100/100 확인. 테스트 1,279 | R2 07-28 snapshot 미생성은 후속 확인 항목. 교훈: 진단 로그의 목적지는 계약이다 — stdout 은 기계의 것 |
| 2026-07-29 | **재발방지 라운드(사용자: "다시 반복 안되게 확실하게")** — 사후수리에서 클래스 봉쇄로 전환. **3중 가드 신설**: ① NUL 프로토콜 fd 봉인(`ops/_stdout_seal.py`) — nul 모드 기동 즉시 fd1을 프로토콜 전용으로 복제하고 fd1 자체를 stderr로 교체, 이후 어떤 모듈의 print/logging도 셸 파서에 도달 불가(적대 e2e: root logging을 stdout에 물리고 오염 주입해도 stdout 순수 증명) ② `prelude-selftest.timer` 07:30 KST — 아침 사이클 전 전수 pytest, 실패는 OnFailure 경보로 승격(회귀를 라이브 이전에 검출, 설치기 신규유닛 not-found 문구 관용 수리 포함, 8타이머 체제) ③ pre-push 훅(`deploy/git-hooks/`) — 전수 통과 전 push 차단. **3축 전수감사(16 findings) 반영**: 근본원인 확정 — CLI 8파일이 import 시점에 root logger를 stdout으로 구성(`basicConfig` 모듈레벨)한 것이 07-28/29 오염 클래스의 뿌리 → 전부 main() 안으로 이동+import 불변식 테스트(`test_import_logging_purity`)로 잠금. heartbeat 'ok' 프로브 2곳 stderr 재지향, backup lock busy를 0→75 fail-loud+최신성 검사 신설, ledger row 프로브 fail-open 수리, 분배 window-밖 skip을 rc 3 fail-loud로, setup 룰 예외 은폐를 집계 경고로 가시화. **테스트 실발송 탈출구 4건 봉쇄**(P1: v2 무소음 정책 회귀 시 실발송 가능 — recorder 봉쇄+0건 단언, spawn 자식 2곳 kill-switch 재주입, limit-markets 가드 테스트 transport 봉쇄), 콜렉터 타이밍 마진 2건 확장 | drift/risk 평가기 미배선 잠복은 DECISIONS #14 등재(기한 08-11, 기본값 제거). v1 btc_regime 'unknown' 관용은 후속 검토 항목. 알림·라벨·판정 문구 무수정 |
| 2026-07-29 | **재발방지 최종 closure(독립 적대 리뷰 3축 2회전 PASS)** — 봉인은 gate `__main__`에만 scope하고 argparse 축약 금지·반복값/help 일치·stdout/stderr 병합 거부·실 Bash `coproc/mapfile`까지 고정. heartbeat의 policy/snapshot은 stdout 문자열 비교를 폐기하고 rc 계약으로 전환. backup은 lock 3000초 bounded wait, 당일 terminal manifest→checksum→archive SHA 결합, missing-only 3300초 catch-up polling(기존 손상/symlink 즉시 실패), heartbeat unit 3900초로 deadline 분리. setup 룰은 부분 오류 집계·전량 오류 RuntimeError, pump-v1 DB 예외는 전파(정상 empty만 `unknown`), distribution late-run은 산출물 쓰기 전 rc3. 미배선 drift/risk는 production health의 거짓 `OK`에서 제거. delivery 테스트 2모듈은 transport 기본 bomb+tmp output 격리, pre-push는 clean HEAD·remote base·Ruff·전수 pytest를 모두 강제 | 검증: 변경 Python 30개 Ruff clean, 셸 syntax/diff-check clean, 전수 `1329 passed`(390.99s), live systemd selftest `ExecMainStatus=0`, installer `--check-only` OK, source↔`/etc` selftest/heartbeat byte 일치, 8 timers active. 동결 파일(`signals/recommend.py`, `notifier/format.py`, `ledger/config.py`) 무변경 — R1 정렬/모델/라벨·pump 규칙/TP/SL·알림 문구 무수정 |
| 2026-07-29 | **closure 교정 재감사** — 위 PASS 뒤 원격 HEAD를 빈 kill-switch·I/O 실패·lazy import 조건으로 다시 감사해 잔여 7경계를 발견·봉쇄: ① `tests/conftest.py`가 빈/임의 `PRELUDE_FORBID_TELEGRAM`도 강제 `1`, pre-push pytest도 동일 강제 ② Telegram/radar 테스트의 kill-switch 해제 모듈 2곳에 기본 HTTP 폭탄 ③ heartbeat no-decision `find\|wc` 실패·비숫자 결과의 부분집계 거짓 정상 제거 ④ installer가 `/usr/bin/timeout`·`ops/backup_manifest.py`를 선검증 ⑤ `signals.recommend` lazy import 하위 3모듈의 module-level stdout logging을 `main()` 전용으로 이동 ⑥ 명시적 `PRELUDE_SKIP_PREPUSH` 우회 제거 ⑦ pre-push 통합 테스트의 가짜 hook `TMPDIR`를 fixture별 격리해 외부 실제 전수 로그 truncate/sparse-hole 방지. 로컬 hook은 정상 `git push`의 우발 회귀 방지이며 Git 자체 `--no-verify`를 막는 원격 정책은 아님(OPS에 한계 명시) | 검증: 빈 kill-switch 관련 회귀 `287 passed`, 전수 `1338 passed`(207.04s), 변경 Python Ruff·셸 문법·diff-check clean, live systemd preflight OK, 오늘 backup manifest/checksum/archive SHA PASS, 8 timers 장전·selftest `Result=success`. 최종 heartbeat 코드의 첫 예약주기 실증은 07-30 10:30 예정이며, 07-29 결손 경보는 당시 실제 데이터 결손 기록으로 보존. 동결 계약 파일 3종 무변경 |
| 2026-07-30 | **셋째 아침 라이브 검증에서 “완료” 주장 철회 후 D1 경계 race 수리** — 사용자 스크린샷의 09:12 `prelude-distribution.service` 실패와 10:30 snapshot-chain/no-decision 경보를 원본 로그·DB·snapshot/receipt/marker로 전수 대조. 경보는 오탐이 아니라 실제 결손: 09:05 D1 1회 수집에서 `KRW-AERO`·`KRW-LAYER` 당일 09:00 봉이 빠졌고 09:08 health가 업스트림에는 존재하는 DB gap `98/100`으로 정확히 차단해 open R1/R2/A1/pump v1/v2가 모두 미생성. 두 행은 09:30 수집에서 복구돼 영구 DB 손상은 아니었음. **수리**: 첫 exact health 실패 시 현재 경계 결손 live identity만 2초 후 targeted refresh(`days=1`)하고 동일 gate를 1회 재검증; pipeline 전체 refresh 1회, 최대 64개(07-30 관측 44+여유 초기값), 초과 광역 결손·지속 실패는 기존처럼 nonzero/OnFailure. fetch 실패와 unresolved를 분리 기록하고 최종 health만 발송 허가 권위로 유지. **실증**: 실 DB 복사본에서 AERO/LAYER 두 행 삭제(0개)→새 CLI가 두 종목만 선택→2/2 복구·`unresolved=0`; 현재 live recommend gate `100/100`; 복구/지속실패/refresh 자체 실패/양채널 동시실패·종료코드/발송차단 회귀 포함 표적 234건 및 전수 `1355 passed`(200.40s), Ruff·Bash syntax·diff-check 통과, 독립 최종 재감사 CRITICAL/MAJOR/MINOR 0. 오늘 놓친 open/pump는 소급 발송하지 않고 no-decision 증거로 보존 | 코드·격리 e2e 기준 수리 완료. **07-31 09:05 예약주기 라이브 실증은 아직 미래라 완료로 가장하지 않음.** R1 정렬/모델/라벨·pump 규칙·TP/SL·알림 문구 무수정 |
| 2026-07-31 | **D1 경계 retry 첫 예약주기 라이브 실증 PASS + heartbeat publish 대용량 오탐 수리** — 09:05 첫 health `97/100`(ATH/PENDLE/TRUMP 결측) 뒤 targeted refresh가 발동해 09:08 exact gate `100/100` 복구; open R1 delivery/ledger 3행, R2 3행, A1 3행, pump-v1 15행, pump-v2 delivery/ledger 4행까지 정상 생성. selftest·preopen·distribution·close·preopen-close·publish·heartbeat 전 유닛 `Result=success`. 10:30 `close no-decision 2026-07-30(cohorts=5)`는 전날 실제 파이프 사망일의 올바른 역사 경보였고, `publish 오늘 세션 없음`은 오탐: 실제 publish는 commit/push 완료했지만 최신 세션이 72,503바이트라 `set -o pipefail` 아래 `echo "$LAST_SESSION" | grep -q`가 grep 조기종료→echo SIGPIPE(rc1)로 성공을 뒤집음. **수리**: publish 로그를 단일 POSIX awk가 EOF까지 읽고 최신 세션의 date/fail/done을 작은 상태값으로 반환하도록 변경. 2,125,912바이트 fixture에서 구 구현 `echo=1/grep=0` 재현 후 새 성공·실패 회귀 모두 PASS; 셸 전파 105건, 전수 `1357 passed`(177.62s), Ruff·Bash syntax·diff-check, 오늘 실로그 직접 판정 `ok`, 독립 적대 재검토 findings 0 | 운영 산출물·R1 정렬/모델/라벨·pump 규칙/TP/SL·사용자 알림 포맷 무수정. heartbeat 소스는 repo 직접 실행이라 systemd 재설치 불필요 |
| 2026-08-03 | **첫 실전 부팅폭풍(기계 08-01 새벽~08-03 10:26 다운) — 캐치업 결함 3계통 수리**: 부팅 즉시 Persistent 타이머 5종 동시 발화로 ① backup 이 close 수집기와 `upbit_15m.db` lock 충돌(`database is locked` → rc 1 경보) → sqlite `.backup` busy-timeout 30s+3회 재시도, backup·selftest 유닛에 close 유닛 After= 직렬화(평시 무영향) ② selftest 가 실 운영 DB를 읽던 비밀폐(non-hermetic) 테스트 2계통(라벨 manifest 의 M15 기본경로 8곳, snapshot D1 provenance)이 수집기 병행 쓰기와 충돌해 실패 → tmp 재지향 + **conftest sqlite hermeticity 가드 신설**(테스트가 `data/*.db` 를 여는 순간 즉시 실패; 부수효과로 라벨·스냅샷 테스트 130s→3.6s) ③ **15m 수집기는 내부 갭을 못 메우는 구조**(최상단 200봉+바닥 연장만) — 07-31 96봉 창이 영구 partial 고착, v2 07-31 원장 no_data 오염, policy_competition rc 1 연쇄 → `--heal-days` 갭치유 모드 신설(UPSERT 멱등 재페이징), close 스크립트 2종을 `--heal-days 3`(초기값)으로 배선, 07-31 창 실치유(96/96·272마켓) 후 close 연쇄 재실행. v2 provenance/scoreboard 검증기가 원장 정당 상태인 `no_data` 를 거부하던 것도 허용으로 정정(병행 codex 세션 작업). 08-01·08-02 는 skip-no-decision 정상 처리(무결정일 설계 검증됨) | 4h/D1 은 페이지 깊이(33일/200일)로 이 클래스 면역. deadman-watch(대시보드 push 72h)·OnFailure 경보 모두 설계대로 발화 — 침묵 없音. 3일+ 다운은 --heal-days 3 초과이므로 수동 heal 필요(데이터가 결정하면 조정) |
| 2026-08-04 | **첫 완전 무인 하루** — 8개 타이머 전 단계 성공, 수동 개입 0: 04:00 backup(manifest ok) → 07:30 selftest(전수 1,359 passed) → 08:50 예고·09:05 확정 발송(delivery_ok, R1 open 2건 기록) → 09:30/10:05 close 2종(marker 2종) → 10:10 대시보드 발행 → 10:30 heartbeat. distribution 채널도 8후보 중 ACTIVE 1건(KRW-ELSA) 기록 | 07-28~08-03 하드닝 라운드(클래스 봉쇄 3종 + 부팅폭풍 내성) 이후 첫 실증. heartbeat 경고 2건은 지난 공백의 정직한 회계(다운타임 무결정일·7일 무신호)로 익일 자동 소멸 |
| 2026-08-05 | **셋째 실전 아침 — 상폐 종목이 close·publish 를 무기한 차단하는 클래스 봉쇄**: AERGO·AQT 상폐(업비트 404 Code not found 실측)로 08-04 preopen 라벨 2행이 창 전체 무봉 → `path_incomplete` 영구 고착 → 라벨 partial(rc 2) → marker 부재 → publish fail-closed 연쇄. **`halted_no_observations` 구조적 종결 상태 신설**: 무봉 행은 업스트림 확인(404 상폐 / 200+빈배열 / 봉 전부 창밖 — 이 셋만 인정, 그 외 fail-closed)이 될 때만 종결로 재분류되고 forward 평가 표본에서 명시 제외(summary.halted). 검증기·평가기 계약 동반 갱신(legacy artifact 호환 유지), 라이브 재실행으로 08-04 preopen complete(labeled 98·halted 2) + close·publish·heartbeat 전부 rc 0 복구. **v2 조기 KILL 은 장애 아님** — 동결 룰(`누적 mean_net<0 → early kill`) 그대로의 자동 집행(n=9, mean −0.097%), 판정문 integrity hash 박제(`radar_terminal_verdict.json`). distribution 아침 실패는 FLUID·TRUMP D1 당일봉 수집 race(첫 거래가 09:06 수집 이후 발생) — 업스트림 확인 게이트가 정확히 fail-closed 로 잡은 것, 자연 치유됨 | 후속: coverage gate 의 "업스트림 존재" 판정 시 타깃 재수집 1회 후 재검(race 클래스 자동 치유), R2 record-only 실패 원인 확인. halted 는 라벨 X/Y 정의 변경이 아니라 평가 가능성 회계이나, 사후 승인 요청 항목으로 보고 |
| 2026-08-05 | (추가) preopen 08:50 게이트 실패 원인 확정·수리: 조용한 아침 15m 경계(08:30봉)에서 top100 중 무거래 7개 실측(DKA·WAVES·POKT 등 — 업스트림에도 봉 부재 확인) — 업스트림 확인 프로브 상한 초기값 5가 정당한 무거래일을 계통 장애로 오판. §2.5 대로 데이터 갱신: `MAX_MISSING_UPSTREAM_PROBES` 5→15 (15 초과는 여전히 fail-closed) | R1 발송 자체는 이날도 정상(legacy 기록 채널만 skip). failed 유닛 잔재 2종은 익일 성공 런이 자연 해소 |
| 2026-08-08 | **사용자 운영 결정 — Prelude/R1은 계속, KILL은 pump-v2에만 적용.** 08-05 불변 KILL 자체는 유지하되 과거 공용 guard가 R1까지 막아 08-06~08에 receipt·ledger가 사라지고 close→publish→heartbeat 경보가 연쇄된 집행 결함을 수정. R1에서 v2 verdict 의존성을 제거하고 기존 live-window/ranking/snapshot/receipt gate는 그대로 유지. pump-v2는 유효 KILL 시 scoring·decision·receipt·ledger·Telegram 전에 rc0 정상 no-op, 손상·불완전 terminal pair와 KILL 뒤 산출물은 계속 fail-closed. close는 `skip-terminal-kill` 및 08-06~08 한정 `skip-policy-blocked(forward_valid=false)`로 분리하고, 기존 v2 no-decision 흔적은 삭제하지 않고 heartbeat 분모에서 terminal retirement로 재분류. v2 scoreboard rc21은 안정 상태로 로그만 남김 | 동결 사전등록 판정문은 수정하지 않음. 이는 08-08 사용자의 명시적 집행 범위 변경이며 R1 정렬·모델·라벨·TP/SL·알림 포맷은 무변경. 08-06~08 누락 알림/receipt/PnL은 소급 생성·재발송하지 않음. 적대 리뷰 후 전수 `1,399 passed`; 운영 네트워크에서 distribution-close·preopen-close·publish 모두 rc0, heartbeat 실동작도 `[ok] all checks pass — silent`로 복구 검증 |
| 2026-08-11 | **두 exact gate 사이 신규거래 race 봉쇄** — 09:05 runner의 recommend 첫 검사에서 `KRW-FIL`이 업스트림 무거래로 확인된 뒤, 4h 수집 중 거래소에 09:00 D1 봉이 생겼다. distribution 검사는 이를 실제 DB gap `99/100`으로 정확히 잡았지만, pipeline 전체 1회 reconcile 상한이 recommend에서 이미 소진돼 record-only distribution을 건너뛰고 service가 rc1로 끝났다. 현재 DB는 09:30 수집으로 `100/100` 자연 복구. **수리**: targeted refresh 상한을 `recommend` 1회와 `distribution` 1회로 분리하되 각 gate 재검증 실패는 계속 fail-closed; 양 gate 연속 race·지속실패 회귀를 격리 테스트로 고정. 오늘 R1 Telegram·receipt·R1/R2/A1/pump-v1 원장은 정상이며, 누락된 legacy distribution 표본은 사후 모델 상태로 소급 생성하지 않고 결손 증거로 보존 | R1 정렬·모델·라벨·TP/SL·알림 문구 무수정. 다음 09:05 예약주기 실증 전에는 운영 closure로 과장하지 않음 |
