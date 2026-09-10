# prelude

> **업비트 KRW 코인 중 하락 가능성은 낮고 상승 가능성은 높은 후보**를 찾아
> KST 08:50·09:05에 알려 주는 개인 트레이딩 보조 레이더.
> 사용자가 직접 판단·매매하며 **자동 주문은 없다.**

![tests](https://img.shields.io/badge/tests-3000%20passed-brightgreen)
![status](https://img.shields.io/badge/verdict-radar--not--strategy-orange)
![evidence](https://img.shields.io/badge/evidence-snapshot%E2%86%92receipt%E2%86%92label-blue)
![judgment](https://img.shields.io/badge/v2%20verdict-KILL%20(early%2C%202026--08--05)-red)

**전체 여정(실패 포함) 공개 보고서** → [soccz.github.io/projects/prelude](https://soccz.github.io/projects/prelude/) ·
**일일 대시보드** → [/dashboard](https://soccz.github.io/projects/prelude/dashboard/) (매일 KST 10:10, PIN 암호화)

**2026-09-10 일일 장애 후속 보완:** 오늘 추천2회와 원장6행은 정상이나 자동 테스트2건이 시간 초과했다.
동시성 테스트의 spawn 준비와 실제 작업 기한을 분리하고, heartbeat에 당일 selftest 완료 확인을 추가했다.
양수인1% 미만 추정값은 `0%` 대신 `<1%`로 표시한다. 점수·순위·모델·과거 발송 증거는 그대로다.
로컬 전수 **3,000 PASS/421.80초** 및 독립 코드 검토를 통과했다. 사용자 실행 후 실제 CPU50%
서비스에서도 **3,000 PASS/878.74초**, 종료exit0과 종합 점검의 `passed` 상태까지 독립 확인했다.
**이번 세 가지 수정의 검증은 완료됐으며 추가 설치·수동 검사는 필요 없다.**
코드·개발일지 공개 준비와 PC/모바일 검증을 마쳤으나 GitHub/Pages에는 아직 게시하지 않았다.
공개 전 과거 문서의 대시보드 PIN 노출을 발견해 본문에서 제거했고,
사용자 결정에 따라 **기존 PIN·암호화 데이터는 유지한다.** 기존 Git 이력은 본문 삭제만으로 보호되지 않는다.
설계·재현 한계·검증·인계는 [PHASES 13차](PHASES.md#실사용-강화-13차--일일-점검-실패확률-표시종합-감시-보완-2026-09-10) 참조.

**2026-09-09 공개 기록 갱신:** 소개 페이지는 아이디어·실패·수리·다음 가설이 이어지는 개발일지다.
9월의 상방 모델/날짜 정합/확률 보정 비교 미채택과 새 체결정보 기록시험을 추가했다.
대시보드는 **현재 R1 장전·장후 전달 증거 / 별도 연구 기록 상태 / 과거 원장**을 구분한다.
“기록 정상”은 “추천 개선 입증”이 아니며, 미관측·미계산 값을0으로 채우지 않는다.
개인 매매 메모는 새 게시 산출물에서 제외한다. 자동 주문·모델 자동 교체·실제 추천 변경은 없다.
전수 **2,915 PASS/436.77초**, 독립 검토·암호화 계약·데스크톱/모바일 가짜 상태 검사를 통과했다.
코드·소개·대시보드 게시를 완료했고, 깨끗한 커밋의 pre-push에서도 **2,915 PASS/431.07초**였다.
Pages 배포 성공과 공개 HTML/암호화5파일의 바이트 일치·동일 생성 세대까지 확인했다.
설계·검증·배포 범위는 [PHASES 12차](PHASES.md#실사용-강화-12차--개발일지현재-대시보드github-공개-반영-2026-09-09) 참조.

**2026-09-09 실제 장애 수리·새 비교 시험 연결:** 첫 정규 수집을 막은 96ns 시각 변환 오류와
전수검사의 스레드 종료 경쟁 조건을 재현해 수정했다. 09-10부터 기존 상위10개에서 체결 정보로 고른3개를
별도 저장하고 원래 추천3개와 같은 날짜에 비교한다. 실패 격리·원본 보존·누락 경고·백업까지 연결했다.
독립 반증과 전수 **2,852 PASS/412.34초**. **실제 추천은 그대로이며 성능 향상은 아직 미검증이다.**

코드 변경은 기존 예약 서비스가 다음 실행에서 읽는다. selftest 때문에 추천이 대기하지 않도록 하는
systemd 한정 설정도 사용자가 `--update-selftest`로 설치했다. 이후 설치19파일의 일치와
loaded 순서 의존 제거·자원상한을 읽기 확인했다. **현재 추가 설치는 필요 없다.**

09-09 실패 원본은 그대로 보존한다.09-10 감사에서는 첫 정상 수집과 두 시험의 진입 전 기록을 확인했다.
성숙한 비교 결과와 실제 추천 개선 여부는 아직 미검증이다.
상세 근거와 정확한 검증 명령은 [PHASES 11차](PHASES.md#실사용-강화-11차--첫-정규-실행-장애-수리추천-경로-격리-2026-09-09),
설치 후 읽기 확인은 [OPS](OPS.md#12-단일-scheduler-계약) 참조.

**2026-09-08 수집 감시 보완(당시 기록):** 수집기가 아예 실행되지 않거나 원본·피처·점수·확정 기록·평가 저장이
중간에 끊긴 경우를 기존10:30 heartbeat가 확인한다. 정상 무교체/관측 결측은 구별하고, 늦은 부팅 시
전일 누락도 확인한다. 장애 주입·독립 검토·전수 **2,531 PASS**. **추가 설치와 추천 변경은 없다.**
운영 기대일09-09와 연구 시작09-08을 구분하며, 검사 범위는 메타데이터 연결·원본 존재/크기다.
원본 내용 전체 무결성·추천 우위·첫 정규 실행 성공을 증명하지 않는다.
검증 명령과 남은 경계는 [PHASES 10차](PHASES.md#실사용-강화-10차--수집-미실행-감시-공백-수리-2026-09-08),
장애 시 확인 방법은 [OPS 수집 감시](OPS.md#수집-미실행중간-중단-감시-2026-09-08-추가) 참조.

**2026-09-08 후속 보완(당시 기록):** 기존 상위10개 안에서 체결 정보로3개를 고르는 **오프라인 선별기**를 구현했다.
당시에는 자동 시험에 연결하지 않았으며09-09의 별도 시험 연결은 위 항목을 따른다. 운영 코드는 R1 원장/라벨 선처리,
연구 명령300초 제한, 수집 전 저장공간 검사(초기5GiB)를 추가했다. 독립 검토와 전수 **2,457 PASS**.
추가 설치는 없으며 다음 예약 실행에서 운영 보완이 사용된다. 기존 실제 백업은 읽기 검증했지만
**원본과 같은 파일시스템**이라 디스크 고장 대비 독립 사본은 아직 없다.
설계·반증·검증 명령·완료 경계는 [PHASES 9차](PHASES.md#실사용-강화-9차--다음-선별안-사전-검증운영-잔여-보완-2026-09-08) 참조.

**2026-09-08 추가 구현:** 공개 체결·호가 수집 → 계산 직전300초 피처 → 별도 시험 순위의 내구 저장 →
기존 추천과 같은 날짜 비교까지 연결했다. 전수2,383개와 후속 백업 회귀21개 PASS,
281시장 실접속·원본 독립 검산도 통과했다. **성능 개선은 미검증이며 R1 추천은 그대로다.**
**새 수집기 설치 확인 완료**:9개 timer 활성, 설치파일19개 SHA 일치.
첫 자동 수집은 **09-09 08:45 KST 실행됐으나 피처 생성에 실패**했다.09-09 수리 사항은 상단을 따른다. 당시 검증·인계 경계는
[PHASES 8차](PHASES.md#실사용-강화-8차--새-정보-수집인과-피처운영-격리-2026-09-07)에 정리했다.

**2026-09-07 운영 강화:** 발송 전 시도 기록으로 crash 후 중복 재발송을 차단하고,
확률을 "검증 중 추정치", 09:00 가격을 "현재 체결가격이 아닌 참고가격"으로 정정했다.
평가는 원래 추천·대조군을 고정하며 결과가 빠졌다고 다음 종목으로 대체하지 않는다.
최종2,165개 테스트와 실자료 독립 검산 PASS. **추천 모델·수치는 그대로이며 추천 성능/수익성 향상의
증명은 아니다.** 범위·한계·검증 명령은 [PHASES.md](PHASES.md#실사용-강화-7차--전달측정신규-후보-기본-잠금-2026-09-07),
상태 확인과 장애 대응은 [OPS.md](OPS.md#전달-불확실성과-안전한-상태-확인-2026-09-07)에 정리했다.

---

## 이 프로젝트가 다른 "코인 봇 레포"와 다른 점

수익률 스크린샷이 없다. 대신 이것들이 있다:

1. **박제된 negative result 22건+** — 12실험 exhaustive, 검증 사슬 4가설, 옵션이론 팬아웃 6트랙,
   최후 챌린저 5축(7후보) — 채택 기준을 통과 못 한 모든 가설이 재현 가능한 형태로 남아 있다.
   "좋아 보이는 백테스트"는 여기서 살아남지 못했다.
2. **증거 사슬** — 성과 주장은 전부 `불변 snapshot → Telegram 서버수락 영수증 → 실제 발송시각 이후
   96봉 라벨 → 감사 평가기`를 통과한 forward 표본에서만 나온다. 09:10 알림이 09:00 봉을
   소급 적중하는 류의 왜곡은 구조적으로 불가능하다.
3. **자기 자신도 못 믿는다는 전제** — 모든 산출물은 content-addressed(SHA-256) + 코드 계보 해시로
   봉인되고, pump v2의 생사는 사전등록 동결 판정(GO/KILL, 불변 터미널, 코드로도 뒤집기 불가)이
   결정하게 했다. 그리고 실제로 그렇게 됐다: **2026-08-05, 조기 사망 조항(누적 mean net < 0)이
   n=9 · mean −0.097%에서 자동 발동해 KILL을 집행하고 판정문을 해시로 박제했다.**
   판정에 불리해도 기준은 안 바꾼다 — 는 원칙이 말이 아니라 실행 기록으로 남았다.
4. **정직한 주장 제한** — 과거 계약 표본의 상방 AUC 0.477과 음수 net 결과를 근거로
   "우수 추천기" 대신 **radar-not-strategy**로 규정했다. 이 과거 수치를 현재 전체 후보의 판별력으로
   재사용하지 않는다.09-07 재평가에서는 전체 후보의 상방 판별력과 실제 Top3 선정 성과를 분리했고,
   비교한 개선안의 저하방·고상방 동시 우위는 입증하지 못했다.

---

## 무엇을 하나

```
active KRW − stablecoin 5종 + D1 PIT 거래대금
              ↓ Top100 exact-boundary freshness gate
    각 gate 첫 실패 시 결손 종목만 1회 재수집·재검증
슬롯당 단일 R1 inference snapshot ──→ 발송 intent → Telegram receipt
              ↓                              ↓
      전 유니버스 score 기록        다음 실행 15분봉부터 새 96봉
              └──────────→ forward label / evaluator
```

**하루 시간표 (KST, systemd 9 timer 설치·활성)**

| 시각 | 동작 |
|---|---|
| 07:30 | 전수 pytest selftest (실패 시 OnFailure 경보) |
| 08:45 | 공개 체결·호가 수집 및 별도 비교 기록 (09-10부터 동점+Top10 시험, 추천 발송과 독립) |
| 08:50 | R1 **장전 후보** 발송 (09:00 참고가격은 아직 미확정) |
| 09:05 | R1 **장후 후보** 발송 + challenger shadow ledger (09:00 가격은 참고용, pump-v2는 KILL 후 정상 no-op) |
| 09:30 | 전일 청산 (−3%SL/+5%TP/EOD, 왕복 0.15% 차감) + 챔피언 재선정 |
| 10:05 | 전 유니버스 forward 라벨 + 감사 평가 |
| 10:10 | 암호화 대시보드 publish |
| 10:30 | heartbeat (이상 시만 알림) · 04:00 content-addressed DB 백업 |

유닛 실패 시 `OnFailure`로 Telegram 경보를 시도한다. 서버·스케줄러·네트워크 전체 장애는
같은 서버의 경보만으로 보장할 수 없으며, 독립 외부 감시는 아직 별도 구축 대상이다.

---

## 정직한 성적표 (2026-08-05)

| 주장 | 증거 | 판정 |
|---|---|---|
| 진입 농축(lift)은 진짜 | pump20 hit 8.1% vs base 1.4% (~6×), 전 fold 일관 | ✅ 생존 |
| 하방 판별력은 진짜 | p_dn5 AUC 0.721 | ✅ 생존 |
| 상방 랭킹이 우수하다 | p_up10 AUC **0.477**, 실측 −0.624%/픽 (몽키 −0.424%) | ❌ 주장 철회 |
| 자동 청산 net 흑자 | 12실험 + 챌린저 7후보 전부 net ≤ 0 | ❌ 구조적 미달 |
| 확률은 calibrated | 독립 head 포함관계 위반 36/100, RR 낙관 편향 | ❌ 정렬용 score일 뿐 |
| v2 급등 레이더는 forward에서 살아남는다 | verified closed 9건 · 누적 mean net **−0.097%** → 동결 조기사망 조항 자동 발동 | ❌ **KILL (2026-08-05 집행)** |

위 표는 당시 계약·표본의 역사적 판정이다. 최신 비교·운영 상태는 PHASES 상단을 따른다.
그래서 결론은 하나다: **이 시스템은 자동 수익기가 아니라 하방 위험을 함께 보여 주는 추천 레이더이며,
수익은 사용자의 진입·청산 판단이 결정한다.** 이 문장을 부정하는 지표가 나오면 문서가 먼저 바뀐다.

---

## 연구 연대기 — 실패가 자산이다

| 기간 | 트랙 | 결과 |
|---|---|---|
| 05-31 | 펌프 선행패턴 역분석 → R1 risk-reward 랭커 탄생 | lift 4~4.7× 확인 |
| 06-01 | 12실험 exhaustive (랭킹·청산·필터·regime·멀티데이·라벨·엔진 sweep) | **net 흑자 0개** — 천장 확정 |
| 06-04→11 | 적대 검증 사슬 (researcher → evaluator → 15m 실경로) | 가짜 "+1.24%" 적발 · hit 엣지만 생존 → 🎯 v2 |
| 06-25 | 옵션이론 팬아웃 (델타-사다리·군중쏠림·exit autopsy·PRPC) | 사다리 deep-loss −38% SHADOW · **exit/entry-timing 소진** |
| 07-25 | 최후 챌린저 5축 + 별도 seed 독립 재검산 | **7후보 전원 REJECT** — "상방을 올리면 하방이 따라온다" 정량 확인 |
| 07-25→26 | Track 1 측정 무결성 (증거 사슬 전면 재건) | historical 탐색 **공식 종료** — 이후 판단은 forward만 |
| 07-27→28 | 3중 장애 (발송 사망·close 마비·침묵) → 적대 리뷰 2×2회전 복구 | CRITICAL 7건 적발·해소 · v2 소생(소급 0/53→**53/53 ok**) |
| 07-30 | 09:05 D1 경계 race 재현·수리 | 결손 종목 1회 표적 재수집 + 동일 gate 재검증, 광역/지속 결손 fail-closed |
| 08-11 | 두 gate 사이 신규거래 race 재현·수리 | recommend·distribution 각각 최대 1회 표적 재수집, 두 번째 실패는 계속 fail-closed |

상세 서사와 각 판정의 원자료: [프로젝트 보고서](https://soccz.github.io/projects/prelude/) ·
[`_workspace/`](_workspace/) (연구 노트·독립 재검산 판정서 40여 건) · [`PHASES.md`](PHASES.md)

---

## 아키텍처 — 신뢰를 코드로 강제하는 장치들

- **단일 불변 snapshot**: 슬롯당 점수 계산은 정확히 1회. 발송·원장·평가가 같은 바이트를 읽는다.
- **발송 intent + receipt**: API 전에 durable 시도를 기록하고 서버 수락 영수증과 연결한다.
  미완료·불확실 시도는 자동 재전송하지 않는다. API 전에 죽어 못 보냈어도 보류될 수 있으므로
  **정확히 한 번 전달 보장은 아니다.** 읽기 전용 상태 조회도 최신 시도와 이전 실패를 구분한다.
- **원후보 고정 평가**: TopN·유동성/ATR 대조군은 결과 연결 전에 고정한다. 필요한 결과가 없으면
  비교 불가와 분모를 공개하며 다른 종목이나 부분 평균으로 메우지 않는다.
- **새 모델 기본 잠금**: 새 등록은 `challenger_only=True`가 기본이며 기존7개 모델 설정은 유지한다.
  이 기본값이 사용자 승인·새 모델의 forward 검증 체계 전체를 대신하지는 않는다.
- **fail-closed + fail-loud**: 수집기 빈 페이지, 손상 state, 부분 백업, DB 재구축 — 전부 시끄러운 실패.
  ("조용한 fail-closed는 조용한 fail-open과 같은 얼굴을 하고 있다" — 07-27 장애의 교훈)
- **provenance 봉인**: champion 상태·정책 비교·메타 모델 전부 payload SHA-256 + 입력 manifest.
  입력이 한 바이트 바뀌면 아티팩트가 무효화된다. pickle은 승인 digest 일치 시에만 실행.
- **시한부 판정 박제 → 집행 완료**: v2의 GO/KILL은 불변 터미널 상태 + 독립 anchor로 봉인돼 있었고,
  동결 조기사망 조항이 **2026-08-05 자동 발동해 KILL로 종결**됐다(n=9 · mean −0.097% ·
  `radar_terminal_verdict.json` integrity hash). KILL 이후 일일 live v2 runner는
  scoring·decision·receipt·ledger·전송 전에 정상 no-op으로 종료한다(명시적 진단 dry-run은 가능).
  **이 판정은 v2에만 적용되며 R1 preopen/open은 계속 운영한다.**
  **무증거로 죽는 실험과 데이터로 판정받는 실험은 다르다** — 이 실험은 후자로 죽었다.
- **위생 4원칙 (유일한 비타협)**: look-ahead 차단 · 유니버스 시간정합 · 거래비용 상시 차감 · 자동주문 금지.

---

## 빠른 시작

```bash
cd /home/soccz/22tb/prelude
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 상태 잡기 (30초) — 운영 머신 기준. 신규 클론엔 forward 산출물(output/*.csv)이 없다
head -60 PHASES.md                                    # 현재 단계·동결 경계
tail -10 output/shadow_ledger_recommend.csv           # 최근 R1 forward 결과
git log --oneline -5

# 안전한 수동 점검 (기록·발송 없음)
python scripts/health_check.py --channel recommend --no-telegram
PYTHONDONTWRITEBYTECODE=1 PRELUDE_FORBID_TELEGRAM=1 python -B scripts/report_recommendation_status.py --format text
PYTHONDONTWRITEBYTECODE=1 python -B -m ops.selftest_status --format text  # 실제 마지막 일일검사 확인
sudo bash deploy/install_systemd.sh --check-only      # 설치본-저장소 정합 검사

# 전체 검증
TMPDIR=/home/soccz/22tb/tmp PYTHONDONTWRITEBYTECODE=1 PRELUDE_FORBID_TELEGRAM=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python -B -m pytest -q -p no:cacheprovider tests/  # 2026-09-10 로컬: 3000 passed
```

---

## 폴더 구조

```
prelude/
├── README.md          # ← 지금 이 파일
├── CLAUDE.md          # 작업 규칙 · SIGNAL.md 시그널 · LEDGER.md 원장 · OPS.md 운영
├── PHASES.md          # 단계·판정 기록 (변경 이력 정본) · NOTES.md 사용자 손글(비공개·로컬 전용)
├── ASSETS.md          # 외부 참조 매핑 · ADDITIONAL_IDEAS.md 검증된 결함·로드맵
├── data/              # D1/4h/15m/Binance 수집 + DB (fail-closed collectors)
├── signals/           # 라벨·피처·모델·snapshot·score 라벨러
├── ledger/            # 경로 판정(path_quality)·원자 CSV·포트폴리오 정본 지표
├── ops/               # 승격 게이트·챔피언·provenance·radar verdict·file lock
├── notifier/          # Telegram + durable 발송 intent + delivery receipt
├── scripts/           # 일일 러너·백테스트·챌린저·감사 평가기
├── deploy/            # systemd 19유닛(9타이머) + 한정 교체/복구 지원 installer
├── tests/             # 2026-09-10 로컬 pytest 3000 PASS (warnings=error)
├── _workspace/        # 연구 노트·설계·독립 재검산 판정서 (negative results 박제)
└── output/            # 산출물 (증거 아티팩트는 gitignore + versioned backup)
```

---

## 어디서부터 읽을지

| 누구냐 | 어디부터 |
|---|---|
| 처음 온 사람 | 이 README → [공개 보고서](https://soccz.github.io/projects/prelude/) |
| 새 세션 시작 Claude | `CLAUDE.md` §0 → `PHASES.md` head |
| "진짜 성과 어때?" | `output/shadow_ledger_recommend.csv` + 평가기 리포트 (forward만 믿는다) |
| 시그널 만지려는 사람 | `SIGNAL.md` (§7.4 확률 정합성 제한 필독) |
| 매일 timer 디버깅 | `OPS.md` → `output/cron_*.log` → `journalctl -u prelude-*` |
| 연구 판정 원자료 | `_workspace/challenger_quant_evaluator_verdict_v1.md` 외 40여 건 |

---

## 현재 작업 경계 (2026-08-05)

- [x] Track 1 측정 무결성 + 적대 감사·보강
- [x] 챌린저 5축 종결 (전원 REJECT) — historical 탐색 공식 종료
- [x] 07-27 3중 장애 수리 + v2 후보 생산 회귀 수리 (적대 리뷰 "신뢰 가능")
- [x] systemd 재설치 + failure-alert 가동 (2026-07-28) → 8타이머 체제(07:30 전수 selftest 포함)
- [x] 실전 8일 하드닝 (07-28→08-05): stdout 오염 클래스 fd 봉인 · 테스트 hermeticity 가드 ·
      부팅폭풍 캐치업 직렬화 · 15m 갭치유(--heal-days) · 상폐 종목 구조적 종결(halted)
- [ ] forward 표본 축적 (새 계약 하 매일 자동)
- [x] **v2 동결 판정 — 2026-08-05 조기 KILL 자동 집행** (조기사망 조항 · 기준 무수정 · 판정문 해시 박제)
- [x] 09-10 신규 시험의 첫 정상 수집·진입 전 기록 확인(운영 증거, 성능 판정 아님)
- [ ] 신규 시험의 성숙한 비교 결과 확인. 모델 변경/실제 추천 승격은 검증 근거와 사용자 승인 필요
- [x] 09-10 selftest 준비/작업 기한 분리·작은 확률 표시·종합 감시 누락 보완(로컬3,000 PASS)
- [x] 수정본의 실제 CPU50% selftest3,000 PASS/878.74초 및 native 상태passed 확인(추가 설치 없음)

---

## 라이선스

개인 사용. 투자 조언 아님. 실거래 손실 책임은 사용자 본인.
이 레포의 가장 큰 자산은 수익률이 아니라 **"무엇이 안 되는지"의 재현 가능한 기록**이다.
