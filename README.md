# prelude

> **업비트 KRW 코인 중 하락 가능성은 낮고 상승 가능성은 높은 후보**를 찾아
> KST 08:50·09:05에 알려 주는 개인 트레이딩 보조 레이더.
> 사용자가 직접 판단·매매하며 **자동 주문은 없다.**

![tests](https://img.shields.io/badge/tests-1400%20passed-brightgreen)
![status](https://img.shields.io/badge/verdict-radar--not--strategy-orange)
![evidence](https://img.shields.io/badge/evidence-snapshot%E2%86%92receipt%E2%86%92label-blue)
![judgment](https://img.shields.io/badge/v2%20verdict-KILL%20(early%2C%202026--08--05)-red)

**전체 여정(실패 포함) 공개 보고서** → [soccz.github.io/projects/prelude](https://soccz.github.io/projects/prelude/) ·
**일일 대시보드** → [/dashboard](https://soccz.github.io/projects/prelude/dashboard/) (매일 KST 10:10, PIN 암호화)

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
4. **정직한 자기 강등** — 신형 forward 측정에서 상방 head AUC 0.477(무작위 이하)이 나오자
   시스템 스스로 주장을 "우수 추천기"에서 **radar-not-strategy**로 낮췄다. 살아있는 것은
   진입 농축(base 대비 ~6배)과 하방 head(AUC 0.721)뿐이라고 문서 전체가 말한다.

---

## 무엇을 하나

```
active KRW − stablecoin 5종 + D1 PIT 거래대금
              ↓ Top100 exact-boundary freshness gate
    각 gate 첫 실패 시 결손 종목만 1회 재수집·재검증
슬롯당 단일 R1 inference snapshot ──→ Telegram delivery receipt
              ↓                              ↓
      전 유니버스 score 기록        다음 실행 15분봉부터 새 96봉
              └──────────→ forward label / evaluator
```

**하루 시간표 (KST, systemd 8 timer, 무인)**

| 시각 | 동작 |
|---|---|
| 07:30 | 전수 pytest selftest (실패 시 OnFailure 경보) |
| 08:50 | R1 **예고** 발송 (진입가 09:00 확정) |
| 09:05 | R1 **확정** 발송 + challenger shadow ledger (pump-v2는 KILL 후 정상 no-op) |
| 09:30 | 전일 청산 (−3%SL/+5%TP/EOD, 왕복 0.15% 차감) + 챔피언 재선정 |
| 10:05 | 전 유니버스 forward 라벨 + 감사 평가 |
| 10:10 | 암호화 대시보드 publish |
| 10:30 | heartbeat (이상 시만 알림) · 04:00 content-addressed DB 백업 |

어떤 유닛이든 죽으면 `OnFailure` 경보가 즉시 텔레그램으로 날아온다 — **조용한 실패는 없다.**

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

그래서 결론은 하나다: **이 시스템은 자동 수익기가 아니라 하방-규율 추천 레이더이며,
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
- **delivery receipt**: 발송은 bool이 아니라 서버 수락 영수증(message_id 대조, ambiguous 분류, exactly-once).
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
python scripts/recommend_today.py --slot open --dry-run
sudo bash deploy/install_systemd.sh --check-only      # 설치본-저장소 정합 검사

# 전체 검증
python -m pytest -q tests/                            # 1400 passed 기대
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
├── notifier/          # Telegram + delivery receipt
├── scripts/           # 일일 러너·백테스트·챌린저·감사 평가기
├── deploy/            # systemd 17유닛(8타이머, 07:30 전수 selftest 포함) + 트랜잭션 installer
├── tests/             # pytest 1362 (warnings=error)
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
- [ ] 사용자 승인 대기: OOF calibration 재구축 → 상방 head 재건 → paired downside veto

---

## 라이선스

개인 사용. 투자 조언 아님. 실거래 손실 책임은 사용자 본인.
이 레포의 가장 큰 자산은 수익률이 아니라 **"무엇이 안 되는지"의 재현 가능한 기록**이다.
