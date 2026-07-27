# prelude

> 업비트 KRW 코인 중 **하락 가능성은 낮고 상승 가능성은 높은 후보**를 찾아
> KST 08:50·09:05에 알려 주는 개인 트레이딩 보조 레이더.
> 사용자가 직접 판단·매매하며 자동 주문은 없다.

```
active KRW - stablecoin 5종 + D1 PIT 거래대금
              ↓ Top100 exact-boundary freshness gate
슬롯당 단일 R1 inference snapshot ─→ Telegram delivery receipt
              ↓                              ↓
      전 유니버스 score 기록       다음 실행 15분봉부터 새 96봉
              └──────────→ forward label/evaluator
```

**현재 상태 (2026-07-26)**:

- 상태: **radar-not-strategy**. R1 preopen/open과 pump v2 알림 운영, R2/A1 등은 record-only
- R1 진단: `p_dn5` 판별력은 살아 있으나 `p_up10` AUC가 `0.477`; 현재 추천 순위가
  무작위보다 낫다고 주장할 수 없음
- Track 1 + 적대 감사 완료: 단일 snapshot/receipt, 실행시각 이후 경로, 원자적 ledger,
  PIT universe, exact freshness, source provenance, terminal verdict, versioned backup
- 추가 challenger 완료: downside veto·upside·safe-up·first-passage·downside
  semivol 전부 REJECT, 채택 0·SHADOW 0·활성 R1 변경 없음
- 실데이터: signal-eligible KRW 266개(USD1/USDC/USDE/USDS/USDT 제외),
  open PIT Top100 D1·4h exact `100/100`
- 검증: 전체 `609 passed`, SQLite 7개 `quick_check=ok`, 변경 production Python
  mypy 52파일·Ruff·compile·shell syntax 통과
- 실제 forward 상태: 2026-07-26 open R1/R2/A1 snapshot 각 100행, R1 receipt 성공,
  새 계약의 complete label은 아직 0개
- pump v2: 기존 scorecard 205행은 digest로 동결했고 2026-07-27부터
  decision→receipt→ledger strict provenance를 강제
- 확률 주의: 현재 독립 head는 포함관계 위반이 있고 R1의 `p_up10`은 과대,
  `p_dn5`는 과소 추정돼 RR를 calibrated probability로 해석하면 안 됨
- 운영 반영 차단: 저장소 unit은 검증됐지만 `/etc/systemd/system` 설치본 15개가
  전부 stale/missing이다. sudo preflight·재설치 전에는 내일 운영에 반영되지 않음

**핵심 원칙 (운영 안전장치)**:

- 시스템은 **알림 + reference ledger**만 제공. 실거래 자동 주문 X
- look-ahead·유니버스 시간 불일치·거래비용 누락은 허용하지 않음
- 발송 성공 시각 이후의 실행 가능한 가격 경로만 forward 성과로 사용
- 같은 날 추천은 equal-weight, 무추천일은 cash 0%, 수익은 비용 차감 후 복리로 집계
- 활성 정렬·라벨·모델 구조·알림 문구 변경은 사용자 승인 후
- 결과가 최우선이지만 현재 약한 모델을 강한 추천기로 포장하지 않음
- 모든 코드는 이 폴더 안에 self-contained (gan_t / xsec_alpha import X)
- 상세 진단과 발전안은 [`ADDITIONAL_IDEAS.md`](ADDITIONAL_IDEAS.md) 참조
- 이번 후보별 최종 결과는
  [`_workspace/challenger_completion_orchestrator_v1.md`](_workspace/challenger_completion_orchestrator_v1.md)
  및 독립 재계산 판정서
  [`_workspace/challenger_quant_evaluator_verdict_v1.md`](_workspace/challenger_quant_evaluator_verdict_v1.md)
  참조

---

## 빠른 시작

### 1. 환경 셋업
```bash
cd /home/soccz/22tb/prelude
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. 매 세션 컨텍스트 잡기 (30초)
```bash
head -60 PHASES.md                              # 현재 단계·동결 경계
tail -30 NOTES.md                               # 사용자 새 항목
tail -10 output/ledger.csv 2>/dev/null          # 최근 가상 결과
git log --oneline -5                            # 최근 커밋
```

### 3. 안전한 수동 점검
```bash
python scripts/health_check.py --channel recommend --no-telegram
python scripts/health_check.py --channel distribution --no-telegram
# 08:45~08:59 KST에만 preopen D1 gate 점검
python scripts/health_check.py --channel recommend-preopen --no-telegram
python scripts/recommend_today.py --slot open --dry-run
sudo bash deploy/install_systemd.sh --check-only
```

### 4. 자주 쓰는 명령
`CLAUDE.md §5` 참조.

---

## 폴더 구조

```
prelude/
├── README.md         # ← 지금 이 파일
├── CLAUDE.md         # Claude 작업 규칙 (매 세션 시작 시 자동 적용)
├── SIGNAL.md         # 시그널 생성 (라벨, 피처, 모델, 추론)
├── LEDGER.md         # 가상 ledger (사이징, 추적, 성과)
├── OPS.md            # 매일 자동 운영 (systemd, freshness, 텔레그램, drift)
├── ASSETS.md         # 다른 폴더 (gan_t/xsec_alpha/fin/APF) 참조 매핑
├── PHASES.md         # Phase 1/2/3 액션 + 체크박스
├── NOTES.md          # 사용자 손글 (실제 매매 일지, 시스템 외 관찰)
├── ADDITIONAL_IDEAS.md # 검증된 결함·추가 개선 로드맵·구현 현황
├── .claude/          # Claude Code 세팅 (권한, 환경)
├── data/             # 업비트 KRW 일봉 / 4h 수집 + DB
├── signals/          # 라벨 / 피처 / 모델 / 추론
├── ledger/           # 가상 포지션 추적
├── ops/              # preflight / drift / ic_gate / retrain
├── notifier/         # 텔레그램
├── scripts/          # 백테스트 / sweep / 일일 추론 등 실행
├── notebooks/        # EDA / 실험
├── output/           # 결과물 (예측 CSV, ledger, drift state)
├── tests/            # pytest
├── deploy/           # systemd 단일 scheduler
├── requirements.txt
└── .gitignore
```

---

## 어디서부터 읽을지 (역할별)

| 누구냐 | 어디부터 |
|---|---|
| **새 세션 시작 Claude** | CLAUDE.md (§0 매 세션 시작 의례) → PHASES.md |
| **6 개월 후 돌아온 사용자** | 이 README → PHASES.md (현재 단계) |
| **시그널 만지려는 Claude** | SIGNAL.md → ASSETS.md (참조 자산) |
| **가상 ledger 보고 싶은 사용자** | LEDGER.md → `output/ledger.csv` |
| **매일 timer 디버깅** | OPS.md → `output/cron_*.log` |
| **실제 매매 일지 적기** | NOTES.md (사용자 직접 적는 곳) |
| **누적 회고 / 시각화** | [soccz.github.io/projects/prelude/dashboard](https://soccz.github.io/projects/prelude/dashboard/) (매일 KST 10:10 자동 갱신) |

---

## 핵심 8개 MD + 추가 아이디어

| MD | 한 줄 | 갱신 빈도 |
|---|---|---|
| README.md | 폴더 안내판 + 핵심 요약 | 큰 변경 시 |
| **CLAUDE.md** | Claude 가 따라야 할 규칙 | 작업 규칙 변경 시 |
| **SIGNAL.md** | 어떤 코인이 오를 것 같다 (라벨 / 피처 / 모델) | 모델·라벨 변경 시 |
| **LEDGER.md** | 가상 자본 어떻게 배분 / 추적 | 사이징 룰 변경 시 |
| **OPS.md** | 매일 자동으로 어떻게 돌고 추적 | 운영 변경 시 |
| **ASSETS.md** | gan_t / xsec_alpha / fin / APF 참조 매핑 | 새 자산 발견 시 |
| **PHASES.md** | Phase 1/2/3 단계별 체크리스트 | 매 작업 단위 (자주) |
| **NOTES.md** | 사용자 손글 (실매매 / 관찰) | 사용자 자유 (자주) |
| **ADDITIONAL_IDEAS.md** | 저하방·고상방 추천을 위한 검증 진단과 발전 로드맵 | 큰 검증·구현 시 |

---

## 현재 작업 경계

- [x] Track 1: 추천 성과를 믿을 수 있게 만드는 측정·재현·운영 수리
- [x] R1 preopen/open 발송과 전용 원장, 전 유니버스 forward 축적 경로
- [x] 신규상장 자동 편입과 PIT Top100 exact-boundary coverage gate
- [x] v2 과거 205행 scorecard digest 동결 + 2026-07-27 strict evidence 전환
- [ ] systemd root preflight·`/etc` 재설치: `sudo bash deploy/install_systemd.sh`
- [ ] 2026-09-01 동결 판정
- [ ] GO 및 사용자 승인 시 expanding OOF → 상방 head → paired downside veto 순으로 진행

상세 진행은 `PHASES.md`, 모델 발전안은 `ADDITIONAL_IDEAS.md`를 기준으로 한다.

---

## Stage 진행

```
Track 1 측정 정상화 ─ 완료
   ↓
전 유니버스 forward 축적 ─ open 1일 시작, complete label 대기
   ↓
2026-09-01 기존 v2 GO/KILL 판정
   ↓ GO + 사용자 승인
진짜 OOF calibration → 상방 head 재구축 → paired downside veto
```

---

## 라이선스

개인 사용. 투자 조언 아님. 실거래 손실 책임은 사용자 본인.
