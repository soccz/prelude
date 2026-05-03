# prelude

> 업비트 KRW 코인의 **오늘 일봉 (KST 09:00 시작) 이 안정적으로 X% 이상 오를지** 매일 KST 08:30 에 텔레그램으로 미리 알리는 **개인 트레이딩 보조 시스템**.

```
KST 08:30 ─→ [어제까지 데이터로 추론] ─→ 텔레그램 알림 + 가상 ledger 자동 기록
                                              ↓
                                       사용자가 알림 보고 직접 매매
                                              ↓
                                       실제 매매 결과는 NOTES.md 에 손글
                                              ↓
                                       시스템 가상 vs 사용자 실제 비교
```

**핵심 원칙**:
- 시스템은 **알림 + 가상 기록** 만. 실거래 자동 주문 X
- 결과 (net PnL) 가 최우선. 학술 표준은 사후 검증용
- 모든 코드는 이 폴더 안에 self-contained (gan_t / xsec_alpha import X)

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
cat PHASES.md | head -60          # 어디까지 왔는지
tail -30 NOTES.md                  # 사용자 새 항목
tail -10 output/ledger.csv         # 가상 ledger 최근 결과
git log --oneline -5               # 최근 커밋
```

### 3. 자주 쓰는 명령
`CLAUDE.md §5` 참조.

---

## 폴더 구조

```
prelude/
├── README.md         # ← 지금 이 파일
├── CLAUDE.md         # Claude 작업 규칙 (매 세션 시작 시 자동 적용)
├── SIGNAL.md         # 시그널 생성 (라벨, 피처, 모델, 추론)
├── LEDGER.md         # 가상 ledger (사이징, 추적, 성과)
├── OPS.md            # 매일 자동 운영 (cron, freshness, 텔레그램, drift)
├── ASSETS.md         # 다른 폴더 (gan_t/xsec_alpha/fin/APF) 참조 매핑
├── PHASES.md         # Phase 1/2/3 액션 + 체크박스
├── NOTES.md          # 사용자 손글 (실제 매매 일지, 시스템 외 관찰)
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
├── deploy/           # cron / systemd
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
| **매일 cron 디버깅** | OPS.md → `output/drift_state.json` |
| **실제 매매 일지 적기** | NOTES.md (사용자 직접 적는 곳) |

---

## 8 개 MD 한눈에

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

---

## 현재 상태

**Phase**: Phase 0 완료 → Phase 1 시작 대기

- [x] 8 개 MD 설계 완료
- [x] 폴더 스켈레톤 + .claude 세팅
- [ ] git init + 첫 커밋
- [ ] Phase 1 — 데이터 수집부터

상세 진행 상태는 `PHASES.md` 참조.

---

## 첫 알림까지의 길

```
[지금 - Phase 0 완료]
   ↓ 사용자 컨펌
[Phase 1 - 1~2주]
   ├─ 업비트 KRW 일봉 3년 백필
   ├─ 라벨 EDA + X/Y 결정 (사용자 컨펌)
   ├─ XGBoost 학습 + Purged WF 백테스트
   ├─ σ-bucket calibration
   ├─ 가상 ledger 셋업
   ├─ 텔레그램 봇 발급 (사용자)
   ├─ dry-run + 포맷 검토
   └─ ★ 첫 KST 08:30 라이브 알림
```

---

## 라이선스

개인 사용. 투자 조언 아님. 실거래 손실 책임은 사용자 본인.
