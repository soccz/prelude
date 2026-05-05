# prelude

> 업비트 KRW 코인의 **오늘 일봉에서 ≥20% tail pump 가능성이 매우 높은 후보**를 매일 KST 09:05 에 텔레그램으로 알리는 **detector beta**.
> 매수 추천 아님 — 사용자 본인 판단 보조용 silence-heavy 레이더.

```
KST 09:05 ─→ [어제까지 데이터 + detector_v1 (binary tail)] ─→ 후보 0~2건 (silence-heavy)
                                                                    ↓
                                                          사용자가 알림 보고 직접 판단/매매
                                                                    ↓
                                                          실제 매매 결과는 NOTES.md 손글
```

**현재 운영 (Stage 1 dry-run 직전)**:
- model: detector_v1 (XGBoost binary, target = next-day max ≥ 20%)
- regime: bull_quiet + bull_volatile (BTC bear 전체 silence)
- threshold: **0.8815 (full panel OOF p99.95) — 운영 코드에서 재계산 금지**
- cap: 2 per day
- mode: detector beta — 자동매매 X, 텔레그램 알림만

**핵심 원칙 (운영 안전장치)**:
- 시스템은 **알림 + reference ledger** 만. 실거래 자동 주문 X
- threshold 는 `output/detector_threshold.json` 의 고정값 그대로. 라이브 quantile 재계산 절대 X
- BTC bear regime → 침묵 (알림 발사 X)
- alert framing: "≥20% tail pump 후보" (매수 추천 아님)
- Stage 1: cron dry-run, telegram off, 로그만. Stage 2 진입은 사용자 명시 컨펌 후
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
cat README.md | sed -n '/현재 상태/,/Stage 진행/p'  # 운영 상태
cat output/detector_threshold.json                    # 운영 룰 (threshold/regime/cap 고정)
tail -30 NOTES.md                                     # 사용자 새 항목
ls -lt output/detector_log_*.json | head -5           # 최근 dry-run 결과
git log --oneline -5                                  # 최근 커밋
```

### 3. dry-run 수동 실행
```bash
python scripts/predict_today.py                  # 오늘 (default dry-run, telegram off)
python scripts/predict_today.py --asof 2025-11-06 # 과거 시뮬
python scripts/predict_today.py --send-telegram   # Stage 2 진입 후만 (텔레그램 발송)
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

## 현재 상태 (2026-05-03)

**Phase**: detector_v1 artifact 완성 → Stage 1 dry-run cron 등록 대기

- [x] Phase 0/1: 데이터 + 6-class 모델 + 가상 ledger 인프라 (legacy 보존)
- [x] Phase X: 6-class 펌프 분포 → ≥20% tail binary detector 재정의
- [x] Phase X-2 sweep (90 조합 regime × threshold × cap)
- [x] Phase X-2-D Fold/Year stability v1/v2/v3 → **C3 채택** (bull_all p99.95 cap2)
- [x] detector_v1 artifact (`signals/models/ckpt/detector_v1.json` + `output/detector_threshold.json`)
- [x] production path (`signals/detector.py`, `scripts/predict_today.py`, `notifier/format.py::format_detector_beta`)
- [ ] Stage 1: cron dry-run 등록 (사용자 수동 — `deploy/crontab.txt`)
- [ ] Stage 2: telegram 발송 활성화 (Stage 1 1~2주 데이터 본 후)
- [ ] Research: Downside guard, 4h confirmation tier (병렬)

상세 진행/lessons 는 `PHASES.md` 참조.

---

## Stage 진행

```
Stage 0 backtest only ─ 완료
   ↓
Stage 1 cron dry-run ──┐ telegram OFF
                       ├─ output/detector_log_YYYYMMDD.json 누적
                       └─ 1~2주 라이브 분포 관찰
   ↓ 사용자 컨펌
Stage 2 telegram beta ─ 자동매매 X, 알림만
   ↓ 사용자 NOTES 평가
Stage 3 threshold/tier 조정
```

---

## 라이선스

개인 사용. 투자 조언 아님. 실거래 손실 책임은 사용자 본인.
