# CLAUDE.md — prelude 작업 규칙

이 문서는 Claude 가 이 프로젝트 (`/home/soccz/22tb/prelude/`) 에서 따라야 할 규칙이다.
글로벌 `~/.claude/CLAUDE.md` 와 auto memory 규칙은 그대로 적용되고, 이 문서는 그 위에 프로젝트 한정 규칙을 더한다.

---

## 0. 매 세션 시작 의례 (30초)

새 세션 시작 시 무조건 이 순서:
1. `cat PHASES.md | head -60` — 마지막 체크박스 확인 (어디까지 왔는지)
2. `cat NOTES.md | tail -30` — 사용자가 새 항목 적었는지 확인
3. `tail -10 output/ledger.csv 2>/dev/null` — 최근 가상 ledger 결과 확인
4. `git log --oneline -5` — 최근 커밋 5 개
5. 사용자가 명시 task 주면 그것부터, 없으면 PHASES 다음 항목

이 의례를 거쳐야 컨텍스트가 빨리 잡힘. 매번 시작 시 5 줄 이하로 보고.

---

## 1. 이 프로젝트의 정체 (한 줄)

업비트 KRW 코인의 **오늘 일봉 (KST 09:00 시작) 이 안정적으로 X% 이상 오를지** 매일 KST 08:30 에 텔레그램으로 알리는 **개인 트레이딩 보조 시스템**.

- **사용자**: 알림 받고 본인 판단으로 직접 매매. 매매 결과는 NOTES.md 에 손글
- **시스템**: 알림 발사 + 가상 ledger 자동 기록 + 가상 성과 추적 (실거래 자동 주문 X)
- **목적**: 트레이딩 PnL 향상. **논문 아님.**

---

## 2. 핵심 작업 규칙

### 2.1 모든 코드는 prelude 안에 self-contained
**다른 폴더 (gan_t, xsec_alpha, fin) 에서 import 하지 않는다.** 참고는 OK 이지만 코드는 prelude 안에 새로 짠다.

이유:
- 다른 폴더 변경 시 prelude 가 깨지지 않음
- 백업·이동·공유 시 prelude 폴더 하나로 됨
- 의존성 명확 (이 폴더 안의 것만 책임)

작업 흐름:
1. 다른 폴더에서 비슷한 코드 발견 → 읽고 이해
2. prelude/ 안에 새 모듈로 작성 (필요한 부분만, 이 프로젝트에 맞게)
3. ASSETS.md 에 "어디서 어떤 아이디어 가져왔는지" 출처만 기록

### 2.2 시그널 / 가상 ledger / 운영 — 책임 분리
| 책임 | 폴더 | MD |
|---|---|---|
| 시그널 생성 (라벨, 피처, 모델, 추론) | `data/`, `signals/` | SIGNAL.md |
| 가상 포지션 추적 (사이징, 진입/청산, 성과) | `ledger/` | LEDGER.md |
| 매일 자동 운영 (cron, freshness, 텔레그램, drift) | `ops/`, `notifier/`, `deploy/` | OPS.md |

한 파일에 두 책임 섞지 말 것. 시그널 코드에서 텔레그램 보내거나, ledger 에서 모델 불러오거나 X.

### 2.3 트레이딩 결과가 최우선, 학술 표준은 조심해서 사용

**방향성 (사용자 경험 기반)**:
- ✅ **결과 → (성공 후) → 논문화**: 트레이딩 결과 좋아서 그게 논문으로 이어지는 건 OK / 좋다
- ❌ **논문 형식 → (먼저 적용해서) → 결과 좋아짐**: 학술 표준 strict 박아서 그게 결과를 좋게 만들 거라 기대하는 방향은 조심. 사용자가 여러 프로그램에서 직접 본 함정 — 다중검정 보정으로 신호 다 죽임, IC 우선 최적화로 PnL 손해, 사전 등록 때문에 발견된 패턴 못 채택 등

**우선순위**:
1. **net Sharpe / Max DD / hit rate / 누적 PnL** — 절대 우선
2. **거래비용 차감 후 net 결과** — 항상 보고
3. **forward (live paper) 검증** — 백테스트만 보고 결정 X
4. (옵션) IC/CRPS — 사후 진단용 (사전 게이트 X)
5. (옵션) 다중검정 보정 — 사후 보고용

다음 학술 도구는 **사후 검증/포장 용도로만**, 사전 lever 로는 X:
- Purged Walk-Forward + embargo (이건 양보 X — 데이터 누수 방어)
- structural_filter 3-stage (사후 검증 가능, 사전 통과 강제 X)
- Holm step-down + DSR (보고할 때 옆에 표기 정도)
- 사전 등록 (선택적 — 발견 채택 막지 말 것)
- IC/CRPS/PI_80 (트레이딩 결과 옆에 진단치로)

**학술 표준이 결과 나쁘게 만들면 즉시 폐기.** 예: structural_filter 통과한 라벨이 PnL 안 나오면 학술 무시하고 PnL 좋은 라벨 채택.

**양보 X (이건 학술 표준이 아니라 위생)**:
- look-ahead 누수 방어 (입력은 t-1 까지, 타겟은 t 이후)
- 유니버스 시간정합성 (fold train 종료 시점 기준)
- 거래비용 항상 차감 (gross 결과는 환상)

### 2.5 데이터가 결정한다 — 모든 숫자는 placeholder

이 프로젝트의 **모든 숫자** — 라벨 X / Y, lookback 격자, 포지션 K, max position, max exposure, BTC regime cut, 거래비용, kill switch, σ-tier cutoff, preflight 임계, retrain cadence, drift cutoff, promotion gate, universe 크기, BTC MA / RV 윈도우 등 — 은 **초기값 (placeholder) 일 뿐**.

원칙:
- **절대적으로 따라야 할 숫자 없음**. 첫 EDA 또는 라이브 결과가 다른 값을 더 좋게 만들면 즉시 변경
- **모든 숫자에 "초기값"** 표기 + "데이터 기반 조정 process" 명시
- "그 결정 왜?" 질문에 항상 데이터 답변 가능해야
- 학술적 사전 등록 (data snooping 방어) 가 트레이딩 결과 손해면 그 사전 등록 무시 (§2.3 와 일관)

**예외 (이 4 가지만 양보 X)**:
- look-ahead 누수 방어 (위생)
- 유니버스 시간정합성 (위생)
- 거래비용 항상 차감 (위생)
- 실거래 자동 주문 X (사용자 명시 전, §3.1)

이 4 가지 외 모든 결정은 데이터 / 결과 기반으로 변경 가능. Claude 가 "이 숫자가 절대값" 으로 오해하지 말 것.

### 2.4 트레이딩 시스템 우선순위
- 백테스트 결과만 보고 의사결정 X — forward + WF 둘 다 통과해야
- net 결과만 보고 (gross 결과는 환상)
- 거래비용: 업비트 KRW 현물 왕복 0.15% (수수료 0.1% + 슬리피지 0.05%) 기본 차감
- Survivorship bias 방어: 상폐 코인도 데이터 포함

---

## 3. 금지/주의 사항

### 3.1 절대 금지
- **다른 폴더에 새 파일 생성 금지** — 모든 새 파일은 prelude 안에
- **gan_t / xsec_alpha 코드 수정 금지** — 그쪽 시스템 깨짐. 필요하면 prelude 로 fork 후 수정
- **다른 폴더에서 import 금지** — `from gan_t.*` `from xsec_alpha.*` 절대 X
- **실거래 API 자동 주문 코드 추가 금지** (사용자 명시 전). 업비트 API key 사용 금지
- **`/tmp` 직접 사용 금지** → `/home/soccz/22tb/tmp` 사용
- **`/mnt/20t` 직접 참조 금지** → `/home/soccz/22tb` 통해 접근

### 3.2 주의
- **자동 학습/재학습 트리거 시 사용자 컨펌**. 매주 1회 retrain OK 이지만, 새 모델 배포는 promotion gate 통과 필요
- **알림 포맷 변경 시 사용자 컨펌**. 매일 보는 거라 안정적이어야
- **라벨 정의 (X/Y) 변경 시 사용자 컨펌**

---

## 4. 의사결정 권한 분리

| 결정 | 누가 |
|---|---|
| 라벨 X/Y/N 변경 | 사용자 (Claude 는 데이터 분석/추천만) |
| 모델 architecture 변경 | 사용자 (Claude 는 비교 실험 후 추천) |
| 새 피처 추가 | Claude OK (단 기여 검증 후 commit) |
| 알림 포맷 변경 | 사용자 |
| 가상 ledger 사이징 룰 변경 | 사용자 |
| Phase 진행 (1→2→3) | 사용자 |
| `notebooks/` 안 자유 실험 | Claude OK |
| 백테스트 스크립트 작성 | Claude OK |
| 자산 참조 (코드 읽기) | Claude OK |

---

## 5. 자주 쓰는 명령

세션 시작 / 작업 중 자주 쓸 거. 외워둘 필요 없음 — 여기서 복사.

```bash
# 1. 매 세션 시작 의례
cat PHASES.md | head -60
tail -30 NOTES.md
tail -10 output/ledger.csv 2>/dev/null
git log --oneline -5

# 2. 데이터 freshness 확인
python -c "import sqlite3; c=sqlite3.connect('data/upbit_d1.db'); print(c.execute('SELECT MAX(timestamp) FROM crypto_data').fetchone())"

# 3. 어제 추천 결과 확인
tail -20 output/predictions_$(date -d yesterday +%Y%m%d).csv 2>/dev/null

# 4. 가상 ledger 누적 PnL
python scripts/ledger_summary.py --days 30

# 5. drift 상태
cat output/drift_state.json 2>/dev/null

# 6. 수동 추론 (오늘 알림 미리)
python scripts/predict_today.py --dry-run

# 7. 백테스트 1개 실행
python scripts/backtest_wf.py --label-x 0.08 --label-y 0.03 --n-folds 5

# 8. 라벨 sweep
python scripts/label_sweep.py --x-grid 0.05,0.08,0.10,0.15 --y-grid 0.02,0.03,0.05
```

---

## 6. 폴더 구조 (코드도 다 prelude 안)

```
prelude/
├── *.md (8 개)              # 설계 문서
├── .claude/                  # Claude Code 세팅
│   ├── settings.local.json   # 권한, 환경
│   └── agents/               # (필요 시) sub-agent
├── data/                     # 데이터 수집 + DB
├── signals/                  # 라벨, 피처, 모델, 추론
├── ledger/                   # 가상 포지션 추적
├── ops/                      # preflight, drift, ic_gate, retrain
├── notifier/                 # 텔레그램
├── scripts/                  # 실행 스크립트 (백테스트, sweep, 일일 추론)
├── notebooks/                # 실험 / EDA
├── output/                   # 결과물 (예측 CSV, ledger, drift state)
├── tests/                    # pytest
├── deploy/                   # cron / systemd
└── requirements.txt
```

세부 구조와 각 모듈 책임은 SIGNAL.md / LEDGER.md / OPS.md 참조.

---

## 7. Claude Code 활용 팁

### 7.1 자주 쓸 도구
- **Read / Edit / Write**: 코드 / MD 수정
- **Bash**: 데이터 / 백테스트 / cron 실행
- **TodoWrite**: 3 단계 이상 task 추적
- **Agent (Explore)**: 코드 탐색 (다른 폴더 — gan_t, fin 등 — 참조 시)
- **Agent (Plan)**: 새 기능 설계
- **WebFetch / WebSearch**: 최신 라이브러리 / API 변경 확인

### 7.2 Sub-agent 후보 (필요 시 .claude/agents/ 추가)
- `signal-debugger`: 시그널 이상 진단 (왜 0 추천? 왜 모든 코인이 같은 점수?)
- `backtest-runner`: 백테스트 자동 실행 + 결과 비교
- `ledger-auditor`: 가상 ledger 일관성 검증 (텔레그램 ↔ ledger ↔ DB)

처음엔 만들지 말고, Phase 진행하다가 같은 task 가 3 번 이상 반복되면 그때 추가.

### 7.3 Plan 모드 활용
새 모델 / 큰 feature 추가 전엔 ExitPlanMode 로 plan 짜고 사용자 승인 받기. 작은 fix 는 바로.

### 7.4 메모리
- 글로벌 auto memory (`~/.claude/projects/-mnt-20t/memory/`) 그대로 적용
- 이 프로젝트 한정 메모리는 `project_prelude_*.md` 형식으로 글로벌 memory 에 저장
- NOTES.md (사용자 손글) 와 별개

---

## 8. 새 MD 추가 / 수정 규칙

- 8 개 핵심 MD (README, SIGNAL, LEDGER, OPS, ASSETS, PHASES, CLAUDE, NOTES) 외 새 MD 추가 시 사용자 컨펌
- README 는 다른 7 개 갱신 후 마지막에 한 번에 갱신
- NOTES.md 는 사용자 손글 — Claude 는 읽기만 (수정 X, 단 사용자 명시 시 OK)
- 큰 변경 (architecture, label 정의, 알림 포맷) 후엔 PHASES.md 에 변경 기록
