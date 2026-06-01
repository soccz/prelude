---
name: prelude-quant
description: "prelude(업비트 KRW 코인 트레이딩 레이더 + AI 퀀트 검증 시스템) 전용 오케스트레이터. 시그널/모델 연구, 백테스트·walk-forward·sweep, 포트폴리오급 평가지표·leak 감사·채택 판정, decision_policy·cron/systemd·텔레그램·dashboard 운영 중 어느 것이든 prelude 관련 작업 요청 시 이 스킬을 사용. 세션 시작 의례를 먼저 돌리고 적합한 전문 에이전트(signal-researcher / quant-evaluator / ops-steward)로 라우팅한다. 후속: 모델 재학습/재평가/정책 갱신/운영 디버깅/이전 결과 개선/부분 재실행/sweep 다시 돌리기 요청 시에도 반드시 사용. 단순 1줄 질문(현재 상태·파일 위치)은 직접 응답 가능."
---

# prelude-quant Orchestrator

업비트 KRW 코인 개인 트레이딩 레이더 + AI 퀀트 검증 시스템(`/home/soccz/22tb/prelude/`)의 전문가 팀을 조율한다. **최우선 목표: 실제 AI 퀀트 포트폴리오로 통할 수준의 신뢰도·평가지표.** 의미 있는 모델(엣지)이 그 다음, 논문적 아이디어는 보너스.

## 실행 모드: 하이브리드 (서브에이전트 기반)

CLAUDE.md §7.2(서브에이전트는 처음부터 만들지 말고 task 3번 반복되면 추가, lean 우선)에 맞춰 **팀 상시가동을 피하고** 서브에이전트 on-demand 를 기본으로 한다.

| 상황 | 모드 | 방법 |
|------|------|------|
| 단일 영역 작업 (연구만 / 운영 디버깅만) | 서브에이전트 1개 | `Agent(subagent_type, model:"opus")` |
| 독립 병렬 작업 (여러 sweep / 여러 라벨 백테스트) | 서브에이전트 fan-out | 단일 메시지에서 N개 `Agent(run_in_background:true)` |
| 모델/시그널 채택 (생성→검증) | 서브에이전트 파이프라인 | signal-researcher → (파일 핸드오프) → quant-evaluator |
| 횡단 채택 결정이 **반복**되어 실시간 토론이 필요해지면 | (보류) 에이전트 팀 | 이 패턴이 3회 이상 반복될 때 도입 검토 — 지금은 안 만든다 |

> 핵심 흐름은 **팬아웃(병렬 연구) + 생성-검증(researcher→evaluator)** 복합 패턴이다. 팀 통신 오버헤드 없이 파일 기반 핸드오프로 조율한다.

## 에이전트 구성

| 에이전트 | subagent_type | 역할 | 스킬 | 출력 |
|---------|--------------|------|------|------|
| signal-researcher | signal-researcher | 엣지 있는 모델/라벨/피처/백테스트 (leak 방어) | signal-research | `signals/`, `output/*.csv`, 연구 노트 |
| quant-evaluator ★ | quant-evaluator | 포트폴리오급 평가지표 + 위생 감사 + ADOPT/SHADOW/REJECT | quant-eval | 판정 카드 |
| ops-steward | ops-steward | decision_policy·cron·텔레그램·dashboard 운영 | ops-steward | `ops/`,`notifier/`,`deploy/` + 정책 버전 |

모든 `Agent` 호출에 **`model: "opus"`** 명시.

## 4대 양보불가 (모든 Phase 강제 — CLAUDE.md §2.5)
이 넷은 결과가 좋아져도 양보 X. 어떤 에이전트든 위반하면 오케스트레이터가 막는다.
1. **look-ahead leak 방어** — 입력 ≤ t-1, 타겟 ≥ t. `next_*`·LEAK_COLS 제외. (이 프로젝트 same-day leak 2번 이력)
2. **유니버스 시간정합성** — fold train 종료 시점 기준.
3. **거래비용 항상 차감** — 0.15% 왕복. gross 결과는 환상.
4. **실거래 자동주문 X** — 사용자 명시 전. 업비트 API key 사용 X.

## 결과 우선 (CLAUDE.md §2.3)
net Sharpe/MaxDD/hit/누적 PnL 이 1순위. PSR/DSR/Holm/IC/사전등록은 **사후 포장용**이지 사전 게이트가 아니다. 학술표준이 PnL 좋은 후보를 죽이려 하면 옆에 표기만 하고 PnL 기준으로 판정. (단 위의 4대 위생은 예외 — 양보 X.)

## 사용자 컨펌 게이트 (Claude 단독 결정 금지)
라벨 X/Y/N 정의 · 모델 architecture · 알림(텔레그램) 포맷 · 가상 ledger 사이징 룰 · 새 모델 배포(promotion) · Phase 진행(1→2→3) · 정책 큰 전환(PROMOTE↔DEMOTE). 이들은 에이전트가 추천/판정까지만 하고 사용자에게 컨펌받는다.

## 워크플로우

### Phase 0: 세션 의례 + 컨텍스트 확인 (먼저, 30초)
prelude 작업이 처음 트리거되면 CLAUDE.md §0 의례를 실행하고 5줄 이하로 보고:
1. `head -60 PHASES.md` — 마지막 체크박스(어디까지 왔는지)
2. `tail -30 NOTES.md` — 사용자 새 항목
3. `tail -10 output/paper_ledger.csv` — 최근 가상 결과 (구 `output/ledger.csv` 는 현재 미존재 레거시)
4. `git log --oneline -5`
5. `_workspace/` 존재 여부로 실행 모드 결정:
   - 미존재 → 초기 실행
   - 존재 + 부분 수정 요청 → 부분 재실행(해당 에이전트만 재호출, 이전 산출물 경로를 프롬프트에 포함)
   - 존재 + 새 입력 → 새 실행(`_workspace/` 를 `_workspace_{YYYYMMDD_HHMMSS}/` 로 이동 후 재생성)

사용자가 명시 task 주면 그것부터, 없으면 PHASES 다음 항목.

### Phase 1: 라우팅
요청을 영역별로 가른다:

| 요청 신호 | 라우팅 |
|----------|--------|
| 라벨/피처/모델/백테스트/WF/sweep/discovery/"엣지 찾아줘" | signal-researcher |
| 성과지표/Sharpe/DD/tear sheet/"이거 채택해도 돼?"/leak 점검/calibration/검증 | quant-evaluator |
| cron/systemd/freshness/drift/텔레그램/dashboard/"왜 0 추천?"/운영 이상 | ops-steward |
| "새 모델 만들어서 운영 넣자" 같은 end-to-end | researcher → evaluator → (ADOPT 시) ops-steward 파이프라인 |

`_workspace/` 에 입력/중간 산출물 저장. 파일명 컨벤션 `{phase}_{agent}_{artifact}.{ext}`.

### Phase 2: 연구/실행 (필요 시 병렬)
- 단일: `Agent(subagent_type:"signal-researcher", model:"opus", prompt:...)`
- 병렬 fan-out(여러 sweep/라벨): 단일 메시지에서 여러 `Agent(run_in_background:true)`. 각 결과는 `output/` + `_workspace/02_signal-researcher_{변형}.md`.

### Phase 3: 검증 (생성-검증 게이트 — 채택 관련 작업이면 필수)
1. signal-researcher 산출물(연구 노트 + output CSV)을 quant-evaluator 에게 넘긴다(파일 경로 전달, 독립 재계산 지시).
2. quant-evaluator 가 판정 카드(ADOPT/SHADOW/REJECT) 생성 → `_workspace/03_quant-evaluator_verdict.md`.
3. 위생 FAIL(leak 등) → REJECT → researcher 로 재작업(최대 2회). 무한 루프 방지.

### Phase 4: 운영 반영 (ADOPT/SHADOW 일 때만)
1. ADOPT → ops-steward 가 decision_policy 반영(PROMOTE), POLICY_VERSION bump. **단 promotion·정책 큰 전환은 사용자 컨펌 게이트.**
2. SHADOW → ops-steward 가 shadow ledger 기록만(ACTIVE 안 함).
3. REJECT → 운영 반영 없음. PHASES.md 에 기록.

### Phase 5: 정리 + 보고
1. `_workspace/` 보존(사후 검증/감사 추적).
2. 큰 변경(architecture/label/알림 포맷)은 PHASES.md 에 기록.
3. 사용자에게 결과 요약(판정·net 결과·사용자 컨펌/sudo 필요 항목) 보고.
4. **피드백 기회 제공**(강요 X): "결과나 팀 구성에서 바꾸고 싶은 점 있나요?"

## 데이터 전달 프로토콜
- **반환값 기반**(서브 결과를 메인이 수집) + **파일 기반**(`_workspace/` 중간 산출물, `output/` 대용량/감사). 최종 산출물만 사용자 지정 경로, 중간(`_workspace/`)은 보존.
- 에이전트 간 직접 통신은 없음(서브에이전트 모드) — 오케스트레이터가 파일 경로를 다음 에이전트 프롬프트에 넣어 핸드오프.

## 에러 핸들링
| 상황 | 전략 |
|------|------|
| 에이전트 1개 실패 | 1회 재시도. 재실패 시 누락 명시하고 진행, 보고서에 기재 |
| researcher 결과 "너무 좋음" | 채택 전 evaluator 에게 leak 검토 명시 요청 |
| evaluator REJECT(위생 FAIL) | researcher 로 재작업(최대 2회), 그래도 FAIL 이면 사용자 보고 |
| 데이터 충돌(주장 vs 재계산) | 재계산 우선 + 출처 병기, 삭제 X |
| 운영 변경이 기존 동작과 충돌 | 변경 전 상태 명시 + 사용자 컨펌 |

## 테스트 시나리오
### 정상 흐름 (end-to-end 채택)
1. "4h feature 추가해서 sustain head(h3/h4) 살릴 수 있나 보고, 되면 운영 넣자"
2. Phase 0 의례 → Phase 1 라우팅(researcher→evaluator→ops 파이프)
3. signal-researcher 가 4h feature 추가 학습 + WF 백테스트, 연구 노트 생성
4. quant-evaluator 가 net 지표 + leak 감사 → 판정 카드(예: SHADOW — forward 표본 부족)
5. SHADOW 이므로 ops-steward 가 shadow ledger 기록만, ACTIVE 안 함. 사용자에게 "forward 표본 N 더 필요" 보고

### 에러 흐름 (leak 재발)
1. researcher 결과 prec 98% (의심스럽게 좋음)
2. quant-evaluator 가 same-day leak(label_date join 오류) 발견 → REJECT(위생 FAIL)
3. 오케스트레이터가 researcher 로 재작업 지시(day-shift join 수정)
4. 재학습 후 prec 16% (진실한 lift) → 재검증 → 판정
5. 보고서에 "1차 leak 발견→수정" 기록

## 참고
- 프로젝트 규칙: `CLAUDE.md`(§2.2 책임분리, §2.3 결과우선, §2.5 placeholder/4대위생, §7.2 lean)
- 설계/내러티브: `README.md`, `RESEARCH.md`(leak 2번·EDA-hit 함정·calibration 등 hard lessons), `PHASES.md`(현재 단계)
- 책임별: `SIGNAL.md` / `LEDGER.md` / `OPS.md`
