---
name: ops-steward
description: "prelude 매일 자동 운영 절차. decision_policy 버전관리(PROMOTE/DEMOTE/SHADOW, POLICY_VERSION bump), cron/systemd timer 7개, 데이터 freshness, drift, 텔레그램 알림, heartbeat, DB 백업, dashboard publish 를 다루고, 일일 파이프 이상(왜 0 추천? 텔레그램 안 감? freshness stale? cron 안 돎?)을 진단할 때 사용. ADOPT 된 후보를 안전하게 라이브에 반영한다. 운영·cron·텔레그램·정책·대시보드 관련 요청이면 이 스킬을 따른다. 후속: 운영 디버깅/정책 갱신/timer 수정/알림 포맷 점검 요청 시에도 사용. 세션 첫 트리거면 prelude-quant 오케스트레이터가 세션 의례를 먼저 돌린 뒤 이 스킬로 라우팅하는 것이 기본."
---

# ops-steward — 매일 도는 시스템을 안전하게 운영한다

prelude 는 매일 텔레그램 알림 **2건**(① **pre-open 08:50 KST** — 09:00 개장 직전, 현재 DEMOTED/shadow ② **distribution 09:05 KST** — 개장 후, ACTIVE) → 가상 ledger → KST 10:10 dashboard publish 로 도는 라이브 시스템이다. 이 스킬은 그 파이프를 안전하게 유지하고, 채택된 후보를 운영에 반영하며, 이상을 진단하는 절차다.

## 0. 운영 안전장치 (양보 X)
- **threshold 라이브 재계산 절대 금지** — `output/detector_threshold.json` 고정값(0.8815 등) 그대로. 라이브 quantile 재계산은 leak.
- **실거래 자동 주문 코드 금지** (사용자 명시 전). 업비트 API key 사용 X. 시스템 = 알림 + reference ledger 만.
- **거래비용 항상 차감** — ledger·dashboard PnL 은 0.15% 왕복 차감 net.
- DB 백업 = sqlite `.backup`(atomic, lock 없이) + `PRAGMA integrity_check`. 1년치 + 상폐 코인은 영구 손실 위험.

## 1. decision_policy 버전관리 (`ops/decision_policy.py`)
채널별 decision: **ACTIVE**(추천 발사) / **WATCH_ONLY**(shadow 기록만) / **SILENCE**(침묵).
현재 상태:
- **distribution = PROMOTE_PAPER** (setup_quality_policy_v1, ACTIVE 가능)
- **preopen = DEMOTED → WATCH_ONLY** (`PREOPEN_DEMOTED=True`, bear_volatile 만 SILENCE). observed -40.8%/88 alerts, replay active 0건이라 강등.
- POLICY_VERSION = `2026-05-26.1`

**정책 변경 절차:**
1. quant-evaluator 판정 카드(ADOPT/SHADOW) 확인 — 판정 없이 PROMOTE 금지.
2. ADOPT → 해당 채널 ACTIVE 분기 반영. SHADOW → WATCH_ONLY(shadow ledger 기록만).
3. **POLICY_VERSION bump** + 변경 사유 주석. `scripts/policy_history.py` 히스토리와 일치.
4. **PROMOTE↔DEMOTE 같은 큰 전환은 사용자 컨펌 게이트.**
5. 재활성 예: preopen → `PREOPEN_DEMOTED=False` 한 줄 변경 + version bump(컨펌 후).

## 2. 텔레그램 운영 원칙 (`notifier/`)
- **매일 2건 발송** — pre-open(08:50, 개장 전) + distribution(09:05, 개장 후). ACTIVE 있으면 추천, 없으면 침묵/상태 메시지. WATCH/SILENCE 후보는 shadow ledger + dashboard 검증용 기록.
- preopen 은 DEMOTED 라 매일 "DEMOTED (shadow only)" 한 줄.
- **알림 포맷 변경은 사용자 컨펌** — 매일 보는 거라 안정적이어야. `notifier/format.py::format_*` 수정 전 컨펌.
- 발송 게이트 테스트: `tests/test_telegram_send_gate.py`, `scripts/verify_telegram.py`.

## 3. systemd timer 맵 (`deploy/`, 7개, KST)
| timer | 시각 | 역할 |
|-------|------|------|
| prelude-distribution | 09:05 (개장 후) | distribution 알림 — ACTIVE (daily_run_distribution.sh) |
| prelude-preopen | 08:50 (개장 직전) | preopen trigger 알림 — 현재 DEMOTED, shadow |
| prelude-close | 09:30경 | paper_ledger close-out (다음날 4h bars 실현) |
| prelude-preopen-close | | preopen ledger close-out |
| prelude-publish-dashboard | 10:10 | build + publish → soccz.github.io |
| prelude-backup | 04:00 | DB 백업 (cron 안 도는 새벽) |
| prelude-heartbeat | 10:30 | 모니터링 (publish 후 20분) |

신규 timer 등록은 사용자 sudo 1회: `sudo bash deploy/install_systemd.sh`.

> **08:50 vs 08:55 주의**: cron **발사 시각은 08:50**(timer OnCalendar 23:50 UTC). 코드/모델 docstring 다수에 보이는 **08:55 는 모델의 설계상 결정 시점**(= 09:00 개장 5분 전, 일봉 거의 마감)으로, collector(255코인)+panel build 가 09:00 을 넘기지 않게 발사를 08:55→08:50 으로 앞당긴 이력(`policy_history.py` "systemd preopen 08:55→08:50") 때문에 둘이 공존한다. **08:55 를 08:50 으로 일괄 치환하면 설계 근거가 사라지므로 금지** — 사용자 향 알림 헤더(`notifier/format.py` 렌더값)만 실제 발사 08:50 과 맞춘다.

## 4. 일일 파이프 디버깅 플레이북

**먼저 시그널 문제인지 운영 문제인지 가른다.** 시그널(모델이 이상한 점수)이면 signal-researcher 로 넘긴다. 운영(freshness/cron/network)이면 여기서 처리.

| 증상 | 점검 순서 |
|------|----------|
| 왜 0 추천? | ① freshness(DB MAX timestamp 어제까지 있나) ② regime(BTC bear 면 정상 침묵) ③ threshold/정책(decision=SILENCE/WATCH 인지) ④ 모델 score 분포(전부 낮음 = 정상 or 모델 이상→researcher) |
| 텔레그램 안 감 | ① send gate(Stage/ACTIVE 조건) ② token/chat_id env ③ network ④ `verify_telegram.py` |
| freshness stale | ① collector cron 로그(`output/cron_*.log`) ② network/거래소 API ③ DB lock → 코드(`data/collector_*.py`) 버그·신규 거래소 추가면 signal-researcher 로 핸드오프(data/ 인프라 소유) |
| cron 안 돎 | ① `systemctl --user status prelude-*.timer` ② 로그 ③ TMPDIR/venv 경로 |
| dashboard 안 올라감 | ① publish.log fail ② git push 권한/충돌(pull --rebase --autostash) ③ build JSON sanitize(NaN/Infinity→null) |

## 5. 모니터링 (`scripts/heartbeat.sh`, KST 10:30)
점검: paper_ledger 어제 row 0(7일 연속 0 시 alert) + DB integrity + disk 90% + publish.log 최근 fail. 이상 시 텔레그램 alert, 정상 시 silent. preopen paper_ledger 빈 건 정상(DEMOTED) → `shadow_ledger_preopen` 검사로 대체.

## 6. 사용자 컨펌 게이트 (Claude 단독 결정 금지)
알림 포맷 · 가상 ledger 사이징 룰 · 새 모델 배포(promotion) · 정책 큰 전환 · 자동 재학습 배포 · Phase 진행. 에이전트는 추천/진단까지, 결정은 사용자.

## 7. 출력 — 운영 진단/변경 보고
```
## 운영 {진단|변경}: {대상}
- 증상/요청: {}
- 시그널 vs 운영: {어느 쪽 문제인지}
- 원인: {로그 근거}
- 조치: {코드/정책 변경, POLICY_VERSION X→Y}
- 사용자 필요: {sudo / 컨펌 항목, 없으면 "없음"}
```

## 8. 테스트
운영 코드 변경 후: `pytest tests/test_decision_policy_shadow.py tests/test_policy_gate.py tests/test_distribution_operational_guards.py tests/test_daily_telegram_scripts.py tests/test_health_check.py`. 정책/게이트/알림 변경은 해당 테스트 통과 확인.
