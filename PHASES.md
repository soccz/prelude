# PHASES.md — 단계별 액션 + 체크리스트

> Phase 별 구체 작업. **체크박스로 진행 추적**. 매 세션 시작 시 이 문서 head 60 줄 보고 어디까지 왔는지 파악 (CLAUDE.md §0).

---

## 현재 상태 (요약, 2026-05-25)

- **현재 운영 후보**: pre-open + distribution beta. decision policy 가 ACTIVE / WATCH_ONLY / SILENCE 로 분리
- **텔레그램 원칙**: 매일 발송. ACTIVE 가 있으면 추천, 없으면 침묵/상태 메시지. WATCH/SILENCE 후보는 shadow ledger + dashboard 검증용으로 기록
- **검증 방향**: 포트폴리오용 아이디어 검증 우선. net PnL / Max DD / hit rate / forward paper 결과로 policy 조정
- **AI quant 포트폴리오 강화**: historical recommendation-quality meta-filter + model card + idea attribution scorecard 추가
- **학습 결과**: `recommendation_quality_meta_label_v1` 학습 완료. holdout 는 손실 축소/정밀도 개선 신호가 있으나 selected n=1 이라 자동 배포는 보류(shadow scoring)
- **아카이브**: detector_v1 / legacy 6-class 모델은 보존, 현재 메인 운영 문맥에서는 후순위

진행 트랙:
- [x] Phase 0 — 8 MD 설계 완료
- [x] Phase 1 — 데이터 수집 / 6-class 모델 / 인프라 (legacy)
- [x] Phase X — leak 발견 → detector 재정의 → C3 채택 → detector_v1 artifact (이번 세션)
- [x] **Phase X+2** — dashboard publish 파이프 (paper_ledger → soccz.github.io 정적 회고) 2026-05-07
- [x] **Phase X+3** — 운영 안전 (DB 백업 + heartbeat 모니터링) 2026-05-26
- [x] **Stage 1 구조 전환** — shadow/paper ledger + ACTIVE-only Telegram + idea validation dashboard
- [x] **Stage 1+ AI quant layer** — recommendation-quality meta-filter + model card/report
- [ ] **Stage 2 live paper 축적** — systemd/cron 운영 후 live shadow 표본 확보
- [ ] **Stage 3** (NOTES 기반 사용자 실제 매매 vs system 추천 비교)
- [x] **Phase X+1 초안** — Distribution head (multi-target) + pre-open trigger 운영 후보
- [x] **Phase X+4** — 정책 채택 (distribution PROMOTE_PAPER + preopen DEMOTE) 2026-05-26 사용자 컨펌
- [x] **Phase X+5** — PUMP hunter rule detector SHADOW 배선 2026-06-04
- [x] **Phase X+6** — exit lab (멀티 청산 잣대) + 운영 강화 (ledger 백업·비용 단일화·CSV 검증) 2026-06-11
- [x] [Research] Cold-start 장중 펌프 (15m 미세동학) — **REJECT** 2026-06-11 (OOS allpick net 전부 음수, robust root 0; _workspace/coldstart_signal-researcher_pump_v1.md)
- [x] [Research] Day-quality gate (죽은 날 침묵) — **REJECT** 2026-06-11 (죽은날 2.8%뿐, replay 개선 permutation p=0.43 noise; 클러스터링 corr 0.52 는 진짜 — pump20 axis 재고 조건)
- [x] [Research] Binance lead-lag (volsurge) — **lift 진짜·배선 보류** 2026-06-11. researcher lift 4.34x (5/5 fold) → evaluator 가 "+1.24% net" 을 일봉 낙관 근사 환상으로 REJECT → 15m 정직 경로 재계산: 전기간 -0.11% (CI 0 포함), **최근 7개월 순수 OOS -0.357% (CI 0 제외), bear_quiet -0.19%**. hit 8.1% vs baseline 5.6% — 증분 피처로는 유효. 다음 주간 retrain 시 b_vol_surge 피처 후보 (모델 변경 = 사용자 컨펌 사안). 독립 룰 배선은 net 양수 확인 전 보류.
- [x] [Research] 델타-사다리 청산 (Veritasium "Trillion Dollar Equation" → Thorp/BSM dynamic 델타헤지 합성) — **SHADOW** 2026-06-25. 4-cell ablation(A=챔피언TP5/SL3, L3/L5=사다리 floor3/5, S5=단일TP5/SL5; 같은 OOF 픽셋 K=3 n=3732, 420일, purged WF embargo5). **핵심: floor 매칭 L5 vs S5 — deep-loss 빈도 0.491→0.303 (-18.7pp, 38%↓) · net +0.0001(동등) · hit +6.4pp, 대신 mean_win -1.7pp(상방 절단)**. = "평균 승자 희생 ↔ deep-loss 빈도↓" 하방-우선 reshaping(downside-first 정합). 비용 robust(conservative+extra5bp 에서도 net차 -0.0005). **ADOPT 아님** — 챔피언 -3% floor 가 절대 하방은 더 타이트(deep-loss 0), 전 cell net 음수(radar-not-strategy 재확인). per-fold 4/4 음수·forward 부재 → ADOPT 상한 SHADOW. quant-reviewer 적대검증 PASS(leak/metric/cost/재현성). 구현: `ledger/exit_lab.py`(walk_ladder_path 외), `scripts/ladder_exit_compare_v1.py`, `tests/test_exit_lab.py`(단일-arm 환원 동등성 회귀 14/14). 설계: `_workspace/ladder_exit_design_v1.md`. 결과: `output/ladder_exit_{compare,perfold,coverage}_v1.*`. 발상: `_workspace/veritasium_trillion_eq_ideas_v1.md`.
- [x] [Research] 군중쏠림 강등 (Veritasium Newton 과확장 함정 → 이미 오른 코인=칼받기) — **SHADOW** 2026-06-25. crowd_index=mean(return_7d/roc_3d/return_5d rank, D-1), honest EOD net(next_close, max-bracket 금지), 5010픽/753일. **full-sample 강함**: HIGH-LOW net차 -0.0126 CI[-0.019,-0.006] 0제외 · deep +0.158 · within-date perm **p=0.002**(day-quality p=0.43 와 달리 통과) · bull_quiet 유의 · score 3밴드 전부 separate(corr(crowd,score)=+0.18=모델 과열 편향). **그러나 quant-evaluator 적대검증에서 운영 생존성 미확립**: (1) deep차(+0.158 최강 주장)는 운영 -3% SL 이 전부 절단 → bracket deep차 **0.000**·net차 -0.0033 CI 0포함, (2) 효과가 **PRE-2025-05(alt 15m 無 구간)에 집중**, 15m 존재 최근 OOS 에선 EOD 효과 자체가 noise(perm **p=0.67**). leak 4대 PASS(binance 낙관근사·day-quality permutation noise 와 다른, 처음부터 정직 EOD). → **ADOPT 아님**(forward 부재 + bracket 소멸 + 최근 OOS noise). DOWNRANK **침묵 배선 보류**(33% 침묵해도 net 0=여전히 못 범). 권고: recommendation_quality record-only flag `crowd_overext=1` + shadow ledger crowd_q 컬럼만, alt 15m 누적 후 forward bracket HIGH-LOW CI 0제외+perm p<0.05 재확인 시 ADOPT 승격. 구현: `scripts/crowding_decay_v1.py`(+`crowding_decay_15m_crosscheck.py`), 판정: `_workspace/crowding_decay_quant-evaluator_verdict.md`, 결과: `output/crowding_decay_{v1,coverage_v1,15m_crosscheck}.*`.
- [x] [Research] Self-impact decay (발사 ACTIVE vs WATCH forward 차 — 영상 효율시장 자가소멸) — **INSUFFICIENT_SAMPLE** 2026-06-25. 하네스(`scripts/self_impact_decay_v1.py`, candles forward 조인) 빌드 완료·재실행 가능. 현재: 30일·전구간 bear(bull 0)·ACTIVE 20(A_TRIPLE) vs WATCH A_TRIPLE **5**. naive ATT(A_TRIPLE)=-0.0341 이나 conf_gap=+47.1 → self-impact(음)와 selection(양) 혼재 분리불가. 힌트: ACTIVE fwd_max +9.9%(레이더는 펌프 포착) vs EOD -5.85%(튀고 덤프) — self-impact 시그니처 후보. 게이트: ACTIVE n≥50 & WATCH A_TRIPLE n≥30(~2-3개월 누적) 후 conf-매칭/RD 로 selection 통제 재추정. `output/self_impact_decay_{v1,coverage_v1}.*`.
- [x] [Research] Exit-timing autopsy (언제 파나 — self_impact 의 +9.9%max→EOD덤프 정조준) — **결론: exit 는 이미 frontier, 변경 불요** 2026-06-25. 장중 autopsy(K=3, 3732픽 420일): 고점 max p50 +5.9%/p90 +21%, **t_max 중앙 0.06(고점은 장 초반에 찍힘)**, EOD p50 −3.4%(이후 하루종일 흘러내림), touched+5% 후 P(EOD>0)=0.54(스파이크 안 유지=코인플립). 청산 토너먼트 13정책: hold_eod −2.14%(최악) → champ TP5/SL3 −0.47%(deep 0) → bestnet TP5/SL8 −0.17%(deep 0.32, −5% 초과로 사용자 성향 위배). **champion 을 net↑·deep≤ 로 동시개선하는 정책 없음** — 사다리(246/123)·부분익절(half5)·trail·TP2~4 전부 못 이김(tp4_sl3 −0.0044 = +0.03pp 노이즈). → **downside-first frontier 에서 champion TP5/SL3 이미 효율적, exit lever 소진. 잃는 건 exit 아니라 진입 엣지.** 사용자 수동매매 규칙: 고점은 장 초반, +5% 익절, EOD 까지 들지 말 것(hold_eod −2.1%). 구현 `scripts/exit_timing_autopsy_v1.py`, `output/exit_timing_autopsy_{v1,coverage_v1}.*`.
- [x] [Research] PRPC 펌프-후 reclaim 확인진입 (entry-timing 마지막 미답축, (b) 데이터/TF 전환) — **DEAD** 2026-06-25. 설계 워크플로(11에이전트) 4후보 중 선정: P4HS(4h돌파) DEAD·SWING-DD(C1+C3 재포장) DEAD·LVG-XS(저변동 시장중립 excess) WEAK·PRPC 선정. kill-test(`scripts/prpc_kill_test_v1.py`, 4h 3.2년, d1 펌프≥12%→과열해소·변동성수축·higher-low consolidation→4h 재돌파, purged 4-fold block-bootstrap): armed 2616/reclaim 825(표본 충분). **B 재돌파진입 net +0.0013 CI[−0.007,+0.009] 0포함·foldPos 2/4 → DEAD. A 즉시눌림목진입(전 armed clean) net −0.0061 CI[−0.011,−0.0002] 유의 음수.** 초기 대조서 즉시진입 +4.4% 는 reclaim-subset 조건부 leak(미래상승 조건)이었고 clean 전수서 소멸. → **entry-timing(관측-후-확인/눌림목)도 exhausted. pump-then-revert 는 timeframe-invariant(4h=일봉=15m), 천장은 진입타이밍 아닌 시장구조.** ★(b)에서 발굴된 유일 실제 엣지 = **LVG-XS cross-sectional excess(진짜·repro 확인, but 현물 long-only 환금불가 — 숏/futures or 불장 필요, prelude 범위 밖)**. 설계 `_workspace/next_research_PRPC_design_v1.md`.
- [ ] [Research] Downside guard / 4h confirmation tier (병렬)
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
- `scripts/pump_detector_v2_today.py` — shadow ledger (`shadow_ledger_pump_hunter_v2.csv`)
  + 🎯 텔레그램. 발사 정책: 후보 ≥1 만 발사 / stale 시 경고 1줄 / 후보 0 정상 = 무소음.
- cron: daily_run_distribution [8/9] binance --days 3 refresh (메인 알림 뒤라 무영향)
  + [9/9] v2 발사. close: v2 ledger 청산 (exit lab 7 잣대 자동).
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
- 아카이브 명시 — collector_1h (37일 stale·미사용), collector_binance_d1 (주간 retrain 전용),
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

**Tear sheet 강화 (2026-05-07 추가)**: pyfolio / quantstats / Bailey & Lopez de Prado (2014) 표준까지 cover. 추가된 metric (총 22+) — Volatility / Skew / Kurt / VaR / CVaR / Tail Ratio / Recovery Factor / Ulcer Index / Common Sense Ratio / W-L streak / **PSR / DSR / MinTRL** / Information Ratio / Beta / Tracking Error vs BTC HODL / Top 5 Drawdowns / Underwater plot / Rolling Sharpe (30d ann) / Monthly returns heatmap / Best & Worst trades / Stratification (regime/setup/score) / Score×PnL scatter / CSV download. PIN 9963 PBKDF2+AES 암호화 + papers viewer 와 동일 패턴.

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

## 비상 / 롤백 절차

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
2. 다음 미체크 액션 1 개 in_progress
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
