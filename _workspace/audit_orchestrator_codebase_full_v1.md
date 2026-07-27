# prelude 전체 코드베이스 감사 v1 (2026-07-25)

> 11-에이전트 워크플로우(정독 7 + 비판 렌즈 4, 1.22M tokens, 240 tool uses) 종합.
> 목적: "뭘 하는 코드인지 하나하나 확인 + 발전 가능성/개선점/단점 보완" (사용자 요청).
> 에이전트 원본 JSON: 세션 scratchpad `agent_00~10_*.json` (요약 손실 없이 이 문서에 통합).

---

## I. 서브시스템 지도

### data/ — 수집기 5종 + SQLite
- `database.py`: candles 단일 테이블, (market,timestamp) PK UPSERT. **timeout/WAL 미설정**. docstring "항상 UTC"는 오기 (업비트=KST-naive, 바이낸스=UTC-naive 혼합).
- `collector_d1/15m/4h`: pyupbit, "최신 1페이지 incremental + oldest 백필" 동일 패턴 복제. `collector_binance(_d1)`: ccxt, quote_volume=close*volume 근사.
- 인벤토리(실측): upbit_d1 252마켓/19.2만행(신선), 15m 270마켓/875만행/1.55GB(신선), 4h 270마켓/111만행(신선), binance_d1 479마켓(신선). upbit_1h·binance_1h는 2026-05-03/04 정지 아카이브.
- 핵심 결함: ① pyupbit이 예외를 삼켜 None 반환 → retry 무발동, 백필이 일시 장애를 "상장 시작 도달"로 오인해 조용히 절단. ② incremental이 now 기준 1페이지만 → 중간 갭 영구 (15m 기준 50시간+ 중단 시). ③ **d1만 --update라 신규상장 18코인 영구 미유입 (252 vs 270)** — R1/pump 레이더 전부 d1 기반이라 관측 밖. ④ retrain_run.sh:19가 존재하지 않는 `collector_4h --update` 호출 (argparse 에러가 `|| true`로 삼켜짐). ⑤ freshness 게이트는 KRW-BTC 단일 샘플.

### signals/ — 시그널 심장부 (3세대 겹층)
- 1세대(legacy): labels/features/xgb_phase1/predict/retrain/validate/calibration — phase1 6-class. 라이브 미사용. predict.py CLI 기본 ckpt(phase1_smoke.json)는 실존하지 않아 즉시 에러. retrain.py promotion gate는 docstring(net Sharpe)과 달리 brier/accuracy만 비교하고 대상이 legacy라 돌아도 무의미.
- 2세대: labels_distribution(7-head h1~h7)/distribution_engine/setups(S01~S04)/detector(binary tail)/bucket_calibration/labels_preopen/precursors — 전부 record-only 또는 challenger 잠금. distribution_engine은 observed -45% vs replay +71.7% 갭으로 발송 영구 잠금(registry에 사유 명시).
- 3세대(현행 라이브): **recommend.py(820줄) score_candidates가 유일한 ACTIVE 발송 모델** — 8개 D-1 피처 cross-section pct-rank 평균 + XGB up/down head 5개(p_up5/10/20, p_dn5/10), R1 = p_up10/max(p_dn5,1e-3) 내림차순 top-3. SL-3%/TP+5% 플랜. pump_detector_v1(record-only)/v2(🎯 radar, roc_7d_rank>0.85 AND binance vol_surge>1.5, net 음수 정직 고지).
- leak 방어 실태: 라이브 R1 경로는 다층 견고 — market별 raw.shift(1)(univariate_precursor_lift_v1.py:146), BTC regime 1일 shift, head/calib train은 feature_date-5일 embargo 이전 컷. 단 **"OOF" 주석의 calibration이 실제론 in-sample**(recommend.py:356 raw_tr=m.predict_proba(Xtr)) — bucket 경계 낙관 편향 가능, forward ledger가 최종 방어선.
- 기타: SL/TP 상수 4곳 중복, embargo 5(recommend) vs 10(validate) 불일치, 피처 엔지니어링 3중 구현(features.py / univariate_precursor_lift / pump_detector 내장), CWD 상대경로 파일 3개(detector/distribution_engine/bucket_calibration), recommendation_quality ckpt가 signals 트리에 있으나 소유는 ops(§2.2 혼재).

### ledger/ — 가상 포지션 추적
- 현행 운영: shadow.py(65컬럼 스냅샷 append, date+channel 멱등) → close_recommend_ledger가 15m 경로 SL3/TP5 청산 + exit_lab 변형 7종 병기. 레거시: tracker/sizing/risk/metrics(운영 미배선 — 머리말 명시, risk_state.json 부재).
- exit_lab.py: 같은 봉 SL 우선, 사다리 arm은 entry×(1+arm) 정확 비율(룩어헤드 차단), 수수료 명목비례 — tests/test_exit_lab.py 14케이스가 walk_path ≡ simulate_path 동등성 강제. 잘 만든 부분.
- 결함: ① **tracker.py:206-218 EOD 청산가가 다음날 09:00 봉의 close(=진입+48h 가격)를 읽는 의도-코드 불일치 의혹** — 레거시/백테스트 경로 look-ahead성. ② TP/SL 체결 정확 레벨 가정(갭 통과 무시) → 저유동 알트 급락 손실 과소평가. ③ config.py:72 LEDGER_CSV="output/ledger.csv"는 실존하지 않는 stale 경로(CLAUDE.md §0 의례도 동일 참조). ④ shadow dedup이 스냅샷 단위 all-or-nothing이라 부분 기록 후 재실행 시 나머지 영구 누락.
- 실데이터: paper_ledger 164행(163 closed), shadow 7종(recommend/r2/sustain 각 ~165, pump v1 780, v2 205, distribution 502, preopen 480) 전부 07-25까지 갱신 중.

### ops/ — 의사결정 정책
- decision_policy: ACTIVE/WATCH_ONLY/SILENCE 정본. bear_volatile→전부 SILENCE, bear_quiet는 A_TRIPLE+hit≥55+edge≥0만 ACTIVE. PREOPEN_DEMOTED=True 하드코딩. POLICY_VERSION 수동 문자열 bump(잊으면 ledger 귀속 왜곡).
- recommendation_quality: 메타필터(강등 전용). 학습 메타모델 DEPLOYED면 증거 기반 tier를 완전히 덮어씀 — 실현 증거가 명백히 음이어도 model_p≥threshold면 ACTIVE 유지. 증거 net_pnl은 SL 미모델링(낙관 편향).
- champion_selector: forward CLOSED만으로 하방-우선 lexicographic(deep_loss↓→net↑→hit↑) + 5거래일 히스테리시스 완전 자동. **그러나 challenger_only=False 모델이 R1 단 1개라 경쟁이 공허** — net_mean -0.15% 모델 무기한 유지. champion_state.json 파손 1회로 히스테리시스 우회 가능.
- policy_competition: 37 cell(모델×발송정책) record-only 감사. CI 없이 최고 cell 노출(max-of-37 선택편향 예비). N+1 쿼리.
- drift_detector/preflight: 판정만 반환, 집행 미배선. FREEZE가 하루 IC 부호 반전에 과민, 다음날 자동 해제. preflight cal 체크(30일)는 legacy 경로에만 배선 — 라이브에서 0회 실행.
- 비용 상수 동명이의: champion_selector.py:63 `ROUND_TRIP_COST_PCT = ROUND_TRIP_COST_PP`(0.15) vs ledger/config(0.0015) — 같은 이름 100배 차이.

### notifier/ + deploy/ + 셸 — 운영 배선
- systemd 7 timer 단일화 실기계 검증(crontab 비어있음, 07-25 정상 발화): 04:00 backup → 08:50 preopen run → 09:05 distribution run → 09:30 dist close → 10:05 preopen close → 10:10 publish → 10:30 heartbeat. preopen/distribution은 Persistent=false(의도 — stale entry 방지).
- telegram.py 단일 창구(4000자 분할, 3회 재시도). **토큰 미설정 시 dry-run 출력 후 True 반환 — 침묵이 '성공'으로 위장되는 단일 실패점.**
- **아침 발송 러너 2종에 실패 알림 전무**(OnFailure 0건, notify_fail 없음) + daily_run_distribution.sh:54 recommend_today 무가드 → 실패 시 R2/A1/PUMP 기록과 pump v2 발송까지 연쇄 스킵. heartbeat는 당일 run 로그를 안 봐서 탐지 지연 최대 7일.
- heartbeat.sh:169 셸 보간(`'''$MSG'''`)이 특수문자에 취약 — 경보가 경보 내용 때문에 죽을 수 있음.
- backup: sqlite .backup + integrity + ledger tar, 그러나 **원본과 같은 물리 디스크**(/mnt/20t 단일) — 디스크 장애 시 forward 검증 누적 전체 유실. 15m 풀본 14개로 백업 27GB.
- drift 측정(measure_run)·주간 재학습(retrain_run)은 스케줄 미등록 — 기능 사장(사고인지 의도인지 문서화 없음).
- git 미커밋: daily_run_*.sh 수정본(07-18 발송 재개)이 커밋본과 다르게 매일 실행 중 — 문서(최소관심 dry-run 강등)·git·실기계 3자 불일치.

### scripts/ 일일 운영 파이프
- predict_today_distribution(record-only, decision window 08:50-09:20) / predict_preopen_trigger / recommend_send(champion-aware dispatcher, R1 시그니처만 구현) / recommend_today(R1/R2/A1 shadow) / close_* 3종 / pump_detector_*_today / health_check / v2_scoreboard(시한부 판정 일일 채점) / train_recommendation_meta(5중 게이트) / idea_validation_report(허브) / build_dashboard(1921줄, PIN 암호화 publish).
- **v2_scoreboard.py:33 비용 이중 차감 확정**: close_recommend_ledger.py:159가 이미 net(0.15% 차감)으로 저장한 realized_pct에서 COST_PCT=0.15를 재차감. 실측: ledger 직접 집계 mean +0.219%(n=202) vs scoreboard 0.069 — 정확히 0.15%p 차이. policy_competition은 0.219로 정상 → 두 평가 계층이 같은 데이터에 다른 답.
- 중복 구현: build_panel 3벌(+연구까지 16벌), ledger append 멱등 5벌, 15m 경로 조회 2벌, regime 한글 매핑 3벌. close_paper/preopen은 N+1 풀스캔. 죽은 경로: predict_today_legacy, verify_telegram(legacy 전용, tp/sl도 구값). CLAUDE.md §5의 ledger_summary.py/backtest_wf.py/label_sweep.py는 실존하지 않음.
- build_dashboard.py:48 DEFAULT_PIN="9963" 평문.

### scripts/ 연구 원오프 (~60개) + tests/
- 8차 연구 burst 전체 지도: precursor 계열(4h/1h/15m) → detector/label discovery(fold_stability v1→v3, C3 채택) → distribution/preopen 빌드 → pump 선행패턴(Angle A-D → pump_detector_v1) → 추천 레이더+downside-first(→ 현행 R1) → bear_quiet path-exit → 챌린저 10종(R2/A1-A4/B2-B3/C1-C4 전패 — "R1 진입집합 내부에선 수익꼬리·손실꼬리 분리 불가") → binance/coldstart/day-quality → Veritasium 6트랙(사다리 SHADOW, crowding SHADOW, self-impact 표본부족, exit autopsy=frontier 확정, PRPC DEAD).
- **역방향 결합**: 라이브 recommend.py:65-72와 close_recommend_ledger.py:40이 연구 원오프 4개 모듈(univariate_precursor_lift/regime_split_precursor/downside_head_riskreward/recommender_downside_exit)을 import — 이것 때문에 60개 원오프 아카이브 정리가 불가능한 구조.
- tests 63개(14파일)는 전부 운영 레이어. **signals.recommend/features/validate는 tests/ 참조 0건** — 양보불가 1번(leak 방어 shift(1))이 테스트로 안 잠김. 좋은 패턴 2개: test_pump_detector_v2의 연구 검증값 잠금(0.85/1.5), test_exit_lab의 시뮬 동등성 격자.

---

## II. 평가/통계 감사 (quant-evaluator 렌즈, 직접 재계산 기반)

1. **[HIGH] v2 시한부 판정 이중 차감** — 위 확정. 수정 시 mean 0.069→0.219, CI[-0.308,0.746]. 조기 KILL(mean<0) 마진 3배 정상화. 사전등록 블록에 정오표 필요.
2. **[HIGH] 시한부 판정 승격 기준이 수학적으로 달성 불가** — n=202, sd 3.818%, 4.59건/일 → 판정일 예상 n≈376. CI95 0 제외에 필요 mean ≥0.389% (실제 0.219%). 현 mean으로 필요한 n≈1,168(8개월 추가). 4중 AND라 판정일 승격 확률 구조적 0 = "판정"이 아니라 지연된 KILL. → GO/EXTEND/KILL 3분법 또는 CI 하한>-0.15% 기준으로 사용자 결정 안건화.
3. **[HIGH] v2 mean>0이 3일 꼬리 의존** — 상위 3일 합 +47.7% > 총합 +44.2%. 나머지 41거래일 순손실. trade mean(+) vs day t(-0.2) 부호 갈등. 판정 채점에 (trade mean, day mean, 꼬리제외 mean) 3종 병기 권고.
4. **[MED] deep_loss_freq 전 참가자 0.000** — SL 정확 레벨 체결 가정 탓에 -5% 이하가 정의상 불가 → champion_selector 1순위 기준이 판별력 제로로 퇴화, 하방-우선 가치가 측정계에서 실종. path_min_pct 활용 worst-fill 병기 컬럼 권고.
5. **[MED] meta-filter deployable이 hot 구간 holdout과 겹침** — holdout 07-07~25가 유일한 흑자 월(+25.15%; 5월 -9.16, 6월 -12.48)과 일치. threshold 41회 시험 미보정. 롤링 3-스플릿 권고.
6. **[MED] forward 표본 실태** — 양수 채널은 ACTIVE 31건(n 부족)과 v2 202건(꼬리 의존)뿐; 대량 표본(R1 162 net -101%, pump v1 765 net -290%)은 전부 음수. 표본 실태표 대시보드 고정 패널 권고.
7. **[MED] calibration 82일 동결** — calibration_h*.csv 05-04 이후 미갱신인데 bucket_calibration이 계속 읽음. 30일 체크는 dead path에만 배선.
8. **[LOW] bootstrap이 trade-level iid** — 현 데이터에선 day-block과 차이 미미함을 실측 확인(정직 보고). 판정용은 day-block 표준화 권고.
9. 해소 확인: meta-filter selected n=1 문제는 현재 n_selected=40 + min_selected 게이트로 해소 상태. bootstrap 구현 자체는 정확. 4대 양보불가 중 leak/비용/자동주문금지 코드 준수 확인.

## III. 엣지 발전 가능성 (signal-researcher 렌즈)

죽은 축(재제안 금지): exit 최적화(TP5/SL3 frontier), entry-timing 확인진입(PRPC DEAD), day-quality 발사게이트(p=0.43), coldstart 15m, dip-buy, multiday, R1 hp 튜닝(C4 전패 → Optuna 백로그 강등 권고).

남은 축(모라토리엄 저촉 여부 표기):
1. **b_vol_surge R1 head 증분 피처 편입** [S, 기승인 백로그 — PHASES:32 명문화, 비저촉] lift 4.34x 검증, binance_d1 신선. 모델 변경=사용자 컨펌 사안.
2. **신규상장 18코인 유니버스 수리** [S, 데이터 위생 — 비저촉] 펌프 base rate 최고 세그먼트가 관측 밖. 70일 히스토리 게이트 때문에 수리 지연=관측 개시 지연.
3. **SHADOW 2건(crowding/self_impact) 승격 게이트 자동 채점기** [S, 비저촉] v2_scoreboard 패턴 복제. self_impact는 radar KILL 조건 분기 변수.
4. 4h D-1 요약 피처 [M, post-09-01] — "confirmation 진입"은 PRPC DEAD와 충돌하므로 폐기, D-1 압축(전일 마지막 4h 모멘텀, 고점 유지율)만 생존형.
5. MTF 피처(15m 거래대금 집중도 등 D-1 마감 기준) [M, post-09-01] — ablation이 h5에서 daily+1h+15m 우위 확인한 근거 있음.
6. Regime split 재론 [M, post-09-01] — day-quality REJECT 판정문 스스로 "클러스터링 corr 0.52는 진짜"를 남김. 발사게이트가 아닌 랭킹 조건부화는 미검증 유일 미답축.
7. 1h 축 결정 필요: 공식 폐기 박제 vs 15m→1h 리샘플 뷰(업비트 한정 최저비용). binance 1h는 재가동 외 대안 없음.

## IV. 우선순위 통합 (전 렌즈)

**P0 (판정 무결성 — 09-01 전 필수):**
1. v2_scoreboard 이중 차감 제거 + 과거 채점 재산출 + 정오표 박제 + "비용 1회 차감" 회귀 테스트
2. 시한부 판정 power 재상정 — DECISIONS 보드 안건(판정일 전 데드라인)
3. 판정 채점에 day-level/꼬리제외 병기

**P1 (조용한 죽음 방지 — 지금 표본이 판정 입력):**
4. OnFailure 템플릿 유닛 1개 + 아침 러너 notify_fail + L54 무가드 수정
5. telegram.py 미설정=False 반환 + heartbeat에 토큰 핑/당일 run 로그/15m integrity/git porcelain 4항목
6. ledger tar + d1.db 타 디바이스 2차 백업(수 MB/일)
7. database.py WAL + busy_timeout (2줄)

**P2 (유니버스/구조):**
8. collector_d1 --all 전환 + 신규상장 18코인 백필
9. 운영 의존 연구 모듈 4개 signals/ 승격 → scripts/archive/ 분리 (R1 골든 스냅샷으로 byte-identical 검증)
10. leak 카나리아 + R1 골든 스냅샷 테스트 2개
11. deep_loss worst-fill 병기 컬럼
12. champion 경쟁 공허 노출(candidates_evaluated 기록 + heartbeat 항목)

**P3 (post-09-01 GO 시):**
13. b_vol_surge 편입(사용자 컨펌) → 4h D-1 피처 → regime split 재론 → MTF
14. meta-filter 롤링 3-스플릿, policy_competition cell CI+채택 프로토콜, calibration 재생성 자동화
15. drift/retrain 가동 여부 DECISIONS 등재(현재 사고/의도 구분 불가)

**분해 보류 판정 박제**: recommend.py(820줄)·build_dashboard.py(1921줄) 분해는 리스크>이득 — 하지 않기로 기록.

**문서 드리프트 목록**: README(detector_v1 시절 동결), CLAUDE.md §0/§5(ledger.csv, 실존하지 않는 스크립트 3개), labels.py 음봉 주석, retrain.py docstring, predict_preopen --allow-late-run help.
