# ADDITIONAL_IDEAS.md 검증 + 발전 로드맵 v1 (2026-07-25)

> 3-에이전트 워크플로우(claims 사실검증 / conflicts 판정충돌 / stats 통계비판, 414k tokens) 종합.
> 원본: 세션 scratchpad `ideas_00~02.json`. 선행 감사: `audit_orchestrator_codebase_full_v1.md`.

## 1. 사실 검증 — §3.2 결함 표 10/10 전부 CONFIRMED

| # | 주장 | 실측 근거 요약 |
|---|---|---|
| P0-1 | 알림 ~09:10 도착 vs 원장 09:00 기산 | 실발송 09:09:39~09:10:14 (cron 로그 4일 실측). 첫 15m봉 \|open→close\| 평균 0.90%, **픽의 ~9%는 알림 전 TP/SL 결판**. signed bias는 +0.004%로 무편향, 분산만 큼 |
| P0-2 | 08:50 R1 전용 원장 없음 | recommend_today에 slot 인자 자체가 없음. registry의 recommend_r1_preopen ledger_path는 실존하지 않는 유령 파일 — preopen forward 표본 영구 0 |
| P0-3 | v2 이중차감 | 감사 확정 재확인. mean 0.069 vs 진짜 0.219. **수정해도 CI_0_제외는 여전히 미충족** (감사 II-2 "판정일 승격 확률 구조적 0" 정합) |
| P0-4 | 불완전 경로 CLOSED | closer 가드는 bars=0뿐. **R1 CLOSED의 81%(132/162)가 봉<96**. 단 다수는 무거래 정상 공백 — closer가 수집장애와 구분 못 하는 게 진짜 문제 |
| P0-5 | 신규상장 18종 누락 | AI, ARX, B3, BABY, DATA, DRV, IO, IRYS, O, OPG, PROS, RE, SLX, SPX, TRAC, UP2, VVV, WIF |
| P1-1 | OOF가 in-sample | recommend.py:356 raw_tr=predict_proba(Xtr) 확정 |
| P1-2 | exit/hit 정의 혼재 비교 | distribution=next_close(SL/TP 無) vs R1 계열=SL3/TP5. deep_loss 1순위가 exit 정의 따라 자동 승패 |
| P1-3 | 무가중 합산 | 3계층 전부: champion(거래단위), policy_comp(무정규화), idea_validation(일합산=픽수 레버리지 + **√252 오사용**) |
| P1-4 | exit 0 은폐 | 셸 6종 전파 표 작성. heartbeat/backup 무조건 exit 0, OnFailure 0건 |
| P2-1 | 재현 manifest 부족 | R1은 모델 아티팩트 자체가 없음(매 호출 즉석 재학습, seed 42가 유일한 앵커). DB 14일 백업 후 byte 재현 불가 |

추가 실측: ① 슬롯당 inference 이미 2회 실행(발송용 09:09:54 / 원장용 09:10:08 — 별개 즉석 학습), ② 07-25 top-3 전원 dump_risk_flag=True인 채 발송(veto 미개입 실례), ③ 봉<96 비율 R1 81% vs v2 36% — champion 비교의 숨은 유동성 비대칭.

## 2. 검증에서 나온 최대 신규 사실 — 급소는 하방이 아니라 상방

- **p_dn5(하방 헤드)는 살아있음**: 실현 dn5 대비 AUC 0.721, 삼분위 실현율 0.127/0.352/0.547 단조. 단 균일 1.49배 과소추정(수준 이동, 폭발 아님 — floor 무발동 실측).
- **p_up10(상방 헤드)은 죽어있음**: 픽 내 실현 P(MFE≥10) 대비 **AUC 0.477(동전 이하)**, corr(rr_ratio, realized)=0.041.
- **몽키 베이스라인**: top100 전 코인 매수 + SL3/TP5 + 비용 = forward 창 -0.424%/픽. **R1 실측 -0.624%로 몽키보다 나쁨.**
- **pseudo-veto 실증(사후, n=162)**: 픽 내 p_dn5 상위 1/3 제거 시 net -0.624→-0.387%, dn3_first 0.586→0.514, 상방 유지(0.253→0.257) — veto 방향은 데이터가 지지.
→ 결론: "더 좋은 모델"의 1순위 타깃은 상방 랭킹 재구축, veto는 SHADOW 후보, 하방 보정은 수준 이동 수리.

## 3. 기존 판정 충돌/재탕 지도 (conflicts 렌즈)

**모라토리엄 판정** (PHASES:482 "신규 연구 착수 시 관심누수 증거 = KILL" 조항 실재 확인):
- Phase A: 대체로 **비저촉**(판정 무결성/위생). 회색 2건 — preopen 전용 원장 신설(DEMOTED+dry-run 채널 인프라 투입), 1분/5분·호가 수집 신설.
- Phase B: B2(단일 snapshot) 비저촉 / B1(진짜 OOF) 회색 — 결함 수리지만 head 재적합=모델 변경=사용자 컨펌 / B3·B4 **저촉**.
- Phase C: **전면 저촉** (비저촉 예외: b_vol_surge 기승인 백로그, 신규상장 수리).
- Phase D: 연구 아님이나 알림 포맷=사용자 컨펌 + 대상 채널이 음소거 상태라 문서로만 존재 가능.
- §11-11 "v2 저하방 재평가" = **동결 위반** — DECISIONS 안건으로만.

**재탕 판정**:
- 2단 veto→ranking 구조 = R3_gate(3변형 전패) + A1(net 음수+상방 절단) + meta-filter + dump_risk_flag 4중 기시도. 벽 통과 새 메커니즘 없음. 단 "calibration 선행 + 상방 유지 종료조건"으로 벽을 게이트화한 것은 신규 프레임.
- utility식 = R2 선형 페널티 일반화 (λ grid 전패 판정 무반박).
- C2/C3 죽은 축 재탕: crowding(SHADOW 판정+승격게이트 기존재), 1m/5m ignition(coldstart REJECT+데이터 부재), 변동성 수축(2회 반증), spread/호가(수집기 부재), dump 계열(A1 기학습), breadth 발사게이트(REJECT — 단 "랭킹 피처"로는 미시도).
- first-passage 라벨: 4h granularity로 이미 존재(labels_distribution h1/h4/h7, distribution 발송 잠금 이력). **진짜 미시도분은 "15m granularity FP를 R1 top-3 정렬키로 학습" 조합뿐.**

**진짜 신규 (genuinely_new)**: ① 발송시각 기준 성과 귀속(sent_at/entry_observable_at/delivery_ok), ② 경로 완결성 게이트, ③ 진짜 expanding OOF, ④ 단일 inference snapshot+manifest, ⑤ delivery receipt 스키마, ⑥ downside semivolatility 피처, ⑦ 유동성-매칭 무작위 baseline, ⑧ 15m FP 정렬키(부분).

## 4. 통계 설계 비판 (stats 렌즈, 실데이터 계산)

- **[HIGH] FP 라벨 = 변동성 재포장 위험**: 45,557 coin-day backfill 실측 — 랭킹 키 전부 vol 십분위와 단조(3.7~7.7배 스프레드), veto 키도 동조 증가(자기모순), net EV는 vol에 평평(-0.29~-0.41%). → ATR-배수 barrier 변형 병행 + within-vol-band lift로만 채택 판정.
- **[HIGH] Phase B 종료조건 검증 불가**: "일관된 방향" 무임계 서술은 현 상태(AUC 0.721 + 1.49배 오보정)도 통과. top-3 기록만으론 bucket 검증에 26개월. **전 유니버스(~100/일) 스코어 일일 기록이 단일 최대 lever** (90픽/월 → ~3,000행/월).
- **[HIGH] 표본 산수**: 날짜 공통충격 ICC≈0.30, DEFF≈30.5 (하루 100코인 = 유효 3.3개). veto 검정은 within-day paired contrast로 설계 시 5.9배 효율(48일 vs 281일).
- **[MED] lexicographic 권고 근거 어긋남**: "비율 폭발" 미발생(floor 무발동). 진짜 문제는 죽은 분자(p_up10)와 균일 오보정(p_dn5). 순서 논쟁 전에 B1+상방 재구축 선행.
- **[MED] §5.3 집계 전환은 v2 판정 소급 변경**: day-eqw 전환 시 v2 +0.219→+0.095 반토막 — 사전등록 위반. 구지표 유지+신지표 병기, 차기 판정부터 전환.
- **[MED] INCOMPLETE 규칙 함정**: complete-only 시 top100 coin-day 30.5% 증발 + 활동성 선택편향(P(up5) 0.254→0.298). → "봉 부재=무거래=flat 경로" 정의 + 수집장애는 KRW-BTC 동시간 완결성 대조로만 판별.
- **[MED] horizon 미고정 + 라벨 중복**: 24h→48h에서 base rate 8~10pp 이동. FP vs 임계 라벨 상관 0.878~0.918 — 신규 정보는 18.4%뿐. 24h 박제 + 출력 3축(하방FP/상방FP/경로규모) 축소.

## 5. 실행 로드맵 (종합)

**Track 1 — 지금 (모라토리엄 비저촉, 감사 P0/P1과 합류)**
1. v2 이중차감 수정 + 정오표(동결 블록 본문 무수정) + 과거 채점 재산출
2. 실패 전파(A5): OnFailure 유닛 + telegram 미설정=False + L54 가드
3. 신규상장 18종 수리(A4)
4. 경로 완결성(A2 수정판): flat 정의 + KRW-BTC 대조 + path_complete 컬럼(소급 기입, CLOSED 유지)
5. **B2 확장판: 단일 snapshot + sent_at/delivery_ok + 전 유니버스 스코어 일일 기록** ← Phase B의 전제조건, 지금부터 쌓여야 09-01 이후 검증 가능
6. 집계 공통 함수(√365, day-eqw, 날짜-cluster CI) 신설 — 대시보드 즉시 적용, v2 판정은 구지표+병기

**Track 2 — DECISIONS 안건 (사용자 결정)**
A. v2 판정 power 재상정(GO/EXTEND/KILL 3분법 or CI 기준 교체) — 판정일 전 데드라인
B. preopen 전용 원장 신설 여부(강등 채널 인프라 투입)
C. §1.3 "추천 품질 > PnL" 거버넌스 비준(CLAUDE.md §2.3 우선순위 역전)
D. (선택) 1분/5분 스냅샷 수집 착수 여부

**Track 3 — post-09-01 GO 시 (사용자 컨펌 게이트)**
1. B1 진짜 expanding OOF calibration (p_dn5 1.49배 수준 수리)
2. **상방 헤드 재구축** — p_up10 AUC 0.477이 진짜 급소. b_vol_surge(기승인) 편입과 결합
3. veto SHADOW: within-day paired 사전등록 게이트(Δdn3_first ≥3pp CI 0제외 AND 상방 유지율 ≥80% AND 90거래일 AND leave-one-month-out 부호 유지)
4. FP 라벨 연구: ATR-배수 barrier vs 고정 % 병행 backfill, within-vol-band lift 판정, horizon 24h 박제
5. B3 baseline 7종 + 몽키(-0.424%/픽 forward) + 유동성-매칭 무작위 박제

**하지 말 것 (제안서에서 기각)**: C2/C3 재탕 피처(crowding 백지 재제안·1m/5m ignition·변동성 수축·spread·dump 라벨 계열), utility식 재탕, lexicographic 조기 채택, complete-only 경로 필터, §5.3의 v2 소급 적용, §11-11 자체 재평가 프레임.
