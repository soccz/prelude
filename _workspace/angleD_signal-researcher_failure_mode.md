# 연구 노트: Angle 4 — failure-mode / anti-pattern mining (layer=coin)

- **가설**: 코어 선행패턴(qv_surge/bounce/momentum/ATR·RV)이 깃발을 세워도 돈을 잃는 것은
  (a) BTC regime, (b) 절대 유동성, (c) 소진상태(이미 많이 오름)에 따라 갈린다. 특히
  bear_volatile 에서 펌프-덤프가 집중되고, TP 도달 전 붕괴가 net 을 죽인다 (RESEARCH §5.2
  "종가유지가 base rate 깎음"의 net-PnL 버전).

- **무엇을 돌렸나**:
  - 스크립트 `scripts/failure_mode_discovery_v1.py` (univariate_precursor_lift_v1 의 leak-free
    panel 재사용 + day-D OHLC outcome 레이어 추가 + BTC regime D-1 join).
  - 7 코어 패턴 × top-decile fire(walk-forward train cutoff, OOS only) → fire 표본의
    hit/dump/fade/sustain + 3 exit(EOD / TP_EOD / TP_SL SL-priority) net(0.15% 차감).
  - 패턴 × 4 regime breakdown (Q2 false-positive 프로파일).
  - 깃발 후 dump vs sustain 을 가르는 15개 D-1 split-cond (Q1 separator).
  - 출력: `output/failure_mode_pattern_regime_v1.csv`, `output/failure_mode_dump_separator_v1.csv`.

- **leak·시간정합성 방어**:
  - 모든 패턴 feature/split-cond = `f_*` (market 별 .shift(1) → D-1 까지만). 코드로 검증: feature set 전부 `f_` prefix.
  - BTC regime = compute_btc_features 후 date 별 .shift(1). 셀프체크: regime_d1[D] == btc_regime[D-1] 전행 True.
  - day-D high/low/close 는 라벨·PnL 경로 시뮬에만. feature/split/pattern 레이어에 day-D OHLC 미사용(grep 확인).
  - fire cutoff = walk-forward train quantile, train 구간 fire=NaN (OOS 평가).
  - 거래비용 0.15% 왕복 항상 차감(net). 진입=day-D open(깃발 D-1 마감 후 09:00 시가).

- **시도 조합 수 (selection deflate)**: 패턴 7 × regime 5 = 35 (Q1/Q2) ; split-cond 15 × 패턴 7 = 105 (Q3) ; exit 3. hand-pick 없음, 전수.

- **1차 결과 (net, 239 markets / 177k coin-days)**:
  - 전체 base: hit_high20=1.7%, spike15 중 pump_then_dump(o2c<0)=9.2%, sustain=74%.
    base net(전체 row): EOD -0.19%, TP_EOD -0.09%, TP_SL -0.34% → **무조건 사면 net 음수.**
  - **net_eod_winrate 가 모든 패턴에서 0.42~0.45 로 균일** → 방향성 깃발 ≠ net 엣지(프로젝트 함정 재확인).
  - **regime 이 failure 를 가른다 (핵심)**:
    - bear_volatile = 최악: dump 12~20%, fade 62~70%, sustain↓ 62~70%, net 음수.
    - bull_volatile / bear_quiet = 최선: dump 5~10%, sustain 76~85%, net 양(+) 다수.
      예) mom_ret_7d bull_volatile TP_EOD +0.49% net / EOD +0.29%; qv_surge_7d bear_quiet TP_EOD +0.46%.
  - **유동성 separator (사용자 직관 반전)**: f_log_qv HIGH(고유동) → net 더 나쁨(-1.1~-1.3% TP_SL),
    f_qv_rank HIGH(저유동) → net 더 좋음. 7/7 패턴 일관. "저유동=덤프위험" 직관과 반대 방향.
  - **exit**: TP_EOD 가 EOD·TP_SL 보다 거의 항상 우월. SL-priority(일봉 보수) TP_SL 은 모든 패턴 net 최악
    → 일봉에서 SL 먼저 박는 청산은 net 을 죽인다. 시간 손절은 15m path 검증 필요.

- **evaluator 검증 요청 (의심 지점)**:
  1. **유동성 반전이 진짜인가, SL-priority 일봉 비관성 artifact 인가**: 저유동 고변동 종목이 +20% high 를
     더 자주 찍어 TP_EOD 가 유리해 보일 수 있음. intrabar 순서 미상 → 15m path 로 TP-before-SL 재검증 권장.
  2. **net 의 절대값이 작다(±0.1~0.5%)** → 거래비용·표본 noise 대비 bootstrap CI 필요. 채택 아닌 SHADOW 후보.
  3. regime 분류는 BTC 1개 시계열 기반 → regime 표본이 시기에 쏠릴 수 있음(2024 bull 집중 등). 시기별 안정성 확인 요망.
  4. fire cutoff decile=0.9 단일값. sweep 안 함(조합수 억제) → 민감도는 추후.

- **라이브러리 부착용 (각 패턴 failure-mode + 권장 exit)**:
  - 모든 코어 패턴: regime gate 권장 — **bear_volatile 에서 발동 억제/사이즈 축소**, bull_volatile/bear_quiet 선호.
  - exit: TP_EOD(고점 TP 도달 시 익절, 아니면 종가) 가 일봉 기준 최선. 하드 -SL 은 일봉에선 net 손해.
  - 유동성: 저유동(고변동) 쪽이 TP 타격률↑ → 단 슬리피지 0.05% 가정의 현실성을 evaluator 가 점검.
