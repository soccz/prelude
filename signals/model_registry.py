"""모델 레지스트리 — champion/challenger 프레임의 모델 카탈로그 (메타데이터만).

champion_selector(ops/champion_selector.py) 가 이 레지스트리를 읽어, 각 모델의 forward
CLOSED ledger 를 하방-우선 점수로 평가하고 slot 별 챔피언을 고른다. ops-steward 의 send
dispatcher 는 선정된 챔피언의 predict 인터페이스(여기 명시)를 호출해 알림을 만든다.

★★★ 이 모듈은 시그널 쪽 메타데이터만 정의한다 (CLAUDE.md §2.2 책임 분리):
    - 텔레그램 발송 X / cron 등록 X / 업비트 자동주문·API key 절대 X.
    - 셀렉터/발송/대시보드 배선은 ops-steward 영역 (이 파일은 그쪽이 읽는 카탈로그).

확장 규율 (새 가설 추가 = dict 한 개 추가):
    MODELS 리스트에 ModelSpec(...) 하나를 append 하면 셀렉터가 자동으로 후보에 포함한다.
    새 모델이 갖춰야 할 것:
      1) forward CLOSED ledger CSV (realized 수익률/하방 컬럼 포함)
      2) slots (어느 시점에 발송 가능한가: preopen/open)
      3) metric_source (ledger 의 어느 컬럼이 net 실현 수익률 / 하방 / hit 인가)
      4) predict_ref (ops dispatcher 가 호출할 import 경로 — 시그니처는 docstring 참조)

★ LEAK 무관 (이 파일은 메타데이터): 실제 leak 방어는 각 모델의 predict 코드
  (signals.recommend 등) 안에서 보장된다. 여기서는 '어디서 실현 결과를 읽나'만 정의.

self-contained: gan_t/xsec_alpha/fin import 0 (prelude 내부 경로 문자열만 보관).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class MetricSource:
    """ledger CSV 에서 하방-우선 점수 계산에 필요한 컬럼 매핑.

    셀렉터는 status_col == closed_value 인 행만(이미 실현된 과거) 평가한다.
    realized_pct_col 은 **net** (거래비용 차감 후) 실현 수익률 % 여야 한다.
    """
    status_col: str            # 청산 여부 컬럼 (예: "status")
    closed_value: str          # CLOSED 를 뜻하는 값 (예: "closed")
    date_col: str              # 진입/추천일 (rolling 윈도 정렬용, 예: "date")
    realized_pct_col: str      # net 실현 수익률 % (왕복비용 차감 후). 하방/net mean 의 기반.
    hit_col: str | None = None  # (옵션) 적중 0/1 컬럼 (3순위 tie-break). 없으면 None.
    # realized_pct_col 이 비용 차감 전(gross)일 때만 채운다. None=이미 net.
    cost_already_deducted: bool = True


@dataclass(frozen=True)
class ModelSpec:
    """champion/challenger 후보 모델 1개의 메타데이터.

    Attributes
    ----------
    id : str
        모델 고유 id (champion_state.json 키).
    name : str
        사람이 읽는 이름.
    ledger_path : str
        forward CLOSED ledger CSV 경로 (prelude 루트 기준 상대).
    slots : list[str]
        이 모델이 발송 가능한 시점 ("preopen" 08:50 / "open" 09:05).
    metric : MetricSource
        ledger 컬럼 매핑 (하방-우선 점수 계산용).
    predict_ref : str
        ops dispatcher 가 호출할 import 경로. 시그니처는 PREDICT_INTERFACES 참조.
    is_backtest_fallback : bool
        forward 게이트(n>=30) 미달 시 기본 챔피언으로 쓸 백테스트-최선 모델인가.
        게이트 통과 모델이 하나도 없으면 이 모델이 항상 발송(SHADOW)된다.
    challenger_only : bool
        True 면 데이터가 충분해도 챔피언 승격 금지(챌린저로만 관찰). 데이터 적은 신모델용.
    hypothesis : str
        "왜 이게 오를/안전한 신호인가" 한 줄 가설 (quant-evaluator 검증 대상).
    notes : str
        운영 메모.
    """
    id: str
    name: str
    ledger_path: str
    slots: list[str]
    metric: MetricSource
    predict_ref: str
    is_backtest_fallback: bool = False
    challenger_only: bool = False
    hypothesis: str = ""
    notes: str = ""

    def abs_ledger_path(self) -> Path:
        p = Path(self.ledger_path)
        return p if p.is_absolute() else (_ROOT / p)


# ==========================================================================
# v1 등록 — R1 (open + preopen) + distribution (open) + preopen(challenger).
#   숫자/경로는 placeholder (CLAUDE.md §2.5): ledger 가 바뀌면 여기만 고친다.
# ==========================================================================
MODELS: list[ModelSpec] = [
    # --- R1 downside-first 레이더 (post-open) ---------------------------------
    ModelSpec(
        id="recommend_r1_open",
        name="R1 risk-reward (downside-first), post-open 09:05",
        ledger_path="output/shadow_ledger_recommend.csv",
        slots=["open"],
        metric=MetricSource(
            status_col="status", closed_value="closed", date_col="date",
            realized_pct_col="realized_pct",   # close_recommend_ledger 가 net(0.15% 차감) 저장
            hit_col="pump20_hit", cost_already_deducted=True),
        predict_ref="signals.recommend:score_candidates",
        is_backtest_fallback=True,   # forward 표본 부족 시 기본 챔피언 (백테스트-최선)
        challenger_only=False,
        hypothesis="D-1 변동성/유동성/모멘텀 cross-section 상위 + de-corr 하락 head 로 "
                   "P(≥10%)/P(≤-5%) 비율(rr)이 높은 코인은 당일 하방이 낮고 상방이 유지된다.",
        notes="현 메인. forward closed 누적 전이라 fallback 챔피언으로 등록. "
              "open slot 은 entry_open(09:00) 확정 후 진입.",
    ),
    # --- R2 downside-penalized 챌린저 (post-open, 별도 shadow ledger) ----------
    #   R1 과 동일 head 확률의 재정렬(p_up10 - λ*p_dn5). r2_challenger_compare_v1
    #   OOS(765일, 15m SL/TP/EOD net): R1 %SL 0.456/no-SL deep 0.135 →
    #   R2 λ=1 %SL 0.272/deep 0.058 (하방 반감, net_mean 유지). 채택은 evaluator 판정 +
    #   forward 누적 후. 지금은 challenger_only(승격 금지) — 자기 ledger record-only.
    ModelSpec(
        id="recommend_r2_open",
        name="R2 risk-reward downside-penalized (challenger), post-open 09:05",
        ledger_path="output/shadow_ledger_recommend_r2.csv",
        slots=["open"],
        metric=MetricSource(
            status_col="status", closed_value="closed", date_col="date",
            realized_pct_col="realized_pct",   # close_recommend_ledger 가 net(0.15% 차감) 저장
            hit_col="pump20_hit", cost_already_deducted=True),
        predict_ref="signals.recommend:score_candidates",   # ranking='R2' 로 호출
        is_backtest_fallback=False,
        challenger_only=True,    # forward 누적 + evaluator 판정 전까지 챔피언 승격 금지
        hypothesis="하방을 선형 패널티(p_up10 - λ*p_dn5)로 더 강하게 누르면 stop-out/"
                   "deep-dump 픽이 상단에서 배제돼 R1(비율) 대비 하방위험이 낮아진다 "
                   "(사용자: 하락최소화 > 상승). OOS 15m net 경로에서 %SL·deep-loss 반감.",
        notes="R1 과 동일 de-corr head 의 결정론적 재정렬 — 새 leak 0. R2_LAMBDA=1.0 "
              "(recommend.py 상수, placeholder). champion_selector 가 30거래일 후 R1 vs "
              "R2 를 동일 스키마 ledger 로 비교. daily_run 배선/활성화는 ops 단계(evaluator "
              "ADOPT 후). 기록: scripts/recommend_today.py --ranking R2.",
    ),
    # --- A1 sustainability-filter 챌린저 (post-open, 별도 shadow ledger) -------
    #   R1 진입(top-3) 위에 dump head(D-1 feature 로 day-D 펌프-후-덤프 확률 학습)로
    #   dump-prone 픽을 강등→다음 R1 후보로 교체. ch_sustainability_v1 OOS(765일, 15m
    #   SL/TP/EOD net): deep-loss(SL 끈 본질 하방) 0.135→0.060 절반(bootstrap 유의),
    #   %SL 0.456→0.326. 단 net Δ 비유의·양쪽 net-negative, prec20 0.037→0.012(상방도
    #   같이 깎임). evaluator SHADOW 판정 → challenger_only(승격 금지) record-only.
    ModelSpec(
        id="recommend_r1_sustain_open",
        name="A1 sustainability-filter (challenger), post-open 09:05",
        ledger_path="output/shadow_ledger_recommend_sustain.csv",
        slots=["open"],
        metric=MetricSource(
            status_col="status", closed_value="closed", date_col="date",
            realized_pct_col="realized_pct",   # close_recommend_ledger 가 net(0.15% 차감) 저장
            hit_col="pump20_hit", cost_already_deducted=True),
        predict_ref="signals.recommend:score_candidates",   # ranking='A1' 로 호출
        is_backtest_fallback=False,
        challenger_only=True,    # forward 누적 + evaluator ADOPT 전까지 챔피언 승격 금지
        hypothesis="D-1 시점 dump head 로 '지속 펌프 vs 펌프-후-덤프'를 갈라 R1 top-3 중 "
                   "dump-prone 픽을 강등·교체하면, deep-loss·%SL 등 하방위험이 낮아진다 "
                   "(사용자: 하락최소화 > 상승). OOS 15m net 경로에서 deep-loss 반감.",
        notes="R1 과 동일 de-corr head 확률 위에 dump head(dump_B, cutoff q0.6) 하나 추가 — "
              "같은 panel/train cutoff, 새 leak 0. A1_DUMP_*/A1_CUTOFF_Q (recommend.py 상수, "
              "placeholder). champion_selector 가 30거래일 후 R1 vs A1 을 동일 스키마 ledger 로 "
              "비교. 기록: scripts/recommend_today.py --ranking A1 (record-only). "
              "trade-off 관측: 하방 개선의 대가로 prec20 도 깎임 → forward 로 재현 확인.",
    ),
    # --- PUMP hunter rule detector (post-open, 별도 shadow ledger) ------------
    #   pump_rule_discovery_v1 에서 채굴한 일봉 급등 rule 을 실제 daily detector 로
    #   배선한다. ML detector/promoted model 이 아니라 해석 가능한 rule-fire watchlist.
    #   forward 표본이 쌓일 때까지 challenger_only=True 로 record-only.
    ModelSpec(
        id="pump_hunter",
        name="PUMP hunter rule detector (challenger), post-open 09:05",
        ledger_path="output/shadow_ledger_pump_hunter.csv",
        slots=["open"],
        metric=MetricSource(
            status_col="status", closed_value="closed", date_col="date",
            realized_pct_col="realized_pct",
            hit_col="pump20_hit", cost_already_deducted=True),
        predict_ref="signals.pump_detector_v1:score_pump_candidates",
        is_backtest_fallback=False,
        challenger_only=True,
        hypothesis="D-1 roc_7d 횡단 rank 상위권(>0.85)과 ATR/과열 제한 rule 이 "
                   "day-D 장중 +20%/+15% 급등 후보를 기존 R1/R2보다 넓게 포착한다. "
                   "룰은 pump_rule_discovery_v1 의 leak-free feature_date=D-1 mining 결과.",
        notes="별도 shadow ledger 에 max 20 watchlist 를 기록. Telegram/ACTIVE 발송 없음. "
              "policy_competition 이 CLOSED forward rows 에서 pump20_recall/net/downside 를 "
              "기존 5개 모델과 자동 비교한다. 충분한 표본 + evaluator 판정 + 사용자 컨펌 "
              "전까지 승격 금지.",
    ),
    # --- PUMP hunter v2 — Binance volsurge 융합 (2026-06-11 사용자 컨펌 radar) --
    #   researcher→evaluator→15m 재검증 체인 통과한 hit 엣지 (최근 7개월 OOS
    #   hit 8.1% vs base 1.4%, lift 4.34x 5/5 fold). 자동 net 은 음수라
    #   champion 승격은 challenger_only 로 차단 — 단 사용자 명시 컨펌으로
    #   🎯 radar 텔레그램은 별도 발사 (메시지에 net 음수 정직 고지 포함).
    ModelSpec(
        id="pump_hunter_v2",
        name="PUMP hunter v2 (Upbit roc7 + Binance volsurge), post-open 09:05",
        ledger_path="output/shadow_ledger_pump_hunter_v2.csv",
        slots=["open"],
        metric=MetricSource(
            status_col="status", closed_value="closed", date_col="date",
            realized_pct_col="realized_pct",
            hit_col="pump20_hit", cost_already_deducted=True),
        predict_ref="signals.pump_detector_v2:score_pump_v2_candidates",
        is_backtest_fallback=False,
        challenger_only=True,
        hypothesis="Upbit roc_7d_rank>0.85 AND Binance D-1 거래량 surge>1.5 가 "
                   "pump20 hit 를 baseline 5.6%→8.1% (base 1.4%) 로 올린다 — "
                   "scripts/binance_leadlag_v1.py 검증 + evaluator 감사 + 15m 재계산.",
        notes="매일 max 5 watchlist + 🎯 텔레그램 radar (사용자 컨펌 2026-06-11). "
              "자동매매 룰 net 음수 — 메시지에 정직 고지, exit 은 사용자 판단. "
              "binance_d1.db daily refresh 필요 (daily_run_distribution [8/9]). "
              "champion 승격은 challenger_only 차단 유지.",
    ),
    # --- R1 pre-open 변형 (08:50 예고, 같은 스코어러 preopen slot) -------------
    ModelSpec(
        id="recommend_r1_preopen",
        name="R1 risk-reward (downside-first), pre-open 08:50 예고",
        ledger_path="output/shadow_ledger_recommend_preopen.csv",
        slots=["preopen"],
        metric=MetricSource(
            status_col="status", closed_value="closed", date_col="date",
            realized_pct_col="realized_pct",
            hit_col="pump20_hit", cost_already_deducted=True),
        predict_ref="signals.recommend:score_candidates",   # slot='preopen' 로 호출
        is_backtest_fallback=False,
        challenger_only=True,    # pre-open shadow 표본 적음 → 챌린저로만 관찰
        hypothesis="post-open R1 과 동일 가설. 단 08:50 예고는 진입가 미확정(None)이라 "
                   "09:00 갭/슬리피지 노출이 추가된다 — forward 로 그 비용을 측정한다.",
        notes="ledger 는 preopen 전용 shadow (open slot 과 분리 평가). 데이터 적으면 "
              "셀렉터가 게이트(n>=30) 미달로 챔피언 승격 안 함.",
    ),
    # --- distribution 7-head (post-open, paper_ledger) ------------------------
    ModelSpec(
        id="distribution_engine",
        name="distribution 7-head + setup library, post-open",
        ledger_path="output/paper_ledger.csv",
        slots=["open"],
        metric=MetricSource(
            status_col="status", closed_value="closed", date_col="date",
            # paper_ledger 는 day-D+1 open→경로 실현을 next_*_return_pct 로 저장.
            # net mean / 하방의 기반 = next_close_return_pct (실현 종가 수익률, %).
            # ★ gross → 셀렉터가 왕복 0.15% 차감 (cost_already_deducted=False).
            realized_pct_col="next_close_return_pct",
            hit_col="hit_h6", cost_already_deducted=False),
        predict_ref="signals.distribution_engine:DistributionEngine.score_panel",
        is_backtest_fallback=False,
        challenger_only=True,    # ★ 발송 절대 불가로 잠금 (2026-06-01): 사용자 명시 "수정한
                                 #   걸로만 와야지 기존은 왜 와" = R1 만 발송. + predict_ref 가
                                 #   score_candidates 와 다른 시그니처(score_panel)라 챔피언
                                 #   되면 발송 깨짐 + observed -45% vs replay +71.7% 갭(신뢰불가).
                                 #   → 영구 challenger(forward 비교·대시보드용만). 승급은 사용자
                                 #   명시 + 발송 어댑터 작성 후에만.
        hypothesis="T×C×DD×H 라벨 공간 7-head 확률 + setup library 교집합이 분포적으로 "
                   "상승/안정 구간을 분리한다. ATR(변동성)이 universal first-split.",
        notes="paper_ledger 실현은 next_close_return_pct(gross %) → 셀렉터가 비용 차감. "
              "open slot 만. observed -45% vs replay +71.7% 갭 있어 policy_gate 관찰 중. "
              "challenger_only=True 로 발송 차단(R1 단일 발송 보장).",
    ),
]


# ==========================================================================
# predict 인터페이스 명세 (ops-steward send dispatcher 가 챔피언 호출용).
#   ★ 이 파일은 호출하지 않는다 — dispatcher 가 predict_ref 를 import 해서 호출할 때
#     기대 시그니처를 문서화한 것. 실제 leak 방어는 각 predict 함수 안에 있다.
# ==========================================================================
PREDICT_INTERFACES = {
    "signals.recommend:score_candidates": (
        "score_candidates(asof_date, limit_markets=None, slot='auto') -> dict\n"
        "  slot='open'    : 09:05 post-open, entry_open=실제 09:00 open.\n"
        "  slot='preopen' : 08:50 pre-open, entry_open=None (D-1 feature 로 다음거래일 예측).\n"
        "  slot='auto'    : asof 당일 봉 존재 여부로 자동 분기.\n"
        "  반환 dict: {'asof','slot','feature_date','top3':[{coin,rr_ratio,p_up10,p_dn5,...}]}"
    ),
    "signals.distribution_engine:DistributionEngine.score_panel": (
        "DistributionEngine.load(ckpt_dir).score_panel(panel: DataFrame) -> DataFrame\n"
        "  panel 은 build_*_features 산출(leak-free D-1 feature). 반환: p_<head> 컬럼 부착.\n"
        "  open slot 전용 (paper_ledger 파이프)."
    ),
    "signals.pump_detector_v1:score_pump_candidates": (
        "score_pump_candidates(asof_date, top_universe=100, max_candidates=20) -> dict\n"
        "  open slot 전용 rule detector. D-1 roc_7d_rank/atr/log_return rule-fire 후보를 "
        "별도 shadow ledger 에 record-only 로 남긴다."
    ),
}


def get_model(model_id: str) -> ModelSpec:
    for m in MODELS:
        if m.id == model_id:
            return m
    raise KeyError(f"unknown model id: {model_id} (registered: {[m.id for m in MODELS]})")


def models_for_slot(slot: str) -> list[ModelSpec]:
    """해당 slot(preopen/open) 에서 발송 가능한 모델 목록."""
    return [m for m in MODELS if slot in m.slots]


def all_slots() -> list[str]:
    seen = []
    for m in MODELS:
        for s in m.slots:
            if s not in seen:
                seen.append(s)
    return seen


def fallback_model(slot: str) -> ModelSpec | None:
    """게이트 통과 모델이 없을 때 쓸 기본 챔피언(백테스트-최선) for slot.
    여러 개면 첫 번째. preopen 에 fallback 이 없으면 open 의 fallback 을 빌려 쓴다
    (R1 은 같은 스코어러라 slot 만 다름)."""
    cands = [m for m in models_for_slot(slot) if m.is_backtest_fallback]
    if cands:
        return cands[0]
    # preopen 에 명시 fallback 이 없으면 R1 open fallback 을 preopen slot 으로 재사용.
    glob = [m for m in MODELS if m.is_backtest_fallback]
    return glob[0] if glob else None


if __name__ == "__main__":
    print("registered models:")
    for m in MODELS:
        print(f"  {m.id:26s} slots={m.slots} fallback={m.is_backtest_fallback} "
              f"challenger_only={m.challenger_only} ledger={m.ledger_path}")
    print("\nslots:", all_slots())
    for s in all_slots():
        fb = fallback_model(s)
        print(f"  slot={s}: candidates={[m.id for m in models_for_slot(s)]} "
              f"fallback={fb.id if fb else None}")
