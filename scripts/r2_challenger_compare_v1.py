"""R2 challenger vs R1 champion — downside-first OOS net comparison (증거 생성).

목적 (signal-researcher 트랙 B):
  현 챔피언 R1(risk-reward 비율 정렬)에 도전하는 첫 챌린저 R2(downside-penalized
  선형 패널티)를 **동일 OOS 기간·동일 유니버스·동일 head 확률·동일 청산경로**에서
  net(왕복 0.15% 차감) 으로 비교한다. 사용자 선호: 하락위험 최소화 > 상승, ~-5%
  손실 수용 → 1순위 지표 = **P(realized net <= -5%) [deep-loss freq] ↓**.

  R1 = p_up10 / max(p_dn5, eps)            (recommend.py 현행 정렬키, byte-identical)
  R2 = p_up10 - λ * p_dn5                   (선형 패널티, λ grid sweep)

★ 왜 이 스크립트가 기존 downside_head_riskreward_v1.py 와 다른가 (핵심):
  - 기존 비교는 realized path = **EOD 종가 수익률**(eod_ret - cost) 이었다.
  - 이 스크립트는 realized path = **recommend.py/close_recommend_ledger.py 와 동일한
    15m SL -3% / TP +5% / EOD 경로청산** (simulate_path) net of 0.15%.
    → 라이브 R1 ledger 가 실제로 겪는 손익경로 그대로 비교 (EDA-hit ≠ ledger 함정 회피).
  - 정렬키만 다르고 head 확률·유니버스·OOS fold 는 R1 과 완전 공유 → 랭킹 효과 isolate.

★ LEAK 방어 (same-day leak 2번 전적 — 양보 X):
  - head feature = build_market_features 의 .shift(1) (D-1 까지), LEAK_COLS/next_* 제외
    (downside_head_riskreward_v1._feats). 라벨 = day-D high/low (미래) — train 에서만 fit.
  - per-fold OOF bucket calibration (train-only). test fold 는 적용만.
  - purged walk-forward + embargo (downside_head_riskreward_v1.walk_forward_heads).
  - 청산 15m 경로는 진입일 D 의 in-trade outcome → leak 아님 (진입 결정은 D-1 까지).
  - R2 는 R1 과 동일 head 확률의 결정론적 재정렬 → 새 leak 유입 구조적 0.

self-contained: prelude 내부 모듈만 import (gan_t/xsec_alpha/fin import 0).

사용:
    python scripts/r2_challenger_compare_v1.py
    python scripts/r2_challenger_compare_v1.py --limit-markets 80   # 개발
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# R1 과 동일한 leak-free head 파이프 재사용 (recommend.py 가 import 하는 바로 그 빌더).
from scripts.downside_head_riskreward_v1 import (  # noqa: E402
    build_panel,
    add_cross_sectional,
    attach_btc_regime,
    walk_forward_heads,
    _feats,
    UP_THRESH,
    DN_THRESH,
)
# 라이브 ledger 와 동일한 15m SL/TP/EOD 경로청산 (recommend.py 청산경로).
from scripts.recommender_downside_exit_v1 import simulate_path  # noqa: E402

D1_DB = str(_ROOT / "data" / "upbit_d1.db")
M15_DB = str(_ROOT / "data" / "upbit_15m.db")
OUT = _ROOT / "output"

ROUND_TRIP_COST = 0.0015     # 왕복 0.15% (수수료 0.1% + 슬리피지 0.05%)
HARD_SL = 0.03               # -3% 손절 (recommend.py / close_recommend_ledger.py 와 동일)
TP = 0.05                    # +5% 익절
DEEP_LOSS = -0.05            # 사용자 기준: net 이보다 깊은 손실 = '나쁜 사건' (1순위)

# 정렬키 anchor (recommend.py 와 동일: p_up10 분자, p_dn5 분모/패널티).
UP_ANCHOR = "lab_up_10"      # P(>=+10%)
DN_ANCHOR = "lab_dn_05"      # P(<=-5%)
RR_EPS = 1e-3                # R1 downside floor (recommend.py RR_EPS 와 동일)
LAMBDA_GRID = [0.5, 1.0, 2.0, 3.0]   # R2 penalized (랜딩 λ=1 포함 — placeholder, 데이터가 결정)

TOP_K = 3                    # 매일 top-3 (recommend.py TOP_K)
# 라이브 R1 유니버스 = 정적 top100 (recommend.py UNIVERSE_TOP_N). 기존 WF script 의
# top100-OR-surge 가 아니라 라이브와 동일하게 top100 으로 제한해 충실 비교.
UNIVERSE_TOP_N = 100
N_FOLDS = 6
EMBARGO = 5

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("r2_challenger")


# ============================================================================
# 1. 15m 경로 로딩 + 진입일 D 경로청산 (라이브 ledger 와 동일)
# ============================================================================
def load_paths(pairs: pd.DataFrame) -> dict:
    """(market, date) → [09:00 D, 09:00 D+1) 의 15m 봉 (open,high,low,close) 리스트."""
    conn = sqlite3.connect(M15_DB)
    paths = {}
    for _, r in pairs.iterrows():
        m, dt = r["market"], pd.Timestamp(r["date"])
        start = dt.strftime("%Y-%m-%d 09:00:00")
        end = (dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d 09:00:00")
        rows = conn.execute(
            "SELECT open,high,low,close FROM candles WHERE market=? AND "
            "timestamp>=? AND timestamp<? ORDER BY timestamp", (m, start, end)
        ).fetchall()
        if rows:
            paths[(m, dt.date())] = rows
    conn.close()
    return paths


def realize_net(bars: list) -> tuple[float, str, float]:
    """라이브와 동일: SL -3% / TP +5% / EOD 경로청산 → net = gross - 왕복비용.

    추가로 no-SL EOD net 도 반환한다. 이유(★ 함정 회피):
      하드 SL -3% 가 손실을 -3.15% 로 floor 하므로 'P(net <= -5%)' 는 모든 정렬키에서
      구조적으로 ~0 → 정렬키를 구분 못 한다. 사용자의 하방-우선 선호를 실제로
      측정하려면 (a) %SL stop-out 빈도(라이브 경로의 '나쁜 사건')와 (b) SL 을 끄면
      그 픽이 얼마나 깊이 빠졌을지(no-SL EOD deep-loss) 둘 다 봐야 한다.
      (b) 는 정렬키의 '본질적 하방 품질'을 드러낸다(SL 에 가려지기 전).
    """
    gross, outcome = simulate_path(bars, HARD_SL, TP, None)
    eod_gross, _ = simulate_path(bars, None, None, None)   # no-SL/no-TP EOD
    if not np.isfinite(gross):
        return np.nan, "nodata", np.nan
    eod_net = (eod_gross - ROUND_TRIP_COST) if np.isfinite(eod_gross) else np.nan
    return gross - ROUND_TRIP_COST, outcome, eod_net


# ============================================================================
# 2. 하방-우선 net 지표 (라이브 ledger 손익경로 기준)
# ============================================================================
def net_metrics(trades: pd.DataFrame) -> dict:
    """trade(date, net, outcome) → 하방-우선 net 지표.
    1순위: deep_loss_freq = P(net <= -5%). 2순위: net_mean. 3순위: hit/precision."""
    d = trades.dropna(subset=["net"]).copy()
    n = len(d)
    if n == 0:
        return {}
    net = d["net"].values
    daily = d.groupby("date")["net"].mean().sort_index()
    eq = (1 + daily).cumprod()
    peak = eq.cummax()
    mdd = float(((eq - peak) / peak).min())
    cum = float(eq.iloc[-1] - 1.0)
    mu = float(net.mean())
    sd = float(daily.std())
    dstd = float(daily[daily < 0].std()) if (daily < 0).any() else np.nan
    k5 = max(1, int(np.ceil(0.05 * n)))
    cvar95 = float(np.sort(net)[:k5].mean())
    oc = d["outcome"]
    # no-SL EOD net (SL 끄면 그 픽이 실제로 얼마나 빠졌나 — 정렬키 본질적 하방 품질).
    eod = d["eod_net"].dropna().values if "eod_net" in d.columns else np.array([])
    return dict(
        n=int(n),
        n_days=int(d["date"].nunique()),
        # ★ 1순위 (라이브 경로의 '나쁜 사건'): -3% 하드 SL stop-out 빈도.
        #   하드 SL 가 손실을 floor 하므로 P(net<=-5%)는 ~0 → 구분 불가. %SL 가 실제 하방.
        pct_sl=float((oc == "sl").mean()),
        # ★ 1순위 보조: SL 끈 EOD 에서 net<=-5% 빈도 (정렬키 본질 하방, SL 가리기 전).
        deep_loss_freq_noSL=float((eod <= DEEP_LOSS).mean()) if len(eod) else np.nan,
        net_mean=mu,                                        # ★ 2순위
        hit=float((net > 0).mean()),                        # ★ 3순위 (net 양수 비율)
        precision_pump20=float(d["pump20_hit"].dropna().mean())
        if d["pump20_hit"].notna().any() else np.nan,       # ★ 3순위 (일봉 +20% 적중)
        # 라이브 경로 deep-loss (참고 — 하드 SL 로 ~0, 갭다운 시만 -5% 초과).
        deep_loss_freq=float((net <= DEEP_LOSS).mean()),
        net_median=float(np.median(net)),
        cvar95=cvar95,
        worst=float(net.min()),
        mdd=mdd,
        cum=cum,
        sharpe=float(mu / sd * np.sqrt(365)) if sd and sd > 0 else np.nan,
        sortino=float(mu / dstd * np.sqrt(365)) if dstd and dstd > 0 else np.nan,
        pct_tp=float((oc == "tp").mean()),
        pct_eod=float((oc == "eod").mean()),
    )


# ============================================================================
# 3. 정렬키별 top-K 픽 (동일 OOS 행에서, 정렬키만 교체)
# ============================================================================
def topk_picks(oos: pd.DataFrame, score_col: str, k: int) -> pd.DataFrame:
    """각 날 score_col 내림차순 top-k. tie-break 은 하방-우선(recommend.py 와 동일 정신):
    deep-dump 낮게(p_dn10 ↑작게), 그 다음 p_up10 ↑크게, exp_downside ↑(덜 깊은 하방)."""
    s = oos.sort_values(
        ["date", score_col, "p_lab_dn_10", "p_lab_up_10", "exp_downside"],
        ascending=[True, False, True, False, False])
    return s.groupby("date").head(k).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-markets", type=int, default=None)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    # --- 1) leak-free panel + head 라벨 (R1 과 동일 빌더) ---
    panel = build_panel(args.limit_markets)
    panel = add_cross_sectional(panel)
    panel = attach_btc_regime(panel)
    # 라이브 R1 유니버스 = 정적 top100 (recommend.py 와 동일).
    panel = panel[panel["f_qv_rank"] <= UNIVERSE_TOP_N].copy()
    panel["in_universe"] = True
    feats = _feats(panel)
    log.info("panel(top%d universe) rows=%d markets=%d dates=%d feats=%d",
             UNIVERSE_TOP_N, len(panel), panel["market"].nunique(),
             panel["date"].nunique(), len(feats))

    up_labels = [f"lab_{n}" for n in UP_THRESH]
    dn_labels = [f"lab_{n}" for n in DN_THRESH]
    label_cols = up_labels + dn_labels

    # --- 2) purged WF heads (R1 과 동일: per-fold OOF bucket calib, train-only) ---
    log.info("=== walk-forward heads (R1 과 동일 빌더, top100 universe) ===")
    oos = walk_forward_heads(panel, feats, label_cols, N_FOLDS, EMBARGO)
    if oos.empty:
        log.error("no OOS folds — abort"); sys.exit(1)
    log.info("OOS rows=%d dates %s..%s folds=%s",
             len(oos), oos["date"].min(), oos["date"].max(),
             sorted(oos["fold"].unique()))

    up = f"p_{UP_ANCHOR}"; dn = f"p_{DN_ANCHOR}"
    oos = oos.dropna(subset=[up, dn]).copy()

    # --- 3) 15m 경로 커버리지 (라이브 ledger 손익경로) ---
    conn = sqlite3.connect(M15_DB)
    m15_min = conn.execute("SELECT MIN(timestamp) FROM candles").fetchone()[0]
    conn.close()
    m15_start = pd.Timestamp(m15_min).date()
    oos = oos[oos["date"] >= m15_start].reset_index(drop=True)
    log.info("OOS in 15m window (>= %s): rows=%d dates=%d",
             m15_start, len(oos), oos["date"].nunique())

    # 모든 후보의 15m 경로 1회 로드 후 net/outcome 부착 (정렬키와 무관 → 픽마다 1회).
    pairs = oos[["market", "date"]].drop_duplicates()
    bars_map = load_paths(pairs)
    oos = oos[oos.apply(lambda r: (r["market"], r["date"]) in bars_map,
                        axis=1)].reset_index(drop=True)
    nets, outs, eods, p20 = [], [], [], []
    for _, r in oos.iterrows():
        net, oc, eod_net = realize_net(bars_map[(r["market"], r["date"])])
        nets.append(net); outs.append(oc); eods.append(eod_net)
        p20.append(1 if (r["up_high_ret"] >= 0.20) else 0)
    oos["net"] = nets
    oos["outcome"] = outs
    oos["eod_net"] = eods
    oos["pump20_hit"] = p20
    oos = oos.dropna(subset=["net"]).reset_index(drop=True)
    log.info("OOS with 15m net path: rows=%d dates=%d (outcome dist: %s)",
             len(oos), oos["date"].nunique(), oos["outcome"].value_counts().to_dict())

    # --- 4) 정렬키 산출 (R1 ratio + R2 λ grid) — 동일 head 확률 ---
    oos["R1"] = oos[up] / np.maximum(oos[dn], RR_EPS)
    for lam in LAMBDA_GRID:
        oos[f"R2_lam{lam}"] = oos[up] - lam * oos[dn]

    # --- 5) 정렬키별 top-3 net 지표 (동일 OOS 행, 동일 날짜) ---
    n_combos = 1 + len(LAMBDA_GRID)   # R1 + R2(λ grid). selection deflate 기록용.
    log.info("SELECTION: 정렬키 %d개 비교 (R1 + R2 λ%s), 동일 OOS·유니버스·청산경로",
             n_combos, LAMBDA_GRID)
    rows = []
    rank_keys = [("R1_ratio", "R1", "-")] + \
                [(f"R2_penalized", f"R2_lam{lam}", f"lam={lam}") for lam in LAMBDA_GRID]
    for policy, col, param in rank_keys:
        picks = topk_picks(oos, col, TOP_K)
        m = net_metrics(picks)
        m.update(policy=policy, ranking_key=col, param=param, K=TOP_K)
        rows.append(m)

    res = pd.DataFrame(rows)
    lead = ["policy", "ranking_key", "param", "K", "n", "n_days",
            "pct_sl", "deep_loss_freq_noSL", "net_mean", "hit", "precision_pump20",
            "deep_loss_freq", "net_median", "cvar95", "worst", "mdd", "cum",
            "sharpe", "sortino", "pct_tp", "pct_eod"]
    res = res[[c for c in lead if c in res.columns]]
    res_path = OUT / "r2_challenger_compare_v1.csv"
    res.to_csv(res_path, index=False)

    # --- 6) best λ 선정 (하방-우선): 라이브 경로 하방 = %SL stop-out + no-SL deep-loss
    #   를 합성한 하방랭크 ↑ 작은 것 우선, 동률이면 net_mean ↑.
    #   (단일 P(net<=-5%) 는 하드 SL 로 ~0 → 구분 불가. §2.3 결과우선: 실제 하방 사건 사용.)
    r1 = res[res["policy"] == "R1_ratio"].iloc[0]
    r2s = res[res["policy"] == "R2_penalized"].copy()
    r2_sorted = r2s.sort_values(
        ["pct_sl", "deep_loss_freq_noSL", "net_mean"],
        ascending=[True, True, False])
    best = r2_sorted.iloc[0]

    # 픽 dump (감사용) — best λ R2 와 R1 의 같은 날짜 side-by-side.
    best_lam = float(best["param"].split("=")[1])
    r1_picks = topk_picks(oos, "R1", TOP_K)
    r2_picks = topk_picks(oos, f"R2_lam{best_lam}", TOP_K)
    dump_cols = ["date", "market", "regime", "fold", up, dn, "p_lab_dn_10",
                 "exp_downside", "R1", f"R2_lam{best_lam}", "net", "outcome",
                 "pump20_hit", "up_high_ret", "down_low_ret"]
    dump_cols = [c for c in dump_cols if c in oos.columns]
    r1_picks_out = r1_picks[dump_cols].assign(policy="R1_ratio")
    r2_picks_out = r2_picks[dump_cols].assign(policy=f"R2_lam{best_lam}")
    pd.concat([r1_picks_out, r2_picks_out], ignore_index=True).to_csv(
        OUT / "r2_challenger_picks_v1.csv", index=False)

    cov = dict(
        oos_rows=int(len(oos)), oos_dates=int(oos["date"].nunique()),
        oos_window=[str(oos["date"].min()), str(oos["date"].max())],
        universe="static_top100", n_folds=N_FOLDS, embargo=EMBARGO,
        n_ranking_keys_compared=int(n_combos), lambda_grid=LAMBDA_GRID,
        exit_path="15m SL-3%/TP+5%/EOD net 0.15%",
        best_lambda=best_lam,
        regime_dist={k: int(v) for k, v in oos["regime"].value_counts().items()},
    )
    (OUT / "r2_challenger_coverage_v1.json").write_text(json.dumps(cov, indent=2))

    # ---- 콘솔 요약 ----
    log.info("\n===== R1 vs R2 (top-3, OOS net 0.15% 차감, 15m SL/TP/EOD 경로) =====")
    log.info("  정렬키       param     n    days  %%SL  deepNoSL net_mean hit  prec20  CVaR95  MaxDD    cum   Sharpe")
    for _, r in res.iterrows():
        log.info("  %-12s %-8s %4d %4d  %.3f  %.3f  %+.4f %.2f  %s  %+.4f %+.4f %+.3f  %s",
                 r["policy"], str(r["param"]), int(r["n"]), int(r["n_days"]),
                 r["pct_sl"], r["deep_loss_freq_noSL"], r["net_mean"], r["hit"],
                 f"{r['precision_pump20']:.3f}" if pd.notna(r["precision_pump20"]) else " nan ",
                 r["cvar95"], r["mdd"], r["cum"],
                 f"{r['sharpe']:+.2f}" if pd.notna(r["sharpe"]) else "nan")

    log.info("\n===== BEST λ (하방-우선: %%SL ↓ → no-SL deep-loss ↓ → net_mean ↑) =====")
    log.info("  R1 baseline : %%SL=%.3f deepNoSL=%.3f net_mean=%+.4f hit=%.2f cum=%+.3f",
             r1["pct_sl"], r1["deep_loss_freq_noSL"], r1["net_mean"], r1["hit"], r1["cum"])
    log.info("  BEST R2 %s : %%SL=%.3f deepNoSL=%.3f net_mean=%+.4f hit=%.2f cum=%+.3f",
             best["param"], best["pct_sl"], best["deep_loss_freq_noSL"],
             best["net_mean"], best["hit"], best["cum"])
    log.info("  Δ(R2-R1): %%SL %+.3f / deepNoSL %+.3f (음수=R2 하방 개선) / net_mean %+.4f",
             best["pct_sl"] - r1["pct_sl"],
             best["deep_loss_freq_noSL"] - r1["deep_loss_freq_noSL"],
             best["net_mean"] - r1["net_mean"])
    log.info("DONE. wrote %s + r2_challenger_picks_v1.csv + r2_challenger_coverage_v1.json",
             res_path)


if __name__ == "__main__":
    main()
