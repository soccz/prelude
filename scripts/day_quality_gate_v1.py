"""
day_quality_gate_v1.py — "어떤 날" gate 연구 (NOT "어떤 코인").

연구 질문: 펌프가 거의 없는 죽은 날을 D-1 까지의 정보만으로 식별해서, 그날 알림 0개로
가는 gate 가 시스템 픽의 net 을 개선하는가?

핵심 가설: 펌프는 시간적으로 클러스터링한다 — D-1 시장 전체의 pump 수가 D 의 pump 풍부도를
예측한다 (n_pump10(D-1) 높음 → D 도 pump-rich).

위생 (양보 불가):
  - 모든 gate FEATURE 는 strictly <= D-1 (daily label 시계열을 만든 뒤 .shift 로 lag feature 생성).
  - 유니버스 랭크도 D-1 quote_volume 기준 (기존 시스템 convention: top-100 by qv.shift(1)).
  - market-level LABEL (n_pump_D) 는 day-D high/open 으로 계산 = 의사결정(D 09:00) 시점엔 미래.
  - replay: 거래비용 차감된 net (ledger realized_pct = TP5/SL3 net; paper_ledger 는 path 로 합성).

self-contained: 다른 폴더 import 없음. 기존 convention 만 mirror (Explore 로 확인):
  - 유니버스: signals/recommend.py / pump_detector_v1.py — top-100 by D-1 quote_volume.
  - pump label: (high_D/open_D - 1) >= 0.10 / 0.20  (univariate_precursor_lift_v1.py).
  - leak 방어: 모든 raw 지표 .shift(1) (univariate_precursor_lift_v1.py:145).
  - 제외: KRW-USDT/USDC (collector_d1.py) — DB 에 이미 없음.

산출:
  output/day_quality_gate_v1_daily.csv      — 일별 label/feature 시계열
  output/day_quality_gate_v1_rules.csv      — 후보 룰 sweep (lift/precision/recall) + WF per-fold
  output/day_quality_gate_v1_replay.csv     — 룰 x ledger x exit 변형 replay net 비교
  output/day_quality_gate_v1_summary.json   — 최종 요약 (판정 입력)
"""
from __future__ import annotations
import sqlite3, json, sys, math
import numpy as np
import pandas as pd

DB = "data/upbit_d1.db"
OUT = "output/day_quality_gate_v1"

# --- placeholders (CLAUDE.md §2.5 — 데이터가 더 좋은 값 주면 바꿈) ---
UNIVERSE_TOP_N = 100        # 기존 시스템과 동일 (D-1 qv top-100)
MIN_UNIVERSE = 80           # 그날 active 코인 < 80 이면 day 제외 (universe 불안정)
PUMP10 = 0.10               # high/open-1 >= 0.10
PUMP20 = 0.20               # high/open-1 >= 0.20
BTC_RV_WIN = 20             # BTC 실현변동성 윈도우
QV_AVG_WIN = 20             # 시장 총 거래대금 20d 평균 대비
EMBARGO = 5                 # purged WF embargo (일)
N_FOLDS = 5
COST_RT = 0.0015            # 왕복 0.15% (수수료0.1+슬리피지0.05) — 합성 exit 에 차감


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# 1) 일별 시장 패널 — 모든 코인, full history
# ---------------------------------------------------------------------------
def load_all_candles() -> pd.DataFrame:
    c = sqlite3.connect(DB)
    df = pd.read_sql(
        "SELECT market, timestamp, open, high, low, close, quote_volume FROM candles",
        c, parse_dates=["timestamp"])
    c.close()
    df["date"] = df["timestamp"].dt.normalize()
    df = df.sort_values(["market", "date"]).reset_index(drop=True)
    # per-market 파생 (전부 raw — shift 는 daily 집계 후)
    g = df.groupby("market", group_keys=False)
    df["pump10"] = (df["high"] / df["open"] - 1.0 >= PUMP10).astype(float)
    df["pump20"] = (df["high"] / df["open"] - 1.0 >= PUMP20).astype(float)
    df["up_day"] = (df["close"] / df["open"] - 1.0 > 0).astype(float)
    df["ret_cc"] = g["close"].apply(lambda s: np.log(s / s.shift(1)))  # close-to-close log ret (D)
    # 유니버스 랭크용: D-1 거래대금 (shift within market) — leak 방어
    df["qv_lag1"] = g["quote_volume"].shift(1)
    return df


# ---------------------------------------------------------------------------
# 2) 일별 집계 — market-level LABEL (day D) + 일별 raw 보조량 (나중에 shift)
# ---------------------------------------------------------------------------
def build_daily(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d, grp in df.groupby("date"):
        active = grp[grp["qv_lag1"].notna()]  # D-1 거래대금 있어야 universe 에 들어옴
        n_active = len(active)
        if n_active < MIN_UNIVERSE:
            continue
        # 유니버스 = D-1 qv top-N (기존 convention)
        topn = min(UNIVERSE_TOP_N, n_active)
        uni = active.nlargest(topn, "qv_lag1")
        # ---- LABEL (day-D, 타겟; 의사결정 시점엔 미래) ----
        n_pump10 = float(uni["pump10"].sum())
        n_pump20 = float(uni["pump20"].sum())
        # ---- 일별 raw 보조량 (D 의 실현값 — feature 로 쓸 땐 반드시 shift) ----
        breadth = float(uni["up_day"].mean())               # 양봉 비율 (D)
        cs_ret_std = float(uni["ret_cc"].std())              # 횡단면 수익 분산 (D close-to-close)
        tot_qv = float(uni["qv_lag1"].sum())                 # 유니버스 총 거래대금 (D-1 기준이라 이미 lag)
        rows.append(dict(date=d, n_uni=topn, n_active=n_active,
                         n_pump10=n_pump10, n_pump20=n_pump20,
                         frac_pump10=n_pump10 / topn, frac_pump20=n_pump20 / topn,
                         breadth=breadth, cs_ret_std=cs_ret_std, tot_qv=tot_qv))
    daily = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # ---- BTC 보조 (D-1 까지) ----
    btc = df[df["market"] == "KRW-BTC"][["date", "close", "open", "high"]].copy()
    btc = btc.sort_values("date")
    btc["btc_ret"] = np.log(btc["close"] / btc["close"].shift(1))   # D close-to-close
    btc["btc_rv"] = btc["btc_ret"].rolling(BTC_RV_WIN).std()        # D 까지 RV
    btc["btc_ma20"] = btc["close"].rolling(20).mean()
    btc["btc_above_ma"] = (btc["close"] > btc["btc_ma20"]).astype(float)
    daily = daily.merge(btc[["date", "btc_ret", "btc_rv", "btc_above_ma"]], on="date", how="left")

    # =========================================================================
    # ★ leak 방어 핵심: feature 는 전부 lag (>= D-1).
    #   daily label 시계열(n_pump10 등)을 만든 뒤 shift → row D 는 D-1 까지만 본다.
    # =========================================================================
    daily["f_n_pump10_l1"] = daily["n_pump10"].shift(1)            # D-1 펌프 수
    daily["f_n_pump10_l2"] = daily["n_pump10"].shift(2)
    daily["f_n_pump10_3d"] = daily["n_pump10"].shift(1).rolling(3).mean()   # 클러스터링 (D-1..D-3)
    daily["f_n_pump10_5d"] = daily["n_pump10"].shift(1).rolling(5).mean()
    daily["f_n_pump20_l1"] = daily["n_pump20"].shift(1)
    daily["f_n_pump20_3d"] = daily["n_pump20"].shift(1).rolling(3).mean()
    daily["f_breadth_l1"] = daily["breadth"].shift(1)
    daily["f_cs_ret_std_l1"] = daily["cs_ret_std"].shift(1)
    # 시장 총 거래대금 vs 20d 평균 (tot_qv 는 D-1 거래대금 합 → 추가로 shift(1) 해서 완전 과거화)
    daily["f_qv_ratio"] = (daily["tot_qv"].shift(1) /
                           daily["tot_qv"].shift(1).rolling(QV_AVG_WIN).mean())
    # BTC: D-1 까지
    daily["f_btc_ret_l1"] = daily["btc_ret"].shift(1)
    daily["f_btc_rv_l1"] = daily["btc_rv"].shift(1)
    daily["f_btc_above_ma_l1"] = daily["btc_above_ma"].shift(1)
    # 펌프 수 autocorr proxy: l1 - 5d 평균 (모멘텀 of 펌프 풍부도)
    daily["f_pump_momo"] = daily["f_n_pump10_l1"] - daily["f_n_pump10_5d"]
    return daily


FEATURES = ["f_n_pump10_l1", "f_n_pump10_l2", "f_n_pump10_3d", "f_n_pump10_5d",
            "f_n_pump20_l1", "f_n_pump20_3d", "f_breadth_l1", "f_cs_ret_std_l1",
            "f_qv_ratio", "f_btc_ret_l1", "f_btc_rv_l1", "f_btc_above_ma_l1",
            "f_pump_momo"]


# ---------------------------------------------------------------------------
# 3) pump-rich day 정의 + 단순 룰 sweep
# ---------------------------------------------------------------------------
def define_richness(daily: pd.DataFrame):
    """pump-rich day = n_pump10_D >= K. K 는 분포 보고 결정 (placeholder: median)."""
    n = daily["n_pump10"].dropna()
    desc = {q: float(n.quantile(q)) for q in [0.1, 0.25, 0.5, 0.75, 0.9]}
    K = max(1, int(round(n.median())))   # 최소 1 — "펌프 최소 1개라도 있는 날"
    log(f"[richness] n_pump10 quantiles: {desc}  -> K(median, >=1) = {K}")
    log(f"[richness] base rate (n_pump10 >= {K}): {float((n >= K).mean()):.3f}")
    log(f"[richness] dead-day rate (n_pump10 == 0): {float((n == 0).mean()):.3f}")
    return K, desc


def rule_metrics(d: pd.DataFrame, gate_on: pd.Series, K: int) -> dict:
    """gate_on(bool) 이 pump-rich(n_pump10>=K) 를 얼마나 잘 맞히나."""
    valid = d["n_pump10"].notna() & gate_on.notna()
    y = (d.loc[valid, "n_pump10"] >= K).astype(int)
    g = gate_on.loc[valid].astype(int)
    base = float(y.mean())
    on_rate = float(g.mean())
    if g.sum() == 0:
        return dict(base=base, on_rate=on_rate, prec_on=np.nan, recall=np.nan,
                    lift=np.nan, dead_off=np.nan, n=int(valid.sum()))
    prec_on = float(y[g == 1].mean())                 # gate ON 일 때 pump-rich 비율
    recall = float(g[y == 1].mean())                  # pump-rich 날 중 gate ON 비율
    lift = prec_on / base if base > 0 else np.nan
    # gate OFF 날의 dead-day(0 펌프) 비율 — OFF 가 진짜 죽은 날 잡는지
    if (g == 0).sum() > 0:
        dead_off = float((d.loc[valid].loc[g == 0, "n_pump10"] == 0).mean())
    else:
        dead_off = np.nan
    return dict(base=base, on_rate=on_rate, prec_on=prec_on, recall=recall,
                lift=lift, dead_off=dead_off, n=int(valid.sum()))


def sweep_rules(daily: pd.DataFrame, K: int):
    """단순 1~2 피처 임계 룰 우선. 시도 조합 수 기록."""
    d = daily.dropna(subset=["n_pump10"]).copy()
    results = []
    n_tried = 0
    # (A) n_pump10(D-1) >= tau  — 가장 단순/해석가능 (핵심 가설)
    for tau in range(0, 8):
        n_tried += 1
        gate = d["f_n_pump10_l1"] >= tau
        m = rule_metrics(d, gate, K); m.update(rule=f"n_pump10_l1>={tau}", family="A_l1", tau=tau)
        results.append(m)
    # (B) 3d 평균 클러스터링
    for tau in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        n_tried += 1
        gate = d["f_n_pump10_3d"] >= tau
        m = rule_metrics(d, gate, K); m.update(rule=f"n_pump10_3d>={tau}", family="B_3d", tau=tau)
        results.append(m)
    # (C) breadth (D-1 양봉 비율)
    for tau in [0.3, 0.4, 0.5, 0.6]:
        n_tried += 1
        gate = d["f_breadth_l1"] >= tau
        m = rule_metrics(d, gate, K); m.update(rule=f"breadth_l1>={tau}", family="C_breadth", tau=tau)
        results.append(m)
    # (D) 2-피처 결합 (l1 AND breadth) — 약간 더 복잡 (조합 수 기록)
    for t1 in [1, 2, 3]:
        for t2 in [0.4, 0.5]:
            n_tried += 1
            gate = (d["f_n_pump10_l1"] >= t1) & (d["f_breadth_l1"] >= t2)
            m = rule_metrics(d, gate, K)
            m.update(rule=f"l1>={t1} & breadth>={t2}", family="D_combo", tau=t1)
            results.append(m)
    # (E) BTC RV 낮을 때 (조용한 날 silence 가설 반대 점검)
    for tau in [0.02, 0.03, 0.04]:
        n_tried += 1
        gate = d["f_btc_rv_l1"] >= tau  # RV 높을 때만 ON
        m = rule_metrics(d, gate, K); m.update(rule=f"btc_rv_l1>={tau}", family="E_btcrv", tau=tau)
        results.append(m)
    res = pd.DataFrame(results)
    log(f"[sweep] tried {n_tried} rule combos (selection-bias note for evaluator)")
    return res, n_tried


# ---------------------------------------------------------------------------
# 4) Purged Walk-Forward — per-fold lift 안정성
# ---------------------------------------------------------------------------
def purged_wf_folds(dates: pd.Series, n_folds: int, embargo: int):
    """시간순 n_folds, 각 fold 는 test 블록. train 은 test 앞쪽(embargo 갭)."""
    uniq = np.sort(dates.unique())
    chunks = np.array_split(uniq, n_folds)
    folds = []
    for i in range(n_folds):
        test_dates = set(pd.to_datetime(chunks[i]))
        test_start = pd.to_datetime(chunks[i][0])
        emb = test_start - pd.Timedelta(days=embargo)
        train_dates = set(pd.to_datetime([dd for dd in uniq if pd.to_datetime(dd) < emb]))
        folds.append((train_dates, test_dates))
    return folds


def wf_rule_stability(daily: pd.DataFrame, K: int, rule_family="A_l1"):
    """베스트 룰류를 fold 마다 OOF threshold 선택 → test lift. 안정성 보고."""
    d = daily.dropna(subset=["n_pump10", "f_n_pump10_l1"]).copy()
    folds = purged_wf_folds(d["date"], N_FOLDS, EMBARGO)
    rows = []
    for fi, (tr, te) in enumerate(folds):
        dtr = d[d["date"].isin(tr)]
        dte = d[d["date"].isin(te)]
        if len(dtr) < 30 or len(dte) < 10:
            rows.append(dict(fold=fi, n_train=len(dtr), n_test=len(dte), skip=True))
            continue
        # train 에서 lift 최대화하는 tau (단순 그리드) — OOF 선택
        best_tau, best_lift = None, -1
        ytr = (dtr["n_pump10"] >= K).astype(int)
        base_tr = ytr.mean()
        for tau in range(0, 7):
            g = (dtr["f_n_pump10_l1"] >= tau).astype(int)
            if g.sum() < 5:
                continue
            prec = ytr[g == 1].mean()
            lift = prec / base_tr if base_tr > 0 else 0
            # on_rate 가 너무 높으면(>0.95) 거의 항상 ON → gate 의미없음, 패널티
            on_rate = g.mean()
            if on_rate > 0.97:
                continue
            if lift > best_lift:
                best_lift, best_tau = lift, tau
        if best_tau is None:
            rows.append(dict(fold=fi, n_train=len(dtr), n_test=len(dte), skip=True))
            continue
        # test 에 적용
        mte = rule_metrics(dte, dte["f_n_pump10_l1"] >= best_tau, K)
        rows.append(dict(fold=fi, n_train=len(dtr), n_test=len(dte),
                         tau=best_tau, train_lift=round(best_lift, 3),
                         test_base=round(mte["base"], 3), test_prec_on=round(mte["prec_on"], 3),
                         test_lift=round(mte["lift"], 3) if not np.isnan(mte["lift"]) else None,
                         test_on_rate=round(mte["on_rate"], 3),
                         test_dead_off=round(mte["dead_off"], 3) if not np.isnan(mte["dead_off"]) else None,
                         skip=False))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5) ★ REPLAY — gate 를 실제 ledger 픽에 소급
# ---------------------------------------------------------------------------
def synth_tp_sl(row, tp, sl):
    """path 극값(next_max/min_return_pct)으로 TP/SL net 합성. TP 우선 보수 가정 X —
    intraday high/low 동시 도달 시 어느쪽 먼저인지 모름 → '비관'(SL 먼저) 가정으로 보수.
    비용 COST_RT 왕복 차감."""
    mx = row.get("next_max_return_pct", np.nan)
    mn = row.get("next_min_return_pct", np.nan)
    cl = row.get("next_close_return_pct", np.nan)
    if pd.isna(mx) or pd.isna(mn):
        return np.nan
    tp_pct, sl_pct = tp * 100, -sl * 100
    hit_tp = mx >= tp_pct
    hit_sl = mn <= sl_pct
    if hit_sl:           # 보수: SL 먼저 가정 (downside-first 와 일치)
        gross = sl_pct
    elif hit_tp:
        gross = tp_pct
    else:
        gross = cl if not pd.isna(cl) else 0.0
    return gross - COST_RT * 100  # net (%)


def load_ledger(path, gate_dates_on: set, label: str):
    """ledger 의 closed 행에 gate 결정(date in gate_dates_on) 부착. net 컬럼 정리."""
    try:
        df = pd.read_csv(path)
    except Exception as e:
        log(f"[replay] {label}: load fail {e}")
        return None
    if "status" in df.columns:
        df = df[df["status"] == "closed"].copy()
    if len(df) == 0:
        return None
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["gate_on"] = df["date"].isin(gate_dates_on)
    # net 컬럼: realized_pct 우선, 없으면 path 로 TP5/SL3 합성
    if "realized_pct" in df.columns:
        df["net_main"] = df["realized_pct"]
    else:
        df["net_main"] = df.apply(lambda r: synth_tp_sl(r, 0.05, 0.03), axis=1)
    # exit 변형 (있으면 그대로, paper_ledger 는 합성)
    if "exit_tp10_nosl_pct" in df.columns:
        df["net_tp10_nosl"] = df["exit_tp10_nosl_pct"]
        df["net_tp5_nosl"] = df["exit_tp5_nosl_pct"]
        df["net_eod"] = df["exit_eod_pct"]
    else:
        df["net_tp10_nosl"] = df.apply(lambda r: synth_tp_sl(r, 0.10, 0.99), axis=1)  # noSL≈sl 99%
        df["net_tp5_nosl"] = df.apply(lambda r: synth_tp_sl(r, 0.05, 0.99), axis=1)
        df["net_eod"] = (df["next_close_return_pct"] - COST_RT * 100) if "next_close_return_pct" in df.columns else np.nan
    df["ledger"] = label
    return df


def replay_table(ledgers: list[pd.DataFrame]):
    """ledger x exit변형 별 gate-ON vs ALL vs OFF net 비교."""
    rows = []
    netcols = [("realized/TP5SL3", "net_main"), ("TP10_noSL", "net_tp10_nosl"),
               ("TP5_noSL", "net_tp5_nosl"), ("EOD", "net_eod")]
    for df in ledgers:
        if df is None:
            continue
        lab = df["ledger"].iloc[0]
        n_dates = df["date"].nunique()
        n_on_dates = df.loc[df["gate_on"], "date"].nunique()
        silence_rate = 1 - n_on_dates / n_dates if n_dates else np.nan
        for cname, col in netcols:
            if col not in df.columns or df[col].isna().all():
                continue
            allp = df[col].dropna()
            onp = df.loc[df["gate_on"], col].dropna()
            offp = df.loc[~df["gate_on"], col].dropna()
            rows.append(dict(
                ledger=lab, exit=cname,
                n_picks_all=len(allp), n_picks_on=len(onp), n_picks_off=len(offp),
                n_dates=n_dates, n_on_dates=n_on_dates, silence_rate=round(silence_rate, 3),
                net_mean_all=round(allp.mean(), 3),
                net_mean_on=round(onp.mean(), 3) if len(onp) else None,
                net_mean_off=round(offp.mean(), 3) if len(offp) else None,
                net_sum_all=round(allp.sum(), 2),
                net_sum_on=round(onp.sum(), 2) if len(onp) else None,
                improve_mean=round(onp.mean() - allp.mean(), 3) if len(onp) else None,
            ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    log("=== day_quality_gate_v1 ===")
    df = load_all_candles()
    log(f"[load] candles rows={len(df)} markets={df['market'].nunique()} "
        f"dates={df['date'].nunique()} span {df['date'].min().date()}..{df['date'].max().date()}")

    daily = build_daily(df)
    log(f"[daily] {len(daily)} days kept (>= {MIN_UNIVERSE} active) "
        f"{daily['date'].min().date()}..{daily['date'].max().date()}")
    daily.to_csv(f"{OUT}_daily.csv", index=False)

    K, desc = define_richness(daily)

    rules, n_tried = sweep_rules(daily, K)
    # 보기 좋게 정렬: lift 내림 (단 on_rate 0.3~0.95 합리 구간 우선)
    rules_show = rules.copy()
    rules_show["sane"] = (rules_show["on_rate"].between(0.3, 0.95)).astype(int)
    rules_show = rules_show.sort_values(["sane", "lift"], ascending=[False, False])
    rules.to_csv(f"{OUT}_rules.csv", index=False)
    log("[sweep] top rules (sane on_rate 0.3-0.95, by lift):")
    cols = ["rule", "family", "base", "on_rate", "prec_on", "recall", "lift", "dead_off", "n"]
    log(rules_show[rules_show["sane"] == 1][cols].head(8).to_string(index=False))

    wf = wf_rule_stability(daily, K)
    wf.to_csv(f"{OUT}_wf.csv", index=False)
    log("[WF] per-fold OOF-threshold stability (rule family A: n_pump10_l1):")
    log(wf.to_string(index=False))

    # ---- gate 선택: 단순 핵심 룰. replay 용 ON-date 집합 2가지 ----
    # (a) full-sample threshold (해석: 전 기간 데이터로 고른 단순 임계)
    # (b) pre-2025-11 threshold (replay 기간과 분리 — leak-clean)
    d_all = daily.dropna(subset=["f_n_pump10_l1"]).copy()
    REPLAY_CUT = pd.Timestamp("2025-11-01")

    # 핵심 룰: n_pump10_l1 >= tau* — tau* 는 (b) pre-cut 데이터에서 lift 최대 (sane on_rate)
    pre = d_all[d_all["date"] < REPLAY_CUT]
    ypre = (pre["n_pump10"] >= K).astype(int); basepre = ypre.mean()
    best = (-1, None)
    for tau in range(0, 7):
        g = (pre["f_n_pump10_l1"] >= tau)
        if not (0.3 <= g.mean() <= 0.95):
            continue
        prec = ypre[g.values == 1].mean()
        lift = prec / basepre if basepre > 0 else 0
        if lift > best[0]:
            best = (lift, tau)
    tau_star = best[1] if best[1] is not None else 1
    log(f"[gate] chosen rule (pre-{REPLAY_CUT.date()} OOF): n_pump10_l1 >= {tau_star} "
        f"(train lift {best[0]:.3f})")

    gate_on_dates = set(d_all.loc[d_all["f_n_pump10_l1"] >= tau_star, "date"])

    # replay 기간 통계 (gate 가 그 기간에 며칠 ON 인가)
    rep_period = d_all[d_all["date"] >= REPLAY_CUT]
    rep_on = rep_period[rep_period["f_n_pump10_l1"] >= tau_star]
    log(f"[gate] replay-period ({REPLAY_CUT.date()}+) gate ON: "
        f"{len(rep_on)}/{len(rep_period)} days = {len(rep_on)/max(1,len(rep_period)):.2%}")

    ledger_paths = [
        ("output/paper_ledger.csv", "distribution"),
        ("output/shadow_ledger_recommend.csv", "recommend_R1"),
        ("output/shadow_ledger_recommend_r2.csv", "recommend_R2"),
        ("output/shadow_ledger_recommend_sustain.csv", "recommend_sustain"),
        ("output/shadow_ledger_pump_hunter.csv", "pump_hunter"),
    ]
    ledgers = [load_ledger(p, gate_on_dates, lab) for p, lab in ledger_paths]
    rep = replay_table(ledgers)
    rep.to_csv(f"{OUT}_replay.csv", index=False)
    log("\n[REPLAY] gate-ON vs ALL net (거래비용 차감):")
    log(rep.to_string(index=False))

    # ---- summary json ----
    summary = dict(
        db_span=[str(df['date'].min().date()), str(df['date'].max().date())],
        n_days_daily=int(len(daily)),
        K_pumprich=int(K), n_pump10_quantiles=desc,
        base_rate_pumprich=float((daily["n_pump10"] >= K).mean()),
        dead_day_rate=float((daily["n_pump10"] == 0).mean()),
        n_rule_combos_tried=int(n_tried),
        gate_rule=f"n_pump10_l1>={tau_star}",
        gate_tau=int(tau_star),
        gate_train_lift_precut=float(best[0]) if best[0] > 0 else None,
        replay_cut=str(REPLAY_CUT.date()),
        replay_period_silence_rate=float(1 - len(rep_on) / max(1, len(rep_period))),
        replay_rows=rep.to_dict("records"),
        wf_test_lifts=[r for r in wf.get("test_lift", pd.Series()).dropna().tolist()],
    )
    with open(f"{OUT}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"\n[done] wrote {OUT}_daily.csv / _rules.csv / _wf.csv / _replay.csv / _summary.json")


if __name__ == "__main__":
    main()
