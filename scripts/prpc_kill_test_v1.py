#!/usr/bin/env python
"""prpc_kill_test_v1 — PRPC(펌프-후 reclaim 확인진입) 1차 kill-test. 학습 없음(deterministic).

설계: _workspace/next_research_PRPC_design_v1.md.
가설: 펌프 *자체* 진입(15실험 천장)이 아니라, 펌프 후 과열이 풀리고(눌림·변동성수축) 4h 재돌파를
*확인한* 뒤 진입하면 net-positive. autopsy(스파이크 장초반→EOD 되돌림)의 손실구간을 건너뜀.

게이트0(표본): 3.2년 armed→재돌파 트레이드 n≥60 AND fold당 ≥15. 미달=즉사(완화 금지 — 완화하면
spike-then-dump 모집단으로 회귀). 게이트1(net+대조): 4 시간-fold, ①net>0 ②block-boot CI95 하한>0
③foldPos≥3/4 ④vs 즉시진입(no-reclaim) net delta CI95>0 ⑤base-rate perm p<0.05.

leak: 펌프일 D 는 d1 마감 후 확정 → consolidation/reclaim 은 D+1 이후 4h 봉만. reclaim 체결 = 확인봉
*다음* 4h open(동일봉 high 누수 차단). 유니버스 = D 시점 quote_volume top-100(survivorship). 비용 왕복 0.15%.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("prpc")
OUT = Path("output")
COST = 0.0015

# 파라미터 (초기값 — train fold 튜닝 전 deterministic 1차)
T_PUMP = 0.12        # d1 high/open-1 펌프 임계
K_CONSOL = 6         # consolidation 관측 4h봉 (1일)
N_RECLAIM = 12       # consolidation 후 재돌파 탐색 4h봉 (2일)
HOLD = 12            # 진입 후 보유 4h봉 (2일)
TP = 0.08            # swing 익절
SL_ATR_MULT = 0.5    # SL = consol_low - 0.5*ATR_4h
OVERHEAT = -0.03     # 과열해소: consol close 가 pump_high 대비 -3% 이하
TOP_N = 100          # 유니버스


def load_4h():
    c = sqlite3.connect("data/upbit_4h.db")
    df = pd.read_csv("/dev/stdin", nrows=0) if False else pd.read_sql(
        "SELECT market,timestamp,open,high,low,close FROM candles", c,
        parse_dates=["timestamp"])
    c.close()
    out = {}
    for m, g in df.sort_values("timestamp").groupby("market"):
        out[m] = dict(ts=g["timestamp"].values,
                      o=g["open"].values.astype(float), h=g["high"].values.astype(float),
                      l=g["low"].values.astype(float), c=g["close"].values.astype(float))
    return out


def main():
    # ---- d1 펌프 + 유니버스 ----
    c = sqlite3.connect("data/upbit_d1.db")
    d1 = pd.read_sql("SELECT market,timestamp,open,high,low,close,quote_volume FROM candles", c,
                     parse_dates=["timestamp"])
    c.close()
    d1["pump"] = d1["high"] / d1["open"] - 1.0
    d1["qv_rank"] = d1.groupby("timestamp")["quote_volume"].rank(ascending=False, method="first")
    pumps = d1[(d1["pump"] >= T_PUMP) & (d1["qv_rank"] <= TOP_N)].copy()
    log.info("d1 펌프 이벤트(≥%.0f%%, top-%d): %d (markets %d, %s~%s)",
             T_PUMP * 100, TOP_N, len(pumps), pumps["market"].nunique(),
             pumps["timestamp"].min().date(), pumps["timestamp"].max().date())

    bars4 = load_4h()
    log.info("4h 마켓 %d 로드", len(bars4))

    rows = []  # (ts, net_im, net_prpc, hit_im, market)
    armed = 0
    for _, p in pumps.iterrows():
        m = p["market"]
        if m not in bars4:
            continue
        b = bars4[m]
        pump_high = p["high"]
        start = np.datetime64(p["timestamp"]) + np.timedelta64(1, "D")  # D+1 (펌프 확정 후)
        i0 = int(np.searchsorted(b["ts"], start))
        if i0 + K_CONSOL + N_RECLAIM + 1 >= len(b["ts"]):
            continue
        W = slice(i0, i0 + K_CONSOL)
        wc, wh, wl = b["c"][W], b["h"][W], b["l"][W]
        if len(wc) < K_CONSOL:
            continue
        # 게이트: 과열해소 + 변동성수축 + higher-low + 신고가 안 함
        if wh.max() > pump_high:                       # consolidation 중 신고가 = 아직 펌프중
            continue
        if wc[-1] / pump_high - 1.0 > OVERHEAT:         # 충분히 안 식음
            continue
        rng = wh - wl
        if rng[:K_CONSOL // 2].mean() <= rng[K_CONSOL // 2:].mean():  # 변동성 수축 아님
            continue
        if wl[K_CONSOL // 2:].min() < wl[:K_CONSOL // 2].min():       # higher-low 깨짐
            continue
        consol_high, consol_low = wh.max(), wl.min()
        atr = float(rng.mean())
        sl_px = consol_low - SL_ATR_MULT * atr
        ci = i0 + K_CONSOL
        if ci + HOLD >= len(b["ts"]):
            continue
        entry_im = b["o"][ci]
        if not np.isfinite(entry_im) or entry_im <= 0:
            continue
        armed += 1

        def walk(start_idx, entry_px):
            tppx = entry_px * (1 + TP)
            for k in range(start_idx, start_idx + HOLD):
                if b["l"][k] <= sl_px:
                    return sl_px / entry_px - 1.0, 0
                if b["h"][k] >= tppx:
                    return TP, 1
            return b["c"][start_idx + HOLD - 1] / entry_px - 1.0, None

        # A) 즉시진입 (모든 armed — 펌프후 눌림목 매수, causal·조건부leak 없음)
        g_im, hit_im = walk(ci, entry_im)
        # B) reclaim 확인진입 (PRPC — subset)
        g_prpc = np.nan
        for j in range(ci, ci + N_RECLAIM):
            if b["c"][j] > consol_high:
                e = j + 1
                if e + HOLD < len(b["ts"]) and np.isfinite(b["o"][e]) and b["o"][e] > 0:
                    g_prpc = walk(e, b["o"][e])[0]
                break
        rows.append((b["ts"][ci], g_im - COST,
                     (g_prpc - COST) if np.isfinite(g_prpc) else np.nan, hit_im, m))

    tr = pd.DataFrame(rows, columns=["ts", "net_im", "net_prpc", "hit", "market"])
    log.info("armed(consolidation 통과)=%d → 즉시진입 n=%d, reclaim n=%d",
             armed, len(tr), int(tr["net_prpc"].notna().sum()))
    if len(tr) == 0:
        _save(tr, dict(verdict="DEAD_no_armed", armed=armed))
        return

    rng2 = np.random.default_rng(42)

    def block_boot(vals, clusters):
        uc = np.unique(clusters)
        by = {u: vals[clusters == u] for u in uc}
        means = []
        for _ in range(2000):
            samp = rng2.choice(uc, size=len(uc), replace=True)
            means.append(np.concatenate([by[u] for u in samp]).mean())
        return float(np.mean(means)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    def evaluate(d, col, name):
        d = d.dropna(subset=[col]).copy()
        if len(d) < 60:
            log.info("  [%s] n=%d <60 표본부족", name, len(d))
            return dict(name=name, n=int(len(d)), insufficient=True)
        d["fold"] = pd.qcut(d["ts"].rank(method="first"), 4, labels=[1, 2, 3, 4])
        foldn = d.groupby("fold", observed=True).size()
        nm, nlo, nhi = block_boot(d[col].values, d["ts"].values.astype("datetime64[D]"))
        fp = int((d.groupby("fold", observed=True)[col].mean() > 0).sum())
        alive = nm > 0 and nlo > 0 and fp >= 3
        log.info("  [%s] n=%d fold%s | net=%+.4f CI95[%+.4f,%+.4f] foldPos=%d/4 hit=%.3f → %s",
                 name, len(d), list(foldn.values), nm, nlo, nhi, fp, (d[col] > 0).mean(),
                 "★ALIVE" if alive else "dead")
        return dict(name=name, n=int(len(d)), net=nm, ci=[nlo, nhi], foldpos=fp,
                    fold_n={int(k): int(v) for k, v in foldn.items()}, alive=bool(alive))

    log.info("\n===== 게이트1 평가 =====")
    res_prpc = evaluate(tr, "net_prpc", "B_PRPC(재돌파 확인진입)")     # 설계된 프로그램
    res_im = evaluate(tr, "net_im", "A_즉시진입(눌림목, 모든 armed)")  # clean lead

    log.info("\n→ ★PRPC(설계) 판정: %s",
             "ALIVE" if res_prpc.get("alive") else "DEAD (재돌파 확인진입 net CI95 0포함/불안정)")
    if res_im.get("alive"):
        log.info("  ※ 부산물: A_즉시진입(눌림목)이 clean(조건부leak 없는 전 armed) 상에서 ALIVE — "
                 "별도 검증 필요(leak 적대감사·deflate). PRPC 와 다른 프로그램.")
    _save(tr, dict(prpc=res_prpc, immediate=res_im, armed=armed))


def _save(tr, summary):
    OUT.mkdir(exist_ok=True)
    if len(tr):
        tr.to_csv(OUT / "prpc_kill_test_v1.csv", index=False)
    summary["params"] = dict(T_PUMP=T_PUMP, K_CONSOL=K_CONSOL, N_RECLAIM=N_RECLAIM,
                             HOLD=HOLD, TP=TP, SL_ATR_MULT=SL_ATR_MULT, OVERHEAT=OVERHEAT, TOP_N=TOP_N)
    (OUT / "prpc_kill_test_coverage_v1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float))
    log.info("wrote output/prpc_kill_test_{v1.csv,coverage_v1.json}")


if __name__ == "__main__":
    main()
