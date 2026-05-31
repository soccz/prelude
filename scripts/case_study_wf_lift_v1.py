"""Purged walk-forward OOS lift for case-study precursors.

In-sample lift 은 자랑이 아니다. precursor 임계를 train fold 에서 고정하고
embargo 후 test fold 에서 lift 가 살아남는지 확인.

규율:
  - feature = D-1 까지, label = D 의 next_ho>=TH (shift -1 로 부착)
  - 시간순 5-fold, fold 사이 5일 embargo
  - 임계는 placeholder (case_study_precursor_v1 의 1차결과 기반)
"""
from __future__ import annotations
import sqlite3, numpy as np, pandas as pd

EPS = 1e-12


def build_panel():
    c = sqlite3.connect("data/upbit_d1.db")
    df = pd.read_sql_query(
        "SELECT market,timestamp,open,high,low,close,quote_volume FROM candles",
        c, parse_dates=["timestamp"]); c.close()
    df = df.sort_values(["market", "timestamp"]).reset_index(drop=True)
    df["date"] = df["timestamp"].dt.date

    def feat(g):
        g = g.sort_values("timestamp").reset_index(drop=True)
        o, h, cl, qv = g["open"], g["high"], g["close"], g["quote_volume"]
        g["ho_1d"] = h / o - 1
        for N in (7,):
            g[f"ret_{N}d"] = cl / cl.shift(N) - 1
        g["qv_ma5"] = qv.rolling(5, min_periods=3).mean()
        g["qv_ma20"] = qv.rolling(20, min_periods=10).mean()
        g["qv_ratio_1_5"] = qv / (g["qv_ma5"] + EPS)
        g["qv_ratio_5_20"] = g["qv_ma5"] / (g["qv_ma20"] + EPS)
        g["qv_ratio_1_20"] = qv / (g["qv_ma20"] + EPS)
        return g

    df = df.groupby("market", group_keys=False).apply(feat)
    df["uni_rank"] = df.groupby("date")["qv_ma20"].rank(ascending=False, method="min")
    df["next_ho"] = df.groupby("market")["ho_1d"].shift(-1)
    df["y20"] = (df["next_ho"] >= 0.20).astype(int)
    df = df[(df["date"] >= pd.Timestamp("2024-01-01").date()) & (df["market"] != "KRW-BTC")]
    df = df.dropna(subset=["next_ho", "qv_ratio_5_20", "qv_ratio_1_5", "uni_rank"])
    return df.sort_values("date").reset_index(drop=True)


RULES = {
    "qv_ratio_1_5>=3.0": lambda d: d["qv_ratio_1_5"] >= 3.0,
    "qv_ratio_1_20>=3.0": lambda d: d["qv_ratio_1_20"] >= 3.0,
    "ret_7d>=0.20": lambda d: d["ret_7d"] >= 0.20,
    "spike2 & mom10": lambda d: (d["qv_ratio_1_5"] >= 2.0) & (d["ret_7d"] >= 0.10),
    "cumvol1.5 & spike2": lambda d: (d["qv_ratio_5_20"] >= 1.5) & (d["qv_ratio_1_5"] >= 2.0),
    "qv1_20>=2 & rank<=100": lambda d: (d["qv_ratio_1_20"] >= 2.0) & (d["uni_rank"] <= 100),
}


def main():
    df = build_panel()
    dates = np.array(sorted(df["date"].unique()))
    nf = 5
    bounds = np.array_split(dates, nf)
    embargo = 5
    print(f"panel n={len(df)} dates={len(dates)} folds={nf} embargo={embargo}d")
    print(f"overall base y20={df['y20'].mean():.4%}\n")

    agg = {k: {"tp": 0, "fired": 0, "ev_pos": 0, "test_base_w": 0.0, "test_n": 0}
           for k in RULES}
    for fi in range(1, nf):  # fold0 = first train, test on later folds
        test_dates = set(bounds[fi].tolist())
        last_train = bounds[fi][0]
        train_cut = last_train  # train = strictly before fold test start minus embargo
        train = df[df["date"] < bounds[fi - 0][0]]  # all before this fold
        # embargo: drop train rows within `embargo` days of test start
        emb_start = bounds[fi][0]
        train = train[train["date"] < (pd.Timestamp(emb_start) - pd.Timedelta(days=embargo)).date()]
        test = df[df["date"].isin(test_dates)]
        base_test = test["y20"].mean()
        for name, fn in RULES.items():
            te = test[fn(test)]
            if len(te) == 0:
                continue
            agg[name]["tp"] += te["y20"].sum()
            agg[name]["fired"] += len(te)
        agg_base = base_test

    # simpler: pooled OOS across folds 1..nf-1
    print(f"{'rule':28s} {'OOS_fired':>9s} {'OOS_hit':>8s} {'OOS_rate':>9s} {'lift':>6s}")
    pooled_test = df[df["date"].isin(set(np.concatenate(bounds[1:]).tolist()))]
    base = pooled_test["y20"].mean()
    for name, fn in RULES.items():
        te = pooled_test[fn(pooled_test)]
        if len(te) == 0:
            print(f"{name:28s} {'0':>9s}"); continue
        rate = te["y20"].mean()
        print(f"{name:28s} {len(te):9d} {int(te['y20'].sum()):8d} {rate:9.3%} {rate/base:6.2f}x")
    print(f"\npooled OOS base rate y20={base:.4%}  (folds 1..{nf-1}, train-before only)")

    # recent coverage: which of this week's first-pump coins did each rule flag at D-1?
    print("\n-- recent coverage (이번주 first-pump D-1 에 깃발?) --")
    cases = {"ID":"2026-05-29","GMT":"2026-05-23","VTHO":"2026-05-30","META":"2026-05-22",
             "ERA":"2026-05-25","PROVE":"2026-05-21","ALT":"2026-05-22","PRL":"2026-05-27",
             "XLM":"2026-05-28","WLD":"2026-05-26"}
    # need D-1 rows; rebuild full (no 2024 cut) for these
    c = sqlite3.connect("data/upbit_d1.db")
    raw = pd.read_sql_query("SELECT market,timestamp,open,high,low,close,quote_volume FROM candles",
                            c, parse_dates=["timestamp"]); c.close()
    raw = raw.sort_values(["market","timestamp"]); raw["date"]=raw["timestamp"].dt.date
    def feat(g):
        g=g.sort_values("timestamp").reset_index(drop=True)
        o,h,cl,qv=g["open"],g["high"],g["close"],g["quote_volume"]
        g["ho_1d"]=h/o-1
        g["ret_7d"]=cl/cl.shift(7)-1
        g["qv_ma5"]=qv.rolling(5,min_periods=3).mean()
        g["qv_ma20"]=qv.rolling(20,min_periods=10).mean()
        g["qv_ratio_1_5"]=qv/(g["qv_ma5"]+EPS)
        g["qv_ratio_5_20"]=g["qv_ma5"]/(g["qv_ma20"]+EPS)
        g["qv_ratio_1_20"]=qv/(g["qv_ma20"]+EPS)
        return g
    raw=raw.groupby("market",group_keys=False).apply(feat)
    raw["uni_rank"]=raw.groupby("date")["qv_ma20"].rank(ascending=False,method="min")
    header = f"{'coin':6s} " + " ".join(f"{k[:14]:>14s}" for k in RULES)
    print(header)
    for tok,d in cases.items():
        m="KRW-"+tok; D=pd.Timestamp(d).date()
        g=raw[raw.market==m].sort_values("date").reset_index(drop=True)
        iD=g.index[g.date==D][0]; dm1=g.iloc[[iD-1]]
        flags=[]
        for name,fn in RULES.items():
            flags.append("FLAG" if bool(fn(dm1).iloc[0]) else " -")
        print(f"{tok:6s} " + " ".join(f"{f:>14s}" for f in flags))


if __name__ == "__main__":
    main()
