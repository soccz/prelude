"""quant-evaluator — R1 baseline byte/value 일관성 독립 확인.
A1/A2/A3/r2 picks dump 의 R1 row 가 동일 (date,market,net) 인지 대조.
"""
import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "output"


def agg(path, col, r1val):
    d = pd.read_csv(OUT / path)
    d = d[d[col] == r1val].dropna(subset=["net"])
    sl = float((d["outcome"].astype(str) == "sl").mean()) if "outcome" in d.columns else np.nan
    return len(d), float(d["net"].mean()), sl


print("=== R1 baseline 재집계 (각 dump) ===")
print("A1 R1 :", agg("ch_sustainability_picks_v1.csv", "policy", "R1_baseline"))
print("A2 R1 :", agg("ch_regime_split_picks_v1.csv", "policy", "R1_baseline"))
print("A3 R1 :", agg("ch_features_picks_v1.csv", "variant", "BASE"))
print("r2 R1 :", agg("r2_challenger_picks_v1.csv", "policy", "R1_ratio"))


def r1set(path, col, val):
    d = pd.read_csv(OUT / path)
    d = d[d[col] == val][["date", "market", "net"]].dropna()
    d["date"] = pd.to_datetime(d["date"]).dt.date
    return d.sort_values(["date", "market"]).reset_index(drop=True)


a1 = r1set("ch_sustainability_picks_v1.csv", "policy", "R1_baseline")
r2 = r1set("r2_challenger_picks_v1.csv", "policy", "R1_ratio")
a2 = r1set("ch_regime_split_picks_v1.csv", "policy", "R1_baseline")
a3 = r1set("ch_features_picks_v1.csv", "variant", "BASE")

sa1 = set(map(tuple, a1[["date", "market"]].values))
sr2 = set(map(tuple, r2[["date", "market"]].values))
sa2 = set(map(tuple, a2[["date", "market"]].values))
sa3 = set(map(tuple, a3[["date", "market"]].values))
print("\n=== (date,market) 집합 일치 ===")
print("A1==r2 :", sa1 == sr2, "| A1==A2 :", sa1 == sa2, "| A1==A3 :", sa1 == sa3)

m = a1.merge(r2, on=["date", "market"], suffixes=("_a1", "_r2"))
print("A1<->r2 merged rows", len(m), "net max|Δ|", float((m.net_a1 - m.net_r2).abs().max()))
m3 = a1.merge(a3, on=["date", "market"], suffixes=("_a1", "_a3"))
print("A1<->A3 merged rows", len(m3), "net max|Δ|", float((m3.net_a1 - m3.net_a3).abs().max()))
m2 = a1.merge(a2, on=["date", "market"], suffixes=("_a1", "_a2"))
print("A1<->A2 merged rows", len(m2), "net max|Δ|", float((m2.net_a1 - m2.net_a2).abs().max()))
