r"""quant-evaluator — A1 degeneracy 본질: 하방 개선이 '진짜 dump 회피'인가
'저변동 교체의 부산물(상방도 깎임)'인가.

A1 picks dump(R1 vs A1_sustain) 을 (date,market) 키로 정렬해:
  - KEPT  = A1 이 R1 과 동일하게 유지한 픽
  - DROPPED = R1 엔 있었는데 A1 이 버린 픽 (R1\A1)
  - ADDED   = A1 이 새로 넣은 교체 픽 (A1\R1)
DROPPED vs ADDED 의 net·하방(eod_net·down_low_ret)·상방(up_high_ret) 분해 →
교체가 '하방 나쁜 픽을 빼고 비슷한 상방의 픽을 넣었나' vs '상·하방 다 작은 저변동으로 갈았나'.
"""
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "output"
d = pd.read_csv(OUT / "ch_sustainability_picks_v1.csv")
d["date"] = pd.to_datetime(d["date"]).dt.date
r1 = d[d.policy == "R1_baseline"].copy()
a1 = d[d.policy == "A1_sustain"].copy()

r1["key"] = list(zip(r1.date, r1.market))
a1["key"] = list(zip(a1.date, a1.market))
r1k, a1k = set(r1.key), set(a1.key)

kept = r1[r1.key.isin(a1k)]
dropped = r1[~r1.key.isin(a1k)]   # R1 had, A1 removed
added = a1[~a1.key.isin(r1k)]     # A1 added new


def summ(df, tag):
    x = df.dropna(subset=["net"])
    print(f"{tag:10s} n={len(x):4d} | net {x.net.mean():+.5f} | "
          f"eod_net {x.eod_net.dropna().mean():+.5f} | "
          f"up_high {x.up_high_ret.dropna().mean():+.4f} | "
          f"down_low {x.down_low_ret.dropna().mean():+.4f} | "
          f"%SL {(x.outcome.astype(str)=='sl').mean():.3f} | "
          f"deepNoSL {(x.eod_net.dropna()<=-0.05).mean():.3f} | "
          f"pump20 {x.pump20_hit.dropna().mean():.4f} | "
          f"pump5 {(x.up_high_ret.dropna()>=0.05).mean():.3f}")


print("=== A1 substitution decomposition (R1 vs A1_sustain = dump_B q0.6) ===")
print(f"R1 picks={len(r1)} A1 picks={len(a1)} | kept={len(kept)} dropped(R1\\A1)={len(dropped)} added(A1\\R1)={len(added)}")
print(f"substitution rate = {len(added)/len(a1):.3f}\n")
summ(r1, "R1_all")
summ(a1, "A1_all")
summ(kept, "KEPT")
summ(dropped, "DROPPED")   # R1 가 갖고있던, A1 이 버린 픽
summ(added, "ADDED")       # A1 이 새로 넣은 픽

print("\n=== 핵심 판별 ===")
dr = dropped.dropna(subset=["net"])
ad = added.dropna(subset=["net"])
print(f"DROPPED(버린) up_high {dr.up_high_ret.mean():+.4f}  vs  ADDED(넣은) up_high {ad.up_high_ret.mean():+.4f}  "
      f"(Δ상방 {ad.up_high_ret.mean()-dr.up_high_ret.mean():+.4f})")
print(f"DROPPED down_low {dr.down_low_ret.mean():+.4f}  vs  ADDED down_low {ad.down_low_ret.mean():+.4f}  "
      f"(Δ하방 {ad.down_low_ret.mean()-dr.down_low_ret.mean():+.4f})")
print(f"DROPPED net {dr.net.mean():+.5f}  vs  ADDED net {ad.net.mean():+.5f}  "
      f"(Δnet {ad.net.mean()-dr.net.mean():+.5f})")
print(f"DROPPED deepNoSL {(dr.eod_net<=-0.05).mean():.3f}  vs  ADDED deepNoSL {(ad.eod_net<=-0.05).mean():.3f}")
print(f"DROPPED pump5 {(dr.up_high_ret>=0.05).mean():.3f}  vs  ADDED pump5 {(ad.up_high_ret>=0.05).mean():.3f}")
# 결론 가늠: ADDED 가 하방 덜빠지면서 상방 유지면 '진짜 회피', 상방도 같이 작으면 '저변동 교체'
print("\n해석 가이드: ADDED 의 down_low 가 DROPPED 보다 덜 깊고(하방개선) up_high 가 비슷/유지면 진짜 dump 회피.")
print("            ADDED 의 up_high 도 DROPPED 보다 작으면 = 저변동 교체(상·하방 동시 절단).")
