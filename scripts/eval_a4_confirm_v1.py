"""quant-evaluator — A4 selective REJECT 재확인: OOF 게이트 net 음수 + fold 부호불안정 + CI 0횡단."""
import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "output"
comp = pd.read_csv(OUT / "ch_selective_compare_v1.csv")
pf = pd.read_csv(OUT / "ch_selective_perfold_v1.csv")

print("=== A4 OOF 게이트 (in-sample ORACLE 제외, 실거래 가능한 OOF 만) ===")
oof = comp[comp.select.str.contains("OOF", na=False) | (comp.label == "R1_baseline_everyday")]
for _, r in oof.iterrows():
    excl0 = (r.boot_npf_lo95 > 0) or (r.boot_npf_hi95 < 0)
    print(f"  {r['label'][:30]:30s} fire={r['fire_rate']:.3f} net/fired={r['net_per_fired']:+.5f} "
          f"Δvs_base={r['delta_net_per_fired']:+.5f} CI95[{r['boot_npf_lo95']:+.5f},{r['boot_npf_hi95']:+.5f}] "
          f"P(npf>0)={r['boot_p_npf_gt0']:.3f} {'*excl0*' if excl0 else '(spans0)'}")

print("\n=== fold 부호 불안정 (OOF 게이트별 delta 부호) ===")
for gate, g in pf.groupby(["gate", "dir"]):
    d = g.dropna(subset=["delta"])
    if len(d) == 0:
        continue
    signs = np.sign(d["delta"].values)
    pos = int((signs > 0).sum()); neg = int((signs < 0).sum())
    print(f"  {str(gate):28s} folds_fired={len(d)} delta+:{pos} delta-:{neg} "
          f"deltas={np.round(d['delta'].values,4)}")
print("\n해석: OOF net/fired 전부 음수 + CI 0 횡단 + fold 부호 혼재 → 견고한 edge 없음 → REJECT.")
