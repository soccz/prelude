"""quant-evaluator — R1 byte-identical & R2 재정렬 독립 실측.

score_candidates(ranking='R1') 가 챔피언 출력과 동일한 top-3 를 내는지,
ranking='R2' 가 같은 head 확률에서 하방 픽을 교체하는지 확인.
self-contained.
"""
from __future__ import annotations
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from signals.recommend import score_candidates, R2_LAMBDA  # noqa: E402

ASOF = "2026-05-30"
print(f"R2_LAMBDA = {R2_LAMBDA}")
try:
    r1 = score_candidates(ASOF, slot="open", ranking="R1")
    r2 = score_candidates(ASOF, slot="open", ranking="R2")
except Exception as e:
    print(f"[ERR] score_candidates failed: {e!r}")
    sys.exit(2)


def top3(res):
    items = res.get("top3") or res.get("items") or []
    out = []
    for it in items[:3]:
        if isinstance(it, dict):
            out.append((it.get("market") or it.get("coin"),
                        it.get("p_up10"), it.get("p_dn5"), it.get("rr_ratio")))
    return out


print("R1 keys:", list(r1.keys()) if isinstance(r1, dict) else type(r1))
print("R1 top3:", top3(r1))
print("R2 top3:", top3(r2))
r1m = [t[0] for t in top3(r1)]
r2m = [t[0] for t in top3(r2)]
print("R1 markets:", r1m)
print("R2 markets:", r2m)
print("top3 markets identical?:", r1m == r2m)
print("any change (R2 reorders/replaces)?:", set(r1m) != set(r2m) or r1m != r2m)
