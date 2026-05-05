"""Calibration tracker — paper ledger 의 closed 행에서 head 별 calibration bucket.

목적 (사용자 caveat):
  raw model probability 90% 가 진짜 90% 인지 검증.
  rare-event 에서 90% 는 거의 항상 과신 — 데이터로 확인.

흐름:
  1. paper_ledger.csv 의 status="closed" 행 로드
  2. 각 head 별 (h2/h5/h6):
     - score 를 10 분위 bucket
     - bucket 별 mean predicted vs actual hit rate
     - 분위별 sample size, calibration error
  3. 출력:
     - output/calibration_<head>.csv
     - 알림 포맷용: 각 head 의 OOF top-bucket actual hit rate

사용:
    python scripts/calibration_paper_ledger.py
    python scripts/calibration_paper_ledger.py --min-samples 30  # 최소 표본
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HEADS_MAP = {
    "h2": ("p_h2_3pct_4h", "hit_h2"),
    "h5": ("p_h5_20pct_tail", "hit_h5"),
    "h6": ("p_h6_5pct_24h", "hit_h6"),
}


def calibration_table(scores: np.ndarray, hits: np.ndarray, n_buckets: int = 10) -> pd.DataFrame:
    """quantile bucket → mean predicted vs actual."""
    df = pd.DataFrame({"score": scores, "hit": hits.astype(int)})
    if len(df) < n_buckets * 5:
        n_buckets = max(2, len(df) // 5)
    df["bucket"] = pd.qcut(df["score"], q=n_buckets, duplicates="drop", labels=False)
    agg = df.groupby("bucket").agg(
        n=("hit", "size"),
        mean_pred_pct=("score", lambda x: x.mean() * 100),
        actual_hit_pct=("hit", lambda x: x.mean() * 100),
        score_min=("score", "min"),
        score_max=("score", "max"),
    ).reset_index()
    agg["calib_error_pp"] = agg["mean_pred_pct"] - agg["actual_hit_pct"]
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-ledger", default="output/paper_ledger.csv")
    parser.add_argument("--out-dir", default="output")
    parser.add_argument("--min-samples", type=int, default=30,
                        help="head 별 최소 closed 행 수 (이하면 스킵)")
    parser.add_argument("--n-buckets", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("calib")

    p = Path(args.paper_ledger)
    if not p.exists():
        log.error(f"ledger missing: {p}")
        sys.exit(1)

    ledger = pd.read_csv(p)
    closed = ledger[ledger["status"] == "closed"].copy()
    log.info(f"closed rows: {len(closed)} / total {len(ledger)}")

    if len(closed) < args.min_samples:
        log.warning(f"closed rows < min_samples ({args.min_samples}) — calibration 신뢰도 낮음")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for head_id, (p_col, hit_col) in HEADS_MAP.items():
        if p_col not in closed.columns or hit_col not in closed.columns:
            continue
        sub = closed[[p_col, hit_col]].dropna()
        if len(sub) < 10:
            log.info(f"{head_id}: too few samples ({len(sub)})")
            continue

        scores = sub[p_col].values.astype(float)
        hits = sub[hit_col].values.astype(int)
        base = float(hits.mean())

        # overall
        log.info(f"{head_id}: n={len(sub)}, base={base*100:.2f}%, "
                 f"mean_pred={scores.mean()*100:.2f}%")

        # per-bucket
        tbl = calibration_table(scores, hits, args.n_buckets)
        out_path = out_dir / f"calibration_{head_id}.csv"
        tbl.to_csv(out_path, index=False)
        log.info(f"  saved {out_path}")

        # top-bucket actual hit rate (for alert framing)
        top = tbl.iloc[-1]
        summary[head_id] = {
            "n_total": int(len(sub)),
            "base_actual_pct": float(base * 100),
            "mean_pred_pct": float(scores.mean() * 100),
            "top_bucket_n": int(top["n"]),
            "top_bucket_pred_range_pct": [float(top["score_min"] * 100), float(top["score_max"] * 100)],
            "top_bucket_mean_pred_pct": float(top["mean_pred_pct"]),
            "top_bucket_actual_hit_pct": float(top["actual_hit_pct"]),
            "top_bucket_calib_error_pp": float(top["calib_error_pp"]),
        }

        print(f"\n=== {head_id} calibration (n={len(sub)}, base={base*100:.2f}%) ===")
        print(tbl.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    summary_path = out_dir / "calibration_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"saved {summary_path}")

    print(f"\n=== Top-bucket actual hit rate (알림 framing 용) ===")
    for h, s in summary.items():
        print(f"  {h}: top bucket pred {s['top_bucket_mean_pred_pct']:.1f}% → actual {s['top_bucket_actual_hit_pct']:.1f}% "
              f"(n={s['top_bucket_n']}, calib_error {s['top_bucket_calib_error_pp']:+.1f}pp)")


if __name__ == "__main__":
    main()
