"""Bucket-based calibration loader — paper ledger backfill 의 calibration_*.csv 활용.

운영 원칙 (사용자 확정 2026-05-04):
  - raw model probability 출력 금지
  - rare event 에서 isotonic 도 극단부 흔들림 → bucket-based actual hit 가 더 안전
  - 표시 = "top decile historical hit: 58%" (n 표기 + rare 표기)

칼리브레이션 데이터:
  output/calibration_h2.csv, calibration_h5.csv, calibration_h6.csv
  scripts/calibration_paper_ledger.py 가 backfill 또는 operational ledger 에서 갱신

사용:
    from signals.bucket_calibration import BucketCalibrator
    calib = BucketCalibrator.load()
    info = calib.lookup("h5", 0.899)
    # → {"bucket": 9, "actual_hit_pct": 11.6, "n": 508, "raw_pred_pct": 60.3}
    text = calib.format_for_alert("h5", 0.899)
    # → "11.6% hist hit (decile 9, n=508, rare)"
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


HEAD_LABELS = {
    "h2": "+3% in 4h",
    "h5": "+20% tail",
    "h6": "+5% in 24h",
}

# rare-event head — 표시에 "rare" 부가
RARE_HEADS = {"h5"}


@dataclass
class BucketCalibrator:
    head_buckets: dict  # head_id -> DataFrame
    summary: Optional[dict] = None

    @classmethod
    def load(cls, calib_dir: str | Path = "output") -> "BucketCalibrator":
        calib_dir = Path(calib_dir)
        head_buckets = {}
        for head_id in HEAD_LABELS:
            p = calib_dir / f"calibration_{head_id}.csv"
            if p.exists():
                df = pd.read_csv(p)
                # ensure ordered by bucket
                df = df.sort_values("bucket").reset_index(drop=True)
                head_buckets[head_id] = df
        summary = None
        sp = calib_dir / "calibration_summary.json"
        if sp.exists():
            import json
            with open(sp) as f:
                summary = json.load(f)
        return cls(head_buckets=head_buckets, summary=summary)

    def is_loaded(self, head_id: str) -> bool:
        return head_id in self.head_buckets and len(self.head_buckets[head_id]) > 0

    def lookup(self, head_id: str, score: float) -> Optional[dict]:
        """raw score → bucket info (actual hit, decile, n)."""
        if head_id not in self.head_buckets:
            return None
        df = self.head_buckets[head_id]
        if len(df) == 0:
            return None
        # find bucket containing score
        match = df[(df["score_min"] <= score) & (score <= df["score_max"])]
        if len(match) == 0:
            # outside range → snap to nearest extreme bucket
            if score < df["score_min"].iloc[0]:
                b = df.iloc[0]
            else:
                b = df.iloc[-1]
        else:
            b = match.iloc[0]
        return {
            "bucket": int(b["bucket"]),
            "n_buckets_total": int(len(df)),
            "actual_hit_pct": float(b["actual_hit_pct"]),
            "n": int(b["n"]),
            "raw_pred_pct": float(b["mean_pred_pct"]),
            "score_range": (float(b["score_min"]), float(b["score_max"])),
        }

    def format_for_alert(self, head_id: str, score: float, show_raw: bool = False) -> str:
        info = self.lookup(head_id, score)
        if info is None:
            return "calibration unavailable — raw score hidden"
        n_buckets = info["n_buckets_total"]
        decile_idx = info["bucket"] + 1  # 1-indexed for human
        rare_tag = ", rare" if head_id in RARE_HEADS else ""
        s = f"{info['actual_hit_pct']:.1f}% hist hit (decile {decile_idx}/{n_buckets}, n={info['n']}{rare_tag})"
        return s
