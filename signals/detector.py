"""Detector v1 — binary tail-pump detector with frozen OOF threshold.

운영 룰 (output/detector_threshold.json 에서 로드):
  - regime ∈ {bull_quiet, bull_volatile} → 게이트 통과
  - score >= threshold (artifact 고정값, 라이브 quantile 재계산 X)
  - per-day top cap (default 2)
  - bear regime → 침묵

핵심 약속 (사용자 운영 원칙):
  threshold 는 artifact 의 고정값을 그대로 쓴다.
  오늘 universe 의 quantile 을 다시 계산하지 않는다.

사용:
    from signals.detector import DetectorV1
    det = DetectorV1.load()
    alerts = det.detect(panel_today)  # → DataFrame: coin, score, regime, ...
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger("detector")


DEFAULT_CONFIG_PATH = "output/detector_threshold.json"


@dataclass
class DetectorConfig:
    version: str
    model_ckpt: str
    feature_cols: list
    target_pct: float
    regimes: list
    threshold: float
    cap: int
    bear_action: str
    framing: str

    @classmethod
    def load(cls, config_path: str | Path = DEFAULT_CONFIG_PATH,
             meta_path: Optional[str | Path] = None) -> "DetectorConfig":
        config_path = Path(config_path)
        with open(config_path) as f:
            cfg = json.load(f)
        meta_path = Path(meta_path or cfg["model_meta"])
        with open(meta_path) as f:
            meta = json.load(f)
        rule = cfg["operational_rule"]
        return cls(
            version=cfg["version"],
            model_ckpt=cfg["model_ckpt"],
            feature_cols=meta["feature_cols"],
            target_pct=rule["target_pct"],
            regimes=rule["regimes"],
            threshold=rule["threshold"],
            cap=rule["cap"],
            bear_action=rule.get("bear_action", "silence"),
            framing=cfg.get("framing", ""),
        )


class DetectorV1:
    """Binary tail detector — frozen threshold + regime gate + per-day cap."""

    def __init__(self, model: xgb.XGBClassifier, config: DetectorConfig):
        self.model = model
        self.config = config

    @classmethod
    def load(cls, config_path: str | Path = DEFAULT_CONFIG_PATH) -> "DetectorV1":
        cfg = DetectorConfig.load(config_path)
        model = xgb.XGBClassifier()
        model.load_model(cfg.model_ckpt)
        logger.info(f"loaded detector {cfg.version}: threshold={cfg.threshold:.4f}, "
                    f"regimes={cfg.regimes}, cap={cfg.cap}")
        return cls(model, cfg)

    def score(self, panel: pd.DataFrame) -> np.ndarray:
        """raw model score on panel rows. panel must have all feature_cols."""
        missing = [c for c in self.config.feature_cols if c not in panel.columns]
        if missing:
            raise ValueError(f"missing features: {missing[:5]}{'...' if len(missing)>5 else ''}")
        X = panel[self.config.feature_cols].astype(float).values
        return self.model.predict_proba(X)[:, 1]

    def detect(self, panel: pd.DataFrame, asof: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """Apply gate to panel.

        Returns DataFrame with columns:
          coin, btc_regime, score, threshold, passes_gate, alert_rank
        Only rows that pass full gate (regime + score + cap) are returned.
        """
        if len(panel) == 0:
            return pd.DataFrame()

        # universe filter: KRW only
        sub = panel[panel["market"].str.startswith("KRW-")].copy()
        if len(sub) == 0:
            return pd.DataFrame()

        # score
        sub["score"] = self.score(sub)
        sub["threshold"] = self.config.threshold

        # regime gate
        in_regime = sub["btc_regime"].isin(self.config.regimes)
        # score gate
        above_thr = sub["score"] >= self.config.threshold
        candidates = sub[in_regime & above_thr].copy()

        if len(candidates) == 0:
            return pd.DataFrame()

        # per-day cap (top score)
        candidates = candidates.sort_values(["timestamp", "score"], ascending=[True, False])
        candidates["alert_rank"] = candidates.groupby("timestamp").cumcount() + 1
        alerts = candidates[candidates["alert_rank"] <= self.config.cap].copy()

        return alerts[["timestamp", "market", "btc_regime", "score",
                       "threshold", "alert_rank"]].rename(columns={"market": "coin"})

    def diagnose(self, panel: pd.DataFrame) -> dict:
        """전체 panel 스코어링 요약 — 운영 모니터링용."""
        sub = panel[panel["market"].str.startswith("KRW-")].copy()
        if len(sub) == 0:
            return {"n": 0}
        sub["score"] = self.score(sub)
        in_regime = sub["btc_regime"].isin(self.config.regimes)
        above_thr = sub["score"] >= self.config.threshold

        return {
            "n_universe": int(len(sub)),
            "n_in_regime": int(in_regime.sum()),
            "n_above_threshold": int(above_thr.sum()),
            "n_pass_both": int((in_regime & above_thr).sum()),
            "score_max": float(sub["score"].max()),
            "score_p99": float(sub["score"].quantile(0.99)),
            "score_p99_5": float(sub["score"].quantile(0.995)),
            "threshold": float(self.config.threshold),
            "regimes_in_universe": sub["btc_regime"].value_counts().to_dict(),
        }
