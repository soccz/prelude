"""Distribution Engine v1 — multi-head loader + setup matching + alert composition.

운영 흐름:
  1. dist_engine_v1 artifact 로드 (signals/models/ckpt/dist_engine_v1/)
     - meta.json : feature_cols, head config
     - <head>.json : XGBoost binary 모델 per head (h1~h7)
  2. panel 입력 → head 별 P(label) 추정
  3. setup 매칭 (signals/setups.py — S01~S04)
  4. universe filter (top50/top100/all)
  5. assemble alerts (top-K by composite score)
  6. format → 텔레그램 / paper ledger

핵심 설계:
  - distribution beta (Stage 1 dry-run): 자동매매 X, 분포 + setup 표시만
  - threshold 고정 X (distribution 자체가 출력)
  - setup 은 사람이 읽는 reason, head 확률은 raw signal

사용:
    from signals.distribution_engine import DistributionEngine
    eng = DistributionEngine.load()
    scored = eng.score_panel(panel)
    alerts = eng.assemble_alerts(scored, top_k=10, universe="top100")
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import xgboost as xgb

from signals.setups import SETUP_LIBRARY, evaluate_setup_rows

logger = logging.getLogger("dist_engine")

DEFAULT_CKPT_DIR = "signals/models/ckpt/dist_engine_v1"


@dataclass
class DistributionEngine:
    models: dict                # head_name -> XGBClassifier
    feature_cols: list
    head_meta: dict             # from meta.json
    setup_library: dict         # signals.setups.SETUP_LIBRARY

    @classmethod
    def load(cls, ckpt_dir: str | Path = DEFAULT_CKPT_DIR) -> "DistributionEngine":
        ckpt_dir = Path(ckpt_dir)
        meta_path = ckpt_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json missing: {meta_path}")
        with open(meta_path) as f:
            meta = json.load(f)
        feature_cols = meta["feature_cols"]
        head_meta = meta["heads"]
        models = {}
        for head_name in head_meta:
            mp = ckpt_dir / f"{head_name}.json"
            if not mp.exists():
                logger.warning(f"missing head ckpt: {head_name}")
                continue
            m = xgb.XGBClassifier()
            m.load_model(str(mp))
            models[head_name] = m
        logger.info(f"loaded {len(models)} heads from {ckpt_dir}")
        return cls(models, feature_cols, head_meta, SETUP_LIBRARY)

    def score_panel(self, panel: pd.DataFrame) -> pd.DataFrame:
        """returns panel with new columns p_<head_name>."""
        missing = [c for c in self.feature_cols if c not in panel.columns]
        if missing:
            raise ValueError(f"missing features: {missing[:5]}{'...' if len(missing)>5 else ''}")
        X = panel[self.feature_cols].astype(float).values
        out = panel.copy()
        for head_name, model in self.models.items():
            out[f"p_{head_name}"] = model.predict_proba(X)[:, 1]
        return out

    def assemble_alerts(self, scored: pd.DataFrame,
                         top_k: int = 10,
                         universe: str = "top100",
                         require_setup: bool = True) -> pd.DataFrame:
        """Score + setup 매칭 → top-K alert 후보.

        composite ranking:
          composite = p_h2 + p_h6 + 5 * p_h5
          (h5 가 rare event 라 weight 가산)

        require_setup=True: S01/S02/S03 중 하나 이상 fire 한 row 만 포함 (S04 는 context)
        """
        sub = scored[scored["market"].str.startswith("KRW-")].copy()

        # universe filter
        if universe.startswith("top") and "liq_rank_daily" in sub.columns:
            n = int(universe.replace("top", ""))
            sub = sub[sub["liq_rank_daily"] <= n]

        if len(sub) == 0:
            return pd.DataFrame()

        # setup detection (per execution batch). A rule that raises for every
        # row is a broken schema/rule contract, not a legitimate zero-fire day.
        setup_matches, _ = evaluate_setup_rows(
            (row for _, row in sub.iterrows()),
            setup_library=self.setup_library,
        )
        sub["setups"] = setup_matches

        # primary setups (S01/S02/S03), exclude S04 (context only)
        sub["primary_setups"] = sub["setups"].apply(
            lambda lst: [s for s in lst if s != "S04"]
        )
        sub["btc_context"] = sub.apply(
            lambda r: "S04" if "S04" in r["setups"] else "—", axis=1
        )

        if require_setup:
            sub = sub[sub["primary_setups"].apply(len) > 0]
        if len(sub) == 0:
            return pd.DataFrame()

        # composite score (rare h5 가산)
        h2 = sub.get("p_h2_hit_3_4h", pd.Series([0]*len(sub))).fillna(0)
        h5 = sub.get("p_h5_tail_20", pd.Series([0]*len(sub))).fillna(0)
        h6 = sub.get("p_h6_hit_5_24h", pd.Series([0]*len(sub))).fillna(0)
        sub["composite"] = h2 + h6 + 5 * h5
        sub = sub.sort_values("composite", ascending=False).head(top_k)
        return sub

    def diagnose(self, scored: pd.DataFrame, universe: str = "top100") -> dict:
        """Universe-level 진단 — 운영 모니터링용."""
        sub = scored[scored["market"].str.startswith("KRW-")].copy()
        n_universe = len(sub)
        if "liq_rank_daily" in sub.columns and universe.startswith("top"):
            n = int(universe.replace("top", ""))
            sub_filt = sub[sub["liq_rank_daily"] <= n].copy()
        else:
            sub_filt = sub.copy()

        setup_matches, setup_error_counts = evaluate_setup_rows(
            (row for _, row in sub_filt.iterrows()),
            setup_library=self.setup_library,
            log_partial_errors=True,
        )
        sub_filt["setups"] = setup_matches
        sub_filt["primary"] = sub_filt["setups"].apply(
            lambda lst: [s for s in lst if s != "S04"]
        )

        setup_counts = {
            sid: int(sub_filt["setups"].apply(lambda ids: sid in ids).sum())
            for sid in self.setup_library
        }

        return {
            "n_universe": n_universe,
            "n_universe_filtered": len(sub_filt),
            "n_with_any_primary_setup": int((sub_filt["primary"].apply(len) > 0).sum()),
            "setup_fire_counts": setup_counts,
            "setup_error_counts": setup_error_counts,
            "p_h2_p99_pct": float(sub_filt.get("p_h2_hit_3_4h", pd.Series([0])).quantile(0.99) * 100)
                             if "p_h2_hit_3_4h" in sub_filt.columns else 0,
            "p_h5_p99_pct": float(sub_filt.get("p_h5_tail_20", pd.Series([0])).quantile(0.99) * 100)
                             if "p_h5_tail_20" in sub_filt.columns else 0,
            "p_h6_p99_pct": float(sub_filt.get("p_h6_hit_5_24h", pd.Series([0])).quantile(0.99) * 100)
                             if "p_h6_hit_5_24h" in sub_filt.columns else 0,
        }
