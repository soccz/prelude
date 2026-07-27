"""Phase 1 모델 — XGBoost multi-class softprob.

설계 (SIGNAL.md §4.1):
  - 라벨: 6-class (signals/labels.py DEFAULT_BINS)
  - objective: multi:softprob → 각 bin 확률 출력
  - sample_weight=balanced (sparse bin 보강)
  - Optuna 튜닝 (mlogloss + per-bin macro F1)
  - 출처: gan_t/training/pump_trainer.py 의 4-class 패턴 → 6-class 확장

핵심 클래스 / 함수:
  - prepare_features(panel) — panel → X / y / feature_names
  - XGBPhase1                — 학습 / 추론 / save / load
  - tune_with_optuna         — HPO
  - train_full               — panel → trained model (단순 wrapper)
  - evaluate                 — metrics dict
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    f1_score,
    log_loss,
    accuracy_score,
)
from sklearn.utils.class_weight import compute_sample_weight

# Optuna verbose off
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================================
# 학습용 피처 / 라벨 분리
# ============================================================================
EXCLUDE_COLS = {
    "market", "timestamp",
    "open", "high", "low", "close", "volume", "quote_volume",
    "label", "max_return",
    "btc_regime",  # str (regime_id 대신)
}

DEFAULT_NUM_CLASS = 6


def prepare_features(panel: pd.DataFrame, num_class: int = DEFAULT_NUM_CLASS) -> tuple:
    """panel → (X, y, feature_names).

    label NaN 또는 비정상 행 제거.
    feature 결측은 NaN 그대로 (XGBoost native 처리).
    """
    df = panel.copy()
    df = df[df["label"].notna()]
    df = df[df["label"].between(0, num_class - 1)]

    feature_names = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_names].astype(float).values
    y = df["label"].astype(int).values
    return X, y, feature_names


# ============================================================================
# 모델 클래스
# ============================================================================
@dataclass
class XGBPhase1:
    num_class: int = DEFAULT_NUM_CLASS
    params: dict = field(default_factory=dict)
    model: Optional[xgb.XGBClassifier] = None
    feature_names: Optional[list[str]] = None

    @staticmethod
    def default_params(num_class: int = DEFAULT_NUM_CLASS) -> dict:
        return {
            "objective": "multi:softprob",
            "num_class": num_class,
            "eval_metric": "mlogloss",
            "n_estimators": 400,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.0,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": 42,
        }

    def __post_init__(self):
        if not self.params:
            self.params = self.default_params(self.num_class)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        eval_set: Optional[list] = None,
        verbose: bool = False,
    ):
        if sample_weight is None:
            sample_weight = compute_sample_weight(class_weight="balanced", y=y)

        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(
            X, y, sample_weight=sample_weight, eval_set=eval_set, verbose=verbose
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained")
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def save(self, ckpt_path: Path | str, config_path: Optional[Path | str] = None):
        ckpt_path = Path(ckpt_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(ckpt_path))

        if config_path:
            config_path = Path(config_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w") as f:
                json.dump({
                    "num_class": self.num_class,
                    "params": self.params,
                    "feature_names": self.feature_names,
                }, f, indent=2, default=str)

    @classmethod
    def load(cls, ckpt_path: Path | str, config_path: Optional[Path | str] = None) -> "XGBPhase1":
        ckpt_path = Path(ckpt_path)
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                cfg = json.load(f)
            obj = cls(num_class=cfg["num_class"], params=cfg["params"], feature_names=cfg.get("feature_names"))
        else:
            obj = cls()

        obj.model = xgb.XGBClassifier(**obj.params)
        obj.model.load_model(str(ckpt_path))
        return obj


# ============================================================================
# Optuna HPO
# ============================================================================
def tune_with_optuna(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_class: int = DEFAULT_NUM_CLASS,
    n_trials: int = 50,
    timeout: Optional[int] = None,
) -> dict:
    """
    Optuna로 XGBoost 하이퍼파라미터 튜닝.
    objective: validation mlogloss + per-bin macro F1 (가중 합).
    """

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "multi:softprob",
            "num_class": num_class,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": 42,
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10, log=True),
        }
        sw = compute_sample_weight(class_weight="balanced", y=y_train)
        m = xgb.XGBClassifier(**params)
        m.fit(X_train, y_train, sample_weight=sw, verbose=False)
        prob = m.predict_proba(X_val)
        pred = prob.argmax(axis=1)

        ll = log_loss(y_val, prob, labels=list(range(num_class)))
        f1 = f1_score(y_val, pred, average="macro", zero_division=0)
        # mlogloss 낮을수록 좋음, F1 높을수록 좋음 → 통합 (낮을수록 좋게 변환)
        score = ll - 0.3 * f1
        return score

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    return study.best_params


# ============================================================================
# 학습 wrapper (단순 train/val split, Purged WF 는 별도)
# ============================================================================
def train_full(
    panel: pd.DataFrame,
    val_ratio: float = 0.2,
    tune: bool = False,
    n_trials: int = 50,
    num_class: int = DEFAULT_NUM_CLASS,
) -> tuple[XGBPhase1, dict]:
    """
    panel → trained XGBPhase1 + metrics.

    val_ratio: 시간 기준 마지막 N% 를 validation 으로 (look-ahead 없음).
    tune: Optuna HPO 수행 여부.
    """
    df = panel[panel["label"].notna()].copy()
    df = df.sort_values("timestamp")

    X, y, feature_names = prepare_features(df, num_class=num_class)
    n = len(X)
    split = int(n * (1 - val_ratio))
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    if tune:
        print(f"Optuna HPO {n_trials} trials...")
        best_params = tune_with_optuna(X_tr, y_tr, X_val, y_val, num_class, n_trials)
        params = XGBPhase1.default_params(num_class)
        params.update(best_params)
    else:
        params = XGBPhase1.default_params(num_class)

    model = XGBPhase1(num_class=num_class, params=params)
    model.feature_names = feature_names
    model.fit(X_tr, y_tr)

    metrics = evaluate(model, X_val, y_val)
    metrics["n_train"] = len(X_tr)
    metrics["n_val"] = len(X_val)
    metrics["n_features"] = len(feature_names)
    return model, metrics


# ============================================================================
# 평가 metrics
# ============================================================================
def evaluate(model: XGBPhase1, X: np.ndarray, y: np.ndarray) -> dict:
    prob = model.predict_proba(X)
    pred = prob.argmax(axis=1)

    n_class = prob.shape[1]
    one_hot = np.eye(n_class)[y]
    brier = float(np.mean(np.sum((prob - one_hot) ** 2, axis=1)))

    return {
        "mlogloss": float(log_loss(y, prob, labels=list(range(n_class)))),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "brier": brier,
        "per_class_count": [int((y == c).sum()) for c in range(n_class)],
        "per_class_predicted": [int((pred == c).sum()) for c in range(n_class)],
        "per_class_acc": [
            float(accuracy_score(y[y == c], pred[y == c])) if (y == c).any() else 0.0
            for c in range(n_class)
        ],
    }


# ============================================================================
# 직접 실행 — 작은 학습 검증
# ============================================================================
if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from data.database import list_markets, load_candles
    from data.market_universe import signal_eligible_markets
    from signals.features import assemble_training_panel

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/upbit_d1.db")
    parser.add_argument("--btc-market", default="KRW-BTC")
    parser.add_argument("--limit", type=int, default=30, help="샘플 코인 갯수")
    parser.add_argument("--tune", action="store_true", help="Optuna HPO 수행")
    parser.add_argument("--n-trials", type=int, default=20)
    args = parser.parse_args()

    print(f"loading {args.limit} markets from {args.db}...")
    markets = signal_eligible_markets(list_markets(args.db))[: args.limit]
    candles = {}
    for m in markets:
        d = load_candles(args.db, m)
        if len(d) > 0:
            candles[m] = d
    btc = load_candles(args.db, args.btc_market)

    print(f"building panel from {len(candles)} markets...")
    panel = assemble_training_panel(candles, btc, normalize=True)
    print(f"Panel: {panel.shape}, label dist: {dict(panel['label'].value_counts().sort_index())}")

    print(f"\ntraining {'(tune)' if args.tune else '(default params)'}...")
    model, metrics = train_full(panel, tune=args.tune, n_trials=args.n_trials)

    print("\n=== Validation metrics ===")
    for k, v in metrics.items():
        if isinstance(v, list):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # save
    ckpt = Path("signals/models/ckpt/phase1_smoke.json")
    cfg = Path("signals/models/configs/phase1_smoke.json")
    model.save(ckpt, cfg)
    print(f"\nSaved: {ckpt}")
