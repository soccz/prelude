"""quant-evaluator independent audit of the Phase-1 pattern library.

목적 (researcher 주장 그대로 안 믿고 독립 재계산):
  1. LEAK 감사: feature shift(1) 실제 검증, BTC regime D-1 정합성 검증
     (regime_d1[D] == regime_raw[D-1]), 라벨이 미래(day-D)인지, cross-section
     rank 가 D-1 입력인지.
  2. portfolio-grade net 백테스트: (pattern × regime) top-decile 진입을
     equal-weight 일별 시계열로 → net Sharpe/Sortino/Calmar/MaxDD/누적/hit.
     ★ SL-first(비관) AND TP-first(낙관) 양쪽 bound + TP_EOD(SL 없음).
     0.15% 왕복 차감. ledger/metrics.py 의 Sharpe/MDD 재사용.
  3. selection-aware DSR: trials= 4각도 총 시도수 반영.

walk-forward OOS: 전역 fold 경계 공유 (regime sub-pop OOS), embargo 5d.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import sys
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from scipy import __version__ as scipy_version
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ops.artifact_provenance import (
    ArtifactSourceChangedError,
    ArtifactValidationError,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    file_set_identity,
    manifest_digest_matches,
    payload_digest,
    resolve_identity_path,
    sha256_bytes,
    strict_json_object,
    with_manifest_digest,
)
from ops.file_lock import file_lock
from data.database import list_markets, load_candles
from data.market_universe import is_excluded_signal_market
from signals.features import compute_btc_features
from scripts.univariate_precursor_lift_v1 import build_market_features, add_cross_sectional
from ledger.metrics import compute_sharpe, compute_mdd
from ledger.portfolio_metrics import annualized_sortino, normalize_kst_date

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "upbit_d1.db"
OUT_CSV = ROOT / "output" / "eval_library_audit_v1.csv"
OUT_JSON = ROOT / "output" / "eval_library_audit_v1.json"
RT = 0.0015
TP, SL = 0.10, 0.05
PPY = 365
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("audit")

REGIMES = ["bull_quiet", "bull_volatile", "bear_quiet", "bear_volatile"]
# 평가 대상 (pattern, feature, direction)
PATTERNS = {
    "qv_surge_30d": ("f_qv_surge_30d", "high"),
    "qv_surge_7d":  ("f_qv_surge_7d", "high"),
    "bounce_7d_low":("f_bounce_off_7d_low", "high"),
    "ret_3d":       ("f_ret_3d", "high"),
    "ret_7d":       ("f_ret_7d", "high"),
    "atr_pct_14":   ("f_atr_pct_14", "high"),
    "rv_21d":       ("f_rv_21d", "high"),
    "atr_xs_decile":("f_atr_xs_decile", "high"),
}
LABEL = "lab_pump20"
EVAL_ARTIFACT_SCHEMA = "eval_library_audit.v2"
EVAL_INPUT_MANIFEST_SCHEMA = "eval_library_audit_inputs.v2"
EVAL_GENERATOR_SOURCES = (
    "scripts/eval_library_audit_v1.py",
    "ops/artifact_provenance.py",
    "data/database.py",
    "data/market_universe.py",
    "signals/features.py",
    "scripts/univariate_precursor_lift_v1.py",
    "ledger/metrics.py",
    "ledger/portfolio_metrics.py",
)


class EvalArtifactError(ArtifactValidationError):
    """The evaluation CSV/JSON pair is absent, stale, or inconsistent."""


def _completed_label_cutoff(now=None) -> pd.Timestamp:
    """Latest fully closed Upbit 09:00-to-09:00 KST trading date."""
    if now is None:
        now_kst = pd.Timestamp.now(tz="Asia/Seoul")
    else:
        now_kst = pd.Timestamp(now)
        if pd.isna(now_kst):
            raise ValueError("now must be a valid timestamp")
        if now_kst.tzinfo is None:
            now_kst = now_kst.tz_localize("Asia/Seoul")
        else:
            now_kst = now_kst.tz_convert("Asia/Seoul")
    current_session = (now_kst - pd.Timedelta(hours=9)).date()
    return pd.Timestamp(current_session) - pd.Timedelta(days=1)


def build(
    db_path: str | Path = DB,
    *,
    completed_through=None,
):
    cutoff = (
        _completed_label_cutoff()
        if completed_through is None
        else normalize_kst_date(completed_through)
    )
    markets = [
        market
        for market in list_markets(db_path)
        if not is_excluded_signal_market(market)
    ]
    frames = []
    for m in markets:
        df = load_candles(db_path, m)
        if df is None or len(df) <= 70:
            continue
        df = df.sort_values("timestamp").reset_index(drop=True).copy()
        timestamps = pd.to_datetime(df["timestamp"], errors="raise")
        if timestamps.dt.tz is not None:
            timestamps = timestamps.dt.tz_convert(
                "Asia/Seoul"
            ).dt.tz_localize(None)
        df["timestamp"] = timestamps
        df = df[df["timestamp"].dt.normalize() <= cutoff].copy()
        if len(df) <= 70:
            continue
        df = df.reset_index(drop=True)
        df["market"] = m
        feat = build_market_features(df)
        g = df
        oc = pd.DataFrame({"market": m, "timestamp": pd.to_datetime(g["timestamp"]),
                           "o": g["open"].values, "h": g["high"].values,
                           "l": g["low"].values, "cl": g["close"].values,
                           "history_prior_bars": np.arange(len(g), dtype=int)})
        feat = feat.copy()
        feat["timestamp"] = pd.to_datetime(feat["timestamp"])
        eligible = feat.merge(
            oc,
            on=["market", "timestamp"],
            how="left",
        )
        eligible = eligible[eligible["history_prior_bars"] >= 70].copy()
        if not eligible.empty:
            frames.append(eligible)
    if not frames:
        raise RuntimeError("no eligible signal markets with >=70 prior bars")
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = panel["timestamp"].dt.date
    panel = panel.sort_values(["date", "market"]).reset_index(drop=True)
    panel = add_cross_sectional(panel)
    return panel


def attach_regime(panel, db_path: str | Path = DB):
    btc = load_candles(db_path, "KRW-BTC")
    bf = compute_btc_features(btc)
    bf["timestamp"] = pd.to_datetime(bf["timestamp"])
    bf = bf.sort_values("timestamp").reset_index(drop=True)
    bf["regime_raw"] = bf["btc_regime"]
    bf["regime_d1"] = bf["btc_regime"].shift(1)        # day-D 가 보는 regime
    bf["date"] = bf["timestamp"].dt.date
    # ★ LEAK TEST A: regime_d1[t] 가 regime_raw[t-1] 와 같은가
    chk = bf[["date", "regime_raw", "regime_d1"]].copy()
    chk["regime_raw_prev"] = chk["regime_raw"].shift(1)
    valid = chk.dropna(subset=["regime_d1", "regime_raw_prev"])
    leak_ok = bool((valid["regime_d1"] == valid["regime_raw_prev"]).all())
    log.info("[LEAK A] regime_d1[D]==regime_raw[D-1] for all rows: %s (n=%d)", leak_ok, len(valid))
    p = panel.merge(bf[["date", "regime_d1"]], on="date", how="left").rename(columns={"regime_d1": "regime"})
    return p, leak_ok


def leak_shift_test(panel, db_path: str | Path = DB):
    """LEAK TEST B: feature 가 정말 D-1 까지인가.
    f_ret_1d[D] (= shift된 어제 ret_1d) 가 raw close 로 재계산한 D-1 ret_1d 와 일치하는지
    무작위 market 샘플로 직접 대조."""
    import random
    markets = panel["market"].dropna().unique().tolist()
    random.seed(0)
    sample = random.sample(markets, min(20, len(markets)))
    mismatches = 0
    checked = 0
    for m in sample:
        df = load_candles(db_path, m).sort_values("timestamp").reset_index(drop=True)
        c = df["close"]
        # raw ret_1d[t] = c[t]/c[t-1]-1 ; feature f_ret_1d[D] 는 이걸 shift(1) → ret_1d[D-1]
        raw_ret1 = (c / c.shift(1) - 1.0)
        ref = raw_ret1.shift(1)  # D-1 값
        sub = panel[panel["market"] == m][["timestamp", "f_ret_1d"]].copy()
        sub["timestamp"] = pd.to_datetime(sub["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        cmp = pd.DataFrame({"timestamp": df["timestamp"], "ref": ref}).merge(sub, on="timestamp", how="inner")
        cmp = cmp.dropna()
        diff = (cmp["ref"] - cmp["f_ret_1d"]).abs()
        mismatches += int((diff > 1e-6).sum())
        checked += len(cmp)
    log.info("[LEAK B] f_ret_1d == shift(1) of raw ret_1d : mismatches=%d / checked=%d", mismatches, checked)
    return mismatches == 0


def folds(dates, n=5, emb=5):
    if n <= 0:
        raise ValueError("n must be positive")
    if emb < 0:
        raise ValueError("emb must be non-negative")

    ordered_dates = pd.Index(dates)
    if not ordered_dates.is_monotonic_increasing or ordered_dates.has_duplicates:
        raise ValueError("dates must be strictly increasing and unique")

    fs = len(ordered_dates) // (n + 1)
    if fs == 0:
        return []

    out = []
    seen_test_dates: set[object] = set()
    for k in range(1, n + 1):
        tr_end = fs * k
        te_start = tr_end + emb
        te_end = (
            len(ordered_dates)
            if k == n
            else min(fs * (k + 1), len(ordered_dates))
        )
        if te_start >= te_end:
            continue

        train_dates = set(ordered_dates[:tr_end])
        test_dates = set(ordered_dates[te_start:te_end])

        # Expanding-window WF invariants.  These assertions intentionally stay
        # next to boundary construction so a later formula edit cannot silently
        # duplicate OOS observations or erase the embargo.
        if not train_dates.isdisjoint(test_dates):
            raise ValueError("walk-forward train/test overlap")
        if not seen_test_dates.isdisjoint(test_dates):
            raise ValueError("walk-forward OOS windows overlap")
        if te_start - tr_end < emb:
            raise ValueError("walk-forward embargo was not preserved")
        if ordered_dates[tr_end - 1] >= ordered_dates[te_start]:
            raise ValueError("walk-forward boundary is not chronological")

        out.append((train_dates, test_dates))
        seen_test_dates.update(test_dates)
    return out


def oos_selected(panel, feat, direction, regime, decile=0.9):
    """walk-forward OOS 로 선택된 (regime) row 모음 → 진입 후보."""
    cols = [feat, "date", "regime", "o", "h", "l", "cl", LABEL]
    d = panel[cols].dropna(subset=[feat, "regime", "o", "h", "l", "cl"])
    d = d[d["regime"] == regime]
    all_dates = np.sort(panel["date"].unique())
    sel = []
    evaluation_dates = []
    for tr_dates, te_dates in folds(all_dates):
        tr = d[d["date"].isin(tr_dates)]
        te = d[d["date"].isin(te_dates)]
        if len(tr) < 200:
            continue
        # The OOS horizon is determined before observing how often the policy
        # fires or how many rows the requested regime contributes.  Sparse and
        # zero-selection folds are real cash exposure, not a reason to delete
        # an inconvenient test fold ex post.
        evaluation_dates.extend(te_dates)
        if direction == "high":
            cut = tr[feat].quantile(decile)
            s = te[te[feat] >= cut]
        else:
            cut = tr[feat].quantile(1 - decile)
            s = te[te[feat] <= cut]
        if len(s):
            sel.append(s)
    if not evaluation_dates:
        return None
    selected = (
        pd.concat(sel, ignore_index=True)
        if sel
        else d.iloc[0:0].copy()
    )
    # Preserve the OOS observation horizon even when the policy has no position
    # near a fold boundary.  net_metrics uses it to count those dates as cash.
    selected.attrs["evaluation_start"] = min(evaluation_dates)
    selected.attrs["evaluation_end"] = max(evaluation_dates)
    return selected


def _naive_calendar_day(value):
    """Normalize a date-like value to the exchange-local, timezone-naive day."""
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return pd.NaT
    if pd.isna(timestamp):
        return pd.NaT
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("Asia/Seoul").tz_localize(None)
    return timestamp.normalize()


def _daily_path_metrics(daily):
    """Annualized path statistics for an already aligned daily return series."""
    daily = pd.to_numeric(daily, errors="coerce").dropna()
    if daily.empty:
        return {
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": np.nan,
            "mdd": 0.0,
            "cum_return": 0.0,
            "annualized_return": 0.0,
            "annualized_arithmetic_return": 0.0,
        }
    equity = (1.0 + daily).cumprod()
    sharpe = compute_sharpe(daily, PPY)
    sortino = annualized_sortino(daily, periods_per_year=PPY)
    mdd = compute_mdd(equity, initial_equity=1.0)
    cumulative = float(equity.iloc[-1] - 1.0)
    ending_equity = float(equity.iloc[-1])
    annualized_return = (
        float(ending_equity ** (PPY / len(daily)) - 1.0)
        if ending_equity > 0
        else np.nan
    )
    annualized_arithmetic_return = float(daily.mean() * PPY)
    calmar = (
        float(annualized_return / abs(mdd))
        if mdd < 0 and np.isfinite(annualized_return)
        else np.nan
    )
    return {
        "sharpe": float(sharpe),
        "sortino": sortino,
        "calmar": calmar,
        "mdd": float(mdd),
        "cum_return": cumulative,
        "annualized_return": annualized_return,
        "annualized_arithmetic_return": annualized_arithmetic_return,
    }


def net_metrics(rows, exit_mode):
    """exit_mode: 'sl_first'(비관) | 'tp_first'(낙관) | 'tp_eod'(SL없음).
    포지션일은 equal-weight, 무포지션 calendar day는 0인 portfolio 지표."""
    if rows is None:
        return None
    o = pd.to_numeric(rows["o"], errors="coerce").to_numpy(float)
    h = pd.to_numeric(rows["h"], errors="coerce").to_numpy(float)
    low = pd.to_numeric(rows["l"], errors="coerce").to_numpy(float)
    cl = pd.to_numeric(rows["cl"], errors="coerce").to_numpy(float)
    observed_dates = pd.Series(
        [_naive_calendar_day(value) for value in rows["date"]],
        index=rows.index,
        dtype="datetime64[ns]",
    )
    tp_px = o * (1 + TP)
    sl_px = o * (1 - SL)
    hit_tp = h >= tp_px
    hit_sl = low <= sl_px
    o2c = cl / o - 1.0
    if exit_mode == "sl_first":
        gross = np.where(hit_sl, -SL, np.where(hit_tp, TP, o2c))
    elif exit_mode == "tp_first":
        gross = np.where(hit_tp, TP, np.where(hit_sl, -SL, o2c))
    elif exit_mode == "tp_eod":
        gross = np.where(hit_tp, TP, o2c)
    else:
        raise ValueError(exit_mode)
    valid = (
        np.isfinite(o)
        & (o > 0)
        & np.isfinite(gross)
        & observed_dates.notna().to_numpy()
    )
    gross = gross[valid]
    net = gross - RT
    dates = observed_dates.to_numpy()[valid]
    td = pd.DataFrame({"date": dates, "net": net})
    trade_daily = td.groupby("date")["net"].mean().sort_index()
    observed = observed_dates.dropna()
    default_start = observed.min() if len(observed) else None
    default_end = observed.max() if len(observed) else None
    calendar_start = _naive_calendar_day(
        rows.attrs.get("evaluation_start", default_start)
    )
    calendar_end = _naive_calendar_day(
        rows.attrs.get("evaluation_end", default_end)
    )
    if pd.isna(calendar_start) or pd.isna(calendar_end):
        if len(rows) == 0:
            return None
        raise ValueError("evaluation calendar bounds must be valid dates")
    if len(observed) and (
        calendar_start > observed.min() or calendar_end < observed.max()
    ):
        raise ValueError("evaluation calendar must contain every observed row")
    if calendar_start > calendar_end:
        raise ValueError("evaluation_start must not be after evaluation_end")

    calendar_index = pd.date_range(calendar_start, calendar_end, freq="D")
    if len(calendar_index) < 3:
        return None
    daily = trade_daily.reindex(calendar_index, fill_value=0.0)
    portfolio = _daily_path_metrics(daily)
    trade_day_only = _daily_path_metrics(trade_daily)

    return {
        "n_trades": int(len(td)),
        "n_days": int(len(daily)),
        "n_calendar_days": int(len(daily)),
        "n_trade_days": int(len(trade_daily)),
        "n_zero_position_days": int(len(daily) - len(trade_daily)),
        "calendar_start": calendar_start.date().isoformat(),
        "calendar_end": calendar_end.date().isoformat(),
        "net_mean_per_trade": float(net.mean()) if len(net) else np.nan,
        "hit_net": float((net > 0).mean()) if len(net) else np.nan,
        "tp_rate": float(hit_tp[valid].mean()) if valid.any() else np.nan,
        "sl_rate": float(hit_sl[valid].mean()) if valid.any() else np.nan,
        **portfolio,
        "trade_day_only_sharpe": trade_day_only["sharpe"],
        "trade_day_only_sortino": trade_day_only["sortino"],
        "trade_day_only_calmar": trade_day_only["calmar"],
        "trade_day_only_mdd": trade_day_only["mdd"],
        "trade_day_only_cum_return": trade_day_only["cum_return"],
        "trade_day_only_annualized_return": trade_day_only["annualized_return"],
        "trade_day_only_annualized_arithmetic_return": trade_day_only[
            "annualized_arithmetic_return"
        ],
    }


def psr_dsr(sharpe_ann, n_obs, trials):
    """PSR(SR*=0) 과 selection-deflated DSR (Bailey & LdP 2014, skew/kurt=정규 가정 간이)."""
    if not np.isfinite(sharpe_ann) or n_obs < 3:
        return np.nan, np.nan
    sr = sharpe_ann / np.sqrt(PPY)  # per-period
    # PSR vs 0
    psr = stats.norm.cdf(sr * np.sqrt(n_obs - 1))
    # DSR: deflated benchmark SR* from trials
    if trials > 1:
        emax = (1 - np.euler_gamma) * stats.norm.ppf(1 - 1.0 / trials) + \
               np.euler_gamma * stats.norm.ppf(1 - 1.0 / (trials * np.e))
        # variance of SR estimates across trials ~ unknown; 보수적으로 1/(n-1) 사용
        sr_star = emax * np.sqrt(1.0 / (n_obs - 1))
        dsr = stats.norm.cdf((sr - sr_star) * np.sqrt(n_obs - 1))
    else:
        dsr = psr
    return float(psr), float(dsr)


def _selection_trials_total() -> int:
    return (46 + 46) + (13 * 2 * 4 + 12) + (8 * 4 * 3) + (7 * 5 + 32)


def _eval_source_state(db_path: str | Path) -> dict[str, Any]:
    database = resolve_identity_path(str(db_path), root=ROOT)
    if not database.is_file():
        raise FileNotFoundError(f"evaluation candle DB missing: {database}")
    captured_files = file_set_identity(
        {
            **{
                f"generator:{relative}": ROOT / relative
                for relative in EVAL_GENERATOR_SOURCES
            },
            "daily_candle_db:main": database,
            "daily_candle_db:wal": Path(f"{database}-wal"),
            "daily_candle_db:journal": Path(f"{database}-journal"),
        },
        root=ROOT,
    )
    generator_sources = {
        relative: captured_files[f"generator:{relative}"]
        for relative in EVAL_GENERATOR_SOURCES
    }
    missing_sources = [
        name
        for name, identity in generator_sources.items()
        if not identity["exists"]
    ]
    if missing_sources:
        raise FileNotFoundError(
            f"eval generator source missing: {missing_sources}"
        )
    return {
        "daily_candle_db": {
            name: captured_files[f"daily_candle_db:{name}"]
            for name in ("main", "wal", "journal")
        },
        "generator_sources": generator_sources,
    }


def _eval_contract() -> dict[str, Any]:
    return {
        "patterns": {
            name: list(value) for name, value in PATTERNS.items()
        },
        "regimes": REGIMES,
        "label": LABEL,
        "round_trip_cost": RT,
        "take_profit": TP,
        "stop_loss": SL,
        "periods_per_year": PPY,
        "walk_forward": {"folds": 5, "embargo_days": 5},
        "selection_trials_total": _selection_trials_total(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy_version,
        },
    }


def build_eval_input_manifest(
    asof: str | date | pd.Timestamp,
    *,
    db_path: str | Path = DB,
    source_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact_asof = str(normalize_kst_date(asof).date())
    state = source_state or _eval_source_state(db_path)
    return with_manifest_digest(
        {
            "schema": EVAL_INPUT_MANIFEST_SCHEMA,
            "asof": artifact_asof,
            "inputs": state,
            "contract": _eval_contract(),
        }
    )


def _eval_run_identity(payload: dict[str, Any]) -> str:
    body = {
        key: value
        for key, value in payload.items()
        if key not in {"generated_at_utc", "run_id", "payload_sha256"}
    }
    return "eval-" + sha256_bytes(canonical_json_bytes(body))[:32]


def _eval_csv_bytes(payload: dict[str, Any]) -> bytes:
    columns = payload.get("columns")
    rows = payload.get("rows")
    if not isinstance(columns, list) or not isinstance(rows, list):
        raise EvalArtifactError("eval artifact rows/columns are invalid")
    frame = pd.DataFrame(rows).reindex(columns=columns)
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _validate_eval_payload(payload: dict[str, Any]) -> None:
    if payload.get("schema") != EVAL_ARTIFACT_SCHEMA:
        raise EvalArtifactError("unsupported eval artifact schema")
    try:
        artifact_asof = date.fromisoformat(str(payload["asof"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise EvalArtifactError("eval artifact asof is invalid") from exc
    try:
        generated = datetime.fromisoformat(
            str(payload["generated_at_utc"])
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise EvalArtifactError(
            "eval artifact generated_at_utc is invalid"
        ) from exc
    if generated.tzinfo is None:
        raise EvalArtifactError(
            "eval artifact generated_at_utc must be timezone-aware"
        )
    manifest = payload.get("input_manifest")
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != EVAL_INPUT_MANIFEST_SCHEMA
        or manifest.get("asof") != artifact_asof.isoformat()
        or not manifest_digest_matches(manifest)
    ):
        raise EvalArtifactError("eval input manifest is invalid")
    columns = payload.get("columns")
    if (
        not isinstance(columns, list)
        or not columns
        or any(not isinstance(column, str) or not column for column in columns)
        or len(columns) != len(set(columns))
    ):
        raise EvalArtifactError("eval columns are invalid")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise EvalArtifactError("eval rows must be a list")
    expected_columns = set(columns)
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != expected_columns:
            raise EvalArtifactError(
                f"eval row schema mismatch at index {index}"
            )
    leak_summary = payload.get("leak_summary")
    if (
        not isinstance(leak_summary, dict)
        or not {"regime_d1_ok", "feature_shift_ok", "label_match"}
        <= set(leak_summary)
    ):
        raise EvalArtifactError("eval leak summary is incomplete")
    if (
        not isinstance(leak_summary["regime_d1_ok"], bool)
        or not isinstance(leak_summary["feature_shift_ok"], bool)
    ):
        raise EvalArtifactError("eval leak checks must be booleans")
    label_match = leak_summary["label_match"]
    if (
        isinstance(label_match, bool)
        or not isinstance(label_match, (int, float))
        or not math.isfinite(float(label_match))
        or not 0.0 <= float(label_match) <= 1.0
    ):
        raise EvalArtifactError("eval label_match is invalid")
    if payload.get("selection_trials_total") != _selection_trials_total():
        raise EvalArtifactError("eval selection trial contract mismatch")
    csv_sha256 = payload.get("csv_sha256")
    if (
        not isinstance(csv_sha256, str)
        or csv_sha256 != sha256_bytes(_eval_csv_bytes(payload))
    ):
        raise EvalArtifactError("eval CSV checksum mismatch")
    if payload.get("run_id") != _eval_run_identity(payload):
        raise EvalArtifactError("eval run_id mismatch")
    if payload.get("payload_sha256") != payload_digest(payload):
        raise EvalArtifactError("eval payload checksum mismatch")


def build_eval_artifact(
    frame: pd.DataFrame,
    *,
    asof: str | date | pd.Timestamp,
    leak_summary: dict[str, Any],
    input_manifest: dict[str, Any],
) -> dict[str, Any]:
    columns = [str(column) for column in frame.columns]
    rows = json.loads(
        frame.to_json(
            orient="records",
            date_format="iso",
            double_precision=15,
        )
    )
    payload: dict[str, Any] = {
        "schema": EVAL_ARTIFACT_SCHEMA,
        "asof": str(normalize_kst_date(asof).date()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "input_manifest": input_manifest,
        "columns": columns,
        "rows": rows,
        "leak_summary": leak_summary,
        "selection_trials_total": _selection_trials_total(),
    }
    payload["csv_sha256"] = sha256_bytes(_eval_csv_bytes(payload))
    payload["run_id"] = _eval_run_identity(payload)
    payload["payload_sha256"] = payload_digest(payload)
    _validate_eval_payload(payload)
    return payload


@contextmanager
def _artifact_lock(path: Path, *, shared: bool) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(
        path.with_name(f".{path.name}.lock"),
        shared=shared,
    ):
        yield


def load_eval_library_artifact(
    csv_path: str | Path = OUT_CSV,
    *,
    json_path: str | Path | None = None,
    asof: str | pd.Timestamp | None = None,
    require_exact_asof: bool = False,
    require_current: bool = True,
    db_path: str | Path = DB,
) -> dict[str, Any]:
    """Validate the sidecar, source manifest, payload digest, and exact CSV."""
    csv_path = Path(csv_path)
    json_path = (
        Path(json_path)
        if json_path is not None
        else csv_path.with_suffix(".json")
    )
    with _artifact_lock(json_path, shared=True):
        if not json_path.is_file():
            raise EvalArtifactError(
                f"eval JSON sidecar is missing: {json_path}"
            )
        try:
            payload = strict_json_object(json_path)
            _validate_eval_payload(payload)
            csv_bytes = csv_path.read_bytes()
        except EvalArtifactError:
            raise
        except ArtifactValidationError as exc:
            raise EvalArtifactError(str(exc)) from exc
        except OSError as exc:
            raise EvalArtifactError(
                "eval CSV/JSON sidecar pair is incomplete"
            ) from exc
        if csv_bytes != _eval_csv_bytes(payload):
            raise EvalArtifactError("eval CSV/JSON payload mismatch")
        if asof is not None:
            cutoff = normalize_kst_date(asof).date()
            artifact_asof = date.fromisoformat(payload["asof"])
            if artifact_asof > cutoff or (
                require_exact_asof and artifact_asof != cutoff
            ):
                raise EvalArtifactError(
                    "eval artifact asof does not satisfy requested cutoff"
                )
        if require_current:
            try:
                current = build_eval_input_manifest(
                    payload["asof"],
                    db_path=db_path,
                )
            except (OSError, ArtifactSourceChangedError) as exc:
                raise EvalArtifactError(
                    "eval inputs required for current validation are missing"
                ) from exc
            if payload["input_manifest"] != current:
                raise EvalArtifactError(
                    "eval input DB or generator semantics changed"
                )
    return payload


def run(
    *,
    db_path: str | Path = DB,
    output_csv: str | Path = OUT_CSV,
    output_json: str | Path = OUT_JSON,
    completed_through=None,
) -> dict[str, Any]:
    source_state = _eval_source_state(db_path)
    panel = build(db_path, completed_through=completed_through)
    panel, leak_a = attach_regime(panel, db_path)
    leak_b = leak_shift_test(panel, db_path)
    # 라벨 미래성 sanity: lab_pump20 == (h/o-1>=0.20)
    lab_chk = ((panel["h"] / panel["o"] - 1.0 >= 0.20).astype(float) == panel[LABEL]).mean()
    log.info("[LEAK C] label==recompute(h/o-1>=.2): match_frac=%.4f (1.0=ok, 라벨은 미래 day-D)", lab_chk)

    rdist = panel["regime"].value_counts(dropna=False)
    log.info("regime dist (D-1):\n%s", rdist.to_string())
    for rg in REGIMES:
        sub = panel[panel["regime"] == rg]
        log.info("  base %s pump20=%.4f n=%d", rg, sub[LABEL].mean(), len(sub))

    # selection: 4각도 총 시도 추정
    trials_total = _selection_trials_total()
    log.info("SELECTION trials_total(approx 4각도)= %d", trials_total)

    rows = []
    for pname, (feat, direction) in PATTERNS.items():
        for rg in REGIMES:
            sel = oos_selected(panel, feat, direction, rg)
            if sel is None:
                continue
            for mode in ["sl_first", "tp_eod", "tp_first"]:
                m = net_metrics(sel, mode)
                if m is None:
                    continue
                psr, dsr = psr_dsr(m["sharpe"], m["n_days"], trials_total)
                rows.append({"pattern": pname, "regime": rg, "exit": mode,
                             **m, "psr": psr, "dsr": dsr})
    res = pd.DataFrame(rows)

    # 요약: regime별 exit별 net Sharpe
    log.info("\n===== NET SHARPE by regime/exit (pattern-mean) =====")
    piv = res.groupby(["regime", "exit"]).agg(
        n=("sharpe", "size"), sharpe=("sharpe", "mean"),
        net_pt=("net_mean_per_trade", "mean"), hit=("hit_net", "mean"),
        mdd=("mdd", "mean"), cum=("cum_return", "mean")).reset_index()
    for _, r in piv.iterrows():
        log.info("  %-14s %-9s sharpe=%+.2f net/trade=%+.4f hit=%.2f mdd=%.3f cum=%+.3f (n_pat=%d)",
                 r["regime"], r["exit"], r["sharpe"], r["net_pt"], r["hit"], r["mdd"], r["cum"], int(r["n"]))

    # 가장 net 양수에 가까운 (pattern×regime×exit) top
    log.info("\n===== TOP net_mean_per_trade (sl_first 비관) =====")
    slf = res[res["exit"] == "sl_first"].sort_values("net_mean_per_trade", ascending=False)
    for _, r in slf.head(10).iterrows():
        log.info("  %-14s %-14s net/tr=%+.4f sharpe=%+.2f hit=%.2f mdd=%.3f cum=%+.3f n=%d dsr=%.3f",
                 r["pattern"], r["regime"], r["net_mean_per_trade"], r["sharpe"], r["hit_net"],
                 r["mdd"], r["cum_return"], int(r["n_trades"]), r["dsr"])
    log.info("\n===== TOP net (tp_eod = SL 없음, 시간손절만) =====")
    te = res[res["exit"] == "tp_eod"].sort_values("net_mean_per_trade", ascending=False)
    for _, r in te.head(10).iterrows():
        log.info("  %-14s %-14s net/tr=%+.4f sharpe=%+.2f hit=%.2f mdd=%.3f cum=%+.3f n=%d dsr=%.3f",
                 r["pattern"], r["regime"], r["net_mean_per_trade"], r["sharpe"], r["hit_net"],
                 r["mdd"], r["cum_return"], int(r["n_trades"]), r["dsr"])
    # artifact 크기: sl_first vs tp_first spread (intrabar 순서 불확실성 폭)
    log.info("\n===== EXIT ARTIFACT spread (tp_first - sl_first) by regime =====")
    for rg in REGIMES:
        a = res[(res["regime"] == rg) & (res["exit"] == "sl_first")]["net_mean_per_trade"].mean()
        b = res[(res["regime"] == rg) & (res["exit"] == "tp_first")]["net_mean_per_trade"].mean()
        log.info("  %-14s sl_first=%+.4f tp_first=%+.4f spread=%.4f", rg, a, b, b - a)

    leak_summary = {
        "regime_d1_ok": bool(leak_a),
        "feature_shift_ok": bool(leak_b),
        "label_match": float(lab_chk),
    }
    artifact_asof = pd.Timestamp(max(panel["date"])).date().isoformat()
    if _eval_source_state(db_path) != source_state:
        raise RuntimeError(
            "eval input DB or generator sources changed during computation"
        )
    manifest = build_eval_input_manifest(
        artifact_asof,
        db_path=db_path,
        source_state=source_state,
    )
    payload = build_eval_artifact(
        res,
        asof=artifact_asof,
        leak_summary=leak_summary,
        input_manifest=manifest,
    )
    csv_path = Path(output_csv)
    json_path = Path(output_json)
    with _artifact_lock(json_path, shared=False):
        atomic_write_bytes(csv_path, _eval_csv_bytes(payload))
        atomic_write_json(json_path, payload)
        if _eval_source_state(db_path) != source_state:
            raise RuntimeError(
                "eval input DB or generator sources changed during publication"
            )
    log.info(
        "wrote %s + %s (%d rows)",
        csv_path,
        json_path,
        len(res),
    )
    print("\nLEAK_SUMMARY", json.dumps(leak_summary))
    return payload


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild the strict Phase-1 evaluation-library audit",
    )
    parser.add_argument("--db", default=str(DB))
    parser.add_argument("--out-csv", default=str(OUT_CSV))
    parser.add_argument("--out-json", default=str(OUT_JSON))
    parser.add_argument(
        "--through-date",
        help=(
            "inclusive completed Upbit trading date; default is the latest "
            "fully closed KST 09:00 session"
        ),
    )
    args = parser.parse_args()
    run(
        db_path=args.db,
        output_csv=args.out_csv,
        output_json=args.out_json,
        completed_through=args.through_date,
    )


if __name__ == "__main__":
    main()
