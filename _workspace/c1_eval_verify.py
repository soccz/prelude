"""Independent adversarial re-verification of C1_sus_net versus R1.

The persisted pick rows are treated as untrusted input.  This script validates
their exact common cohort and cost contract before recomputing metrics, a
circular moving-block bootstrap, and PSR/DSR.  It writes nothing.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
PICKS = ROOT / "output/cc_sustained_label_picks_v1.csv"
COMPARE = ROOT / "output/cc_sustained_label_compare_v1.csv"
COVERAGE = ROOT / "output/cc_sustained_label_coverage_v1.json"
SOURCE = ROOT / "scripts/cc_sustained_label_v1.py"
COST = 0.0015
DEEP = -0.05
SEED = 42
DRAWS = 10_000
BLOCK = 5
EXPECTED_POLICIES = {
    "R1_baseline",
    "C1_sus3",
    "C1_sus3_rr",
    "C1_sus5",
    "C1_sus5_rr",
    "C1_sus_net",
    "C1_sus_net_rr",
    "C1_fwd1",
    "C1_fwd1_rr",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(frame: pd.DataFrame, columns: set[str], *, name: str) -> None:
    for column in sorted(columns):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name}: {column} contains missing/non-finite values")


def _metrics(frame: pd.DataFrame, label_col: str | None = None) -> dict:
    if frame.empty:
        raise ValueError("metrics: empty policy cohort")
    net = frame["net"].to_numpy(float)
    daily = frame.groupby("date", sort=True)["net"].mean()
    equity = (1.0 + daily).cumprod()
    peak = equity.cummax()
    daily_mean = float(daily.mean())
    daily_std = float(daily.std())
    downside_std = float(daily[daily < 0].std()) if (daily < 0).any() else np.nan
    # The original report divided a trade-level mean by a daily-level standard
    # deviation.  Preserve that value only as an explicitly named reconciliation
    # diagnostic; the correctly dimensioned Sharpe uses daily mean and std.
    legacy_mixed_sharpe = (
        float(net.mean() / daily_std * np.sqrt(365))
        if daily_std > 0
        else np.nan
    )
    output = {
        "n": int(len(frame)),
        "n_days": int(len(daily)),
        "net_mean": float(net.mean()),
        "daily_net_mean": daily_mean,
        "net_median": float(np.median(net)),
        "hit": float((net > 0).mean()),
        "sharpe": (
            float(daily_mean / daily_std * np.sqrt(365))
            if daily_std > 0
            else np.nan
        ),
        "legacy_mixed_unit_sharpe": legacy_mixed_sharpe,
        "sortino": (
            float(daily_mean / downside_std * np.sqrt(365))
            if np.isfinite(downside_std) and downside_std > 0
            else np.nan
        ),
        "mdd": float(((equity - peak) / peak).min()),
        "cum": float(equity.iloc[-1] - 1.0),
        "deep_noSL": float(((frame["eod_ret"] - COST) <= DEEP).mean()),
        "pct_sl": float(frame["outcome"].eq("sl").mean()),
        "worst": float(net.min()),
    }
    if label_col:
        label = pd.to_numeric(frame[label_col], errors="coerce")
        if label.isna().any() or not label.isin({0, 1}).all():
            raise ValueError(f"{label_col}: missing or non-binary labels")
        output["prec_self"] = float(label.mean())
    return output


def _sharpe(series: pd.Series) -> float:
    standard_deviation = float(series.std())
    if not np.isfinite(standard_deviation) or standard_deviation <= 0:
        return np.nan
    return float(series.mean() / standard_deviation)


def _bootstrap_stat(sample: pd.DataFrame) -> tuple[float, float, float, float]:
    return (
        float(sample["C1"].mean() - sample["R1"].mean()),
        float(sample["C1d"].mean() - sample["R1d"].mean()),
        float(sample["C1h"].mean() - sample["R1h"].mean()),
        float(_sharpe(sample["C1"]) - _sharpe(sample["R1"])),
    )


def _psr(
    sr_hat: float,
    sr_star: float,
    observations: int,
    skew: float,
    kurtosis: float,
) -> float:
    denominator_squared = (
        1.0 - skew * sr_hat + (kurtosis - 1.0) * sr_hat**2 / 4.0
    )
    if observations < 2 or denominator_squared <= 0:
        raise ValueError("PSR denominator is non-positive")
    z_score = (
        (sr_hat - sr_star)
        * np.sqrt(observations - 1)
        / np.sqrt(denominator_squared)
    )
    return float(stats.norm.cdf(z_score))


def _validate_inputs(
    frame: pd.DataFrame,
    compare: pd.DataFrame,
    coverage: dict,
) -> dict:
    required = {
        "date", "market", "fold", "policy", "net", "outcome",
        "eod_ret", "up_high_ret", "down_low_ret", "pump20_hit",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"picks: missing columns: {missing}")
    if frame.empty:
        raise ValueError("picks: empty artifact")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    if frame["date"].isna().any():
        raise ValueError("picks: invalid/null dates")
    if set(frame["policy"]) != EXPECTED_POLICIES:
        raise ValueError("picks: unexpected or missing policies")
    duplicate = frame.duplicated(["date", "policy", "market"], keep=False)
    if duplicate.any():
        raise ValueError("picks: duplicate date/policy/market rows")
    _finite(
        frame,
        {"fold", "net", "eod_ret", "up_high_ret", "down_low_ret", "pump20_hit"},
        name="picks",
    )
    if not frame["pump20_hit"].isin({0, 1}).all():
        raise ValueError("picks: pump20_hit is not binary")
    if not set(frame["outcome"]).issubset({"tp", "sl", "eod"}):
        raise ValueError("picks: invalid path outcomes")
    barrier = frame["outcome"].isin({"tp", "sl"})
    expected_barrier_net = np.where(
        frame.loc[barrier, "outcome"].eq("tp"),
        0.05 - COST,
        -0.03 - COST,
    )
    cost_error = float(
        np.max(
            np.abs(
                frame.loc[barrier, "net"].to_numpy(float)
                - expected_barrier_net
            )
        )
    )
    if cost_error > 1e-12:
        raise ValueError(f"picks: TP/SL round-trip cost mismatch: {cost_error}")
    eod = frame["outcome"].eq("eod")
    eod_proxy_error = float(
        np.max(
            np.abs(
                frame.loc[eod, "net"].to_numpy(float)
                - (frame.loc[eod, "eod_ret"].to_numpy(float) - COST)
            )
        )
    )

    counts = frame.groupby(["date", "policy"]).size().unstack("policy")
    if counts.isna().any().any():
        raise ValueError("picks: policies do not share a common date cohort")
    if not counts.nunique(axis=1).eq(1).all():
        raise ValueError("picks: policy pick counts differ within dates")
    if not counts.isin({1, 3}).all().all():
        raise ValueError("picks: unexpected per-policy daily pick count")
    if frame.groupby("date")["fold"].nunique().max() != 1:
        raise ValueError("picks: a date belongs to multiple folds")
    fold_dates = (
        frame[["fold", "date"]]
        .drop_duplicates()
        .sort_values(["fold", "date"])
        .groupby("fold")["date"]
        .agg(["min", "max"])
    )
    if list(map(int, fold_dates.index)) != list(range(6)):
        raise ValueError("picks: expected folds 0..5")
    previous_end: pd.Timestamp | None = None
    for row in fold_dates.itertuples():
        if previous_end is not None and row.min <= previous_end:
            raise ValueError("picks: fold test windows overlap or are unordered")
        previous_end = row.max

    if set(compare["policy"]) != EXPECTED_POLICIES:
        raise ValueError("compare: unexpected or missing policies")
    if compare["policy"].duplicated().any():
        raise ValueError("compare: duplicate policy rows")
    compare_required = {
        "net_mean", "sharpe", "mdd", "hit", "deep_loss_freq_noSL",
    }
    missing_compare = sorted(compare_required.difference(compare.columns))
    if missing_compare:
        raise ValueError(f"compare: missing metric columns: {missing_compare}")
    _finite(compare, compare_required, name="compare")
    expected_window = coverage.get("oos_window")
    if not isinstance(expected_window, list) or len(expected_window) != 2:
        raise ValueError("coverage: malformed OOS window")
    if frame["date"].min() != pd.Timestamp(expected_window[0]):
        raise ValueError("picks: start date disagrees with coverage")
    if frame["date"].max() != pd.Timestamp(expected_window[1]):
        raise ValueError("picks: end date disagrees with coverage")
    if int(coverage.get("n_folds", -1)) != 6:
        raise ValueError("coverage: unexpected fold count")
    if int(coverage.get("embargo", -1)) < 5:
        raise ValueError("coverage: embargo shorter than five dates")
    completed_cutoff = (
        datetime.now(ZoneInfo("Asia/Seoul")).date() - pd.Timedelta(days=1)
    )
    if frame["date"].max().date() > completed_cutoff:
        raise ValueError("picks: contains an incomplete/future label date")
    return {
        "tp_sl_cost_once_max_abs_error": cost_error,
        "eod_cost_verification": (
            "UNPROVABLE_FROM_PICKS: source omitted realized 15m eod_net; "
            "persisted eod_ret is a different D1 proxy"
        ),
        "eod_net_minus_d1_proxy_max_abs_delta": eod_proxy_error,
        "common_dates": int(len(counts)),
        "daily_pick_counts": sorted(map(int, np.unique(counts.to_numpy()))),
        "fold_test_windows_nonoverlapping": True,
        "declared_embargo_dates": int(coverage["embargo"]),
        "train_test_overlap_verification": (
            "UNPROVABLE_FROM_ROW_ARTIFACT: training date ranges were not "
            "persisted; source configuration and disjoint OOS fold dates only"
        ),
        "completed_label_cutoff": frame["date"].max().strftime("%Y-%m-%d"),
        "population_scope": {
            "audited_rows": "all persisted picks for all nine ranking policies",
            "common_cohort": (
                "every persisted date, with identical per-policy daily counts"
            ),
            "evidence_boundary": (
                "the full pre-selection candidate universe and training ranges "
                "were not persisted, so selection reconstruction is unprovable"
            ),
        },
        "output_contract": (
            "read-only and stdout-only; partial progress output is invalid "
            "unless the final provenance document is emitted and exit is zero"
        ),
    }


def main() -> None:
    paths = (PICKS, COMPARE, COVERAGE, SOURCE)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing audit inputs: {missing}")
    frame = pd.read_csv(PICKS)
    compare = pd.read_csv(COMPARE)
    coverage = json.loads(COVERAGE.read_text())
    contracts = _validate_inputs(frame, compare, coverage)
    r1 = frame[frame["policy"].eq("R1_baseline")].copy()
    c1 = frame[frame["policy"].eq("C1_sus_net")].copy()

    metrics_r1 = _metrics(r1)
    metrics_c1 = _metrics(c1, "lab_sus_net")
    print("=== INDEPENDENT RE-AGGREGATION (from picks dump) ===")
    keys = [
        "n", "n_days", "net_mean", "daily_net_mean", "sharpe",
        "legacy_mixed_unit_sharpe", "sortino", "mdd", "cum", "hit",
        "deep_noSL", "pct_sl", "prec_self",
    ]
    for key in keys:
        r1_value = metrics_r1.get(key)
        c1_value = metrics_c1.get(key)
        r1_text = f"{r1_value:>12.6f}" if r1_value is not None else f"{'--':>12s}"
        c1_text = f"{c1_value:>12.6f}" if c1_value is not None else f"{'--':>12s}"
        print(f"  {key:25s} R1={r1_text} C1_sus_net={c1_text}")

    reported_r1 = compare[compare["policy"].eq("R1_baseline")].iloc[0]
    reported_c1 = compare[compare["policy"].eq("C1_sus_net")].iloc[0]
    print("\n=== reconcile vs compare CSV (delta should be ~0) ===")
    reconcile = [
        ("net_mean", "net_mean", "net_mean"),
        ("legacy_sharpe", "legacy_mixed_unit_sharpe", "sharpe"),
        ("mdd", "mdd", "mdd"),
        ("hit", "hit", "hit"),
        ("deep_noSL", "deep_noSL", "deep_loss_freq_noSL"),
    ]
    for label, metric_key, reported_key in reconcile:
        print(
            f"  {label:16s} "
            f"R1 reagg-csv={metrics_r1[metric_key] - reported_r1[reported_key]:+.2e} "
            f"C1 reagg-csv={metrics_c1[metric_key] - reported_c1[reported_key]:+.2e}"
        )

    daily = pd.concat(
        [
            r1.groupby("date")["net"].mean().rename("R1"),
            c1.groupby("date")["net"].mean().rename("C1"),
            r1.assign(d=(r1["eod_ret"] - COST <= DEEP))
            .groupby("date")["d"].mean().rename("R1d"),
            c1.assign(d=(c1["eod_ret"] - COST <= DEEP))
            .groupby("date")["d"].mean().rename("C1d"),
            r1.assign(h=r1["net"] > 0)
            .groupby("date")["h"].mean().rename("R1h"),
            c1.assign(h=c1["net"] > 0)
            .groupby("date")["h"].mean().rename("C1h"),
        ],
        axis=1,
    )
    if daily.isna().any().any() or len(daily) != contracts["common_dates"]:
        raise ValueError("paired daily bootstrap cohort is incomplete")
    observed = _bootstrap_stat(daily)
    observations = len(daily)
    blocks = int(np.ceil(observations / BLOCK))
    rng = np.random.default_rng(SEED)
    bootstrap = np.empty((DRAWS, 4))
    block_offsets = np.arange(BLOCK)
    for draw in range(DRAWS):
        starts = rng.integers(0, observations, size=blocks)
        indices = ((starts[:, None] + block_offsets) % observations).ravel()[
            :observations
        ]
        bootstrap[draw] = _bootstrap_stat(daily.iloc[indices])
    if not np.isfinite(bootstrap).all():
        raise ValueError("block bootstrap produced non-finite statistics")
    print(
        f"\n=== CIRCULAR BLOCK bootstrap "
        f"(paired trading-day n={observations}, B={DRAWS}, block={BLOCK}) ==="
    )
    names = [
        "Δnet (C1-R1)",
        "ΔdeepNoSL (C1-R1)",
        "Δhit (C1-R1)",
        "Δdaily Sharpe (C1-R1)",
    ]
    for index, name in enumerate(names):
        low, high = np.quantile(bootstrap[:, index], [0.025, 0.975])
        probability_positive = float((bootstrap[:, index] > 0).mean())
        excludes_zero = "YES" if low > 0 or high < 0 else "no"
        print(
            f"  {name:27s} observed={observed[index]:+.5f} "
            f"CI95=[{low:+.5f},{high:+.5f}] "
            f"P(bootstrap>0)={probability_positive:.3f} excludes0={excludes_zero}"
        )

    daily_c1 = daily["C1"]
    sr = _sharpe(daily_c1)
    skew = float(stats.skew(daily_c1))
    kurtosis = float(stats.kurtosis(daily_c1, fisher=False))
    psr_zero = _psr(sr, 0.0, observations, skew, kurtosis)
    policy_daily = frame.groupby(["policy", "date"])["net"].mean().reset_index()
    sr_by_policy = policy_daily.groupby("policy")["net"].apply(_sharpe)
    if len(sr_by_policy) != 9 or not np.isfinite(sr_by_policy).all():
        raise ValueError("DSR: expected nine finite policy Sharpes")
    sigma_sr = float(sr_by_policy.std())
    trials = 9
    euler_gamma = 0.5772156649015329
    # Bailey/Lopez de Prado expected maximum standard-normal approximation.
    expected_max_z = (
        (1.0 - euler_gamma) * stats.norm.ppf(1.0 - 1.0 / trials)
        + euler_gamma * stats.norm.ppf(1.0 - 1.0 / (trials * np.e))
    )
    sr_star = float(sigma_sr * expected_max_z)
    dsr = _psr(sr, sr_star, observations, skew, kurtosis)
    print(f"\n=== DSR/PSR (daily net, T={observations}, trials={trials}) ===")
    print(f"  daily SR={sr:+.4f} skew={skew:+.3f} kurt={kurtosis:.3f}")
    print(f"  PSR(SR*=0)={psr_zero:.4f}")
    print(
        f"  sigma_SR={sigma_sr:.4f} E[max]z={expected_max_z:.3f} "
        f"SR*_deflated={sr_star:+.4f} DSR={dsr:.4f}"
    )

    r1_keys = set(zip(r1["date"], r1["market"]))
    c1_keys = set(zip(c1["date"], c1["market"]))
    overlap = c1_keys & r1_keys
    new_entries = c1_keys - r1_keys
    dropped = r1_keys - c1_keys
    if not c1_keys or len(c1_keys) != len(c1):
        raise ValueError("C1 key set is empty or non-unique")
    print("\n=== DEGENERACY: substituted vs R1 picks ===")
    print(
        f"  C1 picks={len(c1_keys)} overlap={len(overlap)} "
        f"({len(overlap) / len(c1_keys) * 100:.1f}%) "
        f"new={len(new_entries)} ({len(new_entries) / len(c1_keys) * 100:.1f}%)"
    )
    for tag, subset in (
        ("R1 ALL", r1),
        ("C1 ALL", c1),
        ("C1 NEW entries", c1[c1.set_index(["date", "market"]).index.isin(new_entries)]),
        ("C1 overlap w/R1", c1[c1.set_index(["date", "market"]).index.isin(overlap)]),
        ("R1 DROPPED", r1[r1.set_index(["date", "market"]).index.isin(dropped)]),
    ):
        if subset.empty:
            raise ValueError(f"degeneracy cohort is empty: {tag}")
        print(
            f"  [{tag:18s}] n={len(subset):4d} "
            f"up_high={subset['up_high_ret'].mean():+.4f} "
            f"down_low={subset['down_low_ret'].mean():+.4f} "
            f"|range|={(subset['up_high_ret'] - subset['down_low_ret']).mean():.4f} "
            f"net={subset['net'].mean():+.5f} "
            f"deepNoSL={((subset['eod_ret'] - COST) <= DEEP).mean():.3f} "
            f"pump20={subset['pump20_hit'].mean():.4f}"
        )

    base_rate = float(coverage["base_rates"]["lab_sus_net"])
    precision = metrics_c1["prec_self"]
    print(
        f"\n  prec_self(C1)={precision:.4f} base_rate={base_rate:.4f} "
        f"lift_delta={precision - base_rate:+.4f}"
    )
    provenance = {
        "input_sha256": {
            "picks": _sha(PICKS),
            "compare": _sha(COMPARE),
            "coverage": _sha(COVERAGE),
            "source": _sha(SOURCE),
        },
        "contracts": contracts,
        "bootstrap": {
            "unit": "paired trading date",
            "kind": "circular moving block",
            "seed": SEED,
            "draws": DRAWS,
            "block": BLOCK,
        },
        "metric_unit_note": (
            "reported CSV Sharpe uses trade mean / daily std; corrected Sharpe "
            "above uses daily mean / daily std"
        ),
    }
    print("\n=== PROVENANCE / CONTRACTS ===")
    print(json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
