from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import scripts.binance_leadlag_v1 as binance
import scripts.cc_filtered_multiday_v1 as cc_filtered
import scripts.cc_sustained_label_v1 as cc_sustained
import scripts.ch_features_v1 as ch_features
import scripts.ch_multiday_v1 as ch_multiday
import scripts.ch_regime_split_v1 as ch_regime
import scripts.day_quality_gate_v1 as day_quality
import scripts.downside_aware_recommender_v1 as downside_aware
import scripts.downside_head_riskreward_v1 as downside_head
import scripts.downside_veto_challenger_v1 as downside_veto
import scripts.pump_rule_discovery_v1 as pump_rule
import scripts.recall_universe_recommender_v1 as recall
import scripts.recommendation_scorer_v1 as recommendation
import scripts.regime_split_precursor_v1 as regime
import scripts.safeup_head_challenger_v1 as safeup
import scripts.univariate_precursor_lift_v1 as univariate
from data.market_universe import SIGNAL_EXCLUDED_KRW_MARKETS


class _StopBuilder(RuntimeError):
    pass


LEGACY_LIST_MARKET_SOURCES = (
    "scripts/ablation_common_period_v1.py",
    "scripts/backfill_paper_ledger.py",
    "scripts/backtest_wf_ledger.py",
    "scripts/build_detector_v1.py",
    "scripts/build_distribution_engine_v1.py",
    "scripts/build_preopen_trigger_v1.py",
    "scripts/build_upside_dist_head_v1.py",
    "scripts/calibration_breakdown_v1.py",
    "scripts/execution_sweep_v1.py",
    "scripts/fold_stability_v1.py",
    "scripts/fold_stability_v2.py",
    "scripts/fold_stability_v3.py",
    "scripts/model_vs_random_v1.py",
    "scripts/model_vs_random_v2.py",
    "scripts/pattern_sweep_v1.py",
    "scripts/precursor_15m_v0.py",
    "scripts/precursor_1h_v0.py",
    "scripts/precursor_1h_v1.py",
    "scripts/precursor_4h_v0.py",
    "scripts/predict_today.py",
    "scripts/preopen_first15m_model_v1.py",
    "scripts/regime_threshold_sweep_v1.py",
    "scripts/setup_discovery_v1.py",
    "scripts/sweep_dist_engine_v1.py",
    "scripts/train_phase1_full.py",
    "signals/validate.py",
)


def _call_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


@pytest.mark.parametrize("relative_path", LEGACY_LIST_MARKET_SOURCES)
def test_legacy_list_market_calls_apply_only_the_upbit_signal_universe(
    relative_path: str,
) -> None:
    """Upbit filtering must precede slicing/loading; Binance stays untouched."""
    source_path = Path(__file__).resolve().parents[1] / relative_path
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    parents = {
        id(child): parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node) == "list_markets"
    ]
    assert calls, f"{relative_path}: expected at least one list_markets call"

    for call in calls:
        assert call.args, f"{relative_path}:{call.lineno}: missing DB argument"
        db_arg = ast.unparse(call.args[0]).lower()
        parent = parents.get(id(call))
        filtered = (
            isinstance(parent, ast.Call)
            and _call_name(parent) == "signal_eligible_markets"
            and parent.args
            and parent.args[0] is call
        )
        if "binance" in db_arg:
            assert not filtered, (
                f"{relative_path}:{call.lineno}: Binance universe must remain raw"
            )
        else:
            assert filtered, (
                f"{relative_path}:{call.lineno}: Upbit universe must be filtered "
                "before slicing or loading"
            )


@pytest.mark.parametrize(
    ("module", "builder_name", "loader_name", "market_arg"),
    [
        (univariate, "build_panel", "load_candles", 1),
        (regime, "build_panel", "load_candles", 1),
        (recall, "build_panel", "load_candles", 1),
        (recommendation, "build_panel", "load_candles", 1),
        (downside_aware, "build_panel", "load_candles", 1),
        (downside_head, "build_panel", "load_candles", 1),
        (cc_sustained, "build_panel_c1", "load_candles", 1),
        (safeup, "prepare_panel", "_market_frame", 0),
    ],
)
def test_direct_panel_builders_exclude_signal_markets_before_limit(
    monkeypatch: pytest.MonkeyPatch,
    module,
    builder_name: str,
    loader_name: str,
    market_arg: int,
) -> None:
    requested: list[str] = []
    listed = [*sorted(SIGNAL_EXCLUDED_KRW_MARKETS), "KRW-BTC", "KRW-ETH"]

    monkeypatch.setattr(module, "list_markets", lambda _db: listed)

    def stop_after_first_market(*args, **_kwargs):
        requested.append(str(args[market_arg]))
        raise _StopBuilder

    monkeypatch.setattr(module, loader_name, stop_after_first_market)

    with pytest.raises(_StopBuilder):
        getattr(module, builder_name)(1)

    assert requested == ["KRW-BTC"]


@pytest.mark.parametrize("eligible", [binance.eligible_upbit_markets, pump_rule.eligible_upbit_markets])
def test_exchange_helpers_use_canonical_upbit_universe(
    monkeypatch: pytest.MonkeyPatch,
    eligible,
) -> None:
    module = sys.modules[eligible.__module__]
    listed = [
        *sorted(SIGNAL_EXCLUDED_KRW_MARKETS),
        "BINANCE-BTCUSDT",
        "KRW-BTC",
        "KRW-WBTC",
        "KRW-WETH",
    ]
    monkeypatch.setattr(module, "list_markets", lambda _db: listed)

    assert eligible("ignored.db") == ["KRW-BTC", "KRW-WBTC", "KRW-WETH"]


def test_pump_rule_rejects_non_upbit_or_excluded_signal_targets() -> None:
    pump_rule.validate_upbit_target_panel(
        pd.DataFrame({"market": ["KRW-BTC", "KRW-WBTC"]})
    )

    for contaminated in ("BINANCE-BTCUSDT", "KRW-USDT", "KRW-IP"):
        with pytest.raises(RuntimeError, match="canonical Upbit universe"):
            pump_rule.validate_upbit_target_panel(
                pd.DataFrame({"market": ["KRW-BTC", contaminated]})
            )


@pytest.mark.parametrize(
    ("guard", "message"),
    [
        (ch_regime._reject_contaminated_cache, "rerun with --rebuild-panel"),
        (cc_filtered._reject_contaminated_oos, "rerun without --use-cache"),
        (ch_multiday._reject_contaminated_picks, "regenerate"),
    ],
)
@pytest.mark.parametrize("contaminated", ["KRW-USD1", "KRW-IP"])
def test_contaminated_research_caches_fail_closed(
    guard,
    message: str,
    contaminated: str,
) -> None:
    clean = pd.DataFrame(
        {
            "market": ["KRW-BTC"],
            "up_high_ret": [0.1],
            "down_low_ret": [-0.1],
            "eod_ret": [0.01],
            "lab_dump_B": [0.0],
        }
    )
    guard(clean)
    contaminated_frame = pd.concat(
        [clean, clean.assign(market=contaminated)],
        ignore_index=True,
    )

    with pytest.raises(RuntimeError) as exc_info:
        guard(contaminated_frame)

    error = str(exc_info.value)
    assert "excluded signal markets" in error
    assert message in error


@pytest.mark.parametrize("contaminated", ["KRW-USDE", "KRW-IP"])
def test_downside_oos_rejects_signal_exclusion_before_candidate_ranking(
    tmp_path: Path,
    contaminated: str,
) -> None:
    frame = pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "market": [contaminated],
            "fold": [0],
            "p_lab_up_10": [0.1],
            "p_lab_dn_05": [0.1],
            "p_lab_dn_10": [0.1],
            "exp_downside": [-0.1],
            "up_high_ret": [0.2],
            "down_low_ret": [-0.1],
            "eod_ret": [0.05],
            "f_qv_rank": [1],
        }
    )
    path = tmp_path / "contaminated_oos.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(RuntimeError, match="excluded signal markets"):
        downside_veto._read_oos(path)


@pytest.mark.parametrize(
    ("function", "source_module", "source_name"),
    [
        (ch_features.build_breadth_features, ch_features.mb, "build_coin_pump_flags"),
        (ch_features.build_liq_features, ch_features.liq, "build_features"),
    ],
)
def test_ch_feature_context_filters_signal_exclusions_before_xs_build(
    monkeypatch: pytest.MonkeyPatch,
    function,
    source_module,
    source_name: str,
) -> None:
    raw = pd.DataFrame(
        {
            "market": ["KRW-USDS", "KRW-IP", "KRW-BTC"],
            "timestamp": pd.to_datetime(["2024-01-01"] * 3),
        }
    )
    load_name = "load_candles" if source_module is ch_features.mb else "load_all"
    monkeypatch.setattr(source_module, load_name, lambda *_args: raw.copy())
    captured: list[pd.DataFrame] = []

    def capture(frame: pd.DataFrame):
        captured.append(frame.copy())
        raise _StopBuilder

    monkeypatch.setattr(source_module, source_name, capture)

    with pytest.raises(_StopBuilder):
        function()

    assert captured[0]["market"].tolist() == ["KRW-BTC"]


class _DummyConnection:
    def close(self) -> None:
        return None


def _raw_daily_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market": ["KRW-USDC", "KRW-IP", "KRW-BTC"],
            "timestamp": pd.to_datetime(["2024-01-01 09:00:00"] * 3),
            "open": [1.0, 2.0, 100.0],
            "high": [1.0, 2.1, 101.0],
            "low": [1.0, 1.9, 99.0],
            "close": [1.0, 2.0, 100.5],
            "quote_volume": [1e12, 1e11, 1e9],
        }
    )


def test_day_quality_filters_signal_exclusions_before_daily_aggregation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _raw_daily_rows()
    monkeypatch.setattr(day_quality.sqlite3, "connect", lambda _path: _DummyConnection())
    monkeypatch.setattr(
        day_quality.pd,
        "read_sql",
        lambda *_args, **_kwargs: raw.copy(),
    )

    loaded = day_quality.load_all_candles()

    assert loaded["market"].tolist() == ["KRW-BTC"]


@pytest.mark.parametrize(
    "module",
    [ch_multiday, cc_filtered],
)
def test_multiday_wrappers_filter_signal_exclusions_before_basket_ranking(
    monkeypatch: pytest.MonkeyPatch,
    module,
) -> None:
    raw = _raw_daily_rows()
    monkeypatch.setattr(module.sqlite3, "connect", lambda _path: _DummyConnection())
    monkeypatch.setattr(
        module.pd,
        "read_sql",
        lambda *_args, **_kwargs: raw.copy(),
    )

    loaded = module.load_d1()

    assert loaded["market"].tolist() == ["KRW-BTC"]


def test_day_quality_is_importable_by_path_outside_repo(tmp_path: Path) -> None:
    script = Path(day_quality.__file__).resolve()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import runpy; "
                f"runpy.run_path({str(script)!r}, run_name='research_import_smoke')"
            ),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
