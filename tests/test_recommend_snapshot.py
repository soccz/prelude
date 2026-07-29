from __future__ import annotations

import json
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.recommend_send as recommend_send
import scripts.recommend_today as recommend_today
import signals.recommend as recommend
import signals.recommend_snapshot as recommend_snapshot
from data.market_universe import SIGNAL_EXCLUDED_KRW_MARKETS
from signals.recommend_snapshot import (
    SnapshotError,
    get_or_create_recommend_snapshot,
    load_snapshot,
    snapshot_path,
)


def _candidate(rank: int) -> dict:
    return {
        "coin": f"KRW-T{rank}",
        "rank": rank,
        "score": round(1.0 - rank / 10, 4),
        "pump_prob": 0.02,
        "pump_prob_pct": "2.0%",
        "rr_ratio": 1.0,
        "p_up5": 0.3,
        "p_up10": 0.1,
        "p_up20": 0.02,
        "p_dn5": 0.1,
        "p_dn10": 0.03,
        "exp_downside": -0.02,
        "dump_risk_flag": rank == 1,
        "entry_open": 100.0,
        "sl": -0.03,
        "tp": 0.05,
        "btc_regime": "neutral",
        "feature_values": {
            "f_ret_3d": 0.01 * rank,
            "f_log_qv": 10.0 + rank,
        },
    }


def _score_result(asof: str, slot: str, ranking: str) -> dict:
    universe = [_candidate(i) for i in range(1, 5)]
    feature_date = (
        asof
        if slot == "open"
        else str(date.fromisoformat(asof) - timedelta(days=1))
    )
    cutoff = str(date.fromisoformat(feature_date) - timedelta(days=5))
    if slot == "preopen":
        for candidate in universe:
            candidate["entry_open"] = None
    return {
        "asof": asof,
        "slot": slot,
        "feature_date": feature_date,
        "btc_regime": "neutral",
        "universe_n": len(universe),
        "calibration_source": "bucket_score_pump20",
        "rank_basis": "R1_riskreward(de-corr head)",
        "n_history_dates": 100,
        "ranking": ranking,
        "score_schema_version": "recommend_score.v1",
        "rule_version": "r1_riskreward_v1",
        "model_random_seed": 42,
        "feature_columns": ["f_ret_3d", "f_log_qv"],
        "training": {
            "start": "2025-01-01",
            "end": str(date.fromisoformat(cutoff) - timedelta(days=1)),
            "cutoff_exclusive": cutoff,
            "embargo_days": 5,
            "rows": 1000,
            "dates": 100,
        },
        "universe": universe,
        "top3": universe[:3],
    }


def test_snapshot_reuses_score_once_and_separates_slots(tmp_path):
    calls: list[tuple[str, str, str]] = []

    def scorer(asof, *, limit_markets, slot, ranking):
        calls.append((asof, slot, ranking))
        return _score_result(asof, slot, ranking)

    first = get_or_create_recommend_snapshot(
        "2026-07-25", slot="open", root=tmp_path, scorer=scorer
    )
    second = get_or_create_recommend_snapshot(
        "2026-07-25", slot="open", root=tmp_path, scorer=scorer
    )
    preopen = get_or_create_recommend_snapshot(
        "2026-07-25", slot="preopen", root=tmp_path, scorer=scorer
    )

    assert calls == [
        ("2026-07-25", "open", "R1"),
        ("2026-07-25", "preopen", "R1"),
    ]
    assert first["snapshot_id"] == second["snapshot_id"]
    assert first["top3"] == second["top3"]
    assert first["top3"] == first["universe"][:3]
    assert first["feature_asof"] == "2026-07-25"
    assert preopen["feature_asof"] == "2026-07-24"
    assert first["snapshot_path"] != preopen["snapshot_path"]
    assert first["model"]["random_seed"] == 42
    assert first["training"]["cutoff_exclusive"] == "2026-07-20"
    assert first["decision_started_at"]
    assert first["decision_completed_at"]
    assert first["decision_started_at"] <= first["decision_completed_at"]
    assert first["created_at"] == first["decision_completed_at"]
    assert first["code"]["score_source_sha256"]
    assert first["snapshot_schema"] == "recommend_snapshot.v2"
    assert "data/market_universe.py" in first["code"]["score_source_files"]
    assert first["environment"]["python"]
    assert "xgboost" in first["environment"]["packages"]
    assert first["data"]["path"] == "data/upbit_d1.db"

    raw = Path(first["snapshot_path"]).read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert json.loads(raw)["payload_sha256"]


def test_auto_and_explicit_slot_share_one_creation_lock(tmp_path):
    scorer_entered = threading.Event()
    second_scorer_entered = threading.Event()
    release_scorer = threading.Event()
    call_lock = threading.Lock()
    calls = 0
    errors: list[BaseException] = []
    results: dict[str, dict] = {}

    def scorer(asof, *, limit_markets, slot, ranking):
        nonlocal calls
        with call_lock:
            calls += 1
            call_number = calls
        if call_number == 1:
            scorer_entered.set()
            if not release_scorer.wait(timeout=5):
                raise TimeoutError("test scorer was not released")
        else:
            second_scorer_entered.set()
        resolved_slot = "open" if slot == "auto" else slot
        return _score_result(asof, resolved_slot, ranking)

    def create(name, slot):
        try:
            results[name] = get_or_create_recommend_snapshot(
                "2026-07-25",
                slot=slot,
                root=tmp_path,
                scorer=scorer,
            )
        except BaseException as exc:
            errors.append(exc)

    explicit = threading.Thread(target=create, args=("explicit", "open"))
    automatic = threading.Thread(target=create, args=("auto", "auto"))
    explicit.start()
    assert scorer_entered.wait(timeout=2)
    automatic.start()
    try:
        assert not second_scorer_entered.wait(timeout=0.25)
    finally:
        release_scorer.set()
    explicit.join(timeout=5)
    automatic.join(timeout=5)

    assert not explicit.is_alive()
    assert not automatic.is_alive()
    assert not errors
    assert calls == 1
    assert results["explicit"]["snapshot_id"] == results["auto"]["snapshot_id"]


def test_score_source_hash_rejects_symlink(tmp_path, monkeypatch):
    target = tmp_path / "outside.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    source = tmp_path / "source.py"
    source.symlink_to(target)
    monkeypatch.setattr(recommend_snapshot, "_ROOT", tmp_path)

    with pytest.raises(SnapshotError, match="cannot be hashed safely"):
        recommend_snapshot._source_sha256(("source.py",))


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_score_candidates_rejects_invalid_market_limit_before_data_load(limit):
    with pytest.raises(ValueError, match="limit_markets"):
        recommend.score_candidates("2026-07-25", limit_markets=limit)
    with pytest.raises(ValueError, match="limit_markets"):
        snapshot_path("2026-07-25", "open", limit_markets=limit)


def test_build_panel_excludes_persisted_signal_markets_before_market_limit(
    caplog,
    monkeypatch,
):
    persisted = [
        *sorted(SIGNAL_EXCLUDED_KRW_MARKETS),
        "KRW-BTC",
        "KRW-WBTC",
        "KRW-WETH",
        "KRW-ETH",
    ]
    timestamps = pd.date_range("2026-05-01 09:00:00", periods=80, freq="D")
    candles = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10.0,
            "quote_volume": 1000.0,
        }
    )
    loaded: list[str] = []
    monkeypatch.setattr(recommend, "list_markets", lambda _db: persisted)

    def load_market(_db, market):
        loaded.append(market)
        return candles.copy()

    def build_features(frame):
        return pd.DataFrame(
            {
                "timestamp": frame["timestamp"].to_numpy(),
                "market": frame["market"].to_numpy(),
                "intraday_high_ret": 0.01,
            }
        )

    monkeypatch.setattr(recommend, "load_candles", load_market)
    monkeypatch.setattr(recommend, "build_market_features", build_features)

    with caplog.at_level("INFO", logger="recommend"):
        panel = recommend._build_panel(
            pd.Timestamp("2026-07-25"),
            limit_markets=3,
        )

    assert loaded == ["KRW-BTC", "KRW-WBTC", "KRW-WETH"]
    assert set(panel["market"]) == {"KRW-BTC", "KRW-WBTC", "KRW-WETH"}
    assert "excluding 6 persisted signal market(s)" in caplog.text
    assert all(
        market not in loaded
        for market in SIGNAL_EXCLUDED_KRW_MARKETS
    )
    assert panel["history_prior_bars"].min() == recommend.MIN_HISTORY


def test_build_panel_applies_history_gate_per_market_before_cross_section(
    monkeypatch,
):
    old_dates = pd.date_range("2026-04-01 09:00:00", periods=90, freq="D")
    new_dates = pd.date_range("2026-04-20 09:00:00", periods=71, freq="D")

    def candles(dates):
        return pd.DataFrame(
            {
                "timestamp": dates,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10.0,
                "quote_volume": 1000.0,
            }
        )

    by_market = {
        "KRW-OLD": candles(old_dates),
        "KRW-NEW": candles(new_dates),
    }
    monkeypatch.setattr(
        recommend,
        "list_markets",
        lambda _db: list(by_market),
    )
    monkeypatch.setattr(
        recommend,
        "load_candles",
        lambda _db, market: by_market[market].copy(),
    )
    monkeypatch.setattr(
        recommend,
        "build_market_features",
        lambda frame: pd.DataFrame(
            {
                "timestamp": frame["timestamp"].to_numpy(),
                "market": frame["market"].to_numpy(),
                "intraday_high_ret": 0.01,
            }
        ),
    )

    panel = recommend._build_panel(pd.Timestamp("2026-07-20"))

    assert (panel["history_prior_bars"] >= recommend.MIN_HISTORY).all()
    new_rows = panel[panel["market"] == "KRW-NEW"]
    assert len(new_rows) == 1
    assert new_rows.iloc[0]["history_prior_bars"] == recommend.MIN_HISTORY
    assert new_rows.iloc[0]["timestamp"] == new_dates[recommend.MIN_HISTORY]
    before_eligible = panel[
        panel["timestamp"] < new_dates[recommend.MIN_HISTORY]
    ]
    assert "KRW-NEW" not in set(before_eligible["market"])


def test_build_panel_rejects_market_with_only_69_prior_bars(monkeypatch):
    dates = pd.date_range("2026-05-01 09:00:00", periods=70, freq="D")
    candles = pd.DataFrame(
        {
            "timestamp": dates,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10.0,
            "quote_volume": 1000.0,
        }
    )
    monkeypatch.setattr(recommend, "list_markets", lambda _db: ["KRW-NEW"])
    monkeypatch.setattr(
        recommend,
        "load_candles",
        lambda _db, _market: candles.copy(),
    )

    with pytest.raises(RuntimeError, match="70 prior bars"):
        recommend._build_panel(pd.Timestamp("2026-07-20"))


def test_legacy_v1_snapshot_remains_readable_after_v2_provenance_upgrade(tmp_path):
    snapshot = get_or_create_recommend_snapshot(
        "2026-07-25",
        slot="open",
        root=tmp_path,
        scorer=lambda asof, **kwargs: _score_result(
            asof,
            kwargs["slot"],
            kwargs["ranking"],
        ),
    )
    path = Path(snapshot["snapshot_path"])

    def downgrade_to_v1(document):
        document["snapshot_schema"] = recommend_snapshot.LEGACY_SNAPSHOT_SCHEMA_VERSION
        document["schema"]["snapshot"] = (
            recommend_snapshot.LEGACY_SNAPSHOT_SCHEMA_VERSION
        )
        document["code"]["score_source_files"] = list(
            recommend_snapshot._SCORE_SOURCE_FILES_V1
        )
        document["code"]["score_source_sha256"] = (
            recommend_snapshot._source_sha256(
                recommend_snapshot._SCORE_SOURCE_FILES_V1
            )
        )

    _resign_snapshot(path, downgrade_to_v1)

    loaded = load_snapshot(path)
    assert loaded["snapshot_schema"] == "recommend_snapshot.v1"


def test_score_candidates_preopen_rejects_d2_stale_panel_before_scoring(
    monkeypatch,
):
    stale_panel = pd.DataFrame({"date": [date(2026, 7, 23)]})
    monkeypatch.setattr(
        recommend,
        "_build_panel",
        lambda *_args, **_kwargs: stale_panel,
    )
    for name in (
        "add_cross_sectional",
        "attach_btc_regime",
        "_add_score",
        "_add_universe",
    ):
        monkeypatch.setattr(recommend, name, lambda frame: frame)

    with pytest.raises(RuntimeError, match="exactly asof-1 calendar day"):
        recommend.score_candidates("2026-07-25", slot="preopen")


def test_snapshot_rejects_tampered_id_even_when_payload_checksum_is_unchanged(
    tmp_path,
):
    snapshot = get_or_create_recommend_snapshot(
        "2026-07-25",
        slot="open",
        root=tmp_path,
        scorer=lambda asof, **kwargs: _score_result(
            asof, kwargs["slot"], kwargs["ranking"]
        ),
    )
    path = Path(snapshot["snapshot_path"])
    document = json.loads(path.read_text(encoding="utf-8"))
    document["snapshot_id"] = "recommend-" + "0" * 20
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(SnapshotError, match="snapshot_id mismatch"):
        load_snapshot(path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda document: document.__setitem__(
                "created_at",
                "2026-07-25T23:59:59+00:00",
            ),
            "timestamp chronology",
        ),
        (
            lambda document: (
                document["top3"][1].__setitem__(
                    "coin",
                    document["universe"][0]["coin"],
                ),
                document["universe"][1].__setitem__(
                    "coin",
                    document["universe"][0]["coin"],
                ),
            ),
            "duplicate candidate",
        ),
        (
            lambda document: (
                document["top3"][1].__setitem__("rank", 9),
                document["universe"][1].__setitem__("rank", 9),
            ),
            "ranks are not contiguous",
        ),
        (
            lambda document: (
                document.__setitem__("feature_asof", "2026-07-24"),
                document.__setitem__("feature_date", "2026-07-24"),
            ),
            "open feature_asof",
        ),
    ],
)
def test_snapshot_rejects_internally_checksummed_contract_violation(
    tmp_path,
    mutate,
    message,
):
    snapshot = get_or_create_recommend_snapshot(
        "2026-07-25",
        slot="open",
        root=tmp_path,
        scorer=lambda asof, **kwargs: _score_result(
            asof,
            kwargs["slot"],
            kwargs["ranking"],
        ),
    )
    path = Path(snapshot["snapshot_path"])
    document = json.loads(path.read_text(encoding="utf-8"))
    mutate(document)
    digest = recommend_snapshot._document_digest(document)
    document["payload_sha256"] = digest
    document["snapshot_id"] = f"recommend-{digest[:20]}"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(SnapshotError, match=message):
        load_snapshot(path)


def _resign_snapshot(path: Path, mutate) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    mutate(document)
    digest = recommend_snapshot._document_digest(document)
    document["payload_sha256"] = digest
    document["snapshot_id"] = f"recommend-{digest[:20]}"
    path.write_text(json.dumps(document), encoding="utf-8")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda document: (
                document["top3"][0].__setitem__("score", 1.01),
                document["universe"][0].__setitem__("score", 1.01),
            ),
            "score outside",
        ),
        (
            lambda document: document["top3"].pop(),
            "top3 size",
        ),
        (
            lambda document: document["rule"].__setitem__("tp_pct", 0.06),
            "exit rule metadata",
        ),
        (
            lambda document: (
                document["top3"][0].__setitem__("p_up10", 0.4),
                document["universe"][0].__setitem__("p_up10", 0.4),
            ),
            "probability nesting violated in top-k",
        ),
        (
            lambda document: (
                document["top3"][0].__setitem__("rr_ratio", 99.0),
                document["universe"][0].__setitem__("rr_ratio", 99.0),
            ),
            "rr_ratio is inconsistent",
        ),
        (
            lambda document: document["rule"].__setitem__(
                "rr_ratio_eps",
                0.01,
            ),
            "risk-reward epsilon metadata mismatch",
        ),
        (
            lambda document: document["training"].__setitem__(
                "cutoff_exclusive",
                "2026-07-19",
            ),
            "embargo metadata",
        ),
    ],
)
def test_snapshot_rejects_additional_resigned_semantic_corruption(
    tmp_path,
    mutate,
    message,
):
    snapshot = get_or_create_recommend_snapshot(
        "2026-07-25",
        slot="open",
        root=tmp_path,
        scorer=lambda asof, **kwargs: _score_result(
            asof,
            kwargs["slot"],
            kwargs["ranking"],
        ),
    )
    path = Path(snapshot["snapshot_path"])
    _resign_snapshot(path, mutate)

    with pytest.raises(SnapshotError, match=message):
        load_snapshot(path)


@pytest.mark.parametrize(
    ("raw_up10", "raw_dn5"),
    [
        (0.123449, 0.010049),
        (0.123449, 0.000951),
        (0.123449, 0.001049),
    ],
)
def test_v2_rr_ratio_accepts_values_possible_before_four_decimal_rounding(
    tmp_path,
    raw_up10,
    raw_dn5,
):
    def scorer(asof, **kwargs):
        result = _score_result(asof, kwargs["slot"], kwargs["ranking"])
        for candidate in result["universe"]:
            candidate["p_up10"] = round(raw_up10, 4)
            candidate["p_dn5"] = round(raw_dn5, 4)
            candidate["p_dn10"] = min(candidate["p_dn5"], 0.0005)
            candidate["rr_ratio"] = round(
                raw_up10 / max(raw_dn5, recommend_snapshot.RR_RATIO_EPS),
                4,
            )
        return result

    snapshot = get_or_create_recommend_snapshot(
        "2026-07-25",
        slot="open",
        root=tmp_path,
        scorer=scorer,
    )

    assert load_snapshot(snapshot["snapshot_path"])["snapshot_schema"] == (
        recommend_snapshot.SNAPSHOT_SCHEMA_VERSION
    )


def test_v2_probability_display_accepts_pre_rounding_value(tmp_path):
    def scorer(asof, **kwargs):
        result = _score_result(asof, kwargs["slot"], kwargs["ranking"])
        for candidate in result["universe"]:
            candidate["pump_prob"] = 0.0255
            candidate["pump_prob_pct"] = "2.6%"
            candidate["p_up20"] = 0.0255
        return result

    snapshot = get_or_create_recommend_snapshot(
        "2026-07-25",
        slot="open",
        root=tmp_path,
        scorer=scorer,
    )

    assert load_snapshot(snapshot["snapshot_path"])["top3"][0][
        "pump_prob_pct"
    ] == "2.6%"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda candidate: candidate.__setitem__("rr_ratio", None),
            "rr_ratio is required",
        ),
        (
            lambda candidate: candidate.__setitem__("p_up20", 0.2),
            "probability nesting violated in top-k",
        ),
    ],
)
def test_v2_snapshot_creation_fails_closed_on_degraded_score_vector(
    tmp_path,
    mutate,
    message,
):
    def scorer(asof, **kwargs):
        result = _score_result(asof, kwargs["slot"], kwargs["ranking"])
        for candidate in result["universe"]:
            mutate(candidate)
        return result

    with pytest.raises(SnapshotError, match=message):
        get_or_create_recommend_snapshot(
            "2026-07-25",
            slot="open",
            root=tmp_path,
            scorer=scorer,
        )

    assert not list(tmp_path.rglob("*.json"))


def test_snapshot_tolerates_probability_nesting_violation_as_diagnostic(
    tmp_path,
    caplog,
    monkeypatch,
):
    def scorer(asof, **kwargs):
        result = _score_result(asof, kwargs["slot"], kwargs["ranking"])
        degraded = result["universe"][3]
        degraded["p_up20"] = 0.2
        degraded["pump_prob"] = 0.2
        degraded["pump_prob_pct"] = "20.0%"
        return result

    # 모듈 로거는 운영에서 stderr 전용(propagate=False — NUL/프로브 오염 방지).
    # caplog 는 root 전파에 의존하므로 테스트에서만 전파를 복원한다.
    monkeypatch.setattr(recommend_snapshot.log, "propagate", True)
    with caplog.at_level("WARNING", logger="signals.recommend_snapshot"):
        snapshot = get_or_create_recommend_snapshot(
            "2026-07-25",
            slot="open",
            root=tmp_path,
            scorer=scorer,
        )
        loaded = load_snapshot(snapshot["snapshot_path"])

    assert loaded["universe"][3]["p_up20"] == 0.2
    nesting_warnings = [
        record
        for record in caplog.records
        if "probability nesting violated" in record.getMessage()
    ]
    assert nesting_warnings
    assert "1/4" in nesting_warnings[0].getMessage()
    assert "KRW-T4" in nesting_warnings[0].getMessage()


def test_snapshot_rejects_resigned_preopen_entry_price(tmp_path):
    snapshot = get_or_create_recommend_snapshot(
        "2026-07-25",
        slot="preopen",
        root=tmp_path,
        scorer=lambda asof, **kwargs: _score_result(
            asof,
            kwargs["slot"],
            kwargs["ranking"],
        ),
    )
    path = Path(snapshot["snapshot_path"])

    def add_future_entry(document):
        document["top3"][0]["entry_open"] = 100.0
        document["universe"][0]["entry_open"] = 100.0

    _resign_snapshot(path, add_future_entry)
    with pytest.raises(SnapshotError, match="preopen.*entry_open must be null"):
        load_snapshot(path)


def test_snapshot_rejects_resigned_preopen_stale_feature_day(tmp_path):
    snapshot = get_or_create_recommend_snapshot(
        "2026-07-25",
        slot="preopen",
        root=tmp_path,
        scorer=lambda asof, **kwargs: _score_result(
            asof,
            kwargs["slot"],
            kwargs["ranking"],
        ),
    )
    path = Path(snapshot["snapshot_path"])

    def make_feature_day_stale(document):
        document["feature_asof"] = "2026-07-23"
        document["feature_date"] = "2026-07-23"

    _resign_snapshot(path, make_feature_day_stale)
    with pytest.raises(SnapshotError, match="exactly asof-1d"):
        load_snapshot(path)


def test_snapshot_rejects_resigned_leak_feature_metadata(tmp_path):
    snapshot = get_or_create_recommend_snapshot(
        "2026-07-25",
        slot="open",
        root=tmp_path,
        scorer=lambda asof, **kwargs: _score_result(
            asof,
            kwargs["slot"],
            kwargs["ranking"],
        ),
    )
    path = Path(snapshot["snapshot_path"])

    def add_leak_feature(document):
        document["feature_columns"].append("next_high")
        document["features"]["columns"].append("next_high")
        for collection in ("top3", "universe"):
            for candidate in document[collection]:
                candidate["feature_values"]["next_high"] = 123.0

    _resign_snapshot(path, add_leak_feature)
    with pytest.raises(SnapshotError, match="contain leak fields"):
        load_snapshot(path)


@pytest.mark.parametrize(
    "payload",
    [
        "[]",
        '{"snapshot_schema": NaN}',
        '{"snapshot_schema": "recommend_snapshot.v1", '
        '"snapshot_schema": "recommend_snapshot.v1"}',
    ],
)
def test_snapshot_rejects_non_object_nonstandard_or_duplicate_json(
    tmp_path,
    payload,
):
    path = tmp_path / "invalid.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(SnapshotError):
        load_snapshot(path)


def test_snapshot_rejects_symlink_even_when_target_is_valid(tmp_path):
    snapshot = get_or_create_recommend_snapshot(
        "2026-07-25",
        slot="open",
        root=tmp_path / "canonical",
        scorer=lambda asof, **kwargs: _score_result(
            asof,
            kwargs["slot"],
            kwargs["ranking"],
        ),
    )
    target = Path(snapshot["snapshot_path"])
    alias = tmp_path / "snapshot-alias.json"
    alias.symlink_to(target)

    with pytest.raises(SnapshotError, match="snapshot read failed"):
        load_snapshot(alias)


def test_snapshot_reuse_is_bound_to_requested_model_identity(tmp_path):
    calls = 0

    def scorer(asof, *, slot, ranking, **_kwargs):
        nonlocal calls
        calls += 1
        return _score_result(asof, slot, ranking)

    first = get_or_create_recommend_snapshot(
        "2026-07-25",
        slot="open",
        model_id="model-A",
        root=tmp_path,
        scorer=scorer,
    )
    assert first["model"]["id"] == "model-A"

    with pytest.raises(SnapshotError, match="model identity mismatch"):
        get_or_create_recommend_snapshot(
            "2026-07-25",
            slot="open",
            model_id="model-B",
            root=tmp_path,
            scorer=scorer,
        )
    assert calls == 1


def test_snapshot_rejects_scorer_identity_drift_and_input_race(
    tmp_path, monkeypatch
):
    wrong_slot = lambda asof, **kwargs: _score_result(  # noqa: E731
        asof, "preopen", kwargs["ranking"]
    )
    with pytest.raises(SnapshotError, match="requested 'open'"):
        get_or_create_recommend_snapshot(
            "2026-07-25",
            slot="open",
            root=tmp_path / "slot",
            scorer=wrong_slot,
        )

    manifests = iter([
        {"manifest_id": "before"},
        {"manifest_id": "after"},
    ])
    monkeypatch.setattr(
        recommend_snapshot, "_data_metadata", lambda: next(manifests)
    )
    with pytest.raises(SnapshotError, match="D1 input changed"):
        get_or_create_recommend_snapshot(
            "2026-07-25",
            slot="open",
            root=tmp_path / "race",
            scorer=lambda asof, **kwargs: _score_result(
                asof, kwargs["slot"], kwargs["ranking"]
            ),
        )


def test_dry_run_send_then_ledger_consumes_same_snapshot_without_refit(
    tmp_path, monkeypatch
):
    snapshot_root = tmp_path / "snapshots"
    ledger = tmp_path / "ledger.csv"
    calls = 0
    messages: list[str] = []

    def scorer(asof, *, limit_markets, slot, ranking):
        nonlocal calls
        calls += 1
        return _score_result(asof, slot, ranking)

    spec = SimpleNamespace(
        id="recommend_r1_open",
        predict_ref="signals.recommend:score_candidates",
    )
    monkeypatch.setattr(recommend, "score_candidates", scorer)
    monkeypatch.setattr(
        recommend_send,
        "resolve_champion",
        lambda slot, **_kwargs: (spec, False, "test"),
    )
    monkeypatch.setattr(
        recommend_send, "maybe_notify_champion_change", lambda *a, **k: None
    )
    monkeypatch.setattr(
        recommend_send,
        "send_telegram",
        lambda message, dry_run=False: messages.append(message) or True,
    )
    monkeypatch.setattr(
        recommend_today,
        "read_delivery_receipt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dry-run consumed a delivery receipt")
        ),
    )

    assert recommend_send.send_recommendation(
        "2026-07-25",
        "open",
        dry_run=True,
        snapshot_root=snapshot_root,
    )
    ledger_result = recommend_today.append_today(
        "2026-07-25",
        dry_run=True,
        ledger_path=str(ledger),
        slot="open",
        snapshot_root=snapshot_root,
    )

    assert calls == 1
    assert len(messages) == 1
    assert ledger_result["top3"] == _score_result("2026-07-25", "open", "R1")["top3"]
    assert not ledger.exists()


def test_default_dry_runs_cannot_reserve_canonical_snapshot(
    tmp_path,
    monkeypatch,
):
    calls: list[tuple[str, str, str]] = []

    def scorer(asof, *, limit_markets, slot, ranking):
        calls.append((asof, slot, ranking))
        return _score_result(asof, slot, ranking)

    def reject_snapshot_write(*_args, **_kwargs):
        raise AssertionError("default dry-run touched the snapshot store")

    spec = SimpleNamespace(
        id="recommend_r1_open",
        predict_ref="signals.recommend:score_candidates",
    )
    monkeypatch.setattr(recommend, "score_candidates", scorer)
    monkeypatch.setattr(
        recommend_send,
        "get_or_create_recommend_snapshot",
        reject_snapshot_write,
    )
    monkeypatch.setattr(
        recommend_today,
        "get_or_create_recommend_snapshot",
        reject_snapshot_write,
    )
    monkeypatch.setattr(
        recommend_today,
        "read_delivery_receipt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ephemeral dry-run consumed canonical receipt")
        ),
    )
    monkeypatch.setattr(
        recommend_send,
        "resolve_champion",
        lambda slot, **_kwargs: (spec, False, "test"),
    )
    monkeypatch.setattr(
        recommend_send,
        "maybe_notify_champion_change",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        recommend_send,
        "send_telegram",
        lambda _message, *, dry_run=False: dry_run,
    )
    preview_ledger = tmp_path / "ledger.csv"
    preview_ledger.write_bytes(b"pre-existing canonical bytes\n")

    assert recommend_send.send_recommendation(
        "2026-07-25",
        "open",
        dry_run=True,
    )
    result = recommend_today.append_today(
        "2026-07-25",
        dry_run=True,
        ledger_path=str(preview_ledger),
        slot="open",
    )

    assert calls == [
        ("2026-07-25", "open", "R1"),
        ("2026-07-25", "open", "R1"),
    ]
    assert result["top3"] == _score_result(
        "2026-07-25",
        "open",
        "R1",
    )["top3"]
    assert preview_ledger.read_bytes() == b"pre-existing canonical bytes\n"

    with pytest.raises(
        RuntimeError,
        match="requires a persisted score snapshot",
    ):
        recommend_today.append_today(
            "2026-07-25",
            dry_run=True,
            ledger_path=str(preview_ledger),
            slot="open",
            require_receipt=True,
        )


def test_ledger_dry_run_explicit_canonical_root_stays_ephemeral(
    tmp_path,
    monkeypatch,
):
    calls = 0

    def scorer(asof, *, limit_markets, slot, ranking):
        nonlocal calls
        calls += 1
        return _score_result(asof, slot, ranking)

    monkeypatch.setattr(recommend, "score_candidates", scorer)
    monkeypatch.setattr(
        recommend_today,
        "get_or_create_recommend_snapshot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("explicit canonical root was read or written")
        ),
    )
    ledger = tmp_path / "ledger.csv"

    result = recommend_today.append_today(
        "2026-07-25",
        dry_run=True,
        ledger_path=str(ledger),
        slot="open",
        snapshot_root=recommend_snapshot.DEFAULT_SNAPSHOT_ROOT,
    )

    assert calls == 1
    assert result["top3"]
    assert not ledger.exists()


def test_real_send_writes_delivery_receipt_for_snapshot(tmp_path, monkeypatch):
    snapshot_root = tmp_path / "snapshots"
    receipt_root = tmp_path / "receipts"

    def scorer(asof, *, limit_markets, slot, ranking):
        result = _score_result(asof, slot, ranking)
        result["score_schema_version"] = (
            recommend_send.APPROVED_LIVE_SCORE_SCHEMA
        )
        return result

    class DecisionDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = datetime(2026, 7, 25, 0, 5, tzinfo=timezone.utc)
            return value if tz is None else value.astimezone(tz)

    spec = SimpleNamespace(
        id="recommend_r1_open",
        predict_ref="signals.recommend:score_candidates",
    )
    monkeypatch.setattr(recommend, "score_candidates", scorer)
    monkeypatch.setattr(recommend_snapshot, "datetime", DecisionDatetime)
    monkeypatch.setattr(
        recommend_send,
        "resolve_champion",
        lambda slot, **_kwargs: (spec, False, "test"),
    )
    monkeypatch.setattr(
        recommend_send, "maybe_notify_champion_change", lambda *a, **k: None
    )
    monkeypatch.setattr(
        recommend_send,
        "_assert_live_send_window",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(recommend_send, "send_telegram", lambda *a, **k: True)

    assert recommend_send.send_recommendation(
        "2026-07-25",
        "open",
        snapshot_root=snapshot_root,
        receipt_root=receipt_root,
        radar_verdict_path=tmp_path / "radar-terminal.json",
    )

    receipt_file = receipt_root / "2026-07-25" / "open_r1.json"
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    snapshot = json.loads(
        snapshot_path("2026-07-25", "open", root=snapshot_root).read_text(
            encoding="utf-8"
        )
    )
    assert receipt["delivery_ok"] is True
    assert receipt["sent_at"]
    assert receipt["snapshot_id"] == snapshot["snapshot_id"]
    assert receipt["slot"] == "open"
