from __future__ import annotations

import hashlib
import json

import ops.champion_selector as selector
import pandas as pd
import pytest
from signals.model_registry import MODELS, MetricSource, ModelSpec


def _state() -> dict:
    state = {
        "schema_version": selector.STATE_SCHEMA_VERSION,
        "asof": "2026-07-25",
        "updated_at": "2026-07-25T00:00:00+00:00",
        "config": selector._expected_config(),
        "slots": {
            "open": {
                "champion_id": "recommend_r1_open",
                "since": "2026-07-25",
                "is_fallback": False,
                "metric": None,
                "reason": "test",
            },
            "preopen": {
                "champion_id": "recommend_r1_preopen",
                "since": "2026-07-25",
                "is_fallback": True,
                "metric": None,
                "reason": "test",
            },
        },
        "streaks": {},
        "history": [],
    }
    state["payload_sha256"] = selector._state_digest(state)
    return state


def _legacy_state() -> dict:
    state = _state()
    state["config"] = selector._legacy_config_without_path_or_net_gate()
    state["slots"]["preopen"]["champion_id"] = "recommend_r1_open"
    state["slots"]["preopen"]["since"] = "2026-06-01"
    state["slots"]["open"]["since"] = "2026-06-01"
    state["history"] = [
        {
            "asof": "2026-06-01",
            "slot": "open",
            "from": None,
            "to": "recommend_r1_open",
            "reason": "legacy open assignment",
        },
        {
            "asof": "2026-06-01",
            "slot": "preopen",
            "from": None,
            "to": "recommend_r1_open",
            "reason": "legacy cross-slot fallback",
        },
    ]
    state["payload_sha256"] = selector._state_digest(state)
    return state


def test_intraday_models_use_receipt_aligned_hit_metric():
    intraday_ids = {
        "recommend_r1_open",
        "recommend_r2_open",
        "recommend_r1_sustain_open",
        "recommend_r1_preopen",
        "pump_hunter",
        "pump_hunter_v2",
    }
    specs = {spec.id: spec for spec in MODELS if spec.id in intraday_ids}

    assert set(specs) == intraday_ids
    assert {
        spec.metric.hit_col for spec in specs.values()
    } == {"post_send_pump20_hit"}


def test_champion_reader_fails_closed_on_truncated_or_tampered_state(
    tmp_path,
):
    path = tmp_path / "champion_state.json"
    path.write_text('{"slots":', encoding="utf-8")
    with pytest.raises(selector.ChampionStateError):
        selector.get_champion("open", path)

    state = _state()
    state["slots"]["open"]["champion_id"] = "tampered"
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(selector.ChampionStateError):
        selector.get_champion("open", path)


def test_champion_reader_allows_bootstrap_fallback_only_when_state_is_absent(
    tmp_path,
):
    assert selector.get_champion(
        "open",
        tmp_path / "missing.json",
    ) is None


def test_champion_reader_rejects_dangling_state_symlink(tmp_path):
    path = tmp_path / "champion_state.json"
    path.symlink_to(tmp_path / "missing-target.json")

    with pytest.raises(selector.ChampionStateError, match="read failed"):
        selector.get_champion("open", path)


def test_champion_lock_rejects_symlink_without_touching_target(tmp_path):
    path = tmp_path / "champion_state.json"
    target = tmp_path / "outside.lock"
    target.write_text("do not truncate", encoding="utf-8")
    lock_path = tmp_path / ".champion_state.json.lock"
    lock_path.symlink_to(target)

    with pytest.raises(selector.ChampionStateError, match="lock"):
        with selector._state_lock(path):
            pass

    assert target.read_text(encoding="utf-8") == "do not truncate"


def test_champion_lock_rejects_symlinked_parent(tmp_path):
    actual_parent = tmp_path / "outside"
    actual_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(actual_parent, target_is_directory=True)

    with pytest.raises(selector.ChampionStateError, match="lock"):
        with selector._state_lock(linked_parent / "champion_state.json"):
            pass

    assert list(actual_parent.iterdir()) == []


def test_selector_aborts_when_input_set_changes_during_metrics(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(selector, "STATE_PATH", tmp_path / "champion_state.json")
    monkeypatch.setattr(selector, "MODELS", ())
    captures = [
        {"source": {"exists": True, "size": 1, "sha256": "a" * 64}},
        {"source": {"exists": True, "size": 2, "sha256": "b" * 64}},
    ]
    monkeypatch.setattr(
        selector,
        "file_set_identity",
        lambda *_args, **_kwargs: captures.pop(0),
    )

    with pytest.raises(selector.ChampionStateError, match="changed"):
        selector.run(pd.Timestamp("2026-07-26"), dry_run=True)


def test_champion_state_atomic_writer_round_trips_checksum(tmp_path):
    path = tmp_path / "champion_state.json"
    state = _state()

    selector._atomic_write_state(path, state)

    assert selector.get_champion("open", path)["champion_id"] == (
        "recommend_r1_open"
    )
    assert not list(tmp_path.glob("*.tmp"))


def test_champion_reader_requires_checksum_schema_and_registered_slot(tmp_path):
    path = tmp_path / "champion_state.json"
    state = _state()
    state.pop("payload_sha256")
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(selector.ChampionStateError):
        selector.get_champion("open", path)

    state = _state()
    state["slots"]["preopen"]["champion_id"] = "recommend_r1_open"
    state["payload_sha256"] = selector._state_digest(state)
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(selector.ChampionStateError):
        selector.get_champion("open", path)


def test_champion_reader_can_bind_state_to_decision_asof(tmp_path):
    path = tmp_path / "champion_state.json"
    selector._atomic_write_state(path, _state())

    assert selector.get_champion(
        "open",
        path,
        expected_asof="2026-07-25",
    ) is not None
    assert selector.get_champion(
        "open",
        path,
        expected_asof="2026-07-26",
    ) is not None
    with pytest.raises(selector.ChampionStateError):
        selector.get_champion(
            "open",
            path,
            expected_asof="2026-07-27",
        )


def test_champion_deep_loss_uses_complete_no_sl_path_not_hard_sl_return(
    tmp_path,
):
    ledger = tmp_path / "ledger.csv"
    pd.DataFrame([
        {
            "date": "2026-07-24",
            "status": "closed",
            "realized_pct": -3.15,
            "path_min_pct": -8.0,
            "path_complete": True,
            "hit": 0,
        },
        {
            "date": "2026-07-25",
            "status": "closed",
            "realized_pct": -3.15,
            "path_min_pct": -2.0,
            "path_complete": True,
            "hit": 0,
        },
    ]).to_csv(ledger, index=False)
    spec = ModelSpec(
        id="test-model",
        name="test",
        ledger_path=str(ledger),
        slots=["open"],
        metric=MetricSource(
            status_col="status",
            closed_value="closed",
            date_col="date",
            realized_pct_col="realized_pct",
            hit_col="hit",
            cost_already_deducted=True,
            downside_pct_col="path_min_pct",
            path_complete_col="path_complete",
        ),
        predict_ref="test:predict",
    )

    metric = selector.compute_metric(
        spec,
        pd.Timestamp("2026-07-26"),
    )

    assert metric.deep_loss_freq == 0.5
    assert metric.n_downside == 2
    assert metric.n_downside_days == 2
    assert not metric.gate_pass


@pytest.mark.parametrize(
    ("net_mean_pct", "expected_gate"),
    [
        (-0.01, False),
        (0.01, True),
    ],
)
def test_champion_absolute_gate_requires_positive_cost_deducted_forward_net(
    tmp_path,
    net_mean_pct,
    expected_gate,
):
    ledger = tmp_path / "ledger.csv"
    pd.DataFrame([
        {
            "date": str(day.date()),
            "status": "closed",
            "realized_pct": net_mean_pct,
            "path_min_pct": -1.0,
            "path_complete": True,
            "hit": int(net_mean_pct > 0),
        }
        for day in pd.date_range("2026-06-26", periods=30, freq="D")
    ]).to_csv(ledger, index=False)
    spec = ModelSpec(
        id="test-positive-net-gate",
        name="test",
        ledger_path=str(ledger),
        slots=["open"],
        metric=MetricSource(
            status_col="status",
            closed_value="closed",
            date_col="date",
            realized_pct_col="realized_pct",
            hit_col="hit",
            cost_already_deducted=True,
            downside_pct_col="path_min_pct",
            path_complete_col="path_complete",
        ),
        predict_ref="test:predict",
    )

    metric = selector.compute_metric(spec, pd.Timestamp("2026-07-26"))

    assert metric.n_days == 30
    assert metric.n_downside_days == 30
    assert metric.gate_pass is expected_gate
    if not expected_gate:
        assert "비용차감 forward net_mean" in metric.reason


def test_champion_metric_excludes_same_day_closed_row(tmp_path):
    ledger = tmp_path / "ledger.csv"
    pd.DataFrame([
        {
            "date": "2026-07-25",
            "status": "closed",
            "realized_pct": 99.0,
        }
    ]).to_csv(ledger, index=False)
    spec = ModelSpec(
        id="test-same-day",
        name="test",
        ledger_path=str(ledger),
        slots=["open"],
        metric=MetricSource(
            status_col="status",
            closed_value="closed",
            date_col="date",
            realized_pct_col="realized_pct",
        ),
        predict_ref="test:predict",
    )

    metric = selector.compute_metric(
        spec,
        pd.Timestamp("2026-07-25 09:30:00"),
    )

    assert metric.n_closed == 0
    assert metric.gate_pass is False


def test_champion_metric_excludes_nonfinite_values(tmp_path):
    ledger = tmp_path / "ledger.csv"
    pd.DataFrame([
        {
            "date": "2026-07-23",
            "status": "closed",
            "realized_pct": 1.0,
            "path_min_pct": -6.0,
            "path_complete": True,
            "hit": 1,
        },
        {
            "date": "2026-07-24",
            "status": "closed",
            "realized_pct": float("inf"),
            "path_min_pct": float("-inf"),
            "path_complete": True,
            "hit": float("inf"),
        },
        {
            "date": "2026-07-25",
            "status": "closed",
            "realized_pct": 3.0,
            "path_min_pct": -2.0,
            "path_complete": True,
            "hit": 2,
        },
    ]).to_csv(ledger, index=False)
    spec = ModelSpec(
        id="test-finite-metrics",
        name="test",
        ledger_path=str(ledger),
        slots=["open"],
        metric=MetricSource(
            status_col="status",
            closed_value="closed",
            date_col="date",
            realized_pct_col="realized_pct",
            hit_col="hit",
            cost_already_deducted=True,
            downside_pct_col="path_min_pct",
            path_complete_col="path_complete",
        ),
        predict_ref="test:predict",
    )

    metric = selector.compute_metric(spec, pd.Timestamp("2026-07-26"))

    assert metric.n_closed == 2
    assert metric.n_downside == 2
    assert metric.deep_loss_freq == 0.5
    assert metric.net_mean_pct == 2.0
    assert metric.hit_rate == 1.0


def test_negative_net_incumbent_is_downgraded_to_shadow_fallback():
    metric = selector.ModelMetric(
        model_id="recommend_r1_open",
        n_closed=90,
        n_days=30,
        n_downside=90,
        n_downside_days=30,
        deep_loss_freq=0.0,
        net_mean_pct=-0.3022,
        hit_rate=0.0556,
        gate_pass=False,
        last_date="2026-07-25",
        reason="negative net",
    )
    state = _state()
    state["slots"]["open"]["is_fallback"] = False

    selected = selector.select_for_slot(
        "open",
        {metric.model_id: metric},
        state,
        pd.Timestamp("2026-07-26"),
    )

    assert selected["champion_id"] == "recommend_r1_open"
    assert selected["is_fallback"] is True
    assert "SHADOW 발송" in selected["reason"]


def test_authenticated_legacy_state_migrates_without_losing_history(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "champion_state.json"
    legacy = _legacy_state()
    raw = (json.dumps(
        legacy,
        ensure_ascii=False,
        sort_keys=True,
    ) + "\n").encode("utf-8")
    path.write_bytes(raw)
    monkeypatch.setattr(selector, "STATE_PATH", path)
    monkeypatch.setattr(selector, "MODELS", ())

    migrated = selector.run(
        pd.Timestamp("2026-07-26"),
        dry_run=False,
    )

    digest = hashlib.sha256(raw).hexdigest()
    preserved = tmp_path / f"champion_state.legacy.{digest}.json"
    assert preserved.read_bytes() == raw
    assert migrated["history"] == legacy["history"]
    assert migrated["slots"]["preopen"]["champion_id"] == (
        "recommend_r1_preopen"
    )
    assert migrated["slots"]["open"]["is_fallback"] is True
    assert migrated["slots"]["preopen"]["is_fallback"] is True
    assert migrated["config"] == selector._expected_config()
    assert migrated["migrations"][-1]["from_payload_sha256"] == (
        legacy["payload_sha256"]
    )
    assert migrated["migrations"][-1]["history_entries_preserved"] == 2
    assert selector._validate_state(
        json.loads(path.read_text(encoding="utf-8"))
    )


def test_legacy_state_preservation_rejects_symlink_backup(tmp_path):
    path = tmp_path / "champion_state.json"
    raw = b'{"legacy":true}\n'
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    backup = tmp_path / f"champion_state.legacy.{digest}.json"
    backup.symlink_to(path)

    with pytest.raises(
        selector.ChampionStateError,
        match="backup is unsafe",
    ):
        selector._preserve_legacy_state(path, raw)

    assert backup.is_symlink()
    assert path.read_bytes() == raw


def test_invalid_existing_state_aborts_without_overwrite_or_backup(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "champion_state.json"
    raw = b'{"slots":'
    path.write_bytes(raw)
    monkeypatch.setattr(selector, "STATE_PATH", path)
    monkeypatch.setattr(selector, "MODELS", ())

    with pytest.raises(selector.ChampionStateError, match="refusing overwrite"):
        selector.run(pd.Timestamp("2026-07-26"), dry_run=False)

    assert path.read_bytes() == raw
    assert not list(tmp_path.glob("champion_state.legacy.*.json"))
