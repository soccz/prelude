from __future__ import annotations

import ast
import hashlib
import json
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import ops.recommendation_quality as recommendation_quality
from ops.artifact_provenance import (
    ArtifactSourceChangedError,
    ArtifactValidationError,
    file_set_identity,
    payload_digest,
    with_manifest_digest,
)
from ops.decision_policy import ACTIVE, WATCH_ONLY
from ops.recommendation_quality import (
    APPROVED_META_ARTIFACT_SHA256,
    CATEGORICAL_META_FEATURES,
    META_LEDGER_INPUT_KEYS,
    META_ARTIFACT_SCHEMA,
    META_FEATURES,
    META_TRAINING_LINEAGE_SCHEMA,
    MODEL_ID,
    MODEL_VERSION,
    NUMERIC_META_FEATURES,
    apply_recommendation_quality,
    inspect_trained_meta_model,
    load_trained_meta_model,
    meta_feature_schema_sha256,
    meta_runtime_versions,
    meta_training_row_schema_sha256,
    normalize_history,
)
from scripts import train_recommendation_meta


def _active_candidate() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "market": "KRW-AAA",
            "decision": ACTIVE,
            "idea_id": "dist_h6_b_s03_bull_quiet_v1",
            "setup_quality": "B_S03",
            "btc_regime": "bull_quiet",
            "expected_edge_pct": 1.0,
            "decision_reason": "base active",
        }
    ])


class _FakeModel:
    def __init__(self, p_win: float):
        self.p_win = p_win
        self.feature_names_in_ = np.asarray(META_FEATURES)
        self.classes_ = np.asarray([0, 1])

    def predict_proba(self, X):
        return np.array([[1.0 - self.p_win, self.p_win] for _ in range(len(X))])


def _versioned_artifact(tmp_path, *, p_win=0.1, status="CANDIDATE"):
    model = pickle.dumps(_FakeModel(p_win), protocol=pickle.HIGHEST_PROTOCOL)
    model_hash = hashlib.sha256(model).hexdigest()
    model_name = f"model.{model_hash}.pkl"
    (tmp_path / model_name).write_bytes(model)
    state = {
        "CANDIDATE": (True, False, "AWAITING_USER_APPROVAL"),
        "DEPLOYED": (True, True, "APPROVED"),
        "REJECTED": (False, False, "NOT_ELIGIBLE"),
    }
    validation_passed, deployable, promotion_status = state[status]
    dummy_digest = hashlib.sha256(b"ledger").hexdigest()
    ledgers = {
        key: {
            "path": f"output/{key}.csv",
            "exists": True,
            "size": 1,
            "sha256": dummy_digest,
        }
        for key in META_LEDGER_INPUT_KEYS
    }
    generators = file_set_identity(
        {
            relative: recommendation_quality.PROJECT_ROOT / relative
            for relative in recommendation_quality.META_TRAINING_GENERATOR_SOURCES
        },
        root=recommendation_quality.PROJECT_ROOT,
    )
    lineage = with_manifest_digest(
        {
            "schema_version": META_TRAINING_LINEAGE_SCHEMA,
            "ledger_inputs": with_manifest_digest(
                {"files": ledgers},
                digest_key="bundle_sha256",
            ),
            "generator_sources": with_manifest_digest(
                {"files": generators},
                digest_key="bundle_sha256",
            ),
            "training_rows": {
                "row_schema_sha256": meta_training_row_schema_sha256(),
                "rows_sha256": hashlib.sha256(b"rows").hexdigest(),
                "n_rows": 100,
                "n_dates": 20,
                "date_start": "2026-01-01",
                "date_end": "2026-01-20",
            },
        }
    )
    meta = {
        "artifact_schema": META_ARTIFACT_SCHEMA,
        "artifact_status": status,
        "validation_gate_passed": validation_passed,
        "promotion_status": promotion_status,
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "built_at": "2026-07-26T01:00:00+00:00",
        "deployable": deployable,
        "reason": "test",
        "target": "net_win",
        "threshold": 0.6,
        "model_file": model_name,
        "model_sha256": model_hash,
        "n_samples": 100,
        "date_range": {
            "start": "2026-01-01",
            "end": "2026-01-20",
        },
        "features": META_FEATURES,
        "numeric_features": NUMERIC_META_FEATURES,
        "categorical_features": CATEGORICAL_META_FEATURES,
        "feature_schema_sha256": meta_feature_schema_sha256(),
        "runtime_versions": meta_runtime_versions(),
        "training_lineage": lineage,
    }
    meta["artifact_sha256"] = payload_digest(
        meta,
        digest_key="artifact_sha256",
    )
    (tmp_path / "meta.json").write_text(
        json.dumps(meta, allow_nan=False),
        encoding="utf-8",
    )
    return meta, model_name


def _trainer_args(tmp_path, out_dir):
    ledger_paths = {}
    for key in META_LEDGER_INPUT_KEYS:
        path = tmp_path / f"{key}.csv"
        path.write_text("date,coin\n", encoding="utf-8")
        ledger_paths[key] = str(path)
    return SimpleNamespace(
        out_dir=str(out_dir),
        holdout_frac=0.25,
        min_samples=80,
        min_holdout=20,
        min_selected=5,
        out_validation_csv=str(tmp_path / "validation.csv"),
        out_validation_json=str(tmp_path / "validation.json"),
        paper_ledger=ledger_paths["paper_ledger_distribution"],
        paper_ledger_preopen=ledger_paths["paper_ledger_preopen"],
        shadow_ledger_distribution=ledger_paths["shadow_ledger_distribution"],
        shadow_ledger_preopen=ledger_paths["shadow_ledger_preopen"],
    )


def test_recommendation_quality_demotes_negative_matched_evidence():
    raw_history = pd.DataFrame([
        {
            "date": f"2026-05-{day:02d}",
            "coin": f"KRW-{day}",
            "decision": ACTIVE,
            "idea_id": "dist_h6_b_s03_bull_quiet_v1",
            "setup_quality": "B_S03",
            "btc_regime": "bull_quiet",
            "next_max_return_pct": 1.0,
            "next_close_return_pct": -2.0,
            "status": "closed",
        }
        for day in range(1, 9)
    ])
    history = normalize_history(raw_history, "distribution", "paper")

    out = apply_recommendation_quality(
        _active_candidate(),
        history,
        pd.Timestamp("2026-05-25 09:05"),
        "distribution",
    )

    row = out.iloc[0]
    assert row["base_decision"] == ACTIVE
    assert row["decision"] == WATCH_ONLY
    assert row["confidence_tier"] == "DOWNRANK"
    assert row["blocked_reason"] == "meta_filter_negative_evidence"
    assert row["evidence_n_closed"] == 8


def test_recommendation_quality_keeps_active_without_closed_history():
    out = apply_recommendation_quality(
        _active_candidate(),
        pd.DataFrame(),
        pd.Timestamp("2026-05-25 09:05"),
        "distribution",
    )

    row = out.iloc[0]
    assert row["decision"] == ACTIVE
    assert row["confidence_tier"] == "COLLECT"
    assert row["evidence_n_closed"] == 0


def test_recommendation_quality_ignores_same_day_history():
    raw_history = pd.DataFrame([
        {
            "date": "2026-05-25",
            "coin": f"KRW-{i}",
            "decision": ACTIVE,
            "idea_id": "dist_h6_b_s03_bull_quiet_v1",
            "setup_quality": "B_S03",
            "btc_regime": "bull_quiet",
            "next_max_return_pct": 1.0,
            "next_close_return_pct": -5.0,
            "status": "closed",
        }
        for i in range(8)
    ])
    history = normalize_history(raw_history, "distribution", "paper")

    out = apply_recommendation_quality(
        _active_candidate(),
        history,
        pd.Timestamp("2026-05-25 09:05"),
        "distribution",
    )

    assert out.iloc[0]["decision"] == ACTIVE
    assert out.iloc[0]["confidence_tier"] == "COLLECT"


def test_recommendation_quality_ignores_explicitly_open_outcomes():
    raw_history = pd.DataFrame([
        {
            "date": f"2026-05-{day:02d}",
            "coin": f"KRW-{day}",
            "decision": ACTIVE,
            "idea_id": "dist_h6_b_s03_bull_quiet_v1",
            "setup_quality": "B_S03",
            "btc_regime": "bull_quiet",
            "next_max_return_pct": 1.0,
            "next_close_return_pct": -5.0,
            "status": "open",
        }
        for day in range(1, 9)
    ])

    out = apply_recommendation_quality(
        _active_candidate(),
        normalize_history(raw_history, "distribution", "paper"),
        pd.Timestamp("2026-05-25 09:05"),
        "distribution",
    )

    assert out.iloc[0]["decision"] == ACTIVE
    assert out.iloc[0]["confidence_tier"] == "COLLECT"
    assert out.iloc[0]["evidence_n_closed"] == 0


def test_recommendation_quality_ignores_history_without_coin_identity():
    raw_history = pd.DataFrame([
        {
            "date": f"2026-05-{day:02d}",
            "coin": pd.NA,
            "decision": ACTIVE,
            "idea_id": "dist_h6_b_s03_bull_quiet_v1",
            "setup_quality": "B_S03",
            "btc_regime": "bull_quiet",
            "next_max_return_pct": 0.0,
            "next_close_return_pct": -2.0,
            "status": "closed",
        }
        for day in range(1, 9)
    ])

    out = apply_recommendation_quality(
        _active_candidate(),
        normalize_history(raw_history, "distribution", "paper"),
        pd.Timestamp("2026-05-20"),
        "distribution",
    )

    assert out.iloc[0]["decision"] == ACTIVE
    assert out.iloc[0]["evidence_n_closed"] == 0
    assert out.iloc[0]["confidence_tier"] == "COLLECT"


def test_recommendation_quality_records_non_deployed_model_score_without_demoting():
    artifact = {
        "meta": {
            "model_id": "meta_test",
            "model_version": "v1",
            "deployable": False,
            "threshold": 0.6,
        },
        "model": _FakeModel(0.1),
    }

    out = apply_recommendation_quality(
        _active_candidate(),
        pd.DataFrame(),
        pd.Timestamp("2026-05-25 09:05"),
        "distribution",
        trained_model=artifact,
    )

    row = out.iloc[0]
    assert row["decision"] == ACTIVE
    assert row["trained_model_status"] == "COLLECT"
    assert row["trained_model_p_win"] == 0.1


def test_recommendation_quality_deployed_model_can_downrank():
    artifact = {
        "meta": {
            "model_id": "meta_test",
            "model_version": "v1",
            "deployable": True,
            "threshold": 0.6,
        },
        "model": _FakeModel(0.1),
    }

    out = apply_recommendation_quality(
        _active_candidate(),
        pd.DataFrame(),
        pd.Timestamp("2026-05-25 09:05"),
        "distribution",
        trained_model=artifact,
    )

    row = out.iloc[0]
    assert row["decision"] == WATCH_ONLY
    assert row["confidence_tier"] == "MODEL_DOWNRANK"
    assert row["blocked_reason"] == "trained_meta_model_downrank"


def test_positive_deployed_model_cannot_override_negative_realized_evidence():
    raw_history = pd.DataFrame([
        {
            "date": f"2026-05-{day:02d}",
            "coin": f"KRW-{day}",
            "decision": ACTIVE,
            "idea_id": "dist_h6_b_s03_bull_quiet_v1",
            "setup_quality": "B_S03",
            "btc_regime": "bull_quiet",
            "next_max_return_pct": 1.0,
            "next_close_return_pct": -2.0,
            "status": "closed",
        }
        for day in range(1, 9)
    ])
    artifact = {
        "meta": {
            "model_id": "meta_test",
            "model_version": "v1",
            "deployable": True,
            "threshold": 0.6,
        },
        "model": _FakeModel(0.9),
    }

    out = apply_recommendation_quality(
        _active_candidate(),
        normalize_history(raw_history, "distribution", "paper"),
        pd.Timestamp("2026-05-25 09:05"),
        "distribution",
        trained_model=artifact,
    )

    row = out.iloc[0]
    assert row["decision"] == WATCH_ONLY
    assert row["confidence_tier"] == "DOWNRANK"
    assert row["blocked_reason"] == "meta_filter_negative_evidence"
    assert row["trained_model_p_win"] == 0.9


def test_legacy_unbound_artifact_never_crosses_pickle_boundary(tmp_path, monkeypatch):
    (tmp_path / "meta.json").write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "model_version": MODEL_VERSION,
                "deployable": True,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "model.pkl").write_bytes(b"not safe to execute")
    monkeypatch.setattr(
        "ops.recommendation_quality.pickle.loads",
        lambda _: pytest.fail("legacy pickle must not be loaded"),
    )

    loaded = load_trained_meta_model(tmp_path)

    assert loaded is not None
    assert loaded["model"] is None
    assert loaded["artifact_status"] == "LEGACY_UNBOUND"
    assert loaded["meta"]["deployable"] is False
    assert loaded["meta"]["declared_deployable"] is True


@pytest.mark.parametrize(
    "raw",
    [
        b'{"artifact_schema":"x","artifact_schema":"y"}',
        b'{"artifact_schema":"x","threshold":NaN}',
    ],
)
def test_meta_loader_rejects_non_strict_json(tmp_path, raw):
    (tmp_path / "meta.json").write_bytes(raw)

    with pytest.raises(ArtifactValidationError):
        load_trained_meta_model(tmp_path)


def test_meta_loader_rejects_dangling_metadata_symlink(tmp_path):
    (tmp_path / "meta.json").symlink_to(tmp_path / "missing.json")

    with pytest.raises((ArtifactValidationError, ArtifactSourceChangedError)):
        load_trained_meta_model(tmp_path)


def test_disabled_meta_filter_never_reads_or_validates_artifact(monkeypatch):
    monkeypatch.setattr(
        "ops.recommendation_quality._read_trained_meta_model",
        lambda *_args, **_kwargs: pytest.fail(
            "disabled meta filter must not touch artifact"
        ),
    )

    assert load_trained_meta_model("/does/not/matter", enabled=False) is None


@pytest.mark.parametrize(
    "relative",
    [
        "scripts/predict_today_distribution.py",
        "scripts/predict_preopen_trigger.py",
    ],
)
def test_scoring_callers_bind_artifact_loading_to_meta_filter_flag(relative):
    tree = ast.parse(
        (recommendation_quality.PROJECT_ROOT / relative).read_text(
            encoding="utf-8"
        )
    )
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_trained_meta_model"
    ]

    assert len(calls) == 1
    enabled = next(
        keyword.value
        for keyword in calls[0].keywords
        if keyword.arg == "enabled"
    )
    assert ast.unparse(enabled) == "not args.disable_meta_filter"


def test_unapproved_content_bound_artifact_is_not_unpickled(tmp_path, monkeypatch):
    meta, _ = _versioned_artifact(tmp_path)
    monkeypatch.setattr(
        "ops.recommendation_quality.pickle.loads",
        lambda _: pytest.fail("unapproved pickle must not be loaded"),
    )

    loaded = load_trained_meta_model(tmp_path)

    assert loaded is not None
    assert loaded["meta"]["artifact_sha256"] == meta["artifact_sha256"]
    assert loaded["model"] is None
    assert loaded["artifact_status"] == "UNAPPROVED"


def test_loader_rejects_model_hash_mismatch_before_unpickle(tmp_path, monkeypatch):
    _, model_name = _versioned_artifact(tmp_path)
    (tmp_path / model_name).write_bytes(b"tampered")
    monkeypatch.setattr(
        "ops.recommendation_quality.pickle.loads",
        lambda _: pytest.fail("hash-mismatched pickle must not be loaded"),
    )

    with pytest.raises(ArtifactValidationError, match="model hash mismatch"):
        load_trained_meta_model(tmp_path)


def test_loader_rejects_stale_generator_source_lineage(tmp_path, monkeypatch):
    generator = tmp_path / "generator.py"
    generator.write_text("VERSION = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        recommendation_quality,
        "META_TRAINING_GENERATOR_SOURCES",
        (str(generator),),
    )
    _versioned_artifact(tmp_path)

    generator.write_text("VERSION = 2\n", encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="source lineage is stale"):
        load_trained_meta_model(tmp_path)


def test_loader_rejects_training_row_count_contract_mismatch(tmp_path):
    meta, _ = _versioned_artifact(tmp_path)
    lineage = dict(meta["training_lineage"])
    rows = dict(lineage["training_rows"])
    rows["n_rows"] = 99
    lineage["training_rows"] = rows
    lineage.pop("manifest_sha256")
    meta["training_lineage"] = with_manifest_digest(lineage)
    meta["artifact_sha256"] = payload_digest(
        meta,
        digest_key="artifact_sha256",
    )
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="row counts"):
        load_trained_meta_model(tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("threshold", 1.1, "threshold"),
        ("features", ["wrong"], "feature contract"),
        ("model_version", "unexpected", "model_version"),
        ("validation_gate_passed", False, "promotion state"),
    ],
)
def test_loader_rejects_invalid_live_contract(
    tmp_path,
    field,
    value,
    message,
):
    meta, _ = _versioned_artifact(tmp_path)
    meta[field] = value
    meta["artifact_sha256"] = payload_digest(
        meta,
        digest_key="artifact_sha256",
    )
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match=message):
        load_trained_meta_model(tmp_path)


def test_only_explicitly_approved_complete_artifact_is_unpickled(
    tmp_path,
    monkeypatch,
):
    meta, _ = _versioned_artifact(tmp_path, status="DEPLOYED")
    monkeypatch.setitem(
        APPROVED_META_ARTIFACT_SHA256,
        (MODEL_ID, MODEL_VERSION),
        meta["artifact_sha256"],
    )

    loaded = load_trained_meta_model(tmp_path)
    assert loaded is not None
    assert loaded["artifact_status"] == "DEPLOYED"

    out = apply_recommendation_quality(
        _active_candidate(),
        pd.DataFrame(),
        pd.Timestamp("2026-05-25 09:05"),
        "distribution",
        trained_model=loaded,
    )
    assert out.iloc[0]["decision"] == WATCH_ONLY
    assert out.iloc[0]["trained_model_status"] == "DEPLOYED"


def test_inspection_never_unpickles_even_an_approved_artifact(
    tmp_path,
    monkeypatch,
):
    meta, _ = _versioned_artifact(tmp_path, status="DEPLOYED")
    monkeypatch.setitem(
        APPROVED_META_ARTIFACT_SHA256,
        (MODEL_ID, MODEL_VERSION),
        meta["artifact_sha256"],
    )
    monkeypatch.setattr(
        "ops.recommendation_quality.pickle.loads",
        lambda _: pytest.fail("inspection must never execute pickle"),
    )

    inspected = inspect_trained_meta_model(tmp_path)

    assert inspected is not None
    assert inspected["artifact_status"] == "DEPLOYED"
    assert inspected["model"] is None
    assert inspected["meta"]["deployable"] is True


def test_trainer_publishes_content_bound_candidate_without_auto_deploy(
    tmp_path,
    monkeypatch,
):
    dates = pd.date_range("2026-01-01", periods=100, freq="D")
    target = np.arange(100) % 2
    closed = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "date_dt": dates,
            "channel": "distribution",
            "coin": [f"KRW-{i:03d}" for i in range(100)],
            "net_pnl_pct": np.where(target == 1, 1.0, -1.0),
            "target_net_win": target,
            "outcome_contract": "tp5_sl3_ordered_first_passage_net",
            "promotion_eligible": True,
            "expected_edge_pct": target,
            "setup_quality": "B_S03",
            "btc_regime": "bull_quiet",
        }
    )
    monkeypatch.setattr(
        train_recommendation_meta,
        "build_training_data",
        lambda _args: closed,
    )
    out_dir = tmp_path / "artifact"
    args = _trainer_args(tmp_path, out_dir)

    meta = train_recommendation_meta.train_and_write(args)

    assert meta["validation_gate_passed"] is True
    assert meta["artifact_status"] == "CANDIDATE"
    assert meta["promotion_status"] == "AWAITING_USER_APPROVAL"
    assert meta["deployable"] is False
    assert meta["model_file"] == f"model.{meta['model_sha256']}.pkl"
    assert (out_dir / meta["model_file"]).is_file()
    assert not (out_dir / "model.pkl").exists()
    assert payload_digest(meta, digest_key="artifact_sha256") == meta["artifact_sha256"]
    lineage = meta["training_lineage"]
    assert lineage["training_rows"] == train_recommendation_meta._strict_training_rows(
        closed
    )[1]
    assert set(lineage["ledger_inputs"]["files"]) == set(META_LEDGER_INPUT_KEYS)
    assert set(lineage["generator_sources"]["files"]) == set(
        recommendation_quality.META_TRAINING_GENERATOR_SOURCES
    )

    loaded = load_trained_meta_model(out_dir)
    assert loaded is not None
    assert loaded["artifact_status"] == "UNAPPROVED"
    assert loaded["model"] is None


def test_trainer_publishes_rejection_instead_of_fitting_one_class(
    tmp_path,
    monkeypatch,
):
    dates = pd.date_range("2026-01-01", periods=100, freq="D")
    closed = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "date_dt": dates,
            "channel": "distribution",
            "coin": [f"KRW-{i:03d}" for i in range(100)],
            "net_pnl_pct": 1.0,
            "target_net_win": 1,
            "outcome_contract": "tp5_sl3_ordered_first_passage_net",
            "promotion_eligible": True,
        }
    )
    monkeypatch.setattr(
        train_recommendation_meta,
        "build_training_data",
        lambda _args: closed,
    )
    out_dir = tmp_path / "artifact"
    validation_csv = tmp_path / "validation.csv"
    args = _trainer_args(tmp_path, out_dir)

    meta = train_recommendation_meta.train_and_write(args)

    assert meta["artifact_status"] == "REJECTED"
    assert meta["deployable"] is False
    assert meta["model_file"] is None
    assert "one class" in meta["reason"]
    assert validation_csv.read_text(encoding="utf-8").startswith("date,channel")
    loaded = load_trained_meta_model(out_dir)
    assert loaded is not None
    assert loaded["artifact_status"] == "REJECTED"


def test_training_data_excludes_unordered_proxy_outcomes(monkeypatch):
    enriched = pd.DataFrame(
        [
            {
                "date": "2026-07-01",
                "date_dt": pd.Timestamp("2026-07-01"),
                "channel": "distribution",
                "coin": "KRW-PROXY",
                "net_pnl_pct": 4.85,
                "promotion_eligible": False,
            },
            {
                "date": "2026-07-02",
                "date_dt": pd.Timestamp("2026-07-02"),
                "channel": "distribution",
                "coin": "KRW-ORDERED",
                "net_pnl_pct": -3.15,
                "promotion_eligible": np.bool_(True),
            },
            {
                "date": "bad-date",
                "date_dt": pd.NaT,
                "channel": "distribution",
                "coin": "KRW-NODATE",
                "net_pnl_pct": 1.0,
                "promotion_eligible": True,
            },
        ]
    )
    monkeypatch.setattr(
        train_recommendation_meta,
        "load_candidate_ledger",
        lambda _args: pd.DataFrame({"placeholder": [1]}),
    )
    monkeypatch.setattr(
        train_recommendation_meta,
        "add_result_columns",
        lambda _frame: enriched,
    )

    result = train_recommendation_meta.build_training_data(SimpleNamespace())

    assert result["coin"].tolist() == ["KRW-ORDERED"]
    assert result["target_net_win"].tolist() == [0]


def test_training_snapshot_rejects_ledger_mutation_during_read(
    tmp_path,
    monkeypatch,
):
    args = _trainer_args(tmp_path, tmp_path / "candidate")

    def mutate_during_read(_args):
        Path(args.paper_ledger).write_text(
            "date,coin\n2026-07-26,KRW-TAMPER\n",
            encoding="utf-8",
        )
        return pd.DataFrame()

    monkeypatch.setattr(
        train_recommendation_meta,
        "build_training_data",
        mutate_during_read,
    )

    with pytest.raises(RuntimeError, match="changed during snapshot"):
        train_recommendation_meta._build_training_snapshot(args)


def test_concurrent_trainers_for_same_candidate_slot_are_serialized(
    tmp_path,
    monkeypatch,
):
    state = {"active": 0, "max_active": 0}
    state_lock = threading.Lock()

    def fake_train(_args, _out_dir):
        with state_lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        time.sleep(0.03)
        with state_lock:
            state["active"] -= 1
        return {"ok": True}

    monkeypatch.setattr(
        train_recommendation_meta,
        "_train_and_write_locked",
        fake_train,
    )
    args = SimpleNamespace(out_dir=str(tmp_path / "candidate"))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(train_recommendation_meta.train_and_write, [args, args]))

    assert results == [{"ok": True}, {"ok": True}]
    assert state["max_active"] == 1
