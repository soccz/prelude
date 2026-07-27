from __future__ import annotations

import copy
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

import scripts.validate_dashboard_assets as dashboard_validation
from ops.artifact_provenance import with_manifest_digest
from scripts.build_dashboard import _encrypt_payload
from scripts.idea_validation_report import (
    IDEA_ARTIFACT_SCHEMA,
    report_payload_digest,
)
from scripts.validate_dashboard_assets import (
    DashboardAssetError,
    EXPECTED_ASSETS,
    validate_dashboard_asset_directory,
)


PASSPHRASE = "dashboard-validation-secret"
ASOF = date(2026, 7, 26)
NOW = datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)
GENERATED_AT = "2026-07-26T11:55:00+00:00"
GENERATION_ID = "123e4567-e89b-42d3-a456-426614174000"


def _idea_payload(policy: dict) -> dict:
    payload = {
        "schema": IDEA_ARTIFACT_SCHEMA,
        "asof": ASOF.isoformat(),
        "cutoff_timezone": "Asia/Seoul",
        "generated_at": "2026-07-26T20:55:00",
        "generated_at_utc": GENERATED_AT,
        "dashboard_generation_id": GENERATION_ID,
        "input_lineage": with_manifest_digest(
            {
                "schema_version": 3,
                "files": {"test": {"path": "test", "exists": False}},
                "generator_sources": {
                    "test.py": {
                        "path": "test.py",
                        "exists": True,
                        "size": 1,
                        "sha256": "a" * 64,
                    }
                },
                "contract": {"test": True},
            }
        ),
        "n_candidates": 2,
        "n_closed": 1,
        "policy_competition": policy,
    }
    payload["payload_sha256"] = report_payload_digest(payload)
    return payload


def _payloads() -> dict[str, dict]:
    policy = {
        "schema": "policy_competition.v2",
        "asof": ASOF.isoformat(),
        "run_id": "test-run",
        "input_manifest": {},
        "rows": [],
    }
    idea = _idea_payload(policy)
    summary = {
        "dashboard_generation_id": GENERATION_ID,
        "asof": f"{ASOF.isoformat()}T00:00:00",
        "asof_timezone": "Asia/Seoul",
        "generated_at_utc": GENERATED_AT,
        "channels": {
            "distribution": {
                "n_alerts_total": 0,
                "n_closed": 0,
                "n_pending": 0,
            },
            "preopen": {
                "n_alerts_total": 0,
                "n_closed": 0,
                "n_pending": 0,
            },
            "recommend": {
                "channel": "recommend",
                "n_alerts_total": 0,
            },
        },
        "idea_validation": copy.deepcopy(idea),
        "policy_competition": copy.deepcopy(policy),
        "champion_gate": {
            "asof": ASOF.isoformat(),
            "slots": [
                {
                    "slot": "open",
                    "champion_id": "recommend_r1_open",
                },
                {
                    "slot": "preopen",
                    "champion_id": "recommend_r1_preopen",
                },
            ],
        },
        "pump_hunter_v2": {
            "status": "empty",
            "rows_total": 0,
            "rows_closed": 0,
            "watchlist": [],
        },
    }
    return {
        "summary.json": summary,
        "history.json": {
            "dashboard_generation_id": GENERATION_ID,
            "asof": f"{ASOF.isoformat()}T00:00:00",
            "asof_timezone": "Asia/Seoul",
            "rows": [],
        },
        "accuracy.json": {
            "dashboard_generation_id": GENERATION_ID,
            "asof": f"{ASOF.isoformat()}T00:00:00",
            "asof_timezone": "Asia/Seoul",
            "window_days": 30,
            "distribution": {
                "rolling": [],
                "cum_pnl": [],
                "rolling_sharpe": [],
                "underwater": [],
                "monthly_returns": [],
            },
            "preopen": {
                "rolling": [],
                "cum_pnl": [],
                "rolling_sharpe": [],
                "underwater": [],
                "monthly_returns": [],
            },
            "btc_benchmark": [],
        },
        "idea_validation.json": idea,
        "findings.json": {
            "dashboard_generation_id": GENERATION_ID,
            "asof": ASOF.isoformat(),
            "asof_timezone": "Asia/Seoul",
            "generated_at_utc": GENERATED_AT,
            "champion_leaderboard": {
                "current_champion": {
                    "open": "recommend_r1_open",
                    "preopen": "recommend_r1_preopen",
                },
                "champion_state_identity": {
                    "sha256": "b" * 64,
                    "size": 1,
                },
                "rows": [],
            },
            "policy_competition": {
                "artifact_identity": {
                    "sha256": "c" * 64,
                    "size": 1,
                },
                "database": {},
                "rows": [],
            },
            "honest_caption": "diagnostic only",
            "magnitude_curve": {"thresholds_pct": []},
            "risk_reward": {"labels": []},
            "calibration": {"ideal_line": []},
            "backtest_pumps": {"pumps": []},
            "precursor_lift": {"features": []},
            "regime_baserate": {"regimes": []},
        },
    }


def _write_encrypted_assets(root: Path, payloads: dict[str, dict]) -> None:
    root.mkdir()
    for name in EXPECTED_ASSETS:
        (root / name).write_text(
            json.dumps(_encrypt_payload(payloads[name], PASSPHRASE)),
            encoding="utf-8",
        )


def _validate(root: Path):
    return validate_dashboard_asset_directory(
        root,
        passphrase=PASSPHRASE,
        expected_asof=ASOF,
        now=NOW,
        require_current_sources=False,
    )


def test_five_authenticated_current_assets_pass(tmp_path):
    asset_dir = tmp_path / "assets"
    expected = _payloads()
    _write_encrypted_assets(asset_dir, expected)

    actual = _validate(asset_dir)

    assert actual == expected


def test_plaintext_asset_is_rejected(tmp_path):
    asset_dir = tmp_path / "assets"
    _write_encrypted_assets(asset_dir, _payloads())
    (asset_dir / "summary.json").write_text(
        '{"asof":"2026-07-26"}',
        encoding="utf-8",
    )

    with pytest.raises(DashboardAssetError, match="envelope schema"):
        _validate(asset_dir)


def test_authenticated_ciphertext_tamper_is_rejected(tmp_path):
    asset_dir = tmp_path / "assets"
    _write_encrypted_assets(asset_dir, _payloads())
    path = asset_dir / "history.json"
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["ct"] = (
        ("A" if envelope["ct"][0] != "A" else "B") + envelope["ct"][1:]
    )
    path.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(DashboardAssetError, match="authentication failed"):
        _validate(asset_dir)


def test_wrong_passphrase_is_rejected(tmp_path):
    asset_dir = tmp_path / "assets"
    _write_encrypted_assets(asset_dir, _payloads())

    with pytest.raises(DashboardAssetError, match="authentication failed"):
        validate_dashboard_asset_directory(
            asset_dir,
            passphrase="different-dashboard-secret",
            expected_asof=ASOF,
            now=NOW,
            require_current_sources=False,
        )


def test_stale_generation_is_rejected(tmp_path):
    asset_dir = tmp_path / "assets"
    payloads = _payloads()
    payloads["summary.json"]["generated_at_utc"] = (
        "2026-07-26T01:00:00+00:00"
    )
    _write_encrypted_assets(asset_dir, payloads)

    with pytest.raises(DashboardAssetError, match="stale"):
        _validate(asset_dir)


def test_cross_asset_generation_mismatch_is_rejected(tmp_path):
    asset_dir = tmp_path / "assets"
    payloads = _payloads()
    payloads["summary.json"]["champion_gate"]["slots"][0][
        "champion_id"
    ] = "different"
    _write_encrypted_assets(asset_dir, payloads)

    with pytest.raises(DashboardAssetError, match="champion generations"):
        _validate(asset_dir)


def test_same_day_mixed_publish_generation_is_rejected(tmp_path):
    asset_dir = tmp_path / "assets"
    payloads = _payloads()
    payloads["history.json"]["dashboard_generation_id"] = (
        "223e4567-e89b-42d3-a456-426614174000"
    )
    _write_encrypted_assets(asset_dir, payloads)

    with pytest.raises(DashboardAssetError, match="publish generations"):
        _validate(asset_dir)


def test_shallow_empty_channel_payload_is_rejected(tmp_path):
    asset_dir = tmp_path / "assets"
    payloads = _payloads()
    payloads["summary.json"]["channels"]["distribution"] = {}
    _write_encrypted_assets(asset_dir, payloads)

    with pytest.raises(DashboardAssetError, match="n_alerts_total"):
        _validate(asset_dir)


def test_asset_symlink_is_rejected(tmp_path):
    asset_dir = tmp_path / "assets"
    _write_encrypted_assets(asset_dir, _payloads())
    target = asset_dir / "summary.json"
    outside = tmp_path / "outside.json"
    outside.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(outside)

    with pytest.raises(DashboardAssetError, match="regular non-symlink"):
        _validate(asset_dir)


def test_extra_asset_is_rejected(tmp_path):
    asset_dir = tmp_path / "assets"
    _write_encrypted_assets(asset_dir, _payloads())
    (asset_dir / "extra.json").write_text("{}", encoding="utf-8")

    with pytest.raises(DashboardAssetError, match="exactly the five"):
        _validate(asset_dir)


def test_asset_directory_change_during_validation_is_rejected(
    tmp_path,
    monkeypatch,
):
    asset_dir = tmp_path / "assets"
    _write_encrypted_assets(asset_dir, _payloads())
    original = dashboard_validation.strict_json_object
    changed = False

    def mutate_directory(path):
        nonlocal changed
        if not changed:
            changed = True
            (asset_dir / "extra.json").write_text("{}", encoding="utf-8")
        return original(path)

    monkeypatch.setattr(
        dashboard_validation,
        "strict_json_object",
        mutate_directory,
    )

    with pytest.raises(DashboardAssetError, match="exactly the five"):
        _validate(asset_dir)
