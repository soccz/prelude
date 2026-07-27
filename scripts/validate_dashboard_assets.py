#!/usr/bin/env python3
"""Fail-closed validation for the five encrypted dashboard publish assets."""
from __future__ import annotations

import argparse
import base64
import binascii
import os
import stat
import sys
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cryptography.exceptions import InvalidSignature  # noqa: E402
from cryptography.hazmat.primitives import hashes, hmac as crypto_hmac  # noqa: E402
from cryptography.hazmat.primitives import padding  # noqa: E402
from cryptography.hazmat.primitives.ciphers import (  # noqa: E402
    Cipher,
    algorithms,
    modes,
)
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC  # noqa: E402

from ops.artifact_provenance import (  # noqa: E402
    ArtifactValidationError,
    canonical_json_bytes,
    file_identity,
    strict_json_object,
    strict_json_object_bytes,
)
from ops.champion_selector import (  # noqa: E402
    ChampionStateError,
    load_champion_state_artifact,
)
from ops.policy_competition import (  # noqa: E402
    PolicyArtifactError,
    load_policy_artifact,
)
from scripts.build_dashboard import (  # noqa: E402
    MIN_DASHBOARD_PASSPHRASE_LENGTH,
    PBKDF2_ITERATIONS,
    resolve_dashboard_passphrase,
)
from scripts.idea_validation_report import (  # noqa: E402
    IdeaArtifactError,
    validate_idea_validation_payload,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_ASSETS = (
    "summary.json",
    "history.json",
    "accuracy.json",
    "idea_validation.json",
    "findings.json",
)
ENVELOPE_KEYS = frozenset(
    {
        "encrypted",
        "version",
        "kdf",
        "cipher",
        "iterations",
        "salt",
        "iv",
        "ct",
        "mac",
    }
)
MAX_ENCRYPTED_ASSET_BYTES = 64 * 1024 * 1024
MAX_GENERATION_AGE = timedelta(hours=6)
MAX_FUTURE_SKEW = timedelta(minutes=5)


class DashboardAssetError(ArtifactValidationError):
    """A generated dashboard asset is unsafe, corrupt, stale, or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DashboardAssetError(message)


def _strict_base64(
    value: Any,
    *,
    field: str,
    expected_size: int | None = None,
) -> bytes:
    if type(value) is not str or not value:
        raise DashboardAssetError(f"dashboard envelope {field} must be base64 text")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise DashboardAssetError(
            f"dashboard envelope {field} is invalid base64"
        ) from exc
    if base64.b64encode(decoded).decode("ascii") != value:
        raise DashboardAssetError(
            f"dashboard envelope {field} is not canonical base64"
        )
    if expected_size is not None and len(decoded) != expected_size:
        raise DashboardAssetError(
            f"dashboard envelope {field} has invalid byte length"
        )
    return decoded


def decrypt_dashboard_envelope(
    envelope: Mapping[str, Any],
    passphrase: str,
    *,
    source: str | Path = "<dashboard asset>",
) -> dict[str, Any]:
    """Authenticate, decrypt, unpad, and strictly decode one envelope."""
    if type(envelope) is not dict or set(envelope) != ENVELOPE_KEYS:
        raise DashboardAssetError(
            f"dashboard envelope schema mismatch: {source}"
        )
    _require(envelope["encrypted"] is True, "dashboard asset is not encrypted")
    _require(
        type(envelope["version"]) is int and envelope["version"] == 1,
        "unsupported dashboard envelope version",
    )
    _require(
        envelope["kdf"] == "PBKDF2-HMAC-SHA256",
        "unsupported dashboard envelope KDF",
    )
    _require(
        envelope["cipher"] == "AES-256-CBC-HMAC-SHA256",
        "unsupported dashboard envelope cipher",
    )
    _require(
        type(envelope["iterations"]) is int
        and envelope["iterations"] == PBKDF2_ITERATIONS,
        "dashboard envelope PBKDF2 iteration mismatch",
    )
    _require(
        type(passphrase) is str
        and len(passphrase) >= MIN_DASHBOARD_PASSPHRASE_LENGTH
        and passphrase == passphrase.strip(),
        "dashboard validation passphrase is invalid",
    )

    salt = _strict_base64(envelope["salt"], field="salt", expected_size=16)
    iv = _strict_base64(envelope["iv"], field="iv", expected_size=16)
    ciphertext = _strict_base64(envelope["ct"], field="ct")
    mac = _strict_base64(envelope["mac"], field="mac", expected_size=32)
    _require(
        len(ciphertext) > 0 and len(ciphertext) % 16 == 0,
        "dashboard ciphertext length is invalid",
    )

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=64,
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
    )
    key_material = kdf.derive(passphrase.encode("utf-8"))
    aes_key, mac_key = key_material[:32], key_material[32:]
    verifier = crypto_hmac.HMAC(mac_key, hashes.SHA256())
    verifier.update(salt + iv + ciphertext)
    try:
        verifier.verify(mac)
    except InvalidSignature as exc:
        raise DashboardAssetError(
            f"dashboard envelope authentication failed: {source}"
        ) from exc

    decryptor = Cipher(
        algorithms.AES(aes_key),
        modes.CBC(iv),
    ).decryptor()
    padded = decryptor.update(ciphertext) + decryptor.finalize()
    unpadder = padding.PKCS7(128).unpadder()
    try:
        plaintext = unpadder.update(padded) + unpadder.finalize()
    except ValueError as exc:
        raise DashboardAssetError(
            f"dashboard envelope padding is invalid: {source}"
        ) from exc
    try:
        return strict_json_object_bytes(plaintext, source=source)
    except ArtifactValidationError as exc:
        raise DashboardAssetError(
            f"dashboard plaintext JSON is invalid: {source}"
        ) from exc


def _canonical_asof(value: Any, *, field: str) -> date:
    if type(value) is not str:
        raise DashboardAssetError(f"{field} must be a canonical date")
    try:
        if len(value) == 10:
            parsed = date.fromisoformat(value)
        else:
            timestamp = datetime.fromisoformat(value)
            if (
                timestamp.hour,
                timestamp.minute,
                timestamp.second,
                timestamp.microsecond,
            ) != (0, 0, 0, 0):
                raise ValueError
            parsed = timestamp.date()
    except ValueError as exc:
        raise DashboardAssetError(f"{field} must be a canonical date") from exc
    accepted = {parsed.isoformat(), f"{parsed.isoformat()}T00:00:00"}
    if value not in accepted:
        raise DashboardAssetError(f"{field} must be a canonical date")
    return parsed


def _validate_generated_at(
    value: Any,
    *,
    field: str,
    now: datetime,
) -> None:
    if type(value) is not str:
        raise DashboardAssetError(f"{field} must be a timezone-aware datetime")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise DashboardAssetError(
            f"{field} must be a timezone-aware datetime"
        ) from exc
    if parsed.tzinfo is None:
        raise DashboardAssetError(f"{field} must be timezone-aware")
    utc_value = parsed.astimezone(timezone.utc)
    if utc_value > now + MAX_FUTURE_SKEW:
        raise DashboardAssetError(f"{field} is future-dated")
    if utc_value < now - MAX_GENERATION_AGE:
        raise DashboardAssetError(f"{field} is stale")


def _require_object(
    payload: Mapping[str, Any],
    key: str,
    *,
    context: str,
) -> dict[str, Any]:
    value = payload.get(key)
    if type(value) is not dict:
        raise DashboardAssetError(f"{context}.{key} must be an object")
    return value


def _require_list(
    payload: Mapping[str, Any],
    key: str,
    *,
    context: str,
) -> list[Any]:
    value = payload.get(key)
    if type(value) is not list:
        raise DashboardAssetError(f"{context}.{key} must be an array")
    return value


def _require_nonnegative_int(
    payload: Mapping[str, Any],
    key: str,
    *,
    context: str,
) -> int:
    value = payload.get(key)
    if type(value) is not int or value < 0:
        raise DashboardAssetError(
            f"{context}.{key} must be a non-negative integer"
        )
    return value


def _require_keys(
    payload: Mapping[str, Any],
    keys: set[str],
    *,
    context: str,
) -> None:
    missing = sorted(keys - set(payload))
    if missing:
        raise DashboardAssetError(
            f"{context} schema is missing required keys: {missing}"
        )


def _generation_id(payload: Mapping[str, Any], *, context: str) -> str:
    value = payload.get("dashboard_generation_id")
    if type(value) is not str:
        raise DashboardAssetError(
            f"{context}.dashboard_generation_id must be a canonical UUIDv4"
        )
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise DashboardAssetError(
            f"{context}.dashboard_generation_id must be a canonical UUIDv4"
        ) from exc
    if parsed.version != 4 or str(parsed) != value:
        raise DashboardAssetError(
            f"{context}.dashboard_generation_id must be a canonical UUIDv4"
        )
    return value


def _validate_common_asof(
    payload: Mapping[str, Any],
    *,
    context: str,
    expected_asof: date,
    timezone_key: str = "asof_timezone",
) -> None:
    actual = _canonical_asof(payload.get("asof"), field=f"{context}.asof")
    if actual != expected_asof:
        raise DashboardAssetError(f"{context}.asof does not match publish cutoff")
    if payload.get(timezone_key) != "Asia/Seoul":
        raise DashboardAssetError(
            f"{context}.{timezone_key} must be Asia/Seoul"
        )


def _validate_summary(
    payload: dict[str, Any],
    *,
    expected_asof: date,
    now: datetime,
) -> None:
    _validate_common_asof(
        payload,
        context="summary",
        expected_asof=expected_asof,
    )
    _validate_generated_at(
        payload.get("generated_at_utc"),
        field="summary.generated_at_utc",
        now=now,
    )
    channels = _require_object(payload, "channels", context="summary")
    if set(channels) != {"distribution", "preopen", "recommend"}:
        raise DashboardAssetError("summary.channels schema mismatch")
    if any(type(value) is not dict for value in channels.values()):
        raise DashboardAssetError("summary channel payload must be an object")
    for channel, value in channels.items():
        _require_nonnegative_int(
            value,
            "n_alerts_total",
            context=f"summary.channels.{channel}",
        )
    for channel in ("distribution", "preopen"):
        value = channels[channel]
        _require_nonnegative_int(
            value,
            "n_closed",
            context=f"summary.channels.{channel}",
        )
        _require_nonnegative_int(
            value,
            "n_pending",
            context=f"summary.channels.{channel}",
        )
    if channels["recommend"].get("channel") != "recommend":
        raise DashboardAssetError(
            "summary.channels.recommend.channel must be recommend"
        )
    for key in (
        "idea_validation",
        "policy_competition",
        "champion_gate",
        "pump_hunter_v2",
    ):
        value = _require_object(payload, key, context="summary")
        if not value:
            raise DashboardAssetError(f"summary.{key} must not be empty")
    _require_keys(
        payload["policy_competition"],
        {"schema", "asof", "run_id", "input_manifest", "rows"},
        context="summary.policy_competition",
    )
    _require_keys(
        payload["champion_gate"],
        {"asof", "slots"},
        context="summary.champion_gate",
    )
    _require_keys(
        payload["pump_hunter_v2"],
        {"status", "rows_total", "rows_closed", "watchlist"},
        context="summary.pump_hunter_v2",
    )


def _validate_history(
    payload: dict[str, Any],
    *,
    expected_asof: date,
) -> None:
    _validate_common_asof(
        payload,
        context="history",
        expected_asof=expected_asof,
    )
    rows = _require_list(payload, "rows", context="history")
    for index, row in enumerate(rows):
        if type(row) is not dict:
            raise DashboardAssetError(f"history row {index} must be an object")
        _require_keys(
            row,
            {"date", "channel", "coin", "status"},
            context=f"history.rows[{index}]",
        )
        row_date = _canonical_asof(
            row.get("date"),
            field=f"history.rows[{index}].date",
        )
        if row_date > expected_asof:
            raise DashboardAssetError(f"history row {index} is future-dated")


def _validate_accuracy(
    payload: dict[str, Any],
    *,
    expected_asof: date,
) -> None:
    _validate_common_asof(
        payload,
        context="accuracy",
        expected_asof=expected_asof,
    )
    window = payload.get("window_days")
    if type(window) is not int or window <= 0:
        raise DashboardAssetError("accuracy.window_days must be positive integer")
    for channel in ("distribution", "preopen"):
        channel_payload = _require_object(
            payload,
            channel,
            context="accuracy",
        )
        for key in (
            "rolling",
            "cum_pnl",
            "rolling_sharpe",
            "underwater",
            "monthly_returns",
        ):
            _require_list(
                channel_payload,
                key,
                context=f"accuracy.{channel}",
            )
    _require_list(payload, "btc_benchmark", context="accuracy")


def _validate_idea(
    payload: dict[str, Any],
    *,
    expected_asof: date,
    now: datetime,
    require_current_sources: bool,
) -> None:
    try:
        validate_idea_validation_payload(
            payload,
            asof=expected_asof.isoformat(),
            require_current=require_current_sources,
        )
    except IdeaArtifactError as exc:
        raise DashboardAssetError(str(exc)) from exc
    actual = _canonical_asof(
        payload.get("asof"),
        field="idea_validation.asof",
    )
    if actual != expected_asof:
        raise DashboardAssetError(
            "idea_validation.asof does not match publish cutoff"
        )
    if payload.get("cutoff_timezone") != "Asia/Seoul":
        raise DashboardAssetError(
            "idea_validation.cutoff_timezone must be Asia/Seoul"
        )
    _validate_generated_at(
        payload.get("generated_at_utc"),
        field="idea_validation.generated_at_utc",
        now=now,
    )


def _validate_findings(
    payload: dict[str, Any],
    *,
    expected_asof: date,
    now: datetime,
) -> None:
    _validate_common_asof(
        payload,
        context="findings",
        expected_asof=expected_asof,
    )
    _validate_generated_at(
        payload.get("generated_at_utc"),
        field="findings.generated_at_utc",
        now=now,
    )
    for key in (
        "champion_leaderboard",
        "policy_competition",
        "magnitude_curve",
        "risk_reward",
        "calibration",
        "backtest_pumps",
        "precursor_lift",
        "regime_baserate",
    ):
        value = _require_object(payload, key, context="findings")
        if not value:
            raise DashboardAssetError(f"findings.{key} must not be empty")
    _require_keys(
        payload["champion_leaderboard"],
        {"current_champion", "champion_state_identity", "rows"},
        context="findings.champion_leaderboard",
    )
    _require_keys(
        payload["policy_competition"],
        {"artifact_identity", "database", "rows"},
        context="findings.policy_competition",
    )
    _require_list(
        payload["champion_leaderboard"],
        "rows",
        context="findings.champion_leaderboard",
    )
    _require_list(
        payload["policy_competition"],
        "rows",
        context="findings.policy_competition",
    )
    _require_list(
        payload["magnitude_curve"],
        "thresholds_pct",
        context="findings.magnitude_curve",
    )
    _require_list(
        payload["risk_reward"],
        "labels",
        context="findings.risk_reward",
    )
    _require_list(
        payload["backtest_pumps"],
        "pumps",
        context="findings.backtest_pumps",
    )
    _require_list(
        payload["precursor_lift"],
        "features",
        context="findings.precursor_lift",
    )
    _require_list(
        payload["regime_baserate"],
        "regimes",
        context="findings.regime_baserate",
    )
    caption = payload.get("honest_caption")
    if type(caption) is not str or not caption.strip():
        raise DashboardAssetError("findings.honest_caption must be text")


def _validate_cross_asset_identity(
    payloads: Mapping[str, dict[str, Any]],
) -> None:
    summary = payloads["summary.json"]
    idea = payloads["idea_validation.json"]
    findings = payloads["findings.json"]
    embedded_idea = summary["idea_validation"]
    generation_ids = {
        name: _generation_id(payload, context=name)
        for name, payload in payloads.items()
    }
    if len(set(generation_ids.values())) != 1:
        raise DashboardAssetError(
            "dashboard assets belong to different publish generations"
        )
    if (
        embedded_idea.get("input_lineage") != idea.get("input_lineage")
        or embedded_idea.get("n_candidates") != idea.get("n_candidates")
        or embedded_idea.get("n_closed") != idea.get("n_closed")
    ):
        raise DashboardAssetError(
            "summary and dedicated idea-validation generations disagree"
        )
    summary_policy = summary["policy_competition"]
    idea_policy = idea.get("policy_competition")
    if (
        type(idea_policy) is not dict
        or canonical_json_bytes(summary_policy)
        != canonical_json_bytes(idea_policy)
    ):
        raise DashboardAssetError(
            "summary and idea-validation policy generations disagree"
        )
    champion_gate = summary["champion_gate"]
    leaderboard = findings["champion_leaderboard"]
    summary_champions = {
        str(row.get("slot")): row.get("champion_id")
        for row in champion_gate.get("slots", [])
        if type(row) is dict
    }
    if summary_champions != leaderboard.get("current_champion"):
        raise DashboardAssetError(
            "summary and findings champion generations disagree"
        )


def _validate_current_sources(
    payloads: Mapping[str, dict[str, Any]],
    *,
    expected_asof: date,
) -> None:
    try:
        policy = load_policy_artifact(
            PROJECT_ROOT / "output/policy_competition_summary.json",
            csv_path=PROJECT_ROOT / "output/policy_competition_summary.csv",
            db_path=PROJECT_ROOT / "data/policy_competition.db",
            asof=expected_asof.isoformat(),
            require_exact_asof=True,
            require_current=True,
            candle_db=PROJECT_ROOT / "data/upbit_d1.db",
        )
    except PolicyArtifactError as exc:
        raise DashboardAssetError(
            "current policy competition artifact is invalid"
        ) from exc
    for container in (
        payloads["summary.json"],
        payloads["idea_validation.json"],
    ):
        embedded = container.get("policy_competition")
        if (
            type(embedded) is not dict
            or canonical_json_bytes(embedded) != canonical_json_bytes(policy)
        ):
            raise DashboardAssetError(
                "dashboard policy payload is not the current strict artifact"
            )

    try:
        champion = load_champion_state_artifact(
            PROJECT_ROOT / "output/champion_state.json",
            expected_asof=expected_asof.isoformat(),
        )
    except ChampionStateError as exc:
        raise DashboardAssetError(
            "current champion state is invalid"
        ) from exc
    if champion is None:
        raise DashboardAssetError("current champion state is missing")
    findings_identity = payloads["findings.json"]["champion_leaderboard"].get(
        "champion_state_identity"
    )
    if (
        type(findings_identity) is not dict
        or findings_identity.get("sha256") != champion.identity["sha256"]
        or findings_identity.get("size") != champion.identity["size"]
    ):
        raise DashboardAssetError(
            "findings champion identity is not the current state"
        )
    policy_identity = payloads["findings.json"]["policy_competition"].get(
        "artifact_identity"
    )
    policy_path = PROJECT_ROOT / "output/policy_competition_summary.json"
    current_policy_identity = file_identity(
        policy_path,
        root=PROJECT_ROOT,
    )
    if (
        type(policy_identity) is not dict
        or policy_identity.get("sha256")
        != current_policy_identity.get("sha256")
        or policy_identity.get("size") != current_policy_identity.get("size")
    ):
        raise DashboardAssetError(
            "findings policy identity is not the current artifact"
        )


def _validated_directory_entries(asset_dir: Path) -> dict[str, Path]:
    try:
        directory_stat = asset_dir.lstat()
    except OSError as exc:
        raise DashboardAssetError(
            f"dashboard asset directory cannot be inspected: {asset_dir}"
        ) from exc
    if not stat.S_ISDIR(directory_stat.st_mode):
        raise DashboardAssetError(
            f"dashboard asset directory must be a real directory: {asset_dir}"
        )
    try:
        entries = {entry.name: entry for entry in asset_dir.iterdir()}
    except OSError as exc:
        raise DashboardAssetError(
            f"dashboard asset directory cannot be listed: {asset_dir}"
        ) from exc
    if set(entries) != set(EXPECTED_ASSETS):
        raise DashboardAssetError(
            "dashboard asset directory must contain exactly the five assets"
        )
    for name, path in entries.items():
        try:
            value = path.lstat()
        except OSError as exc:
            raise DashboardAssetError(
                f"dashboard asset cannot be inspected: {name}"
            ) from exc
        if not stat.S_ISREG(value.st_mode):
            raise DashboardAssetError(
                f"dashboard asset must be a regular non-symlink file: {name}"
            )
        if value.st_size <= 0 or value.st_size > MAX_ENCRYPTED_ASSET_BYTES:
            raise DashboardAssetError(
                f"dashboard asset size is invalid: {name}"
            )
    return entries


def _asset_generation_token(
    asset_dir: Path,
    entries: Mapping[str, Path],
) -> tuple[tuple[int, ...], tuple[tuple[str, tuple[int, ...]], ...]]:
    """Bind validation to one directory and one named generation per asset."""

    def token(value: os.stat_result) -> tuple[int, ...]:
        return (
            int(value.st_dev),
            int(value.st_ino),
            int(value.st_mode),
            int(value.st_nlink),
            int(value.st_uid),
            int(value.st_size),
            int(value.st_mtime_ns),
            int(value.st_ctime_ns),
        )

    try:
        directory = asset_dir.lstat()
        members = tuple(
            (name, token(entries[name].lstat()))
            for name in EXPECTED_ASSETS
        )
    except OSError as exc:
        raise DashboardAssetError(
            "dashboard asset generation changed while being inspected"
        ) from exc
    return token(directory), members


def validate_dashboard_asset_directory(
    asset_dir: str | Path,
    *,
    passphrase: str,
    expected_asof: str | date | None = None,
    now: datetime | None = None,
    require_current_sources: bool = True,
) -> dict[str, dict[str, Any]]:
    root = Path(asset_dir)
    entries = _validated_directory_entries(root)
    generation_before = _asset_generation_token(root, entries)
    raw_time = now or datetime.now(timezone.utc)
    if raw_time.tzinfo is None:
        raise DashboardAssetError("validation clock must be timezone-aware")
    current_time = raw_time.astimezone(timezone.utc)
    if expected_asof is None:
        cutoff = datetime.now(ZoneInfo("Asia/Seoul")).date()
    elif type(expected_asof) is date:
        cutoff = expected_asof
    elif isinstance(expected_asof, str):
        cutoff = _canonical_asof(expected_asof, field="expected_asof")
    else:
        raise DashboardAssetError("expected_asof must be a date or date string")

    payloads: dict[str, dict[str, Any]] = {}
    for name in EXPECTED_ASSETS:
        try:
            envelope = strict_json_object(entries[name])
        except ArtifactValidationError as exc:
            raise DashboardAssetError(
                f"dashboard encrypted JSON is invalid: {name}"
            ) from exc
        payloads[name] = decrypt_dashboard_envelope(
            envelope,
            passphrase,
            source=entries[name],
        )

    _validate_summary(
        payloads["summary.json"],
        expected_asof=cutoff,
        now=current_time,
    )
    _validate_history(payloads["history.json"], expected_asof=cutoff)
    _validate_accuracy(payloads["accuracy.json"], expected_asof=cutoff)
    _validate_idea(
        payloads["idea_validation.json"],
        expected_asof=cutoff,
        now=current_time,
        require_current_sources=require_current_sources,
    )
    _validate_idea(
        payloads["summary.json"]["idea_validation"],
        expected_asof=cutoff,
        now=current_time,
        require_current_sources=require_current_sources,
    )
    _validate_findings(
        payloads["findings.json"],
        expected_asof=cutoff,
        now=current_time,
    )
    _validate_cross_asset_identity(payloads)
    if require_current_sources:
        _validate_current_sources(payloads, expected_asof=cutoff)
    entries_after = _validated_directory_entries(root)
    if generation_before != _asset_generation_token(root, entries_after):
        raise DashboardAssetError(
            "dashboard asset generation changed during validation"
        )
    return payloads


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Authenticate and validate the five dashboard publish assets"
    )
    parser.add_argument("--asset-dir", required=True)
    parser.add_argument("--asof")
    parser.add_argument(
        "--pin",
        help="test/manual override; production uses PRELUDE_DASHBOARD_PIN",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        passphrase = resolve_dashboard_passphrase(args.pin)
        validate_dashboard_asset_directory(
            args.asset_dir,
            passphrase=passphrase,
            expected_asof=args.asof,
        )
    except (DashboardAssetError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print("dashboard assets: authenticated, current, and schema-valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
