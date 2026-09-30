"""Public projection contract and bounded/read-only failure behavior."""

import copy
import json
import subprocess
from datetime import timedelta
from types import SimpleNamespace

import pytest

from ops import dashboard_book_validation as display
from ops import recommend_book_validation as io
from test_recommend_book_validation import at, labels_exist, ready as ready  # noqa: F401


def project(ready, tmp_path, n=0):
    root, _, _ = ready
    days = [(at("2026-10-01") + timedelta(days=i)).date().isoformat() for i in range(n)]
    labels_exist(tmp_path, *days)
    now = at() if not n else at(days[-1]) + timedelta(days=1)
    for _ in range(max(1, (n + 1) // 2)):
        io.refresh(root=root, now=now)
    report = io.inspect(root=root, now=now)
    return display.project_report(report, now=now), now


@pytest.mark.parametrize("n", [0, 1, 4, 5, 6])
def test_actual_checked_cache_projection_and_null_ci(ready, tmp_path, n):
    payload, now = project(ready, tmp_path, n)
    assert payload["counts"]["paired_dates"] == n
    assert payload["counts"]["picks_per_arm"] == 3 * n
    assert (payload["metrics"] is None) == (n == 0)
    assert (payload["difference_ci95"] is None) == (n < 5)
    assert (
        payload["automatic_promotion"] is False
        and payload["prospective_pick_records"] == 0
    )
    assert "COIN" not in json.dumps(payload) and str(tmp_path) not in json.dumps(
        payload
    )
    display.validate_book_validation(payload, asof=now.date().isoformat(), now=now)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(private_path="/private"),
        lambda p: p.update(prospective_pick_records=True),
        lambda p: p.update(automatic_promotion=True),
        lambda p: p.update(start_date="2026-09-10"),
        lambda p: p.update(status="waiting_for_first_outcome"),
        lambda p: p.update(attention_required=True),
        lambda p: p.update(report_generated_at="2026-10-01T12:00:00+09:00"),
        lambda p: p["counts"].update(paired_dates=True),
        lambda p: p["counts"].update(pending_dates=1),
        lambda p: p["counts"].update(changed_picks=999),
        lambda p: p["metrics"]["control"].update(dn5=1.01),
        lambda p: p["metrics"]["challenger"].update(eod_return_net=float("nan")),
        lambda p: p["metrics"]["challenger_minus_control"].update(up10=0.9),
        lambda p: p.update(difference_ci95={"eod_return_net": [0, 1]}),
    ],
)
def test_forged_or_inconsistent_display_fails(ready, tmp_path, mutate):
    payload, now = project(ready, tmp_path, 1)
    mutate(payload)
    with pytest.raises((ValueError, TypeError)):
        display.validate_book_validation(payload, asof=now.date().isoformat(), now=now)


def test_incomplete_is_not_zero_or_hidden(ready, tmp_path):
    root, _, _ = ready
    now = at("2026-10-02")
    report = io.refresh(root=root, now=now)
    payload = display.project_report(report, now=now)
    assert payload["status"] == "incomplete" and payload["attention_required"]
    assert payload["counts"]["pending_dates"] == 1 and payload["metrics"] is None


def test_historical_is_unknown_and_does_not_invoke_probe(monkeypatch):
    monkeypatch.setattr(display.subprocess, "run", lambda *a, **k: pytest.fail("probe"))
    payload = display.build_book_validation(asof="2026-09-29", now=at())
    assert payload["status"] == "historical_not_observed" and payload["counts"] is None


@pytest.mark.parametrize(
    "fault", ["timeout", "nonzero", "size", "json", "clock", "valid"]
)
def test_bounded_probe_and_safe_fallback(ready, tmp_path, monkeypatch, fault):
    payload, now = project(ready, tmp_path)
    result = copy.deepcopy(payload)
    if fault == "clock":
        result["observed_at"] = (now - timedelta(seconds=1)).isoformat()

    def child(args, **kwargs):
        assert args[3] == "ops.dashboard_book_validation" and kwargs["timeout"] == 30
        assert "--refresh" not in args and "--initialize" not in args
        if fault == "timeout":
            raise subprocess.TimeoutExpired(args, 30)
        stdout = (
            b"x" * 32769
            if fault == "size"
            else b"PRIVATE ERROR"
            if fault == "json"
            else json.dumps(result).encode()
        )
        return SimpleNamespace(returncode=int(fault == "nonzero"), stdout=stdout)

    monkeypatch.setattr(display.subprocess, "run", child)
    actual = display.build_book_validation(asof=now.date().isoformat(), now=now)
    assert actual["status"] == (
        "waiting_for_first_outcome" if fault == "valid" else "unavailable"
    )
    if fault != "valid":
        assert actual["counts"] is actual["metrics"] is None
    assert "PRIVATE" not in json.dumps(actual)


def test_builder_and_encrypted_asset_validator_call_the_new_contract():
    import ast
    from pathlib import Path
    from scripts import build_dashboard, validate_dashboard_assets

    for module, name in (
        (build_dashboard, "build_book_validation"),
        (validate_dashboard_assets, "validate_book_validation"),
    ):
        tree = ast.parse(Path(module.__file__).read_text())
        assert any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == name
            for n in ast.walk(tree)
        )


def test_cli_unavailable_never_leaks_private_failure_text(monkeypatch, capsys):
    monkeypatch.setattr(
        io, "inspect", lambda **k: (_ for _ in ()).throw(ValueError("/private/SECRET"))
    )
    assert display.main(["--now", at().isoformat()]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "unavailable" and payload["counts"] is None
    assert "SECRET" not in json.dumps(payload)
