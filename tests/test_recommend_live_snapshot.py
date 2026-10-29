from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from signals import recommend_live_snapshot as live
from signals import recommend_snapshot as frozen
from scripts.recommend_send import format_radar
from test_recommend_snapshot import _score_result


def _inconsistent(asof, **kwargs):
    result = _score_result(asof, kwargs["slot"], kwargs["ranking"])
    result["top3"][0]["p_up20"] = 0.2
    result["top3"][0]["pump_prob"] = 0.2
    result["top3"][0]["pump_prob_pct"] = "20.0%"
    return result


def test_inconsistent_delivery_keeps_exact_raw_rows_and_single_inference(tmp_path):
    calls = []

    def scorer(asof, **kwargs):
        calls.append(asof)
        return _inconsistent(asof, **kwargs)

    expected = copy.deepcopy(
        _inconsistent("2026-10-29", slot="open", ranking="R1")["universe"]
    )
    first = live.get_or_create_recommend_snapshot(
        "2026-10-29", root=tmp_path, scorer=scorer
    )
    second = live.get_or_create_recommend_snapshot(
        "2026-10-29", root=tmp_path, scorer=scorer
    )
    assert calls == ["2026-10-29"]
    assert first["snapshot_id"] == second["snapshot_id"]
    assert first["universe"] == expected
    assert first["top3"] == expected[:3]
    assert first["request"]["ranking"] == first["model"]["ranking"] == "R1"
    assert first["probability_quality"]["candidates"] == {"KRW-T1": ["upside"]}
    # No silent reinterpretation of frozen experiments or historical evidence.
    with pytest.raises(frozen.SnapshotError, match="nesting violated"):
        frozen.load_snapshot(first["snapshot_path"])
    text = format_radar(first, "open")
    first_card = text.split("#1 T1", 1)[1].split("#2 T2", 1)[0]
    assert "≥20% 확인불가" in first_card and "≥10% 확인불가" in first_card
    assert "20.0%" not in first_card
    assert "≤-5% 10%" in first_card
    assert "원본 추정치는 기록에 보존" in text


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["top3"][0].update(p_up5=None),
        lambda r: r["top3"][0].update(p_up10=float("nan")),
        lambda r: r["top3"][0].update(p_dn5=1.1),
        lambda r: r["top3"][0].update(rr_ratio=99),
        lambda r: r["top3"][0].update(pump_prob=0.1),
        lambda r: r["training"].update(end="2026-10-30"),
        lambda r: r.update(top3=list(reversed(r["top3"]))),
    ],
)
def test_delivery_exception_does_not_relax_other_guards(tmp_path, mutate):
    def scorer(asof, **kwargs):
        result = _inconsistent(asof, **kwargs)
        mutate(result)
        return result

    with pytest.raises((frozen.SnapshotError, ValueError)):
        live.get_or_create_recommend_snapshot(
            "2026-10-29", root=tmp_path, scorer=scorer
        )
    assert not (tmp_path / "2026-10-29/open_r1.json").exists()


def test_valid_snapshots_keep_frozen_format_and_reader(tmp_path):
    result = live.get_or_create_recommend_snapshot(
        "2026-10-29",
        root=tmp_path,
        scorer=lambda asof, **kw: _score_result(asof, kw["slot"], kw["ranking"]),
    )
    assert "probability_quality" not in result
    assert frozen.load_snapshot(result["snapshot_path"])["top3"] == result["top3"]


def test_concurrent_delivery_and_ledger_share_one_inconsistent_inference(tmp_path):
    calls = []

    def scorer(asof, **kwargs):
        calls.append(asof)
        return _inconsistent(asof, **kwargs)

    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [
            pool.submit(
                live.get_or_create_recommend_snapshot,
                "2026-10-29",
                root=tmp_path,
                scorer=scorer,
            )
            for _ in range(2)
        ]
        results = [job.result() for job in jobs]
    assert calls == ["2026-10-29"]
    assert results[0]["snapshot_id"] == results[1]["snapshot_id"]


def test_quality_marker_cannot_hide_invalid_raw_estimates(tmp_path):
    result = live.get_or_create_recommend_snapshot(
        "2026-10-29", root=tmp_path, scorer=_inconsistent
    )
    path = tmp_path / "2026-10-29/open_r1.json"
    document = json.loads(path.read_text())
    document["probability_quality"]["candidates"] = {}
    digest = frozen._document_digest(document)
    document.update(payload_sha256=digest, snapshot_id=f"recommend-{digest[:20]}")
    path.write_text(json.dumps(document))
    with pytest.raises(frozen.SnapshotError, match="marker"):
        live.load_snapshot(path)
    assert result["top3"][0]["p_up20"] == 0.2
