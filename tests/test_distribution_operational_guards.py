from __future__ import annotations

import sys

import pandas as pd

from notifier.format import format_distribution_beta, format_preopen_beta
from scripts.predict_today_distribution import append_to_paper_ledger, is_decision_window


def _alert_row(market: str = "KRW-AAA") -> pd.DataFrame:
    return pd.DataFrame([
        {
            "market": market,
            "primary_setups": ["S02"],
            "btc_regime": "bull_quiet",
            "btc_context": "S04",
            "p_h2_hit_3_4h": 0.91,
            "p_h6_hit_5_24h": 0.82,
            "p_h5_tail_20": 0.77,
            "p_h1_stable_3_4h": 0.0,
            "p_h7_mid_7_24h": 0.0,
            "setup_quality": "B_S03",
            "expected_edge_pct": 2.31,
            "calibrated_hit_pct": 62.4,
            "calibration_source": "bucket_h6",
            "source_rank": 2,
            "confidence_score": 78.4,
            "confidence_tier": "EVIDENCE_OK",
            "decision_reason": "S03-quality setup with positive h6 edge proxy",
            "policy_id": "setup_quality_policy_v1",
            "idea_id": "dist_h6_b_s03_bull_quiet_v1",
            "log_return_1d": 0.052,
            "atr_pct_14": 0.081,
            "return_5d": 0.94,
            "return_7d": 0.90,
            "vol_5d": 0.88,
            "roc_3d": 0.96,
            "composite": 5.0,
        }
    ])


def test_decision_window_blocks_late_ledger_runs():
    assert not is_decision_window(pd.Timestamp("2026-05-05 08:49:00"))
    assert is_decision_window(pd.Timestamp("2026-05-05 08:50:00"))
    assert is_decision_window(pd.Timestamp("2026-05-05 09:05:00"))
    assert is_decision_window(pd.Timestamp("2026-05-05 09:20:00"))
    assert not is_decision_window(pd.Timestamp("2026-05-05 09:21:00"))
    assert not is_decision_window(pd.Timestamp("2026-05-05 20:54:00"))


def test_append_to_paper_ledger_keeps_one_snapshot_per_date(tmp_path):
    ledger_path = tmp_path / "paper_ledger.csv"
    asof = pd.Timestamp("2026-05-05 09:05:00")

    assert append_to_paper_ledger(_alert_row("KRW-AAA"), asof, str(ledger_path)) == 1
    assert append_to_paper_ledger(_alert_row("KRW-BBB"), asof, str(ledger_path)) == 0

    ledger = pd.read_csv(ledger_path)
    assert ledger["date"].tolist() == ["2026-05-05"]
    assert ledger["coin"].tolist() == ["KRW-AAA"]


def test_distribution_format_uses_unified_design():
    msg = format_distribution_beta(
        alerts=_alert_row(),
        btc_regime="bull_quiet",
        universe_label="top100",
        universe_size=1,
    )

    # 통일 헤더 + BTC 한글 라벨 + composite tier (composite=5.0 → 🔥)
    assert msg.startswith("⚡ distribution")
    assert "🟢 강세 안정" in msg
    assert "🔥" in msg
    # 텔레그램은 raw 모델 점수 대신 policy/edge/검증 hit 중심으로 표시
    assert "raw" not in msg
    assert "edge +2.31%p" in msg
    assert "conf 78/EVIDENCE_OK" in msg
    assert "검증 hit 62.4%" in msg
    assert "policy:" in msg
    # 사용 가이드 + 진단/Setup library 블록은 제거
    assert "━━━ 사용 ━━━" in msg
    assert "Setup library" not in msg
    assert "━━━ 진단 ━━━" not in msg


def test_distribution_silence_uses_unified_design():
    import pandas as pd

    msg = format_distribution_beta(
        alerts=pd.DataFrame(),
        btc_regime="bear_quiet",
        universe_label="top100",
        universe_size=100,
    )
    assert msg.startswith("⚡ distribution")
    assert "🔴 약세 안정" in msg
    assert "━━━ 침묵 ━━━" in msg


def test_preopen_format_uses_unified_design():
    import pandas as pd

    alerts = pd.DataFrame([
        {
            "market": "KRW-AAA",
            "p_first15_3pct": 0.85,
            "p_first15_5pct": 0.70,
            "p_first1h_3pct": 0.80,
            "setup_quality": "PREOPEN",
            "expected_edge_pct": 1.20,
            "calibrated_hit_pct": 46.0,
            "calibration_source": "raw_first1h_5_fallback",
            "source_rank": 1,
            "confidence_score": 52.0,
            "confidence_tier": "COLLECT",
            "decision_reason": "bull regime with strong first1h_5 and composite",
            "composite": 1.7,
            "close": 1234.5,
        }
    ])
    msg = format_preopen_beta(
        alerts=alerts,
        btc_regime="bull_quiet",
        universe_label="top100",
    )
    assert msg.startswith("⚡ pre-open trigger")
    assert "🟢 강세 안정" in msg
    assert "🔥" in msg  # composite 1.7 → fire
    assert "AAA" in msg
    assert "raw" not in msg
    assert "edge +1.20%p" in msg
    assert "conf 52/COLLECT" in msg
    assert "1h +5 signal 46.0%" in msg
    assert "━━━ 사용 ━━━" in msg


def test_close_paper_ledger_no_data_notes_do_not_crash(tmp_path, monkeypatch):
    import scripts.close_paper_ledger as close_mod

    ledger_path = tmp_path / "paper_ledger.csv"
    pd.DataFrame([
        {
            "date": "2026-05-04",
            "coin": "KRW-AAA",
            "setup_ids": "S02",
            "status": "entered",
            "notes": "",
        }
    ]).to_csv(ledger_path, index=False)

    monkeypatch.setattr(close_mod, "compute_realized", lambda *args, **kwargs: {"status": "no_data"})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "close_paper_ledger.py",
            "--paper-ledger",
            str(ledger_path),
            "--asof",
            "2026-05-05 09:30:00",
            "--dry-run",
        ],
    )

    close_mod.main()


def test_close_paper_ledger_also_closes_shadow(tmp_path, monkeypatch):
    import scripts.close_paper_ledger as close_mod

    main_path = tmp_path / "paper_ledger.csv"
    shadow_path = tmp_path / "shadow_ledger_distribution.csv"
    row = {
        "date": "2026-05-04",
        "coin": "KRW-AAA",
        "setup_ids": "S01+S02+S03",
        "status": "entered",
        "notes": "",
    }
    pd.DataFrame([row]).to_csv(main_path, index=False)
    pd.DataFrame([{**row, "decision": "WATCH_ONLY"}]).to_csv(shadow_path, index=False)

    monkeypatch.setattr(
        close_mod,
        "compute_realized",
        lambda *args, **kwargs: {
            "status": "closed",
            "next_open": 100.0,
            "next_high": 106.0,
            "next_low": 99.0,
            "next_close": 103.0,
            "next_max_return_pct": 6.0,
            "next_min_return_pct": -1.0,
            "next_close_return_pct": 3.0,
            "hit_h2": 1,
            "hit_h6": 1,
            "hit_h5": 0,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "close_paper_ledger.py",
            "--paper-ledger",
            str(main_path),
            "--shadow-ledger",
            str(shadow_path),
            "--asof",
            "2026-05-05 09:30:00",
        ],
    )

    close_mod.main()

    assert pd.read_csv(main_path)["status"].tolist() == ["closed"]
    shadow = pd.read_csv(shadow_path)
    assert shadow["status"].tolist() == ["closed"]
    assert shadow["decision"].tolist() == ["WATCH_ONLY"]


def test_preopen_close_waits_for_full_first_hour(monkeypatch):
    import scripts.close_preopen_ledger as close_mod

    df = pd.DataFrame([
        {"timestamp": "2026-05-05 09:00:00", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
        {"timestamp": "2026-05-05 09:15:00", "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0},
    ])
    monkeypatch.setattr(close_mod, "load_candles", lambda *args, **kwargs: df.copy())

    result = close_mod.compute_realized_preopen("KRW-AAA", pd.Timestamp("2026-05-05"))

    assert result["status"] == "pending"
    assert "only 2" in result["notes"]


def test_preopen_close_closes_with_four_15m_bars(monkeypatch):
    import scripts.close_preopen_ledger as close_mod

    df = pd.DataFrame([
        {"timestamp": "2026-05-05 09:00:00", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
        {"timestamp": "2026-05-05 09:15:00", "open": 100.0, "high": 104.0, "low": 99.0, "close": 103.0},
        {"timestamp": "2026-05-05 09:30:00", "open": 103.0, "high": 105.5, "low": 102.0, "close": 105.0},
        {"timestamp": "2026-05-05 09:45:00", "open": 105.0, "high": 106.0, "low": 104.0, "close": 105.5},
    ])
    monkeypatch.setattr(close_mod, "load_candles", lambda *args, **kwargs: df.copy())

    result = close_mod.compute_realized_preopen("KRW-AAA", pd.Timestamp("2026-05-05"))

    assert result["status"] == "closed"
    assert result["hit_first15_3pct"] == 0
    assert result["hit_first30_3pct"] == 1
    assert result["hit_first1h_5pct"] == 1


def test_close_preopen_ledger_also_closes_shadow(tmp_path, monkeypatch):
    import scripts.close_preopen_ledger as close_mod

    main_path = tmp_path / "paper_ledger_preopen.csv"
    shadow_path = tmp_path / "shadow_ledger_preopen.csv"
    row = {
        "date": "2026-05-05",
        "coin": "KRW-AAA",
        "status": "entered",
        "notes": "",
    }
    pd.DataFrame([row]).to_csv(main_path, index=False)
    pd.DataFrame([{**row, "decision": "WATCH_ONLY"}]).to_csv(shadow_path, index=False)

    monkeypatch.setattr(
        close_mod,
        "compute_realized_preopen",
        lambda *args, **kwargs: {
            "status": "closed",
            "first_open": 100.0,
            "first_15m_high": 101.0,
            "first_30m_high": 103.0,
            "first_1h_high": 106.0,
            "first_15m_low": 99.0,
            "first_30m_low": 99.0,
            "first_1h_low": 98.0,
            "first_1h_close": 105.0,
            "first_1h_max_return_pct": 6.0,
            "first_1h_min_return_pct": -2.0,
            "first_1h_close_return_pct": 5.0,
            "hit_first15_3pct": 0,
            "hit_first15_5pct": 0,
            "hit_first30_3pct": 1,
            "hit_first30_5pct": 0,
            "hit_first1h_3pct": 1,
            "hit_first1h_5pct": 1,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "close_preopen_ledger.py",
            "--paper-ledger",
            str(main_path),
            "--shadow-ledger",
            str(shadow_path),
            "--asof",
            "2026-05-05 10:05:00",
        ],
    )

    close_mod.main()

    assert pd.read_csv(main_path)["status"].tolist() == ["closed"]
    shadow = pd.read_csv(shadow_path)
    assert shadow["status"].tolist() == ["closed"]
    assert shadow["decision"].tolist() == ["WATCH_ONLY"]
