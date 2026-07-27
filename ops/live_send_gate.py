"""Single fail-closed contract for retired/manual Telegram send paths."""
from __future__ import annotations

import argparse


def legacy_live_send_message(*, slot: str, scoring_command: str) -> str:
    if slot not in {"preopen", "open"}:
        raise ValueError(f"unsupported canonical sender slot: {slot!r}")
    return (
        "live Telegram sending is disabled for this legacy/manual runner. "
        "Use the canonical sender: "
        f"python scripts/recommend_send.py --slot {slot}. "
        f"For scoring/record-only use: {scoring_command}"
    )


def reject_legacy_live_send(
    parser: argparse.ArgumentParser,
    *,
    requested: bool,
    slot: str,
    scoring_command: str,
) -> None:
    """Exit through argparse before scoring, output writes, or network calls."""
    if requested:
        parser.error(
            legacy_live_send_message(
                slot=slot,
                scoring_command=scoring_command,
            )
        )
