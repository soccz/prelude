"""Shared Upbit KRW signal-universe identity policy.

Raw collection keeps historical market identities.  Signal and research
panels additionally remove stablecoins and deprecated ticker aliases so one
economic asset cannot occupy the cross-section twice.
"""
from __future__ import annotations

from collections.abc import Iterable


STABLECOIN_KRW_MARKETS = frozenset(
    {
        "KRW-USD1",
        "KRW-USDC",
        "KRW-USDE",
        "KRW-USDS",
        "KRW-USDT",
    }
)

# Story (IP) was renamed Data Network (DATA) in July 2026.  Upbit's DATA
# candle endpoint exposes the pre-rename IP history, while the local raw DB
# intentionally retains the old KRW-IP rows for auditability.  Keeping both
# identities in a signal panel double-counts one asset from 2025-08-08 through
# the rename, so the deprecated identity is excluded downstream.
RENAMED_KRW_MARKET_ALIASES = {
    "KRW-IP": "KRW-DATA",
}
DEPRECATED_KRW_MARKETS = frozenset(RENAMED_KRW_MARKET_ALIASES)
SIGNAL_EXCLUDED_KRW_MARKETS = STABLECOIN_KRW_MARKETS | DEPRECATED_KRW_MARKETS


def is_excluded_stablecoin_market(market: str) -> bool:
    """Return whether ``market`` is explicitly excluded as a stablecoin."""
    return market in STABLECOIN_KRW_MARKETS


def is_excluded_signal_market(market: str) -> bool:
    """Return whether ``market`` is excluded from signal/research panels."""
    return market in SIGNAL_EXCLUDED_KRW_MARKETS


def stablecoin_exclusions(markets: Iterable[str]) -> tuple[str, ...]:
    """Return stablecoin identities in deterministic order for audit logs."""
    return tuple(sorted(set(markets) & STABLECOIN_KRW_MARKETS))


def signal_market_exclusions(markets: Iterable[str]) -> tuple[str, ...]:
    """Return excluded signal identities in deterministic order."""
    return tuple(sorted(set(markets) & SIGNAL_EXCLUDED_KRW_MARKETS))


def signal_eligible_markets(markets: Iterable[str]) -> list[str]:
    """Preserve input order while removing excluded signal identities."""
    return [
        market
        for market in markets
        if not is_excluded_signal_market(market)
    ]
