from __future__ import annotations

import pandas as pd
import pytest

from ledger.metrics import compute_mdd


def test_compute_mdd_includes_explicit_initial_equity_in_matching_units():
    normalized = pd.Series([0.9, 0.945])
    krw = pd.Series([9_000_000.0, 9_450_000.0])

    assert compute_mdd(normalized, initial_equity=1.0) == pytest.approx(-0.10)
    assert compute_mdd(krw, initial_equity=10_000_000.0) == pytest.approx(-0.10)


def test_compute_mdd_default_preserves_generic_curve_units():
    normalized = pd.Series([1.0, 0.9, 0.95])

    assert compute_mdd(normalized) == pytest.approx(-0.10)
