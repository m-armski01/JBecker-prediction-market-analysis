"""Tests for the Kalshi taker fee model."""

from __future__ import annotations

import math
from decimal import ROUND_CEILING, Decimal

import numpy as np
import pytest

from src.research.longshot_fade.fees import (
    fee_multiplier,
    series_multipliers,
    taker_fee_cents,
    taker_fee_cents_array,
)


def _reference(contracts: int, cost_cents: int, coef: str = "0.07") -> int:
    p = Decimal(cost_cents) / 100
    dollars = Decimal(coef) * contracts * p * (1 - p)
    return int((dollars * 100).to_integral_value(rounding=ROUND_CEILING))


def test_fee_schedule_examples():
    # Kalshi fee schedule worked examples: 1 contract at 50c -> $0.02; 100 contracts at 5c -> $0.34.
    assert taker_fee_cents(1, 50) == 2
    assert taker_fee_cents(100, 5) == 34


def test_band_examples():
    assert taker_fee_cents(100, 95) == 34  # 0.07*100*0.95*0.05 = $0.3325
    assert taker_fee_cents(100, 99) == 7  # $0.0693
    assert taker_fee_cents(100, 90) == 63  # $0.63 exactly, no rounding up
    assert taker_fee_cents(10, 99) == 1  # $0.00693 -> 1 cent
    assert taker_fee_cents(1000, 93) == 456  # $4.557


@pytest.mark.parametrize("n", [1, 10, 100, 1000])
def test_matches_decimal_reference_all_prices(n):
    for c in range(1, 100):
        assert taker_fee_cents(n, c) == _reference(n, c), (n, c)


def test_symmetry_and_zero():
    for c in range(1, 100):
        assert taker_fee_cents(100, c) == taker_fee_cents(100, 100 - c)
    assert taker_fee_cents(100, 100) == 0
    assert taker_fee_cents(0, 50) == 0


def test_series_multiplier_table():
    for s in ["INXD", "INXU", "KXINXU", "KXINX", "NASDAQ100", "KXNASDAQ100U", "KXINXMAXY"]:
        assert fee_multiplier(s) == 0.5, s
    for s in ["KXJOINXAI", "KXSOLNASDAQ", "KXNFLGAME", "KXBTCD", "", "SB"]:
        assert fee_multiplier(s) == 1.0, s


def test_half_multiplier_matches_reference():
    for c in range(1, 100):
        assert taker_fee_cents(100, c, 0.5) == _reference(100, c, "0.035")


def test_vectorised_matches_scalar():
    n = np.array([1, 10, 100, 1000, 100])
    c = np.array([50, 99, 95, 93, 5])
    m = series_multipliers(["KXBTCD", "KXINXU", "X", "Y", "Z"])
    got = taker_fee_cents_array(n, c, m)
    exp = [taker_fee_cents(a, b, mm) for a, b, mm in zip(n, c, m)]
    assert got.tolist() == exp


def test_invalid_inputs():
    with pytest.raises(ValueError):
        taker_fee_cents(-1, 50)
    with pytest.raises(ValueError):
        taker_fee_cents(1, 101)
    with pytest.raises(ValueError):
        taker_fee_cents(1, 50, multiplier=math.pi)
