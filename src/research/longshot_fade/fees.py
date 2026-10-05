"""Kalshi taker fee model (protocol section 3.4).

fee_cents = ceil(100 * coef * m * C * P * (1 - P)), with P = cost_cents / 100, coef = 0.07 and series multiplier m.
Computed in integer arithmetic: with c = cost_cents, 100 * 0.07 * C * (c/100) * (1 - c/100) = 70 * C * c * (100 - c) / 1e5.
"""

from __future__ import annotations

import re
from typing import Union

import numpy as np

FEE_COEF = 0.07
# coef * multiplier expressed in thousandths: 0.07 -> 70, 0.035 -> 35
_DENOM = 100_000

# Series-specific taker multipliers (S&P 500 and Nasdaq-100 markets at 0.035 since 2022-09-22).
SERIES_MULTIPLIERS: list[tuple[str, float]] = [
    (r"^(KX)?(INX|NASDAQ100)", 0.5),
]
_COMPILED = [(re.compile(p), m) for p, m in SERIES_MULTIPLIERS]

IntLike = Union[int, np.integer]


def fee_multiplier(series: str) -> float:
    """Taker fee multiplier for a series ticker (event_ticker prefix before the first '-')."""
    s = (series or "").upper()
    for pat, mult in _COMPILED:
        if pat.match(s):
            return mult
    return 1.0


def _coef_milli(multiplier: float) -> int:
    val = FEE_COEF * multiplier * 1000
    out = int(round(val))
    if abs(out - val) > 1e-9:
        raise ValueError(f"fee coefficient {FEE_COEF}*{multiplier} is not a whole number of thousandths")
    return out


def taker_fee_cents(contracts: IntLike, cost_cents: IntLike, multiplier: float = 1.0) -> int:
    """Fee in cents for one taker order of `contracts` at `cost_cents` (1-99), rounded up to the cent."""
    c = int(cost_cents)
    n = int(contracts)
    if n < 0 or not 0 <= c <= 100:
        raise ValueError(f"invalid order: contracts={n}, cost_cents={c}")
    num = _coef_milli(multiplier) * n * c * (100 - c)
    return -(-num // _DENOM)


def taker_fee_cents_array(contracts, cost_cents, multipliers=1.0) -> np.ndarray:
    """Vectorised taker_fee_cents; arguments broadcast. Returns int64 cents."""
    n = np.asarray(contracts, dtype=np.int64)
    c = np.asarray(cost_cents, dtype=np.int64)
    m = np.asarray(multipliers, dtype=np.float64)
    milli = np.rint(FEE_COEF * m * 1000).astype(np.int64)
    if np.any(np.abs(milli - FEE_COEF * m * 1000) > 1e-9):
        raise ValueError("fee coefficient is not a whole number of thousandths")
    num = milli * n * c * (100 - c)
    return -(-num // _DENOM)


def series_multipliers(series) -> np.ndarray:
    """Vectorised fee_multiplier over an iterable of series tickers."""
    return np.array([fee_multiplier(s) for s in series], dtype=np.float64)
