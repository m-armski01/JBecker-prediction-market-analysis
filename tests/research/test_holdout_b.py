"""Tests for the holdB normaliser on recorded Kalshi payloads (train-period market, current API schema).

kalshi_trades_sample.json: verbatim response of GET /historical/trades?ticker=KXBTCD-25JAN2615-T104999.99&limit=3.
kalshi_market_sample.json: the subset of GET /historical/markets/KXBTCD-25JAN2615-T104999.99 recorded by the
close_time audit (results/close_time_audit.json), plus event_ticker, market_type and created_time from the local
markets snapshot.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.research.longshot_fade.holdout_b import (
    MARKET_FIELDS,
    market_record,
    normalize_trades_df,
    raw_trade_record,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _norm(trades: list[dict]) -> pd.DataFrame:
    raw = pd.DataFrame([raw_trade_record(t) for t in trades])
    return normalize_trades_df(raw).set_index("trade_id")


def test_recorded_trades_payload():
    payload = json.loads((FIXTURES / "kalshi_trades_sample.json").read_text())
    df = _norm(payload["trades"])
    assert len(df) == 3
    a = df.loc["452c3139-8bfa-4786-b8a8-d726e5ce9b81"]
    assert (a.yes_price, a.no_price, a.taker_side, a["count"]) == (97, 3, "yes", 75.0)
    b = df.loc["9628a1a3-28c4-483e-a837-99c3353a7311"]
    assert (b.yes_price, b.no_price, b.taker_side, b["count"]) == (97, 3, "no", 500.0)
    c = df.loc["f8c04375-3670-41dd-9530-ed8b24b489c7"]
    assert (c.yes_price, c.no_price) == (99, 1)
    assert str(df["created_time"].dt.tz) == "UTC"
    assert a.created_time == pd.Timestamp("2025-01-26T19:58:23.921217Z")
    assert not df["is_block_trade"].any()
    assert ((df.yes_price + df.no_price) == 100).all()


def _t(tid, yes, no, **kw):
    base = {
        "trade_id": tid,
        "ticker": "KXT-1",
        "count_fp": "1.00",
        "yes_price_dollars": yes,
        "no_price_dollars": no,
        "taker_side": "no",
        "created_time": "2026-08-10T00:00:00Z",
        "is_block_trade": False,
    }
    base.update(kw)
    return base


def test_exact_decimal_floor_no_float_truncation():
    # int(float('0.29') * 100) == 28 in floating point; the normaliser must give 29.
    df = _norm([_t("x", "0.2900", "0.7100"), _t("y", "0.5700", "0.4300"), _t("z", "0.5800", "0.4200")])
    assert df.loc["x", ["yes_price", "no_price"]].tolist() == [29, 71]
    assert df.loc["y", ["yes_price", "no_price"]].tolist() == [57, 43]
    assert df.loc["z", ["yes_price", "no_price"]].tolist() == [58, 42]


def test_subpenny_prices_floor_to_sum_99():
    df = _norm([_t("s", "0.0090", "0.9910"), _t("u", "0.0150", "0.9850")])
    assert df.loc["s", ["yes_price", "no_price"]].tolist() == [0, 99]
    assert df.loc["u", ["yes_price", "no_price"]].tolist() == [1, 98]


def test_fractional_counts_legacy_fields_block_trades_and_dedup():
    df = _norm(
        [
            _t("f", "0.9500", "0.0500", count_fp="2.50"),
            {
                "trade_id": "L",
                "ticker": "KXT-1",
                "count": 3,
                "yes_price": 56,
                "no_price": 44,
                "taker_side": "YES",
                "created_time": "2026-08-10T00:00:00Z",
            },
            _t("blk", "0.9000", "0.1000", is_block_trade=True),
            _t("f", "0.9500", "0.0500", count_fp="2.50"),
        ]
    )
    assert len(df) == 3
    assert df.loc["f", "count"] == pytest.approx(2.5)
    assert df.loc["L", ["yes_price", "no_price", "taker_side"]].tolist() == [56, 44, "yes"]
    assert df.loc["L", "count"] == 3.0
    assert bool(df.loc["blk", "is_block_trade"]) and not bool(df.loc["f", "is_block_trade"])


def test_market_record_strips_outcome_fields():
    payload = json.loads((FIXTURES / "kalshi_market_sample.json").read_text())["market"]
    # Synthetic price/settlement fields added here to check they are stripped too.
    payload = {**payload, "settlement_value_dollars": "1.0000", "last_price_dollars": "0.9900"}
    rec = market_record(payload)
    assert list(rec) == MARKET_FIELDS
    assert rec["event_ticker"] == "KXBTCD-25JAN2615"
    for leaked in ["status", "result", "settlement_ts", "settlement_value_dollars", "last_price_dollars"]:
        assert leaked not in rec
