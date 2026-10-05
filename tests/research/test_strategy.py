"""Tests for the entry rule, per-entry P&L and summaries."""

from __future__ import annotations

import math

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.research.longshot_fade import config as C
from src.research.longshot_fade import strategy as ST
from src.research.longshot_fade.fees import taker_fee_cents


def _row(
    tid,
    ticker,
    ts,
    cost,
    *,
    role="taker",
    direction="no",
    purged=False,
    split="train",
    group="Sports",
    notional=1000.0,
    event=None,
    result="no",
    void=False,
    series="KXNFLGAME",
    contracts=50,
):
    return {
        "trade_id": tid,
        "ticker": ticker,
        "event_ticker": event or ticker.rsplit("-", 1)[0],
        "series": series,
        "group": group,
        "category": "x",
        "subcategory": "y",
        "ts": pd.Timestamp(ts, tz="UTC"),
        "quarter": "2025Q1",
        "split": split,
        "purged": purged,
        "role": role,
        "direction": direction,
        "cost_cents": cost,
        "contracts": contracts,
        "hours_since_open": 1.0,
        "prior_notional_usd": notional,
        "prior_trades": 1,
        "prev_yes_price": 100 - cost,
        "hours_to_close": 10.0,
        "close_time": pd.Timestamp("2025-03-01", tz="UTC"),
        "result": result,
        "settled": result in ("yes", "no"),
        "void": void,
        "won": None,
    }


@pytest.fixture()
def con():
    panel = pd.DataFrame(
        [
            # M1: maker row and purged row earlier than the first valid taker-NO print
            _row("m1a", "EV1-M1", "2025-01-01 00:00", 91, role="maker"),
            _row("m1b", "EV1-M1", "2025-01-01 00:01", 93, purged=True),
            _row("m1c", "EV1-M1", "2025-01-01 00:02", 89),
            _row("m1d", "EV1-M1", "2025-01-01 00:03", 94, direction="yes"),
            _row("m1e", "EV1-M1", "2025-01-01 00:04", 92),
            _row("m1f", "EV1-M1", "2025-01-01 00:05", 96),
            _row("m1g", "EV1-M1", "2025-01-01 00:06", 97),
            # M2: Finance market
            _row("m2a", "EV2-M2", "2025-01-02 00:00", 95, group="Finance", series="KXINXU", result="yes"),
            # M3: liquidity floor
            _row("m3a", "EV3-M3", "2025-01-03 00:00", 95, notional=100.0),
            _row("m3b", "EV3-M3", "2025-01-03 00:01", 98, notional=600.0),
            # M4: other split
            _row("m4a", "EV4-M4", "2025-08-01 00:00", 95, split="val"),
            # M5: tie on timestamp broken by trade_id
            _row("m5b", "EV5-M5", "2025-01-05 00:00", 99),
            _row("m5a", "EV5-M5", "2025-01-05 00:00", 90),
        ]
    )
    c = duckdb.connect()
    c.execute("SET TimeZone='UTC'")
    c.register("panel_df", panel)
    return c


def _cfg(lo, cat="all", floor=0.0):
    return C.StrategyConfig(lo, 99, cat, floor)


def test_first_qualifying_taker_no_print(con):
    e = ST.select_entries(con, "panel_df", _cfg(90), "train").set_index("ticker")
    assert set(e.index) == {"EV1-M1", "EV2-M2", "EV3-M3", "EV5-M5"}
    assert e.loc["EV1-M1", "trade_id"] == "m1e"  # skips maker, purged, out-of-band and YES rows
    assert e.loc["EV3-M3", "trade_id"] == "m3a"
    assert e.loc["EV5-M5", "trade_id"] == "m5a"  # same ts: lower trade_id first


def test_band_category_floor_and_second_print(con):
    e95 = ST.select_entries(con, "panel_df", _cfg(95), "train").set_index("ticker")
    assert e95.loc["EV1-M1", "trade_id"] == "m1f"
    exf = ST.select_entries(con, "panel_df", _cfg(90, "ex_finance"), "train")
    assert "EV2-M2" not in set(exf.ticker)
    liq = ST.select_entries(con, "panel_df", _cfg(90, floor=500.0), "train").set_index("ticker")
    assert liq.loc["EV3-M3", "trade_id"] == "m3b"
    second = ST.select_entries(con, "panel_df", _cfg(90), "train", k=2).set_index("ticker")
    assert second.loc["EV1-M1", "trade_id"] == "m1f"
    assert "EV2-M2" not in second.index
    val = ST.select_entries(con, "panel_df", _cfg(90), "val")
    assert val.ticker.tolist() == ["EV4-M4"]


def test_grid_has_12_configs_in_order():
    g = C.grid()
    assert len(g) == 12 and len({x.id for x in g}) == 12
    assert g[0] == C.StrategyConfig(90, 99, "all", 0.0)
    assert g[1] == C.StrategyConfig(90, 99, "all", 500.0)
    assert g[2] == C.StrategyConfig(90, 99, "ex_finance", 0.0)
    assert g[-1] == C.StrategyConfig(95, 99, "ex_finance", 500.0)


def _entries(rows):
    return pd.DataFrame(rows)


def _e(ticker, cost, result, void=False, ts="2025-01-01", close="2025-01-11", series="KXNFLGAME", contracts=500):
    return {
        "ticker": ticker,
        "event_ticker": ticker.split("-")[0],
        "series": series,
        "cost_cents": cost,
        "contracts": contracts,
        "ts": pd.Timestamp(ts, tz="UTC"),
        "close_time": pd.Timestamp(close, tz="UTC"),
        "result": result,
        "void": void,
    }


def test_pnl_no_yes_void_censored():
    df = ST.compute_pnl(
        _entries([_e("A-1", 95, "no"), _e("B-1", 95, "yes"), _e("C-1", 95, None, void=True), _e("D-1", 95, None)]),
        n_contracts=100,
    )
    fee = taker_fee_cents(100, 95)
    cost = 100 * 95 + fee
    assert df.outcome.tolist() == ["no", "yes", "void", "censored"]
    assert df.fee_cents.tolist() == [fee] * 4
    assert df.net_ret[0] == pytest.approx((10000 - cost) / cost)
    assert df.gross_ret[0] == pytest.approx((10000 - 9500) / 9500)
    assert df.net_ret[1] == pytest.approx(-1.0)
    assert df.net_ret[2] == pytest.approx((9500 - cost) / cost)  # refund of N*price, fee not refunded
    assert df.gross_ret[2] == pytest.approx(0.0)
    assert math.isnan(df.net_ret[3])
    assert df.pnl_usd[0] == pytest.approx((10000 - cost) / 100)


def test_hold_days_floor_and_value():
    df = ST.compute_pnl(
        _entries([_e("A-1", 95, "no"), _e("B-1", 95, "no", ts="2025-01-02", close="2025-01-01")]), n_contracts=100
    )
    assert df.hold_days[0] == pytest.approx(10.0)
    assert df.hold_days[1] == pytest.approx(1 / 24)


def test_series_fee_multiplier_and_sensitivities():
    base = ST.compute_pnl(_entries([_e("INX-1", 95, "no", series="KXINXU")]), n_contracts=100)
    assert base.fee_cents[0] == taker_fee_cents(100, 95, 0.5)
    slip = ST.compute_pnl(_entries([_e("A-1", 95, "no")]), n_contracts=100, slippage_cents=1)
    assert slip.price[0] == 96 and slip.fee_cents[0] == taker_fee_cents(100, 96)
    cap = ST.compute_pnl(_entries([_e("A-1", 95, "no", contracts=7)]), n_contracts=100, size_cap=True)
    assert cap.n[0] == 7
    big = ST.compute_pnl(_entries([_e("A-1", 95, "no")]), n_contracts=1000)
    assert big.cost[0] == 1000 * 95 + taker_fee_cents(1000, 95)


def test_summarize_excludes_censored_and_clusters_by_event():
    rows = [
        _e("E1-a", 95, "no"),
        _e("E1-b", 95, "no"),
        _e("E2-a", 95, "yes"),
        _e("E3-a", 95, "no"),
        _e("E4-a", 95, None),
    ]
    for r in rows:
        r["event_ticker"] = r["ticker"].split("-")[0]
    pnl = ST.compute_pnl(_entries(rows), n_contracts=100)
    s = ST.summarize(pnl, iterations=200)
    assert s["n_entries"] == 4 and s["n_events"] == 3 and s["n_censored"] == 1 and s["n_candidates"] == 5
    assert s["win_rate"] == pytest.approx(0.75)
    r = pnl.loc[pnl.outcome != "censored", "net_ret"].to_numpy()
    assert s["mean_net_ret"] == pytest.approx(r.mean())
    assert s["worst_event_loss_usd"] == pytest.approx(pnl.loc[2, "pnl_usd"])
    cap = (pnl.cost * pnl.hold_days)[pnl.outcome != "censored"].sum()
    assert s["mean_net_ret_per_capital_day"] == pytest.approx((pnl.payout - pnl.cost)[:4].sum() / cap)
    assert np.sign(s["t_event"]) == np.sign(s["mean_net_ret"])


def test_summarize_empty():
    s = ST.summarize(ST.compute_pnl(_entries([_e("A-1", 95, None)]), n_contracts=100))
    assert s["n_entries"] == 0 and s["n_censored"] == 1 and math.isnan(s["t_event"])
    assert not ST.passes(s)


def test_passes_rule():
    assert ST.passes({"t_event": 2.0, "boot_ci_lo": 0.0001})
    assert not ST.passes({"t_event": 1.99, "boot_ci_lo": 0.01})
    assert not ST.passes({"t_event": 3.0, "boot_ci_lo": 0.0})


def test_attach_outcomes():
    entries = pd.DataFrame(
        [
            {**_e("A-1", 95, None), "settled": None},
            {**_e("B-1", 95, None), "settled": None},
            {**_e("C-1", 95, None), "settled": None},
        ]
    )
    outcomes = pd.DataFrame(
        [
            {"ticker": "A-1", "result": "no", "void": False, "close_time": pd.Timestamp("2025-01-03", tz="UTC")},
            {"ticker": "B-1", "result": None, "void": True, "close_time": pd.NaT},
        ]
    )
    e = ST.attach_outcomes(entries, outcomes)
    pnl = ST.compute_pnl(e, n_contracts=100)
    assert pnl.outcome.tolist() == ["no", "void", "censored"]
    assert pnl.hold_days[0] == pytest.approx(2.0)  # backfilled close_time overrides
    assert pnl.hold_days[1] == pytest.approx(10.0)  # NaT keeps the panel close_time
