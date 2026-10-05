"""Tests for the positions table builder on a toy trades/markets dataset."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src.analysis.kalshi.util.categories import get_hierarchy
from src.research.longshot_fade import config as C
from src.research.longshot_fade.panel import (
    PANEL_COLUMNS,
    build_panel,
    holdout_file_hashes,
    panel_source,
    sanity_train,
)

UTC = timezone.utc


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def _trade(tid, ticker, ts, yes, no, side, count=1):
    return {
        "trade_id": tid,
        "ticker": ticker,
        "count": count,
        "yes_price": yes,
        "no_price": no,
        "taker_side": side,
        "created_time": _ts(ts),
        "_fetched_at": pd.Timestamp("2025-11-25"),
    }


def _market(ticker, event, status, result, open_t="2024-09-01", close_t="2026-01-01"):
    return {
        "ticker": ticker,
        "event_ticker": event,
        "status": status,
        "result": result,
        "open_time": _ts(open_t),
        "close_time": _ts(close_t),
    }


@pytest.fixture()
def toy(tmp_path: Path):
    trades = pd.DataFrame(
        [
            _trade("a1", "KXNFLGAME-24OCT-A", "2024-09-15 12:00", 10, 90, "yes", 5),  # pre-sample history
            _trade("a2", "KXNFLGAME-24OCT-A", "2024-11-01 10:00", 8, 92, "no", 10),
            _trade("a3", "KXNFLGAME-24OCT-A", "2024-11-01 10:00", 9, 91, "yes", 1),  # same ts, later id
            _trade("a4", "KXNFLGAME-24OCT-A", "2025-01-15 09:00", 5, 95, "no", 2),  # next quarter
            _trade("v1", "KXNFLGAME-24NOV-V", "2024-12-01 09:00", 20, 80, "no", 3),
            _trade("b1", "KXINXU-25AUG-B1", "2025-06-15 09:00", 50, 50, "yes", 4),  # event spans train+val
            _trade("b2", "KXINXU-25AUG-B2", "2025-07-15 09:00", 40, 60, "no", 6),
            _trade("h1", "QQQQQ-25OCT-H", "2025-10-15 09:00", 3, 97, "no", 7),
            _trade("h2", "QQQQQ-25OCT-H", "2025-10-16 09:00", 0, 99, "no", 4),  # sub-cent print (sum 99)
        ]
    )
    markets = pd.DataFrame(
        [
            _market("KXNFLGAME-24OCT-A", "KXNFLGAME-24OCT", "finalized", "no", close_t="2025-02-01"),
            _market("KXNFLGAME-24NOV-V", "KXNFLGAME-24NOV", "finalized", ""),
            _market("KXINXU-25AUG-B1", "KXINXU-25AUG", "finalized", "yes"),
            _market("KXINXU-25AUG-B2", "KXINXU-25AUG", "finalized", "no"),
            _market("QQQQQ-25OCT-H", "QQQQQ-25OCT", "finalized", "yes"),
        ]
    )
    tdir, mdir = tmp_path / "trades", tmp_path / "markets"
    tdir.mkdir()
    mdir.mkdir()
    trades.to_parquet(tdir / "trades_0.parquet", index=False)
    markets.to_parquet(mdir / "markets_0.parquet", index=False)
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    out = tmp_path / "panel"
    pieces = build_panel(con, str(tdir / "*.parquet"), str(mdir / "*.parquet"), out, log=lambda *_: None)
    df = con.execute(
        f"SELECT * FROM read_parquet('{out}/*/*/*.parquet', hive_partitioning = false) ORDER BY trade_id, role"
    ).df()
    return {"con": con, "out": out, "pieces": pieces, "df": df, "root": tmp_path}


def _row(df, tid, role):
    r = df[(df.trade_id == tid) & (df.role == role)]
    assert len(r) == 1
    return r.iloc[0]


def test_two_rows_per_trade_and_columns(toy):
    df = toy["df"]
    assert list(df.columns) == PANEL_COLUMNS
    assert len(df) == 2 * 8  # 8 trades from 2024-10-01, the pre-sample trade is history only
    assert set(df.role) == {"taker", "maker"}
    assert "a1" not in set(df.trade_id)


def test_direction_and_cost(toy):
    df = toy["df"]
    t, m = _row(df, "a2", "taker"), _row(df, "a2", "maker")
    assert (t.direction, t.cost_cents) == ("no", 92)
    assert (m.direction, m.cost_cents) == ("yes", 8)


def test_prior_features_ordering_and_carry_over(toy):
    df = toy["df"]
    a2, a3, a4 = (_row(df, x, "taker") for x in ("a2", "a3", "a4"))
    # a1 (pre-sample) seeds the state: 5 contracts at yes 10 -> $0.50
    assert a2.prior_trades == 1 and a2.prior_notional_usd == pytest.approx(0.50) and a2.prev_yes_price == 10
    # same timestamp: a3 ordered after a2 by trade_id
    assert a3.prior_trades == 2 and a3.prior_notional_usd == pytest.approx(1.30) and a3.prev_yes_price == 8
    # next quarter carries the state
    assert a4.prior_trades == 3 and a4.prior_notional_usd == pytest.approx(1.39) and a4.prev_yes_price == 9
    assert a4.quarter == "2025Q1" and a2.quarter == "2024Q4"


def test_purge_keeps_event_in_latest_split(toy):
    df, pieces = toy["df"], toy["pieces"]
    b1, b2 = _row(df, "b1", "taker"), _row(df, "b2", "taker")
    assert b1.split == "train" and bool(b1.purged)
    assert b2.split == "val" and not bool(b2.purged)
    ps = pieces["purge_summary"]
    assert ps["train"] == {
        "events": 3,
        "events_purged": 1,
        "trade_rows": 5,
        "trade_rows_purged": 1,
        "panel_rows_purged": 2,
    }
    assert ps["val"]["events_purged"] == 0 and ps["holdA"]["events_purged"] == 0


def test_outcomes_visible_in_train_null_in_holdA(toy):
    df = toy["df"]
    a2t, a2m = _row(df, "a2", "taker"), _row(df, "a2", "maker")
    assert a2t.result == "no" and bool(a2t.settled) and not bool(a2t.void)
    assert bool(a2t.won) and not bool(a2m.won)
    v = _row(df, "v1", "taker")
    assert bool(v.void) and not bool(v.settled) and pd.isna(v.won)
    h = df[df.split == "holdA"]
    assert len(h) == 4
    for col in ["result", "settled", "void", "won"]:
        assert h[col].isna().all(), col


def test_subcent_normalization(toy):
    df = toy["df"]
    t, m = _row(df, "h2", "taker"), _row(df, "h2", "maker")
    assert t.cost_cents == 100  # NO cost = 100 - stored yes (0): never understated
    assert m.cost_cents == 1  # YES cost = 100 - stored no (99)
    assert _row(df, "h2", "taker").prior_notional_usd == pytest.approx(0.21)
    assert toy["pieces"]["price_adjusted_trades_by_split"]["holdA"] == 1


def test_categories_and_series(toy):
    df = toy["df"]
    assert get_hierarchy("QQQQQ")[0] == "Other"
    h = _row(df, "h1", "taker")
    assert (h.group, h.series) == ("Other", "QQQQQ")
    a = _row(df, "a2", "taker")
    assert (a.group, a.category, a.subcategory) == get_hierarchy("KXNFLGAME")
    b = _row(df, "b1", "taker")
    assert b.group == get_hierarchy("KXINXU")[0]
    um = toy["pieces"]["unmapped_category"]
    assert um["row_share"] == pytest.approx(4 / 16)


def test_hours_columns(toy):
    df = toy["df"]
    a2 = _row(df, "a2", "taker")
    assert a2.hours_since_open == pytest.approx((_ts("2024-11-01 10:00") - _ts("2024-09-01")).total_seconds() / 3600)
    assert a2.hours_to_close == pytest.approx((_ts("2025-02-01") - _ts("2024-11-01 10:00")).total_seconds() / 3600)


def test_holdout_hashes_and_files(toy):
    hashes = holdout_file_hashes(toy["out"], "holdA", root=toy["root"])
    assert list(hashes) == ["panel/split=holdA/quarter=2025Q4/part-0.parquet"]
    assert all(len(v) == 64 for v in hashes.values())


def test_sanity_train_runs(toy):
    out = sanity_train(toy["con"], toy["out"])
    assert set(out["by_role"]) == {"maker", "taker"}
    # train non-purged settled/void positions: a2, a3, a4 (result no) and v1 (void)
    assert out["by_role"]["taker"]["positions"] == 4


def test_panel_source_reads_split(toy):
    n = toy["con"].execute(f"SELECT COUNT(*) FROM {panel_source(toy['out'], 'val')}").fetchone()[0]
    assert n == 2


def test_holdb_style_build_without_outcomes(tmp_path: Path):
    """No outcome columns in markets, no outcome splits, reverse purge via an exclusion list."""
    trades = pd.DataFrame(
        [
            _trade("x1", "KXBTCD-26AUG-X", "2026-08-10 10:00", 5, 95, "no", 3),
            _trade("y1", "KXOLD-25-Y", "2026-08-11 10:00", 7, 93, "no", 2),
        ]
    )
    markets = pd.DataFrame(
        [
            {
                "ticker": "KXBTCD-26AUG-X",
                "event_ticker": "KXBTCD-26AUG",
                "open_time": _ts("2026-08-01"),
                "close_time": _ts("2026-08-20"),
            },
            {
                "ticker": "KXOLD-25-Y",
                "event_ticker": "KXOLD-25",
                "open_time": _ts("2025-01-01"),
                "close_time": _ts("2026-12-01"),
            },
        ]
    )
    trades.to_parquet(tmp_path / "t.parquet", index=False)
    markets.to_parquet(tmp_path / "m.parquet", index=False)
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    start, end = datetime(2026, 8, 6, tzinfo=UTC), datetime(2026, 10, 5, tzinfo=UTC)
    pieces = build_panel(
        con,
        str(tmp_path / "t.parquet"),
        str(tmp_path / "m.parquet"),
        tmp_path / "pb",
        splits=[("holdB", start, end)],
        outcome_splits=(),
        exclude_events_sql="SELECT 'KXOLD-25' AS event_ticker",
        log=lambda *_: None,
    )
    df = con.execute(f"SELECT * FROM {panel_source(tmp_path / 'pb', 'holdB')}").df()
    assert len(df) == 4
    assert df[df.ticker == "KXOLD-25-Y"].purged.all()
    assert not df[df.ticker == "KXBTCD-26AUG-X"].purged.any()
    assert df.result.isna().all()
    assert pieces["purge_summary"]["holdB"]["events_purged"] == 1
    assert [(q["quarter"], q["rows"]) for q in pieces["quarters"]] == [("2026Q3", 4), ("2026Q4", 0)]


@pytest.mark.slow
@pytest.mark.skipif(not C.PANEL_MANIFEST_JSON.exists() or not C.PANEL_DIR.exists(), reason="needs the built panel")
def test_real_holdA_hashes_match_manifest():
    manifest = C.load_json(C.PANEL_MANIFEST_JSON)
    assert holdout_file_hashes(C.PANEL_DIR, "holdA") == manifest["holdA_files_sha256"]
