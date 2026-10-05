"""Descriptive audit of Kalshi `close_time` semantics (protocol section 4.6).

Question: for finalized markets, is `close_time` the *scheduled* close or the *actual* close?

The audit is restricted to events whose trades (from 2024-10-01) all fall inside the train window,
so outcome information (`result`) is only ever read for train-split events. It writes
`research/longshot_fade/results/close_time_audit.json`.

Run: uv run python -m src.research.longshot_fade.audit_close_time [--no-api]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[3]
TRADES_DIR = ROOT / "data" / "kalshi" / "trades"
MARKETS_DIR = ROOT / "data" / "kalshi" / "markets"
OUT_PATH = ROOT / "research" / "longshot_fade" / "results" / "close_time_audit.json"

SAMPLE_START = "2024-10-01 00:00:00+00"
TRAIN_END = "2025-07-01 00:00:00+00"
API_FIELDS = [
    "ticker",
    "status",
    "result",
    "open_time",
    "close_time",
    "expected_expiration_time",
    "expiration_time",
    "latest_expiration_time",
    "settlement_ts",
    "can_close_early",
    "early_close_condition",
]


def _rows(con: duckdb.DuckDBPyConnection, sql: str) -> list[dict]:
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def build_audit_tables(con: duckdb.DuckDBPyConnection, trades_dir: Path, markets_dir: Path) -> None:
    """Materialise the train-only market table with last-trade timestamps."""
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE m AS
        SELECT ticker, event_ticker, status, result, open_time, close_time
        FROM '{markets_dir}/*.parquet'
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE tt AS
        SELECT ticker, MAX(created_time) AS last_ts, MIN(created_time) AS first_ts, COUNT(*) AS n_trades
        FROM '{trades_dir}/*.parquet'
        GROUP BY ticker
        """
    )
    # Events whose latest trade (from the sample start) is before the end of train.
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE train_events AS
        SELECT m.event_ticker
        FROM tt JOIN m USING (ticker)
        WHERE tt.last_ts >= TIMESTAMPTZ '{SAMPLE_START}'
        GROUP BY m.event_ticker
        HAVING MAX(tt.last_ts) < TIMESTAMPTZ '{TRAIN_END}'
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE a AS
        SELECT m.*, tt.last_ts, tt.first_ts, tt.n_trades,
               EPOCH(m.close_time - tt.last_ts) / 3600.0 AS gap_hours,
               (EXTRACT(second FROM m.close_time) = 0 AND EXTRACT(minute FROM m.close_time) IN (0, 15, 30, 45))
                   AS round_close
        FROM m
        JOIN tt USING (ticker)
        JOIN train_events USING (event_ticker)
        WHERE m.status = 'finalized'
        """
    )


def gap_distribution(con: duckdb.DuckDBPyConnection) -> dict:
    q = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    quant_sql = ", ".join(f"QUANTILE_CONT(gap_hours, {p}) AS q{int(p * 100):02d}" for p in q)
    row = _rows(
        con,
        f"""
        SELECT COUNT(*) AS n_markets,
               {quant_sql},
               AVG((gap_hours < 0)::DOUBLE) AS share_trades_after_close,
               AVG((gap_hours BETWEEN 0 AND 1.0 / 60)::DOUBLE) AS share_close_within_1min_of_last_trade,
               AVG((gap_hours BETWEEN 0 AND 1)::DOUBLE) AS share_close_within_1h_of_last_trade,
               AVG(round_close::DOUBLE) AS share_round_close_time
        FROM a
        """,
    )[0]
    return row


def early_close_by_result(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Within multi-market events: do YES-resolving markets close before their siblings?"""
    return _rows(
        con,
        """
        WITH ev AS (
            SELECT event_ticker, MAX(close_time) AS ev_max_close, COUNT(*) AS n_mkts,
                   COUNT(DISTINCT close_time) AS n_distinct_close
            FROM a WHERE result IN ('yes', 'no')
            GROUP BY event_ticker HAVING COUNT(*) >= 2
        )
        SELECT a.result,
               COUNT(*) AS n_markets,
               AVG((a.close_time < ev.ev_max_close)::DOUBLE) AS share_close_before_event_max,
               QUANTILE_CONT(EPOCH(ev.ev_max_close - a.close_time) / 3600.0, 0.5) AS median_hours_before_event_max,
               QUANTILE_CONT(EPOCH(ev.ev_max_close - a.close_time) / 3600.0, 0.9) AS p90_hours_before_event_max,
               AVG(a.round_close::DOUBLE) AS share_round_close_time,
               AVG((ev.n_distinct_close = 1)::DOUBLE) AS share_in_events_with_single_close_time
        FROM a JOIN ev USING (event_ticker)
        WHERE a.result IN ('yes', 'no')
        GROUP BY a.result ORDER BY a.result
        """,
    )


def sample_tickers(con: duckdb.DuckDBPyConnection, per_result: int = 3) -> list[str]:
    """Deterministic sample of train markets from multi-market events, early- and on-time closers."""
    rows = _rows(
        con,
        f"""
        WITH ev AS (
            SELECT event_ticker, MAX(close_time) AS ev_max_close
            FROM a WHERE result IN ('yes', 'no') GROUP BY event_ticker HAVING COUNT(*) >= 2
        ),
        c AS (
            SELECT a.ticker, a.result, (a.close_time < ev.ev_max_close) AS early,
                   ROW_NUMBER() OVER (PARTITION BY a.result, (a.close_time < ev.ev_max_close)
                                      ORDER BY hash(a.ticker)) AS rn
            FROM a JOIN ev USING (event_ticker)
            WHERE a.result IN ('yes', 'no')
        )
        SELECT ticker FROM c WHERE rn <= {per_result} ORDER BY result, early, rn
        """,
    )
    return [r["ticker"] for r in rows]


def fetch_api_fields(tickers: list[str]) -> list[dict]:
    """Read close-related fields for a few train markets from Kalshi's public API."""
    from src.indexers.kalshi.client import KalshiClient

    out = []
    client = KalshiClient()
    try:
        for t in tickers:
            rec: dict = {"ticker": t}
            for path in (f"/historical/markets/{t}", f"/markets/{t}"):
                try:
                    data = client.http.get(path)
                except Exception as exc:  # record and try the next endpoint
                    rec.setdefault("errors", []).append(f"{path}: {type(exc).__name__}")
                    continue
                mkt = data.get("market", data)
                rec["endpoint"] = path
                rec.update({k: mkt.get(k) for k in API_FIELDS if k != "ticker"})
                rec["all_fields"] = sorted(mkt.keys())
                break
            out.append(rec)
    finally:
        client.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-api", action="store_true", help="skip the Kalshi API field check")
    args = parser.parse_args()

    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    con.execute("SET enable_progress_bar=false")
    build_audit_tables(con, TRADES_DIR, MARKETS_DIR)

    tickers = sample_tickers(con)
    local = _rows(
        con,
        "SELECT ticker, result, open_time, close_time, last_ts, gap_hours, round_close FROM a "
        f"WHERE ticker IN ({', '.join(repr(t) for t in tickers)}) ORDER BY ticker",
    )
    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope": "finalized markets in events whose trades from 2024-10-01 all precede 2025-07-01 (train only)",
        "close_minus_last_trade_hours": gap_distribution(con),
        "early_close_by_result_multi_market_events": early_close_by_result(con),
        "api_sample": {
            "tickers": tickers,
            "local": local,
            "api": [] if args.no_api else fetch_api_fields(tickers),
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(audit, indent=2, default=str) + "\n")
    print(json.dumps(audit, indent=2, default=str))


if __name__ == "__main__":
    main()
