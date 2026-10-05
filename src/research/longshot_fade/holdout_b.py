"""Holdout B: blind pull of fresh Kalshi trades after the freeze (protocol sections 3.1 and 5, commit 7).

Steps (all with frozen code, run at HEAD = `holdout-a-run`):
  pull   - read the live-endpoint cutoff from GET /historical/cutoff, then page through
           GET /markets/trades for [cutoff, pull start) with a resumable cursor; fetch market
           metadata for every traded ticker. No outcome field is stored.
  build  - normalise to the panel schema (integer cents via exact decimal arithmetic, block
           trades dropped), build the holdB panel with panel.build_panel (outcome columns NULL,
           reverse purge of events already seen locally) and write a SHA-256 manifest to
           data/research/longshot_fade/holdB/panel_manifest.json.

Outcomes are fetched later, once, by `unlock.py --holdout B` (at least 7 days after the pull).

Run: uv run python -m src.research.longshot_fade.holdout_b pull|build
"""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import Any

import duckdb
import pandas as pd

from src.research.longshot_fade import config as C
from src.research.longshot_fade.panel import _q, build_panel, panel_glob

RAW_DIR = C.HOLDB_DIR / "raw"
NORM_DIR = C.HOLDB_DIR / "normalized"
PANEL_B_DIR = C.HOLDB_DIR / "panel"
STATE_JSON = C.HOLDB_DIR / "pull_state.json"
MANIFEST_JSON = C.HOLDB_DIR / "panel_manifest.json"

# Raw trade fields kept verbatim (strings); integer legacy fields are kept if the API still sends them.
RAW_TRADE_FIELDS = [
    "trade_id",
    "ticker",
    "count_fp",
    "count",
    "yes_price_dollars",
    "no_price_dollars",
    "yes_price",
    "no_price",
    "taker_side",
    "created_time",
    "is_block_trade",
]
# Market metadata kept (no outcome fields: no status, result, settlement or price fields).
MARKET_FIELDS = ["ticker", "event_ticker", "market_type", "open_time", "close_time", "created_time"]
CHUNK_TRADES = 200_000


def raw_trade_record(t: dict[str, Any]) -> dict[str, str | None]:
    """Keep the raw trade fields as strings (None when absent)."""
    out: dict[str, str | None] = {}
    for k in RAW_TRADE_FIELDS:
        v = t.get(k)
        if isinstance(v, bool):
            v = "true" if v else "false"
        out[k] = None if v is None else str(v)
    return out


def market_record(m: dict[str, Any]) -> dict[str, str | None]:
    """Strip a market payload to non-outcome metadata."""
    return {k: (None if m.get(k) is None else str(m.get(k))) for k in MARKET_FIELDS}


NORMALIZE_TRADES_SQL = """
    SELECT trade_id, ticker,
           COALESCE(TRY_CAST(count_fp AS DOUBLE), TRY_CAST("count" AS DOUBLE)) AS "count",
           CAST(FLOOR(COALESCE(TRY_CAST(yes_price_dollars AS DECIMAL(18, 6)) * 100,
                               TRY_CAST(yes_price AS DECIMAL(18, 6)))) AS INTEGER) AS yes_price,
           CAST(FLOOR(COALESCE(TRY_CAST(no_price_dollars AS DECIMAL(18, 6)) * 100,
                               TRY_CAST(no_price AS DECIMAL(18, 6)))) AS INTEGER) AS no_price,
           lower(taker_side) AS taker_side,
           CAST(created_time AS TIMESTAMPTZ) AS created_time,
           COALESCE(lower(is_block_trade), 'false') = 'true' AS is_block_trade
    FROM {source}
    QUALIFY ROW_NUMBER() OVER (PARTITION BY trade_id ORDER BY created_time) = 1
"""


def normalize_trades(con: duckdb.DuckDBPyConnection, source: str) -> duckdb.DuckDBPyRelation:
    """Exact-decimal normalisation of raw trades to the local schema (integer cents, floor).

    Floors each side to whole cents like the local data; the panel's sum-99 rule then applies the
    same conservative cost to sub-penny prints as for holdA.
    """
    return con.sql(NORMALIZE_TRADES_SQL.format(source=source))


def normalize_trades_df(raw: pd.DataFrame) -> pd.DataFrame:
    """normalize_trades on an in-memory frame of raw string records (used by tests)."""
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    for k in RAW_TRADE_FIELDS:
        if k not in raw.columns:
            raw = raw.assign(**{k: None})
    raw = raw[RAW_TRADE_FIELDS].astype("object")
    con.register("raw_df", raw)
    return normalize_trades(con, "raw_df").df()


# --------------------------------------------------------------------------------------------
# Pull
# --------------------------------------------------------------------------------------------


def _epoch(dt: datetime) -> int:
    return int(dt.timestamp())


def _require_head(tag: str) -> str:
    head = C.git_head()
    tag_commit = C.git("rev-parse", f"{tag}^{{commit}}")
    if head != tag_commit:
        raise SystemExit(f"[holdB] HEAD {head[:10]} is not the commit tagged {tag}; holdB must use frozen code")
    if not C.git_is_clean():
        raise SystemExit("[holdB] working tree is not clean")
    return head


def pull(max_pages: int | None = None) -> dict[str, Any]:
    from src.indexers.kalshi.client import KalshiClient

    head = _require_head(C.TAG_HOLDOUT_A)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    state = C.load_json(STATE_JSON) if STATE_JSON.exists() else {}
    client = KalshiClient()
    try:
        if "window_start_utc" not in state:
            cutoff = client.http.get("/historical/cutoff")
            start = datetime.fromisoformat(cutoff["trades_created_ts"].replace("Z", "+00:00"))
            end = C.utc_now().replace(microsecond=0)
            state = {
                "head": head,
                "cutoff_response": cutoff,
                "window_start_utc": start.isoformat(),
                "window_end_utc": end.isoformat(),
                "pull_started_at_utc": end.isoformat(),
                "trades_cursor": None,
                "trades_done": False,
                "chunk_index": 0,
                "raw_trades": 0,
                "pages": 0,
            }
            C.write_json(STATE_JSON, state)
        start = datetime.fromisoformat(state["window_start_utc"])
        end = datetime.fromisoformat(state["window_end_utc"])

        buf: list[dict] = []
        pages_this_run = 0
        while not state["trades_done"]:
            params: dict[str, Any] = {"limit": 1000, "min_ts": _epoch(start), "max_ts": _epoch(end)}
            if state["trades_cursor"]:
                params["cursor"] = state["trades_cursor"]
            data = client.http.get("/markets/trades", params=params)
            buf.extend(raw_trade_record(t) for t in data.get("trades", []))
            cursor = data.get("cursor") or None
            state["pages"] += 1
            pages_this_run += 1
            done = cursor is None
            if len(buf) >= CHUNK_TRADES or done:
                if buf:
                    path = RAW_DIR / f"trades_{state['chunk_index']:06d}.parquet"
                    pd.DataFrame(buf, columns=RAW_TRADE_FIELDS).to_parquet(path, index=False)
                    state["chunk_index"] += 1
                    state["raw_trades"] += len(buf)
                    buf = []
                state["trades_cursor"] = cursor
                state["trades_done"] = done
                C.write_json(STATE_JSON, state)
                print(f"[holdB] pages={state['pages']} raw_trades={state['raw_trades']:,}")
            if max_pages is not None and pages_this_run >= max_pages and not done:
                # Unsaved buffered pages are re-fetched on resume from the last saved cursor.
                print(f"[holdB] stopping after {pages_this_run} pages (resumable)")
                return state

        if not state.get("markets_done"):
            con = duckdb.connect()
            tickers = [
                r[0]
                for r in con.execute(
                    f"SELECT DISTINCT ticker FROM read_parquet('{_q(RAW_DIR)}/trades_*.parquet') ORDER BY 1"
                ).fetchall()
            ]
            markets = fetch_markets(client, tickers)
            pd.DataFrame(markets, columns=MARKET_FIELDS).to_parquet(C.HOLDB_DIR / "markets.parquet", index=False)
            state["markets_requested"] = len(tickers)
            state["markets_fetched"] = len(markets)
            state["markets_done"] = True
            state["pull_completed_at_utc"] = C.utc_now().isoformat(timespec="seconds")
            C.write_json(STATE_JSON, state)
    finally:
        client.close()
    return state


def fetch_markets(client, tickers: list[str], batch: int = 100) -> list[dict]:
    """Market metadata via GET /markets?tickers=... in batches, falling back to GET /markets/{ticker}."""
    import httpx

    got: dict[str, dict] = {}
    for i in range(0, len(tickers), batch):
        chunk = tickers[i : i + batch]
        try:
            data = client.http.get("/markets", params={"tickers": ",".join(chunk), "limit": len(chunk)})
            for m in data.get("markets", []):
                if m.get("ticker") in chunk:
                    got[m["ticker"]] = market_record(m)
        except httpx.HTTPStatusError:
            pass
        if (i // batch) % 50 == 0:
            print(f"[holdB] markets {len(got):,}/{len(tickers):,}")
    for t in tickers:
        if t in got:
            continue
        for path in (f"/markets/{t}", f"/historical/markets/{t}"):
            try:
                data = client.http.get(path)
            except httpx.HTTPStatusError:
                continue
            got[t] = market_record(data.get("market", data))
            break
    return [got[t] for t in tickers if t in got]


# --------------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------------


def build() -> dict[str, Any]:
    head = _require_head(C.TAG_HOLDOUT_A)
    state = C.load_json(STATE_JSON)
    if not state.get("markets_done"):
        raise SystemExit("[holdB] pull is not complete")
    start = datetime.fromisoformat(state["window_start_utc"])
    end = datetime.fromisoformat(state["window_end_utc"])
    con = C.connect()
    NORM_DIR.mkdir(parents=True, exist_ok=True)

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE mkt AS
        SELECT ticker, event_ticker, market_type,
               CAST(open_time AS TIMESTAMPTZ) AS open_time,
               CAST(close_time AS TIMESTAMPTZ) AS close_time,
               CAST(created_time AS TIMESTAMPTZ) AS created_time
        FROM read_parquet('{_q(C.HOLDB_DIR / "markets.parquet")}')
        """
    )
    raw_src = f"read_parquet('{_q(RAW_DIR)}/trades_*.parquet', union_by_name = true)"
    con.execute(f"CREATE OR REPLACE TEMP TABLE nt AS {NORMALIZE_TRADES_SQL.format(source=raw_src)}")
    counts = con.execute(
        f"""
        SELECT COUNT(*) AS normalized,
               COUNT(*) FILTER (WHERE is_block_trade) AS block_trades,
               COUNT(*) FILTER (WHERE NOT is_block_trade AND ticker NOT IN (SELECT ticker FROM mkt)) AS no_market,
               COUNT(*) FILTER (WHERE created_time < TIMESTAMPTZ '{start.isoformat()}'
                                   OR created_time >= TIMESTAMPTZ '{end.isoformat()}') AS outside_window,
               COUNT(*) FILTER (WHERE yes_price + no_price NOT IN (99, 100)) AS bad_price_sum,
               COUNT(*) FILTER (WHERE taker_side NOT IN ('yes', 'no')) AS bad_side
        FROM nt
        """
    ).fetchone()
    trades_path = NORM_DIR / "trades.parquet"
    con.execute(
        f"""
        COPY (
            SELECT trade_id, ticker, "count", yes_price, no_price, taker_side, created_time
            FROM nt
            WHERE NOT is_block_trade
              AND ticker IN (SELECT ticker FROM mkt)
              AND created_time >= TIMESTAMPTZ '{start.isoformat()}'
              AND created_time < TIMESTAMPTZ '{end.isoformat()}'
              AND yes_price + no_price IN (99, 100)
              AND taker_side IN ('yes', 'no')
        ) TO '{_q(trades_path)}' (FORMAT parquet, COMPRESSION zstd)
        """
    )
    markets_path = NORM_DIR / "markets.parquet"
    con.execute(f"COPY (SELECT * FROM mkt) TO '{_q(markets_path)}' (FORMAT parquet)")

    local_events = f"SELECT DISTINCT event_ticker FROM read_parquet('{_q(panel_glob(C.PANEL_DIR, '*'))}', hive_partitioning = false)"
    pieces = build_panel(
        con,
        str(trades_path),
        str(markets_path),
        PANEL_B_DIR,
        splits=[("holdB", start, end)],
        outcome_splits=(),
        exclude_events_sql=local_events,
    )
    hashes = {C.rel(f): C.sha256_file(f) for f in sorted(PANEL_B_DIR.glob("split=holdB/*/*.parquet"))}
    manifest = {
        "generated_at_utc": C.utc_now().isoformat(timespec="seconds"),
        "built_from_head": head,
        "window": {"start_utc": start.isoformat(), "end_utc_exclusive": end.isoformat()},
        "cutoff_response": state["cutoff_response"],
        "pull_started_at_utc": state["pull_started_at_utc"],
        "pull_completed_at_utc": state["pull_completed_at_utc"],
        "raw_trades": state["raw_trades"],
        "pages": state["pages"],
        "markets_requested": state["markets_requested"],
        "markets_fetched": state["markets_fetched"],
        "normalization": dict(
            zip(["normalized", "block_trades", "no_market", "outside_window", "bad_price_sum", "bad_side"], counts)
        ),
        "row_counts": pieces["quarters"],
        "purge_summary_reverse": pieces["purge_summary"],
        "unmapped_category": pieces["unmapped_category"],
        "price_adjusted_trades_by_split": pieces["price_adjusted_trades_by_split"],
        "holdB_files_sha256": hashes,
    }
    C.write_json(MANIFEST_JSON, manifest)
    print(f"[holdB] manifest -> {C.rel(MANIFEST_JSON)}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Holdout B pull and build (frozen code only).")
    parser.add_argument("step", choices=["pull", "build"])
    parser.add_argument("--max-pages", type=int, default=None, help="stop the trade pull after N pages (resumable)")
    args = parser.parse_args()
    if args.step == "pull":
        state = pull(args.max_pages)
        print({k: v for k, v in state.items() if k != "trades_cursor"})
    else:
        build()


if __name__ == "__main__":
    main()
