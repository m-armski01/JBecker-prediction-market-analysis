"""Build the positions table (protocol section 3.3): two rows (taker, maker) per trade.

Processing is out of core in DuckDB, one calendar quarter at a time. Per-market running state
(cumulative notional, trade count, last yes price) is carried between quarters so that the
`prior_*` features use each market's full available history.

Outcome columns (`result`, `settled`, `void`, `won`) are only joined for outcome-visible splits
(train, val). For holdout splits the market outcome table is never read and the columns are NULL.

Run: uv run python -m src.research.longshot_fade.panel
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import duckdb
import pandas as pd

from src.analysis.kalshi.util.categories import CATEGORY_SQL, get_hierarchy
from src.research.longshot_fade import config as C

PANEL_COLUMNS = [
    "trade_id",
    "ticker",
    "event_ticker",
    "series",
    "group",
    "category",
    "subcategory",
    "ts",
    "quarter",
    "split",
    "purged",
    "role",
    "direction",
    "cost_cents",
    "contracts",
    "hours_since_open",
    "prior_notional_usd",
    "prior_trades",
    "prev_yes_price",
    "hours_to_close",
    "close_time",
    "result",
    "settled",
    "void",
    "won",
]

Split = tuple[str, datetime, Optional[datetime]]


def _ts(dt: datetime) -> str:
    return f"TIMESTAMPTZ '{dt.isoformat()}'"


def _q(path: Any) -> str:
    return str(path).replace("'", "''")


def split_case_sql(splits: Sequence[Split], col: str = "ts") -> str:
    """SQL CASE mapping a timestamp to its split rank (NULL outside every split)."""
    parts = []
    for name, start, end in splits:
        cond = f"{col} >= {_ts(start)}" + (f" AND {col} < {_ts(end)}" if end is not None else "")
        parts.append(f"WHEN {cond} THEN {C.SPLIT_RANK[name]}")
    return "CASE " + " ".join(parts) + " END"


def normalized_trades_sql(trades_glob: str) -> str:
    """Trades with the sub-cent/truncation normalization (protocol section 2): when the stored
    prices sum to 99, each side's cost is 100 minus the other side's stored price."""
    return f"""
        SELECT trade_id, ticker, count, taker_side, created_time AS ts,
               CASE WHEN yes_price + no_price = 99 THEN 100 - no_price ELSE yes_price END AS yes_n,
               CASE WHEN yes_price + no_price = 99 THEN 100 - yes_price ELSE no_price END AS no_n,
               (yes_price + no_price = 99) AS price_adjusted
        FROM read_parquet('{_q(trades_glob)}')
    """


def build_category_map(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Distinct raw categories of the `mk` table mapped to (group, category, subcategory)."""
    raw = [r[0] for r in con.execute("SELECT DISTINCT raw_category FROM mk").fetchall()]
    rows = [(c, *get_hierarchy(c)) for c in raw]
    df = pd.DataFrame(rows, columns=["raw_category", "group", "category", "subcategory"])
    con.register("catmap_df", df)
    con.execute("CREATE OR REPLACE TEMP TABLE catmap AS SELECT * FROM catmap_df")
    con.unregister("catmap_df")
    return df


def build_panel(
    con: duckdb.DuckDBPyConnection,
    trades_glob: str,
    markets_glob: str,
    out_dir: Path,
    splits: Sequence[Split] = tuple(C.LOCAL_SPLITS),
    outcome_splits: Iterable[str] = C.OUTCOME_VISIBLE_SPLITS,
    history_start: datetime | None = None,
    exclude_events_sql: str | None = None,
    log=print,
) -> dict[str, Any]:
    """Write the panel to out_dir/split=<s>/quarter=<q>/part-0.parquet and return manifest pieces.

    splits: ordered (name, start, end) windows; quarters must not straddle split boundaries.
    outcome_splits: splits whose rows get outcome columns; all others get typed NULLs and the
        market outcome table is not read for them.
    history_start: trades before the first split start are used to seed prior_* state
        (None = all earlier trades in trades_glob).
    exclude_events_sql: optional SQL returning a column `event_ticker`; matching events are
        flagged purged (used for the holdB reverse purge).
    """
    outcome_splits = set(outcome_splits)
    out_dir = Path(out_dir)
    first_start = splits[0][1]
    split_case = split_case_sql(splits)

    con.execute(f"CREATE OR REPLACE TEMP VIEW tr AS {normalized_trades_sql(trades_glob)}")
    # Market attributes without outcomes.
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE mk AS
        SELECT ticker, event_ticker, open_time, close_time, {CATEGORY_SQL} AS raw_category
        FROM read_parquet('{_q(markets_glob)}')
        """
    )
    catmap = build_category_map(con)

    # Data extent: last trade timestamp closes an open-ended final split.
    max_ts = con.execute("SELECT MAX(ts) FROM tr").fetchone()[0]
    if max_ts is None:
        raise ValueError(f"no trades found in {trades_glob}")
    data_end = pd.Timestamp(max_ts).to_pydatetime() + pd.Timedelta(microseconds=1).to_pytimedelta()

    # Pass 1: trade presence per (ticker, split) -> event latest split (purge rule, no outcomes).
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE tk_split AS
        SELECT ticker, split_rank, COUNT(*) AS n_trades,
               SUM(price_adjusted::INTEGER) AS n_price_adjusted
        FROM (SELECT ticker, price_adjusted, {split_case} AS split_rank FROM tr WHERE ts >= {_ts(first_start)})
        WHERE split_rank IS NOT NULL
        GROUP BY ALL
        """
    )
    exclude_clause = ""
    if exclude_events_sql:
        con.execute(
            f"CREATE OR REPLACE TEMP TABLE excluded_events AS SELECT DISTINCT event_ticker FROM ({exclude_events_sql})"
        )
        exclude_clause = "OR e.event_ticker IN (SELECT event_ticker FROM excluded_events)"
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE ev_split AS
        SELECT m.event_ticker, MAX(s.split_rank) AS latest_rank
        FROM tk_split s JOIN mk m USING (ticker)
        GROUP BY m.event_ticker
        """
    )
    purge_rows = con.execute(
        f"""
        SELECT s.split_rank,
               COUNT(DISTINCT m.event_ticker) AS n_events,
               COUNT(DISTINCT m.event_ticker) FILTER (WHERE e.latest_rank > s.split_rank {exclude_clause})
                   AS n_events_purged,
               SUM(s.n_trades) AS n_trades,
               COALESCE(SUM(s.n_trades) FILTER (WHERE e.latest_rank > s.split_rank {exclude_clause}), 0)
                   AS n_trades_purged
        FROM tk_split s JOIN mk m USING (ticker) JOIN ev_split e USING (event_ticker)
        GROUP BY s.split_rank ORDER BY s.split_rank
        """
    ).fetchall()
    rank_to_name = {C.SPLIT_RANK[name]: name for name, _, _ in splits}
    purge_summary = {
        rank_to_name[r]: {
            "events": int(ne),
            "events_purged": int(nep),
            "trade_rows": int(nt),
            "trade_rows_purged": int(ntp),
            "panel_rows_purged": 2 * int(ntp),
        }
        for r, ne, nep, nt, ntp in purge_rows
    }

    # Seed per-market state from history before the first split.
    hist_lo = f"AND ts >= {_ts(history_start)}" if history_start is not None else ""
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE state AS
        SELECT ticker,
               SUM(count * yes_n) AS cum_cc,
               COUNT(*) AS cum_trades,
               LAST(yes_n ORDER BY ts, trade_id) AS last_yes
        FROM tr WHERE ts < {_ts(first_start)} {hist_lo}
        GROUP BY ticker
        """
    )

    if outcome_splits:
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE mk_outcome AS
            SELECT ticker, status, result FROM read_parquet('{_q(markets_glob)}')
            """
        )

    quarters_out: list[dict[str, Any]] = []
    for name, start, end in splits:
        rank = C.SPLIT_RANK[name]
        s_end = end if end is not None else max(data_end, start)
        for qlabel, qs, qe in C.quarter_windows(start, s_end):
            path = out_dir / f"split={name}" / f"quarter={qlabel}" / "part-0.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            n_src = con.execute(f"SELECT COUNT(*) FROM tr WHERE ts >= {_ts(qs)} AND ts < {_ts(qe)}").fetchone()[0]
            log(f"[panel] {name} {qlabel}: {n_src:,} trades -> {C.rel(path)}")
            _write_quarter(con, path, name, rank, qlabel, qs, qe, name in outcome_splits, exclude_clause)
            _update_state(con, qs, qe)
            stats = con.execute(
                f"""
                SELECT COUNT(*), COUNT(*) FILTER (WHERE purged), COUNT(DISTINCT ticker),
                       COUNT(DISTINCT event_ticker), COUNT(*) FILTER (WHERE "group" = 'Other')
                FROM read_parquet('{_q(path)}', hive_partitioning = false)
                """
            ).fetchone()
            if stats[0] != 2 * n_src:
                raise RuntimeError(f"{path}: expected {2 * n_src} rows, wrote {stats[0]}")
            quarters_out.append(
                {
                    "split": name,
                    "quarter": qlabel,
                    "window_start_utc": qs.isoformat(),
                    "window_end_utc": qe.isoformat(),
                    "source_trades": int(n_src),
                    "rows": int(stats[0]),
                    "rows_purged": int(stats[1]),
                    "markets": int(stats[2]),
                    "events": int(stats[3]),
                    "rows_group_other": int(stats[4]),
                    "path": C.rel(path),
                }
            )

    total_rows = sum(q["rows"] for q in quarters_out)
    other_rows = sum(q["rows_group_other"] for q in quarters_out)
    mk_other = con.execute(
        """
        SELECT COUNT(*), COUNT(*) FILTER (WHERE c."group" = 'Other')
        FROM mk JOIN catmap c USING (raw_category)
        WHERE ticker IN (SELECT ticker FROM tk_split)
        """
    ).fetchone()
    n_adj = con.execute("SELECT split_rank, SUM(n_price_adjusted) FROM tk_split GROUP BY 1 ORDER BY 1").fetchall()
    return {
        "quarters": quarters_out,
        "purge_summary": purge_summary,
        "unmapped_category": {
            "group_label": "Other",
            "row_share": (other_rows / total_rows) if total_rows else None,
            "market_share": (mk_other[1] / mk_other[0]) if mk_other[0] else None,
            "n_raw_categories": int(len(catmap)),
            "n_raw_categories_other": int((catmap["group"] == "Other").sum()),
        },
        "price_adjusted_trades_by_split": {rank_to_name[r]: int(n) for r, n in n_adj},
        "data_last_trade_utc": pd.Timestamp(max_ts).isoformat(),
    }


def _outcome_select(visible: bool) -> str:
    if visible:
        return """
            o.result AS result,
            (o.result IN ('yes', 'no')) AS settled,
            (o.status = 'finalized' AND (o.result IS NULL OR o.result = '')) AS void,
            CASE WHEN o.result IN ('yes', 'no') THEN direction = o.result END AS won
        """
    return """
            CAST(NULL AS VARCHAR) AS result,
            CAST(NULL AS BOOLEAN) AS settled,
            CAST(NULL AS BOOLEAN) AS void,
            CAST(NULL AS BOOLEAN) AS won
        """


def _write_quarter(
    con: duckdb.DuckDBPyConnection,
    path: Path,
    split: str,
    rank: int,
    qlabel: str,
    qs: datetime,
    qe: datetime,
    outcomes_visible: bool,
    exclude_clause: str,
) -> None:
    outcome_join = "LEFT JOIN mk_outcome o USING (ticker)" if outcomes_visible else ""
    con.execute(
        f"""
        COPY (
            WITH q AS (
                SELECT t.trade_id, t.ticker, t.count, t.taker_side, t.ts, t.yes_n, t.no_n,
                       (COALESCE(s.cum_cc, 0)
                        + COALESCE(SUM(t.count * t.yes_n) OVER w_prev, 0)) / 100.0 AS prior_notional_usd,
                       COALESCE(s.cum_trades, 0) + ROW_NUMBER() OVER w - 1 AS prior_trades,
                       COALESCE(LAG(t.yes_n) OVER w, s.last_yes) AS prev_yes_price
                FROM tr t LEFT JOIN state s USING (ticker)
                WHERE t.ts >= {_ts(qs)} AND t.ts < {_ts(qe)}
                WINDOW w AS (PARTITION BY t.ticker ORDER BY t.ts, t.trade_id),
                       w_prev AS (PARTITION BY t.ticker ORDER BY t.ts, t.trade_id
                                  ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
            ),
            base AS (
                SELECT q.*, m.event_ticker, m.open_time, m.close_time,
                       c."group", c.category, c.subcategory,
                       (e.latest_rank > {rank} {exclude_clause}) AS purged
                FROM q
                JOIN mk m USING (ticker)
                JOIN catmap c USING (raw_category)
                JOIN ev_split e USING (event_ticker)
            ),
            rows AS (
                SELECT base.*, 'taker' AS role, taker_side AS direction FROM base
                UNION ALL
                SELECT base.*, 'maker' AS role,
                       CASE WHEN taker_side = 'yes' THEN 'no' ELSE 'yes' END AS direction
                FROM base
            )
            SELECT
                trade_id, ticker, event_ticker,
                split_part(event_ticker, '-', 1) AS series,
                "group", category, subcategory,
                ts, '{qlabel}' AS quarter, '{split}' AS split, purged,
                role, direction,
                CASE WHEN direction = 'yes' THEN yes_n ELSE no_n END AS cost_cents,
                count AS contracts,
                EPOCH(ts - open_time) / 3600.0 AS hours_since_open,
                prior_notional_usd,
                prior_trades,
                prev_yes_price,
                EPOCH(close_time - ts) / 3600.0 AS hours_to_close,
                close_time,
                {_outcome_select(outcomes_visible)}
            FROM rows {outcome_join}
            ORDER BY ticker, ts, trade_id, role
        ) TO '{_q(path)}' (FORMAT parquet, COMPRESSION zstd)
        """
    )


def _update_state(con: duckdb.DuckDBPyConnection, qs: datetime, qe: datetime) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE state AS
        SELECT COALESCE(s.ticker, a.ticker) AS ticker,
               COALESCE(s.cum_cc, 0) + COALESCE(a.cc, 0) AS cum_cc,
               COALESCE(s.cum_trades, 0) + COALESCE(a.n, 0) AS cum_trades,
               COALESCE(a.last_yes, s.last_yes) AS last_yes
        FROM state s
        FULL OUTER JOIN (
            SELECT ticker, SUM(count * yes_n) AS cc, COUNT(*) AS n, LAST(yes_n ORDER BY ts, trade_id) AS last_yes
            FROM tr WHERE ts >= {_ts(qs)} AND ts < {_ts(qe)}
            GROUP BY ticker
        ) a ON s.ticker = a.ticker
        """
    )


def panel_glob(panel_dir: Path, split: str) -> str:
    return str(Path(panel_dir) / f"split={split}" / "*" / "*.parquet")


def panel_source(panel_dir: Path, split: str) -> str:
    """SQL table expression reading one split of the panel (partition dirs are not used as columns)."""
    return f"read_parquet('{_q(panel_glob(panel_dir, split))}', hive_partitioning = false)"


def holdout_file_hashes(panel_dir: Path, split: str, root: Path = C.ROOT) -> dict[str, str]:
    files = sorted(Path(panel_dir).glob(f"split={split}/*/*.parquet"))
    return {C.rel(f, root): C.sha256_file(f) for f in files}


def sanity_train(con: duckdb.DuckDBPyConnection, panel_dir: Path) -> dict[str, Any]:
    """Becker sign replication on train: mean maker gross_ret minus mean taker gross_ret, per position.

    gross_ret per position = (payout - cost) / cost without fees: (100 - c) / c if won, -1 if lost, 0 if void.
    Uses settled or void, non-purged positions with cost in 1-99 cents.
    """
    rows = con.execute(
        f"""
        WITH p AS (
            SELECT role,
                   CASE WHEN void THEN 0.0
                        WHEN won THEN (100.0 - cost_cents) / cost_cents
                        ELSE -1.0 END AS gross_ret
            FROM {panel_source(panel_dir, "train")}
            WHERE NOT purged AND (settled OR void) AND cost_cents BETWEEN 1 AND 99
        )
        SELECT role, COUNT(*) AS n, AVG(gross_ret) AS mean_gross_ret FROM p GROUP BY role ORDER BY role
        """
    ).fetchall()
    by_role = {r: {"positions": int(n), "mean_gross_ret": float(m)} for r, n, m in rows}
    gap = by_role["maker"]["mean_gross_ret"] - by_role["taker"]["mean_gross_ret"]
    return {
        "split": "train",
        "definition": "mean maker gross_ret minus mean taker gross_ret, per position (non-purged, settled or void, "
        "cost 1-99c); gross_ret = (100-c)/c if won, -1 if lost, 0 if void",
        "by_role": by_role,
        "maker_minus_taker_gross_ret": gap,
        "expected_sign": "positive",
        "passes": bool(gap > 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the train/val/holdA positions panel.")
    parser.add_argument("--memory-limit", default="8GB")
    args = parser.parse_args()

    proto = C.load_protocol()
    con = C.connect(memory_limit=args.memory_limit)
    trades_glob = str(C.TRADES_DIR / "*.parquet")
    markets_glob = str(C.MARKETS_DIR / "*.parquet")
    pieces = build_panel(con, trades_glob, markets_glob, C.PANEL_DIR)

    manifest = {
        "generated_at_utc": C.utc_now().isoformat(timespec="seconds"),
        "built_from_head": C.git_head(),
        "working_tree_clean_at_build": C.git_is_clean(),
        "protocol_json_sha256": C.sha256_file(C.PROTOCOL_JSON),
        "panel_dir": C.rel(C.PANEL_DIR),
        "columns": PANEL_COLUMNS,
        "splits": {
            name: {"start_utc": s.isoformat(), "end_utc_exclusive": e.isoformat() if e else None}
            for name, s, e in C.LOCAL_SPLITS
        },
        "row_counts": pieces["quarters"],
        "purge_summary": pieces["purge_summary"],
        "unmapped_category": pieces["unmapped_category"],
        "price_adjusted_trades_by_split": pieces["price_adjusted_trades_by_split"],
        "data_last_trade_utc": pieces["data_last_trade_utc"],
        "holdA_files_sha256": holdout_file_hashes(C.PANEL_DIR, "holdA"),
    }
    assert proto["protocol_version"] == 1
    C.write_json(C.PANEL_MANIFEST_JSON, manifest)
    C.write_json(C.RESULTS_DIR / "purge_summary.json", pieces["purge_summary"])
    sanity = sanity_train(con, C.PANEL_DIR)
    C.write_json(C.RESULTS_DIR / "sanity_train.json", sanity)
    print(f"[panel] manifest -> {C.rel(C.PANEL_MANIFEST_JSON)}")
    print(
        f"[panel] sanity (train): maker - taker gross_ret = {sanity['maker_minus_taker_gross_ret']:.6f} "
        f"-> {'PASS' if sanity['passes'] else 'FAIL (stop and investigate)'}"
    )


if __name__ == "__main__":
    main()
