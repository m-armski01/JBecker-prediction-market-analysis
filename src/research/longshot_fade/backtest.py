"""Descriptive build-out after the verdict (protocol section 5, commit 8+): portfolio backtest,
sensitivities, breakdowns and the train-only Becker replication.

Uses the frozen configuration when one exists. If validation selected nothing (no frozen.json), it
uses the train rank-1 configuration as a clearly labelled *reference* (DEVIATIONS D18). Each split is
reported separately (never pooled). Holdout outcomes are read only from the files `unlock.py` wrote
(holdA_outcomes/, holdB/outcomes/); when the holdouts were never unlocked those files do not exist and
only train and val are reported. Nothing here changes frozen.json or any committed result.

Run: uv run python -m src.research.longshot_fade.backtest
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.research.longshot_fade import config as C
from src.research.longshot_fade import strategy as ST
from src.research.longshot_fade.panel import panel_source

PRICE_BUCKETS = [(90, 92), (93, 94), (95, 96), (97, 99)]


def reference_config() -> tuple[C.StrategyConfig, str]:
    """Frozen configuration if it exists, else the train rank-1 configuration (validation failed)."""
    if C.FROZEN_JSON.exists():
        return C.StrategyConfig.from_dict(C.load_json(C.FROZEN_JSON)["config"]), "frozen"
    sel = C.load_json(C.RESULTS_DIR / "val_selection.json")
    if sel.get("selected"):
        raise SystemExit("[backtest] a configuration was selected but frozen.json is missing: run freeze.py first")
    by_id = {c.id: c for c in C.grid()}
    return by_id[sel["candidates"][0]], "reference (train rank 1; not frozen - validation failed)"


def split_sources() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {
        "train": {"source": panel_source(C.PANEL_DIR, "train"), "outcomes": None},
        "val": {"source": panel_source(C.PANEL_DIR, "val"), "outcomes": None},
    }
    a = C.HOLDA_OUTCOMES_DIR / "market_outcomes.parquet"
    if a.exists():
        out["holdA"] = {"source": panel_source(C.PANEL_DIR, "holdA"), "outcomes": pd.read_parquet(a)}
    b = C.HOLDB_DIR / "outcomes" / "market_outcomes.parquet"
    if b.exists():
        out["holdB"] = {"source": panel_source(C.HOLDB_DIR / "panel", "holdB"), "outcomes": pd.read_parquet(b)}
    return out


def entries_for(con, spec: dict[str, Any], cfg: C.StrategyConfig, split: str, k: int = 1) -> pd.DataFrame:
    e = ST.select_entries(con, spec["source"], cfg, split, k=k)
    if spec["outcomes"] is not None:
        e = ST.attach_outcomes(e, spec["outcomes"])
    return e


def portfolio(pnl: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """Concurrent positions from entry ts to settlement (close_time); realised P&L on settlement dates."""
    res = pnl[pnl["outcome"] != "censored"].copy()
    if res.empty:
        return {"n_positions": 0}, pd.DataFrame()
    res["entry_ts"] = pd.to_datetime(res["ts"], utc=True)
    res["settle_ts"] = res["entry_ts"] + pd.to_timedelta(res["hold_days"], unit="D")
    ev = pd.concat(
        [
            pd.DataFrame({"t": res["entry_ts"], "capital": res["cost"] / 100.0, "pnl": 0.0}),
            pd.DataFrame({"t": res["settle_ts"], "capital": -res["cost"] / 100.0, "pnl": res["pnl_usd"]}),
        ]
    ).sort_values("t", kind="mergesort")
    ev["capital_in_use"] = ev["capital"].cumsum()
    ev["cum_pnl"] = ev["pnl"].cumsum()
    peak_cap = float(ev["capital_in_use"].max())
    cum = ev["cum_pnl"].to_numpy()
    drawdown = float((np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:] - cum).max())
    total = float(res["pnl_usd"].sum())
    event_pnl = res.groupby("event_ticker")["pnl_usd"].sum().sort_values()
    cap_days = float((res["cost"] / 100.0 * res["hold_days"]).sum())
    daily = (
        ev.assign(date=ev["t"].dt.floor("D"))
        .groupby("date")
        .agg(realized_pnl_usd=("pnl", "sum"), capital_in_use_end_usd=("capital_in_use", "last"))
        .reset_index()
    )
    daily["cum_realized_pnl_usd"] = daily["realized_pnl_usd"].cumsum()
    abs_ev = event_pnl.abs()
    summary = {
        "n_positions": int(len(res)),
        "n_events": int(event_pnl.size),
        "mean_entry_price_cents": float(res["price"].mean()),
        "win_rate": float((res["outcome"] == "no").mean()),
        "breakeven_win_rate_incl_fees": float((res["cost"] / (100.0 * res["n"])).mean()),
        "total_realized_pnl_usd": total,
        "total_cost_usd": float(res["cost"].sum() / 100.0),
        "peak_capital_in_use_usd": peak_cap,
        "return_on_peak_capital": total / peak_cap if peak_cap > 0 else math.nan,
        "return_per_capital_day": total / cap_days if cap_days > 0 else math.nan,
        "max_drawdown_realized_usd": drawdown,
        "worst_10_events": [{"event_ticker": k, "pnl_usd": float(v)} for k, v in event_pnl.head(10).items()],
        "largest_event_share_of_total_pnl": float(event_pnl.max() / total) if total > 0 else math.nan,
        "largest_abs_event_share_of_gross_abs_pnl": float(abs_ev.max() / abs_ev.sum())
        if abs_ev.sum() > 0
        else math.nan,
        "first_entry_utc": res["entry_ts"].min().isoformat(),
        "last_settlement_utc": res["settle_ts"].max().isoformat(),
    }
    return summary, daily


def _summ_row(split: str, variant: str, pnl: pd.DataFrame) -> dict[str, Any]:
    s = ST.summarize(pnl)
    keep = [
        "n_entries",
        "n_events",
        "n_censored",
        "n_void",
        "mean_net_ret",
        "mean_gross_ret",
        "t_event",
        "boot_ci_lo",
        "boot_ci_hi",
        "t_market",
        "win_rate",
        "mean_net_ret_per_capital_day",
        "total_pnl_usd",
        "mean_fee_pct_of_cost",
    ]
    return {"split": split, "variant": variant, **{k: s[k] for k in keep}}


def sensitivities(con, spec, cfg, split: str, base_entries: pd.DataFrame) -> list[dict[str, Any]]:
    rows = [_summ_row(split, "base_N100", ST.compute_pnl(base_entries, n_contracts=C.N_BASE))]
    for n in C.SENSITIVITY_SIZES:
        rows.append(_summ_row(split, f"N{n}", ST.compute_pnl(base_entries, n_contracts=n)))
    rows.append(_summ_row(split, "slippage_plus_1c", ST.compute_pnl(base_entries, slippage_cents=1)))
    second = entries_for(con, spec, cfg, split, k=2)
    rows.append(_summ_row(split, "second_qualifying_print", ST.compute_pnl(second)))
    rows.append(_summ_row(split, "size_cap_min_N_print", ST.compute_pnl(base_entries, size_cap=True)))
    base = ST.compute_pnl(base_entries)
    rows.append(_summ_row(split, "exclude_void", base[base["outcome"] != "void"]))
    return rows


def breakdowns(split: str, pnl: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    pnl = pnl.copy()
    pnl["price_bucket"] = pd.cut(
        pnl["cost_cents"], bins=[89, 92, 94, 96, 99], labels=[f"{a}-{b}" for a, b in PRICE_BUCKETS]
    ).astype(str)
    hso = pnl["hours_since_open"]
    pnl["hours_since_open_bucket"] = np.where(hso < 1, "<1h", np.where(hso <= 24, "1-24h", ">24h"))
    pnl["first_trade_in_market"] = np.where(pnl["prior_trades"] == 0, "first trade", "later trade")
    for dim in ["group", "price_bucket", "quarter", "hours_since_open_bucket", "first_trade_in_market"]:
        for key, g in pnl.groupby(dim):
            s = ST.summarize(g, iterations=100)
            rows.append(
                {
                    "split": split,
                    "dimension": dim,
                    "value": key,
                    "n_entries": s["n_entries"],
                    "n_events": s["n_events"],
                    "mean_net_ret": s["mean_net_ret"],
                    "t_event": s["t_event"],
                    "win_rate": s["win_rate"],
                    "total_pnl_usd": s["total_pnl_usd"],
                }
            )
    return rows


def becker_replication(con) -> tuple[dict[str, Any], pd.DataFrame]:
    """Train only: mean taker/maker gross_ret per position and the YES-NO gap at equal cost basis."""
    q = f"""
        WITH p AS (
            SELECT role, direction, cost_cents,
                   CASE WHEN void THEN 0.0 WHEN won THEN (100.0 - cost_cents) / cost_cents ELSE -1.0 END AS gross_ret,
                   CASE WHEN void THEN 0.0 WHEN won THEN 100.0 - cost_cents ELSE -cost_cents END AS excess_cents
            FROM {panel_source(C.PANEL_DIR, "train")}
            WHERE NOT purged AND (settled OR void) AND cost_cents BETWEEN 1 AND 99
        )
        SELECT role, direction, cost_cents, COUNT(*) AS n, AVG(gross_ret) AS mean_gross_ret,
               AVG(excess_cents) AS mean_excess_cents
        FROM p GROUP BY ALL ORDER BY role, direction, cost_cents
    """
    df = con.execute(q).df()
    by_role = (
        df.assign(w=df["n"] * df["mean_gross_ret"]).groupby("role")[["w", "n"]].sum().assign(m=lambda x: x.w / x.n)["m"]
    )
    piv = (
        df.assign(w=df["n"] * df["mean_gross_ret"])
        .groupby(["direction", "cost_cents"], as_index=False)[["w", "n"]]
        .sum()
        .assign(mean_gross_ret=lambda x: x["w"] / x["n"])
    )
    wide = piv.pivot(index="cost_cents", columns="direction", values=["mean_gross_ret", "n"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide["yes_minus_no_gross_ret"] = wide["mean_gross_ret_yes"] - wide["mean_gross_ret_no"]
    w = wide[["n_yes", "n_no"]].min(axis=1)
    # Context for the entry rule: every taker-NO print at 90-99c (not only the first per market).
    tno = df[(df["role"] == "taker") & (df["direction"] == "no") & (df["cost_cents"] >= 90)]
    taker_no = {
        "positions": int(tno["n"].sum()),
        "mean_gross_ret": float((tno["n"] * tno["mean_gross_ret"]).sum() / tno["n"].sum()),
        "by_cost": {int(c): float(m) for c, m in zip(tno["cost_cents"], tno["mean_gross_ret"])},
    }
    summary = {
        "split": "train",
        "mean_gross_ret_taker": float(by_role.get("taker", math.nan)),
        "mean_gross_ret_maker": float(by_role.get("maker", math.nan)),
        "maker_minus_taker": float(by_role.get("maker", math.nan) - by_role.get("taker", math.nan)),
        "yes_minus_no_gap_weighted_mean": float((wide["yes_minus_no_gross_ret"] * w).sum() / w.sum()),
        "share_of_cost_levels_yes_below_no": float((wide["yes_minus_no_gross_ret"] < 0).mean()),
        "taker_no_all_prints_90_99": taker_no,
        "note": "positions per trade (taker and maker rows); equal-cost comparison pools taker and maker rows",
    }
    return summary, wide


def crosscheck_raw_train(con) -> dict[str, Any]:
    """Recompute b95-99_all_liq0 on train directly from the raw trades/markets files (no panel).

    Train has no sum-99 prices, so raw prices equal normalised ones. Events are train-only when their last
    trade from 2024-10-01 precedes 2025-07-01 (the purge rule).
    """
    row = con.execute(
        f"""
        WITH m AS (SELECT ticker, event_ticker, result FROM read_parquet('{C.MARKETS_DIR}/*.parquet')),
        t AS (
            SELECT t.trade_id, t.ticker, t.created_time, t.taker_side, t.no_price, m.event_ticker, m.result
            FROM read_parquet('{C.TRADES_DIR}/*.parquet') t JOIN m USING (ticker)
            WHERE t.created_time >= TIMESTAMPTZ '{C.SAMPLE_START.isoformat()}'
        ),
        ev AS (SELECT event_ticker FROM t GROUP BY 1 HAVING MAX(created_time) < TIMESTAMPTZ '{C.VAL_START.isoformat()}'),
        cand AS (
            SELECT t.*, ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY created_time, trade_id) AS rn
            FROM t JOIN ev USING (event_ticker)
            WHERE taker_side = 'no' AND no_price BETWEEN 95 AND 99
        )
        SELECT COUNT(*) AS candidates,
               COUNT(*) FILTER (WHERE result IN ('yes', 'no')) AS settled,
               AVG(CASE WHEN result = 'no' THEN 100.0 / no_price - 1 WHEN result = 'yes' THEN -1.0 END) AS gross_settled
        FROM cand WHERE rn = 1
        """
    ).fetchone()
    grid = pd.read_csv(C.RESULTS_DIR / "train_grid.csv").set_index("config_id").loc["b95-99_all_liq0"]
    return {
        "config_id": "b95-99_all_liq0",
        "split": "train",
        "raw_candidates": int(row[0]),
        "raw_settled": int(row[1]),
        "raw_mean_gross_ret_settled": float(row[2]),
        "panel_n_candidates": int(grid["n_candidates"]),
        "panel_n_entries_settled_or_void": int(grid["n_entries"]),
        "panel_n_void": int(grid["n_void"]),
        "panel_mean_gross_ret_incl_void": float(grid["mean_gross_ret"]),
        "match_candidates": int(row[0]) == int(grid["n_candidates"]),
        "match_settled": int(row[1]) == int(grid["n_entries"]) - int(grid["n_void"]),
    }


def main() -> None:
    cfg, label = reference_config()
    con = C.connect()
    sens_rows, bd_rows, port = [], [], {}
    for split, spec in split_sources().items():
        e = entries_for(con, spec, cfg, split)
        pnl = ST.compute_pnl(e, n_contracts=C.N_BASE)
        summary, daily = portfolio(pnl)
        port[split] = summary
        if not daily.empty:
            daily.to_csv(C.RESULTS_DIR / f"pnl_curve_{split}.csv", index=False, float_format="%.6g")
        sens_rows += sensitivities(con, spec, cfg, split, e)
        bd_rows += breakdowns(split, pnl)
        print(f"[backtest] {split}: positions={summary.get('n_positions')} pnl={summary.get('total_realized_pnl_usd')}")
    C.write_json(
        C.RESULTS_DIR / "backtest_portfolio.json",
        {"config": cfg.to_dict(), "config_label": label, "n_contracts": C.N_BASE, "splits": port},
    )
    pd.DataFrame(sens_rows).to_csv(C.RESULTS_DIR / "sensitivities.csv", index=False, float_format="%.8g")
    pd.DataFrame(bd_rows).to_csv(C.RESULTS_DIR / "breakdowns.csv", index=False, float_format="%.8g")
    C.write_json(C.RESULTS_DIR / "crosscheck_train_raw.json", crosscheck_raw_train(con))
    becker, wide = becker_replication(con)
    C.write_json(C.RESULTS_DIR / "becker_replication_train.json", becker)
    wide.to_csv(C.RESULTS_DIR / "becker_yes_no_by_cost_train.csv", index=False, float_format="%.8g")


if __name__ == "__main__":
    main()
