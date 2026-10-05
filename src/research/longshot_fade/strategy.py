"""Entry rule, per-entry P&L and per-configuration summary (protocol sections 3.5 and 4).

Entry: for each market in a split, the k-th (k=1 in the base case) taker row with direction = no,
cost_cents in the band, category and liquidity filters passing, not purged, inside the split window.
P&L in cents per entry of N contracts:
    cost = N * price + fee(N, price);  payout = 100N if NO wins, N * price if void, 0 if YES wins
    net_ret = (payout - cost) / cost;  gross_ret likewise without the fee
"""

from __future__ import annotations

import math
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from src.research.longshot_fade import config as C
from src.research.longshot_fade import stats as S
from src.research.longshot_fade.fees import series_multipliers, taker_fee_cents_array

ENTRY_COLUMNS = [
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
    "cost_cents",
    "contracts",
    "prior_notional_usd",
    "prior_trades",
    "hours_since_open",
    "close_time",
    "result",
    "settled",
    "void",
]


def entry_sql(source: str, cfg: C.StrategyConfig, split: str, k: int = 1) -> str:
    """SQL selecting the k-th qualifying taker-NO print per market from a panel table expression."""
    if cfg.category_filter == "all":
        cat = ""
    elif cfg.category_filter == "ex_finance":
        cat = f"""AND "group" <> '{C.FINANCE_GROUP}'"""
    else:
        raise ValueError(f"unknown category_filter {cfg.category_filter!r}")
    cols = ", ".join(f'"{c}"' for c in ENTRY_COLUMNS)
    return f"""
        WITH q AS (
            SELECT {cols},
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY ts, trade_id) AS entry_rank
            FROM {source}
            WHERE role = 'taker'
              AND direction = 'no'
              AND NOT purged
              AND split = '{split}'
              AND cost_cents BETWEEN {int(cfg.band_lo)} AND {int(cfg.band_hi)}
              AND prior_notional_usd >= {float(cfg.liquidity_floor)}
              {cat}
        )
        SELECT * FROM q WHERE entry_rank = {int(k)} ORDER BY ticker
    """


def select_entries(
    con: duckdb.DuckDBPyConnection, source: str, cfg: C.StrategyConfig, split: str, k: int = 1
) -> pd.DataFrame:
    return con.execute(entry_sql(source, cfg, split, k)).df()


def outcome_label(result, void) -> str:
    """'no' / 'yes' if settled, 'void' if voided, otherwise 'censored' (NULL-safe)."""
    if isinstance(result, str) and result in ("yes", "no"):
        return result
    try:
        is_void = void is not None and not pd.isna(void) and bool(void)
    except (TypeError, ValueError):
        is_void = False
    return "void" if is_void else "censored"


def compute_pnl(
    entries: pd.DataFrame,
    n_contracts: int = C.N_BASE,
    slippage_cents: int = 0,
    size_cap: bool = False,
) -> pd.DataFrame:
    """Per-entry P&L. Expects columns cost_cents, contracts, series, ts, close_time, result, void.

    Rows whose market is neither settled nor void are labelled `censored` and carry NaN returns.
    """
    df = entries.copy()
    if df.empty:
        for col in ["outcome", "n", "price", "fee_cents", "cost", "payout", "net_ret", "gross_ret", "pnl_usd"]:
            df[col] = pd.Series(dtype="float64")
        df["hold_days"] = pd.Series(dtype="float64")
        return df
    df["outcome"] = [outcome_label(r, v) for r, v in zip(df["result"], df["void"])]
    n = np.full(len(df), int(n_contracts), dtype=np.int64)
    if size_cap:
        n = np.minimum(n, np.floor(df["contracts"].to_numpy(dtype=np.float64)).astype(np.int64))
    price = df["cost_cents"].to_numpy(dtype=np.int64) + int(slippage_cents)
    price = np.minimum(price, 100)
    fee = taker_fee_cents_array(n, price, series_multipliers(df["series"]))
    gross_cost = n * price
    cost = gross_cost + fee
    outcome = df["outcome"].to_numpy()
    payout = np.where(outcome == "no", 100 * n, np.where(outcome == "void", n * price, 0)).astype(np.float64)
    resolved = outcome != "censored"
    with np.errstate(divide="ignore", invalid="ignore"):
        net = np.where(resolved & (cost > 0), (payout - cost) / cost, np.nan)
        gross = np.where(resolved & (gross_cost > 0), (payout - gross_cost) / gross_cost, np.nan)
    df["n"] = n
    df["price"] = price
    df["fee_cents"] = fee
    df["cost"] = cost.astype(np.float64)
    df["payout"] = np.where(resolved, payout, np.nan)
    df["net_ret"] = net
    df["gross_ret"] = gross
    df["pnl_usd"] = np.where(resolved, (payout - cost) / 100.0, np.nan)
    close = pd.to_datetime(df["close_time"], utc=True)
    ts = pd.to_datetime(df["ts"], utc=True)
    hold = (close - ts).dt.total_seconds().to_numpy(dtype=np.float64) / 86400.0
    df["hold_days"] = np.maximum(np.nan_to_num(hold, nan=C.HOLD_DAYS_FLOOR), C.HOLD_DAYS_FLOOR)
    return df


def summarize(
    pnl: pd.DataFrame,
    iterations: int = C.BOOT_ITERATIONS,
    seed: int = C.SEED,
) -> dict[str, Any]:
    """Primary and descriptive metrics over resolved (non-censored) entries."""
    n_total = int(len(pnl))
    res = pnl[pnl["outcome"] != "censored"] if n_total else pnl
    n_censored = n_total - int(len(res))
    out: dict[str, Any] = {
        "n_entries": int(len(res)),
        "n_events": int(res["event_ticker"].nunique()) if len(res) else 0,
        "n_censored": n_censored,
        "n_void": int((res["outcome"] == "void").sum()) if len(res) else 0,
        "n_candidates": n_total,
    }
    if len(res) == 0:
        out.update(
            dict.fromkeys(
                [
                    "mean_net_ret",
                    "mean_gross_ret",
                    "se_event",
                    "t_event",
                    "p_one_sided",
                    "boot_ci_lo",
                    "boot_ci_hi",
                    "se_market",
                    "t_market",
                    "mean_net_ret_per_capital_day",
                    "win_rate",
                    "worst_event_loss_usd",
                    "sharpe_per_entry",
                    "total_pnl_usd",
                    "total_cost_usd",
                    "mean_fee_pct_of_cost",
                ],
                math.nan,
            )
        )
        return out
    r = res["net_ret"].to_numpy(dtype=np.float64)
    ev = res["event_ticker"].to_numpy()
    t_ev = S.clustered_mean_test(r, ev)
    t_mk = S.market_clustered_test(r)
    lo, hi = S.cluster_bootstrap_ci(r, ev, iterations=iterations, seed=seed)
    pnl_cents = (res["payout"] - res["cost"]).to_numpy(dtype=np.float64)
    cap_days = (res["cost"] * res["hold_days"]).to_numpy(dtype=np.float64)
    event_pnl = res.assign(_p=res["pnl_usd"]).groupby("event_ticker")["_p"].sum()
    out.update(
        {
            "mean_net_ret": t_ev.mean,
            "mean_gross_ret": float(res["gross_ret"].mean()),
            "se_event": t_ev.se,
            "t_event": t_ev.t,
            "p_one_sided": t_ev.p_one_sided,
            "boot_ci_lo": lo,
            "boot_ci_hi": hi,
            "se_market": t_mk.se,
            "t_market": t_mk.t,
            "mean_net_ret_per_capital_day": float(pnl_cents.sum() / cap_days.sum()) if cap_days.sum() > 0 else math.nan,
            "win_rate": float((r > 0).mean()),
            "worst_event_loss_usd": float(event_pnl.min()),
            "sharpe_per_entry": S.sharpe(r),
            "total_pnl_usd": float(res["pnl_usd"].sum()),
            "total_cost_usd": float(res["cost"].sum() / 100.0),
            "mean_fee_pct_of_cost": float((res["fee_cents"] / res["cost"]).mean() * 100.0),
        }
    )
    return out


def passes(summary: dict[str, Any], t_min: float = C.T_MIN, ci_lo_min: float = C.CI_LO_MIN) -> bool:
    t = summary.get("t_event")
    lo = summary.get("boot_ci_lo")
    if t is None or lo is None or math.isnan(t) or math.isnan(lo):
        return False
    return bool(t >= t_min and lo > ci_lo_min)


def run_config(
    con: duckdb.DuckDBPyConnection,
    source: str,
    cfg: C.StrategyConfig,
    split: str,
    n_contracts: int = C.N_BASE,
    outcomes: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select entries, attach outcomes if given (holdouts), compute P&L and summary."""
    entries = select_entries(con, source, cfg, split)
    if outcomes is not None:
        entries = attach_outcomes(entries, outcomes)
    pnl = compute_pnl(entries, n_contracts=n_contracts)
    return pnl, summarize(pnl)


def attach_outcomes(entries: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Replace the (NULL) outcome columns of holdout entries by a ticker-level outcome table.

    outcomes: columns ticker, result, void and optionally close_time (overrides when not null).
    """
    e = entries.drop(columns=[c for c in ("result", "settled", "void") if c in entries.columns])
    o = outcomes.set_index("ticker")
    e = e.join(o[["result", "void"]], on="ticker")
    if "close_time" in o.columns:
        ct = e["ticker"].map(o["close_time"])
        e["close_time"] = ct.where(ct.notna(), e["close_time"])
    e["settled"] = e["result"].isin(["yes", "no"])
    e["void"] = [False if v is None or pd.isna(v) else bool(v) for v in e["void"]]
    return e
