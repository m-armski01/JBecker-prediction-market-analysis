"""Commit 3: run all 12 configurations on train -> results/train_grid.csv.

Ranking: t_event descending, ties by n_events descending, then protocol grid order. The top 3 go to validation.

Run: uv run python -m src.research.longshot_fade.run_train
"""

from __future__ import annotations

import math
from typing import Any

import duckdb
import pandas as pd

from src.research.longshot_fade import config as C
from src.research.longshot_fade import stats as S
from src.research.longshot_fade import strategy as ST
from src.research.longshot_fade.panel import panel_source

GRID_COLUMNS = [
    "rank",
    "grid_order",
    "config_id",
    "band_lo",
    "band_hi",
    "category_filter",
    "liquidity_floor",
    "n_entries",
    "n_events",
    "mean_net_ret",
    "mean_gross_ret",
    "se_event",
    "t_event",
    "p_one_sided",
    "p_bonferroni",
    "boot_ci_lo",
    "boot_ci_hi",
    "mean_net_ret_per_capital_day",
    "win_rate",
    "worst_event_loss_usd",
    "se_market",
    "t_market",
    "n_censored",
    "n_void",
    "n_candidates",
    "sharpe_per_entry",
    "mean_fee_pct_of_cost",
    "total_pnl_usd",
    "total_cost_usd",
]


def evaluate_grid(con: duckdb.DuckDBPyConnection, split: str, configs: list[C.StrategyConfig]) -> pd.DataFrame:
    source = panel_source(C.PANEL_DIR, split)
    rows: list[dict[str, Any]] = []
    for i, cfg in enumerate(configs):
        _, summ = ST.run_config(con, source, cfg, split)
        rows.append(
            {
                "grid_order": i,
                "config_id": cfg.id,
                "band_lo": cfg.band_lo,
                "band_hi": cfg.band_hi,
                "category_filter": cfg.category_filter,
                "liquidity_floor": cfg.liquidity_floor,
                **summ,
                "p_bonferroni": S.bonferroni(summ["p_one_sided"], C.BONFERRONI_M),
            }
        )
        print(
            f"[{split}] {cfg.id:28s} n={summ['n_entries']:>7} events={summ['n_events']:>6} "
            f"mean_net={summ['mean_net_ret']:+.5f} t={summ['t_event']:+.3f}"
        )
    return pd.DataFrame(rows)


def rank_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Rank by t_event desc, n_events desc, grid order asc (NaN t ranks last)."""
    key_t = df["t_event"].fillna(-math.inf)
    order = sorted(range(len(df)), key=lambda i: (-key_t.iloc[i], -df["n_events"].iloc[i], df["grid_order"].iloc[i]))
    ranks = {idx: r + 1 for r, idx in enumerate(order)}
    out = df.copy()
    out["rank"] = [ranks[i] for i in range(len(df))]
    return out.sort_values("rank").reset_index(drop=True)


def main() -> None:
    proto = C.load_protocol()
    configs = C.grid(proto)
    con = C.connect()
    df = rank_grid(evaluate_grid(con, "train", configs))
    path = C.RESULTS_DIR / "train_grid.csv"
    df[GRID_COLUMNS].to_csv(path, index=False, float_format="%.10g")
    print(f"[train] -> {C.rel(path)}")
    top = df.head(C.TOP_K)
    print("[train] top 3 to validation:", ", ".join(top["config_id"]))


if __name__ == "__main__":
    main()
