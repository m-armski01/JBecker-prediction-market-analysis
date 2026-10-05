"""Commit 4: run the train top 3 on validation and apply the selection rule.

Selection: among the 3, keep those with validation mean_net_ret > 0 and choose the highest validation t_event.
If none qualifies the verdict is "Not supported (failed validation)" and the holdouts stay locked.

Run: uv run python -m src.research.longshot_fade.run_validate
"""

from __future__ import annotations

import math

import pandas as pd

from src.research.longshot_fade import config as C
from src.research.longshot_fade.run_train import GRID_COLUMNS, evaluate_grid

FAILED = "Not supported (failed validation)"


def select(val: pd.DataFrame) -> dict:
    """Apply the selection rule to the validation table (ties: n_events desc, then train rank)."""
    q = val[val["mean_net_ret"] > 0].copy()
    if q.empty:
        return {"selected": None, "verdict": FAILED, "qualifying": []}
    q["_t"] = q["t_event"].fillna(-math.inf)
    q = q.sort_values(["_t", "n_events", "train_rank"], ascending=[False, False, True])
    best = q.iloc[0]
    return {
        "selected": {
            "id": best["config_id"],
            "band_lo": int(best["band_lo"]),
            "band_hi": int(best["band_hi"]),
            "category_filter": best["category_filter"],
            "liquidity_floor": float(best["liquidity_floor"]),
        },
        "verdict": None,
        "qualifying": q["config_id"].tolist(),
    }


def main() -> None:
    proto = C.load_protocol()
    train = pd.read_csv(C.RESULTS_DIR / "train_grid.csv")
    top = train.sort_values("rank").head(C.TOP_K)
    by_id = {c.id: c for c in C.grid(proto)}
    configs = [by_id[i] for i in top["config_id"]]
    con = C.connect()
    val = evaluate_grid(con, "val", configs)
    val["train_rank"] = top["rank"].to_numpy()
    cols = ["train_rank", *[c for c in GRID_COLUMNS if c != "rank"]]
    path = C.RESULTS_DIR / "val_top3.csv"
    val[cols].to_csv(path, index=False, float_format="%.10g")
    sel = select(val)
    sel["rule"] = "keep validation mean_net_ret > 0; choose highest validation t_event"
    sel["candidates"] = top["config_id"].tolist()
    C.write_json(C.RESULTS_DIR / "val_selection.json", sel)
    print(f"[val] -> {C.rel(path)}")
    print(f"[val] selection: {sel['selected']['id'] if sel['selected'] else FAILED}")


if __name__ == "__main__":
    main()
