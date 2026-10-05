"""Commit 5: freeze the selected configuration for the confirmatory holdout tests.

Preconditions: on branch prereg/longshot-fade, clean working tree, HEAD is the validation commit
("val: top-3 validation and selection"), a configuration was selected, frozen.json does not exist yet.

Writes config/frozen.json and results/deflated_sharpe.json (train data only) and appends the
Frozen addendum to PROTOCOL.md. Git commit/tag/push are done by hand afterwards.

Run: uv run python -m src.research.longshot_fade.freeze
"""

from __future__ import annotations

import pandas as pd

from src.research.longshot_fade import config as C
from src.research.longshot_fade import stats as S
from src.research.longshot_fade import strategy as ST
from src.research.longshot_fade.panel import panel_source

VAL_COMMIT_SUBJECT = "val: top-3 validation and selection"


def _plain(row: pd.Series) -> dict:
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in row.to_dict().items()}


def preconditions() -> str:
    branch = C.git("rev-parse", "--abbrev-ref", "HEAD")
    if branch != C.BRANCH:
        raise SystemExit(f"[freeze] on branch {branch!r}, expected {C.BRANCH!r}")
    if not C.git_is_clean():
        raise SystemExit("[freeze] working tree is not clean")
    subject = C.git("log", "-1", "--format=%s")
    if subject != VAL_COMMIT_SUBJECT:
        raise SystemExit(f"[freeze] HEAD subject is {subject!r}, expected the validation commit")
    if C.FROZEN_JSON.exists():
        raise SystemExit("[freeze] frozen.json already exists; the freeze happens once")
    return C.git_head()


def main() -> None:
    commit4 = preconditions()
    proto = C.load_protocol()
    sel = C.load_json(C.RESULTS_DIR / "val_selection.json")
    if not sel.get("selected"):
        raise SystemExit(
            "[freeze] validation selected no configuration: verdict is 'Not supported (failed validation)'"
        )
    cfg = C.StrategyConfig.from_dict(sel["selected"])
    train = pd.read_csv(C.RESULTS_DIR / "train_grid.csv")
    val = pd.read_csv(C.RESULTS_DIR / "val_top3.csv")

    con = C.connect()
    pnl, _ = ST.run_config(con, panel_source(C.PANEL_DIR, "train"), cfg, "train")
    r = pnl.loc[pnl["outcome"] != "censored", "net_ret"].to_numpy()
    dsr = S.deflated_sharpe(r, train["sharpe_per_entry"].tolist(), n_trials=C.BONFERRONI_M)
    dsr.update({"config_id": cfg.id, "split": "train", "status": "descriptive only"})
    C.write_json(C.RESULTS_DIR / "deflated_sharpe.json", dsr)

    frozen = {
        "frozen_at_utc": C.utc_now().isoformat(timespec="seconds"),
        "protocol_version": proto["protocol_version"],
        "config": cfg.to_dict(),
        "n_contracts": C.N_BASE,
        "sensitivity_contracts": list(C.SENSITIVITY_SIZES),
        "fee_model": proto["fees"],
        "seed": C.SEED,
        "bootstrap_iterations": C.BOOT_ITERATIONS,
        "decision_thresholds": proto["decision"]["pass_requires_all"],
        "holdB": {
            "outcome_fetch_min_days_after_pull": C.HOLDB_WAIT_DAYS,
            "power_gate_min_event_clusters": C.HOLDB_MIN_EVENTS,
        },
        "commit4_hash": commit4,
        "panel_manifest_sha256": C.sha256_file(C.PANEL_MANIFEST_JSON),
        "protocol_json_sha256": C.sha256_file(C.PROTOCOL_JSON),
        "train_metrics": _plain(train.loc[train["config_id"] == cfg.id].iloc[0]),
        "val_metrics": _plain(val.loc[val["config_id"] == cfg.id].iloc[0]),
    }
    C.write_json(C.FROZEN_JSON, frozen)

    tm, vm = frozen["train_metrics"], frozen["val_metrics"]
    addendum = f"""

---

## Frozen addendum

Appended by `freeze.py` at {frozen["frozen_at_utc"]}. The validation commit is `{commit4}`.

- **Frozen configuration:** `{cfg.id}`: NO-cost band [{cfg.band_lo}, {cfg.band_hi}]¢, category filter `{cfg.category_filter}`,
  liquidity floor `prior_notional_usd ≥ {cfg.liquidity_floor:g}`. N = {C.N_BASE} contracts.
- **Fee model:** 0.07 × multiplier × C × P(1 − P), rounded up per order; S&P 500 / Nasdaq-100 multiplier 0.5.
- **Seed** {C.SEED}; **bootstrap** {C.BOOT_ITERATIONS} event resamples.
- **Pass rule:** event-clustered one-sided t ≥ {C.T_MIN}, and the 95% bootstrap CI lower bound > {C.CI_LO_MIN}.
- **Train:** n = {int(tm["n_entries"])}, events = {int(tm["n_events"])}, mean net_ret = {tm["mean_net_ret"]:.5f},
  t_event = {tm["t_event"]:.3f}, rank = {int(tm["rank"])} of 12.
- **Validation:** n = {int(vm["n_entries"])}, events = {int(vm["n_events"])}, mean net_ret = {vm["mean_net_ret"]:.5f},
  t_event = {vm["t_event"]:.3f}.
- **Deflated Sharpe ratio** (train, 12 trials, descriptive): {dsr["deflated_sharpe_ratio"]:.4f}
  (`results/deflated_sharpe.json`).
- `config/frozen.json` holds the complete frozen record. Nothing in it changes after the `prereg-frozen` tag.
"""
    with open(C.PROTOCOL_MD, "a") as f:
        f.write(addendum)
    print(f"[freeze] frozen {cfg.id}; wrote {C.rel(C.FROZEN_JSON)} and the PROTOCOL.md addendum")


if __name__ == "__main__":
    main()
