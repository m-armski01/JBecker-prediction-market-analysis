"""Figures for REPORT.md (protocol section 5, commit 8+). Reads committed result files only.

Run: uv run python -m src.research.longshot_fade.report
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.research.longshot_fade import config as C  # noqa: E402

# Validated two-slot categorical palette (light surface) and chart chrome.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
S1 = "#2a78d6"  # train / highlighted
S2 = "#eb6834"  # validation
NEUTRAL = "#c3c2b7"
DPI = 150


def _style(ax, xgrid: bool = True, ygrid: bool = False) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _fig(w: float, h: float, ncols: int = 1):
    fig, axes = plt.subplots(1, ncols, figsize=(w, h), facecolor=SURFACE)
    return fig, axes


def _title(fig, title: str, subtitle: str) -> None:
    fig.text(0.01, 0.97, title, ha="left", va="top", fontsize=11, color=INK, fontweight="bold")
    fig.text(0.01, 0.91, subtitle, ha="left", va="top", fontsize=8, color=INK_2)


def _save(fig, name: str) -> str:
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = C.FIGURES_DIR / name
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    return C.rel(path)


def fig_train_grid(train: pd.DataFrame, top_ids: list[str]) -> str:
    df = train.sort_values("rank", ascending=False)
    fig, ax = _fig(8, 4.8)
    colors = [S1 if c in top_ids else NEUTRAL for c in df["config_id"]]
    y = np.arange(len(df))
    ax.barh(y, df["t_event"], color=colors, height=0.6)
    ax.axvline(0, color=AXIS, linewidth=0.8)
    ax.axvline(C.T_MIN, color=INK_2, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(C.T_MIN + 0.5, len(df) - 1, "pass threshold\nt = 2", color=INK_2, fontsize=7.5, va="center")
    ax.set_yticks(y, [f"#{int(r)}  {c}" for r, c in zip(df["rank"], df["config_id"])], fontsize=7.5)
    for yi, t in zip(y, df["t_event"]):
        ax.text(t - 0.4, yi, f"{t:.1f}", ha="right", va="center", fontsize=7, color=INK_2)
    ax.set_xlim(min(df["t_event"].min() * 1.15, -5), 9)
    ax.set_xlabel("Event-clustered t-statistic of mean net return (train)", color=INK_2, fontsize=8)
    _style(ax)
    _title(
        fig, "Train grid: all 12 configurations lose money", "Blue = top 3 by t sent to validation (protocol ranking)"
    )
    fig.subplots_adjust(left=0.30, right=0.97, top=0.84, bottom=0.12)
    return _save(fig, "train_grid_t.png")


def fig_val_top3(train: pd.DataFrame, val: pd.DataFrame) -> str:
    ids = val["config_id"].tolist()
    fig, ax = _fig(8, 3.4)
    y = np.arange(len(ids))[::-1]
    for off, df, col, lab in [(0.14, train, S1, "Train"), (-0.14, val, S2, "Validation")]:
        d = df.set_index("config_id").loc[ids]
        x = d["mean_net_ret"] * 100
        lo, hi = d["boot_ci_lo"] * 100, d["boot_ci_hi"] * 100
        ax.errorbar(x, y + off, xerr=[x - lo, hi - x], fmt="o", color=col, ms=6, elinewidth=2, capsize=0, label=lab)
    ax.axvline(0, color=INK_2, linewidth=0.9)
    ax.set_yticks(y, ids, fontsize=8)
    ax.set_xlabel("Mean net return per entry, % of capital (95% event-cluster bootstrap CI)", color=INK_2, fontsize=8)
    xmin = min(train["boot_ci_lo"].min(), val["boot_ci_lo"].min()) * 100
    ax.set_xlim(xmin * 1.15, 1.0)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK_2)
    _style(ax)
    _title(
        fig, "Validation: the train top 3 stay negative", "No configuration has a positive validation mean -> no freeze"
    )
    fig.subplots_adjust(left=0.27, right=0.97, top=0.80, bottom=0.17)
    return _save(fig, "val_top3.png")


def fig_estimates_by_split(train: pd.DataFrame, val: pd.DataFrame, ref_id: str) -> str:
    rows = [
        ("train", train.set_index("config_id").loc[ref_id], S1),
        ("val", val.set_index("config_id").loc[ref_id], S2),
    ]
    fig, ax = _fig(8, 3.0)
    labels = ["train", "val", "holdA (semi-blind)", "holdB (blind)"]
    y = np.arange(len(labels))[::-1]
    for (_name, r, col), yi in zip(rows, y[:2]):
        x, lo, hi = r["mean_net_ret"] * 100, r["boot_ci_lo"] * 100, r["boot_ci_hi"] * 100
        ax.errorbar([x], [yi], xerr=[[x - lo], [hi - x]], fmt="o", color=col, ms=6, elinewidth=2, capsize=0)
        ax.text(hi + 0.15, yi, f"{x:.2f}%  [{lo:.2f}, {hi:.2f}]", va="center", fontsize=7.5, color=INK_2)
    for yi in y[2:]:
        ax.text(
            -6.3, yi, "not run: holdouts stay locked after validation failed", va="center", fontsize=7.5, color=MUTED
        )
    ax.axvline(0, color=INK_2, linewidth=0.9)
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlim(-6.5, 1.0)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_xlabel("Mean net return per entry, % (95% event-cluster bootstrap CI)", color=INK_2, fontsize=8)
    _style(ax)
    _title(fig, f"Estimates by split: {ref_id}", "Reference configuration (train rank 1; not frozen)")
    fig.subplots_adjust(left=0.20, right=0.97, top=0.78, bottom=0.18)
    return _save(fig, "estimates_by_split.png")


def fig_cum_pnl() -> str:
    fig, axes = _fig(9, 3.2, ncols=2)
    for ax, split, col in zip(axes, ["train", "val"], [S1, S2]):
        path = C.RESULTS_DIR / f"pnl_curve_{split}.csv"
        d = pd.read_csv(path, parse_dates=["date"])
        ax.plot(d["date"], d["cum_realized_pnl_usd"] / 1000, color=col, linewidth=2)
        ax.axhline(0, color=AXIS, linewidth=0.8)
        ax.set_title(split, loc="left", fontsize=9, color=INK)
        ax.set_ylabel("Cumulative realized P&L ($k)", color=INK_2, fontsize=8)
        ax.tick_params(axis="x", labelrotation=30)
        last = d.iloc[-1]
        ax.text(
            last["date"],
            last["cum_realized_pnl_usd"] / 1000,
            f" {last['cum_realized_pnl_usd'] / 1000:.0f}k",
            fontsize=7.5,
            color=INK_2,
            va="center",
        )
        _style(ax, xgrid=False, ygrid=True)
    _title(
        fig,
        "Cumulative realized P&L, N = 100 contracts per entry",
        "Reference configuration b90-99_ex_finance_liq500; P&L booked at market close; splits not pooled",
    )
    fig.subplots_adjust(left=0.08, right=0.95, top=0.78, bottom=0.2, wspace=0.3)
    return _save(fig, "cum_pnl_by_split.png")


def fig_net_by_group(bd: pd.DataFrame) -> str:
    g = bd[bd["dimension"] == "group"]
    tr = g[g["split"] == "train"].set_index("value")
    va = g[g["split"] == "val"].set_index("value")
    groups = [x for x in tr.sort_values("mean_net_ret").index if tr.loc[x, "n_entries"] >= 30]
    fig, ax = _fig(8, 3.8)
    y = np.arange(len(groups))
    for off, d, col, lab in [(0.15, tr, S1, "Train"), (-0.15, va, S2, "Validation")]:
        xs = [d.loc[x, "mean_net_ret"] * 100 if x in d.index else math.nan for x in groups]
        ax.scatter(xs, y + off, color=col, s=36, label=lab, zorder=3)
    ax.axvline(0, color=INK_2, linewidth=0.9)
    ax.set_yticks(y, [f"{x}  (n={int(tr.loc[x, 'n_entries'])})" for x in groups], fontsize=8)
    ax.set_xlabel("Mean net return per entry, % (reference configuration)", color=INK_2, fontsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK_2)
    _style(ax)
    _title(
        fig,
        "Net return by category group",
        "Groups with at least 30 train entries; n = train entries (validation n can be small, see breakdowns.csv)",
    )
    fig.subplots_adjust(left=0.25, right=0.97, top=0.82, bottom=0.14)
    return _save(fig, "net_return_by_group.png")


def main() -> None:
    train = pd.read_csv(C.RESULTS_DIR / "train_grid.csv")
    val = pd.read_csv(C.RESULTS_DIR / "val_top3.csv")
    bd = pd.read_csv(C.RESULTS_DIR / "breakdowns.csv")
    top_ids = train.sort_values("rank").head(C.TOP_K)["config_id"].tolist()
    ref_id = C.load_json(C.RESULTS_DIR / "backtest_portfolio.json")["config"]["id"]
    out = [
        fig_train_grid(train, top_ids),
        fig_val_top3(train, val),
        fig_estimates_by_split(train, val, ref_id),
        fig_cum_pnl(),
        fig_net_by_group(bd),
    ]
    for p in out:
        print(f"[report] {p}")


if __name__ == "__main__":
    main()
