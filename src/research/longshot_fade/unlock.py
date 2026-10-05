"""Guarded holdout unlock (protocol section 5, commits 6 and 7). Each holdout runs exactly once.

Before anything else, `unlock --holdout A|B` refuses to run unless:
  1. HEAD is exactly the commit tagged `prereg-frozen` (A) or `holdout-a-run` (B);
  2. `git status --porcelain` is empty;
  3. the holdout panel files (exact set) match the SHA-256s in the panel manifest;
  4. HOLDOUT_LOG.md has no earlier entry for this holdout;
  5. (B only) at least 7 days have passed since the holdB pull completed.

This module is the only place where holdout outcomes are read.

Run: uv run python -m src.research.longshot_fade.unlock --holdout A
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.research.longshot_fade import config as C
from src.research.longshot_fade import strategy as ST
from src.research.longshot_fade.panel import panel_source

FINAL_STATUSES = frozenset({"finalized", "settled"})
VOID_RESULTS = frozenset({"", "void"})


class LockError(RuntimeError):
    """Raised when a holdout lock precondition fails; nothing outcome-related has been read."""


# --------------------------------------------------------------------------------------------
# Lock checks
# --------------------------------------------------------------------------------------------


def holdout_logged(log_path: Path, holdout: str) -> bool:
    if not Path(log_path).exists():
        return False
    return re.search(rf"^## Holdout {re.escape(holdout)}\b", Path(log_path).read_text(), flags=re.M) is not None


def verify_files(expected: dict[str, str], panel_dir: Path, split: str, root: Path) -> int:
    """Exact-set SHA-256 check of a holdout panel; returns the number of files verified."""
    actual = sorted(Path(panel_dir).glob(f"split={split}/*/*.parquet"))
    actual_rel = {C.rel(p, root): p for p in actual}
    if not expected:
        raise LockError(f"manifest lists no {split} panel files")
    missing = sorted(set(expected) - set(actual_rel))
    extra = sorted(set(actual_rel) - set(expected))
    if missing or extra:
        raise LockError(f"{split} panel file set differs from manifest (missing={missing}, extra={extra})")
    for relpath, sha in expected.items():
        got = C.sha256_file(actual_rel[relpath])
        if got != sha:
            raise LockError(f"SHA-256 mismatch for {relpath}: manifest {sha[:12]}..., file {got[:12]}...")
    return len(expected)


def check_lock(
    holdout: str,
    *,
    root: Path,
    expected_tag: str,
    expected_files: dict[str, str],
    panel_dir: Path,
    split: str,
    log_path: Path,
    pull_completed_at: datetime | None = None,
    wait_days: int = C.HOLDB_WAIT_DAYS,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Run every precondition; raise LockError on the first failure."""
    try:
        head = C.git("rev-parse", "HEAD", cwd=root)
        tag_commit = C.git("rev-parse", f"{expected_tag}^{{commit}}", cwd=root)
    except Exception as exc:
        raise LockError(f"cannot resolve HEAD or tag {expected_tag}: {exc}") from exc
    if head != tag_commit:
        raise LockError(f"HEAD {head[:10]} is not the commit tagged {expected_tag} ({tag_commit[:10]})")
    if not C.git_is_clean(cwd=root):
        raise LockError("working tree is not clean (git status --porcelain is non-empty)")
    n_files = verify_files(expected_files, panel_dir, split, root)
    if holdout_logged(log_path, holdout):
        raise LockError(f"HOLDOUT_LOG.md already has an entry for holdout {holdout}; it may only run once")
    if holdout == "B":
        if pull_completed_at is None:
            raise LockError("holdB manifest has no pull_completed_at")
        now = now or C.utc_now()
        if now < pull_completed_at + timedelta(days=wait_days):
            raise LockError(
                f"holdB outcome fetch not allowed before {pull_completed_at + timedelta(days=wait_days)} "
                f"({wait_days} days after the pull completed)"
            )
    return {"head": head, "tag": expected_tag, "files_verified": n_files}


def append_log(log_path: Path, holdout: str, head: str, manifest_sha: str, when: datetime, extra: str = "") -> None:
    entry = (
        f"\n## Holdout {holdout}\n\n"
        f"- UTC: {when.isoformat(timespec='seconds')}\n"
        f"- HEAD: {head}\n"
        f"- Manifest SHA-256: {manifest_sha}\n"
        f"{extra}"
        f"- unlocked once\n"
    )
    with open(log_path, "a") as f:
        f.write(entry)


# --------------------------------------------------------------------------------------------
# Outcomes
# --------------------------------------------------------------------------------------------


def outcome_from_market(status: str | None, result: str | None) -> tuple[str | None, bool, bool]:
    """(result, settled, void) under the protocol definitions; non-final markets are unresolved."""
    status = (status or "").lower()
    result = (result or "").lower()
    if result in ("yes", "no"):
        return result, True, False
    if status in FINAL_STATUSES and result in VOID_RESULTS:
        return None, False, True
    return None, False, False


def fetch_market_api(client, ticker: str) -> dict[str, Any]:
    """Live endpoint first, then the historical endpoint. Raises on network failure."""
    import httpx

    last_404 = None
    for path in (f"/markets/{ticker}", f"/historical/markets/{ticker}"):
        try:
            data = client.http.get(path)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                last_404 = path
                continue
            raise
        m = data.get("market", data)
        return {
            "ticker": ticker,
            "endpoint": path,
            "status": m.get("status"),
            "result": m.get("result"),
            "close_time": m.get("close_time"),
            "settlement_ts": m.get("settlement_ts"),
            "fetched_at_utc": C.utc_now().isoformat(),
        }
    return {"ticker": ticker, "endpoint": last_404, "status": None, "result": None, "error": "not_found"}


def backfill(
    tickers: Iterable[str],
    cache_path: Path,
    fetch: Callable[[str], dict[str, Any]],
    log=print,
) -> pd.DataFrame:
    """Fetch market outcomes for tickers, caching one JSON line per ticker (resumable)."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["ticker"]] = rec
    todo = [t for t in tickers if t not in done]
    log(f"[backfill] {len(done)} cached, {len(todo)} to fetch")
    with open(cache_path, "a") as f:
        for i, t in enumerate(todo, 1):
            rec = fetch(t)
            done[t] = rec
            f.write(json.dumps(rec) + "\n")
            if i % 500 == 0:
                f.flush()
                log(f"[backfill] {i}/{len(todo)}")
    return pd.DataFrame([done[t] for t in tickers if t in done])


def api_reachable(client) -> bool:
    try:
        client.http.get("/historical/cutoff")
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------------------------
# Holdout runs
# --------------------------------------------------------------------------------------------


def _frozen() -> tuple[dict[str, Any], C.StrategyConfig]:
    frozen = C.load_json(C.FROZEN_JSON)
    return frozen, C.StrategyConfig.from_dict(frozen["config"])


def _metrics_block(pnl: pd.DataFrame, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        **summary,
        "pass_t_event_ge_2": bool(summary["t_event"] >= C.T_MIN) if summary["n_entries"] else False,
        "pass_boot_ci_lo_gt_0": bool(summary["boot_ci_lo"] > C.CI_LO_MIN) if summary["n_entries"] else False,
        "passes": ST.passes(summary),
    }


def run_holdout_a(allow_unreachable: bool = False) -> dict[str, Any]:
    from src.indexers.kalshi.client import KalshiClient

    manifest = C.load_json(C.PANEL_MANIFEST_JSON)
    lock = check_lock(
        "A",
        root=C.ROOT,
        expected_tag=C.TAG_FROZEN,
        expected_files=manifest["holdA_files_sha256"],
        panel_dir=C.PANEL_DIR,
        split="holdA",
        log_path=C.HOLDOUT_LOG_MD,
    )
    frozen, cfg = _frozen()
    n = int(frozen["n_contracts"])
    con = C.connect()
    entries = ST.select_entries(con, panel_source(C.PANEL_DIR, "holdA"), cfg, "holdA")
    tickers = entries["ticker"].tolist()

    # Local outcomes (allowed here only).
    con.register("entry_tickers", pd.DataFrame({"ticker": tickers}))
    local = con.execute(
        f"""
        SELECT m.ticker, m.status, m.result, m.close_time
        FROM read_parquet('{C.MARKETS_DIR}/*.parquet') m JOIN entry_tickers USING (ticker)
        """
    ).df()
    local_final = local[local["status"] == "finalized"]
    need = sorted(set(tickers) - set(local_final["ticker"]))
    print(f"[holdA] {len(tickers)} entries; {len(local_final)} finalized locally; {len(need)} need backfill")

    client = KalshiClient()
    try:
        reachable = api_reachable(client) if need else True
        if need and not reachable and not allow_unreachable:
            raise LockError(
                "Kalshi API unreachable: aborting before any outcome statistic is computed. Re-run when the API "
                "is reachable, or log a DEVIATIONS entry and pass --allow-unreachable to censor those markets."
            )
        bf = (
            backfill(need, C.BACKFILL_DIR / "holdA_markets.jsonl", lambda t: fetch_market_api(client, t))
            if need and reachable
            else pd.DataFrame(columns=["ticker", "status", "result", "close_time", "endpoint", "error"])
        )
    finally:
        client.close()

    rows = []
    for t, s, r, ct in local_final[["ticker", "status", "result", "close_time"]].itertuples(index=False):
        res, settled, void = outcome_from_market(s, r)
        rows.append({"ticker": t, "result": res, "void": void, "close_time": ct, "source": "local"})
    for rec in bf.to_dict("records"):
        res, settled, void = outcome_from_market(rec.get("status"), rec.get("result"))
        ct = pd.to_datetime(rec.get("close_time"), utc=True) if rec.get("close_time") else pd.NaT
        rows.append({"ticker": rec["ticker"], "result": res, "void": void, "close_time": ct, "source": "backfill"})
    outcomes = pd.DataFrame(rows, columns=["ticker", "result", "void", "close_time", "source"])
    outcomes["close_time"] = pd.to_datetime(outcomes["close_time"], utc=True)
    C.HOLDA_OUTCOMES_DIR.mkdir(parents=True, exist_ok=True)
    outcomes.to_parquet(C.HOLDA_OUTCOMES_DIR / "market_outcomes.parquet", index=False)

    pnl = ST.compute_pnl(ST.attach_outcomes(entries, outcomes), n_contracts=n)
    summary = ST.summarize(pnl)
    bf_resolved = 0
    for s, r in zip(bf.get("status", []), bf.get("result", [])):
        _, settled, void = outcome_from_market(s, r)
        bf_resolved += int(settled or void)
    now = C.utc_now()
    manifest_sha = C.sha256_file(C.PANEL_MANIFEST_JSON)
    out = {
        "holdout": "A",
        "split": "holdA",
        "label": "confirmatory run (semi-blind)",
        "run_at_utc": now.isoformat(timespec="seconds"),
        "head": lock["head"],
        "tag_checked": lock["tag"],
        "panel_files_verified": lock["files_verified"],
        "panel_manifest_sha256": manifest_sha,
        "frozen_config": cfg.to_dict(),
        "n_contracts": n,
        "metrics": _metrics_block(pnl, summary),
        "censoring": {
            "entries_selected": int(len(entries)),
            "resolved_locally": int(len(local_final)),
            "backfill_needed": len(need),
            "api_reachable": bool(reachable),
            "backfill_fetched": int(len(bf)),
            "backfill_resolved": bf_resolved,
            "backfill_still_unresolved": int(len(need) - bf_resolved),
            "censored_entries": int(summary["n_censored"]),
        },
    }
    C.write_json(C.RESULTS_DIR / "holdout_a.json", out)
    append_log(C.HOLDOUT_LOG_MD, "A", lock["head"], manifest_sha, now)
    return out


def run_holdout_b() -> dict[str, Any]:
    from src.indexers.kalshi.client import KalshiClient

    manifest_path = C.HOLDB_DIR / "panel_manifest.json"
    if not manifest_path.exists():
        raise LockError(f"{manifest_path} not found: run holdout_b.py first")
    manifest = C.load_json(manifest_path)
    lock = check_lock(
        "B",
        root=C.ROOT,
        expected_tag=C.TAG_HOLDOUT_A,
        expected_files=manifest["holdB_files_sha256"],
        panel_dir=C.HOLDB_DIR / "panel",
        split="holdB",
        log_path=C.HOLDOUT_LOG_MD,
        pull_completed_at=datetime.fromisoformat(manifest["pull_completed_at_utc"]),
    )
    frozen, cfg = _frozen()
    n = int(frozen["n_contracts"])
    con = C.connect()
    entries = ST.select_entries(con, panel_source(C.HOLDB_DIR / "panel", "holdB"), cfg, "holdB")
    tickers = entries["ticker"].tolist()
    client = KalshiClient()
    try:
        if not api_reachable(client):
            raise LockError("Kalshi API unreachable: holdB outcomes cannot be fetched (status would be 'unavailable')")
        bf = backfill(tickers, C.HOLDB_DIR / "outcomes" / "holdB_markets.jsonl", lambda t: fetch_market_api(client, t))
    finally:
        client.close()
    rows = []
    for rec in bf.to_dict("records"):
        res, settled, void = outcome_from_market(rec.get("status"), rec.get("result"))
        ct = pd.to_datetime(rec.get("close_time"), utc=True) if rec.get("close_time") else pd.NaT
        rows.append({"ticker": rec["ticker"], "result": res, "void": void, "close_time": ct, "source": "api"})
    outcomes = pd.DataFrame(rows, columns=["ticker", "result", "void", "close_time", "source"])
    outcomes["close_time"] = pd.to_datetime(outcomes["close_time"], utc=True)
    outcomes.to_parquet(C.HOLDB_DIR / "outcomes" / "market_outcomes.parquet", index=False)

    pnl = ST.compute_pnl(ST.attach_outcomes(entries, outcomes), n_contracts=n)
    summary = ST.summarize(pnl)
    metrics = _metrics_block(pnl, summary)
    underpowered = summary["n_events"] < C.HOLDB_MIN_EVENTS
    status = "underpowered" if underpowered else ("pass" if metrics["passes"] else "fail")
    now = C.utc_now()
    manifest_sha = C.sha256_file(manifest_path)
    shutil.copyfile(manifest_path, C.RESULTS_DIR / "holdb_panel_manifest.json")
    out = {
        "holdout": "B",
        "split": "holdB",
        "label": "confirmatory run (blind)",
        "run_at_utc": now.isoformat(timespec="seconds"),
        "head": lock["head"],
        "tag_checked": lock["tag"],
        "panel_files_verified": lock["files_verified"],
        "panel_manifest_sha256": manifest_sha,
        "window": manifest.get("window"),
        "frozen_config": cfg.to_dict(),
        "n_contracts": n,
        "metrics": metrics,
        "power_gate_min_events": C.HOLDB_MIN_EVENTS,
        "status": status,
        "censoring": {
            "entries_selected": int(len(entries)),
            "outcomes_fetched": int(len(bf)),
            "censored_entries": int(summary["n_censored"]),
        },
    }
    C.write_json(C.RESULTS_DIR / "holdout_b.json", out)
    append_log(C.HOLDOUT_LOG_MD, "B", lock["head"], manifest_sha, now)
    return out


def mark_b_unavailable(reason: str) -> dict[str, Any]:
    """Record holdB as unavailable (API unreachable or access not permitted). Reads no outcomes."""
    out = {
        "holdout": "B",
        "split": "holdB",
        "status": "unavailable",
        "reason": reason,
        "recorded_at_utc": C.utc_now().isoformat(timespec="seconds"),
        "head": C.git_head(),
    }
    C.write_json(C.RESULTS_DIR / "holdout_b.json", out)
    return out


def verdict(holdout_a_pass: bool | None, holdout_b_status: str | None, validation_failed: bool = False) -> str:
    """Protocol section 6 verdict table. holdout_b_status in {pass, fail, underpowered, unavailable}."""
    if validation_failed:
        return "Not supported (failed validation)"
    if not holdout_a_pass:
        return "Not supported"
    if holdout_b_status == "pass":
        return "Supported (blind replication)"
    if holdout_b_status in ("unavailable", "underpowered", None):
        return "Supported (semi-blind only)"
    return "Not replicated out of sample"


def main() -> None:
    parser = argparse.ArgumentParser(description="Guarded one-time holdout unlock.")
    parser.add_argument("--holdout", choices=["A", "B"])
    parser.add_argument(
        "--allow-unreachable",
        action="store_true",
        help="holdout A only: if the API is unreachable, censor markets without a local outcome "
        "(requires a DEVIATIONS.md entry first)",
    )
    parser.add_argument("--mark-b-unavailable", metavar="REASON", help="record holdB as unavailable and exit")
    args = parser.parse_args()
    if args.mark_b_unavailable:
        print(json.dumps(mark_b_unavailable(args.mark_b_unavailable), indent=2))
        return
    if args.holdout is None:
        parser.error("--holdout is required")
    try:
        out = run_holdout_a(args.allow_unreachable) if args.holdout == "A" else run_holdout_b()
    except LockError as exc:
        raise SystemExit(f"[unlock] refused: {exc}") from None
    m = out["metrics"]
    print(
        f"[unlock] holdout {args.holdout}: n={m['n_entries']} events={m['n_events']} mean_net_ret={m['mean_net_ret']:.5f} "
        f"t_event={m['t_event']:.3f} CI=({m['boot_ci_lo']:.5f}, {m['boot_ci_hi']:.5f}) pass={m['passes']}"
    )


if __name__ == "__main__":
    main()
