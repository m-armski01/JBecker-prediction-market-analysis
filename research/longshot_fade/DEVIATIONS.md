# Deviations from the spec

Every departure from the study spec is listed here, with what changed, why, and the commit that acts on it. Each entry is written before the change is made.

**Entries D1–D12 were all decided before registration.** They are acted on in the commit tagged `protocol-v1`, unless the entry says otherwise. That commit cannot quote its own hash, so its entries cite the tag; later entries cite commit hashes.

## Pre-registration entries

**D1. `.gitignore` exception for the study folder.**
- *What:* added `!research/longshot_fade/` to `.gitignore`.
- *Why:* the repo ignores `research/*`, which would silently leave every study file untracked. The `data/` rule already covers `data/research/`, so nothing else in the ignore rules changed.

**D2. Price normalization for sub-cent and truncated prints.**
- *What:* when `yes_price + no_price = 99`, the YES cost is set to `100 − no_price` and the NO cost to `100 − yes_price`. A position whose cost falls outside 1–99¢ is excluded from return calculations.
- *Spec (§4.3):* `cost_cents` equals the stored price and is assumed to be 1–99.
- *Why:* 267,291 holdA trades come from sub-penny ticks or an indexer float bug, so both stored sides are floored. This rule never understates what a buyer paid (spec §0.4: prefer the conservative option).
- *Effect:* no train or val row is affected. Every sum-99 row is in 2025Q4, i.e. holdA.

**D3. `DEVIATIONS.md` is not empty at commit 1.**
- *Spec:* commit 1 contains an empty `DEVIATIONS.md`.
- *Why:* spec §0.4 requires every uncovered choice to be logged before acting on it, and these choices are acted on in commit 1 (or are registered in it).

**D4. Extra files in commit 1.**
- *What:* commit 1 also contains `src/research/__init__.py`, `src/research/longshot_fade/__init__.py`, `src/research/longshot_fade/audit_close_time.py` and `results/close_time_audit.json`.
- *Why:* the audit numbers quoted in `PROTOCOL.md` Appendix B should trace to committed code and output.
- `audit_close_time.py` is not in the spec's §3 file layout.

**D5. Fee series multiplier table.**
- *What:* S&P 500 and Nasdaq-100 series (`^(KX)?(INX|NASDAQ100)`) use multiplier 0.5, i.e. coefficient 0.035. All other series use 0.07.
- *Verification status:* in `PROTOCOL.md` Appendix A. The absence of any other exception during 2024-10 → 2025-11 is an assumption, because the archived schedules could not be retrieved.

**D6. HoldB operational rules.**
- *Spec:* silent on all of these; they are registered in `PROTOCOL.md` §3 and §5.
- *Rules:*
  - trades come from the live endpoint only, from the cutoff in `GET /historical/cutoff` at pull time to the start of the pull;
  - the pull runs at HEAD = `holdout-a-run`, which is after `prereg-frozen`;
  - market metadata is stored without outcome fields;
  - block trades (`is_block_trade`) are dropped;
  - reverse purge: holdB events that also traded in train, val or holdA are dropped;
  - `prior_*` features use only the trades the live endpoint returns;
  - the holdB manifest stays in `data/research/longshot_fade/holdB/` until unlock, then is copied into `results/` in commit 7;
  - `unlock.py --holdout B` also requires at least 7 days since the pull completed (the owner's decision on 2026-10-05).

**D7. HoldA backfill scope and endpoints.**
- *What:* outcomes are backfilled only for frozen-configuration entry markets whose local status is not `finalized`, since only their outcomes enter the test. Each is tried on the live `/markets/{ticker}` first, then on `/historical/markets/{ticker}`.
- *Why:* Kalshi moved data settled before the cutoff to the historical endpoints.

**D8. Panel conventions.**
- Purged rows are kept and flagged `purged = true`; every statistic excludes them.
- The panel adds a `close_time` column, needed for `hold_days`.
- `won` is null unless the market is settled.
- `cost_cents` may be 100; see D2.
- `prior_*` features use each market's full local history, including trades before 2024-10-01.

**D9. Statistical conventions the spec leaves open.**
- One-sided p-values use Student t with G − 1 degrees of freedom.
- The bootstrap uses `numpy.random.default_rng(20251001)` and percentile CIs.
- Definitions of `mean_net_ret_per_capital_day` (Σ pnl / Σ cost × hold_days), `win_rate` (share with net_ret > 0) and `worst_event_loss_usd` (minimum event P&L in USD).
- Censoring: entries that are neither settled nor void are excluded and counted.

**D10. Local, untracked environment changes.**
- `Claude outputs/` was added to `.git/info/exclude`. This is local and untracked; no tracked file changed. Without it, the untracked folder would make `git status --porcelain` non-empty and trip the freeze and unlock guards.
- The GitHub CLI was installed with Homebrew to open the PR.

**D11. Extra test file.**
- *What:* `tests/research/test_holdout_b.py`, which holds the holdB normalizer test on recorded payloads.
- *Why:* that test has no natural home among the five test files in spec §3.
- The recorded payloads come from `/historical/trades` for a **train-period** market, so no holdout data is exposed.
- *Acted on in:* commit 2.

**D12. Network access to Kalshi.**
- *Observed 2026-10-05:* `api.elections.kalshi.com` answered at first. Later, `kalshi.com`, `api.elections.kalshi.com` and `polymarket.com` were reset during the TLS handshake while other hosts worked.
- The study does not try to get around network restrictions.
- *If the API is unreachable or access is not permitted when a holdout needs it:*
  - holdA markets without a local outcome are censored (excluded and counted);
  - holdB is "unavailable".
- If this happens, a dated entry will be added here.

## Post-registration entries

<!-- append new entries below; never edit the entries above -->

**D13. Kalshi API outcome mapping for holdouts.**
- *What:* in `unlock.py`, a market counts as final if its status is `finalized` or `settled`. The newer API may report `settled`.
- Result `yes`/`no` means settled.
- A final market whose result is empty or `void` counts as void.
- Any other market is unresolved, i.e. censored.
- *Decided:* before the freeze. *Acted on in:* commit 2 (`panel: positions table, fee model, stats, tests`).

**D14. Where the scripts are committed.**
- `unlock.py` and `holdout_b.py` go in commit 2, so the holdouts run on code that existed before the freeze.
- `run_train.py` goes in commit 3.
- `run_validate.py` and `freeze.py` go in commit 4. `freeze.py` checks for a clean tree, so it has to be committed before it runs.
- *Spec:* lists the commit 2 code but does not say where these scripts are committed.

**D15. Behaviour of the holdout runs.**
- `unlock.py --holdout A` aborts if the Kalshi API cannot be reached. It aborts before computing any outcome statistic and before writing the log, so the lock stays intact.
  - `--allow-unreachable` overrides this: markets without a local outcome are then censored. It may only be used after a dated entry in this file.
- The backfill caches one JSON line per ticker, so an interrupted backfill can resume without fetching anything twice.
- For holdA entries, a backfilled `close_time` (the final close) replaces the snapshot's `close_time` for `hold_days`.
- `unlock.py --mark-b-unavailable "<reason>"` records holdB as unavailable without reading any outcome.

**D16. Dropped holdB trades.**
- *What:* trades with no market metadata, outside the window, with a price sum other than 99 or 100, or with an invalid taker side are dropped during normalization. Block trades are dropped too (D6). Each count is reported in the holdB manifest.

**D17. Definition of the sanity check.**
- *What:* `gross_ret` per position is (100 − c)/c if the position won, −1 if it lost and 0 if the market was void.
- *Population:* non-purged, settled-or-void train positions with cost 1–99¢.
- *Statistic:* the sign test is the mean maker `gross_ret` minus the mean taker `gross_ret`, per position.
