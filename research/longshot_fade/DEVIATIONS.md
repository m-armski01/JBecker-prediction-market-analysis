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
