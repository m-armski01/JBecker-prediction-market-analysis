# Pre-registration protocol: longshot-bias fading test (v1)

- **Registered:** 2026-10-05, on branch `prereg/longshot-fade` of the fork `m-armski01/JBecker-prediction-market-analysis`. This file is the content of the commit tagged `protocol-v1`. The GitHub timestamps of that commit and tag are the third-party record of the registration.
- **Machine-readable copy:** `config/protocol_v1.json`. If the two disagree, this document governs and the disagreement is logged in `DEVIATIONS.md`.
- **What was looked at before registration:**
  - non-outcome data checks only: row counts, duplicate keys, the sets of `status`, `result` and `taker_side` values, price ranges, and trade counts per quarter (Appendix C);
  - the train-only `close_time` audit (Appendix B).
  - No return, win rate or other outcome-based statistic was computed for any configuration or split.

---

## 1. Background and question

**Becker (2026), *The Microstructure of Wealth Transfer in Prediction Markets*,** documents three patterns on Kalshi:
- from 2024 Q4 onward, liquidity takers lose money to liquidity makers;
- at the same cost basis, YES positions do worse than NO positions;
- the gap is widest at longshot prices, which he calls an "optimism tax".

Before 2024 Q4 the maker–taker gap had the opposite sign.

**Adamczewski (2026), *Integration Without Leadership*,** supplies the method we copy:
- a hypothesis, estimator and decision rule fixed in timestamped commits before any data are examined;
- separate train, holdout and confirmatory phases;
- inference clustered at the natural dependence unit;
- an explicit comparison of the effect against trading fees.

**Question.** A trader who can only *take* liquidity systematically buys NO against overpriced longshot YES contracts and holds to resolution. Does that trader make money after Kalshi's taker fees, on data not used to choose the strategy?

---

## 2. Data

- **Kalshi trades:** `data/kalshi/trades/*.parquet`. 72,134,741 trades with unique `trade_id`, from 2021-06-30 20:09 UTC to 2025-11-25 22:00 UTC.
- **Kalshi markets:** `data/kalshi/markets/*.parquet`. 7,682,445 markets with unique `ticker`, all `binary`. This is a single snapshot fetched 2025-11-23 18:51 → 2025-11-24 02:40 UTC.
- Schemas are in `docs/SCHEMAS.md`. Nothing is modified: all derived data goes to `data/research/longshot_fade/`, which is gitignored.
- **Known data issues, fixed before any outcome was examined:**
  1. **Sub-cent and truncated prices.** 267,291 trades, all in 2025Q4 and all fetched 2025-11-25, have `yes_price + no_price = 99`. The newer API reports prices as dollar strings. Some trades print at sub-penny ticks near the extremes; the most frequent pairs are (0,99), (1,98) and (2,97). The indexer also has a float-truncation bug, `int(float(x)*100)`, that only hits the strings 0.29, 0.57 and 0.58. In both cases each stored side is the floor of the true price, so the true price of a side is in (stored, 100 − other side's stored price]. **Normalization rule:** when `yes_price + no_price = 99`, set the YES cost to `100 − no_price` and the NO cost to `100 − yes_price`. Otherwise use the stored prices. This rule never understates the price a buyer paid, so it is the conservative choice. Normalized prices feed `cost_cents`, `prior_notional_usd` and `prev_yes_price`. A position whose normalized cost is outside 1–99¢ enters no return calculation. For a NO buyer that only happens at 100¢, which is outside every band.
  2. **Snapshot timing.** Markets that were still open at the snapshot have no result locally. For holdA this is handled by the guarded backfill (§6, commit 6).
  3. **Trade coverage.** The trades indexer skips tickers it has already fetched. A market that was fetched while still trading may therefore be missing its later trades. Entry rules use the *first* qualifying print, so this mainly affects markets with no qualifying print before their fetch. It is listed as a caveat.

---

## 3. Definitions

These are fixed by this commit and cannot change afterwards.

### 3.1 Splits (by trade timestamp, UTC)

| Split | Window |
|---|---|
| `train` | 2024-10-01 00:00 ≤ ts < 2025-07-01 00:00 |
| `val` | 2025-07-01 00:00 ≤ ts < 2025-10-01 00:00 |
| `holdA` (semi-blind) | 2025-10-01 00:00 ≤ ts, through the last local trade (2025-11-25 22:00) |
| `holdB` (blind) | trades pulled from the Kalshi API after the `prereg-frozen` tag (run at HEAD = `holdout-a-run`). The window starts at the live-endpoint cutoff returned by `GET /historical/cutoff` at pull time and ends at the moment the pull starts. All code used is frozen. |

- Trades before 2024-10-01 belong to no split.
- **Why holdA is only semi-blind:** Becker's published results use data through 2025-11-25. The hypothesis was therefore shaped by findings that include this window. holdB is the only fully blind test.
- **Why holdB uses only the live endpoint:** Kalshi moved older data behind `/historical/*` endpoints; on 2026-10-05 the live trades cutoff was 2026-08-06. The spec targets "the earliest date the API returns (≈ last 100 days)", so only the live endpoint is used for holdB trades.

### 3.2 Event family and purging

- The event family is `event_ticker`. It is the cluster unit for all inference.
- **Purge.** For each `event_ticker`, find every split that contains any of its trades. The event stays only in the latest of those splits. Its rows in earlier splits are flagged `purged = true` and excluded from every statistic. Only trade presence is used, never outcomes.
- Dropped events and rows per split are logged to `results/purge_summary.json`.
- **holdB reverse purge.** The holdA panel is hash-frozen at commit 2, so events cannot be moved out of it later. Instead, any holdB event with a trade in train, val or holdA is dropped from holdB.

### 3.3 Positions table

There are two rows per trade: one for the taker and one for the maker.

| Column | Definition |
|---|---|
| `trade_id`, `ticker`, `event_ticker` | from the source tables |
| `series` | the part of `event_ticker` before the first `-` |
| `group`, `category`, `subcategory` | from `src/analysis/kalshi/util/categories.py` (`CATEGORY_SQL` → `get_hierarchy`). Unmapped categories become `"Other"`; their share is logged. |
| `ts`, `quarter`, `split`, `purged` | as in §3.1 and §3.2 |
| `role` | `taker` or `maker` |
| `direction` | taker row: `taker_side`; maker row: the opposite side |
| `cost_cents` | normalized YES price if direction is yes, otherwise normalized NO price (§2). Range 1–100; only 1–99 enters returns. |
| `contracts` | `count` |
| `hours_since_open` | `(ts − open_time)` in hours; null if `open_time` is null |
| `prior_notional_usd` | in the same market, Σ `count × yes_price / 100` over trades strictly before this one (normalized yes price). Uses the market's full local history, including trades before 2024-10-01. In holdB it uses only trades the live endpoint returns. |
| `prior_trades` | number of trades strictly before this one in the same market (same history rule) |
| `prev_yes_price` | normalized `yes_price` of the immediately preceding trade in the market; null for the first trade |
| `hours_to_close` | `(close_time − ts)` in hours. Possibly hindsight (Appendix B). **Descriptive only.** |
| `close_time` | the market's `close_time`. Used only for `hold_days` and descriptive timing. |
| `result`, `settled`, `void`, `won` | `settled` = result ∈ {yes, no}; `void` = status `finalized` with empty result; `won` = (direction = result) when settled, otherwise null. **All null in the holdA and holdB panels.** Outcomes are joined only by `unlock.py`. |

- Within a market, ties on timestamp are ordered by `trade_id`.
- holdB block trades (`is_block_trade = true`) are dropped when the data are normalized. They are negotiated off-book, so they are not evidence that a taker could have traded at that price.

### 3.4 Fee model

- Kalshi's taker fee per order is `fee_cents = ceil(7 × m × C × P × (1 − P))`, where `C` = contracts, `P = cost_cents / 100` and `m` = series multiplier. That is 0.07·m·C·P(1−P) dollars, rounded up to the next cent. The code uses integer arithmetic.
- **Series multipliers:** `m = 0.5` for S&P 500 and Nasdaq-100 series (series matching `^(KX)?(INX|NASDAQ100)`); `m = 1` for everything else.
- Verification and sources are in Appendix A. Apart from the S&P/Nasdaq multiplier, the absence of other series-specific multipliers during 2024-10 → 2025-11 is an **assumption**. If some series actually paid less, using 0.07 overstates fees, which works against H1.
- Maker fees are out of scope: the strategy only takes liquidity.
- **Order size:** base N = 100 contracts; sensitivity sizes N = 10 and N = 1000.

### 3.5 Strategy and entry rule

A configuration is a tuple `(band, category_filter, liquidity_floor)`:

| Parameter | Values |
|---|---|
| `band` (NO cost, cents, inclusive) | `[90,99]`, `[93,99]`, `[95,99]` |
| `category_filter` | `all`, or `ex_finance` (excludes the group labelled exactly `"Finance"` in `categories.py`) |
| `liquidity_floor` | `prior_notional_usd ≥ 0`, or `≥ 500` |

- That is 3 × 2 × 2 = **12 configurations, exactly**. Grid order: band (outer loop), then category filter, then floor, each in the order listed.
- **Entry**, for each market in a split:
  - the first taker row (ordered by `ts`, then `trade_id`) that has `direction = no`, `cost_cents` in the band, passes the category and liquidity filters, is not purged, and has a timestamp inside the split window;
  - one entry per market, of N contracts at that `cost_cents`;
  - only taker-NO prints qualify, because they show that a taker actually bought NO at that price.
- **P&L per entry, in cents:**
  - `cost = N × cost_cents + fee_cents(N, cost_cents)`
  - `payout = 100 × N` if `result = no`; `N × cost_cents` if void (the fee is not refunded); `0` if `result = yes`
  - `net_ret = (payout − cost) / cost`; `gross_ret` is the same without the fee
  - `hold_days = (close_time − entry_ts)` in days, floored at 1/24. This is a proxy for capital lock-up, used only for per-capital-day metrics.
- Entries in markets that are neither settled nor void when outcomes are read are **censored**: they are excluded and their count is reported.
- The unit of observation is the market; clusters are `event_ticker`.

### 3.6 `close_time` audit

- The audit is in Appendix B.
- Conclusion: for finalized markets `close_time` is the **actual** trading close. That is earlier than the scheduled expiry whenever a market closes early.
- `hours_to_close` is never used in a trading rule.

---

## 4. Statistics

**Clustered mean test.**
- Inputs: per-entry `net_ret` values r_i with event clusters g; n entries and G clusters.
- Mean: r̄.
- SE = sqrt( G/(G−1) × Σ_g (Σ_{i∈g} (r_i − r̄))² ) / n.
- t = r̄ / SE, tested one-sided against H1: mean > 0.
- The one-sided p-value uses a Student t distribution with G − 1 degrees of freedom.

**Market-clustered SE.** The same formula with each market as its own cluster. Each market has exactly one entry, so this is the heteroskedasticity-robust SE.

**Cluster bootstrap.**
- Resample events with replacement: 1,000 iterations, seed **20251001**, using `numpy.random.default_rng`.
- Each replicate mean is (Σ of the resampled cluster sums) / (Σ of the resampled cluster sizes).
- Report the 95% percentile CI (2.5th and 97.5th percentiles) of r̄.

**Multiple testing.**
- Bonferroni-adjusted p-value for every train configuration: min(1, 12 × p).
- Deflated Sharpe ratio (Bailey & López de Prado, 2014) for the frozen configuration, using N = 12 trials and the variance of the per-entry train Sharpe ratios of all 12 configurations. Descriptive only.

**Descriptive metrics in every results table.**
- `mean_gross_ret`.
- `mean_net_ret_per_capital_day` = Σ pnl / Σ (cost × hold_days).
- `win_rate` = share of entries with `net_ret > 0`.
- `worst_event_loss_usd` = the minimum over events of the event's summed P&L in USD. Negative means a loss.

---

## 5. Procedure, commits and tags

All work is on branch `prereg/longshot-fade`, created from `origin/main`. The branch and every tag are pushed to `origin` (the fork) as soon as they exist. Nothing is ever pushed to `upstream`. Commits 1–7 are never amended, rebased or force-pushed, and no tag is moved or deleted.

1. **Commit 1, tag `protocol-v1`:** this protocol, `config/protocol_v1.json`, `DEVIATIONS.md`, `HOLDOUT_LOG.md`, the audit script and its output. The draft PR "Longshot-bias fading test" is opened on the fork right after.
2. **Commit 2, positions table and tests.**
   - Code: `config.py`, `fees.py`, `panel.py`, `strategy.py`, `stats.py`, plus `unlock.py` and `holdout_b.py`. The last two are committed now so the holdouts run on frozen code.
   - All unit tests.
   - The panel for train, val and holdA, with null holdA outcomes, and `results/panel_manifest.json`. The manifest holds row counts per split and quarter, the purge summary, the unmapped-category share and the SHA-256 of every holdA panel file.
   - **Train-only sanity check:** mean maker `gross_ret` − mean taker `gross_ret`, per position, must be > 0. It is written to `results/sanity_train.json`. If it is negative, work stops and it is investigated as a data bug. §3 is not changed to make it pass.
3. **Commit 3, train grid:** all 12 configurations on train → `results/train_grid.csv`. Rank by `t_event` (ties: `n_events`, then grid order); the top 3 go to validation.
4. **Commit 4, validation:** the top 3 on val → `results/val_top3.csv`. Selection rule below. If no configuration qualifies, the verdict is "Not supported (failed validation)". Commits 5–7 are then skipped and the holdouts stay locked.
5. **Commit 5, tag `prereg-frozen`:** `freeze.py` writes `config/frozen.json`. It contains the chosen configuration, N, the fee model, the seed, the thresholds, the commit-4 hash and the panel-manifest hash. A Frozen addendum is appended here. The working tree is clean afterwards.
6. **Commit 6, tag `holdout-a-run`.** `unlock.py --holdout A` refuses to run unless all of these hold:
   - HEAD is the `prereg-frozen` commit;
   - `git status --porcelain` is empty;
   - every holdA panel file (exact file set) matches its SHA-256 in the manifest;
   - `HOLDOUT_LOG.md` has no earlier holdout-A entry.

   It then backfills outcomes for frozen-configuration entry markets whose local status is not `finalized`. It tries the live `GET /markets/{ticker}` first, then `GET /historical/markets/{ticker}`, and stores the responses in `data/research/longshot_fade/backfill/`. It runs the frozen configuration once and writes `results/holdout_a.json`. That file holds the train-table metrics, the market-clustered t, the pass flags, and the censored and void counts. It appends the run to `HOLDOUT_LOG.md`. The results are committed unchanged.
7. **Commit 7, tag `holdout-b-run`.**
   - `holdout_b.py`, run at HEAD = `holdout-a-run`, pulls trades for the §3.1 window. It uses the global `GET /markets/trades` feed with a resumable cursor, plus market metadata.
   - The pull stores **no outcome fields**: it keeps only ticker, event ticker and open, close and created times.
   - It normalizes the data to the §3.3 schema (integer cents via exact decimal arithmetic) and builds the holdB panel with the frozen code. It applies the reverse purge and writes a manifest of SHA-256s to `data/research/longshot_fade/holdB/`.
   - `unlock.py --holdout B` runs the same guards, with HEAD = `holdout-a-run`, plus one more: at least **7 days** have passed since the pull completed.
   - It fetches outcomes once, excludes unsettled entries (censored, counted), and writes `results/holdout_b.json`. The holdB manifest is copied into `results/`.
   - **Power gate:** fewer than **300 event clusters** with entries means B is reported as descriptive only, with status "underpowered".
   - If the API is unreachable or access is not permitted, B is "unavailable" and this is recorded in `DEVIATIONS.md`.

Anything after commit 7 (or after commit 4 if validation fails) is descriptive. It cannot modify `frozen.json` or any committed result. If a bug is found after an unlock: fix it in a new commit, log it in `DEVIATIONS.md`, re-run into a new results file labelled `post-unlock correction`, and keep the original file.

---

## 6. Registered items (verbatim)

### Estimator (verbatim from spec §5)

- **Clustered mean test.**
  - Inputs: per-entry `net_ret` values r_i with event clusters g, n entries and G clusters.
  - Mean: r̄.
  - Standard error: SE = sqrt( G/(G−1) × Σ_g (Σ_{i∈g} (r_i − r̄))² ) / n.
  - t = r̄ / SE, tested one-sided (H1: mean > 0).
  - Also report the SE clustered by market. Each market has one entry, so this equals the heteroskedasticity-robust SE.
- **Cluster bootstrap.** Resample events with replacement, 1,000 iterations, seed **20251001**. Report the 95% percentile CI of r̄.

### Selection procedure (verbatim from spec §6)

- Rank the configurations by `t_event` (ties broken by `n_events`, then the order in §4.5). The top 3 go to validation.
- **Selection rule:** among the 3, keep those with validation `mean_net_ret > 0`, and choose the one with the highest validation `t_event`.
- **If none qualifies:** the verdict is **"Not supported (failed validation)"**. Skip commits 5–7, still write `REPORT.md` (§8) and finish the PR (§10). The holdouts are **not** unlocked.

### Hypothesis, decision rule and verdict (verbatim from spec §7)

**H1.** On Kalshi, a liquidity taker who buys NO at a cost basis inside the frozen band (within 90–99¢, i.e. against YES at 1–10¢) and holds to resolution earns a positive mean return per dollar at risk, net of taker fees.

**A holdout passes if both hold:**
1. the one-sided event-clustered t-statistic is ≥ 2.0; **and**
2. the lower bound of the 95% event-cluster bootstrap CI of the mean `net_ret` is > 0.

**Verdict table:**

| holdA | holdB | Verdict |
|---|---|---|
| pass | pass | **Supported (blind replication)** |
| pass | unavailable or underpowered | **Supported (semi-blind only)** |
| pass | fail | **Not replicated out of sample** |
| fail | any | **Not supported** |
| (validation failed) | (not run) | **Not supported (failed validation)** |

Everything else is secondary and descriptive and cannot change the verdict: market-clustered t, breakdowns by category or price bucket, sensitivities, the portfolio backtest, and the per-trade Becker replication.

---

## 7. Clarifications fixed before registration

The spec does not cover these points. Each was decided before any outcome was examined and is also listed in `DEVIATIONS.md`.

1. The price normalization rule (§2), and excluding costs outside 1–99¢ from returns.
2. `won` is null unless the market settled. Entries that are neither settled nor void are censored and counted.
3. p-values use Student t with G − 1 degrees of freedom. The bootstrap is a percentile bootstrap with `numpy.random.default_rng(20251001)`. The descriptive metrics are defined in §4.
4. Purged rows stay in the panel, flagged `purged = true`. The panel also carries `close_time`.
5. `prior_*` features use each market's full local history. In holdB they use only the trades the live endpoint returns.
6. HoldB operations:
   - live endpoint only;
   - reverse purge;
   - no outcome fields stored during the pull;
   - block trades dropped;
   - the manifest kept under `data/` until unlock, then copied to `results/`;
   - a 7-day minimum wait before the single outcome fetch;
   - the unlock HEAD check accepts `holdout-a-run`.
7. The holdA backfill covers only frozen-configuration entry markets whose local status is not `finalized`, since only their outcomes enter the test. Each is tried on the live endpoint, then on the historical endpoint.
8. Fee series multipliers (§3.4, Appendix A).
9. **Network access** (observed 2026-10-05):
   - `api.elections.kalshi.com` answered from this machine early in the session (the `/historical/cutoff` and audit requests in Appendix B). Later, connections to `kalshi.com`, `api.elections.kalshi.com` and `polymarket.com` were reset during the TLS handshake, while other hosts (GitHub, `docs.kalshi.com`) worked. That pattern suggests network-level filtering of prediction-market domains.
   - The study does not try to get around any network restriction.
   - If the API cannot be reached, or access is not permitted, when a holdout needs it: holdA markets without a local outcome are censored (excluded and counted), and holdB is "unavailable".

---

## Appendix A. Fee verification (2024-10 → 2025-11)

| Item | Status | Source |
|---|---|---|
| General taker fee `round up(0.07 × C × P × (1 − P))`, rounded up to the next cent per order | **Verified** at both ends of the window: the 1.75¢ midpoint fee is implied by the 2022 announcement; a 2025 secondary source states it; the current schedule states it | Kalshi, "We're Halving the Fees for Our S&P and Nasdaq Markets", news.kalshi.com/p/were-halving-the-fees, 2022-09-14 ("fees are dropping from 1.75c to 0.875c per contract at the midpoint price" for S&P/Nasdaq, so 1.75¢ = 0.07 × 0.25 was the general rate). A. Courtney, "Maker/Taker Math on Kalshi", whirligigbear.substack.com, 2025-09-17 (taker fee 0.07·P·(1−P), maximum 1.75¢ at 50¢). Kalshi Fee Schedule, kalshi.com/docs/kalshi-fee-schedule.pdf, "Fee Schedule for July 2026 – 7.7.26 Update", as indexed by web search on 2026-10-05 (`fees = round up(0.07 x C x P x (1-P))`; worked examples: 1 contract at 50¢ → $0.02; 100 contracts at 5¢ → $0.34). |
| S&P 500 and Nasdaq-100 markets at 0.035 (multiplier 0.5) | **Verified** from 2022-09-22 (Kalshi primary source). That it continued through 2024–2026 is supported by 2026 secondary summaries; the 2026 primary PDF could not be downloaded from this machine. | Same Kalshi announcement (2022-09-14, effective 2022-09-22). 2026 fee summaries found by web search on 2026-10-05 (e.g. defirate.com/prediction-markets/fees) state that S&P 500 and Nasdaq-100 markets use a 0.035 coefficient. |
| No other series-specific taker multiplier during 2024-10 → 2025-11 | **Assumption** | The archived 2024–2025 schedules could not be retrieved on 2026-10-05: the Internet Archive was offline, and `kalshi.com` reset connections from this machine. Charging 0.07 where the true fee was lower overstates costs, which works against H1. |
| Maker fees | Out of scope | — |

Series matched by `^(KX)?(INX|NASDAQ100)` in the local data: INX, INXD, INXU, INXW, INXY, INXDU, INXI, INXZ, INXM, INXB, NASDAQ100 and its D/U/W/Y/DU/I/Z/M variants, KXINX, KXINXU, KXINXY, KXINXE, KXINXEU, KXINXMINY, KXINXMAXY, KXINXMINW, KXINXHIGH, KXINXPOS, KXNASDAQ100, KXNASDAQ100U, KXNASDAQ100Y, KXNASDAQ100E, KXNASDAQ100EU. The regex does not match KXJOINXAI or KXSOLNASDAQ.

## Appendix B. `close_time` audit (train only)

- **Script:** `src/research/longshot_fade/audit_close_time.py`. **Output:** `results/close_time_audit.json`.
- **Scope:** 111,864 finalized markets in events whose trades from 2024-10-01 all fall before 2025-07-01. `result` was read only for these train events.
- **Close minus last trade (hours).**
  - Quantiles: 1% 0.0004, 5% 0.004, 25% 0.06, median 0.28, 75% 1.84, 95% 39.4, 99% 484.
  - 12.3% of markets close within 1 minute of their last trade; 71.2% within 1 hour.
  - 0.28% have trades timestamped after `close_time`.
  - 73.4% of close times fall on a round quarter-hour; the rest carry sub-second timestamps.
- **Multi-market events.** YES-resolving markets close before their event's latest close in 4.2% of cases (n = 38,803), against 2.6% for NO-resolving markets (n = 69,907). In more than 95% of these events every market shares one `close_time`.
- **API fields** (12 train markets via `/historical/markets/{ticker}`). Markets carry `can_close_early`, `early_close_condition` ("will close and expire early if the event occurs"), `expected_expiration_time`, `latest_expiration_time` and `settlement_ts`. For early-closed markets, `close_time` equals the actual halt rather than the schedule. Examples:
  - `KXVOTERECNCH-26-RMCC` (YES): closed 2025-05-22 15:25:54.238; expected expiration 2026-01-01.
  - `KXSECPRESSMENTION-25OCT12-STOCK` (NO): closed 2025-04-15; expected expiration 2025-10-12.
  - `KXOSCARLAS-25-A`: closed when the Oscars ended (2025-03-03 02:39).
  - Settlement typically follows the close by about 30 minutes.
- **Conclusion.** For finalized markets, `close_time` is the **actual** close.
  - It matches the schedule for markets that run to term and is earlier for markets that close early, whether because the event occurred or the outcome became known. It therefore carries outcome-timing information (hindsight).
  - YES markets close early only modestly more often in this sample.
  - `hours_to_close` stays descriptive and is never used in a rule.
  - `hold_days` uses `close_time` as the end of capital lock-up, which matches when capital is actually released (settlement follows close within minutes to hours).

## Appendix C. Non-outcome data checks run before registration

| Check | Result |
|---|---|
| Trades rows / unique `trade_id` | 72,134,741 / 72,134,741 |
| Markets rows / unique `ticker` / unique `event_ticker` | 7,682,445 / 7,682,445 / 1,197,300 |
| `taker_side` values | yes 49,694,124; no 22,440,617 |
| `count` | min 1, max 3,127,823, no non-positive values |
| `yes_price + no_price ≠ 100` | 267,291 rows, all summing to 99, all in 2025Q4, all fetched 2025-11-25 |
| Rows with a 0 price | 41,705 (41,663 of them are the sum-99 pair (0,99)) |
| Trades per quarter (UTC) | 2024Q3 349,838 · 2024Q4 4,127,877 · 2025Q1 6,048,928 · 2025Q2 7,899,836 · 2025Q3 17,218,565 · 2025Q4 32,634,469 |
| Trades without a matching market (from 2024-10-01) | 0 |
| Market `status` values | finalized 7,320,904 · active 328,865 · initialized 20,536 · closed 11,788 · inactive 342 · determined 9 · disputed 1 |
| `result` values present | `yes`, `no`, empty |
| Null `open_time` / `close_time` / `event_ticker` | 0 / 0 / 0 |
