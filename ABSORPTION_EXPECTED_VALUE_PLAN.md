# Absorption: current behavior and expected-value evaluation plan

Prepared 5 October 2026. This is a code review and implementation proposal only. No signal, label, configuration, or notebook logic has been changed, and no historical performance has been calculated.

Follow-up scope and recommendations are recorded in [ABSORPTION_V2_REFINED_PLAN.md](ABSORPTION_V2_REFINED_PLAN.md). That refinement takes priority for subsequent implementation: v2 only, same-size overlapping baselines, refreshed configurable data coverage, and maximum opportunity before stop separated from realized payoff. Its explicitly unresolved choices remain open.

## 1. What we are trying to answer

The current analysis asks: **Among bars selected by this threshold, how often does the absorption label succeed?**

The proposed analysis adds: **How large are successful and failed outcomes, and what is the average net trade result at this threshold under a stated entry/exit rule?**

For example, a threshold winning 70% of trades with average wins of 5 ticks and average losses of 15 ticks has gross expectancy of `0.70 × 5 − 0.30 × 15 = −1 tick/trade`. A threshold winning 50% with average wins of 12 ticks and average losses of 6 ticks has `+3 ticks/trade`. These are illustrative numbers, not project results.

Scope is absorption only. Keep compression and PVD exhaustion labeling outside this change, as requested. PVD expansion needs its own subsequent specification; do not automatically reuse absorption's contrarian trade direction or exits there.

## 2. Direction terminology

Your intended interpretation is consistent with the labelers:

| Signal name | Meaning intended by the research | Expected price move | Proposed position |
|---|---|---|---|
| BUY absorption | Buy aggression meets opposing liquidity and produces limited price response | Down | Short, position sign `s = −1` |
| SELL absorption | Sell aggression meets opposing liquidity and produces limited price response | Up | Long, position sign `s = +1` |

The signal name describes aggression, not the trade side. These are hypotheses being evaluated; the detection code does not guarantee a reversal or a large move. OHLC and buy/sell volume provide proxies for absorption, rather than direct proof of the liquidity mechanism.

There is an important v2 caveat: its BUY/SELL direction comes from **z_delta**, not necessarily the sign of raw delta. Since `z_delta` subtracts a historical median, BUY can theoretically occur with negative raw delta when it is unusually high relative to history, and conversely for SELL. The eventual analysis should report raw-delta/z-delta sign mismatches. Preserve current semantics for the first comparison; adding a raw-sign requirement would be a separately versioned signal change.

## 3. What the code does today

### 3.1 Configuration and data flow

`config.py` currently enables SM and BO, both with `tick_size = 0.1`, and front five instruments per structure. Structures are consecutive-month spreads and jumbos skipping one or two intervening contract positions. Daily detection horizons are 1d, 2d, 3d, and 5d.

Daily `ABSORPTION` settings:

- Percentile lookback: 60 bars; strength threshold: 0.30.
- Minimum label move: 2 ticks; stop lookback: 10 bars; forward evaluation: 10 bars.
- Score sweep: 0.10 to below 1.00, step 0.05, because `np.arange` excludes the stop.
- Low-count flag below 8 signals; 500 bootstrap samples; seed 42.

Hourly `ABSORPTION_1H` uses a 120-bar score window, 3-tick move, 20-bar stop lookback, and 40 future bars. `run_absorption_1h` also filters volume to strictly greater than 500 by default.

The v2 threshold module has its own `_DEFAULT_PARAMS`, outside central config: volume/ATR normalization windows 5/5, robust baseline 15, label ATR period 14, target and stop multipliers 1.0/1.0, forward window 10. Its default score sweep is 0.05 to below 1.00. The current `threshold_abs_v2.ipynb` overrides target and stop to **1.5/1.5**, uses SM, eight months of history, and sets `COMBINED_CANDLE = 3` for its per-instrument heatmap. These are distinct experiments and must be identified in outputs.

`data/ohlc.py` fetches/caches OHLC with buyvolume and sellvolume. N-day candles are **trailing, overlapping windows**, with one result per underlying day: first open, maximum high, minimum low, last close, and summed volume. A 3d detection candle does not imply stepping forward in three-day increments; the forward label scans subsequent rows.

### 3.2 Percentile scorer: `signals/absorption/calculate.py`

For each bar:

```text
delta = buyvolume − sellvolume
direction = BUY if delta > 0; SELL if delta < 0; otherwise NEUTRAL
aggression_score = percentile of |delta| against previous window bars
price_response = |high − low|
price_response_score = 1 − percentile of price_response against previous bars
absorption_strength = aggression_score × price_response_score
```

The score favors unusually strong net aggression combined with a small total candle range. It lies between 0 and 1 and is not a calibrated probability of success. The first full lookback is warmup. `strength_threshold` controls absorption-volume accounting; it does not prefilter the score sweep. Class labels NONE/POSSIBLE/etc. use separate fixed score boundaries.

### 3.3 Robust z-score scorer: `signals/absorption/calculate_v2.py`

```text
normalized_delta = delta / past rolling mean volume
normalized_price = |close[t] − close[t−1]| / past rolling mean true range
z = (current value − past median) / (1.4826 × past MAD + epsilon)

BUY gate:  z_delta >= X and z_price <= Y
SELL gate: z_delta <= −X and z_price <= Y

delta_strength = sigmoid(|z_delta|)
price_suppression = sigmoid(−z_price)
absorption_score = delta_strength × price_suppression
score set to zero when |z_delta| < 1
```

Lower z_price means a smaller relative **absolute** price response. Both sides use `z_price <= Y`; the old header describing a mirrored SELL inequality is stale. Normalization and statistical baselines use prior bars. The defaults are X=2.0 and Y=0.75, but the threshold module uses dummy gates to compute features/directions before applying search masks.

This detects muted net close-to-close displacement, which can coexist with a large intrabar range. It is therefore economically different from v1's narrow-range measure. Also, a z-price ceiling of +0.75 permits price responses somewhat above the historical median; stricter negative ceilings test stronger suppression.

### 3.4 Rolling scorer: `signals/absorption/calculate_rolling.py`

This builds N-day trailing candles and anchors baselines before the candle's first constituent day. Its v1 compares combined delta/range to prior **1D** histories. Its v2 compares summed delta normalized by a 1D volume baseline and `|combined close − combined open|` normalized by a 1D ATR baseline to prior 1D normalized histories.

Consequences to retain in the plan:

- These comparisons are causal, but distributions differ by combined-candle length. A threshold on 1d and 5d need not mean the same thing.
- Even with `combined_candle = 1`, rolling v2 uses close-to-open response, whereas standalone v2 uses close-to-previous-close response. They are not equivalent scorers.
- Standard `run_absorption` uses previously aggregated candles and baselines on that series. Prior rolling candles share constituent days with the current candle. This is overlap within the detection period, not necessarily future-data leakage, but differs from the rolling scorer's stricter baseline convention.

### 3.5 Three existing label definitions

| Route | Entry reference | Success rule | What failure means |
|---|---|---|---|
| `label_absorption` | Signal close | A future close moves **strictly more than** required ticks in the expected direction, and no stop is touched anywhere in the full window | Stop touched at any time, or insufficient favorable close movement |
| `label_absorption_hourly` | Signal close | Chronological scan finds a close meeting the tick target before a stop | Stop first, or neither level fires before timeout |
| `label_absorption_atr` | Signal close | Future high/low touches the ATR target before the ATR stop | STOP is failure; UNRESOLVED and AMBIGUOUS are excluded from binary metrics |

Daily tick stops use the signal bar and preceding `sl_lookback−1` bars: BUY stop is maximum high plus one tick; SELL stop is minimum low minus one tick. Because the entire future window is checked for a stop, a favorable move followed by a later stop is a daily failure even if a target-based trade could already have exited profitably. The comment in `absorption_threshold.py` calling this daily path profit-first does not match implementation.

Hourly stops use the same historical-extreme construction. Target comparison is inclusive (`>=` or `<=`). If the target close and stop extreme both qualify in one bar, code checks profit first and reports success. OHLC does not establish that ordering; a stop could have occurred earlier within that bar.

ATR labels use an arithmetic rolling mean of **true range**, including the signal bar, despite their docstring describing mean high-low. Targets/stops are symmetric around signal close unless separate multipliers are passed. The labeler can inspect hourly data when both barriers hit on one daily bar, but `_fetch_score_label` currently passes **neither hourly data nor an hourly fetcher**. Such rows therefore remain AMBIGUOUS. UNRESOLVED, AMBIGUOUS, warmup, and insufficient forward history must remain separate statuses in the new analysis.

### 3.6 Runner and threshold-search routes

| Entry point | Scorer | Labeler | Current selection/evaluation |
|---|---|---|---|
| `runner.run_absorption` | v1 on daily or aggregated trailing candles | Daily tick | Strength sweep, classification metrics, CSVs and distributions |
| `runner.run_absorption_1h` | v1 on hourly horizons | Hourly tick | Strength sweep, volume filter, classification metrics |
| `runner.run_absorption_rolling` | Rolling v1/v2 | Daily tick | Strength/score sweep; this v2 route does **not** use ATR labels |
| `absorption_threshold.run_absorption_threshold_analysis` | v1 on raw daily/hourly bars | Daily/hourly tick | Pooled BUY+SELL score table per instrument |
| `absorption_v2_threshold.run_heatmap_analysis` | Standalone or rolling v2 | ATR, forced BUY and SELL separately | X/Y gate heatmap of resolved success rate |
| `absorption_v2_threshold.run_heatmap_by_structure` | Standalone v2 | ATR twice | Pooled X/Y heatmap by structure |
| `absorption_v2_threshold.run_threshold_sweep` | Standalone v2 | ATR twice | Continuous score sweep using direction from dummy gates |

The v2 module's useful optimization is to label each bar twice, once as a hypothetical short and once as a hypothetical long, then select the relevant outcome with each threshold mask. Preserve this for move sizes and trade outcomes when entry/exit settings are unchanged.

`utils/metrics.py` and threshold-specific helpers compute precision, recall, F1, MCC, lift, and precision bootstrap intervals. No payoff size enters these metrics. `Low N` only highlights rows; it does not prevent choosing a threshold. For the ATR heatmap, success rate is conditional on resolved TARGET/STOP outcomes; a cell with many unresolved or ambiguous signals can look attractive while representing only a small portion of selected events.

## 4. Issues to address before interpreting economic results

1. **Runner import blocker:** `runner.py:578` contains a second `from __future__ import annotations` after executable definitions. Python requires future imports at the beginning. This is a source-level syntax blocker; it was not executed during this review. Remove the duplicated patch scaffold in the later implementation.
2. **Rolling label contamination:** rolling scored rows carry N-day highs/lows. At `t+1`, those extrema can include days at or before t. Passing those rows to a forward label scan can report a target/stop from pre-entry movement. Join signal features to raw future 1D execution bars. Compute stop history and label ATR on a explicitly defined raw series, rather than silently on overlapping aggregates.
3. **Scorer comparability:** standalone and rolling v2 use different price-response definitions, and combined candles are compared with 1D baselines. Keep scorer/candle versions visible; do not silently make them equivalent in an expectancy change.
4. **Notebook drift:** `threshold_abs_v2.ipynb` applies its 3D setting only to per-instrument heatmaps. Pooled heatmaps and score sweeps currently have no combined-candle argument and remain 1D. `threshold_abs.ipynb` contains an import of nonexistent `run_absorption_v2_threshold_analysis`, duplicated experimental code, and older SELL masks using `z_price >= −Y` alongside newer symmetric `<= Y` masks. Use the module-based v2 notebook as the maintained route.
5. **Data reproducibility:** the daily cache is keyed by instrument rather than requested date window, and returns existing data without verifying coverage or clipping to the request. Record actual date coverage, validate/refresh as needed, and explicitly clip the research range. Fix a reference date and instrument universe per experiment.
6. **Eligibility differences:** the hourly runner applies a volume floor; the standalone threshold route does not apply that same floor. Shared metrics use a default 11-bar bootstrap block even for hourly config because they read `forward_days`, not `forward_bars`. Record and align filters, warmup, finite-score checks, and evaluation horizons.

These are planned corrections or explicit experiment choices, not changes made in this document. If a correction changes labels, report a separate evaluation version with before/after counts.

## 5. Two complementary measurements

### 5.1 Signal move-size diagnostics, preserving old labels

First add size measurements so we can directly answer the senior's question without replacing existing success definitions.

For entry reference E and direction sign s, define signed price movement in ticks at future close j:

```text
signed_close_move[j] = s × (close[j] − E) / tick_size
terminal_move_ticks = signed_close_move[last evaluation bar]
close_MFE_ticks = max(0, maximum signed_close_move)
close_MAE_ticks = max(0, −minimum signed_close_move)
```

Also calculate intrabar MFE/MAE over raw execution bars: favorable highs and adverse lows for longs; favorable lows and adverse highs for shorts. Store the observation window and whether these are whole-window or pre-exit measurements.

Per threshold, show mean/median favorable excursion for legacy successes, adverse excursion for legacy failures, and the **signed terminal movement for both groups**. A legacy failure can still have a positive terminal move, and a legacy success can finish negative after initially moving favorably. Do not force all failures into a negative payoff or call maximum favorable excursion realized profit. Whole-window MFE can occur after a hypothetical stop and is descriptive opportunity only.

For any explicitly selected signed payoff variable Y, the exact partition is:

```text
E[Y | selected] = P(legacy success | selected) × E[Y | legacy success, selected]
               + P(legacy failure | selected) × E[Y | legacy failure, selected]
```

This supports the requested success/failure size comparison while keeping its economic interpretation honest. Report excluded label statuses and their signed outcomes separately where measurable.

### 5.2 Trade outcomes for profit-based threshold selection

Add a separate trade evaluator. It should simulate an explicit position and return one signed payoff per eligible event. Preserve legacy columns alongside it.

Proposed first trade model:

- Primary executable entry: next raw bar open after the complete detection candle is available. Retain signal close as a separate research-reference model for comparisons with old labels. Evaluate exits on the entry bar too; do not use that bar's close to choose its open entry.
- Stop: existing historical extreme for tick routes or ATR stop for ATR routes, based on information available at the signal close. Reject/flag an entry already beyond its stop. For the ATR model, use signal-time ATR distances around the actual entry and document this departure from close-based legacy levels; optionally compare fixed legacy price levels as a separate model.
- Exit: first target or stop, otherwise close of the final evaluation bar. Fix the window convention: H raw bars from entry inclusive. No exit information after the first exit contributes to trade P&L.
- Target: configurable tick or ATR distance. Initially hold target/stop/horizon fixed while sweeping detection thresholds. A fixed small take-profit cannot monetize a larger move beyond that target; use MFE diagnostics to decide whether a later exit-policy study is warranted.
- Gaps: a stop gapped through fills at the adverse opening price, not the stop level. For a resting take-profit limit, use the target price conservatively when an opening gap crosses it. State order activation and tick rounding rules.
- Same-bar double hits: resolve with complete lower-timeframe data when available. Otherwise mark ambiguous and compute conservative stop-first and optimistic target-first result bounds. Use conservative results for ranking and report ambiguity rates.
- Costs: store gross ticks and subtract configurable round-trip commission/slippage in ticks. Until actual cost inputs are available, report gross results plus an explicitly labeled cost-sensitivity grid. Product tick sizes exist; cash tick values and structure execution costs are not configured, so do not invent currency P&L.

For an unambiguous executed trade:

```text
gross_pnl_ticks = s × (exit_price − entry_price) / tick_size
net_pnl_ticks = gross_pnl_ticks − round_trip_cost_ticks
```

Timeouts are completed trades with a signed result, not censored events. Incomplete data at the end of the dataset is censored and excluded from realized expectancy. Ambiguous trades contribute bounds, rather than being silently dropped from a claim about all selected events.

## 6. Expected-value calculation at each threshold

Use two sets of counts: legacy label success/failure counts and economic win/loss/flat counts. A target hit is not automatically a net win if costs exceed the gain.

For economic wins and losses under the selected trade model:

```text
p_win = N(net_pnl_ticks > 0) / N_executed
p_loss = N(net_pnl_ticks < 0) / N_executed
avg_win_ticks = mean(net_pnl_ticks | net_pnl_ticks > 0)
avg_loss_ticks = mean(−net_pnl_ticks | net_pnl_ticks < 0)

net_expectancy_ticks = p_win × avg_win_ticks − p_loss × avg_loss_ticks
                    = mean(net_pnl_ticks)
```

Flat trades contribute zero. If a class has no members, its conditional mean is unavailable but its aggregate contribution is zero. Costs are already in net payoffs and must not be subtracted again. For gross wins/losses, deduct average costs once to get net expectancy. With only wins/losses and constant gross sizes W/L and cost c, break-even win probability is `(L + c)/(W + L)`.

Add a direct arithmetic check that the weighted calculation equals the average signed payoff. Do the same decomposition by legacy label to show how classification success relates to actual profitability.

For ATR targets/stops, equal multipliers do not imply equal tick gains across events: ATR varies, and timeout exits, gaps, and costs also matter. Report ticks as the requested primary unit; ATR-normalized returns and initial-risk R multiples are useful secondary diagnostics. Pooled ticks are an average per event, not pooled cash profit.

## 7. How the search should work

1. **Separate score thresholds from exit settings.** A score threshold T selects `score >= T`; a v2 gate pair X/Y selects BUY using `z_delta >= X and z_price <= Y`, SELL using `z_delta <= −X and z_price <= Y`. Initially study these as separate searches with fixed exits. Do not add target/stop optimization to the same first grid.
2. **Reuse event calculations.** Compute raw signal features once and short/long outcomes once per entry/exit model. Apply threshold masks afterward. Ensure directions and row identifiers match; reject overlapping BUY/SELL gates when X=0 or adopt an explicit tie rule.
3. **Keep side-level results.** Produce BUY-short, SELL-long, and pooled tables for each product, structure, instrument, and detection horizon. Pool by summing counts/payoffs, not averaging instrument success rates. Include the concentration of trades/profit by instrument.
4. **Distinguish events from a tradable sequence.** Initially every eligible signal can be measured as an independent event, including overlapping forward windows. Label totals as event-study totals. Also run one active position per instrument, accepting signals only when flat and skipping signals during an open trade; this sequence must be rerun for each threshold because selection changes future availability. Do not describe overlapping-event sums as strategy profits. Shared contract legs create additional cross-instrument dependence.
5. **Choose the objective explicitly.** Primary initial ranking: validation mean net ticks per executed trade, with sample-size and stability requirements. Also show validation total net ticks for the one-position sequence, frequency, exposure, and drawdown. Highest expectancy per trade and highest total profit can select different thresholds. If the agreed business objective is total profit under fixed sizing, use that sequence objective for final selection and retain expectancy as its explanation.
6. **Allow no selection.** A threshold with weak support, unstable behavior, or no credible positive net result should be reported as insufficient evidence. The present minimum of 8 is a display threshold, not a sufficient statistical justification.

Proposed threshold table:

| Group of columns | Fields |
|---|---|
| Experiment | Product, structure, instrument, side, candle length, scorer version, entry/exit version, cost scenario, date split |
| Selection | T or X/Y, candidates, selected events, executable trades, skipped overlaps, unique signal dates |
| Coverage | Resolved legacy labels, legacy successes/failures, timeouts, ambiguous, censored, rejected entries |
| Existing diagnostics | Precision, recall, F1, MCC, lift |
| Requested size diagnostics | Mean/median MFE for successes, MAE for failures, signed terminal outcomes by label |
| Economics | Net win/loss/flat counts, probabilities, mean/median win and loss ticks, mean net ticks, sum net ticks, expectancy CI/bounds |
| Risk/support | Profit factor, holding time, sequence drawdown, sample eligibility, instrument concentration |

Charts: expectancy against score threshold with confidence intervals and counts; v2 X/Y heatmap colored by mean net ticks, annotated with N; BUY/SELL views; win/loss size distributions; legacy success rate beside expectancy; coverage/ambiguity view. Keep existing success-rate heatmaps for comparison.

## 8. Validation and uncertainty

Use chronological development, validation, and untouched final test periods. Exact dates depend on actual usable cache coverage and warmup, so do not prescribe a split from requested months alone. If support permits, use expanding-window walk-forward folds.

- Fit/select thresholds on earlier data; choose among candidates on validation; freeze the candidate before evaluating the final test.
- Purge training events whose forward outcomes cross a validation/test boundary. Account for combined-candle overlap and the evaluation window when defining gaps. Prior historical bars may still be used to compute causal test features; future outcome data may not enter selection.
- Bootstrap contiguous blocks of the original calendar/bar sequence, preserving selected-event gaps, rather than compressed valid-row positions. For pooled results, resample date blocks jointly across instruments to preserve simultaneous shocks. Choose block length to cover outcome dependence and combined-candle overlap; examine sensitivity.
- Add intervals for expectancy and conditional payoff sizes, not only precision. Report sparse win/loss classes and low effective sample support.
- Prefer a stable neighborhood of good thresholds across folds over a single isolated best heatmap cell. Show how results change with plausible costs and ambiguity bounds.
- Freeze the instrument universe/reference date and label all conclusions with the tested universe. Current-front instruments alone do not establish historical universe performance.

No best threshold can be claimed from this source review. It requires an agreed payoff model and actual out-of-sample observations.

## 9. Proposed implementation order and file responsibilities

### Phase A: establish a reproducible absorption baseline

Fix the runner import scaffold; inventory effective config and notebook overrides; choose explicit scorer/price-response versions; verify data coverage and filters; map rolling signal dates to raw execution bars. Preserve a legacy-label report for comparison and version any corrected rolling labels separately.

### Phase B: add outcome-size measurements

Create a small absorption-specific evaluation module, proposed `signals/absorption/evaluate.py`. It takes signal features and a separate raw OHLC execution series. It returns stable event identifiers, legacy label references, signed terminal moves, MFE/MAE, and censoring/quality reasons. This delivers the senior's success/failure size analysis before any claim about realized profits.

### Phase C: add explicit trade outcomes

Extend the evaluation module with entry/exit scanning and a ledger: signal time, direction/position, entry/exit time and price, target/stop, exit reason, holding bars, gross/net ticks, cost model, and ambiguity bounds. Store available prices and parameters needed to audit a trade. Keep old labeler interfaces and columns intact.

### Phase D: add expectancy sweeps

Add a dedicated economic aggregator, proposed `utils/expectancy.py`, rather than changing the meaning of shared classification metrics. Integrate it with both absorption threshold modules. Preserve v2's label-twice efficiency, add pooled and side-level tables, and ensure all v2 entry points can use the same combined-candle setting and payoff version.

### Phase E: expose and validate the analysis

Add runner return fields such as `event_outcomes`, `trade_ledger`, `expectancy_table`, and `selection_summary`. Save ledgers/tables plus experiment metadata and plots under the existing product/signal/horizon output structure. Update `threshold_abs_v2.ipynb` to call maintained module functions; retire duplicate notebook logic from the active workflow. Add chronological splits, sequence evaluation, and a frozen final-test report.

In `config.py`, introduce separate absorption v2 and economic-evaluation settings: scorer windows, candle length, entry model, target/stop model, forward bars, costs, ambiguity policy, overlap policy, selection objective, validation dates, sample requirements, and bootstrap settings. Continue allowing notebook overrides, but print/save the effective merged configuration. Leave compression and PVD behavior unchanged.

### Focused checks when implementation is requested

- BUY short gains on falling prices; SELL long gains on rising prices; tick conversion and cost accounting are correct.
- Legacy failure can coexist with a positive economic payoff; weighted expectancy equals the mean ledger payoff.
- Target before stop, stop before target, timeout, double-hit ambiguity, adverse gap, and incomplete forward data behave as specified.
- A rolling signal cannot exit because of a high/low from before entry; raw execution starts only after the signal is available.
- Next-open entry uses no future information, and barriers use the declared signal-time information and rounding rules.
- Zero trades, no winners, no losers, flat trades, missing OHLC, and nonfinite scores have explicit outputs.
- Side/pool counts reconcile; one-position sequence skips overlaps; outcome windows cannot leak across evaluation splits.

## 10. Recommended first deliverable after approval to code

Start with daily absorption v2, supporting 1D and the current 3D detection experiment while executing on raw 1D bars. Preserve old ATR success-rate heatmaps, add legacy success/failure size tables, then add expectancy tables and heatmaps under fixed ATR exits. Keep a next-open trade model and a signal-close reference model clearly separate. Include signed timeout exits, coverage counts, costs/sensitivity, and ambiguity bounds.

Then extend the same evaluation layer to v1 and hourly absorption. This sequence answers the immediate question with a focused implementation and avoids conflating detector changes, label changes, and exit optimization.

The choices to settle when coding starts are the business ranking objective (per-trade expectancy versus total sequence profit), execution costs, primary exit model, and ambiguity treatment. The proposed defaults above make each choice explicit without altering the present code.
