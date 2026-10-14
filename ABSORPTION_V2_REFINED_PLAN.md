# Absorption v2: refined research specification

5 October 2026. Planning only; no Python or notebook changes. This narrows the earlier plan to v2. V1, compression, and PVD are outside implementation scope.

Implementation update: the user has now requested the independent analysis module. See `finding_threshold/expected_profit_analysis.py` and `config.ABSORPTION_EXPECTED_PROFIT`; existing notebook/heatmap code remains unchanged. The user clarified the final rule as **PROFIT FIRST**: required favorable movement before the first stop is SUCCESS, and a later stop cannot overturn it. Success size is maximum opportunity before the first stop or the ten-bar limit; first stop before required profit is FAILURE, sized at configured signal-time ATR distance. The earlier unresolved example in this document is superseded by this clarification. If both first profit and stop occur within one daily bar, chronology is unknown: the initial implementation assumes stop first and exposes an uncertainty flag.

Combined mode now accepts arbitrary positive integer candle lengths. Within each threshold pair and instrument, longest qualifying detection windows are selected first; any overlapping smaller window is suppressed even across directions/dates. Equal-length ties prefer the earlier signal. This is a retrospective research counting rule. Separate mode does not suppress across durations. Each threshold retains both selected and suppressed ledgers for audit.

## 1. Agreed scope and remaining choices

Retain aggregated candles for detection and use raw future 1D bars for outcome evaluation. Sweep two detector thresholds only: one shared positive magnitude X for signed volume-delta z-score, and one price-response ceiling Y. Do not optimize separate BUY and SELL thresholds.

Refresh data coverage whenever the analysis runs, for all currently selected front-five spreads and front-five jumbos per requested product. A configurable history window must replace hardcoded start dates.

Confirmed date convention and remaining payoff clarification:

- Confirmed by the user: N=5 means five calendar months including the current month. Start at the first day of the month N−1 months before the run month. Any October run therefore starts 1 June and ends at the latest completed available trading session at run time. Additional earlier warmup is fetched separately.
- Confirmed: if required profit occurs first, the event remains SUCCESS despite a later stop. If stop occurs first, event is FAILURE. Only pre-stop movement contributes to size.

## 2. Recommended baseline: overlapping candles of the same size

Prefer the user's second method. Build historical k-day candles with a one-trading-bar stride, and compare the current k-day candle with a recent distribution of historical k-day features.

Example: current 2D candle is trading days 11–12. Historical candles must end no later than day 10:

```text
current:    (11,12)
baseline:   (9,10), (8,9), (7,8), (6,7), (5,6),
            (4,5), (3,4), (2,3), (1,2)
```

Ten prior days give nine complete 2D candles, not ten. Ten complete 2D historical candles require eleven prior trading bars.

For W historical k-day candles, the minimum raw history for their aggregation is `W + k − 1` prior trading bars. Including the current k-day signal window gives `W + 2k − 1` total bars. Thus W=15, k=5 needs nineteen prior bars and five current bars: twenty-four in total, plus any extra warmup required by normalization windows and the previous-close feature. Fifteen non-overlapping 5D candles require seventy-five prior bars; that is a different sampling scheme.

The latest baseline candle must end before the current signal candle starts. For a current candle ending at index t, it spans `[t−k+1, t]`; baseline endpoints are `t−k, t−k−1, ..., t−k−W+1`. Each historical candle uses the same k-day feature definition. This excludes constituent-day sharing between the baseline and current signal candle, although historical baseline candles overlap each other.

Why this is preferable to comparing a raw 5D candle with 1D candles:

- Five-day signed delta naturally accumulates; five-day displacement has a different distribution from daily displacement.
- Same-size baselines compare the accumulation with typical accumulation over the same duration.
- Opposite-signed daily deltas cancel in both historical and current k-day candles. Persistent same-signed deltas accumulate in both. Accumulation alone should not make the current candle exceptional.
- A local baseline reduces the age of the observations. It cannot guarantee adjustment to every regime or seasonal change.

Historical overlap is not future leakage, but it creates dependence. Fifteen overlapping 5D windows contain substantially less information than fifteen independent windows. Treat W=15 as a provisional recent baseline, not statistically abundant support. A limited 15/20/30-window sensitivity study is appropriate; select the setting on chronological validation rather than repeatedly tuning on final outcomes. Inspect zero/tiny MAD and require finite, sufficiently supported baseline values; do not let epsilon produce apparently enormous meaningful z-scores from a degenerate distribution.

## 3. Feature definitions and signed direction

Current `calculate_rolling.py` does not yet implement the recommended same-size historical distribution: it scores combined candles against 1D-derived baselines. It also uses close-to-open displacement, while standalone v2 uses close-to-previous-close displacement. The baseline change therefore needs an explicit scorer version.

Recommended consistent price feature for every k:

```text
D_k(t) = sum(buyvolume − sellvolume over the current k trading bars)
P_k(t) = abs(close[t] − close[t−k])
```

Here `close[t−k]` is the close immediately before the first constituent bar. This extends the standalone 1D feature consistently and includes intervening gaps. Record high-low range separately: small net displacement can hide large oscillations.

Retain causal normalization if desired, but apply the identical k-day feature pipeline to each historical candle. A possible delta normalization is `D_k / (k × prior daily mean volume)` and a price normalization is `P_k / prior daily ATR`. Both scales must be available before that candle starts. Baseline historical features retain their own historical scales. The added normalization warmup is separate from the aggregation counts above. Do not compare normalized current features to unnormalized historical ones or introduce duration scaling to only one side of the comparison. Choosing raw versus normalized features remains a documented model setting.

Use the existing robust convention unless explicitly changed:

```text
z_delta = (normalized signed D_k − historical median) / (1.4826 × MAD)
z_price = (normalized absolute P_k − historical median) / (1.4826 × MAD)
```

Do not take absolute delta before building its distribution. Price response is absolute before constructing its distribution; **z_price itself remains signed**. A negative z_price indicates below-median price response.

The user requires positive raw volume delta to mean BUY absorption and negative raw delta to mean SELL absorption. Median-centered z_delta alone does not guarantee the raw sign. Recommended explicit gates:

```text
BUY absorption / short: D_k > 0 AND z_delta >= X  AND z_price <= Y
SELL absorption / long: D_k < 0 AND z_delta <= −X AND z_price <= Y
X > 0; same X for both directions, same Y for both directions
```

The sign checks are not additional thresholds. They preserve the stated meaning and prevent a below-median positive raw delta being assigned SELL. A shared X/Y implements the senior's simplicity requirement even if a skewed signed distribution yields different BUY/SELL frequencies. Report each side and pooled outcomes using that same threshold pair.

## 4. Multi-day detection versus daily outcome evaluation

A 5D signal summarizes five completed trading days. It is available only when the fifth bar closes; do not use it to claim a prediction on its first day. After the signal is available, daily future execution is the correct resolution for chronological stop checks. Daily execution does not turn the detector into a 1D detector.

Fixed k describes the observation window, not the true duration of an absorption episode. The current code detects bars/windows; it does not locate a confirmed start/end of a variable-length period.

For long absorption periods, additionally track runs of same-direction qualifying signals as episodes. Record first qualifying date, ongoing dates, duration, and last qualifying date. At the first signal, future episode duration/end are unknown. Do not retrospectively enter on the eventual last signal or best signal. A clean first comparison reports all events plus a first-event/one-active-position-per-instrument version. Future signals may be marked as continuation; they must not automatically reset the entry, stop, or ten-bar outcome window. Any rule that waits for a causal end-of-absorption confirmation is a separate entry model with different timing.

Study k=1/2/3/5 separately at first. Combining all lengths into one detector requires a later explicit priority/deduplication rule, because windows may signal together or in opposite directions.

## 5. Maximum opportunity before stop

The user's size proposal is primarily a maximum favorable excursion before a stop, over the next H=10 raw trading bars, rather than a fixed take-profit model.

Define entry reference E, signal-time ATR A, and stop multiplier a (e.g. 1.5). Hold A and the stop fixed; do not recompute the stop using future ATR.

| Side | Stop level | Stop test on raw future bar j | Favorable intrabar movement |
|---|---|---|---|
| BUY absorption → short | `E + a × A` | `high[j] >= stop` | `E − low[j]` |
| SELL absorption → long | `E − a × A` | `low[j] <= stop` | `high[j] − E` |

For a short, **maximum high**, not minimum low, determines whether the stop was reached before a favorable low. For a long, minimum low determines stop risk.

Scan chronologically. Once the stop hits, ignore all later bars. Accumulate maximum favorable movement from fully observed pre-stop bars. On the stop-hit bar, the high/low ordering is unknown: its favorable extreme cannot automatically be counted as occurring before the stop. Resolve with complete lower-timeframe data if available; otherwise report pre-stop opportunity bounds or conservatively exclude the favorable extreme from that bar. This temporal resolution is different from a volume filter; no hourly volume floor is implied by using hourly data to resolve ordering.

Store at least:

- `max_favorable_before_stop_ticks` and its time/bar, plus ambiguity flag/bounds.
- `stop_hit`, stop time/bar, `stop_distance_ticks = a × A / tick_size`.
- Whether a chosen minimum opportunity x ticks was reached before stop (a size label).
- Signed terminal movement for no-stop timeouts.
- Censoring and missing-data reasons, independently from failure.

If the first stop is day three, a day-seven low is never part of the short's available opportunity. If the position survives all H bars, opportunity is the maximum favorable movement within those bars. A completed no-stop window with less than x ticks of favorable movement is an insufficient-opportunity event, not an automatic full stop loss.

## 6. Opportunity expectancy versus realized profit

Example: a short first offers +8 ticks, then eventually hits a 15-tick stop.

- Maximum available pre-stop opportunity can be +8 ticks.
- Holding until stop realizes −15 ticks under an ideal stop-level fill.
- Exiting earlier at a predeclared take-profit could realize another amount.

The retrospective peak is not an exit decision available at the time. Thus assigning every success its maximum ten-day opportunity and every stop event its fixed stop loss produces a research payoff score with optimistic exit assumptions; it cannot be described as established trading expectancy.

Recommendation: report two panels. Panel A answers the senior's signal-size question using opportunity distributions and opportunity labels. Panel B uses a defined causal exit policy for signed P&L and threshold profit ranking. A stop-only model with horizon-close exit is one simple option; a fixed target or a defined trailing stop is another, but no target/trailing choice should be silently imposed.

Even if ideal stopped losses are `a × A` in price units, this is not the same tick loss for every event because A varies. Gaps through stops and costs can make realized loss larger. Store ideal stop loss separately from modeled execution loss. If the requested first version deliberately ignores gaps/costs, label that assumption explicitly.

For a complete signed payoff Y and selected events:

```text
mean_payoff = P(success) × mean(Y | success)
            + P(failure) × mean(Y | failure)
```

Use signed Y. Only use `p_success × average_gain − p_failure × average_loss` when all successes are actual gains and all failures actual losses under that specific definition. Include no-stop insufficient-move events and completed timeouts; do not silently exclude them. For opportunity-only scores, use names such as mean pre-stop opportunity and opportunity-versus-stop score, keeping them distinct from realized profit.

## 7. Cache coverage on every run

Compute run date at function invocation, using the configured user/reference timezone. Avoid import-time `today()` defaults that become stale in a long-running notebook.

For each current front-five spread and jumbo:

1. Determine requested analysis start, latest completed market session, and additional pre-start warmup needed for k-day baselines and ATR.
2. Read cache and validate columns, unique sorted timestamps, OHLC consistency, and volume data.
3. Check start/end coverage and internal session gaps. A min/max date check alone cannot establish complete coverage. Use the instrument/exchange session calendar or API availability; weekends and holidays are not missing bars.
4. Fetch missing historical/tail ranges and merge/deduplicate. Refresh the most recent provisional/revisable bars as needed. Newly selected instruments must be fetched even if yesterday's universe was fully cached.
5. Preserve extra valid history in cache, but clip experiment outputs to requested dates after using warmup bars for causal calculations.
6. Report insufficient listing history or unavailable API data rather than claiming coverage. Keep incomplete current-day daily bars out of completed-candle signal analysis.

Use the confirmed calendar-month convention: N=5 on any October run starts 1 June. Compute `analysis_start` as the first day of the run month minus N−1 calendar months; validate N >= 1. End at the latest completed available trading session at run time. Name this convention explicitly in config. A five-month cached dataset and a twenty-bar statistical baseline serve different purposes: data coverage supports many historical observations; each signal's baseline can still use only recent history.

## 8. Revised implementation order

1. Resolve date-window and payoff semantics; save effective configuration.
2. Add run-time coverage validation/refresh and warmup handling.
3. Implement v2 same-size overlapping baselines with a consistent price definition, signed delta distribution, and shared X/Y gates.
4. Map each aggregated signal end to raw 1D future bars; version the corrected outcomes.
5. Compute pre-stop opportunity, stop sizes, chronological outcome statuses, and side-level summaries.
6. Add a separately named causal P&L evaluation once the exit policy is settled.
7. Sweep the X/Y grid using precomputed directional outcomes; show counts, opportunity sizes, stop sizes, coverage, and any selected P&L metric. Keep economic and opportunity ranking names distinct.
8. Compare candidate settings on later chronological data, preserve dependence in uncertainty estimates, and freeze the final candidate before the last test period. Time-series validation must use earlier observations to predict later ones; see [Forecasting: Principles and Practice, time-series cross-validation](https://otexts.com/fpp3/tscv.html). Specific absorption baseline/window choices here are project recommendations, not claims from that source.

No final best W, k, X, or Y is asserted. The overlapping same-size method is the recommended first experiment; validation will determine whether it improves this dataset.
