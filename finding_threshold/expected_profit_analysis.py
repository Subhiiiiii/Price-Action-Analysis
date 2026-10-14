"""V2 absorption expected-size analysis, independent of existing heatmaps.

Public workflow (notebook integration is deliberately left for later)::

    from market_analysis.finding_threshold.expected_profit_analysis import (
        analyze_expected_profit, run_expected_profit_analysis,
    )
    result = analyze_expected_profit(raw_1d, tick_size=0.1,
                                     candle_lengths=[1, 2, 3, 5], mode="combined")
    daily = analyze_expected_profit(raw_1d, tick_size=0.1,
                                    candle_lengths=[1], mode="separate")
    products = run_expected_profit_analysis("SM", mode="combined")

Definitions agreed for this research:
* BUY aggression -> SHORT; SELL aggression -> LONG. One X/Y pair for all k.
* Detect on k-day candles, but evaluate on RAW future 1D bars.
* Required favorable move BEFORE stop -> SUCCESS. A later stop cannot overturn
  that success. Its size is maximum favorable excursion before the first stop
  (or the end of the forward window), not a guaranteed maximum-price exit.
* Stop BEFORE required favorable move -> FAILURE; its size is signal-time ATR
  * configured multiplier / tick, using ideal stop fills.
* No stop + insufficient favorable move -> UNRESOLVED, not an invented stop
  loss. Incomplete no-stop windows -> CENSORED, never premature successes.

Success maxima are hindsight opportunity. `expected_profit_ticks` is the
requested size statistic, NOT realized trading expectancy. Combined counting
is also retrospective: a later long candle can suppress an earlier short
candle. No live entry/exit strategy or portfolio profit is claimed here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ABSORPTION_EXPECTED_PROFIT, DATA_DIR, FRONT_N, OUTPUT_DIR, PRODUCTS

_FIELDS = ["open", "high", "low", "close", "volume", "buyvolume", "sellvolume"]
_RESOLVED = ["SUCCESS", "FAILURE"]


def _settings(params: Mapping | None = None) -> dict:
    """Copy/validate effective config so caller overrides never mutate defaults."""
    cfg = {**ABSORPTION_EXPECTED_PROFIT, **(params or {})}
    for key in ("months_back", "baseline_window", "baseline_min_periods",
                "volume_window", "atr_window", "atr_period", "forward_days"):
        value = cfg[key]
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f"{key} must be a positive integer")
        cfg[key] = int(value)
    if cfg["baseline_min_periods"] > cfg["baseline_window"]:
        raise ValueError("baseline_min_periods cannot exceed baseline_window")
    for key in ("stop_atr_mult", "mad_scale_floor"):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    for key in ("required_move_ticks", "min_signal_count", "warmup_extra_calendar_days"):
        if not np.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"{key} must be nonnegative and finite")
    return cfg


def _lengths(candle_lengths: Sequence[int]) -> list[int]:
    """Validate arbitrary requested durations; no hardcoded 1/2/3/5 branches."""
    values = list(candle_lengths)
    if not values or any(isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer))
                         or k < 1 for k in values):
        raise ValueError("candle_lengths must be a nonempty sequence of positive integers")
    if len(set(values)) != len(values):
        raise ValueError("candle_lengths must not contain duplicates")
    return sorted(int(k) for k in values)


def _raw_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize/validate raw daily bars without silently dropping trading days.

    Duplicate dates or invalid OHLC raise instead of changing k-day durations.
    Missing volume also raises: dropping such rows would bridge a missing day.
    Zero volume is allowed but cannot supply a positive normalization scale.
    """
    data = df.copy()
    data.columns = [str(c).lower() for c in data.columns]
    if data.columns.duplicated().any():
        raise ValueError("Duplicate column names after case normalization")
    missing = set(["date", *_FIELDS]) - set(data.columns)
    if missing:
        raise ValueError(f"Missing raw 1D columns: {sorted(missing)}")
    data = data[["date", *_FIELDS]].rename(columns={"date": "Date"})
    if data.empty:
        raise ValueError("Raw 1D data must contain at least one bar")
    data["Date"] = pd.to_datetime(data["Date"], utc=True, errors="raise").dt.normalize()
    for name in _FIELDS:
        data[name] = pd.to_numeric(data[name], errors="raise")
    if data["Date"].isna().any() or data["Date"].duplicated().any():
        raise ValueError("Raw 1D bars need unique nonmissing session dates")
    if not np.isfinite(data[_FIELDS].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite OHLC/volume values in raw 1D data")
    invalid = ((data["high"] < data[["open", "close", "low"]].max(axis=1))
               | (data["low"] > data[["open", "close", "high"]].min(axis=1))
               | (data[["volume", "buyvolume", "sellvolume"]] < 0).any(axis=1))
    if invalid.any():
        raise ValueError("Invalid OHLC bounds or negative volume")
    return data.sort_values("Date").reset_index(drop=True)


def _true_range(data: pd.DataFrame) -> pd.Series:
    """Raw daily true range, including gaps; first bar uses high minus low."""
    previous = data["close"].shift(1)
    return pd.concat([data["high"] - data["low"],
                      (data["high"] - previous).abs(),
                      (data["low"] - previous).abs()], axis=1).max(axis=1)


def build_v2_features(
    raw_1d: pd.DataFrame, candle_lengths: Sequence[int], params: Mapping | None = None,
) -> pd.DataFrame:
    """Build threshold-independent same-size robust z-scores for each duration.

    Current candle ends at t and starts t-k+1. Its baseline uses W historical
    k-day candles ending t-k, t-k-1, ...; none shares a day with CURRENT candle.
    Historical candles overlap each other, reducing effective independence.
    Each historical normalized feature uses its OWN causal pre-candle scale.
    Price displacement is abs(C[t]-C[t-k]); delta distribution stays signed.
    """
    cfg, lengths, data = _settings(params), _lengths(candle_lengths), _raw_bars(raw_1d)
    w, minp = cfg["baseline_window"], cfg["baseline_min_periods"]
    tr = _true_range(data)
    # These statistics include their indexed bar; shifting by k below makes
    # them stop at t-k, immediately BEFORE the current candle starts.
    mean_volume = data["volume"].rolling(cfg["volume_window"]).mean()
    mean_tr = tr.rolling(cfg["atr_window"]).mean()
    stop_atr = tr.rolling(cfg["atr_period"]).mean()
    frames = []
    for k in lengths:
        delta = (data["buyvolume"] - data["sellvolume"]).rolling(k).sum()
        price = (data["close"] - data["close"].shift(k)).abs()
        if cfg["normalize_features"]:
            delta_scale = k * mean_volume.shift(k)
            price_scale = mean_tr.shift(k)
            nd = delta / delta_scale.where(delta_scale > 0)
            np_ = price / price_scale.where(price_scale > 0)
        else:
            nd, np_ = delta, price

        def zscore(feature: pd.Series) -> tuple[pd.Series, pd.Series]:
            """Exclude the entire current candle, then median/MAD historical k features."""
            history = feature.shift(k).rolling(w, min_periods=minp)
            median = history.median()
            mad = history.apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
            scale = 1.4826 * mad
            # Zero/tiny MAD means an unsupported scale, not infinite evidence.
            z = (feature - median) / scale.where(scale > cfg["mad_scale_floor"])
            return z.replace([np.inf, -np.inf], np.nan), history.count()

        zd, count_d = zscore(nd)
        zp, count_p = zscore(np_)
        frame = pd.DataFrame({
            "Date": data["Date"], "signal_index": np.arange(len(data)),
            "window_start_index": np.arange(len(data)) - k + 1,
            "window_start": data["Date"].shift(k - 1), "candle_length": k,
            "delta": delta, "price_response": price,
            "normalized_delta": nd, "normalized_price": np_,
            "z_delta": zd, "z_price": zp, "delta_baseline_count": count_d,
            "price_baseline_count": count_p,
            "entry_price": data["close"], "atr": stop_atr,
        })
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def evaluate_signal_sizes(
    raw_1d: pd.DataFrame, tick_size: float, params: Mapping | None = None,
) -> pd.DataFrame:
    """Precompute BUY-short and SELL-long outcomes once per raw signal date.

    Entry reference is signal close; ATR includes the completed signal bar.
    PROFIT FIRST: once required opportunity occurs, success is locked. Continue
    only to measure maximum pre-stop opportunity; later stops cannot make it fail.
    On a stop bar, stop is checked before its favorable extreme is counted.
    This avoids assuming an unknown high/low ordering. Failure uses exact
    configured stop distance (ideal fill, ignoring gaps/costs, as requested).
    A known early stop can resolve even a tail event lacking all H future bars.
    """
    if not np.isfinite(tick_size) or tick_size <= 0:
        raise ValueError("tick_size must be positive and finite")
    cfg, data = _settings(params), _raw_bars(raw_1d)
    atr = _true_range(data).rolling(cfg["atr_period"]).mean().to_numpy()
    close, high, low = (data[c].to_numpy(dtype=float) for c in ("close", "high", "low"))
    records = []
    for i in range(len(data)):
        for direction, sign in (("BUY", -1), ("SELL", 1)):
            entry = close[i]
            distance = cfg["stop_atr_mult"] * atr[i]
            stop = entry - sign * distance
            record = {
                "signal_index": i, "absorption_direction": direction,
                "position": "SHORT" if sign == -1 else "LONG",
                "stop_loss": stop, "stop_distance_ticks": distance / tick_size,
                "outcome": "INVALID_ATR", "stop_hit": False,
                "exit_index": np.nan, "exit_date": pd.NaT, "exit_price": np.nan,
                "bars_observed": 0, "profit_bar_offset": np.nan,
                "maximum_profit_bar_offset": np.nan, "profit_hit_first": False,
                "stop_after_profit": False, "size_complete": False,
                "max_favorable_before_stop_ticks": np.nan,
                "max_favorable_ticks": np.nan, "success_size_ticks": np.nan,
                "failure_size_ticks": np.nan, "research_payoff_ticks": np.nan,
                "terminal_move_ticks": np.nan, "stop_or_horizon_pnl_ticks": np.nan,
                "stop_bar_order_unknown": False,
            }
            if not np.isfinite(distance) or distance <= 0:
                records.append(record)
                continue
            best, best_offset, profit_offset = 0.0, np.nan, np.nan
            qualified = False
            end = min(i + cfg["forward_days"], len(data) - 1)
            stopped = False
            for j in range(i + 1, end + 1):
                record["bars_observed"] += 1
                adverse_hit = high[j] >= stop if sign == -1 else low[j] <= stop
                favorable = (entry - low[j] if sign == -1 else high[j] - entry) / tick_size
                if adverse_hit:
                    # Stop ends size observation. Earlier profit locks SUCCESS;
                    # otherwise this is FAILURE. Do not read any later bars.
                    stopped = True
                    record.update(outcome="SUCCESS" if qualified else "FAILURE",
                                  stop_hit=True, stop_after_profit=qualified, size_complete=True, exit_index=j,
                                  exit_date=data.at[j, "Date"], exit_price=stop,
                                  success_size_ticks=best if qualified else np.nan,
                                  failure_size_ticks=np.nan if qualified else distance / tick_size,
                                  research_payoff_ticks=best if qualified else -distance / tick_size,
                                  stop_or_horizon_pnl_ticks=-distance / tick_size,
                                  stop_bar_order_unknown=bool(favorable > best))
                    break
                if favorable > best:
                    best, best_offset = favorable, j - i
                if not qualified and best > 0 and best >= cfg["required_move_ticks"]:
                    qualified, profit_offset = True, j - i
            record["max_favorable_before_stop_ticks"] = best
            record["profit_bar_offset"] = profit_offset
            record["maximum_profit_bar_offset"] = best_offset
            record["profit_hit_first"] = qualified
            if not stopped and end - i < cfg["forward_days"]:
                # Profit-first criterion may already be known, but the requested
                # maximum size is still incomplete. Never rank a partial maximum.
                record["outcome"] = "CENSORED"
            elif not stopped:
                terminal = sign * (close[end] - entry) / tick_size
                record.update(outcome="SUCCESS" if qualified else "UNRESOLVED", size_complete=True,
                              exit_index=end, exit_date=data.at[end, "Date"], exit_price=close[end],
                              max_favorable_ticks=best, terminal_move_ticks=terminal,
                              stop_or_horizon_pnl_ticks=terminal)
                if qualified:
                    record.update(success_size_ticks=best, research_payoff_ticks=best)
            if record["outcome"] == "SUCCESS":
                record["max_favorable_ticks"] = best
            records.append(record)
    columns = ["signal_index", "absorption_direction", "position", "stop_loss",
               "stop_distance_ticks", "outcome", "stop_hit", "exit_index", "exit_date",
               "exit_price", "bars_observed", "profit_bar_offset", "maximum_profit_bar_offset",
               "profit_hit_first", "stop_after_profit", "size_complete", "max_favorable_before_stop_ticks",
               "max_favorable_ticks", "success_size_ticks", "failure_size_ticks",
               "research_payoff_ticks", "terminal_move_ticks", "stop_or_horizon_pnl_ticks",
               "stop_bar_order_unknown"]
    return pd.DataFrame(records, columns=columns)


def select_threshold_signals(features: pd.DataFrame, delta_z: float, price_z: float) -> pd.DataFrame:
    """Apply one symmetric signed-delta threshold and one price ceiling to ALL k.

    Raw-sign guards enforce positive delta=BUY and negative delta=SELL even
    when subtracting the historical median changes a z-score's sign.
    Only signal-time fields influence selection; no outcome labels are read.
    """
    if not np.isfinite(delta_z) or delta_z <= 0 or not np.isfinite(price_z):
        raise ValueError("delta_z must be positive; both thresholds must be finite")
    finite = np.isfinite(features[["z_delta", "z_price", "atr"]]).all(axis=1)
    buy = finite & (features["atr"] > 0) & (features["delta"] > 0) & (features["z_delta"] >= delta_z)
    sell = finite & (features["atr"] > 0) & (features["delta"] < 0) & (features["z_delta"] <= -delta_z)
    muted = features["z_price"] <= price_z
    result = features.loc[(buy | sell) & muted].copy()
    result["absorption_direction"] = np.where(buy.loc[result.index], "BUY", "SELL")
    return result


def deduplicate_combined_signals(signals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep longest detection windows first; suppress ANY overlapping window.

    Operates within one instrument, across directions, using inclusive raw-bar
    intervals [window_start_index, signal_index]. Stops/profits never decide
    which signal wins. Ties in length prefer the earlier end date. Therefore
    5D > 3D > 2D > 1D when those lengths were requested. Non-overlapping windows
    remain separate events. Single-duration SEPARATE mode does not call this.

    This is retrospective research deduplication, not a causal trading policy:
    a future larger signal can supersede an earlier smaller one. Partial overlaps
    are suppressed too, a conservative rule against inflated event counts.
    """
    ordered = signals.sort_values(["candle_length", "signal_index"], ascending=[False, True])
    accepted, rejected, occupied, owners = [], [], {}, {}
    for idx, row in ordered.iterrows():
        covered = set(range(int(row["window_start_index"]), int(row["signal_index"]) + 1))
        overlaps = sorted(covered.intersection(occupied))
        if overlaps:
            rejected.append(idx)
            owners[idx] = occupied[overlaps[0]]
        else:
            accepted.append(idx)
            occupied.update({bar: idx for bar in covered})
    kept = signals.loc[accepted].sort_values("signal_index").copy()
    dropped = signals.loc[rejected].sort_values("signal_index").copy()
    dropped["suppression_reason"] = "overlap_with_longer_or_earlier_equal_window"
    for column in ("signal_index", "candle_length", "window_start_index"):
        dropped[f"suppressed_by_{column}"] = [signals.at[owners[idx], column] for idx in dropped.index]
    return kept, dropped


def summarize_expected_sizes(events: pd.DataFrame, min_signal_count: int = 8) -> dict:
    """Summarize signed requested payoffs and explicitly report their denominator.

    SUCCESS payoff is maximum opportunity; FAILURE payoff is ideal stop loss.
    UNRESOLVED/CENSORED are counted but excluded from the resolved size metric.
    Also expose stop-or-horizon P&L for completed events: unlike maxima, that
    diagnostic uses terminal closes on all no-stop events (still ideal fills).
    """
    outcomes = events["outcome"]
    successful = events.loc[outcomes == "SUCCESS", "success_size_ticks"]
    failed = events.loc[outcomes == "FAILURE", "failure_size_ticks"]
    resolved = events.loc[outcomes.isin(_RESOLVED)]
    n, nr = len(events), len(resolved)
    ns, nf = len(successful), len(failed)
    payoff = resolved["research_payoff_ticks"]
    p_success = ns / nr if nr else np.nan
    p_failure = nf / nr if nr else np.nan
    avg_success = float(successful.mean()) if ns else np.nan
    avg_failure = float(failed.mean()) if nf else np.nan
    # Empty classes have zero CONTRIBUTION, but unknown conditional mean.
    weighted = ((successful.sum() - failed.sum()) / nr) if nr else np.nan
    direct = float(payoff.mean()) if nr else np.nan
    if nr and not np.isclose(weighted, direct):
        raise ValueError("Outcome sizes do not reconcile with signed event payoffs")
    return {
        "n_signals": n, "n_resolved": nr, "n_success": ns, "n_failure": nf,
        "n_unresolved": int((outcomes == "UNRESOLVED").sum()),
        "n_censored": int((outcomes == "CENSORED").sum()),
        "n_invalid_atr": int((outcomes == "INVALID_ATR").sum()),
        "n_stop_after_profit": int(events.get("stop_after_profit", pd.Series(dtype=bool)).sum()),
        "n_stop_bar_order_unknown": int(events.get("stop_bar_order_unknown", pd.Series(dtype=bool)).sum()),
        "n_profit_first_size_censored": int(((outcomes == "CENSORED") &
            events.get("profit_hit_first", pd.Series(False, index=events.index))).sum()),
        "resolved_fraction": nr / n if n else np.nan,
        "success_rate": p_success, "failure_rate": p_failure,
        "avg_success_ticks": avg_success, "avg_failure_ticks": avg_failure,
        "median_success_ticks": float(successful.median()) if ns else np.nan,
        "median_failure_ticks": float(failed.median()) if nf else np.nan,
        "expected_profit_ticks": direct, "weighted_expected_profit_ticks": weighted,
        "total_research_payoff_ticks": float(payoff.sum()) if nr else np.nan,
        "mean_stop_or_horizon_pnl_ticks": float(events["stop_or_horizon_pnl_ticks"].mean()),
        "n_stop_or_horizon_completed": int(events["stop_or_horizon_pnl_ticks"].notna().sum()),
        "low_n": nr < min_signal_count,
    }


def analyze_expected_profit(
    raw_1d: pd.DataFrame, tick_size: float, candle_lengths: Sequence[int] | None = None,
    mode: str = "combined", delta_z_grid: Sequence[float] | None = None,
    price_z_grid: Sequence[float] | None = None, params: Mapping | None = None,
    analysis_start=None, analysis_end=None,
) -> dict:
    """Pure in-memory analysis for ONE instrument, without network or disk writes.

    `combined`: one X/Y pair selects all requested k, then longest-window-first
    dedup runs afresh for that pair. `separate`: each requested k has its own
    event table and grid without cross-duration suppression; pass [1] for 1D only.
    Return auditable event/suppression ledgers for every threshold and side-level
    summaries. No automatic 'best threshold' hides low counts or missing outcomes.
    All optional date limits filter SIGNAL END dates, retaining earlier warmup.
    """
    cfg = _settings(params)
    lengths = _lengths(cfg["candle_lengths"] if candle_lengths is None else candle_lengths)
    if mode not in ("combined", "separate"):
        raise ValueError("mode must be 'combined' or 'separate'")
    dx = list(cfg["delta_z_grid"] if delta_z_grid is None else delta_z_grid)
    py = list(cfg["price_z_grid"] if price_z_grid is None else price_z_grid)
    if not dx or not py or len(set(dx)) != len(dx) or len(set(py)) != len(py):
        raise ValueError("Threshold grids must be nonempty and unique")
    if any(not np.isfinite(x) or x <= 0 for x in dx) or any(not np.isfinite(y) for y in py):
        raise ValueError("Delta thresholds must be positive; grids must be finite")
    features = build_v2_features(raw_1d, lengths, cfg)
    outcomes = evaluate_signal_sizes(raw_1d, tick_size, cfg)
    if analysis_start is not None:
        features = features.loc[features["Date"] >= pd.to_datetime(analysis_start, utc=True)]
    if analysis_end is not None:
        features = features.loc[features["Date"] <= pd.to_datetime(analysis_end, utc=True)]
    summaries, by_side, ledgers, suppressed = [], [], {}, {}
    groups = [("combined", features)] if mode == "combined" else [
        (f"{k}d", features.loc[features["candle_length"] == k]) for k in lengths]
    for group, subset in groups:
        for x in dx:
            for y in py:
                candidates = select_threshold_signals(subset, x, y)
                if mode == "combined":
                    selected, dropped = deduplicate_combined_signals(candidates)
                else:
                    selected, dropped = candidates, candidates.iloc[:0].copy()
                # Join AFTER selection, so failures/censored windows never lose
                # priority to lower-duration windows with nicer known outcomes.
                events = selected.merge(outcomes, on=["signal_index", "absorption_direction"],
                                        how="left", validate="many_to_one")
                key = (group, float(x), float(y))
                ledgers[key], suppressed[key] = events, dropped
                info = {"analysis_group": group, "delta_z_threshold": x, "price_z_threshold": y,
                        "n_candidates": len(candidates), "n_suppressed": len(dropped)}
                summaries.append({**info, **summarize_expected_sizes(events, cfg["min_signal_count"])})
                for side in ("BUY", "SELL"):
                    side_events = events.loc[events["absorption_direction"] == side]
                    by_side.append({"analysis_group": group, "delta_z_threshold": x,
                                    "price_z_threshold": y, "absorption_direction": side,
                                    **summarize_expected_sizes(side_events, cfg["min_signal_count"])})
    return {
        "features": features.reset_index(drop=True), "directional_outcomes": outcomes,
        "threshold_table": pd.DataFrame(summaries), "side_table": pd.DataFrame(by_side),
        "event_ledgers": ledgers, "suppressed_ledgers": suppressed,
        "metadata": {"mode": mode, "candle_lengths": lengths, "params": cfg,
                     "entry_reference": "signal_close", "stop_fill": "ideal_fixed_stop",
                     "metric": "maximum_success_opportunity_minus_stopped_loss",
                     "is_realized_profit": False,
                     "deduplication": "retrospective_longest_first_any_overlap" if mode == "combined" else "none",
                     "analysis_start": str(analysis_start), "analysis_end": str(analysis_end)},
    }


def analysis_date_range(as_of=None, params: Mapping | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Calendar-month-inclusive start and end boundary excluding today's unfinished bar.

    For N=5 and any October run, start is 1 June. End is yesterday's date;
    actual completed sessions come from the supplied API data, not a guessed
    weekday calendar. `as_of` represents the run date, not an inclusive bar date.
    """
    cfg = _settings(params)
    today = pd.Timestamp.now(tz=cfg["timezone"]) if as_of is None else pd.Timestamp(as_of)
    if today.tzinfo is not None:
        today = today.tz_convert(cfg["timezone"])
    day = pd.Timestamp(today.date(), tz="UTC")
    start = day.replace(day=1) - pd.DateOffset(months=cfg["months_back"] - 1)
    return start, day - pd.Timedelta(days=1)


def refresh_expected_profit_data(
    instrument: str, product_code: str, candle_lengths: Sequence[int],
    params: Mapping | None = None, as_of=None, fetcher: Callable | None = None,
    cache_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Verify each run against fresh API data; preserve extra cache history.

    Refresh the full requested range plus warmup rather than trusting cache
    min/max dates: this also reconciles internal missing/revised API bars.
    Fresh API rows are authoritative for the requested slice. No cache deletion
    and no fallback to silently stale data if the API fails. Session completeness
    relative to an exchange calendar is NOT claimed without that calendar.
    The old fetch helper's cache early-return is deliberately not used.
    """
    from ..utils.contracts import instrument_to_api_str
    if fetcher is None:
        from ..data.ohlc import getData_OHLC_v2
        fetcher = getData_OHLC_v2
    cfg, lengths = _settings(params), _lengths(candle_lengths)
    start, end = analysis_date_range(as_of, cfg)
    # Full-baseline feature warmup also needs each historical candle's causal
    # normalization history and one pre-window close. Add holiday padding.
    warmup = cfg["baseline_window"] + 2 * max(lengths) + max(
        cfg["volume_window"], cfg["atr_window"], cfg["atr_period"])
    fetch_start = start - pd.offsets.BDay(warmup) - pd.Timedelta(days=cfg["warmup_extra_calendar_days"])
    folder = Path(DATA_DIR if cache_dir is None else cache_dir) / product_code.lower()
    cache = folder / f"{instrument}_1d.csv"
    cached = _raw_bars(pd.read_csv(cache)) if cache.exists() else None
    fetched = fetcher(instrs=instrument_to_api_str(instrument),
                      start_date=fetch_start.strftime("%Y-%m-%d"),
                      end_date=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                      timeinterval="1D")
    if fetched is None or fetched.empty:
        raise ValueError(f"{instrument}: API returned no fresh data; cache coverage unverified")
    fresh = _raw_bars(fetched)
    fresh = fresh.loc[fresh["Date"].between(fetch_start, end)].reset_index(drop=True)
    analysis = fresh.loc[fresh["Date"].between(start, end)]
    if analysis.empty:
        raise ValueError(f"{instrument}: no completed bars in requested analysis range")
    # Remove the old requested slice, then merge the authoritative replacement.
    # Rows outside that slice remain cached for other analysis windows.
    outside = cached.loc[~cached["Date"].between(fetch_start, end)] if cached is not None else None
    merged = pd.concat([outside, fresh], ignore_index=True) if outside is not None else fresh.copy()
    merged = merged.sort_values("Date").reset_index(drop=True)
    folder.mkdir(parents=True, exist_ok=True)
    merged.to_csv(cache, index=False, encoding="utf-8-sig")
    old_dates = set(cached["Date"]) if cached is not None else set()
    return fresh, {
        "cache_path": str(cache), "analysis_start": start, "analysis_end": end,
        "fetch_start": fetch_start, "actual_start": fresh["Date"].min(),
        "actual_end": fresh["Date"].max(), "analysis_bars": len(analysis),
        "warmup_bars_available": int((fresh["Date"] < start).sum()),
        "warmup_bars_requested": warmup,
        "warmup_sufficient": int((fresh["Date"] < start).sum()) >= warmup,
        "new_dates": len(set(fresh["Date"]) - old_dates),
        "refresh_policy": "full_requested_range_plus_warmup_every_run",
        "coverage_basis": "fresh_api_response; exchange_calendar_completeness_unverified",
    }


def run_expected_profit_analysis(
    product_code: str, candle_lengths: Sequence[int] | None = None,
    mode: str = "combined", params: Mapping | None = None,
    delta_z_grid: Sequence[float] | None = None, price_z_grid: Sequence[float] | None = None,
    as_of=None, front_n: int = FRONT_N, fetcher: Callable | None = None,
    cache_dir: Path | str | None = None, save_outputs: bool = False,
    instrument_names: Sequence[str] | None = None,
) -> dict:
    """Run the independent analysis for front-N spreads and jumbos of one product.

    All durations/instruments use the same supplied X/Y grid. Dedup stays within
    each instrument. Return per-instrument results plus coverage/error reports.
    No notebook changes, existing heatmaps, clean-cache flags, or runner imports.
    Optional CSV/metadata export allows reviewing this implementation first.
    """
    from ..utils.contracts import get_all_front_instruments
    cfg = _settings(params)
    lengths = _lengths(cfg["candle_lengths"] if candle_lengths is None else candle_lengths)
    if mode not in ("combined", "separate"):
        raise ValueError("mode must be 'combined' or 'separate'")
    if isinstance(front_n, bool) or int(front_n) != front_n or front_n < 1:
        raise ValueError("front_n must be a positive integer")
    if product_code not in PRODUCTS:
        raise ValueError(f"Unknown product {product_code!r}")
    start, end = analysis_date_range(as_of, cfg)
    reference_date = (end + pd.Timedelta(days=1)).date()
    universe = get_all_front_instruments(product_code, reference_date, int(front_n))
    # LLM resume can omit contracts already completed by every requested model.
    # Preserve the full universe/ranks in the returned result.
    requested = None if instrument_names is None else set(instrument_names)
    unknown = set() if requested is None else requested - {i for members in universe.values() for i in members}
    if unknown:
        raise ValueError(f"Instruments outside the front universe: {sorted(unknown)}")
    results, errors = {}, {}
    for structure, instruments in universe.items():
        for instrument in instruments:
            if requested is not None and instrument not in requested:
                continue
            try:
                data, coverage = refresh_expected_profit_data(
                    instrument, product_code, lengths, cfg, as_of, fetcher, cache_dir)
                result = analyze_expected_profit(
                    data, PRODUCTS[product_code]["tick_size"], lengths, mode,
                    delta_z_grid, price_z_grid, cfg, start, end)
                result["coverage"] = coverage
                result["metadata"].update(product=product_code, structure=structure, instrument=instrument)
                results[instrument] = result
                print(f"{instrument}: {coverage['analysis_bars']} raw bars; "
                      f"warmup sufficient={coverage['warmup_sufficient']}")
                if save_outputs:
                    _save_analysis_result(result, instrument, product_code, mode)
            except (ValueError, OSError, RuntimeError) as exc:
                errors[instrument] = str(exc)
                print(f"[SKIP] {instrument}: {exc}")
    return {"instruments": results, "errors": errors, "universe": universe,
            "analysis_start": start, "analysis_end": end, "mode": mode, "candle_lengths": lengths,
            "product_code": product_code, "params": cfg}


def pool_expected_profit_analysis(run_result: dict, structure: str | None = None) -> dict:
    """Pool actual event payoffs across instruments, keeping BUY and SELL folded.

    `structure=None` combines all front-N spreads and jumbos. Pass 'spread' or
    'jumbo' for structure-only views. Recompute means from joined event ledgers;
    never average instrument means or success rates with unequal event counts.
    Durations were already deduplicated WITHIN each instrument and threshold;
    independent instruments must not suppress one another just for sharing dates.
    Return the same plotting/table interface as single-instrument analysis.
    """
    universe = run_result["universe"]
    if structure is not None and structure not in universe:
        raise ValueError(f"Unknown structure {structure!r}; choose from {list(universe)}")
    expected = [inst for name, instruments in universe.items()
                if structure is None or name == structure for inst in instruments]
    members = {inst: result for inst, result in run_result["instruments"].items() if inst in expected}
    if not members:
        raise ValueError("No successfully analyzed instruments available to pool")
    first = next(iter(members.values()))
    keys = list(first["event_ledgers"])
    cfg = first["metadata"]["params"]
    for result in members.values():
        if set(result["event_ledgers"]) != set(keys):
            raise ValueError("Cannot pool instruments with different threshold grids/groups")
        if result["metadata"]["params"] != cfg:
            raise ValueError("Cannot pool instruments with different effective settings")
    tables, side_tables, ledgers, suppressed = [], [], {}, {}
    for key in keys:
        group, x, y = key
        event_frames, suppression_frames, candidate_count = [], [], 0
        for inst, result in members.items():
            # Keep provenance so a pooled cell can be traced to its instruments.
            name = result["metadata"]["structure"]
            event_frames.append(result["event_ledgers"][key].assign(instrument=inst, structure=name))
            suppression_frames.append(result["suppressed_ledgers"][key].assign(instrument=inst, structure=name))
            table = result["threshold_table"]
            row = table.loc[(table["analysis_group"] == group) &
                            (table["delta_z_threshold"] == x) & (table["price_z_threshold"] == y)].iloc[0]
            candidate_count += int(row["n_candidates"])
        events = pd.concat(event_frames, ignore_index=True)
        dropped = pd.concat(suppression_frames, ignore_index=True)
        ledgers[key], suppressed[key] = events, dropped
        info = {"analysis_group": group, "delta_z_threshold": x, "price_z_threshold": y,
                "n_candidates": candidate_count, "n_suppressed": len(dropped),
                "n_instruments": len(members)}
        tables.append({**info, **summarize_expected_sizes(events, cfg["min_signal_count"])})
        for side in ("BUY", "SELL"):
            side_tables.append({**info, "absorption_direction": side,
                                **summarize_expected_sizes(events.loc[events["absorption_direction"] == side],
                                                           cfg["min_signal_count"])})
    scope = "spreads + jumbos" if structure is None else structure
    product = run_result.get("product_code", first["metadata"].get("product", ""))
    return {
        "threshold_table": pd.DataFrame(tables), "side_table": pd.DataFrame(side_tables),
        "event_ledgers": ledgers, "suppressed_ledgers": suppressed,
        "metadata": {**first["metadata"], "instrument": None, "structure": structure,
                     "pooled": True, "instruments": list(members),
                     "expected_instruments": expected,
                     "missing_instruments": [inst for inst in expected if inst not in members],
                     "plot_title": f"{product} | {scope} | {len(members)}/{len(expected)} instruments"},
    }


def _save_analysis_result(result: dict, instrument: str, product_code: str, mode: str) -> None:
    """Export auditable event/suppression ledgers, grid tables, and effective metadata."""
    import json
    lengths = "_".join(map(str, result["metadata"]["candle_lengths"]))
    folder = OUTPUT_DIR / product_code.lower() / "expected_profit_analysis" / f"{mode}_{lengths}" / instrument
    folder.mkdir(parents=True, exist_ok=True)
    for key in ("features", "directional_outcomes", "threshold_table", "side_table"):
        result[key].to_csv(folder / f"{key}.csv", index=False, encoding="utf-8-sig")
    for category in ("event_ledgers", "suppressed_ledgers"):
        for (group, x, y), frame in result[category].items():
            frame.to_csv(folder / f"{category}_{group}_dz{x:g}_pz{y:g}.csv", index=False, encoding="utf-8-sig")
    (folder / "metadata.json").write_text(
        json.dumps({"experiment": result["metadata"], "coverage": result["coverage"]},
                   default=str, indent=2), encoding="utf-8")


def plot_expected_profit_heatmap(result: dict, analysis_group: str = "combined", ax=None):
    """Optional new heatmap of requested expected sizes; old heatmaps stay intact.

    Cell annotations expose resolved/selected counts. Color is hindsight size
    expectancy, not success rate or realized P&L. Import plotting only on demand.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    table = result["threshold_table"]
    subset = table.loc[table["analysis_group"] == analysis_group]
    if subset.empty:
        raise ValueError(f"No analysis group {analysis_group!r}")
    xs = list(dict.fromkeys(subset["delta_z_threshold"]))
    ys = list(dict.fromkeys(subset["price_z_threshold"]))
    matrix = subset.pivot(index="price_z_threshold", columns="delta_z_threshold",
                          values="expected_profit_ticks").reindex(index=ys, columns=xs)
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, len(xs)), max(4, len(ys) * 0.65)))
    values = matrix.to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    limit = max(float(np.abs(finite_values).max()), 1e-9) if finite_values.size else 1.0
    # Zero is always neutral, losses red and gains green, even for one-sign grids.
    image = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap="RdYlGn",
                      norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
    ax.set_xticks(range(len(xs)), [f"{x:g}" for x in xs])
    ax.set_yticks(range(len(ys)), [f"{y:g}" for y in ys])
    indexed = subset.set_index(["price_z_threshold", "delta_z_threshold"])
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            row = indexed.loc[(y, x)]
            value = row["expected_profit_ticks"]
            label = f"{value:.2f}" if np.isfinite(value) else "N/A"
            marker = "*" if row["low_n"] else ""
            color = "white" if np.isfinite(value) and abs(value) > 0.65 * limit else "black"
            ax.text(j, i, f"{label}{marker} ticks\nN:{int(row['n_signals'])}  R:{int(row['n_resolved'])}",
                    ha="center", va="center", fontsize=8, color=color)
    metadata = result.get("metadata", {})
    instrument_title = " | ".join(str(metadata[key]) for key in ("product", "structure", "instrument")
                                   if metadata.get(key))
    title = metadata.get("plot_title", instrument_title or "V2 absorption")
    ax.set(xlabel="Delta z threshold X: BUY >= +X; SELL <= -X",
           ylabel="Price-response z ceiling Y: z_price <= Y",
           title=f"{title}\n{analysis_group}: expected size | N=selected; R=resolved; * low R")
    ax.figure.colorbar(image, ax=ax, label="Expected success/failure size (ticks)")
    ax.figure.tight_layout()
    return ax.figure
