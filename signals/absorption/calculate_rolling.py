# market_analysis/signals/absorption/calculate_rolling.py
"""
Rolling N-day candle absorption scorer.

The Problem with naive aggregation
------------------------------------
The existing ohlc.py already builds rolling candles via _aggregate_trailing():

    3D candle on Wednesday  =  open(Mon) | high(max Mon-Wed) | low(min Mon-Wed)
                                | close(Wed) | volume(sum Mon-Wed)

But when calculate_absorption() receives that DataFrame and computes rolling
z-scores / percentile baselines, it uses the *last* bar of the window (Wednesday)
as its reference point.  The causal baseline then only guarantees "no look-ahead
beyond Wednesday", but it happily uses Tuesday's and Monday's values — which are
part of the candle itself — as part of the rolling statistics.

The correct rule:
    "When I am on Wednesday and analysing its 3D candle, I should only use
     data *before Monday* to build any statistical baseline."

This file implements that rule via a two-phase approach:

    Phase 1  — build rolling candles
        Identical to _aggregate_trailing() in data/ohlc.py.
        Each output row is tagged with both:
            date_close  — last bar's date (the output row's timestamp)
            date_open   — first bar's date in the window (the lookback cutoff)

    Phase 2  — score on 1D bars, aligned by date_open
        For every combined candle we want to score, we find the correct
        *previous* 1D bar (the bar before date_open) and use the
        population of 1D bars up to and including that bar as the baseline.

        Why 1D bars for the baseline?
            A 3D candle compresses three days of information into one row.
            Using other 3D candles in the baseline would double- or
            triple-count overlapping days.  Baselines built on the
            *underlying* 1D bars are stationary, non-overlapping, and
            scale correctly regardless of window size.

Public API
----------
    calculate_absorption_rolling(df_1d, combined_candle=1, method="v2", ...)
        df_1d           : raw 1D OHLC DataFrame (the base bars)
        combined_candle : 1 | 2 | 3 | 5  (N-day rolling window)
        method          : "v1" = percentile scorer (calculate.py)
                          "v2" = z-score scorer   (calculate_v2.py)  [default]

    drop-in replacement for calculate_absorption / calculate_absorption_v2
    when you want multi-day candles with a correctly anchored baseline.

Column conventions mirror calculate_v2.py so label.py and label_hourly.py
work without modification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-9

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — ROLLING CANDLE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_rolling_candles(df_1d: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Build rolling n-day OHLC candles from a sorted 1D DataFrame.

    Each output row covers bars [i-n+1 .. i]:
        open        = bar[i-n+1].open
        high        = max(bar[i-n+1 .. i].high)
        low         = min(bar[i-n+1 .. i].low)
        close       = bar[i].close
        volume      = sum(bar[i-n+1 .. i].volume)
        buyvolume   = sum(...)
        sellvolume  = sum(...)
        Date        = bar[i].Date         ← label the row by its LAST bar
        date_open   = bar[i-n+1].Date     ← first bar in the window

    date_open is the key for causal baseline alignment in Phase 2.
    """
    df = df_1d.copy().reset_index(drop=True)

    rows: list[dict] = []
    for i in range(n - 1, len(df)):
        window = df.iloc[i - n + 1 : i + 1]
        _do_raw = df.at[i - n + 1, "Date"]
        _do_ts  = pd.Timestamp(_do_raw)
        _do_utc = _do_ts if _do_ts.tzinfo is not None else _do_ts.tz_localize("UTC")
        rows.append({
            "Date"       : df.at[i, "Date"],
            "date_open"  : _do_utc,   # ← cutoff anchor, always UTC Timestamp
            "open"       : float(window["open"].iloc[0]),
            "high"       : float(window["high"].max()),
            "low"        : float(window["low"].min()),
            "close"      : float(window["close"].iloc[-1]),
            "volume"     : float(window["volume"].sum()),
            "buyvolume"  : float(window["buyvolume"].sum()),
            "sellvolume" : float(window["sellvolume"].sum()),
        })

    return pd.DataFrame(rows).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2a — CAUSAL BASELINE HELPERS (shared by v1 and v2 scorers)
# ─────────────────────────────────────────────────────────────────────────────

def _build_1d_baselines_v1(df_1d: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Pre-compute 1D percentile baselines for the v1 (percentile) scorer.

    For each 1D bar at position i we store:
        abs_delta_sorted     — sorted history of |delta| from bars [i-window .. i-1]
        price_range_sorted   — same for price range
        n_valid              — number of usable bars in that history

    These are later retrieved by date to answer:
        "what percentile is this candle's abs_delta / price_range
         relative to the 1D bars that came BEFORE its first bar?"

    Returns a dict keyed by Date string → (abs_delta_hist, price_range_hist).
    """
    df = df_1d.copy().reset_index(drop=True)
    df["_abs_delta"]    = (df["buyvolume"] - df["sellvolume"]).abs()
    df["_price_range"]  = (df["high"] - df["low"]).abs()
    dates = df["Date"].astype(str).tolist()

    baselines: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    abs_d_arr    = df["_abs_delta"].to_numpy(float)
    price_r_arr  = df["_price_range"].to_numpy(float)

    for i, date in enumerate(dates):
        start = max(0, i - window)
        # history = bars [start .. i-1] — strictly before bar i
        hist_abs   = abs_d_arr[start : i]
        hist_range = price_r_arr[start : i]
        hist_abs   = hist_abs[np.isfinite(hist_abs)]
        hist_range = hist_range[np.isfinite(hist_range)]
        baselines[date] = (
            np.sort(hist_abs),
            np.sort(hist_range),
        )
    return baselines


def _percentile_of(value: float, sorted_hist: np.ndarray) -> float:
    """Fraction of sorted_hist values <= value."""
    if len(sorted_hist) == 0 or not np.isfinite(value):
        return np.nan
    return float(np.searchsorted(sorted_hist, value, side="right")) / len(sorted_hist)


def _build_1d_baselines_v2(
    df_1d: pd.DataFrame,
    volume_window: int,
    atr_window: int,
    baseline_window: int,
    min_periods: int,
) -> dict[str, dict]:
    """
    Pre-compute 1D z-score baselines for the v2 scorer.

    For each 1D bar i, store everything needed to compute z_delta and z_price
    for an arbitrary (combined) candle whose window *starts* at bar i.

    Stored per date:
        norm_delta_med, norm_delta_mad   — for z_delta
        norm_price_med, norm_price_mad   — for z_price
        avg_vol                          — rolling avg volume (for normalising delta)
        atr                              — rolling ATR        (for normalising price)

    All rolling windows look back from bar i-1 (strictly causal).
    """
    df = df_1d.copy().reset_index(drop=True)

    buy_v   = pd.to_numeric(df["buyvolume"],  errors="coerce")
    sell_v  = pd.to_numeric(df["sellvolume"], errors="coerce")
    volume  = pd.to_numeric(df["volume"],     errors="coerce")
    high    = pd.to_numeric(df["high"],       errors="coerce")
    low     = pd.to_numeric(df["low"],        errors="coerce")
    close   = pd.to_numeric(df["close"],      errors="coerce")

    # ── normalised delta (per 1D bar) ────────────────────────────────────────
    delta_1d = buy_v - sell_v
    avg_vol  = (
        volume.shift(1)
        .rolling(volume_window, min_periods=max(3, volume_window // 2))
        .mean()
    )
    norm_delta_1d = delta_1d / (avg_vol + _EPS)

    # ── normalised price response (per 1D bar) ────────────────────────────────
    prev_close    = close.shift(1)
    price_resp_1d = (close - prev_close).abs()
    tr            = pd.concat(
        [
            high - low,
            (high  - prev_close).abs(),
            (low   - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_1d = (
        tr.shift(1)
        .rolling(atr_window, min_periods=max(3, atr_window // 2))
        .mean()
    )
    norm_price_1d = price_resp_1d / (atr_1d + _EPS)

    # ── rolling median / MAD (causal: shift-1 so bar i only sees 0..i-1) ────
    def _rolling_med_mad(s: pd.Series) -> tuple[pd.Series, pd.Series]:
        prev = s.shift(1)
        roll = prev.rolling(baseline_window, min_periods=min_periods)
        med  = roll.median()
        def _mad(x: np.ndarray) -> float:
            m = np.median(x)
            return float(np.median(np.abs(x - m)))
        mad = roll.apply(_mad, raw=True)
        return med, mad

    nd_med, nd_mad = _rolling_med_mad(norm_delta_1d)
    np_med, np_mad = _rolling_med_mad(norm_price_1d)

    dates = df["Date"].astype(str).tolist()

    baselines: dict[str, dict] = {}
    for i, date in enumerate(dates):
        baselines[date] = {
            "avg_vol"       : float(avg_vol.iloc[i])   if pd.notna(avg_vol.iloc[i])  else np.nan,
            "atr"           : float(atr_1d.iloc[i])    if pd.notna(atr_1d.iloc[i])   else np.nan,
            "nd_med"        : float(nd_med.iloc[i])    if pd.notna(nd_med.iloc[i])   else np.nan,
            "nd_mad"        : float(nd_mad.iloc[i])    if pd.notna(nd_mad.iloc[i])   else np.nan,
            "np_med"        : float(np_med.iloc[i])    if pd.notna(np_med.iloc[i])   else np.nan,
            "np_mad"        : float(np_mad.iloc[i])    if pd.notna(np_mad.iloc[i])   else np.nan,
        }
    return baselines


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2b — ROW-LEVEL SCORING
# ─────────────────────────────────────────────────────────────────────────────

def _score_row_v1(
    row: pd.Series,
    baselines: dict,
    window: int,
    tick_size: float,
    strength_threshold: float,
) -> dict:
    """
    Score one combined candle using v1 (percentile) logic.

    Baseline is looked up by date_open (the first bar of the window),
    which guarantees only data before that date was used.
    """
    try:
        date_open_str = pd.Timestamp(row["date_open"]).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    except Exception:
        date_open_str = str(row["date_open"])
    baseline      = baselines.get(date_open_str)

    abs_delta    = abs(float(row["buyvolume"]) - float(row["sellvolume"]))
    price_range  = abs(float(row["high"]) - float(row["low"]))
    delta_val    = float(row["buyvolume"]) - float(row["sellvolume"])
    total_vol    = float(row["volume"])

    out = {
        "delta"           : delta_val,
        "delta_pct"       : delta_val / total_vol * 100 if total_vol > 0 else np.nan,
        "price_range"     : price_range,
        "price_response"  : price_range,
        "price_response_ticks": price_range / tick_size,
        "buy_volume"      : float(row["buyvolume"]),
        "sell_volume"     : float(row["sellvolume"]),
        "total_volume"    : total_vol,
    }

    direction = "BUY" if delta_val > 0 else ("SELL" if delta_val < 0 else "NEUTRAL")
    out["absorption_direction"] = direction

    if baseline is None or len(baseline[0]) < max(4, window // 4):
        out.update({
            "aggression_score"        : np.nan,
            "price_range_percentile"  : np.nan,
            "price_response_score"    : np.nan,
            "absorption_strength"     : np.nan,
            "absorption_class"        : "NONE",
            "absorption_score"        : np.nan,
            "absorption_raw"          : np.nan,
        })
        return out

    abs_d_hist, price_r_hist = baseline
    aggression_score  = _percentile_of(abs_delta, abs_d_hist)
    range_pct         = _percentile_of(price_range, price_r_hist)
    price_resp_score  = 1.0 - range_pct if np.isfinite(range_pct) else np.nan

    strength = (
        aggression_score * price_resp_score
        if np.isfinite(aggression_score) and np.isfinite(price_resp_score)
        else np.nan
    )

    def _classify(s: float) -> str:
        if not np.isfinite(s): return "NONE"
        if s < 0.25: return "NONE"
        if s < 0.40: return "POSSIBLE"
        if s < 0.60: return "MODERATE"
        if s < 0.80: return "STRONG"
        return "EXTREME"

    out.update({
        "aggression_score"       : aggression_score,
        "price_range_percentile" : range_pct,
        "price_response_score"   : price_resp_score,
        "absorption_strength"    : strength,
        "absorption_class"       : _classify(strength if strength is not None else np.nan),
        # v2-compatible aliases (NaN for v1 — avoids column mismatch downstream)
        "absorption_score"       : strength,
        "absorption_raw"         : np.nan,
        "z_delta"                : np.nan,
        "z_price"                : np.nan,
        "delta_strength"         : np.nan,
        "price_suppression"      : np.nan,
    })
    return out


def _score_row_v2(
    row: pd.Series,
    baselines: dict,
    delta_z_threshold: float,
    price_z_threshold: float,
) -> dict:
    """
    Score one combined candle using v2 (z-score) logic.

    Baseline is looked up by date_open so only data before the candle's
    first constituent bar feeds the statistics.

    The combined candle's delta and price_response are normalised the same
    way as in calculate_v2.py:
        norm_delta  = (buy - sell) / avg_vol_before_candle_open
        norm_price  = |close - prev_close_of_first_bar| / ATR_before_candle_open

    "prev_close of first bar" is not directly available in the combined row,
    so we use |high - low| / ATR as the price normalisation — this is the
    intra-candle range normalised by the ATR baseline, which is appropriate
    because the combined candle's open is the first bar's open and its close
    is the last bar's close; the net move is close - open.
    We therefore use abs(close - open) as the price response for combined
    candles (net directional move over the N-day window), which is more
    meaningful than the bar's own high-low range.
    """
    # Normalise to the same ISO key format used when building baselines:
    # "YYYY-MM-DDTHH:MM:SS+00:00"  (str() gives spaces, not T-separator)
    try:
        date_open_str = pd.Timestamp(row["date_open"]).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    except Exception:
        date_open_str = str(row["date_open"])
    b             = baselines.get(date_open_str)

    buy_v   = float(row["buyvolume"])
    sell_v  = float(row["sellvolume"])
    close_  = float(row["close"])
    open_   = float(row["open"])
    delta   = buy_v - sell_v

    out = {"delta": delta}

    if b is None:
        out.update({
            "z_delta"              : np.nan,
            "z_price"              : np.nan,
            "delta_strength"       : np.nan,
            "price_suppression"    : np.nan,
            "absorption_raw"       : np.nan,
            "absorption_score"     : np.nan,
            "absorption_direction" : "NEUTRAL",
            # v1-compatible aliases
            "absorption_strength"  : np.nan,
        })
        return out

    avg_vol = b["avg_vol"]
    atr     = b["atr"]
    nd_med  = b["nd_med"]
    nd_mad  = b["nd_mad"]
    np_med  = b["np_med"]
    np_mad  = b["np_mad"]

    # ── normalised delta ───────────────────────────────────────────────────────
    norm_delta = delta / (avg_vol + _EPS)

    # ── normalised price response for combined candle ─────────────────────────
    # Net move: |close - open| / ATR  (the directional move over the N-day window)
    net_move   = abs(close_ - open_)
    norm_price = net_move / (atr + _EPS)

    # ── robust z-scores ────────────────────────────────────────────────────────
    def _rz(x: float, med: float, mad: float) -> float:
        if not np.isfinite(med) or not np.isfinite(mad):
            return np.nan
        return (x - med) / (1.4826 * mad + _EPS)

    z_delta = _rz(norm_delta, nd_med, nd_mad) if np.isfinite(norm_delta) else np.nan
    z_price = _rz(norm_price, np_med, np_mad) if np.isfinite(norm_price) else np.nan

    # ── direction + signal conditions ─────────────────────────────────────────
    if np.isfinite(z_delta) and np.isfinite(z_price):
        buy_signal  = (z_delta >=  delta_z_threshold) and (z_price <= price_z_threshold)
        sell_signal = (z_delta <= -delta_z_threshold) and (z_price <= price_z_threshold)
    else:
        buy_signal = sell_signal = False

    if buy_signal:
        direction = "BUY"
    elif sell_signal:
        direction = "SELL"
    else:
        direction = "NEUTRAL"

    # ── continuous score ───────────────────────────────────────────────────────
    if np.isfinite(z_delta) and np.isfinite(z_price):
        delta_strength    = 1.0 / (1.0 + np.exp(-abs(z_delta)))
        price_suppression = 1.0 / (1.0 + np.exp(z_price))
        abs_score         = delta_strength * price_suppression
        # suppress score if |z_delta| < 1 (small delta not meaningful)
        absorption_score  = abs_score if abs(z_delta) >= 1.0 else 0.0
        absorption_raw    = abs(z_delta) * price_suppression
    else:
        delta_strength = price_suppression = absorption_score = absorption_raw = np.nan

    out.update({
        "z_delta"              : z_delta,
        "z_price"              : z_price,
        "delta_strength"       : delta_strength,
        "price_suppression"    : price_suppression,
        "absorption_raw"       : absorption_raw,
        "absorption_score"     : absorption_score,
        "absorption_direction" : direction,
        # v1-compatible alias — downstream label.py may look for absorption_strength
        "absorption_strength"  : absorption_score,
    })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def calculate_absorption_rolling(
    df_1d: pd.DataFrame,
    combined_candle: int = 1,
    method: str = "v2",
    # ── v1 params ──────────────────────────────────────────────────────────────
    window: int = 60,
    tick_size: float = 0.1,
    strength_threshold: float = 0.60,
    # ── v2 params ──────────────────────────────────────────────────────────────
    volume_window: int = 5,
    atr_window: int = 5,
    baseline_window: int = 15,
    min_periods: int | None = None,
    delta_z_threshold: float = 2.0,
    price_z_threshold: float = 0.75,
) -> pd.DataFrame:
    """
    Build rolling N-day candles and score them for absorption with a correctly
    anchored, look-ahead-free baseline.

    Parameters
    ----------
    df_1d            : Raw 1D OHLC DataFrame.  Must contain:
                       Date, open, high, low, close, volume, buyvolume, sellvolume.
    combined_candle  : Number of days to roll up into each candle.
                       1 = standard 1D bars (identical to existing behaviour).
                       2 / 3 / 5 = rolling 2/3/5-day candles.
    method           : "v1" — percentile scorer (matches calculate.py output columns)
                       "v2" — z-score scorer   (matches calculate_v2.py output columns)
                              [default; recommended for robustness]

    v1-specific
    -----------
    window               : Percentile lookback in 1D bars (default 60).
    tick_size            : Product tick size for price_response_ticks (default 0.1).
    strength_threshold   : Threshold used to compute absorption_volume_pct.

    v2-specific
    -----------
    volume_window        : Rolling avg-volume lookback (default 5).
    atr_window           : ATR lookback (default 5).
    baseline_window      : Median/MAD baseline window (default 15).
    min_periods          : Min observations for rolling stats.
    delta_z_threshold    : z_delta must exceed ±X to qualify (default 2.0).
    price_z_threshold    : z_price must be ≤ Y to qualify (default 0.75).

    Returns
    -------
    DataFrame with one row per combined candle (indexed by the candle's last
    bar Date), including all original OHLC columns plus absorption signal
    columns.  'date_open' column is retained for auditability.

    The output is directly compatible with label.py and label_hourly.py:
    those functions use absorption_direction and close/high/low columns,
    which are all present and correctly populated.

    Example
    -------
    >>> from market_analysis.signals.absorption.calculate_rolling import (
    ...     calculate_absorption_rolling
    ... )
    >>> scored_3d = calculate_absorption_rolling(df_1d, combined_candle=3, method="v2")
    >>> labeled_3d = label_absorption(scored_3d, tick=0.1, ...)
    """
    if combined_candle < 1:
        raise ValueError(f"combined_candle must be >= 1, got {combined_candle}")
    if method not in ("v1", "v2"):
        raise ValueError(f"method must be 'v1' or 'v2', got '{method}'")
    if min_periods is None:
        min_periods = max(8, baseline_window // 2)

    # ── normalise input ────────────────────────────────────────────────────────
    df = df_1d.copy()
    df.columns = [c.lower() for c in df.columns]
    if "date" in df.columns:
        df = df.rename(columns={"date": "Date"})
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)

    for col in ("open", "high", "low", "close", "volume", "buyvolume", "sellvolume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required = {"open", "high", "low", "close", "volume", "buyvolume", "sellvolume"}
    missing  = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"calculate_absorption_rolling: missing columns {missing}")

    # ── phase 1: build rolling candles ────────────────────────────────────────
    if combined_candle == 1:
        # No aggregation needed — work on raw 1D bars.
        # Still need date_open column for baseline lookup.
        df_combined = df.copy()
        df_combined["date_open"] = df_combined["Date"]
    else:
        df_combined = _build_rolling_candles(df, combined_candle)

    # Ensure date_open is a string for dict-key lookup
    df_combined["_date_open_str"] = pd.to_datetime(
        df_combined["date_open"], utc=True, errors="coerce"
    ).dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    # Also store 1D date_open as ISO string matching the 1D dates
    df["_date_str"] = df["Date"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    # ── phase 2: score ─────────────────────────────────────────────────────────
    if method == "v1":
        # Build baselines keyed by 1D date ISO string
        baselines_1d = _build_1d_baselines_v1(
            df.rename(columns={"_date_str": "_key"}),
            window=window,
        )
        # Re-key by ISO string for matching
        baselines = {}
        for i, row in df.iterrows():
            key = row["Date"].strftime("%Y-%m-%dT%H:%M:%S+00:00")
            # store with raw Date as well as string
            baselines[key] = baselines_1d.get(str(row["Date"].date()), (np.array([]), np.array([])))

        # Rebuild properly keyed by ISO datetime string
        baselines_keyed: dict = {}
        abs_d_arr   = df["buyvolume"].sub(df["sellvolume"]).abs().to_numpy(float)
        price_r_arr = df["high"].sub(df["low"]).abs().to_numpy(float)

        for i in range(len(df)):
            key   = df.at[i, "Date"].strftime("%Y-%m-%dT%H:%M:%S+00:00")
            start = max(0, i - window)
            ha    = abs_d_arr[start:i];  ha = ha[np.isfinite(ha)]
            hr    = price_r_arr[start:i]; hr = hr[np.isfinite(hr)]
            baselines_keyed[key] = (np.sort(ha), np.sort(hr))

        score_rows = []
        for _, row in df_combined.iterrows():
            scored = _score_row_v1(
                row, baselines_keyed,
                window=window, tick_size=tick_size,
                strength_threshold=strength_threshold,
            )
            score_rows.append(scored)

    else:  # v2
        baselines_1d_v2 = _build_1d_baselines_v2(
            df,
            volume_window=volume_window,
            atr_window=atr_window,
            baseline_window=baseline_window,
            min_periods=min_periods,
        )
        # Re-key by ISO datetime string
        baselines_keyed_v2: dict = {}
        for i, date_val in enumerate(df["Date"]):
            key = date_val.strftime("%Y-%m-%dT%H:%M:%S+00:00")
            baselines_keyed_v2[key] = baselines_1d_v2.get(
                str(date_val.date()), {}
            )

        # Rebuild: key by ISO string from df directly
        baselines_keyed_v2 = {}
        for i in range(len(df)):
            key = df.at[i, "Date"].strftime("%Y-%m-%dT%H:%M:%S+00:00")
            baselines_keyed_v2[key] = {
                "avg_vol" : np.nan,
                "atr"     : np.nan,
                "nd_med"  : np.nan,
                "nd_mad"  : np.nan,
                "np_med"  : np.nan,
                "np_mad"  : np.nan,
            }

        # Build the full v2 baselines properly
        buy_v   = df["buyvolume"]
        sell_v  = df["sellvolume"]
        volume  = df["volume"]
        high    = df["high"]
        low     = df["low"]
        close   = df["close"]

        avg_vol_series = (
            volume.shift(1)
            .rolling(volume_window, min_periods=max(3, volume_window // 2))
            .mean()
        )
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr_series = (
            tr.shift(1)
            .rolling(atr_window, min_periods=max(3, atr_window // 2))
            .mean()
        )

        delta_1d       = buy_v - sell_v
        norm_delta_1d  = delta_1d / (avg_vol_series + _EPS)
        price_resp_1d  = (close - prev_close).abs()
        norm_price_1d  = price_resp_1d / (atr_series + _EPS)

        def _roll_med_mad(s: pd.Series):
            prev = s.shift(1)
            roll = prev.rolling(baseline_window, min_periods=min_periods)
            med  = roll.median()
            def _mad(x):
                m = np.median(x)
                return float(np.median(np.abs(x - m)))
            mad = roll.apply(_mad, raw=True)
            return med, mad

        nd_med_s, nd_mad_s = _roll_med_mad(norm_delta_1d)
        np_med_s, np_mad_s = _roll_med_mad(norm_price_1d)

        for i in range(len(df)):
            key = df.at[i, "Date"].strftime("%Y-%m-%dT%H:%M:%S+00:00")
            baselines_keyed_v2[key] = {
                "avg_vol" : float(avg_vol_series.iloc[i]) if pd.notna(avg_vol_series.iloc[i]) else np.nan,
                "atr"     : float(atr_series.iloc[i])     if pd.notna(atr_series.iloc[i])     else np.nan,
                "nd_med"  : float(nd_med_s.iloc[i])       if pd.notna(nd_med_s.iloc[i])       else np.nan,
                "nd_mad"  : float(nd_mad_s.iloc[i])       if pd.notna(nd_mad_s.iloc[i])       else np.nan,
                "np_med"  : float(np_med_s.iloc[i])       if pd.notna(np_med_s.iloc[i])       else np.nan,
                "np_mad"  : float(np_mad_s.iloc[i])       if pd.notna(np_mad_s.iloc[i])       else np.nan,
            }

        score_rows = []
        for _, row in df_combined.iterrows():
            scored = _score_row_v2(
                row, baselines_keyed_v2,
                delta_z_threshold=delta_z_threshold,
                price_z_threshold=price_z_threshold,
            )
            score_rows.append(scored)

    # ── assemble output ────────────────────────────────────────────────────────
    score_df = pd.DataFrame(score_rows)
    result   = df_combined.drop(columns=["_date_open_str"], errors="ignore").copy()

    for col in score_df.columns:
        result[col] = score_df[col].values

    # ── absorption_volume_pct (v1 convention) ──────────────────────────────────
    if method == "v1" and "absorption_strength" in result.columns:
        result["absorption_volume"] = np.where(
            result["absorption_strength"].fillna(0) >= strength_threshold,
            result["volume"],
            0.0,
        )
        total_vol = float(result["volume"].sum())
        abs_vol   = float(result["absorption_volume"].sum())
        result["absorption_volume_pct"] = (
            (abs_vol / total_vol * 100) if total_vol > 0 else np.nan
        )

    # final sort
    result = result.sort_values("Date").reset_index(drop=True)
    return result