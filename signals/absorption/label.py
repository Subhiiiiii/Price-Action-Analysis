# market_analysis/signals/absorption/label.py
"""
Stop-loss-aware outcome labeling for absorption signals.

Takes the output of calculate_absorption() and adds:
  stop_loss  : computed stop level (NaN = censored)
  stop_hit   : 1 = stop triggered, 0 = clean, NaN = censored
  sl_success : 1 = success (move >= required AND stop not hit)
               0 = failure
               NaN = censored (excluded from all metric calculations)

Functions
---------
label_absorption        -- original tick-based stop-loss labeler (unchanged)
label_absorption_atr    -- ATR-based labeler with hourly same-bar disambiguation
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from ...config import ABSORPTION


# =============================================================================
# ORIGINAL LABELER -- do NOT modify
# =============================================================================

def label_absorption(
    df: pd.DataFrame,
    tick:                float | None = None,
    required_move_ticks: int   | None = None,
    sl_lookback:         int   | None = None,
    forward_days:        int   | None = None,
) -> pd.DataFrame:
    """
    Label each row with a stop-loss-aware success flag.

    SELL direction (expect price UP):
        stop_loss = min(Low [t-sl_lookback+1 .. t]) - tick
        success   = max(Close[t+1..t+k] - Close[t]) > required_move_ticks x tick
                    AND min(Low[t+1..t+k]) > stop_loss

    BUY direction (expect price DOWN):
        stop_loss = max(High[t-sl_lookback+1 .. t]) + tick
        success   = min(Close[t+1..t+k] - Close[t]) < -(required_move_ticks x tick)
                    AND max(High[t+1..t+k]) < stop_loss

    Parameters fall back to config.ABSORPTION values when None.
    """
    tick     = tick                 if tick                 is not None else ABSORPTION["tick_size"] if "tick_size" in ABSORPTION else 0.1
    req_move = (required_move_ticks if required_move_ticks  is not None else ABSORPTION["required_move_ticks"]) * tick
    sl_lb    = sl_lookback          if sl_lookback           is not None else ABSORPTION["sl_lookback"]
    k        = forward_days         if forward_days           is not None else ABSORPTION["forward_days"]

    df = df.copy().sort_values("Date").reset_index(drop=True)
    n  = len(df)

    def _get(name: str) -> np.ndarray:
        for c in df.columns:
            if c.lower() == name.lower():
                return df[c].values
        raise KeyError(f"Column '{name}' not found. Available: {list(df.columns)}")

    close_ = _get("close")
    high_  = _get("high")
    low_   = _get("low")
    dir_   = _get("absorption_direction")   # added by calculate_absorption

    stop_loss_out = np.full(n, np.nan)
    stop_hit_out  = np.full(n, np.nan)
    success_out   = np.full(n, np.nan)

    for i in range(n):
        past_start = i - sl_lb + 1
        if past_start < 0 or (i + k) >= n:
            continue

        direction = str(dir_[i]).upper()
        c0        = close_[i]

        if direction == "SELL":
            sl       = float(low_[past_start : i + 1].min()) - tick
            stop_hit = bool(low_[i+1 : i+k+1].min()        <= sl)
            move_ok  = bool((close_[i+1:i+k+1] - c0).max()  >  req_move)
        elif direction == "BUY":
            sl       = float(high_[past_start : i + 1].max()) + tick
            stop_hit = bool(high_[i+1 : i+k+1].max()       >= sl)
            move_ok  = bool((close_[i+1:i+k+1] - c0).min()  < -req_move)
        else:
            continue   # NEUTRAL -- skip

        stop_loss_out[i] = round(sl, 8)
        stop_hit_out[i]  = int(stop_hit)
        success_out[i]   = 0.0 if stop_hit else (1.0 if move_ok else 0.0)

    df["stop_loss"]  = stop_loss_out
    df["stop_hit"]   = stop_hit_out
    df["sl_success"] = success_out
    return df


# =============================================================================
# ATR LABELER WITH HOURLY SAME-BAR DISAMBIGUATION
# =============================================================================

def _resolve_ambiguity_hourly(
    ambiguous_date: str,
    direction: str,
    target: float,
    stop: float,
    hourly_df: "pd.DataFrame | None",
    hourly_fetcher,
) -> str:
    """
    Resolve a same-bar ambiguity using hourly OHLC for ONE calendar date only.

    Called only when a future daily bar has BOTH daily_target_hit=True AND
    daily_stop_hit=True.  Extracts / fetches hourly candles for that date alone,
    walks them chronologically, and returns:

        "TARGET"    -- target was hit before stop
        "STOP"      -- stop was hit before target
        "AMBIGUOUS" -- both hit in the same hourly candle, or no hourly data
        "NEITHER"   -- hourly data exists but neither level was touched
                       (caller should continue to next daily bar)

    Parameters
    ----------
    ambiguous_date   : "YYYY-MM-DD" string
    direction        : "BUY" or "SELL"
    target           : target price level
    stop             : stop price level
    hourly_df        : pre-loaded 1H DataFrame (Date col = datetime or str).
                       Filtered to ambiguous_date internally.  If None,
                       hourly_fetcher is used instead.
    hourly_fetcher   : callable(date_str: str) -> pd.DataFrame | None
                       Must return hourly bars for that single day with at
                       least Date, high, low columns.  Pass None if unavailable.
    """
    hourly_slice = None

    if hourly_df is not None:
        dates = pd.to_datetime(hourly_df["Date"], utc=True, errors="coerce")
        mask  = dates.dt.strftime("%Y-%m-%d") == ambiguous_date
        hourly_slice = hourly_df.loc[mask].copy()
    elif hourly_fetcher is not None:
        try:
            fetched = hourly_fetcher(ambiguous_date)
            if fetched is not None and len(fetched) > 0:
                hourly_slice = fetched.copy()
        except Exception:
            pass

    if hourly_slice is None or len(hourly_slice) == 0:
        return "AMBIGUOUS"

    # normalise column names to lowercase
    hourly_slice.columns = [c.lower() for c in hourly_slice.columns]
    if "date" in hourly_slice.columns:
        hourly_slice = hourly_slice.rename(columns={"date": "Date"})
    hourly_slice = hourly_slice.sort_values("Date").reset_index(drop=True)

    for _, row in hourly_slice.iterrows():
        h_high = float(row["high"])
        h_low  = float(row["low"])

        if direction == "BUY":
            # BUY: expect price DOWN  => target below, stop above
            target_hit_h = h_low  <= target
            stop_hit_h   = h_high >= stop
        else:
            # SELL: expect price UP   => target above, stop below
            target_hit_h = h_high >= target
            stop_hit_h   = h_low  <= stop

        if target_hit_h and stop_hit_h:
            return "AMBIGUOUS"   # both in same hourly candle
        if target_hit_h:
            return "TARGET"
        if stop_hit_h:
            return "STOP"

    return "NEITHER"   # no level touched in any hourly candle


def label_absorption_atr(
    df: pd.DataFrame,
    atr_period:       int   = 14,
    target_atr_mult:  float = 1.0,
    stop_atr_mult:    float = 1.0,
    forward_days:     int   = 10,
    hourly_df:        "pd.DataFrame | None" = None,
    hourly_fetcher=None,
) -> pd.DataFrame:
    """
    ATR-based outcome labeler with hourly same-bar disambiguation.

    Target / Stop computation
    -------------------------
    ATR[t]   = mean(High[j] - Low[j]) for j in [t-atr_period+1 .. t]

    BUY  (absorption of selling -- expect price DOWN):
        target = Close[t] - ATR[t] * target_atr_mult
        stop   = Close[t] + ATR[t] * stop_atr_mult

    SELL (absorption of buying  -- expect price UP):
        target = Close[t] + ATR[t] * target_atr_mult
        stop   = Close[t] - ATR[t] * stop_atr_mult

    Forward scan (daily bars t+1 .. t+forward_days)
    ------------------------------------------------
    For each future daily bar j:

        BUY:  daily_target_hit = Low[j]  <= target
              daily_stop_hit   = High[j] >= stop

        SELL: daily_target_hit = High[j] >= target
              daily_stop_hit   = Low[j]  <= stop

        Case 1 -- target only:       outcome="TARGET",    sl_success=1, stop_hit=0
        Case 2 -- stop only:         outcome="STOP",      sl_success=0, stop_hit=1
        Case 3 -- BOTH (ambiguous):  drill into hourly OHLC for that date only
            hourly="TARGET"   -> outcome="TARGET",    sl_success=1, stop_hit=0
            hourly="STOP"     -> outcome="STOP",      sl_success=0, stop_hit=1
            hourly="AMBIGUOUS"-> outcome="AMBIGUOUS", sl_success=NaN
            hourly="NEITHER"  -> continue to next daily bar (treat as clean)
        Case 4 -- neither:           continue to next daily bar

    If window expires with no hit -> outcome="UNRESOLVED", sl_success=NaN.

    Parameters
    ----------
    df               : Output of calculate_absorption().  Must contain:
                       Date, open, high, low, close, absorption_direction.
    atr_period       : Number of bars for ATR calculation (default 14).
    target_atr_mult  : ATR multiplier for profit target distance (default 1.0).
    stop_atr_mult    : ATR multiplier for stop distance (default 1.0).
    forward_days     : Maximum daily bars to scan forward (default 10).
    hourly_df        : Optional pre-loaded 1H OHLC DataFrame.
                       Must have columns: Date (datetime or str), high, low.
                       Used to resolve same-bar ambiguity -- only the rows
                       matching the ambiguous calendar date are accessed.
                       Takes priority over hourly_fetcher.
    hourly_fetcher   : Optional callable(date_str: str) -> pd.DataFrame | None.
                       Called ONLY for ambiguous dates; returns hourly bars for
                       that one day.  Ignored if hourly_df is provided.

    Returns
    -------
    df with additional columns:
        atr          float  -- ATR at signal bar
        target_price float  -- computed profit target
        stop_loss    float  -- computed stop level
        stop_hit     float  -- 1=stop hit, 0=target/unresolved, NaN=ambiguous/censored
        sl_success   float  -- 1=success, 0=failure, NaN=ambiguous/censored
        outcome      str    -- "TARGET" | "STOP" | "AMBIGUOUS" | "UNRESOLVED" | ""
        stop_logic   str    -- "ATR" for rows processed here, "" otherwise
    """
    df = df.copy().sort_values("Date").reset_index(drop=True)
    n  = len(df)

    def _get(name: str) -> np.ndarray:
        for c in df.columns:
            if c.lower() == name.lower():
                return df[c].values
        raise KeyError(f"Column '{name}' not found. Available: {list(df.columns)}")

    close_  = _get("close")
    high_   = _get("high")
    low_    = _get("low")
    dir_    = _get("absorption_direction")
    dates_  = df["Date"].values

    # -- pre-compute ATR for all bars ------------------------------------------
    # Full Wilder True Range: max(H-L, |H-prevClose|, |L-prevClose|).
    # Bar 0 has no prior close, so we fall back to H-L for that bar.
    tr_  = np.empty(n)
    tr_[0] = high_[0] - low_[0]
    for i in range(1, n):
        tr_[i] = max(
            high_[i] - low_[i],
            abs(high_[i] - close_[i - 1]),
            abs(low_[i]  - close_[i - 1]),
        )
    atr_ = np.full(n, np.nan)
    for i in range(atr_period - 1, n):
        atr_[i] = float(tr_[i - atr_period + 1 : i + 1].mean())

    # -- output arrays ---------------------------------------------------------
    atr_out      = np.full(n, np.nan)
    target_out   = np.full(n, np.nan)
    stop_out     = np.full(n, np.nan)
    stop_hit_out = np.full(n, np.nan)
    success_out  = np.full(n, np.nan)
    outcome_out  = [""] * n
    logic_out    = [""] * n

    for i in range(n):
        if np.isnan(atr_[i]) or (i + forward_days) >= n:
            continue

        direction = str(dir_[i]).upper()
        if direction not in ("SELL", "BUY"):
            continue

        c0        = float(close_[i])
        atr_val   = float(atr_[i])
        target_d  = atr_val * target_atr_mult
        stop_d    = atr_val * stop_atr_mult

        if direction == "BUY":
            target = c0 - target_d
            stop   = c0 + stop_d
        else:  # SELL
            target = c0 + target_d
            stop   = c0 - stop_d

        atr_out[i]    = round(atr_val, 8)
        target_out[i] = round(target, 8)
        stop_out[i]   = round(stop, 8)
        logic_out[i]  = "ATR"

        # -- forward daily scan ------------------------------------------------
        final_outcome   = "UNRESOLVED"
        final_stop_hit  = np.nan
        final_success   = np.nan

        for offset in range(1, forward_days + 1):
            j = i + offset
            if j >= n:
                break

            j_high = float(high_[j])
            j_low  = float(low_[j])

            if direction == "BUY":
                daily_target_hit = j_low  <= target
                daily_stop_hit   = j_high >= stop
            else:  # SELL
                daily_target_hit = j_high >= target
                daily_stop_hit   = j_low  <= stop

            if daily_target_hit and not daily_stop_hit:
                # Case 1: target only
                final_outcome  = "TARGET"
                final_stop_hit = 0.0
                final_success  = 1.0
                break

            elif daily_stop_hit and not daily_target_hit:
                # Case 2: stop only
                final_outcome  = "STOP"
                final_stop_hit = 1.0
                final_success  = 0.0
                break

            elif daily_target_hit and daily_stop_hit:
                # Case 3: same-bar ambiguity -- resolve with hourly data
                raw_date = dates_[j]
                try:
                    amb_date_str = pd.Timestamp(raw_date).strftime("%Y-%m-%d")
                except Exception:
                    amb_date_str = str(raw_date)[:10]

                hourly_result = _resolve_ambiguity_hourly(
                    ambiguous_date = amb_date_str,
                    direction      = direction,
                    target         = target,
                    stop           = stop,
                    hourly_df      = hourly_df,
                    hourly_fetcher = hourly_fetcher,
                )

                if hourly_result == "TARGET":
                    final_outcome  = "TARGET"
                    final_stop_hit = 0.0
                    final_success  = 1.0
                    break
                elif hourly_result == "STOP":
                    final_outcome  = "STOP"
                    final_stop_hit = 1.0
                    final_success  = 0.0
                    break
                elif hourly_result == "AMBIGUOUS":
                    final_outcome  = "AMBIGUOUS"
                    final_stop_hit = np.nan
                    final_success  = np.nan
                    break
                else:
                    # "NEITHER": hourly data covered the day but neither level
                    # was touched -- treat as clean bar and keep scanning
                    continue

            # Case 4: neither hit -- continue scanning

        stop_hit_out[i] = final_stop_hit
        success_out[i]  = final_success
        outcome_out[i]  = final_outcome

    df["atr"]          = atr_out
    df["target_price"] = target_out
    df["stop_loss"]    = stop_out
    df["stop_hit"]     = stop_hit_out
    df["sl_success"]   = success_out
    df["outcome"]      = outcome_out
    df["stop_logic"]   = logic_out
    return df
