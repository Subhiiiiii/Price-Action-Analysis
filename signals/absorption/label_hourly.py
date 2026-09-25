# market_analysis/signals/absorption/label_hourly.py
"""
Profit-first outcome labeling for 1H absorption signals.

Problem with the original label_absorption():
    If price hits the profit target at bar t+3 and *then* reverses into the
    stop at bar t+7, the original logic wrongly counts it as a failure
    because stop_hit=True is evaluated over the whole forward window.

New logic — "profit-first" scan:
    Walk forward bars one at a time.  Record which event fires FIRST.

    SELL absorption (expect price UP):
        profit_target = close[t] + required_move
        stop_level    = min(low[t-sl_lb : t+1]) - tick

        scan t+1, t+2, ... t+k in order:
          - if close[bar] >= profit_target  → profit_hit = True,  break
          - if low[bar]   <= stop_level     → stop_hit   = True,  break

    BUY absorption (expect price DOWN):
        profit_target = close[t] - required_move
        stop_level    = max(high[t-sl_lb : t+1]) + tick

        scan t+1, t+2, ... t+k in order:
          - if close[bar] <= profit_target  → profit_hit = True,  break
          - if high[bar]  >= stop_level     → stop_hit   = True,  break

    sl_success:
        1   if profit_hit (regardless of what happens after)
        0   if stop_hit (and profit_hit = False)
        0   if neither fires within k bars
        NaN if censored (not enough forward bars)

Extra diagnostic columns:
    profit_hit        (bool int)  — 1 if profit target was reached first
    stop_hit_first    (bool int)  — 1 if stop fired before profit target
    profit_bar_offset (int | NaN) — how many bars after t until profit was hit

Parameters fall back to config.ABSORPTION_1H when None.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...config import ABSORPTION_1H


def label_absorption_hourly(
    df: pd.DataFrame,
    tick:                float | None = None,
    required_move_ticks: int   | None = None,
    sl_lookback:         int   | None = None,
    forward_bars:        int   | None = None,
) -> pd.DataFrame:
    """
    Profit-first labeling for 1H bars.

    Parameters
    ----------
    df                  Output of calculate_absorption() on 1H OHLC data.
    tick                Tick size (falls back to config.ABSORPTION_1H default).
    required_move_ticks Min bars of move to count as success.
    sl_lookback         Past bars used to set the stop level.
    forward_bars        Max bars to scan forward for outcome (≈2.5 trading days
                        at default of 40 with 16 bars/day).

    Returns
    -------
    df with additional columns:
        stop_loss         float  — computed stop level
        profit_hit        int    — 1 if profit target fired first
        stop_hit_first    int    — 1 if stop fired first (profit_hit=0)
        profit_bar_offset int    — bars until profit hit (NaN if not hit)
        sl_success        float  — 1/0/NaN outcome label
    """
    tick     = tick                 if tick                 is not None else 0.1
    req_move = (
        (required_move_ticks if required_move_ticks is not None
         else ABSORPTION_1H["required_move_ticks"]) * tick
    )
    sl_lb = sl_lookback  if sl_lookback  is not None else ABSORPTION_1H["sl_lookback"]
    k     = forward_bars if forward_bars is not None else ABSORPTION_1H["forward_bars"]

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
    dir_   = _get("absorption_direction")

    stop_loss_out       = np.full(n, np.nan)
    profit_hit_out      = np.full(n, np.nan)
    stop_hit_first_out  = np.full(n, np.nan)
    profit_bar_off_out  = np.full(n, np.nan)
    success_out         = np.full(n, np.nan)

    for i in range(n):
        past_start = i - sl_lb + 1

        # censored: not enough past bars for stop, or not enough forward bars
        if past_start < 0 or (i + k) >= n:
            continue

        direction = str(dir_[i]).upper()
        c0        = close_[i]

        if direction not in ("SELL", "BUY"):
            continue   # NEUTRAL — skip

        # ── compute stop level ────────────────────────────────────────────────
        if direction == "SELL":
            profit_target = c0 + req_move
            sl            = float(low_[past_start : i + 1].min()) - tick
        else:  # BUY
            profit_target = c0 - req_move
            sl            = float(high_[past_start : i + 1].max()) + tick

        stop_loss_out[i] = round(sl, 8)

        # ── profit-first forward scan ─────────────────────────────────────────
        profit_hit     = False
        stop_hit_first = False
        profit_bar_off = np.nan

        for offset in range(1, k + 1):
            j = i + offset
            if j >= n:
                break  # censored mid-scan — treat as neither

            if direction == "SELL":
                profit_fired = close_[j] >= profit_target
                stop_fired   = low_[j]   <= sl
            else:
                profit_fired = close_[j] <= profit_target
                stop_fired   = high_[j]  >= sl

            if profit_fired:
                profit_hit     = True
                profit_bar_off = offset
                break
            if stop_fired:
                stop_hit_first = True
                break
            # neither yet — continue scanning

        profit_hit_out[i]     = int(profit_hit)
        stop_hit_first_out[i] = int(stop_hit_first)
        profit_bar_off_out[i] = profit_bar_off

        # ── outcome ───────────────────────────────────────────────────────────
        if profit_hit:
            success_out[i] = 1.0   # profit reached first — success
        elif stop_hit_first:
            success_out[i] = 0.0   # stop hit before profit — failure
        else:
            success_out[i] = 0.0   # k bars elapsed, neither triggered

    df["stop_loss"]        = stop_loss_out
    df["profit_hit"]       = profit_hit_out
    df["stop_hit_first"]   = stop_hit_first_out
    df["profit_bar_offset"]= profit_bar_off_out
    df["sl_success"]       = success_out
    return df
