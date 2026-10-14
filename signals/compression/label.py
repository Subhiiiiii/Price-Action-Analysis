# market_analysis/signals/compression/label.py
"""
Compression outcome labeling.

success = max(abs(Close[t+i] - Close[t]), i=1..forward_days) > k × tick_size
No direction, no stop loss — compression predicts volatility expansion only.
Zero compression_score rows are dropped before labeling (no signal).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from ...config import COMPRESSION


def label_compression(
    df: pd.DataFrame,
    tick:         float | None = None,
    k:            int   | None = None,
    forward_days: int   | None = None,
    keep_all_bars: bool = True,
) -> pd.DataFrame:
    """
    Label each row using the actual next `forward_days` trading days.

    Priority 1 Fix:
    Labels are calculated across the continuous sequence of actual trading days
    BEFORE filtering any signal rows. This guarantees that Close[t+1..t+fwd]
    corresponds to real market trading days rather than non-zero signal bars.

    Columns added:
      max_move_Nd  : max absolute close move over the forward window
      success      : 1 = expansion happened (> k × tick), 0 = no, NaN = censored
    """
    tick_     = tick         if tick         is not None else COMPRESSION.get("tick_size", 0.1)
    k_        = k            if k            is not None else COMPRESSION["k"]
    fwd       = forward_days if forward_days is not None else COMPRESSION["forward_days"]
    req       = k_ * tick_
    col_name  = f"max_move_{fwd}d"

    # Sort strictly by Date on the FULL sequence of trading days
    df_full = df.copy().sort_values("Date").reset_index(drop=True)
    n       = len(df_full)
    close_v = pd.to_numeric(df_full["close"], errors="coerce").values
    mm      = np.full(n, np.nan)
    succ    = np.full(n, np.nan)

    for i in range(n):
        if i + fwd >= n:
            continue
        fwd_closes = close_v[i + 1 : i + fwd + 1]
        if len(fwd_closes) == 0 or np.isnan(fwd_closes).all() or np.isnan(close_v[i]):
            continue
        move       = float(np.nanmax(np.abs(fwd_closes - close_v[i])))
        mm[i]      = move
        succ[i]    = 1.0 if move > req else 0.0

    df_full[col_name]  = mm
    df_full["success"] = succ

    if not keep_all_bars:
        df_full = df_full[df_full.get("compression_score", 0) > 0].reset_index(drop=True)

    return df_full
