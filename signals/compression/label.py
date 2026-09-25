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
) -> pd.DataFrame:
    """
    Label each non-zero compression row.

    Columns added:
      max_move_Nd  : max absolute close move over the forward window
      success      : 1 = expansion happened, 0 = no, NaN = censored
    """
    tick_     = tick         if tick         is not None else COMPRESSION.get("tick_size", 0.1)
    k_        = k            if k            is not None else COMPRESSION["k"]
    fwd       = forward_days if forward_days is not None else COMPRESSION["forward_days"]
    req       = k_ * tick_
    col_name  = f"max_move_{fwd}d"

    # drop zero-score rows — they carry no signal
    df = (
        df[df["compression_score"] > 0]
        .copy()
        .sort_values("Date")
        .reset_index(drop=True)
    )
    n       = len(df)
    close_v = df["close"].values
    mm      = np.full(n, np.nan)
    succ    = np.full(n, np.nan)

    for i in range(n):
        if i + fwd >= n:
            continue
        fwd_closes = close_v[i + 1 : i + fwd + 1]
        move       = np.abs(fwd_closes - close_v[i]).max()
        mm[i]      = move
        succ[i]    = 1.0 if move > req else 0.0

    df[col_name]   = mm
    df["success"]  = succ
    return df
