# market_analysis/signals/compression/calculate_multiday.py
"""
Builds n-day rolling candles and scores compression in two ways.

track="candle"  (Track A)
    Rolling n-day candle series (one per day, overlapping) is treated as an
    ordinary "daily" series. Baseline = previous `range_baseline` rolling candles.

track="custom"  (Track B)
    Baseline is made of NON-overlapping n-day candles anchored at the current
    candle. For a 3D candle Tue+Wed+Thu with range_baseline=2, the baseline is
    Thu-Fri-Mon (previous week) and the one before it, etc.
    Implementation: the rolling candle series is split into n "phases"
    (every n-th candle). Each phase is a clean, non-overlapping n-day series,
    so calculate_compression() runs on it unchanged. Phases are then merged
    back into one daily-stepping series. Fully causal.
"""
from __future__ import annotations
import pandas as pd
from .calculate import calculate_compression
from ...data.ohlc import _aggregate_trailing


def build_scored_candles(
    df_1d: pd.DataFrame,
    n: int,
    track: str = "candle",
    range_baseline: int = 20,
    atr_period: int = 5,
    rv_period: int = 5,
    persistence_window: int = 20,
    compression_threshold: float = 0.30,
    label_atr_period: int = 10,
) -> pd.DataFrame:
    if track not in ("candle", "custom"):
        raise ValueError("track must be 'candle' or 'custom'")

    kw = dict(
        range_baseline=range_baseline, atr_period=atr_period, rv_period=rv_period,
        persistence_window=persistence_window,
        compression_threshold=compression_threshold,
        label_atr_period=label_atr_period,
    )

    candles = df_1d.copy() if n == 1 else _aggregate_trailing(df_1d, n)
    candles = candles.sort_values("Date").reset_index(drop=True)

    if track == "candle" or n == 1:
        scored, _ = calculate_compression(candles, **kw)
        scored["phase"] = 0
        return scored

    parts = []
    for phase in range(n):
        sub = candles.iloc[phase::n]          # candles exactly n days apart
        if len(sub) < 3:
            continue
        scored, _ = calculate_compression(sub, **kw)
        scored["phase"] = phase
        parts.append(scored)

    if not parts:
        return pd.DataFrame()
    return (pd.concat(parts, ignore_index=True)
              .sort_values("Date").reset_index(drop=True))