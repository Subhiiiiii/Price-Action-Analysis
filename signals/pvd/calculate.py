# market_analysis/signals/pvd/calculate.py
"""
PVD (Price-Volume Divergence) score calculation.

Adapted from the internal engine PVD module.  All Bar/EngineResult/Metric
schema dependencies have been removed so this works directly with plain OHLC
DataFrames, consistent with the absorption and compression modules.

Input DataFrame must have columns (any case):
    Date, Close (or close), Volume (or volume)

Output DataFrame adds columns:
    pvd_z          -- standardised residual z-score
    pvd_score      -- 0..1 magnitude score (= min(100, 50*|z|) / 100)
    volume_ratio   -- current horizon volume / recent average horizon volume
    pvd_direction  -- "UP" | "DOWN" | "NEUTRAL"  (based on price_change sign)
    pvd_signal     -- "UP_TREND_EXHAUSTION" | "DOWN_TREND_EXHAUSTION"
                      "UP_PRICE_EXPANSION"  | "DOWN_PRICE_EXPANSION"
                      "NORMAL"
    pvd_strength   -- "EXTREME" (|z|>=2) | "STRONG" (|z|>=1.5) | "NONE"
    pvd_atr        -- ATR[t] = mean(High - Low) over atr_period bars
                      (used downstream by label_pvd for outcome calculation)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...config import PVD

EPSILON: float = 1e-10


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Case-insensitive column lookup."""
    for c in df.columns:
        if c.lower() == name.lower():
            return pd.to_numeric(df[c], errors="coerce")
    raise KeyError(f"Column '{name}' not found. Available: {list(df.columns)}")


def calculate_pvd(
    df: pd.DataFrame,
    price_horizon_h:    int   | None = None,
    volatility_window:  int   | None = None,
    pvd_model_window:   int   | None = None,
    atr_period:         int   | None = None,
    exhaustion_z_threshold: float | None = None,
    expansion_z_threshold:  float | None = None,
    eps: float = EPSILON,
) -> pd.DataFrame:
    """
    Calculate PVD scores and signals from an OHLC DataFrame.

    Parameters fall back to config.PVD values when None.

    Parameters
    ----------
    df                      : DataFrame with Date, Close/close, Volume/volume,
                              and optionally High/high, Low/low (needed for ATR).
    price_horizon_h         : Bar count for price-change window.
    volatility_window       : Rolling window for 1-bar volatility sigma.
    pvd_model_window        : Rolling window for OLS beta/alpha and residual stats.
    atr_period              : Bars for ATR calculation (High - Low mean).
    exhaustion_z_threshold  : z <= this threshold triggers TREND_EXHAUSTION.
                              Default from config (negative value, e.g. -1.5).
    expansion_z_threshold   : z >= this threshold triggers PRICE_EXPANSION.
                              Default from config (positive value, e.g. +1.5).
    eps                     : Small constant to prevent division by zero.

    Returns
    -------
    Original df with additional columns listed in the module docstring.
    """
    # -- resolve params from config --------------------------------------------
    horizon       = int(price_horizon_h   if price_horizon_h   is not None else PVD["price_horizon_h"])
    vol_win       = int(volatility_window  if volatility_window  is not None else PVD["volatility_window"])
    model_win     = int(pvd_model_window   if pvd_model_window   is not None else PVD["pvd_model_window"])
    atr_per       = int(atr_period         if atr_period         is not None else PVD["atr_period"])
    exh_z         = float(exhaustion_z_threshold if exhaustion_z_threshold is not None
                          else PVD["exhaustion_z_threshold"])
    exp_z         = float(expansion_z_threshold  if expansion_z_threshold  is not None
                          else PVD["expansion_z_threshold"])

    if min(horizon, vol_win, model_win, atr_per) <= 0:
        raise ValueError("All PVD window parameters must be positive integers")

    df = df.copy().sort_values("Date").reset_index(drop=True)

    close  = _col(df, "close")
    volume = _col(df, "volume")

    # -- price change & 1-bar volatility ---------------------------------------
    price_change  = close - close.shift(horizon)
    one_bar_delta = close - close.shift(1)

    vol_min = max(5, vol_win // 2)
    sigma   = (
        one_bar_delta.shift(1)
        .rolling(vol_win, min_periods=vol_min)
        .std()
    )
    normalized_price_response = price_change.abs() / (sigma + eps)

    # -- volume ratio ----------------------------------------------------------
    model_min = max(5, model_win // 2)
    horizon_volume     = volume.rolling(horizon, min_periods=1).sum()
    horizon_volume_avg = (
        horizon_volume.shift(1)
        .rolling(model_win, min_periods=model_min)
        .mean()
    )
    volume_ratio = horizon_volume / (horizon_volume_avg + eps)

    # -- past-only OLS: log-log model ------------------------------------------
    x      = np.log1p(volume_ratio)
    y      = np.log1p(normalized_price_response)
    past_x = x.shift(1)
    past_y = y.shift(1)

    cov     = past_x.rolling(model_win, min_periods=model_min).cov(past_y)
    var_x   = past_x.rolling(model_win, min_periods=model_min).var()
    beta    = cov / (var_x + eps)
    alpha   = (
        past_y.rolling(model_win, min_periods=model_min).mean()
        - beta * past_x.rolling(model_win, min_periods=model_min).mean()
    )

    expected_npr = np.exp((alpha + beta * x).clip(-20, 20)) - 1
    expected_npr = expected_npr.clip(lower=-0.999999)

    # -- residual z-score ------------------------------------------------------
    residual = np.log1p(normalized_price_response) - np.log1p(expected_npr)
    res_mean = residual.shift(1).rolling(model_win, min_periods=model_min).mean()
    res_std  = residual.shift(1).rolling(model_win, min_periods=model_min).std()
    z        = (residual - res_mean) / (res_std + eps)

    pvd_score = (np.minimum(100, 50 * z.abs()) / 100.0).clip(0, 1)

    # -- ATR -------------------------------------------------------------------
    # Full Wilder True Range: max(H-L, |H-prevClose|, |L-prevClose|).
    # close.shift(1) gives the previous bar's close; for the very first bar
    # the prev-close is NaN so pd.concat max naturally falls back to H-L.
    try:
        high  = _col(df, "high")
        low   = _col(df, "low")
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low  - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
    except KeyError:
        tr = pd.Series(np.nan, index=df.index)

    atr_min = max(3, atr_per // 2)
    pvd_atr = tr.rolling(atr_per, min_periods=atr_min).mean()

    # -- direction / signal / strength -----------------------------------------
    direction = np.select(
        [price_change > 0, price_change < 0],
        ["UP", "DOWN"],
        default="NEUTRAL",
    )
    signal = np.select(
        [
            (z <= exh_z) & (price_change > 0),
            (z <= exh_z) & (price_change < 0),
            (z >= exp_z) & (price_change > 0),
            (z >= exp_z) & (price_change < 0),
        ],
        [
            "UP_TREND_EXHAUSTION",
            "DOWN_TREND_EXHAUSTION",
            "UP_PRICE_EXPANSION",
            "DOWN_PRICE_EXPANSION",
        ],
        default="NORMAL",
    )
    strength = np.select(
        [z.abs() >= 2.0, z.abs() >= 1.5],
        ["EXTREME", "STRONG"],
        default="NONE",
    )

    df["pvd_z"]         = z
    df["pvd_score"]     = pvd_score
    df["volume_ratio"]  = volume_ratio
    df["pvd_direction"] = direction
    df["pvd_signal"]    = signal
    df["pvd_strength"]  = strength
    df["pvd_atr"]       = pvd_atr
    return df
