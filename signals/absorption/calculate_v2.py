# market_analysis/signals/absorption/calculate_v2.py
"""
Absorption score — Method 2 (z-score based).

Replaces the percentile-of-|delta| approach in calculate.py with a
two-threshold z-score model:

    BUY  absorption: z_delta >= +X  AND  z_price <=  +Y
                     (huge buy-delta, price barely moves up → selling absorbed)

    SELL absorption: z_delta <= -X  AND  z_price >= -Y
                     (huge sell-delta, price barely moves down → buying absorbed)

All rolling statistics are CAUSAL (shift-1 before rolling) so there is
zero look-ahead bias.

Input DataFrame must have columns (case-insensitive):
    Open, High, Low, Close, Volume, BuyVolume, SellVolume, Date

Output adds these columns alongside the originals:
    absorption_score       float [0, 1]   — continuous signal strength
    absorption_direction   str            — "BUY" | "SELL" | "NEUTRAL"
    absorption_raw         float          — pre-sigmoid raw score
    delta                  float          — buyvolume - sellvolume
    z_delta                float          — robust z-score of normalised delta
    z_price                float          — robust z-score of normalised price response

Column-name convention mirrors calculate.py so label.py and
label_hourly.py work without modification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_median_mad_causal(s: pd.Series, W: int, minp: int):
    """
    Causal rolling median and MAD — uses only bars strictly before the
    current bar (shift-1 so bar i only sees bars 0..i-1).
    """
    prev = s.shift(1)
    roll = prev.rolling(W, min_periods=minp)
    med  = roll.median()

    def _mad(x: np.ndarray) -> float:
        m = np.median(x)
        return float(np.median(np.abs(x - m)))

    mad = roll.apply(_mad, raw=True)
    return med, mad


def _robust_z(x: pd.Series, med: pd.Series, mad: pd.Series, eps: float = _EPS) -> pd.Series:
    """Robust z-score: (x - median) / (1.4826 * MAD + eps)."""
    return (x - med) / (1.4826 * mad + eps)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def calculate_absorption_v2(
    df: pd.DataFrame,
    # z-score model windows
    volume_window:    int   = 5,
    atr_window:       int   = 5,
    baseline_window:  int   = 15,
    min_periods:      int | None = None,
    # signal thresholds
    delta_z_threshold: float = 2.0,   # X — how extreme the delta must be
    price_z_threshold: float = 0.75,  # Y — how muted the price response must be
) -> pd.DataFrame:
    """
    Calculate Method-2 absorption signals from OHLC bars.

    Parameters
    ----------
    df                 : DataFrame with OHLC + buyvolume/sellvolume columns.
    volume_window      : Look-back for rolling avg volume used to normalise delta.
    atr_window         : Look-back for ATR used to normalise price response.
    baseline_window    : Rolling window (W) for median / MAD z-score baseline.
    min_periods        : Minimum valid observations in rolling windows.
                         Defaults to max(8, baseline_window // 2).
    delta_z_threshold  : z_delta must exceed ±X to qualify (default 2.0).
    price_z_threshold  : z_price must be within ±Y to qualify (default 0.75).

    Returns
    -------
    Input DataFrame with additional columns:
        absorption_score, absorption_direction, absorption_raw,
        delta, z_delta, z_price
    """
    if min_periods is None:
        min_periods = max(8, baseline_window // 2)

    data = df.copy()

    # ── Normalise column names to lowercase, keep 'Date' capitalised ─────────
    data.columns = [c.lower() for c in data.columns]
    if "date" in data.columns:
        data = data.rename(columns={"date": "Date"})

    # ── Sort chronologically ──────────────────────────────────────────────────
    for col in ("Date", "time", "timestamp"):
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], utc=True, errors="coerce")
            data = data.sort_values(col).reset_index(drop=True)
            break

    # ── Coerce numerics ───────────────────────────────────────────────────────
    for col in ("open", "high", "low", "close", "volume", "buyvolume", "sellvolume"):
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    required = {"open", "high", "low", "close", "volume", "buyvolume", "sellvolume"}
    missing  = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"calculate_absorption_v2: missing columns {missing}")

    buy_v  = data["buyvolume"]
    sell_v = data["sellvolume"]
    volume = data["volume"]
    high   = data["high"]
    low    = data["low"]
    close  = data["close"]

    # ── Delta ─────────────────────────────────────────────────────────────────
    delta = buy_v - sell_v

    # ── 1. Normalised delta = delta / rolling_avg_volume ──────────────────────
    #avg_vol=1 without normalisation
    avg_vol = (
        volume
        .shift(1)
        .rolling(volume_window, min_periods=max(3, volume_window // 2))
        .mean()
    )
    normalised_delta = delta / (avg_vol + _EPS)

    # ── 2. Normalised price response = (close - prev_close) / ATR ────────────
    price_response = (close - close.shift(1)).abs()
    prev_close     = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = (
        tr.shift(1)
        .rolling(atr_window, min_periods=max(3, atr_window // 2))
        .mean()
    )
    #atr=1
    normalised_price = price_response / (atr + _EPS)

    # ── 3. Causal robust z-scores ─────────────────────────────────────────────
    d_med, d_mad = _rolling_median_mad_causal(normalised_delta, baseline_window, min_periods)
    p_med, p_mad = _rolling_median_mad_causal(normalised_price, baseline_window, min_periods)

    z_delta = _robust_z(normalised_delta, d_med, d_mad)
    z_price = _robust_z(normalised_price, p_med, p_mad)

    # ── 4. Absorption signal conditions ───────────────────────────────────────
    #
    # Price response is ABSOLUTE:
    #     price_response = abs(close_t - close_{t-1})
    #
    # Therefore z_price measures only the MAGNITUDE of price movement.
    # Direction comes entirely from z_delta.
    #
    # BUY absorption:
    #     unusually large positive delta
    #     + unusually small price movement
    #
    # SELL absorption:
    #     unusually large negative delta
    #     + unusually small price movement

    buy_signal = (
        (z_delta >= delta_z_threshold) &
        (z_price <= price_z_threshold)
    )

    sell_signal = (
        (z_delta <= -delta_z_threshold) &
        (z_price <= price_z_threshold)
    )

    direction = np.select(
        [buy_signal, sell_signal],
        ["BUY", "SELL"],
        default="NEUTRAL",
    )


    # ── 5. Continuous absorption score ────────────────────────────────────────
    #
    # Score represents:
    #
    #     LARGE DELTA  ×  SMALL PRICE RESPONSE
    #
    # Delta strength:
    #     How extreme is |z_delta|?
    #
    # Price suppression:
    #     How unusually small is z_price?
    #
    # Both components are mapped to [0, 1].

    # 5a. Delta extremeness
    delta_strength = 1 / (1 + np.exp(-abs(z_delta)))

    # 5b. Price suppression
    #
    # Lower z_price = stronger absorption.
    # Around z_price = 0.5, suppression = 0.5.
    price_suppression = 1.0 / (1.0 + np.exp(z_price))

    # 5c. Final continuous score
    absorption_score = (
        delta_strength * price_suppression
    )

    # Optional minimum delta requirement:
    # don't call small delta + small price movement "absorption".
    absorption_score = absorption_score.where(
        np.abs(z_delta) >= 1.0,
        0.0
    )

    # Raw score before [0,1] scaling
    absorption_raw = (
        np.abs(z_delta) * price_suppression
    )
    # ── 6. Attach columns ─────────────────────────────────────────────────────

    data["delta"]                = delta.values
    data["z_delta"]              = z_delta.values
    data["z_price"]              = z_price.values
    data["delta_strength"]       = delta_strength.values
    data["price_suppression"]    = price_suppression.values
    data["absorption_raw"]       = absorption_raw.values
    data["absorption_score"]     = absorption_score.values
    data["absorption_direction"] = direction

    
    return data
