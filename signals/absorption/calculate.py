# market_analysis/signals/absorption/calculate.py
"""
Absorption score calculation — extracted from the core engine.
calculate_absorption() takes a plain OHLC DataFrame and returns it
with all absorption columns added. No external schemas required.

Input DataFrame must have lowercase columns:
  open, high, low, close, volume, buyvolume, sellvolume, Date
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd


def calculate_absorption(
    df: pd.DataFrame,
    window: int = 60,
    tick_size: float = 0.25,
    strength_threshold: float = 0.60,
) -> pd.DataFrame:
    """
    Calculate absorption signals from OHLC bars with buyVolume/sellVolume.

    Required columns:
        open, close, high, low, volume, buyvolume, sellvolume, Date (timestamp)

    Returns a DataFrame with one row per bar, containing all absorption metrics.
    All percentile baselines are past-only (shift-1 semantics).
    """
    if window < 1:
        raise ValueError(f"Absorption percentile window must be >= 1, got {window}")

    data = df.copy()

    # Sort chronologically
    for col in ("Date", "time", "timestamp"):
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], utc=True, errors="coerce")
            data = data.sort_values(col).reset_index(drop=True)
            break

    # Coerce numerics
    for col in ("open", "close", "high", "low", "volume", "buyvolume", "sellvolume"):
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    required = {"open", "close", "high", "low", "volume", "buyvolume", "sellvolume"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Absorption is missing OHLC fields: {', '.join(missing)}")

    # ─── Volume ───────────────────────────────────────────────────────────────
    data["buy_volume"]   = data["buyvolume"]
    data["sell_volume"]  = data["sellvolume"]
    data["total_volume"] = data["volume"]

    # ─── Delta / Delta% ───────────────────────────────────────────────────────
    data["delta"] = data["buyvolume"] - data["sellvolume"]
    data["delta_pct"] = np.where(
        data["total_volume"] > 0,
        data["delta"] / data["total_volume"] * 100,
        np.nan,
    )

    # ─── Direction ────────────────────────────────────────────────────────────
    data["absorption_direction"] = np.select(
        [data["delta"] > 0, data["delta"] < 0],
        ["BUY", "SELL"],
        default="NEUTRAL",
    )

    # ─── Price range ──────────────────────────────────────────────────────────
    data["price_range"]         = (data["high"] - data["low"]).abs()
    data["price_response"]      = data["price_range"]
    data["price_response_ticks"] = data["price_response"] / tick_size

    # ─── Aggression score: past-only percentile of |delta| ───────────────────
    abs_delta       = data["delta"].abs().to_numpy(dtype=float)
    aggression_score = np.full(len(data), np.nan)

    for i in range(len(data)):
        if i < window:
            continue
        history = abs_delta[i - window: i]
        valid   = history[np.isfinite(history)]
        if len(valid) < window:
            continue
        current = abs_delta[i]
        if not np.isfinite(current):
            continue
        aggression_score[i] = float(np.sum(valid <= current)) / len(valid)

    data["aggression_score"] = aggression_score

    # ─── Price range score: 1 - past-only percentile of (high-low) ───────────
    range_vals = data["price_range"].to_numpy(dtype=float)
    range_pct  = np.full(len(data), np.nan)

    for i in range(len(data)):
        if i < window:
            continue
        history = range_vals[i - window: i]
        valid   = history[np.isfinite(history)]
        if len(valid) < window:
            continue
        current = range_vals[i]
        if not np.isfinite(current):
            continue
        range_pct[i] = float(np.sum(valid <= current)) / len(valid)

    data["price_range_percentile"] = range_pct
    data["price_response_score"]   = 1.0 - data["price_range_percentile"]

    # ─── Final absorption strength ─────────────────────────────────────────────
    data["absorption_strength"] = (
        data["aggression_score"] * data["price_response_score"]
    )

    # ─── Classification ───────────────────────────────────────────────────────
    def _classify(s: float) -> str:
        if s < 0.25:  return "NONE"
        if s < 0.40:  return "POSSIBLE"
        if s < 0.60:  return "MODERATE"
        if s < 0.80:  return "STRONG"
        return "EXTREME"

    data["absorption_class"] = data["absorption_strength"].apply(
        lambda s: _classify(float(s)) if np.isfinite(s) else "NONE"
    )

    # ─── Absorption volume % ──────────────────────────────────────────────────
    data["absorption_volume"] = np.where(
        data["absorption_strength"] >= strength_threshold,
        data["total_volume"],
        0.0,
    )
    total_vol = float(data["total_volume"].sum())
    abs_vol   = float(data["absorption_volume"].sum())
    data["absorption_volume_pct"] = (
        (abs_vol / total_vol * 100) if total_vol > 0 else np.nan
    )

    return data