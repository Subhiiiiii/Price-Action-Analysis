# market_analysis/signals/compression/calculate.py
"""
Compression score calculation — extracted from the core engine.
calculate_compression() takes a plain OHLC DataFrame and returns it
with all compression columns added.

Input DataFrame must have lowercase columns:
  open, high, low, close, (volume optional), Date
"""

from __future__ import annotations
import math
import numpy as np
import pandas as pd

FEATURE_NAMES = (
    "compression_score",
    "range_component",
    "atr_component",
    "rv_component",
    "volume_range_efficiency",
    "compression_persistence",
)

_ROBUST_K = 1.4826


def _rolling_mad(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    def mad(values: np.ndarray) -> float:
        median = np.median(values)
        return float(np.median(np.abs(values - median)))
    return series.rolling(window, min_periods=min_periods).apply(mad, raw=True)


def _robust_zscore(
    values: pd.Series,
    window: int,
    min_periods: int,
    epsilon: float = 1e-9,
) -> pd.Series:
    median = values.rolling(window, min_periods=min_periods).median().shift(1)
    mad    = _rolling_mad(values, window, min_periods).shift(1)
    return (values - median) / (_ROBUST_K * mad + epsilon)


def calculate_compression(
    df: pd.DataFrame,
    range_baseline: int = 20,
    atr_period: int = 5,
    rv_period: int = 5,
    persistence_window: int = 20,
    compression_threshold: float = 0.60,
    epsilon: float = 1e-9,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Calculate the causal four-component compression score.

    Returns (data_with_columns, feature_names_list).
    """
    data = df.copy()
    for column in ("Date", "time"):
        if column in data.columns:
            data[column] = pd.to_datetime(data[column])
            data = data.sort_values(column).reset_index(drop=True)
            break

    # coerce numerics
    for col in ("open", "high", "low", "close", "volume"):
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data["range"]       = (data["high"] - data["low"]).clip(lower=epsilon)
    data["volume_safe"] = data.get("volume", pd.Series(1.0, index=data.index)).clip(lower=epsilon)
    data["price_change"] = data["close"].diff()
    min_periods = min(range_baseline, max(5, range_baseline // 2))

    log_range   = np.log1p(data["range"])
    range_z     = _robust_zscore(log_range, range_baseline, min_periods, epsilon)
    data["range_component"] = np.clip(-range_z / 2.5, 0.0, 1.0)

    previous_close = data["close"].shift(1)
    data["true_range"] = pd.Series(
        np.maximum.reduce([
            data["range"].values,
            (data["high"] - previous_close).abs().values,
            (data["low"]  - previous_close).abs().values,
        ]),
        index=data.index,
    ).clip(lower=epsilon)
    data["atr"] = data["true_range"].rolling(
        atr_period, min_periods=max(2, atr_period // 2)
    ).mean()
    atr_z = _robust_zscore(np.log1p(data["atr"]), range_baseline, min_periods, epsilon)
    data["atr_component"] = np.clip(-atr_z / 2.5, 0.0, 1.0)

    realized_variance = (
        data["price_change"].fillna(0.0).pow(2)
        .rolling(rv_period, min_periods=max(2, rv_period // 2))
        .mean()
    )
    data["rv"] = realized_variance.apply(
        lambda v: math.sqrt(v) if v is not None and not math.isnan(v) and v >= 0 else np.nan
    ).clip(lower=epsilon)
    rv_z = _robust_zscore(np.log1p(data["rv"]), range_baseline, min_periods, epsilon)
    data["rv_component"] = np.clip(-rv_z / 2.5, 0.0, 1.0)

    efficiency = np.log1p(data["volume_safe"]) / (np.log1p(data["range"]) + epsilon)
    efficiency_z = _robust_zscore(efficiency, range_baseline, min_periods, epsilon)
    data["volume_range_efficiency"] = np.clip(efficiency_z / 2.5, 0.0, 1.0)

    raw_score = (
        0.30 * data["range_component"].fillna(0.0)
        + 0.25 * data["atr_component"].fillna(0.0)
        + 0.25 * data["rv_component"].fillna(0.0)
        + 0.20 * data["volume_range_efficiency"].fillna(0.0)
    )

    minimum_component = compression_threshold * (0.40 / 0.60)
    range_confirmed   = data["range_component"] >= minimum_component
    secondary_count   = (
        (data["atr_component"]            >= minimum_component).astype(int)
        + (data["rv_component"]           >= minimum_component).astype(int)
        + (data["volume_range_efficiency"] >= minimum_component).astype(int)
    )
    confirmed = range_confirmed & (secondary_count >= 2)
    data["compression_score"] = np.where(confirmed, raw_score.clip(0.0, 1.0), 0.0)

    warming = (
        data["range_component"].isna()
        | data["atr_component"].isna()
        | data["rv_component"].isna()
    )
    data.loc[warming, "compression_score"] = np.nan
    data["compression_state"] = (
        data["compression_score"] >= compression_threshold
    ).astype(int)
    data.loc[data["compression_score"].isna(), "compression_state"] = 0
    data["compression_persistence"] = data["compression_state"].rolling(
        persistence_window, min_periods=max(3, persistence_window // 3)
    ).mean()

    return data, list(FEATURE_NAMES)
