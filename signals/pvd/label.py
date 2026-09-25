# market_analysis/signals/pvd/label.py
"""
PVD outcome labeling -- TREND_EXHAUSTION signals only.

For each bar flagged as UP_TREND_EXHAUSTION or DOWN_TREND_EXHAUSTION:

    ATR     = pvd_atr[t]  (pre-computed by calculate_pvd)
    distance = ATR * PVD_OUTCOME_ATR_MULTIPLIER

    UP_TREND_EXHAUSTION   (price moved UP abnormally -- expect DOWN reversal):
        target        = Close[t] - distance
        failure_level = Close[t] + distance

    DOWN_TREND_EXHAUSTION (price moved DOWN abnormally -- expect UP reversal):
        target        = Close[t] + distance
        failure_level = Close[t] - distance

Forward scan using CLOSE ONLY over PVD_OUTCOME_HORIZON_DAYS daily bars:

    target reached first        -> outcome = "SUCCESS",   pvd_success = 1.0
    failure_level reached first -> outcome = "FAILURE",   pvd_success = 0.0
    neither within horizon      -> outcome = "UNRESOLVED", pvd_success = NaN

PRICE_EXPANSION signals are NOT labeled (pvd_success = NaN, outcome = "").

Parameters fall back to config.PVD when None.

Output columns added:
    pvd_outcome_target   float  -- target price level (NaN if not an exhaustion)
    pvd_outcome_stop     float  -- failure level     (NaN if not an exhaustion)
    pvd_success          float  -- 1 / 0 / NaN
    pvd_outcome          str    -- "SUCCESS" | "FAILURE" | "UNRESOLVED" | ""
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...config import PVD

_EXHAUSTION_SIGNALS = {"UP_TREND_EXHAUSTION", "DOWN_TREND_EXHAUSTION"}


def label_pvd(
    df: pd.DataFrame,
    atr_multiplier:   float | None = None,
    horizon_days:     int   | None = None,
) -> pd.DataFrame:
    """
    Add outcome labels to PVD TREND_EXHAUSTION signals.

    Must be called on the output of calculate_pvd() (which adds pvd_signal
    and pvd_atr columns).

    Parameters
    ----------
    df              : Output of calculate_pvd().
    atr_multiplier  : ATR multiplier for target/failure distance.
                      Falls back to config.PVD["outcome_atr_multiplier"].
    horizon_days    : Max daily bars to scan for outcome.
                      Falls back to config.PVD["outcome_horizon_days"].

    Returns
    -------
    df with additional columns:
        pvd_outcome_target  float
        pvd_outcome_stop    float
        pvd_success         float  (1 / 0 / NaN)
        pvd_outcome         str
    """
    atr_mult = float(
        atr_multiplier if atr_multiplier is not None
        else PVD["outcome_atr_multiplier"]
    )
    horizon = int(
        horizon_days if horizon_days is not None
        else PVD["outcome_horizon_days"]
    )

    df = df.copy().sort_values("Date").reset_index(drop=True)
    n  = len(df)

    def _get(name: str) -> np.ndarray:
        for c in df.columns:
            if c.lower() == name.lower():
                return df[c].values
        raise KeyError(f"Column '{name}' not found. Available: {list(df.columns)}")

    close_   = _get("close")
    signal_  = _get("pvd_signal")
    atr_     = _get("pvd_atr")

    target_out  = np.full(n, np.nan)
    stop_out    = np.full(n, np.nan)
    success_out = np.full(n, np.nan)
    outcome_out = [""] * n

    for i in range(n):
        sig = str(signal_[i]).strip()
        if sig not in _EXHAUSTION_SIGNALS:
            continue   # PRICE_EXPANSION or NORMAL -- skip

        atr_val = float(atr_[i])
        if np.isnan(atr_val) or atr_val <= 0:
            continue   # ATR not yet warmed up -- skip

        c0       = float(close_[i])
        distance = atr_val * atr_mult

        if sig == "UP_TREND_EXHAUSTION":
            # price was abnormally UP -- expect DOWN reversal
            target        = c0 - distance
            failure_level = c0 + distance
        else:
            # DOWN_TREND_EXHAUSTION -- expect UP reversal
            target        = c0 + distance
            failure_level = c0 - distance

        target_out[i] = round(target, 8)
        stop_out[i]   = round(failure_level, 8)

        # -- not enough forward bars to evaluate (censored) --------------------
        if (i + horizon) >= n:
            outcome_out[i] = "UNRESOLVED"
            continue

        # -- forward scan using Close only ------------------------------------
        final_outcome = "UNRESOLVED"
        final_success = np.nan

        for offset in range(1, horizon + 1):
            j = i + offset
            if j >= n:
                break
            c_j = float(close_[j])

            if sig == "UP_TREND_EXHAUSTION":
                target_hit  = c_j <= target
                failure_hit = c_j >= failure_level
            else:
                target_hit  = c_j >= target
                failure_hit = c_j <= failure_level

            if target_hit and failure_hit:
                # same bar -- use whichever is closer to c0 (conservative)
                dist_to_target  = abs(c_j - target)
                dist_to_failure = abs(c_j - failure_level)
                if dist_to_target <= dist_to_failure:
                    final_outcome = "SUCCESS"
                    final_success = 1.0
                else:
                    final_outcome = "FAILURE"
                    final_success = 0.0
                break
            elif target_hit:
                final_outcome = "SUCCESS"
                final_success = 1.0
                break
            elif failure_hit:
                final_outcome = "FAILURE"
                final_success = 0.0
                break
            # else: neither -- continue

        success_out[i] = final_success
        outcome_out[i] = final_outcome

    df["pvd_outcome_target"] = target_out
    df["pvd_outcome_stop"]   = stop_out
    df["pvd_success"]        = success_out
    df["pvd_outcome"]        = outcome_out
    return df
