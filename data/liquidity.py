
# market_analysis/data/liquidity.py
"""
Liquidity qualification layer for commodity spreads and outrights.

Phase 1 & Phase 2 Implementation:
- Default analytical universe: Front-2 spreads for softs.
- Pipeline monitoring universe: Front-3 to Front-5 spreads.
- Metrics calculated:
    - 20-day median volume (MedVol_20d)
    - 20-day relative volume (RVol_20d = 5d mean / 20d median)
    - Zero-volume days in last 20 days (ZeroVol_20d)
    - Transaction count / proxy (where available, else daily volume delta / EOD notice)
    - Bid-Ask spread / proxy (where available, else EOD tick/range notice)
- Status classification: ELIGIBLE, MONITOR, ILLIQUID
"""

from __future__ import annotations

import datetime
from typing import Mapping, Sequence
import numpy as np
import pandas as pd

from ..config import DB_CONFIG, PRODUCTS
from ..utils.contracts import get_all_front_instruments
from .ohlc import fetch_ohlc_from_db, getData_OHLC_v2, fetch_ohlc_1d

# ─────────────────────────────────────────────────────────────────────────────
# PROVISIONAL & CALIBRATED LIQUIDITY THRESHOLDS BY COMMODITY
# ─────────────────────────────────────────────────────────────────────────────
LIQUIDITY_THRESHOLDS: dict[str, dict] = {
    "DEFAULT": {
        "min_median_vol_eligible": 500,
        "min_median_vol_monitor" : 100,
        "max_zero_vol_days"      : 1,
        "min_bars"               : 30,
    },
    "SB": {  # Sugar No. 11
        "min_median_vol_eligible": 1000,
        "min_median_vol_monitor" : 250,
        "max_zero_vol_days"      : 1,
        "min_bars"               : 30,
    },
    "KC": {  # Coffee "C"
        "min_median_vol_eligible": 300,
        "min_median_vol_monitor" : 80,
        "max_zero_vol_days"      : 1,
        "min_bars"               : 30,
    },
    "CC": {  # Cocoa
        "min_median_vol_eligible": 500,
        "min_median_vol_monitor" : 100,
        "max_zero_vol_days"      : 1,
        "min_bars"               : 30,
    },
    "CT": {  # Cotton No. 2
        "min_median_vol_eligible": 500,
        "min_median_vol_monitor" : 50,
        "max_zero_vol_days"      : 1,
        "min_bars"               : 20,
    },
    "RC": {  # Robusta Coffee
        "min_median_vol_eligible": 500,
        "min_median_vol_monitor" : 100,
        "max_zero_vol_days"      : 1,
        "min_bars"               : 20,
    },
    "SM": {  # Soybean Meal
        "min_median_vol_eligible": 1000,
        "min_median_vol_monitor" : 250,
        "max_zero_vol_days"      : 0,
        "min_bars"               : 30,
    },
    "BO": {  # Soybean Oil
        "min_median_vol_eligible": 1000,
        "min_median_vol_monitor" : 250,
        "max_zero_vol_days"      : 0,
        "min_bars"               : 30,
    },
}


def qualify_instrument_liquidity(
    df: pd.DataFrame | None,
    product_code: str,
    instrument: str,
    rank: int = 1,
    analytical_max_rank: int = 2,
    custom_thresholds: Mapping | None = None,
) -> dict:
    """
    Evaluate liquidity metrics and classify a spread into ELIGIBLE, MONITOR, or ILLIQUID.
    """
    thresholds = {
        **LIQUIDITY_THRESHOLDS.get(product_code, LIQUIDITY_THRESHOLDS["DEFAULT"]),
        **(custom_thresholds or {}),
    }

    universe = "Front-2 (Analytical)" if rank <= analytical_max_rank else "Front-3..5 (Pipeline)"

    if df is None or len(df) == 0:
        return {
            "product": product_code,
            "rank": f"Front-{rank}",
            "instrument": instrument,
            "universe": universe,
            "bars": 0,
            "med_vol_20d": 0.0,
            "rvol_20d": 0.0,
            "zero_vol_days_20d": 20,
            "tx_count": "N/A (no data)",
            "bid_ask_spread": "N/A (no data)",
            "status": "ILLIQUID",
            "reason": "No data returned from database or API",
            "is_eligible": False,
        }

    n_bars = len(df)
    v = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0)

    # 20-day window metrics
    v20 = v.tail(20)
    med_20d = float(v20.median()) if len(v20) > 0 else 0.0
    mean_20d = float(v20.mean()) if len(v20) > 0 else 0.0
    zero_days_20d = int((v20 == 0).sum())

    # Relative volume (recent 5d average / 20d median)
    v5_mean = float(v.tail(5).mean()) if len(v) >= 5 else float(v.mean())
    rvol_20d = float(v5_mean / med_20d) if med_20d > 0 else 0.0

    # Transaction count & Bid-Ask spread proxies where tick-level data is unavailable in daily DB
    # If buy_volume and sell_volume exist, we measure directional volume split
    if "buyvolume" in df.columns and "sellvolume" in df.columns:
        recent_delta = float((df["buyvolume"].tail(5) - df["sellvolume"].tail(5)).abs().mean())
        tx_proxy = f"Daily agg (delta={recent_delta:.0f})"
    else:
        tx_proxy = "N/A (daily EOD)"

    # Effective spread proxy: (High - Low) relative to tick size
    tick_size = PRODUCTS.get(product_code, {}).get("tick_size", 1.0)
    if "high" in df.columns and "low" in df.columns:
        mean_range_ticks = float(((df["high"].tail(20) - df["low"].tail(20)) / tick_size).median())
        ba_proxy = f"Range Med: {mean_range_ticks:.1f} ticks"
    else:
        ba_proxy = "N/A (daily EOD)"

    # Classification logic
    min_bars = thresholds["min_bars"]
    min_med_elig = thresholds["min_median_vol_eligible"]
    min_med_mon = thresholds["min_median_vol_monitor"]
    max_zero_days = thresholds["max_zero_vol_days"]

    if n_bars < min_bars:
        if med_20d >= min_med_elig:
            status = "MONITOR"
            reason = f"Short history ({n_bars} < {min_bars} bars) despite liquid volume (20d MedVol={med_20d:.0f})"
        else:
            status = "ILLIQUID"
            reason = f"Insufficient history ({n_bars} < {min_bars} bars) & low volume"
    elif zero_days_20d > max_zero_days:
        if med_20d >= min_med_mon:
            status = "MONITOR"
            reason = f"Intermittent trading ({zero_days_20d} zero-vol days in trailing 20d)"
        else:
            status = "ILLIQUID"
            reason = f"High zero-volume days ({zero_days_20d}/20 days dead)"
    elif med_20d >= min_med_elig:
        status = "ELIGIBLE"
        reason = f"Liquid (20d MedVol={med_20d:.0f} >= {min_med_elig})"
    elif med_20d >= min_med_mon:
        status = "MONITOR"
        reason = f"Moderate volume ({min_med_mon} <= MedVol={med_20d:.0f} < {min_med_elig})"
    else:
        status = "ILLIQUID"
        reason = f"Low volume (MedVol={med_20d:.0f} < {min_med_mon})"

    return {
        "product": product_code,
        "rank": f"Front-{rank}",
        "instrument": instrument,
        "universe": universe,
        "bars": n_bars,
        "med_vol_20d": med_20d,
        "rvol_20d": round(rvol_20d, 2),
        "zero_vol_days_20d": zero_days_20d,
        "tx_count": tx_proxy,
        "bid_ask_spread": ba_proxy,
        "status": status,
        "reason": reason,
        "is_eligible": (status == "ELIGIBLE"),
    }


def evaluate_softs_liquidity_universe(
    products: Sequence[str] = ("SB", "KC", "CC", "CT", "RC"),
    front_n: int = 5,
    analytical_max_rank: int = 2,
    start_date: str = "2024-01-01",
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Evaluate and qualify liquidity across all softs contracts from Front-1 to Front-N.
    """
    if end_date is None:
        end_date = datetime.date.today().strftime("%Y-%m-%d")

    records = []
    for prod in products:
        all_front = get_all_front_instruments(prod, front_n=front_n)
        spreads = all_front.get("spread", [])
        for rank, inst in enumerate(spreads, 1):
            df = fetch_ohlc_from_db(inst, start_date, end_date)
            # fallback to internal API if DB empty
            if df is None or len(df) == 0:
                df = fetch_ohlc_1d(inst, prod, getData_OHLC_v2, start_date, end_date)

            qual = qualify_instrument_liquidity(
                df=df,
                product_code=prod,
                instrument=inst,
                rank=rank,
                analytical_max_rank=analytical_max_rank,
            )
            records.append(qual)

    return pd.DataFrame(records)


def style_liquidity_table(df: pd.DataFrame, caption: str = "Liquidity Qualification Matrix") -> object:
    """
    Format and colorize the liquidity qualification matrix for Jupyter display.
    """
    def highlight_status(row):
        st = row.get("status", "")
        if st == "ELIGIBLE":
            color = "#d4edda; color: #155724; font-weight: bold;"
        elif st == "MONITOR":
            color = "#fff3cd; color: #856404; font-weight: bold;"
        else:
            color = "#f8d7da; color: #721c24; font-weight: bold;"
        return [f"background-color: {color}" if c == "status" else "" for c in row.index]

    styled = (
        df.style
        .format({
            "med_vol_20d": "{:,.0f}",
            "rvol_20d": "{:.2f}",
            "bars": "{:,d}",
            "zero_vol_days_20d": "{:d}",
        })
        .apply(highlight_status, axis=1)
        .set_caption(caption)
    )
    return styled
