# market_analysis/config.py
"""
Central config — edit ONLY this file when changing paths, params, or products.
"""

from pathlib import Path
import datetime

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

# Folder where raw CSVs fetched from the OHLC API are cached per-instrument.
# Sub-folder per product: DATA_DIR / "sm" / "SMZ26-SMF27_1d.csv" etc.
DATA_DIR = Path(
    r"C:\Users\shresth.shankhdhar\OneDrive - hertshtengroup.com"
    r"\Documents\market_analysis\data"
)

# All output CSVs and PNG plots go here.
# Sub-folders are created automatically: OUTPUT_DIR / "sm" / "1d" / ...
OUTPUT_DIR = Path(
    r"C:\Users\shresth.shankhdhar\OneDrive - hertshtengroup.com"
    r"\Documents\market_analysis\output"
)

# ─────────────────────────────────────────────────────────────────────────────
# CLEAN PREVIOUS DATA
# Set to True to delete cached CSVs and re-fetch + recalculate from scratch.
# ─────────────────────────────────────────────────────────────────────────────
CLEAN_PREVIOUS_DATA: bool = False

# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE DATE — used to compute front-N contracts
# ─────────────────────────────────────────────────────────────────────────────
REFERENCE_DATE: datetime.date = datetime.date.today()

# ─────────────────────────────────────────────────────────────────────────────
# FRONT-N — how many spreads / jumbos per structure type
# ─────────────────────────────────────────────────────────────────────────────
FRONT_N: int = 5

# ─────────────────────────────────────────────────────────────────────────────
# API AUTH TOKEN
# Paste your bearer token here. Never commit this file to version control.
# ─────────────────────────────────────────────────────────────────────────────
API_TOKEN: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJCUWVPTHlzdVZnckllbmRabnJWVWpZeExTYm1xTVZjMVFDakdtUEMzd0ZBIiwidHlwZSI6InBlcnNvbmFsIiwiaWF0IjoxNzg5NTcxNTUyLCJleHAiOjI0MjAyOTE1NTIsInN1YiI6IjQxOCIsImVtYWlsIjoic2hyZXN0aC5zaGFua2hkaGFyQGhlcnRzaHRlbmdyb3VwLmNvbSIsInVzZXJuYW1lIjoic2hyZXN0aC5zaGFua2hkaGFyIn0.QSbD84uB4B2wj7mNV5jakzHkRR8L5hnwjFjEwGYLOHg"

# ─────────────────────────────────────────────────────────────────────────────
# OHLC FETCH WINDOW
# How far back to pull OHLC data from the API (for score calculation).
# ─────────────────────────────────────────────────────────────────────────────
OHLC_START_DATE: str = "2026-01-01"   # YYYY-MM-DD
OHLC_END_DATE:   str = datetime.date.today().strftime("%Y-%m-%d")

# ─────────────────────────────────────────────────────────────────────────────
# PRODUCT REGISTRY
# Add a product here; everything else picks it up automatically.
# ─────────────────────────────────────────────────────────────────────────────
PRODUCTS: dict[str, dict] = {
    "SM": {
        "traded_months"   : [1, 3, 5, 7, 8, 9, 10, 12],
        "exchange_prefix" : "ZM",    # used in Excel sheet names
        "tick_size"       : 0.1,
        "api_prefix"      : "SM",    # instrument strings: SM + month_code + year
    },
    "BO":{
        "traded_months"   : [1, 3, 5, 7, 8, 9, 10, 12],
        "exchange_prefix" : "ZL",
        "tick_size"       : 0.1,
        "api_prefix"      : "BO",
    },
    # "C": {
    #     "traded_months"   : [3, 5, 7, 9, 12],
    #     "exchange_prefix" : "ZC",
    #     "tick_size"       : 0.25,
    #     "api_prefix"      : "C",
    # },
}

# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURES
# Maps structure name -> (front_leg_offset, list[back_leg_offsets])
# For each front index i, all back offsets are generated and interleaved
# in order (i, bo) sorted by i first then bo.
# ─────────────────────────────────────────────────────────────────────────────
STRUCTURES: dict[str, tuple[int, list[int]]] = {
    "spread": (0, [1]),      # consecutive months only: i → i+1
    "jumbo" : (0, [2, 3]),   # skip-1 AND skip-2: i→i+2, i→i+3, interleaved by front
}

# ─────────────────────────────────────────────────────────────────────────────
# HORIZONS to compute and analyse
# ─────────────────────────────────────────────────────────────────────────────
HORIZONS: list[str] = ["1d", "2d", "3d", "5d"]

HORIZON_BARS: dict[str, int] = {   # how many 1D bars to roll up
    "1d": 1,
    "2d": 2,
    "3d": 3,
    "5d": 5,
}

# 1H multi-horizon bar counts (passed to _aggregate_trailing_1h)
HORIZON_BARS_1H: dict[str, int] = {
    "1h" :  1,
    "2h" :  2,
    "4h" :  4,
    "8h" :  8,
    "16h": 16,
}

# Session-break threshold for 1H aggregation
# Consecutive bars separated by more than this are treated as a new session
# (e.g. Friday close → Monday open gap).
HOURLY_SESSION_GAP: str = "4h"   # pd.Timedelta-compatible string

# ─────────────────────────────────────────────────────────────────────────────
# ABSORPTION SIGNAL PARAMS
# ─────────────────────────────────────────────────────────────────────────────
ABSORPTION: dict = {
    "window"              : 60,    # percentile lookback (bars)
    "strength_threshold"  : 0.30,  # classification threshold (not used in sweep)
    # labeling
    "required_move_ticks" : 2,     # min move for success = 2 × tick_size
    "sl_lookback"         : 10,    # past days for stop-loss level
    "forward_days"        : 10,    # days ahead to evaluate outcome
    # threshold sweep
    "threshold_grid_start": 0.10,
    "threshold_grid_stop" : 1.00,
    "threshold_grid_step" : 0.05,
    "min_signal_count"    : 8,
    "n_bootstrap"         : 500,
    "random_seed"         : 42,
}

# ─────────────────────────────────────────────────────────────────────────────
# COMPRESSION SIGNAL PARAMS
# ─────────────────────────────────────────────────────────────────────────────
COMPRESSION: dict = {
    "range_baseline"        : 20,
    "atr_period"            : 5,
    "rv_period"             : 5,
    "persistence_window"    : 20,
    "compression_threshold" : 0.30,
    # labeling
    "k"                     : 4,   # required move = k × tick_size
    "forward_days"          : 10,
    # threshold sweep
    "threshold_line"        : 0.40,
    "threshold_grid_start"  : 0.10,
    "threshold_grid_stop"   : 1.00,
    "threshold_grid_step"   : 0.05,
    "min_signal_count"      : 8,
    "n_bootstrap"           : 500,
    "random_seed"           : 42,
}

# ─────────────────────────────────────────────────────────────────────────────
# ABSORPTION 1H SIGNAL PARAMS
# Tuned for hourly bars.  Uses profit-first labeling (label_absorption_hourly).
# ─────────────────────────────────────────────────────────────────────────────
ABSORPTION_1H: dict = {
    "window"              : 120,   # percentile lookback in 1H bars (~7.5 trading days)
    "strength_threshold"  : 0.30,
    # labeling (profit-first logic — see signals/absorption/label_hourly.py)
    "required_move_ticks" : 3,     # min move = 3 × tick_size
    "sl_lookback"         : 20,    # past 1H bars for stop-loss level
    "forward_bars"        : 40,    # ≈ 2.5 trading days at 16 bars/day
    # threshold sweep
    "threshold_grid_start": 0.10,
    "threshold_grid_stop" : 1.00,
    "threshold_grid_step" : 0.05,
    "min_signal_count"    : 8,
    "n_bootstrap"         : 500,
    "random_seed"         : 42,
}

# Volume minimum for 1H bars (lower than 1D because volume is spread over bars)
VOLUME_MIN_1H: int = 500

# ─────────────────────────────────────────────────────────────────────────────
# PVD (Price-Volume Divergence) SIGNAL PARAMS
# ─────────────────────────────────────────────────────────────────────────────
PVD: dict = {
    # ---- score calculation windows ----------------------------------------
    # price_horizon_h   : bar count for price-change measurement
    # volatility_window : rolling window for 1-bar sigma (normalises price move)
    # pvd_model_window  : rolling window for OLS fit and residual z-score
    # atr_period        : bars for ATR = mean(High - Low)
    "price_horizon_h"        : 3,
    "volatility_window"      : 15,
    "pvd_model_window"       : 15,
    "atr_period"             : 10,

    # ---- signal thresholds -------------------------------------------------
    # z <= exhaustion_z_threshold  -> TREND_EXHAUSTION  (negative, e.g. -1.5)
    # z >= expansion_z_threshold   -> PRICE_EXPANSION   (positive, e.g. +1.5)
    # These replace the hardcoded ±1.5 in the original engine code.
    "exhaustion_z_threshold" : -1.5,
    "expansion_z_threshold"  :  1.5,

    # ---- outcome labeling (TREND_EXHAUSTION only) ---------------------------
    # outcome_atr_multiplier : ATR * this = distance to target / failure level
    # outcome_horizon_days   : max future daily bars to scan for outcome
    "outcome_atr_multiplier" : 1.5,
    "outcome_horizon_days"   : 10,

    # ---- threshold sweep (for notebook analysis) ----------------------------
    "threshold_grid_start"   : 0.10,
    "threshold_grid_stop"    : 1.00,
    "threshold_grid_step"    : 0.05,
    "min_signal_count"       : 8,
    "n_bootstrap"            : 500,
    "random_seed"            : 42,

    # ---- exhaustion outcome treatment ---------------------------------------
    # exhaustion_merge_unresolved:
    #   False (default) → strict 3-class: SUCCESS / FAILURE / UNRESOLVED
    #                     UNRESOLVED rows are EXCLUDED from F1/Precision/Recall.
    #   True            → UNRESOLVED is counted as SUCCESS
    #                     (senior's suggestion: horizon expired without stop hit
    #                      ≈ trend did not reverse, so we were right).
    "exhaustion_merge_unresolved" : True,
}