# Copy this complete block into a notebook cell. The existing notebook is not edited.
import sys
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

PROJECT_DIR = Path(
    r"C:\Users\shresth.shankhdhar\OneDrive - hertshtengroup.com\Documents\market_analysis"
)
if str(PROJECT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR.parent))

# Reload edited config/module even if this notebook kernel imported older versions.
config = importlib.import_module("market_analysis.config")
importlib.reload(config)
analysis = importlib.import_module("market_analysis.finding_threshold.expected_profit_analysis")
importlib.reload(analysis)

PRODUCT_CODE = "SM"                     # "SM" or "BO"
FRONT_N = 5                            # five spreads AND five jumbos
CANDLE_LENGTHS = [1, 2, 3, 5]           # any positive integer lengths
MODE = "combined"                      # "separate" + [1] for only 1D

# X displays POSITIVE magnitudes but selects BOTH signed z_delta tails.
# Y is a ceiling on z-score of ABSOLUTE price displacement, on BOTH sides.
# np.arange excludes its stop; change these ranges/steps as you like.
DELTA_Z_GRID = np.round(np.arange(1.0, 3.25, 0.25), 6)
PRICE_Z_GRID = np.round(np.arange(0.0, -1.75, -0.25), 6)

PARAMS = dict(config.ABSORPTION_EXPECTED_PROFIT)
PARAMS.update({
    "months_back": 5,                  # October run -> 1 June onward
    "baseline_window": 15,
    "baseline_min_periods": 15,        # must not exceed baseline_window
    "volume_window": 5,
    "atr_window": 5,
    "atr_period": 14,
    "stop_atr_mult": 1.5,              # configurable, fixed at signal time
    "required_move_ticks": 2,          # required profit BEFORE first stop
    "forward_days": 10,               # future RAW 1D trading bars
})

run = analysis.run_expected_profit_analysis(
    product_code=PRODUCT_CODE,
    candle_lengths=CANDLE_LENGTHS,
    mode=MODE,
    params=PARAMS,
    delta_z_grid=DELTA_Z_GRID,
    price_z_grid=PRICE_Z_GRID,
    front_n=FRONT_N,
    save_outputs=False,
)

print(f"Analysis: {run['analysis_start'].date()} to {run['analysis_end'].date()}")
if run["errors"]:
    display(pd.DataFrame(list(run["errors"].items()), columns=["Instrument", "Error"]))

# ONE heatmap per individual instrument in combined mode: five spreads plus
# five jumbos, when all instruments have data. BUY and SELL fold within each
# instrument only. No pooling across instruments or structure types.
for structure, instruments in run["universe"].items():
    print(f"\n{structure.upper()}")
    for instrument in instruments:
        result = run["instruments"].get(instrument)
        if result is None:
            continue  # The fetch/analysis error is already reported above.
        print(instrument)
        groups = list(dict.fromkeys(result["threshold_table"]["analysis_group"]))
        for group in groups:
            figure = analysis.plot_expected_profit_heatmap(result, analysis_group=group)
            plt.show()
            plt.close(figure)

        # Exact counts/payoff sizes behind this instrument's cells.
        display(result["threshold_table"][[
            "analysis_group", "delta_z_threshold", "price_z_threshold",
            "expected_profit_ticks", "n_signals", "n_resolved", "n_success", "n_failure",
            "avg_success_ticks", "avg_failure_ticks", "n_unresolved", "n_censored",
            "n_suppressed",
        ]])
