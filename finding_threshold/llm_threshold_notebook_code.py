"""Copy these cells into a notebook after filling GROQ_API_KEYS in config.py.

Runs live data refresh and paid/provider-limited Groq requests when executed.
Each product produces one GPT-OSS CSV and one Qwen CSV, with separate rows for
each spread/jumbo. No backtest is performed here.
"""

# %% Imports: usable even when the notebook starts inside finding_threshold.
import importlib
import sys
from pathlib import Path

import numpy as np
from IPython.display import display

PROJECT_ROOT = Path(
    r"C:\Users\shresth.shankhdhar\OneDrive - hertshtengroup.com\Documents\market_analysis"
)
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

# Reload all three modules so edited configuration applies in an existing kernel.
config = importlib.import_module("market_analysis.config")
importlib.reload(config)
analysis = importlib.import_module("market_analysis.finding_threshold.expected_profit_analysis")
importlib.reload(analysis)
selection = importlib.import_module("market_analysis.finding_threshold.llm_threshold_selection")
importlib.reload(selection)

# %% Editable sweeps. Positive X controls BOTH volume tails; signed Y is the
# ceiling for z-score of ABSOLUTE price response. Change ranges as you prefer.
CANDLE_LENGTHS = [1, 2, 3, 5]
VOLUME_Z_SWEEP = np.round(np.arange(0.5, 3.01, 0.25), 4)
PRICE_Z_SWEEP = np.round(np.arange(-2.0, 0.51, 0.25), 4)


def select_product_thresholds(product, update=False):
    """Normally fill missing selections; explicit update refreshes every contract."""
    runner = (selection.update_llm_threshold_selection if update else
              selection.run_llm_threshold_selection)
    result = runner(
        product,
        candle_lengths=CANDLE_LENGTHS,
        mode="combined",
        delta_z_grid=VOLUME_Z_SWEEP,
        price_z_grid=PRICE_Z_SWEEP,
        front_n=config.FRONT_N,
    )
    for model_name in ("gpt_oss", "qwen"):
        print(f"{product} / {model_name}: {result['csv_paths'][model_name]}")
        display(result["tables"][model_name])
    return result


def update_product_thresholds(product):
    """Fetch/reanalyze ALL front-N spreads/jumbos and rerun BOTH models."""
    return select_product_thresholds(product, update=True)


# %% Run BO independently. Re-running reuses successes, even on another day.
bo_results = select_product_thresholds("BO")

# %% Run SM independently.
sm_results = select_product_thresholds("SM")

# %% Run these explicitly only when you want fresh thresholds for all contracts.
# bo_results = update_product_thresholds("BO")
# sm_results = update_product_thresholds("SM")
