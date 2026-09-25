# market_analysis/finding_threshold/absorption_threshold.py
"""
Absorption score threshold analysis.

Entry point for the notebook:
    from finding_threshold.absorption_threshold import run_absorption_threshold_analysis

    run_absorption_threshold_analysis(
        product_code   = "SM",
        timeframe      = "1d",          # "1d" | "1h"
        months_back    = 9,
        score_grid     = np.arange(0.10, 1.0, 0.05),
        front_n        = 5,
    )

What it does per instrument (spread × 5  +  jumbo × 5):
  1. Fetch 1D or 1H OHLC via project OHLC helpers.
  2. Run calculate_absorption() on the raw bars.
  3. Run label_absorption() (1d) or label_absorption_hourly() (1h)
     with the labeling params from config.ABSORPTION / ABSORPTION_1H.
  4. Sweep absorption_strength thresholds across score_grid, producing
     the full Precision / Recall / F1 / MCC / Lift table per instrument.
  5. Display a distribution plot: SUCCESS / FAILURE count vs score bucket.

Both BUY and SELL rows are pooled into a single sweep table (same convention
as the reference notebook).  The distribution plot shows both directions in
a single 3-panel chart (Success | Failure | Overlay).
"""

from __future__ import annotations

import sys
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython.display import display

# ── project root ──────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── project imports ───────────────────────────────────────────────────────────
from market_analysis.config import (
    PRODUCTS, FRONT_N,
    ABSORPTION as ABS_CFG,
    ABSORPTION_1H as ABS_1H_CFG,
)
from market_analysis.utils.contracts import get_all_front_instruments
from market_analysis.data.ohlc import (
    fetch_ohlc_1d, fetch_ohlc_1h, getData_OHLC_v2,
)
from market_analysis.signals.absorption.calculate import calculate_absorption
from market_analysis.signals.absorption.label     import label_absorption
from market_analysis.signals.absorption.label_hourly import label_absorption_hourly
from market_analysis.utils.metrics import _block_bootstrap_ci

# ── colours ───────────────────────────────────────────────────────────────────
_C_SUCCESS = "seagreen"
_C_FAILURE = "tomato"
_C_SWEEP   = "steelblue"


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _months_back_start(months: int) -> str:
    today = datetime.date.today()
    year  = today.year
    month = today.month - months
    while month <= 0:
        month += 12
        year  -= 1
    import calendar
    day = min(today.day, calendar.monthrange(year, month)[1])
    return datetime.date(year, month, day).strftime("%Y-%m-%d")


def _make_threshold_table(
    labeled_df     : pd.DataFrame,
    score_col      : str,
    success_col    : str,
    score_grid     : np.ndarray,
    min_n          : int   = 8,
    n_bootstrap    : int   = 500,
    random_seed    : int   = 42,
    forward_bars   : int   = 10,
) -> tuple[pd.DataFrame, float]:
    """Sweep score_grid and return (table_df, base_rate)."""
    valid = labeled_df.dropna(subset=[success_col]).reset_index(drop=True)
    if len(valid) == 0:
        return pd.DataFrame(), np.nan

    base_rate  = float(valid[success_col].mean())
    block_len  = forward_bars + 1
    scores_arr = valid[score_col].values
    labels_arr = valid[success_col].values

    rows = []
    for T in score_grid:
        sig = scores_arr >= T
        lab = labels_arr.astype(int)

        tp  = int(( sig & (lab == 1)).sum())
        fp  = int(( sig & (lab == 0)).sum())
        fn  = int((~sig & (lab == 1)).sum())
        tn  = int((~sig & (lab == 0)).sum())
        ns  = tp + fp

        prec = tp / ns          if ns > 0                else np.nan
        rec  = tp / (tp + fn)   if (tp + fn) > 0         else np.nan
        f1   = (2 * prec * rec / (prec + rec)
                if (np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0)
                else np.nan)
        lift = prec / base_rate if (np.isfinite(prec) and base_rate > 0) else np.nan

        try:
            from sklearn.metrics import matthews_corrcoef
            mcc = (matthews_corrcoef(lab, sig.astype(int))
                   if sig.sum() > 0 and (~sig).sum() > 0 else np.nan)
        except Exception:
            mcc = np.nan

        ci     = _block_bootstrap_ci(sig, labels_arr, n_bootstrap, block_len, random_seed)
        ci_str = (f"[{ci[0]:.3f}, {ci[1]:.3f}]"
                  if np.isfinite(ci[0]) else "n/a")

        rows.append({
            "Threshold"   : round(T, 3),
            "N"           : ns,
            "Precision"   : prec,
            "Prec CI 95%" : ci_str,
            "Recall"      : rec,
            "F1"          : f1,
            "MCC"         : mcc,
            "Lift"        : lift,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Low N"       : ns < min_n,
        })

    return pd.DataFrame(rows), base_rate


def _styled_table(table: pd.DataFrame, caption: str = "") -> object:
    float_cols = {c: "{:.3f}" for c in table.columns if table[c].dtype == float}
    float_cols["Threshold"] = "{:.3f}"
    return (
        table.style
        .format(float_cols, na_rep="—")
        .apply(
            lambda row: ["background-color: #fff3cd" if row["Low N"] else ""] * len(row),
            axis=1,
        )
        .set_caption(caption)
    )


def _distribution_plot(
    labeled_df  : pd.DataFrame,
    score_col   : str,
    success_col : str,
    instrument  : str,
    timeframe   : str,
    tick_size   : float,
) -> plt.Figure:
    """
    3-panel distribution: Success histogram | Failure histogram | Density overlay.
    Buckets of width 0.05 across [0, 1].
    """
    valid   = labeled_df.dropna(subset=[success_col])
    n_valid = len(valid)
    if n_valid == 0:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, "No valid labelled rows", ha="center", va="center",
                transform=ax.transAxes, fontsize=13)
        ax.set_title(f"{instrument} [{timeframe}] — no data")
        return fig

    n_succ = int((valid[success_col] == 1.0).sum())
    n_fail = int((valid[success_col] == 0.0).sum())
    base   = n_succ / n_valid if n_valid > 0 else np.nan

    s_scores = valid.loc[valid[success_col] == 1.0, score_col]
    f_scores = valid.loc[valid[success_col] == 0.0, score_col]
    bins = np.arange(0, 1.05, 0.05)

    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig)
    ax1, ax2, ax3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

    for ax, data, label, color in [
        (ax1, s_scores, f"Success  n={n_succ}",  _C_SUCCESS),
        (ax2, f_scores, f"Failure  n={n_fail}", _C_FAILURE),
    ]:
        ax.hist(data, bins=bins, color=color, edgecolor="white", alpha=0.85)
        ax.set_title(label, fontweight="bold", fontsize=11)
        ax.set_xlabel("Absorption Strength")
        ax.set_ylabel("Count")
        ax.set_xlim([0, 1])
        ax.grid(axis="y", alpha=0.3)

    # overlay / density
    ax3.hist(s_scores, bins=bins, color=_C_SUCCESS, edgecolor="white",
             alpha=0.55, density=True, label=f"Success n={n_succ}")
    ax3.hist(f_scores, bins=bins, color=_C_FAILURE, edgecolor="white",
             alpha=0.55, density=True, label=f"Failure n={n_fail}")
    ax3.set_title("Overlay (density)", fontweight="bold", fontsize=11)
    ax3.set_xlabel("Absorption Strength")
    ax3.set_ylabel("Density")
    ax3.set_xlim([0, 1])
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    fig.suptitle(
        f"{instrument}  [{timeframe}]  —  Absorption Score Distribution by Outcome\n"
        f"Valid={n_valid}  Success={n_succ}  Failure={n_fail}  "
        f"Base rate={base:.3f}  tick_size={tick_size}",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    return fig


def _print_instrument_header(
    instrument   : str,
    structure    : str,
    product_code : str,
    timeframe    : str,
    start_date   : str,
    end_date     : str,
    cfg          : dict,
    tick_size    : float,
    score_grid   : np.ndarray,
    n_rows       : int,
) -> None:
    sep = "=" * 72
    fwd_key = "forward_bars" if "forward_bars" in cfg else "forward_days"
    print(f"\n{sep}")
    print(f"  INSTRUMENT : {instrument}  ({structure.upper()})")
    print(f"  PRODUCT    : {product_code}   TIMEFRAME : {timeframe}")
    print(f"  DATA RANGE : {start_date}  →  {end_date}   ({n_rows} bars)")
    print(f"  ── Absorption params ──────────────────────────────────────────")
    print(f"  window              : {cfg.get('window', 'n/a')}")
    print(f"  required_move_ticks : {cfg.get('required_move_ticks', 'n/a')}")
    print(f"  sl_lookback         : {cfg.get('sl_lookback', 'n/a')}")
    print(f"  forward             : {cfg.get(fwd_key, 'n/a')} {'bars' if '1h' in timeframe else 'days'}")
    print(f"  tick_size           : {tick_size}")
    print(f"  ── Score threshold sweep ──────────────────────────────────────")
    print(f"  score_grid          : {score_grid[0]:.2f} → {score_grid[-1]:.2f}  "
          f"({len(score_grid)} steps)")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_absorption_threshold_analysis(
    product_code     : str,
    timeframe        : str        = "1d",
    months_back      : int        = 9,
    score_grid       : np.ndarray | None = None,
    front_n          : int        = 5,
    min_signal_count : int        = 8,
    n_bootstrap      : int        = 500,
    random_seed      : int        = 42,
    show_plots       : bool       = True,
) -> dict[str, dict]:
    """
    Run absorption score threshold analysis for all spreads and jumbos.

    Parameters
    ----------
    product_code     : e.g. "SM"
    timeframe        : "1d" for daily bars, "1h" for hourly bars.
    months_back      : How many calendar months of history to pull.
    score_grid       : Array of absorption_strength thresholds to sweep,
                       e.g. np.arange(0.10, 1.0, 0.05).
                       Defaults to config values.
    front_n          : Number of spreads and jumbos each.
    min_signal_count : N below this is flagged "Low N".
    n_bootstrap      : Bootstrap replications for precision CI.
    random_seed      : RNG seed.
    show_plots       : Display distribution plots inline.

    Returns
    -------
    results : dict keyed by instrument string, each containing:
        "labeled_df"  : fully labelled DataFrame
        "sweep_table" : threshold sweep DataFrame
        "base_rate"   : success base rate
    """
    is_1h = timeframe == "1h"
    cfg   = ABS_1H_CFG if is_1h else ABS_CFG

    if score_grid is None:
        score_grid = np.round(
            np.arange(
                cfg["threshold_grid_start"],
                cfg["threshold_grid_stop"],
                cfg["threshold_grid_step"],
            ),
            3,
        )

    start_date = _months_back_start(months_back)
    end_date   = datetime.date.today().strftime("%Y-%m-%d")

    tick_size = PRODUCTS[product_code]["tick_size"]

    all_instruments = get_all_front_instruments(
        product_code = product_code,
        front_n      = front_n,
    )

    results: dict[str, dict] = {}

    for structure, instruments in all_instruments.items():
        print(f"\n{'#' * 72}")
        print(f"#  STRUCTURE : {structure.upper()}  —  {product_code}  [{timeframe}]")
        print(f"{'#' * 72}")

        for instrument in instruments:
            # ── 1. Fetch OHLC ─────────────────────────────────────────────
            if is_1h:
                df_raw = fetch_ohlc_1h(
                    instrument      = instrument,
                    product_code    = product_code,
                    getData_OHLC_v2 = getData_OHLC_v2,
                    start_date      = start_date,
                    end_date        = end_date,
                )
            else:
                df_raw = fetch_ohlc_1d(
                    instrument      = instrument,
                    product_code    = product_code,
                    getData_OHLC_v2 = getData_OHLC_v2,
                    start_date      = start_date,
                    end_date        = end_date,
                )

            if df_raw is None or len(df_raw) < 30:
                print(f"\n  [SKIP] {instrument} — insufficient data "
                      f"({0 if df_raw is None else len(df_raw)} bars)")
                continue

            # ── 2. Calculate absorption scores ────────────────────────────
            df_abs = calculate_absorption(
                df               = df_raw,
                window           = cfg["window"],
                tick_size        = tick_size,
                strength_threshold = cfg["strength_threshold"],
            )

            # ── 3. Label outcomes ─────────────────────────────────────────
            if is_1h:
                df_labeled = label_absorption_hourly(
                    df                   = df_abs,
                    tick                 = tick_size,
                    required_move_ticks  = cfg["required_move_ticks"],
                    sl_lookback          = cfg["sl_lookback"],
                    forward_bars         = cfg["forward_bars"],
                )
                success_col  = "sl_success"
                forward_bars = cfg["forward_bars"]
            else:
                # 1d label_absorption uses the profit-first logic from the
                # reference code — adapts to your existing label module.
                # If your label module is named differently adjust here.
                df_labeled = label_absorption(
                    df                   = df_abs,
                    tick                 = tick_size,
                    required_move_ticks  = cfg["required_move_ticks"],
                    sl_lookback          = cfg["sl_lookback"],
                    forward_days         = cfg["forward_days"],
                )
                success_col  = "sl_success"
                forward_bars = cfg["forward_days"]

            _print_instrument_header(
                instrument   = instrument,
                structure    = structure,
                product_code = product_code,
                timeframe    = timeframe,
                start_date   = start_date,
                end_date     = end_date,
                cfg          = cfg,
                tick_size    = tick_size,
                score_grid   = score_grid,
                n_rows       = len(df_raw),
            )

            # ── 4. Outcome summary ────────────────────────────────────────
            valid_rows = df_labeled.dropna(subset=[success_col])
            n_total    = len(df_labeled)
            n_censored = int(df_labeled[success_col].isna().sum())
            n_valid    = len(valid_rows)
            n_succ     = int((valid_rows[success_col] == 1.0).sum())
            n_fail     = int((valid_rows[success_col] == 0.0).sum())

            print(f"\n  Total bars          : {n_total}")
            print(f"  Censored (NaN)      : {n_censored}")
            print(f"  Valid rows          : {n_valid}")
            print(f"  ├── Success  (1)    : {n_succ}   "
                  f"({100*n_succ/n_valid:.1f}%)" if n_valid else "  ├── Success  (1)    : 0")
            print(f"  └── Failure  (0)    : {n_fail}   "
                  f"({100*n_fail/n_valid:.1f}%)" if n_valid else "  └── Failure  (0)    : 0")

            if "stop_hit" in df_labeled.columns:
                n_stopped = int((df_labeled["stop_hit"] == 1.0).sum())
                req_pts   = cfg["required_move_ticks"] * tick_size
                print(f"       ├── stop hit          : {n_stopped}")
                print(f"       └── insufficient move : {n_fail - n_stopped}  "
                      f"(< {cfg['required_move_ticks']} ticks = {req_pts:.3f} pts)")
            elif "stop_hit_first" in df_labeled.columns:
                n_stopped = int((df_labeled["stop_hit_first"] == 1.0).sum())
                req_pts   = cfg["required_move_ticks"] * tick_size
                print(f"       ├── stop hit first    : {n_stopped}")
                print(f"       └── neither fired     : {n_fail - n_stopped}")

            # direction breakdown
            dir_col = None
            for c in df_labeled.columns:
                if c.lower() == "absorption_direction":
                    dir_col = c
                    break
            if dir_col:
                for d in ["BUY", "SELL"]:
                    sub = df_labeled[
                        df_labeled[dir_col].str.upper() == d
                    ].dropna(subset=[success_col])
                    if len(sub):
                        sr = (sub[success_col] == 1.0).mean()
                        print(f"  {d}  valid={len(sub)}  "
                              f"success_rate={sr:.3f}")

            # ── 5. Threshold sweep ────────────────────────────────────────
            sweep, base_rate = _make_threshold_table(
                labeled_df   = df_labeled,
                score_col    = "absorption_strength",
                success_col  = success_col,
                score_grid   = score_grid,
                min_n        = min_signal_count,
                n_bootstrap  = n_bootstrap,
                random_seed  = random_seed,
                forward_bars = forward_bars,
            )

            print(f"\n  ── Combined BUY + SELL threshold sweep  "
                  f"(base rate = {base_rate:.3f if np.isfinite(base_rate) else 'n/a'})")

            if sweep.empty:
                print("     No valid labelled rows for sweep.")
            else:
                req_ticks = cfg["required_move_ticks"]
                req_pts   = req_ticks * tick_size
                fwd_key   = "forward_bars" if is_1h else "forward_days"
                caption   = (
                    f"{instrument}  [{timeframe}]  |  "
                    f"required_move={req_ticks}×tick ({req_pts:.3f} pts)  |  "
                    f"sl_lookback={cfg['sl_lookback']}  |  "
                    f"{fwd_key}={cfg[fwd_key]}  |  "
                    f"months_back={months_back}  |  "
                    f"base rate={base_rate:.3f}  |  "
                    f"Yellow = N < {min_signal_count}"
                )
                display(_styled_table(sweep, caption))

            # ── 6. Distribution plot ──────────────────────────────────────
            if show_plots and not sweep.empty:
                fig = _distribution_plot(
                    labeled_df  = df_labeled,
                    score_col   = "absorption_strength",
                    success_col = success_col,
                    instrument  = instrument,
                    timeframe   = timeframe,
                    tick_size   = tick_size,
                )
                plt.show()
                plt.close(fig)

            # ── store ─────────────────────────────────────────────────────
            results[instrument] = {
                "labeled_df" : df_labeled,
                "sweep_table": sweep,
                "base_rate"  : base_rate,
            }

    return results
