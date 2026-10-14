# market_analysis/finding_threshold/compression_threshold.py
"""
Compression score threshold analysis.

Entry point for the notebook:

    from finding_threshold.compression_threshold import run_compression_threshold_analysis

    results = run_compression_threshold_analysis(
        product_code = "SM",
        months_back  = 9,
        score_grid   = np.arange(0.10, 1.0, 0.05),   # optional, defaults from config
        front_n      = 5,
    )

What it does per instrument (spread × front_n  +  jumbo × front_n):
  1. Fetch 1D OHLC via project OHLC helpers.
  2. Run calculate_compression() on the raw bars.
  3. Run label_compression() with the labeling params from config.COMPRESSION.
  4. Sweep compression_score thresholds across score_grid producing
     the full Precision / Recall / F1 / MCC / Lift table per instrument.
  5. Display a distribution plot: SUCCESS / FAILURE count vs score bucket
     (3-panel: Success | Failure | Density overlay).

Both the sweep table and the distribution chart are identical in style
to the absorption and pvd reference notebooks.
"""

from __future__ import annotations

import sys
import datetime
import calendar
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
try:
    from IPython.display import display
except Exception:
    display = print

# -- project root on sys.path -------------------------------------------------
_HERE = Path(__file__).resolve().parent       # .../finding_threshold/
_ROOT = _HERE.parent                          # .../market_analysis/
_DOCS = _ROOT.parent                          # .../Documents/
for _p in [str(_ROOT), str(_DOCS), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import types
if "market_analysis" not in sys.modules:
    pkg = types.ModuleType("market_analysis")
    pkg.__path__ = [str(_ROOT)]
    sys.modules["market_analysis"] = pkg

# -- project imports ----------------------------------------------------------
try:
    from market_analysis.config import (
        PRODUCTS, FRONT_N,
        COMPRESSION as COMP_CFG,
        OHLC_START_DATE, OHLC_END_DATE,
    )
    from market_analysis.utils.contracts import get_all_front_instruments
    from market_analysis.data.ohlc import fetch_ohlc_1d, getData_OHLC_v2
    from market_analysis.signals.compression.calculate import calculate_compression
    from market_analysis.signals.compression.label     import label_compression
    from market_analysis.utils.metrics import _block_bootstrap_ci, wilson_score_interval
    from market_analysis.data.liquidity import qualify_instrument_liquidity, style_liquidity_table
except ImportError:
    from config import (
        PRODUCTS, FRONT_N,
        COMPRESSION as COMP_CFG,
        OHLC_START_DATE, OHLC_END_DATE,
    )
    from utils.contracts import get_all_front_instruments
    from data.ohlc import fetch_ohlc_1d, getData_OHLC_v2
    from signals.compression.calculate import calculate_compression
    from signals.compression.label     import label_compression
    from utils.metrics import _block_bootstrap_ci, wilson_score_interval
    from data.liquidity import qualify_instrument_liquidity, style_liquidity_table

# -- colours ------------------------------------------------------------------
_C_SUCCESS = "seagreen"
_C_FAILURE = "tomato"


# =============================================================================
# HELPERS
# =============================================================================

def _months_back_start(months: int) -> str:
    today = datetime.date.today()
    year, month = today.year, today.month - months
    while month <= 0:
        month += 12
        year  -= 1
    day = min(today.day, calendar.monthrange(year, month)[1])
    return datetime.date(year, month, day).strftime("%Y-%m-%d")


# =============================================================================
# THRESHOLD SWEEP TABLE
# =============================================================================

def _collapse_to_episodes(
    sig: np.ndarray,
    mode: str = "first_bar",
    min_gap: int = 1,
) -> np.ndarray:
    """
    Handle overlapping forward windows by collapsing contiguous compression runs.

    Modes:
      - 'first_bar': Trigger signal only on the entry/first day of each compression run.
      - 'non_overlapping': Enforce at least `min_gap` (forward_days) bars between signals.
      - 'all_bars': Retain all individual bar triggers.
    """
    if mode == "all_bars":
        return sig.copy()

    n = len(sig)
    ep_sig = np.zeros(n, dtype=bool)
    gap = min_gap if mode == "non_overlapping" else 1
    last_trigger = -gap - 1

    for i in range(n):
        if sig[i]:
            if (i - last_trigger) > gap:
                ep_sig[i] = True
                last_trigger = i
    return ep_sig


def _make_threshold_table(
    labeled_df  : pd.DataFrame,
    score_col   : str,
    success_col : str,
    score_grid  : np.ndarray,
    min_n       : int   = 8,
    n_bootstrap : int   = 500,
    random_seed : int   = 42,
    forward_days: int   = 10,
    episode_mode: str   = "first_bar",
) -> tuple[pd.DataFrame, float]:
    """
    Sweep score_grid and return (table_df, base_rate).

    Priority 2 Fix: Rows with zero scores are fully included in evaluating
    negative signals, specificity, true base rate, and lift.
    Priority 3 Fix: Clustered compression signals with overlapping forward windows
    are handled via episode collapsing (`episode_mode='first_bar'`).
    Priority 4 Fix: Wilson 95% Confidence Interval included as supplementary check.
    """
    valid = labeled_df.dropna(subset=[success_col]).reset_index(drop=True)
    if len(valid) == 0:
        return pd.DataFrame(), np.nan

    base_rate   = float(valid[success_col].mean())
    block_len   = forward_days + 1
    scores_arr  = valid[score_col].values
    labels_arr  = valid[success_col].values

    rows = []
    for T in score_grid:
        sig_raw = scores_arr >= T
        # Handle overlapping windows / compression episodes
        sig = _collapse_to_episodes(sig_raw, mode=episode_mode, min_gap=forward_days)
        lab = labels_arr.astype(int)

        tp  = int(( sig & (lab == 1)).sum())
        fp  = int(( sig & (lab == 0)).sum())
        fn  = int((~sig & (lab == 1)).sum())
        tn  = int((~sig & (lab == 0)).sum())
        ns  = tp + fp
        raw_bars = int(sig_raw.sum())

        prec = tp / ns          if ns > 0               else np.nan
        rec  = tp / (tp + fn)   if (tp + fn) > 0        else np.nan
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

        wilson = wilson_score_interval(tp, ns, 0.95)
        wilson_str = (f"[{wilson[0]:.3f}, {wilson[1]:.3f}]"
                      if np.isfinite(wilson[0]) else "n/a")

        rows.append({
            "Threshold"    : round(T, 3),
            "N"            : ns,
            "N (raw bars)" : raw_bars,
            "Precision"    : prec,
            "Prec CI (Boot)": ci_str,
            "Wilson CI 95%": wilson_str,
            "Recall"       : rec,
            "F1"           : f1,
            "MCC"          : mcc,
            "Lift"         : lift,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Low N"        : ns < min_n,
        })

    return pd.DataFrame(rows), base_rate


# =============================================================================
# STYLED TABLE
# =============================================================================

def _styled_table(table: pd.DataFrame, caption: str = "") -> object:
    float_cols = {c: "{:.3f}" for c in table.columns if table[c].dtype == float}
    float_cols["Threshold"] = "{:.3f}"
    return (
        table.style
        .format(float_cols, na_rep="--")
        .apply(
            lambda row: ["background-color: #fff3cd" if row["Low N"] else ""] * len(row),
            axis=1,
        )
        .background_gradient(subset=["F1"],        cmap="RdYlGn", vmin=0.3, vmax=0.7)
        .background_gradient(subset=["Precision"], cmap="Blues",  vmin=0.3, vmax=0.8)
        .set_caption(caption)
    )


# =============================================================================
# DISTRIBUTION PLOT
# =============================================================================

def _distribution_plot(
    labeled_df  : pd.DataFrame,
    score_col   : str,
    success_col : str,
    instrument  : str,
    tick_size   : float,
    k           : int,
    forward_days: int,
) -> plt.Figure:
    """
    3-panel distribution chart:
      Panel 1 -- Success histogram
      Panel 2 -- Failure histogram
      Panel 3 -- Density overlay

    Buckets of width 0.05 across [0, 1].
    A dashed red vertical line marks the config reference threshold_line.
    """
    valid   = labeled_df.dropna(subset=[success_col])
    n_valid = len(valid)

    if n_valid == 0:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, "No valid labelled rows", ha="center", va="center",
                transform=ax.transAxes, fontsize=13)
        ax.set_title(f"{instrument} -- Compression -- no labelled data")
        return fig

    n_succ = int((valid[success_col] == 1.0).sum())
    n_fail = int((valid[success_col] == 0.0).sum())
    base   = n_succ / n_valid if n_valid > 0 else np.nan

    s_scores = valid.loc[valid[success_col] == 1.0, score_col]
    f_scores = valid.loc[valid[success_col] == 0.0, score_col]
    bins     = np.arange(0, 1.05, 0.05)
    ref      = COMP_CFG.get("threshold_line", 0.40)

    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    for ax, data, label, color in [
        (ax1, s_scores, f"Success  n={n_succ}",  _C_SUCCESS),
        (ax2, f_scores, f"Failure  n={n_fail}", _C_FAILURE),
    ]:
        ax.hist(data, bins=bins, color=color, edgecolor="white", alpha=0.85)
        ax.set_title(label, fontweight="bold", fontsize=11)
        ax.set_xlabel("Compression Score")
        ax.set_ylabel("Count")
        ax.set_xlim([0, 1])
        ax.grid(axis="y", alpha=0.3)
        ax.axvline(ref, color="red", linestyle="--", linewidth=1.4,
                   label=f"config ref={ref:.2f}")
        ax.legend(fontsize=8)

    # overlay / density
    ax3.hist(s_scores, bins=bins, color=_C_SUCCESS, edgecolor="white",
             alpha=0.55, density=True, label=f"Success n={n_succ}")
    ax3.hist(f_scores, bins=bins, color=_C_FAILURE, edgecolor="white",
             alpha=0.55, density=True, label=f"Failure n={n_fail}")
    ax3.axvline(ref, color="red", linestyle="--", linewidth=1.4,
                label=f"config ref={ref:.2f}")
    ax3.set_title("Overlay (density)", fontweight="bold", fontsize=11)
    ax3.set_xlabel("Compression Score")
    ax3.set_ylabel("Density")
    ax3.set_xlim([0, 1])
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    req_pts  = k * tick_size
    base_str = f"{base:.3f}" if np.isfinite(base) else "n/a"
    fig.suptitle(
        f"{instrument}  --  Compression Score Distribution by Outcome\n"
        f"Valid={n_valid}  Success={n_succ}  Failure={n_fail}  "
        f"Base rate={base_str}  "
        f"Required move = {k}x tick ({req_pts:.3f} pts)  "
        f"Forward window = {forward_days}d",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    return fig


# =============================================================================
# INSTRUMENT HEADER
# =============================================================================

def _print_instrument_header(
    instrument   : str,
    structure    : str,
    product_code : str,
    start_date   : str,
    end_date     : str,
    cfg          : dict,
    tick_size    : float,
    score_grid   : np.ndarray,
    n_rows       : int,
) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  INSTRUMENT : {instrument}  ({structure.upper()})")
    print(f"  PRODUCT    : {product_code}")
    print(f"  DATA RANGE : {start_date}  -->  {end_date}   ({n_rows} bars)")
    print(f"  -- Compression signal params ----------------------------------------")
    print(f"  range_baseline     : {cfg.get('range_baseline', 'n/a')}")
    print(f"  atr_period         : {cfg.get('atr_period', 'n/a')}")
    print(f"  rv_period          : {cfg.get('rv_period', 'n/a')}")
    print(f"  persistence_window : {cfg.get('persistence_window', 'n/a')}")
    print(f"  compression_thresh : {cfg.get('compression_threshold', 'n/a')}  (score gate)")
    print(f"  -- Labeling params --------------------------------------------------")
    k = cfg.get('k', 4)
    print(f"  k (move required)  : {k}  -->  {k} x {tick_size} = {k * tick_size:.3f} pts")
    print(f"  forward_days       : {cfg.get('forward_days', 'n/a')}")
    print(f"  -- Score threshold sweep --------------------------------------------")
    print(f"  score_grid         : {score_grid[0]:.2f} --> {score_grid[-1]:.2f}  "
          f"({len(score_grid)} steps)")
    print(f"  config ref line    : {cfg.get('threshold_line', 'n/a')}")
    print(sep)


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

def run_compression_threshold_analysis(
    product_code          : str,
    months_back           : int               = 9,
    score_grid            : "np.ndarray|None" = None,
    front_n               : int               = 5,    # Pipeline tracking universe
    analytical_front_n    : int               = 2,    # Analytical universe (Front-2 for softs)
    require_eligible      : bool              = True, # Enforce liquidity qualification
    include_monitor       : bool              = False,# Include MONITOR contracts in sweep
    episode_mode          : str               = "first_bar", # Priority 3: Resolve forward window overlap
    min_signal_count      : int               = 8,
    n_bootstrap           : int               = 500,
    random_seed           : int               = 42,
    show_plots            : bool              = True,
) -> dict[str, dict]:
    """
    Run compression score threshold analysis for front spreads and jumbos.

    Phase 1 & Phase 2 Enhancements:
      - Sets Front-2 as the default analytical universe for soft commodities.
      - Retains Front-3 to Front-5 in the data pipeline for monitoring.
      - Validates actual liquidity of each spread via the Liquidity Qualification Layer
        (20d median volume, relative volume, zero-volume days).
      - Classifies into ELIGIBLE, MONITOR, ILLIQUID.
      - Resolves overlapping forward windows via `episode_mode='first_bar'`.
      - Includes zero-score days for robust true base rate and lift calculation.
      - Evaluates both Bootstrap and Wilson 95% Confidence Intervals.

    Parameters
    ----------
    product_code          : e.g. "SB", "KC", "CC", "CT", "RC", "SM"
    months_back           : Calendar months of history to pull.
    score_grid            : Array of compression_score thresholds to sweep.
    front_n               : Total pipeline universe size (default 5).
    analytical_front_n    : Analytical universe depth (default 2 for softs).
    require_eligible      : If True, only sweeps ELIGIBLE (and opt-in MONITOR) contracts.
    include_monitor       : If True, sweeps MONITOR contracts in addition to ELIGIBLE.
    episode_mode          : 'first_bar' (default), 'non_overlapping', or 'all_bars'.
    min_signal_count      : N below this is flagged "Low N" (yellow) in table.
    n_bootstrap           : Bootstrap replications for Precision CI.
    random_seed           : RNG seed.
    show_plots            : Display distribution plots inline.

    Returns
    -------
    dict keyed by instrument string (plus "_liquidity_matrix" summary DataFrame).
    """
    if score_grid is None:
        score_grid = np.round(
            np.arange(
                COMP_CFG["threshold_grid_start"],
                COMP_CFG["threshold_grid_stop"],
                COMP_CFG["threshold_grid_step"],
            ),
            3,
        )

    start_date = _months_back_start(months_back)
    end_date   = datetime.date.today().strftime("%Y-%m-%d")

    tick_size = PRODUCTS[product_code]["tick_size"]
    k          = COMP_CFG["k"]
    fwd        = COMP_CFG["forward_days"]

    all_instruments = get_all_front_instruments(
        product_code = product_code,
        front_n      = front_n,
    )

    results: dict[str, dict] = {}
    liquidity_rows: list[dict] = []

    for structure, instruments in all_instruments.items():
        print(f"\n{'#' * 72}")
        print(f"#  STRUCTURE : {structure.upper()}  --  {product_code}")
        print(f"#  Analytical Universe: Front-{analytical_front_n}  |  Pipeline: Front-{front_n}")
        print(f"{'#' * 72}")

        # ── Pre-fetch & Liquidity Qualification for all front_n instruments ──
        raw_cache: dict[str, pd.DataFrame | None] = {}
        qual_cache: dict[str, dict] = {}
        for rank, instrument in enumerate(instruments, 1):
            df_raw = fetch_ohlc_1d(
                instrument      = instrument,
                product_code    = product_code,
                getData_OHLC_v2 = getData_OHLC_v2,
                start_date      = start_date,
                end_date        = end_date,
            )
            raw_cache[instrument] = df_raw
            qual = qualify_instrument_liquidity(
                df=df_raw,
                product_code=product_code,
                instrument=instrument,
                rank=rank,
                analytical_max_rank=analytical_front_n,
            )
            qual_cache[instrument] = qual
            liquidity_rows.append(qual)

        # Display structure liquidity qualification table
        struct_liq_df = pd.DataFrame([qual_cache[ins] for ins in instruments])
        display(style_liquidity_table(
            struct_liq_df,
            caption=f"Liquidity Qualification Matrix — {product_code} {structure.upper()} (Front-{analytical_front_n} Analytical vs Pipeline)"
        ))

        for rank, instrument in enumerate(instruments, 1):
            qual = qual_cache[instrument]
            df_raw = raw_cache[instrument]
            status = qual["status"]

            # Phase 1: Analytical Universe Check
            if rank > analytical_front_n:
                print(f"\n  [PIPELINE MONITORING] {instrument} (Rank Front-{rank} > Front-{analytical_front_n}) -- "
                      f"Retained in pipeline. Status={status} ({qual['reason']})")
                continue

            # Phase 2: Liquidity Qualification Gate
            if require_eligible:
                allow_sweep = (status == "ELIGIBLE") or (include_monitor and status == "MONITOR")
                if not allow_sweep:
                    print(f"\n  [LIQUIDITY GATE] {instrument} (Rank Front-{rank}, Status={status}) -- "
                          f"Skipped from sweep: {qual['reason']}")
                    continue

            if df_raw is None or len(df_raw) < 20:
                nb = 0 if df_raw is None else len(df_raw)
                print(f"\n  [SKIP] {instrument} -- insufficient data ({nb} bars)")
                continue

            # 2. Calculate compression scores
            df_comp, _ = calculate_compression(
                df                    = df_raw,
                range_baseline        = COMP_CFG["range_baseline"],
                atr_period            = COMP_CFG["atr_period"],
                rv_period             = COMP_CFG["rv_period"],
                persistence_window    = COMP_CFG["persistence_window"],
                compression_threshold = COMP_CFG["compression_threshold"],
            )

            # 3. Label outcomes (drops zero-score rows internally)
            df_labeled = label_compression(
                df           = df_comp,
                tick         = tick_size,
                k            = k,
                forward_days = fwd,
            )
            success_col = "success"

            # 4. Print header
            _print_instrument_header(
                instrument   = instrument,
                structure    = structure,
                product_code = product_code,
                start_date   = start_date,
                end_date     = end_date,
                cfg          = COMP_CFG,
                tick_size    = tick_size,
                score_grid   = score_grid,
                n_rows       = len(df_raw),
            )

            # 5. Outcome summary
            n_total    = len(df_labeled)
            n_nonzero  = int((pd.to_numeric(df_labeled["compression_score"], errors="coerce") > 0).sum())
            n_censored = int(df_labeled[success_col].isna().sum())
            valid_rows = df_labeled.dropna(subset=[success_col])
            n_valid    = len(valid_rows)
            n_succ     = int((valid_rows[success_col] == 1.0).sum())
            n_fail     = int((valid_rows[success_col] == 0.0).sum())

            print(f"\n  Total trading bars in sample          : {n_total}")
            print(f"  Bars with non-zero compression score  : {n_nonzero} ({100*n_nonzero/max(1,n_total):.1f}%)")
            print(f"  Censored (forward window not complete): {n_censored}")
            print(f"  Valid trading bars evaluated          : {n_valid}")
            print(f"  Episode mode (forward window overlap) : {episode_mode}")
            if n_valid:
                print(f"  |-- Breakout Success (1) : {n_succ}   "
                      f"({100 * n_succ / n_valid:.1f}%)")
                print(f"  |-- No Breakout      (0) : {n_fail}   "
                      f"({100 * n_fail / n_valid:.1f}%)")
            else:
                print("  |-- Breakout Success (1) : 0")
                print("  |-- No Breakout      (0) : 0")

            # 6. Threshold sweep table
            sweep, base_rate = _make_threshold_table(
                labeled_df   = df_labeled,
                score_col    = "compression_score",
                success_col  = success_col,
                score_grid   = score_grid,
                min_n        = min_signal_count,
                n_bootstrap  = n_bootstrap,
                random_seed  = random_seed,
                forward_days = fwd,
                episode_mode = episode_mode,
            )

            br_str = f"{base_rate:.3f}" if np.isfinite(base_rate) else "n/a"
            print(f"\n  -- Combined threshold sweep  (base rate = {br_str})")

            if sweep.empty:
                print("     No valid labelled rows for sweep.")
            else:
                req_pts = k * tick_size
                caption = (
                    f"{instrument}  |  "
                    f"required_move = {k}x tick ({req_pts:.3f} pts)  |  "
                    f"forward_days = {fwd}  |  "
                    f"months_back = {months_back}  |  "
                    f"base rate = {br_str}  |  "
                    f"Yellow = N < {min_signal_count}"
                )
                display(_styled_table(sweep, caption))

            # 7. Distribution plot
            if show_plots and not sweep.empty:
                fig = _distribution_plot(
                    labeled_df   = df_labeled,
                    score_col    = "compression_score",
                    success_col  = success_col,
                    instrument   = instrument,
                    tick_size    = tick_size,
                    k            = k,
                    forward_days = fwd,
                )
                plt.show()
                plt.close(fig)

            # store
            results[instrument] = {
                "labeled_df"  : df_labeled,
                "sweep_table" : sweep,
                "base_rate"   : base_rate,
            }

    results["_liquidity_matrix"] = pd.DataFrame(liquidity_rows)
    return results
