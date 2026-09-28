# market_analysis/finding_threshold/absorption_v2_threshold.py
"""
Absorption Method-2 — threshold / parameter search.

THREE analysis modes, all callable from the notebook:

    run_heatmap_analysis()         — per-instrument z-score heatmap
                                     (delta_z_threshold × price_z_threshold grid)
                                     + score distribution plots.

    run_threshold_sweep()          — per-instrument 1D sweep over absorption_score,
                                     producing Precision / Recall / F1 / MCC / Lift
                                     table (same as absorption_v2_threshold.py but
                                     consistent with v2 internals).

    run_heatmap_by_structure()     — same as run_heatmap_analysis() but pools all
                                     instruments per structure type (spread / jumbo)
                                     into one heatmap instead of one per instrument.

All three share the same fetch-once / label-twice trick:
    1. score every bar with calculate_absorption_v2()  (dummy thresholds so
       every bar gets z_delta / z_price computed)
    2. run label_absorption_atr() twice — once forcing BUY, once forcing SELL
       — so every bar has a valid BUY outcome AND a valid SELL outcome
    3. apply the actual threshold masks at query time (no re-labeling needed)

This means the expensive labeling loop runs exactly twice per instrument,
not once per threshold combination.

Usage
-----
    from market_analysis.finding_threshold.absorption_v2_threshold import (
        run_heatmap_analysis,
        run_threshold_sweep,
        run_heatmap_by_structure,
    )

    # Heatmap — one plot per instrument
    run_heatmap_analysis(product_code="SM", months_back=8)

    # Threshold sweep table
    run_threshold_sweep(product_code="SM", months_back=8)

    # Pooled heatmap per structure type
    run_heatmap_by_structure(product_code="SM", months_back=8)
"""

from __future__ import annotations

import datetime
import calendar
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from IPython.display import display

# ── project root resolution ───────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent   # .../finding_threshold/
_ROOT = _HERE.parent                       # .../market_analysis/
_DOCS = _ROOT.parent                       # .../Documents/
for _p in [str(_DOCS), str(_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from market_analysis.config import PRODUCTS, FRONT_N
from market_analysis.utils.contracts import get_all_front_instruments
from market_analysis.data.ohlc import fetch_ohlc_1d, getData_OHLC_v2
from market_analysis.signals.absorption.calculate_v2 import calculate_absorption_v2
from market_analysis.signals.absorption.calculate_rolling import calculate_absorption_rolling
from market_analysis.signals.absorption.label import label_absorption_atr
from market_analysis.utils.metrics import _block_bootstrap_ci

try:
    from sklearn.metrics import matthews_corrcoef, precision_recall_curve, auc as sk_auc
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── palette ───────────────────────────────────────────────────────────────────
_C_SUCCESS = "seagreen"
_C_FAILURE  = "tomato"

# ── default scorer + labeler params ───────────────────────────────────────────
_DEFAULT_PARAMS: dict = {
    # z-score scorer windows
    "volume_window"    : 5,
    "atr_window"       : 5,
    "baseline_window"  : 15,
    # dummy thresholds used only to score every bar (real thresholds swept)
    "delta_z_threshold": 0,
    "price_z_threshold": 9999,
    # ATR labeler
    "atr_period"       : 14,
    "target_atr_mult"  : 1.0,
    "stop_atr_mult"    : 1.0,
    "forward_days"     : 10,
}


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _start_date(months_back: int) -> str:
    today = datetime.date.today()
    y, m  = today.year, today.month - months_back
    while m <= 0:
        m += 12; y -= 1
    d = min(today.day, calendar.monthrange(y, m)[1])
    return datetime.date(y, m, d).strftime("%Y-%m-%d")


def _fetch_score_label(
    instrument: str,
    product_code: str,
    start_date: str,
    end_date: str,
    params: dict,
    combined_candle: int = 1,          # ← new param
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:

    df = fetch_ohlc_1d(instrument, product_code, getData_OHLC_v2,
                        start_date, end_date)
    if df is None or len(df) < 30:
        return None, None, None

    # ── score (with rolling candle support) ───────────────────────────────────
    try:
        if combined_candle == 1:
            # standard path — identical to before
            df_scored = calculate_absorption_v2(
                df,
                volume_window      = params["volume_window"],
                atr_window         = params["atr_window"],
                baseline_window    = params["baseline_window"],
                delta_z_threshold  = params["delta_z_threshold"],
                price_z_threshold  = params["price_z_threshold"],
            )
        else:
            # rolling N-day candle path — uses calculate_rolling.py
            df_scored = calculate_absorption_rolling(
                df,
                combined_candle    = combined_candle,
                method             = "v2",
                volume_window      = params["volume_window"],
                atr_window         = params["atr_window"],
                baseline_window    = params["baseline_window"],
                delta_z_threshold  = params["delta_z_threshold"],
                price_z_threshold  = params["price_z_threshold"],
            )
    except Exception as e:
        print(f"  [WARN] Scoring failed for {instrument}: {e}")
        return None, None, None

    real_direction = df_scored["absorption_direction"].copy()

    df_scored["absorption_direction"] = "BUY"
    try:
        df_buy = label_absorption_atr(
            df_scored,
            atr_period      = params["atr_period"],
            target_atr_mult = params["target_atr_mult"],
            stop_atr_mult   = params["stop_atr_mult"],
            forward_days    = params["forward_days"],
        )
    except Exception as e:
        print(f"  [WARN] BUY labeling failed for {instrument}: {e}")
        return None, None, None

    df_scored["absorption_direction"] = "SELL"
    try:
        df_sell = label_absorption_atr(
            df_scored,
            atr_period      = params["atr_period"],
            target_atr_mult = params["target_atr_mult"],
            stop_atr_mult   = params["stop_atr_mult"],
            forward_days    = params["forward_days"],
        )
    except Exception as e:
        print(f"  [WARN] SELL labeling failed for {instrument}: {e}")
        return None, None, None

    df_scored["absorption_direction"] = real_direction
    return df_scored, df_buy, df_sell




def _heatmap_matrices(
    df_scored    : pd.DataFrame,
    df_buy       : pd.DataFrame,
    df_sell      : pd.DataFrame,
    delta_z_grid : np.ndarray,
    price_z_grid : np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (heat_rate, annot_labels) matrices for the given threshold grids.

    heat_rate[i, j] = success rate when  |z_delta| >= delta_z_grid[j]
                                    AND   z_price   <= price_z_grid[i]

    Annotation format per cell: "S: N\\nF: N\\nN: N\\nAvg: 0.XX"
        S = successes (BUY + SELL)
        F = failures
        N = total valid
        Avg = mean absorption_score of qualifying bars
    """
    n_price = len(price_z_grid)
    n_delta = len(delta_z_grid)

    heat_rate    = np.full((n_price, n_delta), np.nan)
    annot_labels = np.empty((n_price, n_delta), dtype=object)

    z_delta_col = df_scored["z_delta"].values
    z_price_col = df_scored["z_price"].values
    score_col   = df_scored["absorption_score"].values
    buy_success = df_buy["sl_success"].values
    sell_success= df_sell["sl_success"].values

    for i, pz in enumerate(price_z_grid):
        for j, dz in enumerate(delta_z_grid):

            buy_mask  = (z_delta_col >= dz)  & (z_price_col <= pz)
            sell_mask = (z_delta_col <= -dz) & (z_price_col <= pz)

            # outcomes from the pre-labeled frames
            buy_succ  = int(np.nansum((buy_success[buy_mask]   == 1.0)))
            buy_fail  = int(np.nansum((buy_success[buy_mask]   == 0.0)))
            sell_succ = int(np.nansum((sell_success[sell_mask] == 1.0)))
            sell_fail = int(np.nansum((sell_success[sell_mask] == 0.0)))

            tot_succ  = buy_succ  + sell_succ
            tot_fail  = buy_fail  + sell_fail
            tot_valid = tot_succ  + tot_fail

            combined_mask = buy_mask | sell_mask
            avg_score = (
                float(np.nanmean(score_col[combined_mask]))
                if combined_mask.sum() > 0 else np.nan
            )

            if tot_valid > 0:
                heat_rate[i, j]    = tot_succ / tot_valid
                annot_labels[i, j] = (
                    f"S:{tot_succ} F:{tot_fail}\n"
                    f"N:{tot_valid} Avg:{avg_score:.2f}"
                )
            else:
                heat_rate[i, j]    = np.nan
                annot_labels[i, j] = "N/A"

    return heat_rate, annot_labels


def _plot_heatmap(
    heat_rate    : np.ndarray,
    annot_labels : np.ndarray,
    delta_z_grid : np.ndarray,
    price_z_grid : np.ndarray,
    title        : str,
    params       : dict,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 7))

    sns.heatmap(
        heat_rate,
        annot  = annot_labels,
        fmt    = "",
        cmap   = "RdYlGn",
        vmin   = 0.30,
        vmax   = 0.70,
        ax     = ax,
        cbar_kws = {"label": "Success Rate"},
        linewidths = 0.4,
        linecolor  = "white",
    )

    ax.set_xticklabels([f"{dz:.2f}" for dz in delta_z_grid], rotation=0)
    ax.set_yticklabels([f"{pz:.2f}" for pz in price_z_grid], rotation=0)
    ax.invert_yaxis()

    ax.set_xlabel("Min |z_delta|  (larger = rarer, stronger delta signal)", labelpad=8)
    ax.set_ylabel("Max z_price   (lower = smaller price move = stronger absorption)", labelpad=8)

    fwd = params["forward_days"]
    T   = params["target_atr_mult"]
    S   = params["stop_atr_mult"]
    ap  = params["atr_period"]
    ax.set_title(
        f"{title}\n"
        f"ATR period={ap}  target×{T}  stop×{S}  fwd={fwd}d  "
        f"vol_win={params['volume_window']}  atr_win={params['atr_window']}  "
        f"base_win={params['baseline_window']}",
        fontsize=11,
    )
    plt.tight_layout()
    return fig


def _plot_distribution(
    df_scored    : pd.DataFrame,
    df_buy       : pd.DataFrame,
    df_sell      : pd.DataFrame,
    title        : str,
) -> plt.Figure:
    """
    3-panel success/failure distribution of absorption_score
    using the real direction from df_scored.
    """
    real_dir  = df_scored["absorption_direction"]
    buy_mask  = real_dir == "BUY"
    sell_mask = real_dir == "SELL"

    # Assign outcomes using real direction
    final_success = pd.Series(np.nan, index=df_scored.index)
    final_success.loc[buy_mask]  = df_buy.loc[buy_mask,  "sl_success"].values
    final_success.loc[sell_mask] = df_sell.loc[sell_mask, "sl_success"].values

    valid = df_scored.copy()
    valid["_suc"] = final_success
    valid = valid.dropna(subset=["_suc"])

    n_valid = len(valid)
    if n_valid == 0:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, "No valid labelled rows", ha="center", va="center",
                transform=ax.transAxes, fontsize=13)
        ax.set_title(title)
        return fig

    s_scores = valid.loc[valid["_suc"] == 1.0, "absorption_score"]
    f_scores = valid.loc[valid["_suc"] == 0.0, "absorption_score"]
    n_succ   = len(s_scores)
    n_fail   = len(f_scores)
    base     = n_succ / n_valid

    bins = np.arange(0, 1.05, 0.05)

    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
    ax1, ax2, ax3 = (fig.add_subplot(gs[k]) for k in range(3))

    for ax, data, label, color in [
        (ax1, s_scores, f"Success  n={n_succ}", _C_SUCCESS),
        (ax2, f_scores, f"Failure  n={n_fail}", _C_FAILURE),
    ]:
        ax.hist(data, bins=bins, color=color, edgecolor="white", alpha=0.85)
        ax.set_title(label, fontweight="bold", fontsize=11)
        ax.set_xlabel("Absorption Score")
        ax.set_ylabel("Count")
        ax.set_xlim([0, 1])
        ax.grid(axis="y", alpha=0.3)

    ax3.hist(s_scores, bins=bins, color=_C_SUCCESS, edgecolor="white",
             alpha=0.55, density=True, label=f"Success n={n_succ}")
    ax3.hist(f_scores, bins=bins, color=_C_FAILURE, edgecolor="white",
             alpha=0.55, density=True, label=f"Fail n={n_fail}")
    ax3.set_title("Overlay (density)", fontweight="bold", fontsize=11)
    ax3.set_xlabel("Absorption Score")
    ax3.set_ylabel("Density")
    ax3.set_xlim([0, 1])
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    fig.suptitle(
        f"{title}  |  valid={n_valid}  success={n_succ}  fail={n_fail}  base={base:.3f}",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    return fig


def _make_sweep_table(
    df_scored    : pd.DataFrame,
    df_buy       : pd.DataFrame,
    df_sell      : pd.DataFrame,
    score_grid   : np.ndarray,
    min_n        : int,
    n_bootstrap  : int,
    random_seed  : int,
    forward_days : int,
) -> tuple[pd.DataFrame, float]:
    """
    Sweep absorption_score thresholds and compute Precision/Recall/F1/MCC/Lift.

    Uses the real absorption_direction to pick BUY or SELL outcomes per row.
    """
    real_dir      = df_scored["absorption_direction"]
    buy_mask      = real_dir == "BUY"
    sell_mask     = real_dir == "SELL"

    final_success = pd.Series(np.nan, index=df_scored.index)
    final_success.loc[buy_mask]  = df_buy.loc[buy_mask,  "sl_success"].values
    final_success.loc[sell_mask] = df_sell.loc[sell_mask, "sl_success"].values

    valid = df_scored.copy()
    valid["_suc"]   = final_success
    valid["_score"] = valid["absorption_score"]
    valid = valid.dropna(subset=["_suc"]).reset_index(drop=True)

    if len(valid) == 0:
        return pd.DataFrame(), np.nan

    base_rate  = float(valid["_suc"].mean())
    block_len  = forward_days + 1
    scores_arr = valid["_score"].values
    labels_arr = valid["_suc"].values

    rows = []
    for T in score_grid:
        sig = scores_arr >= T
        lab = labels_arr.astype(int)

        tp  = int(( sig & (lab == 1)).sum())
        fp  = int(( sig & (lab == 0)).sum())
        fn  = int((~sig & (lab == 1)).sum())
        tn  = int((~sig & (lab == 0)).sum())
        ns  = tp + fp

        prec = tp / ns        if ns > 0              else np.nan
        rec  = tp / (tp + fn) if (tp + fn) > 0       else np.nan
        f1   = (2 * prec * rec / (prec + rec)
                if np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0
                else np.nan)
        lift = prec / base_rate if (np.isfinite(prec) and base_rate > 0) else np.nan

        mcc = np.nan
        if _SKLEARN and sig.sum() > 0 and (~sig).sum() > 0:
            try:
                mcc = float(matthews_corrcoef(lab, sig.astype(int)))
            except Exception:
                pass

        ci     = _block_bootstrap_ci(sig, labels_arr, n_bootstrap, block_len, random_seed)
        ci_str = f"[{ci[0]:.3f},{ci[1]:.3f}]" if np.isfinite(ci[0]) else "n/a"

        rows.append({
            "Score Thresh" : round(float(T), 3),
            "N"            : ns,
            "Precision"    : prec,
            "Prec CI 95%"  : ci_str,
            "Recall"       : rec,
            "F1"           : f1,
            "MCC"          : mcc,
            "Lift"         : lift,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Low N"        : ns < min_n,
        })

    return pd.DataFrame(rows), base_rate


def _styled_sweep_table(table: pd.DataFrame, caption: str = "") -> object:
    float_cols = {
        c: "{:.3f}" for c in table.columns
        if pd.api.types.is_float_dtype(table[c])
    }
    float_cols["Score Thresh"] = "{:.3f}"
    styler = (
        table.style
        .format(float_cols, na_rep="—")
        .apply(
            lambda row: ["background-color: #fff3cd" if row["Low N"] else ""] * len(row),
            axis=1,
        )
        .set_caption(caption)
    )
    if _SKLEARN:
        styler = (
            styler
            .background_gradient(subset=["F1"],        cmap="RdYlGn", vmin=0.3, vmax=0.7)
            .background_gradient(subset=["Precision"], cmap="Blues",  vmin=0.3, vmax=0.8)
        )
    return styler


def _pr_auc(df_scored, df_buy, df_sell) -> float:
    if not _SKLEARN:
        return np.nan
    real_dir      = df_scored["absorption_direction"]
    buy_mask      = real_dir == "BUY"
    sell_mask     = real_dir == "SELL"
    final_success = pd.Series(np.nan, index=df_scored.index)
    final_success.loc[buy_mask]  = df_buy.loc[buy_mask,  "sl_success"].values
    final_success.loc[sell_mask] = df_sell.loc[sell_mask, "sl_success"].values
    valid = df_scored.copy()
    valid["_suc"] = final_success
    valid = valid.dropna(subset=["_suc", "absorption_score"])
    if valid["_suc"].nunique() < 2:
        return np.nan
    try:
        p, r, _ = precision_recall_curve(valid["_suc"].astype(int),
                                          valid["absorption_score"])
        return float(sk_auc(r, p))
    except Exception:
        return np.nan


def _header(instrument, structure, product_code, timeframe, start_date, end_date,
             params, tick_size, n_rows, combined_candle=1):
    candle_label = f"{combined_candle}D rolling" if combined_candle > 1 else "1D"
    print(f"\n{'─'*70}")
    print(f"  {structure.upper()}  ▶  {instrument}  [{timeframe}  {candle_label}]  ({product_code})")
    print(f"{'─'*70}")
    print(f"  Data            : {start_date} → {end_date}  ({n_rows} bars)")
    print(f"  Candle          : {candle_label}")
    print(f"  Tick size       : {tick_size}")
    print(f"  volume_window   : {params['volume_window']}")
    print(f"  atr_window      : {params['atr_window']}")
    print(f"  baseline_window : {params['baseline_window']}")
    print(f"  atr_period      : {params['atr_period']}")
    print(f"  target×stop     : {params['target_atr_mult']} / {params['stop_atr_mult']}")
    print(f"  forward_days    : {params['forward_days']}")
    print(f"{'─'*70}")


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINTS
# ─────────────────────────────────────────────────────────────────────────────

def run_heatmap_analysis(
    product_code     : str,
    months_back      : int               = 8,
    front_n          : int               = FRONT_N,
    combined_candle  : int               = 1,
    delta_z_grid     : np.ndarray | None = None,
    price_z_grid     : np.ndarray | None = None,
    params           : dict | None       = None,
    show_dist_plot   : bool              = True,
    show_heatmap     : bool              = True,
) -> dict:
    """
    Per-instrument z-score threshold heatmap + absorption score distribution.

    One heatmap + one distribution plot per instrument.

    Parameters
    ----------
    product_code    : e.g. "SM" or "BO"
    months_back     : Calendar months of history to fetch.
    front_n         : Front-N instruments per structure (spread + jumbo).
    combined_candle : 1 = standard 1D bars (default).
                      2 / 3 / 5 = rolling N-day candles with causal baseline
                      anchored before the candle's first constituent bar.
    delta_z_grid    : 1-D array of |z_delta| thresholds to test on the X-axis.
                      Default: np.arange(1.0, 3.25, 0.25)
    price_z_grid    : 1-D array of z_price thresholds to test on the Y-axis.
                      Default: np.arange(0.5, -1.75, -0.25)
    params          : Override any scorer / labeler param. Keys:
                      volume_window, atr_window, baseline_window,
                      atr_period, target_atr_mult, stop_atr_mult, forward_days.
    show_dist_plot  : Display score distribution plots.
    show_heatmap    : Display heatmap plots.

    Returns
    -------
    dict keyed by instrument:
        "df_scored"  : scored DataFrame
        "df_buy"     : BUY-labeled DataFrame
        "df_sell"    : SELL-labeled DataFrame
        "heat_rate"  : numpy matrix of success rates
        "annot"      : numpy matrix of annotations
    """
    p = dict(_DEFAULT_PARAMS)
    if params:
        p.update(params)

    if delta_z_grid is None:
        delta_z_grid = np.arange(1.0, 3.25, 0.25)
    if price_z_grid is None:
        price_z_grid = np.arange(0.5, -1.75, -0.25)

    start        = _start_date(months_back)
    end          = datetime.date.today().strftime("%Y-%m-%d")
    tick         = PRODUCTS[product_code]["tick_size"]
    candle_label = f"{combined_candle}D rolling" if combined_candle > 1 else "1D"
    all_instruments = get_all_front_instruments(product_code, front_n=front_n)

    results = {}

    for structure, instruments in all_instruments.items():
        print(f"\n{'#'*72}")
        print(f"#  STRUCTURE : {structure.upper()}  —  {product_code}  [{candle_label}]")
        print(f"{'#'*72}")

        for inst in instruments:
            _header(inst, structure, product_code, "1d", start, end, p, tick,
                    n_rows=0, combined_candle=combined_candle)

            df_scored, df_buy, df_sell = _fetch_score_label(
                inst, product_code, start, end, p,
                combined_candle=combined_candle)
            if df_scored is None:
                print(f"  [SKIP] {inst} — no data / scoring failed")
                continue

            print(f"  Bars fetched    : {len(df_scored)}")

            # ── heatmap ───────────────────────────────────────────────────────
            heat_rate, annot = _heatmap_matrices(
                df_scored, df_buy, df_sell, delta_z_grid, price_z_grid)

            if show_heatmap:
                title = (f"{structure.upper()} ▶ {inst}  [{candle_label}]  |  "
                         f"Absorption v2 Heatmap  (BUY+SELL pooled)")
                fig_h = _plot_heatmap(heat_rate, annot, delta_z_grid,
                                      price_z_grid, title, p)
                plt.show(); plt.close(fig_h)

            # ── distribution ──────────────────────────────────────────────────
            if show_dist_plot:
                title_d = (f"{structure.upper()} ▶ {inst}  [{candle_label}]  |  "
                           f"Absorption Score Distribution")
                fig_d   = _plot_distribution(df_scored, df_buy, df_sell, title_d)
                plt.show(); plt.close(fig_d)

            results[inst] = {
                "df_scored" : df_scored,
                "df_buy"    : df_buy,
                "df_sell"   : df_sell,
                "heat_rate" : heat_rate,
                "annot"     : annot,
            }

    return results


def run_heatmap_by_structure(
    product_code     : str,
    months_back      : int               = 8,
    front_n          : int               = FRONT_N,
    delta_z_grid     : np.ndarray | None = None,
    price_z_grid     : np.ndarray | None = None,
    params           : dict | None       = None,
    show_dist_plot   : bool              = True,
    show_heatmap     : bool              = True,
) -> dict:
    """
    Same as run_heatmap_analysis() but pools all instruments within each
    structure type (spread / jumbo) into a single heatmap.

    Useful for seeing structure-level signal quality without per-instrument noise.

    Returns
    -------
    dict keyed by structure name ("spread", "jumbo"):
        "heat_rate"   : pooled success-rate matrix
        "annot"       : pooled annotation matrix
        "instruments" : list of instruments included
    """
    p = dict(_DEFAULT_PARAMS)
    if params:
        p.update(params)

    if delta_z_grid is None:
        delta_z_grid = np.arange(1.0, 3.25, 0.25)
    if price_z_grid is None:
        price_z_grid = np.arange(0.5, -1.75, -0.25)

    start = _start_date(months_back)
    end   = datetime.date.today().strftime("%Y-%m-%d")
    all_instruments = get_all_front_instruments(product_code, front_n=front_n)

    results = {}

    for structure, instruments in all_instruments.items():
        print(f"\n{'#'*72}")
        print(f"#  POOLED HEATMAP — {structure.upper()}  ({product_code})")
        print(f"{'#'*72}")

        n_price = len(price_z_grid)
        n_delta = len(delta_z_grid)

        # accumulate successes/failures across instruments
        pool_succ = np.zeros((n_price, n_delta))
        pool_fail = np.zeros((n_price, n_delta))
        pool_score_sum = np.zeros((n_price, n_delta))
        pool_score_cnt = np.zeros((n_price, n_delta))
        valid_instruments = []

        for inst in instruments:
            print(f"  Loading {inst}...")
            df_scored, df_buy, df_sell = _fetch_score_label(
                inst, product_code, start, end, p)
            if df_scored is None:
                print(f"    [SKIP]")
                continue

            valid_instruments.append(inst)
            z_delta_col  = df_scored["z_delta"].values
            z_price_col  = df_scored["z_price"].values
            score_col    = df_scored["absorption_score"].values
            buy_success  = df_buy["sl_success"].values
            sell_success = df_sell["sl_success"].values

            for i, pz in enumerate(price_z_grid):
                for j, dz in enumerate(delta_z_grid):
                    buy_mask  = (z_delta_col >= dz)  & (z_price_col <= pz)
                    sell_mask = (z_delta_col <= -dz) & (z_price_col <= pz)

                    pool_succ[i,j] += (
                        np.nansum(buy_success[buy_mask]   == 1.0) +
                        np.nansum(sell_success[sell_mask] == 1.0)
                    )
                    pool_fail[i,j] += (
                        np.nansum(buy_success[buy_mask]   == 0.0) +
                        np.nansum(sell_success[sell_mask] == 0.0)
                    )
                    comb = score_col[buy_mask | sell_mask]
                    pool_score_sum[i,j] += np.nansum(comb)
                    pool_score_cnt[i,j] += np.isfinite(comb).sum()

        # build matrices
        heat_rate    = np.full((n_price, n_delta), np.nan)
        annot_labels = np.empty((n_price, n_delta), dtype=object)

        for i in range(n_price):
            for j in range(n_delta):
                tot_valid = pool_succ[i,j] + pool_fail[i,j]
                avg_score = (pool_score_sum[i,j] / pool_score_cnt[i,j]
                             if pool_score_cnt[i,j] > 0 else np.nan)
                if tot_valid > 0:
                    heat_rate[i,j]    = pool_succ[i,j] / tot_valid
                    annot_labels[i,j] = (
                        f"S:{int(pool_succ[i,j])} F:{int(pool_fail[i,j])}\n"
                        f"N:{int(tot_valid)} Avg:{avg_score:.2f}"
                    )
                else:
                    heat_rate[i,j]    = np.nan
                    annot_labels[i,j] = "N/A"

        if show_heatmap:
            title = (f"POOLED {structure.upper()}  |  {product_code}  "
                     f"|  {len(valid_instruments)} instruments")
            fig_h = _plot_heatmap(heat_rate, annot_labels, delta_z_grid,
                                   price_z_grid, title, p)
            plt.show(); plt.close(fig_h)

        results[structure] = {
            "heat_rate"   : heat_rate,
            "annot"       : annot_labels,
            "instruments" : valid_instruments,
        }

    return results


def run_threshold_sweep(
    product_code     : str,
    months_back      : int               = 8,
    front_n          : int               = FRONT_N,
    score_grid       : np.ndarray | None = None,
    params           : dict | None       = None,
    min_signal_count : int               = 8,
    n_bootstrap      : int               = 500,
    random_seed      : int               = 42,
    show_plots       : bool              = True,
) -> dict:
    """
    Per-instrument absorption_score threshold sweep.

    Sweeps the continuous absorption_score (not the z-thresholds) and
    produces a Precision / Recall / F1 / MCC / Lift table for each instrument.

    Parameters
    ----------
    score_grid       : Thresholds to sweep. Default np.arange(0.05, 1.0, 0.05).
    params           : Override scorer / labeler params.
    min_signal_count : Rows with N < this are highlighted yellow.
    n_bootstrap      : Bootstrap iterations for Precision CI.
    random_seed      : RNG seed.
    show_plots       : Display distribution plots.

    Returns
    -------
    dict keyed by instrument:
        "sweep_table" : threshold DataFrame
        "base_rate"   : float
        "pr_auc"      : float
    """
    p = dict(_DEFAULT_PARAMS)
    if params:
        p.update(params)

    if score_grid is None:
        score_grid = np.round(np.arange(0.05, 1.0, 0.05), 3)

    start = _start_date(months_back)
    end   = datetime.date.today().strftime("%Y-%m-%d")
    tick  = PRODUCTS[product_code]["tick_size"]
    all_instruments = get_all_front_instruments(product_code, front_n=front_n)

    results = {}

    for structure, instruments in all_instruments.items():
        print(f"\n{'#'*72}")
        print(f"#  STRUCTURE : {structure.upper()}  —  {product_code}  (threshold sweep)")
        print(f"{'#'*72}")

        for inst in instruments:
            _header(inst, structure, product_code, "1d", start, end, p, tick, 0)

            df_scored, df_buy, df_sell = _fetch_score_label(
                inst, product_code, start, end, p)
            if df_scored is None:
                print(f"  [SKIP] {inst}")
                continue

            # direction summary
            real_dir = df_scored["absorption_direction"]
            for d in ("BUY", "SELL"):
                sub = df_scored[real_dir == d]
                if len(sub):
                    out = (df_buy if d == "BUY" else df_sell)
                    suc = (out.loc[real_dir == d, "sl_success"] == 1.0).sum()
                    tot = out.loc[real_dir == d, "sl_success"].notna().sum()
                    print(f"  {d:<4}  directional bars={len(sub)}  "
                          f"valid outcomes={tot}  "
                          f"success_rate={suc/tot:.3f}" if tot else
                          f"  {d:<4}  directional bars={len(sub)}  valid=0")

            # sweep
            sweep, base_rate = _make_sweep_table(
                df_scored, df_buy, df_sell,
                score_grid, min_signal_count, n_bootstrap, random_seed,
                p["forward_days"],
            )
            auc_val = _pr_auc(df_scored, df_buy, df_sell)

            br_str  = f"{base_rate:.3f}" if np.isfinite(base_rate) else "n/a"
            auc_str = f"{auc_val:.3f}"   if np.isfinite(auc_val)   else "n/a"
            print(f"\n  Threshold sweep  (base={br_str}  PR-AUC={auc_str})")

            if not sweep.empty:
                caption = (
                    f"{inst}  |  "
                    f"Δz_thresh=dummy  Pz_thresh=dummy  |  "
                    f"ATR×{p['target_atr_mult']}/×{p['stop_atr_mult']}  "
                    f"fwd={p['forward_days']}d  months={months_back}  "
                    f"base={br_str}  Yellow=N<{min_signal_count}"
                )
                display(_styled_sweep_table(sweep, caption))

            if show_plots:
                title_d = f"{structure.upper()} ▶ {inst}  |  Absorption Score Distribution"
                fig_d   = _plot_distribution(df_scored, df_buy, df_sell, title_d)
                plt.show(); plt.close(fig_d)

            results[inst] = {
                "sweep_table" : sweep,
                "base_rate"   : base_rate,
                "pr_auc"      : auc_val,
            }

    return results