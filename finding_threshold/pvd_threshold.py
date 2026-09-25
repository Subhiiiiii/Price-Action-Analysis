# market_analysis/finding_threshold/pvd_threshold.py
"""
PVD threshold analysis â€” exhaustion + expansion.

Entry point for the notebook:

    from finding_threshold.pvd_threshold import run_pvd_threshold_analysis

    results = run_pvd_threshold_analysis(
        product_code   = "SM",
        months_back    = 9,
        z_grid_neg     = np.arange(-3.0, -0.5, 0.25),   # exhaustion sweep
        z_grid_pos     = np.arange( 0.5,  3.5, 0.25),   # expansion sweep
        atr_multiplier = 1.5,
        horizon_days   = 10,
        front_n        = 5,
    )

EXHAUSTION section (per instrument):
    - ONE combined F1/Precision/Recall/MCC/Lift table for UP + DOWN together.
      Discriminant = |pvd_z|. Higher = stronger signal regardless of direction.
    - ONE combined distribution chart (UP and DOWN side by side per |z| bucket).
    - Vertical dashed line marks the config reference |z| threshold.

EXPANSION section (per instrument):
    - For PRICE_EXPANSION signals (positive z), visualise the real residual
      in terms of k * ATR per |z| bucket.
      real_residual_atr = |pvd_z * pvd_atr|
      (proportional to how much the actual price move exceeded model expectation,
       expressed in ATR units)
    - Bar chart: mean |real residual| per |z| bucket (UP and DOWN bars).
    - Scatter: pvd_z vs real residual coloured by direction.
"""

from __future__ import annotations

import sys
import datetime
import calendar
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

# â”€â”€ project root on sys.path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_HERE = Path(__file__).resolve().parent       # .../finding_threshold/
_ROOT = _HERE.parent                          # .../market_analysis/
_DOCS = _ROOT.parent                          # .../Documents/
for _p in [str(_DOCS), str(_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# â”€â”€ project imports â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from market_analysis.config import (
    PRODUCTS, FRONT_N, PVD as PVD_CFG,
    OHLC_START_DATE, OHLC_END_DATE,
)
from market_analysis.utils.contracts import get_all_front_instruments
from market_analysis.data.ohlc import fetch_ohlc_1d, getData_OHLC_v2
from market_analysis.signals.pvd.calculate import calculate_pvd
from market_analysis.signals.pvd.label    import label_pvd
from market_analysis.utils.metrics import _block_bootstrap_ci

# â”€â”€ colour palette â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_CUP_S   = "#2ecc71"   # UP   SUCCESS
_CUP_F   = "#e74c3c"   # UP   FAILURE
_CUP_U   = "#3498db"   # UP   UNRESOLVED
_CDN_S   = "#27ae60"   # DOWN SUCCESS
_CDN_F   = "#c0392b"   # DOWN FAILURE
_CDN_U   = "#2980b9"   # DOWN UNRESOLVED
_CREF    = "red"       # reference line

_EXHAUSTION_SIGNALS = {"UP_TREND_EXHAUSTION", "DOWN_TREND_EXHAUSTION"}
_EXPANSION_SIGNALS  = {"UP_PRICE_EXPANSION",  "DOWN_PRICE_EXPANSION"}

# â”€â”€ shared bucket edges â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_EDGES  = list(np.arange(0.5, 2.5, 0.05)) + [np.inf]
_LABELS = [
    f"{_EDGES[i]:.1f}-{_EDGES[i+1]:.1f}" if _EDGES[i+1] != np.inf
    else f">{_EDGES[i]:.1f}"
    for i in range(len(_EDGES) - 1)
]


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


def _assign_bucket(s: pd.Series) -> pd.Categorical:
    """Assign |z| values to fixed buckets; result always has all categories."""
    return pd.Categorical(
        pd.cut(s, bins=_EDGES, labels=_LABELS, right=False),
        categories=_LABELS, ordered=True,
    )


def _counts_by_bucket(
    sub             : pd.DataFrame,
    success_col     : str,
    merge_unresolved: bool = False,
) -> pd.DataFrame:
    """
    Return (n_labels x 3) count table always with all bucket rows.

    merge_unresolved=True  : UNRESOLVED rows are reclassified as SUCCESS
                             (senior's treatment: expired horizon = no reversal = win).
    merge_unresolved=False : strict 3-class counts (default).
    """
    if len(sub) == 0:
        return pd.DataFrame(0, index=_LABELS,
                            columns=["SUCCESS", "FAILURE", "UNRESOLVED"])
    s = sub.copy()
    s["z_abs"] = s["pvd_z"].abs()
    s["_cat"]  = "UNRESOLVED"
    s.loc[s[success_col] == 1.0, "_cat"] = "SUCCESS"
    s.loc[s[success_col] == 0.0, "_cat"] = "FAILURE"
    # When merging, flip UNRESOLVED → SUCCESS before counting
    if merge_unresolved:
        s.loc[s["_cat"] == "UNRESOLVED", "_cat"] = "SUCCESS"
    s["_bucket"] = _assign_bucket(s["z_abs"])
    return (
        s.groupby(["_bucket", "_cat"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=_LABELS, fill_value=0)
        .reindex(columns=["SUCCESS", "FAILURE", "UNRESOLVED"], fill_value=0)
    )


# =============================================================================
# EXHAUSTION â€” COMBINED THRESHOLD TABLE (UP + DOWN)
# =============================================================================

def _make_combined_threshold_table(
    labeled_df      : pd.DataFrame,
    z_abs_grid      : np.ndarray,
    success_col     : str  = "pvd_success",
    min_n           : int  = 8,
    n_bootstrap     : int  = 500,
    random_seed     : int  = 42,
    horizon_days    : int  = 10,
    merge_unresolved: bool = False,
) -> tuple[pd.DataFrame, float]:
    """
    Sweep |pvd_z| threshold over combined UP + DOWN exhaustion rows.
    A row is 'signalled' when |pvd_z| >= T.
    Returns (table_df, base_rate).
    """
    sub = labeled_df[labeled_df["pvd_outcome"] != ""].copy()

    if merge_unresolved:
        # Include UNRESOLVED rows and treat them as SUCCESS (label = 1)
        valid = sub.copy().reset_index(drop=True)
        # Fill NaN success_col (= UNRESOLVED) with 1.0
        valid[success_col] = valid[success_col].fillna(1.0)
    else:
        # Strict: exclude UNRESOLVED rows entirely
        valid = sub.dropna(subset=[success_col]).reset_index(drop=True)

    if len(valid) == 0:
        return pd.DataFrame(), np.nan

    base_rate  = float(valid[success_col].mean())
    block_len  = horizon_days + 1
    z_abs_arr  = valid["pvd_z"].abs().values
    labels_arr = valid[success_col].values

    rows = []
    for T in z_abs_grid:
        sig = z_abs_arr >= T
        lab = labels_arr.astype(int)

        tp  = int(( sig & (lab == 1)).sum())
        fp  = int(( sig & (lab == 0)).sum())
        fn  = int((~sig & (lab == 1)).sum())
        tn  = int((~sig & (lab == 0)).sum())
        ns  = tp + fp

        prec = tp / ns        if ns > 0               else np.nan
        rec  = tp / (tp + fn) if (tp + fn) > 0        else np.nan
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
        ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if np.isfinite(ci[0]) else "n/a"

        rows.append({
            "|z| Threshold": round(T, 3),
            "N (UP+DN)"    : ns,
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


def _styled_table(table: pd.DataFrame, caption: str = "") -> object:
    float_cols = {c: "{:.3f}" for c in table.columns if table[c].dtype == float}
    float_cols["|z| Threshold"] = "{:.3f}"
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
# EXHAUSTION â€” COMBINED DISTRIBUTION CHART (UP + DOWN)
# =============================================================================

def _exhaustion_distribution_plot(
    labeled_df      : pd.DataFrame,
    instrument      : str,
    z_ref           : float,
    success_col     : str  = "pvd_success",
    merge_unresolved: bool = False,
) -> plt.Figure:
    """
    6-bar-series chart grouped by |pvd_z| bucket:
    UP-SUCCESS / UP-FAILURE / UP-UNRESOLVED / DN-SUCCESS / DN-FAILURE / DN-UNRESOLVED

    When merge_unresolved=True the UNRESOLVED bars collapse into SUCCESS.
    """
    up  = labeled_df[labeled_df["pvd_signal"] == "UP_TREND_EXHAUSTION"]
    dn  = labeled_df[labeled_df["pvd_signal"] == "DOWN_TREND_EXHAUSTION"]
    cup = _counts_by_bucket(up, success_col, merge_unresolved=merge_unresolved)
    cdn = _counts_by_bucket(dn, success_col, merge_unresolved=merge_unresolved)

    def _tot(sub, cat):
        if len(sub) == 0: return 0
        s = sub.copy()
        s["_cat"] = "UNRESOLVED"
        s.loc[s[success_col] == 1.0, "_cat"] = "SUCCESS"
        s.loc[s[success_col] == 0.0, "_cat"] = "FAILURE"
        return int((s["_cat"] == cat).sum())

    n_up_s = _tot(up, "SUCCESS");    n_up_f = _tot(up, "FAILURE");   n_up_u = _tot(up, "UNRESOLVED")
    n_dn_s = _tot(dn, "SUCCESS");    n_dn_f = _tot(dn, "FAILURE");   n_dn_u = _tot(dn, "UNRESOLVED")
    up_base = n_up_s / (n_up_s + n_up_f) if (n_up_s + n_up_f) > 0 else np.nan
    dn_base = n_dn_s / (n_dn_s + n_dn_f) if (n_dn_s + n_dn_f) > 0 else np.nan

    x = np.arange(len(_LABELS))
    w = 0.13
    g = 0.04  # gap between UP and DOWN cluster

    series = [
        (cup, "UP-SUCCESS",    _CUP_S, -2.5*w - g),
        (cup, "UP-FAILURE",    _CUP_F, -1.5*w - g),
        (cup, "UP-UNRESOLVED", _CUP_U, -0.5*w - g),
        (cdn, "DN-SUCCESS",    _CDN_S, +0.5*w + g),
        (cdn, "DN-FAILURE",    _CDN_F, +1.5*w + g),
        (cdn, "DN-UNRESOLVED", _CDN_U, +2.5*w + g),
    ]
    col_map = {
        "UP-SUCCESS": "SUCCESS", "UP-FAILURE": "FAILURE", "UP-UNRESOLVED": "UNRESOLVED",
        "DN-SUCCESS": "SUCCESS", "DN-FAILURE": "FAILURE", "DN-UNRESOLVED": "UNRESOLVED",
    }

    fig, ax = plt.subplots(figsize=(16, 6))
    for cnt_df, label, color, offset in series:
        col  = col_map[label]
        vals = cnt_df[col].values if col in cnt_df.columns else np.zeros(len(_LABELS))
        ax.bar(x + offset, vals, width=w, label=label, color=color,
               edgecolor="white", alpha=0.85)

    # reference line
    z_abs_ref = abs(z_ref)
    ref_x     = np.interp(z_abs_ref, _EDGES[:-1], np.arange(len(_LABELS)))
    ax.axvline(x=ref_x, color=_CREF, linestyle="--", linewidth=1.8,
               label=f"|z| = {z_abs_ref:.2f} (config ref)")

    ax.set_xticks(x)
    ax.set_xticklabels(_LABELS, rotation=30, ha="right", fontsize=9)
    ax.set_xlabel("|pvd_z| bucket", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    up_br = f"{up_base:.3f}" if np.isfinite(up_base) else "n/a"
    dn_br = f"{dn_base:.3f}" if np.isfinite(dn_base) else "n/a"
    ax.set_title(
        f"{instrument}  --  EXHAUSTION distribution  (UP + DOWN)\n"
        f"UP : S={n_up_s}  F={n_up_f}  U={n_up_u}  base={up_br}    "
        f"DOWN : S={n_dn_s}  F={n_dn_f}  U={n_dn_u}  base={dn_br}",
        fontsize=11,
    )
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


# =============================================================================
# EXPANSION â€” REAL RESIDUAL ANALYSIS
# =============================================================================

def _compute_real_residual(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add _real_res_atr = pvd_z * pvd_atr  and  _abs_res_atr = |_real_res_atr|.

    Rationale:
      pvd_z = (log_residual - mu_res) / sigma_res
      => log_residual = pvd_z * sigma_res + mu_res
      The residual is in log-normalised price space; to convert back to
      real ATR units we use pvd_atr as the price-scale reference:
        real_residual ~ pvd_z * pvd_atr
      This is directional (positive = actual move exceeded model expectation)
      and proportional to the true over/under-shoot in ATR multiples.
    """
    df = df.copy()
    df["_real_res_atr"] = df["pvd_z"] * df["pvd_atr"].fillna(0)
    df["_abs_res_atr"]  = df["_real_res_atr"].abs()
    return df


def _expansion_analysis_plots(
    df_with_res: pd.DataFrame,
    instrument : str,
    z_grid_pos : np.ndarray,
) -> list[plt.Figure]:
    """
    Two charts for PRICE_EXPANSION signals:
      1. Bar chart: mean |real residual| (ATR units) per |z| bucket, UP vs DOWN.
      2. Scatter:   pvd_z vs real residual, coloured by direction.
    """
    exp_rows = df_with_res[df_with_res["pvd_signal"].isin(_EXPANSION_SIGNALS)].copy()
    figs: list[plt.Figure] = []

    if len(exp_rows) == 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5,
                f"No PRICE_EXPANSION signals found\n{instrument}",
                ha="center", va="center", transform=ax.transAxes, fontsize=13)
        ax.set_title(f"{instrument}  --  EXPANSION analysis (no data)")
        figs.append(fig)
        return figs

    exp_rows["z_abs"]   = exp_rows["pvd_z"].abs()
    exp_rows["_bucket"] = _assign_bucket(exp_rows["z_abs"])
    exp_rows["_dir"]    = exp_rows["pvd_signal"].map({
        "UP_PRICE_EXPANSION"  : "UP",
        "DOWN_PRICE_EXPANSION": "DOWN",
    }).fillna("UP")

    # â”€â”€ Chart 1: mean |residual| per bucket â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    x = np.arange(len(_LABELS))
    w = 0.38

    for k, (direction, color) in enumerate([("UP", "#8e44ad"), ("DOWN", "#2980b9")]):
        sub = exp_rows[exp_rows["_dir"] == direction]
        grp = (
            sub.groupby("_bucket", observed=False)["_abs_res_atr"]
            .agg(["mean", "std", "count"])
            .reindex(_LABELS, fill_value=0)
        )
        means  = grp["mean"].fillna(0).values
        stds   = grp["std"].fillna(0).values
        counts = grp["count"].values
        sems   = np.where(counts > 1, stds / np.sqrt(np.maximum(counts, 1)), 0)

        ax1.bar(x + (k - 0.5) * w, means, width=w,
                label=f"{direction} EXPANSION  (n={len(sub)})",
                color=color, edgecolor="white", alpha=0.82,
                yerr=sems, capsize=4, error_kw={"linewidth": 1.2})

    ax1.set_xticks(x)
    ax1.set_xticklabels(_LABELS, rotation=30, ha="right", fontsize=9)
    ax1.set_xlabel("|pvd_z| bucket", fontsize=10)
    ax1.set_ylabel("Mean |real residual|  =  mean( |pvd_z x ATR| )", fontsize=10)
    ax1.set_title(
        f"{instrument}  --  PRICE_EXPANSION  |  mean real residual by |z| bucket\n"
        f"Residual unit: |pvd_z x pvd_atr|  (k x ATR proxy for over/undershoot)\n"
        f"Total: UP={int((exp_rows['_dir']=='UP').sum())}  "
        f"DOWN={int((exp_rows['_dir']=='DOWN').sum())}",
        fontsize=11,
    )
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    figs.append(fig1)

    # â”€â”€ Chart 2: scatter pvd_z vs real residual â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for direction, color, marker in [
        ("UP",   "#8e44ad", "^"),
        ("DOWN", "#2980b9", "v"),
    ]:
        sub = exp_rows[exp_rows["_dir"] == direction]
        if len(sub) == 0:
            continue
        ax2.scatter(sub["pvd_z"], sub["_real_res_atr"],
                    c=color, marker=marker, alpha=0.6, s=45,
                    label=f"{direction} (n={len(sub)})",
                    edgecolors="white", linewidths=0.3)

    ax2.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax2.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    ax2.axvspan(float(z_grid_pos[0]), float(z_grid_pos[-1]), alpha=0.06, color="#9b59b6")
    ax2.set_xlabel("pvd_z", fontsize=10)
    ax2.set_ylabel("Real residual  (pvd_z x pvd_atr)", fontsize=10)
    ax2.set_title(
        f"{instrument}  --  PRICE_EXPANSION scatter\n"
        f"Positive = actual price move EXCEEDED model expectation",
        fontsize=11,
    )
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25)
    plt.tight_layout()
    figs.append(fig2)

    return figs


# =============================================================================
# INSTRUMENT HEADER
# =============================================================================

def _print_instrument_header(
    instrument   : str,
    structure    : str,
    product_code : str,
    start_date   : str,
    end_date     : str,
    pvd_cfg      : dict,
    z_grid_neg   : np.ndarray,
    z_grid_pos   : np.ndarray,
    atr_mult     : float,
    horizon_days : int,
    n_rows       : int,
) -> None:
    sep = "=" * 72
    abs_neg = np.sort(np.abs(z_grid_neg))
    print(f"\n{sep}")
    print(f"  INSTRUMENT : {instrument}  ({structure.upper()})")
    print(f"  PRODUCT    : {product_code}")
    print(f"  DATA RANGE : {start_date}  -->  {end_date}   ({n_rows} bars)")
    print(f"  -- PVD signal params -----------------------------------------")
    print(f"  price_horizon_h   : {pvd_cfg['price_horizon_h']}")
    print(f"  volatility_window : {pvd_cfg['volatility_window']}")
    print(f"  pvd_model_window  : {pvd_cfg['pvd_model_window']}")
    print(f"  atr_period        : {pvd_cfg['atr_period']}")
    print(f"  -- Exhaustion outcome labelling ------------------------------")
    print(f"  atr_multiplier    : {atr_mult}")
    print(f"  horizon_days      : {horizon_days}")
    _merge_flag = pvd_cfg.get('exhaustion_merge_unresolved', False)
    print(f"  merge_unresolved  : {_merge_flag}  "
          f"{'(UNRESOLVED → SUCCESS)' if _merge_flag else '(UNRESOLVED excluded)'}")
    print(f"  -- Exhaustion |z| sweep (combined UP+DOWN) ------------------")
    print(f"  |z| range         : {abs_neg[0]:.2f} --> {abs_neg[-1]:.2f}  ({len(abs_neg)} steps)")
    print(f"  config ref |z|    : {abs(pvd_cfg['exhaustion_z_threshold']):.2f}")
    print(f"  -- Expansion z range ----------------------------------------")
    print(f"  z_grid_pos        : {float(z_grid_pos[0]):.2f} --> {float(z_grid_pos[-1]):.2f}")
    print(sep)


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

def run_pvd_threshold_analysis(
    product_code    : str,
    months_back     : int               = 9,
    z_grid_neg      : "np.ndarray|None" = None,
    z_grid_pos      : "np.ndarray|None" = None,
    atr_multiplier  : float             = 1.5,
    horizon_days    : int               = 10,
    front_n         : int               = 5,
    min_signal_count: int               = 8,
    n_bootstrap     : int               = 500,
    random_seed     : int               = 42,
    show_plots      : bool              = True,
) -> dict:
    """
    Run PVD threshold analysis for all front-N spreads and jumbos.

    Parameters
    ----------
    product_code      : e.g. "SM"
    months_back       : Calendar months of history to fetch.
    z_grid_neg        : Exhaustion z-score sweep  -- NEGATIVE values
                        e.g. np.arange(-3.0, -0.5, 0.25).
    z_grid_pos        : Expansion z-score range   -- POSITIVE values
                        e.g. np.arange(0.5, 3.5, 0.25).
    atr_multiplier    : ATR x this = target/stop for outcome labelling.
    horizon_days      : Max future daily bars to scan for outcome.
    front_n           : Front-N spreads and jumbos each.
    min_signal_count  : N below this flagged yellow in table.
    n_bootstrap       : Bootstrap replications for Precision CI.
    random_seed       : RNG seed.
    show_plots        : Display plots inline.

    Returns
    -------
    dict keyed by instrument:
        "labeled_df"   : fully labelled DataFrame
        "sweep_table"  : combined UP+DOWN exhaustion threshold table
        "base_rate"    : success base rate (exhaustion)
        "expansion_df" : PRICE_EXPANSION rows with real residual column
    """
    if z_grid_neg is None:
        z_grid_neg = np.arange(-3.0, -0.5, 0.25)
    if z_grid_pos is None:
        z_grid_pos = np.arange(0.5, 3.5, 0.25)

    z_abs_grid         = np.sort(np.abs(z_grid_neg))        # ascending positives
    z_ref              = float(PVD_CFG["exhaustion_z_threshold"])
    exh_thresh_for_calc = float(np.max(z_grid_neg))          # least extreme = most inclusive
    exp_thresh_for_calc = float(np.min(z_grid_pos))          # least extreme = most inclusive

    start_date = _months_back_start(months_back)
    end_date   = datetime.date.today().strftime("%Y-%m-%d")

    all_instruments = get_all_front_instruments(
        product_code = product_code,
        front_n      = front_n,
    )

    results: dict = {}

    for structure, instruments in all_instruments.items():
        print(f"\n{'#' * 72}")
        print(f"#  STRUCTURE : {structure.upper()}  --  {product_code}")
        print(f"{'#' * 72}")

        for instrument in instruments:

            # 1. Fetch OHLC
            df1d = fetch_ohlc_1d(
                instrument      = instrument,
                product_code    = product_code,
                getData_OHLC_v2 = getData_OHLC_v2,
                start_date      = start_date,
                end_date        = end_date,
            )
            if df1d is None or len(df1d) < 30:
                nb = 0 if df1d is None else len(df1d)
                print(f"\n  [SKIP] {instrument} -- insufficient data ({nb} bars)")
                continue

            # 2. Calculate PVD
            df_pvd = calculate_pvd(
                df                     = df1d,
                price_horizon_h        = PVD_CFG["price_horizon_h"],
                volatility_window      = PVD_CFG["volatility_window"],
                pvd_model_window       = PVD_CFG["pvd_model_window"],
                atr_period             = PVD_CFG["atr_period"],
                exhaustion_z_threshold = exh_thresh_for_calc,
                expansion_z_threshold  = exp_thresh_for_calc,
            )

            # 3. Label exhaustion outcomes
            df_labeled = label_pvd(
                df             = df_pvd,
                atr_multiplier = atr_multiplier,
                horizon_days   = horizon_days,
            )

            # 4. Add real residual (for expansion section)
            df_labeled = _compute_real_residual(df_labeled)

            _print_instrument_header(
                instrument   = instrument,
                structure    = structure,
                product_code = product_code,
                start_date   = start_date,
                end_date     = end_date,
                pvd_cfg      = PVD_CFG,
                z_grid_neg   = z_grid_neg,
                z_grid_pos   = z_grid_pos,
                atr_mult     = atr_multiplier,
                horizon_days = horizon_days,
                n_rows       = len(df1d),
            )

            # 5. Exhaustion signal counts
            exh_rows = df_labeled[df_labeled["pvd_signal"].isin(_EXHAUSTION_SIGNALS)]
            exp_rows = df_labeled[df_labeled["pvd_signal"].isin(_EXPANSION_SIGNALS)]

            n_succ  = int((exh_rows["pvd_success"] == 1.0).sum())
            n_fail  = int((exh_rows["pvd_success"] == 0.0).sum())
            n_unres = int((exh_rows["pvd_outcome"] == "UNRESOLVED").sum())

            print(f"\n  -- EXHAUSTION SIGNALS ------------------------------------------")
            print(f"  Total  : {len(exh_rows)}   SUCCESS={n_succ}  FAILURE={n_fail}  UNRESOLVED={n_unres}")
            for sig in ("UP_TREND_EXHAUSTION", "DOWN_TREND_EXHAUSTION"):
                sub = exh_rows[exh_rows["pvd_signal"] == sig]
                s   = int((sub["pvd_success"] == 1.0).sum())
                f   = int((sub["pvd_success"] == 0.0).sum())
                u   = int((sub["pvd_outcome"] == "UNRESOLVED").sum())
                print(f"    {sig:<28}: S={s}  F={f}  U={u}  total={len(sub)}")

            print(f"\n  -- EXPANSION SIGNALS -------------------------------------------")
            for sig in ("UP_PRICE_EXPANSION", "DOWN_PRICE_EXPANSION"):
                sub = exp_rows[exp_rows["pvd_signal"] == sig]
                print(f"    {sig:<28}: total={len(sub)}")

            # 6. Combined threshold table
            _merge_unresolved = bool(PVD_CFG.get("exhaustion_merge_unresolved", False))
            sweep_table, base_rate = _make_combined_threshold_table(
                labeled_df       = df_labeled,
                z_abs_grid       = z_abs_grid,
                success_col      = "pvd_success",
                min_n            = min_signal_count,
                n_bootstrap      = n_bootstrap,
                random_seed      = random_seed,
                horizon_days     = horizon_days,
                merge_unresolved = _merge_unresolved,
            )

            _br = f"{base_rate:.3f}" if np.isfinite(base_rate) else "n/a"
            _mode = "UNRESOLVED→SUCCESS" if _merge_unresolved else "strict 3-class"
            print(f"\n  -- EXHAUSTION THRESHOLD TABLE (UP+DOWN combined)  "
                  f"base rate={_br}  [{_mode}]")
            if sweep_table.empty:
                print("     No labelled rows available.")
            else:
                caption = (
                    f"{instrument}  |  EXHAUSTION UP+DOWN  "
                    f"|  ATR={atr_multiplier}  |  horizon={horizon_days}d  "
                    f"|  data={months_back}m  |  base={_br}  |  {_mode}"
                )
                display(_styled_table(sweep_table, caption))

            # 7. Distribution plot
            if show_plots:
                fig = _exhaustion_distribution_plot(
                    labeled_df       = df_labeled,
                    instrument       = instrument,
                    z_ref            = z_ref,
                    merge_unresolved = _merge_unresolved,
                )
                plt.show()
                plt.close(fig)

            # 8. Expansion analysis plots
            if show_plots:
                for f in _expansion_analysis_plots(df_labeled, instrument, z_grid_pos):
                    plt.show()
                    plt.close(f)

            results[instrument] = {
                "labeled_df"   : df_labeled,
                "sweep_table"  : sweep_table,
                "base_rate"    : base_rate,
                "expansion_df" : exp_rows.copy(),
            }

    return results

