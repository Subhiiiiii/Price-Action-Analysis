# market_analysis/finding_threshold/absorption_v2_threshold.py
"""
Absorption Method-2 threshold analysis.

Entry point for the notebook:

    from market_analysis.finding_threshold.absorption_v2_threshold import (
        run_absorption_v2_threshold_analysis,
    )

    results = run_absorption_v2_threshold_analysis(
        product_code = "BO",
        timeframe    = "1d",         # "1d" | "1h"
        months_back  = 7,
        score_grid   = np.arange(0.05, 1.0, 0.05),
        front_n      = 5,
    )

What it does per instrument (spread × front_n + jumbo × front_n):
  1. Fetch 1D or 1H OHLC (same helpers as the rest of the project).
  2. Run calculate_absorption_v2() — Method-2 z-score scorer.
  3. Label with label_absorption_atr() (ATR-based, profit-first).
  4. Sweep absorption_score thresholds → F1 / Precision / Recall / MCC / Lift
     table per instrument (BUY + SELL pooled).
  5. Show a 3-panel distribution plot: Success | Failure | Overlay.

ATR labeling logic
------------------
    BUY  direction (expect price DOWN):
        target = close[t] - ATR[t] * target_atr_mult
        stop   = close[t] + ATR[t] * stop_atr_mult

    SELL direction (expect price UP):
        target = close[t] + ATR[t] * target_atr_mult
        stop   = close[t] - ATR[t] * stop_atr_mult

    Forward scan: profit-first (target hit before stop = success).
    Ambiguous same-bar hits resolved with hourly data when available.

All config defaults come from config.ABSORPTION_V2 (or fall back to
ABSORPTION if that key is absent).
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
from IPython.display import display

# ── project root ──────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent   # .../finding_threshold/
_ROOT = _HERE.parent                       # .../market_analysis/
_DOCS = _ROOT.parent                       # .../Documents/
for _p in [str(_DOCS), str(_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

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
from market_analysis.signals.absorption.calculate_v2 import calculate_absorption_v2
from market_analysis.signals.absorption.label import label_absorption_atr
from market_analysis.utils.metrics import _block_bootstrap_ci

try:
    from sklearn.metrics import matthews_corrcoef, precision_recall_curve, auc as sk_auc
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

# ── colours ───────────────────────────────────────────────────────────────────
_C_SUCCESS = "seagreen"
_C_FAILURE = "tomato"
_C_SWEEP   = "steelblue"


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

# Default Method-2 params — override any key via the v2_params argument.
_DEFAULT_V2_PARAMS: dict = {
    # z-score model
    "volume_window"      : 5,
    "atr_window"         : 5,
    "baseline_window"    : 15,
    "delta_z_threshold"  : 2.0,
    "price_z_threshold"  : 0.75,
    # ATR labeling
    "atr_period"         : 14,
    "target_atr_mult"    : 1.0,
    "stop_atr_mult"      : 1.0,
    "forward_days"       : 10,
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _months_back_start(months: int) -> str:
    today = datetime.date.today()
    year  = today.year
    month = today.month - months
    while month <= 0:
        month += 12
        year  -= 1
    day = min(today.day, calendar.monthrange(year, month)[1])
    return datetime.date(year, month, day).strftime("%Y-%m-%d")


def _make_threshold_table(
    labeled_df   : pd.DataFrame,
    score_col    : str,
    success_col  : str,
    score_grid   : np.ndarray,
    min_n        : int   = 8,
    n_bootstrap  : int   = 500,
    random_seed  : int   = 42,
    forward_bars : int   = 10,
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

        mcc = np.nan
        if _SKLEARN and sig.sum() > 0 and (~sig).sum() > 0:
            try:
                mcc = float(matthews_corrcoef(lab, sig.astype(int)))
            except Exception:
                pass

        ci     = _block_bootstrap_ci(sig, labels_arr, n_bootstrap, block_len, random_seed)
        ci_str = (f"[{ci[0]:.3f}, {ci[1]:.3f}]" if np.isfinite(ci[0]) else "n/a")

        rows.append({
            "Score Thresh" : round(T, 3),
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


def _pr_auc(labeled_df: pd.DataFrame, score_col: str, success_col: str) -> float:
    if not _SKLEARN:
        return np.nan
    valid = labeled_df.dropna(subset=[success_col])
    if valid[success_col].nunique() < 2:
        return np.nan
    prec_c, rec_c, _ = precision_recall_curve(
        valid[success_col].astype(int), valid[score_col]
    )
    return float(sk_auc(rec_c, prec_c))


def _styled_table(table: pd.DataFrame, caption: str = "") -> object:
    float_cols = {
        c: "{:.3f}" for c in table.columns
        if pd.api.types.is_float_dtype(table[c])
    }
    float_cols["Score Thresh"] = "{:.3f}"
    return (
        table.style
        .format(float_cols, na_rep="—")
        .apply(
            lambda row: [
                "background-color: #fff3cd" if row["Low N"] else ""
            ] * len(row),
            axis=1,
        )
        .background_gradient(subset=["F1"],        cmap="RdYlGn", vmin=0.3, vmax=0.7)
        .background_gradient(subset=["Precision"], cmap="Blues",  vmin=0.3, vmax=0.8)
        .set_caption(caption)
    )


def _distribution_plot(
    labeled_df  : pd.DataFrame,
    score_col   : str,
    success_col : str,
    instrument  : str,
    timeframe   : str,
    v2_params   : dict,
) -> plt.Figure:
    """
    3-panel distribution:
        Panel 1  — Success histogram
        Panel 2  — Failure histogram
        Panel 3  — Density overlay of both
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
    base   = n_succ / n_valid

    s_scores = valid.loc[valid[success_col] == 1.0, score_col]
    f_scores = valid.loc[valid[success_col] == 0.0, score_col]
    bins     = np.arange(0, 1.05, 0.05)

    fig = plt.figure(figsize=(17, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    for ax, data, label, color in [
        (ax1, s_scores, f"Success  n={n_succ}", _C_SUCCESS),
        (ax2, f_scores, f"Failure  n={n_fail}", _C_FAILURE),
    ]:
        ax.hist(data, bins=bins, color=color, edgecolor="white", alpha=0.85)
        ax.set_title(label, fontweight="bold", fontsize=11)
        ax.set_xlabel("Absorption Score (Method 2)")
        ax.set_ylabel("Count")
        ax.set_xlim([0, 1])
        ax.grid(axis="y", alpha=0.3)

    ax3.hist(s_scores, bins=bins, color=_C_SUCCESS, edgecolor="white",
             alpha=0.55, density=True, label=f"Success n={n_succ}")
    ax3.hist(f_scores, bins=bins, color=_C_FAILURE, edgecolor="white",
             alpha=0.55, density=True, label=f"Fail n={n_fail}")
    ax3.set_title("Overlay (density)", fontweight="bold", fontsize=11)
    ax3.set_xlabel("Absorption Score (Method 2)")
    ax3.set_ylabel("Density")
    ax3.set_xlim([0, 1])
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    X = v2_params["delta_z_threshold"]
    Y = v2_params["price_z_threshold"]
    T = v2_params["target_atr_mult"]
    S = v2_params["stop_atr_mult"]
    fwd = v2_params["forward_days"]

    fig.suptitle(
        f"{instrument}  [{timeframe}]  —  Absorption Method 2  "
        f"|  Δz≥{X}  Pz≤{Y}  |  ATR×{T}/×{S}  |  fwd={fwd}d\n"
        f"valid={n_valid}  success={n_succ}  fail={n_fail}  base={base:.3f}",
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
    v2_params    : dict,
    tick_size    : float,
    n_rows       : int,
) -> None:
    line = "─" * 70
    print(f"\n{line}")
    print(f"  {structure.upper()}  ▶  {instrument}  [{timeframe}]  ({product_code})")
    print(line)
    print(f"  Data window       : {start_date}  →  {end_date}  ({n_rows} bars)")
    print(f"  Tick size         : {tick_size}")
    print(f"  ── Method-2 scorer ──────────────────────────────────────")
    print(f"  volume_window     : {v2_params['volume_window']}")
    print(f"  atr_window        : {v2_params['atr_window']}")
    print(f"  baseline_window   : {v2_params['baseline_window']}")
    print(f"  delta_z_threshold : {v2_params['delta_z_threshold']}")
    print(f"  price_z_threshold : {v2_params['price_z_threshold']}")
    print(f"  ── ATR labeling ─────────────────────────────────────────")
    print(f"  atr_period        : {v2_params['atr_period']}")
    print(f"  target_atr_mult   : {v2_params['target_atr_mult']}")
    print(f"  stop_atr_mult     : {v2_params['stop_atr_mult']}")
    print(f"  forward_days      : {v2_params['forward_days']}")
    print(line)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_absorption_v2_threshold_analysis(
    product_code     : str,
    timeframe        : str               = "1d",
    months_back      : int               = 7,
    score_grid       : np.ndarray | None = None,
    front_n          : int               = 5,
    v2_params        : dict | None       = None,
    min_signal_count : int               = 8,
    n_bootstrap      : int               = 500,
    random_seed      : int               = 42,
    show_plots       : bool              = True,
    hourly_df_map    : dict | None       = None,
) -> dict:
    """
    Run Method-2 absorption threshold analysis for all front-N instruments.

    Parameters
    ----------
    product_code      : e.g. "BO" or "SM".
    timeframe         : "1d" (daily OHLC) or "1h" (hourly OHLC).
    months_back       : Calendar months of history to fetch.
    score_grid        : Threshold grid for the sweep (default 0.05..0.95 step 0.05).
    front_n           : Front-N spreads and jumbos each.
    v2_params         : Override any Method-2 or ATR labeling param.
                        Keys: volume_window, atr_window, baseline_window,
                              delta_z_threshold, price_z_threshold,
                              atr_period, target_atr_mult, stop_atr_mult,
                              forward_days.
    min_signal_count  : N < this is flagged yellow in the sweep table.
    n_bootstrap       : Bootstrap replications for Precision CI.
    random_seed       : RNG seed.
    show_plots        : Display distribution plots inline.
    hourly_df_map     : Optional {instrument: hourly_df} pre-loaded 1H data
                        for same-bar ambiguity resolution in the ATR labeler
                        (only relevant when timeframe="1d").

    Returns
    -------
    dict keyed by instrument string:
        "labeled_df"   : fully labelled DataFrame
        "sweep_table"  : threshold sweep DataFrame
        "base_rate"    : success base rate
        "pr_auc"       : PR-AUC
        "v2_params"    : the params used for this run
    """
    # ── merge user params over defaults ───────────────────────────────────────
    params = dict(_DEFAULT_V2_PARAMS)
    if v2_params:
        params.update(v2_params)

    # ── score grid ────────────────────────────────────────────────────────────
    if score_grid is None:
        score_grid = np.round(np.arange(0.05, 1.00, 0.05), 3)

    start_date = _months_back_start(months_back)
    end_date   = datetime.date.today().strftime("%Y-%m-%d")
    is_1h      = timeframe == "1h"
    tick_size  = PRODUCTS[product_code]["tick_size"]

    all_instruments = get_all_front_instruments(
        product_code = product_code,
        front_n      = front_n,
    )

    results: dict[str, dict] = {}

    for structure, instruments in all_instruments.items():
        print(f"\n{'#' * 72}")
        print(f"#  STRUCTURE : {structure.upper()}  —  {product_code}  [{timeframe}]  (Method 2)")
        print(f"{'#' * 72}")

        for instrument in instruments:

            # ── 1. Fetch OHLC ─────────────────────────────────────────────────
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
                nb = 0 if df_raw is None else len(df_raw)
                print(f"\n  [SKIP] {instrument} — insufficient data ({nb} bars)")
                continue

            _print_instrument_header(
                instrument   = instrument,
                structure    = structure,
                product_code = product_code,
                timeframe    = timeframe,
                start_date   = start_date,
                end_date     = end_date,
                v2_params    = params,
                tick_size    = tick_size,
                n_rows       = len(df_raw),
            )

            # ── 2. Score — Method 2 ───────────────────────────────────────────
            try:
                df_scored = calculate_absorption_v2(
                    df_raw,
                    volume_window      = params["volume_window"],
                    atr_window         = params["atr_window"],
                    baseline_window    = params["baseline_window"],
                    delta_z_threshold  = params["delta_z_threshold"],
                    price_z_threshold  = params["price_z_threshold"],
                )
            except Exception as exc:
                print(f"  [WARN] Scoring failed for {instrument}: {exc}")
                continue

            # ── 3. Label — ATR profit-first ───────────────────────────────────
            h_df = (hourly_df_map or {}).get(instrument)

            try:
                df_labeled = label_absorption_atr(
                    df_scored,
                    atr_period      = params["atr_period"],
                    target_atr_mult = params["target_atr_mult"],
                    stop_atr_mult   = params["stop_atr_mult"],
                    forward_days    = params["forward_days"],
                    hourly_df       = h_df,
                )
            except Exception as exc:
                print(f"  [WARN] Labeling failed for {instrument}: {exc}")
                continue

            # ── 4. Filter to directional rows only ────────────────────────────
            df_labeled = df_labeled[
                df_labeled["absorption_direction"].isin(["BUY", "SELL"])
            ].dropna(subset=["absorption_score"]).copy()

            success_col = "sl_success"

            # ── 5. Outcome summary ────────────────────────────────────────────
            valid_rows = df_labeled.dropna(subset=[success_col])
            n_total    = len(df_labeled)
            n_valid    = len(valid_rows)
            n_succ     = int((valid_rows[success_col] == 1.0).sum())
            n_fail     = int((valid_rows[success_col] == 0.0).sum())
            n_amb      = int((df_labeled.get("outcome", pd.Series()) == "AMBIGUOUS").sum())
            n_unres    = int((df_labeled.get("outcome", pd.Series()) == "UNRESOLVED").sum())

            print(f"\n  Total directional bars  : {n_total}")
            print(f"  Valid (non-NaN success) : {n_valid}")
            print(f"  ├── Success   (1)       : {n_succ}"
                  + (f"   ({100*n_succ/n_valid:.1f}%)" if n_valid else ""))
            print(f"  ├── Failure   (0)       : {n_fail}"
                  + (f"   ({100*n_fail/n_valid:.1f}%)" if n_valid else ""))
            print(f"  ├── Ambiguous (NaN)     : {n_amb}")
            print(f"  └── Unresolved (NaN)    : {n_unres}")

            if n_valid == 0:
                print("  [SKIP] No valid labelled rows.")
                continue

            # direction breakdown
            for d in ("BUY", "SELL"):
                sub = df_labeled[
                    df_labeled["absorption_direction"] == d
                ].dropna(subset=[success_col])
                if len(sub):
                    sr = float((sub[success_col] == 1.0).mean())
                    print(f"  {d:<4} valid={len(sub)}  success_rate={sr:.3f}")

            # ── 6. Threshold sweep ────────────────────────────────────────────
            sweep, base_rate = _make_threshold_table(
                labeled_df   = df_labeled,
                score_col    = "absorption_score",
                success_col  = success_col,
                score_grid   = score_grid,
                min_n        = min_signal_count,
                n_bootstrap  = n_bootstrap,
                random_seed  = random_seed,
                forward_bars = params["forward_days"],
            )

            auc_val = _pr_auc(df_labeled, "absorption_score", success_col)

            br_str = f"{base_rate:.3f}" if np.isfinite(base_rate) else "n/a"
            auc_str = f"{auc_val:.3f}"  if np.isfinite(auc_val) else "n/a"
            print(f"\n  ── Combined BUY + SELL sweep  "
                  f"(base rate={br_str}  PR-AUC={auc_str})")

            if sweep.empty:
                print("  No valid labelled rows for sweep.")
            else:
                caption = (
                    f"{instrument}  [{timeframe}]  Method 2  "
                    f"|  Δz≥{params['delta_z_threshold']}  Pz≤{params['price_z_threshold']}  "
                    f"|  ATR×{params['target_atr_mult']}/×{params['stop_atr_mult']}  "
                    f"|  fwd={params['forward_days']}d  "
                    f"|  months={months_back}  |  base={br_str}  "
                    f"|  Yellow=N<{min_signal_count}"
                )
                display(_styled_table(sweep, caption))

            # ── 7. Distribution plot ──────────────────────────────────────────
            if show_plots and not sweep.empty:
                fig = _distribution_plot(
                    labeled_df  = df_labeled,
                    score_col   = "absorption_score",
                    success_col = success_col,
                    instrument  = instrument,
                    timeframe   = timeframe,
                    v2_params   = params,
                )
                plt.show()
                plt.close(fig)

            results[instrument] = {
                "labeled_df"  : df_labeled,
                "sweep_table" : sweep,
                "base_rate"   : base_rate,
                "pr_auc"      : auc_val,
                "v2_params"   : dict(params),
            }

    print(f"\n{'═'*70}")
    print(f"  ABSORPTION V2 COMPLETE  ▸  {product_code}  [{timeframe}]  "
          f"▸  {len(results)} instrument(s)")
    print(f"{'═'*70}")
    return results
