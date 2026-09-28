# market_analysis/runner.py
"""
Main notebook interface.

Usage:
    import sys; sys.path.insert(0, r"C:\path\to\parent_of_market_analysis")
    from market_analysis.runner import run_absorption, run_compression

    abs_results = run_absorption("SM", horizons=["1d","2d","3d","5d"])
    comp_results = run_compression("SM", horizons=["1d","2d","3d","5d"])

Each function:
  1. Fetches OHLC from the API (cached; wipe with CLEAN_PREVIOUS_DATA=True in config)
  2. Calculates the signal score
  3. Labels each row with success/fail
  4. Displays: distribution plot (inline) + styled F1/Precision/MCC table (inline)
  5. Saves: labeled CSV, sweep CSV, distribution PNG per instrument × horizon

Output is organised by structure type (spread / jumbo) with clear print separators.
Results dict: results[instrument][horizon] = {"table", "base_rate", "pr_auc",
                                               "scored", "fig_dist", "styled_table"}
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import (
    PRODUCTS, STRUCTURES, HORIZONS, FRONT_N, REFERENCE_DATE,
    DATA_DIR, OUTPUT_DIR, CLEAN_PREVIOUS_DATA,
    ABSORPTION, COMPRESSION,
    ABSORPTION_1H, HORIZON_BARS_1H, VOLUME_MIN_1H,
)
from .utils.contracts import get_all_front_instruments
from .data.ohlc import (
    get_all_horizon_ohlc, getData_OHLC_v2 as _default_fetch,
    fetch_ohlc_1h, get_all_horizon_ohlc_1h,
)
from .signals.absorption.calculate import calculate_absorption
from .signals.absorption.label import label_absorption
from .signals.absorption.label_hourly import label_absorption_hourly
from .signals.compression.calculate import calculate_compression
from .signals.compression.label import label_compression
from .utils.metrics import threshold_sweep, pr_auc
from .utils.plotting import plot_distribution, styled_sweep_table


# ── helpers ───────────────────────────────────────────────────────────────────

def _setup_output(product_code: str, signal: str, horizon: str) -> Path:
    d = OUTPUT_DIR / product_code.lower() / signal / horizon
    d.mkdir(parents=True, exist_ok=True)
    return d


def _clean_if_needed(product_code: str) -> None:
    if CLEAN_PREVIOUS_DATA:
        for folder in [DATA_DIR / product_code.lower(),
                        OUTPUT_DIR / product_code.lower()]:
            if folder.exists():
                shutil.rmtree(folder)
                print(f"  [CLEAN] {folder}")


def _section(text: str, char: str = "═", width: int = 70) -> None:
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def _subsection(text: str, width: int = 70) -> None:
    print(f"\n{'─' * width}")
    print(f"  {text}")
    print(f"{'─' * width}")


def _print_table(tbl: pd.DataFrame, base_rate: float, pr_auc_val: float) -> None:
    """Print the sweep table cleanly as plain text."""
    display_cols = ["Threshold", "N", "Precision", "Prec CI 95%",
                    "Recall", "F1", "MCC", "Lift"]
    t = tbl[display_cols].copy()
    # format floats
    for col in ["Precision", "Recall", "F1", "MCC", "Lift"]:
        t[col] = t[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    t["N"] = t["N"].astype(int)
    print(t.to_string(index=False))
    print(f"\n  Base rate: {base_rate:.3f}  |  PR-AUC: {pr_auc_val:.3f}  "
          f"|  Lift > 1 and MCC > 0 = signal beats chance")


# ── absorption ────────────────────────────────────────────────────────────────

def run_absorption(
    product_code: str,
    horizons: list[str] | None = None,
    getData_OHLC_v2: Callable | None = None,
    front_n: int = FRONT_N,
    show_plots: bool = True,
    save_outputs: bool = True,
    cfg: dict | None = None,
) -> dict:
    """
    Run absorption analysis for front_n spreads + front_n jumbos.

    Outputs per instrument × horizon:
      - Distribution plot (Success vs Fail, shown inline)
      - F1 / Precision / Recall / MCC / Lift table (printed)
    Saves labeled CSV, sweep CSV, distribution PNG to OUTPUT_DIR.
    """
    if cfg is None:     cfg = ABSORPTION
    if horizons is None: horizons = HORIZONS
    if getData_OHLC_v2 is None: getData_OHLC_v2 = _default_fetch

    _clean_if_needed(product_code)
    product  = PRODUCTS[product_code]
    tick     = product["tick_size"]
    by_struct = get_all_front_instruments(product_code, REFERENCE_DATE, front_n)

    _section(f"ABSORPTION  ▸  {product_code}  ▸  horizons: {horizons}")
    for struct, instrs in by_struct.items():
        print(f"  {struct.upper():8s}: {instrs}")

    results: dict[str, dict] = {}

    for struct_name, instruments in by_struct.items():

        _section(f"STRUCTURE: {struct_name.upper()}  ({product_code}  absorption)",
                 char="=")

        for instrument in instruments:
            _subsection(f"{struct_name.upper()}  ▸  {instrument}")

            horizon_ohlc = get_all_horizon_ohlc(
                instrument, product_code, horizons, getData_OHLC_v2)
            if not horizon_ohlc:
                print(f"  [SKIP] No OHLC data for {instrument}")
                continue

            results.setdefault(instrument, {})

            for horizon in horizons:
                ohlc = horizon_ohlc.get(horizon)
                if ohlc is None or len(ohlc) == 0:
                    continue

                print(f"\n  ◆ {instrument}  [{horizon}]  ({len(ohlc)} bars)")

                # score
                try:
                    scored = calculate_absorption(
                        ohlc,
                        window             = cfg.get("window", 60),
                        tick_size          = tick,
                        strength_threshold = cfg.get("strength_threshold", 0.60),
                    )
                except Exception as e:
                    print(f"    [WARN] Score failed: {e}"); continue

                # label
                labeled = label_absorption(
                    scored,
                    tick                = tick,
                    required_move_ticks = cfg["required_move_ticks"],
                    sl_lookback         = cfg["sl_lookback"],
                    forward_days        = cfg["forward_days"],
                )
                labeled = labeled[
                    labeled["absorption_direction"].isin(["BUY", "SELL"])
                ].dropna(subset=["absorption_strength"]).copy()
                labeled["success"] = labeled["sl_success"]

                valid  = labeled.dropna(subset=["success"])
                n_val  = len(valid)
                n_succ = int((valid["success"] == 1).sum())
                if n_val == 0:
                    print("    [SKIP] No valid labeled rows."); continue

                print(f"    valid={n_val}  success={n_succ}  "
                      f"fail={n_val-n_succ}  "
                      f"base rate={n_succ/n_val:.3f}")

                # sweep
                tbl, br = threshold_sweep(labeled, "absorption_strength",
                                          success_col="success", cfg=dict(cfg))
                auc_val = pr_auc(labeled, "absorption_strength", "success")

                # distribution plot (only graph)
                title  = f"{struct_name.upper()} ▸ {instrument} ▸ {horizon} ▸ absorption"
                fig_d  = plot_distribution(
                    labeled, "absorption_strength", "success",
                    cfg.get("threshold_line"), title)
                if show_plots:
                    plt.figure(fig_d.number); plt.show()

                # F1 / metric table
                print(f"\n    ── Threshold Table  [{instrument}  {horizon}] ──")
                _print_table(tbl, br, auc_val)

                # save
                if save_outputs:
                    out  = _setup_output(product_code, "absorption", horizon)
                    safe = instrument.replace("-", "_")
                    labeled.to_csv(out / f"{safe}_labeled.csv",
                                   index=False, encoding="utf-8-sig")
                    tbl.to_csv(out / f"{safe}_sweep.csv",
                               index=False, encoding="utf-8-sig")
                    fig_d.savefig(out / f"{safe}_{struct_name}_dist.png",
                                  dpi=120, bbox_inches="tight")

                results[instrument][horizon] = {
                    "scored"      : labeled,
                    "table"       : tbl,
                    "base_rate"   : br,
                    "pr_auc"      : auc_val,
                    "fig_dist"    : fig_d,
                    "styled_table": styled_sweep_table(
                        tbl, f"{instrument} {horizon} | base={br:.3f}"
                    ),
                }
                plt.close("all")

    _section(f"ABSORPTION COMPLETE  ▸  {product_code}  ▸  "
             f"{sum(len(v) for v in results.values())} results")
    return results


# ── compression ───────────────────────────────────────────────────────────────

def run_compression(
    product_code: str,
    horizons: list[str] | None = None,
    getData_OHLC_v2: Callable | None = None,
    front_n: int = FRONT_N,
    show_plots: bool = True,
    save_outputs: bool = True,
    cfg: dict | None = None,
) -> dict:
    """
    Run compression analysis for front_n spreads + front_n jumbos.

    Outputs per instrument × horizon:
      - Distribution plot (Success vs Fail, shown inline)
      - F1 / Precision / Recall / MCC / Lift table (printed)
    """
    if cfg is None:      cfg = COMPRESSION
    if horizons is None: horizons = HORIZONS
    if getData_OHLC_v2 is None: getData_OHLC_v2 = _default_fetch

    _clean_if_needed(product_code)
    product   = PRODUCTS[product_code]
    tick      = product["tick_size"]
    by_struct = get_all_front_instruments(product_code, REFERENCE_DATE, front_n)

    _section(f"COMPRESSION  ▸  {product_code}  ▸  horizons: {horizons}")
    for struct, instrs in by_struct.items():
        print(f"  {struct.upper():8s}: {instrs}")

    results: dict[str, dict] = {}

    for struct_name, instruments in by_struct.items():

        _section(f"STRUCTURE: {struct_name.upper()}  ({product_code}  compression)",
                 char="=")

        for instrument in instruments:
            _subsection(f"{struct_name.upper()}  ▸  {instrument}")

            horizon_ohlc = get_all_horizon_ohlc(
                instrument, product_code, horizons, getData_OHLC_v2)
            if not horizon_ohlc:
                print(f"  [SKIP] No OHLC data for {instrument}")
                continue

            results.setdefault(instrument, {})

            for horizon in horizons:
                ohlc = horizon_ohlc.get(horizon)
                if ohlc is None or len(ohlc) == 0:
                    continue

                print(f"\n  ◆ {instrument}  [{horizon}]  ({len(ohlc)} bars)")

                # score
                try:
                    scored_df, _ = calculate_compression(
                        ohlc,
                        range_baseline        = cfg.get("range_baseline", 20),
                        atr_period            = cfg.get("atr_period", 5),
                        rv_period             = cfg.get("rv_period", 5),
                        persistence_window    = cfg.get("persistence_window", 20),
                        compression_threshold = cfg.get("compression_threshold", 0.60),
                    )
                except Exception as e:
                    print(f"    [WARN] Score failed: {e}"); continue

                # label
                labeled = label_compression(
                    scored_df, tick=tick,
                    k=cfg["k"], forward_days=cfg["forward_days"])

                valid  = labeled.dropna(subset=["success"])
                n_val  = len(valid)
                n_succ = int((valid["success"] == 1).sum())
                if n_val == 0:
                    print("    [SKIP] No valid labeled rows."); continue

                print(f"    valid={n_val}  success={n_succ}  "
                      f"fail={n_val-n_succ}  "
                      f"base rate={n_succ/n_val:.3f}")

                # sweep
                tbl, br = threshold_sweep(labeled, "compression_score",
                                          success_col="success", cfg=cfg)
                auc_val = pr_auc(labeled, "compression_score", "success")

                # distribution plot
                title = f"{struct_name.upper()} ▸ {instrument} ▸ {horizon} ▸ compression"
                fig_d = plot_distribution(
                    labeled, "compression_score", "success",
                    cfg.get("threshold_line", 0.40), title)
                if show_plots:
                    plt.figure(fig_d.number); plt.show()

                # F1 / metric table
                print(f"\n    ── Threshold Table  [{instrument}  {horizon}] ──")
                _print_table(tbl, br, auc_val)

                # save
                if save_outputs:
                    out  = _setup_output(product_code, "compression", horizon)
                    safe = instrument.replace("-", "_")
                    labeled.to_csv(out / f"{safe}_labeled.csv",
                                   index=False, encoding="utf-8-sig")
                    tbl.to_csv(out / f"{safe}_sweep.csv",
                               index=False, encoding="utf-8-sig")
                    fig_d.savefig(out / f"{safe}_{struct_name}_dist.png",
                                  dpi=120, bbox_inches="tight")

                results[instrument][horizon] = {
                    "scored"      : labeled,
                    "table"       : tbl,
                    "base_rate"   : br,
                    "pr_auc"      : auc_val,
                    "fig_dist"    : fig_d,
                    "styled_table": styled_sweep_table(
                        tbl, f"{instrument} {horizon} | base={br:.3f}"
                    ),
                }
                plt.close("all")

    _section(f"COMPRESSION COMPLETE  ▸  {product_code}  ▸  "
             f"{sum(len(v) for v in results.values())} results")
    return results


# ── 1H absorption ─────────────────────────────────────────────────────────────

def run_absorption_1h(
    product_code: str,
    horizons: list[str] | None = None,
    getData_OHLC_v2: Callable | None = None,
    front_n: int = FRONT_N,
    show_plots: bool = True,
    save_outputs: bool = True,
    cfg: dict | None = None,
    volume_min: int | None = None,
    start_date: str | None = None,   # per-run override; defaults to last 6 months
    end_date:   str | None = None,
) -> dict:
    """
    Run 1H absorption analysis using profit-first labeling.

    Key differences vs run_absorption():
      - Fetches interval=1H bars via fetch_ohlc_1h().
      - Aggregates multi-bar horizons with _aggregate_trailing_1h()
        which respects session breaks (Fri-close → Mon-open gaps).
      - Labels with label_absorption_hourly() — profit-first scan.
      - Default horizons: ["1h","2h","4h","8h","16h"].
      - Default volume filter: VOLUME_MIN_1H = 500 (lower than daily 3000).
      - calculate_absorption window: 120 bars (≈7.5 trading days).
      - Extra columns in results: profit_hit, stop_hit_first, profit_bar_offset.

    Result dict shape mirrors run_absorption():
      results[instrument][horizon] = {
          "scored", "table", "base_rate", "pr_auc", "fig_dist", "styled_table"
      }
    """
    if cfg is None:      cfg = ABSORPTION_1H
    if horizons is None: horizons = list(HORIZON_BARS_1H.keys())  # ["1h","2h","4h","8h","16h"]
    if getData_OHLC_v2 is None: getData_OHLC_v2 = _default_fetch
    vol_min = volume_min if volume_min is not None else VOLUME_MIN_1H

    _clean_if_needed(product_code)
    product   = PRODUCTS[product_code]
    tick      = product["tick_size"]
    by_struct = get_all_front_instruments(product_code, REFERENCE_DATE, front_n)

    _section(f"ABSORPTION 1H  ▸  {product_code}  ▸  horizons: {horizons}")
    for struct, instrs in by_struct.items():
        print(f"  {struct.upper():8s}: {instrs}")
    print(f"  Volume filter : >{vol_min}  (1H)")

    results: dict[str, dict] = {}

    for struct_name, instruments in by_struct.items():

        _section(f"STRUCTURE: {struct_name.upper()}  ({product_code}  absorption 1H)",
                 char="=")

        for instrument in instruments:
            _subsection(f"{struct_name.upper()}  ▸  {instrument}  [1H]")

            horizon_ohlc = get_all_horizon_ohlc_1h(
                instrument, product_code, horizons, getData_OHLC_v2,
                start_date=start_date, end_date=end_date,
            )
            if not horizon_ohlc:
                print(f"  [SKIP] No 1H OHLC data for {instrument}")
                continue

            results.setdefault(instrument, {})

            for horizon in horizons:
                ohlc = horizon_ohlc.get(horizon)
                if ohlc is None or len(ohlc) == 0:
                    continue

                # warm-up filter: first window bars are unreliable
                warmup = cfg.get("window", 120)
                if len(ohlc) <= warmup:
                    print(f"  [SKIP] {instrument} [{horizon}]: only {len(ohlc)} bars, "
                          f"need >{warmup} for warm-up")
                    continue

                print(f"\n  ◆ {instrument}  [{horizon}]  ({len(ohlc)} bars)")

                # ── score ────────────────────────────────────────────────────
                try:
                    scored = calculate_absorption(
                        ohlc,
                        window             = cfg.get("window", 120),
                        tick_size          = tick,
                        strength_threshold = cfg.get("strength_threshold", 0.30),
                    )
                except Exception as e:
                    print(f"    [WARN] Score failed: {e}"); continue

                # ── label (profit-first) ──────────────────────────────────────
                labeled = label_absorption_hourly(
                    scored,
                    tick                = tick,
                    required_move_ticks = cfg["required_move_ticks"],
                    sl_lookback         = cfg["sl_lookback"],
                    forward_bars        = cfg["forward_bars"],
                )

                # filter: known direction, volume, drop warm-up rows
                labeled = labeled[
                    labeled["absorption_direction"].isin(["BUY", "SELL"])
                ].dropna(subset=["absorption_strength"]).copy()

                # warm-up: iloc[warmup:] already scored as NaN via calculate_absorption
                # but label rows may still be NaN in sl_success — that's correct (censored)
                labeled["success"] = labeled["sl_success"]

                # volume filter
                if "volume" in labeled.columns:
                    labeled = labeled[labeled["volume"] > vol_min].copy()

                valid  = labeled.dropna(subset=["success"])
                n_val  = len(valid)
                n_succ = int((valid["success"] == 1).sum())
                n_ph   = int(valid.get("profit_hit", pd.Series(0)).sum())

                if n_val == 0:
                    print("    [SKIP] No valid labeled rows."); continue

                print(f"    valid={n_val}  success={n_succ}  "
                      f"fail={n_val - n_succ}  "
                      f"base rate={n_succ / n_val:.3f}  "
                      f"profit_hit_first={n_ph}")

                # per-instrument event count (log for overlap awareness)
                print(f"    instrument={instrument}  horizon={horizon}  "
                      f"events_contributing={n_val}")

                # ── threshold sweep ───────────────────────────────────────────
                tbl, br = threshold_sweep(labeled, "absorption_strength",
                                          success_col="success", cfg=dict(cfg))
                auc_val = pr_auc(labeled, "absorption_strength", "success")

                # ── distribution plot ─────────────────────────────────────────
                title = (f"{struct_name.upper()} ▸ {instrument} ▸ {horizon} "
                         f"▸ absorption 1H  (profit-first)")
                fig_d = plot_distribution(
                    labeled, "absorption_strength", "success",
                    cfg.get("threshold_line"), title)
                if show_plots:
                    plt.figure(fig_d.number); plt.show()

                print(f"\n    ── Threshold Table  [{instrument}  {horizon}] ──")
                _print_table(tbl, br, auc_val)

                # ── save ─────────────────────────────────────────────────────
                if save_outputs:
                    out  = _setup_output(product_code, "absorption_1h", horizon)
                    safe = instrument.replace("-", "_")
                    labeled.to_csv(out / f"{safe}_1h_labeled.csv",
                                   index=False, encoding="utf-8-sig")
                    tbl.to_csv(out / f"{safe}_1h_sweep.csv",
                               index=False, encoding="utf-8-sig")
                    fig_d.savefig(out / f"{safe}_{struct_name}_1h_dist.png",
                                  dpi=120, bbox_inches="tight")

                results[instrument][horizon] = {
                    "scored"      : labeled,
                    "table"       : tbl,
                    "base_rate"   : br,
                    "pr_auc"      : auc_val,
                    "fig_dist"    : fig_d,
                    "styled_table": styled_sweep_table(
                        tbl, f"{instrument} {horizon} 1H | base={br:.3f}"
                    ),
                }
                plt.close("all")

    _section(f"ABSORPTION 1H COMPLETE  ▸  {product_code}  ▸  "
             f"{sum(len(v) for v in results.values())} results")
    return results

# ── ADD THIS TO market_analysis/runner.py ─────────────────────────────────────
#
# Paste this function at the bottom of runner.py (after run_absorption_1h).
# Also add to the imports at the top of runner.py:
#
#   from .signals.absorption.calculate_rolling import calculate_absorption_rolling
#
# ─────────────────────────────────────────────────────────────────────────────
"""
run_absorption_rolling()
========================
Drop-in companion to run_absorption() that uses rolling N-day candles
(combined_candle = 1 / 2 / 3 / 5) with a correctly anchored causal baseline.

Key difference vs run_absorption():
    run_absorption() calls get_all_horizon_ohlc() which builds the rolling
    candles in ohlc.py and then passes them to calculate_absorption().
    That scorer's rolling windows look back from the *last* bar of each
    combined candle, so the baseline can "see inside" the candle.

    run_absorption_rolling() feeds the raw 1D bars to
    calculate_absorption_rolling() which:
        1. Builds the rolling N-day candle (same logic as _aggregate_trailing).
        2. Anchors every statistical baseline at the bar *before* the candle's
           first constituent day — so a Wednesday 3D candle (Mon-Tue-Wed) only
           uses data up to Sunday.

Usage (from notebook)
---------------------
    from market_analysis.runner import run_absorption_rolling

    results = run_absorption_rolling(
        "SM",
        horizons        = ["1d", "2d", "3d", "5d"],
        method          = "v2",       # "v1" for percentile scorer
        show_plots      = True,
        save_outputs    = True,
    )
    # results[instrument][horizon] = {"scored", "table", "base_rate", ...}
"""

from __future__ import annotations
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# These imports already exist in runner.py — repeated here only for the
# standalone patch context.
from .config import (
    PRODUCTS, STRUCTURES, HORIZONS, FRONT_N, REFERENCE_DATE,
    DATA_DIR, OUTPUT_DIR,
    ABSORPTION, HORIZON_BARS,
)
from .utils.contracts import get_all_front_instruments
from .data.ohlc import fetch_ohlc_1d, getData_OHLC_v2 as _default_fetch
from .signals.absorption.calculate_rolling import calculate_absorption_rolling
from .signals.absorption.label import label_absorption
from .utils.metrics import threshold_sweep, pr_auc
from .utils.plotting import plot_distribution, styled_sweep_table


def run_absorption_rolling(
    product_code: str,
    horizons: list[str] | None = None,
    method: str = "v2",
    getData_OHLC_v2: Callable | None = None,
    front_n: int = FRONT_N,
    show_plots: bool = True,
    save_outputs: bool = True,
    cfg: dict | None = None,
    # v2 scorer params (ignored when method="v1")
    volume_window: int = 5,
    atr_window: int = 5,
    baseline_window: int = 15,
    delta_z_threshold: float = 2.0,
    price_z_threshold: float = 0.75,
) -> dict:
    """
    Rolling N-day candle absorption analysis with causal-baseline correction.

    Parameters
    ----------
    product_code       : e.g. "SM", "BO"  (must be in config.PRODUCTS)
    horizons           : list of horizon strings, e.g. ["1d","2d","3d","5d"]
                         Maps via HORIZON_BARS to combined_candle values.
    method             : "v2" (z-score, default) or "v1" (percentile)
    getData_OHLC_v2    : OHLC fetch function; uses internal API default if None.
    front_n            : How many front spreads / jumbos per structure.
    show_plots         : Display distribution plots inline.
    save_outputs       : Write labeled CSVs / sweep CSVs / PNG plots to OUTPUT_DIR.
    cfg                : Override config.ABSORPTION params (labeling, sweep grid).
    volume_window      : (v2) Rolling avg-volume lookback for delta normalisation.
    atr_window         : (v2) ATR lookback for price normalisation.
    baseline_window    : (v2) Median/MAD rolling window.
    delta_z_threshold  : (v2) Signal threshold on z_delta.
    price_z_threshold  : (v2) Signal threshold on z_price.

    Returns
    -------
    results[instrument][horizon] = {
        "scored"       : labeled DataFrame,
        "table"        : threshold-sweep DataFrame,
        "base_rate"    : float,
        "pr_auc"       : float,
        "fig_dist"     : matplotlib Figure,
        "styled_table" : styled pandas Styler,
    }
    """
    if cfg is None:      cfg = ABSORPTION
    if horizons is None: horizons = HORIZONS
    if getData_OHLC_v2 is None: getData_OHLC_v2 = _default_fetch

    product   = PRODUCTS[product_code]
    tick      = product["tick_size"]
    by_struct = get_all_front_instruments(product_code, REFERENCE_DATE, front_n)

    _section(f"ABSORPTION ROLLING ({method.upper()})  ▸  {product_code}  ▸  horizons: {horizons}")
    for struct, instrs in by_struct.items():
        print(f"  {struct.upper():8s}: {instrs}")

    results: dict[str, dict] = {}

    for struct_name, instruments in by_struct.items():

        _section(
            f"STRUCTURE: {struct_name.upper()}  ({product_code}  absorption rolling)",
            char="=",
        )

        for instrument in instruments:
            _subsection(f"{struct_name.upper()}  ▸  {instrument}  [rolling]")

            # ── fetch raw 1D bars (one network/cache hit per instrument) ──────
            df_1d = fetch_ohlc_1d(
                instrument, product_code, getData_OHLC_v2,
            )
            if df_1d is None or len(df_1d) == 0:
                print(f"  [SKIP] No 1D OHLC data for {instrument}")
                continue

            results.setdefault(instrument, {})

            for horizon in horizons:
                combined_candle = HORIZON_BARS.get(horizon, 1)

                print(f"\n  ◆ {instrument}  [{horizon}]  "
                      f"(combined_candle={combined_candle}, 1D bars={len(df_1d)})")

                # ── score ──────────────────────────────────────────────────────
                try:
                    scored = calculate_absorption_rolling(
                        df_1d,
                        combined_candle    = combined_candle,
                        method             = method,
                        # v1 params
                        window             = cfg.get("window", 60),
                        tick_size          = tick,
                        strength_threshold = cfg.get("strength_threshold", 0.60),
                        # v2 params
                        volume_window      = volume_window,
                        atr_window         = atr_window,
                        baseline_window    = baseline_window,
                        delta_z_threshold  = delta_z_threshold,
                        price_z_threshold  = price_z_threshold,
                    )
                except Exception as e:
                    print(f"    [WARN] Score failed: {e}")
                    import traceback; traceback.print_exc()
                    continue

                # ── label ──────────────────────────────────────────────────────
                try:
                    labeled = label_absorption(
                        scored,
                        tick                = tick,
                        required_move_ticks = cfg["required_move_ticks"],
                        sl_lookback         = cfg["sl_lookback"],
                        forward_days        = cfg["forward_days"],
                    )
                except Exception as e:
                    print(f"    [WARN] Label failed: {e}")
                    continue

                # Score column name differs by method
                score_col = "absorption_score" if method == "v2" else "absorption_strength"

                labeled = labeled[
                    labeled["absorption_direction"].isin(["BUY", "SELL"])
                ].dropna(subset=[score_col]).copy()
                labeled["success"] = labeled["sl_success"]

                valid  = labeled.dropna(subset=["success"])
                n_val  = len(valid)
                n_succ = int((valid["success"] == 1).sum())
                if n_val == 0:
                    print("    [SKIP] No valid labeled rows.")
                    continue

                print(f"    candles_scored={len(scored)}  valid={n_val}  "
                      f"success={n_succ}  fail={n_val-n_succ}  "
                      f"base rate={n_succ/n_val:.3f}")

                # ── threshold sweep ────────────────────────────────────────────
                tbl, br = threshold_sweep(
                    labeled, score_col,
                    success_col="success", cfg=dict(cfg),
                )
                auc_val = pr_auc(labeled, score_col, "success")

                # ── distribution plot ──────────────────────────────────────────
                title = (
                    f"{struct_name.upper()} ▸ {instrument} ▸ {horizon} "
                    f"▸ absorption rolling ({method.upper()})"
                )
                fig_d = plot_distribution(
                    labeled, score_col, "success",
                    cfg.get("threshold_line"), title,
                )
                if show_plots:
                    plt.figure(fig_d.number); plt.show()

                print(f"\n    ── Threshold Table  [{instrument}  {horizon}] ──")
                _print_table(tbl, br, auc_val)

                # ── save outputs ───────────────────────────────────────────────
                if save_outputs:
                    tag  = f"absorption_rolling_{method}"
                    out  = _setup_output(product_code, tag, horizon)
                    safe = instrument.replace("-", "_")
                    labeled.to_csv(
                        out / f"{safe}_{horizon}_labeled.csv",
                        index=False, encoding="utf-8-sig",
                    )
                    tbl.to_csv(
                        out / f"{safe}_{horizon}_sweep.csv",
                        index=False, encoding="utf-8-sig",
                    )
                    fig_d.savefig(
                        out / f"{safe}_{struct_name}_{horizon}_dist.png",
                        dpi=120, bbox_inches="tight",
                    )

                results[instrument][horizon] = {
                    "scored"       : labeled,
                    "table"        : tbl,
                    "base_rate"    : br,
                    "pr_auc"       : auc_val,
                    "fig_dist"     : fig_d,
                    "styled_table" : styled_sweep_table(
                        tbl,
                        f"{instrument} {horizon} rolling {method} | base={br:.3f}",
                    ),
                }
                plt.close("all")

    _section(
        f"ABSORPTION ROLLING COMPLETE  ▸  {product_code}  ▸  "
        f"{sum(len(v) for v in results.values())} results"
    )
    return results

