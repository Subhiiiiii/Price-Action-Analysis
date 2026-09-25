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