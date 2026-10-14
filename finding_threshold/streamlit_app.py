# market_analysis/finding_threshold/streamlit_app.py
"""
Interactive Streamlit Application for Compression Signal Analysis & Liquidity Qualification.

Features:
- Phase 1 & 2: Front-2 Analytical Universe vs Front-3..5 Pipeline Monitoring.
- Liquidity Qualification Matrix with commodity-calibrated thresholds.
- Priority 1: 10-day forward outcome labeling on actual continuous trading days.
- Priority 2: Full sample evaluation including zero-score days for true base rate & lift.
- Priority 3: Episode collapsing ('first_bar', 'non_overlapping', 'all_bars') to eliminate forward window overlap.
- Priority 4: Block-bootstrap CI and Wilson 95% confidence intervals.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_DOCS = _ROOT.parent
for p in [str(_ROOT), str(_DOCS), str(_HERE)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import os
import importlib.util
link_path = _ROOT / "market_analysis"
if not link_path.exists():
    try:
        os.symlink(str(_ROOT), str(link_path), target_is_directory=True)
    except Exception:
        pass

if "market_analysis" not in sys.modules or not getattr(sys.modules["market_analysis"], "__spec__", None):
    init_file = _ROOT / "__init__.py"
    if not init_file.exists():
        init_file.write_text("# market_analysis\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location("market_analysis", str(init_file), submodule_search_locations=[str(_ROOT)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["market_analysis"] = pkg
    spec.loader.exec_module(pkg)

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    from market_analysis.config import (
        PRODUCTS, FRONT_N, COMPRESSION as COMP_CFG,
    )
    from market_analysis.utils.contracts import get_all_front_instruments
    from market_analysis.data.ohlc import fetch_ohlc_1d, getData_OHLC_v2
    from market_analysis.data.liquidity import (
        qualify_instrument_liquidity,
        evaluate_softs_liquidity_universe,
        LIQUIDITY_THRESHOLDS,
    )
    from market_analysis.signals.compression.calculate import calculate_compression
    from market_analysis.signals.compression.label import label_compression
    from market_analysis.finding_threshold.compression_threshold import (
        _months_back_start,
        _make_threshold_table,
        _distribution_plot,
    )
    from market_analysis.utils.metrics import wilson_score_interval, _block_bootstrap_ci
except ImportError:
    from config import (
        PRODUCTS, FRONT_N, COMPRESSION as COMP_CFG,
    )
    from utils.contracts import get_all_front_instruments
    from data.ohlc import fetch_ohlc_1d, getData_OHLC_v2
    from data.liquidity import (
        qualify_instrument_liquidity,
        evaluate_softs_liquidity_universe,
        LIQUIDITY_THRESHOLDS,
    )
    from signals.compression.calculate import calculate_compression
    from signals.compression.label import label_compression
    from finding_threshold.compression_threshold import (
        _months_back_start,
        _make_threshold_table,
        _distribution_plot,
    )
    from utils.metrics import wilson_score_interval, _block_bootstrap_ci

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Commodity Price Action — Compression Thresholds",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Compression Signal & Liquidity Threshold Explorer")
st.markdown(
    """
    **Comprehensive Research & Diagnostics Dashboard for Commodity Calendar Spreads**  
    *Phases 1 & 2 Liquidity Qualification • Actual 10-Day Horizon Labeling • Compression Episodes (No Overlap) • Wilson & Bootstrap CIs*
    """
)

# ── Sidebar Controls ──────────────────────────────────────────────────────────
st.sidebar.header("🎯 Universe & Parameter Controls")

product_options = ["SB", "KC", "CC", "CT", "RC", "SM", "BO", "W"]
selected_product = st.sidebar.selectbox("Commodity Product", product_options, index=0)

months_back = st.sidebar.slider("Historical Lookback (Months)", min_value=3, max_value=24, value=9, step=1)

st.sidebar.subheader("💧 Phase 1 & 2: Liquidity Qualification")
analytical_n = st.sidebar.slider("Analytical Universe Depth (Front-N)", min_value=1, max_value=5, value=2,
                                  help="Phase 1: Default Front-2 for Softs; remaining contracts monitored in pipeline.")
pipeline_n = st.sidebar.slider("Total Pipeline Tracking Depth", min_value=analytical_n, max_value=5, value=5)
require_eligible = st.sidebar.checkbox("Enforce Liquidity Gate (Filter ILLIQUID)", value=True)
include_monitor = st.sidebar.checkbox("Include MONITOR Spreads in Sweep", value=False)

st.sidebar.subheader("🔬 Signal & Horizon Parameters")
k_val = st.sidebar.number_input("Move Requirement k (x tick size)", min_value=1, max_value=20, value=COMP_CFG.get("k", 4))
fwd_days = st.sidebar.number_input("Forward Evaluation Window (Trading Days)", min_value=1, max_value=30, value=COMP_CFG.get("forward_days", 10))

episode_mode = st.sidebar.selectbox(
    "Episode Mode (Resolve Window Overlap)",
    ["first_bar", "non_overlapping", "all_bars"],
    index=0,
    help="Priority 3: 'first_bar' triggers on entry bar of compression episode. 'non_overlapping' enforces independent windows."
)

st.sidebar.subheader("📊 Sweep Threshold Grid")
grid_start = st.sidebar.number_input("Grid Start", min_value=0.05, max_value=0.50, value=0.10, step=0.05)
grid_stop = st.sidebar.number_input("Grid Stop", min_value=0.50, max_value=1.00, value=0.95, step=0.05)
grid_step = st.sidebar.number_input("Grid Step", min_value=0.01, max_value=0.10, value=0.05, step=0.01)
score_grid = np.round(np.arange(grid_start, grid_stop + grid_step / 2, grid_step), 3)

# ── Cached Data Fetching & Processing ─────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner=False)
def load_product_data(product: str, months: int, p_depth: int, a_depth: int):
    start_d = _months_back_start(months)
    import datetime
    end_d = datetime.date.today().strftime("%Y-%m-%d")
    instruments_dict = get_all_front_instruments(product, front_n=p_depth)

    cached_ohlc = {}
    liquidity_list = []

    for structure, inst_list in instruments_dict.items():
        for rank, inst in enumerate(inst_list, 1):
            df_raw = fetch_ohlc_1d(
                instrument=inst,
                product_code=product,
                getData_OHLC_v2=getData_OHLC_v2,
                start_date=start_d,
                end_date=end_d,
            )
            cached_ohlc[inst] = df_raw
            qual = qualify_instrument_liquidity(
                df=df_raw,
                product_code=product,
                instrument=inst,
                rank=rank,
                analytical_max_rank=a_depth,
            )
            qual["structure"] = structure
            liquidity_list.append(qual)

    return cached_ohlc, pd.DataFrame(liquidity_list), start_d, end_d


with st.spinner(f"Loading & Qualifying Market Data for {selected_product}..."):
    cached_ohlc, liq_df, start_date, end_date = load_product_data(
        selected_product, months_back, pipeline_n, analytical_n
    )

tick_size = PRODUCTS[selected_product]["tick_size"]

# ── Top Metric Banner ─────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
eligible_count = int((liq_df["status"] == "ELIGIBLE").sum()) if not liq_df.empty else 0
monitor_count = int((liq_df["status"] == "MONITOR").sum()) if not liq_df.empty else 0
illiquid_count = int((liq_df["status"] == "ILLIQUID").sum()) if not liq_df.empty else 0
total_contracts = len(liq_df)

col1.metric("Selected Product", selected_product, f"Tick: {tick_size}")
col2.metric("Total Contracts", total_contracts, f"Front-1..{pipeline_n}")
col3.metric("Eligible", eligible_count, "Ready for Sweep", delta_color="normal")
col4.metric("Monitor", monitor_count, "Pipeline Tracked", delta_color="off")
col5.metric("Illiquid", illiquid_count, "Filtered Out", delta_color="inverse")

# ── Main Tabs ─────────────────────────────────────────────────────────────────
tab_liq, tab_sweep, tab_dist, tab_raw = st.tabs([
    "💧 Liquidity Qualification (Phase 1 & 2)",
    "🎯 Threshold Sweep & Performance",
    "📊 Distributions (Breakout vs Failure)",
    "🔍 Feature Components & Signals",
])

# ── TAB 1: LIQUIDITY QUALIFICATION ────────────────────────────────────────────
with tab_liq:
    st.subheader(f"Phase 1 & 2 Liquidity Qualification Matrix — {selected_product}")
    st.markdown(
        """
        * **Phase 1 (Spread Coverage)**: Front-2 is designated as the analytical universe; Front-3 to Front-5 are preserved for pipeline monitoring.  
        * **Phase 2 (Liquidity Screening)**: Evaluates 20-day median volume, relative volume ratio, zero-volume days, and spread proxies.
        """
    )

    if not liq_df.empty:
        # Style status column
        def color_status(val):
            if val == "ELIGIBLE":
                return "background-color: #d4edda; color: #155724; font-weight: bold;"
            elif val == "MONITOR":
                return "background-color: #fff3cd; color: #856404; font-weight: bold;"
            else:
                return "background-color: #f8d7da; color: #721c24; font-weight: bold;"

        styler = (
            liq_df[[
                "product", "structure", "rank", "instrument", "universe",
                "bars", "med_vol_20d", "rvol_20d", "zero_vol_days_20d", "status", "reason"
            ]].style
            .format({"med_vol_20d": "{:,.0f}", "rvol_20d": "{:.2f}", "bars": "{:,d}"})
        )
        if hasattr(styler, "map"):
            styler = styler.map(color_status, subset=["status"])
        else:
            styler = styler.applymap(color_status, subset=["status"])

        st.dataframe(
            styler,
            use_container_width=True,
            height=380,
        )

        st.subheader("Liquidity Profile by Tenor")
        c_left, c_right = st.columns(2)
        with c_left:
            spreads_only = liq_df[liq_df["structure"] == "spread"]
            fig_vol, ax_vol = plt.subplots(figsize=(8, 3.5))
            colors = ["seagreen" if s == "ELIGIBLE" else "goldenrod" if s == "MONITOR" else "tomato" for s in spreads_only["status"]]
            ax_vol.bar(spreads_only["instrument"], spreads_only["med_vol_20d"], color=colors, edgecolor="black", alpha=0.85)
            ax_vol.set_ylabel("20-Day Median Daily Volume")
            ax_vol.set_title(f"{selected_product} Calendar Spreads — 20-Day Median Volume", fontweight="bold", fontsize=10)
            ax_vol.grid(axis="y", alpha=0.3)
            plt.xticks(rotation=20)
            st.pyplot(fig_vol)
            plt.close(fig_vol)

        with c_right:
            fig_rvol, ax_rvol = plt.subplots(figsize=(8, 3.5))
            ax_rvol.bar(spreads_only["instrument"], spreads_only["rvol_20d"], color="steelblue", edgecolor="black", alpha=0.85)
            ax_rvol.axhline(1.0, color="red", linestyle="--", label="Baseline (1.0x)")
            ax_rvol.set_ylabel("Relative Volume (5d / 20d)")
            ax_rvol.set_title(f"{selected_product} Calendar Spreads — Relative Volume", fontweight="bold", fontsize=10)
            ax_rvol.grid(axis="y", alpha=0.3)
            ax_rvol.legend(fontsize=8)
            plt.xticks(rotation=20)
            st.pyplot(fig_rvol)
            plt.close(fig_rvol)


# ── TAB 2: THRESHOLD SWEEP & PERFORMANCE ──────────────────────────────────────
with tab_sweep:
    st.subheader("Compression Threshold Sweeps on Qualified Contracts")

    # Filter eligible instruments
    qualified_rows = liq_df[
        (liq_df["rank"].isin([f"Front-{i}" for i in range(1, analytical_n + 1)])) &
        (
            (liq_df["status"] == "ELIGIBLE") if require_eligible and not include_monitor else
            (liq_df["status"].isin(["ELIGIBLE", "MONITOR"])) if require_eligible and include_monitor else
            (liq_df["status"].notna())
        )
    ]

    if qualified_rows.empty:
        st.warning("No instruments passed the selected liquidity and analytical universe filters. Consider enabling MONITOR spreads or expanding analytical depth.")
    else:
        inst_list = qualified_rows["instrument"].tolist()
        selected_inst = st.selectbox("Select Qualified Instrument for Deep Dive", inst_list, index=0)

        raw_df = cached_ohlc.get(selected_inst)
        if raw_df is None or len(raw_df) < 20:
            st.error(f"Insufficient bars for {selected_inst} ({0 if raw_df is None else len(raw_df)} bars).")
        else:
            # Calculate Compression
            df_comp, _ = calculate_compression(
                df=raw_df,
                range_baseline=COMP_CFG["range_baseline"],
                atr_period=COMP_CFG["atr_period"],
                rv_period=COMP_CFG["rv_period"],
                persistence_window=COMP_CFG["persistence_window"],
                compression_threshold=COMP_CFG["compression_threshold"],
            )

            # Priority 1: Label using next 10 actual trading days before filtering
            df_labeled = label_compression(
                df=df_comp,
                tick=tick_size,
                k=k_val,
                forward_days=fwd_days,
                keep_all_bars=True, # Priority 2: keep all trading bars
            )

            # Threshold sweep with Priority 3 (episodes) and Priority 4 (Wilson CI)
            sweep_table, base_rate = _make_threshold_table(
                labeled_df=df_labeled,
                score_col="compression_score",
                success_col="success",
                score_grid=score_grid,
                min_n=8,
                forward_days=fwd_days,
                episode_mode=episode_mode,
            )

            # Performance summary cards
            best_f1_row = sweep_table.loc[sweep_table["F1"].idxmax()] if not sweep_table.empty and sweep_table["F1"].notna().any() else None

            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Market Base Rate", f"{base_rate:.1%}" if np.isfinite(base_rate) else "N/A", "Breakout Frequency")
            if best_f1_row is not None:
                m_col2.metric("Optimal Threshold (Max F1)", f"{best_f1_row['Threshold']:.2f}", f"F1: {best_f1_row['F1']:.3f}")
                m_col3.metric("Precision @ Optimal", f"{best_f1_row['Precision']:.1%}", f"Wilson: {best_f1_row['Wilson CI 95%']}")
                m_col4.metric("Lift @ Optimal", f"{best_f1_row['Lift']:.2f}x", f"Signals: N={int(best_f1_row['N'])}")

            st.write(f"**Sweep Table: {selected_inst}** (Move Required = {k_val}x tick = {k_val*tick_size:.3f} pts, Horizon = {fwd_days}d, Mode = {episode_mode})")

            # Style sweep table
            def highlight_f1(row):
                return ["background-color: #fff3cd" if row["Low N"] else "" for _ in row.index]

            st.dataframe(
                sweep_table.style
                .format({
                    "Threshold": "{:.3f}",
                    "Precision": "{:.3f}",
                    "Recall": "{:.3f}",
                    "F1": "{:.3f}",
                    "MCC": "{:.3f}",
                    "Lift": "{:.2f}",
                }, na_rep="--")
                .background_gradient(subset=["Precision"], cmap="Blues", vmin=0.3, vmax=0.9)
                .background_gradient(subset=["F1"], cmap="RdYlGn", vmin=0.3, vmax=0.7)
                .apply(highlight_f1, axis=1),
                use_container_width=True,
                height=450,
            )

            # Interactive Plot: Precision, Recall, F1 curves
            st.subheader("Performance Metric Curves across Score Thresholds")
            fig_perf, ax_perf = plt.subplots(figsize=(10, 4))
            ax_perf.plot(sweep_table["Threshold"], sweep_table["Precision"], label="Precision", color="dodgerblue", lw=2, marker="o", ms=4)
            ax_perf.plot(sweep_table["Threshold"], sweep_table["Recall"], label="Recall", color="orange", lw=2, marker="s", ms=4)
            ax_perf.plot(sweep_table["Threshold"], sweep_table["F1"], label="F1 Score", color="forestgreen", lw=2.5, marker="^", ms=5)
            ax_perf.axhline(base_rate, color="grey", linestyle="--", label=f"Base Rate ({base_rate:.2f})")
            ax_perf.set_xlabel("Compression Score Threshold")
            ax_perf.set_ylabel("Metric Value")
            ax_perf.set_title(f"{selected_inst} — Precision, Recall & F1 vs Threshold", fontweight="bold")
            ax_perf.set_xlim([grid_start, grid_stop])
            ax_perf.set_ylim([0, 1.05])
            ax_perf.grid(alpha=0.3)
            ax_perf.legend(loc="best")
            st.pyplot(fig_perf)
            plt.close(fig_perf)


# ── TAB 3: DISTRIBUTIONS ──────────────────────────────────────────────────────
with tab_dist:
    st.subheader(f"Signal & Breakout Outcome Distribution — {selected_inst if 'selected_inst' in locals() else ''}")

    if 'df_labeled' in locals() and not df_labeled.empty:
        fig_dist = _distribution_plot(
            labeled_df=df_labeled,
            score_col="compression_score",
            success_col="success",
            instrument=selected_inst,
            tick_size=tick_size,
            k=k_val,
            forward_days=fwd_days,
        )
        st.pyplot(fig_dist)
        plt.close(fig_dist)

        # Bucket crosstab
        st.subheader("Score-Bucket Breakdown (0.05 Intervals)")
        valid_b = df_labeled.dropna(subset=["success"]).copy()
        valid_b["score_bucket"] = pd.cut(
            valid_b["compression_score"],
            bins=np.round(np.arange(0, 1.05, 0.05), 2),
            include_lowest=True,
            right=False,
        )
        ct = pd.crosstab(valid_b["score_bucket"], valid_b["success"], dropna=False).rename(columns={0.0: "No Breakout (0)", 1.0: "Breakout (1)"})
        ct["Total"] = ct.sum(axis=1)
        ct["Success Rate"] = ct["Breakout (1)"] / ct["Total"].replace(0, np.nan)
        st.dataframe(
            ct.style.format({"Success Rate": "{:.1%}"}, na_rep="--")
            .background_gradient(subset=["Success Rate"], cmap="Greens", vmin=0.3, vmax=0.9),
            use_container_width=True,
        )


# ── TAB 4: FEATURE COMPONENTS & SIGNALS ───────────────────────────────────────
with tab_raw:
    st.subheader(f"Feature Component Inspection & Price Chart — {selected_inst if 'selected_inst' in locals() else ''}")

    if 'df_comp' in locals() and not df_comp.empty:
        # Price with signals
        fig_ts, (ax_p, ax_sc) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [2.5, 1.5]})
        ax_p.plot(pd.to_datetime(df_comp["Date"]), df_comp["close"], label="Close Price", color="black", lw=1.2)

        # highlight signals above 0.4
        sig_bars = df_comp[df_comp["compression_score"] >= 0.40]
        if not sig_bars.empty:
            ax_p.scatter(pd.to_datetime(sig_bars["Date"]), sig_bars["close"], color="crimson", s=35, label="Compression Score >= 0.40", zorder=5)

        ax_p.set_title(f"{selected_inst} Price History & Compression Signals", fontweight="bold")
        ax_p.set_ylabel("Price")
        ax_p.grid(alpha=0.3)
        ax_p.legend(loc="upper left")

        # Compression components
        ax_sc.plot(pd.to_datetime(df_comp["Date"]), df_comp["compression_score"], label="Composite Score", color="crimson", lw=1.8)
        if "volume_range_efficiency" in df_comp.columns:
            ax_sc.plot(pd.to_datetime(df_comp["Date"]), df_comp["volume_range_efficiency"], label="Vol-Range Efficiency", color="purple", lw=1.0, ls="--", alpha=0.7)
        if "range_component" in df_comp.columns:
            ax_sc.plot(pd.to_datetime(df_comp["Date"]), df_comp["range_component"], label="Range Component", color="darkgreen", lw=1.0, ls=":", alpha=0.7)

        ax_sc.axhline(0.40, color="red", linestyle="--", alpha=0.8, label="Ref Line (0.40)")
        ax_sc.set_ylabel("Score / Component")
        ax_sc.set_ylim([-0.05, 1.05])
        ax_sc.grid(alpha=0.3)
        ax_sc.legend(loc="upper left", fontsize=8)

        plt.tight_layout()
        st.pyplot(fig_ts)
        plt.close(fig_ts)

        st.subheader("Raw Data Table")
        st.dataframe(df_labeled.tail(50), use_container_width=True)
