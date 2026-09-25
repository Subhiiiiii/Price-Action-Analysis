"""
absorption_regime_viz.py
========================
Drop this file next to your existing market_analysis package and run it in
a Jupyter cell or as a script.

PURPOSE
-------
Visualise the THREE regimes that live inside "absorption":
  1. Pure Buying     — high buy delta  + high price rise    → no absorption
  2. Absorption → Reversal   — high buy delta  + flat price → price falls after
  3. Absorption → Continuation — high buy delta + flat price → price rises after

All metrics are computed strictly from past data; forward windows are only
used to assign post-hoc regime labels (no leakage into scores).

USAGE
-----
# In your notebook
from absorption_regime_viz import AbsorptionRegimeExplorer
explorer = AbsorptionRegimeExplorer(df)   # df = your scored+labeled DataFrame
explorer.run_all()                        # prints + shows every plot
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── tiny colour palette ────────────────────────────────────────────────────
C_BUY   = "#2196F3"   # blue   – pure buying
C_REV   = "#F44336"   # red    – absorption → reversal
C_CONT  = "#4CAF50"   # green  – absorption → continuation
C_NEUT  = "#9E9E9E"   # grey   – unresolved / neutral
ALPHA   = 0.55


# ════════════════════════════════════════════════════════════════════════════
class AbsorptionRegimeExplorer:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Output of calculate_absorption() followed by label_absorption().
        Required columns:
            delta, delta_pct, price_range, total_volume, close,
            absorption_strength, absorption_direction, sl_success,
            Date (or time / timestamp)

    forward_window : int
        How many future bars to look ahead when computing post-event
        delta persistence and lambda shift.  Default 5.

    vol_window : int
        Rolling window (bars) for realised volatility.  Default 20.

    direction_filter : str | None
        "BUY", "SELL", or None (both).  When None both directions are
        included but the 2D scatter axes flip so BUY absorption is always
        on the "positive" side.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        forward_window: int = 5,
        vol_window: int = 20,
        direction_filter: str | None = None,
        absorption_threshold: float = 0.60,
    ):
        self.fw  = forward_window
        self.vw  = vol_window
        self.thr = absorption_threshold

        d = df.copy()
        # standardise date column
        for col in ("Date", "time", "timestamp"):
            if col in d.columns:
                d[col] = pd.to_datetime(d[col], utc=True, errors="coerce")
                d = d.sort_values(col).reset_index(drop=True)
                d = d.rename(columns={col: "Date"})
                break

        # normalise column names to lower
        d.columns = [c.lower() for c in d.columns]

        if direction_filter:
            d = d[d["absorption_direction"].str.upper() == direction_filter.upper()]
        self.df = d.reset_index(drop=True)
        self._enrich()

    # ── feature engineering (no forward leakage into scores) ────────────────
    def _enrich(self):
        d = self.df
        n = len(d)

        close  = d["close"].to_numpy(dtype=float)
        delta  = d["delta"].to_numpy(dtype=float)
        high   = d["high"].to_numpy(dtype=float)   if "high"  in d else close
        low    = d["low"].to_numpy(dtype=float)    if "low"   in d else close
        volume = d["total_volume"].to_numpy(dtype=float)

        # ── 1. Kyle lambda  (past-only: rolling, no forward look) ──────────
        #     λ_t  =  |ΔP_t|  /  (|delta_t| + ε)
        price_chg = np.abs(np.diff(close, prepend=np.nan))
        abs_delta = np.abs(delta)
        lam = price_chg / (abs_delta + 1e-9)
        d["kyle_lambda"] = lam

        # ── 2. Realised volatility (rolling std of returns, past-only) ──────
        ret   = np.diff(np.log(np.where(close > 0, close, np.nan)), prepend=np.nan)
        rvol  = pd.Series(ret).rolling(self.vw, min_periods=3).std().to_numpy()
        d["rvol"] = rvol

        # ── 3. Absorption Efficiency  |delta| / (|ΔP| + ε) ─────────────────
        #    normalised by rolling baselines so it's comparable across regimes
        roll_vol_delta = pd.Series(abs_delta).rolling(self.vw, min_periods=3).median()
        roll_rvol      = pd.Series(rvol).rolling(self.vw, min_periods=3).median()
        ae = (abs_delta / (roll_vol_delta + 1e-9)) / \
             (price_chg  / (roll_rvol      + 1e-9) + 1e-9)
        d["absorption_efficiency"] = ae

        # ── 4. Post-event features (label-only: excluded from signal) ────────
        #    These are used ONLY to colour/bucket points after the fact.
        #    They must NEVER feed back into the absorption_strength score.

        delta_persist   = np.full(n, np.nan)
        lambda_shift    = np.full(n, np.nan)
        delta_accel     = np.full(n, np.nan)
        fwd_price_chg   = np.full(n, np.nan)   # raw: sign depends on direction

        for i in range(n - self.fw):
            fw_slice_delta = delta[i+1 : i+self.fw+1]
            fw_slice_high  = high[i+1 : i+self.fw+1]
            fw_slice_low   = low[i+1 : i+self.fw+1]
            fw_close_arr   = close[i+1 : i+self.fw+1]

            # delta persistence: fraction of forward bars with same-sign delta
            sign_now = np.sign(delta[i])
            same_sign = np.sum(np.sign(fw_slice_delta) == sign_now)
            delta_persist[i] = same_sign / self.fw

            # lambda shift: avg forward lambda vs current lambda
            fw_pc  = np.abs(np.diff(np.concatenate([[close[i]], fw_close_arr])))
            fw_ad  = np.abs(fw_slice_delta) + 1e-9
            fw_lam = fw_pc / fw_ad
            if np.isfinite(lam[i]) and lam[i] > 1e-12:
                lambda_shift[i] = np.nanmean(fw_lam) / (lam[i] + 1e-9)

            # delta acceleration: mean(fw delta) - current delta (normalised)
            delta_accel[i] = (np.nanmean(fw_slice_delta) - delta[i]) / \
                             (np.std(delta[:i+1]) + 1e-9)

            # forward price change
            fwd_price_chg[i] = fw_close_arr[-1] - close[i]

        d["delta_persist"]   = delta_persist
        d["lambda_shift"]    = lambda_shift
        d["delta_accel"]     = delta_accel
        d["fwd_price_chg"]   = fwd_price_chg

        # ── 5. Regime label (for colouring only) ─────────────────────────────
        #    Uses sl_success from label_absorption if available,
        #    else falls back to fwd_price_chg sign.
        regime = np.full(n, "UNRESOLVED", dtype=object)

        has_label = "sl_success" in d.columns
        abs_str   = d["absorption_strength"].to_numpy(dtype=float)
        direction = d["absorption_direction"].str.upper().to_numpy()

        for i in range(n - self.fw):
            is_abs = np.isfinite(abs_str[i]) and abs_str[i] >= self.thr
            dir_i  = direction[i]

            if not is_abs:
                # below absorption threshold → pure directional flow
                if dir_i == "BUY"  and fwd_price_chg[i] > 0:
                    regime[i] = "PURE_BUYING"
                elif dir_i == "SELL" and fwd_price_chg[i] < 0:
                    regime[i] = "PURE_SELLING"
                continue

            # absorption detected — determine resolution
            if has_label and np.isfinite(d["sl_success"].iloc[i]):
                success = d["sl_success"].iloc[i]
                if dir_i == "SELL":
                    # SELL absorption → expect UP
                    regime[i] = "ABS_CONT"  if success == 1 else "ABS_REV"
                else:
                    # BUY absorption → expect DOWN
                    regime[i] = "ABS_CONT"  if success == 1 else "ABS_REV"
            else:
                # fallback: forward price direction
                if dir_i == "BUY":
                    regime[i] = "ABS_REV"   if fwd_price_chg[i] < 0 else "ABS_CONT"
                else:
                    regime[i] = "ABS_CONT"  if fwd_price_chg[i] > 0 else "ABS_REV"

        d["regime"] = regime
        self.df = d

    # ── colour mapping ───────────────────────────────────────────────────────
    @staticmethod
    def _regime_color(regime_series: pd.Series) -> list[str]:
        cmap = {
            "PURE_BUYING":  C_BUY,
            "PURE_SELLING": "#FF9800",
            "ABS_REV":      C_REV,
            "ABS_CONT":     C_CONT,
            "UNRESOLVED":   C_NEUT,
        }
        return [cmap.get(r, C_NEUT) for r in regime_series]

    # ════════════════════════════════════════════════════════════════════════
    # PLOT 1 — 2D Pressure vs Response map  (the core diagnostic)
    # ════════════════════════════════════════════════════════════════════════
    def plot_pressure_response_map(self, ax=None, max_pts: int = 3000):
        """
        X = normalised delta_pct  (signed)
        Y = normalised absorption_efficiency
        Colour = regime

        Quadrants:
          top-right   → huge buy flow, efficient absorption  → BUY absorption
          bottom-right → huge buy flow, price rises normally → pure buying
          top-left    → huge sell flow, efficient absorption → SELL absorption
          bottom-left  → pure selling
        """
        d = self.df.dropna(subset=["delta_pct", "absorption_efficiency", "regime"])
        if len(d) > max_pts:
            d = d.sample(max_pts, random_state=42)

        if ax is None:
            _, ax = plt.subplots(figsize=(9, 7))

        colors = self._regime_color(d["regime"])

        # clip outliers for display
        x = d["delta_pct"].clip(-3*d["delta_pct"].std(), 3*d["delta_pct"].std())
        y = np.log1p(d["absorption_efficiency"].clip(0))

        ax.scatter(x, y, c=colors, alpha=ALPHA, s=18, linewidths=0)

        ax.axvline(0, color="white", lw=0.8, alpha=0.4)
        ax.axhline(np.log1p(1), color="white", lw=0.8, ls="--", alpha=0.4)

        ax.set_xlabel("Delta %  (signed,  + = net buying)", fontsize=10)
        ax.set_ylabel("log(1 + Absorption Efficiency)", fontsize=10)
        ax.set_title("Pressure vs Response Map", fontsize=12, fontweight="bold")

        # legend
        from matplotlib.patches import Patch
        legend_map = {
            "Pure Buying":    C_BUY,
            "Pure Selling":   "#FF9800",
            "Abs → Reversal": C_REV,
            "Abs → Cont.":    C_CONT,
            "Unresolved":     C_NEUT,
        }
        handles = [Patch(fc=v, label=k, alpha=0.8) for k, v in legend_map.items()]
        ax.legend(handles=handles, fontsize=8, loc="upper left")
        ax.set_facecolor("#111111")
        return ax

    # ════════════════════════════════════════════════════════════════════════
    # PLOT 2 — Delta Persistence distribution by regime
    # ════════════════════════════════════════════════════════════════════════
    def plot_delta_persistence_dist(self, ax=None):
        """
        KDE of delta_persist split by ABS_REV vs ABS_CONT.
        High persistence → continuation hypothesis.
        Low  persistence → reversal hypothesis.
        """
        d = self.df.dropna(subset=["delta_persist", "regime"])

        rev_dp  = d.loc[d["regime"] == "ABS_REV",  "delta_persist"].to_numpy()
        cont_dp = d.loc[d["regime"] == "ABS_CONT", "delta_persist"].to_numpy()

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        for arr, label, color in [
            (rev_dp,  "Abs → Reversal",     C_REV),
            (cont_dp, "Abs → Continuation", C_CONT),
        ]:
            if len(arr) < 4:
                continue
            xs = np.linspace(0, 1, 200)
            try:
                kde  = gaussian_kde(arr, bw_method="scott")
                ys   = kde(xs)
                ax.fill_between(xs, ys, alpha=0.35, color=color)
                ax.plot(xs, ys, color=color, lw=2, label=f"{label}  (n={len(arr)})")
            except Exception:
                ax.hist(arr, bins=20, color=color, alpha=0.5,
                        density=True, label=f"{label}  (n={len(arr)})")

        ax.set_xlabel("Delta Persistence  (fraction of forward bars same-sign)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title("Delta Persistence: Reversal vs Continuation", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_facecolor("#111111")
        return ax

    # ════════════════════════════════════════════════════════════════════════
    # PLOT 3 — Kyle Lambda Shift distribution
    # ════════════════════════════════════════════════════════════════════════
    def plot_lambda_shift_dist(self, ax=None):
        """
        KDE of lambda_shift by regime.
        lambda_shift > 1  → market becoming MORE sensitive to flow after abs
                          → price is about to move (continuation or reversal)
        lambda_shift < 1  → flow still being absorbed
        """
        d = self.df.dropna(subset=["lambda_shift", "regime"])
        d = d[np.isfinite(d["lambda_shift"]) & (d["lambda_shift"] > 0)]
        log_ls = np.log(d["lambda_shift"])

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        for regime, label, color in [
            ("ABS_REV",  "Abs → Reversal",     C_REV),
            ("ABS_CONT", "Abs → Continuation", C_CONT),
        ]:
            arr = log_ls[d["regime"] == regime].to_numpy()
            if len(arr) < 4:
                continue
            try:
                xs  = np.linspace(arr.min(), arr.max(), 200)
                kde = gaussian_kde(arr, bw_method="scott")
                ys  = kde(xs)
                ax.fill_between(xs, ys, alpha=0.35, color=color)
                ax.plot(xs, ys, color=color, lw=2, label=f"{label}  (n={len(arr)})")
            except Exception:
                ax.hist(arr, bins=20, color=color, alpha=0.5,
                        density=True, label=f"{label}  (n={len(arr)})")

        ax.axvline(0, color="white", lw=1.2, ls="--", alpha=0.6,
                   label="λ_shift = 1  (no change)")
        ax.set_xlabel("log(Lambda Shift)  — log(λ_post / λ_absorption)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title("Kyle λ Shift after Absorption", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_facecolor("#111111")
        return ax

    # ════════════════════════════════════════════════════════════════════════
    # PLOT 4 — Delta Acceleration scatter
    # ════════════════════════════════════════════════════════════════════════
    def plot_delta_acceleration(self, ax=None):
        """
        delta_accel vs absorption_strength, coloured by regime.
        Positive acceleration after absorption → continuation.
        Negative acceleration                 → reversal.
        """
        d = self.df.dropna(subset=["delta_accel", "absorption_strength", "regime"])
        d = d[d["absorption_strength"] >= self.thr]

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        colors = self._regime_color(d["regime"])
        ax.scatter(
            d["absorption_strength"],
            d["delta_accel"].clip(-5, 5),
            c=colors, alpha=ALPHA, s=18, linewidths=0,
        )
        ax.axhline(0, color="white", lw=1, ls="--", alpha=0.5)
        ax.set_xlabel("Absorption Strength", fontsize=10)
        ax.set_ylabel("Delta Acceleration  (normalised)", fontsize=10)
        ax.set_title("Delta Acceleration vs Absorption Strength", fontsize=12, fontweight="bold")

        from matplotlib.patches import Patch
        handles = [
            Patch(fc=C_REV,  label="Abs → Reversal",     alpha=0.8),
            Patch(fc=C_CONT, label="Abs → Continuation", alpha=0.8),
        ]
        ax.legend(handles=handles, fontsize=9)
        ax.set_facecolor("#111111")
        return ax

    # ════════════════════════════════════════════════════════════════════════
    # PLOT 5 — Absorption Efficiency vs Realised Vol  (regime-coloured)
    # ════════════════════════════════════════════════════════════════════════
    def plot_ae_vs_rvol(self, ax=None, max_pts: int = 2000):
        """
        High vol + high AE  → loud market being absorbed.
        Low  vol + high AE  → quiet market with unusual order flow.
        """
        d = self.df.dropna(subset=["absorption_efficiency", "rvol", "regime"])
        d = d[np.isfinite(d["absorption_efficiency"]) & (d["rvol"] > 0)]
        if len(d) > max_pts:
            d = d.sample(max_pts, random_state=42)

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        colors = self._regime_color(d["regime"])
        ax.scatter(
            d["rvol"],
            np.log1p(d["absorption_efficiency"].clip(0)),
            c=colors, alpha=ALPHA, s=16, linewidths=0,
        )
        ax.set_xlabel("Realised Volatility  (rolling return std)", fontsize=10)
        ax.set_ylabel("log(1 + Absorption Efficiency)", fontsize=10)
        ax.set_title("Absorption Efficiency vs Market Volatility", fontsize=12, fontweight="bold")

        from matplotlib.patches import Patch
        legend_map = {
            "Pure Buying":    C_BUY,
            "Pure Selling":   "#FF9800",
            "Abs → Reversal": C_REV,
            "Abs → Cont.":    C_CONT,
            "Unresolved":     C_NEUT,
        }
        handles = [Patch(fc=v, label=k, alpha=0.8) for k, v in legend_map.items()]
        ax.legend(handles=handles, fontsize=8)
        ax.set_facecolor("#111111")
        return ax

    # ════════════════════════════════════════════════════════════════════════
    # PLOT 6 — Regime counts + base rates per absorption_class
    # ════════════════════════════════════════════════════════════════════════
    def plot_regime_by_class(self, ax=None):
        """
        Stacked bar: for each absorption_class bucket,
        show the fraction of ABS_REV vs ABS_CONT.
        Tells you whether stronger absorption is more likely to reverse.
        """
        d = self.df[self.df["regime"].isin(["ABS_REV", "ABS_CONT"])]
        if "absorption_class" not in d.columns or len(d) == 0:
            print("[SKIP] absorption_class column missing or no data.")
            return ax

        tbl = (
            d.groupby(["absorption_class", "regime"])
             .size()
             .unstack(fill_value=0)
        )
        order = ["POSSIBLE", "MODERATE", "STRONG", "EXTREME"]
        tbl   = tbl.reindex([o for o in order if o in tbl.index])
        total = tbl.sum(axis=1)
        frac  = tbl.div(total, axis=0)

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))

        x = np.arange(len(frac))
        w = 0.55

        bottom = np.zeros(len(frac))
        for col, color, label in [
            ("ABS_REV",  C_REV,  "Abs → Reversal"),
            ("ABS_CONT", C_CONT, "Abs → Continuation"),
        ]:
            if col in frac.columns:
                vals = frac[col].to_numpy()
                ax.bar(x, vals, w, bottom=bottom, color=color,
                       alpha=0.85, label=label)
                bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(frac.index, fontsize=10)
        ax.set_xlabel("Absorption Class", fontsize=10)
        ax.set_ylabel("Fraction of cases", fontsize=10)
        ax.set_title("Reversal vs Continuation by Absorption Class", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

        # annotate with sample counts
        for xi, (idx, row) in zip(x, frac.iterrows()):
            n = int(total.loc[idx])
            ax.text(xi, 1.02, f"n={n}", ha="center", va="bottom",
                    fontsize=8, color="white")

        ax.set_facecolor("#111111")
        ax.set_ylim(0, 1.12)
        return ax

    # ════════════════════════════════════════════════════════════════════════
    # PLOT 7 — Simple numeric summary table
    # ════════════════════════════════════════════════════════════════════════
    def print_summary(self):
        d = self.df
        regimes = ["PURE_BUYING", "PURE_SELLING", "ABS_REV", "ABS_CONT", "UNRESOLVED"]
        rows = []
        for r in regimes:
            sub = d[d["regime"] == r]
            if len(sub) == 0:
                continue
            rows.append({
                "Regime": r,
                "Count":  len(sub),
                "Pct":    f"{len(sub)/len(d)*100:.1f}%",
                "Mean AE":   f"{sub['absorption_efficiency'].mean():.2f}",
                "Mean DP":   f"{sub['delta_persist'].mean():.2f}",
                "Mean λShift": f"{sub['lambda_shift'].mean():.2f}" if "lambda_shift" in sub else "–",
                "Mean Δaccel": f"{sub['delta_accel'].mean():.2f}",
            })
        print("\n── Regime Summary ──────────────────────────────────────────────")
        print(pd.DataFrame(rows).to_string(index=False))
        print()

    # ════════════════════════════════════════════════════════════════════════
    # run_all : convenience method
    # ════════════════════════════════════════════════════════════════════════
    def run_all(self, figsize_big=(18, 14)):
        plt.style.use("dark_background")
        self.print_summary()

        fig = plt.figure(figsize=figsize_big, constrained_layout=True)
        fig.suptitle("Absorption Regime Explorer", fontsize=15, fontweight="bold")

        gs = gridspec.GridSpec(3, 3, figure=fig)

        self.plot_pressure_response_map(ax=fig.add_subplot(gs[0:2, 0:2]))
        self.plot_delta_persistence_dist(ax=fig.add_subplot(gs[0, 2]))
        self.plot_lambda_shift_dist(ax=fig.add_subplot(gs[1, 2]))
        self.plot_delta_acceleration(ax=fig.add_subplot(gs[2, 0]))
        self.plot_ae_vs_rvol(ax=fig.add_subplot(gs[2, 1]))
        self.plot_regime_by_class(ax=fig.add_subplot(gs[2, 2]))

        plt.show()
        return fig


# ════════════════════════════════════════════════════════════════════════════
# NOTEBOOK USAGE EXAMPLE
# ════════════════════════════════════════════════════════════════════════════
"""
# ── In your Jupyter notebook ──────────────────────────────────────────────

from market_analysis.signals.absorption.calculate import calculate_absorption
from market_analysis.signals.absorption.label import label_absorption
from absorption_regime_viz import AbsorptionRegimeExplorer

# Step 1: get your raw OHLC (example — replace with your fetch call)
ohlc = getData_OHLC_v2("ES-H25", horizon="1d")

# Step 2: score
scored = calculate_absorption(ohlc, window=60, tick_size=0.25)

# Step 3: label  (forward labels — used only for colouring, not for scoring)
labeled = label_absorption(
    scored,
    tick=0.25,
    required_move_ticks=4,
    sl_lookback=10,
    forward_days=5,
)

# Step 4: explore
explorer = AbsorptionRegimeExplorer(
    labeled,
    forward_window=5,      # how many bars to look ahead for post-event features
    vol_window=20,         # rolling window for realised vol
    direction_filter=None, # "BUY", "SELL", or None for both
    absorption_threshold=0.60,
)
explorer.run_all()

# ── Individual plots ──────────────────────────────────────────────────────
# explorer.plot_pressure_response_map()
# explorer.plot_delta_persistence_dist()
# explorer.plot_lambda_shift_dist()
# explorer.plot_delta_acceleration()
# explorer.plot_ae_vs_rvol()
# explorer.plot_regime_by_class()
"""
