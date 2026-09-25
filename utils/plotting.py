# market_analysis/utils/plotting.py
"""
Shared plotting functions for absorption and compression.
All functions return matplotlib Figure — caller calls plt.show() or savefig().
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import precision_recall_curve, auc as sk_auc

_S_COL = "seagreen"
_F_COL = "tomato"
_C_COL = "steelblue"


def plot_distribution(
    labeled_df: pd.DataFrame,
    score_col: str,
    success_col: str = "success",
    threshold_line: float | None = None,
    title: str = "",
) -> plt.Figure:
    """
    3-panel distribution: Success hist | Fail hist | Density overlay.
    Works for both absorption and compression.
    """
    valid   = labeled_df.dropna(subset=[success_col])
    n_valid = len(valid)
    n_succ  = int((valid[success_col] == 1.0).sum())
    n_fail  = int((valid[success_col] == 0.0).sum())
    base    = n_succ / n_valid if n_valid > 0 else 0.0
    s_sc    = valid.loc[valid[success_col] == 1.0, score_col]
    f_sc    = valid.loc[valid[success_col] == 0.0, score_col]
    bins    = np.arange(0, 1.05, 0.05)

    fig = plt.figure(figsize=(16, 5))
    axes = [fig.add_subplot(gridspec.GridSpec(1, 3, figure=fig)[i]) for i in range(3)]

    for ax, (data, lbl, color) in zip(axes[:2], [
        (s_sc, f"Success  n={len(s_sc)}", _S_COL),
        (f_sc, f"Fail  n={len(f_sc)}",    _F_COL),
    ]):
        ax.hist(data, bins=bins, color=color, edgecolor="white", alpha=0.85)
        if threshold_line is not None:
            ax.axvline(threshold_line, color="red", linestyle="--",
                       linewidth=1.8, label=f"T={threshold_line}")
        ax.set_title(lbl, fontweight="bold", fontsize=11)
        ax.set_xlabel(score_col); ax.set_ylabel("Count")
        ax.set_xlim([0, 1]); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    ax3 = axes[2]
    ax3.hist(s_sc, bins=bins, color=_S_COL, edgecolor="white",
             alpha=0.55, density=True, label=f"Success n={len(s_sc)}")
    ax3.hist(f_sc, bins=bins, color=_F_COL, edgecolor="white",
             alpha=0.55, density=True, label=f"Fail n={len(f_sc)}")
    if threshold_line is not None:
        ax3.axvline(threshold_line, color="red", linestyle="--",
                    linewidth=1.8, label=f"T={threshold_line}")
    ax3.set_title("Overlay (density)", fontweight="bold", fontsize=11)
    ax3.set_xlabel(score_col); ax3.set_ylabel("Density")
    ax3.set_xlim([0, 1]); ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    fig.suptitle(
        f"{title}\nValid={n_valid}  Success={n_succ}  Fail={n_fail}  "
        f"Base={base:.3f}"
        + (f"  T={threshold_line}" if threshold_line is not None else ""),
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    return fig


def plot_plateau(
    sweep_table: pd.DataFrame,
    base_rate: float,
    title: str = "",
) -> plt.Figure:
    """2×2 plateau plots: Precision / F1 / MCC / Lift vs Threshold."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(title, fontsize=11, y=1.01)
    t_vals = sweep_table["Threshold"].values
    low_n  = sweep_table["Low N"].values

    for ax, (metric, hline) in zip(axes.flat, [
        ("Precision", None), ("F1", None), ("MCC", 0.0), ("Lift", 1.0)
    ]):
        if low_n.any():
            low_start = sweep_table.loc[sweep_table["Low N"], "Threshold"].min()
            ax.axvspan(low_start, t_vals.max() + 0.02,
                       alpha=0.07, color="grey", label="Low-N zone")

        ax.plot(t_vals, sweep_table[metric], marker="o",
                color=_C_COL, linewidth=2, label=metric)

        if metric == "Precision":
            try:
                lo = sweep_table["Prec CI 95%"].str.extract(r"\[(.+),").astype(float)
                hi = sweep_table["Prec CI 95%"].str.extract(r", (.+)\]").astype(float)
                ax.fill_between(t_vals, lo.iloc[:,0].values, hi.iloc[:,0].values,
                                alpha=0.15, color=_C_COL, label="95% CI")
            except Exception:
                pass
            ax.axhline(base_rate, color="tomato", linestyle="--",
                       linewidth=1, label=f"Base ({base_rate:.3f})")

        if hline is not None:
            ax.axhline(hline, color="grey", linestyle=":", linewidth=1)

        ax.set_xlabel("Threshold", fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(metric, fontsize=11)
        ax.set_xlim([t_vals.min()-0.02, t_vals.max()+0.02])
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_pr_curve(
    labeled_df: pd.DataFrame,
    score_col: str,
    base_rate: float,
    success_col: str = "success",
    title: str = "",
) -> plt.Figure:
    """PR curve with AUC annotation."""
    fig, ax = plt.subplots(figsize=(7, 5))
    valid   = labeled_df.dropna(subset=[success_col])
    y_true  = valid[success_col].astype(int)
    y_score = valid[score_col]
    if y_true.nunique() >= 2:
        prec_c, rec_c, _ = precision_recall_curve(y_true, y_score)
        pr_auc_val        = sk_auc(rec_c, prec_c)
        ax.plot(rec_c, prec_c, color=_C_COL, linewidth=2,
                label=f"PR curve (AUC={pr_auc_val:.3f})")
        ax.axhline(base_rate, color="tomato", linestyle="--", linewidth=1,
                   label=f"Base ({base_rate:.3f})")
    else:
        ax.text(0.5, 0.5, "Not enough class variation",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(title, fontsize=11)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.05])
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def styled_sweep_table(sweep_table: pd.DataFrame, caption: str = ""):
    """Pandas Styler with yellow Low-N rows. Call display() on the result."""
    float_fmt = {c: "{:.3f}" for c in sweep_table.columns
                 if sweep_table[c].dtype == float}
    float_fmt["Threshold"] = "{:.3f}"
    return (
        sweep_table.style
        .format(float_fmt, na_rep="—")
        .apply(lambda row: [
            "background-color: #fff3cd" if row["Low N"] else ""
        ] * len(row), axis=1)
        .set_caption(caption)
    )
