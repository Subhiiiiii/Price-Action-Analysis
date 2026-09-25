# market_analysis/utils/metrics.py
"""
Threshold sweep: Precision / Recall / F1 / MCC / Lift + block-bootstrap CI.
Works for both absorption (score col = "absorption_strength")
and compression (score col = "compression_score").
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, precision_recall_curve, auc as sk_auc


# ── bootstrap CI ─────────────────────────────────────────────────────────────

def _block_bootstrap_ci(
    signals: np.ndarray,
    labels: np.ndarray,
    n_boot: int,
    block_len: int,
    seed: int,
) -> tuple[float, float]:
    """Block-bootstrap 95% CI on Precision (positional indexing, no OOB)."""
    rng    = np.random.default_rng(seed)
    valid  = ~pd.isna(np.asarray(labels))
    sv     = np.asarray(signals)[valid].astype(bool)
    lv     = np.asarray(labels)[valid]
    n      = len(sv)
    if n == 0:
        return (np.nan, np.nan)
    precs    = []
    n_blocks = int(np.ceil(n / block_len))
    for _ in range(n_boot):
        pos = []
        for _ in range(n_blocks):
            s = rng.integers(0, max(1, n - block_len + 1))
            pos.extend(range(s, min(s + block_len, n)))
        pos = pos[:n]
        sv_ = sv[pos]; lv_ = lv[pos]; ns = sv_.sum()
        if ns > 0:
            precs.append((sv_ & (lv_ == 1)).sum() / ns)
    if not precs:
        return (np.nan, np.nan)
    return float(np.percentile(precs, 2.5)), float(np.percentile(precs, 97.5))


# ── threshold sweep ───────────────────────────────────────────────────────────

def threshold_sweep(
    labeled_df: pd.DataFrame,
    score_col: str,
    success_col: str = "success",
    cfg: dict | None = None,
) -> tuple[pd.DataFrame, float]:
    """
    Sweep a threshold grid and compute Precision/Recall/F1/MCC/Lift + CI.

    Parameters
    ----------
    labeled_df  : DataFrame with score_col and success_col (1/0/NaN).
    score_col   : "absorption_strength" or "compression_score"
    success_col : "sl_success" for absorption, "success" for compression
    cfg         : dict with keys threshold_grid_start/stop/step,
                  min_signal_count, n_bootstrap, random_seed, forward_days.
                  Falls back to COMPRESSION config if None.

    Returns
    -------
    (sweep_table DataFrame, base_rate float)
    """
    from ..config import COMPRESSION
    if cfg is None:
        cfg = COMPRESSION

    tgrid = np.round(
        np.arange(
            cfg["threshold_grid_start"],
            cfg["threshold_grid_stop"],
            cfg["threshold_grid_step"],
        ),
        3,
    )
    min_n     = cfg["min_signal_count"]
    n_boot    = cfg["n_bootstrap"]
    seed      = cfg["random_seed"]
    block_len = cfg.get("forward_days", 10) + 1

    valid      = labeled_df.dropna(subset=[success_col]).reset_index(drop=True)
    base_rate  = float(valid[success_col].mean()) if len(valid) > 0 else np.nan
    scores_arr = valid[score_col].values
    labels_arr = valid[success_col].values

    rows = []
    for T in tgrid:
        sig = scores_arr >= T
        lab = labels_arr.astype(int)

        tp   = int(( sig & (lab == 1)).sum())
        fp   = int(( sig & (lab == 0)).sum())
        fn   = int((~sig & (lab == 1)).sum())
        tn   = int((~sig & (lab == 0)).sum())
        ns   = tp + fp

        prec = tp / ns       if ns > 0           else np.nan
        rec  = tp / (tp+fn)  if (tp+fn) > 0      else np.nan
        f1   = (2*prec*rec/(prec+rec)
                if (prec==prec and rec==rec and (prec+rec)>0) else np.nan)
        lift = prec / base_rate if (prec==prec and base_rate) else np.nan
        try:
            mcc = (matthews_corrcoef(lab, sig.astype(int))
                   if sig.sum()>0 and (~sig).sum()>0 else np.nan)
        except Exception:
            mcc = np.nan

        ci     = _block_bootstrap_ci(sig, labels_arr, n_boot, block_len, seed)
        ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci[0]==ci[0] else "n/a"

        rows.append({
            "Threshold"  : T,
            "N"          : ns,
            "Precision"  : prec,
            "Prec CI 95%": ci_str,
            "Recall"     : rec,
            "F1"         : f1,
            "MCC"        : mcc,
            "Lift"       : lift,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Low N"      : ns < min_n,
        })

    return pd.DataFrame(rows), base_rate


def pr_auc(labeled_df: pd.DataFrame, score_col: str, success_col: str = "success") -> float:
    """PR-AUC for a labeled DataFrame."""
    valid   = labeled_df.dropna(subset=[success_col])
    y_true  = valid[success_col].astype(int)
    y_score = valid[score_col]
    if y_true.nunique() < 2:
        return np.nan
    prec_c, rec_c, _ = precision_recall_curve(y_true, y_score)
    return float(sk_auc(rec_c, prec_c))