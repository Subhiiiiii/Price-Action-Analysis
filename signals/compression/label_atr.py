# market_analysis/signals/compression/label_atr.py
"""
ATR-based compression labeling.

For each scored bar t:
  req      = k_atr × ATR[t]                      (ATR = mean true range, causal)
  move     = max_{i=1..fwd} |Close[t+i] - Close[t]|
  outcome  = CONTINUATION  if any of the next `lookahead` bars is also compressed
             SUCCESS       if move > req
             FAILURE       otherwise
             CENSORED      forward window incomplete
             WARMUP/NO_ATR indicators not ready

Forward moves are computed on the FULL series (no row dropping first).
Zero-score rows are filtered only AFTER labeling.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

_MODES = {"exclude": np.nan, "as_success": 1.0, "as_failure": 0.0}


def label_compression_atr(
    df: pd.DataFrame,
    k_atr: float = 1.0,
    forward_days: int = 10,
    atr_col: str = "atr_label",
    score_col: str = "compression_score",
    compression_threshold: float = 0.30,
    continuation_threshold: float | None = None,
    continuation_lookahead: int = 1,
    continuation_mode: str = "exclude",
    include_zero_scores: bool = False,
) -> pd.DataFrame:
    if continuation_mode not in (*_MODES, "ignore"):
        raise ValueError(f"continuation_mode must be one of {[*_MODES, 'ignore']}")
    cont_thr = compression_threshold if continuation_threshold is None else continuation_threshold
    L = min(continuation_lookahead, forward_days)

    d = df.sort_values("Date").reset_index(drop=True).copy()
    n = len(d)
    close = d["close"].to_numpy(float)
    atr   = d[atr_col].to_numpy(float)
    score = d[score_col].to_numpy(float)

    max_move   = np.full(n, np.nan)
    success    = np.full(n, np.nan)
    next_score = np.full(n, np.nan)
    is_cont    = np.zeros(n, dtype=bool)
    outcome    = np.full(n, "WARMUP", dtype=object)

    for i in range(n):
        if i + 1 < n:
            next_score[i] = score[i + 1]
        if np.isnan(score[i]):
            continue                                   # WARMUP
        if not np.isfinite(atr[i]) or atr[i] <= 0:
            outcome[i] = "NO_ATR"; continue
        if i + forward_days >= n:
            outcome[i] = "CENSORED"; continue

        move = np.abs(close[i + 1 : i + forward_days + 1] - close[i]).max()
        max_move[i] = move
        cont = bool(np.nansum(score[i + 1 : i + 1 + L] >= cont_thr) > 0)
        is_cont[i] = cont
        raw = 1.0 if move > k_atr * atr[i] else 0.0

        if cont and continuation_mode != "ignore":
            outcome[i] = "CONTINUATION"
            success[i] = _MODES[continuation_mode]
        else:
            outcome[i] = "SUCCESS" if raw == 1.0 else "FAILURE"
            success[i] = raw

    d["atr_at_signal"]         = atr
    d["req_move"]              = k_atr * atr
    d["max_move"]              = max_move
    d["move_in_atr"]           = max_move / atr
    d["next_compression_score"]= next_score
    d["is_continuation"]       = is_cont
    d["outcome"]               = outcome
    d["success"]               = success

    d = d[d[score_col].notna()]
    if not include_zero_scores:
        d = d[d[score_col] > 0]
    return d.reset_index(drop=True)