
# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1 — SETUP & IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\shresth.shankhdhar\OneDrive - hertshtengroup.com\Documents")

import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score
from itertools import combinations

from market_analysis.config import (
    PRODUCTS, FRONT_N, OHLC_START_DATE, OHLC_END_DATE, ABSORPTION, COMPRESSION
)
from market_analysis.data.ohlc import getData_OHLC_v2, fetch_ohlc_1d, get_ohlc_for_horizon
from market_analysis.utils.contracts import (
    MONTH_CODES, next_n_contract_months, get_front_instruments, fmt_contract
)
from market_analysis.signals.absorption.calculate import calculate_absorption
from market_analysis.signals.absorption.label import label_absorption
from market_analysis.signals.compression.calculate import calculate_compression

# ── Constants ──────────────────────────────────────────────────────────────────
PRODUCT_CODE      = "SM"
TICK_SIZE         = PRODUCTS[PRODUCT_CODE]["tick_size"]          # 0.1
ABS_THRESHOLD     = 0.25    # absorption_strength filter
COMP_THRESHOLD    = 0.30    # compression_score filter
VOLUME_MIN        = 3000    # ignore bars with total volume below this
HORIZONS          = ["1d", "2d", "3d", "5d"]
TODAY             = datetime.date.today()

# Label params (from your best sweep row)
REQUIRED_MOVE_TICKS = 5     # 5 × tick = 0.5 pts
SL_LOOKBACK         = 10
FORWARD_DAYS        = 10

# ── OHLC fetch window per instrument ─────────────────────────────────────────
# Each spread's data starts 7 months before the front-leg's expiry month.
# e.g. SMZ26-SMF27  → front leg = Dec 2026 → start = 2026-05-01
MONTHS_BEFORE_EXPIRY = 7

# ── RV Ratio config ──────────────────────────────────────────────────────────
# All combinations of (numerator_window, denominator_window) will be computed
# as separate features rv_ratio_{N}d_{D}d and each will get its own plot.
# Convention: short_window / long_window  → >1 = vol expanding, <1 = compressing
RV_NUMERATOR_WINDOWS   = [3,4, 5, 10]      # short lookback options (days)
RV_DENOMINATOR_WINDOWS = [15, 20, 40]    # long  lookback options (days)

# ── Kyle Lambda z-score window ───────────────────────────────────────────────
# Rolling window (in bars) used to compute the mean and std of lambda_t
# for the z-score normalisation. Larger = more stable baseline, slower to adapt.
LAMBDA_ZSCORE_WINDOW: int = 20

# ── Kyle Lambda ratio window ───────────────────────────────────────────────
# lambda_ratio = lambda_t / mean(lambda_{t-LAMBDA_RATIO_WINDOW : t-1})
# Measures how current price-impact compares to recent average depth.
# < 1  → market currently harder to move than usual (deep, absorbing flow)
# > 1  → market thinner than usual (each unit of delta moves price more)
LAMBDA_RATIO_WINDOW: int = 3     # how many past bars to average for the denominator
# ─────────────────────────────────────────────────────────────────────────────

print("✅ Setup complete")
print(f"   Product       : {PRODUCT_CODE}  tick={TICK_SIZE}")
print(f"   Abs threshold : {ABS_THRESHOLD}   Comp threshold: {COMP_THRESHOLD}")
print(f"   Volume filter : >{VOLUME_MIN}    Forward days: {FORWARD_DAYS}")
print(f"   Today         : {TODAY}")
print(f"   Fetch window  : {MONTHS_BEFORE_EXPIRY} months before front-leg expiry")
print(f"   RV numerators : {RV_NUMERATOR_WINDOWS}")
print(f"   RV denominators: {RV_DENOMINATOR_WINDOWS}")
print(f"   Lambda zscore window : {LAMBDA_ZSCORE_WINDOW}d")
print(f"   Lambda ratio  window : {LAMBDA_RATIO_WINDOW}d")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2 — INSTRUMENT SELECTION (front 4 + recently expired spreads)
# ═══════════════════════════════════════════════════════════════════════════════

def get_expired_spreads(product_code: str, n_expired: int = 3) -> list[str]:
    """
    Return the last `n_expired` calendar-consecutive spread pairs that expired
    before today. Uses the same traded_months list from config.
    """
    product  = PRODUCTS[product_code]
    traded   = sorted(product["traded_months"])
    prefix   = product["api_prefix"]

    # walk backwards from today to collect recent contract months
    y, m     = TODAY.year, TODAY.month
    collected = []
    while len(collected) < n_expired + 4:   # collect extra buffer
        m -= 1
        if m < 1:
            m = 12; y -= 1
        if m in traded:
            collected.append((y, m))

    collected.reverse()   # chronological order

    pairs = []
    for i in range(len(collected) - 1):
        front = fmt_contract(prefix, *collected[i])
        back  = fmt_contract(prefix, *collected[i + 1])
        pairs.append(f"{front}-{back}")

    # take the most recent n_expired pairs
    return pairs[-n_expired:]


def instr_fetch_start(instrument: str, months_before: int = MONTHS_BEFORE_EXPIRY) -> str:
    """
    Derive the OHLC fetch start date for a spread instrument.

    Logic: parse the front-leg contract → find its expiry month (year, month)
           → subtract `months_before` calendar months → return "YYYY-MM-01".

    Example:
        "SMZ26-SMF27"  → front leg SMZ26 → Dec 2026
                       → 7 months before Dec = May 2026
                       → start = "2026-05-01"
    """
    import re
    # Reverse MONTH_CODES so we can go from letter → month number
    CODE_TO_MONTH = {v: k for k, v in MONTH_CODES.items()}

    front_leg = instrument.split("-")[0]          # e.g. "SMZ26"
    m_obj     = re.match(r"^[A-Za-z]+([A-Z])(\d{2})$", front_leg)
    if not m_obj:
        # Fallback: use global OHLC_START_DATE if we can't parse
        return OHLC_START_DATE

    month_code = m_obj.group(1)                   # e.g. "Z"
    year_2dig  = int(m_obj.group(2))              # e.g. 26
    year       = 2000 + year_2dig                 # e.g. 2026
    month      = CODE_TO_MONTH.get(month_code, 1) # e.g. 12 (December)

    # subtract months_before months
    start_month = month - months_before
    start_year  = year
    while start_month < 1:
        start_month += 12
        start_year  -= 1

    return f"{start_year}-{start_month:02d}-01"


# Front 4 spreads starting from today
front_4 = get_front_instruments(PRODUCT_CODE, "spread",
                                reference_date=TODAY, front_n=4)

# ~3 recently expired spreads for extra data
expired_3 = get_expired_spreads(PRODUCT_CODE, n_expired=3)

ALL_INSTRUMENTS = expired_3 + front_4

print("═" * 60)
print("INSTRUMENTS SELECTED")
print("═" * 60)
print(f"  Expired spreads : {expired_3}")
print(f"  Front 4 spreads : {front_4}")
print(f"  Total           : {len(ALL_INSTRUMENTS)} instruments")
print()
print("  Derived fetch windows (7 months before front-leg expiry):")
for instr in ALL_INSTRUMENTS:
    print(f"    {instr:22s}  start={instr_fetch_start(instr)}")
print("═" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3 — FETCH, SCORE (absorption + compression) FOR ALL INSTRUMENTS
# ═══════════════════════════════════════════════════════════════════════════════

print("═" * 60)
print("FETCHING & SCORING OHLC DATA")
print("═" * 60)

# Store per-instrument scored 1D DataFrames (before aggregation)
scored_1d = {}      # {instrument: df_with_absorption_1d}
scored_nd = {}      # {instrument: {horizon: df}}

for instr in ALL_INSTRUMENTS:
    print(f"\n  ▸ {instr}")

    # Per-instrument fetch window: 7 months before front-leg expiry
    instr_start = instr_fetch_start(instr)
    print(f"    fetch_start={instr_start}")

    # --- 1D fetch & absorption score ---
    df1d = fetch_ohlc_1d(instr, PRODUCT_CODE, getData_OHLC_v2,
                          start_date=instr_start, end_date=OHLC_END_DATE)
    if df1d is None or len(df1d) < 70:
        print(f"    [SKIP] not enough data (rows={len(df1d) if df1d is not None else 0})")
        continue

    df1d_abs  = calculate_absorption(df1d, window=60,
                                      tick_size=TICK_SIZE, strength_threshold=ABS_THRESHOLD)
    df1d_abs  = label_absorption(df1d_abs, tick=TICK_SIZE,
                                  required_move_ticks=REQUIRED_MOVE_TICKS,
                                  sl_lookback=SL_LOOKBACK,
                                  forward_days=FORWARD_DAYS)
    scored_1d[instr] = df1d_abs

    # --- compression on 1D ---
    df1d_comp, _ = calculate_compression(df1d.copy(),
                                          compression_threshold=COMP_THRESHOLD)

    # merge compression_score into absorption df by Date
    comp_merge = df1d_comp[["Date", "compression_score"]].copy()
    comp_merge["Date"] = pd.to_datetime(comp_merge["Date"]).dt.strftime("%Y-%m-%d")
    df1d_abs["Date"]   = pd.to_datetime(df1d_abs["Date"]).dt.strftime("%Y-%m-%d")
    df1d_abs = df1d_abs.merge(comp_merge.rename(columns={"compression_score": "comp_score_1d"}),
                               on="Date", how="left")
    scored_1d[instr] = df1d_abs

    # --- multi-horizon absorption scores (for consistency feature) ---
    scored_nd[instr] = {}
    for hz in HORIZONS:
        from market_analysis.data.ohlc import _aggregate_trailing
        from market_analysis.config import HORIZON_BARS
        n_bars = HORIZON_BARS[hz]
        dfh    = df1d.copy() if n_bars == 1 else _aggregate_trailing(df1d.copy(), n_bars)
        if dfh is None or len(dfh) < 65:
            continue
        dfh_sc = calculate_absorption(dfh, window=60,
                                       tick_size=TICK_SIZE, strength_threshold=ABS_THRESHOLD)
        scored_nd[instr][hz] = dfh_sc
    print(f"    rows={len(df1d_abs)}  "
          f"abs_events={(df1d_abs['absorption_strength'] > ABS_THRESHOLD).sum()}")

print("\n✅ Scoring complete")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4 — FEATURE ENGINEERING
# (all past-only, no forward leakage, computed at bar t)
# ═══════════════════════════════════════════════════════════════════════════════

def build_resolution_features(df: pd.DataFrame,
                                scored_nd_for_instr: dict,
                                tick: float = 0.1,
                                epsilon: float = 1e-9) -> pd.DataFrame:
    """
    Add Stage-2 characterization features to a scored+labeled 1D DataFrame.
    All features use only [t-N : t] — strictly past-only.
    """
    d = df.copy().sort_values("Date").reset_index(drop=True)

    # ── 1. Intrabar: close position within bar ──────────────────────────────
    d["close_pos"] = (d["close"] - d["low"]) / (d["high"] - d["low"] + epsilon)
    # Signed for direction:
    #   BUY  absorption → reversal = price drops → close near LOW is BAD for buyers
    #     close_pos LOW (< 0.3) → sellers won intrabar → reversal hint → signed negative
    #   SELL absorption → reversal = price rises → close near HIGH is BAD for sellers
    # Convention: signed_close_pos > 0 → continuation, < 0 → reversal
    d["signed_close_pos"] = np.where(
        d["absorption_direction"] == "BUY",
        d["close_pos"] - 0.5,          # BUY: high → continuation (+), low → reversal (-)
        0.5 - d["close_pos"],          # SELL: low → continuation (+), high → reversal (-)
    )

    # ── 2. Intrabar: wick asymmetry ──────────────────────────────────────────
    d["upper_wick"] = d["high"] - d[["open", "close"]].max(axis=1)
    d["lower_wick"] = d[["open", "close"]].min(axis=1) - d["low"]
    d["wick_ratio"] = d["upper_wick"] / (d["lower_wick"] + epsilon)
    # BUY absorption: large upper wick → price pushed up but rejected → reversal
    # SELL absorption: large lower wick → price tested down but rejected → reversal
    # Sign: negative → reversal hint
    d["signed_wick_ratio"] = np.where(
        d["absorption_direction"] == "BUY",
        -(d["wick_ratio"] - 1.0),       # BUY: wick_ratio >> 1 → reversal → negative
        (d["wick_ratio"] - 1.0),        # SELL: wick_ratio << 1 → reversal → positive issue
    )
    # simpler: just keep raw wick_ratio and let model learn direction via dir dummy

    # ── 3. Delta ratio (how one-sided) ───────────────────────────────────────
    d["delta_ratio"] = d["buyvolume"] / (d["sellvolume"] + epsilon)

    # ── 4. Delta momentum: is aggressive flow accel or dying? ────────────────
    d["delta_accel"] = d["delta"] - d["delta"].shift(1).rolling(3, min_periods=1).mean()
    # Sign to success: BUY absorption — if delta_accel positive → buyers not exhausted → continuation
    # SELL absorption — if delta_accel negative (sell delta growing) → sellers not exhausted → continuation
    d["signed_delta_accel"] = np.where(
        d["absorption_direction"] == "BUY",
        d["delta_accel"],
        -d["delta_accel"],
    )

    # ── 5. Volume deceleration ───────────────────────────────────────────────
    vol_roll5 = d["volume"].shift(1).rolling(5, min_periods=2).mean()
    d["vol_ratio"] = d["volume"] / (vol_roll5 + epsilon)
    # vol_ratio < 1 → volume drying up → exhaustion → reversal (negative for our sign)
    # signed: positive → continuation hint
    d["signed_vol_ratio"] = d["vol_ratio"] - 1.0   # >0 = escalating, <0 = dying

    # ── 6. Price drift into absorption ───────────────────────────────────────
    d["drift_5"] = (d["close"] - d["close"].shift(5)) / (5 * tick + epsilon)
    # BUY absorption after falling prices (drift_5 < 0) → counter-trend buyers → reversal likely
    # SELL absorption after rising prices (drift_5 > 0) → counter-trend sellers → reversal likely
    # Sign: same direction as absorption = continuation, opposite = reversal
    d["signed_drift_5"] = np.where(
        d["absorption_direction"] == "BUY",
        d["drift_5"],          # rising drift + BUY absorption → continuation
        -d["drift_5"],         # falling drift + SELL absorption → continuation
    )

    # ── 7. Kyle Lambda: market depth anomaly ─────────────────────────────────
    price_change = d["close"].diff().abs()
    abs_delta    = d["delta"].abs()
    d["lambda_t"] = price_change / (abs_delta + epsilon)

    # lambda_ratio: current vs recent average depth
    #   = lambda_t / mean(lambda[t-LAMBDA_RATIO_WINDOW : t-1])
    #   < 1 → market deeper than usual → flow absorbed without moving price much
    #   > 1 → market thinner than usual → each delta unit moves price more
    _lam_ratio_mp          = max(2, LAMBDA_RATIO_WINDOW // 2)
    d["lambda_ratio_avg"]  = (
        d["lambda_t"].shift(1)
        .rolling(LAMBDA_RATIO_WINDOW, min_periods=_lam_ratio_mp)
        .mean()
    )
    d["lambda_ratio"] = d["lambda_t"] / (d["lambda_ratio_avg"] + epsilon)

    # lambda_zscore: z-normalised over a longer baseline window
    _lam_mp    = max(10, LAMBDA_ZSCORE_WINDOW // 3)
    lam_mean   = d["lambda_t"].shift(1).rolling(LAMBDA_ZSCORE_WINDOW, min_periods=_lam_mp).mean()
    lam_std    = d["lambda_t"].shift(1).rolling(LAMBDA_ZSCORE_WINDOW, min_periods=_lam_mp).std()
    d["lambda_zscore"] = (d["lambda_t"] - lam_mean) / (lam_std + epsilon)
    # very negative lambda_zscore = unusually deep market = strong absorption event
    # (window controlled by LAMBDA_ZSCORE_WINDOW; ratio window by LAMBDA_RATIO_WINDOW)

    # ── 8. Realized volatility ratios (all configured combinations) ──────────
    # Config lists RV_NUMERATOR_WINDOWS and RV_DENOMINATOR_WINDOWS (defined in Cell 1).
    # For each (num_w, den_w) pair where num_w < den_w we compute:
    #   rv_ratio_{num_w}d_{den_w}d = rv(num_w) / rv(den_w)
    # >1 = vol expanding (short > long)  |  <1 = vol compressing
    log_ret = np.log(d["close"].replace(0, np.nan) / d["close"].shift(1).replace(0, np.nan))

    # Pre-compute each unique window std once
    all_windows = sorted(set(RV_NUMERATOR_WINDOWS + RV_DENOMINATOR_WINDOWS))
    rv_cache = {}
    for w in all_windows:
        mp = max(2, w // 2)
        rv_cache[w] = log_ret.rolling(w, min_periods=mp).std()

    for num_w in RV_NUMERATOR_WINDOWS:
        for den_w in RV_DENOMINATOR_WINDOWS:
            if num_w >= den_w:
                continue   # skip: numerator must be shorter than denominator
            col = f"rv_ratio_{num_w}d_{den_w}d"
            d[col] = rv_cache[num_w] / (rv_cache[den_w] + epsilon)

    # Keep the original rv_ratio name pointing at the first valid combo
    # so downstream code that references "rv_ratio" still works
    _first_num = next((n for n in RV_NUMERATOR_WINDOWS for den in RV_DENOMINATOR_WINDOWS if n < den), None)
    _first_den = next((den for n in RV_NUMERATOR_WINDOWS for den in RV_DENOMINATOR_WINDOWS if n < den), None)
    if _first_num is not None:
        d["rv_ratio"] = d[f"rv_ratio_{_first_num}d_{_first_den}d"]
    # rv_ratio > 1 → short-window vol > long-window vol → expanding (high-vol regime)
    # rv_ratio < 1 → short-window vol < long-window vol → compressing (quiet)

    # ── 9. Multi-timeframe consistency ───────────────────────────────────────
    # For each bar date, check if absorption_strength > ABS_THRESHOLD on other horizons
    d["Date_str"] = pd.to_datetime(d["Date"]).dt.strftime("%Y-%m-%d")
    tf_scores = {hz: {} for hz in HORIZONS}
    for hz, dfh in scored_nd_for_instr.items():
        if dfh is None or "absorption_strength" not in dfh.columns:
            continue
        dfh_copy = dfh.copy()
        dfh_copy["Date_str"] = pd.to_datetime(dfh_copy["Date"]).dt.strftime("%Y-%m-%d")
        for _, row in dfh_copy.iterrows():
            tf_scores[hz][row["Date_str"]] = row["absorption_strength"]

    consistency_vals = []
    for _, row in d.iterrows():
        date_key = row["Date_str"]
        fires = [
            1 if tf_scores[hz].get(date_key, 0) > ABS_THRESHOLD else 0
            for hz in HORIZONS
        ]
        consistency_vals.append(np.mean(fires))
    d["tf_consistency"] = consistency_vals

    # ── 10. Dual signal: both absorption AND compression fire ─────────────────
    d["dual_signal"] = (
        (d["absorption_strength"] > ABS_THRESHOLD) &
        (d.get("comp_score_1d", pd.Series(0, index=d.index)).fillna(0) > COMP_THRESHOLD)
    ).astype(float)

    # ── 11. Direction dummy (BUY=1, SELL=0) ──────────────────────────────────
    d["is_buy_absorption"] = (d["absorption_direction"] == "BUY").astype(float)

    return d


# Apply feature engineering to all instruments
print("═" * 60)
print("FEATURE ENGINEERING")
print("═" * 60)

featured = {}
for instr, df in scored_1d.items():
    nd_for_instr = scored_nd.get(instr, {})
    featured[instr] = build_resolution_features(df, nd_for_instr, tick=TICK_SIZE)
    print(f"  ✓ {instr}  columns={len(featured[instr].columns)}")

print("✅ Feature engineering complete")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5 — POOL ABSORPTION EVENTS & SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("═" * 60)
print("POOLING ABSORPTION EVENTS (volume > 3000, strength > 0.25)")
print("═" * 60)

FEATURE_COLS = [
    "signed_close_pos",
    "signed_wick_ratio",
    "signed_delta_accel",
    "signed_vol_ratio",
    "signed_drift_5",
    "lambda_zscore",
    "rv_ratio",
    "tf_consistency",
    "dual_signal",
    "is_buy_absorption",
    # raw (unsigned) versions also included for exploration
    "close_pos",
    "wick_ratio",
    "vol_ratio",
    "delta_ratio",
    "lambda_ratio",
]

all_events = []
for instr, df in featured.items():
    mask = (
        (df["absorption_strength"] > ABS_THRESHOLD) &
        (df["volume"] > VOLUME_MIN) &
        (df["sl_success"].notna())
    )
    subset = df[mask].copy()
    subset["instrument"] = instr
    all_events.append(subset)

if not all_events:
    print("❌ No events found — check thresholds or data fetch")
else:
    pool = pd.concat(all_events, ignore_index=True)
    pool = pool.sort_values("Date").reset_index(drop=True)

    n_total   = len(pool)
    n_success = int(pool["sl_success"].sum())
    n_fail    = int((pool["sl_success"] == 0).sum())
    base_rate = n_success / n_total if n_total > 0 else 0

    print(f"\n  Total absorption events : {n_total}")
    print(f"  Success (reversal)      : {n_success}  ({100*base_rate:.1f}%)")
    print(f"  Failure                 : {n_fail}")
    print(f"\n  Per instrument:")
    for instr in pool["instrument"].unique():
        sub = pool[pool["instrument"] == instr]
        s = int(sub["sl_success"].sum())
        print(f"    {instr:20s}  n={len(sub):3d}  success={s}")
    print()

    # ensure features are numeric
    for fc in FEATURE_COLS:
        if fc in pool.columns:
            pool[fc] = pd.to_numeric(pool[fc], errors="coerce")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6 — DISTRIBUTION PLOTS: Feature separation (success vs fail)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Helper: plot a set of features and return separation stats ────────────────
def _plot_feature_grid(pool_df, feature_list, title_prefix=""):
    """
    Plots histograms for each (feat, label) tuple in feature_list.
    Returns a list of separation stat dicts.
    """
    suc = pool_df[pool_df["sl_success"] == 1]
    fai = pool_df[pool_df["sl_success"] == 0]
    stats_out = []

    n_cols = 3
    n_rows = (len(feature_list) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.5))
    axes = axes.flatten()

    for ax, (feat, label) in zip(axes, feature_list):
        if feat not in pool_df.columns:
            ax.set_visible(False); continue
        s_vals = suc[feat].dropna()
        f_vals = fai[feat].dropna()
        if len(s_vals) == 0 or len(f_vals) == 0:
            ax.set_visible(False); continue

        ax.hist(f_vals, bins=20, alpha=0.5, color="#e74c3c", density=True,
                label=f"Fail (n={len(f_vals)})")
        ax.hist(s_vals, bins=20, alpha=0.5, color="#2ecc71", density=True,
                label=f"Success (n={len(s_vals)})")
        ks_stat, ks_pval = stats.ks_2samp(s_vals, f_vals)
        mean_diff = s_vals.mean() - f_vals.mean()
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_xlabel(f"KS={ks_stat:.2f}  p={ks_pval:.3f}  Δmean={mean_diff:+.3f}", fontsize=8)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        stats_out.append({"feature": feat, "ks_stat": ks_stat, "ks_pval": ks_pval,
                           "mean_success": s_vals.mean(), "mean_fail": f_vals.mean(),
                           "mean_diff": mean_diff})

    for ax in axes[len(feature_list):]:
        ax.set_visible(False)
    plt.suptitle(f"{title_prefix}\n"
                 f"(Abs≥{ABS_THRESHOLD}, Vol>{VOLUME_MIN}, "
                 f"n={n_total}, base={base_rate:.1%})",
                 fontsize=10, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()
    return stats_out


# ── Section A: core signed features ──────────────────────────────────────────
CORE_PLOT_FEATURES = [
    ("signed_close_pos",   "Signed Close Position\n(+→ continuation, -→ reversal)"),
    ("signed_wick_ratio",  "Signed Wick Ratio\n(+→ continuation, -→ reversal)"),
    ("signed_delta_accel", "Signed Delta Accel\n(+→ flow accel, -→ flow dying)"),
    ("signed_vol_ratio",   "Signed Vol Ratio\n(+→ vol escalating, -→ vol dying)"),
    ("signed_drift_5",     "Signed 5-Day Drift\n(+→ with trend, -→ counter trend)"),
    ("lambda_zscore",      "Kyle Lambda Z-Score\n(- = unusually deep market)"),
    ("lambda_ratio",       f"Lambda Ratio (w={LAMBDA_RATIO_WINDOW}d)\n(<1=deep/absorbing, >1=thin/reactive)"),
    ("tf_consistency",     "Multi-TF Consistency\n(fraction of HZs that fire)"),
    ("dual_signal",        "Dual Signal\n(Abs + Compression both fire)"),
]

print("═" * 60)
print("SECTION A — CORE FEATURE SEPARATION (success vs failure)")
print("═" * 60)
sep_stats = _plot_feature_grid(pool, CORE_PLOT_FEATURES,
                                title_prefix="Core Features: Success vs Failure")

sep_df = pd.DataFrame(sep_stats).sort_values("ks_stat", ascending=False)
print("\nCore Feature Separation (KS test, sorted by KS stat):")
print(sep_df[["feature", "ks_stat", "ks_pval", "mean_success", "mean_fail", "mean_diff"]].to_string(index=False))


# ── Section B: RV Ratio — all window combinations ────────────────────────────
print()
print("═" * 60)
print("SECTION B — RV RATIO COMBINATIONS (all num/den window pairs)")
print(f"  Numerator  windows: {RV_NUMERATOR_WINDOWS}")
print(f"  Denominator windows: {RV_DENOMINATOR_WINDOWS}")
print("═" * 60)

RV_PLOT_FEATURES = []
for num_w in RV_NUMERATOR_WINDOWS:
    for den_w in RV_DENOMINATOR_WINDOWS:
        if num_w >= den_w:
            continue
        col   = f"rv_ratio_{num_w}d_{den_w}d"
        label = (f"RV Ratio {num_w}d / {den_w}d\n"
                 f"(>1=vol expanding, <1=compressing)")
        RV_PLOT_FEATURES.append((col, label))

if RV_PLOT_FEATURES:
    rv_sep_stats = _plot_feature_grid(pool, RV_PLOT_FEATURES,
                                       title_prefix="RV Ratio Combinations: Success vs Failure")
    rv_sep_df = pd.DataFrame(rv_sep_stats).sort_values("ks_stat", ascending=False)
    print("\nRV Ratio Separation (KS test, sorted by KS stat):")
    print(rv_sep_df[["feature", "ks_stat", "ks_pval",
                     "mean_success", "mean_fail", "mean_diff"]].to_string(index=False))
    print(f"\n  → Best RV combo: {rv_sep_df.iloc[0]['feature']}  "
          f"KS={rv_sep_df.iloc[0]['ks_stat']:.3f}")
    # Store best for use in model sweep
    BEST_RV_FEATURE = rv_sep_df.iloc[0]["feature"]
else:
    print("  No valid RV combos (all numerators >= denominators). Check config.")
    BEST_RV_FEATURE = "rv_ratio"

# Combined separation table (all features)
all_sep_df = pd.concat([sep_df, rv_sep_df if RV_PLOT_FEATURES else pd.DataFrame()],
                        ignore_index=True).sort_values("ks_stat", ascending=False)
print("\n── ALL FEATURES combined (sorted by KS stat) ──")
print(all_sep_df[["feature", "ks_stat", "ks_pval", "mean_diff"]].to_string(index=False))
success_events = pool[pool["sl_success"] == 1]
fail_events    = pool[pool["sl_success"] == 0]


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7 — WALK-FORWARD TRAIN / TEST SPLIT
# ═══════════════════════════════════════════════════════════════════════════════

pool_sorted = pool.sort_values("Date").reset_index(drop=True)
split_idx   = int(len(pool_sorted) * 0.70)
train_df    = pool_sorted.iloc[:split_idx].copy()
test_df     = pool_sorted.iloc[split_idx:].copy()

print("═" * 60)
print("WALK-FORWARD SPLIT")
print("═" * 60)
print(f"  Train: {len(train_df)} events | {train_df['Date'].min()} → {train_df['Date'].max()}")
print(f"  Test : {len(test_df)} events  | {test_df['Date'].min()} → {test_df['Date'].max()}")
print(f"  Train base rate: {train_df['sl_success'].mean():.1%}")
print(f"  Test  base rate: {test_df['sl_success'].mean():.1%}")


def get_Xy(df: pd.DataFrame, feats: list[str]):
    """Extract X, y matrices from a pool DataFrame, dropping NaN rows."""
    sub  = df[feats + ["sl_success"]].dropna()
    X    = sub[feats].values.astype(float)
    y    = sub["sl_success"].values.astype(int)
    return X, y, sub.index


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 8 — FEATURE COMBO SWEEP (SGD Logistic Regression)
# ═══════════════════════════════════════════════════════════════════════════════

print("═" * 60)
print("FEATURE COMBINATION SWEEP (SGD Logistic Regression)")
print("═" * 60)

# Signed feature candidates (theoretically motivated).
# rv_ratio is replaced by the best-KS combo found in Cell 6 Section B.
SIGNED_FEATS = [
    "signed_close_pos",
    "signed_delta_accel",
    "signed_vol_ratio",
    "signed_drift_5",
    "lambda_zscore",
    "lambda_ratio",        # depth ratio vs recent avg (LAMBDA_RATIO_WINDOW)
    BEST_RV_FEATURE,       # best RV window combo by KS separation
    "tf_consistency",
    "signed_wick_ratio",
]

# Filter to only features that exist and have enough variance
available_feats = [
    f for f in SIGNED_FEATS
    if f in train_df.columns and train_df[f].dropna().std() > 1e-6
]
print(f"  Available features: {available_feats}\n")

# Combos of size 2–6 (theoretically motivated pairs first, then expand)
# Plus one full-feature model
combo_sizes = [2, 3, 4, 5]
combos      = []
for size in combo_sizes:
    combos.extend(list(combinations(available_feats, size)))
# Add full model
combos.append(tuple(available_feats))

print(f"  Total combos to evaluate: {len(combos)}")

def make_sgd_pipeline():
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=0.01,
        max_iter=1000,
        tol=1e-4,
        random_state=42,
        class_weight="balanced",
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


results = []
for combo in combos:
    feats = list(combo)
    X_tr, y_tr, _ = get_Xy(train_df, feats)
    X_te, y_te, _ = get_Xy(test_df,  feats)

    if len(X_tr) < 20 or len(X_te) < 5:
        continue
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        continue

    pipe = make_sgd_pipeline()
    try:
        pipe.fit(X_tr, y_tr)
    except Exception:
        continue

    y_pred_prob = pipe.predict_proba(X_te)[:, 1]
    # Use 0.5 as default threshold
    y_pred = (y_pred_prob >= 0.5).astype(int)

    prec   = precision_score(y_te, y_pred, zero_division=0)
    rec    = recall_score(y_te,    y_pred, zero_division=0)
    f1     = f1_score(y_te,        y_pred, zero_division=0)
    # Lift over base rate
    base   = y_te.mean() if y_te.mean() > 0 else 1e-9
    lift   = prec / base

    results.append({
        "features"  : feats,
        "n_feats"   : len(feats),
        "precision" : round(prec, 4),
        "recall"    : round(rec,  4),
        "f1"        : round(f1,   4),
        "lift"      : round(lift, 4),
        "n_train"   : len(X_tr),
        "n_test"    : len(X_te),
    })

results_df = pd.DataFrame(results).sort_values("precision", ascending=False).reset_index(drop=True)

print("\n  Top 15 combos by Precision on TEST set:")
print("─" * 80)
for i, row in results_df.head(15).iterrows():
    print(f"  [{i+1:2d}] n_feats={row['n_feats']}  "
          f"P={row['precision']:.3f}  R={row['recall']:.3f}  "
          f"F1={row['f1']:.3f}  Lift={row['lift']:.2f}")
    print(f"       features: {row['features']}")

print(f"\n  Base rate on test: {test_df['sl_success'].mean():.1%}")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 9 — TOP MODEL DEEP DIVE (threshold sweep + coefficients)
# ═══════════════════════════════════════════════════════════════════════════════

# Pick top combo by precision
best_row   = results_df.iloc[0]
best_feats = best_row["features"]
print("═" * 60)
print(f"TOP MODEL ANALYSIS")
print("═" * 60)
print(f"  Features: {best_feats}")
print(f"  Train n={best_row['n_train']}  Test n={best_row['n_test']}")
print(f"  Precision={best_row['precision']:.3f}  Recall={best_row['recall']:.3f}  "
      f"F1={best_row['f1']:.3f}  Lift={best_row['lift']:.2f}")

X_tr, y_tr, _ = get_Xy(train_df, best_feats)
X_te, y_te, _ = get_Xy(test_df,  best_feats)

best_pipe = make_sgd_pipeline()
best_pipe.fit(X_tr, y_tr)
y_prob_te = best_pipe.predict_proba(X_te)[:, 1]

# ── Coefficients ──────────────────────────────────────────────────────────────
coefs = best_pipe.named_steps["clf"].coef_[0]
coef_df = pd.DataFrame({
    "feature": best_feats,
    "coefficient": coefs,
    "abs_coef": np.abs(coefs)
}).sort_values("abs_coef", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Coefficient bar chart
ax = axes[0]
colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in coef_df["coefficient"]]
ax.barh(coef_df["feature"], coef_df["coefficient"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Model Coefficients\n(+ → predicts success/continuation)", fontweight="bold")
ax.set_xlabel("Coefficient (scaled)")
ax.grid(alpha=0.3, axis="x")

# Probability distribution by outcome
ax = axes[1]
ax.hist(y_prob_te[y_te == 0], bins=15, alpha=0.6, color="#e74c3c", density=True, label="Fail")
ax.hist(y_prob_te[y_te == 1], bins=15, alpha=0.6, color="#2ecc71", density=True, label="Success")
ax.axvline(0.5, color="black", linestyle="--", label="Threshold=0.5")
ax.set_title("Predicted Probability Distribution\n(Test set)", fontweight="bold")
ax.set_xlabel("P(success)")
ax.legend()
ax.grid(alpha=0.3)

plt.suptitle(f"Top Model Deep Dive  |  Features: {best_feats}", fontsize=9)
plt.tight_layout()
plt.show()

print("\n  Coefficients (sorted by |coef|):")
print(coef_df[["feature", "coefficient"]].to_string(index=False))


# ── Threshold sweep on test set ───────────────────────────────────────────────
thresholds = np.arange(0.30, 0.85, 0.05)
sweep_rows = []
for t in thresholds:
    y_pred_t = (y_prob_te >= t).astype(int)
    n_pred   = y_pred_t.sum()
    if n_pred == 0:
        continue
    p = precision_score(y_te, y_pred_t, zero_division=0)
    r = recall_score(y_te,    y_pred_t, zero_division=0)
    f = f1_score(y_te,        y_pred_t, zero_division=0)
    sweep_rows.append({"threshold": round(t, 2), "N_predicted": n_pred,
                        "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)})

sweep_res = pd.DataFrame(sweep_rows)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(sweep_res["threshold"], sweep_res["precision"], "g-o", label="Precision", linewidth=2)
ax.plot(sweep_res["threshold"], sweep_res["recall"],    "b-s", label="Recall",    linewidth=2)
ax.plot(sweep_res["threshold"], sweep_res["f1"],        "r-^", label="F1",        linewidth=2)
ax2 = ax.twinx()
ax2.bar(sweep_res["threshold"], sweep_res["N_predicted"], alpha=0.2, color="gray",
        width=0.03, label="N predicted")
ax2.set_ylabel("N predicted signals", color="gray")
ax.axhline(test_df["sl_success"].mean(), color="black", linestyle=":", label=f"Base rate")
ax.set_xlabel("Probability threshold")
ax.set_ylabel("Score")
ax.set_title("Threshold Sweep on TEST Set\n(Top Model)", fontweight="bold")
ax.legend(loc="upper left"); ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\n  Threshold Sweep Table (TEST):")
print(sweep_res.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 10 — BUY vs SELL ABSORPTION SPLIT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

print("═" * 60)
print("BUY vs SELL ABSORPTION SPLIT")
print("═" * 60)

for direction in ["BUY", "SELL"]:
    sub = pool[pool["absorption_direction"] == direction].copy()
    n   = len(sub)
    if n < 15:
        print(f"  {direction}: too few events ({n}), skipping")
        continue

    s_sub = sub[sub["sl_success"] == 1]
    f_sub = sub[sub["sl_success"] == 0]
    print(f"\n  {direction} ABSORPTION  n={n}  "
          f"success={len(s_sub)} ({100*len(s_sub)/n:.1f}%)  "
          f"fail={len(f_sub)}")

    feat_means = []
    for feat in available_feats:
        if feat not in sub.columns:
            continue
        sv = s_sub[feat].dropna()
        fv = f_sub[feat].dropna()
        if len(sv) < 3 or len(fv) < 3:
            continue
        ks, pv = stats.ks_2samp(sv, fv)
        feat_means.append({
            "feature": feat,
            "mean_success": round(sv.mean(), 4),
            "mean_fail"   : round(fv.mean(), 4),
            "diff"        : round(sv.mean() - fv.mean(), 4),
            "ks_stat"     : round(ks, 3),
            "p_value"     : round(pv, 3),
        })

    feat_means_df = pd.DataFrame(feat_means).sort_values("ks_stat", ascending=False)
    print(feat_means_df.to_string(index=False))

    # Train direction-specific model if enough data
    sub_sorted = sub.sort_values("Date").reset_index(drop=True)
    sp = int(len(sub_sorted) * 0.70)
    tr_d = sub_sorted.iloc[:sp]
    te_d = sub_sorted.iloc[sp:]
    if len(tr_d) < 15 or len(te_d) < 5:
        print(f"  [{direction}] Not enough data for direction model")
        continue

    X_tr_d, y_tr_d, _ = get_Xy(tr_d, best_feats)
    X_te_d, y_te_d, _ = get_Xy(te_d, best_feats)
    if len(np.unique(y_tr_d)) < 2:
        print(f"  [{direction}] Only one class in train, skipping")
        continue

    pipe_d = make_sgd_pipeline()
    pipe_d.fit(X_tr_d, y_tr_d)
    if len(np.unique(y_te_d)) < 2:
        print(f"  [{direction}] Only one class in test set, skipping metrics")
        continue
    y_pred_d = pipe_d.predict(X_te_d)
    print(f"\n  [{direction}] Direction-specific model (same features):")
    print(f"    Precision={precision_score(y_te_d, y_pred_d, zero_division=0):.3f}  "
          f"Recall={recall_score(y_te_d, y_pred_d, zero_division=0):.3f}  "
          f"F1={f1_score(y_te_d, y_pred_d, zero_division=0):.3f}")
    coefs_d = pipe_d.named_steps["clf"].coef_[0]
    print(f"    Coefficients: " +
          "  ".join(f"{f}={c:+.3f}" for f, c in zip(best_feats, coefs_d)))


print("\n" + "═" * 60)
print("ANALYSIS COMPLETE")
print("═" * 60)
