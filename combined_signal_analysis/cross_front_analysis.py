"""Descriptive same-signal timing across fronts/products; reuse RUN, no API calls."""
from itertools import combinations
from pathlib import Path
import re

import numpy as np
import pandas as pd

SIGNALS = ("ABS_BUY", "ABS_SELL", "COMPRESSION", "UP_TREND_EXHAUSTION",
           "DOWN_TREND_EXHAUSTION", "UP_PRICE_EXPANSION", "DOWN_PRICE_EXPANSION")


def _dates(values):
    """One timezone/date convention for raw bars, indicators and events."""
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True)).normalize()


def _coordinates(instrument, product):
    """Calendar-month distances describe how near the two contract legs are."""
    match = re.fullmatch(rf"{re.escape(product)}([FGHJKMNQUVXZ])(\d{{2}})-(?:{re.escape(product)})?([FGHJKMNQUVXZ])(\d{{2}})", instrument)
    if not match:
        raise ValueError(f"Cannot read contract months: {instrument}")
    months = "FGHJKMNQUVXZ"
    a, ya, b, yb = match.groups()
    return ((2000 + int(ya)) * 12 + months.index(a),
            (2000 + int(yb)) * 12 + months.index(b))


def _signal_inputs(run):
    """Extract signal dates AND scored dates, excluding warmup from denominators.

    Absorption uses already deduplicated events at detection END, not every
    candidate duration. Compression/PVD count daily observations, not episodes.
    """
    from market_analysis.finding_threshold import expected_profit_analysis as ep
    records, ready, info, calendar = [], {}, {}, set()
    for instrument, result in run["instruments"].items():
        daily = result["daily"].copy()
        daily["Date"] = _dates(daily["Date"])
        calendar.update(daily["Date"])
        meta = result["absorption_metadata"]
        start, end = meta.get("analysis_start"), meta.get("analysis_end")
        start = daily.Date.min() if start in (None, "None") else pd.to_datetime(start, utc=True).normalize()
        end = daily.Date.max() if end in (None, "None") else pd.to_datetime(end, utc=True).normalize()
        daily = daily.loc[daily.Date.between(start, end)]
        product = result["threshold"]["product"]
        info[instrument] = {"product": product, "structure": result["threshold"]["structure"],
                            "coordinates": _coordinates(instrument, product)}
        features = ep.build_v2_features(result["raw"], result["settings"]["candle_lengths"], result["abs_params"])
        valid = np.isfinite(features[["z_delta", "z_price", "atr"]]).all(axis=1) & features.atr.gt(0)
        absorption_dates = set(_dates(features.loc[valid & features.Date.between(start, end), "Date"]))
        comp_dates = set(daily.loc[np.isfinite(daily.compression_score), "Date"])
        pvd_dates = set(daily.loc[np.isfinite(daily.pvd_z), "Date"])
        for signal in SIGNALS:
            ready[instrument, signal] = absorption_dates if signal.startswith("ABS_") else comp_dates if signal == "COMPRESSION" else pvd_dates
        events = result["events"]
        if not events.empty:
            for event in events.itertuples():
                date = _dates([event.Date])[0]
                if start <= date <= end:
                    records.append((instrument, product, "ABS_" + event.absorption_direction, date))
        for row in daily.itertuples():
            if row.compression_score >= result["settings"]["compression_threshold"]:
                records.append((instrument, product, "COMPRESSION", row.Date))
            if row.pvd_signal in SIGNALS:
                records.append((instrument, product, row.pvd_signal, row.Date))
    ledger = pd.DataFrame(records, columns=["instrument", "product", "signal", "Date"]).drop_duplicates()
    return ledger, ready, info, sorted(calendar)


def _compare(ledger, ready, info, calendar, windows, related_product_groups):
    """Each eligible anchor matches the nearest same-type partner once.

    Symmetric windows include later signals: descriptive research, not a live
    entry filter. Full target-window coverage is required; missing scores/data
    cannot silently become 'no match'. A partner may serve multiple anchors.
    """
    position = {date: i for i, date in enumerate(calendar)}
    groups = [set(group) for group in related_product_groups]
    pairs, matches, pools = [], [], {}
    for a, b in combinations(sorted(info), 2):
        pa, pb = info[a]["product"], info[b]["product"]
        if pa != pb and not any({pa, pb} <= group for group in groups):
            continue
        scope = "same_product:" + pa if pa == pb else "related_products:" + "_".join(sorted([pa, pb]))
        ca, cb = info[a]["coordinates"], info[b]["coordinates"]
        distance = {"front_month_gap": abs(ca[0] - cb[0]), "back_month_gap": abs(ca[1] - cb[1]),
                    "tenor_month_gap": abs((ca[1] - ca[0]) - (cb[1] - cb[0]))}
        for signal in SIGNALS:
            dates = {inst: sorted(ledger.loc[(ledger.instrument == inst) & (ledger.signal == signal), "Date"])
                     for inst in (a, b)}
            for window in windows:
                row = dict(scope=scope, instrument_a=a, instrument_b=b, structure_a=info[a]["structure"],
                           structure_b=info[b]["structure"], signal=signal, within_trading_bars=window, **distance)
                pool = pools.setdefault((scope, signal, window), {"total": set(), "eligible": set(), "matched": set()})
                gaps = []
                for source, target, suffix in ((a, b, "a"), (b, a, "b")):
                    eligible, hit = 0, 0
                    target_ready = ready[target, signal]
                    for date in dates[source]:
                        anchor = (source, signal, date)
                        pool["total"].add(anchor)
                        i = position[date]
                        if date not in ready[source, signal] or i - window < 0 or i + window >= len(calendar):
                            continue
                        if not set(calendar[i-window:i+window+1]) <= target_ready:
                            continue
                        eligible += 1
                        pool["eligible"].add(anchor)
                        candidates = [d for d in dates[target] if abs(position[d] - i) <= window]
                        if not candidates:
                            continue
                        buddy = min(candidates, key=lambda d: (abs(position[d] - i), d))
                        lag = position[buddy] - i
                        hit += 1
                        pool["matched"].add(anchor)
                        gaps.append(abs(lag))
                        matches.append(dict(scope=scope, signal=signal, within_trading_bars=window,
                                            source_instrument=source, target_instrument=target,
                                            source_date=date, target_date=buddy, lag_trading_bars=lag,
                                            absolute_gap_trading_bars=abs(lag), **distance))
                    row.update({f"total_signals_{suffix}": len(dates[source]), f"eligible_signals_{suffix}": eligible,
                                f"matched_signals_{suffix}": hit,
                                f"match_ratio_{suffix}": hit / eligible if eligible else np.nan})
                row["mean_matched_gap_bars"] = float(np.mean(gaps)) if gaps else np.nan
                pairs.append(row)
    summaries = []
    for (scope, signal, window), pool in pools.items():
        n, hits = len(pool["eligible"]), len(pool["matched"])
        summaries.append(dict(scope=scope, signal=signal, within_trading_bars=window,
                              total_anchor_signals=len(pool["total"]), eligible_anchor_signals=n,
                              matched_anchor_signals=hits, match_ratio=hits / n if n else np.nan))
    return {"pair_table": pd.DataFrame(pairs), "product_summary": pd.DataFrame(summaries),
            "matches": pd.DataFrame(matches), "signal_ledger": ledger}


def analyze_cross_front_signals(run, windows=(0, 1, 2, 5), related_product_groups=(("SM", "BO"),), save=True):
    """Return pair ratios, deduplicated product ratios and exact closest matches.

    windows=0 means same date; 2 means +/-2 observed trading sessions. The
    calendar is the union of RUN bar dates, so weekends do not add a bar.
    Product summary counts an anchor once even if it matches several fronts.
    This is conditional signal co-occurrence, not profitability/causation.
    """
    windows = list(windows)
    if not windows or any(isinstance(w, bool) or not isinstance(w, (int, np.integer)) or w < 0 for w in windows):
        raise ValueError("windows must contain non-negative integer trading-bar gaps")
    ledger, ready, info, calendar = _signal_inputs(run)
    output = _compare(ledger, ready, info, calendar, sorted(set(windows)), related_product_groups)
    if save:
        folder = Path(run["output_dir"]) / "cross_front"
        folder.mkdir(parents=True, exist_ok=True)
        for name, table in output.items():
            table.to_csv(folder / f"{name}.csv", index=False)
    return output
