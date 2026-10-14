"""Copy this entire file into a notebook, or paste each # %% cell separately.

Definitions are import-safe. The final cells run only in a notebook/script,
not when imported for testing. No Groq calls. Excel is read, never modified.
Requires numpy/pandas/openpyxl; plots use matplotlib and notebook display uses
IPython. Research maxima are opportunity measurements, not realized trade P&L.
"""

# %% 1. Imports and editable inputs
from pathlib import Path
import importlib
import json
import re
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\shresth.shankhdhar\OneDrive - hertshtengroup.com\Documents\market_analysis")
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))
config = importlib.import_module("market_analysis.config")
ep = importlib.import_module("market_analysis.finding_threshold.expected_profit_analysis")
importlib.reload(config)
importlib.reload(ep)
ohlc = importlib.import_module("market_analysis.data.ohlc")
importlib.reload(ohlc)
from market_analysis.signals.compression.calculate import calculate_compression
from market_analysis.signals.pvd.calculate import calculate_pvd
from market_analysis.utils.contracts import MONTH_CODES

ABS_CSV_PATHS = {
    "SM": PROJECT_ROOT / "output/llm_thresholds/SM_2026-10-07_gpt_oss_thresholds.csv",
    "BO": PROJECT_ROOT / "output/llm_thresholds/BO_2026-10-06_gpt_oss_thresholds.csv",
}
MANUAL_XLSX = PROJECT_ROOT / "manual_threshold/threshold_manual.xlsx"
SETTINGS = {
    "as_of": None,                       # None = local run date; completed bars only
    "months_back": config.ABSORPTION_EXPECTED_PROFIT["months_back"],
    "candle_lengths": [1, 2, 3, 5],
    "before_bars": 5,                    # raw trading bars before absorption START
    "forward_bars": 10,                  # raw trading bars after absorption END
    "compression_threshold": 0.30,
    "min_pattern_events": 4,             # flag low N; do not hide rare patterns
    "data_mode": "auto",                 # cache first; fetch only missing head/tail
    "allow_cached_on_fetch_error": True,  # show stale-coverage warning, keep analysis usable
    "output_dir": PROJECT_ROOT / "combined_signal_analysis/outputs",
}
ABS_PARAMS = dict(config.ABSORPTION_EXPECTED_PROFIT)  # stop/move/window overrides here
PVD_PARAMS = dict(config.PVD)                      # score windows, not Excel thresholds


# %% 2. Threshold loading and explicit nearest-contract interpolation
def contract_coordinates(instrument, product):
    """Month coordinates + traded-month gap: adjacent legs=spread, wider=jumbo."""
    instrument = str(instrument).strip().upper().replace(" ", "")
    pattern = rf"{re.escape(product)}([FGHJKMNQUVXZ])(\d{{2}})-(?:{re.escape(product)})?([FGHJKMNQUVXZ])(\d{{2}})"
    match = re.fullmatch(pattern, instrument)
    if not match:
        raise ValueError(f"Invalid instrument for {product}: {instrument}")
    month_numbers = {v: k for k, v in MONTH_CODES.items()}
    a, ya, b, yb = match.groups()
    front = (2000 + int(ya)) * 12 + month_numbers[a] - 1
    back = (2000 + int(yb)) * 12 + month_numbers[b] - 1
    if back <= front:
        raise ValueError(f"Back leg must be later: {instrument}")
    traded = config.PRODUCTS[product]["traded_months"]
    gap = sum((v % 12 + 1) in traded for v in range(front + 1, back + 1))
    return front, back, "spread" if gap == 1 else "jumbo"


def canonical_instrument(value, product):
    """Accept short/legacy full back legs without combining unrelated products."""
    value = str(value).strip().upper().replace(" ", "")
    front, back = value.split("-", 1)
    if back.startswith(product):
        back = back[len(product):]
    result = front + "-" + back
    contract_coordinates(result, product)
    return result


def read_manual_pvd_tables(path, as_of=None, products=None):
    """Read only PVD threshold numbers. Dates/working flags do not affect selection.

    Duplicate instruments use their LAST populated numeric row in sheet order.
    Dates are retained as unparsed source labels, including malformed labels.
    as_of affects the raw-data analysis elsewhere, never this manual lookup.
    """
    try:
        book = pd.ExcelFile(path, engine="openpyxl")
    except PermissionError:
        raise PermissionError(f"Cannot read {path}. Save/close Excel or supply an accessible local workbook copy.") from None
    tables, issues = {}, []
    for kind, sign in (("exhaustion", -1), ("expansion", 1)):
        candidates = [s for s in book.sheet_names if "pvd" in s.lower() and
                      (("exh" in s.lower()) if kind == "exhaustion" else "expan" in s.lower())]
        if len(candidates) != 1:
            raise ValueError(f"Expected one PVD {kind} sheet; found {candidates}. Sheets: {book.sheet_names}")
        sheet = candidates[0]
        data = pd.read_excel(book, sheet_name=sheet)
        data.columns = [str(c).strip().lower().replace(" ", "_") for c in data.columns]
        needed = {"product", "structure", "threshold_zscore"}
        if not needed.issubset(data.columns):
            raise ValueError(f"{sheet}: missing columns {needed - set(data.columns)}")
        if "date" not in data:
            data["date"] = ""
        data = data.dropna(subset=["product", "structure"]).copy()
        data["product"] = data["product"].astype(str).str.strip().str.upper()
        if products is not None:
            data = data.loc[data["product"].isin(products)].copy()
        missing = data.threshold_zscore.isna()
        for _, source in data.loc[missing].iterrows():
            issues.append({"sheet": sheet, "product": source["product"], "instrument": source["structure"],
                           "source_date": str(source["date"]), "issue": "missing numeric threshold"})
        data = data.loc[~missing].copy()
        data["threshold_zscore"] = pd.to_numeric(data["threshold_zscore"], errors="raise")

        if not np.isfinite(data.threshold_zscore).all() or not (sign * data.threshold_zscore > 0).all():
            raise ValueError(f"{sheet}: exhaustion must be negative and expansion positive")
        data["instrument"] = [canonical_instrument(s, p) for s, p in zip(data.structure, data["product"])]
        data = data.drop_duplicates(["product", "instrument"], keep="last")
        data["structure_type"] = [contract_coordinates(s, p)[2] for s, p in zip(data.instrument, data["product"])]
        data["sheet"] = sheet
        tables[kind] = data
    book.close()
    tables["issues"] = pd.DataFrame(issues)
    return tables


def resolve_pvd_threshold(table, instrument, product, structure):
    """Exact match, otherwise mean of TWO closest same-product/same-type rows.

    Distance = absolute front month distance + absolute back month distance.
    Calendar month distance, ties by instrument name; never cross product/type.
    If fewer than two usable neighbors exist, report missing threshold explicitly.
    """
    pool = table.loc[(table["product"] == product) & (table.structure_type == structure)].copy()
    exact = pool.loc[pool.instrument == instrument]
    if len(exact):
        donors, method = exact.iloc[:1], "exact"
    else:
        if len(pool) < 2:
            raise ValueError(f"{product}/{instrument}: need TWO {structure} neighbors in {table.sheet.iloc[0] if len(table) else 'manual sheet'}")
        front, back, _ = contract_coordinates(instrument, product)
        pool["distance_months"] = [abs(contract_coordinates(s, product)[0] - front) +
                                  abs(contract_coordinates(s, product)[1] - back) for s in pool.instrument]
        donors = pool.sort_values(["distance_months", "instrument"]).iloc[:2]
        method = "mean_two_closest_same_product_type"
    return float(donors.threshold_zscore.mean()), {
        "method": method, "donors": "; ".join(donors.instrument),
        "donor_values": "; ".join(map(str, donors.threshold_zscore)),
        "donor_dates": "; ".join(donors["date"].fillna("").astype(str)),
        "sheet": str(donors.iloc[0].sheet),
    }


def load_thresholds(csv_paths=ABS_CSV_PATHS, workbook=MANUAL_XLSX, settings=None):
    """Only instruments in supplied GPT-OSS CSVs; unusable rows become audit errors."""
    cfg = {**SETTINGS, **(settings or {})}
    manual = read_manual_pvd_tables(workbook, cfg["as_of"], products=list(csv_paths))
    if not manual["issues"].empty:
        print(f"Manual workbook: {len(manual['issues'])} blank numeric-threshold rows skipped. Dates ignored. Inspect THRESHOLDS.attrs['manual_issues'].")
    rows = []
    for product, path in csv_paths.items():
        data = pd.read_csv(path, keep_default_na=False)
        for _, source in data.iterrows():
            record = {"product": product, "instrument": str(source.instrument),
                      "structure": source.structure, "abs_source": str(path),
                      "abs_selection_date": source.get("run_date", ""), "status": "error", "error": ""}
            try:
                if source.get("selection_status") != "ok" or source.get("model") != "openai/gpt-oss-120b":
                    raise ValueError("Absorption threshold is missing/not a completed GPT-OSS choice")
                if source.get("analysis_group") != "combined":
                    raise ValueError("This notebook expects combined absorption thresholds")
                instrument = canonical_instrument(source.instrument, product)
                if contract_coordinates(instrument, product)[2] != source.structure:
                    raise ValueError("Structure type does not match contract legs")
                x, y = float(source.z_volume_delta), float(source.z_price_response)
                if not np.isfinite([x, y]).all() or x <= 0:
                    raise ValueError("Invalid absorption X/Y")
                record.update(instrument=instrument, abs_volume_z=x, abs_price_z=y,
                              compression_threshold=cfg["compression_threshold"])
                for kind in ("exhaustion", "expansion"):
                    value, provenance = resolve_pvd_threshold(manual[kind], instrument, product, source.structure)
                    record[f"pvd_{kind}_z"] = value
                    record.update({f"{kind}_{k}": v for k, v in provenance.items()})
                record["status"] = "ok"
            except (ValueError, KeyError, TypeError) as exc:
                record["error"] = str(exc)
            rows.append(record)
    audit = pd.DataFrame(rows)
    if audit.duplicated(["product", "instrument"]).any():
        raise ValueError("Duplicate instrument rows in absorption inputs: choose one combined threshold per instrument")
    audit.attrs["manual_issues"] = manual["issues"]
    return audit


# %% 3. Signal flags and forward movement (raw daily bars only)
FLAG_NAMES = ("compression", "exhaustion_aligned", "exhaustion_opposite",
              "expansion_aligned", "expansion_opposite")


def relative_flags(daily, sign, compression_threshold):
    """PVD exhaustion implies reversal; expansion describes its actual move direction."""
    direction = np.where(daily.pvd_direction == "UP", 1,
                         np.where(daily.pvd_direction == "DOWN", -1, 0))
    exhaustion = daily.pvd_signal.str.contains("EXHAUSTION")
    expansion = daily.pvd_signal.str.contains("EXPANSION")
    return pd.DataFrame({
        "compression": daily.compression_score >= compression_threshold,
        "exhaustion_aligned": exhaustion & (-direction == sign),
        "exhaustion_opposite": exhaustion & (-direction == -sign),
        "expansion_aligned": expansion & (direction == sign),
        "expansion_opposite": expansion & (direction == -sign),
    }, index=daily.index)


def window_movement(raw, origin, end, sign, tick, required, complete):
    """MFE/MAE ignores stops: describes price path, not the absorption trade payoff."""
    if not complete or end <= origin:
        return {"mfe_ticks": np.nan, "mae_ticks": np.nan, "end_move_ticks": np.nan,
                "favorable_move": np.nan, "adverse_move": np.nan, "any_move": np.nan}
    future = raw.iloc[origin + 1:end + 1]
    entry = float(raw.iloc[origin].close)
    favorable = (future.high - entry) if sign == 1 else (entry - future.low)
    adverse = (entry - future.low) if sign == 1 else (future.high - entry)
    mfe, mae = max(0.0, float(favorable.max()) / tick), max(0.0, float(adverse.max()) / tick)
    return {"mfe_ticks": mfe, "mae_ticks": mae,
            "end_move_ticks": sign * (float(raw.iloc[end].close) - entry) / tick,
            "favorable_move": bool(mfe > 0 and mfe >= required),
            "adverse_move": bool(mae > 0 and mae >= required),
            "any_move": bool(max(mfe, mae) > 0 and max(mfe, mae) >= required)}


def make_event_context(raw, daily, events, threshold, cfg, abs_params, tick):
    """Entry-time patterns and later patterns are recorded in separate columns.

    AFTER uses the full planned H-bar window even after the trade stops, for
    market-path research. Landmark rows explicitly flag if the original stop
    already occurred, and report continuation metrics separately for active trades.
    """
    records, landmarks, occurrences = [], [], []
    horizon, before = cfg["forward_bars"], cfg["before_bars"]
    required = abs_params["required_move_ticks"]
    for _, event in events.iterrows():
        start, end = int(event.window_start_index), int(event.signal_index)
        sign = 1 if event.absorption_direction == "SELL" else -1
        flags = relative_flags(daily, sign, cfg["compression_threshold"])
        event_id = f"{threshold['instrument']}:{pd.Timestamp(event.Date).strftime('%Y-%m-%d')}:{event.absorption_direction}:{int(event.candle_length)}d"
        last = min(end + horizon, len(raw) - 1)
        complete = end + horizon < len(raw)
        row = {**event.to_dict(), "event_id": event_id, "product": threshold["product"],
               "instrument": threshold["instrument"], "structure": threshold["structure"],
               "full_forward_window": complete, "observed_forward_bars": last - end,
               "direction_sign": sign}
        row.update({f"forward_{k}": v for k, v in window_movement(raw, end, last, sign, tick, required, complete).items()})
        signatures = {}
        ranges = {"before": (max(0, start - before), start - 1), "during": (start, end),
                  "after": (end + 1, last)}
        for phase, (left, right) in ranges.items():
            indexes = list(range(left, right + 1))
            observed = daily.iloc[indexes]
            ready = (len(indexes) > 0 and observed[["compression_score", "pvd_z"]].notna().all().all())
            if phase == "before":
                ready = ready and len(indexes) == before
            if phase == "after":
                ready = ready and complete
            row[f"{phase}_ready"] = bool(ready)
            tokens = []
            for name in FLAG_NAMES:
                hits = [j for j in indexes if bool(flags.at[j, name])]
                row[f"{phase}_{name}"] = bool(hits) if ready else pd.NA
                if hits:
                    tokens.append(name)
                    for j in hits:
                        occurrences.append({"event_id": event_id, "instrument": threshold["instrument"],
                                            "phase": phase, "signal_type": name, "signal_index": j,
                                            "signal_date": raw.at[j, "Date"], "offset_from_abs_end": j - end})
                if phase == "after" and hits:
                    j = hits[0]  # one first landmark per event/type, not every repeated day
                    stop_before = bool(event.stop_hit and float(event.exit_index) <= j)
                    profit_before = bool(event.profit_hit_first and float(event.profit_bar_offset) <= j - end)
                    landmark = {"event_id": event_id, "product": threshold["product"],
                                "instrument": threshold["instrument"], "structure": threshold["structure"],
                                "absorption_direction": event.absorption_direction,
                                "signal_type": name, "signal_index": j, "signal_date": raw.at[j, "Date"],
                                "delay_bars": j - end, "remaining_bars": end + horizon - j,
                                "stop_already_hit": stop_before, "profit_already_hit": profit_before,
                                "move_at_signal_ticks": sign * (float(raw.at[j, "close"]) - float(event.entry_price)) / tick,
                                "full_remaining_window": complete, "original_outcome": event.outcome}
                    landmark.update(window_movement(raw, j, last, sign, tick, required, complete))
                    # Original stop/entry retained for remaining trade opportunity.
                    # Entry at marker close is NOT simulated as a new trade here.
                    best, stopped, ambiguous = 0.0, stop_before, False
                    if not stopped:
                        for k in range(j + 1, last + 1):
                            stop_hit = raw.at[k, "high"] >= event.stop_loss if sign == -1 else raw.at[k, "low"] <= event.stop_loss
                            favorable = ((float(raw.at[k, "high"]) - float(raw.at[j, "close"])) if sign == 1 else
                                         (float(raw.at[j, "close"]) - float(raw.at[k, "low"]))) / tick
                            if stop_hit:
                                stopped, ambiguous = True, favorable > best
                                break
                            best = max(best, favorable)
                    landmark["remaining_active_trade_mfe_ticks"] = best if not stop_before and (complete or stopped) else np.nan
                    landmark["remaining_stop_bar_order_unknown"] = ambiguous
                    landmarks.append(landmark)
            signatures[phase] = "+".join(tokens) if ready and tokens else ("none" if ready else "unknown")
            sequence, previous_state = [], None
            for j in indexes:
                state = str(daily.at[j, "pvd_signal"]) if pd.notna(daily.at[j, "pvd_z"]) else "UNKNOWN"
                if state != "NORMAL" and state != previous_state:
                    sequence.append(state)
                previous_state = state  # Normal gaps separate repeated occurrences.
            row[f"{phase}_pvd_sequence"] = " > ".join(sequence) or "none"
        row["entry_signature"] = f"before:{signatures['before']} / during:{signatures['during']}"
        row["post_signature"] = f"after:{signatures['after']}"
        row["joint_signature"] = row["entry_signature"] + " / " + row["post_signature"]
        row["during_has_both_exhaustion_and_expansion"] = bool(
            signatures["during"] != "unknown" and "exhaustion" in signatures["during"] and "expansion" in signatures["during"])
        row["after_has_both_expansion_directions"] = bool(
            complete and row["after_ready"] and bool(row["after_expansion_aligned"]) and bool(row["after_expansion_opposite"]))
        records.append(row)
    return pd.DataFrame(records), pd.DataFrame(landmarks), pd.DataFrame(occurrences)


# %% 4. One-instrument analysis; no notebook/API required for this pure function
def analyze_combined_instrument(raw_1d, threshold, settings=None, abs_params=None, pvd_params=None,
                                analysis_start=None, analysis_end=None):
    """Use saved absorption gate, daily compression and daily PVD on identical bars."""
    cfg = {**SETTINGS, **(settings or {})}
    ap = {**ABS_PARAMS, **(abs_params or {}), "forward_days": cfg["forward_bars"]}
    pp = {**PVD_PARAMS, **(pvd_params or {})}
    raw = ep._raw_bars(raw_1d)
    tick = config.PRODUCTS[threshold["product"]]["tick_size"]
    comp = config.COMPRESSION
    daily, _ = calculate_compression(raw, range_baseline=comp["range_baseline"],
        atr_period=comp["atr_period"], rv_period=comp["rv_period"],
        persistence_window=comp["persistence_window"], compression_threshold=cfg["compression_threshold"],
        label_atr_period=config.COMPRESSION_ATR["label_atr_period"])
    daily = calculate_pvd(daily, price_horizon_h=pp["price_horizon_h"],
        volatility_window=pp["volatility_window"], pvd_model_window=pp["pvd_model_window"],
        atr_period=pp["atr_period"], exhaustion_z_threshold=threshold["pvd_exhaustion_z"],
        expansion_z_threshold=threshold["pvd_expansion_z"])
    absorption = ep.analyze_expected_profit(raw, tick, cfg["candle_lengths"], "combined",
        [threshold["abs_volume_z"]], [threshold["abs_price_z"]], ap, analysis_start, analysis_end)
    key = ("combined", float(threshold["abs_volume_z"]), float(threshold["abs_price_z"]))
    events, landmarks, occurrences = make_event_context(raw, daily, absorption["event_ledgers"][key],
                                                       threshold, cfg, ap, tick)
    valid = absorption["features"].loc[absorption["features"][["z_delta", "z_price", "atr"]].notna().all(axis=1)]
    exposure = 0.0
    if not valid.empty:
        first, final = valid.Date.min().tz_localize(None).normalize(), raw.Date.max().tz_localize(None).normalize()
        if analysis_end is not None:
            final = min(final, pd.Timestamp(analysis_end).tz_localize(None).normalize())
        for month in pd.period_range(first, final, freq="M"):
            a, b = max(first, month.start_time.normalize()), min(final, month.end_time.normalize())
            exposure += ((b - a).days + 1) / month.days_in_month
    if not events.empty:
        events["exposure_months"] = exposure
    return {"raw": raw, "daily": daily, "events": events, "landmarks": landmarks,
            "occurrences": occurrences, "suppressed": absorption["suppressed_ledgers"][key],
            "threshold": dict(threshold), "tick_size": tick, "abs_params": ap, "pvd_params": pp, "settings": cfg,
            "absorption_metadata": absorption["metadata"]}


# %% 5. Frequency distributions and descriptive conditional probabilities
def event_statistics(events, total_events, baseline=None, min_n=4):
    """Every probability states its denominator; unresolved is not counted as success."""
    n = len(events)
    resolved = events.loc[events.outcome.isin(["SUCCESS", "FAILURE"])]
    successes = events.loc[events.outcome == "SUCCESS"]
    failures = events.loc[events.outcome == "FAILURE"]
    complete = events.loc[events.full_forward_window]
    valid_move = complete.loc[complete.forward_favorable_move.notna()]
    known_profit = events.loc[events.profit_hit_first]
    result = {"n_events": n, "pattern_probability": n / total_events if total_events else np.nan,
              "n_resolved": len(resolved), "n_success": len(successes), "n_failure": len(failures),
              "n_unresolved": int((events.outcome == "UNRESOLVED").sum()),
              "n_censored": int((events.outcome == "CENSORED").sum()),
              "n_full_forward": len(valid_move), "n_stop_bar_ambiguous": int(events.stop_bar_order_unknown.sum()),
              "p_success_given_resolved": len(successes) / len(resolved) if len(resolved) else np.nan,
              "expected_opportunity_ticks": resolved.research_payoff_ticks.mean(),
              "avg_success_ticks": successes.success_size_ticks.mean(),
              "avg_failure_ticks": failures.failure_size_ticks.mean(),
              "avg_profit_delay_bars_known": known_profit.profit_bar_offset.mean(),
              "p_favorable_move_full_window": valid_move.forward_favorable_move.astype(float).mean(),
              "p_adverse_move_full_window": valid_move.forward_adverse_move.astype(float).mean(),
              "p_any_move_full_window": valid_move.forward_any_move.astype(float).mean(),
              "mean_mfe_full_window_ticks": valid_move.forward_mfe_ticks.mean(),
              "mean_mae_full_window_ticks": valid_move.forward_mae_ticks.mean(),
              "mean_end_move_full_window_ticks": valid_move.forward_end_move_ticks.mean(),
              "low_n": len(resolved) < min_n}
    exposure = (baseline["exposure_months"] if baseline is not None else
                (events.drop_duplicates("instrument").exposure_months.sum() if "exposure_months" in events else np.nan))
    result["exposure_months"] = exposure
    result["signals_per_instrument_month"] = n / exposure if exposure > 0 else np.nan
    if baseline is not None:
        result["delta_expected_ticks_vs_abs_alone"] = result["expected_opportunity_ticks"] - baseline["expected_opportunity_ticks"]
        result["delta_success_rate_vs_abs_alone"] = result["p_success_given_resolved"] - baseline["p_success_given_resolved"]
    return result


def summarize_scope(events, cfg):
    """Predefined overlapping groups + mutually exclusive observed signatures."""
    if events.empty:
        return {"baseline": pd.DataFrame(), "patterns": pd.DataFrame(), "entry_distribution": pd.DataFrame(),
                "joint_distribution": pd.DataFrame(), "sequences": pd.DataFrame(), "by_duration": pd.DataFrame(),
                "by_absorption_side": pd.DataFrame()}
    total = len(events)
    baseline = event_statistics(events, total, min_n=cfg["min_pattern_events"])
    rows = []
    for phase in ("before", "during", "after"):
        for name in FLAG_NAMES:
            mask = events[f"{phase}_{name}"].eq(True).fillna(False)
            sub = events.loc[mask]
            rows.append({"pattern": f"{phase}: {name}", "timing": "entry_known" if phase != "after" else "after_only_research",
                         **event_statistics(sub, total, baseline, cfg["min_pattern_events"]),
                         "n_without_pattern": int((events[f"{phase}_{name}"].eq(False).fillna(False)).sum()),
                         "n_unknown_pattern": int(events[f"{phase}_{name}"].isna().sum()),
                         "expected_without_pattern_ticks": events.loc[events[f"{phase}_{name}"].eq(False).fillna(False) &
                                                                      events.outcome.isin(["SUCCESS", "FAILURE"]), "research_payoff_ticks"].mean()})
    for name, mask in {
        "prior aligned expansion THEN during aligned exhaustion": events.before_expansion_aligned.eq(True) & events.during_exhaustion_aligned.eq(True),
        "during compression AND aligned exhaustion": events.during_compression.eq(True) & events.during_exhaustion_aligned.eq(True),
        "during aligned exhaustion THEN after aligned expansion": events.during_exhaustion_aligned.eq(True) & events.after_expansion_aligned.eq(True),
        "after BOTH aligned and opposite expansion": events.after_has_both_expansion_directions,
        "during BOTH exhaustion and expansion": events.during_has_both_exhaustion_and_expansion,
    }.items():
        rows.append({"pattern": name, "timing": "after_only_research" if "after" in name else "entry_known",
                     **event_statistics(events.loc[mask.fillna(False)], total, baseline, cfg["min_pattern_events"])})

    def distribution(column):
        # Exact signatures partition events: frequencies sum to one, unlike marginal groups.
        return pd.DataFrame([{"pattern": signature, **event_statistics(group, total, baseline, cfg["min_pattern_events"])}
                             for signature, group in events.groupby(column, dropna=False)]).sort_values("n_events", ascending=False)

    sequences = events.groupby(["before_pvd_sequence", "during_pvd_sequence", "after_pvd_sequence"], dropna=False)
    sequence_table = pd.DataFrame([{ "before_pvd": a, "during_pvd": b, "after_pvd": c,
        **event_statistics(group, total, baseline, cfg["min_pattern_events"])} for (a, b, c), group in sequences])
    return {"baseline": pd.DataFrame([baseline]), "patterns": pd.DataFrame(rows),
            "by_duration": pd.DataFrame([{"pattern": f"{int(length)}D absorption",
                **event_statistics(group, total, baseline, cfg["min_pattern_events"])}
                for length, group in events.groupby("candle_length")]),
            "by_absorption_side": pd.DataFrame([{"pattern": f"{side} absorption",
                **event_statistics(group, total, baseline, cfg["min_pattern_events"])}
                for side, group in events.groupby("absorption_direction")]),
            "entry_distribution": distribution("entry_signature"),
            "joint_distribution": distribution("joint_signature"),
            "sequences": sequence_table.sort_values("n_events", ascending=False)}


def summarize_landmarks(landmarks, min_n=4):
    """Condition on a later signal and measure ONLY movement after its close.

    The original H-bar deadline stays fixed. Multiple signal categories can share
    an absorption event, so these row counts must not be summed into trade counts.
    """
    rows = []
    if landmarks.empty:
        return pd.DataFrame()
    for signal_type, group in landmarks.groupby("signal_type"):
        valid = group.loc[group.full_remaining_window & group.favorable_move.notna()]
        active = valid.loc[~valid.stop_already_hit]
        rows.append({"signal_type": signal_type, "n_first_markers": len(group),
                     "n_complete_remaining": len(valid), "n_original_trade_active": len(active),
                     "n_stop_already_hit": int(group.stop_already_hit.sum()),
                     "n_profit_already_hit": int(group.profit_already_hit.sum()),
                     "mean_delay_bars": group.delay_bars.mean(), "mean_remaining_bars": group.remaining_bars.mean(),
                     "mean_move_already_at_marker_ticks": group.move_at_signal_ticks.mean(),
                     "p_remaining_favorable_full_window": valid.favorable_move.astype(float).mean(),
                     "mean_remaining_mfe_ticks": valid.mfe_ticks.mean(),
                     "mean_remaining_mae_ticks": valid.mae_ticks.mean(),
                     "p_remaining_favorable_original_trade_active": active.favorable_move.astype(float).mean(),
                     "mean_active_pre_stop_remaining_mfe_ticks": active.remaining_active_trade_mfe_ticks.mean(),
                     "low_n": len(valid) < min_n})
    return pd.DataFrame(rows)


# %% 6. Runner: exact CSV universe, auditable outputs, refresh/cache option
def run_combined_signal_analysis(settings=None, csv_paths=None, workbook=None,
                                 abs_params=None, pvd_params=None, raw_by_instrument=None, threshold_audit=None):
    """Analyze only input CSV instruments; raw_by_instrument enables offline tests.

    Current/manual thresholds applied to earlier history make this descriptive
    in-sample research; no out-of-sample trading validation is claimed.
    """
    cfg = {**SETTINGS, **(settings or {})}
    for name in ("before_bars", "forward_bars", "min_pattern_events"):
        if isinstance(cfg[name], bool) or not isinstance(cfg[name], int) or cfg[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if cfg["data_mode"] not in ("auto", "refresh", "cache"):
        raise ValueError("data_mode must be auto, refresh or cache")
    ap = {**ABS_PARAMS, **(abs_params or {}), "months_back": cfg["months_back"], "forward_days": cfg["forward_bars"]}
    pp = {**PVD_PARAMS, **(pvd_params or {})}
    # Pad fetch history for PVD's volatility/model/residual warmup and PRE context.
    calendar_pad = ap["warmup_extra_calendar_days"]
    extra = pp["volatility_window"] + 2 * pp["pvd_model_window"] + pp["price_horizon_h"] + cfg["before_bars"]
    ap["warmup_extra_calendar_days"] = max(ap["warmup_extra_calendar_days"], int(np.ceil(extra * 7 / 5)))
    start, end = ep.analysis_date_range(cfg["as_of"], ap)
    thresholds = (load_thresholds(csv_paths or ABS_CSV_PATHS, workbook or MANUAL_XLSX, cfg)
                  if threshold_audit is None else threshold_audit.copy())
    manual_issues = thresholds.attrs.get("manual_issues", pd.DataFrame())
    outputs, coverage_rows, errors = {}, [], []
    authentication_error = None

    def fetch_missing(**kwargs):
        """Use current config token; avoid repeating known auth failure for every contract."""
        nonlocal authentication_error
        if authentication_error is not None:
            raise ohlc.OHLCFetchError("Earlier OHLC authentication failure in this run; missing range not fetched", authentication_error)
        try:
            return ohlc.getData_OHLC_v2(**kwargs, token=config.API_TOKEN, raise_on_error=True)
        except ohlc.OHLCFetchError as exc:
            if exc.status_code in (401, 403):
                authentication_error = exc.status_code
            raise
    for _, threshold in thresholds.iterrows():
        instrument, product = threshold.instrument, threshold["product"]
        if threshold.status != "ok":
            errors.append({"instrument": instrument, "stage": "threshold", "error": threshold.error})
            print(f"SKIP {instrument}: {threshold.error}")
            continue
        print(f"Analyzing {instrument}...", flush=True)
        try:
            if raw_by_instrument is not None:
                raw = ep._raw_bars(raw_by_instrument[instrument])
                raw = raw.loc[raw.Date <= end].reset_index(drop=True)
                coverage = {"coverage_basis": "supplied_raw_data; not API verified"}
            elif cfg["data_mode"] == "auto":
                warmup = ap["baseline_window"] + 2 * max(cfg["candle_lengths"]) + max(
                    ap["volume_window"], ap["atr_window"], ap["atr_period"])
                warmup = max(warmup, extra + max(cfg["candle_lengths"]),
                             config.COMPRESSION["range_baseline"] + max(config.COMPRESSION["atr_period"],
                             config.COMPRESSION["rv_period"]) + cfg["before_bars"] + max(cfg["candle_lengths"]))
                fetch_start = start - pd.offsets.BDay(warmup) - pd.Timedelta(days=calendar_pad)
                raw, coverage = ohlc.fetch_ohlc_1d_incremental(
                    instrument, product, fetch_start, end, fetcher=fetch_missing,
                    allow_cached_on_fetch_error=cfg["allow_cached_on_fetch_error"])
                coverage.update(analysis_start=start, analysis_end=end,
                                warmup_bars_available=int((raw.Date < start).sum()),
                                warmup_bars_requested=warmup,
                                warmup_sufficient=int((raw.Date < start).sum()) >= warmup)
                if coverage["coverage_warning"]:
                    print(f"  COVERAGE WARNING: {coverage['coverage_warning']}", flush=True)
            elif cfg["data_mode"] == "refresh":
                raw, coverage = ep.refresh_expected_profit_data(instrument, product, cfg["candle_lengths"], ap, cfg["as_of"])
            else:
                path = config.DATA_DIR / product.lower() / f"{instrument}_1d.csv"
                raw = ep._raw_bars(pd.read_csv(path))
                raw = raw.loc[raw.Date <= end].reset_index(drop=True)
                coverage = {"cache_path": str(path), "coverage_basis": "cached data; refresh disabled; completeness unverified"}
            if raw.empty or not raw.Date.between(start, end).any():
                raise ValueError("No raw bars in requested analysis period")
            result = analyze_combined_instrument(raw, threshold, cfg, ap, pp, start, end)
            result["coverage"] = coverage
            outputs[instrument] = result
            coverage_rows.append({"instrument": instrument, "product": product,
                                  "raw_start": raw.Date.min(), "raw_end": raw.Date.max(),
                                  "analysis_bars": int(raw.Date.between(start, end).sum()),
                                  "abs_events": len(result["events"]), **coverage})
            print(f"  {len(result['events'])} unique absorption events; {len(result['landmarks'])} first after-signal markers", flush=True)
        except (ValueError, KeyError, OSError, RuntimeError) as exc:
            errors.append({"instrument": instrument, "stage": "analysis", "error": str(exc)})
            print(f"SKIP {instrument}: {exc}", flush=True)

    def combine(name):
        frames = [r[name] for r in outputs.values() if not r[name].empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    events, landmarks, occurrences = combine("events"), combine("landmarks"), combine("occurrences")
    reports = {}
    if not events.empty:
        for instrument, group in events.groupby("instrument"):
            reports[instrument] = summarize_scope(group, cfg)
        # Optional pooled view is explicitly labelled; per-instrument results stay primary.
        for product, group in events.groupby("product"):
            reports[f"POOLED_{product}"] = summarize_scope(group, cfg)
    day = pd.Timestamp.now(tz="Asia/Calcutta") if cfg["as_of"] is None else pd.Timestamp(cfg["as_of"])
    folder = Path(cfg["output_dir"]) / day.strftime("%Y-%m-%d")
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in {"threshold_audit": thresholds, "coverage": pd.DataFrame(coverage_rows),
                        "manual_input_issues": manual_issues,
                        "errors": pd.DataFrame(errors), "events": events, "after_signal_landmarks": landmarks,
                        "signal_occurrences": occurrences}.items():
        frame.to_csv(folder / f"{name}.csv", index=False, encoding="utf-8-sig")
    for scope, report in reports.items():
        for name, table in report.items():
            table.to_csv(folder / f"{scope}_{name}.csv", index=False, encoding="utf-8-sig")
        subset = landmarks.loc[landmarks.instrument == scope] if not landmarks.empty and not scope.startswith("POOLED_") else (
            landmarks.loc[landmarks["product"] == scope[7:]] if not landmarks.empty else landmarks)
        report["remaining_move"] = summarize_landmarks(subset, cfg["min_pattern_events"])
        report["remaining_move"].to_csv(folder / f"{scope}_remaining_move.csv", index=False, encoding="utf-8-sig")
    for instrument, result in outputs.items():
        result["daily"].to_csv(folder / f"{instrument}_daily_signals.csv", index=False, encoding="utf-8-sig")
        result["suppressed"].to_csv(folder / f"{instrument}_suppressed_absorption.csv", index=False, encoding="utf-8-sig")
    metadata = {"settings": cfg, "abs_params": ap, "pvd_params": pp, "analysis_start": start,
                "analysis_end": end, "threshold_sources": "provided_threshold_audit" if threshold_audit is not None else (csv_paths or ABS_CSV_PATHS),
                "manual_workbook": workbook or MANUAL_XLSX,
                "counting": "retrospective longest-first absorption window suppression; separate events may share forward bars",
                "interpretation": "descriptive in-sample associations; after signals are unavailable at entry; hindsight maxima are not realized P&L"}
    (folder / "metadata.json").write_text(json.dumps(metadata, default=str, indent=2), encoding="utf-8")
    return {"instruments": outputs, "events": events, "landmarks": landmarks, "occurrences": occurrences,
            "threshold_audit": thresholds, "coverage": pd.DataFrame(coverage_rows), "errors": pd.DataFrame(errors),
            "reports": reports, "settings": cfg, "output_dir": folder}


# %% 7. Labelled candlestick examples
def plot_event(result, event_id, context_bars=7, figsize=(17, 11)):
    """Plot one event with chronological signals; post-stop candles are market context."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    event = result["events"].loc[result["events"].event_id == event_id].iloc[0]
    raw, daily = result["raw"], result["daily"]
    start, end = int(event.window_start_index), int(event.signal_index)
    horizon = result["settings"]["forward_bars"]
    left, right = max(0, start - context_bars), min(len(raw) - 1, end + horizon + 2)
    positions = np.arange(left, right + 1)
    bars = raw.iloc[left:right + 1]
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True,
                             gridspec_kw={"height_ratios": [4, 1.2, 1.5, 1.2]})
    ax = axes[0]
    for j in positions:
        bar = raw.iloc[j]
        color = "#18864b" if bar.close >= bar.open else "#c53c3c"
        ax.vlines(j, bar.low, bar.high, color=color, linewidth=1)
        body = max(abs(bar.close - bar.open), result["tick_size"] * 0.04)
        ax.add_patch(Rectangle((j - 0.32, min(bar.open, bar.close)), 0.64, body, color=color, alpha=.85))
    ax.axvspan(start - .5, end + .5, color="#527ed1", alpha=.18, label="Absorption detection window")
    ax.axvspan(end + .5, min(end + horizon, right) + .5, color="#b68bce", alpha=.08, label="Forward research window")
    ax.axhline(event.entry_price, color="black", linestyle=":", label="Signal-close entry reference")
    ax.axhline(event.stop_loss, color="#c53c3c", linestyle="--", label="Original configured ATR stop")
    target = event.entry_price + event.direction_sign * result["abs_params"]["required_move_ticks"] * result["tick_size"]
    ax.axhline(target, color="#18864b", linestyle="--", label="Required profit opportunity")
    ax.annotate(f"{int(event.candle_length)}D {event.absorption_direction} absorption\n{event.position} hypothesis",
                (end, event.entry_price), xytext=(10, 28), textcoords="offset points",
                arrowprops={"arrowstyle": "->"}, fontsize=9)
    padding = max(float(bars.high.max() - bars.low.min()) * .04, result["tick_size"])
    compressed = daily.iloc[positions].compression_score >= result["settings"]["compression_threshold"]
    ax.scatter(positions[compressed], bars.low.to_numpy()[compressed] - padding, marker="s", s=35, color="#3575b9", label="Compression >= 0.30")
    styles = {"UP_TREND_EXHAUSTION": ("v", "#d18120"), "DOWN_TREND_EXHAUSTION": ("^", "#d18120"),
              "UP_PRICE_EXPANSION": ("^", "#8723a6"), "DOWN_PRICE_EXPANSION": ("v", "#8723a6")}
    for name, (marker, color) in styles.items():
        mask = daily.iloc[positions].pvd_signal.eq(name).to_numpy()
        y = bars.high.to_numpy()[mask] + padding if name.startswith("UP") else bars.low.to_numpy()[mask] - 2 * padding
        ax.scatter(positions[mask], y, marker=marker, color=color, s=65, label=name)
    if event.profit_hit_first:
        profit_j = end + int(event.profit_bar_offset)
        ax.axvline(profit_j, color="#18864b", linestyle=":", label="First required-profit bar")
    if event.stop_hit:
        ax.axvline(int(event.exit_index), color="#c53c3c", linestyle=":", label="First stop bar; later bars excluded from trade size")
    legends = result["landmarks"]
    if not legends.empty:
        markers = legends.loc[legends.event_id == event_id]
        for j in markers.signal_index.drop_duplicates():
            ax.scatter(int(j), raw.at[int(j), "close"], marker="D", color="black", s=35)
        ax.add_line(Line2D([], [], color="black", marker="D", linestyle="", label="First after-signal close: remaining move starts NEXT bar"))
    ax.set_title(f"{event.instrument} | {event.Date.strftime('%Y-%m-%d')} | {event.outcome} | "
                 f"research opportunity={event.research_payoff_ticks:.2f} ticks\n"
                 "Arrows: price-direction signal names. Later confirmations are not available at the absorption entry.", fontsize=11)
    ax.set_ylabel("Price")
    ax.legend(loc="upper left", fontsize=7, ncol=3)
    axes[1].plot(positions, daily.iloc[positions].compression_score, color="#3575b9")
    axes[1].axhline(result["settings"]["compression_threshold"], color="black", linestyle="--")
    axes[1].set_ylabel("Compression\nscore / 0.30 gate")
    axes[1].set_ylim(-.05, 1.05)
    axes[2].plot(positions, daily.iloc[positions].pvd_z, color="#8723a6")
    axes[2].axhline(result["threshold"]["pvd_exhaustion_z"], color="#d18120", linestyle="--", label="Manual exhaustion cutoff")
    axes[2].axhline(result["threshold"]["pvd_expansion_z"], color="#8723a6", linestyle="--", label="Manual expansion cutoff")
    axes[2].axhline(0, color="grey", linewidth=.6)
    axes[2].set_ylabel("Signed PVD z")
    axes[2].legend(fontsize=8, loc="upper left")
    delta = bars.buyvolume - bars.sellvolume
    axes[3].bar(positions, delta, color=np.where(delta >= 0, "#18864b", "#c53c3c"), alpha=.8)
    axes[3].axhline(0, color="grey", linewidth=.6)
    axes[3].set_ylabel("Raw daily\nbuy - sell volume")
    for panel in axes:
        panel.axvline(end, color="black", linewidth=.8, alpha=.5)
        panel.grid(alpha=.18)
    ticks = positions[::max(1, len(positions) // 10)]
    axes[3].set_xticks(ticks, [raw.at[j, "Date"].strftime("%d-%b") for j in ticks], rotation=30)
    axes[3].set_xlabel("Raw trading bars (calendar gaps compressed). Absorption shaded window ends at black line.")
    fig.tight_layout()
    return fig


def plot_pattern_examples(run, instrument=None, max_examples=6):
    """Different categories/outcomes selected by date, not cherry-picked maximum gains."""
    import matplotlib.pyplot as plt
    if run["events"].empty:
        print("No absorption events: no candlestick examples available.")
        return []
    events = run["events"].sort_values("Date")
    if instrument is not None:
        events = events.loc[events.instrument == instrument]
    categories = [
        ("Failure example: inspect adverse sequence", events.outcome.eq("FAILURE")),
        ("Multi-day absorption detection window", events.candle_length.ge(2)),
        ("Absorption + compression", events.during_compression.eq(True)),
        ("Aligned exhaustion + absorption", events.during_exhaustion_aligned.eq(True)),
        ("Prior expansion then absorption", events.before_expansion_aligned.eq(True)),
        ("Aligned expansion AFTER absorption", events.after_expansion_aligned.eq(True)),
        ("Opposite expansion / possible adverse path", events.after_expansion_opposite.eq(True)),
        ("Both expansion directions / choppy path", events.after_has_both_expansion_directions.eq(True)),
    ]
    selected, used = [], set()
    for label, mask in categories:
        candidates = events.loc[mask.fillna(False) & ~events.event_id.isin(used)]
        if not candidates.empty and len(selected) < max_examples:
            row = candidates.iloc[0]
            selected.append((label, row))
            used.add(row.event_id)
    # Include a failure/unresolved example where available, not just successes.
    for outcome in ("FAILURE", "UNRESOLVED", "SUCCESS", "CENSORED"):
        candidates = events.loc[events.outcome.eq(outcome) & ~events.event_id.isin(used)]
        if len(selected) < max_examples and not candidates.empty:
            row = candidates.iloc[0]
            selected.append((f"{outcome} reference example", row))
            used.add(row.event_id)
    for label, event in selected:
        print(f"EXAMPLE: {label}\n{event.event_id}\n{event.entry_signature}\n{event.post_signature}")
        figure = plot_event(run["instruments"][event.instrument], event.event_id)
        plt.show()
        plt.close(figure)
    return [row.event_id for _, row in selected]


# %% 8. Readable cell-output helpers and definitions
def explain(title, text):
    """Notebook Markdown where available; plain text fallback for scripts."""
    try:
        from IPython.display import Markdown, display
        display(Markdown(f"### {title}\n\n{text}"))
    except ImportError:
        print(f"\n{title}\n{text}\n")


def show_table(table, columns=None, limit=30):
    if table.empty:
        print("No observations available for this table.")
        return
    view = table if columns is None else table[[c for c in columns if c in table.columns]]
    try:
        from IPython.display import display
        display(view.head(limit).style.format(precision=3, na_rep="n/a"))
    except ImportError:
        print(view.head(limit).to_string(index=False))
    if len(view) > limit:
        print(f"Showing {limit}/{len(view)} rows. Full table is in RUN['reports'] and saved CSV.")


KEY_COLUMNS = ["pattern", "timing", "n_events", "pattern_probability", "n_resolved", "n_full_forward",
               "signals_per_instrument_month",
               "p_success_given_resolved", "expected_opportunity_ticks", "delta_expected_ticks_vs_abs_alone",
               "p_favorable_move_full_window", "p_adverse_move_full_window", "p_any_move_full_window",
               "avg_success_ticks", "avg_failure_ticks", "low_n"]


# %% 9. Threshold audit: check exact/fallback sources BEFORE running analysis
if __name__ == "__main__":
    explain("1. Threshold sources", "Sirf supplied GPT-OSS CSV instruments. Compression=0.30. "
            "PVD exact Excel match, warna same product/type ke two closest front/back month distances ka average. "
            "Manual dates/working columns ignore hote hain; duplicate instruments ki last filled numeric-threshold row use hoti hai. "
            "Error rows skip hongi; absorption thresholds fabricate nahi hote.")
    THRESHOLDS = load_thresholds()
    show_table(THRESHOLDS, ["product", "instrument", "structure", "status", "abs_volume_z", "abs_price_z",
                           "pvd_exhaustion_z", "exhaustion_method", "exhaustion_donors",
                           "pvd_expansion_z", "expansion_method", "expansion_donors", "error"], limit=100)
    show_table(THRESHOLDS.attrs["manual_issues"], limit=15)

# %% 10. Run once: later display/plot cells reuse RUN, so they do not refetch data
if __name__ == "__main__":
    explain("2. Analysis run", "before=5 trading bars before absorption START; during=its 1/2/3/5D window; "
            "after=10 trading bars after its END. Compression/PVD computed on raw 1D series. "
            "Absorption counting uses existing retrospective longest-window preference. "
            "Auto data mode uses saved OHLC first, fetching only missing ranges. Check actual_end / target_date_reached / coverage_warning. "
            "Historical thresholds are in-sample inputs; research payoff is maximum pre-stop opportunity, not realized profit.")
    RUN = run_combined_signal_analysis(threshold_audit=THRESHOLDS)
    print("Saved outputs:", RUN["output_dir"])
    show_table(RUN["coverage"], limit=100)
    show_table(RUN["errors"], limit=100)

# %% 11. Per-instrument baseline and overlapping pattern associations
if __name__ == "__main__":
    for SCOPE, REPORT in RUN["reports"].items():
        if SCOPE.startswith("POOLED_"):
            continue                         # primary view: every instrument separately
        explain(f"3. {SCOPE}: absorption alone", "Ye baseline hai. n_resolved=SUCCESS+FAILURE. "
                "UNRESOLVED/CENSORED expected opportunity se excluded, counts retain hote hain.")
        show_table(REPORT["baseline"])
        explain(f"3a. {SCOPE}: absorption duration and side", "Ye same deduplicated events ka breakdown hai: "
                "1D/2D/3D/5D aur BUY/SELL. Extra trades add nahi ho rahe. Forward windows overlap kar sakti hain; "
                "frequencies descriptive hain, independent samples ya causation ka proof nahi.")
        show_table(REPORT["by_duration"], KEY_COLUMNS)
        show_table(REPORT["by_absorption_side"], KEY_COLUMNS)
        explain(f"4. {SCOPE}: overlapping patterns", "pattern_probability=N pattern / ALL selected absorption events. "
                "Rows overlap; probabilities ko add mat karo. Entry-known groups entry par available hain. "
                "After groups sirf later association batati hain. Delta ticks absorption-alone ke comparison mein hai. "
                "Forward movement probabilities full 10-bar paths par hain, stop ke baad bhi; trade success alag metric hai. "
                "Aligned exhaustion ka reversal absorption direction se match karta hai; aligned expansion ka actual move match karta hai.")
        show_table(REPORT["patterns"], KEY_COLUMNS, limit=40)

# %% 12. Observed distributions and sequences: unplanned combinations included
if __name__ == "__main__":
    for SCOPE, REPORT in RUN["reports"].items():
        if SCOPE.startswith("POOLED_"):
            continue
        explain(f"5. {SCOPE}: entry pattern probability distribution", "Exact signatures mutually exclusive hain: "
                "pattern probabilities sum to 1. Unknown ka matlab full before/during indicator history nahi mili. "
                "Rare new combinations bhi retain hote hain; low_n=True ko strong conclusion mat samajhna.")
        show_table(REPORT["entry_distribution"], KEY_COLUMNS, limit=20)
        explain(f"6. {SCOPE}: joint before/during/after patterns", "Ye after-window information include karti hain, "
                "isliye live entry filter nahi hain. Choppy opposite expansions aur exhaustion-to-expansion patterns inspect karo.")
        show_table(REPORT["joint_distribution"], KEY_COLUMNS, limit=20)
        explain(f"7. {SCOPE}: observed PVD sequences", "Before/during/after order retain hai. "
                "Repeated consecutive PVD signal names collapse hote hain; full occurrence dates separate ledger mein hain.")
        show_table(REPORT["sequences"], ["before_pvd", "during_pvd", "after_pvd", *KEY_COLUMNS[2:]], limit=20)

# %% 13. After-signal remaining opportunity: answers 'move already done ya baaki?'
if __name__ == "__main__":
    for SCOPE, REPORT in RUN["reports"].items():
        if SCOPE.startswith("POOLED_"):
            continue
        explain(f"8. {SCOPE}: movement AFTER the later signal", "First marker per absorption event/type. "
                "Marker close ke NEXT bar se original day-10 deadline tak measurement hai. "
                "Stop-already-hit aur profit-already-hit counts dekho. Remaining MFE/MAE market path hai; "
                "active-pre-stop MFE original stop ke pehle ki remaining opportunity hai. "
                "Day-10 marker ke baad zero bars bache hain, isliye remaining metrics n/a, zero success nahi.")
        show_table(REPORT["remaining_move"])

# %% 14. Candlestick examples and direct inspection
if __name__ == "__main__":
    explain("9. Visual examples", "Blue shaded area=absorption detection window. Black end line=signal-close reference. "
            "Squares=compression; orange arrows=exhaustion; purple arrows=expansion; black diamonds=first later markers. "
            "Green/red dotted lines=first profit/stop bar. Candles beyond stop are market context. "
            "Examples selected by category/date, not maximum profit.")
    EXAMPLE_IDS = plot_pattern_examples(RUN, max_examples=6)
    # More examples for one instrument:
    # plot_pattern_examples(RUN, instrument="SMZ26-F27", max_examples=4)
    # Plot any exact event from RUN['events']:
    # fig = plot_event(RUN['instruments']['SMZ26-F27'], RUN['events'].iloc[0].event_id)
    # display(fig)
    # Full event/landmark ledgers:
    # display(RUN['events'])
    # display(RUN['landmarks'])
    # Optional pooled report (front instruments are not independent observations):
    # display(RUN['reports']['POOLED_SM']['patterns'])

# %% 15. Same signals across fronts and related products: reuse RUN, no fetch
if __name__ == "__main__":
    import market_analysis.combined_signal_analysis.cross_front_analysis as cross_front
    importlib.reload(cross_front)
    CROSS = cross_front.analyze_cross_front_signals(
        RUN, windows=[0, 1, 2, 5], related_product_groups=[["SM", "BO"]], save=True)
    explain("10. Cross-front / cross-product signals", "0=same day; 2=plus/minus 2 observed trading bars. "
            "BUY/SELL absorption and each directional PVD type match separately. Absorption date=detection END. "
            "Compression/PVD counts signal days, not episodes. Product ratio=unique matched anchors / eligible anchors; "
            "one anchor matching several fronts counts once. Eligible requires scored partner data throughout the window. "
            "Later matches are descriptive confirmation, not information available at entry. "
            "Front/back month gaps show contract proximity. Positive lag means target signal came later.")
    show_table(CROSS["product_summary"], limit=100)
    show_table(CROSS["pair_table"].sort_values(
        ["front_month_gap", "back_month_gap", "signal", "within_trading_bars"]), limit=100)
    show_table(CROSS["matches"], limit=30)
    print("Full tables saved:", Path(RUN["output_dir"]) / "cross_front")
