"""Five prespecified context hypotheses on existing RUN; no API calls.

All candidate features stop at absorption detection END. Returns small product
matrices and auditable event/group tables. Payoff is the existing hindsight
pre-stop opportunity, not realized P&L. Resolved-only averages are explicit.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULTS = dict(range_lookback=20, volume_lookback=20, recovery_cutoff=.5,
                relative_volume_cutoff=1.5, min_resolved_per_group=10,
                min_retained_fraction=.30, pvd_horizon=None)

DESCRIPTIONS = {
    "H1_REJECTION": "Absorption + during-window rejection of a prior range boundary vs no rejection",
    "H2_EXHAUSTION_BEFORE": "Latest before exhaustion reversal aligned with absorption vs opposite",
    "H2_EXHAUSTION_DURING": "Latest during exhaustion reversal aligned with absorption vs opposite",
    "H3_EXPANSION_RECOVERY": "Before expansion opposite absorption direction: high recovery vs low recovery",
    "H4_COMPRESSION": "During compression present vs absent; larger future move is the primary question",
    "H5_EXPANSION_VOLUME": "Same before expansion-to-absorption sequence: high total volume vs ordinary",
}


def _horizon(run, result, cfg):
    """Prefer parameters used to create RUN, including existing saved metadata."""
    if cfg["pvd_horizon"] is not None:
        return int(cfg["pvd_horizon"])
    if "pvd_params" in result:
        return int(result["pvd_params"]["price_horizon_h"])
    metadata = Path(run.get("output_dir", ".")) / "metadata.json"
    if metadata.is_file():
        return int(json.loads(metadata.read_text(encoding="utf-8"))["pvd_params"]["price_horizon_h"])
    raise ValueError("Old RUN has no PVD horizon metadata. Pass pvd_horizon equal to the horizon used to score PVD.")


def _latest(part, category):
    hits = part.loc[part.pvd_signal.astype(str).str.endswith(category)]
    return None if hits.empty else hits.iloc[-1]


def extract_features(run, settings=None):
    """One event row; past reference ranges exclude the entire current absorption.

    H1: sell absorption requires low below prior low and that day's close back
    above it; buy absorption mirrored. H3 latest expansion in BEFORE is used
    without searching backwards for a more favorable matching signal. Its
    directional price displacement uses the original PVD model horizon.
    """
    cfg = {**DEFAULTS, **(settings or {})}
    for key in ("range_lookback", "volume_lookback", "min_resolved_per_group"):
        if isinstance(cfg[key], bool) or int(cfg[key]) != cfg[key] or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not 0 <= cfg["recovery_cutoff"] <= 1 or cfg["relative_volume_cutoff"] <= 0:
        raise ValueError("Invalid recovery/relative-volume cutoff")
    if not 0 <= cfg["min_retained_fraction"] <= 1:
        raise ValueError("min_retained_fraction must be between 0 and 1")
    records = []
    for instrument, result in run["instruments"].items():
        raw, daily = result["raw"].reset_index(drop=True), result["daily"].reset_index(drop=True)
        if len(raw) != len(daily) or not pd.to_datetime(raw.Date, utc=True).equals(pd.to_datetime(daily.Date, utc=True)):
            raise ValueError(f"{instrument}: raw/daily bars must align")
        if result["events"].empty:
            continue
        h = _horizon(run, result, cfg)
        if h < 1:
            raise ValueError("PVD horizon must be positive")
        tick, before, forward = float(result["tick_size"]), int(result["settings"]["before_bars"]), int(result["settings"]["forward_bars"])
        for event in result["events"].itertuples():
            start, end = int(event.window_start_index), int(event.signal_index)
            sign = 1 if event.absorption_direction == "SELL" else -1
            during = daily.iloc[start:end+1]
            part_before = daily.iloc[max(0,start-before):start]
            row = dict(event_id=event.event_id, product=result["threshold"]["product"], instrument=instrument,
                       Date=raw.iloc[end].Date, absorption_side=event.absorption_direction,
                       duration=int(event.candle_length), outcome=event.outcome,
                       research_payoff_ticks=event.research_payoff_ticks,
                       full_forward=end+forward < len(raw), pvd_horizon_used=h,
                       before_pvd_known=len(part_before)==before and before>0 and np.isfinite(part_before.pvd_z).all(),
                       during_pvd_known=len(during)>0 and np.isfinite(during.pvd_z).all())
            # Baseline range ends BEFORE absorption START, never includes future bars.
            prior = raw.iloc[max(0,start-int(cfg["range_lookback"])):start]
            row["range_known"] = len(prior)==cfg["range_lookback"]
            row["rejection"] = False
            if row["range_known"]:
                lo, hi = prior.low.min(), prior.high.max()
                row["rejection"] = bool(((during.low < lo) & (during.close > lo)).any() if sign==1 else
                                        ((during.high > hi) & (during.close < hi)).any())
            row["compression_known"] = np.isfinite(during.compression_score).all()
            row["compression"] = bool((during.compression_score >= result["settings"]["compression_threshold"]).any())
            for phase, part in (("before",part_before),("during",during)):
                latest = _latest(part, "EXHAUSTION")
                row[phase+"_exhaustion_state"] = "UNKNOWN" if not row[phase+"_pvd_known"] else "NO_SIGNAL"
                if row[phase+"_pvd_known"] and latest is not None:
                    reversal = 1 if latest.pvd_signal.startswith("DOWN") else -1
                    row[phase+"_exhaustion_state"] = "ALIGNED" if reversal==sign else "OPPOSITE"
            row.update(expansion_sequence=False, recovery_fraction=np.nan, relative_volume=np.nan,
                       expansion_name=None, expansion_date=pd.NaT)
            expansion = _latest(part_before, "EXPANSION") if row["before_pvd_known"] else None
            if expansion is not None:
                j = int(expansion.name)
                exp_sign = 1 if expansion.pvd_signal.startswith("UP") else -1
                row["expansion_name"], row["expansion_date"] = expansion.pvd_signal, raw.iloc[j].Date
                row["expansion_sequence"] = exp_sign == -sign
                if j >= h:
                    displacement = exp_sign*(float(raw.iloc[j].close)-float(raw.iloc[j-h].close))
                    if displacement > 0:
                        row["recovery_fraction"] = -exp_sign*(float(raw.iloc[end].close)-float(raw.iloc[j].close))/displacement
                baseline_volume = raw.iloc[max(0,j-int(cfg["volume_lookback"])):j].volume
                if len(baseline_volume)==cfg["volume_lookback"] and np.isfinite(baseline_volume).all() and baseline_volume.median()>0:
                    row["relative_volume"] = float(raw.iloc[j].volume)/float(baseline_volume.median())
            # Raw market opportunity, independent of original stop; includes NO-MOVE events.
            row["forward_mfe_ticks"], row["forward_move_ticks"] = np.nan, np.nan
            if row["full_forward"]:
                future = raw.iloc[end+1:end+forward+1]
                entry = float(raw.iloc[end].close)
                up, down = max(0.,(future.high.max()-entry)/tick), max(0.,(entry-future.low.min())/tick)
                row["forward_mfe_ticks"] = up if sign==1 else down
                row["forward_move_ticks"] = max(up,down)
            records.append(row)
    return pd.DataFrame(records), cfg


def _stats(data):
    """No invented losses for UNRESOLVED/CENSORED; always expose their count."""
    resolved = data.loc[data.outcome.isin(["SUCCESS","FAILURE"])]
    full = data.loc[data.full_forward]
    return dict(signals=len(data), resolved=len(resolved), unresolved=int(data.outcome.eq("UNRESOLVED").sum()),
                censored=int(data.outcome.eq("CENSORED").sum()),
                success_pct=100*resolved.outcome.eq("SUCCESS").mean() if len(resolved) else np.nan,
                expected_ticks=resolved.research_payoff_ticks.mean(),
                median_payoff_ticks=resolved.research_payoff_ticks.median(),
                total_opportunity_ticks=resolved.research_payoff_ticks.sum() if len(resolved) else np.nan,
                full_forward=len(full), median_move_ticks=full.forward_move_ticks.median(),
                median_mfe_ticks=full.forward_mfe_ticks.median())


def _groups(events, cfg):
    """Fixed eligible universes; absent/unknown features do not become negative examples."""
    yield "H1_REJECTION", events.range_known, events.rejection, "rejection", "no_rejection"
    for phase in ("before","during"):
        states = events[phase+"_exhaustion_state"]
        yield "H2_EXHAUSTION_"+phase.upper(), states.isin(["ALIGNED","OPPOSITE"]), states.eq("ALIGNED"), "aligned", "opposite"
    sequence = events.expansion_sequence
    yield "H3_EXPANSION_RECOVERY", sequence & events.recovery_fraction.notna(), events.recovery_fraction.ge(cfg["recovery_cutoff"]), "high_recovery", "low_recovery"
    yield "H4_COMPRESSION", events.compression_known, events.compression, "compression", "no_compression"
    yield "H5_EXPANSION_VOLUME", sequence & events.relative_volume.notna(), events.relative_volume.ge(cfg["relative_volume_cutoff"]), "high_volume", "ordinary_volume"


def test_multi_signal_hypotheses(run, settings=None, save=True):
    """Product matrices + subgroup/membership ledgers. No threshold optimization.

    Verdict is descriptive screening, NOT significance or out-of-sample evidence.
    Main payoff gain compares Selected vs Comparison, not vs all absorptions.
    H2 omits no-exhaustion cases; H3/H5 restrict to opposite-expansion sequences.
    Median and per-instrument agreement flag potentially concentrated mean gains.
    """
    events, cfg = extract_features(run, settings)
    if events.empty:
        raise ValueError("RUN contains no retained absorption events")
    main, detail, members, stability, baselines = [], [], [], [], []
    for product, data in events.groupby("product", sort=True):
        baselines.append(dict(Product=product, **_stats(data)))
        for hypothesis, eligible_mask, selected_mask, sel_name, cmp_name in _groups(data,cfg):
            eligible = data.loc[eligible_mask]
            selected = data.loc[eligible_mask & selected_mask]
            comparison = data.loc[eligible_mask & ~selected_mask]
            a, b = _stats(selected), _stats(comparison)
            gain, success_gain = a["expected_ticks"]-b["expected_ticks"], a["success_pct"]-b["success_pct"]
            median_gain = a["median_payoff_ticks"]-b["median_payoff_ticks"]
            move_gain = a["median_move_ticks"]-b["median_move_ticks"]
            retained = len(selected)/len(eligible) if len(eligible) else np.nan
            count_a, count_b = (a["full_forward"],b["full_forward"]) if hypothesis=="H4_COMPRESSION" else (a["resolved"],b["resolved"])
            if min(count_a,count_b) < cfg["min_resolved_per_group"]:
                verdict = "INSUFFICIENT_SAMPLE"
            elif retained < cfg["min_retained_fraction"]:
                verdict = "LOW_SIGNAL_RETENTION"
            elif hypothesis=="H4_COMPRESSION":
                verdict = "LARGER_MOVE_AND_PAYOFF" if move_gain>0 and gain>0 else "LARGER_MOVE_ONLY" if move_gain>0 else "NO_LARGER_MOVE"
            elif gain>0 and success_gain>=0 and median_gain>=0:
                verdict = "PROMISING_IN_SAMPLE"
            elif gain>0:
                verdict = "MIXED_MEAN_GAIN"
            else:
                verdict = "NO_PAYOFF_GAIN"
            stable_count, stable_positive = 0,0
            for instrument, subset in eligible.groupby("instrument"):
                selected_ids = set(selected.event_id)
                sa = _stats(subset.loc[subset.event_id.isin(selected_ids)])
                sb = _stats(subset.loc[~subset.event_id.isin(selected_ids)])
                d = sa["expected_ticks"]-sb["expected_ticks"]
                if sa["resolved"] and sb["resolved"]:
                    stable_count += 1
                    stable_positive += int(d>0)
                stability.append(dict(Product=product,Hypothesis=hypothesis,Instrument=instrument,
                                      Selected_resolved=sa["resolved"],Comparison_resolved=sb["resolved"],Gain_ticks=d))
            main.append(dict(Product=product,Hypothesis=hypothesis,Total_abs=len(data),Eligible=len(eligible),
                             Selected=len(selected),Comparison=len(comparison),Keep_pct=100*retained,
                             Resolved_S=a["resolved"],Resolved_C=b["resolved"],Success_S_pct=a["success_pct"],
                             Success_C_pct=b["success_pct"],Ticks_S=a["expected_ticks"],Ticks_C=b["expected_ticks"],
                             Gain_ticks=gain,Median_gain_ticks=median_gain,Move_gain_ticks=move_gain,
                             Positive_instruments=f"{stable_positive}/{stable_count}",Verdict=verdict))
            for name, group in ((sel_name,selected),(cmp_name,comparison)):
                detail.append(dict(Product=product,Hypothesis=hypothesis,Group=name,**_stats(group)))
            for name, group in (("Selected",selected),("Comparison",comparison),("Excluded",data.loc[~eligible_mask])):
                for row in group.itertuples():
                    members.append(dict(Product=product,Hypothesis=hypothesis,Group=name,event_id=row.event_id))
    output = dict(matrix=pd.DataFrame(main),groups=pd.DataFrame(detail),baseline=pd.DataFrame(baselines),
                  events=events,membership=pd.DataFrame(members),instrument_audit=pd.DataFrame(stability),settings=cfg)
    if save:
        folder = Path(run["output_dir"])/"multi_signal_hypotheses"
        folder.mkdir(parents=True,exist_ok=True)
        for name, value in output.items():
            if isinstance(value,pd.DataFrame):
                value.to_csv(folder/(name+".csv"),index=False)
        (folder/"settings.json").write_text(json.dumps(cfg,indent=2),encoding="utf-8")
    return output


def display_multi_signal_tests(output):
    """Compact matrices, with optional full tables accessible in output."""
    from IPython.display import display, Markdown
    display(Markdown("## Multisignal screening by product\n\n"
        "Selected vs Comparison test hai; Gain_ticks = Ticks_S - Ticks_C. "
        "Keep_pct ka denominator Eligible hai, Total_abs nahi. "
        "Success = original profit-before-configured-ATR-stop label. "
        "Expected ticks resolved-only maximum pre-stop opportunity hai, realized profit nahi. "
        "H2 uses exhaustion only; before direction automatically invert nahi hoti. "
        "PROMISING means in-sample indication, proof nahi.\n\n" +
        "\n".join(f"- **{name}**: {description}" for name,description in DESCRIPTIONS.items())))
    display(Markdown("### Quick verdict matrix\n\n"
        "Products side-by-side. INSUFFICIENT_SAMPLE means conclusion pending, pattern failed nahi. "
        "PROMISING_IN_SAMPLE means explore further; statistically confirmed strategy nahi."))
    display(output["matrix"].pivot(index="Hypothesis",columns="Product",values="Verdict"))
    for product, table in output["matrix"].groupby("Product"):
        display(Markdown(f"### {product}: kya useful lag raha hai?\n\n"
            "Pehle Resolved_S / Resolved_C, phir Gain_ticks aur Verdict dekho. "
            "Selected/Comparison total signal counts hain; unresolved/censored averages mein excluded. "
            "Positive_instruments = positive mean gain wale instruments / both groups resolved wale instruments; "
            "ye significance test nahi. H4 ka main result next movement table mein hai."))
        cols = ["Hypothesis","Selected","Comparison","Resolved_S","Resolved_C","Success_S_pct", "Success_C_pct",
                "Ticks_S","Ticks_C","Gain_ticks","Verdict"]
        display(table[cols].style.format(precision=2,na_rep="n/a").hide(axis="index"))
    display(Markdown("### Signal coverage aur consistency\n\n"
        "Eligible = valid features + required sequence. Excluded cases ko negative examples nahi banaya. "
        "Keep_pct hypothetical selected-only filter ka eligible retention hai; currently koi trades remove nahi hue. "
        "Median_gain_ticks check karo: mean gain kuch large opportunities se toh nahi?"))
    display(output["matrix"][["Product","Hypothesis","Total_abs","Eligible","Keep_pct","Median_gain_ticks","Positive_instruments"]]
            .style.format(precision=2,na_rep="n/a").hide(axis="index"))
    display(Markdown("### H4: compression ke baad move bada tha?\n\n"
        "median_move_ticks = next full horizon mein larger of UP/DOWN excursions, stop ignore karke. "
        "NO-MOVE cases include hain. full_forward = is measurement ka sample size. "
        "Positive move gain reversal/trade profit ki guarantee nahi."))
    h4 = output["groups"].loc[output["groups"].Hypothesis.eq("H4_COMPRESSION")]
    display(h4[["Product","Group","signals","full_forward","median_move_ticks","median_mfe_ticks"]]
            .style.format(precision=2,na_rep="n/a").hide(axis="index"))
    display(Markdown("### Original baseline / unresolved counts\n\n"
        "Original pooled absorption performance. Hypothesis gain apne eligible Comparison se hai, "
        "is overall baseline se direct fair comparison nahi. Unresolved/Censored exclusion visible hai."))
    display(output["baseline"].style.format(precision=2,na_rep="n/a").hide(axis="index"))
    print("Share matrix + coverage + H4 tables. Full audit: TESTS['groups'], TESTS['events'], TESTS['instrument_audit'].")
