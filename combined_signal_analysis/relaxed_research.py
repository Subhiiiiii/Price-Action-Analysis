"""Cache-first BO/SM/W research with two prespecified experimental profiles.

Independent of production GPT-OSS CSV / Excel thresholds. Never fabricates data.
First profile fetches missing ranges; others reuse the identical raw bars.
"""
import importlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

from . import combined_signal_notebook as core
from . import multi_signal_hypotheses as multi

PROFILES = {
    "reference_research": dict(abs_volume_z=1.50, abs_price_z=-0.25,
                               pvd_exhaustion_z=-1.50, pvd_expansion_z=1.50, compression_threshold=.30),
    "relaxed_research": dict(abs_volume_z=1.00, abs_price_z=0.25,
                             pvd_exhaustion_z=-1.00, pvd_expansion_z=1.00, compression_threshold=.30),
}


def build_universe(products=("BO","SM","W"), front_n=5, structures=("spread","jumbo"), as_of=None):
    """Generate today's front N of each requested structure using product config."""
    if isinstance(front_n,bool) or not isinstance(front_n,int) or front_n<1:
        raise ValueError("front_n must be a positive integer")
    if not products or not structures:
        raise ValueError("products and structures cannot be empty")
    if len(set(products))!=len(products) or len(set(structures))!=len(structures):
        raise ValueError("Duplicate products/structures are not allowed")
    # contracts imports PRODUCTS at module load; refresh after a config edit.
    contracts = importlib.import_module("market_analysis.utils.contracts")
    importlib.reload(contracts)
    when = pd.Timestamp(as_of).date() if as_of is not None else pd.Timestamp.now(tz="Asia/Kolkata").date()
    rows=[]
    for product in products:
        if product not in core.config.PRODUCTS:
            raise ValueError(f"{product} missing from config.PRODUCTS; reload config/kernel after editing")
        tick = float(core.config.PRODUCTS[product]["tick_size"])
        if not np.isfinite(tick) or tick<=0:
            raise ValueError(f"Invalid tick_size for {product}")
        for structure in structures:
            for rank,instrument in enumerate(contracts.get_front_instruments(product,structure,when,front_n),start=1):
                rows.append(dict(product=product,instrument=instrument,structure=structure,front_rank=rank,tick_size=tick))
    return pd.DataFrame(rows)


def research_thresholds(universe, name, profile):
    """Explicit experimental values for all contracts, no manual/GPT-OSS fallback."""
    required=set(PROFILES["reference_research"])
    if set(profile)!=required or not all(np.isfinite(v) for v in profile.values()):
        raise ValueError(f"{name}: provide exactly {sorted(required)} as finite numbers")
    if profile["abs_volume_z"]<0 or profile["pvd_exhaustion_z"]>=0 or profile["pvd_expansion_z"]<=0:
        raise ValueError(f"{name}: invalid signed z cutoffs")
    if not 0<=profile["compression_threshold"]<=1:
        raise ValueError("Compression threshold must be between 0 and 1")
    table=universe.copy()
    for column,value in profile.items():
        table[column]=value
    table["status"],table["error"]="ok",""
    table["profile"],table["threshold_source"]=name,"experimental_prespecified_not_validated"
    return table


def run_relaxed_research(products=("BO","SM","W"),front_n=5,structures=("spread","jumbo"),
                         as_of=None,months_back=None,candle_lengths=(1,2,3,5),profiles=None,
                         data_mode="auto",hypothesis_settings=None,output_dir=None,
                         raw_by_instrument=None):
    """Run every profile on the same front universe and first-run data snapshot.

    Snapshot = current fronts back through requested months, not a historical
    rolling-front portfolio. Relaxing z thresholds does NOT guarantee higher
    deduplicated event counts, because longest-window selections can change.
    """
    when = str(pd.Timestamp(as_of).date()) if as_of is not None else str(pd.Timestamp.now(tz="Asia/Kolkata").date())
    universe=build_universe(products,front_n,structures,when)
    profiles=PROFILES if profiles is None else profiles
    if not profiles:
        raise ValueError("At least one research profile required")
    for name in profiles:
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
            raise ValueError("Profile names must be simple letters/numbers/underscores/hyphens")
    folder=Path(output_dir) if output_dir else core.PROJECT_ROOT/"combined_signal_analysis"/"outputs"/"relaxed_research"/when
    folder.mkdir(parents=True,exist_ok=True)
    audits={name:research_thresholds(universe,name,profile) for name,profile in profiles.items()}
    runs,tests,baselines,matrices,coverages,errors={}, {}, [], [], [], []
    shared_raw=raw_by_instrument
    first_coverage=None
    for name,profile in profiles.items():
        cfg={**core.SETTINGS,"as_of":when,"data_mode":data_mode,"candle_lengths":list(candle_lengths),
             "compression_threshold":profile["compression_threshold"],"output_dir":folder/name}
        if months_back is not None:
            cfg["months_back"]=months_back
        print(f"\nPROFILE {name}: experimental thresholds, not production choices",flush=True)
        run=core.run_combined_signal_analysis(settings=cfg,threshold_audit=audits[name],raw_by_instrument=shared_raw)
        runs[name]=run
        if shared_raw is None:
            shared_raw={inst:r["raw"].copy() for inst,r in run["instruments"].items()}
            first_coverage=run["coverage"].copy()
        coverage=run["coverage"].copy()
        coverage["Profile"]=name
        coverages.append(coverage)
        err=run["errors"].copy()
        err["Profile"]=name
        errors.append(err)
        for product in products:
            results=[r for r in run["instruments"].values() if r["threshold"]["product"]==product]
            pieces=[r["events"] for r in results if not r["events"].empty]
            events=pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame()
            resolved=events.loc[events.outcome.isin(["SUCCESS","FAILURE"])] if not events.empty else events
            counts=[len(r["events"]) for r in results]
            per_month=[]
            for result in results:
                meta=result["absorption_metadata"]
                a=pd.Timestamp(meta["analysis_start"])
                b=pd.Timestamp(meta["analysis_end"])
                exposure=max(0,(b-a).days+1)/30.4375
                per_month.append(len(result["events"])/exposure if exposure else np.nan)
            baselines.append(dict(Profile=name,Product=product,Requested=int(universe["product"].eq(product).sum()),
                                  Loaded=len(results),Signals=sum(counts),Zero_signal_instruments=sum(n==0 for n in counts),
                                  Median_signals_per_instrument_month=float(np.nanmedian(per_month)) if per_month else np.nan,
                                  Resolved=len(resolved),Success_pct=100*resolved.outcome.eq("SUCCESS").mean() if len(resolved) else np.nan,
                                  Expected_ticks=resolved.research_payoff_ticks.mean() if len(resolved) else np.nan,
                                  Unresolved=int(events.outcome.eq("UNRESOLVED").sum()) if not events.empty else 0,
                                  Censored=int(events.outcome.eq("CENSORED").sum()) if not events.empty else 0))
        if any(not r["events"].empty for r in run["instruments"].values()):
            tested=multi.test_multi_signal_hypotheses(run,settings=hypothesis_settings,save=True)
            tests[name]=tested
            matrix=tested["matrix"].copy()
            matrix["Profile"]=name
            matrices.append(matrix)
        else:
            print(f"{name}: no absorption events; inspect coverage/errors. No hypothesis verdict fabricated.")
    baseline=pd.DataFrame(baselines)
    matrix=pd.concat(matrices,ignore_index=True) if matrices else pd.DataFrame()
    combined_errors=pd.concat(errors,ignore_index=True) if errors else pd.DataFrame()
    coverage=pd.concat(coverages,ignore_index=True) if coverages else pd.DataFrame()
    for name,table in dict(universe=universe,baseline=baseline,hypotheses=matrix,errors=combined_errors,coverage=coverage).items():
        table.to_csv(folder/(name+".csv"),index=False)
    if first_coverage is not None:
        first_coverage.to_csv(folder/"shared_data_snapshot_coverage.csv",index=False)
    (folder/"research_settings.json").write_text(json.dumps(dict(as_of=when,products=list(products),front_n=front_n,
        structures=list(structures),profiles=profiles,months_back=months_back or core.SETTINGS["months_back"],
        candle_lengths=list(candle_lengths),hypothesis_settings=hypothesis_settings or multi.DEFAULTS,
        note="Experimental profiles; current-front historical snapshot; conditional research opportunity, not realized P&L"),indent=2),encoding="utf-8")
    return dict(runs=runs,tests=tests,baseline=baseline,matrix=matrix,coverage=coverage,errors=combined_errors,
                universe=universe,output_dir=folder)


def load_saved_research(folder):
    """Display saved research without fetching data or recalculating indicators."""
    folder=Path(folder)
    result={"output_dir":folder}
    for key,filename in {"baseline":"baseline.csv","matrix":"hypotheses.csv","coverage":"coverage.csv",
                         "errors":"errors.csv","universe":"universe.csv"}.items():
        try:
            result[key]=pd.read_csv(folder/filename)
        except pd.errors.EmptyDataError:
            result[key]=pd.DataFrame()
    return result


def display_relaxed_research(research):
    """Small pooled output with Wheat data failures visible even if no events."""
    from IPython.display import display,Markdown
    display(Markdown("## BO / SM / W relaxed research\n\n"
        "reference_research aur relaxed_research dono experimental hain; old optimized thresholds se comparison nahi. "
        "Same instruments/raw bars par comparison. Historical front snapshot hai, rolling front portfolio nahi. "
        "Expected_ticks = resolved-only maximum pre-stop opportunity. Raw ticks different products mein directly comparable nahi. "
        "Relaxation mein absorption/PVD cutoffs change hote hain; compression 0.30 same hai."))
    display(research["baseline"].style.format(precision=2,na_rep="n/a").hide(axis="index"))
    display(Markdown("Loaded < Requested ho toh product result incomplete hai. "
        "Median monthly frequency requested calendar span ka rough per-instrument rate hai; missing history/warmup se affected ho sakta hai."))
    if not research["matrix"].empty:
        display(Markdown("### Hypotheses by product/profile\n\n"
            "Gain_ticks = selected vs comparison, relaxed minus reference nahi. "
            "PROMISING means exploratory, INSufficient sample means conclusion pending."))
        display(research["matrix"].pivot(index="Hypothesis",columns=["Product","Profile"],values="Verdict"))
        display(research["matrix"][["Profile","Product","Hypothesis","Selected","Comparison","Resolved_S","Resolved_C","Gain_ticks","Verdict"]]
                .style.format(precision=2,na_rep="n/a").hide(axis="index"))
    display(Markdown("### Coverage / errors\n\nW absent ho toh errors dekho; zero-signal aur missing-data alag cases hain. "
        "Later profiles same cached snapshot use karte hain; initial profile coverage freshness check karo."))
    display(research["coverage"])
    display(research["errors"])
    print("Saved:",research["output_dir"])
