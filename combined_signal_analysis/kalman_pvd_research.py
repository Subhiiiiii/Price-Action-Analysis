"""Offline, causal Kalman context and 1:1 absorption screening.

Uses frozen daily-score/event snapshots. Does not change production thresholds,
old maximum-opportunity labels, or retrospective absorption event selection.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize

DEFAULTS = dict(training_bars=60, confidence_z=1.64, weakening_ratio=.5,
                volume_lookback=20, sudden_lookback=5, low_volume_ratio=.3,
                sudden_drop_ratio=.5, pressure_imbalance=.3,
                min_completed=10, min_retention=.2,
                roundtrip_cost_ticks={}, products=["BO", "SM", "W"])


def volume_features(raw, cfg):
    """Past-only baselines; invalid volume is UNKNOWN, never low activity."""
    d = raw.copy()
    v, buy, sell = (pd.to_numeric(d[k], errors="coerce") for k in
                    ("volume", "buyvolume", "sellvolume"))
    valid = np.isfinite(v) & np.isfinite(buy) & np.isfinite(sell)
    valid &= (v > 0) & (buy >= 0) & (sell >= 0)
    valid &= np.isclose(buy + sell, v, rtol=.01, atol=1)
    clean = v.where(valid)
    base = clean.shift(1).rolling(cfg["volume_lookback"], min_periods=cfg["volume_lookback"]).median()
    recent = clean.shift(1).rolling(cfg["sudden_lookback"], min_periods=cfg["sudden_lookback"]).median()
    d["imbalance"] = ((buy-sell)/(buy+sell)).where(valid)
    d["relative_volume"] = clean/base
    d["sudden_ratio"] = clean/recent
    d["volume_known"] = valid & (base > 0) & (recent > 0)
    d["low_activity"] = d.volume_known & d.relative_volume.lt(cfg["low_volume_ratio"])
    d["sudden_drop"] = d.volume_known & d.sudden_ratio.lt(cfg["sudden_drop_ratio"])
    return d


def _filter(y, x, theta, return_states=False):
    """Local level/slope; optional lagged volume input enters level increment.

    Standardized price units. Noise variances are positive. One-sided Joseph
    covariance update. Initial state is diffuse; first 5 innovations burn in.
    """
    r, qlevel, qslope = np.exp(theta[:3])
    gamma = np.asarray(theta[3:])
    f = np.array([[1., 1.], [0., 1.]])
    q = np.diag([qlevel, qslope])
    state, cov = np.array([y[0], 0.]), np.diag([10., 1.])
    history, likelihood = [], 0.
    for i, price in enumerate(y):
        if i:
            state = f @ state
            if len(gamma):
                state[0] += float(x[i] @ gamma)
            cov = f @ cov @ f.T + q
        error, variance = price-state[0], cov[0, 0]+r
        if i >= 5:
            likelihood += .5*(np.log(variance)+error*error/variance)
        gain = cov[:, 0]/variance
        state = state + gain*error
        a = np.eye(2)-np.outer(gain, [1., 0.])
        cov = a @ cov @ a.T + r*np.outer(gain, gain)
        if return_states:
            history.append((state[0], state[1], np.sqrt(max(cov[1, 1], 0.)),
                            error, np.sqrt(variance)))
    return np.asarray(history) if return_states else float(likelihood)


def fit_context(d, cfg, volume=False):
    """Fit once on initial prefix, freeze parameters, then filter sequentially.

    The calibrated prefix is not eligible for event context. Parameters and
    scaling never use later evaluation bars. Volume coefficients describe a
    lagged association, not causal price impact or order-book imbalance.
    """
    train = int(cfg["training_bars"])
    if len(d) <= train+1:
        raise ValueError(f"Need more than {train+1} raw bars")
    prices = d.close.to_numpy(float)
    if not np.isfinite(prices).all():
        raise ValueError("Non-finite price")
    scale = float(np.std(np.diff(prices[:train]), ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Flat/invalid training prices")
    y = (prices-prices[0])/scale
    # Both values are lagged one bar. Missing values are neutral inputs, flagged.
    xv = np.column_stack([d.imbalance.shift(1),
                          np.log(d.relative_volume.clip(lower=.01, upper=100)).shift(1)])
    input_known = np.isfinite(xv).all(axis=1)
    x = np.nan_to_num(xv, nan=0., posinf=0., neginf=0.) if volume else np.zeros((len(d), 0))
    if volume and input_known[:train].sum() < 20:
        raise ValueError("Insufficient valid volume training inputs")
    starts = [np.log([.3, .03, .001]), np.log([.8, .1, .01])]
    bounds = [(-9, 4)]*3 + ([(-3, 3)]*2 if volume else [])
    fits = []
    for initial in starts:
        initial = np.r_[initial, [0., 0.]] if volume else initial
        fit = minimize(
            lambda t: _filter(y[:train], x[:train], t), initial,
            method="L-BFGS-B", bounds=bounds, options={"maxiter": 180})
        if fit.success and np.isfinite(fit.fun):
            fits.append(fit)
    if not fits:
        raise ValueError("Kalman prefix optimization did not converge")
    fit = min(fits, key=lambda a: a.fun)
    h = _filter(y, x, fit.x, True)*scale
    out = pd.DataFrame(h, columns=["level", "slope", "slope_se", "innovation", "innovation_se"], index=d.index)
    out["level"] += prices[0]  # undo centering; slope/errors only need scale conversion
    out["slope_z"] = out.slope/out.slope_se
    out["state"] = np.select([out.slope_z.ge(cfg["confidence_z"]),
                               out.slope_z.le(-cfg["confidence_z"])], [1, -1], default=0)
    out["ready"] = np.arange(len(d)) >= train
    if volume:
        out["ready"] &= input_known
    audit = dict(train_bars=train, train_end=str(d.iloc[train-1].Date),
                 train_objective=float(fit.fun), observation_variance=float(np.exp(fit.x[0])*scale**2),
                 level_variance=float(np.exp(fit.x[1])*scale**2),
                 slope_variance=float(np.exp(fit.x[2])*scale**2),
                 at_parameter_bound=bool(any(abs(v-l)<.01 or abs(v-u)<.01 for v,(l,u) in zip(fit.x,bounds))),
                 imbalance_coefficient=float(fit.x[3]*scale) if volume else np.nan,
                 log_volume_coefficient=float(fit.x[4]*scale) if volume else np.nan)
    return out, audit


def trade_1r(raw, index, sign, risk, tick, horizon, cost_ticks=0.):
    """Fixed +/-1R ideal barriers; open checked first, intrabar ties stop-first.

    Gap-adjusted sensitivity uses stop open if beyond stop, target conservatively
    fills target. Timeouts exit horizon close. Early barriers resolve even when
    later horizon data are absent. No horizon maxima used for payoff.
    """
    if not np.isfinite(risk) or risk <= 0 or tick <= 0:
        raise ValueError("Invalid risk/tick")
    entry = float(raw.iloc[index].close)
    result = dict(status="CENSORED", gross_R=np.nan, net_R=np.nan, gap_net_R=np.nan,
                  payoff_ticks=np.nan, ambiguous=False, gap=False, exit_index=np.nan)
    last = min(index+horizon, len(raw)-1)
    for j in range(index+1, last+1):
        b = raw.iloc[j]
        if not np.isfinite([b.open,b.high,b.low,b.close]).all():
            result["status"] = "DATA_INVALID"
            return result
        op = sign*(b.open-entry)/risk
        upper = (b.high-entry)/risk if sign == 1 else (entry-b.low)/risk
        lower = (b.low-entry)/risk if sign == 1 else (entry-b.high)/risk
        if op <= -1:
            status, r, gap_r = "STOP", -1., float(op)
            result["gap"] = op < -1
        elif op >= 1:
            status, r, gap_r = "TARGET", 1., 1.
            result["gap"] = op > 1
        elif lower <= -1:
            status, r, gap_r = "STOP", -1., -1.
            result["ambiguous"] = upper >= 1
        elif upper >= 1:
            status, r, gap_r = "TARGET", 1., 1.
        elif j == index+horizon:
            r = float(sign*(b.close-entry)/risk)
            status, gap_r = "TIMEOUT", r
        else:
            continue
        cost_r = float(cost_ticks)*tick/risk
        result.update(status=status, gross_R=r, net_R=r-cost_r, gap_net_R=gap_r-cost_r,
                      payoff_ticks=r*risk/tick-cost_ticks, exit_index=j)
        return result
    return result


def _stats(g):
    """All completed trades including timeouts; never discard non-barrier trades."""
    done = g.loc[g.status.isin(["TARGET", "STOP", "TIMEOUT"])]
    return dict(Signals=len(g), Completed=len(done), Target=int(g.status.eq("TARGET").sum()),
                Stop=int(g.status.eq("STOP").sum()), Timeout=int(g.status.eq("TIMEOUT").sum()),
                Censored=int(g.status.eq("CENSORED").sum()), Invalid=int(g.status.eq("DATA_INVALID").sum()),
                Ambiguous=int(g.ambiguous.sum()), Gaps=int(g.gap.sum()),
                Target_pct=100*done.status.eq("TARGET").mean(), Mean_R=done.net_R.mean(),
                Median_R=done.net_R.median(), Mean_ticks=done.payoff_ticks.mean(),
                Gap_mean_R=done.gap_net_R.mean())


def _patterns(g):
    """Hold the basic context fixed in each selected/comparison contrast."""
    meaningful = g.prior_state.ne(0)
    yield "TREND_WEAKENING_EXHAUSTION", meaningful, g.weakening & g.exh_matches_prior
    counter = meaningful & g.prior_state.eq(-g.trade_sign)
    yield "COUNTERTREND_WEAKENING", counter, g.weakening
    yield "PRESSURE_STALL", counter & g.volume_known & g.pressure, g.weakening
    yield "LOW_VOLUME_APPARENT_EXHAUSTION", g.volume_known & g.has_exhaustion & g.end_state.eq(0), g.low_any


def run_experiment(research, profiles=("relaxed_research",), settings=None):
    """Return compact tables and full audit ledgers. Offline saved snapshots only."""
    cfg = {**DEFAULTS, **(settings or {})}
    for name in ("training_bars","volume_lookback","sudden_lookback","min_completed"):
        if isinstance(cfg[name],bool) or int(cfg[name])!=cfg[name] or cfg[name]<1:
            raise ValueError(f"{name} must be a positive integer")
    if cfg["training_bars"] < 30 or cfg["confidence_z"] <= 0:
        raise ValueError("Need training_bars>=30 and positive confidence_z")
    if not 0 < cfg["weakening_ratio"] < 1:
        raise ValueError("weakening_ratio must be between 0 and 1")
    if any(not np.isfinite(v) or v < 0 for v in cfg["roundtrip_cost_ticks"].values()):
        raise ValueError("Costs must be nonnegative")
    root = Path(research["output_dir"])
    universe = pd.read_csv(root/"universe.csv").set_index("instrument")
    records, trade_records, audits, errors, daily = [], [], [], [], {}
    exposure = {}
    for profile in profiles:
        paths = list((root/profile).glob("*/multi_signal_hypotheses/events.csv"))
        if len(paths) != 1:
            raise ValueError(f"{profile}: expected exactly one saved snapshot")
        folder = paths[0].parent.parent
        meta = json.loads((folder/"metadata.json").read_text(encoding="utf-8"))
        features = pd.read_csv(paths[0])
        features["rejection"] = features.rejection.astype(str).str.lower().map({"true":True,"false":False})
        events = pd.read_csv(folder/"events.csv")
        if features.event_id.duplicated().any() or events.event_id.duplicated().any():
            raise ValueError("Duplicate saved event ids")
        features = features.set_index("event_id")
        startdate, enddate = (pd.Timestamp(meta[k]) for k in ("analysis_start", "analysis_end"))
        exposure[profile] = ((enddate-startdate).days+1)/30.4375
        horizon = int(meta["settings"]["forward_bars"])
        for instrument in universe.index:
            product = universe.at[instrument,"product"]
            if product not in cfg["products"]:
                continue
            print(f"{profile} | {instrument}: fitting causal Kalman contexts...", flush=True)
            try:
                d = pd.read_csv(folder/f"{instrument}_daily_signals.csv")
                d["Date"] = pd.to_datetime(d.Date, utc=True)
                d = volume_features(d, cfg)
                d["analysis_period"] = d.Date.between(startdate,enddate)
                previous=d.close.shift(1)
                tr=pd.concat([d.high-d.low,(d.high-previous).abs(),(d.low-previous).abs()],axis=1).max(axis=1)
                d["lag_atr"]=tr.rolling(int(meta["abs_params"]["atr_period"])).mean().shift(1)
                tick = float(universe.at[instrument,"tick_size"])
                own = events.loc[events.instrument.eq(instrument)].copy()
                outcomes = {}
                for e in own.itertuples():
                    end = int(e.signal_index)
                    if pd.Timestamp(e.Date) != d.iloc[end].Date:
                        raise ValueError("Event index/date mismatch")
                    sign = 1 if e.absorption_direction == "SELL" else -1
                    risk = float(e.atr)*float(meta["abs_params"]["stop_atr_mult"])
                    result = trade_1r(d,end,sign,risk,tick,horizon,cfg["roundtrip_cost_ticks"].get(product,0.))
                    outcomes[e.event_id] = dict(Profile=profile,Product=product,Instrument=instrument,
                        event_id=e.event_id,Date=e.Date,Month=pd.Timestamp(e.Date).strftime("%Y-%m"),
                        side=e.absorption_direction,trade_sign=sign,risk_ticks=risk/tick,**result)
                    trade_records.append(outcomes[e.event_id])
                for variant, use_volume in [("PRICE_KALMAN",False),("VOLUME_KALMAN",True)]:
                    try:
                        k, audit = fit_context(d,cfg,use_volume)
                        daily[(profile,instrument,variant)] = pd.concat([d,k.add_prefix("kalman_")],axis=1)
                        audit.update(Profile=profile,Product=product,Instrument=instrument,Variant=variant,
                                     Events=len(own),Eligible=0,Invalid_volume_bars=int((~d.volume_known).sum()))
                        for e in own.itertuples():
                            a,b = int(e.window_start_index),int(e.signal_index)
                            if a<1 or not bool(k.loc[a-1:b,"ready"].all()):
                                continue
                            f = features.loc[e.event_id]
                            before, current = k.iloc[a-1],k.iloc[b]
                            prior_state = int(before.state)
                            window = d.iloc[a:b+1]
                            exhaust = window.loc[window.pvd_signal.astype(str).str.endswith("EXHAUSTION")]
                            exh_trend = 0 if exhaust.empty else (1 if exhaust.iloc[-1].pvd_signal.startswith("UP") else -1)
                            sign = outcomes[e.event_id]["trade_sign"]
                            vol_known = bool(window.volume_known.all())
                            # Aggregate imbalance avoids giving a tiny-volume day equal influence.
                            imbalance = float((window.buyvolume.sum()-window.sellvolume.sum())/window.volume.sum()) if vol_known else np.nan
                            rv = float(window.relative_volume.mean()) if vol_known else np.nan
                            records.append(dict(**outcomes[e.event_id],Variant=variant,
                                prior_state=prior_state,end_state=int(current.state),
                                prior_slope=float(before.slope),end_slope=float(current.slope),
                                weakening=bool(prior_state and prior_state*current.slope <= cfg["weakening_ratio"]*abs(before.slope)),
                                has_exhaustion=not exhaust.empty,exh_matches_prior=bool(prior_state and exh_trend==prior_state),
                                pvd_known=bool(np.isfinite(window.pvd_z).all()),
                                pvd_aligned=bool(exh_trend and -exh_trend==sign),
                                volume_known=vol_known,imbalance=imbalance,relative_volume=rv,
                                low_any=bool(window.low_activity.any() or window.sudden_drop.any()),
                                pressure=bool(vol_known and rv>=1 and sign*imbalance<=-cfg["pressure_imbalance"]),
                                rejection=f.rejection,range_known=str(f.range_known).lower()=="true",
                                specific_pvd=(f.during_exhaustion_state=="ALIGNED") if product=="BO" else (f.before_exhaustion_state=="OPPOSITE")))
                            audit["Eligible"] += 1
                        audits.append(audit)
                    except Exception as exc:
                        errors.append(dict(Profile=profile,Instrument=instrument,Variant=variant,Error=str(exc)))
            except Exception as exc:
                errors.append(dict(Profile=profile,Instrument=instrument,Variant="DATA",Error=str(exc)))
    ev, trades = pd.DataFrame(records),pd.DataFrame(trade_records)
    if ev.empty:
        raise ValueError(f"No eligible contexts; errors={errors[:3]}")
    # Fair variant comparison: intersection, NOT different warmup/volume cohorts.
    counts = ev.groupby(["Profile","event_id"]).Variant.nunique()
    paired = set(counts[counts==2].index)
    ev["paired"] = [(p,e) in paired for p,e in zip(ev.Profile,ev.event_id)]
    groups, matrix, monthly, stability, strategies = [],[],[],[],[]
    for (profile,product,variant), g in ev.loc[ev.paired].groupby(["Profile","Product","Variant"]):
        g = g.loc[g.pvd_known].copy()
        base = dict(Profile=profile,Product=product,Variant=variant)
        for name,eligible,selected in _patterns(g):
            a,b = g.loc[eligible & selected],g.loc[eligible & ~selected]
            sa,sb = _stats(a),_stats(b)
            gain = sa["Mean_R"]-sb["Mean_R"]
            verdict = "INSUFFICIENT_SAMPLE" if min(sa["Completed"],sb["Completed"])<cfg["min_completed"] else (
                "LOW_RETENTION" if len(a)/max(1,int(eligible.sum()))<cfg["min_retention"] else
                "POSITIVE_IN_SAMPLE" if gain>0 else "NO_MEAN_GAIN")
            matrix.append(dict(**base,Pattern=name,Eligible=int(eligible.sum()),Selected=len(a),Comparison=len(b),
                               Completed_S=sa["Completed"],Completed_C=sb["Completed"],
                               Target_S_pct=sa["Target_pct"],Target_C_pct=sb["Target_pct"],
                               R_S=sa["Mean_R"],R_C=sb["Mean_R"],Gain_R=gain,
                               Gap_gain_R=sa["Gap_mean_R"]-sb["Gap_mean_R"],Verdict=verdict))
            for label,part in [("Selected",a),("Comparison",b)]:
                groups.append(dict(**base,Pattern=name,Group=label,**_stats(part)))
            for inst in sorted(g.Instrument.unique()):
                ia,ib=_stats(a.loc[a.Instrument.eq(inst)]),_stats(b.loc[b.Instrument.eq(inst)])
                stability.append(dict(**base,Pattern=name,Instrument=inst,Completed_S=ia["Completed"],Completed_C=ib["Completed"],Gain_R=ia["Mean_R"]-ib["Mean_R"]))
            for month in sorted(g.Month.unique()):
                ma,mb=_stats(a.loc[a.Month.eq(month)]),_stats(b.loc[b.Month.eq(month)])
                monthly.append(dict(**base,Pattern=name,Month=month,Completed_S=ma["Completed"],Completed_C=mb["Completed"],Gain_R=ma["Mean_R"]-mb["Mean_R"]))
        # Incremental filters on identical original events; no threshold retuning.
        confirm = g.pvd_aligned & g.exh_matches_prior & g.weakening
        masks = {"Paired baseline":pd.Series(True,index=g.index),
                 "Raw PVD aligned":g.pvd_aligned,
                 "Kalman confirmed PVD":confirm,
                 "Confirmed + normal volume":confirm & g.volume_known & ~g.low_any,
                 "Rejection OR raw aligned":g.range_known & (g.rejection | g.pvd_aligned),
                 "Rejection OR confirmed":g.range_known & (g.rejection | confirm),
                 "Existing product OR":g.range_known & (g.rejection | g.specific_pvd)}
        loaded=int((universe["product"]==product).sum())
        for label,mask in masks.items():
            part=g.loc[mask]
            strategies.append(dict(**base,Strategy=label,Keep_pct=100*len(part)/max(1,len(g)),
                                   Signals_per_instrument_month=len(part)/(loaded*exposure[profile]),**_stats(part)))
    baseline=[dict(Profile=p,Product=pr,**_stats(g)) for (p,pr),g in trades.groupby(["Profile","Product"])]
    diagnostics=[]
    signal_diagnostics=[]
    for (profile,instrument,variant), d in daily.items():
        ready=d.loc[d.kalman_ready & d.analysis_period]
        names=ready.pvd_signal.astype(str)
        exhaustion=names.str.endswith("EXHAUSTION")
        episodes=exhaustion & names.ne(names.shift(1))
        flips=ready.kalman_state.mul(ready.kalman_state.shift(1)).eq(-1)
        signal_diagnostics.append(dict(Profile=profile,Product=universe.at[instrument,"product"],
            Variant=variant,Bars=len(ready),PVD_exhaustion_days=int(exhaustion.sum()),
            PVD_contiguous_episodes=int(episodes.sum()),Kalman_adjacent_direction_flips=int(flips.sum()),
            Uncertain_bars=int(ready.kalman_state.eq(0).sum())))
        # Evaluation innovations use only one-step predictions; volume groups are descriptive.
        part=d.loc[d.kalman_ready & d.volume_known & d.analysis_period & d.lag_atr.gt(0)].copy()
        part["activity"]=np.where(part.low_activity | part.sudden_drop,"LOW_OR_DROP","NORMAL")
        part["standardized_error"]=part.kalman_innovation/part.kalman_innovation_se
        for activity,g in part.groupby("activity"):
            diagnostics.append(dict(Profile=profile,Product=universe.at[instrument,"product"],
                Instrument=instrument,Variant=variant,Activity=activity,Bars=len(g),
                Squared_errors=float((g.standardized_error**2).sum()),
                Abs_error_ATR_sum=float((g.kalman_innovation.abs()/g.lag_atr).sum()),
                Large_errors=int(g.standardized_error.abs().gt(2).sum()),
                Uncertain=int(g.kalman_state.eq(0).sum())))
    diagnostics=pd.DataFrame(diagnostics)
    signal_diagnostics=pd.DataFrame(signal_diagnostics)
    if len(signal_diagnostics):
        signal_diagnostics=signal_diagnostics.groupby(["Profile","Product","Variant"],as_index=False).sum(numeric_only=True)
    if len(diagnostics):
        diagnostics=diagnostics.groupby(["Profile","Product","Variant","Activity"],as_index=False)[
            ["Bars","Squared_errors","Abs_error_ATR_sum","Large_errors","Uncertain"]].sum()
        diagnostics["MAE_in_ATR"]=diagnostics.Abs_error_ATR_sum/diagnostics.Bars
        diagnostics["RMS_standardized_error"]=np.sqrt(diagnostics.Squared_errors/diagnostics.Bars)
        diagnostics["Large_error_pct"]=100*diagnostics.Large_errors/diagnostics.Bars
        diagnostics["Uncertain_pct"]=100*diagnostics.Uncertain/diagnostics.Bars
        diagnostics=diagnostics.drop(columns=["Squared_errors","Abs_error_ATR_sum","Large_errors","Uncertain"])
    output=dict(events=ev,trades=trades,baseline=pd.DataFrame(baseline),matrix=pd.DataFrame(matrix),
                groups=pd.DataFrame(groups),strategies=pd.DataFrame(strategies),monthly=pd.DataFrame(monthly),
                instrument_audit=pd.DataFrame(stability),fit_audit=pd.DataFrame(audits),errors=pd.DataFrame(errors),
                volume_diagnostics=diagnostics,signal_diagnostics=signal_diagnostics,daily=daily,settings=cfg)
    dest=root/"kalman_pvd_1r"
    dest.mkdir(parents=True,exist_ok=True)
    for key,value in output.items():
        if isinstance(value,pd.DataFrame):
            value.to_csv(dest/f"{key}.csv",index=False)
    (dest/"settings.json").write_text(json.dumps(cfg,indent=2),encoding="utf-8")
    output["output_dir"]=dest
    return output


def display_experiment(result):
    """Small views plus copyable tables for interpretation in chat."""
    from IPython.display import display,Markdown
    display(Markdown("## Kalman + PVD + volume | 1:1 research\n\n"
        "Original absorption events; 1R target/stop, 10-bar configured horizon timeout included. "
        "Entry is signal close (ideal reference). Same-bar ties stop-first; gap-adjusted mean also shown. "
        "Costs default ZERO; Mean_R is simulated payoff, not maximum opportunity. "
        "Kalman fit only on initial prefix; paired variants use identical eligible events. "
        "Existing event selection is retrospective longest-window preference: this is exploratory, not a live backtest."))
    display(Markdown("**Baseline:** poore absorption set par naya 1:1 result. "
        "**Strategies:** Keep_pct aur frequency ke saath average R compare karo. "
        "**Patterns:** selected vs matched-context comparison; positive Gain_R selected group better. "
        "LOW_VOLUME pattern mein positive gain ka matlab low-volume group better tha, filter useful tha ye nahi. "
        "Low N means pending evidence. Uncertain slope ka matlab automatically range nahi."))
    for key in ("baseline","strategies","matrix"):
        display(result[key].round(3))
    audit=result["fit_audit"]
    fit_summary=audit.groupby(["Profile","Product","Variant"],as_index=False).agg(
        Fits=("Instrument","count"),Events=("Events","sum"),Eligible=("Eligible","sum"),
        Boundary_fits=("at_parameter_bound","sum"))
    print("COPY 1R BASELINE\n"+result["baseline"].round(4).to_csv(index=False))
    print("COPY KALMAN STRATEGIES\n"+result["strategies"].round(4).to_csv(index=False))
    print("COPY KALMAN PATTERNS\n"+result["matrix"].round(4).to_csv(index=False))
    print("COPY KALMAN COVERAGE\n"+fit_summary.to_csv(index=False))
    print("COPY VOLUME DIAGNOSTICS\n"+result["volume_diagnostics"].round(4).to_csv(index=False))
    print("COPY SIGNAL DIAGNOSTICS\n"+result["signal_diagnostics"].to_csv(index=False))
    for key in ("monthly","instrument_audit"):
        t=result[key]
        if t.empty:
            continue
        valid=t.loc[(t.Completed_S>0)&(t.Completed_C>0)].copy()
        valid["positive"]=valid.Gain_R>0
        small=valid.groupby(["Profile","Product","Variant","Pattern"],as_index=False).agg(
            Comparable=("Gain_R","count"),Positive=("positive","sum"))
        print("COPY CONSISTENCY "+key+"\n"+small.to_csv(index=False))
    print("COPY ERRORS\n"+(result["errors"].to_csv(index=False) if len(result["errors"]) else "NONE"))
    print("Saved full audit:",result["output_dir"])


def plot_context(result, instrument, profile="relaxed_research", variant="PRICE_KALMAN", last_bars=70):
    """Visual diagnosis: raw candles, causal level/slope uncertainty, PVD and activity."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    d=result["daily"][(profile,instrument,variant)]
    d=d.loc[d.analysis_period].tail(int(last_bars))
    if d.empty:
        raise ValueError("No analysis-period bars for this instrument")
    x=np.arange(len(d));z=result["settings"]["confidence_z"]
    fig,axes=plt.subplots(4,1,figsize=(14,10),sharex=True,
                          gridspec_kw={"height_ratios":[3,1.5,1.2,1.2]})
    for i,b in enumerate(d.itertuples()):
        color="seagreen" if b.close>=b.open else "indianred"
        axes[0].vlines(i,b.low,b.high,color=color)
        axes[0].add_patch(Rectangle((i-.3,min(b.open,b.close)),.6,max(abs(b.close-b.open),1e-8),color=color))
        if b.low_activity or b.sudden_drop:
            for ax in axes:
                ax.axvspan(i-.5,i+.5,color="orange",alpha=.15)
    valid=d.kalman_ready.to_numpy(bool)
    axes[0].plot(x,np.where(valid,d.kalman_level,np.nan),color="blue",label="Causal Kalman level")
    for e in result["events"].loc[(result["events"].Instrument==instrument)&
            (result["events"].Profile==profile)&(result["events"].Variant==variant)].itertuples():
        match=np.flatnonzero(d.Date.eq(pd.Timestamp(e.Date)))
        if len(match):
            j=int(match[0]);axes[0].scatter(j,d.iloc[j].close,marker="^" if e.trade_sign==1 else "v",color="purple",s=60)
    slope=d.kalman_slope.where(valid);se=d.kalman_slope_se.where(valid)
    axes[1].plot(x,slope,color="blue",label="Slope")
    axes[1].fill_between(x,slope-z*se,slope+z*se,color="blue",alpha=.15,label=f"Conditional +/- {z:g} SE")
    axes[1].axhline(0,color="black",linewidth=.8)
    axes[2].plot(x,d.pvd_z,color="teal",label="Original PVD z")
    axes[2].axhline(0,color="grey",linewidth=.8)
    axes[3].bar(x,d.relative_volume,color="grey",alpha=.7,label="Relative volume")
    axes[3].axhline(result["settings"]["low_volume_ratio"],color="orange",linestyle="--",label="Low activity threshold")
    labels=["Price","Slope (price/bar)","PVD z","Volume / past median"]
    for ax,label in zip(axes,labels):
        ax.set_ylabel(label);ax.legend(loc="best",fontsize=8);ax.grid(alpha=.15)
    positions=np.unique(np.linspace(0,len(d)-1,min(10,len(d))).astype(int))
    axes[-1].set_xticks(positions, [d.iloc[j].Date.strftime("%d-%b") for j in positions],rotation=30)
    axes[0].set_title(f"{instrument} | {variant} | {profile}\nPurple: absorption expected direction; orange background: low/drop volume")
    fig.tight_layout();plt.show()
    return fig
