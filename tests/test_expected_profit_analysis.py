"""Focused synthetic checks: no market API calls and no notebook dependencies."""

import sys
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from market_analysis.finding_threshold.expected_profit_analysis import (
    analysis_date_range, analyze_expected_profit, build_v2_features,
    deduplicate_combined_signals, evaluate_signal_sizes,
    refresh_expected_profit_data, select_threshold_signals, summarize_expected_sizes,
    run_expected_profit_analysis, plot_expected_profit_heatmap,
    pool_expected_profit_analysis,
)


def bars(size=12):
    """Flat synthetic prices give exactly 2 price units of daily true range."""
    return pd.DataFrame({"Date": pd.bdate_range("2026-01-01", periods=size),
                         "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
                         "volume": 100.0, "buyvolume": 60.0, "sellvolume": 40.0})


class OutcomeTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {"atr_period": 2, "forward_days": 3, "stop_atr_mult": 1.5,
                    "required_move_ticks": 2}

    def event(self, data, side="BUY", index=2):
        out = evaluate_signal_sizes(data, 1.0, self.cfg)
        return out.loc[(out.signal_index == index) & (out.absorption_direction == side)].iloc[0]

    def test_profit_before_stop_is_success_and_later_extremes_ignored(self):
        data = bars()
        data.loc[3, "low"] = 96.0  # short reaches required profit before stop
        data.loc[4, "high"] = 104.0  # later hits stop at 103
        data.loc[4, "low"] = 80.0  # stop-bar low order unknown: do not count
        data.loc[5, "low"] = 50.0  # after exit: never count
        event = self.event(data)
        self.assertEqual(event.outcome, "SUCCESS")
        self.assertEqual(event.success_size_ticks, 4.0)
        self.assertTrue(event.stop_after_profit)
        self.assertTrue(event.profit_hit_first)
        self.assertEqual(event.profit_bar_offset, 1)
        self.assertEqual(event.exit_index, 4)

    def test_stop_before_profit_is_configured_loss(self):
        data = bars()
        data.loc[3, "high"] = 104.0
        data.loc[4, "low"] = 80.0
        event = self.event(data)
        self.assertEqual(event.outcome, "FAILURE")
        self.assertEqual(event.failure_size_ticks, 3.0)
        self.assertEqual(event.research_payoff_ticks, -3.0)
        self.assertFalse(event.profit_hit_first)

    def test_long_mirror_and_maximum_until_stop(self):
        data = bars()
        data.loc[3, "high"] = 104.0
        data.loc[4, "high"] = 107.0
        data.loc[5, "low"] = 96.0
        event = self.event(data, "SELL")
        self.assertEqual(event.position, "LONG")
        self.assertEqual(event.outcome, "SUCCESS")
        self.assertEqual(event.success_size_ticks, 7.0)
        self.assertEqual(event.maximum_profit_bar_offset, 2)

    def test_same_bar_first_profit_and_stop_conservatively_fails(self):
        data = bars()
        data.loc[3, ["high", "low"]] = [104.0, 95.0]
        event = self.event(data)
        self.assertEqual(event.outcome, "FAILURE")
        self.assertTrue(event.stop_bar_order_unknown)

    def test_maximum_no_stop_and_insufficient_move(self):
        data = bars()
        data.loc[4, "low"] = 93.0
        self.assertEqual(self.event(data).success_size_ticks, 7.0)
        unresolved = self.event(bars())
        self.assertEqual(unresolved.outcome, "UNRESOLVED")
        self.assertTrue(np.isnan(unresolved.failure_size_ticks))

    def test_censored_maximum_and_known_early_stop(self):
        data = bars(5)
        data.loc[3, "low"] = 95.0
        partial = self.event(data)
        self.assertEqual(partial.outcome, "CENSORED")
        self.assertTrue(partial.profit_hit_first)
        self.assertTrue(np.isnan(partial.research_payoff_ticks))
        data.loc[4, "high"] = 104.0
        self.assertEqual(self.event(data).outcome, "SUCCESS")
        self.assertEqual(self.event(data, index=3).outcome, "CENSORED")
        data.loc[3, "high"] = 104.0
        self.assertEqual(self.event(data).outcome, "FAILURE")

    def test_stop_multiplier_and_tick_conversion(self):
        self.cfg["stop_atr_mult"] = 2.0
        data = bars()
        data.loc[3, "high"] = 105.0
        out = evaluate_signal_sizes(data, 0.1, self.cfg)
        event = out.loc[(out.signal_index == 2) & (out.absorption_direction == "BUY")].iloc[0]
        self.assertEqual(event.failure_size_ticks, 40.0)

    def test_expected_value_counts_and_empty_classes(self):
        events = pd.DataFrame({"outcome": ["SUCCESS", "FAILURE", "UNRESOLVED", "CENSORED"],
                               "success_size_ticks": [5.0, np.nan, np.nan, np.nan],
                               "failure_size_ticks": [np.nan, 3.0, np.nan, np.nan],
                               "research_payoff_ticks": [5.0, -3.0, np.nan, np.nan],
                               "stop_or_horizon_pnl_ticks": [1.0, -3.0, 0.5, np.nan]})
        summary = summarize_expected_sizes(events)
        self.assertEqual(summary["expected_profit_ticks"], 1.0)
        self.assertEqual(summary["weighted_expected_profit_ticks"], 1.0)
        self.assertEqual(summary["resolved_fraction"], 0.5)
        self.assertEqual(summary["n_unresolved"], 1)
        self.assertEqual(summarize_expected_sizes(events.iloc[:1])["expected_profit_ticks"], 5.0)
        self.assertTrue(np.isnan(summarize_expected_sizes(events.iloc[:0])["expected_profit_ticks"]))


class FeatureAndCountingTests(unittest.TestCase):
    def variable_bars(self):
        data = bars(100)
        rng = np.random.default_rng(32)
        close = 100 + rng.normal(0, 0.4, len(data)).cumsum()
        data["close"] = close
        data["open"] = np.r_[close[0], close[:-1]]
        data["high"] = data[["open", "close"]].max(axis=1) + 1
        data["low"] = data[["open", "close"]].min(axis=1) - 1
        delta = rng.uniform(-40, 40, len(data))
        data["buyvolume"], data["sellvolume"] = (100 + delta) / 2, (100 - delta) / 2
        return data

    def test_same_size_baseline_excludes_current_constituents_and_future(self):
        data = self.variable_bars()
        cfg = {"normalize_features": False, "baseline_window": 4, "baseline_min_periods": 4,
               "atr_period": 2}
        k, t = 3, 40
        features = build_v2_features(data, [k], cfg)
        current = features.loc[features.signal_index == t].iloc[0]
        # Historical 3D endpoint indices are t-3,t-4,t-5,t-6.
        history = features.loc[features.signal_index.between(t - k - 3, t - k), "delta"]
        expected = (current.delta - history.median()) / (1.4826 * np.median(np.abs(history - history.median())))
        self.assertAlmostEqual(current.z_delta, expected)
        self.assertEqual(current.price_response, abs(data.at[t, "close"] - data.at[t-k, "close"]))
        truncated = build_v2_features(data.iloc[:t+1], [k], cfg)
        pd.testing.assert_series_equal(current, truncated.loc[truncated.signal_index == t].iloc[0])

    def test_longest_wins_across_dates_and_directions_without_outcome_peeking(self):
        candidates = pd.DataFrame({"window_start_index": [11, 10, 9, 7, 20],
                                   "signal_index": [11, 11, 11, 11, 20],
                                   "candle_length": [1, 2, 3, 5, 1],
                                   "absorption_direction": ["SELL", "BUY", "BUY", "SELL", "BUY"]})
        kept, dropped = deduplicate_combined_signals(candidates)
        self.assertEqual(kept.candle_length.tolist(), [5, 1])
        self.assertEqual(len(dropped), 3)
        self.assertEqual(dropped.suppressed_by_candle_length.tolist(), [5, 5, 5])
        # A later 5D candle suppresses the earlier 1D constituent signal too.
        earlier = pd.DataFrame({"window_start_index": [8, 7], "signal_index": [8, 11],
                                "candle_length": [1, 5]})
        self.assertEqual(deduplicate_combined_signals(earlier)[0].candle_length.tolist(), [5])

    def test_equal_length_tie_and_partial_overlap(self):
        signals = pd.DataFrame({"window_start_index": [1, 2, 5], "signal_index": [3, 4, 7],
                                "candle_length": [3, 3, 3]})
        kept, _ = deduplicate_combined_signals(signals)
        self.assertEqual(kept.signal_index.tolist(), [3, 7])

    def test_shared_gate_preserves_raw_delta_sign(self):
        frame = pd.DataFrame({"delta": [5, -5, -5, 5], "z_delta": [2, -2, 2, -2],
                              "z_price": [-1]*4, "atr": [1]*4})
        selected = select_threshold_signals(frame, 1.5, 0)
        self.assertEqual(selected.absorption_direction.tolist(), ["BUY", "SELL"])

    def test_custom_lengths_separate_and_combined_counts(self):
        data = self.variable_bars()
        cfg = {"baseline_window": 5, "baseline_min_periods": 5, "atr_period": 2,
               "volume_window": 2, "atr_window": 2, "forward_days": 3}
        args = dict(tick_size=0.1, candle_lengths=[1, 4], params=cfg,
                    delta_z_grid=[0.1, 1.0], price_z_grid=[1.0])
        separate = analyze_expected_profit(data, mode="separate", **args)
        combined = analyze_expected_profit(data, mode="combined", **args)
        self.assertEqual(set(separate["threshold_table"].analysis_group), {"1d", "4d"})
        for key, events in combined["event_ledgers"].items():
            occupied = set()
            for _, row in events.iterrows():
                interval = set(range(row.window_start_index, row.signal_index + 1))
                self.assertFalse(interval & occupied)
                occupied |= interval
            summary = combined["threshold_table"].loc[
                combined["threshold_table"].delta_z_threshold == key[1]].iloc[0]
            self.assertEqual(summary.n_candidates, len(events) + len(combined["suppressed_ledgers"][key]))
        daily = analyze_expected_profit(data, mode="separate", **{**args, "candle_lengths": [1]})
        self.assertEqual(set(daily["features"].candle_length), {1})

    def test_invalid_and_degenerate_data(self):
        data = bars()
        self.assertTrue(build_v2_features(data, [1], {"baseline_window": 2, "baseline_min_periods": 2})
                        .z_delta.isna().all())
        with self.assertRaises(ValueError):
            build_v2_features(data, [1, 1])
        with self.assertRaises(ValueError):
            build_v2_features(pd.concat([data, data.iloc[:1]]), [1])
        with self.assertRaises(ValueError):
            select_threshold_signals(pd.DataFrame(), 0, 0)
        with self.assertRaises(ValueError):
            build_v2_features(data.iloc[:0], [1])

    def test_threshold_specific_priority_changes_when_long_signal_does_not_qualify(self):
        frame = pd.DataFrame({"window_start_index": [11, 7], "signal_index": [11, 11],
                              "candle_length": [1, 5], "delta": [5, -5],
                              "z_delta": [3, -1.5], "z_price": [-1, -1], "atr": [1, 1]})
        loose, _ = deduplicate_combined_signals(select_threshold_signals(frame, 1, 0))
        strict, _ = deduplicate_combined_signals(select_threshold_signals(frame, 2, 0))
        self.assertEqual(loose.candle_length.tolist(), [5])
        self.assertEqual(strict.candle_length.tolist(), [1])

    @unittest.skipUnless(find_spec("matplotlib"), "Optional matplotlib is not installed in this runtime")
    def test_optional_heatmap_smoke(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        result = analyze_expected_profit(self.variable_bars(), 0.1, [1, 3],
            params={"baseline_window": 5, "baseline_min_periods": 5},
            delta_z_grid=[1.0, 2.0], price_z_grid=[0.0, -0.5])
        figure = plot_expected_profit_heatmap(result)
        figure.canvas.draw()
        self.assertEqual(len(figure.axes), 2)  # grid plus colorbar
        plt.close(figure)


class CoverageTests(unittest.TestCase):
    def test_june_to_october_and_year_boundary(self):
        start, end = analysis_date_range("2026-10-05")
        self.assertEqual(str(start.date()), "2026-06-01")
        self.assertEqual(str(end.date()), "2026-10-04")
        start, _ = analysis_date_range("2026-01-15")
        self.assertEqual(str(start.date()), "2025-09-01")

    def test_refresh_reconciles_stale_internal_rows_and_clips_today(self):
        fetched = bars(180)
        fetched["Date"] = pd.bdate_range("2026-03-01", periods=180)
        # Omit one real API session; an old cache row for that date must not be
        # claimed authoritative merely because it lies between cached min/max.
        missing_date = fetched.at[90, "Date"]
        old = fetched.copy()
        fetched = fetched.drop(index=90)
        calls = []

        def fake_fetch(**kwargs):
            calls.append(kwargs)
            return fetched

        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "sm"
            folder.mkdir()
            old.to_csv(folder / "SMZ26-F27_1d.csv", index=False)
            data, coverage = refresh_expected_profit_data(
                "SMZ26-F27", "SM", [1, 2, 3, 5], as_of="2026-10-05",
                fetcher=fake_fetch, cache_dir=temporary)
            cached = pd.read_csv(folder / "SMZ26-F27_1d.csv")
            self.assertNotIn(missing_date.strftime("%Y-%m-%d"), cached.Date.str[:10].tolist())
            self.assertTrue((data.Date < pd.Timestamp("2026-10-05", tz="UTC")).all())
            self.assertEqual(calls[0]["timeinterval"], "1D")
            self.assertEqual(str(coverage["analysis_start"].date()), "2026-06-01")
            refresh_expected_profit_data("SMZ26-F27", "SM", [1], as_of="2026-10-05",
                                         fetcher=fake_fetch, cache_dir=temporary)
            self.assertEqual(len(calls), 2)  # a complete cache doesn't skip verification

    def test_api_failure_never_claims_fresh_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "unverified"):
                refresh_expected_profit_data("SMZ26-F27", "SM", [1], as_of="2026-10-05",
                                             fetcher=lambda **kw: None, cache_dir=temporary)

    def test_product_runner_uses_configurable_lengths_and_current_universe(self):
        data = FeatureAndCountingTests().variable_bars()
        with tempfile.TemporaryDirectory() as temporary:
            result = run_expected_profit_analysis("SM", [1, 4], mode="combined", front_n=1,
                as_of="2026-05-20", fetcher=lambda **kw: data, cache_dir=temporary,
                delta_z_grid=[1.0], price_z_grid=[0.0])
        self.assertEqual(result["errors"], {})
        self.assertEqual(len(result["instruments"]), 2)  # one spread and one jumbo
        self.assertEqual(result["candle_lengths"], [1, 4])
        for entry in result["instruments"].values():
            self.assertEqual(entry["metadata"]["candle_lengths"], [1, 4])
        pooled = pool_expected_profit_analysis(result)
        self.assertEqual(pooled["metadata"]["missing_instruments"], [])
        self.assertEqual(pooled["threshold_table"].iloc[0].n_instruments, 2)


class PoolingTests(unittest.TestCase):
    def synthetic_run(self):
        """Unequal counts distinguish proper event pooling from averaging means."""
        key = ("combined", 1.0, -0.5)
        results = {}
        for instrument, structure, payoffs in (("A", "spread", [10.0]), ("B", "jumbo", [-2.0]*3)):
            events = pd.DataFrame({
                "outcome": ["SUCCESS" if p > 0 else "FAILURE" for p in payoffs],
                "absorption_direction": ["BUY"]*len(payoffs),
                "success_size_ticks": [p if p > 0 else np.nan for p in payoffs],
                "failure_size_ticks": [-p if p < 0 else np.nan for p in payoffs],
                "research_payoff_ticks": payoffs, "stop_or_horizon_pnl_ticks": payoffs})
            results[instrument] = {
                "event_ledgers": {key: events}, "suppressed_ledgers": {key: events.iloc[:0]},
                "threshold_table": pd.DataFrame([{"analysis_group": key[0], "delta_z_threshold": key[1],
                                                  "price_z_threshold": key[2], "n_candidates": len(events)}]),
                "metadata": {"structure": structure, "product": "SM", "params": {"min_signal_count": 1}},
            }
        return {"universe": {"spread": ["A"], "jumbo": ["B"]}, "instruments": results}

    def test_pools_counts_and_signed_sizes_not_instrument_means(self):
        pooled = pool_expected_profit_analysis(self.synthetic_run())
        row = pooled["threshold_table"].iloc[0]
        self.assertEqual(row.n_signals, 4)
        self.assertEqual(row.n_success, 1)
        self.assertEqual(row.n_failure, 3)
        self.assertEqual(row.expected_profit_ticks, 1.0)  # (10-2-2-2)/4; NOT (10-2)/2
        self.assertEqual(row.n_instruments, 2)
        self.assertEqual(set(pooled["event_ledgers"][("combined", 1.0, -0.5)].instrument), {"A", "B"})

    def test_structure_and_incomplete_universe_are_explicit(self):
        run = self.synthetic_run()
        spread = pool_expected_profit_analysis(run, structure="spread")
        self.assertEqual(spread["threshold_table"].iloc[0].expected_profit_ticks, 10.0)
        del run["instruments"]["B"]
        pooled = pool_expected_profit_analysis(run)
        self.assertEqual(pooled["metadata"]["missing_instruments"], ["B"])
        with self.assertRaises(ValueError):
            pool_expected_profit_analysis(run, structure="jumbo")


if __name__ == "__main__":
    unittest.main()
