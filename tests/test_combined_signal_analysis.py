"""Offline checks for threshold provenance, timing, counting and remaining moves."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from combined_signal_analysis import combined_signal_notebook as cs


def context_fixture(stop_hit=False, length=15):
    dates = pd.bdate_range("2026-06-01", periods=length, tz="UTC")
    close = 100 + np.arange(length, dtype=float)
    raw = pd.DataFrame({"Date": dates, "open": close - .2, "close": close,
                        "high": close + 1, "low": close - 1,
                        "volume": 100, "buyvolume": 60, "sellvolume": 40})
    raw.at[6, "high"] = 200  # giant move on marker day must NOT count after marker
    daily = raw.copy()
    daily["compression_score"] = .1
    daily["pvd_z"] = 0.
    daily["pvd_direction"] = "NEUTRAL"
    daily["pvd_signal"] = "NORMAL"
    for index, name, direction, z in [(1, "DOWN_TREND_EXHAUSTION", "DOWN", -2),
                                      (3, "DOWN_TREND_EXHAUSTION", "DOWN", -2),
                                      (4, "UP_TREND_EXHAUSTION", "UP", -2),
                                      (6, "UP_PRICE_EXPANSION", "UP", 2)]:
        daily.loc[index, ["pvd_signal", "pvd_direction", "pvd_z"]] = [name, direction, z]
    daily.at[3, "compression_score"] = .3
    event = {"Date": dates[4], "window_start_index": 3, "signal_index": 4, "candle_length": 2,
             "absorption_direction": "SELL", "entry_price": 104., "stop_loss": 90.,
             "stop_hit": stop_hit, "exit_index": 6 if stop_hit else 14,
             "profit_hit_first": True, "profit_bar_offset": 1, "outcome": "SUCCESS"}
    threshold = {"instrument": "SMZ26-F27", "product": "SM", "structure": "spread"}
    return raw, daily, pd.DataFrame([event]), threshold


class ThresholdTests(unittest.TestCase):
    def test_actual_workbook_sources_and_missing_abs_row(self):
        thresholds = cs.load_thresholds(settings={"as_of": "2026-10-07"})
        sm = thresholds.loc[thresholds.instrument == "SMZ26-F27"].iloc[0]
        self.assertEqual(sm.pvd_exhaustion_z, -1.1)  # last populated row; date is just a label
        self.assertIn("28-09-264", sm.exhaustion_donor_dates)
        self.assertEqual(sm.pvd_expansion_z, 1.3)
        self.assertEqual(sm.exhaustion_method, "exact")
        missing = thresholds.loc[thresholds.instrument == "BOZ26-H27"].iloc[0]
        self.assertEqual(missing.status, "error")
        self.assertGreater(len(thresholds.attrs["manual_issues"]), 0)

    def test_two_closest_never_cross_product_or_type(self):
        table = pd.DataFrame([
            {"product": "BO", "instrument": "BOK27-N27", "structure_type": "spread", "threshold_zscore": -1.3},
            {"product": "BO", "instrument": "BOH27-K27", "structure_type": "spread", "threshold_zscore": -1.2},
            {"product": "BO", "instrument": "BOK27-Q27", "structure_type": "jumbo", "threshold_zscore": -9},
            {"product": "SM", "instrument": "SMN27-Q27", "structure_type": "spread", "threshold_zscore": -9},
        ])
        table["date"], table["sheet"] = pd.Timestamp("2026-09-22"), "pvd_exhaustion"
        value, source = cs.resolve_pvd_threshold(table, "BON27-Q27", "BO", "spread")
        self.assertAlmostEqual(value, -1.25)
        self.assertEqual(source["donors"], "BOK27-N27; BOH27-K27")
        with self.assertRaisesRegex(ValueError, "TWO"):
            cs.resolve_pvd_threshold(table.iloc[:1], "BON27-Q27", "BO", "spread")


class TimingTests(unittest.TestCase):
    def test_phases_direction_alignment_and_marker_day_exclusion(self):
        raw, daily, events, threshold = context_fixture()
        ledger, markers, occurrence = cs.make_event_context(raw, daily, events, threshold,
            {**cs.SETTINGS, "before_bars": 3}, {"required_move_ticks": 2}, .1)
        row = ledger.iloc[0]
        self.assertTrue(row.before_exhaustion_aligned)
        self.assertTrue(row.during_exhaustion_aligned)
        self.assertTrue(row.during_exhaustion_opposite)
        self.assertTrue(row.during_compression)
        self.assertTrue(row.after_expansion_aligned)
        self.assertFalse(row.after_expansion_opposite)
        self.assertIn("during:compression", row.entry_signature)
        self.assertNotIn("expansion", row.entry_signature)
        marker = markers.loc[markers.signal_type == "expansion_aligned"].iloc[0]
        self.assertEqual(marker.delay_bars, 2)
        self.assertEqual(marker.remaining_bars, 8)
        self.assertAlmostEqual(marker.mfe_ticks, (115 - 106) / .1)
        self.assertAlmostEqual(row.forward_mfe_ticks, (200 - 104) / .1)
        self.assertTrue(marker.profit_already_hit)
        self.assertFalse(marker.stop_already_hit)

    def test_stopped_trade_is_excluded_from_remaining_active_opportunity(self):
        raw, daily, events, threshold = context_fixture(stop_hit=True)
        _, markers, _ = cs.make_event_context(raw, daily, events, threshold,
            {**cs.SETTINGS, "before_bars": 3}, {"required_move_ticks": 2}, .1)
        marker = markers.loc[markers.signal_type == "expansion_aligned"].iloc[0]
        self.assertTrue(marker.stop_already_hit)
        self.assertTrue(np.isnan(marker.remaining_active_trade_mfe_ticks))
        summary = cs.summarize_landmarks(markers)
        self.assertEqual(summary.iloc[0].n_original_trade_active, 0)

    def test_incomplete_after_phase_is_unknown_not_absence(self):
        raw, daily, events, threshold = context_fixture(length=10)
        ledger, markers, _ = cs.make_event_context(raw, daily, events, threshold,
            {**cs.SETTINGS, "before_bars": 3}, {"required_move_ticks": 2}, .1)
        self.assertFalse(ledger.iloc[0].full_forward_window)
        self.assertTrue(pd.isna(ledger.iloc[0].after_expansion_aligned))
        self.assertEqual(ledger.iloc[0].post_signature, "after:unknown")
        self.assertTrue(markers.mfe_ticks.isna().all())

    def test_last_day_marker_has_no_remaining_observation(self):
        raw, daily, events, threshold = context_fixture()
        daily.loc[6, ["pvd_signal", "pvd_z", "pvd_direction"]] = ["NORMAL", 0., "NEUTRAL"]
        daily.loc[14, ["pvd_signal", "pvd_z", "pvd_direction"]] = ["UP_PRICE_EXPANSION", 2., "UP"]
        _, markers, _ = cs.make_event_context(raw, daily, events, threshold,
            {**cs.SETTINGS, "before_bars": 3}, {"required_move_ticks": 2}, .1)
        self.assertEqual(markers.iloc[0].remaining_bars, 0)
        self.assertTrue(np.isnan(markers.iloc[0].mfe_ticks))


class IntegrationTests(unittest.TestCase):
    def test_auto_mode_keeps_analysis_and_stops_repeating_auth_failure(self):
        thresholds = cs.load_thresholds(settings={"as_of": "2026-10-07"})
        thresholds = thresholds.loc[thresholds.instrument.isin(["BOF27-H27", "BOH27-K27"])].copy()
        with tempfile.TemporaryDirectory() as directory, patch.object(
            cs.ohlc, "getData_OHLC_v2", side_effect=cs.ohlc.OHLCFetchError("OHLC API HTTP 401", 401)
        ) as api:
            result = cs.run_combined_signal_analysis(settings={"data_mode": "auto", "as_of": "2026-10-07", "output_dir": directory}, threshold_audit=thresholds)
        self.assertEqual(api.call_count, 1)
        self.assertGreater(len(result["events"]), 0)
        self.assertTrue(result["errors"].empty)
        self.assertTrue(result["coverage"].cache_fallback_used.all())
        self.assertFalse(result["coverage"].target_date_reached.any())

    def test_cached_data_distributions_and_resolved_opportunity_reconcile(self):
        with tempfile.TemporaryDirectory() as directory:
            result = cs.run_combined_signal_analysis(settings={"data_mode": "cache", "as_of": "2026-10-07", "output_dir": directory})
            self.assertGreater(len(result["events"]), 0)
            self.assertFalse(result["events"].event_id.duplicated().any())
            self.assertEqual(set(result["errors"].instrument), {"BOZ26-H27"})
            for scope, report in result["reports"].items():
                self.assertAlmostEqual(report["entry_distribution"].pattern_probability.sum(), 1.)
                self.assertAlmostEqual(report["joint_distribution"].pattern_probability.sum(), 1.)
                events = result["events"].loc[result["events"].instrument == scope] if not scope.startswith("POOLED_") else result["events"].loc[result["events"]["product"] == scope[7:]]
                resolved = events.loc[events.outcome.isin(["SUCCESS", "FAILURE"])]
                baseline = report["baseline"].iloc[0]
                self.assertEqual(baseline.n_resolved, len(resolved))
                if len(resolved):
                    self.assertAlmostEqual(baseline.expected_opportunity_ticks, resolved.research_payoff_ticks.mean())
            self.assertTrue((Path(directory) / "2026-10-07/manual_input_issues.csv").exists())
            # Export keeps signal-time and future-observation columns explicitly separate.
            self.assertTrue((Path(directory) / "2026-10-07/after_signal_landmarks.csv").exists())

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib unavailable in bundled runtime")
    def test_candlestick_plot_smoke(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        threshold = cs.load_thresholds(settings={"as_of": "2026-10-07"}).iloc[0]
        raw = pd.read_csv(cs.config.DATA_DIR / "sm/SMZ26-F27_1d.csv")
        result = cs.analyze_combined_instrument(raw, threshold)
        if len(result["events"]):
            figure = cs.plot_event(result, result["events"].iloc[0].event_id)
            self.assertEqual(len(figure.axes), 4)
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
