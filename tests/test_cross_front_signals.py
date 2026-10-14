"""Offline checks for timing, coverage and unique anchor denominators."""
import sys
import unittest
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from combined_signal_analysis.cross_front_analysis import _compare, SIGNALS, analyze_cross_front_signals


class CrossFrontTests(unittest.TestCase):
    def test_actual_cached_bars_offline(self):
        from combined_signal_analysis import combined_signal_notebook as cs
        thresholds = cs.load_thresholds()
        results = {}
        for name in ["SMZ26-F27", "BOZ26-F27"]:
            row = thresholds.loc[thresholds.instrument.eq(name)].iloc[0].to_dict()
            raw = pd.read_csv(cs.PROJECT_ROOT / "data" / row["product"].lower() / (name + "_1d.csv"))
            results[name] = cs.analyze_combined_instrument(
                raw, row, analysis_start="2026-06-01", analysis_end="2026-10-06")
        output = analyze_cross_front_signals({"instruments": results}, save=False)
        self.assertEqual(len(output["pair_table"]), len(SIGNALS) * 4)
        self.assertGreater(len(output["signal_ledger"]), 0)
        self.assertTrue(output["signal_ledger"].Date.ge(pd.Timestamp("2026-06-01", tz="UTC")).all())
        rates = output["product_summary"].match_ratio.dropna()
        self.assertTrue(rates.between(0, 1).all())

    def fixture(self):
        dates = list(pd.bdate_range("2026-10-01", periods=8, tz="UTC"))
        info = {"SMZ26-F27": {"product": "SM", "structure": "spread", "coordinates": (1, 2)},
                "SMF27-H27": {"product": "SM", "structure": "spread", "coordinates": (2, 4)},
                "SMH27-K27": {"product": "SM", "structure": "spread", "coordinates": (4, 6)},
                "BOZ26-F27": {"product": "BO", "structure": "spread", "coordinates": (1, 2)}}
        ready = {(inst, sig): set(dates) for inst in info for sig in SIGNALS}
        rows = [(i, v["product"], "ABS_BUY", dates[2]) for i, v in info.items()]
        ledger = pd.DataFrame(rows, columns=["instrument", "product", "signal", "Date"])
        return ledger, ready, info, dates

    def test_many_partner_matches_count_anchor_once(self):
        args = self.fixture()
        out = _compare(*args, [0], [["SM", "BO"]])
        sm = out["product_summary"].query("scope == 'same_product:SM' and signal == 'ABS_BUY'").iloc[0]
        self.assertEqual(sm.eligible_anchor_signals, 3)
        self.assertEqual(sm.matched_anchor_signals, 3)
        self.assertEqual(sm.match_ratio, 1)
        related = out["product_summary"].query("scope == 'related_products:BO_SM' and signal == 'ABS_BUY'").iloc[0]
        self.assertEqual(related.matched_anchor_signals, 4)

    def test_weekend_uses_trading_bars_and_signal_side_is_exact(self):
        ledger, ready, info, dates = self.fixture()
        ledger.loc[ledger.instrument == "SMZ26-F27", "Date"] = dates[1]  # Friday
        ledger.loc[ledger.instrument == "SMH27-K27", "signal"] = "ABS_SELL"
        out = _compare(ledger, ready, info, dates, [0, 1], [])
        pair = out["pair_table"].query("instrument_a == 'SMF27-H27' and instrument_b == 'SMZ26-F27' and signal == 'ABS_BUY'")
        self.assertEqual(pair.query("within_trading_bars == 0").iloc[0].matched_signals_a, 0)
        self.assertEqual(pair.query("within_trading_bars == 1").iloc[0].matched_signals_a, 1)
        self.assertTrue(out["matches"].absolute_gap_trading_bars.eq(1).all())

    def test_unscored_partner_window_is_not_a_failed_match(self):
        ledger, ready, info, dates = self.fixture()
        ready["SMZ26-F27", "ABS_BUY"].remove(dates[3])
        out = _compare(ledger, ready, info, dates, [1], [])
        row = out["pair_table"].query("instrument_a == 'SMF27-H27' and instrument_b == 'SMZ26-F27' and signal == 'ABS_BUY'").iloc[0]
        self.assertEqual(row.total_signals_a, 1)
        self.assertEqual(row.eligible_signals_a, 0)
        self.assertTrue(pd.isna(row.match_ratio_a))


if __name__ == "__main__":
    unittest.main()
