"""Offline Wheat inclusion and identical-data experiment checks."""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
from market_analysis.combined_signal_analysis import relaxed_research as research


class RelaxedTests(unittest.TestCase):
    def test_wheat_universe(self):
        universe=research.build_universe(as_of="2026-10-09")
        self.assertEqual(len(universe),30)
        wheat=universe.loc[universe["product"].eq("W")]
        self.assertEqual(len(wheat),10)
        self.assertEqual(wheat.iloc[0].instrument,"WZ26-H27")
        self.assertTrue(wheat.tick_size.eq(.25).all())

    def test_profiles_and_custom_validation(self):
        universe=research.build_universe(products=["W"],front_n=1,structures=["spread"],as_of="2026-10-09")
        a=research.PROFILES["reference_research"]
        b=research.PROFILES["relaxed_research"]
        self.assertLess(b["abs_volume_z"],a["abs_volume_z"])
        self.assertGreater(b["abs_price_z"],a["abs_price_z"])
        table=research.research_thresholds(universe,"relaxed",b)
        self.assertEqual(table.iloc[0].threshold_source,"experimental_prespecified_not_validated")
        with self.assertRaises(ValueError):
            research.build_universe(front_n=0)

    def test_wheat_passes_through_full_pipeline_without_api(self):
        universe=research.build_universe(products=["W"],front_n=1,structures=["spread"],as_of="2026-10-09")
        # Test fixture ONLY: reuse real OHLC shape to exercise code under W naming.
        # This is never saved as Wheat market data and is NOT a Wheat result.
        fixture=pd.read_csv(ROOT/"data/bo/BOZ26-F27_1d.csv")
        raw={universe.iloc[0].instrument:fixture}
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            with patch.object(research.core.ohlc,"getData_OHLC_v2",side_effect=AssertionError("No API in offline test")):
                result=research.run_relaxed_research(products=["W"],front_n=1,structures=["spread"],
                    as_of="2026-10-09",output_dir=folder,raw_by_instrument=raw)
            self.assertEqual(len(result["baseline"]),2)
            self.assertTrue(result["baseline"].Product.eq("W").all())
            self.assertTrue(result["baseline"].Loaded.eq(1).all())
            self.assertTrue(result["errors"].empty)
            self.assertTrue(result["matrix"].Product.eq("W").all())
            self.assertFalse(result["matrix"].empty)
            runs=list(result["runs"].values())
            name=universe.iloc[0].instrument
            pd.testing.assert_frame_equal(runs[0]["instruments"][name]["raw"],runs[1]["instruments"][name]["raw"])


if __name__=="__main__":
    unittest.main()
