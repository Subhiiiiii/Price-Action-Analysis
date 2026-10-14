"""Meaningful offline checks: denominators, no-move retention and causal context."""
import json
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from combined_signal_analysis import combined_signal_notebook as core
from combined_signal_analysis import multi_signal_hypotheses as multi


class MultiSignalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        instruments = {}
        for name in ("SMZ26-F27","BOZ26-F27"):
            threshold = dict(instrument=name,product=name[:2],structure="spread",abs_volume_z=1.,abs_price_z=.5,
                             pvd_exhaustion_z=-1.1,pvd_expansion_z=1.3)
            raw = pd.read_csv(ROOT/"data"/name[:2].lower()/(name+"_1d.csv"))
            instruments[name] = core.analyze_combined_instrument(raw,threshold,analysis_start="2026-06-01",analysis_end="2026-10-06")
        cls.run_data = dict(instruments=instruments)

    def test_real_cache_and_partition_denominators(self):
        output = multi.test_multi_signal_hypotheses(self.run_data,save=False)
        matrix = output["matrix"]
        self.assertEqual(len(matrix),12)
        self.assertTrue((matrix.Selected+matrix.Comparison).eq(matrix.Eligible).all())
        self.assertTrue(matrix.Eligible.le(matrix.Total_abs).all())
        self.assertTrue(matrix.Resolved_S.le(matrix.Selected).all())
        self.assertTrue(matrix.Resolved_C.le(matrix.Comparison).all())
        self.assertTrue(np.allclose(matrix.Gain_ticks,matrix.Ticks_S-matrix.Ticks_C,equal_nan=True))
        for _, group in output["groups"].iterrows():
            self.assertEqual(group.signals,group.resolved+group.unresolved+group.censored)
        self.assertTrue(output["events"].Date.ge(pd.Timestamp("2026-06-01",tz="UTC")).all())

    def test_unresolved_no_move_included_only_in_movement(self):
        events = pd.DataFrame(dict(outcome=["SUCCESS","FAILURE","UNRESOLVED","CENSORED"],
                                   research_payoff_ticks=[4.,-2.,np.nan,np.nan],
                                   full_forward=[True,True,True,False],
                                   forward_move_ticks=[5.,3.,0.,np.nan],forward_mfe_ticks=[4.,1.,0.,np.nan]))
        stat = multi._stats(events)
        self.assertEqual(stat["resolved"],2)
        self.assertEqual(stat["success_pct"],50)
        self.assertEqual(stat["expected_ticks"],1)
        self.assertEqual(stat["full_forward"],3)
        self.assertEqual(stat["median_move_ticks"],3)

    def test_features_do_not_read_future_prices(self):
        result = self.run_data["instruments"]["SMZ26-F27"]
        first = result["events"].iloc[[0]]
        end = int(first.iloc[0].signal_index)
        original = {**result,"events":first}
        changed = {**original,"raw":result["raw"].copy(),"daily":result["daily"].copy()}
        for frame in (changed["raw"],changed["daily"]):
            for column in ("open","high","low","close"):
                frame.loc[frame.index>end,column] += 1000
        a,_ = multi.extract_features({"instruments":{"SMZ26-F27":original}})
        b,_ = multi.extract_features({"instruments":{"SMZ26-F27":changed}})
        features = ["rejection","compression","before_exhaustion_state","during_exhaustion_state",
                    "expansion_sequence","recovery_fraction","relative_volume"]
        pd.testing.assert_frame_equal(a[features],b[features])

    def test_notebook_cells_compile(self):
        notebook = json.loads((ROOT/"combined_signal_analysis/multi_signal_tests.ipynb").read_text(encoding="utf-8"))
        for i,cell in enumerate(notebook["cells"]):
            if cell["cell_type"]=="code":
                compile("".join(cell["source"]),f"cell_{i}","exec")


if __name__=="__main__":
    unittest.main()
