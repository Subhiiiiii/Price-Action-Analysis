"""Offline notebook smoke check; no network, plots or output overwrites."""
import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from combined_signal_analysis import combined_signal_notebook as core
import numpy as np

notebook = json.loads((ROOT / "combined_signal_analysis/absorption_pvd_hypotheses.ipynb").read_text(encoding="utf-8"))
sources = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
for i, source in enumerate(sources):
    compile(source, f"notebook_cell_{i}", "exec")
instruments = {}
for name in ["SMZ26-F27", "BOZ26-F27"]:
    # Fixed smoke-test cutoffs avoid requiring access to an open Excel workbook.
    threshold = dict(instrument=name, product=name[:2], structure="spread",
                     abs_volume_z=1., abs_price_z=.5,
                     pvd_exhaustion_z=-1.1, pvd_expansion_z=1.3)
    raw = pd.read_csv(ROOT / "data" / threshold["product"].lower() / (name + "_1d.csv"))
    instruments[name] = core.analyze_combined_instrument(
        raw, threshold, analysis_start="2026-06-01", analysis_end="2026-10-06")
namespace = dict(pd=pd, np=np, RUN={"instruments": instruments}, FLAT_BAND_TICKS=0., MIN_SAMPLE=10,
                 view=lambda *a, **k: None)
for source in sources:
    if source.startswith(("PVD_NAMES", "frequency,", "def summarize_prediction", "def candidate_prediction")):
        exec(source, namespace)
events = namespace["EVENTS"]
assert len(events) > 0
for _, group in namespace["COMBINATIONS"].groupby(["instrument", "phase"]):
    assert np.isclose(group.pct_all_absorptions.sum(), 100)
h = namespace["HYPOTHESES"]
assert h.accuracy_pct.dropna().between(0, 100).all()
assert h.n_evaluated.le(h.n_predictions).all()
assert namespace["DIRECTION"]["DOWN_PRICE_EXPANSION"] == 1
assert namespace["DIRECTION"]["UP_TREND_EXHAUSTION"] == -1
candidate = namespace["candidate_prediction"]
row = pd.Series(dict(before_state="UP", during_state="DOWN", before_consensus=1,
                     during_consensus=-1, abs_prediction=-1))
assert candidate(row) == (-1, "AGREE")
row["during_consensus"] = 1
assert candidate(row)[1] == "CONFLICT"
print(f"PASS: {len(sources)} cells compile; {len(events)} cached events; frequency, direction and conflict checks passed.")
