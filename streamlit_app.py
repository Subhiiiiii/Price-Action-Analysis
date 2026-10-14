# Root entrypoint for Streamlit Community Cloud
import sys
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_DOCS = _ROOT.parent
for p in [str(_ROOT), str(_DOCS)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Create 'market_analysis' virtual package shim for environments
# where the repo is cloned under a different directory name (e.g. Streamlit Cloud)
if "market_analysis" not in sys.modules:
    pkg = types.ModuleType("market_analysis")
    pkg.__path__ = [str(_ROOT)]
    sys.modules["market_analysis"] = pkg

# Execute the application located in finding_threshold
app_path = _ROOT / "finding_threshold" / "streamlit_app.py"
with open(app_path, "r", encoding="utf-8") as f:
    code = f.read()

exec(code, globals())
