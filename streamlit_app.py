# Root entrypoint for Streamlit Community Cloud
import sys
import os
import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_DOCS = _ROOT.parent
for p in [str(_ROOT), str(_DOCS)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Create a filesystem symlink on Linux/Streamlit Cloud if possible
link_path = _ROOT / "market_analysis"
if not link_path.exists():
    try:
        os.symlink(str(_ROOT), str(link_path), target_is_directory=True)
    except Exception:
        pass

# Ensure 'market_analysis' package spec exists in sys.modules
if "market_analysis" not in sys.modules or not getattr(sys.modules["market_analysis"], "__spec__", None):
    init_file = _ROOT / "__init__.py"
    if not init_file.exists():
        init_file.write_text("# market_analysis\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location("market_analysis", str(init_file), submodule_search_locations=[str(_ROOT)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["market_analysis"] = pkg
    spec.loader.exec_module(pkg)

# Execute the application located in finding_threshold
app_path = _ROOT / "finding_threshold" / "streamlit_app.py"
with open(app_path, "r", encoding="utf-8") as f:
    code = f.read()

exec(code, globals())
