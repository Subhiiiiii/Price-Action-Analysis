# Root entrypoint for Streamlit Community Cloud
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_DOCS = _ROOT.parent
for p in [str(_DOCS), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Execute the application located in finding_threshold
app_path = _ROOT / "finding_threshold" / "streamlit_app.py"
with open(app_path, "r", encoding="utf-8") as f:
    code = f.read()

exec(code, globals())
