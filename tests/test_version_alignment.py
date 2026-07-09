from pathlib import Path
import re
import vetgraph

def test_version_matches_pyproject():
    pp = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pp.read_text(encoding="utf-8")
    m = re.search(r'version\s*=\s*"([^"]+)"', text)
    assert m
    assert vetgraph.__version__ == m.group(1)
