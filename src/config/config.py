from pathlib import Path
import yaml  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

with open(PROJECT_ROOT / "config.yaml") as f:
    _cfg = yaml.safe_load(f)

PATHS = {name: PROJECT_ROOT / rel_path for name, rel_path in _cfg["paths"].items()}
