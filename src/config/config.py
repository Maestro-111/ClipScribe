from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

with open(PROJECT_ROOT / "config.yaml") as f:
    _cfg = yaml.safe_load(f)