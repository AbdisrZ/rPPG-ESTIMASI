from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[2]


def muat(path=None) -> dict:
    p = Path(path) if path else RAIZ / "config" / "default.yaml"
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)
