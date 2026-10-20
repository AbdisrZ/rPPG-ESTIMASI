from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[2]


def _gabung(dasar: dict, timpa: dict) -> dict:
    hasil = dict(dasar)
    for k, v in timpa.items():
        hasil[k] = _gabung(hasil[k], v) if isinstance(v, dict) and isinstance(hasil.get(k), dict) else v
    return hasil


def muat(path=None, lokal=None) -> dict:
    """Muat config. Tanpa `path`: default.yaml lalu ditimpa config/lokal.yaml bila ada."""
    if path:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    with open(RAIZ / "config" / "default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    p_lokal = Path(lokal) if lokal else RAIZ / "config" / "lokal.yaml"
    if p_lokal.exists():
        with open(p_lokal, encoding="utf-8") as f:
            cfg = _gabung(cfg, yaml.safe_load(f) or {})
    return cfg
