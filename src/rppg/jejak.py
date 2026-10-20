"""Jejak reproduksibilitas untuk meta hasil: commit, hash config, versi Python dan paket."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime
from importlib import metadata

from .config import RAIZ

PAKET = ["numpy", "scipy", "pandas", "matplotlib", "opencv-contrib-python", "mediapipe"]


def _git(*arg):
    try:
        return subprocess.run(["git", *arg], cwd=RAIZ, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def hash_commit():
    return _git("rev-parse", "--short", "HEAD")


def hash_config(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def versi_paket() -> dict:
    hasil = {}
    for p in PAKET:
        try:
            hasil[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            hasil[p] = None
    return hasil


def meta_dasar(cfg: dict) -> dict:
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"waktu": datetime.now().isoformat(timespec="seconds"), "commit": hash_commit(),
            "ada_perubahan_belum_commit": None if status is None else bool(status),
            "hash_config": hash_config(cfg), "config": cfg,
            "python": platform.python_version(), "os": platform.platform(), "paket": versi_paket()}
