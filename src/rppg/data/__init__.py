"""Pemuat dataset dengan antarmuka sama: Rekaman.frames() -> iterator (frame RGB uint8, t detik),
Rekaman.referensi() -> (t_ref detik, hr_ref BPM). t video dan t referensi memakai nol yang sama.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np


class DatasetTidakAda(RuntimeError):
    pass


class FormatTidakSesuai(RuntimeError):
    pass


@dataclass
class Rekaman:
    dataset: str
    id: str
    frames: Callable[[], Iterator[tuple[np.ndarray, float]]]
    referensi: Callable[[], tuple[np.ndarray, np.ndarray]]


def muat_dataset(nama: str, root) -> list[Rekaman]:
    from . import pure, ubfc

    pemuat = {"ubfc": ubfc.daftar, "pure": pure.daftar}
    if nama not in pemuat:
        raise ValueError(f"dataset tidak dikenal: {nama}")
    if not root:
        raise DatasetTidakAda(f"jalur dataset '{nama}' belum diisi (config/lokal.yaml, kunci dataset.{nama})")
    root = Path(root)
    if not root.is_dir():
        raise DatasetTidakAda(f"folder dataset '{nama}' tidak ditemukan: {root}")
    rek = pemuat[nama](root)
    if not rek:
        raise DatasetTidakAda(f"tidak ada rekaman '{nama}' yang dikenali di {root}")
    return rek
