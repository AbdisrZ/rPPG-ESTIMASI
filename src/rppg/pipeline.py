"""Inti streaming: push_rgb(rgb, t) -> Keadaan. Dipakai bersama oleh demo & evaluasi offline."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import muat
from .sinyal import METODE, estimasi_bpm, resampel_seragam


@dataclass
class Keadaan:
    t: float
    n_sampel: int
    bpm: float | None = None
    confidence: float | None = None
    diperbarui: bool = False  # True bila bpm dihitung ulang pada push ini


class StreamingProcessor:
    def __init__(self, cfg: dict | None = None, perbarui_tiap: float = 1.0, fs_grid: float = 30.0,
                 metode: str = "chrom"):
        if metode not in METODE:
            raise ValueError(f"metode tidak dikenal: {metode}")
        self.metode = metode
        self.cfg = cfg or muat()
        self.jendela = float(self.cfg["sinyal"]["jendela_detik"])
        self.band = tuple(self.cfg["sinyal"]["band_hz"])
        self.perbarui_tiap = perbarui_tiap
        self.fs_grid = fs_grid
        self._buf: deque = deque()
        self._t_hitung = -np.inf
        self._bpm = None
        self._conf = None

    def push_rgb(self, rgb, t: float) -> Keadaan:
        self._buf.append((float(t), np.asarray(rgb, dtype=float)))
        while self._buf and t - self._buf[0][0] > self.jendela:
            self._buf.popleft()
        durasi = self._buf[-1][0] - self._buf[0][0]
        baru = durasi >= 0.8 * self.jendela and t - self._t_hitung >= self.perbarui_tiap
        if baru:
            self._t_hitung = t
            ts = np.array([b[0] for b in self._buf])
            xs = np.stack([b[1] for b in self._buf])
            _, grid_rgb = resampel_seragam(ts, xs, self.fs_grid)
            self._bpm, self._conf = estimasi_bpm(METODE[self.metode](grid_rgb, self.fs_grid, self.band),
                                                 self.fs_grid, self.band)
        return Keadaan(t=float(t), n_sampel=len(self._buf), bpm=self._bpm, confidence=self._conf,
                       diperbarui=baru)
