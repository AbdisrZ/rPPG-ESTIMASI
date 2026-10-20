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
    # ok | mengumpulkan | direset | timestamp_mundur | sampel_tidak_valid
    status: str = "mengumpulkan"
    durasi_s: float = 0.0  # rentang waktu sampel di buffer


class StreamingProcessor:
    """Kausal: hanya sampel dengan t lebih besar dari sampel terakhir yang diterima.

    Ketahanan (config ketahanan): sampel non-finite atau timestamp mundur/sama dibuang tanpa
    mengubah buffer; jeda antar sampel > jeda_maks_detik (mis. wajah hilang) mengosongkan buffer
    dan BPM, sehingga sinyal sebelum dan sesudah jeda tidak tersambung.
    """

    def __init__(self, cfg: dict | None = None, perbarui_tiap: float = 1.0, fs_grid: float = 30.0,
                 metode: str = "chrom"):
        if metode not in METODE:
            raise ValueError(f"metode tidak dikenal: {metode}")
        self.metode = metode
        self.cfg = cfg or muat()
        self.jendela = float(self.cfg["sinyal"]["jendela_detik"])
        self.band = tuple(self.cfg["sinyal"]["band_hz"])
        self.jeda_maks = float(self.cfg["ketahanan"]["jeda_maks_detik"])
        self.perbarui_tiap = perbarui_tiap
        self.fs_grid = fs_grid
        self.reset()

    def reset(self):
        self._buf: deque = deque()
        self._t_hitung = -np.inf
        self._bpm = None
        self._conf = None
        self._pulsa = None

    @property
    def pulsa(self):
        """Sinyal pulsa (grid seragam fs_grid) dari perhitungan BPM terakhir, atau None."""
        return self._pulsa

    @property
    def t_terakhir(self) -> float | None:
        return self._buf[-1][0] if self._buf else None

    def keadaan(self, t, status, diperbarui=False) -> Keadaan:
        """Keadaan buffer saat ini tanpa menambah sampel."""
        durasi = self._buf[-1][0] - self._buf[0][0] if self._buf else 0.0
        return Keadaan(t=float(t), n_sampel=len(self._buf), bpm=self._bpm, confidence=self._conf,
                       diperbarui=diperbarui, status=status, durasi_s=float(durasi))

    def push_rgb(self, rgb, t: float) -> Keadaan:
        t = float(t)
        x = np.asarray(rgb, dtype=float)
        if not (np.isfinite(t) and np.all(np.isfinite(x))):
            return self.keadaan(t, "sampel_tidak_valid")
        if self._buf and t <= self._buf[-1][0]:
            return self.keadaan(t, "timestamp_mundur")
        direset = bool(self._buf) and t - self._buf[-1][0] > self.jeda_maks
        if direset:
            self.reset()
        self._buf.append((t, x))
        while self._buf and t - self._buf[0][0] > self.jendela:
            self._buf.popleft()
        durasi = self._buf[-1][0] - self._buf[0][0]
        baru = durasi >= 0.8 * self.jendela and t - self._t_hitung >= self.perbarui_tiap
        if baru:
            self._t_hitung = t
            ts = np.array([b[0] for b in self._buf])
            xs = np.stack([b[1] for b in self._buf])
            _, grid_rgb = resampel_seragam(ts, xs, self.fs_grid)
            self._pulsa = METODE[self.metode](grid_rgb, self.fs_grid, self.band)
            self._bpm, self._conf = estimasi_bpm(self._pulsa, self.fs_grid, self.band)
        status = "direset" if direset else ("ok" if self._bpm is not None else "mengumpulkan")
        return self.keadaan(t, status, baru)
