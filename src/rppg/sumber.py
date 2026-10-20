"""Sumber frame bertimestamp. SumberKamera: backend dari config, frame duplikat dibuang."""
from __future__ import annotations

import time

import numpy as np

from .config import muat


def _buka_opencv(indeks: int, backend: str):
    import cv2  # impor malas: inti dan tes tidak wajib bergantung pada cv2

    return cv2.VideoCapture(indeks, getattr(cv2, f"CAP_{backend}"))


class KameraGagal(RuntimeError):
    pass


def baca_video(path):
    """Berkas video -> iterator (frame RGB, t = indeks / fps kontainer). Galat bila tidak bisa dibuka."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0
    if not fps or fps <= 0:
        cap.release()
        raise RuntimeError(f"video tidak bisa dibuka atau fps tidak terbaca: {path}")
    i = 0
    try:
        while True:
            ok, f = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(f, cv2.COLOR_BGR2RGB), i / fps
            i += 1
    finally:
        cap.release()


class SumberKamera:
    """Membaca webcam; tiap frame diberi timestamp time.perf_counter saat read() kembali.

    Frame yang identik piksel demi piksel dengan frame sebelumnya dianggap duplikat
    (driver mengulang frame lama) dan tidak diteruskan ke pemakai.
    `buka` dan `jam` bisa diganti untuk tes dengan kamera tiruan.
    """

    def __init__(self, cfg: dict | None = None, buka=_buka_opencv, jam=time.perf_counter):
        cfg = cfg or muat()
        k = cfg["kamera"]
        self.gagal_maks = int(cfg["ketahanan"]["kamera_gagal_maks"])
        self.backend = k["backend"]
        self.jam = jam
        self.cap = buka(int(k["indeks"]), self.backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"kamera {k['indeks']} gagal dibuka dengan backend {self.backend}")
        self.cap.set(3, k["lebar"])   # CAP_PROP_FRAME_WIDTH
        self.cap.set(4, k["tinggi"])  # CAP_PROP_FRAME_HEIGHT
        for _ in range(int(k["pemanasan_frame"])):
            self.cap.read()
        self._sebelum = None
        self.n_dibaca = 0
        self.n_duplikat = 0

    def info(self) -> dict:
        """Nilai yang dilaporkan driver (bukan hasil ukur)."""
        return {"backend": self.backend,
                "lebar": int(self.cap.get(3)), "tinggi": int(self.cap.get(4)),
                "fps_driver": float(self.cap.get(5))}  # CAP_PROP_FPS

    def baca_mentah(self):
        """Satu read(): (frame, t, duplikat) atau None bila kamera gagal."""
        ok, f = self.cap.read()
        t = self.jam()
        if not ok or f is None:
            return None
        dup = self._sebelum is not None and np.array_equal(f, self._sebelum)
        self._sebelum = f
        self.n_dibaca += 1
        self.n_duplikat += int(dup)
        return f, t, dup

    def __iter__(self):
        """Frame unik (frame, t). read() gagal sesaat dilewati; gagal berturut-turut
        >= ketahanan.kamera_gagal_maks melempar KameraGagal."""
        gagal = 0
        while True:
            r = self.baca_mentah()
            if r is None:
                gagal += 1
                if gagal >= self.gagal_maks:
                    raise KameraGagal(f"kamera gagal membaca {gagal} kali berturut-turut")
                continue
            gagal = 0
            if not r[2]:
                yield r[0], r[1]

    def tutup(self):
        self.cap.release()
