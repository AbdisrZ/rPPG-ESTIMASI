import csv
import json

import numpy as np
import pytest

from rppg.cek_kamera import jalankan_cek, ringkas, tulis_hasil
from rppg.sumber import SumberKamera

CFG = {"kamera": {"indeks": 0, "backend": "DSHOW", "lebar": 640, "tinggi": 480, "pemanasan_frame": 2}}


class CapTiruan:
    """Pengganti cv2.VideoCapture: memutar daftar frame, lalu gagal."""

    def __init__(self, frames, buka=True):
        self.frames = list(frames)
        self.buka = buka
        self.dilepas = False
        self.setelan = {}

    def isOpened(self):
        return self.buka

    def set(self, prop, val):
        self.setelan[prop] = val
        return True

    def get(self, prop):
        return self.setelan.get(prop, 0.0)

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        self.dilepas = True


class JamTiruan:
    """Jam deterministik: tiap panggilan maju sebesar langkah berikutnya."""

    def __init__(self, langkah):
        self.langkah = list(langkah)
        self.t = 0.0

    def __call__(self):
        if self.langkah:
            self.t += self.langkah.pop(0)
        return self.t


def frame(nilai):
    return np.full((4, 4, 3), nilai, dtype=np.uint8)


def test_sumber_buang_duplikat_dan_timestamp():
    # 2 frame pemanasan, lalu A, A(duplikat), B, C
    cap = CapTiruan([frame(9), frame(9), frame(1), frame(1), frame(2), frame(3)])
    jam = JamTiruan([0.1] * 20)
    s = SumberKamera(CFG, buka=lambda i, b: cap, jam=jam)
    hasil = list(s)
    assert [int(f[0, 0, 0]) for f, _ in hasil] == [1, 2, 3]
    ts = [t for _, t in hasil]
    assert ts == sorted(ts)
    assert s.n_dibaca == 4 and s.n_duplikat == 1
    s.tutup()
    assert cap.dilepas


def test_sumber_gagal_buka():
    with pytest.raises(RuntimeError):
        SumberKamera(CFG, buka=lambda i, b: CapTiruan([], buka=False))


def test_ringkas_angka_diketahui():
    # interval frame unik: 0.1, 0.1, 0.1, 0.2 -> median 0.1
    t = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    dup = [0, 1, 0, 0, 0, 0]
    r = ringkas(t, dup)
    assert r["n_dibaca"] == 6 and r["n_unik"] == 5 and r["n_duplikat"] == 1
    assert np.isclose(r["rasio_duplikat"], 1 / 6)
    assert np.isclose(r["fps_nyata"], 4 / 0.5)
    assert np.isclose(r["dt_rata_ms"], 125.0)
    assert np.isclose(r["dt_p95_ms"], np.percentile([100, 100, 100, 200], 95))
    # jitter = |dt - median(dt)| = 0, 0, 0, 100 ms
    assert np.isclose(r["jitter_rata_ms"], 25.0)
    assert np.isclose(r["jitter_p95_ms"], np.percentile([0, 0, 0, 100], 95))


def test_ringkas_terlalu_sedikit_frame():
    r = ringkas([0.0], [0])
    assert r["fps_nyata"] is None and r["jitter_p95_ms"] is None


def test_jalankan_dan_tulis(tmp_path):
    frames = [frame(0), frame(0)] + [frame(k % 2 * 50 + k) for k in range(100)]
    cap = CapTiruan(frames)
    s = SumberKamera(CFG, buka=lambda i, b: cap, jam=JamTiruan([1 / 15] * 500))
    log = jalankan_cek(s, detik=2.0)
    assert 0 < len(log) <= 100
    folder = tulis_hasil(log, tmp_path, meta={"backend": "DSHOW"})
    with open(folder / "frame.csv", encoding="utf-8") as f:
        baris = list(csv.DictReader(f))
    assert len(baris) == len(log)
    assert set(baris[0]) == {"indeks", "t", "dt_ms", "duplikat"}
    ring = json.loads((folder / "ringkasan.json").read_text(encoding="utf-8"))
    assert ring["meta"]["backend"] == "DSHOW"
    assert np.isclose(ring["statistik"]["fps_nyata"], 15.0)
