"""WP4: wajah hilang, kamera gagal di tengah sesi, frame kosong, timestamp mundur, status antarmuka."""
import numpy as np
import pytest

from rppg.aliran import PESAN, Aliran
from rppg.pipeline import StreamingProcessor
from rppg.sumber import KameraGagal, SumberKamera
from rppg.wajah import PengekstrakRGB

CFG = {"roi": {"lebar": 0.40, "tinggi": 0.15, "geser_y": 0.05, "deteksi_tiap_n_frame": 10},
       "sinyal": {"jendela_detik": 10, "band_hz": [0.7, 4.0]},
       "kamera": {"indeks": 0, "backend": "DSHOW", "lebar": 640, "tinggi": 480, "pemanasan_frame": 0},
       "ketahanan": {"jeda_maks_detik": 1.0, "kamera_gagal_maks": 3}}


def rgb_pulsa(t, bpm=72):
    p = np.sin(2 * np.pi * bpm / 60 * t)
    return np.array([0.60 + 0.004 * p, 0.45 + 0.008 * p, 0.40 + 0.002 * p])


def isi(proc, t_awal, t_akhir, fps=15):
    k = None
    for t in np.arange(t_awal, t_akhir, 1 / fps):
        k = proc.push_rgb(rgb_pulsa(t), t)
    return k


# --- wajah hilang: jeda antar sampel > batas -> buffer dikosongkan
def test_jeda_panjang_mengosongkan_buffer():
    p = StreamingProcessor(CFG)
    k = isi(p, 0, 12)
    assert k.bpm is not None
    k = p.push_rgb(rgb_pulsa(14.0), 14.0)  # wajah hilang 2 s
    assert k.status == "direset" and k.n_sampel == 1 and k.bpm is None and k.confidence is None
    k = isi(p, 14.0 + 1 / 15, 20)
    assert k.bpm is None  # belum 0.8 x jendela sejak reset
    assert isi(p, 20, 23).bpm is not None


def test_jeda_pendek_tidak_mereset():
    p = StreamingProcessor(CFG)
    isi(p, 0, 12)
    p.push_rgb(rgb_pulsa(12.0), 12.0)
    k = p.push_rgb(rgb_pulsa(12.5), 12.5)
    assert k.status == "ok" and k.bpm is not None and k.durasi_s > 9.0


# --- timestamp mundur atau sama: sampel dibuang, buffer tetap
@pytest.mark.parametrize("t_baru", [4.0, 5.0])
def test_timestamp_mundur_dibuang(t_baru):
    p = StreamingProcessor(CFG)
    isi(p, 0, 5.01)
    n = p.push_rgb(rgb_pulsa(5.5), 5.5).n_sampel
    k = p.push_rgb(rgb_pulsa(t_baru), t_baru)
    assert k.status == "timestamp_mundur" and k.n_sampel == n
    assert p.push_rgb(rgb_pulsa(5.6), 5.6).n_sampel == n + 1


@pytest.mark.parametrize("buruk", [[np.nan, 0.4, 0.4], [0.5, np.inf, 0.4]])
def test_sampel_tidak_valid_dibuang(buruk):
    p = StreamingProcessor(CFG)
    n = isi(p, 0, 3).n_sampel
    k = p.push_rgb(buruk, 3.1)
    assert k.status == "sampel_tidak_valid" and k.n_sampel == n


# --- frame kosong
class PencariPenuh:
    def cari(self, frame):
        return (0, 0, frame.shape[1], frame.shape[0])


@pytest.mark.parametrize("frame", [None, np.zeros((0, 0, 3), np.uint8)])
def test_frame_kosong_tidak_crash(frame):
    ex = PengekstrakRGB(PencariPenuh(), CFG["roi"])
    assert ex.proses(frame) is None
    assert ex.proses(np.full((20, 20, 3), 9, np.uint8)) is not None


# --- kamera gagal di tengah sesi
class CapGagal:
    """read() mengikuti pola: True = frame baru, False = gagal."""

    def __init__(self, pola):
        self.pola, self.i = list(pola), 0

    def isOpened(self):
        return True

    def set(self, *a):
        return True

    def get(self, *a):
        return 0.0

    def read(self):
        if not self.pola:
            return False, None
        ok = self.pola.pop(0)
        self.i += 1
        return (True, np.full((4, 4, 3), self.i % 256, np.uint8)) if ok else (False, None)

    def release(self):
        pass


def test_kamera_gagal_sesaat_dilanjutkan():
    cap = CapGagal([True, True, False, False, True, True])
    s = SumberKamera(CFG, buka=lambda i, b: cap, jam=iter(np.arange(100.0)).__next__)
    frames = []
    with pytest.raises(KameraGagal):  # setelah pola habis kamera terus gagal
        for f, t in s:
            frames.append(t)
    assert len(frames) == 4


def test_kamera_gagal_terus_menerus_melempar():
    cap = CapGagal([True] + [False] * 10)
    s = SumberKamera(CFG, buka=lambda i, b: cap, jam=iter(np.arange(100.0)).__next__)
    it = iter(s)
    next(it)
    with pytest.raises(KameraGagal, match="3"):
        next(it)


# --- status dan pesan untuk antarmuka
class PencariSaklar:
    def __init__(self):
        self.ada = True

    def cari(self, frame):
        return (0, 0, frame.shape[1], frame.shape[0]) if self.ada else None


def frame_pulsa(t):
    return np.clip(rgb_pulsa(t) * 255 + np.zeros((24, 24, 3)), 0, 255).astype(np.uint8)


def test_aliran_status_berurutan():
    pc = PencariSaklar()
    cfg = dict(CFG, roi=dict(CFG["roi"], deteksi_tiap_n_frame=1))
    a = Aliran(cfg, pc, "chrom")
    h = a.proses(frame_pulsa(0.0), 0.0)
    assert h.status == "mengumpulkan" and h.ada_wajah and "/10 s" in h.pesan
    assert a.proses(None, 0.05).status == "frame_kosong"
    for t in np.arange(1 / 15, 12, 1 / 15):
        h = a.proses(frame_pulsa(t), t)
    assert h.status == "ok" and h.bpm is not None
    pc.ada = False
    h = a.proses(frame_pulsa(12.1), 12.1)
    assert h.status == "wajah_hilang" and not h.ada_wajah
    pc.ada = True
    h = a.proses(frame_pulsa(14.0), 14.0)
    assert h.status == "direset" and h.bpm is None
    assert a.proses(frame_pulsa(13.0), 13.0).status == "timestamp_mundur"
    assert set(PESAN) >= {"ok", "mengumpulkan", "direset", "timestamp_mundur", "sampel_tidak_valid",
                          "wajah_hilang", "frame_kosong"}
    assert all(isinstance(v, str) and v for v in PESAN.values())
