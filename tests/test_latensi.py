import json

import numpy as np
import pandas as pd

from rppg.latensi import Pengukur, ringkas, tulis_hasil, ukur_pipeline

CFG = {"roi": {"lebar": 0.40, "tinggi": 0.15, "geser_y": 0.05, "deteksi_tiap_n_frame": 10},
       "sinyal": {"jendela_detik": 10, "band_hz": [0.7, 4.0]},
       "ketahanan": {"jeda_maks_detik": 1.0, "kamera_gagal_maks": 15}}


class JamManual:
    """Jam yang hanya maju bila diminta, supaya durasi tiap tahap diketahui pasti."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def maju(self, s):
        self.t += s


class PencariLambat:
    def __init__(self, jam, detik=0.020, ada=True):
        self.jam, self.detik, self.ada = jam, detik, ada

    def cari(self, frame):
        self.jam.maju(self.detik)
        return (0, 0, frame.shape[1], frame.shape[0]) if self.ada else None


def frames_sintetis(bpm=72, durasi=15, fps=15, seed=0):
    rng = np.random.default_rng(seed)
    for ti in np.arange(0, durasi, 1 / fps):
        p = np.sin(2 * np.pi * bpm / 60 * ti)
        warna = np.array([150 * (1 + 0.006 * p), 110 * (1 + 0.012 * p), 90 * (1 + 0.003 * p)])
        yield np.clip(warna + rng.normal(0, 2, (32, 32, 3)), 0, 255).astype(np.uint8), float(ti)


def test_pengukur_konteks_dan_ringkasan():
    jam = JamManual()
    p = Pengukur(jam)
    for d in [0.001, 0.002, 0.003, 0.010]:
        with p.ukur("x"):
            jam.maju(d)
    r = ringkas(p.sampel).set_index("tahap")
    assert r.loc["x", "n"] == 4
    assert np.isclose(r.loc["x", "median_ms"], 2.5)
    assert np.isclose(r.loc["x", "p95_ms"], np.percentile([1, 2, 3, 10], 95))


def test_ukur_pipeline_tahap_dan_bpm_pertama():
    jam = JamManual()
    p, info = ukur_pipeline(frames_sintetis(), PencariLambat(jam), CFG, "chrom", jam=jam,
                            ke_rgb=lambda f: f)
    r = ringkas(p.sampel).set_index("tahap")
    n = 15 * 15
    assert info["n_frame"] == n and info["n_frame_tanpa_roi"] == 0
    assert r.loc["deteksi_wajah", "n"] == int(np.ceil(n / 10))
    assert np.isclose(r.loc["deteksi_wajah", "median_ms"], 20.0)
    assert r.loc["roi_rgb", "n"] == n and r.loc["konversi_rgb", "n"] == n and r.loc["total_frame", "n"] == n
    assert r.loc["push_hitung", "n"] == info["n_estimasi"] > 0
    assert r.loc["push_hitung", "n"] + r.loc["push_tanpa_hitung", "n"] == n
    # BPM pertama butuh 0.8 x jendela (8 s) data pada timestamp sumber
    assert 8.0 <= info["waktu_sampai_bpm_pertama_s"] < 8.0 + 2 / 15
    assert info["bpm_pertama"] is not None


def test_ukur_pipeline_batas_detik_dan_tanpa_wajah():
    jam = JamManual()
    p, info = ukur_pipeline(frames_sintetis(), PencariLambat(jam, ada=False), CFG, "chrom",
                            detik=2.0, jam=jam)
    assert info["n_frame"] == 30 and info["n_frame_tanpa_roi"] == 30
    assert info["waktu_sampai_bpm_pertama_s"] is None and info["n_estimasi"] == 0
    assert "push_hitung" not in set(ringkas(p.sampel).tahap)


def test_tulis_hasil(tmp_path):
    jam = JamManual()
    p, info = ukur_pipeline(frames_sintetis(durasi=3), PencariLambat(jam), CFG, "chrom", jam=jam)
    folder = tulis_hasil(p, info, tmp_path / "run", {"sumber": "tiruan"})
    s = pd.read_csv(folder / "sampel.csv")
    assert list(s.columns) == ["indeks_frame", "tahap", "ms"] and len(s) == len(p.sampel)
    r = pd.read_csv(folder / "ringkasan.csv")
    assert {"tahap", "n", "median_ms", "p95_ms"} <= set(r.columns)
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["sumber"] == "tiruan" and meta["info"]["n_frame"] == 45
