import json

import numpy as np
import pandas as pd
import pytest

from rppg.data import Rekaman
from rppg.evaluasi import (BELUM, ekstrak_rgb, evaluasi_rekaman, holm, hr_ref_jendela,
                           ke_markdown, kurangi_fps, ringkas, tabel1, tulis_keluaran, uji_wilcoxon)
from rppg.wajah import roi_dahi

CFG = {"roi": {"lebar": 0.40, "tinggi": 0.15, "geser_y": 0.05, "deteksi_tiap_n_frame": 10},
       "sinyal": {"jendela_detik": 10, "band_hz": [0.7, 4.0]},
       "evaluasi": {"kondisi_fps": [15], "metode": ["chrom", "pos", "hijau"], "batas_galat_bpm": 5,
                    "hr_ref_valid_bpm": [30, 240], "folder_hasil": "results/evaluasi"}}


class PencariTiruan:
    """Wajah selalu memenuhi seluruh frame."""

    def __init__(self, ada=True):
        self.ada, self.n = ada, 0

    def cari(self, frame):
        self.n += 1
        return (0, 0, frame.shape[1], frame.shape[0]) if self.ada else None


def video_sintetis(bpm=72, durasi=20, fps=30, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, durasi, 1 / fps)
    p = np.sin(2 * np.pi * bpm / 60 * t)
    for ti, pi in zip(t, p):
        warna = np.array([150 * (1 + 0.006 * pi), 110 * (1 + 0.012 * pi), 90 * (1 + 0.003 * pi)])
        f = np.clip(warna + rng.normal(0, 2, (64, 64, 3)), 0, 255).astype(np.uint8)
        yield f, float(ti)


def rekaman_sintetis(id_, bpm, **kw):
    return Rekaman(dataset="sintetis", id=id_, frames=lambda: video_sintetis(bpm, **kw),
                   referensi=lambda: (np.arange(0, 21, 1 / 60.0), np.full(21 * 60, float(bpm))))


def test_kurangi_fps_berbasis_timestamp():
    rng = np.random.default_rng(0)
    t = np.arange(0, 10, 1 / 30) + rng.uniform(-0.003, 0.003, 300)
    hasil = [ti for _, ti in kurangi_fps(((None, ti) for ti in t), 15)]
    assert abs(len(hasil) - 150) <= 2
    assert np.all(np.diff(hasil) > 1 / 30 + 0.005)  # tidak pernah dua frame berurutan asli


def test_kurangi_fps_lebih_tinggi_dari_asli_tidak_menambah():
    t = np.arange(0, 2, 1 / 15)
    assert len(list(kurangi_fps(((None, ti) for ti in t), 30))) == len(t)


def test_roi_dahi():
    x0, y0, x1, y1 = roi_dahi((100, 50, 200, 200), (480, 640), CFG["roi"])
    assert (x1 - x0, y1 - y0) == (80, 30)
    assert x0 == 160 and y0 == 60
    assert roi_dahi((630, 470, 200, 200), (480, 640), CFG["roi"]) is None  # di luar frame


def test_ekstrak_rgb_deteksi_tiap_n_dan_wajah_hilang():
    frames = [(np.full((20, 20, 3), 7, np.uint8), i / 30) for i in range(25)]
    pc = PencariTiruan()
    hasil = list(ekstrak_rgb(iter(frames), pc, CFG["roi"]))
    assert pc.n == 3 and np.allclose(hasil[0][1], 7)
    assert all(rgb is None for _, rgb in ekstrak_rgb(iter(frames), PencariTiruan(ada=False), CFG["roi"]))


def test_hr_ref_jendela():
    t = np.arange(0, 20, 0.5)
    hr = np.where(t < 10, 60.0, 90.0)
    hr[3] = 0.0  # tidak valid, diabaikan
    assert np.isclose(hr_ref_jendela(t, hr, 9.9, 10, (30, 240)), 60.0)
    assert np.isclose(hr_ref_jendela(t, hr, 19.5, 5, (30, 240)), 90.0)
    assert np.isnan(hr_ref_jendela(t, hr, 50.0, 10, (30, 240)))


def test_evaluasi_rekaman_menemukan_bpm():
    baris = evaluasi_rekaman(rekaman_sintetis("v1", 72), 15, ["chrom", "pos", "hijau"], PencariTiruan(), CFG)
    df = pd.DataFrame(baris)
    assert set(df.metode) == {"chrom", "pos", "hijau"}
    assert (df.kondisi_fps == 15).all() and (df.hr_ref == 72).all()
    assert df.groupby("metode").size().min() >= 5
    assert (df.galat.abs() <= 3).mean() > 0.9


def test_holm():
    p_adj = holm([0.01, 0.04, 0.03])
    assert np.allclose(p_adj, [0.03, 0.06, 0.06])
    assert np.isnan(holm([np.nan, 0.02])[0])


def df_buatan():
    rng = np.random.default_rng(3)
    baris = []
    for v in range(6):
        for m, sd in [("chrom", 1.0), ("pos", 1.5), ("hijau", 8.0)]:
            for k in range(10):
                baris.append(dict(dataset="ubfc", video=f"s{v}", kondisi_fps=15, metode=m, t=10.0 + k,
                                  bpm=72 + rng.normal(0, sd), confidence=0.5, hr_ref=72.0))
    df = pd.DataFrame(baris)
    df["galat"] = df.bpm - df.hr_ref
    return df


def test_ringkas_dan_wilcoxon():
    df = df_buatan()
    r = ringkas(df, 5)
    assert set(r.columns) >= {"dataset", "kondisi_fps", "metode", "n", "mae", "rmse",
                              "proporsi_dalam_batas", "bias", "loa_bawah", "loa_atas"}
    baris = r.set_index("metode")
    assert baris.loc["chrom", "n"] == 60 and baris.loc["chrom", "mae"] < baris.loc["hijau", "mae"]
    w = uji_wilcoxon(df)
    assert len(w) == 3 and (w.n_video == 6).all()
    assert (w.p_holm >= w.p).all()


def test_tulis_keluaran(tmp_path):
    df = df_buatan()
    folder = tulis_keluaran(df, tmp_path / "run", {"dataset": ["ubfc"]}, 5)
    for nama in ["estimasi.csv", "ringkasan.csv", "wilcoxon.csv", "meta.json",
                 "bland_altman_ubfc_15fps_chrom.png"]:
        assert (folder / nama).exists(), nama
    assert json.loads((folder / "meta.json").read_text(encoding="utf-8"))["dataset"] == ["ubfc"]
    assert len(pd.read_csv(folder / "estimasi.csv")) == len(df)


def test_tabel1_belum_diukur_dan_terisi():
    t = tabel1(None, ["ubfc", "pure"], ["chrom", "pos", "hijau"], 15, 5)
    assert t.shape == (6, 5) and (t.iloc[:, 2:] == BELUM).all().all()
    t = tabel1(ringkas(df_buatan(), 5), ["ubfc", "pure"], ["chrom", "pos", "hijau"], 15, 5)
    ubfc = t[t.Dataset == "ubfc"]
    assert (ubfc.iloc[:, 2:] != BELUM).all().all()
    assert (t[t.Dataset == "pure"].iloc[:, 2:] == BELUM).all().all()
    md = ke_markdown(t)
    assert md.splitlines()[0].startswith("| Dataset") and BELUM in md
