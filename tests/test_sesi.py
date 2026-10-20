"""WP5: logika aplikasi tanpa Qt (sesi, perekaman CSV, tempo video, argumen, teks tampilan)."""
import csv
import json

import numpy as np
import pytest

from rppg.aliran import PESAN
from rppg.pipeline import StreamingProcessor
from rppg.sesi import (KOLOM_SESI, TEKS_BUKAN_MEDIS, PerekamSesi, Sesi, atur_tempo, buat_parser,
                       teks_tampilan)

CFG = {"roi": {"lebar": 0.40, "tinggi": 0.15, "geser_y": 0.05, "deteksi_tiap_n_frame": 1},
       "sinyal": {"jendela_detik": 10, "band_hz": [0.7, 4.0]},
       "kamera": {"indeks": 0, "backend": "DSHOW", "lebar": 640, "tinggi": 480, "pemanasan_frame": 0},
       "ketahanan": {"jeda_maks_detik": 1.0, "kamera_gagal_maks": 3},
       "aplikasi": {"fps": 15, "metode": "chrom", "latensi_jendela_frame": 4, "folder_sesi": "results/sesi"}}


class PencariPenuh:
    def cari(self, frame):
        return (0, 0, frame.shape[1], frame.shape[0])


def frame_pulsa(t, bpm=72):
    p = np.sin(2 * np.pi * bpm / 60 * t)
    rgb = np.array([0.60 + 0.004 * p, 0.45 + 0.008 * p, 0.40 + 0.002 * p])
    return np.clip(rgb * 255 + np.zeros((24, 24, 3)), 0, 255).astype(np.uint8)


class JamTiruan:
    """Tiap panggilan maju `langkah` detik."""

    def __init__(self, langkah=0.002):
        self.t, self.langkah = 0.0, langkah

    def __call__(self):
        self.t += self.langkah
        return self.t


def jalankan(sesi, t_awal, t_akhir, fps=15):
    tp = None
    for t in np.arange(t_awal, t_akhir, 1 / fps):
        tp = sesi.langkah(frame_pulsa(t), t)
    return tp


def test_pipeline_menyimpan_pulsa_dan_reset():
    p = StreamingProcessor(CFG)
    assert p.pulsa is None
    for t in np.arange(0, 12, 1 / 15):
        p.push_rgb(frame_pulsa(t)[0, 0] / 255, t)
    assert p.pulsa is not None and p.pulsa.ndim == 1 and len(p.pulsa) > 200
    p.reset()
    assert p.pulsa is None


def test_sesi_langkah_ms_proses_dan_pulsa():
    s = Sesi(CFG, PencariPenuh(), "chrom", jam=JamTiruan(0.002))
    tp = s.langkah(frame_pulsa(0.0), 0.0)
    assert tp.ms_proses == pytest.approx(2.0)
    assert tp.hasil.bpm is None and tp.pulsa is None and tp.metode == "chrom"
    assert tp.roi is not None
    tp = jalankan(s, 1 / 15, 12)
    assert tp.hasil.status == "ok" and 65 < tp.hasil.bpm < 79
    assert tp.pulsa is not None and tp.fs_pulsa > 0
    assert tp.ms_median == pytest.approx(2.0) and tp.ms_p95 == pytest.approx(2.0)


def test_sesi_ke_rgb_ikut_diukur():
    s = Sesi(CFG, PencariPenuh(), "chrom", jam=JamTiruan(0.001))
    dipanggil = []
    tp = s.langkah(frame_pulsa(0.0), 0.0, ke_rgb=lambda f: dipanggil.append(1) or f[..., ::-1])
    assert dipanggil and tp.ms_proses == pytest.approx(1.0)
    assert np.array_equal(tp.frame, frame_pulsa(0.0)[..., ::-1])  # pratinjau memakai frame RGB
    assert s.langkah(None, 0.1, ke_rgb=lambda f: 1 / 0).hasil.status == "frame_kosong"


def test_jendela_latensi_dari_config():
    s = Sesi(CFG, PencariPenuh(), "chrom", jam=JamTiruan(0.001))
    jalankan(s, 0, 1)
    assert len(s.ms) == CFG["aplikasi"]["latensi_jendela_frame"]


def test_ganti_metode_mengulang_pengukuran():
    s = Sesi(CFG, PencariPenuh(), "chrom")
    assert jalankan(s, 0, 12).hasil.bpm is not None
    s.minta_metode("pos")
    tp = s.langkah(frame_pulsa(12.0), 12.0)
    assert tp.metode == "pos" and tp.hasil.bpm is None and tp.pulsa is None
    with pytest.raises(ValueError):
        s.minta_metode("tidak_ada")


def test_perekam_menulis_csv(tmp_path):
    s = Sesi(CFG, PencariPenuh(), "chrom", jam=JamTiruan(0.002))
    folder = s.mulai_rekam(tmp_path, meta={"sumber": {"jenis": "tiruan"}})
    jalankan(s, 0, 12)
    s.minta_metode("hijau")
    s.langkah(frame_pulsa(12.0), 12.0)
    s.langkah(None, 12.05)
    s.berhenti_rekam()
    s.langkah(frame_pulsa(12.1), 12.1)  # setelah berhenti: tidak ditulis
    with open(folder / "sesi.csv", newline="", encoding="utf-8") as f:
        baris = list(csv.DictReader(f))
    assert list(baris[0]) == KOLOM_SESI
    assert len(baris) == len(np.arange(0, 12, 1 / 15)) + 2
    assert baris[0]["bpm"] == "" and baris[0]["confidence"] == ""
    assert float(baris[-3]["bpm"]) > 0 and 0 <= float(baris[-3]["confidence"]) <= 1
    assert baris[-2]["metode"] == "hijau" and baris[-1]["status"] == "frame_kosong"
    assert float(baris[1]["timestamp"]) == pytest.approx(1 / 15)
    assert float(baris[0]["ms_proses"]) == pytest.approx(2.0)
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["sumber"]["jenis"] == "tiruan" and "kolom" in meta and meta["n_baris"] == len(baris)


def test_perekam_folder_unik(tmp_path):
    a = PerekamSesi(tmp_path, meta={}, nama="x")
    b = PerekamSesi(tmp_path, meta={}, nama="x")
    assert a.folder != b.folder
    a.tutup()
    b.tutup()
    a.tutup()  # tutup dua kali aman


def test_atur_tempo_menunggu_sesuai_timestamp():
    jam = JamTiruan(0.0)
    tidur = []

    def tidur_tiruan(d):
        tidur.append(d)
        jam.t += d

    frames = [(None, 10.0), (None, 10.5), (None, 11.0)]
    keluar = list(atur_tempo(iter(frames), jam=jam, tidur=tidur_tiruan))
    assert [t for _, t in keluar] == [10.0, 10.5, 11.0]
    assert tidur == pytest.approx([0.5, 0.5])


def test_atur_tempo_tidak_menunggu_bila_terlambat():
    jam = JamTiruan(1.0)  # pemrosesan lebih lambat dari video
    tidur = []
    list(atur_tempo(iter([(None, 0.0), (None, 0.1), (None, 0.2)]), jam=jam, tidur=tidur.append))
    assert tidur == []


def test_parser_bawaan_dari_config():
    a = buat_parser(CFG).parse_args([])
    assert a.video is None and a.metode == "chrom" and a.fps == 15 and not a.rekam
    a = buat_parser(CFG).parse_args(["--video", "x.mp4", "--metode", "pos", "--fps", "0", "--rekam"])
    assert a.video == "x.mp4" and a.metode == "pos" and a.fps == 0 and a.rekam


def test_teks_tampilan():
    s = Sesi(CFG, PencariPenuh(), "chrom", jam=JamTiruan(0.002))
    teks = teks_tampilan(s.langkah(frame_pulsa(0.0), 0.0))
    assert teks["bpm"] == "--" and teks["confidence"] == "--" and "Mengumpulkan" in teks["status"]
    assert "ms" in teks["latensi"]
    teks = teks_tampilan(jalankan(s, 1 / 15, 12))
    assert teks["bpm"].isdigit() and teks["confidence"].startswith("0.")


def test_teks_bukan_alat_medis_dan_pesan_akhir():
    assert "bukan alat medis" in TEKS_BUKAN_MEDIS.lower()
    assert PESAN["video_selesai"] and PESAN["kamera_gagal"]
