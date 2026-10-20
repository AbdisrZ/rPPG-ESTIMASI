"""Logika aplikasi tanpa Qt: sesi pengukuran, perekaman CSV, tempo video, sumber, argumen, teks tampilan.

app.py hanya menampilkan; semua yang bisa dites ada di sini.
CSV sesi (results/sesi/<waktu>/sesi.csv), satu baris per frame yang diproses:
- timestamp : timestamp sumber (s); kamera: time.perf_counter, video: indeks / fps kontainer.
- bpm, confidence : estimasi terakhir yang berlaku pada frame itu (kosong bila belum/tidak ada).
- metode    : chrom | pos | hijau.
- ms_proses : waktu pemrosesan frame (BGR->RGB bila ada, ROI, push_rgb) dengan time.perf_counter, ms.
- status    : kode status Aliran (lihat aliran.PESAN).
"""
from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .aliran import Aliran, Hasil
from .jejak import meta_dasar
from .sinyal import METODE

TEKS_BUKAN_MEDIS = "Bukan alat medis. Angka hanya untuk penelitian, jangan dipakai untuk diagnosis."
KOLOM_SESI = ["timestamp", "bpm", "confidence", "metode", "ms_proses", "status"]


@dataclass
class Tampilan:
    hasil: Hasil
    metode: str
    ms_proses: float
    ms_median: float
    ms_p95: float
    n_ms: int  # banyak frame dalam median/p95
    pulsa: np.ndarray | None
    fs_pulsa: float
    roi: tuple | None
    frame: object = None  # frame RGB yang diproses, untuk pratinjau


class PerekamSesi:
    """Menulis sesi.csv (lihat docstring modul) dan meta.json ke <folder_induk>/<nama>/."""

    def __init__(self, folder_induk, meta: dict, nama: str | None = None):
        nama = nama or datetime.now().strftime("%Y%m%d-%H%M%S")
        folder, i = Path(folder_induk) / nama, 2
        while folder.exists():
            folder, i = Path(folder_induk) / f"{nama}-{i}", i + 1
        folder.mkdir(parents=True)
        self.folder = folder
        self.meta = meta | {"kolom": KOLOM_SESI, "definisi": __doc__.strip(), "n_baris": None}
        self._tulis_meta()
        self._f = open(folder / "sesi.csv", "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow(KOLOM_SESI)
        self.n_baris = 0

    def _tulis_meta(self):
        (self.folder / "meta.json").write_text(json.dumps(self.meta, indent=2, ensure_ascii=False, default=str),
                                               encoding="utf-8")

    def tulis(self, h: Hasil, metode: str, ms: float):
        kosong = lambda v, fmt: "" if v is None else format(v, fmt)
        self._w.writerow([repr(float(h.t)), kosong(h.bpm, ".2f"), kosong(h.confidence, ".4f"), metode,
                          f"{ms:.3f}", h.status])
        self._f.flush()
        self.n_baris += 1

    def tutup(self):
        if self._f.closed:
            return
        self._f.close()
        self.meta["n_baris"] = self.n_baris
        self._tulis_meta()


class Sesi:
    """Aliran + pengukuran ms proses + ganti metode + perekaman.

    `langkah` dipanggil dari utas pemroses; `minta_metode`, `mulai_rekam`, `berhenti_rekam`
    aman dipanggil dari utas antarmuka.
    """

    def __init__(self, cfg: dict, pencari, metode: str, jam=time.perf_counter):
        self.cfg, self.pencari, self.jam = cfg, pencari, jam
        self.ms: deque = deque(maxlen=int(cfg["aplikasi"]["latensi_jendela_frame"]))
        self._kunci = threading.Lock()
        self._perekam: PerekamSesi | None = None
        self._diminta = None
        self._pasang(metode)

    def _pasang(self, metode: str):
        self.metode = metode
        self.aliran = Aliran(self.cfg, self.pencari, metode)

    def minta_metode(self, metode: str):
        """Metode diganti pada langkah berikutnya; buffer dan BPM diulang dari awal."""
        if metode not in METODE:
            raise ValueError(f"metode tidak dikenal: {metode}")
        self._diminta = metode

    @property
    def merekam(self) -> bool:
        return self._perekam is not None

    def mulai_rekam(self, folder_induk, meta: dict) -> Path:
        with self._kunci:
            if self._perekam is None:
                self._perekam = PerekamSesi(folder_induk, meta_dasar(self.cfg) | meta)
            return self._perekam.folder

    def berhenti_rekam(self) -> Path | None:
        with self._kunci:
            p, self._perekam = self._perekam, None
        if p is None:
            return None
        p.tutup()
        return p.folder

    def langkah(self, frame, t: float, ke_rgb=None) -> Tampilan:
        diminta, self._diminta = self._diminta, None
        if diminta is not None and diminta != self.metode:
            self._pasang(diminta)
        a = self.jam()
        if ke_rgb is not None and frame is not None and frame.size:
            frame = ke_rgb(frame)
        h = self.aliran.proses(frame, t)
        ms = (self.jam() - a) * 1000.0
        self.ms.append(ms)
        with self._kunci:
            if self._perekam is not None:
                self._perekam.tulis(h, self.metode, ms)
        arr = np.fromiter(self.ms, float)
        proc = self.aliran.proc
        return Tampilan(h, self.metode, ms, float(np.median(arr)), float(np.percentile(arr, 95)),
                        len(arr), proc.pulsa, proc.fs_grid, self.aliran.ex.roi_terakhir, frame)


def atur_tempo(frames, jam=time.perf_counter, tidur=time.sleep):
    """Putar (frame, t) dari berkas pada kecepatan nyata: tunggu sampai jam dinding menyusul t.

    Bila pemrosesan lebih lambat dari video, frame tidak dibuang; video berjalan lebih lambat.
    """
    awal = None
    for f, t in frames:
        if awal is None:
            awal = (jam(), t)
        tunggu = (t - awal[1]) - (jam() - awal[0])
        if tunggu > 0:
            tidur(tunggu)
        yield f, t


@dataclass
class Sumber:
    frames: object          # iterator (frame, t)
    ke_rgb: object          # fungsi frame -> RGB, atau None bila sumber sudah RGB
    tutup: object
    info: dict


def buka_sumber(cfg: dict, video: str | None, fps: float) -> Sumber:
    """Kamera (SumberKamera, BGR) atau berkas video (RGB, diputar real-time); kurangi_fps bila fps > 0."""
    from .evaluasi import kurangi_fps
    from .sumber import SumberKamera, baca_video

    def batasi(fr):
        return kurangi_fps(fr, fps) if fps > 0 else fr

    if video:
        return Sumber(batasi(atur_tempo(baca_video(video))), None, lambda: None,
                      {"jenis": "video", "berkas": str(video), "fps_target": fps or None})
    import cv2

    kamera = SumberKamera(cfg)
    info = {"jenis": "kamera", "driver": kamera.info(), "fps_target": fps or None}
    return Sumber(batasi(iter(kamera)), lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2RGB), kamera.tutup, info)


def buat_parser(cfg: dict) -> argparse.ArgumentParser:
    ap_cfg = cfg["aplikasi"]
    ap = argparse.ArgumentParser(prog="python -m rppg.app", description="Estimasi denyut jantung rPPG. "
                                 + TEKS_BUKAN_MEDIS)
    ap.add_argument("--video", default=None, help="berkas video; tanpa ini memakai kamera dari config")
    ap.add_argument("--metode", default=ap_cfg["metode"], choices=list(METODE))
    ap.add_argument("--fps", type=float, default=float(ap_cfg["fps"]),
                    help="kurangi_fps pada sumber; 0 = mati")
    ap.add_argument("--rekam", action="store_true", help=f"langsung rekam sesi ke {ap_cfg['folder_sesi']}")
    return ap


def teks_tampilan(tp: Tampilan) -> dict:
    h = tp.hasil
    return {"bpm": "--" if h.bpm is None else f"{h.bpm:.0f}",
            "confidence": "--" if h.confidence is None else f"{h.confidence:.2f}",
            "status": h.pesan,
            "latensi": f"Proses per frame: {tp.ms_proses:.1f} ms (median {tp.ms_median:.1f} ms, "
                       f"p95 {tp.ms_p95:.1f} ms, {tp.n_ms} frame terakhir)"}
