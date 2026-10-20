"""Pengukuran latensi per tahap dengan time.perf_counter.

Tahap per frame (ms):
- konversi_rgb      : BGR -> RGB (hanya bila sumber memberi BGR, mis. kamera).
- deteksi_wajah     : pencari.cari(), hanya pada frame deteksi (tiap roi.deteksi_tiap_n_frame).
- roi_rgb           : ROI dahi + rata-rata RGB.
- push_hitung       : push_rgb yang menghitung ulang BPM (Keadaan.diperbarui).
- push_tanpa_hitung : push_rgb yang hanya menambah buffer.
- total_frame       : seluruh pemrosesan satu frame setelah frame diterima (tanpa menunggu kamera).
Waktu sampai BPM pertama: dari timestamp sumber (frame pertama -> frame yang pertama memberi BPM)
dan dari jam dinding sejak loop dimulai.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from .pipeline import StreamingProcessor
from .wajah import PengekstrakRGB

TAHAP = ["konversi_rgb", "deteksi_wajah", "roi_rgb", "push_hitung", "push_tanpa_hitung", "total_frame"]


class Pengukur:
    def __init__(self, jam=time.perf_counter):
        self.jam = jam
        self.frame = -1
        self.sampel: list[tuple[int, str, float]] = []

    def catat(self, tahap: str, ms: float):
        self.sampel.append((self.frame, tahap, float(ms)))

    @contextmanager
    def ukur(self, tahap: str):
        t0 = self.jam()
        try:
            yield
        finally:
            self.catat(tahap, (self.jam() - t0) * 1000.0)


def ringkas(sampel) -> pd.DataFrame:
    df = pd.DataFrame(sampel, columns=["indeks_frame", "tahap", "ms"])
    baris = []
    for tahap in [t for t in TAHAP if t in set(df.tahap)] + sorted(set(df.tahap) - set(TAHAP)):
        ms = df.ms[df.tahap == tahap].to_numpy()
        baris.append(dict(tahap=tahap, n=len(ms), median_ms=float(np.median(ms)),
                          p95_ms=float(np.percentile(ms, 95)), rata_ms=float(ms.mean()), maks_ms=float(ms.max())))
    return pd.DataFrame(baris, columns=["tahap", "n", "median_ms", "p95_ms", "rata_ms", "maks_ms"])


def ukur_pipeline(frames, pencari, cfg: dict, metode: str, detik: float | None = None,
                  jam=time.perf_counter, ke_rgb=None):
    """Jalankan jalur live (ROI -> push_rgb) pada `frames` (iterator (frame, t)) sambil mengukur.

    `detik` membatasi durasi menurut timestamp sumber. Kembalikan (Pengukur, info).
    """
    p = Pengukur(jam)
    ex = PengekstrakRGB(pencari, cfg["roi"], p)
    proc = StreamingProcessor(cfg, metode=metode)
    info = dict(metode=metode, n_frame=0, n_frame_tanpa_roi=0, n_estimasi=0, bpm_pertama=None,
                waktu_sampai_bpm_pertama_s=None, waktu_dinding_sampai_bpm_pertama_s=None)
    dinding0, t0 = jam(), None
    for i, (f, t) in enumerate(frames):
        if t0 is None:
            t0 = t
        if detik is not None and t - t0 >= detik:
            break
        p.frame = i
        info["n_frame"] += 1
        with p.ukur("total_frame"):
            if ke_rgb is not None:
                with p.ukur("konversi_rgb"):
                    f = ke_rgb(f)
            rgb = ex.proses(f)
            if rgb is None:
                info["n_frame_tanpa_roi"] += 1
                continue
            a = jam()
            k = proc.push_rgb(rgb, t)
            p.catat("push_hitung" if k.diperbarui else "push_tanpa_hitung", (jam() - a) * 1000.0)
        if k.diperbarui:
            info["n_estimasi"] += 1
        if k.bpm is not None and info["bpm_pertama"] is None:
            info.update(bpm_pertama=k.bpm, waktu_sampai_bpm_pertama_s=float(t - t0),
                        waktu_dinding_sampai_bpm_pertama_s=float(jam() - dinding0))
    return p, info


def tulis_hasil(pengukur: Pengukur, info: dict, folder, meta: dict) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pengukur.sampel, columns=["indeks_frame", "tahap", "ms"]).to_csv(folder / "sampel.csv", index=False)
    ringkas(pengukur.sampel).to_csv(folder / "ringkasan.csv", index=False)
    isi = meta | {"definisi": __doc__.strip(), "info": info}
    (folder / "meta.json").write_text(json.dumps(isi, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return folder
