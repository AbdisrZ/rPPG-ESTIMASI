"""Latensi per tahap: python scripts/latensi.py [--video berkas] [--detik N] [--metode chrom]

Tanpa --video memakai SumberKamera (config kamera). Hasil: results/latensi/<waktu>/
{sampel.csv, ringkasan.csv, meta.json}. Hanya waktu yang disimpan, bukan frame.
"""
import argparse
import sys
from datetime import datetime

from rppg.config import RAIZ, muat
from rppg.jejak import meta_dasar
from rppg.latensi import ringkas, tulis_hasil, ukur_pipeline
from rppg.sumber import SumberKamera, baca_video
from rppg.wajah import PencariMediaPipe

cfg = muat()
lt = cfg["latensi"]
ap = argparse.ArgumentParser()
ap.add_argument("--video", default=None)
ap.add_argument("--detik", type=float, default=float(lt["detik_bawaan"]))
ap.add_argument("--metode", default=lt["metode"], choices=["chrom", "pos", "hijau"])
a = ap.parse_args()

pencari = PencariMediaPipe(cfg)
mulai = datetime.now()
if a.video:
    sumber_info = {"jenis": "video", "berkas": a.video}
    try:
        p, info = ukur_pipeline(baca_video(a.video), pencari, cfg, a.metode, detik=a.detik)
    except RuntimeError as e:
        sys.exit(f"Gagal: {e}")
else:
    import cv2

    try:
        kamera = SumberKamera(cfg)
    except RuntimeError as e:
        sys.exit(f"Gagal: {e}. Pengukuran latensi belum dijalankan.")
    try:
        p, info = ukur_pipeline(iter(kamera), pencari, cfg, a.metode, detik=a.detik,
                                ke_rgb=lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        sumber_info = {"jenis": "kamera", "driver": kamera.info(),
                       "n_dibaca": kamera.n_dibaca, "n_duplikat": kamera.n_duplikat}
    finally:
        kamera.tutup()

if info["n_frame"] == 0:
    sys.exit("Gagal: sumber tidak menghasilkan frame. Pengukuran latensi belum dijalankan.")

meta = meta_dasar(cfg) | {"waktu_mulai": mulai.isoformat(timespec="seconds"), "sumber": sumber_info,
                          "detik_diminta": a.detik}
folder = tulis_hasil(p, info, RAIZ / lt["folder_hasil"] / mulai.strftime("%Y%m%d-%H%M%S"), meta)
print(ringkas(p.sampel).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
print({k: v for k, v in info.items()})
print(f"Tersimpan di {folder}")
