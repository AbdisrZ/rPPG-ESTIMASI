"""Latensi per tahap: python scripts/latensi.py [--video berkas] [--detik N] [--metode chrom] [--fps 15]

Tanpa --video memakai SumberKamera (config kamera). --fps (bawaan latensi.fps) menerapkan kurangi_fps
berbasis timestamp pada sumber sebelum diproses; --fps 0 mematikannya.
Hasil: results/latensi/<waktu>/{sampel.csv, ringkasan.csv, meta.json}. Hanya waktu yang disimpan, bukan frame.
"""
import argparse
import sys
from datetime import datetime

from rppg.config import RAIZ, muat
from rppg.evaluasi import kurangi_fps
from rppg.jejak import meta_dasar
from rppg.latensi import ringkas, tulis_hasil, ukur_pipeline
from rppg.sumber import KameraGagal, SumberKamera, baca_video
from rppg.wajah import PencariMediaPipe

cfg = muat()
lt = cfg["latensi"]
ap = argparse.ArgumentParser()
ap.add_argument("--video", default=None)
ap.add_argument("--detik", type=float, default=float(lt["detik_bawaan"]))
ap.add_argument("--metode", default=lt["metode"], choices=["chrom", "pos", "hijau"])
ap.add_argument("--fps", type=float, default=float(lt["fps"]))
a = ap.parse_args()


def batasi(frames):
    return kurangi_fps(frames, a.fps) if a.fps > 0 else frames


pencari = PencariMediaPipe(cfg)
mulai = datetime.now()
if a.video:
    sumber_info = {"jenis": "video", "berkas": a.video}
    try:
        p, info = ukur_pipeline(batasi(baca_video(a.video)), pencari, cfg, a.metode, detik=a.detik)
    except RuntimeError as e:
        sys.exit(f"Gagal: {e}")
else:
    import cv2

    try:
        kamera = SumberKamera(cfg)
    except RuntimeError as e:
        sys.exit(f"Gagal: {e}. Pengukuran latensi belum dijalankan.")
    try:
        p, info = ukur_pipeline(batasi(iter(kamera)), pencari, cfg, a.metode, detik=a.detik,
                                ke_rgb=lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        sumber_info = {"jenis": "kamera", "driver": kamera.info(),
                       "n_dibaca": kamera.n_dibaca, "n_duplikat": kamera.n_duplikat}
    except KameraGagal as e:
        sys.exit(f"Gagal di tengah sesi: {e}. Hasil sesi ini tidak disimpan.")
    finally:
        kamera.tutup()

if info["n_frame"] == 0:
    sys.exit("Gagal: sumber tidak menghasilkan frame. Pengukuran latensi belum dijalankan.")

meta = meta_dasar(cfg) | {"waktu_mulai": mulai.isoformat(timespec="seconds"), "sumber": sumber_info,
                          "detik_diminta": a.detik, "fps_target": a.fps or None,
                          "catatan_fps": "fps_target dicapai dengan kurangi_fps berbasis timestamp pada "
                                         "aliran sumber; frame yang dibuang tidak diproses dan tidak diukur."}
folder = tulis_hasil(p, info, RAIZ / lt["folder_hasil"] / mulai.strftime("%Y%m%d-%H%M%S"), meta)
print(ringkas(p.sampel).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
print({k: v for k, v in info.items()})
print(f"Tersimpan di {folder}")
