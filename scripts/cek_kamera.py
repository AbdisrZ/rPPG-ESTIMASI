"""Uji kamera tercatat: python scripts/cek_kamera.py [--detik N]

Hasil: results/kamera/<waktu>/frame.csv dan ringkasan.json.
"""
import argparse
import platform
import subprocess
import sys
from datetime import datetime

from rppg.cek_kamera import jalankan_cek, tulis_hasil
from rppg.config import RAIZ, muat
from rppg.sumber import SumberKamera


def hash_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=RAIZ,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


cfg = muat()
ap = argparse.ArgumentParser()
ap.add_argument("--detik", type=float, default=float(cfg["cek_kamera"]["detik_bawaan"]))
a = ap.parse_args()

try:
    sumber = SumberKamera(cfg)
except RuntimeError as e:
    sys.exit(f"Gagal: {e}. Uji kamera belum dijalankan.")

mulai = datetime.now()
try:
    log = jalankan_cek(sumber, a.detik)
    info = sumber.info()
finally:
    sumber.tutup()

if len(log) < 2:
    sys.exit("Gagal: kamera tidak menghasilkan frame. Uji kamera belum dijalankan.")

meta = {"waktu_mulai": mulai.isoformat(timespec="seconds"), "detik_diminta": a.detik,
        "kamera_config": cfg["kamera"], "driver": info, "commit": hash_commit(),
        "python": platform.python_version(), "os": platform.platform()}
folder = tulis_hasil(log, RAIZ / cfg["cek_kamera"]["folder_hasil"] / mulai.strftime("%Y%m%d-%H%M%S"), meta)
print(f"Tersimpan di {folder}")
print((folder / "ringkasan.json").read_text(encoding="utf-8"))
