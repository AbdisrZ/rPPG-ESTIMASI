"""Bangkitkan Tabel 1 dari ringkasan.csv: python scripts/tabel1.py [ringkasan.csv] [--fps 15] [--keluar berkas.md]

Tanpa ringkasan.csv semua sel ditulis "belum diukur".
"""
import argparse

import pandas as pd

from rppg.config import muat
from rppg.evaluasi import ke_markdown, tabel1

cfg = muat()
ev = cfg["evaluasi"]
ap = argparse.ArgumentParser()
ap.add_argument("ringkasan", nargs="?", default=None)
ap.add_argument("--fps", type=float, default=float(ev["kondisi_fps"][0]))
ap.add_argument("--keluar", default=None)
a = ap.parse_args()

r = pd.read_csv(a.ringkasan) if a.ringkasan else None
md = ke_markdown(tabel1(r, ["ubfc", "pure"], list(ev["metode"]), a.fps, float(ev["batas_galat_bpm"])))
print(f"Tabel 1 ({a.fps:g} fps; sumber: {a.ringkasan or 'tidak ada'})\n{md}")
if a.keluar:
    with open(a.keluar, "w", encoding="utf-8") as f:
        f.write(md + "\n")
