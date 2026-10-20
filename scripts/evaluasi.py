"""Evaluasi offline: python scripts/evaluasi.py [--dataset ubfc pure] [--fps 15] [--metode chrom pos hijau]

Jalur dataset dari config/lokal.yaml (kunci dataset.<nama>). Dataset yang tidak ada dilewati dengan
pesan; bila tidak ada satu pun, skrip berhenti tanpa menulis hasil.
Keluaran: results/evaluasi/<waktu>/{estimasi.csv, ringkasan.csv, wilcoxon.csv, bland_altman_*.png,
tabel1_<fps>fps.md, meta.json}.
"""
import argparse
import sys
from datetime import datetime

import pandas as pd

from rppg.config import RAIZ, muat
from rppg.data import DatasetTidakAda, FormatTidakSesuai, muat_dataset
from rppg.evaluasi import evaluasi_rekaman, ke_markdown, ringkas, tabel1, tulis_keluaran
from rppg.jejak import meta_dasar
from rppg.wajah import PencariMediaPipe

cfg = muat()
ev = cfg["evaluasi"]
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", nargs="+", default=["ubfc", "pure"], choices=["ubfc", "pure"])
ap.add_argument("--fps", nargs="+", type=float, default=[float(x) for x in ev["kondisi_fps"]])
ap.add_argument("--metode", nargs="+", default=list(ev["metode"]), choices=["chrom", "pos", "hijau"])
ap.add_argument("--maks-video", type=int, default=None, help="batasi jumlah video per dataset (uji cepat)")
a = ap.parse_args()

rekaman, tidak_ada = {}, {}
for ds in a.dataset:
    try:
        rekaman[ds] = muat_dataset(ds, cfg["dataset"].get(ds))[:a.maks_video]
    except DatasetTidakAda as e:
        tidak_ada[ds] = str(e)
        print(f"[lewati] {e}")
if not rekaman:
    sys.exit("Berhenti: tidak ada dataset yang tersedia. Tidak ada hasil yang ditulis; "
             "semua sel Tabel 1 tetap 'belum diukur' (lihat scripts/tabel1.py).")

mulai = datetime.now()
pencari = PencariMediaPipe(cfg)
baris = []
for ds, daftar in rekaman.items():
    for i, rek in enumerate(daftar, 1):
        for fps in a.fps:
            print(f"{ds} {i}/{len(daftar)} {rek.id} @ {fps:g} fps ...", flush=True)
            try:
                baris += evaluasi_rekaman(rek, fps, a.metode, pencari, cfg)
            except FormatTidakSesuai as e:
                sys.exit(f"Berhenti: format {ds} tidak sesuai asumsi pemuat: {e}")
if not baris:
    sys.exit("Berhenti: tidak ada estimasi yang dihasilkan (wajah tidak terdeteksi atau video terlalu pendek).")

df = pd.DataFrame(baris)
batas = float(ev["batas_galat_bpm"])
meta = meta_dasar(cfg) | {
    "waktu_mulai": mulai.isoformat(timespec="seconds"),
    "dataset": {ds: [r.id for r in d] for ds, d in rekaman.items()},
    "dataset_tidak_ada": tidak_ada, "kondisi_fps": a.fps, "metode": a.metode, "batas_galat_bpm": batas,
    "catatan_fps": "Kondisi fps di bawah fps asli dataset (30) dibuat dengan kurangi_fps berbasis "
                   "timestamp, bukan rekaman kamera dengan fps tersebut."}
folder = tulis_keluaran(df, RAIZ / ev["folder_hasil"] / mulai.strftime("%Y%m%d-%H%M%S"), meta, batas)
r = ringkas(df, batas)
for fps in a.fps:
    md = ke_markdown(tabel1(r, ["ubfc", "pure"], ["chrom", "pos", "hijau"], fps, batas))
    (folder / f"tabel1_{fps:g}fps.md").write_text(md + "\n", encoding="utf-8")
    print(f"\nTabel 1 ({fps:g} fps):\n{md}")
print(f"\nTersimpan di {folder}")
