"""Pemuat UBFC-rPPG DATASET_2 (Bobbia dkk., 2017).

ASUMSI FORMAT, perlu diverifikasi terhadap berkas asli dan pembaca resmi yang disertakan di unduhan.
Halaman resmi (sites.google.com/view/ybenezeth/ubfcrppg) hanya menyebut 30 fps, 640x480,
dan oksimeter CMS50E; nama dan isi berkas tidak dijelaskan di sana.
- <root>/<subjek>/vid.avi
- <root>/<subjek>/ground_truth.txt: tiga baris angka dipisah spasi, berurutan
  sinyal PPG, HR (BPM), timestamp (detik).
- Waktu frame video = indeks / fps kontainer (avi tidak menyimpan timestamp per frame).
Pemeriksaan: tiga baris sama panjang, timestamp tidak menurun, dan rentang timestamp
sebanding dengan durasi video; bila tidak, FormatTidakSesuai.
DATASET_1 (gtdump.xmp) belum didukung.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import FormatTidakSesuai, Rekaman


def _buka(p: Path):
    import cv2

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise FormatTidakSesuai(f"video tidak bisa dibuka: {p}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        cap.release()
        raise FormatTidakSesuai(f"fps kontainer tidak terbaca: {p}")
    return cap, fps


def _frames(p: Path):
    import cv2

    cap, fps = _buka(p)
    i = 0
    try:
        while True:
            ok, f = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(f, cv2.COLOR_BGR2RGB), i / fps
            i += 1
    finally:
        cap.release()


def _durasi_video(p: Path) -> float:
    import cv2

    cap, fps = _buka(p)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return n / fps


def _referensi(folder: Path):
    p = folder / "ground_truth.txt"
    baris = [np.array(b.split(), dtype=float) for b in p.read_text().splitlines() if b.strip()]
    if len(baris) < 3 or len({len(b) for b in baris[:3]}) != 1 or len(baris[0]) < 2:
        raise FormatTidakSesuai(f"{p}: diharapkan 3 baris (PPG, HR, timestamp) sama panjang")
    _, hr, t = baris[:3]
    if np.any(np.diff(t) < 0):
        raise FormatTidakSesuai(f"{p}: baris ke-3 bukan timestamp naik")
    durasi = _durasi_video(folder / "vid.avi")
    rentang = t[-1] - t[0]
    if durasi > 0 and not (0.5 <= rentang / durasi <= 2.0):
        raise FormatTidakSesuai(f"{p}: rentang timestamp {rentang:.1f} tidak sebanding durasi video "
                                f"{durasi:.1f} s; satuan mungkin bukan detik")
    return t - t[0], hr


def daftar(root: Path) -> list[Rekaman]:
    hasil = []
    for vid in sorted(root.glob("*/vid.avi")):
        d = vid.parent
        if (d / "ground_truth.txt").exists():
            hasil.append(Rekaman("ubfc", d.name, lambda v=vid: _frames(v), lambda d=d: _referensi(d)))
    return hasil
