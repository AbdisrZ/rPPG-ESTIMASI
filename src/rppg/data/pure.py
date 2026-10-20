"""Pemuat PURE (Stricker dkk., 2014).

ASUMSI FORMAT, perlu diverifikasi terhadap berkas asli (akses lewat permintaan ke TU Ilmenau).
Halaman resmi hanya menyebut 30 Hz, 640x480, oksimeter CMS50E 60 Hz; struktur berkas tidak dijelaskan.
- <root>/<id>/<id>.json dan gambar <root>/<id>/<id>/Image<timestamp_ns>.png
  (atau gambar langsung di <root>/<id>/).
- JSON: "/FullPackage" = daftar {"Timestamp": ns, "Value": {"pulseRate": BPM, ...}},
  "/Image" = daftar {"Timestamp": ns}.
- Nol waktu = timestamp gambar pertama; frame dan referensi memakai nol yang sama.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from . import FormatTidakSesuai, Rekaman

POLA = re.compile(r"Image(\d+)\.png$")


def _gambar(folder: Path) -> list[tuple[int, Path]]:
    g = [(int(m.group(1)), p) for p in folder.glob("Image*.png") if (m := POLA.search(p.name))]
    return sorted(g)


def _frames(gambar):
    import cv2

    t0 = gambar[0][0]
    for ts, p in gambar:
        f = cv2.imread(str(p))
        if f is None:
            raise FormatTidakSesuai(f"gambar tidak terbaca: {p}")
        yield cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (ts - t0) / 1e9


def _referensi(p_json: Path, t0: int):
    isi = json.loads(p_json.read_text())
    try:
        paket = isi["/FullPackage"]
        t = np.array([x["Timestamp"] for x in paket], dtype=np.int64)
        hr = np.array([x["Value"]["pulseRate"] for x in paket], dtype=float)
    except (KeyError, TypeError) as e:
        raise FormatTidakSesuai(f"{p_json}: kunci /FullPackage[].Timestamp/Value.pulseRate tidak ada ({e})")
    if len(t) < 2 or np.any(np.diff(t) < 0):
        raise FormatTidakSesuai(f"{p_json}: timestamp referensi kosong atau tidak naik")
    return (t - t0) / 1e9, hr


def daftar(root: Path) -> list[Rekaman]:
    hasil = []
    for p_json in sorted(root.rglob("*.json")):
        gambar = _gambar(p_json.parent / p_json.stem) or _gambar(p_json.parent)
        if not gambar:
            continue
        hasil.append(Rekaman("pure", p_json.stem, lambda g=gambar: _frames(g),
                             lambda p=p_json, t0=gambar[0][0]: _referensi(p, t0)))
    return hasil
