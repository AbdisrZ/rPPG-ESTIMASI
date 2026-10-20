"""Uji kamera tercatat: rekam log read(), ringkas fps nyata, rasio duplikat, dan jitter.

Definisi (dihitung hanya dari frame unik, kecuali rasio duplikat):
- dt          : selisih timestamp antar frame unik berurutan.
- fps_nyata   : (n_unik - 1) / (t_unik_terakhir - t_unik_pertama).
- jitter      : |dt - median(dt)|; dilaporkan rata-rata dan persentil 95.
- rasio_duplikat : n_duplikat / n_dibaca (semua read() yang berhasil).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def jalankan_cek(sumber, detik: float) -> list[tuple[float, bool]]:
    """Baca sumber selama `detik` (diukur dari timestamp frame pertama). Kembalikan [(t, duplikat)]."""
    log = []
    while (r := sumber.baca_mentah()) is not None:
        _, t, dup = r
        if log and t - log[0][0] >= detik:
            break
        log.append((t, bool(dup)))
    return log


def _ms(x):
    return None if x is None else float(x) * 1000.0


def ringkas(t, duplikat) -> dict:
    t = np.asarray(t, dtype=float)
    dup = np.asarray(duplikat, dtype=bool)
    tu = t[~dup]
    n = len(t)
    r = {"n_dibaca": int(n), "n_unik": int(len(tu)), "n_duplikat": int(dup.sum()),
         "rasio_duplikat": float(dup.mean()) if n else None,
         "durasi_s": float(t[-1] - t[0]) if n > 1 else None,
         "fps_baca": None, "fps_nyata": None,
         "dt_rata_ms": None, "dt_median_ms": None, "dt_p95_ms": None,
         "jitter_rata_ms": None, "jitter_p95_ms": None}
    if n > 1 and t[-1] > t[0]:
        r["fps_baca"] = float((n - 1) / (t[-1] - t[0]))
    if len(tu) > 1:
        dt = np.diff(tu)
        jit = np.abs(dt - np.median(dt))
        r.update(fps_nyata=float((len(tu) - 1) / (tu[-1] - tu[0])),
                 dt_rata_ms=_ms(dt.mean()), dt_median_ms=_ms(np.median(dt)),
                 dt_p95_ms=_ms(np.percentile(dt, 95)),
                 jitter_rata_ms=_ms(jit.mean()), jitter_p95_ms=_ms(np.percentile(jit, 95)))
    return r


def tulis_hasil(log, folder, meta: dict) -> Path:
    """Tulis frame.csv (satu baris per read) dan ringkasan.json ke `folder`."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    t = [x[0] for x in log]
    dup = [x[1] for x in log]
    t0 = t[0] if t else 0.0
    with open(folder / "frame.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["indeks", "t", "dt_ms", "duplikat"])
        for i, (ti, di) in enumerate(log):
            dt = "" if i == 0 else f"{(ti - t[i - 1]) * 1000:.3f}"
            w.writerow([i, f"{ti - t0:.6f}", dt, int(di)])
    isi = {"meta": meta, "definisi": __doc__.strip(), "statistik": ringkas(t, dup)}
    (folder / "ringkasan.json").write_text(json.dumps(isi, indent=2, ensure_ascii=False), encoding="utf-8")
    return folder
