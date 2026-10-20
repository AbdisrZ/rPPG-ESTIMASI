"""Evaluasi offline: dataset -> (kurangi fps) -> wajah/ROI -> RGB -> StreamingProcessor -> metrik.

Definisi:
- kurangi_fps: grid waktu berjangkar pada timestamp frame pertama dengan jarak 1/fps; tiap slot
  mengambil paling banyak satu frame, yaitu frame pertama dengan t >= slot - toleransi (1/4 periode
  target). Ini simulasi berbasis timestamp, bukan kamera dengan fps tersebut.
- Penyelarasan: estimasi yang dihitung pada waktu t (sampel terakhir jendela) dibandingkan dengan
  rata-rata HR referensi dalam [t - sinyal.jendela_detik, t]; sampel referensi di luar
  evaluasi.hr_ref_valid_bpm diabaikan. Tanpa sampel valid -> hr_ref NaN (tidak dihitung).
- Hanya push yang menghitung ulang BPM (Keadaan.diperbarui) yang dicatat sebagai estimasi.
- Frame tanpa wajah/ROI dilewati (tidak di-push).
- Uji Wilcoxon bertanda (berpasangan) antar metode pada MAE per video, per dataset dan kondisi fps;
  p dikoreksi Holm di dalam tiap kelompok tersebut.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrik import bland_altman, mae, persen_dalam, rmse
from .pipeline import StreamingProcessor
from .wajah import PengekstrakRGB

BELUM = "belum diukur"
TOLERANSI_SLOT = 0.25  # bagian dari periode target; definisi algoritma, bukan parameter eksperimen


def kurangi_fps(frames, fps: float):
    periode = 1.0 / fps
    t0, slot = None, 0
    for f, t in frames:
        if t0 is None:
            t0 = t
        if t >= t0 + slot * periode - TOLERANSI_SLOT * periode:
            yield f, t
            slot = max(slot + 1, int(round((t - t0) / periode)) + 1)


def ekstrak_rgb(frames, pencari, cfg_roi: dict):
    """Yield (t, rata-rata RGB ROI dahi) atau (t, None) bila wajah/ROI tidak ada (lihat PengekstrakRGB)."""
    ex = PengekstrakRGB(pencari, cfg_roi)
    for f, t in frames:
        yield t, ex.proses(f)


def hr_ref_jendela(t_ref, hr_ref, t: float, jendela: float, rentang) -> float:
    t_ref, hr_ref = np.asarray(t_ref), np.asarray(hr_ref, dtype=float)
    m = (t_ref >= t - jendela) & (t_ref <= t) & (hr_ref >= rentang[0]) & (hr_ref <= rentang[1])
    return float(hr_ref[m].mean()) if m.any() else float("nan")


def evaluasi_rekaman(rek, fps: float, metode: list[str], pencari, cfg: dict) -> list[dict]:
    jendela = float(cfg["sinyal"]["jendela_detik"])
    rentang = cfg["evaluasi"]["hr_ref_valid_bpm"]
    t_ref, hr_ref = rek.referensi()
    proc = {m: StreamingProcessor(cfg, metode=m) for m in metode}
    baris = []
    for t, rgb in ekstrak_rgb(kurangi_fps(rek.frames(), fps), pencari, cfg["roi"]):
        if rgb is None:
            continue
        for m, p in proc.items():
            k = p.push_rgb(rgb, t)
            if k.diperbarui:
                hr = hr_ref_jendela(t_ref, hr_ref, t, jendela, rentang)
                baris.append(dict(dataset=rek.dataset, video=rek.id, kondisi_fps=fps, metode=m, t=t,
                                  bpm=k.bpm, confidence=k.confidence, hr_ref=hr, galat=k.bpm - hr))
    return baris


KUNCI = ["dataset", "kondisi_fps", "metode"]


def ringkas(df: pd.DataFrame, batas: float) -> pd.DataFrame:
    baris = []
    for (ds, fps, m), g in df.dropna(subset=["hr_ref"]).groupby(KUNCI, sort=False):
        ba = bland_altman(g.bpm, g.hr_ref) if len(g) > 1 else {"bias": np.nan, "loa_bawah": np.nan,
                                                               "loa_atas": np.nan}
        baris.append(dict(dataset=ds, kondisi_fps=fps, metode=m, n=len(g), n_video=g.video.nunique(),
                          mae=mae(g.bpm, g.hr_ref), rmse=rmse(g.bpm, g.hr_ref),
                          proporsi_dalam_batas=persen_dalam(g.bpm, g.hr_ref, batas), **ba))
    return pd.DataFrame(baris)


def holm(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    hasil = np.full_like(p, np.nan)
    idx = np.where(np.isfinite(p))[0]
    urut = idx[np.argsort(p[idx])]
    m, maks = len(urut), 0.0
    for i, j in enumerate(urut):
        maks = max(maks, min(1.0, (m - i) * p[j]))
        hasil[j] = maks
    return hasil


def uji_wilcoxon(df: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import wilcoxon

    baris = []
    d = df.dropna(subset=["hr_ref"]).assign(ag=lambda x: x.galat.abs())
    for (ds, fps), g in d.groupby(["dataset", "kondisi_fps"], sort=False):
        per_video = g.groupby(["video", "metode"]).ag.mean().unstack()
        kelompok = []
        for a, b in itertools.combinations(list(dict.fromkeys(g.metode)), 2):
            pas = per_video[[a, b]].dropna()
            try:
                stat, p = wilcoxon(pas[a], pas[b])
            except ValueError:  # terlalu sedikit pasangan atau semua selisih nol
                stat, p = np.nan, np.nan
            kelompok.append(dict(dataset=ds, kondisi_fps=fps, metode_a=a, metode_b=b,
                                 n_video=len(pas), statistik=float(stat), p=float(p)))
        for k, ph in zip(kelompok, holm([k["p"] for k in kelompok])):
            k["p_holm"] = float(ph)
        baris += kelompok
    return pd.DataFrame(baris, columns=["dataset", "kondisi_fps", "metode_a", "metode_b", "n_video",
                                        "statistik", "p", "p_holm"])


def plot_bland_altman(g: pd.DataFrame, judul: str, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ba = bland_altman(g.bpm, g.hr_ref)
    rata, beda = (g.bpm + g.hr_ref) / 2, g.bpm - g.hr_ref
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(rata, beda, s=8, alpha=0.5)
    for y, gaya in [(ba["bias"], "-"), (ba["loa_bawah"], "--"), (ba["loa_atas"], "--")]:
        ax.axhline(y, color="k", linestyle=gaya, linewidth=1)
    ax.set_xlabel("Rata-rata estimasi dan referensi (BPM)")
    ax.set_ylabel("Estimasi - referensi (BPM)")
    ax.set_title(judul)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def tulis_keluaran(df: pd.DataFrame, folder, meta: dict, batas: float) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder / "estimasi.csv", index=False)
    ringkas(df, batas).to_csv(folder / "ringkasan.csv", index=False)
    uji_wilcoxon(df).to_csv(folder / "wilcoxon.csv", index=False)
    for (ds, fps, m), g in df.dropna(subset=["hr_ref"]).groupby(KUNCI, sort=False):
        if len(g) > 1:
            fps_s = f"{fps:g}"
            plot_bland_altman(g, f"{ds} {fps_s} fps {m} (n={len(g)})",
                              folder / f"bland_altman_{ds}_{fps_s}fps_{m}.png")
    (folder / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str),
                                      encoding="utf-8")
    return folder


def tabel1(ringkasan: pd.DataFrame | None, dataset, metode, fps: float, batas: float) -> pd.DataFrame:
    """Tabel 1: MAE, RMSE, proporsi |galat| <= batas per dataset dan metode pada satu kondisi fps."""
    kolom = ["Dataset", "Metode", "MAE (BPM)", "RMSE (BPM)", f"<={batas:g} BPM (%)"]
    idx = {}
    if ringkasan is not None and len(ringkasan):
        for r in ringkasan[np.isclose(ringkasan.kondisi_fps, fps)].itertuples():
            idx[(r.dataset, r.metode)] = r
    baris = []
    for ds in dataset:
        for m in metode:
            r = idx.get((ds, m))
            if r is None or r.n == 0:
                baris.append([ds, m, BELUM, BELUM, BELUM])
            else:
                baris.append([ds, m, f"{r.mae:.2f}", f"{r.rmse:.2f}", f"{100 * r.proporsi_dalam_batas:.1f}"])
    return pd.DataFrame(baris, columns=kolom)


def ke_markdown(df: pd.DataFrame) -> str:
    kepala = "| " + " | ".join(df.columns) + " |"
    garis = "|" + "|".join("---" for _ in df.columns) + "|"
    isi = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([kepala, garis, *isi])
