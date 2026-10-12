"""Fungsi murni pemrosesan sinyal rPPG (tanpa state, mudah diuji)."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def resampel_seragam(t: np.ndarray, x: np.ndarray, fs: float):
    """Interpolasi linear sampel bertimestamp tak teratur ke grid seragam fs Hz."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    grid = np.arange(t[0], t[-1], 1.0 / fs)
    if x.ndim == 1:
        return grid, np.interp(grid, t, x)
    return grid, np.stack([np.interp(grid, t, x[:, k]) for k in range(x.shape[1])], axis=1)


def bandpass(x: np.ndarray, fs: float, band=(0.7, 4.0), orde: int = 3) -> np.ndarray:
    nyq = fs / 2.0
    hi = min(band[1], 0.95 * nyq)
    b, a = butter(orde, [band[0] / nyq, hi / nyq], btype="band")
    pad = 3 * max(len(a), len(b))
    if len(x) <= pad:
        raise ValueError("sinyal terlalu pendek untuk filter")
    return filtfilt(b, a, x)


def chrom(rgb: np.ndarray, fs: float, band=(0.7, 4.0)) -> np.ndarray:
    """CHROM (de Haan & Jeanne, 2013). rgb: (N,3) pada grid seragam fs."""
    norm = rgb / np.mean(rgb, axis=0, keepdims=True)
    r, g, b = norm[:, 0], norm[:, 1], norm[:, 2]
    xs = 3 * r - 2 * g
    ys = 1.5 * r + g - 1.5 * b
    xf, yf = bandpass(xs, fs, band), bandpass(ys, fs, band)
    alpha = np.std(xf) / (np.std(yf) + 1e-12)
    return xf - alpha * yf


def estimasi_bpm(pulsa: np.ndarray, fs: float, band=(0.7, 4.0), nfft_min: int = 4096):
    """Puncak spektrum (zero-padding + interpolasi parabola) -> (bpm, confidence).

    confidence = rasio daya di sekitar puncak (±0.1 Hz) + harmonik-2 terhadap
    total daya dalam pita (0..1).
    """
    x = (pulsa - np.mean(pulsa)) * np.hanning(len(pulsa))
    nfft = max(nfft_min, int(2 ** np.ceil(np.log2(len(x)))))
    spek = np.abs(np.fft.rfft(x, nfft)) ** 2
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    m = (f >= band[0]) & (f <= band[1])
    idx = np.where(m)[0]
    k = idx[np.argmax(spek[idx])]
    if 0 < k < len(spek) - 1:  # interpolasi parabola
        a, b, c = np.log(spek[k - 1] + 1e-20), np.log(spek[k] + 1e-20), np.log(spek[k + 1] + 1e-20)
        d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
        fp = f[k] + d * (f[1] - f[0])
    else:
        fp = f[k]
    total = spek[idx].sum() + 1e-20
    dekat = m & ((np.abs(f - fp) <= 0.1) | (np.abs(f - 2 * fp) <= 0.1))
    conf = float(spek[dekat].sum() / total)
    return float(fp * 60.0), min(conf, 1.0)


def hijau(rgb: np.ndarray, fs: float, band=(0.7, 4.0)) -> np.ndarray:
    """Pembanding: kanal hijau ternormalisasi lalu band-pass."""
    g = rgb[:, 1] / np.mean(rgb[:, 1])
    return bandpass(g, fs, band)


def pos(rgb: np.ndarray, fs: float, band=(0.7, 4.0)) -> np.ndarray:
    """POS (Wang dkk., 2017), versi jendela tunggal (tanpa overlap-add 1,6 detik)."""
    c = rgb / np.mean(rgb, axis=0, keepdims=True)
    s1 = c[:, 1] - c[:, 2]
    s2 = c[:, 1] + c[:, 2] - 2 * c[:, 0]
    s1f, s2f = bandpass(s1, fs, band), bandpass(s2, fs, band)
    alpha = np.std(s1f) / (np.std(s2f) + 1e-12)
    return s1f + alpha * s2f


METODE = {"chrom": chrom, "pos": pos, "hijau": hijau}
