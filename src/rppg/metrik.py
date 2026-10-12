"""Metrik evaluasi estimasi BPM terhadap referensi."""
from __future__ import annotations

import numpy as np


def _pasangan(est, ref):
    est, ref = np.asarray(est, float), np.asarray(ref, float)
    ok = np.isfinite(est) & np.isfinite(ref)
    return est[ok], ref[ok]


def mae(est, ref) -> float:
    e, r = _pasangan(est, ref)
    return float(np.mean(np.abs(e - r)))


def rmse(est, ref) -> float:
    e, r = _pasangan(est, ref)
    return float(np.sqrt(np.mean((e - r) ** 2)))


def persen_dalam(est, ref, batas: float = 5.0) -> float:
    """Proporsi estimasi dengan |galat| <= batas BPM (0..1)."""
    e, r = _pasangan(est, ref)
    return float(np.mean(np.abs(e - r) <= batas))


def bland_altman(est, ref) -> dict:
    e, r = _pasangan(est, ref)
    d = e - r
    bias, sd = float(np.mean(d)), float(np.std(d, ddof=1))
    return {"bias": bias, "loa_bawah": bias - 1.96 * sd, "loa_atas": bias + 1.96 * sd}
