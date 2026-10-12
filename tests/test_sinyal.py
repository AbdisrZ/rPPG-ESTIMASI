import numpy as np
import pytest

from rppg.pipeline import StreamingProcessor
from rppg.sinyal import chrom, estimasi_bpm


def sintetis(bpm, durasi=30, fps=15, noise=0.002, seed=0, jitter=0.01):
    rng = np.random.default_rng(seed)
    t = np.arange(0, durasi, 1 / fps) + rng.uniform(-jitter, jitter, int(durasi * fps))
    t = np.sort(t)
    p = np.sin(2 * np.pi * bpm / 60 * t)
    rgb = np.stack([0.60 + 0.004 * p, 0.45 + 0.008 * p, 0.40 + 0.002 * p], axis=1)
    return t, rgb + rng.normal(0, noise, rgb.shape)


@pytest.mark.parametrize("bpm", [55, 72, 100, 140])
def test_streaming_bpm_akurat(bpm):
    t, rgb = sintetis(bpm)
    p = StreamingProcessor()
    k = None
    for ti, ci in zip(t, rgb):
        k = p.push_rgb(ci, ti)
    assert k.bpm is not None
    assert abs(k.bpm - bpm) <= 2.0
    assert k.confidence > 0.3


def test_belum_cukup_data_bpm_none():
    t, rgb = sintetis(72, durasi=3)
    p = StreamingProcessor()
    for ti, ci in zip(t, rgb):
        k = p.push_rgb(ci, ti)
    assert k.bpm is None


def test_derau_murni_confidence_rendah():
    rng = np.random.default_rng(1)
    t = np.arange(0, 30, 1 / 15)
    rgb = 0.5 + rng.normal(0, 0.01, (len(t), 3))
    p = StreamingProcessor()
    for ti, ci in zip(t, rgb):
        k = p.push_rgb(ci, ti)
    assert k.confidence < 0.5


def test_chrom_dan_estimasi_langsung():
    fs = 30
    t = np.arange(0, 10, 1 / fs)
    p = np.sin(2 * np.pi * 1.2 * t)
    rgb = np.stack([0.6 + 0.004 * p, 0.45 + 0.008 * p, 0.4 + 0.002 * p], axis=1)
    bpm, _ = estimasi_bpm(chrom(rgb, fs), fs)
    assert abs(bpm - 72) < 1.0


@pytest.mark.parametrize("metode", ["chrom", "pos", "hijau"])
def test_semua_metode_menemukan_72bpm(metode):
    t, rgb = sintetis(72)
    p = StreamingProcessor(metode=metode)
    for ti, ci in zip(t, rgb):
        k = p.push_rgb(ci, ti)
    assert abs(k.bpm - 72) <= 2.0


def test_metode_tidak_dikenal():
    with pytest.raises(ValueError):
        StreamingProcessor(metode="xyz")
