"""Pemuat dataset diuji dengan folder sintetis yang meniru format asumsi (lihat docstring pemuat)."""
import json

import cv2
import numpy as np
import pytest

from rppg.data import DatasetTidakAda, FormatTidakSesuai, muat_dataset


def tulis_ubfc(root, nama="subject1", n=45, fps=30.0):
    d = root / nama
    d.mkdir(parents=True)
    vw = cv2.VideoWriter(str(d / "vid.avi"), cv2.VideoWriter_fourcc(*"MJPG"), fps, (32, 24))
    for i in range(n):
        vw.write(np.full((24, 32, 3), (10, 20, 200), dtype=np.uint8))  # BGR: merah dominan
    vw.release()
    t = np.arange(n) / fps
    ppg = np.sin(2 * np.pi * 1.2 * t)
    hr = np.full(n, 72.0)
    with open(d / "ground_truth.txt", "w") as f:
        for baris in (ppg, hr, t):
            f.write("  ".join(f"{v:.6e}" for v in baris) + "\n")
    return d


def tulis_pure(root, nama="01-01", n=12, fps=30.0):
    d = root / nama
    (d / nama).mkdir(parents=True)
    t0 = 1_400_000_000_000_000_000
    ts_img = [t0 + int(i * 1e9 / fps) for i in range(n)]
    for ts in ts_img:
        cv2.imwrite(str(d / nama / f"Image{ts}.png"), np.full((24, 32, 3), (10, 20, 200), np.uint8))
    paket = [{"Timestamp": t0 + int(k * 1e9 / 60),
              "Value": {"pulseRate": 70 + (k % 2), "o2saturation": 98, "signalStrength": 4,
                        "waveform": 50}} for k in range(2 * n)]
    isi = {"/FullPackage": paket, "/Image": [{"Timestamp": ts} for ts in ts_img]}
    (d / f"{nama}.json").write_text(json.dumps(isi))
    return d


def test_ubfc_frame_rgb_dan_referensi(tmp_path):
    tulis_ubfc(tmp_path)
    rek = muat_dataset("ubfc", tmp_path)
    assert [r.id for r in rek] == ["subject1"]
    frames = list(rek[0].frames())
    assert len(frames) == 45
    f0, t0 = frames[0]
    assert f0.shape == (24, 32, 3) and f0[..., 0].mean() > f0[..., 2].mean()  # RGB, bukan BGR
    assert np.isclose(frames[1][1] - t0, 1 / 30)
    t, hr = rek[0].referensi()
    assert len(t) == 45 and np.allclose(hr, 72)


def test_ubfc_format_salah(tmp_path):
    d = tulis_ubfc(tmp_path)
    (d / "ground_truth.txt").write_text("1 2 3\n")
    with pytest.raises(FormatTidakSesuai):
        muat_dataset("ubfc", tmp_path)[0].referensi()


def test_pure_timestamp_ns_ke_detik(tmp_path):
    tulis_pure(tmp_path)
    rek = muat_dataset("pure", tmp_path)
    assert [r.id for r in rek] == ["01-01"]
    frames = list(rek[0].frames())
    assert len(frames) == 12
    assert frames[0][1] == 0.0 and np.isclose(frames[1][1], 1 / 30, atol=1e-6)
    assert frames[0][0][..., 0].mean() > frames[0][0][..., 2].mean()
    t, hr = rek[0].referensi()
    assert len(t) == 24 and np.isclose(t[1], 1 / 60, atol=1e-6) and set(hr) == {70.0, 71.0}


def test_pure_kunci_hilang(tmp_path):
    d = tulis_pure(tmp_path)
    (d / "01-01.json").write_text(json.dumps({"lain": []}))
    with pytest.raises(FormatTidakSesuai):
        muat_dataset("pure", tmp_path)[0].referensi()


@pytest.mark.parametrize("nama", ["ubfc", "pure"])
def test_dataset_tidak_ada(tmp_path, nama):
    with pytest.raises(DatasetTidakAda):
        muat_dataset(nama, None)
    with pytest.raises(DatasetTidakAda):
        muat_dataset(nama, tmp_path / "tidak_ada")
    with pytest.raises(DatasetTidakAda):
        muat_dataset(nama, tmp_path)  # folder ada tapi kosong
