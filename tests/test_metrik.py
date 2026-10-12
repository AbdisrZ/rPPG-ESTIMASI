import numpy as np

from rppg.metrik import bland_altman, mae, persen_dalam, rmse


def test_metrik_dasar():
    est, ref = [70, 75, 80], [72, 72, 72]
    assert np.isclose(mae(est, ref), (2 + 3 + 8) / 3)
    assert np.isclose(rmse(est, ref), np.sqrt((4 + 9 + 64) / 3))
    assert np.isclose(persen_dalam(est, ref, 5), 2 / 3)


def test_nan_diabaikan():
    assert mae([70, np.nan], [72, 72]) == 2.0


def test_bland_altman():
    r = bland_altman([71, 73, 75], [70, 72, 74])
    assert np.isclose(r["bias"], 1.0)
