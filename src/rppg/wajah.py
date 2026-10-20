"""Deteksi wajah (MediaPipe Tasks FaceDetector) dan ROI dahi."""
from __future__ import annotations

from .config import RAIZ


def roi_dahi(kotak, bentuk, cfg_roi: dict):
    """kotak wajah (x, y, w, h) -> ROI dahi (x0, y0, x1, y1) terpotong ke frame, atau None.

    Lebar ROI = roi.lebar * w, berpusat horizontal pada wajah; tinggi = roi.tinggi * h;
    tepi atas = y + roi.geser_y * h.
    """
    x, y, w, h = kotak
    tinggi_f, lebar_f = bentuk[:2]
    cx, rw, rh = x + w / 2, cfg_roi["lebar"] * w, cfg_roi["tinggi"] * h
    y0f = y + cfg_roi["geser_y"] * h
    x0, x1 = max(0, round(cx - rw / 2)), min(lebar_f, round(cx + rw / 2))
    y0, y1 = max(0, round(y0f)), min(tinggi_f, round(y0f + rh))
    if x1 <= x0 or y1 <= y0:
        return None
    return int(x0), int(y0), int(x1), int(y1)


class PencariMediaPipe:
    """cari(frame RGB) -> (x, y, w, h) wajah terbesar, atau None."""

    def __init__(self, cfg: dict):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        opts = vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(RAIZ / cfg["wajah"]["model"])),
            running_mode=vision.RunningMode.IMAGE)
        self._det = vision.FaceDetector.create_from_options(opts)

    def cari(self, frame):
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame)
        hasil = self._det.detect(img).detections
        if not hasil:
            return None
        b = max((d.bounding_box for d in hasil), key=lambda b: b.width * b.height)
        return b.origin_x, b.origin_y, b.width, b.height
