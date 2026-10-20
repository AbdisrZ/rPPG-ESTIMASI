"""Deteksi wajah (MediaPipe Tasks FaceDetector) dan ROI dahi."""
from __future__ import annotations

from contextlib import nullcontext

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


class PengekstrakRGB:
    """Frame RGB -> rata-rata RGB ROI dahi, atau None bila wajah/ROI tidak ada.

    Deteksi wajah tiap roi.deteksi_tiap_n_frame; kotak terakhir dipakai di antaranya.
    `pengukur` (opsional) mencatat tahap deteksi_wajah dan roi_rgb.
    """

    def __init__(self, pencari, cfg_roi: dict, pengukur=None):
        self.pencari, self.cfg_roi = pencari, cfg_roi
        self.n = int(cfg_roi["deteksi_tiap_n_frame"])
        self._ukur = pengukur.ukur if pengukur else (lambda _: nullcontext())
        self._i, self._kotak = 0, None
        self.roi_terakhir = None  # (x0, y0, x1, y1) frame terakhir, untuk pratinjau antarmuka

    def proses(self, frame):
        self.roi_terakhir = None
        if frame is None or frame.size == 0:  # frame kosong: tidak dihitung dalam irama deteksi
            return None
        if self._i % self.n == 0:
            with self._ukur("deteksi_wajah"):
                self._kotak = self.pencari.cari(frame)
        self._i += 1
        with self._ukur("roi_rgb"):
            roi = roi_dahi(self._kotak, frame.shape, self.cfg_roi) if self._kotak is not None else None
            self.roi_terakhir = roi
            if roi is None:
                return None
            x0, y0, x1, y1 = roi
            return frame[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)


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
