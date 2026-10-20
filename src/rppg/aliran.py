"""Loop per frame tanpa Qt: frame -> ROI -> StreamingProcessor -> Hasil dengan status dan pesan.

Lapisan ini dipakai antarmuka (WP5) supaya logika dan pesan status bisa dites tanpa Qt.
"""
from __future__ import annotations

from dataclasses import dataclass

from .pipeline import StreamingProcessor
from .wajah import PengekstrakRGB

PESAN = {
    "ok": "Mengukur",
    "mengumpulkan": "Mengumpulkan sinyal ({durasi:.0f}/{jendela:.0f} s)",
    "direset": "Wajah sempat hilang lebih dari {jeda:g} s, pengukuran diulang",
    "timestamp_mundur": "Timestamp frame mundur, frame dilewati",
    "sampel_tidak_valid": "Nilai warna tidak valid, frame dilewati",
    "wajah_hilang": "Wajah tidak terdeteksi",
    "frame_kosong": "Frame kosong dari sumber",
    "kamera_gagal": "Kamera berhenti mengirim frame",
}


@dataclass
class Hasil:
    t: float
    bpm: float | None
    confidence: float | None
    status: str
    pesan: str
    ada_wajah: bool
    n_sampel: int = 0


class Aliran:
    def __init__(self, cfg: dict, pencari, metode: str = "chrom"):
        self.cfg = cfg
        self.ex = PengekstrakRGB(pencari, cfg["roi"])
        self.proc = StreamingProcessor(cfg, metode=metode)
        self.jendela = float(cfg["sinyal"]["jendela_detik"])
        self.jeda = float(cfg["ketahanan"]["jeda_maks_detik"])

    def pesan(self, status: str, durasi: float = 0.0) -> str:
        return PESAN[status].format(durasi=durasi, jendela=self.jendela, jeda=self.jeda)

    def _tanpa_sampel(self, t: float, status: str) -> Hasil:
        # wajah/frame hilang lebih lama dari batas: BPM lama tidak lagi ditampilkan
        tt = self.proc.t_terakhir
        if tt is not None and t - tt > self.jeda:
            self.proc.reset()
        k = self.proc.keadaan(t, status)
        return Hasil(t, k.bpm, k.confidence, status, self.pesan(status), False, k.n_sampel)

    def proses(self, frame, t: float) -> Hasil:
        if frame is None or frame.size == 0:
            return self._tanpa_sampel(t, "frame_kosong")
        rgb = self.ex.proses(frame)
        if rgb is None:
            return self._tanpa_sampel(t, "wajah_hilang")
        k = self.proc.push_rgb(rgb, t)
        return Hasil(t, k.bpm, k.confidence, k.status, self.pesan(k.status, k.durasi_s), True, k.n_sampel)
