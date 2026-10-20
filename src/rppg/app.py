"""Antarmuka tipis PySide6 + pyqtgraph: python -m rppg.app [--video berkas] [--metode m] [--fps n] [--rekam]

Hanya tampilan dan kabel sinyal Qt; logika ada di rppg.sesi (dites tanpa Qt).
Pembacaan sumber dan pemrosesan berjalan di QThread supaya antarmuka tidak macet saat kamera menunggu frame.
"""
from __future__ import annotations

import sys

import numpy as np

# mediapipe harus diimpor sebelum PySide6: hook impor shiboken bentrok dengan `six`
# (ditarik mediapipe -> matplotlib -> dateutil) dan melempar AttributeError '_SixMetaPathImporter'.
from .wajah import PencariMediaPipe  # noqa: E402  (impor mediapipe terjadi di sini)
import mediapipe  # noqa: E402,F401

import pyqtgraph as pg  # noqa: E402
from PySide6 import QtCore, QtGui, QtWidgets

from .aliran import PESAN
from .config import RAIZ, muat
from .sesi import TEKS_BUKAN_MEDIS, Sesi, buat_parser, buka_sumber, teks_tampilan
from .sinyal import METODE
from .sumber import KameraGagal

KAMERA = "Kamera"
VIDEO = "Berkas video..."


class Pekerja(QtCore.QThread):
    terbuka = QtCore.Signal(dict)       # info sumber
    tampil = QtCore.Signal(object)      # Tampilan
    berakhir = QtCore.Signal(str)       # pesan akhir

    def __init__(self, sesi: Sesi, cfg: dict, video: str | None, fps: float):
        super().__init__()
        self.sesi, self.cfg, self.video, self.fps = sesi, cfg, video, fps
        self._henti = False

    def hentikan(self):
        self._henti = True

    def run(self):
        try:
            src = buka_sumber(self.cfg, self.video, self.fps)
        except RuntimeError as e:
            self.berakhir.emit(f"Sumber gagal dibuka: {e}")
            return
        self.terbuka.emit(src.info)
        pesan = PESAN["video_selesai"] if self.video else "Kamera berhenti"
        try:
            for f, t in src.frames:
                if self._henti:
                    pesan = "Dihentikan"
                    break
                self.tampil.emit(self.sesi.langkah(f, t, src.ke_rgb))
        except KameraGagal as e:
            pesan = f"{PESAN['kamera_gagal']}: {e}"
        except Exception as e:  # tampilkan, jangan biarkan utas mati diam-diam
            pesan = f"Galat: {e!r}"
        finally:
            src.tutup()
        self.berakhir.emit(pesan)


class Jendela(QtWidgets.QWidget):
    def __init__(self, cfg: dict, pencari, args):
        super().__init__()
        self.cfg, self.pencari = cfg, pencari
        self.video, self.fps, self.rekam_awal = args.video, args.fps, args.rekam
        self.pekerja: Pekerja | None = None
        self.sesi: Sesi | None = None
        self.info_sumber: dict = {}
        self._pulsa_terakhir = None
        self.setWindowTitle("rPPG CHROM - estimasi denyut jantung")

        self.pilih_sumber = QtWidgets.QComboBox()
        self.pilih_sumber.addItems([KAMERA, VIDEO])
        self.pilih_metode = QtWidgets.QComboBox()
        self.pilih_metode.addItems(list(METODE))
        self.pilih_metode.setCurrentText(args.metode)
        self.tombol_jalan = QtWidgets.QPushButton("Mulai")
        self.tombol_rekam = QtWidgets.QPushButton("Rekam sesi")
        self.tombol_rekam.setCheckable(True)
        self.label_sumber = QtWidgets.QLabel()

        atas = QtWidgets.QHBoxLayout()
        for teks, w in [("Sumber", self.pilih_sumber), ("Metode", self.pilih_metode)]:
            atas.addWidget(QtWidgets.QLabel(teks))
            atas.addWidget(w)
        atas.addWidget(self.tombol_jalan)
        atas.addWidget(self.tombol_rekam)
        atas.addWidget(self.label_sumber, 1)

        self.pratinjau = QtWidgets.QLabel("Belum ada frame")
        self.pratinjau.setAlignment(QtCore.Qt.AlignCenter)
        self.pratinjau.setMinimumSize(320, 240)
        self.pratinjau.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)

        self.label_bpm = QtWidgets.QLabel("--")
        f = self.label_bpm.font()
        f.setPointSize(48)
        f.setBold(True)
        self.label_bpm.setFont(f)
        self.label_conf = QtWidgets.QLabel("Confidence: --")
        self.label_status = QtWidgets.QLabel("Siap")
        self.label_status.setWordWrap(True)
        self.label_latensi = QtWidgets.QLabel("Proses per frame: --")
        self.label_latensi.setWordWrap(True)
        self.label_rekam = QtWidgets.QLabel("Tidak merekam")
        self.label_rekam.setWordWrap(True)
        self.plot = pg.PlotWidget(title="Sinyal pulsa (jendela terakhir)")
        self.plot.setLabel("bottom", "waktu", "s")
        self.plot.hideAxis("left")
        self.kurva = self.plot.plot(pen=pg.mkPen(width=2))

        kanan = QtWidgets.QVBoxLayout()
        kanan.addWidget(QtWidgets.QLabel("BPM"))
        kanan.addWidget(self.label_bpm)
        for w in [self.label_conf, self.label_status, self.label_latensi, self.label_rekam]:
            kanan.addWidget(w)
        kanan.addStretch(1)

        tengah = QtWidgets.QHBoxLayout()
        tengah.addWidget(self.pratinjau, 3)
        tengah.addLayout(kanan, 2)

        self.label_medis = QtWidgets.QLabel(TEKS_BUKAN_MEDIS)
        self.label_medis.setStyleSheet("color: #b00020; font-weight: bold;")
        self.label_medis.setWordWrap(True)

        tata = QtWidgets.QVBoxLayout(self)
        tata.addLayout(atas)
        tata.addLayout(tengah, 3)
        tata.addWidget(self.plot, 2)
        tata.addWidget(self.label_medis)

        self.pilih_sumber.setCurrentText(VIDEO if self.video else KAMERA)
        self.pilih_sumber.activated.connect(self._ganti_sumber)
        self.pilih_metode.currentTextChanged.connect(self._ganti_metode)
        self.tombol_jalan.clicked.connect(self._jalan_henti)
        self.tombol_rekam.toggled.connect(self._rekam)
        self.resize(1000, 750)
        self._mulai()

    # --- sumber dan pekerja
    def _mulai(self):
        self._henti()
        self.sesi = Sesi(self.cfg, self.pencari, self.pilih_metode.currentText())
        self._pulsa_terakhir = None
        self.kurva.setData([], [])
        self.label_sumber.setText(self.video or f"kamera {self.cfg['kamera']['indeks']} "
                                                f"({self.cfg['kamera']['backend']})")
        self.pekerja = Pekerja(self.sesi, self.cfg, self.video, self.fps)
        self.pekerja.terbuka.connect(self._terbuka)
        self.pekerja.tampil.connect(self._tampil)
        self.pekerja.berakhir.connect(self._berakhir)
        self.pekerja.start()
        self.tombol_jalan.setText("Henti")

    def _henti(self):
        if self.pekerja is not None:
            self.pekerja.hentikan()
            self.pekerja.wait(3000)
            self.pekerja = None
        self.tombol_rekam.setChecked(False)
        self.tombol_jalan.setText("Mulai")

    def _jalan_henti(self):
        if self.pekerja is not None and self.pekerja.isRunning():
            self._henti()
        else:
            self._mulai()

    def _ganti_sumber(self):
        if self.pilih_sumber.currentText() == VIDEO:
            berkas, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Pilih video", str(RAIZ),
                                                              "Video (*.mp4 *.avi *.mov *.mkv);;Semua (*)")
            if not berkas:
                self.pilih_sumber.setCurrentText(VIDEO if self.video else KAMERA)
                return
            self.video = berkas
        else:
            self.video = None
        self._mulai()

    def _ganti_metode(self, metode: str):
        if self.sesi is not None:
            self.sesi.minta_metode(metode)
            self._pulsa_terakhir = None
            self.kurva.setData([], [])

    # --- perekaman
    def _rekam(self, nyala: bool):
        if self.sesi is None:
            return
        if nyala:
            folder = self.sesi.mulai_rekam(RAIZ / self.cfg["aplikasi"]["folder_sesi"],
                                           {"sumber": self.info_sumber, "metode_awal": self.sesi.metode})
            self.label_rekam.setText(f"Merekam ke {folder}")
            self.tombol_rekam.setText("Berhenti rekam")
        else:
            folder = self.sesi.berhenti_rekam()
            self.label_rekam.setText(f"Tersimpan di {folder}" if folder else "Tidak merekam")
            self.tombol_rekam.setText("Rekam sesi")

    # --- slot dari pekerja
    def _terbuka(self, info: dict):
        self.info_sumber = info
        if self.rekam_awal:
            self.rekam_awal = False
            self.tombol_rekam.setChecked(True)

    def _berakhir(self, pesan: str):
        self.label_status.setText(pesan)
        self.tombol_rekam.setChecked(False)
        self.tombol_jalan.setText("Mulai")

    def _tampil(self, tp):
        teks = teks_tampilan(tp)
        self.label_bpm.setText(teks["bpm"])
        self.label_conf.setText(f"Confidence: {teks['confidence']}  |  metode: {tp.metode}")
        self.label_status.setText(teks["status"])
        self.label_latensi.setText(teks["latensi"])
        if tp.pulsa is not self._pulsa_terakhir:
            self._pulsa_terakhir = tp.pulsa
            if tp.pulsa is None:
                self.kurva.setData([], [])
            else:
                self.kurva.setData(np.arange(len(tp.pulsa)) / tp.fs_pulsa, tp.pulsa)
        if tp.frame is not None and tp.frame.size:
            self._pratinjau(np.ascontiguousarray(tp.frame), tp.roi)

    def _pratinjau(self, frame, roi):
        h, w = frame.shape[:2]
        img = QtGui.QImage(frame.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
        if roi is not None:
            x0, y0, x1, y1 = roi
            p = QtGui.QPainter(img)
            p.setPen(QtGui.QPen(QtGui.QColor(0, 220, 0), 2))
            p.drawRect(x0, y0, x1 - x0, y1 - y0)
            p.end()
        self.pratinjau.setPixmap(QtGui.QPixmap.fromImage(img).scaled(
            self.pratinjau.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def closeEvent(self, e):
        self._henti()
        super().closeEvent(e)


def main(argv=None) -> int:
    cfg = muat()
    args = buat_parser(cfg).parse_args(argv)
    app = QtWidgets.QApplication(sys.argv[:1])
    jendela = Jendela(cfg, PencariMediaPipe(cfg), args)
    jendela.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
