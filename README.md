# rPPG Real-Time: Evaluasi Akurasi dan Latensi CHROM

Evaluasi akurasi dan latensi algoritma CHROM real-time untuk estimasi
denyut jantung dari video wajah (Python, OpenCV, MediaPipe, PySide6).

## Prasyarat
- Windows 10/11, Python 3.12 (python.org, dengan launcher `py`), Git
- Webcam (opsional, untuk demo)

## Penyiapan
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

## Verifikasi lingkungan
python tes_lingkungan.py
Hasil yang diharapkan: Python 3.12.x, tanpa error impor,
kamera True dengan bentuk frame (480, 640, 3).
python tes_wajah.py   (tekan q untuk keluar)

## Struktur
src/      kode pustaka inti
data/     dataset (tidak di-commit)
models/   model deteksi wajah (diunduh otomatis)
results/  keluaran CSV dan grafik

## Catatan
- MediaPipe 1.1.0 tidak memiliki mp.solutions, kode memakai Tasks API.
- Webcam laptop yang diuji berjalan 15 fps asli, gunakan backend DSHOW.