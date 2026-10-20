# Tes yang lolos

Dicatat dari keluaran nyata `python -m pytest -v`, bukan ringkasan tangan.

- Tanggal: 2026-10-20
- Commit: `a1e7d4e` (feat(ketahanan): tangani wajah hilang, kamera gagal, frame kosong, timestamp mundur)
- Lingkungan: Windows 11, Python 3.12.0, pytest 9.1.1, venv `.venv`
- Hasil: **53 passed in 4.38s**, 0 gagal, 0 dilewati

Semua tes memakai data sintetis atau tiruan (kamera tiruan, jam tiruan, pencari wajah tiruan, folder dataset buatan). Tidak ada tes yang membutuhkan webcam, dataset UBFC/PURE, atau model MediaPipe, jadi hasilnya sama di mesin mana pun.

## Daftar tes

### tests/test_cek_kamera.py (5) — uji kamera tercatat (WP1)

| Tes | Yang diuji |
|---|---|
| test_sumber_buang_duplikat_dan_timestamp | SumberKamera membuang frame duplikat, timestamp naik, kamera dilepas saat ditutup |
| test_sumber_gagal_buka | Kamera yang gagal dibuka memunculkan galat |
| test_ringkas_angka_diketahui | fps nyata, rasio duplikat, dt dan jitter (rata-rata, p95) dari angka yang diketahui |
| test_ringkas_terlalu_sedikit_frame | Satu frame saja tidak menghasilkan angka palsu (None) |
| test_jalankan_dan_tulis | `frame.csv` dan `ringkasan.json` ditulis dengan kolom dan fps yang benar |

### tests/test_data.py (6) — pemuat dataset (WP2)

| Tes | Yang diuji |
|---|---|
| test_ubfc_frame_rgb_dan_referensi | Pemuat UBFC: frame RGB (bukan BGR), timestamp 1/fps, HR referensi |
| test_ubfc_format_salah | ground_truth.txt yang tidak sesuai asumsi → FormatTidakSesuai |
| test_pure_timestamp_ns_ke_detik | Pemuat PURE: timestamp nanodetik → detik, HR pulseRate |
| test_pure_kunci_hilang | JSON tanpa kunci yang diharapkan → FormatTidakSesuai |
| test_dataset_tidak_ada[ubfc] | Jalur kosong, folder tidak ada, atau folder kosong → DatasetTidakAda |
| test_dataset_tidak_ada[pure] | Sama, untuk PURE |

### tests/test_evaluasi.py (13) — evaluasi offline (WP2)

| Tes | Yang diuji |
|---|---|
| test_kurangi_fps_berbasis_timestamp | 30 → 15 fps memotong rapi setiap frame kedua |
| test_kurangi_fps_sumber_sudah_15fps_tertunda_tidak_dibuang | Sumber 15 fps dengan penundaan baca tidak kehilangan frame (regresi bug 366/450) |
| test_kurangi_fps_sumber_macet_tanpa_ledakan_frame | Setelah sumber macet 1 s tidak ada ledakan frame |
| test_kurangi_fps_laju_rata_rata_tidak_melebihi_target | 22 → 15 fps menghasilkan rata-rata 15 fps |
| test_kurangi_fps_lebih_tinggi_dari_asli_tidak_menambah | Target lebih tinggi dari sumber tidak menambah frame |
| test_roi_dahi | Ukuran dan posisi ROI dahi, serta ROI di luar frame → None |
| test_ekstrak_rgb_deteksi_tiap_n_dan_wajah_hilang | Deteksi wajah tiap n frame; tanpa wajah → None |
| test_hr_ref_jendela | Rata-rata HR referensi per jendela, nilai tidak valid diabaikan |
| test_evaluasi_rekaman_menemukan_bpm | Ujung ke ujung pada video sintetis 72 BPM untuk chrom, pos, hijau |
| test_holm | Koreksi Holm terhadap nilai yang dihitung tangan |
| test_ringkas_dan_wilcoxon | Ringkasan MAE/RMSE/Bland-Altman dan uji Wilcoxon per video |
| test_tulis_keluaran | estimasi.csv, ringkasan.csv, wilcoxon.csv, PNG Bland-Altman, meta.json |
| test_tabel1_belum_diukur_dan_terisi | Tabel 1: sel "belum diukur" bila data tidak ada, terisi bila ada |

### tests/test_ketahanan.py (11) — ketahanan (WP4)

| Tes | Yang diuji |
|---|---|
| test_jeda_panjang_mengosongkan_buffer | Wajah hilang > jeda_maks_detik → buffer dan BPM dikosongkan |
| test_jeda_pendek_tidak_mereset | Jeda di bawah batas tidak mereset |
| test_timestamp_mundur_dibuang[4.0] | Timestamp mundur dibuang, buffer tetap |
| test_timestamp_mundur_dibuang[5.0] | Timestamp sama dibuang, buffer tetap |
| test_sampel_tidak_valid_dibuang[buruk0] | Nilai RGB NaN dibuang |
| test_sampel_tidak_valid_dibuang[buruk1] | Nilai RGB inf dibuang |
| test_frame_kosong_tidak_crash[None] | Frame None tidak membuat crash |
| test_frame_kosong_tidak_crash[frame1] | Frame berukuran 0 tidak membuat crash |
| test_kamera_gagal_sesaat_dilanjutkan | Kamera gagal sesaat → sesi lanjut |
| test_kamera_gagal_terus_menerus_melempar | Kamera gagal berturut-turut → KameraGagal |
| test_aliran_status_berurutan | Status dan pesan antarmuka: mengumpulkan, frame kosong, ok, wajah hilang, direset, timestamp mundur |

### tests/test_latensi.py (4) — latensi (WP3)

| Tes | Yang diuji |
|---|---|
| test_pengukur_konteks_dan_ringkasan | Pengukur mencatat durasi; median dan p95 benar |
| test_ukur_pipeline_tahap_dan_bpm_pertama | Semua tahap tercatat dengan jumlah benar; BPM pertama sekitar 8 s |
| test_ukur_pipeline_batas_detik_dan_tanpa_wajah | Batas durasi dipatuhi; tanpa wajah tidak ada estimasi |
| test_tulis_hasil | sampel.csv, ringkasan.csv, meta.json |

### tests/test_metrik.py (3) — metrik

| Tes | Yang diuji |
|---|---|
| test_metrik_dasar | MAE, RMSE, proporsi galat ≤ 5 BPM |
| test_nan_diabaikan | Pasangan NaN diabaikan |
| test_bland_altman | Bias Bland-Altman |

### tests/test_sinyal.py (11) — sinyal dan pipeline

| Tes | Yang diuji |
|---|---|
| test_streaming_bpm_akurat[55] | Pipeline streaming menemukan 55 BPM (±2) |
| test_streaming_bpm_akurat[72] | Sama, 72 BPM |
| test_streaming_bpm_akurat[100] | Sama, 100 BPM |
| test_streaming_bpm_akurat[140] | Sama, 140 BPM |
| test_belum_cukup_data_bpm_none | Data kurang dari jendela → BPM None |
| test_derau_murni_confidence_rendah | Derau murni → confidence rendah |
| test_chrom_dan_estimasi_langsung | CHROM + estimasi spektrum pada sinyal bersih |
| test_semua_metode_menemukan_72bpm[chrom] | CHROM menemukan 72 BPM |
| test_semua_metode_menemukan_72bpm[pos] | POS menemukan 72 BPM |
| test_semua_metode_menemukan_72bpm[hijau] | Hijau menemukan 72 BPM |
| test_metode_tidak_dikenal | Nama metode salah → ValueError |

## Cara menjalankan ulang

Dari akar repo, di PowerShell:

```powershell
.venv\Scripts\activate
pip install -e .            # sekali saja, atau setelah pyproject.toml berubah
python -m pytest -q         # semua tes, ringkas
python -m pytest -v         # semua tes, satu baris per tes (bentuk daftar di atas)
```

Menjalankan sebagian:

```powershell
python -m pytest tests/test_ketahanan.py -v                 # satu berkas
python -m pytest -k kurangi_fps -v                          # tes yang namanya memuat kata ini
python -m pytest "tests/test_sinyal.py::test_streaming_bpm_akurat[72]"   # satu tes
```

Bila ada yang gagal, `python -m pytest -x -q` berhenti di kegagalan pertama dan menampilkan detailnya.

## Yang tidak termasuk tes

Skrip berikut butuh perangkat atau data nyata, jadi dijalankan terpisah dan hasilnya masuk `results/` (tidak di-commit):

- `python scripts/cek_kamera.py --detik 30`: webcam (WP1)
- `python scripts/latensi.py --detik 30`: webcam, bawaan 15 fps (WP3)
- `python scripts/evaluasi.py`: dataset UBFC/PURE di `config/lokal.yaml` (WP2); tanpa dataset berhenti dengan pesan
- `python scripts/tabel1.py [ringkasan.csv]`: Tabel 1 dari hasil evaluasi

Daftar ini perlu diperbarui bila tes ditambah atau diubah.
