# Proter — Monitor Lalu Lintas Stereo

Sistem pemantauan lalu lintas menggunakan **dua kamera** (*stereo vision*):
mendeteksi kendaraan, melacaknya dengan ID yang tetap, mengukur **posisi 3D
(meter)** dan **kecepatan (m/s)**, lalu menyimpan semuanya ke file data yang
mudah dibaca program lain.

Panduan lengkap untuk pemula tersedia di **`notebooks/proter.ipynb`** (buka dengan
Jupyter/VS Code). Dokumen ini ringkasan teknisnya.

## Cara kerja

```
Kamera kiri + kanan (atau dua file video)
        |
        v
[1] Rectification    - luruskan distorsi lensa, sejajarkan kedua gambar
[2] Deteksi          - YOLO menemukan kendaraan di gambar kiri
[3] Pelacakan        - ByteTrack memberi ID yang tetap antar-frame
[4] Stereo matching  - cari kendaraan yang sama di gambar kanan (disparity)
[5] Triangulasi      - ubah disparity menjadi jarak dalam meter
[6] Estimasi kecepatan - regresi linear pada riwayat posisi
        |
        v
CSV + 2 video + JSON metadata  (satu folder per sesi rekaman)
```

## Instalasi

Butuh Python 3.9+ (tkinter ikut hadir pada instalasi Python standar).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

Bobot YOLO (`yolov9c.pt`, ±25 MB) diunduh otomatis saat pertama kali dipakai.

## Penggunaan

Semua perintah lewat `stereo_traffic_monitor.py`:

```bash
# Antarmuka grafis (tab Live + tab Analysis + tab Calibrate)
python stereo_traffic_monitor.py gui

# Kalibrasi dari foto papan catur (perkiraan 10-15 pasang foto)
python stereo_traffic_monitor.py calibrate --left-dir foto/kiri --right-dir foto/kanan \
    --board 9x6 --square-m 0.025 --out calibration.json

# Pipeline tanpa tampilan (dua file video ATAU dua indeks kamera)
python stereo_traffic_monitor.py run --left vid_l.mp4 --right vid_r.mp4 \
    --calibration calibration.json --out traffic_monitoring --preview

# Render ulang overlay (ID, kecepatan, lintasan) dari sebuah rekaman ke video
python stereo_traffic_monitor.py overlay recording_001 --camera left --unit km/h

# Kalibrasi manual (approximasi) dari pengukuran tangan, tanpa foto papan catur
python tools/make_manual_calibration.py --focal-px 1333.333 --baseline-m 1.0 --out Calibration/calibration.json
```

GUI juga bisa dijalankan langsung: `python stereo_traffic_gui.py`.
Di tab **Analysis**: `Space` = play/pause, `kiri/kanan` = mundur/maju 1 frame.
Tab **Calibrate** menjalankan kalibrasi dari dalam GUI dan otomatis mengarahkan
tab Live ke file hasilnya.

## Struktur keluaran

Setiap sesi rekaman menghasilkan satu folder:

```
traffic_monitoring/
└── recording_001/
    ├── left_camera.mp4          video kiri hasil rectification
    ├── right_camera.mp4         video kanan hasil rectification
    ├── tracking_data.csv        satu baris per kendaraan per frame
    ├── calibration.json         matriks kamera, distorsi, baseline, sistem koordinat
    └── recording_metadata.json  waktu mulai, fps, resolusi, info sinkronisasi
```

### Kolom utama `tracking_data.csv`

| Kolom | Arti |
|---|---|
| `timestamp_ns` | waktu tangkap (nanodetik sejak Unix epoch) |
| `frame_id` | nomor frame **mulai dari 1** — kunci sinkronisasi CSV ↔ video |
| `object_id` | ID pelacakan; hanya berlaku di dalam satu rekaman |
| `class`, `confidence` | jenis kendaraan (COCO) dan **confidence deteksi YOLO** |
| `world_x/y/z_m` | posisi 3D dalam meter (X kanan, Y bawah, Z kedalaman) |
| `velocity_x/y/z_ms`, `speed_ms` | komponen kecepatan (m/s) dan besar kecepatannya |
| `left_u/v`, `right_u/v` | pusat kendaraan dalam piksel gambar *rectified* |
| `bbox_x/y/w/h` | bounding box pada gambar kiri rectified |

Definisi sistem koordinat dan semua satuan juga tersimpan di dalam
`calibration.json` dan `recording_metadata.json`, sehingga program lain bisa
membaca data tanpa memahami kode computer vision-nya.

## Batasan yang perlu diketahui

- **Sinkronisasi kamera USB tidak hardware-sync** — dua frame bisa berselisih
  hingga ~1 frame; ini memengaruhi akurasi kedalaman objek yang bergerak cepat.
- **`confidence` = confidence deteksi YOLO**, bukan skor kualitas pengukuran 3D.
- Bounding box gambar kanan adalah **kotak kiri yang digeser disparity**, bukan
  hasil deteksi independen.
- Akurasi jarak/kecepatan bergantung mutlak pada **kualitas kalibrasi**;
  validasilah dengan pengukuran pembanding di lapangan.
- Objek tanpa pasangan stereo yang andal (oklusi, tekstur lemah) **dibuang**
  dari hasil 3D, bukan ditebak.

## Pengujian

Pengujian otomatis tanpa kamera (geometri triangulasi, kecepatan, round-trip
CSV, pemetaan frame CSV ↔ video, kalibrasi sintetis):

```bash
python tests/test_tracking_data.py
# atau: python -m pytest tests/ -v
```

## Membuat aplikasi `.exe` (beta, Windows)

Distribusi berupa **folder** (bukan satu `.exe` raksasa) karena dependensi
OpenCV + PyTorch sangat besar:

```bash
.venv\Scripts\pyinstaller Proter.spec --noconfirm --clean
Copy-Item yolov9c.pt dist\Proter\          # bobot model agar jalan offline
Copy-Item calibration.json dist\Proter\ -ErrorAction SilentlyContinue
```

Hasil: `dist\Proter\Proter.exe` (jalankan dari foldernya, jangan dipindah
sendirian — folder `_internal` wajib ikut). Jika ada masalah saat startup,
jejak kesalahannya ditulis ke `proter_error.log` di folder yang sama.

> **Status: prototipe beta.** Angka jarak/kecepatan dari exe ini belum
> divalidasi terhadap alat ukur pembanding di lapangan.

## Struktur proyek

```
proter/
├── notebooks/proter.ipynb           panduan berbahasa Indonesia (mulai dari sini)
├── stereo_traffic_monitor.py        inti pemrosesan + perintah CLI
├── stereo_traffic_gui.py            antarmuka grafis (Live + Analysis + Calibrate)
├── proter_app.py                    titik masuk untuk build .exe
├── Proter.spec                      konfigurasi PyInstaller
├── tools/make_manual_calibration.py buat calibration.json manual dari pengukuran
├── requirements.txt
├── Calibration/                     foto papan catur + calibration.json (lokal, tidak diunggah)
├── tests/test_tracking_data.py      pengujian otomatis lapisan data
└── dist/Proter/                     hasil build .exe beta (tidak diunggah ke repo)
```
