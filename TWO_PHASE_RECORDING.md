# Two-Phase Recording System

## Overview
Sistem ini memisahkan proses recording dari marker detection untuk menjaga FPS tetap tinggi saat merekam.

**Masalah:** Saat menjalankan `freeaHandDrawing.py`, proses marker detection real-time menyebabkan FPS turun drastis.

**Solusi:** Gunakan dua fase terpisah:
1. **Fase 1 (Pure Recording)**: Hanya merekam video kamera tanpa marker detection → FPS tinggi ✓
2. **Fase 2 (Post-Processing)**: Analisis video yang sudah direkam untuk marker detection dan generate CSV

---

## Cara Penggunaan

### Fase 1: Pure Recording

```bash
python recorder_pure.py
```

**Fitur:**
- Merekam video dari kamera tanpa marker detection
- Capture screenshot layar dalam interval (untuk reference)
- Monitor FPS untuk memastikan recording berkualitas
- Button detection untuk start/stop recording

**Output:**
```
dataMarker/session_YYYYMMDD_HHMMSS/
├── raw_camera.mp4       # Video kamera murni (high FPS)
└── screen_captures/     # Tangkapan layar (opsional, dapat dihapus)
    ├── screen_000000.png
    ├── screen_000001.png
    └── ...
```

**Keterangan:**
- Press ESC untuk keluar dari program
- Green button di UI menandakan recording aktif
- FPS seharusnya **>25 FPS** (jika <20 FPS, kurangi resolusi/kompres)

---

### Fase 2: Post-Processing

Setelah selesai recording, jalankan processor untuk menganalisis video:

```bash
# Proses session terbaru
python processor_video.py -l

# Atau proses session spesifik
python processor_video.py -s dataMarker/session_20250115_184137
```

**Fitur:**
- Membaca `raw_camera.mp4`
- Melakukan marker detection untuk setiap frame
- Generate video tertracking dengan trajectory visualization
- Generate CSV dengan semua data pose

**Output:**
```
dataMarker/session_YYYYMMDD_HHMMSS/
├── raw_camera.mp4          # Video asli (dari fase 1)
├── tracked.mp4             # Video dengan marker dan trajectory
└── data.csv                # Data pose dan pen tip tracking
```

**CSV Columns:**
- `timestamp`: ISO format timestamp
- `frame`: Frame number
- `marker_id`: ID marker yang terdeteksi
- `tip_x, tip_y, tip_z`: Relative pen tip position
- `marker_x, marker_y, marker_z`: Relative marker position
- `angle`: Rotation angle
- `axis_x, axis_y, axis_z`: Rotation axis

---

## Workflow Lengkap

1. **Siapkan session:**
   ```bash
   python recorder_pure.py
   ```
   - Program akan menunggu button detection
   - Tekan tombol ON untuk mulai recording
   - Lakukan drawing/tracking sesuai kebutuhan
   - Tekan tombol OFF untuk stop recording

2. **Analisis video yang direkam:**
   ```bash
   python processor_video.py -l
   ```
   - Program akan membaca `raw_camera.mp4` dari session terbaru
   - Melakukan marker detection frame-by-frame
   - Menghasilkan `tracked.mp4` dan `data.csv`

3. **Cek hasil:**
   ```
   dataMarker/session_YYYYMMDD_HHMMSS/
   ├── raw_camera.mp4  ← Video rekaman asli
   ├── tracked.mp4     ← Video dengan analisis marker
   └── data.csv        ← Data untuk analysis lebih lanjut
   ```

---

## Perbandingan: Lama vs Baru

### Sebelumnya (freeaHandDrawing.py)
```
Proses dalam 1 program:
Recording + Marker Detection + Screen Capture + CSV Generation
                    ↓
            FPS DROP DRASTIS (5-15 FPS)
```

### Sekarang (2-Phase System)
```
Phase 1 - Pure Recording:     Phase 2 - Post-Processing:
  • Video capture only      →   • Video analysis (no time pressure)
  • FPS: 25-30 FPS          →   • Generate tracked.mp4
  • Cepat selesai           →   • Generate data.csv
```

**Keuntungan:**
✓ FPS tetap tinggi saat recording (25-30 FPS vs 5-15 FPS)
✓ Data recording lebih akurat (tidak ada frame drop)
✓ Post-processing bisa dilakukan kapan saja (off-the-shelf processing)
✓ Mudah di-debug (pisah concern)

---

## Advanced Options

### Mengubah Screen Capture Interval
Edit di `recorder_pure.py`:
```python
SCREEN_CAPTURE_INTERVAL = 2  # Capture setiap 2 frame (lebih jarang = storage lebih kecil)
```

### Mengubah Crop Region untuk Screen
Edit di kedua file:
```python
CROP_INDEX = 1  # 0-6 pilihan crop region yang berbeda
```

### Mengubah Marker Detection Parameter
Edit di `processor_video.py` - ubah camera matrix dan dist coeffs sesuai kalibrasi terbaru.

---

## Troubleshooting

**Q: FPS masih drop di Phase 1?**
- Kurangi resolusi kamera (e.g., 640x480 instead 1280x720)
- Disable screen capture dengan set `SCREEN_CAPTURE_INTERVAL = 999999`

**Q: Marker tidak terdeteksi di Phase 2?**
- Pastikan lighting cukup saat recording
- Cek marker IDs di `processor_video.py` (currently: [1, 5, 7])
- Verify camera calibration parameters

**Q: Video file terlalu besar?**
- Gunakan compression settings berbeda di fourcc
- Reduce frame rate jika acceptable

---

## File Structure

```
aruco_stylus/
├── recorder_pure.py          # Phase 1: Pure recording
├── processor_video.py        # Phase 2: Post-processing
├── freeaHandDrawing.py       # Original (legacy)
├── TWO_PHASE_RECORDING.md    # This file
└── dataMarker/
    ├── session_20250115_184137/
    │   ├── raw_camera.mp4
    │   ├── tracked.mp4
    │   ├── data.csv
    │   └── screen_captures/
    └── ...
```

---

## Migrasi dari freeaHandDrawing.py

Jika ingin kembali ke sistem lama:
```bash
python freeaHandDrawing.py
```

Namun disarankan menggunakan 2-phase system karena:
- FPS lebih stabil
- Recording quality lebih baik
- Processing dapat dioptimasi terpisah

---

## Tips Performance Tuning

1. **Untuk recording ultra-clean:**
   - Disable screen capture: `SCREEN_CAPTURE_INTERVAL = 999999`
   - Gunakan 1080p resolution
   - Dedicated SSD untuk video storage

2. **Untuk processing lebih cepat:**
   - GPU acceleration dapat ditambahkan di `processor_video.py`
   - Multi-threading untuk frame processing

3. **Storage optimization:**
   - Hapus `screen_captures/` folder setelah selesai (tidak essential)
   - Compress `raw_camera.mp4` dengan ffmpeg jika perlu archive

---

**Last Updated:** 2025-01-15
**Tested on:** Windows 10/11, Python 3.8+, OpenCV 4.5+
