## VTuber Live Avatar – MediaPipe + OpenCV + Tkinter

Project ini adalah aplikasi **VTuber 2D Live Avatar** yang menggunakan **MediaPipe** untuk melacak pose, tangan, dan wajah secara real-time, lalu menggambar karakter anime menggunakan **OpenCV Shapes**. Hampir semua tubuh digambar memakai kode, dan hanya dua file aset yang digunakan:

- **baju2.png** → Aset baju
- **rok.png** → Aset rok

Aplikasi menampilkan dua panel:
- **Kiri** : Webcam + Tracking
- **Kanan** : Avatar anime yang mengikuti gerakan tubuh

---

## Konsep Utama (Tracking + Avatar Drawing)

Aplikasi menggabungkan 3 modul utama MediaPipe:

### 1️⃣ Pose Tracking  
Mendeteksi posisi bahu, siku, pinggul, lutut, dan pergelangan tangan. Digunakan untuk menggambar tubuh karakter & menempatkan pakaian.

### 2️⃣ Hand Tracking  
Mengambil koordinat jari untuk menggambar bentuk tangan dan gesture.

### 3️⃣ Face Mesh  
Menghitung nilai:
- **EAR (Eye Aspect Ratio)** → untuk kedipan
- **MAR (Mouth Aspect Ratio)** → untuk mulut terbuka
- Posisi pipi, mata, alis, hidung untuk ekspresi anime

Avatar digambar seluruhnya menggunakan OpenCV shape (ellipse, polygon, line) kecuali baju & rok.

---

## Aset yang Digunakan

Aplikasi hanya membutuhkan dua aset PNG dengan transparansi:

| Aset | Fungsi |
|------|--------|
| **baju2.png** | Overlay baju pada torso |
| **rok.png** | Overlay rok pada pinggul |

Semua rambut, wajah, badan, tangan, kaki, dan ekspresi digambar memakai shape.

---

## Background Avatar

Avatar mendukung:
- Background gradient (default)
- Background gambar dari folder `assets/backgrounds/`

Tekan **B** untuk mengganti background.

---

