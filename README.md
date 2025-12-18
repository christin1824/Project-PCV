# Laporan Project Akhir PCV Vtuber

## Pendahuluan

Project ini adalah aplikasi **VTuber 2D Live Avatar** yang menggunakan **MediaPipe** untuk melacak pose, tangan, dan wajah secara real-time, lalu menggambar karakter animasi menggunakan **OpenCV Shapes**. Proyek ini merupakan aplikasi pelacakan gerakan real-time yang menghasilkan **Animated Cartoon Avatar** berdasarkan input dari webcam, mentranslasikan gerakan pengguna ke karakter 2D yang digambar di layar. Aplikasi ini dibangun menggunakan **OpenCV** untuk pemrosesan gambar, **MediaPipe** untuk tracking, dan **Tkinter** untuk antarmuka pengguna grafis (GUI).

Aplikasi menampilkan dua windows:
* **Kiri**: Webcam + Tracking (Menampilkan landmark MediaPipe)
* **Kanan**: Avatar animasi yang mengikuti gerakan tubuh 

---
## Video Demo

Berikut merupakan video demo vtuber : https://drive.google.com/file/d/17j6cfuJ04dZ1s7ZGbjQ7qD6UbnBeW1jG/view?usp=sharing

---

## Konsep Utama (Tracking & Rendering)

Hampir semua tubuh, wajah, dan rambut avatar digambar memakai kode primitif OpenCV (lingkaran, garis, poligon). Hanya dua file aset yang digunakan sebagai *overlay* pakaian. Aplikasi menggabungkan 3 modul utama MediaPipe:

### 1. Pose Tracking
Mendeteksi 33 landmark tubuh (bahu, siku, pinggul, lutut, dll.). Digunakan untuk:
* Menggambar tubuh karakter dan anggota badan (lengan, kaki).
* Menentukan posisi torso untuk menempatkan *overlay* pakaian.

### 2️. Hand Tracking
Mengambil 21 koordinat jari per tangan. Digunakan untuk:
* Menggambar bentuk tangan dan jari-jari pada avatar.

### 3️. Face Mesh (Deteksi Ekspresi)
Mendeteksi ratusan landmark pada wajah, yang kemudian digunakan untuk menghitung nilai dan posisi ekspresi:
* **EAR (Eye Aspect Ratio)**: Digunakan untuk mendeteksi kedipan mata. Jika $\text{EAR}$ turun di bawah *threshold* (0.2), mata avatar akan tertutup.
* **MAR (Mouth Aspect Ratio)**: Digunakan untuk mendeteksi seberapa lebar mulut terbuka. Jika $\text{MAR}$ melewati *threshold* (0.5), mulut avatar digambar terbuka.
* **Posisi Wajah:** Digunakan untuk menempatkan mata, alis, hidung, dan pipi yang merona (blush) secara simetris, menciptakan ekspresi anime sederhana.

---

## Fitur Utama dan Implementasi

### 1. Ekspresi Wajah Dinamis

Aplikasi ini mampu menerjemahkan gerakan wajah dasar menjadi ekspresi avatar:
* **Kedipan Mata (Blink):** Menggunakan **Eye Aspect Ratio (EAR)** yang dihitung dari landmark Face Mesh untuk menutup mata avatar secara otomatis saat pengguna berkedip. Rumus yang digunakan adalah:

  $$\text{EAR} = \frac{|p_2 - p_6| + |p_3 - p_5|}{2 \cdot |p_1 - p_4|}$$

  (Jika $\text{EAR} < 0.2$, mata avatar tertutup.)

* **Buka Mulut:** Menggunakan **Mouth Aspect Ratio (MAR)**. Jika $\text{MAR} > 0.5$, mulut avatar akan digambar terbuka (oval sederhana).

* **Wajah Simetris:** Wajah avatar (mata, hidung, mulut) digambar dalam posisi simetris relatif terhadap pusat kepala yang diestimasi.

---

### 2. Integrasi Aset Pakaian (Overlay)

Avatar digambar seluruhnya menggunakan shape, kecuali untuk dua *overlay* pakaian yang diimpor. Aset PNG dimuat dengan saluran alfa (`cv2.IMREAD_UNCHANGED`) dan diterapkan ke avatar:

| Aset File | Fungsi | Detail Implementasi |
| :--- | :--- | :--- |
| **`baju2.png`** | Overlay Baju Atasan | Disesuaikan skalanya (`TOP_SCALE_X = 1.60`) dan ditempatkan di atas torso vektor. |
| **`rok.png`** | Overlay Rok | Disesuaikan skalanya dan ditempatkan di area pinggul/kaki bagian atas. |

---

### 3. Latar Belakang Interaktif

Pengguna dapat mengganti latar belakang tampilan avatar saat *runtime* dengan menekan tombol **'B'**:
* Mendukung latar belakang **Gradien** (default) dan **Gambar Statis** yang dimuat dari folder (misalnya, folder `backgrounds/`).

---

### 4. Rendering Kartun Sederhana

Anggota badan (lengan dan kaki) digambar sebagai elips dan garis tebal menggunakan warna kulit (`SKIN_COLOR_MAIN`, `SKIN_COLOR_SHADOW`) sebelum *overlay* pakaian untuk menciptakan efek lapisan.

---

## Teknologi dan Dependensi

| Kategori | Teknologi | Tujuan Proyek Ini |
| :--- | :--- | :--- |
| **Tracking** | **MediaPipe** | Deteksi Landmark Wajah, Pose, dan Tangan. |
| **Vision & Processing** | **OpenCV (`cv2`)** | Pengambilan frame, Rendering Grafis 2D, dan blending aset. |
| **GUI** | **Tkinter** | Menampilkan Output Webcam dan Avatar dalam dua panel. |
| **Matematika & Grafis** | **NumPy & PIL/Pillow** | Perhitungan vektor, manipulasi gambar, dan konversi untuk Tkinter. |

---

## Cara Instalasi dan Eksekusi

### 1. Prasyarat
* Python 
* Webcam / Camera Laptop

### 2. Struktur Direktori
Pastikan file kode dan aset Anda berada dalam struktur berikut:
```
Project Folder
├──  main.py
├──  baju2.png
├──  rok.png
└──  backgrounds/
    └── 1yg.png
    └── hybe.png
    └── jyp2.png
    └── panggung1.png
```

**Catatan:**
- Semua file PNG aset (seperti `baju2.png`, `rok.png`, dll.) harus berada satu level dengan `main.py`.
- Folder `backgrounds/` bersifat opsional. Jika ada, aplikasi akan otomatis membaca semua gambar di dalamnya untuk fitur ganti background (tekan tombol **B**).

### 3. Clone atau Download Project

Masuk ke folder tempat Anda ingin menyimpan project:

```bash
cd d:\ProjectPCV
```
### 4. Instalasi Dependensi

Instal *library* Python yang diperlukan:

```bash
pip install opencv-python mediapipe numpy pillow. 
```

### 5. Menjalankan Program

```bash
python main.py
```

### 6. Instruksi Penggunaan
- Arahkan wajah dan tubuh ke depan kamera.
- Tekan B pada keyboard untuk mengganti background.
- Jika asset hilang, pastikan file PNG berada dalam folder yang sama dengan main.py.
---
## Detail Konfigurasi

| Konstanta | Default | Fungsi |
| :--- | :--- | :--- |
| **SKIN_COLOR_MAIN** | (165, 218, 255) | Warna kulit dasar avatar. |
| **HAIR_COLOR_MAIN** | (50, 100, 150) | Warna rambut avatar. |
| **SKIRT_SCALE** | 1.25 | Skala overlay rok. |
| **TOP_SCALE_X** | 1.60 | Tingkat pelebaran baju overlay. |
| **EAR_THRESHOLD** | 0.2 | Batas deteksi kedipan mata. |
| **MAR_THRESHOLD** | 0.5 | Batas deteksi buka mulut. |
---
## Limitasi Aplikasi

- **Ekspresi Mulut Terbatas :**  Mulut hanya mendukung dua kondisi (terbuka / tertutup). Belum ada variasi ekspresi atau lip-sync.

- **Belum Mendukung Rotasi Kepala 3D:** Wajah hanya ditampilkan front-facing karena tracking masih berbasis 2D.

- **Animasi Masih Sederhana:**  Avatar digambar menggunakan primitive shapes, sehingga gerakan terlihat kaku dan minim detail.

- **Overlay PNG Bergantung Skala Manual:** Proporsi baju/rok harus disesuaikan manual dan bisa sedikit bergeser.

## Rencana Pengembangan

- Menambahkan smoothing (EMA/One Euro Filter)  
- Implementasi rotasi kepala 3D menggunakan `solvePnP`  
- Ekspresi wajah dan mulut yang lebih lengkap  
- Animasi idle dan motion yang lebih halus  
- Sistem layering pakaian dan aksesori

## Kesimpulan 
Aplikasi VTuber 2D ini berhasil menggabungkan MediaPipe (Pose, Face Mesh, dan Hand Tracking) dengan OpenCV untuk membangun avatar kartun real-time yang responsif dan menarik. Meskipun sebagian besar avatar digambar menggunakan primitive shapes, aplikasi ini bisa menjadi fondasi awal untuk proyek VTuber lebih kompleks seperti model avatar 3D.
