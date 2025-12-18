# Laporan Project Akhir PCV Vtuber Dengan Menggunakan Shape Untuk Animasi

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

---

# Laporan Project Akhir PCV Vtuber Dengan Menggunakan Vtube Studio Untuk Animasi

## Pendahuluan

Project ini merupakan aplikasi **VTuber Face Tracking Real-Time** yang mengintegrasikan **MediaPipe Face Mesh** dengan **VTube Studio** untuk menggerakkan avatar 2D secara langsung berdasarkan ekspresi dan pergerakan wajah pengguna. Sistem ini berfungsi sebagai **plugin eksternal face tracker**, di mana data hasil deteksi wajah dari webcam diproses menggunakan MediaPipe, kemudian dikirimkan ke **VTube Studio melalui VTube Studio Public API berbasis WebSocket**.

Berbeda dengan proyek VTuber sebelumnya yang menggambar avatar menggunakan **primitive shapes OpenCV**, pada project ini seluruh proses animasi dan rendering avatar ditangani sepenuhnya oleh **VTube Studio**, sementara aplikasi Python hanya bertugas melakukan:

- Tracking wajah  
- Perhitungan parameter ekspresi  
- Pengiriman data parameter ke model Live2D di VTube Studio  

Aplikasi menampilkan satu jendela utama berupa:

- GUI Webcam + Face Mesh Tracking (Tkinter)  
- Avatar Live2D ditampilkan terpisah di aplikasi VTube Studio  

---

## Video Demo

Berikut merupakan video demo : https://drive.google.com/file/d/17j6cfuJ04dZ1s7ZGbjQ7qD6UbnBeW1jG/view?usp=sharing  

---

## Konsep Utama (Tracking & Parameter Injection)

Konsep utama project ini adalah mengonversi **landmark wajah MediaPipe** menjadi **parameter numerik** yang sesuai dengan parameter standar **VTube Studio**, lalu mengirimkannya secara *real-time* melalui **WebSocket API**.

Sistem terdiri dari tiga komponen utama:

- Face Tracking (MediaPipe)  
- Parameter Processing & Smoothing  
- Parameter Injection ke VTube Studio  

---

## 1. Face Tracking (MediaPipe Face Mesh)

Aplikasi menggunakan **MediaPipe Face Mesh** untuk mendeteksi ratusan landmark wajah secara real-time. Landmark ini digunakan untuk menghitung:

- Rotasi Kepala (X, Y, Z)  
- Bukaan Mulut  
- Keterbukaan Mata Kiri & Kanan  

MediaPipe dipilih karena:

- Ringan dan real-time  
- Stabil untuk single-face tracking  
- Cocok untuk aplikasi VTuber  

---

## 2. Perhitungan Parameter Wajah

### a. Rotasi Kepala (Head Rotation)

Rotasi kepala dihitung menggunakan beberapa landmark utama:

- Ujung hidung (*nose tip*)  
- Jembatan hidung  
- Sudut mata kiri dan kanan  

Parameter yang dihasilkan:

- **Head X** → Gerakan menoleh kiri/kanan  
- **Head Y** → Gerakan menunduk/menengadah  
- **Head Z** → Kemiringan kepala (*tilt*)  

Nilai dinormalisasi ke rentang **-1 hingga 1**, kemudian diskalakan ke rentang **-30 sampai 30** agar sesuai dengan standar VTube Studio:
- FaceAngleX
- FaceAngleY
- FaceAngleZ


---

### b. Bukaan Mulut (Mouth Open)

Bukaan mulut dihitung dari jarak vertikal antara:

- Landmark bibir atas  
- Landmark bibir bawah  

Nilai ini dinormalisasi ke rentang **0 – 1** dan dipetakan ke parameter:
- MouthOpen


Semakin besar jarak bibir, semakin terbuka mulut avatar di VTube Studio.

---

### c. Keterbukaan Mata (Eye Open)

Untuk setiap mata, sistem menghitung jarak vertikal antara:

- Kelopak mata atas  
- Kelopak mata bawah  

Parameter yang dikirim:
- EyeOpenLeft
- EyeOpenRight


Nilai ini memungkinkan avatar:

- Berkedip  
- Menutup mata  
- Membuka mata secara natural  

---

## 3. Smoothing dan Stabilitas Gerakan

Agar animasi avatar terlihat halus dan tidak bergetar, digunakan metode **Exponential Moving Average (EMA)**:
```
smoothed = old_value * (1 - alpha) + new_value * alpha
```


Smoothing diterapkan pada:

- Rotasi kepala  
- Bukaan mulut  
- Bukaan mata  

Hal ini membuat pergerakan avatar di VTube Studio terlihat lebih natural dan stabil.

---

## Integrasi dengan VTube Studio

### 1. Koneksi WebSocket

Aplikasi terhubung ke:
```
ws://localhost:8001
```

Menggunakan **VTube Studio Public API**, dengan proses:

- Authentication Token Request  
- Authentication Request  
- Parameter Injection  

---

### 2. Mapping Parameter

| Parameter MediaPipe | Parameter VTube Studio |
|--------------------|------------------------|
| Head X | FaceAngleX |
| Head Y | FaceAngleY |
| Head Z | FaceAngleZ |
| Mouth Open | MouthOpen |
| Eye Left | EyeOpenLeft |
| Eye Right | EyeOpenRight |

Parameter dikirim menggunakan:
```
InjectParameterDataRequest
```

---

### 3. Heartbeat & Reconnect System

Aplikasi dilengkapi dengan:

- Heartbeat otomatis setiap 5 detik  
- Auto reconnect jika koneksi terputus  
- Monitoring status koneksi di GUI  

Fitur ini memastikan aplikasi tetap stabil saat digunakan dalam sesi VTuber yang panjang.

---

## Antarmuka Pengguna (GUI)

GUI dibuat menggunakan **Tkinter**, dengan fitur:

- Tampilan webcam real-time  
- Visualisasi Face Mesh MediaPipe  
- Status koneksi ke VTube Studio  
- Informasi parameter yang sedang aktif  

GUI berfungsi sebagai alat monitoring, sementara animasi avatar ditampilkan langsung di VTube Studio.

---

## Teknologi dan Dependensi

| Kategori | Teknologi | Fungsi |
|--------|----------|-------|
| Face Tracking | MediaPipe Face Mesh | Deteksi landmark wajah |
| Computer Vision | OpenCV | Capture webcam & visualisasi |
| GUI | Tkinter | Antarmuka monitoring |
| API Communication | WebSocket | Komunikasi dengan VTube Studio |
| Image Processing | PIL (Pillow) | Konversi frame ke Tkinter |
| Math Processing | Python Math | Normalisasi & rotasi |

---

## Cara Instalasi dan Eksekusi

### 1. Prasyarat

- Python  
- Webcam  
- VTube Studio (Running & Model Loaded)  

---

### 2. Instalasi Dependensi

```bash
pip install opencv-python mediapipe numpy pillow websocket-client
```
### 3. Menjalankan Program

- Buka VTube Studio
- Aktifkan Allow API Access
- Jalankan program:
  ```
  python barukepala.py
  ```
- Izinkan plugin saat pop-up autentikasi muncul di VTube Studio

---

## Limitasi Aplikasi
- Tidak ada Lip-Sync Audio : Bukaan mulut hanya berbasis visual, belum terintegrasi suara.
- Ekspresi Terbatas Parameter Default : Bergantung pada parameter yang tersedia di model Live2D.
- Tracking Masih 2D : Belum menggunakan solvePnP untuk rotasi 3D yang presisi.
- Single Face Only : Sistem hanya mendukung satu wajah.

## Rencana Pengembangan
- Integrasi Audio-Based Lip Sync
- Implementasi Hand Pose
- Support Custom Parameter Live2D
- Gesture-based expression (alis, senyum, marah)
---

## Kesimpulan
Project VTuber ini berhasil mengimplementasikan sistem Face Tracking real-time menggunakan MediaPipe yang terintegrasi langsung dengan VTube Studio melalui WebSocket API. Dengan memisahkan proses tracking dan rendering, aplikasi ini mampu memanfaatkan kualitas animasi Live2D secara maksimal, sekaligus mempertahankan fleksibilitas pengolahan data wajah di sisi Python.

Proyek ini menjadi fondasi yang kuat untuk pengembangan sistem VTuber profesional, baik untuk kebutuhan streaming, virtual presenter, maupun penelitian lanjutan di bidang Computer Vision dan Human–Computer Interaction.
