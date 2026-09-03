# 🕷️ Rumah123 Web Scraper

Scraper Python sederhana namun powerful untuk mengambil data listing rumah dari [rumah123.com](https://www.rumah123.com) khusus area Yogyakarta. Data hasil scraping akan disimpan dalam format CSV yang bisa digunakan untuk analisis data, machine learning, atau keperluan lainnya.

## 📦 Fitur

- Scrape otomatis seluruh halaman listing rumah.
- Ambil informasi penting seperti:
  - Judul
  - Harga
  - Jumlah kamar tidur, kamar mandi, dan carport
  - Luas tanah dan bangunan
  - Lokasi, deskripsi, dan agen properti
  - Badges/promosi
- Penanganan rate-limit otomatis (HTTP 429) dengan backoff algoritma.
- Simpan data ke dalam file CSV siap pakai.
- Menampilkan sampel data setelah scraping.

---

## 🛠️ Instalasi

1. **Clone repository ini**

   ```bash
   git clone https://github.com/username/rumah123-scraper.git
   cd machine-learning/dataset-scraper
   ```

2. **Buat virtual environment (opsional namun direkomendasikan)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Tambahkan file `ua.txt`**  
   File ini harus berisi daftar User-Agent (satu per baris). Contoh isi `ua.txt`:
   ```
   Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 ...
   Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 ...
   ```

---

## 🚀 Menjalankan Scraper

Jalankan script utama:

```bash
python scraper.py
```

Scraper akan menelusuri setiap halaman hingga tidak ada properti lagi yang ditemukan dan menyimpan hasilnya ke:

```
../dataset/houses.csv
```

---

## 🗃️ Contoh Output (CSV)

```csv
title,price,bedroom,bathroom,carport,LT,LB,badges,agent,updated,location,link
"Termurah Dan Terbaik ! Rumah Cantik Dengan Kontruksi Mewah Harga Murah Di Perumahan Mustika Sedayu Jalan Wates Km 13","Rp 319 Juta","2","1","1",": 97 m²",": 36 m²","Rumah, Premier, Dekat Sekolah","Ayanti Sahabat Properti Jogja","Diperbarui 6 hari yang lalu oleh","Sedayu, Bantul","/properti/bantul/hos15809287/"
```

---

## 🧩 Dependencies

- `beautifulsoup4`
- `tqdm`
- `colorama`

Install dengan:

```bash
pip install -r requirements.txt
```

---

## 📁 Struktur Folder

```
dataset-scraper/
├── scraper.py
├── ua.txt
├── requirements.txt
```

---

## 📝 Catatan

- Website dapat berubah sewaktu-waktu. Jika struktur HTML berubah, scraper ini mungkin perlu diperbarui.
- Harap gunakan scraper ini secara etis dan sesuai dengan [robots.txt](https://www.rumah123.com/robots.txt) serta TOS situs terkait.

---

## 📬 Kontak

Dikembangkan oleh [Fajar Jati Nugroho](https://github.com/FjrREPO).  
Feel free to open an issue or pull request!

---

## 📜 Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE).