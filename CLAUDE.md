# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Layout Repo (penting)

Direktori kerja `D:\7. Semester\HousePrediction` hanyalah pembungkus. Seluruh kode, dataset, dan git repository ada di subfolder `house-price-prediction/`. Jalankan semua perintah dari dalam subfolder tersebut.

Ada beberapa virtual environment: `venv/` dan `.venv/` di dalam proyek (Python 3.13), serta `../.venv` di tingkat pembungkus (Python 3.11).`requirements.txt` ada di root proyek.

## Perintah Umum

Semua dari `house-price-prediction/`:

```bash
# Aktifkan venv (Windows)
venv\Scripts\activate

# Jalankan aplikasi Streamlit — HARUS dari root proyek (pages/ memakai path relatif seperti glob "processed_data/...")
streamlit run Home.py

# Preprocessing dataset mentah → processed_data/ + reports/
python train/preprocess.py --input dataset/houses.csv --city Yogyakarta
# --input, --output-dir, --city opsional; default: dataset/houses.csv, kota "Yogyakarta"

# Training lengkap: base RF → optimasi GA → RF teroptimasi → ensemble → simpan model
python train/train.py

# Scraper data rumah123.com (lihat README di folder tersebut)
python dataset/dataset-scraper/scraper.py
```

Tidak ada test suite maupun linter yang dikonfigurasi.

Catatan:
- `train/train.py` tidak punya CLI argument (README lama menyebut `--use-ga-optimization` — usang; GA selalu dijalankan). Jika tidak menemukan file preprocessed, skrip memanggil `input()` interaktif — selalu jalankan preprocessing dulu.
- Urutan wajib: preprocess → train → baru aplikasi prediksi berfungsi dengan model baru.

## Arsitektur

Pipeline satu arah: **scrape → preprocess → train → predict**, dengan artefak di antara tahap (bukan komunikasi langsung antar modul).

### Alur data

1. **Scrape** — `dataset/dataset-scraper/scraper.py` mengambil listing rumah123.com (Yogyakarta) → `dataset/houses.csv`. Kolom mentah: `title`, `price` (string "Rp 1,5 Miliar"), `bedroom`, `bathroom`, `carport`, `LT`/`LB` (string "150 m²"), `location` ("Kecamatan, Kabupaten"), `updated` ("3 hari lalu").

2. **Preprocessing** — `train/preprocess.py` (CLI wrapper) memanggil fungsi kanonik `preprocess_data()` di `utils/preprocessing.py`, yang melakukan: parsing string harga/luas/tanggal relatif → split location jadi `kecamatan`/`kabupaten_kota` → buang judul hotel/kost, dedup judul, buang listing >60 hari, filter kota → cek kualitas + imputasi median → outlier removal (IQR + z-score, threshold per kolom di dict `IQR_THRESHOLDS`/`Z_SCORE_THRESHOLDS`) → feature engineering (`price_per_m2_*`, `building_efficiency`, `listing_age_days`, dll.) → fuzzy logic → label encoding (`kecamatan_encoded`, mapping disimpan ke `reports/encoder_mappings.json`). Output: `processed_data/preprocessed_data_<timestamp>.pkl` (dict berisi `data`/`feature_columns`/`preprocessing_info`) + CSV + metadata JSON.

3. **Training** — `train/train.py` memuat file `.pkl` terbaru di `processed_data/` (diurut mtime), lalu melakukan feature engineering tahap kedua: buang outlier harga IQR lagi, target `log1p(price)`, bangun ~24 fitur, **mengecualikan semua fitur turunan harga** (`price_per_m2_*`, `price`) untuk mencegah data leakage. Fitur diskalakan `StandardScaler` (disimpan sebagai `model/feature_scaler.pkl`). Lalu berurutan: base RF → `GAOptimizer` (`train/ga_optimizer.py`: tournament selection, crossover aritmetika, mutasi gaussian adaptif, elitisme, early stopping; fitness = composite MAPE/R²/RMSE/MAE yang dihitung di skala asli setelah `expm1`) → RF teroptimasi → tiga ensemble (`train/ensemble_model.py`: voting, stacking dengan meta-learner Ridge, BlendedEnsemble berbobot inverse-MAPE). Model terbaik disimpan sebagai `model/best_<tipe>_ensemble.pkl`, RF sebagai `model/optimized_rf_model.pkl`, plus metadata JSON/TXT dan plot di `model/evaluation/`. Log di `logs/`.

4. **Aplikasi** — Streamlit multipage: `Home.py` entry point, `pages/1_Dataset.py` (EDA), `pages/2_Preprocessing.py` (tampilkan hasil preprocessing), `pages/3_Model.py` (statistik/history training), `pages/4_Prediction.py` (form prediksi). Halaman prediksi mereplikasi daftar 24 fitur yang sama dengan `train/train.py` secara hardcoded, men-scale dengan `feature_scaler.pkl`, memprediksi, lalu `np.expm1` kembali ke rupiah. Encoding kecamatan dibaca dari `reports/encoder_mappings.json` (fallback: list hardcoded). Ada fallback kalkulasi heuristik jika model gagal.

### Kontrak log-transform (kritis)

Semua training memakai target `y = np.log1p(price)`; evaluator (`rf_model.py`, `ensemble_model.py`), fitness GA, dan halaman prediksi mengembalikan ke skala rupiah dengan `np.expm1()`. Model dilatih ulang dari `train/train.py` menandai dirinya dengan atribut custom `_is_log_transformed = True`. Jika mengubah salah satu sisi (training/evaluasi/prediksi), ketiga sisi lain harus konsisten — mencampur skala log dan rupiah menghasilkan metrik/prediksi yang salah diam-diam.

### Fuzzy logic

`utils/preprocessing.py:create_fuzzy_systems()` membangun 4 sistem scikit-fuzzy: kategorisasi harga, kualitas ruang (bedroom × bathroom), kualitas area (LT × LB), dan evaluasi carport. Hasilnya menjadi fitur `fuzzy_area_quality` dan `fuzzy_quality_score` (rata-rata metrik fuzzy). Fungsi yang sama dipanggil ulang saat prediksi untuk satu rumah — perubahan membership/rule harus tetap sinkron karena model lama dilatih dengan nilai fuzzy lama.

### Duplikasi kode yang harus disinkronkan

Logika parsing/bersih-bersih ada dalam 3 salinan dengan perilaku & threshold berbeda:
- `utils/preprocessing.py` — kanonik, dipakai pipeline preprocessing.
- `utils/helper.py` — salinan lama; TIDAK dipakai pipeline training (threshold berbeda: mis. price max 20e9 vs 10e9).
- `train/preprocess.py:clean_numeric` — parsing kasar sebelum masuk `preprocess_data()`.

Demikian pula daftar fitur engineered: `train/train.py` (`base_features` + blok derivasi) vs `pages/4_Prediction.py:predict_house_price()` (replika manual, termasuk ambang tier LT/LB). Mengubah fitur di satu tempat tanpa tempat lain membuat model dan app prediksi tidak selaras.

### Quirk / bug yang diketahui

- `pages/4_Prediction.py` mendefinisikan `ENSEMBLE_PATH` sebagai `best_stacking_esemble.pkl` (typo) sedangkan file aktual `model/best_stacking_ensemble.pkl` — sehingga app selalu jatuh ke `optimized_rf_model.pkl`.
- `pages/2_Preprocessing.py` meng-hardcode nama file metadata/pickle tertentu (`preprocessing_metadata_20250506_140027.json`), bukan yang terbaru.
- Metrik & feature-importance di README.md berasal dari versi model lama yang masih memakai fitur price-derived (price 40%, price_per_m2_land 38%) — pipeline saat ini mengecualikan fitur tersebut demi menghindari leakage, jadi angka README tidak merefleksikan model sekarang.
- Metrik evaluasi dihitung di skala rupiah asli (setelah inverse log), bukan di ruang log.

### Konvensi import

Skrip di `train/` saling import secara datar (`from rf_model import ...`, `from ga_optimizer import GAOptimizer`) dan mengandalkan sys.path otomatis dari direktori skrip, plus `sys.path.insert` eksplisit untuk parent agar `from utils.preprocessing import ...` bekerja. Halaman Streamlit memakai `from train.rf_model import ...` setelah insert parent dir. `.vscode/settings.json` menambahkan `./train` ke `python.analysis.extraPaths`.
