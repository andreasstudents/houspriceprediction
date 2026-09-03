import streamlit as st

st.set_page_config(page_title="Home", page_icon="🏠", layout="wide")

st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .highlight {
        background-color: #F0F7FF;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1E3A8A;
        margin-bottom: 1rem;
    }
    .feature-card {
        background-color: #FFFFFF;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .tech-badge {
        display: inline-block;
        background-color: #E5E7EB;
        color: #374151;
        padding: 5px 10px;
        border-radius: 15px;
        margin: 5px;
        font-size: 0.9rem;
    }
    .footer {
        margin-top: 3rem;
        text-align: center;
        color: #6B7280;
        font-size: 0.8rem;
    }
    .sidebar-content {
        padding: 15px;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 class='main-header'>🏠 Implementasi Fuzzy Logic dan Optimasi Algoritma Genetika pada Random Forest untuk Prediksi Harga Rumah di Kota Yogyakarta</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    """
    Selamat datang di sistem **prediksi harga rumah** untuk wilayah **Kota Yogyakarta**.

    Aplikasi ini memanfaatkan pendekatan hybrid cerdas yang menggabungkan:
    
    - 🔍 **Fuzzy Logic** — untuk menangani variabel kualitatif seperti jumlah kamar atau ketidakpastian pada luas bangunan.
    - 🌲 **Random Forest** — model pembelajaran mesin berbasis ensemble.
    - 🧬 **Genetic Algorithm** — digunakan untuk mengoptimalkan parameter model agar akurasi meningkat.
    
    Sistem ini membantu pengguna memperkirakan harga rumah berdasarkan fitur utama seperti:
    - Luas tanah (`LT`)
    - Luas bangunan (`LB`)
    - Jumlah kamar tidur (`bedroom`)
    - Jumlah kamar mandi (`bathroom`)
    - Lokasi (`kecamatan` dan `kabupaten/kota`)
    - Waktu terakhir listing diperbarui (`updated`)
    """
)

st.markdown("---")

st.subheader("🔍 Github")
st.markdown(
    """
    <div class="highlight">
    <p style="font-size: 1.2rem; font-weight: bold;">Kunjungi repositori GitHub untuk melihat kode lebih lanjut:</p>
    <a href="https://github.com/FjrREPO/house-price-prediction" target="_blank">
    <button style="background-color: #1E3A8A; color: white; border: none; padding: 10px 20px; border-radius: 5px; font-weight: bold; font-size: 1rem; cursor: pointer;">Lihat di GitHub</button>
    </a>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

st.subheader("🎯 Tujuan Aplikasi")
st.markdown(
    """
    Aplikasi ini dikembangkan sebagai bagian dari penelitian untuk prediksi harga properti residensial 
    secara lebih adaptif dan realistis, dengan cakupan terbatas hanya pada wilayah Kota Yogyakarta. 
    Aplikasi ini tidak ditujukan untuk wilayah selain Kota Yogyakarta maupun di luar wilayah tersebut.

    Dengan kombinasi teknik fuzzy dan pembelajaran mesin, sistem ini dapat memahami pola harga rumah 
    yang seringkali tidak linier atau tidak pasti.
    """
)

st.markdown("---")

st.sidebar.title("📌 Navigasi")
st.sidebar.markdown("Gunakan menu di atas atau sidebar untuk berpindah halaman.")

st.sidebar.info(
    """
### Tentang Aplikasi
Versi ini menampilkan halaman utama sebagai pengantar sistem.

Untuk mencoba prediksi harga, silakan buka halaman **Prediksi** di menu samping.
"""
)

st.markdown(
    "<h2 class='sub-header'>🛠️ Teknologi yang Digunakan</h2>", unsafe_allow_html=True
)

st.markdown(
    """
<div style="text-align: center; padding: 20px;">
    <span class="tech-badge">🐍 Python</span>
    <span class="tech-badge">📊 Pandas</span>
    <span class="tech-badge">🔢 NumPy</span>
    <span class="tech-badge">🤖 Scikit-learn</span>
    <span class="tech-badge">🧠 Scikit-Fuzzy</span>
    <span class="tech-badge">🧬 DEAP (Genetic Algorithm)</span>
    <span class="tech-badge">📈 Matplotlib</span>
    <span class="tech-badge">🗺️ Folium</span>
    <span class="tech-badge">📦 Streamlit</span>
    <span class="tech-badge">🧹 NLTK</span>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div style="background-color: #1E3A8A; padding: 30px; border-radius: 10px; color: white; text-align: center; margin-top: 40px;">
    <h2>Siap untuk Memprediksi Harga Rumah?</h2>
    <p style="font-size: 1.2rem;">Gunakan fitur prediksi kami untuk memperkirakan harga properti berdasarkan preferensi Anda.</p>
    <a href="/Prediction" target="_self">
    <button style="background-color: white; color: #1E3A8A; border: none; padding: 12px 24px; border-radius: 5px; font-weight: bold; font-size: 1.1rem; cursor: pointer; margin-top: 15px;">Mulai Prediksi Sekarang</button>
    </a>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="footer">
    <p>© 2025 Machine Learning Prediksi Harga Rumah Yogyakarta | Data terakhir diperbarui: 12 April 2025</p>
</div>
""",
    unsafe_allow_html=True,
)
