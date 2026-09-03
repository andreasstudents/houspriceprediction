import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Data Properti", page_icon="🏠", layout="wide")
current_dir = os.path.dirname(os.path.abspath(__file__))

st.title("📚 Raw Dataset")
st.markdown(
    """
    Data ini merupakan hasil scraping dari [rumah123.com](https://www.rumah123.com) 
    untuk wilayah Yogyakarta, yang mencakup berbagai informasi properti.
    """
)


@st.cache_data
def load_data():
    try:
        df = pd.read_csv(os.path.join(current_dir, "../dataset/houses.csv"))
        return df
    except FileNotFoundError:
        st.error(
            "❌ Dataset tidak ditemukan. Pastikan file `houses-cleaned.csv` ada di folder yang benar."
        )
        return None


df = load_data()
if df is None:
    st.stop()

with st.expander("🗂️ Deskripsi Dataset", expanded=False):
    st.markdown("Berikut adalah penjelasan dari setiap kolom dalam dataset:")
    deskripsi_kolom = pd.DataFrame(
        {
            "Kolom": [
                "title",
                "price",
                "bedroom",
                "bathroom",
                "carport",
                "LT",
                "LB",
                "badges",
                "agent",
                "updated",
                "location",
                "link",
            ],
            "Deskripsi": [
                "Judul listing rumah",
                "Harga rumah (dalam satuan tertentu)",
                "Jumlah kamar tidur",
                "Jumlah kamar mandi",
                "Jumlah carport",
                "Luas tanah (m²)",
                "Luas bangunan (m²)",
                "Label tambahan seperti 'Dekat Sekolah', dll",
                "Nama agen penjual",
                "Tanggal terakhir pembaruan",
                "Lokasi rumah",
                "Link ke halaman detail listing",
            ],
        }
    )
    st.table(deskripsi_kolom)

st.subheader("🧩 Filter Dataset")

col1, col2 = st.columns([2, 2])
with col1:
    selected_columns = st.multiselect(
        "Pilih kolom untuk ditampilkan:",
        options=df.columns.tolist(),
        default=df.columns.tolist(),
    )

with col2:
    if "Harga" in df.columns:
        min_price, max_price = st.slider(
            "Filter berdasarkan harga:",
            min_value=int(df["Harga"].min()),
            max_value=int(df["Harga"].max()),
            value=(int(df["Harga"].min()), int(df["Harga"].max())),
        )
        df = df[(df["Harga"] >= min_price) & (df["Harga"] <= max_price)]

filtered_df = df[selected_columns]

st.subheader("📋 Tabel Dataset")
st.dataframe(filtered_df.head(10), use_container_width=True)

st.subheader("📈 Statistik Deskriptif")
st.dataframe(df.describe(), use_container_width=True)

st.subheader("🔎 Informasi Dataset")
st.markdown(
    f"""
- Jumlah baris: **{df.shape[0]}**
- Jumlah kolom: **{df.shape[1]}**
- Nama kolom: `{', '.join(df.columns)}`
"""
)

st.subheader("🧼 Cek Nilai Null / Missing")
missing_info = df.isnull().sum().reset_index()
missing_info.columns = ["Kolom", "Jumlah Nilai Kosong"]
missing_info["Persentase"] = (missing_info["Jumlah Nilai Kosong"] / len(df)) * 100
missing_info = missing_info[missing_info["Jumlah Nilai Kosong"] > 0]

if missing_info.empty:
    st.success("✅ Tidak ada nilai kosong dalam dataset.")
else:
    st.warning("⚠️ Ditemukan nilai kosong pada beberapa kolom.")
    st.table(missing_info)

st.subheader("ℹ️ Informasi Kolom")
column_info = pd.DataFrame(
    {
        "Kolom": df.columns,
        "Tipe Data": df.dtypes.values,
        "Jumlah Nilai Unik": df.nunique().values,
    }
)
st.table(column_info)

if "Harga" in df.columns:
    st.subheader("📊 Distribusi Harga Rumah")
    fig, ax = plt.subplots(figsize=(10, 4))
    df["Harga"].plot(kind="hist", bins=20, alpha=0.7, ax=ax, color="#1f77b4")
    ax.set_title("Distribusi Harga Rumah")
    ax.set_xlabel("Harga")
    ax.set_ylabel("Frekuensi")
    st.pyplot(fig)


st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
    Raw Dataset
    </div>
    """,
    unsafe_allow_html=True,
)
