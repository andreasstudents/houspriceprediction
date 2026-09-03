import streamlit as st

# Judul Web
st.title("Simulasi Prediksi Harga Rumah Sederhana")

# Input Field dari User
luas_tanah = st.number_input("Masukkan Luas Tanah (m2):", min_value=0)

# Button untuk Prediksi
if st.button("Prediksi Harga"):
    # Function Simulasi: Harga = Luas x 1 Juta
    harga_prediksi = luas_tanah * 1000000
    st.success(f"Perkiraan Harga Rumah: Rp {harga_prediksi:,.0f}")
