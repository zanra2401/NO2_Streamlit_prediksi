import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np # Diperlukan untuk typing

# ==============================
# 🔧 Fungsi bantu
# ==============================

# Berat molar NO2 (g/mol). Sumber: IUPAC
MOLAR_MASS_NO2_G_PER_MOL = 46.0055

# Konversi Gram (g) ke Mikrogram (µg)
G_TO_UG = 1_000_000

#
def fitur_column(lag: int):
    """Mengembalikan daftar nama kolom fitur sesuai jumlah lag."""
    # Nama kolom harus konsisten di seluruh skrip
    return [f"lag_{i}" for i in range(1, lag + 1)]

def konversi_kolom_ke_volume(mol_per_m2: float, ketinggian_m: float = 100.0) -> float:
    """
    Mengonversi Kepadatan Kolom (mol/m^2) menjadi Konsentrasi Volume (mol/m^3)
    dengan membaginya dengan ketinggian kolom udara (PBL).
    
    Ini adalah 'skala standar' yang kita gunakan di aplikasi Streamlit.
    """
    if ketinggian_m <= 0:
        return 0.0
    return mol_per_m2 / ketinggian_m

def konversi_ke_skala_standar(mol_per_m2: float) -> float:
    """
    Mengonversi Konsentrasi Volume (mol/m^3) menjadi Konsentrasi Massa (µg/m^3).
    """
    # Langkah 1: (mol/m^3) * (g/mol) = g/m^3
    g_per_m3 = konversi_kolom_ke_volume(mol_per_m2) * MOLAR_MASS_NO2_G_PER_MOL
    
    # Langkah 2: (g/m^3) * (µg/g) = µg/m^3
    ug_per_m3 = g_per_m3 * G_TO_UG
    
    return ug_per_m3


# ==============================
# Load Model & Scaler dari Path
# ==============================

MODEL_PATH = "knn_lag_3.pkl"
SCALER_PATH = "scaler_lag_3.pkl"

st.title("Prediksi NO₂ (Lag 3 Hari) Menggunakan Model KNN")

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error(f"File model '{MODEL_PATH}' atau scaler '{SCALER_PATH}' tidak ditemukan. Pastikan path sudah benar.")
    st.stop()

# Load model dan scaler
try:
    with open(MODEL_PATH, "rb") as f:
        model = joblib.load(f)

    with open(SCALER_PATH, "rb") as f:
        scaler = joblib.load(f)
    
    st.success("Model dan Scaler berhasil dimuat.")
    
except Exception as e:
    st.error(f"Gagal memuat file .pkl: {e}")
    st.stop()


# ==============================
# Input Data
# ==============================

st.subheader("Masukkan nilai NO₂ 3 hari terakhir (dalam mol/m²)")

col1, col2, col3 = st.columns(3)
# Urutan input disesuaikan dengan lag (lag_1 adalah kemarin)
no2_lag_1_input = col1.number_input("NO₂ Hari -1 (Kemarin)", min_value=0.0, value=0.000012, format="%.6f", step=0.000001)
no2_lag_2_input = col2.number_input("NO₂ Hari -2", min_value=0.0, value=0.000014, format="%.6f", step=0.000001)
no2_lag_3_input = col3.number_input("NO₂ Hari -3", min_value=0.0, value=0.000017, format="%.6f", step=0.000001)

# DAPATKAN NAMA FITUR YANG KONSISTEN
feature_names = fitur_column(3) # -> ['no2_lag_1', 'no2_lag_2', 'no2_lag_3']

# FIX 1: Gunakan nama kolom yang konsisten
# Pastikan urutan data sesuai dengan urutan nama kolom
input_df = pd.DataFrame({
    feature_names[0]: [no2_lag_1_input], # no2_lag_1
    feature_names[1]: [no2_lag_2_input], # no2_lag_2
    feature_names[2]: [no2_lag_3_input], # no2_lag_3
})

# Susun ulang kolom agar sesuai urutan training (jika diperlukan)
# Seringkali model dilatih dengan urutan lag terlama dulu
# Ganti baris ini jika urutan training Anda berbeda
try:
    expected_order = ['lag_3', 'lag_2', 'lag_1']
    input_df = input_df[expected_order]
    st.write("Data Input Awal (sebelum konversi):")
    st.dataframe(input_df)
except KeyError:
    st.error("Nama kolom input tidak sesuai dengan 'expected_order'. Cek fungsi fitur_column.")
    st.stop()


st.subheader("2. Prediksi")

if st.button("Jalankan Prediksi"):
    try:
        # 2a. Scaling input
        # scaler.transform mengembalikan NumPy array
        X_scaled_array = scaler.transform(input_df)

        # FIX 2: Ubah array kembali ke DataFrame dengan nama kolom yang benar
        # Ini akan menghilangkan UserWarning
        X_scaled_df = pd.DataFrame(X_scaled_array, columns=input_df.columns)

        st.write("Data setelah scaling (Z-score):")
        st.dataframe(X_scaled_df)

        # 2b. Prediksi
        # Gunakan DataFrame yang sudah di-scaling
        y_pred = model.predict(X_scaled_df)[0]

        st.success("Prediksi berhasil dilakukan!")

        st.metric("Perkiraan NO₂ (skala standar)", f"{konversi_ke_skala_standar(y_pred):.1f} µg/m³")
        st.metric("Perkiraan NO₂ (mol/m² asli)", f"{y_pred:.6f} mol/m²")
        # Ambil hasil prediksi dalam µg/m³
        no2_ug_m3 = konversi_ke_skala_standar(y_pred)

        # Berdasarkan panduan WHO 2021:
        # - Baik  : ≤ 25 µg/m³ (rata-rata 24 jam)
        # - Buruk : > 25 µg/m³
        if no2_ug_m3 <= 25:
            kualitas = "Baik ✅"
            warna = "green"
            deskripsi = "Kualitas udara aman menurut standar WHO (≤ 25 µg/m³)."
        else:
            kualitas = "Buruk ⚠️"
            warna = "red"
            deskripsi = "Kualitas udara melebihi batas aman WHO (> 25 µg/m³)."

        st.subheader("3. Kualitas Udara Berdasarkan Prediksi")
        st.markdown(f"<h4 style='color:{warna}'>{kualitas}</h4>", unsafe_allow_html=True)
        st.write(deskripsi)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
