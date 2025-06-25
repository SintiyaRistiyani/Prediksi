import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# --------------- Fungsi bantu (opsional bisa dikembangkan) ---------------
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return None

def check_stationarity(df, column):
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(df[column].dropna())
    return result

# ----------------- Sidebar Navigasi -----------------
st.sidebar.title("📊 Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", [
    "Home", 
    "Input Data", 
    "Data Preprocessing", 
    "Stasioneritas", 
    "Model", 
    "Prediksi dan Visualisasi", 
    "Interpretasi dan Saran"
])

# ----------------- Halaman Home -----------------
if menu == "Home":
    st.title("📈 Aplikasi Prediksi Harga Saham")
    st.markdown("""
        Selamat datang di aplikasi prediksi harga saham berbasis **Streamlit**.  
        Silakan gunakan menu di samping untuk mengakses berbagai fitur mulai dari input data hingga interpretasi hasil model.
    """)

# ----------------- Halaman Input Data -----------------
elif menu == "Input Data":
    st.title("📥 Input Data")
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:
            st.success("Data berhasil dimuat!")
            st.dataframe(df.head())

# ----------------- Halaman Data Preprocessing -----------------
elif menu == "Data Preprocessing":
    st.title("🧹 Data Preprocessing")
    if 'df' not in locals():
        st.warning("Silakan upload data terlebih dahulu di halaman Input Data.")
    else:
        st.markdown("Contoh preprocessing: mengisi missing value.")
        st.write(f"Jumlah nilai kosong sebelum: \n{df.isnull().sum()}")
        df.fillna(method='ffill', inplace=True)
        st.write(f"Jumlah nilai kosong setelah: \n{df.isnull().sum()}")
        st.dataframe(df.head())

# ----------------- Halaman Stasioneritas -----------------
elif menu == "Stasioneritas":
    st.title("📉 Uji Stasioneritas")
    if 'df' not in locals():
        st.warning("Silakan upload data terlebih dahulu di halaman Input Data.")
    else:
        kolom = st.selectbox("Pilih kolom untuk diuji:", df.columns)
        result = check_stationarity(df, kolom)
        st.markdown(f"""
        **ADF Statistic**: {result[0]:.4f}  
        **p-value**: {result[1]:.4f}  
        """)
        if result[1] < 0.05:
            st.success("Data stasioner (tolak H0)")
        else:
            st.error("Data tidak stasioner (gagal tolak H0)")

# ----------------- Halaman Model -----------------
elif menu == "Model":
    st.title("🔧 Pemodelan")
    st.markdown("Di sini Anda bisa menerapkan model seperti ARIMA, MAR, atau model lainnya.")
    st.info("Silakan integrasikan model Anda di sini.")

# ----------------- Halaman Prediksi dan Visualisasi -----------------
elif menu == "Prediksi dan Visualisasi":
    st.title("📊 Prediksi dan Visualisasi")
    st.markdown("Tampilkan hasil prediksi dan visualisasinya di sini.")
    # Dummy data contoh
    t = np.arange(100)
    pred = np.sin(t/10)
    real = pred + np.random.normal(0, 0.1, size=100)
    plt.figure(figsize=(10,4))
    plt.plot(t, real, label="Real")
    plt.plot(t, pred, label="Prediksi")
    plt.legend()
    st.pyplot(plt)

# ----------------- Halaman Interpretasi dan Saran -----------------
elif menu == "Interpretasi dan Saran":
    st.title("📝 Interpretasi dan Saran")
    st.markdown("""
        #### Interpretasi:
        - Model menunjukkan performa yang cukup baik dalam menangkap tren harga saham.
        - Terdapat fluktuasi yang masih bisa diperbaiki pada komponen residual.

        #### Saran:
        - Lakukan tuning parameter pada model.
        - Pertimbangkan faktor eksternal seperti berita pasar atau sentimen investor.
        - Gunakan model lanjutan seperti **Mixture Autoregressive (MAR)** untuk data yang memiliki switching regime.
    """)
