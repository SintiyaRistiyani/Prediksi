import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# ----------------- Fungsi bantu -----------------
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
menu = st.sidebar.radio("Pilih Halaman:", (
    "Home", 
    "Input Data", 
    "Data Preprocessing", 
    "Stasioneritas", 
    "Model", 
    "Prediksi dan Visualisasi", 
    "Interpretasi dan Saran"
))

# ----------------- Halaman Home -----------------
if menu == "Home":
    st.title("📈 Aplikasi Prediksi Harga Saham Menggunakan Model ARIMA dan MAR")
    st.markdown("""
        Selamat datang di aplikasi prediksi harga saham berbasis **Streamlit**.  
        Silakan gunakan menu di samping untuk mengakses berbagai fitur mulai dari input data hingga interpretasi hasil model.
        
        Ketentuan :
        1. file harus dalam bentuk csv
        2. data harus memiliki kolom date/tanggal dan harga saham

    """)

# ----------------- Halaman Input Data -----------------
elif menu == "Input Data":
    st.title("📥 Input Data")
    st.markdown("""
        Ketentuan :
        1. file harus dalam bentuk csv
        2. data harus memiliki kolom date/tanggal dan harga saham
    """)
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:
            st.session_state['df'] = df  # simpan ke session
            st.success("Data berhasil dimuat!")
            st.dataframe(df.head())

# ----------------- Halaman Data Preprocessing -----------------
elif menu == "Data Preprocessing":
    st.title("🧹 Data Preprocessing")
    
    if 'df' not in st.session_state:
        st.warning("Silakan upload data terlebih dahulu di halaman Input Data.")
    else:
        df = st.session_state['df']
        st.markdown("### 1️⃣ Pilih Kolom Data Saham yang Ingin Dianalisis")

        selected_column = st.selectbox("Pilih kolom perusahaan / harga:", df.columns)
        st.session_state['selected_column'] = selected_column

        st.markdown(f"Data asli dari kolom **{selected_column}**:")
        st.line_chart(df[selected_column])

        # Missing Value Handling
        st.markdown("### 2️⃣ Penanganan Missing Value")
        st.write(f"Jumlah nilai kosong sebelum: {df[selected_column].isnull().sum()}")
        method = st.selectbox("Metode pengisian missing value", ['ffill', 'bfill', 'interpolate', 'drop'])

        if method == 'drop':
            df_clean = df[selected_column].dropna()
        elif method == 'interpolate':
            df_clean = df[selected_column].interpolate()
        else:
            df_clean = df[selected_column].fillna(method=method)

        st.write(f"Jumlah nilai kosong setelah: {df_clean.isnull().sum()}")

        # Log Return
        st.markdown("### 3️⃣ Hitung Log Return")
        log_return = np.log(df_clean / df_clean.shift(1)).dropna()
        st.dataframe(log_return.head())
        st.session_state['log_return'] = log_return

        # Visualisasi
        st.markdown("### 4️⃣ Visualisasi Data dan Log Return")
        st.line_chart(df_clean, use_container_width=True)
        st.line_chart(log_return, use_container_width=True)

# ----------------- Halaman Stasioneritas -----------------
elif menu == "Stasioneritas":
    st.title("📉 Uji Stasioneritas")
    if 'df' not in st.session_state:
        st.warning("Silakan upload data terlebih dahulu di halaman Input Data.")
    else:
        df = st.session_state['df']
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
