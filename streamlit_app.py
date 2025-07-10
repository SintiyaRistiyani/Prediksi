import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
from scipy.stats import skew, kurtosis
import seaborn as sns


from sklearn.cluster import KMeans
from scipy.stats import gennorm, norm, kstest
from statsmodels.tools.tools import add_constant
from statsmodels.stats.diagnostic import acorr_ljungbox, het_white
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_percentage_error

# Utility
from io import StringIO

# ----------------- Fungsi bantu -----------------
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return None

def check_stationarity(series):
    result = adfuller(series.dropna())
    return result

def diagnostik_saham(series, nama_saham):
    st.markdown(f"## 🧪 Uji Diagnostik Distribusi: {nama_saham}")

    if series is None or len(series) == 0:
        st.warning("Series log return kosong.")
        return

    series = series.dropna()

    # Skewness & Kurtosis
    skw = skew(series)
    krt = kurtosis(series)
    st.write(f"**Skewness:** {skw:.4f}")
    st.write(f"**Kurtosis:** {krt:.4f}")

    # Visualisasi histogram + KDE
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(series, kde=True, bins=30, color='skyblue', ax=ax)
    ax.set_title(f'Distribusi Log Return {nama_saham}')
    ax.set_xlabel('Log Return')
    ax.set_ylabel('Frekuensi')
    st.pyplot(fig)

# ----------------- Sidebar Navigasi -----------------
st.sidebar.title("📊 Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", (
    "Home", 
    "Input Data", 
    "Data Preprocessing", 
    "Stasioneritas", 
    "Model", 
    "Uji Signifikansi dan Residual", 
    "Prediksi dan Visualisasi", 
    "Interpretasi dan Saran"
))

# ----------------- Halaman Home -----------------
if menu == "Home":
    st.title("Aplikasi Prediksi Harga Saham Menggunakan Model Mixture Autoregressive (MAR)")
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
    uploaded_file = st.file_uploader("📂 Upload File CSV (delimiter = ';')", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, delimiter=';')
            df.columns = df.columns.str.strip()
            if 'Date' not in df.columns:
                st.error("Kolom 'Date' tidak ditemukan.")
                st.stop()
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date']).sort_values('Date')
            harga_col = st.selectbox("Pilih kolom harga:", df.columns)
            date_col = st.selectbox("Pilih kolom tanggal:",df.columns)
            df[harga_col] = pd.to_numeric(df[harga_col]
                                          .astype(str)
                                          .str.replace('.', '', regex=False)
                                          .str.replace(',', '.', regex=False)
                                          .str.replace('[^0-9.-]', '', regex=True),
                                          errors='coerce')
            df = df.dropna(subset=[harga_col])
            st.session_state['df'] = df
            st.session_state['harga_col'] = harga_col
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error: {e}")
            
# ----------------- Halaman Preprocessing -----------------
elif menu == "Data Preprocessing":
    st.title("⚙️ Preprocessing Data")

    if 'df' not in st.session_state or 'harga_col' not in st.session_state:
        st.warning("Upload data terlebih dahulu.")
        st.stop()

    df = st.session_state['df'].copy()
    harga_col = st.session_state['harga_col']

    # Fungsi format ke rupiah Indonesia (29.153,00)
    def format_harga_idr(x):
        return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    # Bersihkan kolom harga (jika masih format "29.153,00" → float)
    def clean_price_column(series):
        return series.astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)

    # Bersihkan dan hitung log return
    df[harga_col] = clean_price_column(df[harga_col])
    df['Harga'] = df[harga_col].apply(format_harga_idr)

    # Hitung log return
    df['Log Return'] = np.log(df[harga_col] / df[harga_col].shift(1))
    df = df.dropna().reset_index(drop=True)

    # Tampilkan tabel 5 baris pertama
    st.markdown("### Tabel 5 Baris Pertama")
    st.dataframe(df[['Date', 'Harga', 'Log Return']].head())

    # ----------------- Split Data -----------------
    st.markdown("### Split Data (Train/Test)")
    n_test = 30
    train, test = df[:-n_test], df[-n_test:]
    st.session_state['log_return_train'] = train['Log Return']
    st.session_state['train'] = train
    st.session_state['test'] = test

    # Visualisasi log return
    st.line_chart({
        'Train': train.set_index('Date')['Log Return'],
        'Test': test.set_index('Date')['Log Return']
    })
    
# ----------------- Halaman Uji Stasioneritas -----------------
elif menu == "Stasioneritas":
    st.title("📉 Uji Stasioneritas dan Diagnostik Distribusi")

    # Validasi data
    if 'train' not in st.session_state or 'harga_col' not in st.session_state:
        st.warning("Silakan lakukan preprocessing terlebih dahulu.")
        st.stop()

    train = st.session_state['train']
    harga_col = st.session_state['harga_col']
    
    # Panggil fungsi dengan log return dari data train
    diagnostik_saham(train['Log Return'], harga_col)

    # === Fungsi Uji ADF ===
    from statsmodels.tsa.stattools import adfuller

    def check_stationarity(series):
        result = adfuller(series.dropna())
        return result

    # Hitung ADF
    adf_result = check_stationarity(train['Log Return'])
    st.markdown("### 🔍 Hasil Uji ADF")
    st.write(f"**ADF Statistic:** {adf_result[0]:.4f}")
    st.write(f"**p-value:** {adf_result[1]:.4f}")
    st.write("**Kesimpulan:**", 
             "✅ Stasioner (p < 0.05)" if adf_result[1] < 0.05 else "⚠️ Tidak Stasioner (p ≥ 0.05)")

    # === Plot ACF & PACF ===
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    import matplotlib.pyplot as plt

    st.markdown("### 🔁 ACF dan PACF Plot")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(train['Log Return'], ax=ax[0], lags=20)
    plot_pacf(train['Log Return'], ax=ax[1], lags=20, method='ywm')
    ax[0].set_title("ACF")
    ax[1].set_title("PACF")
    st.pyplot(fig)

    from scipy.stats import skew, kurtosis, shapiro, jarque_bera

    def diagnostik_saham(series, nama_saham):
        st.markdown(f"## 🧪 Uji Diagnostik Distribusi: {nama_saham}")
        series = series.dropna()
        
        # Skewness & Kurtosis
        skw = skew(series)
        krt = kurtosis(series)
        st.write(f"**Skewness:** {skw:.4f}")
        st.write(f"**Kurtosis:** {krt:.4f}")
    
        # Visualisasi histogram + KDE
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(series, kde=True, bins=30, color='skyblue', ax=ax)
        ax.set_title(f'Distribusi Log Return {nama_saham}')
        ax.set_xlabel('Log Return')
        ax.set_ylabel('Frekuensi')
        st.pyplot(fig)

# =============================== MODEL ==============================
elif menu == "Model":
    st.title("🏗️ Pemodelan Mixture Autoregressive (MAR)")

    if 'log_return_train' not in st.session_state:
        st.warning("Lakukan preprocessing terlebih dahulu.")
        st.stop()

    X = st.session_state['log_return_train'].values

    model_choice = st.selectbox("Pilih Jenis Model:", ["MAR-Normal", "MAR-GED"])
    metode_pemodelan = st.radio("Pilih Metode Pemodelan:", ["Otomatis (EM + BIC)", "Manual"])

    if metode_pemodelan == "Otomatis (EM + BIC)":
        max_p = st.slider("Max order p:", 1, 5, 3)
        max_K = st.slider("Max jumlah komponen K:", 1, 5, 3)

        if st.button("🔁 Cari Struktur Terbaik (p & K)"):
            with st.spinner("Mencari struktur terbaik..."):
                if model_choice == "MAR-Normal":
                    _, best_p, best_k = best_structure_mar_normal(X, max_p, max_K)
                else:
                    _, best_p, best_k = best_structure_mar_ged(X, max_p, max_K)

                if best_p and best_k:
                    st.success(f"Struktur terbaik: p = {best_p}, K = {best_k}")
                    st.session_state['selected_p'] = best_p
                    st.session_state['selected_k'] = best_k
                    st.session_state['model_type'] = model_choice
                else:
                    st.error("Gagal menentukan struktur terbaik.")

        if 'selected_p' in st.session_state and 'selected_k' in st.session_state:
            best_p = st.session_state['selected_p']
            best_k = st.session_state['selected_k']
            if st.button("📌 Latih Model dengan p & K Terbaik"):
                with st.spinner("Melatih model final..."):
                    if model_choice == "MAR-Normal":
                        model = em_mar_normal(X, best_p, best_k)
                        st.session_state['best_model'] = model
                        st.session_state['best_p'] = best_p
                        st.session_state['best_k'] = best_k
                        st.session_state['model_type'] = "MAR-Normal"
                        st.session_state['ar_params'] = model['ar_params']
                        st.session_state['sigmas'] = model['sigmas']
                        st.session_state['weights'] = model['weights']
                    else:
                        model = em_mar_ged(X, best_p, best_k)
                        st.session_state['best_model_ged'] = model
                        st.session_state['best_p_ged'] = best_p
                        st.session_state['best_k_ged'] = best_k
                        st.session_state['model_type'] = "MAR-GED"
                        st.session_state['ar_params'] = model['ar_params']
                        st.session_state['sigmas'] = model['sigmas']
                        st.session_state['weights'] = model['weights']
                    st.success(f"Model dilatih: p = {best_p}, K = {best_k}")

    elif metode_pemodelan == "Manual":
        p_manual = st.number_input("Masukkan ordo AR (p):", min_value=1, max_value=5, value=1)
        K_manual = st.number_input("Masukkan jumlah komponen (K):", min_value=1, max_value=5, value=2)

        if st.button("🧠 Latih Model Secara Manual"):
            with st.spinner("Melatih model dengan parameter manual..."):
                if model_choice == "MAR-Normal":
                    model = em_mar_normal(X, p_manual, K_manual)
                    st.session_state['best_model'] = model
                    st.session_state['best_p'] = p_manual
                    st.session_state['best_k'] = K_manual
                    st.session_state['model_type'] = "MAR-Normal"
                    st.session_state['ar_params'] = model['ar_params']
                    st.session_state['sigmas'] = model['sigmas']
                    st.session_state['weights'] = model['weights']
                else:
                    model = em_mar_ged(X, p_manual, K_manual)
                    st.session_state['best_model_ged'] = model
                    st.session_state['best_p_ged'] = p_manual
                    st.session_state['best_k_ged'] = K_manual
                    st.session_state['model_type'] = "MAR-GED"
                    st.session_state['ar_params'] = model['ar_params']
                    st.session_state['sigmas'] = model['sigmas']
                    st.session_state['weights'] = model['weights']
                st.success(f"Model dilatih: p = {p_manual}, K = {K_manual}")

# =================== PREDIKSI DAN VISUALISASI===========================
