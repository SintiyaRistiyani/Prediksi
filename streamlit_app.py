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
from scipy.stats import skew, kurtosis, shapiro, jarque_bera


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
# ===================== FUNGSI PENDUKUNG =====================
def parameter_significance(model):
    """
    Hitung t‑stat & p‑value sederhana (normal approximation) 
    untuk setiap koefisien phi.
    """
    phi   = model['phi']
    sigma = model['sigma']
    X     = model['X']
    T_eff = X.shape[0]
    p     = phi.shape[1]

    se_phi = []
    for k in range(model['K']):
        XtX_inv = np.linalg.inv(X.T @ X + 1e-6*np.eye(p))
        se = np.sqrt(sigma[k]**2 * np.diag(XtX_inv))
        se_phi.append(se)

    rows = []
    for k in range(model['K']):
        for j in range(p):
            t_stat = phi[k, j] / se_phi[k][j]
            p_val  = 2 * (1 - norm.cdf(abs(t_stat)))
            rows.append({"Komponen": k+1, "Phi_j": f"phi{j+1}", 
                         "Estimate": phi[k, j], "t": t_stat, "p‑value": p_val})
    return pd.DataFrame(rows)

def diag_residual(resid):

    st.subheader("🧪 Diagnostik Residual")
    # ACF plot
    fig, ax = plt.subplots()
    plot_acf(resid, lags=40, ax=ax)
    st.pyplot(fig)

    # Ljung‑Box
    lb = acorr_ljungbox(resid, lags=[10, 20], return_df=True)
    st.write("Ljung‑Box test:")
    st.dataframe(lb.round(4))

def forecast_mar(model, series, n_steps):
    """
    Prediksi n_steps ke depan untuk MAR‑Normal/GED sederhana.
    Hasil: DataFrame [Tanggal, Prediksi, Aktual, Error]
    """
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
    # Hitung ADF
    adf_result = check_stationarity(train['Log Return'])
    st.markdown("### 🔍 Hasil Uji ADF")
    st.write(f"**ADF Statistic:** {adf_result[0]:.4f}")
    st.write(f"**p-value:** {adf_result[1]:.4f}")
    st.write("**Kesimpulan:**", 
             "✅ Stasioner (p < 0.05)" if adf_result[1] < 0.05 else "⚠️ Tidak Stasioner (p ≥ 0.05)")

    # === Plot ACF & PACF ===

    st.markdown("### 🔁 ACF dan PACF Plot")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(train['Log Return'], ax=ax[0], lags=20)
    plot_pacf(train['Log Return'], ax=ax[1], lags=20, method='ywm')
    ax[0].set_title("ACF")
    ax[1].set_title("PACF")
    st.pyplot(fig)
        
    skw = skew(train['Log Return'])
    krt = kurtosis(train['Log Return'])
    st.write(f"**Skewness:** {skw:.4f}")
    st.write(f"**Kurtosis:** {krt:.4f}")

    
    # Visualisasi histogram + KDE
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(series, kde=True, bins=30, color='skyblue', ax=ax)
    ax.set_title(f'Distribusi Log Return {nama_saham}')
    ax.set_xlabel('Log Return')
    ax.set_ylabel('Frekuensi')
    st.pyplot(fig)

# ===================== MENU: Model =====================
elif menu == "Model":

    st.title("🏗️ Pemodelan Mixture Autoregressive (MAR)")

    # ------- pastikan data sudah ada ----------
    if 'log_return_train' not in st.session_state:
        st.warning("Lakukan preprocessing terlebih dahulu.")
        st.stop()

    series = st.session_state['log_return_train'].values

    # 1. user pilih jenis distribusi
    model_choice = st.selectbox("Jenis Distribusi Komponen:", 
                                ["MAR-Normal", "MAR-GED"])

    # 2. user masukkan orde AR (p) hasil observasi ACF‑PACF
    p_input = st.number_input("Orde AR (p) hasil analisis ACF/PACF:", 
                              min_value=1, max_value=5, value=1, step=1)

    # 3. batas maksimum K yang akan di‑grid‑search
    max_K = st.slider("Pencarian K maksimal:", 2, 5, 3)

    # 4. tombol eksekusi penuh
    if st.button("🔍 Cari K Terbaik & Estimasi Model"):
        with st.spinner("Menjalankan pemodelan ..."):

            # =====================================================
            # A) Pencarian K terbaik   ---------------------------
            # =====================================================
            if model_choice == "MAR-Normal":
                best_model, df_k = find_best_K(series, p_input, 
                                               range(1, max_K + 1))
            else:
                # fungsi find_best_K_ged Anda, konsep sama
                best_model, df_k = find_best_K_ged(series, p_input, 
                                                   range(1, max_K + 1))

            best_k = best_model["K"]
            st.success(f"K terbaik: {best_k} (BIC = {best_model['BIC']:.2f})")

            # Simpan hasil ke session_state
            st.session_state.update({
                'best_model' : best_model,
                'best_p'     : p_input,
                'best_k'     : best_k,
                'model_type' : model_choice,
                'ar_params'  : best_model['phi'],
                'sigmas'     : best_model['sigma'],
                'weights'    : best_model['pi']
            })

            # =====================================================
            # B) Uji signifikansi parameter   --------------------
            # =====================================================
            #  (contoh cepat: t‑stat = phi / se; se = sqrt(σ² * (X'X)⁻¹) )
            sig_df = parameter_significance(best_model)      # definisikan fungsi ini
            st.subheader("📊 Uji Signifikansi Parameter")
            st.dataframe(sig_df.style.format("{:.4f}"))

            # =====================================================
            # C) Diagnostik Residual -----------------------------
            # =====================================================
            resid = best_model['y'] - (best_model['X'] @ 
                                       best_model['phi'][np.argmax(best_model['pi'])])
            diag_residual(resid)     # fungsi: plot ACF resid, Ljung‑Box, JB, dsb.

            # =====================================================
            # D) Prediksi Out‑of‑Sample --------------------------
            # =====================================================
            n_forecast = st.number_input("Horizon Prediksi:", 1, 60, 30)
            if st.button("🚀 Prediksi Out‑of‑Sample"):
                pred_df = forecast_mar(best_model, series, n_forecast)
                st.subheader("📈 Prediksi vs Aktual")
                st.line_chart(pred_df.set_index("Tanggal"))

                mape = np.mean(np.abs(pred_df['Error'] / pred_df['Aktual'])) * 100
                st.write(f"MAPE: **{mape:.2f}%**")

# =================== PREDIKSI DAN VISUALISASI===========================
