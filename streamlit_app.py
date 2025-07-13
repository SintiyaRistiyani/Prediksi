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
    Prediksi n_steps ke depan untuk MAR-Normal atau MAR-GED.
    """
    series = series.copy()
    phi = model['phi']
    pi = model['pi']
    K = model['K']
    p = phi.shape[1]

    history = list(series[-p:])  # gunakan p terakhir untuk awal prediksi
    preds = []

    for _ in range(n_steps):
        pred_per_komponen = []
        for k in range(K):
            ar_part = np.dot(phi[k], history[-p:][::-1])  # lag p dibalik urutannya
            pred_per_komponen.append(pi[k] * ar_part)

        pred_value = np.sum(pred_per_komponen)
        preds.append(pred_value)
        history.append(pred_value)

    # Buat dataframe hasil
    dates = pd.date_range(start=pd.to_datetime("today").normalize(), periods=n_steps+1, freq='D')[1:]
    pred_df = pd.DataFrame({
        "Tanggal": dates,
        "Prediksi": preds,
        "Aktual": [np.nan]*n_steps,
        "Error": [np.nan]*n_steps
    })
    return pred_df

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
        st.warning("Upload data terlebih dahulu di menu Input Data.")
        st.stop()

    df = st.session_state['df'].copy()
    harga_col = st.session_state['harga_col']

    # --- Format harga ke Rupiah ---
    def format_harga_idr(x):
        return f"Rp {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    df['Harga Angka'] = df[harga_col]  # simpan harga numerik untuk perhitungan
    df['Harga Format Rupiah'] = df['Harga Angka'].apply(format_harga_idr)

    # --- Hitung Log Return ---
    df['Log Return'] = np.log(df['Harga Angka'] / df['Harga Angka'].shift(1))
    df = df.dropna().reset_index(drop=True)

    # --- Tampilkan 5 data pertama ---
    st.markdown("### 📋 Tabel 5 Data Pertama")
    st.dataframe(df[['Date', 'Harga Format Rupiah', 'Log Return']].head())

    # --- Split Data Train/Test ---
    st.markdown("### ✂️ Split Data (Train/Test)")
    n_test = 30
    train, test = df[:-n_test], df[-n_test:]

    st.session_state['log_return_train'] = train['Log Return']
    st.session_state['train'] = train
    st.session_state['test'] = test

    # --- Visualisasi Log Return Split ---
    st.markdown("#### Visualisasi Log Return (Train/Test)")
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(train['Date'], train['Log Return'], label='Train', color='blue')
    ax.plot(test['Date'], test['Log Return'], label='Test', color='orange')
    ax.set_title("Log Return: Train vs Test Split")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Log Return")
    ax.legend()
    st.pyplot(fig)

    
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

    # --- Uji Stasioneritas ADF ---
    st.markdown("### 🧪 Uji Stasioneritas ADF")

    from statsmodels.tsa.stattools import adfuller

    def adf_test(series, name):
        result = adfuller(series)
        st.write(f'**ADF Test untuk {name}:**')
        st.write(f'- ADF Statistic : {result[0]:.4f}')
        st.write(f'- p-value       : {result[1]:.4f}')
        st.write(f'- Stationary?   : {"✅ Ya" if result[1] < 0.05 else "⚠️ Tidak"}')
        st.write("---")

    adf_test(train['Log Return'], harga_col)
    
    # === Plot ACF & PACF ===
    st.markdown("### 🔁 ACF dan PACF Plot")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    plot_acf(train['Log Return'], ax=axes[0], lags=20)
    axes[0].set_title("ACF")

    plot_pacf(train['Log Return'], ax=axes[1], lags=20, method='ywm')
    axes[1].set_title("PACF")

    st.pyplot(fig)

    # Skewness dan Kurtosis
    skw = skew(train['Log Return'])
    krt = kurtosis(train['Log Return'])
    st.write(f"**Skewness:** {skw:.4f}")
    st.write(f"**Kurtosis:** {krt:.4f}")

    # Visualisasi histogram + KDE
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(train['Log Return'], kde=True, bins=30, color='skyblue', ax=ax)
    ax.set_title(f'Distribusi Log Return {harga_col}')
    ax.set_xlabel('Log Return')
    ax.set_ylabel('Frekuensi')
    st.pyplot(fig)

elif menu == "Model":

    st.title("🏗️ Pemodelan Mixture Autoregressive (MAR)")

    # Validasi data
    if 'log_return_train' not in st.session_state:
        st.warning("Lakukan preprocessing terlebih dahulu.")
        st.stop()

    series = st.session_state['log_return_train'].values

    # Pilihan model
    model_choice = st.selectbox("Pilih Jenis Distribusi Komponen:", 
                                ["MAR-Normal", "MAR-GED"])

    # Input p dan K_max
    p_input = st.number_input("Masukkan orde AR (p):", min_value=1, max_value=5, value=1)
    k_max = st.slider("Pilih K maksimal (jumlah komponen):", min_value=2, max_value=5, value=3)

    # Eksekusi pencarian model terbaik
    if st.button("🔍 Cari K Terbaik & Estimasi Model"):
        with st.spinner("Menjalankan proses EM dan pencarian K..."):

            if model_choice == "MAR-Normal":
                best_model, df_bic = find_best_K(series, p_input, range(1, k_max + 1))

                st.success(f"✅ Model MAR-Normal terbaik: K={best_model['K']} (BIC={best_model['BIC']:.2f})")

                st.markdown("### 📊 Tabel BIC (MAR-Normal)")
                st.dataframe(df_bic.style.format({"LogLik": "{:.2f}", "AIC": "{:.2f}", "BIC": "{:.2f}"}))

                # Parameter output
                phi = best_model['phi']
                sigma = best_model['sigma']
                pi = best_model['pi']

                param_data = []
                for k in range(best_model['K']):
                    row = {f"phi{j+1}": phi[k, j] for j in range(p_input)}
                    row.update({"Komponen": k+1, "sigma": sigma[k], "pi": pi[k]})
                    param_data.append(row)

                st.markdown("### 🔧 Parameter MAR-Normal")
                st.dataframe(pd.DataFrame(param_data).round(4))

            else:  # MAR-GED
                best_model, df_bic = find_best_K_mar_ged(series, p_input, range(1, k_max + 1))

                st.success(f"✅ Model MAR-GED terbaik: K={best_model['K']} (BIC={best_model['BIC']:.2f})")

                st.markdown("### 📊 Tabel BIC (MAR-GED)")
                st.dataframe(df_bic.style.format({"LogLik": "{:.2f}", "AIC": "{:.2f}", "BIC": "{:.2f}"}))

                # Parameter output
                phi = best_model['phi']
                sigma = best_model['sigma']
                beta = best_model['beta']
                pi = best_model['pi']

                param_data = []
                for k in range(best_model['K']):
                    row = {f"phi{j+1}": phi[k, j] for j in range(p_input)}
                    row.update({"Komponen": k+1, "sigma": sigma[k], "beta": beta[k], "pi": pi[k]})
                    param_data.append(row)

                st.markdown("### 🔧 Parameter MAR-GED")
                st.dataframe(pd.DataFrame(param_data).round(4))

            # Simpan ke session_state untuk keperluan selanjutnya (uji residual, prediksi, dll)
            st.session_state['best_model'] = best_model
            st.session_state['model_choice'] = model_choice
            st.session_state['best_k'] = best_model['K']
            st.session_state['best_p'] = p_input


# =================== PREDIKSI DAN VISUALISASI===========================
