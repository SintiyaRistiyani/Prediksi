import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from scipy.stats import gennorm, norm, kstest
from statsmodels.tools.tools import add_constant
from statsmodels.stats.diagnostic import acorr_ljungbox, het_white
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

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
    "Uji Signifikansi dan Residual", 
    "Prediksi dan Visualisasi", 
    "Interpretasi dan Saran"
))

# ----------------- Halaman Home -----------------
if menu == "Home":
    st.title("📈 Aplikasi Prediksi Harga Saham Menggunakan Model Mixture Autoregressive (MAR)")
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
        **Ketentuan**:
        1. File harus dalam bentuk **CSV**
        2. Data harus memiliki kolom **tanggal** dan **harga saham**
        3. Nilai harga sebaiknya berupa angka tanpa simbol (misal: tanpa `Rp`, `%`, atau pemisah ribuan)
    """)

    uploaded_file = st.file_uploader("📂 Upload File CSV (delimiter = ';')", type=["csv"])

    if uploaded_file:
        try:
            # Membaca file dengan delimiter ;
            df = pd.read_csv(uploaded_file, delimiter=';')
            df.columns = df.columns.str.strip()  # Bersihkan nama kolom

            st.markdown("### ✅ Pilih Kolom Harga")
            harga_col = st.selectbox("Pilih kolom harga saham:", df.columns)
            
            # Konversi harga ke numerik
            df[harga_col] = (
                df[harga_col]
                .astype(str)
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
                .str.replace("[^0-9.-]", "", regex=True)
            )
            df[harga_col] = pd.to_numeric(df[harga_col], errors="coerce")

            st.session_state['df'] = df
            st.session_state['selected_price_col'] = harga_col

            st.success("✅ Data berhasil dimuat dan kolom harga dikonversi ke numerik.")
            st.markdown("#### 📋 Preview Data")
            st.dataframe(df.head())

        except Exception as e:
            st.error(f"Gagal membaca atau memproses file: {e}")

# ----------------- Halaman Preprocessing -----------------
elif menu == "Data Preprocessing":
    st.title("⚙️ Preprocessing Data")

    if 'df' not in st.session_state or 'selected_price_col' not in st.session_state:
        st.warning("Mohon upload dan pilih kolom harga terlebih dahulu di halaman 'Input Data'.")
        st.stop()

    df = st.session_state['df'].copy()
    price_col = st.session_state['selected_price_col']

    # Konversi tanggal dan urutkan
    if 'tanggal' in df.columns:
        df['tanggal'] = pd.to_datetime(df['tanggal'], errors='coerce')
        df = df.dropna(subset=['tanggal'])
        df = df.sort_values(by='tanggal')
    else:
        st.error("Kolom 'tanggal' tidak ditemukan.")
        st.stop()

    df['log_return'] = np.log(df[price_col] / df[price_col].shift(1))
    df = df.dropna()

    # Simpan ke session_state
    st.session_state['preprocessed_df'] = df
    st.session_state['log_return'] = df['log_return']

       # ----------------- Split Data -----------------
    st.markdown("### ✂️ Split Data (Train/Test)")

    n_test = 30
    log_return = df['log_return']
    log_return_train = log_return[:-n_test]
    log_return_test = log_return[-n_test:]

    st.session_state['log_return_train'] = log_return_train
    st.session_state['log_return_test'] = log_return_test
    st.session_state['original_df'] = df.copy()

    st.info(f"Data telah dibagi: {len(log_return_train)} data untuk training, {len(log_return_test)} data untuk testing (30 hari terakhir).")

    st.line_chart({
        "Train": log_return_train,
        "Test": pd.Series(log_return_test, index=range(len(log_return_train), len(log_return_train) + len(log_return_test)))
    })

# ----------------- Halaman Stasioneritas -----------------
elif menu == "Uji Stasioneritas":
    st.title("📉 Uji Stasioneritas Log Return")

    if 'log_return_train' not in st.session_state:
        st.warning("Data train belum tersedia. Harap lakukan preprocessing terlebih dahulu.")
        st.stop()

    log_return_train = st.session_state['log_return_train']
    st.markdown("### 🧪 Uji ADF (Augmented Dickey-Fuller)")
    adf_result = adfuller(log_return_train)
    st.write(f"ADF Statistic: {adf_result[0]:.4f}")
    st.write(f"p-value: {adf_result[1]:.4f}")
    st.write("Kesimpulan:", "✅ Stasioner" if adf_result[1] < 0.05 else "❌ Tidak Stasioner")

    st.markdown("### 🔁 Plot ACF dan PACF (Train Only)")

    fig1, ax1 = plt.subplots()
    plot_acf(log_return_train, ax=ax1, lags=20)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    plot_pacf(log_return_train, ax=ax2, lags=20, method='ywm')
    st.pyplot(fig2)

    # Simpan hasil ke session_state
    st.session_state['stationary_return'] = log_return_train

# ----------------- Halaman Model -----------------
# --- Inisialisasi parameter MAR-Normal ---
def initialize_parameters_mar_normal(X, p, K):
    N = len(X)
    weights = np.full(K, 1 / K)
    km = KMeans(n_clusters=K, random_state=42).fit(X[p:].reshape(-1, 1))
    labels = km.labels_

    ar_params = np.zeros((K, p))
    sigmas = np.zeros(K)

    for k in range(K):
        idx = np.where(labels == k)[0]
        X_k = np.array([X[i - p:i] for i in idx + p if i >= p])
        y_k = X[idx + p][idx + p >= p]

        if len(y_k) == 0:
            ar_params[k] = np.zeros(p)
            sigmas[k] = np.std(X)
        else:
            beta_hat = np.linalg.lstsq(X_k, y_k, rcond=None)[0]
            residuals = y_k - X_k @ beta_hat
            ar_params[k] = beta_hat
            sigmas[k] = np.std(residuals) + 1e-6

    return weights, ar_params, sigmas

# --- EM untuk MAR-Normal ---
def em_mar_normal(X, p, K, max_iter=100, tol=1e-6):
    N = len(X)
    X_lagged = np.array([X[i - p:i] for i in range(p, N)])
    y = X[p:]

    weights, ar_params, sigmas = initialize_parameters_mar_normal(X, p, K)
    loglik_old = -np.inf

    for iteration in range(max_iter):
        pdfs = np.zeros((N - p, K))
        for k in range(K):
            mu = X_lagged @ ar_params[k]
            sigma = sigmas[k]
            pdfs[:, k] = weights[k] * (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((y - mu) / sigma) ** 2)

        responsibilities = pdfs / (pdfs.sum(axis=1, keepdims=True) + 1e-12)

        for k in range(K):
            r_k = responsibilities[:, k]
            N_k = r_k.sum()
            weights[k] = N_k / (N - p)

            W = np.diag(r_k)
            XWX = X_lagged.T @ W @ X_lagged
            XWy = X_lagged.T @ W @ y
            try:
                ar_params[k] = np.linalg.solve(XWX + 1e-6 * np.eye(p), XWy)
            except np.linalg.LinAlgError:
                ar_params[k] = np.zeros(p)

            mu_k = X_lagged @ ar_params[k]
            residuals = y - mu_k
            sigmas[k] = np.sqrt(np.sum(r_k * residuals ** 2) / N_k) + 1e-6

        loglik = np.sum(np.log(pdfs.sum(axis=1) + 1e-12))
        if abs(loglik - loglik_old) < tol:
            break
        loglik_old = loglik

    num_params = (K - 1) + K * (p + 1)
    bic = -2 * loglik + num_params * np.log(N - p)

    return {
        'weights': weights,
        'ar_params': ar_params,
        'sigmas': sigmas,
        'log_likelihood': loglik,
        'bic': bic
    }

# --- Cari struktur terbaik MAR-Normal ---
def best_structure_mar_normal(X, max_p=5, max_K=5):
    best_model = None
    best_bic = np.inf
    best_p, best_k = None, None

    for p in range(1, max_p + 1):
        for K in range(1, max_K + 1):
            try:
                model = em_mar_normal(X, p, K)
                if model['bic'] < best_bic:
                    best_bic = model['bic']
                    best_model = model
                    best_p = p
                    best_k = K
            except Exception:
                continue

    return best_model, best_p, best_k

# --- Inisialisasi parameter MAR-GED ---
def initialize_parameters_mar_ged(X, p, K):
    N = len(X)
    weights = np.full(K, 1 / K)
    km = KMeans(n_clusters=K, random_state=42).fit(X[p:].reshape(-1, 1))
    labels = km.labels_

    ar_params = np.zeros((K, p))
    sigmas = np.zeros(K)
    betas = np.full(K, 2.0)

    for k in range(K):
        idx = np.where(labels == k)[0]
        X_k = np.array([X[i - p:i] for i in idx + p if i >= p])
        y_k = X[idx + p][idx + p >= p]

        if len(y_k) == 0:
            ar_params[k] = np.zeros(p)
            sigmas[k] = np.std(X)
        else:
            beta_hat = np.linalg.lstsq(X_k, y_k, rcond=None)[0]
            residuals = y_k - X_k @ beta_hat
            ar_params[k] = beta_hat
            sigmas[k] = np.std(residuals) + 1e-6

    return weights, ar_params, sigmas, betas

# --- EM untuk MAR-GED ---
def em_mar_ged(X, p, K, max_iter=100, tol=1e-6):
    N = len(X)
    X_lagged = np.array([X[i - p:i] for i in range(p, N)])
    y = X[p:]

    weights, ar_params, sigmas, betas = initialize_parameters_mar_ged(X, p, K)
    loglik_old = -np.inf

    for iteration in range(max_iter):
        pdfs = np.zeros((N - p, K))
        for k in range(K):
            mu = X_lagged @ ar_params[k]
            scale = sigmas[k]
            beta = betas[k]
            pdfs[:, k] = weights[k] * gennorm.pdf(y, beta, loc=mu, scale=scale)

        responsibilities = pdfs / (pdfs.sum(axis=1, keepdims=True) + 1e-12)

        for k in range(K):
            r_k = responsibilities[:, k]
            N_k = r_k.sum()
            weights[k] = N_k / (N - p)

            W = np.diag(r_k)
            XWX = X_lagged.T @ W @ X_lagged
            XWy = X_lagged.T @ W @ y
            try:
                ar_params[k] = np.linalg.solve(XWX + 1e-6 * np.eye(p), XWy)
            except np.linalg.LinAlgError:
                ar_params[k] = np.zeros(p)

            mu_k = X_lagged @ ar_params[k]
            residuals = y - mu_k
            sigmas[k] = np.sqrt(np.sum(r_k * residuals ** 2) / N_k) + 1e-6
            betas[k] = 2.0

        loglik = np.sum(np.log(pdfs.sum(axis=1) + 1e-12))
        if abs(loglik - loglik_old) < tol:
            break
        loglik_old = loglik

    num_params = (K - 1) + K * (p + 2)
    bic = -2 * loglik + num_params * np.log(N - p)

    return {
        'weights': weights,
        'ar_params': ar_params,
        'sigmas': sigmas,
        'beta': betas,
        'log_likelihood': loglik,
        'bic': bic
    }

# --- Cari struktur terbaik MAR-GED ---
def best_structure_mar_ged(X, max_p=3, max_K=3):
    best_model = None
    best_bic = np.inf
    best_p, best_k = None, None

    for p in range(1, max_p + 1):
        for K in range(1, max_K + 1):
            try:
                model = em_mar_ged(X, p, K)
                if model['bic'] < best_bic:
                    best_bic = model['bic']
                    best_model = model
                    best_p = p
                    best_k = K
            except Exception:
                continue

    return best_model, best_p, best_k
    
def predict_mar_normal(model, X_init, n_steps=30):
    """
    Prediksi n_steps ke depan menggunakan komponen utama dari model MAR-Normal.
    """
    ar_params = model['ar_params']
    weights = model['weights']
    p = ar_params.shape[1]

    # Komponen dominan (komponen dengan bobot terbesar)
    main_k = np.argmax(weights)
    phi = ar_params[main_k]

    # Prediksi iteratif
    preds = []
    X_curr = list(X_init[-p:])  # ambil p nilai terakhir sebagai input awal

    for _ in range(n_steps):
        x_lag = np.array(X_curr[-p:])[::-1]  # urutan lag terbaru ke lama
        next_val = np.dot(phi, x_lag)
        preds.append(next_val)
        X_curr.append(next_val)

    return np.array(preds)

def predict_mar_ged(model, X_init, n_steps=30):
    """
    Prediksi n_steps ke depan menggunakan komponen utama dari model MAR-GED.
    """
    ar_params = model['ar_params']
    weights = model['weights']
    sigmas = model['sigmas']
    betas = model['beta']
    p = ar_params.shape[1]

    main_k = np.argmax(weights)
    phi = ar_params[main_k]
    sigma = sigmas[main_k]
    beta = betas[main_k]

    preds = []
    X_curr = list(X_init[-p:])

    np.random.seed(42)  # agar hasil reproducible

    for _ in range(n_steps):
        x_lag = np.array(X_curr[-p:])[::-1]  # gunakan urutan lag terbaru ke lama
        next_val = np.dot(phi, x_lag)
        noise = gennorm.rvs(beta, loc=0, scale=sigma)
        next_val += noise
        preds.append(next_val)
        X_curr.append(next_val)

    return np.array(preds)

if menu == "Model":
    st.title("🏗️ Pemodelan Mixture Autoregressive (MAR)")

    if 'log_return' not in st.session_state:
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

# ----------------- Halaman Uji Signifikansi dan Residual -----------------
if menu == "Uji Signifikansi dan Residual":
    st.title("📌 Uji Signifikansi Parameter & Asumsi Residual")

    if 'model_type' not in st.session_state:
        st.warning("Silakan latih model terlebih dahulu di halaman Model.")
        st.stop()

    X = st.session_state['log_return_train'].values
    model_type = st.session_state['model_type']

    if model_type == "MAR-Normal":
        model = st.session_state['best_model']
        p = st.session_state['best_p']
    else:
        model = st.session_state['best_model_ged']
        p = st.session_state['best_p_ged']

    ar_params = model['ar_params']
    sigmas = model['sigmas']
    weights = model['weights']
    K = len(weights)

    X_lagged = np.array([X[i - p:i] for i in range(p, len(X))])
    y = X[p:]

    st.markdown("### 🧪 Uji Signifikansi Parameter AR")

    signif_components = []
    signifikansi_data = []

    for k in range(K):
        phi = ar_params[k]
        sigma = sigmas[k]

        se = sigma / (np.sqrt(np.sum(X_lagged[:, -1] ** 2)) + 1e-6)
        z_score = phi[-1] / se
        p_value = 2 * (1 - norm.cdf(np.abs(z_score)))

        signif = p_value < 0.05
        if signif:
            signif_components.append(k)

        signifikansi_data.append({
            "Komponen": k + 1,
            "AR": round(phi[-1], 6),
            "SE": round(se, 6),
            "z-score": round(z_score, 4),
            "p-value": round(p_value, 4),
            "Status": "Signifikan ✅" if signif else "Tidak Signifikan ❌"
        })

    df_signif = pd.DataFrame(signifikansi_data)
    st.dataframe(df_signif)

    if len(signif_components) == 0:
        st.info("Tidak ada parameter signifikan. Uji residual tidak dilakukan.")
        st.stop()

    st.markdown("### 🔎 Uji Asumsi Residual (Hanya Komponen Signifikan)")

    residual_results = []

    for k in signif_components:
        phi = ar_params[k]
        mu = X_lagged @ phi
        residuals = y - mu

        lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
        lb_p = lb_test['lb_pvalue'].values[0]

        res_std = (residuals - residuals.mean()) / (residuals.std() + 1e-6)
        ks_stat, ks_p = kstest(res_std, 'norm')

        white_test = het_white(residuals, add_constant(X_lagged))
        white_p = white_test[1]

        residual_results.append({
            "Komponen": k + 1,
            "Ljung-Box p": round(lb_p, 4),
            "KS-test p": round(ks_p, 4),
            "White p": round(white_p, 4),
            "Autokorelasi": "❌ Ya" if lb_p < 0.05 else "✅ Tidak",
            "Normalitas": "❌ Tidak Normal" if ks_p < 0.05 else "✅ Normal",
            "Heteroskedastisitas": "❌ Ya" if white_p < 0.05 else "✅ Tidak"
        })

        st.subheader(f"📉 ACF Residual - Komponen {k + 1}")
        fig, ax = plt.subplots()
        plot_acf(residuals, ax=ax, lags=20)
        st.pyplot(fig)

    df_residual = pd.DataFrame(residual_results)
    st.dataframe(df_residual)

    # Simpan ke session_state
    st.session_state['uji_signif'] = df_signif
    st.session_state['uji_residual'] = df_residual

# ----------------- Halaman Prediksi dan Visualisasi -----------------
if menu == "Prediksi dan Visualisasi":
    st.title("📈 Prediksi dan Visualisasi Harga Saham")

    if 'model_type' not in st.session_state or 'log_return_train' not in st.session_state or 'original_df' not in st.session_state or 'selected_price_col' not in st.session_state:
        st.warning("Pastikan data sudah diproses dan model sudah dilatih.")
        st.stop()

    model_type = st.session_state['model_type']
    log_return_train = st.session_state['log_return_train']
    df = st.session_state['original_df']
    selected_price_col = st.session_state['selected_price_col']

    if selected_price_col not in df.columns:
        st.error("Kolom harga tidak ditemukan dalam data asli.")
        st.stop()

    harga_terakhir = df.iloc[-1][selected_price_col]

    n_steps = st.slider("🔢 Jumlah Hari Prediksi:", 5, 60, 30)

    if st.button("🔮 Prediksi"):
        with st.spinner("Melakukan prediksi..."):
            if model_type == "MAR-Normal":
                model = st.session_state['best_model']
                pred_log_return = predict_mar_normal(model, log_return_train.values, n_steps=n_steps)
            else:
                model = st.session_state['best_model_ged']
                pred_log_return = predict_mar_ged(model, log_return_train.values, n_steps=n_steps)

            # Hitung harga dari log return
            harga_prediksi = [harga_terakhir]
            for r in pred_log_return:
                harga_prediksi.append(harga_prediksi[-1] * np.exp(r))
            harga_prediksi = harga_prediksi[1:]  # buang harga awal

            tanggal_awal = df['Tanggal'].iloc[-1] if 'Tanggal' in df.columns else df.iloc[-1][0]
            tanggal_prediksi = pd.date_range(start=tanggal_awal + pd.Timedelta(days=1), periods=n_steps, freq='B')

            df_prediksi = pd.DataFrame({
                'Tanggal': tanggal_prediksi,
                'Log Return': pred_log_return_train,
                'Prediksi Harga': harga_prediksi
            })

            # Simpan ke session_state
            st.session_state['hasil_prediksi'] = df_prediksi
            st.session_state['n_steps_prediksi'] = n_steps
            st.session_state['pred_log_return_train'] = pred_log_return_train
            st.session_state['pred_harga'] = harga_prediksi

            # Visualisasi
            st.line_chart(df_prediksi.set_index('Tanggal')['Prediksi Harga'])

            # Tabel hasil
            st.dataframe(df_prediksi.style.format({
                'Log Return': '{:.6f}',
                'Prediksi Harga': 'Rp{:,.2f}'
            }))

            # Unduh CSV
            csv = df_prediksi.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Unduh Hasil Prediksi", data=csv, file_name='prediksi_harga.csv', mime='text/csv')
            
# ----------------- Halaman Interpretasi Hasil -----------------
if menu == "Interpretasi Hasil":
    st.title("🧠 Interpretasi Hasil Model & Prediksi")

    if 'model_type' not in st.session_state or 'hasil_prediksi' not in st.session_state:
        st.warning("Pastikan Anda sudah melakukan pelatihan dan prediksi model.")
        st.stop()

    model_type = st.session_state['model_type']
    st.subheader("📄 Ringkasan Model")

    if model_type == "MAR-Normal":
        model = st.session_state['best_model']
        p = st.session_state['best_p']
        K = st.session_state['best_k']
        st.write("**Jenis Model:** MAR-Normal")
    else:
        model = st.session_state['best_model_ged']
        p = st.session_state['best_p_ged']
        K = st.session_state['best_k_ged']
        st.write("**Jenis Model:** MAR-GED")

    st.markdown(f"""
    - **Ordo AR (p):** {p}
    - **Jumlah Komponen (K):** {K}
    - **Log-Likelihood:** {model['log_likelihood']:.2f}
    - **BIC:** {model['bic']:.2f}
    """)

    st.subheader("📌 Parameter Model")
    for k in range(K):
        st.markdown(f"**Komponen {k+1}**")
        st.write(f"**Bobot (π):** {model['weights'][k]:.4f}")
        st.write(f"**AR Coef (ϕ):** {model['ar_params'][k]}")
        st.write(f"**Sigma (σ):** {model['sigmas'][k]:.6f}")
        if model_type == "MAR-GED":
            st.write(f"**Beta (β):** {model['beta'][k]:.2f}")
        st.markdown("---")

    st.subheader("📈 Visualisasi Hasil Prediksi")
    df_prediksi = st.session_state['hasil_prediksi']
    st.line_chart(df_prediksi.set_index('Tanggal')['Prediksi Harga'])

    st.dataframe(df_prediksi.style.format({
        'Log Return': '{:.6f}',
        'Prediksi Harga': 'Rp{:,.2f}'
    }))

    st.subheader("🧠 Interpretasi Singkat")
    harga_awal = df_prediksi['Prediksi Harga'].iloc[0]
    harga_akhir = df_prediksi['Prediksi Harga'].iloc[-1]
    perubahan = ((harga_akhir - harga_awal) / harga_awal) * 100

    arah = "naik 📈" if perubahan > 0 else "turun 📉"
    st.success(f"Model memprediksi bahwa harga saham akan **{arah} sebesar {perubahan:.2f}%** dalam {len(df_prediksi)} hari ke depan.")
    st.subheader("🔍 Perbandingan Harga Aktual vs Prediksi")

    if 'original_df' in st.session_state and 'selected_price_col' in st.session_state:
        df_asli = st.session_state['original_df'].copy()
        kolom_harga = st.session_state['selected_price_col']

        # Ambil harga historis 60 hari terakhir (jika ada)
        df_asli['Tanggal'] = pd.to_datetime(df_asli['Tanggal'])
        df_asli = df_asli.sort_values('Tanggal')
        df_asli = df_asli[-60:] if len(df_asli) > 60 else df_asli

        # Gabungkan dengan prediksi
        df_prediksi['Tipe'] = 'Prediksi'
        df_asli_show = df_asli[['Tanggal', kolom_harga]].rename(columns={kolom_harga: 'Harga'})
        df_asli_show['Tipe'] = 'Aktual'

        df_pred_show = df_prediksi[['Tanggal', 'Prediksi Harga']].rename(columns={'Prediksi Harga': 'Harga'})

        df_gabung = pd.concat([df_asli_show, df_pred_show], ignore_index=True)

        import altair as alt

        chart = alt.Chart(df_gabung).mark_line(point=True).encode(
            x='Tanggal:T',
            y='Harga:Q',
            color=alt.Color('Tipe:N', scale=alt.Scale(scheme='category10')),
            tooltip=['Tanggal:T', 'Harga:Q', 'Tipe:N']
        ).properties(
            width=800,
            height=400,
            title="Harga Aktual vs Harga Prediksi"
        ).interactive()

        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("Data asli tidak tersedia untuk perbandingan.")

    # Tombol unduh hasil lengkap
    csv = df_prediksi.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Unduh Tabel Prediksi", data=csv, file_name='tabel_prediksi.csv', mime='text/csv')
