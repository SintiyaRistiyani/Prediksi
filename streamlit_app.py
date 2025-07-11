import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ML & Statistik
from sklearn.cluster import KMeans
from scipy.stats import gennorm
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
# ================= Fungsi-fungsi =================
# --- Fungsi Estimasi MAR-Normal ---
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

def best_structure_mar_normal(X, max_p=3, max_K=3):
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

# --- Fungsi Estimasi MAR-GED ---
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

# --- Fungsi Prediksi MAR-Normal ---
def predict_mar_normal(model, X_init, n_steps=30):
    ar_params = model['ar_params']
    weights = model['weights']
    p = ar_params.shape[1]
    main_k = np.argmax(weights)
    phi = ar_params[main_k]
    preds = []
    X_curr = list(X_init[-p:])
    for _ in range(n_steps):
        x_lag = np.array(X_curr[-p:])[::-1]
        next_val = np.dot(phi, x_lag)
        preds.append(next_val)
        X_curr.append(next_val)
    return np.array(preds)

# --- Fungsi Prediksi MAR-GED ---
def predict_mar_ged(model, X_init, n_steps=30):
    pred = []
    X_curr = list(X_init[-model['ar_params'].shape[1]:])
    main_comp = np.argmax(model['weights'])
    phi = model['ar_params'][main_comp]
    sigma = model['sigmas'][main_comp]
    beta = model['beta'][main_comp]
    np.random.seed(42)
    for _ in range(n_steps):
        next_val = np.dot(phi, X_curr[-len(phi):])
        noise = gennorm.rvs(beta, loc=0, scale=sigma)
        next_val += noise
        pred.append(next_val)
        X_curr.append(next_val)
    return np.array(pred)

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

    uploaded_file = st.file_uploader("Upload file CSV (delimiter = ';')", type=["csv"])

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
            st.dataframe(df.head())

        except Exception as e:
            st.error(f"Gagal membaca atau memproses file: {e}")

# ----------------- Halaman Data Preprocessing -----------------
elif menu == "Data Preprocessing":
    st.title("🧹 Data Preprocessing")

    if 'df' not in st.session_state or 'selected_price_col' not in st.session_state:
        st.warning("Silakan upload data terlebih dahulu di halaman Input Data.")
    else:
        df = st.session_state['df']
        selected_column = st.session_state['selected_price_col']

        # Pilih kolom tanggal
        date_cols = [col for col in df.columns if 'tgl' in col.lower() or 'date' in col.lower()]
        selected_date_col = st.selectbox("Pilih kolom tanggal:", date_cols)

        df[selected_date_col] = pd.to_datetime(df[selected_date_col], errors='coerce')
        df = df.dropna(subset=[selected_date_col])
        df = df.sort_values(by=selected_date_col).reset_index(drop=True)

        st.markdown(f"### 📈 Data Asli dari Kolom **{selected_column}**")
        df_plot = df[[selected_date_col, selected_column]].dropna().set_index(selected_date_col)
        st.line_chart(df_plot)

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

        df_clean = pd.to_numeric(df_clean, errors='coerce')
        st.write(f"Jumlah nilai kosong setelah: {df_clean.isnull().sum()}")

        # ----------------- Log Return -----------------
        st.markdown("### 3️⃣ Hitung Log Return")
        df['Log_Return'] = np.log(df[selected_column] / df[selected_column].shift(1))
        df = df.dropna(subset=['Log_Return'])
        st.session_state['log_return'] = df['Log_Return']
        st.session_state['df'] = df  # simpan update dataframe dengan log return

        st.dataframe(df[[selected_column, 'Log_Return']].head())

        # Visualisasi
        st.markdown("### 4️⃣ Visualisasi Data dan Log Return")

        log_return = st.session_state['log_return']

        # Gabungkan untuk visual
        df_viz = pd.DataFrame({
            'Tanggal': df[selected_date_col],
            'Harga': df_clean,
            'LogReturn': log_return
        }).dropna().set_index('Tanggal')

        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        ax[0].plot(df_viz.index, df_viz['Harga'], color='blue')
        ax[0].set_title("Harga Saham")

        ax[1].plot(df_viz.index, df_viz['LogReturn'], color='green')
        ax[1].set_title("Log Return")

        for axis in ax:
            axis.grid(True)
            axis.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        st.pyplot(fig)

# ----------------- Halaman Stasioneritas -----------------
elif menu == "Stasioneritas":
    st.title("📉 Uji Stasioneritas (ADF Test - Log Return)")

    if 'log_return' not in st.session_state or 'selected_price_col' not in st.session_state:
        st.warning("Silakan lakukan preprocessing terlebih dahulu agar log return tersedia.")
        st.stop()

    log_return = st.session_state['log_return']
    selected_col = st.session_state['selected_price_col']

    st.markdown(f"Kolom yang dianalisis: **{selected_col}**")
    st.markdown("Uji stasioneritas dilakukan terhadap **log return** dari data harga saham.")

    # Visualisasi log return
    st.markdown("### 🔍 Visualisasi Log Return")
    st.line_chart(log_return)

    # Siapkan dataframe log return untuk ADF test
    df_test = pd.DataFrame({selected_col: log_return.values}, index=log_return.index)

    from statsmodels.tsa.stattools import adfuller
    result = adfuller(df_test[selected_col].dropna())

    # Tampilkan hasil ADF
    st.markdown("### 📋 Hasil Uji ADF (Augmented Dickey-Fuller)")
    st.markdown(f"""
    - **ADF Statistic**: `{result[0]:.4f}`  
    - **p-value**: `{result[1]:.4f}`
    - **Lags Used**: `{result[2]}`
    - **Jumlah Observasi Efektif**: `{result[3]}`
    """)

    st.markdown("**Critical Values:**")
    for key, value in result[4].items():
        st.markdown(f"- `{key}`: `{value:.4f}`")

    # Interpretasi
    if result[1] < 0.05:
        st.success("✅ Log return **stasioner** (p-value < 0.05 → tolak H0: tidak ada akar unit).")
    else:
        st.error("❌ Log return **tidak stasioner** (p-value ≥ 0.05 → gagal tolak H0: ada akar unit).")

# ----------------- Halaman Model -----------------
# ================= Model ===================
if menu == "Model":
    st.title("🏗️ Training Model MAR")

    if 'log_return' not in st.session_state:
        st.warning("Lakukan preprocessing terlebih dahulu.")
        st.stop()

    X = st.session_state['log_return'].values
    model_choice = st.selectbox("Pilih Model:", ["MAR-Normal", "MAR-GED"])
    max_p = st.slider("Max order p:", 1, 5, 3)
    max_K = st.slider("Max jumlah komponen K:", 1, 5, 3)

    if st.button("Jalankan Training"):
        with st.spinner("Sedang melatih model..."):
            if model_choice == "MAR-Normal":
                model, best_p, best_k = best_structure_mar_normal(X, max_p, max_K)
                if model is not None:
                    st.success(f"Training selesai: p={best_p}, K={best_k}, BIC={model['bic']:.2f}")
                    st.session_state['best_model'] = model
                    st.session_state['best_p'] = best_p
                    st.session_state['best_k'] = best_k
                    st.session_state['model_type'] = "MAR-Normal"
                else:
                    st.error("Training gagal, coba parameter lain.")

            else:  # MAR-GED
                model, best_p, best_k = best_structure_mar_ged(X, max_p, max_K)
                if model is not None:
                    st.success(f"Training selesai: p={best_p}, K={best_k}, BIC={model['bic']:.2f}")
                    st.session_state['best_model_ged'] = model
                    st.session_state['best_p_ged'] = best_p
                    st.session_state['best_k_ged'] = best_k
                    st.session_state['model_type'] = "MAR-GED"
                else:
                    st.error("Training gagal, coba parameter lain.")

# ================= Prediksi dan Visualisasi ===================
elif menu == "Prediksi dan Visualisasi":
    st.title("🔮 Prediksi dan Visualisasi")

    if 'model_type' not in st.session_state or 'log_return' not in st.session_state or 'df' not in st.session_state:
        st.warning("Model atau data belum siap. Pastikan sudah training dan preprocessing.")
        st.stop()

    model_type = st.session_state['model_type']
    log_return = st.session_state['log_return'].dropna()
    df = st.session_state['df']
    selected_col = st.session_state['selected_price_col']

    st.write(f"Model yang digunakan: **{model_type}**")

    n_steps = st.number_input("Jumlah langkah prediksi ke depan (hari):", min_value=1, max_value=365, value=30)

    if st.button("Prediksi"):
        with st.spinner("Memproses prediksi..."):
            if model_type == "MAR-Normal":
                if 'best_model' not in st.session_state:
                    st.error("Model MAR-Normal belum tersedia.")
                    st.stop()
                model = st.session_state['best_model']
                pred_log_return = predict_mar_normal(model, log_return.values, n_steps=n_steps)

            elif model_type == "MAR-GED":
                if 'best_model_ged' not in st.session_state:
                    st.error("Model MAR-GED belum tersedia.")
                    st.stop()
                model = st.session_state['best_model_ged']
                pred_log_return = predict_mar_ged(model, log_return.values, n_steps=n_steps)

            else:
                st.error("Model tidak dikenali.")
                st.stop()

            # Konversi log return ke harga
            last_price = df[selected_col].dropna().iloc[-1]
            future_prices = [last_price]
            for r in pred_log_return:
                future_prices.append(future_prices[-1] * np.exp(r))
            future_prices = future_prices[1:]

            last_date = df.index[-1]
            future_dates = pd.date_range(start=last_date, periods=n_steps + 1, freq='D')[1:]

            df_future = pd.DataFrame({'Tanggal': future_dates, 'Prediksi Harga': future_prices}).set_index('Tanggal')

            st.success("Prediksi selesai!")
            st.line_chart(df_future)
            st.dataframe(df_future)

            st.session_state['prediksi_harga'] = df_future

# ================= Interpretasi dan Saran ===================
elif menu == "Interpretasi dan Saran":
    st.title("📌 Interpretasi dan Saran")
    st.write("Isi interpretasi dan saran di sini sesuai kebutuhan.")
