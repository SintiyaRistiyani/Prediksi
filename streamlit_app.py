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
            betas[k] = 2.0  # tetap fixed (bisa dikembangkan jika perlu)

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

if menu == "Model":
    st.title("🏗️ Pemodelan Mixture Autoregressive (MAR)")

    if 'log_return' not in st.session_state:
        st.warning("Lakukan preprocessing terlebih dahulu.")
        st.stop()

    X = st.session_state['log_return'].values

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
                    else:
                        model = em_mar_ged(X, best_p, best_k)
                        st.session_state['best_model_ged'] = model
                        st.session_state['best_p_ged'] = best_p
                        st.session_state['best_k_ged'] = best_k
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
                else:
                    model = em_mar_ged(X, p_manual, K_manual)
                    st.session_state['best_model_ged'] = model
                    st.session_state['best_p_ged'] = p_manual
                    st.session_state['best_k_ged'] = K_manual
                    st.session_state['model_type'] = "MAR-GED"
                st.success(f"Model dilatih: p = {p_manual}, K = {K_manual}")

    st.markdown("### 📌 Ringkasan Parameter Model")
    if 'model_type' in st.session_state:
        if st.session_state['model_type'] == "MAR-Normal" and 'best_model' in st.session_state:
            model = st.session_state['best_model']
            st.write(f"**p**: {st.session_state['best_p']}, **K**: {st.session_state['best_k']}")
            st.write("**Weights:**", model['weights'])
            st.write("**AR Params:**", model['ar_params'])
            st.write("**Sigma (Variansi):**", model['sigmas'])
            st.write(f"**Log-Likelihood**: {model['log_likelihood']:.2f}")
            st.write(f"**BIC**: {model['bic']:.2f}")

        elif st.session_state['model_type'] == "MAR-GED" and 'best_model_ged' in st.session_state:
            model = st.session_state['best_model_ged']
            st.write(f"**p**: {st.session_state['best_p_ged']}, **K**: {st.session_state['best_k_ged']}")
            st.write("**Weights:**", model['weights'])
            st.write("**AR Params:**", model['ar_params'])
            st.write("**Sigma (Scale):**", model['sigmas'])
            st.write("**Beta (Shape):**", model['beta'])
            st.write(f"**Log-Likelihood**: {model['log_likelihood']:.2f}")
            st.write(f"**BIC**: {model['bic']:.2f}")

# ----------------- Halaman Uji Signifikansi dan Residual -----------------
if menu == "Uji Signifikansi dan Residual":
    st.title("📌 Uji Signifikansi Parameter & Asumsi Residual")

    if 'model_type' not in st.session_state:
        st.warning("Silakan latih model terlebih dahulu di halaman Model.")
        st.stop()

    X = st.session_state['log_return'].values
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

# ----------------- Halaman Prediksi dan Visualisasi -----------------
if menu == "Prediksi dan Visualisasi":
    st.title("📈 Prediksi dan Visualisasi Harga")

    if 'model_type' not in st.session_state or 'log_return' not in st.session_state or 'original_df' not in st.session_state:
        st.warning("Pastikan data sudah di-preprocessing dan model sudah dilatih.")
        st.stop()

    model_type = st.session_state['model_type']
    log_return = st.session_state['log_return']
    df = st.session_state['original_df']
    harga_terakhir = df.iloc[-1][st.session_state['selected_col']]

    n_steps = st.slider("Jumlah Hari Prediksi:", 5, 60, 30)

    if st.button("🔮 Prediksi"):
        with st.spinner("Melakukan prediksi..."):
            if model_type == "MAR-Normal":
                model = st.session_state['best_model']
                pred_log_return = predict_mar_normal(model, log_return.values, n_steps=n_steps)
            else:
                model = st.session_state['best_model_ged']
                pred_log_return = predict_mar_ged(model, log_return.values, n_steps=n_steps)

            pred_log_return = np.array(pred_log_return)
            harga_prediksi = [harga_terakhir]
            for r in pred_log_return:
                harga_prediksi.append(harga_prediksi[-1] * np.exp(r))
            harga_prediksi = harga_prediksi[1:]

            tanggal_mulai = df.index[-1]
            tanggal_prediksi = pd.date_range(start=tanggal_mulai + pd.Timedelta(days=1), periods=n_steps, freq='B')

            df_prediksi = pd.DataFrame({
                'Tanggal': tanggal_prediksi,
                'Prediksi Harga': harga_prediksi,
                'Log Return': pred_log_return
            })

            st.line_chart(df_prediksi.set_index('Tanggal')['Prediksi Harga'])
            st.dataframe(df_prediksi)

            csv = df_prediksi.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Unduh Hasil Prediksi", data=csv, file_name='prediksi_harga.csv', mime='text/csv')

# ----------------- Halaman Interpretasi dan Saran -----------------
if menu == "Interpretasi dan Saran":
    st.title("🧭 Interpretasi dan Saran")

    if 'model_type' not in st.session_state:
        st.warning("Model belum tersedia. Silakan latih model terlebih dahulu.")
        st.stop()

    st.subheader("📌 Interpretasi Model")
    model_type = st.session_state['model_type']
    st.markdown(f"Model yang digunakan: **{model_type}**")

    if model_type == "MAR-Normal":
        model = st.session_state['best_model']
        ar_params = model['ar_params']
        signif_ar = np.any(np.abs(ar_params) > 0.1)
        if signif_ar:
            st.write("Model menunjukkan adanya efek autoregresif yang cukup kuat pada beberapa komponen.")
        else:
            st.write("Mayoritas parameter AR tidak signifikan. Pergerakan harga bersifat acak atau didominasi noise.")
    else:
        model = st.session_state['best_model_ged']
        st.write("Model menggunakan distribusi Generalized Error (GED) yang cocok untuk data dengan kurtosis tinggi atau heavy-tail.")

    st.subheader("📈 Prediksi dan Risiko")
    st.markdown("Hasil prediksi dapat digunakan sebagai acuan pergerakan harga jangka pendek. Namun, tetap perlu waspada terhadap volatilitas dan ketidakpastian pasar.")

    st.subheader("🧠 Saran Penggunaan")
    st.markdown("""
    - Gunakan model ini untuk simulasi dan eksplorasi, bukan sebagai satu-satunya acuan keputusan investasi.
    - Perbarui model secara berkala untuk menangkap dinamika pasar terbaru.
    - Lakukan validasi out-of-sample untuk mengukur akurasi jangka panjang.
    - Kombinasikan dengan analisis teknikal/fundamental untuk hasil yang lebih komprehensif.
    """)
