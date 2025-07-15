import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.stats import gennorm, norm, kstest
from statsmodels.tools.tools import add_constant
from statsmodels.stats.diagnostic import acorr_ljungbox, het_white
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_percentage_error
from scipy.stats import skew, kurtosis, shapiro, jarque_bera
from numpy.linalg import LinAlgError
from scipy.optimize import minimize
from numpy.linalg import LinAlgError

# Utility
from io import StringIO

# === FUNGSI PENDUKUNG ===
# === ESTIMASI BETA GED ===
def estimate_beta(residuals, weights, sigma_init=1.0):
    def neg_log_likelihood(beta):
        if beta <= 0:
            return np.inf
        pdf_vals = gennorm.pdf(residuals, beta, loc=0, scale=sigma_init)
        logpdf = np.log(pdf_vals + 1e-12)
        return -np.sum(weights * logpdf)

    result = minimize(neg_log_likelihood, x0=np.array([2.0]), bounds=[(0.1, 10)])
    return result.x[0] if result.success else 2.0

# === EM ALGORITHM UNTUK MAR-GED ===
def em_mar_ged_manual(series, p, K, max_iter=100, tol=1e-6, seed=42):
    np.random.seed(seed)
    n = len(series)
    y = series[p:]
    X = np.column_stack([series[p - i - 1: n - i - 1] for i in range(p)])
    T_eff = len(y)

    phi = np.random.randn(K, p) * 0.1
    sigma = np.random.rand(K) * 0.05 + 1e-3
    beta = np.full(K, 2.0)
    pi = np.ones(K) / K
    ll_old = -np.inf

    for iteration in range(max_iter):
        log_tau = np.zeros((T_eff, K))
        for k in range(K):
            mu_k = X @ phi[k]
            log_pdf = gennorm.logpdf(y, beta[k], loc=mu_k, scale=np.maximum(sigma[k], 1e-6))
            log_tau[:, k] = np.log(np.maximum(pi[k], 1e-8)) + log_pdf

        log_tau_max = np.max(log_tau, axis=1, keepdims=True)
        tau = np.exp(log_tau - log_tau_max)
        tau /= tau.sum(axis=1, keepdims=True)

        for k in range(K):
            w = tau[:, k]
            W = np.diag(w)
            XtWX = X.T @ W @ X
            XtWy = X.T @ (w * y)
            try:
                XtWX += 1e-6 * np.eye(p)
                phi[k] = np.linalg.solve(XtWX, XtWy)
            except LinAlgError:
                phi[k] = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]

            mu_k = X @ phi[k]
            resid = y - mu_k
            sigma[k] = max(np.sqrt(np.sum(w * resid**2) / np.sum(w)), 1e-6)
            beta[k] = estimate_beta(resid, w, sigma_init=sigma[k])

        pi = tau.mean(axis=0)
        pi = np.maximum(pi, 1e-8)
        pi /= pi.sum()

        ll_new = np.sum(np.log(np.sum(np.exp(log_tau - log_tau_max), axis=1)) + log_tau_max.flatten())
        if np.abs(ll_new - ll_old) < tol:
            break
        ll_old = ll_new

    num_params = K * (p + 2) + (K - 1)
    aic = -2 * ll_new + 2 * num_params
    bic = -2 * ll_new + np.log(T_eff) * num_params

    return {
        'K': K,
        'phi': phi,
        'sigma': sigma,
        'pi': pi,
        'beta': beta,
        'loglik': ll_new,
        'AIC': aic,
        'BIC': bic,
        'tau': tau,
        'X': X,
        'y': y
    }

# === GRID SEARCH UNTUK K TERTENTU ===
def find_best_K_mar_ged(series, p, K_range, max_iter=100, tol=1e-6):
    results = []
    for K in K_range:
        model = em_mar_ged_manual(series, p, K, max_iter=max_iter, tol=tol)
        results.append(model)

    best_model = min(results, key=lambda x: x['BIC'])
    return best_model, pd.DataFrame({
        'K': [m['K'] for m in results],
        'LogLik': [m['loglik'] for m in results],
        'AIC': [m['AIC'] for m in results],
        'BIC': [m['BIC'] for m in results]
    })

# === TAMPILKAN PARAMETER ===
def show_mar_ged_params(model):
    p = model['phi'].shape[1]
    df = pd.DataFrame(model['phi'], columns=[f"phi{i+1}" for i in range(p)])
    df['sigma'] = model['sigma']
    df['beta'] = model['beta']
    df['pi'] = model['pi']
    st.dataframe(df.style.format("{:.4f}"))
    st.write(f"LogLik: {model['loglik']:.4f}, BIC: {model['BIC']:.2f}, AIC: {model['AIC']:.2f}")
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

# ---------------------------- Halaman Home ----------------------------------------------
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

# -------------------------- Halaman Preprocessing ----------------------------
elif menu == "Data Preprocessing":
    st.title("⚙️ Preprocessing Data")

    if 'df' not in st.session_state or 'harga_col' not in st.session_state:
        st.warning("Upload data terlebih dahulu di menu Input Data.")
        st.stop()

    df = st.session_state['df'].copy()
    harga_col = st.session_state['harga_col']

    # Format harga ke Rupiah
    def format_harga_idr(x):
        return f"Rp {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    df['Harga Angka'] = df[harga_col]
    df['Harga Format Rupiah'] = df['Harga Angka'].apply(format_harga_idr)

    # Hitung Log Return
    df['Log Return'] = np.log(df['Harga Angka'] / df['Harga Angka'].shift(1))
    df = df.dropna().reset_index(drop=True)

    # Split Train/Test
    n_test = 30
    train, test = df[:-n_test], df[-n_test:]

    # Simpan ke session_state
    st.session_state['df'] = df
    st.session_state['train'] = train
    st.session_state['test'] = test
    st.session_state['log_return_train'] = train[['Date', 'Log Return']]
    st.session_state['log_return_test'] = test[['Date', 'Log Return']]
    st.session_state['harga_col'] = harga_col

    # Tampilkan 5 data pertama
    st.markdown("### 📋 Tabel 5 Data Pertama")
    st.dataframe(df[['Date', 'Harga Format Rupiah', 'Log Return']].head())

    # Visualisasi Train/Test
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
    st.title("📉 Uji Stasioneritas & Diagnostik Distribusi")

    # Validasi data
    if 'log_return_train' not in st.session_state:
        st.warning("Silakan lakukan preprocessing terlebih dahulu.")
        st.stop()

    log_return_train = st.session_state['log_return_train']

    st.subheader("📌 Diagnostik Log Return")

    series = log_return_train['Log Return'].dropna()

    # === Uji Stasioneritas ADF ===
    st.markdown("#### 🧪 Uji Stasioneritas ADF")
    result = adfuller(series)
    st.write(f"- **ADF Statistic** : {result[0]:.4f}")
    st.write(f"- **p-value**       : {result[1]:.4f}")
    st.write(f"- **Stationary?**   : {'✅ Ya' if result[1] < 0.05 else '⚠️ Tidak'}")

    # === Uji Skewness & Kurtosis ===
    st.markdown("#### 📊 Skewness & Kurtosis")
    skw = skew(series)
    krt = kurtosis(series)
    st.write(f"- **Skewness** : {skw:.4f}")
    st.write(f"- **Kurtosis** : {krt:.4f}")

    # === Visualisasi Distribusi (Histogram + KDE) ===
    st.markdown("#### 📈 Distribusi Log Return")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(series, kde=True, bins=30, color='skyblue', ax=ax)
    ax.set_title('Distribusi Log Return')
    ax.set_xlabel('Log Return')
    ax.set_ylabel('Frekuensi')
    st.pyplot(fig)

    # === Plot ACF & PACF ===
    st.markdown("#### 🔁 Plot ACF & PACF")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(series, lags=20, ax=axes[0])
    axes[0].set_title('ACF')
    plot_pacf(series, lags=20, ax=axes[1], method='ywm')
    axes[1].set_title('PACF')
    st.pyplot(fig)

    # === Visualisasi Time Series Log Return ===
    st.markdown("#### 🕒 Plot Log Return")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(log_return_train['Date'], series, color='red')
    ax.set_title('Log Return Train')
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Log Return')
    ax.grid(True)
    st.pyplot(fig)

# ---------------------------------- Halaman Model (MAR‑GED) -----------------------------------------------
elif menu == "Model":
    st.title("⚙️ Estimasi Model MAR‑GED")

    if 'log_return_train' not in st.session_state:
        st.warning("Silakan lakukan preprocessing terlebih dahulu.")
        st.stop()

    train_df = st.session_state['log_return_train'].copy()
    series = train_df['Log Return'].values

    st.sidebar.subheader("🔧 Mode Estimasi")
    mode = st.sidebar.radio("Pilih Mode Estimasi:", ["Manual", "Cari Otomatis"], horizontal=True)

    if mode == "Manual":
        p = st.sidebar.number_input("Ordo AR (p)", min_value=1, max_value=10, value=2)
        K = st.sidebar.number_input("Jumlah Komponen (K)", min_value=2, max_value=6, value=2)
        max_iter = st.sidebar.slider("Maks Iterasi EM", 50, 500, 200, 50)

        if st.button("🚀 Estimasi MAR‑GED"):
            with st.spinner("Menjalankan estimasi EM..."):
                model = em_mar_ged_manual(series, p=p, K=K, max_iter=max_iter)
                st.session_state['mar_ged_model'] = model
                st.session_state['mar_ged_p'] = p
                st.session_state['mar_ged_K'] = K
                st.success("Estimasi selesai.")
                 # Sinkronisasi dengan halaman lain
                st.session_state['best_model'] = model
                st.session_state['model_choice'] = "MAR-GED"

        if 'mar_ged_model' in st.session_state:
            show_mar_ged_params(st.session_state['mar_ged_model'])

    else:
        p_max = st.sidebar.number_input("p Maks", min_value=1, max_value=10, value=5)
        K_range = range(2, 6)
        max_iter = st.sidebar.slider("Maks Iterasi EM", 50, 500, 200, 50)

        if st.button("🔍 Cari p & K Terbaik"):
            with st.spinner("Menjalankan pencarian grid..."):
                best_bic = np.inf
                best_model = None
                for p in range(1, p_max+1):
                    try:
                        model, summary = find_best_K_mar_ged(series, p=p, K_range=K_range, max_iter=max_iter)
                        if model['BIC'] < best_bic:
                            best_bic = model['BIC']
                            best_model = model
                            best_p = p
                    except Exception as e:
                        continue

                if best_model:
                    st.session_state['mar_ged_model'] = best_model
                    st.session_state['mar_ged_p'] = best_p
                    st.session_state['mar_ged_K'] = best_model['K']
                    st.success(f"Model terbaik ditemukan: p={best_p}, K={best_model['K']}, BIC={best_model['BIC']:.2f}")
                    show_mar_ged_params(best_model)
                else:
                    st.error("Gagal menemukan model yang konvergen.")


# ======================================== UJI SIGNIFIKANSI DAN RESIDUAL =======================================
elif menu == "Uji Signifikansi dan Residual":

    st.title("🧪 Uji Signifikansi Parameter & Diagnostik Residual")

    if 'best_model' not in st.session_state:
        st.warning("Lakukan pemodelan terlebih dahulu di menu 'Model'.")
        st.stop()

    model = st.session_state['best_model']
    model_choice = st.session_state['model_choice']

    st.header("📌 Uji Signifikansi Parameter")

    # Fungsi uji signifikansi MAR-Normal
    def test_significance_mar(result):
        from scipy.stats import norm

        phi = result['phi']         # (K x p)
        sigma = result['sigma']     # (K,)
        pi = result['pi']           # (K,)
        tau = result['tau']         # (T x K)
        X = result['X']             # (T x p)
        y = result['y']             # (T,)
        T_eff = len(y)

        K, p = phi.shape
        sig_results = []

        for k in range(K):
            r_k = tau[:, k]
            W = np.diag(r_k)

            try:
                XtWX = X.T @ W @ X
                XtWX += 1e-6 * np.eye(p)
                cov_phi = sigma[k]**2 * np.linalg.inv(XtWX)
                se_phi = np.sqrt(np.diag(cov_phi))
            except np.linalg.LinAlgError:
                se_phi = np.full(p, np.nan)

            z_phi = phi[k] / se_phi
            pval_phi = 2 * (1 - norm.cdf(np.abs(z_phi)))

            se_sigma = sigma[k] / np.sqrt(2 * np.sum(r_k))
            z_sigma = sigma[k] / se_sigma
            pval_sigma = 2 * (1 - norm.cdf(np.abs(z_sigma)))

            se_pi = np.sqrt(pi[k] * (1 - pi[k]) / T_eff)
            z_pi = pi[k] / se_pi
            pval_pi = 2 * (1 - norm.cdf(np.abs(z_pi)))

            for j in range(p):
                sig_results.append({
                    'Komponen': k + 1,
                    'Parameter': f'phi_{j+1}',
                    'Estimate': phi[k, j],
                    'Std.Err': se_phi[j],
                    'z-value': z_phi[j],
                    'p-value': pval_phi[j]
                })

            sig_results.append({
                'Komponen': k + 1,
                'Parameter': 'sigma',
                'Estimate': sigma[k],
                'Std.Err': se_sigma,
                'z-value': z_sigma,
                'p-value': pval_sigma
            })
            sig_results.append({
                'Komponen': k + 1,
                'Parameter': 'pi',
                'Estimate': pi[k],
                'Std.Err': se_pi,
                'z-value': z_pi,
                'p-value': pval_pi
            })

        return pd.DataFrame(sig_results)

    # Fungsi uji signifikansi MAR-GED (AR params saja)
    def test_significance_ar_params_mar(X, y, phi, sigma, tau):
        from scipy.stats import norm

        K, p = phi.shape
        result = []

        for k in range(K):
            for j in range(p):
                nom = phi[k, j]
                denom = np.sum(tau[:, k] * X[:, j]**2)
                if denom > 0:
                    se = np.sqrt(sigma[k]**2 / denom)
                    z = nom / se
                    p_value = 2 * (1 - norm.cdf(np.abs(z)))
                    result.append({
                        'Komponen': k + 1,
                        'AR Index': f'phi_{j+1}',
                        'Estimate': nom,
                        'Std Error': se,
                        'z-value': z,
                        'p-value': p_value,
                        'Signifikan': '✅' if p_value < 0.05 else '❌'
                    })

        return pd.DataFrame(result)

    # Fungsi hitung residual MAR-Normal
    def compute_mar_residuals(result):
        import numpy as np

        tau = result['tau']         # (T x K)
        X = result['X']             # (T x p)
        y = result['y']             # (T,)
        phi = result['phi']         # (K x p)

        dominant_comp = np.argmax(tau, axis=1)
        residuals = np.zeros(len(y))

        for t in range(len(y)):
            k = dominant_comp[t]
            residuals[t] = y[t] - X[t] @ phi[k]

        return residuals

    # Fungsi uji residual MAR-Normal
    def test_residual_assumptions(result, lags=10, alpha=0.05):
        from scipy.stats import kstest
        from statsmodels.stats.diagnostic import acorr_ljungbox

        residuals = compute_mar_residuals(result)
        resid_std = (residuals - np.mean(residuals)) / np.std(residuals)

        ks_stat, ks_pvalue = kstest(resid_std, 'norm')
        ks_decision = 'Tolak H0 (Tidak Normal)' if ks_pvalue < alpha else 'Gagal Tolak H0 (Normal)'

        lb_result = acorr_ljungbox(residuals, lags=lags, return_df=True)
        lb_stat = lb_result['lb_stat'].values[-1]
        lb_pvalue = lb_result['lb_pvalue'].values[-1]
        lb_decision = 'Tolak H0 (Ada Autokorelasi)' if lb_pvalue < alpha else 'Gagal Tolak H0 (Tidak Ada Autokorelasi)'

        result_summary = pd.DataFrame({
            'Test': ['Kolmogorov-Smirnov', 'Ljung-Box'],
            'Statistic': [ks_stat, lb_stat],
            'p-value': [ks_pvalue, lb_pvalue],
            'Hipotesis Nol (H0)': ['Residual mengikuti distribusi normal', 'Tidak ada autokorelasi residual'],
            'Keputusan': [ks_decision, lb_decision]
        })

        return result_summary, residuals

    # Fungsi hitung residual MAR-GED
    def compute_residuals_mar(model):
        import numpy as np

        tau = model['tau']
        X = model['X']
        y = model['y']
        phi = model['phi']

        dominant = np.argmax(tau, axis=1)
        residuals = np.zeros(len(y))
        for t in range(len(y)):
            k = dominant[t]
            y_pred = X[t] @ phi[k]
            residuals[t] = y[t] - y_pred

        return residuals

    # Fungsi uji residual MAR-GED
    def test_residual_assumptions_mar(model, lags=10):
        from scipy.stats import kstest
        from statsmodels.stats.diagnostic import acorr_ljungbox

        residuals = compute_residuals_mar(model)
        residuals_std = (residuals - np.mean(residuals)) / np.std(residuals)

        ks_stat, ks_pval = kstest(residuals_std, 'norm')

        lb_result = acorr_ljungbox(residuals, lags=lags, return_df=True)
        lb_stat = lb_result['lb_stat'].values[-1]
        lb_pval = lb_result['lb_pvalue'].values[-1]

        result = pd.DataFrame({
            'Test': ['Kolmogorov-Smirnov', 'Ljung-Box'],
            'Statistic': [ks_stat, lb_stat],
            'p-value': [ks_pval, lb_pval],
            'Hipotesis Nol (H0)': ['Residual mengikuti distribusi normal',
                                   'Tidak ada autokorelasi residual'],
            'Keputusan': ['Tolak H0 (Tidak Normal)' if ks_pval < 0.05 else 'Gagal Tolak H0 (Normal)',
                          'Tolak H0 (Ada Autokorelasi)' if lb_pval < 0.05 else 'Gagal Tolak H0 (Tidak Ada Autokorelasi)']
        })

        return result, residuals

    if model_choice == "MAR-Normal":
        st.markdown("##### Model: **MAR-Normal**")
        df_sig = test_significance_mar(model)
        st.dataframe(df_sig.style.format({
            "Estimate": "{:.4f}",
            "Std.Err": "{:.4f}",
            "z-value": "{:.4f}",
            "p-value": "{:.4f}"
        }))
    else:
        st.markdown("##### Model: **MAR-GED**")
        df_sig = test_significance_ar_params_mar(
            model['X'], model['y'], model['phi'], model['sigma'], model['tau'])
        st.dataframe(df_sig.style.format({
            "Estimate": "{:.4f}",
            "Std Error": "{:.4f}",
            "z-value": "{:.4f}",
            "p-value": "{:.4f}"
        }))

    st.markdown("""
    **Interpretasi:**  
    - p-value < 0.05 → **Signifikan**  
    - p-value ≥ 0.05 → **Tidak signifikan**
    """)

    st.header("📊 Diagnostik Residual")

    if model_choice == "MAR-Normal":
        st.markdown("##### Residual (Komponen Dominan) - MAR-Normal")
        result_summary, residuals = test_residual_assumptions(model)
        st.dataframe(result_summary.style.format({"Statistic": "{:.4f}", "p-value": "{:.4f}"}))
    else:
        st.markdown("##### Residual (Komponen Dominan) - MAR-GED")
        result_summary, residuals = test_residual_assumptions_mar(model)
        st.dataframe(result_summary.style.format({"Statistic": "{:.4f}", "p-value": "{:.4f}"}))

# ------------------------------- PREDIKSI DAN VISUALISASI ---------------------------------------------
elif menu == "Prediksi dan Visualisasi":
    st.header("🔮 Prediksi Harga Saham dengan Model MAR")

    # Fungsi prediksi MAR Normal (komponen dominan)
    def predict_mar_normal(model, X_init, n_steps=30):
        phi = model['phi']
        pi = model['pi']
        p = phi.shape[1]
        main_k = np.argmax(pi)
        phi_main = phi[main_k]
        preds = []
        X_curr = list(X_init[-p:])
        for _ in range(n_steps):
            x_lag = np.array(X_curr[-p:])[::-1]
            next_val = np.dot(phi_main, x_lag)
            preds.append(next_val)
            X_curr.append(next_val)
        return np.array(preds)

    # Fungsi prediksi MAR GED (komponen dominan)
    def predict_mar_ged(model, X_init, n_steps=30, seed=42):
        from scipy.stats import gennorm
        phi = model['phi']
        pi = model['pi']
        sigma = model['sigma']
        beta = model['beta']
        p = phi.shape[1]
        main_comp = np.argmax(pi)
        phi_main = phi[main_comp]
        sigma_main = sigma[main_comp]
        beta_main = beta[main_comp]
        np.random.seed(seed)
        preds = []
        X_curr = list(X_init[-p:])
        for _ in range(n_steps):
            next_val = np.dot(phi_main, X_curr[-p:][::-1])
            noise = gennorm.rvs(beta_main, loc=0, scale=sigma_main)
            next_val += noise
            preds.append(next_val)
            X_curr.append(next_val)
        return np.array(preds)

    # Validasi dan ambil data
    required_keys = ['log_return_train', 'df', 'best_model', 'harga_col']
    for key in required_keys:
        if key not in st.session_state:
            st.error(f"❌ Data '{key}' belum tersedia. Silakan lakukan input, preprocessing, dan estimasi model terlebih dahulu.")
            st.stop()

    log_return_train = st.session_state['log_return_train']
    df = st.session_state['df']
    best_model = st.session_state['best_model']
    harga_col = st.session_state['harga_col']

    st.markdown(f"📌 **Saham yang Dipilih:** {harga_col}")

    matched_col = 'Log Return'

    n_steps = st.number_input("📅 Masukkan Jumlah Hari Prediksi:", min_value=1, max_value=90, value=30)
    show_as = st.radio("📊 Tampilkan Hasil Sebagai:", ['Log-Return', 'Harga'])

    if st.button("▶️ Prediksi"):
        X_init = log_return_train[matched_col].dropna().values
        model = best_model

        dist = model.get('dist', 'normal').lower()

        if dist == 'normal':
            preds_log = predict_mar_normal(model, X_init, n_steps=n_steps)
        elif dist == 'ged':
            preds_log = predict_mar_ged(model, X_init, n_steps=n_steps)
        else:
            st.error(f"❌ Distribusi model '{dist}' tidak dikenali atau fungsi prediksi belum tersedia.")
            st.stop()

        st.success(f"✅ Prediksi {n_steps} hari ke depan untuk {matched_col} selesai.")

        if show_as == 'Harga':
            if harga_col in df.columns:
                last_price = df.loc[df.index[-1], harga_col]
            else:
                all_harga_cols = [col for col in df.columns if col != 'Date']
                if len(all_harga_cols) > 0:
                    last_price = df.loc[df.index[-1], all_harga_cols[0]]
                    st.warning(f"⚠️ Kolom harga '{harga_col}' tidak ditemukan di df. Menggunakan kolom {all_harga_cols[0]} sebagai harga terakhir.")
                else:
                    st.error("❌ Tidak ada kolom harga yang tersedia di df.")
                    st.stop()

            def convert_logreturn_to_price(last_price, logreturns):
                prices = [last_price]
                for r in logreturns:
                    prices.append(prices[-1] * np.exp(r))
                return np.array(prices[1:])

            preds_price = convert_logreturn_to_price(last_price, preds_log)

            df_pred = pd.DataFrame({
                'Hari ke': np.arange(1, n_steps + 1),
                'Harga Prediksi': preds_price
            })
            st.write(f"### 📋 Tabel Prediksi Harga Saham {harga_col}")
            st.dataframe(df_pred.style.format({"Harga Prediksi": "Rp {:,.2f}".format}))

            fig, ax = plt.subplots(figsize=(12, 5))
            harga_hist = df[harga_col].dropna() if harga_col in df.columns else df.iloc[:, 1].dropna()
            ax.plot(harga_hist.index, harga_hist.values, label='Harga Historis', color='blue')
            future_idx = np.arange(harga_hist.index[-1] + 1, harga_hist.index[-1] + n_steps + 1)
            ax.plot(future_idx, preds_price, label='Harga Prediksi', linestyle='--', color='orange')
            ax.set_title(f"📈 Prediksi Harga Saham {harga_col}")
            ax.set_xlabel("Hari")
            ax.set_ylabel("Harga (Rupiah)")
            ax.legend()
            st.pyplot(fig)

        else:  # show_as == 'Log-Return'
            df_pred = pd.DataFrame({
                'Hari ke': np.arange(1, n_steps + 1),
                'Log-Return Prediksi': preds_log
            })
            st.write(f"### 📋 Tabel Prediksi Log-Return Saham {harga_col}")
            st.dataframe(df_pred.style.format({"Log-Return Prediksi": "{:.6f}"}))

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(np.arange(len(X_init)), X_init, label='Log-Return Historis', color='green')
            future_idx = np.arange(len(X_init), len(X_init) + n_steps)
            ax.plot(future_idx, preds_log, label='Log-Return Prediksi', linestyle='--', color='red')
            ax.set_title(f"📈 Prediksi Log-Return Saham {harga_col}")
            ax.set_xlabel("Hari")
            ax.set_ylabel("Log-Return")
            ax.legend()
            st.pyplot(fig)
# ========================================= INTERPRETASI DAN SARAN ===========================================================
elif menu == "Interpretasi dan Saran":

    st.title("📈 Interpretasi Model dan Rekomendasi")

    if 'best_model' not in st.session_state or 'model_choice' not in st.session_state:
        st.warning("⚠️ Silakan lakukan estimasi model dulu di halaman Model.")
        st.stop()

    model = st.session_state['best_model']
    model_choice = st.session_state['model_choice']
    p = st.session_state.get('best_p', 1)
    K = st.session_state.get('best_k', model['K'])

    st.markdown(f"### Model Terpilih: {model_choice} dengan K={K} komponen dan orde AR(p)={p}")

    # Tampilkan parameter phi per komponen
    st.subheader("Parameter AR (phi) per Komponen")
    phi_df = pd.DataFrame(model['phi'], columns=[f'phi_{i+1}' for i in range(p)])
    phi_df.index = [f'Komponen {k+1}' for k in range(K)]
    st.dataframe(phi_df.round(4))

    # Tampilkan parameter sigma dan pi
    st.subheader("Parameter Varians (sigma) dan Proporsi Komponen (pi)")
    params_df = pd.DataFrame({
        'sigma': model['sigma'],
        'pi': model['pi']
    }, index=[f'Komponen {k+1}' for k in range(K)])
    if model_choice == "MAR-GED":
        params_df['beta'] = model['beta']
    st.dataframe(params_df.round(4))

    # Interpretasi sederhana
    st.markdown("### Interpretasi Singkat")
    st.markdown("""
    - **Parameter AR (phi)** menunjukkan kekuatan hubungan lag pada masing-masing komponen model.
    - **Sigma** menggambarkan volatilitas/residual standar pada komponen tersebut.
    - **Pi** adalah proporsi kontribusi setiap komponen dalam campuran.
    """)

    if model_choice == "MAR-GED":
        st.markdown("- **Beta** pada GED mengatur ketebalan ekor distribusi, nilai beta < 2 menunjukkan ekor yang lebih berat dibanding normal.")

    # Saran penggunaan model
    st.subheader("Saran Penggunaan Model")
    st.markdown("""
    - Gunakan model ini untuk memprediksi harga saham dengan mempertimbangkan adanya campuran beberapa proses autoregresif yang berbeda.
    - Jika model MAR-GED dipilih, model ini cocok untuk data dengan distribusi heavy-tailed atau outlier.
    - Selalu cek asumsi residual dan lakukan validasi model untuk memastikan performa prediksi yang baik.
    """)

    # Tombol untuk simpan hasil interpretasi sebagai file txt (optional)
    if st.button("💾 Simpan Interpretasi ke File"):
        interpretasi_text = f"""
        Model Terpilih: {model_choice} dengan K={K} komponen dan orde AR(p)={p}\n
        Parameter AR (phi):\n{phi_df.round(4).to_string()}\n
        Parameter Sigma dan Pi:\n{params_df.round(4).to_string()}\n
        """
        if model_choice == "MAR-GED":
            interpretasi_text += f"Beta:\n{params_df['beta'].round(4).to_string()}\n"

        interpretasi_text += """
        Interpretasi:
        - Parameter AR (phi) menunjukkan kekuatan hubungan lag pada masing-masing komponen model.
        - Sigma menggambarkan volatilitas/residual standar pada komponen tersebut.
        - Pi adalah proporsi kontribusi setiap komponen dalam campuran.
        """

        if model_choice == "MAR-GED":
            interpretasi_text += "- Beta pada GED mengatur ketebalan ekor distribusi, nilai beta < 2 menunjukkan ekor yang lebih berat dibanding normal.\n"

        interpretasi_text += """
        Saran:
        - Gunakan model ini untuk prediksi harga saham dengan campuran beberapa proses autoregresif.
        - Cek asumsi residual dan validasi model secara berkala.
        """

        with open("interpretasi_saran.txt", "w") as f:
            f.write(interpretasi_text)

        st.success("✅ File interpretasi_saran.txt berhasil disimpan di direktori aplikasi.")
