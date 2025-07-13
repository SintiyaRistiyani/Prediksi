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
from scipy.stats import norm
from numpy.linalg import LinAlgError
from scipy.stats import gennorm
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error


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


# Test Uji Signifikansi
from scipy.stats import norm
import pandas as pd
import numpy as np

def test_significance_ar_params_mar(X, y, phi, sigma, tau):
    """
    Uji signifikansi parameter AR untuk model MAR (Normal atau GED).
    Digunakan untuk evaluasi parameter phi.

    Parameters:
    - X: matriks lag (T x p)
    - y: vektor observasi (T,)
    - phi: matriks parameter AR (K x p)
    - sigma: vektor sigma tiap komponen (K,)
    - tau: probabilitas posterior (T x K)

    Output:
    - DataFrame berisi estimasi, standard error, z-value, p-value, dan keputusan.
    """
    K, p = phi.shape
    T = len(y)
    result = []

    for k in range(K):
        idx_k = tau[:, k] > 1e-3  # Hanya komponen aktif (probabilitas cukup besar)

        for j in range(p):
            Xj = X[:, j]
            nom = phi[k, j]
            denom = np.sum(tau[:, k] * Xj**2)

            if denom > 0:
                se = np.sqrt(sigma[k]**2 / denom)
                z = nom / se
                p_value = 2 * (1 - norm.cdf(np.abs(z)))

                result.append({
                    'Komponen': k+1,
                    'AR Index': f'phi_{j+1}',
                    'Estimate': nom,
                    'Std Error': se,
                    'z-value': z,
                    'p-value': p_value,
                    'Signifikan': '✅' if p_value < 0.05 else '❌'
                })

    return pd.DataFrame(result)


# ===================== FUNGSI PENDUKUNG MAR NORMAL=====================
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

# Untuk MAR-Normal
def em_mar_normal_manual(series, p, K, max_iter=100, tol=1e-6, seed=42):
    """
    EM Algorithm untuk MAR-Normal
    """
    np.random.seed(seed)
    n = len(series)
    y = series[p:]
    X = np.column_stack([series[p - i - 1: n - i - 1] for i in range(p)])
    T_eff = len(y)

    phi = np.random.randn(K, p) * 0.1
    sigma = np.random.rand(K) * 0.05 + 1e-3
    pi = np.ones(K) / K
    ll_old = -np.inf

    for iteration in range(max_iter):
        # E-Step
        log_tau = np.zeros((T_eff, K))
        for k in range(K):
            mu_k = X @ phi[k]
            log_pdf = norm.logpdf(y, loc=mu_k, scale=np.maximum(sigma[k], 1e-6))
            log_tau[:, k] = np.log(np.maximum(pi[k], 1e-8)) + log_pdf

        log_tau_max = np.max(log_tau, axis=1, keepdims=True)
        tau = np.exp(log_tau - log_tau_max)
        tau /= tau.sum(axis=1, keepdims=True)

        # M-Step
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

        pi = tau.mean(axis=0)
        pi = np.maximum(pi, 1e-8)
        pi /= pi.sum()

        ll_new = np.sum(np.log(np.sum(np.exp(log_tau - log_tau_max), axis=1)) + log_tau_max.flatten())

        if np.abs(ll_new - ll_old) < tol:
            break
        ll_old = ll_new

    # Hitung jumlah parameter untuk AIC/BIC
    num_params = K * (p + 1) + (K - 1)  # phi, sigma, pi
    aic = -2 * ll_new + 2 * num_params
    bic = -2 * ll_new + np.log(T_eff) * num_params

    return {
        'K': K,
        'phi': phi,
        'sigma': sigma,
        'pi': pi,
        'loglik': ll_new,
        'AIC': aic,
        'BIC': bic,
        'tau': tau,
        'X': X,
        'y': y
    }

# MAR-NORMAL
from scipy.stats import norm
import numpy as np
import pandas as pd

def test_significance_mar(result):
    """
    Uji signifikansi parameter MAR-Normal (phi, sigma, pi)
    """
    phi = result['phi']         # (K x p)
    sigma = result['sigma']     # (K,)
    pi = result['pi']           # (K,)
    tau = result['tau']         # (T x K)
    X = result['X']             # (T x p)
    y = result['y']             # (T,)

    K, p = phi.shape
    T_eff = len(y)

    sig_results = []

    for k in range(K):
        r_k = tau[:, k]
        W = np.diag(r_k)

        # --- Covariance phi ---
        try:
            XtWX = X.T @ W @ X
            XtWX += 1e-6 * np.eye(p)  # regularisasi numerik agar stabil
            cov_phi = sigma[k]**2 * np.linalg.inv(XtWX)
            se_phi = np.sqrt(np.diag(cov_phi))
        except np.linalg.LinAlgError:
            se_phi = np.full(p, np.nan)

        # --- Z-test untuk phi ---
        z_phi = phi[k] / se_phi
        pval_phi = 2 * (1 - norm.cdf(np.abs(z_phi)))

        # --- Standard error sigma ---
        se_sigma = sigma[k] / np.sqrt(2 * np.sum(r_k))
        z_sigma = sigma[k] / se_sigma
        pval_sigma = 2 * (1 - norm.cdf(np.abs(z_sigma)))

        # --- Standard error pi ---
        se_pi = np.sqrt(pi[k] * (1 - pi[k]) / T_eff)
        z_pi = pi[k] / se_pi
        pval_pi = 2 * (1 - norm.cdf(np.abs(z_pi)))

        # --- Simpan hasil untuk phi ---
        for j in range(p):
            sig_results.append({
                'Komponen': k + 1,
                'Parameter': f'phi_{j+1}',
                'Estimate': phi[k, j],
                'Std.Err': se_phi[j],
                'z-value': z_phi[j],
                'p-value': pval_phi[j],
                'Signifikan': '✅' if pval_phi[j] < 0.05 else '❌'
            })

        # --- Simpan hasil untuk sigma ---
        sig_results.append({
            'Komponen': k + 1,
            'Parameter': 'sigma',
            'Estimate': sigma[k],
            'Std.Err': se_sigma,
            'z-value': z_sigma,
            'p-value': pval_sigma,
            'Signifikan': '✅' if pval_sigma < 0.05 else '❌'
        })

        # --- Simpan hasil untuk pi ---
        sig_results.append({
            'Komponen': k + 1,
            'Parameter': 'pi',
            'Estimate': pi[k],
            'Std.Err': se_pi,
            'z-value': z_pi,
            'p-value': pval_pi,
            'Signifikan': '✅' if pval_pi < 0.05 else '❌'
        })

    return pd.DataFrame(sig_results)


# MAR-Normal
def find_best_K(series, p, K_range, max_iter=100, tol=1e-6):
    """
    Cari jumlah komponen K terbaik untuk MAR-Normal berdasarkan BIC
    """
    results = []
    for K in K_range:
        print(f"🔄 Estimasi MAR-Normal untuk K={K}...")
        model = em_mar_normal_manual(series, p, K, max_iter, tol)
        results.append(model)

    best_model = min(results, key=lambda x: x['BIC'])
    print(f"✅ Model terbaik: K={best_model['K']} (BIC={best_model['BIC']:.2f})")

    df_bic = pd.DataFrame({
        'K': [m['K'] for m in results],
        'LogLik': [m['loglik'] for m in results],
        'AIC': [m['AIC'] for m in results],
        'BIC': [m['BIC'] for m in results]
    })

    return best_model, df_bic


def find_best_K(series, p, K_range, max_iter=100, tol=1e-6):
    results = []
    for K in K_range:
        model = em_mar_normal_manual(series, p, K, max_iter, tol)
        results.append(model)
    best_model = min(results, key=lambda x: x['BIC'])
    df_bic = pd.DataFrame({
        'K': [m['K'] for m in results],
        'LogLik': [m['loglik'] for m in results],
        'AIC': [m['AIC'] for m in results],
        'BIC': [m['BIC'] for m in results]
    })
    return best_model, df_bic


# ==================== FUNGSI PENDUKUNG MAR GED ==================================
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

    num_params = K * (p + 2) + (K - 1)  # phi, sigma, beta, pi
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

# === GRID SEARCH UNTUK MENCARI K TERBAIK ===
def find_best_K_mar_ged(series, p, K_range, max_iter=100, tol=1e-6):
    results = []
    for K in K_range:
        print(f"🔄 Estimasi MAR-GED untuk K={K}...")
        model = em_mar_ged_manual(series, p, K, max_iter=max_iter, tol=tol)
        results.append(model)

    best_model = min(results, key=lambda x: x['BIC'])
    print(f"✅ Model terbaik: K={best_model['K']} (BIC={best_model['BIC']:.2f})")

    return best_model, pd.DataFrame({
        'K': [m['K'] for m in results],
        'LogLik': [m['loglik'] for m in results],
        'AIC': [m['AIC'] for m in results],
        'BIC': [m['BIC'] for m in results]
    })
# Signifikan
from scipy.stats import norm
import numpy as np
import pandas as pd

def test_significance_ar_params_mar_ged(X, y, phi, sigma, beta, tau):
    """
    Uji signifikansi parameter AR untuk model MAR-GED (phi, sigma, beta).
    Hanya untuk phi di fungsi ini, sigma dan beta bisa dibuat terpisah jika perlu.
    """
    K, p = phi.shape
    T = len(y)
    result = []

    for k in range(K):
        idx_k = tau[:, k] > 1e-3  # hanya gunakan data yang signifikan di komponen k

        for j in range(p):
            Xj = X[:, j]
            nom = phi[k, j]
            denom = np.sum(tau[:, k] * Xj**2)
            if denom > 0:
                # Standard error memperhitungkan scale dari GED
                se = np.sqrt(sigma[k]**2 / denom)
                z = nom / se
                p_value = 2 * (1 - norm.cdf(np.abs(z)))
                result.append({
                    'Komponen': k+1,
                    'Parameter': f'phi_{j+1}',
                    'Estimate': nom,
                    'Std.Error': se,
                    'z-value': z,
                    'p-value': p_value,
                    'Signifikan': '✅' if p_value < 0.05 else '❌'
                })

    return pd.DataFrame(result)
    
    
# Residual
from scipy.stats import kstest, norm, gennorm
from statsmodels.stats.diagnostic import acorr_ljungbox
import numpy as np
import pandas as pd

def compute_residuals_mar(model):
    """
    Hitung residual berdasarkan komponen dominan (argmax dari tau)
    """
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

def test_residual_assumptions_mar(model, lags=10, ged=False):
    """
    Uji asumsi residual MAR (Normal atau GED):
    - K-S test (Normal jika ged=False, GED jika ged=True)
    - Ljung-Box test (autokorelasi)
    """
    residuals = compute_residuals_mar(model)
    residuals_std = (residuals - np.mean(residuals)) / np.std(residuals)

    if ged:
        # Jika MAR-GED, gunakan rata-rata beta untuk CDF GED
        beta_avg = np.mean(model['beta'])
        ks_stat, ks_pval = kstest(residuals_std, lambda x: gennorm.cdf(x, beta_avg))
        hipotesis = 'Residual mengikuti distribusi GED'
        keputusan = 'Tolak H0 (Tidak GED)' if ks_pval < 0.05 else 'Gagal Tolak H0 (GED)'
    else:
        ks_stat, ks_pval = kstest(residuals_std, 'norm')
        hipotesis = 'Residual mengikuti distribusi normal'
        keputusan = 'Tolak H0 (Tidak Normal)' if ks_pval < 0.05 else 'Gagal Tolak H0 (Normal)'

    # Ljung-Box Test
    lb_result = acorr_ljungbox(residuals, lags=lags, return_df=True)
    lb_stat = lb_result['lb_stat'].values[-1]
    lb_pval = lb_result['lb_pvalue'].values[-1]
    keputusan_lb = 'Tolak H0 (Ada Autokorelasi)' if lb_pval < 0.05 else 'Gagal Tolak H0 (Tidak Ada Autokorelasi)'

    result = pd.DataFrame({
        'Test': ['Kolmogorov-Smirnov', 'Ljung-Box'],
        'Statistic': [ks_stat, lb_stat],
        'p-value': [ks_pval, lb_pval],
        'Hipotesis Nol (H0)': [hipotesis, 'Tidak ada autokorelasi residual'],
        'Keputusan': [keputusan, keputusan_lb]
    })

    return result, residuals


# FORECAST
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
#======== PENDUKUNG ===================
from scipy.stats import kstest, norm
from statsmodels.stats.diagnostic import acorr_ljungbox
import numpy as np
import pandas as pd

def compute_mar_residuals(result):
    """
    Hitung residual MAR-Normal berdasarkan komponen dominan tiap waktu t.
    """
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


def test_residual_assumptions(result, lags=10, alpha=0.05):
    """
    Uji asumsi residual MAR-Normal:
    - Normalitas (Kolmogorov-Smirnov)
    - Autokorelasi (Ljung-Box)
    """
    residuals = compute_mar_residuals(result)
    n = len(residuals)

    # Normalisasi residual sebelum K–S
    resid_std = (residuals - np.mean(residuals)) / np.std(residuals)

    # --- Kolmogorov-Smirnov Test (Normalitas) ---
    ks_stat, ks_pvalue = kstest(resid_std, 'norm')
    ks_decision = 'Tolak H0 (Tidak Normal)' if ks_pvalue < alpha else 'Gagal Tolak H0 (Normal)'

    # --- Ljung-Box Test (Autokorelasi) ---
    lb_result = acorr_ljungbox(residuals, lags=lags, return_df=True)
    lb_stat = lb_result['lb_stat'].values[-1]
    lb_pvalue = lb_result['lb_pvalue'].values[-1]
    lb_decision = 'Tolak H0 (Ada Autokorelasi)' if lb_pvalue < alpha else 'Gagal Tolak H0 (Tidak Ada Autokorelasi)'

    # Buat tabel hasil
    result_summary = pd.DataFrame({
        'Test': ['Kolmogorov-Smirnov', 'Ljung-Box'],
        'Statistic': [ks_stat, lb_stat],
        'p-value': [ks_pvalue, lb_pvalue],
        'Hipotesis Nol (H0)': ['Residual mengikuti distribusi normal', 'Tidak ada autokorelasi residual'],
        'Keputusan': [ks_decision, lb_decision]
    })

    return result_summary, residuals



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


# =================== UJI SIGNIFIKANDI DAN RESIDUAL===========================
elif menu == "Uji Signifikansi dan Residual":

    st.title("🧪 Uji Signifikansi Parameter & Diagnostik Residual")

    if 'best_model' not in st.session_state:
        st.warning("Lakukan pemodelan terlebih dahulu di menu 'Model'.")
        st.stop()

    model = st.session_state['best_model']
    model_choice = st.session_state['model_choice']

    st.header("📌 Uji Signifikansi Parameter")

    if model_choice == "MAR-Normal":
        st.markdown("##### Model: **MAR-Normal**")

        df_sig = test_significance_mar(model)

        st.dataframe(df_sig.style.format({"Estimate": "{:.4f}", "Std.Err": "{:.4f}", 
                                          "z-value": "{:.4f}", "p-value": "{:.4f}"}))

    elif model_choice == "MAR-GED":
        st.markdown("##### Model: **MAR-GED**")

        df_sig = test_significance_ar_params_mar(model['X'], model['y'], model['phi'], model['sigma'], model['tau'])

        st.dataframe(df_sig.style.format({"Estimate": "{:.4f}", "Std Error": "{:.4f}", 
                                          "z-value": "{:.4f}", "p-value": "{:.4f}"}))

    st.markdown("""
    **Interpretasi:**  
    - p-value < 0.05 → **Signifikan**  
    - p-value ≥ 0.05 → **Tidak signifikan**
    """)

    # ------------------ DIAGNOSTIK RESIDUAL -------------------
    st.header("📊 Diagnostik Residual")

    if model_choice == "MAR-Normal":
        st.markdown("##### Residual (Komponen Dominan) - MAR-Normal")

        result_summary, residuals = test_residual_assumptions(model)

        st.dataframe(result_summary.style.format({"Statistic": "{:.4f}", "p-value": "{:.4f}"}))

        # Plot residual waktu
        st.markdown("#### 🕒 Plot Residual Waktu")
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(residuals, label="Residual", color='purple')
        ax.axhline(0, linestyle='--', color='gray')
        ax.set_title("Plot Residual MAR-Normal (Komponen Dominan)")
        st.pyplot(fig)

        # Histogram
        st.markdown("#### 🔍 Histogram Residual")
        fig, ax = plt.subplots(figsize=(8,4))
        sns.histplot(residuals, kde=True, bins=30, color='skyblue', ax=ax)
        ax.set_title("Distribusi Residual MAR-Normal")
        st.pyplot(fig)

    elif model_choice == "MAR-GED":
        st.markdown("##### Residual (Komponen Dominan) - MAR-GED")

        result_summary, residuals = test_residual_assumptions_mar(model)

        st.dataframe(result_summary.style.format({"Statistic": "{:.4f}", "p-value": "{:.4f}"}))

        # Plot residual waktu
        st.markdown("#### 🕒 Plot Residual Waktu")
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(residuals, label="Residual", color='darkgreen')
        ax.axhline(0, linestyle='--', color='gray')
        ax.set_title("Plot Residual MAR-GED (Komponen Dominan)")
        st.pyplot(fig)

        # Histogram
        st.markdown("#### 🔍 Histogram Residual")
        fig, ax = plt.subplots(figsize=(8,4))
        sns.histplot(residuals, kde=True, bins=30, color='lightgreen', ax=ax)
        ax.set_title("Distribusi Residual MAR-GED")
        st.pyplot(fig)

===================================== PREDIKSI DAN VISUALISASI===============================================
# Fungsi Konversi Log-Return ke Harga
# ---------------------------
def convert_logreturn_to_price(last_price, log_returns):
    prices = [last_price]
    for r in log_returns:
        prices.append(prices[-1] * np.exp(r))
    return np.array(prices[1:])

# ---------------------------
# Fungsi Performa Harga
# ---------------------------
def compute_price_metrics(actual, pred):
    mape = mean_absolute_percentage_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    return mape, rmse, mae

# ---------------------------
# Halaman Prediksi
elif menu == "Prediksi dan Visualisasi":
    st.header("🔮 Prediksi Harga & Visualisasi")

    nama_saham = st.selectbox("Pilih Saham", list(log_return_train.columns))

    mode_prediksi = st.radio("Pilih Mode Prediksi", ['Out-of-Sample', 'Forecast Masa Depan'])
    show_as = st.radio("Tampilkan Dalam", ['Log-Return', 'Harga'])

    model = best_models[nama_saham]
    data_train_saham = log_return_train[nama_saham].dropna().values

if mode_prediksi == 'Out-of-Sample':
    st.subheader(f"📈 Out-of-Sample Prediction: {nama_saham}")

    y_test_actual = log_return_test[nama_saham].dropna().values
    y_test_pred = model['y_pred_outsample']  # Pastikan output ini ada di model kamu

    if show_as == 'Log-Return':
        # Visualisasi Log-Return
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(y_test_actual, label='Aktual', color='black')
        ax.plot(y_test_pred, label='Prediksi', color='blue', linestyle='--')
        ax.set_title(f'Aktual vs Prediksi Out-of-Sample (Log-Return) - {nama_saham}')
        ax.legend()
        st.pyplot(fig)

    else:
        # Visualisasi Harga
        first_test_idx = log_return_test.index[0]
        idx_loc = df.index.get_loc(first_test_idx)
        last_price = df.iloc[idx_loc - 1][nama_saham]

        actual_price = convert_logreturn_to_price(last_price, y_test_actual)
        pred_price = convert_logreturn_to_price(last_price, y_test_pred)

        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(actual_price, label='Harga Aktual', color='black')
        ax.plot(pred_price, label='Harga Prediksi', color='red', linestyle='--')
        ax.set_title(f'Out-of-Sample Harga: {nama_saham}')
        ax.legend()
        st.pyplot(fig)

        # Performa Harga
        mape, rmse, mae = compute_price_metrics(actual_price, pred_price)

        st.write("📊 **Tabel Performa Out-of-Sample (Harga)**")
        df_perf = pd.DataFrame({
            'MAPE (%)': [mape*100],
            'RMSE': [rmse],
            'MAE': [mae]
        })
        st.dataframe(df_perf)

    # Download CSV
    df_outsample = pd.DataFrame({
        'Aktual': y_test_actual if show_as == 'Log-Return' else actual_price,
        'Prediksi': y_test_pred if show_as == 'Log-Return' else pred_price
    })
    csv_out = df_outsample.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Out-of-Sample (CSV)", csv_out, file_name=f'out_sample_{nama_saham}.csv')

else:
    st.subheader(f"🔮 Forecast {nama_saham} 30 Langkah ke Depan")

    n_steps = 30
    pred_log = predict_mar_normal(model, data_train_saham, n_steps=n_steps)

    if show_as == 'Log-Return':
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(np.arange(len(data_train_saham)), data_train_saham, label='Data Historis', color='black')
        ax.plot(np.arange(len(data_train_saham), len(data_train_saham)+n_steps), pred_log,
                label='Forecast', color='red', linestyle='--')
        ax.set_title(f'Forecasting Log-Return: {nama_saham}')
        ax.legend()
        st.pyplot(fig)

    else:
        # Konversi ke Harga
        last_price = df.iloc[-1][nama_saham]
        pred_price = convert_logreturn_to_price(last_price, pred_log)

        harga_hist = df[nama_saham].dropna().values

        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(np.arange(len(harga_hist)), harga_hist, label='Data Historis', color='black')
        ax.plot(np.arange(len(harga_hist), len(harga_hist)+n_steps), pred_price,
                label='Forecast Harga', color='red', linestyle='--')
        ax.set_title(f'Forecasting Harga: {nama_saham}')
        ax.legend()
        st.pyplot(fig)

    # Download CSV
    df_forecast = pd.DataFrame({
        'Step': np.arange(1, n_steps+1),
        'Prediksi': pred_log if show_as == 'Log-Return' else pred_price
    })
    st.write("📊 **Tabel Forecasting**")
    st.dataframe(df_forecast)

    csv_forecast = df_forecast.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Forecast (CSV)", csv_forecast, file_name=f'forecast_{nama_saham}.csv')


