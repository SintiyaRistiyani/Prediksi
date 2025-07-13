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

elif menu == "Model":

    st.title("🏗️ Pemodelan Mixture Autoregressive (MAR)")

    if 'log_return_train' not in st.session_state:
        st.warning("⚠️ Silakan lakukan preprocessing terlebih dahulu.")
        st.stop()

    series = st.session_state['log_return_train']['Log Return'].values

    model_choice = st.selectbox("Pilih Jenis Distribusi Komponen:", ["MAR-Normal", "MAR-GED"])

    p_input = st.number_input("Masukkan orde AR (p):", min_value=1, max_value=5, value=1)
    k_max = st.slider("Pilih K maksimal (jumlah komponen):", min_value=2, max_value=5, value=3)

    # ===== Fungsi-fungsi pembantu EM MAR-Normal & MAR-GED =====
    import numpy as np

    def build_X(y, p):
        # buat matrix X dari y lag p
        T = len(y)
        X = np.column_stack([y[(p - i - 1):(T - i - 1)] for i in range(p)])
        return X

    def em_mar_normal_manual(y, p, K, max_iter=100, tol=1e-6):
        T = len(y)
        X = build_X(y, p)
        y_p = y[p:]
        n = len(y_p)

        # Initialize parameters
        np.random.seed(42)
        phi = np.random.randn(K, p) * 0.1
        sigma = np.ones(K) * np.std(y_p)
        pi = np.ones(K) / K

        tau = np.zeros((n, K))

        log_likelihood_old = -np.inf

        for iteration in range(max_iter):
            # E-step
            for k in range(K):
                mu = X @ phi[k]
                pdf = (1 / (np.sqrt(2 * np.pi) * sigma[k])) * np.exp(-0.5 * ((y_p - mu) / sigma[k]) ** 2)
                tau[:, k] = pi[k] * pdf
            tau_sum = tau.sum(axis=1, keepdims=True)
            tau = tau / tau_sum

            # M-step
            Nk = tau.sum(axis=0)
            pi = Nk / n
            for k in range(K):
                W = np.diag(tau[:, k])
                # Weighted least squares
                XtW = X.T @ W
                try:
                    phi[k] = np.linalg.solve(XtW @ X, XtW @ y_p)
                except np.linalg.LinAlgError:
                    phi[k] = np.linalg.lstsq(XtW @ X, XtW @ y_p, rcond=None)[0]
                mu = X @ phi[k]
                sigma[k] = np.sqrt(np.sum(tau[:, k] * (y_p - mu) ** 2) / Nk[k])

            # Log-likelihood
            ll = 0
            for k in range(K):
                mu = X @ phi[k]
                ll += pi[k] * (1 / (np.sqrt(2 * np.pi) * sigma[k])) * np.exp(-0.5 * ((y_p - mu) / sigma[k]) ** 2)
            log_likelihood = np.sum(np.log(ll + 1e-12))  # Add small constant untuk stabilitas

            if np.abs(log_likelihood - log_likelihood_old) < tol:
                break
            log_likelihood_old = log_likelihood

        # Info criteria
        n_params = K * (p + 2) - 1  # phi (p*K), sigma (K), pi (K-1)
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + n_params * np.log(n)

        return {
            'K': K,
            'phi': phi,
            'sigma': sigma,
            'pi': pi,
            'LogLik': log_likelihood,
            'AIC': aic,
            'BIC': bic,
            'tau': tau,
            'y': y_p,
            'X': X,
            'dist': 'normal',
            'p': p
        }

    def ged_pdf(x, mu, sigma, beta):
        c = beta / (2 * sigma * np.exp(np.log(2) / beta) * np.math.gamma(1 / beta))
        return c * np.exp(- (np.abs((x - mu) / sigma)) ** beta)

    def em_mar_ged_manual(y, p, K, max_iter=100, tol=1e-6):
        from scipy.special import gamma
        T = len(y)
        X = build_X(y, p)
        y_p = y[p:]
        n = len(y_p)

        # Initialize parameters
        np.random.seed(42)
        phi = np.random.randn(K, p) * 0.1
        sigma = np.ones(K) * np.std(y_p)
        beta = np.ones(K) * 1.5  # shape parameter GED, 1.5 awal
        pi = np.ones(K) / K

        tau = np.zeros((n, K))

        log_likelihood_old = -np.inf

        for iteration in range(max_iter):
            # E-step
            for k in range(K):
                mu = X @ phi[k]
                # pdf GED per titik
                c = beta[k] / (2 * sigma[k] * gamma(1 / beta[k]))
                pdf = c * np.exp(- (np.abs((y_p - mu) / sigma[k])) ** beta[k])
                tau[:, k] = pi[k] * pdf
            tau_sum = tau.sum(axis=1, keepdims=True)
            tau = tau / tau_sum

            # M-step
            Nk = tau.sum(axis=0)
            pi = Nk / n
            for k in range(K):
                W = np.diag(tau[:, k])
                # Weighted least squares
                XtW = X.T @ W
                try:
                    phi[k] = np.linalg.solve(XtW @ X, XtW @ y_p)
                except np.linalg.LinAlgError:
                    phi[k] = np.linalg.lstsq(XtW @ X, XtW @ y_p, rcond=None)[0]
                mu = X @ phi[k]
                # Update sigma numerically
                def sigma_obj(s):
                    if s <= 0:
                        return np.inf
                    res = tau[:, k] * (np.abs((y_p - mu) / s) ** beta[k])
                    return np.sum(res) / Nk[k] - gamma(1 / beta[k]) / gamma(3 / beta[k])
                res_sigma = minimize(sigma_obj, sigma[k], bounds=[(1e-6, None)])
                sigma[k] = res_sigma.x[0]

                # Update beta numerically
                def beta_obj(b):
                    if b <= 0:
                        return np.inf
                    val = -n * (np.log(b) - np.log(2) - np.log(sigma[k]) - np.log(gamma(1 / b))) + \
                          np.sum(tau[:, k] * (np.abs((y_p - mu) / sigma[k]) ** b) * np.log(np.abs((y_p - mu) / sigma[k]) + 1e-12))
                    return val
                res_beta = minimize(beta_obj, beta[k], bounds=[(0.1, 10)])
                beta[k] = res_beta.x[0]

            # Log-likelihood
            ll = 0
            for k in range(K):
                mu = X @ phi[k]
                c = beta[k] / (2 * sigma[k] * gamma(1 / beta[k]))
                pdf = c * np.exp(- (np.abs((y_p - mu) / sigma[k])) ** beta[k])
                ll += pi[k] * pdf
            log_likelihood = np.sum(np.log(ll + 1e-12))

            if np.abs(log_likelihood - log_likelihood_old) < tol:
                break
            log_likelihood_old = log_likelihood

        n_params = K * (p + 3) - 1  # phi(p*K), sigma(K), beta(K), pi(K-1)
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + n_params * np.log(n)

        return {
            'K': K,
            'phi': phi,
            'sigma': sigma,
            'beta': beta,
            'pi': pi,
            'LogLik': log_likelihood,
            'AIC': aic,
            'BIC': bic,
            'tau': tau,
            'y': y_p,
            'X': X,
            'dist': 'ged',
            'p': p
        }

    def find_best_K(series, p, K_range):
        results = []
        best_model = None
        best_bic = np.inf
        for K in K_range:
            try:
                model = em_mar_normal_manual(series, p, K)
                bic = model['BIC']
                results.append({'K': K, 'LogLik': model['LogLik'], 'AIC': model['AIC'], 'BIC': bic})
                if bic < best_bic:
                    best_bic = bic
                    best_model = model
            except Exception as e:
                st.error(f"Error estimasi MAR-Normal K={K}: {e}")
        df_bic = pd.DataFrame(results)
        return best_model, df_bic

    def find_best_K_mar_ged(series, p, K_range):
        results = []
        best_model = None
        best_bic = np.inf
        for K in K_range:
            try:
                model = em_mar_ged_manual(series, p, K)
                bic = model['BIC']
                results.append({'K': K, 'LogLik': model['LogLik'], 'AIC': model['AIC'], 'BIC': bic})
                if bic < best_bic:
                    best_bic = bic
                    best_model = model
            except Exception as e:
                st.error(f"Error estimasi MAR-GED K={K}: {e}")
        df_bic = pd.DataFrame(results)
        return best_model, df_bic

    # ===== Akhir fungsi pembantu =====

    if st.button("🔍 Cari K Terbaik & Estimasi Model"):
        with st.spinner("⏳ Menjalankan EM Algorithm & Pencarian K Terbaik..."):

            if model_choice == "MAR-Normal":
                best_model, df_bic = find_best_K(series, p_input, range(1, k_max + 1))
            else:
                best_model, df_bic = find_best_K_mar_ged(series, p_input, range(1, k_max + 1))

            if best_model:
                st.success(f"✅ Model {model_choice} terbaik ditemukan: K={best_model['K']} (BIC={best_model['BIC']:.2f})")

                st.markdown("### 📊 Tabel BIC")
                st.dataframe(df_bic.style.format({"LogLik": "{:.2f}", "AIC": "{:.2f}", "BIC": "{:.2f}"}))

                st.markdown(f"### 🔧 Parameter {model_choice}")
                phi = best_model['phi']
                param_data = []
                for k in range(best_model['K']):
                    row = {f"phi{j+1}": phi[k, j] for j in range(p_input)}
                    row.update({
                        "Komponen": k+1,
                        "sigma": best_model['sigma'][k],
                        "pi": best_model['pi'][k]
                    })
                    if model_choice == "MAR-GED":
                        row["beta"] = best_model['beta'][k]
                    param_data.append(row)
                st.dataframe(pd.DataFrame(param_data).round(4))

                # Simpan hasil model di session_state
                st.session_state['best_model'] = best_model
                st.session_state['model_choice'] = model_choice
                st.session_state['best_k'] = best_model['K']
                st.session_state['best_p'] = p_input
            else:
                st.error("❌ Tidak ada model yang berhasil diestimasi.")

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

    # Validasi data dan model
    required_keys = ['log_return_train', 'df', 'best_model', 'harga_col']
    for key in required_keys:
        if key not in st.session_state:
            st.error(f"❌ Data '{key}' belum tersedia. Silakan lakukan input, preprocessing, dan estimasi model terlebih dahulu.")
            st.stop()

    log_return_train = st.session_state['log_return_train']
    df = st.session_state['df']
    best_model = st.session_state['best_model']
    harga_col = st.session_state['harga_col']  # Nama saham yang dipilih user di halaman Input Data

    st.markdown(f"📌 **Saham yang Dipilih:** {harga_col}")

    n_steps = st.number_input("📅 Masukkan Jumlah Hari Prediksi:", min_value=1, max_value=90, value=30)
    show_as = st.radio("📊 Tampilkan Hasil Sebagai:", ['Log-Return', 'Harga'])

    if st.button("▶️ Prediksi"):
        # Ambil data historis log return saham terpilih
        if harga_col in log_return_train.columns:
            X_init = log_return_train[harga_col].dropna().values
        else:
            X_init = log_return_train.iloc[:, 0].dropna().values  # fallback ambil kolom pertama

        model = best_model  # Tidak pakai [harga_col] karena best_model sudah untuk 1 saham
        dist = model.get('dist', 'normal').lower()

        # Fungsi prediksi sesuai distribusi
        if dist == 'normal':
            preds_log = predict_mar_normal(model, X_init, n_steps=n_steps)
        elif dist == 'ged':
            preds_log = predict_mar_ged(model, X_init, n_steps=n_steps)
        else:
            st.error(f"❌ Distribusi model '{dist}' tidak dikenali.")
            st.stop()

        st.success(f"✅ Prediksi {n_steps} hari ke depan untuk {harga_col} selesai.")

        # Fungsi konversi log-return ke harga
        def logreturn_to_price(last_price, logreturns):
            prices = []
            current_price = last_price
            for lr in logreturns:
                next_price = current_price * np.exp(lr)
                prices.append(next_price)
                current_price = next_price
            return np.array(prices)

        if show_as == 'Harga':
            # Ambil harga terakhir aktual
            last_price = df.loc[df.index[-1], harga_col]
            preds_price = logreturn_to_price(last_price, preds_log)

            # Tampilkan tabel harga prediksi
            df_pred = pd.DataFrame({
                'Hari ke': np.arange(1, n_steps+1),
                'Harga Prediksi': preds_price
            })
            st.write(f"### 📋 Tabel Prediksi Harga Saham {harga_col}")
            st.dataframe(df_pred.style.format({"Harga Prediksi": "Rp {:,.2f}".format}))

            # Plot harga prediksi bersama harga historis
            fig, ax = plt.subplots(figsize=(12, 5))
            harga_hist = df[harga_col].dropna()
            ax.plot(harga_hist.index, harga_hist.values, label='Harga Historis', color='blue')
            future_idx = np.arange(harga_hist.index[-1]+1, harga_hist.index[-1]+n_steps+1)
            ax.plot(future_idx, preds_price, label='Harga Prediksi', linestyle='--', color='orange')
            ax.set_title(f"📈 Prediksi Harga Saham {harga_col}")
            ax.set_xlabel("Hari")
            ax.set_ylabel("Harga (Rupiah)")
            ax.legend()
            st.pyplot(fig)

        else:
            # Tampilkan tabel log-return prediksi
            df_pred = pd.DataFrame({
                'Hari ke': np.arange(1, n_steps+1),
                'Log-Return Prediksi': preds_log
            })
            st.write(f"### 📋 Tabel Prediksi Log-Return Saham {harga_col}")
            st.dataframe(df_pred.style.format({"Log-Return Prediksi": "{:.6f}"}))

            # Plot log-return prediksi bersama data historis
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(np.arange(len(X_init)), X_init, label='Log-Return Historis', color='green')
            future_idx = np.arange(len(X_init), len(X_init)+n_steps)
            ax.plot(future_idx, preds_log, label='Log-Return Prediksi', linestyle='--', color='red')
            ax.set_title(f"📈 Prediksi Log-Return Saham {harga_col}")
            ax.set_xlabel("Hari")
            ax.set_ylabel("Log-Return")
            ax.legend()
            st.pyplot(fig)
