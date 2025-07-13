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

# ----------------- Halaman Preprocessing -----------------
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
    if 'log_return_train' not in st.session_state or 'harga_cols' not in st.session_state:
        st.warning("Silakan lakukan preprocessing terlebih dahulu.")
        st.stop()

    log_return_train = st.session_state['log_return_train']
    harga_cols = st.session_state['harga_cols']

    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from scipy.stats import skew, kurtosis
    import seaborn as sns

    # Loop setiap saham
    for col in harga_cols:
        st.subheader(f"📌 Diagnostik untuk {col}")

        series = log_return_train[col].dropna()

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
        ax.set_title(f'Distribusi Log Return {col}')
        ax.set_xlabel('Log Return')
        ax.set_ylabel('Frekuensi')
        st.pyplot(fig)

        # === Plot ACF & PACF ===
        st.markdown("#### 🔁 Plot ACF & PACF")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        plot_acf(series, lags=20, ax=axes[0])
        axes[0].set_title(f'ACF - {col}')
        plot_pacf(series, lags=20, ax=axes[1], method='ywm')
        axes[1].set_title(f'PACF - {col}')
        st.pyplot(fig)

        # === Visualisasi Time Series Log Return ===
        st.markdown("#### 🕒 Plot Log Return")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(series, color='red')
        ax.set_title(f'Log Return Train - {col}')
        ax.set_xlabel('Tanggal')
        ax.set_ylabel('Log Return')
        ax.grid(True)
        st.pyplot(fig)

        st.markdown("---")  # Pemisah antar saham

# ==================================== HALAMAN MODEL =======================================================
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

            # Jika multi saham, bisa buat dict
            if 'best_models' not in st.session_state:
                st.session_state['best_models'] = {}

            st.session_state['best_models']['Saham'] = best_model  # atau ganti 'Saham' dengan nama kolom jika multi

# ============================== UJI SIGNIFIKANDI DAN RESIDUAL===================================================
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

# ============================ HALAMAN PREDIKSI DAN VISUALISASI ============================

if menu == "Prediksi dan Visualisasi":
    st.header("🔮 Prediksi Harga & Visualisasi")

    # Cek apakah data sudah tersedia di session state
    required_keys = ['log_return_train', 'test', 'df', 'best_models']
    for key in required_keys:
        if key not in st.session_state:
            st.error(f"❌ Data {key} belum tersedia. Silakan lakukan preprocessing dan estimasi model terlebih dahulu.")
            st.stop()

    # Ambil data dari session state
    log_return_train = st.session_state['log_return_train']
    log_return_test = st.session_state['test']['Log Return']
    df = st.session_state['df']
    best_models = st.session_state['best_models']

    nama_saham = st.selectbox("Pilih Saham", list(log_return_train.columns))

    mode_prediksi = st.radio("Pilih Mode Prediksi", ['Out-of-Sample', 'Forecast Masa Depan'])
    show_as = st.radio("Tampilkan Dalam", ['Log-Return', 'Harga'])

    model = best_models[nama_saham]
    data_train_saham = log_return_train[nama_saham].dropna().values

    if mode_prediksi == 'Out-of-Sample':
        st.subheader(f"📈 Out-of-Sample Prediction: {nama_saham}")

        y_test_actual = log_return_test[nama_saham].dropna().values
        y_test_pred = model['y_pred_outsample']  # Pastikan output ini sudah disimpan saat estimasi

        if show_as == 'Log-Return':
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(y_test_actual, label='Aktual', color='black')
            ax.plot(y_test_pred, label='Prediksi', color='blue', linestyle='--')
            ax.set_title(f'Aktual vs Prediksi Out-of-Sample (Log-Return) - {nama_saham}')
            ax.legend()
            st.pyplot(fig)

        else:
            first_test_idx = st.session_state['test'].index[0]
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

            # Hitung MAPE, RMSE, MAE
            mape, rmse, mae = compute_price_metrics(actual_price, pred_price)

            st.write("📊 **Tabel Performa Out-of-Sample (Harga)**")
            df_perf = pd.DataFrame({
                'MAPE (%)': [mape*100],
                'RMSE': [rmse],
                'MAE': [mae]
            })
            st.dataframe(df_perf)

        # Download CSV Out-of-Sample
        df_outsample = pd.DataFrame({
            'Aktual': y_test_actual if show_as == 'Log-Return' else actual_price,
            'Prediksi': y_test_pred if show_as == 'Log-Return' else pred_price
        })
        csv_out = df_outsample.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Out-of-Sample (CSV)", csv_out, file_name=f'out_sample_{nama_saham}.csv')

    else:
        st.subheader(f"🔮 Forecast {nama_saham} 30 Langkah ke Depan")

        n_steps = 30

        # Pilih fungsi prediksi sesuai distribusi
        if model['dist'] == 'normal':
            pred_log = predict_mar_normal(model, data_train_saham, n_steps=n_steps)
        elif model['dist'] == 'ged':
            pred_log = predict_mar_ged(model, data_train_saham, n_steps=n_steps)
        else:
            st.error("Distribusi model tidak dikenali.")
            st.stop()

        if show_as == 'Log-Return':
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(np.arange(len(data_train_saham)), data_train_saham, label='Data Historis', color='black')
            ax.plot(np.arange(len(data_train_saham), len(data_train_saham)+n_steps), pred_log,
                    label='Forecast', color='red', linestyle='--')
            ax.set_title(f'Forecasting Log-Return: {nama_saham}')
            ax.legend()
            st.pyplot(fig)

        else:
            last_price = df[nama_saham].dropna().values[-1]
            pred_price = convert_logreturn_to_price(last_price, pred_log)

            harga_hist = df[nama_saham].dropna().values

            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(np.arange(len(harga_hist)), harga_hist, label='Data Historis', color='black')
            ax.plot(np.arange(len(harga_hist), len(harga_hist)+n_steps), pred_price,
                    label='Forecast Harga', color='red', linestyle='--')
            ax.set_title(f'Forecasting Harga: {nama_saham}')
            ax.legend()
            st.pyplot(fig)

        df_forecast = pd.DataFrame({
            'Step': np.arange(1, n_steps+1),
            'Prediksi': pred_log if show_as == 'Log-Return' else pred_price
        })

        st.write("📊 **Tabel Forecasting**")
        st.dataframe(df_forecast)

        csv_forecast = df_forecast.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Forecast (CSV)", csv_forecast, file_name=f'forecast_{nama_saham}.csv')
