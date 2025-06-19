import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import math
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm # Untuk Ljung-Box, Jarque-Bera
from scipy import stats # Untuk Jarque-Bera test

# ----------------- Konfigurasi Dashboard -----------------
st.set_page_config(page_title="Prediksi Harga Saham", layout="wide")
st.title("📈 Dashboard Prediksi Harga Saham")
st.markdown("Prediksi harga saham menggunakan model **Mixture Autoregressive (MAR)** dengan distribusi **GED** dan estimasi parameter **EM**.")

# ----------------- Menu Navigasi Sidebar -----------------
st.sidebar.markdown("#### MENU NAVIGASI 🧭")
menu_items = {
    "HOME": "home",
    "INPUT DATA": "input_data",
    "DATA PREPROCESSING": "data_preprocessing",
    "STASIONERITAS DATA": "stasioneritas_data",
    "DATA SPLITTING": "data_splitting",
    "MODEL ARIMA": "pemodelan_arima",
    "PREDIKSI ARIMA": "prediksi_arima",
    "MODEL NGARCH": "pemodelan_ngarch",
    "PREDIKSI NGARCH": "prediksi_ngarch",
    "INTERPRETASI & SARAN": "interpretasi_saran",
}

# ----------------- State Session -----------------
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'home'
if 'selected_currency' not in st.session_state:
    st.session_state['selected_currency'] = None
if 'variable_name' not in st.session_state:
    st.session_state['variable_name'] = "Nama Variabel"

for item, key in menu_items.items():
    if st.sidebar.button(item, key=key):
        st.session_state['current_page'] = key

# ----------------- Fungsi Preprocess -----------------
def preprocess(df, col):
    series = df[col].dropna()
    return series

# ----------------- Fungsi Estimasi MAR-GED dengan EM -----------------
def fit_mar_em_ged(series, n_components=2, ar_order=1, max_iter=100, tol=1e-4):
    n = len(series)
    X = np.column_stack([series.shift(i) for i in range(1, ar_order + 1)])
    X = X[ar_order:]
    Y = series[ar_order:]

    X = X.values
    Y = Y.values.reshape(-1, 1)

    pis = np.ones(n_components) / n_components
    betas = np.random.randn(n_components, ar_order)
    coefs = np.random.randn(n_components, 1)
    scales = np.ones(n_components)

    log_likelihood_old = -np.inf
    for iteration in range(max_iter):
        # E-step
        responsibilities = np.zeros((len(Y), n_components))
        for k in range(n_components):
            mean_k = X @ betas[k].reshape(-1, 1)
            resid = Y - mean_k
            pdf_vals = gennorm.pdf(resid.flatten(), beta=1.5, loc=0, scale=scales[k])
            responsibilities[:, k] = pis[k] * pdf_vals
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)

        # M-step
        Nk = responsibilities.sum(axis=0)
        pis = Nk / len(Y)

        for k in range(n_components):
            W = np.diag(responsibilities[:, k])
            beta_k = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Y
            betas[k] = beta_k.flatten()
            resid = Y - X @ beta_k
            scales[k] = np.sqrt((responsibilities[:, k] * resid.flatten() ** 2).sum() / Nk[k])

        # Cek konvergensi
        log_likelihood = np.sum(np.log(responsibilities.sum(axis=1)))
        if np.abs(log_likelihood - log_likelihood_old) < tol:
            break
        log_likelihood_old = log_likelihood

    return pis, coefs, betas, scales, ar_order

# ----------------- Fungsi Prediksi MAR-GED -----------------
def predict_mar_ged(pis, coefs, betas, scales, series, steps=10):
    ar_order = betas.shape[1]
    preds = []
    history = series.values.tolist()

    for _ in range(steps):
        forecast_components = []
        for k in range(len(pis)):
            recent_vals = np.array(history[-ar_order:][::-1])
            pred = np.dot(betas[k], recent_vals)
            forecast_components.append(pis[k] * pred)
        forecast = np.sum(forecast_components)
        preds.append(forecast)
        history.append(forecast)

    return preds

# ----------------- Upload dan Prediksi -----------------
uploaded_file = st.file_uploader("Unggah file CSV harga saham", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, delimiter=';')

    required_columns = {'Date'}
    if not required_columns.issubset(df.columns):
        st.error("❌ CSV harus memiliki kolom 'Date' dan setidaknya satu kolom harga saham.")
        st.stop()

    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df.set_index('Date', inplace=True)

    saham_list = df.columns.tolist()
    saham = st.selectbox("Pilih perusahaan", saham_list)

series = preprocess(df, saham)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(series.index, series.values, label=saham, color='blue')
ax.set_title(f"Harga Saham {saham} dari Waktu ke Waktu")
ax.set_xlabel("Tanggal")
ax.set_ylabel("Harga")
ax.grid(True)
ax.legend()

st.pyplot(fig)

    st.subheader(f"📊 Statistik Deskriptif {saham}")
    st.write(series.describe())

    st.subheader("🔮 Prediksi Harga dengan MAR-GED")
    pred_days = st.slider("Jumlah hari ke depan", min_value=5, max_value=60, value=30)

    if st.button("Mulai Prediksi"):
        with st.spinner("Melakukan estimasi model dan prediksi..."):
            train_series = series[:-pred_days]
            pis, coefs, betas, scales, ar_order = fit_mar_em_ged(train_series, n_components=2, ar_order=2)
            preds = predict_mar_ged(pis, coefs, betas, scales, train_series, steps=pred_days)
            future_dates = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=pred_days, freq='B')
            pred_df = pd.DataFrame({f'Prediksi {saham}': preds}, index=future_dates)

        st.line_chart(pred_df, use_container_width=True)

        with st.expander("📥 Unduh hasil prediksi"):
            st.dataframe(pred_df)
            csv = pred_df.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Download sebagai CSV",
                data=csv,
                file_name=f'prediksi_{saham.lower()}.csv',
                mime='text/csv',
            )
else:
    st.info("Silakan unggah file CSV yang harus memiliki kolom : Date, Harga Saham ")

st.markdown("---")
st.caption("© 2025 Prediksi Harga Saham | Dibuat dengan Streamlit")
