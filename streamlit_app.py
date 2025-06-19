import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gennorm

st.set_page_config(page_title="Prediksi Harga Saham", layout="wide")
st.title("📈 Dashboard Prediksi Harga Saham")
st.markdown("Prediksi menggunakan model **Mixture Autoregressive (MAR)** dengan distribusi **GED** dan estimasi parameter **EM**.")
# --- Sidebar Menu ---
st.sidebar.markdown("#### MENU NAVIGASI 🧭")

menu_items = {
    "HOME": "home",
    "INPUT DATA": "input_data",
    "DATA PREPROCESSING": "data_preprocessing",
    "STASIONERITAS DATA": "stasioneritas_data",
    "DATA SPLITTING": "data_splitting",
    "MODEL ARIMA": "pemodelan_arima", # Diubah namanya
    "PREDIKSI ARIMA": "prediksi_arima", # Tambahan menu prediksi untuk ARIMA
    "MODEL NGARCH": "pemodelan_ngarch", # Diubah namanya
    "PREDIKSI NGARCH": "prediksi_ngarch", # Tambahan menu prediksi untuk NGARCH
    "INTERPRETASI & SARAN": "interpretasi_saran",
}

if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'home'
if 'selected_currency' not in st.session_state:
    st.session_state['selected_currency'] = None
if 'variable_name' not in st.session_state:
    st.session_state['variable_name'] = "Nama Variabel"

for item, key in menu_items.items():
    if st.sidebar.button(item, key=key):
        st.session_state['current_page'] = key

uploaded_file = st.file_uploader("Unggah file CSV harga saham", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, delimiter=';')
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df.set_index('Date', inplace=True)

    saham_list = df.columns.tolist()
    saham = st.selectbox("Pilih perusahaan", saham_list)

    series = preprocess(df, saham)
    st.line_chart(series, use_container_width=True)

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
    st.info("Silakan unggah file CSV  yang harus memiliki kolom : Date, Harga Saham ")

st.markdown("---")
st.caption("© 2025 Prediksi Harga Saham | Dibuat dengan Streamlit")
