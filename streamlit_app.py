import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gennorm
import warnings
warnings.filterwarnings("ignore")

# (tambahkan fungsi MAR-GED di sini seperti sebelumnya...)

st.set_page_config(page_title="Prediksi Harga Saham", layout="wide")

st.title("📈 Dashboard Prediksi Harga Saham")
st.markdown("Prediksi menggunakan model **Mixture Autoregressive (MAR)** dengan distribusi **GED**.")

uploaded_file = st.file_uploader("Unggah file CSV harga saham", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, delimiter=';')
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df.set_index('Date', inplace=True)

    saham_list = df.columns.tolist()
    saham = st.selectbox("Pilih perusahaan", saham_list)

    if st.button("🔍 Analisis"):
        series = preprocess(df, saham)

        st.subheader(f"📊 Statistik {saham}")
        st.write(series.describe())

        st.line_chart(series, use_container_width=True)

        pred_days = st.slider("Jumlah hari ke depan", min_value=5, max_value=60, value=30)

        with st.spinner("Melakukan estimasi model dan prediksi..."):
            train_series = series[:-pred_days]
            pis, coefs, betas, scales, ar_order = fit_mar_em_ged(train_series, n_components=2, ar_order=2)
            preds = predict_mar_ged(pis, coefs, betas, scales, train_series, steps=pred_days)
            future_dates = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=pred_days, freq='B')
            pred_df = pd.DataFrame({f'Prediksi {saham}': preds}, index=future_dates)

        st.subheader("📈 Hasil Prediksi")
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
    st.info("Silakan unggah file CSV dengan format: `Date; GUDANG GARAM; SAMPOERNA; WISMILAK` ...")

st.markdown("---")
st.caption("© 2025 Prediksi Saham Rokok | Dibuat dengan Streamlit")
