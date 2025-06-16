import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Prediksi Harga Saham", layout="wide")

st.title("📈 Dashboard Prediksi Harga Saham Rokok")
st.markdown("Prediksi menggunakan model **Mixture Autoregressive (MAR)** dengan distribusi **GED**.")

uploaded_file = st.file_uploader("Unggah file CSV harga saham", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    saham_list = df.columns.tolist()
    saham = st.selectbox("Pilih perusahaan", saham_list)
    
    st.line_chart(df[saham], use_container_width=True)

    st.subheader(f"📊 Statistik {saham}")
    st.write(df[saham].describe())

    # Simulasi prediksi dummy
    st.subheader("🔮 Prediksi Harga")
    pred_hari = st.slider("Jumlah hari ke depan", min_value=1, max_value=30, value=7)
    
    # Dummy prediksi (gantilah dengan model MAR asli)
    last_value = df[saham].iloc[-1]
    noise = np.random.normal(0, 1, size=pred_hari)
    pred = last_value + np.cumsum(noise)

    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=pred_hari)
    pred_df = pd.DataFrame({f'Prediksi {saham}': pred}, index=future_dates)

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
    st.info("Silakan unggah file CSV dengan format: `Date`, `GUDANG GARAM`, `SAMPOERNA`, `WISMILAK`, ...")

st.markdown("---")
st.caption("© 2025 Prediksi Saham Rokok | Dibuat dengan Streamlit")
