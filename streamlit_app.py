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
elif menu == "Prediksi dan Visualisasi":
    st.title("📊 Prediksi dan Visualisasi")

    if 'model_type' not in st.session_state or 'log_return' not in st.session_state:
        st.warning("Model belum dilatih atau data log return tidak tersedia. Silakan jalankan training terlebih dahulu.")
        st.stop()

    model_type = st.session_state['model_type']
    log_return = st.session_state['log_return'].dropna()
    selected_col = st.session_state['selected_price_col']
    df = st.session_state['df']

    st.markdown(f"### 🔮 Prediksi Harga dengan **{model_type}**")
    n_steps = st.number_input("Jumlah langkah prediksi ke depan (hari):", min_value=1, max_value=365, value=30)

    # Tombol Prediksi
    if st.button("📈 Prediksi"):
        st.info("⏳ Sedang memproses prediksi...")

        # Fungsi prediksi MAR-Normal
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

        # Fungsi prediksi MAR-GED
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

        # Jalankan prediksi sesuai model
        if model_type == "MAR-Normal":
            model = st.session_state['best_model']
            pred_log_return = predict_mar_normal(model, log_return.values, n_steps=n_steps)
        elif model_type == "MAR-GED":
            model = st.session_state['best_model_ged']
            pred_log_return = predict_mar_ged(model, log_return.values, n_steps=n_steps)
        else:
            st.error("Model tidak dikenali.")
            st.stop()

        # Transformasi log-return → harga
        last_price = df[selected_col].dropna().iloc[-1]
        future_prices = [last_price]
        for r in pred_log_return:
            future_prices.append(future_prices[-1] * np.exp(r))
        future_prices = future_prices[1:]

        future_dates = pd.date_range(start=df.index[-1], periods=n_steps + 1, freq='D')[1:]
        df_future = pd.DataFrame({'Tanggal': future_dates, 'Prediksi Harga': future_prices}).set_index('Tanggal')

        # Tampilkan hasil
        st.success("✅ Prediksi selesai!")
        st.line_chart(df_future)
        st.dataframe(df_future)

        # Simpan prediksi ke session_state (jika ingin dipakai di halaman lain)
        st.session_state['prediksi_harga'] = df_future
        
# ----------------- Halaman Prediksi dan Visualisasi -----------------
elif menu == "Prediksi dan Visualisasi":
    st.title("📊 Prediksi dan Visualisasi")

    if 'model_type' not in st.session_state or 'log_return' not in st.session_state:
        st.warning("Model belum dilatih atau log return tidak tersedia.")
        st.stop()

    model_type = st.session_state['model_type']
    df = st.session_state['df']
    log_return = st.session_state['log_return'].dropna()
    selected_col = st.session_state['selected_price_col']

    st.markdown(f"### 🔮 Prediksi Harga Menggunakan Model: **{model_type}**")

    # Input jumlah langkah prediksi
    n_steps = st.number_input("Jumlah langkah prediksi ke depan (n_steps):", min_value=1, max_value=365, value=30, step=1)

    # ----------------- Fungsi Prediksi MAR-Normal -----------------
    def predict_mar_normal(model, X_init, n_steps=30):
        ar_params = model['ar_params']
        weights = model['weights']
        p = ar_params.shape[1]

        # Komponen dominan (komponen dengan bobot terbesar)
        main_k = np.argmax(weights)
        phi = ar_params[main_k]

        # Prediksi iteratif
        preds = []
        X_curr = list(X_init[-p:])  # ambil p nilai terakhir sebagai input awal

        for _ in range(n_steps):
            x_lag = np.array(X_curr[-p:])[::-1]  # urutan lag terbaru ke lama
            next_val = np.dot(phi, x_lag)
            preds.append(next_val)
            X_curr.append(next_val)

        return np.array(preds)

    # ----------------- Prediksi -----------------
# ----------------- Halaman Prediksi dan Visualisasi -----------------
elif menu == "Prediksi dan Visualisasi":
    st.title("📊 Prediksi dan Visualisasi")

    if 'model_type' not in st.session_state or 'log_return' not in st.session_state:
        st.warning("Model belum dilatih atau log return tidak tersedia.")
        st.stop()

    model_type = st.session_state['model_type']
    df = st.session_state['df']
    log_return = st.session_state['log_return'].dropna()
    selected_col = st.session_state['selected_price_col']

    st.markdown(f"### 🔮 Prediksi Harga Menggunakan Model: **{model_type}**")

    # Input jumlah langkah prediksi
    n_steps = st.number_input("Jumlah langkah prediksi ke depan (n_steps):", min_value=1, max_value=365, value=30, step=1)

    # ----------------- Fungsi Prediksi MAR-Normal -----------------
    def predict_mar_normal(model, X_init, n_steps=30):
        ar_params = model['ar_params']
        weights = model['weights']
        p = ar_params.shape[1]

        # Komponen dominan (komponen dengan bobot terbesar)
        main_k = np.argmax(weights)
        phi = ar_params[main_k]

        preds = []
        X_curr = list(X_init[-p:])  # ambil p nilai terakhir sebagai input awal

        for _ in range(n_steps):
            x_lag = np.array(X_curr[-p:])[::-1]  # urutan lag terbaru ke lama
            next_val = np.dot(phi, x_lag)
            preds.append(next_val)
            X_curr.append(next_val)

        return np.array(preds)

    # ----------------- Fungsi Prediksi MAR-GED -----------------
    def predict_mar_ged(model, X_init, n_steps=30):
        pred = []
        X_curr = list(X_init[-model['ar_params'].shape[1]:])  # gunakan p terakhir
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

    # ----------------- Prediksi -----------------
    if model_type == "MAR-Normal":
        if 'best_model' not in st.session_state or 'best_p' not in st.session_state:
            st.error("Model MAR-Normal belum tersedia. Jalankan training terlebih dahulu.")
            st.stop()

        model = st.session_state['best_model']
        X = log_return.values

        pred_log_return = predict_mar_normal(model, X, n_steps=n_steps)

    elif model_type == "MAR-GED":
        if 'best_model_ged' not in st.session_state or 'best_p_ged' not in st.session_state:
            st.error("Model MAR-GED belum tersedia. Jalankan training terlebih dahulu.")
            st.stop()

        model = st.session_state['best_model_ged']
        X = log_return.values

        pred_log_return = predict_mar_ged(model, X, n_steps=n_steps)

    else:
        st.warning("Model tidak dikenali.")
        st.stop()

    # ----------------- Transformasi log-return → harga -----------------
    last_price = df[selected_col].dropna().iloc[-1]
    future_prices = [last_price]
    for r in pred_log_return:
        future_prices.append(future_prices[-1] * np.exp(r))
    future_prices = future_prices[1:]

    # Buat tanggal untuk prediksi
    last_date = df[df[selected_col].notna()].iloc[-1].name
    future_dates = pd.date_range(start=last_date, periods=n_steps+1, freq='D')[1:]

    df_future = pd.DataFrame({
        'Tanggal': future_dates,
        'Prediksi Harga': future_prices
    }).set_index('Tanggal')

    # ----------------- Visualisasi -----------------
    st.markdown("### 📈 Prediksi Harga Ke Depan")
    st.line_chart(df_future)

    # ----------------- Tabel Prediksi -----------------
    st.markdown("### 📋 Tabel Hasil Prediksi")
    st.dataframe(df_future)
    
# ----------------- Halaman Interpretasi dan Saran -----------------
elif menu == "Interpretasi dan Saran":
    st.title("📌 Interpretasi dan Saran")

    st.markdown("## 📈 Ringkasan Kinerja Model")
    for saham in stock_names:  # contoh: ['GUDANG GARAM', 'SAMPOERNA', 'WISMILAK']
        st.subheader(f"📊 {saham}")

        # Ambil metrik dari session_state
        model_used = st.session_state.get(f"{saham}_model_type", "N/A")
        mape = st.session_state.get(f"{saham}_mape", None)
        loglik = st.session_state.get(f"{saham}_loglik", None)

        # Tampilkan hasil evaluasi
        st.markdown(f"**Model yang digunakan:** {model_used}")
        if mape is not None:
            st.markdown(f"**MAPE:** {mape:.2f}%")
        if loglik is not None:
            st.markdown(f"**Log-Likelihood:** {loglik:.2f}")
        
        # Rekomendasi sederhana berdasarkan MAPE
        if mape is not None:
            if mape < 5:
                st.success("✅ Model memiliki akurasi sangat baik untuk prediksi jangka pendek.")
            elif mape < 10:
                st.info("ℹ️ Model cukup baik, meskipun masih dapat ditingkatkan.")
            else:
                st.warning("⚠️ Akurasi masih perlu ditingkatkan. Pertimbangkan pemodelan ulang.")

    st.markdown("---")
    st.markdown("## 💡 Saran Pengembangan")

    st.markdown("""
    - Coba bandingkan performa model ARIMA dan MAR secara menyeluruh, tidak hanya dari MAPE tapi juga dari log-likelihood dan visualisasi tren.
    - Gunakan validasi silang (cross-validation) pada data time series untuk menghindari overfitting.
    - Tambahkan eksternal variabel (seperti IHSG, inflasi, nilai tukar) jika ingin membangun model regresi multivariat (misalnya VAR atau MAR multivariat).
    - Untuk jangka panjang, pertimbangkan model machine learning seperti LSTM jika data lebih banyak tersedia.
    """)

    st.markdown("## 🗂️ Dokumentasi dan Referensi")
    st.markdown("""
    - Rasyid (2018). *Model Autoregressive dengan Distribusi Student-t dan Generalized Error Distribution untuk Prediksi Harga Saham.*
    - Historini (2010). *Model Mixture Autoregressive untuk Peramalan Deret Waktu Keuangan.*
    - Asrini (2013). *Analisis Peramalan Harga Saham dengan Model MAR.*
    """)

    st.markdown("Terima kasih telah menggunakan aplikasi ini. 🙏")
