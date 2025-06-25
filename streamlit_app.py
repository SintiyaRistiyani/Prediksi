import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    st.title("📈 Aplikasi Prediksi Harga Saham Menggunakan Model ARIMA dan MAR")
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
    df['Log_Return'] = np.log(df[selected_column] / df[selected_column].shift(1)).dropna()
    df = df.dropna(subset=['Log_Return'])
    st.session_state['log_return'] = df['Log_Return']
    st.session_state['df'] = df  # simpan update dataframe dengan log return

    st.dataframe(df[[selected_column, 'Log_Return']].head()

        # Visualisasi
        st.markdown("### 4️⃣ Visualisasi Data dan Log Return")

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

    # Validasi apakah log return tersedia
    if 'log_return' not in st.session_state or 'selected_price_col' not in st.session_state:
        st.warning("Silakan lakukan preprocessing terlebih dahulu agar log return tersedia.")
        st.stop()

    log_return = st.session_state['log_return']
    selected_col = st.session_state['selected_price_col']

    st.markdown(f"Kolom yang dianalisis: **{selected_col}**")
    st.markdown("Uji stasioneritas dilakukan terhadap **log return** dari data harga saham.")

    # Siapkan dataframe log return untuk uji ADF
    df_test = pd.DataFrame(log_return, columns=[selected_col])

     # Fungsi ADF Test
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(df_test[selected_col].dropna())

    # Tampilkan hasil
    st.markdown(f"""
    **Hasil Uji ADF (Augmented Dickey-Fuller):**
    - **ADF Statistic**: {result[0]:.4f}  
    - **p-value**: {result[1]:.4f}  
    - **Critical Values**:
    """)
    for key, value in result[4].items():
        st.markdown(f"- {key}: {value:.4f}")

    # Interpretasi
    if result[1] < 0.05:
        st.success("✅ Log return stasioner (tolak H0 - tidak ada akar unit).")
    else:
        st.error("❌ Log return tidak stasioner (gagal tolak H0 - ada akar unit).")

# ----------------- Halaman Model -----------------
elif menu == "Model":
    st.title("🔧 Pemodelan")

    if 'log_return' not in st.session_state or 'selected_price_col' not in st.session_state:
        st.warning("Silakan lakukan preprocessing terlebih dahulu untuk mendapatkan log return.")
        st.stop()

    log_return = st.session_state['log_return']
    selected_col = st.session_state['selected_price_col']

    st.markdown("### 📌 Pilih Model yang Ingin Digunakan")
    model_type = st.radio("Pilih jenis model:", ["ARIMA", "Mixture Autoregressive (MAR)"])
    st.session_state['model_type'] = model_type

    # --------------------- ARIMA ---------------------
    if model_type == "ARIMA":
        st.markdown("### ⚙️ Parameter ARIMA (p, d, q)")
        p = st.number_input("p (Autoregressive term)", min_value=0, value=1, step=1)
        d = st.number_input("d (Differencing term)", min_value=0, value=0, step=1)
        q = st.number_input("q (Moving average term)", min_value=0, value=1, step=1)

        if st.button("🚀 Jalankan Model ARIMA"):
            from statsmodels.tsa.arima.model import ARIMA
            import warnings
            warnings.filterwarnings("ignore")

            st.markdown("⏳ Melatih model ARIMA...")
            try:
                model = ARIMA(log_return, order=(p, d, q))
                model_fit = model.fit()

                st.session_state['arima_model'] = model_fit
                pred = model_fit.predict()
                st.session_state['arima_pred'] = pred

                st.success("✅ Model ARIMA berhasil dilatih!")
                st.markdown("### 📋 Ringkasan Model")
                st.text(model_fit.summary())

            except Exception as e:
                st.error(f"❌ Gagal melatih model ARIMA: {e}")

    # --------------------- MAR ---------------------
elif model_type == "Mixture Autoregressive (MAR)":
    st.markdown("### ⚙️ Pilih Metode Pelatihan Model MAR")

    mar_method = st.radio("Pilih metode pelatihan:", [
        "Hitung Otomatis (EM Algorithm)", 
        "Masukkan Parameter Manual"
    ])

    if mar_method == "Hitung Otomatis (EM Algorithm)":
        k = st.number_input("Jumlah Komponen (k)", min_value=1, value=2, step=1)
        p = st.number_input("Order AR (p)", min_value=1, value=1, step=1)

        if st.button("🚀 Jalankan EM untuk MAR (GED)"):
            try:
                from numpy.linalg import inv
                from scipy.special import gamma
                import numpy as np

                def ged_pdf(x, mu, sigma, nu):
                    beta = sigma * (gamma(1/nu) / gamma(3/nu))**0.5
                    coeff = nu / (2 * beta * gamma(1/nu))
                    z = np.abs((x - mu) / beta)
                    return coeff * np.exp(-z**nu)

                log_ret = st.session_state['log_return'].dropna().values
                X = np.column_stack([log_ret[i:-(p - i)] for i in range(p)])
                Y = log_ret[p:]
                n = len(Y)

                np.random.seed(42)
                pis = np.full(k, 1/k)
                betas = [np.random.randn(p) for _ in range(k)]
                sigmas = np.full(k, np.std(Y))
                nus = np.full(k, 2.0)  # GED shape parameter (ν)

                max_iter = 100
                for _ in range(max_iter):
                    gamma_mat = np.zeros((n, k))
                    for j in range(k):
                        mu = X @ betas[j]
                        gamma_mat[:, j] = pis[j] * ged_pdf(Y, mu, sigmas[j], nus[j])
                    gamma_mat /= gamma_mat.sum(axis=1, keepdims=True)

                    pis = gamma_mat.mean(axis=0)
                    for j in range(k):
                        W = np.diag(gamma_mat[:, j])
                        XW = X.T @ W
                        betas[j] = inv(XW @ X) @ XW @ Y
                        residual = Y - X @ betas[j]
                        sigmas[j] = np.sqrt((gamma_mat[:, j] * residual**2).sum() / gamma_mat[:, j].sum())

                st.session_state['mar_model'] = {
                    'pis': pis, 'betas': betas, 'sigmas': sigmas, 'nus': nus,
                    'k': k, 'p': p
                }

                st.success("✅ MAR (GED) berhasil dilatih dengan EM!")
                for j in range(k):
                    st.markdown(f"#### Komponen {j+1}")
                    st.write(f"Koefisien AR: {betas[j]}")
                    st.write(f"σ²: {sigmas[j]**2:.6f}")
                    st.write(f"Proporsi: {pis[j]:.4f}")
                    st.write(f"ν (shape): {nus[j]:.2f}")

            except Exception as e:
                st.error(f"Gagal saat EM training: {e}")

    elif mar_method == "Masukkan Parameter Manual":
        st.markdown("#### ✍️ Masukkan Parameter MAR secara Manual")
        k = st.number_input("Jumlah Komponen (k)", min_value=1, value=2, step=1)
        p = st.number_input("Order AR (p)", min_value=1, value=1, step=1)

        betas = []
        sigmas = []
        pis = []
        nus = []

        for i in range(k):
            st.markdown(f"##### Komponen {i+1}")
            beta_input = st.text_input(f"Koefisien AR (pisahkan dengan koma) Komponen {i+1}", value="0.5")
            sigma_input = st.number_input(f"Sigma Komponen {i+1}", value=0.1)
            pi_input = st.number_input(f"Proporsi Komponen {i+1}", min_value=0.0, max_value=1.0, value=1/k)
            nu_input = st.number_input(f"GED Shape ν Komponen {i+1}", min_value=1.0, value=2.0)

            beta_array = np.fromstring(beta_input, sep=',')
            if len(beta_array) != p:
                st.error(f"Jumlah koefisien AR untuk Komponen {i+1} harus sama dengan order p ({p})")
                st.stop()

            betas.append(beta_array)
            sigmas.append(sigma_input)
            pis.append(pi_input)
            nus.append(nu_input)

        if st.button("🚀 Simpan Parameter MAR Manual"):
            pis = np.array(pis)
            pis /= pis.sum()

            st.session_state['mar_model'] = {
                'pis': pis,
                'betas': betas,
                'sigmas': sigmas,
                'nus': nus,
                'k': k,
                'p': p
            }

            st.success("✅ Parameter MAR berhasil disimpan!")
            for j in range(k):
                st.markdown(f"#### Komponen {j+1}")
                st.write(f"Koefisien AR: {betas[j]}")
                st.write(f"σ²: {sigmas[j]**2:.6f}")
                st.write(f"Proporsi: {pis[j]:.4f}")
                st.write(f"ν (shape): {nus[j]:.2f}")
                    
# ----------------- Halaman Prediksi dan Visualisasi -----------------
elif menu == "Prediksi dan Visualisasi":
    st.title("📊 Prediksi dan Visualisasi")

    if 'model_type' not in st.session_state or 'log_return' not in st.session_state:
        st.warning("Model belum dilatih atau log return tidak tersedia.")
        st.stop()

    model_type = st.session_state['model_type']
    log_return = st.session_state['log_return'].dropna()
    st.markdown(f"### 🔮 Hasil Prediksi Menggunakan Model: **{model_type}**")
    
    forecast_steps = st.number_input("Masukkan jumlah langkah prediksi ke depan:"
                                     
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # ----------------- Prediksi dengan ARIMA -----------------
    if model_type == "ARIMA":
        if 'arima_model' not in st.session_state:
            st.warning("Model ARIMA belum tersedia.")
            st.stop()

        model_fit = st.session_state['arima_model']
        pred = model_fit.predict(start=0, end=len(log_return)-1)

        df_pred = pd.DataFrame({
            "Aktual": log_return.values,
            "Prediksi": pred.values
        }, index=log_return.index).dropna()

    # ----------------- Prediksi dengan MAR -----------------
    elif model_type == "Mixture Autoregressive (MAR)":
        if 'mar_model' not in st.session_state:
            st.warning("Model MAR belum tersedia.")
            st.stop()

        model = st.session_state['mar_model']
        pis = model['pis']
        betas = model['betas']
        sigmas = model['sigmas']
        nus = model['nus']
        k = model['k']
        p = model['p']

        data = log_return.values
        if len(data) <= p:
            st.error("Data tidak cukup untuk membuat prediksi MAR dengan order p.")
            st.stop()

        # Membentuk data X untuk prediksi
        X_pred = np.column_stack([data[i:-(p - i)] for i in range(p)])
        y_actual = data[p:]
        pred_len = len(X_pred)

        y_pred = np.zeros(pred_len)
        for j in range(k):
            y_pred += pis[j] * (X_pred @ betas[j])

        df_pred = pd.DataFrame({
            "Aktual": y_actual,
            "Prediksi": y_pred
        }, index=log_return.index[p:]).dropna()

    # ----------------- Visualisasi -----------------
    st.markdown("### 📈 Visualisasi Prediksi vs Aktual")
    st.line_chart(df_pred)

    # ----------------- Tabel Prediksi -----------------
    st.markdown("### 📋 Tabel Hasil Prediksi (10 Baris Terakhir)")
    st.dataframe(df_pred.tail(10))

    # ----------------- Evaluasi -----------------
    st.markdown("### ✅ Evaluasi Akurasi (MAPE)")
    mape = mean_absolute_percentage_error(df_pred["Aktual"], df_pred["Prediksi"])
    st.write(f"**MAPE (Mean Absolute Percentage Error):** {mape:.2f}%")

# ----------------- Halaman Interpretasi dan Saran -----------------
elif menu == "Interpretasi dan Saran":
    st.title("📝 Interpretasi dan Saran")
    st.markdown("""
        #### Interpretasi:
        - Model menunjukkan performa yang cukup baik dalam menangkap tren harga saham.
        - Terdapat fluktuasi yang masih bisa diperbaiki pada komponen residual.

        #### Saran:
        - Lakukan tuning parameter pada model.
        - Pertimbangkan faktor eksternal seperti berita pasar atau sentimen investor.
        - Gunakan model lanjutan seperti **Mixture Autoregressive (MAR)** untuk data yang memiliki switching regime.
    """)
