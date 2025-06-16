import streamlit as st
import base64
import pandas as pd
import numpy as np
from scipy.stats import gennorm
import warnings
warnings.filterwarnings("ignore")

# ===================== Fungsi MAR-GED ===================== #
def preprocess(df, col):
    df = df.dropna(subset=[col])
    series = df[col].copy()
    series = series.astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
    series.index = pd.to_datetime(df.index)
    return series

def fit_mar_em_ged(series, n_components=2, ar_order=1, max_iter=100, tol=1e-4):
    X = np.column_stack([series.shift(i) for i in range(1, ar_order + 1)])
    X = X[ar_order:]
    y = series[ar_order:]

    X = np.hstack([np.ones((X.shape[0], 1)), X])
    y = y.values

    beta_init = np.linalg.pinv(X) @ y
    residuals = y - X @ beta_init

    pis = np.ones(n_components) / n_components
    coefs = [beta_init.copy() for _ in range(n_components)]
    scales = [np.std(residuals)] * n_components
    betas = [1.5] * n_components
    weights = np.ones((len(y), n_components)) / n_components
    log_likelihoods = []

    for iteration in range(max_iter):
        for k in range(n_components):
            mu_k = X @ coefs[k]
            pdf_k = gennorm.pdf(y, beta=betas[k], loc=mu_k, scale=scales[k])
            weights[:, k] = pis[k] * pdf_k
        weights /= weights.sum(axis=1, keepdims=True)
        pis = weights.mean(axis=0)
        for k in range(n_components):
            W = np.diag(weights[:, k])
            XtW = X.T @ W
            beta_k = np.linalg.pinv(XtW @ X) @ XtW @ y
            coefs[k] = beta_k
            residuals = y - X @ coefs[k]
            beta_est, loc_est, scale_est = gennorm.fit(residuals)
            betas[k], scales[k] = beta_est, scale_est
        ll = np.sum(np.log(np.sum([
            pis[k] * gennorm.pdf(y, beta=betas[k], loc=X @ coefs[k], scale=scales[k])
            for k in range(n_components)
        ], axis=0)))
        log_likelihoods.append(ll)
        if iteration > 1 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            break
    return pis, coefs, betas, scales, ar_order

def predict_mar_ged(pis, coefs, betas, scales, series, steps=10):
    ar_order = len(coefs[0]) - 1
    history = list(series[-ar_order:])
    preds = []
    for _ in range(steps):
        X_t = np.array([1.0] + history[-ar_order:][::-1])
        pred_k = np.array([X_t @ coefs[k] for k in range(len(coefs))])
        final_pred = np.sum(pis * pred_k)
        preds.append(final_pred)
        history.append(final_pred)
    return np.array(preds)

# ===================== Setup UI ===================== #

st.set_page_config(layout="wide", page_title="Prediksi Harga Saham")

if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    # Background image
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as file:
            encoded = base64.b64encode(file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded}");
                background-size: cover;
                background-position: center;
            }}
            .button {{
                background-color: #333;
                border: none;
                color: white;
                padding: 12px 24px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 18px;
                border-radius: 30px;
                margin-top: 30px;
            }}
            .center {{
                text-align: center;
                margin-top: 200px;
            }}
            </style>
            <div class="center">
                <h1>prediksi harga saham</h1>
                <form action="" method="post">
                    <button class="button">TRY NOW</button>
                </form>
            </div>
            """,
            unsafe_allow_html=True
        )

    add_bg_from_local("e41209f6-d8fd-4940-8c86-92915d6d624d.png")

    if st.form("next"):
        st.session_state.page = "predict"
        st.experimental_rerun()

elif st.session_state.page == "predict":
    st.title("📈 Dashboard Prediksi Harga Saham")
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
