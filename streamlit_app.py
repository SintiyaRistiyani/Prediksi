import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
# Set page config with a light clean style
st.set_page_config(
    page_title="Prediksi Harga Saham Produksi Rokok - Mixture Autoregressive",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for styling according to the default inspiration guidelines
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@600;800&display=swap');

        html, body, #root, .main {
            background: #ffffff;
            color: #6b7280;
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 0;
        }

        .block-container {
            max-width: 1200px;
            margin-left: auto;
            margin-right: auto;
            padding-top: 4rem;
            padding-bottom: 4rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        h1 {
            font-size: 48px;
            font-weight: 800;
            color: #111827;
            margin-bottom: 0.25rem;
        }

        h2 {
            font-weight: 600;
            color: #374151;
            margin-top: 0;
            margin-bottom: 1.5rem;
            font-size: 24px;
        }

        .intro-text {
            font-size: 18px;
            max-width: 700px;
            line-height: 1.5;
            margin-bottom: 3rem;
            color: #4b5563;
        }

        .card {
            background: #f9fafb;
            border-radius: 0.75rem;
            box-shadow: 0 1px 3px rgb(0 0 0 / 0.1);
            padding: 2rem;
            margin-bottom: 2rem;
        }

        .sticky-header {
            position: sticky;
            top: 0;
            background: white;
            z-index: 1000;
            border-bottom: 1px solid #e5e7eb;
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
        }

        .nav-links a {
            text-decoration: none;
            color: #6b7280;
            margin-left: 1.5rem;
            transition: color 0.3s ease;
            font-weight: 600;
        }
        .nav-links a:hover {
            color: #111827;
        }

        .btn-primary {
            background-color: #111827;
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 0.5rem;
            font-weight: 700;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .btn-primary:hover {
            background-color: #374151;
        }
        /* Line chart filter style */
        .filter-container {
            margin-bottom: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header / Navigation Bar
st.markdown(
    """
    <header class="sticky-header" role="banner" aria-label="Main Navigation">
        <div class="logo" style="font-size: 1.75rem; color:#111827;">RokokStockPred</div>
        <nav class="nav-links" role="navigation" aria-label="Primary Navigation">
            <a href="#overview">Overview</a>
            <a href="#data-sample">Data Sample</a>
            <a href="#model-prediction">Prediction</a>
        </nav>
    </header>
    """,
    unsafe_allow_html=True,
)

# Main container
with st.container():
    st.markdown('<section id="overview">', unsafe_allow_html=True)

    st.markdown("<h1>Prediksi Harga Saham Produksi Rokok</h1>", unsafe_allow_html=True)
    st.markdown("<h2>Model Mixture Autoregressive</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        <p class="intro-text">
        Penelitian ini bertujuan memprediksi harga saham industri rokok menggunakan
        model statistik Mixture Autoregressive (MAR). Aplikasi ini memungkinkan visualisasi 
        data harga saham, eksplorasi data, serta prediksi harga masa depan secara interaktif.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('</section>', unsafe_allow_html=True)

# Helper function to load example data
@st.cache_data
def load_stock_data():
    # For demonstration, we'll generate synthetic sample data here.
    # In practice, replace this with a read from CSV or database.
    np.random.seed(42)
    dates = pd.date_range(start='2017-01-01', periods=200, freq='B')  # Trading days
    prices = 100 + np.cumsum(np.random.normal(0, 1, size=len(dates)))  # synthetic price series

    df = pd.DataFrame({'Date': dates, 'Close': prices})
    return df

# Load data
stock_data = load_stock_data()

with st.container():
    st.markdown('<section id="data-sample">', unsafe_allow_html=True)
    st.markdown("<h2>Sample Data Harga Saham</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        <p>
        Contoh data harga saham produksi rokok mulai dari tahun 2017 hingga saat ini.
        Data ini digunakan sebagai dasar analisis dan prediksi.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Show data sample as a card
    st.write(stock_data.head(10))

    # Plot price line chart
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(stock_data['Date'], stock_data['Close'], color="#111827", linewidth=2)
    ax.set_title("Harga Saham Produksi Rokok (Close)", fontsize=18, fontweight='600', loc='left')
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Harga (IDR)")
    ax.grid(alpha=0.2)
    plt.xticks(rotation=45)

    st.pyplot(fig)
    st.markdown('</section>', unsafe_allow_html=True)

# Placeholder for model prediction inputs and outputs
with st.container():
    st.markdown('<section id="model-prediction">', unsafe_allow_html=True)
    st.markdown("<h2>Prediksi Harga Saham</h2>", unsafe_allow_html=True)

    st.markdown(
        """
        <p>
        Pilih parameter model dan jangkauan prediksi untuk melakukan simulasi prediksi harga saham 
        menggunakan model Mixture Autoregressive.
        </p>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        periods = st.slider(
            label="Jumlah periode hari perdagangan untuk prediksi",
            min_value=5,
            max_value=60,
            value=30,
            step=1
        )
        st.write(f"Prediksi untuk {periods} hari kedepan.")

    with col2:
        components = st.selectbox(
            label="Jumlah komponen Mixture Autoregressive",
            options=[1, 2, 3, 4, 5],
            index=1,
            help="Jumlah komponen campuran dalam model MAR."
        )

    # Placeholder prediction data
    # For demo, we simulate a naive prediction by continuing last close price with noise
    last_price = stock_data['Close'].iloc[-1]
    pred_index = pd.date_range(start=stock_data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=periods, freq='B')
    np.random.seed(1)
    noise = np.random.normal(0, 0.5, size=periods)
    pred_prices = last_price + np.cumsum(noise)  # simple random walk prediction

    pred_df = pd.DataFrame({'Date': pred_index, 'Predicted Close': pred_prices})

    # Plot prediction
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(stock_data['Date'], stock_data['Close'], label="Observed", color="#6b7280", alpha=0.6)
    ax2.plot(pred_df['Date'], pred_df['Predicted Close'], label="Predicted", color="#111827", linestyle='--', linewidth=2)
    ax2.set_title("Prediksi Harga Saham dengan Model Mixture Autoregressive", fontsize=18, fontweight='600', loc='left')
    ax2.set_xlabel("Tanggal")
    ax2.set_ylabel("Harga (IDR)")
    ax2.legend()
    ax2.grid(alpha=0.15)
    plt.xticks(rotation=45)

    st.pyplot(fig2)

    st.markdown('</section>', unsafe_allow_html=True)

# Footer with minimal
st.markdown(
    """
    <footer style="text-align:center; padding:2rem 0; color:#9ca3af; font-size:14px;">
      © 2024 Penelitian Prediksi Harga Saham Produksi Rokok
    </footer>
    """,
    unsafe_allow_html=True,
)

