import streamlit as st
import pandas as pd
import math
from pathlib import Path
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Konfigurasi Halaman (Hanya dipanggil sekali di awal) ---
st.set_page_config(
    page_title='Prediksi Harga Saham - Mixture Autoregressive', # Tambah emoji di sini
    page_icon='📈', # Ini adalah icon untuk tab browser
    layout="wide"
)
        color: #374151; /* neutral gray-700 */
    }
    h1 {
        font-weight: 700;
        font-size: 3rem;
        margin-bottom: 0.25rem;
        color: #111827; /* neutral gray-900 */
    }
    h2 {
        font-weight: 500;
        font-size: 1.25rem;
        margin-top: 0;
        margin-bottom: 2rem;
        color: #6b7280; /* neutral gray-500 */
    }
    </style>
    """,
    unsafe_allow_html=True,
)
with st.container():
    st.markdown(
        """
        <div class="main-container">
            <h1>Prediksi Harga Saham Rokok</h1>
            <h2>Model Mixture Autoregressive dan ARIMA</h2>
            <p>
                Penelitian ini bertujuan untuk melakukan prediksi harga saham industri rokok menggunakan pendekatan model statistik canggih
                yaitu Mixture Autoregressive dan ARIMA. Aplikasi ini memberikan visualisasi dan analisis data harga saham secara interaktif.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# :earth_americas: GDP dashboard

Browse GDP data from the [World Bank Open Data](https://data.worldbank.org/) website. As you'll
notice, the data only goes to 2022 right now, and datapoints for certain years are often missing.
But it's otherwise a great (and did I mention _free_?) source of data.
'''

# Add some spacing
''
''

min_value = gdp_df['Year'].min()
max_value = gdp_df['Year'].max()

from_year, to_year = st.slider(
    'Which years are you interested in?',
    min_value=min_value,
    max_value=max_value,
    value=[min_value, max_value])

countries = gdp_df['Country Code'].unique()

if not len(countries):
    st.warning("Select at least one country")

selected_countries = st.multiselect(
    'Which countries would you like to view?',
    countries,
    ['DEU', 'FRA', 'GBR', 'BRA', 'MEX', 'JPN'])

''
''
''

# Filter the data
filtered_gdp_df = gdp_df[
    (gdp_df['Country Code'].isin(selected_countries))
    & (gdp_df['Year'] <= to_year)
    & (from_year <= gdp_df['Year'])
]

st.header('GDP over time', divider='gray')

''

st.line_chart(
    filtered_gdp_df,
    x='Year',
    y='GDP',
    color='Country Code',
)

''
''


first_year = gdp_df[gdp_df['Year'] == from_year]
last_year = gdp_df[gdp_df['Year'] == to_year]

st.header(f'GDP in {to_year}', divider='gray')

''

cols = st.columns(4)

for i, country in enumerate(selected_countries):
    col = cols[i % len(cols)]

    with col:
        first_gdp = first_year[first_year['Country Code'] == country]['GDP'].iat[0] / 1000000000
        last_gdp = last_year[last_year['Country Code'] == country]['GDP'].iat[0] / 1000000000

        if math.isnan(first_gdp):
            growth = 'n/a'
            delta_color = 'off'
        else:
            growth = f'{last_gdp / first_gdp:,.2f}x'
            delta_color = 'normal'

        st.metric(
            label=f'{country} GDP',
            value=f'{last_gdp:,.0f}B',
            delta=growth,
            delta_color=delta_color
        )
