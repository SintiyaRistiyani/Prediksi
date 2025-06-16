import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image
import base64

# Set full screen layout
st.set_page_config(layout="wide")

# CSS custom styling
st.markdown("""
    <style>
        .main {
            background: linear-gradient(to top, #a0d468 50%, #dff0ea 50%);
            height: 100vh;
        }
        .cloud {
            position: absolute;
            top: 100px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 50px;
            color: white;
        }
        .title {
            position: absolute;
            top: 200px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 30px;
            font-weight: bold;
            color: black;
        }
        .button {
            position: absolute;
            top: 270px;
            left: 50%;
            transform: translateX(-50%);
        }
        .logos {
            position: absolute;
            top: 20px;
            left: 20px;
            display: flex;
            gap: 20px;
        }
        .logo-img {
            width: 50px;
            height: 50px;
        }
    </style>
""", unsafe_allow_html=True)

# Inject HTML for layout
st.markdown(f"""
    <div class="main">
        <div class="logos">
            <img src="data:image/png;base64,{base64.b64encode(open("logo1.png", "rb").read()).decode()}" class="logo-img">
            <img src="data:image/png;base64,{base64.b64encode(open("logo2.png", "rb").read()).decode()}" class="logo-img">
        </div>
        <div class="title">prediksi harga saham</div>
        <div class="button">
            <a href="#" target="_self">
                <button style="background-color:black; color:white; border:none; padding:10px 20px; border-radius:10px;">
                    TRY NOW
                </button>
            </a>
        </div>
    </div>
""", unsafe_allow_html=True)


