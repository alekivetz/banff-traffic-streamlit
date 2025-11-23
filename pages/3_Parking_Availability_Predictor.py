import streamlit as st
import pandas as pd
import joblib

from utils.display_images import display_banner

# --- UI Setup ---
st.set_page_config(page_title='Parking Availability Predictor', page_icon='⏱️', layout='wide')
display_banner()

# --- Header ---
st.title('Parking Availability Predictor - 60 Minute Forecast')
st.write("""
    Welcome to the **Banff Parking Availability Predictor**, an interactive dashboard that leverages machine learning 
    to **analyze and forecast traffic delays** across Banff National Park.  
""")

