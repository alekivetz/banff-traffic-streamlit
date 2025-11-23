import streamlit as st
import pandas as pd
import joblib
import io
import requests
import json
from datetime import datetime

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

from utils.display_images import display_image, display_banner
from utils.route_info import ROUTE_DESCRIPTIONS

# --- Connect to Google Drive ---
def connect_drive():
    """Authenticate with Google Drive using the service account."""
    creds = service_account.Credentials.from_service_account_info(
        dict(st.secrets['gcp_service_account']),
        scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
    return build('drive', 'v3', credentials=creds)

def download_from_drive(file_id):
    """Download a file from Google Drive."""
    drive = connect_drive()
    request = drive.files().get_media(fileId=file_id)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    buffer.seek(0)
    return buffer.read()

# --- Load models and data ---
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_parquet_from_drive():
    """Download and cache the Parquet data file from Google Drive."""
    file_id = st.secrets['ROUTES_WITH_LAGS_ID']
    try:
        data_bytes = download_from_drive(file_id)
        return pd.read_parquet(io.BytesIO(data_bytes), engine='fastparquet')
    except Exception as e:
        st.warning(f'Could not load from Google Drive ({e}). Loading local copy instead.')
        return pd.read_parquet('data/routes_with_lags.parquet', engine='fastparquet')
    

@st.cache_resource(show_spinner=False)
def fetch_classifier():
    """Download and cache the classifier model from Drive."""   
    file_id = st.secrets['CLASSIFIER_MODEL_ID']
    try:
        model_bytes = download_from_drive(file_id)
        model = joblib.load(io.BytesIO(model_bytes))
        return model
    except Exception as e:
        st.error(f'Failed top load classifier from Drive ({e}).')
        raise


@st.cache_resource(show_spinner=False)
def fetch_regressors():
    """Download and cache all per-route models from Drive."""
    route_dict = json.loads(st.secrets['ROUTE_MODELS_IDS'])
    models = {}
    for route, file_id in route_dict.items():
        try:
            model_bytes = download_from_drive(file_id)
            models[route] = joblib.load(io.BytesIO(model_bytes))
        except Exception as e:
            st.warning(f'Could not load model for {route}: {e}')
    return models


def get_models():
    """Load classifier and regressors once per session."""  
    if 'classifier' not in st.session_state:    
        with st.spinner('Loading models...'):
            st.session_state.classifier = fetch_classifier()
            st.session_state.regressors = fetch_regressors()
    return st.session_state.classifier, st.session_state.regressors


def load_data():
    """Load the data once per session (avoid refetching on widget changes)."""
    if 'routes_df' not in st.session_state:
        with st.spinner('Loading dataset...'):
            df = fetch_parquet_from_drive()
            if 'timestamp' in df.columns and df['timestamp'].dtype == 'object':
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            st.session_state.routes_df = df
    return st.session_state.routes_df


clf, regressors = get_models()
df = load_data()

st.success('Models and data loaded successfully.')

# --- UI Setup ---
st.set_page_config(page_title='Banff Delay Predictor', page_icon='⏱️', layout='wide')
display_banner()

# --- Header ---
st.title('Banff Traffic Delay Predictor')
st.write("""
    Welcome to the **Banff Traffic Delay Predictor**, an interactive dashboard that leverages machine learning 
    to **analyze and forecast traffic delays** across Banff National Park.  
""")

st.subheader('Model Overview')
st.info("""

    1. **Overall Delay Classifier**: A probabilistic model estimating the likelihood of **no delay**, **minor delay**, or **major congestion** using temporal and seasonal patterns across all routes.

    2. **Per-Route Delay Regressor**: A route-specific regression model trained on historical data, including **lag** and **rolling window** features, to predict the **expected delay duration (in minutes)** for a chosen route.
""")

st.markdown('---')
st.subheader('Select Prediction Parameters')

# --- Input Controls ---
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    route = st.selectbox('Select Route:', sorted(ROUTE_DESCRIPTIONS.keys(), key=lambda x: int(x.split()[-1])))
    date_input = st.date_input('Select Date:')
    time_input = st.time_input('Select Time:')
with col2: 
    with st.expander('View Route Descriptions', expanded=False):
        for r, desc in ROUTE_DESCRIPTIONS.items():
            st.write(f'**{r}** - {desc}')
with col3:
    image = display_image('banff_map.png')
    st.image(image, width=400)

st.markdown('---')

# --- Models ---
if st.button('Predict Delay'):
    dt = datetime.combine(date_input, time_input)

    col1, col2 = st.columns([1, 1])

    # --- Left: Classifier
    with col1:
        st.header('Overall Congestion Risk (Classifier)')

        clf_features = pd.DataFrame([{
            'route': route,
            'hour': dt.hour,
            'month': dt.month,
            'day_of_week': dt.weekday(),
            'is_weekend': int(dt.weekday() in [5, 6]),
            'season': (
                'winter' if dt.month in [12, 1, 2] else
                'spring' if dt.month in [3, 4, 5] else
                'summer' if dt.month in [6, 7, 8] else 'fall')
        }])

        probs = clf.predict_proba(clf_features)[0]
        no_delay, minor_delay, major_delay = probs

        st.subheader('Predicted Delay Probabilities')
        st.write(f'🟢 No Delay: {no_delay:.2%}')
        st.progress(no_delay)
        st.write(f'🟡 Minor Delay: {minor_delay:.2%}')
        st.progress(minor_delay)
        st.write(f'🔴 Major Delay: {major_delay:.2%}')
        st.progress(major_delay)

        st.subheader('Overall Risk Assessment')
        if major_delay > 0.6:
            st.error('🔴 **High chance of major congestion.**')
        elif minor_delay > 0.6 or major_delay > 0.3:
            st.warning('🟡 **Moderate congestion expected.**')
        else:
            st.success('🟢 **Low congestion expected.**')

    # --- Right: Regressor ---
    with col2:
        st.header('Route-Specific Delay (Regressor)')

        dt = datetime.combine(date_input, time_input)
        route_df = df[df['route'] == route].copy().sort_values('timestamp')
        route_df = route_df[route_df['timestamp'] <= dt]

        if route_df.empty:
            st.error('No available past data for this route and time.')
        else:
            closest_row = route_df.iloc[-1]
            regressor = regressors.get(route)

            if regressor is None:
                st.error(f'No trained model found for {route}.')
            else:
                X_new = pd.DataFrame([closest_row])[regressor.feature_names_in_]
                pred = regressor.predict(X_new)[0]

                st.success(f'Predicted Delay: {pred:.2f} minutes')

st.markdown('---')
st.caption("""
    Data Source: Banff Traffic Management Project (2025)
    Created by Alpine Analysts · NorQuest College MLAD · Fall 2025
""")

