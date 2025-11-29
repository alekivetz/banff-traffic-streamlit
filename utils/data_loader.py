import io
import joblib
import json
import streamlit as st
import pandas as pd

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

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

# --- Generic Parquet Data Loader ---
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_parquet(file_id):

    try:
        data_bytes = download_from_drive(file_id)
        df = pd.read_parquet(io.BytesIO(data_bytes), engine='fastparquet')
    except Exception as e:
        st.warning(f'Could not load from Google Drive ({e}).')
    return df

# --- Load Datasets ---
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_routes_model():
    """Routes dataset for ML modeling."""
    return fetch_parquet(st.secrets['ROUTES_WITH_LAGS_ID'])


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_routes_vis_chatbot():
    """Routes dataset shared by visualization dashboards and chatbot."""
    df = fetch_parquet(st.secrets['ROUTES_VIS_ID'])
    if not df.empty and 'calculation time' in df.columns:
        df['calculation time'] = pd.to_datetime(df['calculation time'], errors='coerce')
        df = df.dropna(subset=['calculation time'])
        df['route'] = df['route'].astype(str)
        df['hour'] = df['calculation time'].dt.hour
        df['day_of_week'] = df['calculation time'].dt.day_name()
        df['month'] = df['calculation time'].dt.strftime('%B')
    return df

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_parking_vis():
    """Parking dataset for visual dashboards."""
    return fetch_parquet(st.secrets['PARKING_VIS_ID'])


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_parking_chatbot():
    """Merged parking dataset for chatbot analysis."""
    return fetch_parquet(st.secrets['PARKING_CHATBOT_ID'])


# --- Model Loaders ---
@st.cache_resource(show_spinner=False)
def fetch_classifier():
    file_id = st.secrets['CLASSIFIER_MODEL_ID']
    model_bytes = download_from_drive(file_id)
    return joblib.load(io.BytesIO(model_bytes))


@st.cache_resource(show_spinner=False)
def fetch_regressors():
    route_dict = json.loads(st.secrets['ROUTE_MODELS_IDS'])
    models = {}
    for route, file_id in route_dict.items():
        try:
            model_bytes = download_from_drive(file_id)
            model = joblib.load(io.BytesIO(model_bytes))
            models[route] = model
        except Exception as e:
            st.warning(f'Could not load model for {route}: {e}')
    return models


@st.cache_data(ttl=86400, show_spinner=False)
def load_parking_resources():
    """Load the parking model, encoder, and modeling dataset from Google Drive."""
    try:
        model = joblib.load(io.BytesIO(download_from_drive(st.secrets['PARKING_MODEL_ID'])))
        encoder = joblib.load(io.BytesIO(download_from_drive(st.secrets['PARKING_ENCODER_ID'])))
        df = fetch_parquet(st.secrets['PARKING_DATA_ID'])
        return model, encoder, df
    except Exception as e:
        st.warning(f'Failed to load parking resources: {e}')
        return None, None, pd.DataFrame()


# --- App Initialization ---
def init_app_state():
    """Initialize and store data/models in session_state."""
    with st.spinner('Initializing Banff Traffic Management project...'):
        # ROUTES DATASETS
        st.session_state.routes_df_model = st.session_state.get('routes_df_model') or fetch_routes_model()
        st.session_state.routes_df_vis_chatbot = (
            st.session_state.get('routes_df_vis_chatbot') or fetch_routes_vis_chatbot()
        )
        # ROUTES MODELS
        st.session_state.classifier = st.session_state.get('classifier') or fetch_classifier()
        st.session_state.regressors = st.session_state.get('regressors') or fetch_regressors()  

        # PARKING DATASETS
        st.session_state.parking_df_vis = st.session_state.get('parking_df_vis') or fetch_parking_vis()
        st.session_state.parking_df_chatbot = st.session_state.get('parking_df_chatbot') or fetch_parking_chatbot()

        # PARKING MODEL + ENCODER
        if 'parking_df_model' not in st.session_state:
            model, encoder, df = load_parking_resources()
            st.session_state.parking_model = model
            st.session_state.parking_encoder = encoder
            st.session_state.parking_df_model = df

    st.success('All datasets and models loaded successfully!')
