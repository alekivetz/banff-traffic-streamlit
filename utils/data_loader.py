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


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_parquet_from_drive():
    file_id = st.secrets['ROUTES_WITH_LAGS_ID']
    try:
        data_bytes = download_from_drive(file_id)
        df = pd.read_parquet(io.BytesIO(data_bytes), engine='fastparquet')
    except Exception as e:
        st.warning(f'Could not load from Google Drive ({e}). Loading local copy instead.')
        df = pd.read_parquet('data/routes_with_lags.parquet', engine='fastparquet')

    return df


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
def load_parking_data():
    """Load model, encoder, and data from Google Drive."""
    model_bytes = download_from_drive(st.secrets['PARKING_MODEL_ID'])
    encoder_bytes = download_from_drive(st.secrets['PARKING_ENCODER_ID'])
    data_bytes = download_from_drive(st.secrets['PARKING_DATA_ID'])

    model = joblib.load(io.BytesIO(model_bytes))
    unit_encoder = joblib.load(io.BytesIO(encoder_bytes))   
    features_df = pd.read_parquet(io.BytesIO(data_bytes), engine='fastparquet')

    return model, unit_encoder, features_df


def init_app_state():
    """Initialize and store data/models in session_state."""
    if 'routes_df' not in st.session_state:
        with st.spinner('Loading dataset...'):
            st.session_state.routes_df = fetch_parquet_from_drive()
    if 'classifier' not in st.session_state:
        with st.spinner('Loading models...'):
            st.session_state.classifier = fetch_classifier()
            st.session_state.regressors = fetch_regressors()
    if 'parking_model' not in st.session_state:
        with st.spinner('Loading parking resources...'):
            try:
                parking_model, parking_encoder, parking_df = load_parking_data()
                st.session_state.parking_model = parking_model
                st.session_state.parking_encoder = parking_encoder
                st.session_state.parking_df = parking_df
            except Exception as e:
                st.warning(f'Could not load parking resources: {e}')
