import streamlit as st
import pandas as pd
import joblib
import io

from utils.display_images import display_banner
from utils.google_drive_helpers import download_from_drive

# --- Load models and data ---
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

try: 
    model, unit_encoder, features_df = load_parking_data()
except Exception as e:
    st.error(f'Failed to load parking data from Google Drive ({e}).')
    st.stop()

# --- UI Setup ---
st.set_page_config(page_title='Parking Availability Predictor', page_icon='⏱️', layout='wide')
display_banner()

# --- Header ---
st.title('Parking Availability Predictor - 60 Minute Forecast')
st.write("""
This interactive dashboard uses a machine learning model to **forecast parking occupancy 60 minutes ahead** 
for various lots in the town of Banff, helping visitors and planners manage congestion more effectively.
""")
st.markdown('---')  

st.write('Select a parking lot to view its predicted occupancy 60 minutes ahead.')  

# --- Select Parking Lot ---
units = sorted(features_df['unit'].unique())    
selected_unit = st.selectbox('Select Parking Lot', units)

unit_data = features_df[features_df['unit'] == selected_unit].copy()
if unit_data.empty:
    st.error('No data available for this parking lot.')
    st.stop()

unit_data = unit_data.sort_values('ts')
latest_row = unit_data.iloc[-1]
st.write(f'Using latest data timestamp for this los: **{latest_row["ts"]}**')

# --- Prepare Features ---
if 'unit_encoded' not in unit_data.columns:
    unit_data['unit_encoded'] = unit_encoder.transform(unit_data['unit'])
    latest_row = unit_data.iloc[-1] 

FEATURE_COLS = [
    'occupancy', 'max_capacity',
    'lag_1', 'lag_2', 'lag_3', 'lag_4',
    'roll_mean_4', 'roll_mean_8',
    'hour', 'dayofweek', 'is_weekend', 'month',
    'unit_encoded'
]

X_current = pd.DataFrame([latest_row[FEATURE_COLS].to_dict()])

# --- Predict ---
pred_occ = int(model.predict(X_current)[0])
pred_occ = max(0, min(pred_occ, int(latest_row['max_capacity'])))

available = int(round(latest_row['max_capacity'] - pred_occ))

# --- Display ---
st.subheader(f'📍 {selected_unit}')
st.metric('Current occupancy', f'{int(latest_row["occupancy"])}/{int(latest_row["max_capacity"])}')

st.subheader('⏳ Forecast for 60 minutes ahead')
st.metric('Predicted occupancy in 60 min', f'{pred_occ}/{int(latest_row["max_capacity"])}')
st.metric('Estimated free spaces in 60 min', available)

if available <= 0:
    st.error('🔴 Lot will likely be FULL in 60 minutes.')
elif available < latest_row['max_capacity'] * 0.1:
    st.warning('🟡 Lot will be almost full in 60 minutes.')
else:
    st.success('🟢 Spaces should be available in 60 minutes.')

st.caption('Forecasts generated using an XGBoost regression model trained on 15-minute occupancy intervals.')

# --- Footer ---
st.markdown('---')
st.caption("""
    Data Source: Banff Traffic Management Project (2025)
           
    Created by Alpine Analysts · NorQuest College MLAD · Fall 2025
""")
