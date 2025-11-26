import streamlit as st
import pandas as pd
import shap
import joblib
import io
import json
import matplotlib.pyplot as plt

from utils.display_images import display_banner
from utils.data_loader import download_from_drive

# --- Page Setup ---
st.set_page_config(page_title='Model Explainability', page_icon='🔍', layout='wide')
display_banner()

st.title('Model Explainability')
st.write('''
This page provides insights into how each per-route Random Forest model makes predictions.
It uses SHAP (SHapley Additive exPlanations) to visualize which input features most strongly influence
predicted traffic delays.
''')

st.markdown('---')

# --- Load Models ---
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
def fetch_parquet_from_drive():
    file_id = st.secrets['ROUTES_WITH_LAGS_ID']
    data_bytes = download_from_drive(file_id)
    df = pd.read_parquet(io.BytesIO(data_bytes), engine='fastparquet')
    return df


# --- Load resources ---
regressors = fetch_regressors()
df = fetch_parquet_from_drive()

# --- Route selection ---
route = st.selectbox('Select Route to Explain:', sorted(regressors.keys(), key=lambda x: int(x.split()[-1])))

model = regressors.get(route)
route_df = df[df['route'] == route].copy()

if model is None:
    st.error(f'No model found for {route}')
else:
    st.success(f'Model loaded for {route}')

    # --- Sample data ---
    sample_size = st.slider('Number of recent samples to explain:', min_value=50, max_value=500, value=200, step=50)
    sample = route_df.drop(columns=['timestamp', 'route']).tail(sample_size)
    st.write(f'Explaining last {len(sample)} samples.')

    # --- SHAP computation ---
    with st.spinner('Computing SHAP values...'):
        explainer = shap.Explainer(model)
        shap_values = explainer(sample)

    # --- Global Feature Importance ---
    st.subheader('Global Feature Importance')
    fig, _ = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, sample, show=False)
    st.pyplot(fig)
    plt.clf()

    # --- Per-instance Force Plot ---
    st.subheader('Individual Prediction Explanation')
    row_idx = st.slider('Select sample index to inspect:', 0, len(sample) - 1, 0)
    shap_html = shap.force_plot(
        explainer.expected_value,
        shap_values.values[row_idx, :],
        sample.iloc[row_idx, :],
        matplotlib=False
    )
    st.components.v1.html(shap.getjs() + shap_html.html(), height=250)
