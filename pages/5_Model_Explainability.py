import streamlit as st
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.display_images import display_banner
from utils.data_loader import init_app_state


# --- PAGE CONFIG ---
st.set_page_config(page_title='Model Explainability', page_icon='🧠', layout='wide')
display_banner()

st.title('Model Explainability')
st.write('''
This page illustrates **how each machine learning model makes its predictions** by highlighting the most influential temporal, spatial, and behavioral factors driving congestion and parking demand across Banff.
''')

st.markdown('---')

# --- LOAD DATA & MODELS ---
init_app_state()
df = st.session_state.routes_df
clf = st.session_state.classifier
regressors = st.session_state.regressors


# --- HELPER: Compute SHAP values ---
def compute_shap(model, X):
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # Handle classifier vs regressor shape
    if len(shap_values.values.shape) == 3:
        mean_abs = np.mean(np.abs(shap_values.values), axis=2)
    else:
        mean_abs = np.abs(shap_values.values)

    shap_df = (
        pd.DataFrame({
            'Feature': X.columns,
            'Mean |SHAP value|': np.mean(mean_abs, axis=0)
        })
        .sort_values('Mean |SHAP value|', ascending=False)
        .reset_index(drop=True)
    )
    return shap_values, shap_df


# --- GLOBAL CLASSIFIER ---
st.header('1. Global Delay Classifier')

expected_cols = ['route', 'season', 'hour', 'day_of_week', 'is_weekend', 'month']
clf_sample = df.copy()

# add 'season' if missing
if 'month' in clf_sample.columns and 'season' not in clf_sample.columns:
    clf_sample['season'] = clf_sample['month'].map({
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'fall', 10: 'fall', 11: 'fall'
    })

clf_sample = clf_sample[[c for c in expected_cols if c in clf_sample.columns]].dropna()
clf_sample = clf_sample.sample(min(500, len(clf_sample)), random_state=42)

# unwrap pipeline if needed
if hasattr(clf, 'named_steps') and 'model' in clf.named_steps:
    model = clf.named_steps['model']
    preproc = clf.named_steps['preprocessor']
    X_trans = preproc.transform(clf_sample)
    feature_names = preproc.get_feature_names_out()
    X_df = pd.DataFrame(X_trans, columns=feature_names)
else:
    model = clf
    X_df = clf_sample.select_dtypes(include=['number', 'bool'])

# compute SHAP
shap_values, shap_df = compute_shap(model, X_df)

# layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader('SHAP Summary Plot')
    fig, _ = plt.subplots(figsize=(7, 4))
    shap.summary_plot(shap_values, X_df, show=False)
    st.pyplot(fig)
    plt.clf()

with col2:
    st.subheader('Top Feature Importances')
    st.dataframe(shap_df.head(10), hide_index=True, width='stretch')

st.markdown('''
### Interpretation
The global classifier primarily depends on **temporal** and **seasonal** signals, indicating that congestion patterns are strongly tied to **when** and **where** vehicles travel.  

- **`num__hour`** - the most influential feature, reflecting predictable daily traffic peaks during morning and evening hours.  
- **`num__month`** and **`cat__season_summer`** - capture broader **seasonal variations**, where summer months bring increased visitor volumes and heavier congestion.  
- **Route indicators** (e.g., `cat__route_Route 4`, `cat__route_Route 6`) - show that some routes consistently experience higher congestion, contributing spatial context to predictions.  
- **`num__day_of_week`** and **`is_weekend`** - differentiate between weekday commuter traffic and weekend tourism-driven flows.  

''')

st.markdown('---')


# --- PER-ROUTE REGRESSORS ---
st.header('2. Per-Route Delay Regressors')
st.write('''
    This model predicts **short-term route-level delays** by learning how real-time speed and travel time trends relate to congestion buildup.
''')

# --- Route selector ---
available_routes = sorted(regressors.keys(), key=lambda x: int(x.split()[-1]))
selected_route = st.selectbox('Select a route', available_routes)

# --- Load model & data for selected route ---
model = regressors.get(selected_route)
route_df = (
    df[df['route'] == selected_route]
    .drop(columns=['timestamp', 'route'], errors='ignore')
    .select_dtypes(include=['number', 'bool'])
    .tail(100)
)

if model is None or route_df.empty:
    st.warning(f'Model or data missing for {selected_route}.')
else:
    # Align columns to model expectations
    if hasattr(model, 'feature_names_in_'):
        X_aligned = route_df.reindex(columns=model.feature_names_in_, fill_value=0)
    else:
        X_aligned = route_df.copy()

    # Compute SHAP values
    shap_values, shap_df = compute_shap(model, X_aligned)

    # Compute model importances (if available)
    if hasattr(model, 'feature_importances_'):
        feature_names = getattr(model, 'feature_names_in_', X_aligned.columns)
        rf_imp = pd.DataFrame({
            'Feature': feature_names,
            'Model Importance': model.feature_importances_[:len(feature_names)]
        })
    else:
        rf_imp = pd.DataFrame({'Feature': X_aligned.columns, 'Model Importance': np.nan})

    # --- Merge safely ---
    table = (
        pd.merge(shap_df, rf_imp, on='Feature', how='left')
        .sort_values('Mean |SHAP value|', ascending=False)
        .round(5)
        .drop_duplicates(subset='Feature')
    )

    # --- Layout ---
    st.subheader(f'{selected_route}')
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('**SHAP Summary Plot**')
        fig, _ = plt.subplots(figsize=(7, 4))
        shap.summary_plot(shap_values, X_aligned, show=False)
        st.pyplot(fig)
        plt.clf()

    with col2:
        st.markdown('**Top Feature Importances**')
        st.dataframe(table.head(10), hide_index=True, width='stretch')

    # --- Interpretation ---
    st.markdown('''
    ### Interpretation
    Each route-specific regressor focuses on **direct traffic flow indicators**, highlighting how local dynamics drive short-term delay forecasts.  

    - **`mean_travel_time`** and **`speed_kmh`** - the dominant predictors, capturing real-time congestion intensity.  
    - **`delay_lag_1`** and **`travel_lag_1`** - represent immediate historical effects, showing persistence in traffic conditions.  
    - **Rolling averages** and **trend indicators** - provide context for gradual buildup or easing of congestion.  
    ''')

st.markdown('---')

    # --- PARKING DEMAND REGRESSOR ---
st.header('3. Parking Availability Regressor')
st.write('''
This model predicts **parking occupancy and demand trends** by analyzing recent occupancy patterns and time-based variables.
''')

parking_model = st.session_state.parking_model
parking_df = st.session_state.parking_df

# --- Prepare numeric data ---
numeric_cols = parking_df.select_dtypes(include=['number', 'bool']).columns.tolist()
parking_sample = parking_df[numeric_cols].dropna().tail(100)

# Align to model features if needed
if hasattr(parking_model, 'feature_names_in_'):
    X_parking = parking_sample.reindex(columns=parking_model.feature_names_in_, fill_value=0)
else:
    X_parking = parking_sample.copy()

# --- Compute SHAP (cached + fast) ---
@st.cache_resource(show_spinner=False)
def get_shap_for_parking(_model, X):
    try:
        explainer = shap.TreeExplainer(_model)
    except Exception:
        explainer = shap.Explainer(_model)

    shap_values = explainer(X)
    if len(shap_values.values.shape) == 3:
        mean_abs = np.mean(np.abs(shap_values.values), axis=2)
    else:
        mean_abs = np.abs(shap_values.values)

    shap_df = (
        pd.DataFrame({
            'Feature': X.columns,
            'Mean |SHAP value|': np.mean(mean_abs, axis=0)
        })
        .sort_values('Mean |SHAP value|', ascending=False)
        .reset_index(drop=True)
    )
    return shap_values, shap_df

p_shap, p_shap_df = get_shap_for_parking(parking_model, X_parking)

# --- Compute model importances (if available) ---
if hasattr(parking_model, 'feature_importances_'):
    feats = getattr(parking_model, 'feature_names_in_', X_parking.columns)
    rf_imp = pd.DataFrame({
        'Feature': feats,
        'Model Importance': parking_model.feature_importances_[:len(feats)]
    })
else:
    rf_imp = pd.DataFrame({'Feature': X_parking.columns, 'Model Importance': np.nan})

# Safe merge
p_table = (
    pd.merge(p_shap_df, rf_imp, on='Feature', how='left')
    .sort_values('Mean |SHAP value|', ascending=False)
    .round(5)
    .drop_duplicates(subset='Feature')
)

# --- Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('**SHAP Summary Plot**')
    fig, _ = plt.subplots(figsize=(7, 4))
    shap.summary_plot(p_shap, X_parking, show=False)
    st.pyplot(fig)
    plt.clf()

with col2:
    st.markdown('**Top Feature Importances**')
    st.dataframe(p_table.head(10), hide_index=True, width='stretch')

# --- Interpretation ---
st.markdown('''
### Interpretation
The parking demand regressor relies most heavily on **current occupancy** and **recent lag features**, showing that short-term parking behavior is primarily driven by how full the lots already are.  

- **`occupancy`** - the strongest predictor, reflecting the direct carry-over of current lot fill levels into short-term forecasts.  
- **`lag_1`** and **`lag_2`** - capture the short-term momentum in parking activity (how quickly lots are filling or clearing).  
- **`hour`** and **`month`** - capture recurring **daily and seasonal demand cycles**, such as midday peaks and higher summer usage.  
- **`max_capacity`** - keeps predictions realistic by accounting for the physical capacity limits of each lot.  

''')


# --- Footer ---
st.markdown('---')
st.caption("""
    Data Source: Banff Traffic Management Project (2025)
           
    Created by Alpine Analysts · NorQuest College MLAD · Fall 2025
""")


