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
st.write("""
This page illustrates **how each machine learning model makes its predictions** by highlighting the most influential temporal, spatial, and behavioral factors driving congestion and parking demand across Banff.
""")
st.markdown('---')

# --- LOAD DATA & MODELS ---
df = st.session_state.routes_df_model
clf = st.session_state.classifier
regressors = st.session_state.regressors

# --- HELPER FUNCTIONS ---
def compute_shap_light(model, X, sample_n=100):
    X_sample = X.sample(min(sample_n, len(X)), random_state=42)
    explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent', approximate=True)
    shap_values = explainer(X_sample, check_additivity=False)

    # Always flatten multi-dimensional SHAP outputs
    shap_array = np.abs(shap_values.values)
    if shap_array.ndim > 2:
        shap_array = shap_array.mean(axis=2)
    if shap_array.ndim > 1:
        mean_abs = shap_array.mean(axis=0)
    else:
        mean_abs = shap_array  # already 1D

    shap_df = (
        pd.DataFrame({
            'Feature': X_sample.columns,
            'Mean |SHAP value|': mean_abs.flatten()
        })
        .sort_values('Mean |SHAP value|', ascending=False)
        .reset_index(drop=True)
    )

    return shap_values, shap_df, X_sample


def plot_shap_summary(values, X):
    fig, _ = plt.subplots(figsize=(7, 4))
    shap.summary_plot(values, X, show=False)
    st.pyplot(fig)
    plt.close(fig)

# --- GLOBAL CLASSIFIER ---
st.header('1. Global Delay Classifier')
st.markdown("""
A **Random Forest classifier** categorizes congestion levels into three classes:
**0 = Low**, **1 = Medium**, and **2 = High** based on features such as average speed, delay trends, and temporal indicators.

**Evaluation results (macro averages):**
- Accuracy: 0.70  
- Precision: 0.37  
- Recall: 0.52  
- F1-Score: 0.34  

The model performs well for **low congestion** (97 % precision) and captures **high congestion** with **65 % recall**, though performance is limited for medium delays due to data imbalance.
""")

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

shap_values, shap_df, X_used = compute_shap_light(model, X_df, sample_n=200)

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader('SHAP Summary Plot')
    plot_shap_summary(shap_values, X_used)
with col2:
    st.subheader('Top Feature Importances')
    st.dataframe(shap_df.head(10), hide_index=True, width='stretch')

st.markdown('---')

# --- PER-ROUTE REGRESSORS ---
st.header('2. Per-Route Delay Regressors')
st.markdown("""
Per-route **Random Forest regressors** forecast continuous travel delays using time-based, spatial, and lagged traffic features.

**Average performance:**
- R²: 0.93  
- MAE: 0.004 minutes  
- RMSE: 0.235 minutes  
""")

all_shap_tables, all_shap_values, all_X = [], [], []

for route, model in regressors.items():
    route_df = (
        df[df['route'] == route]
        .drop(columns=['timestamp', 'route'], errors='ignore')
        .select_dtypes(include=['number', 'bool'])
        .tail(100)
    )
    if model is None or route_df.empty:
        continue

    if hasattr(model, 'feature_names_in_'):
        X_aligned = route_df.reindex(columns=model.feature_names_in_, fill_value=0)
    else:
        X_aligned = route_df.copy()

    shap_values, shap_df, X_used = compute_shap_light(model, X_aligned, sample_n=80)
    shap_df['route'] = route
    all_shap_tables.append(shap_df)
    all_shap_values.append(np.abs(shap_values.values))
    all_X.append(X_used)

if all_shap_tables:
    combined_shap = (
        pd.concat(all_shap_tables)
        .groupby('Feature', as_index=False)['Mean |SHAP value|']
        .mean()
        .sort_values('Mean |SHAP value|', ascending=False)
        .reset_index(drop=True)
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('**SHAP Summary Plot (All Routes)**')
        X_all = pd.concat(all_X, axis=0).reset_index(drop=True)
        shap_all = np.concatenate(all_shap_values, axis=0)
        n = min(len(X_all), shap_all.shape[0], 800)
        X_all = X_all.iloc[:n]
        shap_all = shap_all[:n, :X_all.shape[1]]
        plot_shap_summary(shap_all, X_all)
    with col2:
        st.markdown('**Top 10 Feature Importances**')
        st.dataframe(combined_shap.head(10), hide_index=True, width='stretch')

st.markdown('---')

# --- PARKING REGRESSOR ---
st.header('3. Parking Availability Regressor')
st.markdown("""
An **XGBoost regressor** predicts parking occupancy and duration based on hourly transaction data and time-based variables.

**Performance:**
- R²: 0.95  
- MAE: 0.969 hours  
""")

parking_model = st.session_state.parking_model
parking_df = st.session_state.parking_df_model

numeric_cols = parking_df.select_dtypes(include=['number', 'bool']).columns.tolist()
X_parking = parking_df[numeric_cols].dropna().tail(200)
if hasattr(parking_model, 'feature_names_in_'):
    X_parking = X_parking.reindex(columns=parking_model.feature_names_in_, fill_value=0)

p_shap, p_shap_df, X_used = compute_shap_light(parking_model, X_parking, sample_n=150)

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown('**SHAP Summary Plot**')
    plot_shap_summary(p_shap, X_used)
with col2:
    st.markdown('**Top Feature Importances**')
    st.dataframe(p_shap_df.head(10), hide_index=True, width='stretch')

st.markdown('---')
st.caption("""
Data Source: Banff Traffic Management Project (2025)  
Created by Alpine Analysts · NorQuest College MLAD · Fall 2025
""")
