import streamlit as st
from utils.display_images import display_banner

def main():
    st.set_page_config(
        page_title='About the Project',
        page_icon='📘',
        layout='wide',
    )

    # --- Banner ---
    display_banner()

    # --- Header ---
    st.title('About the Banff Traffic Management Project')
    st.write("""
    The **Banff Traffic Management** project was developed by **Team Alpine Analysts** 
    as part of the Machine Learning Analyst Diploma program at NorQuest College (Fall 2025).  
    The project applies **machine learning and data visualization** to analyze and forecast **traffic congestion** 
    and **parking availability** across the town of Banff.

    The system integrates multiple predictive and analytical modules into a single Streamlit interface, 
    transforming real transportation data into actionable insights for smarter mobility management.
    """)

    st.markdown('---')

    # --- Project Motivation ---
    st.subheader('Project Motivation')
    st.write("""
    Banff is one of Canada’s most popular tourist destinations, drawing millions of visitors each year. 
    As traffic volumes increase, congestion at key routes and entrances has become a significant challenge 
    for both travelers and park management.  
             
    The goal of this project is to create a **decision-support platform** that helps stakeholders:
    - Monitor and forecast real-time route conditions.  
    - Anticipate congestion patterns before they occur.  
    - Analyze parking demand, usage, and revenue trends.  
    - Support sustainable mobility through data-driven insights.
    """)

    st.markdown('---')

    # --- Application Overview ---
    st.subheader('Application Overview')
    st.write("""
    The interactive dashboard now consists of **four integrated components**, each designed 
    to address a different aspect of Banff’s mobility system:
    """)

    st.markdown("""
    | Page | Description |
    |------|-------------|
    | **1. Traffic Delay Predictor** | Machine learning–powered model that estimates congestion probability and predicts per-route delay durations. |
    | **2. Traffic Analysis Dashboard** | Interactive visualizations of historical speed and delay trends across Banff’s main routes. |
    | **3. Parking Availability Predictor** | Predicts parking lot occupancy 60 minutes into the future using an XGBoost regression model trained on 15-minute intervals. |
    | **4. Parking Analysis Dashboard** | Aggregates parking session data to visualize sessions, revenue, duration, and demand by time and location. |
    """)

    st.markdown('---')

    # --- Technical Overview ---
    st.subheader('Technical Overview')
    st.write("""
    The Banff Traffic Management System combines predictive modeling with interactive visualization.  
    It was developed entirely in **Python**, using:

    - **Machine Learning:** scikit-learn, XGBoost, Random Forest Regressor 
    - **Data Processing:** pandas, NumPy  
    - **Visualization:** Plotly, Streamlit  
    - **Data Storage:** Google Drive (private data hosting)  
    - **Deployment:** Streamlit Cloud  
    """)

    st.write("""
    **Core Models:**
    - **Delay Risk Classifier** – Predicts whether a route will experience *no delay*, *minor delay*, or *major congestion*.  
    - **Per-Route Delay Regressor** – Forecasts the expected delay duration (minutes) per route using temporal and lag-based features.  
    - **Parking Occupancy Forecaster** – Uses a regression-based XGBoost model to predict 60-minute-ahead lot occupancy.
    """)

    st.markdown('---')

    # --- Workflow ---
    st.subheader('Machine Learning Workflow')
    st.write("""
    The project followed the **CRISP-DM** methodology:

    1. **Data Understanding** – Explored over a year of traffic and parking data from Banff National Park.  
    2. **Data Preparation** – Cleaned and merged datasets, engineered lag and rolling-window features, and derived temporal variables.  
    3. **Modeling** –  
       • *Traffic Models:* Random Forest and XGBoost regressors for route-specific delays and congestion levels.  
       • *Parking Models:* XGBoost regressors to forecast occupancy based on recent usage trends.  
    4. **Evaluation** – Measured performance using MAE, RMSE, and R² for regression; precision and recall for classification.  
    5. **Deployment** – Integrated all models into a unified Streamlit dashboard for real-time prediction and visualization.
    """)

    st.markdown('---')

    # --- Team Members ---
    st.subheader('Team Alpine Analysts')
    st.write("""
    - **Angela Lekivetz** – Data Analysis · Model Development · Streamlit Integration
    - **Christine Joyce Moraleja** – Data Cleaning · Documentation · Project Coordination
    - **Victoriia Biaragova** – Model Development · Feature Engineering · Dashboard Design
    - **Sirjana Chauhan** – Model Evaluation · Visualization · Testing
    """)

    # --- Acknowledgment ---
    st.markdown('---')
    st.caption("""
    Developed as part of **CMPT 3835 – Work Integrated Learning 2**  
    
    NorQuest College · Fall 2025 · Instructors: Uchenna Mgbaja, Palwasha Afsar
    """)

if __name__ == '__main__':
    main()
