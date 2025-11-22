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
    This initiative explores how machine learning can support smarter transportation planning 
    within Banff National Park by predicting both **traffic congestion** and **parking availability**.
    """)

    st.markdown('---')

    # --- Project Motivation ---
    st.subheader('Project Motivation')
    st.write("""
    Banff is one of Canada’s most popular tourist destinations, drawing millions of visitors each year. 
    As traffic volumes increase, congestion at key routes and entrances has become a significant challenge 
    for both travelers and park management.  
    The goal of this initiative is to create a **decision-support tool** that helps stakeholders 
    monitor, visualize, and forecast traffic conditions to improve visitor flow and reduce bottlenecks.
    """)

    # --- Application Overview ---
    st.subheader('Application Overview')
    st.write("""
    The dashboard consists of three main components:
    - **Traffic Delay Predictor** – A machine learning–powered page that estimates the probability 
      of congestion and predicts route-specific delays.  
    - **Traffic Analysis** – An interactive visualization page for exploring historical 
      route data, delay distributions, and seasonal trends.  
    - **About the Project** – A reference page explaining the data sources, methodology, and team behind the app.
    """)

    # --- Technical Overview ---
    st.subheader('Technical Overview')
    st.write("""
    The project leverages historical traffic data from Banff’s monitored routes, 
    including metrics such as **speed**, **travel time**, **current delay**, and **volume trends**.

    Two main models were developed:
    - **Delay Risk Classifier** – a probabilistic model that estimates the likelihood of  
      *no delay*, *minor delay*, or *major congestion* using temporal and seasonal variables.  
    - **Per-Route Delay Regressor** – a route-level model trained with lag and rolling window 
      features to predict the expected delay duration in minutes.

    Both models were implemented in Python using **pandas**, **scikit-learn**, and **XGBoost**, 
    with deployment via **Streamlit** for interactive visualization.
    """)

    # --- Workflow ---
    st.subheader('Machine Learning Workflow')
    st.write("""
    The project followed the **CRISP-DM** process:
    1. **Data Understanding:** Examined over a year of traffic data from multiple Banff routes.  
    2. **Data Preparation:** Cleaned, merged, and feature-engineered temporal and rolling statistics.  
    3. **Modeling:** Trained and evaluated Random Forest and XGBoost models per route.  
    4. **Evaluation:** Compared metrics using Mean Absolute Error (MAE) and R² to assess accuracy.  
    5. **Deployment:** Integrated the final models into an interactive Streamlit application.
    """)

    # --- Team Members ---
    st.subheader('Team Alpine Analysts')
    st.write("""
    - **Angela Lekivetz**  
    - **Christine Joyce Moraleja**
    - **Victoriia Biaragova**
    - **Sirjana Chauhan**
    """)

    # --- Acknowledgment ---
    st.markdown('---')
    st.caption("""
    Developed as part of **CMPT 3835 – Work Integrated Learning 2**  
    NorQuest College · Fall 2025 · Instructors: Uchenna Mgbaja, Palwasha Afsar
    """)

if __name__ == '__main__':
    main()
