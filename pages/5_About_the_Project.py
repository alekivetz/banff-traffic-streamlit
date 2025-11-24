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
    The interactive dashboard now consists of **four integrated components**, each designed 
    to address a different aspect of Banff’s mobility system:
    
    - **Traffic Delay Predictor** – A machine learning–powered module that estimates overall 
      congestion risk and predicts per-route delay durations.  
    - **Traffic Analysis** – An interactive data visualization page for exploring 
      route performance, speed trends, and seasonal delay patterns.  
    - **Parking Availability Predictor** – A predictive model that estimates **lot occupancy 60 minutes ahead**, 
      helping to anticipate demand and prevent overcrowding.  
    - **Parking Analysis** – A detailed analytical dashboard providing insights into 
      **sessions, revenue, payment methods, and duration trends** across all parking units.  
    """)

    # --- Technical Overview ---
    st.subheader('Technical Overview')
    st.write("""
    The Banff Traffic Management system integrates several predictive and analytical components 
    developed using **Python**, **pandas**, **scikit-learn**, **XGBoost**, and **Streamlit**.  
    Each model focuses on a specific aspect of transportation management:

    • **Traffic Delay Models** – Predict both the probability of congestion and the expected delay duration 
      for each monitored route. These models incorporate temporal patterns, rolling averages, and historical 
      traffic metrics to deliver accurate short-term forecasts.

    • **Parking Occupancy Forecast** – Uses a regression-based XGBoost model trained on 15-minute interval data 
      to estimate lot occupancy one hour into the future.  

    All models were integrated into a single Streamlit application that provides interactive filtering, 
    live forecasting, and data visualizations.
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
