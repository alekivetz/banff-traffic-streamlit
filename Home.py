import streamlit as st
from google.oauth2 import service_account
from utils.display_images import display_banner

creds = service_account.Credentials.from_service_account_info(
    dict(st.secrets["gcp_service_account"]),
    scopes=["https://www.googleapis.com/auth/drive.readonly"]
)
st.success("✅ Google service account loaded successfully!")

def main():
    st.set_page_config(
        page_title='Banff Traffic Management',
        page_icon='🚗',
        layout='wide',
    )

    # --- Banner ---
    display_banner()

    # --- Header ---
    st.title('Alpine Analysts: Banff Traffic Management')
    st.write("""
    Welcome to the **Banff Traffic Management** system, a machine learning-powered dashboard 
    that helps analyze and forecast traffic conditions and parking availability 
    across the town of Banff.
    """)

    st.markdown('---')
    st.subheader('Project Overview')
    st.write("""
    This tool uses real traffic data from Banff National Park to estimate:

    * **Overall delay risk probabilities** 
    * **Route-specific delay duration**
    * **Visualization and analysis** of travel times and delay patterns
    """)

    st.info("""
    Use the **sidebar** to navigate the application:

    * **Traffic Delay Predictor** – explore overall congestion risk and 
      route-specific delay predictions
    * **Traffic Analysis** - Visualize historical route data to uncover patterns
    * **About the Project** – learn about the data sources, modeling approach, and the Alpine Analysts team
    """)

    st.markdown('---')
    st.caption('Created by Alpine Analysts · NorQuest College MLAD · Fall 2025')

if __name__ == '__main__':
    main()
