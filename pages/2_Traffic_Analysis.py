import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import io

from utils.display_images import display_banner
from utils.google_drive_helpers import download_from_drive

# --- Page Config ---
st.set_page_config(page_title='Banff Traffic Analysis', page_icon='📈', layout='wide')

# --- Header ---
display_banner()
st.title('Banff Traffic Congestion Analysis')
st.write('This interactive dashboard visualizes historical traffic data to uncover delay patterns, peak-hour behavior, and route performance within the town of Banff.')

# --- Top Controls: Parameters & Filters ---
st.markdown('---')
st.subheader('Simulation Parameters and Filters')

col1, col2, col3, col4 = st.columns(4)
with col1:
    base_speed_kmh = st.slider('Base Speed (km/h)', 30, 100, 60, 5)
with col2:
    rush_hour_slowdown = st.slider('Rush Hour Slowdown (%)', 10, 90, 50, 5)
with col3:
    weekend_speedup = st.slider('Weekend Speed Increase (%)', 0, 30, 10, 5)
with col4:
    base_delay = st.slider('Base Delay (minutes)', 0, 20, 5, 1)

st.markdown('')

f1, f2, f3 = st.columns([1.3, 1, 0.6])
with f1:
    st.markdown('**Select Date Range**')
    min_date = datetime(2024, 12, 1).date()
    max_date = datetime(2025, 9, 7).date()
    selected_dates = st.date_input('Select Date Range', value=[min_date, max_date],
                                   min_value=min_date, max_value=max_date,
                                   label_visibility='collapsed')
with f2:
    st.markdown('**Select Routes**')
    available_routes = ['Route 1','Route 2','Route 3','Route 4','Route 5','Route 6',
                        'Route 7','Route 8','Route 10','Route 11','Route 12','Route 13']
    selected_routes = st.multiselect('Select Routes', options=available_routes,
                                     default=available_routes[:3],
                                     label_visibility='collapsed')
with f3:
    st.write('')
    if st.button('Reset All'):
        st.rerun()

st.markdown('---')

# --- Load Data ---
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_routes_vis():
    """Download routes visualization data Google Drive."""
    try:
        file_id = st.secrets['ROUTES_VIS_ID']
        data_bytes = download_from_drive(file_id)
        df = pd.read_parquet(io.BytesIO(data_bytes), engine='fastparquet')

        # --- Parse and clean ---
        df['calculation time'] = pd.to_datetime(df['calculation time'], errors='coerce')
        df = df.dropna(subset=['calculation time'])
        df['route'] = df['route'].astype(str)

        # --- Derive temporal features ---
        df['hour'] = df['calculation time'].dt.hour
        df['day_of_week'] = df['calculation time'].dt.day_name()
        df['month'] = df['calculation time'].dt.strftime('%B')

        return df
    
    except Exception as e:
        st.warning(f'Could not load from Google Drive ({e})')
        return pd.DataFrame()


def get_routes_vis():
    """Load once per user session."""
    if 'df_vis' not in st.session_state:
        with st.spinner('Loading route visualization data...'):
            st.session_state.df_vis = fetch_routes_vis()
    return st.session_state.df_vis

df = get_routes_vis()

# --- Apply Filters ---
filtered_df = df[
    (df['calculation time'].dt.date >= selected_dates[0]) &
    (df['calculation time'].dt.date <= selected_dates[1]) &
    (df['route'].isin(selected_routes))
]

# --- 1. Average Speed by Route and Hour ---
st.subheader('1. Average Speed by Route and Hour of Day')
heatmap_data = filtered_df.pivot_table(index='route', columns='hour', values='speed(km/h)', aggfunc='mean')
fig1 = px.imshow(heatmap_data, labels=dict(x='Hour of Day', y='Route', color='Speed (km/h)'),
                 x=heatmap_data.columns, y=heatmap_data.index, aspect='auto',
                 color_continuous_scale='RdYlGn_r', title='Average Speed by Route and Hour of Day')
fig1.update_xaxes(side='bottom')
st.plotly_chart(fig1, width='stretch')

st.markdown('---')

# --- 2. Daily Delay Trends ---
st.subheader('2. Daily Delay Trends by Route')
fig2 = px.line(filtered_df.groupby(['route', pd.Grouper(key='calculation time', freq='D')])['actual delay (mins)'].mean().reset_index(),
               x='calculation time', y='actual delay (mins)', color='route',
               title='Average Daily Delay by Route',
               labels={'calculation time': 'Date', 'actual delay (mins)': 'Average Delay (minutes)'})
st.plotly_chart(fig2, width='stretch')

st.markdown('---')

# --- 3. Delay Distribution by Day ---
st.subheader('3. Delay Distribution by Day of Week')
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
filtered_df.loc[:, 'day_of_week'] = pd.Categorical(filtered_df['day_of_week'], categories=weekday_order, ordered=True)
fig3 = px.box(filtered_df, x='day_of_week', y='actual delay (mins)', color='day_of_week',
              title='Delay Distribution by Day of Week',
              category_orders={'day_of_week': weekday_order},
              labels={'day_of_week': 'Day of Week', 'actual delay (mins)': 'Delay (minutes)'})
fig3.update_layout(showlegend=False)
st.plotly_chart(fig3, width='stretch')

st.markdown('---')

# --- 4. Speed vs Delay ---
st.subheader('4. Speed vs. Delay Relationship')
sample_df = filtered_df.sample(min(1000, len(filtered_df)), random_state=42)
fig4 = px.scatter(sample_df, x='speed(km/h)', y='actual delay (mins)', color='route',
                  title='Speed vs. Delay by Route', trendline='lowess',
                  labels={'speed(km/h)': 'Speed (km/h)', 'actual delay (mins)': 'Delay (minutes)'},
                  hover_data=['calculation time'])
st.plotly_chart(fig4, width='stretch')

st.markdown('---')

# --- 5. Monthly Delay Trends ---
st.subheader('5. Monthly Delay Trends')
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
filtered_df.loc[:, 'month'] = pd.Categorical(filtered_df['month'], categories=month_order, ordered=True)
monthly_avg = filtered_df.groupby(['month', 'route'], observed=False)['actual delay (mins)'].mean().reset_index()
fig5 = px.bar(monthly_avg, x='month', y='actual delay (mins)', color='route', barmode='group',
              title='Average Monthly Delay by Route',
              labels={'month': 'Month', 'actual delay (mins)': 'Average Delay (minutes)'})
st.plotly_chart(fig5, width='stretch')


# --- 6. XAI Residual & Correlation Diagnostics -----------------------
st.markdown('---')
st.subheader('6. Explainable AI: Diagnostics and Feature Insights')


st.markdown('**Residual-style Diagnostic Plot**')
# approximate residual = expected base delay vs actual
filtered_df['expected_delay'] = base_delay + np.random.normal(0, 1, len(filtered_df))
filtered_df['residual'] = filtered_df['actual delay (mins)'] - filtered_df['expected_delay']

fig_resid = px.scatter(
    filtered_df, x='expected_delay', y='residual', color='route',
    title='Residual-style Plot (Actual - Expected Delay)',
    labels={'expected_delay': 'Expected Delay (minutes)', 'residual': 'Residual (minutes)'},
    opacity=0.7
)
fig_resid.add_hline(y=0, line_dash='dash', line_color='gray')
st.plotly_chart(fig_resid, width='stretch') 

# --- Footer ---
st.markdown('---')
st.caption("""
    Data Source: Banff Traffic Management Project (2025)
           
    Created by Alpine Analysts · NorQuest College MLAD · Fall 2025
""")