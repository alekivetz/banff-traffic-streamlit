import streamlit as st
import pandas as pd
import plotly.express as px
import io
from utils.display_images import display_banner
from utils.google_drive_helpers import download_from_drive


# --- UI ---
st.set_page_config(page_title='Banff Parking Analysis', page_icon='🚗', layout='wide')
display_banner()

# --- Header ---
st.title('Banff Parking Analysis')
st.write("""
The **Banff Parking Dashboard** provides an interactive overview of parking activity across the town of Banff.  
Explore key trends in **sessions, revenue, duration, and occupancy** to understand visitor behaviour and identify 
peak usage periods throughout the year.
""")

st.markdown('---')

# --- Load Data ---
@st.cache_data(ttl=86400, show_spinner=False)
def load_data():
    '''Load the merged and cleaned parking dataset from Google Drive.'''
    file_id = st.secrets['PARKING_VIS_ID']
    data_bytes = download_from_drive(file_id)
    df = pd.read_parquet(io.BytesIO(data_bytes), engine='fastparquet')
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f'❌ Failed to load data from Google Drive: {e}')
    st.stop()

# --- Filters ---
st.subheader('Filter Options')

col1, col2, col3, col4 = st.columns(4)
with col1:
    payment_filter = st.multiselect(
        'Payment Type',
        options=df['Payment type'].unique(),
        default=df['Payment type'].unique()
    )

with col2:
    unit_filter = st.multiselect(
        'Parking Lot',
        options=df['Unit'].unique(),
        default=df['Unit'].unique()
    )

with col3:
    month_filter = st.multiselect(
        'Month',
        options=df['month'].unique(),
        default=df['month'].unique()
    )

with col4:
    unit_search = st.text_input('Search Unit (optional)', placeholder='e.g., Bear Street')

col5, col6 = st.columns(2)
with col5:
    type_search = st.text_input('Search Fee or Type', placeholder='e.g., Paystation or Event')


# --- Apply Filters ---
df_filtered = df[
    (df['Payment type'].isin(payment_filter)) &
    (df['Unit'].isin(unit_filter)) &
    (df['month'].isin(month_filter))
]

if unit_search:
    df_filtered = df_filtered[df_filtered['Unit'].str.contains(unit_search, case=False, na=False)]

if type_search:
    df_filtered = df_filtered[
        df_filtered['Fee'].str.contains(type_search, case=False, na=False) |
        df_filtered['Type'].str.contains(type_search, case=False, na=False)
    ]

# --- Top KPIs ---
st.markdown('---')
st.subheader('Key Performance Indicators')

col1, col2, col3 = st.columns(3)

with col1:
    st.metric('Total Sessions', f'{len(df_filtered):,}')

with col2:
    st.metric('Total Revenue ($)', f'{df_filtered["Amount"].sum():,.2f}')

with col3:
    dur = df_filtered[df_filtered['duration'] != 'NO']['duration'].astype(float)
    avg_dur = round(dur.mean(), 2) if len(dur) > 0 else 0
    st.metric('Avg Duration (min)', avg_dur)

st.markdown('---')

# --- Monthly Sessions Trend ---
st.subheader('Monthly Parking Sessions Trend')

monthly_df = df.groupby('month').size().reset_index(name='sessions')

monthly_fig = px.line(
    monthly_df,
    x='month',
    y='sessions',
    title='Monthly Parking Sessions (Total Count)',
    markers=True
)

st.plotly_chart(monthly_fig, width='stretch')

# --- Day of Week Trend ---
st.markdown('---')
st.subheader('Day of Week Parking Trend (Overall)')

weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_df = df.groupby('day_of_week').size().reset_index(name='sessions')
dow_df['day_of_week'] = pd.Categorical(dow_df['day_of_week'], categories=weekday_order, ordered=True)
dow_df = dow_df.sort_values('day_of_week')

dow_fig = px.bar(
    dow_df,
    x='day_of_week',
    y='sessions',
    title='Parking Sessions by Day of Week (Overall)'
)

st.plotly_chart(dow_fig, width='stretch')

# --- Monthly Revenue Trend ---
st.markdown('---')
st.subheader('Monthly Revenue Trend (Overall)')

month_order = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]

rev_month_df = df.groupby('month')['Amount'].sum().reset_index()
rev_month_df['month'] = pd.Categorical(rev_month_df['month'], categories=month_order, ordered=True)
rev_month_df = rev_month_df.sort_values('month')

rev_month_fig = px.line(
    rev_month_df,
    x='month',
    y='Amount',
    title='Total Revenue by Month ($)',
    markers=True
)

st.plotly_chart(rev_month_fig, width='stretch')

# --- Busiest Units (Overall) ---
st.markdown('---')
st.subheader('Top Busiest Parking Units (Overall)')

unit_counts = (
    df['Unit']
    .value_counts()
    .reset_index(name='count')
    .rename(columns={'index': 'Unit'})
)

unit_fig = px.bar(
    unit_counts,
    x='Unit',
    y='count',
    title='Top Busiest Parking Units',
    labels={'Unit': 'Parking Lot', 'count': 'Sessions'}
)

st.plotly_chart(unit_fig, width='stretch')

# --- Payment Type Distribution ---
st.markdown('---')
st.subheader('Payment Type Distribution (Overall)')

payment_fig = px.pie(
    df,
    names='Payment type',
    title='Payment Method Breakdown (All Data)',
    hole=0.4
)

st.plotly_chart(payment_fig, width='stretch')

# --- Duration Distribution ---
st.markdown('---')
st.subheader('Duration Distribution (Filtered Data)')

dur_df = df_filtered[df_filtered['duration'] != 'NO'].copy()
dur_df['duration'] = dur_df['duration'].astype(float)

duration_fig = px.histogram(
    dur_df,
    x='duration',
    nbins=50,
    title='Distribution of Parking Duration (Minutes)',
    color_discrete_sequence=['#0072B2']
)

st.plotly_chart(duration_fig, width='stretch')

# --- Hour × Day-of-Week Heatmap ---
st.markdown('---')
st.subheader('Hour × Day-of-Week Parking Intensity Heatmap (Overall)')

heat_df = df[df['duration'] != 'NO'].copy()
heat_df['duration'] = heat_df['duration'].astype(float)

pivot_df = heat_df.groupby(['day_of_week', 'hour']).size().reset_index(name='sessions')
pivot_df['day_of_week'] = pd.Categorical(pivot_df['day_of_week'], categories=weekday_order, ordered=True)
pivot_table = pivot_df.pivot(index='day_of_week', columns='hour', values='sessions').fillna(0)

heatmap_fig = px.imshow(
    pivot_table,
    labels=dict(x='Hour of Day', y='Day of Week', color='Sessions'),
    title='Hourly Parking Activity by Day of Week',
    aspect='auto',
    color_continuous_scale='Blues'
)

st.plotly_chart(heatmap_fig, width='stretch')

# --- Footer ---
st.markdown('---')
st.caption("""
    Data Source: Banff Traffic Management Project (2025)
           
    Created by Alpine Analysts · NorQuest College MLAD · Fall 2025
""")
