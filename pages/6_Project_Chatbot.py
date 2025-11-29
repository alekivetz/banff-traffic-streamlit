import re
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from utils.display_images import display_banner

# -----------------------------------------------------------
# PAGE CONFIG & STYLE
# -----------------------------------------------------------
st.set_page_config(page_title='Banff Project Chatbot', page_icon='💬', layout='wide')
display_banner()
st.title('💬 Banff Traffic Management Chatbot')
st.write('''
Ask questions about the **Banff Traffic Management** project — including 
its goals, traffic patterns, and parking analytics.  
The chatbot blends **retrieval-augmented generation (RAG)** with **data analysis**.
''')
st.markdown('---')

st.markdown("""
<style>
    .stChatFloatingInputContainer {
        position: fixed;
        bottom: 2rem;
        left: 50%;
        transform: translateX(-50%);
        width: 80%;
        max-width: 800px;
        background: white;
        padding: 1rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        z-index: 100;
    }
    .main .block-container { padding-bottom: 8rem; }
    .stChatMessage { max-height: 70vh; overflow-y: auto; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------
if 'messages' not in st.session_state:
    st.session_state.messages = []

# -----------------------------------------------------------
# LOAD DATA (once per session)
# -----------------------------------------------------------
@st.cache_data
def load_route_data():
    df = pd.read_csv('cleaned_routes.csv')
    if 'calculation time' in df.columns:
        df['calculation time'] = pd.to_datetime(df['calculation time'])
        df['hour'] = df['calculation time'].dt.hour
        df['day_name'] = df['calculation time'].dt.day_name()
        df['date'] = df['calculation time'].dt.date
    df['adjusted_speed'] = df['speed(km/h)']
    df['adjusted_delay'] = df['actual delay (mins)']
    return df

@st.cache_data
def load_parking_data():
    df = pd.read_csv('merged_df_cleaned.csv')
    df['Starting date'] = pd.to_datetime(df['Starting date'])
    return df

if 'routes_df' not in st.session_state:
    try:
        st.session_state.routes_df = load_route_data()
    except Exception:
        st.session_state.routes_df = pd.DataFrame()

if 'parking_df' not in st.session_state:
    try:
        st.session_state.parking_df = load_parking_data()
    except Exception:
        st.session_state.parking_df = pd.DataFrame()

# -----------------------------------------------------------
# LOAD EMBEDDINGS + GENERATOR (RAG)
# -----------------------------------------------------------
project_docs = {
    'overview': '''
        The Banff Traffic Management project supports smarter mobility in Banff National Park.
        It predicts route delays and parking occupancy using Random Forest and XGBoost models
        trained on historical and weather data.
    ''',
    'explainability': '''
        Model explainability uses SHAP values to highlight temporal, spatial, and behavioral
        features influencing congestion and parking forecasts.
    ''',
    'parking': '''
        The Parking Analysis component visualizes occupancy, demand, and revenue,
        forecasting parking needs 60 minutes ahead using 15-minute intervals.
    '''
}

@st.cache_resource
def load_embeddings():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    doc_embeddings = {k: embedder.encode(v, convert_to_tensor=True) for k, v in project_docs.items()}
    return embedder, doc_embeddings

@st.cache_resource
def load_generator():
    return pipeline('text2text-generation', model='google/flan-t5-small')

embedder, doc_embeddings = load_embeddings()
generator = load_generator()

def retrieve_context(query, top_k=2):
    query_emb = embedder.encode(query, convert_to_tensor=True)
    scores = {k: util.pytorch_cos_sim(query_emb, emb).item() for k, emb in doc_embeddings.items()}
    top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return '\n\n'.join(project_docs[d] for d, _ in top_docs)

def query_llm(query, context):
    prompt = (
        'You are a helpful assistant for the Banff Traffic Management project.\n\n'
        f'Context:\n{context}\n\nUser Query: {query}\n\nAnswer:'
    )
    output = generator(prompt, max_new_tokens=180, do_sample=True, temperature=0.7)
    return output[0]['generated_text'].replace(prompt, '').strip()

# -----------------------------------------------------------
# PARKING HELPERS
# -----------------------------------------------------------
def extract_iso_date(query):
    match = re.search(r'\d{4}-\d{2}-\d{2}', query)
    return match.group(0) if match else None

def handle_parking_query(q):
    df = st.session_state.parking_df
    q_lower = q.lower()
    date_str = extract_iso_date(q_lower)

    if date_str and any(k in q_lower for k in ['how many', 'how often', 'number of']):
        count = len(df[df['Starting date'].dt.date == pd.to_datetime(date_str).date()])
        return f'On {date_str}, there were {count} parking sessions.'

    if date_str and 'average' in q_lower and 'duration' in q_lower:
        filtered = df[df['Starting date'].dt.date == pd.to_datetime(date_str).date()]
        if filtered.empty:
            return f'No parking sessions found on {date_str}.'
        avg = filtered['duration'].mean()
        return f'On {date_str}, {len(filtered)} sessions had an average duration of {avg:.2f} hours.'

    match = re.search(r'(banff\d+)', q_lower)
    if match and 'average' in q_lower and 'duration' in q_lower:
        key = match.group(1)
        mask = df['Unit'].astype(str).str.contains(key, case=False, na=False)
        filtered = df[mask]
        if filtered.empty:
            return f'No sessions found for unit matching "{key}".'
        avg = filtered['duration'].mean()
        unit_name = filtered['Unit'].iloc[0]
        return f'{unit_name} has {len(filtered)} sessions with an average duration of {avg:.2f} hours.'

    if 'which unit' in q_lower and any(k in q_lower for k in ['longest', 'longer']):
        grouped = df.groupby('Unit')['duration'].agg(['mean', 'count']).reset_index()
        row = grouped.loc[grouped['mean'].idxmax()]
        return f'The unit with the longest average duration is {row["Unit"]}, with {int(row["count"])} sessions and an average duration of {row["mean"]:.2f} hours.'

    return None  # No match found

# -----------------------------------------------------------
# TRAFFIC QUERY HANDLER
# -----------------------------------------------------------
def handle_traffic_query(q):
    df = st.session_state.routes_df
    if df.empty:
        return 'Traffic data unavailable.'

    q_lower = q.lower()
    if 'delay' in q_lower:
        return f'Max delay: {df["adjusted_delay"].max():.2f} min, Min delay: {df["adjusted_delay"].min():.2f} min.'
    if 'speed' in q_lower:
        return f'Max speed: {df["adjusted_speed"].max():.1f} km/h, Min speed: {df["adjusted_speed"].min():.1f} km/h.'
    return None

# -----------------------------------------------------------
# SMART ROUTER
# -----------------------------------------------------------
def get_bot_response(query):
    if any(k in query.lower() for k in ['delay', 'speed', 'route']):
        resp = handle_traffic_query(query)
        if resp:
            return resp
    if any(k in query.lower() for k in ['parking', 'duration', 'banff']):
        resp = handle_parking_query(query)
        if resp:
            return resp
    context = retrieve_context(query)
    return query_llm(query, context)

# -----------------------------------------------------------
# CHAT UI
# -----------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

if prompt := st.chat_input('Ask me about Banff traffic or parking...'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)
    with st.chat_message('assistant'):
        with st.spinner('Analyzing...'):
            reply = get_bot_response(prompt)
            st.markdown(reply)
    st.session_state.messages.append({'role': 'assistant', 'content': reply})
