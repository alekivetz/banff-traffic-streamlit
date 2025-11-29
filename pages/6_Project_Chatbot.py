import re
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from utils.display_images import display_banner

# --- Page Config & Style ---
st.set_page_config(page_title='Banff Project Chatbot', page_icon='💬', layout='wide')
display_banner()
st.title('Banff Traffic Management Chatbot')
st.write("""
Ask questions about **traffic conditions**, **parking analytics**, or the **project itself**.  
This chatbot blends **retrieval-augmented generation (RAG)** with **data-driven insights** to provide both factual and conceptual answers.
""")

# --- Example Queries ---
# Show expanded only before first chat
expanded_state = not st.session_state.get('messages')

with st.expander("Try asking questions like these:", expanded=expanded_state):
    st.markdown("""
    **Parking Queries**
    - What is the average parking duration?  
    - How many parking sessions were there on 2025-08-31?  
    - What is the average parking duration for BANFF01?  
    - Which unit has the longest average duration?

    **Traffic Queries**
    - What is the average delay across all routes?  
    - What’s the maximum speed recorded on Route 3?  
    - Show me the delay trends for November.  
    - Which routes have the highest congestion?

    **Project & Model Queries**
    - How does the Banff Traffic Management model work?  
    - What algorithms are used for prediction?  
    - How do SHAP values explain congestion predictions?  
     - What factors influence parking demand the most?
    """)


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

# --- Initialization ----
if 'messages' not in st.session_state:
    st.session_state.messages = []

routes_df = st.session_state.routes_df_vis_chatbot
parking_df = st.session_state.parking_df_chatbot

# --- Context Documents ---
project_overview = """
The Banff Traffic Management project supports smarter mobility in Banff National Park.
It predicts route-level congestion and parking occupancy 60 minutes ahead,
using Random Forest and XGBoost models trained on time-series and weather data.
"""

model_explainability = """
Model explainability relies on SHAP values to identify the most influential
temporal, spatial, and behavioral factors contributing to congestion and parking demand.
"""

parking_analysis = """
The Parking Analysis dashboard visualizes occupancy, duration, and revenue trends.
Forecasts use 15-minute intervals of historical data merged with weather and event indicators.
"""

model_performance = """
The Banff Traffic Management project evaluates model performance for both traffic and parking prediction tasks.

For **traffic congestion forecasting**, Random Forest and XGBoost regressors were trained per route using time-based, spatial, and weather-related features.  
Average performance across all routes:
- R²: 0.93  
- MAE: 0.12 minutes  
- RMSE: 0.31 minutes  

Top-performing routes include Route 1, Route 3, and Route 12, all with R² values above 0.98.  
More variable routes, such as Route 5 and Route 11, show higher error due to irregular traffic patterns.

For **parking demand forecasting**, an ensemble regression model was trained using parking duration, weather, and time-of-day features.  
Results:
- R²: 0.95
- MAE: 0.969
"""

documents = {
    'overview': project_overview,
    'explainability': model_explainability,
    'parking': parking_analysis,
    'performance': model_performance,
}

# --- Embeddings + Generator (RAG) ---
@st.cache_resource
def load_embeddings():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    doc_embeddings = {k: embedder.encode(v, convert_to_tensor=True) for k, v in documents.items()}
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
    return '\n\n'.join(documents[d] for d, _ in top_docs)

# --- Intent Matching ---
@st.cache_resource
def load_intent_imbedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

intent_embedder = load_intent_imbedder()

# Predefined intent categories

INTENTS = {
    'parking': [
        'parking duration',
        'how long people park',
        'parking sessions',
        'which parking lot',
        'banff parking',
        'average stay',
        'parking occupancy',
        'busiest parking lot'
    ],
    'traffic': [
        'traffic delay',
        'vehicle speed',
        'route congestion',
        'road conditions',
        'travel time',
        'rush hour',
        'busiest routes',
        'traffic volume'
    ],
    'project': [
        'project overview',
        'machine learning model',
        'random forest',
        'xgboost',
        'explainability',
        'model training',
        'shap values',
        'feature importance'
    ],
}

# Encode all examples once
INTENT_EMBEDDINGS = {
    k: intent_embedder.encode(v, convert_to_tensor=True) for k, v in INTENTS.items()
}

def detect_intent(query):
    """Return best-matching intent based on cosine similarity."""
    q_emb = intent_embedder.encode(query, convert_to_tensor=True)
    scores = {
        intent: float(util.pytorch_cos_sim(q_emb, emb).max())
        for intent, emb in INTENT_EMBEDDINGS.items()
    }
    best_intent = max(scores, key=scores.get)
    return best_intent if scores[best_intent] > 0.35 else 'project'  # fallback to RAG

def query_llm(query, context):
    prompt = (
        'You are a helpful assistant for the Banff Traffic Management project.\n\n'
        f'Context:\n{context}\n\nUser Query: {query}\n\nAnswer:'
    )
    output = generator(prompt, max_new_tokens=180, do_sample=True, temperature=0.7)
    return output[0]['generated_text'].replace(prompt, '').strip()

# --- Rule-Based Data Queries ---
def extract_iso_date(query):
    match = re.search(r'\d{4}-\d{2}-\d{2}', query)
    return match.group(0) if match else None

# --- Parking Queries
def handle_parking_query(q):
    df = parking_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['Starting date']):
        df['Starting date'] = pd.to_datetime(df['Starting date'], errors='coerce')

    q_lower = q.lower()
    date_str = extract_iso_date(q_lower)

    # Count sessions by day
    if date_str and any(k in q_lower for k in ['how many', 'how often', 'number of']):
        count = len(df[df['Starting date'].dt.date == pd.to_datetime(date_str).date()])
        return f'On {date_str}, there were {count} parking sessions.'

    # Average duration by day   
    if date_str and 'average' in q_lower and 'duration' in q_lower:
        filtered = df[df['Starting date'].dt.date == pd.to_datetime(date_str).date()]
        if filtered.empty:
            return f'No parking sessions found on {date_str}.'
        avg = filtered['duration'].mean()
        return f'On {date_str}, {len(filtered)} sessions had an average duration of {avg:.2f} hours.'

    # Average duration for specific unit
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

    # Longest average duration (unit/lot synonyms)
    if any(k in q_lower for k in ['unit', 'lot', 'location', 'parkade', 'parking lot']) and \
       any(k in q_lower for k in ['longest', 'longer', 'highest', 'most time']):
        grouped = df.groupby('Unit')['duration'].agg(['mean', 'count']).reset_index()
        row = grouped.loc[grouped['mean'].idxmax()]
        return (
            f'The parking lot with the longest average duration is **{row["Unit"]}**, '
            f'with {int(row["count"])} sessions and an average stay of {row["mean"]:.2f} hours.'
        )
    
    # Overall average parking duration - no date or unit
    if 'average' in q_lower and 'duration' in q_lower:
        avg = df['duration'].mean()
        total = len(df)
        return (
            f'The overall average parking duration across all {total:,} sessions '
            f'is {avg:.2f} hours.'
        )
    return None  # No match found

# --- Traffic Queries
def handle_traffic_query(q):
    df = routes_df.copy()
    q_lower = q.lower()

    # Delay analysis 
    if any(k in q_lower for k in ['delay', 'travel time', 'congestion']):
        max_delay = df['actual delay (mins)'].max()
        min_delay = df['actual delay (mins)'].min()
        avg_delay = df['actual delay (mins)'].mean()
        return (
            f'Average delay: {avg_delay:.2f} min\n'
            f'Max delay: {max_delay:.2f} min | Min delay: {min_delay:.2f} min.'
        )

    # Speed analysis
    if any(k in q_lower for k in ['speed', 'velocity', 'pace']):
        max_speed = df['speed(km/h)'].max()
        min_speed = df['speed(km/h)'].min()
        avg_speed = df['speed(km/h)'].mean()
        return (
            f'Average speed: {avg_speed:.1f} km/h\n'
            f'Highest: {max_speed:.1f} km/h | Lowest: {min_speed:.1f} km/h.'
        )
    
    # Most congested routes
    if any(k in q_lower for k in ['busiest', 'most congested', 'worst traffic']):
        busiest = df.groupby('route')['actual delay (mins)'].mean().nlargest(3).index.tolist()
        return f'The most congested routes are typically {", ".join(busiest)}.'



    return None


# --- Routing Logic ---
def get_bot_response(query):
    intent = detect_intent(query)

    if intent == 'parking':
        resp = handle_parking_query(query)
        if resp:
            return resp

    if intent == 'traffic':
        resp = handle_traffic_query(query)
        if resp:
            return resp

    # fallback: project or general
    context = retrieve_context(query)
    return query_llm(query, context)


# --- Chat UI ---
if 'messages' not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

if prompt := st.chat_input('Ask me about Banff traffic, parking, or our Banff Traffic Management project...'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)

    with st.chat_message('assistant'):
        with st.spinner('Analyzing your question...'):
            reply = get_bot_response(prompt)
            st.markdown(reply)

    st.session_state.messages.append({'role': 'assistant', 'content': reply})