import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

from utils.display_images import display_banner

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Project Chatbot", page_icon="💬", layout="wide")
display_banner()
st.title("Project Chatbot")
st.write("""
Ask questions about the **Banff Traffic Management** project — 
including its goals, models, data sources, and key insights.
The chatbot uses a Retrieval-Augmented Generation (RAG) approach 
to combine contextual knowledge with language understanding.
""")
st.markdown("---")

# -----------------------------------------------------------------------------
# 1. Context documents (you can replace these with real text later)
# -----------------------------------------------------------------------------
project_overview = """
The Banff Traffic Management project supports smarter mobility in Banff National Park.
It predicts traffic delays per route and parking occupancy 60 minutes ahead.
Models include Random Forest and XGBoost regressors trained on time-series features.
"""

model_explainability = """
Model explainability uses SHAP values to highlight the most influential features 
affecting congestion and parking predictions, including temporal, spatial, and behavioral drivers.
"""

parking_analysis = """
The Parking Analysis Dashboard visualizes revenue, demand, and occupancy across time and locations.
Parking forecasts rely on historical 15-minute occupancy data and weather factors.
"""

documents = {
    "overview": project_overview,
    "explainability": model_explainability,
    "parking": parking_analysis,
}

# -----------------------------------------------------------------------------
# 2. Embed documents
# -----------------------------------------------------------------------------
@st.cache_resource
def load_embeddings():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    doc_embeddings = {
        k: embedder.encode(v, convert_to_tensor=True)
        for k, v in documents.items()
    }
    return embedder, doc_embeddings

embedder, doc_embeddings = load_embeddings()

def retrieve_context(query, top_k=2):
    query_emb = embedder.encode(query, convert_to_tensor=True)
    scores = {
        k: util.pytorch_cos_sim(query_emb, emb).item()
        for k, emb in doc_embeddings.items()
    }
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_doc_ids = [doc for doc, _ in sorted_docs[:top_k]]
    return "\n\n".join(documents[doc] for doc in top_doc_ids)

# -----------------------------------------------------------------------------
# 3. Set up language model
# -----------------------------------------------------------------------------
@st.cache_resource
def load_generator():
    return pipeline("text2text-generation", model="google/flan-t5-small")

generator = load_generator()

def query_llm(query, context):
    prompt = (
        "You are a helpful assistant for the Banff Traffic Management project. "
        "Use the provided context to answer clearly and accurately.\n\n"
        f"Context:\n{context}\n\nUser Query: {query}\n\nAnswer:"
    )
    outputs = generator(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
    response = outputs[0]["generated_text"]
    return response.replace(prompt, "").strip()

# -----------------------------------------------------------------------------
# 4. Streamlit chatbot UI
# -----------------------------------------------------------------------------
user_query = st.text_input("Ask a question about the project:", placeholder="e.g. How do you predict parking demand?")

if user_query:
    with st.spinner("Generating response..."):
        context = retrieve_context(user_query)
        answer = query_llm(user_query, context)
        st.success(answer)
