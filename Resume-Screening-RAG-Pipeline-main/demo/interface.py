import sys, os
sys.dont_write_bytecode = True

import time
import pandas as pd
import streamlit as st
import numpy as np

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings

from llm_agent import ChatBot
from ingest_data import ingest
from retriever import SelfQueryRetriever
import chatbot_verbosity as chatbot_verbosity

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH       = "data/main-data/synthetic-resumes.csv"
FAISS_PATH      = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

st.set_page_config(page_title="Resume Screening GPT", page_icon="📄")
st.title("📄 Resume Screening GPT")
st.caption("Powered by Llama3 (local) + RAG — 100% free, no API key needed")

# ── Session state ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "resume_list" not in st.session_state:
    st.session_state.resume_list = []

if "embedding_model" not in st.session_state:
    with st.spinner("Loading embedding model..."):
        st.session_state.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )

if "df" not in st.session_state:
    try:
        st.session_state.df = pd.read_csv(DATA_PATH)
    except Exception as e:
        st.error(f"❌ Could not load resume data: {e}")
        st.stop()

if "rag_pipeline" not in st.session_state:
    try:
        vectordb = FAISS.load_local(
            FAISS_PATH,
            st.session_state.embedding_model,
            distance_strategy=DistanceStrategy.COSINE,
            allow_dangerous_deserialization=True,
        )
    except Exception:
        st.warning("⚠️ Vectorstore not found — building one now (this may take a few minutes)...")
        vectordb = ingest(st.session_state.df, "Resume", st.session_state.embedding_model)
        vectordb.save_local(FAISS_PATH)
        st.success("✅ Vectorstore built and saved!")

    st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, st.session_state.df)

# FIX Bug 2 — cache ChatBot once; never recreate on every Streamlit rerun
if "llm" not in st.session_state:
    st.session_state.llm = ChatBot()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.info("🦙 Running Llama3 locally via Ollama — no API key required!")

    st.selectbox("RAG Mode", ["Generic RAG", "RAG Fusion"], key="rag_selection")

    st.markdown("---")
    st.markdown("### 📂 Upload Your Own Resumes")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        try:
            df_new = pd.read_csv(uploaded_file)
            if "Resume" not in df_new.columns or "ID" not in df_new.columns:
                st.error("CSV must contain 'Resume' and 'ID' columns.")
            else:
                with st.spinner("Indexing resumes..."):
                    vectordb = ingest(df_new, "Resume", st.session_state.embedding_model)
                    st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, df_new)
                    st.session_state.df = df_new
                st.success("✅ Uploaded and indexed!")
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")
    if st.button("🗑️ Clear Conversation"):
        st.session_state.chat_history = []
        st.session_state.resume_list = []
        st.rerun()

    st.markdown("---")
    st.markdown("### 💡 Example Queries")
    st.markdown("""
- `Find Python developers with ML experience`
- `Hire a senior backend engineer with API skills`
- `Show applicant 101`
- `Who has TensorFlow experience?`
- `Compare the top 3 candidates`
    """)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
This is a RAG-powered resume screening assistant.  
Built as a Bachelor's thesis project by [Hungreeee](https://github.com/Hungreeee).  
Modified to run 100% locally using **Ollama + Llama3**.
    """)

# ── Chat history display ───────────────────────────────────────────────────────
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)
    elif isinstance(message, tuple):
        with st.chat_message("AI"):
            message[0].render(*message[1:])

# ── Chat input ────────────────────────────────────────────────────────────────
user_query = st.chat_input("Ask about candidates or paste a job description...")

# Pull from session state — never reinstantiate on each rerun (Bug 2 fix)
retriever = st.session_state.rag_pipeline
llm       = st.session_state.llm

if user_query:
    with st.chat_message("Human"):
        st.markdown(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("AI"):
        start = time.time()

        with st.spinner("Retrieving candidates..."):
            docs = retriever.retrieve_docs(user_query, llm, st.session_state.rag_selection)
            query_type = retriever.meta_data["query_type"]
            st.session_state.resume_list = docs

        # Show retrieved candidates inline for JD queries
        if query_type == "retrieve_applicant_jd" and docs:
            with st.expander(f"📋 Retrieved {len(docs)} candidate(s)", expanded=False):
                for i, doc in enumerate(docs[:5]):
                    st.markdown(f"**Candidate {i+1}**")
                    st.markdown(doc[:500] + "..." if len(doc) > 500 else doc)
                    st.markdown("---")

        with st.spinner("Generating response..."):
            # Pass real chat history — not empty [] (Bug 2 fix)
            history = st.session_state.chat_history[:-1]  # exclude the just-appended user message
            stream  = llm.generate_message_stream(user_query, docs, history, query_type)
            response = st.write_stream(stream)

        end = time.time()

        # Verbosity expander
        chatbot_verbosity.render(docs, retriever.meta_data, end - start)

        st.session_state.chat_history.append(AIMessage(content=response))
        st.session_state.chat_history.append(
            (chatbot_verbosity, docs, retriever.meta_data, end - start)
        )