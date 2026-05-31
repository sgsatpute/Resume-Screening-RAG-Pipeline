import os
import sys
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf import PdfReader

import chatbot_verbosity as chatbot_verbosity
from ingest_data import ingest
from llm_agent import ChatBot
from retriever import SelfQueryRetriever

sys.dont_write_bytecode = True
load_dotenv()

DATA_PATH = "data/main-data/synthetic-resumes.csv"
FAISS_PATH = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_PROVIDERS = ["Ollama", "Gemini"]
OLLAMA_MODELS = ["llama3", "gemma3:4b", "mistral", "llama3.1"]
GEMINI_MODELS = ["gemini-3.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]


def options_with_default(options: list[str], default: str) -> list[str]:
    return [default] + [option for option in options if option != default]


def extract_pdf_text(uploaded_file) -> str:
    uploaded_file.seek(0)
    reader = PdfReader(uploaded_file)
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n\n".join(page for page in pages if page).strip()


def load_uploaded_resumes(uploaded_file) -> pd.DataFrame:
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        df_new = pd.read_csv(uploaded_file)
        if "Resume" not in df_new.columns or "ID" not in df_new.columns:
            raise ValueError("CSV must contain 'Resume' and 'ID' columns.")
        return df_new

    if file_name.endswith(".pdf"):
        resume_text = extract_pdf_text(uploaded_file)
        if not resume_text:
            raise ValueError("Could not extract text from this PDF.")
        return pd.DataFrame([{"ID": 1, "Resume": resume_text}])

    raise ValueError("Unsupported file type. Upload a CSV or PDF file.")


DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "Ollama").strip().title()
if DEFAULT_PROVIDER not in LLM_PROVIDERS:
    DEFAULT_PROVIDER = "Ollama"

DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3").strip()
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()

st.set_page_config(page_title="Resume Screening GPT", page_icon=":page_facing_up:")
st.title("Resume Screening GPT")
st.caption("Powered by selectable Ollama/Gemini models + RAG")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "resume_list" not in st.session_state:
    st.session_state.resume_list = []

if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = DEFAULT_PROVIDER

if "ollama_model" not in st.session_state:
    st.session_state.ollama_model = DEFAULT_OLLAMA_MODEL

if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = DEFAULT_GEMINI_MODEL

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
        st.error(f"Could not load resume data: {e}")
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
        st.warning("Vectorstore not found. Building one now; this may take a few minutes.")
        vectordb = ingest(st.session_state.df, "Resume", st.session_state.embedding_model)
        vectordb.save_local(FAISS_PATH)
        st.success("Vectorstore built and saved.")

    st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, st.session_state.df)

with st.sidebar:
    st.markdown("## Control Panel")

    st.selectbox("LLM Provider", LLM_PROVIDERS, key="llm_provider")

    if st.session_state.llm_provider == "Ollama":
        st.selectbox(
            "Ollama Model",
            options_with_default(OLLAMA_MODELS, st.session_state.ollama_model),
            key="ollama_model",
        )
        selected_model = st.session_state.ollama_model
        st.info(f"Running Ollama model `{selected_model}` locally. No API key required.")
    else:
        st.selectbox(
            "Gemini Model",
            options_with_default(GEMINI_MODELS, st.session_state.gemini_model),
            key="gemini_model",
        )
        selected_model = st.session_state.gemini_model
        st.info(f"Running Gemini model `{selected_model}`. Requires GOOGLE_API_KEY or GEMINI_API_KEY.")

    st.selectbox("RAG Mode", ["Generic RAG", "RAG Fusion"], key="rag_selection")

    st.markdown("---")
    st.markdown("### Upload Your Own Resumes")
    uploaded_file = st.file_uploader("Upload CSV or PDF", type=["csv", "pdf"])

    if uploaded_file:
        try:
            df_new = load_uploaded_resumes(uploaded_file)
            with st.spinner("Indexing resumes..."):
                vectordb = ingest(df_new, "Resume", st.session_state.embedding_model)
                st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, df_new)
                st.session_state.df = df_new
            st.success("Uploaded and indexed.")
        except Exception as e:
            st.error(f"Upload error: {e}")

    st.markdown("---")
    if st.button("Clear Conversation"):
        st.session_state.chat_history = []
        st.session_state.resume_list = []
        st.rerun()

    st.markdown("---")
    st.markdown("### Example Queries")
    st.markdown(
        """
- `Find Python developers with ML experience`
- `Hire a senior backend engineer with API skills`
- `Show applicant 101`
- `Who has TensorFlow experience?`
- `Compare the top 3 candidates`
        """
    )

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        """
This is a RAG-powered resume screening assistant.
Built as a Bachelor's thesis project by [Hungreeee](https://github.com/Hungreeee).
Modified to support **Ollama** and **Gemini** model backends.
        """
    )

selected_provider = st.session_state.llm_provider.lower()
selected_model = (
    st.session_state.ollama_model
    if st.session_state.llm_provider == "Ollama"
    else st.session_state.gemini_model
)
llm_signature = (selected_provider, selected_model)

if "llm" not in st.session_state or st.session_state.get("llm_signature") != llm_signature:
    try:
        st.session_state.llm = ChatBot(provider=selected_provider, model=selected_model)
        st.session_state.llm_signature = llm_signature
    except Exception as e:
        st.error(f"Could not initialize {st.session_state.llm_provider} model `{selected_model}`: {e}")
        st.stop()

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

user_query = st.chat_input("Ask about candidates or paste a job description...")

retriever = st.session_state.rag_pipeline
llm = st.session_state.llm

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

        if query_type == "retrieve_applicant_jd" and docs:
            with st.expander(f"Retrieved {len(docs)} candidate(s)", expanded=False):
                for i, doc in enumerate(docs[:5]):
                    st.markdown(f"**Candidate {i + 1}**")
                    st.markdown(doc[:500] + "..." if len(doc) > 500 else doc)
                    st.markdown("---")

        with st.spinner("Generating response..."):
            history = st.session_state.chat_history[:-1]
            stream = llm.generate_message_stream(user_query, docs, history, query_type)
            response = st.write_stream(stream)

        end = time.time()
        chatbot_verbosity.render(docs, retriever.meta_data, end - start)

        st.session_state.chat_history.append(AIMessage(content=response))
        st.session_state.chat_history.append(
            (chatbot_verbosity, docs, retriever.meta_data, end - start)
        )
