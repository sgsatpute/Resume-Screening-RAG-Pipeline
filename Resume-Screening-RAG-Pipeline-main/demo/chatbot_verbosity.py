import sys
sys.dont_write_bytecode = True

import streamlit as st
import numpy as np


def render(document_list: list, meta_data: dict, time_elapsed: float):
    message_map = {
        "retrieve_applicant_jd": "📄 **Job description detected** — RAG retrieval used.",
        "retrieve_applicant_id": "🔍 **Applicant ID(s) detected** — exact ID lookup used.",
        "no_retrieve":           "💬 **General question** — answered from model knowledge.",
    }

    with st.expander(f"🔎 Verbosity  ({np.round(time_elapsed, 2)}s)", expanded=False):
        st.markdown(message_map.get(meta_data["query_type"], ""))

        if meta_data["query_type"] == "retrieve_applicant_jd":
            st.markdown(f"**Mode:** {meta_data['rag_mode']}")
            st.markdown(f"**Top {len(document_list[:5])} resumes retrieved.**")

            cols = st.columns(min(len(document_list[:5]), 5), gap="small")
            for i, doc in enumerate(document_list[:5]):
                with cols[i], st.popover(f"Resume {i+1}"):
                    st.markdown(doc)

            if meta_data.get("subquestion_list"):
                st.markdown("**Sub-queries used:**")
                for q in meta_data["subquestion_list"]:
                    st.markdown(f"- `{q}`")

            if meta_data.get("retrieved_docs_with_scores"):
                st.markdown("**Re-ranking scores:**")
                scores = meta_data["retrieved_docs_with_scores"]
                top = list(scores.items())[:5]
                for doc_id, score in top:
                    st.markdown(f"- ID `{doc_id}` → `{round(score, 4)}`")

        elif meta_data["query_type"] == "retrieve_applicant_id":
            cols = st.columns(min(len(document_list[:5]), 5), gap="small")
            for i, doc in enumerate(document_list[:5]):
                with cols[i], st.popover(f"Resume {i+1}"):
                    st.markdown(doc)
            st.markdown(f"**Extracted IDs:** `{meta_data.get('extracted_input', '')}`")