# Data & AI Challenge: Intelligent Candidate Discovery

Team Name: ResumeRAG

Team Leader Name: Saurav Satpute

GitHub Repository: https://github.com/sgsatpute/Resume-Screening-RAG-Pipeline

## Problem Statement

Hiring teams receive large volumes of unstructured resumes and long job descriptions. Traditional keyword screening can miss semantically relevant candidates, reward keyword stuffing, and provide weak explanations for why a candidate was shortlisted. The goal is to convert a JD into a ranked, explainable, evidence-backed candidate list.

## Solution Overview

The proposed solution is a RAG-powered resume screening assistant. It indexes resume chunks in FAISS using sentence-transformer embeddings, retrieves candidates for a JD, optionally expands the JD with RAG Fusion sub-queries, re-ranks candidates, maps candidate IDs back to full resumes, computes readable fit scores, and asks an LLM to produce a grounded recruiter-friendly answer.

The differentiator is that the LLM does not search from memory or invent candidates. It receives a ranked evidence pack containing applicant IDs, match scores, extracted skills, and resume text, then summarizes only those retrieved candidates.

## JD Understanding & Candidate Evaluation

Key JD requirements extracted:
- Role intent: developer, engineer, analyst, designer, manager, job, role, seniority.
- Hard skills: programming languages, frameworks, ML/data tools, cloud, databases, DevOps, Agile/Scrum.
- Experience signals: years of experience, hands-on delivery, senior/junior terms.
- Domain/tooling: industry context, product area, databases, cloud stack, APIs.
- Soft signals: communication, collaboration, ownership, problem solving.

Most important candidate signals:
- Semantic similarity between JD and resume chunks.
- Matched skills from the maintained skills list.
- Phrase-level/bigram matches for requirements such as machine learning or project manager.
- Full resume evidence after chunk retrieval.
- Data-quality flags such as short resumes, no skill match, placeholder contact details, or weak profile structure.

## Ranking Methodology

The system retrieves, scores, and ranks candidates in four steps:
1. Query classification detects JD search, applicant ID lookup, or general recruitment question.
2. FAISS retrieves top candidate resume chunks using sentence-transformer embeddings.
3. In RAG Fusion mode, the LLM generates 3-4 focused sub-queries and reciprocal-rank fusion combines the candidate ID lists.
4. Retrieved IDs are mapped back to full resumes and sorted using `compute_score`.

Models, algorithms, and heuristics:
- FAISS vector search.
- `sentence-transformers/all-MiniLM-L6-v2` embeddings.
- RAG Fusion query expansion.
- Reciprocal rank fusion: `1 / (rank + 50)`.
- Final readable score: unigram overlap + 2x bigram overlap + skill-density bonus.
- LLM providers: local Ollama or Gemini.

## Explainability & Data Validation

Ranking decisions are explained through applicant ID, score, matched skills, and evidence snippets from retrieved resumes. The LLM prompt explicitly says not to reorder candidates, not to invent extra candidates, and not to output missing IDs or scores.

Hallucinations are reduced by grounding the response in retrieved resume context and by preserving candidate order from deterministic retrieval/scoring. Unsupported justifications are avoided by using only retrieved resume text as the evidence source.

Inconsistent or suspicious profiles are not silently removed. The ranked workbook flags short resumes, no skill match, placeholder contact details, and weak profile structure so a recruiter can review data quality before acting.

## End-to-End Workflow

1. User pastes a JD or candidate query into the Streamlit UI.
2. The retriever classifies the query.
3. JD queries trigger FAISS retrieval.
4. RAG Fusion optionally generates focused sub-queries.
5. Candidate ID lists are fused and re-ranked.
6. Candidate IDs map back to full resume text.
7. Final score and matched skills are computed.
8. The ranked context is sent to the LLM.
9. The LLM streams the shortlist and explanation.
10. The verbosity panel shows query type, RAG mode, sub-queries, retrieved candidates, RRF scores, and elapsed time.

## System Architecture

The architecture has six layers:
- Streamlit UI for chat, upload, provider selection, and RAG mode selection.
- Session state for cached embeddings, active dataframe, retriever, and LLM wrapper.
- Query router for JD, applicant ID, or general question paths.
- FAISS vectorstore with sentence-transformer resume chunk embeddings.
- Ranking layer using RAG Fusion, reciprocal-rank fusion, keyword/bigram/skill scoring.
- LLM generation layer using Ollama or Gemini, plus a verbosity trace for validation.

## Results & Performance

The improved current app retriever was rerun on all 500 evaluation rows:
- Top-1 accuracy / Hit@1: 0.8160
- Hit@3: 0.8960
- Hit@5: 0.9100
- Mean reciprocal rank: 0.8559
- Top-5 misses reduced from 166 to 45

Compared with the previous current app path, Hit@1 improved from 0.6240 to 0.8160 and Hit@5 improved from 0.6680 to 0.9100.

The saved RAGAS research outputs are still included for continuity. The best saved RAGAS run, GPT-4 + RAG Fusion, achieved:
- Context precision: 0.7927
- Context recall: 0.8434
- Faithfulness: 0.9450
- Answer similarity: 0.7329

Gemini and Ollama were both tested on a five-query generation sample using the improved retriever. Gemini and Ollama `gemma3:4b` both achieved 0.8000 generated Hit@1 on the sample, while Gemini had lower average latency on this machine.

The system meets runtime and compute constraints by using FAISS for efficient vector search, a compact CPU-friendly embedding model, cached Streamlit session objects, and local Ollama support for no-API-key execution.

## Technologies Used

- Streamlit: fast interactive screening UI.
- LangChain: document loading, splitting, retrieval, and LLM integration.
- FAISS: local vector indexing and similarity search.
- Hugging Face sentence-transformers: semantic resume embeddings.
- Ollama: local LLM runtime.
- Gemini: optional cloud LLM backend.
- Pandas: CSV resume loading and uploaded dataset handling.
- PyPDF: PDF resume text extraction.

## Submission Assets

- GitHub repository: https://github.com/sgsatpute/Resume-Screening-RAG-Pipeline
- Deck PDF: `candidate_discovery_submission.pdf`
- Editable deck: `candidate_discovery_submission.pptx`
- Ranked output workbook: `ranked_candidate_output.xlsx`

Note: the official Hack2Skill PPT template and ranked-output spreadsheet template were not present in the local workspace, so the generated deck is a clean from-scratch deck and the workbook uses explicit challenge-relevant columns.
