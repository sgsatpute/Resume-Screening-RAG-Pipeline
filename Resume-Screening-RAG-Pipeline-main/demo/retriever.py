import sys
import re
sys.dont_write_bytecode = True

RAG_K_THRESHOLD = 5

# ── Skills list ───────────────────────────────────────────────────────────────
SKILLS = [
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "ruby", "go",
    "rust", "php", "swift", "kotlin", "scala", "r", "matlab",
    # Web
    "react", "angular", "vue", "node.js", "django", "flask", "fastapi",
    "spring", "express", "html", "css", "rest", "graphql",
    # Data / ML / AI
    "machine learning", "deep learning", "nlp", "computer vision",
    "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
    "data science", "statistics", "reinforcement learning", "llm",
    # Cloud / DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "ci/cd", "terraform",
    "jenkins", "git", "linux", "bash",
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "cassandra", "oracle", "nosql",
    # Other
    "api", "microservices", "agile", "scrum", "excel", "power bi", "tableau",
    "spark", "hadoop", "kafka", "airflow",
]

# ── Query type detection ──────────────────────────────────────────────────────
JD_KEYWORDS = [
    "find", "looking for", "hire", "hiring", "need", "want", "search",
    "candidate", "developer", "engineer", "analyst", "designer", "manager",
    "experience with", "skills in", "proficient", "background in",
    "who has", "who knows", "team member", "position", "role", "job",
    "years of experience", "expertise", "senior", "junior", "mid-level",
]


def detect_query_type(question: str):
    q = question.lower()

    # ID lookup: support uploaded single-resume IDs like "applicant 1"
    # without treating "top 3 candidates" as candidate ID 3.
    explicit_ids = []
    for pattern in [
        r"\b(?:applicant|candidate|resume)\s*(?:id\s*)?#?\s*(\d+)\b",
        r"\bid\s*#?\s*(\d+)\b",
    ]:
        explicit_ids.extend(re.findall(pattern, q))

    ids = list(dict.fromkeys(explicit_ids)) if explicit_ids else re.findall(r"\b\d{3,}\b", question)
    if ids:
        return "retrieve_applicant_id", {"id_list": ids}

    # JD lookup: keyword match
    if any(kw in q for kw in JD_KEYWORDS):
        return "retrieve_applicant_jd", {"job_description": question}

    # Skill mention without explicit JD keywords → treat as JD search
    detected_skills = [s for s in SKILLS if s in q]
    if detected_skills:
        return "retrieve_applicant_jd", {"job_description": question}

    return "no_retrieve", {}


# ── Scoring helpers ───────────────────────────────────────────────────────────
def compute_score(query: str, resume_text: str) -> float:
    """
    Keyword + bigram overlap match score.

    Bug 3 fix: bigrams added so "machine learning" scores as one phrase hit
    (weighted 2x) rather than two weak unigram hits.

    Bug 4 fix: removed arbitrary (len % 7) tiebreak. Replaced with a
    skills-density bonus (max +2 pts) so richer resumes win ties meaningfully.
    """
    if not query.strip() or not resume_text.strip():
        return 0.0

    q_lower = query.lower()
    r_lower = resume_text.lower()

    # Unigrams
    query_words  = set(q_lower.split())
    resume_words = set(r_lower.split())

    # Bigrams
    q_tokens = q_lower.split()
    r_tokens = r_lower.split()
    query_bigrams  = set(zip(q_tokens, q_tokens[1:]))
    resume_bigrams = set(zip(r_tokens, r_tokens[1:]))

    unigram_overlap = len(query_words  & resume_words)
    bigram_overlap  = len(query_bigrams & resume_bigrams)

    total_query_terms = len(query_words) + len(query_bigrams)
    if total_query_terms == 0:
        return 0.0

    base_score = ((unigram_overlap + bigram_overlap * 2) / total_query_terms) * 100

    # Tiebreak: skills density in resume (max +2 pts)
    skill_count = len(extract_skills(resume_text))
    tiebreak = min(skill_count / 50, 1.0) * 2.0

    return round(base_score + tiebreak, 2)


def extract_skills(text: str) -> list:
    text = text.lower()
    return [skill for skill in SKILLS if skill in text]


# ── RAG retriever ─────────────────────────────────────────────────────────────
class RAGRetriever():
    def __init__(self, vectorstore_db, df):
        self.vectorstore = vectorstore_db
        self.df = df

    def __reciprocal_rank_fusion__(self, document_rank_list: list, k=50):
        fused_scores = {}
        for doc_list in document_rank_list:
            for rank, (doc, _) in enumerate(doc_list.items()):
                fused_scores.setdefault(doc, 0)
                fused_scores[doc] += 1 / (rank + k)
        return dict(sorted(fused_scores.items(), key=lambda x: x[1], reverse=True))

    def __retrieve_docs_id__(self, question: str, k=50):
        docs_score = self.vectorstore.similarity_search_with_score(question, k=k)
        return {str(doc.metadata["ID"]): score for doc, score in docs_score}

    def retrieve_id_and_rerank(self, subquestion_list: list):
        ranks = [self.__retrieve_docs_id__(q, RAG_K_THRESHOLD) for q in subquestion_list]
        return self.__reciprocal_rank_fusion__(ranks)

    def retrieve_documents_with_id(self, doc_id_with_score: dict, threshold=5, query=""):
        id_resume_dict = dict(zip(self.df["ID"].astype(str), self.df["Resume"]))
        top_ids = sorted(doc_id_with_score, key=doc_id_with_score.get, reverse=True)[:threshold]

        results = []
        for rid in top_ids:
            if rid not in id_resume_dict:
                continue
            resume = id_resume_dict[rid]
            score  = compute_score(query, resume)
            skills = extract_skills(resume)
            formatted = (
                f"Applicant ID: {rid}\n"
                f"Match Score: {score}%\n"
                f"Skills: {', '.join(skills) if skills else 'N/A'}\n\n"
                f"{resume}"
            )
            results.append(formatted)

        # Sort highest score first so LLM sees best candidates at top
        results.sort(
            key=lambda x: float(re.search(r"Match Score: ([\d.]+)%", x).group(1))
                if re.search(r"Match Score: ([\d.]+)%", x) else 0,
            reverse=True,
        )
        return results


# ── Self-query retriever ───────────────────────────────────────────────────────
class SelfQueryRetriever(RAGRetriever):
    def __init__(self, vectorstore_db, df):
        super().__init__(vectorstore_db, df)
        self.meta_data = {
            "rag_mode":                   "",
            "query_type":                 "no_retrieve",
            "extracted_input":            "",
            "subquestion_list":           [],
            "retrieved_docs_with_scores": [],
        }

    def retrieve_docs(self, question: str, llm, rag_mode: str):
        self.meta_data["rag_mode"] = rag_mode

        action, parsed = detect_query_type(question)
        self.meta_data["query_type"]      = action
        self.meta_data["extracted_input"] = parsed

        # ── ID lookup ──────────────────────────────────────────────────────
        if action == "retrieve_applicant_id":
            results = []
            for rid in parsed.get("id_list", []):
                try:
                    row = self.df[self.df["ID"].astype(str) == str(rid)].iloc[0]
                    results.append(f"Applicant ID {row['ID']}\n{row['Resume']}")
                except Exception:
                    continue
            return results

        # ── JD lookup ──────────────────────────────────────────────────────
        elif action == "retrieve_applicant_jd":
            subqueries = [question]
            if rag_mode == "RAG Fusion":
                try:
                    subqueries += llm.generate_subquestions(question)
                except Exception:
                    pass  # fallback to single query

            self.meta_data["subquestion_list"]           = subqueries
            retrieved_ids                                 = self.retrieve_id_and_rerank(subqueries)
            self.meta_data["retrieved_docs_with_scores"] = retrieved_ids
            return self.retrieve_documents_with_id(retrieved_ids, query=question)

        # ── No retrieval ───────────────────────────────────────────────────
        else:
            return []
