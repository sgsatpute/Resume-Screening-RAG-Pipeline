import sys
import re
sys.dont_write_bytecode = True

from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.llms import Ollama


class ChatBot():
    def __init__(self):
        self.llm = Ollama(model="llama3")

    def generate_subquestions(self, question: str) -> list[str]:
        prompt = f"""You are a recruitment expert. Break this job description into 3-4 focused search queries to find matching resumes.
Return ONLY the queries, one per line, no numbering, no bullet points, no extra text.

Job Description:
{question}
"""
        response = self.llm.invoke(prompt)

        # Bug 6 fix — strip numbering/bullets Llama3 adds despite instructions
        lines = [q.strip() for q in response.split("\n") if q.strip()]
        lines = [re.sub(r'^[\d\-\.\*\)]+\s*', '', q) for q in lines]
        lines = [q for q in lines if len(q) > 10]  # drop stray single words
        return lines[:4]

    def generate_message_stream(self, question: str, docs: list, history: list, prompt_cls: str):
        context = "\n\n".join(docs)

        if prompt_cls == "retrieve_applicant_jd":
            prompt = f"""You are an expert ATS (Applicant Tracking System) helping a hiring manager screen resumes.

Retrieved Resumes (already ranked by relevance):
{context}

Hiring Manager's Request:
{question}

Instructions:
- Candidates are pre-sorted by Match Score (highest first). DO NOT reorder them.
- Show the top 3 candidates with their Applicant ID and score.
- Pick the single best candidate and explain why in 2-3 bullet points.
- Be concise and professional.

Format your response EXACTLY like this:

Top Candidates:
1. Applicant ID: <ID> → Score: <score>%
2. Applicant ID: <ID> → Score: <score>%
3. Applicant ID: <ID> → Score: <score>%

Best Candidate: Applicant ID: <ID> | Score: <score>%

Reason:
- <reason 1>
- <reason 2>
- <reason 3>
"""

        elif prompt_cls == "retrieve_applicant_id":
            prompt = f"""You are a resume analyst helping a hiring manager evaluate a specific candidate.

Candidate Resume:
{context}

Question:
{question}

Provide a structured analysis with:
- Key Strengths (3 bullet points)
- Potential Weaknesses (2 bullet points)
- Overall Recommendation (1-2 sentences)

Be honest, concise and professional.
"""

        else:
            # Bug 7 fix — format history objects into readable text, not raw list
            if history:
                history_lines = []
                for msg in history[-6:]:  # last 3 turns (user + AI pairs)
                    if isinstance(msg, HumanMessage):
                        history_lines.append(f"User: {msg.content}")
                    elif isinstance(msg, AIMessage):
                        history_lines.append(f"Assistant: {msg.content}")
                    # tuple entries are verbosity objects — skip them
                history_text = "\n".join(history_lines) if history_lines else "No previous messages."
            else:
                history_text = "No previous messages."

            prompt = f"""You are a helpful recruitment assistant with expertise in HR and talent acquisition.

Chat History:
{history_text}

Question:
{question}

Answer clearly and professionally. If the question is about resumes or candidates,
use your expertise to give actionable advice.
"""

        return self.llm.stream(prompt)