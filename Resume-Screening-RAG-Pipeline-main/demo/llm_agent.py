import os
import re
import sys

sys.dont_write_bytecode = True

from langchain_core.messages import AIMessage, HumanMessage


class ChatBot:
    def __init__(self, provider: str = "ollama", model: str | None = None):
        self.provider = provider.lower().strip()
        self.model = model or self._default_model(self.provider)
        self.llm = self._build_llm()

    def _default_model(self, provider: str) -> str:
        if provider == "gemini":
            return os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
        return os.getenv("OLLAMA_MODEL", "llama3")

    def _build_llm(self):
        if self.provider == "ollama":
            return self._build_ollama()

        if self.provider == "gemini":
            return self._build_gemini()

        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def _build_ollama(self):
        try:
            from langchain_ollama import OllamaLLM

            return OllamaLLM(model=self.model)
        except ImportError:
            from langchain_community.llms import Ollama

            return Ollama(model=self.model)

    def _build_gemini(self):
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Gemini selected but no API key was found. Set GOOGLE_API_KEY "
                "or GEMINI_API_KEY in your environment or .env file."
            )

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise RuntimeError(
                "Gemini selected but langchain-google-genai is not installed. "
                "Install it with: pip install langchain-google-genai"
            ) from exc

        return ChatGoogleGenerativeAI(
            model=self.model,
            google_api_key=api_key,
            temperature=0.1,
        )

    def _to_text(self, response) -> str:
        if isinstance(response, str):
            return response

        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "".join(parts)

        return str(content)

    def _stream_text(self, prompt: str):
        for chunk in self.llm.stream(prompt):
            text = self._to_text(chunk)
            if text:
                yield text

    def generate_subquestions(self, question: str) -> list[str]:
        prompt = f"""You are a recruitment expert. Break this job description into 3-4 focused search queries to find matching resumes.
Return ONLY the queries, one per line, no numbering, no bullet points, no extra text.

Job Description:
{question}
"""
        response = self.llm.invoke(prompt)
        response_text = self._to_text(response)

        # Strip numbering and bullets models may add despite instructions.
        lines = [q.strip() for q in response_text.split("\n") if q.strip()]
        lines = [re.sub(r"^[\d\-\.\*\)]+\s*", "", q) for q in lines]
        lines = [q for q in lines if len(q) > 10]
        return lines[:4]

    def generate_message_stream(
        self,
        question: str,
        docs: list,
        history: list,
        prompt_cls: str,
    ):
        context = "\n\n".join(docs)
        candidate_count = len(docs)

        if prompt_cls == "retrieve_applicant_jd":
            prompt = f"""You are an expert ATS (Applicant Tracking System) helping a hiring manager screen resumes.

Retrieved Resumes (already ranked by relevance):
{context}

Hiring Manager's Request:
{question}

Instructions:
- Candidates are pre-sorted by Match Score (highest first). DO NOT reorder them.
- Show only the {candidate_count} retrieved candidate(s). Do not invent extra candidates.
- Never write Applicant ID: N/A or Score: N/A.
- Pick the single best candidate and explain why in 2-3 bullet points.
- Be concise and professional.

Format your response EXACTLY like this:

Top Candidates:
1. Applicant ID: <ID from retrieved resume> -> Score: <score from retrieved resume>%

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
            if history:
                history_lines = []
                for msg in history[-6:]:
                    if isinstance(msg, HumanMessage):
                        history_lines.append(f"User: {msg.content}")
                    elif isinstance(msg, AIMessage):
                        history_lines.append(f"Assistant: {msg.content}")
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

        return self._stream_text(prompt)
