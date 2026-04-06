"""
RAG Service — GENERAL_ACADEMIC_QUERY handler

Embedding: HuggingFace Inference API (HTTP) — no local model, no sentence_transformers.
           Same multilingual-e5-large model used during indexing.
           Add HF_API_KEY to .env for higher rate limits (free tier works fine).

Pipeline:
    question → HF API embed → Pinecone query → Groq LLM → markdown answer
"""

from __future__ import annotations

import os
import time
from typing import List, Dict, Optional

import requests
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from llm_client import llm_call_text

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_3VW5oW_EeqNfRBYYHyobURtEVAB299vvjEL86og3wuAbuWWE7jxKS2ZfuqFZTPKh2P2Q2F")
INDEX_NAME       = os.getenv("PINECONE_INDEX",   "bnu-bylaws")
TOP_K            = 4
MIN_SCORE        = 0.30

HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL   = "intfloat/multilingual-e5-large"

# ── HuggingFace InferenceClient — handles endpoint routing automatically ──────
_hf_client: Optional[InferenceClient] = None

def _get_hf_client() -> Optional[InferenceClient]:
    global _hf_client
    if _hf_client is None:
        if not HF_API_KEY:
            print("❌  HF_API_KEY not set. Add it to your .env file.")
            print("    Get a free token at: https://huggingface.co/settings/tokens")
            return None
        _hf_client = InferenceClient(token=HF_API_KEY)
    return _hf_client


# ═════════════════════════════════════════════════════════════════════════════
# Embedding via huggingface_hub InferenceClient (no hardcoded URLs)
# ═════════════════════════════════════════════════════════════════════════════

def _embed(text: str, retries: int = 3) -> Optional[List[float]]:
    """
    Embed *text* using huggingface_hub.InferenceClient.feature_extraction().
    Applies the 'query: ' prefix required by multilingual-e5-large.
    Returns a list of 1024 floats, or None on failure.

    Uses the official HF Python client instead of raw HTTP requests —
    no hardcoded URLs that break when HuggingFace changes their API.
    Requires HF_API_KEY in .env.
    """
    client = _get_hf_client()
    if client is None:
        return None

    for attempt in range(retries):
        try:
            result = client.feature_extraction(
                text=f"query: {text}",
                model=HF_MODEL,
            )
            # result may be a numpy array — convert to plain Python list
            if hasattr(result, "tolist"):
                result = result.tolist()
            # Unwrap [[...]] → [...] for single-string input
            if isinstance(result, list) and result and isinstance(result[0], list):
                result = result[0]
            if not isinstance(result, list) or not result:
                print(f"⚠️  Unexpected embedding shape: {type(result)}")
                return None
            return result

        except Exception as e:
            print(f"⚠️  HF embed attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2)

    return None


# ═════════════════════════════════════════════════════════════════════════════
# RAG Service
# ═════════════════════════════════════════════════════════════════════════════

class RAGService:
    """Singleton — retrieval-augmented generation for bylaw/academic questions."""

    def __init__(self):
        self._pinecone_idx = None
        self._ready        = False

    # ── Lazy init: only Pinecone client, no local model to load ──────────────

    def _ensure_ready(self) -> bool:
        if self._ready:
            return True
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=PINECONE_API_KEY)
            self._pinecone_idx = pc.Index(INDEX_NAME)
            stats = self._pinecone_idx.describe_index_stats()
            print(f"✅ Pinecone '{INDEX_NAME}' ready — {stats['total_vector_count']} vectors.")
            self._ready = True
            return True
        except Exception as e:
            print(f"⚠️  RAGService init failed: {e}")
            return False

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, question: str) -> List[Dict]:
        """Embed question → Pinecone → return relevant chunks."""
        vector = _embed(question)
        if vector is None:
            print("⚠️  Embedding failed — skipping retrieval.")
            return []

        results = self._pinecone_idx.query(
            vector=vector, top_k=TOP_K, include_metadata=True
        )
        return [
            {
                "topic": m["metadata"].get("topic", ""),
                "text":  m["metadata"].get("text",  ""),
                "score": round(m["score"], 3),
            }
            for m in results.get("matches", [])
            if m["score"] >= MIN_SCORE
        ]

    # ── Generation ────────────────────────────────────────────────────────────

    def _llm_answer(
        self,
        question: str,
        chunks: List[Dict],
        history: Optional[List[Dict]] = None,
    ) -> str:
        context = "\n\n---\n\n".join(
            f"[{c['topic']}]\n{c['text']}" for c in chunks
        )

        history_block = ""
        if history:
            turns = "\n".join(
                f"{m['role'].capitalize()}: {m['content']}" for m in history[-4:]
            )
            history_block = f"\n\nRecent conversation:\n{turns}"

        system = (
            "You are a friendly, knowledgeable academic advisor for BNU "
            "(Badr University in Cairo). "
            "Answer the student's question clearly and concisely using ONLY "
            "the provided bylaw/regulation context. "
            "Use **bold** for key terms and bullet points for lists. "
            "Keep answers short — 3-6 sentences or a brief bullet list. "
            "NEVER mention article numbers, article names, or any source references. "
            "NEVER say things like 'according to Article X' or 'as stated in Article Y'. "
            "Just give the answer directly as a knowledgeable advisor would. "
            "If the context doesn't cover the question, say so honestly and "
            "suggest contacting the registrar's office."
        )

        user_msg = (
            f"Context from BNU bylaws:\n{context}"
            f"{history_block}\n\n"
            f"Student question: {question}\n\n"
            "Give a concise, well-formatted answer."
        )

        try:
            return llm_call_text(system=system, user=user_msg, temperature=0.2, max_tokens=500)
        except Exception as e:
            return f"⚠️  Could not generate answer: {e}"

    # ── Public entry point ────────────────────────────────────────────────────

    def answer(
        self,
        question: str,
        history: Optional[List[Dict]] = None,
    ) -> str:
        if not self._ensure_ready():
            return (
                "⚠️  The knowledge base is currently unavailable. "
                "Please contact the registrar's office directly."
            )

        chunks = self.retrieve(question)
        if not chunks:
            return (
                "I couldn't find specific information about that in the BNU bylaws. "
                "Please contact the registrar's office or check the official student handbook."
            )

        return self._llm_answer(question, chunks, history)


# ── Singleton ─────────────────────────────────────────────────────────────────

_instance: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    global _instance
    if _instance is None:
        _instance = RAGService()
    return _instance


def handle_general_query(
    question: str,
    history: Optional[List[Dict]] = None,
) -> str:
    """Entry point called by the execution engine."""
    return get_rag_service().answer(question, history)