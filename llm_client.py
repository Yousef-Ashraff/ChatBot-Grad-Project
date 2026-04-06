"""
llm_client.py — Centralized Groq LLM Client with 5-Key Fallback + Streaming
=============================================================================

All LLM calls in the project import from here.
The client holds 5 independent Groq API key slots (groq1 … groq5).
When a key hits its rate limit, the next key is tried automatically.

.env setup — add as many keys as you have:
    GROQ_API_KEY=gsk_...      ← slot 1 (primary)
    GROQ_API_KEY2=gsk_...     ← slot 2 (first fallback)
    GROQ_API_KEY3=gsk_...     ← slot 3
    GROQ_API_KEY4=gsk_...     ← slot 4
    GROQ_API_KEY5=gsk_...     ← slot 5

Slots with no key set are silently skipped.
The active model is controlled by GROQ_MODEL_XXX in .env (default: meta-llama/llama-4-scout-17b-16e-instruct).
The agent model is controlled by GROQ_MODEL_AGENT in .env (default: openai/gpt-oss-120b).

Usage:
    from llm_client import llm_call_json, llm_call_text, llm_call_stream_text

    # For JSON output (query engine, context resolver)
    raw_json = llm_call_json("Extract course name from: 'I want to take ML'")

    # For plain text (blocking)
    text = llm_call_text(system="You are an advisor.", user="What is my GPA?")

    # For streaming plain text — yields str chunks
    for chunk in llm_call_stream_text(system="...", user="..."):
        print(chunk, end="", flush=True)
"""

from __future__ import annotations

import os
import time
import logging
from typing import List, Dict, Optional, Generator

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Model selection
# ─────────────────────────────────────────────────────────────────────────────
# GROQ_MODEL_XXX: used for judge, reformulate, preprocessor, entity extraction, etc.
# GROQ_MODEL_AGENT: used by the agent LLM (set in agent.py via GROQ_MODEL_AGENT env var)
GROQ_MODEL = os.getenv("GROQ_MODEL_XXX", "meta-llama/llama-4-scout-17b-16e-instruct")

# ─────────────────────────────────────────────────────────────────────────────
# 5 independent key slots
# ─────────────────────────────────────────────────────────────────────────────
_KEY_SLOTS: List[Dict[str, str]] = [
    {"slot": "groq1", "env_key": "GROQ_API_KEY"},
    {"slot": "groq2", "env_key": "GROQ_API_KEY2"},
    {"slot": "groq3", "env_key": "GROQ_API_KEY3"},
    {"slot": "groq4", "env_key": "GROQ_API_KEY4"},
    {"slot": "groq5", "env_key": "GROQ_API_KEY5"},
]

_RATE_LIMIT_SIGNALS = (
    "rate_limit", "rate limit", "quota", "429",
    "overloaded", "capacity", "too many requests",
    "resource_exhausted", "insufficient_quota", "ratelimit",
)


def _is_rate_limit(exc: Exception) -> bool:
    return any(sig in str(exc).lower() for sig in _RATE_LIMIT_SIGNALS)


# ─────────────────────────────────────────────────────────────────────────────
# Low-level Groq callers
# ─────────────────────────────────────────────────────────────────────────────

def _call_groq(api_key: str, messages: List[Dict], temperature: float, max_tokens: int) -> str:
    """Blocking full-response call."""
    from groq import Groq
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def _call_groq_stream(
    api_key: str,
    messages: List[Dict],
    temperature: float,
    max_tokens: int,
) -> Generator[str, None, None]:
    """Streaming call — yields text chunks as they arrive from Groq."""
    from groq import Groq
    client = Groq(api_key=api_key)
    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ─────────────────────────────────────────────────────────────────────────────
# Public blocking API
# ─────────────────────────────────────────────────────────────────────────────

def llm_call(
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 600,
    retry_delay: float = 0.5,
) -> str:
    """
    Blocking chat request — cycles through key slots on rate-limit errors.
    Raises RuntimeError if every configured key slot fails.
    """
    attempted: List[str] = []
    last_error: Optional[Exception] = None

    for slot in _KEY_SLOTS:
        api_key = os.getenv(slot["env_key"], "").strip()
        if not api_key:
            continue

        slot_name = slot["slot"]
        attempted.append(slot_name)

        try:
            result = _call_groq(api_key, messages, temperature, max_tokens)
            if len(attempted) > 1:
                logger.info("LLM succeeded on fallback key slot '%s'", slot_name)
            return result

        except Exception as exc:
            last_error = exc
            if _is_rate_limit(exc):
                logger.debug(
                    "LLM key slot '%s' rate-limited — trying next key. (%s)",
                    slot_name, type(exc).__name__,
                )
                time.sleep(retry_delay)
                continue
            else:
                logger.debug("LLM key slot '%s' error: %s — trying next key.", slot_name, exc)
                continue

    configured = [s["slot"] for s in _KEY_SLOTS if os.getenv(s["env_key"], "").strip()]
    raise RuntimeError(
        f"All Groq key slots exhausted. "
        f"Configured slots: {configured or 'NONE — check your .env!'}. "
        f"Last error: {last_error}"
    )


def llm_call_json(
    prompt: str,
    system: str = "You output ONLY valid JSON. No explanations, no markdown, just JSON.",
    temperature: float = 0,
    max_tokens: int = 500,
) -> str:
    """JSON-only blocking call — used by query_engine and context_resolver."""
    return llm_call(
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )


def llm_call_text(
    system: str,
    user: str,
    temperature: float = 0.3,
    max_tokens: int = 600,
) -> str:
    """Blocking text/conversation call — used by rag_service and response_generator."""
    return llm_call(
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public streaming API (sync generators)
# ─────────────────────────────────────────────────────────────────────────────

def llm_call_stream(
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 600,
    retry_delay: float = 0.5,
) -> Generator[str, None, None]:
    """
    Sync generator — yields text chunks from Groq streaming API.
    Cycles through key slots on rate-limit errors.

    Usage:
        for chunk in llm_call_stream(messages):
            print(chunk, end="", flush=True)
    """
    last_error: Optional[Exception] = None

    for slot in _KEY_SLOTS:
        api_key = os.getenv(slot["env_key"], "").strip()
        if not api_key:
            continue

        slot_name = slot["slot"]

        try:
            for chunk in _call_groq_stream(api_key, messages, temperature, max_tokens):
                yield chunk
            return  # Streaming completed successfully on this slot

        except Exception as exc:
            last_error = exc
            if _is_rate_limit(exc):
                logger.debug(
                    "LLM stream key slot '%s' rate-limited — trying next key.", slot_name
                )
                time.sleep(retry_delay)
                continue
            else:
                logger.debug(
                    "LLM stream key slot '%s' error: %s — trying next.", slot_name, exc
                )
                continue

    configured = [s["slot"] for s in _KEY_SLOTS if os.getenv(s["env_key"], "").strip()]
    raise RuntimeError(
        f"All Groq key slots exhausted (streaming). "
        f"Configured slots: {configured or 'NONE — check your .env!'}. "
        f"Last error: {last_error}"
    )


def llm_call_stream_text(
    system: str,
    user: str,
    temperature: float = 0.3,
    max_tokens: int = 600,
) -> Generator[str, None, None]:
    """
    Streaming text/conversation call — yields str chunks.
    Used by response_generator for token-by-token SSE streaming to the client.
    """
    yield from llm_call_stream(
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics — run: python llm_client.py
# ─────────────────────────────────────────────────────────────────────────────

def check_slots() -> Dict[str, bool]:
    return {
        s["slot"]: bool(os.getenv(s["env_key"], "").strip())
        for s in _KEY_SLOTS
    }


if __name__ == "__main__":
    print("=== Groq Key Slot Status ===")
    configured_count = 0
    for slot_info in _KEY_SLOTS:
        active = bool(os.getenv(slot_info["env_key"], "").strip())
        status = "✅ key set" if active else "⬜ no key"
        print(f"  {slot_info['slot']}  ({slot_info['env_key']:<16})  {status}")
        if active:
            configured_count += 1

    print(f"\nModel: {GROQ_MODEL}")
    print(f"Active slots: {configured_count} / 5")

    if configured_count == 0:
        print("\n❌  No Groq keys configured!")
        print("    Add GROQ_API_KEY (and optionally GROQ_API_KEY2…GROQ_API_KEY5) to .env")
    else:
        print("\nRunning blocking test call...")
        try:
            result = llm_call_json('Return this exact JSON: {"ok": true}')
            print(f"✅ Blocking test passed: {result}")
        except RuntimeError as e:
            print(f"❌ Blocking test failed: {e}")

        print("\nRunning streaming test call...")
        try:
            collected = []
            for tok in llm_call_stream_text(
                system="You are a helpful assistant.",
                user="Say exactly: Hello streaming world",
                max_tokens=20,
            ):
                collected.append(tok)
                print(tok, end="", flush=True)
            print(f"\n✅ Streaming test passed ({len(collected)} chunks)")
        except RuntimeError as e:
            print(f"\n❌ Streaming test failed: {e}")