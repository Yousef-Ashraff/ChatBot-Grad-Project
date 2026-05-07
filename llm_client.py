"""
llm_client.py — Centralized LLM Client with Groq + OpenRouter routing
=======================================================================

All LLM calls in the project import from here.

Routing:
  MODEL_AGENT  (openai/gpt-oss-120b) → OpenRouter (OPENROUTER_API_KEY)
  GROQ_MODEL_XXX    (utility tasks)        → Groq (GROQ_API_KEY with 5-key fallback)

.env setup:
    GROQ_API_KEY=        # required (utility model)
    GROQ_API_KEY2=       # optional rate-limit fallback
    GROQ_API_KEY3=
    GROQ_API_KEY4=
    GROQ_API_KEY5=
    OPENROUTER_API_KEY=  # required (agent model)

    MODEL_AGENT=openai/gpt-oss-120b
    GROQ_MODEL_XXX=meta-llama/llama-4-scout-17b-16e-instruct

Usage:
    from llm_client import llm_call_json, llm_call_text, llm_call_stream_text
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, Generator, List, Optional

import openai
from dotenv import load_dotenv
from groq import Groq, RateLimitError

load_dotenv()

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

_GROQ_KEYS: List[str] = [
    k for k in [
        os.getenv("GROQ_API_KEY"),
        os.getenv("GROQ_API_KEY2"),
        os.getenv("GROQ_API_KEY3"),
        os.getenv("GROQ_API_KEY4"),
        os.getenv("GROQ_API_KEY5"),
    ] if k
]

if not _GROQ_KEYS:
    logger.warning(
        "No GROQ_API_KEY found. Set GROQ_API_KEY (and optionally GROQ_API_KEY2-5) in .env"
    )

# Two-model strategy:
#   MODEL_AGENT → agent tool-selection + final answer synthesis (OpenRouter)
#   GROQ_MODEL_XXX   → all utility tasks (judge, reformulate, preprocess, RAG, translate) (Groq)
MODEL_AGENT     = os.getenv("MODEL_AGENT", "openai/gpt-oss-120b")
GROQ_MODEL_XXX       = os.getenv("GROQ_MODEL_XXX",   "meta-llama/llama-4-scout-17b-16e-instruct")

# Aliases for modules that import these names directly
GROQ_MODEL           = GROQ_MODEL_XXX
GROQ_MODEL_TRANSLATE = GROQ_MODEL_XXX
OLLAMA_MODEL         = GROQ_MODEL_XXX       # legacy alias
OLLAMA_MODEL_TRANSLATE = GROQ_MODEL_XXX     # legacy alias

_OPENROUTER_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

if not _OPENROUTER_KEY:
    logger.warning(
        "No OPENROUTER_API_KEY found. Agent model calls will fail. Add it to .env"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Key rotation (Groq)
# ─────────────────────────────────────────────────────────────────────────────

_current_key_idx: int = 0


def _get_groq_client() -> Groq:
    if not _GROQ_KEYS:
        raise RuntimeError(
            "No Groq API keys configured. Add GROQ_API_KEY to your .env file."
        )
    return Groq(api_key=_GROQ_KEYS[_current_key_idx])


# Keep the old name as an alias so nothing outside this file breaks
_get_client = _get_groq_client


def _rotate_key() -> bool:
    global _current_key_idx
    if _current_key_idx + 1 < len(_GROQ_KEYS):
        _current_key_idx += 1
        logger.warning("[llm_client] Rate limit — rotating to key #%d", _current_key_idx + 1)
        return True
    logger.error("[llm_client] All %d Groq keys rate-limited.", len(_GROQ_KEYS))
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Low-level callers — OpenRouter (agent model)
# ─────────────────────────────────────────────────────────────────────────────

def _call_openrouter(messages: List[Dict], temperature: float, max_tokens: int, model: str) -> str:
    if not _OPENROUTER_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not configured. Add it to your .env file.")
    try:
        client = openai.OpenAI(api_key=_OPENROUTER_KEY, base_url=_OPENROUTER_BASE_URL)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        raise RuntimeError(f"OpenRouter call failed: {exc}") from exc


def _call_openrouter_stream(
    messages: List[Dict], temperature: float, max_tokens: int, model: str
) -> Generator[str, None, None]:
    if not _OPENROUTER_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not configured. Add it to your .env file.")
    try:
        client = openai.OpenAI(api_key=_OPENROUTER_KEY, base_url=_OPENROUTER_BASE_URL)
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
    except Exception as exc:
        raise RuntimeError(f"OpenRouter stream failed: {exc}") from exc


# ─────────────────────────────────────────────────────────────────────────────
# Low-level callers — Groq (utility model)
# ─────────────────────────────────────────────────────────────────────────────

def _call_groq(messages: List[Dict], temperature: float, max_tokens: int, model: str) -> str:
    last_exc = None
    for _ in range(len(_GROQ_KEYS) + 1):
        try:
            client = _get_groq_client()
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except RateLimitError as exc:
            last_exc = exc
            if not _rotate_key():
                break
            time.sleep(0.5)
        except Exception as exc:
            raise RuntimeError(f"Groq call failed: {exc}") from exc
    raise RuntimeError(f"All Groq keys exhausted. Last error: {last_exc}")


def _call_groq_stream(
    messages: List[Dict], temperature: float, max_tokens: int, model: str
) -> Generator[str, None, None]:
    last_exc = None
    for _ in range(len(_GROQ_KEYS) + 1):
        try:
            client = _get_groq_client()
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
            return
        except RateLimitError as exc:
            last_exc = exc
            if not _rotate_key():
                break
            time.sleep(0.5)
        except Exception as exc:
            raise RuntimeError(f"Groq stream failed: {exc}") from exc
    raise RuntimeError(f"All Groq keys exhausted. Last error: {last_exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Router — sends agent model to OpenRouter, utility model to Groq
# ─────────────────────────────────────────────────────────────────────────────

def _route_call(messages: List[Dict], temperature: float, max_tokens: int, model: str) -> str:
    if model == MODEL_AGENT:
        return _call_openrouter(messages, temperature, max_tokens, model)
    return _call_groq(messages, temperature, max_tokens, model)


def _route_stream(
    messages: List[Dict], temperature: float, max_tokens: int, model: str
) -> Generator[str, None, None]:
    if model == MODEL_AGENT:
        yield from _call_openrouter_stream(messages, temperature, max_tokens, model)
    else:
        yield from _call_groq_stream(messages, temperature, max_tokens, model)


# ─────────────────────────────────────────────────────────────────────────────
# Public blocking API
# ─────────────────────────────────────────────────────────────────────────────

def llm_call(
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 600,
    retry_delay: float = 0.5,
    model: str = None,
) -> str:
    """Blocking chat. Pass model= to override default (e.g. MODEL_AGENT)."""
    chosen_model = model or GROQ_MODEL_XXX
    try:
        return _route_call(messages, temperature, max_tokens, chosen_model)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"LLM call failed: {exc}") from exc


def llm_call_json(
    prompt: str,
    system: str = "You output ONLY valid JSON. No explanations, no markdown, just JSON.",
    temperature: float = 0,
    max_tokens: int = 500,
    model: str = None,
) -> str:
    """JSON-only blocking call — used by judge, preprocessor, context_resolver."""
    return llm_call(
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        model=model or GROQ_MODEL_XXX,
    )


def llm_call_text(
    system: str,
    user: str,
    temperature: float = 0.3,
    max_tokens: int = 600,
    model: str = None,
) -> str:
    """Blocking text/conversation call."""
    return llm_call(
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        model=model or GROQ_MODEL_XXX,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public streaming API
# ─────────────────────────────────────────────────────────────────────────────

def llm_call_stream(
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 600,
    retry_delay: float = 0.5,
    model: str = None,
) -> Generator[str, None, None]:
    """Sync generator — yields text chunks."""
    chosen_model = model or GROQ_MODEL_XXX
    try:
        yield from _route_stream(messages, temperature, max_tokens, chosen_model)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"LLM stream failed: {exc}") from exc


def llm_call_stream_text(
    system: str,
    user: str,
    temperature: float = 0.3,
    max_tokens: int = 600,
    model: str = None,
) -> Generator[str, None, None]:
    """Streaming text/conversation call — yields str chunks."""
    yield from llm_call_stream(
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics — run: python llm_client.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== LLM Client Status ===")
    print(f"  Groq keys    : {len(_GROQ_KEYS)} (GROQ_API_KEY through GROQ_API_KEY5)")
    print(f"  OpenRouter   : {'configured' if _OPENROUTER_KEY else 'MISSING'}")
    print(f"  Agent model  : {MODEL_AGENT}  → OpenRouter")
    print(f"  Utility model: {GROQ_MODEL_XXX}  → Groq")

    if not _GROQ_KEYS:
        print("\n❌  No API keys found — add GROQ_API_KEY to your .env file.")
        exit(1)

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