"""
chatbot_api.py — Mobile App API Layer  (LangGraph Agent Version)
=================================================================

This is the single entry point your mobile app (and api_server.py) calls.

Public functions
────────────────
  chat(student_id, message)              → {ok, student_id, response}
  get_student_info(student_id)           → {ok, student_id, first_name, gpa, …}
  clear_history(student_id)              → {ok, student_id}
  get_disambiguation_options(sid, term)  → {ok, student_id, candidates}

How chat() works
────────────────
  1. Load the student's recent chat history from Supabase.
  2. Route message:
       a. Pending ambiguity session? → resolve disambiguation reply
       b. Normal query              → preprocess → BNUAdvisorAgent
  3. Save both the user message and the agent's response to Supabase.
"""

from __future__ import annotations

import json
import logging
import os
import re
import traceback
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
from eligibility import TRACK_MAP as _TRACK_MAP

load_dotenv()

# ── In-memory ambiguity session cache ────────────────────────────────────────
# Maps student_id → PendingAmbiguity object.
# Set when the preprocessor finds multiple close-matching courses and needs
# the student to pick one.  Cleared as soon as the student replies.
_ambiguity_sessions: Dict[str, Any] = {}


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════

def chat(student_id: str, message: str) -> Dict[str, Any]:
    """
    Main entry point for every student message.

    Args:
        student_id: Student's unique identifier from the sign-in session.
        message:    The student's text message.

    Returns:
        {
            "ok":         bool,
            "student_id": str,
            "response":   str,   # present when ok=True
            "error":      str,   # present when ok=False
            "trace":      str,   # present when ok=False (stack trace)
        }
    """
    if not student_id or not message:
        return {
            "ok":         False,
            "student_id": student_id,
            "error":      "Both student_id and message are required.",
        }

    try:
        from chatbot_connector import ChatbotConnector
        from language_service import (
            detect_and_translate_input,
            translate_to_arabic,
            translate_history_to_english,
        )
        connector = ChatbotConnector()

        # ── 1. Load chat history for context continuity ───────────────────
        chat_data = connector.get_chat_history(student_id)
        history   = chat_data.get("chat_history", []) if chat_data else []

        # ── 2. Detect language + translate input to English ───────────────
        original_lang, english_message = detect_and_translate_input(message)

        # ── 2b. Inherit language from a pending ambiguity session ──────────
        # When the student answers a disambiguation question with "1", "2",
        # or any neutral text, detection returns "english" even though the
        # original query was Arabic.  Use the language stored with the
        # pending session so the final answer is still translated back.
        _stored = _ambiguity_sessions.get(student_id)
        if isinstance(_stored, dict) and _stored.get("lang", "english") != "english":
            original_lang = _stored["lang"]

        # ── 3. Translate history window to English if needed ──────────────
        english_history = (
            translate_history_to_english(history)
            if original_lang != "english"
            else history
        )

        # ── 4. Route through pipeline (always English) ────────────────────
        response = _route_message(student_id, english_message, english_history, original_lang)

        # ── 5. Translate response back to student's language if needed ─────
        if original_lang != "english":
            final_response = translate_to_arabic(response)
            response_lang  = "arabic"
        else:
            final_response = response
            response_lang  = "english"

        # ── 6. Persist to Supabase ────────────────────────────────────────
        # Store the original (untranslated) user message so the mobile app
        # displays what the student actually typed.
        connector.add_message(student_id, "user",      message,        lang=original_lang)
        connector.add_message(student_id, "assistant", final_response, lang=response_lang)

        return {
            "ok":         True,
            "student_id": student_id,
            "response":   final_response,
        }

    except Exception as exc:
        # Clear any pending ambiguity session so the next message
        # doesn't get misrouted as a disambiguation reply.
        _ambiguity_sessions.pop(student_id, None)
        return {
            "ok":         False,
            "student_id": student_id,
            "error":      str(exc),
            "trace":      traceback.format_exc(),
        }


def get_student_info(student_id: str) -> Dict[str, Any]:
    """
    Fetch student profile + academic details from Supabase.
    Call this once when the student logs in to populate the app UI.

    Returns:
        {ok, student_id, first_name, last_name, track, university_year,
         gpa, earned_credits, completed_courses}
    """
    try:
        from chatbot_connector import ChatbotConnector
        connector = ChatbotConnector()

        student = connector.get_student_data(student_id)
        if not student:
            return {
                "ok":         False,
                "student_id": student_id,
                "error":      "Student not found.",
            }

        academic = connector.get_or_initialize_academic_details(student_id)
        return {
            "ok":                True,
            "student_id":        student_id,
            "first_name":        student.get("first_name",       ""),
            "last_name":         student.get("last_name",        ""),
            "track":             _TRACK_MAP.get((student.get("track") or "").strip().upper(),
                                             student.get("track", "")),
            "university_year":   student.get("university_year",  0),
            "gpa":               academic.get("gpa",             0.0),
            "earned_credits":    academic.get("earned_credits",  0),
            "completed_courses": academic.get("completed_courses", []),
        }

    except Exception as exc:
        return {"ok": False, "student_id": student_id, "error": str(exc)}


def clear_history(student_id: str) -> Dict[str, Any]:
    """
    Clear the student's chat history and any pending disambiguation session.
    Call this on logout or when the student starts a fresh conversation.
    """
    try:
        from chatbot_connector import ChatbotConnector
        ChatbotConnector().clear_chat_history(student_id)

        # Cancel any pending ambiguity session
        _ambiguity_sessions.pop(student_id, None)

        return {"ok": True, "student_id": student_id}

    except Exception as exc:
        return {"ok": False, "student_id": student_id, "error": str(exc)}


def get_disambiguation_options(student_id: str, term: str) -> Dict[str, Any]:
    """
    Return candidate course names when the student typed an ambiguous name.
    Use this to power an autocomplete/picker in the mobile UI.

    Returns:
        {"ok": True, "candidates": [{"name": ..., "code": ..., "confidence": ...}]}
    """
    try:
        from course_name_mapper import get_ambiguous_matches
        return {
            "ok":         True,
            "student_id": student_id,
            "candidates": get_ambiguous_matches(term),
        }
    except Exception as exc:
        return {"ok": False, "student_id": student_id, "error": str(exc)}


# ═════════════════════════════════════════════════════════════════════════════
# Internal routing
# ═════════════════════════════════════════════════════════════════════════════

def _route_message(
    student_id:    str,
    message:       str,
    history:       list,
    original_lang: str = "english",
) -> str:
    """
    Route the student's message through the full pipeline:

    Priority order:
    1. Pending ambiguity  → student is answering a disambiguation question
    2. Normal query       → preprocess → agent
    """

    # ── 1. Resolve pending ambiguity ──────────────────────────────────────
    stored = _ambiguity_sessions.get(student_id)
    if stored is not None:
        pending = stored["pending"] if isinstance(stored, dict) else stored
        return _resolve_ambiguity_reply(student_id, message, pending, original_lang)

    # ── 2. Preprocess → agent ─────────────────────────────────────────────
    return _preprocess_and_run(student_id, message, history, original_lang)


def _preprocess_and_run(
    student_id:    str,
    message:       str,
    history:       list,
    original_lang: str = "english",
) -> str:
    """
    Run the preprocessing pipeline, then the agent.

    If preprocessing finds an ambiguous course name, it stores a
    PendingAmbiguity in _ambiguity_sessions and returns the clarification
    question to the student.  The agent is NOT called yet.

    On the student's next message, _resolve_ambiguity_reply() picks up.
    """
    from preprocessor import get_preprocessor
    from chatbot_connector import ChatbotConnector

    # Fetch the student's enrolled track so the preprocessor can use it as
    # a secondary hint when resolving track-vs-course conflicts.
    student_track: Optional[str] = None
    try:
        student_data  = ChatbotConnector().get_student_data(student_id)
        if student_data:
            student_track = _TRACK_MAP.get(
                (student_data.get("track") or "").strip().upper(), None
            )
    except Exception:
        pass  # heuristic only — safe to skip on error

    pre    = get_preprocessor()
    result = pre.process(message, history, student_track=student_track)

    if result.status == "ambiguous":
        # Store pending state + original language so disambiguation replies
        # are translated back to the student's language correctly.
        _ambiguity_sessions[student_id] = {"pending": result.pending, "lang": original_lang}
        return result.clarification

    # "ready" or "passthrough" — we have a clean query
    clean_query = result.clean_query or message

    # ── Lecture mode — bypass the agent entirely ───────────────────────────
    # When a lecture PDF is attached, answer directly with llm_call_text.
    # The agent is NOT used: its system prompt + tool schemas alone cost
    # ~6 000 tokens, which overflows the 8 000 TPM cap when lecture text
    # is added on top.  A direct LLM call uses only the lecture + question.
    try:
        import lecture_service
        lecture = lecture_service.get_lecture_context(student_id)
        if lecture:
            print(f"\n📖 LECTURE MODE active for {student_id!r} — '{lecture['name']}' "
                  f"({len(lecture['text'])} chars) — bypassing agent")
            return _answer_from_lecture(clean_query, lecture, history)
    except Exception as exc:
        logger.warning("[chatbot_api] lecture bypass error: %s", exc)
        print(f"⚠️  lecture bypass error: {exc}")
        # Fall through to normal agent if something goes wrong

    sub_queries = _analyze_and_split(clean_query)
    if len(sub_queries) > 1:
        return _split_and_run(student_id, clean_query, sub_queries, history)

    from agent import BNUAdvisorAgent
    agent = BNUAdvisorAgent(student_id=student_id)
    return agent.run(query=clean_query, history=history)


def _resolve_ambiguity_reply(
    student_id:    str,
    reply:         str,
    pending:       any,
    original_lang: str = "english",
) -> str:
    """
    The student has answered a disambiguation question.
    Resolve the ambiguity, clear the pending state, and run the agent
    with the now-fully-resolved query.
    """
    from preprocessor import get_preprocessor

    # Remove pending state immediately so we don't re-enter this on next msg
    _ambiguity_sessions.pop(student_id, None)

    pre    = get_preprocessor()
    result = pre.resolve_ambiguity(pending, reply)

    # Another course in the original query was also ambiguous — ask again
    if result.status == "ambiguous":
        # Preserve the original language for the chained disambiguation turn
        _ambiguity_sessions[student_id] = {"pending": result.pending, "lang": original_lang}
        return result.clarification

    clean_query = result.clean_query or pending.original_query

    sub_queries = _analyze_and_split(clean_query)
    if len(sub_queries) > 1:
        return _split_and_run(student_id, clean_query, sub_queries, pending.history)

    from agent import BNUAdvisorAgent
    agent = BNUAdvisorAgent(student_id=student_id)
    return agent.run(query=clean_query, history=pending.history)


def _answer_from_lecture(question: str, lecture: dict, history: list) -> str:
    """
    Answer a student question using ONLY the attached lecture PDF content.
    Bypasses the LangGraph agent entirely — uses a direct llm_call_text()
    so we never send tool schemas or agent system prompts alongside the
    lecture text (which would exceed the 8 000-token TPM limit).

    Args:
        question: The student's (preprocessed) question.
        lecture:  Dict with keys "text" (str) and "name" (str).
        history:  Recent conversation turns for multi-turn context.

    Returns:
        The LLM's answer as a plain string.
    """
    from llm_client import llm_call_text, GROQ_MODEL_XXX

    lecture_name = lecture["name"]
    lecture_text = lecture["text"]

    # Fit within ~5 000 chars of lecture text to leave room for the question,
    # history, and the system prompt while staying well under 8 000 tokens.
    MAX_LECTURE_CHARS = 5_000
    if len(lecture_text) > MAX_LECTURE_CHARS:
        lecture_text = lecture_text[:MAX_LECTURE_CHARS] + "\n… [truncated]"

    # Build a compact history block (last 3 turns max)
    history_block = ""
    if history:
        turns = []
        for turn in history[-6:]:
            role = "Student" if turn.get("role") == "user" else "Advisor"
            turns.append(f"{role}: {turn.get('content', '')[:200]}")
        if turns:
            history_block = "\n\nPrevious conversation:\n" + "\n".join(turns)

    system = (
        "You are the BNU Academic Advisor helping a student understand their lecture material. "
        "Answer ONLY based on the lecture content provided — do not use outside knowledge. "
        "If the lecture does not contain enough information to answer, say so clearly. "
        "Be concise and use bullet points or bold text where helpful. "
        "Never mention tools, databases, or system internals."
    )

    user = (
        f"Lecture: {lecture_name}\n\n"
        f"--- LECTURE CONTENT ---\n{lecture_text}\n--- END OF LECTURE ---"
        f"{history_block}\n\n"
        f"Student question: {question}"
    )

    try:
        return llm_call_text(
            system=system,
            user=user,
            temperature=0.3,
            max_tokens=1000,
            model=GROQ_MODEL_XXX,   # utility model — lower token cost, no tool schema overhead
        )
    except Exception as exc:
        logger.error("[_answer_from_lecture] LLM call failed: %s", exc)
        return (
            "I had trouble reading the lecture content. "
            "Please try again or rephrase your question."
        )


def _analyze_and_split(clean_query: str) -> List[str]:
    """
    Ask the utility LLM to split the query into one sub-query per user intent.

    Intent-driven splitting — never permutation/cartesian expansion of non-factual clauses:
      COMPARISON clause  → always ONE sub-query (atomic, never split by items compared)
      RECOMMEND  clause  → always ONE sub-query (atomic, never split by options)
      CAUSAL clause      → always ONE sub-query (premise "I finished/passed X" + consequence
                           "what does that unlock?" are one intent, never split)
      FACTUAL clause     → N courses → N; 1 course × M programs → M; N×M only for pure factual
      Independent topics → one per distinct intent

    Returns a list of sub-query strings.  Single-element list → no split,
    normal single-agent path is taken.
    """
    from llm_client import llm_call_json

    try:
        raw = llm_call_json(
            system=(
                "You split student academic advisor queries into the minimum number of "
                "independent sub-queries, one per distinct user intent. "
                "Output ONLY valid JSON — no markdown, no explanations, no prose."
            ),
            max_tokens=1500,
            prompt=(
                "Split the query below into sub-queries, one per user intent.\n\n"
                "══ STEP 0 — RESOLVE INTRA-QUERY REFERENCES ══\n"
                "Before splitting, resolve any pronouns or vague references that point to an entity\n"
                "already named within the SAME query. Replace the pronoun with the referent.\n"
                "The referent can be anything: a course, program, track, elective, academic rule,\n"
                "concept, or any other entity. Use the surrounding context to decide.\n"
                "  Pronouns to check: it, this, that, them, those, one, ones, there, here, …\n"
                "  Examples:\n"
                "    'what is NLP and I love it'       → 'what is NLP and I love NLP'\n"
                "    'tell me about ML, is it hard?'   → 'tell me about ML, is ML hard?'\n"
                "    'what are the electives and when can I take them?' → already clear, no change\n"
                "    'compare AIM and SAD, which is better for me?'    → already clear, no change\n"
                "  If a pronoun is ambiguous or has no referent in the query, leave it as-is.\n"
                "  Work on the RESOLVED query in all subsequent steps.\n\n"
                "══ STEP 1 — FIND INTENT BOUNDARIES ══\n"
                "Identify every intent-bearing verb or phrase and what it governs:\n"
                "  COMPARISON  — 'compare', 'vs', 'difference between', 'X or Y (choice)'\n"
                "  RECOMMEND   — 'which should I choose', 'what is best for me', 'recommend'\n"
                "  CAUSAL      — a PREMISE clause ('I finished/passed/completed/got/took X [and Y…]',\n"
                "                'I have X done', 'now that I passed X') paired with a CONSEQUENCE\n"
                "                question ('what does that unlock?', 'what can I take now?',\n"
                "                'what am I eligible for?', 'what does it close?', 'what comes next?',\n"
                "                'what does finishing X do?', 'what opens up?').\n"
                "                The premise supplies context FOR the question — they are one intent.\n"
                "  FACTUAL     — everything else (what is, when can I take, prerequisites, etc.)\n\n"
                "  KEY: every course/program that appears as an argument of a COMPARISON, RECOMMEND,\n"
                "  or CAUSAL clause is OWNED BY that clause and must stay inside it. Never extract\n"
                "  those entities as separate sub-queries.\n\n"
                "══ STEP 2 — ONE SUB-QUERY PER INTENT ══\n"
                "  COMPARISON clause → exactly ONE sub-query containing the whole clause.\n"
                "    – All compared items (courses, programs, tracks, any mix) stay together.\n"
                "    – Never split 'compare X and Y and Z' into sub-queries for X, Y, Z.\n"
                "  RECOMMEND  clause → exactly ONE sub-query containing the whole clause.\n"
                "  CAUSAL clause → exactly ONE sub-query containing BOTH the premise and the question.\n"
                "    – Never split the completion/premise half from its consequence question.\n"
                "    – 'I finished X and Y, what does that unlock?' → one sub-query, not two.\n"
                "    – Even if multiple courses appear in the premise, keep all of them together\n"
                "      with the consequence question in one sub-query.\n"
                "  FACTUAL clause (only truly independent factual questions):\n"
                "    • 1 entity                           → 1 sub-query\n"
                "    • N independent courses, same intent → N sub-queries (one per course)\n"
                "    • 1 course × M programs, same intent → M sub-queries (one per program)\n"
                "    • Already atomic                     → return as-is\n"
                "    • Multiple attributes of the SAME entity (student profile, one course,\n"
                "      one program) — e.g. 'my completed courses and their grades',\n"
                "      'ML credits and prerequisites' — → exactly ONE sub-query.\n"
                "      A possessive pronoun ('their', 'its') referencing an entity in the\n"
                "      same query is a strong signal that both parts form one compound\n"
                "      question about the same thing.\n\n"
                "  NEVER apply cartesian expansion to items that are arguments of a COMPARISON\n"
                "  or RECOMMEND clause, even if those items look like independent entities.\n\n"
                "══ STEP 3 — SMELL TEST (apply before finalizing) ══\n"
                "  ✗ Did the user actually ask for each generated sub-query? If no → remove it.\n"
                "  ✗ Is any sub-query one of the arguments of a comparison/recommend clause\n"
                "    extracted on its own? → the splitter broke an atomic clause. Merge it back.\n"
                "  ✗ Are all sub-queries about attributes of the same entity (same course,\n"
                "    same student profile, same program)? → collapse into one sub-query.\n"
                "  ✗ Does any sub-query contain a pronoun ('one', 'the other', 'it', 'this',\n"
                "    'that', 'they', 'them', 'those', 'either') whose referent appears ONLY in\n"
                "    a different sub-query, not within itself? → those sub-queries must be merged\n"
                "    into a single sub-query with the pronoun replaced by its referent.\n"
                "    A sub-query must be fully self-contained and independently understandable.\n"
                "  ✗ Do all sub-queries together faithfully reconstruct the original intent\n"
                "    with no added or lost meaning? If no → revise.\n"
                "  ✗ Is one sub-query a course-completion premise ('I finished/passed/got X')\n"
                "    and another sub-query is the consequence question ('what does X unlock?',\n"
                "    'what can I take?') for those SAME courses? → that is a CAUSAL clause;\n"
                "    merge both halves into a single sub-query.\n\n"
                "══ OUTPUT RULES ══\n"
                "  • Write sub-queries using the RESOLVED query text from Step 0 (pronouns replaced).\n"
                "  • Otherwise copy verbatim — no extra rephrasing or reformatting.\n"
                "  • NEVER convert 'and' to '&' or '&' to 'and'.\n"
                "  • NEVER merge two distinct intents into one sub-query.\n\n"
                "══ PROGRAM NAME PROTECTION ══\n"
                "These names contain '&' and must NEVER be split across tokens:\n"
                "  • artificial intelligence & machine learning\n"
                "  • software & application development\n"
                "  • data science\n\n"
                "══ EXAMPLES ══\n"
                '  COMPARISON with mixed entity types (program + courses) — stays atomic:\n'
                '  "compare artificial intelligence & machine learning program and artificial intelligence course and natural language processing course"\n'
                '  → ["compare artificial intelligence & machine learning program and artificial intelligence course and natural language processing course"]\n\n'
                '  COMPARISON between two programs — stays atomic:\n'
                '  "compare artificial intelligence & machine learning and software & application development"\n'
                '  → ["compare artificial intelligence & machine learning and software & application development"]\n\n'
                '  COMPARISON + separate FACTUAL (two distinct intents):\n'
                '  "compare NLP and image processing in artificial intelligence & machine learning and what is the GPA"\n'
                '  → ["compare NLP and image processing in artificial intelligence & machine learning",\n'
                '     "what is the GPA"]\n\n'
                '  FACTUAL, 2 independent courses:\n'
                '  "what is artificial intelligence and what is machine learning"\n'
                '  → ["what is artificial intelligence", "what is machine learning"]\n\n'
                '  FACTUAL, 1 course × 2 programs:\n'
                '  "when can I take ML at artificial intelligence & machine learning and software & application development"\n'
                '  → ["when can I take ML at artificial intelligence & machine learning",\n'
                '     "when can I take ML at software & application development"]\n\n'
                '  Pronoun resolved, stays atomic:\n'
                '  "what is \'natural language processing\' course and i love it"\n'
                '  → Step 0: "it" → "\'natural language processing\' course"\n'
                '  → ["what is \'natural language processing\' course and i love \'natural language processing\' course"]\n\n'
                '  FACTUAL, multiple attributes of the same entity — stays atomic:\n'
                '  "what are my completed courses and their grades"\n'
                '  → ["what are my completed courses and their grades"]\n\n'
                '  FACTUAL, multiple attributes of one course — stays atomic:\n'
                '  "how many credits is ML and what are its prerequisites"\n'
                '  → ["how many credits is ML and what are its prerequisites"]\n\n'
                '  CAUSAL — premise + consequence stays atomic:\n'
                '  "I finished \'data structures\' and \'design & analysis of algorithms\' and \'object oriented programming\' — what does that unlock in artificial intelligence & machine learning?"\n'
                '  → ["I finished \'data structures\' and \'design & analysis of algorithms\' and \'object oriented programming\' — what does that unlock in artificial intelligence & machine learning?"]\n\n'
                '  CAUSAL — single course completion + question, stays atomic:\n'
                '  "I passed machine learning, what am I eligible for now?"\n'
                '  → ["I passed machine learning, what am I eligible for now?"]\n\n'
                '  CAUSAL + separate FACTUAL (two distinct intents):\n'
                '  "I finished machine learning, what does it unlock, and also what is deep learning about?"\n'
                '  → ["I finished machine learning, what does it unlock?",\n'
                '     "what is deep learning about?"]\n\n'
                '  Already atomic:\n'
                '  "what is ML about?" → ["what is ML about?"]\n\n'
                f'Query: "{clean_query}"\n\n'
                'Apply the rules above silently. Output ONLY this JSON and nothing else:\n'
                '{"sub_queries": ["..."]}'
            ),
        )
        # llm_call_json returns a raw string — extract the JSON object robustly.
        # Use raw_decode so we stop at the first complete JSON object and ignore
        # any trailing text (extra objects, explanatory notes, etc.).
        raw = raw.strip()
        # Find the first '{' to start decoding from
        brace = raw.find('{')
        if brace == -1:
            raise ValueError(f"No JSON object found in LLM response: {raw!r}")
        data, _ = json.JSONDecoder().raw_decode(raw, brace)
        sub_queries = [q.strip() for q in data.get("sub_queries", []) if q.strip()]

        from debug_box import box as _box
        _box(
            "🔀  QUERY SPLIT DECISION",
            [f"Input  : {clean_query}"]
            + [f"Sub [{i+1}]: {q}" for i, q in enumerate(sub_queries)],
            force=True,
        )

        if sub_queries:
            return sub_queries
    except Exception as exc:
        from debug_box import box as _box
        _box("🔀  QUERY SPLIT DECISION", [f"ERROR: {exc}", f"Input: {clean_query}"], force=True)
        logger.warning("_analyze_and_split failed (%s) — using original query", exc)

    return [clean_query]


def _split_and_run(
    student_id:  str,
    clean_query: str,
    sub_queries: List[str],
    history:     list,
    verbose:     bool = False,
) -> str:
    """
    Run the agent pipeline once per sub-query to collect tool context,
    then synthesise ONE final answer from the combined context.

    When verbose=True each sub-query run streams its debug boxes so the
    user can follow tool selection, results, and judge verdicts.
    """
    from agent import BNUAdvisorAgent, _get_answer_llm, _ANSWER_SYSTEM
    from langchain_core.messages import SystemMessage, HumanMessage
    from debug_box import box as _box, is_verbose as _is_verbose

    verbose = verbose or _is_verbose()

    if len(sub_queries) == 1:
        agent = BNUAdvisorAgent(student_id=student_id)
        return agent.run(query=sub_queries[0], history=history, verbose=verbose)

    # Collect tool context for every sub-query
    all_contexts: List[str] = []
    for i, sub_query in enumerate(sub_queries, 1):
        if verbose:
            _box(
                f"🔀  SUB-QUERY {i} / {len(sub_queries)}",
                [sub_query],
                force=True,
            )
        agent = BNUAdvisorAgent(student_id=student_id)
        ctx = agent.run_and_get_context(sub_query, history, verbose=verbose)
        all_contexts.extend(ctx)

    if not all_contexts:
        # No tool results at all — fall back to a single agent run
        agent = BNUAdvisorAgent(student_id=student_id)
        return agent.run(query=clean_query, history=history, verbose=verbose)

    if verbose:
        _box("✅  ANSWER NODE  →  generating final response", [], force=True)

    # One final synthesis from all collected context
    combined = "\n\n---\n\n".join(all_contexts)
    return _get_answer_llm().invoke([
        SystemMessage(content=_ANSWER_SYSTEM),
        HumanMessage(content=(
            f"Student question: {clean_query}\n\n"
            f"Information from the BNU database:\n{combined}\n\n"
            "Answer the question clearly and concisely based on the information above."
        )),
    ]).content


# ═════════════════════════════════════════════════════════════════════════════
# Smoke test — run: python chatbot_api.py
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    TEST_ID = os.getenv("STUDENT_ID", "22030094")

    print("=" * 60)
    print("Chatbot API Smoke Test  (LangGraph Agent)")
    print("=" * 60)

    print("\n── Student info ──")
    info = get_student_info(TEST_ID)
    print(json.dumps(info, indent=2, default=str))

    print("\n── Test chat query ──")
    result = chat(TEST_ID, "What are the prerequisites for Machine Learning?")
    print(json.dumps(result, indent=2, default=str))