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
  2. Check if there is an active multi-turn course planning session:
       YES  → forward the student's reply to PlanningOrchestrator.advance()
       NO   → run the BNUAdvisorAgent (LangGraph ReAct loop)
  3. If the agent calls the start_course_planning tool, the PlanningState
     is captured via a patched version of the tool (see bottom of this file)
     and cached in _planning_sessions[student_id].
  4. Save both the user message and the agent's response to Supabase.

Planning session lifecycle
──────────────────────────
  start_course_planning tool called
        │
        ▼  chatbot_api captures PlanningState
  _planning_sessions[student_id] = state
        │
        ▼  next student message
  PlanningOrchestrator.advance(state, reply)
        │
        ▼  when state.current_step == PlanStep.COMPLETE
  del _planning_sessions[student_id]
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

# ── In-memory planning session cache ─────────────────────────────────────────
# Maps student_id → active PlanningState object.
# PlanningState is produced by PlanningOrchestrator.start() and consumed
# turn-by-turn by PlanningOrchestrator.advance().
# Cleared on clear_history() or when planning completes.
_planning_sessions: Dict[str, Any] = {}

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
        connector = ChatbotConnector()

        # ── 1. Load chat history for context continuity ───────────────────
        # Supabase stores the last 3 user + 3 assistant messages.
        # We pass them to the agent so it can continue naturally mid-session.
        chat_data = connector.get_chat_history(student_id)
        history   = chat_data.get("chat_history", []) if chat_data else []

        # ── 2. Route message ──────────────────────────────────────────────
        response = _route_message(student_id, message, history)

        # ── 3. Persist to Supabase ────────────────────────────────────────
        connector.add_message(student_id, "user",      message)
        connector.add_message(student_id, "assistant", response)

        return {
            "ok":         True,
            "student_id": student_id,
            "response":   response,
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
    Clear the student's chat history and cancel any active planning session.
    Call this on logout or when the student starts a fresh conversation.
    """
    try:
        from chatbot_connector import ChatbotConnector
        ChatbotConnector().clear_chat_history(student_id)

        # Cancel any in-progress planning or ambiguity session
        _planning_sessions.pop(student_id, None)
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
    student_id: str,
    message:    str,
    history:    list,
) -> str:
    """
    Route the student's message through the full pipeline:

    Priority order:
    1. Pending ambiguity  → student is answering a disambiguation question
    2. Active planning    → student is in a multi-turn planning session
    3. Normal query       → preprocess → agent
    """

    # ── 1. Resolve pending ambiguity ──────────────────────────────────────
    pending_ambiguity = _ambiguity_sessions.get(student_id)
    if pending_ambiguity is not None:
        return _resolve_ambiguity_reply(student_id, message, pending_ambiguity)

    # ── 2. Active planning session ────────────────────────────────────────
    active_state = _planning_sessions.get(student_id)
    if active_state is not None:
        return _advance_planning(student_id, message, active_state)

    # ── 3. Preprocess → agent ─────────────────────────────────────────────
    return _preprocess_and_run(student_id, message, history)


def _preprocess_and_run(
    student_id: str,
    message:    str,
    history:    list,
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
        # Store the pending state and ask the student to clarify
        _ambiguity_sessions[student_id] = result.pending
        return result.clarification

    # "ready" or "passthrough" — we have a clean query
    clean_query = result.clean_query or message

    sub_queries = _analyze_and_split(clean_query)
    if len(sub_queries) > 1:
        return _split_and_run(student_id, clean_query, sub_queries, history)

    from agent import BNUAdvisorAgent
    agent = BNUAdvisorAgent(student_id=student_id)
    return agent.run(query=clean_query, history=history)


def _resolve_ambiguity_reply(
    student_id: str,
    reply:      str,
    pending:    any,
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
        _ambiguity_sessions[student_id] = result.pending
        return result.clarification

    clean_query = result.clean_query or pending.original_query

    sub_queries = _analyze_and_split(clean_query)
    if len(sub_queries) > 1:
        return _split_and_run(student_id, clean_query, sub_queries, pending.history)

    from agent import BNUAdvisorAgent
    agent = BNUAdvisorAgent(student_id=student_id)
    return agent.run(query=clean_query, history=pending.history)


def _analyze_and_split(clean_query: str) -> List[str]:
    """
    Ask the utility LLM to split the query into one sub-query per user intent.

    Intent-driven splitting — never permutation/cartesian expansion of non-factual clauses:
      COMPARISON clause  → always ONE sub-query (atomic, never split by items compared)
      RECOMMEND  clause  → always ONE sub-query (atomic, never split by options)
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
                "independent sub-queries, one per distinct user intent."
            ),
            prompt=(
                "Split the query below into sub-queries, one per user intent.\n\n"
                "══ STEP 1 — CLASSIFY EACH CLAUSE ══\n"
                "Read the query and label every clause with one of:\n"
                "  COMPARISON  — contains: compare, vs, difference between, X or Y (choice)\n"
                "  RECOMMEND   — contains: which should I choose, what is best for me, recommend\n"
                "  FACTUAL     — everything else (what is, when can I take, prerequisites, etc.)\n\n"
                "══ STEP 2 — GENERATE SUB-QUERIES BY INTENT ══\n"
                "  COMPARISON clause → exactly ONE sub-query (the whole clause, never split by items)\n"
                "  RECOMMEND  clause → exactly ONE sub-query (the whole clause, never split by options)\n"
                "  FACTUAL clause:\n"
                "    • 1 entity                → 1 sub-query\n"
                "    • N courses, no program   → N sub-queries (one per course)\n"
                "    • 1 course × M programs   → M sub-queries (one per program)\n"
                "    • N courses × M programs  → N×M sub-queries (cartesian — ONLY for FACTUAL)\n"
                "    • Already atomic          → return as-is\n\n"
                "══ STEP 3 — SMELL TEST (apply before finalizing) ══\n"
                "  ✗ Did the user actually ask for each generated sub-query? If no → remove it.\n"
                "  ✗ Is any sub-query a duplicate of another with one variable swapped?\n"
                "    → the splitter cross-product-expanded a non-FACTUAL clause. Fix it.\n"
                "  ✗ Do all sub-queries together faithfully reconstruct the original intent\n"
                "    with no added or lost meaning? If no → revise.\n\n"
                "══ OUTPUT RULES ══\n"
                "  • Extract sub-queries verbatim — copy words EXACTLY from the input.\n"
                "  • Do NOT rephrase, reword, reformat, or add/remove punctuation.\n"
                "  • NEVER convert 'and' to '&' or '&' to 'and'.\n"
                "  • NEVER merge two clauses into one sub-query.\n\n"
                "══ PROGRAM NAME PROTECTION ══\n"
                "These names contain '&' and must NEVER be split across tokens:\n"
                "  • artificial intelligence & machine learning\n"
                "  • software & application development\n"
                "  • data science\n\n"
                "══ EXAMPLES ══\n"
                '  FACTUAL, 2 courses:\n'
                '  "what is artificial intelligence and machine learning"\n'
                '  → ["what is artificial intelligence", "what is machine learning"]\n\n'
                '  FACTUAL, 1 course × 2 programs:\n'
                '  "when can I take ML at artificial intelligence & machine learning and software & application development"\n'
                '  → ["when can I take ML at artificial intelligence & machine learning",\n'
                '     "when can I take ML at software & application development"]\n\n'
                '  FACTUAL, 2 courses × 2 programs:\n'
                '  "when can I take NLP and image processing at artificial intelligence & machine learning and software & application development"\n'
                '  → ["when can I take NLP at artificial intelligence & machine learning",\n'
                '     "when can I take NLP at software & application development",\n'
                '     "when can I take image processing at artificial intelligence & machine learning",\n'
                '     "when can I take image processing at software & application development"]\n\n'
                '  COMPARISON (atomic):\n'
                '  "compare artificial intelligence & machine learning and software & application development"\n'
                '  → ["compare artificial intelligence & machine learning and software & application development"]\n\n'
                '  COMPARISON + separate FACTUAL:\n'
                '  "compare NLP and image processing in artificial intelligence & machine learning and what is the GPA"\n'
                '  → ["compare NLP and image processing in artificial intelligence & machine learning",\n'
                '     "what is the GPA"]\n\n'
                '  COMPARISON + separate COMPARISON (two distinct intents):\n'
                '  "compare course A and course B in program X and program X or program Y"\n'
                '  → ["compare course A and course B in program X",\n'
                '     "program X or program Y"]\n\n'
                '  Already atomic:\n'
                '  "what is ML about?" → ["what is ML about?"]\n\n'
                f'Query: "{clean_query}"\n\n'
                'Return JSON only: {"sub_queries": ["..."]}'
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
    from agent import BNUAdvisorAgent
    from llm_client import llm_call_text
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

    # ── Planning special case ─────────────────────────────────────────────
    # If start_course_planning was called in any sub-query, the planning
    # session is now cached in _planning_sessions[student_id].  We must
    # surface the planning output directly so the student sees the
    # interactive prompt and the session continues normally on the next turn.
    _PLANNING_SIG = "STUDENT COURSE PLANNING SYSTEM"
    planning_ctxs    = [c for c in all_contexts if _PLANNING_SIG in c]
    non_planning_ctxs = [c for c in all_contexts if _PLANNING_SIG not in c]

    if planning_ctxs:
        planning_output = planning_ctxs[0]
        if non_planning_ctxs:
            # Synthesise the non-planning parts first, then append the
            # planning prompt so the student gets both pieces of info.
            if verbose:
                _box("✅  ANSWER NODE  →  generating final response", [], force=True)
            other_answer = llm_call_text(
                system=(
                    "You are the BNU Academic Advisor. "
                    "Write a clear, helpful answer using ONLY the provided database context. "
                    "Use **bold** for key terms, bullet points for lists. "
                    "Do NOT cite source or tool names."
                ),
                user=(
                    f"Student question: {clean_query}\n\n"
                    f"Information from the BNU database:\n"
                    + "\n\n---\n\n".join(non_planning_ctxs)
                    + "\n\nAnswer only the non-planning part of the question."
                ),
            )
            return f"{other_answer}\n\n---\n\n{planning_output}"
        # Only planning context — return it directly
        return planning_output

    if verbose:
        _box("✅  ANSWER NODE  →  generating final response", [], force=True)

    # One final synthesis from all collected context
    combined = "\n\n---\n\n".join(all_contexts)
    return llm_call_text(
        system=(
            "You are the BNU Academic Advisor. "
            "Write a clear, helpful answer to the student's question using ONLY the "
            "provided database context. "
            "Match answer length to the question: short for simple queries, detailed "
            "for complex ones. "
            "Use **bold** for key terms, bullet points for lists. "
            "Do NOT cite article numbers, source names, or tool names. "
            "If the context is insufficient, say so and suggest the student contact "
            "the registrar's office."
        ),
        user=(
            f"Student question: {clean_query}\n\n"
            f"Information from the BNU database:\n{combined}\n\n"
            "Answer the question clearly and concisely based on the information above."
        ),
    )


def _advance_planning(student_id: str, reply: str, state: Any) -> str:
    """
    Drive the next turn of an active planning session.

    Args:
        student_id: The student's ID.
        reply:      The student's latest answer to the planner's prompt.
        state:      PlanningState cached from the previous turn.

    Returns:
        The planner's next message, or a completion notice when done.
    """
    try:
        from planning_service import PlanningOrchestrator, PlanStep

        next_message, new_state = PlanningOrchestrator.advance(state, reply)

        if new_state is None or new_state.current_step == PlanStep.COMPLETE:
            # Planning is done — remove the cached session
            _planning_sessions.pop(student_id, None)
            return (
                next_message
                or "✅ Your course plan is complete! Feel free to ask me anything else."
            )

        # Update cached state for the next turn
        _planning_sessions[student_id] = new_state
        return next_message or "Please continue with the planning session."

    except Exception as exc:
        # If the planner breaks, clean up and report the error gracefully
        _planning_sessions.pop(student_id, None)
        return (
            f"The planning session encountered an issue ({exc}). "
            "Please try starting a new planning session."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Planning tool patch
# ─────────────────────────────────────────────────────────────────────────────
#
# The LangGraph agent calls tools.start_course_planning, which by default
# discards the PlanningState object.  We patch it here at startup to instead
# route through _start_and_cache_planning(), which captures the state and
# stores it in _planning_sessions so subsequent messages can call advance().
#
# This keeps tools.py and agent.py free of any chatbot_api imports while
# still wiring up the multi-turn planning lifecycle correctly.
# ─────────────────────────────────────────────────────────────────────────────

def _start_and_cache_planning() -> str:
    """
    Start a PlanningOrchestrator session, cache the state, return first message.
    This replaces the default start_course_planning tool implementation.
    Student ID is read from tools._ACTIVE_STUDENT_ID (same as all other tools).
    """
    try:
        from planning_service import PlanningOrchestrator, PlanStep
        from chatbot_connector import ChatbotConnector
        import tools as _tools

        student_id               = _tools._get_student_id()
        supabase_client          = ChatbotConnector().client
        first_message, state     = PlanningOrchestrator.start(student_id, supabase_client)

        # Cache state only if planning is not already complete
        if state is not None and state.current_step != PlanStep.COMPLETE:
            _planning_sessions[student_id] = state

        return first_message or "Course planning session started."

    except Exception as exc:
        return f"Could not start course planning: {exc}"


def _patch_planning_tool() -> None:
    """
    Monkey-patch the start_course_planning tool in tools.py to use
    _start_and_cache_planning() so the PlanningState is captured.
    """
    try:
        import tools as _tools
        for t in _tools.ALL_TOOLS:
            if t.name == "start_course_planning":
                # Replace the underlying function while preserving tool metadata
                t.func = _start_and_cache_planning
                break
    except Exception:
        pass  # Non-critical — falls back to the original (stateless) impl.


# Apply the patch immediately when this module is imported
_patch_planning_tool()


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