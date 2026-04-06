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

import os
import traceback
from typing import Any, Dict, Optional

from dotenv import load_dotenv

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
            "track":             student.get("track",            ""),
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

    pre   = get_preprocessor()
    result = pre.process(message, history)

    if result.status == "ambiguous":
        # Store the pending state and ask the student to clarify
        _ambiguity_sessions[student_id] = result.pending
        return result.clarification

    # "ready" or "passthrough" — we have a clean query
    clean_query = result.clean_query or message

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

    clean_query = result.clean_query or pending.original_query

    from agent import BNUAdvisorAgent
    agent = BNUAdvisorAgent(student_id=student_id)
    return agent.run(query=clean_query, history=pending.history)


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

def _start_and_cache_planning(student_id: str) -> str:
    """
    Start a PlanningOrchestrator session, cache the state, return first message.
    This replaces the default start_course_planning tool implementation.
    """
    try:
        from planning_service import PlanningOrchestrator, PlanStep
        from chatbot_connector import ChatbotConnector

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