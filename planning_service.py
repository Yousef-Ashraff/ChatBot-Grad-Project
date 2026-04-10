"""
Planning Service — wraps planning() for chatbot integration.

The planning() function is fully interactive (uses input() / print()).
This module runs it in a background thread and intercepts all I/O so the
chatbot can drive it turn-by-turn without ANY changes to planning().

Public surface (matches what execution_engine.py already imports):
    PlanStep            — Enum with IN_PROGRESS / COMPLETE
    PlanningState       — dataclass holding the running session
    PlanningOrchestrator.start(student_id, supabase_client)
                        → (first_message: str, state: PlanningState | None)
    PlanningOrchestrator.advance(state, user_reply: str)
                        → (next_message: str, state: PlanningState)

How the thread bridge works
───────────────────────────
1.  planning() runs inside a daemon thread.
2.  A custom _ThreadRouter replaces sys.stdout; writes from the planning
    thread are buffered in a per-thread StringIO, all other threads write
    normally to the real stdout.
3.  builtins.input is replaced by _smart_input; when the planning thread
    calls input(prompt) it:
        a) flushes its buffer + the prompt string into _out_q
        b) blocks on _in_q until the chatbot sends the user's reply
4.  The chatbot (execution_engine) calls start() to kick off the thread
    and gets the first message, then calls advance(state, reply) for each
    subsequent turn.
5.  When planning() returns, the thread puts a "done" event in _out_q and
    state.current_step is set to PlanStep.COMPLETE.
"""

from __future__ import annotations

import builtins
import io
import os
import queue
import sys
import threading
import traceback
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

# ── Course name normalisation (resolves user-typed names to Neo4j names) ──────
try:
    from course_name_mapper import map_course_name as _map_course_name
    _MAPPER_OK = True
except ImportError:
    _MAPPER_OK = False
    def _map_course_name(name, threshold=0.3):  # type: ignore
        return None


def _normalise_course(name: str) -> str:
    """Map a user-typed course name to the canonical Neo4j name if possible."""
    if not name or not isinstance(name, str):
        return name
    mapped = _map_course_name(name.strip())
    if mapped:
        print(f"📚 [Planning] Normalised '{name}' → '{mapped}'")
        return mapped.lower()
    return name.lower()


# ── planning() and helper functions now live in planning.py ─────────────────
# Importing here keeps PlanningOrchestrator working without changes.
try:
    from planning import planning
    _PLANNING_OK = True
except ImportError as _pe:
    print(f'⚠️  planning_service: could not import planning.py: {_pe}')
    _PLANNING_OK = False
    def planning(student_id): return None

# ═════════════════════════════════════════════════════════════════════════════
# Thread-aware I/O bridge
# ═════════════════════════════════════════════════════════════════════════════

class _ThreadRouter(io.RawIOBase):
    """
    Replaces sys.stdout.  Writes from a registered thread are captured in a
    per-thread StringIO buffer; all other threads write to the real stdout.
    """

    def __init__(self, real_stdout):
        self._real   = real_stdout
        self._bufs: Dict[int, io.StringIO] = {}
        self._lock   = threading.Lock()

    # ── Registration ──────────────────────────────────────────────────────

    def register(self, tid: int) -> None:
        with self._lock:
            self._bufs[tid] = io.StringIO()

    def unregister(self, tid: int) -> None:
        with self._lock:
            self._bufs.pop(tid, None)

    def flush_buffer(self, tid: int) -> str:
        """Return and clear the buffer for *tid*."""
        with self._lock:
            buf = self._bufs.get(tid)
            if buf is None:
                return ""
            text = buf.getvalue()
            self._bufs[tid] = io.StringIO()
            return text

    # ── io.TextIOBase interface ───────────────────────────────────────────

    def write(self, text: str) -> int:
        tid = threading.get_ident()
        with self._lock:
            buf = self._bufs.get(tid)
        if buf is not None:
            buf.write(text)
            return len(text)
        # Not a registered thread — fall through to real stdout
        return self._real.write(text)

    def flush(self) -> None:
        self._real.flush()

    def fileno(self) -> int:
        try:
            return self._real.fileno()
        except Exception:
            return -1

    def isatty(self) -> bool:
        return False

    # Make print() happy
    @property
    def encoding(self):
        return getattr(self._real, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self._real, "errors", "replace")


# Module-level router (installed once, stays for the process lifetime)
_router: Optional[_ThreadRouter] = None
_router_install_lock = threading.Lock()

# Per-thread input handlers
_input_handlers: Dict[int, Callable] = {}
_ih_lock = threading.Lock()
_original_input = builtins.input   # captured before any patch


def _install_router() -> _ThreadRouter:
    """Install the thread-aware stdout router (idempotent)."""
    global _router
    with _router_install_lock:
        if _router is None:
            _router = _ThreadRouter(sys.__stdout__)
            sys.stdout = _router
    return _router


def _install_input_patch() -> None:
    """Replace builtins.input with our smart dispatcher (idempotent)."""
    if builtins.input is not _smart_input:
        builtins.input = _smart_input


def _smart_input(prompt: str = "") -> str:
    """Dispatch input() calls to per-thread handlers when registered."""
    tid = threading.get_ident()
    with _ih_lock:
        handler = _input_handlers.get(tid)
    if handler is not None:
        return handler(prompt)
    return _original_input(prompt)


def _register_input_handler(tid: int, handler: Callable) -> None:
    with _ih_lock:
        _input_handlers[tid] = handler


def _unregister_input_handler(tid: int) -> None:
    with _ih_lock:
        _input_handlers.pop(tid, None)


# ═════════════════════════════════════════════════════════════════════════════
# Public planning API
# ═════════════════════════════════════════════════════════════════════════════

class PlanStep(Enum):
    IN_PROGRESS = auto()
    COMPLETE    = auto()


@dataclass
class PlanningState:
    current_step: PlanStep                = PlanStep.IN_PROGRESS
    result:       Optional[Any]           = None
    # Planning thread → chatbot:  ("need_input", buffered_text, prompt_str)
    #                          or ("done",        buffered_text, result_dict | None)
    _out_q: queue.Queue = field(default_factory=queue.Queue)
    # Chatbot → planning thread:  user reply string
    _in_q:  queue.Queue = field(default_factory=queue.Queue)
    _thread: Optional[threading.Thread] = field(default=None, repr=False)


def _format_message(buffered: str, event: str, extra: Any) -> str:
    """
    Combine buffered terminal output + prompt/completion note into the
    message that will be displayed to the student.

    The planning function already produces nicely formatted output
    (=== headers, bullet points, numbered lists) so we preserve it as-is
    and just append the prompt on a new line.
    """
    parts: List[str] = []

    text = buffered.strip() if buffered else ""
    if text:
        parts.append(text)

    if event == "need_input" and extra:
        prompt = extra.strip()
        if prompt:
            # Visually separate the prompt from the preceding output
            parts.append(f"\n➡️  **{prompt}**")

    elif event == "done":
        if extra is not None:
            # planning() returned a result dict — session is complete
            parts.append("\n✅ **Your course plan has been finalised!**")
        # if extra is None it means planning() returned None (student not found, etc.)
        # — the error was already printed to the buffer above

    return "\n".join(parts) if parts else "⏳ Planning in progress…"


class PlanningOrchestrator:
    """
    Static orchestrator that starts and advances planning sessions.

    Usage (mirrors what execution_engine.py already does):

        msg, state = PlanningOrchestrator.start(student_id, supabase_client)
        # → show msg to student, wait for reply …

        msg, state = PlanningOrchestrator.advance(state, user_reply)
        # → repeat until state.current_step == PlanStep.COMPLETE
    """

    _TIMEOUT = 90   # seconds to wait for each planning event

    @staticmethod
    def start(
        student_id: str,
        supabase_client: Any,
    ) -> Tuple[str, Optional[PlanningState]]:
        """
        Spin up a planning thread and return the first message to show the student.

        Returns:
            (message, state)   — on success
            (error_msg, None)  — on hard failure (e.g. DB unreachable)
        """
        router = _install_router()
        _install_input_patch()

        state = PlanningState()

        def _run() -> None:
            tid = threading.get_ident()
            router.register(tid)

            def handle_input(prompt: str) -> str:
                # Flush whatever was printed since last input call
                buffered = router.flush_buffer(tid)
                state._out_q.put(("need_input", buffered, prompt))
                # Block until the chatbot sends the student's reply
                return state._in_q.get()

            _register_input_handler(tid, handle_input)

            try:
                result = planning(student_id, supabase_client)
                buffered = router.flush_buffer(tid)
                state.result = result
                state.current_step = PlanStep.COMPLETE
                state._out_q.put(("done", buffered, result))

            except Exception as exc:
                buffered = router.flush_buffer(tid)
                err = f"{buffered}\n❌ Planning error: {exc}\n{traceback.format_exc()}"
                state.current_step = PlanStep.COMPLETE
                state._out_q.put(("done", err, None))

            finally:
                router.unregister(tid)
                _unregister_input_handler(tid)

        state._thread = threading.Thread(target=_run, daemon=True, name="planning-worker")
        state._thread.start()

        # Wait for the first event (first input() call or immediate completion)
        try:
            event, buffered, extra = state._out_q.get(
                timeout=PlanningOrchestrator._TIMEOUT
            )
        except queue.Empty:
            return (
                "⚠️ Planning timed out while fetching your student data. "
                "Please try again or contact the registrar.",
                None,
            )

        if event == "done":
            state.current_step = PlanStep.COMPLETE

        return _format_message(buffered, event, extra), state

    @staticmethod
    def advance(
        state: PlanningState,
        user_reply: str,
    ) -> Tuple[str, PlanningState]:
        """
        Forward the student's reply to the planning thread and return the
        next message (either the next prompt or the completion summary).

        Args:
            state:      The active PlanningState returned by start() or a previous advance()
            user_reply: The student's raw text reply

        Returns:
            (message, updated_state)
        """
        if state.current_step == PlanStep.COMPLETE:
            # Guard: session already finished
            return "✅ Planning session is already complete.", state

        # Send reply to the blocked planning thread
        state._in_q.put(user_reply)

        # Wait for the next event
        try:
            event, buffered, extra = state._out_q.get(
                timeout=PlanningOrchestrator._TIMEOUT
            )
        except queue.Empty:
            state.current_step = PlanStep.COMPLETE
            return (
                "⚠️ Planning timed out waiting for a response. "
                "The session has been ended. Please start a new plan.",
                state,
            )

        if event == "done":
            state.current_step = PlanStep.COMPLETE

        return _format_message(buffered, event, extra), state