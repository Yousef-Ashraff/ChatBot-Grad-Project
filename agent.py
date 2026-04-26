"""
agent.py — BNU Academic Advisor  ·  LangGraph Agent with Judging Loop
======================================================================

Architecture
────────────
Each student query runs through a multi-step judging loop rather than a
simple ReAct loop:

  START
    │
  [agent]  ←─────────────────────────────────────────┐
    │ (has tool_calls)                                 │
  [tools]                                             │
    │                                                 │
  [collect]  ← appends tool results to               │
    │          accumulated_context                    │
  [judge]  ← LLM checks if context fully answers     │
    │         the original query                      │
    ├─ satisfied         → [answer] → END             │
    ├─ tools_this_round < MAX  ────────────────────────┘  (call another tool)
    ├─ reformulations < MAX   → [reformulate] → [agent]  (try new angle)
    └─ reformulations ≥ MAX   → [clarify]    → END        (give up, ask user)

State
─────
  messages              – full message history (add_messages reducer)
  accumulated_context   – list of tool results (operator.add reducer — APPENDS)
  tool_calls_this_round – how many tool rounds done for the current_query
  query_reformulations  – how many times the query was reformulated
  original_query        – never changes after initialization
  current_query         – may be replaced by the reformulate node
  satisfied             – set by judge node

Limits
──────
  MAX_TOOL_CALLS_PER_ROUND = 3   tool rounds per query before reformulating
  MAX_REFORMULATIONS       = 3   reformulations before giving up

student_id injection
────────────────────
Tools that need student_id declare `student_id: Annotated[str, InjectedToolArg]`.
The value is passed via RunnableConfig.configurable:
    config = {"configurable": {"student_id": "22030094"}}
"""

from __future__ import annotations

import json
import logging
import operator
import os
import re
from typing import Annotated, Dict, List, Literal, Optional

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage,
)
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from tools import ALL_TOOLS
from debug_box import box as _box, set_verbose as _set_verbose, is_verbose as _is_verbose
from llm_client import llm_call_json, llm_call_text

load_dotenv()
logger = logging.getLogger(__name__)

GROQ_MODEL_AGENT         = os.getenv("GROQ_MODEL_AGENT", "openai/gpt-oss-120b")
MAX_TOOL_CALLS_PER_ROUND = 3   # max tool call rounds per query/reformulation
MAX_REFORMULATIONS       = 3   # max reformulations before giving up


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """
    Full state flowing through the judging loop graph.

    Fields with operator.add APPEND on every update.
    All other fields use the default replace reducer (last-write wins).
    """
    messages:                  Annotated[list[BaseMessage], add_messages]
    accumulated_context:       Annotated[List[str], operator.add]
    silent_context:            Annotated[List[str], operator.add]
    # "tool_name|canonicalized_args" — used to skip exact duplicate calls
    called_tools:              Annotated[List[str], operator.add]
    # Every distinct reformulation attempted so far (including original)
    previous_reformulations:   Annotated[List[str], operator.add]
    tool_calls_this_round:     int   # incremented by collect; reset by reformulate
    query_reformulations:      int   # total reformulations done
    original_query:            str   # never changes (preprocessed, resolved)
    current_query:             str   # replaced by reformulate node
    satisfied:                 bool  # set by judge node
    judge_missing:             str   # set by judge — what was still missing
    judge_missing_source:      str   # "llm" or "python_override"
    judge_deps_check_info:     str   # result of _multi_course_deps_missing check
    judge_tools_this_round:    int   # set by judge — snapshot of tool_calls_this_round



# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

_AGENT_SYSTEM = (
    "You are a tool-selection assistant for the BNU Academic Advisor. "
    "Your ONLY job is to decide which tool to call next to gather data that "
    "answers the student's query. "
    "Every query you receive has already been fully preprocessed:\n"
    "  - All pronouns and references are resolved to their actual subjects.\n"
    "  - All course and track names are canonical.\n"
    "  - No ambiguity remains — use the query text as-is.\n"
    "  - Course names are marked with single quotes followed by the word 'course': "
    "'<name>' course  (e.g. 'machine learning' course). "
    "Program names are marked with single quotes followed by the word 'program': "
    "'<name>' program  (e.g. 'artificial intelligence & machine learning' program). "
    "The ENTIRE quoted string is the exact entity name — never split or truncate it.\n"
    "  - CRITICAL: Terms that are NOT in single quotes are NOT course or program names "
    "(they are descriptors, search words, or context). "
    "For example, 'special topics' is NOT a course name — only 'special topics in advanced "
    "machine learning' course (fully quoted) would be. "
    "Never pass an unquoted term as a course_name or program_name parameter.\n"
    "\nRULES:\n"
    "1. Call exactly one tool — never write a prose answer.\n"
    "2. Never call the same tool with the same parameters as a previous call.\n"
    "3. Choose the tool whose description BEST matches the query intent.\n"
    "4. If context has already been collected, pick a DIFFERENT tool.\n"
    "5. Use the exact course/program name from the query (the full single-quoted string) "
    "— do NOT substitute, split, or paraphrase.\n"
    "5b. Do NOT add a program_name parameter unless the query explicitly names a program. "
    "If the query asks about a course without specifying a program, call the tool "
    "without program_name. Adding an uninstructed program filter is not 'trying a different "
    "angle' — it artificially restricts results and counts as a wasted round.\n"
    "6. For course planning requests, call start_course_planning — it is a "
    "   complete interactive session, no other tools are needed alongside it.\n"
    "7. COURSE COMPLETION STATEMENT — ONLY applies when the student's message contains "
    "explicit first-person language stating they finished a course: 'I got X', 'I passed X', "
    "'I completed X', 'I finished X', 'I have X done'. A QUESTION about a course ('what "
    "close X?', 'what does X unlock?', 'what courses need X?') is NEVER a course completion "
    "statement, even if it mentions a course name. When the pattern matches: call "
    "get_course_dependencies with prereq=False, dependents=True (show what completing X "
    "enables). Do NOT call store_preference for course completion statements.\n"
    "7b. CHOOSING FLAGS for get_course_dependencies — ask ONE question:\n"
    "  'Is the student asking about what comes BEFORE X, or what comes AFTER X?'\n"
    "  BEFORE X (courses the student must complete to reach/access X) → prereq=True, dependents=False\n"
    "  AFTER X (courses that become available once X is completed)    → prereq=False, dependents=True\n"
    "  Both directions or uncertain                                   → both=True (default — safest)\n"
    "\n"
    "  To determine BEFORE vs AFTER: reconstruct the expected answer.\n"
    "    Expected answer = courses you must finish TO REACH X → BEFORE → prereq.\n"
    "    Expected answer = courses you can take HAVING COMPLETED X → AFTER → dependents.\n"
    "\n"
    "  When direction is genuinely uncertain, always use both=True rather than guessing.\n"
    "  No surface keyword (close, open, unlock, need, require) reliably signals direction.\n"
    "  Always resolve from meaning: what would a correct answer to this query look like?\n"
    "8. PREFERENCE DETECTION — call store_preference whenever the student expresses "
    "interest, love, skill, weakness, background, or emotion about any subject "
    "(e.g. 'I love NLP', 'I'm good at math', 'I hate theory', 'coding is easy for me'). "
    "Exclude course completion statements (rule 7).\n"
    "   8a. PURE PREFERENCE (no factual question): if the message is ONLY a sentiment "
    "statement with no factual question embedded, call store_preference and DO NOT call "
    "any other tool. The judge will mark the query satisfied after store_preference alone. "
    "Do not fetch course info, prerequisites, or any other data.\n"
    "   8b. PREFERENCE + FACTUAL QUESTION: if the message expresses a preference AND asks "
    "a factual question (e.g. 'I love NLP, what are its prerequisites?'), call "
    "store_preference FIRST. After it completes you will be called again — then call the "
    "appropriate factual tool to answer the question.\n"
)

_ANSWER_SYSTEM = (
    "You are the BNU Academic Advisor. "
    "Write a clear, helpful answer to the student's question using ONLY the "
    "provided database context. "
    "Match answer length to the question: short for simple queries, detailed "
    "for complex ones. "
    "Use **bold** for key terms, bullet points for lists. "
    "Do NOT cite article numbers, source names, or tool names. "
    "If the context is insufficient, say so and suggest the student contact "
    "the registrar's office.\n"
    "COURSE COMPLETION ACKNOWLEDGMENT: ONLY trigger this when the student's message "
    "explicitly states in first person that they completed, passed, finished, or earned "
    "a course — the message must contain language like 'I got X', 'I passed X', "
    "'I completed X', 'I finished X', 'I have X'. Do NOT trigger this for questions "
    "that merely ask about prerequisites or what a course unlocks — a question is not "
    "a completion statement, even if the tool returned a `dependents` list. "
    "When triggered: start with a warm, brief congratulation, then follow these steps:\n"
    "  STEP 1 — Build COMPLETED_SET: all course names the student explicitly stated "
    "finishing in this message (e.g. 'I finished X, Y, Z' → COMPLETED_SET = {X, Y, Z}).\n"
    "  STEP 2 — Classify every dependent course from the tool's 'dependents' list:\n"
    "    • ALREADY DONE — the dependent's name is itself in COMPLETED_SET.\n"
    "    • FULLY UNLOCKED — every course in that dependent's 'dep_prereq' list is in "
    "COMPLETED_SET (or dep_prereq is empty), AND the dependent is not already done.\n"
    "    • GETTING CLOSER — at least one course in dep_prereq is NOT in COMPLETED_SET "
    "and the dependent is not already done.\n"
    "  STEP 3 — Render up to THREE sections (omit any section that is empty):\n"
    "    Section 1 '✅ Already completed:' — ALREADY DONE courses, each marked ✅ with "
    "'(you mentioned finishing this)'.\n"
    "    Section 2 '🎯 Fully unlocked — you meet all prerequisites:' — FULLY UNLOCKED "
    "courses with their code and type. End this section with: *Use the eligibility "
    "checker to confirm enrollment — credit-hour requirements may also apply.*\n"
    "    Section 3 '⏳ Getting closer — more prerequisites needed:' — GETTING CLOSER "
    "courses. Under each course list the dep_prereq items NOT in COMPLETED_SET as "
    "'Still needs: <course name>'. Never respond with the course's own description.\n"
    "STUDENT PREFERENCES AND INTERESTS: If the student expressed interest, love, "
    "enthusiasm, or a personal connection to a subject (e.g. 'I love NLP', 'I enjoy math'), "
    "acknowledge it warmly at the start of your response. Briefly mention why that subject "
    "is valuable and how it connects to their studies or career — then continue with the "
    "factual answer. If the message is ONLY a preference statement with no factual question, "
    "respond warmly, affirm their interest, and highlight the benefits and career relevance "
    "of that subject. Do NOT mention databases, preference profiles, or system internals.\n"
    "IMPORTANT: When listing courses, always include 'Elective Slot' entries "
    "exactly like any other course — do NOT skip or omit them. An elective slot "
    "is a scheduled course position (not a specific elective course itself) and "
    "must appear in the list.\n\n"
    "SPECIALIZED COURSES SCOPE RULE:\n"
    "- If the student asks about 'specialized courses' (no qualifier) refer to BOTH "
    "  the specialized core courses AND the specialized elective catalogue.\n"
    "- If the student asks about 'specialized core courses' refer to core only.\n"
    "- If the student asks about 'specialized elective courses' refer to elective catalogue only.\n\n"
    "FULL COURSE LIST FORMATTING RULES (apply when listing all courses for a program):\n"
    "1. SPECIALIZED CORE COURSES — label the section as required for that specific program. "
    "   Elective Slot entries belong under this section and are ALSO required for the program "
    "   (they are reserved positions in the schedule, not optional). Do NOT put them in a "
    "   separate section or imply they are optional. "
    "   Format every Elective Slot as a regular bullet-point list item (same style as real courses). "
    "   Each slot entry has a 'count' field — always show it as '(×N slots)' directly after "
    "   'Elective Slot', e.g.: '- **Elective Slot (×2)** – ELECTIVE – 6 cr – Fourth Year / First Sem'.\n"
    "2. SPECIALIZED ELECTIVE COURSES — label the section as the program's own elective "
    "   catalogue (e.g. 'AIM Elective Catalogue'). These are specific to the program — each "
    "   program has its own distinct set of elective courses.\n"
    "3. GENERAL / HUMANITIES COURSES — label the section header as "
    "   'required for ALL programs (12 cr)'.\n"
    "4. MATHEMATICS & BASIC SCIENCES — label the section header as "
    "   'required for ALL programs (24 cr)'.\n"
    "5. BASIC COMPUTING SCIENCES — label the section header as "
    "   'required for ALL programs (36 cr)'. Then note any program-specific exceptions "
    "   inline: 'Technical Report Writing (BCS112)' exists only in AIM and SAD; "
    "   'Fundamentals of Data Science' exists only in Data Science. "
    "   Do NOT say these BCS courses are 'specific to AIM' in the section header.\n\n"
    "RESPONSE DEPTH RULE:\n"
    "Answer every part of the query fully using the data you already have.\n"
    "- If the query has multiple parts, address each one completely and separately.\n"
    "- Include all relevant fields for every item — never summarize or truncate a list.\n"
    "- Do not refer the student to an advisor or external source as a substitute for "
    "  information you already have access to.\n\n"
    "DEPENDENCY QUERY RULE:\n"
    "Whenever the student's question asks about prerequisites, what a course requires, "
    "what a course unlocks, what a course closes, or any form of dependency information, "
    "ALWAYS output BOTH Section A and Section B — NEVER skip or omit either section. "
    "The content inside each section depends on exactly one of three states:\n"
    "  Section A '📋 Prerequisites:'\n"
    "    STATE 1 — key present in context AND list is non-empty → list every prerequisite course.\n"
    "    STATE 2 — key present in context AND list is explicitly empty (\"prerequisites\": []) → write: "
    "'This course has no prerequisites — anyone can enroll.'\n"
    "    STATE 3 — key NOT present in context at all → write: "
    "'Want to know what you need before taking this course? Just ask and I will look it up for you!'\n"
    "  Section B '🔓 Unlocks (Dependents):'\n"
    "    STATE 1 — key present in context AND list is non-empty → for each dependent course show it "
    "as a sub-entry with ALL of its prerequisites (the full dep_prereq list including the "
    "current course itself). Format: '- **<Name>** — requires: <prereq1>, <prereq2>, ...'. "
    "If a dependent's dep_prereq list is empty write '— requires: (none listed)'.\n"
    "    STATE 2 — key present in context AND list is explicitly empty (\"dependents\": []) → write: "
    "'This course does not directly unlock any other courses.'\n"
    "    STATE 3 — key NOT present in context at all → write: "
    "'Want to know which courses this one unlocks? Just ask and I will look it up for you!'\n"
    "ABSOLUTE RULES:\n"
    "  • NEVER remove or skip Section A or Section B regardless of which state applies.\n"
    "  • STATE 2 (empty list []) and STATE 3 (key absent) are completely different — never confuse them.\n"
    "  • An empty list means the data was fetched and the answer is 'none'. A missing key means the data was never fetched.\n\n"
    "COMPARISON FOLLOW-UP RULE:\n"
    "Whenever the context was gathered using compare_programs or compare_courses, "
    "end your response with a friendly follow-up question asking the student whether "
    "they would like a personalised recommendation based on the comparison — for example: "
    "'Would you like me to make a personalised recommendation for you based on this comparison?'\n"
)


# ─────────────────────────────────────────────────────────────────────────────
# LLM factory
# ─────────────────────────────────────────────────────────────────────────────

_LLM_INSTANCE: Optional[ChatGroq] = None


def _get_llm() -> ChatGroq:
    """
    Return a ChatGroq instance with automatic fallback across up to 5 API keys.
    Cached as a module-level singleton (no student-specific state inside it).
    """
    global _LLM_INSTANCE
    if _LLM_INSTANCE is not None:
        return _LLM_INSTANCE

    key_env_vars = [
        "GROQ_API_KEY", "GROQ_API_KEY2", "GROQ_API_KEY3",
        "GROQ_API_KEY4", "GROQ_API_KEY5",
    ]
    active_keys = [
        os.getenv(v, "").strip()
        for v in key_env_vars
        if os.getenv(v, "").strip()
    ]
    if not active_keys:
        raise ValueError(
            "No Groq API keys found. "
            "Set GROQ_API_KEY (and optionally GROQ_API_KEY2…5) in .env"
        )

    instances = [
        ChatGroq(api_key=k, model=GROQ_MODEL_AGENT, temperature=0.1, max_tokens=1500)
        for k in active_keys
    ]
    primary = instances[0]
    _LLM_INSTANCE = (
        primary.with_fallbacks(instances[1:])
        if len(instances) > 1
        else primary
    )
    return _LLM_INSTANCE


# ─────────────────────────────────────────────────────────────────────────────
# Graph nodes
# ─────────────────────────────────────────────────────────────────────────────

def _make_agent_node(llm_with_tools):
    """
    Agent node — selects and calls the next tool.

    Builds a focused prompt that shows:
      • The current query (possibly reformulated)
      • The original query (for context)
      • A compact summary of already-collected context
      • Recent tool call history (so the agent avoids repeating itself)

    Then invokes the LLM-with-tools to produce the next tool call.
    Falls back once on tool_use_failed (Groq 400) with a correction prompt.
    """
    _RETRY_MSG = (
        "Your previous response was invalid because you wrote prose instead "
        "of a proper tool call. "
        "You MUST call exactly one tool using the tool-call format — no prose. "
        "Please try again now."
    )

    def agent_node(state: AgentState) -> dict:
        original_query = state.get("original_query", "")
        current_query  = state.get("current_query",  "")
        accum_ctx      = state.get("accumulated_context", [])
        messages       = state.get("messages", [])

        # ── Build query block ─────────────────────────────────────────────
        reformulation_note = (
            f"\n(Reformulated as: {current_query})"
            if current_query != original_query
            else ""
        )

        # ── Build compact context summary ─────────────────────────────────
        context_block = ""
        if accum_ctx:
            summaries = []
            for c in accum_ctx:
                summaries.append(c[:250] + "…" if len(c) > 250 else c)
            context_block = (
                "\n\nContext already collected:\n"
                + "\n---\n".join(summaries)
            )

        # ── Build already-called-tools block ──────────────────────────────
        called_sigs = state.get("called_tools", [])
        called_block = ""
        if called_sigs:
            import json as _j
            lines = []
            for sig in called_sigs:
                parts = sig.split("|", 1)
                if len(parts) == 2:
                    try:
                        args = _j.loads(parts[1])
                        lines.append(f"  - {parts[0]}({', '.join(f'{k}={v}' for k,v in args.items())})")
                    except Exception:
                        lines.append(f"  - {sig}")
                else:
                    lines.append(f"  - {sig}")
            called_block = (
                "\n\nALREADY CALLED (do NOT repeat these exact calls):\n"
                + "\n".join(lines)
                + "\nYou MUST choose a different tool OR different parameters."
            )

        # ── Build judge hint block (only when judge already ran and flagged missing info)
        judge_missing = state.get("judge_missing", "").strip()
        judge_hint = (
            f"\n\nJUDGE FEEDBACK: The previous tool round was not sufficient. "
            f"Still missing: '{judge_missing}'. "
            f"Your next tool call MUST target this missing information."
            if judge_missing else ""
        )

        user_msg = (
            f"Student query: {original_query}"
            f"{reformulation_note}"
            f"{context_block}"
            f"{called_block}"
            f"{judge_hint}\n\n"
            "Call the next most useful tool to answer this query. "
            "Do NOT repeat any already-called tool with the same parameters."
        )

        # ── Include recent tool-call history so agent can see what was tried
        # Find the last agent→tools cycle (AIMessage + its ToolMessages)
        recent_history: List[BaseMessage] = []
        for m in reversed(messages):
            if isinstance(m, (AIMessage, ToolMessage)):
                recent_history.insert(0, m)
                if isinstance(m, AIMessage) and m.tool_calls:
                    break
            else:
                break  # Stop at non-AI/tool message

        prompt_messages: List[BaseMessage] = [SystemMessage(content=_AGENT_SYSTEM)]
        if recent_history:
            prompt_messages += recent_history
        prompt_messages.append(HumanMessage(content=user_msg))

        # ── Invoke LLM with retry on tool_use_failed ─────────────────────
        try:
            response = llm_with_tools.invoke(prompt_messages)
            return {"messages": [response]}
        except Exception as exc:
            err_str = str(exc)
            if "tool_use_failed" in err_str or "400" in err_str:
                logger.warning("[Agent] tool_use_failed — retrying with correction")
                retry_msgs = prompt_messages + [HumanMessage(content=_RETRY_MSG)]
                try:
                    response = llm_with_tools.invoke(retry_msgs)
                    return {"messages": [response]}
                except Exception as retry_exc:
                    logger.error("[Agent] Retry also failed: %s", retry_exc)
                    raise retry_exc
            raise

    return agent_node


def _collect_node(state: AgentState) -> dict:
    """
    Collect node — runs immediately after the ToolNode.

    Scans the message list for the ToolMessages produced by the most recent
    tools node invocation (all ToolMessages after the last AIMessage), formats
    them, and appends to accumulated_context.

    Also increments tool_calls_this_round by 1 (one increment per
    agent→tools→collect cycle, not per individual tool result).
    """
    messages              = state["messages"]
    tool_calls_this_round = state.get("tool_calls_this_round", 0)

    # Walk backwards to find ToolMessages and the AIMessage of this round
    new_context: List[str] = []
    new_silent:  List[str] = []
    new_tool_sigs: List[str] = []
    triggering_ai: Optional[AIMessage] = None

    import json as _j

    _SILENT_TOOLS = {"store_preference"}

    # Find the triggering AIMessage and build tool_call_id → args map
    tc_args_map: dict = {}
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            triggering_ai = m
            if m.tool_calls:
                for tc in m.tool_calls:
                    tc_id  = tc.get("id", "")
                    args   = tc.get("args", {})
                    tc_args_map[tc_id] = args
                    name     = tc.get("name", "?")
                    args_str = _j.dumps(args, sort_keys=True, ensure_ascii=False)
                    new_tool_sigs.append(f"{name}|{args_str}")
            break

    # Collect ToolMessages from the current round (after the triggering AIMessage)
    for m in reversed(messages):
        if isinstance(m, ToolMessage):
            tool_name    = getattr(m, "name", "tool")
            tool_content = getattr(m, "content", "") or ""
            tool_call_id = getattr(m, "tool_call_id", "")
            args         = tc_args_map.get(tool_call_id, {})
            if args:
                args_label = ", ".join(f"{k}={v}" for k, v in args.items() if v is not None)
                header = f"[{tool_name}({args_label})]"
            else:
                header = f"[{tool_name}]"
            if tool_name in _SILENT_TOOLS:
                new_silent.insert(0, f"{header}:\n{tool_content}")
            else:
                new_context.insert(0, f"{header}:\n{tool_content}")
        elif isinstance(m, AIMessage):
            break

    return {
        "accumulated_context":   new_context,           # operator.add appends
        "silent_context":        new_silent,            # operator.add appends
        "called_tools":          new_tool_sigs,         # operator.add appends
        "tool_calls_this_round": tool_calls_this_round + 1,
    }


def _multi_course_deps_missing(original_query: str, accumulated_context: list) -> str:
    """
    When a query involves get_course_dependencies for multiple courses (completion,
    prereq, unlock, or mixed), check that every quoted course in the query has its
    own get_course_dependencies header in context (any direction).

    Gate: only activates when the context already contains at least one
    get_course_dependencies header — meaning the agent already identified this as a
    deps-type query. Direction-agnostic: works for prereq-only, dependents-only, both.

    Program names are detected by the pattern 'TERM' followed by the word 'program'
    in the query text and are excluded from the coverage check.

    Returns the first uncovered course name, or "" if all covered or not applicable.
    """
    # Gate: only fire if the agent already made at least one get_course_dependencies call
    if not any('[get_course_dependencies(' in ctx for ctx in accumulated_context):
        return ""

    quoted = re.findall(r"'([^']+)'", original_query)
    if not quoted:
        return ""

    # Build set of course_names that had get_course_dependencies called (any direction)
    queried_deps: set = set()
    for ctx in accumulated_context:
        m = re.search(r'\[get_course_dependencies\(([^)]+)\)', ctx)
        if m:
            cn = re.search(r'course_name=([^,)]+)', m.group(1))
            if cn:
                queried_deps.add(cn.group(1).strip().lower())

    for term in quoted:
        is_program = bool(re.search(
            r"'" + re.escape(term) + r"'\s*program",
            original_query, re.IGNORECASE
        ))
        if is_program:
            continue
        if term.lower() not in queried_deps:
            return term
    return ""



def _make_judge_node():
    """
    Judge node — evaluates whether accumulated_context fully answers the
    original query.

    Uses llm_call_json (direct Groq call, no LangChain overhead) to ask:
    "Does this context satisfy the query? {satisfied: true/false}"

    Returns {"satisfied": bool} — does NOT add to messages.
    """
    def judge_node(state: AgentState) -> dict:
        original_query = state.get("original_query", "")
        accum_ctx      = state.get("accumulated_context", [])
        silent_ctx     = state.get("silent_context", [])
        tools_used     = state.get("tool_calls_this_round", 0)
        called_tools   = state.get("called_tools", [])

        # ── Special case: planning tool is interactive ────────────────────────
        # Check THREE independent sources for planning output — whichever is
        # available guarantees we detect it regardless of state timing issues.

        # Source 1: called_tools signatures (populated by dedup + collect nodes)
        planning_sig = any("start_course_planning" in sig for sig in called_tools)

        # Source 2: accumulated_context list
        planning_ctx = " ".join(accum_ctx)

        # Source 3: ToolMessages in state["messages"] — scan ALL, not just reversed to AIMessage
        # The previous version broke early at AIMessage, missing the tool msg sometimes
        tool_msg_content = ""
        all_tool_contents = []
        for m in state.get("messages", []):
            if isinstance(m, ToolMessage):
                c = getattr(m, "content", "") or ""
                all_tool_contents.append(c)
                if getattr(m, "name", "") == "start_course_planning":
                    tool_msg_content = c

        # Combine ALL sources
        full_planning_ctx = " ".join([planning_ctx, tool_msg_content] + all_tool_contents)

        planning_was_called = (
            planning_sig
            or "STUDENT COURSE PLANNING SYSTEM" in full_planning_ctx
            or "Planning for: Year" in full_planning_ctx
            or "STUDENT COURSE PLANNING" in full_planning_ctx
        )

        logger.warning(
            "[Judge] planning_sig=%s tool_msg_len=%d ctx_len=%d was_called=%s",
            planning_sig, len(tool_msg_content), len(planning_ctx), planning_was_called
        )

        if planning_was_called:
            planning_failed = (
                "Planning error:" in full_planning_ctx
                or "❌ Error: Student ID" in full_planning_ctx
                or "Error invoking tool" in full_planning_ctx
                or "Could not start course planning" in full_planning_ctx
            )
            logger.warning("[Judge] planning_failed=%s", planning_failed)
            return {
                "satisfied":             True,
                "judge_missing":         "",
                "judge_missing_source":  "llm",
                "judge_deps_check_info": "skipped — planning tool detected",
                "judge_tools_this_round": tools_used,
            }

        main_ctx = "\n\n---\n\n".join(accum_ctx) if accum_ctx else "(no data collected yet)"
        if silent_ctx:
            side_block = "\n".join(silent_ctx)
            context = f"{main_ctx}\n\n--- [Side-effects] ---\n{side_block}"
        else:
            context = main_ctx

        if _is_verbose():
            _box(
                "🔍  JUDGE INPUT  —  context sent to judge LLM",
                context.splitlines(),
            )

        # Structured chain-of-thought judge: the model must enumerate and type
        # every required entity before it can set satisfied=true/false.
        # This prevents the LLM from confusing a COURSE named "X" with a
        # PROGRAM named "X" (e.g. "artificial intelligence" course ≠
        # "artificial intelligence & machine learning" program).
        prompt = (
            f'Student query: "{original_query}"\n\n'
            f'Data collected from the BNU database:\n{context}\n\n'
            f'Determine whether the collected data is sufficient to fully answer '
            f'the student query.\n'
            f'\n'
            f'--- TOOL RESULT FIELD SEMANTICS ---\n'
            f'Use these mappings when deciding if context satisfies the query intent:\n'
            f'\n'
            f'GENERAL RULE — TOOL CALL HEADER = ENTITY PRESENT:\n'
            f'  Every context entry starts with a header like `[tool_name(course_name=X)]`.\n'
            f'  That header proves the tool ran FOR entity X — entity X IS present in context.\n'
            f'  Do NOT require X to appear as a literal word inside the result body.\n'
            f'  The courses/programs listed INSIDE the result body are OTHER entities\n'
            f'  (e.g., dependents of X, prerequisites of X, courses compared to X) — NOT X itself.\n'
            f'  NEVER mark X as absent or missing because X does not appear in the result body.\n'
            f'\n'
            f'• get_course_dependencies has two fields:\n'
            f'  `prerequisites` = what must be done BEFORE X (X\'s prerequisites).\n'
            f'    Satisfies any query about what the student must complete to reach/access X.\n'
            f'  `dependents` = what becomes available AFTER X is completed (courses X unlocks).\n'
            f'    Satisfies any query about what completing X makes accessible.\n'
            f'  To judge which field is expected: reconstruct what the correct answer contains.\n'
            f'    "Courses you must finish to reach X" → BEFORE → expect `prerequisites`.\n'
            f'    "Courses you can take once you have X" → AFTER → expect `dependents`.\n'
            f'  EMPTY RESULT MEANING — read this carefully:\n'
            f'    `"dependents": []`  → course X unlocks NOTHING further. This IS a complete,\n'
            f'      correct answer to "what does completing X enable?". Mark present_in_context=true.\n'
            f'      Do NOT ask for another tool call. Do NOT mark the entity as missing.\n'
            f'    `"prerequisites": []` → course X has NO prerequisites. This IS a complete,\n'
            f'      correct answer to "what must I finish before X?". Mark present_in_context=true.\n'
            f'      Do NOT ask for another tool call. Do NOT mark the entity as missing.\n'
            f'  ONLY `prerequisites` is sufficient for a BEFORE-X query; ONLY `dependents` is\n'
            f'  sufficient for an AFTER-X query — do not mark as missing because the other field\n'
            f'  is absent.\n'
            f'  SATISFYING A DEPENDENCY QUERY — MECHANICAL CHECK (follow exactly):\n'
            f'  A. Find a call header `get_course_dependencies(course_name=X, ...)` in context.\n'
            f'  B. Determine direction from MEANING, not surface keywords:\n'
            f'       AFTER-X: X is the STARTING POINT (already completed/possessed).\n'
            f'         Student asks what X opens up, gives access to, or closes for them.\n'
            f'         (e.g. "what does X close", "what does passing X unlock", "I finished X, now what?")\n'
            f'         → need dependents=True in header OR "dependents" key in result body.\n'
            f'       BEFORE-X: X is the DESTINATION/GOAL (not yet achieved).\n'
            f'         Student asks what is required to reach X, what closes/unlocks X itself.\n'
            f'         (e.g. "what closes X", "what courses unlock X", "what must I finish before X")\n'
            f'         → need prereq=True in header OR "prerequisites" key in result body.\n'
            f'  C. If A and B hold → present_in_context=true for X. STOP. Do not inspect\n'
            f'     nested sub-fields (dep_prereq, etc.) to re-evaluate presence.\n'
            f'  dep_prereq inside a dependents entry lists prerequisites of THAT dependent\n'
            f'  course — NOT prerequisites of X. X appearing there is expected and changes\n'
            f'  nothing about whether X\'s own dependency query is satisfied.\n'
            f'  Surface keywords never reliably indicate BEFORE vs AFTER — reason from meaning.\n'
            f'• Course description / info / credits:\n'
            f'  → satisfied by a get_course_info result for course X.\n'
            f'• Timing / semester / year offered:\n'
            f'  → satisfied by a get_course_timing result for course X.\n'
            f'• Eligibility ("can I take X"):\n'
            f'  → satisfied by a check_course_eligibility result for course X,\n'
            f'    regardless of whether the verdict is eligible or not eligible.\n'
            f'\n'
            f'--- STEP 1: DERIVE ENTITIES FROM THE QUERY — ignore the collected data entirely ---\n'
            f'Read the student query above and answer: "What is this student explicitly asking\n'
            f'to learn about?" List ONLY those things. Do NOT look at the tool results or\n'
            f'tool call parameters yet — they cannot add new entities.\n'
            f'\n'
            f'Entity types:\n'
            f'  type="course"  → a specific course the query is asking about\n'
            f'  type="program" → an academic track/program the query is asking about\n'
            f'  type="other"   → anything else (GPA policy, graduation rules, bylaws, etc.)\n'
            f'\n'
            f'DATA TYPES ARE NOT ENTITIES: Terms like `prerequisites`, `dependents`,\n'
            f'`eligibility`, `credits`, `timing`, `closes`, `unlocks` describe WHAT information\n'
            f'to retrieve about an entity — they are query intents, not entities themselves.\n'
            f'List only COURSES, PROGRAMS, or TOPIC AREAS (e.g. bylaws, GPA policy) as entities.\n'
            f'\n'
            f'SUBJECT vs FILTER:\n'
            f'  Ask: "Is this name the SUBJECT of the question, or just a scope/filter?"\n'
            f'  Subject  → student wants data ABOUT it          → list as entity\n'
            f'  Filter   → narrows WHERE to look, not WHAT to learn → do NOT list as entity\n'
            f'\n'
            f'  Filters are never entities:\n'
            f'  • A program name that scopes a course query ("in the AIM program", "for AIM")\n'
            f'    is a filter — NOT an entity. The courses are the subject.\n'
            f'  • "What is the AIM program?" / "Describe AIM"  → AIM IS the subject → entity.\n'
            f'  • "Compare AIM and SAD"                        → AIM, SAD are subjects → entities.\n'
            f'  • "Compare course A and B in AIM"              → A, B are subjects; AIM is a filter.\n'
            f'\n'
            f'CRITICAL: course data ≠ program data.\n'
            f'  "artificial intelligence" (course) ≠ "artificial intelligence & machine learning" (program).\n'
            f'  A course result that mentions which program it belongs to is NOT program data.\n'
            f'  Program data requires: total credits, credit distribution, multi-year curriculum.\n'
            f'\n'
            f'Your entity list is now FIXED. Do NOT add anything from the tool results or\n'
            f'tool call parameters in the steps below.\n'
            f'\n'
            f'--- STEP 2: CHECK PRESENCE — for each entity from Step 1 only ---\n'
            f'Mark present_in_context=true if the collected data semantically answers the query\n'
            f'intent for that entity. Use the TOOL RESULT FIELD SEMANTICS above.\n'
            f'Do NOT require literal word matches between the query and the field names.\n'
            f'\n'
            f'--- SATISFACTION RULES ---\n'
            f'1. satisfied=true  ONLY when every entity has present_in_context=true.\n'
            f'2. satisfied=false if ANY entity is absent — name the missing one.\n'
            f'3. For comparison queries, ALL compared items must be present.\n'
            f'4. Tool errors or missing student_id alone do NOT make satisfied=false '
            f'   if the other results already answer the query.\n'
            f'5. General answers (not student-specific) are fine — if data exists to '
            f'   give a helpful answer, that is enough.\n'
            f'6. EMPTY DEPENDENCY ARRAYS — MANDATORY RULE:\n'
            f'   `"dependents": []` means the course unlocks NOTHING. This IS the answer.\n'
            f'   `"prerequisites": []` means the course has NO prerequisites. This IS the answer.\n'
            f'   An empty array [] for any dependency field ALWAYS means present_in_context=true.\n'
            f'   NEVER mark an entity as missing or absent because its dependency result is empty.\n'
            f'   NEVER require a second tool call to "confirm" an empty result.\n'
            f'7. PURE PREFERENCE STATEMENT RULE — HIGHEST PRIORITY:\n'
            f'   If the query is solely expressing a personal feeling, interest, or attitude\n'
            f'   (e.g. "I love X", "I enjoy Y", "I hate Z", "X is easy for me", "I am good at Y")\n'
            f'   with NO embedded factual question (no "what is", "can I take", "when is",\n'
            f'   "prerequisites", "eligibility", etc.), then:\n'
            f'   → List ZERO entities (nothing factual to look up).\n'
            f'   → satisfied=true as soon as store_preference appears in the side-effects context.\n'
            f'   → Do NOT require any factual course/program data. The preference was stored — done.\n'
            f'   DISTINCTION: "I love X" = pure preference (satisfied after store_preference).\n'
            f'                "I love X, what are its prerequisites?" = preference + factual question\n'
            f'                (satisfied only after store_preference AND the factual answer).\n'
            f'                "I passed/completed/got X" = course completion, NOT a preference\n'
            f'                (needs get_course_dependencies result — see TOOL RESULT SEMANTICS).\n'
            f'\n'
            f'Respond with ONLY this exact JSON structure — no other text:\n'
            f'{{\n'
            f'  "entities": [\n'
            f'    {{"name": "entity name", "type": "course|program|other", '
            f'"present_in_context": true}},\n'
            f'    ...\n'
            f'  ],\n'
            f'  "satisfied": true,\n'
            f'  "missing": ""\n'
            f'}}'
        )

        satisfied = False
        missing   = ""
        raw       = ""
        try:
            raw = llm_call_json(prompt, temperature=0, max_tokens=800)
            # Strip markdown fences if present
            raw = re.sub(r"```json|```", "", raw).strip()
            # Extract the outermost {...} blob (handles extra surrounding text)
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                raw = m.group(0)
            data = json.loads(raw)

            # Primary read: top-level satisfied field
            satisfied = bool(data.get("satisfied", False))
            missing   = data.get("missing", "")

            # Safety cross-check: if any entity is marked present_in_context=false,
            # override a (hallucinated) satisfied=true.
            entities = data.get("entities", [])
            if entities:
                absent = [
                    e.get("name", "unknown")
                    for e in entities
                    if not e.get("present_in_context", True)
                ]
                if absent and satisfied:
                    satisfied = False
                    missing   = missing or f"missing data for: {', '.join(absent)}"
                    logger.warning(
                        "[Judge] overriding satisfied=true — absent entities: %s", absent
                    )
            else:
                # Model returned no entity list — fall back to strict False
                # unless the model explicitly said satisfied=true with a reason
                # (trust the model only when it provides entities)
                if satisfied and not missing:
                    # no entities parsed but model said true — keep it only if
                    # something was actually collected (factual or silent)
                    if not accum_ctx and not silent_ctx:
                        satisfied = False
                        missing   = "no data collected yet"

        except Exception as exc:
            logger.warning("[Judge] parse failed — raw=%r  err=%s", raw[:200], exc)
            satisfied = False
            missing   = "parse error — defaulting to not satisfied"

        # Python post-check: if LLM says satisfied but a course mentioned in a
        # deps-type query has no own get_course_dependencies call, override to False.
        missing_source    = "llm"
        deps_check_info   = "skipped — LLM already not satisfied"
        if satisfied:
            missing_course = _multi_course_deps_missing(original_query, accum_ctx)
            if missing_course:
                satisfied      = False
                missing        = missing_course
                missing_source = "python_override"
                deps_check_info = f"ran → override triggered, missing: {missing_course}"
                logger.warning(
                    "[Judge] Python override: '%s' has no get_course_dependencies call in context",
                    missing_course,
                )
            else:
                deps_check_info = "ran → all courses covered ✓"

        # Store round count in a valid state field for display
        return {
            "satisfied":             satisfied,
            "judge_missing":         missing,
            "judge_missing_source":  missing_source,
            "judge_deps_check_info": deps_check_info,
            "judge_tools_this_round": tools_used,
        }

    return judge_node


def _make_answer_node(llm):
    """
    Answer node — generates the final student-facing answer.

    Synthesises the accumulated_context into a natural language answer
    using the plain LLM (no tools).  This is always the last step before END.
    """
    def answer_node(state: AgentState) -> dict:
        original_query = state.get("original_query", "")
        accum_ctx      = state.get("accumulated_context", [])
        called_tools   = state.get("called_tools", [])

        # ── Planning tool: relay its output directly — no LLM synthesis needed
        # Check all three sources just like the judge node does
        tool_msg_planning = ""
        for m in reversed(state.get("messages", [])):
            if isinstance(m, ToolMessage):
                if getattr(m, "name", "") == "start_course_planning":
                    tool_msg_planning = getattr(m, "content", "") or ""
                    break
            elif isinstance(m, AIMessage):
                break

        ctx_str = " ".join(accum_ctx) + " " + tool_msg_planning
        planning_was_called = (
            any("start_course_planning" in sig for sig in called_tools)
            or "STUDENT COURSE PLANNING SYSTEM" in ctx_str
            or "Planning for: Year" in ctx_str
        )
        if planning_was_called:
            # Return the planning output — prefer ToolMessage content if richer
            planning_output = tool_msg_planning or "\n".join(accum_ctx)
            from langchain_core.messages import AIMessage as _AI
            return {"messages": [_AI(content=planning_output)]}

        context = (
            "\n\n---\n\n".join(accum_ctx)
            if accum_ctx
            else "(no database context was collected)"
        )

        user_prompt = (
            f"Student question: {original_query}\n\n"
            f"Information from the BNU database:\n{context}\n\n"
            "Answer the question clearly and concisely based on the information above."
        )

        response = llm.invoke([
            SystemMessage(content=_ANSWER_SYSTEM),
            HumanMessage(content=user_prompt),
        ])
        return {"messages": [response]}

    return answer_node


def _make_reformulate_node():
    """
    Reformulate node — generates a new current_query that explores a
    hidden or implicit aspect of the original query.

    Uses llm_call_text (direct Groq call).  Returns:
      current_query:         the new reformulated query string
      tool_calls_this_round: reset to 0 (fresh round for the new query)
      query_reformulations:  incremented by 1
    """
    # Ordered list of angles to explore — each reformulation picks the next
    # one not yet tried.  Having a fixed list prevents the LLM from looping.
    _ANGLES = [
        "what year and semester is this offered in for each program",
        "what are the prerequisites required before taking this",
        "what courses does completing this course unlock or enable",
        "which programs or tracks include this as a required course",
        "am I eligible to take this course based on my academic record",
        "how many credit hours is this course worth",
        "what is the content and description of this course",
    ]

    def reformulate_node(state: AgentState) -> dict:
        original_query        = state.get("original_query", "")
        accum_ctx             = state.get("accumulated_context", [])
        reformulations        = state.get("query_reformulations", 0)
        previous_reforms      = state.get("previous_reformulations", [])
        judge_missing         = state.get("judge_missing", "").strip()

        # All queries tried so far (original + every previous reformulation)
        all_tried_set = {q.lower().strip() for q in [original_query] + list(previous_reforms)}
        all_tried     = [original_query] + list(previous_reforms)
        tried_block   = "\n".join(f"  {i+1}. {q}" for i, q in enumerate(all_tried))

        # Compact context summary (first 3 items, truncated)
        ctx_summary = "\n".join(
            f"  - {c[:120]}…" if len(c) > 120 else f"  - {c}"
            for c in accum_ctx[:3]
        ) or "  (none)"

        # If the judge told us what is missing, use that directly as the angle.
        # Otherwise fall back to the fixed angle list.
        if judge_missing:
            next_angle = judge_missing
        else:
            # Pick the next unused angle from the fixed list
            # This guarantees variety even if the LLM ignores the instruction
            next_angle = None
            for angle in reformulate_node._ANGLES:
                if not any(angle.split()[0] in tried.lower() for tried in all_tried):
                    next_angle = angle
                    break
            if next_angle is None:
                next_angle = "provide any information available about this topic"

        # Label the angle source for clarity in the prompt
        angle_label = (
            "What the judge says is still missing (focus on THIS):"
            if judge_missing
            else "Suggested new angle to explore:"
        )

        prompt = (
            "The student's original query (already reference-resolved): \n"
            f"  {original_query}\n\n"
            "Queries already tried - you MUST NOT produce any of these:\n"
            f"{tried_block}\n\n"
            "Context collected so far (to understand what is already known):\n"
            f"{ctx_summary}\n\n"
            f"{angle_label} {next_angle}\n\n"
            "Rewrite the original query to directly target the missing information above.\n"
            "Return ONLY the rewritten query as one plain sentence."
        )

        new_query = None
        system = (
            "You rewrite BNU academic advisor queries to explore a specific angle.\n"
            "Rules:\n"
            "1. Return ONLY one plain sentence — the rewritten query.\n"
            "2. The output MUST be different from every query listed as already tried.\n"
            "3. Never use pronouns or references (it, them, this, that, the course). "
            "Always spell out the full explicit name of every course or program."
        )

        # Try up to 3 times to get a query that is not a duplicate
        for attempt in range(3):
            try:
                raw = llm_call_text(
                    system=system,
                    user=prompt,
                    temperature=0.4 + attempt * 0.3,   # 0.4, 0.7, 1.0
                    max_tokens=200,
                ).strip().strip('"').strip("'").strip()
                if len(raw) >= 5 and raw.lower().strip() not in all_tried_set:
                    new_query = raw
                    break
                logger.debug("[Reformulate] attempt %d produced duplicate: %r", attempt+1, raw)
            except Exception as exc:
                logger.debug("[Reformulate] attempt %d error: %s", attempt+1, exc)

        # Hard fallback: force a unique query by appending the angle
        if new_query is None:
            new_query = f"{original_query} — focusing on: {next_angle}"

        return {
            "current_query":           new_query,
            "tool_calls_this_round":   0,
            "query_reformulations":    reformulations + 1,
            "previous_reformulations": [new_query],     # operator.add appends
        }

    # Attach _ANGLES as a function attribute so it is accessible inside
    reformulate_node._ANGLES = _ANGLES

    return reformulate_node


def _make_clarify_node(llm):
    """
    Clarify node — generates a polite clarification request to the student
    after all reformulation attempts have been exhausted.
    """
    def clarify_node(state: AgentState) -> dict:
        original_query = state.get("original_query", "")

        user_prompt = (
            f'A BNU student asked: "{original_query}"\n\n'
            "We tried multiple approaches but could not find a complete answer "
            "in the database. "
            "Generate a polite, specific clarification request. "
            "Suggest what additional detail would help: exact course name/code, "
            "program/track, or a clearer phrasing of what they need to know. "
            "Be brief and friendly."
        )

        response = llm.invoke([
            SystemMessage(content=_ANSWER_SYSTEM),
            HumanMessage(content=user_prompt),
        ])
        return {"messages": [response]}

    return clarify_node


# ─────────────────────────────────────────────────────────────────────────────
# Routing functions
# ─────────────────────────────────────────────────────────────────────────────

def _route_after_agent(
    state: AgentState,
) -> Literal["tools", "answer"]:
    """
    After agent node:
    - LLM made tool calls  → tools
    - LLM gave direct text → answer  (shouldn't normally happen given the prompt)
    """
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "answer"


def _route_after_judge(
    state: AgentState,
) -> Literal["answer", "agent", "reformulate", "clarify"]:
    """
    After judge node — implements the judging loop decision:

    ┌─────────────────────────────────────────────────────────────────┐
    │  satisfied?                    → answer                         │
    │  not satisfied:                                                 │
    │    tool_calls < MAX            → agent  (try another tool)      │
    │    tool_calls ≥ MAX            →                                │
    │      reformulations < MAX      → reformulate (new query angle)  │
    │      reformulations ≥ MAX      → clarify  (give up)            │
    └─────────────────────────────────────────────────────────────────┘
    """
    if state.get("satisfied", False):
        return "answer"

    tools_used     = state.get("tool_calls_this_round", 0)
    reformulations = state.get("query_reformulations",  0)

    if tools_used < MAX_TOOL_CALLS_PER_ROUND:
        return "agent"
    elif reformulations < MAX_REFORMULATIONS:
        return "reformulate"
    else:
        return "clarify"


# ─────────────────────────────────────────────────────────────────────────────
# Debug printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_step(
    node_name:      str,
    node_output:    dict,
    original_query: str = "",
    verbose:        bool = True,
    state:          dict = None,
) -> None:
    """
    Print a debug box for one graph node execution.

    node_output is the raw state delta returned by the node (not the full
    accumulated state), so accumulated_context contains only the NEW items
    added in this step, and integers reflect the updated values.

    state (optional) is the full accumulated state — used by nodes that need
    fields set by a prior node (e.g. reformulate reading judge_missing).
    """
    if not verbose:
        return

    import json as _json

    messages = node_output.get("messages", [])

    # ── AGENT ─────────────────────────────────────────────────────────────
    if node_name == "agent":
        last = messages[-1] if messages else None
        if not isinstance(last, AIMessage):
            return
        if last.tool_calls:
            body = []
            for i, tc in enumerate(last.tool_calls, 1):
                body.append(f"[{i}] Tool  :  {tc.get('name', 'unknown')}")
                for k, v in tc.get("args", {}).items():
                    v_str = (
                        _json.dumps(v, ensure_ascii=False)
                        if not isinstance(v, str)
                        else v
                    )
                    body.append(f"    {k}  =  {v_str}")
                if i < len(last.tool_calls):
                    body.append("")
            _box("🤖  AGENT  →  calling tool(s)", body)
        else:
            _box("🤖  AGENT  →  direct answer (no tool)", [])

    # ── TOOLS ─────────────────────────────────────────────────────────────
    elif node_name == "tools":
        for m in messages:
            tool_name = getattr(m, "name", "unknown_tool")
            raw       = getattr(m, "content", "") or ""
            try:
                parsed  = _json.loads(raw)
                preview = _json.dumps(parsed, ensure_ascii=False, indent=2)
            except Exception:
                preview = raw
            if len(preview) > 500:
                preview = preview[:500] + "\n… (truncated)"
            _box(f"🔧  TOOL RESULT  —  {tool_name}", [""] + preview.splitlines())

    # ── DEDUP ─────────────────────────────────────────────────────────────
    elif node_name == "dedup":
        msgs = node_output.get("messages", [])
        skipped_names = [
            getattr(m, "name", "?") for m in msgs
            if isinstance(m, ToolMessage) and "[SKIPPED" in getattr(m, "content", "")
        ]
        if skipped_names:
            _box(
                "⏭️   DEDUP  →  duplicate call(s) skipped",
                [f"Skipped: {', '.join(skipped_names)}  (already called with same params)"],
            )

    # ── COLLECT ───────────────────────────────────────────────────────────
    elif node_name == "collect":
        new_ctx    = node_output.get("accumulated_context", [])
        new_silent = node_output.get("silent_context", [])
        tools_used = node_output.get("tool_calls_this_round", 0)
        lines = [f"{len(new_ctx)} new item(s) added to accumulated context."]
        if new_silent:
            lines.append(f"{len(new_silent)} side-effect result(s) stored in silent context.")
        _box(
            f"📦  CONTEXT COLLECTED  (round {tools_used} / {MAX_TOOL_CALLS_PER_ROUND})",
            lines,
        )

    # ── JUDGE ─────────────────────────────────────────────────────────────
    elif node_name == "judge":
        satisfied       = node_output.get("satisfied", False)
        missing         = node_output.get("judge_missing", "")
        src             = node_output.get("judge_missing_source", "llm")
        tools_used      = node_output.get("judge_tools_this_round", 0)
        deps_check_info = node_output.get("judge_deps_check_info", "")
        body = [f"Tools used this round : {tools_used} / {MAX_TOOL_CALLS_PER_ROUND}"]
        if deps_check_info:
            body.append(f"Deps check    : {deps_check_info}")
        if not satisfied and missing:
            prefix = "[Python override]" if src == "python_override" else "[LLM]"
            body.append(f"Missing {prefix} : {missing}")
        _box(
            f"⚖️   JUDGE  →  {'satisfied ✅' if satisfied else 'not yet satisfied ❌'}",
            body,
        )

    # ── REFORMULATE ───────────────────────────────────────────────────────
    elif node_name == "reformulate":
        new_query      = node_output.get("current_query", "?")
        reformulations = node_output.get("query_reformulations", 0)
        # Read judge_missing from state (not node_output — reformulate doesn't echo it)
        judge_missing_val = state.get("judge_missing", "") if state else ""
        body = [f"Original     : {original_query}"]
        if reformulations > 1:
            body.append(f"Previous     : (see above attempts)")
        if judge_missing_val:
            body.append(f"Judge said   : {judge_missing_val}")
        body.append(f"New query    : {new_query}")
        _box(
            f"🔄  REFORMULATION  (attempt {reformulations} / {MAX_REFORMULATIONS})",
            body,
        )

    # ── ANSWER ────────────────────────────────────────────────────────────
    elif node_name == "answer":
        _box("✅  ANSWER NODE  →  generating final response", [])

    # ── CLARIFY ───────────────────────────────────────────────────────────
    elif node_name == "clarify":
        _box(
            "❓  CLARIFY  →  exhausted all attempts, asking student",
            [f"Tried {MAX_REFORMULATIONS} reformulations × "
             f"{MAX_TOOL_CALLS_PER_ROUND} tools each = "
             f"{MAX_REFORMULATIONS * MAX_TOOL_CALLS_PER_ROUND} total tool rounds."],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

_APP = None



def _dedup_node(state: AgentState) -> dict:
    """
    Deduplication node — sits between agent and tools.

    Inspects the tool calls in the last AIMessage.  For any call whose
    canonical signature (tool_name|sorted_args_json) already exists in
    state["called_tools"], inject a fake ToolMessage that says SKIPPED
    instead of actually calling the tool.

    This means the ToolNode never runs for duplicate calls — they are
    counted as a used round but produce no new context.

    Returns the injected SKIPPED ToolMessages (so LangGraph can route
    to collect → judge normally), plus the signatures added to called_tools.
    """
    import json as _j

    messages      = state["messages"]
    called_sigs   = set(state.get("called_tools", []))

    # Find the last AIMessage
    last_ai: Optional[AIMessage] = None
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            last_ai = m
            break

    if not last_ai or not last_ai.tool_calls:
        return {}

    skipped_msgs: List[ToolMessage] = []
    new_sigs:     List[str]         = []

    for tc in last_ai.tool_calls:
        name     = tc.get("name", "?")
        args     = tc.get("args", {})
        tc_id    = tc.get("id", name)
        args_str = _j.dumps(args, sort_keys=True, ensure_ascii=False)
        sig      = f"{name}|{args_str}"

        if sig in called_sigs:
            # Build a fake ToolMessage so LangGraph sees a response
            skipped_msgs.append(
                ToolMessage(
                    content=f"[SKIPPED — already called with identical parameters: {sig}]",
                    tool_call_id=tc_id,
                    name=name,
                )
            )
        else:
            new_sigs.append(sig)

    if skipped_msgs:
        return {
            "messages":    skipped_msgs,   # inject as ToolMessages
            "called_tools": new_sigs,      # record only newly seen sigs
        }
    # No duplicates — let tools run normally (routing will call tools)
    return {"called_tools": new_sigs}


def _route_after_dedup(state: AgentState) -> Literal["tools", "collect"]:
    """
    After dedup node:
    - If the last messages contain SKIPPED ToolMessages → collect (bypass tools)
    - Otherwise → tools (run normally)
    """
    messages = state["messages"]
    # Walk back to find the last AIMessage and what follows it
    saw_ai = False
    for m in reversed(messages):
        if isinstance(m, ToolMessage):
            content = getattr(m, "content", "")
            if "[SKIPPED" in content:
                return "collect"   # at least one was skipped; bypass ToolNode
            saw_ai = False  # reset — keep looking
        elif isinstance(m, AIMessage):
            break
    return "tools"


def _build_graph():
    """
    Compile the LangGraph StateGraph for the judging loop.
    Called once — cached in _APP.
    """
    llm            = _get_llm()
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    graph = StateGraph(AgentState)

    graph.add_node("agent",       _make_agent_node(llm_with_tools))
    graph.add_node("dedup",       _dedup_node)
    graph.add_node("tools",       ToolNode(ALL_TOOLS))
    graph.add_node("collect",     _collect_node)
    graph.add_node("judge",       _make_judge_node())
    graph.add_node("answer",      _make_answer_node(llm))
    graph.add_node("reformulate", _make_reformulate_node())
    graph.add_node("clarify",     _make_clarify_node(llm))

    # ── Edges ────────────────────────────────────────────────────────────
    graph.add_edge(START, "agent")

    # After agent: always go to dedup first (checks for duplicate calls)
    graph.add_conditional_edges(
        "agent",
        _route_after_agent,
        {"tools": "dedup", "answer": "answer"},
    )

    # After dedup: run tools normally OR skip to collect if all duplicates
    graph.add_conditional_edges(
        "dedup",
        _route_after_dedup,
        {"tools": "tools", "collect": "collect"},
    )

    # After tools: collect context, then judge
    graph.add_edge("tools",   "collect")
    graph.add_edge("collect", "judge")

    # After judge: the heart of the judging loop
    graph.add_conditional_edges(
        "judge",
        _route_after_judge,
        {
            "answer":     "answer",
            "agent":      "agent",
            "reformulate": "reformulate",
            "clarify":    "clarify",
        },
    )

    # After reformulate: start a fresh tool-call round
    graph.add_edge("reformulate", "agent")

    # Terminal nodes
    graph.add_edge("answer",  END)
    graph.add_edge("clarify", END)

    return graph.compile()


def _get_app():
    global _APP
    if _APP is None:
        _APP = _build_graph()
    return _APP


# ─────────────────────────────────────────────────────────────────────────────
# BNUAdvisorAgent — public API
# ─────────────────────────────────────────────────────────────────────────────

class BNUAdvisorAgent:
    """
    BNU Academic Advisor powered by a LangGraph judging-loop agent.

    Usage
    ─────
        agent = BNUAdvisorAgent(student_id="22030094")
        answer = agent.run(
            query="what does probability and statistics close?",
            history=[...],
        )
        # With debug output:
        answer = agent.run(query="...", history=[...], verbose=True)
    """

    def __init__(self, student_id: str):
        self.student_id = student_id

    def run(
        self,
        query:   str,
        history: Optional[List[Dict]] = None,
        verbose: bool = False,
    ) -> str:
        """
        Run the judging loop for one student query.

        Args:
            query:   The (preprocessed, canonical) student query.
            history: Recent conversation turns for agent context.
            verbose: If True, prints step-by-step debug boxes.

        Returns:
            Final answer string.
        """
        history = history or []

        # ── Load conversation history into messages ───────────────────────
        # history[-6:] = last 6 messages from Supabase = last 3 Q/A pairs.
        # Each "turn" is ONE message: either a student question or an
        # advisor answer.  So 6 turns = 3 student questions + 3 advisor answers.
        #
        # Example with 3 Q/A pairs loaded:
        #   [0] HumanMessage  "what is ml?"           ← student turn 1
        #   [1] AIMessage     "ML is a course..."     ← advisor turn 1
        #   [2] HumanMessage  "when is it offered?"   ← student turn 2
        #   [3] AIMessage     "It is in year 3..."    ← advisor turn 2
        #   [4] HumanMessage  "what does it close?"   ← student turn 3
        #   [5] AIMessage     "It closes deep..."     ← advisor turn 3
        #   [6] HumanMessage  <<current query>>       ← appended last
        #
        # The agent LLM sees ALL of these as conversation context.
        HISTORY_WINDOW = 6   # last 6 messages = 3 Q/A pairs
        history_slice  = history[-HISTORY_WINDOW:]

        messages: List[BaseMessage] = []
        for turn in history_slice:
            role         = turn.get("role", "")
            turn_content = turn.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=turn_content))
            elif role == "assistant":
                messages.append(AIMessage(content=turn_content))
        messages.append(HumanMessage(content=query))   # current query always last

        # ── Debug: show loaded history (verbose mode only) ────────────────
        if verbose or _is_verbose():
            history_lines = []
            if history_slice:
                for i, turn in enumerate(history_slice):
                    role    = turn.get("role", "user")
                    text    = turn.get("content", "")
                    label   = "Student" if role == "user" else "Advisor"
                    # Truncate long advisor answers for readability
                    preview = (text[:120] + " …") if len(text) > 120 else text
                    history_lines.append(f"[{i+1}] {label}: {preview}")
                history_lines.append("")
                history_lines.append(
                    f"Total: {len(history_slice)} message(s) loaded  "
                    f"= {len(history_slice)//2} complete Q/A pair(s)"
                    + (f"  + {len(history_slice)%2} partial" if len(history_slice)%2 else "")
                )
            else:
                history_lines.append("(no previous conversation — fresh session)")
            history_lines.append(f"Current query: {query}")
            _box("💬  AGENT CONTEXT  —  conversation history loaded", history_lines)

        # Initial state — include all AgentState fields
        initial_state: AgentState = {
            "messages":                  messages,
            "accumulated_context":       [],
            "silent_context":            [],
            "called_tools":              [],
            "previous_reformulations":   [],
            "tool_calls_this_round":     0,
            "query_reformulations":      0,
            "original_query":            query,
            "current_query":             query,
            "satisfied":                 False,
            "judge_missing":             "",
            "judge_missing_source":      "llm",
            "judge_deps_check_info":     "",
            "judge_tools_this_round":    0,
        }

        # RunnableConfig — student_id reaches tools via configurable
        run_config = {
            "configurable": {"student_id": self.student_id},
            # Allow enough steps for the full judging loop
            # Each round: agent + dedup + tools + collect + judge = 5 nodes
            # Total rounds: (1 initial + MAX_REFORMULATIONS) × MAX_TOOL_CALLS_PER_ROUND
            # Plus: reformulate nodes + answer/clarify + buffer
            "recursion_limit": (
                (MAX_REFORMULATIONS + 1)          # reformulation cycles
                * MAX_TOOL_CALLS_PER_ROUND        # tool rounds per cycle
                * 6                               # nodes per round (with dedup)
                + MAX_REFORMULATIONS              # reformulate nodes
                + 10                              # answer/clarify/start/buffer
            ),
        }

        app = _get_app()

        # Set the active student ID in tools.py so all tools that need
        # student_id can read it directly without relying on RunnableConfig
        # forwarding (which is unreliable across LangGraph versions).
        from tools import set_active_student_id
        set_active_student_id(self.student_id)

        try:
            if verbose:
                return self._run_streaming(app, initial_state, run_config, query)
            else:
                return self._run_blocking(app, initial_state, run_config)
        except Exception as exc:
            logger.error("Graph execution error: %s", exc, exc_info=True)
            return (
                "I'm experiencing a technical issue right now. "
                "Please try again in a moment."
            )

    def run_and_get_context(
        self,
        query:   str,
        history: Optional[List[Dict]] = None,
        verbose: bool = False,
    ) -> List[str]:
        """
        Run the full judging loop for one sub-query and return the
        accumulated tool context (list of raw tool-result strings) WITHOUT
        generating a final answer text.

        Used by chatbot_api._split_and_run() to collect context from
        multiple sub-queries before synthesising one combined answer.
        """
        history = history or []
        history_slice = history[-6:]

        messages: List[BaseMessage] = []
        for turn in history_slice:
            role    = turn.get("role", "")
            content = turn.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=query))

        initial_state: AgentState = {
            "messages":                messages,
            "accumulated_context":     [],
            "silent_context":          [],
            "called_tools":            [],
            "previous_reformulations": [],
            "tool_calls_this_round":   0,
            "query_reformulations":    0,
            "original_query":          query,
            "current_query":           query,
            "satisfied":               False,
            "judge_missing":           "",
            "judge_missing_source":    "llm",
            "judge_deps_check_info":   "",
            "judge_tools_this_round":  0,
        }

        run_config = {
            "configurable": {"student_id": self.student_id},
            "recursion_limit": (
                (MAX_REFORMULATIONS + 1) * MAX_TOOL_CALLS_PER_ROUND * 6
                + MAX_REFORMULATIONS + 10
            ),
        }

        app = _get_app()
        from tools import set_active_student_id
        set_active_student_id(self.student_id)

        try:
            if verbose:
                # Stream and print debug boxes for tool-selection/judging nodes,
                # but SKIP the answer/clarify nodes — a single combined answer
                # will be synthesised from all sub-query contexts after the loop.
                _SKIP_IN_SUBQUERY = {"answer", "clarify"}
                all_context: List[str] = []
                _last_judge_missing = ""
                for event in app.stream(
                    initial_state, config=run_config, stream_mode="updates"
                ):
                    for node_name, node_output in event.items():
                        if node_name == "judge":
                            _last_judge_missing = node_output.get("judge_missing", "")
                        if node_name not in _SKIP_IN_SUBQUERY:
                            _print_step(
                                node_name,
                                node_output,
                                original_query=query,
                                verbose=True,
                                state={"judge_missing": _last_judge_missing},
                            )
                        if node_name == "collect":
                            all_context.extend(
                                node_output.get("accumulated_context", [])
                            )
                return all_context
            else:
                final_state = app.invoke(initial_state, config=run_config)
                return final_state.get("accumulated_context", [])
        except Exception as exc:
            logger.error("run_and_get_context error: %s", exc, exc_info=True)
            return []

    # ── Execution modes ───────────────────────────────────────────────────

    def _run_blocking(self, app, initial_state, config) -> str:
        """Run the graph without streaming — returns final answer."""
        final_state = app.invoke(initial_state, config=config)
        return self._extract_answer(final_state)

    def _run_streaming(self, app, initial_state, config, original_query: str) -> str:
        """
        Run the graph with streaming — prints a debug box per node,
        returns the final answer from the answer/clarify node.

        Uses stream_mode="updates" which yields ONLY the delta per node,
        NOT the full accumulated state.  This means:
          - node_output["accumulated_context"] = only NEW items this step
          - node_output["tool_calls_this_round"] = current total value
        """
        last_ai_answer = None
        _last_judge_missing = ""

        for event in app.stream(
            initial_state,
            config=config,
            stream_mode="updates",
        ):
            for node_name, node_output in event.items():
                new_messages = node_output.get("messages", [])

                # Track judge_missing so reformulate debug box can show it
                if node_name == "judge":
                    _last_judge_missing = node_output.get("judge_missing", "")

                # Print debug box for this node
                _print_step(
                    node_name,
                    node_output,
                    original_query=original_query,
                    verbose=True,
                    state={"judge_missing": _last_judge_missing},
                )

                # Capture the final AI answer (from answer or clarify nodes)
                for m in new_messages:
                    if isinstance(m, AIMessage) and not m.tool_calls:
                        last_ai_answer = m.content

        if last_ai_answer is not None:
            return last_ai_answer

        return (
            "I could not generate a complete answer. "
            "Please try rephrasing your question with more detail."
        )

    # ── Answer extraction ─────────────────────────────────────────────────

    @staticmethod
    def _extract_answer(final_state: dict) -> str:
        """Extract the last non-tool AIMessage content as the final answer."""
        ai_messages = [
            m for m in final_state.get("messages", [])
            if isinstance(m, AIMessage) and not m.tool_calls
        ]
        if ai_messages:
            return ai_messages[-1].content or ""
        # Fallback
        last = final_state["messages"][-1]
        return getattr(last, "content", str(last))


# ─────────────────────────────────────────────────────────────────────────────
# Interactive CLI  —  run: python agent.py
# ─────────────────────────────────────────────────────────────────────────────
#
# Usage:
#   python agent.py                      → interactive chat, uses .env STUDENT_ID
#   python agent.py --student 22030094   → override student ID
#   python agent.py --verbose            → show judging loop debug boxes
#   python agent.py --test               → one smoke test query, then exit
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BNU Academic Advisor — CLI chat")
    parser.add_argument(
        "--student",
        default=os.getenv("STUDENT_ID", "22030094"),
        help="Student ID (default: STUDENT_ID from .env)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show judging loop: tool calls, judge verdicts, reformulations",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run one smoke-test query then exit",
    )
    args = parser.parse_args()

    # ── Logging: silence third-party noise, enable ours if verbose ────────
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s | %(name)s | %(message)s",
    )
    for _noisy in ("neo4j", "neo4j.io", "neo4j.pool", "groq",
                   "httpx", "httpcore", "httpcore.http11", "urllib3"):
        logging.getLogger(_noisy).setLevel(logging.ERROR)
    if args.verbose:
        logging.getLogger("tools").setLevel(logging.DEBUG)
        logging.getLogger("agent").setLevel(logging.DEBUG)
        logging.getLogger("preprocessor").setLevel(logging.DEBUG)

    # ── Set global verbose flag (used by debug_box.box) ───────────────────
    _set_verbose(args.verbose)

    student_id      = args.student
    agent           = BNUAdvisorAgent(student_id=student_id)
    session_history: List[Dict] = []
    pending_ambiguity = None

    # Import planning session routing (monkey-patch applied on chatbot_api import)
    try:
        import chatbot_api as _chatbot_api
        from planning_service import PlanningOrchestrator, PlanStep
        _planning_available = True
    except Exception:
        _planning_available = False

    print()
    print("=" * 60)
    print("  🎓  BNU Academic Advisor  (LangGraph  ·  Judging Loop)")
    print("=" * 60)
    print(f"  Student ID  : {student_id}")
    print(f"  Model (agent) : {GROQ_MODEL_AGENT}")
    print(f"  Max tools   : {MAX_TOOL_CALLS_PER_ROUND} / round, "
          f"{MAX_REFORMULATIONS} reformulations max")
    print(f"  Debug mode  : {'ON — full judging loop shown' if args.verbose else 'OFF (use --verbose)'}")
    print("  Type 'exit' or 'quit' to end.")
    print("=" * 60)
    print()

    # ── Smoke test ────────────────────────────────────────────────────────
    if args.test:
        from preprocessor import get_preprocessor
        q = "what does probability and statistical methods close"
        print(f"[SMOKE TEST] {q}\n{'-' * 60}")
        pre    = get_preprocessor()
        result = pre.process(q, history=[])
        clean  = result.clean_query or q
        ans    = agent.run(clean, history=[], verbose=True)
        print(f"\nAnswer:\n{ans}")
        print(f"{'-' * 60}\n[SMOKE TEST] Done ✅")
        exit(0)

    # ── Interactive loop ──────────────────────────────────────────────────
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "bye", "q"}:
            print("\nGoodbye! 👋")
            break

        try:
            # ── Route to active planning session first ────────────────────
            if (
                _planning_available
                and student_id in _chatbot_api._planning_sessions
            ):
                state = _chatbot_api._planning_sessions[student_id]
                response, state = PlanningOrchestrator.advance(state, user_input)
                if state.current_step == PlanStep.COMPLETE:
                    del _chatbot_api._planning_sessions[student_id]
            else:
                from preprocessor import get_preprocessor
                pre = get_preprocessor()

                # ── Resolve pending ambiguity ─────────────────────────────
                if pending_ambiguity is not None:
                    result          = pre.resolve_ambiguity(pending_ambiguity, user_input)
                    pending_ambiguity = None
                else:
                    result = pre.process(user_input, session_history)

                # ── Route based on preprocessor outcome ──────────────────
                if result.status == "ambiguous":
                    pending_ambiguity = result.pending
                    response          = result.clarification
                else:
                    clean_query = result.clean_query or user_input
                    if args.verbose:
                        print()   # blank line before agent boxes

                    # Decompose into sub-queries if the query asks about
                    # multiple independent topics (two courses, course + bylaw, etc.)
                    split_done = False
                    if _planning_available:
                        sub_queries = _chatbot_api._analyze_and_split(clean_query)
                        if len(sub_queries) > 1:
                            response   = _chatbot_api._split_and_run(
                                student_id  = student_id,
                                clean_query = clean_query,
                                sub_queries = sub_queries,
                                history     = session_history,
                                verbose     = args.verbose,
                            )
                            split_done = True

                    if not split_done:
                        response = agent.run(
                            query   = clean_query,
                            history = session_history,
                            verbose = args.verbose,
                        )

        except Exception as exc:
            pending_ambiguity = None   # clear stale state on error
            response = f"⚠️  Error: {exc}"

        if args.verbose:
            print("\n" + "─" * 60)
            print("Answer:")
        else:
            print()
        print(f"Advisor: {response}")
        print()

        # Keep last 6 messages (3 user + 3 assistant) — mirrors Supabase
        session_history.append({"role": "user",      "content": user_input})
        session_history.append({"role": "assistant",  "content": response})
        session_history = session_history[-6:]