"""
tools.py - LangChain Tool Definitions for the BNU Academic Advisor Agent
=========================================================================

HOW student_id IS HANDLED
──────────────────────────
Tools that need the student's ID call _get_student_id() internally.
The LLM never sees student_id as a parameter — it is NOT in any tool schema.

Priority chain:
  1. _ACTIVE_STUDENT_ID  - set by agent.run() via set_active_student_id()
                           before each graph execution (production/API path).
  2. os.getenv("STUDENT_ID") - .env fallback for local CLI development.

Note: InjectedToolArg / RunnableConfig was removed because LangGraph's
ToolNode does not reliably forward RunnableConfig.configurable to tool
function parameters across all versions.  The module-level variable
approach is simpler and works everywhere.

COURSE NAME NORMALISATION
─────────────────────────

COURSE NAME NORMALISATION
─────────────────────────
Every tool that accepts a course_name applies fuzzy matching internally via
_normalize_course(). The LLM can pass the name exactly as the student typed
it (partial names, abbreviations, typos) — the tool handles normalisation.

ALL tools return plain strings. LangGraph's ToolNode passes them back to the
LLM as ToolMessage content.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_course(raw_name: str) -> str:
    """
    Silent safety-net fuzzy match. The preprocessor handles this before the
    agent runs, so this function is rarely needed. No logging — the
    preprocessor already showed the user any relevant mapping decisions.
    """
    if not raw_name:
        return raw_name
    try:
        from course_name_mapper import map_course_name
        mapped = map_course_name(raw_name.strip(), threshold=0.3)
        if mapped:
            return mapped
    except Exception:
        pass
    return raw_name


# ── Active student ID ────────────────────────────────────────────────────────
#
# Priority chain (highest lowest):
#   1. _ACTIVE_STUDENT_ID   - set by agent.run() via set_active_student_id()
#                             before each graph invocation.
#   2. os.getenv("STUDENT_ID") - .env fallback for local CLI development.
#
# Thread safety: agent.run() sets this before calling app.invoke/stream
# and the graph executes synchronously, so there is no race condition in
# single-threaded deployments.  For async/multi-threaded production use,
# replace with a contextvars.ContextVar.

_ACTIVE_STUDENT_ID: str = ""


def set_active_student_id(student_id: str) -> None:
    """Called by agent.run() before every graph execution."""
    global _ACTIVE_STUDENT_ID
    _ACTIVE_STUDENT_ID = student_id or os.getenv("STUDENT_ID", "")


def _get_student_id() -> str:
    """
    Return the active student ID.
    Falls back to STUDENT_ID from .env if not set by the agent.
    """
    return _ACTIVE_STUDENT_ID or os.getenv("STUDENT_ID", "")


def _to_str(result) -> str:
    """Serialise any return value to a human-readable string."""
    if result is None:
        return "No information found for the given parameters."
    if isinstance(result, (dict, list)):
        return json.dumps(result, ensure_ascii=False, indent=2)
    return str(result)


# ─────────────────────────────────────────────────────────────────────────────
# ── Student profile ──────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_student_info() -> str:
    """
    The sole source for the current student's personal academic record.

    Covers everything that belongs to this student specifically: identity,
    program, year, GPA, total earned credits, completed courses with grades,
    and semester-by-semester GPA history.

    Does not answer questions about courses, curriculum, or policies — those
    live in Neo4j and the bylaws. Accepts no parameters; student ID is
    injected automatically.
    """
    sid = _get_student_id()
    try:
        from student_functions import get_student_details
        return _to_str(get_student_details(sid))
    except Exception as exc:
        return f"Error fetching student info: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# ── Course information ───────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_course_info(course_name: str, program_name: Optional[str] = None) -> str:
    """
    The source for static facts about a single course: what it is, its credit
    value, its code, and its role in the curriculum (core or elective).

    Belongs to queries about the nature of a course itself — content,
    description, weight, classification.

    Does not cover when it is offered (get_course_timing), what leads to or
    from it (get_course_dependencies), or whether the student can take it
    (check_course_eligibility). program_name is optional; only affects code
    resolution for the six courses whose codes differ per program.
    """
    try:
        from neo4j_course_functions import get_course_info as _fn
        return _to_str(_fn(_normalize_course(course_name), program_name=program_name))
    except Exception as exc:
        return f"Error fetching course info: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# ── Prerequisites ────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_course_dependencies(
    course_name: str,
    program_name: Optional[str] = None,
    prereq: bool = True,
    dependents: bool = True,
) -> str:
    """
    Get prerequisite and/or dependent information for a course.

    prereq=True     → what must be done BEFORE X can be taken (X's prerequisites).
    dependents=True → what becomes available AFTER X is completed (courses X unlocks).

    Choosing flags — ask ONE question:
        "Is the student asking about what comes BEFORE X, or what comes AFTER X?"

        BEFORE X → prereq=True,  dependents=False
        AFTER X  → prereq=False, dependents=True
        Both / uncertain → both=True (default — always safe when unsure)

    To resolve BEFORE vs AFTER, reconstruct what the correct answer would look like:
        Answer = courses the student must finish TO REACH X          → BEFORE → prereq
        Answer = courses the student can take HAVING COMPLETED X     → AFTER  → dependents

    No surface keyword reliably indicates direction. Always reason from meaning:
    what would a complete and correct answer to this query actually contain?

    Args:
        course_name:  Course name
        program_name: Student's program/track (optional).
        prereq:       If True (default), include prerequisites in the result.
        dependents:   If True (default), include dependent/closes courses in the result.
    """
    try:
        from neo4j_course_functions import get_course_dependencies
        return _to_str(
            get_course_dependencies(
                _normalize_course(course_name), program_name,
                prereq=prereq, dependents=dependents,
            )
        )
    except Exception as exc:
        return f"Error fetching prerequisites: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# ── Course timing ────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_course_timing(
    course_name: str,
    program_name: Optional[str] = None,
) -> str:
    """
    The only source for where in the academic timeline a course is scheduled:
    its year level (1–4) and semester (1 or 2).

    Belongs to "when is this course offered?" — positioning within the
    curriculum calendar.

    Does not list all courses in a given term (get_courses_by_term). Does not
    say whether the student can enroll now (check_course_eligibility). Accepts
    course names; program_name optional but improves accuracy when a
    course's position differs across programs.
    """
    try:
        from neo4j_course_functions import get_course_timing as _fn
        return _to_str(_fn(_normalize_course(course_name), program_name))
    except Exception as exc:
        return f"Error fetching course timing: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# ── Eligibility check ────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@tool
def check_course_eligibility(
    course_name: str,
) -> str:
    """
    The only tool that answers whether the current student is allowed to
    enroll in a specific course right now, given their actual academic record.

    Belongs to all enrollment-gate questions: prerequisite gaps, credit-hour
    thresholds, program membership.

    Evaluates one course at a time. Does not list all courses the student is
    eligible for. Does not suggest what to take next (start_course_planning).
    The missing prerequisites in the result are only what the student currently
    lacks, not the full prerequisite list (use get_course_dependencies for
    that). course_name accepts input; student data is read automatically.
    """
    sid = _get_student_id()
    try:
        from eligibility import check_course_eligibility as _check
        return _to_str(
            _check(
                student_id=sid,
                course_name=_normalize_course(course_name),
            )
        )
    except Exception as exc:
        return f"Error checking eligibility: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# ── Courses by term ──────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_courses_by_term(
    level: int,
    semester: int,
    program_name: Optional[str] = None,
) -> str:
    """
    Returns every course scheduled in a single academic term (one year +
    semester pair) as it appears in the curriculum.

    Belongs to "what's in year X semester Y?" — the static curriculum snapshot
    for one time slot.

    For queries spanning multiple terms, use get_courses_by_multiple_terms to
    avoid separate round trips. Does not reflect the student's progress or
    eligibility. level is 1–4; semester is 1 or 2; program_name optional
    (omitting it returns all programs' courses for that term).
    """
    try:
        from neo4j_course_functions import get_courses_by_term as _fn
        return _to_str(_fn(level, semester, program_name))
    except Exception as exc:
        return f"Error fetching courses by term: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# ── Courses by multiple terms ────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_courses_by_multiple_terms(
    terms: List[Dict],
    program_name: Optional[str] = None,
) -> str:
    """
    Retrieves curriculum data for several academic terms in one call — the
    batch form of get_courses_by_term.

    Belongs to queries spanning more than one semester. Use this instead of
    calling get_courses_by_term multiple times.

    terms is a list of {"level": int, "semester": int} objects; a full year
    requires two entries (semester 1 and 2). program_name optional.
    """
    try:
        from neo4j_course_functions import get_courses_by_multiple_terms as _fn
        return _to_str(_fn(terms, program_name))
    except Exception as exc:
        return f"Error fetching courses for multiple terms: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# ── Electives ────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_all_electives(program_name: str) -> str:
    """
    The complete elective course catalogue for a given program — every course
    available to fill elective slots.

    Belongs to "what elective courses can I choose from?" — the content menu
    of optional courses.

    Does not answer when elective slots occur or how many exist
    (get_elective_slots_time_and_occ). program_name is required.
    """
    try:
        from neo4j_course_functions import get_all_electives_by_program, run_cypher_query

        # Use the full query (credit_hours + description) directly to avoid
        # depending on whichever version of get_all_electives_by_program is installed.
        prog = program_name.lower().strip() if program_name else program_name
        query = """
        MATCH (c:Course)-[r:BELONGS_TO]->(p:Program {name: $program_name})
        WHERE r.elective = 'yes'
        RETURN
            c.name AS course_name,
            COALESCE(c.code, r.code) AS course_code,
            c.description AS description,
            c.credit_hours AS credit_hours
        ORDER BY COALESCE(c.code, r.code)
        """
        result = run_cypher_query(query, {"program_name": prog})
        return _to_str(result)
    except Exception as exc:
        return f"Error fetching electives: {exc}"


@tool
def get_elective_slots_time_and_occ(program_name: str) -> str:
    """
    Returns the schedule of elective slots in a program: which year/semester
    they fall in and how many slots are available at each point.

    Belongs to questions about the timing and count of elective opportunities,
    not their content.

    Does not list the courses that fill those slots (get_all_electives or
    get_specialized_elective_courses). program_name required.
    """
    try:
        from neo4j_course_functions import get_elective_slots_time_and_occ as _fn
        return _to_str(_fn(program_name))
    except Exception as exc:
        return f"Error fetching elective slots: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# ── Filter / search courses ──────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@tool
def filter_courses(
    program_name: Optional[str] = None,
    filters: Optional[Dict] = None,
    course_types: Optional[List[str]] = None,
    return_fields: Optional[List[str]] = None,
    course_list: Optional[List[str]] = None,
) -> str:
    """
    A flexible multi-criteria search engine over the course graph — finds
    courses matching a combination of attributes.

    Belongs to exploratory or attribute-filtered searches where the target is
    a set of courses matching conditions, not a known category.

    Use category-specific tools (get_all_specialized_courses, get_general_courses,
    etc.) when you want a well-defined category wholesale. Use get_course_info
    for a single known course. All parameters are optional; apply only the
    filters relevant to the query.

    filters: dict of attribute conditions, e.g. {"credit_hours": 3}.
    course_types: ["core"], ["elective"], or both.
    course_list: restricts filtering to a specific set of course names.
    return_fields: limits which fields appear in the output.
    """
    try:
        from neo4j_course_functions import filter_courses as _fn
        return _to_str(
            _fn(
                filters=filters,
                course_types=course_types,
                return_fields=return_fields,
                program_name=program_name,
                course_list=course_list,
            )
        )
    except Exception as exc:
        return f"Error filtering courses: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# ── Program / track info ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_program_total_credits(program_name: str) -> str:
    """
    Returns the single total graduation credit-hour requirement for a program.

    Belongs to "how many credits do I need to graduate?" — one scalar answer
    per program.

    Does not break the total into categories (get_credit_hour_distribution).
    Does not compare with the student's earned credits. program_name required.
    """
    try:
        from neo4j_track_functions import get_program_total_credits as _fn
        return _to_str(_fn(program_name))
    except Exception as exc:
        return f"Error fetching program credit requirement: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# ── BNU bylaws / regulations (RAG) ───────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@tool
def answer_academic_question(question: str) -> str:
    """
    The RAG interface to BNU's official bylaws and academic regulations — the
    authoritative source for all institutional policy questions.

    Belongs to questions about university rules: GPA thresholds, graduation
    conditions, probation, attendance limits, credit transfer, course
    withdrawal, course repetition, academic warnings, leave of absence,
    disciplinary procedures — anything that lives in official documents.

    Does not know course details, program structures, or student records —
    those are Neo4j/Supabase tools. Answers are bounded by what the bylaws
    cover; if a topic is absent from the regulations, the tool will say so.
    question should be a full, natural-language policy question.
    """
    try:
        from rag_service import handle_general_query
        return handle_general_query(question=question, history=[])
    except Exception as exc:
        return f"Error searching knowledge base: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# ── Course planning ───────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@tool
def start_course_planning() -> str:
    """
    The entry point for a personalised, multi-turn interactive course planning
    session that builds a study plan tailored to the student's record and goals.

    Belongs to requests for a planned path: semester recommendations,
    identifying remaining graduation requirements, personalised scheduling.

    Calling this tool ends the current tool-selection loop — the student then
    enters a dedicated planner conversation. Do not call any other tool in the
    same round as this one. Not for one-off prerequisite checks
    (get_course_dependencies) or single-course eligibility
    (check_course_eligibility). No parameters; uses the student record
    automatically.
    """
    # NOTE: chatbot_api.py patches this function at startup to also capture
    # the PlanningState for multi-turn continuation.
    sid = _get_student_id()
    try:
        from planning_service import PlanningOrchestrator
        from chatbot_connector import ChatbotConnector
        supabase_client = ChatbotConnector().client
        message, _ = PlanningOrchestrator.start(sid, supabase_client)
        return message or "Course planning session started."
    except Exception as exc:
        return f"Could not start course planning: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# ── Program / track info ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_program_info(
    prg: str,
    course_info: bool = True,
    desc_info: bool = True,
) -> str:
    """
    The comprehensive data package for one academic program: description,
    credit category breakdown, full years-3/4 curriculum, elective slot
    schedule, and elective catalogue.

    Belongs to "tell me about program X" — the full structural profile of a
    single track.

    For comparing programs, use compare_programs (which calls this per program
    atomically — do not call manually per-program for a comparison). For
    the credit breakdown alone, get_credit_hour_distribution is lighter.

    prg is the full program name. course_info=False skips curriculum/elective
    data when only the description is needed. desc_info=False skips the
    description.
    """
    try:
        from neo4j_track_functions import get_program_info as _fn
        return _to_str(_fn(prg, course_info=course_info, desc_info=desc_info))
    except Exception as exc:
        return f"Error fetching program info: {exc}"


@tool
def get_credit_hour_distribution() -> str:
    """
    The faculty-wide credit breakdown by category (GEN, BAS, BCS, specialized,
    field training, graduation projects) — identical across all three programs.

    Belongs to questions about how the 136 total credits are allocated across
    categories.

    Program-independent: call once, never per-program. For the single
    graduation total, use get_program_total_credits. For the actual courses
    within a category, use the category tools. No parameters.
    """
    try:
        from neo4j_track_functions import get_credit_hour_distribution as _fn
        return _to_str(_fn())
    except Exception as exc:
        return f"Error fetching credit hour distribution: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# ── Course-category tools ────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_specialized_core_courses(
    prg: str,
    year_flag: bool = False,
    sem_flag: bool = False,
) -> str:
    """
    The mandatory, non-elective courses unique to a specific program —
    identified by the program's code prefix (AIM/SAD/DAS) and marked
    non-elective. Part of the 51-credit specialized category.

    Belongs to questions about the obligatory specialized work that defines
    a track.

    "Specialized" means program-prefixed, not merely advanced or upper-level.
    Does not include GEN, BAS, or BCS courses. For electives alongside these,
    use get_all_specialized_courses. prg required; year_flag/sem_flag add
    per-course scheduling data.
    """
    try:
        from neo4j_track_functions import get_specialized_core_courses as _fn
        return _to_str(_fn(prg, year_flag=year_flag, sem_flag=sem_flag))
    except Exception as exc:
        return f"Error fetching specialized core courses: {exc}"


@tool
def get_specialized_elective_courses(
    prg: str,
    year_flag: bool = False,
    sem_flag: bool = False,
) -> str:
    """
    The elective courses unique to a specific program — program-prefixed
    (AIM/SAD/DAS) and marked elective.

    Belongs to "what optional program-specific courses does this track offer?"

    Mandatory counterpart is get_specialized_core_courses; both together are
    get_all_specialized_courses. When year_flag or sem_flag is True, returns
    the elective slot schedule rather than per-course positioning, because
    elective courses have no fixed year/semester placement in the curriculum.
    prg required.
    """
    try:
        from neo4j_track_functions import get_specialized_elective_courses as _fn
        return _to_str(_fn(prg, year_flag=year_flag, sem_flag=sem_flag))
    except Exception as exc:
        return f"Error fetching specialized elective courses: {exc}"


@tool
def get_all_specialized_courses(
    prg: str,
    year_flag: bool = False,
    sem_flag: bool = False,
) -> str:
    """
    All program-specific courses — mandatory and elective — for a track,
    covering the full 51-credit specialized category.

    Belongs to "what makes this program distinct?" or "what courses are unique
    to this track?" — the complete set of courses that separates one program
    from the others.

    Use the split tools (get_specialized_core_courses, get_specialized_elective_courses)
    only when one subset is specifically needed. Does not include shared
    courses (GEN/BAS/BCS). prg required; year_flag/sem_flag add scheduling
    (elective courses return slot schedule, not per-course timing).
    """
    try:
        from neo4j_track_functions import get_all_specialized_courses as _fn
        return _to_str(_fn(prg, year_flag=year_flag, sem_flag=sem_flag))
    except Exception as exc:
        return f"Error fetching all specialized courses: {exc}"


@tool
def get_general_courses(
    year_flag: bool = False,
    sem_flag: bool = False,
) -> str:
    """
    The humanities and general education courses (GEN prefix, 12 credits) —
    identical across all three programs.

    Belongs to questions specifically about GEN courses.

    Program-independent: the result is the same regardless of track; do not
    call per-program. For the full shared curriculum (GEN + BAS + BCS), use
    get_all_not_specialized_courses. No prg parameter; year_flag/sem_flag add
    scheduling.
    """
    try:
        from neo4j_track_functions import get_general_courses as _fn
        return _to_str(_fn(year_flag=year_flag, sem_flag=sem_flag))
    except Exception as exc:
        return f"Error fetching general courses: {exc}"


@tool
def get_math_and_basic_science_courses(
    year_flag: bool = False,
    sem_flag: bool = False,
) -> str:
    """
    The mathematics and basic sciences courses (BAS prefix, 24 credits) —
    identical across all three programs.

    Belongs to questions specifically about BAS courses.

    Program-independent: do not call per-program. For the full shared
    curriculum, use get_all_not_specialized_courses. No prg parameter;
    year_flag/sem_flag add scheduling.
    """
    try:
        from neo4j_track_functions import get_MathAndBasicScience_courses as _fn
        return _to_str(_fn(year_flag=year_flag, sem_flag=sem_flag))
    except Exception as exc:
        return f"Error fetching math and basic science courses: {exc}"


@tool
def get_basic_computing_sciences_courses(
    prg: str = None,
    year_flag: bool = False,
    sem_flag: bool = False,
) -> str:
    """
    The core computing foundation courses (BCS prefix, 36 credits) — mostly
    identical across programs, with one exception: Data Science substitutes
    "Fundamentals of Data Science" for "Technical Report Writing" (found in
    AIM and SAD).

    Belongs to questions specifically about BCS courses.

    Pass prg when the query is program-specific, especially for Data Science,
    to get the exact correct list. Omitting prg returns all BCS courses with a
    note about the difference. year_flag/sem_flag add scheduling.
    """
    try:
        from neo4j_track_functions import get_BasicComputingSciences_courses as _fn
        return _to_str(_fn(prg=prg, year_flag=year_flag, sem_flag=sem_flag))
    except Exception as exc:
        return f"Error fetching basic computing sciences courses: {exc}"


@tool
def get_all_types_courses(
    prg: str,
    year_flag: bool = False,
    sem_flag: bool = False,
) -> str:
    """
    The complete curriculum for a program, organized into all four course
    categories: specialized, GEN, BAS, and BCS.

    Belongs to full-curriculum queries — "what does this program contain?"
    with no filtering on type or category.

    For only mandatory courses, use get_all_core_courses. For only the shared
    (non-specialized) portion, use get_all_not_specialized_courses. prg
    required; year_flag/sem_flag add scheduling.
    """
    try:
        from neo4j_track_functions import get_all_types_courses as _fn
        return _to_str(_fn(prg, year_flag=year_flag, sem_flag=sem_flag))
    except Exception as exc:
        return f"Error fetching all course types: {exc}"


@tool
def get_all_core_courses(
    prg: str,
    year_flag: bool = False,
    sem_flag: bool = False,
) -> str:
    """
    All mandatory (non-elective) courses for a program, spanning all four
    categories: specialized core, GEN, BAS, and BCS.

    Belongs to "what courses must I take?" — the complete non-optional
    curriculum.

    Excludes electives. For the full curriculum including electives, use
    get_all_types_courses. For only the specialized mandatory subset, use
    get_specialized_core_courses. prg required; year_flag/sem_flag add
    scheduling.
    """
    try:
        from neo4j_track_functions import get_all_core_courses as _fn
        return _to_str(_fn(prg, year_flag=year_flag, sem_flag=sem_flag))
    except Exception as exc:
        return f"Error fetching all core courses: {exc}"


@tool
def get_all_not_specialized_courses(
    prg: str = None,
    year_flag: bool = False,
    sem_flag: bool = False,
) -> str:
    """
    The courses shared across all programs — GEN + BAS + BCS (72 credits) —
    the common foundation independent of track choice.

    Belongs to "what would I take regardless of which program I pick?" or
    "what courses are common to all tracks?"

    Excludes specialized (program-prefixed) courses. For the full program
    curriculum including specialized, use get_all_types_courses. Pass prg
    when the query is for Data Science specifically, to get the correct BCS
    variant. year_flag/sem_flag add scheduling.
    """
    try:
        from neo4j_track_functions import get_all_not_specialized_courses as _fn
        return _to_str(_fn(prg=prg, year_flag=year_flag, sem_flag=sem_flag))
    except Exception as exc:
        return f"Error fetching non-specialized courses: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# ── Preference storage ───────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@tool
def store_preference(preferences: Dict[str, float]) -> str:
    """
    The write interface for the student's persistent preference profile —
    records inferred academic interests, strengths, and dislikes derived from
    what the student says about themselves.

    Belongs to any moment the student reveals a genuine personal signal:
    enthusiasm, skill, background, certification, or aversion toward any
    academic subject — even when expressed implicitly or indirectly.

    Does not apply to course completion or neutral factual discussion. Do not
    call for casual course mentions; only genuine signals about the student as
    a learner. Cannot be queried — it only writes.

    preferences is a dict of {category_key: delta_float}. Deltas are additive
    on top of existing scores, clamped to [0.0, 1.0]. Positive = interest or
    strength; negative = dislike or weakness. One call may cover multiple
    categories. Only these 12 keys are valid (others are silently ignored):

      math                  — calculus, linear algebra, discrete math, numerical methods
      probability_statistics — probability, statistics, stochastic processes
      programming           — coding, algorithms, data structures, competitive programming
      software_engineering  — design patterns, web/mobile dev, system design, SDLC
      ai_ml                 — machine learning, deep learning, AI broadly
      data_management       — databases, SQL, data warehousing, data engineering
      data_analysis         — analytics, BI, visualization, insight extraction
      theory                — automata, complexity, formal methods, logic
      networking_systems    — networks, OS, security, infrastructure, CCNA
      visual_computing      — image processing, computer graphics, geometry
      language_text         — NLP, linguistics, text processing, translation
      optimization          — operations research, numerical optimization

    Delta magnitude guide:
      ±0.10  encountered it / studied briefly
      ±0.15  comfortable or uncomfortable with it
      ±0.20  good at it / studies regularly / noticeable dislike
      ±0.25  passionate about it / strong aversion
      ±0.30  professional certification or work experience
    """
    sid = _get_student_id()
    try:
        from preference_service import update_ai_preference, VALID_CATEGORIES
        ignored = [k for k in preferences if k not in VALID_CATEGORIES]
        updated = update_ai_preference(sid, preferences)
        msg = f"Preference profile updated: {updated}"
        if ignored:
            msg += f" (ignored unknown categories: {ignored})"
        return msg
    except Exception as exc:
        return f"Error storing preference: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# ── Comparison tools ─────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@tool
def compare_programs(program_names: List[str]) -> str:
    """
    Aggregates full program data for two or more programs in one call,
    enabling side-by-side comparison.

    Belongs to any request to compare, contrast, or decide between academic
    tracks.

    Do not use for a single-program query (get_program_info). Never
    decompose a comparison into separate per-program calls — this tool does
    it atomically and must be used as a unit. Does not synthesize the
    comparison; the answer LLM generates that from the collected data.
    program_names is a list of full program name strings.
    """
    from neo4j_track_functions import get_program_info as _fn
    results = {}
    for prg in program_names:
        try:
            results[prg] = _fn(prg, course_info=True, desc_info=True)
        except Exception as exc:
            results[prg] = f"Error fetching info for '{prg}': {exc}"
    return _to_str(results)


@tool
def compare_courses(course_names: List[str], program_name: Optional[str] = None) -> str:
    """
    Aggregates course data for two or more courses in one call, enabling side-by-side comparison.
    for two or more courses in one call,
    enabling side-by-side comparison.

    Belongs to any request to compare, contrast, evaluate, or choose between
    courses.

    Do not use for a single-course query (get_course_info +
    get_course_dependencies). Never decompose a comparison into individual
    per-course calls — this tool does it atomically. Does not synthesize the
    comparison; the answer LLM generates that from the collected data.
    course_names is a list of course name strings. program_name optional;
    improves code accuracy for the six courses with program-specific codes.
    """
    from neo4j_course_functions import (
        get_course_info as _info_fn,
        get_course_dependencies as _dep_fn,
    )
    results = {}
    for raw_name in course_names:
        name = _normalize_course(raw_name)
        entry: dict = {}
        try:
            entry["info"] = _info_fn(name, program_name=program_name)
        except Exception as exc:
            entry["info"] = f"Error: {exc}"
        try:
            entry["prerequisites_and_dependents"] = _dep_fn(name, program_name)
        except Exception as exc:
            entry["prerequisites_and_dependents"] = f"Error: {exc}"
        results[name] = entry
    return _to_str(results)


# ─────────────────────────────────────────────────────────────────────────────
# Tool list exported to agent.py
# ─────────────────────────────────────────────────────────────────────────────

ALL_TOOLS = [
    get_student_info,
    get_course_info,
    get_course_dependencies,
    get_course_timing,
    check_course_eligibility,
    get_courses_by_term,
    get_courses_by_multiple_terms,
    get_all_electives,
    get_elective_slots_time_and_occ,
    filter_courses,
    get_program_total_credits,
    answer_academic_question,
    start_course_planning,
    get_program_info,
    get_credit_hour_distribution,
    # Course-category tools
    get_specialized_core_courses,
    get_specialized_elective_courses,
    get_all_specialized_courses,
    get_general_courses,
    get_math_and_basic_science_courses,
    get_basic_computing_sciences_courses,
    get_all_types_courses,
    get_all_core_courses,
    get_all_not_specialized_courses,
    # Comparison tools
    compare_programs,
    compare_courses,
    # Preference storage
    store_preference,
]