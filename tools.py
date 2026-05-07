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
def get_student_info(fields: Optional[List[str]] = None) -> str:
    """
    The sole source for the current student's personal academic record.

    Always pass ONLY the fields relevant to the query — never request
    everything when a subset suffices.

    Available fields:
      "gpa"                — current cumulative GPA (single number)
      "track"              — enrolled program / track name
      "university_year"    — current year level (1–4)
      "total_hours_earned" — total credit hours completed
      "full_name"          — student's full name
      "courses_degrees"    — full course history with grades and percentages
      "completed_courses"  — list of completed course names only (no grades)
      "semester_gpas"      — GPA broken down by semester

    When fields is None or empty, all fields are returned (avoid for simple
    queries — it floods the context with irrelevant data).

    Examples:
      "What is my GPA?"                  → fields=["gpa"]
      "What year am I in?"               → fields=["university_year"]
      "What courses have I completed?"   → fields=["completed_courses"]
      "Show me my grades"                → fields=["courses_degrees"]
      "What's my GPA history?"           → fields=["semester_gpas", "gpa"]
      "Tell me about my profile"         → fields=None  (full profile)
    """
    sid = _get_student_id()
    try:
        from student_functions import get_student_details
        data = get_student_details(sid)
        if data and fields:
            data = {k: v for k, v in data.items() if k in fields}
        return _to_str(data)
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
    student_filters: Optional[List[str]] = None,
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
    student_filters: optional list of student-context filters applied after the
        Neo4j query:
        - "not_completed": exclude courses the student has already completed.
        - "eligible": keep only courses the student is currently eligible for
          (all prerequisites met and credit-hour requirements satisfied).
        Both can be combined, e.g. ["not_completed", "eligible"].
    """
    try:
        from neo4j_course_functions import filter_courses as _fn
        courses = _fn(
            filters=filters,
            course_types=course_types,
            return_fields=return_fields,
            program_name=program_name,
            course_list=course_list,
        )

        if student_filters and courses:
            active_filters = {f.lower() for f in student_filters}
            sid = _get_student_id()

            if active_filters & {'not_completed', 'eligible'}:
                from eligibility import get_student_context
                ctx = get_student_context(sid)
                completed = ctx['completed_courses']

                if 'not_completed' in active_filters:
                    courses = [
                        c for c in courses
                        if c.get('name', '').lower() not in completed
                    ]

                if 'eligible' in active_filters and courses:
                    from neo4j_course_functions import check_course_eligibility as _elig_fn
                    completed_list = list(completed)
                    earned = ctx['total_hours_earned']
                    prog = ctx['program_name']
                    eligible_courses = []
                    for c in courses:
                        name = c.get('name', '').lower()
                        if not name:
                            continue
                        result = _elig_fn(
                            course_name=name,
                            completed_courses=completed_list,
                            earned_credits=earned,
                            program_name=prog,
                        )
                        if result.get('eligible'):
                            eligible_courses.append(c)
                    courses = eligible_courses

        return _to_str(courses)
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
    Build a personalised, fully automated course plan for the student based on
    their academic record, track, GPA, and course preferences.

    Use for requests about semester plans, study schedules, remaining graduation
    requirements, or personalised course sequencing. Not for one-off prerequisite
    checks (get_course_dependencies) or single-course eligibility
    (check_course_eligibility). No parameters; uses the student record automatically.
    """
    sid = _get_student_id()
    try:
        from planning import planning
        from chatbot_connector import ChatbotConnector
        supabase_client = ChatbotConnector().client
        result = planning(sid, supabase_client)
        if not result:
            return "Could not generate a course plan — student record may be incomplete."

        lines = [
            "STUDENT COURSE PLAN",
            f"Year {result['year']} | {result['semester']} Semester | {result['track'].title()}",
            f"Planned Credits: {result['planned_credits']} / {result['available_credits']} available",
        ]

        notes = result.get('advisor_notes', [])
        if notes:
            lines.append("")
            lines.append("Planning notes:")
            for note in notes:
                lines.append(f"  • {note}")

        courses = result.get('planned_courses', [])
        if courses:
            lines.append("")
            lines.append("Recommended courses:")
            for i, course in enumerate(courses, 1):
                name  = course.get('course_name', 'Unknown')
                cr    = course.get('credit_hours', '?')
                ctype = course.get('course_type', 'course')
                code  = course.get('course_code', '')
                tag   = f" [{code}]" if code else ""
                lines.append(f"  {i}. {name}{tag} — {cr} cr ({ctype})")
        else:
            lines.append("")
            lines.append("No courses could be planned for this term.")

        return "\n".join(lines)

    except Exception as exc:
        return f"Could not generate a course plan: {exc}"


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
# ── Recommendation tools ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_program_recommendation() -> str:
    """
    Recommend the best-fitting academic program (AIM / SAD / Data Science)
    for the current student based on their preference profile.

    Combines three preference sources with weighted scoring:
      - degree_preference (45%) — inferred from uploaded transcript grades
      - user_preference   (35%) — filled by student at signup
      - ai_preference     (20%) — inferred by agent during conversation

    Returns a ranked list of all 3 programs with match percentages and the
    key interest areas that drove each score.

    Use this when the student asks:
    - "Which program should I choose?"
    - "AIM or SAD — which suits me better?"
    - "What track fits my interests?"
    - "Recommend a program for me"
    - "Which is better for me — data science or AI?"
    """
    sid = _get_student_id()
    try:
        from recommendation_service import recommend_programs
        return recommend_programs(sid)
    except Exception as exc:
        return f"Error generating program recommendation: {exc}"


@tool
def get_elective_recommendation(top_n: Optional[int] = None) -> str:
    """
    Recommend the best-fitting elective courses for the current student
    within their program, based on their preference profile.

    Scores every elective in the programme catalogue using the student's
    merged preference vector (degree + user + ai preferences) and returns
    the top 5 matches with match percentages and matching interest areas.
    If top_n is set, returns only that many top matches instead of the default 5.

    Use this when the student asks:
    - "Which electives should I take?"
    - "What electives fit my interests?"
    - "Recommend electives for me"
    - "Which optional courses suit me best?"
    """
    sid = _get_student_id()
    try:
        from eligibility import get_student_context
        from recommendation_service import recommend_electives
        program_name = get_student_context(sid).get("program_name")
        if not program_name:
            return "Could not determine your program. Please contact the registrar."
        if top_n is not None:
            return recommend_electives(sid, program_name, top_n=top_n)
        return recommend_electives(sid, program_name)
    except Exception as exc:
        return f"Error generating elective recommendation: {exc}"


def _recommend_core(
    sid: str,
    course_names: Optional[List[str]] = None,
    top_n: Optional[int] = None,
) -> str:
    try:
        from eligibility import check_course_eligibility as _check_elig, get_student_context
        from neo4j_course_functions import (
            get_courses_by_term as _term_fn,
            get_course_closes as _closes_fn,
            get_course_info as _info_fn,
            normalize_level,
            normalize_semester,
        )

        ctx = get_student_context(sid)
        program_name = ctx.get("program_name")
        university_year = ctx.get("university_year")
        current_term = ctx.get("current_term") or 1  # 1 or 2 from DB, default to 1

        if not program_name:
            return "Could not determine your program. Please contact the registrar."
        if not university_year:
            return "Could not determine your current academic year. Please contact the registrar."

        current_year_str = normalize_level(university_year)
        current_sem_str = normalize_semester(current_term)   # "First" or "Second"
        current_term_label = f"{current_year_str}, {current_sem_str} Semester"

        # ── helpers ──────────────────────────────────────────────────────────

        def _get_info(name: str) -> dict:
            try:
                info_list = _info_fn(name, program_name=program_name)
                return info_list[0] if info_list else {}
            except Exception:
                return {}

        def _get_closes_count(name: str) -> int:
            try:
                return len(_closes_fn(name, program_name=program_name))
            except Exception:
                return 0

        def _elig(name: str) -> dict:
            try:
                return _check_elig(sid, name)
            except Exception:
                return {"eligible": False, "missing_prerequisites": [], "credit_requirement_met": True}

        def _missing_str(elig_result: dict) -> list:
            parts = [p.get("name", "?") for p in elig_result.get("missing_prerequisites", [])]
            if not elig_result.get("credit_requirement_met", True):
                cr = elig_result.get("credit_requirement")
                earned = elig_result.get("earned_credits", 0)
                parts.append(f"need {cr} credit hours (have {earned})")
            return parts

        def _technical_score(course_code) -> int:
            """0 for general-track (GEN prefix), 1 for all technical courses.
            Used as a tiebreaker so technical courses rank above general ones
            when they share the same closes count."""
            if not course_code:
                return 1
            return 0 if str(course_code).upper().startswith("GEN") else 1

        def _sort_key(c):
            return (c["closes_courses_count"], _technical_score(c.get("course_code", "")))

        # ── MODE 1 ───────────────────────────────────────────────────────────
        if not course_names:
            # Fetch the full year (both semesters) so we can split and note
            # other-semester courses instead of silently dropping them.
            year_data = _term_fn(university_year, semester=None, program_name=program_name)

            current_term_mandatory = []   # courses in student's current semester
            other_term_mandatory = []     # mandatory courses scheduled for the other semester

            for _yr, semesters in year_data.items():
                for sem, programs in semesters.items():
                    for _prog, courses in programs.items():
                        for c in courses:
                            if c["course_type"] == "mandatory":
                                entry = {
                                    "course_name": c["course_name"],
                                    "course_code": c["course_code"],
                                    "credit_hours": c["credit_hours"],
                                    "semester": sem,
                                }
                                if sem == current_sem_str:
                                    current_term_mandatory.append(entry)
                                else:
                                    other_term_mandatory.append(entry)

            if not current_term_mandatory and not other_term_mandatory:
                return f"No mandatory courses found for {program_name} in {current_year_str}."

            eligible_courses = []
            not_eligible_list = []

            # Check eligibility only for current-semester courses
            for course in current_term_mandatory:
                er = _elig(course["course_name"])
                if er["eligible"]:
                    eligible_courses.append(dict(course))
                else:
                    not_eligible_list.append({
                        "course_name": course["course_name"],
                        "missing_prerequisites": _missing_str(er),
                    })

            # Enrich eligible courses with description + closes count
            for course in eligible_courses:
                info = _get_info(course["course_name"])
                course["description"] = info.get("description") or ""
                course["motivation"] = info.get("motivation") or ""
                course["closes_courses_count"] = _get_closes_count(course["course_name"])

            # Sort by closes count desc; technical courses break ties over general ones
            eligible_courses.sort(key=_sort_key, reverse=True)

            def _ne_note(entries):
                if not entries:
                    return "none"
                parts = []
                for ne in entries:
                    m = ", ".join(ne["missing_prerequisites"]) if ne["missing_prerequisites"] else "unmet requirements"
                    parts.append(f"{ne['course_name'].title()} (missing: {m})")
                return "; ".join(parts)

            not_elig_note = _ne_note(not_eligible_list)

            # Other-semester note
            other_term_note = (
                "; ".join(c["course_name"].title() for c in other_term_mandatory)
                if other_term_mandatory else "none"
            )
            other_sem_str = "Second" if current_sem_str == "First" else "First"

            def _closes_label(c):
                n = c["closes_courses_count"]
                return f"{c['course_name'].title()} (unlocks {n} future course{'s' if n != 1 else ''})"

            # ── Mode 1a: return all eligible with closes counts ───────────────
            if top_n is None or top_n >= len(eligible_courses):
                output_data = {
                    "student_year": current_year_str,
                    "current_semester": current_sem_str,
                    "program": program_name,
                    "eligible_core_courses": eligible_courses,
                    "not_eligible_courses": not_eligible_list,
                    "not_current_term_courses": [c["course_name"] for c in other_term_mandatory],
                }
                output = _to_str(output_data)
                output += (
                    f"\n\n[ADVISOR NOTE] Present to the student: "
                    f"These are all the eligible mandatory (core) courses in your current term ({current_term_label}), "
                    f"listed with how many future courses each one unlocks. "
                    f"List each eligible course with its name, semester, credit hours, description, and closes count. "
                    f"Then guide the student: 'If you can take all of them this term, go ahead! "
                    f"But if you need to choose only some, prioritize the ones that unlock the most future "
                    f"courses in coming years and semesters — they give you the most academic momentum. "
                    f"When two courses unlock the same number of future courses, prefer the more technical one.' "
                    f"Then inform: 'These mandatory courses are not in your current term ({other_sem_str} Semester) — "
                    f"you will take them next: {other_term_note}.' "
                    f"Finally inform: 'You cannot enroll in these courses yet: {not_elig_note}.' "
                    f"Explain why each blocked course is blocked (which prerequisites are missing)."
                )

            # ── Mode 1b: top_n by closes count ───────────────────────────────
            else:
                top_courses = eligible_courses[:top_n]
                remaining = eligible_courses[top_n:]

                top_note = ", ".join(_closes_label(c) for c in top_courses)
                remaining_note = (
                    "; ".join(_closes_label(c) for c in remaining) if remaining else "none"
                )

                output_data = {
                    "student_year": current_year_str,
                    "current_semester": current_sem_str,
                    "program": program_name,
                    "recommended_core_courses": top_courses,
                    "other_eligible_not_recommended": remaining,
                    "not_eligible_courses": not_eligible_list,
                    "not_current_term_courses": [c["course_name"] for c in other_term_mandatory],
                }
                output = _to_str(output_data)
                output += (
                    f"\n\n[ADVISOR NOTE] Present to the student: "
                    f"These are the top {top_n} recommended mandatory (core) courses for your current term "
                    f"({current_term_label}). They are recommended because you are eligible for them "
                    f"AND they unlock the most future courses in coming years and semesters: {top_note}. "
                    f"When courses tied on future unlocks, more technical courses were preferred over general-track ones. "
                    f"Then say: 'I do not recommend the remaining eligible courses as strongly: "
                    f"{remaining_note} — each of them unlocks fewer future courses than the ones I recommended.' "
                    f"Then inform: 'These mandatory courses are not in your current term ({other_sem_str} Semester) — "
                    f"you will take them next: {other_term_note}.' "
                    f"Finally say: 'You cannot enroll in these courses yet: {not_elig_note}.'"
                )

            return output

        # ── MODE 2 ───────────────────────────────────────────────────────────
        else:
            advisor_note_parts = []
            normalized = [_normalize_course(n) for n in course_names]

            # ── Step 0: filter out courses not in student's program ───────────
            in_program_names: List[str] = []
            not_in_program_names: List[str] = []
            course_info_cache: Dict[str, dict] = {}

            for name in normalized:
                info = _get_info(name)
                course_info_cache[name] = info
                in_prog = any(
                    (off.get("program") or "").lower() == program_name.lower()
                    for off in info.get("program_offerings", [])
                )
                if in_prog:
                    in_program_names.append(name)
                else:
                    not_in_program_names.append(name)

            if not_in_program_names:
                not_prog_str = ", ".join(n.title() for n in not_in_program_names)
                advisor_note_parts.append(
                    f"These courses are not in your program ({program_name.title()}): {not_prog_str}"
                )

            # ── Step 1: separate current-year vs other-year (in-program only) ─
            course_term_map: Dict[str, dict] = {}
            for name in in_program_names:
                info = course_info_cache[name]
                year_found = None
                sem_found = None
                for off in info.get("program_offerings", []):
                    if (off.get("program") or "").lower() == program_name.lower():
                        year_found = off.get("year")
                        sem_found = off.get("semester")
                        break
                course_term_map[name] = {
                    "year": year_found,
                    "semester": sem_found,
                    "info": info,
                }

            current_term_courses: List[str] = []
            not_current_term_courses: List[str] = []

            current_sem_str = normalize_semester(current_term) if current_term else None

            for name, data in course_term_map.items():
                year_match = data["year"] == current_year_str
                sem_match = (current_sem_str is None) or (data["semester"] == current_sem_str)
                if year_match and sem_match:
                    current_term_courses.append(name)
                else:
                    not_current_term_courses.append(name)

            if not_current_term_courses:
                year_labels = []
                for n in not_current_term_courses:
                    yr = course_term_map[n]["year"] or "unknown year"
                    sem = course_term_map[n]["semester"]
                    label = f"{n.title()} ({yr}{', Semester: ' + sem if sem else ''})"
                    year_labels.append(label)
                advisor_note_parts.append(
                    f"These courses are not in your current term ({current_term_label}): "
                    + "; ".join(year_labels)
                )

            # ── Step 2: check eligibility for current-term courses ────────────
            eligible_current: List[str] = []
            not_eligible_current: List[dict] = []

            for name in current_term_courses:
                er = _elig(name)
                if er["eligible"]:
                    eligible_current.append(name)
                else:
                    not_eligible_current.append({
                        "course_name": name,
                        "missing_prerequisites": _missing_str(er),
                    })

            if not_eligible_current:
                parts = []
                for ne in not_eligible_current:
                    m = ", ".join(ne["missing_prerequisites"]) if ne["missing_prerequisites"] else "unmet requirements"
                    parts.append(f"{ne['course_name'].title()} (missing: {m})")
                advisor_note_parts.append(
                    "You are not eligible to enroll in these current-term courses: "
                    + "; ".join(parts)
                )

            # Build base output data
            output_data: dict = {
                "student_year": current_year_str,
                "program": program_name,
                "not_in_program_courses": not_in_program_names,
                "not_current_term_courses": [
                    {
                        "course_name": n,
                        "year": course_term_map[n]["year"],
                        "semester": course_term_map[n]["semester"],
                    }
                    for n in not_current_term_courses
                ],
                "not_eligible_courses": not_eligible_current,
            }

            if not eligible_current:
                output_data["eligible_current_term_courses"] = []
                output = _to_str(output_data)
                full_note = " | ".join(advisor_note_parts) if advisor_note_parts else ""
                output += (
                    f"\n\n[ADVISOR NOTE] {full_note} "
                    f"None of the provided courses are both in your current term and eligible for enrollment."
                )
                return output

            # ── Step 3: get closes counts, sort, build recommendation ─────────
            eligible_with_data = []
            for name in eligible_current:
                info = course_term_map[name]["info"]
                eligible_with_data.append({
                    "course_name": name,
                    "course_code": info.get("course_code", ""),
                    "credit_hours": info.get("credit_hours"),
                    "description": info.get("description") or "",
                    "motivation": info.get("motivation") or "",
                    "semester": course_term_map[name]["semester"] or "",
                    "closes_courses_count": _get_closes_count(name),
                })

            eligible_with_data.sort(key=_sort_key, reverse=True)

            best = eligible_with_data[0]
            rest = eligible_with_data[1:]

            best_label = (
                f"{best['course_name'].title()} "
                f"(unlocks {best['closes_courses_count']} future course"
                f"{'s' if best['closes_courses_count'] != 1 else ''})"
            )
            rest_labels = ", ".join(
                f"{r['course_name'].title()} "
                f"(unlocks {r['closes_courses_count']} future course"
                f"{'s' if r['closes_courses_count'] != 1 else ''})"
                for r in rest
            )

            tiebreaker_note = (
                " When courses tied on future unlocks, more technical courses were preferred over general-track ones."
            )
            advisor_note_parts.append(
                f"I recommend {best_label} because it unlocks the most future courses "
                f"among your eligible current-term options.{tiebreaker_note}"
                + (
                    f" Not recommended as strongly: {rest_labels} — "
                    f"each unlocks fewer future courses than the one I recommended."
                    if rest else ""
                )
            )

            output_data["eligible_current_term_courses_sorted_by_impact"] = eligible_with_data
            output = _to_str(output_data)
            output += "\n\n[ADVISOR NOTE] " + " | ".join(advisor_note_parts)
            return output

    except Exception as exc:
        return f"Error in recommend_core: {exc}"


@tool
def recommend_core(top_n: Optional[int] = None) -> str:
    """
    Recommends mandatory (core) courses for the current student based on
    eligibility and how many future courses each one unlocks.

    Scans ALL mandatory courses in the student's current academic year,
    checks eligibility for each, and returns eligible ones with descriptions.
    If top_n is set and smaller than the eligible count, sorts by how many
    future courses each eligible course unlocks and returns only the top_n
    highest-impact picks.

    Use this when the student asks:
    - "What core courses should I take this term?"
    - "Which mandatory courses can I register for now?"
    - "Recommend the most important core courses for me"
    - "What's the best core course to take next?"
    """
    sid = _get_student_id()
    return _recommend_core(sid, top_n=top_n)


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
    shared_note = None
    for prg in program_names:
        try:
            data = _fn(prg, course_info=True, desc_info=True)
            if isinstance(data, dict):
                shared_note = shared_note or data.pop("shared_courses_note", None)
            results[prg] = data
        except Exception as exc:
            results[prg] = f"Error fetching info for '{prg}': {exc}"

    output_data = results
    if shared_note:
        output_data = {"shared_courses_note": shared_note, **results}

    output = _to_str(output_data)

    sid = _get_student_id()
    student_prog = None
    try:
        from eligibility import get_student_context
        student_prog = get_student_context(sid).get("program_name")
    except Exception:
        pass

    # Presentation instruction — always appended regardless of specialization status.
    output += (
        "\n\n[ADVISOR NOTE — RESPONSE STYLE] You are explaining these programs to a student "
        "who may have no prior knowledge of what each track involves. Be thorough and clear: "
        "list ALL courses in every category (humanities, math/science, basic computing, specialized core, electives) "
        "by name — do not summarise or truncate; "
        "explain what each category means in plain terms; "
        "and describe what studying each program actually looks like in practice, not just its label."
    )

    if not student_prog:
        # Student not yet specialized (Year 1/2) — recommend a program based on their profile.
        try:
            from recommendation_service import recommend_programs
            rec = recommend_programs(sid)
            output += "\n\n" + rec
        except Exception:
            pass
        output += (
            "\n\n[ADVISOR NOTE] This student has not yet chosen a specialization. "
            "Use the program recommendation above (if available) to suggest the best-fit "
            "track based on their preference profile. Encourage them to share their "
            "interests if no preference data exists yet."
        )
    else:
        output += (
            f"\n\n[ADVISOR NOTE] This student is already enrolled in the "
            f"{student_prog.title()} program. Do not recommend switching programs. "
            f"Ask if they would like to know more about their program's courses, "
            f"electives, prerequisites, or schedule."
        )

    return output


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

    output = _to_str(results)

    # Classify compared courses by type.
    all_electives = [
        name for name, entry in results.items()
        if isinstance(entry.get("info"), list)
        and any(
            off.get("course_type") == "yes"
            for row in entry["info"]
            for off in row.get("program_offerings", [])
        )
    ]
    all_cores = [
        name for name, entry in results.items()
        if isinstance(entry.get("info"), list)
        and any(
            off.get("course_type") == "no"
            for row in entry["info"]
            for off in row.get("program_offerings", [])
        )
    ]

    if len(all_electives) >= 2 or len(all_cores) >= 2:
        sid = _get_student_id()
        prog = program_name
        if not prog:
            try:
                from eligibility import get_student_context
                prog = get_student_context(sid).get("program_name")
            except Exception:
                prog = None

        if prog:
            prog_lower = prog.lower().strip()

            # ── Elective recommendation ───────────────────────────────────────
            if len(all_electives) >= 2:
                in_prog_elective = [
                    name for name in all_electives
                    if any(
                        off.get("course_type") == "yes"
                        and (off.get("program") or "").lower().strip() == prog_lower
                        for row in results[name]["info"]
                        for off in row.get("program_offerings", [])
                    )
                ]
                out_of_prog_elective = [n for n in all_electives if n not in in_prog_elective]

                if len(in_prog_elective) >= 2:
                    rec = ""
                    try:
                        from recommendation_service import recommend_electives
                        rec = recommend_electives(sid, prog, course_names=in_prog_elective, skip_course_info=True)
                        output += "\n\n" + rec
                    except Exception:
                        pass
                    if out_of_prog_elective:
                        out_t = ", ".join(n.title() for n in out_of_prog_elective)
                        in_t  = ", ".join(n.title() for n in in_prog_elective)
                        covers_suffix = (
                            f" The elective recommendation above covers only: {in_t}."
                            if rec and "No preference data" not in rec
                            else ""
                        )
                        output += (
                            f"\n\n[ADVISOR NOTE] {out_t} "
                            f"{'is' if len(out_of_prog_elective) == 1 else 'are'} not offered as an elective in the student's program "
                            f"({prog.title()}); do not recommend {'it' if len(out_of_prog_elective) == 1 else 'them'}."
                            f"{covers_suffix}"
                        )
                elif len(in_prog_elective) == 1:
                    in_t  = in_prog_elective[0].title()
                    out_t = ", ".join(n.title() for n in out_of_prog_elective)
                    elig_suffix = ""
                    try:
                        from recommendation_service import _eligibility_for
                        elig = _eligibility_for(sid, in_prog_elective[0])
                        if elig.get("eligible") is True:
                            elig_suffix = (
                                f" The student has completed all prerequisites for {in_t} "
                                f"and can enroll in it."
                            )
                        elif elig.get("eligible") is False:
                            missing = [p.get("name", "?") for p in elig.get("missing_prerequisites", [])]
                            credit_req = elig.get("credit_requirement")
                            credit_met = elig.get("credit_requirement_met", True)
                            earned = elig.get("earned_credits", "?")
                            reasons = []
                            if missing:
                                reasons.append(f"has not completed the required prerequisites: {', '.join(missing)}")
                            if not credit_met:
                                reasons.append(f"needs {credit_req} credit hours but only has {earned}")
                            reasons_str = " and ".join(reasons) if reasons else "prerequisites are not met"
                            elig_suffix = (
                                f" However, the student cannot enroll in {in_t} yet "
                                f"because the student {reasons_str}."
                            )
                    except Exception:
                        pass
                    output += (
                        f"\n\n[ADVISOR NOTE] Only {in_t} is an elective in the student's program "
                        f"({prog.title()}). "
                        f"{out_t} {'is' if len(out_of_prog_elective) == 1 else 'are'} not offered as an elective in their program. "
                        f"Inform the student that {in_t} is the only elective from these courses that belongs to their program. "
                        f"{elig_suffix}".rstrip()
                    )
                else:  # 0 electives in student's program
                    all_t = ", ".join(n.title() for n in all_electives)
                    output += (
                        f"\n\n[ADVISOR NOTE] None of the compared courses ({all_t}) "
                        f"are offered as electives in the student's program ({prog.title()}). "
                        f"Do not recommend any of them; inform the student these courses "
                        f"are not available as electives in their program."
                    )

            # ── Core recommendation ───────────────────────────────────────────
            if len(all_cores) >= 2:
                in_prog_core = [
                    name for name in all_cores
                    if any(
                        off.get("course_type") == "no"
                        and (off.get("program") or "").lower().strip() == prog_lower
                        for row in results[name]["info"]
                        for off in row.get("program_offerings", [])
                    )
                ]
                out_of_prog_core = [n for n in all_cores if n not in in_prog_core]

                if len(in_prog_core) >= 2:
                    rec = ""
                    try:
                        rec = _recommend_core(sid, course_names=in_prog_core)
                        output += "\n\n" + rec
                    except Exception:
                        pass
                    if out_of_prog_core:
                        out_t = ", ".join(n.title() for n in out_of_prog_core)
                        in_t  = ", ".join(n.title() for n in in_prog_core)
                        covers_suffix = (
                            f" The core recommendation above covers only: {in_t}."
                            if "eligible_current_term_courses_sorted_by_impact" in rec
                            else ""
                        )
                        output += (
                            f"\n\n[ADVISOR NOTE] {out_t} "
                            f"{'is' if len(out_of_prog_core) == 1 else 'are'} not offered as a core course in the student's program "
                            f"({prog.title()}); do not recommend {'it' if len(out_of_prog_core) == 1 else 'them'}."
                            f"{covers_suffix}"
                        )
                elif len(in_prog_core) == 1:
                    in_t  = in_prog_core[0].title()
                    out_t = ", ".join(n.title() for n in out_of_prog_core)
                    elig_suffix = ""
                    try:
                        from recommendation_service import _eligibility_for
                        elig = _eligibility_for(sid, in_prog_core[0])
                        if elig.get("eligible") is True:
                            elig_suffix = (
                                f" The student has completed all prerequisites for {in_t} "
                                f"and can enroll in it."
                            )
                        elif elig.get("eligible") is False:
                            missing = [p.get("name", "?") for p in elig.get("missing_prerequisites", [])]
                            credit_req = elig.get("credit_requirement")
                            credit_met = elig.get("credit_requirement_met", True)
                            earned = elig.get("earned_credits", "?")
                            reasons = []
                            if missing:
                                reasons.append(f"has not completed the required prerequisites: {', '.join(missing)}")
                            if not credit_met:
                                reasons.append(f"needs {credit_req} credit hours but only has {earned}")
                            reasons_str = " and ".join(reasons) if reasons else "prerequisites are not met"
                            elig_suffix = (
                                f" However, the student cannot enroll in {in_t} yet "
                                f"because the student {reasons_str}."
                            )
                    except Exception:
                        pass
                    output += (
                        f"\n\n[ADVISOR NOTE] Only {in_t} is a core course in the student's program "
                        f"({prog.title()}). "
                        f"{out_t} {'is' if len(out_of_prog_core) == 1 else 'are'} not offered as a core course in their program. "
                        f"Inform the student that {in_t} is the only core course from these compared courses that belongs to their program. "
                        f"{elig_suffix}".rstrip()
                    )
                else:  # 0 core courses in student's program
                    all_t = ", ".join(n.title() for n in all_cores)
                    output += (
                        f"\n\n[ADVISOR NOTE] None of the compared courses ({all_t}) "
                        f"are offered as core courses in the student's program ({prog.title()}). "
                        f"Do not recommend any of them; inform the student these courses "
                        f"are not available as core courses in their program."
                    )

            if len(all_electives) >= 2 and len(all_cores) >= 2:
                output += (
                    f"\n\n[ADVISOR NOTE] The output above contains two independent recommendations. "
                    f"Present them to the student as two clearly separated sections: "
                    f"(1) Elective recommendation — based on preference fit. "
                    f"(2) Core recommendation — based on eligibility and how many future courses each unlocks. "
                    f"Do not merge them into a single list."
                )

    return output


# ─────────────────────────────────────────────────────────────────────────────
# ── GPA tools ────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@tool
def calculate_projected_gpa(new_courses: List[dict]) -> str:
    """
    Projects the student's GPA if they score specific percentages in new courses.

    Belongs to hypothetical "what if" GPA questions:
    "If I get 85 in machine learning and 90 in networks, what will my GPA be?"

    new_courses must be a list of objects, each with:
      - "name":       exact course name (string)
      - "percentage": numeric score 0–100

    Example: [{"name": "machine learning", "percentage": 85},
              {"name": "computer networks", "percentage": 90}]

    Credit hours for each course are looked up automatically from the knowledge
    graph. Does not require any other parameters.
    """
    sid = _get_student_id()
    try:
        from gpa_service import project_gpa_with_new_courses as _fn
        return _to_str(_fn(sid, new_courses))
    except Exception as exc:
        return f"Error projecting GPA: {exc}"


@tool
def calculate_target_gpa(
    target_gpa: float,
    optimization: str = "maximize_credits",
) -> str:
    """
    Analyzes what combination of courses and minimum grades per credit type
    are needed to reach a target GPA.

    Belongs to goal-oriented GPA questions:
    "I want my GPA to be 3.2", "How do I increase my GPA to 3.5?",
    "What do I need to improve my GPA?"

    target_gpa: the desired cumulative GPA (0.0–4.0).

    optimization controls the course-count strategy:
      - "maximize_credits" (default): fill the available credit limit with as
        many eligible courses as possible — maximizes GPA leverage.
        Use when the student does not restrict course count.
      - "minimize_grade": use the minimum number of credits needed to make
        the target achievable — fewer courses but higher grade requirements.
        Use when the student says "I don't want too many courses" or similar.

    Returns:
      - The recommended combination (n1 one-credit, n2 two-credit, n3 three-credit)
      - Per-credit-type minimum grade (floor), where ALL types must be met together
      - Whether the target is achievable; if not, the maximum possible GPA
      - Advisor notes with follow-up options
    """
    sid = _get_student_id()
    try:
        from gpa_service import analyze_target_gpa as _fn
        return _to_str(_fn(sid, target_gpa, optimization))
    except Exception as exc:
        return f"Error analyzing target GPA: {exc}"


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
    # Recommendation tools
    get_program_recommendation,
    get_elective_recommendation,
    recommend_core,
    # Preference storage
    store_preference,
    # GPA tools
    calculate_projected_gpa,
    calculate_target_gpa,
]