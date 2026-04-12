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
    Retrieve the student's complete academic profile from the database.
    Returns: full name, program/track, university year, GPA, total earned
    credits, and the complete list of completed courses.

    Use this when the student asks about:
    - Their GPA or academic standing
    - How many credits they have earned
    - Which courses they have already completed
    - Their current program or track
    - Any question about their personal academic status
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
def get_course_info(course_name: str) -> str:
    """
    Get general information about a specific course.
    Returns: full course name, description, credit hours, and course type
    (core / elective).

    Use this when the student wants to know what a course is about,
    how many credits it gives, or whether it is core or elective.
    Examples: "What is Machine Learning about?", "How many credits is SE?"

    Args:
        course_name: Course name, abbreviation, or partial name. Fuzzy
                     matching is applied automatically (e.g. "ml", "soft eng").
    """
    try:
        from neo4j_course_functions import get_course_info as _fn
        return _to_str(_fn(_normalize_course(course_name)))
    except Exception as exc:
        return f"Error fetching course info: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# ── Prerequisites ────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_course_prerequisites(
    course_name: str,
    program_name: Optional[str] = None,
) -> str:
    """
    Get BOTH the prerequisites for a course AND the courses that depend on it.

    Returns TWO sections:
    - "prerequisites": courses the student must complete BEFORE enrolling in
      this course (what the student needs first).
    - "dependents": courses that REQUIRE this course as a prerequisite — i.e.,
      what this course UNLOCKS, CLOSES, or ENABLES for the student.

    Use this for ANY of these question types:
    - "What do I need before taking X?" (prerequisites section)
    - "What are the prerequisites for X?" (prerequisites section)
    - "What does X close / unlock / enable?" (dependents section)
    - "What courses require X?" (dependents section)
    - "What comes after completing X?" (dependents section)
    - "What closes if I complete X?" (dependents section)
    - "What courses does X lead to?" (dependents section)
    - "What is X a prerequisite for?" (dependents section)

    Args:
        course_name:  Course name or abbreviation. Fuzzy matching is applied.
        program_name: Student's program/track (optional). Improves accuracy.
                      E.g. "artificial intelligence & machine learning",
                      "software & application development", "data science".
    """
    try:
        from neo4j_course_functions import get_course_dependencies
        return _to_str(
            get_course_dependencies(_normalize_course(course_name), program_name)
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
    Find out which academic year level and semester a course is offered in.
    Returns: year level (1–4) and semester (1 or 2) in the curriculum.

    Use this for questions like:
    - "When is Machine Learning taught?"
    - "Which semester is Operating Systems?"
    - "What year do students take Data Structures?"

    Args:
        course_name:  Course name or abbreviation. Fuzzy matching is applied.
        program_name: Student's program/track (optional).
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
    Check whether the current student is eligible to enroll in a specific
    course right now.

    Checks:
    - Whether the course belongs to the student's program
    - Whether all prerequisite courses have been completed
    - Whether any credit-hour requirements are met

    Returns a clear eligible / not-eligible result with:
    - The eligibility verdict (true/false)
    - List of any missing prerequisite courses
    - Credit-hour requirement status if applicable

    Use this for questions like:
    - "Can I take Machine Learning this semester?"
    - "Am I eligible for Operating Systems?"
    - "Can I register for Advanced AI?"
    - "Do I have the prerequisites for Deep Learning?"

    Args:
        course_name: Course name or abbreviation. Fuzzy matching is applied.
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
    Get all courses scheduled in a specific academic term.
    Returns: list of courses in that year/semester with names, codes, credits.

    Use this for questions like:
    - "What courses are in year 2 semester 1?"
    - "Show me the third-year second-semester curriculum"
    - "What do first-year students study?"

    Args:
        level:        Academic year: 1, 2, 3, or 4.
        semester:     1 (first/Fall) or 2 (second/Spring).
        program_name: Student's program/track (optional).
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
    Get courses for several academic terms in a single call.

    Use this INSTEAD of calling get_courses_by_term multiple times when the
    student asks about courses spanning more than one semester.

    Example: "What do I study in years 2 and 3?" → call with:
    terms = [{"level": 2, "semester": 1}, {"level": 2, "semester": 2},
             {"level": 3, "semester": 1}, {"level": 3, "semester": 2}]

    Args:
        terms:        List of {"level": int, "semester": int} objects.
        program_name: Student's program/track (optional).
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
    List all available elective courses for a given program.
    Returns: full list of elective options with names, codes, and credit hours.

    Use this when the student asks:
    - "What electives are available in my program?"
    - "Show me elective options for the AI track"
    - "What are my elective choices?"

    Args:
        program_name: The student's program/track. E.g. "artificial
                      intelligence & machine learning", "software &
                      application development", "data science".
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
def get_elective_slots(program_name: str) -> str:
    """
    Get the elective slot schedule for a program: which year/semester
    students can take electives and how many slots are available.

    Use this for questions like:
    - "When can I take electives?"
    - "How many elective slots do I have?"
    - "In which semesters are elective courses offered?"

    Args:
        program_name: The student's program/track.
    """
    try:
        from neo4j_course_functions import get_elective_slots_time
        return _to_str(get_elective_slots_time(program_name))
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
    Search and filter courses using multiple criteria simultaneously.
    Use this for complex or exploratory course search queries, such as:
    - "Show me all 3-credit courses in my program"
    - "What core courses are in the AI program?"
    - "List all electives with 2 credit hours"

    Args:
        program_name:  Student's program/track (optional).
        filters:       Dict of criteria, e.g. {"credit_hours": 3} (optional).
        course_types:  Types to include, e.g. ["elective"] or ["core"] (opt.).
        return_fields: Fields to include in the result (optional).
        course_list:   Specific course names to filter from (optional).
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
    Return the total number of credit hours required to graduate from a
    specific program (track).

    Use this when the student asks:
    - "How many credits do I need to graduate?"
    - "What is the total credit requirement for the AI track?"
    - "How many credit hours are required in my program?"
    - "What are the total credits needed for data science?"

    Args:
        program_name: The program/track name. E.g. "artificial intelligence
                      & machine learning", "software & application
                      development", "data science".
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
    Search the BNU bylaws and academic regulations knowledge base.
    Covers: graduation requirements, grading/GPA policies, academic probation,
    attendance rules, credit transfer, registration/withdrawal, disciplinary
    policies, academic warnings, leave of absence, and all official BNU rules.

    Use this for ANY question about BNU policies or academic regulations:
    - "What is the minimum GPA to avoid probation?"
    - "How many absences are allowed per course?"
    - "What are the graduation requirements?"
    - "Can I withdraw from a course after the add/drop period?"
    - "What happens if I fail a course twice?"

    Args:
        question: The specific policy or regulation question to look up.
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
    Start a personalised multi-turn course planning session for the student.
    Builds a tailored study plan based on completed courses, GPA, earned
    credits, and remaining graduation requirements.

    Use this when the student asks:
    - "Help me plan my courses"
    - "What should I take next semester?"
    - "Make a study plan for me"
    - "What courses do I still need to graduate?"
    - "Give me a recommended course schedule"

    After calling this tool, the student enters a multi-turn planning session
    and should answer the planner's follow-up prompts.
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
    Get comprehensive information about a specific academic program/track.

    Returns any combination of:
    - Program description (from Neo4j)
    - Credit hour distribution (always included — shared across all programs)
    - Full curriculum for years 3 and 4 (core + elective courses)
    - Hardcoded year-1/2 courses that differ between tracks
    - Elective slot schedule (which year/semester electives are offered)
    - Full elective course catalogue

    Use this for questions like:
    - "What is the AI / SAD / Data Science program?"
    - "Tell me about the artificial intelligence track"
    - "Compare the AI and SAD programs" (call once per program)

    Args:
        prg:         Program name 'full names'.
        course_info: Set False to skip curriculum/elective data (e.g. when
                     only the description is needed).
        desc_info:   Set False to skip the program description.
    """
    try:
        from neo4j_track_functions import get_program_info as _fn
        return _to_str(_fn(prg, course_info=course_info, desc_info=desc_info))
    except Exception as exc:
        return f"Error fetching program info: {exc}"


@tool
def get_credit_hour_distribution() -> str:
    """
    Return the faculty-wide credit hour distribution shared by ALL programs.

    This breakdown is identical across all three tracks (AIM, SAD, Data Science)
    and should be called ONCE — not once per program.

    Use this for questions like:
    - "How are credit hours distributed?"
    - "How many humanities credits are required?"
    - "What is the breakdown of the 136 credit hours?"
    - "How many credits are for field training / graduation project?"
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
    Return the specialized **core** (non-elective) courses for a program.

    These are courses whose code starts with the program's prefix
    (AIM / SAD / DAS) and are mandatory (not elective).
    They correspond to the "applied / specialized courses" (51 cr) category.

    Use this when the student asks:
    - "What specialized courses are NOT elective in the SAD program?"
    - "What mandatory specialized courses does AIM have?"
    - "List the core courses specific to the data science track"

    Args:
        prg:       Program name. E.g. "artificial intelligence & machine learning",
                   "software & application development", "data science" (or aliases
                   like "AIM", "SAD", "DS").
        year_flag: Set True to include the year level for each course.
        sem_flag:  Set True to include the semester for each course.
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
    Return the specialized **elective** courses available in a program.

    These are courses whose code starts with the program's prefix
    (AIM / SAD / DAS) and are elective (students choose some of them).
    When year_flag or sem_flag is True the elective *slot schedule* is
    returned (the semesters when students may fill elective slots), since
    individual elective courses have no fixed placement in the curriculum.

    Use this when the student asks:
    - "What electives does the AI track offer?"
    - "What program-specific electives are in data science?"
    - "Show me elective options for SAD"

    Args:
        prg:       Program name or alias.
        year_flag: Set True to include the elective slot year schedule.
        sem_flag:  Set True to include the elective slot semester schedule.
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
    Return ALL specialized courses (core + elective) for a program.

    These are courses unique to that track, identified by the program code
    prefix (AIM / SAD / DAS).  Combines both mandatory and elective courses.
    The applied/specialized category totals 51 credits.

    Use this when the student asks:
    - "What courses are special to the data science program?"
    - "What courses are in the data science program but not in others?"
    - "What specialised courses does SAD offer?"
    - "What courses differ between AIM and SAD?" (call once per program)
    - "What specialised courses are in year 3 of SAD?" (return all, LLM filters by year)

    Args:
        prg:       Program name or alias.
        year_flag: Include year for core courses; elective slot years for electives.
        sem_flag:  Include semester for core courses; elective slot semesters for electives.
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
    Return the humanities / general education courses (code prefix: GEN).

    These courses are **identical across all three programs** and total 12 credits.

    Use this when the student asks:
    - "What general courses are there?"
    - "What humanities courses do I take?"
    - "What GEN courses are in the curriculum?"
    - "What general courses are in the data science program?" (same for all)

    Args:
        year_flag: Set True to include the year level for each course.
        sem_flag:  Set True to include the semester for each course.
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
    Return the mathematics and basic sciences courses (code prefix: BAS).

    These courses are **identical across all three programs** and total 24 credits.

    Use this when the student asks:
    - "What math courses are there?"
    - "What mathematics and basic science courses do I study?"
    - "Show me BAS courses"
    - "What math courses are in the data science program?" (same for all)

    Args:
        year_flag: Set True to include the year level for each course.
        sem_flag:  Set True to include the semester for each course.
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
    Return the basic computing sciences courses (code prefix: BCS).

    Mostly identical across programs (36 credits), with one difference:
    - Data Science has **"Fundamentals of Data Science"** (not in AIM/SAD).
    - AIM and SAD have **"Technical Report Writing"** (not in Data Science).

    Provide *prg* to get the exact BCS list for that program.
    Omit *prg* to get all BCS courses (with a note about the difference).

    Use this when the student asks:
    - "What computing science courses are there?"
    - "What BCS courses does the AI track have?"
    - "List basic computing sciences courses for data science"

    Args:
        prg:       Program name or alias (optional).
        year_flag: Set True to include the year level for each course.
        sem_flag:  Set True to include the semester for each course.
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
    Return every course in the curriculum for a program, grouped by category.

    Covers all four course categories:
      • Applied / specialized (51 cr)  — core + elective, program-specific
      • Humanities / general  (12 cr)  — GEN prefix, same for all programs
      • Math & basic sciences (24 cr)  — BAS prefix, same for all programs
      • Basic computing sci.  (36 cr)  — BCS prefix, mostly shared
    (Field training 6 cr and graduation projects 7 cr bring the full degree
    to 136 credits; those are noted but not listed as courses.)

    Use this when the student asks:
    - "What courses are in the data science program?"
    - "Show me the full curriculum for AIM"
    - "What do students study in the SAD track?"

    Args:
        prg:       Program name or alias.
        year_flag: Include year level for each course.
        sem_flag:  Include semester for each course.
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
    Return all **core (non-elective)** courses for a program, grouped by category.

    Covers:
      • Specialized core (51 cr)  — mandatory program-specific courses
      • Humanities / general (12 cr)
      • Math & basic sciences (24 cr)
      • Basic computing sciences (36 cr)
    (Field training 6 cr and graduation projects 7 cr noted but not listed.)
    Total degree: 136 credits.

    Use this when the student asks:
    - "What are the core courses in data science?"
    - "What courses are NOT elective in the AIM program?"
    - "Show me only mandatory courses for SAD"

    Args:
        prg:       Program name or alias.
        year_flag: Include year level for each course.
        sem_flag:  Include semester for each course.
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
    Return the courses that are **shared across all programs** (non-specialized).

    Covers:
      • Humanities / general  (12 cr)  — GEN prefix
      • Math & basic sciences (24 cr)  — BAS prefix
      • Basic computing sci.  (36 cr)  — BCS prefix (mostly shared)
    Total: 72 cr in courses + 13 cr field training/grad projects = 85 cr non-specialized.

    Provide *prg* to get the BCS list specific to that program (handles the
    DS vs AIM/SAD one-course difference).

    Use this when the student asks:
    - "What courses are NOT special to the data science program?"
    - "What courses are shared between all programs?"
    - "What courses would I take regardless of which track I choose?"

    Args:
        prg:       Program name or alias (optional).
        year_flag: Include year level for each course.
        sem_flag:  Include semester for each course.
    """
    try:
        from neo4j_track_functions import get_all_not_specialized_courses as _fn
        return _to_str(_fn(prg=prg, year_flag=year_flag, sem_flag=sem_flag))
    except Exception as exc:
        return f"Error fetching non-specialized courses: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Tool list exported to agent.py
# ─────────────────────────────────────────────────────────────────────────────

ALL_TOOLS = [
    get_student_info,
    get_course_info,
    get_course_prerequisites,
    get_course_timing,
    check_course_eligibility,
    get_courses_by_term,
    get_courses_by_multiple_terms,
    get_all_electives,
    get_elective_slots,
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
]