"""
eligibility.py — Course Eligibility Checker
=============================================

Updated implementation that fetches student context directly from Supabase
(one DB call) and checks prerequisites against the Neo4j knowledge graph.

Public API
──────────
    check_course_eligibility(student_id, course_name) → dict

    get_student_context(student_id) → dict
        Returns: {completed_courses, program_name, total_hours_earned}

Internal helpers
────────────────
    _get_student_row(student_id) → dict | None
    course_belongs_to_program(course_name, program_name) → bool

Differences from the old interactive_eligibility_check
───────────────────────────────────────────────────────
- Single Supabase query (not two separate calls)
- Reads credit hours directly from courses_degrees[].credit_hours
- Checks course_belongs_to_program before looking up prerequisites
- Returns a clean dict — no interactive prompts, no side effects
"""

from __future__ import annotations

import os
import logging
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Program code → canonical Neo4j name ──────────────────────────────────────

TRACK_MAP: Dict[str, str] = {
    "SAD": "software & application development",
    "AIM": "artificial intelligence & machine learning",
    "DAS": "data science",
}


def _norm(name: str) -> str:
    """Normalize course name for comparison: canonical form uses '&'.
    Handles OCR/Supabase data that may store either ' and ' or ' & '.
    Safe: requires spaces on both sides so words like 'understanding' are unaffected.
    """
    return name.replace(" and ", " & ")


# ── Supabase client (lazy singleton) ─────────────────────────────────────────

_supabase = None


def _get_supabase():
    global _supabase
    if _supabase is None:
        from supabase import create_client
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
        _supabase = create_client(url, key)
    return _supabase


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_student_row(student_id: str) -> Optional[dict]:
    """
    Fetch a single student row from Supabase.
    Returns the row dict or None if not found.
    """
    try:
        supabase = _get_supabase()
        response = (
            supabase.table("students")
            .select(
                "student_id, first_name, last_name, track, "
                "courses_degrees, total_hours_earned, university_year, gpa"#, current_term"
            )
            .eq("student_id", str(student_id))
            .single()
            .execute()
        )
        return response.data
    except Exception as exc:
        logger.warning("_get_student_row failed for %s: %s", student_id, exc)
        return None


def get_student_context(student_id: str) -> dict:
    """
    Fetch all needed student data in a single DB call.

    Returns:
        {
            "completed_courses":  list[str],   # lowercase course names
            "program_name":       str | None,  # full canonical KG program name
            "total_hours_earned": int,
            "university_year":    int | None,
            "current_term":       int | None,
            "gpa":                float,
            "first_name":         str | None,
            "last_name":          str | None,
        }

    Note: total_hours_earned is read directly from the DB column (set by the
    admin / grading system).  It is NOT recomputed from courses_degrees because
    the DB value is the authoritative source.
    """
    row = _get_student_row(student_id)

    if not row:
        return {
            "completed_courses":  [],
            "program_name":       None,
            "total_hours_earned": 0,
            "university_year":    None,
            "current_term":       None,
            "gpa":                0.0,
            "first_name":         None,
            "last_name":          None,
        }

    courses_degrees = row.get("courses_degrees") or []

    completed_courses = [
        _norm(course["name"].lower())
        for course in courses_degrees
        if isinstance(course, dict) and course.get("name")
    ]

    track_code   = (row.get("track") or "").strip().upper()
    program_name = TRACK_MAP.get(track_code)

    return {
        "completed_courses":  completed_courses,
        "program_name":       program_name,
        "total_hours_earned": row.get("total_hours_earned") or 0,
        "university_year":    row.get("university_year"),
        "current_term":       row.get("current_term") or None, # for testing
        "gpa":                row.get("gpa") or 0.0,
        "first_name":         row.get("first_name"),
        "last_name":          row.get("last_name"),
    }


def course_belongs_to_program(course_name: str, program_name: str) -> bool:
    """
    Check if a course belongs to a given program in the knowledge graph.

    Args:
        course_name:  Lowercase course name.
        program_name: Full program name (from TRACK_MAP).

    Returns:
        True if the course is linked to the program.
    """
    try:
        from neo4j_course_functions import run_cypher_query
        query = """
        MATCH (c:Course {name: $course_name})-[:BELONGS_TO]->(p:Program {name: $program_name})
        RETURN count(c) > 0 AS belongs
        """
        result = run_cypher_query(
            query,
            {"course_name": course_name, "program_name": program_name},
        )
        return result[0]["belongs"] if result else False
    except Exception as exc:
        logger.warning("course_belongs_to_program failed: %s", exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main eligibility function
# ─────────────────────────────────────────────────────────────────────────────

def check_course_eligibility(student_id: str, course_name: str) -> dict:
    """
    Check if a student is eligible to take a course based on prerequisites.

    Args:
        student_id:  The student's ID string.
        course_name: Name of the course to check (will be lowercased internally).

    Returns:
        {
            "eligible":               bool,
            "course":                 str,
            "message":                str,
            "missing_prerequisites":  list,
            "credit_requirement":     int | None,
            "credit_requirement_met": bool,
            "earned_credits":         int
        }
    """
    course_name = course_name.lower()

    # ── Gather student context ────────────────────────────────────────────────
    ctx               = get_student_context(student_id)
    completed_courses = ctx["completed_courses"]
    program_name      = ctx["program_name"]
    earned_credits    = ctx["total_hours_earned"]   # authoritative DB value

    # ── Guard: unknown program ────────────────────────────────────────────────
    if not program_name:
        return {
            "eligible":               False,
            "course":                 course_name,
            "message":                "❌ Could not determine your program. Please contact the registrar.",
            "missing_prerequisites":  [],
            "credit_requirement":     None,
            "credit_requirement_met": False,
            "earned_credits":         earned_credits,
        }

    # ── Guard: course must belong to the student's program ───────────────────
    if not course_belongs_to_program(course_name, program_name):
        return {
            "eligible":               False,
            "course":                 course_name,
            "message":                (
                f"❌ '{course_name}' does not belong to your program "
                f"({program_name})."
            ),
            "missing_prerequisites":  [],
            "credit_requirement":     None,
            "credit_requirement_met": False,
            "earned_credits":         earned_credits,
        }

    # ── Fetch prerequisites from the knowledge graph ──────────────────────────
    from neo4j_course_functions import get_course_dependencies
    dep_info      = get_course_dependencies(course_name, program_name, dependents=False)
    prerequisites = dep_info.get("prerequisites", [])

    # ── No prerequisites → immediately eligible ───────────────────────────────
    if not prerequisites:
        return {
            "eligible":               True,
            "course":                 course_name,
            "message":                f"✅ No prerequisites required for '{course_name}'. You can take this course!",
            "missing_prerequisites":  [],
            "credit_requirement":     None,
            "credit_requirement_met": True,
            "earned_credits":         earned_credits,
        }

    # ── Evaluate each prerequisite ────────────────────────────────────────────
    missing_courses        = []
    credit_requirement     = None
    credit_requirement_met = True

    for prereq in prerequisites:
        if "Required_Credit_Hours" in prereq:
            # Credit-hour gate (e.g. Graduation Project needs 100 ch)
            credit_requirement = int(prereq["Required_Credit_Hours"])
            if earned_credits < credit_requirement:
                credit_requirement_met = False
        else:
            # Course prerequisite gate
            prereq_name = (prereq.get("name") or "").lower()
            if prereq_name and prereq_name not in completed_courses:
                missing_courses.append(prereq)

    # ── Build result ──────────────────────────────────────────────────────────
    eligible = len(missing_courses) == 0 and credit_requirement_met

    result = {
        "eligible":               eligible,
        "course":                 course_name,
        "missing_prerequisites":  missing_courses,
        "credit_requirement":     credit_requirement,
        "credit_requirement_met": credit_requirement_met,
        "earned_credits":         earned_credits,
    }

    if eligible:
        result["message"] = f"✅ You are eligible to take '{course_name}'!"
    else:
        reasons = []
        if missing_courses:
            names = [p.get("name", "?") for p in missing_courses]
            reasons.append(f"Missing prerequisite courses: {', '.join(names)}")
        if not credit_requirement_met:
            reasons.append(
                f"Credit-hour requirement: need {credit_requirement} ch, "
                f"you have {earned_credits} ch"
            )
        result["message"] = (
            f"❌ You are NOT eligible to take '{course_name}'. "
            + " | ".join(reasons)
        )

    return result