"""
Neo4j Track/Program Functions

This module provides functions to interact with the Neo4j knowledge graph
for program-level (track) data, such as total credits required per program.
"""

from neo4j_course_functions import (
    run_cypher_query,
    get_courses_by_multiple_terms,
    get_elective_slots_time_and_occ,
    get_elective_slots_time,          # backward-compat alias used by get_program_info
    get_all_electives_by_program,
)

# ──────────────────────────────────────────────────────────────────────────────
# Program metadata
# ──────────────────────────────────────────────────────────────────────────────

# Canonical program name → course code prefix
PROGRAM_CODE_PREFIX = {
    "artificial intelligence & machine learning": "AIM",
    "software & application development": "SAD",
    "data science": "DAS",
}



def _query_courses_by_code_prefix(
    code_prefix: str,
    elective: str = None,
    program_name: str = None,
    year_flag: bool = False,
    sem_flag: bool = False,
) -> list:
    """
    Query courses from Neo4j whose code starts with *code_prefix*.

    Args:
        code_prefix:   Course-code prefix to match (e.g. 'AIM', 'GEN', 'BAS', 'BCS').
        elective:      'yes' → elective only, 'no' → core only, None → both.
        program_name:  If given, restrict to that program; else deduplicate across all.
        year_flag:     Add 'year' field to each returned course dict.
        sem_flag:      Add 'semester' field to each returned course dict.

    Returns:
        List of course dicts (course_name, course_code, credit_hours, [year], [semester]).
    """
    conditions = ["(c.code STARTS WITH $code_prefix OR r.code STARTS WITH $code_prefix)"]
    params: dict = {"code_prefix": code_prefix}

    if elective is not None:
        conditions.append("r.elective = $elective")
        params["elective"] = elective

    if program_name is not None:
        conditions.append("toLower(p.name) = $program_name")
        params["program_name"] = program_name.lower()

    where_clause = " AND ".join(conditions)

    # Always fetch year/sem from DB so we can populate the flags;
    # we just won't include them in the output dict if the flags are off.
    query = f"""
    MATCH (c:Course)-[r:BELONGS_TO]->(p:Program)
    WHERE {where_clause}
    RETURN c.name AS course_name,
           COALESCE(c.code, r.code) AS course_code,
           c.credit_hours AS credit_hours,
           r.year_name AS year,
           r.semester AS semester
    ORDER BY COALESCE(c.code, r.code)
    """

    rows = run_cypher_query(query, params)

    courses = []
    seen_codes: set = set()
    for row in rows:
        code = row["course_code"]
        # When no program filter, deduplicate by course code (same course appears
        # for each program it belongs to; year/sem are identical across programs
        # for GEN/BAS courses, so the first row is fine).
        if program_name is None and code in seen_codes:
            continue
        seen_codes.add(code)

        course: dict = {
            "course_name": row["course_name"],
            "course_code": row["course_code"],
            "credit_hours": row["credit_hours"],
        }
        if year_flag:
            course["year"] = row.get("year")
        if sem_flag:
            course["semester"] = row.get("semester")
        courses.append(course)

    return courses


def _parse_elective_slot(slot_str: str) -> dict:
    """
    Parse a slot string such as 'Third Year / Second Sem' into its parts.

    Returns a dict with keys 'year' (e.g. 'Third Year') and
    'semester' (e.g. 'Second').  Unknown formats yield None values.
    """
    # Expected format: "{Ordinal} Year / {First|Second} Sem"
    try:
        left, right = slot_str.split(" / ", 1)
        year = left.strip()                          # "Third Year"
        semester = right.strip().replace(" Sem", "").strip()  # "Second"
        return {"year": year, "semester": semester}
    except (ValueError, AttributeError):
        return {"year": None, "semester": None}




# ──────────────────────────────────────────────────────────────────────────────
# Public course-category functions
# ──────────────────────────────────────────────────────────────────────────────

def get_specialized_core_courses(
    prg: str,
    year_flag: bool = False,
    sem_flag: bool = False,
) -> dict:
    """
    Return specialized **core** (non-elective) courses for *prg*, plus the
    elective slot entries appended at the end of the courses list.

    Identified by: code starts with the program prefix (AIM / SAD / DAS)
                   AND r.elective = 'no'.

    Each elective slot in the program's schedule is included as a course
    entry with course_name='Elective Slot' and course_code='ELECTIVE'.
    When year_flag / sem_flag is True, the slot's year and semester (parsed
    from get_elective_slots_time) are added to that entry exactly as they
    are added for every real course.

    These map to the "applied / specialized courses" (51 cr) category.

    Args:
        prg:       Program name or alias.
        year_flag: Include 'year' field in each course dict (real + slot entries).
        sem_flag:  Include 'semester' field in each course dict (real + slot entries).
    """
    code_prefix = PROGRAM_CODE_PREFIX.get(prg)
    if not code_prefix:
        return {"error": f"Unknown program '{prg}'. Valid: {list(PROGRAM_CODE_PREFIX.keys())}"}

    # ── Real core courses ────────────────────────────────────────────────────
    courses = _query_courses_by_code_prefix(
        code_prefix=code_prefix,
        elective="no",
        program_name=prg,
        year_flag=year_flag,
        sem_flag=sem_flag,
    )

    # ── Elective slot entries ────────────────────────────────────────────────
    slot_credit_hours = 3
    slots = get_elective_slots_time_and_occ(prg)   # list of {"slot": "Year / Sem", "count": N}
    if not isinstance(slots, list):
        slots = []

    for slot_info in slots:
        slot_str = slot_info["slot"]
        count = slot_info["count"]
        parsed = _parse_elective_slot(slot_str)
        slot_entry: dict = {
            "course_name": "Elective Slot",
            "course_code": "ELECTIVE",
            "credit_hours": slot_credit_hours * count,   # total credits for this time position
            "count": count,                              # number of slots at this time
            "slot_schedule": slot_str,
        }
        if year_flag:
            slot_entry["year"] = parsed["year"]
        if sem_flag:
            slot_entry["semester"] = parsed["semester"]
        courses.append(slot_entry)

    total_credits = sum(c["credit_hours"] or 0 for c in courses)

    return {
        "program": prg,
        "type": "specialized_core",
        "total_credits": total_credits,
        "courses": courses,
    }


def get_specialized_elective_courses(
    prg: str,
    year_flag: bool = False,
    sem_flag: bool = False,
) -> dict:
    """
    Return specialized **elective** courses available in *prg*.

    Identified by: code starts with the program prefix (AIM / SAD / DAS)
                   AND r.elective = 'yes'.

    When year_flag / sem_flag is True the elective *slot* schedule is
    returned instead of per-course year/semester values (elective courses
    have no fixed placement in the curriculum — only the slots do).

    Args:
        prg:       Program name or alias.
        year_flag: Include elective slot year info.
        sem_flag:  Include elective slot semester info.
    """
    code_prefix = PROGRAM_CODE_PREFIX.get(prg)
    if not code_prefix:
        return {"error": f"Unknown program '{prg}'. Valid: {list(PROGRAM_CODE_PREFIX.keys())}"}

    courses = _query_courses_by_code_prefix(
        code_prefix=code_prefix,
        elective="yes",
        program_name=prg,
        year_flag=False,   # no fixed year/sem per elective course
        sem_flag=False,
    )
    # total_credits = sum(c["credit_hours"] or 0 for c in courses)

    result: dict = {
        "program": prg,
        "type": "specialized_elective",
        "total_credits": 0,
        "courses": courses,
    }

    if year_flag or sem_flag:
        result["elective_slots"] = get_elective_slots_time_and_occ(prg)
        result["note"] = (
            "Elective courses have no fixed year/semester. "
            "'elective_slots' shows when students may fill these slots."
        )

    return result


def get_all_specialized_courses(
    prg: str,
    year_flag: bool = False,
    sem_flag: bool = False,
) -> dict:
    """
    Return **all** specialized courses (core + elective) for *prg*.

    Combines get_specialized_core_courses + get_specialized_elective_courses.
    The applied/specialized category totals 51 credits.

    Args:
        prg:       Program name or alias.
        year_flag: Include year for core courses; elective slot years for electives.
        sem_flag:  Include semester for core courses; elective slot sems for electives.
    """

    core_data = get_specialized_core_courses(prg, year_flag=year_flag, sem_flag=sem_flag)
    if "error" in core_data:
        return core_data

    elective_data = get_specialized_elective_courses(prg, year_flag=year_flag, sem_flag=sem_flag)

    total_credits = (core_data["total_credits"] or 0) + (elective_data["total_credits"] or 0)

    result: dict = {
        "program": prg,
        "type": "all_specialized",
        "total_credits": total_credits,
        "core_courses": {
            "count": len(core_data["courses"]),
            "total_credits": core_data["total_credits"],
            "courses": core_data["courses"],
        },
        "elective_courses": {
            "count": len(elective_data["courses"]),
            "total_credits": elective_data["total_credits"],
            "courses": elective_data["courses"],
        },
    }

    if year_flag or sem_flag:
        result["elective_slots"] = elective_data.get("elective_slots")
        result["note"] = elective_data.get("note")

    return result


def get_general_courses(
    year_flag: bool = False,
    sem_flag: bool = False,
) -> dict:
    """
    Return humanities / general courses (code prefix: GEN).

    These are **identical across all three programs** (12 credits total).

    Args:
        year_flag: Include 'year' field in each course dict.
        sem_flag:  Include 'semester' field in each course dict.
    """
    courses = _query_courses_by_code_prefix(
        code_prefix="GEN",
        elective=None,
        program_name=None,
        year_flag=year_flag,
        sem_flag=sem_flag,
    )
    total_credits = sum(c["credit_hours"] or 0 for c in courses)

    return {
        "type": "general (humanities)",
        "note": "Same for all programs — 12 credits",
        "total_credits": total_credits,
        "courses": courses,
    }


def get_MathAndBasicScience_courses(
    year_flag: bool = False,
    sem_flag: bool = False,
) -> dict:
    """
    Return mathematics and basic sciences courses (code prefix: BAS).

    These are **identical across all three programs** (24 credits total).

    Args:
        year_flag: Include 'year' field in each course dict.
        sem_flag:  Include 'semester' field in each course dict.
    """
    courses = _query_courses_by_code_prefix(
        code_prefix="BAS",
        elective=None,
        program_name=None,
        year_flag=year_flag,
        sem_flag=sem_flag,
    )
    total_credits = sum(c["credit_hours"] or 0 for c in courses)

    return {
        "type": "mathematics and basic sciences",
        "note": "Same for all programs — 24 credits",
        "total_credits": total_credits,
        "courses": courses,
    }


def get_BasicComputingSciences_courses(
    prg: str = None,
    year_flag: bool = False,
    sem_flag: bool = False,
) -> dict:
    """
    Return basic computing sciences courses (code prefix: BCS).

    Mostly identical across programs (36 credits), with one program-specific
    difference:
    - Data Science has **"Fundamentals of Data Science"** (absent from AIM/SAD).
    - AIM and SAD have **"Technical Report Writing"** (absent from Data Science).

    When *prg* is provided the returned list is filtered to that program's
    BCS courses, which naturally handles the difference above.
    When *prg* is None, all BCS courses are returned (with the note above).

    Args:
        prg:       Program name or alias (optional).
        year_flag: Include 'year' field in each course dict.
        sem_flag:  Include 'semester' field in each course dict.
    """

    courses = _query_courses_by_code_prefix(
        code_prefix="BCS",
        elective=None,
        program_name=prg,
        year_flag=year_flag,
        sem_flag=sem_flag,
    )
    total_credits = sum(c["credit_hours"] or 0 for c in courses)

    note = (
        f"BCS courses for program: {prg}"
        if prg
        else (
            "BCS courses (all programs). Note: Data Science has "
            "'Fundamentals of Data Science' in place of 'Technical Report Writing' "
            "which exists in AIM and SAD."
        )
    )

    return {
        "type": "basic computing sciences",
        "program": prg,
        "note": note,
        "total_credits": total_credits,
        "courses": courses,
    }


def get_all_types_courses(
    prg: str,
    year_flag: bool = False,
    sem_flag: bool = False,
) -> dict:
    """
    Return ALL course categories for *prg* — every course in the curriculum.

    Combines (with expected credit totals from the curriculum):
      • Applied / specialized courses  — 51 cr  (get_all_specialized_courses)
      • Humanities / general           — 12 cr  (get_general_courses)
      • Mathematics & basic sciences   — 24 cr  (get_MathAndBasicScience_courses)
      • Basic computing sciences       — 36 cr  (get_BasicComputingSciences_courses)
      • Field training                 —  6 cr  (note only — no course list)
      • Graduation projects            —  7 cr  (note only — no course list)
      ─────────────────────────────────────────
      Total degree requirement         136 cr

    Args:
        prg:       Program name or alias.
        year_flag: Include year for each course.
        sem_flag:  Include semester for each course.
    """

    specialized = get_all_specialized_courses(prg, year_flag=year_flag, sem_flag=sem_flag)
    general = get_general_courses(year_flag=year_flag, sem_flag=sem_flag)
    math_sci = get_MathAndBasicScience_courses(year_flag=year_flag, sem_flag=sem_flag)
    bcs = get_BasicComputingSciences_courses(prg=prg, year_flag=year_flag, sem_flag=sem_flag)

    courses_total = (
        (specialized.get("total_credits") or 0)
        + (general.get("total_credits") or 0)
        + (math_sci.get("total_credits") or 0)
        + (bcs.get("total_credits") or 0)
    )

    return {
        "program": prg,
        "type": "all_types",
        "total_credits_in_courses": courses_total,
        "full_degree_total_credits": 136,
        "note": (
            "Field training (6 cr) and graduation projects (7 cr) are not listed "
            "as courses here but count toward the 136-credit degree requirement."
        ),
        "specialized_courses": specialized,
        "general_courses": general,
        "math_and_basic_sciences": math_sci,
        "basic_computing_sciences": bcs,
    }


def get_all_core_courses(
    prg: str,
    year_flag: bool = False,
    sem_flag: bool = False,
) -> dict:
    """
    Return all **core (non-elective)** courses for *prg* across all categories.

    Combines:
      • Specialized core               — 51 cr  (get_specialized_core_courses)
      • Humanities / general           — 12 cr  (get_general_courses)
      • Mathematics & basic sciences   — 24 cr  (get_MathAndBasicScience_courses)
      • Basic computing sciences       — 36 cr  (get_BasicComputingSciences_courses)
      • Field training                 —  6 cr  (note only)
      • Graduation projects            —  7 cr  (note only)
      ─────────────────────────────────────────
      Total degree requirement         136 cr

    Args:
        prg:       Program name or alias.
        year_flag: Include year for each course.
        sem_flag:  Include semester for each course.
    """

    spec_core = get_specialized_core_courses(prg, year_flag=year_flag, sem_flag=sem_flag)
    if "error" in spec_core:
        return spec_core

    general = get_general_courses(year_flag=year_flag, sem_flag=sem_flag)
    math_sci = get_MathAndBasicScience_courses(year_flag=year_flag, sem_flag=sem_flag)
    bcs = get_BasicComputingSciences_courses(prg=prg, year_flag=year_flag, sem_flag=sem_flag)

    courses_total = (
        (spec_core.get("total_credits") or 0)
        + (general.get("total_credits") or 0)
        + (math_sci.get("total_credits") or 0)
        + (bcs.get("total_credits") or 0)
    )

    return {
        "program": prg,
        "type": "all_core",
        "total_credits_in_courses": courses_total,
        "full_degree_total_credits": 136,
        "note": (
            "Elective courses are excluded. "
            "Field training (6 cr) and graduation projects (7 cr) are not listed "
            "as courses here but count toward the 136-credit degree requirement."
        ),
        "specialized_core": spec_core,
        "general_courses": general,
        "math_and_basic_sciences": math_sci,
        "basic_computing_sciences": bcs,
    }


def get_all_not_specialized_courses(
    prg: str = None,
    year_flag: bool = False,
    sem_flag: bool = False,
) -> dict:
    """
    Return all courses that are **shared across programs** (non-specialized).

    Combines:
      • Humanities / general           — 12 cr  (get_general_courses)
      • Mathematics & basic sciences   — 24 cr  (get_MathAndBasicScience_courses)
      • Basic computing sciences       — 36 cr  (get_BasicComputingSciences_courses)
      ─────────────────────────────────────────
      Subtotal                          72 cr
      Field training + grad projects   +13 cr  (note only)
      ─────────────────────────────────────────
      Non-specialized degree total      85 cr

    When *prg* is provided, the BCS list is filtered to that program
    (handles the Data-Science-specific vs AIM/SAD-specific BCS difference).

    Args:
        prg:       Program name or alias (optional).
        year_flag: Include year for each course.
        sem_flag:  Include semester for each course.
    """

    general = get_general_courses(year_flag=year_flag, sem_flag=sem_flag)
    math_sci = get_MathAndBasicScience_courses(year_flag=year_flag, sem_flag=sem_flag)
    bcs = get_BasicComputingSciences_courses(prg=prg, year_flag=year_flag, sem_flag=sem_flag)

    courses_total = (
        (general.get("total_credits") or 0)
        + (math_sci.get("total_credits") or 0)
        + (bcs.get("total_credits") or 0)
    )

    result: dict = {
        "type": "all_not_specialized",
        "total_credits_in_courses": courses_total,
        "non_specialized_degree_total": 85,
        "note": (
            "These courses are common to all programs. "
            "Field training (6 cr) and graduation projects (7 cr) bring the "
            "non-specialized total to 85 cr out of the 136-credit degree."
        ),
        "general_courses": general,
        "math_and_basic_sciences": math_sci,
        "basic_computing_sciences": bcs,
    }

    if prg:
        result["program"] = prg

    return result


def get_credit_hour_distribution() -> dict:
    """
    Return the faculty-wide credit hour distribution shared by all programs.

    This breakdown is identical across all three tracks (AIM, SAD, Data Science).
    Call this function once when credit distribution info is needed — do NOT
    call it once per program as it is program-agnostic.

    Returns:
        Dict with category → credit hours mapping and a total.
    """
    return {
        "distribution": {
            "humanities": 12,
            "mathematics and basic sciences": 24,
            "basic computing sciences": 36,
            "applied / specialized courses": 51,
            "field training": 6,
            "graduation projects": 7,
        }
    }


def get_program_info(prg: str, course_info: bool = True, desc_info: bool = True) -> dict:
    """
    Get comprehensive information about a specific program/track.

    Args:
        prg:         Program name or alias (e.g. "AIM", "data science", "SAD").
        course_info: If True, include curriculum data:
                     - All specialized courses (core + elective) via get_all_specialized_courses
                     - Elective slot schedule (with occurrence counts)
                     - Note about shared course types across all programs
        desc_info:   If True, fetch the program description from Neo4j.

    Returns:
        Dict with keys populated based on the flags set.
        Always includes credit_hour_distribution (shared across all programs).
    """

    if prg is None:
        return {"error": "Program name must be provided. Use 'AIM', 'SAD', or 'data science'."}

    result: dict = {
        "program": prg,
        "credit_hour_distribution": get_credit_hour_distribution(),
    }

    # ── Course information ────────────────────────────────────────────────────
    if course_info:
        result["specialized_courses"] = get_all_specialized_courses(prg)

        result["shared_courses_note"] = (
            "All three programs (AIM, SAD, Data Science) share the same courses in the "
            "following categories: General Courses (humanities), Math & Basic Science, and "
            "Basic Computing Sciences — EXCEPT in Basic Computing Sciences: "
            "'data science' has 'Fundamentals of Data Science' which is absent from "
            "AIM and SAD; AIM and SAD have 'Technical Report Writing' which is absent "
            "from Data Science."
        )

    # ── Description / program overview ───────────────────────────────────────
    if desc_info:
        query = "MATCH (p:Program {name: $program_name}) RETURN p.description AS description"
        rows = run_cypher_query(query, {"program_name": prg})
        result["description"] = rows[0]["description"] if rows else None

    return result


def get_program_total_credits(program_name: str) -> dict:
    """
    Return the total_credits_required for a given program node.

    Args:
        program_name: Canonical program name, e.g.
                      "artificial intelligence and machine learning",
                      "software and application development",
                      "data science".

    Returns:
        dict with keys:
          - program_name (str)
          - total_credits_required (int | None)
        or an error dict if the program is not found.
    """
    query = """
    MATCH (p:Program)
    WHERE toLower(p.name) = toLower($program_name)
    RETURN p.name AS program_name, p.total_credits_required AS total_credits_required
    """
    rows = run_cypher_query(query, {"program_name": program_name.strip()})

    if not rows:
        return {
            "error": f"No program found matching '{program_name}'.",
            "program_name": program_name,
            "total_credits_required": None,
        }

    return {
        "program_name": rows[0]["program_name"],
        "total_credits_required": rows[0]["total_credits_required"],
    }
