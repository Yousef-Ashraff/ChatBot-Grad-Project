"""
Neo4j Track/Program Functions

This module provides functions to interact with the Neo4j knowledge graph
for program-level (track) data, such as total credits required per program.
"""

from neo4j_course_functions import (
    run_cypher_query,
    get_courses_by_multiple_terms,
    get_elective_slots_time,
    get_all_electives_by_program,
)


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
                     - Core + elective courses for years 3 and 4 (from Neo4j)
                     - Hardcoded year-1/2 courses that differ between tracks
                     - Elective slot schedule
                     - Full elective catalogue
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
        # Year 3 + 4 courses from the knowledge graph
        terms_3_4 = [
            {"level": 3, "semester": 1},
            {"level": 3, "semester": 2},
            {"level": 4, "semester": 1},
            {"level": 4, "semester": 2},
        ]
        curriculum_3_4 = get_courses_by_multiple_terms(terms_3_4, program_name=prg)

        # Hardcoded year-1/2 courses that differ between programs.
        # All programs share the same year-1 and year-2 curriculum EXCEPT:
        #   • "data science" has "fundamentals of data science"
        #     which is absent from AIM and SAD.
        #   • AIM and SAD have "technical report writing"
        #     which is absent from data science.
        _UNIQUE_YEAR12: dict = {
            "artificial intelligence & machine learning": [
                {
                    "course_name": "technical report writing",
                    "note": "present in AIM and SAD only (not in data science)",
                }
            ],
            "software & application development": [
                {
                    "course_name": "technical report writing",
                    "note": "present in AIM and SAD only (not in data science)",
                }
            ],
            "data science": [
                {
                    "course_name": "fundamentals of data science",
                    "note": "present in data science only (not in AIM or SAD)",
                }
            ],
        }

        result["curriculum"] = {
            "years_3_and_4": curriculum_3_4,
            "unique_year_1_2_courses": _UNIQUE_YEAR12.get(prg, []),
        }

        result["elective_slots"] = get_elective_slots_time(prg)
        result["electives"] = get_all_electives_by_program(prg)

    # ── Description / program overview ───────────────────────────────────────
    if desc_info:
        query = "MATCH (p:Program {name: $program_name}) RETURN p.description AS description"
        rows = run_cypher_query(query, {"program_name": prg})
        result["description"] = rows[0]["description"] if rows else None

    return result


def get_specialized_core_courses(prg: str) -> dict:
    """
    Return the specialized (upper-level) core curriculum for a program.

    "Specialized" courses are:
    - All core and elective courses in Years 3 and 4 (from Neo4j).
    - Program-specific early courses that are unique to the track, even though
      they appear in Years 1–2:
        • data science           → "fundamentals of data science" (Year 2, Sem 2)
        • AIM / SAD              → "technical report writing"      (Year 1, Sem 2)
    - Elective slot schedule for the program.

    Args:
        prg: Canonical program name, e.g.
             "artificial intelligence & machine learning",
             "software & application development",
             "data science".

    Returns:
        Dict with keys:
          - program
          - years_3_and_4          : output of get_courses_by_multiple_terms
          - program_specific_early  : dict describing the unique early course
          - elective_slots          : output of get_elective_slots_time
    """
    prg = prg.lower().strip()

    # ── Year 3 & 4 from the knowledge graph ──────────────────────────────────
    terms_3_4 = [
        {"level": 3, "semester": 1},
        {"level": 3, "semester": 2},
        {"level": 4, "semester": 1},
        {"level": 4, "semester": 2},
    ]
    curriculum_3_4 = get_courses_by_multiple_terms(terms_3_4, program_name=prg)

    # ── Program-specific early specialization course ──────────────────────────
    # These courses appear in Years 1–2 but are unique to a specific track.
    # Use get_courses_by_multiple_terms for year 2 sem 2 to get live KG data
    # for the data science case.
    if prg == "data science":
        y2_t2_data = get_courses_by_multiple_terms(
            [{"level": 2, "semester": 2}], program_name=prg
        )
        program_specific_early = {
            "course_name": "fundamentals of data science",
            "year": 2,
            "semester": 2,
            "note": "unique to data science — not offered in AIM or SAD",
            "term_data": y2_t2_data,
        }
    elif prg in (
        "artificial intelligence & machine learning",
        "software & application development",
    ):
        program_specific_early = {
            "course_name": "technical report writing",
            "year": 1,
            "semester": 2,
            "note": "unique to AIM and SAD — not offered in data science",
        }
    else:
        program_specific_early = None

    return {
        "program": prg,
        "years_3_and_4": curriculum_3_4,
        "program_specific_early": program_specific_early,
        "elective_slots": get_elective_slots_time(prg),
    }


def get_general_core_courses(prg: str) -> dict:
    """
    Return the general (foundational) core curriculum for a program.

    "General" courses are all courses in Years 1 and 2.  Almost all of these
    are shared across every track; the only exceptions are:
      • "fundamentals of data science" (Year 2, Sem 2) — data science ONLY
      • "technical report writing"     (Year 1, Sem 2) — AIM and SAD ONLY

    Notes about these track-specific courses are appended so the LLM can
    communicate them accurately to the student.

    Args:
        prg: Canonical program name.

    Returns:
        Dict with keys:
          - program
          - years_1_and_2        : output of get_courses_by_multiple_terms
          - program_specific_notes: list of dicts describing unique courses
    """
    prg = prg.lower().strip()

    # ── Year 1 & 2 from the knowledge graph ──────────────────────────────────
    terms_1_2 = [
        {"level": 1, "semester": 1},
        {"level": 1, "semester": 2},
        {"level": 2, "semester": 1},
        {"level": 2, "semester": 2},
    ]
    curriculum_1_2 = get_courses_by_multiple_terms(terms_1_2, program_name=prg)

    # ── Notes about program-specific courses in Years 1–2 ────────────────────
    if prg == "data science":
        program_specific_notes = [
            {
                "course_name": "fundamentals of data science",
                "year": 2,
                "semester": 2,
                "note": "present in data science only — not offered in AIM or SAD",
            }
        ]
    elif prg in (
        "artificial intelligence & machine learning",
        "software & application development",
    ):
        program_specific_notes = [
            {
                "course_name": "technical report writing",
                "year": 1,
                "semester": 2,
                "note": "present in AIM and SAD only — not offered in data science",
            }
        ]
    else:
        program_specific_notes = [
            {
                "note": (
                    "fundamentals of data science (Year 2, Sem 2) exists only in "
                    "data science; technical report writing (Year 1, Sem 2) exists "
                    "only in AIM and SAD."
                )
            }
        ]

    return {
        "program": prg,
        "years_1_and_2": curriculum_1_2,
        "program_specific_notes": program_specific_notes,
    }


def get_all_core_courses(prg: str) -> dict:
    """
    Return the complete core curriculum for a program (Years 1–4) in one call.

    Combines get_general_core_courses (Years 1–2) and
    get_specialized_core_courses (Years 3–4) into a single, structured result.

    Args:
        prg: Canonical program name.

    Returns:
        Dict with keys:
          - program
          - general_core           : years 1 & 2 curriculum
          - general_program_notes  : track-specific notes for years 1–2
          - specialized_core       : years 3 & 4 curriculum
          - program_specific_early : unique early specialization course entry
          - elective_slots         : when electives are available
    """
    prg = prg.lower().strip()

    general = get_general_core_courses(prg)
    specialized = get_specialized_core_courses(prg)

    return {
        "program": prg,
        "general_core (years 1-2)": general["years_1_and_2"],
        "general_program_notes (years 1-2)": general["program_specific_notes"],
        "specialized_core (years 3-4)": specialized["years_3_and_4"],
        "program_specific_early_course": specialized["program_specific_early"],
        "elective_slots": specialized["elective_slots"],
    }


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
