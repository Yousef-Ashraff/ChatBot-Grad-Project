"""
gpa_service.py — GPA Computation and Analysis Service
======================================================

Scenarios handled:
  2. Project GPA given new courses with hypothetical grades
  3. Find combination + per-type minimum grades needed to reach a target GPA

GPA formula (BNU bylaw Art. 13):
  Cumulative GPA = Σ(points × credit_hours) / Σ(credit_hours)

Grade table:
  ≥96% → 4.00 (A+)  |  ≥92% → 3.70 (A)  |  ≥88% → 3.40 (A-)
  ≥84% → 3.20 (B+)  |  ≥80% → 3.00 (B)  |  ≥76% → 2.80 (B-)
  ≥72% → 2.60 (C+)  |  ≥68% → 2.40 (C)  |  ≥64% → 2.20 (C-)
  ≥60% → 2.00 (D+)  |  ≥55% → 1.50 (D)  |  ≥50% → 1.00 (D-)
  <50%  → 0.00 (F)
"""

from __future__ import annotations

import math
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Grade table ───────────────────────────────────────────────────────────────
# Each entry: (min_percentage, grade_points, grade_letter)
# Sorted highest → lowest threshold.

GRADE_TABLE: List[Tuple[int, float, str]] = [
    (96, 4.00, "A+"),
    (92, 3.70, "A"),
    (88, 3.40, "A-"),
    (84, 3.20, "B+"),
    (80, 3.00, "B"),
    (76, 2.80, "B-"),
    (72, 2.60, "C+"),
    (68, 2.40, "C"),
    (64, 2.20, "C-"),
    (60, 2.00, "D+"),
    (55, 1.50, "D"),
    (50, 1.00, "D-"),
    (0,  0.00, "F"),
]


def percentage_to_points(pct: float) -> Tuple[float, str]:
    """Convert a percentage score to GPA grade points and letter."""
    for min_pct, pts, grade in GRADE_TABLE:
        if pct >= min_pct:
            return pts, grade
    return 0.00, "F"


def points_to_grade_requirement(required_pts: float) -> Tuple[str, int, float]:
    """
    Find the LOWEST grade (minimum percentage) that gives AT LEAST required_pts.
    Scans from lowest points upward so we return the easiest grade that satisfies
    the requirement.

    Returns (grade_letter, min_percentage, awarded_points).
    E.g. points_to_grade_requirement(2.5) → ("C+", 72, 2.60)
    """
    if required_pts <= 0:
        return "F", 0, 0.00
    for min_pct, pts, grade in reversed(GRADE_TABLE):
        if pts >= required_pts:
            return grade, min_pct, pts
    return "A+", 96, 4.00


def _grade_range_str(grade_letter: str, min_pct: int) -> str:
    """Return a human-readable percentage range string, e.g. '80–83% (B)'."""
    for i, (threshold, _, letter) in enumerate(GRADE_TABLE):
        if letter == grade_letter:
            if i == 0:
                return f"≥{min_pct}% ({grade_letter})"
            upper = GRADE_TABLE[i - 1][0] - 1
            return f"{min_pct}–{upper}% ({grade_letter})"
    return f"≥{min_pct}% ({grade_letter})"


# ── GPA computation ───────────────────────────────────────────────────────────

def compute_gpa_from_courses(courses_degrees: list) -> Tuple[float, int, float]:
    """
    Compute GPA directly from the courses_degrees JSON column.
    F grades contribute 0 quality points but still count in the credit total.

    Returns (gpa, total_credits, total_quality_points).
    """
    total_qp = 0.0
    total_cr = 0
    for c in (courses_degrees or []):
        if not isinstance(c, dict):
            continue
        pct = float(c.get("degree") or 0)
        cr  = int(c.get("credit_hours") or 0)
        pts, _ = percentage_to_points(pct)
        total_qp += pts * cr
        total_cr += cr
    gpa = round(total_qp / total_cr, 4) if total_cr > 0 else 0.0
    return gpa, total_cr, total_qp


# ── Semester inference ────────────────────────────────────────────────────────

def infer_current_semester(courses_degrees: list) -> Optional[str]:
    """
    Infer the UPCOMING semester from the most recently completed semester.
    Skips "Summer" entries and moves to the previous element.

    Most recent completed "Fall"   → next upcoming is Spring → "Second"
    Most recent completed "Spring" → next upcoming is Fall   → "First"
    Returns None if undetermined.
    """
    import re
    best_key = (-1, -1)
    result = None

    _SEASON_PRIORITY = {"spring": 1, "fall": 3}  # summer (2) skipped

    for c in (courses_degrees or []):
        sem = (c.get("semester") or "").strip().lower()
        if "summer" in sem:
            continue
        m = re.search(r"\d{4}", sem)
        year = int(m.group()) if m else 0
        for season, priority in _SEASON_PRIORITY.items():
            if season in sem:
                if (year, priority) > best_key:
                    best_key = (year, priority)
                    # completed Fall → next is Spring (Second)
                    # completed Spring → next is Fall (First)
                    result = "Second" if season == "fall" else "First"
                break

    return result


# ── Eligible-course helper ────────────────────────────────────────────────────

def _get_eligible_not_completed(
    credit_hours: int,
    program_name: str,
    completed,
    earned_credits: int,
    semester: Optional[str],
) -> List[dict]:
    """
    Return eligible, not-completed CORE (mandatory) courses of a given
    credit_hours value for the student's program, optionally filtered by semester.
    """
    from neo4j_course_functions import (
        filter_courses as _filter_courses,
        check_course_eligibility as _elig_fn,
    )

    courses = _filter_courses(
        filters={"credit_hours": credit_hours},
        course_types=["core"],
        program_name=program_name,
        semester=semester,
    )
    if not courses:
        return []

    courses = [c for c in courses if c.get("name", "").lower() not in completed]
    if not courses:
        return []

    completed_list = list(completed)
    eligible = []
    for c in courses:
        name = c.get("name", "").lower()
        if not name:
            continue
        result = _elig_fn(
            course_name=name,
            completed_courses=completed_list,
            earned_credits=earned_credits,
            program_name=program_name,
        )
        if result.get("eligible"):
            eligible.append(c)
    return eligible


def _build_remaining_elective_slots(
    program_name: str,
    completed,
) -> dict:
    """
    Mirror planning.py's pre-computed remaining_slots logic.

    Returns a dict: {(year: int, semester: str): remaining_slots: int}
    Only terms with at least one total slot are included.
    Completed electives drain the earliest available slot first.
    """
    from planning import get_elective_slots
    from neo4j_course_functions import get_all_electives_by_program

    # Build the ordered list of terms that have any slots
    slot_terms = [
        (y, s)
        for y in range(1, 5)
        for s in ["First", "Second"]
        if get_elective_slots(program_name, y, s) > 0
    ]
    remaining_slots = {
        (y, s): get_elective_slots(program_name, y, s)
        for (y, s) in slot_terms
    }

    # Drain one slot per completed elective (earliest term first)
    all_electives = get_all_electives_by_program(program_name) or []
    all_elective_names = {e["course_name"].lower() for e in all_electives}
    for name in completed:
        if name in all_elective_names:
            for ys in slot_terms:
                if remaining_slots[ys] > 0:
                    remaining_slots[ys] -= 1
                    break

    return remaining_slots


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2 — Project GPA with hypothetical new-course grades
# ─────────────────────────────────────────────────────────────────────────────

def project_gpa_with_new_courses(student_id: str, new_courses: List[dict]) -> dict:
    """
    Project the student's GPA if they score specific percentages in new courses.

    new_courses: [{"name": "machine learning", "percentage": 80}, ...]
    Credit hours for each course are looked up from Neo4j.
    """
    from eligibility import _get_student_row
    row = _get_student_row(student_id)
    if not row:
        return {"error": "Student not found."}

    courses_degrees = row.get("courses_degrees") or []
    _, existing_credits, existing_qp = compute_gpa_from_courses(courses_degrees)
    current_gpa = float(row.get("gpa") or 0.0)

    from neo4j_course_functions import get_course_info as _get_course_info

    new_entries   = []
    lookup_errors = []
    for item in new_courses:
        name = str(item.get("name") or "").strip().lower()
        pct  = float(item.get("percentage") or 0)
        if not name:
            continue

        # Look up credit hours from Neo4j
        cr = 3  # safe fallback
        try:
            info_list = _get_course_info(name)
            if info_list and isinstance(info_list, list) and info_list[0]:
                cr = int(info_list[0].get("credit_hours") or 3)
        except Exception:
            lookup_errors.append(name)

        pts, grade_letter = percentage_to_points(pct)
        new_entries.append({
            "name":         name,
            "percentage":   pct,
            "grade":        grade_letter,
            "credit_hours": cr,
            "points":       pts,
        })

    if not new_entries:
        return {"error": "No valid new courses provided."}

    new_qp = sum(e["points"] * e["credit_hours"] for e in new_entries)
    new_cr = sum(e["credit_hours"]               for e in new_entries)
    total_cr    = existing_credits + new_cr
    proj_gpa    = round((existing_qp + new_qp) / total_cr, 3) if total_cr > 0 else 0.0

    result = {
        "current_gpa":      round(current_gpa, 3),
        "projected_gpa":    proj_gpa,
        "existing_credits": existing_credits,
        "new_credits":      new_cr,
        "total_credits":    total_cr,
        "new_courses":      new_entries,
    }
    if lookup_errors:
        result["note"] = (
            f"Credit hours defaulted to 3 for course(s) not found in the "
            f"knowledge graph: {', '.join(lookup_errors)}"
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3 — Analyze what it takes to reach a target GPA
# ─────────────────────────────────────────────────────────────────────────────

def _find_combination(
    count_1cr: int, count_2cr: int, count_3cr: int, count_4cr: int,
    credit_limit: int,
    existing_credits: int, existing_qp: float,
    target_gpa: float,
    optimization: str,
) -> dict:
    """
    Find (n1, n2, n3, n4) based on the chosen optimization strategy.

    maximize_credits: fill the credit limit — highest-credit courses first
                      (4cr → 3cr → 2cr → 1cr).
    minimize_grade  : use the fewest credits that still make the target
                      achievable (Q_needed ≤ 4.0×T), same priority order.
    """
    def _greedy_fill(budget: int) -> Tuple[int, int, int, int]:
        n4 = min(count_4cr, budget // 4)
        rem = budget - n4 * 4
        n3 = min(count_3cr, rem // 3)
        rem -= n3 * 3
        n2 = min(count_2cr, rem // 2)
        rem -= n2 * 2
        n1 = min(count_1cr, rem)
        return n1, n2, n3, n4

    if optimization == "minimize_grade":
        if target_gpa >= 4.0:
            t_min = credit_limit
        else:
            numerator = target_gpa * existing_credits - existing_qp
            t_min = max(0, math.ceil(numerator / (4.0 - target_gpa)))

        for budget in range(min(t_min, credit_limit), credit_limit + 1):
            n1, n2, n3, n4 = _greedy_fill(budget)
            T = n1 + n2 * 2 + n3 * 3 + n4 * 4
            if T >= t_min:
                break
        else:
            n1, n2, n3, n4 = _greedy_fill(credit_limit)
    else:  # maximize_credits (default)
        n1, n2, n3, n4 = _greedy_fill(credit_limit)

    T          = n1 + n2 * 2 + n3 * 3 + n4 * 4
    Q_needed   = target_gpa * (existing_credits + T) - existing_qp
    achievable = T > 0 and Q_needed <= 4.0 * T
    if T > 0:
        max_gpa = round((existing_qp + 4.0 * T) / (existing_credits + T), 3)
    else:
        max_gpa = round(existing_qp / existing_credits, 3) if existing_credits else 0.0

    return {
        "n1": n1, "n2": n2, "n3": n3, "n4": n4,
        "total_credits": T,
        "Q_needed":      Q_needed,
        "achievable":    achievable,
        "max_achievable_gpa": max_gpa,
    }


def _compute_type_floors(
    Q_needed: float, T: int, n1: int, n2: int, n3: int, n4: int,
) -> dict:
    """
    For each active credit type compute the per-course minimum grade,
    assuming ALL OTHER TYPES score 4.0 (100%).

    Formula per type:
        floor_type = (Q_needed - 4.0 × (T - n_type × ch)) / (n_type × ch)

    All courses within the same credit type share the same floor.
    All four floors must be met simultaneously.
    """
    floors = {}
    for ch, n, key in [(1, n1, "1cr"), (2, n2, "2cr"), (3, n3, "3cr"), (4, n4, "4cr")]:
        if n == 0:
            continue
        type_credits  = n * ch
        other_credits = T - type_credits
        raw_floor     = (Q_needed - 4.0 * other_credits) / type_credits
        floor_pts     = max(0.0, min(4.0, raw_floor))
        grade, min_pct, _ = points_to_grade_requirement(floor_pts)
        floors[key] = {
            "count":          n,
            "credit_hours":   ch,
            "floor_points":   round(floor_pts, 3),
            "min_percentage": min_pct,
            "grade":          grade,
            "range_str":      _grade_range_str(grade, min_pct),
        }
    return floors


def _advisor_course_notes(
    notes: list,
    n1: int, n2: int, n3: int, n4: int,
    courses_1cr: list, courses_2cr: list, courses_3cr: list, courses_4cr: list,
    open_slots_by_term: Optional[dict] = None,
) -> None:
    """Append notes listing eligible course names by credit type and elective slots."""
    for n, courses, label, is_3cr in [
        (n1, courses_1cr, "1-credit", False),
        (n2, courses_2cr, "2-credit", False),
        (n3, courses_3cr, "3-credit", True),
        (n4, courses_4cr, "4-credit", False),
    ]:
        if n == 0:
            continue
        parts = []
        if courses:
            parts.append(", ".join(c.get("name", "?") for c in courses))
        if is_3cr and open_slots_by_term:
            slot_strs = [
                f"{slots} elective slot(s) in Year {yr} {sem} semester"
                for (yr, sem), slots in sorted(open_slots_by_term.items())
            ]
            parts.append("; ".join(slot_strs))
        if parts:
            notes.append(f"Eligible {label} course(s) available this semester: {', '.join(parts)}.")


def analyze_target_gpa(
    student_id: str,
    target_gpa: float,
    optimization: str = "maximize_credits",
) -> dict:
    """
    Scenario 3: analyze what combination of eligible courses and minimum
    grade per credit type is needed to reach target_gpa.

    optimization: "maximize_credits" (default) | "minimize_grade"
    """
    from eligibility import _get_student_row, get_student_context

    row = _get_student_row(student_id)
    if not row:
        return {"error": "Student not found."}

    courses_degrees = row.get("courses_degrees") or []
    current_gpa     = float(row.get("gpa") or 0.0)

    ctx          = get_student_context(student_id)
    program_name = ctx["program_name"]
    completed    = ctx["completed_courses"]
    earned       = ctx["total_hours_earned"]

    # Use authoritative DB values — compute_gpa_from_courses treats courses
    # with degree=None as F (0 pts) which under-counts existing_qp vs. DB GPA.
    existing_cr = earned
    existing_qp = current_gpa * existing_cr

    if existing_cr == 0:
        return {"error": "No completed courses found to establish a GPA basis."}

    if not program_name:
        return {"error": "Could not determine your program."}

    # Credit limit based on current GPA
    credit_limit = 15 if current_gpa < 2.0 else (18 if current_gpa < 3.0 else 21)

    # Infer upcoming semester from the most recently completed semester
    current_semester = infer_current_semester(courses_degrees)

    # Eligible not-completed CORE courses by credit type, filtered by semester
    courses_1cr = _get_eligible_not_completed(1, program_name, completed, earned, current_semester)
    courses_2cr = _get_eligible_not_completed(2, program_name, completed, earned, current_semester)
    courses_3cr = _get_eligible_not_completed(3, program_name, completed, earned, current_semester)
    courses_4cr = _get_eligible_not_completed(4, program_name, completed, earned, current_semester)

    # Elective slots: treat open slots as additional 3-credit placeholders
    remaining_slots = _build_remaining_elective_slots(program_name, completed)
    # Filter to same semester type as current_semester (any year level)
    if current_semester:
        open_slots_by_term = {
            (yr, sem): slots
            for (yr, sem), slots in remaining_slots.items()
            if sem == current_semester and slots > 0
        }
    else:
        open_slots_by_term = {}
    open_slots = sum(open_slots_by_term.values())

    count_1cr = len(courses_1cr)
    count_2cr = len(courses_2cr)
    count_3cr = len(courses_3cr) + open_slots   # elective slots counted as 3-cr mandatory
    count_4cr = len(courses_4cr)

    # Find best combination
    combo = _find_combination(
        count_1cr, count_2cr, count_3cr, count_4cr,
        credit_limit, existing_cr, existing_qp,
        target_gpa, optimization,
    )
    n1, n2, n3, n4 = combo["n1"], combo["n2"], combo["n3"], combo["n4"]
    T          = combo["total_credits"]
    Q_needed   = combo["Q_needed"]
    achievable = combo["achievable"]
    max_gpa    = combo["max_achievable_gpa"]

    # No eligible courses available at all
    if T == 0:
        sem_note = f" in Semester {current_semester}" if current_semester else ""
        return {
            "current_gpa":  round(current_gpa, 3),
            "target_gpa":   target_gpa,
            "credit_limit": credit_limit,
            "error": (
                f"No eligible not-completed courses found{sem_note}. "
                "Consider completing prerequisites to unlock more courses."
            ),
        }

    combination_info = {
        "n1_one_credit":   n1,
        "n2_two_credit":   n2,
        "n3_three_credit": n3,
        "n4_four_credit":  n4,
        "total_credits":   T,
    }

    def _available_courses_dict():
        return {
            "1_credit_options": [c.get("name") for c in courses_1cr],
            "2_credit_options": [c.get("name") for c in courses_2cr],
            "3_credit_options": {
                "courses": [c.get("name") for c in courses_3cr],
                "elective_slots": [
                    {"year": yr, "semester": sem, "available_slots": slots}
                    for (yr, sem), slots in sorted(open_slots_by_term.items())
                ],
            },
            "4_credit_options": [c.get("name") for c in courses_4cr],
        }

    # ── Already at / above target — show maintenance minimum ─────────────────
    if current_gpa >= target_gpa:
        Q_maint = target_gpa * (existing_cr + T) - existing_qp
        floors  = _compute_type_floors(Q_maint, T, n1, n2, n3, n4)
        p_raw   = Q_maint / T if T else 0.0
        grade_m, pct_m, _ = points_to_grade_requirement(max(0.0, p_raw))

        advisor_notes = [
            f"Your current GPA ({current_gpa:.3f}) already meets your target ({target_gpa}). "
            f"With {T} new credit(s), score at least {pct_m}% ({grade_m}) overall "
            f"to maintain GPA ≥ {target_gpa}.",
            "⚠️ All credit-type minimums must be met simultaneously — "
            "if any type falls below its floor, the others cannot compensate "
            "(they would need to exceed 100%).",
        ]
        _advisor_course_notes(advisor_notes, n1, n2, n3, n4,
                              courses_1cr, courses_2cr, courses_3cr, courses_4cr,
                              open_slots_by_term)

        return {
            "current_gpa":       round(current_gpa, 3),
            "target_gpa":        target_gpa,
            "credit_limit":      credit_limit,
            "already_at_target": True,
            "combination":       combination_info,
            "available_courses": _available_courses_dict(),
            "type_floors":       floors,
            "current_semester":  current_semester,
            "advisor_notes":     advisor_notes,
        }

    # ── Target not achievable — show max possible GPA ─────────────────────────
    if not achievable:
        advisor_notes = [
            f"Reaching a GPA of {target_gpa} is not possible this semester "
            f"with the available {T} credit(s).",
            f"Maximum achievable GPA this semester: {max_gpa:.3f}. "
            f"Aim for this — next semester you can continue toward {target_gpa}.",
        ]
        _advisor_course_notes(advisor_notes, n1, n2, n3, n4,
                              courses_1cr, courses_2cr, courses_3cr, courses_4cr,
                              open_slots_by_term)

        return {
            "current_gpa":        round(current_gpa, 3),
            "target_gpa":         target_gpa,
            "credit_limit":       credit_limit,
            "achievable":         False,
            "max_achievable_gpa": max_gpa,
            "combination":        combination_info,
            "available_courses":  _available_courses_dict(),
            "current_semester":   current_semester,
            "advisor_notes":      advisor_notes,
        }

    # ── Achievable — compute per-type floors ──────────────────────────────────
    floors = _compute_type_floors(Q_needed, T, n1, n2, n3, n4)

    advisor_notes = [
        "⚠️ All credit-type minimums must be met simultaneously — "
        "if any type falls below its floor, the others cannot compensate "
        "(they would need to exceed 100%).",
    ]
    _advisor_course_notes(advisor_notes, n1, n2, n3, n4,
                          courses_1cr, courses_2cr, courses_3cr, courses_4cr,
                          open_slots_by_term)

    return {
        "current_gpa":        round(current_gpa, 3),
        "target_gpa":         target_gpa,
        "credit_limit":       credit_limit,
        "achievable":         True,
        "max_achievable_gpa": max_gpa,
        "combination":        combination_info,
        "available_courses":  _available_courses_dict(),
        "type_floors":        floors,
        "current_semester":   current_semester,
        "advisor_notes":      advisor_notes,
    }
