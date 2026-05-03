"""
planning.py — Student Course Planning Function
===============================================
Fully automated course planner. No print() or input() calls.
Returns a dict with planned_courses, advisor_notes, and summary fields.

Stages:
  1. Previous terms (same semester type) — backlog of missed mandatory courses
  2. Current year / semester
  3. Future years (same semester), year+1 → Year 4
  4. Fill leftover 1-2 credits from closest future year

Overflow resolution removes courses by ascending dependent count, prioritising
courses that do not close anything in the upcoming term(s) first.
Elective picks always have 0 effective dependents → removed before mandatory.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# get_elective_slots
# ─────────────────────────────────────────────────────────────────────────────

def get_elective_slots(track, year, semester):
    """Number of elective slots for a given track, year, and semester."""
    track = track.lower()

    if isinstance(year, str):
        year_map = {
            'first year': 4, 'second year': 3, 'third year': 2, 'fourth year': 1,
            'first': 4, 'second': 3, 'third': 2, 'fourth': 1,
            '1': 4, '2': 3, '3': 2, '4': 1
        }
        year_num = year_map.get(year.lower(), int(year) if year.isdigit() else 0)
    else:
        year_num = 5 - int(year)   # 1→4, 2→3, 3→2, 4→1

    if 'artificial intelligence' in track:
        if year_num == 1 and semester == 'First':    return 2
        elif year_num == 1 and semester == 'Second': return 3
        else:                                         return 0

    elif 'software' in track or 'data science' in track:
        if year_num == 2 and semester == 'Second':   return 1
        elif year_num == 1 and semester == 'First':  return 2
        elif year_num == 1 and semester == 'Second': return 2
        else:                                         return 0

    return 0


# ─────────────────────────────────────────────────────────────────────────────
# get_prerequisites — alias for get_course_dependencies (prereqs only)
# ─────────────────────────────────────────────────────────────────────────────

def get_prerequisites(course_name, program_name):
    from neo4j_course_functions import get_course_dependencies
    return get_course_dependencies(course_name, program_name, dependents=False)


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _next_term(year: int, sem: str) -> Tuple[int, str]:
    """Return the term that immediately follows (year, sem)."""
    return (year, 'Second') if sem == 'First' else (year + 1, 'First')


def _check_prereqs(
    course_name: str,
    track: str,
    completed: Set[str],
    earned: int,
) -> bool:
    prereqs = get_prerequisites(course_name, track)
    for p in prereqs.get('prerequisites', []):
        if 'Required_Credit_Hours' in p:
            if earned < int(p['Required_Credit_Hours']):
                return False
        else:
            if p.get('name', '') not in completed:
                return False
    return True


def _get_term_mandatory(
    year: int,
    sem: str,
    track: str,
    completed: Set[str],
    planned: Set[str],
    earned: int,
) -> List[dict]:
    """Eligible mandatory courses for a term, excluding completed and already planned."""
    from neo4j_course_functions import get_courses_by_term
    term_data = get_courses_by_term(year, sem, track)
    result: List[dict] = []
    if not term_data:
        return result
    for _, semesters in term_data.items():
        for _, programs in semesters.items():
            if track in programs:
                for course in programs[track]:
                    if course.get('course_type') != 'mandatory':
                        continue
                    name = course['course_name']
                    if name in completed or name in planned:
                        continue
                    if _check_prereqs(name, track, completed, earned):
                        result.append(course)
    return result


def _get_term_all_names(year: int, sem: str, track: str) -> Set[str]:
    """All course names scheduled in a given term."""
    from neo4j_course_functions import get_courses_by_term
    term_data = get_courses_by_term(year, sem, track)
    names: Set[str] = set()
    if term_data:
        for _, semesters in term_data.items():
            for _, programs in semesters.items():
                if track in programs:
                    for course in programs[track]:
                        names.add(course['course_name'])
    return names


def _get_dependents(course_name: str, track: str) -> Set[str]:
    """Names of courses that require course_name as a prerequisite."""
    from neo4j_course_functions import get_course_dependencies
    result = get_course_dependencies(course_name, track, prereq=False, dependents=True)
    if not isinstance(result, dict):
        return set()
    names: Set[str] = set()
    for d in result.get('dependents', []):
        if isinstance(d, dict):
            n = d.get('name', '')
            if n:
                names.add(n)
        elif isinstance(d, str):
            names.add(d)
    return names


def _resolve_overflow(
    courses: List[dict],
    avail: int,
    track: str,
    advisor_notes: List[str],
    priority_term_sets: List[Set[str]],
) -> Tuple[List[dict], List[dict]]:
    """
    Remove courses until sum(credit_hours) <= avail.
    Returns (remaining, removed).

    Removal order:
      For each set in priority_term_sets:
        - Candidates: courses whose dependents don't intersect the set
        - Sort candidates by ascending effective dependents count
        - Remove smallest first until resolved or candidates exhausted
      Final pass: all remaining courses, ascending dependents
    Elective picks (is_elective_pick=True) always score 0 → removed first.
    """
    remaining = list(courses)
    removed:   List[dict] = []

    def total_cr(lst: List[dict]) -> int:
        return sum(c['credit_hours'] for c in lst)

    def eff_deps(c: dict) -> int:
        if c.get('is_elective_pick'):
            return 0
        return len(_get_dependents(c['course_name'], track))

    # Priority layers
    for term_names in priority_term_sets:
        if total_cr(remaining) <= avail:
            break
        candidates: List[Tuple[dict, int]] = []
        for c in remaining:
            closes = set() if c.get('is_elective_pick') else _get_dependents(c['course_name'], track)
            if not closes.intersection(term_names):
                candidates.append((c, eff_deps(c)))
        candidates.sort(key=lambda x: x[1])
        for c, d in candidates:
            if total_cr(remaining) <= avail:
                break
            remaining.remove(c)
            removed.append(c)
            advisor_notes.append(
                f"Removed '{c['course_name']}' to fit credit limit "
                f"(dependents: {d}, does not close priority term courses)"
            )

    # Final pass: all remaining
    if total_cr(remaining) > avail:
        with_deps = [(c, eff_deps(c)) for c in remaining]
        with_deps.sort(key=lambda x: x[1])
        for c, d in with_deps:
            if total_cr(remaining) <= avail:
                break
            remaining.remove(c)
            removed.append(c)
            advisor_notes.append(
                f"Removed '{c['course_name']}' to fit credit limit (dependents: {d})"
            )

    return remaining, removed


_YEAR_LEVEL_MAP = {
    1: 'First Year', 2: 'Second Year', 3: 'Third Year', 4: 'Fourth Year',
}


def _stage4_candidates(
    avail: int,
    year: int,
    sem: str,
    track: str,
    completed: Set[str],
    planned: Set[str],
    earned: int,
) -> List[dict]:
    """
    Mandatory courses from a specific future term with credit_hours <= avail,
    excluding completed/planned and courses failing prereqs.
    Uses filter_courses() with year_level + semester — planning-internal params.
    """
    from neo4j_course_functions import filter_courses as _fc
    raw = _fc(
        filters={'credit_hours': {'<=': avail}},
        course_types=['mandatory'],
        return_fields=['name', 'code', 'credit_hours'],
        program_name=track,
        year_level=_YEAR_LEVEL_MAP.get(year),
        semester=sem,
    ) or []
    result: List[dict] = []
    for c in raw:
        name = (c.get('name') or '').lower()
        if not name or name in completed or name in planned:
            continue
        if not _check_prereqs(name, track, completed, earned):
            continue
        result.append({
            'course_name':  name,
            'credit_hours': c.get('credit_hours', 0),
            'course_type':  'mandatory',
            'course_code':  c.get('code', ''),
        })
    return result


def _build_result(
    student_id:    str,
    year:          int,
    semester:      str,
    track:         str,
    original_avail: int,
    suggest_courses: List[dict],
    advisor_notes:  List[str],
) -> dict:
    planned_credits = sum(c['credit_hours'] for c in suggest_courses)
    return {
        'student_id':       student_id,
        'year':             year,
        'semester':         semester,
        'track':            track,
        'available_credits': original_avail,
        'planned_credits':  planned_credits,
        'planned_courses':  suggest_courses,
        'advisor_notes':    advisor_notes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# planning()
# ─────────────────────────────────────────────────────────────────────────────

def planning(student_id: str, supabase_client) -> Optional[dict]:
    """
    Build a fully automated course plan for the student.

    Args:
        student_id:      The student's ID.
        supabase_client: Supabase client instance (unused directly; passed for
                         compatibility with planning_service).

    Returns:
        dict with planned_courses, advisor_notes, and summary fields,
        or None if the student record is missing.
    """
    from neo4j_course_functions import get_all_electives_by_program
    from eligibility import get_student_context
    from recommendation_service import recommend_electives

    advisor_notes:   List[str]  = []
    suggest_courses: List[dict] = []

    # ── Student data ──────────────────────────────────────────────────────────
    ctx = get_student_context(student_id)
    if not ctx or not ctx.get('program_name'):
        return None

    completed: Set[str] = set(ctx['completed_courses'])
    track:     str      = ctx['program_name']
    earned:    int      = ctx['total_hours_earned']
    gpa:       float    = ctx['gpa'] or 0.0
    uni_year:  int      = ctx['university_year']
    sem_num:   int      = ctx['current_term'] or 1
    semester:  str      = 'First' if sem_num == 1 else 'Second'

    # ── Credit limit ──────────────────────────────────────────────────────────
    if gpa >= 3.0:
        original_avail = 21
    elif gpa >= 2.0:
        original_avail = 18
    else:
        original_avail = 15

    avail_credits: int = original_avail
    advisor_notes.append(
        f"Planning Year {uni_year}, {semester} Semester | "
        f"Track: {track} | GPA: {gpa} → {avail_credits}-credit limit"
    )

    # ── Shared elective pool (shrinks as stages consume electives) ────────────
    all_electives = get_all_electives_by_program(track)
    elective_pool: List[dict] = [
        e for e in all_electives
        if e['course_name'] not in completed
        and _check_prereqs(e['course_name'], track, completed, earned)
    ]

    # ── Pre-compute remaining elective slots (total minus already completed) ──
    # Build ordered list of all terms that have slots (Year 1→4, First→Second).
    _slot_terms = [
        (y, s)
        for y in range(1, 5)
        for s in ['First', 'Second']
        if get_elective_slots(track, y, s) > 0
    ]
    remaining_slots: Dict[Tuple[int, str], int] = {
        (y, s): get_elective_slots(track, y, s) for (y, s) in _slot_terms
    }
    # Assign each completed elective to the first slot with remaining capacity.
    all_elective_names: Set[str] = {e['course_name'] for e in all_electives}
    for _cname in completed:
        if _cname in all_elective_names:
            for _ys in _slot_terms:
                if remaining_slots[_ys] > 0:
                    remaining_slots[_ys] -= 1
                    break

    # ── Helpers ───────────────────────────────────────────────────────────────

    def planned_names() -> Set[str]:
        return {c['course_name'] for c in suggest_courses}

    def pick_electives(n: int) -> List[dict]:
        """Pick n electives from pool using recommendation scoring."""
        if n <= 0:
            return []
        available = len(elective_pool)
        if available == 0:
            advisor_notes.append(f"No eligible electives available for {n} slot(s)")
            return []
        to_pick = min(n, available)
        if available < n:
            advisor_notes.append(
                f"Only {available} eligible elective(s) for {n} slot(s) — "
                f"{n - available} slot(s) could not be filled"
            )
        chosen = recommend_electives(
            student_id, track, top_n=to_pick, eligible_electives=list(elective_pool)
        )
        for c in chosen:
            elective_pool[:] = [e for e in elective_pool if e['course_name'] != c['course_name']]
            advisor_notes.append(f"Selected elective '{c['course_name']}' based on preferences")
        return [{**c, 'course_type': 'elective', 'is_elective_pick': True} for c in chosen]

    def return_to_pool(removed: List[dict]) -> None:
        """Put removed elective picks back in the pool for later stages."""
        for c in removed:
            if c.get('is_elective_pick'):
                base = {k: v for k, v in c.items() if k not in ('is_elective_pick',)}
                elective_pool.append(base)

    def process_stage(
        mandatory:          List[dict],
        elective_slots:     int,
        priority_term_sets: List[Set[str]],
        label:              str,
    ) -> bool:
        """
        Add eligible courses for one planning stage.
        Returns True if planning should terminate (credit limit reached or overflow).
        """
        nonlocal avail_credits

        if not mandatory and elective_slots == 0:
            return False

        elective_picks = pick_electives(elective_slots)
        candidates     = list(mandatory) + elective_picks
        total          = sum(c['credit_hours'] for c in candidates)

        if total == 0:
            return False

        if total <= avail_credits:
            suggest_courses.extend(candidates)
            avail_credits -= total
            note = f"{label}: added {len(mandatory)} mandatory course(s)"
            if elective_picks:
                note += f" + {len(elective_picks)} elective(s)"
            advisor_notes.append(note)
            if avail_credits == 0:
                advisor_notes.append("Credit limit reached.")
                return True
            return False

        else:  # overflow
            advisor_notes.append(
                f"{label}: {total} credits exceeds {avail_credits} available — resolving overflow"
            )
            resolved, removed = _resolve_overflow(
                candidates, avail_credits, track, advisor_notes, priority_term_sets
            )
            return_to_pool(removed)
            suggest_courses.extend(resolved)
            avail_credits -= sum(c['credit_hours'] for c in resolved)
            return True  # always terminate on overflow

    # ── Stage 1: Previous terms (same semester type) ──────────────────────────
    prev_mandatory:      List[dict] = []
    prev_elective_slots: int        = 0

    for y in range(1, uni_year + 1):
        for s in ['First', 'Second']:
            if y == uni_year and s == semester:
                break
            if s != semester:
                continue
            prev_mandatory.extend(
                _get_term_mandatory(y, s, track, completed, planned_names(), earned)
            )
            prev_elective_slots += remaining_slots.get((y, s), 0)

    if prev_mandatory or prev_elective_slots > 0:
        cur_names          = _get_term_all_names(uni_year, semester, track)
        next_y, next_s     = _next_term(uni_year, semester)
        next_names         = _get_term_all_names(next_y, next_s, track)
        if process_stage(
            prev_mandatory, prev_elective_slots,
            [cur_names, next_names],
            "Previous terms backlog",
        ):
            return _build_result(student_id, uni_year, semester, track,
                                  original_avail, suggest_courses, advisor_notes)

    # ── Stage 2: Current term ─────────────────────────────────────────────────
    cur_mandatory      = _get_term_mandatory(
        uni_year, semester, track, completed, planned_names(), earned
    )
    cur_elective_slots = remaining_slots.get((uni_year, semester), 0)

    if cur_mandatory or cur_elective_slots > 0:
        next_y, next_s = _next_term(uni_year, semester)
        next_names     = _get_term_all_names(next_y, next_s, track)
        if process_stage(
            cur_mandatory, cur_elective_slots,
            [next_names],
            f"Year {uni_year} {semester} Semester",
        ):
            return _build_result(student_id, uni_year, semester, track,
                                  original_avail, suggest_courses, advisor_notes)

    # ── Stage 3: Future years (same semester), uni_year+1 → 4 ────────────────
    for future_y in range(uni_year + 1, 5):
        fut_mandatory      = _get_term_mandatory(
            future_y, semester, track, completed, planned_names(), earned
        )
        fut_elective_slots = remaining_slots.get((future_y, semester), 0)

        if not fut_mandatory and fut_elective_slots == 0:
            continue

        if process_stage(
            fut_mandatory, fut_elective_slots,
            [],
            f"Year {future_y} {semester} Semester (advanced)",
        ):
            return _build_result(student_id, uni_year, semester, track,
                                  original_avail, suggest_courses, advisor_notes)

    # ── Stage 4: Fill leftover 1-2 credits from closest future year ───────────
    if 0 < avail_credits <= 2:
        pnames = planned_names()
        for future_y in range(uni_year + 1, 5):
            candidates = _stage4_candidates(
                avail_credits, future_y, semester, track, completed, pnames, earned
            )
            if not candidates:
                continue

            if avail_credits == 2:
                two = [c for c in candidates if c.get('credit_hours') == 2]
                if two:
                    suggest_courses.append(two[0])
                    advisor_notes.append(
                        f"Filled 2 remaining credits with '{two[0]['course_name']}' (Year {future_y})"
                    )
                    break
                one = [c for c in candidates if c.get('credit_hours') == 1]
                if len(one) >= 2:
                    suggest_courses.extend(one[:2])
                    advisor_notes.append(
                        f"Filled 2 remaining credits with 2 one-credit courses (Year {future_y})"
                    )
                    break
                if one:
                    suggest_courses.append(one[0])
                    advisor_notes.append(
                        f"Filled 1 of 2 remaining credits with '{one[0]['course_name']}' (Year {future_y})"
                    )
                    break

            elif avail_credits == 1:
                one = [c for c in candidates if c.get('credit_hours') == 1]
                if one:
                    suggest_courses.append(one[0])
                    advisor_notes.append(
                        f"Filled 1 remaining credit with '{one[0]['course_name']}' (Year {future_y})"
                    )
                    break

    return _build_result(
        student_id, uni_year, semester, track,
        original_avail, suggest_courses, advisor_notes,
    )
