"""
planning.py — Student Course Planning Function
===============================================
Fully automated course planner. No print() or input() calls.
Returns a dict with planned_courses, advisor_notes, course_details, and summary fields.

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
# Unspecialized-student constants
# ─────────────────────────────────────────────────────────────────────────────

PROXY_TRACK = "artificial intelligence & machine learning"

# Courses shared across ALL programs per term (Year 3/4).
# Names use the PROXY_TRACK (AIM) canonical form so Neo4j queries are consistent.
SHARED_COURSES_BY_TERM: Dict[Tuple[int, str], List[str]] = {
    (3, 'First'):  ['current social issues in egypt', 'artificial intelligence', 'software engineering 1'],
    (3, 'Second'): ['machine learning'],
    (4, 'First'):  ['professional ethics', 'graduation project (1)'],
    (4, 'Second'): ['entrepreneurship', 'graduation project (2)'],
}


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


def _get_missing_prereqs(
    course_name: str,
    track: str,
    completed: Set[str],
    earned: int,
) -> List[str]:
    """Return human-readable names of prerequisites the student hasn't met yet."""
    prereqs = get_prerequisites(course_name, track)
    missing: List[str] = []
    for p in prereqs.get('prerequisites', []):
        if 'Required_Credit_Hours' in p:
            req = int(p['Required_Credit_Hours'])
            if earned < req:
                missing.append(f"{req} earned credit hours (you have {earned})")
        else:
            pname = (p.get('name') or '').lower()
            if pname and pname not in completed:
                missing.append(pname)
    return missing


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
                    name = course['course_name'].lower()
                    if name in completed or name in planned:
                        continue
                    if _check_prereqs(name, track, completed, earned):
                        result.append({**course, 'course_name': name})
    return result


def _get_all_term_mandatory_raw(year: int, sem: str, track: str) -> List[dict]:
    """All mandatory courses for a term regardless of prereq status or completion."""
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
                    name = course['course_name'].lower()
                    result.append({**course, 'course_name': name})
    return result


def _get_shared_term_mandatory(
    year: int,
    sem: str,
    completed: Set[str],
    planned: Set[str],
    earned: int,
) -> List[dict]:
    """
    Eligible shared courses for a Y3/Y4 term for unspecialized students.
    Fetches credit_hours live from Neo4j (PROXY_TRACK); prereqs checked dynamically.
    """
    from neo4j_course_functions import get_course_info
    shared_names = SHARED_COURSES_BY_TERM.get((year, sem), [])
    result: List[dict] = []
    for name in shared_names:
        if name in completed or name in planned:
            continue
        if not _check_prereqs(name, PROXY_TRACK, completed, earned):
            continue
        try:
            info_list = get_course_info(name, PROXY_TRACK)
            info = info_list[0] if info_list else {}
        except Exception:
            info = {}
        credit_hours = info.get('credit_hours', 0)
        if not credit_hours:
            continue
        result.append({
            'course_name':  name,
            'credit_hours': credit_hours,
            'course_type':  'mandatory',
            'course_code':  info.get('code', ''),
        })
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
                        names.add(course['course_name'].lower())
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

    # Pass 0 — Protect combined union of all priority terms.
    # Removes courses that don't unlock anything in ANY priority term before
    # the individual-term passes start. Only active when 2+ sets are given
    # (Stage 1 backlog: cur_names + next_names).
    if len(priority_term_sets) > 1 and total_cr(remaining) > avail:
        combined = set().union(*priority_term_sets)
        pass0_candidates: List[Tuple[dict, int]] = []
        for c in remaining:
            closes = set() if c.get('is_elective_pick') else _get_dependents(c['course_name'], track)
            if not closes.intersection(combined):
                pass0_candidates.append((c, eff_deps(c)))
        pass0_candidates.sort(key=lambda x: x[1])
        for c, d in pass0_candidates:
            if total_cr(remaining) <= avail:
                break
            remaining.remove(c)
            removed.append(c)
            advisor_notes.append(
                f"Removed '{c['course_name']}' to fit credit limit "
                f"(dependents: {d}, does not close any combined priority term courses)"
            )

    # Priority layers (Pass 1, Pass 2, …)
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
    student_id:      str,
    year:            int,
    semester:        str,
    track:           str,
    original_avail:  int,
    suggest_courses: List[dict],
    advisor_notes:   List[str],
    course_details:  Dict[str, dict],
) -> dict:
    planned_credits = sum(c['credit_hours'] for c in suggest_courses)
    return {
        'student_id':        student_id,
        'year':              year,
        'semester':          semester,
        'track':             track,
        'available_credits': original_avail,
        'planned_credits':   planned_credits,
        'planned_courses':   suggest_courses,
        'advisor_notes':     advisor_notes,
        'course_details':    course_details,
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
        dict with planned_courses, advisor_notes, course_details, and summary fields,
        or None if the student record is missing.
    """
    from neo4j_course_functions import get_all_electives_by_program, get_course_info
    from eligibility import get_student_context
    from recommendation_service import (
        recommend_electives,
        merge_preferences,
        _top_matching_categories,
        ELECTIVE_CATALOGUES,
        _NEO4J_PROGRAM,
    )

    advisor_notes:   List[str]  = []
    suggest_courses: List[dict] = []

    # ── Student data ──────────────────────────────────────────────────────────
    ctx = get_student_context(student_id)
    if not ctx:
        return None

    completed:  Set[str] = set(ctx['completed_courses'])
    earned:     int      = ctx['total_hours_earned']
    gpa:        float    = ctx['gpa'] or 0.0
    uni_year:   int      = ctx.get('university_year') or 0
    sem_num:    int      = ctx['current_term'] or 2
    semester:   str      = 'First' if sem_num == 1 else 'Second'

    # ── Failed courses (for tagging backlog and classifying unplanned courses) ─
    failed_courses_set: Set[str] = ctx.get('failed_courses', set())

    # ── Track / unspecialized detection ──────────────────────────────────────
    raw_track:        Optional[str] = ctx.get('program_name')
    is_unspecialized: bool          = False

    if not raw_track:
        if uni_year in (3, 4):
            # Data error — Year 3/4 students must have a declared track
            return {
                'student_id':        student_id,
                'year':              uni_year,
                'semester':          semester,
                'track':             '',
                'available_credits': 0,
                'planned_credits':   0,
                'planned_courses':   [],
                'advisor_notes': [
                    "⚠️ Your specialization track is missing from our records. "
                    "Year 3 and Year 4 courses are program-specific and cannot be "
                    "planned without a declared track. Please contact the registrar "
                    "to have your track assigned before we can generate a plan for you."
                ],
                'course_details': {},
            }
        if not uni_year:
            return None  # incomplete record — cannot plan
        is_unspecialized = True
        track = PROXY_TRACK
    else:
        track = raw_track

    # Display label shown in the result dict and advisor notes (not used for queries)
    _display_track = "Pre-Specialization (Universal Curriculum)" if is_unspecialized else track

    # ── Internal state for note generation ───────────────────────────────────
    _termination_info:       dict       = {}   # set by process_stage on termination
    _stage_overflow_removed: List[dict] = []   # all courses dropped by overflow
    _has_backlog:  bool = False                # set after Stage 1 scan
    _has_current:  bool = False                # set after Stage 2 scan

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
        f"Track: {_display_track} | GPA: {gpa} → {avail_credits}-credit limit"
    )

    if is_unspecialized:
        advisor_notes.append(
            "📋 You have not declared a specialization track yet. This plan covers "
            "your universal Year 1–2 curriculum plus courses shared across all programs "
            "in Years 3–4. Course names reflect the standard curriculum and may vary "
            "slightly depending on your final track. Once you specialize, a full "
            "track-specific plan will be available."
        )

    # ── Student preference vector (for elective reasoning) ────────────────────
    _student_vec, _ = merge_preferences(student_id)
    _cat_key        = _NEO4J_PROGRAM.get(track.lower(), track.lower())
    _catalogue      = ELECTIVE_CATALOGUES.get(_cat_key, ELECTIVE_CATALOGUES.get(track.lower(), {}))

    # ── Shared elective pool (shrinks as stages consume electives) ────────────
    all_electives = get_all_electives_by_program(track)
    # Normalise names to lowercase so they match the completed set (which is also lowercase)
    for e in all_electives:
        e['course_name'] = e['course_name'].lower()
    elective_pool: List[dict] = [
        e for e in all_electives
        if e['course_name'] not in completed
        and _check_prereqs(e['course_name'], track, completed, earned)
    ]

    # ── Pre-compute remaining elective slots (total minus already completed) ──
    _slot_terms = [
        (y, s)
        for y in range(1, 5)
        for s in ['First', 'Second']
        if get_elective_slots(track, y, s) > 0
    ]
    remaining_slots: Dict[Tuple[int, str], int] = {
        (y, s): get_elective_slots(track, y, s) for (y, s) in _slot_terms
    }
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
        result = []
        for c in chosen:
            elective_pool[:] = [e for e in elective_pool if e['course_name'] != c['course_name']]
            # Compute top matching preference categories for this elective
            name     = c['course_name'].lower()
            profile  = _catalogue.get(name, {}).get('profile', {})
            if profile and _student_vec:
                top_cats = _top_matching_categories(_student_vec, profile, n=2)
                cat_str  = " & ".join(k.replace('_', ' ') for k in top_cats) if top_cats else "your academic interests"
            else:
                cat_str = "your academic interests"
            result.append({
                **c,
                'course_type':       'elective',
                'is_elective_pick':  True,
                '_stage':            'elective',
                '_preference_cats':  cat_str,
            })
        return result

    def return_to_pool(removed: List[dict]) -> None:
        """Put removed elective picks back in the pool for later stages."""
        for c in removed:
            if c.get('is_elective_pick'):
                base = {k: v for k, v in c.items()
                        if k not in ('is_elective_pick', '_stage', '_preference_cats')}
                elective_pool.append(base)

    def process_stage(
        mandatory:          List[dict],
        elective_slots:     int,
        priority_term_sets: List[Set[str]],
        label:              str,
        stage_type:         str = '',
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

        def _tag(courses: List[dict]) -> List[dict]:
            for c in courses:
                if '_stage' not in c:
                    c['_stage'] = stage_type
            return courses

        if total <= avail_credits:
            suggest_courses.extend(_tag(candidates))
            avail_credits -= total
            if avail_credits == 0:
                _termination_info.update({'reason': 'exact', 'stage_type': stage_type})
                return True
            return False

        else:  # overflow — redirect _resolve_overflow's internal notes to a sink
            _avail_before   = avail_credits
            _overflow_sink: List[str] = []
            resolved, removed = _resolve_overflow(
                candidates, avail_credits, track, _overflow_sink, priority_term_sets
            )
            _stage_overflow_removed.extend(removed)
            return_to_pool(removed)
            suggest_courses.extend(_tag(resolved))
            avail_credits -= sum(c['credit_hours'] for c in resolved)
            _termination_info.update({
                'reason':       'overflow',
                'stage_type':   stage_type,
                'total_needed': total,
                'avail_before': _avail_before,
                'resolved':     list(resolved),
                'removed':      list(removed),
            })
            return True

    # ── Note generators (called after process_stage returns True) ─────────────

    def _generate_overflow_note() -> None:
        """Append a student-friendly explanation of the overflow decision."""
        total_needed = _termination_info.get('total_needed', 0)
        avail_before = _termination_info.get('avail_before', 0)
        resolved     = _termination_info.get('resolved', [])
        removed      = _termination_info.get('removed', [])
        stage_type   = _termination_info.get('stage_type', '')

        elec_removed = [c for c in removed if     c.get('is_elective_pick')]
        mand_removed = [c for c in removed if not c.get('is_elective_pick')]
        mand_kept    = [c for c in resolved if not c.get('is_elective_pick')]

        stage_labels = {
            'backlog': 'previous-semester backlog',
            'current': f'Year {uni_year} {semester} Semester (current term)',
            'future':  'advanced future courses',
        }
        stage_label = stage_labels.get(stage_type, stage_type)

        advisor_notes.append(
            f"⚠️ Credit limit situation ({stage_label}): you had "
            f"{total_needed} credits worth of eligible courses but only "
            f"{avail_before} credits were available in your allowance."
        )
        advisor_notes.append(
            "How the selection was made: Elective picks are always removed first "
            "because they do not block any required course. Among mandatory courses, "
            "those that unlock courses in your upcoming terms are protected first. "
            "The remaining are sorted by downstream impact — courses with fewer future "
            "dependencies are removed before those with more."
        )
        if elec_removed:
            names = ', '.join(c['course_name'].title() for c in elec_removed)
            advisor_notes.append(
                f"Elective(s) set aside this session: {names}. "
                f"They will be available again in your next planning session."
            )
        if mand_kept:
            names = ', '.join(c['course_name'].title() for c in mand_kept)
            advisor_notes.append(
                f"✅ Mandatory courses kept in your plan: {names}. "
                f"These were prioritised because they open doors to your upcoming "
                f"terms or have the most downstream dependencies."
            )
        if mand_removed:
            names = ', '.join(c['course_name'].title() for c in mand_removed)
            advisor_notes.append(
                f"🚫 Mandatory course(s) moved to your next session: {names}. "
                f"These had less immediate impact on your upcoming path and will "
                f"be given priority again in your next semester's plan."
            )

    def _generate_termination_note() -> None:
        """Append a note explaining what subsequent stages were blocked."""
        reason     = _termination_info.get('reason', '')
        stage_type = _termination_info.get('stage_type', '')

        if stage_type == 'backlog':
            if reason == 'overflow':
                advisor_notes.append(
                    f"📌 Because the credit limit was reached during backlog processing, "
                    f"your current-term courses (Year {uni_year}, {semester} Semester) "
                    f"and any advanced future courses could not be added this session. "
                    f"Clearing your backlog is the top priority right now."
                )
            else:
                advisor_notes.append(
                    f"📌 Your backlog courses filled your entire {original_avail}-credit "
                    f"allowance completely. No credit remained for your current-term courses "
                    f"(Year {uni_year}, {semester} Semester) or any advanced future courses. "
                    f"Once you clear this backlog, those will take centre stage."
                )

        elif stage_type == 'current':
            backlog_part = " backlog and" if _has_backlog else ""
            if reason == 'overflow':
                advisor_notes.append(
                    f"📌 The credit limit was reached while planning your current-term "
                    f"courses (Year {uni_year}, {semester} Semester). "
                    f"No advanced future courses could be pulled forward this session."
                )
            else:
                advisor_notes.append(
                    f"📌 Your{backlog_part} current-term courses "
                    f"(Year {uni_year}, {semester} Semester) filled your entire "
                    f"{original_avail}-credit allowance. "
                    f"No advanced future courses were added this session."
                )

        elif stage_type == 'future':
            if reason == 'overflow':
                advisor_notes.append(
                    f"📌 The credit limit was reached while pulling forward advanced "
                    f"future courses. Your plan is now complete within your "
                    f"{original_avail}-credit allowance."
                )
            else:
                advisor_notes.append(
                    f"📌 Your {original_avail}-credit allowance is fully used — "
                    f"your plan is complete for this semester."
                )

        else:
            advisor_notes.append(
                f"📌 Credit allowance of {original_avail} credits fully used."
            )

    # ── Unplanned course classifier ───────────────────────────────────────────

    def _add_unplanned_notes() -> None:
        """
        Scan all past and current terms to classify every unplanned mandatory
        course and append student-friendly notes to advisor_notes.
        """
        def _term_idx(y: int, s: str) -> int:
            return y * 2 + (0 if s == 'First' else 1)

        current_idx            = _term_idx(uni_year, semester)
        planned                = planned_names()
        overflow_removed_names = {c['course_name'] for c in _stage_overflow_removed}

        failed_wrong_sem:   List[dict] = []
        eligible_wrong_sem: List[dict] = []
        blocked_wrong_sem:  List[dict] = []
        blocked_same_sem:   List[dict] = []

        # ── Scan all previous terms (any semester type) ───────────────────────
        for y in range(1, uni_year + 1):
            for s in ['First', 'Second']:
                if _term_idx(y, s) >= current_idx:
                    break  # reached current term — stop inner loop

                raw_courses = _get_all_term_mandatory_raw(y, s, track)
                for course in raw_courses:
                    name = course['course_name']
                    if name in completed or name in planned:
                        continue
                    if name in overflow_removed_names:
                        continue  # already explained in the overflow note

                    is_failed   = name in failed_courses_set
                    is_same_sem = (s == semester)
                    prereqs_ok  = _check_prereqs(name, track, completed, earned)

                    if is_same_sem:
                        # Was a candidate for the backlog stage.
                        # If prereqs were met it should have been planned or overflow-removed.
                        # Only reach here if prereqs are NOT met.
                        if not prereqs_ok:
                            missing = _get_missing_prereqs(name, track, completed, earned)
                            blocked_same_sem.append({
                                'course_name':       name,
                                'year':              y,
                                'semester':          s,
                                'missing_prereqs':   missing,
                                'previously_failed': is_failed,
                            })
                    else:
                        # Different semester type — cannot be planned this term at all.
                        if is_failed:
                            failed_wrong_sem.append({
                                'course_name': name, 'year': y, 'semester': s,
                            })
                        elif prereqs_ok:
                            eligible_wrong_sem.append({
                                'course_name': name, 'year': y, 'semester': s,
                            })
                        else:
                            missing = _get_missing_prereqs(name, track, completed, earned)
                            blocked_wrong_sem.append({
                                'course_name':     name,
                                'year':            y,
                                'semester':        s,
                                'missing_prereqs': missing,
                            })

        # ── Scan current term for prereq-blocked courses ───────────────────────
        current_term_blocked: List[dict] = []
        cur_all_raw = _get_all_term_mandatory_raw(uni_year, semester, track)
        for course in cur_all_raw:
            name = course['course_name']
            if name in completed or name in planned:
                continue
            if name in overflow_removed_names:
                continue
            if not _check_prereqs(name, track, completed, earned):
                missing = _get_missing_prereqs(name, track, completed, earned)
                current_term_blocked.append({
                    'course_name':     name,
                    'year':            uni_year,
                    'semester':        semester,
                    'missing_prereqs': missing,
                })

        # ── Generate notes ────────────────────────────────────────────────────

        if failed_wrong_sem:
            advisor_notes.append(
                f"── Courses you previously failed that are not offered in the "
                f"{semester} semester ──"
            )
            for entry in failed_wrong_sem:
                name = entry['course_name'].title()
                y, s = entry['year'], entry['semester']
                advisor_notes.append(
                    f"❌ You previously failed {name} (Year {y}, {s} Semester). "
                    f"This course is only offered in the {s} semester — not your "
                    f"current {semester} semester, so it cannot be planned right now. "
                    f"Your options: check whether it runs in summer by visiting the "
                    f"Community → Q&A tab in the app (the team replies in under 12 hours), "
                    f"or take it next time the {s} semester comes around."
                )

        if eligible_wrong_sem:
            advisor_notes.append(
                f"── Courses you haven't completed yet, prerequisites met, "
                f"but offered in the other semester ──"
            )
            for entry in eligible_wrong_sem:
                name = entry['course_name'].title()
                y, s = entry['year'], entry['semester']
                advisor_notes.append(
                    f"📅 You haven't completed {name} yet (Year {y}, {s} Semester) — "
                    f"and your prerequisites ARE met, so you are fully eligible. "
                    f"However, it is only offered in the {s} semester, not your current "
                    f"{semester} semester. "
                    f"You can check summer availability via the Community → Q&A tab "
                    f"(reply in under 12 hours), or take it next {s} semester."
                )

        if blocked_wrong_sem:
            advisor_notes.append(
                f"── Courses not yet complete, prerequisites missing, "
                f"and offered in the other semester ──"
            )
            for entry in blocked_wrong_sem:
                name    = entry['course_name'].title()
                y, s    = entry['year'], entry['semester']
                prereqs = (', '.join(p.title() for p in entry['missing_prereqs'])
                           or 'certain prerequisites')
                advisor_notes.append(
                    f"📅 You haven't completed {name} yet (Year {y}, {s} Semester). "
                    f"You are not eligible yet — you are still missing: {prereqs}. "
                    f"On top of that, it is only offered in the {s} semester "
                    f"(not your current {semester} semester), so it could not be planned "
                    f"right now regardless. Focus on completing the prerequisites first."
                )

        if blocked_same_sem:
            advisor_notes.append(
                f"── Courses from previous {semester} semesters that you can't take "
                f"yet due to missing prerequisites ──"
            )
            for entry in blocked_same_sem:
                name       = entry['course_name'].title()
                y, s       = entry['year'], entry['semester']
                prereqs    = (', '.join(p.title() for p in entry['missing_prereqs'])
                              or 'certain prerequisites')
                was_failed = entry['previously_failed']
                if was_failed:
                    advisor_notes.append(
                        f"🔒 You previously failed {name} (Year {y}, {s} Semester). "
                        f"You are not eligible to retake it this term because you are "
                        f"still missing: {prereqs}. "
                        f"Complete those courses first to unlock it."
                    )
                else:
                    advisor_notes.append(
                        f"🔒 You haven't completed {name} yet (Year {y}, {s} Semester). "
                        f"You are not eligible this term because you are still missing: "
                        f"{prereqs}. Complete those prerequisites first to unlock it."
                    )

        if current_term_blocked:
            advisor_notes.append(
                f"── Current-term courses (Year {uni_year}, {semester} Semester) "
                f"you can't take yet due to missing prerequisites ──"
            )
            for entry in current_term_blocked:
                name    = entry['course_name'].title()
                prereqs = (', '.join(p.title() for p in entry['missing_prereqs'])
                           or 'certain prerequisites')
                advisor_notes.append(
                    f"⛔ {name} is part of your Year {uni_year}, {semester} Semester "
                    f"curriculum, but you cannot take it this term because you are still "
                    f"missing: {prereqs}. "
                    f"Complete those prerequisites to unlock it in a future term."
                )

        # ── Scan future terms (same semester type) for prereq-blocked courses ──
        future_blocked: List[dict] = []
        for future_y in range(uni_year + 1, 5):
            if is_unspecialized and future_y >= 3:
                # Only shared courses for unspecialized students
                for name in SHARED_COURSES_BY_TERM.get((future_y, semester), []):
                    if name in completed or name in planned or name in overflow_removed_names:
                        continue
                    if not _check_prereqs(name, track, completed, earned):
                        missing = _get_missing_prereqs(name, track, completed, earned)
                        future_blocked.append({
                            'course_name':     name,
                            'year':            future_y,
                            'semester':        semester,
                            'missing_prereqs': missing,
                        })
            else:
                for course in _get_all_term_mandatory_raw(future_y, semester, track):
                    name = course['course_name']
                    if name in completed or name in planned or name in overflow_removed_names:
                        continue
                    if not _check_prereqs(name, track, completed, earned):
                        missing = _get_missing_prereqs(name, track, completed, earned)
                        future_blocked.append({
                            'course_name':     name,
                            'year':            future_y,
                            'semester':        semester,
                            'missing_prereqs': missing,
                        })

        if future_blocked:
            advisor_notes.append(
                f"── Future {semester} semester courses you are not yet eligible for ──"
            )
            for entry in future_blocked:
                name    = entry['course_name'].title()
                y       = entry['year']
                prereqs = (', '.join(p.title() for p in entry['missing_prereqs'])
                           or 'certain prerequisites')
                advisor_notes.append(
                    f"🔒 {name} is scheduled for Year {y}, {semester} Semester — a future term. "
                    f"You are not yet eligible because you are still missing: {prereqs}. "
                    f"Work on completing these prerequisites now so the course is fully "
                    f"unlocked when that semester arrives."
                )

    def _finalize_with_unplanned() -> Dict[str, dict]:
        _add_unplanned_notes()
        return _finalize()

    # ── Post-stage enrichment ─────────────────────────────────────────────────

    def _finalize() -> Dict[str, dict]:
        """
        Batch-fetch course info from Neo4j and append per-course advisor notes.
        Each note explains WHY this student should take this course (personal,
        motivational), WHAT they will learn, and HOW it connects to their future.
        Returns course_details dict: {course_name: {description, motivation, code}}.
        """
        # Maps preference categories → career / study areas for elective notes
        _CAT_CAREER: Dict[str, str] = {
            'ai_ml':                  'AI and machine learning engineering',
            'programming':            'software development and application engineering',
            'visual_computing':       'computer vision, game development, and graphics',
            'networking_systems':     'IoT, embedded systems, and network engineering',
            'data_analysis':          'data analytics and business intelligence',
            'data_management':        'database engineering and data architecture',
            'software_engineering':   'software architecture and professional practice',
            'math':                   'algorithm design and technical research',
            'probability_statistics': 'data science, research, and statistical modeling',
            'language_text':          'natural language processing and conversational AI',
            'theory':                 'academic research and advanced algorithm design',
            'optimization':           'operations research and performance engineering',
        }

        course_details:   Dict[str, dict] = {}
        per_course_notes: List[str]        = []

        for c in suggest_courses:
            name  = c['course_name']
            stage = c.get('_stage', '')
            code  = c.get('course_code', '')
            cr    = c.get('credit_hours', '?')

            # Fetch description + motivation from Neo4j
            try:
                info_list = get_course_info(name, track)
                info      = info_list[0] if info_list else {}
            except Exception:
                info = {}

            desc = (info.get('description') or '').strip()
            mot  = (info.get('motivation')  or '').strip()

            course_details[name] = {'description': desc, 'motivation': mot, 'code': code}

            title    = name.title()
            code_tag = f" [{code}]" if code else ""
            what     = mot if mot else desc   # prefer the motivation (why-focused) over raw description

            if stage == 'backlog':
                from_y     = c.get('_from_year', '?')
                from_s     = c.get('_from_sem', '?')
                was_failed = c.get('_previously_failed', False)
                if was_failed:
                    history = (
                        f"You previously attempted and failed this course "
                        f"(Year {from_y}, {from_s} Semester). "
                        f"This is your opportunity to retake it and clear it from your record."
                    )
                else:
                    history = (
                        f"This mandatory course from Year {from_y}, {from_s} Semester "
                        f"was not completed at the time. "
                        f"Taking it now keeps your academic record clean and on track."
                    )
                note = (
                    f"📚 {title}{code_tag} ({cr} cr) — BACKLOG: {history} "
                    f"Completing mandatory backlog courses protects your GPA and unblocks "
                    f"future courses that depend on this one as a prerequisite."
                )
                if desc:
                    note += f" What you will study: {desc}"

            elif stage == 'current':
                note = (
                    f"📖 {title}{code_tag} ({cr} cr) — CURRENT TERM mandatory "
                    f"(Year {uni_year}, {semester} Semester). "
                    f"This course is part of your official curriculum for this semester. "
                    f"Completing it on schedule keeps your graduation plan on track "
                    f"and ensures you do not accumulate backlog in future terms."
                )
                if what:
                    note += f" What you will gain: {what}"

            elif stage == 'future':
                from_y = c.get('_from_year', '?')
                note = (
                    f"🔮 {title}{code_tag} ({cr} cr) — ADVANCED from Year {from_y}, "
                    f"{semester} Semester. "
                    f"You still have available credit space this term, so this future course "
                    f"is pulled forward to make the most of your allowance. "
                    f"Taking it now lightens your workload in upcoming semesters and keeps "
                    f"you ahead of schedule."
                )
                if what:
                    note += f" What you will gain: {what}"

            elif stage == 'elective':
                cats     = c.get('_preference_cats', '')
                cat_list = [ct.strip() for ct in cats.split('&')] if cats else []
                career   = ' and '.join(
                    _CAT_CAREER.get(ct.replace(' ', '_'), ct) for ct in cat_list
                ) if cat_list else 'your academic interests'
                note = (
                    f"⭐ {title}{code_tag} ({cr} cr) — ELECTIVE chosen specifically for you "
                    f"based on your strengths in {cats if cats else 'your interests'}. "
                    f"Your profile shows a strong affinity for these areas, and this course "
                    f"builds directly on that foundation."
                )
                if what:
                    note += f" In this course you will learn: {what}"
                note += (
                    f" These skills will prepare you for careers in {career} "
                    f"and strengthen your profile for jobs and graduate study in this field."
                )

            elif stage == 'stage4':
                note = (
                    f"✨ {title}{code_tag} ({cr} cr) — CREDIT FILLER: you had {cr} credit(s) "
                    f"remaining in your allowance, so this course is added to make full use of "
                    f"your available load."
                )
                if what:
                    note += f" {what}"

            else:
                note = f"• {title}{code_tag} ({cr} cr)"
                if what:
                    note += f" — {what}"

            per_course_notes.append(note)

        advisor_notes.extend(per_course_notes)
        return course_details

    # ── Stage 1: Previous terms (same semester type) ──────────────────────────
    prev_mandatory:      List[dict] = []
    prev_elective_slots: int        = 0

    for y in range(1, uni_year + 1):
        for s in ['First', 'Second']:
            if y == uni_year and s == semester:
                break
            if s != semester:
                continue
            term_courses = _get_term_mandatory(y, s, track, completed, planned_names(), earned)
            for c in term_courses:
                c['_from_year']         = y
                c['_from_sem']          = s
                c['_previously_failed'] = c['course_name'] in failed_courses_set
            prev_mandatory.extend(term_courses)
            prev_elective_slots += remaining_slots.get((y, s), 0)

    _has_backlog = bool(prev_mandatory or prev_elective_slots > 0)

    if _has_backlog:
        cur_names      = _get_term_all_names(uni_year, semester, track)
        next_y, next_s = _next_term(uni_year, semester)
        if is_unspecialized and next_y >= 3:
            next_names = set(SHARED_COURSES_BY_TERM.get((next_y, next_s), []))
        else:
            next_names = _get_term_all_names(next_y, next_s, track)
        if process_stage(
            prev_mandatory, prev_elective_slots,
            [cur_names, next_names],
            "Previous terms backlog",
            stage_type='backlog',
        ):
            if _termination_info.get('reason') == 'overflow':
                _generate_overflow_note()
            _generate_termination_note()
            cd = _finalize_with_unplanned()
            return _build_result(student_id, uni_year, semester, _display_track,
                                  original_avail, suggest_courses, advisor_notes, cd)
    else:
        advisor_notes.append(
            f"✅ Great news — you have completed all mandatory courses from your previous "
            f"{semester} semesters! Your backlog is completely clear. Well done!"
        )

    # ── Stage 2: Current term ─────────────────────────────────────────────────
    cur_mandatory = _get_term_mandatory(
        uni_year, semester, track, completed, planned_names(), earned
    )
    for c in cur_mandatory:
        c['_from_year'] = uni_year
        c['_from_sem']  = semester
    cur_elective_slots = remaining_slots.get((uni_year, semester), 0)

    _has_current = bool(cur_mandatory or cur_elective_slots > 0)

    if _has_current:
        next_y, next_s = _next_term(uni_year, semester)
        if is_unspecialized and next_y >= 3:
            next_names = set(SHARED_COURSES_BY_TERM.get((next_y, next_s), []))
        else:
            next_names = _get_term_all_names(next_y, next_s, track)
        if process_stage(
            cur_mandatory, cur_elective_slots,
            [next_names],
            f"Year {uni_year} {semester} Semester",
            stage_type='current',
        ):
            if _termination_info.get('reason') == 'overflow':
                _generate_overflow_note()
            _generate_termination_note()
            cd = _finalize_with_unplanned()
            return _build_result(student_id, uni_year, semester, _display_track,
                                  original_avail, suggest_courses, advisor_notes, cd)
    else:
        advisor_notes.append(
            f"⭐ You have already completed all mandatory courses for your current term "
            f"(Year {uni_year}, {semester} Semester). That is exceptional progress — "
            f"you are ahead of your curriculum!"
        )

    # ── Stage 3: Future years (same semester), uni_year+1 → 4 ────────────────
    for future_y in range(uni_year + 1, 5):
        if is_unspecialized and future_y >= 3:
            # Only shared courses — no track-specific electives
            fut_mandatory = _get_shared_term_mandatory(
                future_y, semester, completed, planned_names(), earned
            )
            for c in fut_mandatory:
                c['_from_year'] = future_y
                c['_from_sem']  = semester
            fut_elective_slots = 0
        else:
            fut_mandatory = _get_term_mandatory(
                future_y, semester, track, completed, planned_names(), earned
            )
            for c in fut_mandatory:
                c['_from_year'] = future_y
                c['_from_sem']  = semester
            fut_elective_slots = remaining_slots.get((future_y, semester), 0)

        if not fut_mandatory and fut_elective_slots == 0:
            continue

        if process_stage(
            fut_mandatory, fut_elective_slots,
            [],
            f"Year {future_y} {semester} Semester (advanced)",
            stage_type='future',
        ):
            if _termination_info.get('reason') == 'overflow':
                _generate_overflow_note()
            _generate_termination_note()
            cd = _finalize_with_unplanned()
            return _build_result(student_id, uni_year, semester, _display_track,
                                  original_avail, suggest_courses, advisor_notes, cd)

    # ── Stage 4: Fill leftover 1-2 credits from closest future year ───────────
    if 0 < avail_credits <= 2:
        pnames = planned_names()
        for future_y in range(uni_year + 1, 5):
            if is_unspecialized and future_y >= 3:
                continue  # shared Y3/4 courses were handled in Stage 3
            candidates = _stage4_candidates(
                avail_credits, future_y, semester, track, completed, pnames, earned
            )
            if not candidates:
                continue

            added = None
            if avail_credits == 2:
                two = [c for c in candidates if c.get('credit_hours') == 2]
                if two:
                    added = [two[0]]
                    advisor_notes.append(
                        f"Filled 2 remaining credits with '{two[0]['course_name']}' (Year {future_y})"
                    )
                else:
                    one = [c for c in candidates if c.get('credit_hours') == 1]
                    if len(one) >= 2:
                        added = one[:2]
                        advisor_notes.append(
                            f"Filled 2 remaining credits with 2 one-credit courses (Year {future_y})"
                        )
                    elif one:
                        added = [one[0]]
                        advisor_notes.append(
                            f"Filled 1 of 2 remaining credits with '{one[0]['course_name']}' (Year {future_y})"
                        )

            elif avail_credits == 1:
                one = [c for c in candidates if c.get('credit_hours') == 1]
                if one:
                    added = [one[0]]
                    advisor_notes.append(
                        f"Filled 1 remaining credit with '{one[0]['course_name']}' (Year {future_y})"
                    )

            if added:
                for c in added:
                    c['_stage']     = 'stage4'
                    c['_from_year'] = future_y
                suggest_courses.extend(added)
                break

    cd = _finalize_with_unplanned()
    return _build_result(
        student_id, uni_year, semester, _display_track,
        original_avail, suggest_courses, advisor_notes, cd,
    )
