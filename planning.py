"""
planning.py — Student Course Planning Function
===============================================

Interactive course planner driven by the chatbot via planning_service.py.
Uses print() / input() which are intercepted by the thread bridge in
planning_service.py so the chatbot can drive it turn-by-turn.

Local helpers defined here:
  - get_elective_slots(track, year, semester)  — elective slot counts per term
  - get_prerequisites(course_name, program)    — alias for get_course_dependencies
"""

from __future__ import annotations


# ─────────────────────────────────────────────────────────────────────────────
# get_elective_slots
# Returns the number of elective slots for a given track / year / semester.
# ─────────────────────────────────────────────────────────────────────────────

def get_elective_slots(track, year, semester):
    """
    Get the number of elective slots for a given track, year, and semester.

    Args:
        track:    Program name/track (full canonical name)
        year:     Year level (integer 1-4)
        semester: 'First' or 'Second'

    Returns:
        Number of elective slots (int)
    """
    track = track.lower()

    # Convert year to numeric index (1 = Fourth Year, 4 = First Year)
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
        if year_num == 1 and semester == 'First':    # Year 4, Sem 1
            return 2
        elif year_num == 1 and semester == 'Second': # Year 4, Sem 2
            return 3
        else:
            return 0

    elif 'software' in track or 'data science' in track:
        if year_num == 2 and semester == 'Second':   # Year 3, Sem 2
            return 1
        elif year_num == 1 and semester == 'First':  # Year 4, Sem 1
            return 2
        elif year_num == 1 and semester == 'Second': # Year 4, Sem 2
            return 2
        else:
            return 0

    return 0


# ─────────────────────────────────────────────────────────────────────────────
# get_prerequisites — alias for get_course_dependencies
# ─────────────────────────────────────────────────────────────────────────────

def get_prerequisites(course_name, program_name):
    """Alias for get_course_dependencies — prerequisites only."""
    from neo4j_course_functions import get_course_dependencies
    return get_course_dependencies(course_name, program_name, dependents=False)


# ─────────────────────────────────────────────────────────────────────────────
# planning()
# ─────────────────────────────────────────────────────────────────────────────

def planning(student_id, supabase_client):
    """
    Create a course plan for a student for a specific semester.

    Args:
        student_id:      The student's ID
        supabase_client: Supabase client instance (received from PlanningOrchestrator)

    Returns:
        Dictionary with suggested courses and planning details
    """
    from neo4j_course_functions import (
        get_courses_by_term,
        get_all_electives_by_program,
    )
    from eligibility import get_student_context

    print("="*80)
    print("STUDENT COURSE PLANNING SYSTEM")
    print("="*80)
    print()

    # Step 1: Fetch student data
    print("Fetching student information...")

    ctx = get_student_context(student_id)
    completed_courses = ctx["completed_courses"]
    track             = ctx["program_name"]
    earned_credits    = ctx["total_hours_earned"]
    current_gpa       = ctx["gpa"]
    university_year   = ctx["university_year"]
    current_term      = ctx["current_term"]

    if current_gpa is None:
        current_gpa = 0

    if current_term is None:
        semester = 1
    else:
        semester = current_term

    print(f"✓ Student: {ctx.get('first_name', '')} {ctx.get('last_name', '')}")
    print(f"✓ Track: {track.upper()}")
    print(f"✓ Current Year: {university_year}")
    print()

    if not track:
        print(f"❌ Error: Student ID {student_id} not found!")
        return None

    if semester == 1:
        semester = 'First'
    elif semester == 2:
        semester = 'Second'

    print(f"\n📅 Planning for: Year {university_year}, {semester} Semester")
    print()

    # Step 3: Fetch and display academic details
    print("="*80)
    print("CURRENT ACADEMIC STATUS")
    print("="*80)

    print(f"Current GPA: {current_gpa}")
    print(f"Earned Credits: {earned_credits}")
    print(f"Completed Courses: {len(completed_courses)} courses")
    if completed_courses:
        print(f"  Latest: {', '.join(completed_courses[-3:])}")
    print()

    # Validate electives in completed courses
    print("\nValidating elective courses...")

    all_track_electives = get_all_electives_by_program(track)
    elective_names_set = {e['course_name'].lower() for e in all_track_electives}

    # Get all elective slots from previous terms (stop AT current planning term)
    elective_slots_needed = []
    for year in range(1, university_year + 1):
        for sem in ['First', 'Second']:
            if year == university_year and sem == semester:
                break
            expected_slots = get_elective_slots(track, year, sem)
            if expected_slots > 0:
                for slot_num in range(expected_slots):
                    elective_slots_needed.append({
                        'year': year,
                        'semester': sem,
                        'slot_num': slot_num + 1
                    })

    completed_electives = [c for c in completed_courses if c.lower() in elective_names_set]

    filled_slots = []
    unfilled_slots = []

    for i, slot in enumerate(elective_slots_needed):
        if i < len(completed_electives):
            filled_slots.append({**slot, 'course': completed_electives[i]})
        else:
            unfilled_slots.append(slot)

    print(f"   Total elective slots expected so far: {len(elective_slots_needed)}")
    print(f"   Total electives completed: {len(completed_electives)}")
    print()

    if filled_slots:
        print(f"   ✓ Filled elective slots:")
        for slot in filled_slots:
            print(f"      • Year {slot['year']}, {slot['semester']}: {slot['course']}")
        print()

    if unfilled_slots:
        print(f"   ⚠️  Missing elective slots ({len(unfilled_slots)}):")
        for slot in unfilled_slots:
            print(f"      • Year {slot['year']}, {slot['semester']} Semester: Slot {slot['slot_num']}")
        print()
    else:
        print(f"   ✓ All required electives completed!")
        print()

    print()
    print()

    # Step 5: Calculate available credits based on GPA
    if current_gpa >= 3.0:
        aval_credits = 21
    elif current_gpa >= 2.0:
        aval_credits = 18
    else:
        aval_credits = 15

    print("="*80)
    print("CREDIT ALLOCATION")
    print("="*80)
    print(f"Based on your GPA ({current_gpa}), you can register for: {aval_credits} credit hours")
    print()

    # Step 6: Find courses not completed from previous years/semesters
    print("="*80)
    print("ANALYZING PREVIOUS YEARS")
    print("="*80)

    suggest_courses = []
    mandatory_missing = []
    same_semester_missing = []
    elective_missing = []

    all_track_electives = get_all_electives_by_program(track)
    elective_names = {e['course_name'].lower() for e in all_track_electives}

    elective_slots_all = []

    for year in range(1, university_year + 1):
        for sem in ['First', 'Second']:
            if year == university_year and sem == semester:
                break

            expected_elective_slots = get_elective_slots(track, year, sem)
            for slot_i in range(expected_elective_slots):
                elective_slots_all.append({
                    'year': year,
                    'semester': sem,
                    'slot_num': slot_i + 1
                })

            year_courses = get_courses_by_term(year, sem, track)

            if year_courses:
                for year_key, semesters in year_courses.items():
                    for sem_key, programs in semesters.items():
                        if track in programs:
                            for course in programs[track]:
                                course_name = course['course_name']

                                if course_name not in completed_courses:
                                    if course['course_type'] == 'mandatory':
                                        mandatory_missing.append({
                                            **course,
                                            'from_year': year,
                                            'from_semester': sem
                                        })

                                        if sem == semester:
                                            prereqs = get_prerequisites(course_name, track)
                                            can_take = True

                                            if prereqs['prerequisites']:
                                                for prereq in prereqs['prerequisites']:
                                                    if 'Required_Credit_Hours' in prereq:
                                                        required_credits = int(prereq['Required_Credit_Hours'])
                                                        if earned_credits < required_credits:
                                                            can_take = False
                                                            break
                                                    else:
                                                        prereq_name = prereq['name']
                                                        if prereq_name not in completed_courses:
                                                            can_take = False
                                                            break

                                            if can_take:
                                                same_semester_missing.append(course)
                                                suggest_courses.append(course)

    completed_electives = [c for c in completed_courses if c.lower() in elective_names]

    unfilled_slots = []
    for slot_idx, slot in enumerate(elective_slots_all):
        if slot_idx >= len(completed_electives):
            unfilled_slots.append(slot)

    all_elective_placeholders = []

    for slot in unfilled_slots:
        elective_placeholder = {
            'course_name': f'elective slot {slot["slot_num"]}',
            'course_code': f'ELEC-{slot["year"]}{slot["semester"][0]}',
            'credit_hours': 3,
            'course_type': 'elective',
            'from_year': slot['year'],
            'from_semester': slot['semester'],
            'is_placeholder': True
        }
        all_elective_placeholders.append(elective_placeholder)

        if slot['semester'] == semester:
            same_semester_missing.append(elective_placeholder)
            suggest_courses.append(elective_placeholder)

    if mandatory_missing or all_elective_placeholders:
        total_missing = len(mandatory_missing) + len(all_elective_placeholders)
        print(f"📋 ALL Missing Requirements from Previous Terms ({total_missing}):")
        print()

        for course in mandatory_missing:
            print(f"   [{course['course_code']}] {course['course_name']} ({course['credit_hours']} credits)")
            print(f"      From: Year {course['from_year']}, {course['from_semester']} Semester")

        for course in all_elective_placeholders:
            print(f"   [ELECTIVE] Missing elective slot ({course['credit_hours']} credits)")
            print(f"      From: Year {course['from_year']}, {course['from_semester']} Semester")

        print()

        if same_semester_missing:
            mandatory_same_sem = [c for c in same_semester_missing if not c.get('is_placeholder')]
            elective_same_sem  = [c for c in same_semester_missing if c.get('is_placeholder')]

            applicable_mandatory = []
            for course in mandatory_same_sem:
                course_name = course['course_name']
                prereqs = get_prerequisites(course_name, track)
                can_take = True

                if prereqs['prerequisites']:
                    for prereq in prereqs['prerequisites']:
                        if 'Required_Credit_Hours' in prereq:
                            required_credits = int(prereq['Required_Credit_Hours'])
                            if earned_credits < required_credits:
                                can_take = False
                                break
                        else:
                            prereq_name = prereq['name']
                            if prereq_name not in completed_courses:
                                can_take = False
                                break

                if can_take:
                    applicable_mandatory.append(course)

            total_applicable = len(applicable_mandatory) + len(elective_same_sem)

            if total_applicable > 0:
                print(f"✓ Applicable to current plan ({semester} Semester): {total_applicable} requirement(s)")

                if applicable_mandatory:
                    print(f"   Mandatory courses:")
                    for course in applicable_mandatory:
                        print(f"   • [{course['course_code']}] {course['course_name']}")

                if elective_same_sem:
                    print(f"   Elective slots to fill: {len(elective_same_sem)}")
                    for course in elective_same_sem:
                        print(f"   • Year {course['from_year']} {course['from_semester']} semester elective")

                print()
            else:
                print(f"✓ No applicable {semester} semester requirements")
                print(f"   (All missing courses have unmet prerequisites)")
                print()
        else:
            print(f"✓ No missing {semester} semester requirements from previous years")
            print()
    else:
        print("✓ All previous requirements completed!")

    print()

    # Step 7: Check if previous term courses already fill available credits
    total_previous_term_credits = sum(c['credit_hours'] for c in same_semester_missing)

    print("="*80)
    print(f"COURSES FOR YEAR {university_year}, {semester.upper()} SEMESTER")
    print("="*80)

    if total_previous_term_credits >= aval_credits:
        print(f"⚠️  Previous term courses ({total_previous_term_credits} credits) already meet/exceed your credit limit ({aval_credits} credits)")
        print("📋 Focusing on completing previous term courses only")
        print()
        print("Current term courses will be skipped for this planning session.")
        print("You should prioritize completing missing courses from previous terms first.")
        print()
        skip_current_term = True
    else:
        skip_current_term = False
        current_term_courses = get_courses_by_term(university_year, semester, track)
        term_courses = []

        if current_term_courses:
            for year_key, semesters in current_term_courses.items():
                for sem_key, programs in semesters.items():
                    if track in programs:
                        term_courses = programs[track]

        print(f"Found {len(term_courses)} courses scheduled for this term")
        print(f"Previous term courses: {total_previous_term_credits} credits")
        print(f"Remaining credits available: {aval_credits - total_previous_term_credits} credits")
        print()

    # Step 8: Filter courses where prerequisites are met (only if not skipping)
    if not skip_current_term:
        print("Checking prerequisites...")

        for course in term_courses:
            course_name = course['course_name']

            if course_name in completed_courses:
                continue

            if any(c['course_name'] == course_name for c in suggest_courses):
                continue

            prereqs = get_prerequisites(course_name, track)
            can_take = True

            if prereqs['prerequisites']:
                for prereq in prereqs['prerequisites']:
                    if 'Required_Credit_Hours' in prereq:
                        required_credits = int(prereq['Required_Credit_Hours'])
                        if earned_credits < required_credits:
                            can_take = False
                            print(f"   ✗ {course_name}: Need {required_credits} credits (have {earned_credits})")
                            break
                    else:
                        prereq_name = prereq['name']
                        if prereq_name not in completed_courses:
                            can_take = False
                            print(f"   ✗ {course_name}: Missing prerequisite '{prereq_name}'")
                            break

            if can_take and course['course_type'] == 'mandatory':
                suggest_courses.append(course)
                print(f"   ✓ {course_name}: Prerequisites met")

        print()
    else:
        print("⏭️  Skipping current term course analysis (focus on previous terms)")
        print()

    # Step 9: Handle electives — TWO SEPARATE PHASES
    print("="*80)
    print("ELECTIVE COURSES")
    print("="*80)

    num_elective_slots   = get_elective_slots(track, university_year, semester)
    num_missing_electives = sum(1 for c in suggest_courses if c.get('is_placeholder'))
    current_total_credits = sum(c['credit_hours'] for c in suggest_courses)

    if skip_current_term:
        print(f"⚠️  Previous term courses use {current_total_credits} credits")
        if current_total_credits >= aval_credits:
            print(f"No room for electives (already at/over {aval_credits} credit limit)")
            suggest_courses = [c for c in suggest_courses if not c.get('is_placeholder')]
            print(f"   Note: Elective requirements cannot be fulfilled this semester due to credit limit.")
            num_missing_electives = 0
            num_elective_slots = 0
        else:
            remaining = aval_credits - current_total_credits
            print(f"Remaining credits: {remaining}")

    # PHASE 1: Fill previous term elective slots FIRST (if any)
    if num_missing_electives > 0:
        print()
        print("="*80)
        print("PHASE 1: PREVIOUS TERM ELECTIVE REQUIREMENTS")
        print("="*80)
        print(f"⚠️  You have {num_missing_electives} missing elective(s) from previous {semester} semester terms")
        print(f"   These are MANDATORY and must be filled like any other previous term course.")
        print()

        all_electives = get_all_electives_by_program(track)
        suggest_elective_courses = []

        for elective in all_electives:
            course_name = elective['course_name']
            if course_name in completed_courses:
                continue
            prereqs = get_prerequisites(course_name, track)
            can_take = True
            if prereqs['prerequisites']:
                for prereq in prereqs['prerequisites']:
                    if 'Required_Credit_Hours' in prereq:
                        required_credits = int(prereq['Required_Credit_Hours'])
                        if earned_credits < required_credits:
                            can_take = False
                            break
                    else:
                        prereq_name = prereq['name']
                        if prereq_name not in completed_courses:
                            can_take = False
                            break
            if can_take:
                suggest_elective_courses.append(elective)

        print(f"{len(suggest_elective_courses)} elective(s) available with prerequisites met:")
        print()

        for i, elective in enumerate(suggest_elective_courses, 1):
            print(f"{i}. [{elective['course_code']}] {elective['course_name']}")
            print(f"   Credits: {elective['credit_hours']}")
            desc = elective.get('description', 'No description available')
            if desc and len(desc) > 100:
                desc = desc[:97] + "..."
            print(f"   Description: {desc}")
            print()

        print(f"Please choose {num_missing_electives} elective(s) to fulfill PREVIOUS term requirements.")
        print("Enter the numbers separated by commas (e.g., 1,3,5)")
        print(f"You need to select exactly {num_missing_electives} elective(s), or enter 'q' to quit:")

        while True:
            choices = input("Your choices: ").strip()

            if choices.lower() == 'q':
                print("⚠️  Skipping previous term elective selection.")
                suggest_courses = [c for c in suggest_courses if not c.get('is_placeholder')]
                print(f"   Note: {num_missing_electives} previous term elective requirement(s) will not be fulfilled.")
                num_missing_electives = 0
                break
            elif choices:
                try:
                    indices = [int(x.strip()) - 1 for x in choices.split(',')]

                    if len(indices) != len(set(indices)):
                        duplicates = [str(i+1) for i in indices if indices.count(i) > 1]
                        print(f"❌ Duplicate elective numbers detected: {', '.join(set(duplicates))}")
                        print(f"   Please enter each elective number only once:")
                        continue

                    invalid = [str(i+1) for i in indices if i < 0 or i >= len(suggest_elective_courses)]
                    if invalid:
                        print(f"❌ Invalid elective numbers: {', '.join(invalid)}")
                        print(f"   Valid range is 1 to {len(suggest_elective_courses)}. Please try again:")
                        continue

                    chosen_electives = [suggest_elective_courses[i] for i in indices]

                    if len(chosen_electives) != num_missing_electives:
                        print(f"❌ You selected {len(chosen_electives)} elective(s), but you need exactly {num_missing_electives}.")
                        print(f"Please try again or enter 'q' to quit:")
                        continue

                    suggest_courses = [c for c in suggest_courses if not c.get('is_placeholder')]

                    for elective in chosen_electives:
                        elective_course = {
                            'course_name': elective['course_name'],
                            'course_code': elective['course_code'],
                            'credit_hours': elective['credit_hours'],
                            'course_type': 'mandatory',
                            'from_previous_elective': True
                        }
                        suggest_courses.append(elective_course)
                        same_semester_missing.append(elective_course)

                    print(f"\n✓ Added {len(chosen_electives)} elective(s) to PREVIOUS TERM requirements")
                    num_missing_electives = 0
                    break
                except ValueError:
                    print("❌ Invalid input. Please enter numbers separated by commas (e.g., 1,3,5):")
            else:
                print(f"❌ Please select {num_missing_electives} elective(s) or enter 'q' to quit:")

    # PHASE 2: Fill current term elective slots (if any)
    if num_elective_slots > 0:
        print()
        print("="*80)
        print("PHASE 2: CURRENT TERM ELECTIVE SELECTION")
        print("="*80)
        print(f"Current term has {num_elective_slots} elective slot(s) available.")
        print()

        all_electives = get_all_electives_by_program(track)
        suggest_elective_courses = []

        for elective in all_electives:
            course_name = elective['course_name']
            if course_name in completed_courses:
                continue
            if any(c['course_name'] == course_name for c in suggest_courses if c.get('from_previous_elective')):
                continue
            prereqs = get_prerequisites(course_name, track)
            can_take = True
            if prereqs['prerequisites']:
                for prereq in prereqs['prerequisites']:
                    if 'Required_Credit_Hours' in prereq:
                        required_credits = int(prereq['Required_Credit_Hours'])
                        if earned_credits < required_credits:
                            can_take = False
                            break
                    else:
                        prereq_name = prereq['name']
                        if prereq_name not in completed_courses:
                            can_take = False
                            break
            if can_take:
                suggest_elective_courses.append(elective)

        print(f"{len(suggest_elective_courses)} elective(s) available with prerequisites met:")
        print()

        for i, elective in enumerate(suggest_elective_courses, 1):
            print(f"{i}. [{elective['course_code']}] {elective['course_name']}")
            print(f"   Credits: {elective['credit_hours']}")
            desc = elective.get('description', 'No description available')
            if desc and len(desc) > 100:
                desc = desc[:97] + "..."
            print(f"   Description: {desc}")
            print()

        print(f"Please choose {num_elective_slots} elective(s) for CURRENT term.")
        print("Enter the numbers separated by commas (e.g., 1,3,5)")
        print(f"You need to select exactly {num_elective_slots} elective(s), or enter 'q' to quit:")

        while True:
            choices = input("Your choices: ").strip()

            if choices.lower() == 'q':
                print("⚠️  Skipping current term elective selection.")
                break
            elif choices:
                try:
                    indices = [int(x.strip()) - 1 for x in choices.split(',')]

                    if len(indices) != len(set(indices)):
                        duplicates = [str(i+1) for i in indices if indices.count(i) > 1]
                        print(f"❌ Duplicate elective numbers detected: {', '.join(set(duplicates))}")
                        print(f"   Please enter each elective number only once:")
                        continue

                    invalid = [str(i+1) for i in indices if i < 0 or i >= len(suggest_elective_courses)]
                    if invalid:
                        print(f"❌ Invalid elective numbers: {', '.join(invalid)}")
                        print(f"   Valid range is 1 to {len(suggest_elective_courses)}. Please try again:")
                        continue

                    chosen_electives = [suggest_elective_courses[i] for i in indices]

                    if len(chosen_electives) != num_elective_slots:
                        print(f"❌ You selected {len(chosen_electives)} elective(s), but you need exactly {num_elective_slots}.")
                        print(f"Please try again or enter 'q' to quit:")
                        continue

                    for elective in chosen_electives:
                        suggest_courses.append({
                            'course_name': elective['course_name'],
                            'course_code': elective['course_code'],
                            'credit_hours': elective['credit_hours'],
                            'course_type': 'elective'
                        })

                    print(f"\n✓ Added {len(chosen_electives)} elective(s) to CURRENT TERM courses")
                    break
                except ValueError:
                    print("❌ Invalid input. Please enter numbers separated by commas (e.g., 1,3,5):")
            else:
                print(f"❌ Please select {num_elective_slots} elective(s) or enter 'q' to quit:")

    # Safety check: remove any remaining placeholders
    suggest_courses = [c for c in suggest_courses if not c.get('is_placeholder')]

    print()

    # Step 10: Check if total credits exceed available
    previous_term_credits = sum(
        c['credit_hours'] for c in suggest_courses
        if c in same_semester_missing or c.get('from_previous_elective')
    )
    current_term_credits = sum(
        c['credit_hours'] for c in suggest_courses
        if c not in same_semester_missing and not c.get('from_previous_elective')
    )
    total_credits = sum(c['credit_hours'] for c in suggest_courses)

    print("="*80)
    print("CREDIT CALCULATION")
    print("="*80)
    print(f"Previous term courses: {previous_term_credits} credits")
    print(f"Current term courses: {current_term_credits} credits")
    print(f"Total suggested: {total_credits} credits")
    print(f"Available credits: {aval_credits} credits")
    print()

    if previous_term_credits >= aval_credits:
        print("⚠️  Previous term courses already fill your available credits!")
        print("    Current term courses will be excluded from this semester's plan.")
        print("    You should focus on completing previous term requirements first.")
        print()

        previous_electives = [c for c in suggest_courses if c.get('from_previous_elective')]
        suggest_courses = same_semester_missing.copy()
        suggest_courses.extend(previous_electives)

        previous_term_credits = sum(c['credit_hours'] for c in suggest_courses)

        if previous_term_credits > aval_credits:
            print(f"    Even previous term courses exceed limit by {previous_term_credits - aval_credits} credits.")
            print(f"    You must defer some courses to next semester.")
            print()
            print("    Previous term courses:")

            all_previous = [c for c in suggest_courses if c in same_semester_missing or c.get('from_previous_elective')]
            for i, course in enumerate(all_previous, 1):
                elec_note = " - Previous elective" if course.get('from_previous_elective') else ""
                print(f"    {i}. [{course['course_code']}] {course['course_name']} ({course['credit_hours']} credits){elec_note}")
            print()
            print(f"    You need to remove at least {previous_term_credits - aval_credits} credits worth of courses.")
            print("    Enter the numbers of courses to DEFER (comma-separated):")

            while True:
                remove_choices = input("    Defer courses: ").strip()

                if remove_choices:
                    try:
                        indices = [int(x.strip()) - 1 for x in remove_choices.split(',')]

                        if len(indices) != len(set(indices)):
                            duplicates = [str(i+1) for i in indices if indices.count(i) > 1]
                            print(f"    ❌ Duplicate course numbers detected: {', '.join(set(duplicates))}")
                            print(f"       Please enter each course number only once:")
                            continue

                        invalid = [str(i+1) for i in indices if i < 0 or i >= len(all_previous)]
                        if invalid:
                            print(f"    ❌ Invalid course numbers: {', '.join(invalid)}")
                            print(f"       Valid range is 1 to {len(all_previous)}. Please try again:")
                            continue

                        to_remove = [all_previous[i] for i in indices]
                        credits_to_remove = sum(c['credit_hours'] for c in to_remove)
                        new_total = previous_term_credits - credits_to_remove

                        if new_total > aval_credits:
                            print(f"    ❌ Deferring those courses gives {new_total} credits, still over {aval_credits} limit.")
                            print(f"       You deferred {credits_to_remove} credits, but need to defer at least {previous_term_credits - aval_credits} credits.")
                            print("       Please try again:")
                            continue

                        for course in to_remove:
                            suggest_courses = [c for c in suggest_courses if c['course_name'] != course['course_name']]

                        total_credits = sum(c['credit_hours'] for c in suggest_courses)
                        print(f"\n    ✓ Deferred {len(to_remove)} course(s). New total: {total_credits} credits")
                        break
                    except ValueError:
                        print("    ❌ Invalid input. Please enter numbers separated by commas (e.g., 1,3,5):")
                    except Exception:
                        print("    ❌ Invalid selection. Please enter valid numbers separated by commas:")
                else:
                    print("    ❌ You must defer courses to meet the credit limit. Please try again:")

        remaining_placeholders = [c for c in suggest_courses if c.get('is_placeholder')]
        if remaining_placeholders:
            print()
            print("    ⚠️  You have elective slot(s) remaining after deferral.")
            print(f"    These {len(remaining_placeholders)} elective(s) must be filled with actual courses.")
            print()

            all_electives = get_all_electives_by_program(track)
            suggest_elective_courses = []

            for elective in all_electives:
                course_name = elective['course_name']
                if course_name in completed_courses:
                    continue
                prereqs = get_prerequisites(course_name, track)
                can_take = True
                if prereqs['prerequisites']:
                    for prereq in prereqs['prerequisites']:
                        if 'Required_Credit_Hours' in prereq:
                            required_credits = int(prereq['Required_Credit_Hours'])
                            if earned_credits < required_credits:
                                can_take = False
                                break
                        else:
                            prereq_name = prereq['name']
                            if prereq_name not in completed_courses:
                                can_take = False
                                break
                if can_take:
                    suggest_elective_courses.append(elective)

            print(f"    {len(suggest_elective_courses)} elective(s) available:")
            for i, elective in enumerate(suggest_elective_courses, 1):
                print(f"    {i}. [{elective['course_code']}] {elective['course_name']} ({elective['credit_hours']} credits)")
            print()

            print(f"    Please choose {len(remaining_placeholders)} elective(s) for the remaining slot(s).")
            print(f"    Enter the numbers separated by commas:")

            while True:
                elec_choices = input("    Your choices: ").strip()

                if elec_choices:
                    try:
                        indices = [int(x.strip()) - 1 for x in elec_choices.split(',')]

                        if len(indices) != len(set(indices)):
                            duplicates = [str(i+1) for i in indices if indices.count(i) > 1]
                            print(f"    ❌ Duplicate elective numbers detected: {', '.join(set(duplicates))}")
                            print(f"       Please enter each elective number only once:")
                            continue

                        invalid = [str(i+1) for i in indices if i < 0 or i >= len(suggest_elective_courses)]
                        if invalid:
                            print(f"    ❌ Invalid elective numbers: {', '.join(invalid)}")
                            print(f"       Valid range is 1 to {len(suggest_elective_courses)}. Please try again:")
                            continue

                        chosen_electives = [suggest_elective_courses[i] for i in indices]

                        if len(chosen_electives) != len(remaining_placeholders):
                            print(f"    ❌ You selected {len(chosen_electives)}, but need exactly {len(remaining_placeholders)}.")
                            print(f"    Please try again:")
                            continue

                        suggest_courses = [c for c in suggest_courses if not c.get('is_placeholder')]

                        for elective in chosen_electives:
                            suggest_courses.append({
                                'course_name': elective['course_name'],
                                'course_code': elective['course_code'],
                                'credit_hours': elective['credit_hours'],
                                'course_type': 'mandatory',
                                'from_previous_elective': True
                            })

                        print(f"\n    ✓ Added {len(chosen_electives)} elective(s) to your plan")
                        break
                    except ValueError:
                        print("    ❌ Invalid input. Please enter numbers separated by commas (e.g., 1,3,5):")
                else:
                    print(f"    ❌ Please select {len(remaining_placeholders)} elective(s):")

    elif total_credits > aval_credits:
        print("="*80)
        print("FINAL COURSE SELECTION")
        print("="*80)
        print(f"⚠️  Total suggested courses ({total_credits} credits) exceed your limit ({aval_credits} credits)!")
        print(f"    You need to remove {total_credits - aval_credits} credits worth of courses.")
        print()
        print("Please choose which courses to prioritize:")
        print()

        previous_term_courses_list = [
            c for c in suggest_courses
            if c in same_semester_missing or c.get('from_previous_elective')
        ]

        print(f"📌 MANDATORY from previous terms ({previous_term_credits} credits):")
        for course in previous_term_courses_list:
            elec_note = " (Previous elective)" if course.get('from_previous_elective') else ""
            print(f"   • [{course['course_code']}] {course['course_name']} ({course['credit_hours']} credits){elec_note}")
        print()

        current_term_courses_list = [
            c for c in suggest_courses
            if c not in same_semester_missing and not c.get('from_previous_elective')
        ]

        if current_term_courses_list:
            print(f"Current term courses (can be deferred):")
            for i, course in enumerate(current_term_courses_list, 1):
                print(f"{i}. [{course['course_code']}] {course['course_name']} ({course['credit_hours']} credits)")

            print()
            print(f"You need to remove {total_credits - aval_credits} credits worth of courses.")
            print("Enter the numbers of courses to REMOVE (comma-separated):")

            while True:
                remove_choices = input("Remove courses: ").strip()

                if remove_choices:
                    try:
                        indices = [int(x.strip()) - 1 for x in remove_choices.split(',')]

                        if len(indices) != len(set(indices)):
                            duplicates = [str(i+1) for i in indices if indices.count(i) > 1]
                            print(f"❌ Duplicate course numbers detected: {', '.join(set(duplicates))}")
                            print(f"   Please enter each course number only once:")
                            continue

                        invalid = [str(i+1) for i in indices if i < 0 or i >= len(current_term_courses_list)]
                        if invalid:
                            print(f"❌ Invalid course numbers: {', '.join(invalid)}")
                            print(f"   Valid range is 1 to {len(current_term_courses_list)}. Please try again:")
                            continue

                        to_remove = [current_term_courses_list[i] for i in indices]
                        credits_to_remove = sum(c['credit_hours'] for c in to_remove)
                        new_total = total_credits - credits_to_remove

                        if new_total > aval_credits:
                            print(f"❌ Removing those courses gives {new_total} credits, still over {aval_credits} limit.")
                            print(f"   You removed {credits_to_remove} credits, but need to remove at least {total_credits - aval_credits} credits.")
                            print("   Please try again:")
                            continue

                        for course in to_remove:
                            suggest_courses = [c for c in suggest_courses if c['course_name'] != course['course_name']]

                        total_credits = sum(c['credit_hours'] for c in suggest_courses)
                        print(f"\n✓ Removed {len(to_remove)} course(s). New total: {total_credits} credits")
                        break
                    except ValueError:
                        print("❌ Invalid input. Please enter numbers separated by commas (e.g., 1,3,5):")
                    except Exception:
                        print("❌ Invalid selection. Please enter valid numbers separated by commas:")
                else:
                    print("❌ You must remove courses to meet the credit limit. Please try again:")
    else:
        print("="*80)
        print("FINAL COURSE SELECTION")
        print("="*80)
        print(f"✓ Total credits ({total_credits}) within limit ({aval_credits})")
        print()

    # Final summary
    print()

    # FINAL SAFETY CHECK: remove any remaining placeholders
    suggest_courses = [c for c in suggest_courses if not c.get('is_placeholder')]

    print("="*80)
    print("YOUR COURSE PLAN")
    print("="*80)
    print(f"Year: {university_year} | Semester: {semester} | Track: {track.upper()}")
    print(f"Total Credits: {sum(c['credit_hours'] for c in suggest_courses)}/{aval_credits}")
    print()

    for i, course in enumerate(suggest_courses, 1):
        if course.get('from_previous_elective'):
            course_type_label = "MANDATORY (Previous elective)"
        else:
            course_type_label = "MANDATORY" if course['course_type'] == 'mandatory' else "ELECTIVE"

        print(f"{i}. [{course['course_code']}] {course['course_name']}")
        print(f"   {course['credit_hours']} credits | {course_type_label}")
        print()

    print("="*80)

    return {
        'student_id': student_id,
        'year': university_year,
        'semester': semester,
        'track': track,
        'available_credits': aval_credits,
        'planned_courses': suggest_courses,
        'total_credits': sum(c['credit_hours'] for c in suggest_courses)
    }
