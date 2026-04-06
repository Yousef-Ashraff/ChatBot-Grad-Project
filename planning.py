"""
planning.py — Student Course Planning Function
===============================================

Exact copy of the notebook planning() function with only these
mechanical substitutions applied:

  1. Student data: supabase direct queries replaced by
     get_student_context() from eligibility.py (new schema).

  2. get_elective_slots() renamed to get_elective_slots_planning()
     to avoid collision with the neo4j_course_functions version.

  3. get_prerequisites() alias removed; calls replaced directly with
     get_course_dependencies() from neo4j_course_functions.

  4. get_all_electives_by_program() and get_courses_by_term() imported
     directly from neo4j_course_functions.

Everything else — all print statements, input prompts, loop logic,
variable names, credit calculations, deferral flows — is identical to
the notebook.
"""

from __future__ import annotations


# ─────────────────────────────────────────────────────────────────────────────
# get_elective_slots_planning
# Renamed from get_elective_slots to avoid collision with the version
# that already exists in neo4j_course_functions.
# Logic is byte-for-byte identical to the notebook's get_elective_slots().
# ─────────────────────────────────────────────────────────────────────────────

def get_elective_slots_planning(track, year, semester):
    """
    Get the number of elective slots for a given track, year, and semester.

    Args:
        track:    Program name/track
        year:     Year level (1-4 or 'First Year', etc.)
        semester: 'First' or 'Second'

    Returns:
        Number of elective slots
    """
    # Normalize inputs
    track = track.lower()

    # Normalize year to number
    if isinstance(year, str):
        year_map = {
            'first year': 4, 'second year': 3, 'third year': 2, 'fourth year': 1,
            'first': 4, 'second': 3, 'third': 2, 'fourth': 1,
            '1': 4, '2': 3, '3': 2, '4': 1
        }
        year_num = year_map.get(year.lower(), int(year) if year.isdigit() else 0)
    else:
        year_num = 5 - int(year)  # Convert 1->4, 2->3, 3->2, 4->1

    # AI & ML has different slots
    if 'artificial intelligence' in track:
        if year_num == 1 and semester == 'First':   # 4th year, 1st semester
            return 2
        elif year_num == 1 and semester == 'Second': # 4th year, 2nd semester
            return 3
        else:
            return 0

    # SAD and DAS have same slots
    elif 'software' in track or 'data science' in track:
        if year_num == 2 and semester == 'Second':  # 3rd year, 2nd semester
            return 1
        elif year_num == 1 and semester == 'First': # 4th year, 1st semester
            return 2
        elif year_num == 1 and semester == 'Second':# 4th year, 2nd semester
            return 2
        else:
            return 0

    return 0


# ─────────────────────────────────────────────────────────────────────────────
# planning()
# ─────────────────────────────────────────────────────────────────────────────

def planning(student_id):
    """
    Create a course plan for a student for a specific semester.

    Args:
        student_id: The student's ID

    Returns:
        Dictionary with suggested courses and planning details
    """
    from neo4j_course_functions import (
        get_courses_by_term,
        get_course_dependencies,
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

    if not track:
        print(f"❌ Error: Student ID {student_id} not found!")
        return None

    # Resolve semester from current_term (1 or 2), default First
    semester = 'Second' if current_term == 2 else 'First'


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
        print(f"  Latest: {', '.join(completed_courses[-3:])}")  # Show last 3
    print()

    # Validate electives in completed courses
    print("\nValidating elective courses...")

    # Get all electives for this track
    all_track_electives = get_all_electives_by_program(track)
    elective_names_set = {e['course_name'].lower() for e in all_track_electives}

    # Get all elective slots from previous terms (in chronological order)
    elective_slots_needed = []
    for year in range(1, university_year + 1):
        for sem in ['First', 'Second']:
            # Skip current planning term
            if year == university_year and sem == semester:
                continue

            expected_slots = get_elective_slots_planning(track, year, sem)
            if expected_slots > 0:
                for slot_num in range(expected_slots):
                    elective_slots_needed.append({
                        'year': year,
                        'semester': sem,
                        'slot_num': slot_num + 1
                    })

    # Get completed electives
    completed_electives = [c for c in completed_courses if c.lower() in elective_names_set]

    # Assign completed electives to slots (earliest first)
    filled_slots = []
    unfilled_slots = []

    for i, slot in enumerate(elective_slots_needed):
        if i < len(completed_electives):
            filled_slots.append({
                **slot,
                'course': completed_electives[i]
            })
        else:
            unfilled_slots.append(slot)

    # Display results
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
    mandatory_missing = []      # All missing mandatory courses (for display)
    same_semester_missing = []  # Missing courses from same semester type (will be added to plan)
    elective_missing = []       # Missing electives from previous terms

    # Get all electives for this track
    all_track_electives = get_all_electives_by_program(track)
    elective_names = {e['course_name'].lower() for e in all_track_electives}

    # Get all elective slots chronologically (same logic as validation)
    elective_slots_all = []

    # Get all courses from previous years and current year's previous semester
    for year in range(1, university_year + 1):
        for sem in ['First', 'Second']:
            # Skip current planning semester
            if year == university_year and sem == semester:
                break

            # Check elective requirements for this term
            expected_elective_slots = get_elective_slots_planning(track, year, sem)
            for slot_i in range(expected_elective_slots):
                elective_slots_all.append({
                    'year': year,
                    'semester': sem,
                    'slot_num': slot_i + 1
                })

            year_courses = get_courses_by_term(year, sem, track)

            # Extract courses for this track
            if year_courses:
                for year_key, semesters in year_courses.items():
                    for sem_key, programs in semesters.items():
                        if track in programs:
                            for course in programs[track]:
                                course_name = course['course_name']

                                # If not completed and mandatory, add to missing list
                                if course_name not in completed_courses:
                                    if course['course_type'] == 'mandatory':
                                        mandatory_missing.append({
                                            **course,
                                            'from_year': year,
                                            'from_semester': sem
                                        })

                                        # Only add to suggest_courses if same semester type AND prerequisites are met
                                        if sem == semester:
                                            # Check prerequisites before adding
                                            prereqs = get_course_dependencies(course_name, track)
                                            can_take = True

                                            if prereqs['prerequisites']:
                                                for prereq in prereqs['prerequisites']:
                                                    # Handle credit hour requirements
                                                    if 'Required_Credit_Hours' in prereq:
                                                        required_credits = int(prereq['Required_Credit_Hours'])
                                                        if earned_credits < required_credits:
                                                            can_take = False
                                                            break
                                                    else:
                                                        # Regular course prerequisite
                                                        prereq_name = prereq['name']
                                                        if prereq_name not in completed_courses:
                                                            can_take = False
                                                            break

                                            # Only add if prerequisites are met
                                            if can_take:
                                                same_semester_missing.append(course)
                                                suggest_courses.append(course)

    # Assign completed electives to earliest slots
    completed_electives = [c for c in completed_courses if c.lower() in elective_names]

    unfilled_slots = []
    for slot_idx, slot in enumerate(elective_slots_all):
        if slot_idx >= len(completed_electives):
            unfilled_slots.append(slot)

    # Track all elective placeholders (for display) and same-semester ones (for planning)
    all_elective_placeholders = []

    # Add placeholders for unfilled slots only
    for slot in unfilled_slots:
        elective_placeholder = {
            'course_name': f'elective slot {slot["slot_num"]}',
            'course_code': f'ELEC-{slot["year"]}{slot["semester"][0]}',
            'credit_hours': 3,  # Standard elective credit hours
            'course_type': 'elective',
            'from_year': slot['year'],
            'from_semester': slot['semester'],
            'is_placeholder': True
        }
        all_elective_placeholders.append(elective_placeholder)

        # Only add to suggest_courses if same semester type
        if slot['semester'] == semester:
            same_semester_missing.append(elective_placeholder)
            suggest_courses.append(elective_placeholder)

    # Display all missing courses
    if mandatory_missing or all_elective_placeholders:
        total_missing = len(mandatory_missing) + len(all_elective_placeholders)
        print(f"📋 ALL Missing Requirements from Previous Terms ({total_missing}):")
        print()

        # Show mandatory courses
        for course in mandatory_missing:
            print(f"   [{course['course_code']}] {course['course_name']} ({course['credit_hours']} credits)")
            print(f"      From: Year {course['from_year']}, {course['from_semester']} Semester")

        # Show ALL elective placeholders (including different semester types)
        for course in all_elective_placeholders:
            print(f"   [ELECTIVE] Missing elective slot ({course['credit_hours']} credits)")
            print(f"      From: Year {course['from_year']}, {course['from_semester']} Semester")

        print()

        # Show which ones will be added to plan (same semester only)
        # BUT ONLY if prerequisites are met!
        if same_semester_missing:
            mandatory_same_sem = [c for c in same_semester_missing if not c.get('is_placeholder')]
            elective_same_sem = [c for c in same_semester_missing if c.get('is_placeholder')]

            # Filter mandatory courses by prerequisites
            applicable_mandatory = []
            for course in mandatory_same_sem:
                course_name = course['course_name']

                # Get prerequisites
                prereqs = get_course_dependencies(course_name, track)

                # Check if prerequisites are met
                can_take = True

                if prereqs['prerequisites']:
                    for prereq in prereqs['prerequisites']:
                        # Handle credit hour requirements
                        if 'Required_Credit_Hours' in prereq:
                            required_credits = int(prereq['Required_Credit_Hours'])
                            if earned_credits < required_credits:
                                can_take = False
                                break
                        else:
                            # Regular course prerequisite
                            prereq_name = prereq['name']
                            if prereq_name not in completed_courses:
                                can_take = False
                                break

                if can_take:
                    applicable_mandatory.append(course)

            # Calculate total applicable (mandatory + elective slots)
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

    # If previous term courses meet or exceed available credits, don't add current term courses
    if total_previous_term_credits >= aval_credits:
        print(f"⚠️  Previous term courses ({total_previous_term_credits} credits) already meet/exceed your credit limit ({aval_credits} credits)")
        print("📋 Focusing on completing previous term courses only")
        print()
        print("Current term courses will be skipped for this planning session.")
        print("You should prioritize completing missing courses from previous terms first.")
        print()

        # Skip to electives section
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

            # Skip if already completed
            if course_name in completed_courses:
                continue

            # Skip if already in suggest_courses
            if any(c['course_name'] == course_name for c in suggest_courses):
                continue

            # Get prerequisites
            prereqs = get_course_dependencies(course_name, track)

            # Check if prerequisites are met
            can_take = True

            if prereqs['prerequisites']:
                for prereq in prereqs['prerequisites']:
                    # Handle credit hour requirements (graduation project, field training)
                    if 'Required_Credit_Hours' in prereq:
                        required_credits = int(prereq['Required_Credit_Hours'])
                        if earned_credits < required_credits:
                            can_take = False
                            print(f"   ✗ {course_name}: Need {required_credits} credits (have {earned_credits})")
                            break
                    else:
                        # Regular course prerequisite
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

    # Step 9: Handle electives - TWO SEPARATE PHASES
    print("="*80)
    print("ELECTIVE COURSES")
    print("="*80)

    num_elective_slots = get_elective_slots_planning(track, university_year, semester)

    # Count missing electives from previous same-semester terms (these are already in suggest_courses as placeholders)
    num_missing_electives = sum(1 for c in suggest_courses if c.get('is_placeholder'))

    # Check if we have room for electives
    current_total_credits = sum(c['credit_hours'] for c in suggest_courses)

    if skip_current_term:
        print(f"⚠️  Previous term courses use {current_total_credits} credits")
        if current_total_credits >= aval_credits:
            print(f"No room for electives (already at/over {aval_credits} credit limit)")
            # Remove placeholders since we can't add electives
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

        # Get all electives for track
        all_electives = get_all_electives_by_program(track)

        # Filter electives where prerequisites are met
        suggest_elective_courses = []

        for elective in all_electives:
            course_name = elective['course_name']

            # Skip if already completed
            if course_name in completed_courses:
                continue

            # Get prerequisites
            prereqs = get_course_dependencies(course_name, track)

            # Check if prerequisites are met
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

        # Display electives with descriptions
        for i, elective in enumerate(suggest_elective_courses, 1):
            print(f"{i}. [{elective['course_code']}] {elective['course_name']}")
            print(f"   Credits: {elective['credit_hours']}")
            desc = elective.get('description', 'No description available')
            if desc and len(desc) > 100:
                desc = desc[:97] + "..."
            print(f"   Description: {desc}")
            print()

        # Let user choose electives for PREVIOUS terms
        print(f"Please choose {num_missing_electives} elective(s) to fulfill PREVIOUS term requirements.")
        print("Enter the numbers separated by commas (e.g., 1,3,5)")
        print(f"You need to select exactly {num_missing_electives} elective(s), or enter 'q' to quit:")

        while True:
            choices = input("Your choices: ").strip()

            if choices.lower() == 'q':
                print("⚠️  Skipping previous term elective selection.")
                # Remove placeholder electives since user didn't select any
                suggest_courses = [c for c in suggest_courses if not c.get('is_placeholder')]
                print(f"   Note: {num_missing_electives} previous term elective requirement(s) will not be fulfilled.")
                num_missing_electives = 0  # Mark as handled
                break
            elif choices:
                try:
                    # Parse indices
                    indices = [int(x.strip()) - 1 for x in choices.split(',')]

                    # Check for duplicates
                    if len(indices) != len(set(indices)):
                        duplicates = [str(i+1) for i in indices if indices.count(i) > 1]
                        print(f"❌ Duplicate elective numbers detected: {', '.join(set(duplicates))}")
                        print(f"   Please enter each elective number only once:")
                        continue

                    # Check for invalid indices
                    invalid = [str(i+1) for i in indices if i < 0 or i >= len(suggest_elective_courses)]
                    if invalid:
                        print(f"❌ Invalid elective numbers: {', '.join(invalid)}")
                        print(f"   Valid range is 1 to {len(suggest_elective_courses)}. Please try again:")
                        continue

                    # Get chosen electives (now guaranteed no duplicates)
                    chosen_electives = [suggest_elective_courses[i] for i in indices]

                    # Validate exact number
                    if len(chosen_electives) != num_missing_electives:
                        print(f"❌ You selected {len(chosen_electives)} elective(s), but you need exactly {num_missing_electives}.")
                        print(f"Please try again or enter 'q' to quit:")
                        continue

                    # Remove placeholder electives from suggest_courses
                    suggest_courses = [c for c in suggest_courses if not c.get('is_placeholder')]

                    # Add chosen electives to suggest_courses as MANDATORY from previous terms
                    # Also add to same_semester_missing so they're treated as previous term courses
                    for elective in chosen_electives:
                        elective_course = {
                            'course_name': elective['course_name'],
                            'course_code': elective['course_code'],
                            'credit_hours': elective['credit_hours'],
                            'course_type': 'mandatory',  # Treat as mandatory!
                            'from_previous_elective': True  # Mark as from previous elective slot
                        }
                        suggest_courses.append(elective_course)
                        same_semester_missing.append(elective_course)

                    print(f"\n✓ Added {len(chosen_electives)} elective(s) to PREVIOUS TERM requirements")
                    num_missing_electives = 0  # Mark as handled
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

        # Get all electives for track
        all_electives = get_all_electives_by_program(track)

        # Filter electives where prerequisites are met
        suggest_elective_courses = []

        for elective in all_electives:
            course_name = elective['course_name']

            # Skip if already completed
            if course_name in completed_courses:
                continue

            # Skip if already selected in Phase 1
            if any(c['course_name'] == course_name for c in suggest_courses if c.get('from_previous_elective')):
                continue

            # Get prerequisites
            prereqs = get_course_dependencies(course_name, track)

            # Check if prerequisites are met
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

        # Display electives with descriptions
        for i, elective in enumerate(suggest_elective_courses, 1):
            print(f"{i}. [{elective['course_code']}] {elective['course_name']}")
            print(f"   Credits: {elective['credit_hours']}")
            desc = elective.get('description', 'No description available')
            if desc and len(desc) > 100:
                desc = desc[:97] + "..."
            print(f"   Description: {desc}")
            print()

        # Let user choose electives for CURRENT term
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
                    # Parse indices
                    indices = [int(x.strip()) - 1 for x in choices.split(',')]

                    # Check for duplicates
                    if len(indices) != len(set(indices)):
                        duplicates = [str(i+1) for i in indices if indices.count(i) > 1]
                        print(f"❌ Duplicate elective numbers detected: {', '.join(set(duplicates))}")
                        print(f"   Please enter each elective number only once:")
                        continue

                    # Check for invalid indices
                    invalid = [str(i+1) for i in indices if i < 0 or i >= len(suggest_elective_courses)]
                    if invalid:
                        print(f"❌ Invalid elective numbers: {', '.join(invalid)}")
                        print(f"   Valid range is 1 to {len(suggest_elective_courses)}. Please try again:")
                        continue

                    # Get chosen electives (now guaranteed no duplicates)
                    chosen_electives = [suggest_elective_courses[i] for i in indices]

                    # Validate exact number
                    if len(chosen_electives) != num_elective_slots:
                        print(f"❌ You selected {len(chosen_electives)} elective(s), but you need exactly {num_elective_slots}.")
                        print(f"Please try again or enter 'q' to quit:")
                        continue

                    # Add chosen electives to suggest_courses as CURRENT term electives
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

    # Safety check: Remove any remaining placeholders (shouldn't happen, but just in case)
    suggest_courses = [c for c in suggest_courses if not c.get('is_placeholder')]

    print()

    # Step 10: Check if total credits exceed available
    # Calculate credits from previous terms and current term separately
    # Previous term includes both mandatory and previous electives (marked with from_previous_elective)
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

    # If previous term courses alone meet or exceed available credits
    if previous_term_credits >= aval_credits:
        print("⚠️  Previous term courses already fill your available credits!")
        print("    Current term courses will be excluded from this semester's plan.")
        print("    You should focus on completing previous term requirements first.")
        print()

        # Keep only previous term courses (both mandatory and previous electives)
        previous_electives = [c for c in suggest_courses if c.get('from_previous_elective')]
        suggest_courses = same_semester_missing.copy()
        suggest_courses.extend(previous_electives)

        # Recalculate previous term credits
        previous_term_credits = sum(c['credit_hours'] for c in suggest_courses)

        # If still over limit, user must choose what to defer
        if previous_term_credits > aval_credits:
            print(f"    Even previous term courses exceed limit by {previous_term_credits - aval_credits} credits.")
            print(f"    You must defer some courses to next semester.")
            print()
            print("    Previous term courses:")

            # Combine mandatory and previous electives for display
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
                        # Parse indices
                        indices = [int(x.strip()) - 1 for x in remove_choices.split(',')]

                        # Check for duplicates
                        if len(indices) != len(set(indices)):
                            duplicates = [str(i+1) for i in indices if indices.count(i) > 1]
                            print(f"    ❌ Duplicate course numbers detected: {', '.join(set(duplicates))}")
                            print(f"       Please enter each course number only once:")
                            continue

                        # Check for invalid indices
                        invalid = [str(i+1) for i in indices if i < 0 or i >= len(all_previous)]
                        if invalid:
                            print(f"    ❌ Invalid course numbers: {', '.join(invalid)}")
                            print(f"       Valid range is 1 to {len(all_previous)}. Please try again:")
                            continue

                        # Get courses to remove (now guaranteed no duplicates)
                        to_remove = [all_previous[i] for i in indices]

                        # Calculate credits to be removed
                        credits_to_remove = sum(c['credit_hours'] for c in to_remove)
                        new_total = previous_term_credits - credits_to_remove

                        # Validate that enough credits are removed
                        if new_total > aval_credits:
                            print(f"    ❌ Deferring those courses gives {new_total} credits, still over {aval_credits} limit.")
                            print(f"       You deferred {credits_to_remove} credits, but need to defer at least {previous_term_credits - aval_credits} credits.")
                            print("       Please try again:")
                            continue

                        # Remove selected courses
                        for course in to_remove:
                            suggest_courses = [c for c in suggest_courses if c['course_name'] != course['course_name']]

                        total_credits = sum(c['credit_hours'] for c in suggest_courses)
                        print(f"\n    ✓ Deferred {len(to_remove)} course(s). New total: {total_credits} credits")
                        break
                    except ValueError:
                        print("    ❌ Invalid input. Please enter numbers separated by commas (e.g., 1,3,5):")
                    except:
                        print("    ❌ Invalid selection. Please enter valid numbers separated by commas:")
                else:
                    print("    ❌ You must defer courses to meet the credit limit. Please try again:")

        # Check if there are remaining elective placeholders after deferral
        remaining_placeholders = [c for c in suggest_courses if c.get('is_placeholder')]
        if remaining_placeholders:
            print()
            print("    ⚠️  You have elective slot(s) remaining after deferral.")
            print(f"    These {len(remaining_placeholders)} elective(s) must be filled with actual courses.")
            print()

            # Get all electives for track
            all_electives = get_all_electives_by_program(track)

            # Filter electives where prerequisites are met
            suggest_elective_courses = []

            for elective in all_electives:
                course_name = elective['course_name']

                # Skip if already completed
                if course_name in completed_courses:
                    continue

                # Get prerequisites
                prereqs = get_course_dependencies(course_name, track)

                # Check if prerequisites are met
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
                        # Parse indices
                        indices = [int(x.strip()) - 1 for x in elec_choices.split(',')]

                        # Check for duplicates
                        if len(indices) != len(set(indices)):
                            duplicates = [str(i+1) for i in indices if indices.count(i) > 1]
                            print(f"    ❌ Duplicate elective numbers detected: {', '.join(set(duplicates))}")
                            print(f"       Please enter each elective number only once:")
                            continue

                        # Check for invalid indices
                        invalid = [str(i+1) for i in indices if i < 0 or i >= len(suggest_elective_courses)]
                        if invalid:
                            print(f"    ❌ Invalid elective numbers: {', '.join(invalid)}")
                            print(f"       Valid range is 1 to {len(suggest_elective_courses)}. Please try again:")
                            continue

                        # Get chosen electives (now guaranteed no duplicates)
                        chosen_electives = [suggest_elective_courses[i] for i in indices]

                        # Validate exact number
                        if len(chosen_electives) != len(remaining_placeholders):
                            print(f"    ❌ You selected {len(chosen_electives)}, but need exactly {len(remaining_placeholders)}.")
                            print(f"    Please try again:")
                            continue

                        # Remove placeholders
                        suggest_courses = [c for c in suggest_courses if not c.get('is_placeholder')]

                        # Add chosen electives as mandatory from previous terms
                        for elective in chosen_electives:
                            suggest_courses.append({
                                'course_name': elective['course_name'],
                                'course_code': elective['course_code'],
                                'credit_hours': elective['credit_hours'],
                                'course_type': 'mandatory',  # Treat as mandatory from previous
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

        # Separate mandatory from previous terms and current term courses
        previous_term_courses = [
            c for c in suggest_courses
            if c in same_semester_missing or c.get('from_previous_elective')
        ]

        print(f"📌 MANDATORY from previous terms ({previous_term_credits} credits):")
        for course in previous_term_courses:
            elec_note = " (Previous elective)" if course.get('from_previous_elective') else ""
            print(f"   • [{course['course_code']}] {course['course_name']} ({course['credit_hours']} credits){elec_note}")
        print()

        # Show current term courses that can be deferred
        current_term_courses = [
            c for c in suggest_courses
            if c not in same_semester_missing and not c.get('from_previous_elective')
        ]

        if current_term_courses:
            print(f"Current term courses (can be deferred):")
            for i, course in enumerate(current_term_courses, 1):
                print(f"{i}. [{course['course_code']}] {course['course_name']} ({course['credit_hours']} credits)")

            print()
            print(f"You need to remove {total_credits - aval_credits} credits worth of courses.")
            print("Enter the numbers of courses to REMOVE (comma-separated):")

            while True:
                remove_choices = input("Remove courses: ").strip()

                if remove_choices:
                    try:
                        # Parse indices
                        indices = [int(x.strip()) - 1 for x in remove_choices.split(',')]

                        # Check for duplicates
                        if len(indices) != len(set(indices)):
                            duplicates = [str(i+1) for i in indices if indices.count(i) > 1]
                            print(f"❌ Duplicate course numbers detected: {', '.join(set(duplicates))}")
                            print(f"   Please enter each course number only once:")
                            continue

                        # Check for invalid indices
                        invalid = [str(i+1) for i in indices if i < 0 or i >= len(current_term_courses)]
                        if invalid:
                            print(f"❌ Invalid course numbers: {', '.join(invalid)}")
                            print(f"   Valid range is 1 to {len(current_term_courses)}. Please try again:")
                            continue

                        # Get courses to remove (now guaranteed no duplicates)
                        to_remove = [current_term_courses[i] for i in indices]

                        # Calculate credits to be removed
                        credits_to_remove = sum(c['credit_hours'] for c in to_remove)
                        new_total = total_credits - credits_to_remove

                        # Validate that enough credits are removed
                        if new_total > aval_credits:
                            print(f"❌ Removing those courses gives {new_total} credits, still over {aval_credits} limit.")
                            print(f"   You removed {credits_to_remove} credits, but need to remove at least {total_credits - aval_credits} credits.")
                            print("   Please try again:")
                            continue

                        # Remove selected courses
                        for course in to_remove:
                            suggest_courses = [c for c in suggest_courses if c['course_name'] != course['course_name']]

                        total_credits = sum(c['credit_hours'] for c in suggest_courses)
                        print(f"\n✓ Removed {len(to_remove)} course(s). New total: {total_credits} credits")
                        break
                    except ValueError:
                        print("❌ Invalid input. Please enter numbers separated by commas (e.g., 1,3,5):")
                    except:
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

    # FINAL SAFETY CHECK: Remove any remaining placeholders
    suggest_courses = [c for c in suggest_courses if not c.get('is_placeholder')]

    print("="*80)
    print("YOUR COURSE PLAN")
    print("="*80)
    print(f"Year: {university_year} | Semester: {semester} | Track: {track.upper()}")
    print(f"Total Credits: {sum(c['credit_hours'] for c in suggest_courses)}/{aval_credits}")
    print()

    for i, course in enumerate(suggest_courses, 1):
        # Previous electives are shown as MANDATORY
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