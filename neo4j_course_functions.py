"""
Neo4j Course Management Functions

This module provides functions to interact with a Neo4j knowledge graph
for course prerequisites, dependencies, and program management.

Updated with improved functions from Grad_project_2__2_.ipynb
"""

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Neo4j Connection Configuration from environment variables
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE', 'neo4j')  # Default to 'neo4j' if not specified

# Initialize Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


# Year level mapping (accepts both formats)
YEAR_LEVEL_MAP = {
    # Number format
    '1': 'First Year',
    '2': 'Second Year',
    '3': 'Third Year',
    '4': 'Fourth Year',
    # Ordinal format
    'first': 'First Year',
    'second': 'Second Year',
    'third': 'Third Year',
    'fourth': 'Fourth Year',
    # Full format (already correct)
    'first year': 'First Year',
    'second year': 'Second Year',
    'third year': 'Third Year',
    'fourth year': 'Fourth Year',
}

SEMESTER_MAP = {
    # String format
    'first': 'First',
    'second': 'Second',
    # Number format
    '1': 'First',
    '2': 'Second',
    1: 'First',
    2: 'Second',
    'first semester': 'First',
    'second semester': 'Second'
}


def _resolve_code(codes):
    """
    Deserialize the Case-3 code structure produced by no-program-context queries.

    Cypher queries that run without a single Program context return code as:
      [{program: null, code: "X"}]         → code lives on the Course node
                                             → returns "X"  (plain string)
      [{program: "aim", code: "A"}, ...]   → code lives on BELONGS_TO relationships
                                             → returns the list as-is

    Args:
        codes: list of {program, code} dicts from the Cypher CASE expression, or None.

    Returns:
        str   – when there is exactly one entry and its program key is None
        list  – when multiple per-program entries are present
        None  – when codes is empty / None
    """
    if not codes:
        return None
    if len(codes) == 1 and codes[0].get("program") is None:
        return codes[0]["code"]
    return codes


def test_connection():
    """Test the connection to Neo4j Aura database."""
    with driver.session(database="neo4j") as session:
        result = session.run("RETURN 'connected to Aura' AS msg")
        for record in result:
            print(record["msg"])


def run_cypher_query(query, params=None):
    """
    Execute a Cypher query on the Neo4j database.
    
    Args:
        query: Cypher query string
        params: Dictionary of parameters for the query
        
    Returns:
        List of record dictionaries
    """
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(query, params or {})
        return [record.data() for record in result]


def normalize_level(level):
    """Normalize level input to 'First Year', 'Second Year', etc."""
    if isinstance(level, int):
        level = str(level)
    normalized = YEAR_LEVEL_MAP.get(level.lower().strip())
    if not normalized:
        raise ValueError(f"Invalid level: '{level}'. Use 1-4, 'First'/'Second'/'Third'/'Fourth', or 'First Year' etc.")
    return normalized


def normalize_semester(semester):
    """Normalize semester input to 'First' or 'Second'."""
    if semester is None:
        return None
    key = semester if isinstance(semester, int) else semester.lower().strip()
    normalized = SEMESTER_MAP.get(key)
    if not normalized:
        raise ValueError(f"Invalid semester: '{semester}'. Use 1/2 or 'First'/'Second'.")
    return normalized


def get_courses_by_term(level, semester=None, program_name=None):
    """
    Get all courses for a given level and semester across programs.

    Args:
        level: Academic year level. Accepts:
               - Integer: 1, 2, 3, 4
               - String ordinal: 'First', 'Second', 'Third', 'Fourth'
               - Full string: 'First Year', 'Second Year', etc.
        semester: Optional semester. Accepts:
               - Integer: 1, 2
               - String: 'First', 'Second'
               - None: returns both semesters
        program_name: Optional program name (string or list) to filter by

    Returns:
        Dictionary organized by level → semester → program → courses
    """

    # Normalize level (must be provided)
    if level is None:
        raise ValueError("Level must be specified.")

    normalized_level = normalize_level(level)
    normalized_semester = normalize_semester(semester)

    # Handle program filter
    if program_name is None:
        program_names = [
            'artificial intelligence & machine learning',
            'software & application development',
            'data science'
        ]
    elif isinstance(program_name, str):
        program_names = [program_name.lower()]
    else:
        program_names = [p.lower() for p in program_name]

    # Build query based on whether semester is specified
    if normalized_semester:
        query = """
        MATCH (c:Course)-[r:BELONGS_TO]->(p:Program)
        WHERE r.year_name = $year_name
          AND r.semester = $semester
          AND p.name IN $program_names
        RETURN
            r.year_name AS year,
            r.semester AS semester,
            p.name AS program,
            c.name AS course_name,
            COALESCE(c.code, r.code) AS course_code,
            c.credit_hours AS credit_hours,
            r.elective AS course_type
        ORDER BY p.name, COALESCE(c.code, r.code)
        """
        params = {
            "year_name": normalized_level,
            "semester": normalized_semester,
            "program_names": program_names
        }
    else:
        query = """
        MATCH (c:Course)-[r:BELONGS_TO]->(p:Program)
        WHERE r.year_name = $year_name
          AND p.name IN $program_names
        RETURN
            r.year_name AS year,
            r.semester AS semester,
            p.name AS program,
            c.name AS course_name,
            COALESCE(c.code, r.code) AS course_code,
            c.credit_hours AS credit_hours,
            r.elective AS course_type
        ORDER BY p.name, r.semester, COALESCE(c.code, r.code)
        """
        params = {
            "year_name": normalized_level,
            "program_names": program_names
        }

    result = run_cypher_query(query, params)

    # Organize results by level → semester → program → courses
    organized = {}

    for record in result:
        year = record["year"]
        sem = record["semester"]
        prog = record["program"]

        if year not in organized:
            organized[year] = {}

        if sem not in organized[year]:
            organized[year][sem] = {}

        if prog not in organized[year][sem]:
            organized[year][sem][prog] = []

        organized[year][sem][prog].append({
            "course_name": record["course_name"],
            "course_code": record["course_code"],
            "credit_hours": record["credit_hours"],
            "course_type": "elective" if record["course_type"] == "yes" else "mandatory"
        })

    return organized


def get_courses_by_multiple_terms(terms, program_name=None):
    """
    Get courses for multiple levels and their corresponding semesters.

    Args:
        terms: List of tuples or dicts specifying level and optional semester.
               Examples:
               - [(1, 'First'), (2, None), (3, 'Second')]
               - [{'level': 1, 'semester': 'First'}, {'level': 2}]
               - [(1, 1), (2, 2), (3, None)]
        program_name: Optional program name (string or list) to filter by

    Returns:
        Dictionary organized by level → semester → program → courses
    """

    combined_results = {}

    for term in terms:
        # Handle both tuple and dict formats
        if isinstance(term, dict):
            level = term.get('level')
            semester = term.get('semester', None)
        elif isinstance(term, (list, tuple)):
            level = term[0]
            semester = term[1] if len(term) > 1 else None
        else:
            raise ValueError(f"Invalid term format: {term}. Use tuple (level, semester) or dict {{'level': ..., 'semester': ...}}")

        # Get courses for this term
        result = get_courses_by_term(level, semester, program_name)

        # Merge into combined results
        for year, semesters in result.items():
            if year not in combined_results:
                combined_results[year] = {}
            for sem, programs in semesters.items():
                if sem not in combined_results[year]:
                    combined_results[year][sem] = {}
                combined_results[year][sem].update(programs)

    return combined_results


def get_course_dependencies(course_name, program_name=None, prereq=True, dependents=True):
    """
    Get dependency information for a course from the knowledge graph.

    Uses track property if available, otherwise falls back to intersection of programs.

    Args:
        course_name:  Name of the course (case-insensitive).
        program_name: Optional program name (string) or list of program names to filter.
        prereq:       If True (default), include prerequisites in the result.
        dependents:   If True (default), include dependent/closes courses in the result.

    Returns:
        Dictionary with any combination of:
            - "prerequisites": courses needed BEFORE taking this course (when prereq=True)
            - "dependents":    courses unlocked AFTER completing this course (when dependents=True)
    """

    # Convert to lowercase to match the KG format
    course_name = course_name.lower()

    # Special cases with credit hour requirements
    if course_name == 'graduation project (1)':
        result = {}
        if prereq:
            result["prerequisites"] = [{'Required_Credit_Hours': '100',
                'tracks': ['software & application development',
                   'artificial intelligence & machine learning',
                   'data science']}]
        if dependents:
            result["dependents"] = get_course_closes(course_name)
        return result

    elif course_name == 'field training (1)':
        result = {}
        if prereq:
            result["prerequisites"] = [{'Required_Credit_Hours': '60',
                'tracks': ['software & application development',
                   'artificial intelligence & machine learning',
                   'data science']}]
        if dependents:
            result["dependents"] = []
        return result

    elif course_name == 'field training (2)':
        result = {}
        if prereq:
            result["prerequisites"] = [{'Required_Credit_Hours': '90',
                'tracks': ['software & application development',
                   'artificial intelligence & machine learning',
                   'data science']}]
        if dependents:
            result["dependents"] = []
        return result

    if program_name:
        # Handle both string and list inputs
        if isinstance(program_name, str):
            program_names = [program_name.lower()]
        else:
            program_names = [name.lower() for name in program_name]

        # Get program-specific prerequisites with fallback to intersection
        query = """
        MATCH (c:Course {name: $course_name})-[r:HAS_PREREQUISITE]->(prereq:Course)

        // Get programs for input course
        OPTIONAL MATCH (c)-[bc:BELONGS_TO]->(input_prog:Program)
        WHERE input_prog.name IN $program_names

        // Get programs for prerequisite course
        OPTIONAL MATCH (prereq)-[bp:BELONGS_TO]->(prereq_prog:Program)
        WHERE prereq_prog.name IN $program_names

        WITH prereq, r, input_prog, prereq_prog, bc, bp
        WHERE (r.track IS NOT NULL AND r.track CONTAINS input_prog.name)
           OR (r.track IS NULL AND input_prog.name = prereq_prog.name)

        WITH prereq,
             collect(DISTINCT {
                 program: input_prog.name,
                 course_type: bc.elective
             }) AS program_details,
             collect(DISTINCT bp.code)[0] AS rel_code

        WHERE size(program_details) > 0

        RETURN
            prereq.name AS prerequisite_name,
            COALESCE(prereq.code, rel_code) AS prerequisite_code,
            prereq.credit_hours AS credit_hours,
            program_details
        ORDER BY prereq.name
        """
        params = {"course_name": course_name, "program_names": program_names}

        result_dict = {}

        if prereq:
            # Execute prerequisites query on Neo4j
            result = run_cypher_query(query, params)

            # Build prerequisites list
            prerequisites = []
            for record in result:
                prereq_data = {
                    "name": record["prerequisite_name"],
                    "code": record["prerequisite_code"],
                    "credit_hours": record["credit_hours"],
                    "tracks": []
                }

                for detail in record["program_details"]:
                    if detail and detail.get("program"):
                        prereq_data["tracks"].append({
                            "program": detail["program"],
                            "course_type": "elective" if detail["course_type"] == "yes" else "mandatory"
                        })

                if prereq_data["tracks"]:
                    prerequisites.append(prereq_data)

            result_dict["prerequisites"] = prerequisites

        if dependents:
            result_dict["dependents"] = get_course_closes(course_name, program_name)

        return result_dict
    else:
        # Get all prerequisites — respecting r.track on HAS_PREREQUISITE and the
        # intersection of programs that both the main course and the prereq belong to.
        #
        # Root cause of the old bug: the previous query only looked at the PREREQUISITE
        # course's own BELONGS_TO memberships, ignoring the r.track property entirely.
        # That caused general courses (e.g. Structured Programming) to appear as a
        # prerequisite in every program they happen to belong to, even when the
        # HAS_PREREQUISITE relationship is track-restricted or when the main course
        # only exists in a subset of those programs.
        #
        # Fix: mirror the program-specific branch logic but without restricting to a
        # single program:
        #   • If r.track IS NOT NULL  → only include programs named in r.track
        #   • If r.track IS NULL      → only include programs where BOTH courses appear
        query = """
        MATCH (c:Course {name: $course_name})-[r:HAS_PREREQUISITE]->(prereq:Course)

        // Programs the MAIN course belongs to
        OPTIONAL MATCH (c)-[:BELONGS_TO]->(main_prog:Program)

        // Programs the PREREQ course belongs to (with elective flag)
        OPTIONAL MATCH (prereq)-[bp:BELONGS_TO]->(prereq_prog:Program)

        WITH prereq, r,
             collect(DISTINCT main_prog.name) AS main_prog_names,
             collect(DISTINCT {
                 program:     prereq_prog.name,
                 course_type: CASE WHEN bp.elective = 'yes' THEN 'elective' ELSE 'mandatory' END
             }) AS all_prereq_details,
             collect(DISTINCT {program: prereq_prog.name, code: bp.code}) AS all_rel_codes

        // Keep only the program entries where the prereq relationship actually applies:
        //   - track-specific (r.track set)  → program must appear in r.track string
        //   - track-agnostic (r.track null)  → program must be shared by both courses
        WITH prereq, all_rel_codes,
             [detail IN all_prereq_details WHERE
                 detail.program IS NOT NULL AND (
                     (r.track IS NOT NULL AND r.track CONTAINS detail.program)
                     OR  (r.track IS NULL   AND detail.program IN main_prog_names)
                 )
             ] AS program_details,
             [rc IN all_rel_codes WHERE
                 rc.program IS NOT NULL AND (
                     (r.track IS NOT NULL AND r.track CONTAINS rc.program)
                     OR  (r.track IS NULL   AND rc.program IN main_prog_names)
                 )
             ] AS filtered_rel_codes

        WHERE size(program_details) > 0

        // Case 3: no program context.
        // code is on the node  → [{program: null, code: prereq.code}]
        // code is on BELONGS_TO → filtered_rel_codes (one entry per valid program)
        RETURN
            prereq.name  AS prerequisite_name,
            CASE WHEN prereq.code IS NOT NULL
                 THEN [{program: null, code: prereq.code}]
                 ELSE filtered_rel_codes
            END AS prerequisite_code,
            prereq.credit_hours AS credit_hours,
            program_details
        ORDER BY prereq.name
        """
        params = {"course_name": course_name}

        result_dict = {}

        if prereq:
            # Execute prerequisites query on Neo4j
            preq_result = run_cypher_query(query, params)

            # Build prerequisites list — same dict-track format as the program-specific path.
            # _resolve_code() unwraps the Case-3 code structure:
            #   [{program: null, code: "X"}] → "X"  (node-level code, plain string)
            #   [{program: "aim", code: ...}, ...] → list of per-program dicts
            prerequisites = []
            for record in preq_result:
                prereq_data = {
                    "name":         record["prerequisite_name"],
                    "code":         _resolve_code(record["prerequisite_code"]),
                    "credit_hours": record["credit_hours"],
                    "tracks":       [],
                }
                for detail in record["program_details"]:
                    if detail and detail.get("program"):
                        prereq_data["tracks"].append({
                            "program":     detail["program"],
                            "course_type": detail["course_type"],
                        })
                prerequisites.append(prereq_data)

            result_dict["prerequisites"] = prerequisites

        if dependents:
            result_dict["dependents"] = get_course_closes(course_name, program_name)

        return result_dict


def get_course_closes(course_name, program_name=None):
    """
    Get courses that have this course as a prerequisite (courses that this course "closes" or "opens up").
    In other words, find what courses become available after completing this course.
    Args:
        course_name: Name of the course (case-insensitive, will be converted to lowercase)
        program_name: Optional program name (string) or list of program names to get program-specific courses
    Returns:
        List of courses that require this course as a prerequisite
    """
    # Convert to lowercase to match the KG format
    course_name = course_name.lower()
    if program_name is None:
        program_name = ['artificial intelligence & machine learning','software & application development', 'data science']
    # Handle both string and list inputs
    if isinstance(program_name, str):
        program_names = [program_name.lower()]
    else:
        program_names = [name.lower() for name in program_name]
    # Get program-specific courses that require this prerequisite.
    # Each program_details entry carries the per-program COALESCE(c.code, bc.code)
    # so the Python layer can derive the correct code for Case 2 (single program →
    # plain string) and Case 3 (no program → string when unique, list when multiple
    # different codes exist, which happens for the 6 relationship-code courses).
    query = """
    MATCH (prereq:Course {name: $course_name})<-[r:HAS_PREREQUISITE]-(c:Course)
    // Get programs for the courses that have this prerequisite
    OPTIONAL MATCH (c)-[bc:BELONGS_TO]->(prog:Program)
    WHERE prog.name IN $program_names
    WITH c, r, prog, bc
    WHERE (r.track IS NOT NULL AND r.track CONTAINS prog.name)
       OR (r.track IS NULL)
    WITH c,
         collect(DISTINCT {
             program:     prog.name,
             course_type: bc.elective,
             code:        COALESCE(c.code, bc.code)
         }) AS program_details
    WHERE size(program_details) > 0
    RETURN
        c.name AS course_name,
        c.credit_hours AS credit_hours,
        program_details
    ORDER BY c.name
    """
    params = {"course_name": course_name, "program_names": program_names}
    # Execute query on Neo4j
    result = run_cypher_query(query, params)
    # Return list of courses with their program details.
    # Derive the top-level "code" field from per-program codes:
    #   - All programs agree on the same code (or only one program) → plain string
    #   - Programs have different codes (relationship-code courses)  → list of
    #     {program, code} dicts (Case 3 behaviour)
    output = []
    for record in result:
        details = record["program_details"]
        all_codes = [d["code"] for d in details if d and d.get("code") is not None]
        unique_codes = list(dict.fromkeys(all_codes))   # deduplicate, preserve order

        if not unique_codes:
            code = None
        elif len(unique_codes) == 1:
            code = unique_codes[0]
        else:
            code = [{"program": d["program"], "code": d.get("code")} for d in details if d]

        course_data = {
            "name": record["course_name"],
            "code": code,
            "credit_hours": record["credit_hours"],
            "tracks": []
        }
        for detail in details:
            if detail and detail.get("program"):
                course_data["tracks"].append({
                    "program": detail["program"],
                    "course_type": "elective" if detail["course_type"] == "yes" else "mandatory"
                })
        if course_data["tracks"]:
            output.append(course_data)
    return output


def get_course_info(course_name, program_name=None):
    """
    Get course information from the knowledge graph.

    Args:
        course_name: Name of the course (case-insensitive, will be converted to lowercase)
        program_name: Optional program name (string) or list of program names to get program-specific information

    Returns:
        Course information including program-specific details if program is provided
    """

    # Convert to lowercase to match the KG format
    course_name = course_name.lower()

    if program_name:
        # Handle both string and list inputs
        if isinstance(program_name, str):
            program_names = [program_name.lower()]
        else:
            program_names = [name.lower() for name in program_name]

        # Get program-specific course information
        query = """
        MATCH (c:Course {name: $course_name})-[r:BELONGS_TO]->(p:Program)
        WHERE p.name IN $program_names
        WITH c,
             collect(DISTINCT {
                 program: p.name,
                 year: r.year_name,
                 semester: r.semester,
                 course_type: r.elective,
                 min_academic_load: r.min_academic_load,
                 max_standard_academic_load: r.max_standard_academic_load,
                 required_core_credits: r.required_core_credits
             }) AS program_offerings,
             collect(r.code)[0] AS rel_code
        RETURN
            c.name AS course_name,
            COALESCE(c.code, rel_code) AS course_code,
            c.credit_hours AS credit_hours,
            c.description AS description,
            c.motivation AS motivation,
            c.min_hours_to_enroll AS min_hours_to_enroll,
            program_offerings
        """
        params = {"course_name": course_name, "program_names": program_names}
    else:
        # Case 3: no program context — return code as structured list so the caller
        # can distinguish a single node-level code from per-program relationship codes.
        query = """
        MATCH (c:Course {name: $course_name})
        OPTIONAL MATCH (c)-[r:BELONGS_TO]->(p:Program)
        WITH c,
             collect(DISTINCT {
                 program: p.name,
                 year: r.year_name,
                 semester: r.semester,
                 course_type: r.elective,
                 min_academic_load: r.min_academic_load,
                 max_standard_academic_load: r.max_standard_academic_load,
                 required_core_credits: r.required_core_credits
             }) AS program_offerings,
             collect(DISTINCT {program: p.name, code: r.code}) AS rel_codes
        RETURN
            c.name AS course_name,
            CASE WHEN c.code IS NOT NULL
                 THEN [{program: null, code: c.code}]
                 ELSE rel_codes
            END AS course_code,
            c.credit_hours AS credit_hours,
            c.description AS description,
            c.motivation AS motivation,
            c.min_hours_to_enroll AS min_hours_to_enroll,
            program_offerings
        """
        params = {"course_name": course_name}

    result = run_cypher_query(query, params)

    if not program_name:
        # Unwrap the Case-3 code structure: plain string when code is on the node,
        # list of {program, code} dicts when code lives on BELONGS_TO relationships.
        for row in result:
            row['course_code'] = _resolve_code(row['course_code'])

    return result


def get_course_timing(course_name, program_name=None):
    """
    Get timing information (year, semester) for a course using get_course_info().

    Args:
        course_name: Name of the course (case-insensitive, will be converted to lowercase)
        program_name: Optional program name (string) or list of program names to get program-specific timing

    Returns:
        Dictionary with course timing information organized by program
    """

    # Convert to lowercase to match the KG format
    course_name = course_name.lower()

    # Get course information using get_course_info()
    course_info = get_course_info(course_name, program_name)

    if not course_info or len(course_info) == 0:
        return {
            "course_name": course_name,
            "found": False,
            "message": f"Course '{course_name}' not found in the knowledge graph."
        }

    # Extract the first result (should only be one for a specific course)
    info = course_info[0]
    # Build the timing response
    timing_info = {
        "course_name": info['course_name'],
        "course_code": info['course_code'],
        "credit_hours": info['credit_hours'],
        "found": True,
        "programs": []
    }
    sad_das_elective_slots = ['Third Year / Second Sem', 'Fourth Year / First Sem', 'Fourth Year / Second Sem']
    ai_elective_slots = ['Fourth Year / First Sem', 'Fourth Year / Second Sem']

    # If program_name was specified, we have program_offerings
    if 'program_offerings' in info:
        for offering in info['program_offerings']:
            if offering['program']:  # Only include if program is not None
                if offering['course_type'] == 'yes':
                    if offering['program'] == 'data science' or offering['program'] == 'software & application development':
                          offering['semester'] =  ', '.join(sad_das_elective_slots)
                    else:
                        offering['semester'] = ', '.join(ai_elective_slots)
                    offering['year'] = 'None'
                timing_info['programs'].append({
                    "program": offering['program'],
                    "year": offering['year'] if offering['year'] != 'None'else None,
                    "semester": offering['semester'],
                    # "semester_number": offering['semester_number'],
                    "course_type": "elective" if offering['course_type'] == 'yes' else "mandatory"
                })


    return timing_info


def get_elective_slots_time_and_occ(program_name=None):
    """
    Get elective course slots with occurrence counts for each program.

    Args:
        program_name: Optional program name (string) or list of program names

    Returns:
        For a single program: list of dicts [{"slot": "Year / Sem", "count": N}, ...]
        For multiple/all programs: dict mapping program name -> list of dicts
    """

    elective_slots = {
        'software & application development': [
            {"slot": "Third Year / Second Sem", "count": 1},
            {"slot": "Fourth Year / First Sem",  "count": 2},
            {"slot": "Fourth Year / Second Sem", "count": 2},
        ],
        'data science': [
            {"slot": "Third Year / Second Sem", "count": 1},
            {"slot": "Fourth Year / First Sem",  "count": 2},
            {"slot": "Fourth Year / Second Sem", "count": 2},
        ],
        'artificial intelligence & machine learning': [
            {"slot": "Fourth Year / First Sem",  "count": 2},
            {"slot": "Fourth Year / Second Sem", "count": 3},
        ],
    }

    # If no program specified, return all
    if program_name is None:
        return elective_slots

    # Handle both string and list inputs
    if isinstance(program_name, str):
        program_names = [program_name.lower()]
    else:
        program_names = [name.lower() for name in program_name]

    # Filter and return requested programs
    result = {}
    for prog in program_names:
        if prog in elective_slots:
            result[prog] = elective_slots[prog]
        else:
            result[prog] = None

    # If only one program requested, return just that list
    if len(result) == 1:
        return list(result.values())[0]

    return result


# backward-compat alias — keep until get_program_info is recreated
get_elective_slots_time = get_elective_slots_time_and_occ


def get_all_electives_by_program(program_name=None):
    """
    Get all elective courses for specified programs.
    
    Args:
        program_name: Optional program name (string or list) to filter by
    
    Returns:
        List of elective courses or dictionary mapping programs to their electives
    """
    # Handle program filter
    if program_name is None:
        program_names = ['artificial intelligence & machine learning',
                        'software & application development',
                        'data science']
    elif isinstance(program_name, str):
        program_names = [program_name.lower()]
    else:
        program_names = [name.lower() for name in program_name]

    results = {}

    for prog in program_names:
        # Query to get all electives for this program (no semester filtering)
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

        params = {"program_name": prog}
        electives = run_cypher_query(query, params)

        results[prog] = electives

    # If only one program, return just that program's data
    if len(results) == 1:
        return list(results.values())[0]

    return results


def check_course_eligibility(course_name, prerequisites=None, completed_courses=None, earned_credits=None, program_name=None):
    """
    Check if a student is eligible to take a course based on prerequisites.
    
    Args:
        course_name: Name of the course to check eligibility for
        prerequisites: Optional list of prerequisites (if None, will fetch from database)
        completed_courses: List of course names the student has completed
        earned_credits: Total credit hours the student has earned
        program_name: Student's program (optional, for program-specific prerequisites)
    
    Returns:
        Dictionary with eligibility status and details
    """
    
    # Convert to lowercase
    course_name = course_name.lower()
    if completed_courses:
        completed_courses = [c.lower() for c in completed_courses]
    
    # Get prerequisites for the course if not provided
    if not prerequisites:
        prerequisites = get_course_dependencies(course_name, program_name, dependents=False)['prerequisites']

    if not prerequisites:
        return {
            "eligible": True,
            "course": course_name,
            "message": f"No prerequisites required for {course_name}. You can take this course!",
            "missing_prerequisites": [],
            "credit_requirement_met": True
        }
    
    missing_courses = []
    credit_requirement = None
    credit_requirement_met = True
    
    for prereq in prerequisites:
        # Check if this is a credit requirement
        if 'Required_Credit_Hours' in prereq:
            credit_requirement = int(prereq['Required_Credit_Hours'])
            if earned_credits is None:
                credit_requirement_met = False
            elif earned_credits < credit_requirement:
                credit_requirement_met = False
        else:
            # This is a course prerequisite
            prereq_name = prereq.get('name', '').lower()
            if completed_courses is None or prereq_name not in completed_courses:
                missing_courses.append(prereq)
    
    # Determine eligibility
    eligible = len(missing_courses) == 0 and credit_requirement_met
    
    # Build response
    result = {
        "eligible": eligible,
        "course": course_name,
        "missing_prerequisites": missing_courses,
        "credit_requirement": credit_requirement,
        "credit_requirement_met": credit_requirement_met,
        "earned_credits": earned_credits
    }
    
    # Build message
    if eligible:
        result["message"] = f"✅ You are eligible to take {course_name}!"
    else:
        reasons = []
        if missing_courses:
            course_names = [p['name'] for p in missing_courses]
            reasons.append(f"Missing prerequisite courses: {', '.join(course_names)}")
        if not credit_requirement_met:
            if earned_credits is None:
                reasons.append(f"Credit requirement: {credit_requirement} credits needed (your credits not provided)")
            else:
                reasons.append(f"Credit requirement: {credit_requirement} credits needed (you have {earned_credits})")
        result["message"] = f"❌ You are NOT eligible to take {course_name}. " + " | ".join(reasons)
    
    return result


def filter_courses(filters=None, course_types=None, return_fields=None, program_name=None, course_list=None):
    """
    Filter courses based on min_hours_to_enroll and credit_hours.

    Args:
        filters: Dict with field:value pairs. Supported fields:
            - 'min_hours_to_enroll': Minimum credit hours required to enroll
            - 'credit_hours': Number of credit hours the course is worth

            Can use exact match or comparison operators:
            {'min_hours_to_enroll': 0}            # Exact match
            {'credit_hours': {'>=': 2, '<=': 3}}  # Comparison operators

        course_types: List of course types to include:
            - ['core'] or ['mandatory']: Only mandatory/core courses
            - ['elective']:             Only elective courses
            - ['core', 'elective'] or None: All courses (default)

        return_fields: List of fields to return, e.g., ['name', 'code', 'credit_hours']
                       If None, returns: ['name', 'code', 'credit_hours', 'min_hours_to_enroll']
                       Special field 'program_details': Includes program and course type info.

        program_name: String or list of program names to filter by.
                      If None, uses all programs.

        course_list: Optional list of course names (strings) to limit the search to.

    Returns:
        List of course dictionaries matching the filter criteria.
    """
    # Default return fields
    if return_fields is None:
        return_fields = ['name', 'code', 'credit_hours', 'min_hours_to_enroll']

    # Default course types — all courses
    if course_types is None:
        course_types = ['core', 'elective']

    # Normalize course types to Neo4j elective flag ('yes'/'no')
    normalized_types = []
    for ct in course_types:
        if ct.lower() in ['core', 'mandatory']:
            normalized_types.append('no')   # elective='no' means mandatory/core
        elif ct.lower() == 'elective':
            normalized_types.append('yes')  # elective='yes' means elective

    # Default programs
    if program_name is None:
        program_names = [
            'artificial intelligence & machine learning',
            'software & application development',
            'data science',
        ]
    elif isinstance(program_name, str):
        program_names = [program_name.lower()]
    else:
        program_names = [name.lower() for name in program_name]

    # Build query
    query_parts = [
        "MATCH (c:Course)",
        "MATCH (c)-[b:BELONGS_TO]->(p:Program)",
        "WHERE p.name IN $program_names",
    ]
    params = {'program_names': program_names, 'course_types': normalized_types}

    # Optional: filter by course name list
    if course_list is not None:
        course_list_lower = [name.lower() for name in course_list]
        query_parts.append("AND toLower(c.name) IN $course_list")
        params['course_list'] = course_list_lower

    # Optional: filter by course type
    if len(normalized_types) == 1:
        query_parts.append("AND b.elective IN $course_types")

    # Supported filter fields
    SUPPORTED_FIELDS = ['min_hours_to_enroll', 'credit_hours']
    where_conditions = []

    if filters:
        for field, value in filters.items():
            if field not in SUPPORTED_FIELDS:
                raise ValueError(
                    f"Unsupported filter field: '{field}'. Allowed: {SUPPORTED_FIELDS}"
                )
            if isinstance(value, dict):
                for operator, op_value in value.items():
                    param_name = (
                        f"{field}_{operator.replace('>', 'gt').replace('<', 'lt').replace('=', 'eq')}"
                    )
                    where_conditions.append(f"c.{field} {operator} ${param_name}")
                    params[param_name] = op_value
            else:
                param_name = field
                where_conditions.append(f"c.{field} = ${param_name}")
                params[param_name] = value

    if where_conditions:
        query_parts.append("AND " + " AND ".join(where_conditions))

    # Decide whether to include program_details in the output
    include_program_details = (
        'program_details' in return_fields
        or len(program_names) > 1
        or len(normalized_types) == 2
    )

    base_fields = [f for f in return_fields if f != 'program_details']

    if include_program_details:
        if 'code' in base_fields:
            query_parts.append(
                "WITH c, collect(DISTINCT {"
                "program: p.name, "
                "course_type: CASE WHEN b.elective = 'yes' THEN 'elective' ELSE 'core' END"
                "}) AS program_details, collect(b.code)[0] AS rel_code"
            )
        else:
            query_parts.append(
                "WITH c, collect(DISTINCT {"
                "program: p.name, "
                "course_type: CASE WHEN b.elective = 'yes' THEN 'elective' ELSE 'core' END"
                "}) AS program_details"
            )
        return_parts = [
            "COALESCE(c.code, rel_code) AS code" if field == 'code' else f"c.{field} AS {field}"
            for field in base_fields
        ]
        return_list = ", ".join(return_parts)
        query_parts.append(f"RETURN {return_list}, program_details")
    else:
        return_parts = [
            "COALESCE(c.code, b.code) AS code" if field == 'code' else f"c.{field} AS {field}"
            for field in base_fields
        ]
        return_clause = ", ".join(return_parts)
        query_parts.append(f"RETURN DISTINCT {return_clause}")

    if 'code' in base_fields:
        query_parts.append("ORDER BY code")
    elif 'name' in base_fields:
        query_parts.append("ORDER BY c.name")

    full_query = "\n".join(query_parts)
    return run_cypher_query(full_query, params)


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


def interactive_eligibility_check(course_name, program_name=None, student_id=os.getenv("STUDENT_ID")):
    """
    Interactive function that asks user for their progress and checks eligibility.
    Optionally saves academic details to Supabase if student_id is provided.
    
    Args:
        course_name: Name of the course to check
        program_name: Student's program (optional)
        student_id: Student ID for saving to Supabase (optional)
    
    Returns:
        Eligibility result dictionary
    """

    # ── Normalise the course name before anything else ───────────────────────
    try:
        from course_name_mapper import map_course_name as _ecmap
        _mapped = _ecmap(course_name)
        if _mapped and _mapped.lower() != course_name.lower():
            print(f"📚 [Eligibility] Normalised course: '{course_name}' → '{_mapped}'")
            course_name = _mapped
    except Exception:
        pass
    
    print(f"\n{'='*80}")
    print(f"ELIGIBILITY CHECK FOR: {course_name.upper()}")
    print(f"{'='*80}\n")
    
    # Get existing academic details if student_id provided
    existing_completed = []
    existing_credits = None
    existing_gpa = None
    
    if student_id:
        try:
            from chatbot_connector import ChatbotConnector
            connector = ChatbotConnector()
            academic_details = connector.get_academic_details(student_id)
            
            if academic_details:
                existing_completed = [c.lower() for c in academic_details.get('completed_courses', [])]
                existing_credits = academic_details.get('earned_credits')
                existing_gpa = academic_details.get('gpa')
                
                if existing_completed:
                    print(f"📚 Your completed courses on record: {', '.join(existing_completed)}")
                if existing_credits:
                    print(f"📊 Your earned credits on record: {existing_credits}")
                if existing_gpa:
                    print(f"🎓 Your GPA on record: {existing_gpa}")
                if existing_completed or existing_credits or existing_gpa:
                    print()
        except Exception as e:
            print(f"⚠ Could not retrieve existing academic details: {e}\n")
    
    # Get prerequisites first
    prerequisites = get_course_dependencies(course_name, program_name, dependents=False)['prerequisites']

    if not prerequisites:
        print(f"✅ No prerequisites required for {course_name}!")
        return {
            "eligible": True,
            "course": course_name,
            "message": "No prerequisites required",
            "missing_prerequisites": []
        }
    
    print(f"Found {len(prerequisites)} prerequisite(s):\n")
    
    # Display prerequisites
    course_prerequisites = []
    credit_requirement = None
    
    for idx, prereq in enumerate(prerequisites, 1):
        if 'Required_Credit_Hours' in prereq:
            credit_requirement = int(prereq['Required_Credit_Hours'])
            print(f"{idx}. Credit Requirement: {credit_requirement} credit hours")
        else:
            course_prerequisites.append(prereq)
            tracks = prereq.get('tracks', [])
            if isinstance(tracks, list) and tracks:
                track_info = f" (for {', '.join([t['program'] if isinstance(t, dict) else t for t in tracks])})"
            elif isinstance(tracks, str):
                track_info = f" (for {tracks})"
            else:
                track_info = ""
            
            # Mark if already completed
            prereq_name = prereq.get('name', 'Unknown').lower()
            status = " ✅ (Already completed)" if prereq_name in existing_completed else ""
            print(f"{idx}. [{prereq.get('code', 'N/A')}] {prereq.get('name', 'Unknown')}{track_info}{status}")
    
    print(f"\n{'='*80}\n")
    
    # Ask for user's progress
    completed_courses = existing_completed.copy()  # Start with existing courses
    earned_credits = existing_credits
    gpa = existing_gpa
    
    if course_prerequisites:
        # Find which prerequisites are NOT already in completed courses
        prereq_names = [p.get('name', '').lower() for p in course_prerequisites]
        missing_prereqs = [p for p in prereq_names if p not in existing_completed]
        
        if missing_prereqs:
            print("Please enter any additional courses you have completed from the prerequisites above (comma-separated):")
            print("Example: machine learning, databases")
            print("(Press Enter if you haven't completed any additional courses)")
            user_input = input("Additional completed courses: ").strip()
            
            if user_input:
                new_courses = [c.strip().lower() for c in user_input.split(',')]
                # ── Normalise each course name to canonical Neo4j form ────
                try:
                    from course_name_mapper import map_course_name as _icmap
                    normalised_new = []
                    for c in new_courses:
                        if c:
                            mapped = _icmap(c)
                            normalised_new.append((mapped or c).lower())
                    new_courses = normalised_new
                except Exception:
                    pass
                # Add new courses to the existing list (avoiding duplicates)
                for course in new_courses:
                    if course and course not in completed_courses:
                        completed_courses.append(course)
        else:
            print("✅ All prerequisite courses are already marked as completed!")
    
    if credit_requirement:
        if existing_credits is not None:
            print(f"\nYour current earned credits on record: {existing_credits}")
            user_input = input(f"Want to update? (Enter new value or press Enter to keep {existing_credits}): ").strip()
            if user_input:
                try:
                    earned_credits = int(user_input)
                except ValueError:
                    print(f"Invalid input. Keeping current value: {existing_credits}")
                    earned_credits = existing_credits
        else:
            user_input = input(f"\nHow many total credit hours have you earned? ").strip()
            if user_input:
                try:
                    earned_credits = int(user_input)
                except ValueError:
                    print("Invalid input. Assuming 0 credits.")
                    earned_credits = 0
    
    # Ask for GPA if we're saving to Supabase
    if student_id:
        if existing_gpa is not None:
            user_input = input(f"\nYour current GPA is {existing_gpa}. Want to update? (Enter new value or press Enter to keep): ").strip()
            if user_input:
                try:
                    gpa = float(user_input)
                except ValueError:
                    print(f"Invalid input. Keeping current GPA: {existing_gpa}")
                    gpa = existing_gpa
        else:
            user_input = input(f"\nWhat is your GPA? (optional, press Enter to skip): ").strip()
            if user_input:
                try:
                    gpa = float(user_input)
                except ValueError:
                    print("Invalid input. GPA not saved.")
                    gpa = None
    
    # Check eligibility
    result = check_course_eligibility(
        course_name,
        prerequisites=prerequisites,
        completed_courses=completed_courses,
        earned_credits=earned_credits,
        program_name=program_name
    )
    
    # Save to Supabase if student_id provided
    if student_id:
        try:
            from chatbot_connector import ChatbotConnector
            connector = ChatbotConnector()
            
            # Update academic details (this will merge with existing data)
            connector.update_student_progress(
                student_id=student_id,
                completed_courses=completed_courses,
                earned_credits=earned_credits,
                gpa=gpa
            )
            print(f"\n✓ Academic details saved to database for student {student_id}")
        except Exception as e:
            print(f"\n⚠ Warning: Could not save to database: {e}")
    
    # Display result
    print(f"\n{'='*80}")
    print(result["message"])
    print(f"{'='*80}\n")
    
    if not result["eligible"]:
        if result["missing_prerequisites"]:
            print("Missing prerequisite courses:")
            for prereq in result["missing_prerequisites"]:
                print(f"  ❌ [{prereq.get('code', 'N/A')}] {prereq.get('name', 'Unknown')}")
        
        if not result["credit_requirement_met"]:
            print(f"\nCredit Requirement:")
            print(f"  Required: {result['credit_requirement']} credits")
            print(f"  Your Credits: {result['earned_credits'] or 0} credits")
            print(f"  Shortfall: {result['credit_requirement'] - (result['earned_credits'] or 0)} credits")
    
    return result