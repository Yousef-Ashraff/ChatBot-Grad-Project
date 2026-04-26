"""
degree_preference.py — Transcript-to-preference mapper
=======================================================
Takes OCR-extracted transcript grades and maps them to the 12 preference
categories, storing the result in the degree_preference column of the
student_preferences Supabase table.

Input format expected:
    [
        {"course": "machine learning",        "grade": "A"},
        {"course": "differential & integral calculus", "grade": "85"},
        {"course": "data structures",          "grade": "3.5"},
        ...
    ]

Grade formats handled automatically:
    Letter  : A+, A, A-, B+, B, B-, C+, C, C-, D+, D, F
    Percent : 0–100  (e.g. 85, 92.5)
    GPA     : 0.0–4.0 (e.g. 3.7)
"""

from __future__ import annotations

import os
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()

# ── Category taxonomy (mirrors preference_service.VALID_CATEGORIES) ──────────

VALID_CATEGORIES = {
    "math", "probability_statistics", "programming", "software_engineering",
    "ai_ml", "data_management", "data_analysis", "theory",
    "networking_systems", "visual_computing", "language_text", "optimization",
}

# ── Complete course → category-weight mapping ─────────────────────────────────
# Built from get_all_types_courses() output for all 3 programs.
# Format: "lowercase course name": {"category": weight, ...}
# Weights sum to 1.0 per course. Courses with no signal are absent.

COURSE_CATEGORY_MAP: Dict[str, Dict[str, float]] = {

    # ── BAS: Mathematics & Basic Sciences ────────────────────────────────────
    "differential & integral calculus":     {"math": 1.0},
    "physics":                              {"math": 0.3, "networking_systems": 0.3},
    "discrete mathematics":                 {"math": 0.6, "theory": 0.4},
    "linear algebra":                       {"math": 1.0},
    "electronics":                          {"networking_systems": 0.7, "math": 0.3},
    "probability & statistical methods":    {"probability_statistics": 1.0},
    "differential equations":              {"math": 1.0},
    "statistical analysis":                {"probability_statistics": 1.0},

    # ── BCS: Basic Computing Sciences ────────────────────────────────────────
    "computer science fundamentals":        {"programming": 0.6, "theory": 0.4},
    "structured programming":               {"programming": 1.0},
    "object oriented programming":          {"programming": 0.7, "software_engineering": 0.3},
    "computer programming with matlab":     {"programming": 0.7, "math": 0.3},
    "computer networks fundamentals":       {"networking_systems": 1.0},
    "fundamentals of databases":            {"data_management": 1.0},
    "data structures":                      {"programming": 0.6, "theory": 0.4},
    "logic design":                         {"theory": 0.7, "networking_systems": 0.3},
    "design & analysis of algorithms":      {"programming": 0.5, "theory": 0.5},
    "computer architecture":                {"networking_systems": 0.7, "theory": 0.3},
    "operating systems":                    {"networking_systems": 0.8, "theory": 0.2},
    "fundamentals of data science":         {"data_analysis": 0.7, "data_management": 0.3},
    "artificial intelligence":              {"ai_ml": 1.0},
    # technical report writing → no domain signal, omitted

    # ── AIM: Specialized Core ─────────────────────────────────────────────────
    "software engineering":                 {"software_engineering": 1.0},
    "microprocessors & assembly language":  {"networking_systems": 0.7, "programming": 0.3},
    "systems analysis & design":            {"software_engineering": 0.8, "data_management": 0.2},
    "machine learning":                     {"ai_ml": 1.0},
    "knowledge representation & reasoning 1": {"ai_ml": 0.7, "theory": 0.3},
    "robotics":                             {"ai_ml": 0.5, "optimization": 0.3, "networking_systems": 0.2},
    "neural networks":                      {"ai_ml": 1.0},
    "natural language processing":          {"language_text": 0.7, "ai_ml": 0.3},
    "image processing":                     {"visual_computing": 1.0},
    "deep learning":                        {"ai_ml": 1.0},
    "computer vision":                      {"visual_computing": 0.7, "ai_ml": 0.3},
    "data science":                         {"data_analysis": 0.6, "ai_ml": 0.4},
    # field training / graduation project → no domain signal, omitted

    # ── AIM: Electives ────────────────────────────────────────────────────────
    "natural language understanding":       {"language_text": 0.8, "ai_ml": 0.2},
    "advanced computer vision":             {"visual_computing": 0.8, "ai_ml": 0.2},
    "robot kinematics & dynamics":          {"optimization": 0.6, "math": 0.4},
    "planning techniques for robotics":     {"ai_ml": 0.6, "optimization": 0.4},
    "speech processing":                    {"language_text": 0.8, "ai_ml": 0.2},
    "pattern recognition":                  {"ai_ml": 0.8, "probability_statistics": 0.2},
    "special topics in advanced artificial intelligence": {"ai_ml": 1.0},
    "deep reinforcement learning":          {"ai_ml": 1.0},
    "internet of things":                   {"networking_systems": 0.8, "programming": 0.2},
    "natural language processing with deep learning": {"language_text": 0.6, "ai_ml": 0.4},
    "big data analysis":                    {"data_analysis": 0.7, "data_management": 0.3},
    "data mining":                          {"data_analysis": 0.8, "ai_ml": 0.2},
    "special topics in advanced machine learning": {"ai_ml": 1.0},
    "advanced deep learning":               {"ai_ml": 1.0},
    "analysis of time series data":         {"data_analysis": 0.6, "probability_statistics": 0.4},
    "stochastic processes":                 {"probability_statistics": 1.0},
    "knowledge representation & reasoning 2": {"ai_ml": 0.7, "theory": 0.3},
    "embedded systems":                     {"networking_systems": 0.8, "programming": 0.2},

    # ── SAD: Specialized Core ─────────────────────────────────────────────────
    "web development":                      {"software_engineering": 1.0},
    "computer graphics":                    {"visual_computing": 1.0},
    "software project management":          {"software_engineering": 1.0},
    "mobile application development":       {"software_engineering": 1.0},
    "software requirement analysis":        {"software_engineering": 1.0},
    "information storage & management":     {"data_management": 1.0},
    "software security":                    {"networking_systems": 0.7, "software_engineering": 0.3},
    "software testing & quality assurance": {"software_engineering": 1.0},

    # ── SAD: Electives ────────────────────────────────────────────────────────
    "assembly language":                    {"networking_systems": 0.7, "programming": 0.3},
    "open-source software development":     {"software_engineering": 0.8, "programming": 0.2},
    "software design & architecture":       {"software_engineering": 1.0},
    "software construction":                {"software_engineering": 0.7, "programming": 0.3},
    "advanced databases":                   {"data_management": 1.0},
    "ethical hacking":                      {"networking_systems": 1.0},
    "cloud computing":                      {"networking_systems": 0.8, "software_engineering": 0.2},
    "enterprise architecture":              {"software_engineering": 1.0},
    "big data analytics":                   {"data_analysis": 0.7, "data_management": 0.3},
    "agile methods":                        {"software_engineering": 1.0},
    "software engineering 2":              {"software_engineering": 1.0},
    "distributed systems":                  {"networking_systems": 0.7, "software_engineering": 0.3},
    "game development":                     {"software_engineering": 0.7, "visual_computing": 0.3},
    "software engineering for internet applications": {"software_engineering": 1.0},
    "human computer interaction":           {"software_engineering": 0.8, "visual_computing": 0.2},
    # social media & digital marketing → no technical domain signal, omitted

    # ── DAS: Specialized Core ─────────────────────────────────────────────────
    "data visualization & data-driven decision-making": {"data_analysis": 1.0},
    "numerical methods":                    {"math": 0.6, "optimization": 0.4},
    "big data technologies":                {"data_management": 0.7, "data_analysis": 0.3},
    "applied regression methods":           {"probability_statistics": 0.7, "data_analysis": 0.3},
    "modeling & simulation":                {"math": 0.4, "probability_statistics": 0.4, "optimization": 0.2},
    "optimization methods":                 {"optimization": 1.0},
    "text mining":                          {"language_text": 0.6, "data_analysis": 0.4},
    "large-scale data analysis":            {"data_analysis": 0.8, "data_management": 0.2},

    # ── DAS: Electives ────────────────────────────────────────────────────────
    "applied multivariate analysis":        {"probability_statistics": 0.8, "data_analysis": 0.2},
    "biostatistics methods":                {"probability_statistics": 1.0},
    "applied data science for cyber security": {"data_analysis": 0.6, "networking_systems": 0.4},
    "selected topics in data science":      {"data_analysis": 1.0},
    "advanced optimization methods":        {"optimization": 1.0},
    "mathematical modeling with applications": {"math": 0.6, "optimization": 0.4},
    "decision support systems & business intelligence": {"data_analysis": 0.8, "data_management": 0.2},
    "advanced machine learning":            {"ai_ml": 1.0},
    "stochastic methods":                   {"probability_statistics": 1.0},
    "swarm intelligence algorithms":        {"optimization": 0.7, "ai_ml": 0.3},
    "social network analysis":              {"data_analysis": 0.7, "networking_systems": 0.3},
}

# ── Grade normalization ───────────────────────────────────────────────────────

_LETTER_GRADE_MAP = {
    "a+": 1.00, "a": 0.97, "a-": 0.93,
    "b+": 0.87, "b": 0.83, "b-": 0.80,
    "c+": 0.77, "c": 0.73, "c-": 0.70,
    "d+": 0.67, "d": 0.63,
    "f": 0.00, "fail": 0.00, "w": None, "i": None,  # W=withdrawn, I=incomplete → skip
}


def _normalize_grade(raw: str) -> float | None:
    """
    Convert any grade representation to a float in [0.0, 1.0].
    Returns None if the grade should be skipped (withdrawn, incomplete, etc.).
    """
    if raw is None:
        return None

    s = str(raw).strip().lower()

    # Letter grade
    if s in _LETTER_GRADE_MAP:
        return _LETTER_GRADE_MAP[s]

    # Numeric
    try:
        val = float(s.replace("%", ""))
    except ValueError:
        return None

    if val > 4.0:
        # Percentage scale (0–100)
        return max(0.0, min(1.0, val / 100.0))
    else:
        # GPA scale (0.0–4.0)
        return max(0.0, min(1.0, val / 4.0))


def _normalize_course_name(name: str) -> str:
    """Lowercase + strip for map lookup."""
    return name.strip().lower()


# ── Core processing function ──────────────────────────────────────────────────

def compute_degree_preference(
    transcript: List[Dict],
) -> Dict[str, float]:
    """
    Convert a list of {course, grade} entries into a degree_preference dict.

    Algorithm:
        For each category, collect all (grade_score × category_weight) values
        from courses that touch it, then take the weighted average.

        category_score = Σ(grade_i × w_i) / Σ(w_i)

    Args:
        transcript: list of {"course": str, "grade": str} dicts.

    Returns:
        Dict[category, score] — only categories with at least one matched course.
        Scores are floated and rounded to 3 decimal places.
    """
    # numerator and denominator per category for weighted average
    num: Dict[str, float] = {}
    den: Dict[str, float] = {}

    for entry in transcript:
        course_raw = entry.get("course", "")
        grade_raw  = entry.get("grade", "")

        grade = _normalize_grade(grade_raw)
        if grade is None:
            continue  # skip withdrawn / incomplete

        course_key = _normalize_course_name(course_raw)
        categories = COURSE_CATEGORY_MAP.get(course_key)

        if not categories:
            continue  # unmapped course — skip silently

        for cat, weight in categories.items():
            contribution = grade * weight
            num[cat] = num.get(cat, 0.0) + contribution
            den[cat] = den.get(cat, 0.0) + weight

    return {
        cat: round(num[cat] / den[cat], 3)
        for cat in num
        if den[cat] > 0
    }


def save_degree_preference(student_id: str, transcript: List[Dict]) -> Dict[str, float]:
    """
    Compute degree_preference from transcript and upsert into Supabase.

    Args:
        student_id: The student's ID.
        transcript: list of {"course": str, "grade": str} dicts.

    Returns:
        The computed degree_preference dict.
    """
    scores = compute_degree_preference(transcript)
    if not scores:
        return {}

    from supabase import create_client
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in .env")

    client = create_client(url, key)
    client.table("student_preferences").upsert(
        {"student_id": student_id, "degree_preference": scores},
        on_conflict="student_id",
    ).execute()

    return scores
