"""
recommendation_service.py — Program & elective recommendation engine
=====================================================================
Scores programs and electives against a merged student preference vector
built from three sources:
    degree_preference  (0.45) — derived from uploaded transcript grades
    user_preference    (0.35) — filled by student at signup
    ai_preference      (0.20) — inferred by agent during chat

All profiles use the same 12-category taxonomy as preference_service.py.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

# ── Category taxonomy (12 keys) ───────────────────────────────────────────────

CATEGORIES = [
    "math",
    "probability_statistics",
    "programming",
    "software_engineering",
    "ai_ml",
    "data_management",
    "data_analysis",
    "theory",
    "networking_systems",
    "visual_computing",
    "language_text",
    "optimization",
]

# ── Source weights ─────────────────────────────────────────────────────────────

SOURCE_WEIGHTS = {
    "degree_preference": 0.45,
    "user_preference":   0.35,
    "ai_preference":     0.20,
}

# ── Program profile vectors ───────────────────────────────────────────────────

PROGRAM_PROFILES: Dict[str, Dict[str, float]] = {
    "artificial intelligence and machine learning": {
        "math":                   0.85,
        "probability_statistics": 0.75,
        "programming":            0.65,
        "software_engineering":   0.20,
        "ai_ml":                  1.00,
        "data_management":        0.25,
        "data_analysis":          0.40,
        "theory":                 0.55,
        "networking_systems":     0.20,
        "visual_computing":       0.70,
        "language_text":          0.65,
        "optimization":           0.60,
    },
    "software and application development": {
        "math":                   0.30,
        "probability_statistics": 0.10,
        "programming":            1.00,
        "software_engineering":   1.00,
        "ai_ml":                  0.15,
        "data_management":        0.65,
        "data_analysis":          0.30,
        "theory":                 0.40,
        "networking_systems":     0.60,
        "visual_computing":       0.25,
        "language_text":          0.10,
        "optimization":           0.15,
    },
    "data science": {
        "math":                   0.75,
        "probability_statistics": 0.90,
        "programming":            0.55,
        "software_engineering":   0.30,
        "ai_ml":                  0.55,
        "data_management":        0.75,
        "data_analysis":          1.00,
        "theory":                 0.35,
        "networking_systems":     0.25,
        "visual_computing":       0.25,
        "language_text":          0.40,
        "optimization":           0.65,
    },
}

# ── Elective catalogues ───────────────────────────────────────────────────────
# Each entry: {name: {code, credits, profile: {cat: weight}}}
# Electives with no meaningful category signal (e.g. english language) are omitted.

AIM_ELECTIVES: Dict[str, dict] = {
    "natural language understanding": {
        "code": "AIM411", "credits": 2,
        "profile": {"language_text": 0.9, "ai_ml": 0.7, "programming": 0.4},
    },
    "advanced computer vision": {
        "code": "AIM412", "credits": 2,
        "profile": {"visual_computing": 1.0, "ai_ml": 0.8, "math": 0.5},
    },
    "robot kinematics & dynamics": {
        "code": "AIM413", "credits": 2,
        "profile": {"math": 0.8, "optimization": 0.7, "ai_ml": 0.5},
    },
    "planning techniques for robotics": {
        "code": "AIM414", "credits": 2,
        "profile": {"ai_ml": 0.7, "optimization": 0.7, "theory": 0.5, "math": 0.4},
    },
    "speech processing": {
        "code": "AIM415", "credits": 2,
        "profile": {"language_text": 0.8, "ai_ml": 0.7, "math": 0.5},
    },
    "pattern recognition": {
        "code": "AIM416", "credits": 2,
        "profile": {"ai_ml": 0.8, "math": 0.6, "probability_statistics": 0.6, "visual_computing": 0.4},
    },
    "special topics in advanced artificial intelligence": {
        "code": "AIM417", "credits": 3,
        "profile": {"ai_ml": 1.0, "theory": 0.5},
    },
    "deep reinforcement learning": {
        "code": "AIM418", "credits": 3,
        "profile": {"ai_ml": 1.0, "optimization": 0.8, "math": 0.6, "programming": 0.5},
    },
    "internet of things": {
        "code": "AIM419", "credits": 2,
        "profile": {"networking_systems": 0.8, "programming": 0.6, "software_engineering": 0.4},
    },
    "natural language processing with deep learning": {
        "code": "AIM420", "credits": 2,
        "profile": {"language_text": 1.0, "ai_ml": 0.9, "programming": 0.5},
    },
    "big data analysis": {
        "code": "AIM421", "credits": 2,
        "profile": {"data_analysis": 0.9, "data_management": 0.7, "programming": 0.5},
    },
    "data mining": {
        "code": "AIM422", "credits": 2,
        "profile": {"data_analysis": 0.8, "ai_ml": 0.6, "math": 0.5},
    },
    "special topics in advanced machine learning": {
        "code": "AIM423", "credits": 2,
        "profile": {"ai_ml": 1.0, "math": 0.6, "programming": 0.5},
    },
    "advanced deep learning": {
        "code": "AIM424", "credits": 2,
        "profile": {"ai_ml": 1.0, "math": 0.6, "programming": 0.6},
    },
    "analysis of time series data": {
        "code": "AIM425", "credits": 3,
        "profile": {"probability_statistics": 0.8, "math": 0.7, "data_analysis": 0.7},
    },
    "stochastic processes": {
        "code": "AIM426", "credits": 3,
        "profile": {"probability_statistics": 1.0, "math": 0.8, "optimization": 0.4},
    },
    "knowledge representation & reasoning 2": {
        "code": "AIM427", "credits": 3,
        "profile": {"theory": 0.9, "ai_ml": 0.7, "math": 0.4},
    },
    "embedded systems": {
        "code": "AIM428", "credits": 3,
        "profile": {"networking_systems": 0.8, "programming": 0.7, "software_engineering": 0.4},
    },
    "computer programming with matlab": {
        "code": "BCS104", "credits": 3,
        "profile": {"programming": 0.8, "math": 0.6, "optimization": 0.4},
    },
}

SAD_ELECTIVES: Dict[str, dict] = {
    "internet of things": {
        "code": "AIM419", "credits": 2,
        "profile": {"networking_systems": 0.8, "programming": 0.6, "software_engineering": 0.4},
    },
    "computer programming with matlab": {
        "code": "BCS104", "credits": 3,
        "profile": {"programming": 0.8, "math": 0.6, "optimization": 0.4},
    },
    "assembly language": {
        "code": "SAD302", "credits": 3,
        "profile": {"programming": 0.9, "networking_systems": 0.5, "theory": 0.4},
    },
    "social media & digital marketing": {
        "code": "SAD314", "credits": 3,
        "profile": {"software_engineering": 0.5, "data_analysis": 0.3},
    },
    "open-source software development": {
        "code": "SAD315", "credits": 3,
        "profile": {"programming": 0.8, "software_engineering": 0.7},
    },
    "software design & architecture": {
        "code": "SAD316", "credits": 3,
        "profile": {"software_engineering": 1.0, "theory": 0.5, "programming": 0.5},
    },
    "software construction": {
        "code": "SAD417", "credits": 3,
        "profile": {"software_engineering": 0.9, "programming": 0.8},
    },
    "advanced databases": {
        "code": "SAD418", "credits": 3,
        "profile": {"data_management": 1.0, "programming": 0.5, "software_engineering": 0.4},
    },
    "ethical hacking": {
        "code": "SAD419", "credits": 3,
        "profile": {"networking_systems": 0.9, "programming": 0.6, "theory": 0.4},
    },
    "cloud computing": {
        "code": "SAD420", "credits": 3,
        "profile": {"networking_systems": 0.8, "software_engineering": 0.6, "programming": 0.4},
    },
    "enterprise architecture": {
        "code": "SAD421", "credits": 3,
        "profile": {"software_engineering": 0.9, "theory": 0.5, "data_management": 0.4},
    },
    "big data analytics": {
        "code": "SAD422", "credits": 3,
        "profile": {"data_analysis": 0.9, "data_management": 0.7, "programming": 0.4},
    },
    "agile methods": {
        "code": "SAD425", "credits": 3,
        "profile": {"software_engineering": 1.0, "programming": 0.4},
    },
    "software engineering 2": {
        "code": "SAD426", "credits": 3,
        "profile": {"software_engineering": 1.0, "programming": 0.7, "theory": 0.4},
    },
    "distributed systems": {
        "code": "SAD427", "credits": 3,
        "profile": {"networking_systems": 0.8, "software_engineering": 0.7, "programming": 0.5},
    },
    "game development": {
        "code": "SAD428", "credits": 3,
        "profile": {"programming": 0.9, "visual_computing": 0.7, "software_engineering": 0.5},
    },
    "software engineering for internet applications": {
        "code": "SAD429", "credits": 3,
        "profile": {"software_engineering": 0.9, "programming": 0.7, "networking_systems": 0.4},
    },
    "human computer interaction": {
        "code": "SAD430", "credits": 3,
        "profile": {"software_engineering": 0.7, "visual_computing": 0.5, "programming": 0.3},
    },
    "data mining": {
        "code": None, "credits": 2,
        "profile": {"data_analysis": 0.8, "ai_ml": 0.5, "math": 0.4},
    },
}

DAS_ELECTIVES: Dict[str, dict] = {
    "deep reinforcement learning": {
        "code": "AIM418", "credits": 3,
        "profile": {"ai_ml": 1.0, "optimization": 0.8, "math": 0.6, "programming": 0.5},
    },
    "analysis of time series data": {
        "code": "AIM425", "credits": 3,
        "profile": {"probability_statistics": 0.8, "math": 0.7, "data_analysis": 0.7},
    },
    "computer programming with matlab": {
        "code": "BCS104", "credits": 3,
        "profile": {"programming": 0.8, "math": 0.6, "optimization": 0.4},
    },
    "applied multivariate analysis": {
        "code": "DAS313", "credits": 3,
        "profile": {"probability_statistics": 0.9, "math": 0.8, "data_analysis": 0.7},
    },
    "biostatistics methods": {
        "code": "DAS314", "credits": 3,
        "profile": {"probability_statistics": 0.9, "math": 0.6, "data_analysis": 0.7},
    },
    "applied data science for cyber security": {
        "code": "DAS315", "credits": 3,
        "profile": {"data_analysis": 0.8, "networking_systems": 0.6, "programming": 0.4},
    },
    "natural language processing": {
        "code": "DAS316", "credits": 3,
        "profile": {"language_text": 1.0, "ai_ml": 0.7, "programming": 0.4},
    },
    "selected topics in data science": {
        "code": "DAS416", "credits": 3,
        "profile": {"data_analysis": 1.0, "math": 0.5, "ai_ml": 0.4},
    },
    "advanced optimization methods": {
        "code": "DAS417", "credits": 3,
        "profile": {"optimization": 1.0, "math": 0.9},
    },
    "mathematical modeling with applications": {
        "code": "DAS418", "credits": 3,
        "profile": {"math": 0.9, "optimization": 0.7, "probability_statistics": 0.4},
    },
    "decision support systems & business intelligence": {
        "code": "DAS419", "credits": 4,
        "profile": {"data_analysis": 0.9, "data_management": 0.5, "software_engineering": 0.5},
    },
    "advanced machine learning": {
        "code": "DAS420", "credits": 3,
        "profile": {"ai_ml": 1.0, "math": 0.7, "programming": 0.5},
    },
    "stochastic methods": {
        "code": "DAS422", "credits": 3,
        "profile": {"probability_statistics": 1.0, "math": 0.8, "optimization": 0.5},
    },
    "swarm intelligence algorithms": {
        "code": "DAS423", "credits": 3,
        "profile": {"optimization": 0.9, "ai_ml": 0.7, "math": 0.5},
    },
    "social network analysis": {
        "code": "DAS425", "credits": 3,
        "profile": {"data_analysis": 0.8, "networking_systems": 0.5, "math": 0.4},
    },
    "distributed systems": {
        "code": "SAD427", "credits": 3,
        "profile": {"networking_systems": 0.8, "software_engineering": 0.6, "programming": 0.5},
    },
    "computer vision": {
        "code": None, "credits": 3,
        "profile": {"visual_computing": 1.0, "ai_ml": 0.8, "math": 0.5},
    },
}

ELECTIVE_CATALOGUES = {
    "artificial intelligence & machine learning": AIM_ELECTIVES,
    "software & application development":        SAD_ELECTIVES,
    "data science":                                DAS_ELECTIVES,
}


# ── Core math helpers ─────────────────────────────────────────────────────────

def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = set(a) | set(b)
    dot  = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    norm_a = math.sqrt(sum(v ** 2 for v in a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return round(dot / (norm_a * norm_b), 4)


def _top_matching_categories(
    student: Dict[str, float],
    profile: Dict[str, float],
    n: int = 3,
) -> List[str]:
    """Return the n category names that contribute most to the cosine score."""
    contrib = {
        k: student.get(k, 0.0) * profile.get(k, 0.0)
        for k in set(student) | set(profile)
        if student.get(k, 0.0) > 0
    }
    return [k for k, _ in sorted(contrib.items(), key=lambda x: -x[1])][:n]


# Maps recommendation_service canonical names → Neo4j programme names (use & not and)
_NEO4J_PROGRAM = {
    "artificial intelligence and machine learning": "artificial intelligence & machine learning",
    "software and application development":        "software & application development",
    "data science":                                "data science",
}


def _get_course_context(course_name: str, program_name: str) -> str:
    """
    Fetch description + motivation from Neo4j for one course.
    Returns a single string ready to embed in the output, or "" on failure.
    """
    try:
        from neo4j_course_functions import get_course_info
        neo4j_prog = _NEO4J_PROGRAM.get(program_name, program_name)
        result = get_course_info(course_name, program_name=neo4j_prog)
        if not result:
            return ""
        # get_course_info returns a list when no program or a dict when program given
        info = result[0] if isinstance(result, list) else result
        parts = []
        desc = (info.get("description") or "").strip()
        mot  = (info.get("motivation")  or "").strip()
        if desc:
            parts.append(desc)
        if mot and mot.lower() != desc.lower():
            parts.append(mot)
        return "  ".join(parts)        # single line, two spaces between if both exist
    except Exception:
        return ""


def _eligibility_for(student_id: str, course_name: str) -> dict:
    """
    Call eligibility.check_course_eligibility and return the result dict.
    Returns a safe fallback dict on any error so the recommendation still renders.
    """
    try:
        from eligibility import check_course_eligibility
        return check_course_eligibility(student_id, course_name)
    except Exception:
        return {"eligible": None, "missing_prerequisites": [], "message": ""}


def _eligibility_line(result: dict) -> Tuple[str, str]:
    """
    Return (inline_tag, detail_line) for a given eligibility result dict.
      inline_tag  — appended to the score line, e.g. "✓ eligible"
      detail_line — shown on the next line when not eligible, e.g. "Missing: NLP"
    """
    if result.get("eligible") is None:
        return "", ""
    if result["eligible"]:
        return "✓ eligible", ""
    missing = [p.get("name", "?") for p in result.get("missing_prerequisites", [])]
    credit_req = result.get("credit_requirement")
    credit_met = result.get("credit_requirement_met", True)
    parts = []
    if missing:
        parts.append(f"missing: {', '.join(missing)}")
    if not credit_met:
        parts.append(
            f"need {credit_req} cr, have {result.get('earned_credits', '?')} cr"
        )
    detail = f"   Not eligible — {' | '.join(parts)}" if parts else ""
    return "✗ not eligible", detail



# ── Merge preferences ─────────────────────────────────────────────────────────

def merge_preferences(student_id: str) -> Tuple[Dict[str, float], List[str]]:
    """
    Read all 3 preference sources from Supabase and merge into one vector.

    Returns:
        (merged_vector, sources_used)
        merged_vector — {category: score 0–1}
        sources_used  — list of source names that had data
    """
    from preference_service import get_preferences

    prefs = get_preferences(student_id)
    sources_used: List[str] = []
    weighted_sum: Dict[str, float] = {}
    weight_total: Dict[str, float] = {}

    for source_key, weight in SOURCE_WEIGHTS.items():
        source_data: Dict[str, float] = prefs.get(source_key) or {}
        if not source_data:
            continue
        sources_used.append(source_key)
        for cat in CATEGORIES:
            val = float(source_data.get(cat, 0.0))
            weighted_sum[cat]  = weighted_sum.get(cat, 0.0)  + val * weight
            weight_total[cat]  = weight_total.get(cat, 0.0)  + weight

    if not sources_used:
        return {}, []

    merged = {
        cat: round(weighted_sum.get(cat, 0.0) / weight_total[cat], 3)
        for cat in CATEGORIES
        if weight_total.get(cat, 0.0) > 0
    }
    return merged, sources_used


# ── Program recommendation ────────────────────────────────────────────────────

def recommend_programs(student_id: str) -> str:
    """
    Score all 3 programs against the student's merged preference vector.
    Returns a formatted string for the answer LLM.
    Program names are canonical (pre-resolved by the preprocessor).
    """
    student_vec, sources = merge_preferences(student_id)

    if not student_vec:
        return (
            "No preference data found for this student yet. "
            "Ask them about their interests, strengths, or background to build a profile."
        )

    scores: List[Tuple[str, float, List[str]]] = []
    for prog_name, prog_profile in PROGRAM_PROFILES.items():
        score  = _cosine(student_vec, prog_profile)
        top_cats = _top_matching_categories(student_vec, prog_profile)
        scores.append((prog_name, score, top_cats))

    scores.sort(key=lambda x: -x[1])

    lines = [
        "PROGRAM RECOMMENDATION",
        "=" * 40,
        f"Sources used: {', '.join(s.replace('_', ' ') for s in sources)}",
        "",
    ]
    for rank, (prog, score, cats) in enumerate(scores, 1):
        pct = int(score * 100)
        cat_str = ", ".join(cats) if cats else "general profile"
        label = " ← top match" if rank == 1 else ""
        lines.append(f"{rank}. {prog.title()} — {pct}% match{label}")
        lines.append(f"   Key matching areas: {cat_str}")

    lines += [
        "",
        f"Top recommendation: {scores[0][0].title()}",
        f"Student preference highlights: " +
        ", ".join(
            f"{k}={v}"
            for k, v in sorted(student_vec.items(), key=lambda x: -x[1])[:4]
            if v > 0
        ),
    ]
    return "\n".join(lines)


# ── Elective recommendation ───────────────────────────────────────────────────

def recommend_electives(
    student_id: str,
    program_name: str,
    top_n: int = 5,
    course_names: Optional[List[str]] = None,
    skip_course_info: bool = False,
) -> str:
    """
    Two modes:

    Mode 1 — no course_names:
        Score all electives, walk down by score checking eligibility until
        top_n eligible ones are found. Only eligible electives are shown.

    Mode 2 — course_names provided:
        Head-to-head comparison of specific electives with eligibility status.
        Used by recommend_courses() after confirming the courses are electives.
        course_names and program_name are pre-resolved by the preprocessor.

    skip_course_info: set True only when called internally by compare_courses(),
        which already includes full course descriptions. Never pass True from
        an agent tool call — the agent should never set this flag.

    Returns a formatted string for the answer LLM.
    """
    canonical  = program_name.lower().strip()
    catalogue  = ELECTIVE_CATALOGUES.get(canonical, {})

    student_vec, sources = merge_preferences(student_id)
    if not student_vec:
        return (
            "No preference data found yet. "
            "Share your interests or background so I can recommend electives."
        )

    sources_str = ", ".join(s.replace("_", " ") for s in sources)

    # ── Mode 2: head-to-head comparison ──────────────────────────────────────
    if course_names:
        found: List[Tuple[str, float, dict, dict]] = []  # name, score, data, elig
        not_found: List[str] = []

        for raw in course_names:
            key  = raw.lower().strip()
            data = catalogue.get(key)
            if data is None:
                not_found.append(raw)
                continue
            score = _cosine(student_vec, data["profile"])
            elig  = _eligibility_for(student_id, key)
            found.append((key, score, data, elig))

        if not found:
            return f"Could not find any of the requested electives in the {canonical.title()} catalogue: {course_names}"

        found.sort(key=lambda x: -x[1])

        lines = ["ELECTIVE COMPARISON", "=" * 40, f"Sources used: {sources_str}", ""]

        for rank, (name, score, data, elig) in enumerate(found, 1):
            code     = data.get("code") or "—"
            credits  = data.get("credits", "?")
            pct      = int(score * 100)
            cats     = ", ".join(_top_matching_categories(student_vec, data["profile"])) or "general"
            tag, det = _eligibility_line(elig)
            label    = " ← better fit" if rank == 1 and len(found) > 1 else ""
            elig_tag = f"  {tag}" if tag else ""
            if skip_course_info:
                about = None
            else:
                about    = _get_course_context(name, canonical)
            lines.append(f"{rank}. {name.title()} ({code}, {credits} cr) — {pct}% match{elig_tag}{label}")
            lines.append(f"   Matching areas: {cats}")
            if about:
                lines.append(f"   About: {about}")
            if det:
                lines.append(det)

        lines.append("")

        best      = found[0]
        runner    = found[1] if len(found) > 1 else None
        all_ineligible = all(e[3].get("eligible") is False for e in found)

        if all_ineligible:
            rec = (
                f"You're not eligible for either option yet — both require prerequisites "
                f"you haven't completed. {best[0].title()} would be the better fit once "
                f"eligible. Would you like to see electives you can take right now?"
            )
        elif best[3].get("eligible") is True:
            rec = f"{best[0].title()} — better match and you're eligible to take it now."
        elif runner and runner[3].get("eligible") is True:
            rec = (
                f"{runner[0].title()} — lower match score but you're eligible now. "
                f"{best[0].title()} is a better fit but requires prerequisites first."
            )
        else:
            rec = f"{best[0].title()} fits your profile better."

        lines.append(f"Recommendation: {rec}")
        if not_found:
            lines.append(f"Could not find in catalogue: {', '.join(not_found)}")

        return "\n".join(lines)

    # ── Mode 1: walk down by score, stop once top_n eligible found ───────────
    if not catalogue:
        return f"No elective catalogue found for '{canonical}'."

    scored: List[Tuple[str, float, dict]] = sorted(
        [(name, _cosine(student_vec, data["profile"]), data) for name, data in catalogue.items()],
        key=lambda x: -x[1],
    )

    eligible_rows: List[Tuple[str, float, dict]] = []
    for name, score, data in scored:
        if len(eligible_rows) >= top_n:
            break
        elig = _eligibility_for(student_id, name)
        if elig.get("eligible") is not False:   # True or None (check error → include)
            eligible_rows.append((name, score, data))

    lines = [
        f"ELECTIVE RECOMMENDATION — {canonical.title()}",
        "=" * 40,
        f"Sources used: {sources_str}",
        "",
    ]

    if eligible_rows:
        for rank, (name, score, data) in enumerate(eligible_rows, 1):
            code    = data.get("code") or "—"
            credits = data.get("credits", "?")
            pct     = int(score * 100)
            cats    = ", ".join(_top_matching_categories(student_vec, data["profile"])) or "general"
            about   = _get_course_context(name, canonical)
            lines.append(f"{rank}. {name.title()} ({code}, {credits} cr) — {pct}% match")
            lines.append(f"   Matching areas: {cats}")
            if about:
                lines.append(f"   About: {about}")
        lines += ["", f"Top pick: {eligible_rows[0][0].title()} ({eligible_rows[0][2].get('code', '—')})"]
    else:
        lines.append("No eligible electives found based on your completed courses.")

    return "\n".join(lines)
