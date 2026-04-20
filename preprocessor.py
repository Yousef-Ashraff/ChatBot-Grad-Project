"""
preprocessor.py — Query Preprocessing Pipeline
================================================

Runs BEFORE the agent on every user message.  The agent only ever receives
a fully resolved, canonically-named query — it never has to deal with
abbreviations, pronouns, typos, or ambiguous names.

Pipeline (in order)
───────────────────
  Step 1 — Reference resolution
      "what about it?" + history → "what about machine learning?"
      Uses the LLM to replace pronouns / vague references with the actual
      entity they refer to, based on the last few conversation turns.

  Step 2 — Entity extraction
      Uses the LLM to pull out every course mention and track mention
      from the (now-dereferenced) query as a JSON list.
      Example: "prereqs of ml in aim" → courses=["ml"], tracks=["aim"]

  Step 3 — Course name mapping
      For each extracted course mention:
        a) Check COURSE_ALIASES first  → instant, no DB call
        b) If not found → fuzzy-match against live Neo4j course list
        c) If ONE clear winner  → replace silently
        d) If MULTIPLE close matches  → STOP, ask the user to pick one
           (agent never runs until the user clarifies)

  Step 4 — Track name mapping
      Same logic, but uses TRACK_ALIASES + the Neo4j Program list.
      Tracks are less ambiguous so we always auto-pick the best match.

  Step 5 — Query rewriting
      Uses the LLM to substitute the resolved names back into the original
      query naturally, producing a clean sentence for the agent.

Return values
─────────────
  PreprocessResult with .status ==

    "ready"     → .clean_query  is the fully resolved query string.
                  Pass this directly to BNUAdvisorAgent.run().

    "ambiguous" → .clarification  is a question string to show the student.
                  .pending  holds everything needed to continue once the
                  student answers.  Store it in chatbot_api._ambiguity_sessions.

    "passthrough" → no entities found, query is passed as-is.  Happens for
                    policy questions ("what is the GPA rule?") that contain
                    no course or track names.

Ambiguity continuation
──────────────────────
  When the student answers the clarification question, call:
      preprocessor.resolve_ambiguity(pending, student_reply)
  This returns a fresh PreprocessResult (always "ready" or "passthrough").
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

from llm_client import llm_call_json, llm_call_text
from debug_box import box

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Alias dictionaries — checked FIRST, no DB call needed
# ═════════════════════════════════════════════════════════════════════════════

# ── Course aliases ────────────────────────────────────────────────────────────
# Map every common abbreviation / shorthand → canonical Neo4j course name.
# All keys are lowercase.  Add more as students use them.

COURSE_ALIASES: Dict[str, str] = {
    # AI / ML
    "ml":              "machine learning",
    "ai":              "artificial intelligence",
    "dl":              "deep learning",
    "cv":              "computer vision",
    "nlp":             "natural language processing",
    "rl":              "reinforcement learning",
    "nn":              "neural networks",
    "aml":             "advanced machine learning",

    # Data science
    "ds":              "data science",
    "da":              "data analytics",
    "dw":              "data warehousing",
    "dm":              "data mining",
    "bi":              "business intelligence",
    "ts":              "analysis of time series data",
    "mv":              "applied multivariate analysis",

    # Software / systems
    "se":              "software engineering",
    "swe":             "software engineering",
    "os":              "operating systems",
    "db":              "database systems",
    "dbs":             "database systems",
    "cn":              "computer networks",
    "net":             "computer networks",
    "oop":             "object oriented programming",
    "dsa":             "data structures & algorithms",
    "algo":            "algorithms",
    "alg":             "algorithms",
    "ds&a":            "data structures & algorithms",

    # Core / foundations
    "prob":            "probability & statistical methods",
    "stats":           "probability & statistical methods",
    "stat":            "probability & statistical methods",
    "calc":            "calculus",
    "la":              "linear algebra",
    "linalg":          "linear algebra",
    "dm":              "discrete mathematics",
    "discmath":        "discrete mathematics",

    # Specific BNU course codes (students often type the code)
    "bcs311":          "artificial intelligence",
    "bas201":          "probability & statistical methods",
    "aim401":          "deep learning",
    "aim402":          "computer vision",
    "aim403":          "data science",
    "aim416":          "pattern recognition",
    "aim415":          "speech processing",
    "aim425":          "analysis of time series data",
    "aim423":          "special topics in advanced machine learning",
    "das420":          "advanced machine learning",
    "das313":          "applied multivariate analysis",
}

# ── Track / program aliases ───────────────────────────────────────────────────
# Map shorthand → canonical Neo4j program name (all lowercase).

TRACK_ALIASES: Dict[str, str] = {
    # AI & ML track
    "aim":                      "artificial intelligence & machine learning",
    "ai":                       "artificial intelligence & machine learning",
    "ai track":                 "artificial intelligence & machine learning",
    "ai and ml":                "artificial intelligence & machine learning",
    "ai & ml":                  "artificial intelligence & machine learning",
    "aiml":                     "artificial intelligence & machine learning",
    "machine learning track":   "artificial intelligence & machine learning",
    "ml track":                 "artificial intelligence & machine learning",

    # Software track
    "sad":                      "software & application development",
    "software":                 "software & application development",
    "software dev":             "software & application development",
    "software development":     "software & application development",
    "app dev":                  "software & application development",
    "sw":                       "software & application development",

    # Data science track
    "das":                      "data science",
    "data science":             "data science",
    "data":                     "data science",
    "ds track":                 "data science",
}

# ── Pre-defined course candidates for known track/course conflict terms ───────
# Checked in _map_course() BEFORE the live Neo4j fuzzy search.
# Benefits:
#   1. Faster — no DB round-trip for these common terms.
#   2. Complete — ensures ALL relevant courses are shown (fuzzy search has a
#      top-N cap and score threshold that may miss some).
#
# Resolution rules in _map_course():
#   • Single entry      OR
#     first entry name == search term (exact collision)
#       → status="resolved"  (single clear winner, no disambiguation needed)
#   • Multiple entries AND first entry name ≠ search term
#       → status="ambiguous" (show the full list to the student)
#
# Courses are ordered by relevance (most likely interpretation first).

KNOWN_CONFLICT_COURSES: Dict[str, List[Dict]] = {

    # ── "ai" → single winner: Artificial Intelligence (BCS311) ────────────────
    "ai": [
        {"name": "artificial intelligence", "code": "BCS311", "confidence": 1.00},
    ],

    # ── "soft" / "software" → all courses containing "software" ───────────────
    "soft": [
        {"name": "software engineering",                          "code": "AIM301", "confidence": 0.92},
        {"name": "software engineering 2",                        "code": "SAD426", "confidence": 0.88},
        {"name": "software project management",                   "code": "SAD306", "confidence": 0.87},
        {"name": "software requirement analysis",                 "code": "SAD308", "confidence": 0.87},
        {"name": "software design & architecture",                "code": "SAD316", "confidence": 0.85},
        {"name": "software construction",                         "code": "SAD417", "confidence": 0.84},
        {"name": "software security",                             "code": "SAD410", "confidence": 0.84},
        {"name": "software testing & quality assurance",          "code": "SAD412", "confidence": 0.83},
        {"name": "open-source software development",              "code": "SAD315", "confidence": 0.80},
        {"name": "software engineering for internet applications","code": "SAD429", "confidence": 0.80},
    ],

    # ── "data" → all courses with "data" as a primary word ────────────────────
    "data": [
        {"name": "data science",                                    "code": "AIM403", "confidence": 0.94},
        {"name": "data mining",                                     "code": "AIM422", "confidence": 0.92},
        {"name": "data visualization & data-driven decision-making","code": "DAS303", "confidence": 0.88},
        {"name": "fundamentals of data science",                    "code": "BCS213", "confidence": 0.87},
        {"name": "applied data science for cyber security",         "code": "DAS315", "confidence": 0.85},
        {"name": "selected topics in data science",                 "code": "DAS416", "confidence": 0.83},
        {"name": "big data analysis",                               "code": "AIM421", "confidence": 0.82},
        {"name": "big data analytics",                              "code": "SAD422", "confidence": 0.82},
        {"name": "big data technologies",                           "code": "DAS306", "confidence": 0.80},
        {"name": "large-scale data analysis",                       "code": "DAS412", "confidence": 0.78},
        {"name": "data structures",                                 "code": "BCS206", "confidence": 0.72},
    ],

    # ── "data science" → exact collision: one dominant course (AIM403) ────────
    # Single entry → always "resolved" directly; no multi-course disambiguation.
    # Students who mean a different data-science course type the full name.
    "data science": [
        {"name": "data science", "code": "AIM403", "confidence": 1.00},
    ],
}

# "software" reuses the same candidate list as "soft"
KNOWN_CONFLICT_COURSES["software"] = KNOWN_CONFLICT_COURSES["soft"]


# ── Track / Course conflict resolution ───────────────────────────────────────
# Terms where the canonical course name equals the canonical track name exactly.
# These need a higher confidence threshold to auto-resolve (the names are
# identical, so any mistake is invisible to the student).
EXACT_COLLISION_TERMS: set = {"data science"}

# Minimum intent-signal confidence needed to auto-resolve a track/course conflict.
INTENT_SIGNAL_THRESHOLD   = 0.70
# Higher bar when the course and track share the exact same canonical name.
EXACT_COLLISION_THRESHOLD = 0.85

# ── Intent signal patterns ────────────────────────────────────────────────────
# Used by _score_intent_context() to identify course- vs track-level context.

_COURSE_INTENT_PATTERNS: List[Tuple[str, float]] = [
    (r"\bcan\s+i\s+take\b",                    0.92),
    (r"\b(take|register\s+for|enroll\s+in)\b", 0.90),
    (r"\bwhen\s+(is|can\s+i|do\s+i)\b",        0.88),
    (r"\b(prerequisites?|prereqs?)\b",          0.90),
    (r"\beligibl",                              0.90),
    (r"\bhow\s+many\s+(credits?|hours?)\b",     0.85),
    (r"\b(unlocks?|closes?|opens?)\b",          0.85),
    (r"\bdescription\b",                        0.75),
    (r"\bwhat\s+is\b",                          0.68),
]

_TRACK_INTENT_PATTERNS: List[Tuple[str, float]] = [
    (r"\b(tracks?|programs?|majors?)\b",                      1.00),
    (r"\belectives?\b",                                       0.87),
    (r"\bcurriculum\b",                                       0.87),
    (r"\bcourses?\s+in\b",                                    0.87),
    (r"\bcourses?\s+(?:\w+\s+){0,3}in\b",                     0.87),  # "courses are in", "courses offered in", "courses available in"
    (r"\bcourses?\s+(?:\w+\s+){0,3}of\b",                    0.87),  # "courses of SAD", "courses part of"
    (r"\bstudents?\b",                                        0.85),
    (r"\bi'?m\s+(in|enrolled)\b",                             0.85),
    (r"\benrolled\s+in\b",                                    0.85),
    (r"\bgraduat",                                            0.80),
    (r"\bin\s+the\b",                                         0.78),
]


def _score_intent_context(text: str) -> Tuple[float, float]:
    """
    Score a text window for course-vs-track intent signals.
    Returns (course_confidence, track_confidence).
    Module-level so it can be used without a class instance.
    """
    cs = max((c for p, c in _COURSE_INTENT_PATTERNS if re.search(p, text)), default=0.0)
    ts = max((c for p, c in _TRACK_INTENT_PATTERNS  if re.search(p, text)), default=0.0)
    return cs, ts


# ── Entity blocklist ──────────────────────────────────────────────────────────
# Words the LLM sometimes extracts as course or track names but are NOT
# actual entity names.  Any extracted term whose lowercase form appears in
# this set is silently discarded before course/track mapping begins.

ENTITY_BLOCKLIST: set = {
    # Query structural words
    "courses", "course", "electives", "elective", "subjects", "subject",
    "classes", "class", "modules", "module", "lecture", "lectures",
    # Academic calendar words
    "semester", "semesters", "term", "terms", "year", "years",
    "first", "second", "third", "fourth",
    "fall", "spring", "summer",
    # Program / university structure
    "program", "programs", "track", "tracks", "major", "majors",
    "university", "college", "department", "faculty", "school",
    # Prerequisite / requirement words
    "prerequisites", "prerequisite", "prereq", "prereqs",
    "requirements", "requirement", "credits", "credit", "hours",
    # General question words that leak into entity lists
    "information", "details", "list", "overview", "description",
    "time", "timing", "schedule", "date", "when", "what", "how",
    "all", "any", "some", "these", "those", "this", "that",
}


# ═════════════════════════════════════════════════════════════════════════════
# Result types
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class PendingAmbiguity:
    """
    Stored in chatbot_api._ambiguity_sessions while waiting for the student
    to resolve an ambiguous course name OR a course-vs-track conflict.

    ambiguity_type == "course_name"   (original behaviour)
        candidates holds the close-matching course names.

    ambiguity_type == "course_vs_track"  (new)
        course_canonical / track_canonical hold the two options.
        candidates is empty ([]).
        pending_courses holds remaining unmapped course terms.
        pending_track_course_conflicts holds other unresolved course-vs-track
          conflicts in the same query: {original_term: (course_canon, track_canon)}.
    """
    original_query:   str                    # raw user message
    dereferenced:     str                    # after Step 0 (reference resolution)
    ambiguous_term:   str                    # the term the student typed, e.g. "ai"
    candidates:       List[Dict]             # [{name, code, confidence}, ...] — course_name type
    resolved_courses: Dict[str, str]         # already-resolved course mappings so far
    pending_courses:  List[Tuple[str, int, str]]  # remaining course occurrences not yet mapped: [(original, char_pos, deduped), ...]
    resolved_tracks:  Dict[str, str]         # already-resolved track mappings
    history:          List[Dict]             # conversation history at time of query
    term_position:    int = -1               # char position of ambiguous_term in dereferenced query (-1 = unknown)
    # ── New fields (all optional with defaults for backward compatibility) ──
    ambiguity_type:   str = "course_name"    # "course_name" | "course_vs_track"
    course_canonical: Optional[str] = None  # for course_vs_track: the course option
    track_canonical:  Optional[str] = None  # for course_vs_track: the track option
    student_track:    Optional[str] = None  # student's enrolled canonical track (for heuristic)
    pending_track_course_conflicts: List[Tuple[str, int, str, str]] = field(default_factory=list)
    # ^ remaining unresolved course-vs-track conflicts: [(original, char_pos, course_canon, track_canon)]
    #   List (not dict) so two occurrences of the same surface text at different positions
    #   are stored independently.
    resolved_course_occ: List[Tuple[str, int, str]] = field(default_factory=list)
    # ^ (original_term, char_pos, canonical_course_name) for every occurrence resolved as a course
    #   used for position-aware skip checks and positional rewriting
    pending_track_occurrences: List[Tuple[str, int, str]] = field(default_factory=list)
    # ^ (original_term, char_pos, deduped_term) for track occurrences still needing Step 4 mapping
    #   populated when process() returns early so the mapping happens post-disambiguation

    def __post_init__(self) -> None:
        """Normalize None → safe defaults (guards against accidental None assignment)."""
        if self.term_position is None:          # type: ignore[comparison-overlap]
            self.term_position = -1


@dataclass
class PreprocessResult:
    """Return value of QueryPreprocessor.process()."""
    status:           str                       # "ready" | "ambiguous" | "passthrough"
    clean_query:      str          = ""         # fully resolved query (when ready/passthrough)
    clarification:    str          = ""         # question to show student (when ambiguous)
    pending:          Optional[PendingAmbiguity] = None
    resolved_courses: Dict[str, str] = field(default_factory=dict)  # original → canonical


# ═════════════════════════════════════════════════════════════════════════════
# Fuzzy matching helpers
# ═════════════════════════════════════════════════════════════════════════════

def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _prefix_score(inp: str, name: str) -> float:
    """Bonus score when the input is a clear prefix of the name."""
    i, n = inp.lower().strip(), name.lower().strip()
    if len(i) < 3:
        return 0.0
    if n.startswith(i):
        return 0.50 + (len(i) / max(len(n), 1)) * 0.40
    return 0.0


def _score_course(term: str, course_name: str, course_code: str) -> float:
    """Overall match score for one (term, course) pair."""
    term_upper = term.upper().strip()
    if term_upper == (course_code or ""):
        return 1.0
    if term.lower() == course_name.lower():
        return 1.0

    sim     = _similarity(term, course_name)
    prefix  = _prefix_score(term, course_name)
    keyword = _keyword_overlap(term, course_name)
    seq_kw  = sim * 0.6 + keyword * 0.4
    return max(seq_kw, prefix)


def _keyword_overlap(term: str, name: str) -> float:
    stop = {"and", "the", "with", "for", "of", "in", "to", "a", "an"}
    tw = {w for w in term.lower().split() if w not in stop}
    nw = {w for w in name.lower().split() if w not in stop}
    if not tw:
        return 0.0
    return len(tw & nw) / len(tw)


# ═════════════════════════════════════════════════════════════════════════════
# Neo4j data loaders (cached in module scope)
# ═════════════════════════════════════════════════════════════════════════════

_COURSE_CACHE: Optional[List[Dict]] = None
_TRACK_CACHE:  Optional[List[str]]  = None


def _load_courses() -> List[Dict]:
    """Return [{name, code, name_lower}, ...] from Neo4j, cached."""
    global _COURSE_CACHE
    if _COURSE_CACHE is not None:
        return _COURSE_CACHE
    try:
        from neo4j_course_functions import run_cypher_query
        rows = run_cypher_query(
            "MATCH (c:Course) OPTIONAL MATCH (c)-[r:BELONGS_TO]->(:Program) "
            "WITH c, collect(r.code)[0] AS rel_code "
            "RETURN c.name AS name, COALESCE(c.code, rel_code) AS code ORDER BY c.name"
        )
        _COURSE_CACHE = [
            {"name": r["name"], "code": r["code"] or "",
             "name_lower": (r["name"] or "").lower()}
            for r in (rows or [])
        ]
        logger.info("Preprocessor: loaded %d courses from Neo4j", len(_COURSE_CACHE))
    except Exception as exc:
        logger.warning("Preprocessor: could not load course list — %s", exc)
        _COURSE_CACHE = []
    return _COURSE_CACHE


def _load_tracks() -> List[str]:
    """Return list of canonical program names from Neo4j, cached."""
    global _TRACK_CACHE
    if _TRACK_CACHE is not None:
        return _TRACK_CACHE
    try:
        from neo4j_course_functions import run_cypher_query
        rows = run_cypher_query(
            "MATCH (p:Program) RETURN p.name AS name ORDER BY p.name"
        )
        _TRACK_CACHE = [(r["name"] or "").lower() for r in (rows or []) if r["name"]]
        logger.info("Preprocessor: loaded %d tracks from Neo4j", len(_TRACK_CACHE))
    except Exception as exc:
        logger.warning("Preprocessor: could not load track list — %s", exc)
        _TRACK_CACHE = []
    return _TRACK_CACHE


def refresh_caches() -> None:
    """Force reload of course and track caches (e.g. after DB update)."""
    global _COURSE_CACHE, _TRACK_CACHE
    _COURSE_CACHE = None
    _TRACK_CACHE  = None
    _load_courses()
    _load_tracks()


# ═════════════════════════════════════════════════════════════════════════════
# Main preprocessor class
# ═════════════════════════════════════════════════════════════════════════════

class QueryPreprocessor:
    """
    Stateless — create once, call process() on every user message.
    All per-request state is passed in / returned, never stored on self.
    """

    # Ambiguity threshold: two candidates are "too close to pick automatically"
    # when their confidence scores are within this delta of each other.
    AMBIGUITY_DELTA = 0.08

    # Minimum confidence to even consider a fuzzy match.
    MATCH_THRESHOLD = 0.30

    # Maximum candidates to show the student in a disambiguation prompt.
    MAX_CANDIDATES = 5

    # ── Public entry point ────────────────────────────────────────────────

    def process(
        self,
        query:         str,
        history:       List[Dict],
        student_track: Optional[str] = None,
    ) -> PreprocessResult:
        """
        Preprocessing pipeline — 5 active steps:

          Step 0 — Reference resolution
              Replace pronouns / vague references ("it", "them", "this", etc.)
              with the actual entity from the conversation history.
              All downstream steps operate on the resolved query, not the raw one.

          Step 1 — Entity extraction
              Extract every course / track term from the RESOLVED query.

          Step 2 — Char deduplication
              Collapse repeated chars: "mmml" → "ml", "artifccial" → "artifcial".

          Step 3 — Course mapping
              alias table first → fuzzy DB match → ambiguity check.

          Step 4 — Track mapping
              Same logic, word-level fuzzy matching included.

          Step 5 — Query rewriting
              Substitute canonical names into the resolved query.
        """

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Step 0 — Reference resolution
        # Resolve BEFORE anything else so all downstream steps work on the
        # dereferenced text.  The resolved query is used everywhere below
        # instead of the original raw query.
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        resolved = self._resolve_references(query, history)

        box(
            "📝  STEP 0 — Reference Resolution",
            [
                f"Original : {query}",
                f"Resolved : {resolved}"
                + ("  (no change)" if resolved == query else "  ✓ changed"),
            ],
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Step 1 — Entity extraction  (runs on resolved query)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Returns List[Tuple[str, int]] per category — one entry per
        # OCCURRENCE: the same term appearing twice → two independent entries.
        raw_courses, raw_tracks = self._extract_entities(resolved)
        # raw_courses / raw_tracks: [(term, char_pos), ...]

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Step 2 — Entity filtering + Char deduplication
        # ─────────────────────────────────────────────────────────────────
        # Filter: discard any term whose lowercase form is in ENTITY_BLOCKLIST
        # (generic words like "electives", "semester", "year" that are not
        # actual course or track names).
        # Occurrence lists: [(original, char_pos, deduped), ...]
        # Each occurrence is INDEPENDENT even when the surface text repeats.
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        raw_courses = [(t, p) for t, p in raw_courses if t.lower().strip() not in ENTITY_BLOCKLIST]
        raw_tracks  = [(t, p) for t, p in raw_tracks  if t.lower().strip() not in ENTITY_BLOCKLIST]

        # Build occurrence triples: (original, char_pos, deduped)
        course_occurrences: List[Tuple[str, int, str]] = [
            (t, p, self._dedupe_chars(t)) for t, p in raw_courses
        ]
        track_occurrences: List[Tuple[str, int, str]] = [
            (t, p, self._dedupe_chars(t)) for t, p in raw_tracks
        ]

        dedup_notes = (
            [f'Course  "{o}" (pos {p}) -> "{d}"' for o, p, d in course_occurrences if o != d]
            + [f'Track   "{o}" (pos {p}) -> "{d}"' for o, p, d in track_occurrences if o != d]
        )

        step12_lines = [
            f"Working query : {resolved}",
            f"Courses found : {[(t, p) for t, p, _ in course_occurrences] or '(none)'}",
            f"Tracks  found : {[(t, p) for t, p, _ in track_occurrences]  or '(none)'}",
        ]
        if dedup_notes:
            step12_lines += ["", "Char deduplication:"] + [f"  {n}" for n in dedup_notes]

        box("📝  STEP 1&2 — Entity Extraction + Char Deduplication", step12_lines)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Step 2.5 — Track / Course conflict resolution
        # Each OCCURRENCE is resolved INDEPENDENTLY using the local ±6-word
        # context window around that occurrence's position.
        # Same surface text at different positions may receive different
        # resolutions (e.g. "ai at pos 25" → track, "ai at pos 44" → course).
        #
        # Two sources of conflict:
        #   Part A — course occurrences also in TRACK_ALIASES
        #             (e.g. "ai" in both COURSE_ALIASES and TRACK_ALIASES)
        #   Part B — track occurrences that also fuzzy-match a course
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        conflict_debug: List[str] = []
        # (original, pos, deduped) occurrences to move between lists
        to_move_to_track:  List[Tuple[str, int, str]] = []
        to_move_to_course: List[Tuple[str, int, str]] = []
        # unresolved: (original, pos, deduped, course_canon, track_canon, clarification)
        unresolved: List[Tuple[str, int, str, str, str, str]] = []

        # ── Part A: course occurrences that also match a track ────────────────
        for original, pos, deduped in course_occurrences:
            ded_lower = deduped.lower().strip()

            if ded_lower in COURSE_ALIASES and ded_lower in TRACK_ALIASES:
                # A1: explicit dual-alias
                course_canon = COURSE_ALIASES[ded_lower]
                track_canon  = TRACK_ALIASES[ded_lower]

            elif ded_lower in KNOWN_CONFLICT_COURSES:
                # A2: known conflict term — check whether it also matches a track
                track_canon, _ = self._map_track_with_method(deduped)
                if not track_canon:
                    continue
                course_canon = self._has_course_match(deduped)
                if not course_canon:
                    continue

            else:
                continue  # no conflict possible for this occurrence

            # Resolve using LOCAL context around this specific occurrence
            resolution, _, clarification = self._resolve_track_course_conflict(
                original, resolved, course_canon, track_canon, student_track, char_pos=pos
            )
            conflict_debug.append(
                f'"{original}" @{pos}  →  course "{course_canon}"  OR  track "{track_canon}"'
                f"  [{resolution.upper()}]"
            )
            if resolution == "track":
                to_move_to_track.append((original, pos, deduped))
            elif resolution == "ambiguous":
                unresolved.append((original, pos, deduped, course_canon, track_canon, clarification))

        # ── Part B: track occurrences that also match a course ────────────────
        course_occurrence_keys = {(o, p) for o, p, _ in course_occurrences}
        for original, pos, deduped in track_occurrences:
            if (original, pos) in course_occurrence_keys:
                continue  # handled by Part A
            track_canon, _ = self._map_track_with_method(deduped)
            if not track_canon:
                continue
            course_canon = self._has_course_match(deduped) or self._has_course_match(track_canon)
            if not course_canon:
                continue
            # Resolve using LOCAL context around this specific occurrence
            resolution, _, clarification = self._resolve_track_course_conflict(
                original, resolved, course_canon, track_canon, student_track, char_pos=pos
            )
            conflict_debug.append(
                f'"{original}" @{pos}  →  course "{course_canon}"  OR  track "{track_canon}"'
                f"  [{resolution.upper()}]"
            )
            if resolution == "course":
                to_move_to_course.append((original, pos, deduped))
            elif resolution == "ambiguous":
                unresolved.append((original, pos, deduped, course_canon, track_canon, clarification))

        # Apply auto-resolutions (after iteration to avoid mutation-during-loop)
        move_to_track_keys  = {(o, p) for o, p, _ in to_move_to_track}
        move_to_course_keys = {(o, p) for o, p, _ in to_move_to_course}
        course_occurrences = [
            (o, p, d) for o, p, d in course_occurrences
            if (o, p) not in move_to_track_keys
        ]
        new_track_occ = [
            (o, p, d) for o, p, d in track_occurrences
            if (o, p) not in move_to_course_keys
        ]
        for (o, p, d) in to_move_to_track:
            new_track_occ.append((o, p, d))
        track_occurrences = new_track_occ

        # ── Process occurrences resolved as "course" in Part B ───────────────
        # Each was a track term resolved to "course" intent; now find which course.
        # (original, pos, deduped, canonical) for already-resolved ones
        step25_resolved_occ: List[Tuple[str, int, str, str]] = []
        step25_course_resolutions: Dict[str, str] = {}   # orig → canonical
        # unresolved_courses: (orig, pos, deduped, candidates)
        unresolved_courses: List[Tuple[str, int, str, List[Dict]]] = []

        for (orig, pos, ded) in to_move_to_course:
            track_can, _ = self._map_track_with_method(ded)
            candidates   = self._get_course_candidates(ded, track_can)

            if len(candidates) == 1:
                canon = candidates[0]["name"]
                step25_course_resolutions[orig] = canon
                step25_resolved_occ.append((orig, pos, ded, canon))
                conflict_debug.append(
                    f'"{orig}" @{pos}  →  auto-resolved course "{canon}"  [COURSE-AUTO]'
                )
            elif len(candidates) > 1:
                unresolved_courses.append((orig, pos, ded, candidates))
                conflict_debug.append(
                    f'"{orig}" @{pos}  →  {len(candidates)} possible courses  [COURSE-AMBIGUOUS]'
                )
            else:
                # No course found — let Step 3 handle it
                course_occurrences.append((orig, pos, ded))

        if conflict_debug:
            box("📝  STEP 2.5 — Track/Course Conflict", conflict_debug)

        # ── Handle multi-course ambiguity from Part B ─────────────────────────
        if unresolved_courses:
            orig0, pos0, ded0, cands0 = unresolved_courses[0]
            rest_courses = unresolved_courses[1:]
            remaining_occ = list(course_occurrences) + [
                (ro, rp, rd) for ro, rp, rd, _ in rest_courses
            ]
            step25_course_occ = [(o, p, c) for o, p, d, c in step25_resolved_occ]
            pending = PendingAmbiguity(
                original_query   = query,
                dereferenced     = resolved,
                ambiguous_term   = orig0,
                term_position    = pos0,
                candidates       = cands0,
                resolved_courses = step25_course_resolutions,
                pending_courses  = remaining_occ,
                resolved_tracks  = {},
                history          = history,
                resolved_course_occ      = step25_course_occ,
                pending_track_occurrences = list(track_occurrences),
            )
            return PreprocessResult(
                status        = "ambiguous",
                clarification = self._build_clarification(orig0, cands0, resolved, pos0),
                pending       = pending,
            )

        # ── Return ambiguity for first unresolvable track-vs-course conflict ──
        if unresolved:
            orig0, pos0, ded0, course0, track0, clarif0 = unresolved[0]
            rest = unresolved[1:]
            unresolved_occ_keys = {(r[0], r[1]) for r in unresolved}
            remaining_occ = [
                (o, p, d) for o, p, d in course_occurrences
                if (o, p) not in unresolved_occ_keys
            ]
            # Track occurrences that were auto-resolved as TRACK (not in unresolved)
            auto_track_occ = [
                (o, p, d) for o, p, d in track_occurrences
                if (o, p) not in unresolved_occ_keys
            ]
            step25_course_occ = [(o, p, c) for o, p, d, c in step25_resolved_occ]
            pending = PendingAmbiguity(
                original_query   = query,
                dereferenced     = resolved,
                ambiguous_term   = orig0,
                term_position    = pos0,
                candidates       = [],
                resolved_courses = {},
                pending_courses  = remaining_occ,
                resolved_tracks  = {},
                history          = history,
                ambiguity_type   = "course_vs_track",
                course_canonical = course0,
                track_canonical  = track0,
                student_track    = student_track,
                pending_track_course_conflicts = [
                    (r[0], r[1], r[3], r[4]) for r in rest
                ],
                resolved_course_occ      = step25_course_occ,
                pending_track_occurrences = auto_track_occ,
            )
            return PreprocessResult(
                status        = "ambiguous",
                clarification = clarif0,
                pending       = pending,
            )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Step 3 — Course mapping (per-occurrence, independent)
        # resolved_courses: original_term → canonical (for PreprocessResult)
        # resolved_course_occ: (term, pos, canonical) for positional rewriting
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        resolved_courses: Dict[str, str] = dict(step25_course_resolutions)
        # Position-keyed: (term, pos, canonical) — accumulates as we map courses
        resolved_course_occ: List[Tuple[str, int, str]] = [
            (o, p, c) for o, p, d, c in step25_resolved_occ
        ]
        course_debug_lines: List[str] = [
            f'"{orig}"  →  "{canon}"  [from Step 2.5]'
            for orig, canon in step25_course_resolutions.items()
        ]

        for original, pos, deduped in course_occurrences:
            result = self._map_course(deduped)

            if result["status"] == "resolved":
                canonical = result["canonical"]
                method    = result.get("method", "fuzzy match")
                resolved_courses[original] = canonical
                resolved_course_occ.append((original, pos, canonical))
                label = "(same)" if deduped.lower() == canonical.lower() else f"→  {canonical}"
                course_debug_lines.append(
                    f'"{original}" @{pos}'
                    + (f' [deduped: "{deduped}"]' if original != deduped else "")
                    + f"  {label}  [{method}]"
                )

            elif result["status"] == "ambiguous":
                course_debug_lines.append(
                    f'"{original}" @{pos} (deduped: "{deduped}")  →  AMBIGUOUS — '
                    f"{len(result['candidates'])} close matches found"
                )
                for i, c in enumerate(result["candidates"], 1):
                    course_debug_lines.append(
                        f"    {i}. {c['name']} ({c['code']})  "
                        f"confidence={c['confidence']:.2f}"
                    )
                box("📝  STEP 3 — Course Mapping", course_debug_lines)

                # Remaining course occurrences after this one
                cur_idx = next(
                    (i for i, (o, p, d) in enumerate(course_occurrences)
                     if o == original and p == pos),
                    len(course_occurrences) - 1,
                )
                remaining_occ = list(course_occurrences[cur_idx + 1:])

                pending = PendingAmbiguity(
                    original_query   = query,
                    dereferenced     = resolved,
                    ambiguous_term   = original,
                    term_position    = pos,
                    candidates       = result["candidates"],
                    resolved_courses = resolved_courses,
                    pending_courses  = remaining_occ,
                    resolved_tracks  = {},
                    history          = history,
                    resolved_course_occ      = list(resolved_course_occ),
                    pending_track_occurrences = list(track_occurrences),
                )
                return PreprocessResult(
                    status        = "ambiguous",
                    clarification = self._build_clarification(original, result["candidates"], resolved, pos),
                    pending       = pending,
                )
            else:
                course_debug_lines.append(
                    f'"{original}" @{pos}'
                    + (f' [deduped: "{deduped}"]' if original != deduped else "")
                    + "  →  not found"
                )

        if course_occurrences:
            box("📝  STEP 3 — Course Mapping", course_debug_lines)
        else:
            box("📝  STEP 3 — Course Mapping", ["No course terms to map."])

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Step 4 — Track mapping (per-occurrence, independent)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        resolved_tracks: Dict[str, str] = {}
        track_debug_lines: List[str] = []

        for original, pos, deduped in track_occurrences:
            canonical, method = self._map_track_with_method(deduped)
            if canonical:
                resolved_tracks[original] = canonical
                label = "(same)" if deduped.lower() == canonical.lower() else f"→  {canonical}"
                track_debug_lines.append(
                    f'"{original}" @{pos}'
                    + (f' [deduped: "{deduped}"]' if original != deduped else "")
                    + f"  {label}  [{method}]"
                )
            else:
                track_debug_lines.append(
                    f'"{original}" @{pos}'
                    + (f' [deduped: "{deduped}"]' if original != deduped else "")
                    + "  →  not found"
                )

        if track_occurrences:
            box("📝  STEP 4 — Track Mapping", track_debug_lines)
        else:
            box("📝  STEP 4 — Track Mapping", ["No track terms to map."])

        # ── Step 4b: retry failed-track terms as courses ──────────────────
        # When the LLM puts a term in tracks[] but no track name matches it,
        # it may actually be a course (LLM misclassified "special topics",
        # "data visualization", etc.).  Try _map_course() on those terms;
        # if a course match exists, handle it exactly like Step 3 would have.
        step4b_debug: List[str] = []
        step4b_ambiguous: List[Tuple[str, int, str, List[Dict]]] = []

        # Build resolved-track spans for overlap detection below
        resolved_track_spans = [
            (pos, pos + len(orig))
            for orig, pos, _d in track_occurrences
            if orig in resolved_tracks
        ]

        for original, pos, deduped in track_occurrences:
            if original in resolved_tracks:
                continue   # already resolved as track — skip
            # Skip if another track occurrence with overlapping span already mapped
            span_end = pos + len(original)
            if any(
                pos < rend and rstart < span_end
                for rstart, rend in resolved_track_spans
            ):
                continue  # overlapping term already resolved as a track — skip fallback
            course_result = self._map_course(deduped)
            if course_result["status"] == "resolved":
                canonical = course_result["canonical"]
                resolved_courses[original] = canonical
                resolved_course_occ.append((original, pos, canonical))
                step4b_debug.append(
                    f'"{original}" @{pos}  →  course "{canonical}"  [track-fallback]'
                )
            elif course_result["status"] == "ambiguous":
                step4b_ambiguous.append((original, pos, deduped, course_result["candidates"]))
                step4b_debug.append(
                    f'"{original}" @{pos}  →  AMBIGUOUS as course  [track-fallback]'
                )
            else:
                step4b_debug.append(
                    f'"{original}" @{pos}  →  not a course either  [track-fallback]'
                )

        if step4b_debug:
            box("📝  STEP 4b — Track→Course Fallback", step4b_debug)

        # Handle ambiguous track-fallback terms (ask student, chain remaining)
        if step4b_ambiguous:
            orig0, pos0, ded0, cands0 = step4b_ambiguous[0]
            rest_4b = step4b_ambiguous[1:]
            # remaining pending: the other ambiguous fallback occs
            remaining_4b = [(ro, rp, rd) for ro, rp, rd, _ in rest_4b]
            pending = PendingAmbiguity(
                original_query   = query,
                dereferenced     = resolved,
                ambiguous_term   = orig0,
                term_position    = pos0,
                candidates       = cands0,
                resolved_courses = resolved_courses,
                pending_courses  = remaining_4b,
                resolved_tracks  = resolved_tracks,
                history          = history,
                resolved_course_occ      = list(resolved_course_occ),
                pending_track_occurrences = [],  # track occs already exhausted
            )
            return PreprocessResult(
                status        = "ambiguous",
                clarification = self._build_clarification(orig0, cands0, resolved, pos0),
                pending       = pending,
            )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Step 5 — Query rewriting
        # Build suffix-annotated positional replacements from all resolved entities,
        # then apply them right-to-left so earlier positions are not shifted.
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        annotated_replacements: List[Tuple[int, int, str]] = []

        # Course occurrences (Step 2.5 auto + Step 3) → " course" suffix
        for o, p, ded, canon in step25_resolved_occ:
            annotated_replacements.append((p, p + len(o), f"'{canon}' course"))
        for original, pos, deduped in course_occurrences:
            if original in resolved_courses:
                canon = resolved_courses[original]
                annotated_replacements.append((pos, pos + len(original), f"'{canon}' course"))
            elif original != deduped:
                annotated_replacements.append((pos, pos + len(original), deduped))

        # Track occurrences (Step 4) → " program" suffix
        for original, pos, deduped in track_occurrences:
            if original in resolved_tracks:
                canon = resolved_tracks[original]
                annotated_replacements.append((pos, pos + len(original), f"'{canon}' program"))
            elif original != deduped:
                annotated_replacements.append((pos, pos + len(original), deduped))

        if not annotated_replacements:
            if resolved != query:
                box(
                    "📝  STEP 5 — Clean Query (sent to agent)",
                    [f"Original : {query}", f"Clean    : {resolved}  (references resolved)"],
                )
                return PreprocessResult(status="ready", clean_query=resolved)
            box("📝  STEP 5 — Query Rewrite",
                ["Nothing changed — query passed through as-is."])
            return PreprocessResult(status="passthrough", clean_query=query)

        # Apply positional replacements right-to-left
        clean = self._rewrite_query_positional(resolved, annotated_replacements)

        # LLM fallback: if any expected canonical is still missing from the
        # result, the term was not literally found in text (implied / abbreviated).
        dict_replacements: Dict[str, str] = {}
        for k, v in resolved_courses.items():
            dict_replacements[k] = f"'{v}' course"
        for o, p, _d, c in step25_resolved_occ:
            dict_replacements[o] = f"'{c}' course"
        for k, v in resolved_tracks.items():
            if k not in dict_replacements:
                dict_replacements[k] = f"'{v}' program"

        still_unreplaced = any(
            replacement.lower() not in clean.lower()
            for orig, replacement in dict_replacements.items()
            if orig.lower() != replacement.lower()
        )
        if still_unreplaced:
            clean = self._rewrite_query(resolved, dict_replacements)

        box(
            "📝  STEP 5 — Clean Query (sent to agent)",
            [f"Original : {query}", f"Clean    : {clean}"],
        )

        return PreprocessResult(status="ready", clean_query=clean, resolved_courses=resolved_courses)

    def resolve_ambiguity(
        self,
        pending: PendingAmbiguity,
        student_reply: str,
    ) -> PreprocessResult:
        """
        Called when the student answers a disambiguation question.

        Dispatches to the appropriate handler based on ambiguity_type:
          "course_name"    — student picks from multiple close-matching courses
          "course_vs_track" — student picks whether the term is a course or track
        """
        if pending.ambiguity_type == "course_vs_track":
            return self._resolve_course_vs_track(pending, student_reply)

        # ── "course_name" handler (original behaviour) ────────────────────
        # 1. Parse the student's reply to pick the chosen course.
        # 2. Merge with already-resolved courses in pending.
        # 3. Re-run track mapping on the original query.
        # 4. Rewrite the original query with all resolved names.
        # 5. Show debug boxes for the resolution.
        chosen = self._pick_from_reply(student_reply, pending.candidates)

        # Student chose to keep original term unchanged
        is_keep_original = (chosen == self._KEEP_ORIGINAL)
        if is_keep_original:
            chosen = pending.ambiguous_term

        box(
            "📝  AMBIGUITY RESOLVED",
            [
                f'Student replied : "{student_reply}"',
                f'Chosen course   : "{chosen}"',
            ],
        )

        # Merge chosen course (key = ORIGINAL ambiguous term)
        all_courses = dict(pending.resolved_courses)
        all_courses[pending.ambiguous_term] = chosen

        # Build resolved_course_occ incrementally so every chained PendingAmbiguity
        # receives the full position-keyed history (critical for correct post-chain
        # track mapping: prevents a TRACK-resolved occurrence of the same surface
        # text from being skipped as if it were a course).
        chain_course_occ: List[Tuple[str, int, str]] = list(pending.resolved_course_occ)
        # When student chose "keep as is" (option N+1), don't add a positional
        # replacement — the original term stays verbatim without a 'course' suffix.
        if pending.term_position >= 0 and not is_keep_original:
            chain_course_occ.append(
                (pending.ambiguous_term, pending.term_position, chosen)
            )

        # Process remaining course occurrences that were skipped when this ambiguity was hit.
        # pending_courses is now List[Tuple[str, int, str]] = [(original, char_pos, deduped), ...]
        for idx, (rem_orig, rem_pos, rem_deduped) in enumerate(pending.pending_courses):
            rem_result = self._map_course(rem_deduped)
            if rem_result["status"] == "resolved":
                all_courses[rem_orig] = rem_result["canonical"]
                chain_course_occ.append((rem_orig, rem_pos, rem_result["canonical"]))
            elif rem_result["status"] == "ambiguous":
                # Chain: another occurrence is ambiguous — ask the student again
                next_rem = list(pending.pending_courses[idx + 1:])
                new_pending = PendingAmbiguity(
                    original_query   = pending.original_query,
                    dereferenced     = pending.dereferenced,
                    ambiguous_term   = rem_orig,
                    term_position    = rem_pos,
                    candidates       = rem_result["candidates"],
                    resolved_courses = all_courses,
                    pending_courses  = next_rem,
                    resolved_tracks  = {},
                    history          = pending.history,
                    resolved_course_occ       = list(chain_course_occ),
                    pending_track_occurrences  = list(pending.pending_track_occurrences),
                )
                clarification = self._build_clarification(
                    rem_orig, rem_result["candidates"], pending.dereferenced, rem_pos
                )
                return PreprocessResult(
                    status        = "ambiguous",
                    clarification = clarification,
                    pending       = new_pending,
                )
            # else "not_found": leave it for the agent to handle

        # chain_pending_track_occ mirrors chain_course_occ for the track side.
        # Starts from the parent's pending_track_occurrences; if any
        # pending_track_course_conflicts auto-resolve as TRACK below, those
        # occurrences are appended so downstream rewriters can map them.
        chain_pending_track_occ: List[Tuple[str, int, str]] = list(pending.pending_track_occurrences)

        # Process any pending_track_course_conflicts forwarded from a previous
        # course_vs_track session (e.g. when student picked "course" for one
        # conflict causing a course_name chain, but more conflicts remain).
        extra_tracks: Dict[str, str] = {}
        for rem_term, rem_pos_c, rem_course, rem_track in pending.pending_track_course_conflicts:
            resolution, _, clarification = self._resolve_track_course_conflict(
                rem_term, pending.original_query, rem_course, rem_track, pending.student_track,
                char_pos=rem_pos_c if rem_pos_c >= 0 else None,
            )
            if resolution == "course":
                all_courses[rem_term] = rem_course
                # Track the position so the rewriter includes it
                if rem_pos_c >= 0 and not any(o == rem_term and p == rem_pos_c for o, p, _ in chain_course_occ):
                    chain_course_occ.append((rem_term, rem_pos_c, rem_course))
            elif resolution == "track":
                extra_tracks[rem_term] = rem_track
                # Add this occurrence to chain_pending_track_occ if not already there
                deduped_rc = self._dedupe_chars(rem_term)
                if (rem_term, rem_pos_c) not in {(o, p) for o, p, _ in chain_pending_track_occ}:
                    chain_pending_track_occ.append((rem_term, rem_pos_c, deduped_rc))
            else:
                remaining = [
                    item for item in pending.pending_track_course_conflicts
                    if item[0] != rem_term or item[1] != rem_pos_c
                ]
                new_pending = PendingAmbiguity(
                    original_query   = pending.original_query,
                    dereferenced     = pending.dereferenced,
                    ambiguous_term   = rem_term,
                    term_position    = rem_pos_c,
                    candidates       = [],
                    resolved_courses = all_courses,
                    pending_courses  = [],
                    resolved_tracks  = {**pending.resolved_tracks, **extra_tracks},
                    history          = pending.history,
                    ambiguity_type   = "course_vs_track",
                    course_canonical = rem_course,
                    track_canonical  = rem_track,
                    student_track    = pending.student_track,
                    pending_track_course_conflicts = remaining,
                    resolved_course_occ       = list(chain_course_occ),
                    pending_track_occurrences  = list(chain_pending_track_occ),
                )
                return PreprocessResult(
                    status        = "ambiguous",
                    clarification = clarification,
                    pending       = new_pending,
                )

        # chain_course_occ already contains every course-resolved occurrence:
        #   pending.resolved_course_occ  (chain history)
        #   + current term (added at the top of this function)
        #   + courses from pending_courses loop above
        #   + courses from pending_track_course_conflicts auto-resolved as course
        # Using it directly ensures pending_courses resolutions end up in the
        # positional replacement list — fixing the bug where "data visualization"
        # (or any course resolved in the chaining loop) was skipped from rewriting.
        all_course_occ: List[Tuple[str, int, str]] = chain_course_occ
        course_resolved_pos = {(o, p) for o, p, _ in all_course_occ}

        # Map remaining track occurrences using pending.pending_track_occurrences
        # (position-aware so we skip only the specific occurrence resolved as course).
        # Fall back to _extract_entities if pending_track_occurrences is not populated
        # (e.g. for PendingAmbiguity objects created before this field was added).
        if pending.pending_track_occurrences:
            raw_track_occ = pending.pending_track_occurrences
        else:
            _, raw_tracks_fallback = self._extract_entities(pending.original_query)
            raw_track_occ = [(t, p, self._dedupe_chars(t)) for t, p in raw_tracks_fallback]

        resolved_track_occ: List[Tuple[str, int, str]] = []  # (term, pos, canonical)
        # Carry forward any extra track resolutions from pending_track_course_conflicts
        for term, canon in extra_tracks.items():
            # Find position(s) in the query — use first found
            for occ_t, occ_p, _ in raw_track_occ:
                if occ_t == term:
                    resolved_track_occ.append((term, occ_p, canon))
                    break
            else:
                resolved_track_occ.append((term, -1, canon))  # position unknown

        track_debug: List[str] = []
        # Track-to-course fallback: collect failed track terms to try as courses
        track_fallback_ambiguous_pa: List[Tuple[str, int, str, List[Dict]]] = []
        for orig, pos, ded in raw_track_occ:
            if (orig, pos) in course_resolved_pos:
                track_debug.append(f'"{orig}" @{pos}  →  skipped (resolved as course)')
                continue
            canonical, method = self._map_track_with_method(ded)
            if canonical:
                resolved_track_occ.append((orig, pos, canonical))
                track_debug.append(f'"{orig}" @{pos}  →  {canonical}  [{method}]')
            else:
                # Track mapping failed — try as a course (LLM misclassification)
                cr = self._map_course(ded)
                if cr["status"] == "resolved":
                    all_courses[orig] = cr["canonical"]
                    all_course_occ.append((orig, pos, cr["canonical"]))
                    course_resolved_pos = {(o2, p2) for o2, p2, _ in all_course_occ}
                    track_debug.append(
                        f'"{orig}" @{pos}  →  course "{cr["canonical"]}"  [track-fallback]'
                    )
                elif cr["status"] == "ambiguous":
                    track_fallback_ambiguous_pa.append((orig, pos, ded, cr["candidates"]))
                    track_debug.append(f'"{orig}" @{pos}  →  AMBIGUOUS as course  [track-fallback]')
                else:
                    track_debug.append(f'"{orig}" @{pos}  →  not found')

        if raw_track_occ or extra_tracks:
            box("📝  TRACK MAPPING (post-disambiguation)", track_debug)

        # Handle ambiguous track-fallback terms — ask the student, chain remainder
        if track_fallback_ambiguous_pa:
            orig0, pos0, ded0, cands0 = track_fallback_ambiguous_pa[0]
            rest_fb = track_fallback_ambiguous_pa[1:]
            remaining_fb = [(ro, rp, rd) for ro, rp, rd, _ in rest_fb]
            new_pending = PendingAmbiguity(
                original_query   = pending.original_query,
                dereferenced     = pending.dereferenced,
                ambiguous_term   = orig0,
                term_position    = pos0,
                candidates       = cands0,
                resolved_courses = all_courses,
                pending_courses  = remaining_fb,
                resolved_tracks  = {**pending.resolved_tracks, **extra_tracks,
                                     **{o2: c2 for o2, p2, c2 in resolved_track_occ}},
                history          = pending.history,
                resolved_course_occ      = list(all_course_occ),
                pending_track_occurrences = [],
            )
            return PreprocessResult(
                status        = "ambiguous",
                clarification = self._build_clarification(
                    orig0, cands0, pending.dereferenced, pos0
                ),
                pending       = new_pending,
            )

        # Build positional replacements — each occurrence replaced independently.
        pos_replacements: List[Tuple[int, int, str]] = []
        for o, p, canon in all_course_occ:
            if p >= 0:
                pos_replacements.append((p, p + len(o), f"'{canon}' course"))
        for o, p, canon in resolved_track_occ:
            if p >= 0:
                pos_replacements.append((p, p + len(o), f"'{canon}' program"))
        # Deduped fallback for unresolved terms
        for o, p, ded in raw_track_occ:
            if (o, p) not in course_resolved_pos and o != ded and p >= 0:
                if not any(r[0] == p for r in pos_replacements):  # not already replaced
                    pos_replacements.append((p, p + len(o), ded))

        if pos_replacements:
            clean = self._rewrite_query_positional(pending.original_query, pos_replacements)
            # LLM fallback if any expected canonical is still missing
            dict_rep: Dict[str, str] = {}
            for o, p, canon in all_course_occ:
                dict_rep[o] = f"'{canon}' course"
            for o, p, canon in resolved_track_occ:
                if o not in dict_rep:
                    dict_rep[o] = f"'{canon}' program"
            still_unreplaced = any(
                v.lower() not in clean.lower()
                for k, v in dict_rep.items()
                if k.lower() != v.lower()
            )
            if still_unreplaced:
                clean = self._rewrite_query(pending.original_query, dict_rep)
        else:
            clean = pending.original_query

        box(
            "📝  CLEAN QUERY (sent to agent)",
            [
                f"Original : {pending.original_query}",
                f"Clean    : {clean}",
            ],
        )

        return PreprocessResult(status="ready", clean_query=clean, resolved_courses=all_courses)

    # ── Step 1: Reference resolution ─────────────────────────────────────

    def _resolve_references(self, query: str, history: List[Dict]) -> str:
        """
        Replace pronouns / vague references with their actual referents.

        Only called when the query contains a reference word AND history
        exists.  Returns the original query unchanged if nothing to resolve.
        """
        # Words that signal a vague reference needing resolution.
        # Uses whole-word matching to avoid false positives
        # (e.g. "that" in "I want that course" vs "data" which contains "tha").
        REFERENCE_WORDS = {
            "it", "its", "this", "that", "these", "those",
            "them", "they",                              # ← was missing
            "the course", "the subject", "the track", "the program",
            "same course", "that course", "this course",
        }
        query_lower = query.lower()

        # Use word-boundary matching to avoid false positives:
        # "data" should NOT match "that", "it" should NOT match "bit", etc.
        import re as _re_ref
        has_ref = any(
            _re_ref.search(rf"\b{_re_ref.escape(word)}\b", query_lower)
            for word in REFERENCE_WORDS
        )

        if not has_ref or not history:
            return query

        # Build a compact history block (last 4 turns max)
        history_text = "\n".join(
            f"{'Student' if m['role'] == 'user' else 'Advisor'}: {m['content']}"
            for m in history[-4:]
        )

        prompt = f"""You are resolving references in a student's university academic question.

Conversation so far:
{history_text}

Student's new question: "{query}"

Task: Replace any pronoun or vague reference with the MOST SPECIFIC subject
it contextually refers to, based on the full conversation above.

References to resolve: "it", "its", "this", "that", "them", "they",
"these", "those", "the course", "the subject", "the track", "the program"

IMPORTANT — Always use the MOST SPECIFIC contextual meaning, not the
grammatically minimal one. Students are asking follow-up questions about
the topic of the conversation, so:
  - If the conversation was about "electives at data science", then
    "it" / "them" → "electives at data science"  (NOT just "electives")
  - If the conversation was about "machine learning prerequisites", then
    "it" → "machine learning"  (NOT just "course")
  - Include the program/track if it was mentioned alongside the subject.

The reference can point to ANYTHING from the conversation:
  - A course with its context  (e.g. "it" → "machine learning in the AI track")
  - A program/track            (e.g. "it" → "data science track")
  - A concept with context     (e.g. "them" → "electives in data science")
  - Multiple specific items    (e.g. "them" → "machine learning and data structures")

Rules:
  1. Scan the FULL conversation to find what the reference contextually points to.
  2. Prefer the MOST SPECIFIC subject — include any qualifiers (program, year, etc.)
     that were mentioned alongside the subject in the conversation.
  3. If nothing needs resolving, return the question EXACTLY as-is.
  4. Do NOT add information that was never in the conversation.
  5. Return ONLY the resolved question — nothing else.

Examples:
  Conversation: Student asked "what electives at das?" / Advisor answered about DAS electives
  Student asks: "when can i take it?"
  Resolved:     "when can i take electives at data science?"   ← includes "data science"

  Conversation: Advisor said "you can take electives in year 3 in the AI program"
  Student asks: "when can i take them?"
  Resolved:     "when can i take electives in the AI program?"  ← includes "AI program"

Resolved question:"""

        try:
            resolved = llm_call_text(
                system=(
                    "You resolve pronouns and vague references in student questions. "
                    "Return only the rewritten question as plain text. No explanation."
                ),
                user=prompt,
                temperature=0,
                max_tokens=200,
            ).strip().strip('"')
            # Sanity check
            if not resolved or len(resolved) > len(query) * 5:
                return query
            return resolved
        except Exception:
            return query


    # ── Step 2: Entity extraction ─────────────────────────────────────────

    @staticmethod
    def _dedupe_chars(term: str) -> str:
        """
        Collapse runs of 3 or more consecutive identical characters.

        Threshold = 3 so that legitimate English double-letters are
        preserved:
            "mmml"           → "ml"    (3 m's = keyboard hold)
            "arrrtttiiii"    → "arti"  (3+ each = keyboard hold)
            "daaata"         → "data"  (3 a's = keyboard hold)
            "Differential"   → "Differential"  (double-l = real word, kept)
            "processing"     → "processing"    (double-s = real word, kept)
            "ssoftware"      → "software"      (3 s's? no — 2 s's kept as-is)

        Wait — "ssoftware" has only 2 s's, so it is kept unchanged.
        That's fine: "ss" at the start of a word is unusual enough that
        fuzzy matching will still find "software" with high confidence.

        Algorithm: scan with a run-length counter; emit the char once
        for any run, but only if the run length >= 3.  Runs of 1 or 2
        are emitted as-is.
        """
        if not term:
            return term

        import re as _re

        def _collapse(m):
            ch  = m.group(0)[0]
            run = len(m.group(0))
            # Collapse only if 3 or more consecutive identical chars
            return ch if run >= 3 else m.group(0)

        # Match any run of 2+ of the same char (case-sensitive)
        return _re.sub(r"(.)\1+", _collapse, term)

    def _extract_entities(
        self, query: str
    ) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        """
        Extract course mentions and track/program mentions from the query.

        Returns: (course_occurrences, track_occurrences)
            Each is a list of (term, char_position) tuples — one entry per
            occurrence of the term in the query text.  The same surface text
            (e.g. "ai") may appear multiple times and will produce multiple
            independent entries.

        Uses the LLM for term identification, then finds ALL char positions
        of each extracted term in the query via substring search.
        """
        prompt = f"""Extract course names and program/track names from this student question.

Question: "{query}"

Return ONLY this JSON (no explanation, no markdown):
{{
  "courses": ["term1", "term2"],
  "tracks": ["term1"]
}}

Rules:
- courses: every course name, code, or abbreviation (e.g. "ml", "BCS311", "machine learning", "soft eng", "probability")
- tracks: every program/track name or abbreviation — these are the ONLY valid track abbreviations:
    aim, ai, aiml → artificial intelligence & machine learning
    sad, software, sw, app dev → software & application development
    das, ds, data science, data → data science
  If the question contains any of these, put them in tracks[], NOT courses[].
- Use the EXACT wording from the question.
- Use [] if nothing found for a category.

Examples:
  "what electives at das?" → {{"courses": [], "tracks": ["das"]}}
  "what electives at ai?" → {{"courses": [], "tracks": ["ai"]}}
  "can i take ml in sad?" → {{"courses": ["ml"], "tracks": ["sad"]}}
  "prereqs for BCS311?" → {{"courses": ["BCS311"], "tracks": []}}"""

        try:
            raw = llm_call_json(prompt, temperature=0, max_tokens=200)
            # Strip markdown fences if the model wraps with ```json
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(raw)
            courses = [str(c).strip() for c in data.get("courses", []) if c]
            tracks  = [str(t).strip() for t in data.get("tracks",  []) if t]

            # Post-process: move course terms that are track aliases to tracks,
            # BUT only when they are NOT also in COURSE_ALIASES.
            # Dual-alias terms (e.g. "ai": in both dicts) STAY in courses —
            # Step 2.5 will resolve the track/course conflict via intent signals.
            reclassified = [
                c for c in courses
                if c.lower().strip() in TRACK_ALIASES
                and c.lower().strip() not in COURSE_ALIASES
            ]
            courses = [
                c for c in courses
                if not (c.lower().strip() in TRACK_ALIASES
                        and c.lower().strip() not in COURSE_ALIASES)
            ]
            tracks = tracks + reclassified

            # Remove from tracks any term that is already in courses.
            # The LLM sometimes puts ambiguous terms (e.g. "ai", "soft") in
            # both lists; Part A of Step 2.5 handles course-list terms, so
            # having them also in tracks causes duplicate conflict detection.
            courses_lower = {c.lower().strip() for c in courses}
            tracks = [t for t in tracks if t.lower().strip() not in courses_lower]

            # ── Post-process extracted course terms ───────────────────────────
            # 1. Strip trailing blocklist words: "ai courses" → "ai"
            # 2. Split "X in/at/for Y" where Y is a track alias:
            #    "special topics in ai" → course "special topics" + track "ai"
            processed_courses: List[str] = []
            extra_tracks_from_split: List[str] = []
            for term in courses:
                # Step 1: strip trailing blocklist words
                stripped = self._strip_trailing_blocklist(term)
                if not stripped:
                    continue
                # Step 2: split "X in/at/for track_alias"
                split = self._split_course_track_compound(stripped)
                if split:
                    course_part, track_part = split
                    if course_part:
                        processed_courses.append(course_part)
                    if track_part and track_part.lower().strip() not in courses_lower:
                        extra_tracks_from_split.append(track_part)
                else:
                    processed_courses.append(stripped)
            courses = processed_courses
            tracks  = tracks + extra_tracks_from_split

            # Find ALL char positions of each unique term in the query.
            # This is the key change: the same term appearing twice produces
            # two independent (term, position) entries.
            course_occurrences = self._find_all_char_occurrences(query, courses)
            track_occurrences  = self._find_all_char_occurrences(query, tracks)

            return course_occurrences, track_occurrences
        except Exception as exc:
            logger.debug("Entity extraction failed: %s", exc)
            return [], []

    @staticmethod
    def _find_all_char_occurrences(
        text: str, terms: List[str]
    ) -> List[Tuple[str, int]]:
        """
        For each term, find ALL character positions where it appears in text
        as a whole word (case-insensitive, word-boundary aware).

        "ai" will NOT match inside "explain", "said", "fails" etc. — the char
        immediately before and after the match must not be alphanumeric or '_'.

        Returns a flat list of (term, start_char_pos) sorted by position ascending.
        Each unique occurrence is returned independently — if "ai" appears at
        positions 8 and 23, the result contains both entries.
        """
        result: List[Tuple[str, int]] = []
        text_lower = text.lower()
        for term in terms:
            t_lower = term.lower()
            t_len   = len(t_lower)
            start   = 0
            while True:
                pos = text_lower.find(t_lower, start)
                if pos == -1:
                    break
                end = pos + t_len
                # Word-boundary check: char before and after must not be
                # alphanumeric (or underscore), otherwise "ai" would match
                # inside "explain", "said", "training", etc.
                before_ok = (pos == 0) or not (text_lower[pos - 1].isalnum() or text_lower[pos - 1] == "_")
                after_ok  = (end >= len(text_lower)) or not (text_lower[end].isalnum() or text_lower[end] == "_")
                if before_ok and after_ok:
                    result.append((term, pos))
                start = pos + t_len  # non-overlapping occurrences
        result.sort(key=lambda x: x[1])
        return result

    # ── Step 3: Course mapping ────────────────────────────────────────────

    def _map_course(self, term: str) -> Dict:
        """
        Map one course term to its canonical Neo4j name.

        Returns:
            {"status": "resolved",  "canonical": "machine learning"}
            {"status": "ambiguous", "candidates": [{name, code, confidence}, ...]}
            {"status": "not_found"}
        """
        term_lower = term.lower().strip()

        # ── a) Check aliases first ────────────────────────────────────────
        if term_lower in COURSE_ALIASES:
            return {"status": "resolved", "canonical": COURSE_ALIASES[term_lower], "method": "alias"}

        # ── b) Check pre-defined conflict candidates ──────────────────────
        # Faster than a DB fuzzy search AND ensures the complete relevant list
        # is shown for known ambiguous terms (e.g. "soft" → all software courses).
        if term_lower in KNOWN_CONFLICT_COURSES:
            candidates = KNOWN_CONFLICT_COURSES[term_lower]
            # Single entry OR exact name match → one clear winner, no disambiguation.
            if len(candidates) == 1 or candidates[0]["name"].lower() == term_lower:
                return {
                    "status":    "resolved",
                    "canonical": candidates[0]["name"],
                    "method":    "known_conflict",
                }
            return {"status": "ambiguous", "candidates": candidates}

        # ── c) Fuzzy match against live course list ───────────────────────
        courses = _load_courses()
        if not courses:
            return {"status": "not_found"}

        scored = []
        for c in courses:
            s = _score_course(term, c["name"], c["code"])
            if s >= self.MATCH_THRESHOLD:
                scored.append({
                    "name":       c["name"],
                    "code":       c["code"],
                    "confidence": round(s, 3),
                })
        scored.sort(key=lambda x: x["confidence"], reverse=True)

        if not scored:
            return {"status": "not_found"}

        best = scored[0]

        # ── d) Exact or single clear winner ──────────────────────────────
        if best["confidence"] >= 0.95:
            return {"status": "resolved", "canonical": best["name"], "method": f"fuzzy match (confidence={best['confidence']:.2f})"}

        # ── e) Check for ambiguity (two close scores) ─────────────────────
        if len(scored) >= 2:
            second = scored[1]
            delta  = best["confidence"] - second["confidence"]
            if delta <= self.AMBIGUITY_DELTA:
                # Too close — need the student to pick
                candidates = [
                    m for m in scored[:self.MAX_CANDIDATES]
                    if best["confidence"] - m["confidence"] <= self.AMBIGUITY_DELTA
                ]
                # Always include the top match even if it's just above the delta
                if best not in candidates:
                    candidates.insert(0, best)
                return {"status": "ambiguous", "candidates": candidates}

        # ── f) Single winner with reasonable confidence ───────────────────
        return {"status": "resolved", "canonical": best["name"], "method": f"fuzzy match (confidence={best['confidence']:.2f})"}

    # ── Step 4: Track mapping ─────────────────────────────────────────────

    def _map_track(self, term: str) -> Optional[str]:
        """
        Map one track/program term to its canonical Neo4j program name.
        Returns the canonical name, or None if no confident match found.
        Tracks are less ambiguous so we always auto-pick.
        """
        term_lower = term.lower().strip()

        # a) Check aliases
        if term_lower in TRACK_ALIASES:
            return TRACK_ALIASES[term_lower]

        # b) Check each track alias value for substring / similarity
        tracks = _load_tracks()
        if not tracks:
            return None

        scored = []
        for t in tracks:
            s = max(
                _similarity(term, t),
                _prefix_score(term, t),
                _keyword_overlap(term, t),
            )
            if s >= self.MATCH_THRESHOLD:
                scored.append((t, s))
        scored.sort(key=lambda x: x[1], reverse=True)

        if scored and scored[0][1] >= 0.5:
            return scored[0][0]
        return None

    def _map_track_with_method(self, term: str):
        """
        Map a (possibly deduped) track term to its canonical Neo4j program name.
        Returns (canonical_name, method_string) or (None, None).

        Scoring tries four strategies and takes the best:
          1. Exact alias table lookup  → instant, highest priority
          2. Full-string similarity against the whole track name
          3. Prefix score (term is a prefix of the track name)
          4. Word-level similarity: compare term against each INDIVIDUAL WORD
             of the track name — this is what catches "artifcial" → "artificial
             intelligence and machine learning" because:
             _similarity("artifcial", "artificial") ≈ 0.88  (one char diff)
        """
        term_lower = term.lower().strip()

        # 1. Alias table
        if term_lower in TRACK_ALIASES:
            return TRACK_ALIASES[term_lower], "alias"

        tracks = _load_tracks()
        if not tracks:
            return None, None

        scored = []
        for t in tracks:
            # Full-string scores
            full_sim   = _similarity(term_lower, t)
            prefix     = _prefix_score(term_lower, t)
            kw_overlap = _keyword_overlap(term_lower, t)

            # Word-level score: best similarity between term and any word in track name
            word_sim = max(
                (_similarity(term_lower, word) for word in t.split()),
                default=0.0,
            )

            best = max(full_sim, prefix, kw_overlap, word_sim)
            if best >= self.MATCH_THRESHOLD:
                scored.append((t, best, word_sim))

        scored.sort(key=lambda x: x[1], reverse=True)

        if scored and scored[0][1] >= 0.5:
            t, score, word_s = scored[0]
            method = (
                f"word-level fuzzy (confidence={word_s:.2f})"
                if word_s == score
                else f"fuzzy match (confidence={score:.2f})"
            )
            return t, method

        return None, None

    # ── Track/Course conflict resolution helpers ──────────────────────────

    def _has_course_match(self, term: str) -> Optional[str]:
        """
        Return the best course canonical name if `term` could refer to a course.
        Returns the top candidate even when the match is ambiguous (multiple
        close courses).  Returns None if no course matches above threshold.
        """
        term_lower = term.lower().strip()
        if term_lower in COURSE_ALIASES:
            return COURSE_ALIASES[term_lower]
        result = self._map_course(term)
        if result["status"] == "resolved":
            return result["canonical"]
        if result["status"] == "ambiguous":
            return result["candidates"][0]["name"]
        return None

    def _get_course_candidates(
        self,
        deduped:     str,
        track_canon: Optional[str] = None,
    ) -> List[Dict]:
        """
        Return the full list of course candidates for a term that Step 2.5 has
        decided is a course reference.  Checked in order:
          1. COURSE_ALIASES            → single winner
          2. KNOWN_CONFLICT_COURSES    → all listed candidates
          3. fuzzy _map_course(deduped)
          4. fuzzy _map_course(track_canon)  (when deduped itself matched nothing)
        Returns [] when no course match exists.
        """
        ded_lower = deduped.lower().strip()

        if ded_lower in COURSE_ALIASES:
            return [{"name": COURSE_ALIASES[ded_lower], "code": "", "confidence": 1.0}]

        if ded_lower in KNOWN_CONFLICT_COURSES:
            return list(KNOWN_CONFLICT_COURSES[ded_lower])

        result = self._map_course(deduped)
        if result["status"] == "resolved":
            return [{"name": result["canonical"], "code": result.get("code", ""), "confidence": 1.0}]
        if result["status"] == "ambiguous":
            return result["candidates"]

        # deduped itself matched nothing — try the resolved track name
        if track_canon and track_canon.lower().strip() != ded_lower:
            tc_lower = track_canon.lower().strip()
            if tc_lower in COURSE_ALIASES:
                return [{"name": COURSE_ALIASES[tc_lower], "code": "", "confidence": 1.0}]
            if tc_lower in KNOWN_CONFLICT_COURSES:
                return list(KNOWN_CONFLICT_COURSES[tc_lower])
            # Check if any individual word of the track name has a KNOWN_CONFLICT_COURSES
            # entry (e.g. "software & application development" → word "software" → 10 courses).
            # This catches track abbreviations like "sad"/"das" that map to a track whose
            # name contains a known ambiguous course keyword.
            for word in tc_lower.split():
                if word in KNOWN_CONFLICT_COURSES:
                    return list(KNOWN_CONFLICT_COURSES[word])
            # Last resort: fuzzy match against the track's canonical name.
            # We deliberately do NOT return a single fuzzy "resolved" hit as an
            # auto-resolved winner here, because a track abbreviation typed as a
            # course query warrants confirmation even when one fuzzy result exists.
            result2 = self._map_course(track_canon)
            if result2["status"] == "ambiguous":
                return result2["candidates"]

        return []

    def _detect_intent_signal(
        self,
        term:     str,
        query:    str,
        char_pos: Optional[int] = None,
    ) -> Tuple[str, float]:
        """
        Detect whether `term` is used as a course or track reference in `query`.

        When `char_pos` is given, only the ±6-word window around the occurrence
        at that character position is scored — enabling per-occurrence resolution
        when the same term appears multiple times with different meanings.

        When `char_pos` is None (default), the best score across ALL occurrences
        is returned (original behaviour).

        Returns:
            ("course", confidence)  |  ("track", confidence)  |  ("unknown", 0.0)
        """
        q_lower = query.lower()
        t_lower = term.lower()
        q_words = q_lower.split()
        t_words = t_lower.split()

        WINDOW = 6

        # Find ALL word-level positions of the term
        all_word_positions: List[int] = []
        for i in range(len(q_words) - len(t_words) + 1):
            if q_words[i: i + len(t_words)] == t_words:
                all_word_positions.append(i)

        NARROW_WINDOW = 3   # words each side for intent scoring (per-occurrence mode)

        def _score_at_word_pos(word_pos: int, narrow: bool = False) -> Tuple[float, float]:
            """
            Score the context around a single word position.

            When `narrow=True` (per-occurrence mode):
              - Structural checks (preposition / container) examine only the
                IMMEDIATELY ADJACENT word so they cannot match a different
                occurrence of the same term nearby.
              - Intent scoring uses a ±NARROW_WINDOW context.

            When `narrow=False` (global mode):
              - Structural checks run on the full ±WINDOW context.
              - Intent scoring uses the full ±WINDOW context.
            """
            if narrow:
                # --- Structural: check only the one word directly before/after ---
                before = q_words[word_pos - 1] if word_pos > 0 else ""
                after_idx = word_pos + len(t_words)
                after = q_words[after_idx] if after_idx < len(q_words) else ""

                if before in {"of", "in", "at", "from", "for"}:
                    return 0.0, 0.92   # preposition directly before → strong track
                if after in {"courses", "electives", "curriculum", "subjects", "classes"}:
                    return 0.0, 0.92   # plural/collective container word directly after → strong track
                if after in {"course", "elective", "subject", "class"}:
                    return 0.92, 0.0   # singular "X course" pattern → "the AI course" → strong course

                # Intent signal: narrow window only
                start = max(0, word_pos - NARROW_WINDOW)
                end   = min(len(q_words), word_pos + len(t_words) + NARROW_WINDOW)
            else:
                # --- Structural: regex on the ±WINDOW context ---
                start = max(0, word_pos - WINDOW)
                end   = min(len(q_words), word_pos + len(t_words) + WINDOW)
                ctx   = " ".join(q_words[start:end])
                prep_re = r'\b(?:of|in|at|from|for)\s+' + re.escape(t_lower) + r'\b'
                if re.search(prep_re, ctx):
                    return 0.0, 0.92
                cont_re = re.escape(t_lower) + r'\s+(?:courses|electives?|curriculum|subjects|classes)'
                if re.search(cont_re, ctx):
                    return 0.0, 0.92
                sing_re = re.escape(t_lower) + r'\s+(?:course|subject|class)\b'
                if re.search(sing_re, ctx):
                    return 0.92, 0.0

            ctx = " ".join(q_words[start:end])
            c, t = _score_intent_context(ctx)
            return c, t

        if char_pos is not None and all_word_positions:
            # ── Position-specific scoring (narrow, occurrence-isolated) ───────
            # Convert char_pos to a word index:
            # len(query[:char_pos].split()) = number of complete words before char_pos
            # = 0-based index of the word that starts at char_pos.
            target_word_idx = len(query[:char_pos].split())
            best_word_pos = min(
                all_word_positions, key=lambda p: abs(p - target_word_idx)
            )
            c, t = _score_at_word_pos(best_word_pos, narrow=True)
            if t > 0 and t >= c:
                return "track", t
            if c > 0 and c > t:
                return "course", c
            return "unknown", 0.0

        # ── Global scoring (no char_pos): structural checks first, then best ──
        preposition_re = r'\b(?:of|in|at|from|for)\s+' + re.escape(t_lower) + r'\b'
        if re.search(preposition_re, q_lower):
            return "track", 0.92

        container_re = re.escape(t_lower) + r'\s+(?:courses?|electives?|curriculum|subjects?|classes?)'
        if re.search(container_re, q_lower):
            return "track", 0.92

        if all_word_positions:
            best_c, best_t = 0.0, 0.0
            for wpos in all_word_positions:
                c, t   = _score_at_word_pos(wpos, narrow=False)
                best_c = max(best_c, c)
                best_t = max(best_t, t)
        else:
            best_c, best_t = _score_intent_context(q_lower)

        if best_c > best_t and best_c > 0:
            return "course", best_c
        if best_t > best_c and best_t > 0:
            return "track", best_t
        return "unknown", 0.0

    def _resolve_track_course_conflict(
        self,
        term:             str,
        query:            str,
        course_canonical: str,
        track_canonical:  str,
        student_track:    Optional[str],
        char_pos:         Optional[int] = None,
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Decide whether `term` refers to a course or a track in `query`.

        Resolution layers (applied in order):
          1. Intent signal from the ±6-word context window around `char_pos`
             (when provided) or the best score across all occurrences (default).
          2. Enrolled-track heuristic (confidence boost when signal exists)
          3. Fallback → ask the student

        Returns one of:
          ("course",    course_canonical, None)
          ("track",     track_canonical,  None)
          ("ambiguous", None,             clarification_question)
        """
        is_exact = term.lower().strip() in EXACT_COLLISION_TERMS
        threshold = EXACT_COLLISION_THRESHOLD if is_exact else INTENT_SIGNAL_THRESHOLD

        intent, confidence = self._detect_intent_signal(term, query, char_pos=char_pos)

        if intent != "unknown":
            # Enrolled-track heuristic: boost confidence when the signal aligns
            # with whether the term matches the student's own program.
            if student_track and track_canonical:
                same_track = track_canonical.lower() == student_track.lower()
                if intent == "track" and same_track:
                    confidence = min(confidence + 0.15, 1.0)
                elif intent == "course" and not same_track:
                    confidence = min(confidence + 0.10, 1.0)

            if confidence >= threshold:
                if intent == "course":
                    return ("course", course_canonical, None)
                else:
                    return ("track", track_canonical, None)

        # No clear signal → ask the student
        return (
            "ambiguous",
            None,
            self._build_track_course_clarification(term, track_canonical, query, char_pos),
        )

    def _build_track_course_clarification(
        self,
        term:            str,
        track_canonical: str,
        query:           Optional[str] = None,
        char_pos:        int           = -1,
    ) -> str:
        """Build a friendly course-vs-track disambiguation question.
        When query and char_pos are provided, a ±1-word context snippet is
        included so the student knows which occurrence is being asked about.
        """
        if query and char_pos >= 0:
            snippet     = self._context_snippet(query, term, char_pos)
            term_header = f'**"{term}"** (in: "{snippet}")'
        else:
            term_header = f'**"{term}"**'
        return (
            f'{term_header} can refer to either a course or a track/program:\n\n'
            f'  **1.** A **course**\n'
            f'  **2.** The *{track_canonical.title()}* **track/program**\n\n'
            'Which did you mean? (Reply with 1, 2, "course", or "track")'
        )

    @staticmethod
    def _pick_course_vs_track_from_reply(reply: str) -> str:
        """
        Parse the student's answer to a course-vs-track question.
        Returns "course" or "track".  Defaults to "course" when unclear.
        """
        r     = reply.strip().lower()
        words = set(r.split())

        # Number-based takes priority
        if words & {"1", "one", "first"}:
            return "course"
        if words & {"2", "two", "second"}:
            return "track"

        # Keyword vote
        course_votes = sum(1 for w in ["course", "class", "subject"] if w in words)
        track_votes  = sum(1 for w in ["track", "program", "major"]  if w in words)

        if course_votes > track_votes:
            return "course"
        if track_votes > course_votes:
            return "track"

        return "course"  # default

    def _resolve_course_vs_track(
        self,
        pending:       PendingAmbiguity,
        student_reply: str,
    ) -> PreprocessResult:
        """
        Handle the student's answer to a course-vs-track disambiguation question.

        After resolving the flagged conflict this method also:
          1. Processes any other pending track-vs-course conflicts.
          2. Runs Step-3-style mapping for remaining pending_courses.
          3. Re-runs track mapping from the dereferenced query.
          4. Rewrites and returns the clean query.
        """
        choice = self._pick_course_vs_track_from_reply(student_reply)

        all_courses = dict(pending.resolved_courses)
        all_tracks  = dict(pending.resolved_tracks)

        if choice == "track":
            all_tracks[pending.ambiguous_term] = pending.track_canonical
            box(
                "📝  COURSE vs TRACK RESOLVED",
                [
                    f'Term           : "{pending.ambiguous_term}"',
                    f"Student chose  : TRACK",
                    f'Resolved to    : "{pending.track_canonical}"',
                ],
            )
        else:
            # Student picked "course" — run full course mapping to check
            # whether the term is ambiguous between MULTIPLE courses.
            deduped_term  = self._dedupe_chars(pending.ambiguous_term)
            course_result = self._map_course(deduped_term)

            if course_result["status"] == "ambiguous":
                # Multiple courses match — chain into course-name disambiguation
                # so the student can pick the specific course they meant.
                box(
                    "📝  COURSE vs TRACK RESOLVED",
                    [
                        f'Term           : "{pending.ambiguous_term}"',
                        f"Student chose  : COURSE",
                        f"Multiple course matches — asking student to pick one",
                    ],
                )
                # term_position not known yet (student still picking which course)
                # — propagate the chain history as-is, without adding this term yet.
                new_pending = PendingAmbiguity(
                    original_query   = pending.original_query,
                    dereferenced     = pending.dereferenced,
                    ambiguous_term   = pending.ambiguous_term,
                    term_position    = pending.term_position,
                    candidates       = course_result["candidates"],
                    resolved_courses = all_courses,
                    pending_courses  = pending.pending_courses,
                    resolved_tracks  = all_tracks,
                    history          = pending.history,
                    ambiguity_type   = "course_name",
                    student_track    = pending.student_track,
                    pending_track_course_conflicts = pending.pending_track_course_conflicts,
                    resolved_course_occ       = list(pending.resolved_course_occ),
                    pending_track_occurrences  = list(pending.pending_track_occurrences),
                )
                return PreprocessResult(
                    status        = "ambiguous",
                    clarification = self._build_clarification(
                        pending.ambiguous_term, course_result["candidates"],
                        pending.dereferenced, pending.term_position,
                    ),
                    pending       = new_pending,
                )

            # Single winner (or alias — already in course_canonical)
            canonical = (
                course_result.get("canonical") or pending.course_canonical
            )
            box(
                "📝  COURSE vs TRACK RESOLVED",
                [
                    f'Term           : "{pending.ambiguous_term}"',
                    f"Student chose  : COURSE",
                    f'Resolved to    : "{canonical}"',
                ],
            )
            all_courses[pending.ambiguous_term] = canonical

        # Build position-keyed occurrence lists so every chained PendingAmbiguity
        # carries the full resolution history for both courses AND tracks.
        #
        # chain_course_occ — occurrences resolved as COURSE (for skip-guard and rewriter)
        # IMPORTANT: only add the current term when choice == "course".
        # The same surface text may have been resolved as COURSE at a different position
        # (e.g. "ai" @8 → course); that entry is already in pending.resolved_course_occ.
        # Adding it here for a TRACK choice would mis-label this occurrence as a course.
        chain_course_occ: List[Tuple[str, int, str]] = list(pending.resolved_course_occ)
        if choice == "course" and pending.term_position >= 0:
            chain_course_occ.append(
                (pending.ambiguous_term, pending.term_position,
                 all_courses[pending.ambiguous_term])
            )

        # chain_pending_track_occ — occurrences that still need track mapping.
        # Start from the auto-TRACK list carried by the parent pending, then
        # add the occurrence the student JUST resolved as TRACK (if any) so
        # the downstream rewriter can replace it too.
        chain_pending_track_occ: List[Tuple[str, int, str]] = list(pending.pending_track_occurrences)
        if choice == "track" and pending.term_position >= 0:
            deduped_t = self._dedupe_chars(pending.ambiguous_term)
            chain_pending_track_occ.append(
                (pending.ambiguous_term, pending.term_position, deduped_t)
            )

        # ── 1. Process remaining track-vs-course conflicts ─────────────────
        for rem_term, rem_pos_c, rem_course, rem_track in pending.pending_track_course_conflicts:
            resolution, _, clarification = self._resolve_track_course_conflict(
                rem_term, pending.dereferenced, rem_course, rem_track, pending.student_track,
                char_pos=rem_pos_c if rem_pos_c >= 0 else None,
            )
            if resolution == "course":
                all_courses[rem_term] = rem_course
                # Track this occurrence as course-resolved so the rewriter skips it
                if rem_pos_c >= 0 and not any(o == rem_term and p == rem_pos_c for o, p, _ in chain_course_occ):
                    chain_course_occ.append((rem_term, rem_pos_c, rem_course))
            elif resolution == "track":
                all_tracks[rem_term] = rem_track
                # Add to pending track occurrences for downstream rewriter
                deduped_rc = self._dedupe_chars(rem_term)
                if rem_pos_c >= 0 and (rem_term, rem_pos_c) not in {(o, p) for o, p, _ in chain_pending_track_occ}:
                    chain_pending_track_occ.append((rem_term, rem_pos_c, deduped_rc))
            else:
                # Still ambiguous → ask about this one
                remaining_conflicts = [
                    item for item in pending.pending_track_course_conflicts
                    if item[0] != rem_term or item[1] != rem_pos_c
                ]
                new_pending = PendingAmbiguity(
                    original_query   = pending.original_query,
                    dereferenced     = pending.dereferenced,
                    ambiguous_term   = rem_term,
                    term_position    = rem_pos_c,
                    candidates       = [],
                    resolved_courses = all_courses,
                    pending_courses  = pending.pending_courses,
                    resolved_tracks  = all_tracks,
                    history          = pending.history,
                    ambiguity_type   = "course_vs_track",
                    course_canonical = rem_course,
                    track_canonical  = rem_track,
                    student_track    = pending.student_track,
                    pending_track_course_conflicts = remaining_conflicts,
                    resolved_course_occ       = list(chain_course_occ),
                    pending_track_occurrences  = list(chain_pending_track_occ),
                )
                return PreprocessResult(
                    status        = "ambiguous",
                    clarification = clarification,
                    pending       = new_pending,
                )

        # ── 2. Run Step-3-style mapping for remaining course occurrences ──────
        # pending_courses is List[Tuple[str, int, str]] = [(original, char_pos, deduped), ...]
        for idx, (rem_orig, rem_pos, rem_deduped) in enumerate(pending.pending_courses):
            rem_result = self._map_course(rem_deduped)
            if rem_result["status"] == "resolved":
                all_courses[rem_orig] = rem_result["canonical"]
                chain_course_occ.append((rem_orig, rem_pos, rem_result["canonical"]))
            elif rem_result["status"] == "ambiguous":
                next_rem = list(pending.pending_courses[idx + 1:])
                new_pending = PendingAmbiguity(
                    original_query   = pending.original_query,
                    dereferenced     = pending.dereferenced,
                    ambiguous_term   = rem_orig,
                    term_position    = rem_pos,
                    candidates       = rem_result["candidates"],
                    resolved_courses = all_courses,
                    pending_courses  = next_rem,
                    resolved_tracks  = all_tracks,
                    history          = pending.history,
                    ambiguity_type   = "course_name",
                    student_track    = pending.student_track,
                    resolved_course_occ       = list(chain_course_occ),
                    pending_track_occurrences  = list(chain_pending_track_occ),
                )
                return PreprocessResult(
                    status        = "ambiguous",
                    clarification = self._build_clarification(
                        rem_orig, rem_result["candidates"], pending.dereferenced, rem_pos
                    ),
                    pending       = new_pending,
                )
            # else "not_found": leave it for the agent

        # ── 3. Build full course-resolved occurrence list ─────────────────────
        # chain_course_occ already contains every course-resolved occurrence:
        #   pending.resolved_course_occ  (chain history from parent rounds)
        #   + current term if choice=="course"  (added above)
        #   + courses from pending_track_course_conflicts auto-resolved as course (step 1)
        #   + courses from pending_courses (step 2)
        # Using it directly avoids the "in all_courses" check that fired even for
        # TRACK choices when the same surface text had been resolved as course elsewhere.
        all_course_occ: List[Tuple[str, int, str]] = chain_course_occ
        course_resolved_pos = {(o, p) for o, p, _ in all_course_occ}

        # ── Map remaining track occurrences (position-aware) ─────────────────
        # chain_pending_track_occ is always the superset: it starts from
        # pending.pending_track_occurrences and appends any term just resolved
        # as TRACK in this call. Always use it so the newly-resolved occurrence
        # is included in the positional replacement pass.
        if chain_pending_track_occ:
            raw_track_occ = chain_pending_track_occ
        else:
            _, raw_tracks_fb = self._extract_entities(pending.dereferenced)
            raw_track_occ = [(t, p, self._dedupe_chars(t)) for t, p in raw_tracks_fb]

        resolved_track_occ: List[Tuple[str, int, str]] = []
        # Carry forward auto-resolved tracks from step 1
        for term, canon in all_tracks.items():
            for o2, p2, _ in raw_track_occ:
                if o2 == term and (o2, p2) not in course_resolved_pos:
                    resolved_track_occ.append((term, p2, canon))
                    break

        track_debug: List[str] = []
        # track-fallback collection for _resolve_course_vs_track path
        track_fallback_ambiguous_cvt: List[Tuple[str, int, str, List[Dict]]] = []
        for orig, pos, ded in raw_track_occ:
            if (orig, pos) in course_resolved_pos:
                track_debug.append(f'"{orig}" @{pos}  →  skipped (resolved as course)')
                continue
            if any(o == orig and p == pos for o, p, _ in resolved_track_occ):
                continue  # already in list from all_tracks carry-forward above
            canonical, method = self._map_track_with_method(ded)
            if canonical:
                all_tracks[orig] = canonical
                resolved_track_occ.append((orig, pos, canonical))
                track_debug.append(f'"{orig}" @{pos}  →  {canonical}  [{method}]')
            else:
                # Track mapping failed — try as a course (LLM misclassification)
                cr = self._map_course(ded)
                if cr["status"] == "resolved":
                    all_courses[orig] = cr["canonical"]
                    all_course_occ.append((orig, pos, cr["canonical"]))
                    course_resolved_pos = {(o2, p2) for o2, p2, _ in all_course_occ}
                    track_debug.append(
                        f'"{orig}" @{pos}  →  course "{cr["canonical"]}"  [track-fallback]'
                    )
                elif cr["status"] == "ambiguous":
                    track_fallback_ambiguous_cvt.append((orig, pos, ded, cr["candidates"]))
                    track_debug.append(f'"{orig}" @{pos}  →  AMBIGUOUS as course  [track-fallback]')
                else:
                    track_debug.append(f'"{orig}" @{pos}  →  not found')

        if raw_track_occ:
            box("📝  TRACK MAPPING (post-disambiguation)", track_debug)

        # Handle ambiguous track-fallback terms — ask the student, chain remainder
        if track_fallback_ambiguous_cvt:
            orig0, pos0, ded0, cands0 = track_fallback_ambiguous_cvt[0]
            rest_cvt = track_fallback_ambiguous_cvt[1:]
            remaining_cvt = [(ro, rp, rd) for ro, rp, rd, _ in rest_cvt]
            new_pending = PendingAmbiguity(
                original_query   = pending.original_query,
                dereferenced     = pending.dereferenced,
                ambiguous_term   = orig0,
                term_position    = pos0,
                candidates       = cands0,
                resolved_courses = all_courses,
                pending_courses  = remaining_cvt,
                resolved_tracks  = all_tracks,
                history          = pending.history,
                resolved_course_occ      = list(all_course_occ),
                pending_track_occurrences = [],
            )
            return PreprocessResult(
                status        = "ambiguous",
                clarification = self._build_clarification(
                    orig0, cands0, pending.dereferenced, pos0
                ),
                pending       = new_pending,
            )

        # ── 4. Rewrite the query positionally ─────────────────────────────────
        pos_replacements: List[Tuple[int, int, str]] = []
        for o, p, canon in all_course_occ:
            if p >= 0:
                pos_replacements.append((p, p + len(o), f"'{canon}' course"))
        for o, p, canon in resolved_track_occ:
            if p >= 0:
                pos_replacements.append((p, p + len(o), f"'{canon}' program"))
        # Deduped fallback for unresolved terms
        for o, p, ded in raw_track_occ:
            if (o, p) not in course_resolved_pos and o != ded and p >= 0:
                if not any(r[0] == p for r in pos_replacements):
                    pos_replacements.append((p, p + len(o), ded))

        if pos_replacements:
            clean = self._rewrite_query_positional(pending.dereferenced, pos_replacements)
            dict_rep: Dict[str, str] = {}
            for o, p, canon in all_course_occ:
                dict_rep[o] = f"'{canon}' course"
            for o, p, canon in resolved_track_occ:
                if o not in dict_rep:
                    dict_rep[o] = f"'{canon}' program"
            still_unreplaced = any(
                v.lower() not in clean.lower()
                for k, v in dict_rep.items()
                if k.lower() != v.lower()
            )
            if still_unreplaced:
                clean = self._rewrite_query(pending.dereferenced, dict_rep)
        else:
            clean = pending.dereferenced

        box(
            "📝  CLEAN QUERY (sent to agent)",
            [f"Original : {pending.original_query}", f"Clean    : {clean}"],
        )

        return PreprocessResult(status="ready", clean_query=clean, resolved_courses=all_courses)

    # ── Step 5: Query rewriting ───────────────────────────────────────────

    @staticmethod
    def _rewrite_query_positional(
        text:         str,
        replacements: List[Tuple[int, int, str]],
    ) -> str:
        """
        Apply substring replacements at exact character positions, right-to-left.

        replacements: [(start_pos, end_pos, replacement_text), ...]

        Applying right-to-left (descending start position) ensures that
        replacements at earlier positions are not shifted by later ones.
        Overlapping replacements are silently skipped.
        """
        result    = text
        covered: List[Tuple[int, int]] = []  # (start, end) regions already replaced

        for start, end, replacement in sorted(replacements, key=lambda x: x[0], reverse=True):
            # Skip if this region overlaps one already processed
            if any(s <= start < e or s < end <= e for s, e in covered):
                continue
            result  = result[:start] + replacement + result[end:]
            covered.append((start, end))

        return result

    def _rewrite_query(
        self,
        query:        str,
        replacements: Dict[str, str],
    ) -> str:
        """
        Substitute resolved names back into the query naturally.
        replacements: {original_term -> canonical_name_or_deduped_spelling}

        Uses simple regex substitution first (fast path), then falls back to
        the LLM for cases where simple replacement leaves the term unchanged
        (e.g. multi-word partial matches like "soft eng" -> "software engineering").
        """
        if not replacements:
            return query

        all_replacements = replacements

        # Try simple case-insensitive substitution first (fast path)
        result = query
        for term, canonical in all_replacements.items():
            # Case-insensitive whole-word-ish replace
            import re
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            result  = pattern.sub(canonical, result)

        # Fall back to LLM if any canonical name is missing from the result.
        # This handles two failure modes:
        #   1. Partial matches: "soft eng" → "software engineering" (original still in result)
        #   2. Implied terms: "training 1 and 2" → entity "training 2" never existed as a
        #      literal substring, so regex substitution silently did nothing.
        still_unreplaced = any(
            canonical.lower() not in result.lower()
            for term, canonical in all_replacements.items()
            if term.lower() != canonical.lower()
        )
        if still_unreplaced:
            result = self._llm_rewrite(query, all_replacements)

        return result

    def _llm_rewrite(self, query: str, replacements: Dict[str, str]) -> str:
        """Ask the LLM to rewrite the query with canonical names."""
        mapping_text = "\n".join(
            f'  "{term}" → "{canonical}"'
            for term, canonical in replacements.items()
        )
        prompt = f"""Rewrite this student question by replacing the terms below with their canonical names.

Original question: "{query}"

Replacements:
{mapping_text}

Rules:
- Keep the meaning and tone identical.
- Only replace the specified terms, change nothing else.
- Return ONLY the rewritten question, no explanation.

Rewritten question:"""

        try:
            rewritten = llm_call_text(
                system="Rewrite student questions substituting course/track names as instructed. Return only the rewritten question.",
                user=prompt,
                temperature=0,
                max_tokens=150,
            ).strip().strip('"')
            if len(rewritten) < 3:
                return query
            return rewritten
        except Exception:
            return query

    # ── Disambiguation helpers ────────────────────────────────────────────

    @staticmethod
    def _context_snippet(query: str, term: str, char_pos: int, window: int = 1) -> str:
        """
        Return a ±window-word context string around `term` at `char_pos`.

        Example (window=1):
            query   = "what is special courses at ai and what is ai"
            term    = "ai", char_pos = 27  →  'at **ai** and'
            term    = "ai", char_pos = 42  →  'is **ai**'
        """
        if char_pos < 0:
            return f"**{term}**"
        words     = query.split()
        word_idx  = len(query[:char_pos].split())          # words before term
        t_wcount  = len(term.split())
        before    = words[max(0, word_idx - window): word_idx]
        after     = words[word_idx + t_wcount: word_idx + t_wcount + window]
        parts: List[str] = []
        if before:
            parts.append(" ".join(before))
        parts.append(f"**{term}**")
        if after:
            parts.append(" ".join(after))
        return " ".join(parts)

    @staticmethod
    def _strip_trailing_blocklist(term: str) -> str:
        """
        Remove trailing blocklist words from an extracted term.

        Examples:
            "ai courses"     → "ai"
            "data course"    → "data"
            "ml electives"   → "ml"
            "special topics" → "special topics"  (no blocklist word at end)
        """
        words = term.strip().split()
        while words and words[-1].lower() in ENTITY_BLOCKLIST:
            words.pop()
        return " ".join(words)

    @staticmethod
    def _split_course_track_compound(term: str) -> Optional[Tuple[str, str]]:
        """
        If `term` looks like "X in/at/for Y" where Y is a TRACK_ALIASES key,
        split it into (course_part=X, track_part=Y).
        Returns None when the term does not match this pattern.

        Examples:
            "special topics in ai"   → ("special topics", "ai")
            "electives in aim"        → ("electives", "aim")   ← course_part may still be blocklist
            "machine learning in sad" → ("machine learning", "sad")
            "data visualization"     → None  (no preposition + track alias)
        """
        term_lower = term.lower()
        for prep in (" in ", " at ", " for "):
            idx = term_lower.find(prep)
            if idx == -1:
                continue
            right = term[idx + len(prep):].strip()
            if right.lower() in TRACK_ALIASES:
                left = term[:idx].strip()
                if left:
                    return left, right
        return None

    def _build_clarification(
        self,
        term:       str,
        candidates: List[Dict],
        query:      Optional[str] = None,
        char_pos:   int           = -1,
    ) -> str:
        """
        Build a friendly disambiguation question to show the student.
        When `query` and `char_pos` are provided, a ±1-word context snippet
        is included so the student knows which occurrence is being asked about.
        The last option always lets the student keep the original term as-is.
        """
        if query and char_pos >= 0:
            snippet = self._context_snippet(query, term, char_pos)
            header  = f'I found several courses matching **"{term}"** (in: "{snippet}"). Which one did you mean?\n'
        else:
            header  = f'I found several courses matching **"{term}"**. Which one did you mean?\n'
        lines = [header]
        for i, c in enumerate(candidates, 1):
            code = f"({c['code']})" if c.get("code") else ""
            lines.append(f"  **{i}.** {c['name'].title()} {code}")
        # Extra option: keep original term unchanged
        keep_idx = len(candidates) + 1
        lines.append(f"  **{keep_idx}.** {term}")
        lines.append(
            "\nPlease reply with the number or name of the course you meant."
        )
        return "\n".join(lines)

    # Sentinel returned when the student chooses to keep their original term.
    _KEEP_ORIGINAL = "__KEEP_ORIGINAL__"

    def _pick_from_reply(self, reply: str, candidates: List[Dict]) -> str:
        """
        Parse the student's disambiguation reply and return the chosen name.

        Handles:
          - Number: "1", "first", "the first one"
            * N+1  (last option) → _KEEP_ORIGINAL sentinel
          - Name: "software engineering", "soft eng"
          - Fallback: best fuzzy match against candidate names
        """
        reply_strip = reply.strip().lower()
        keep_idx = len(candidates) + 1  # index of the "keep original" option

        # Number-based selection
        number_words = {
            "1": 1, "one": 1, "first": 1,
            "2": 2, "two": 2, "second": 2,
            "3": 3, "three": 3, "third": 3,
            "4": 4, "four": 4, "fourth": 4,
            "5": 5, "five": 5, "fifth": 5,
        }
        for word, idx in number_words.items():
            if word in reply_strip.split() or reply_strip == word:
                if idx == keep_idx:
                    return self._KEEP_ORIGINAL
                if 1 <= idx <= len(candidates):
                    return candidates[idx - 1]["name"]

        # Name-based selection — fuzzy match against candidate names
        scored = [
            (c["name"], _similarity(reply_strip, c["name"]))
            for c in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        if scored and scored[0][1] >= 0.3:
            return scored[0][0]

        # Default: first candidate
        return candidates[0]["name"]


# ═════════════════════════════════════════════════════════════════════════════
# Module-level singleton
# ═════════════════════════════════════════════════════════════════════════════

_PREPROCESSOR: Optional[QueryPreprocessor] = None


def get_preprocessor() -> QueryPreprocessor:
    global _PREPROCESSOR
    if _PREPROCESSOR is None:
        _PREPROCESSOR = QueryPreprocessor()
    return _PREPROCESSOR