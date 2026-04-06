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
    "dsa":             "data structures and algorithms",
    "algo":            "algorithms",
    "alg":             "algorithms",
    "ds&a":            "data structures and algorithms",

    # Core / foundations
    "prob":            "probability and statistical methods",
    "stats":           "probability and statistical methods",
    "stat":            "probability and statistical methods",
    "calc":            "calculus",
    "la":              "linear algebra",
    "linalg":          "linear algebra",
    "dm":              "discrete mathematics",
    "discmath":        "discrete mathematics",

    # Specific BNU course codes (students often type the code)
    "bcs311":          "artificial intelligence",
    "bas201":          "probability and statistical methods",
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
    "aim":                      "artificial intelligence and machine learning",
    "ai":                       "artificial intelligence and machine learning",
    "ai track":                 "artificial intelligence and machine learning",
    "ai and ml":                "artificial intelligence and machine learning",
    "ai & ml":                  "artificial intelligence and machine learning",
    "aiml":                     "artificial intelligence and machine learning",
    "machine learning track":   "artificial intelligence and machine learning",
    "ml track":                 "artificial intelligence and machine learning",

    # Software track
    "sad":                      "software and application development",
    "software":                 "software and application development",
    "software dev":             "software and application development",
    "software development":     "software and application development",
    "app dev":                  "software and application development",
    "sw":                       "software and application development",

    # Data science track
    "das":                      "data science",
    "data science":             "data science",
    "data":                     "data science",
    "ds track":                 "data science",
}

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
    to resolve an ambiguous course name.
    """
    original_query:   str                    # raw user message
    dereferenced:     str                    # after Step 1 (reference resolution)
    ambiguous_term:   str                    # the term the student typed, e.g. "soft"
    candidates:       List[Dict]             # [{name, code, confidence}, ...]
    resolved_courses: Dict[str, str]         # already-resolved course mappings so far
    resolved_tracks:  Dict[str, str]         # already-resolved track mappings
    history:          List[Dict]             # conversation history at time of query


@dataclass
class PreprocessResult:
    """Return value of QueryPreprocessor.process()."""
    status:        str                       # "ready" | "ambiguous" | "passthrough"
    clean_query:   str          = ""         # fully resolved query (when ready/passthrough)
    clarification: str          = ""         # question to show student (when ambiguous)
    pending:       Optional[PendingAmbiguity] = None


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
            "MATCH (c:Course) RETURN c.name AS name, c.code AS code ORDER BY c.name"
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
        query:   str,
        history: List[Dict],
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
        raw_courses, raw_tracks = self._extract_entities(resolved)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Step 2 — Entity filtering + Char deduplication
        # ─────────────────────────────────────────────────────────────────
        # Filter: discard any term whose lowercase form is in ENTITY_BLOCKLIST
        # (generic words like "electives", "semester", "year" that are not
        # actual course or track names).
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        raw_courses = [t for t in raw_courses if t.lower().strip() not in ENTITY_BLOCKLIST]
        raw_tracks  = [t for t in raw_tracks  if t.lower().strip() not in ENTITY_BLOCKLIST]

        # original → deduped (may equal original when no dupe chars found)
        course_orig_to_dedup: Dict[str, str] = {
            t: self._dedupe_chars(t) for t in raw_courses
        }
        track_orig_to_dedup: Dict[str, str] = {
            t: self._dedupe_chars(t) for t in raw_tracks
        }

        dedup_notes = (
            [f'Course  "{o}" -> "{d}"' for o, d in course_orig_to_dedup.items() if o != d]
            + [f'Track   "{o}" -> "{d}"' for o, d in track_orig_to_dedup.items() if o != d]
        )

        step12_lines = [
            f"Working query : {resolved}",
            f"Courses found : {list(course_orig_to_dedup.keys()) or '(none)'}",
            f"Tracks  found : {list(track_orig_to_dedup.keys())  or '(none)'}",
        ]
        if dedup_notes:
            step12_lines += ["", "Char deduplication:"] + [f"  {n}" for n in dedup_notes]

        box("📝  STEP 1&2 — Entity Extraction + Char Deduplication", step12_lines)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Step 3 — Course mapping
        # resolved_courses: original_term → canonical_name
        # This ensures _rewrite_query replaces the exact substring in query.
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        resolved_courses: Dict[str, str] = {}  # original → canonical
        course_debug_lines: List[str] = []

        for original, deduped in course_orig_to_dedup.items():
            result = self._map_course(deduped)

            if result["status"] == "resolved":
                canonical = result["canonical"]
                method    = result.get("method", "fuzzy match")
                resolved_courses[original] = canonical          # key = ORIGINAL
                label = "(same)" if deduped.lower() == canonical.lower() else f"→  {canonical}"
                course_debug_lines.append(
                    f'"{original}"'
                    + (f' [deduped: "{deduped}"]' if original != deduped else "")
                    + f"  {label}  [{method}]"
                )

            elif result["status"] == "ambiguous":
                course_debug_lines.append(
                    f'"{original}" (deduped: "{deduped}")  →  AMBIGUOUS — '
                    f"{len(result['candidates'])} close matches found"
                )
                for i, c in enumerate(result["candidates"], 1):
                    course_debug_lines.append(
                        f"    {i}. {c['name']} ({c['code']})  "
                        f"confidence={c['confidence']:.2f}"
                    )
                box("📝  STEP 3 — Course Mapping", course_debug_lines)

                pending       = PendingAmbiguity(
                    original_query   = query,
                    dereferenced     = resolved,        # use resolved version
                    ambiguous_term   = original,        # ORIGINAL term
                    candidates       = result["candidates"],
                    resolved_courses = resolved_courses,
                    resolved_tracks  = {},
                    history          = history,
                )
                clarification = self._build_clarification(original, result["candidates"])
                return PreprocessResult(
                    status        = "ambiguous",
                    clarification = clarification,
                    pending       = pending,
                )
            else:
                course_debug_lines.append(
                    f'"{original}"'
                    + (f' [deduped: "{deduped}"]' if original != deduped else "")
                    + "  →  not found"
                )

        if course_orig_to_dedup:
            box("📝  STEP 3 — Course Mapping", course_debug_lines)
        else:
            box("📝  STEP 3 — Course Mapping", ["No course terms to map."])

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Step 4 — Track mapping
        # resolved_tracks: original_term → canonical_name
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        resolved_tracks: Dict[str, str] = {}
        track_debug_lines: List[str] = []

        for original, deduped in track_orig_to_dedup.items():
            canonical, method = self._map_track_with_method(deduped)
            if canonical:
                resolved_tracks[original] = canonical           # key = ORIGINAL
                label = "(same)" if deduped.lower() == canonical.lower() else f"→  {canonical}"
                track_debug_lines.append(
                    f'"{original}"'
                    + (f' [deduped: "{deduped}"]' if original != deduped else "")
                    + f"  {label}  [{method}]"
                )
            else:
                track_debug_lines.append(
                    f'"{original}"'
                    + (f' [deduped: "{deduped}"]' if original != deduped else "")
                    + "  →  not found"
                )

        if track_orig_to_dedup:
            box("📝  STEP 4 — Track Mapping", track_debug_lines)
        else:
            box("📝  STEP 4 — Track Mapping", ["No track terms to map."])

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Step 5 — Query rewriting
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Build the full replacement table:
        #   mapped terms    → canonical name
        #   unmapped terms  → deduped spelling (minimum cleanup)
        all_replacements: Dict[str, str] = {}

        # Mapped courses and tracks
        all_replacements.update(resolved_courses)
        all_replacements.update(resolved_tracks)

        # Unmapped terms that were at least deduped (original != deduped)
        for original, deduped in {**course_orig_to_dedup, **track_orig_to_dedup}.items():
            if original not in all_replacements and original != deduped:
                all_replacements[original] = deduped

        if not all_replacements:
            # Even with no name mappings, use the resolved query (refs replaced)
            if resolved != query:
                box(
                    "📝  STEP 5 — Clean Query (sent to agent)",
                    [f"Original : {query}", f"Clean    : {resolved}  (references resolved)"],
                )
                return PreprocessResult(status="ready", clean_query=resolved)
            box("📝  STEP 5 — Query Rewrite",
                ["Nothing changed — query passed through as-is."])
            return PreprocessResult(status="passthrough", clean_query=query)

        # Rewrite on the resolved text (not the raw query) so the canonical
        # names are substituted into the dereferenced version
        clean = self._rewrite_query(resolved, all_replacements)

        box(
            "📝  STEP 5 — Clean Query (sent to agent)",
            [f"Original : {query}", f"Clean    : {clean}"],
        )

        return PreprocessResult(status="ready", clean_query=clean)

    def resolve_ambiguity(
        self,
        pending: PendingAmbiguity,
        student_reply: str,
    ) -> PreprocessResult:
        """
        Called when the student answers a disambiguation question.

        1. Parse the student's reply to pick the chosen course.
        2. Merge with already-resolved courses in pending.
        3. Re-run track mapping on the original query.
        4. Rewrite the original query with all resolved names.
        5. Show debug boxes for the resolution.
        """
        chosen = self._pick_from_reply(student_reply, pending.candidates)

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

        # Re-run track mapping on the original query
        _, raw_tracks = self._extract_entities(pending.original_query)
        resolved_tracks: Dict[str, str] = {}
        track_debug: List[str] = []
        for t in raw_tracks:
            deduped = self._dedupe_chars(t)
            canonical, method = self._map_track_with_method(deduped)
            if canonical:
                resolved_tracks[t] = canonical
                track_debug.append(f'"{t}"  →  {canonical}  [{method}]')
            else:
                track_debug.append(f'"{t}"  →  not found')

        if raw_tracks:
            box("📝  TRACK MAPPING (post-disambiguation)", track_debug)

        # Rewrite the original query
        all_replacements = {**all_courses, **resolved_tracks}
        # Also replace any unmapped deduped terms
        orig_to_dedup = {t: self._dedupe_chars(t) for t in
                         list(all_courses.keys()) + list(raw_tracks)}
        for orig, ded in orig_to_dedup.items():
            if orig not in all_replacements and orig != ded:
                all_replacements[orig] = ded

        clean = self._rewrite_query(pending.original_query, all_replacements)

        box(
            "📝  CLEAN QUERY (sent to agent)",
            [
                f"Original : {pending.original_query}",
                f"Clean    : {clean}",
            ],
        )

        return PreprocessResult(status="ready", clean_query=clean)

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

    def _extract_entities(self, query: str) -> Tuple[List[str], List[str]]:
        """
        Extract course mentions and track/program mentions from the query.

        Returns: (list_of_course_terms, list_of_track_terms)

        Uses the LLM for robust extraction from casual language.
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
    aim, ai, aiml → artificial intelligence and machine learning
    sad, software, sw, app dev → software and application development
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

            # Post-process: if a "course" term is actually a known track alias,
            # move it to tracks. This handles the case where the LLM misclassifies
            # "das", "ai", "sad" etc. as course abbreviations.
            reclassified = [c for c in courses if c.lower().strip() in TRACK_ALIASES]
            courses = [c for c in courses if c.lower().strip() not in TRACK_ALIASES]
            tracks  = tracks + reclassified

            return courses, tracks
        except Exception as exc:
            logger.debug("Entity extraction failed: %s", exc)
            return [], []

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

        # ── b) Fuzzy match against live course list ───────────────────────
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

        # ── c) Exact or single clear winner ──────────────────────────────
        if best["confidence"] >= 0.95:
            return {"status": "resolved", "canonical": best["name"], "method": f"fuzzy match (confidence={best['confidence']:.2f})"}

        # ── d) Check for ambiguity (two close scores) ─────────────────────
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

        # ── e) Single winner with reasonable confidence ───────────────────
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

    # ── Step 5: Query rewriting ───────────────────────────────────────────

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

        # If the result still contains an unresolved original term, use LLM
        # (handles partial matches like "soft eng" -> "software engineering")
        still_unreplaced = any(
            term.lower() in result.lower() and term.lower() != canonical.lower()
            for term, canonical in all_replacements.items()
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

    def _build_clarification(self, term: str, candidates: List[Dict]) -> str:
        """
        Build a friendly disambiguation question to show the student.
        """
        lines = [f'I found several courses matching **"{term}"**. Which one did you mean?\n']
        for i, c in enumerate(candidates, 1):
            code = f"({c['code']})" if c.get("code") else ""
            lines.append(f"  **{i}.** {c['name'].title()} {code}")
        lines.append(
            "\nPlease reply with the number or name of the course you meant."
        )
        return "\n".join(lines)

    def _pick_from_reply(self, reply: str, candidates: List[Dict]) -> str:
        """
        Parse the student's disambiguation reply and return the chosen name.

        Handles:
          - Number: "1", "first", "the first one"
          - Name: "software engineering", "soft eng"
          - Fallback: best fuzzy match against candidate names
        """
        reply_strip = reply.strip().lower()

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