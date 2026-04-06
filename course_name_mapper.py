"""
Course Name Mapper - FIXED VERSION with lazy loading, error handling,
and prefix-match scoring so partial inputs like "soft" map to
"software engineering" even when sequence-similarity is low.
"""

import os
import logging
from typing import List, Dict, Optional
from difflib import SequenceMatcher
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class CourseNameMapper:
    """Maps user input to actual Neo4j course names using fuzzy matching."""

    def __init__(self, lazy_load: bool = True):
        self.driver = None
        self._course_cache = None
        self._initialized = False

        if not lazy_load:
            self._initialize()

    def _initialize(self):
        if self._initialized:
            return True
        try:
            uri      = os.getenv("NEO4J_URI")
            user     = os.getenv("NEO4J_USERNAME")
            password = os.getenv("NEO4J_PASSWORD")

            if not all([uri, user, password]):
                print("⚠️  Neo4j credentials not found in .env - course mapper disabled")
                return False

            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self._load_courses()
            self._initialized = True
            return True

        except Exception as e:
            print(f"⚠️  Failed to initialize course mapper: {e}")
            print("   Falling back to hardcoded aliases only")
            return False

    def _load_courses(self):
        query = """
        MATCH (c:Course)
        RETURN c.name AS name, c.code AS code
        ORDER BY c.name
        """
        try:
            with self.driver.session() as session:
                result = session.run(query)
                self._course_cache = [
                    {
                        "name":       record["name"],
                        "code":       record["code"],
                        "name_lower": record["name"].lower() if record["name"] else "",
                    }
                    for record in result
                ]
            logger.debug("Loaded %d courses from Neo4j", len(self._course_cache))
        except Exception as e:
            print(f"⚠️  Failed to load courses: {e}")
            self._course_cache = []

    def refresh_cache(self):
        if not self._initialized:
            self._initialize()
        if self._initialized:
            self._load_courses()

    def _ensure_initialized(self) -> bool:
        if not self._initialized:
            return self._initialize()
        return True

    def _similarity_score(self, str1: str, str2: str) -> float:
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

    def _extract_keywords(self, text: str) -> List[str]:
        stop_words = {'and', 'the', 'with', 'for', 'of', 'in', 'to', 'a', 'an'}
        words = text.lower().split()
        return [w for w in words if w not in stop_words]

    def _keyword_match_score(self, user_input: str, course_name: str) -> float:
        user_keywords   = set(self._extract_keywords(user_input))
        course_keywords = set(self._extract_keywords(course_name))

        if not user_keywords:
            return 0.0

        overlap = len(user_keywords & course_keywords)
        return overlap / len(user_keywords)

    def _code_match_score(self, user_input: str, course_code: str) -> float:
        if not course_code:
            return 0.0

        user_upper = user_input.upper().strip()

        if user_upper == course_code:
            return 1.0
        if user_upper.replace(" ", "") == course_code:
            return 1.0
        if course_code in user_upper:
            return 0.9

        return 0.0

    def _prefix_match_score(self, user_input: str, course_name_lower: str) -> float:
        """
        Score based on whether the user's input is a prefix of the course name.

        This fixes "soft" → "software engineering" (and similar partial inputs)
        that score too low on pure sequence-similarity.

        Rules:
        - Minimum input length: 3 chars (avoid false positives for "db", "os", etc.
          which are handled by COURSE_ALIASES).
        - Score = 0.5 + (prefix_length_ratio × 0.4)  → range [0.50, 0.90]
          A very short prefix (3 chars / 20 char name = 0.15 ratio) → 0.56
          A long prefix (12 chars / 20 chars = 0.60 ratio)          → 0.74
        - Also checks word-prefix: first word of user input starts the course name
          (handles "software eng" → "software engineering").
        """
        user_lower = user_input.lower().strip()
        inp_len    = len(user_lower)

        if inp_len < 3:
            return 0.0

        # Full string prefix match
        if course_name_lower.startswith(user_lower):
            prefix_ratio = inp_len / max(len(course_name_lower), 1)
            return 0.50 + (prefix_ratio * 0.40)

        # Word-level prefix: user's words each start a word in the course name
        user_words   = user_lower.split()
        course_words = course_name_lower.split()
        if (len(user_words) >= 1 and len(course_words) >= len(user_words)
                and all(cw.startswith(uw)
                        for uw, cw in zip(user_words, course_words))):
            word_ratio = len(user_words) / max(len(course_words), 1)
            return 0.45 + (word_ratio * 0.40)

        return 0.0

    def find_best_match(
        self,
        user_input: str,
        threshold:  float = 0.3,
        top_n:      int   = 5,
    ) -> Optional[Dict]:
        """
        Find the best matching course name.

        Scoring (highest wins):
          1. Exact match             → 1.0
          2. Course-code match       → 0.9 – 1.0
          3. Prefix match            → 0.50 – 0.90   ← NEW (fixes "soft" → "software …")
          4. Sequence + keyword mix  → 0.0 – 1.0

        Args:
            user_input: User's course name/abbreviation
            threshold:  Minimum score to return a result
            top_n:      Internal — candidates to rank before selecting best

        Returns:
            Dict with 'name', 'code', and 'confidence', or None if no match.
        """
        if not self._ensure_initialized():
            return None
        if not user_input or not self._course_cache:
            return None

        user_input     = user_input.strip()
        user_lower     = user_input.lower()
        scored_courses = []

        for course in self._course_cache:
            exact_match   = user_lower == course["name_lower"]
            code_score    = self._code_match_score(user_input, course["code"])
            similarity    = self._similarity_score(user_input, course["name"])
            keyword_score = self._keyword_match_score(user_input, course["name"])
            prefix_score  = self._prefix_match_score(user_input, course["name_lower"])

            if exact_match:
                final_score = 1.0
            elif code_score > 0:
                final_score = code_score
            else:
                seq_kw_score = (similarity * 0.6) + (keyword_score * 0.4)
                # Take the best of sequence/keyword and prefix scoring
                final_score = max(seq_kw_score, prefix_score)

            if final_score >= threshold:
                scored_courses.append({
                    "name":       course["name"],
                    "code":       course["code"],
                    "confidence": final_score,
                    "exact":      exact_match,
                })

        scored_courses.sort(key=lambda x: x["confidence"], reverse=True)

        return scored_courses[0] if scored_courses else None

    def find_ambiguous_matches(
        self,
        user_input:      str,
        threshold:       float = 0.3,
        ambiguity_delta: float = 0.08,
        max_candidates:  int   = 5,
    ) -> List[Dict]:
        """
        Returns a list of candidate courses when the top matches are too close
        in confidence to pick one automatically (i.e. the result is ambiguous).

        Returns an EMPTY list when the best match is clearly dominant.
        """
        if not self._ensure_initialized():
            return []

        all_matches = self.find_all_matches(user_input, threshold=threshold)

        if len(all_matches) < 2:
            return []

        best_score   = all_matches[0]["confidence"]
        second_score = all_matches[1]["confidence"]

        if best_score - second_score > ambiguity_delta:
            return []

        candidates = [
            m for m in all_matches[:max_candidates]
            if best_score - m["confidence"] <= ambiguity_delta
        ]
        return candidates

    def find_all_matches(
        self,
        user_input: str,
        threshold:  float = 0.3,
    ) -> List[Dict]:
        """Find all possible matching courses above threshold."""
        if not self._ensure_initialized():
            return []
        if not user_input or not self._course_cache:
            return []

        user_lower = user_input.lower().strip()
        matches    = []

        for course in self._course_cache:
            similarity    = self._similarity_score(user_input, course["name"])
            keyword_score = self._keyword_match_score(user_input, course["name"])
            code_score    = self._code_match_score(user_input, course["code"])
            prefix_score  = self._prefix_match_score(user_input, course["name_lower"])

            seq_kw_score = (similarity * 0.6) + (keyword_score * 0.4)
            final_score  = max(code_score, seq_kw_score, prefix_score)

            if final_score >= threshold:
                matches.append({
                    "name":       course["name"],
                    "code":       course["code"],
                    "confidence": final_score,
                })

        matches.sort(key=lambda x: x["confidence"], reverse=True)
        return matches

    def get_all_courses(self) -> List[Dict]:
        if not self._ensure_initialized():
            return []
        return (
            [{"name": c["name"], "code": c["code"]} for c in self._course_cache]
            if self._course_cache else []
        )

    def get_course_suggestions(self, partial_input: str, limit: int = 10) -> List[Dict]:
        if not partial_input:
            return self.get_all_courses()[:limit]
        matches = self.find_all_matches(partial_input, threshold=0.2)
        return matches[:limit]

    def close(self):
        if self.driver:
            self.driver.close()
            self.driver = None
            self._initialized = False


# ── Singleton ─────────────────────────────────────────────────────────────────

_mapper_instance = None


def get_course_mapper() -> CourseNameMapper:
    global _mapper_instance
    if _mapper_instance is None:
        _mapper_instance = CourseNameMapper(lazy_load=True)
    return _mapper_instance


def map_course_name(user_input: str, threshold: float = 0.3) -> Optional[str]:
    """
    Convenience function to map a course name.

    RULE: call this only when the input is NOT already in COURSE_ALIASES.
    If the input IS in COURSE_ALIASES, use the alias value directly and skip
    this mapper.  If it is NOT in aliases, always call this mapper.

    Args:
        user_input: User's input (e.g., "soft", "machine learning", "AIM302")
        threshold:  Minimum confidence threshold (default 0.3)

    Returns:
        Normalized course name, or None if no confident match found.
    """
    try:
        mapper = get_course_mapper()
        match  = mapper.find_best_match(user_input, threshold)
        return match["name"] if match else None
    except Exception as e:
        print(f"⚠️  Mapping error: {e}")
        return None


def get_ambiguous_matches(
    user_input:      str,
    threshold:       float = 0.3,
    ambiguity_delta: float = 0.08,
) -> List[Dict]:
    """
    Convenience function: return ambiguous course candidates for *user_input*.

    Returns a list of candidate dicts [{name, code, confidence}] when two or
    more courses score within `ambiguity_delta` of each other, otherwise [].
    """
    try:
        mapper = get_course_mapper()
        return mapper.find_ambiguous_matches(
            user_input,
            threshold=threshold,
            ambiguity_delta=ambiguity_delta,
        )
    except Exception as e:
        print(f"⚠️  Ambiguity check error: {e}")
        return []