"""
preference_service.py — Student preference storage layer
=========================================================
Manages the ai_preference column in the student_preferences Supabase table.

Table schema expected in Supabase:
    student_preferences (
        student_id        text PRIMARY KEY,
        ai_preference     jsonb DEFAULT '{}',
        user_preference   jsonb DEFAULT '{}',
        degree_preference jsonb DEFAULT '{}',
        updated_at        timestamptz DEFAULT now()
    )
"""

from __future__ import annotations

import os
from typing import Dict

from dotenv import load_dotenv

load_dotenv()

# ── Valid category taxonomy ───────────────────────────────────────────────────

VALID_CATEGORIES: set[str] = {
    "math",                  # calculus, linear algebra, discrete math
    "probability_statistics",# probability, stats, stochastic processes
    "programming",           # coding, algorithms, data structures
    "software_engineering",  # design patterns, web/mobile dev, SDLC
    "ai_ml",                 # machine learning, deep learning, AI
    "data_management",       # databases, SQL, data warehousing
    "data_analysis",         # analytics, BI, visualization
    "theory",                # automata, complexity, formal methods
    "networking_systems",    # networks, OS, security, infrastructure
    "visual_computing",      # image processing, graphics, geometry
    "language_text",         # NLP, linguistics, text processing
    "optimization",          # operations research, numerical optimization
}


# ── Supabase client (lazy singleton) ─────────────────────────────────────────

_client = None


def _get_client():
    global _client
    if _client is None:
        from supabase import create_client
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
        _client = create_client(url, key)
    return _client


# ── Public API ────────────────────────────────────────────────────────────────

def get_preferences(student_id: str) -> Dict:
    """Return all three preference dicts for a student."""
    client = _get_client()
    resp = (
        client.table("student_preferences")
        .select("ai_preference, user_preference, degree_preference")
        .eq("student_id", student_id)
        .execute()
    )
    if resp.data:
        import json as _j

        def _parse(val):
            if isinstance(val, str):
                try:
                    val = _j.loads(val)
                except Exception:
                    return {}
            return val if isinstance(val, dict) else {}

        row = resp.data[0]
        return {
            "ai_preference":     _parse(row.get("ai_preference")),
            "user_preference":   _parse(row.get("user_preference")),
            "degree_preference": _parse(row.get("degree_preference")),
        }
    return {"ai_preference": {}, "user_preference": {}, "degree_preference": {}}


def update_ai_preference(student_id: str, deltas: Dict[str, float]) -> Dict[str, float]:
    """
    Merge delta scores into the student's ai_preference.

    - Ignores any key not in VALID_CATEGORIES.
    - Clamps each resulting score to [0.0, 1.0].
    - Creates the row if it does not exist (upsert).

    Returns the full updated ai_preference dict.
    """
    valid_deltas = {k: float(v) for k, v in deltas.items() if k in VALID_CATEGORIES}
    if not valid_deltas:
        return {}

    client = _get_client()

    # Read current scores
    resp = (
        client.table("student_preferences")
        .select("ai_preference")
        .eq("student_id", student_id)
        .execute()
    )
    current: Dict[str, float] = {}
    if resp.data:
        raw = resp.data[0].get("ai_preference") or {}
        if isinstance(raw, str):
            import json as _j
            try:
                raw = _j.loads(raw)
            except Exception:
                raw = {}
        current = raw if isinstance(raw, dict) else {}

    # Merge
    updated = dict(current)
    for cat, delta in valid_deltas.items():
        updated[cat] = round(max(0.0, min(1.0, updated.get(cat, 0.0) + delta)), 3)

    # Upsert — creates row on first call, updates on subsequent calls
    client.table("student_preferences").upsert(
        {"student_id": student_id, "ai_preference": updated},
        on_conflict="student_id",
    ).execute()

    return updated
