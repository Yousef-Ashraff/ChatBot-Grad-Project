"""
Student-related functions that interact with Supabase backend
"""

from chatbot_connector import ChatbotConnector
from eligibility import TRACK_MAP as _TRACK_MAP
import os

def get_student_details(student_id = os.getenv("STUDENT_ID")):
    """
    Get student details from Supabase.

    Returns a dict with all relevant student columns, or None if not found.
    """
    try:
        connector = ChatbotConnector()
        student_data = connector.get_student_data(student_id)

        if not student_data:
            return None

        courses_degrees = student_data.get("courses_degrees") or []

        return {
            "student_id":        student_id,
            # ── Identity ──────────────────────────────────────────────────
            "full_name":         student_data.get("full_name", ""),
            "track":             _TRACK_MAP.get(
                                     (student_data.get("track") or "").strip().upper(),
                                     student_data.get("track", "")
                                 ),
            "university_year":   student_data.get("university_year"),
            # ── Academic progress ─────────────────────────────────────────
            "gpa":               student_data.get("gpa", 0.0),
            "total_hours_earned": student_data.get("total_hours_earned", 0),
            "courses_degrees":   courses_degrees,
            "completed_courses": [c["name"] for c in courses_degrees if isinstance(c, dict) and "name" in c],
            "semester_gpas":     student_data.get("semester_gpas") or {},
            # ── LinkedIn ──────────────────────────────────────────────────
            "linkedin":          student_data.get("linkedin", ""),
            "linkedin_summary":  student_data.get("linkedin_summary", ""),
        }

    except Exception as e:
        print(f"Error fetching student details: {e}")
        return None