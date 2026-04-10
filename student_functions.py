"""
Student-related functions that interact with Supabase backend
"""

from chatbot_connector import ChatbotConnector
from eligibility import TRACK_MAP as _TRACK_MAP
import os

def get_student_details(student_id = os.getenv("STUDENT_ID")):
    """
    Get student's academic details from Supabase.
    
    Args:
        student_id: The unique student ID
        
    Returns:
        dict: Academic details containing:
            - gpa: float
            - earned_credits: int
            - completed_courses: list of course names
        Returns None if student not found or has no academic details
    """
    try:
        connector = ChatbotConnector()

        # Fetch both the student profile row AND the academic_details sub-object
        student_data     = connector.get_student_data(student_id)
        academic_details = connector.get_or_initialize_academic_details(student_id)

        if academic_details:
            return {
                "student_id":        student_id,
                # ── Identity fields (name, track) ─────────────────────────
                "first_name":        student_data.get("first_name", "")        if student_data else "",
                "last_name":         student_data.get("last_name",  "")        if student_data else "",
                "track":             _TRACK_MAP.get((student_data.get("track") or "").strip().upper(),
                                               student_data.get("track", "")) if student_data else "",
                "university_year":   student_data.get("university_year", None) if student_data else None,
                # ── Academic progress fields ──────────────────────────────
                "gpa":               academic_details.get("gpa",               0.0),
                "earned_credits":    academic_details.get("earned_credits",    0),
                "completed_courses": academic_details.get("completed_courses", []),
            }

        return None
        
    except Exception as e:
        print(f"Error fetching student details: {e}")
        return None