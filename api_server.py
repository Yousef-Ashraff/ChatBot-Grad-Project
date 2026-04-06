"""
api_server.py — FastAPI HTTP Server for the BNU Academic Advisor Chatbot
=========================================================================

Exposes chatbot_api.py as REST endpoints for the mobile app.

Run with:
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

Install:
    pip install fastapi uvicorn langchain-groq langgraph langchain-core

Endpoints
─────────
  POST /chat                    — Student sends a message, gets AI response
  GET  /student/{student_id}    — Fetch student profile (call once on login)
  POST /clear-history           — Clear history + cancel planning (on logout)
  POST /disambiguation          — Autocomplete for ambiguous course names
  GET  /health                  — Health check
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import chatbot_api

load_dotenv()

app = FastAPI(
    title="BNU Academic Advisor API",
    description=(
        "AI Academic Advisor powered by a LangGraph ReAct Agent for BNU students."
    ),
    version="2.0.0",
)

# Allow requests from the mobile app — tighten origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / response models ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    student_id: str
    message: str

class StudentRequest(BaseModel):
    student_id: str

class DisambiguationRequest(BaseModel):
    student_id: str
    term: str       # Partial course name typed by the student


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.post("/chat")
def chat(req: ChatRequest):
    """
    Send a student message and receive the AI advisor's response.

    The LangGraph ReAct agent reasons over the question, invokes the
    appropriate tools (course DB, bylaws KB, student profile, …), and
    returns a grounded, context-aware answer in natural language.

    Body:    {"student_id": "22030094", "message": "Can I take ML?"}
    Returns: {"ok": true, "student_id": "...", "response": "..."}
    """
    result = chatbot_api.chat(req.student_id, req.message)
    if not result["ok"]:
        raise HTTPException(status_code=500, detail=result.get("error"))
    return result


@app.get("/student/{student_id}")
def get_student_info(student_id: str):
    """
    Fetch student profile and academic details from Supabase.
    Call this once when the student logs in to populate the mobile UI.

    Returns:
        {ok, student_id, first_name, last_name, track, university_year,
         gpa, earned_credits, completed_courses}
    """
    result = chatbot_api.get_student_info(student_id)
    if not result["ok"]:
        raise HTTPException(status_code=404, detail=result.get("error"))
    return result


@app.post("/clear-history")
def clear_history(req: StudentRequest):
    """
    Clear chat history and cancel any active planning session.
    Call on logout or when starting a fresh conversation.

    Body:    {"student_id": "22030094"}
    Returns: {"ok": true, "student_id": "..."}
    """
    result = chatbot_api.clear_history(req.student_id)
    if not result["ok"]:
        raise HTTPException(status_code=500, detail=result.get("error"))
    return result


@app.post("/disambiguation")
def get_disambiguation_options(req: DisambiguationRequest):
    """
    Return candidate course names when the student typed an ambiguous term.
    Use this to power autocomplete or a picker in the mobile UI.

    Body:    {"student_id": "22030094", "term": "soft"}
    Returns: {"ok": true, "candidates": [{"name": "...", "confidence": 0.9}]}
    """
    result = chatbot_api.get_disambiguation_options(req.student_id, req.term)
    if not result["ok"]:
        raise HTTPException(status_code=500, detail=result.get("error"))
    return result


@app.get("/health")
def health():
    """Simple health check."""
    return {"status": "ok", "version": "2.0.0"}
