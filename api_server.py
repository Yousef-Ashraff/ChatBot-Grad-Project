"""
api_server.py — FastAPI HTTP Server for the BNU Academic Advisor Chatbot
=========================================================================

Endpoints
─────────
  POST /chat                    — Student sends a message, gets AI response
  GET  /student/{student_id}    — Fetch student profile (call once on login)
  POST /clear-history           — Clear history + cancel planning (on logout)
  POST /clear-lecture           — Clear any active lecture/reading session
  POST /disambiguation          — Autocomplete for ambiguous course names
  GET  /health                  — Health check
"""

import traceback
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import chatbot_api

load_dotenv()

logger = logging.getLogger(__name__)

app = FastAPI(
    title="BNU Academic Advisor API",
    description="AI Academic Advisor powered by a LangGraph ReAct Agent for BNU students.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global error handler — prints full traceback for every 500 ────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    logger.error("Unhandled exception on %s %s:\n%s", request.method, request.url.path, tb)
    print(f"\n❌ UNHANDLED ERROR on {request.method} {request.url.path}:\n{tb}")
    return JSONResponse(
        status_code=500,
        content={"ok": False, "error": str(exc), "trace": tb},
    )


# ── Request / response models ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    student_id: str
    message: str

class StudentRequest(BaseModel):
    student_id: str

class DisambiguationRequest(BaseModel):
    student_id: str
    term: str

class SetLectureRequest(BaseModel):
    student_id: str
    lecture_pdf: str   # base64-encoded PDF bytes
    lecture_name: str = ""


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.post("/chat")
def chat(req: ChatRequest):
    """
    Send a student message and receive the AI advisor's response.

    Body:    {"student_id": "22030094", "message": "Can I take ML?"}
    Returns: {"ok": true, "student_id": "...", "response": "..."}
    """
    result = chatbot_api.chat(req.student_id, req.message)
    if not result["ok"]:
        error_detail = result.get("error", "Unknown error")
        trace        = result.get("trace", "")
        # Always print to console so you can see it in the uvicorn terminal
        print(f"\n❌ /chat error for student {req.student_id}:\n{error_detail}")
        if trace:
            print(trace)
        raise HTTPException(status_code=500, detail=error_detail)
    return result


@app.get("/student/{student_id}")
def get_student_info(student_id: str):
    """
    Fetch student profile and academic details from Supabase.
    Call this once when the student logs in to populate the mobile UI.
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
        error_detail = result.get("error", "Unknown error")
        print(f"\n❌ /clear-history error for student {req.student_id}: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)
    return result


@app.post("/set-lecture")
def set_lecture(req: SetLectureRequest):
    """
    Receive a base64-encoded PDF lecture from the mobile app, extract its
    text, and store it in memory so /chat answers questions about it.

    Body:    {"student_id": "22030094", "lecture_pdf": "<base64>", "lecture_name": "Week 3"}
    Returns: {"ok": true, "student_id": "...", "chars": <int>}
    """
    import lecture_service
    print(f"\n📚 /set-lecture — student={req.student_id!r} name={req.lecture_name!r} "
          f"b64_len={len(req.lecture_pdf)}")
    try:
        text = lecture_service.set_lecture(
            req.student_id, req.lecture_pdf, req.lecture_name
        )
        print(f"✅ /set-lecture — stored {len(text)} chars for {req.student_id!r}")
        return {"ok": True, "student_id": req.student_id, "chars": len(text)}
    except RuntimeError as exc:
        print(f"❌ /set-lecture FAILED for {req.student_id!r}: {exc}")
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/clear-lecture")
def clear_lecture(req: StudentRequest):
    """
    Clear any active lecture/reading session for the student.

    Body:    {"student_id": "22030094"}
    Returns: {"ok": true, "student_id": "..."}
    """
    import lecture_service
    lecture_service.clear_lecture(req.student_id)
    return {"ok": True, "student_id": req.student_id}


@app.post("/disambiguation")
def get_disambiguation_options(req: DisambiguationRequest):
    """
    Return candidate course names when the student typed an ambiguous term.

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