# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BNU Academic Advisor AI Chatbot — a LangGraph-based agent that provides academic guidance to university students via a FastAPI REST API consumed by a mobile app. Integrates Neo4j (course knowledge graph), Supabase (student data + chat history), Pinecone (bylaws RAG), and Groq LLM.

## Commands

```bash
# Install dependencies (uv manages the venv automatically)
uv sync

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Run the API server
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

# Quick local test (set STUDENT_ID in .env first)
python -c "from chatbot_api import chat; print(chat('22030158', 'What courses can I take?'))"
```

No test suite or linter is configured.

## Architecture

### Request Flow

```
Mobile App → POST /chat (api_server.py)
  → chatbot_api.chat(student_id, message)
    → chatbot_connector: load Supabase chat history
    → preprocessor.preprocess(message)       # normalize before agent sees it
    → [active planning session?]
        YES → PlanningOrchestrator.advance()
        NO  → agent.run()
    → chatbot_connector: save to Supabase
```

### LangGraph Judging Loop (`agent.py`)

Not a standard ReAct loop — uses a custom **multi-turn judging loop**:

```
[agent node] → tool calls (max 3/round)
[tools node] → execute tools
[collect node] → append to accumulated_context
[judge node] → LLM evaluates: "is the context sufficient?"
  satisfied=True       → [answer node] → END
  tool_calls < MAX     → loop back to [agent]
  reformulations < MAX → [reformulate node] → [agent]
  else                 → [clarify node] → END
```

Key state fields in `AgentState` (TypedDict):
- `messages` — full LangChain message history
- `accumulated_context` — all tool results (append-only, never discarded)
- `called_tools` — dedup cache (`"tool_name|args"`)
- `original_query` / `current_query` — original is immutable; current may be reformulated
- Loop counters: `tool_calls_this_round` (max 3), `query_reformulations` (max 3)

### Preprocessing Pipeline (`preprocessor.py`)

Runs **before** the agent on every query. Five steps:
1. **Reference resolution** — replace pronouns using LLM + chat history
2. **Entity extraction** — extract course/track names via LLM
3. **Course name mapping** — fuzzy match against `COURSE_ALIASES` + Neo4j; single match → substitute silently; multiple → trigger ambiguity session
4. **Track name mapping** — same for program names
5. **Query rewriting** — LLM substitutes resolved names back naturally

Returns `PreprocessResult` with status: `"ready"` | `"ambiguous"` | `"passthrough"`

### Tools (`tools.py`)

All tools are LangChain `@tool` functions. Student ID is injected via module-level `_ACTIVE_STUDENT_ID` (set by `agent.run()` before each call).

| Category | Tool |
|---|---|
| Student | `get_student_info()` |
| Courses | `get_course_info()`, `get_course_prerequisites()` |
| Eligibility | `check_course_eligibility()` |
| Planning | `start_course_planning()` |
| Programs | `get_program_requirements()`, `get_program_courses()` |
| Bylaws RAG | `query_academic_bylaws()` |

### Course Planning (`planning.py` + `planning_service.py`)

Interactive planning algorithm (ported from a Jupyter notebook) runs in a **daemon background thread**. `PlanningOrchestrator` intercepts `print()`/`input()` via custom I/O routers and drives the session turn-by-turn. Active sessions stored in `chatbot_api._planning_sessions[student_id]`.

### Data Sources

| Source | Purpose | Connection |
|---|---|---|
| **Neo4j** | Course graph (prereqs, programs, dependencies) | `NEO4J_URI` in `.env` |
| **Supabase** | Student profiles, chat history | `SUPABASE_URL` / `SUPABASE_KEY` |
| **Pinecone** | Academic bylaws vector index (`bnu-bylaws`) | `PINECONE_API_KEY` |
| **HuggingFace** | Embeddings: `intfloat/multilingual-e5-large` | `HF_API_KEY` |

### LLM: Groq with 5-Key Fallback (`llm_client.py`)

Supports `GROQ_API_KEY` through `GROQ_API_KEY5`. When one key hits a rate limit, `_call_groq()` auto-rotates to the next available key. Two model slots configured via env:
- `GROQ_MODEL_AGENT` — used by the agent node
- `GROQ_MODEL_XXX` — used by judge, preprocessor, and query engine nodes

## Key Global State Patterns

- `tools._ACTIVE_STUDENT_ID` — set before each `agent.run()` call; not thread-safe across concurrent students
- `chatbot_api._planning_sessions` — active planning sessions by student_id
- `chatbot_api._ambiguity_sessions` — pending course disambiguation by student_id
- `eligibility._supabase` — lazy singleton Supabase client

## API Endpoints (`api_server.py`)

- `POST /chat` — main chat endpoint
- `GET /student/{student_id}` — fetch student profile
- `POST /clear-history` — clear session and cancel active planning
- `POST /disambiguation` — course name autocomplete
- `GET /health` — health check

## Environment Variables (`.env`)

Required keys: `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `GROQ_API_KEY` (+ optionally `GROQ_API_KEY2`–`5`), `SUPABASE_URL`, `SUPABASE_KEY`, `PINECONE_API_KEY`, `PINECONE_INDEX`, `HF_API_KEY`, `GOOGLE_SHEET_ID`.

Debug flags: `DEBUG_MODE`, `DEBUG_TO_FILE`, `DEBUG_LEVEL` — set `DEBUG_MODE=true` to see preprocessor and agent reasoning via `debug_box.py`.

`service-account.json` must be present for Google Sheets logging (`google_sheets_logger.py`).
