# CLAUDE.md — BNU Academic Advisor Chatbot

## Project Overview

AI-powered academic advisor chatbot for the Faculty of Computer Science, Benha University (BNU = Badr University in Cairo). Answers student questions about courses, prerequisites, eligibility, curriculum planning, electives, and university bylaws. Deployed as a REST API consumed by a mobile app.

## Commands

```bash
# Install dependencies
uv sync

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Run the API server
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

# CLI dev mode
python agent.py --student 22030094 --verbose

# Smoke test
python agent.py --test

# Quick API test
python chatbot_api.py

# Direct chat test
python -c "from chatbot_api import chat; print(chat('22030158', 'What courses can I take?'))"
```

No test suite or linter is configured.

## File Structure

```
project/
├── api_server.py              # FastAPI HTTP server (port 8000)
├── chatbot_api.py             # Main orchestration layer + session management
├── agent.py                   # LangGraph judging-loop agent
├── tools.py                   # 12 LangChain tool definitions
├── preprocessor.py            # Query preprocessing pipeline (5 steps)
├── course_name_mapper.py      # Fuzzy course name → Neo4j canonical name
├── neo4j_course_functions.py  # All Neo4j/KG query functions
├── eligibility.py             # Prerequisite + credit-hour eligibility checker
├── rag_service.py             # RAG pipeline (Pinecone + HF + Groq)
├── planning.py                # Interactive course planning function
├── planning_service.py        # Thread bridge wrapping planning() for chatbot
├── chatbot_connector.py       # Supabase client (history + student data)
├── student_functions.py       # Student profile query helper
├── llm_client.py              # Groq LLM client with 5-key fallback
├── debug_box.py               # Unicode box printer for debug output
├── google_sheets_logger.py    # Optional conversation logging to Sheets
└── .env                       # All credentials and config
```

## Architecture

### High-Level

```
[Mobile App / Student]
        │
        ▼
[api_server.py]          ← FastAPI HTTP server (uvicorn)
        │
        ▼
[chatbot_api.py]         ← Main orchestration layer (routing, sessions, patching)
        │
   ┌────┴───────────────────────────┐
   │                                │
   ▼                                ▼
[preprocessor.py]          [planning_service.py]
(query cleaning)           (multi-turn planning sessions)
   │
   ▼
[agent.py]               ← LangGraph ReAct agent with Judging Loop
   │
   ▼
[tools.py]               ← 12 LangChain tools
   │
   ├─► [neo4j_course_functions.py]  ← Knowledge Graph (Neo4j Aura)
   ├─► [eligibility.py]            ← Prerequisite + credit-hour checks (Supabase)
   ├─► [rag_service.py]            ← RAG pipeline (Pinecone + HuggingFace + Groq)
   ├─► [student_functions.py]      ← Student profile queries (Supabase)
   └─► [planning_service.py]       ← Interactive multi-turn course planner
```

### Request Flow (End-to-End)

```
Student sends message → POST /chat
    │
    ▼
chatbot_api.chat(student_id, message)
    │
    ├─ Load chat history from Supabase (last 3 user + 3 assistant = 6 messages)
    │
    ├─ _route_message():
    │      Priority 1: Pending ambiguity session? → _resolve_ambiguity_reply()
    │      Priority 2: Active planning session?   → _advance_planning()
    │      Priority 3: Normal query               → _preprocess_and_run()
    │
    │   [Normal path: _preprocess_and_run()]
    │      │
    │      ▼
    │   preprocessor.process(message, history)   ← 5-step pipeline
    │      │
    │      ▼
    │   Returns PreprocessResult:
    │     status="ready"      → clean_query + resolved_courses
    │     status="ambiguous"  → clarification string → returned to student
    │     status="passthrough"→ query as-is → agent.run()
    │
    ▼
_analyze_and_split(clean_query)   ← LLM decomposes multi-requirement queries
    │
    ├─ 1 sub-query  → BNUAdvisorAgent.run(query, history)
    │
    └─ N sub-queries → _split_and_run():
           for each sub-query:
             BNUAdvisorAgent.run_and_get_context() → accumulated_context list
           combine all contexts → single llm_call_text() synthesis
           (if start_course_planning triggered → surface planning output directly)
    │
    ▼
Final answer string
    │
    ▼
chatbot_api persists to Supabase (user msg + assistant response)
    │
    ▼
Returns {"ok": true, "response": "..."}
```

### LangGraph Judging Loop (`agent.py`)

Not a standard ReAct loop — uses a custom multi-turn judging loop:

```
START → agent → dedup → tools → collect → judge
            ↑                              │
            │          ┌──────── satisfied=true → answer → END
            │          │
            │          ├──── tool_calls < 3 → agent (next tool)
            │          │
            │          └──── tool_calls ≥ 3:
            │                 reformulations < 3 → reformulate → agent
            │                 reformulations ≥ 3 → clarify → END
```

**Routing logic:**
- after agent: tool_calls → dedup; no tool_calls → answer
- after dedup: any SKIPPED → collect (bypass tools); all new → tools
- after tools: → collect (always)
- after collect: → judge (always)
- after judge: satisfied=true → answer; satisfied=false → see counters above

**Limits:** `MAX_TOOL_CALLS_PER_ROUND = 3`, `MAX_REFORMULATIONS = 3`

**`AgentState` TypedDict fields:**
- `messages` — full LangChain message history (add_messages reducer)
- `accumulated_context` — all tool result strings (append-only, never discarded)
- `called_tools` — dedup signatures `"tool_name|sorted_args_json"` (append-only)
- `previous_reformulations` — prior query reformulations tried
- `tool_calls_this_round` — counter of tool rounds done (int, replace)
- `query_reformulations` — total reformulations attempted (int, replace)
- `original_query` — never changes after init
- `current_query` — may be replaced by reformulate node
- `satisfied` — set by judge node (bool)
- `judge_missing`, `judge_tools_this_round` — judge metadata

**Graph nodes:**
- **agent** — selects next tool. Uses `GROQ_MODEL_AGENT`. Retries on `tool_use_failed` (Groq 400).
- **dedup** — injects fake `[SKIPPED]` ToolMessage for duplicate calls.
- **tools** — `ToolNode(ALL_TOOLS)`, executes tool calls.
- **collect** — appends ToolMessage content to `accumulated_context`, increments round counter.
- **judge** — LLM evaluates if context fully answers `original_query`. If `start_course_planning` was called → always `satisfied=true`.
- **answer** — generates final answer from `accumulated_context`. If planning tool was called → relays planning output directly.
- **reformulate** — generates new `current_query` with a different angle (7 fixed angles in order). Resets `tool_calls_this_round` to 0.
- **clarify** — polite last-resort clarification request after all reformulations exhausted.

### Preprocessing Pipeline (`preprocessor.py`)

Runs before every agent invocation. Five steps:

1. **Reference resolution** — LLM resolves pronouns/vague refs using conversation history ("what about it?" → "what about machine learning?")
2. **Entity extraction** — LLM extracts course names and track/program names as JSON
3. **Course name mapping** — check `COURSE_ALIASES` first, then fuzzy match Neo4j. Single winner → auto-substitute; multiple close matches → `"ambiguous"` → store `PendingAmbiguity`, return clarification question. If more courses remain after the ambiguous one, they are saved in `PendingAmbiguity.pending_courses` and processed after disambiguation (chained ambiguity support).
4. **Track name mapping** — check `TRACK_ALIASES`, then fuzzy match programs. Always auto-resolves.
5. **Query rewriting** — substitute canonical names into resolved query (regex first; LLM fallback triggered when canonical name is absent from result — handles implied terms like "training 1 and 2")

**Fuzzy scoring:** `score = max(code_exact→1.0, similarity*0.6 + keyword_overlap*0.4, prefix_score)`; threshold=0.30; ambiguity_delta=0.08 (two candidates within 0.08 → ask user)

**COURSE_ALIASES examples:** `"ml"→"machine learning"`, `"os"→"operating systems"`, `"oop"→"object oriented programming"`, `"dsa"→"data structures and algorithms"`, course codes like `"bcs311"→"artificial intelligence"`

**TRACK_ALIASES:** `"aim"/"ai"/"aiml"→"artificial intelligence and machine learning"`, `"sad"/"software"/"sw"→"software and application development"`, `"das"/"ds"/"data science"→"data science"`

**ENTITY_BLOCKLIST:** words filtered after extraction to prevent false matches: `courses`, `electives`, `semester`, `year`, `prerequisites`, `credits`, `hours`, `information`, etc.

**PendingAmbiguity** (stored in `_ambiguity_sessions[student_id]`): `original_query`, `dereferenced`, `ambiguous_term`, `candidates [{name,code,confidence}]`, `resolved_courses`, `pending_courses` (remaining unprocessed courses), `resolved_tracks`, `history`

**PreprocessResult** fields: `status`, `clean_query`, `clarification`, `pending`, `resolved_courses: Dict[str,str]` (original→canonical for every course mapped in this run)

**Chained ambiguity flow:** When multiple courses are ambiguous, the preprocessor asks about them one at a time. After each disambiguation reply, `resolve_ambiguity()` processes remaining `pending_courses`; if another is ambiguous it returns `status="ambiguous"` again. `_resolve_ambiguity_reply()` in `chatbot_api.py` detects this and stores the new pending instead of running the agent.

## Tools (`tools.py`) — 12 Tools

Student ID injected via module-level `_ACTIVE_STUDENT_ID` (set by `set_active_student_id()` before each graph run). All tools return plain strings. Course name fuzzy matching applied automatically via `_normalize_course()`.

| Tool | When to use |
|---|---|
| `get_student_info` | "What's my GPA?", "What have I completed?" |
| `get_course_info` | "What is ML about?", "How many credits is SE?" |
| `get_course_prerequisites` | "What does X close?", "What are prereqs for X?", "What does X unlock?" |
| `get_course_timing` | "When is ML taught?", "Which semester is OS?" |
| `check_course_eligibility` | "Can I take ML?", "Am I eligible for OS?" |
| `get_courses_by_term` | "What's in year 2 semester 1?" |
| `get_courses_by_multiple_terms` | "What do I study in years 2 and 3?" |
| `get_all_electives` | "What electives are in the AI track?" |
| `get_elective_slots` | "When can I take electives?", "How many elective slots?" |
| `filter_courses` | "Show all 3-credit courses in AIM", "What core courses in SAD?" |
| `answer_academic_question` | "What's the minimum GPA?", "Graduation requirements?" — RAG over bylaws |
| `start_course_planning` | "Make a plan for me", "What should I take next semester?" |

## Data Sources

### Neo4j Knowledge Graph

```
Nodes:    (Course) — name, code, description, credit_hours
          (Program) — name
          (GraphEmbedding) — embedding [indexed]

Rels:     (Course)-[:BELONGS_TO]->(Program)        rel props: elective='yes'/'no', year_level, semester
          (Course)-[:HAS_PREREQUISITE]->(Course)
          (Course)-[:PREREQUISITE_OF]->(Course)
          (Course)-[:SIMILAR_TO]->(Course)
```

**Programs (3 tracks):** `artificial intelligence and machine learning` (AIM), `software and application development` (SAD), `data science` (DAS)

**Year levels:** `First Year`, `Second Year`, `Third Year`, `Fourth Year`
**Semesters:** `First`, `Second`

### Supabase (PostgreSQL) — `students` table

| Column | Type | Notes |
|---|---|---|
| `student_id` | string | unique |
| `first_name`, `last_name` | string | |
| `track` | string | `"AIM"`, `"SAD"`, or `"DS"` |
| `university_year` | int | 1–4 |
| `gpa` | float | |
| `total_hours_earned` | int | **authoritative** — set by registrar, never recomputed |
| `courses_degrees` | JSON array | `[{"name": "machine learning", "credit_hours": 3}, ...]` |
| `chat_history` | JSON | `{"conversation_id": "uuid", "chat_history": [{role, content, timestamp}]}` |
| `academic_details` | JSON | legacy, kept for compatibility |

**TRACK_MAP:** `"SAD"→"software and application development"`, `"AIM"→"artificial intelligence and machine learning"`, `"DS"→"data science"`

**Chat history window:** last 3 user + last 3 assistant messages (6 total)

### Pinecone (RAG)

- Index: `bnu-bylaws`, top_k=4, min_score=0.30
- Metadata per vector: `topic`, `text`
- Used for: GPA policy, attendance, graduation requirements, academic probation, withdrawal, credit transfer, disciplinary policies

## LLM Client (`llm_client.py`)

**5-key fallback:** `GROQ_API_KEY` → `GROQ_API_KEY2` → ... → `GROQ_API_KEY5`. On rate limit (429) → rotate to next key. Raises `RuntimeError` only if ALL keys fail.

**Model split:**
- `GROQ_MODEL_AGENT` — agent node (tool selection + answer synthesis). Default: `openai/gpt-oss-120b`
- `GROQ_MODEL_XXX` — utility tasks (preprocessing, judging, reformulation, RAG generation). Default: `meta-llama/llama-4-scout-17b-16e-instruct`

**Public API:** `llm_call()`, `llm_call_json()`, `llm_call_text()`, `llm_call_stream()`, `llm_call_stream_text()`

## Course Planning (`planning.py` + `planning_service.py`)

`planning()` is an interactive function using `print()`/`input()` (originally from a notebook). `PlanningOrchestrator` wraps it for chatbot use via a thread bridge:

- `planning()` runs in a **daemon thread**
- `_ThreadRouter` replaces `sys.stdout` — captures thread output into per-thread StringIO buffers
- `_smart_input` replaces `builtins.input` — flushes output to `_out_q`, blocks on `_in_q` for student reply
- `chatbot_api` calls `advance(state, reply)` per student message turn
- Timeout: 90 seconds per planning event

**Session lifecycle:**
1. `start_course_planning` tool called → `_start_and_cache_planning(student_id)` (monkey-patched in `chatbot_api.py`)
2. `PlanningOrchestrator.start()` → first prompt + `PlanningState`
3. State stored in `_planning_sessions[student_id]`
4. Each subsequent student message → `PlanningOrchestrator.advance(state, reply)`
5. On `PlanStep.COMPLETE` → delete from `_planning_sessions`
6. On `clear_history()` → also clears `_planning_sessions[student_id]`

## Multi-Requirement Query Splitting (`chatbot_api.py`)

Before every agent run, `_analyze_and_split(clean_query)` is called to detect whether the query asks about multiple independent topics.

**`_analyze_and_split(clean_query) → List[str]`**
- Calls the utility LLM with a structured prompt covering all combinatorial cases:
  - N courses → N sub-queries
  - 1 course × M programs → M sub-queries
  - N courses × M programs → N×M sub-queries (cartesian product)
  - Independent mixed questions (course + bylaw) → one each
  - Already atomic → `[clean_query]` unchanged
- Returns a list of sub-query strings. Single-item list → no split.

**`_split_and_run(student_id, clean_query, sub_queries, history, verbose)`**
- For each sub-query: calls `BNUAdvisorAgent.run_and_get_context()` to collect tool results without generating a per-sub-query answer.
- Combines all `accumulated_context` lists.
- **Planning special case:** if `start_course_planning` was triggered in any sub-query (`STUDENT COURSE PLANNING SYSTEM` in context), the planning output is returned directly (so the interactive session continues). Any non-planning context is synthesised and prepended.
- Otherwise: single `llm_call_text()` synthesis from all combined context.

**`BNUAdvisorAgent.run_and_get_context(query, history, verbose)`** (in `agent.py`)
- Runs the full judging loop graph (agent → tools → judge) for one sub-query.
- Returns `accumulated_context: List[str]` (raw tool result strings) without calling the answer node.
- When `verbose=True`: streams with debug boxes for all nodes **except** `answer`/`clarify` (those are suppressed per sub-query; one combined answer box is shown at the end by `_split_and_run`).

**CLI loop** (`agent.py __main__`) also calls `_analyze_and_split` / `_split_and_run` via the already-imported `chatbot_api` module, so CLI and API entry points behave identically.

## Session Management (in-memory, `chatbot_api.py`)

```python
_planning_sessions:   Dict[str, PlanningState]    # active multi-turn planning
_ambiguity_sessions:  Dict[str, PendingAmbiguity] # waiting for disambiguation reply
```

Routing priority in `_route_message()`:
1. `_ambiguity_sessions[student_id]` → `_resolve_ambiguity_reply()`
2. `_planning_sessions[student_id]` → `_advance_planning()`
3. Neither → `_preprocess_and_run()`

`_resolve_ambiguity_reply()` now checks `result.status == "ambiguous"` before proceeding — if `resolve_ambiguity()` found another chained ambiguous course, it stores the new `PendingAmbiguity` and returns the next clarification question instead of running the agent.

## API Endpoints (`api_server.py`)

Port 8000, CORS open (`"*"`).

| Endpoint | Method | Body | Description |
|---|---|---|---|
| `/chat` | POST | `{student_id, message}` | Main chat |
| `/student/{student_id}` | GET | — | Student profile for mobile login |
| `/clear-history` | POST | `{student_id}` | Clear Supabase history + sessions |
| `/disambiguation` | POST | `{student_id, term}` | Course name autocomplete → `{ok, candidates}` |
| `/health` | GET | — | `{status: "ok", version: "2.0.0"}` |

## Eligibility Checker (`eligibility.py`)

`check_course_eligibility(student_id, course_name)`:
1. Single Supabase query → student row
2. Build `completed_courses` from `courses_degrees[].name` (lowercased)
3. Convert track code → canonical program name via `TRACK_MAP`
4. Guard: `course_belongs_to_program()` via Neo4j
5. `get_course_dependencies(course_name, program_name)` from Neo4j
6. If no prerequisites → immediately eligible
7. For each prerequisite: if `"Required_Credit_Hours"` key → credit-hour gate; else → must be in `completed_courses`

`get_student_context(student_id)` — shared helper returning `{completed_courses, program_name, total_hours_earned, university_year, current_term, gpa, first_name, last_name}`

## RAG Service (`rag_service.py`)

1. Embed question via HuggingFace `intfloat/multilingual-e5-large` (adds `"query: "` prefix)
2. Query Pinecone (top_k=4, min_score=0.30)
3. Format chunks as context (topic + text, separated by `---`)
4. Include last 4 conversation turns as `history_block`
5. Call Groq LLM — answer ONLY from bylaw context, 3–6 sentences or bullet list, no article numbers

## Chat History (`chatbot_connector.py`)

- `get_chat_history(student_id)` → `{"conversation_id": "uuid", "chat_history": [...]}`
- `add_message(student_id, role, content)` — keeps rolling window: last 3 user + last 3 assistant messages
- Optionally logs to Google Sheets (buffers user msg, logs complete turn on assistant reply)
- `clear_chat_history(student_id)` → new `conversation_id`, clears messages

## Course Name Mapper (`course_name_mapper.py`)

Lazy-initialized class `CourseNameMapper`:
- Loads all `(Course)` nodes from Neo4j at init (cached)
- Scores each course: exact match (1.0), code match (0.9–1.0), prefix score (0.50–0.90), sequence+keyword (0.0–1.0)
- `find_best_match(user_input)` → single best match or None
- `find_ambiguous_matches(user_input)` → candidates when top scores within `ambiguity_delta=0.08`
- Module-level singleton via `get_course_mapper()`
- Convenience: `map_course_name(user_input)`, `get_ambiguous_matches(user_input)`

**RULE:** Always check `COURSE_ALIASES` first. Only call mapper if input is NOT in `COURSE_ALIASES`.

## Debug System (`debug_box.py`)

Unicode box-drawing output. Global verbose flag via `set_verbose(True)`.

```
╔══════════════════════════════════════════════════════════════════════╗
║                     🤖  MY TITLE                                     ║
╠══════════════════════════════════════════════════════════════════════╣
║  line 1                                                              ║
╚══════════════════════════════════════════════════════════════════════╝
```

Boxes printed by: preprocessor (steps 0–5) and agent nodes (Agent, Dedup, Collect, Judge, Reformulate, Answer, Clarify). `force=True` for critical events (always print).

## Key Global State Patterns (not thread-safe across concurrent students)

- `tools._ACTIVE_STUDENT_ID` — set before each `agent.run()` call
- `chatbot_api._planning_sessions` — active planning sessions by student_id
- `chatbot_api._ambiguity_sessions` — pending course disambiguation by student_id
- `eligibility._supabase` — lazy singleton Supabase client
- `course_name_mapper` — module-level singleton, cached Neo4j course list

## Environment Variables (`.env`)

```env
# Supabase
SUPABASE_URL=
SUPABASE_KEY=

# Neo4j Aura
NEO4J_URI=
NEO4J_USERNAME=
NEO4J_PASSWORD=
NEO4J_DATABASE=neo4j

# Groq (up to 5 keys for rate-limit fallback)
GROQ_API_KEY=
GROQ_API_KEY2=
GROQ_API_KEY3=
GROQ_API_KEY4=
GROQ_API_KEY5=

# Model selection
GROQ_MODEL_XXX=meta-llama/llama-4-scout-17b-16e-instruct  # utility LLM
GROQ_MODEL_AGENT=openai/gpt-oss-120b                       # agent LLM

# RAG
PINECONE_API_KEY=
PINECONE_INDEX=bnu-bylaws
HF_API_KEY=

# Optional: Google Sheets logging
GOOGLE_SHEET_ID=
# + service-account.json in project root

# Dev: default student for CLI testing
STUDENT_ID=22030094

# Debug flags
DEBUG_MODE=false       # set true to see preprocessor and agent reasoning
DEBUG_TO_FILE=false
DEBUG_LEVEL=
```

## Key Design Decisions

1. **Student ID never in LLM prompts** — injected via `_ACTIVE_STUDENT_ID` module variable, never passed as a tool parameter.
2. **Dual LLM model strategy** — cheap/fast model for utility tasks; powerful model only for agent tool selection and final answer synthesis.
3. **Judging loop over simple ReAct** — judge node ensures the agent keeps trying until context is genuinely sufficient, not just until one tool was called.
4. **Preprocessing before agent** — agent never deals with abbreviations, pronouns, or ambiguous names; preprocessor resolves everything first.
5. **Planning tool monkey-patching** — `chatbot_api.py` patches `start_course_planning` at import time to capture `PlanningState`, without touching `tools.py` or `agent.py`.
6. **Thread bridge for interactive planning** — `planning.py` kept unchanged with its `print`/`input` calls; thread bridge transparently captures and routes all I/O.
7. **Deduplication node** — prevents the agent from calling the same tool with the same args twice.
8. **Ambiguity handled at preprocessor level** — ambiguous course names stop the pipeline before the agent runs; student picks, then pipeline continues with the resolved name.
9. **Chained ambiguity** — when a query mentions N ambiguous courses, the preprocessor asks about them one at a time. Remaining courses are stored in `PendingAmbiguity.pending_courses` and processed after each disambiguation reply; the pipeline supports full multi-step chaining.
10. **Multi-requirement query splitting** — instead of relying on a single agent run to handle every topic in a complex query, an LLM-based splitter decomposes the query into atomic sub-queries. Each sub-query runs its own agent pipeline to collect tool context; all contexts are then combined for one final synthesis. This guarantees every independent topic receives a dedicated lookup.
11. **Planning in split queries** — when `start_course_planning` fires as a sub-query of a split, its output is surfaced directly (not summarised) so the interactive session continues correctly. Any co-located non-planning answer is prepended.
12. **Chat history is a sliding window** — only last 3 user + 3 assistant messages stored in Supabase and passed to agent.
13. **Authoritative credit count from DB** — `total_hours_earned` read directly from Supabase, never recomputed from `courses_degrees`.
