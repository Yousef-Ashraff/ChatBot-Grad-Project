# BNU Academic Advisor Chatbot — Full Project Description
## (Feed this to Claude Code to generate CLAUDE.md)

---

## 1. Project Identity

**Project Name:** BNU Academic Advisor Chatbot
**Institution:** Faculty of Computer Science, Benha University (BNU = Badr University in Cairo)
**Purpose:** An AI-powered academic advisor chatbot that answers students' questions about courses, prerequisites, eligibility, curriculum planning, electives, and university bylaws/regulations.
**Deployment:** Mobile app (REST API) + CLI dev mode

---

## 2. High-Level Architecture

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

[chatbot_connector.py]   ← Supabase client (chat history + student data)
[llm_client.py]          ← Groq LLM client with 5-key fallback + streaming
[course_name_mapper.py]  ← Fuzzy course name matching (Neo4j-backed)
[debug_box.py]           ← Unicode box printer for verbose/debug mode
[google_sheets_logger.py]← Optional: log all Q&A pairs to Google Sheets
```

---

## 3. Technology Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| Agent Framework | LangGraph (StateGraph) + LangChain |
| LLM (Agent) | Groq — `openai/gpt-oss-120b` (configurable via `GROQ_MODEL_AGENT`) |
| LLM (Utilities) | Groq — `meta-llama/llama-4-scout-17b-16e-instruct` (configurable via `GROQ_MODEL_XXX`) |
| Knowledge Graph | Neo4j Aura (course catalog, prerequisites, programs) |
| Student Database | Supabase (PostgreSQL) — student profiles, chat history, academic records |
| RAG Vector DB | Pinecone (`bnu-bylaws` index) |
| Embeddings | HuggingFace `intfloat/multilingual-e5-large` via Inference API |
| Fuzzy Matching | Python `difflib.SequenceMatcher` + custom scoring |
| Optional Logging | Google Sheets via `gspread` + Service Account |

---

## 4. Data Stores

### 4.1 Neo4j Knowledge Graph

**Schema:**
```
Nodes:
  (Course)        — name, code, description, credit_hours
  (Program)       — name
  (GraphEmbedding)— embedding [indexed]

Relationships:
  (Course)-[:BELONGS_TO]->(Program)          — course belongs to a program
                                               rel has property: elective = 'yes'/'no'
  (Course)-[:HAS_PREREQUISITE]->(Course)     — this course requires that course first
  (Course)-[:PREREQUISITE_OF]->(Course)      — this course is needed before that course
  (Course)-[:SIMILAR_TO]->(Course)           — semantic similarity (from embeddings)
```

**Programs (3 tracks):**
- `artificial intelligence & machine learning` (AIM track)
- `software & application development` (SAD track)
- `data science` (DAS track)

**Year levels stored as:** `First Year`, `Second Year`, `Third Year`, `Fourth Year`
**Semesters stored as:** `First`, `Second`

Course nodes also store timing data (year_level, semester) on the `BELONGS_TO` relationship.

### 4.2 Supabase (Students Table)

**`students` table columns:**
- `student_id` — unique string ID
- `first_name`, `last_name`
- `track` — `"AIM"`, `"SAD"`, or `"DS"` (short code)
- `university_year` — integer 1–4
- `gpa` — float
- `total_hours_earned` — integer (authoritative credit count, set by registrar)
- `courses_degrees` — JSON array: `[{"name": "machine learning", "credit_hours": 3}, ...]`
- `chat_history` — JSON: `{"conversation_id": "uuid", "chat_history": [{role, content, timestamp}]}`
- `academic_details` — JSON: `{"completed_courses": [...], "earned_credits": N, "gpa": X}` ← legacy, kept for compatibility

**Track code → canonical program name mapping (TRACK_MAP in eligibility.py):**
```python
"SAD" → "software and application development"
"AIM" → "artificial intelligence and machine learning"
"DS"  → "data science"
```

### 4.3 Pinecone (RAG — BNU Bylaws)

- Index name: `bnu-bylaws`
- Top-K retrieval: 4 chunks
- Minimum similarity score: 0.30
- Metadata fields per vector: `topic`, `text`
- Used for bylaw/regulation questions (GPA policy, attendance, graduation requirements, etc.)

---

## 5. Request Flow (End-to-End)

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
    │   preprocessor.process(message, history)
    │      │
    │      ├─ Step 0: Reference resolution
    │      │     "what about it?" + history → "what about machine learning?"
    │      │     LLM resolves pronouns/vague refs using conversation history
    │      │
    │      ├─ Step 1: Entity extraction (LLM → JSON)
    │      │     Extract course names and track/program names from resolved query
    │      │
    │      ├─ Step 2: Entity filtering + char deduplication
    │      │     Remove ENTITY_BLOCKLIST words ("courses", "semester", "year", etc.)
    │      │     Collapse repeated chars: "mmml" → "ml", "arrrtttiiii" → "arti"
    │      │
    │      ├─ Step 3: Course name mapping
    │      │     a) Check COURSE_ALIASES table (instant, no DB call)
    │      │     b) Fuzzy match against Neo4j course list (prefix + sequence + keyword)
    │      │     c) Single clear winner → auto-resolve
    │      │     d) Multiple close matches → STOP, return "ambiguous" status
    │      │        → Store PendingAmbiguity in _ambiguity_sessions[student_id]
    │      │        → Return clarification question to student
    │      │
    │      ├─ Step 4: Track name mapping
    │      │     Check TRACK_ALIASES, then fuzzy match Neo4j programs
    │      │     Always auto-resolves (tracks are less ambiguous)
    │      │
    │      └─ Step 5: Query rewriting
    │            Substitute canonical names into resolved query
    │            First: regex substitution (fast)
    │            Fallback: LLM rewrite if partial matches remain
    │
    │   Returns PreprocessResult:
    │     status="ready"      → clean_query → agent.run()
    │     status="ambiguous"  → clarification string → returned to student
    │     status="passthrough"→ query as-is → agent.run()
    │
    ▼
BNUAdvisorAgent.run(query, history)   [agent.py]
    │
    ▼ [LangGraph StateGraph — Judging Loop]
    │
    START → agent → dedup → tools → collect → judge
                ↑                              │
                │          ┌──────── satisfied=true → answer → END
                │          │
                │          ├──── tool_calls < 3 → agent (next tool)
                │          │
                │          └──── tool_calls ≥ 3:
                │                 reformulations < 3 → reformulate → agent
                │                 reformulations ≥ 3 → clarify → END
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

---

## 6. LangGraph Agent — Judging Loop (agent.py)

### 6.1 Agent State (AgentState TypedDict)

```python
messages                # Full message history (add_messages reducer — appends)
accumulated_context     # List of tool result strings (operator.add — appends)
called_tools            # "tool_name|sorted_args_json" dedup signatures (operator.add)
previous_reformulations # Previous query reformulations tried (operator.add)
tool_calls_this_round   # Counter: tool rounds done in current query (int, replace)
query_reformulations    # Total reformulations attempted (int, replace)
original_query          # Never changes after init (str)
current_query           # May be replaced by reformulate node (str)
satisfied               # Set by judge node (bool)
judge_missing           # What was still missing (str)
judge_tools_this_round  # Snapshot of round count at judge time (int)
```

### 6.2 Graph Nodes

**agent** — Selects the next tool to call. Builds a focused prompt showing: current query, original query, compact summary of already-collected context, list of already-called tools (with args). Invokes `llm_with_tools`. Has automatic retry on `tool_use_failed` (Groq 400 errors). Uses `GROQ_MODEL_AGENT`.

**dedup** — Sits between agent and tools. Inspects tool calls in last AIMessage. For any call whose signature (`tool_name|sorted_args_json`) already exists in `called_tools`, injects a fake ToolMessage saying "[SKIPPED]" instead of calling the tool. Routes to `collect` if all duplicates, or `tools` if new calls exist.

**tools** — LangGraph `ToolNode(ALL_TOOLS)`. Executes the tool calls from the agent AIMessage.

**collect** — Runs after tools. Scans messages for ToolMessages from the last cycle. Appends their content to `accumulated_context`. Increments `tool_calls_this_round` by 1.

**judge** — Evaluates whether `accumulated_context` fully answers `original_query`. Uses `llm_call_json` (direct Groq, no LangChain). Returns `{"satisfied": true/false}`. Special case: if `start_course_planning` was called (detected via 3 independent checks), always returns `satisfied=true` (planning is self-contained interactive).

**answer** — Generates final natural-language answer using plain LLM (no tools). Uses `accumulated_context` + `original_query`. Special case: if planning tool was called, relays the planning output directly without LLM synthesis.

**reformulate** — Generates a new `current_query` that explores a different angle. Has a fixed list of 7 angles to explore in order (year/semester, prerequisites, dependents, programs, eligibility, credits, description). Resets `tool_calls_this_round` to 0. Tries up to 3 times with increasing temperature to avoid duplicates.

**clarify** — Last resort after all reformulations exhausted. Generates a polite clarification request to the student suggesting what additional detail would help.

### 6.3 Routing Logic

```
after agent:  tool_calls → dedup; no tool_calls → answer
after dedup:  any SKIPPED → collect (bypass tools); all new → tools
after tools:  → collect (always)
after collect:→ judge (always)
after judge:  satisfied=true → answer
              satisfied=false:
                tool_calls_this_round < 3 → agent
                tool_calls_this_round ≥ 3 AND reformulations < 3 → reformulate
                tool_calls_this_round ≥ 3 AND reformulations ≥ 3 → clarify
after reformulate: → agent (always)
answer: → END
clarify:→ END
```

### 6.4 Limits
- `MAX_TOOL_CALLS_PER_ROUND = 3` (tool rounds per query/reformulation)
- `MAX_REFORMULATIONS = 3` (reformulation attempts before giving up)
- History window: last 6 messages (3 user + 3 assistant) passed into agent
- `recursion_limit` = `(MAX_REFORMULATIONS + 1) * MAX_TOOL_CALLS_PER_ROUND * 6 + MAX_REFORMULATIONS + 10`

---

## 7. Tools (tools.py) — 12 Tools

All tools: return plain strings. Course name fuzzy matching applied automatically via `_normalize_course()`. Student ID injected via module-level `_ACTIVE_STUDENT_ID` (set by `set_active_student_id()` before each graph run).

| Tool | Function | Neo4j/DB call | When to use |
|---|---|---|---|
| `get_student_info` | Student GPA, credits, completed courses, track, year | Supabase | "What's my GPA?", "What have I completed?" |
| `get_course_info` | Course name, description, credit hours, type (core/elective) | Neo4j | "What is ML about?", "How many credits is SE?" |
| `get_course_prerequisites` | Both prerequisites (what's needed) AND dependents (what this unlocks/closes) | Neo4j | "What does X close?", "What are prereqs for X?", "What does X unlock?" |
| `get_course_timing` | Year level + semester a course is offered | Neo4j | "When is ML taught?", "Which semester is OS?" |
| `check_course_eligibility` | Eligible to take a course? Checks: program membership, completed prereqs, credit-hour gates | Supabase + Neo4j | "Can I take ML?", "Am I eligible for OS?" |
| `get_courses_by_term` | All courses in a specific year+semester | Neo4j | "What's in year 2 semester 1?" |
| `get_courses_by_multiple_terms` | Courses for several terms in one call | Neo4j | "What do I study in years 2 and 3?" |
| `get_all_electives` | All elective courses for a program (with credits, descriptions) | Neo4j | "What electives are in the AI track?" |
| `get_elective_slots` | When/how many elective slots per semester in a program | Neo4j | "When can I take electives?", "How many elective slots?" |
| `filter_courses` | Search courses by multiple criteria (credits, type, program, list) | Neo4j | "Show all 3-credit courses in AIM", "What core courses in SAD?" |
| `answer_academic_question` | RAG search over BNU bylaws/regulations | Pinecone + Groq | "What's the minimum GPA?", "How many absences allowed?", "Graduation requirements?" |
| `start_course_planning` | Start multi-turn interactive course planning session | Supabase + Neo4j | "Make a plan for me", "What should I take next semester?" |

---

## 8. Preprocessor (preprocessor.py)

### Course Aliases (COURSE_ALIASES)
A hardcoded dict mapping common abbreviations to canonical Neo4j course names. Examples:
- `"ml"` → `"machine learning"`
- `"ai"` → `"artificial intelligence"`
- `"os"` → `"operating systems"`
- `"db"` / `"dbs"` → `"database systems"`
- `"oop"` → `"object oriented programming"`
- `"dsa"` → `"data structures and algorithms"`
- `"prob"` / `"stats"` / `"stat"` → `"probability and statistical methods"`
- Course codes like `"bcs311"` → `"artificial intelligence"`, `"aim401"` → `"deep learning"`, etc.

### Track Aliases (TRACK_ALIASES)
- `"aim"` / `"ai"` / `"aiml"` → `"artificial intelligence and machine learning"`
- `"sad"` / `"software"` / `"sw"` → `"software and application development"`
- `"das"` / `"ds"` / `"data science"` → `"data science"`

### Entity Blocklist (ENTITY_BLOCKLIST)
Words filtered out after entity extraction to prevent false course/track matches:
`courses`, `electives`, `semester`, `year`, `prerequisites`, `credits`, `hours`, `information`, `details`, `list`, `schedule`, `when`, `what`, `how`, `all`, `any`, etc.

### Fuzzy Matching Scoring Formula
```python
score = max(
    code_exact_match → 1.0,
    similarity * 0.6 + keyword_overlap * 0.4,   # sequence + keyword
    prefix_score  # 0.50 + (prefix_len_ratio * 0.40)
)
threshold = 0.30
ambiguity_delta = 0.08  # two candidates within 0.08 → ask user to pick
```

### PendingAmbiguity Object
Stored in `chatbot_api._ambiguity_sessions[student_id]` when ambiguous:
```python
original_query    # raw user message
dereferenced      # after reference resolution
ambiguous_term    # the term that was ambiguous
candidates        # [{name, code, confidence}, ...]
resolved_courses  # already-resolved courses so far
resolved_tracks   # already-resolved tracks
history           # conversation history at time of query
```

---

## 9. Course Name Mapper (course_name_mapper.py)

A separate, lazy-initialized class `CourseNameMapper` that:
- Loads all `(Course)` nodes from Neo4j at initialization (cached)
- Scores each course using: exact match (1.0), code match (0.9–1.0), prefix score (0.50–0.90), sequence+keyword (0.0–1.0)
- `find_best_match(user_input)` → single best match or None
- `find_ambiguous_matches(user_input)` → list of candidates when top scores within `ambiguity_delta=0.08` of each other
- `find_all_matches(user_input)` → all matches above threshold
- Module-level singleton via `get_course_mapper()`
- Convenience function `map_course_name(user_input)` → canonical name or None
- Convenience function `get_ambiguous_matches(user_input)` → candidates list

**RULE:** Always check COURSE_ALIASES first. Only call this mapper if the input is NOT in COURSE_ALIASES.

---

## 10. Eligibility Checker (eligibility.py)

**`check_course_eligibility(student_id, course_name)`** performs:
1. Single Supabase query to get student row (name, track, `courses_degrees`, `total_hours_earned`, `university_year`, `gpa`)
2. Build `completed_courses` from `courses_degrees[].name` (lowercased)
3. Convert `track` code (SAD/AIM/DS) to canonical program name via `TRACK_MAP`
4. Guard: unknown program → error
5. Guard: `course_belongs_to_program()` via Neo4j Cypher — course must be in student's program
6. `get_course_dependencies(course_name, program_name)` from Neo4j
7. If no prerequisites → immediately eligible
8. For each prerequisite:
   - If `"Required_Credit_Hours"` key present → credit-hour gate (compare vs `total_hours_earned`)
   - Otherwise → must be in `completed_courses`

Returns dict with: `eligible`, `course`, `message`, `missing_prerequisites`, `credit_requirement`, `credit_requirement_met`, `earned_credits`

**`get_student_context(student_id)`** — shared helper used by eligibility AND planning:
Returns `{completed_courses, program_name, total_hours_earned, university_year, current_term, gpa, first_name, last_name}`

---

## 11. RAG Service (rag_service.py)

**Pipeline:**
1. Embed question via HuggingFace `intfloat/multilingual-e5-large` (adds `"query: "` prefix)
2. Query Pinecone `bnu-bylaws` index (top_k=4, min_score=0.30)
3. Format retrieved chunks as context (topic + text, separated by `---`)
4. Include last 4 conversation turns as `history_block`
5. Call Groq LLM with strict instructions:
   - Answer using ONLY the bylaw context
   - Never mention article numbers, source references
   - 3–6 sentences or brief bullet list
   - If not covered → suggest contacting registrar

Used for questions about: graduation requirements, GPA policies, academic probation, attendance rules, credit transfer, withdrawal, disciplinary policies, leave of absence, academic warnings.

---

## 12. Planning System (planning.py + planning_service.py)

### The planning() function (planning.py)
- Interactive function using `print()` and `input()` — originally designed for notebooks
- Fetches student context via `get_student_context(student_id)` from eligibility.py
- Asks student: what year/semester to plan for, which electives they prefer
- Shows current academic status (GPA, credits, completed courses)
- Fetches courses for the target term from Neo4j
- Checks prerequisite eligibility for each course
- Handles elective slots per program and semester
- Returns a complete course plan dict

### PlanningOrchestrator (planning_service.py)
Wraps the interactive `planning()` for chatbot use via a thread bridge:

**Architecture:**
- `planning()` runs in a daemon thread
- `_ThreadRouter` replaces `sys.stdout` — captures output from the planning thread into per-thread StringIO buffers
- `_smart_input` replaces `builtins.input` — when planning thread calls `input(prompt)`:
  1. Flushes buffer + prompt string into `_out_q`
  2. Blocks on `_in_q` waiting for chatbot's reply
- `chatbot_api` calls `advance(state, reply)` for each student message turn

**PlanningState** holds:
- `current_step`: `PlanStep.IN_PROGRESS` or `PlanStep.COMPLETE`
- `result`: the final plan dict when complete
- `_out_q`: thread → chatbot queue (events: `"need_input"` or `"done"`)
- `_in_q`: chatbot → thread queue (student reply strings)
- `_thread`: the daemon thread reference

**Session lifecycle in chatbot_api.py:**
1. `start_course_planning` tool called → `_start_and_cache_planning(student_id)` (monkey-patched)
2. `PlanningOrchestrator.start()` → first message + PlanningState
3. State stored in `_planning_sessions[student_id]`
4. Each subsequent student message → `PlanningOrchestrator.advance(state, reply)`
5. When `PlanStep.COMPLETE` → `del _planning_sessions[student_id]`
6. On `clear_history()` → also clears `_planning_sessions[student_id]`

**Timeout:** 90 seconds per planning event

---

## 13. LLM Client (llm_client.py)

### Key Design: 5-Key Groq Fallback
```
.env keys: GROQ_API_KEY, GROQ_API_KEY2, GROQ_API_KEY3, GROQ_API_KEY4, GROQ_API_KEY5
```
On rate limit (`429`, `"rate_limit"`, `"quota"`, etc.) → try next key automatically.
Raises `RuntimeError` only when ALL configured keys fail.

### Model Split
- `GROQ_MODEL` (from `GROQ_MODEL_XXX` env) — utility LLM: preprocessing, judging, reformulation, entity extraction, RAG generation. Default: `"meta-llama/llama-4-scout-17b-16e-instruct"`
- `GROQ_MODEL_AGENT` (from `GROQ_MODEL_AGENT` env) — agent LLM (tool selection, answer synthesis). Default: `"openai/gpt-oss-120b"`

### Public API
- `llm_call(messages, temperature, max_tokens)` → blocking, returns str
- `llm_call_json(prompt, system, temperature, max_tokens)` → blocking, JSON-only
- `llm_call_text(system, user, temperature, max_tokens)` → blocking, text
- `llm_call_stream(messages, ...)` → sync generator yielding str chunks
- `llm_call_stream_text(system, user, ...)` → sync generator yielding str chunks

---

## 14. Chatbot Connector (chatbot_connector.py)

Supabase client wrapper. Key behaviors:
- `get_chat_history(student_id)` → `{"conversation_id": "uuid", "chat_history": [...]}`
- `add_message(student_id, role, content)` — maintains rolling window:
  - Keeps last 3 user messages + last 3 assistant messages (sorted by timestamp)
  - Optionally logs to Google Sheets: buffers user message, logs complete turn when assistant message arrives
- `get_or_initialize_academic_details(student_id)` → legacy compatibility
- `clear_chat_history(student_id)` → generates new `conversation_id`, clears messages

---

## 15. API Server (api_server.py)

FastAPI app, runs on port 8000, CORS open (`"*"`).

| Endpoint | Method | Description |
|---|---|---|
| `/chat` | POST | `{student_id, message}` → `{ok, student_id, response}` |
| `/student/{student_id}` | GET | Student profile for mobile login UI |
| `/clear-history` | POST | `{student_id}` → clears Supabase history + sessions |
| `/disambiguation` | POST | `{student_id, term}` → `{ok, candidates: [{name, code, confidence}]}` |
| `/health` | GET | `{status: "ok", version: "2.0.0"}` |

---

## 16. Session Types in chatbot_api.py

Three in-memory dictionaries:

```python
_planning_sessions:   Dict[str, PlanningState]      # active multi-turn planning
_ambiguity_sessions:  Dict[str, PendingAmbiguity]   # waiting for disambiguation reply
```

**Routing priority in `_route_message()`:**
1. `_ambiguity_sessions[student_id]` present → `_resolve_ambiguity_reply()`
2. `_planning_sessions[student_id]` present → `_advance_planning()`
3. Neither → `_preprocess_and_run()`

---

## 17. Debug System (debug_box.py)

All debug output uses Unicode box-drawing characters. Global verbose flag set via `set_verbose(True)`.

Box style:
```
╔══════════════════════════════════════════════════════════════════════╗
║                     🤖  MY TITLE                                     ║
╠══════════════════════════════════════════════════════════════════════╣
║  line 1                                                              ║
╚══════════════════════════════════════════════════════════════════════╝
```

Each preprocessor step and agent node prints its own box when verbose=True. Boxes use `force=True` for critical events (always print) or default behavior (only when verbose).

Preprocessor boxes: Step 0 (reference resolution), Step 1&2 (entity extraction + dedup), Step 3 (course mapping), Step 4 (track mapping), Step 5 (clean query).

Agent boxes: Agent (tool calls), Tool Result, Dedup (skipped), Collect (context count), Judge (satisfied?), Reformulate (new query), Answer, Clarify.

---

## 18. Question Types Supported

The chatbot handles ALL of these question categories:

**Course Prerequisites & Dependencies:**
- "What are the prerequisites for Machine Learning?"
- "What does Probability and Statistics close?" (= what courses require it)
- "What does completing Operating Systems unlock?"
- "What is Deep Learning a prerequisite for?"
- "What courses depend on Data Structures?"

**Eligibility:**
- "Can I take Machine Learning?"
- "Am I eligible for Advanced AI?"
- "Do I have the prerequisites for Deep Learning?"
- "Can I register for Operating Systems this semester?"

**Course Information:**
- "What is Computer Vision about?"
- "How many credits is Software Engineering?"
- "Is Machine Learning a core or elective course?"
- "How many credit hours do I need for Graduation Project 1?"
- "How many credit hours do I need for Graduation Project 2?"

**Timing / Schedule:**
- "When is Machine Learning taught?"
- "Which semester is Operating Systems?"
- "What year do students take Data Structures?"
- "What courses are in year 3 semester 2?"
- "Show me the second-year curriculum"

**Electives:**
- "What electives are available in my program?"
- "What electives are in the AIM track?"
- "What are the elective options for SAD?"
- "When can I take electives in the data science program?"
- "How many elective slots do I have in year 4?"

**Filtering / Comparisons:**
- "What courses have more than 2 credit hours?" (from a given list)
- "What courses from [list] are in the AI program?"
- "Show me all 3-credit courses in the SAD track"
- "What core courses are in the AIM program?"
- "What electives have 2 credit hours?"

**Student Profile:**
- "What's my GPA?"
- "How many credits have I earned?"
- "What courses have I completed?"
- "What track am I in?"
- "What year am I in?"

**Course Planning:**
- "Help me plan my courses"
- "What should I take next semester?"
- "Make a study plan for me"
- "What courses do I still need to graduate?"
- "Give me a recommended course schedule"

**Bylaws & Regulations (RAG):**
- "What is the minimum GPA to avoid academic probation?"
- "How many absences are allowed per course?"
- "What are the graduation requirements?"
- "Can I withdraw from a course after add/drop?"
- "What happens if I fail a course twice?"
- "What are the credit transfer policies?"
- "What is the academic warning policy?"

**Multi-turn / Context-aware:**
- "What about it?" (references previous course discussed)
- "When can I take them?" (references electives from previous answer)
- "Is that a prerequisite for anything?" (references course from context)

---

## 19. Environment Variables (.env)

```
# Supabase
SUPABASE_URL=
SUPABASE_KEY=

# Neo4j Aura
NEO4J_URI=
NEO4J_USERNAME=
NEO4J_PASSWORD=
NEO4J_DATABASE=neo4j

# Groq (up to 5 keys for fallback)
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
# + service-account.json file in project root

# Dev: default student for CLI testing
STUDENT_ID=22030094
```

---

## 20. File Structure

```
project/
├── api_server.py           # FastAPI HTTP server
├── chatbot_api.py          # Main orchestration layer + session management
├── agent.py                # LangGraph judging-loop agent
├── tools.py                # 12 LangChain tool definitions
├── preprocessor.py         # Query preprocessing pipeline (5 steps)
├── course_name_mapper.py   # Fuzzy course name → Neo4j canonical name
├── neo4j_course_functions.py  # All Neo4j/KG query functions
├── eligibility.py          # Prerequisite + credit-hour eligibility checker
├── rag_service.py          # RAG pipeline (Pinecone + HF + Groq)
├── planning.py             # Interactive course planning function
├── planning_service.py     # Thread bridge wrapping planning() for chatbot
├── chatbot_connector.py    # Supabase client (history + student data)
├── student_functions.py    # Student profile query helper
├── llm_client.py           # Groq LLM client with 5-key fallback
├── debug_box.py            # Unicode box printer for debug output
├── google_sheets_logger.py # Optional conversation logging to Sheets
└── .env                    # All credentials and config
```

---

## 21. Running the Project

```bash
# Install dependencies
pip install fastapi uvicorn langchain-groq langgraph langchain-core \
            supabase python-dotenv neo4j pinecone-client huggingface_hub \
            requests gspread google-auth

# Start API server
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

# CLI mode (for development)
python agent.py --student 22030094 --verbose

# Smoke test
python agent.py --test

# Direct API test
python chatbot_api.py
```

---

## 22. Key Design Decisions & Patterns

1. **Student ID never in LLM prompts** — injected via module-level variable `_ACTIVE_STUDENT_ID` in tools.py, set before each graph invocation. LLM never sees it as a tool parameter.

2. **Dual LLM model strategy** — cheap/fast model for utility tasks (preprocessing, judging, reformulation), powerful model only for agent tool selection and final answer synthesis.

3. **Judging loop over simple ReAct** — the judge node ensures the agent keeps trying until it has genuinely enough data, not just until one tool was called. This catches cases where the first tool result was insufficient.

4. **Preprocessing before agent** — the agent NEVER deals with abbreviations, pronouns, or ambiguous names. The preprocessor resolves everything first. This keeps agent prompts clean and reduces wasted tool calls.

5. **Planning tool monkey-patching** — `chatbot_api.py` patches `start_course_planning` at import time to capture the `PlanningState` for multi-turn sessions, without touching `tools.py` or `agent.py`.

6. **Thread bridge for interactive planning** — `planning.py` is kept unchanged (with its `print`/`input` calls), the thread bridge transparently captures and routes all I/O.

7. **Deduplication node** — prevents the agent from calling the same tool with the same args twice, wasting rounds.

8. **Ambiguity handled at preprocessor level** — ambiguous course names stop the pipeline before the agent runs. The student picks, then the pipeline continues with the resolved name.

9. **Chat history is a sliding window** — Supabase stores only the last 3 user + 3 assistant messages per student. Agent receives the last 6 messages as context.

10. **Authoritative credit count from DB** — `total_hours_earned` is read directly from Supabase, NOT recomputed from `courses_degrees`. The registrar's system sets it.
