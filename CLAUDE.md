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
├── tools.py                   # 27 LangChain tool definitions
├── preprocessor.py            # Query preprocessing pipeline (5 steps)
├── course_name_mapper.py      # Fuzzy course name → Neo4j canonical name
├── neo4j_course_functions.py  # Core Neo4j/KG query functions + _resolve_code()
├── neo4j_track_functions.py   # Program/track-level KG functions (credit distribution, program info)
├── eligibility.py             # Prerequisite + credit-hour eligibility checker
├── rag_service.py             # RAG pipeline (Pinecone + HF + Groq)
├── planning.py                # Fully automated course planner (no I/O, returns dict)
├── planning_service.py        # Thin re-export of planning() for import compatibility
├── recommendation_service.py  # Elective/program recommendation via preference scoring
├── chatbot_connector.py       # Supabase client (history + student data)
├── student_functions.py       # Student profile query helper
├── llm_client.py              # Groq LLM client with 5-key fallback
├── debug_box.py               # Unicode box printer for debug output
├── preference_service.py      # Student preference CRUD (ai_preference column in Supabase)
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
[chatbot_api.py]         ← Main orchestration layer (routing, sessions)
        │
        ▼
[preprocessor.py]        ← Query cleaning (5-step pipeline)
        │
        ▼
[agent.py]               ← LangGraph ReAct agent with Judging Loop
        │
        ▼
[tools.py]               ← 27 LangChain tools
        │
        ├─► [neo4j_course_functions.py]  ← Knowledge Graph (Neo4j Aura)
        ├─► [eligibility.py]            ← Prerequisite + credit-hour checks (Supabase)
        ├─► [rag_service.py]            ← RAG pipeline (Pinecone + HuggingFace + Groq)
        ├─► [student_functions.py]      ← Student profile queries (Supabase)
        ├─► [planning.py]               ← Fully automated course planner
        ├─► [recommendation_service.py] ← Elective/program recommendation
        └─► [preference_service.py]     ← Student preference storage (Supabase)
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
    │      Priority 2: Normal query               → _preprocess_and_run()
    │
    │   [Normal path: _preprocess_and_run()]
    │      │
    │      ▼
    │   preprocessor.process(message, history, student_track)   ← 5-step pipeline
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
- `silent_context` — results from "silent" tools like `store_preference` (append-only); shown to the judge but not included in the main context block passed to the answer LLM
- `called_tools` — dedup signatures `"tool_name|sorted_args_json"` (append-only)
- `previous_reformulations` — prior query reformulations tried
- `tool_calls_this_round` — counter of tool rounds done (int, replace)
- `query_reformulations` — total reformulations attempted (int, replace)
- `original_query` — never changes after init
- `current_query` — may be replaced by reformulate node
- `satisfied` — set by judge node (bool)
- `judge_missing` — what the judge (or Python post-check) says is still missing
- `judge_missing_source` — `"llm"` or `"python_override"` (who set `judge_missing`)
- `judge_deps_check_info` — result string from `_multi_course_deps_missing` check; shown in debug box
- `judge_tools_this_round` — judge metadata snapshot of tool round counter

**Graph nodes:**
- **agent** — selects next tool. Uses `GROQ_MODEL_AGENT`. Retries on `tool_use_failed` (Groq 400).
- **dedup** — injects fake `[SKIPPED]` ToolMessage for duplicate calls.
- **tools** — `ToolNode(ALL_TOOLS)`, executes tool calls.
- **collect** — appends ToolMessage content to `accumulated_context` (or `silent_context` for tools in `_SILENT_TOOLS = {"store_preference"}`), increments round counter.
- **judge** — LLM evaluates if context fully answers `original_query`. Handles PURE PREFERENCE STATEMENT rule: if query is only a sentiment/interest statement and `store_preference` ran → `satisfied=true` immediately. Recognises `start_course_planning` result (`STUDENT COURSE PLAN` header) as a complete answer to any planning query. After the LLM verdict, runs the `_multi_course_deps_missing` Python post-check (see below).
- **answer** — generates final answer from `accumulated_context` via LLM synthesis. All tool results including planning go through the LLM — no special-casing or direct relay.
- **reformulate** — generates new `current_query` with a different angle (7 fixed angles in order). Resets `tool_calls_this_round` to 0.
- **clarify** — polite last-resort clarification request after all reformulations exhausted.

**Judge Python post-check — `_multi_course_deps_missing`:**
Runs after the LLM judge returns `satisfied=true`. Deterministically verifies that every quoted course name in `original_query` has its own `[get_course_dependencies(...)]` header in `accumulated_context`.

- **Gate:** only activates when `accumulated_context` already contains at least one `get_course_dependencies` header (meaning the agent already identified this as a deps-type query). Direction-agnostic: works for prereq-only, dependents-only, or both.
- **Program filter:** quoted terms immediately followed by the word `"program"` in the query text are skipped (they are program-scope filters, not courses to check).
- **Override:** if any quoted course lacks its own header, `satisfied` is flipped to `False` and `judge_missing` is set to that course name. `judge_missing_source` is set to `"python_override"`.
- **Why needed:** the LLM judge can be fooled into marking a course as covered when that course name appears only inside the result *body* of another course's dependency query (e.g. in a `dep_prereq` list). The Python check enforces coverage purely on tool-call headers, which is unambiguous.
- **Debug box:** the judge box always shows a `Deps check` line with one of three states:
  - `skipped — LLM already not satisfied` (LLM returned False; check never ran)
  - `ran → all courses covered ✓` (LLM returned True; no override needed)
  - `ran → override triggered, missing: <course>` (LLM returned True; Python flipped to False)
- The `Missing` line in the judge box is prefixed with `[LLM]` or `[Python override]` accordingly.

### Preprocessing Pipeline (`preprocessor.py`)

Runs before every agent invocation. Five steps:

1. **Reference resolution** — LLM resolves pronouns/vague refs using conversation history ("what about it?" → "what about machine learning?")
2. **Entity extraction** — LLM extracts course names and track/program names as JSON
3. **Course name mapping** — check `COURSE_ALIASES` first, then fuzzy match Neo4j. Single winner → auto-substitute; multiple close matches → `"ambiguous"` → store `PendingAmbiguity`, return clarification question. If more courses remain after the ambiguous one, they are saved in `PendingAmbiguity.pending_courses` and processed after disambiguation (chained ambiguity support).
4. **Track name mapping** — check `TRACK_ALIASES`, then fuzzy match programs. Always auto-resolves.
5. **Query rewriting** — substitute canonical names into resolved query (regex first; LLM fallback triggered when canonical name is absent from result — handles implied terms like "training 1 and 2")

**Fuzzy scoring:** `score = max(code_exact→1.0, similarity*0.6 + keyword_overlap*0.4, prefix_score)`; threshold=0.30; ambiguity_delta=0.08 (two candidates within 0.08 → ask user)

**`_load_courses()` query** (internal cache used for entity extraction): `MATCH (c:Course) OPTIONAL MATCH (c)-[r:BELONGS_TO]->(:Program) WITH c, collect(r.code)[0] AS rel_code RETURN c.name AS name, COALESCE(c.code, rel_code) AS code ORDER BY c.name` — the OPTIONAL MATCH ensures the 6 affected courses (whose `c.code` is null) still get a code for code-string matching in the preprocessor entity blocklist and alias checks.

**COURSE_ALIASES examples:** `"ml"→"machine learning"`, `"os"→"operating systems"`, `"oop"→"object oriented programming"`, `"dsa"→"data structures and algorithms"`, course codes like `"bcs311"→"artificial intelligence"`

**TRACK_ALIASES:** `"aim"/"ai"/"aiml"→"artificial intelligence and machine learning"`, `"sad"/"software"/"sw"→"software and application development"`, `"das"/"ds"/"data science"→"data science"`

**ENTITY_BLOCKLIST:** words filtered after extraction to prevent false matches: `courses`, `electives`, `semester`, `year`, `prerequisites`, `credits`, `hours`, `information`, etc.

**PendingAmbiguity** (stored in `_ambiguity_sessions[student_id]`): `original_query`, `dereferenced`, `ambiguous_term`, `candidates [{name,code,confidence}]`, `resolved_courses`, `pending_courses` (remaining unprocessed courses), `resolved_tracks`, `history`

**PreprocessResult** fields: `status`, `clean_query`, `clarification`, `pending`, `resolved_courses: Dict[str,str]` (original→canonical for every course mapped in this run)

**`process(message, history, student_track=None)`** — `student_track` is the student's canonical program name (fetched from Supabase by `chatbot_api._preprocess_and_run()` before calling the preprocessor). Used as a secondary hint for resolving track-vs-course conflicts.

**Chained ambiguity flow:** When multiple courses are ambiguous, the preprocessor asks about them one at a time. After each disambiguation reply, `resolve_ambiguity()` processes remaining `pending_courses`; if another is ambiguous it returns `status="ambiguous"` again. `_resolve_ambiguity_reply()` in `chatbot_api.py` detects this and stores the new pending instead of running the agent.

## Tools (`tools.py`) — 27 Tools

Student ID injected via module-level `_ACTIVE_STUDENT_ID` (set by `set_active_student_id()` before each graph run). All tools return plain strings. Course name fuzzy matching applied automatically via `_normalize_course()`.

The inline electives Cypher query inside `get_all_electives` uses `COALESCE(c.code, r.code) AS course_code` and `ORDER BY COALESCE(c.code, r.code)` (Case 1 — program is always provided).

**Silent tools** (`_SILENT_TOOLS = {"store_preference"}`): their results go to `silent_context` instead of `accumulated_context`, so they are visible to the judge but not included in the LLM answer prompt.

| Tool | When to use |
|---|---|
| `get_student_info` | "What's my GPA?", "What have I completed?" |
| `get_course_info` | "What is ML about?", "How many credits is SE?" |
| `get_course_dependencies` | "What does X close?", "What are prereqs for X?", "What does X unlock?", "I passed X, what can I take now?" |
| `get_course_timing` | "When is ML taught?", "Which semester is OS?" |
| `check_course_eligibility` | "Can I take ML?", "Am I eligible for OS?" |
| `get_courses_by_term` | "What's in year 2 semester 1?" |
| `get_courses_by_multiple_terms` | "What do I study in years 2 and 3?" |
| `get_all_electives` | "What electives are in the AI track?" |
| `get_elective_slots_time_and_occ` | "When can I take electives?", "How many elective slots?" |
| `filter_courses` | "Show all 3-credit courses in AIM", "What core courses in SAD?" |
| `get_program_total_credits` | "How many credits to graduate?", "Total credit requirement for AIM?" |
| `answer_academic_question` | "What's the minimum GPA?", "Graduation requirements?" — RAG over bylaws |
| `start_course_planning` | "Make a plan for me", "What should I take next semester?" |
| `get_program_info` | "Tell me about the AIM program", "What is the SAD track?" |
| `get_credit_hour_distribution` | "How are the 136 credits broken down?", "How many humanities credits?" |
| `get_specialized_core_courses` | "What mandatory specialized courses does AIM have?" |
| `get_specialized_elective_courses` | "What electives does the AI track offer?" |
| `get_all_specialized_courses` | "What courses are unique to data science?", "What differs between AIM and SAD?" |
| `get_general_courses` | "What GEN/humanities courses are there?" |
| `get_math_and_basic_science_courses` | "What math/BAS courses do I study?" |
| `get_basic_computing_sciences_courses` | "What BCS courses does AIM have?" |
| `get_all_types_courses` | "Show me the full curriculum for AIM" |
| `get_all_core_courses` | "What mandatory courses are in data science?" |
| `get_all_not_specialized_courses` | "What courses are shared between all programs?" |
| `compare_programs` | "Compare AIM and SAD", "What's the difference between DAS and AIM?" |
| `compare_courses` | "Compare ML and DL", "What's the difference between OS and networks?" |
| `recommend_core` | "What core courses should I take?", "Which mandatory courses can I register for?", "Recommend the most important core course for me", "Of [X, Y, Z], which should I prioritize?" |
| `store_preference` *(silent)* | Automatically called when student expresses interest/skill/dislike ("I love NLP", "I'm good at math", "I hate theory") |

## Data Sources

### Neo4j Knowledge Graph

```
Nodes:    (Course) — name, code*, description, credit_hours
          (Program) — name
          (GraphEmbedding) — embedding [indexed]

Rels:     (Course)-[:BELONGS_TO]->(Program)        rel props: elective='yes'/'no', year_level, semester, code*
          (Course)-[:HAS_PREREQUISITE]->(Course)   rel props: track (optional, restricts to specific programs)
          (Course)-[:PREREQUISITE_OF]->(Course)
          (Course)-[:SIMILAR_TO]->(Course)
```

**Programs (3 tracks):** `artificial intelligence & machine learning` (AIM), `software & application development` (SAD), `data science` (DAS)

**Year levels:** `First Year`, `Second Year`, `Third Year`, `Fourth Year`
**Semesters:** `First`, `Second`

#### Course Code Schema (IMPORTANT)

The `code` property is split across two locations depending on the course:

- **Code on the Course node** (`c.code`) — courses that belong to one program, or multiple programs all sharing the **same** code. The vast majority of courses.
- **Code on the BELONGS_TO relationship** (`r.code`) — courses that belong to multiple programs with a **different** code per program. The node's `code` property is `null` for these courses.

**6 affected courses (code lives on `BELONGS_TO`, not on the node):**
- `machine learning`
- `natural language processing`
- `image processing`
- `deep learning`
- `computer vision`
- `data mining`

**3-case resolution rule** applied in every Cypher query that reads a course code:

| Case | Condition | Cypher pattern | Python result |
|---|---|---|---|
| **1 / 2** | Program context available (required or passed) | `COALESCE(c.code, r.code) AS code` | plain `str` |
| **3** | No program context (`program_name=None`) | `CASE WHEN c.code IS NOT NULL THEN [{program: null, code: c.code}] ELSE collect({program: p.name, code: r.code}) END` | `str` if single node code; `[{program, code}, …]` if relationship codes |

**`_resolve_code(codes)`** (module-level helper in `neo4j_course_functions.py`) unwraps the Case-3 list:
- `[{program: None, code: "X"}]` → `"X"` (plain string)
- `[{program: "aim", code: "AIM304"}, …]` → returns the list as-is
- `None / []` → `None`

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

### Supabase — `student_preferences` table

| Column | Type | Notes |
|---|---|---|
| `student_id` | text PK | |
| `ai_preference` | jsonb | AI-inferred interest scores `{category: float 0–1}` |
| `user_preference` | jsonb | Reserved for explicit user-set preferences |
| `degree_preference` | jsonb | Reserved for degree-level preference data |
| `updated_at` | timestamptz | Auto-updated |

**Valid `ai_preference` categories (12):** `math`, `probability_statistics`, `programming`, `software_engineering`, `ai_ml`, `data_management`, `data_analysis`, `theory`, `networking_systems`, `visual_computing`, `language_text`, `optimization`

Scores are clamped to [0.0, 1.0]. Updated via `update_ai_preference(student_id, deltas)` which upserts (creates row on first call).

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

`planning()` is a fully automated function — no `print()` or `input()` calls. Called directly by the `start_course_planning` tool; returns a structured dict which the answer LLM synthesises into a natural response like any other tool result.

**Return dict:**
```python
{
  'student_id':        str,
  'year':              int,
  'semester':          str,       # 'First' or 'Second'
  'track':             str,
  'available_credits': int,       # credit limit based on GPA (15 / 18 / 21)
  'planned_credits':   int,
  'planned_courses':   List[dict],
  'advisor_notes':     List[str], # human-readable decisions for the LLM
}
```

**Credit limit by GPA:** GPA ≥ 3.0 → 21 cr | GPA ≥ 2.0 → 18 cr | else → 15 cr

**Planning stages:**

| Stage | What it adds | Overflow priority sets |
|---|---|---|
| **1 — Backlog** | Missed mandatory + elective slots from previous same-semester terms | `[cur_term_names, next_term_names]` |
| **2 — Current term** | Mandatory + elective slots for `(uni_year, semester)` | `[next_term_names]` |
| **3 — Future years** | `uni_year+1` → Year 4, same semester, mandatory + elective slots | `[]` |
| **4 — Leftover fill** | 1–2 spare credits: mandatory course from closest future year (same semester) | — |

**Overflow resolution (`_resolve_overflow`):** for each priority set, removes courses whose dependents don't intersect that term's course names (ascending `len(dependents)`, electives score 0 → removed first). Then a final pass over all remaining courses. Removed elective picks are returned to the shared pool. Overflow always terminates planning.

**Shared elective pool:** built once from `get_all_electives_by_program(track)`, filtered to eligible (not completed + prereqs met). Shrinks as stages consume electives via `pick_electives(n)` → `recommend_electives(..., eligible_electives=pool)` (Mode 3). Removed elective picks are returned to the pool via `return_to_pool()`.

**Pre-computed remaining elective slots:** before any stage runs, `remaining_slots: Dict[Tuple[int,str], int]` is built by:
1. Collecting all terms with slots in year/semester order
2. Iterating completed courses — each completed elective decrements the first slot with remaining capacity
This ensures stages never try to fill already-completed elective slots.

**Stage 4 detail:** uses `filter_courses()` directly from `neo4j_course_functions` with `year_level` and `semester` params (planning-internal, not exposed to the agent tool). Also checks prereqs. Priority: 1 course of 2 cr > 2 courses of 1 cr > 1 course of 1 cr > nothing.

**`_next_term(year, sem)`:** Term 1 → same year Term 2; Term 2 → next year Term 1.

**`planning_service.py`** is now a thin re-export (`from planning import planning`) kept only for import-compatibility.

## Recommendation Service (`recommendation_service.py`)

Scores electives and programs against a merged student preference vector.

**Preference sources and weights:**
- `degree_preference` — 0.45 (from transcript grades)
- `user_preference` — 0.35 (student signup)
- `ai_preference` — 0.20 (agent inference during chat)

**`recommend_electives(student_id, track, top_n, eligible_electives=None)`**

Three modes based on the `eligible_electives` parameter:

| Mode | Trigger | Output |
|---|---|---|
| **Mode 1** | `eligible_electives=None`, called by agent for general recommendation | Formatted string ranked by cosine similarity |
| **Mode 2** | `eligible_electives=None`, head-to-head comparison | Formatted string |
| **Mode 3** | `eligible_electives=List[dict]` (pre-filtered pool passed by planning) | `List[dict]` top-n electives sorted by cosine similarity — for planning-internal use only |

**Mode 3 detail:** planning passes its `elective_pool` directly, bypassing the eligibility check inside the service. The service scores each pool elective against the merged student vector and returns the top-n as structured dicts. This avoids a redundant Neo4j query since the pool is already built and filtered.

**Cosine similarity** between student preference vector and elective's category profile determines ranking.

## Multi-Requirement Query Splitting (`chatbot_api.py`)

Before every agent run, `_analyze_and_split(clean_query)` is called to detect whether the query asks about multiple independent topics.

**`_analyze_and_split(clean_query) → List[str]`**
- Calls the utility LLM with a 4-step structured prompt (resolve intra-query pronouns → find intent boundaries → assign one sub-query per intent → smell test).
- **Intent types:**
  - `COMPARISON` clause (`compare`, `vs`, `difference between`) → always ONE atomic sub-query, never split by items compared
  - `RECOMMEND` clause (`which should I choose`, `recommend`) → always ONE atomic sub-query
  - `FACTUAL` clause: N independent courses → N sub-queries; 1 course × M programs → M sub-queries; N×M only for pure factual; already atomic → unchanged
- Comparison/recommend arguments are **never** extracted as separate sub-queries even if they look like independent entities.
- Returns a list of sub-query strings. Single-item list → no split.

**`_split_and_run(student_id, clean_query, sub_queries, history, verbose)`**
- For each sub-query: calls `BNUAdvisorAgent.run_and_get_context()` to collect tool results without generating a per-sub-query answer.
- Combines all `accumulated_context` lists into one final `llm_call_text()` synthesis.
- Planning output (from `start_course_planning`) is treated identically to any other tool context — no special casing.

**`BNUAdvisorAgent.run_and_get_context(query, history, verbose)`** (in `agent.py`)
- Runs the full judging loop graph (agent → tools → judge) for one sub-query.
- Returns `accumulated_context: List[str]` (raw tool result strings) without calling the answer node.
- When `verbose=True`: streams with debug boxes for all nodes **except** `answer`/`clarify` (those are suppressed per sub-query; one combined answer box is shown at the end by `_split_and_run`).

**CLI loop** (`agent.py __main__`) also calls `_analyze_and_split` / `_split_and_run` via the already-imported `chatbot_api` module, so CLI and API entry points behave identically.

## Session Management (in-memory, `chatbot_api.py`)

```python
_ambiguity_sessions:  Dict[str, PendingAmbiguity] # waiting for disambiguation reply
```

Routing priority in `_route_message()`:
1. `_ambiguity_sessions[student_id]` → `_resolve_ambiguity_reply()`
2. Neither → `_preprocess_and_run()`

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

## Neo4j Course Functions (`neo4j_course_functions.py`)

Core KG query layer. All functions accept lowercase course/program names. Code resolution follows the 3-case rule above.

**`_resolve_code(codes)`** — module-level helper. Unwraps the Case-3 `[{program, code}]` structure into a plain string (single node code) or a list of per-program dicts (relationship codes). Always call this on Case-3 query results before returning to callers.

**Key functions and their code-resolution case:**

| Function | Program param | Case | Code return type |
|---|---|---|---|
| `get_courses_by_term(level, semester, program_name)` | optional, defaults to all 3 | 2 | `str` via `COALESCE` |
| `get_course_info(course_name, program_name)` | optional | 2 if given → `str`; 3 if None → `str` or `list` |
| `get_course_dependencies(course_name, program_name)` | optional | 2 if given → `str`; 3 if None → `str` or `list` per prereq |
| `get_course_closes(course_name, program_name)` | optional, defaults to all 3 | smart: single code if all programs agree, `list` if codes differ |
| `get_all_electives_by_program(program_name)` | required | 1 | `str` via `COALESCE` |
| `filter_courses(filters, course_types, return_fields, program_name, year_level, semester)` | optional, defaults to all 3 | 2 | `str` via `COALESCE` |

**`filter_courses` planning-internal params:** `year_level` (e.g. `'Third Year'`) and `semester` (`'First'`/`'Second'`) filter by `b.year_level` and `b.semester` on the `BELONGS_TO` relationship. These params are NOT exposed in the agent's `filter_courses` tool — only `planning.py` uses them directly via `from neo4j_course_functions import filter_courses`.

**`get_course_closes()` smart code derivation:** The query embeds `COALESCE(c.code, bc.code)` per program inside `program_details`. Python then collects all per-program codes; if all are identical (or only one program) → returns a plain string (Case 2); if they differ (the 6 affected courses queried across multiple programs) → returns `[{program, code}, …]` (Case 3 behaviour).

**`get_course_dependencies()` general branch (no program):** Collects `{program, code}` pairs from all `bp:BELONGS_TO` relationships, filters them with the same track/intersection logic used for `program_details`, then returns a `CASE WHEN prereq.code IS NOT NULL` structure. `_resolve_code()` applied in the Python serializer.

**`get_course_info()` no-program branch:** Collects `{program, code}` pairs from all `r:BELONGS_TO` relationships, then `CASE WHEN c.code IS NOT NULL THEN [{program: null, code: c.code}] ELSE rel_codes END`. `_resolve_code()` applied before returning.

**`get_credit_hour_distribution()`** — returns the faculty-wide credit breakdown (same for all programs): humanities 12 cr, math/sci 24 cr, basic computing 36 cr, applied/specialized 51 cr, field training 6 cr, grad projects 7 cr = 136 cr total.

**`get_program_info(prg, course_info, desc_info)`** — comprehensive program data: credit distribution, years-3/4 curriculum, unique year-1/2 courses, elective slots, elective catalogue, program description.

## Neo4j Track Functions (`neo4j_track_functions.py`)

Program/track-level query layer. Imports from `neo4j_course_functions`. All code resolution uses Case 1/2 (program prefix always known).

**`PROGRAM_CODE_PREFIX`** — maps canonical program name → course code prefix: `"artificial intelligence & machine learning"→"AIM"`, `"software & application development"→"SAD"`, `"data science"→"DAS"`.

**`_query_courses_by_code_prefix(code_prefix, elective, program_name, year_flag, sem_flag)`**
- WHERE condition: `(c.code STARTS WITH $code_prefix OR r.code STARTS WITH $code_prefix)` — checks both node and relationship code so the 6 affected courses (whose codes live on `BELONGS_TO`) are found correctly.
- RETURN: `COALESCE(c.code, r.code) AS course_code`
- Python deduplication (when no `program_name`): by `course_code` value; safe because shared courses (GEN/BAS) always have `c.code` on the node.

**Course category functions** (all return `{program, type, total_credits, courses:[…]}`):
- `get_specialized_core_courses(prg)` — AIM/SAD/DAS core courses + elective slot placeholders
- `get_specialized_elective_courses(prg)` — AIM/SAD/DAS elective courses
- `get_all_specialized_courses(prg)` — combines the two above (51 cr)
- `get_general_courses()` — GEN prefix, identical across all programs (12 cr)
- `get_MathAndBasicScience_courses()` — BAS prefix, identical across all programs (24 cr)
- `get_BasicComputingSciences_courses(prg)` — BCS prefix; Data Science has "Fundamentals of Data Science" instead of "Technical Report Writing" (36 cr)
- `get_all_types_courses(prg)` — all 4 categories combined
- `get_all_core_courses(prg)` — all non-elective courses
- `get_all_not_specialized_courses(prg)` — GEN + BAS + BCS only (72 cr)

**`get_credit_hour_distribution()`** — same as the one in `neo4j_course_functions.py` (duplicate, kept for callers that import from this module).

**`get_program_info(prg, course_info, desc_info)`** — same as the one in `neo4j_course_functions.py` (canonical version; `neo4j_track_functions` re-exports it).

**`get_program_total_credits(program_name)`** — queries `p.total_credits_required` from the Program node.

## Course Name Mapper (`course_name_mapper.py`)

Lazy-initialized class `CourseNameMapper`:
- Loads all `(Course)` nodes from Neo4j at init (cached) using an OPTIONAL MATCH fallback: `MATCH (c:Course) OPTIONAL MATCH (c)-[r:BELONGS_TO]->(:Program) WITH c, collect(r.code)[0] AS rel_code RETURN c.name, COALESCE(c.code, rel_code) AS code` — ensures the 6 affected courses get a code (any one of their program codes) for fuzzy matching by code string.
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

## Preference Service (`preference_service.py`)

Manages the `ai_preference` column in the `student_preferences` Supabase table.

**`VALID_CATEGORIES`** — the 12 allowed keys for the `ai_preference` dict (see Tools section above).

**`get_preferences(student_id) → Dict`** — returns all three preference dicts (`ai_preference`, `user_preference`, `degree_preference`). Parses JSON strings if Supabase returns them as strings.

**`update_ai_preference(student_id, deltas: Dict[str, float]) → Dict[str, float]`**
- Filters out any key not in `VALID_CATEGORIES`.
- Reads existing `ai_preference` scores from Supabase.
- Merges: `updated[cat] = clamp(current.get(cat, 0.0) + delta, 0.0, 1.0)`.
- Upserts the full updated dict (creates row on first call, updates on subsequent calls).
- Returns the updated dict.

**How it's called:** The `store_preference` LangChain tool calls `update_ai_preference` with LLM-inferred delta scores whenever the student expresses genuine interest, skill, background, or dislike about a subject category.

## Key Global State Patterns (not thread-safe across concurrent students)

- `tools._ACTIVE_STUDENT_ID` — set before each `agent.run()` call
- `chatbot_api._ambiguity_sessions` — pending course disambiguation by student_id
- `eligibility._supabase` — lazy singleton Supabase client
- `preference_service._client` — lazy singleton Supabase client
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
5. **Deduplication node** — prevents the agent from calling the same tool with the same args twice.
6. **Ambiguity handled at preprocessor level** — ambiguous course names stop the pipeline before the agent runs; student picks, then pipeline continues with the resolved name.
7. **Chained ambiguity** — when a query mentions N ambiguous courses, the preprocessor asks about them one at a time. Remaining courses are stored in `PendingAmbiguity.pending_courses` and processed after each disambiguation reply; the pipeline supports full multi-step chaining.
8. **Multi-requirement query splitting** — instead of relying on a single agent run to handle every topic in a complex query, an LLM-based splitter decomposes the query into atomic sub-queries. Each sub-query runs its own agent pipeline to collect tool context; all contexts are then combined for one final synthesis. This guarantees every independent topic receives a dedicated lookup.
9. **Planning is fully automated** — `planning()` has no `print()`/`input()` calls. It runs synchronously, returns a structured dict, and the `start_course_planning` tool formats it as a string. The answer LLM synthesises a natural response from it like any other tool result — no thread bridge, no monkey-patching, no interactive session management.
10. **Unified answer path for planning** — planning output goes through the same LLM answer node as all other tools. The judge recognises `STUDENT COURSE PLAN` as a complete answer to planning queries (zero entities to check). No special-casing in `_split_and_run` or the answer node.
11. **Chat history is a sliding window** — only last 3 user + 3 assistant messages stored in Supabase and passed to agent.
12. **Authoritative credit count from DB** — `total_hours_earned` read directly from Supabase, never recomputed from `courses_degrees`.
13. **Silent context for side-effects** — `store_preference` results go to `silent_context` (not `accumulated_context`) so the judge can see the side-effect was performed, but the preference update text is not included in the answer prompt. This keeps preference storage invisible to the student-facing response.
14. **Preference inference from conversation** — the agent system prompt includes rules for detecting preference signals ("I love X", "I'm good at Y", "I hate Z") and routing them to `store_preference` before (or instead of) any factual tool calls. Pure preference statements (no factual question) are fully satisfied after `store_preference` alone — the judge marks them satisfied immediately.
15. **Agent rules for preference vs. course completion** — the agent distinguishes "I got ML" (course completion → call `get_course_dependencies`) from "I love NLP" (preference → call `store_preference`). A `COURSE COMPLETION STATEMENT` triggers dependency lookup; a `PREFERENCE DETECTION` trigger only applies when the student expresses genuine interest/skill, not course completion.
16. **Comparison tools kept atomic** — `compare_programs` and `compare_courses` gather full data for all items in a single call. The query splitter treats any comparison/recommend clause as a single atomic sub-query; it never breaks out individual compared items as separate sub-queries.
17. **Python post-check for multi-course dependency coverage** — the LLM judge can be fooled into marking `satisfied=true` when a course name appears in the *body* of another course's dependency result (e.g. in a `dep_prereq` list). `_multi_course_deps_missing()` runs after the LLM verdict and deterministically verifies every quoted course in the query has its own `get_course_dependencies` header in context. If not, it overrides the verdict to `satisfied=false`. This prevents the judge from skipping uncovered courses while remaining direction-agnostic (works for prereq, dependents, or both queries). The debug box labels the missing source as `[LLM]` or `[Python override]` for traceability.
18. **Course code split across node vs. relationship** — six courses (`machine learning`, `natural language processing`, `image processing`, `deep learning`, `computer vision`, `data mining`) have program-specific codes that differ per program. Their `code` property was removed from the Course node and migrated to the `BELONGS_TO` relationship (`r.code`). All Cypher queries follow a 3-case rule: (1/2) program context available → `COALESCE(c.code, r.code)` returning a plain string; (3) no program context → `CASE WHEN c.code IS NOT NULL` returning a `[{program, code}]` list, unwrapped by `_resolve_code()`. The `_query_courses_by_code_prefix()` filter in `neo4j_track_functions.py` uses `(c.code STARTS WITH $prefix OR r.code STARTS WITH $prefix)` so these courses are still found by prefix. The fuzzy code matchers (`course_name_mapper.py`, `preprocessor.py`) use `OPTIONAL MATCH + collect(r.code)[0]` to get any one code for matching, which is sufficient for string-identity matching.
19. **Planning elective recommendation via Mode 3** — `recommend_electives` has a planning-internal Mode 3 triggered by passing `eligible_electives=pool`. This receives the pre-filtered elective pool directly (skipping redundant eligibility checks) and returns a ranked `List[dict]` rather than a formatted string. The shared pool shrinks across stages so each elective is picked at most once across the entire plan.
20. **Pre-computed elective slot counts** — before any planning stage runs, `remaining_slots` is computed once: total slots per term minus completed electives assigned sequentially. Stages read `remaining_slots.get((year, sem), 0)` instead of calling `get_elective_slots()` directly, ensuring already-filled slots are never re-planned.
21. **Planning Stage 4 uses extended `filter_courses`** — `filter_courses()` in `neo4j_course_functions` accepts optional `year_level` and `semester` params that filter on `BELONGS_TO` relationship properties. These are not exposed in the agent's `filter_courses` tool; only `planning.py` imports the function directly and uses them. This avoids a separate helper function while keeping the agent's API unchanged.
