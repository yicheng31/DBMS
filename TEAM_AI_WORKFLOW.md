# Team AI Workflow Guide — TransitFlow

A practical guide for three students working together on TransitFlow using any AI coding assistant (Claude Code, GitHub Copilot, Cursor, Gemini Code Assist, etc.).

**Read this before you write a single line of code.**

---

## Table of Contents

- [Part 0: Before Anyone Writes Code — The Schema-First Rule](#part-0-before-anyone-writes-code--the-schema-first-rule)
- [Part 1: Team Coordination with AI](#part-1-team-coordination-with-ai)
- [Part 2: The AI-Integrated Workflow Loop](#part-2-the-ai-integrated-workflow-loop)
- [Part 3: Small Working Examples](#part-3-small-working-examples)
- [Part 4: Prompts That Work](#part-4-prompts-that-work)
- [Appendix: Pre-Session Checklist](#appendix-pre-session-checklist)

---

## Part 0: Before Anyone Writes Code — The Schema-First Rule

> **Critical:** Every query function in `databases/relational/queries.py` and `databases/graph/queries.py` runs SQL or Cypher against your database. That SQL references table names and column names that **you** design. If one person's AI generates `SELECT * FROM stations` and another person's generates `SELECT * FROM metro_stations`, nothing will work together.
>
> **The rule: agree on `databases/relational/schema.sql` as a team before anyone implements a single query function.**

### Step 0.1 — Run the Schema Design Workshop Together

Do this once as a team, before splitting work. It takes about 90 minutes.

**Preparation (each person, before the meeting):**
1. Read `train-mock-data/metro_stations.json` and `train-mock-data/bookings.json`
2. Read the stub function signatures in `databases/relational/queries.py` — the function names and their docstrings tell you exactly what data the queries need to return
3. Skim `train-mock-data/national_rail_schedules.json`, `train-mock-data/registered_users.json`, `train-mock-data/payments.json`

**During the workshop:**
1. Each person asks their AI assistant: *"Given this JSON data [paste 10–20 lines], what SQL tables would you design?"*
2. Compare the three AI outputs as a team — they will differ
3. Discuss and decide together (AI proposes options; humans decide)
4. Write the agreed schema into `databases/relational/schema.sql`

See [Example 1](#example-1-schema-design-workshop) in Part 3 for a concrete walkthrough.

### Step 0.2 — Commit and Lock the Schema

Once your team agrees on the schema, one person commits it:

```bash
git checkout -b feature/schema-design
git add databases/relational/schema.sql
git commit -m "Add agreed relational schema - team reviewed"
```

Open a Pull Request and have all three teammates approve it before merging to main. After it merges, **do not rename tables or columns without telling the whole team** — it will break everyone else's queries.

### Step 0.3 — Do the Same for the Graph Schema

The graph queries in `databases/graph/queries.py` (e.g., `query_shortest_route`, `query_station_connections`) need a Neo4j node/relationship schema. Read `train-mock-data/metro_stations.json` and `train-mock-data/national_rail_stations.json`, decide on node labels (`Station`, `MetroStation`, etc.) and relationship types (`CONNECTS_TO`, `INTERCHANGE`, etc.) as a team before implementing graph queries.

---

## Part 1: Team Coordination with AI

### 1.1 — Who Owns What

Use this as a starting point. Adjust it to your team.

| Area | Files to implement | Shared dependency |
|---|---|---|
| Relational schema | `databases/relational/schema.sql` | **Whole team — agree together** |
| Relational queries | `databases/relational/queries.py` | Schema must be finalized first |
| Graph schema + queries | `databases/graph/queries.py` | Station IDs from relational schema |
| Seeding & testing | `skeleton/seed_postgres.py`, `skeleton/seed_neo4j.py` | Both schemas |

**Document your assignments.** Create a `TEAM.md` file at the project root:

```markdown
# Team Assignments

| Name  | Primary responsibility                          |
|-------|-------------------------------------------------|
| Alice | Relational schema + relational query functions  |
| Bob   | Graph schema + graph query functions            |
| Carol | Seeding scripts + integration testing           |
```

### 1.2 — Git Basics (Step by Step)

If you are new to Git, follow this pattern every time you start working:

**One-time setup:**
```bash
# Clone the shared repo (do this once)
git clone <your-repo-url>
cd transitflow-demo
```

**Every time you start a work session:**
```bash
# 1. Make sure you have the latest code from your teammates
git checkout main
git pull origin main

# 2. Create a branch for what you're about to do
git checkout -b feature/alice/metro-schedules-query
```

**While working:**
```bash
# Save your progress frequently
git add databases/relational/queries.py
git commit -m "Implement query_metro_schedules - returns schedules by origin/destination"
```

**When you're done with a feature:**
```bash
# Push your branch to GitHub
git push origin feature/alice/metro-schedules-query
# Then open a Pull Request on GitHub and ask a teammate to review
```

**Branch naming convention:** `feature/<your-name>/<what-youre-doing>`

Examples:
- `feature/alice/relational-schema`
- `feature/bob/graph-shortest-route`
- `feature/carol/seed-postgres`

### 1.3 — The Shared AI Context File

> **The single most impactful thing you can do for consistency.**

Create `AI_SESSION_CONTEXT.md` in the repo root (a template is provided — see [AI_SESSION_CONTEXT.md](AI_SESSION_CONTEXT.md)). Every time someone opens an AI chat session, they **paste the contents of this file as the first message**.

This file contains:
- The project's agreed coding conventions
- Your finalized schema (once decided)
- The function signatures you're implementing
- Your team's decisions log

The AI will then know your table names, column names, return types, and style — and will generate code that fits your codebase instead of inventing its own conventions.

**Who updates it:** Whoever merges a schema change or makes an architectural decision updates `AI_SESSION_CONTEXT.md` in the same commit. Treat it like a living document.

### 1.4 — The Before-You-Start Ritual

Before opening your AI assistant each session:

1. `git pull origin main` — get your teammates' latest merged work
2. Check GitHub for open Pull Requests — is anything waiting for your review?
3. Tell your teammates (via your team chat) what you're about to work on: *"Working on query_metro_schedules today"*
4. Paste `AI_SESSION_CONTEXT.md` into your AI chat as the first message

This takes two minutes and prevents three people asking AI to solve the same problem three different ways.

### 1.5 — Agree on a Definition of Done Per Stub

Before implementing any stub function, answer these questions as a team:

- What input does it receive? (already documented in the docstring)
- What should it return? (already documented — look at the `Returns:` section)
- What does a correct output look like for a known input?

Write this down. For example, for `query_metro_schedules("MS01", "MS09")`:
- *"Should return at least one schedule. Each dict must have keys `schedule_id`, `line`, `departure_time`, `stops_list`."*

This is your acceptance criterion. When your AI generates code, test it against this criterion before marking the task done.

---

## Part 2: The AI-Integrated Workflow Loop

For every feature or function you implement, follow this five-stage loop. Never skip straight to Implementation.

```
Analysis & Planning → Options Evaluation → Minimal Implementation → Testing → Merging
         ↑                                                                        |
         └────────────────────────────────────────────────────────────────────────┘
                            (loop back if tests fail or reveal new requirements)
```

### Stage 1 — Analysis & Planning

**What you do:** Understand the problem before asking AI to solve it.

1. Read the stub function's docstring — it tells you exactly what the function must do
2. Look at the mock data that the function will query
3. Trace which table(s) you'll need (from your agreed schema)

**AI's role at this stage:** Ask AI to *explain*, not generate. Example:

> *"I need to implement `query_metro_schedules(origin_id, destination_id)`. It should return schedules that serve both stations in the correct order. My schema has a `metro_schedules` table with columns: `schedule_id, line, direction, stops (JSONB array)`. Can you explain what SQL approach I'd use to find schedules where both station IDs appear in the stops array in the right order?"*

**Human decision point:** Do you understand the approach before proceeding? If not, ask AI to explain further — don't ask it to generate code yet.

### Stage 2 — Options Evaluation

**What you do:** Ask AI for 2–3 approaches and compare them with your teammate.

Example prompt:

> *"Give me two different SQL approaches to find metro schedules where MS01 comes before MS09 in a JSONB array of stop IDs. Show the tradeoffs."*

AI might propose:
- Option A: Use `jsonb_array_elements` with position tracking
- Option B: Use `@>` containment operator + position comparison

Compare with your teammate. Pick the one that matches your schema and your team's SQL comfort level. Document the decision in `AI_SESSION_CONTEXT.md`:
> *"Metro schedule stop-order checking: using jsonb_array_elements approach (Option A) — clearer to read, easier to debug"*

### Stage 3 — Minimal Implementation

**What you do:** Implement one function at a time. Get it working before moving to the next.

**Before generating code, prepare your prompt:**
1. Paste your `AI_SESSION_CONTEXT.md` contents (if you haven't already)
2. Paste the exact stub function signature and docstring
3. Paste the relevant table definition from your schema

Example prompt structure (see [Part 4](#part-4-prompts-that-work) for templates):

> *[paste AI_SESSION_CONTEXT.md]*
>
> *Now implement this function. Match the signature exactly — do not change parameter names or return types:*
> *[paste stub function]*
>
> *My schema for the relevant tables:*
> *[paste CREATE TABLE statements]*

**Review the AI output before using it:**
- Does it use the table names from your schema? (not invented ones)
- Does it match the return type described in the docstring?
- Does it follow the `_connect()` / `RealDictCursor` pattern from `example_query()`?

See [Example 2](#example-2-implementing-a-relational-query-stub) in Part 3 for a full walkthrough.

### Stage 4 — Testing

**What you do:** Manually run the function and verify it returns what you expect.

You do not need a formal test framework. Open a Python shell:

```python
# From the project root, with your virtual environment active
python

>>> from databases.relational.queries import query_metro_schedules
>>> result = query_metro_schedules("MS01", "MS09")
>>> print(result)
>>> # Does it return a list? Does each item have the expected keys?
>>> # Is the result non-empty for a route that exists in your seed data?
```

**What to check:**
- Does it return a list (not None, not an error)?
- Does each dict have the keys the agent expects?
- For a station pair you know exists, does it return sensible results?
- For a station pair that doesn't exist, does it return an empty list (not crash)?

If the function raises an error, paste the error and your code back into the AI chat and ask it to fix the issue.

### Stage 5 — Merging

**What you do:** Get your work reviewed by a teammate and merge it.

1. Push your branch: `git push origin feature/alice/metro-schedules-query`
2. Open a Pull Request on GitHub
3. Ask a teammate to review — see [Example 4](#example-4-pr-review-and-merging) in Part 3
4. Address any feedback
5. Merge once approved
6. Update `AI_SESSION_CONTEXT.md` if any architectural decisions changed

**Update the main branch after merging:**
```bash
git checkout main
git pull origin main
```

---

## Part 3: Small Working Examples

### Example 1: Schema Design Workshop

**Scenario:** Your team is designing the `metro_stations` table from the mock data.

**Step 1 — Look at the mock data** (`train-mock-data/metro_stations.json`):

```json
{
  "station_id": "MS01",
  "name": "Central Square",
  "lines": ["M1", "M2"],
  "is_interchange_metro": true,
  "interchange_metro_lines": ["M1", "M2"],
  "is_interchange_national_rail": true,
  "interchange_national_rail_station_id": "NR01",
  "adjacent_stations": [
    { "station_id": "MS05", "line": "M1", "travel_time_min": 3 },
    { "station_id": "MS02", "line": "M1", "travel_time_min": 3 }
  ]
}
```

**Step 2 — Ask AI to propose a schema:**

> *"Here is one entry from our metro station data: [paste the JSON above]. Design a PostgreSQL schema to store this data. Note that `adjacent_stations` represents graph relationships — we have a separate Neo4j database for those, so you do not need to store them in SQL. Focus on what the relational database needs."*

**Step 3 — AI might produce something like:**

```sql
CREATE TABLE metro_stations (
    station_id          VARCHAR(10) PRIMARY KEY,
    name                TEXT NOT NULL,
    is_interchange_metro         BOOLEAN DEFAULT FALSE,
    is_interchange_national_rail BOOLEAN DEFAULT FALSE,
    interchange_nr_station_id    VARCHAR(10) REFERENCES national_rail_stations(station_id)
);

CREATE TABLE metro_station_lines (
    station_id  VARCHAR(10) REFERENCES metro_stations(station_id),
    line        VARCHAR(5) NOT NULL,
    PRIMARY KEY (station_id, line)
);
```

**Step 4 — Team discussion questions:**
- Do we need `metro_station_lines` as a separate table, or can we store lines as a simple array? (Hint: look at what queries will need to filter by line)
- Should `interchange_nr_station_id` be a foreign key constraint now, or added after both tables exist?
- What will `query_metro_schedules` need from this table?

**Human decision:** The team decides — AI proposes options. Normalization choices affect everyone's query functions, so everyone must agree.

---

### Example 2: Implementing a Relational Query Stub

**Scenario:** Alice is implementing `query_metro_schedules`.

**Step 1 — Alice reads the stub** (`databases/relational/queries.py`, lines 110–118):

```python
def query_metro_schedules(origin_id: str, destination_id: str) -> list[dict]:
    """
    Return metro schedules that serve both origin and destination in the correct order.

    Args:
        origin_id:       e.g. "MS01"
        destination_id:  e.g. "MS09"
    """
    raise NotImplementedError("TODO: implement after designing your schema")
```

**Step 2 — Alice prepares her prompt:**

```
[paste AI_SESSION_CONTEXT.md first]

Now implement this Python function. Rules:
- Use the _connect() helper and psycopg2.extras.RealDictCursor pattern shown in example_query()
- Match the stub's signature exactly — do not change parameter names or return types
- Use only table/column names from the schema below

Stub to implement:
[paste the stub above]

My schema (relevant tables):
CREATE TABLE metro_schedules (
    schedule_id  VARCHAR(20) PRIMARY KEY,
    line         VARCHAR(5) NOT NULL,
    direction    VARCHAR(10),
    stops        JSONB NOT NULL   -- ordered list of station_ids, e.g. ["MS01","MS02","MS09"]
);
```

**Step 3 — AI generates code. Alice checks:**
- Does it use `_connect()` from the module? ✓ or ✗
- Does it use `RealDictCursor`? ✓ or ✗
- Does it return `list[dict]`, not a single row? ✓ or ✗
- Does it reference `metro_schedules` (not an invented table name)? ✓ or ✗

**Step 4 — Alice tests it:**

```python
python

>>> from databases.relational.queries import query_metro_schedules
>>> result = query_metro_schedules("MS01", "MS09")
>>> print(type(result))      # should be <class 'list'>
>>> print(result)            # should show schedule dicts
>>> print(result[0].keys())  # check key names
```

---

### Example 3: Implementing a Graph Query Stub

**Scenario:** Bob is implementing `query_station_connections`.

**The stub** (`databases/graph/queries.py`, lines 159–166):

```python
def query_station_connections(station_id: str) -> list[dict]:
    """
    List all direct connections from a given station.

    Args:
        station_id: e.g. "MS01" or "NR01"
    """
    raise NotImplementedError("TODO: implement after designing your graph schema")
```

**Bob's prompt:**

```
[paste AI_SESSION_CONTEXT.md first]

Implement this Neo4j query function. Rules:
- Use the _driver() helper and the session pattern shown in example_count_nodes()
- Match the stub's signature exactly
- Use the node labels and relationship types from our agreed graph schema below

Stub to implement:
[paste stub above]

Our graph schema:
- Node label: Station, properties: {station_id, name, network}
- Relationship: CONNECTS_TO, properties: {line, travel_time_min}
```

**Bob checks the AI output:**
- Does it use `_driver()` from the module? ✓ or ✗
- Does it use `with driver.session() as session:`? ✓ or ✗
- Does the Cypher use `Station` as the node label (not `Node` or `stop`)? ✓ or ✗
- Does it return `list[dict]`? ✓ or ✗

**Bob tests it:**

```python
python

>>> from databases.graph.queries import query_station_connections
>>> result = query_station_connections("MS01")
>>> print(result)
>>> # MS01 (Central Square) connects to MS05, MS02, MS06, MS07 per the mock data
>>> # Check that your results match
```

---

### Example 4: PR Review and Merging

**Scenario:** Alice has pushed `feature/alice/metro-schedules-query` and opened a PR.

**Bob reviews the PR. He checks:**

1. Does the function match the stub's signature? (no extra or changed parameters)
2. Does it use table/column names from the agreed schema?
3. Does it follow the `_connect()` / `RealDictCursor` pattern?
4. Does it handle the empty-result case (no schedules found)?

**If Bob spots an issue**, he leaves a comment on GitHub:
> *"Line 45: your query uses `stations` but our schema calls this table `metro_stations`. Also the return dict is missing the `departure_time` key that `query_metro_fare` expects."*

**Alice fixes it**, pushes a new commit, and replies to the comment.

**After Bob approves**, Alice merges the PR:
- Click "Merge Pull Request" on GitHub
- Then locally: `git checkout main && git pull origin main`

---

### Example 5: Catching an AI Inconsistency

**Scenario:** Carol asks her AI to implement `query_national_rail_fare`. The AI generates:

```python
cur.execute("SELECT * FROM fares WHERE route_id = %s", (schedule_id,))
```

But the agreed schema has no `fares` table — the fare is calculated from `national_rail_schedules.base_fare_usd` and `national_rail_schedules.per_stop_rate_usd`.

**How to catch it:**
- The code runs, but returns `[]` or throws a `psycopg2.errors.UndefinedTable` error
- Carol compares the table name in the AI output against her schema — mismatch found

**Fix:** Carol updates her prompt to paste the exact `CREATE TABLE` statements and says:
> *"Do not invent table or column names. Use only what appears in the schema below."*

**Lesson:** Always paste your schema into the AI prompt. AI will make up plausible-sounding names if you don't give it the real ones.

---

## Part 4: Prompts That Work

These are tool-agnostic templates. Paste them into any AI assistant (Claude, Copilot, Cursor, Gemini, etc.).

### Template A: Schema Design

```
I'm a student working on a database project. Here is one sample entry from our
raw data file [filename]:

[paste 1–3 JSON objects from the mock data]

Design a PostgreSQL schema to store this data. Constraints:
- Use snake_case for all table and column names
- Use VARCHAR for IDs (they look like "MS01", "NR_SCH01")
- Avoid storing graph/network relationships (those go in Neo4j)
- Include PRIMARY KEY and NOT NULL where appropriate
- Show the CREATE TABLE statement only, no explanation

Note: this schema will be shared with two teammates. Table names must be agreed
before anyone writes query functions.
```

### Template B: Query Function Implementation

```
I'm implementing a Python function for a PostgreSQL database project.
Follow these rules strictly:
- Use only the table and column names in the schema below — do not invent names
- Use the _connect() helper function already defined in the module
- Use psycopg2.extras.RealDictCursor (so rows come back as dicts)
- Match the stub signature exactly — do not change parameter names or return type
- Return an empty list [] (not None) when no rows are found
- Do not add try/except unless the docstring specifically asks for error handling

[paste AI_SESSION_CONTEXT.md here]

Stub to implement:
[paste the stub function with its docstring]

Schema (relevant tables only):
[paste the CREATE TABLE statements your function will query]
```

### Template C: Code Review

```
Review this Python database function against the stub contract and schema below.
Check for:
1. Does it use only table/column names from the schema?
2. Does it match the stub's return type and key names?
3. Does it follow the _connect() / RealDictCursor pattern?
4. Does it handle the empty-result case gracefully?
5. Any SQL injection risk (are all user inputs parameterised with %s)?

Report only real issues — no style suggestions.

Stub (the contract):
[paste the original stub]

Implementation to review:
[paste your code]

Schema:
[paste relevant CREATE TABLE statements]
```

### Template D: Debugging

```
This Python function is raising an error. Help me fix it.

Error:
[paste the full traceback]

Function:
[paste your code]

Schema:
[paste relevant CREATE TABLE statements]

What I expected it to do:
[one sentence]
```

### How to Share Prompts That Worked

When you find a prompt that produces good output, add it to the **Prompts log** section of `AI_SESSION_CONTEXT.md`. Your teammates can reuse it instead of spending time writing their own.

---

## Appendix: Pre-Session Checklist

Run through this before every AI-assisted work session.

```
[ ] git checkout main && git pull origin main
[ ] Check GitHub for open Pull Requests — anything needing your review?
[ ] Confirm Docker containers are running: docker compose ps
    (should show postgres, neo4j, pgadmin as "Up")
[ ] Confirm your virtual environment is active: python -c "import psycopg2; print('ok')"
[ ] Open AI_SESSION_CONTEXT.md and paste its contents into your AI chat
[ ] Tell your teammates what you're about to work on
```

If Docker isn't running: `docker compose up -d` from the project root.

If your venv is missing: see the [Python Virtual Environments](README.md#python-virtual-environments) section of README.md.

---

## Quick Reference

| Question | Where to look |
|---|---|
| What functions do I need to implement? | `databases/relational/queries.py`, `databases/graph/queries.py` — read the stubs and docstrings |
| What data do I have to work with? | `train-mock-data/` — JSON files for every entity |
| What does the agent call my function with? | `skeleton/agent.py` — the `TOOLS` list shows the exact parameters |
| Where do I design the schema? | `databases/relational/schema.sql` — currently empty, you fill it in |
| What do I paste into AI at the start? | `AI_SESSION_CONTEXT.md` — the shared context file |
| Generic team practices and checklists | `TEAM_PROJECT_GUIDE.md` |
