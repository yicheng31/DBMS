# TransitFlow 團隊 AI 規範

本文件規範本團隊在 TransitFlow 專案中使用 AI 工具的方式。適用工具包含 Codex、ChatGPT、GitHub Copilot、Cursor、Gemini 或其他 AI coding assistant。

AI 可以協助分析、產生範例、撰寫初稿、除錯與 review，但不得取代團隊對資料庫設計、系統架構與程式正確性的判斷。

## Project Overview

TransitFlow 是一個鐵路與捷運 AI 助理系統。使用者可以詢問班次、票價、座位、訂票、取消訂票、路線規劃與補償政策等問題。

本專案的核心目標是設計並實作支援 AI 助理的資料庫層。AI 助理本身不直接操作資料庫，而是透過 Python query functions 呼叫 PostgreSQL、pgvector 與 Neo4j。

資料庫責任分工如下：

| Component | Responsibility |
|---|---|
| PostgreSQL | 使用者、憑證、票種、班次、座位、訂票、付款、回饋、政策等結構化與交易資料 |
| PostgreSQL + pgvector | 政策文件 embedding、語意搜尋與 RAG |
| Neo4j | 站點連線、轉乘、最短路徑、替代路線、延誤影響範圍 |

重要原則：

- PostgreSQL 負責 relational / transactional data。
- Neo4j 負責 graph traversal / route planning。
- pgvector 負責 policy semantic search。
- AI 不得混淆以上責任。
- AI 不得自行發明新的 schema、graph model 或 function contract。

## Tech Stack

本專案使用以下技術：

| Area | Technology |
|---|---|
| Language | Python 3.11+ |
| Relational DB | PostgreSQL |
| Vector DB | PostgreSQL pgvector extension |
| Graph DB | Neo4j |
| SQL Driver | psycopg2 |
| Graph Driver | Neo4j Python Driver |
| UI | Gradio |
| LLM Provider | Gemini or Ollama |
| Environment | Docker, `.env`, `.env.example` |
| Version Control | Git and GitHub |

AI 產生的程式碼必須維持在上述 tech stack 內。

除非團隊明確決定，AI 不得新增：

- ORM
- backend framework
- frontend framework
- new database
- external service
- unnecessary package

## Coding Convention

### General

- 遵守現有專案結構。
- 不任意 rename file、function、table、column、label 或 relationship。
- 不修改不相關檔案。
- 一次只處理一個明確任務。
- 保持程式簡單、可讀，適合 database management course project。
- 若 schema 或需求不足，AI 應先指出問題，不得自行補 invent。

### Python

- Follow PEP 8。
- 使用 `snake_case` 命名 function、variable、file。
- 使用 type hints。
- 保留既有 function signature。
- query functions 回傳 `list[dict]`、`dict`、`Optional[dict]`、`bool` 或 docstring 指定的格式。
- 查無資料時依照 contract 回傳 `[]` 或 `None`，不要把正常查無資料當成 exception。

### PostgreSQL / SQL

AI 寫 SQL 前必須先參考：

- `databases/relational/schema.sql`
- `databases/relational/queries.py`
- relevant seed data in `train-mock-data/`

SQL 規範：

- 使用實際存在的 table / column。
- 使用 `%s` placeholders。
- 不得把使用者輸入直接 string concatenation 到 SQL。
- 使用 explicit columns，避免正式 query 使用 `SELECT *`。
- 保留 primary key、foreign key、unique、check constraints。
- booking、cancellation、payment 等寫入操作必須考慮 transaction。

PostgreSQL 適合處理：

- 使用者資料
- 登入與註冊
- 班次與票價
- 座位與訂票
- 付款與取消訂票
- feedback
- policy metadata

### Neo4j / Cypher

AI 寫 Cypher 前必須先參考：

- `skeleton/seed_neo4j.py`
- `databases/graph/queries.py`

Cypher 規範：

- 使用 Cypher parameters。
- 不得把使用者輸入直接 string concatenation 到 Cypher。
- node label 使用 `PascalCase`。
- relationship type 使用 `UPPER_SNAKE_CASE`。
- 回傳 plain Python `dict` 或 `list[dict]`。

Neo4j 適合處理：

- shortest route
- cheapest route approximation
- alternative routes
- interchange path
- delay ripple
- station connections

### Cross-Database Rules

- 不要用 PostgreSQL joins 實作 route planning。
- 不要把所有 PostgreSQL tables 複製到 Neo4j。
- 不要用 Neo4j 處理使用者密碼、付款、訂票交易資料。
- PostgreSQL 與 Neo4j 的 station IDs 必須一致。
- SQL 只用於 PostgreSQL。
- Cypher 只用於 Neo4j。

## Examples

### Good AI Request

```text
Read AI_SESSION_CONTEXT.md first.

I need to implement query_metro_schedules(origin_id, destination_id).
Before writing code, explain which PostgreSQL tables are needed, how to check that origin appears before destination, and what edge cases I should test.
Do not write code yet.
```

Why this is good:

- It gives AI project context.
- It asks for analysis before implementation.
- It limits scope to one function.
- It prevents AI from jumping directly into invented code.

### Bad AI Request

```text
Help me finish all database queries.
```

Why this is bad:

- Scope is too broad.
- AI may invent table names.
- AI may change function signatures.
- AI may mix PostgreSQL and Neo4j responsibilities.

### Good Implementation Request

```text
Implement only query_available_seats in databases/relational/queries.py.

Constraints:
- Keep the existing function signature unchanged.
- Use _connect() and RealDictCursor.
- Use %s placeholders.
- Match databases/relational/schema.sql exactly.
- Return the shape described in the docstring.
- Do not modify unrelated functions.
```

### Good Review Request

```text
Review this function for:
- schema mismatch
- SQL injection risk
- wrong return type
- function signature changes
- behavior that violates AI_SESSION_CONTEXT.md

List only actionable issues.
```

## Analysis Prompt

Use this prompt before asking AI to write code:

```text
Read AI_SESSION_CONTEXT.md first and follow it as the project contract.

I need to work on:
<function_or_file_name>

Before writing code, analyze:
1. Whether this belongs to PostgreSQL, pgvector, or Neo4j
2. Which files should be inspected first
3. Which tables, columns, labels, or relationships are involved
4. What the expected return format should be
5. What edge cases should be tested
6. Whether any team decision is needed before implementation

Do not write implementation code yet.
```

## Review Prompt

Use this prompt after AI or a teammate produces code:

```text
Review the following changes against AI_SESSION_CONTEXT.md and this project's existing schema.

Focus only on actionable problems:
- incorrect table or column names
- incorrect Neo4j labels or relationships
- SQL or Cypher injection risk
- changed function signatures
- wrong return shape
- missing empty-result handling
- transaction problems
- unnecessary dependencies
- unrelated file changes
- code that mixes PostgreSQL and Neo4j responsibilities incorrectly

Do not rewrite the whole file. Suggest the smallest safe fix for each issue.
```

## Prohibitions

The following are prohibited when using AI in this project:

- Do not accept AI-generated code without reading it.
- Do not let AI rename existing functions, files, tables, columns, labels, or relationships.
- Do not let AI change function signatures unless the team explicitly agrees.
- Do not let AI invent missing schema.
- Do not let AI add new frameworks, databases, ORMs, or external services.
- Do not commit `.env`, API keys, database passwords, tokens, or secrets.
- Do not concatenate user input into SQL or Cypher.
- Do not use PostgreSQL for route traversal logic that belongs in Neo4j.
- Do not use Neo4j for transactional booking, payment, or authentication logic.
- Do not modify vector/RAG behavior unless the task specifically involves policy search.
- Do not ask AI to rewrite large unrelated parts of the project.
- Do not mark AI-generated code as done before testing or manually checking it.

## Definition of Done

AI-assisted work is complete only when:

- The implementation follows `AI_SESSION_CONTEXT.md`.
- The function signature is unchanged.
- SQL or Cypher matches the actual schema.
- Dynamic values use parameters.
- Return values match the docstring.
- Empty or not-found cases are handled.
- The code has been manually tested or reviewed with seed data assumptions.
- No secrets are committed.
- Any new team decision is recorded in `AI_SESSION.md`.