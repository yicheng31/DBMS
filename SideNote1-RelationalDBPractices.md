# Side Note 1 — Relational Database Best Practices in Production

> **Disclaimer**
> This document was co-authored with the assistance of multiple AI tools. While every effort has been made to ensure the accuracy of the content, some unintended errors may still be present. If you spot any mistakes, please [submit an issue on GitHub](https://github.com/NCUIM-Lab710-Teaching/IM2002-DBMGT-Train-v2/issues).

---

> **Who is this for?**
> This note is for students who have just started working with databases.
> You have seen SQL, you have run some queries in Python — now let's look at how real production systems handle this properly.

---

## Why Does This Matter?

The code in `databases/relational/queries.py` works perfectly well for a teaching environment. But production systems — apps used by real people at scale — have stricter requirements around **performance**, **security**, and **maintainability**. This note walks through each gap, explains *why* it exists, and shows you what the production version looks like.

---

## 1. Connection Pooling

### What is a database connection?

Every time your Python code talks to PostgreSQL, it first has to open a **connection** — a dedicated communication channel between your app and the database server. This involves a TCP handshake, authentication, and resource allocation on both sides.

In the teaching code, every query function does this:

```python
def _connect():
    conn = psycopg2.connect(PG_DSN)  # opens a brand new connection every time
    conn.autocommit = True
    return conn
```

This means if 100 users search for train seats at the same time, your app tries to open 100 separate connections simultaneously. PostgreSQL has a default limit of around 100 connections — your app would start refusing users.

### The production solution: Connection Pools

A **connection pool** keeps a set of connections open and ready. Instead of opening and closing one per query, your code borrows a connection from the pool, uses it, and returns it.

```python
from psycopg2 import pool

# Created once when the app starts — keeps 2 to 10 connections ready
_pool = pool.ThreadedConnectionPool(minconn=2, maxconn=10, dsn=PG_DSN)

def query_national_rail_availability(origin_id, destination_id, travel_date=None):
    conn = _pool.getconn()       # borrow a connection
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (origin_id, destination_id))
            return cur.fetchall()
    finally:
        _pool.putconn(conn)      # always return it, even if an error occurs
```

For very large-scale apps, a separate tool called **PgBouncer** sits between your app and PostgreSQL and manages thousands of connections more efficiently than any in-process pool can.

### Learn more
- [psycopg2 Connection Pools (official docs)](https://www.psycopg.org/docs/pool.html)
- [PgBouncer — Official site and documentation](https://www.pgbouncer.org/)
- [What is Connection Pooling? (CockroachDB blog)](https://www.cockroachlabs.com/blog/what-is-connection-pooling/)

---

## 2. How SQL Is Organised

The teaching code stores SQL as inline strings inside each Python function. This is fine for learning, but production teams have stronger conventions.

### Option A — ORM (Object-Relational Mapper)

An ORM lets you write Python objects instead of raw SQL. The library translates your Python into SQL automatically.

The most popular Python ORM is **SQLAlchemy**.

```python
# Instead of writing SQL, you define Python classes that map to tables
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import select

# Step 1: create a shared base — all models inherit from this
class Base(DeclarativeBase):
    pass

# Step 2: define your model by inheriting from Base
class TrainService(Base):
    __tablename__ = "train_services"

    id: Mapped[int] = mapped_column(primary_key=True)
    service_code: Mapped[str]
    available_seats: Mapped[int]

# Then query using Python — no SQL string at all
stmt = select(TrainService).where(TrainService.available_seats > 0)
services = session.execute(stmt).scalars().all()
```

**Pros:** Type-safe, editor autocomplete, database-agnostic (you can switch from PostgreSQL to MySQL with one config line), built-in migration support.

**Cons:** Hides SQL from you (a problem when learning), can produce inefficient queries for complex joins.

### Option B — Separate `.sql` Files

Another pattern used in data-heavy systems is to keep SQL in dedicated files:

```
databases/
  relational/
    queries/
      seat_availability.sql
      user_bookings.sql
      ticket_prices.sql
```

A library called **aiosql** loads these files at startup and exposes them as Python functions automatically. DBAs (database administrators) can tune the SQL without touching Python code, and you can test queries directly in pgAdmin.

### Option C — Query Builder (SQLAlchemy Core)

A middle ground — you write something *close* to SQL but in a structured, composable Python style:

```python
from sqlalchemy import select, and_

stmt = (
    select(train_services, stations)
    .join(stations, train_services.c.origin_id == stations.c.id)
    .where(
        and_(
            stations.c.code == origin_code,
            train_services.c.available_seats > 0
        )
    )
)
```

### Learn more
- [SQLAlchemy ORM Tutorial (official)](https://docs.sqlalchemy.org/en/20/orm/quickstart.html)
- [aiosql — SQL in .sql files for Python](https://nackjicholson.github.io/aiosql/)
- [SQLAlchemy Core Tutorial (official)](https://docs.sqlalchemy.org/en/20/core/tutorial.html)
- [Full Stack Python — SQLAlchemy overview](https://www.fullstackpython.com/sqlalchemy.html)

---

## 3. Asynchronous Database Access

### What does "synchronous" mean here?

In the teaching code, when a query runs, Python **stops and waits** for PostgreSQL to respond before doing anything else. This is called **blocking** or synchronous I/O.

For a single user, this is fine. For a web API serving hundreds of users, it means one slow database query can hold up everyone else waiting behind it.

### The production solution: Async I/O

Modern Python web frameworks like **FastAPI** are built around `async`/`await`. When combined with an async database driver like **asyncpg**, your app can handle many requests concurrently without waiting:

```python
import asyncpg

async def query_seat_availability(origin_code: str, dest_code: str):
    async with pool.acquire() as conn:          # async pool — does not block
        rows = await conn.fetch(sql, origin_code, dest_code)
        return [dict(row) for row in rows]
```

Think of it like a waiter in a restaurant. A synchronous waiter takes one order, walks to the kitchen, stands there until the food is ready, then comes back. An async waiter takes many orders, drops them all at the kitchen, and handles whichever is ready first.

### Learn more
- [asyncpg — Fast PostgreSQL client for Python (GitHub)](https://github.com/MagicStack/asyncpg)
- [FastAPI with Databases tutorial (official FastAPI docs)](https://fastapi.tiangolo.com/tutorial/sql-databases/)
- [Real Python — Async IO in Python](https://realpython.com/async-io-python/)

---

## 4. Password Security

### What is wrong with plain-text passwords?

The teaching code stores and checks passwords exactly as the user typed them:

```python
# In register_user()
INSERT INTO users (..., password, ...) VALUES (..., %s, ...)

# In login_user()
WHERE u.email = %s AND u.password = %s
```

If an attacker ever gains read access to your database (via SQL injection, a backup leak, or a misconfigured cloud bucket), every user's password is immediately exposed — including any other site where they reused that password.

### The production solution: Password Hashing

Passwords should be passed through a **one-way hashing function** before storage. The hash cannot be reversed back into the original password. When a user logs in, you hash what they typed and compare the two hashes — you never compare plain text.

```python
from argon2 import PasswordHasher

ph = PasswordHasher()

# On registration — store the hash, not the password
hashed = ph.hash(plain_password)
# e.g. "$argon2id$v=19$m=65536,t=3,p=4$..."

# On login — verify the input against the stored hash
try:
    ph.verify(stored_hash, input_password)  # raises exception if wrong
    return True
except Exception:
    return False
```

**argon2** is the current gold standard, recommended by OWASP (the Open Web Application Security Project). **bcrypt** is also widely used and acceptable.

### Why not just use Python's built-in `hashlib`?

Functions like `hashlib.sha256()` are designed to be *fast* — great for checksums, bad for passwords. Attackers can test billions of guesses per second against a SHA-256 hash. Argon2 and bcrypt are deliberately slow and memory-intensive, making brute-force attacks impractical.

### Learn more
- [OWASP Password Storage Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html)
- [argon2-cffi — Python library (docs)](https://argon2-cffi.readthedocs.io/en/stable/)
- [Python hashlib module — official docs (explains why it is not suited for passwords)](https://docs.python.org/3/library/hashlib.html)

---

## 5. The Repository Pattern

### What is the problem?

In the teaching code, the agent calls query functions directly:

```python
# Inside agent.py
results = query_user_bookings(user_email)
```

This is fine, but it tightly **couples** your business logic to your database layer. If you ever want to:
- Swap PostgreSQL for a different database
- Write automated tests without a real database
- Change how bookings are fetched without touching every caller

...you would have to find and edit every place that calls `query_user_bookings`.

### The production solution: Repository classes

A **repository** is a class that owns all the database operations for one domain concept. The rest of the app only talks to the repository, never to the database directly.

```python
class BookingRepository:
    def __init__(self, session):
        self.session = session

    def get_by_user(self, user_email: str) -> list[dict]:
        # all the SQL lives here
        ...

    def create(self, user_id: int, service_id: int, travel_date: str) -> dict:
        ...

# In tests, you can replace it with a fake version
class FakeBookingRepository:
    def get_by_user(self, user_email: str) -> list[dict]:
        return [{"booking_ref": "TEST001", ...}]   # no database needed
```

This follows a software design principle called **Separation of Concerns** — each part of the code has one clear responsibility.

### Learn more
- [Martin Fowler — Repository Pattern (reference)](https://martinfowler.com/eaaCatalog/repository.html)
- [ArjanCodes — Repository Pattern in Python (YouTube)](https://www.youtube.com/watch?v=9pymbjfqfNs)
- [Cosmic Python — Repository Pattern (free book)](https://www.cosmicpython.com/book/chapter_02_repository.html)

---

## 6. Database Migrations

### What is the problem?

In development, when you want to add a new column to a table, you might drop the whole database and re-run `schema.sql`. In production, **you cannot do this** — the database holds real user data that cannot be deleted.

### The production solution: Migration tools

A **migration** is a versioned script that describes *one incremental change* to the schema. Every change — adding a column, creating a table, adding an index — gets its own migration file.

A migration tool (**Alembic** for SQLAlchemy, **Flyway** for Java, **Django migrations** for Django) tracks which scripts have already run and only applies new ones. Every environment — developer laptop, staging server, production server — runs the exact same migration history and ends up with the identical schema.

Each tool uses its own file format. Flyway uses plain `.sql` files with a version prefix:

```
migrations/
  V1__initial_schema.sql          ← creates all base tables
  V2__add_delay_records.sql       ← adds the delay_records table
  V3__add_user_accounts.sql       ← adds the user_accounts table
  V4__add_railcard_expiry.sql     ← adds a new column to users
```

Alembic (the Python/SQLAlchemy tool) generates Python scripts instead, stored in a `versions/` folder with auto-generated filenames:

```
alembic/versions/
  a1b2c3d4_initial_schema.py
  e5f6a7b8_add_delay_records.py
```

Both approaches use the same CLI-driven workflow:

```bash
# Apply all pending migrations
alembic upgrade head

# Roll back the last migration if something went wrong
alembic downgrade -1
```

### Learn more
- [Alembic Tutorial (official)](https://alembic.sqlalchemy.org/en/latest/tutorial.html)
- [Alembic — Auto Generating Migrations (official)](https://alembic.sqlalchemy.org/en/latest/autogenerate.html)
- [Flyway — Database migrations documentation (Redgate)](https://documentation.red-gate.com/fd)

---

## Summary

| Topic | Teaching Code | Production Approach |
|---|---|---|
| **Connections** | New connection per query | Connection pool (psycopg2 pool / PgBouncer) |
| **SQL location** | Inline strings in functions | ORM, `.sql` files, or query builder |
| **I/O model** | Synchronous (blocking) | Async (`asyncpg` + `async`/`await`) |
| **Passwords** | Plain text | `argon2` or `bcrypt` hash |
| **DB layer structure** | Standalone functions | Repository pattern (classes) |
| **Schema changes** | Drop and recreate | Versioned migrations (Alembic / Flyway) |

None of these make the teaching code "wrong" — they solve problems that only appear at scale or in security-sensitive contexts. Understanding *why* each practice exists is more important than memorising the tools.

---

## Recommended Starting Points

If you want to go further after this course, these free resources are a good next step:

| Resource | What you will learn |
|---|---|
| [SQLAlchemy Tutorial (official)](https://docs.sqlalchemy.org/en/20/orm/quickstart.html) | ORM and query builder from scratch |
| [FastAPI SQL Databases Guide](https://fastapi.tiangolo.com/tutorial/sql-databases/) | Connecting a real async API to PostgreSQL |
| [Cosmic Python (free book)](https://www.cosmicpython.com/) | Architecture patterns including Repository and Unit of Work |
| [OWASP Top 10](https://owasp.org/www-project-top-ten/) | The ten most common web security mistakes (SQL injection is #3) |
| [Real Python — Databases](https://realpython.com/tutorials/databases/) | Practical Python + database tutorials at all levels |
| [PostgreSQL Official Docs](https://www.postgresql.org/docs/current/) | The authoritative reference for everything PostgreSQL |
