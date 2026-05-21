# Side Note 3 — Graph Database Best Practices in Production

> **Disclaimer**
> This document was co-authored with the assistance of multiple AI tools. While every effort has been made to ensure the accuracy of the content, some unintended errors may still be present. If you spot any mistakes, please [submit an issue on GitHub](https://github.com/NCUIM-Lab710-Teaching/IM2002-DBMGT-Train-v2/issues).

---

> **Who is this for?**
> This note is for students who have worked with Neo4j in this project.
> You have written Cypher queries to find train routes and model station connections — now let's look at how production systems manage graph databases properly.

---

## What Does the Teaching Code Do?

The TransitFlow project uses **Neo4j** to model the physical rail network. Stations are **nodes** and rail connections between them are **relationships**. Cypher queries find the shortest path, avoid closed stations, and ripple delay information through the network.

The teaching code creates a new driver for every query like this:

```python
def _driver():
    """Return a Neo4j driver. Caller is responsible for closing."""
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def query_shortest_route(origin_id: str, destination_id: str, network: str = "auto") -> dict:
    with _driver() as driver:           # new driver created here
        with driver.session() as session:
            result = session.run(cypher, ...)
    # driver closed here — connection pool torn down and rebuilt next call
```

This is functional for a single-user teaching environment. In production, it has several problems that this note explains and fixes.

---

## 1. Driver Management: One Driver for the Whole App

### What is the Neo4j driver?

The Neo4j Python driver is not just a connection — it is a **connection pool manager**. When you create a driver with `GraphDatabase.driver(...)`, it:
- Opens a pool of TCP connections to Neo4j
- Manages routing for clustered Neo4j deployments
- Handles authentication and TLS

### The problem with the teaching code

Every query function calls `_driver()`, which creates and then immediately destroys a fresh pool. This is the same problem as opening a new `psycopg2.connect()` per query — expensive setup cost on every call.

### The production solution: a singleton driver

Create the driver **once** when the application starts, and reuse it for every query:

```python
from neo4j import GraphDatabase
from skeleton.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# Created once at module load — shared across all queries
_DRIVER = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD),
    max_connection_pool_size=50,     # how many connections to keep open
)

def query_shortest_route(origin_id: str, destination_id: str, network: str = "auto") -> dict:
    with _DRIVER.session() as session:    # borrows from the pool — no new driver
        result = session.run(cypher, ...)
        return dict(result.single()) if result else {"found": False}

# In a web framework, close the driver when the app shuts down:
# _DRIVER.close()
```

The driver's built-in pool means you don't need an external tool like PgBouncer — Neo4j's Python driver handles connection reuse natively.

### Learn more
- [Neo4j Python Driver — Connection and authentication (official docs)](https://neo4j.com/docs/python-manual/current/)
- [Neo4j Python Driver — Advanced connection options](https://neo4j.com/docs/python-manual/current/connect-advanced/)

---

## 2. Explicit Transactions

### What the teaching code does

Every query uses `session.run()` directly, which runs in **auto-commit mode**. Each statement is its own transaction and either fully succeeds or fails independently.

For read queries this is acceptable. For write operations — like inserting a new booking or updating a station status — auto-commit gives you no way to rollback multiple related changes if one of them fails partway through.

### The production solution: explicit transactions

The Neo4j driver has three transaction modes:

#### Auto-commit (teaching code — fine for reads)
```python
session.run(cypher, params)
```

#### Managed transactions (recommended for writes)
```python
def create_station_tx(tx, station_id, name, lines):
    tx.run(
        "MERGE (s:MetroStation {station_id: $station_id}) SET s.name = $name, s.lines = $lines",
        station_id=station_id, name=name, lines=lines,
    )

with _DRIVER.session() as session:
    session.execute_write(create_station_tx, "MS21", "New Quarter", ["M2"])
    # automatically retried on transient errors, rolled back on exceptions
```

#### Explicit transactions (for complex multi-step operations)
```python
with _DRIVER.session() as session:
    with session.begin_transaction() as tx:
        tx.run("MATCH (s:NationalRailStation {station_id: $sid}) SET s.active = false", sid="NR03")
        tx.run("MATCH (:NationalRailStation {station_id: $sid})-[r:RAIL_LINK]-() SET r.active = false", sid="NR03")
        tx.commit()   # both changes committed together, or neither if an error occurs
```

### Learn more
- [Neo4j Python Driver — Transactions (official docs)](https://neo4j.com/docs/python-manual/current/transactions/)

---

## 3. Indexes and Constraints

### The problem

The teaching code runs queries like:

```cypher
MATCH (s:Station {code: $code}) ...
```

Without an index on `Station.code`, Neo4j must scan **every Station node** to find the one with the matching code. This is fast with 20 stations; it is unacceptably slow with 20,000.

### The production solution: define indexes and constraints

In production, you define indexes and constraints in your schema setup scripts — not in your query code.

#### Unique constraint (also creates an index)
```cypher
// Run these once when setting up the database
CREATE CONSTRAINT metro_station_id_unique
FOR (s:MetroStation)
REQUIRE s.station_id IS UNIQUE;

CREATE CONSTRAINT nr_station_id_unique
FOR (s:NationalRailStation)
REQUIRE s.station_id IS UNIQUE;
```

After this, `MATCH (s:MetroStation {station_id: "MS01"})` uses the index and runs in O(1) time regardless of how many stations exist.

#### Range index (for non-unique properties)
```cypher
CREATE INDEX metro_station_name_index
FOR (s:MetroStation)
ON (s.name);
```

#### Relationship index
```cypher
CREATE INDEX metro_link_line_index
FOR ()-[r:METRO_LINK]-()
ON (r.line);
```

### Learn more
- [Neo4j Cypher Manual — Indexes](https://neo4j.com/docs/cypher-manual/current/indexes/)
- [Neo4j Cypher Manual — Constraints](https://neo4j.com/docs/cypher-manual/current/constraints/)

---

## 4. Graph Data Modelling

### What is graph data modelling?

When you design a relational database, you think in tables and foreign keys. When you design a graph database, you think in **nodes**, **relationships**, and **properties**.

The most important decision in graph modelling is: **what should be a node, and what should be a relationship?**

#### The TransitFlow model (simplified)
```
(:MetroStation {station_id, name, lines[]})
    -[:METRO_LINK {line, travel_time_min, base_fare_usd, per_stop_rate_usd}]->
(:MetroStation {station_id, name, lines[]})

(:NationalRailStation {station_id, name, lines[]})
    -[:RAIL_LINK {line, travel_time_min, standard_fare_usd, first_fare_usd}]->
(:NationalRailStation {station_id, name, lines[]})

(:MetroStation)-[:INTERCHANGE_TO {transfer_time_min}]->(:NationalRailStation)
```

This is a good model because:
- Stations are entities (they have their own identity and properties)
- Transit links are relationships (they only exist *between* two stations)
- Fare data is stored on edges, enabling cost-weighted Dijkstra without joining PostgreSQL
- Route-finding algorithms traverse relationships naturally

#### A common modelling mistake: making relationships into nodes

A beginner might model an interchange like this:

```
(:MetroStation)-[:HAS_INTERCHANGE]->(:Interchange)-[:CONNECTS_TO]->(:NationalRailStation)
```

But if an interchange simply connects two stations, a direct relationship is cleaner:

```
(:MetroStation)-[:INTERCHANGE_TO {transfer_time_min: 5}]->(:NationalRailStation)
```

Rule of thumb: **if something connects exactly two things and has properties, it is a relationship, not a node.**

### Learn more
- [Neo4j — What is a graph database? (Getting Started)](https://neo4j.com/docs/getting-started/graph-database/)
- [Neo4j — Graph data modelling guide](https://neo4j.com/docs/getting-started/data-modeling/)

---

## 5. Graph Algorithms (GDS Plugin)

### What the teaching code uses

The teaching code uses Cypher's built-in `shortestPath()`, which finds the path with the fewest hops. This is correct for simple route finding.

```cypher
MATCH path = shortestPath((start)-[:RAIL_LINK*]-(end))
```

### What production systems add: the Graph Data Science library

Neo4j's **Graph Data Science (GDS)** plugin provides a library of advanced graph algorithms that are far more powerful than what Cypher expressions alone can compute:

#### Weighted shortest path (Dijkstra's algorithm)
The built-in `shortestPath()` counts hops. Dijkstra's finds the path that minimises a numeric weight — such as travel time or fare. The TransitFlow project uses the **APOC** plugin (`apoc.algo.dijkstra`) for this, which must be enabled in `docker-compose.yml`. GDS provides a more feature-complete alternative:

```cypher
MATCH (start:MetroStation {station_id: 'MS01'}),
      (end:MetroStation   {station_id: 'MS14'})
CALL gds.shortestPath.dijkstra.stream('metro-network', {
    sourceNode: start,
    targetNode: end,
    relationshipWeightProperty: 'travel_time_min'
})
YIELD path
RETURN path
```

#### PageRank — find the most important stations
Identifies which stations are most central to the network (high-traffic interchange hubs):

```cypher
CALL gds.pageRank.stream('metro-network')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS station, score
ORDER BY score DESC
LIMIT 10
```

#### Community detection — find natural clusters
Groups stations that are more connected to each other than to the rest of the network — useful for identifying line clusters or service zones:

```cypher
CALL gds.louvain.stream('metro-network')
YIELD nodeId, communityId
RETURN gds.util.asNode(nodeId).name AS station, communityId
ORDER BY communityId
```

These algorithms would be **extremely complex to reproduce in SQL** — they are the core reason to choose a graph database for network-type problems.

### Learn more
- [Neo4j Graph Data Science library — official documentation](https://neo4j.com/docs/graph-data-science/current/)

---

## 6. Cypher Query Organisation

### The same problem as SQL

Just as the relational `queries.py` stores SQL inline, the teaching graph code stores Cypher inline as strings. At scale, the same alternatives apply:

- **Centralise queries** in a dedicated module (already done well in `databases/graph/queries.py`)
- **Repository pattern** — wrap query functions in a class so tests can swap in a fake
- **Parameterise everything** — the teaching code already does this correctly (uses `$param` syntax, never string formatting)

The teaching code already avoids the biggest Cypher security risk: **Cypher injection**. Never format values directly into a Cypher string:

```python
# DANGEROUS — never do this
cypher = f"MATCH (s:MetroStation {{station_id: '{user_input}'}}) RETURN s"

# SAFE — always use parameters
cypher = "MATCH (s:MetroStation {station_id: $station_id}) RETURN s"
session.run(cypher, station_id=user_input)
```

### Learn more
- [Neo4j Cypher Manual — Introduction](https://neo4j.com/docs/cypher-manual/current/)

---

## 7. When to Use a Graph Database vs a Relational Database

This is one of the most important questions in database design. A graph database is the right choice when:

| Scenario | Why graph wins |
|---|---|
| **Route finding** (shortest path, avoid a node) | Graph traversal is native; SQL requires recursive CTEs |
| **Ripple / impact analysis** (delay at one station affects others) | N-hop traversal is one line of Cypher; very slow in SQL |
| **Recommendations** (people who took this route also took...) | Relationship patterns across millions of nodes are fast |
| **Fraud detection** (shared accounts, common addresses) | Detecting connected subgraphs is core graph theory |
| **Knowledge graphs** (entities and their relationships) | Flexible schema is a natural fit |

A relational database is still the right choice for:
- Tabular, structured data with clear schema (bookings, pricing, users)
- Aggregation-heavy queries (SUM, GROUP BY, window functions)
- Transactions that modify many rows at once

**TransitFlow uses both for a reason**: the route network is a graph problem; the booking history is a relational problem. Using the right tool for each is exactly what a real production system does.

---

## 8. Alternatives to Neo4j

Neo4j is the most widely-used graph database, but it is not the only option:

| Database | Key difference |
|---|---|
| **Amazon Neptune** | Fully managed on AWS; supports both property graphs and RDF (knowledge graphs) |
| **ArangoDB** | Multi-model: graph + document + key-value in one engine |
| **TigerGraph** | Built for very large-scale graphs (billions of edges); used in fraud detection |
| **Apache AGE** | Graph extension for PostgreSQL (like pgvector but for graphs) |

For most applications that do not require AWS lock-in, Neo4j Community Edition (free and open-source) is the standard starting point.

### Learn more
- [Amazon Neptune — official page](https://aws.amazon.com/neptune/)
- [ArangoDB — official site](https://arango.ai/)

---

## Summary

| Topic | Teaching Code | Production Approach |
|---|---|---|
| **Driver lifecycle** | New driver per query | Singleton driver, shared across app |
| **Transactions** | Auto-commit via `session.run()` | Managed or explicit transactions for writes |
| **Indexes** | None defined | `CREATE CONSTRAINT` / `CREATE INDEX` in setup scripts |
| **Path algorithms** | Built-in `shortestPath()` (hop count) | GDS Dijkstra (weighted), PageRank, community detection |
| **Cypher location** | Inline strings in functions | Centralised module + repository pattern |
| **Security** | Parameterised (correct) | Parameterised — never string-format user input into Cypher |

---

## Recommended Starting Points

| Resource | What you will learn |
|---|---|
| [Neo4j Python Driver Manual](https://neo4j.com/docs/python-manual/current/) | Driver setup, sessions, and transactions in Python |
| [Neo4j Cypher Manual](https://neo4j.com/docs/cypher-manual/current/) | Complete Cypher language reference |
| [Neo4j — What is a graph database?](https://neo4j.com/docs/getting-started/graph-database/) | Concepts: nodes, relationships, properties |
| [Neo4j — Graph data modelling](https://neo4j.com/docs/getting-started/data-modeling/) | How to design a graph schema |
| [Neo4j GDS Library](https://neo4j.com/docs/graph-data-science/current/) | PageRank, Dijkstra, community detection, and more |
| [Neo4j Cypher — Indexes and Constraints](https://neo4j.com/docs/cypher-manual/current/indexes/) | How to make queries fast |
