# Side Note 2 — Vector Database Best Practices in Production

> **Disclaimer**
> This document was co-authored with the assistance of multiple AI tools. While every effort has been made to ensure the accuracy of the content, some unintended errors may still be present. If you spot any mistakes, please [submit an issue on GitHub](https://github.com/NCUIM-Lab710-Teaching/IM2002-DBMGT-Train-v2/issues).

---

> **Who is this for?**
> This note is for students who have worked with the pgvector RAG pipeline in this project.
> You have seen how embeddings are stored in PostgreSQL and searched with cosine similarity — now let's look at how production systems do this properly at scale.

---

## What Does the Teaching Code Do?

The TransitFlow project stores policy documents as embedding vectors inside PostgreSQL using the **pgvector** extension. When a user asks a question, the agent converts it to an embedding, then runs this query:

```python
sql = """
    SELECT title, category, content,
           1 - (embedding <=> %s::vector) AS similarity
    FROM policy_documents
    WHERE 1 - (embedding <=> %s::vector) > %s
    ORDER BY embedding <=> %s::vector
    LIMIT %s
"""
```

The `<=>` operator is pgvector's cosine distance. This is a perfectly valid approach for a small dataset. Production systems — handling millions of documents and thousands of queries per second — need to handle this differently.

---

## 1. Dedicated Vector Databases vs pgvector

### What is a vector database?

A **vector database** is a database built specifically to store, index, and search embedding vectors at scale. Unlike PostgreSQL + pgvector (which adds vector capability on top of a relational database), a dedicated vector DB is designed from the ground up for this task.

### When is pgvector enough?

pgvector is a good choice when:
- Your document collection is small (under a few million)
- You already use PostgreSQL and don't want to add another system
- You need to combine vector search with relational queries in the same database

This is exactly the case in TransitFlow — it is the right tool for the job here.

### When do you need a dedicated vector database?

When your use case involves:
- Tens of millions of vectors
- Sub-millisecond search latency requirements
- Horizontal scaling across multiple servers
- Built-in support for real-time vector updates

The most widely used dedicated vector databases are:

| Database | Best for | Hosted option? |
|---|---|---|
| **Pinecone** | Fully managed, zero infrastructure | Yes (cloud-only) |
| **Qdrant** | High performance, open-source, Rust-based | Yes + self-host |
| **Weaviate** | Combined vector + keyword search, multi-modal | Yes + self-host |
| **ChromaDB** | Local development and prototyping | Self-host only |

```python
# Example: using Qdrant instead of pgvector
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(url="http://localhost:6333")

# Create a collection (equivalent to a table)
client.create_collection(
    collection_name="policy_documents",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

# Search
results = client.search(
    collection_name="policy_documents",
    query_vector=embedding,
    limit=5,
)
```

### Learn more
- [pgvector — GitHub repository and documentation](https://github.com/pgvector/pgvector)
- [Pinecone — official documentation](https://docs.pinecone.io/)
- [Qdrant — official documentation](https://qdrant.tech/documentation/)
- [Weaviate — official documentation](https://docs.weaviate.io/weaviate)
- [ChromaDB — official documentation](https://docs.trychroma.com/)

---

## 2. Indexing: Why Your Search Will Be Slow Without It

### The problem with the teaching code

The teaching code does **no indexing** on the `embedding` column. Every search performs a full table scan — it compares the query vector against every single row in the table. This is called **exact nearest neighbour** (exact NN) search.

For 100 policy documents, this is fast. For 10 million documents, this takes several seconds per query.

### The production solution: Approximate Nearest Neighbour (ANN) indexing

ANN indexes trade a tiny amount of accuracy for a massive speedup. The two most common algorithms used by pgvector (and most vector databases) are:

#### HNSW — Hierarchical Navigable Small World

The default and best choice for most applications. Builds a multi-layer graph structure over the vectors. Search is fast even on very large collections.

```sql
-- Add HNSW index to the embedding column in PostgreSQL + pgvector
CREATE INDEX ON policy_documents
USING hnsw (embedding vector_cosine_ops);
```

#### IVFFlat — Inverted File with Flat quantisation

Divides vectors into clusters (lists) and searches only within the most relevant clusters. Faster to build than HNSW but slightly less accurate.

```sql
CREATE INDEX ON policy_documents
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

**Rule of thumb:** Use HNSW unless you have a very large collection and tight memory constraints.

### Learn more
- [pgvector HNSW and IVFFlat indexing (GitHub README)](https://github.com/pgvector/pgvector)

---

## 3. Choosing an Embedding Model

### What the teaching code does

The teaching code generates embeddings externally (via `llm.embed()` in `skeleton/llm_provider.py`) and passes the resulting float list into the query. This is correct — the model should live outside the database layer.

### What matters when choosing a model in production

In production, the choice of embedding model affects accuracy, speed, and cost. The main options are:

#### Option A — Hosted API (OpenAI, Cohere, etc.)
You send your text to an external API and get an embedding back. No GPU needed, easy to use, but you pay per token and add network latency.

```python
import openai

response = openai.embeddings.create(
    model="text-embedding-3-small",
    input="What is the delay repay policy?"
)
embedding = response.data[0].embedding  # list of 1536 floats
```

#### Option B — Local model (Sentence Transformers)
You run an open-source model on your own machine or server. No API cost, fully offline, but requires more compute.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode("What is the delay repay policy?")
```

**Important:** Whatever model you use to *embed your documents* at index time, you **must use the same model** to embed queries at search time. Mixing models produces nonsense results.

### Learn more
- [Sentence Transformers — documentation and pretrained models](https://www.sbert.net/)
- [HuggingFace NLP Course — Semantic Search with Embeddings](https://huggingface.co/learn/nlp-course/chapter5/6)

---

## 4. Chunking: How You Split Documents Matters

### The problem

The teaching code stores each policy document as a single large text block. If a user asks about the minimum fare rule, the entire railcard guide (500+ words) is returned as one chunk. The LLM then has to read everything to find the one relevant sentence.

In production, you **chunk** documents into smaller pieces before embedding them — each chunk covers one coherent idea.

### Common chunking strategies

#### Fixed-size chunking
Split text every N characters or tokens, with a small overlap to preserve context across boundaries.

```python
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap   # overlap ensures context is not lost at boundaries
    return chunks
```

#### Recursive / semantic chunking
Split on natural boundaries first (paragraphs → sentences → words). Only split smaller if the chunk is still too large. This produces more meaningful chunks than fixed-size splitting.

#### Rule of thumb for chunk sizes

| Use case | Recommended chunk size |
|---|---|
| Short factual Q&A | 100–200 tokens |
| Policy / legal documents | 300–500 tokens |
| Long-form content | 500–1000 tokens |

### Learn more
- [LangChain — Build a RAG application (official tutorial)](https://docs.langchain.com/oss/python/langchain/rag)

---

## 5. Metadata Filtering

### What the teaching code does

The teaching code searches all policy documents regardless of category. If a user asks about a refund, the search might return an accessibility guide if it happens to be similar to the query vector.

### The production solution: pre-filter by metadata

Most production vector databases support **metadata filters** — you can narrow the search to a specific category, date range, or any other field before doing the vector similarity comparison.

```python
# Example: only search within the "refund" category using Qdrant
from qdrant_client.models import Filter, FieldCondition, MatchValue

results = client.search(
    collection_name="policy_documents",
    query_vector=embedding,
    query_filter=Filter(
        must=[FieldCondition(key="category", match=MatchValue(value="refund"))]
    ),
    limit=5,
)
```

In pgvector you can achieve this with a simple `WHERE` clause added before the vector comparison:

```sql
SELECT title, content, 1 - (embedding <=> %s::vector) AS similarity
FROM policy_documents
WHERE category = 'refund'                           -- metadata filter first
  AND 1 - (embedding <=> %s::vector) > %s
ORDER BY embedding <=> %s::vector
LIMIT %s
```

---

## 6. Reranking

### What is reranking?

After the vector search returns the top K results, a **reranker** re-scores them using a more accurate (but slower) model called a **cross-encoder**. The initial vector search is fast and casts a wide net. The reranker then picks the best results from that net.

Think of it in two stages:
1. **Vector search** — retrieves the top 20 candidates quickly (fast, approximate)
2. **Reranker** — reads each candidate against the query and re-scores all 20, returning the best 5 (slow, accurate)

```python
import cohere

co = cohere.Client("your-api-key")

# Step 1: retrieve candidates with vector search (top 20)
candidates = query_policy_vector_search(embedding, top_k=20)

# Step 2: rerank the candidates against the original question
reranked = co.rerank(
    model="rerank-english-v3.0",
    query="What is the delay repay policy?",
    documents=[c["content"] for c in candidates],
    top_n=5,
)
```

Reranking significantly improves the quality of results for long-form questions or when the initial vector search returns results that are semantically similar but not actually relevant.

### Learn more
- [Cohere Rerank — API reference](https://docs.cohere.com/reference/rerank)

---

## 7. Embedding Cache

### The problem

In the teaching code, every time the agent answers a question, it re-embeds the query. If the same question is asked twice, the model runs twice — wasting time and API cost.

### The production solution: cache embeddings

For queries you expect to see repeatedly, cache the embedding in Redis or a simple in-memory store:

```python
import hashlib, json
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_cached_embedding(text: str) -> tuple[float, ...]:
    embedding = llm.embed(text)           # only calls the model on first occurrence
    return tuple(embedding)               # tuples are hashable, lists are not
```

For production, Redis is preferred over `lru_cache` because it survives server restarts and is shared across multiple processes.

---

## Summary

| Topic | Teaching Code | Production Approach |
|---|---|---|
| **Vector storage** | pgvector in PostgreSQL | Dedicated DB (Qdrant, Pinecone, Weaviate) or pgvector with HNSW index |
| **Search type** | Full table scan (exact NN) | ANN index (HNSW or IVFFlat) |
| **Document size** | One chunk per document | Multiple smaller chunks per document |
| **Filtering** | Similarity threshold only | Metadata filter + similarity threshold |
| **Result quality** | Top K raw results | Top K reranked by cross-encoder |
| **Embedding compute** | Re-embedded every query | Cached (lru_cache or Redis) |

---

## Recommended Starting Points

| Resource | What you will learn |
|---|---|
| [pgvector GitHub](https://github.com/pgvector/pgvector) | HNSW/IVFFlat indexing for PostgreSQL |
| [Sentence Transformers docs](https://www.sbert.net/) | Running embedding models locally |
| [HuggingFace NLP Course — Semantic Search](https://huggingface.co/learn/nlp-course/chapter5/6) | End-to-end embedding + search tutorial |
| [LangChain RAG tutorial](https://docs.langchain.com/oss/python/langchain/rag) | Full RAG pipeline with chunking and retrieval |
| [Qdrant documentation](https://qdrant.tech/documentation/) | Purpose-built vector database from scratch |
| [Cohere Rerank API](https://docs.cohere.com/reference/rerank) | Improving RAG result quality with reranking |
