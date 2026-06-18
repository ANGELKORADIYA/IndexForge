# MemorySearch Setup Guide 🔍

This guide covers development and production setup for all completed phases.

---

## 1. Prerequisites

| Tool | Version | Required For |
|---|---|---|
| Docker + Docker Compose | Any recent | Recommended quickstart |
| Rust (rustup) | 1.75+ (2021 edition) | Local development |
| PostgreSQL | 16+ | Metadata + vector storage |
| pgvector extension | 0.7+ | HNSW vector search |

---

## 2. Quick Start (Docker — Recommended)

```bash
# Start DB + dev container with live code mounting
docker compose --profile dev up

# Or production (optimized binary)
docker compose --profile prod up --build
```

---

## 3. Manual Local Setup

### Step 1 — Clone & configure

```bash
git clone <repository-url>
cd memory-search
cp .env.example .env
```

Edit `.env` — the important variables:

```env
DATABASE_URL=postgres://postgres:postgres@localhost:5432/memory_search
TANTIVY_PATH=./data/tantivy-index
```

### Step 2 — Start the database

```bash
# DB only via Docker
docker compose up db -d

# Or use a local PostgreSQL 16+ instance with pgvector
```

### Step 3 — Apply migrations

```bash
# Option A: sqlx-cli (recommended)
cargo install sqlx-cli --no-default-features --features postgres
sqlx migrate run --database-url "postgres://postgres:postgres@localhost:5432/memory_search"

# Option B: psql directly
psql postgres://postgres:postgres@localhost:5432/memory_search \
     -f migrations/20240101000000_init.sql
```

This creates:
- `documents` table (one row per indexed file)
- `chunks` table with `embedding VECTOR(384)` column
- HNSW index on `embedding` for ANN search
- GIN full-text index on `text`

### Step 4 — Build

```bash
cargo build --package ms-cli
```

---

## 4. CLI Usage (Phase 1 + Phase 2)

### Index a directory

Chunks files, generates embeddings, stores in both Tantivy (BM25) and pgvector:

```bash
cargo run --package ms-cli -- index ./my-notes
cargo run --package ms-cli -- index ./my-notes --mode notes
cargo run --package ms-cli -- index ./my-codebase --mode codebase
```

Available modes: `notes`, `codebase`, `wikipedia`, `default`

### BM25 search (fast, no DB needed after indexing)

```bash
cargo run --package ms-cli -- search "rust ownership"
cargo run --package ms-cli -- search "machine learning" --mode notes --top-k 5
```

### 3-arm search — BM25 + Fuzzy + Semantic with RRF merge

Loads fuzzy index from DB, embeds query, runs all 3 arms in parallel:

```bash
cargo run --package ms-cli -- search-all "rust ownership"
cargo run --package ms-cli -- search-all "machne learnng" --mode notes   # typo tolerant
```

### Advanced Search: Cross-Encoder Re-ranking

You can append `--rerank` to pass the RRF results through a Cross-Encoder model (`BGERerankerBase`) for significantly higher accuracy:

```bash
cargo run --package ms-cli -- search-all "what is ownership?" --rerank
```

### Retrieval-Augmented Generation (RAG)

You can append `--rag` to pass the top search results to an LLM to generate a natural language answer. By default, it uses a local Ollama instance:

```bash
cargo run --package ms-cli -- search-all "what is ownership?" --rerank --rag
```

Output format (Search-All):
```
1. [RRF 0.0328] chunk preview text... | arms: BM25=1.234, Semantic=0.891
2. [RRF 0.0291] another chunk...      | arms: Fuzzy=0.412, Semantic=0.743

🤖 Generating answer with RAG...

=== RAG Answer (Model: llama3.2) ===
Ownership is a set of rules that governs how a Rust program manages memory...
```

---

## 5. How Each Search Arm Works

### Arm 1 — BM25 (Tantivy)
- Inverted index, TF-IDF weighted term frequency
- Exact keyword matching
- Sub-millisecond on medium datasets

### Arm 2 — Fuzzy (Trigram + Jaccard)
- Character trigrams: "hello" → `[" he", "hel", "ell", "llo", "lo "]`
- Jaccard similarity: |A ∩ B| / |A ∪ B|
- Naturally tolerates typos: 1-char error only invalidates 3 of N trigrams
- Pure-Rust, no external dictionary needed

### Arm 3 — Semantic (pgvector HNSW)
- Query embedded via `all-MiniLM-L6-v2` (384-dim, ~86MB, downloaded once)
- Cosine ANN via pgvector `<=>` operator on HNSW index
- Finds results with zero keyword overlap

### Merge — RRF (Reciprocal Rank Fusion)
- Score: `Σ_arm 1 / (60 + rank_i)`
- Deduplicates across arms, preserves per-arm scores
- No model needed, ~1ms overhead

## 6. Phase 3: Reranker & RAG Setup

### Cross-Encoder Re-ranker
When you use the `--rerank` flag, the engine downloads the `BGERerankerBase` ONNX model (~85MB) automatically on the first run. It re-scores the combined results from the 3 arms to provide the most accurate final ranking.

### RAG LLM Providers
When you use the `--rag` flag, the `ms-rag` layer selects an LLM provider based on your `.env` configuration. It checks in the following priority order:

1. **Gemini (Google API)**: Used if `GEMINI_API_KEY` is set.
   - Default model: `gemini-1.5-pro-latest`
2. **OpenRouter (Cloud API)**: Used if `OPENROUTER_API_KEY` is set.
   - Default model: `meta-llama/llama-3-8b-instruct`
3. **Ollama (100% Offline)**: Used if no API keys are found.
   - Default host: `http://localhost:11434`
   - Default model: `llama3.2`

You must have the Ollama server running locally with `ollama run llama3.2` to use the offline fallback.

---

## 7. Phase 4: Full-Stack Web UI Setup

### Starting the Axum REST API (`ms-server`)
The REST API runs on port 8080 by default and serves endpoints for searching and indexing.
```bash
cargo run --package ms-server
```

### Starting the React Frontend (`ui`)
The frontend is built with React and Vite. It expects the `ms-server` to be running at `localhost:8080`.
```bash
cd ui
npm install
npm run dev
```
Navigate to `http://localhost:5173` to access the MemorySearch web interface.

---

## 7. Model Download

Embeddings model (`all-MiniLM-L6-v2`, ~86MB ONNX) is downloaded **automatically** on first `index` run by fastembed into `./.fastembed_cache/`.

Override cache location via env:
```env
FASTEMBED_CACHE_DIR=./models
```

Or set `HF_HOME` to use a Hugging Face mirror.

---

## 7. Common Troubleshooting

### `relation "documents" does not exist`
Run the SQL migration (Step 3 above). The schema has not been applied yet.

### `DATABASE_URL must be set`
Ensure `.env` exists (copied from `.env.example`) or export `DATABASE_URL` in your shell.

### `RelativeUrlWithoutBase` on model download
Was a fastembed 3.x bug — fixed by upgrading to fastembed 5.x (already done).

### `ort_sys::size_t` not found
Was a dep version conflict between `ort` rc.4 and `ort-sys` rc.9 — fixed by upgrading to fastembed 5.x.

### `cannot borrow self.model as mutable`
fastembed 5.x `embed()` takes `&mut self`. Fixed in `embedder.rs` and call sites.

### `pgvector` extension not found
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Port 5432 already in use
Change `POSTGRES_PORT` in `.env` and update the Docker Compose port mapping.

### Slow first Rust build in Docker
Expected — the Dockerfile caches deps via a dummy build. Only first build is slow.

---

## 8. Project Structure

```
crates/
  ms-core/       # Shared types: Chunk, SearchResult, SearchArm, Config
  ms-ingest/     # Chunker (SentenceChunker) + Embedder (fastembed 5.x)
  ms-index/      # BM25 (Tantivy), Vector (pgvector), Fuzzy (trigram)
  ms-search/     # Query router (tokio::join!) + RRF merge
  ms-rerank/     # Cross-encoder re-ranker (Phase 3)
  ms-rag/        # RAG layer / Ollama (Phase 3)
  ms-server/     # Axum REST API (Phase 4)
  ms-cli/        # CLI: index, search, search-all
  ms-python/     # Python Library bindings (PyO3)
ui/              # React + Vite frontend UI
data/            # Tantivy index files (gitignored)
migrations/      # SQL schema migrations
plans/           # Per-phase task briefs, reports, reviews
models/          # Offline ML models (gitignored)
.env.example     # Environment variable template
```

---

## 9. Python Library (`memory_search`)

We expose a natively compiled Python library powered by `PyO3` and `maturin`.

### Installation
```bash
cd crates/ms-python
python -m venv .venv
# source .venv/bin/activate  (Linux/Mac)
.\.venv\Scripts\activate   # (Windows)
pip install maturin
maturin develop --release
```

### Usage
```python
import memory_search
import json

ms = memory_search.MemorySearch()
ms.index("./data", mode="wikipedia")

results = ms.search("How does vector search work?", top_k=3, rerank=True)
print(json.loads(results))
```
