# 🔍 IndexForge

> **Local-first semantic search engine** — search anything like Google with exact keyword match, fuzzy tolerance, and semantic intent understanding — entirely offline and built in Rust.

![Status](https://img.shields.io/badge/status-production--ready-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)
![Rust](https://img.shields.io/badge/rust-1.75%2B-orange?style=flat-square)

---

## ✨ Features

### 🎯 Three Search Arms in Parallel
- **Exact Keyword (BM25)** — Fast, deterministic full-text search via Tantivy
- **Fuzzy Match** — Typo-tolerant search using character trigrams + SymSpell
- **Semantic Intent** — Vector-based search using HNSW ANN for zero-keyword-overlap results

### 🚀 Performance
- Sub-10ms search latency on medium datasets (up to 500K chunks)
- Sub-100ms with re-ranking
- 100% offline by default — all models bundled locally
- Parallel query execution via Tokio async runtime

### 📚 Multi-Modal Data Support
- **Notes** — Markdown, plaintext, obsidian vaults
- **Codebase** — Language-aware AST chunking via tree-sitter
- **Wikipedia** — Streaming `.zim` and XML dump parsing
- **PDFs, DOCX, CSV, JSON** — Native loaders with metadata extraction

### 🔄 Flexible Re-ranking
- **RRF (Reciprocal Rank Fusion)** — Fast, no-model merge of 3 arms
- **Cross-Encoder** — BGERerankerBase for highest quality
- **LLM Re-ranker** — Local Ollama or cloud API (Gemini, OpenRouter)

### 🤖 RAG Integration
- Generate natural language answers from search results
- Support for local (Ollama), cloud (Gemini, OpenRouter), and offline modes
- Configurable prompt templates

### 🐍 Python Bindings
- Full PyO3 library for seamless Python integration
- Native compiled performance with Rust internals

### 🌐 Full-Stack Web UI
- React + Vite frontend with glassmorphism design
- REST API via Axum with `/api/index` and `/api/search` endpoints
- Real-time search with arm score visualization

---

## 🚀 Quick Start

### Prerequisites
| Tool | Version | Purpose |
|------|---------|---------|
| Rust | 1.75+ | Build from source |
| PostgreSQL | 16+ | Metadata & vector storage |
| pgvector | 0.7+ | HNSW vector indexing |
| Docker (optional) | Latest | Containerized setup |

### 1️⃣ Docker (Recommended)
```bash
# Clone the repository
git clone https://github.com/yourusername/indexforge
cd indexforge

# Start with dev profile (live code reloading)
docker compose --profile dev up

# Or production (optimized binary)
docker compose --profile prod up --build

# Access at http://localhost:5173 (UI) and http://localhost:8080 (API)
```

### 2️⃣ Local Manual Setup

#### Step 1: Clone & Configure
```bash
git clone https://github.com/yourusername/indexforge
cd indexforge
cp .env.example .env
```

#### Step 2: Start PostgreSQL
```bash
# Via Docker (simplest)
docker run -d \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=memory_search \
  -p 5432:5432 \
  pgvector/pgvector:latest

# Or use your local PostgreSQL 16+ with pgvector extension
psql -U postgres -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

#### Step 3: Apply Migrations
```bash
# Install sqlx-cli (one-time)
cargo install sqlx-cli --no-default-features --features postgres

# Run migrations
sqlx migrate run --database-url "postgres://postgres:postgres@localhost:5432/memory_search"
```

#### Step 4: Build & Run

**CLI Only:**
```bash
cargo build --release --package ms-cli
```

**Full Stack (API + UI):**
```bash
# Terminal 1: Start API
cargo run --release --package ms-server

# Terminal 2: Start React frontend
cd ui
npm install
npm run dev

# Navigate to http://localhost:5173
```

---

## 📖 Usage Examples

### Index Your Data

```bash
# Index a directory of notes
cargo run --package ms-cli -- index ./my-notes

# Index a codebase with language-aware chunking
cargo run --package ms-cli -- index ./src --mode codebase

# Index Wikipedia
cargo run --package ms-cli -- index ./wikipedia.zim --mode wikipedia
```

### Search

**Fast BM25 keyword search:**
```bash
cargo run --package ms-cli -- search "rust ownership"
```

**Parallel 3-arm search with RRF merge:**
```bash
cargo run --package ms-cli -- search-all "ownership rules"
```

**With re-ranking (Cross-Encoder):**
```bash
cargo run --package ms-cli -- search-all "what is ownership?" --rerank
```

**With RAG answer generation:**
```bash
cargo run --package ms-cli -- search-all "ownership in rust" --rerank --rag
```

### Output Example
```
1. [RRF 0.0328] Rust uses an ownership system... | arms: BM25=1.234, Semantic=0.891
2. [RRF 0.0291] Ownership rules prevent double... | arms: Fuzzy=0.412, Semantic=0.743

🤖 Generating answer with RAG...

=== RAG Answer (Model: llama3.2) ===
Ownership is a set of rules that governs how a Rust program manages memory...
```

### Python Library

```python
import memory_search

ms = memory_search.MemorySearch()

# Index
ms.index("./my-notes", mode="notes")

# Search with all 3 arms
results = ms.search_all("rust ownership", top_k=5, rerank=True)

for result in results:
    print(f"Score: {result['score']:.4f}")
    print(f"Text: {result['text'][:200]}...")
    print(f"Arms: {result['arms']}")
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│         Data Sources (Modes)            │
│  Wikipedia | Notes | Codebase | PDFs    │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│      Ingestion Pipeline (Rust)          │
│  Chunker → Embedder → BM25 Indexer      │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│         Storage Layer                   │
│  PostgreSQL + pgvector | Tantivy        │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│      Query Router (3 Parallel Arms)     │
│  BM25 | Fuzzy (Trigrams) | Vector ANN   │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│    Re-ranker (RRF | Cross-Encoder)      │
│  Merge & score results                  │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│      RAG Layer (Optional)               │
│  LLM → Natural Language Answer          │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│      Output Interfaces                  │
│  CLI | REST API | Web UI | Python Lib   │
└─────────────────────────────────────────┘
```

For detailed architecture, see [architecture.md](architecture.md).

---

## 📁 Project Structure

```
crates/
  ms-core/       # Shared types: Chunk, SearchResult, SearchArm, Config
  ms-cli/        # CLI binary: index, search, search-all commands
  ms-ingest/     # Chunker, loader, document parser
  ms-index/      # BM25 (Tantivy), Vector (pgvector), Fuzzy (trigrams)
  ms-search/     # Query router, RRF merge, arm coordination
  ms-rerank/     # Cross-encoder re-ranker (BGERerankerBase)
  ms-rag/        # LLM providers, RAG pipeline
  ms-server/     # Axum REST API server
  ms-python/     # Python bindings (PyO3)

ui/              # React + Vite frontend
migrations/      # PostgreSQL schema migrations
data/            # Tantivy index & uploads (gitignored)
.env.example     # Environment variables template
Dockerfile       # Multi-stage build (dev + prod profiles)
docker-compose.yml
```

---

## ⚙️ Technology Stack

| Layer | Technology |
|-------|-----------|
| **Core Engine** | Rust 2021, Tokio async runtime |
| **Full-Text Search** | Tantivy (BM25 + inverted index) |
| **Vector Search** | pgvector HNSW or FAISS |
| **Metadata Store** | PostgreSQL 16+ |
| **Embeddings** | all-MiniLM-L6-v2 via fastembed (384-dim) |
| **Fuzzy Search** | Character trigrams + Jaccard similarity |
| **Re-ranking** | Cross-encoder (BGERerankerBase ONNX) |
| **LLM Integration** | Ollama (local) or cloud APIs |
| **REST API** | Axum web framework |
| **Frontend** | React 19 + Vite + TailwindCSS |
| **Python Binding** | PyO3 + maturin |

---

## 📊 Performance Targets

| Scenario | Latency | Dataset Size |
|----------|---------|--------------|
| BM25 search | < 5ms | 500K chunks |
| 3-arm search (no rerank) | < 10ms | 500K chunks |
| With cross-encoder rerank | < 100ms | 100K chunks |
| With LLM rerank | 2–5s | Top-20 results |
| Vector ANN only | < 50ms | 5M vectors |

---

## 🔧 Configuration

Copy `.env.example` to `.env` and customize:

```env
# Database
DATABASE_URL=postgres://postgres:postgres@localhost:5432/memory_search

# Search
TANTIVY_PATH=./data/tantivy-index
DEFAULT_TOP_K=5
DEFAULT_MODE=notes

# Embeddings (auto-downloaded on first run)
FASTEMBED_CACHE_DIR=./.fastembed_cache

# RAG (optional — falls back to Ollama at localhost:11434)
GEMINI_API_KEY=
OPENROUTER_API_KEY=
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# Web Server
SERVER_HOST=0.0.0.0
SERVER_PORT=8080
```

---

## 📚 Detailed Documentation

- [**SETUP_GUIDE.md**](SETUP_GUIDE.md) — Step-by-step installation & troubleshooting
- [**architecture.md**](architecture.md) — System design, open-source inspirations, data flow
- [**PROJECT.md**](PROJECT.md) — Phase-by-phase deliverables, cross-module decisions
- [**progress.md**](progress.md) — Current status and completed phases

---

## 🛠️ Development

### Build from Source
```bash
# Debug build (fast compile, slower runtime)
cargo build

# Release build (optimized for production)
cargo build --release

# Run tests
cargo test

# Run specific crate tests
cargo test --package ms-index
```

### Running Tests with Docker
```bash
docker compose --profile dev up
# Tests run automatically via docker-compose override
```

### Code Organization
- **Type definitions**: `ms-core/src/lib.rs`
- **CLI commands**: `ms-cli/src/main.rs`
- **Search arms**: `ms-index/src/{bm25,fuzzy,vector}.rs`
- **REST endpoints**: `ms-server/src/main.rs`
- **React components**: `ui/src/components/`

---

## 🚢 Deployment

### Docker Production Build
```bash
docker compose --profile prod up --build

# Or build only
docker build -t indexforge:latest -f Dockerfile --target prod .

# Run
docker run -d \
  -e DATABASE_URL="postgres://..." \
  -p 8080:8080 \
  indexforge:latest
```

### Kubernetes (via Helm)
```bash
helm chart coming soon...
```

---

## 📝 License

MIT License — see [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/amazing-feature`
3. Commit your changes: `git commit -am 'Add amazing feature'`
4. Push to the branch: `git push origin feat/amazing-feature`
5. Open a Pull Request

See [contributing guidelines](CONTRIBUTING.md) for more details (coming soon).

---

## 🙋 Support & Community

- **Issues**: [GitHub Issues](https://github.com/yourusername/indexforge/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/indexforge/discussions)
- **Chat**: [Discord](https://discord.gg/yourdiscord) (coming soon)

---

## 📖 How Each Search Arm Works

### Arm 1: Exact Keyword (BM25 via Tantivy)
Inverted index with TF-IDF scoring. Fast, deterministic, best for exact matches.
```
Query: "rust ownership"
Matches: All chunks containing "rust" OR "ownership"
Score: TF-IDF weighted term frequency
Latency: < 5ms
```

### Arm 2: Fuzzy Match (Trigrams + Jaccard)
Character n-grams with Jaccard similarity. Naturally tolerates typos.
```
Query: "russt ownersheep"
Converts to trigrams: [" ru", "rus", "uss", "sst", ...]
Matches chunks where 60%+ of trigrams overlap
Latency: < 5ms
```

### Arm 3: Semantic Intent (Vector ANN)
Query embedded via `all-MiniLM-L6-v2`, then ANN search via HNSW.
```
Query: "memory management mechanism"
Embedded to: [0.123, -0.456, ..., 0.789] (384 dims)
Matches: Most similar vectors via cosine distance
Latency: < 50ms
```

### Merge: RRF (Reciprocal Rank Fusion)
Combines ranks from all 3 arms, deduplicates, preserves per-arm scores.
```
Score = Σ 1 / (60 + rank_i)
Deduplicates: If chunk appears in multiple arms, highest rank wins
Latency: < 1ms
```

---

## 🎯 Roadmap

- [x] **Phase 1** — Core engine (Tantivy BM25, PostgreSQL, CLI)
- [x] **Phase 2** — Fuzzy + Semantic arms (Trigrams, pgvector HNSW)
- [x] **Phase 3** — Re-rankers (Cross-Encoder, LLM re-ranker, RAG)
- [x] **Phase 4** — Full-stack web UI (React + Axum)
- [x] **Phase 5** — Advanced modes (Wikipedia, PDF, Codebase AST)
- [x] **Phase 6** — Python library & advanced retrieval
- [ ] **Phase 7** — Kubernetes operator & distributed indexing
- [ ] **Phase 8** — Multi-language support & UI internationalization

---

## 🔗 Inspiration & Credits

Built on the shoulders of giants:
- **[LlamaIndex](https://github.com/run-llama/llama_index)** — Chunking & node parser patterns
- **[FAISS](https://github.com/facebookresearch/faiss)** — Vector indexing via pgvector HNSW
- **[Typesense](https://github.com/typesense/typesense)** — BM25 + fuzzy search inspiration
- **[AnythingLLM](https://github.com/Mintplex-Labs/anything-llm)** — Multi-arm retrieval & LLM integration
- **[OpenWebUI](https://github.com/open-webui/open-webui)** — Re-ranker UI patterns

---

## 📞 Contact

For questions, feature requests, or bugs:
- Open an [issue](https://github.com/yourusername/indexforge/issues)
- Start a [discussion](https://github.com/yourusername/indexforge/discussions)

---

**Built with ❤️ in Rust**
