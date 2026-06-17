# MemorySearch 🔍

> A local-first, offline-capable semantic search engine that mimics Google — supporting exact keyword match, fuzzy match, and intent/semantic search — built in Rust with a React + FastAPI web UI.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Open Source Inspirations — What We Steal From Each](#2-open-source-inspirations--what-we-steal-from-each)
3. [System Architecture](#3-system-architecture)
4. [Tech Stack](#4-tech-stack)
5. [Directory Structure](#5-directory-structure)
6. [Data Modes](#6-data-modes)
7. [Ingestion Pipeline](#7-ingestion-pipeline)
8. [Storage Layer](#8-storage-layer)
9. [The 3 Search Arms](#9-the-3-search-arms)
10. [Merge & Re-Ranking Layer](#10-merge--re-ranking-layer)
11. [RAG Layer (Optional)](#11-rag-layer-optional)
12. [Output Interfaces](#12-output-interfaces)
13. [Build Phases (Roadmap)](#13-build-phases-roadmap)
14. [Phase 1 — Setup & Core Engine](#14-phase-1--setup--core-engine)
15. [Phase 2 — Fuzzy + Semantic Arms](#15-phase-2--fuzzy--semantic-arms)
16. [Phase 3 — Re-Rankers](#16-phase-3--re-rankers)
17. [Phase 4 — Full-Stack Web UI](#17-phase-4--full-stack-web-ui)
18. [Phase 5 — Data Modes](#18-phase-5--data-modes)
19. [Configuration Reference](#19-configuration-reference)
20. [API Reference](#20-api-reference)
21. [Offline Model Downloads](#21-offline-model-downloads)
22. [Performance Benchmarks & Targets](#22-performance-benchmarks--targets)
23. [Contributing](#23-contributing)

---

## 1. Project Overview

MemorySearch is a **multi-modal local search engine** designed to work on your own data — notes, PDFs, code, Wikipedia dumps, web pages, and structured files — entirely offline.

### Core philosophy

- **100% offline by default.** All models (embedding, re-ranking) are downloaded once and run locally. No data leaves your machine.
- **Flexible optionally.** You can plug in a local LLM (via Ollama) or a cloud API (Anthropic / OpenAI) for the RAG answer layer.
- **Three search arms, one result.** Exact keyword (BM25), fuzzy/typo-tolerant, and semantic/intent search all run in parallel, then get merged and re-ranked.
- **Mode-aware.** Different data types (Wikipedia, Notes, Codebase) get different chunking, indexing, and search strategies.
- **Performance-first.** The core engine is Rust. Sub-10ms search on medium datasets (up to 500K chunks).

### What you get

```
User query: "how does rust ownership work"
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   BM25 exact    Fuzzy match  Semantic ANN
   (tantivy)     (symspell)   (all-MiniLM)
        └───────────┼───────────┘
                    ▼
             Re-ranker (RRF / cross-encoder / LLM)
                    ▼
         Top-K chunks + metadata + LLM answer
```

---

## 2. Open Source Inspirations — What We Steal From Each

### LlamaIndex (`llama_index`)

**Repo:** https://github.com/run-llama/llama_index

**What it does well:**
- Modular document ingestion pipeline (loaders, transformers, nodes)
- Flexible chunking strategies (`SentenceSplitter`, `TokenTextSplitter`, `CodeSplitter`)
- Node metadata enrichment (source file, page number, chunk index)
- Retrieval abstractions (vector store, keyword store, knowledge graph)

**What we mimic:**
- `SimpleNodeParser` → our `Chunker` struct in Rust
- Overlap-aware sentence chunking (each chunk shares N sentences with prev/next)
- Metadata schema: `{ source_file, page, line_start, line_end, chunk_index, doc_id }`
- The concept of a "retriever" that abstracts over multiple backends

**Key code pattern to replicate (Python → Rust logic):**
```python
# LlamaIndex sentence splitter concept
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = splitter.get_nodes_from_documents(documents)
# Each node: { text, metadata: { file_path, page_label } }
```

```rust
// Our Rust equivalent
pub struct Chunk {
    pub id: Uuid,
    pub text: String,
    pub embedding: Option<Vec<f32>>,
    pub metadata: ChunkMetadata,
}

pub struct ChunkMetadata {
    pub source_file: String,
    pub page: Option<u32>,
    pub line_start: Option<u32>,
    pub line_end: Option<u32>,
    pub chunk_index: u32,
    pub doc_id: Uuid,
    pub mode: DataMode,
}
```

---

### FAISS (`faiss`)

**Repo:** https://github.com/facebookresearch/faiss

**What it does well:**
- Billion-scale approximate nearest neighbor (ANN) search
- HNSW (Hierarchical Navigable Small World) graph index — best recall/speed tradeoff
- IVF (Inverted File Index) — good for massive datasets with GPU
- Pure in-memory, extremely fast

**What we mimic:**
- HNSW index via `pgvector` (PostgreSQL extension) for medium datasets
- FAISS flat index via `faiss-rs` (Rust bindings) for large/massive datasets
- The query pattern: embed query → `index.search(query_vector, top_k)`

**Index selection by mode:**
| Mode | Dataset size | Index type |
|------|-------------|-----------|
| Notes / Codebase | < 500K chunks | `pgvector` HNSW (M=16, ef=64) |
| Wikipedia | 5M+ chunks | FAISS IVF-PQ (GPU optional) |

**Key parameters to understand:**
```
HNSW M=16        → each node connects to 16 neighbors (higher = better recall, more RAM)
HNSW ef_search=64 → search explores 64 candidates (higher = better recall, slower)
IVF nlist=1024   → 1024 Voronoi cells (√N rule: √5M ≈ 2236, use 2048)
PQ m=8           → 8 sub-quantizers (compresses 384-dim float32 to 8 bytes per vector)
```

---

### AnythingLLM (`anything-llm`)

**Repo:** https://github.com/Mintplex-Labs/anything-llm

**What it does well:**
- Multi-source document ingestion with a clean workspace concept
- Parallel retrieval combining vector search + keyword search
- LLM-agnostic: swap between Ollama, OpenAI, Anthropic at runtime
- Clean separation of "collector" (ingest) vs "retriever" (search)

**What we mimic:**
- The "workspace" = our "mode" (Notes, Codebase, Wikipedia)
- Running vector search AND keyword search in parallel, then merging
- LLM provider abstraction: `LLMProvider` trait with `OllamaProvider`, `AnthropicProvider`, `OpenAIProvider`
- The collector pipeline pattern: `raw_file → loader → chunker → embedder → store`

**Our Rust trait inspired by AnythingLLM:**
```rust
#[async_trait]
pub trait LLMProvider: Send + Sync {
    async fn complete(&self, prompt: &str, context: &[&str]) -> Result<String>;
    fn model_name(&self) -> &str;
    fn is_local(&self) -> bool;
}

pub struct OllamaProvider { model: String, base_url: String }
pub struct AnthropicProvider { api_key: String, model: String }
pub struct NoOpProvider; // returns empty string, for pure search mode
```

---

### Typesense (`typesense`)

**Repo:** https://github.com/typesense/typesense

**What it does well:**
- BM25 full-text search with typo tolerance built in
- Trigram-based fuzzy matching (very fast)
- Faceted search, filters, sorting
- Instant search (sub-10ms)
- Schema-based collections

**What we mimic:**
- BM25 scoring via **Tantivy** (pure Rust, Lucene-inspired, same algorithmic family as Typesense's core)
- Trigram fuzzy matching: index every token's character 3-grams, query by 3-gram overlap
- SymSpell for query correction before BM25 (Typesense does this internally)
- The "collection" concept → our `index` per mode

**Tantivy vs Typesense:**
```
Typesense:  built-in server, HTTP API, managed collections
Tantivy:    Rust library, embedded, you control everything
We use:     Tantivy embedded in our Rust binary (no separate process)
```

**BM25 score formula (what Tantivy implements):**
```
Score(D,Q) = Σ IDF(qi) × (f(qi,D) × (k1+1)) / (f(qi,D) + k1 × (1 - b + b × |D|/avgdl))

where:
  IDF(qi) = log((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
  f(qi,D) = term frequency of qi in document D
  k1 = 1.2  (term saturation)
  b  = 0.75 (length normalization)
  N  = total number of documents
  n(qi) = documents containing qi
  |D| = length of D, avgdl = average document length
```

---

### OpenWebUI (`open-webui`)

**Repo:** https://github.com/open-webui/open-webui

**What it does well:**
- Clean, modern chat + search UI
- Model selection UI (pick which LLM to use)
- RAG pipeline with configurable retrieval settings
- Re-ranker selection exposed as a UI setting

**What we mimic:**
- The re-ranker selection UX: user picks "Fast (RRF)", "Quality (cross-encoder)", "Smart (LLM)"
- Streaming responses in the UI
- The "knowledge base" management panel: add/remove sources, see index status
- Search result cards showing source + chunk text + relevance score

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DATA SOURCES                          │
│  Wikipedia │ Notes/MD │ Codebase │ PDFs/URLs │ JSON/CSV │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              INGESTION PIPELINE (Rust)                   │
│  Loader → Chunker → Embedder → BM25 Indexer → Metadata  │
└──────┬─────────────────┬──────────────────┬─────────────┘
       │                 │                  │
┌──────▼──────┐  ┌───────▼──────┐  ┌───────▼──────────┐
│ PostgreSQL  │  │ FAISS / HNSW │  │  Tantivy Index   │
│ + pgvector  │  │ vector index │  │  BM25 full-text  │
│ (chunks +   │  │ (ANN search) │  │  + trigram fuzzy │
│  metadata)  │  └───────┬──────┘  └───────┬──────────┘
└──────┬──────┘          │                  │
       │                 │                  │
┌──────▼─────────────────▼──────────────────▼──────────┐
│                  QUERY ROUTER (Rust)                   │
│  normalize → expand → detect intent → fan-out         │
└──────┬─────────────────┬──────────────────┬───────────┘
       │                 │                  │
┌──────▼──────┐  ┌───────▼──────┐  ┌───────▼──────┐
│  BM25 ARM   │  │  FUZZY ARM   │  │SEMANTIC ARM  │
│  (Tantivy)  │  │(SymSpell +   │  │(all-MiniLM + │
│  TF-IDF     │  │ trigrams)    │  │ HNSW cosine) │
└──────┬──────┘  └───────┬──────┘  └───────┬──────┘
       └─────────────────┼──────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│            RE-RANKER (selectable)                    │
│  [RRF merge] [cross-encoder] [LLM ranker] [none]    │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│              RAG LAYER (optional)                    │
│  top-K chunks → prompt → LLM → answer + sources     │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│                OUTPUT INTERFACES                     │
│  Web UI (React) │ REST API │ Python lib │ CLI        │
└─────────────────────────────────────────────────────┘
```

---

## 4. Tech Stack

| Layer | Technology | Reason |
|-------|-----------|--------|
| Core engine | **Rust** (2021 edition) | Performance, memory safety, no GC pauses |
| Async runtime | **Tokio** | Parallel search arms, async I/O |
| Web server | **Axum** | Ergonomic Rust HTTP, built on Tokio |
| Full-text search | **Tantivy** | Pure Rust BM25, Lucene-quality |
| Vector store | **pgvector** (PostgreSQL ext) | HNSW index, SQL metadata queries |
| Large-scale ANN | **faiss-rs** | FAISS bindings for Wikipedia mode |
| Metadata DB | **PostgreSQL** | ACID, relations, pgvector in same DB |
| ORM / SQL | **SQLx** | Async, compile-time checked queries |
| Embeddings | **fastembed-rs** | all-MiniLM-L6-v2, fully offline |
| Fuzzy | **symspell-rs** + trigrams | Sub-ms typo correction |
| Re-ranker | **ort** (ONNX Runtime) | ms-marco cross-encoder, offline ONNX |
| Python bridge | **PyO3** | Expose Rust core as Python lib |
| API layer | **FastAPI** (Python) | Bridge UI ↔ Rust core |
| Frontend | **React** + **TypeScript** | Web UI |
| Local LLM | **Ollama** | llama3, mistral, etc. — local RAG |
| Config | **TOML** + **serde** | Human-friendly config files |
| Logging | **tracing** + **tracing-subscriber** | Structured async-aware logs |
| CLI | **clap** | Rust argument parsing |

---

## 5. Directory Structure

```
memory-search/
├── Cargo.toml                  # Rust workspace
├── Cargo.lock
├── config/
│   ├── default.toml            # Default configuration
│   ├── wikipedia.toml          # Wikipedia mode overrides
│   ├── notes.toml              # Notes mode overrides
│   └── codebase.toml           # Codebase mode overrides
│
├── crates/
│   ├── ms-core/                # Core library (no I/O)
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── chunk.rs        # Chunk + ChunkMetadata types
│   │   │   ├── score.rs        # SearchResult, scoring types
│   │   │   └── config.rs       # Config structs
│   │
│   ├── ms-ingest/              # Ingestion pipeline
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── loader/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── text.rs     # Plain text loader
│   │   │   │   ├── pdf.rs      # PDF → text (pdfium-render)
│   │   │   │   ├── html.rs     # HTML → text (scraper)
│   │   │   │   ├── code.rs     # Code files (tree-sitter)
│   │   │   │   ├── csv.rs      # CSV/JSON structured data
│   │   │   │   └── wiki.rs     # Wikipedia XML dump parser
│   │   │   ├── chunker/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── sentence.rs # Sentence-aware chunking
│   │   │   │   ├── token.rs    # Token-window chunking
│   │   │   │   └── code.rs     # AST-aware code chunking
│   │   │   └── embedder.rs     # fastembed-rs wrapper
│   │
│   ├── ms-index/               # Index management
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── bm25.rs         # Tantivy BM25 index
│   │   │   ├── vector.rs       # pgvector / FAISS index
│   │   │   ├── fuzzy.rs        # Trigram + SymSpell index
│   │   │   └── postgres.rs     # PostgreSQL metadata store
│   │
│   ├── ms-search/              # Search engine
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── router.rs       # Query router + fan-out
│   │   │   ├── arms/
│   │   │   │   ├── bm25.rs     # BM25 search arm
│   │   │   │   ├── fuzzy.rs    # Fuzzy search arm
│   │   │   │   └── semantic.rs # Vector search arm
│   │   │   └── merge.rs        # RRF + result merging
│   │
│   ├── ms-rerank/              # Re-ranking layer
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── rrf.rs          # Reciprocal Rank Fusion
│   │   │   ├── cross_encoder.rs # ONNX cross-encoder (ms-marco)
│   │   │   └── llm_rerank.rs   # Ollama LLM re-ranker
│   │
│   ├── ms-rag/                 # RAG answer generation
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── provider/
│   │   │   │   ├── mod.rs      # LLMProvider trait
│   │   │   │   ├── ollama.rs
│   │   │   │   ├── anthropic.rs
│   │   │   │   └── openai.rs
│   │   │   └── rag.rs          # Context assembly + prompting
│   │
│   ├── ms-server/              # Axum HTTP server
│   │   ├── src/
│   │   │   ├── main.rs
│   │   │   ├── routes/
│   │   │   │   ├── search.rs   # POST /search
│   │   │   │   ├── ingest.rs   # POST /ingest
│   │   │   │   └── health.rs   # GET /health
│   │   │   └── state.rs        # AppState (shared engine)
│   │
│   └── ms-cli/                 # CLI binary
│       ├── src/
│       │   └── main.rs         # clap commands: index, search, serve
│
├── python/                     # PyO3 Python bindings
│   ├── ms_python/
│   │   ├── __init__.py
│   │   └── search.pyi          # Type stubs
│   ├── Cargo.toml
│   └── src/lib.rs              # PyO3 #[pymodule]
│
├── api/                        # FastAPI bridge
│   ├── main.py
│   ├── routes/
│   │   ├── search.py
│   │   └── ingest.py
│   └── requirements.txt
│
├── ui/                         # React frontend
│   ├── package.json
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── SearchBar.tsx
│   │   │   ├── ResultCard.tsx
│   │   │   ├── ModeSelector.tsx
│   │   │   └── RerankerPicker.tsx
│   │   └── api/
│   │       └── client.ts
│   └── public/
│
├── models/                     # Downloaded offline models (gitignored)
│   ├── all-MiniLM-L6-v2/
│   └── ms-marco-MiniLM-L6-v2/
│
├── data/                       # Local data storage (gitignored)
│   ├── tantivy-index/
│   ├── trigram-index/
│   └── uploads/
│
├── scripts/
│   ├── download_models.sh      # Download all offline models
│   ├── setup_postgres.sh       # Init PostgreSQL + pgvector
│   └── ingest_wikipedia.sh     # Stream Wikipedia XML dump
│
└── docs/
    ├── architecture.md
    ├── chunking-strategies.md
    └── benchmark-results.md
```

---

## 6. Data Modes

MemorySearch is **mode-aware**. Each mode has different loaders, chunking strategies, index settings, and search defaults.

### Notes mode (second brain)

**Target:** Obsidian vaults, plain markdown, Notion exports, daily journals.

```toml
# config/notes.toml
[mode]
name = "notes"
chunk_size = 256          # smaller chunks, more precise recall
chunk_overlap = 32
chunker = "sentence"
embed_model = "all-MiniLM-L6-v2"
vector_index = "pgvector-hnsw"

[search]
default_top_k = 10
fuzzy_enabled = true
semantic_weight = 0.6
bm25_weight = 0.3
fuzzy_weight = 0.1
```

### Codebase mode

**Target:** Source code files, docstrings, README files, inline comments.

```toml
# config/codebase.toml
[mode]
name = "codebase"
chunker = "code"          # AST-aware via tree-sitter
chunk_size = 512
embed_model = "all-MiniLM-L6-v2"
# code embed alternative: "jinaai/jina-embeddings-v2-base-code"

[search]
default_top_k = 8
fuzzy_enabled = true      # typos in function names
semantic_weight = 0.5
bm25_weight = 0.4         # code has precise keywords (fn names, types)
fuzzy_weight = 0.1
```

### Wikipedia mode

**Target:** Full Wikipedia XML dump (~22GB compressed, ~90GB uncompressed).

```toml
# config/wikipedia.toml
[mode]
name = "wikipedia"
chunker = "token"
chunk_size = 512
chunk_overlap = 64
embed_model = "all-MiniLM-L6-v2"
vector_index = "faiss-ivfpq"      # not pgvector — too large
faiss_nlist = 4096
faiss_pq_m = 16

[ingest]
stream_xml = true                 # streaming XML parser, not load-all
batch_size = 1000                 # embed 1000 chunks at a time
workers = 4
```

### PDF / Web mode

**Target:** Research papers, documentation sites, saved web pages.

```toml
# config/default.toml (covers PDF/web)
[mode]
name = "default"
chunker = "sentence"
chunk_size = 400
chunk_overlap = 50
```

---

## 7. Ingestion Pipeline

### Step-by-step flow

```
File / URL / dump
      │
      ▼
[1] LOADER         Raw bytes → clean UTF-8 text + page/line metadata
      │             • PDF: pdfium-render → text per page
      │             • HTML: scraper crate → strip tags
      │             • Code: read_to_string → preserve line numbers
      │             • Wiki: quick-xml streaming parser
      ▼
[2] CHUNKER        Long text → Vec<RawChunk{ text, start_byte, end_byte }>
      │             • Sentence: unicode-segmentation crate
      │             • Token: tiktoken-rs (cl100k_base)
      │             • Code: tree-sitter AST — split at function/class boundaries
      ▼
[3] EMBEDDER       Vec<RawChunk> → Vec<Chunk{ text, embedding: [f32; 384] }>
      │             • fastembed-rs: batch encode, 32 chunks/batch
      │             • Model: all-MiniLM-L6-v2 (ONNX, ~80MB)
      │             • Runs on CPU by default, CUDA optional
      ▼
[4] INDEXER        Chunk → 3 indexes simultaneously
      │             • Tantivy: add_document({ id, text, mode, source })
      │             • pgvector: INSERT INTO chunks (id, text, embedding, metadata)
      │             • Trigram: build in-memory then flush to disk
      ▼
[5] METADATA       Store source provenance in PostgreSQL
                   • doc_id, source_file, url, ingested_at, mode, chunk_count
```

### Chunker implementations

#### Sentence chunker (notes, PDFs)

```rust
// crates/ms-ingest/src/chunker/sentence.rs
use unicode_segmentation::UnicodeSegmentation;

pub struct SentenceChunker {
    pub chunk_size: usize,   // target chars per chunk
    pub overlap: usize,      // overlap chars with adjacent chunk
}

impl SentenceChunker {
    pub fn chunk(&self, text: &str) -> Vec<String> {
        let sentences: Vec<&str> = text.unicode_sentences().collect();
        let mut chunks = Vec::new();
        let mut current = String::new();

        for sentence in &sentences {
            if current.len() + sentence.len() > self.chunk_size && !current.is_empty() {
                chunks.push(current.clone());
                // keep overlap: take last N chars
                let overlap_start = current.len().saturating_sub(self.overlap);
                current = current[overlap_start..].to_string();
            }
            current.push_str(sentence);
        }
        if !current.is_empty() { chunks.push(current); }
        chunks
    }
}
```

#### Code chunker (AST-aware)

```rust
// crates/ms-ingest/src/chunker/code.rs
// Uses tree-sitter to split at function/class/module boundaries

pub struct CodeChunker {
    pub language: SupportedLanguage,
    pub max_chunk_tokens: usize,
}

pub enum SupportedLanguage {
    Rust, Python, TypeScript, JavaScript, Go, Java, Cpp,
}

impl CodeChunker {
    pub fn chunk(&self, source: &str) -> Vec<CodeChunk> {
        let mut parser = tree_sitter::Parser::new();
        parser.set_language(self.language.ts_language()).unwrap();
        let tree = parser.parse(source, None).unwrap();

        // Walk AST, collect function_definition / function_item / class_definition nodes
        let mut chunks = Vec::new();
        self.collect_nodes(tree.root_node(), source, &mut chunks);
        chunks
    }
}
```

---

## 8. Storage Layer

### PostgreSQL schema

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm; -- for trigram fuzzy in SQL (optional)

-- Documents table (one row per ingested file/URL)
CREATE TABLE documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_path TEXT NOT NULL,
    source_url  TEXT,
    mode        TEXT NOT NULL,  -- 'notes' | 'codebase' | 'wikipedia' | 'default'
    title       TEXT,
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    chunk_count INT DEFAULT 0,
    hash        TEXT            -- SHA256 of content (skip re-ingest if unchanged)
);

-- Chunks table (one row per chunk, with embedding)
CREATE TABLE chunks (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id      UUID REFERENCES documents(id) ON DELETE CASCADE,
    text        TEXT NOT NULL,
    embedding   VECTOR(384),   -- all-MiniLM-L6-v2 produces 384-dim
    chunk_index INT NOT NULL,
    page        INT,
    line_start  INT,
    line_end    INT,
    mode        TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for ANN search (pgvector)
CREATE INDEX chunks_embedding_hnsw
    ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Full-text index (PostgreSQL tsvector, backup to Tantivy)
CREATE INDEX chunks_fts ON chunks USING GIN (to_tsvector('english', text));

-- Metadata index
CREATE INDEX chunks_mode_idx ON chunks (mode);
CREATE INDEX chunks_doc_idx  ON chunks (doc_id);
```

### Tantivy schema (full-text BM25)

```rust
// crates/ms-index/src/bm25.rs
use tantivy::{schema::*, Index};

pub fn build_schema() -> Schema {
    let mut builder = Schema::builder();
    builder.add_text_field("id",   STRING | STORED);   // UUID
    builder.add_text_field("text", TEXT | STORED);     // chunk text (BM25 indexed)
    builder.add_text_field("mode", STRING | STORED);   // filter by mode
    builder.add_text_field("source", STRING | STORED); // source file path
    builder.add_u64_field("chunk_index", STORED);
    builder.build()
}

pub struct BM25Index {
    index: Index,
    schema: Schema,
    reader: IndexReader,
}

impl BM25Index {
    pub fn search(&self, query: &str, mode: &str, top_k: usize) -> Vec<BM25Result> {
        let searcher = self.reader.searcher();
        let text_field = self.schema.get_field("text").unwrap();
        let mode_field = self.schema.get_field("mode").unwrap();

        // BM25 query on text field, filtered by mode
        let text_query = QueryParser::for_index(&self.index, vec![text_field])
            .parse_query(query).unwrap();
        let mode_query = TermQuery::new(
            Term::from_field_text(mode_field, mode),
            IndexRecordOption::Basic
        );
        let combined = BooleanQuery::new(vec![
            (Occur::Must, text_query),
            (Occur::Must, Box::new(mode_query)),
        ]);

        let top_docs = searcher.search(&combined, &TopDocs::with_limit(top_k)).unwrap();
        top_docs.iter().map(|(score, doc_address)| {
            let doc = searcher.doc(*doc_address).unwrap();
            BM25Result {
                id: doc.get_first(self.schema.get_field("id").unwrap())
                    .unwrap().as_text().unwrap().to_string(),
                score: *score,
            }
        }).collect()
    }
}
```

---

## 9. The 3 Search Arms

All 3 arms run in **parallel** via `tokio::join!`.

```rust
// crates/ms-search/src/router.rs

pub async fn search(
    query: &str,
    mode: &str,
    top_k: usize,
    state: &AppState,
) -> Vec<SearchResult> {
    // Normalize query
    let normalized = normalize_query(query);

    // Embed query once (reuse across semantic arm)
    let query_embedding = state.embedder.embed_one(&normalized).await;

    // Run all 3 arms in parallel
    let (bm25_results, fuzzy_results, semantic_results) = tokio::join!(
        search_bm25(&normalized, mode, top_k * 2, &state.bm25_index),
        search_fuzzy(&normalized, mode, top_k * 2, &state.fuzzy_index),
        search_semantic(&query_embedding, mode, top_k * 2, &state.vector_index),
    );

    // Merge and re-rank
    merge_results(bm25_results, fuzzy_results, semantic_results, top_k, &state.config)
}
```

### Arm 1: BM25 exact keyword

```rust
// crates/ms-search/src/arms/bm25.rs
pub async fn search_bm25(
    query: &str,
    mode: &str,
    top_k: usize,
    index: &BM25Index,
) -> Vec<ScoredCandidate> {
    index.search(query, mode, top_k)
        .into_iter()
        .map(|r| ScoredCandidate {
            chunk_id: r.id,
            score: r.score as f64,
            arm: SearchArm::BM25,
        })
        .collect()
}
```

### Arm 2: Fuzzy match (trigram + SymSpell)

```rust
// crates/ms-search/src/arms/fuzzy.rs
//
// Step 1: SymSpell corrects the query ("resurch" → "research")
// Step 2: Corrected query is decomposed into character 3-grams
// Step 3: 3-gram overlap score = |query_grams ∩ doc_grams| / |query_grams ∪ doc_grams| (Jaccard)
// Step 4: Return top-K by Jaccard score

pub async fn search_fuzzy(
    query: &str,
    mode: &str,
    top_k: usize,
    index: &FuzzyIndex,
) -> Vec<ScoredCandidate> {
    // SymSpell correction
    let corrected = index.symspell.lookup_compound(query, 2); // max edit distance = 2

    // Trigram query
    let query_grams: HashSet<String> = trigrams(&corrected.term);
    index.search_by_trigrams(&query_grams, mode, top_k)
}

fn trigrams(s: &str) -> HashSet<String> {
    let padded = format!("  {}  ", s.to_lowercase());
    padded.chars()
        .collect::<Vec<_>>()
        .windows(3)
        .map(|w| w.iter().collect::<String>())
        .collect()
}
```

### Arm 3: Semantic / intent (vector ANN)

```rust
// crates/ms-search/src/arms/semantic.rs
//
// Uses pgvector's HNSW index for cosine similarity ANN search.
// For Wikipedia (massive mode), swaps to FAISS IVF-PQ.

pub async fn search_semantic(
    query_embedding: &[f32; 384],
    mode: &str,
    top_k: usize,
    pool: &PgPool,
) -> Vec<ScoredCandidate> {
    // pgvector cosine similarity query
    let rows = sqlx::query!(
        r#"
        SELECT id::text, 1 - (embedding <=> $1::vector) AS score
        FROM chunks
        WHERE mode = $2
        ORDER BY embedding <=> $1::vector
        LIMIT $3
        "#,
        query_embedding as &[f32],
        mode,
        top_k as i64,
    )
    .fetch_all(pool)
    .await
    .unwrap();

    rows.iter().map(|r| ScoredCandidate {
        chunk_id: r.id.clone().unwrap(),
        score: r.score.unwrap_or(0.0),
        arm: SearchArm::Semantic,
    }).collect()
}
```

---

## 10. Merge & Re-Ranking Layer

### RRF (Reciprocal Rank Fusion) — default fast mode

```rust
// crates/ms-search/src/merge.rs
//
// RRF formula: score(d) = Σ_arm  1 / (k + rank_of_d_in_arm)
// k = 60 (standard constant, reduces impact of top ranks)
//
// This is purely rank-based — no score normalization needed.
// Works well even when arm scores have very different scales.

const RRF_K: f64 = 60.0;

pub fn rrf_merge(
    bm25: Vec<ScoredCandidate>,
    fuzzy: Vec<ScoredCandidate>,
    semantic: Vec<ScoredCandidate>,
    top_k: usize,
) -> Vec<String> {
    let mut scores: HashMap<String, f64> = HashMap::new();

    for (arm_results, _arm) in [&bm25, &fuzzy, &semantic] {
        for (rank, candidate) in arm_results.iter().enumerate() {
            let rrf_score = 1.0 / (RRF_K + rank as f64 + 1.0);
            *scores.entry(candidate.chunk_id.clone()).or_insert(0.0) += rrf_score;
        }
    }

    let mut ranked: Vec<(String, f64)> = scores.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    ranked.into_iter().take(top_k).map(|(id, _)| id).collect()
}
```

### Cross-encoder re-ranker (quality mode)

```rust
// crates/ms-rerank/src/cross_encoder.rs
//
// Model: cross-encoder/ms-marco-MiniLM-L6-v2
// Input: (query, chunk_text) pair
// Output: relevance score (higher = more relevant)
// Runtime: ONNX via ort crate (CPU, ~50-200ms for 20 candidates)

use ort::{Environment, Session, Value};

pub struct CrossEncoderReranker {
    session: Session,
    tokenizer: Tokenizer,  // tokenizers crate
}

impl CrossEncoderReranker {
    pub fn rerank(&self, query: &str, candidates: &[(String, String)]) -> Vec<f32> {
        // candidates: Vec<(chunk_id, chunk_text)>
        // Returns scores in same order as candidates
        let pairs: Vec<(&str, &str)> = candidates
            .iter()
            .map(|(_, text)| (query, text.as_str()))
            .collect();

        // Tokenize all pairs as [CLS] query [SEP] passage [SEP]
        let encodings = self.tokenizer.encode_batch(
            pairs.iter().map(|(q, p)| format!("{} [SEP] {}", q, p)).collect(),
            true
        ).unwrap();

        // Run ONNX inference (batched)
        let scores = self.session.run(inputs).unwrap();
        scores.iter().map(|s| *s).collect()
    }
}
```

### LLM re-ranker (smart mode, slowest)

```rust
// crates/ms-rerank/src/llm_rerank.rs
//
// Sends query + top-N chunks to local Ollama model
// Prompt asks model to rank chunks 1-N by relevance

pub async fn llm_rerank(
    query: &str,
    candidates: &[(String, String)],  // (chunk_id, chunk_text)
    provider: &dyn LLMProvider,
) -> Vec<String> {
    let chunks_text = candidates.iter().enumerate()
        .map(|(i, (_, text))| format!("[{}] {}", i+1, &text[..200.min(text.len())]))
        .collect::<Vec<_>>()
        .join("\n\n");

    let prompt = format!(
        "Query: {}\n\nRank these passages from most to least relevant. \
         Reply ONLY with the numbers in order, comma-separated (e.g. 3,1,4,2):\n\n{}",
        query, chunks_text
    );

    let response = provider.complete(&prompt, &[]).await.unwrap();
    // Parse "3,1,4,2" → reorder candidates
    parse_ranking(&response, candidates)
}
```

---

## 11. RAG Layer (Optional)

```rust
// crates/ms-rag/src/rag.rs

pub struct RAGEngine {
    provider: Box<dyn LLMProvider>,
    top_k: usize,
    max_context_tokens: usize,
}

impl RAGEngine {
    pub async fn answer(
        &self,
        query: &str,
        chunks: &[Chunk],
    ) -> RAGResponse {
        // Assemble context from top-K chunks
        let context = chunks.iter()
            .map(|c| format!(
                "Source: {}\n{}\n---",
                c.metadata.source_file, c.text
            ))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "Answer the question based on the provided context. \
             If the answer is not in the context, say so.\n\n\
             Context:\n{}\n\nQuestion: {}\n\nAnswer:",
            context, query
        );

        let answer = self.provider.complete(&prompt, &[]).await.unwrap();

        RAGResponse {
            answer,
            sources: chunks.iter().map(|c| c.metadata.source_file.clone()).collect(),
            chunks: chunks.to_vec(),
        }
    }
}
```

### LLM provider implementations

```rust
// crates/ms-rag/src/provider/ollama.rs

pub struct OllamaProvider {
    base_url: String,  // default: "http://localhost:11434"
    model: String,     // e.g. "llama3", "mistral", "phi3"
}

#[async_trait]
impl LLMProvider for OllamaProvider {
    async fn complete(&self, prompt: &str, _context: &[&str]) -> Result<String> {
        let resp = reqwest::Client::new()
            .post(format!("{}/api/generate", self.base_url))
            .json(&serde_json::json!({
                "model": self.model,
                "prompt": prompt,
                "stream": false
            }))
            .send().await?
            .json::<serde_json::Value>().await?;
        Ok(resp["response"].as_str().unwrap_or("").to_string())
    }

    fn is_local(&self) -> bool { true }
}
```

---

## 12. Output Interfaces

### REST API (Axum)

```
POST /api/search
{
  "query": "how does rust ownership work",
  "mode": "codebase",
  "top_k": 10,
  "reranker": "cross_encoder",  // "rrf" | "cross_encoder" | "llm" | "none"
  "rag": true                   // include LLM answer
}

Response:
{
  "query": "how does rust ownership work",
  "answer": "Rust ownership is...",  // null if rag=false
  "results": [
    {
      "chunk_id": "...",
      "text": "Ownership is Rust's most unique feature...",
      "score": 0.94,
      "metadata": {
        "source_file": "docs/ownership.md",
        "page": null,
        "line_start": 42,
        "line_end": 89,
        "chunk_index": 3
      },
      "arm_scores": {
        "bm25": 8.4,
        "fuzzy": 0.72,
        "semantic": 0.89
      }
    }
  ],
  "latency_ms": {
    "embed": 12,
    "bm25": 3,
    "fuzzy": 2,
    "semantic": 8,
    "rerank": 94,
    "total": 121
  }
}

POST /api/ingest
{ "path": "/path/to/files", "mode": "codebase" }

GET  /api/modes          → list active modes + chunk counts
GET  /api/health         → { "status": "ok", "version": "..." }
DELETE /api/index/:mode  → clear an entire mode index
```

### Python library (PyO3)

```python
# After: pip install memory-search

from memory_search import MemorySearch, SearchConfig

ms = MemorySearch("./my-index")

# Ingest
ms.ingest("/path/to/notes", mode="notes")

# Search
results = ms.search(
    query="how does attention work",
    mode="notes",
    top_k=10,
    reranker="cross_encoder"
)

for r in results:
    print(r.score, r.text[:100], r.metadata.source_file)
```

### CLI

```bash
# Index a directory
ms-search index ./my-notes --mode notes

# Index a Wikipedia dump
ms-search index ~/wikipedia/enwiki-latest-articles.xml.bz2 --mode wikipedia

# Search
ms-search search "how does attention work" --mode notes --top-k 10

# Search with re-ranker
ms-search search "rust ownership" --mode codebase --reranker cross_encoder

# Start web server
ms-search serve --port 8080

# Show index stats
ms-search stats
```

---

## 13. Build Phases (Roadmap)

| Phase | Goal | Key deliverable |
|-------|------|----------------|
| 1 | Core engine + CLI | Rust crate, Tantivy BM25, pgvector, RRF merge, `ms-search search` CLI |
| 2 | All 3 arms | Fuzzy arm (SymSpell + trigram), semantic arm (all-MiniLM), parallel Tokio |
| 3 | Re-rankers | Cross-encoder (ONNX, offline), LLM re-ranker (Ollama), re-ranker picker |
| 4 | Web UI | React + FastAPI, search UI, result cards, mode selector, ingest panel |
| 5 | Data modes | Wikipedia XML streaming, code AST chunking (tree-sitter), Notes Markdown |
| 6 | Python lib | PyO3 bindings, `pip install memory-search` |
| 7 | Performance | FAISS IVF-PQ for Wikipedia, batch ingest, benchmark suite |

---

## 14. Phase 1 — Setup & Core Engine

### Prerequisites

```bash
# Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update stable

# PostgreSQL with pgvector
# macOS
brew install postgresql@16
brew install pgvector  # or: follow https://github.com/pgvector/pgvector

# Ubuntu/Debian
sudo apt install postgresql-16 postgresql-16-pgvector

# Start PostgreSQL
brew services start postgresql@16  # macOS
sudo systemctl start postgresql     # Linux
```

### Initialize database

```bash
# Run setup script
chmod +x scripts/setup_postgres.sh
./scripts/setup_postgres.sh

# Or manually:
psql -U postgres -c "CREATE DATABASE memory_search;"
psql -U postgres -d memory_search -c "CREATE EXTENSION vector;"
psql -U postgres -d memory_search -f schema.sql
```

### Download offline models

```bash
chmod +x scripts/download_models.sh
./scripts/download_models.sh

# What this downloads (~350MB total):
# models/all-MiniLM-L6-v2/         ~80MB  (embedding model, ONNX)
# models/ms-marco-MiniLM-L6-v2/   ~80MB  (cross-encoder re-ranker, ONNX)
# data/symspell-dict/frequency.txt ~30MB  (SymSpell frequency dictionary)
```

### Cargo workspace setup

```toml
# Cargo.toml (root workspace)
[workspace]
members = [
    "crates/ms-core",
    "crates/ms-ingest",
    "crates/ms-index",
    "crates/ms-search",
    "crates/ms-rerank",
    "crates/ms-rag",
    "crates/ms-server",
    "crates/ms-cli",
]
resolver = "2"

[workspace.dependencies]
tokio       = { version = "1", features = ["full"] }
serde       = { version = "1", features = ["derive"] }
serde_json  = "1"
sqlx        = { version = "0.7", features = ["postgres", "uuid", "runtime-tokio-native-tls"] }
tantivy     = "0.21"
fastembed   = "3"
uuid        = { version = "1", features = ["v4"] }
axum        = "0.7"
clap        = { version = "4", features = ["derive"] }
tracing     = "0.1"
tracing-subscriber = "0.3"
anyhow      = "1"
async-trait = "0.1"
reqwest     = { version = "0.11", features = ["json"] }
```

### Phase 1 key crate: ms-core

```toml
# crates/ms-core/Cargo.toml
[dependencies]
serde      = { workspace = true }
uuid       = { workspace = true }
```

```rust
// crates/ms-core/src/lib.rs

pub mod chunk;
pub mod score;
pub mod config;

// Re-exports
pub use chunk::{Chunk, ChunkMetadata, DataMode};
pub use score::{SearchResult, SearchArm, ScoredCandidate};
```

### First working CLI command

```bash
# Build everything
cargo build --release

# Index a test directory
./target/release/ms-search index ./test-data --mode notes

# Run a search
./target/release/ms-search search "machine learning" --mode notes

# Expected output:
# [0.94] machine_learning_notes.md:42  "Machine learning is a subset of AI..."
# [0.87] ml_resources.md:12            "Key ML concepts include supervised..."
# [0.81] deep_learning.md:67           "Neural networks form the basis of..."
```

---

## 15. Phase 2 — Fuzzy + Semantic Arms

### Download embedding model (fastembed-rs)

```rust
// crates/ms-ingest/src/embedder.rs
use fastembed::{EmbeddingModel, FlagEmbedding, InitOptions};

pub struct Embedder {
    model: FlagEmbedding,
}

impl Embedder {
    pub fn new(model_dir: &str) -> Self {
        let model = FlagEmbedding::try_new(InitOptions {
            model_name: EmbeddingModel::AllMiniLML6V2,
            cache_dir: PathBuf::from(model_dir),
            show_download_progress: true,
            ..Default::default()
        }).unwrap();
        Self { model }
    }

    pub fn embed_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        self.model.embed(texts.to_vec(), Some(32)).unwrap()
    }

    pub fn embed_one(&self, text: &str) -> Vec<f32> {
        self.embed_batch(&[text]).remove(0)
    }
}
```

### SymSpell dictionary build

```bash
# Build frequency dictionary from your corpus (optional, default dict included)
ms-search build-symspell-dict --corpus ./data/corpus.txt --output ./data/symspell-dict/
```

---

## 16. Phase 3 — Re-Rankers

### Cross-encoder model (ONNX, offline)

```bash
# Convert HuggingFace model to ONNX (run once)
pip install optimum[onnxruntime]
optimum-cli export onnx \
    --model cross-encoder/ms-marco-MiniLM-L6-v2 \
    --task text-classification \
    ./models/ms-marco-MiniLM-L6-v2/
```

### Ollama setup (LLM re-ranker + RAG)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model (pick one based on your hardware)
ollama pull llama3          # 4.7GB  — best quality
ollama pull mistral         # 4.1GB  — fast + good
ollama pull phi3            # 2.2GB  — lightweight
ollama pull nomic-embed-text # alternative embedding model

# Verify
ollama list
```

---

## 17. Phase 4 — Full-Stack Web UI

### FastAPI bridge

```bash
cd api/
pip install fastapi uvicorn httpx

# Start API server (talks to Rust server on :8080)
uvicorn main:app --reload --port 8000
```

### React UI setup

```bash
cd ui/
npm create vite@latest . -- --template react-ts
npm install
npm run dev  # starts on :5173
```

### Key UI components

```typescript
// ui/src/components/SearchBar.tsx
// Features:
// - Debounced search as-you-type (300ms)
// - Mode selector (Notes / Codebase / Wikipedia / All)
// - Re-ranker picker (Fast / Quality / Smart)
// - RAG toggle (pure search vs answer + sources)

// ui/src/components/ResultCard.tsx
// Shows:
// - Chunk text with query term highlighting
// - Source file + line numbers
// - Per-arm scores (BM25 / Fuzzy / Semantic) as mini bars
// - Expand/collapse long chunks
// - Copy chunk button
```

---

## 18. Phase 5 — Data Modes

### Wikipedia XML streaming ingest

```bash
# Download Wikipedia dump (~22GB compressed)
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

# Stream-ingest (does not load full XML into memory)
ms-search index enwiki-latest-pages-articles.xml.bz2 --mode wikipedia

# This will:
# 1. Stream-parse XML with quick-xml
# 2. Extract article text, skip talk/user pages
# 3. Chunk into 512-token windows
# 4. Batch-embed (32 chunks/batch, fastembed)
# 5. Insert into FAISS IVF-PQ index
# Expected time: 6-12 hours on modern CPU
# Expected index size: ~50GB (FAISS PQ compressed from ~150GB raw vectors)
```

### Codebase tree-sitter setup

```toml
# crates/ms-ingest/Cargo.toml (add tree-sitter grammars)
[dependencies]
tree-sitter       = "0.22"
tree-sitter-rust  = "0.21"
tree-sitter-python = "0.21"
tree-sitter-typescript = "0.21"
tree-sitter-go    = "0.21"
```

---

## 19. Configuration Reference

```toml
# config/default.toml

[database]
url = "postgresql://localhost/memory_search"
max_connections = 10

[indexes]
tantivy_path = "./data/tantivy-index"
trigram_path = "./data/trigram-index"
faiss_path   = "./data/faiss-index"

[embedding]
model_path = "./models/all-MiniLM-L6-v2"
batch_size = 32
device = "cpu"  # "cpu" | "cuda" | "metal"

[search]
default_top_k = 10
default_reranker = "rrf"  # "rrf" | "cross_encoder" | "llm" | "none"
bm25_weight    = 0.35
fuzzy_weight   = 0.15
semantic_weight = 0.50

[rerank]
cross_encoder_path = "./models/ms-marco-MiniLM-L6-v2"
cross_encoder_max_candidates = 20

[rag]
enabled = true
provider = "ollama"           # "ollama" | "anthropic" | "openai" | "none"
ollama_base_url = "http://localhost:11434"
ollama_model = "llama3"
anthropic_model = "claude-sonnet-4-20250514"
max_context_chunks = 5

[server]
host = "0.0.0.0"
port = 8080

[logging]
level = "info"  # "trace" | "debug" | "info" | "warn" | "error"
```

---

## 20. API Reference

| Method | Endpoint | Description |
|--------|---------|-------------|
| `POST` | `/api/search` | Main search endpoint |
| `POST` | `/api/ingest` | Ingest files/directories |
| `GET`  | `/api/modes` | List modes + chunk counts |
| `GET`  | `/api/documents` | List ingested documents |
| `DELETE` | `/api/documents/:id` | Remove a document |
| `DELETE` | `/api/index/:mode` | Wipe a mode's index |
| `GET`  | `/api/health` | Health check |
| `GET`  | `/api/stats` | Index stats + performance |
| `GET`  | `/api/config` | Current configuration |
| `POST` | `/api/config` | Update configuration at runtime |

---

## 21. Offline Model Downloads

```bash
#!/bin/bash
# scripts/download_models.sh

set -e
mkdir -p models data/symspell-dict

echo "Downloading all-MiniLM-L6-v2 (embedding model)..."
# fastembed-rs downloads this automatically on first use
# Or manually via huggingface-hub:
pip install huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='sentence-transformers/all-MiniLM-L6-v2',
    local_dir='./models/all-MiniLM-L6-v2'
)
"

echo "Downloading ms-marco cross-encoder (re-ranker)..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='cross-encoder/ms-marco-MiniLM-L6-v2',
    local_dir='./models/ms-marco-MiniLM-L6-v2'
)
"

echo "Downloading SymSpell frequency dictionary..."
curl -L https://raw.githubusercontent.com/wolfgarbe/SymSpell/master/SymSpell/frequency_dictionary_en_82_765.txt \
    -o data/symspell-dict/frequency.txt

echo "All models downloaded."
echo "Total size: ~$(du -sh models/ | cut -f1)"
```

---

## 22. Performance Benchmarks & Targets

| Mode | Dataset | Index time | Search latency (p50) | Search latency (p99) |
|------|---------|-----------|---------------------|---------------------|
| Notes | 10K chunks | ~30s | 8ms | 25ms |
| Codebase | 50K chunks | ~3min | 12ms | 40ms |
| Wikipedia | 5M chunks | ~8hrs | 45ms | 120ms |

**With re-rankers (on top of base search):**

| Re-ranker | Added latency (20 candidates) |
|-----------|-------------------------------|
| RRF | +1ms |
| Cross-encoder (ONNX, CPU) | +80–200ms |
| LLM (Ollama llama3, CPU) | +2,000–5,000ms |
| LLM (Ollama llama3, GPU) | +200–500ms |

**Memory footprint:**

| Component | RAM usage |
|-----------|----------|
| all-MiniLM-L6-v2 (ONNX) | ~250MB |
| Tantivy BM25 index (50K chunks) | ~200MB |
| pgvector HNSW (50K × 384-dim) | ~1.2GB |
| ms-marco cross-encoder | ~100MB |
| Total (codebase mode) | ~1.8GB |

---

## 23. Contributing

### Development setup

```bash
git clone https://github.com/your-org/memory-search
cd memory-search
cp config/default.toml config/local.toml  # edit local config
./scripts/setup_postgres.sh
./scripts/download_models.sh
cargo build
cargo test
```

### Running tests

```bash
cargo test                      # all unit tests
cargo test -p ms-search         # search crate only
cargo test -- --test-thread 1   # single-threaded (for DB tests)
```

### Code style

```bash
cargo fmt        # format
cargo clippy     # lint
cargo doc --open # docs
```

---

## License

MIT License — see `LICENSE`.

---

*Built with inspiration from LlamaIndex, FAISS, AnythingLLM, Typesense, and OpenWebUI.*
*Core engine: Rust. Fast by default. Offline always.*