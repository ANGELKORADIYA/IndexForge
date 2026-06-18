# MemorySearch Cross-Module Knowledge Base

This file serves as a central repository for cross-module decisions, architectural boundaries, and phase-to-phase knowledge transfer.

## Phase 1: Setup & Core Engine
**Status:** Complete
**Deliverables:**
- Rust 2021 Workspace with 8 core crates.
- Initial PostgreSQL schema with `pgvector` extension and indexing.
- `SentenceChunker` (Unicode-safe char-based) in `ms-ingest`.
- `BM25Index` wrapper for Tantivy in `ms-index`.
- `ms-cli` supporting `index` and `search` (BM25 only).

**Technical Decisions:**
- **Unicode Safety:** `SentenceChunker` uses character-based counting and slicing to avoid panics on multi-byte characters.
- **Tantivy Integration:** Uses a boolean query to filter by `mode` while searching `text`.
- **Core Types:** Centralized in `ms-core` (`Chunk`, `SearchResult`, `SearchArm`, `DataMode`) to ensure consistency across search arms.
- **Project Structure:** All index files stored in `./data/tantivy-index` by default.

## Phase 1.5: Infrastructure & Docker
**Status:** Complete
**Deliverables:**
- `Dockerfile`: Optimized for simplicity and workspace-wide builds.
- `docker-compose.yml`: Includes `db` (pgvector), `app-dev` (with volumes), and `app-prod` profiles.
- `scripts/download_models.sh`: Orchestrates model downloads (currently handles SymSpell dictionary).
- `SETUP_GUIDE.md`: Comprehensive guide for Docker and manual local setup.
- `.env.example`: Template for environment configuration.


## Phase 2: Fuzzy & Semantic Search Arms
**Status:** Complete
**Deliverables:**
- `ms-index/src/vector.rs`: pgvector HNSW semantic search (cosine distance via `<=>` operator).
- `ms-index/src/fuzzy.rs`: Pure-Rust trigram fuzzy index (character n-grams + Jaccard similarity).
- `ms-search/src/router.rs`: 3-arm parallel query router using `tokio::join!`.
- `ms-search/src/merge.rs`: RRF (Reciprocal Rank Fusion) merge with k=60.
- `ms-cli`: New `search-all` command wiring all 3 arms + RRF.

**Technical Decisions:**
- **Fuzzy Arm:** Pure-Rust character trigrams (no symspell). Space-padded trigrams handle boundary matching. 1-char typo only invalidates 3/N trigrams, giving natural tolerance.
- **Vector Arm:** Delegates to pgvector HNSW `<=>` cosine distance. Score = `1 - distance`.
- **RRF Constant k=60:** Standard value from original Cormack et al. paper. Stable performance across arm count.
- **Semantic failure is non-fatal:** If pgvector query fails, BM25+fuzzy results are returned without error.
- **fastembed 5.x:** Upgraded from 3.x. Uses `TextInitOptions` builder API. `embed()` now takes `&mut self`.
- **FuzzyIndex loading:** At search-all time, fuzzy index is built from DB `chunks` table for the queried mode.

## Phase 3: Re-rankers & RAG Layer
**Status:** Complete
**Deliverables:**
- `ms-rerank/src/cross_encoder.rs`: Cross-encoder re-ranker using `fastembed`'s `TextRerank` model (`BGERerankerBase`).
- `ms-rag/src/provider.rs`: `LLMProvider` trait for abstracting LLM backends.
- Multiple RAG providers: `OllamaProvider` (default, local), `OpenRouterProvider` (if `OPENROUTER_API_KEY` is present), and `GeminiProvider` (if `GEMINI_API_KEY` is present).
- `ms-rag/src/pipeline.rs`: Assembles prompts using the top-K chunks and retrieves answers.
- `ms-cli`: Updated `search-all` command with `--rerank` and `--rag` flags.

**Technical Decisions:**
- **Re-ranker:** Switched from manually loading `ms-marco` via `ort` to using `fastembed`'s built-in `TextRerank` (`BGERerankerBase`). This simplifies the dependency graph and ensures compatibility.
- **RAG LLM routing:** Automatically routes to the highest-priority API key found in the environment, falling back to local `localhost:11434` (Ollama) to preserve the "100% offline default" rule while allowing easy cloud integration.

## Phase 4: Full-Stack Web UI
**Status:** Complete
**Deliverables:**
- `ms-server`: Axum-based REST API server handling `/api/index` and `/api/search`.
- `ui`: React + Vite frontend with glassmorphism UI, search filters, and visualization of RAG answers and RRF arm scores.
- Concurrent model state sharing using `tokio::sync::Mutex` for `fastembed`'s mutable constraints.

## Phase 5: Advanced Data Modes & Python Lib
**Status:** Complete
**Deliverables:**
- `ms-ingest/loader`: Native parsers for Wikipedia (`.zim`), PDF, DOCX, JSON, and CSV.
- Python bindings (`PyO3` / `maturin`): Built `ms-python` crate exposing a synchronous Python class for indexing and 3-arm searching.

## Phase 6: Advanced Capabilities (LlamaIndex / Typesense parity)
**Status:** Complete
**Deliverables:**
- **AST Code Chunking**: Integrated `tree-sitter` to parse code files into semantic blocks (functions, classes) instead of blindly slicing text.
- **Token-Aware Chunking**: Integrated `tiktoken-rs` (cl100k_base) to chunk purely by token limits for optimal density.
- **SymSpell Corrector**: Pure-Rust O(1) dictionary edit-distance spell checker to fix massive query typos before database hits.
- **LLM Re-ranker**: Alternative to cross-encoder. Feeds top chunks to local LLMs with a strict JSON prompt to intelligently re-sort the candidates.

---

## Architectural Rules
1. **100% Offline Default:** No external API calls by default. Models must be loaded from local files.
2. **Modular Workspace:** Functionality must be strictly encapsulated within its respective crate (`ms-core`, `ms-ingest`, `ms-index`, etc.).
3. **Database Dependency:** Requires PostgreSQL with `pgvector` enabled.