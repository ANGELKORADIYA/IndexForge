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
*Pending...*

## Phase 3: Re-rankers & RAG Layer
*Pending...*

## Phase 4: Full-Stack Web UI
*Pending...*

## Phase 5: Advanced Data Modes & Python Lib
*Pending...*

---

## Architectural Rules
1. **100% Offline Default:** No external API calls by default. Models must be loaded from local files.
2. **Modular Workspace:** Functionality must be strictly encapsulated within its respective crate (`ms-core`, `ms-ingest`, `ms-index`, etc.).
3. **Database Dependency:** Requires PostgreSQL with `pgvector` enabled.