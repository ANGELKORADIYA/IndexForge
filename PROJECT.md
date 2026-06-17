# MemorySearch Cross-Module Knowledge Base

This file serves as a central repository for cross-module decisions, architectural boundaries, and phase-to-phase knowledge transfer.

## Phase 1: Setup & Core Engine
*Pending execution...*

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