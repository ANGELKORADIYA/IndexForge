## Phase 1 (Complete)
Task 1: Workspace Scaffolding: complete (commits a41723b..HEAD, review clean)
Task 2: Database Initialization: complete (commits 49bd269..HEAD, review clean)
Task 3: Ingestion Pipeline (Text): complete (commits 027481d..HEAD, review clean)
Task 4: Search & CLI (Phase 1 Final): complete (commits d055dd2..HEAD, review clean)
Task 5: Infrastructure & Docker Setup: complete (Setup Guide, Dockerfile, Compose, Scripts)

## Phase 2 (Complete)
Task 1: Semantic Embedding Integration: complete
  - fastembed upgraded 3.x → 5.x (TextInitOptions builder API, &mut self embed)
  - ort/ort-sys version mismatch fixed by upgrading fastembed
  - Embedder wired into ms-cli index command; embeddings stored in pgvector
  - Config::default() fixed to read DATABASE_URL / TANTIVY_PATH from env

Task 2: Semantic Search Arm (pgvector): complete
  - ms-index/src/vector.rs: cosine ANN via <=> operator against HNSW index
  - Returns (1 - cosine_distance) as similarity score

Task 3: Fuzzy Search Arm: complete
  - ms-index/src/fuzzy.rs: pure-Rust character trigrams + Jaccard similarity
  - No external dict dependency; space-padded trigrams for boundary matching
  - Typo tolerance: 1-char error only invalidates 3 of N trigrams

Task 4: Query Router & RRF Merge: complete
  - ms-search/src/router.rs: tokio::join! runs all 3 arms concurrently
  - ms-search/src/merge.rs: Reciprocal Rank Fusion k=60 deduplicates across arms
  - ms-cli: new `search-all` subcommand exposes 3-arm merged search
  - Semantic arm failure is non-fatal (warns, falls back to BM25+fuzzy)

## Phase 3 (Pending)
Task 1: Cross-Encoder Re-ranker (ms-rerank)
Task 2: RAG Layer with Ollama (ms-rag)

## Phase 4 (Pending)
Task 1: REST API via Axum (ms-server)
Task 2: React Web UI (ui/)

## Phase 5 (Pending)
Task 1: Advanced Loaders (Wikipedia XML, tree-sitter AST)
Task 2: Python Bindings (PyO3)
