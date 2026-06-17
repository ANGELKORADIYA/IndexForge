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
