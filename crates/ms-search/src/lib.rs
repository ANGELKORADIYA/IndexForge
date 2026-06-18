pub mod router;
pub mod merge;

// Re-export the BM25 arm for ms-cli backward compatibility
pub use router::search_bm25;
