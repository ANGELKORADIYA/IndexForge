use serde::{Deserialize, Serialize};
use crate::chunk::ChunkMetadata;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub chunk_id: String,
    pub text: String,
    pub score: f64,
    pub metadata: ChunkMetadata,
    pub arm_scores: std::collections::HashMap<SearchArm, f64>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SearchArm {
    BM25,
    Fuzzy,
    Semantic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredCandidate {
    pub chunk_id: String,
    pub text: String,
    pub score: f64,
    pub arm: SearchArm,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagAnswer {
    pub answer: String,
    pub sources: Vec<SearchResult>,
    pub model: String,
}
