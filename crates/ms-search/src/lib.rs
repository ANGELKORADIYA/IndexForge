use ms_core::{ScoredCandidate, SearchArm};
use ms_index::bm25::BM25Index;

pub async fn search_bm25(
    query: &str,
    mode: &str,
    top_k: usize,
    index: &BM25Index,
) -> Vec<ScoredCandidate> {
    index.search(query, mode, top_k)
        .unwrap_or_default()
        .into_iter()
        .map(|r| ScoredCandidate {
            chunk_id: r.id,
            text: r.text,
            score: r.score as f64,
            arm: SearchArm::BM25,
        })
        .collect()
}
