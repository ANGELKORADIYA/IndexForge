use std::collections::HashMap;
use ms_core::{ScoredCandidate, SearchArm, SearchResult, ChunkMetadata, DataMode};
use uuid::Uuid;

/// Reciprocal Rank Fusion merge.
///
/// Formula: score(d) = Σ_arm  1 / (k + rank_i(d))
/// where k=60 (standard RRF constant).
///
/// Deduplicates by chunk_id across arms, returns top_k SearchResults.
pub fn rrf_merge(
    arms: Vec<Vec<ScoredCandidate>>,
    top_k: usize,
) -> Vec<SearchResult> {
    const K: f64 = 60.0;

    // chunk_id -> (rrf_score, text, arm_scores)
    let mut scores: HashMap<String, (f64, String, HashMap<SearchArm, f64>)> = HashMap::new();

    for arm_results in &arms {
        for (rank, candidate) in arm_results.iter().enumerate() {
            let rrf = 1.0 / (K + (rank + 1) as f64);
            let entry = scores
                .entry(candidate.chunk_id.clone())
                .or_insert_with(|| (0.0, candidate.text.clone(), HashMap::new()));
            entry.0 += rrf;
            entry.2.insert(candidate.arm, candidate.score);
        }
    }

    // Sort by accumulated RRF score descending
    let mut ranked: Vec<(String, f64, String, HashMap<SearchArm, f64>)> = scores
        .into_iter()
        .map(|(id, (rrf, text, arm_scores))| (id, rrf, text, arm_scores))
        .collect();

    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    ranked
        .into_iter()
        .take(top_k)
        .map(|(chunk_id, rrf_score, text, arm_scores)| SearchResult {
            chunk_id: chunk_id.clone(),
            text,
            score: rrf_score,
            arm_scores,
            // Minimal metadata — enrichment from DB happens at the API layer (Phase 4)
            metadata: ChunkMetadata {
                source_file: String::new(),
                page: None,
                line_start: None,
                line_end: None,
                chunk_index: 0,
                doc_id: Uuid::nil(),
                mode: DataMode::Default,
            },
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidates(arm: SearchArm, ids: &[&str]) -> Vec<ScoredCandidate> {
        ids.iter()
            .enumerate()
            .map(|(i, id)| ScoredCandidate {
                chunk_id: id.to_string(),
                text: format!("text for {id}"),
                score: 1.0 / (i + 1) as f64,
                arm,
            })
            .collect()
    }

    #[test]
    fn test_rrf_deduplication() {
        let bm25 = make_candidates(SearchArm::BM25, &["a", "b", "c"]);
        let sem = make_candidates(SearchArm::Semantic, &["a", "c", "d"]);
        let results = rrf_merge(vec![bm25, sem], 5);

        // "a" appears in both arms — should rank highest
        assert_eq!(results[0].chunk_id, "a");
        // Each chunk_id is unique in output
        let ids: Vec<&str> = results.iter().map(|r| r.chunk_id.as_str()).collect();
        let unique: std::collections::HashSet<&str> = ids.iter().copied().collect();
        assert_eq!(ids.len(), unique.len());
    }

    #[test]
    fn test_rrf_arm_scores_captured() {
        let bm25 = make_candidates(SearchArm::BM25, &["x"]);
        let fuzzy = make_candidates(SearchArm::Fuzzy, &["x"]);
        let results = rrf_merge(vec![bm25, fuzzy], 1);
        assert!(results[0].arm_scores.contains_key(&SearchArm::BM25));
        assert!(results[0].arm_scores.contains_key(&SearchArm::Fuzzy));
    }
}
