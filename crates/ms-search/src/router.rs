use ms_core::{ScoredCandidate, SearchArm, SearchResult};
use ms_index::bm25::BM25Index;
use ms_index::fuzzy::FuzzyIndex;
use ms_index::vector::search_vector;
use ms_ingest::embedder::Embedder;
use sqlx::PgPool;
use anyhow::Result;

use crate::merge::rrf_merge;

/// Full 3-arm parallel search with RRF merge.
///
/// Runs BM25, fuzzy, and semantic arms concurrently via tokio::join!,
/// then merges with Reciprocal Rank Fusion.
fn candidate_to_result(c: &ScoredCandidate) -> SearchResult {
    let mut arm_scores = std::collections::HashMap::new();
    arm_scores.insert(c.arm, c.score);
    SearchResult {
        chunk_id: c.chunk_id.clone(),
        text: c.text.clone(),
        score: c.score,
        arm_scores,
        metadata: ms_core::ChunkMetadata {
            source_file: String::new(),
            page: None,
            line_start: None,
            line_end: None,
            chunk_index: 0,
            doc_id: uuid::Uuid::nil(),
            mode: ms_core::DataMode::Default,
        },
    }
}

pub async fn search_with_arms(
    query: &str,
    mode: &str,
    top_k: usize,
    bm25_index: &BM25Index,
    fuzzy_index: &FuzzyIndex,
    embedder: &mut Embedder,
    pool: &PgPool,
    include_arms: bool,
    arm_top_k: usize,
) -> Result<(Vec<SearchResult>, Option<ms_core::PerArmResults>)> {
    let query_embedding = embedder.embed_one(query)?;

    let (bm25_results, fuzzy_results, semantic_results) = tokio::join!(
        async { search_bm25(query, mode, top_k * 2, bm25_index).await },
        async { search_fuzzy(query, mode, top_k * 2, fuzzy_index).await },
        search_vector(pool, query_embedding, mode, top_k * 2),
    );

    let sem_candidates: Vec<ScoredCandidate> = match semantic_results {
        Ok(sem) => sem
            .into_iter()
            .map(|r| ScoredCandidate {
                chunk_id: r.chunk_id,
                text: r.text,
                score: r.score,
                arm: SearchArm::Semantic,
            })
            .collect(),
        Err(e) => {
            eprintln!("[warn] Semantic arm failed: {e}");
            vec![]
        }
    };

    let arm_results = if include_arms {
        Some(ms_core::PerArmResults {
            bm25: bm25_results.iter().take(arm_top_k).map(candidate_to_result).collect(),
            fuzzy: fuzzy_results.iter().take(arm_top_k).map(candidate_to_result).collect(),
            semantic: sem_candidates.iter().take(arm_top_k).map(candidate_to_result).collect(),
        })
    } else {
        None
    };

    let all: Vec<Vec<ScoredCandidate>> = vec![bm25_results, fuzzy_results, sem_candidates];
    Ok((rrf_merge(all, top_k), arm_results))
}

/// Full 3-arm parallel search with RRF merge.
pub async fn search(
    query: &str,
    mode: &str,
    top_k: usize,
    bm25_index: &BM25Index,
    fuzzy_index: &FuzzyIndex,
    embedder: &mut Embedder,
    pool: &PgPool,
) -> Result<Vec<SearchResult>> {
    let (merged, _) = search_with_arms(query, mode, top_k, bm25_index, fuzzy_index, embedder, pool, false, top_k).await?;
    Ok(merged)
}

/// BM25 exact keyword arm.
pub async fn search_bm25(
    query: &str,
    mode: &str,
    top_k: usize,
    index: &BM25Index,
) -> Vec<ScoredCandidate> {
    index
        .search(query, mode, top_k)
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

/// Fuzzy trigram + SymSpell arm.
pub async fn search_fuzzy(
    query: &str,
    mode: &str,
    top_k: usize,
    index: &FuzzyIndex,
) -> Vec<ScoredCandidate> {
    index
        .search(query, mode, top_k)
        .into_iter()
        .map(|r| ScoredCandidate {
            chunk_id: r.chunk_id,
            text: r.text,
            score: r.score,
            arm: SearchArm::Fuzzy,
        })
        .collect()
}
