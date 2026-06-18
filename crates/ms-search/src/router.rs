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
pub async fn search(
    query: &str,
    mode: &str,
    top_k: usize,
    bm25_index: &BM25Index,
    fuzzy_index: &FuzzyIndex,
    embedder: &mut Embedder,
    pool: &PgPool,
) -> Result<Vec<SearchResult>> {
    // Embed the query once — reused by the semantic arm
    let query_embedding = embedder.embed_one(query)?;

    // Run all 3 arms in parallel
    let (bm25_results, fuzzy_results, semantic_results) = tokio::join!(
        async { search_bm25(query, mode, top_k * 2, bm25_index).await },
        async { search_fuzzy(query, mode, top_k * 2, fuzzy_index).await },
        search_vector(pool, query_embedding, mode, top_k * 2),
    );

    let mut all: Vec<Vec<ScoredCandidate>> = vec![bm25_results, fuzzy_results];

    match semantic_results {
        Ok(sem) => {
            let sem_candidates: Vec<ScoredCandidate> = sem
                .into_iter()
                .map(|r| ScoredCandidate {
                    chunk_id: r.chunk_id,
                    text: r.text,
                    score: r.score,
                    arm: SearchArm::Semantic,
                })
                .collect();
            all.push(sem_candidates);
        }
        Err(e) => {
            eprintln!("[warn] Semantic arm failed: {e}");
        }
    }

    Ok(rrf_merge(all, top_k))
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
