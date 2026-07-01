use axum::{extract::{State, Query}, Json};
use serde::{Deserialize, Serialize};
use ms_core::{SearchResult, RagAnswer};
use ms_index::fuzzy::FuzzyIndex;
use crate::state::AppState;

#[derive(Deserialize)]
pub struct SearchQuery {
    pub q: String,
    pub mode: String,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default)]
    pub rerank: bool,
    #[serde(default)]
    pub rag: bool,
    #[serde(default, alias = "show_arm_results")]
    pub arms: bool,
}

fn default_top_k() -> usize {
    10
}

#[derive(Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rag_answer: Option<RagAnswer>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arm_results: Option<ms_core::PerArmResults>,
}

pub async fn search_handler(
    State(state): State<AppState>,
    Query(query): Query<SearchQuery>,
) -> Result<Json<SearchResponse>, (axum::http::StatusCode, String)> {
    let index_path = std::env::var("TANTIVY_PATH").unwrap_or_else(|_| "./data/tantivy-index".to_string());
    let schema = ms_index::bm25::build_schema();
    
    let index = tantivy::Index::open_in_dir(&index_path)
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("Tantivy error: {}", e)))?;
    let reader = index.reader()
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("Reader error: {}", e)))?;
    let bm25 = ms_index::bm25::BM25Index { index, schema, reader };

    let mut fuzzy = FuzzyIndex::new();
    let rows = sqlx::query!("SELECT id::text, text FROM chunks WHERE mode = $1", query.mode)
        .fetch_all(&state.pool)
        .await
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("DB error: {}", e)))?;
    
    for row in rows {
        fuzzy.add(row.id.unwrap_or_default(), row.text);
    }

    let k_search = if query.rerank && query.arms { query.top_k * 5 } else if query.rerank { query.top_k * 2 } else { query.top_k };
    let mut embedder = state.embedder.lock().await;

    let (mut results, arm_results) = ms_search::router::search_with_arms(
        &query.q,
        &query.mode,
        k_search,
        &bm25,
        &fuzzy,
        &mut *embedder,
        &state.pool,
        query.arms,
        query.top_k,
    ).await.map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("Search error: {}", e)))?;

    drop(embedder);

    if query.rerank {
        if let Some(ref arms) = arm_results {
            let mut shown_ids = std::collections::HashSet::new();
            for r in &arms.bm25 { shown_ids.insert(r.chunk_id.clone()); }
            for r in &arms.fuzzy { shown_ids.insert(r.chunk_id.clone()); }
            for r in &arms.semantic { shown_ids.insert(r.chunk_id.clone()); }
            results.retain(|r| !shown_ids.contains(&r.chunk_id));
        }
        let mut reranker = state.cross_encoder.lock().await;
        results = reranker.rerank(&query.q, results)
            .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("Rerank error: {}", e)))?;
        results.truncate(query.top_k);
    }

    let mut rag_answer = None;
    if query.rag && !results.is_empty() {
        let provider = ms_rag::get_provider()
            .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("LLM provider error: {}", e)))?;
        let pipeline = ms_rag::RagPipeline::new(provider);
        let top_k_for_rag = std::cmp::min(results.len(), 5);
        let answer = pipeline.answer(&query.q, &results[..top_k_for_rag]).await
            .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("RAG error: {}", e)))?;
        rag_answer = Some(answer);
    }

    Ok(Json(SearchResponse {
        results,
        rag_answer,
        arm_results,
    }))
}
