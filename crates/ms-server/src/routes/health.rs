use axum::{extract::State, Json};
use serde_json::json;
use crate::state::AppState;

pub async fn health_check() -> Json<serde_json::Value> {
    Json(json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION")
    }))
}

pub async fn stats_handler(
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    // We will query pgvector tables for counts.
    let doc_count: (i64,) = sqlx::query_as("SELECT count(*) FROM documents")
        .fetch_one(&state.pool)
        .await
        .unwrap_or((0,));

    let chunk_count: (i64,) = sqlx::query_as("SELECT count(*) FROM chunks")
        .fetch_one(&state.pool)
        .await
        .unwrap_or((0,));

    // For breakdown by mode:
    let mode_counts: Vec<(String, i64)> = sqlx::query_as("SELECT mode, count(*) FROM chunks GROUP BY mode")
        .fetch_all(&state.pool)
        .await
        .unwrap_or_default();

    let mut modes = serde_json::Map::new();
    for (mode, count) in mode_counts {
        modes.insert(mode, json!(count));
    }

    Json(json!({
        "total_documents": doc_count.0,
        "total_chunks": chunk_count.0,
        "modes": modes,
        "arms": {
            "bm25": "Tantivy Index (Syncs 1:1 with DB Chunks)",
            "semantic": "pgvector HNSW (Syncs 1:1 with DB Chunks)",
            "fuzzy": "DB Trigrams (Generated on-the-fly from Chunks)"
        }
    }))
}
