use axum::{extract::State, Json};
use serde::Deserialize;
use std::path::PathBuf;
use uuid::Uuid;
use walkdir::WalkDir;
use tantivy::doc;
use pgvector::Vector;
use ms_ingest::chunker::sentence::SentenceChunker;
use ms_index::bm25::build_schema;
use crate::state::AppState;

#[derive(Deserialize)]
pub struct IngestRequest {
    pub file_path: String,
    pub mode: String,
}

pub async fn ingest_handler(
    State(state): State<AppState>,
    Json(payload): Json<IngestRequest>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    let path = PathBuf::from(&payload.file_path);
    if !path.exists() {
        return Err((axum::http::StatusCode::BAD_REQUEST, "File does not exist".to_string()));
    }

    let index_path = std::env::var("TANTIVY_PATH").unwrap_or_else(|_| "./data/tantivy-index".to_string());
    std::fs::create_dir_all(&index_path).map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("Tantivy path error: {}", e)))?;
    
    let schema = build_schema();
    let index = if std::path::Path::new(&index_path).join("meta.json").exists() {
        tantivy::Index::open_in_dir(&index_path).map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("Tantivy error: {}", e)))?
    } else {
        tantivy::Index::create_in_dir(&index_path, schema.clone()).map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("Tantivy error: {}", e)))?
    };

    let mut index_writer = index.writer(50_000_000).map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("Writer error: {}", e)))?;
    let chunker = SentenceChunker { chunk_size: 500, overlap: 50 };

    let id_field     = schema.get_field("id").unwrap();
    let text_field   = schema.get_field("text").unwrap();
    let mode_field   = schema.get_field("mode").unwrap();
    let source_field = schema.get_field("source").unwrap();
    let ci_field     = schema.get_field("chunk_index").unwrap();

    let mut total_docs = 0usize;
    let mut total_chunks = 0usize;

    let mut embedder = state.embedder.lock().await;

    for entry in WalkDir::new(&path).into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() { continue; }
        let Ok(content) = std::fs::read_to_string(entry.path()) else { continue };
        if content.trim().is_empty() { continue; }

        let doc_id = Uuid::new_v4();
        let source_path = entry.path().display().to_string();

        sqlx::query(
            "INSERT INTO documents (id, source_path, mode) VALUES ($1, $2, $3)",
        )
        .bind(doc_id)
        .bind(&source_path)
        .bind(&payload.mode)
        .execute(&state.pool)
        .await
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("DB error: {}", e)))?;

        let chunks = chunker.chunk(&content);
        if chunks.is_empty() { continue; }

        let embeddings = embedder.embed_batch(&chunks).map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("Embed error: {}", e)))?;

        for (i, (chunk_text, embedding)) in
            chunks.into_iter().zip(embeddings.into_iter()).enumerate()
        {
            let chunk_id = Uuid::new_v4();

            // 1. Tantivy BM25
            index_writer.add_document(doc!(
                id_field     => chunk_id.to_string(),
                text_field   => chunk_text.clone(),
                mode_field   => payload.mode.clone(),
                source_field => source_path.clone(),
                ci_field     => i as u64
            )).map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("Tantivy add_doc error: {}", e)))?;

            // 2. pgvector
            sqlx::query(
                "INSERT INTO chunks (id, doc_id, text, embedding, chunk_index, mode) \
                 VALUES ($1, $2, $3, $4, $5, $6)",
            )
            .bind(chunk_id)
            .bind(doc_id)
            .bind(&chunk_text)
            .bind(Vector::from(embedding))
            .bind(i as i32)
            .bind(&payload.mode)
            .execute(&state.pool)
            .await
            .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("DB chunk error: {}", e)))?;

            total_chunks += 1;
        }
        total_docs += 1;
    }

    index_writer.commit().map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("Commit error: {}", e)))?;

    Ok(Json(serde_json::json!({
        "status": "success",
        "message": format!("Ingested {} documents, {} chunks", total_docs, total_chunks)
    })))
}
