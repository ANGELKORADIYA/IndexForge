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
        
        let extracted_docs = match ms_ingest::loader::load_file(entry.path()) {
            Ok(docs) => docs,
            Err(_) => continue,
        };

        for extracted in extracted_docs {
            let doc_id = Uuid::new_v4();
            
            sqlx::query(
                "INSERT INTO documents (id, source_path, mode) VALUES ($1, $2, $3)",
            )
            .bind(doc_id)
            .bind(&extracted.source)
            .bind(&payload.mode)
            .execute(&state.pool)
            .await
            .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("DB error: {}", e)))?;

            let chunks = chunker.chunk(&extracted.text);
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
                    source_field => extracted.source.clone(),
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
    }

    index_writer.commit().map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("Commit error: {}", e)))?;

    Ok(Json(serde_json::json!({
        "status": "success",
        "message": format!("Ingested {} documents, {} chunks", total_docs, total_chunks)
    })))
}

use axum::extract::Multipart;
use tokio::io::AsyncWriteExt;

pub async fn upload_handler(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    let upload_dir = PathBuf::from("./data/uploads");
    std::fs::create_dir_all(&upload_dir)
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let mut saved_paths = Vec::new();
    let mut mode = "default".to_string();

    while let Some(field) = multipart.next_field().await.unwrap_or(None) {
        let name = field.name().unwrap_or("").to_string();
        
        if name == "mode" {
            if let Ok(text) = field.text().await {
                mode = text;
            }
            continue;
        }

        if name == "file" {
            let file_name = field.file_name().unwrap_or("unnamed").to_string();
            let dest_path = upload_dir.join(&file_name);
            
            let data = field.bytes().await.map_err(|e| (axum::http::StatusCode::BAD_REQUEST, e.to_string()))?;
            let mut file = tokio::fs::File::create(&dest_path).await.map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
            file.write_all(&data).await.map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
            
            saved_paths.push(dest_path);
        }
    }

    if saved_paths.is_empty() {
        return Err((axum::http::StatusCode::BAD_REQUEST, "No files uploaded".to_string()));
    }

    // Process the uploaded files the same way as ingest_handler.
    // For simplicity, we just trigger the same logic but for each file individually or point ingest_handler to the uploads dir.
    // Let's just reuse the exact same ingest loop by calling a helper or copying the inner loop logic.
    let index_path = std::env::var("TANTIVY_PATH").unwrap_or_else(|_| "./data/tantivy-index".to_string());
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

    for path in saved_paths {
        let extracted_docs = match ms_ingest::loader::load_file(&path) {
            Ok(docs) => docs,
            Err(_) => continue,
        };

        for extracted in extracted_docs {
            let doc_id = Uuid::new_v4();
            sqlx::query("INSERT INTO documents (id, source_path, mode) VALUES ($1, $2, $3)")
                .bind(doc_id).bind(&extracted.source).bind(&mode).execute(&state.pool).await.unwrap();

            let chunks = chunker.chunk(&extracted.text);
            if chunks.is_empty() { continue; }
            let embeddings = embedder.embed_batch(&chunks).unwrap();

            for (i, (chunk_text, embedding)) in chunks.into_iter().zip(embeddings.into_iter()).enumerate() {
                let chunk_id = Uuid::new_v4();
                index_writer.add_document(doc!(id_field=>chunk_id.to_string(), text_field=>chunk_text.clone(), mode_field=>mode.clone(), source_field=>extracted.source.clone(), ci_field=>i as u64)).unwrap();
                sqlx::query("INSERT INTO chunks (id, doc_id, text, embedding, chunk_index, mode) VALUES ($1, $2, $3, $4, $5, $6)")
                    .bind(chunk_id).bind(doc_id).bind(&chunk_text).bind(Vector::from(embedding)).bind(i as i32).bind(&mode).execute(&state.pool).await.unwrap();
                total_chunks += 1;
            }
            total_docs += 1;
        }
        
        // Optional: remove file after parsing
        let _ = std::fs::remove_file(path);
    }

    index_writer.commit().map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("Commit error: {}", e)))?;

    Ok(Json(serde_json::json!({
        "status": "success",
        "message": format!("Uploaded and indexed {} documents, {} chunks", total_docs, total_chunks)
    })))
}
