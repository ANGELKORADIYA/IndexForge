use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use std::path::PathBuf;
use tokio::runtime::Runtime;
use walkdir::WalkDir;
use uuid::Uuid;
use pgvector::Vector;
use tantivy::doc;
use sqlx::postgres::PgPoolOptions;

use ms_ingest::chunker::sentence::SentenceChunker;
use ms_ingest::embedder::Embedder;
use ms_index::bm25::build_schema;
use ms_search::router;
use ms_rerank::CrossEncoder;

#[pyclass]
pub struct MemorySearch {
    rt: Runtime,
    db_url: String,
    tantivy_path: String,
}

#[pymethods]
impl MemorySearch {
    #[new]
    #[pyo3(signature = (db_url=None, tantivy_path=None))]
    fn new(db_url: Option<String>, tantivy_path: Option<String>) -> PyResult<Self> {
        let db = db_url.unwrap_or_else(|| {
            std::env::var("DATABASE_URL")
                .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5432/memory_search".to_string())
        });
        let tp = tantivy_path.unwrap_or_else(|| {
            std::env::var("TANTIVY_PATH").unwrap_or_else(|_| "./data/tantivy-index".to_string())
        });

        let rt = Runtime::new().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(MemorySearch {
            rt,
            db_url: db,
            tantivy_path: tp,
        })
    }

    fn index(&self, path: String, mode: String) -> PyResult<String> {
        self.rt.block_on(async {
            let pool = PgPoolOptions::new().connect(&self.db_url).await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            
            std::fs::create_dir_all(&self.tantivy_path).unwrap();
            let schema = build_schema();
            let index = if std::path::Path::new(&self.tantivy_path).join("meta.json").exists() {
                tantivy::Index::open_in_dir(&self.tantivy_path).unwrap()
            } else {
                tantivy::Index::create_in_dir(&self.tantivy_path, schema.clone()).unwrap()
            };
            let mut index_writer = index.writer(50_000_000).unwrap();
            
            let chunker = SentenceChunker { chunk_size: 500, overlap: 50 };
            let mut embedder = Embedder::new(None).unwrap();

            let id_field     = schema.get_field("id").unwrap();
            let text_field   = schema.get_field("text").unwrap();
            let mode_field   = schema.get_field("mode").unwrap();
            let source_field = schema.get_field("source").unwrap();
            let ci_field     = schema.get_field("chunk_index").unwrap();

            let mut total_chunks = 0;

            for entry in WalkDir::new(&path).into_iter().filter_map(|e| e.ok()) {
                if !entry.file_type().is_file() { continue; }
                let extracted_docs = match ms_ingest::loader::load_file(entry.path()) {
                    Ok(docs) => docs,
                    Err(_) => continue,
                };

                for extracted in extracted_docs {
                    let doc_id = Uuid::new_v4();
                    sqlx::query("INSERT INTO documents (id, source_path, mode) VALUES ($1, $2, $3)")
                        .bind(doc_id).bind(&extracted.source).bind(&mode).execute(&pool).await.unwrap();

                    let chunks = chunker.chunk(&extracted.text);
                    if chunks.is_empty() { continue; }
                    let embeddings = embedder.embed_batch(&chunks).unwrap();

                    for (i, (chunk_text, embedding)) in chunks.into_iter().zip(embeddings.into_iter()).enumerate() {
                        let chunk_id = Uuid::new_v4();
                        index_writer.add_document(doc!(id_field=>chunk_id.to_string(), text_field=>chunk_text.clone(), mode_field=>mode.clone(), source_field=>extracted.source.clone(), ci_field=>i as u64)).unwrap();
                        sqlx::query("INSERT INTO chunks (id, doc_id, text, embedding, chunk_index, mode) VALUES ($1, $2, $3, $4, $5, $6)")
                            .bind(chunk_id).bind(doc_id).bind(&chunk_text).bind(Vector::from(embedding)).bind(i as i32).bind(&mode).execute(&pool).await.unwrap();
                        total_chunks += 1;
                    }
                }
            }
            index_writer.commit().unwrap();
            Ok(format!("Successfully indexed {} chunks into mode '{}'", total_chunks, mode))
        })
    }

    #[pyo3(signature = (query, mode="default".to_string(), top_k=10, rerank=false, rag=false))]
    fn search(
        &self,
        query: String,
        mode: String,
        top_k: usize,
        rerank: bool,
        rag: bool,
    ) -> PyResult<String> {
        self.rt.block_on(async {
            let pool = PgPoolOptions::new().connect(&self.db_url).await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let mut embedder = Embedder::new(None).unwrap();
            let mut cross_encoder = if rerank { Some(CrossEncoder::new().unwrap()) } else { None };

            let schema = build_schema();
            let index = tantivy::Index::open_in_dir(&self.tantivy_path).unwrap();
            let reader = index.reader().unwrap();
            
            let mut results = router::route_query(
                &query, &mode, &pool, &reader, &schema, &mut embedder
            ).await.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            if let Some(ce) = &mut cross_encoder {
                results = ms_rerank::rerank(&query, results, ce).unwrap();
            }

            results.truncate(top_k);
            let json_results = serde_json::to_string_pretty(&results).unwrap();
            
            Ok(json_results)
        })
    }
}

#[pymodule]
fn memory_search(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<MemorySearch>()?;
    Ok(())
}
