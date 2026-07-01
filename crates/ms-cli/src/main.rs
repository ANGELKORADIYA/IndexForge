use clap::{Parser, Subcommand};
use ms_ingest::chunker::sentence::SentenceChunker;
use ms_ingest::embedder::Embedder;
use ms_index::bm25::build_schema;
use ms_index::fuzzy::FuzzyIndex;
use std::path::PathBuf;
use walkdir::WalkDir;
use tantivy::doc;
use uuid::Uuid;
use sqlx::postgres::PgPoolOptions;
use pgvector::Vector;
use dotenvy::dotenv;

fn clean_preview(text: &str, _max_chars: usize) -> String {
    let mut cleaned = String::with_capacity(text.len());
    let mut last_space = false;
    for c in text.chars() {
        if c.is_control() || c.is_whitespace() {
            if !last_space {
                cleaned.push(' ');
                last_space = true;
            }
        } else {
            cleaned.push(c);
            last_space = false;
        }
    }
    cleaned.trim().to_string()
}

#[derive(Parser)]
#[command(name = "ms-cli")]
#[command(about = "MemorySearch CLI — index and search your local data", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Ingest a directory into all indexes (Tantivy BM25 + pgvector + fuzzy)
    Index {
        path: PathBuf,
        #[arg(short, long, default_value = "default")]
        mode: String,
    },
    /// Search using BM25 only (fast, no DB needed)
    Search {
        query: String,
        #[arg(short, long, default_value = "default")]
        mode: String,
        #[arg(short, long, default_value_t = 10)]
        top_k: usize,
        #[arg(long, default_missing_value = "all", num_args = 0..=1, alias = "show-arm-results")]
        arms: Option<String>,
    },
    /// Search using all 3 arms in parallel (BM25 + Fuzzy + Semantic) with RRF merge
    SearchAll {
        query: String,
        #[arg(short, long, default_value = "default")]
        mode: String,
        #[arg(short, long, default_value_t = 10)]
        top_k: usize,
        #[arg(long, default_value_t = false)]
        rerank: bool,
        #[arg(long, default_value_t = false)]
        rag: bool,
        #[arg(long, default_missing_value = "all", num_args = 0..=1, alias = "show-arm-results")]
        arms: Option<String>,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv().ok();
    let cli = Cli::parse();
    let index_path = std::env::var("TANTIVY_PATH").unwrap_or_else(|_| "./data/tantivy-index".to_string());
    std::fs::create_dir_all(&index_path)?;

    let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&database_url)
        .await?;

    let mut embedder = Embedder::new(None)?;

    match &cli.command {
        Commands::Index { path, mode } => {
            let schema = build_schema();
            let index = if std::path::Path::new(&index_path).join("meta.json").exists() {
                tantivy::Index::open_in_dir(&index_path)?
            } else {
                tantivy::Index::create_in_dir(&index_path, schema.clone())?
            };

            let mut index_writer = index.writer(50_000_000)?;
            let chunker = SentenceChunker { chunk_size: 500, overlap: 50 };

            let id_field     = schema.get_field("id").unwrap();
            let text_field   = schema.get_field("text").unwrap();
            let mode_field   = schema.get_field("mode").unwrap();
            let source_field = schema.get_field("source").unwrap();
            let ci_field     = schema.get_field("chunk_index").unwrap();

            let mut total_docs = 0usize;
            let mut total_chunks = 0usize;

            for entry in WalkDir::new(path).into_iter().filter_map(|e| e.ok()) {
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
                    .bind(mode)
                    .execute(&pool)
                    .await?;

                    let chunks = chunker.chunk(&extracted.text);
                    if chunks.is_empty() { continue; }

                    let embeddings = embedder.embed_batch(&chunks)?;

                    for (i, (chunk_text, embedding)) in
                        chunks.into_iter().zip(embeddings.into_iter()).enumerate()
                    {
                        let chunk_id = Uuid::new_v4();

                        // 1. Tantivy BM25
                        index_writer.add_document(doc!(
                            id_field     => chunk_id.to_string(),
                            text_field   => chunk_text.clone(),
                            mode_field   => mode.clone(),
                            source_field => extracted.source.clone(),
                            ci_field     => i as u64
                        ))?;

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
                        .bind(mode)
                        .execute(&pool)
                        .await?;

                        total_chunks += 1;
                    }
                    total_docs += 1;
                }
            }

            index_writer.commit()?;
            println!("✅ Indexed {total_docs} documents, {total_chunks} chunks into mode='{mode}'.");
        }

        Commands::Search { query, mode, top_k, arms } => {
            let schema = build_schema();
            let index  = tantivy::Index::open_in_dir(&index_path)?;
            let reader = index.reader()?;
            let bm25   = ms_index::bm25::BM25Index { index, schema, reader };

            let results = ms_search::search_bm25(query, mode, *top_k, &bm25).await;
            if results.is_empty() {
                println!("No results found.");
            } else {
                if arms.is_some() {
                    println!("\n\x1b[1;36m┌─────────────────────────────────────────────────────────────────────────────┐\x1b[0m");
                    println!("\x1b[1;36m│ 📝 ARM 1: BM25 (Exact Keyword Match) — Top {} Results                        │\x1b[0m", results.len());
                    println!("\x1b[1;36m└─────────────────────────────────────────────────────────────────────────────┘\x1b[0m");
                    for (i, res) in results.iter().enumerate() {
                        let preview = clean_preview(&res.text, 140);
                        println!("  \x1b[1;36m#{:02}\x1b[0m │ ID: \x1b[1m{}\x1b[0m │ Score: \x1b[1;36m{:.4}\x1b[0m", i + 1, &res.chunk_id[..8], res.score);
                        println!("      └─ \"{}\"\n", preview);
                    }
                } else {
                    println!("=== BM25 Results for \"{}\" (mode: {}) ===", query, mode);
                    for (i, res) in results.iter().enumerate() {
                        let preview = clean_preview(&res.text, 140);
                        println!("  #{:02} │ ID: {} │ Score: {:.4}", i + 1, &res.chunk_id[..8], res.score);
                        println!("      └─ \"{}\"\n", preview);
                    }
                }
            }
        }

        Commands::SearchAll { query, mode, top_k, rerank, rag, arms } => {
            // Build BM25 index
            let schema = build_schema();
            let index  = tantivy::Index::open_in_dir(&index_path)?;
            let reader = index.reader()?;
            let bm25   = ms_index::bm25::BM25Index { index, schema, reader };

            // Build fuzzy index from DB chunks
            println!("⚡ Loading fuzzy index from database...");
            let mut fuzzy = FuzzyIndex::new();
            let rows = sqlx::query!("SELECT id::text, text FROM chunks WHERE mode = $1", mode)
                .fetch_all(&pool)
                .await?;
            for row in rows {
                fuzzy.add(row.id.unwrap_or_default(), row.text);
            }

            // Run all 3 arms
            let show_arms = arms.is_some();
            let arm_filter = arms.as_deref().unwrap_or("all").to_lowercase();
            let show_all = arm_filter == "all" || arm_filter == "true" || arm_filter == "yes" || arm_filter.is_empty();

            let k_search = if *rerank && show_arms { (*top_k) * 5 } else if *rerank { (*top_k) * 2 } else { *top_k };
            println!("🔍 Running 3-arm search for \"{}\" (mode: {})...", query, mode);
            let (mut results, arm_results) = ms_search::router::search_with_arms(
                query, mode, k_search,
                &bm25, &fuzzy, &mut embedder, &pool, show_arms, *top_k,
            ).await?;

            if let Some(ref arms_data) = arm_results {
                if show_all || arm_filter == "bm25" {
                    println!("\n\x1b[1;36m┌─────────────────────────────────────────────────────────────────────────────┐\x1b[0m");
                    println!("\x1b[1;36m│ 📝 ARM 1: BM25 (Exact Keyword Match) — Top {} Results                        │\x1b[0m", arms_data.bm25.len());
                    println!("\x1b[1;36m└─────────────────────────────────────────────────────────────────────────────┘\x1b[0m");
                    for (i, res) in arms_data.bm25.iter().enumerate() {
                        let preview = clean_preview(&res.text, 140);
                        println!("  \x1b[1;36m#{:02}\x1b[0m │ ID: \x1b[1m{}\x1b[0m │ Score: \x1b[1;36m{:.4}\x1b[0m", i + 1, &res.chunk_id[..8], res.score);
                        println!("      └─ \"{}\"\n", preview);
                    }
                }

                if show_all || arm_filter == "fuzzy" {
                    println!("\n\x1b[1;33m┌─────────────────────────────────────────────────────────────────────────────┐\x1b[0m");
                    println!("\x1b[1;33m│ 🔤 ARM 2: Fuzzy (Trigram Typo-Tolerant) — Top {} Results                     │\x1b[0m", arms_data.fuzzy.len());
                    println!("\x1b[1;33m└─────────────────────────────────────────────────────────────────────────────┘\x1b[0m");
                    for (i, res) in arms_data.fuzzy.iter().enumerate() {
                        let preview = clean_preview(&res.text, 140);
                        println!("  \x1b[1;33m#{:02}\x1b[0m │ ID: \x1b[1m{}\x1b[0m │ Score: \x1b[1;33m{:.4}\x1b[0m", i + 1, &res.chunk_id[..8], res.score);
                        println!("      └─ \"{}\"\n", preview);
                    }
                }

                if show_all || arm_filter == "semantic" {
                    println!("\n\x1b[1;32m┌─────────────────────────────────────────────────────────────────────────────┐\x1b[0m");
                    println!("\x1b[1;32m│ 🧠 ARM 3: Semantic (Vector Embedding) — Top {} Results                      │\x1b[0m", arms_data.semantic.len());
                    println!("\x1b[1;32m└─────────────────────────────────────────────────────────────────────────────┘\x1b[0m");
                    for (i, res) in arms_data.semantic.iter().enumerate() {
                        let preview = clean_preview(&res.text, 140);
                        println!("  \x1b[1;32m#{:02}\x1b[0m │ ID: \x1b[1m{}\x1b[0m │ Score: \x1b[1;32m{:.4}\x1b[0m", i + 1, &res.chunk_id[..8], res.score);
                        println!("      └─ \"{}\"\n", preview);
                    }
                }
            }

            if *rerank {
                if let Some(ref arms_data) = arm_results {
                    let mut shown_ids = std::collections::HashSet::new();
                    for r in &arms_data.bm25 { shown_ids.insert(r.chunk_id.clone()); }
                    for r in &arms_data.fuzzy { shown_ids.insert(r.chunk_id.clone()); }
                    for r in &arms_data.semantic { shown_ids.insert(r.chunk_id.clone()); }
                    results.retain(|r| !shown_ids.contains(&r.chunk_id));
                }
                println!("🧠 Re-ranking top {} unique remaining candidates with Cross-Encoder...", results.len());
                let mut cross_encoder = ms_rerank::CrossEncoder::new()?;
                results = cross_encoder.rerank(query, results)?;
                results.truncate(*top_k);
            }

            if results.is_empty() {
                println!("No results found.");
            } else {
                let title = if *rerank && show_arms {
                    "🏆 FINAL RESULTS: Cross-Encoder Re-ranked (Unique candidates not overlapping)"
                } else if *rerank {
                    "🏆 FINAL RESULTS: Cross-Encoder Re-ranked"
                } else {
                    "🏆 FINAL RESULTS: BM25 + Fuzzy + Semantic (RRF Merged)"
                };
                println!("\x1b[1;35m┌───────────────────────────────────────────────────────────────────────────────────────────┐\x1b[0m");
                println!("\x1b[1;35m│ {}\x1b[0m", title);
                println!("\x1b[1;35m└───────────────────────────────────────────────────────────────────────────────────────────┘\x1b[0m");
                for (i, res) in results.iter().enumerate() {
                    let arm_info: Vec<String> = res.arm_scores
                        .iter()
                        .map(|(arm, s)| format!("{:?}={:.3}", arm, s))
                        .collect();
                    let preview = clean_preview(&res.text, 160);
                    println!(
                        "  \x1b[1;35m#{:02}\x1b[0m │ ID: \x1b[1m{}\x1b[0m │ Score: \x1b[1;35m{:.4}\x1b[0m │ Arms: {}",
                        i + 1, &res.chunk_id[..8], res.score, arm_info.join(", ")
                    );
                    println!("      └─ \"{}\"\n", preview);
                }

                if *rag {
                    println!("\n🤖 Generating answer with RAG...");
                    let provider = ms_rag::get_provider()?;
                    let pipeline = ms_rag::RagPipeline::new(provider);
                    let top_k_for_rag = std::cmp::min(results.len(), 5);
                    let answer = pipeline.answer(query, &results[..top_k_for_rag]).await?;
                    println!("\n=== RAG Answer (Model: {}) ===\n{}", answer.model, answer.answer);
                }
            }
        }
    }
    Ok(())
}
