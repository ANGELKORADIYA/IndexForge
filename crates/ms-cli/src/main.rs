use clap::{Parser, Subcommand};
use ms_ingest::chunker::sentence::SentenceChunker;
use ms_ingest::embedder::Embedder;
use ms_index::bm25::build_schema;
use std::path::PathBuf;
use walkdir::WalkDir;
use tantivy::doc;
use uuid::Uuid;
use sqlx::postgres::PgPoolOptions;
use pgvector::Vector;
use dotenvy::dotenv;

#[derive(Parser)]
#[command(name = "ms-cli")]
#[command(about = "Memory Search CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Index {
        path: PathBuf,
        #[arg(short, long, default_value = "default")]
        mode: String,
    },
    Search {
        query: String,
        #[arg(short, long, default_value = "default")]
        mode: String,
        #[arg(short, long, default_value_t = 10)]
        top_k: usize,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv().ok();
    let cli = Cli::parse();
    let index_path = "./data/tantivy-index";
    std::fs::create_dir_all(index_path)?;

    let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&database_url)
        .await?;

    let mut embedder = Embedder::new(None)?;

    match &cli.command {
        Commands::Index { path, mode } => {
            let schema = build_schema();
            let index = if std::path::Path::new(index_path).join("meta.json").exists() {
                tantivy::Index::open_in_dir(index_path)?
            } else {
                tantivy::Index::create_in_dir(index_path, schema.clone())?
            };

            let mut index_writer = index.writer(50_000_000)?;
            let chunker = SentenceChunker {
                chunk_size: 500,
                overlap: 50,
            };

            let id_field = schema.get_field("id").unwrap();
            let text_field = schema.get_field("text").unwrap();
            let mode_field = schema.get_field("mode").unwrap();
            let source_field = schema.get_field("source").unwrap();
            let chunk_index_field = schema.get_field("chunk_index").unwrap();

            for entry in WalkDir::new(path).into_iter().filter_map(|e| e.ok()) {
                if entry.file_type().is_file() {
                    if let Ok(content) = std::fs::read_to_string(entry.path()) {
                        let doc_id = Uuid::new_v4();
                        let source_path = entry.path().display().to_string();

                        sqlx::query(
                            "INSERT INTO documents (id, source_path, mode) VALUES ($1, $2, $3)",
                        )
                        .bind(doc_id)
                        .bind(&source_path)
                        .bind(mode)
                        .execute(&pool)
                        .await?;

                        let chunks = chunker.chunk(&content);
                        if chunks.is_empty() {
                            continue;
                        }

                        let embeddings = embedder.embed_batch(&chunks)?;

                        for (i, (chunk_text, embedding)) in chunks.into_iter().zip(embeddings.into_iter()).enumerate() {
                            let chunk_id = Uuid::new_v4();
                            
                            // 1. Tantivy index
                            index_writer.add_document(doc!(
                                id_field => chunk_id.to_string(),
                                text_field => chunk_text.clone(),
                                mode_field => mode.clone(),
                                source_field => source_path.clone(),
                                chunk_index_field => i as u64
                            ))?;

                            // 2. PostgreSQL + pgvector
                            sqlx::query(
                                "INSERT INTO chunks (id, doc_id, text, embedding, chunk_index, mode) VALUES ($1, $2, $3, $4, $5, $6)",
                            )
                            .bind(chunk_id)
                            .bind(doc_id)
                            .bind(&chunk_text)
                            .bind(Vector::from(embedding))
                            .bind(i as i32)
                            .bind(mode)
                            .execute(&pool)
                            .await?;
                        }
                    }
                }
            }
            index_writer.commit()?;
            println!("Indexing complete.");
        }
        Commands::Search { query, mode, top_k } => {
            let schema = build_schema();
            let index = tantivy::Index::open_in_dir(index_path)?;
            let reader = index.reader()?;
            let bm25_index = ms_index::bm25::BM25Index {
                index,
                schema,
                reader,
            };

            let results = ms_search::search_bm25(query, mode, *top_k, &bm25_index).await;
            if results.is_empty() {
                println!("No results found.");
            } else {
                for (i, res) in results.iter().enumerate() {
                    println!("{}. [score: {:.4}] {}", i + 1, res.score, res.text);
                }
            }
        }
    }
    Ok(())
}
