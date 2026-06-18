use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub tantivy_path: String,
    pub database_url: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            tantivy_path: std::env::var("TANTIVY_PATH")
                .unwrap_or_else(|_| "./data/tantivy-index".to_string()),
            database_url: std::env::var("DATABASE_URL")
                .unwrap_or_else(|_| {
                    "postgres://postgres:postgres@127.0.0.1:5432/memory_search".to_string()
                }),
        }
    }
}
