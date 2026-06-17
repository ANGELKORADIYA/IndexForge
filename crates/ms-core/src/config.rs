use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub tantivy_path: String,
    pub database_url: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            tantivy_path: "./data/tantivy-index".to_string(),
            database_url: "postgresql://localhost/memory_search".to_string(),
        }
    }
}
