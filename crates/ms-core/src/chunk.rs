use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: Uuid,
    pub text: String,
    pub embedding: Option<Vec<f32>>,
    pub metadata: ChunkMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    pub source_file: String,
    pub page: Option<u32>,
    pub line_start: Option<u32>,
    pub line_end: Option<u32>,
    pub chunk_index: u32,
    pub doc_id: Uuid,
    pub mode: DataMode,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DataMode {
    Notes,
    Codebase,
    Wikipedia,
    Default,
}

impl From<&str> for DataMode {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "notes" => DataMode::Notes,
            "codebase" => DataMode::Codebase,
            "wikipedia" => DataMode::Wikipedia,
            _ => DataMode::Default,
        }
    }
}
