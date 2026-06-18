pub mod text;
pub mod pdf;
pub mod docx;
pub mod zim;
pub mod structured;

use std::path::Path;

pub struct ExtractedDoc {
    pub source: String,
    pub text: String,
}

pub fn load_file(path: &Path) -> anyhow::Result<Vec<ExtractedDoc>> {
    let ext = path.extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();

    match ext.as_str() {
        "pdf" => pdf::load(path),
        "docx" => docx::load(path),
        "json" => structured::load_json(path),
        "csv" => structured::load_csv(path),
        "zim" => zim::load(path),
        _ => text::load(path), // Default to plain text
    }
}
