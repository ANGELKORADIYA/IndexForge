use super::ExtractedDoc;
use std::path::Path;

pub fn load(path: &Path) -> anyhow::Result<Vec<ExtractedDoc>> {
    // pdf_extract::extract_text takes a path, not bytes.
    let text = pdf_extract::extract_text(path)
        .unwrap_or_else(|_| String::new());

    if text.trim().is_empty() {
        return Ok(vec![]);
    }

    Ok(vec![ExtractedDoc {
        source: path.display().to_string(),
        text,
    }])
}
