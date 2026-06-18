use super::ExtractedDoc;
use std::path::Path;

pub fn load(path: &Path) -> anyhow::Result<Vec<ExtractedDoc>> {
    let bytes = std::fs::read(path)?;
    
    // pdf_extract::extract_text can extract plain text from PDFs.
    let text = pdf_extract::extract_text(&bytes)
        .unwrap_or_else(|_| String::new());

    if text.trim().is_empty() {
        return Ok(vec![]);
    }

    Ok(vec![ExtractedDoc {
        source: path.display().to_string(),
        text,
    }])
}
