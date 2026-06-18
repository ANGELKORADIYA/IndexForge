use super::ExtractedDoc;
use std::path::Path;

pub fn load(path: &Path) -> anyhow::Result<Vec<ExtractedDoc>> {
    let content = std::fs::read_to_string(path)?;
    if content.trim().is_empty() {
        return Ok(vec![]);
    }

    Ok(vec![ExtractedDoc {
        source: path.display().to_string(),
        text: content,
    }])
}
