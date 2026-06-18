use super::ExtractedDoc;
use std::path::Path;
use serde_json::Value;

pub fn load_json(path: &Path) -> anyhow::Result<Vec<ExtractedDoc>> {
    let content = std::fs::read_to_string(path)?;
    let parsed: Value = serde_json::from_str(&content)?;

    let mut docs = Vec::new();

    // If it's an array, treat each object as a separate document
    if let Some(arr) = parsed.as_array() {
        for (i, item) in arr.iter().enumerate() {
            let text = serde_json::to_string_pretty(item).unwrap_or_default();
            docs.push(ExtractedDoc {
                source: format!("{}#[{}]", path.display(), i),
                text,
            });
        }
    } else {
        // If it's an object, dump the whole thing
        let text = serde_json::to_string_pretty(&parsed).unwrap_or_default();
        docs.push(ExtractedDoc {
            source: path.display().to_string(),
            text,
        });
    }

    Ok(docs)
}

pub fn load_csv(path: &Path) -> anyhow::Result<Vec<ExtractedDoc>> {
    let mut reader = csv::Reader::from_path(path)?;
    let headers = reader.headers()?.clone();

    let mut docs = Vec::new();
    for (i, result) in reader.records().enumerate() {
        if let Ok(record) = result {
            let mut text_parts = Vec::new();
            for (header, field) in headers.iter().zip(record.iter()) {
                text_parts.push(format!("{}: {}", header, field));
            }
            docs.push(ExtractedDoc {
                source: format!("{}#row={}", path.display(), i + 1),
                text: text_parts.join("\n"),
            });
        }
    }

    Ok(docs)
}
