use super::ExtractedDoc;
use std::path::Path;
use zim::Zim;

pub fn load(path: &Path) -> anyhow::Result<Vec<ExtractedDoc>> {
    // Zim parsing can be heavy; we will extract the text from the html inside the articles.
    // Note: To avoid running out of memory, a real production system might stream this
    // directly into the index. For simplicity, we'll collect up to a certain limit or 
    // yield them. Wait, if a ZIM file is 4GB, returning a Vec<ExtractedDoc> will OOM!
    // Since our `load_file` signature returns `Vec<ExtractedDoc>`, let's implement a safeguard limit.
    // A better approach would be to refactor `load_file` to return an Iterator, but for now:
    
    let zim_file = Zim::new(path)?;
    let mut docs = Vec::new();

    // Iterate through all articles.
    for article in zim_file.iterate_by_urls() {
        if docs.len() >= 1000 {
            // Safety break for now: process first 1000 articles so we don't OOM
            break;
        }

        // We only care about content (usually text/html)
        if article.mime_type.as_deref() != Some("text/html") {
            continue;
        }

        let content = article.get_content(&zim_file).unwrap_or_default();
        let html_string = String::from_utf8_lossy(&content).to_string();
        
        // Strip HTML (extremely rudimentary for speed and simplicity)
        // In a real system, we might use a proper HTML parser like tl or scraper.
        let text = strip_html(&html_string);

        if !text.trim().is_empty() {
            docs.push(ExtractedDoc {
                source: format!("{}#{}", path.display(), article.url),
                text,
            });
        }
    }

    Ok(docs)
}

fn strip_html(html: &str) -> String {
    let mut result = String::with_capacity(html.len());
    let mut in_tag = false;

    for c in html.chars() {
        match c {
            '<' => in_tag = true,
            '>' => {
                in_tag = false;
                result.push(' ');
            }
            _ if !in_tag => result.push(c),
            _ => {}
        }
    }

    // Compress whitespace
    result.split_whitespace().collect::<Vec<_>>().join(" ")
}
