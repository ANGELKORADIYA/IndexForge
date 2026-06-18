use super::ExtractedDoc;
use std::path::Path;
use zim::Zim;

pub fn load(path: &Path) -> anyhow::Result<Vec<ExtractedDoc>> {
    let zim_file = Zim::new(path).map_err(|e| anyhow::anyhow!("ZIM Error: {:?}", e))?;
    let mut docs = Vec::new();

    // Iterate through all articles.
    for article in zim_file.iterate_by_urls() {
        if docs.len() >= 1000 {
            // Safety break for now: process first 1000 articles so we don't OOM
            break;
        }

        // Check if MIME type is text/html
        match article.mime_type {
            zim::MimeType::Type(ref t) if t == "text/html" => {},
            _ => continue,
        }

        let content = match article.target {
            Some(zim::Target::Cluster(cluster_idx, blob_idx)) => {
                let cluster = zim_file.get_cluster(cluster_idx).map_err(|e| anyhow::anyhow!("{:?}", e))?;
                let blob = cluster.get_blob(blob_idx).map_err(|e| anyhow::anyhow!("{:?}", e))?;
                blob.to_vec()
            }
            _ => continue,
        };

        let html_string = String::from_utf8_lossy(&content).to_string();
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
