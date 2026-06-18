/// Pure-Rust fuzzy index using character trigrams + Jaccard similarity.
/// No external dependency needed — works 100% offline.
///
/// Query typo tolerance is handled by character n-gram overlap, which naturally
/// tolerates single-character substitutions/transpositions without a dictionary.

pub struct FuzzyResult {
    pub chunk_id: String,
    pub text: String,
    pub score: f64, // Jaccard trigram similarity [0.0, 1.0]
}

/// In-memory fuzzy index.
pub struct FuzzyIndex {
    /// (chunk_id, text, trigram_set)
    entries: Vec<(String, String, Vec<String>)>,
}

impl FuzzyIndex {
    pub fn new() -> Self {
        FuzzyIndex { entries: Vec::new() }
    }

    /// Add a chunk to the in-memory fuzzy index.
    pub fn add(&mut self, chunk_id: String, text: String) {
        let tg = trigrams(&text.to_lowercase());
        self.entries.push((chunk_id, text, tg));
    }

    /// Search: rank chunks by trigram Jaccard similarity against the query.
    ///
    /// The `mode_filter` parameter is accepted for API symmetry with other arms
    /// but the fuzzy index is not mode-partitioned (mode filtering happens at
    /// load time in ms-cli when building the index).
    pub fn search(&self, query: &str, _mode_filter: &str, top_k: usize) -> Vec<FuzzyResult> {
        let query_trigrams = trigrams(&query.to_lowercase());
        if query_trigrams.is_empty() {
            return vec![];
        }
        let query_set: std::collections::HashSet<&str> =
            query_trigrams.iter().map(|s| s.as_str()).collect();

        let mut scored: Vec<(&str, &str, f64)> = self
            .entries
            .iter()
            .map(|(id, text, doc_tg)| {
                let doc_set: std::collections::HashSet<&str> =
                    doc_tg.iter().map(|s| s.as_str()).collect();
                let intersection = query_set.intersection(&doc_set).count();
                let union = query_set.union(&doc_set).count();
                let jaccard = if union == 0 { 0.0 } else { intersection as f64 / union as f64 };
                (id.as_str(), text.as_str(), jaccard)
            })
            .collect();

        scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        scored
            .into_iter()
            .take(top_k)
            .filter(|(_, _, s)| *s > 0.0)
            .map(|(id, text, score)| FuzzyResult {
                chunk_id: id.to_string(),
                text: text.to_string(),
                score,
            })
            .collect()
    }
}

impl Default for FuzzyIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate character-level trigrams.
/// "hello" → ["hel", "ell", "llo"]
/// Pads short strings with spaces: "hi" → [" hi", "hi "]
fn trigrams(s: &str) -> Vec<String> {
    let padded = format!(" {} ", s);
    let chars: Vec<char> = padded.chars().collect();
    if chars.len() < 3 {
        return vec![padded];
    }
    chars.windows(3).map(|w| w.iter().collect()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trigrams_basic() {
        let t = trigrams("hello");
        assert!(t.contains(&" he".to_string()));
        assert!(t.contains(&"hel".to_string()));
        assert!(t.contains(&"llo".to_string()));
    }

    #[test]
    fn test_fuzzy_ranks_relevant_first() {
        let mut idx = FuzzyIndex::new();
        idx.add("c1".into(), "Rust ownership and borrowing rules".into());
        idx.add("c2".into(), "Python garbage collection memory management".into());

        let results = idx.search("rust ownersh", "", 5);
        assert!(!results.is_empty());
        assert_eq!(results[0].chunk_id, "c1");
    }

    #[test]
    fn test_fuzzy_typo_tolerance() {
        let mut idx = FuzzyIndex::new();
        idx.add("c1".into(), "machine learning algorithms".into());
        idx.add("c2".into(), "cooking recipes".into());

        // "machne learnng" (2 chars dropped) should still match c1 better
        let results = idx.search("machne learnng", "", 2);
        assert!(!results.is_empty());
        assert_eq!(results[0].chunk_id, "c1");
    }
}
