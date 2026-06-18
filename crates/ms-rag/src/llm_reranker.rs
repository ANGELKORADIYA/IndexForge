use crate::provider::LLMProvider;
use ms_core::score::SearchResult;

/// LLM-based re-ranker that uses a language model to reorder search
/// results by semantic relevance to a query.
///
/// The re-ranker sends a structured prompt to the LLM containing the
/// query and up to 20 text chunks, asks for a JSON array of indices
/// sorted by relevance, and reorders the results accordingly.
pub struct LlmReranker {
    provider: Box<dyn LLMProvider>,
}

impl LlmReranker {
    /// Creates a new `LlmReranker` backed by the given LLM provider.
    pub fn new(provider: Box<dyn LLMProvider>) -> Self {
        Self { provider }
    }

    /// Re-ranks `results` by asking the LLM to judge relevance to `query`.
    ///
    /// Returns a new vector of `SearchResult` sorted from most to least
    /// relevant, with freshly assigned scores (1.0 for the top result,
    /// decreasing linearly).
    pub async fn rerank(
        &self,
        query: &str,
        results: Vec<SearchResult>,
    ) -> anyhow::Result<Vec<SearchResult>> {
        if results.is_empty() {
            return Ok(Vec::new());
        }

        // Cap at 20 chunks to stay within reasonable prompt sizes.
        let capped: Vec<SearchResult> = results.into_iter().take(20).collect();
        let total = capped.len();

        // Build the prompt.
        let prompt = build_prompt(query, &capped);

        // Ask the LLM.
        let response = self.provider.generate_answer(&prompt).await?;

        // Parse the ranking from the response.
        let ranking = parse_ranking(&response, total);

        // Reorder results according to the LLM's ranking.
        let reordered = apply_ranking(capped, &ranking, total);

        Ok(reordered)
    }
}

/// Builds a structured prompt for the LLM re-ranking task.
fn build_prompt(query: &str, results: &[SearchResult]) -> String {
    let mut prompt = String::new();
    prompt.push_str(
        "You are a search relevance expert. Given a query and a list of text chunks, \
         rank the chunks by relevance to the query.\n\n",
    );
    prompt.push_str(&format!("Query: {}\n\n", query));
    prompt.push_str("Chunks:\n");

    for (i, result) in results.iter().enumerate() {
        // Truncate chunk text to 500 characters to keep the prompt concise.
        let truncated: String = result.text.chars().take(500).collect();
        prompt.push_str(&format!("[{}] {}\n", i, truncated));
    }

    prompt.push_str(
        "\nRespond with ONLY a JSON array of chunk indices sorted from most relevant \
         to least relevant. Example: [2, 0, 1]\n\
         Do not include any other text, explanation, or formatting.\n",
    );

    prompt
}

/// Attempts to parse a JSON array of integer indices from the LLM response.
///
/// Handles common wrapping patterns (e.g. markdown code blocks) and
/// gracefully falls back to original ordering on parse failure.
fn parse_ranking(response: &str, total: usize) -> Vec<usize> {
    // Try to extract a JSON array from the response.
    // The LLM might wrap it in markdown code blocks like ```json ... ```.
    let cleaned = extract_json_array(response);

    // Parse as a JSON array of numbers.
    let parsed: Result<Vec<serde_json::Value>, _> = serde_json::from_str(&cleaned);

    let indices: Vec<usize> = match parsed {
        Ok(values) => values
            .iter()
            .filter_map(|v| v.as_u64().map(|n| n as usize))
            .filter(|&idx| idx < total)
            .collect(),
        Err(_) => {
            // Fallback: return original order.
            return (0..total).collect();
        }
    };

    if indices.is_empty() {
        return (0..total).collect();
    }

    // Deduplicate while preserving order.
    let mut seen = std::collections::HashSet::new();
    indices
        .into_iter()
        .filter(|idx| seen.insert(*idx))
        .collect()
}

/// Extracts the first JSON array substring (`[...]`) from `text`,
/// stripping surrounding markdown fences if present.
fn extract_json_array(text: &str) -> String {
    // Strip markdown code fences.
    let stripped = text
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    // Find the first `[` and the last `]`.
    if let (Some(start), Some(end)) = (stripped.find('['), stripped.rfind(']')) {
        if start < end {
            return stripped[start..=end].to_string();
        }
    }

    stripped.to_string()
}

/// Applies the parsed ranking to the results, assigning linearly
/// decreasing scores.
///
/// Any indices not mentioned by the LLM are appended at the end in their
/// original order.
fn apply_ranking(
    results: Vec<SearchResult>,
    ranking: &[usize],
    total: usize,
) -> Vec<SearchResult> {
    let mut mentioned: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut reordered: Vec<SearchResult> = Vec::with_capacity(total);

    // Place ranked results first.
    for &idx in ranking {
        if idx < results.len() && mentioned.insert(idx) {
            reordered.push(results[idx].clone());
        }
    }

    // Append any results not mentioned by the LLM.
    for (i, _) in results.iter().enumerate() {
        if !mentioned.contains(&i) {
            reordered.push(results[i].clone());
        }
    }

    // Assign new scores: 1.0 for best, decreasing linearly.
    let count = reordered.len();
    if count > 0 {
        for (i, result) in reordered.iter_mut().enumerate() {
            result.score = if count == 1 {
                1.0
            } else {
                1.0 - (i as f64 / count as f64)
            };
        }
    }

    reordered
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_json_array_plain() {
        let input = "[3, 1, 2, 0]";
        assert_eq!(extract_json_array(input), "[3, 1, 2, 0]");
    }

    #[test]
    fn test_extract_json_array_with_markdown() {
        let input = "```json\n[3, 1, 2, 0]\n```";
        assert_eq!(extract_json_array(input), "[3, 1, 2, 0]");
    }

    #[test]
    fn test_extract_json_array_with_extra_text() {
        let input = "Here is the ranking:\n[3, 1, 2, 0]\nHope this helps!";
        assert_eq!(extract_json_array(input), "[3, 1, 2, 0]");
    }

    #[test]
    fn test_parse_ranking_valid() {
        let response = "[2, 0, 1]";
        let ranking = parse_ranking(response, 3);
        assert_eq!(ranking, vec![2, 0, 1]);
    }

    #[test]
    fn test_parse_ranking_with_invalid_indices() {
        let response = "[2, 99, 0, 1]";
        let ranking = parse_ranking(response, 3);
        assert_eq!(ranking, vec![2, 0, 1]);
    }

    #[test]
    fn test_parse_ranking_garbage() {
        let response = "I don't know what you mean";
        let ranking = parse_ranking(response, 3);
        assert_eq!(ranking, vec![0, 1, 2]); // fallback to original order
    }

    #[test]
    fn test_parse_ranking_deduplication() {
        let response = "[1, 1, 0, 2]";
        let ranking = parse_ranking(response, 3);
        assert_eq!(ranking, vec![1, 0, 2]);
    }

    #[test]
    fn test_build_prompt_structure() {
        let results = vec![make_result("chunk0"), make_result("chunk1")];
        let prompt = build_prompt("test query", &results);
        assert!(prompt.contains("Query: test query"));
        assert!(prompt.contains("[0] chunk0"));
        assert!(prompt.contains("[1] chunk1"));
        assert!(prompt.contains("JSON array"));
    }

    fn make_result(text: &str) -> SearchResult {
        SearchResult {
            chunk_id: "test-id".to_string(),
            text: text.to_string(),
            score: 0.5,
            metadata: ms_core::chunk::ChunkMetadata {
                source_file: "test.txt".to_string(),
                page: None,
                line_start: None,
                line_end: None,
                chunk_index: 0,
                doc_id: uuid::Uuid::nil(),
                mode: ms_core::chunk::DataMode::Default,
            },
            arm_scores: std::collections::HashMap::new(),
        }
    }
}
