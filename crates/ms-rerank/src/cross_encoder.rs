use anyhow::{Context, Result};
use fastembed::{RerankInitOptions, TextRerank};
use ms_core::SearchResult;

pub struct CrossEncoder {
    model: TextRerank,
}

impl CrossEncoder {
    /// Initialize the model, downloading it if necessary.
    pub fn new() -> Result<Self> {
        let model = TextRerank::try_new(
            RerankInitOptions::new(fastembed::RerankerModel::BGERerankerBase)
                .with_show_download_progress(true)
        )
        .context("Failed to initialize TextRerank")?;

        Ok(Self { model })
    }

    /// Re-score a list of (query, text) pairs.
    pub fn score(&mut self, query: &str, texts: &[&str]) -> Result<Vec<f32>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        
        let results = self.model.rerank(query, texts, false, None)?;
        
        let scores = results.into_iter().map(|r| r.score).collect();

        Ok(scores)
    }

    /// Re-rank SearchResults by cross-encoder score (descending).
    pub fn rerank(
        &mut self,
        query: &str,
        mut results: Vec<SearchResult>,
    ) -> Result<Vec<SearchResult>> {
        if results.is_empty() {
            return Ok(results);
        }

        let texts: Vec<&str> = results.iter().map(|r| r.text.as_str()).collect();
        let scores = self.score(query, &texts)?;

        for (res, &score) in results.iter_mut().zip(scores.iter()) {
            res.score = score as f64;
        }

        // Sort descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results)
    }
}
