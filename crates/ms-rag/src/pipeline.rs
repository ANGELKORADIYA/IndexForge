use crate::provider::LLMProvider;
use anyhow::Result;
use ms_core::{RagAnswer, SearchResult};

pub struct RagPipeline {
    provider: Box<dyn LLMProvider>,
}

impl RagPipeline {
    pub fn new(provider: Box<dyn LLMProvider>) -> Self {
        Self { provider }
    }

    pub async fn answer(
        &self,
        query: &str,
        context_chunks: &[SearchResult],
    ) -> Result<RagAnswer> {
        let mut context_text = String::new();
        for (i, chunk) in context_chunks.iter().enumerate() {
            context_text.push_str(&format!("Chunk {}:\n{}\n---\n", i + 1, chunk.text));
        }

        let prompt = format!(
            "You are a helpful assistant. Answer the question using ONLY the provided context. \
            If the context does not contain enough information, say \"I don't know based on the context.\"\n\n\
            Context:\n{}\n\
            Question: {}\n\n\
            Answer:",
            context_text, query
        );

        let answer_text = self.provider.generate_answer(&prompt).await?;

        Ok(RagAnswer {
            answer: answer_text,
            sources: context_chunks.to_vec(),
            model: self.provider.model_name().to_string(),
        })
    }
}
