use crate::{gemini, ollama, openrouter};
use std::env;

#[async_trait::async_trait]
pub trait LLMProvider: Send + Sync {
    async fn generate_answer(&self, prompt: &str) -> anyhow::Result<String>;
    fn model_name(&self) -> &str;
}

pub fn get_provider() -> anyhow::Result<Box<dyn LLMProvider>> {
    if let Ok(key) = env::var("GEMINI_API_KEY") {
        Ok(Box::new(gemini::GeminiProvider::new(key)))
    } else if let Ok(key) = env::var("OPENROUTER_API_KEY") {
        Ok(Box::new(openrouter::OpenRouterProvider::new(key)))
    } else {
        Ok(Box::new(ollama::OllamaProvider::new()))
    }
}
