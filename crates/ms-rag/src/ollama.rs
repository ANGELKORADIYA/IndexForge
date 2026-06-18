use crate::provider::LLMProvider;
use anyhow::{Context, Result};
use reqwest::Client;
use serde_json::json;

pub struct OllamaProvider {
    client: Client,
    host: String,
    model: String,
}

impl OllamaProvider {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            host: std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string()),
            model: std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "llama3.2".to_string()),
        }
    }
}

#[async_trait::async_trait]
impl LLMProvider for OllamaProvider {
    async fn generate_answer(&self, prompt: &str) -> Result<String> {
        let url = format!("{}/api/generate", self.host);
        
        let req_body = json!({
            "model": self.model,
            "prompt": prompt,
            "stream": false
        });

        let resp = self.client
            .post(&url)
            .json(&req_body)
            .send()
            .await
            .context("Failed to connect to Ollama")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("Ollama error ({}): {}", status, text);
        }

        let json: serde_json::Value = resp.json().await?;
        let answer = json["response"]
            .as_str()
            .context("Missing 'response' field in Ollama output")?
            .to_string();

        Ok(answer)
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}
