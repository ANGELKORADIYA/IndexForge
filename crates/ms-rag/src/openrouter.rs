use crate::provider::LLMProvider;
use anyhow::{Context, Result};
use reqwest::Client;
use serde_json::json;

pub struct OpenRouterProvider {
    client: Client,
    api_key: String,
    model: String,
}

impl OpenRouterProvider {
    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model: std::env::var("OPENROUTER_MODEL").unwrap_or_else(|_| "meta-llama/llama-3-8b-instruct".to_string()),
        }
    }
}

#[async_trait::async_trait]
impl LLMProvider for OpenRouterProvider {
    async fn generate_answer(&self, prompt: &str) -> Result<String> {
        let url = "https://openrouter.ai/api/v1/chat/completions";

        let req_body = json!({
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        });

        let resp = self.client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("HTTP-Referer", "http://localhost:3000") // Required by OpenRouter
            .header("X-Title", "MemorySearch")
            .json(&req_body)
            .send()
            .await
            .context("Failed to connect to OpenRouter API")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("OpenRouter error ({}): {}", status, text);
        }

        let json: serde_json::Value = resp.json().await?;
        let answer = json["choices"][0]["message"]["content"]
            .as_str()
            .context("Unexpected JSON structure from OpenRouter")?
            .to_string();

        Ok(answer)
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}
