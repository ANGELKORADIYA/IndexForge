use crate::provider::LLMProvider;
use anyhow::{Context, Result};
use reqwest::Client;
use serde_json::json;

pub struct GeminiProvider {
    client: Client,
    api_key: String,
    model: String,
}

impl GeminiProvider {
    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model: std::env::var("GEMINI_MODEL").unwrap_or_else(|_| "gemini-1.5-pro-latest".to_string()),
        }
    }
}

#[async_trait::async_trait]
impl LLMProvider for GeminiProvider {
    async fn generate_answer(&self, prompt: &str) -> Result<String> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            self.model, self.api_key
        );

        let req_body = json!({
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        });

        let resp = self.client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&req_body)
            .send()
            .await
            .context("Failed to connect to Gemini API")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("Gemini error ({}): {}", status, text);
        }

        let json: serde_json::Value = resp.json().await?;
        
        let answer = json["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .context("Unexpected JSON structure from Gemini API")?
            .to_string();

        Ok(answer)
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}
