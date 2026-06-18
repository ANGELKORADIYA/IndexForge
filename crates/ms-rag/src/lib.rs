pub mod provider;
pub mod ollama;
pub mod gemini;
pub mod openrouter;
pub mod pipeline;

pub use provider::{LLMProvider, get_provider};
pub use pipeline::RagPipeline;
