use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use std::path::PathBuf;

pub struct Embedder {
    model: TextEmbedding,
}

impl Embedder {
    pub fn new(model_dir: Option<PathBuf>) -> anyhow::Result<Self> {
        let mut opts = TextInitOptions::new(EmbeddingModel::AllMiniLML6V2)
            .with_show_download_progress(true);

        if let Some(dir) = model_dir {
            opts = opts.with_cache_dir(dir);
        }

        let model = TextEmbedding::try_new(opts)?;
        Ok(Self { model })
    }

    pub fn embed_batch(&mut self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        Ok(self.model.embed(refs, None)?)
    }

    pub fn embed_one(&mut self, text: &str) -> anyhow::Result<Vec<f32>> {
        let embeddings = self.model.embed(vec![text], None)?;
        Ok(embeddings.into_iter().next().unwrap())
    }
}
