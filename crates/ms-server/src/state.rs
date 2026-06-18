use ms_core::Config;
use ms_ingest::embedder::Embedder;
use ms_rerank::CrossEncoder;
use sqlx::PgPool;
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct AppStateInner {
    pub pool: PgPool,
    pub config: Config,
    // Note: BM25Index isn't strictly thread-safe to share if we rebuild it often, 
    // but for search-only sharing an Arc<BM25Index> is fine.
    // However, if we ingest while running, we'd need a RwLock or reload it.
    // For now, we'll keep it simple and instantiate it per request if needed, 
    // or just lock it if it's cheap to load. Let's just lock it.
    pub embedder: Mutex<Embedder>,
    pub cross_encoder: Mutex<CrossEncoder>,
}

pub type AppState = Arc<AppStateInner>;
