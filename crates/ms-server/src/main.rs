mod routes;
mod state;

use axum::{
    routing::{get, post},
    Router,
};
use dotenvy::dotenv;
use ms_core::Config;
use ms_ingest::embedder::Embedder;
use ms_rerank::CrossEncoder;
use sqlx::postgres::PgPoolOptions;
use std::env;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv().ok();

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "ms_server=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let db_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&db_url)
        .await?;

    let config = Config::default();
    let embedder = Embedder::new(None)?;
    let cross_encoder = CrossEncoder::new()?;

    let app_state = Arc::new(state::AppStateInner {
        pool,
        config,
        embedder: Mutex::new(embedder),
        cross_encoder: Mutex::new(cross_encoder),
    });

    let app = Router::new()
        .route("/health", get(routes::health::health_check))
        .route("/api/search", get(routes::search::search_handler))
        .route("/api/index", post(routes::ingest::ingest_handler))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(app_state);

    let port = env::var("SERVER_PORT").unwrap_or_else(|_| "8080".to_string());
    let addr = format!("0.0.0.0:{}", port);
    
    tracing::info!("Starting memory search server on {}", addr);
    let listener = TcpListener::bind(&addr).await?;
    
    axum::serve(listener, app).await?;

    Ok(())
}
