use pgvector::Vector;
use sqlx::PgPool;
use anyhow::Result;

pub struct VectorResult {
    pub chunk_id: String,
    pub text: String,
    pub score: f64, // cosine similarity (1 - distance)
}

/// Semantic ANN search via pgvector HNSW index.
/// Returns top_k chunks ordered by cosine similarity to `query_embedding`.
pub async fn search_vector(
    pool: &PgPool,
    query_embedding: Vec<f32>,
    mode: &str,
    top_k: usize,
) -> Result<Vec<VectorResult>> {
    let vec = Vector::from(query_embedding);

    // pgvector uses <=> for cosine distance (lower = more similar).
    // We convert distance → similarity = 1 - distance.
    let rows = sqlx::query!(
        r#"
        SELECT id::text, text,
               1.0 - (embedding <=> $1) AS similarity
        FROM chunks
        WHERE mode = $2
          AND embedding IS NOT NULL
        ORDER BY embedding <=> $1
        LIMIT $3
        "#,
        vec as Vector,
        mode,
        top_k as i64
    )
    .fetch_all(pool)
    .await?;

    let results = rows
        .into_iter()
        .map(|r| VectorResult {
            chunk_id: r.id.unwrap_or_default(),
            text: r.text,
            score: r.similarity.unwrap_or(0.0),
        })
        .collect();

    Ok(results)
}
