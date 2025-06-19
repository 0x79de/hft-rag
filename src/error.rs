use thiserror::Error;

pub type Result<T> = std::result::Result<T, RagError>;

#[derive(Error, Debug)]
pub enum RagError {
    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Ingestion error: {0}")]
    Ingestion(String),

    #[error("Query processing error: {0}")]
    Query(String),

    #[error("Retrieval error: {0}")]
    Retrieval(String),

    #[error("Integration error: {0}")]
    Integration(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("HTTP error: {0}")]
    Http(#[from] axum::http::Error),

    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),

    #[error("HuggingFace Hub error: {0}")]
    HfHub(#[from] hf_hub::api::tokio::ApiError),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Qdrant client error: {0}")]
    Qdrant(#[from] anyhow::Error),
    
    #[error("Qdrant error: {0}")]
    QdrantClient(Box<qdrant_client::QdrantError>),
}

impl From<qdrant_client::QdrantError> for RagError {
    fn from(err: qdrant_client::QdrantError) -> Self {
        RagError::QdrantClient(Box::new(err))
    }
}