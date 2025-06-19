use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagConfig {
    pub server: ServerConfig,
    pub embedding: EmbeddingConfig,
    pub storage: StorageConfig,
    pub ingestion: IngestionConfig,
    pub retrieval: RetrievalConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub workers: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub provider: EmbeddingProvider,
    pub model_path: Option<String>,
    pub api_key: Option<String>,
    pub batch_size: usize,
    pub cache_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmbeddingProvider {
    Candle,
    OpenAI,
    Cohere,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub qdrant_url: String,
    pub qdrant_api_key: Option<String>,
    pub postgres_url: String,
    pub collection_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionConfig {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub batch_size: usize,
    pub supported_formats: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    pub top_k: usize,
    pub similarity_threshold: f32,
    pub max_context_length: usize,
    pub timeout_ms: u64,
    // Dynamic mode switching configuration
    pub full_context_threshold: usize,  // Number of docs below which to use full context
    pub rag_mode_scale_factor: usize,   // Multiplier for RAG mode capacity (10x scaling)
    pub context_switch_latency_ms: u64, // Max latency before switching modes
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                host: "0.0.0.0".to_string(),
                port: 8080,
                workers: None,
            },
            embedding: EmbeddingConfig {
                provider: EmbeddingProvider::Candle,
                model_path: None,
                api_key: None,
                batch_size: 32,
                cache_size: 1000,
            },
            storage: StorageConfig {
                qdrant_url: "http://localhost:6333".to_string(),
                qdrant_api_key: None,
                postgres_url: "postgresql://localhost/hft_rag".to_string(),
                collection_name: "documents".to_string(),
            },
            ingestion: IngestionConfig {
                chunk_size: 512,
                chunk_overlap: 64,
                batch_size: 100,
                supported_formats: vec!["txt".to_string(), "md".to_string(), "json".to_string(), "csv".to_string()],
            },
            retrieval: RetrievalConfig {
                top_k: 10,
                similarity_threshold: 0.7,
                max_context_length: 4000,
                timeout_ms: 5000,
                full_context_threshold: 100,      // Switch to full context if < 100 docs
                rag_mode_scale_factor: 10,        // 10x scaling capability
                context_switch_latency_ms: 50,    // Switch modes if latency > 50ms
            },
        }
    }
}