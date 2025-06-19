use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::Path;
use crate::{Result, RagError};

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

impl RagConfig {
    /// Load configuration from environment variables and optional config file
    pub fn from_env() -> Result<Self> {
        let mut config = Self::default();
        
        // Load from environment variables
        config.load_from_env()?;
        
        // Load from config file if specified
        if let Ok(config_path) = env::var("RAG_CONFIG_PATH") {
            if Path::new(&config_path).exists() {
                config = Self::from_file(&config_path)?;
                // Override with environment variables
                config.load_from_env()?;
            }
        }
        
        config.validate()?;
        Ok(config)
    }
    
    /// Load configuration from a TOML file
    pub fn from_file(path: &str) -> Result<Self> {
        let content = fs::read_to_string(path)
            .map_err(|e| RagError::Config(format!("Failed to read config file {}: {}", path, e)))?;
        
        toml::from_str(&content)
            .map_err(|e| RagError::Config(format!("Failed to parse config file {}: {}", path, e)))
    }
    
    /// Load configuration values from environment variables
    fn load_from_env(&mut self) -> Result<()> {
        // Server configuration
        if let Ok(host) = env::var("RAG_SERVER_HOST") {
            self.server.host = host;
        }
        if let Ok(port) = env::var("RAG_SERVER_PORT") {
            self.server.port = port.parse()
                .map_err(|e| RagError::Config(format!("Invalid server port: {}", e)))?;
        }
        if let Ok(workers) = env::var("RAG_SERVER_WORKERS") {
            self.server.workers = Some(workers.parse()
                .map_err(|e| RagError::Config(format!("Invalid server workers: {}", e)))?);
        }
        
        // Embedding configuration
        if let Ok(provider) = env::var("RAG_EMBEDDING_PROVIDER") {
            self.embedding.provider = match provider.to_lowercase().as_str() {
                "candle" => EmbeddingProvider::Candle,
                "openai" => EmbeddingProvider::OpenAI,
                "cohere" => EmbeddingProvider::Cohere,
                _ => return Err(RagError::Config(format!("Invalid embedding provider: {}", provider))),
            };
        }
        if let Ok(model_path) = env::var("RAG_EMBEDDING_MODEL_PATH") {
            self.embedding.model_path = Some(model_path);
        }
        if let Ok(api_key) = env::var("RAG_EMBEDDING_API_KEY") {
            self.embedding.api_key = Some(api_key);
        }
        if let Ok(batch_size) = env::var("RAG_EMBEDDING_BATCH_SIZE") {
            self.embedding.batch_size = batch_size.parse()
                .map_err(|e| RagError::Config(format!("Invalid embedding batch size: {}", e)))?;
        }
        if let Ok(cache_size) = env::var("RAG_EMBEDDING_CACHE_SIZE") {
            self.embedding.cache_size = cache_size.parse()
                .map_err(|e| RagError::Config(format!("Invalid embedding cache size: {}", e)))?;
        }
        
        // Storage configuration
        if let Ok(qdrant_url) = env::var("RAG_QDRANT_URL") {
            self.storage.qdrant_url = qdrant_url;
        }
        if let Ok(qdrant_api_key) = env::var("RAG_QDRANT_API_KEY") {
            self.storage.qdrant_api_key = Some(qdrant_api_key);
        }
        if let Ok(postgres_url) = env::var("RAG_POSTGRES_URL") {
            self.storage.postgres_url = postgres_url;
        }
        if let Ok(collection_name) = env::var("RAG_COLLECTION_NAME") {
            self.storage.collection_name = collection_name;
        }
        
        // Retrieval configuration
        if let Ok(top_k) = env::var("RAG_RETRIEVAL_TOP_K") {
            self.retrieval.top_k = top_k.parse()
                .map_err(|e| RagError::Config(format!("Invalid retrieval top_k: {}", e)))?;
        }
        if let Ok(threshold) = env::var("RAG_RETRIEVAL_THRESHOLD") {
            self.retrieval.similarity_threshold = threshold.parse()
                .map_err(|e| RagError::Config(format!("Invalid similarity threshold: {}", e)))?;
        }
        if let Ok(max_context) = env::var("RAG_RETRIEVAL_MAX_CONTEXT") {
            self.retrieval.max_context_length = max_context.parse()
                .map_err(|e| RagError::Config(format!("Invalid max context length: {}", e)))?;
        }
        if let Ok(timeout) = env::var("RAG_RETRIEVAL_TIMEOUT_MS") {
            self.retrieval.timeout_ms = timeout.parse()
                .map_err(|e| RagError::Config(format!("Invalid retrieval timeout: {}", e)))?;
        }
        
        // Ingestion configuration
        if let Ok(chunk_size) = env::var("RAG_INGESTION_CHUNK_SIZE") {
            self.ingestion.chunk_size = chunk_size.parse()
                .map_err(|e| RagError::Config(format!("Invalid chunk size: {}", e)))?;
        }
        if let Ok(chunk_overlap) = env::var("RAG_INGESTION_CHUNK_OVERLAP") {
            self.ingestion.chunk_overlap = chunk_overlap.parse()
                .map_err(|e| RagError::Config(format!("Invalid chunk overlap: {}", e)))?;
        }
        if let Ok(batch_size) = env::var("RAG_INGESTION_BATCH_SIZE") {
            self.ingestion.batch_size = batch_size.parse()
                .map_err(|e| RagError::Config(format!("Invalid ingestion batch size: {}", e)))?;
        }
        
        Ok(())
    }
    
    /// Validate configuration values
    fn validate(&self) -> Result<()> {
        if self.server.port == 0 {
            return Err(RagError::Config("Server port cannot be 0".to_string()));
        }
        
        if self.embedding.batch_size == 0 {
            return Err(RagError::Config("Embedding batch size cannot be 0".to_string()));
        }
        
        if self.ingestion.chunk_size == 0 {
            return Err(RagError::Config("Chunk size cannot be 0".to_string()));
        }
        
        if self.ingestion.chunk_overlap >= self.ingestion.chunk_size {
            return Err(RagError::Config("Chunk overlap must be less than chunk size".to_string()));
        }
        
        if self.retrieval.similarity_threshold < 0.0 || self.retrieval.similarity_threshold > 1.0 {
            return Err(RagError::Config("Similarity threshold must be between 0.0 and 1.0".to_string()));
        }
        
        if self.retrieval.top_k == 0 {
            return Err(RagError::Config("Retrieval top_k cannot be 0".to_string()));
        }
        
        // Validate URLs
        if !self.storage.qdrant_url.starts_with("http://") && !self.storage.qdrant_url.starts_with("https://") {
            return Err(RagError::Config("Qdrant URL must start with http:// or https://".to_string()));
        }
        
        Ok(())
    }
    
    /// Save configuration to a TOML file
    pub fn save_to_file(&self, path: &str) -> Result<()> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| RagError::Config(format!("Failed to serialize config: {}", e)))?;
        
        fs::write(path, content)
            .map_err(|e| RagError::Config(format!("Failed to write config file {}: {}", path, e)))?;
        
        Ok(())
    }
}