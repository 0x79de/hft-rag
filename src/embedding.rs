use async_trait::async_trait;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use tokenizers::Tokenizer;
use hf_hub::api::tokio::Api;
use std::sync::Arc;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use dashmap::DashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use crate::{Result, RagError};

#[async_trait]
pub trait EmbeddingService: Send + Sync {
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>>;
    async fn embed_batch(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>>;
    fn dimension(&self) -> usize;
}

pub struct CandleEmbedding {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    dimension: usize,
    cache: Arc<DashMap<u64, Vec<f32>>>,
}

impl CandleEmbedding {
    pub async fn new() -> Result<Self> {
        let device = Device::Cpu; // Use CPU for compatibility
        
        // Download model from HuggingFace
        let api = Api::new()?;
        let repo = api.model("sentence-transformers/all-MiniLM-L6-v2".to_string());
        
        // Download tokenizer
        let tokenizer_path = repo.get("tokenizer.json").await?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| RagError::Embedding(format!("Failed to load tokenizer: {}", e)))?;
        
        // Download model config
        let config_path = repo.get("config.json").await?;
        let config_content = std::fs::read_to_string(config_path)?;
        let config: Config = serde_json::from_str(&config_content)?;
        
        // Download model weights - try safetensors first, fallback to pytorch
        let weights_path = match repo.get("model.safetensors").await {
            Ok(path) => path,
            Err(_) => repo.get("pytorch_model.bin").await?,
        };
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], candle_core::DType::F32, &device)? };
        
        // Load model
        let model = BertModel::load(vb, &config)?;
        
        Ok(Self {
            model,
            tokenizer,
            device,
            dimension: 384, // MiniLM-L6-v2 output dimension
            cache: Arc::new(DashMap::new()),
        })
    }
    
    pub async fn new_with_cache_size(_cache_size: usize) -> Result<Self> {
        // For cases where we want to limit cache size
        let embedding = Self::new().await?;
        // Note: DashMap doesn't have built-in size limits, but we could implement LRU
        Ok(embedding)
    }
    
    fn get_cache_key(&self, text: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        hasher.finish()
    }
    
    async fn compute_embedding(&self, text: &str) -> Result<Vec<f32>> {
        // Tokenize the text
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| RagError::Embedding(format!("Tokenization failed: {}", e)))?;
        
        let tokens = encoding.get_ids();
        let token_ids = Tensor::new(tokens, &self.device)?.unsqueeze(0)?;
        let attention_mask = Tensor::ones((1, tokens.len()), candle_core::DType::U32, &self.device)?;
        
        // Get model embeddings
        let embeddings = self.model.forward(&token_ids, &attention_mask, None)?;
        
        // Mean pooling to get sentence embedding
        let pooled = embeddings.mean(1)?;
        let embedding_vec = pooled.to_vec2::<f32>()?[0].clone();
        
        // Normalize the embedding
        let norm = embedding_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normalized: Vec<f32> = embedding_vec.iter().map(|x| x / norm).collect();
        
        Ok(normalized)
    }
}

#[async_trait]
impl EmbeddingService for CandleEmbedding {
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        // Check cache first
        let cache_key = self.get_cache_key(text);
        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(cached.clone());
        }
        
        // Compute embedding
        let embedding = self.compute_embedding(text).await?;
        
        // Cache the result
        self.cache.insert(cache_key, embedding.clone());
        
        Ok(embedding)
    }

    async fn embed_batch(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::new();
        let mut uncached_texts = Vec::new();
        let mut cache_keys = Vec::new();
        
        // Check cache for all texts
        for text in &texts {
            let cache_key = self.get_cache_key(text);
            cache_keys.push(cache_key);
            
            if let Some(cached) = self.cache.get(&cache_key) {
                embeddings.push(Some(cached.clone()));
            } else {
                embeddings.push(None);
                uncached_texts.push(*text);
            }
        }
        
        // Compute embeddings for uncached texts in parallel
        let uncached_results = futures::future::try_join_all(
            uncached_texts.into_iter().map(|text| self.compute_embedding(text))
        ).await?;
        
        // Fill in the uncached results and update cache
        let mut uncached_idx = 0;
        let mut final_embeddings = Vec::new();
        
        for (i, embedding_opt) in embeddings.into_iter().enumerate() {
            if let Some(embedding) = embedding_opt {
                final_embeddings.push(embedding);
            } else {
                let embedding = uncached_results[uncached_idx].clone();
                self.cache.insert(cache_keys[i], embedding.clone());
                final_embeddings.push(embedding);
                uncached_idx += 1;
            }
        }
        
        Ok(final_embeddings)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

#[derive(Serialize)]
struct OpenAIEmbeddingRequest {
    input: Vec<String>,
    model: String,
}

#[derive(Deserialize)]
struct OpenAIEmbeddingResponse {
    data: Vec<OpenAIEmbeddingData>,
}

#[derive(Deserialize)]
struct OpenAIEmbeddingData {
    embedding: Vec<f32>,
}

pub struct OpenAIEmbedding {
    client: Client,
    api_key: String,
    model: String,
    dimension: usize,
    cache: Arc<DashMap<u64, Vec<f32>>>,
}

impl OpenAIEmbedding {
    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model: "text-embedding-3-small".to_string(),
            dimension: 1536,
            cache: Arc::new(DashMap::new()),
        }
    }
    
    pub fn with_model(api_key: String, model: String, dimension: usize) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model,
            dimension,
            cache: Arc::new(DashMap::new()),
        }
    }
    
    fn get_cache_key(&self, text: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        self.model.hash(&mut hasher);
        hasher.finish()
    }
    
    async fn call_openai_api(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let request = OpenAIEmbeddingRequest {
            input: texts,
            model: self.model.clone(),
        };
        
        let response = self.client
            .post("https://api.openai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| RagError::Embedding(format!("OpenAI API call failed: {}", e)))?;
        
        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(RagError::Embedding(format!("OpenAI API error: {}", error_text)));
        }
        
        let embedding_response: OpenAIEmbeddingResponse = response.json().await
            .map_err(|e| RagError::Embedding(format!("Failed to parse OpenAI response: {}", e)))?;
        
        Ok(embedding_response.data.into_iter().map(|d| d.embedding).collect())
    }
}

#[async_trait]
impl EmbeddingService for OpenAIEmbedding {
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let cache_key = self.get_cache_key(text);
        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(cached.clone());
        }
        
        let embeddings = self.call_openai_api(vec![text.to_string()]).await?;
        let embedding = embeddings.into_iter().next()
            .ok_or_else(|| RagError::Embedding("No embedding returned from OpenAI".to_string()))?;
        
        self.cache.insert(cache_key, embedding.clone());
        Ok(embedding)
    }

    async fn embed_batch(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::new();
        let mut uncached_texts = Vec::new();
        let mut cache_keys = Vec::new();
        let mut indices = Vec::new();
        
        // Check cache for all texts
        for (i, text) in texts.iter().enumerate() {
            let cache_key = self.get_cache_key(text);
            cache_keys.push(cache_key);
            
            if let Some(cached) = self.cache.get(&cache_key) {
                embeddings.push(Some(cached.clone()));
            } else {
                embeddings.push(None);
                uncached_texts.push(text.to_string());
                indices.push(i);
            }
        }
        
        // Process uncached texts in batches (OpenAI has batch limits)
        let batch_size = 100; // OpenAI batch limit
        let mut uncached_results = Vec::new();
        
        for chunk in uncached_texts.chunks(batch_size) {
            let mut batch_results = self.call_openai_api(chunk.to_vec()).await?;
            uncached_results.append(&mut batch_results);
        }
        
        // Fill in the uncached results and update cache
        let mut uncached_idx = 0;
        let mut final_embeddings = Vec::new();
        
        for (i, embedding_opt) in embeddings.into_iter().enumerate() {
            if let Some(embedding) = embedding_opt {
                final_embeddings.push(embedding);
            } else {
                let embedding = uncached_results[uncached_idx].clone();
                self.cache.insert(cache_keys[i], embedding.clone());
                final_embeddings.push(embedding);
                uncached_idx += 1;
            }
        }
        
        Ok(final_embeddings)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

// Fallback embedding service that tries Candle first, then OpenAI
pub struct FallbackEmbedding {
    candle: Option<CandleEmbedding>,
    openai: Option<OpenAIEmbedding>,
    preferred_dimension: usize,
}

impl FallbackEmbedding {
    pub async fn new(openai_api_key: Option<String>) -> Result<Self> {
        let candle = match CandleEmbedding::new().await {
            Ok(service) => Some(service),
            Err(e) => {
                tracing::warn!("Failed to initialize Candle embedding service: {}", e);
                None
            }
        };
        
        let openai = openai_api_key.map(OpenAIEmbedding::new);
        
        let preferred_dimension = candle.as_ref().map(|c| c.dimension()).unwrap_or(1536);
        
        if candle.is_none() && openai.is_none() {
            return Err(RagError::Embedding("No embedding service available".to_string()));
        }
        
        Ok(Self {
            candle,
            openai,
            preferred_dimension,
        })
    }
}

#[async_trait]
impl EmbeddingService for FallbackEmbedding {
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        // Try Candle first (local, faster)
        if let Some(candle) = &self.candle {
            match candle.embed_text(text).await {
                Ok(embedding) => return Ok(embedding),
                Err(e) => tracing::warn!("Candle embedding failed, falling back to OpenAI: {}", e),
            }
        }
        
        // Fallback to OpenAI
        if let Some(openai) = &self.openai {
            return openai.embed_text(text).await;
        }
        
        Err(RagError::Embedding("No embedding service available".to_string()))
    }

    async fn embed_batch(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>> {
        // Try Candle first
        if let Some(candle) = &self.candle {
            match candle.embed_batch(texts.clone()).await {
                Ok(embeddings) => return Ok(embeddings),
                Err(e) => tracing::warn!("Candle batch embedding failed, falling back to OpenAI: {}", e),
            }
        }
        
        // Fallback to OpenAI
        if let Some(openai) = &self.openai {
            return openai.embed_batch(texts).await;
        }
        
        Err(RagError::Embedding("No embedding service available".to_string()))
    }

    fn dimension(&self) -> usize {
        self.preferred_dimension
    }
}