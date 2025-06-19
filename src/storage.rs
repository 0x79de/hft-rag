use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::collections::HashMap;
use crate::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: Uuid,
    pub content: String,
    pub metadata: HashMap<String, String>,
    pub embedding: Vec<f32>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    pub embedding: Vec<f32>,
    pub top_k: usize,
    pub threshold: f32,
    pub filters: Option<HashMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub document: Document,
    pub score: f32,
}

#[async_trait]
pub trait VectorStorage: Send + Sync {
    async fn insert_document(&self, document: Document) -> Result<()>;
    async fn insert_batch(&self, documents: Vec<Document>) -> Result<()>;
    async fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>>;
    async fn delete_document(&self, id: Uuid) -> Result<()>;
    async fn get_document(&self, id: Uuid) -> Result<Option<Document>>;
}

// In-memory storage implementation for development and testing
// Replace with actual Qdrant implementation once API compatibility is resolved
pub struct QdrantStorage {
    documents: std::sync::Arc<dashmap::DashMap<Uuid, Document>>,
    _url: String,
    _api_key: Option<String>,
}

impl QdrantStorage {
    pub async fn new(url: &str, api_key: Option<&str>) -> Result<Self> {
        tracing::warn!("Using in-memory storage implementation. Replace with Qdrant for production.");
        
        Ok(Self {
            documents: std::sync::Arc::new(dashmap::DashMap::new()),
            _url: url.to_string(),
            _api_key: api_key.map(|s| s.to_string()),
        })
    }
    
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
}

#[async_trait]
impl VectorStorage for QdrantStorage {
    async fn insert_document(&self, document: Document) -> Result<()> {
        self.documents.insert(document.id, document);
        Ok(())
    }

    async fn insert_batch(&self, documents: Vec<Document>) -> Result<()> {
        for document in documents {
            self.documents.insert(document.id, document);
        }
        Ok(())
    }

    async fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>> {
        let mut candidates = Vec::new();
        
        // Get all documents and calculate similarities
        for doc_ref in self.documents.iter() {
            let document = doc_ref.value();
            
            // Apply filters if provided
            if let Some(ref filters) = query.filters {
                let mut matches = true;
                for (key, value) in filters {
                    if let Some(doc_value) = document.metadata.get(key) {
                        if doc_value != value {
                            matches = false;
                            break;
                        }
                    } else {
                        matches = false;
                        break;
                    }
                }
                if !matches {
                    continue;
                }
            }
            
            // Calculate similarity
            let similarity = Self::cosine_similarity(&query.embedding, &document.embedding);
            
            if similarity >= query.threshold {
                candidates.push(SearchResult {
                    document: document.clone(),
                    score: similarity,
                });
            }
        }
        
        // Sort by similarity (descending) and take top k
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(query.top_k);
        
        Ok(candidates)
    }

    async fn delete_document(&self, id: Uuid) -> Result<()> {
        self.documents.remove(&id);
        Ok(())
    }

    async fn get_document(&self, id: Uuid) -> Result<Option<Document>> {
        Ok(self.documents.get(&id).map(|doc_ref| doc_ref.value().clone()))
    }
}