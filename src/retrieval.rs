use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use crate::Result;
use crate::storage::{VectorStorage, SearchQuery, SearchResult, Document};
use crate::query::EnhancedQuery;
use crate::embedding::EmbeddingService;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalContext {
    pub results: Vec<SearchResult>,
    pub total_score: f32,
    pub query_metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    pub top_k: usize,
    pub similarity_threshold: f32,
    pub max_context_length: usize,
    pub rerank_top_n: usize,
    pub temporal_decay_factor: f32,
    // Dynamic mode switching
    pub full_context_threshold: usize,
    pub rag_mode_scale_factor: usize,
    pub context_switch_latency_ms: u64,
}

#[derive(Debug, Clone)]
pub enum RetrievalMode {
    FullContext,    // Use all available context when content is limited
    RagMode,        // Use retrieval-based mode for large content sets
    Adaptive,       // Dynamically switch based on performance
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            similarity_threshold: 0.7,
            max_context_length: 4000,
            rerank_top_n: 20,
            temporal_decay_factor: 0.1,
            full_context_threshold: 100,
            rag_mode_scale_factor: 10,
            context_switch_latency_ms: 50,
        }
    }
}

#[async_trait]
pub trait RetrievalPipeline: Send + Sync {
    async fn retrieve(&self, query: EnhancedQuery) -> Result<RetrievalContext>;
}

#[allow(dead_code)]
pub struct MultiStageRetrieval {
    storage: Arc<dyn VectorStorage>,
    embedding_service: Arc<dyn EmbeddingService>,
    config: RetrievalConfig,
    current_mode: RetrievalMode,
    total_document_count: Arc<std::sync::atomic::AtomicUsize>,
}

#[allow(dead_code)]
impl MultiStageRetrieval {
    pub fn new(
        storage: Arc<dyn VectorStorage>,
        embedding_service: Arc<dyn EmbeddingService>,
        config: RetrievalConfig,
    ) -> Self {
        Self {
            storage,
            embedding_service,
            config,
            current_mode: RetrievalMode::Adaptive,
            total_document_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }
    
    async fn determine_retrieval_mode(&self, query: &EnhancedQuery) -> RetrievalMode {
        let doc_count = self.total_document_count.load(std::sync::atomic::Ordering::Relaxed);
        
        // Switch to full context mode if document count is below threshold
        if doc_count < self.config.full_context_threshold {
            return RetrievalMode::FullContext;
        }
        
        // For high-priority queries, use adaptive mode with performance monitoring
        if matches!(query.intent, crate::query::QueryIntent::RiskAssessment) {
            return RetrievalMode::Adaptive;
        }
        
        // Default to RAG mode for large document sets
        RetrievalMode::RagMode
    }
    
    async fn retrieve_full_context(&self, query: &EnhancedQuery) -> Result<RetrievalContext> {
        // Load all documents (up to threshold) with metadata filtering
        let search_query = SearchQuery {
            embedding: query.embedding.clone(),
            top_k: self.config.full_context_threshold,
            threshold: 0.0, // Lower threshold for full context
            filters: self.build_metadata_filters(query),
        };
        
        let results = self.storage.search(search_query).await?;
        Ok(RetrievalContext {
            results,
            total_score: 1.0, // Full context always gets max score
            query_metadata: serde_json::json!({
                "mode": "full_context",
                "document_count": self.total_document_count.load(std::sync::atomic::Ordering::Relaxed)
            }),
        })
    }
    
    async fn retrieve_with_performance_monitoring(&self, query: &EnhancedQuery) -> Result<RetrievalContext> {
        let start_time = Instant::now();
        
        // Try standard RAG retrieval first
        let result = self.retrieve_rag_mode(query).await;
        let latency = start_time.elapsed().as_millis() as u64;
        
        // If latency exceeds threshold, switch to simpler mode
        if latency > self.config.context_switch_latency_ms {
            tracing::warn!("High retrieval latency {}ms, switching to simplified mode", latency);
            return self.retrieve_simplified_mode(query).await;
        }
        
        result
    }
    
    async fn retrieve_rag_mode(&self, query: &EnhancedQuery) -> Result<RetrievalContext> {
        // Enhanced RAG with intelligent filtering (already implemented)
        let results = self.vector_search(query).await?;
        let reranked_results = self.rerank_by_relevance_and_recency(results);
        Ok(self.assemble_context(reranked_results))
    }
    
    async fn retrieve_simplified_mode(&self, query: &EnhancedQuery) -> Result<RetrievalContext> {
        // Simplified retrieval with reduced processing for low latency
        let search_query = SearchQuery {
            embedding: query.embedding.clone(),
            top_k: (self.config.top_k / 2).max(5), // Reduce top_k for speed
            threshold: self.config.similarity_threshold + 0.1, // Higher threshold
            filters: None, // Skip complex filtering for speed
        };
        
        let results = self.storage.search(search_query).await?;
        Ok(RetrievalContext {
            results,
            total_score: 0.8, // Slightly lower score for simplified mode
            query_metadata: serde_json::json!({
                "mode": "simplified",
                "performance_fallback": true
            }),
        })
    }

    async fn vector_search(&self, query: &EnhancedQuery) -> Result<Vec<SearchResult>> {
        // Intelligent retrieval: dynamic top-k based on query complexity and available content
        let dynamic_top_k = self.calculate_dynamic_top_k(query);
        let dynamic_threshold = self.calculate_dynamic_threshold(query);
        
        let search_query = SearchQuery {
            embedding: query.embedding.clone(),
            top_k: dynamic_top_k,
            threshold: dynamic_threshold,
            filters: self.build_metadata_filters(query),
        };

        // Use timeout for vector search to ensure sub-20ms performance
        let search_future = self.storage.search(search_query);
        let timeout_duration = tokio::time::Duration::from_millis(15); // Leave 5ms for post-processing
        
        let results = match tokio::time::timeout(timeout_duration, search_future).await {
            Ok(Ok(results)) => results,
            Ok(Err(e)) => return Err(e),
            Err(_) => {
                tracing::warn!("Vector search timeout after 15ms, returning empty results");
                return Ok(vec![]);
            }
        };
        
        // Apply intelligent filtering: only return results above relevance threshold
        Ok(self.filter_by_intelligent_relevance(results, query))
    }
    
    fn calculate_dynamic_top_k(&self, query: &EnhancedQuery) -> usize {
        let base_k = self.config.top_k; // Use config top_k instead of rerank_top_n
        
        // For performance optimization, limit complexity multipliers
        let complexity_multiplier = if query.market_filters.len() > 2 || 
                                      query.temporal_context.is_some() ||
                                      query.expanded_text.len() > 50 {
            1.5 // Reduced from 2.0 for performance
        } else {
            1.0
        };
        
        // Scale based on query intent - optimize for speed
        let intent_multiplier = match query.intent {
            crate::query::QueryIntent::TradingStrategy => 1.3, // Reduced from 1.5
            crate::query::QueryIntent::MarketData => 1.1,      // Reduced from 1.2
            crate::query::QueryIntent::RiskAssessment => 1.4,  // Reduced from 1.8
            _ => 1.0,
        };
        
        // Cap at 50 for better performance (was 100)
        ((base_k as f64 * complexity_multiplier * intent_multiplier) as usize).min(50)
    }
    
    fn calculate_dynamic_threshold(&self, query: &EnhancedQuery) -> f32 {
        let base_threshold = self.config.similarity_threshold;
        
        // Lower threshold for broad market analysis, higher for specific queries
        match query.intent {
            crate::query::QueryIntent::MarketData => base_threshold - 0.05,
            crate::query::QueryIntent::TradingStrategy => base_threshold + 0.02,
            crate::query::QueryIntent::RiskAssessment => base_threshold + 0.03,
            _ => base_threshold,
        }
    }
    
    fn filter_by_intelligent_relevance(&self, mut results: Vec<SearchResult>, query: &EnhancedQuery) -> Vec<SearchResult> {
        if results.is_empty() {
            return results;
        }
        
        // Optimize for speed: reduce cutoff calculation overhead
        let top_score = results[0].score; // Direct access instead of optional
        let relevance_cutoff = top_score * 0.75; // Slightly higher cutoff (was 0.7) for fewer results
        
        // Apply temporal boost only for time-sensitive queries (optimized)
        if query.temporal_context.is_some() {
            let now_timestamp = chrono::Utc::now().timestamp();
            for result in &mut results {
                if let Some(timestamp_str) = result.document.metadata.get("timestamp") {
                    if let Ok(ts) = timestamp_str.parse::<i64>() {
                        let age_hours = (now_timestamp - ts) / 3600;
                        if age_hours < 24 {
                            result.score *= 1.15; // Slightly reduced boost (was 1.2) for performance
                        }
                    }
                }
            }
            // Use unstable sort for better performance
            results.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        }
        
        // Filter by intelligent relevance threshold - optimized for speed
        results.into_iter()
            .filter(|r| r.score >= relevance_cutoff)
            .take(15) // Reduced from 20 for better performance
            .collect()
    }

    fn build_metadata_filters(&self, query: &EnhancedQuery) -> Option<std::collections::HashMap<String, String>> {
        if query.market_filters.is_empty() {
            return None;
        }

        let mut filters = std::collections::HashMap::new();
        
        for market_filter in &query.market_filters {
            if let Some(symbol) = &market_filter.symbol {
                filters.insert("symbol".to_string(), symbol.clone());
            }
            if let Some(asset_class) = &market_filter.asset_class {
                filters.insert("asset_class".to_string(), asset_class.clone());
            }
        }

        if filters.is_empty() {
            None
        } else {
            Some(filters)
        }
    }

    fn rerank_by_relevance_and_recency(&self, results: Vec<SearchResult>) -> Vec<SearchResult> {
        let mut scored_results: Vec<(SearchResult, f32)> = results
            .into_iter()
            .map(|result| {
                let recency_score = self.calculate_recency_score(&result.document);
                let combined_score = result.score * 0.7 + recency_score * 0.3;
                (result, combined_score)
            })
            .collect();

        // Sort by combined score (descending)
        scored_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top_k and remove the combined score
        scored_results
            .into_iter()
            .take(self.config.top_k)
            .map(|(mut result, combined_score)| {
                result.score = combined_score;
                result
            })
            .collect()
    }

    fn calculate_recency_score(&self, document: &Document) -> f32 {
        let now = chrono::Utc::now();
        let age_hours = (now - document.timestamp).num_hours() as f32;
        
        // Exponential decay: more recent documents get higher scores
        (-age_hours * self.config.temporal_decay_factor).exp()
    }

    fn assemble_context(&self, results: Vec<SearchResult>) -> RetrievalContext {
        let mut total_length = 0;
        let mut selected_documents = Vec::new();
        let mut total_score = 0.0;

        for result in results {
            let content_length = result.document.content.len();
            
            if total_length + content_length <= self.config.max_context_length {
                total_length += content_length;
                total_score += result.score;
                selected_documents.push(result);
            } else {
                break;
            }
        }

        let num_docs = selected_documents.len();
        RetrievalContext {
            results: selected_documents,
            total_score: total_score / num_docs as f32,
            query_metadata: serde_json::json!({
                "total_documents": num_docs,
                "total_content_length": total_length,
                "average_score": total_score / num_docs as f32,
            }),
        }
    }
}

#[async_trait]
impl RetrievalPipeline for MultiStageRetrieval {
    async fn retrieve(&self, query: EnhancedQuery) -> Result<RetrievalContext> {
        // Stage 1: Vector similarity search with metadata filtering
        let initial_results = self.vector_search(&query).await?;
        
        if initial_results.is_empty() {
            return Ok(RetrievalContext {
                results: vec![],
                total_score: 0.0,
                query_metadata: serde_json::json!({
                    "message": "No documents found matching the query",
                    "total_documents": 0,
                }),
            });
        }

        // Stage 2: Re-rank by relevance and recency
        let reranked_results = self.rerank_by_relevance_and_recency(initial_results);

        // Stage 3: Assemble context within length limits
        let context = self.assemble_context(reranked_results);

        Ok(context)
    }
}

pub struct HybridRetrieval {
    vector_retrieval: MultiStageRetrieval,
    // TODO: Add full-text search component (Tantivy)
}

impl HybridRetrieval {
    pub fn new(
        storage: Arc<dyn VectorStorage>,
        embedding_service: Arc<dyn EmbeddingService>,
        config: RetrievalConfig,
    ) -> Self {
        Self {
            vector_retrieval: MultiStageRetrieval::new(storage, embedding_service, config),
        }
    }
}

#[async_trait]
impl RetrievalPipeline for HybridRetrieval {
    async fn retrieve(&self, query: EnhancedQuery) -> Result<RetrievalContext> {
        // For now, just use vector retrieval
        // TODO: Combine with full-text search results
        self.vector_retrieval.retrieve(query).await
    }
}