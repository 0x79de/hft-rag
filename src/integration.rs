use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;
use crate::{Result, RagError};
use crate::query::{Query, QueryProcessor};
use crate::retrieval::{RetrievalPipeline, RetrievalContext};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketEvent {
    pub id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub event_type: MarketEventType,
    pub symbol: String,
    pub data: serde_json::Value,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketEventType {
    Trade,
    Quote,
    OrderBook,
    News,
    Signal,
    Alert,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HftRequest {
    pub request_id: String,
    pub request_type: HftRequestType,
    pub query: String,
    pub context: Option<HashMap<String, String>>,
    pub priority: Priority,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HftRequestType {
    StrategyAugmentation,
    RiskAssessment,
    MarketAnalysis,
    ContextualSearch,
    RealTimeQuery,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Priority {
    Critical,
    High,
    Normal,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HftResponse {
    pub request_id: String,
    pub context: RetrievalContext,
    pub processing_time_ms: u64,
    pub status: ResponseStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseStatus {
    Success,
    PartialSuccess,
    Failed,
    Timeout,
}

#[async_trait]
pub trait HftIntegration: Send + Sync {
    async fn process_request(&self, request: HftRequest) -> Result<HftResponse>;
    async fn ingest_market_event(&self, event: MarketEvent) -> Result<()>;
    async fn health_check(&self) -> Result<bool>;
}

pub struct McpBridge {
    query_processor: Arc<QueryProcessor>,
    retrieval_pipeline: Arc<dyn RetrievalPipeline>,
    event_ingestion_tx: mpsc::UnboundedSender<MarketEvent>,
    max_processing_time_ms: u64,
}

impl McpBridge {
    pub fn new(
        query_processor: Arc<QueryProcessor>,
        retrieval_pipeline: Arc<dyn RetrievalPipeline>,
    ) -> (Self, mpsc::UnboundedReceiver<MarketEvent>) {
        let (event_tx, event_rx) = mpsc::unbounded_channel();

        let bridge = Self {
            query_processor,
            retrieval_pipeline,
            event_ingestion_tx: event_tx,
            max_processing_time_ms: 50, // 50ms latency target for HFT
        };

        (bridge, event_rx)
    }

    async fn process_hft_query(&self, request: &HftRequest) -> Result<RetrievalContext> {
        let query = Query {
            text: request.query.clone(),
            filters: request.context.clone(),
            top_k: Some(self.get_top_k_for_priority(&request.priority)),
            threshold: Some(self.get_threshold_for_request_type(&request.request_type)),
            context_window: Some(self.get_context_window_for_request_type(&request.request_type)),
        };

        let enhanced_query = self.query_processor.process_query(query).await?;
        self.retrieval_pipeline.retrieve(enhanced_query).await
    }

    fn get_top_k_for_priority(&self, priority: &Priority) -> usize {
        match priority {
            Priority::Critical => 5,  // Fewer docs for faster processing
            Priority::High => 10,
            Priority::Normal => 15,
            Priority::Low => 20,
        }
    }

    fn get_threshold_for_request_type(&self, request_type: &HftRequestType) -> f32 {
        match request_type {
            HftRequestType::StrategyAugmentation => 0.8, // High precision required
            HftRequestType::RiskAssessment => 0.85,      // Very high precision
            HftRequestType::MarketAnalysis => 0.7,       // Balanced
            HftRequestType::ContextualSearch => 0.6,     // More recall
            HftRequestType::RealTimeQuery => 0.75,       // Fast but accurate
        }
    }

    fn get_context_window_for_request_type(&self, request_type: &HftRequestType) -> usize {
        match request_type {
            HftRequestType::StrategyAugmentation => 2000, // Compact context
            HftRequestType::RiskAssessment => 1500,       // Focus on key info
            HftRequestType::MarketAnalysis => 3000,       // More comprehensive
            HftRequestType::ContextualSearch => 4000,     // Full context
            HftRequestType::RealTimeQuery => 1000,        // Minimal for speed
        }
    }

    async fn _check_processing_timeout(&self, start_time: std::time::Instant) -> bool {
        start_time.elapsed().as_millis() as u64 > self.max_processing_time_ms
    }
}

#[async_trait]
impl HftIntegration for McpBridge {
    async fn process_request(&self, request: HftRequest) -> Result<HftResponse> {
        let start_time = std::time::Instant::now();

        // Check for timeout early for critical requests
        if matches!(request.priority, Priority::Critical) {
            tokio::select! {
                result = self.process_hft_query(&request) => {
                    let processing_time = start_time.elapsed().as_millis() as u64;
                    
                    match result {
                        Ok(context) => Ok(HftResponse {
                            request_id: request.request_id,
                            context,
                            processing_time_ms: processing_time,
                            status: ResponseStatus::Success,
                        }),
                        Err(e) => Ok(HftResponse {
                            request_id: request.request_id.clone(),
                            context: RetrievalContext {
                                results: vec![],
                                total_score: 0.0,
                                query_metadata: serde_json::json!({"error": e.to_string()}),
                            },
                            processing_time_ms: processing_time,
                            status: ResponseStatus::Failed,
                        }),
                    }
                }
                _ = tokio::time::sleep(tokio::time::Duration::from_millis(self.max_processing_time_ms)) => {
                    Ok(HftResponse {
                        request_id: request.request_id,
                        context: RetrievalContext {
                            results: vec![],
                            total_score: 0.0,
                            query_metadata: serde_json::json!({"error": "Request timeout"}),
                        },
                        processing_time_ms: self.max_processing_time_ms,
                        status: ResponseStatus::Timeout,
                    })
                }
            }
        } else {
            // Normal processing for non-critical requests
            let context = self.process_hft_query(&request).await?;
            let processing_time = start_time.elapsed().as_millis() as u64;

            Ok(HftResponse {
                request_id: request.request_id,
                context,
                processing_time_ms: processing_time,
                status: ResponseStatus::Success,
            })
        }
    }

    async fn ingest_market_event(&self, event: MarketEvent) -> Result<()> {
        // Send event to ingestion pipeline asynchronously
        self.event_ingestion_tx
            .send(event)
            .map_err(|e| RagError::Integration(format!("Failed to queue market event: {}", e)))?;
        
        Ok(())
    }

    async fn health_check(&self) -> Result<bool> {
        // TODO: Implement comprehensive health checks
        // - Check vector storage connectivity
        // - Check embedding service availability  
        // - Check ingestion pipeline status
        // - Verify processing latency is within limits
        Ok(true)
    }
}

pub struct RealTimeEventProcessor {
    event_rx: mpsc::UnboundedReceiver<MarketEvent>,
    // TODO: Add ingestion pipeline reference
}

impl RealTimeEventProcessor {
    pub fn new(event_rx: mpsc::UnboundedReceiver<MarketEvent>) -> Self {
        Self { event_rx }
    }

    pub async fn start_processing(&mut self) -> Result<()> {
        while let Some(event) = self.event_rx.recv().await {
            if let Err(e) = self.process_event(event).await {
                tracing::error!("Failed to process market event: {}", e);
                // Continue processing other events
            }
        }
        Ok(())
    }

    async fn process_event(&self, event: MarketEvent) -> Result<()> {
        // TODO: Implement real-time event processing
        // 1. Convert market event to document
        // 2. Generate embedding
        // 3. Store in vector database
        // 4. Update indexes
        // Target: Complete processing within 10ms
        
        tracing::debug!("Processing market event: {:?} for {}", event.event_type, event.symbol);
        Ok(())
    }
}