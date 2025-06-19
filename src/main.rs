use hft_rag::{
    api::{create_router, AppState},
    config::RagConfig,
    embedding::CandleEmbedding,
    query::QueryProcessor,
    retrieval::{HybridRetrieval, RetrievalConfig},
    storage::QdrantStorage,
    integration::{McpBridge, RealTimeEventProcessor},
};
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing::{info, error};
use tracing_subscriber::{self, EnvFilter};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info"))
        )
        .init();

    info!("Starting HFT-RAG system...");

    // Load configuration
    let config = RagConfig::default();
    
    // Initialize components
    let embedding_service = Arc::new(CandleEmbedding::new().await?);
    let storage = Arc::new(QdrantStorage::new(&config.storage.qdrant_url, None).await?);
    let query_processor = Arc::new(QueryProcessor::new());
    
    let retrieval_config = RetrievalConfig {
        top_k: config.retrieval.top_k,
        similarity_threshold: config.retrieval.similarity_threshold,
        max_context_length: config.retrieval.max_context_length,
        ..Default::default()
    };
    
    let retrieval_pipeline = Arc::new(HybridRetrieval::new(
        storage.clone(),
        embedding_service.clone(),
        retrieval_config,
    ));

    // Create MCP bridge for HFT integration
    let (_mcp_bridge, event_rx) = McpBridge::new(
        query_processor.clone(),
        retrieval_pipeline.clone(),
    );
    
    // Start real-time event processor
    let mut event_processor = RealTimeEventProcessor::new(event_rx);
    tokio::spawn(async move {
        if let Err(e) = event_processor.start_processing().await {
            error!("Event processor failed: {}", e);
        }
    });

    // Create API router
    let app_state = AppState {
        query_processor,
        retrieval_pipeline,
    };
    
    let app = create_router(app_state);

    // Start server
    let addr = format!("{}:{}", config.server.host, config.server.port);
    let listener = TcpListener::bind(&addr).await?;
    
    info!("HFT-RAG server listening on {}", addr);
    info!("Health check: http://{}/health", addr);
    info!("Query endpoint: http://{}/query", addr);

    axum::serve(listener, app).await?;

    Ok(())
}
