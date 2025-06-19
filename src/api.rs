use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, error};
use crate::query::{Query, QueryProcessor};
use crate::retrieval::RetrievalPipeline;

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub version: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QueryRequest {
    pub query: String,
    pub filters: Option<HashMap<String, String>>,
    pub top_k: Option<usize>,
    pub threshold: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QueryResponse {
    pub query: String,
    pub documents: Vec<DocumentResponse>,
    pub metadata: serde_json::Value,
    pub processing_time_ms: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DocumentResponse {
    pub id: String,
    pub content: String,
    pub metadata: HashMap<String, String>,
    pub score: f32,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
    pub code: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

pub struct AppState {
    pub query_processor: Arc<QueryProcessor>,
    pub retrieval_pipeline: Arc<dyn RetrievalPipeline>,
}

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/query", post(query_documents))
        .route("/status", get(system_status))
        .with_state(Arc::new(state))
}

async fn health_check() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        timestamp: chrono::Utc::now(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

async fn system_status(State(_state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "operational",
        "timestamp": chrono::Utc::now(),
        "components": {
            "query_processor": "healthy",
            "retrieval_pipeline": "healthy",
            "vector_storage": "healthy"
        }
    }))
}

async fn query_documents(
    State(state): State<Arc<AppState>>,
    Json(request): Json<QueryRequest>,
) -> std::result::Result<Json<QueryResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start_time = std::time::Instant::now();
    
    info!("Processing query: {}", request.query);

    // Process the query
    let query = Query {
        text: request.query.clone(),
        filters: request.filters,
        top_k: request.top_k,
        threshold: request.threshold,
        context_window: None,
    };

    let enhanced_query = match state.query_processor.process_query(query).await {
        Ok(query) => query,
        Err(e) => {
            error!("Query processing failed: {}", e);
            return Err(create_error_response(
                StatusCode::BAD_REQUEST,
                "Query processing failed",
                &e.to_string(),
            ));
        }
    };

    // Retrieve relevant documents
    let retrieval_context = match state.retrieval_pipeline.retrieve(enhanced_query).await {
        Ok(context) => context,
        Err(e) => {
            error!("Document retrieval failed: {}", e);
            return Err(create_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Document retrieval failed",
                &e.to_string(),
            ));
        }
    };

    let processing_time = start_time.elapsed().as_millis() as u64;

    // Convert to response format
    let documents: Vec<DocumentResponse> = retrieval_context
        .documents
        .into_iter()
        .map(|doc| DocumentResponse {
            id: doc.id.to_string(),
            content: doc.content,
            metadata: doc.metadata,
            score: 0.0, // TODO: Add score from retrieval
            timestamp: doc.timestamp,
        })
        .collect();

    let response = QueryResponse {
        query: request.query,
        documents,
        metadata: retrieval_context.query_metadata,
        processing_time_ms: processing_time,
    };

    info!("Query processed in {}ms", processing_time);
    Ok(Json(response))
}

fn create_error_response(
    status: StatusCode,
    code: &str,
    message: &str,
) -> (StatusCode, Json<ErrorResponse>) {
    (
        status,
        Json(ErrorResponse {
            error: message.to_string(),
            code: code.to_string(),
            timestamp: chrono::Utc::now(),
        }),
    )
}

// Middleware for request logging and metrics
pub async fn logging_middleware(
    request: axum::http::Request<axum::body::Body>,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let method = request.method().clone();
    let uri = request.uri().clone();
    let start = std::time::Instant::now();

    let response = next.run(request).await;

    let latency = start.elapsed();
    let status = response.status();

    info!(
        method = %method,
        uri = %uri,
        status = %status,
        latency_ms = %latency.as_millis(),
        "Request processed"
    );

    response
}