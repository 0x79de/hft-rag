use axum::{
    extract::{Path, State},
    http::StatusCode,
    middleware,
    response::Json,
    routing::{delete, get, post, put},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, error};
use uuid::Uuid;
use crate::query::{Query, QueryProcessor};
use crate::retrieval::RetrievalPipeline;
use crate::storage::{Document, VectorStorage};
use crate::embedding::EmbeddingService;

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
    pub storage: Arc<dyn VectorStorage>,
    pub embedding_service: Arc<dyn EmbeddingService>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DocumentRequest {
    pub content: String,
    pub metadata: Option<HashMap<String, String>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DocumentCreateResponse {
    pub id: String,
    pub status: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BatchDocumentRequest {
    pub documents: Vec<DocumentRequest>,
}

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/status", get(system_status))
        .route("/query", post(query_documents))
        .route("/documents", post(create_document))
        .route("/documents/batch", post(create_documents_batch))
        .route("/documents/:id", get(get_document))
        .route("/documents/:id", put(update_document))
        .route("/documents/:id", delete(delete_document))
        .layer(middleware::from_fn(logging_middleware))
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
        .results
        .into_iter()
        .map(|result| DocumentResponse {
            id: result.document.id.to_string(),
            content: result.document.content,
            metadata: result.document.metadata,
            score: result.score,
            timestamp: result.document.timestamp,
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

async fn create_document(
    State(state): State<Arc<AppState>>,
    Json(request): Json<DocumentRequest>,
) -> std::result::Result<Json<DocumentCreateResponse>, (StatusCode, Json<ErrorResponse>)> {
    let id = Uuid::new_v4();
    
    // Generate embedding for the document
    let embedding = match state.embedding_service.embed_text(&request.content).await {
        Ok(emb) => emb,
        Err(e) => {
            error!("Failed to generate embedding: {}", e);
            return Err(create_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Embedding generation failed",
                &e.to_string(),
            ));
        }
    };
    
    let document = Document {
        id,
        content: request.content,
        metadata: request.metadata.unwrap_or_default(),
        embedding,
        timestamp: chrono::Utc::now(),
    };
    
    if let Err(e) = state.storage.insert_document(document).await {
        error!("Failed to store document: {}", e);
        return Err(create_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Document storage failed",
            &e.to_string(),
        ));
    }
    
    info!("Document created with ID: {}", id);
    Ok(Json(DocumentCreateResponse {
        id: id.to_string(),
        status: "created".to_string(),
    }))
}

async fn create_documents_batch(
    State(state): State<Arc<AppState>>,
    Json(request): Json<BatchDocumentRequest>,
) -> std::result::Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    if request.documents.is_empty() {
        return Err(create_error_response(
            StatusCode::BAD_REQUEST,
            "Empty batch",
            "Batch request must contain at least one document",
        ));
    }
    
    let mut documents = Vec::new();
    let mut created_ids = Vec::new();
    
    for doc_request in request.documents {
        let id = Uuid::new_v4();
        created_ids.push(id.to_string());
        
        // Generate embedding for each document
        let embedding = match state.embedding_service.embed_text(&doc_request.content).await {
            Ok(emb) => emb,
            Err(e) => {
                error!("Failed to generate embedding for document: {}", e);
                return Err(create_error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Batch embedding generation failed",
                    &e.to_string(),
                ));
            }
        };
        
        let document = Document {
            id,
            content: doc_request.content,
            metadata: doc_request.metadata.unwrap_or_default(),
            embedding,
            timestamp: chrono::Utc::now(),
        };
        
        documents.push(document);
    }
    
    if let Err(e) = state.storage.insert_batch(documents).await {
        error!("Failed to store document batch: {}", e);
        return Err(create_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Batch storage failed",
            &e.to_string(),
        ));
    }
    
    info!("Batch of {} documents created", created_ids.len());
    Ok(Json(serde_json::json!({
        "status": "created",
        "count": created_ids.len(),
        "ids": created_ids
    })))
}

async fn get_document(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> std::result::Result<Json<DocumentResponse>, (StatusCode, Json<ErrorResponse>)> {
    let uuid = match Uuid::parse_str(&id) {
        Ok(uuid) => uuid,
        Err(_) => {
            return Err(create_error_response(
                StatusCode::BAD_REQUEST,
                "Invalid ID format",
                "Document ID must be a valid UUID",
            ));
        }
    };
    
    match state.storage.get_document(uuid).await {
        Ok(Some(document)) => {
            Ok(Json(DocumentResponse {
                id: document.id.to_string(),
                content: document.content,
                metadata: document.metadata,
                score: 1.0, // Not applicable for direct retrieval
                timestamp: document.timestamp,
            }))
        }
        Ok(None) => {
            Err(create_error_response(
                StatusCode::NOT_FOUND,
                "Document not found",
                "No document found with the specified ID",
            ))
        }
        Err(e) => {
            error!("Failed to retrieve document: {}", e);
            Err(create_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Document retrieval failed",
                &e.to_string(),
            ))
        }
    }
}

async fn update_document(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(request): Json<DocumentRequest>,
) -> std::result::Result<Json<DocumentCreateResponse>, (StatusCode, Json<ErrorResponse>)> {
    let uuid = match Uuid::parse_str(&id) {
        Ok(uuid) => uuid,
        Err(_) => {
            return Err(create_error_response(
                StatusCode::BAD_REQUEST,
                "Invalid ID format",
                "Document ID must be a valid UUID",
            ));
        }
    };
    
    // Check if document exists
    if state.storage.get_document(uuid).await.map_err(|e| {
        error!("Failed to check document existence: {}", e);
        create_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Database error",
            &e.to_string(),
        )
    })?.is_none() {
        return Err(create_error_response(
            StatusCode::NOT_FOUND,
            "Document not found",
            "No document found with the specified ID",
        ));
    }
    
    // Generate new embedding for updated content
    let embedding = match state.embedding_service.embed_text(&request.content).await {
        Ok(emb) => emb,
        Err(e) => {
            error!("Failed to generate embedding: {}", e);
            return Err(create_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Embedding generation failed",
                &e.to_string(),
            ));
        }
    };
    
    let updated_document = Document {
        id: uuid,
        content: request.content,
        metadata: request.metadata.unwrap_or_default(),
        embedding,
        timestamp: chrono::Utc::now(),
    };
    
    if let Err(e) = state.storage.insert_document(updated_document).await {
        error!("Failed to update document: {}", e);
        return Err(create_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Document update failed",
            &e.to_string(),
        ));
    }
    
    info!("Document updated with ID: {}", uuid);
    Ok(Json(DocumentCreateResponse {
        id: uuid.to_string(),
        status: "updated".to_string(),
    }))
}

async fn delete_document(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> std::result::Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let uuid = match Uuid::parse_str(&id) {
        Ok(uuid) => uuid,
        Err(_) => {
            return Err(create_error_response(
                StatusCode::BAD_REQUEST,
                "Invalid ID format",
                "Document ID must be a valid UUID",
            ));
        }
    };
    
    // Check if document exists
    if state.storage.get_document(uuid).await.map_err(|e| {
        error!("Failed to check document existence: {}", e);
        create_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Database error",
            &e.to_string(),
        )
    })?.is_none() {
        return Err(create_error_response(
            StatusCode::NOT_FOUND,
            "Document not found",
            "No document found with the specified ID",
        ));
    }
    
    if let Err(e) = state.storage.delete_document(uuid).await {
        error!("Failed to delete document: {}", e);
        return Err(create_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Document deletion failed",
            &e.to_string(),
        ));
    }
    
    info!("Document deleted with ID: {}", uuid);
    Ok(Json(serde_json::json!({
        "status": "deleted",
        "id": uuid.to_string()
    })))
}