use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::collections::HashMap;
use crate::Result;
use qdrant_client::{
    Qdrant,
    qdrant::{
        CreateCollection, VectorParams, Distance, 
        PointStruct, SearchPoints, Filter, Condition, FieldCondition, Match,
        ScoredPoint, UpsertPoints, DeletePoints, PointsSelector, PointId,
        Value, RetrievedPoint, GetPoints, WithPayloadSelector,
    },
};
use std::sync::Arc;

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

pub struct QdrantStorage {
    client: Arc<Qdrant>,
    collection_name: String,
}

impl QdrantStorage {
    pub async fn new(url: &str, api_key: Option<&str>) -> Result<Self> {
        let client = if let Some(key) = api_key {
            Qdrant::from_url(url).api_key(key).build()?
        } else {
            Qdrant::from_url(url).build()?
        };

        let collection_name = "hft_documents".to_string();
        let storage = Self {
            client: Arc::new(client),
            collection_name,
        };

        // Initialize collection if it doesn't exist
        storage.ensure_collection_exists().await?;
        
        Ok(storage)
    }

    async fn ensure_collection_exists(&self) -> Result<()> {
        let collections = self.client.list_collections().await?;
        
        let collection_exists = collections
            .collections
            .iter()
            .any(|c| c.name == self.collection_name);

        if !collection_exists {
            tracing::info!("Creating Qdrant collection '{}'", self.collection_name);
            
            let create_collection = CreateCollection {
                collection_name: self.collection_name.clone(),
                vectors_config: Some(VectorParams {
                    size: 384, // Standard embedding dimension
                    distance: Distance::Cosine.into(),
                    ..Default::default()
                }.into()),
                ..Default::default()
            };

            self.client.create_collection(create_collection).await?;
            tracing::info!("Successfully created collection '{}'", self.collection_name);
        }

        Ok(())
    }
    
    fn document_to_point(&self, document: &Document) -> PointStruct {
        let mut payload = HashMap::new();
        
        // Add document fields to payload
        payload.insert("content".to_string(), Value::from(document.content.clone()));
        payload.insert("timestamp".to_string(), Value::from(document.timestamp.to_rfc3339()));
        
        // Add metadata
        for (key, value) in &document.metadata {
            payload.insert(key.clone(), Value::from(value.clone()));
        }
        
        PointStruct::new(
            document.id.to_string(),
            document.embedding.clone(),
            payload,
        )
    }
    
    fn point_to_document(&self, point: &ScoredPoint) -> Option<Document> {
        let id_str = match &point.id.as_ref()?.point_id_options {
            Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(uuid)) => uuid.clone(),
            Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(num)) => num.to_string(),
            _ => return None,
        };
        let id = Uuid::parse_str(&id_str).ok()?;
        
        let payload = &point.payload;
        let content = payload.get("content")?.as_str()?.to_string();
        let timestamp_str = payload.get("timestamp")?.as_str()?;
        let timestamp = chrono::DateTime::parse_from_rfc3339(timestamp_str)
            .ok()?
            .with_timezone(&chrono::Utc);
        
        let mut metadata = HashMap::new();
        for (key, value) in payload {
            if key != "content" && key != "timestamp" {
                if let Some(str_val) = value.as_str() {
                    metadata.insert(key.clone(), str_val.to_string());
                }
            }
        }
        
        // Extract embedding from vectors
        let embedding = match &point.vectors.as_ref()?.vectors_options {
            Some(qdrant_client::qdrant::vectors_output::VectorsOptions::Vector(vector)) => {
                vector.data.clone()
            },
            _ => return None,
        };
        
        Some(Document {
            id,
            content,
            metadata,
            embedding,
            timestamp,
        })
    }

    fn retrieved_point_to_document(&self, point: &RetrievedPoint) -> Option<Document> {
        let id_str = match &point.id.as_ref()?.point_id_options {
            Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(uuid)) => uuid.clone(),
            Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(num)) => num.to_string(),
            _ => return None,
        };
        let id = Uuid::parse_str(&id_str).ok()?;
        
        let payload = &point.payload;
        let content = payload.get("content")?.as_str()?.to_string();
        let timestamp_str = payload.get("timestamp")?.as_str()?;
        let timestamp = chrono::DateTime::parse_from_rfc3339(timestamp_str)
            .ok()?
            .with_timezone(&chrono::Utc);
        
        let mut metadata = HashMap::new();
        for (key, value) in payload {
            if key != "content" && key != "timestamp" {
                if let Some(str_val) = value.as_str() {
                    metadata.insert(key.clone(), str_val.to_string());
                }
            }
        }
        
        // Extract embedding from vectors
        let embedding = match &point.vectors.as_ref()?.vectors_options {
            Some(qdrant_client::qdrant::vectors_output::VectorsOptions::Vector(vector)) => {
                vector.data.clone()
            },
            _ => return None,
        };
        
        Some(Document {
            id,
            content,
            metadata,
            embedding,
            timestamp,
        })
    }
}

#[async_trait]
impl VectorStorage for QdrantStorage {
    async fn insert_document(&self, document: Document) -> Result<()> {
        let point = self.document_to_point(&document);
        
        let upsert_points = UpsertPoints {
            collection_name: self.collection_name.clone(),
            points: vec![point],
            ..Default::default()
        };
        
        self.client.upsert_points(upsert_points).await?;
        Ok(())
    }

    async fn insert_batch(&self, documents: Vec<Document>) -> Result<()> {
        if documents.is_empty() {
            return Ok(());
        }
        
        let points: Vec<PointStruct> = documents
            .iter()
            .map(|doc| self.document_to_point(doc))
            .collect();
        
        let upsert_points = UpsertPoints {
            collection_name: self.collection_name.clone(),
            points,
            ..Default::default()
        };
        
        self.client.upsert_points(upsert_points).await?;
        Ok(())
    }

    async fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>> {
        let mut filter = None;
        
        // Build filter from query filters
        if let Some(ref filters) = query.filters {
            let mut conditions = Vec::new();
            
            for (key, value) in filters {
                conditions.push(Condition {
                    condition_one_of: Some(FieldCondition {
                        key: key.clone(),
                        r#match: Some(Match {
                            match_value: Some(value.clone().into()),
                        }),
                        ..Default::default()
                    }.into()),
                });
            }
            
            if !conditions.is_empty() {
                filter = Some(Filter {
                    must: conditions,
                    ..Default::default()
                });
            }
        }
        
        let search_points = SearchPoints {
            collection_name: self.collection_name.clone(),
            vector: query.embedding,
            limit: query.top_k as u64,
            score_threshold: Some(query.threshold),
            filter,
            with_payload: Some(true.into()),
            with_vectors: Some(true.into()),
            ..Default::default()
        };
        
        let search_result = self.client.search_points(search_points).await?;
        
        let mut results = Vec::new();
        for scored_point in search_result.result {
            if let Some(document) = self.point_to_document(&scored_point) {
                results.push(SearchResult {
                    document,
                    score: scored_point.score,
                });
            }
        }
        
        Ok(results)
    }

    async fn delete_document(&self, id: Uuid) -> Result<()> {
        let delete_points = DeletePoints {
            collection_name: self.collection_name.clone(),
            points: Some(PointsSelector {
                points_selector_one_of: Some(vec![PointId {
                    point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(id.to_string())),
                }].into()),
            }),
            ..Default::default()
        };
        
        self.client.delete_points(delete_points).await?;
        Ok(())
    }

    async fn get_document(&self, id: Uuid) -> Result<Option<Document>> {
        let get_request = GetPoints {
            collection_name: self.collection_name.clone(),
            ids: vec![PointId {
                point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(id.to_string())),
            }],
            with_payload: Some(WithPayloadSelector {
                selector_options: Some(true.into()),
            }),
            with_vectors: Some(true.into()),
            ..Default::default()
        };
        
        let result = self.client.get_points(get_request).await?;
        
        if let Some(point) = result.result.first() {
            Ok(self.retrieved_point_to_document(point))
        } else {
            Ok(None)
        }
    }
}