#[cfg(test)]
mod unit_tests {

    mod query_tests {
        use crate::query::{QueryProcessor, Query, QueryIntent};

        #[tokio::test]
        async fn test_query_processor_intent_classification() {
            let processor = QueryProcessor::new();
            
            let market_query = Query {
                text: "What is the current price of AAPL?".to_string(),
                filters: None,
                top_k: None,
                threshold: None,
                context_window: None,
            };
            
            let result = processor.process_query(market_query).await.unwrap();
            assert!(matches!(result.intent, QueryIntent::MarketData));
        }

        #[tokio::test]
        async fn test_financial_term_expansion() {
            let processor = QueryProcessor::new();
            
            let query = Query {
                text: "Show me vol and pnl data".to_string(),
                filters: None,
                top_k: None,
                threshold: None,
                context_window: None,
            };
            
            let result = processor.process_query(query).await.unwrap();
            assert!(result.expanded_text.contains("volatility"));
            assert!(result.expanded_text.contains("profit and loss"));
        }

        #[tokio::test]
        async fn test_symbol_extraction() {
            let processor = QueryProcessor::new();
            
            let query = Query {
                text: "Compare AAPL and MSFT performance".to_string(),
                filters: None,
                top_k: None,
                threshold: None,
                context_window: None,
            };
            
            let result = processor.process_query(query).await.unwrap();
            assert!(result.market_filters.len() >= 2);
            assert!(result.market_filters.iter().any(|f| f.symbol.as_ref().is_some_and(|s| s == "AAPL")));
            assert!(result.market_filters.iter().any(|f| f.symbol.as_ref().is_some_and(|s| s == "MSFT")));
        }
    }

    mod config_tests {
        use crate::config::{RagConfig, EmbeddingProvider};

        #[test]
        fn test_default_config() {
            let config = RagConfig::default();
            assert_eq!(config.server.host, "0.0.0.0");
            assert_eq!(config.server.port, 8080);
            assert!(matches!(config.embedding.provider, EmbeddingProvider::Candle));
        }

        #[test]
        fn test_config_basic_values() {
            let config = RagConfig::default();
            assert!(config.embedding.batch_size > 0);
            assert!(config.retrieval.top_k > 0);
            assert!(config.retrieval.similarity_threshold >= 0.0);
            assert!(config.retrieval.similarity_threshold <= 1.0);
        }
    }

    mod embedding_tests {
        use crate::embedding::{OpenAIEmbedding, EmbeddingService};

        #[tokio::test]
        async fn test_openai_embedding_dimension() {
            let embedding = OpenAIEmbedding::new("test-key".to_string());
            assert_eq!(embedding.dimension(), 1536);
        }

        #[tokio::test]
        async fn test_embedding_with_custom_model() {
            let embedding = OpenAIEmbedding::with_model("test-key".to_string(), "custom-model".to_string(), 768);
            assert_eq!(embedding.dimension(), 768);
        }
    }

    mod storage_tests {
        use crate::storage::{Document, SearchQuery};
        use uuid::Uuid;
        use std::collections::HashMap;

        #[test]
        fn test_document_creation() {
            let doc = Document {
                id: Uuid::new_v4(),
                content: "Test content".to_string(),
                metadata: HashMap::new(),
                embedding: vec![0.1, 0.2, 0.3],
                timestamp: chrono::Utc::now(),
            };
            
            assert_eq!(doc.content, "Test content");
            assert_eq!(doc.embedding.len(), 3);
        }

        #[test]
        fn test_search_query() {
            let query = SearchQuery {
                embedding: vec![0.1, 0.2, 0.3],
                top_k: 10,
                threshold: 0.7,
                filters: None,
            };
            
            assert_eq!(query.top_k, 10);
            assert_eq!(query.threshold, 0.7);
        }
    }

    mod ingestion_tests {
        use crate::ingestion::{TextParser, DocumentParser, IngestionPipeline};
        use std::fs;
        use tempfile::tempdir;

        #[tokio::test]
        async fn test_text_parser() {
            let dir = tempdir().unwrap();
            let file_path = dir.path().join("test.txt");
            fs::write(&file_path, "This is test content").unwrap();
            
            let parser = TextParser;
            let chunks = parser.parse(&file_path).await.unwrap();
            
            assert_eq!(chunks.len(), 1);
            assert_eq!(chunks[0].content, "This is test content");
            assert_eq!(chunks[0].metadata.get("file_type").unwrap(), "text");
        }

        #[tokio::test]
        async fn test_json_parser() {
            let dir = tempdir().unwrap();
            let file_path = dir.path().join("test.json");
            let json_data = r#"{"symbol": "AAPL", "price": 150.0, "timestamp": "2024-01-01T00:00:00Z"}"#;
            fs::write(&file_path, json_data).unwrap();
            
            let parser = crate::ingestion::JsonParser;
            let chunks = parser.parse(&file_path).await.unwrap();
            
            assert_eq!(chunks.len(), 1);
            assert!(chunks[0].content.contains("AAPL"));
            assert_eq!(chunks[0].metadata.get("symbol").unwrap(), "AAPL");
        }

        #[test]
        fn test_ingestion_pipeline_creation() {
            let _pipeline = IngestionPipeline::new();
            let _pipeline_custom = IngestionPipeline::with_chunk_config(1024, 128);
            
            // Just test that they can be created successfully
            // Test passes if no panic occurs during validation
        }
    }

    mod api_tests {
        use crate::api::{QueryRequest, DocumentRequest};
        use std::collections::HashMap;

        #[test]
        fn test_query_request_serialization() {
            let request = QueryRequest {
                query: "test query".to_string(),
                filters: None,
                top_k: Some(10),
                threshold: Some(0.8),
            };
            
            let json = serde_json::to_string(&request).unwrap();
            let deserialized: QueryRequest = serde_json::from_str(&json).unwrap();
            
            assert_eq!(deserialized.query, "test query");
            assert_eq!(deserialized.top_k, Some(10));
            assert_eq!(deserialized.threshold, Some(0.8));
        }

        #[test]
        fn test_document_request() {
            let mut metadata = HashMap::new();
            metadata.insert("type".to_string(), "financial".to_string());
            
            let request = DocumentRequest {
                content: "Financial document content".to_string(),
                metadata: Some(metadata),
            };
            
            assert_eq!(request.content, "Financial document content");
            assert!(request.metadata.is_some());
        }
    }

    mod integration_tests {
        use crate::integration::HftRequestType;

        #[test]
        fn test_request_type_variants() {
            // Test that all variants exist and can be matched
            let variants = [
                HftRequestType::StrategyAugmentation,
                HftRequestType::RiskAssessment,
                HftRequestType::MarketAnalysis,
                HftRequestType::ContextualSearch,
                HftRequestType::RealTimeQuery,
            ];
            
            assert_eq!(variants.len(), 5);
        }
    }
}