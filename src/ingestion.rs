use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use regex::Regex;
use csv::ReaderBuilder;
use crate::{Result, RagError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub content: String,
    pub metadata: HashMap<String, String>,
    pub chunk_index: usize,
    pub source_id: String,
}

#[async_trait]
pub trait DocumentParser: Send + Sync {
    async fn parse(&self, file_path: &Path) -> Result<Vec<DocumentChunk>>;
    fn supported_extensions(&self) -> &[&str];
}

pub struct TextParser;

#[async_trait]
impl DocumentParser for TextParser {
    async fn parse(&self, file_path: &Path) -> Result<Vec<DocumentChunk>> {
        let content = tokio::fs::read_to_string(file_path).await?;
        let mut metadata = HashMap::new();
        metadata.insert("file_name".to_string(), file_path.file_name().unwrap().to_string_lossy().to_string());
        metadata.insert("file_type".to_string(), "text".to_string());
        
        Ok(vec![DocumentChunk {
            content,
            metadata,
            chunk_index: 0,
            source_id: file_path.to_string_lossy().to_string(),
        }])
    }

    fn supported_extensions(&self) -> &[&str] {
        &["txt", "md"]
    }
}

pub struct JsonParser;

#[async_trait]
impl DocumentParser for JsonParser {
    async fn parse(&self, file_path: &Path) -> Result<Vec<DocumentChunk>> {
        let content = tokio::fs::read_to_string(file_path).await?;
        let json_value: serde_json::Value = serde_json::from_str(&content)?;
        
        let mut metadata = HashMap::new();
        metadata.insert("file_name".to_string(), file_path.file_name().unwrap().to_string_lossy().to_string());
        metadata.insert("file_type".to_string(), "json".to_string());

        // Extract relevant fields for HFT data
        if let Some(timestamp) = json_value.get("timestamp") {
            metadata.insert("timestamp".to_string(), timestamp.to_string());
        }
        if let Some(symbol) = json_value.get("symbol") {
            metadata.insert("symbol".to_string(), symbol.as_str().unwrap_or("").to_string());
        }

        Ok(vec![DocumentChunk {
            content: json_value.to_string(),
            metadata,
            chunk_index: 0,
            source_id: file_path.to_string_lossy().to_string(),
        }])
    }

    fn supported_extensions(&self) -> &[&str] {
        &["json"]
    }
}

pub struct CsvParser {
    financial_symbols: Regex,
}

impl Default for CsvParser {
    fn default() -> Self {
        Self {
            financial_symbols: Regex::new(r"\b[A-Z]{3,5}\b").unwrap(),
        }
    }
}

#[async_trait]
impl DocumentParser for CsvParser {
    async fn parse(&self, file_path: &Path) -> Result<Vec<DocumentChunk>> {
        let content = tokio::fs::read_to_string(file_path).await?;
        let mut reader = ReaderBuilder::new().from_reader(content.as_bytes());
        let headers = reader.headers()?.clone();
        
        let mut chunks = Vec::new();
        let mut chunk_index = 0;
        
        // Process CSV in batches for better chunking
        let mut current_batch = Vec::new();
        let batch_size = 100; // rows per chunk
        
        for result in reader.records() {
            let record = result?;
            current_batch.push(record);
            
            if current_batch.len() >= batch_size {
                let chunk = self.create_csv_chunk(&headers, &current_batch, file_path, chunk_index)?;
                chunks.push(chunk);
                current_batch.clear();
                chunk_index += 1;
            }
        }
        
        // Process remaining records
        if !current_batch.is_empty() {
            let chunk = self.create_csv_chunk(&headers, &current_batch, file_path, chunk_index)?;
            chunks.push(chunk);
        }
        
        Ok(chunks)
    }

    fn supported_extensions(&self) -> &[&str] {
        &["csv"]
    }
}

impl CsvParser {
    fn create_csv_chunk(&self, headers: &csv::StringRecord, records: &[csv::StringRecord], file_path: &Path, chunk_index: usize) -> Result<DocumentChunk> {
        let mut metadata = HashMap::new();
        metadata.insert("file_name".to_string(), file_path.file_name().unwrap().to_string_lossy().to_string());
        metadata.insert("file_type".to_string(), "csv".to_string());
        metadata.insert("chunk_type".to_string(), "market_data".to_string());
        
        // Extract financial metadata
        let mut symbols = Vec::new();
        let mut timestamps = Vec::new();
        
        for record in records {
            // Look for timestamp columns
            for (i, header) in headers.iter().enumerate() {
                if header.to_lowercase().contains("time") || header.to_lowercase().contains("date") {
                    if let Some(value) = record.get(i) {
                        timestamps.push(value.to_string());
                    }
                }
                
                // Look for symbol columns
                if header.to_lowercase().contains("symbol") || header.to_lowercase().contains("ticker") {
                    if let Some(value) = record.get(i) {
                        symbols.push(value.to_string());
                    }
                } else if let Some(value) = record.get(i) {
                    // Extract symbols using regex
                    for cap in self.financial_symbols.captures_iter(value) {
                        symbols.push(cap[0].to_string());
                    }
                }
            }
        }
        
        if !symbols.is_empty() {
            metadata.insert("symbols".to_string(), symbols.join(","));
        }
        
        if !timestamps.is_empty() {
            metadata.insert("timestamp_range".to_string(), format!("{} to {}", timestamps.first().unwrap_or(&"".to_string()), timestamps.last().unwrap_or(&"".to_string())));
        }
        
        // Create readable content from CSV data
        let mut content = format!("Headers: {}\n\n", headers.iter().collect::<Vec<_>>().join(", "));
        for record in records {
            content.push_str(&format!("{}\n", record.iter().collect::<Vec<_>>().join(", ")));
        }
        
        Ok(DocumentChunk {
            content,
            metadata,
            chunk_index,
            source_id: format!("{}-{}", file_path.to_string_lossy(), chunk_index),
        })
    }
}

// PDF Parser for financial documents
pub struct PdfParser;

#[async_trait]
impl DocumentParser for PdfParser {
    async fn parse(&self, file_path: &Path) -> Result<Vec<DocumentChunk>> {
        let content = pdf_extract::extract_text(file_path)
            .map_err(|e| RagError::Ingestion(format!("PDF extraction failed: {}", e)))?;
        
        let mut metadata = HashMap::new();
        metadata.insert("file_name".to_string(), file_path.file_name().unwrap().to_string_lossy().to_string());
        metadata.insert("file_type".to_string(), "pdf".to_string());
        
        // Extract financial metadata from PDF content
        let financial_regex = Regex::new(r"\b[A-Z]{3,5}\b").unwrap();
        let date_regex = Regex::new(r"\d{4}-\d{2}-\d{2}").unwrap();
        
        let symbols: Vec<String> = financial_regex.find_iter(&content)
            .map(|m| m.as_str().to_string())
            .collect();
        
        let dates: Vec<String> = date_regex.find_iter(&content)
            .map(|m| m.as_str().to_string())
            .collect();
        
        if !symbols.is_empty() {
            metadata.insert("symbols".to_string(), symbols.join(","));
        }
        
        if !dates.is_empty() {
            metadata.insert("dates".to_string(), dates.join(","));
        }
        
        Ok(vec![DocumentChunk {
            content,
            metadata,
            chunk_index: 0,
            source_id: file_path.to_string_lossy().to_string(),
        }])
    }

    fn supported_extensions(&self) -> &[&str] {
        &["pdf"]
    }
}

pub struct IngestionPipeline {
    parsers: Vec<Box<dyn DocumentParser>>,
    chunk_size: usize,
    chunk_overlap: usize,
}

impl IngestionPipeline {
    pub fn new() -> Self {
        Self {
            parsers: vec![
                Box::new(TextParser),
                Box::new(JsonParser),
                Box::new(CsvParser::default()),
                Box::new(PdfParser),
            ],
            chunk_size: 512,
            chunk_overlap: 64,
        }
    }
    
    pub fn with_chunk_config(chunk_size: usize, chunk_overlap: usize) -> Self {
        let mut pipeline = Self::new();
        pipeline.chunk_size = chunk_size;
        pipeline.chunk_overlap = chunk_overlap;
        pipeline
    }

    pub async fn process_file(&self, file_path: &Path) -> Result<Vec<DocumentChunk>> {
        let extension = file_path
            .extension()
            .and_then(|ext| ext.to_str())
            .ok_or_else(|| RagError::Ingestion("No file extension".to_string()))?;

        for parser in &self.parsers {
            if parser.supported_extensions().contains(&extension) {
                let chunks = parser.parse(file_path).await?;
                return self.chunk_documents(chunks);
            }
        }

        Err(RagError::Ingestion(format!("Unsupported file type: {}", extension)))
    }

    fn chunk_documents(&self, chunks: Vec<DocumentChunk>) -> Result<Vec<DocumentChunk>> {
        let mut result = Vec::new();
        
        for chunk in chunks {
            if chunk.content.len() <= self.chunk_size {
                result.push(chunk);
            } else {
                // Advanced chunking for financial documents
                result.extend(self.smart_chunk(&chunk)?);
            }
        }
        
        Ok(result)
    }
    
    fn smart_chunk(&self, chunk: &DocumentChunk) -> Result<Vec<DocumentChunk>> {
        let mut result = Vec::new();
        let content = &chunk.content;
        
        // For financial data, prefer sentence-based chunking
        let sentences: Vec<&str> = content.split(&['.', '!', '?', '\n']).collect();
        let mut current_chunk = String::new();
        let mut chunk_index = 0;
        
        for sentence in sentences {
            let trimmed = sentence.trim();
            if trimmed.is_empty() {
                continue;
            }
            
            if current_chunk.len() + trimmed.len() + 1 > self.chunk_size {
                if !current_chunk.is_empty() {
                    result.push(self.create_chunk_from_content(
                        &current_chunk,
                        chunk,
                        chunk_index,
                    ));
                    
                    // Handle overlap
                    if self.chunk_overlap > 0 {
                        let overlap_text = self.get_overlap_text(&current_chunk);
                        current_chunk = overlap_text + " " + trimmed;
                    } else {
                        current_chunk = trimmed.to_string();
                    }
                    
                    chunk_index += 1;
                } else {
                    current_chunk = trimmed.to_string();
                }
            } else {
                if !current_chunk.is_empty() {
                    current_chunk.push(' ');
                }
                current_chunk.push_str(trimmed);
            }
        }
        
        // Add final chunk
        if !current_chunk.is_empty() {
            result.push(self.create_chunk_from_content(
                &current_chunk,
                chunk,
                chunk_index,
            ));
        }
        
        Ok(result)
    }
    
    fn create_chunk_from_content(&self, content: &str, original: &DocumentChunk, index: usize) -> DocumentChunk {
        let mut metadata = original.metadata.clone();
        metadata.insert("parent_chunk_id".to_string(), original.source_id.clone());
        metadata.insert("chunk_size".to_string(), content.len().to_string());
        
        DocumentChunk {
            content: content.to_string(),
            metadata,
            chunk_index: index,
            source_id: format!("{}-{}", original.source_id, index),
        }
    }
    
    fn get_overlap_text(&self, text: &str) -> String {
        let words: Vec<&str> = text.split_whitespace().collect();
        let overlap_words = (words.len() * self.chunk_overlap / self.chunk_size).min(words.len());
        
        if overlap_words > 0 {
            words[words.len() - overlap_words..].join(" ")
        } else {
            String::new()
        }
    }
    
    pub async fn process_directory(&self, dir_path: &Path) -> Result<Vec<DocumentChunk>> {
        let mut all_chunks = Vec::new();
        let mut entries = tokio::fs::read_dir(dir_path).await?;
        
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.is_file() {
                match self.process_file(&path).await {
                    Ok(mut chunks) => all_chunks.append(&mut chunks),
                    Err(e) => tracing::warn!("Failed to process file {:?}: {}", path, e),
                }
            }
        }
        
        Ok(all_chunks)
    }
}