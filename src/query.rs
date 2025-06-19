use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Query {
    pub text: String,
    pub filters: Option<HashMap<String, String>>,
    pub top_k: Option<usize>,
    pub threshold: Option<f32>,
    pub context_window: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedQuery {
    pub original_text: String,
    pub expanded_text: String,
    pub intent: QueryIntent,
    pub temporal_context: Option<TemporalContext>,
    pub market_filters: Vec<MarketFilter>,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryIntent {
    MarketData,
    TradingStrategy,
    RiskAssessment,
    HistoricalAnalysis,
    RealTimeQuery,
    GeneralInformation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalContext {
    pub time_range: Option<(chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>)>,
    pub relative_time: Option<String>, // "last hour", "today", etc.
    pub market_session: Option<String>, // "pre-market", "trading", "after-hours"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketFilter {
    pub symbol: Option<String>,
    pub asset_class: Option<String>,
    pub market_condition: Option<String>,
    pub volatility_range: Option<(f32, f32)>,
}

pub struct QueryProcessor {
    financial_terms: HashMap<String, Vec<String>>,
}

impl QueryProcessor {
    pub fn new() -> Self {
        let mut financial_terms = HashMap::new();
        
        // Add common financial term expansions
        financial_terms.insert("vol".to_string(), vec!["volatility".to_string(), "volume".to_string()]);
        financial_terms.insert("pnl".to_string(), vec!["profit and loss".to_string(), "p&l".to_string()]);
        financial_terms.insert("vwap".to_string(), vec!["volume weighted average price".to_string()]);
        financial_terms.insert("rsi".to_string(), vec!["relative strength index".to_string()]);
        financial_terms.insert("macd".to_string(), vec!["moving average convergence divergence".to_string()]);
        
        Self { financial_terms }
    }

    pub async fn process_query(&self, query: Query) -> Result<EnhancedQuery> {
        let expanded_text = self.expand_financial_terms(&query.text);
        let intent = self.classify_intent(&query.text);
        let temporal_context = self.extract_temporal_context(&query.text);
        let market_filters = self.extract_market_filters(&query.text, &query.filters);

        // TODO: Generate embedding for the expanded query
        let embedding = vec![0.0; 384]; // Placeholder

        Ok(EnhancedQuery {
            original_text: query.text,
            expanded_text,
            intent,
            temporal_context,
            market_filters,
            embedding,
        })
    }

    fn expand_financial_terms(&self, text: &str) -> String {
        let mut expanded = text.to_lowercase();
        
        for (abbrev, expansions) in &self.financial_terms {
            if expanded.contains(abbrev) {
                // Add the first expansion to the text
                if let Some(expansion) = expansions.first() {
                    expanded = expanded.replace(abbrev, &format!("{} {}", abbrev, expansion));
                }
            }
        }
        
        expanded
    }

    fn classify_intent(&self, text: &str) -> QueryIntent {
        let text_lower = text.to_lowercase();
        
        if text_lower.contains("market data") || text_lower.contains("price") || text_lower.contains("quote") {
            QueryIntent::MarketData
        } else if text_lower.contains("strategy") || text_lower.contains("trading") || text_lower.contains("algorithm") {
            QueryIntent::TradingStrategy
        } else if text_lower.contains("risk") || text_lower.contains("exposure") || text_lower.contains("var") {
            QueryIntent::RiskAssessment
        } else if text_lower.contains("historical") || text_lower.contains("past") || text_lower.contains("trend") {
            QueryIntent::HistoricalAnalysis
        } else if text_lower.contains("now") || text_lower.contains("current") || text_lower.contains("real-time") {
            QueryIntent::RealTimeQuery
        } else {
            QueryIntent::GeneralInformation
        }
    }

    fn extract_temporal_context(&self, text: &str) -> Option<TemporalContext> {
        let text_lower = text.to_lowercase();
        let _now = chrono::Utc::now();
        
        let relative_time = if text_lower.contains("last hour") {
            Some("last hour".to_string())
        } else if text_lower.contains("today") {
            Some("today".to_string())
        } else if text_lower.contains("yesterday") {
            Some("yesterday".to_string())
        } else if text_lower.contains("last week") {
            Some("last week".to_string())
        } else {
            None
        };

        let market_session = if text_lower.contains("pre-market") {
            Some("pre-market".to_string())
        } else if text_lower.contains("after-hours") {
            Some("after-hours".to_string())
        } else {
            None
        };

        if relative_time.is_some() || market_session.is_some() {
            Some(TemporalContext {
                time_range: None, // TODO: Convert relative time to actual time range
                relative_time,
                market_session,
            })
        } else {
            None
        }
    }

    fn extract_market_filters(&self, text: &str, filters: &Option<HashMap<String, String>>) -> Vec<MarketFilter> {
        let mut market_filters = Vec::new();
        
        // Extract from text
        let text_upper = text.to_uppercase();
        let symbols = self.extract_symbols(&text_upper);
        
        for symbol in symbols {
            market_filters.push(MarketFilter {
                symbol: Some(symbol),
                asset_class: None,
                market_condition: None,
                volatility_range: None,
            });
        }

        // Extract from filters
        if let Some(filters) = filters {
            if let Some(symbol) = filters.get("symbol") {
                market_filters.push(MarketFilter {
                    symbol: Some(symbol.clone()),
                    asset_class: filters.get("asset_class").cloned(),
                    market_condition: filters.get("market_condition").cloned(),
                    volatility_range: None,
                });
            }
        }

        market_filters
    }

    fn extract_symbols(&self, text: &str) -> Vec<String> {
        // Simple regex-like extraction for common stock symbols
        let mut symbols = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();
        
        for word in words {
            // Check if word looks like a stock symbol (2-5 uppercase letters)
            if word.len() >= 2 && word.len() <= 5 && word.chars().all(|c| c.is_ascii_uppercase()) {
                symbols.push(word.to_string());
            }
        }
        
        symbols
    }
}