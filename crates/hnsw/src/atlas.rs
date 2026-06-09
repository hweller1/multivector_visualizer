use anyhow::{anyhow, Result};
use futures::TryStreamExt;
use mongodb::{bson::doc, Client, Collection};
use mongodb::bson::Document;

pub struct HnswLayerStat {
    pub layer: u8,
    pub node_count: u32,
    pub avg_degree: f32,
}

pub struct AtlasClient {
    collection: Collection<Document>,
    index_name: &'static str,
    voyage_api_key: String,
    http: reqwest::Client,
}

impl AtlasClient {
    pub async fn from_env() -> Result<Self> {
        // Load .env at workspace root — no-op if absent
        let _ = dotenvy::dotenv();

        let uri = std::env::var("MONGODB_URI")
            .map_err(|_| anyhow!("MONGODB_URI not set — check .env at workspace root"))?;
        let voyage_api_key = std::env::var("VOYAGE_API_KEY")
            .map_err(|_| anyhow!("VOYAGE_API_KEY not set — check .env at workspace root"))?;

        let client = Client::with_uri_str(&uri).await?;
        let collection = client
            .database("multivector")
            .collection::<Document>("multivector_demo");

        Ok(Self {
            collection,
            index_name: "vector_index",
            voyage_api_key,
            http: reqwest::Client::new(),
        })
    }

    pub async fn index_doc(&self, doc_id: u32, text: &str) -> Result<Vec<f32>> {
        let embedding = self.embed(text).await?;

        let embedding_bson: Vec<mongodb::bson::Bson> = embedding
            .iter()
            .map(|&f| mongodb::bson::Bson::Double(f as f64))
            .collect();

        self.collection
            .insert_one(doc! {
                "doc_id": doc_id as i64,
                "text": text,
                "embedding": embedding_bson,
            })
            .await?;

        Ok(embedding)
    }

    pub async fn query(&self, embedding: &[f32], top_k: usize) -> Result<Vec<(u32, f32)>> {
        let query_vec: Vec<mongodb::bson::Bson> = embedding
            .iter()
            .map(|&f| mongodb::bson::Bson::Double(f as f64))
            .collect();

        let pipeline = vec![
            doc! {
                "$vectorSearch": {
                    "index": self.index_name,
                    "path": "embedding",
                    "queryVector": query_vec,
                    "numCandidates": (top_k as i64) * 10,
                    "limit": top_k as i64,
                }
            },
            doc! {
                "$project": {
                    "doc_id": 1,
                    "score": { "$meta": "vectorSearchScore" },
                    "_id": 0,
                }
            },
        ];

        let mut cursor = self.collection.aggregate(pipeline).await?;
        let mut results = Vec::new();
        while let Some(doc) = cursor.try_next().await? {
            let doc_id = doc.get_i64("doc_id").unwrap_or(0) as u32;
            let score = doc.get_f64("score").unwrap_or(0.0) as f32;
            results.push((doc_id, score));
        }
        Ok(results)
    }

    pub async fn graph_stats(&self) -> Result<Vec<HnswLayerStat>> {
        // Atlas Vector Search does not expose HNSW graph stats via the driver API
        Ok(vec![])
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let resp = self
            .http
            .post("https://ai.mongodb.com/v1/embeddings")
            .bearer_auth(&self.voyage_api_key)
            .json(&serde_json::json!({
                "model": "voyage-4-large",
                "input": [text]
            }))
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        let embedding: Vec<f32> = resp["data"][0]["embedding"]
            .as_array()
            .ok_or_else(|| anyhow!("Voyage API: missing embedding in response"))?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();

        if embedding.is_empty() {
            return Err(anyhow!("Voyage API returned empty embedding"));
        }
        Ok(embedding)
    }

    /// Embed a query text for use in vectorSearch.
    pub async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.embed(text).await
    }
}
