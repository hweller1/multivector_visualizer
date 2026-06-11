use anyhow::{anyhow, Result};
use common::corpus::SHARED_CORPUS;
use std::collections::HashMap;
use std::path::PathBuf;

/// Default embedding model. Override with `VOYAGE_MODEL=<other>` if needed.
pub const DEFAULT_MODEL: &str = "voyage-4-large";
const BASE_URL: &str = "https://ai.mongodb.com/v1";
const CACHE_PATH: &str = "cache/voyage_corpus_embeddings.json";

pub struct VoyageClient {
    api_key: String,
    pub model: String,
    http: reqwest::Client,
    cache_path: PathBuf,
}

impl VoyageClient {
    /// Load API key and optional model from environment / `.env`.
    /// Returns `None` if `VOYAGE_API_KEY` is absent.
    /// Set `VOYAGE_MODEL` to override the default model (e.g. `voyage-4-large`).
    pub fn from_env() -> Option<Self> {
        let _ = dotenvy::dotenv();
        let api_key = std::env::var("VOYAGE_API_KEY").ok()?.trim().to_string();
        if api_key.is_empty() { return None; }
        let model = std::env::var("VOYAGE_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
        // Cache is keyed by model so different models don't collide.
        let cache_path = PathBuf::from(format!("cache/voyage_{model}_embeddings.json"));
        Some(Self { api_key, model, http: reqwest::Client::new(), cache_path })
    }

    /// Return all 20 corpus embeddings, loading from local cache when possible.
    /// On a cache miss (or partial cache), calls the Voyage API for missing docs
    /// then writes the updated cache to disk.
    pub async fn load_corpus_embeddings(&self) -> Result<HashMap<u32, Vec<f32>>> {
        let mut embeddings = self.load_cache().unwrap_or_default();

        let missing: Vec<(u32, &str)> = SHARED_CORPUS
            .iter()
            .filter(|(id, _)| !embeddings.contains_key(id))
            .map(|(id, text)| (*id, *text))
            .collect();

        if missing.is_empty() {
            println!("  [voyage] loaded {} embeddings from cache", embeddings.len());
            return Ok(embeddings);
        }

        println!(
            "  [voyage] fetching {} doc{} from {} ({} cached)…",
            missing.len(),
            if missing.len() == 1 { "" } else { "s" },
            self.model,
            embeddings.len(),
        );

        // Batch all missing docs in a single API call (Voyage supports arrays).
        let texts: Vec<&str> = missing.iter().map(|(_, t)| *t).collect();
        let batch = self.embed_batch(&texts).await?;

        for ((doc_id, _), emb) in missing.iter().zip(batch.into_iter()) {
            embeddings.insert(*doc_id, emb);
        }

        self.save_cache(&embeddings)?;
        println!("  [voyage] cached {} embeddings ({}) to {:?}", embeddings.len(), self.model, self.cache_path);
        Ok(embeddings)
    }

    /// Embed a single query string.
    pub async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let mut batch = self.embed_batch(&[text]).await?;
        batch.pop().ok_or_else(|| anyhow!("empty embedding batch"))
    }

    // ── batch API ──────────────────────────────────────────────────────────

    pub async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let input: Vec<&str> = texts.to_vec();
        let body = serde_json::json!({
            "model": self.model,
            "input": input,
        });

        let raw = self
            .http
            .post(format!("{BASE_URL}/embeddings"))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        let status = raw.status();
        let resp = raw.json::<serde_json::Value>().await?;

        if !status.is_success() {
            return Err(anyhow!(
                "Voyage API HTTP {status} (model={}, {} input{}) — {resp}\n\
                 Hint: Voyage keys start with 'pa-'. Check VOYAGE_API_KEY in .env.",
                self.model,
                texts.len(),
                if texts.len() == 1 { "" } else { "s" },
            ));
        }

        let data = resp["data"]
            .as_array()
            .ok_or_else(|| anyhow!("Voyage API: unexpected response shape: {resp}"))?;

        data.iter()
            .map(|item| {
                item["embedding"]
                    .as_array()
                    .ok_or_else(|| anyhow!("missing embedding field"))
                    .map(|arr| arr.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect())
            })
            .collect()
    }

    fn load_cache(&self) -> Option<HashMap<u32, Vec<f32>>> {
        let raw = std::fs::read_to_string(&self.cache_path).ok()?;
        // Stored as {"0": [...], "1": [...]} — string keys from JSON map.
        let map: HashMap<String, Vec<f32>> = serde_json::from_str(&raw).ok()?;
        Some(map.into_iter().filter_map(|(k, v)| k.parse::<u32>().ok().map(|id| (id, v))).collect())
    }

    fn save_cache(&self, embeddings: &HashMap<u32, Vec<f32>>) -> Result<()> {
        if let Some(parent) = self.cache_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // Serialize with string keys for portability.
        let string_keyed: HashMap<String, &Vec<f32>> =
            embeddings.iter().map(|(k, v)| (k.to_string(), v)).collect();
        std::fs::write(&self.cache_path, serde_json::to_string(&string_keyed)?)?;
        Ok(())
    }
}
