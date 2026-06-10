use anyhow::{anyhow, Result};
use common::token::{TokenMatrix, TOKEN_DIM};
use std::collections::HashMap;

const BASE_URL: &str = "https://api.jina.ai/v1";
pub const MODEL: &str = "jina-colbert-v2";
pub const CACHE_PATH: &str = "cache/jina_colbert_embeddings.json";

pub struct JinaColBertClient {
    api_key: String,
    http: reqwest::Client,
}

impl JinaColBertClient {
    pub fn from_env() -> Option<Self> {
        let _ = dotenvy::dotenv();
        let key = std::env::var("JINA_API_KEY").ok()?.trim().to_string();
        if key.is_empty() { return None; }
        Some(Self { api_key: key, http: reqwest::Client::new() })
    }

    /// Fetch per-token ColBERT embeddings for a batch of texts.
    /// Returns one TokenMatrix per input text (rows = per-token 128-dim vectors).
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<TokenMatrix>> {
        let body = serde_json::json!({
            "model": MODEL,
            "input": texts,
            "output_format": "token_embeddings",
            "truncate_dim": TOKEN_DIM,
        });

        let raw = self.http
            .post(format!("{BASE_URL}/embeddings"))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        let status = raw.status();
        let resp: serde_json::Value = raw.json().await?;

        if !status.is_success() {
            return Err(anyhow!(
                "Jina API HTTP {status} — {}\nHint: check JINA_API_KEY in .env",
                resp
            ));
        }

        let data = resp["data"].as_array()
            .ok_or_else(|| anyhow!("Jina: unexpected response: {resp}"))?;

        data.iter().map(|item| {
            // Jina ColBERT returns per-token embeddings in the "embeddings" field.
            let emb_arr = item.get("embeddings")
                .or_else(|| item.get("token_embeddings"))
                .and_then(|v| v.as_array())
                .ok_or_else(|| anyhow!("Jina: missing 'embeddings' in response item: {item}"))?;

            let rows: Result<Vec<[f32; TOKEN_DIM]>> = emb_arr.iter().map(|tok| {
                let arr = tok.as_array()
                    .ok_or_else(|| anyhow!("token embedding not an array"))?;
                let dim = arr.len();
                if dim != TOKEN_DIM {
                    return Err(anyhow!("Jina: expected {TOKEN_DIM}-dim, got {dim}"));
                }
                let mut v = [0f32; TOKEN_DIM];
                for (i, x) in arr.iter().enumerate() {
                    v[i] = x.as_f64().unwrap_or(0.0) as f32;
                }
                Ok(v)
            }).collect();

            let rows = rows?;
            // Jina does not return token strings — use positional placeholders.
            // ColBertEncoder.encode() replaces these with WordPiece strings when counts match.
            let tokens: Vec<String> = (0..rows.len()).map(|i| format!("[{i}]")).collect();
            Ok(TokenMatrix { tokens, rows })
        }).collect()
    }

    /// Ensure all `texts` have cached embeddings, fetching any that are missing.
    /// Saves the updated cache to disk and returns it.
    pub async fn refresh_cache(&self, texts: &[&str]) -> Result<HashMap<String, TokenMatrix>> {
        let mut cache = load_cache();
        let missing: Vec<&str> = texts.iter().copied()
            .filter(|t| !cache.contains_key(*t))
            .collect();

        if missing.is_empty() {
            return Ok(cache);
        }

        print!("  [jina] fetching {} texts via {}… ", missing.len(), MODEL);
        let _ = std::io::Write::flush(&mut std::io::stdout());

        // Batch in groups of 32 to stay within request size limits.
        for chunk in missing.chunks(32) {
            let matrices = self.embed_batch(chunk).await?;
            for (text, matrix) in chunk.iter().zip(matrices) {
                cache.insert(text.to_string(), matrix);
            }
        }

        save_cache(&cache)?;
        println!("cached → {CACHE_PATH}");
        Ok(cache)
    }
}

pub fn load_cache() -> HashMap<String, TokenMatrix> {
    let raw = match std::fs::read_to_string(CACHE_PATH) {
        Ok(s) => s,
        Err(_) => return HashMap::new(),
    };
    let map: HashMap<String, serde_json::Value> = match serde_json::from_str(&raw) {
        Ok(m) => m,
        Err(_) => return HashMap::new(),
    };
    map.into_iter().filter_map(|(text, v)| {
        let tokens: Vec<String> = v["tokens"].as_array()?
            .iter().map(|t| t.as_str().unwrap_or("?").to_string()).collect();
        let rows: Option<Vec<[f32; TOKEN_DIM]>> = v["rows"].as_array()?.iter().map(|row| {
            let arr = row.as_array()?;
            if arr.len() != TOKEN_DIM { return None; }
            let mut v = [0f32; TOKEN_DIM];
            for (i, x) in arr.iter().enumerate() {
                v[i] = x.as_f64().unwrap_or(0.0) as f32;
            }
            Some(v)
        }).collect();
        let rows = rows?;
        Some((text, TokenMatrix { tokens, rows }))
    }).collect()
}

pub fn save_cache(cache: &HashMap<String, TokenMatrix>) -> Result<()> {
    std::fs::create_dir_all("cache")?;
    let ser: HashMap<&str, serde_json::Value> = cache.iter().map(|(text, m)| {
        let tokens: Vec<&str> = m.tokens.iter().map(|s| s.as_str()).collect();
        let rows: Vec<Vec<f32>> = m.rows.iter().map(|r| r.to_vec()).collect();
        (text.as_str(), serde_json::json!({ "tokens": tokens, "rows": rows }))
    }).collect();
    std::fs::write(CACHE_PATH, serde_json::to_string(&ser)?)?;
    Ok(())
}
