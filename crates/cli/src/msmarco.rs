//! MS MARCO passage embedding pipeline.
//!
//! Embeds 8.8M passages with:
//! - Jina ColBERT v2: per-token 128-dim embeddings (reused by PLAID / WARP / TACHIOM / ColBERT)
//! - Voyage-4-large: 1024-dim sentence embeddings (HNSW only)
//!
//! Output layout under `<out_dir>`:
//!   jina/offsets.bin  — u64[N]: byte offsets into jina/data.bin
//!   jina/lengths.bin  — u32[N]: token counts per passage
//!   jina/data.bin     — f32[Σtokens × 128]: packed token embeddings  (~35 GB)
//!   jina/meta.json    — {"count": N, "dim": 128, "model": "jina-colbert-v2"}
//!   voyage/data.bin   — f32[N × 1024]: sentence embeddings            (~36 GB)
//!   voyage/meta.json  — {"count": N, "dim": 1024, "model": "voyage-4-large"}
//!
//! Resume: progress is derived from file sizes on disk; rerun to continue after interruption.

use std::{
    fs::{File, OpenOptions},
    io::{BufRead, BufReader, Seek, SeekFrom, Write},
    path::Path,
    sync::Arc,
    time::Duration,
};

use anyhow::{anyhow, Context, Result};
use colbert::JinaColBertClient;
use common::{TokenMatrix, TOKEN_DIM};
use hnsw::voyage::VoyageClient;
use tokio::{sync::Semaphore, task::JoinSet};

const JINA_BATCH: usize = 32;
const VOYAGE_BATCH: usize = 128;
const JINA_CONCURRENCY: usize = 2;  // Jina free tier: 2 concurrent requests max
const VOYAGE_CONCURRENCY: usize = 8;
const MACRO_BATCH: usize = 3_200;
const VOYAGE_DIM: usize = 1024;
const TOTAL_FULL: usize = 8_841_823;

pub async fn run_embed_msmarco(
    collection_tsv: &Path,
    out_dir: &Path,
    jina_only: bool,
    voyage_only: bool,
    limit: Option<usize>,
) -> Result<()> {
    let total = limit.unwrap_or(TOTAL_FULL).min(TOTAL_FULL);
    let do_jina = !voyage_only;
    let do_voyage = !jina_only;

    let jina_dir = out_dir.join("jina");
    let voyage_dir = out_dir.join("voyage");

    if do_jina {
        std::fs::create_dir_all(&jina_dir)?;
    }
    if do_voyage {
        std::fs::create_dir_all(&voyage_dir)?;
    }

    let jina_done = if do_jina { jina_passages_done(&jina_dir) } else { usize::MAX };
    let voyage_done = if do_voyage { voyage_passages_done(&voyage_dir) } else { usize::MAX };

    println!("  MS MARCO embedding pipeline");
    println!("  Input:  {}", collection_tsv.display());
    println!("  Output: {}", out_dir.display());
    println!("  Target: {total} passages");
    if do_jina {
        println!("  Jina done:   {jina_done} / {total}");
    }
    if do_voyage {
        println!("  Voyage done: {voyage_done} / {total}");
    }

    // Each model makes an independent sequential pass over the TSV.
    // Reads the file twice but avoids complex interleaved-progress logic.
    if do_jina && jina_done < total {
        let client = Arc::new(
            JinaColBertClient::from_env()
                .ok_or_else(|| anyhow!("JINA_API_KEY required — set it in .env"))?,
        );
        embed_jina_pass(collection_tsv, &jina_dir, jina_done, total, client).await?;
    } else if do_jina {
        println!("  [jina] already complete ({jina_done} passages)");
    }

    if do_voyage && voyage_done < total {
        let client = Arc::new(
            VoyageClient::from_env()
                .ok_or_else(|| anyhow!("VOYAGE_API_KEY required — set it in .env"))?,
        );
        embed_voyage_pass(collection_tsv, &voyage_dir, voyage_done, total, client).await?;
    } else if do_voyage {
        println!("  [voyage] already complete ({voyage_done} passages)");
    }

    // Write / update meta.json with final counts.
    if do_jina {
        let n = jina_passages_done(&jina_dir);
        write_meta(&jina_dir, n, TOKEN_DIM, colbert::jina::MODEL)?;
        println!("  [jina]   meta.json: count={n}");
    }
    if do_voyage {
        let n = voyage_passages_done(&voyage_dir);
        let model = VoyageClient::from_env()
            .map(|c| c.model.clone())
            .unwrap_or_else(|| hnsw::voyage::DEFAULT_MODEL.to_string());
        write_meta(&voyage_dir, n, VOYAGE_DIM, &model)?;
        println!("  [voyage] meta.json: count={n}");
    }

    println!("  Done.");
    Ok(())
}

// ─── Jina pass ───────────────────────────────────────────────────────────────

async fn embed_jina_pass(
    tsv: &Path,
    out_dir: &Path,
    start_from: usize,
    total: usize,
    client: Arc<JinaColBertClient>,
) -> Result<()> {
    println!("  [jina] embedding from passage {start_from} …");

    let mut data_f = open_append(&out_dir.join("data.bin"))?;
    let mut off_f = open_append(&out_dir.join("offsets.bin"))?;
    let mut len_f = open_append(&out_dir.join("lengths.bin"))?;

    let mut data_offset: u64 = data_f.seek(SeekFrom::End(0))?;
    let mut count = start_from;
    let mut batch: Vec<String> = Vec::with_capacity(MACRO_BATCH);

    let reader = BufReader::new(
        File::open(tsv).with_context(|| format!("opening {}", tsv.display()))?,
    );

    for (line_no, line_result) in reader.lines().enumerate().skip(start_from).take(total - start_from) {
        let line = line_result?;
        batch.push(parse_text(&line, line_no)?);

        if batch.len() >= MACRO_BATCH {
            data_offset =
                flush_jina(&client, &batch, &mut data_f, &mut off_f, &mut len_f, data_offset)
                    .await?;
            count += batch.len();
            batch.clear();
            print_progress("[jina]", count, total);
        }
    }

    if !batch.is_empty() {
        flush_jina(&client, &batch, &mut data_f, &mut off_f, &mut len_f, data_offset).await?;
        count += batch.len();
    }

    println!("\r  [jina] done — {count} passages embedded        ");
    Ok(())
}

async fn flush_jina(
    client: &Arc<JinaColBertClient>,
    texts: &[String],
    data_f: &mut File,
    off_f: &mut File,
    len_f: &mut File,
    mut data_offset: u64,
) -> Result<u64> {
    let matrices = embed_jina_concurrent(client, texts).await?;

    let mut data_buf = Vec::<u8>::new();
    let mut off_buf = Vec::<u8>::with_capacity(matrices.len() * 8);
    let mut len_buf = Vec::<u8>::with_capacity(matrices.len() * 4);

    for m in &matrices {
        off_buf.extend_from_slice(&data_offset.to_le_bytes());
        len_buf.extend_from_slice(&(m.rows.len() as u32).to_le_bytes());
        for row in &m.rows {
            for &v in row.iter() {
                data_buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        data_offset += (m.rows.len() * TOKEN_DIM * 4) as u64;
    }

    // Write data before the index so partial writes leave the index consistent.
    data_f.write_all(&data_buf)?;
    off_f.write_all(&off_buf)?;
    len_f.write_all(&len_buf)?;

    Ok(data_offset)
}

async fn embed_jina_concurrent(
    client: &Arc<JinaColBertClient>,
    texts: &[String],
) -> Result<Vec<TokenMatrix>> {
    let chunks: Vec<Vec<String>> = texts
        .chunks(JINA_BATCH)
        .map(|c| c.iter().map(|s| s.clone()).collect())
        .collect();
    let n = chunks.len();

    let sem = Arc::new(Semaphore::new(JINA_CONCURRENCY));
    let mut js: JoinSet<(usize, Result<Vec<TokenMatrix>>)> = JoinSet::new();

    for (i, chunk) in chunks.into_iter().enumerate() {
        let permit = sem.clone().acquire_owned().await?;
        let client = Arc::clone(client);
        js.spawn(async move {
            let texts: Vec<&str> = chunk.iter().map(|s| s.as_str()).collect();
            let result = 'retry: {
                let mut delay = Duration::from_millis(500);
                for _ in 0..8u32 {
                    match client.embed_batch(&texts).await {
                        Ok(v) => break 'retry Ok(v),
                        Err(e) => {
                            eprintln!("\n  [jina retry] {e}");
                            tokio::time::sleep(delay).await;
                            delay = (delay * 2).min(Duration::from_secs(30));
                        }
                    }
                }
                client.embed_batch(&texts).await
            };
            drop(permit);
            (i, result)
        });
    }

    let mut results: Vec<Option<Vec<TokenMatrix>>> = vec![None; n];
    while let Some(joined) = js.join_next().await {
        let (i, r) = joined.map_err(|e| anyhow!("jina task panicked: {e}"))?;
        results[i] = Some(r?);
    }

    Ok(results.into_iter().flat_map(|r| r.unwrap()).collect())
}

// ─── Voyage pass ─────────────────────────────────────────────────────────────

async fn embed_voyage_pass(
    tsv: &Path,
    out_dir: &Path,
    start_from: usize,
    total: usize,
    client: Arc<VoyageClient>,
) -> Result<()> {
    println!("  [voyage] embedding from passage {start_from} …");

    let mut data_f = open_append(&out_dir.join("data.bin"))?;
    let mut count = start_from;
    let mut batch: Vec<String> = Vec::with_capacity(MACRO_BATCH);

    let reader = BufReader::new(
        File::open(tsv).with_context(|| format!("opening {}", tsv.display()))?,
    );

    for (line_no, line_result) in reader.lines().enumerate().skip(start_from).take(total - start_from) {
        let line = line_result?;
        batch.push(parse_text(&line, line_no)?);

        if batch.len() >= MACRO_BATCH {
            flush_voyage(&client, &batch, &mut data_f).await?;
            count += batch.len();
            batch.clear();
            print_progress("[voyage]", count, total);
        }
    }

    if !batch.is_empty() {
        flush_voyage(&client, &batch, &mut data_f).await?;
        count += batch.len();
    }

    println!("\r  [voyage] done — {count} passages embedded        ");
    Ok(())
}

async fn flush_voyage(
    client: &Arc<VoyageClient>,
    texts: &[String],
    data_f: &mut File,
) -> Result<()> {
    let embeddings = embed_voyage_concurrent(client, texts).await?;
    let mut buf = Vec::<u8>::with_capacity(embeddings.len() * VOYAGE_DIM * 4);
    for emb in &embeddings {
        if emb.len() != VOYAGE_DIM {
            return Err(anyhow!("Voyage returned {}-dim, expected {VOYAGE_DIM}", emb.len()));
        }
        for &v in emb.iter() {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    data_f.write_all(&buf)?;
    Ok(())
}

async fn embed_voyage_concurrent(
    client: &Arc<VoyageClient>,
    texts: &[String],
) -> Result<Vec<Vec<f32>>> {
    let chunks: Vec<Vec<String>> = texts
        .chunks(VOYAGE_BATCH)
        .map(|c| c.iter().map(|s| s.clone()).collect())
        .collect();
    let n = chunks.len();

    let sem = Arc::new(Semaphore::new(VOYAGE_CONCURRENCY));
    let mut js: JoinSet<(usize, Result<Vec<Vec<f32>>>)> = JoinSet::new();

    for (i, chunk) in chunks.into_iter().enumerate() {
        let permit = sem.clone().acquire_owned().await?;
        let client = Arc::clone(client);
        js.spawn(async move {
            let texts: Vec<&str> = chunk.iter().map(|s| s.as_str()).collect();
            let result = 'retry: {
                let mut delay = Duration::from_millis(500);
                for _ in 0..8u32 {
                    match client.embed_batch(&texts).await {
                        Ok(v) => break 'retry Ok(v),
                        Err(e) => {
                            eprintln!("\n  [voyage retry] {e}");
                            tokio::time::sleep(delay).await;
                            delay = (delay * 2).min(Duration::from_secs(30));
                        }
                    }
                }
                client.embed_batch(&texts).await
            };
            drop(permit);
            (i, result)
        });
    }

    let mut results: Vec<Option<Vec<Vec<f32>>>> = vec![None; n];
    while let Some(joined) = js.join_next().await {
        let (i, r) = joined.map_err(|e| anyhow!("voyage task panicked: {e}"))?;
        results[i] = Some(r?);
    }

    Ok(results.into_iter().flat_map(|r| r.unwrap()).collect())
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Number of passages already written to the Jina binary store (from offsets.bin size).
fn jina_passages_done(dir: &Path) -> usize {
    std::fs::metadata(dir.join("offsets.bin"))
        .map(|m| m.len() as usize / 8) // each u64 offset = 8 bytes
        .unwrap_or(0)
}

/// Number of passages already written to the Voyage binary store (from data.bin size).
fn voyage_passages_done(dir: &Path) -> usize {
    std::fs::metadata(dir.join("data.bin"))
        .map(|m| m.len() as usize / (VOYAGE_DIM * 4)) // each embedding = 1024 × f32
        .unwrap_or(0)
}

fn open_append(path: &Path) -> Result<File> {
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("opening {}", path.display()))
}

fn parse_text(line: &str, line_no: usize) -> Result<String> {
    line.splitn(2, '\t')
        .nth(1)
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("missing text column in TSV line {line_no}"))
}

fn write_meta(dir: &Path, count: usize, dim: usize, model: &str) -> Result<()> {
    let meta = serde_json::json!({
        "count": count,
        "dim": dim,
        "model": model,
    });
    std::fs::write(
        dir.join("meta.json"),
        serde_json::to_string_pretty(&meta)?,
    )?;
    Ok(())
}

fn print_progress(tag: &str, count: usize, total: usize) {
    let pct = (count as f64 / total as f64 * 100.0).min(100.0);
    print!("\r  {tag} {pct:5.1}%  {count}/{total}");
    let _ = std::io::stdout().flush();
}
