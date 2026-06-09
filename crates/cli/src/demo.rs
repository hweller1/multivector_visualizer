use anyhow::Result;
use colbert::ColBertEngine;
use common::{Engine, ScenarioRunner};
use hnsw::HnswEngine;
use plaid::PlaidEngine;
use std::path::PathBuf;
use std::time::Instant;
use tachiom::TachiomEngine;
use warp::WarpEngine;

async fn render_log_visual(log: &common::TraceLog) {
    if log.events.is_empty() {
        return;
    }
    let delay = common::viz_delay_ms();
    common::render_trace(log, delay).await;
}

fn vocab_path() -> PathBuf {
    std::env::var("MULTIVECTOR_VOCAB")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::current_dir()
                .unwrap()
                .join("vocab/wordpiece_vocab.txt")
        })
}

pub async fn run_demo(name: &str, dry_run: bool, _trace_json: Option<PathBuf>) -> Result<()> {
    if name == "compare" {
        crate::compare::run_compare().await?;
        return Ok(());
    }

    let scenario_path = PathBuf::from(format!("scenarios/{name}.toml"));
    let mut runner = ScenarioRunner::from_path(&scenario_path)?;
    runner.dry_run = dry_run;

    let engine_name = runner.scenario.meta.engine.clone();
    let vp = vocab_path();

    match engine_name.as_str() {
        "colbert" => {
            let mut engine = ColBertEngine::new(&vp)?;
            run_with_engine(&runner, &mut engine).await?;
        }
        "plaid" => {
            let mut engine = PlaidEngine::new(&vp)?;
            run_with_engine(&runner, &mut engine).await?;
        }
        "warp" => {
            let mut engine = WarpEngine::new(&vp)?;
            run_with_engine(&runner, &mut engine).await?;
        }
        "tachiom" => {
            let mut engine = TachiomEngine::new(&vp)?;
            run_with_engine(&runner, &mut engine).await?;
        }
        "hnsw" => {
            let mut engine = HnswEngine::new_local_with_voyage().await;
            run_with_engine(&runner, &mut engine).await?;
        }
        "compare" => {
            crate::compare::run_compare().await?;
        }
        other => {
            println!("engine '{other}' not yet implemented");
        }
    }

    Ok(())
}

async fn run_with_engine<E: Engine>(runner: &ScenarioRunner, engine: &mut E) -> Result<()> {
    runner
        .run(|op, args| {
            let engine_ptr: *mut E = engine as *mut E;
            async move {
                // Safety: single-threaded demo; engine outlives all futures.
                let e = unsafe { &mut *engine_ptr };
                dispatch_op(e, &op, args).await?;
                let pause_ms = common::viz_delay_ms() * 3;
                if pause_ms > 0 {
                    tokio::time::sleep(std::time::Duration::from_millis(pause_ms)).await;
                }
                Ok(())
            }
        })
        .await
}

fn fmt_ms(ms: f64) -> String {
    if ms < 1.0 { format!("{:.0}µs", ms * 1000.0) } else { format!("{:.2}ms", ms) }
}

async fn dispatch_op<E: Engine>(engine: &mut E, op: &str, args: Vec<String>) -> anyhow::Result<()> {
    match op {
        "index" => {
            let arg = args
                .first()
                .ok_or_else(|| anyhow::anyhow!("index: missing doc_id argument"))?;

            if arg == "all" {
                println!("  Indexing {} documents:", common::SHARED_CORPUS.len());
                let t_index = Instant::now();
                for (i, (doc_id, text)) in common::SHARED_CORPUS.iter().enumerate() {
                    let snippet: String = text.chars().take(50).collect();
                    println!("  [{:>2}] \"{}{}\"",
                        doc_id,
                        snippet,
                        if text.len() > 50 { "…" } else { "" });
                    let log = engine.index(*doc_id, text).await?;
                    // Show per-doc trace only for small logs (e.g. HNSW 2 events) — suppress verbose tokenize/embed streams
                    if log.events.len() <= 4 {
                        render_log_visual(&log).await;
                    }
                    let _ = i;
                }
                let index_ms = t_index.elapsed().as_secs_f64() * 1000.0;
                println!("  ✓ {} docs indexed.  \x1b[2m⏱ {index_ms:.1}ms total  ({:.2}ms/doc avg)\x1b[0m",
                    common::SHARED_CORPUS.len(),
                    index_ms / common::SHARED_CORPUS.len() as f64);
            } else {
                let doc_id: u32 = arg
                    .parse()
                    .map_err(|e| anyhow::anyhow!("index: invalid doc_id: {e}"))?;
                let text = common::SHARED_CORPUS
                    .iter()
                    .find(|(id, _)| *id == doc_id)
                    .map(|(_, text)| *text)
                    .ok_or_else(|| anyhow::anyhow!("index: doc_id {doc_id} not in SHARED_CORPUS"))?;
                let t_index = Instant::now();
                let log = engine.index(doc_id, text).await?;
                let index_ms = t_index.elapsed().as_secs_f64() * 1000.0;
                render_log_visual(&log).await;
                println!("  \x1b[2m⏱ indexed doc {doc_id} in {index_ms:.2}ms\x1b[0m");
            }
        }

        "query" => {
            // If the last arg is a plain integer, treat it as top_k.
            let (query_args, top_k) = match args.last().and_then(|a| a.parse::<usize>().ok()) {
                Some(k) => (args[..args.len() - 1].to_vec(), k),
                None    => (args.clone(), 10),
            };
            let text = query_args.join(" ");
            let t_query = Instant::now();
            let (results, log) = engine.query(&text, top_k).await?;
            let query_ms = t_query.elapsed().as_secs_f64() * 1000.0;

            let relevant = common::corpus::relevant_docs_for(&text);
            let label = common::corpus::relevance_label_for(&text);

            // Header — show ground truth context when available
            let rule = "═".repeat(62);
            println!("  ╔{rule}╗");
            println!("  ║  Query: \"{}\"", text);
            if let (Some(docs), Some(lbl)) = (relevant, label) {
                let doc_list: Vec<String> = docs.iter().map(|d| format!("{d}")).collect();
                println!("  ║  Relevant ({lbl}): docs [{}]", doc_list.join(", "));
            }
            println!("  ╚{rule}╝");

            let top_k_display = results.len();
            let mut hits = 0usize;
            let score_max = results.iter().map(|(_, s)| *s).fold(0.0f32, f32::max).max(0.01);

            for (i, (doc_id, score)) in results.iter().enumerate() {
                let snippet = common::corpus::SHARED_CORPUS
                    .iter()
                    .find(|(id, _)| *id == *doc_id)
                    .map(|(_, t)| {
                        let s: String = t.chars().take(52).collect();
                        if t.len() > 52 { format!("{s}…") } else { s }
                    })
                    .unwrap_or_else(|| format!("doc {doc_id}"));

                let (marker, color_on, color_off) = match relevant {
                    Some(r) if r.contains(doc_id) => {
                        hits += 1;
                        ("✓", "\x1b[32m", "\x1b[0m")  // green check
                    }
                    Some(_) => ("✗", "\x1b[31m", "\x1b[0m"),  // red cross
                    None    => ("·", "", ""),
                };
                let bar_filled = ((score / score_max) * 12.0).round() as usize;
                let bar_str = format!("{}{}", "█".repeat(bar_filled.min(12)), "░".repeat(12 - bar_filled.min(12)));
                println!("  {color_on}{marker}{color_off}  {}.  {score:.4}  \x1b[2m{bar_str}\x1b[0m  [{doc_id:>2}] \"{snippet}\"",
                    i + 1);
            }

            println!();
            if let Some(r) = relevant {
                let prec_bar_filled = (hits * 12 / top_k_display.max(1)).min(12);
                let prec_bar = format!("{}{}", "█".repeat(prec_bar_filled), "░".repeat(12 - prec_bar_filled));
                println!("  Precision@{top_k_display}: \x1b[2m{prec_bar}\x1b[0m  \x1b[1m{hits}/{top_k_display}\x1b[0m relevant  \x1b[2m(ground truth = {} docs)\x1b[0m",
                    r.len());
            }

            // Prefer engine-reported timing over outer wall-clock (avoids including API latency in HNSW).
            let timing_line = if let Some(t) = &log.timing {
                let search_str = fmt_ms(t.search_ms.unwrap_or(query_ms));
                let scored_str = t.docs_scored
                    .map(|n| format!("{n}/{} docs scored", common::SHARED_CORPUS.len()))
                    .unwrap_or_default();
                if let Some(embed_ms) = t.embed_ms {
                    // HNSW: separate embed (API) from search
                    format!("search \x1b[1m{search_str}\x1b[0m  \x1b[2m+ embed {embed_str}  {scored_str}\x1b[0m",
                        embed_str = fmt_ms(embed_ms))
                } else {
                    format!("\x1b[1m{search_str}\x1b[0m  \x1b[2m{scored_str}\x1b[0m")
                }
            } else {
                format!("\x1b[1m{}\x1b[0m  \x1b[2m(wall clock)\x1b[0m", fmt_ms(query_ms))
            };
            println!("  \x1b[2m⏱  query:\x1b[0m  {timing_line}");
            println!();

            render_log_visual(&log).await;
        }

        "inspect" => {
            let target = args.first().map(|s| s.as_str());
            let out = engine.inspect(target).await?;
            println!("{out}");
        }

        "trace" => {
            let filter = args.first().map(|s| s.as_str()).unwrap_or("all");
            let key = format!("__trace__{filter}");
            match engine.inspect(Some(&key)).await {
                Ok(json) if json.trim_start().starts_with('{') => {
                    match serde_json::from_str::<common::TraceLog>(&json) {
                        Ok(log) if !log.events.is_empty() => {
                            println!("  ({} phase events)", log.events.len());
                            render_log_visual(&log).await;
                        }
                        _ => println!("  (no trace events for '{filter}' — run index first)"),
                    }
                }
                _ => println!("  (trace not available for this engine)"),
            }
        }

        other => {
            println!("  unknown op: {other}");
        }
    }
    Ok(())
}
