use anyhow::Result;
use colbert::ColBertEngine;
use common::{Engine, ScenarioRunner};
use hnsw::HnswEngine;
use plaid::PlaidEngine;
use std::path::PathBuf;
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

    // Resolve scenarios/<name>.toml
    let scenario_path = PathBuf::from(format!("scenarios/{name}.toml"));
    let mut runner = ScenarioRunner::from_path(&scenario_path)?;
    runner.dry_run = dry_run;

    // Determine which engine to construct based on scenario.meta.engine
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
            let mut engine = HnswEngine::new_local();
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
            // We need to dispatch into the engine — but we can't capture a mutable ref
            // in a closure that outlives itself. Use a raw pointer trick for single-thread demo.
            let engine_ptr: *mut E = engine as *mut E;
            async move {
                // Safety: single-threaded demo; engine outlives all futures.
                let e = unsafe { &mut *engine_ptr };
                dispatch_op(e, &op, args).await?;
                // Post-step pause: give time to read the narration
                let pause_ms = common::viz_delay_ms() * 3;
                if pause_ms > 0 {
                    tokio::time::sleep(std::time::Duration::from_millis(pause_ms)).await;
                }
                Ok(())
            }
        })
        .await
}

async fn dispatch_op<E: Engine>(engine: &mut E, op: &str, args: Vec<String>) -> anyhow::Result<()> {
    match op {
        "index" => {
            let arg = args
                .first()
                .ok_or_else(|| anyhow::anyhow!("index: missing doc_id argument"))?;

            if arg == "all" {
                for (doc_id, text) in common::SHARED_CORPUS {
                    let log = engine.index(*doc_id, text).await?;
                    println!("  indexed doc {doc_id} ({} trace events)", log.events.len());
                    render_log_visual(&log).await;
                }
            } else {
                let doc_id: u32 = arg
                    .parse()
                    .map_err(|e| anyhow::anyhow!("index: invalid doc_id: {e}"))?;

                let text = common::SHARED_CORPUS
                    .iter()
                    .find(|(id, _)| *id == doc_id)
                    .map(|(_, text)| *text)
                    .ok_or_else(|| {
                        anyhow::anyhow!("index: doc_id {doc_id} not in SHARED_CORPUS")
                    })?;

                let log = engine.index(doc_id, text).await?;
                if !log.events.is_empty() {
                    println!("  indexed doc {doc_id} ({} trace events)", log.events.len());
                }
                render_log_visual(&log).await;
            }
        }
        "query" => {
            let text = args.join(" ");
            let (results, log) = engine.query(&text, 10).await?;
            println!("  query: {text}");
            for (i, (doc_id, score)) in results.iter().enumerate() {
                println!("    {}. doc_id={doc_id}  score={score:.4}", i + 1);
            }
            render_log_visual(&log).await;
        }
        "inspect" => {
            let target = args.first().map(|s| s.as_str());
            let out = engine.inspect(target).await?;
            println!("{out}");
        }
        "trace" => {
            println!("  (trace step — replay last trace log)");
        }
        other => {
            println!("  unknown op: {other}");
        }
    }
    Ok(())
}
