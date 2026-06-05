use std::path::PathBuf;
use anyhow::Result;
use common::{Engine, ScenarioRunner, TraceLog};
use colbert::ColBertEngine;

pub async fn run_demo(name: &str, dry_run: bool, _trace_json: Option<PathBuf>) -> Result<()> {
    if name == "compare" {
        println!("compare not yet implemented");
        return Ok(());
    }

    // Resolve scenarios/<name>.toml
    let scenario_path = PathBuf::from(format!("scenarios/{name}.toml"));
    let mut runner = ScenarioRunner::from_path(&scenario_path)?;
    runner.dry_run = dry_run;

    // Determine which engine to construct based on scenario.meta.engine
    let engine_name = runner.scenario.meta.engine.clone();

    match engine_name.as_str() {
        "colbert" => {
            let vocab_path = std::env::var("MULTIVECTOR_VOCAB")
                .map(PathBuf::from)
                .unwrap_or_else(|_| {
                    std::env::current_dir()
                        .unwrap()
                        .join("vocab/wordpiece_vocab.txt")
                });
            let mut engine = ColBertEngine::new(&vocab_path)?;
            run_with_engine(&runner, &mut engine).await?;
        }
        other => {
            println!("engine '{other}' not yet implemented");
        }
    }

    Ok(())
}

async fn run_with_engine<E: Engine>(runner: &ScenarioRunner, engine: &mut E) -> Result<()> {
    runner.run(|op, args| {
        // We need to dispatch into the engine — but we can't capture a mutable ref
        // in a closure that outlives itself. Use a raw pointer trick for single-thread demo.
        let engine_ptr: *mut E = engine as *mut E;
        async move {
            // Safety: single-threaded demo; engine outlives all futures.
            let e = unsafe { &mut *engine_ptr };
            dispatch_op(e, &op, args).await
        }
    }).await
}

async fn dispatch_op<E: Engine>(engine: &mut E, op: &str, args: Vec<String>) -> anyhow::Result<()> {
    match op {
        "index" => {
            // args[0] is a doc_id string → look up in SHARED_CORPUS
            let doc_id: u32 = args.get(0)
                .ok_or_else(|| anyhow::anyhow!("index: missing doc_id argument"))?
                .parse()
                .map_err(|e| anyhow::anyhow!("index: invalid doc_id: {e}"))?;

            let text = common::SHARED_CORPUS.iter()
                .find(|(id, _)| *id == doc_id)
                .map(|(_, text)| *text)
                .ok_or_else(|| anyhow::anyhow!("index: doc_id {doc_id} not in SHARED_CORPUS"))?;

            let log = engine.index(doc_id, text).await?;
            if !log.events.is_empty() {
                println!("  indexed doc {doc_id} ({} trace events)", log.events.len());
            }
        }
        "query" => {
            let text = args.join(" ");
            let (results, _log) = engine.query(&text, 10).await?;
            println!("  query: {text}");
            for (i, (doc_id, score)) in results.iter().enumerate() {
                println!("    {}. doc_id={doc_id}  score={score:.4}", i + 1);
            }
        }
        "inspect" => {
            let target = args.get(0).map(|s| s.as_str());
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
