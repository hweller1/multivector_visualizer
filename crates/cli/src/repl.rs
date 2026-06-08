use common::{Engine, TraceLog, VizRepl};
use std::io::{self, BufRead, Write};

pub async fn run_repl(engine: &mut dyn Engine, mut viz: VizRepl) -> anyhow::Result<()> {
    println!("  {} REPL — type 'help' for commands", engine.name());
    let stdin = io::stdin();
    let mut next_doc_id: u32 = 0;
    loop {
        print!("{}> ", engine.name());
        io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break; // EOF
        }
        let trimmed = line.trim();
        let parts: Vec<&str> = trimmed.splitn(2, ' ').collect();
        match parts.as_slice() {
            ["quit"] | ["exit"] | ["q"] => break,
            ["help"] | ["h"] => print_help(engine.name()),
            ["index", rest] => {
                let log = engine.index(next_doc_id, rest).await?;
                next_doc_id += 1;
                render_log(&log).await;
            }
            ["query", rest] => {
                let (results, log) = engine.query(rest, 10).await?;
                render_results(&results);
                render_log(&log).await;
            }
            ["inspect"] => {
                let out = engine.inspect(None).await?;
                println!("{out}");
            }
            ["inspect", target] => {
                let out = engine.inspect(Some(target)).await?;
                println!("{out}");
            }
            ["trace"] => {
                println!("(trace replay not yet implemented)");
            }
            ["trace", _filter] => {
                println!("(trace replay not yet implemented)");
            }
            [""] | [] => {}
            _ => eprintln!("unknown command; type 'help'"),
        }
        viz.print_suggestion();
    }
    Ok(())
}

pub async fn render_log(log: &TraceLog) {
    if log.events.is_empty() {
        return;
    }
    let delay = common::viz_delay_ms();
    common::render_trace(log, delay).await;
}

pub fn render_results(results: &[(u32, f32)]) {
    if results.is_empty() {
        println!("No results.");
        return;
    }
    for (i, (doc_id, score)) in results.iter().enumerate() {
        println!("  {}. doc_id={doc_id}  score={score:.4}", i + 1);
    }
}

pub fn print_help(engine_name: &str) {
    println!(
        r#"  {engine_name} REPL commands:
    index <text>         — index text as next document
    query <text>         — query top-10 results
    inspect              — show index summary
    inspect <target>     — engine-specific inspect (e.g., tokens)
    trace                — replay last trace log
    trace <filter>       — filter trace by stage name
    help / h             — show this help
    quit / exit / q      — exit REPL"#
    );
}
