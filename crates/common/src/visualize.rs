use crate::trace::{TailClass, TraceEvent, TraceLog};
use std::io::Write;

// ANSI codes
const CYAN: &str = "\x1b[36m";
const DIM: &str = "\x1b[2m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

/// Read delay from env var, defaulting to 120ms.
pub fn viz_delay_ms() -> u64 {
    std::env::var("MULTIVECTOR_VIZ_DELAY_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(120)
}

/// Render a filled/empty bar string.
fn bar(value: f32, max: f32, width: usize) -> String {
    let filled = ((value / max.max(1e-6)) * width as f32).round() as usize;
    let filled = filled.min(width);
    format!("{}{}", "█".repeat(filled), "░".repeat(width - filled))
}

/// Truncate a token string to max 10 chars, appending ".." if longer.
fn trunc(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("{}..", &s[..max.saturating_sub(2)])
    } else {
        s.to_string()
    }
}

/// Convert a similarity value to a block character for intensity display.
fn intensity_block(v: f32) -> char {
    if v >= 0.8 {
        '█'
    } else if v >= 0.6 {
        '▓'
    } else if v >= 0.4 {
        '▒'
    } else if v >= 0.2 {
        '░'
    } else {
        ' '
    }
}

async fn sleep_ms(ms: u64) {
    if ms > 0 {
        tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
    }
}

fn flush() {
    std::io::stdout().flush().ok();
}

/// Render all trace events with per-event ASCII diagrams and a configurable delay.
pub async fn render_trace(log: &TraceLog, delay_ms: u64) {
    for (_ts, event) in &log.events {
        render_event(event, delay_ms).await;
    }
}

async fn render_event(event: &TraceEvent, delay_ms: u64) {
    match event {
        TraceEvent::HnswInsert {
            doc_id,
            layer,
            neighbors,
        } => {
            let neighbor_str = if neighbors.is_empty() {
                "(none)".to_string()
            } else {
                neighbors
                    .iter()
                    .map(|n| n.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            // Build graph state line showing existing nodes + new one
            let graph_nodes: String = {
                let mut parts: Vec<String> = neighbors.iter().map(|n| format!("[{}]", n)).collect();
                parts.push(format!("{}[{}←new]{}", BOLD, doc_id, RESET));
                parts.join(" ")
            };
            println!(
                "{CYAN} ┌─ HNSW Insert ────────────────────────────────────┐{RESET}"
            );
            println!(
                "{CYAN} │{RESET}  {BOLD}doc_id:{RESET} {doc_id:<40}{CYAN}│{RESET}"
            );
            println!(
                "{CYAN} │{RESET}  {BOLD}layer: {RESET} {layer:<40}{CYAN}│{RESET}"
            );
            println!(
                "{CYAN} │{RESET}  edges → {neighbor_str:<38}{CYAN}│{RESET}"
            );
            println!("{CYAN} │{RESET}                                                   {CYAN}│{RESET}");
            println!(
                "{CYAN} │{RESET}  {DIM}graph state:{RESET} {graph_nodes}"
            );
            println!(
                "{CYAN} └───────────────────────────────────────────────────┘{RESET}"
            );
            flush();
            sleep_ms(delay_ms).await;
        }

        TraceEvent::HnswQuery {
            hop,
            current,
            candidates,
        } => {
            println!(
                "{CYAN} ┌─ HNSW Query hop {hop} ──────────────────────────────┐{RESET}"
            );
            println!(
                "{CYAN} │{RESET}  current: {BOLD}doc {current}{RESET:<40}{CYAN}│{RESET}"
            );
            println!("{CYAN} │{RESET}  candidates:                                       {CYAN}│{RESET}");
            for (id, score) in candidates.iter().take(6) {
                let b = bar(*score, 1.0, 24);
                println!(
                    "{CYAN} │{RESET}    doc {id:<4} │ {DIM}{b}{RESET} │ {score:.3}  {CYAN}│{RESET}"
                );
            }
            println!(
                "{CYAN} └───────────────────────────────────────────────────┘{RESET}"
            );
            flush();
            sleep_ms(delay_ms).await;
        }

        TraceEvent::HnswLayerStats {
            layer,
            node_count,
            avg_degree,
        } => {
            println!(
                "{CYAN} ┌─ HNSW Layer Stats ────────────────────────────────┐{RESET}"
            );
            println!(
                "{CYAN} │{RESET}  layer:      {BOLD}{layer}{RESET:<40}{CYAN}│{RESET}"
            );
            println!(
                "{CYAN} │{RESET}  node_count: {node_count:<40}{CYAN}│{RESET}"
            );
            println!(
                "{CYAN} │{RESET}  avg_degree: {avg_degree:.3}{RESET}"
            );
            println!(
                "{CYAN} └───────────────────────────────────────────────────┘{RESET}"
            );
            flush();
            sleep_ms(delay_ms).await;
        }

        TraceEvent::Tokenize { doc_id, tokens } => {
            let n = tokens.len();
            let display: Vec<String> = tokens.iter().take(8).map(|t| trunc(t, 10)).collect();
            println!(
                "{CYAN} ┌─ Tokenize doc {doc_id} ──────────────────────────────┐{RESET}"
            );
            println!(
                "{CYAN} │{RESET}  {BOLD}{n} tokens:{RESET:<42}{CYAN}│{RESET}"
            );
            for (i, tok) in display.iter().enumerate() {
                println!(
                    "{CYAN} │{RESET}   [{i}] {tok:<44}{CYAN}│{RESET}"
                );
            }
            if n > 8 {
                let more = n - 8;
                println!(
                    "{CYAN} │{RESET}   {DIM}... +{more} more{RESET:<42}{CYAN}│{RESET}"
                );
            }
            println!(
                "{CYAN} └───────────────────────────────────────────────────┘{RESET}"
            );
            flush();
            sleep_ms(delay_ms).await;
        }

        TraceEvent::TokenEmbed {
            doc_id: _,
            token,
            embedding_preview,
        } => {
            let tok = trunc(token, 10);
            let e = embedding_preview;
            println!(
                "  {tok:<12} → [ {}{:+.3}{RESET}  {:+.3}  {:+.3}  {DIM}… ] (128-dim){RESET}",
                BOLD, e[0], e[1], e[2]
            );
            flush();
            // TokenEmbed fires rapidly — use short 30ms delay
            let short = if delay_ms == 0 { 0 } else { 30 };
            sleep_ms(short).await;
        }

        TraceEvent::MaxSimMatrix {
            query_tokens,
            doc_id,
            matrix,
            row_maxima,
            score,
        } => {
            // Truncate to 6 query and 6 doc tokens
            let max_q = 6usize;
            let max_d = 6usize;
            let n_q = query_tokens.len().min(max_q);
            let n_d = matrix.first().map(|r| r.len()).unwrap_or(0).min(max_d);

            // Derive doc token count from matrix columns (no doc_tokens field in event)
            // We'll label doc columns as d0..d{n_d-1}
            println!(
                "{CYAN} ┌─ MaxSim: query × doc {doc_id} ─────────────────────┐{RESET}"
            );
            println!("{CYAN} │{RESET}");

            // Header row: doc token labels
            let mut header = format!("{CYAN} │{RESET}  {:<8} │", "");
            for j in 0..n_d {
                header.push_str(&format!(" {DIM}d{j:<7}{RESET}"));
            }
            println!("{header}");

            // Matrix rows
            for i in 0..n_q {
                let qtok = trunc(&query_tokens[i], 8);
                let row_max = row_maxima.get(i).copied().unwrap_or(0.0);
                let mut row_line = format!("{CYAN} │{RESET} {BOLD}{qtok:<8}{RESET} │");
                if let Some(row) = matrix.get(i) {
                    for j in 0..n_d {
                        let v = row.get(j).copied().unwrap_or(0.0);
                        let blk = intensity_block(v);
                        row_line.push_str(&format!("    {blk}    "));
                    }
                }
                row_line.push_str(&format!("│ max={row_max:.2}"));
                println!("{row_line}");
            }

            println!("{CYAN} │{RESET}   {DIM}──────────{RESET}");
            println!(
                "{CYAN} │{RESET}                                    {BOLD}Score = {score:.3}{RESET}"
            );
            println!(
                "{CYAN} └──────────────────────────────────────────────────────┘{RESET}"
            );
            flush();
            sleep_ms(delay_ms).await;
        }

        TraceEvent::CentroidAssign {
            doc_id: _,
            token,
            centroid_id,
        } => {
            let tok = trunc(token, 10);
            println!(
                "  {tok:<12} → {DIM}centroid {RESET}{BOLD}{centroid_id}{RESET}"
            );
            flush();
            // Fast per-token event
            let short = if delay_ms == 0 { 0 } else { 40 };
            sleep_ms(short).await;
        }

        TraceEvent::CentroidAnn {
            query_token,
            top_centroids,
        } => {
            let qtok = trunc(query_token, 20);
            println!(
                "{CYAN} ┌─ Centroid ANN: \"{qtok}\" ──────────────────────┐{RESET}"
            );
            println!("{CYAN} │{RESET}  top centroids:                                    {CYAN}│{RESET}");
            for (id, score) in top_centroids.iter().take(6) {
                let b = bar(*score, 1.0, 20);
                println!(
                    "{CYAN} │{RESET}    c{id:<4} │ {DIM}{b}{RESET} │ {score:.3}  {CYAN}│{RESET}"
                );
            }
            println!(
                "{CYAN} └───────────────────────────────────────────────────┘{RESET}"
            );
            flush();
            sleep_ms(delay_ms).await;
        }

        TraceEvent::CandidateExpand {
            centroid_ids,
            candidate_doc_ids,
            pruned_count,
        } => {
            let n_cands = candidate_doc_ids.len() as f32;
            let n_pruned = *pruned_count as f32;
            let total = n_cands + n_pruned;
            let (filled_bar, empty_bar) = if total > 0.0 {
                (bar(n_cands, total, 20), bar(n_pruned, total, 20))
            } else {
                (bar(0.0, 1.0, 20), bar(0.0, 1.0, 20))
            };
            let pct = if total > 0.0 {
                (n_pruned / total) * 100.0
            } else {
                0.0
            };
            println!(
                "{CYAN} ┌─ Candidate Expansion ────────────────────────────┐{RESET}"
            );
            println!(
                "{CYAN} │{RESET}  centroids activated: {BOLD}{:<27}{RESET}{CYAN}│{RESET}",
                centroid_ids.len()
            );
            println!(
                "{CYAN} │{RESET}  candidates:  {n_cands:<4.0} docs  {DIM}{filled_bar}{RESET} {CYAN}│{RESET}"
            );
            println!(
                "{CYAN} │{RESET}  pruned:      {n_pruned:<4.0} docs  {DIM}{empty_bar}{RESET}  {CYAN}│{RESET}"
            );
            println!(
                "{CYAN} │{RESET}  savings: {BOLD}{pct:.0}%{RESET:<42}{CYAN}│{RESET}"
            );
            println!(
                "{CYAN} └───────────────────────────────────────────────────┘{RESET}"
            );
            flush();
            sleep_ms(delay_ms).await;
        }

        TraceEvent::PlaidMaxSim {
            candidate_count,
            scored_count,
            top_k,
        } => {
            println!(
                "{CYAN} ┌─ PLAID MaxSim ────────────────────────────────────┐{RESET}"
            );
            println!(
                "{CYAN} │{RESET}  scored {BOLD}{scored_count}{RESET} of {candidate_count} candidates{:<20}{CYAN}│{RESET}",
                ""
            );
            println!("{CYAN} │{RESET}  top results:                                       {CYAN}│{RESET}");
            for (id, score) in top_k.iter().take(6) {
                let b = bar(*score, 1.0, 24);
                println!(
                    "{CYAN} │{RESET}    doc {id:<4} │ {DIM}{b}{RESET} │ {score:.3}  {CYAN}│{RESET}"
                );
            }
            println!(
                "{CYAN} └────────────────────────────────────────────────────┘{RESET}"
            );
            flush();
            sleep_ms(delay_ms).await;
        }

        TraceEvent::XtrScore {
            query_token_id,
            token_scores,
        } => {
            // Sort by score descending, take top 6
            let mut sorted = token_scores.clone();
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top6 = sorted.iter().take(6);
            println!(
                "{CYAN} ┌─ Xtr scores for token_id={query_token_id} ────────────────┐{RESET}"
            );
            for (id, score) in top6 {
                let b = bar(*score, 1.0, 20);
                println!(
                    "{CYAN} │{RESET}  doc {id:<4} │ {DIM}{b}{RESET} │ {score:.3}  {CYAN}│{RESET}"
                );
            }
            println!(
                "{CYAN} └────────────────────────────────────────────────────┘{RESET}"
            );
            flush();
            sleep_ms(delay_ms).await;
        }

        TraceEvent::CandidateGather {
            gathered,
            overlap_with_gt: _,
            fraction_promoted,
        } => {
            let pct = fraction_promoted * 100.0;
            let b = bar(*fraction_promoted, 1.0, 24);
            println!(
                "{CYAN} ┌─ Candidate Gather ────────────────────────────────┐{RESET}"
            );
            println!(
                "{CYAN} │{RESET}  gathered: {BOLD}{}{RESET} docs{:<35}{CYAN}│{RESET}",
                gathered.len(),
                ""
            );
            println!(
                "{CYAN} │{RESET}  promoted: {BOLD}{pct:.0}%{RESET}  {DIM}{b}{RESET}  {CYAN}│{RESET}"
            );
            println!(
                "{CYAN} └────────────────────────────────────────────────────┘{RESET}"
            );
            flush();
            sleep_ms(delay_ms).await;
        }

        TraceEvent::MaxSimRefine {
            candidate_count,
            top_k,
        } => {
            println!(
                "{CYAN} ┌─ MaxSim Refine ({candidate_count} candidates) ────────────────┐{RESET}"
            );
            for (id, score) in top_k.iter().take(6) {
                let b = bar(*score, 1.0, 24);
                println!(
                    "{CYAN} │{RESET}    doc {id:<4} │ {DIM}{b}{RESET} │ {score:.3}  {CYAN}│{RESET}"
                );
            }
            println!(
                "{CYAN} └────────────────────────────────────────────────────┘{RESET}"
            );
            flush();
            sleep_ms(delay_ms).await;
        }

        TraceEvent::TailHandle {
            token_type,
            count,
            classification,
        } => {
            // TAU = 256
            let tau: f32 = 256.0;
            let b = bar(*count as f32, tau, 20);
            let class_str = match classification {
                TailClass::Tail => "Tail  ",
                TailClass::Normal => "Normal",
                TailClass::Heavy => "Heavy ",
            };
            let tok = trunc(token_type, 10);
            println!(
                "  {tok:<12} {DIM}freq={RESET}{BOLD}{count:<5}{RESET} {DIM}{b}{RESET}  {class_str}"
            );
            flush();
            sleep_ms(delay_ms).await;
        }

        TraceEvent::DampedScore {
            token_type,
            variance,
            weight,
        } => {
            let b = bar(*weight, 1.0, 20);
            let tok = trunc(token_type, 10);
            println!(
                "  {tok:<12} {DIM}var={RESET}{variance:.3}  {BOLD}w={weight:.3}{RESET}  {DIM}{b}{RESET}"
            );
            flush();
            sleep_ms(delay_ms).await;
        }

        TraceEvent::BudgetBound {
            token_type,
            raw_kappa,
            floored,
            ceiled,
            final_kappa,
        } => {
            // THETA = 32 (typical budget per token)
            let theta: f32 = 32.0;
            let b = bar(*final_kappa as f32, theta, 20);
            let tok = trunc(token_type, 10);
            println!(
                "  {tok:<12} {DIM}raw={RESET}{raw_kappa:.1} → [{floored},{ceiled}] → {BOLD}κⱼ={final_kappa}{RESET}  {DIM}{b}{RESET}"
            );
            flush();
            sleep_ms(delay_ms).await;
        }

        TraceEvent::BudgetReconcile {
            total_budget,
            allocated,
            redistributed,
        } => {
            let b = bar(*allocated as f32, *total_budget as f32, 24);
            println!(
                "{CYAN} ┌─ Budget Reconcile ────────────────────────────────┐{RESET}"
            );
            println!(
                "{CYAN} │{RESET}  total:         {BOLD}{total_budget:<33}{RESET}{CYAN}│{RESET}"
            );
            println!(
                "{CYAN} │{RESET}  allocated:     {allocated:<4}    {DIM}{b}{RESET}  {CYAN}│{RESET}"
            );
            println!(
                "{CYAN} │{RESET}  redistributed: {redistributed:<33}{CYAN}│{RESET}"
            );
            println!(
                "{CYAN} └────────────────────────────────────────────────────┘{RESET}"
            );
            flush();
            sleep_ms(delay_ms).await;
        }

        TraceEvent::PqInspect {
            level,
            dimensions,
            subquantizer_count,
            code_bits,
        } => {
            println!(
                "{CYAN} ┌─ PQ Inspect ──────────────────────────────────────┐{RESET}"
            );
            println!(
                "{CYAN} │{RESET}  level:             {BOLD}{level:<30}{RESET}{CYAN}│{RESET}"
            );
            println!(
                "{CYAN} │{RESET}  dimensions:        {dimensions:<30}{CYAN}│{RESET}"
            );
            println!(
                "{CYAN} │{RESET}  subquantizers:     {subquantizer_count:<30}{CYAN}│{RESET}"
            );
            println!(
                "{CYAN} │{RESET}  code_bits:         {code_bits:<30}{CYAN}│{RESET}"
            );
            println!(
                "{CYAN} └────────────────────────────────────────────────────┘{RESET}"
            );
            flush();
            sleep_ms(delay_ms).await;
        }

        TraceEvent::TachiomSearch { timings } => {
            println!(
                "{CYAN} ┌─ TACHIOM Search ──────────────────────────────────┐{RESET}"
            );
            println!(
                "{CYAN} │{RESET}  gather:  {BOLD}{:.1}ms{RESET}{:<36}{CYAN}│{RESET}",
                timings.gather_ms, ""
            );
            println!(
                "{CYAN} │{RESET}  refine:  {BOLD}{:.1}ms{RESET}{:<36}{CYAN}│{RESET}",
                timings.refine_ms, ""
            );
            println!(
                "{CYAN} │{RESET}  total:   {BOLD}{:.1}ms{RESET}{:<36}{CYAN}│{RESET}",
                timings.total_ms, ""
            );
            println!(
                "{CYAN} └────────────────────────────────────────────────────┘{RESET}"
            );
            flush();
            sleep_ms(delay_ms).await;
        }
    }
}
