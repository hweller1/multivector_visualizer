use crate::corpus::SHARED_CORPUS;
use crate::trace::{TailClass, TraceEvent, TraceLog};
use std::io::Write;

const CYAN: &str = "\x1b[36m";
const DIM: &str = "\x1b[2m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";

pub fn viz_delay_ms() -> u64 {
    std::env::var("MULTIVECTOR_VIZ_DELAY_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(120)
}

fn bar(value: f32, max: f32, width: usize) -> String {
    let filled = ((value / max.max(1e-6)) * width as f32).round() as usize;
    let filled = filled.min(width);
    format!("{}{}", "█".repeat(filled), "░".repeat(width - filled))
}

fn trunc(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("{}..", &s[..max.saturating_sub(2)])
    } else {
        s.to_string()
    }
}

fn intensity_block(v: f32) -> char {
    if v >= 0.8 { '█' } else if v >= 0.6 { '▓' } else if v >= 0.4 { '▒' } else if v >= 0.2 { '░' } else { ' ' }
}

/// Look up doc text from SHARED_CORPUS, truncated to `max` chars.
fn doc_snippet(doc_id: u32, max: usize) -> String {
    SHARED_CORPUS
        .iter()
        .find(|(id, _)| *id == doc_id)
        .map(|(_, text)| {
            let s: String = text.chars().take(max).collect();
            if text.len() > max { format!("{s}…") } else { s }
        })
        .unwrap_or_else(|| format!("doc {doc_id}"))
}

async fn sleep_ms(ms: u64) {
    if ms > 0 {
        tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
    }
}

fn flush() {
    std::io::stdout().flush().ok();
}

// ─── Public entry points ─────────────────────────────────────────────────────

pub async fn render_trace(log: &TraceLog, delay_ms: u64) {
    let events: Vec<&TraceEvent> = log.events.iter().map(|(_, e)| e).collect();
    let mut i = 0;
    while i < events.len() {
        match events[i] {
            TraceEvent::TailHandle { .. } => {
                let run: Vec<&TraceEvent> = events[i..]
                    .iter()
                    .copied()
                    .take_while(|e| matches!(e, TraceEvent::TailHandle { .. }))
                    .collect();
                render_tail_phase(&run, delay_ms).await;
                i += run.len();
            }
            TraceEvent::DampedScore { .. } => {
                let run: Vec<&TraceEvent> = events[i..]
                    .iter()
                    .copied()
                    .take_while(|e| matches!(e, TraceEvent::DampedScore { .. }))
                    .collect();
                render_damped_phase(&run, delay_ms).await;
                i += run.len();
            }
            TraceEvent::BudgetBound { .. } => {
                let bound_run: Vec<&TraceEvent> = events[i..]
                    .iter()
                    .copied()
                    .take_while(|e| matches!(e, TraceEvent::BudgetBound { .. }))
                    .collect();
                let j = i + bound_run.len();
                let reconcile = events
                    .get(j)
                    .copied()
                    .filter(|e| matches!(e, TraceEvent::BudgetReconcile { .. }));
                render_budget_phase(&bound_run, reconcile, delay_ms).await;
                i = j + if reconcile.is_some() { 1 } else { 0 };
            }
            _ => {
                render_event(events[i], delay_ms).await;
                i += 1;
            }
        }
    }
}

// ─── Phase batch renderers ────────────────────────────────────────────────────

async fn render_tail_phase(events: &[&TraceEvent], delay_ms: u64) {
    // Sort by freq desc
    let mut rows: Vec<(&str, u32, &TailClass)> = events
        .iter()
        .filter_map(|e| {
            if let TraceEvent::TailHandle { token_type, count, classification } = e {
                Some((token_type.as_str(), *count, classification))
            } else {
                None
            }
        })
        .collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1));

    let max_freq = rows.first().map(|r| r.1).unwrap_or(1).max(1);
    let n_tail = rows.iter().filter(|r| matches!(r.2, TailClass::Tail)).count();
    let n_normal = rows.iter().filter(|r| matches!(r.2, TailClass::Normal)).count();
    let n_heavy = rows.iter().filter(|r| matches!(r.2, TailClass::Heavy)).count();

    println!("{CYAN} ┌─ Phase 1 · Tail Classification ─────────────────────────────┐{RESET}");
    println!("{CYAN} │{RESET}  Thresholds: TAIL < μ=128  ≤  NORMAL < τ=256  ≤  HEAVY       {CYAN}│{RESET}");
    println!("{CYAN} │{RESET}                                                              {CYAN}│{RESET}");
    println!("{CYAN} │{RESET}  {DIM}Token        Freq   ── freq / τ ──────────  Class{RESET}         {CYAN}│{RESET}");

    let show = rows.len().min(15);
    for (tok, count, class) in &rows[..show] {
        let b = bar(*count as f32, 256.0, 20);
        let class_str = match class {
            TailClass::Tail   => format!("{DIM}TAIL{RESET}  "),
            TailClass::Normal => format!("{CYAN}NORMAL{RESET}"),
            TailClass::Heavy  => format!("{YELLOW}HEAVY{RESET} "),
        };
        let t = trunc(tok, 10);
        println!("{CYAN} │{RESET}  {t:<12} {count:<5}  {DIM}{b}{RESET}  {class_str}  {CYAN}│{RESET}");
        sleep_ms(if delay_ms == 0 { 0 } else { 35 }).await;
    }
    if rows.len() > show {
        println!("{CYAN} │{RESET}  {DIM}… +{} more token types{RESET}                                      {CYAN}│{RESET}", rows.len() - show);
    }
    println!("{CYAN} │{RESET}                                                              {CYAN}│{RESET}");
    println!("{CYAN} │{RESET}  {DIM}TAIL:{RESET} {n_tail}   {CYAN}NORMAL:{RESET} {n_normal}   {YELLOW}HEAVY:{RESET} {n_heavy}                          {CYAN}│{RESET}");
    println!("{CYAN} └──────────────────────────────────────────────────────────────┘{RESET}");
    flush();
    sleep_ms(delay_ms).await;
    let _ = max_freq;
}

async fn render_damped_phase(events: &[&TraceEvent], delay_ms: u64) {
    let mut rows: Vec<(&str, f32, f32)> = events
        .iter()
        .filter_map(|e| {
            if let TraceEvent::DampedScore { token_type, variance, weight } = e {
                Some((token_type.as_str(), *variance, *weight))
            } else {
                None
            }
        })
        .collect();
    rows.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    let max_w = rows.first().map(|r| r.2).unwrap_or(1.0).max(1e-6);

    println!("{CYAN} ┌─ Phase 2 · Damped Scoring  wⱼ = √nⱼ · sⱼ ────────────────────┐{RESET}");
    println!("{CYAN} │{RESET}  High-variance types earn more centroid budget.               {CYAN}│{RESET}");
    println!("{CYAN} │{RESET}                                                              {CYAN}│{RESET}");
    println!("{CYAN} │{RESET}  {DIM}Token        var     wⱼ     ── wⱼ / max(w) ──{RESET}             {CYAN}│{RESET}");

    let show = rows.len().min(15);
    for (tok, var, w) in &rows[..show] {
        let b = bar(*w, max_w, 20);
        let t = trunc(tok, 10);
        println!("{CYAN} │{RESET}  {t:<12} {var:.3}   {BOLD}{w:.3}{RESET}  {DIM}{b}{RESET}  {CYAN}│{RESET}");
        sleep_ms(if delay_ms == 0 { 0 } else { 35 }).await;
    }
    if rows.len() > show {
        println!("{CYAN} │{RESET}  {DIM}… +{} more{RESET}                                                  {CYAN}│{RESET}", rows.len() - show);
    }
    println!("{CYAN} └──────────────────────────────────────────────────────────────┘{RESET}");
    flush();
    sleep_ms(delay_ms).await;
}

async fn render_budget_phase(
    bound_events: &[&TraceEvent],
    reconcile: Option<&TraceEvent>,
    delay_ms: u64,
) {
    let mut rows: Vec<(&str, f32, u32, u32, u32)> = bound_events
        .iter()
        .filter_map(|e| {
            if let TraceEvent::BudgetBound { token_type, raw_kappa, floored, ceiled, final_kappa } = e {
                Some((token_type.as_str(), *raw_kappa, *floored, *ceiled, *final_kappa))
            } else {
                None
            }
        })
        .collect();
    rows.sort_by(|a, b| b.4.cmp(&a.4));

    let theta = 39u32;

    println!("{CYAN} ┌─ Phase 3+4 · Budget Allocation  B=200, ε=4, θ=39 ─────────────┐{RESET}");
    println!("{CYAN} │{RESET}  raw κⱼ = B · wⱼ/Σw,  then clamp to [ε=4, θ=39]              {CYAN}│{RESET}");
    println!("{CYAN} │{RESET}                                                              {CYAN}│{RESET}");
    println!("{CYAN} │{RESET}  {DIM}Token        raw_κ  →  κⱼ   ── κⱼ / θ ────────{RESET}           {CYAN}│{RESET}");

    let show = rows.len().min(15);
    for (tok, raw, _fl, _ce, kappa) in &rows[..show] {
        let b = bar(*kappa as f32, theta as f32, 20);
        let flag = if *kappa == theta {
            format!("{YELLOW}▲CAPPED{RESET}")
        } else if *kappa <= 4 {
            format!("{DIM}▼FLOOR {RESET}")
        } else {
            "       ".to_string()
        };
        let t = trunc(tok, 10);
        println!("{CYAN} │{RESET}  {t:<12} {raw:5.1} → {BOLD}{kappa:>2}{RESET}   {DIM}{b}{RESET}  {flag}  {CYAN}│{RESET}");
        sleep_ms(if delay_ms == 0 { 0 } else { 35 }).await;
    }
    if rows.len() > show {
        println!("{CYAN} │{RESET}  {DIM}… +{} more{RESET}                                                  {CYAN}│{RESET}", rows.len() - show);
    }

    if let Some(TraceEvent::BudgetReconcile { total_budget, allocated, redistributed }) = reconcile {
        println!("{CYAN} │{RESET}                                                              {CYAN}│{RESET}");
        let b = bar(*allocated as f32, *total_budget as f32, 24);
        println!("{CYAN} │{RESET}  {DIM}allocated:{RESET} {BOLD}{allocated}{RESET}/{total_budget}  {DIM}{b}{RESET}  redistributed: {redistributed}  {CYAN}│{RESET}");
    }
    println!("{CYAN} └──────────────────────────────────────────────────────────────┘{RESET}");
    flush();
    sleep_ms(delay_ms).await;
}

// ─── Per-event renderers ──────────────────────────────────────────────────────

async fn render_event(event: &TraceEvent, delay_ms: u64) {
    match event {
        TraceEvent::HnswInsert { doc_id, doc_text, layers } => {
            let snippet = trunc(doc_text, 56);
            let promoted = layers.len() > 1;
            let layer_desc = if promoted {
                let ls: Vec<String> = layers.iter().map(|(l, _)| format!("L{l}")).collect();
                format!("★ Promoted → {}", ls.join(" + "))
            } else {
                format!("Layer {}", layers.first().map(|(l, _)| *l).unwrap_or(0))
            };

            println!("{CYAN} ┌─ Insert doc {doc_id} · {layer_desc} ──────────────────────────────────┐{RESET}");
            println!("{CYAN} │{RESET}  \"{snippet}\"");
            println!("{CYAN} │{RESET}  {DIM}↳ 1024-dim Voyage embedding → single point in vector space{RESET}");
            println!("{CYAN} │{RESET}");

            for (l, nbrs) in layers.iter() {
                let (role, role_note) = if *l == 0 {
                    ("fine-grained", "dense proximity — final recall layer")
                } else {
                    ("highway", "sparse long-range — skip ahead at query time")
                };
                if nbrs.is_empty() {
                    println!("{CYAN} │{RESET}  {DIM}L{l} {role}:{RESET}  {GREEN}first node on this layer{RESET}  {DIM}({role_note}){RESET}");
                } else {
                    let edge_str: String = nbrs.iter().take(6)
                        .map(|n| format!("[{n}]"))
                        .collect::<Vec<_>>()
                        .join("─");
                    let more = if nbrs.len() > 6 { format!(" +{} more", nbrs.len() - 6) } else { String::new() };
                    println!("{CYAN} │{RESET}  {DIM}L{l} {role}:{RESET}  {edge_str}─[{doc_id}]{more}  {DIM}({role_note}){RESET}");
                    for n in nbrs.iter().take(2) {
                        let ns = doc_snippet(*n, 44);
                        println!("{CYAN} │{RESET}    {DIM}└ [{n}]{RESET} \"{ns}\"");
                    }
                }
            }

            if promoted {
                println!("{CYAN} │{RESET}");
                println!("{CYAN} │{RESET}  {YELLOW}★ Promoted to {}/{} layers — future queries will enter here{RESET}", layers.len(), layers.len());
                println!("{CYAN} │{RESET}  {DIM}  High-layer nodes are \"skyscrapers\": visible from far away,{RESET}");
                println!("{CYAN} │{RESET}  {DIM}  enabling long jumps before descending to the dense L0 graph.{RESET}");
            }
            println!("{CYAN} └────────────────────────────────────────────────────────────────┘{RESET}");
            flush();
            sleep_ms(delay_ms).await;
        }

        TraceEvent::HnswQuery { layer, entry_doc, candidates, greedy_best } => {
            let (role, role_note) = if *layer == 0 {
                ("L0 Fine-grained", "dense layer — pick best neighbors here")
            } else {
                ("Highway layer", "sparse — skip large distances, then descend")
            };
            let entry_str = if *entry_doc == u32::MAX {
                format!("{DIM}query vector (top of graph — starting point){RESET}")
            } else {
                let s = doc_snippet(*entry_doc, 36);
                format!("{BOLD}doc {entry_doc}{RESET} · \"{s}\"")
            };
            let max_score = candidates.iter().map(|(_, s)| *s).fold(f32::NEG_INFINITY, f32::max).max(0.01);

            println!("{CYAN} ┌─ Query · {role} · {role_note} ─────────────────────────────┐{RESET}");
            println!("{CYAN} │{RESET}  Entry: {entry_str}");
            println!("{CYAN} │{RESET}  Scoring {BOLD}{}{RESET} neighbor{} — greedily move to the highest-scoring one:",
                candidates.len(),
                if candidates.len() == 1 { "" } else { "s" });
            for (id, score) in candidates.iter().take(10) {
                let b = bar(*score, max_score, 14);
                let snippet = doc_snippet(*id, 36);
                let marker = if *id == *greedy_best {
                    format!(" {GREEN}← best{RESET}")
                } else {
                    String::new()
                };
                println!("{CYAN} │{RESET}    {BOLD}{score:+.3}{RESET}  {DIM}{b}{RESET}  [{id}] \"{snippet}\"{marker}");
                sleep_ms(if delay_ms == 0 { 0 } else { 40 }).await;
            }
            if candidates.len() > 10 {
                println!("{CYAN} │{RESET}    {DIM}… +{} more{RESET}", candidates.len() - 10);
            }
            println!("{CYAN} │{RESET}");
            if *layer == 0 {
                let best_s = doc_snippet(*greedy_best, 38);
                println!("{CYAN} │{RESET}  {GREEN}─→ Top result: doc {greedy_best}{RESET} · \"{best_s}\"");
                println!("{CYAN} │{RESET}  {DIM}   (all remaining candidates returned as top-k results){RESET}");
            } else {
                let best_s = doc_snippet(*greedy_best, 38);
                println!("{CYAN} │{RESET}  {GREEN}─→ Move to doc {greedy_best}{RESET} · \"{best_s}\"");
                println!("{CYAN} │{RESET}  {DIM}   (descend to next layer from this node){RESET}");
            }
            println!("{CYAN} └────────────────────────────────────────────────────────────────┘{RESET}");
            flush();
            sleep_ms(delay_ms).await;
        }

        TraceEvent::HnswLayerStats { layer, node_count, avg_degree } => {
            let role = if *layer == 0 { "dense — every node" } else { "highway — promoted nodes" };
            let b = bar(*node_count as f32, 20.0, 20);
            let deg_b = bar(*avg_degree as f32, 4.0, 8);
            println!("{CYAN} ┌─ Layer {layer} · {role} ──────────────────────────────────────────┐{RESET}");
            println!("{CYAN} │{RESET}  {BOLD}{node_count}{RESET} nodes  {DIM}{b}{RESET}  {DIM}(of 20 total){RESET}");
            println!("{CYAN} │{RESET}  avg {BOLD}{avg_degree:.1}{RESET} neighbors/node  {DIM}{deg_b}{RESET}  {DIM}(max M=4){RESET}");
            println!("{CYAN} └────────────────────────────────────────────────────────────────┘{RESET}");
            flush();
            sleep_ms(if delay_ms == 0 { 0 } else { 60 }).await;
        }

        TraceEvent::Tokenize { doc_id, tokens } => {
            let n = tokens.len();
            println!("{CYAN} ┌─ Tokenize doc {doc_id} ─────────────────────────────────────────┐{RESET}");
            let snippet = doc_snippet(*doc_id, 55);
            println!("{CYAN} │{RESET}  \"{snippet}\"");
            println!("{CYAN} │{RESET}  {BOLD}{n} tokens:{RESET}");
            let show = tokens.len().min(8);
            for (i, tok) in tokens[..show].iter().enumerate() {
                println!("{CYAN} │{RESET}    [{i}] {tok}");
            }
            if n > 8 {
                println!("{CYAN} │{RESET}    {DIM}... +{} more{RESET}", n - 8);
            }
            println!("{CYAN} └───────────────────────────────────────────────────────────┘{RESET}");
            flush();
            sleep_ms(delay_ms).await;
        }

        TraceEvent::TokenEmbed { doc_id: _, token, embedding_preview } => {
            let tok = trunc(token, 12);
            let e = embedding_preview;
            println!("  {tok:<14} → [ {}{:+.3}{RESET}  {:+.3}  {:+.3}  {DIM}… ] (128-dim){RESET}",
                BOLD, e[0], e[1], e[2]);
            flush();
            sleep_ms(if delay_ms == 0 { 0 } else { 30 }).await;
        }

        TraceEvent::MaxSimMatrix { query_tokens, doc_tokens, doc_id, matrix, row_maxima, score } => {
            let max_q = 6usize;
            let max_d = 6usize;
            let n_q = query_tokens.len().min(max_q);
            let n_d = matrix.first().map(|r| r.len()).unwrap_or(0).min(max_d);
            let doc_snippet_str = doc_snippet(*doc_id, 45);

            println!("{CYAN} ┌─ MaxSim: query × doc {doc_id} ─────────────────────────────────┐{RESET}");
            println!("{CYAN} │{RESET}  \"{doc_snippet_str}\"");
            println!("{CYAN} │{RESET}");

            let mut header = format!("{CYAN} │{RESET}  {:<10} │", "");
            for j in 0..n_d {
                let lbl = doc_tokens.get(j).map(|t| trunc(t, 7)).unwrap_or_default();
                header.push_str(&format!(" {DIM}{lbl:<7}{RESET}"));
            }
            println!("{header}");

            for i in 0..n_q {
                let qtok = trunc(&query_tokens[i], 10);
                let row_max = row_maxima.get(i).copied().unwrap_or(0.0);
                let best_j = matrix.get(i).and_then(|row| {
                    row.iter().take(n_d).enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(j, _)| j)
                });
                let best_tok = best_j.and_then(|j| doc_tokens.get(j)).map(|t| trunc(t, 8)).unwrap_or_default();
                let mut row_line = format!("{CYAN} │{RESET} {BOLD}{qtok:<10}{RESET} │");
                if let Some(row) = matrix.get(i) {
                    for j in 0..n_d {
                        let v = row.get(j).copied().unwrap_or(0.0);
                        row_line.push_str(&format!("    {}    ", intensity_block(v)));
                    }
                }
                row_line.push_str(&format!("│ {BOLD}{row_max:.2}{RESET} ← {DIM}\"{best_tok}\"{RESET}"));
                println!("{row_line}");
            }
            println!("{CYAN} │{RESET}   {DIM}──────────{RESET}");
            println!("{CYAN} │{RESET}                           {BOLD}Score = {score:.3}{RESET}");
            println!("{CYAN} └────────────────────────────────────────────────────────────┘{RESET}");
            flush();
            sleep_ms(delay_ms).await;
        }

        TraceEvent::CentroidAssign { doc_id: _, token, centroid_id } => {
            let tok = trunc(token, 12);
            println!("  {tok:<14} → {DIM}centroid {RESET}{BOLD}{centroid_id}{RESET}");
            flush();
            sleep_ms(if delay_ms == 0 { 0 } else { 40 }).await;
        }

        TraceEvent::CentroidAnn { query_token, top_centroids } => {
            let qtok = trunc(query_token, 20);
            println!("{CYAN} ┌─ Centroid ANN: \"{qtok}\" ──────────────────────────────────┐{RESET}");
            for (id, score) in top_centroids.iter().take(6) {
                let b = bar(*score, 1.0, 20);
                println!("{CYAN} │{RESET}    c{id:<5}  {DIM}{b}{RESET}  {score:.3}");
            }
            println!("{CYAN} └────────────────────────────────────────────────────────────┘{RESET}");
            flush();
            sleep_ms(delay_ms).await;
        }

        TraceEvent::CandidateExpand { centroid_ids, candidate_doc_ids, pruned_count } => {
            let n_c = candidate_doc_ids.len() as f32;
            let n_p = *pruned_count as f32;
            let total = n_c + n_p;
            let pct = if total > 0.0 { (n_p / total) * 100.0 } else { 0.0 };
            let b_keep = bar(n_c, total.max(1.0), 20);
            let b_prune = bar(n_p, total.max(1.0), 20);
            println!("{CYAN} ┌─ PLAID: Centroid Expansion ────────────────────────────────┐{RESET}");
            println!("{CYAN} │{RESET}  centroids activated: {BOLD}{}{RESET}", centroid_ids.len());
            println!("{CYAN} │{RESET}  kept:   {BOLD}{n_c:<4.0}{RESET}  {DIM}{b_keep}{RESET}");
            println!("{CYAN} │{RESET}  pruned: {BOLD}{n_p:<4.0}{RESET}  {DIM}{b_prune}{RESET}  {BOLD}{pct:.0}%{RESET} saved");
            println!("{CYAN} └────────────────────────────────────────────────────────────┘{RESET}");
            flush();
            sleep_ms(delay_ms).await;
        }

        TraceEvent::PlaidMaxSim { candidate_count, scored_count, top_k } => {
            let max_score = top_k.iter().map(|(_, s)| *s).fold(0.0f32, f32::max).max(0.01);
            println!("{CYAN} ┌─ PLAID MaxSim ─────────────────────────────────────────────┐{RESET}");
            println!("{CYAN} │{RESET}  scored {BOLD}{scored_count}{RESET} of {candidate_count} candidates");
            for (id, score) in top_k.iter().take(8) {
                let b = bar(*score, max_score, 16);
                let snippet = doc_snippet(*id, 35);
                println!("{CYAN} │{RESET}  {BOLD}{score:.3}{RESET}  {DIM}{b}{RESET}  \"{snippet}\"");
            }
            println!("{CYAN} └────────────────────────────────────────────────────────────┘{RESET}");
            flush();
            sleep_ms(delay_ms).await;
        }

        TraceEvent::XtrScore { query_token_id, query_token, token_scores } => {
            let mut sorted = token_scores.clone();
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let max_s = sorted.first().map(|(_, s)| *s).unwrap_or(1.0).max(0.01);
            let qtok = if query_token.is_empty() {
                format!("q-token {query_token_id}")
            } else {
                format!("\"{query_token}\" (q{query_token_id})")
            };
            println!("{CYAN} ┌─ Xtr: {qtok} ──────────────────────────────────────────┐{RESET}");
            for (id, score) in sorted.iter().take(6) {
                let b = bar(*score, max_s, 16);
                let snippet = doc_snippet(*id, 38);
                println!("{CYAN} │{RESET}  {BOLD}{score:.3}{RESET}  {DIM}{b}{RESET}  \"{snippet}\"");
            }
            println!("{CYAN} └────────────────────────────────────────────────────────────┘{RESET}");
            flush();
            sleep_ms(delay_ms).await;
        }

        TraceEvent::CandidateGather { gathered, fraction_promoted, .. } => {
            let pct = fraction_promoted * 100.0;
            let b = bar(*fraction_promoted, 1.0, 24);
            println!("{CYAN} ┌─ WARP: Candidate Gather ───────────────────────────────────┐{RESET}");
            println!("{CYAN} │{RESET}  {BOLD}{}{RESET} docs gathered  {DIM}{b}{RESET}  {BOLD}{pct:.0}%{RESET} of index", gathered.len());
            println!("{CYAN} └────────────────────────────────────────────────────────────┘{RESET}");
            flush();
            sleep_ms(delay_ms).await;
        }

        TraceEvent::MaxSimRefine { candidate_count, top_k } => {
            let max_score = top_k.iter().map(|(_, s)| *s).fold(0.0f32, f32::max).max(0.01);
            println!("{CYAN} ┌─ MaxSim Refine  ({candidate_count} candidates) ────────────────────────┐{RESET}");
            for (id, score) in top_k.iter().take(8) {
                let b = bar(*score, max_score, 16);
                let snippet = doc_snippet(*id, 35);
                println!("{CYAN} │{RESET}  {BOLD}{score:.3}{RESET}  {DIM}{b}{RESET}  \"{snippet}\"");
            }
            println!("{CYAN} └────────────────────────────────────────────────────────────┘{RESET}");
            flush();
            sleep_ms(delay_ms).await;
        }

        // Phase events rendered individually only when not in a batch run
        TraceEvent::TailHandle { token_type, count, classification } => {
            let b = bar(*count as f32, 256.0, 18);
            let cs = match classification {
                TailClass::Tail => "TAIL  ",
                TailClass::Normal => "NORMAL",
                TailClass::Heavy => "HEAVY ",
            };
            let t = trunc(token_type, 12);
            println!("  {t:<14}  freq={BOLD}{count:<5}{RESET}  {DIM}{b}{RESET}  {cs}");
            flush();
            sleep_ms(if delay_ms == 0 { 0 } else { 35 }).await;
        }

        TraceEvent::DampedScore { token_type, variance, weight } => {
            let b = bar(*weight, 5.0, 18);
            let t = trunc(token_type, 12);
            println!("  {t:<14}  var={variance:.3}  {BOLD}w={weight:.3}{RESET}  {DIM}{b}{RESET}");
            flush();
            sleep_ms(if delay_ms == 0 { 0 } else { 35 }).await;
        }

        TraceEvent::BudgetBound { token_type, raw_kappa, floored: _, ceiled: _, final_kappa } => {
            let b = bar(*final_kappa as f32, 39.0, 18);
            let t = trunc(token_type, 12);
            println!("  {t:<14}  raw={raw_kappa:.1} → {BOLD}κⱼ={final_kappa}{RESET}  {DIM}{b}{RESET}");
            flush();
            sleep_ms(if delay_ms == 0 { 0 } else { 35 }).await;
        }

        TraceEvent::BudgetReconcile { total_budget, allocated, redistributed } => {
            let b = bar(*allocated as f32, *total_budget as f32, 24);
            println!("{CYAN} ┌─ Budget Reconcile ─────────────────────────────────────────┐{RESET}");
            println!("{CYAN} │{RESET}  {allocated}/{total_budget} allocated  {DIM}{b}{RESET}  redistributed: {redistributed}");
            println!("{CYAN} └────────────────────────────────────────────────────────────┘{RESET}");
            flush();
            sleep_ms(delay_ms).await;
        }

        TraceEvent::PqInspect { level, dimensions, subquantizer_count, code_bits } => {
            println!("{CYAN} ┌─ PQ Level {level} ─────────────────────────────────────────────┐{RESET}");
            println!("{CYAN} │{RESET}  {dimensions}-dim  /  {subquantizer_count} subquantizers  /  {code_bits}-bit codes");
            println!("{CYAN} └────────────────────────────────────────────────────────────┘{RESET}");
            flush();
            sleep_ms(delay_ms).await;
        }

        TraceEvent::TachiomSearch { timings } => {
            println!("{CYAN} ┌─ TACHIOM Search ───────────────────────────────────────────┐{RESET}");
            println!("{CYAN} │{RESET}  gather: {BOLD}{:.1}ms{RESET}   refine: {BOLD}{:.1}ms{RESET}   total: {BOLD}{:.1}ms{RESET}",
                timings.gather_ms, timings.refine_ms, timings.total_ms);
            println!("{CYAN} └────────────────────────────────────────────────────────────┘{RESET}");
            flush();
            sleep_ms(delay_ms).await;
        }
    }
}
