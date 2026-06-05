use std::collections::HashSet;

pub fn recall_at_k(retrieved: &[u32], ground_truth: &[u32], k: usize) -> f32 {
    let retrieved_k: HashSet<_> = retrieved.iter().take(k).collect();
    let relevant: HashSet<_> = ground_truth.iter().take(k).collect();
    let intersection = retrieved_k.intersection(&relevant).count();
    intersection as f32 / k.min(relevant.len()) as f32
}

pub fn latency_percentiles(latencies_ms: &mut [f32]) -> (f32, f32, f32) {
    latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = latencies_ms.len();
    let p50 = latencies_ms[n / 2];
    let p95 = latencies_ms[(n as f32 * 0.95) as usize];
    let p99 = latencies_ms[(n as f32 * 0.99) as usize];
    (p50, p95, p99)
}
