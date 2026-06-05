use common::trace::{TraceEvent, TraceLog};

pub struct GatherStats {
    pub gathered: Vec<u32>,
    pub fraction_promoted: f32,
}

pub fn gather_candidates(
    xtr_results: &[(u32, f32)],
    total_docs: usize,
) -> (Vec<u32>, GatherStats, TraceLog) {
    let mut log = TraceLog::default();

    let gathered: Vec<u32> = xtr_results.iter().map(|(id, _)| *id).collect();
    let fraction_promoted = if total_docs == 0 {
        0.0
    } else {
        gathered.len() as f32 / total_docs as f32
    };

    log.push(TraceEvent::CandidateGather {
        gathered: gathered.clone(),
        overlap_with_gt: fraction_promoted, // Used as a proxy here
        fraction_promoted,
    });

    let stats = GatherStats {
        gathered: gathered.clone(),
        fraction_promoted,
    };

    (gathered, stats, log)
}
