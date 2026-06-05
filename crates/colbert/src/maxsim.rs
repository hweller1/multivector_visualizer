use common::{TokenMatrix, TOKEN_DIM};

/// Dot product of two L2-normalized vectors = cosine similarity.
pub fn cosine(a: &[f32; TOKEN_DIM], b: &[f32; TOKEN_DIM]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// MaxSim(Q, D) = Σ_i max_j cosine(Q[i], D[j])
pub fn maxsim(query: &TokenMatrix, doc: &TokenMatrix) -> f32 {
    if query.rows.is_empty() || doc.rows.is_empty() {
        return 0.0;
    }
    query.rows.iter().map(|q_tok| {
        doc.rows.iter()
            .map(|d_tok| cosine(q_tok, d_tok))
            .fold(f32::NEG_INFINITY, f32::max)
    }).sum()
}

/// Returns (score, full matrix rows×cols, row_maxima) for MaxSimMatrix trace event.
pub fn maxsim_with_matrix(
    query: &TokenMatrix,
    doc: &TokenMatrix,
) -> (f32, Vec<Vec<f32>>, Vec<f32>) {
    if query.rows.is_empty() || doc.rows.is_empty() {
        return (0.0, vec![], vec![]);
    }
    let mut matrix: Vec<Vec<f32>> = Vec::with_capacity(query.rows.len());
    let mut row_maxima: Vec<f32> = Vec::with_capacity(query.rows.len());

    for q_tok in &query.rows {
        let row: Vec<f32> = doc.rows.iter().map(|d_tok| cosine(q_tok, d_tok)).collect();
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        row_maxima.push(max_val);
        matrix.push(row);
    }

    let score: f32 = row_maxima.iter().sum();
    (score, matrix, row_maxima)
}
