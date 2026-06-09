/// (doc_id, text) pairs. doc_id is stable and referenced by scenario TOML.
pub const SHARED_CORPUS: &[(u32, &str)] = &[
    (
        0,
        "The river bank was slippery after the spring flood receded.",
    ),
    (1, "She opened a savings account at the bank downtown."),
    (2, "The logging truck carried a full trunk of oak timber."),
    (
        3,
        "He packed his winter clothes into the car trunk before the road trip.",
    ),
    (
        4,
        "The surgeon operated on the nerve trunk in the patient's lower back.",
    ),
    (
        5,
        "The hiking trail runs along the left bank of the Colorado River.",
    ),
    (
        6,
        "Interest rates at the central bank rose sharply this quarter.",
    ),
    (
        7,
        "The elephant wrapped its trunk around the tree to pull it down.",
    ),
    (
        8,
        "She wore a light cotton dress on the warm summer afternoon.",
    ),
    (
        9,
        "The physics lab measured the speed of light using interferometry.",
    ),
    (
        10,
        "The crane operator lowered the steel beam with precision.",
    ),
    (11, "The paper crane origami requires 25 precise folds."),
    (
        12,
        "Venture capital firms invested heavily in financial technology startups.",
    ),
    (
        13,
        "The geological fault line runs beneath the river delta.",
    ),
    (
        14,
        "He pitched the tent on the flat bank beside the stream.",
    ),
    (
        15,
        "The investment bank underwrote the government bond issuance.",
    ),
    (
        16,
        "The trunk road connects the capital city to the northern province.",
    ),
    (
        17,
        "Scientists detected gravitational waves using laser light pulses.",
    ),
    (
        18,
        "A flock of cranes migrated south along the river valley.",
    ),
    (
        19,
        "The reserve bank adjusted monetary policy after the inflation report.",
    ),
];

/// Ground truth relevance sets for the two canonical demo queries.
///
/// These define which documents are genuinely relevant for each query's intended meaning.
/// "bank" is intentionally ambiguous — two distinct meanings share the same word.
/// The demo shows how each engine handles this ambiguity differently.
pub const DEMO_GROUND_TRUTH: &[(&str, &str, &[u32])] = &[
    // (query, description of what "relevant" means, relevant doc_ids)
    (
        "river erosion along the bank",
        "river / waterway / outdoor",
        &[0, 5, 14, 13, 18],
        // doc 0:  "The river bank was slippery after the spring flood receded."
        // doc 5:  "The hiking trail runs along the left bank of the Colorado River."
        // doc 14: "He pitched the tent on the flat bank beside the stream."
        // doc 13: "The geological fault line runs beneath the river delta."
        // doc 18: "A flock of cranes migrated south along the river valley."
    ),
    (
        "open a checking account at the bank",
        "financial institution / banking",
        &[1, 6, 15, 19, 12],
        // doc 1:  "She opened a savings account at the bank downtown."
        // doc 6:  "Interest rates at the central bank rose sharply this quarter."
        // doc 15: "The investment bank underwrote the government bond issuance."
        // doc 19: "The reserve bank adjusted monetary policy after the inflation report."
        // doc 12: "Venture capital firms invested heavily in financial technology startups."
    ),
    (
        "bank interest rate",
        "financial institution / banking",
        &[6, 19, 15, 1, 12],
    ),
    (
        "river bank hiking",
        "river / waterway / outdoor",
        &[0, 5, 14, 13, 18],
    ),
];

/// Return the ground-truth relevant doc IDs for a query, if known.
pub fn relevant_docs_for(query: &str) -> Option<&'static [u32]> {
    DEMO_GROUND_TRUTH
        .iter()
        .find(|(q, _, _)| *q == query)
        .map(|(_, _, docs)| *docs)
}

/// Return the relevance description label for a query, if known.
pub fn relevance_label_for(query: &str) -> Option<&'static str> {
    DEMO_GROUND_TRUTH
        .iter()
        .find(|(q, _, _)| *q == query)
        .map(|(_, label, _)| *label)
}

/// Ground-truth top-1 document per query (used by all verify modules).
pub const VERIFY_QUERIES: &[(&str, u32)] = &[
    ("river erosion along the bank", 0),
    ("open a checking account at the bank", 1),
    ("lumber loaded on a logging truck", 2),
    ("packing luggage into the car before travel", 3),
    ("neural trunk anatomy in spinal surgery", 4),
    ("hiking trail beside a river", 5),
    ("central bank interest rate decision", 6),
    ("elephant using its trunk", 7),
    ("summer fashion lightweight clothing", 8),
    ("speed of light measurement experiment", 9),
    ("construction crane lifting steel", 10),
    ("paper folding origami bird", 11),
    ("fintech startup venture funding", 12),
    ("geological fault beneath river delta", 13),
    ("tent camping beside a stream", 14),
    ("bond underwriting investment banking", 15),
    ("arterial road connecting cities", 16),
    ("laser light pulse experiment", 17),
    ("bird migration along river valley", 18),
    ("monetary policy inflation central bank", 19),
];
