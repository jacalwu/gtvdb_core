//! P1-4 SQL acceptance: `gtv-pattern` temporal motifs have a SQL exit.
//!
//! Before this change `gtv-engine` declared `gtv-pattern` as a dependency but
//! never referenced it, so AML wash-trade rings could only be matched in Rust.
//! The new UDTFs (`motif` / `motif_from`) and the joinable batch tables
//! (`gtv_motif_matches` / `gtv_motif_edges`) must be indistinguishable from
//! the Rust kernel apart from speed. Every test below compares row-for-row.
//!
//! Data discipline: all graphs are built in memory; no file is written.

use std::sync::Arc;

use arrow::array::{
    ArrayRef, Int64Array, RecordBatch, TimestampNanosecondArray, UInt16Array, UInt32Array,
    UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use gtv_core::TemporalCSR;
use gtv_engine::{
    find_matches, match_schema, GtvContext, MotifShape, MotifSpec,
};
use gtv_pattern::{Match, Pattern, WindowedMotif};

const T: i64 = 100;
const SEEDS: &[u64] = &[0, 1, 3, 7];

/// Three wash-trade rings plus expired / not-yet-valid edges. Event times are
/// strictly increasing along each ring so `Pattern::ring(3)` event orders hold.
/// The 0 -> 3 -> 4 -> 0 ring spans 50 time units, which the 25-unit rolling
/// window must reject.
fn ring_graph() -> TemporalCSR {
    let edges: Vec<(u64, u64, i64, i64, u16)> = vec![
        (0, 1, 10, 200, 1),
        (0, 3, 40, 200, 1),
        (1, 2, 20, 200, 2),
        (1, 5, 70, 200, 1),
        (2, 0, 30, 200, 3),
        (2, 7, 5, 50, 0), // expired at T
        (3, 4, 50, 200, 2),
        (4, 0, 90, 200, 3), // span 50 -> window rejects
        (5, 6, 80, 200, 2),
        (6, 1, 95, 200, 3), // span 25 -> window accepts (inclusive)
        (7, 8, 100, 200, 0),
        (8, 2, 110, 200, 0), // starts after T
    ];
    let src: Vec<u64> = edges.iter().map(|e| e.0).collect();
    let dst: Vec<u64> = edges.iter().map(|e| e.1).collect();
    let vf: Vec<i64> = edges.iter().map(|e| e.2).collect();
    let vt: Vec<i64> = edges.iter().map(|e| e.3).collect();
    let et: Vec<u16> = edges.iter().map(|e| e.4).collect();
    TemporalCSR::from_arrays(
        &UInt64Array::from(src),
        &UInt64Array::from(dst),
        &TimestampNanosecondArray::from(vf),
        &TimestampNanosecondArray::from(vt),
        &UInt16Array::from(et),
        9,
    )
    .unwrap()
}

fn col_u64(b: &RecordBatch, i: usize) -> Vec<u64> {
    b.column(i)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .values()
        .to_vec()
}

fn col_u32(b: &RecordBatch, i: usize) -> Vec<u32> {
    b.column(i)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .unwrap()
        .values()
        .to_vec()
}

fn col_u16(b: &RecordBatch, i: usize) -> Vec<u16> {
    b.column(i)
        .as_any()
        .downcast_ref::<UInt16Array>()
        .unwrap()
        .values()
        .to_vec()
}

fn col_i64(b: &RecordBatch, i: usize) -> Vec<i64> {
    b.column(i)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .values()
        .to_vec()
}

/// One match row, ignoring `match_id` (pointwise UDTFs restart it at zero).
type MatchRow = (Vec<u64>, i64, i64);

fn match_rows(batches: &[RecordBatch], num_vars: usize) -> Vec<MatchRow> {
    let mut out = Vec::new();
    for b in batches {
        assert_eq!(b.num_columns(), num_vars + 3, "match schema width");
        let nodes: Vec<Vec<u64>> = (0..num_vars).map(|v| col_u64(b, 1 + v)).collect();
        let emin = col_i64(b, 1 + num_vars);
        let emax = col_i64(b, 2 + num_vars);
        for i in 0..b.num_rows() {
            let ns = (0..num_vars).map(|v| nodes[v][i]).collect();
            out.push((ns, emin[i], emax[i]));
        }
    }
    out.sort_unstable();
    out
}

fn rust_rows(matches: &[Match], num_vars: usize) -> Vec<MatchRow> {
    let mut out: Vec<MatchRow> = matches
        .iter()
        .map(|m| {
            let mut ns = m.nodes.clone();
            ns.resize(num_vars, 0);
            let emin = m.edges.iter().map(|e| e.valid_from).min().unwrap_or(0);
            let emax = m.edges.iter().map(|e| e.valid_from).max().unwrap_or(0);
            (ns, emin, emax)
        })
        .collect();
    out.sort_unstable();
    out
}

async fn sql(ctx: &GtvContext, query: &str) -> Vec<RecordBatch> {
    ctx.sql(query).await.unwrap()
}

async fn pointwise_motif_from(ctx: &GtvContext, shape: &str, seeds: &[u64], limit: usize) -> Vec<MatchRow> {
    let nv = MotifShape::parse(shape).unwrap().num_vars();
    let mut out = Vec::new();
    for &s in seeds {
        let q = format!("SELECT * FROM motif_from('{shape}', {s}, {T}, {limit})");
        out.extend(match_rows(&sql(ctx, &q).await, nv));
    }
    out.sort_unstable();
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pointwise_udtf_matches_rust_find_from_row_for_row() {
    let csr = ring_graph();
    let ctx = GtvContext::new();
    ctx.register_motif(&csr);
    let pattern = Pattern::ring(3);

    for &s in SEEDS {
        let got = match_rows(
            &sql(
                &ctx,
                &format!("SELECT * FROM motif_from('ring3', {s}, {T}, 500)"),
            )
            .await,
            3,
        );
        let expected = rust_rows(
            &gtv_pattern::find_from(&csr, &pattern, s, T, 500).unwrap(),
            3,
        );
        assert_eq!(got, expected, "seed {s} must match the Rust kernel");
    }

    let all = pointwise_motif_from(&ctx, "ring3", SEEDS, 500).await;
    assert!(!all.is_empty(), "fixture must contain wash-trade rings");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn full_scan_udtf_matches_rust_find() {
    let csr = ring_graph();
    let ctx = GtvContext::new();
    ctx.register_motif(&csr);

    let got = match_rows(&sql(&ctx, "SELECT * FROM motif('ring3', 100, 500)").await, 3);
    let expected = rust_rows(&gtv_pattern::find(&csr, &Pattern::ring(3), T, 500).unwrap(), 3);
    assert!(!got.is_empty());
    assert_eq!(got, expected, "find() ring3 fast path must agree with SQL");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn batch_matches_equals_n_pointwise_udtf_calls() {
    let csr = ring_graph();
    let ctx = GtvContext::new();
    ctx.register_motif(&csr);
    let spec = MotifSpec::new(MotifShape::Ring(3), T).with_limit(500);
    ctx.register_motif_batch(&csr, &spec, SEEDS).unwrap();

    let batched = match_rows(&sql(&ctx, "SELECT * FROM gtv_motif_matches").await, 3);
    let pointwise = pointwise_motif_from(&ctx, "ring3", SEEDS, 500).await;
    assert!(!batched.is_empty());
    assert_eq!(batched, pointwise, "one batch must equal N pointwise UDTFs");

    let expected = rust_rows(&find_matches(&csr, &spec, Some(SEEDS)).unwrap(), 3);
    assert_eq!(batched, expected);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn edge_table_reconstructs_every_matched_edge() {
    let csr = ring_graph();
    let ctx = GtvContext::new();
    let spec = MotifSpec::new(MotifShape::Ring(3), T).with_limit(500);
    ctx.register_motif_batch(&csr, &spec, SEEDS).unwrap();

    let matches = find_matches(&csr, &spec, Some(SEEDS)).unwrap();
    let edge_batches = sql(&ctx, "SELECT * FROM gtv_motif_edges").await;

    let mut rebuilt: Vec<Vec<(u64, u32, u64, u64, u16, i64, i64)>> =
        vec![Vec::new(); matches.len()];
    for b in &edge_batches {
        let (id, idx, src, dst, et, vf, vt) = (
            col_u64(b, 0),
            col_u32(b, 1),
            col_u64(b, 2),
            col_u64(b, 3),
            col_u16(b, 4),
            col_i64(b, 5),
            col_i64(b, 6),
        );
        for i in 0..b.num_rows() {
            rebuilt[id[i] as usize].push((id[i], idx[i], src[i], dst[i], et[i], vf[i], vt[i]));
        }
    }
    for group in &mut rebuilt {
        group.sort_by_key(|r| r.1);
    }

    assert_eq!(rebuilt.len(), matches.len());
    for (i, m) in matches.iter().enumerate() {
        let expected: Vec<_> = m
            .edges
            .iter()
            .enumerate()
            .map(|(j, e)| {
                (
                    i as u64,
                    j as u32,
                    e.src,
                    e.dst,
                    e.edge_type,
                    e.valid_from,
                    e.valid_to,
                )
            })
            .collect();
        assert_eq!(rebuilt[i], expected, "match {i} edge projection");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rolling_window_matches_windowed_kernel_and_filters() {
    let csr = ring_graph();
    let ctx = GtvContext::new();
    ctx.register_motif(&csr);

    let plain = MotifSpec::new(MotifShape::Ring(3), T).with_limit(500);
    let windowed = plain.clone().with_window(25);
    ctx.register_motif_batch(&csr, &windowed, SEEDS).unwrap();

    let got = match_rows(&sql(&ctx, "SELECT * FROM gtv_motif_matches").await, 3);
    let motif = WindowedMotif::new(Pattern::ring(3)).with_window(25);
    let mut expected: Vec<Match> = Vec::new();
    for &s in SEEDS {
        let remaining = 500 - expected.len();
        expected.extend(gtv_pattern::find_windowed_from(&csr, &motif, s, T, remaining).unwrap());
    }
    assert_eq!(got, rust_rows(&expected, 3));

    let unwindowed = rust_rows(&find_matches(&csr, &plain, Some(SEEDS)).unwrap(), 3);
    assert!(
        got.len() < unwindowed.len(),
        "the 25-unit window must reject the 0->3->4->0 ring (span 50)"
    );
    for row in &got {
        assert!(row.2 - row.1 <= 25, "event span must respect the window");
        assert!(unwindowed.contains(row), "windowed rows must be a subset");
    }

    // The SQL UDTF carries the same optional window argument.
    let udtf = match_rows(
        &sql(&ctx, "SELECT * FROM motif_from('ring3', 0, 100, 500, 25)").await,
        3,
    );
    let via_kernel = rust_rows(
        &gtv_pattern::find_windowed_from(&csr, &motif, 0, T, 500).unwrap(),
        3,
    );
    assert_eq!(udtf, via_kernel);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn edge_type_sequence_and_shape_variants() {
    let csr = ring_graph();
    let ctx = GtvContext::new();

    // Our rings use types 1,2,3 in event-time order.
    let ok = MotifSpec::new(MotifShape::Ring(3), T)
        .with_limit(500)
        .with_sequence(vec![1, 2, 3]);
    ctx.register_motif_batch(&csr, &ok, SEEDS).unwrap();
    let rows = match_rows(&sql(&ctx, "SELECT * FROM gtv_motif_matches").await, 3);
    assert_eq!(rows, rust_rows(&find_matches(&csr, &ok, Some(SEEDS)).unwrap(), 3));
    assert!(!rows.is_empty(), "type sequence 1,2,3 must match");

    let rejected = ok.clone().with_sequence(vec![9, 9, 9]);
    ctx.register_motif_batch(&csr, &rejected, SEEDS).unwrap();
    let empty = sql(&ctx, "SELECT * FROM gtv_motif_matches").await;
    assert_eq!(empty.iter().map(|b| b.num_rows()).sum::<usize>(), 0);

    // Every built-in shape exposes a correctly sized SQL schema.
    for (name, nv) in [("path2", 3), ("path3", 4), ("ring4", 4), ("diamond", 4)] {
        let spec = MotifSpec::new(MotifShape::parse(name).unwrap(), T).with_limit(50);
        ctx.register_motif_batch(&csr, &spec, SEEDS).unwrap();
        let out = sql(&ctx, "SELECT * FROM gtv_motif_matches").await;
        assert_eq!(out[0].num_columns(), nv + 3, "{name} schema width");
        assert_eq!(match_schema(nv).fields().len(), nv + 3);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn batch_table_joins_projects_and_filters() {
    let csr = ring_graph();
    let ctx = GtvContext::new();
    ctx.register_motif(&csr);
    let spec = MotifSpec::new(MotifShape::Ring(3), T).with_limit(500);
    ctx.register_motif_batch(&csr, &spec, &[0]).unwrap();

    let schema = Arc::new(Schema::new(vec![
        Field::new("case_id", DataType::UInt64, false),
        Field::new("account_id", DataType::UInt64, false),
    ]));
    let cases = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(vec![101u64])) as ArrayRef,
            Arc::new(UInt64Array::from(vec![0u64])) as ArrayRef,
        ],
    )
    .unwrap();
    ctx.register_batches("cases", schema, vec![cases]).unwrap();

    let out = sql(
        &ctx,
        "SELECT c.case_id, m.n1, m.n2 FROM cases c \
         JOIN gtv_motif_matches m ON m.n0 = c.account_id \
         WHERE m.event_max - m.event_min <= 25 ORDER BY m.n1, m.n2",
    )
    .await;
    let mut rows = Vec::new();
    for b in &out {
        assert_eq!(b.num_columns(), 3, "projection must narrow");
        let (cid, n1, n2) = (col_u64(b, 0), col_u64(b, 1), col_u64(b, 2));
        for i in 0..b.num_rows() {
            rows.push((cid[i], n1[i], n2[i]));
        }
    }
    assert!(!rows.is_empty(), "seed 0 has a ring within the 25-unit span");
    assert!(rows.iter().all(|r| r.0 == 101));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn empty_seeds_zero_limit_and_reregistration() {
    let csr = ring_graph();
    let ctx = GtvContext::new();
    let spec = MotifSpec::new(MotifShape::Ring(3), T).with_limit(500);

    ctx.register_motif_batch(&csr, &spec, &[]).unwrap();
    for table in ["gtv_motif_matches", "gtv_motif_edges"] {
        let out = sql(&ctx, &format!("SELECT * FROM {table}")).await;
        assert_eq!(out.iter().map(|b| b.num_rows()).sum::<usize>(), 0);
        let expected = if table == "gtv_motif_matches" { 6 } else { 7 };
        assert_eq!(out[0].num_columns(), expected, "{table} typed empty");
    }

    ctx.register_motif_batch(&csr, &spec, SEEDS).unwrap();
    let before = match_rows(&sql(&ctx, "SELECT * FROM gtv_motif_matches").await, 3);
    assert!(!before.is_empty());

    ctx.register_motif_batch(&csr, &spec.clone().with_limit(0), SEEDS).unwrap();
    let after = sql(&ctx, "SELECT * FROM gtv_motif_matches").await;
    assert_eq!(after.iter().map(|b| b.num_rows()).sum::<usize>(), 0);

    // A shared graph image gives the same answer as the owned path.
    let shared = Arc::new(csr);
    ctx.register_motif_batch_shared(&shared, &spec, SEEDS).unwrap();
    let again = match_rows(&sql(&ctx, "SELECT * FROM gtv_motif_matches").await, 3);
    assert_eq!(again, before);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invalid_shape_is_rejected() {
    let csr = ring_graph();
    let ctx = GtvContext::new();
    ctx.register_motif(&csr);
    let err = ctx.sql("SELECT * FROM motif('not_a_shape', 100)").await;
    assert!(err.is_err(), "unknown shape must fail loudly");
}
