//! M6 SQL acceptance: batched temporal-graph tables.
//!
//! The pointwise `neighbors(src,t)` / `khop(src,k,t)` UDTFs accept literals
//! only, so a batch of seeds used to cost N parse+plan+collect cycles. The new
//! `gtv_batch_neighbors` / `gtv_batch_khop` tables are ordinary relations:
//! one planning cycle, joinable, projectable, filterable.
//!
//! Every test below compares the batch answer **row-for-row** against the
//! pointwise UDTFs — the kernel (`csr.rs`, `endpoint.rs`, GTVSeg v1) is not
//! modified, so the two paths must be indistinguishable apart from speed.
//!
//! Data discipline: all graphs are built in memory; no file is written.

use std::sync::Arc;

use arrow::array::{
    ArrayRef, Int64Array, RecordBatch, TimestampNanosecondArray, UInt16Array, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use gtv_core::TemporalCSR;
use gtv_engine::GtvContext;

const T: i64 = 100;
const K: usize = 2;
const SEEDS: &[u64] = &[0, 1, 3, 7, 11];

/// 12 nodes, finite intervals, open-ended relationships and expired edges.
/// `valid_at = 100` exercises the half-open `[valid_from, valid_to)` rule.
fn small_graph() -> TemporalCSR {
    let edges: Vec<(u64, u64, i64, i64, u16)> = vec![
        (0, 1, 0, 100, 0),       // expired at T (half-open)
        (0, 2, 0, i64::MAX, 1),  // enduring
        (0, 3, 50, 150, 2),
        (1, 4, 0, 200, 0),
        (1, 5, 100, 300, 1),
        (2, 9, 0, i64::MAX, 0),
        (3, 6, 0, i64::MAX, 2),
        (3, 7, 10, 80, 0),       // expired at T
        (5, 10, 0, i64::MAX, 0),
        (7, 8, 0, i64::MAX, 1),
        (11, 0, 0, i64::MAX, 0),
        (8, 11, 0, i64::MAX, 2),
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
        12,
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

type NeighborRow = (u64, u64, u16, i64, i64);
type KhopRow = (u64, u64, u64);

async fn sql_neighbors(ctx: &GtvContext, sql: &str) -> Vec<NeighborRow> {
    let mut out = Vec::new();
    for b in ctx.sql(sql).await.unwrap() {
        let (s, d, et, vf, vt) = (
            col_u64(&b, 0),
            col_u64(&b, 1),
            col_u16(&b, 2),
            col_i64(&b, 3),
            col_i64(&b, 4),
        );
        for i in 0..b.num_rows() {
            out.push((s[i], d[i], et[i], vf[i], vt[i]));
        }
    }
    out.sort_unstable();
    out
}

async fn sql_khop(ctx: &GtvContext, sql: &str) -> Vec<KhopRow> {
    let mut out = Vec::new();
    for b in ctx.sql(sql).await.unwrap() {
        let (s, h, d) = (col_u64(&b, 0), col_u64(&b, 1), col_u64(&b, 2));
        for i in 0..b.num_rows() {
            out.push((s[i], h[i], d[i]));
        }
    }
    out.sort_unstable();
    out
}

async fn pointwise_neighbors(ctx: &GtvContext, seeds: &[u64], t: i64) -> Vec<NeighborRow> {
    let mut out = Vec::new();
    for &s in seeds {
        let sql = format!(
            "SELECT src, dst, edge_type, valid_from, valid_to FROM neighbors({s}, {t})"
        );
        out.extend(sql_neighbors(ctx, &sql).await);
    }
    out.sort_unstable();
    out
}

async fn pointwise_khop(ctx: &GtvContext, seeds: &[u64], k: usize, t: i64) -> Vec<KhopRow> {
    let mut out = Vec::new();
    for &s in seeds {
        let sql = format!("SELECT hop, dst FROM khop({s}, {k}, {t})");
        for b in ctx.sql(&sql).await.unwrap() {
            let (h, d) = (col_u64(&b, 0), col_u64(&b, 1));
            for i in 0..b.num_rows() {
                out.push((s, h[i], d[i]));
            }
        }
    }
    out.sort_unstable();
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn batch_neighbors_matches_pointwise_udtf_row_for_row() {
    let csr = small_graph();
    let ctx = GtvContext::new();
    ctx.register_neighbors(&csr);
    ctx.register_batch_graph(&csr, SEEDS, T, K).unwrap();

    let pointwise = pointwise_neighbors(&ctx, SEEDS, T).await;
    let batched = sql_neighbors(&ctx, "SELECT * FROM gtv_batch_neighbors").await;
    assert!(!pointwise.is_empty(), "fixture must produce neighbours");
    assert_eq!(batched, pointwise);

    // The time predicate is the kernel's, not a SQL-level filter: a different
    // snapshot must agree too (and must differ from the first one).
    ctx.register_batch_graph(&csr, SEEDS, 20, K).unwrap();
    let early = sql_neighbors(&ctx, "SELECT * FROM gtv_batch_neighbors").await;
    let early_pointwise = pointwise_neighbors(&ctx, SEEDS, 20).await;
    assert_eq!(early, early_pointwise);
    assert_ne!(early, pointwise, "valid_at must actually filter");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn batch_khop_matches_pointwise_udtf_row_for_row() {
    let csr = small_graph();
    let ctx = GtvContext::new();
    ctx.register_neighbors(&csr);
    ctx.register_batch_graph(&csr, SEEDS, T, K).unwrap();

    let pointwise = pointwise_khop(&ctx, SEEDS, K, T).await;
    let batched = sql_khop(&ctx, "SELECT seed, hop, dst FROM gtv_batch_khop").await;
    assert!(!pointwise.is_empty(), "fixture must reach nodes");
    assert_eq!(batched, pointwise);

    // Every row must belong to a requested seed, and hops must be 1..=K.
    for &(s, h, _) in &batched {
        assert!(SEEDS.contains(&s));
        assert!((1..=K as u64).contains(&h));
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shared_and_owned_registration_agree() {
    let csr = Arc::new(small_graph());
    let ctx = GtvContext::new();

    ctx.register_batch_graph(csr.as_ref(), SEEDS, T, K).unwrap();
    let owned_n = sql_neighbors(&ctx, "SELECT * FROM gtv_batch_neighbors").await;
    let owned_k = sql_khop(&ctx, "SELECT * FROM gtv_batch_khop").await;

    ctx.register_batch_graph_shared(&csr, SEEDS, T, K).unwrap();
    let shared_n = sql_neighbors(&ctx, "SELECT * FROM gtv_batch_neighbors").await;
    let shared_k = sql_khop(&ctx, "SELECT * FROM gtv_batch_khop").await;

    assert_eq!(owned_n, shared_n);
    assert_eq!(owned_k, shared_k);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reregistering_replaces_the_previous_seed_set() {
    let csr = small_graph();
    let ctx = GtvContext::new();
    ctx.register_batch_graph(&csr, SEEDS, T, K).unwrap();
    let before = sql_neighbors(&ctx, "SELECT * FROM gtv_batch_neighbors").await;

    let smaller = [0u64];
    ctx.register_batch_graph(&csr, &smaller, T, K).unwrap();
    let after_n = sql_neighbors(&ctx, "SELECT * FROM gtv_batch_neighbors").await;
    let after_k = sql_khop(&ctx, "SELECT * FROM gtv_batch_khop").await;

    assert!(after_n.len() < before.len());
    assert!(after_n.iter().all(|r| r.0 == 0));
    assert!(after_k.iter().all(|r| r.0 == 0));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn batch_table_joins_with_an_ordinary_sql_relation() {
    let csr = small_graph();
    let ctx = GtvContext::new();
    ctx.register_batch_graph(&csr, &[0, 3], T, K).unwrap();

    let schema = Arc::new(Schema::new(vec![
        Field::new("case_id", DataType::UInt64, false),
        Field::new("account_id", DataType::UInt64, false),
    ]));
    let cases = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(vec![101u64, 102])) as ArrayRef,
            Arc::new(UInt64Array::from(vec![0u64, 3])) as ArrayRef,
        ],
    )
    .unwrap();
    ctx.register_batches("cases", schema, vec![cases]).unwrap();

    // A literal-only UDTF cannot appear on the right side of this join.
    let out = ctx
        .sql(
            "SELECT c.case_id, b.dst FROM cases c \
             JOIN gtv_batch_neighbors b ON b.src = c.account_id \
             ORDER BY c.case_id, b.dst",
        )
        .await
        .unwrap();
    let mut rows = Vec::new();
    for b in out {
        let (cid, dst) = (col_u64(&b, 0), col_u64(&b, 1));
        for i in 0..b.num_rows() {
            rows.push((cid[i], dst[i]));
        }
    }
    assert_eq!(
        rows,
        vec![(101, 2), (101, 3), (102, 6)],
        "join must see exactly the active neighbours at T=100"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn projection_and_filter_are_forwarded_to_the_memory_source() {
    let csr = small_graph();
    let ctx = GtvContext::new();
    ctx.register_batch_graph(&csr, SEEDS, T, K).unwrap();

    let out = ctx
        .sql("SELECT dst FROM gtv_batch_neighbors WHERE dst >= 6 ORDER BY dst")
        .await
        .unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].num_columns(), 1, "projection must narrow to one column");
    let dst = col_u64(&out[0], 0);
    assert!(dst.windows(2).all(|w| w[0] <= w[1]));
    assert!(dst.iter().all(|&d| d >= 6));

    let hop = ctx
        .sql("SELECT seed, dst FROM gtv_batch_khop WHERE hop = 1")
        .await
        .unwrap();
    assert!(hop.iter().all(|b| b.num_columns() == 2));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn empty_seed_set_yields_typed_empty_tables() {
    let csr = small_graph();
    let ctx = GtvContext::new();
    ctx.register_batch_graph(&csr, &[], T, K).unwrap();

    for table in ["gtv_batch_neighbors", "gtv_batch_khop"] {
        let out = ctx.sql(&format!("SELECT * FROM {table}")).await.unwrap();
        let rows: usize = out.iter().map(|b| b.num_rows()).sum();
        assert_eq!(rows, 0, "{table} must be empty");
        let expected = if table == "gtv_batch_neighbors" { 5 } else { 3 };
        assert_eq!(out[0].num_columns(), expected);
    }
}
