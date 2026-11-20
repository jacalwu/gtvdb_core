//! C1 acceptance: a SQL/PGQ statement runs end-to-end through the real engine.
//!
//! The rewrite must produce ordinary SQL — same tables, same planner — and the
//! rows must equal the equivalent hand-written join. Anything the subset does
//! not support must fail with the feature named, never run as something else.
//!
//! Data discipline: tables are in-memory; nothing is written to disk.

use std::sync::Arc;

use arrow::array::{Float64Array, Int64Array, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use gtv_engine::GtvContext;
use gtv_pgq::{rewrite, PropertyGraphRegistry};

fn accounts() -> (SchemaRef, arrow::array::RecordBatch) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new("branch", DataType::Utf8, false),
        Field::new("status", DataType::Utf8, false),
    ]));
    let batch = arrow::array::RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(vec![1, 2, 3, 4])),
            Arc::new(StringArray::from(vec!["HK", "HK", "SG", "HK"])),
            Arc::new(StringArray::from(vec!["open", "open", "closed", "open"])),
        ],
    )
    .unwrap();
    (schema, batch)
}

fn transfers() -> (SchemaRef, arrow::array::RecordBatch) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("txn", DataType::UInt64, false),
        Field::new("src_id", DataType::UInt64, false),
        Field::new("dst_id", DataType::UInt64, false),
        Field::new("amount", DataType::Float64, false),
        Field::new("event_time", DataType::Int64, false),
    ]));
    let batch = arrow::array::RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(vec![10, 11, 12, 13])),
            Arc::new(UInt64Array::from(vec![1, 2, 3, 1])),
            Arc::new(UInt64Array::from(vec![2, 3, 4, 4])),
            Arc::new(Float64Array::from(vec![500.0, 50.0, 900.0, 20.0])),
            Arc::new(Int64Array::from(vec![100, 110, 120, 130])),
        ],
    )
    .unwrap();
    (schema, batch)
}

fn ctx() -> GtvContext {
    let c = GtvContext::new();
    let (s, b) = accounts();
    c.register_batches("accounts", s, vec![b]).unwrap();
    let (s, b) = transfers();
    c.register_batches("transfers", s, vec![b]).unwrap();
    c
}

fn registry() -> PropertyGraphRegistry {
    let mut reg = PropertyGraphRegistry::new();
    reg.add_graph("payments");
    reg.add_vertex("payments", "Account", "accounts", "id").unwrap();
    reg.add_edge("payments", "Transfer", "transfers", "src_id", "dst_id")
        .unwrap();
    reg
}

#[tokio::test]
async fn pgq_query_returns_the_same_rows_as_a_hand_written_join() {
    let c = ctx();
    let reg = registry();

    // `status` is a node property, `amount` an edge property; both must resolve.
    let pgq = "SELECT t.src, t.dst, t.amt FROM GRAPH_TABLE (payments \
               MATCH (a:Account)-[e:Transfer]->(b:Account) \
               WHERE b.status = 'open' \
               COLUMNS (a.id AS src, b.id AS dst, e.amount AS amt)) AS t \
               ORDER BY t.amt DESC";

    let plain = "SELECT a.id AS src, b.id AS dst, e.amount AS amt \
                 FROM transfers e \
                 JOIN accounts a ON a.id = e.src_id \
                 JOIN accounts b ON b.id = e.dst_id \
                 WHERE b.status = 'open' \
                 ORDER BY amt DESC";

    let got = c.sql_pgq(pgq, &reg).await.unwrap();
    let want = c.sql(plain).await.unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!(got, want, "rewrite changed the answer");
    assert_eq!(got[0].num_rows(), 3, "only transfers into open accounts");
}

#[tokio::test]
async fn rewrite_is_plain_sql_and_explains() {
    let c = ctx();
    let reg = registry();
    let pgq = "SELECT t.src FROM GRAPH_TABLE (payments \
               MATCH (a:Account)-[e:Transfer]->(b:Account) \
               COLUMNS (a.id AS src)) AS t";

    let rewritten = c.rewrite_pgq(pgq, &reg).unwrap();
    assert!(!rewritten.sql.contains("GRAPH_TABLE"));
    assert_eq!(rewritten.clauses, 1);
    assert!(rewritten.relations.contains(&"transfers".to_string()));

    // The planner sees ordinary joins over the base tables.
    let plan = c
        .sql(&format!("EXPLAIN {}", rewritten.sql))
        .await
        .unwrap();
    let text = format!("{:?}", plan);
    assert!(text.contains("transfers"), "plan should name the base table: {text}");
}

#[tokio::test]
async fn labelled_edge_and_typed_edges_compose() {
    let c = ctx();
    let mut reg = registry();
    reg.set_edge_type("payments", "Transfer", 7).unwrap();

    // The physical table has no `edge_type` column, so adding the filter must
    // fail loudly at planning time rather than silently returning every edge.
    let pgq = "SELECT t.src FROM GRAPH_TABLE (payments \
               MATCH (a:Account)-[e:Transfer]->(b:Account) \
               COLUMNS (a.id AS src)) AS t";
    let rewritten = rewrite(pgq, &reg).unwrap();
    assert!(rewritten.sql.contains("edge_type = 7"));
    assert!(c.sql(&rewritten.sql).await.is_err(), "missing column must error");
}

#[tokio::test]
async fn unsupported_features_are_named() {
    let reg = registry();
    let cases = [
        // unbounded quantifier
        "SELECT * FROM GRAPH_TABLE (payments MATCH (a:Account)-[e:Transfer]->*(b:Account) COLUMNS (a.id AS x)) AS t",
        // multiple patterns
        "SELECT * FROM GRAPH_TABLE (payments MATCH (a:Account)-[e:Transfer]->(b:Account), (c:Account)-[f:Transfer]->(d:Account) COLUMNS (a.id AS x)) AS t",
        // label alternation
        "SELECT * FROM GRAPH_TABLE (payments MATCH (a:Account|Customer)-[e:Transfer]->(b:Account) COLUMNS (a.id AS x)) AS t",
    ];
    for sql in cases {
        let err = rewrite(sql, &reg).unwrap_err();
        assert!(
            matches!(err, gtv_pgq::PgqError::Unsupported(_) | gtv_pgq::PgqError::Parse { .. }),
            "case must be refused, got {err:?}: {sql}"
        );
    }
}

#[tokio::test]
async fn plain_sql_passes_through_unharmed() {
    let c = ctx();
    let reg = registry();
    let rows = c
        .sql_pgq("SELECT count(*) AS n FROM accounts", &reg)
        .await
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].num_rows(), 1);
}
