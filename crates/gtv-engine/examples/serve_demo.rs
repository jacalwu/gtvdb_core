//! A minimal protocol-v1 server, for client development.
//!
//!     cargo run -p gtv-engine --example serve_demo
//!     GTV_ADDR=0.0.0.0:50051 cargo run -p gtv-engine --example serve_demo
//!
//! It exists because the deployment binary (`gtv-server`, in the extension
//! repository) carries authentication, TLS and the demo workload, which is more
//! than a client author needs in order to try the wire contract. This example
//! is the smallest thing that speaks `gtvquery.QueryService` v1:
//!
//! * `Negotiate` — reports the supported version range;
//! * `ExecuteQuery` — SQL + `as_of` + `snapshot_id` in, Arrow IPC + receipt out;
//! * `Execute` — the v0 shape, answered from the same handler.
//!
//! The tables are plain in-memory batches, **not** a temporal store, so a
//! pinned `as_of` is echoed into the receipt but does not filter anything. That
//! is the one thing this server does not demonstrate; everything else on the
//! wire is real.

use std::sync::Arc;

use arrow::array::{Float64Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use gtv_core::{Freshness, KnowledgeClock};
use gtv_engine::GtvContext;
use gtv_proto::{bind, ErrorCode, QueryHandler, RemoteQuery, WireError};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let addr = std::env::var("GTV_ADDR").unwrap_or_else(|_| "127.0.0.1:50051".to_string());

    let ctx = Arc::new(demo_context()?);

    // Pretend the engine was last refreshed at a fixed instant. Receipts quote
    // it as `freshness.as_known_at`, which is what lets a client tell "no rows
    // matched" from "the data has not arrived yet".
    ctx.set_freshness(Freshness::new(
        1_700_000_000_000_000_000,
        KnowledgeClock::SegmentSystemTs,
    ));

    let handler: QueryHandler = {
        let ctx = Arc::clone(&ctx);
        Arc::new(move |query: RemoteQuery| {
            let ctx = Arc::clone(&ctx);
            Box::pin(async move {
                let query_id = if query.request_id.is_empty() {
                    "anonymous".to_string()
                } else {
                    query.request_id.clone()
                };
                // `sql_with_receipt` hashes the optimized plan and stamps
                // tri-state + snapshot + freshness; `encode_response` then
                // repeats the receipt in the Arrow schema metadata.
                match ctx
                    .sql_with_receipt(query_id, &query.sql, query.as_of, query.snapshot_id, &[])
                    .await
                {
                    Ok(outcome) => Ok((outcome.batches, outcome.receipt)),
                    Err(e) => Err(WireError::new(ErrorCode::Internal, e.to_string())),
                }
            })
        })
    };

    let (bound, _server) = bind(addr.parse()?, handler).await?;
    println!("gtv demo server listening on http://{bound} (protocol v1)");
    println!("  tables: numbers (n, square), prices (t, price)");
    println!("  try:    SELECT * FROM numbers WHERE n <= 3");
    println!("  stop:   Ctrl-C");

    // Park forever; Ctrl-C terminates the process. `bind` owns the server task.
    std::future::pending::<()>().await;
    Ok(())
}

fn demo_context() -> anyhow::Result<GtvContext> {
    let ctx = GtvContext::new();

    let numbers_schema = Arc::new(Schema::new(vec![
        Field::new("n", DataType::Int64, false),
        Field::new("square", DataType::Float64, false),
    ]));
    let n: Vec<i64> = (1..=10).collect();
    let numbers = RecordBatch::try_new(
        numbers_schema.clone(),
        vec![
            Arc::new(Int64Array::from(n.clone())),
            Arc::new(Float64Array::from(
                n.iter().map(|x| (*x as f64).powi(2)).collect::<Vec<_>>(),
            )),
        ],
    )?;
    ctx.register_batches("numbers", numbers_schema, vec![numbers])?;

    let prices_schema = Arc::new(Schema::new(vec![
        Field::new("t", DataType::Int64, false),
        Field::new("price", DataType::Float64, false),
    ]));
    let prices = RecordBatch::try_new(
        prices_schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![0i64, 10, 20, 30, 40, 50])),
            Arc::new(Float64Array::from(vec![100.0, 101.0, 99.0, 102.0, 103.0, 104.0])),
        ],
    )?;
    ctx.register_batches("prices", prices_schema, vec![prices])?;

    Ok(ctx)
}
