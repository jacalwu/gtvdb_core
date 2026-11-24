//! In-process Python binding for the gtv temporal-graph engine.
//!
//! This is the native half of the `pygtvdb-local` distribution: the same
//! `gtv-engine` that backs `gtv-server`, embedded directly in the Python
//! process (DuckDB-style). No server, no Docker, no protoc on the user's
//! machine. The GIL is released while queries run.
//!
//! Results cross the FFI boundary as **Arrow IPC stream bytes** and are decoded
//! to `pyarrow.Table` by the pure-Python layer. This keeps the binding thin and
//! avoids pulling `arrow-pyarrow` (whose version would otherwise force a
//! DataFusion-restart). The all-static link means a single wheel carries the
//! whole engine with no system libraries beyond libc/libm/libgcc.

use arrow::datatypes::Schema;
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use gtv_engine::GtvContext;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use tokio::runtime::Runtime;

/// An in-process gtv engine. Each connection owns one isolated `GtvContext`;
/// multiple connections may coexist in one Python process.
#[pyclass]
struct LocalConnection {
    ctx: GtvContext,
    /// The engine's SQL path is async; Python's is not. A dedicated runtime
    /// per connection keeps queries off any ambient/global executor.
    rt: Runtime,
}

/// Serialize result batches into an Arrow IPC stream. The schema is passed in
/// rather than inferred from the first batch: a zero-row result has no batch,
/// and inferring from nothing would ship an empty schema — the Python side then
/// reports no columns at all instead of "no rows".
fn encode_ipc(schema: &Schema, batches: &[RecordBatch]) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, schema)
            .map_err(|e| format!("ipc writer: {e}"))?;
        for b in batches {
            writer.write(b).map_err(|e| format!("ipc write: {e}"))?;
        }
        writer.finish().map_err(|e| format!("ipc finish: {e}"))?;
    }
    Ok(buf)
}

#[pymethods]
impl LocalConnection {
    #[new]
    fn new() -> PyResult<Self> {
        let rt =
            Runtime::new().map_err(|e| PyRuntimeError::new_err(format!("runtime init: {e}")))?;
        Ok(Self {
            ctx: GtvContext::new(),
            rt,
        })
    }

    /// Execute SQL; return Arrow IPC stream bytes. The GIL is released for the
    /// duration of the query (PyO3 0.29: `Python::detach`, previously
    /// `allow_threads`), so distinct connections can run in parallel.
    fn sql<'py>(&self, py: Python<'py>, query: &str) -> PyResult<Bound<'py, PyBytes>> {
        let (schema, batches) = py.detach(|| {
            self.rt
                .block_on(self.ctx.sql_with_schema(query))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        let buf = encode_ipc(&schema, &batches).map_err(PyRuntimeError::new_err)?;
        Ok(PyBytes::new(py, &buf))
    }

    /// Attach a filesystem catalog (committed Parquet snapshots) so SQL can
    /// see persisted tables. Returns nothing; use `sync_catalog` to publish.
    fn attach_catalog(&self, root: &str) -> PyResult<()> {
        self.ctx
            .attach_catalog(root)
            .map(|_| ())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Register a Parquet file as a SQL table in this session.
    fn register_parquet(&self, path: &str, name: &str) -> PyResult<()> {
        self.ctx
            .register_parquet(path, name)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Publish all committed catalog tables into the SQL session. Returns the
    /// number of tables registered.
    fn sync_catalog(&self) -> PyResult<usize> {
        self.ctx
            .sync_catalog()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

#[pymodule]
fn _gtv_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LocalConnection>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field};
    use arrow::ipc::reader::StreamReader;

    #[test]
    fn encode_ipc_roundtrips_schema_and_rows() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("n", DataType::Int64, false),
            Field::new("s", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec![Some("a"), None, Some("c")])),
            ],
        )
        .unwrap();

        let buf = encode_ipc(&schema, std::slice::from_ref(&batch)).unwrap();
        let cursor = std::io::Cursor::new(buf);
        let mut reader = StreamReader::try_new(cursor, None).unwrap();
        let out = reader.next().unwrap().unwrap();
        assert_eq!(out.schema(), schema);
        assert_eq!(out.num_rows(), 3);

        // A zero-row result must still carry its schema: without it Python
        // reports an empty `description` instead of the column list.
        let empty = encode_ipc(&schema, &[]).unwrap();
        let mut r = StreamReader::try_new(std::io::Cursor::new(empty), None).unwrap();
        assert_eq!(r.schema().as_ref(), schema.as_ref());
        assert!(r.next().is_none());
    }
}
