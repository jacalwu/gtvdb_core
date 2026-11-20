//! gtv-proto: versioned query dispatch over gRPC + Arrow IPC (A1 / B1 / B5).
//!
//! # What crosses the boundary
//!
//! | v0 (`Execute`) | v1 (`ExecuteQuery`) |
//! |---|---|
//! | SQL in, Arrow IPC out | SQL + `as_of` + `snapshot_id` + `request_id` in |
//! | errors as free text | `ErrorCode` numeric + `error` text |
//! | no completeness signal | `TriState` + full `QueryReceipt` |
//!
//! v1 keeps every v0 field number, so a v0 client keeps working against a v1
//! server for one compatibility cycle (B5). A v1 client that reaches a v0
//! server falls back to `Execute` and is told, in the returned outcome, that no
//! receipt is available — it is never given a receipt that was not computed.
//!
//! # Why the receipt is in two places
//!
//! `QueryResponse.receipt_json` is for servers/clients that speak v1; the same
//! receipt is also stamped into the Arrow schema metadata, so a tool that only
//! reads Arrow (a notebook, `arrow-tools`, a Flight consumer) still carries it.
//! [`attach_receipt_metadata`] and [`receipt_from_ipc`] are the two halves.
//!
//! # What this crate deliberately does not do
//!
//! It does not know how to execute SQL: [`QueryHandler`] is a closure supplied
//! by the embedding application, so `gtv-proto` stays free of DataFusion and of
//! every domain crate. TLS termination is likewise the host's job (see
//! `gtv-server` in the extension repository); [`serve`] is plaintext for tests
//! and for sidecar use on loopback.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::Schema;
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use tonic::metadata::MetadataValue;
use tonic::transport::{Certificate, ClientTlsConfig, Endpoint};

pub mod gtvquery {
    tonic::include_proto!("gtvquery");
}

pub use gtvquery::{ErrorCode, QueryRequest, QueryResponse, TriState as WireTriState};
pub use gtv_core::{is_supported_version, negotiate_version, QueryReceipt, TriState};

pub use gtvquery::query_service_server::{QueryService, QueryServiceServer};
pub use gtvquery::query_service_client::QueryServiceClient;

/// Protocol version this crate speaks.
pub const PROTOCOL_VERSION: u32 = gtv_core::PROTOCOL_VERSION;

/// Oldest version accepted on `ExecuteQuery`.
pub const MIN_SUPPORTED_PROTOCOL: u32 = gtv_core::MIN_SUPPORTED_PROTOCOL;

/// Version reported for calls that arrive through the v0 `Execute` RPC.
pub const LEGACY_PROTOCOL_VERSION: u32 = 0;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// A failure with a stable wire code.
///
/// Clients branch on [`WireError::code`]; `message` is for humans and may be
/// reworded between releases.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WireError {
    pub code: ErrorCode,
    pub message: String,
}

impl WireError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    pub fn invalid_argument(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::InvalidArgument, message)
    }

    pub fn unsupported_version(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::UnsupportedVersion, message)
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::Internal, message)
    }

    /// A statement that did not finish. Carries the reason in `message`; the
    /// structured reason travels in the receipt.
    pub fn incomplete(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::ResultIncomplete, message)
    }

    pub fn snapshot_mismatch(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::SnapshotMismatch, message)
    }
}

impl std::fmt::Display for WireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.code, self.message)
    }
}

impl std::error::Error for WireError {}

impl From<gtv_core::GtvError> for WireError {
    fn from(e: gtv_core::GtvError) -> Self {
        use gtv_core::GtvError::*;
        match e {
            Arrow(inner) => Self::internal(inner.to_string()),
            InvalidArgument(msg) | Schema(msg) => Self::invalid_argument(msg),
            NodeOutOfRange(id) => Self::invalid_argument(format!("node id out of range: {id}")),
            MetricMismatch { index, query } => Self::invalid_argument(format!(
                "metric mismatch: index={index:?}, query={query:?}"
            )),
            DimensionMismatch { index, query } => Self::invalid_argument(format!(
                "dimension mismatch: index={index}, query={query}"
            )),
            BudgetExceeded {
                stage,
                limit,
                observed,
            } => Self::incomplete(format!(
                "budget exceeded during {stage}: limit={limit}, observed={observed}"
            )),
            SnapshotMismatch { expected, got } => {
                Self::snapshot_mismatch(format!("snapshot mismatch: expected {expected}, got {got}"))
            }
            UniverseMismatch { expected, got } => Self::invalid_argument(format!(
                "universe mismatch: expected {expected}, got {got}"
            )),
            HighDegreeNode { node, degree, .. } => {
                Self::invalid_argument(format!("high-degree node {node} (degree {degree})"))
            }
            Cancelled => Self::new(ErrorCode::Cancelled, "query cancelled"),
        }
    }
}

impl WireError {
    /// Map a tonic transport status whose code we cannot know into `Internal`,
    /// except for the two cases that mean "the peer is not there / too old".
    pub fn from_status(status: &tonic::Status) -> Self {
        match status.code() {
            tonic::Code::Unimplemented => Self::unsupported_version(
                "server does not implement ExecuteQuery; it predates protocol v1",
            ),
            tonic::Code::DeadlineExceeded => {
                Self::new(ErrorCode::DeadlineExceeded, status.message().to_string())
            }
            tonic::Code::Cancelled => Self::new(ErrorCode::Cancelled, status.message().to_string()),
            tonic::Code::InvalidArgument => {
                Self::invalid_argument(status.message().to_string())
            }
            _ => Self::internal(status.message().to_string()),
        }
    }
}

/// Convert a generated numeric code into the enum, defaulting to `Internal`
/// for codes this build does not know (forward compatibility).
pub fn error_code_from_i32(raw: i32) -> ErrorCode {
    ErrorCode::try_from(raw).unwrap_or(ErrorCode::Internal)
}

pub fn tri_state_to_wire(state: TriState) -> WireTriState {
    match state {
        TriState::Found => WireTriState::Found,
        TriState::NoMatch => WireTriState::NoMatch,
        TriState::Incomplete => WireTriState::Incomplete,
    }
}

/// `None` when the peer is a v0 server that did not compute a tri-state.
pub fn tri_state_from_wire(raw: i32) -> Option<TriState> {
    match WireTriState::try_from(raw) {
        Ok(WireTriState::Found) => Some(TriState::Found),
        Ok(WireTriState::NoMatch) => Some(TriState::NoMatch),
        Ok(WireTriState::Incomplete) => Some(TriState::Incomplete),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Request / outcome
// ---------------------------------------------------------------------------

/// What a client asks for, independent of the wire shape.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RemoteQuery {
    pub sql: String,
    /// Caller correlation id, echoed into the receipt.
    pub request_id: String,
    pub as_of: Option<i64>,
    pub snapshot_id: Option<u64>,
    pub timeout_ms: Option<u64>,
}

impl RemoteQuery {
    pub fn new(sql: impl Into<String>, request_id: impl Into<String>) -> Self {
        Self {
            sql: sql.into(),
            request_id: request_id.into(),
            ..Default::default()
        }
    }

    pub fn with_as_of(mut self, as_of: i64) -> Self {
        self.as_of = Some(as_of);
        self
    }

    pub fn with_snapshot(mut self, snapshot_id: u64) -> Self {
        self.snapshot_id = Some(snapshot_id);
        self
    }

    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }

    /// v1 wire form.
    pub fn to_request(&self) -> QueryRequest {
        QueryRequest {
            sql: self.sql.clone(),
            protocol_version: PROTOCOL_VERSION,
            as_of: self.as_of,
            snapshot_id: self.snapshot_id,
            request_id: self.request_id.clone(),
            timeout_ms: self.timeout_ms,
        }
    }

    /// v0 wire form: only `sql` exists.
    pub fn to_legacy_request(&self) -> QueryRequest {
        QueryRequest {
            sql: self.sql.clone(),
            ..Default::default()
        }
    }
}

/// Rows plus whatever the server could prove about them.
#[derive(Debug, Clone)]
pub struct RemoteOutcome {
    pub batches: Vec<RecordBatch>,
    /// `None` only when the peer is a v0 server. Never synthesised client-side.
    pub receipt: Option<QueryReceipt>,
    pub tri_state: Option<TriState>,
    pub protocol_version: u32,
}

impl RemoteOutcome {
    /// Total rows across batches.
    pub fn rows(&self) -> usize {
        self.batches.iter().map(|b| b.num_rows()).sum()
    }

    /// `true` when the server proved the answer is complete.
    ///
    /// A v0 peer answers `false`: without a receipt we cannot distinguish
    /// "nothing found" from "did not finish".
    pub fn is_conclusive(&self) -> bool {
        matches!(
            self.tri_state,
            Some(TriState::Found) | Some(TriState::NoMatch)
        )
    }
}

// ---------------------------------------------------------------------------
// Arrow IPC
// ---------------------------------------------------------------------------

/// Encode a sequence of batches into a single Arrow IPC stream.
pub fn encode_batches(batches: &[RecordBatch]) -> arrow::error::Result<Vec<u8>> {
    let schema = batches
        .first()
        .map(|b| b.schema())
        .unwrap_or_else(|| Arc::new(Schema::empty()));
    let mut buf = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, &schema)?;
        for batch in batches {
            writer.write(batch)?;
        }
        writer.finish()?;
    }
    Ok(buf)
}

/// Decode an Arrow IPC stream back into batches.
pub fn decode_batches(data: &[u8]) -> arrow::error::Result<Vec<RecordBatch>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }
    let cursor = std::io::Cursor::new(data);
    let reader = StreamReader::try_new(cursor, None)?;
    reader.collect()
}

/// Stamp `receipt` into every batch's schema metadata, preserving existing keys.
pub fn attach_receipt_metadata(
    batches: Vec<RecordBatch>,
    receipt: &QueryReceipt,
) -> arrow::error::Result<Vec<RecordBatch>> {
    let meta = receipt.to_arrow_metadata();
    batches
        .into_iter()
        .map(|batch| {
            let schema = batch.schema();
            let mut merged: HashMap<String, String> = schema.metadata().clone();
            merged.extend(meta.clone());
            let new_schema = Arc::new(Schema::new_with_metadata(schema.fields().clone(), merged));
            RecordBatch::try_new(new_schema, batch.columns().to_vec())
        })
        .collect()
}

/// Read the receipt a peer stamped into the IPC stream, if any.
pub fn receipt_from_ipc(data: &[u8]) -> Option<gtv_core::Result<QueryReceipt>> {
    let batches = decode_batches(data).ok()?;
    let first = batches.first()?;
    match QueryReceipt::from_arrow_metadata(first.schema().metadata()) {
        Ok(Some(receipt)) => Some(Ok(receipt)),
        Ok(None) => None,
        Err(e) => Some(Err(e)),
    }
}

// ---------------------------------------------------------------------------
// Response decoding (pure — no network)
// ---------------------------------------------------------------------------

/// Turn a wire response into an outcome, or a typed error.
///
/// Pure so it can be tested against hand-built responses: the interesting
/// cases (partial result, version mismatch, missing receipt) should not need a
/// live server to verify.
pub fn decode_response(response: QueryResponse) -> Result<RemoteOutcome, WireError> {
    let code = error_code_from_i32(response.code as i32);
    let receipt = if response.receipt_json.is_empty() {
        None
    } else {
        match QueryReceipt::from_json(&response.receipt_json) {
            Ok(r) => Some(r),
            Err(e) => return Err(WireError::internal(format!("malformed receipt: {e}"))),
        }
    };

    if code != ErrorCode::Ok {
        // A partial answer is still an answer: return the rows *and* the error,
        // so a caller can keep the evidence while knowing it is incomplete.
        if code == ErrorCode::ResultIncomplete {
            let batches = decode_batches(&response.arrow_ipc)
                .map_err(|e| WireError::internal(format!("bad Arrow payload: {e}")))?;
            let tri_state = tri_state_from_wire(response.tri_state as i32)
                .or_else(|| receipt.as_ref().map(|r| r.tri_state));
            return Ok(RemoteOutcome {
                batches,
                receipt,
                tri_state,
                protocol_version: response.protocol_version,
            });
        }
        return Err(WireError::new(code, response.error));
    }

    let batches = decode_batches(&response.arrow_ipc)
        .map_err(|e| WireError::internal(format!("bad Arrow payload: {e}")))?;
    let tri_state =
        tri_state_from_wire(response.tri_state as i32).or_else(|| receipt.as_ref().map(|r| r.tri_state));
    Ok(RemoteOutcome {
        batches,
        receipt,
        tri_state,
        protocol_version: response.protocol_version,
    })
}

/// Build a wire response from an outcome. Used by servers and by tests.
pub fn encode_response(
    outcome: Result<(Vec<RecordBatch>, QueryReceipt), WireError>,
) -> QueryResponse {
    match outcome {
        Ok((batches, receipt)) => {
            let meta_batches = attach_receipt_metadata(batches, &receipt).unwrap_or_default();
            let ipc = encode_batches(&meta_batches).unwrap_or_default();
            QueryResponse {
                arrow_ipc: ipc,
                error: String::new(),
                protocol_version: PROTOCOL_VERSION,
                code: ErrorCode::Ok as i32,
                tri_state: tri_state_to_wire(receipt.tri_state) as i32,
                receipt_json: receipt.to_json(),
            }
        }
        Err(e) => QueryResponse {
            arrow_ipc: Vec::new(),
            error: e.message,
            protocol_version: PROTOCOL_VERSION,
            code: e.code as i32,
            tri_state: WireTriState::Unspecified as i32,
            receipt_json: String::new(),
        },
    }
}

/// Check that a request's version can be served, returning the version to use.
pub fn accept_version(requested: u32) -> Result<u32, WireError> {
    if requested == LEGACY_PROTOCOL_VERSION {
        return Err(WireError::unsupported_version(
            "ExecuteQuery requires protocol v1 or later; use Execute for v0 clients",
        ));
    }
    negotiate_version(requested).ok_or_else(|| {
        WireError::unsupported_version(format!(
            "server supports v{MIN_SUPPORTED_PROTOCOL}..v{PROTOCOL_VERSION}, client sent v{requested}"
        ))
    })
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

/// What a host application must provide: execute one query.
///
/// Boxed future so the type stays object-safe; `gtv-proto` owns no runtime.
pub type QueryHandler = Arc<
    dyn Fn(RemoteQuery) -> Pin<Box<dyn Future<Output = Result<(Vec<RecordBatch>, QueryReceipt), WireError>> + Send>>
        + Send
        + Sync,
>;

/// `QueryService` implementation over a [`QueryHandler`].
#[derive(Clone)]
pub struct QueryServiceImpl {
    handler: QueryHandler,
}

impl QueryServiceImpl {
    pub fn new(handler: QueryHandler) -> Self {
        Self { handler }
    }

    /// Convenience constructor from an async function.
    pub fn from_fn<F, Fut>(f: F) -> Self
    where
        F: Fn(RemoteQuery) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<(Vec<RecordBatch>, QueryReceipt), WireError>> + Send + 'static,
    {
        Self::new(Arc::new(move |q| Box::pin(f(q))))
    }

    /// The handler this service wraps, for handing to [`bind`] / [`serve`].
    pub fn into_handler(self) -> QueryHandler {
        self.handler
    }

    pub fn handler(&self) -> QueryHandler {
        self.handler.clone()
    }
}

#[tonic::async_trait]
impl QueryService for QueryServiceImpl {
    async fn execute(
        &self,
        request: tonic::Request<QueryRequest>,
    ) -> Result<tonic::Response<QueryResponse>, tonic::Status> {
        let req = request.into_inner();
        let query = RemoteQuery::new(req.sql, req.request_id);
        match (self.handler)(query).await {
            Ok((batches, receipt)) => {
                // v0 has no receipt field. The receipt is still stamped into
                // the Arrow metadata, so a client that later upgrades can read
                // it without the server changing.
                let mut resp = encode_response(Ok((batches, receipt)));
                resp.protocol_version = LEGACY_PROTOCOL_VERSION;
                Ok(tonic::Response::new(resp))
            }
            Err(e) => Ok(tonic::Response::new(encode_response(Err(e)))),
        }
    }

    async fn execute_query(
        &self,
        request: tonic::Request<QueryRequest>,
    ) -> Result<tonic::Response<QueryResponse>, tonic::Status> {
        let req = request.into_inner();
        if let Err(e) = accept_version(req.protocol_version) {
            return Ok(tonic::Response::new(encode_response(Err(e))));
        }
        let query = RemoteQuery {
            sql: req.sql,
            request_id: req.request_id,
            as_of: req.as_of,
            snapshot_id: req.snapshot_id,
            timeout_ms: req.timeout_ms,
        };
        let resp = encode_response((self.handler)(query).await);
        Ok(tonic::Response::new(resp))
    }

    async fn negotiate(
        &self,
        request: tonic::Request<gtvquery::VersionRequest>,
    ) -> Result<tonic::Response<gtvquery::VersionResponse>, tonic::Status> {
        let req = request.into_inner();
        let chosen = negotiate_version(req.protocol_version);
        let error = if chosen.is_none() {
            format!(
                "no overlap: server supports v{MIN_SUPPORTED_PROTOCOL}..v{PROTOCOL_VERSION}, client sent v{}",
                req.protocol_version
            )
        } else {
            String::new()
        };
        Ok(tonic::Response::new(gtvquery::VersionResponse {
            protocol_version: chosen.unwrap_or(0),
            min_supported: MIN_SUPPORTED_PROTOCOL,
            max_supported: PROTOCOL_VERSION,
            engine_version: gtv_core::ENGINE_VERSION.to_string(),
            error,
        }))
    }
}

/// Serve plaintext on `addr`, returning the bound address (useful with `:0`).
///
/// TLS is the host's responsibility: `gtv-server` terminates TLS and PQC ahead
/// of this service. Loopback + sidecar is the intended production shape of the
/// embedded engine, so plaintext here is a deployment choice, not an oversight.
pub async fn serve(addr: std::net::SocketAddr, handler: QueryHandler) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let local = listener.local_addr()?;
    tracing_free_log(&format!("gtv-proto listening on {local}"));
    tonic::transport::Server::builder()
        .add_service(QueryServiceServer::new(QueryServiceImpl::new(handler)))
        .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
        .await?;
    Ok(())
}

/// Bind and return the address without awaiting the server loop.
pub async fn bind(
    addr: std::net::SocketAddr,
    handler: QueryHandler,
) -> anyhow::Result<(std::net::SocketAddr, tokio::task::JoinHandle<()>)> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let local = listener.local_addr()?;
    let service = QueryServiceServer::new(QueryServiceImpl::new(handler));
    let handle = tokio::spawn(async move {
        let _ = tonic::transport::Server::builder()
            .add_service(service)
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
            .await;
    });
    Ok((local, handle))
}

/// Minimal stderr trace so the crate does not depend on a logging facade.
fn tracing_free_log(msg: &str) {
    if std::env::var_os("GTV_PROTO_QUIET").is_none() {
        eprintln!("{msg}");
    }
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

async fn connect(
    addr: &str,
    ca_cert_pem: Option<&str>,
) -> anyhow::Result<QueryServiceClient<tonic::transport::Channel>> {
    let scheme = if ca_cert_pem.is_some() {
        "https"
    } else {
        "http"
    };
    let endpoint = Endpoint::from_shared(format!("{scheme}://{addr}"))?;
    let endpoint = match ca_cert_pem {
        Some(ca) => {
            endpoint.tls_config(ClientTlsConfig::new().ca_certificate(Certificate::from_pem(ca)))?
        }
        None => endpoint,
    };
    let channel = endpoint.connect().await?;
    Ok(QueryServiceClient::new(channel)
        .max_decoding_message_size(usize::MAX)
        .max_encoding_message_size(usize::MAX))
}

fn with_token<T>(mut request: tonic::Request<T>, token: Option<&str>) -> anyhow::Result<tonic::Request<T>> {
    if let Some(token) = token {
        let value = MetadataValue::try_from(format!("Bearer {token}"))?;
        request.metadata_mut().insert("authorization", value);
    }
    Ok(request)
}

/// Execute `sql` on a remote gtv-server at `addr` (e.g. `127.0.0.1:50051`).
///
/// Reads `GTV_AUTH_TOKEN` (Bearer token) and `GTV_TLS_CA` (PEM CA certificate)
/// from the environment, so CLI call sites get auth/TLS for free. Use
/// [`query_remote_with`] to pass them explicitly.
pub async fn query_remote(addr: &str, sql: &str) -> anyhow::Result<Vec<RecordBatch>> {
    let token = std::env::var("GTV_AUTH_TOKEN").ok();
    let ca = std::env::var("GTV_TLS_CA").ok();
    query_remote_with(addr, sql, token.as_deref(), ca.as_deref()).await
}

/// v0 client path: SQL in, batches out, no receipt.
///
/// Kept verbatim so existing callers keep working (B5).
pub async fn query_remote_with(
    addr: &str,
    sql: &str,
    token: Option<&str>,
    ca_cert_pem: Option<&str>,
) -> anyhow::Result<Vec<RecordBatch>> {
    let mut client = connect(addr, ca_cert_pem).await?;
    let request = with_token(tonic::Request::new(QueryRequest {
        sql: sql.to_string(),
        ..Default::default()
    }), token)?;
    let response = client.execute(request).await?.into_inner();
    if !response.error.is_empty() {
        anyhow::bail!("remote error: {}", response.error);
    }
    Ok(decode_batches(&response.arrow_ipc)?)
}

/// v1 client path: full query, full outcome, with v0 fallback.
///
/// The fallback is deliberately explicit in the result: when the server is too
/// old, `outcome.receipt` is `None` and `protocol_version` is `0`.
pub async fn query_remote_full(
    addr: &str,
    query: &RemoteQuery,
    token: Option<&str>,
    ca_cert_pem: Option<&str>,
) -> Result<RemoteOutcome, WireError> {
    let mut client = connect(addr, ca_cert_pem)
        .await
        .map_err(|e| WireError::internal(e.to_string()))?;

    let request = with_token(tonic::Request::new(query.to_request()), token)
        .map_err(|e| WireError::invalid_argument(e.to_string()))?;

    match client.execute_query(request).await {
        Ok(response) => {
            let resp = response.into_inner();
            // A v1-capable server that rejects our version is a hard error.
            if resp.code == ErrorCode::UnsupportedVersion as i32 {
                return Err(WireError::unsupported_version(resp.error));
            }
            decode_response(resp)
        }
        Err(status) if status.code() == tonic::Code::Unimplemented => {
            // v0 server: fall back, and say so by returning no receipt.
            let legacy = with_token(tonic::Request::new(query.to_legacy_request()), token)
                .map_err(|e| WireError::invalid_argument(e.to_string()))?;
            let resp = client
                .execute(legacy)
                .await
                .map_err(|s| WireError::from_status(&s))?
                .into_inner();
            if !resp.error.is_empty() {
                return Err(WireError::internal(resp.error));
            }
            let batches = decode_batches(&resp.arrow_ipc)
                .map_err(|e| WireError::internal(format!("bad Arrow payload: {e}")))?;
            let tri = if batches.iter().any(|b| b.num_rows() > 0) {
                TriState::Found
            } else {
                TriState::NoMatch
            };
            Ok(RemoteOutcome {
                batches,
                receipt: None,
                tri_state: Some(tri),
                protocol_version: LEGACY_PROTOCOL_VERSION,
            })
        }
        Err(status) => Err(WireError::from_status(&status)),
    }
}
