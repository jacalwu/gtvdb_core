//! gtv-proto: gRPC query dispatch + Arrow IPC (Flight wire format) transport.
//!
//! Defines the [`gtvquery`] service and a tiny client/server helper that
//! serializes result [`RecordBatch`]es to/from an Arrow IPC stream, so query
//! results can cross a node boundary as bytes.

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

pub use gtvquery::{QueryRequest, QueryResponse};

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

/// Execute `sql` on a remote gtv-server with explicit transport options.
///
/// When `ca_cert_pem` is `Some`, the connection uses TLS (`https://`) and trusts
/// the supplied CA; otherwise it is plaintext (`http://`). When `token` is
/// `Some`, it is sent as `authorization: Bearer <token>`.
pub async fn query_remote_with(
    addr: &str,
    sql: &str,
    token: Option<&str>,
    ca_cert_pem: Option<&str>,
) -> anyhow::Result<Vec<RecordBatch>> {
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

    // Lift tonic's default 4 MiB message cap so large result sets (TC-10) can
    // be transported as a single Arrow IPC payload.
    let mut client = gtvquery::query_service_client::QueryServiceClient::new(channel)
        .max_decoding_message_size(usize::MAX)
        .max_encoding_message_size(usize::MAX);

    let mut request = tonic::Request::new(QueryRequest {
        sql: sql.to_string(),
    });
    if let Some(token) = token {
        let value = MetadataValue::try_from(format!("Bearer {token}"))?;
        request.metadata_mut().insert("authorization", value);
    }

    let response = client.execute(request).await?.into_inner();
    if !response.error.is_empty() {
        anyhow::bail!("remote error: {}", response.error);
    }
    Ok(decode_batches(&response.arrow_ipc)?)
}
