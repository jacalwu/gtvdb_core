//! Thin Arrow ↔ Parquet round-trip helpers.
//!
//! Writers use a shared [`default_writer_properties`] profile (ZSTD, smaller
//! row groups, dictionary encoding). Readers use the HDB mmap path so every
//! call site shares OS page-cache / huge-page behaviour.

use std::fs::File;
use std::path::Path;

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use crate::error::Result;
use crate::hdb::read_parquet_mmap;

/// Default max rows per Parquet row group (~256k). Smaller than parquet-rs'
/// 1 048 576 default so min/max statistics prune more aggressively on CRM /
/// as-of filters.
pub const DEFAULT_MAX_ROW_GROUP_ROWS: usize = 256 * 1024;

/// Shared writer profile for all gtv Parquet outputs.
///
/// * `GTV_PARQUET_COMPRESSION` — `zstd` (default), `lz4` / `lz4_raw`, `snappy`,
///   or `uncompressed` / `none`.
/// * `GTV_PARQUET_ROW_GROUP_ROWS` — positive integer; default
///   [`DEFAULT_MAX_ROW_GROUP_ROWS`].
pub fn default_writer_properties() -> WriterProperties {
    let compression = match std::env::var("GTV_PARQUET_COMPRESSION")
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "lz4" | "lz4_raw" => Compression::LZ4_RAW,
        "snappy" => Compression::SNAPPY,
        "uncompressed" | "none" => Compression::UNCOMPRESSED,
        // default + explicit "zstd"
        _ => Compression::ZSTD(Default::default()),
    };
    let max_row_group = std::env::var("GTV_PARQUET_ROW_GROUP_ROWS")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_MAX_ROW_GROUP_ROWS);

    WriterProperties::builder()
        .set_compression(compression)
        .set_max_row_group_row_count(Some(max_row_group))
        .set_dictionary_enabled(true)
        .build()
}

/// Write a single [`RecordBatch`] to `path` as a Parquet file.
pub fn write_batch(path: &str, batch: &RecordBatch) -> Result<()> {
    let file = File::create(path)?;
    let props = default_writer_properties();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))?;
    writer.write(batch)?;
    writer.close()?;
    Ok(())
}

/// Read all row groups of a Parquet file into [`RecordBatch`]es via mmap
/// (same path as HDB hot reads).
pub fn read_batches(path: &str) -> Result<Vec<RecordBatch>> {
    read_parquet_mmap(Path::new(path))
}

/// Read only the Arrow schema of a Parquet file (no data scan).
pub fn parquet_schema(path: &str) -> Result<SchemaRef> {
    let file = File::open(path)?;
    let builder = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)?;
    Ok(builder.schema().clone())
}

/// Infer the Arrow schema of a CSV file from its header + up to `max_records`
/// rows (used to register external CSV references).
pub fn infer_csv_schema(path: &str, max_records: usize) -> Result<SchemaRef> {
    use arrow::csv::reader::Format;
    let file = File::open(path)?;
    let mut reader = std::io::BufReader::new(file);
    let (schema, _) = Format::default()
        .with_header(true)
        .infer_schema(&mut reader, Some(max_records))?;
    Ok(std::sync::Arc::new(schema))
}
