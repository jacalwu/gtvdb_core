//! In-memory, native fixed-width delta for the P1 write path.
//!
//! `MemSegment` is deliberately boring: a `Vec` of ingest records, no Arrow
//! arrays, no encoding, no compression. It exists so the ingest hot path only
//! does `WAL::append` + `Vec::push`. All structural work (one sort, CSR column
//! construction, GTVSeg encoding) is deferred to `flush()`.
//!
//! This is the "LSM write, CSR read" split from
//! `docs/storage-mmap-design.md` §3.5: multi-layer deltas may exist *while
//! ingesting*, but no k-way merge ever enters `neighbor_iter`'s hot loop.
//! A version query materialises a single contiguous `TemporalCSR` instead.

use std::collections::BTreeMap;

use gtv_core::csr::TemporalCSR;

use crate::error::{Result, StorageError};
use crate::wal::EdgeRecord;

/// An edge plus the routing / transaction metadata it was ingested with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IngestedEdge {
    pub txn_id: u64,
    pub table_id: u32,
    pub partition_id: u32,
    pub edge: EdgeRecord,
}

#[derive(Debug, Default)]
pub struct MemSegment {
    edges: Vec<IngestedEdge>,
}

impl MemSegment {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    pub fn len(&self) -> usize {
        self.edges.len()
    }

    /// Approximate resident bytes, used for the flush threshold.
    pub fn approx_bytes(&self) -> usize {
        self.edges.len() * std::mem::size_of::<IngestedEdge>()
    }

    pub fn push_batch(
        &mut self,
        txn_id: u64,
        table_id: u32,
        partition_id: u32,
        edges: &[EdgeRecord],
    ) {
        self.edges.reserve(edges.len());
        for &edge in edges {
            self.edges.push(IngestedEdge {
                txn_id,
                table_id,
                partition_id,
                edge,
            });
        }
    }

    pub fn extend(&mut self, other: impl IntoIterator<Item = IngestedEdge>) {
        self.edges.extend(other);
    }

    pub fn clear(&mut self) {
        self.edges.clear();
    }

    pub fn max_system_ts(&self) -> Option<i64> {
        self.edges.iter().map(|e| e.edge.system_ts).max()
    }

    pub fn txn_range(&self) -> Option<(u64, u64)> {
        let mut it = self.edges.iter().map(|e| e.txn_id);
        let first = it.next()?;
        let mut lo = first;
        let mut hi = first;
        for t in it {
            lo = lo.min(t);
            hi = hi.max(t);
        }
        Some((lo, hi))
    }

    /// Group the delta by `(table_id, partition_id)`. Each group becomes one
    /// immutable GTVSeg on flush; untouched historical segments are never
    /// reopened or rewritten (macroblock reuse).
    pub fn groups(&self) -> BTreeMap<(u32, u32), Vec<EdgeRecord>> {
        let mut out: BTreeMap<(u32, u32), Vec<EdgeRecord>> = BTreeMap::new();
        for e in &self.edges {
            out.entry((e.table_id, e.partition_id))
                .or_default()
                .push(e.edge);
        }
        out
    }

    /// Edges visible at system time `T` (bitemporal `as_of_system`).
    pub fn edges_at_system(&self, t: i64) -> Vec<EdgeRecord> {
        self.edges
            .iter()
            .filter(|e| e.edge.system_ts <= t)
            .map(|e| e.edge)
            .collect()
    }

    pub fn records(&self) -> &[IngestedEdge] {
        &self.edges
    }

    /// Build a single contiguous CSR from native edge records. This is the
    /// **only** place a flush pays for a sort.
    pub fn to_csr(edges: &[EdgeRecord], node_count: usize) -> Result<TemporalCSR> {
        let n = edges.len();
        let mut src = Vec::with_capacity(n);
        let mut dst = Vec::with_capacity(n);
        let mut valid_from = Vec::with_capacity(n);
        let mut valid_to = Vec::with_capacity(n);
        let mut edge_type = Vec::with_capacity(n);
        for e in edges {
            src.push(e.src);
            dst.push(e.dst);
            valid_from.push(e.valid_from);
            valid_to.push(e.valid_to);
            edge_type.push(e.edge_type);
        }
        TemporalCSR::from_edge_columns(
            &src,
            &dst,
            &valid_from,
            &valid_to,
            &edge_type,
            node_count,
        )
        .map_err(|e| StorageError::Msg(format!("invalid delta CSR: {e}")))
    }
}
