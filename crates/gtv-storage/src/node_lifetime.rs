//! Immutable, fixed-width node-lifetime snapshots (`nodes_<S>.bin`).
//!
//! A retroactive `valid_to` correction publishes a **new file** under a new
//! system-time stamp; the old file stays reachable through `AS OF SYSTEM`.
//! This is the whole bitemporal mechanism for nodes — no per-row system
//! interval, no version-id index, no retract lookup on the read path.
//!
//! Layout (little-endian):
//!
//! ```text
//! header[64]  magic / version / node_count / reserved
//! body        valid_from[i64; N] || valid_to[i64; N]
//! trailer[32] blake3(header || body)
//! ```
//!
//! M1 copies the whole column (O(V)) on publish; a production store would use
//! page-level COW. The *semantics* do not change.

use gtv_core::endpoint::NodeLifetime;

use crate::error::{Result, StorageError};

fn err(msg: impl Into<String>) -> StorageError {
    StorageError::Msg(msg.into())
}

pub const NODE_MAGIC: [u8; 8] = *b"GTVNODE1";
pub const NODE_FORMAT_VERSION: u16 = 1;
pub const NODE_HEADER_SIZE: usize = 64;
pub const NODE_CHECKSUM_SIZE: usize = 32;

pub fn encode_node_lifetime(lifetime: &NodeLifetime) -> Result<Vec<u8>> {
    let (from, to) = lifetime.columns();
    let n = from.len();
    let mut out = Vec::with_capacity(NODE_HEADER_SIZE + n * 16 + NODE_CHECKSUM_SIZE);
    out.extend_from_slice(&NODE_MAGIC);
    out.extend_from_slice(&NODE_FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&[0u8; 6]);
    out.extend_from_slice(&(n as u64).to_le_bytes());
    out.extend_from_slice(&[0u8; 40]);
    debug_assert_eq!(out.len(), NODE_HEADER_SIZE);
    for v in from {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for v in to {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(blake3::hash(&out).as_bytes());
    Ok(out)
}

pub fn decode_node_lifetime(bytes: &[u8]) -> Result<NodeLifetime> {
    if bytes.len() < NODE_HEADER_SIZE + NODE_CHECKSUM_SIZE {
        return Err(err("node lifetime file shorter than its header + checksum"));
    }
    if bytes[0..8] != NODE_MAGIC {
        return Err(err("node lifetime file bad magic"));
    }
    if u16::from_le_bytes(bytes[8..10].try_into().unwrap()) != NODE_FORMAT_VERSION {
        return Err(err("unsupported node lifetime format version"));
    }
    let body_end = bytes.len() - NODE_CHECKSUM_SIZE;
    if blake3::hash(&bytes[..body_end]).as_bytes() != &bytes[body_end..] {
        return Err(err("node lifetime checksum mismatch"));
    }
    let n = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
    if body_end != NODE_HEADER_SIZE + n * 16 {
        return Err(err("node lifetime body length does not match node_count"));
    }
    let mut from = Vec::with_capacity(n);
    let mut to = Vec::with_capacity(n);
    for i in 0..n {
        let a = NODE_HEADER_SIZE + i * 8;
        from.push(i64::from_le_bytes(bytes[a..a + 8].try_into().unwrap()));
    }
    let to_base = NODE_HEADER_SIZE + n * 8;
    for i in 0..n {
        let a = to_base + i * 8;
        to.push(i64::from_le_bytes(bytes[a..a + 8].try_into().unwrap()));
    }
    NodeLifetime::new(from, to).map_err(|e| err(format!("invalid node lifetime file: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_lifetime_file_roundtrip() {
        let l = NodeLifetime::new(vec![0, 100, i64::MIN], vec![10, 200, i64::MAX]).unwrap();
        let bytes = encode_node_lifetime(&l).unwrap();
        assert_eq!(bytes.len(), NODE_HEADER_SIZE + 3 * 16 + NODE_CHECKSUM_SIZE);
        assert_eq!(decode_node_lifetime(&bytes).unwrap(), l);
        let mut torn = bytes.clone();
        let last = torn.len() - 1;
        torn[last] ^= 0x01;
        assert!(decode_node_lifetime(&torn).unwrap_err().to_string().contains("checksum"));
    }
}
