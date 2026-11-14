//! The host backend — always available, never fails for lack of a device.
//!
//! It is not a stub: it runs the same kernels the device path is measured
//! against (`gtv_index::exact_topk` for K-NN, [`crate::sweep::asof_match`] for
//! as-of), so "fall back to the host" degrades to the optimized CPU path rather
//! than to something slow that makes the GPU look good by comparison.

use std::sync::OnceLock;

use crate::backend::{
    AccelBackend, AccelError, AccelOp, AsofRequest, DeviceInfo, DeviceKind, IvfRequest, KnnRequest,
    RawHit,
};
use crate::sweep::asof_match;

/// CPU execution backend.
#[derive(Debug, Default, Clone, Copy)]
pub struct HostBackend;

impl HostBackend {
    pub fn new() -> Self {
        Self
    }
}

fn cpu_info() -> &'static DeviceInfo {
    static INFO: OnceLock<DeviceInfo> = OnceLock::new();
    INFO.get_or_init(DeviceInfo::cpu)
}

impl AccelBackend for HostBackend {
    fn device(&self) -> &DeviceInfo {
        cpu_info()
    }

    fn supports(&self, _op: AccelOp) -> bool {
        true
    }

    fn exact_knn(&self, req: &KnnRequest<'_>) -> Result<Vec<RawHit>, AccelError> {
        if req.dim == 0 || req.query.len() != req.dim {
            return Err(AccelError::Invalid(format!(
                "query dim {} != corpus dim {}",
                req.query.len(),
                req.dim
            )));
        }
        if req.data.len() != req.ids.len() * req.dim {
            return Err(AccelError::Invalid(format!(
                "corpus has {} floats, expected {} x {}",
                req.data.len(),
                req.ids.len(),
                req.dim
            )));
        }
        Ok(gtv_index::exact_topk(
            gtv_index::CorpusView {
                ids: req.ids,
                data: req.data,
                dim: req.dim,
                metric: req.metric,
                normalized: req.normalized,
            },
            req.query,
            req.k,
            req.mask,
        )
        .into_iter()
        .map(|(distance, id)| RawHit { distance, id })
        .collect())
    }

    fn ivf_probe(&self, req: &IvfRequest<'_>) -> Result<Vec<RawHit>, AccelError> {
        if req.dim == 0 || req.query.len() != req.dim {
            return Err(AccelError::Invalid(format!(
                "query dim {} != corpus dim {}",
                req.query.len(),
                req.dim
            )));
        }
        if req.data.len() != req.ids.len() * req.dim {
            return Err(AccelError::Invalid(format!(
                "ivf corpus has {} floats, expected {} x {}",
                req.data.len(),
                req.ids.len(),
                req.dim
            )));
        }
        Ok(gtv_index::exact_topk_positions(
            gtv_index::CorpusView {
                ids: req.ids,
                data: req.data,
                dim: req.dim,
                metric: req.metric,
                normalized: req.normalized,
            },
            req.positions,
            req.query,
            req.k,
        )
        .into_iter()
        .map(|(distance, id)| RawHit { distance, id })
        .collect())
    }

    fn asof_join(&self, req: &AsofRequest<'_>) -> Result<Vec<Option<u32>>, AccelError> {
        Ok(asof_match(
            req.left,
            req.right_times,
            req.tolerance_ns,
        ))
    }
}

impl HostBackend {
    /// Device kind, for callers that only hold the trait object.
    pub fn kind(&self) -> DeviceKind {
        DeviceKind::Cpu
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gtv_core::Metric;

    #[test]
    fn host_knn_ranks_nearest_first() {
        let ids = vec![0u64, 1, 2];
        let data = vec![0.0f32, 0.0, 1.0, 1.0, 5.0, 5.0];
        let q = [0.1f32, 0.1];
        let hits = HostBackend::new()
            .exact_knn(&KnnRequest {
                name: "t",
                version: 1,
                ids: &ids,
                data: &data,
                dim: 2,
                metric: Metric::L2,
                normalized: false,
                query: &q,
                k: 3,
                mask: None,
            })
            .unwrap();
        assert_eq!(hits.iter().map(|h| h.id).collect::<Vec<_>>(), vec![0, 1, 2]);
    }

    #[test]
    fn host_knn_rejects_a_mismatched_corpus() {
        let ids = vec![0u64, 1];
        let data = vec![0.0f32, 0.0];
        let err = HostBackend::new()
            .exact_knn(&KnnRequest {
                name: "t",
                version: 1,
                ids: &ids,
                data: &data,
                dim: 2,
                metric: Metric::L2,
                normalized: false,
                query: &[0.0, 0.0],
                k: 1,
                mask: None,
            })
            .unwrap_err();
        assert!(matches!(err, AccelError::Invalid(_)));
    }

    #[test]
    fn host_ivf_probe_scans_only_the_selected_positions() {
        let ids = vec![10u64, 11, 12, 13];
        let data = vec![
            0.0f32, 0.0, // 10
            1.0, 1.0, // 11
            2.0, 2.0, // 12
            9.0, 9.0, // 13
        ];
        let q = [9.0f32, 9.0];
        let hits = HostBackend::new()
            .ivf_probe(&IvfRequest {
                name: "ivf",
                version: 1,
                ids: &ids,
                data: &data,
                dim: 2,
                metric: Metric::L2,
                normalized: false,
                query: &q,
                k: 2,
                // Position 3 (the nearest row) is deliberately NOT probed.
                positions: &[0, 1, 2],
            })
            .unwrap();
        assert_eq!(hits.iter().map(|h| h.id).collect::<Vec<_>>(), vec![12, 11]);
    }

    #[test]
    fn host_asof_matches_the_series() {
        let right = vec![0i64, 10, 20];
        let got = HostBackend::new()
            .asof_join(&AsofRequest {
                name: "s",
                version: 1,
                left: &[5, 10, 25],
                right_times: &right,
                payload_bytes: 0,
                tolerance_ns: None,
            })
            .unwrap();
        assert_eq!(got, vec![Some(0), Some(1), Some(2)]);
    }

    /// Not part of CI: the contiguous + bounded-selection scan against the
    /// `Vec<Vec<f32>>` + full-sort shape it replaced.
    ///
    /// Run with
    /// `cargo test -p gtv-accelerator --release -- --ignored --nocapture bench_knn`.
    #[test]
    #[ignore = "measurement, not a correctness gate"]
    fn bench_knn_contiguous_vs_vec_of_vecs() {
        use std::time::Instant;

        const N: usize = 200_000;
        const DIM: usize = 128;
        const K: usize = 10;

        let mut seed = 0x9E3779B9_7F4A7C15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / 1024.0
        };

        let rows: Vec<Vec<f32>> = (0..N)
            .map(|_| (0..DIM).map(|_| next()).collect())
            .collect();
        let mut flat = Vec::with_capacity(N * DIM);
        for r in &rows {
            flat.extend_from_slice(r);
        }
        let ids: Vec<u64> = (0..N as u64).collect();
        let query: Vec<f32> = (0..DIM).map(|_| next()).collect();

        // Shape of the code that was replaced: score every vector, then sort all
        // N of them and truncate.
        let t0 = Instant::now();
        let mut old: Vec<(f32, u64)> = rows
            .iter()
            .enumerate()
            .map(|(i, v)| (gtv_core::Metric::L2.distance(&query, v), ids[i]))
            .collect();
        old.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
        old.truncate(K);
        let slow = t0.elapsed();

        let t1 = Instant::now();
        let new = gtv_index::exact_topk(
            gtv_index::CorpusView {
                ids: &ids,
                data: &flat,
                dim: DIM,
                metric: gtv_core::Metric::L2,
                normalized: false,
            },
            &query,
            K,
            None,
        );
        let fast = t1.elapsed();

        assert_eq!(
            old.iter().map(|h| h.1).collect::<Vec<_>>(),
            new.iter().map(|h| h.1).collect::<Vec<_>>(),
            "both paths must return the same top-k"
        );
        println!(
            "knn N={N} dim={DIM} k={K}: contiguous+select {:?} vs vec-of-vec+full sort {:?} ({:.2}x)",
            fast,
            slow,
            slow.as_secs_f64() / fast.as_secs_f64().max(1e-9)
        );
    }
}
