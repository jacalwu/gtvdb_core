//! Minimal on-device smoke check, used while bringing up the acceptance suite.
//! Compiled only with `--features cuda`.
#![cfg(feature = "cuda")]

use gtv_accelerator::{cuda::CudaBackend, AccelBackend, AccelError, AsofRequest, KnnRequest};
use gtv_core::Metric;

#[test]
fn device_opens_and_both_kernels_launch() {
    let backend = match CudaBackend::probe(0) {
        Ok(b) => b,
        // Only a genuinely absent device is a skip. A device that is present but
        // whose kernel will not compile/launch must fail the gate.
        Err(AccelError::NoDevice(msg)) => {
            if std::env::var("GTV_CUDA_REQUIRE_DEVICE").as_deref() == Ok("1") {
                panic!("CI requires a CUDA device but none is usable: {msg}");
            }
            eprintln!("SKIP: no usable CUDA device: {msg}");
            return;
        }
        Err(e) => panic!("device present but backend unusable: {e}"),
    };
    println!("device = {} / {} MiB / kind={:?}", backend.device().name,
             backend.device().total_bytes / (1024 * 1024), backend.device().kind);

    let ids: Vec<u64> = vec![0, 1, 2];
    let data: Vec<f32> = vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0];
    let query = [0.1f32, 0.1];
    let hits = backend
        .exact_knn(&KnnRequest {
            name: "smoke",
            version: 1,
            ids: &ids,
            data: &data,
            dim: 2,
            metric: Metric::L2,
            normalized: false,
            query: &query,
            k: 3,
            mask: None,
        })
        .expect("knn kernel must launch");
    println!("knn hits = {hits:?}");
    assert_eq!(hits.iter().map(|h| h.id).collect::<Vec<_>>(), vec![0, 1, 2]);

    let right = vec![0i64, 10, 20];
    let left = vec![5i64, 10, 25];
    let matched = backend
        .asof_join(&AsofRequest {
            name: "smoke",
            version: 1,
            left: &left,
            right_times: &right,
            payload_bytes: 3 * 8,
            tolerance_ns: None,
        })
        .expect("asof kernel must launch");
    println!("asof matched = {matched:?}");
    assert_eq!(matched, vec![Some(0), Some(1), Some(2)]);
}
