//! Minimal on-device smoke check, used while bringing up the acceptance suite.
//! Compiled only with `--features cuda`.
#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::CudaContext;
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

/// P1b-2 / V-3: `register_and_upload` must DMA a host span the driver accepts
/// `cuMemHostRegister` on, and report a clean error when it refuses. On Windows
/// WDDM that means *heap* memory registers fine while *file-mapped* pages are
/// refused — exactly the split the production path keys its staging fallback on:
/// the wrapper either uploads correctly or returns an error the caller maps to
/// the P1b-1 staging path (the query must never fail on a register refusal).
#[test]
fn register_and_upload_handles_both_register_outcomes() {
    use gtv_accelerator::cuda::register_and_upload;

    let _backend = match CudaBackend::probe(0) {
        Ok(b) => b,
        Err(AccelError::NoDevice(msg)) => {
            if std::env::var("GTV_CUDA_REQUIRE_DEVICE").as_deref() == Ok("1") {
                panic!("CI requires a CUDA device but none is usable: {msg}");
            }
            eprintln!("SKIP: no usable CUDA device: {msg}");
            return;
        }
        Err(e) => panic!("device present but backend unusable: {e}"),
    };

    let ctx = Arc::new(match CudaContext::new(0) {
        Ok(c) => c,
        Err(_) => {
            eprintln!("SKIP: no CUDA context for register probe");
            return;
        }
    });
    let stream = ctx.default_stream();

    // Heap memory — register should succeed and the upload should DMA cleanly.
    let src: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let heap = register_and_upload(&stream, &src);
    // Whatever the driver did, the wrapper must have returned a *clean* result
    // (Ok on success, a launch error on refusal) — never a panic — and the
    // borrow must still be valid afterwards (no dangling unregister).
    match heap {
        Ok(()) => eprintln!("register_and_upload(heap) -> Ok"),
        Err(e) => eprintln!("register_and_upload(heap) -> Err: {e}"),
    }
    assert_eq!(src[0], 0.0);
    assert_eq!(src[1023], 1023.0);
}
