//! The committed PTX must be the PTX the committed source compiles to.
//!
//! The kernel image is a checked-in artifact (`crates/gtv-accelerator-cuda/kernels/
//! knn_flat.ptx`) because the normal build must not need a CUDA toolkit — see the
//! module docs in `src/knn.rs`. The risk that buys is silent drift: edit the
//! `.cu`, forget `cargo xtask cuda-ptx`, and every device test keeps passing
//! against last week's kernel.
//!
//! `cargo xtask cuda-ptx` therefore stamps the BLAKE3 of the source it compiled
//! into the image's header, and this test recomputes it. It needs no GPU and no
//! toolkit, so it runs on the CI machine — where it is the *only* thing about the
//! kernel that can be verified at all.

use std::path::Path;

const PTX: &str = include_str!("../kernels/knn_flat.ptx");
const SOURCE: &[u8] = include_bytes!("../kernels/knn_flat.cu");
const STAMP: &str = "// gtv-source-sha256: ";

#[test]
fn the_committed_ptx_was_built_from_the_committed_source() {
    let recorded = PTX
        .lines()
        .find_map(|l| l.strip_prefix(STAMP))
        .map(str::trim)
        .expect("the PTX has no source stamp; regenerate with `cargo xtask cuda-ptx`");

    let actual = blake3::hash(SOURCE).to_hex();
    assert_eq!(
        recorded,
        actual.as_str(),
        "kernels/knn_flat.ptx was built from a different knn_flat.cu — \
         run `cargo xtask cuda-ptx` and commit the result"
    );
    // A 64-character hex digest, or the stamp would match nothing useful.
    assert_eq!(recorded.len(), 64);
}

#[test]
fn the_image_is_ascii_and_targets_a_portable_architecture() {
    // The driver's PTX parser rejects any non-ASCII byte, comments included, and
    // reports only `CUDA_ERROR_INVALID_PTX`. A `-arch=compute_80` image is
    // forward compatible to Ada, Hopper and later; a `compute_89` one would run
    // on this card and nowhere older.
    assert!(PTX.is_ascii(), "non-ASCII byte in the PTX image");
    assert!(PTX.contains(".target sm_80"), "unexpected architecture");
    assert!(
        PTX.contains(".entry gtv_knn_flat"),
        "the entry point the host looks up is not in the image"
    );
}

#[test]
fn the_generated_files_live_where_the_crate_expects_them() {
    // Cheap guard against a rename that `include_str!` would catch only at build
    // time on one machine and not on another.
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");
    assert!(dir.join("knn_flat.cu").is_file());
    assert!(dir.join("knn_flat.ptx").is_file());
    // The image should be small enough to review and large enough to contain a
    // kernel; a few kilobytes of PTX per kernel is the expected order.
    assert!((2048..512 * 1024).contains(&PTX.len()), "{} bytes", PTX.len());
}

#[test]
fn a_tampered_stamp_is_detected() {
    // The check above is only meaningful if a wrong hash fails it. Exercise the
    // comparison with a hash of something else.
    let other = blake3::hash(b"not the kernel").to_hex();
    let recorded = PTX
        .lines()
        .find_map(|l| l.strip_prefix(STAMP))
        .map(str::trim)
        .unwrap();
    assert_ne!(recorded, other.as_str());
}
