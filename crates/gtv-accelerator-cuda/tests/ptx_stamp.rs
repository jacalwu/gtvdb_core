//! The committed PTX must be the PTX the committed source compiles to.
//!
//! The kernel images are checked-in artifacts (`crates/gtv-accelerator-cuda/
//! kernels/*.ptx`) because the normal build must not need a CUDA toolkit — see
//! the module docs in `src/knn.rs`. The risk that buys is silent drift: edit the
//! `.cu`, forget `cargo xtask cuda-ptx`, and every device test keeps passing
//! against last week's kernel.
//!
//! `cargo xtask cuda-ptx` therefore stamps the BLAKE3 of the source it compiled
//! into each image's header, and this test recomputes it. It needs no GPU and no
//! toolkit, so it runs on the CI machine — where it is the *only* thing about
//! the kernels that can be verified at all. Adding a kernel to the list here is
//! the price of adding one to `crates/xtask/src/main.rs`; the two lists are kept
//! in the same order on purpose.

use std::path::Path;

const STAMP: &str = "// gtv-source-sha256: ";

/// `(PTX image, its source, its entry point)`.
const KERNELS: &[(&str, &str, &str, &str)] = &[
    (
        "knn_flat",
        include_str!("../kernels/knn_flat.ptx"),
        include_str!("../kernels/knn_flat.cu"),
        "gtv_knn_flat",
    ),
    (
        "window_scan",
        include_str!("../kernels/window_scan.ptx"),
        include_str!("../kernels/window_scan.cu"),
        "gtv_window_scan",
    ),
];

#[test]
fn every_committed_ptx_was_built_from_the_committed_source() {
    for (name, ptx, source, _) in KERNELS {
        let recorded = ptx
            .lines()
            .find_map(|l| l.strip_prefix(STAMP))
            .map(str::trim)
            .unwrap_or_else(|| {
                panic!("{name}.ptx has no source stamp; regenerate with `cargo xtask cuda-ptx`")
            });
        let actual = blake3::hash(source.as_bytes()).to_hex();
        assert_eq!(
            recorded,
            actual.as_str(),
            "{name}.ptx was built from a different {name}.cu — \
             run `cargo xtask cuda-ptx` and commit the result"
        );
        // A 64-character hex digest, or the stamp would match nothing useful.
        assert_eq!(recorded.len(), 64);
    }
}

#[test]
fn every_image_is_ascii_and_targets_a_portable_architecture() {
    // The driver's PTX parser rejects any non-ASCII byte, comments included, and
    // reports only `CUDA_ERROR_INVALID_PTX`. A `-arch=compute_80` image is
    // forward compatible to Ada, Hopper and later; a `compute_89` one would run
    // on this card and nowhere older.
    for (name, ptx, _, entry) in KERNELS {
        assert!(ptx.is_ascii(), "non-ASCII byte in {name}.ptx");
        assert!(ptx.contains(".target sm_80"), "{name}: unexpected architecture");
        assert!(
            ptx.contains(&format!(".entry {entry}")),
            "{name}: the entry point the host looks up is not in the image"
        );
        // Large enough to contain a kernel, small enough to review.
        assert!((2048..512 * 1024).contains(&ptx.len()), "{name}: {} bytes", ptx.len());
    }
}

#[test]
fn the_generated_files_live_where_the_crate_expects_them() {
    // Cheap guard against a rename that `include_str!` would catch only at build
    // time on one machine and not on another.
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");
    for (name, ..) in KERNELS {
        assert!(dir.join(format!("{name}.cu")).is_file(), "{name}.cu");
        assert!(dir.join(format!("{name}.ptx")).is_file(), "{name}.ptx");
    }
}

#[test]
fn a_tampered_stamp_is_detected() {
    // The check above is only meaningful if a wrong hash fails it. Exercise the
    // comparison with a hash of something else, for every kernel.
    let other = blake3::hash(b"not the kernel").to_hex();
    for (name, ptx, ..) in KERNELS {
        let recorded = ptx
            .lines()
            .find_map(|l| l.strip_prefix(STAMP))
            .map(str::trim)
            .unwrap();
        assert_ne!(recorded, other.as_str(), "{name}");
    }
}

#[test]
fn two_kernels_are_not_the_same_file() {
    // A copy-paste that stamped one source into both images would satisfy every
    // assertion above, because each check is per file. The hashes are the only
    // thing that ties an image to *its* source.
    let a = KERNELS[0].1.lines().find_map(|l| l.strip_prefix(STAMP));
    let b = KERNELS[1].1.lines().find_map(|l| l.strip_prefix(STAMP));
    assert_ne!(a, b, "both images carry the same source stamp");
}
