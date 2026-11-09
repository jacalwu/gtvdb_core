//! Build tasks that need a CUDA toolkit, kept out of the normal build.
//!
//! `cargo test --workspace` on a machine with no CUDA must stay green, so the
//! kernel compiler cannot be part of the crate's build script. Instead the PTX
//! is compiled here, on a machine that has `nvcc`, and committed. The crate
//! embeds it with `include_str!` and the driver JITs it — one `compute_80` image
//! covers Ampere and everything after it, which is why a single file is enough.
//!
//! # What the stamp is for
//!
//! A checked-in binary artifact that claims to come from a source file is a lie
//! waiting to happen: edit the `.cu`, forget the rebuild, and the tests keep
//! passing against last week's kernel. So every generated `.ptx` carries the
//! BLAKE3 of the `.cu` it came from, and `tests/ptx_stamp.rs` recomputes that
//! hash from the committed source on every `cargo test` — including on the GPU-less
//! CI machine, where it is the only thing about the kernel that *can* be
//! verified.
//!
//! Usage:
//!
//! ```text
//! cargo xtask cuda-ptx          # regenerate every kernel
//! cargo xtask cuda-ptx --check  # verify the committed PTX matches its source
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;

/// Every kernel: `(package dir, source file, PTX file, entry point)`.
///
/// The entry point is not decoration — a module loads successfully even if the
/// function inside it was renamed, so the only way to catch that at build time
/// is to look for the symbol in the generated PTX.
const KERNELS: &[(&str, &str, &str, &str)] = &[(
    "crates/gtv-accelerator-cuda",
    "kernels/knn_flat.cu",
    "kernels/knn_flat.ptx",
    "gtv_knn_flat",
)];

/// The virtual architecture the PTX targets.
///
/// `compute_80` rather than the local card's `compute_89`: PTX is forward
/// compatible, so one image runs on Ada, Hopper and Blackwell alike, and a
/// kernel this simple gains nothing from an architecture-specific target.
const ARCH: &str = "compute_80";

/// The marker line the stamp test greps for. Keep the two in sync.
pub const STAMP_PREFIX: &str = "// gtv-source-sha256: ";

fn main() {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("cuda-ptx") => {
            let check = args.any(|a| a == "--check");
            let root = workspace_root();
            if let Err(e) = cuda_ptx(&root, check) {
                eprintln!("cuda-ptx: {e}");
                std::process::exit(1);
            }
        }
        Some(other) => {
            eprintln!("unknown task {other:?}");
            eprintln!("usage: cargo xtask cuda-ptx [--check]");
            std::process::exit(2);
        }
        None => {
            eprintln!("usage: cargo xtask cuda-ptx [--check]");
            std::process::exit(2);
        }
    }
}

/// The workspace root: this crate lives at `<root>/crates/xtask`.
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/xtask is two levels below the root")
        .to_path_buf()
}

fn cuda_ptx(root: &Path, check: bool) -> Result<(), String> {
    let nvcc = find_nvcc()?;
    let mut stale = 0;
    for (pkg, src, out, entry) in KERNELS {
        let src_path = root.join(pkg).join(src);
        let out_path = root.join(pkg).join(out);
        let source = std::fs::read(&src_path)
            .map_err(|e| format!("read {}: {e}", src_path.display()))?;
        let hash = blake3::hash(&source).to_hex();

        if check {
            let committed = std::fs::read_to_string(&out_path)
                .map_err(|e| format!("read {}: {e}", out_path.display()))?;
            match stamped_hash(&committed) {
                Some(h) if h == hash.as_str() => {
                    println!("{}: up to date ({})", out_path.display(), &hash[..12]);
                }
                Some(h) => {
                    stale += 1;
                    eprintln!(
                        "{}: STALE — source is {hash}, ptx was built from {h}",
                        out_path.display()
                    );
                }
                None => {
                    stale += 1;
                    eprintln!("{}: no stamp", out_path.display());
                }
            }
            continue;
        }

        let ptx = compile(&nvcc, &src_path)?;
        if !ptx.contains(entry) {
            return Err(format!(
                "{src}: nvcc produced no `{entry}` symbol — the entry point was renamed"
            ));
        }
        let mut text = String::new();
        // ASCII only, and not by preference: the driver's PTX parser rejects any
        // non-ASCII byte, comments included (`CUDA_ERROR_INVALID_PTX`, with no
        // hint about why). An em dash in this header cost one debugging cycle.
        text.push_str("// GENERATED FILE - do not edit. Regenerate with `cargo xtask cuda-ptx`.\n");
        text.push_str(&format!("{STAMP_PREFIX}{}\n", hash.as_str()));
        text.push_str(&format!("// Source: {pkg}/{src}\n"));
        text.push_str(&format!("// Target: -arch={ARCH}\n"));
        text.push_str(&ptx);
        std::fs::write(&out_path, text)
            .map_err(|e| format!("write {}: {e}", out_path.display()))?;
        println!(
            "{}: {} bytes, sha256 {}",
            out_path.display(),
            ptx.len(),
            &hash[..12]
        );
    }
    if stale > 0 {
        return Err(format!("{stale} kernel(s) out of date; run `cargo xtask cuda-ptx`"));
    }
    Ok(())
}

/// The hash recorded in a generated PTX header, if there is one.
fn stamped_hash(ptx: &str) -> Option<&str> {
    ptx.lines()
        .find_map(|l| l.strip_prefix(STAMP_PREFIX))
        .map(|h| h.trim())
}

fn compile(nvcc: &Path, src: &Path) -> Result<String, String> {
    let out = std::env::temp_dir().join("gtv-xtask.ptx");
    let status = Command::new(nvcc)
        .arg("-ptx")
        .arg(format!("-arch={ARCH}"))
        .arg("-o")
        .arg(&out)
        .arg(src)
        .status()
        .map_err(|e| format!("spawn {}: {e}", nvcc.display()))?;
    if !status.success() {
        return Err(format!(
            "nvcc failed for {} — on Windows nvcc needs the MSVC environment \
             (run from a Developer Command Prompt, or set INCLUDE/PATH to the \
             MSVC and Windows SDK directories)",
            src.display()
        ));
    }
    // PTX is UTF-8 by construction (it is ASCII), so a lossy read can only be
    // wrong if the file is not what we think it is — which is worth failing on.
    let bytes = std::fs::read(&out).map_err(|e| format!("read {}: {e}", out.display()))?;
    let text =
        String::from_utf8(bytes).map_err(|_| format!("{} is not UTF-8 PTX", out.display()))?;
    if !text.is_ascii() {
        return Err(format!(
            "{} contains non-ASCII bytes; the driver's PTX parser rejects them \
             even inside comments",
            out.display()
        ));
    }
    Ok(text)
}

/// `nvcc` from `CUDA_PATH` if it is set, else from `PATH`.
fn find_nvcc() -> Result<PathBuf, String> {
    let exe = if cfg!(windows) { "nvcc.exe" } else { "nvcc" };
    for var in ["CUDA_PATH", "CUDA_HOME"] {
        if let Some(dir) = std::env::var_os(var) {
            let candidate = Path::new(&dir).join("bin").join(exe);
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }
    // Bare name: let the OS resolve it. `Command` reports a spawn failure with
    // the name, so a missing toolkit produces "spawn nvcc: ... not found".
    let on_path = PathBuf::from(exe);
    if Command::new(&on_path)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
    {
        return Ok(on_path);
    }
    Err(format!(
        "no nvcc: set CUDA_PATH or put {exe} on PATH. \
         The normal build does not need it — only regenerating the committed PTX does."
    ))
}
