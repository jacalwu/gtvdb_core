//! GPUDirect Storage seam — the *decision*, not the transfer.
//!
//! `docs/storage-tiering-ssd-ram-vram.md` §6 scopes this narrowly: on a Linux
//! host with `nvidia-fs` loaded and a supported NVMe topology, a **large**
//! corpus upload may go straight from the file cache to the device and skip the
//! RAM staging round-trip. Every other case — another OS, no `nvidia-fs`, a
//! transfer below `ACCEL_GPUDIRECT_THRESHOLD` — must keep using the staged
//! pinned-buffer path and must record that it did.
//!
//! So this module answers exactly one question — *which upload path?* — and is
//! deliberately pure: no FFI, no I/O, no `unsafe`. That keeps the policy
//! testable on every platform, including the GPU-less CI host, and keeps the
//! honesty property checkable: [`probe`] reports `Unavailable` until a `cufile`
//! binding is actually compiled in, so a build can never *claim* a zero-copy
//! transfer it is not able to perform.
//!
//! When the binding lands, it flips [`DIRECT_TRANSFER_COMPILED`] and implements
//! the `Direct` arm of the upload in `cuda.rs`; nothing else here needs to
//! change, and the counters already in place start reporting direct transfers.

#[cfg(target_os = "linux")]
use std::path::Path;

/// Whether a direct (zero-copy) transfer is compiled into this build.
///
/// This is flipped **together with** the `cufile` binding that performs it,
/// never on its own: a build that reports `Available` without a working
/// transfer would silently turn every large upload into an error or a lie.
#[cfg(target_os = "linux")]
const DIRECT_TRANSFER_COMPILED: bool = false;

/// Whether the host can execute a direct transfer at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectStatus {
    /// The host is provisioned *and* this build can issue direct transfers.
    Available,
    /// The direct path cannot run here; the reason is kept for telemetry.
    Unavailable(&'static str),
}

impl DirectStatus {
    pub fn is_available(self) -> bool {
        matches!(self, Self::Available)
    }

    /// The operator-facing reason (`"available"` when it is).
    pub fn reason(self) -> &'static str {
        match self {
            Self::Available => "available",
            Self::Unavailable(why) => why,
        }
    }
}

/// Which path an upload takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UploadPath {
    /// Stage through the host-pinned buffer. Always available, every platform.
    Staged,
    /// Skip the staging copy via GPUDirect Storage.
    Direct,
}

impl UploadPath {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Staged => "staged",
            Self::Direct => "direct",
        }
    }
}

/// Probe the host for GPUDirect Storage support.
///
/// Cheap and side-effect free, so it is called once when the backend is opened
/// and the answer is cached on it. It is never on a query path.
pub fn probe() -> DirectStatus {
    #[cfg(target_os = "linux")]
    {
        if !nvidia_fs_present() {
            return DirectStatus::Unavailable("no /dev/nvidia-fs* device (nvidia-fs not loaded)");
        }
        if !DIRECT_TRANSFER_COMPILED {
            return DirectStatus::Unavailable("built without cufile bindings");
        }
        DirectStatus::Available
    }
    #[cfg(not(target_os = "linux"))]
    {
        // cufile's compatible mode still needs the Linux module; on Windows /
        // macOS the staged path is the only one, by construction.
        DirectStatus::Unavailable("GPUDirect Storage is Linux-only")
    }
}

/// Whether `/dev` exposes the `nvidia-fs` character devices.
///
/// One node per supporting file system; with none, there is nothing for a
/// direct transfer to talk to.
#[cfg(target_os = "linux")]
fn nvidia_fs_present() -> bool {
    std::fs::read_dir(Path::new("/dev"))
        .map(|entries| {
            entries.flatten().any(|e| {
                e.file_name()
                    .to_string_lossy()
                    .starts_with("nvidia-fs")
            })
        })
        .unwrap_or(false)
}

/// Decide the upload path for a transfer of `bytes`.
///
/// `threshold` is `ACCEL_GPUDIRECT_THRESHOLD`. Below it the staged path wins
/// outright: opening the file range and aligning the descriptor costs more than
/// the copy it would save, so GPUDirect only pays for itself on large corpora.
pub fn upload_plan(
    enabled: bool,
    status: DirectStatus,
    bytes: u64,
    threshold: u64,
) -> UploadPath {
    if enabled && status.is_available() && bytes >= threshold {
        UploadPath::Direct
    } else {
        UploadPath::Staged
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_disabled_setting_always_stages() {
        assert_eq!(
            upload_plan(false, DirectStatus::Available, 1 << 40, 0),
            UploadPath::Staged
        );
    }

    #[test]
    fn a_small_transfer_stages_even_when_direct_is_available() {
        // 512 MiB against the 1 GiB default: not worth the setup cost.
        assert_eq!(
            upload_plan(true, DirectStatus::Available, 512 << 20, 1 << 30),
            UploadPath::Staged
        );
        assert_eq!(
            upload_plan(true, DirectStatus::Available, 1 << 30, 1 << 30),
            UploadPath::Direct,
            "the threshold is inclusive"
        );
    }

    #[test]
    fn an_unavailable_host_never_plans_direct() {
        let status = DirectStatus::Unavailable("no /dev/nvidia-fs* device");
        assert_eq!(upload_plan(true, status, 8 << 30, 0), UploadPath::Staged);
        assert!(!status.is_available());
        assert_eq!(DirectStatus::Available.reason(), "available");
        assert_eq!(status.reason(), "no /dev/nvidia-fs* device");
    }

    #[test]
    fn this_build_does_not_claim_direct_support() {
        // The honesty property: no `cufile` binding is compiled in, so the probe
        // must not report a direct path on any platform.
        assert!(!probe().is_available());
        #[cfg(not(target_os = "linux"))]
        assert_eq!(probe().reason(), "GPUDirect Storage is Linux-only");
    }
}
