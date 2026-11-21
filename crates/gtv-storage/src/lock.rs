//! Data-directory ownership: who may open a directory, and what a refusal says.
//!
//! `SegmentStore` is a single-writer structure. It appends to one WAL, replaces
//! one manifest in place and unlinks segments the manifest no longer references;
//! two processes interleaving those operations cannot be made correct by
//! ordering them better. The lock *is* the mechanism, so it has to be an OS
//! advisory lock rather than a file that merely claims to be in use:
//!
//! * An advisory lock dies with its holder, so `kill -9` leaves nothing for the
//!   next start to trip over. A pid file cannot do that without inventing
//!   liveness detection, and pid reuse defeats it anyway.
//! * The lock file's **contents are diagnostics only** — pid, host, start time,
//!   mode, kept in `OWNER.json` beside an empty `OWNER.lock`. They are read to
//!   explain a refusal ("who holds it", design R5) and never consulted to
//!   decide whether the directory is free. The record can lag by the microseconds
//!   between taking the lock and writing it, and it survives `kill -9` — both
//!   are acceptable for a hint and fatal for an authority.
//!
//! The lock file is never unlinked, not even on drop. On Unix the lock belongs
//! to the open file description, so removing the path would let the next
//! process create a *different* inode and "acquire" that while the first still
//! holds the old one. A stale-looking lock file is the price of correctness; a
//! stale *lock* cannot exist.

use std::fs::{self, File, OpenOptions, TryLockError};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{Result, StorageError};

/// Name of the lock file inside a data directory. Its bytes are never read:
/// on Windows `LockFileEx` is a *mandatory* lock over the whole file, so a
/// second process cannot even open it for reading while the holder lives.
pub const LOCK_FILE: &str = "OWNER.lock";

/// Name of the holder record. Separate from the lock because of the paragraph
/// above: the record has to stay readable by the process that is being refused.
/// It is a **hint**, not the authority — see the module docs.
pub const OWNER_FILE: &str = "OWNER.json";

/// How a data directory is being opened.
///
/// The three modes exist because "open a directory" answers two different
/// questions: *may I write?* and *must I be alone?* Conflating them is what
/// makes a read-only replica silently become a second writer's problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum OpenMode {
    /// Read + write + flush + GC, and no other writer may hold the directory.
    /// The only mode that may modify files; a network-mode primary and a local
    /// single-user session both use it.
    ReadWriteExclusive,
    /// Read-only, no lock. The manifest is pinned when the store opens, so the
    /// answer set cannot drift underneath a query — but nothing stops another
    /// process from writing to the directory meanwhile.
    ReadOnlyShared,
    /// Read-only, shared lock: other readers may hold it too, a
    /// [`OpenMode::ReadWriteExclusive`] holder may not.
    ReadOnlySharedLocked,
}

impl OpenMode {
    /// Whether this mode may modify the directory.
    pub fn is_read_only(self) -> bool {
        !matches!(self, Self::ReadWriteExclusive)
    }

    /// Stable spelling for CLI flags, error messages and receipts.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ReadWriteExclusive => "rw-exclusive",
            Self::ReadOnlyShared => "ro-shared",
            Self::ReadOnlySharedLocked => "ro-shared-locked",
        }
    }

    /// Parse the spelling produced by [`Self::as_str`].
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "rw-exclusive" | "rw" => Some(Self::ReadWriteExclusive),
            "ro-shared" | "ro" => Some(Self::ReadOnlyShared),
            "ro-shared-locked" => Some(Self::ReadOnlySharedLocked),
            _ => None,
        }
    }
}

impl std::fmt::Display for OpenMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What the lock file claims about its holder. Diagnostics only — see the
/// module docs for why this must never be the authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LockOwner {
    pub pid: u32,
    pub host: String,
    /// Unix seconds at acquisition. Wall clock, so it is comparable across
    /// processes but not monotonic.
    pub started_at: i64,
    pub mode: OpenMode,
}

impl LockOwner {
    /// One-line description used in "already open by …" errors.
    pub fn describe(&self) -> String {
        format!(
            "pid {} on {} ({}, since unix {})",
            self.pid, self.host, self.mode, self.started_at
        )
    }
}

/// The guard that keeps a data directory owned for as long as it lives.
///
/// Dropping it releases the OS lock (the `File` closes); the lock file itself
/// stays on disk. `ro-shared` acquires no lock at all and is represented by the
/// same type so callers can hold "whatever this mode required" uniformly.
#[derive(Debug)]
pub struct StoreLock {
    /// `OWNER.lock` — the authority.
    path: PathBuf,
    /// `OWNER.json` — the readable hint about who holds `path`.
    owner_path: PathBuf,
    /// `None` for [`OpenMode::ReadOnlyShared`], which takes no lock.
    file: Option<File>,
    mode: OpenMode,
}

impl StoreLock {
    /// Acquire ownership of `root` under `mode`.
    ///
    /// For a writable mode the directory is created if missing; a read-only
    /// mode refuses to create one, because "open a directory that does not
    /// exist" is a typo, not an empty database.
    pub fn acquire(root: impl AsRef<Path>, mode: OpenMode) -> Result<Self> {
        let root = root.as_ref();
        let path = root.join(LOCK_FILE);
        let owner_path = root.join(OWNER_FILE);

        if mode.is_read_only() && !root.is_dir() {
            return Err(StorageError::Msg(format!(
                "no data directory at `{}` ({} mode does not create one)",
                root.display(),
                mode
            )));
        }
        if !mode.is_read_only() {
            fs::create_dir_all(root).map_err(|e| {
                StorageError::Msg(format!("cannot create data directory `{}`: {e}", root.display()))
            })?;
        }

        // `ro-shared` deliberately takes no lock: it reads a manifest snapshot
        // and never writes. Taking a lock here would turn a report replica into
        // a process that can block the writer for no benefit.
        if mode == OpenMode::ReadOnlyShared {
            return Ok(Self { path, owner_path, file: None, mode });
        }

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(|e| {
                StorageError::Msg(format!("cannot open lock file `{}`: {e}", path.display()))
            })?;

        let attempt = match mode {
            OpenMode::ReadWriteExclusive => file.try_lock(),
            OpenMode::ReadOnlySharedLocked => file.try_lock_shared(),
            OpenMode::ReadOnlyShared => unreachable!("handled above"),
        };
        match attempt {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => {
                return Err(StorageError::Locked {
                    path: path.display().to_string(),
                    holder: holder_description(&owner_path),
                    mode,
                })
            }
            Err(TryLockError::Error(e)) => {
                return Err(StorageError::Msg(format!(
                    "cannot lock `{}` as {mode}: {e}",
                    path.display()
                )))
            }
        }

        let mut lock = Self { path, owner_path, file: Some(file), mode };
        lock.write_owner();
        Ok(lock)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn mode(&self) -> OpenMode {
        self.mode
    }

    /// The recorded holder of this directory (itself, after acquisition).
    pub fn owner(&self) -> Option<LockOwner> {
        read_owner(&self.owner_path)
    }

    /// Best-effort record of who we are, for the *next* process's error message.
    ///
    /// A failure here must not fail the open: the lock is already held and the
    /// record only improves diagnostics. Losing it degrades a refusal to
    /// "held by another process" instead of naming the pid.
    fn write_owner(&mut self) {
        if self.file.is_none() {
            return;
        }
        let owner = LockOwner {
            pid: std::process::id(),
            host: hostname(),
            started_at: unix_now(),
            mode: self.mode,
        };
        let Ok(bytes) = serde_json::to_vec_pretty(&owner) else { return };
        // Atomic so a refused process never reads half a record.
        let _ = crate::atomic::write_atomic(&self.owner_path, &bytes);
    }
}

/// Read the recorded holder from `OWNER.json`, if there is one.
pub fn read_owner(path: impl AsRef<Path>) -> Option<LockOwner> {
    let bytes = fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn holder_description(owner_path: &Path) -> String {
    match read_owner(owner_path) {
        Some(o) => o.describe(),
        // A lock is held but the record is missing (crash between the lock and
        // the record), truncated or from a future format. Say exactly that.
        None => "another process (holder record unavailable)".to_string(),
    }
}

fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// `std` has no hostname API and pulling in a crate for a diagnostic string is
/// not worth it: the shell export is what the operator would see anyway.
fn hostname() -> String {
    std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown-host".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root(label: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("gtv_lock_{}_{}", std::process::id(), label));
        let _ = fs::remove_dir_all(&p);
        p
    }

    #[test]
    fn mode_round_trips_through_its_spelling() {
        for m in [
            OpenMode::ReadWriteExclusive,
            OpenMode::ReadOnlyShared,
            OpenMode::ReadOnlySharedLocked,
        ] {
            assert_eq!(OpenMode::parse(m.as_str()), Some(m));
            assert_eq!(m.is_read_only(), m != OpenMode::ReadWriteExclusive);
        }
        assert_eq!(OpenMode::parse("nonsense"), None);
    }

    #[test]
    fn exclusive_lock_excludes_a_second_exclusive_and_names_the_holder() {
        let root = root("excl");
        let first = StoreLock::acquire(&root, OpenMode::ReadWriteExclusive).unwrap();
        let pid = std::process::id();

        let err = StoreLock::acquire(&root, OpenMode::ReadWriteExclusive).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains(&format!("pid {pid}")),
            "refusal must name the holder, got: {msg}"
        );
        assert!(msg.contains("rw-exclusive"), "got: {msg}");

        drop(first);
        let _again = StoreLock::acquire(&root, OpenMode::ReadWriteExclusive).unwrap();
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn shared_locked_readers_coexist_but_exclude_a_writer() {
        let root = root("shared");
        // Read-only modes never initialise a directory, so the test does.
        fs::create_dir_all(&root).unwrap();
        let a = StoreLock::acquire(&root, OpenMode::ReadOnlySharedLocked).unwrap();
        let b = StoreLock::acquire(&root, OpenMode::ReadOnlySharedLocked).unwrap();

        let err = StoreLock::acquire(&root, OpenMode::ReadWriteExclusive).unwrap_err();
        assert!(matches!(err, StorageError::Locked { .. }), "got: {err}");

        drop((a, b));
        let _w = StoreLock::acquire(&root, OpenMode::ReadWriteExclusive).unwrap();
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn unlocked_read_only_ignores_a_held_exclusive_lock() {
        // `ro-shared` promises a pinned manifest, not exclusivity. Refusing it
        // would make a report replica impossible on a live writer.
        let root = root("ro_shared");
        let _w = StoreLock::acquire(&root, OpenMode::ReadWriteExclusive).unwrap();
        let ro = StoreLock::acquire(&root, OpenMode::ReadOnlyShared).unwrap();
        assert_eq!(ro.mode(), OpenMode::ReadOnlyShared);
        // It wrote no record of its own, so the file still describes the writer.
        assert_eq!(
            ro.owner().map(|o| o.mode),
            Some(OpenMode::ReadWriteExclusive)
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn read_only_modes_do_not_create_a_missing_directory() {
        let missing = root("missing").join("nope");
        let err = StoreLock::acquire(&missing, OpenMode::ReadOnlyShared).unwrap_err();
        assert!(err.to_string().contains("no data directory"), "got: {err}");
        assert!(!missing.exists(), "read-only open must not create the dir");
    }

    #[test]
    fn a_stale_record_never_blocks_a_restart() {
        // Simulates `kill -9`: the file survives with a dead holder's record,
        // but the OS lock is gone. The record must not be trusted.
        let root = root("stale");
        fs::create_dir_all(&root).unwrap();
        let stale = LockOwner {
            pid: 999_999,
            host: "dead-host".into(),
            started_at: 1,
            mode: OpenMode::ReadWriteExclusive,
        };
        fs::write(root.join(OWNER_FILE), serde_json::to_vec(&stale).unwrap()).unwrap();
        fs::write(root.join(LOCK_FILE), b"").unwrap();

        let lock = StoreLock::acquire(&root, OpenMode::ReadWriteExclusive)
            .expect("stale lock file must not block startup");
        assert_eq!(lock.owner().unwrap().pid, std::process::id());
        let _ = fs::remove_dir_all(&root);
    }
}
