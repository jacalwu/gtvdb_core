//! P2 directory ownership: one writer, honest refusals, and a readable replica.
//!
//! These are integration tests because the property under test is *across*
//! handles and across a simulated crash — the unit tests in `lock.rs` cover the
//! guard itself, and cannot show that the store replays the WAL exactly once or
//! that a read-only handle leaves the directory byte-identical.
//!
//! Data discipline: every edge is built in memory, every file lives under a
//! tempdir, no CSV is written.

use std::fs;
use std::path::{Path, PathBuf};

use gtv_storage::store::SegmentStore;
use gtv_storage::{EdgeRecord, OpenMode, StorageError, LOCK_FILE};

fn temp_root(label: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("gtv_p2_{}_{}", std::process::id(), label));
    let _ = fs::remove_dir_all(&p);
    p
}

fn edges(start: u64, count: u64) -> Vec<EdgeRecord> {
    (start..start + count)
        .map(|i| EdgeRecord::new(i % 16, (i * 7 + 3) % 16, 100 + i as i64, 100_000, 1, 0))
        .collect()
}

fn cleanup(root: &Path) {
    let _ = fs::remove_dir_all(root);
}

/// R5 + R6 in one shape: the second writer is refused *by the lock*, that
/// refusal cannot have touched the WAL, and after the holder goes away the
/// frames are replayed exactly once.
#[test]
fn a_refused_open_does_not_replay_the_wal_and_recovery_happens_once() {
    let root = temp_root("replay_once");

    let mut writer = SegmentStore::open(&root, 16).unwrap();
    for i in 0..3u32 {
        writer.ingest(1, 0, &edges(i as u64 * 8, 4)).unwrap();
    }
    assert_eq!(writer.pending_edges(), 12);

    // The second writer must be refused, and the refusal must name the holder.
    let err = SegmentStore::open(&root, 16).unwrap_err();
    let msg = err.to_string();
    assert!(
        matches!(err, StorageError::Locked { .. }),
        "expected a locked error, got: {msg}"
    );
    assert!(
        msg.contains(&format!("pid {}", std::process::id())),
        "refusal must name the holder: {msg}"
    );
    // Nothing was replayed into a second in-memory segment behind our back.
    assert_eq!(writer.pending_edges(), 12);

    drop(writer);

    // Restart: the three frames come back exactly once.
    let store = SegmentStore::open_existing(&root, OpenMode::ReadWriteExclusive).unwrap();
    assert_eq!(store.recovery().frames_replayed, 3);
    assert_eq!(store.recovery().edges_replayed, 12);
    assert_eq!(store.pending_edges(), 12);

    // Flushing checkpoints them, so the next open replays nothing.
    let mut store = store;
    store.flush_at(1_000).unwrap();
    drop(store);

    let store = SegmentStore::open_existing(&root, OpenMode::ReadWriteExclusive).unwrap();
    assert_eq!(store.recovery().frames_replayed, 0);
    // `flush_at` checkpoints, which truncates the log: there is nothing left to
    // replay, so "exactly once" holds by having nothing to replay twice.
    assert_eq!(store.recovery().frames_scanned, 0);
    assert_eq!(store.pending_edges(), 0);
    drop(store);

    cleanup(&root);
}

/// A read-only handle must refuse every mutation, not merely "not be given
/// `&mut`". The manifest is compared byte-for-byte afterwards: a refusal that
/// still wrote something is worse than no refusal.
#[test]
fn a_read_only_handle_refuses_every_write() {
    let root = temp_root("read_only_writes");
    {
        let mut s = SegmentStore::open(&root, 16).unwrap();
        s.ingest(1, 0, &edges(0, 8)).unwrap();
        s.flush_at(500).unwrap();
        s.create_tag("baseline", 500).unwrap();
    }
    let manifest_path = root.join("manifest.json");
    let before = fs::read(&manifest_path).unwrap();
    let segments_before = fs::read_dir(root.join("segments")).unwrap().count();

    let mut ro = SegmentStore::open_existing(&root, OpenMode::ReadOnlyShared).unwrap();
    assert_eq!(ro.mode(), OpenMode::ReadOnlyShared);
    assert_eq!(ro.system_ts(), 500);

    let refusals = [
        ro.ingest(1, 0, &edges(100, 2)).map(|_| ()),
        ro.flush_at(600).map(|_| ()),
        ro.create_tag("after", 500).map(|_| ()),
        ro.drop_tag("baseline").map(|_| ()),
        ro.gc(gtv_storage::GcPolicy::collect(0)).map(|_| ()),
        ro.publish_observation("cpi", 600, vec![]).map(|_| ()),
    ];
    for r in &refusals {
        let err = r.as_ref().unwrap_err().to_string();
        assert!(err.contains("refusing to write"), "got: {err}");
        assert!(err.contains("ro-shared"), "the mode must be named: {err}");
    }

    assert_eq!(
        fs::read(&manifest_path).unwrap(),
        before,
        "a refused write must leave the manifest untouched"
    );
    assert_eq!(
        fs::read_dir(root.join("segments")).unwrap().count(),
        segments_before,
        "a refused write must not publish a segment"
    );
    drop(ro);
    cleanup(&root);
}

/// `ro-shared` pins the manifest it read: a concurrent writer's new snapshot is
/// invisible to the replica. That is the whole point of the mode — a report
/// must not change shape halfway through.
#[test]
fn a_read_only_handle_keeps_the_snapshot_it_opened_with() {
    let root = temp_root("pinned");
    {
        let mut s = SegmentStore::open(&root, 16).unwrap();
        s.ingest(1, 0, &edges(0, 8)).unwrap();
        s.flush_at(1_000).unwrap();
    }

    let ro = SegmentStore::open_existing(&root, OpenMode::ReadOnlyShared).unwrap();
    assert_eq!(ro.system_ts(), 1_000);
    let segments = ro.segments().len();
    assert!(segments >= 1);

    // A writer may still open the directory: `ro-shared` takes no lock.
    {
        let mut w = SegmentStore::open_existing(&root, OpenMode::ReadWriteExclusive).unwrap();
        w.ingest(1, 0, &edges(50, 8)).unwrap();
        w.flush_at(2_000).unwrap();
    }

    assert_eq!(ro.system_ts(), 1_000, "the replica must not follow the writer");
    assert_eq!(ro.segments().len(), segments);
    drop(ro);
    cleanup(&root);
}

/// `ro-shared-locked` is the mode for a replica that must not have its files
/// replaced underneath it: it takes a shared lock, so the writer is refused
/// while it lives, and released again when it drops.
#[test]
fn a_shared_locked_replica_excludes_the_writer_until_it_closes() {
    let root = temp_root("shared_locked");
    {
        let mut s = SegmentStore::open(&root, 16).unwrap();
        s.ingest(1, 0, &edges(0, 8)).unwrap();
        s.flush_at(1_000).unwrap();
    }

    let replica = SegmentStore::open_existing(&root, OpenMode::ReadOnlySharedLocked).unwrap();
    let err = SegmentStore::open_existing(&root, OpenMode::ReadWriteExclusive).unwrap_err();
    assert!(matches!(err, StorageError::Locked { .. }), "got: {err}");
    assert_eq!(replica.mode(), OpenMode::ReadOnlySharedLocked);

    drop(replica);
    let w = SegmentStore::open_existing(&root, OpenMode::ReadWriteExclusive).unwrap();
    assert_eq!(w.mode(), OpenMode::ReadWriteExclusive);
    drop(w);
    cleanup(&root);
}

/// A server restart must not need the operator to restate `node_count`: it is
/// in the manifest, and disagreeing with it is an error either way.
#[test]
fn open_existing_reads_the_node_count_from_the_manifest() {
    let root = temp_root("node_count");
    {
        let s = SegmentStore::open(&root, 24).unwrap();
        assert_eq!(s.node_count(), 24);
    }

    let s = SegmentStore::open_existing(&root, OpenMode::ReadWriteExclusive).unwrap();
    assert_eq!(s.node_count(), 24);
    drop(s);

    let err = SegmentStore::open(&root, 8).unwrap_err().to_string();
    assert!(err.contains("node_count"), "got: {err}");
    cleanup(&root);
}

/// "Open the data directory" on a path that holds no data is a typo. A
/// read-only mode must say so and must not create anything.
#[test]
fn a_read_only_open_never_initialises_a_directory() {
    let root = temp_root("not_a_dir").join("missing");
    let err = SegmentStore::open_existing(&root, OpenMode::ReadOnlyShared).unwrap_err();
    assert!(err.to_string().contains("no data directory"), "got: {err}");
    assert!(!root.exists(), "read-only open created the directory");
}

/// The lock file outlives its holder on purpose (unlinking it would break the
/// lock on Unix). A restart must therefore tolerate — and ignore — its presence.
#[test]
fn the_lock_file_survives_and_never_blocks_a_restart() {
    let root = temp_root("survives");
    {
        let s = SegmentStore::open(&root, 8).unwrap();
        assert!(root.join(LOCK_FILE).exists());
        drop(s);
    }
    assert!(
        root.join(LOCK_FILE).exists(),
        "the lock file must not be unlinked on drop"
    );

    let s = SegmentStore::open(&root, 8).unwrap();
    assert_eq!(s.lock().map(|l| l.mode()), Some(OpenMode::ReadWriteExclusive));
    drop(s);
    cleanup(&root);
}
