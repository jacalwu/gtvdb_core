//! P1 write-path integration tests.
//!
//! Data discipline: every edge below is constructed in memory and every file
//! lives under a tempdir. **No CSV is written anywhere** — `assert_no_csv`
//! walks the whole store root at the end of each test. The repository has a
//! hard rule against committing CSV (`.gitignore` + the `generated-data` CI
//! job), so the tests must not create a file that someone could `git add -f`.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use gtv_storage::{EdgeRecord, SegmentStore, Wal, WalFrame};

fn temp_root(label: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "gtv_p1_{}_{}",
        std::process::id(),
        label
    ));
    let _ = fs::remove_dir_all(&p);
    p
}

fn edges(start: u64, count: u64, system_ts: i64) -> Vec<EdgeRecord> {
    (start..start + count)
        .map(|i| {
            EdgeRecord::new(
                i % 128,
                (i * 7 + 3) % 128,
                100 + i as i64,
                100_000 + i as i64,
                (i % 3) as u16,
                system_ts,
            )
        })
        .collect()
}

fn assert_no_csv(root: &Path) {
    fn walk(dir: &Path, found: &mut Vec<PathBuf>) {
        if let Ok(rd) = fs::read_dir(dir) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    walk(&p, found);
                } else if p.extension().and_then(|s| s.to_str()) == Some("csv") {
                    found.push(p);
                }
            }
        }
    }
    let mut found = Vec::new();
    walk(root, &mut found);
    assert!(
        found.is_empty(),
        "P1 wrote CSV data (hard rule violation): {found:?}"
    );
}

#[test]
fn flush_publishes_mmap_segments_and_reuses_history() {
    let root = temp_root("reuse");
    let mut store = SegmentStore::open(&root, 128).unwrap();
    store.set_flush_threshold(usize::MAX);

    let a = edges(0, 100, 10);
    store.ingest(0, 0, &a).unwrap();
    let r1 = store.flush_at(10).unwrap();
    assert_eq!(r1.new_segments, 1);
    assert_eq!(r1.reused_segments, 0);
    assert_eq!(r1.edges_flushed, 100);
    assert_eq!(r1.checkpoint_txn, 1);
    assert!(r1.wal_bytes_reclaimed > 0);
    assert_eq!(store.verify_all().unwrap(), 1);

    let first_path = store.segment_path(&store.segments()[0]);
    let first_bytes = fs::read(&first_path).unwrap();

    // A second flush touches only a different partition. The first segment
    // must be byte-for-byte untouched (OceanBase-style macroblock reuse).
    let b = edges(100, 50, 20);
    store.ingest(0, 7, &b).unwrap();
    let r2 = store.flush_at(20).unwrap();
    assert_eq!(r2.new_segments, 1);
    assert_eq!(r2.reused_segments, 1);
    assert_eq!(r2.reuse_ratio, 0.5);
    assert_eq!(r2.checkpoint_txn, 2);
    assert_eq!(fs::read(&first_path).unwrap(), first_bytes);
    assert_eq!(store.verify_all().unwrap(), 2);

    // as_of_system: T=15 sees only the first baseline; T=25 sees both.
    let g15 = store.graph_at(15).unwrap();
    assert_eq!(g15.edge_count(), 100);
    let g25 = store.graph_at(25).unwrap();
    assert_eq!(g25.edge_count(), 150);
    assert_eq!(g25.node_count(), 128);
    assert!(g25.degree(0).unwrap() >= 1);
    assert!(g25.neighbors(0, 500).unwrap().count() >= 1);

    // The read path is a single contiguous CSR, not a k-way merged view.
    assert_eq!(store.entries_at_system(25).len(), 2);
    assert_eq!(store.entries_at_system(15).len(), 1);

    assert_no_csv(&root);
    drop(store);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn crash_before_flush_replays_wal_then_flushes_exactly_once() {
    let root = temp_root("replay");

    {
        let mut store = SegmentStore::open(&root, 128).unwrap();
        store.set_flush_threshold(usize::MAX);
        store.ingest(0, 0, &edges(0, 40, 1)).unwrap();
        store.ingest(0, 0, &edges(40, 30, 2)).unwrap();
        store.ingest(1, 2, &edges(70, 20, 3)).unwrap();
        assert_eq!(store.pending_edges(), 90);
        // Drop without flush == kill -9. WAL frames are already fsync'd.
    }

    let mut store = SegmentStore::open(&root, 128).unwrap();
    store.set_flush_threshold(usize::MAX);
    let rec = store.recovery().clone();
    assert_eq!(rec.frames_scanned, 3);
    assert_eq!(rec.frames_replayed, 3);
    assert_eq!(rec.edges_replayed, 90);
    assert_eq!(rec.torn_tail_bytes, 0);
    assert_eq!(store.pending_edges(), 90);
    assert_eq!(store.graph_at(i64::MAX).unwrap().edge_count(), 90);

    let r = store.flush_at(100).unwrap();
    assert_eq!(r.edges_flushed, 90);
    assert_eq!(r.new_segments, 2); // two (table, partition) groups
    assert_eq!(r.checkpoint_txn, 3);
    assert_eq!(store.pending_edges(), 0);

    // Reopen after checkpoint: no frame is replayed, no edge is duplicated.
    let store2 = SegmentStore::open(&root, 128).unwrap();
    assert_eq!(store2.recovery().frames_replayed, 0);
    assert_eq!(store2.pending_edges(), 0);
    assert_eq!(store2.graph_at(i64::MAX).unwrap().edge_count(), 90);
    assert_eq!(store2.verify_all().unwrap(), 2);
    assert_no_csv(&root);

    drop(store);
    drop(store2);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn torn_wal_tail_is_truncated_and_ingest_continues() {
    let root = temp_root("torn");
    {
        let mut store = SegmentStore::open(&root, 128).unwrap();
        store.set_flush_threshold(usize::MAX);
        store.ingest(0, 0, &edges(0, 10, 1)).unwrap();
        store.ingest(0, 0, &edges(10, 10, 2)).unwrap();
    }
    // Simulate a process dying halfway through the third frame.
    {
        let mut f = fs::OpenOptions::new()
            .append(true)
            .open(root.join("wal.log"))
            .unwrap();
        f.write_all(b"half-written-frame").unwrap();
        f.sync_all().unwrap();
    }

    let mut store = SegmentStore::open(&root, 128).unwrap();
    store.set_flush_threshold(usize::MAX);
    assert_eq!(store.recovery().torn_tail_bytes, 18);
    assert_eq!(store.recovery().frames_replayed, 2);
    assert_eq!(store.pending_edges(), 20);
    assert!(store.wal_len() > 64);

    // Appending after truncation must land on a clean frame boundary.
    store.ingest(0, 0, &edges(20, 5, 3)).unwrap();
    store.flush_at(50).unwrap();
    let reopened = SegmentStore::open(&root, 128).unwrap();
    assert_eq!(reopened.recovery().torn_tail_bytes, 0);
    assert_eq!(reopened.recovery().frames_replayed, 0);
    assert_eq!(reopened.graph_at(i64::MAX).unwrap().edge_count(), 25);
    assert_no_csv(&root);

    drop(store);
    drop(reopened);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn checkpoint_skips_stale_frames_and_orphan_segments_are_ignored() {
    let root = temp_root("checkpoint");
    {
        let mut store = SegmentStore::open(&root, 128).unwrap();
        store.set_flush_threshold(usize::MAX);
        store.ingest(0, 0, &edges(0, 25, 5)).unwrap();
        store.flush_at(10).unwrap();
        let checkpoint = store.checkpoint_txn();

        // A crash between the manifest write and WAL truncation leaves a
        // frame that is already covered by the published segment.
        let (mut wal, _) = Wal::open(root.join("wal.log")).unwrap();
        wal.append(&WalFrame::insert(
            checkpoint,
            0,
            0,
            edges(0, 25, 5),
        ))
        .unwrap();

        // An orphan .seg from a failed flush is not in the manifest.
        let entry = store.segments()[0].clone();
        let orphan = root.join("segments").join("seg_99999999_t0_p0.seg");
        fs::copy(store.segment_path(&entry), &orphan).unwrap();
    }

    let store = SegmentStore::open(&root, 128).unwrap();
    assert_eq!(store.recovery().frames_skipped_checkpoint, 1);
    assert_eq!(store.recovery().frames_replayed, 0);
    assert_eq!(store.segments().len(), 1);
    assert_eq!(store.graph_at(i64::MAX).unwrap().edge_count(), 25);
    assert_no_csv(&root);

    drop(store);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn corrupted_segment_is_rejected_by_manifest_checksum() {
    let root = temp_root("tamper");
    {
        let mut store = SegmentStore::open(&root, 128).unwrap();
        store.set_flush_threshold(usize::MAX);
        store.ingest(0, 0, &edges(0, 30, 1)).unwrap();
        store.flush_at(10).unwrap();
    }
    let store = SegmentStore::open(&root, 128).unwrap();
    let path = store.segment_path(&store.segments()[0]);
    // Corrupt a `dst` value (column 3). Changing an `offsets` word instead
    // would fail CSR structural validation before the checksum is reached.
    let at = {
        let seg = store.open_entry(&store.segments()[0]).unwrap();
        seg.header().columns[3].offset + 8
    };
    let mut bytes = fs::read(&path).unwrap();
    bytes[at] ^= 0xff;
    fs::write(&path, &bytes).unwrap();
    let e = store.verify_all().unwrap_err();
    assert!(
        e.to_string().contains("data checksum mismatch"),
        "unexpected error: {e}"
    );
    assert_no_csv(&root);

    drop(store);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn empty_flush_is_a_noop_and_node_count_is_pinned() {
    let root = temp_root("noop");
    let mut store = SegmentStore::open(&root, 128).unwrap();
    let r = store.flush_at(42).unwrap();
    assert_eq!(r.new_segments, 0);
    assert_eq!(r.edges_flushed, 0);
    assert_eq!(r.reuse_ratio, 1.0);
    assert!(store.segments().is_empty());
    drop(store);

    let e = SegmentStore::open(&root, 256).unwrap_err();
    assert!(e.to_string().contains("node_count"), "unexpected error: {e}");
    assert_no_csv(&root);
    let _ = fs::remove_dir_all(&root);
}
