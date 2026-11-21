//! G9 acceptance: the store can report the knowledge state a result came from.
//!
//! The receipt needs a watermark and a retention floor; both already exist in
//! the store (manifest `system_ts`, `gc_floor`), so the only risk is that they
//! are read from the wrong place or reported as something they are not. These
//! tests pin the mapping:
//!
//! * `as_known_at` is the newest *published* segment, not the newest ingested
//!   edge (an unflushed WAL record is not yet answerable);
//! * the clock is reported as `SegmentSystemTs` — core must not pretend a
//!   `recorded_at` watermark it never saw;
//! * `history_floor` appears only once delayed GC has actually set one.
//!
//! Files live in a tempdir outside the repository.

use std::fs;
use std::path::PathBuf;

use gtv_core::KnowledgeClock;
use gtv_storage::{EdgeRecord, SegmentStore};

const NODE_COUNT: usize = 8;

fn temp_root(label: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("gtv_fresh_{}_{}", std::process::id(), label));
    let _ = fs::remove_dir_all(&p);
    p
}

#[test]
fn watermark_tracks_publish_not_ingest() {
    let root = temp_root("publish");
    let mut store = SegmentStore::open(&root, NODE_COUNT).unwrap();
    store
        .ingest(7, 3, &[EdgeRecord::new(1, 2, 0, 1_000, 7, 10)])
        .unwrap();

    // Ingested but not flushed: nothing is answerable yet.
    assert_eq!(store.freshness().as_known_at, 0);

    store.flush_at(10).unwrap();
    let freshness = store.freshness();
    assert_eq!(freshness.as_known_at, 10);
    assert_eq!(freshness.clock, KnowledgeClock::SegmentSystemTs);
    assert!(freshness.validate().is_ok());
}

#[test]
fn freshness_reports_business_max_and_has_no_floor_by_default() {
    let root = temp_root("floor");
    let mut store = SegmentStore::open(&root, NODE_COUNT).unwrap();
    store
        .ingest(7, 3, &[EdgeRecord::new(1, 2, 0, 1_000, 7, 10)])
        .unwrap();
    store.flush_at(10).unwrap();

    let freshness = store.freshness();
    assert_eq!(freshness.history_floor, None, "full history retained");
    assert_eq!(freshness.business_max, Some(1_000));
    assert!(freshness.retains_business_time(1));
    // Reading at its own watermark is current; one tick later is stale.
    assert!(freshness.verdict(10).is_current());
    assert!(!freshness.verdict(11).is_current());

    let _ = fs::remove_dir_all(&root);
}
