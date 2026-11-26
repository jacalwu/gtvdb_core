//! Integration tests for the versioned index store (B2-2).

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use gtv_core::{Metric, VectorIndex, VectorSpace};
use gtv_index::{AnyIndex, BuildOptions};
use gtv_index_store::{IndexMeta, IndexStore};

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn f32(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn vectors(n: usize, dim: usize, seed: u64) -> (Vec<u64>, Vec<Vec<f32>>) {
    let mut rng = Lcg(seed);
    let ids = (0..n as u64).collect();
    let vs = (0..n)
        .map(|_| (0..dim).map(|_| rng.f32()).collect())
        .collect();
    (ids, vs)
}

fn root(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("gtv_idx_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn meta() -> IndexMeta {
    IndexMeta {
        model_id: "bge".into(),
        model_version: "1".into(),
        embedding_model: "bge-small".into(),
        ..Default::default()
    }
}

fn save_load_roundtrip(tag: &str, options: BuildOptions, metric: Metric) {
    let store = IndexStore::open(root(tag)).unwrap();
    let (ids, vs) = vectors(128, 8, 7);
    let index = AnyIndex::build(ids, vs.clone(), metric, &options).unwrap();
    let v = store.save(tag, &index, &options, &meta()).unwrap();
    assert_eq!(v.version, 1);
    assert_eq!(store.current_version(tag).unwrap(), Some(1));

    let loaded = store.load(tag, None).unwrap();
    assert_eq!(loaded.manifest.row_count, 128);
    assert_eq!(loaded.manifest.metric, metric.as_str());
    assert_eq!(loaded.index.metric(), metric);
    assert_eq!(loaded.index.len(), 128);
    // The embedding space is derived from (model, version, feature, metric, dim)
    // and survives the save/load round-trip unchanged.
    let expected_space = VectorSpace::new("bge", "1", "", metric, 8).id();
    assert_eq!(loaded.vector_space_id(), Some(expected_space));

    let q = vs[3].clone();
    let a: Vec<u64> = index
        .search(&q, 5, None)
        .unwrap()
        .iter()
        .map(|h| h.id)
        .collect();
    let b: Vec<u64> = loaded
        .index
        .search(&q, 5, None)
        .unwrap()
        .iter()
        .map(|h| h.id)
        .collect();
    assert_eq!(a, b);
    // Exact indexes must recover the query vector itself.
    if !matches!(options, BuildOptions::Hnsw { .. }) {
        assert_eq!(b[0], 3);
    }
}

#[test]
fn flat_round_trip() {
    save_load_roundtrip("flat", BuildOptions::Flat, Metric::L2);
}

#[test]
fn ivf_round_trip() {
    save_load_roundtrip(
        "ivf",
        BuildOptions::Ivf {
            nlist: 8,
            nprobe: 8,
        },
        Metric::L2,
    );
}

#[test]
fn hnsw_round_trip() {
    save_load_roundtrip(
        "hnsw",
        BuildOptions::Hnsw {
            m: 16,
            ef_construction: 100,
            ef_search: 100,
        },
        Metric::Cosine,
    );
}

#[test]
fn shadow_build_then_swap_and_rollback() {
    let tag = "swap";
    let store = IndexStore::open(root(tag)).unwrap();
    let (ids, vs) = vectors(64, 4, 1);
    let v1_index = AnyIndex::build(ids.clone(), vs.clone(), Metric::L2, &BuildOptions::Flat).unwrap();
    store
        .save(tag, &v1_index, &BuildOptions::Flat, &meta())
        .unwrap();

    // Shadow build v2 (different corpus size) without touching CURRENT.
    let (ids2, vs2) = vectors(96, 4, 2);
    let v2_index = AnyIndex::build(ids2, vs2, Metric::L2, &BuildOptions::Flat).unwrap();
    let v2 = store
        .build(tag, &v2_index, &BuildOptions::Flat, &meta())
        .unwrap();
    assert_eq!(v2.version, 2);
    assert_eq!(store.current_version(tag).unwrap(), Some(1));
    assert_eq!(store.load(tag, None).unwrap().manifest.row_count, 64);

    // Swap to v2, then roll back to v1.
    store.activate(tag, 2).unwrap();
    assert_eq!(store.current_version(tag).unwrap(), Some(2));
    assert_eq!(store.load(tag, None).unwrap().manifest.row_count, 96);
    store.rollback(tag, 1).unwrap();
    assert_eq!(store.load(tag, None).unwrap().manifest.row_count, 64);

    assert_eq!(store.versions(tag).unwrap(), vec![1, 2]);
}

#[test]
fn rebuild_from_vectors_matches_direct_build() {
    let store = IndexStore::open(root("rebuild")).unwrap();
    let (ids, vs) = vectors(200, 6, 5);
    let opts = BuildOptions::Ivf {
        nlist: 16,
        nprobe: 4,
    };
    let built = AnyIndex::build(ids.clone(), vs.clone(), Metric::L2, &opts).unwrap();
    store.save("rb", &built, &opts, &meta()).unwrap();
    let loaded = store.load("rb", None).unwrap();

    // "Rebuild" = build again from the same authoritative vectors.
    let rebuilt = AnyIndex::build(ids, vs.clone(), Metric::L2, &opts).unwrap();
    let q = vs[10].clone();
    let a: Vec<u64> = loaded.index.search(&q, 5, None).unwrap().iter().map(|h| h.id).collect();
    let b: Vec<u64> = rebuilt.search(&q, 5, None).unwrap().iter().map(|h| h.id).collect();
    assert_eq!(a, b);
}

#[test]
fn corruption_is_detected() {
    let tag = "corrupt";
    let store = IndexStore::open(root(tag)).unwrap();
    let (ids, vs) = vectors(32, 4, 9);
    let index = AnyIndex::build(ids, vs, Metric::L2, &BuildOptions::Flat).unwrap();
    store.save(tag, &index, &BuildOptions::Flat, &meta()).unwrap();

    let id = store.resolve(tag).unwrap();
    let path = store
        .root()
        .join(id.to_string())
        .join("v1")
        .join("index.gtvidx");
    let mut bytes = std::fs::read(&path).unwrap();
    let n = bytes.len();
    // Flip a byte in the trailing payload/checksum region. For a GTVVEC payload
    // this lands in the data column (past the header), which `load` does *not*
    // hash on the O(1) cold-open path — but `verify` must catch it.
    bytes[n - 40] ^= 0xff;
    std::fs::write(&path, &bytes).unwrap();

    // Cold open stays O(1) and does not detect the data-column corruption...
    assert!(store.load(tag, None).is_ok());
    // ...but the explicit integrity check does.
    assert!(store.verify(tag, 1).is_err());
}

#[test]
fn gtvvec_header_corruption_fails_load() {
    let tag = "hcorrupt";
    let store = IndexStore::open(root(tag)).unwrap();
    let (ids, vs) = vectors(32, 4, 9);
    let index = AnyIndex::build(ids, vs, Metric::L2, &BuildOptions::Flat).unwrap();
    store.save(tag, &index, &BuildOptions::Flat, &meta()).unwrap();

    let id = store.resolve(tag).unwrap();
    let path = store
        .root()
        .join(id.to_string())
        .join("v1")
        .join("index.gtvidx");
    let bytes = std::fs::read(&path).unwrap();
    let mut corrupted = bytes.clone();
    // The GTVVEC payload begins at `16 + manifest_len`; flip a byte inside its
    // 4096-byte header (row_count field lives at payload offset 12).
    let manifest_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let payload_offset = 16 + manifest_len;
    corrupted[payload_offset + 12] ^= 0xff;
    std::fs::write(&path, &corrupted).unwrap();

    // Header corruption is caught on open (header_checksum is always verified).
    assert!(store.load(tag, None).is_err());
}

#[test]
fn unknown_version_and_name_are_typed_errors() {
    let store = IndexStore::open(root("missing")).unwrap();
    assert!(store.resolve("nope").is_err());
    let (ids, vs) = vectors(8, 2, 1);
    let index = AnyIndex::build(ids, vs, Metric::L2, &BuildOptions::Flat).unwrap();
    store.save("x", &index, &BuildOptions::Flat, &meta()).unwrap();
    assert!(store.load("x", Some(99)).is_err());
    assert!(store.activate("x", 99).is_err());
}

/// V-4: an IVF index is persisted as a compound `GIVFCMP` payload (cell layout
/// + GTVVEC corpus), so `load` must take the mmap cold-open path and still
/// produce bit-identical query results.
#[test]
fn ivf_uses_the_compound_mmap_payload() {
    let tag = "ivfcompound";
    let store = IndexStore::open(root(tag)).unwrap();
    let (ids, vs) = vectors(256, 8, 11);
    let opts = BuildOptions::Ivf {
        nlist: 8,
        nprobe: 8,
    };
    let index = AnyIndex::build(ids, vs.clone(), Metric::L2, &opts).unwrap();
    store.save(tag, &index, &opts, &meta()).unwrap();

    // The container payload must carry the compound magic.
    let id = store.resolve(tag).unwrap();
    let path = store
        .root()
        .join(id.to_string())
        .join("v1")
        .join("index.gtvidx");
    let bytes = std::fs::read(&path).unwrap();
    let manifest_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let payload_offset = 16 + manifest_len;
    assert_eq!(&bytes[payload_offset..payload_offset + 8], b"GIVFCMP\0");

    // The store recognizes it and mmaps it.
    let kind = gtv_index_store::container_payload_kind(&path).unwrap();
    assert_eq!(kind, gtv_index_store::PayloadKind::IvfCompound);

    // Reopening must reproduce the index exactly (corpus order included).
    let loaded = store.load(tag, None).unwrap();
    let original = index.as_ivf().unwrap();
    let reopened = loaded.index.as_ivf().unwrap();
    assert_eq!(reopened.ids(), original.ids());
    assert_eq!(reopened.data(), original.data());
    assert_eq!(reopened.centroids(), original.centroids());
    assert_eq!(reopened.list_offsets(), original.list_offsets());
    assert_eq!(reopened.orig_pos(), original.orig_pos());
    assert_eq!(reopened.nlist(), original.nlist());
    assert_eq!(reopened.nprobe(), original.nprobe());

    // And the query answers must be identical.
    let q = vs[17].clone();
    let a: Vec<u64> = index
        .search(&q, 10, None)
        .unwrap()
        .iter()
        .map(|h| h.id)
        .collect();
    let b: Vec<u64> = loaded
        .index
        .search(&q, 10, None)
        .unwrap()
        .iter()
        .map(|h| h.id)
        .collect();
    assert_eq!(a, b);

    // `verify` runs the full data-integrity check over the mapped corpus.
    assert!(store.verify(tag, 1).is_ok());
}

/// V-4: corrupting the IVF corpus must be caught by `verify` (the cold open
/// deliberately only validates the header).
#[test]
fn ivf_corpus_corruption_is_caught_by_verify() {
    let tag = "ivfcorrupt";
    let store = IndexStore::open(root(tag)).unwrap();
    let (ids, vs) = vectors(64, 4, 3);
    let opts = BuildOptions::Ivf {
        nlist: 4,
        nprobe: 4,
    };
    let index = AnyIndex::build(ids, vs, Metric::L2, &opts).unwrap();
    store.save(tag, &index, &opts, &meta()).unwrap();

    let id = store.resolve(tag).unwrap();
    let path = store
        .root()
        .join(id.to_string())
        .join("v1")
        .join("index.gtvidx");
    let mut bytes = std::fs::read(&path).unwrap();
    let n = bytes.len();
    bytes[n - 40] ^= 0xff;
    std::fs::write(&path, &bytes).unwrap();

    assert!(store.verify(tag, 1).is_err());
}

/// B2-2: a shadow build + atomic swap must never break a live reader. Four
/// reader threads hammer `load(name, None)` (always the active version) while
/// v2 is built and `CURRENT` is swapped; every read must succeed and return
/// either the old or the new row count.
#[test]
fn concurrent_readers_survive_shadow_swap() {
    let tag = "concurrent";
    let store = IndexStore::open(root(tag)).unwrap();
    let (ids, vs) = vectors(64, 4, 1);
    let v1 = AnyIndex::build(ids, vs, Metric::L2, &BuildOptions::Flat).unwrap();
    store.save(tag, &v1, &BuildOptions::Flat, &meta()).unwrap();

    let stop = Arc::new(AtomicBool::new(false));
    let errors = Arc::new(AtomicUsize::new(0));
    let reads = Arc::new(AtomicUsize::new(0));
    let mut handles = Vec::new();
    for _ in 0..4 {
        let store = store.clone();
        let stop = stop.clone();
        let errors = errors.clone();
        let reads = reads.clone();
        handles.push(std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                match store.load(tag, None) {
                    Ok(loaded) => {
                        let n = loaded.manifest.row_count;
                        if n != 64 && n != 96 {
                            errors.fetch_add(1, Ordering::Relaxed);
                        }
                        if loaded.index.search(&[0.0f32; 4], 3, None).is_err() {
                            errors.fetch_add(1, Ordering::Relaxed);
                        }
                        reads.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(_) => {
                        errors.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }));
    }

    // Shadow build v2, let readers observe v1, then atomically activate v2.
    let (ids2, vs2) = vectors(96, 4, 2);
    let v2 = AnyIndex::build(ids2, vs2, Metric::L2, &BuildOptions::Flat).unwrap();
    store.build(tag, &v2, &BuildOptions::Flat, &meta()).unwrap();
    std::thread::sleep(Duration::from_millis(20));
    store.activate(tag, 2).unwrap();
    std::thread::sleep(Duration::from_millis(50));
    stop.store(true, Ordering::Relaxed);
    for h in handles {
        h.join().unwrap();
    }

    assert!(reads.load(Ordering::Relaxed) > 0, "readers made progress");
    assert_eq!(errors.load(Ordering::Relaxed), 0, "no reader was disrupted");
    assert_eq!(store.current_version(tag).unwrap(), Some(2));
}

/// The vector-space id is a first-class manifest field: derived at build time
/// from the descriptor, distinct for a same-dimension different model, and
/// preserved verbatim when an explicit id is supplied (save/load round-trip).
#[test]
fn vector_space_id_is_persisted_model_sensitive_and_overridable() {
    let store = IndexStore::open(root("vspace")).unwrap();
    let (ids, vs) = vectors(16, 8, 3);
    let index = AnyIndex::build(ids, vs, Metric::Cosine, &BuildOptions::Flat).unwrap();

    let mut m1 = meta();
    m1.feature_version = "feat-v1".into();
    store.save("a", &index, &BuildOptions::Flat, &m1).unwrap();

    let mut m2 = meta();
    m2.model_id = "other-model".into();
    m2.feature_version = "feat-v1".into();
    store.save("b", &index, &BuildOptions::Flat, &m2).unwrap();

    let a = store.load("a", None).unwrap().vector_space_id().unwrap();
    let b = store.load("b", None).unwrap().vector_space_id().unwrap();
    assert_eq!(
        a,
        VectorSpace::new("bge", "1", "feat-v1", Metric::Cosine, 8).id()
    );
    assert_ne!(a, b, "same dim/metric but a different model must be a different space");

    // An explicit id passes straight through, so re-saving a bound index does
    // not re-key its space.
    let mut m3 = meta();
    m3.vector_space_id = Some(0xDEAD_BEEF);
    store.save("c", &index, &BuildOptions::Flat, &m3).unwrap();
    assert_eq!(
        store.load("c", None).unwrap().vector_space_id(),
        Some(0xDEAD_BEEF)
    );

    // The sibling-manifest cold-open helper surfaces the same id.
    let container = store
        .root()
        .join(store.resolve("a").unwrap().to_string())
        .join("v1")
        .join("index.gtvidx");
    let (_, manifest) = gtv_index_store::load_index_with_manifest(&container).unwrap();
    assert_eq!(manifest.and_then(|m| m.vector_space_id), Some(a));
}
