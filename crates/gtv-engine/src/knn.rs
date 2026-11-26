//! Exact K-NN over named vector collections, exposed to SQL as a table function.
//!
//! `knn(name, query, k [, label])` runs a brute-force L2 nearest-neighbor search
//! over a collection registered ahead of time (via [`GtvContext::register_knn`]),
//! returning `(id, distance)` rows ordered nearest-first (ties broken by
//! ascending id, matching the reference oracle in `testcase/data/gen_data.py`).
//!
//! The optional 4th argument filters by a per-vector metadata label — this is
//! the gtvdb counterpart of KDB.AI's `metadata_filtering` (e.g. genre over
//! song embeddings).

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use arrow::array::BooleanArray;
use datafusion::catalog::{TableFunctionArgs, TableFunctionImpl};
use datafusion::datasource::TableProvider;
use datafusion::error::{DataFusionError, Result};
use gtv_accelerator::{
    AcceleratorRegistry, IvfRequest, KnnRequest, RawHit, ResidentKind, PREFETCH_HEAT_FLOOR,
};
use gtv_core::{GtvError, Metric};
use gtv_index::{AnyIndex, BuildOptions, CorpusView, FlatIndex, IvfIndex, ProbePlan};

use crate::expr_util::{expr_to_i64, expr_to_string};

/// Options for registering an IVF-backed `knn` collection.
///
/// Bundled as a value so the registration entry point stays under the
/// argument-count threshold and the three knobs can be passed together.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IvfOptions {
    /// Distance metric for the collection.
    pub metric: Metric,
    /// Number of Voronoi cells in the coarse quantizer.
    pub nlist: usize,
    /// Cells probed per query (clamped to `nlist` by `IvfIndex`).
    pub nprobe: usize,
}

impl IvfOptions {
    pub fn new(metric: Metric, nlist: usize, nprobe: usize) -> Self {
        Self {
            metric,
            nlist,
            nprobe,
        }
    }
}

/// Which index backs a collection.
///
/// An IVF index keeps the corpus **cell-ordered**: each probed cell is a
/// contiguous run, which is exactly the access shape a device can scan without
/// pointer chasing. The two reprs are kept distinct because their row order —
/// and therefore the meaning of a returned position — is different; the
/// accelerator caches them under different resident keys.
#[derive(Debug, Clone)]
enum Repr {
    Flat(FlatIndex),
    Ivf(IvfIndex),
}

impl Repr {
    fn ids(&self) -> &[u64] {
        match self {
            Repr::Flat(i) => i.ids(),
            Repr::Ivf(i) => i.ids(),
        }
    }

    fn data(&self) -> &[f32] {
        match self {
            Repr::Flat(i) => i.data(),
            Repr::Ivf(i) => i.data(),
        }
    }

    fn dim(&self) -> usize {
        match self {
            Repr::Flat(i) => i.dim(),
            Repr::Ivf(i) => i.dim(),
        }
    }

    fn metric(&self) -> Metric {
        match self {
            Repr::Flat(i) => i.metric(),
            Repr::Ivf(i) => i.metric(),
        }
    }

    fn normalized(&self) -> bool {
        match self {
            Repr::Flat(i) => i.is_normalized(),
            Repr::Ivf(i) => i.is_normalized(),
        }
    }

    fn view(&self) -> CorpusView<'_> {
        CorpusView {
            ids: self.ids(),
            data: self.data(),
            dim: self.dim(),
            metric: self.metric(),
            normalized: self.normalized(),
        }
    }

    /// `Some` for an IVF collection: the rows this query must scan.
    fn probe_plan(&self, query: &[f32], mask: Option<&BooleanArray>) -> Option<ProbePlan> {
        match self {
            Repr::Flat(_) => None,
            Repr::Ivf(i) => Some(i.probe_positions(query, i.nprobe(), mask)),
        }
    }
}

/// A named, immutable set of vectors (plus optional metadata labels) to search.
///
/// The corpus is a single contiguous row-major buffer, not a `Vec<Vec<f32>>`:
/// one streaming pass instead of a pointer chase per vector, AVX2+FMA distances
/// where available, and — the part that matters most at scale — a **bounded**
/// top-K per thread flushed with `select_nth_unstable`, so an `N`-row scan never
/// materializes or sorts an `N`-entry score vector. Label filtering is compiled
/// to a `BooleanArray` bitmask once per query rather than checked inside the scan.
#[derive(Debug, Clone)]
pub struct KnnCollection {
    index: Repr,
    /// Per-vector metadata label (e.g. genre), in **original input order**.
    /// For IVF this is also the mask domain that `probe_positions` expects.
    labels: Option<Vec<String>>,
    /// Content version for the device cache. Assigned at registration; a
    /// re-registered collection gets a new one.
    version: u64,
    /// Embedding-space/snapshot this collection was built for. `None` for an
    /// ad-hoc collection that did not declare one; such a collection cannot be
    /// checked against a query that names a space (that is refused, not
    /// ignored), but still answers unqualified queries unchanged.
    binding: Option<gtv_core::VectorSpaceBinding>,
}

impl KnnCollection {
    /// New L2 collection.
    pub fn new(
        ids: Vec<u64>,
        vectors: Vec<Vec<f32>>,
        labels: Option<Vec<String>>,
    ) -> Result<Self> {
        Self::with_metric(ids, vectors, labels, Metric::L2)
    }

    /// New collection with an explicit metric.
    ///
    /// For [`Metric::Cosine`] the rows are unit-normalized here (by `FlatIndex`),
    /// after which the distance is `1 - dot` and reuses the SIMD dot kernel.
    pub fn with_metric(
        ids: Vec<u64>,
        vectors: Vec<Vec<f32>>,
        labels: Option<Vec<String>>,
        metric: Metric,
    ) -> Result<Self> {
        if ids.len() != vectors.len() {
            return Err(DataFusionError::Execution(
                "knn collection: ids/vectors length mismatch".into(),
            ));
        }
        if let Some(labels) = &labels {
            if labels.len() != vectors.len() {
                return Err(DataFusionError::Execution(
                    "knn collection: labels/vectors length mismatch".into(),
                ));
            }
        }
        if vectors.is_empty() {
            return Err(DataFusionError::Execution(
                "knn collection: empty vector set".into(),
            ));
        }
        let index = FlatIndex::with_metric(ids, vectors, metric)
            .map_err(|e| DataFusionError::Execution(format!("knn collection: {e}")))?;
        Ok(Self {
            index: Repr::Flat(index),
            labels,
            version: 0,
            binding: None,
        })
    }

    /// New IVF-backed collection from a flat row-major `f32` corpus.
    ///
    /// `nlist` Voronoi cells are trained on the corpus; each query probes the
    /// `nprobe` nearest cells. The corpus is reordered so a probe is a handful
    /// of sequential runs (~64 MB at 1M × 512, `nlist=1024`, `nprobe=32`)
    /// instead of a 2 GB streaming scan. Distances stay exact `f32` — IVF
    /// prunes cells, it does not quantize vectors.
    pub fn with_ivf(
        ids: Vec<u64>,
        vectors: Vec<Vec<f32>>,
        labels: Option<Vec<String>>,
        metric: Metric,
        nlist: usize,
        nprobe: usize,
    ) -> Result<Self> {
        if ids.len() != vectors.len() {
            return Err(DataFusionError::Execution(
                "knn collection: ids/vectors length mismatch".into(),
            ));
        }
        if let Some(labels) = &labels {
            if labels.len() != vectors.len() {
                return Err(DataFusionError::Execution(
                    "knn collection: labels/vectors length mismatch".into(),
                ));
            }
        }
        if vectors.is_empty() {
            return Err(DataFusionError::Execution(
                "knn collection: empty vector set".into(),
            ));
        }
        let dim = vectors[0].len();
        let mut data = Vec::with_capacity(ids.len() * dim);
        for v in &vectors {
            if v.len() != dim {
                return Err(DataFusionError::Execution(
                    "knn collection: inconsistent vector dimensions".into(),
                ));
            }
            data.extend_from_slice(v);
        }
        let index = IvfIndex::with_metric(ids, data, dim, nlist, nprobe, metric)
            .map_err(|e| DataFusionError::Execution(format!("knn ivf collection: {e}")))?;
        Ok(Self {
            index: Repr::Ivf(index),
            labels,
            version: 0,
            binding: None,
        })
    }

    /// Build a collection from a (possibly mmap-backed) persisted index, so the
    /// `knn()` SQL surface can serve a corpus loaded cold from the index store
    /// without re-decoding it into owned `Vec<Vec<f32>>`.
    ///
    /// `Flat` and `Ivf` indexes become zero-copy views over the mapping; `Hnsw`
    /// is rejected because the `knn()` surface only has `Flat`/`Ivf`
    /// representations (use `ann()` for Hnsw).
    ///
    /// `labels`, if supplied, are in the **original input order** the corpus was
    /// registered with — for both reprs. That is the same domain
    /// [`Self::label_mask`] produces and, for an `Ivf`, the domain
    /// `IvfIndex::probe_positions` maps into cell order via `orig_pos` (which
    /// the `GIVFCMP` payload persists, so a cold-opened index keeps the
    /// mapping). Hence no cell-order reordering is needed here.
    pub fn from_any_index(
        any: AnyIndex,
        labels: Option<Vec<String>>,
    ) -> Result<Self> {
        let index = match any {
            AnyIndex::Flat(flat) => Repr::Flat(flat),
            AnyIndex::Ivf(ivf) => Repr::Ivf(ivf),
            AnyIndex::Hnsw(_) => {
                return Err(DataFusionError::Execution(
                    "knn collection: Hnsw index is not supported by the knn() surface; use ann()"
                        .into(),
                ))
            }
        };
        if let Some(labels) = &labels {
            if labels.len() != index.ids().len() {
                return Err(DataFusionError::Execution(format!(
                    "knn collection: labels/index length mismatch ({} labels vs {} vectors)",
                    labels.len(),
                    index.ids().len()
                )));
            }
        }
        Ok(Self {
            index,
            labels,
            version: 0,
            binding: None,
        })
    }

    /// Attach the embedding-space binding this collection answers for.
    ///
    /// A bound collection refuses a query that names a different space (see
    /// [`Self::check_binding`]); an unbound one answers unqualified queries as
    /// before. Builder form so a cold-open path can chain it onto the
    /// constructors without a second mutable pass.
    pub fn with_binding(mut self, binding: gtv_core::VectorSpaceBinding) -> Self {
        self.binding = Some(binding);
        self
    }

    /// The binding this collection carries, if any.
    pub fn binding(&self) -> Option<gtv_core::VectorSpaceBinding> {
        self.binding
    }

    /// Snapshot this collection as an [`AnyIndex`] so it can be persisted.
    ///
    /// This is the save-direction counterpart of [`Self::from_any_index`]: the
    /// returned index shares the same POD layout (`Col` views are `Arc`-backed
    /// for mmap corpora and clone-once for in-memory corpora), so a caller can
    /// hand it to `gtv_index_store::IndexStore::save` (or a bare-file writer)
    /// and the payload is re-encoded as a mmap-friendly `GTVVEC` / `GIVFCMP`
    /// container. A later `.knn-index` cold-open then serves it with no
    /// re-decode.
    pub fn to_any_index(&self) -> AnyIndex {
        match &self.index {
            Repr::Flat(i) => AnyIndex::Flat(i.clone()),
            Repr::Ivf(i) => AnyIndex::Ivf(i.clone()),
        }
    }

    /// Build parameters that match the backing index, for the persisted
    /// manifest. IVF carries its trained `nlist` / `nprobe`; flat carries none.
    pub fn build_options(&self) -> BuildOptions {
        match &self.index {
            Repr::Flat(_) => BuildOptions::Flat,
            Repr::Ivf(i) => BuildOptions::Ivf {
                nlist: i.nlist(),
                nprobe: i.nprobe(),
            },
        }
    }

    /// Whether this collection uses an IVF probe rather than a flat scan.
    pub fn is_ivf(&self) -> bool {
        matches!(self.index, Repr::Ivf(_))
    }

    /// Which device cache this collection belongs in.
    ///
    /// Flat and IVF corpora are stored in different device buffers on purpose:
    /// their row order differs, so a shared buffer would silently score the
    /// wrong rows.
    pub fn resident_kind(&self) -> ResidentKind {
        if self.is_ivf() {
            ResidentKind::IvfCorpus
        } else {
            ResidentKind::Corpus
        }
    }

    /// Vector dimension.
    pub fn dim(&self) -> usize {
        self.index.dim()
    }

    /// The metric every query against this collection uses.
    pub fn metric(&self) -> Metric {
        self.index.metric()
    }

    /// The collection's ids (exposed for the Quantum MCP surface). For an IVF
    /// collection this is the cell order, consistent with [`Self::vectors`].
    pub fn ids(&self) -> &[u64] {
        self.index.ids()
    }

    /// The corpus as one contiguous row-major `f32` buffer.
    ///
    /// This is the buffer the accelerator uploads (cell order for an IVF
    /// collection, parallel to [`Self::ids`]) — handed over by reference, with
    /// no per-vector copy and no `Vec<Vec<f32>>` rebuild.
    pub fn data(&self) -> &[f32] {
        self.index.data()
    }

    /// The collection's vectors (exposed for the Quantum MCP surface).
    ///
    /// Yields the canonical corpus — for a cosine collection that is the
    /// unit-normalized form the distances were computed from.
    pub fn vectors(&self) -> impl Iterator<Item = &[f32]> + '_ {
        let dim = self.index.dim();
        self.index.data().chunks(dim)
    }

    /// Content version used as the device-cache key.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Stamp the collection with a version (called once at registration).
    pub fn assign_version(&mut self, version: u64) {
        self.version = version;
    }

    /// Reject a caller-supplied metric that differs from the index metric.
    pub fn check_metric(&self, requested: Metric) -> std::result::Result<(), GtvError> {
        if requested != self.metric() {
            return Err(GtvError::MetricMismatch {
                index: self.metric(),
                query: requested,
            });
        }
        Ok(())
    }

    /// Refuse a query that names a different embedding space (or snapshot)
    /// than this collection answers for.
    ///
    /// Semantics, in order:
    /// * The collection is bound → a named space/snapshot must match exactly;
    ///   a mismatch is [`GtvError::VectorSpaceMismatch`] /
    ///   [`GtvError::SnapshotMismatch`], never a fallback.
    /// * The collection is **unbound** but the query names a space → refused,
    ///   because "cannot verify" is not "compatible". Unqualified queries
    ///   (`None`) still pass.
    ///
    /// `snapshot` is the corpus snapshot the caller expects, not the graph
    /// snapshot: a vector index annotates a different snapshot than the
    /// temporal graph, so the caller decides whether to pin it.
    pub fn check_binding(
        &self,
        snapshot: Option<u64>,
        vector_space_id: Option<u64>,
    ) -> std::result::Result<(), GtvError> {
        match &self.binding {
            Some(binding) => binding.check(snapshot, vector_space_id),
            None => {
                if vector_space_id.is_some() {
                    Err(GtvError::InvalidArgument(
                        "knn collection: query names a vector space but the collection is \
                         unbound; re-register it with a binding to use strong checks"
                            .into(),
                    ))
                } else {
                    Ok(())
                }
            }
        }
    }

    /// Compile the label filter into an aligned bitmask, if a label was asked
    /// for. `None` means "no filter" (and costs nothing in the scan).
    fn label_mask(&self, label: Option<&str>) -> Option<BooleanArray> {
        let want = label?;
        let labels = self.labels.as_ref()?;
        Some(BooleanArray::from(
            labels.iter().map(|l| l == want).collect::<Vec<bool>>(),
        ))
    }

    /// Exact K-NN under this collection's metric, on the host.
    ///
    /// Returns `(id, distance)` pairs nearest-first (L2 reports the Euclidean
    /// distance, others the metric distance).
    #[cfg(test)]
    fn search(&self, query: &[f32], k: usize, label: Option<&str>) -> Vec<(u64, f64)> {
        self.search_with("", None, query, k, label)
    }

    /// Exact K-NN with an optional accelerator.
    ///
    /// With a registry the device path is tried first (the corpus is already
    /// contiguous, so it is handed over without a copy) and the host kernel runs
    /// when the device declines or fails. Without one this is a pure CPU scan.
    /// For an IVF collection both surfaces consume the *same* probe plan, so
    /// the cell choice can never diverge between host and device.
    pub fn search_with(
        &self,
        name: &str,
        accel: Option<&AcceleratorRegistry>,
        query: &[f32],
        k: usize,
        label: Option<&str>,
    ) -> Vec<(u64, f64)> {
        let mask = self.label_mask(label);
        let plan = self.index.probe_plan(query, mask.as_ref());
        let kind = self.resident_kind();

        // P1c 预取预告 from the one place that actually knows the corpus: the
        // scan is imminent, so fault its pages in now instead of inside the
        // upload (or the host kernel). Skipped when the device already holds
        // the block — there is no host read to warm for.
        if let Some(reg) = accel {
            if !reg.is_resident(kind, name) {
                reg.prefetch_hint(kind, name, self.index.data());
            }
        }

        let raw: Vec<RawHit> = match (plan.as_ref(), accel) {
            // IVF: only the (small) position list moves per query; the whole
            // cell-ordered corpus stays resident on the device.
            (Some(plan), Some(reg)) => reg.ivf_probe(&IvfRequest {
                name,
                version: self.version,
                ids: self.index.ids(),
                data: self.index.data(),
                dim: self.index.dim(),
                metric: self.index.metric(),
                normalized: self.index.normalized(),
                query,
                k,
                positions: &plan.positions,
            }),
            (Some(plan), None) => gtv_index::exact_topk_positions(
                self.index.view(),
                &plan.positions,
                query,
                k,
            )
            .into_iter()
            .map(|(distance, id)| RawHit { distance, id })
            .collect(),
            (None, Some(reg)) => reg.exact_knn(&KnnRequest {
                name,
                version: self.version,
                ids: self.index.ids(),
                data: self.index.data(),
                dim: self.index.dim(),
                metric: self.index.metric(),
                normalized: self.index.normalized(),
                query,
                k,
                mask: mask.as_ref(),
            }),
            (None, None) => gtv_index::exact_topk(self.index.view(), query, k, mask.as_ref())
                .into_iter()
                .map(|(distance, id)| RawHit { distance, id })
                .collect(),
        };

        let metric = self.metric();
        raw.into_iter()
            .map(|h| (h.id, crate::vec_surface::report_distance(metric, h.distance)))
            .collect()
    }
}

/// The `knn` table function over a shared, mutable set of collections.
#[derive(Debug)]
pub struct KnnTableFunction {
    collections: Arc<RwLock<HashMap<String, KnnCollection>>>,
    /// Shared device cache. Present even on a GPU-less host, where it routes
    /// straight to the host kernel.
    accelerators: Arc<AcceleratorRegistry>,
}

impl KnnTableFunction {
    pub fn new(
        collections: Arc<RwLock<HashMap<String, KnnCollection>>>,
        accelerators: Arc<AcceleratorRegistry>,
    ) -> Self {
        Self {
            collections,
            accelerators,
        }
    }
}

/// P2 heat pump: pull the hottest non-resident collections onto the device while
/// no query is waiting on them.
///
/// Candidates come from the accelerator's shadow ledger — collections a query
/// was *refused*, ranked by decayed heat — and each one goes through
/// [`AcceleratorRegistry::prefetch_upload`], which admits it into VRAM that is
/// free right now. The pass is bounded by the registry's `ACCEL_PREFETCH_BLOCKS`
/// (`ACCEL_PREFETCH=OFF` ⇒ `0` ⇒ no-op), so a deployment that did not opt in
/// pays a single branch here.
///
/// The pass starts with a **reclaim sweep**: a collection dropped from the map
/// keeps its device copy until someone says otherwise, and those bytes can never
/// be a hit again. Releasing them is what gives the pump something to fill — the
/// scenario is `DROP TABLE t` followed by queries that keep asking for `u`. It is
/// also the only reason a slack-only pump is useful at all: a pump that evicts
/// would have to throw out a block a query is using (measured: +71% transferred
/// bytes, p99 × 1.41 on a 3-corpus rotation — see `bench_heat_pump_rotate`).
///
/// Nothing about the pass can fail a query: a candidate whose collection is gone
/// is dropped from the ledger, one whose representation differs is skipped, and
/// a refused admission is simply counted. Returns how many uploads completed.
pub fn pump_heat_prefetch(
    collections: &HashMap<String, KnnCollection>,
    accel: &AcceleratorRegistry,
) -> usize {
    let budget = accel.heat_pump_blocks();
    if budget == 0 || !accel.is_device_backed() {
        return 0;
    }
    reclaim_dropped(collections, accel);
    let mut uploaded = 0;
    // Over-fetch: a candidate may name a dropped collection or the other
    // representation, and the pass should still fill its budget from the
    // next-hottest name rather than stop at the first miss.
    for key in accel.prefetch_candidates(PREFETCH_HEAT_FLOOR, budget * 4) {
        if uploaded >= budget {
            break;
        }
        let Some(collection) = collections.get(&key.name) else {
            // The dataset no longer exists; never pump it again.
            accel.forget_candidate(key.kind, &key.name);
            continue;
        };
        if collection.resident_kind() != key.kind {
            continue;
        }
        if accel.prefetch_upload(
            key.kind,
            &key.name,
            collection.ids(),
            collection.data(),
            collection.dim(),
            collection.version(),
        ) {
            uploaded += 1;
        }
    }
    uploaded
}

/// Return the device copy of every corpus whose collection is gone.
///
/// `collections` is the engine's authoritative set of live knn datasets, and the
/// two corpus resident kinds are written only by this module, so "resident but
/// absent from the map" means the dataset was dropped. The block can never be a
/// hit again; holding it only shrinks the working set the pump may use.
fn reclaim_dropped(collections: &HashMap<String, KnnCollection>, accel: &AcceleratorRegistry) {
    for (kind, name) in accel.resident_blocks() {
        if !matches!(kind, ResidentKind::Corpus | ResidentKind::IvfCorpus) {
            continue;
        }
        if collections.contains_key(&name) {
            continue;
        }
        accel.release(kind, &name);
        accel.forget_candidate(kind, &name);
    }
}

impl TableFunctionImpl for KnnTableFunction {
    fn call_with_args(&self, args: TableFunctionArgs) -> Result<Arc<dyn TableProvider>> {
        let exprs = args.exprs();
        let name = expr_to_string(
            exprs
                .first()
                .ok_or_else(|| DataFusionError::Execution("knn(name, query, k [, label]): missing `name`".into()))?,
        )?;
        let query = expr_to_string(
            exprs
                .get(1)
                .ok_or_else(|| DataFusionError::Execution("knn(name, query, k [, label]): missing `query`".into()))?,
        )?;
        let k = expr_to_i64(
            exprs
                .get(2)
                .ok_or_else(|| DataFusionError::Execution("knn(name, query, k [, label]): missing `k`".into()))?,
        )? as usize;
        let label = exprs.get(3).map(expr_to_string).transpose()?;
        let requested_metric = exprs.get(4).map(expr_to_string).transpose()?;

        // `'*'` means "do not filter by label" so the metric (5th arg) can be
        // supplied for collections that carry no labels.
        let label = match label.as_deref() {
            Some("*") => None,
            _ => label,
        };

        let query = crate::vec_surface::parse_query("knn", &query)?;

        let collections = self.collections.read().map_err(|_| {
            DataFusionError::Execution("knn: collection registry poisoned".into())
        })?;
        let collection = collections.get(&name).ok_or_else(|| {
            DataFusionError::Execution(format!("knn: unknown collection `{name}`"))
        })?;

        crate::vec_surface::check_requested_metric(
            "knn",
            collection.metric(),
            requested_metric.as_deref(),
        )?;
        crate::vec_surface::check_dim("knn", &query, collection.dim())?;

        // Optional 6th argument: the vector-space id the caller expects. It is
        // a u64 (a 64-bit hash), so it is accepted as a decimal string rather
        // than an SQL integer, which would truncate values above i64::MAX.
        let requested_space = exprs
            .get(5)
            .map(expr_to_string)
            .transpose()?
            .map(|s| {
                s.trim().parse::<u64>().map_err(|_| {
                    DataFusionError::Execution(format!(
                        "knn: vector_space_id `{s}` is not a decimal u64"
                    ))
                })
            })
            .transpose()?;
        collection
            .check_binding(None, requested_space)
            .map_err(|e| DataFusionError::Execution(e.to_string()))?;

        let rows = collection.search_with(
            &name,
            Some(self.accelerators.as_ref()),
            &query,
            k,
            label.as_deref(),
        );
        // P2: the query has now either filled the device cache or left shadow
        // evidence; spend the idle window pulling in the hottest refused
        // collection so the *next* query finds it resident.
        pump_heat_prefetch(&collections, self.accelerators.as_ref());
        crate::vec_surface::hits_table(&rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ivf_collection_matches_flat_on_a_full_probe() {
        // nprobe = nlist makes the IVF probe scan every cell, so the answer
        // must be bit-identical to the flat reference — including distances.
        let ids: Vec<u64> = (0..12).collect();
        let vectors: Vec<Vec<f32>> = (0..12)
            .map(|i| vec![i as f32 * 0.5, (11 - i) as f32 * 0.25])
            .collect();
        let flat = KnnCollection::with_metric(ids.clone(), vectors.clone(), None, Metric::L2).unwrap();
        let ivf =
            KnnCollection::with_ivf(ids, vectors, None, Metric::L2, 4, 4).unwrap();
        assert!(ivf.is_ivf());
        assert_eq!(flat.search(&[2.0, 1.0], 5, None), ivf.search(&[2.0, 1.0], 5, None));
    }

    #[test]
    fn ivf_label_mask_is_in_original_input_order() {
        let ids: Vec<u64> = (0..8).collect();
        let vectors: Vec<Vec<f32>> = (0..8)
            .map(|i| vec![i as f32, 0.0])
            .collect();
        let labels: Vec<String> = (0..8)
            .map(|i| if i % 2 == 0 { "a" } else { "b" }.to_string())
            .collect();
        let flat =
            KnnCollection::with_metric(ids.clone(), vectors.clone(), Some(labels.clone()), Metric::L2)
                .unwrap();
        let ivf = KnnCollection::with_ivf(ids, vectors, Some(labels), Metric::L2, 4, 4).unwrap();
        // Every returned id must be even (label "a"), on both representations.
        for label in ["a", "b"] {
            let f: Vec<u64> = flat.search(&[3.0, 0.0], 8, Some(label))
                .into_iter().map(|(id, _)| id).collect();
            let v: Vec<u64> = ivf.search(&[3.0, 0.0], 8, Some(label))
                .into_iter().map(|(id, _)| id).collect();
            assert_eq!(f, v, "label {label} must select the same rows");
            assert!(v.iter().all(|id| (id % 2 == 0) == (label == "a")));
        }
    }

    #[test]
    fn ivf_cold_open_keeps_label_filtering() {
        // A cold-opened IVF must still honour labels. The mask is indexed by
        // `orig_pos` (cell position → original input position), and that column
        // is part of the persisted `GIVFCMP` payload, so labels stay in
        // original input order through save/load — no cell-order remap needed.
        let ids: Vec<u64> = (0..12).collect();
        let vectors: Vec<Vec<f32>> = (0..12)
            .map(|i| vec![i as f32, (11 - i) as f32 * 0.5])
            .collect();
        let labels: Vec<String> = (0..12)
            .map(|i| if i % 3 == 0 { "x" } else { "y" }.to_string())
            .collect();

        let built = KnnCollection::with_ivf(
            ids,
            vectors,
            Some(labels.clone()),
            Metric::L2,
            4,
            4,
        )
        .unwrap();
        let cold = KnnCollection::from_any_index(built.to_any_index(), Some(labels)).unwrap();
        assert!(cold.is_ivf());

        for label in [None, Some("x"), Some("y")] {
            let want = built.search(&[5.0, 3.0], 12, label);
            let got = cold.search(&[5.0, 3.0], 12, label);
            assert_eq!(got, want, "label {label:?} must select the same rows");
            if let Some(label) = label {
                let want_x = label == "x";
                for (id, _) in &got {
                    assert_eq!(*id % 3 == 0, want_x, "id {id} violates label {label}");
                }
            }
        }
    }

    #[test]
    fn from_any_index_rejects_label_length_mismatch() {
        let c = songs();
        let err = KnnCollection::from_any_index(c.to_any_index(), Some(vec!["pop".into()]))
            .unwrap_err();
        assert!(
            err.to_string().contains("labels/index length mismatch"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn bound_collection_refuses_a_different_space() {
        use gtv_core::{VectorSpace, VectorSpaceBinding};

        let space = VectorSpace::new("model-a", "v1", "feat-v1", Metric::L2, 2);
        let other = VectorSpace::new("model-b", "v1", "feat-v1", Metric::L2, 2);
        let bound = songs().with_binding(VectorSpaceBinding::new(Some(9), space.id()));

        // No constraint, matching space+snapshot, and a matching space with an
        // unspecified snapshot all pass.
        assert!(bound.check_binding(None, None).is_ok());
        assert!(bound.check_binding(None, Some(space.id())).is_ok());
        assert!(bound.check_binding(Some(9), Some(space.id())).is_ok());

        // A different model at the same dimension must be refused, never
        // silently answered.
        assert!(matches!(
            bound.check_binding(None, Some(other.id())),
            Err(GtvError::VectorSpaceMismatch { .. })
        ));
        assert!(matches!(
            bound.check_binding(Some(10), Some(space.id())),
            Err(GtvError::SnapshotMismatch { .. })
        ));
    }

    #[test]
    fn unbound_collection_refuses_a_named_space() {
        use gtv_core::VectorSpace;

        let c = songs();
        // (Metric comes in through `use super::*`.)
        // Unqualified query: unchanged behaviour.
        assert!(c.check_binding(None, None).is_ok());
        // A named space against an unbound collection cannot be verified, so it
        // is refused rather than treated as compatible.
        let space = VectorSpace::new("m", "v1", "f", Metric::L2, 2);
        assert!(matches!(
            c.check_binding(None, Some(space.id())),
            Err(GtvError::InvalidArgument(_))
        ));
    }

    fn songs() -> KnnCollection {
        // 10 songs, 2-dim embeddings, with genre labels (mirrors gen_data.py).
        let ids: Vec<u64> = (0..10).collect();
        let vectors = vec![
            vec![0.0, 0.0], // 0 pop
            vec![0.5, 0.5], // 1 pop
            vec![5.0, 5.0], // 2 rock
            vec![5.2, 5.1], // 3 rock
            vec![1.0, 1.0], // 4 jazz
            vec![1.1, 1.0], // 5 jazz
            vec![9.0, 9.0], // 6 classical
            vec![9.1, 9.0], // 7 classical
            vec![0.2, 0.1], // 8 pop
            vec![4.9, 5.0], // 9 rock
        ];
        let labels = vec![
            "pop", "pop", "rock", "rock", "jazz", "jazz", "classical", "classical", "pop",
            "rock",
        ]
        .into_iter()
        .map(str::to_string)
        .collect();
        KnnCollection::new(ids, vectors, Some(labels)).unwrap()
    }

    #[test]
    fn songs_knn_matches_oracle() {
        let c = songs();
        // TC-11 oracle: knn(q=[0.1,0.1], k=3, no filter) = [8, 0, 1]
        let got = c.search(&[0.1, 0.1], 3, None);
        assert_eq!(got.iter().map(|(id, _)| *id).collect::<Vec<_>>(), vec![8, 0, 1]);
    }

    #[test]
    fn songs_knn_pop_filter_matches_oracle() {
        let c = songs();
        // TC-11 oracle: mask=pop = [8, 0, 1]
        let got = c.search(&[0.1, 0.1], 3, Some("pop"));
        assert_eq!(got.iter().map(|(id, _)| *id).collect::<Vec<_>>(), vec![8, 0, 1]);
    }

    #[test]
    fn songs_knn_rock_filter_matches_oracle() {
        let c = songs();
        // TC-11 oracle: q=[5.0,5.0], mask=rock = [2, 9, 3]
        let got = c.search(&[5.0, 5.0], 3, Some("rock"));
        assert_eq!(got.iter().map(|(id, _)| *id).collect::<Vec<_>>(), vec![2, 9, 3]);
    }

    #[test]
    fn tss_knn_matches_oracle() {
        // TC-07 oracle: knn(q=noisy_uptrend, k=4) = [0, 2, 1, 3]
        let vectors = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], // 0 uptrend
            vec![5.0, 4.0, 2.0, 2.0, 4.0, 5.0], // 1 dip
            vec![3.0, 3.0, 3.0, 3.0, 3.0, 3.0], // 2 flat
            vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0], // 3 downtrend
        ];
        let c = KnnCollection::new((0..4).collect(), vectors, None).unwrap();
        let q = [1.1, 2.0, 3.1, 4.0, 5.1, 6.0];
        let got = c.search(&q, 4, None);
        assert_eq!(got.iter().map(|(id, _)| *id).collect::<Vec<_>>(), vec![0, 2, 1, 3]);
        let top1 = c.search(&q, 1, None);
        assert_eq!(top1.iter().map(|(id, _)| *id).collect::<Vec<_>>(), vec![0]);
    }

    #[test]
    fn metric_defaults_to_l2_and_is_reported() {
        let c = songs();
        assert_eq!(c.metric(), Metric::L2);
    }

    #[test]
    fn check_metric_rejects_mismatch() {
        let c = songs();
        assert!(c.check_metric(Metric::L2).is_ok());
        assert!(matches!(
            c.check_metric(Metric::Cosine),
            Err(GtvError::MetricMismatch { .. })
        ));
    }

    #[test]
    fn cosine_collection_orders_by_direction() {
        // Same directions as vectors 0/1/2 but wildly different magnitudes:
        // L2 would rank by magnitude, cosine must rank by direction.
        let vectors = vec![
            vec![100.0f32, 0.0],
            vec![0.9, 0.1],
            vec![0.0, 1.0],
        ];
        let c = KnnCollection::with_metric((0..3).collect(), vectors, None, Metric::Cosine)
            .unwrap();
        assert_eq!(c.metric(), Metric::Cosine);
        let got = c.search(&[1.0, 0.0], 3, None);
        assert_eq!(got.iter().map(|(id, _)| *id).collect::<Vec<_>>(), vec![0, 1, 2]);
    }

    #[test]
    fn knn_collection_from_any_index_flat_serves_mmap_corpus() {
        // The shape `IndexStore::load` returns for a GTVVEC payload: a `Flat`
        // AnyIndex wrapping a possibly mmap-backed `FlatIndex`. Wrapping it into
        // a `KnnCollection` must serve the corpus without re-decoding into
        // `Vec<Vec<f32>>`.
        let ids: Vec<u64> = (0..10).collect();
        let vectors = vec![
            vec![0.0, 0.0],
            vec![0.5, 0.5],
            vec![5.0, 5.0],
            vec![5.2, 5.1],
            vec![1.0, 1.0],
            vec![1.1, 1.0],
            vec![9.0, 9.0],
            vec![9.1, 9.0],
            vec![0.2, 0.1],
            vec![4.9, 5.0],
        ];
        let flat = FlatIndex::with_metric(ids.clone(), vectors.clone(), Metric::L2).unwrap();
        let c = KnnCollection::from_any_index(AnyIndex::Flat(flat), None).unwrap();
        assert!(!c.is_ivf());
        // Same oracle as `songs_knn_matches_oracle`: q=[0.1,0.1], k=3 = [8,0,1].
        let got = c.search(&[0.1, 0.1], 3, None);
        assert_eq!(got.iter().map(|(id, _)| *id).collect::<Vec<_>>(), vec![8, 0,1]);
    }

    #[test]
    fn knn_collection_from_any_index_ivf_serves_mmap_corpus() {
        let ids: Vec<u64> = (0..12).collect();
        let vectors: Vec<Vec<f32>> = (0..12)
            .map(|i| vec![i as f32 * 0.5, (11 - i) as f32 * 0.25])
            .collect();
        let dim = 2;
        let mut data = Vec::with_capacity(ids.len() * dim);
        for v in &vectors {
            data.extend_from_slice(v);
        }
        let ivf = IvfIndex::with_metric(ids.clone(), data, dim, 4, 4, Metric::L2).unwrap();
        let c = KnnCollection::from_any_index(AnyIndex::Ivf(ivf), None).unwrap();
        assert!(c.is_ivf());
        // A full probe (nprobe == nlist) must match the flat reference
        // bit-for-bit, including distances.
        let flat = KnnCollection::with_metric(ids, vectors, None, Metric::L2).unwrap();
        assert_eq!(flat.search(&[2.0, 1.0], 5, None), c.search(&[2.0, 1.0], 5, None));
    }

    // ---- P2: engine-side prefetch hints and the heat pump -------------------

    use gtv_accelerator::{
        AccelCache, AccelError, AccelOp, AccelBackend, AsofRequest, DeviceInfo, DeviceKind,
        HostBackend, PrefetchRequest, ResidentRegistry, VramBudget,
    };
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    /// A "device" that runs the host kernels and counts calls: enough to drive
    /// the pump's routing without hardware.
    #[derive(Debug)]
    struct FakeDevice {
        info: DeviceInfo,
        calls: AtomicU64,
        prefetches: AtomicU64,
    }

    impl FakeDevice {
        fn new() -> Self {
            Self {
                info: DeviceInfo {
                    name: "fake-gpu".to_string(),
                    kind: DeviceKind::Cuda,
                    total_bytes: 1 << 30,
                },
                calls: AtomicU64::new(0),
                prefetches: AtomicU64::new(0),
            }
        }
        fn calls(&self) -> u64 {
            self.calls.load(Ordering::Relaxed)
        }
        fn prefetches(&self) -> u64 {
            self.prefetches.load(Ordering::Relaxed)
        }
    }

    impl AccelBackend for FakeDevice {
        fn device(&self) -> &DeviceInfo {
            &self.info
        }
        fn supports(&self, _op: AccelOp) -> bool {
            true
        }
        fn exact_knn(&self, req: &KnnRequest<'_>) -> std::result::Result<Vec<RawHit>, AccelError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            HostBackend::new().exact_knn(req)
        }
        fn ivf_probe(&self, req: &IvfRequest<'_>) -> std::result::Result<Vec<RawHit>, AccelError> {
            HostBackend::new().ivf_probe(req)
        }
        fn asof_join(
            &self,
            req: &AsofRequest<'_>,
        ) -> std::result::Result<Vec<Option<u32>>, AccelError> {
            HostBackend::new().asof_join(req)
        }
        fn prefetch_corpus(
            &self,
            _req: &PrefetchRequest<'_>,
        ) -> std::result::Result<bool, AccelError> {
            self.prefetches.fetch_add(1, Ordering::Relaxed);
            Ok(true)
        }
    }

    fn fake_registry(capacity: u64, pump_blocks: usize) -> (Arc<FakeDevice>, AcceleratorRegistry) {
        let fake = Arc::new(FakeDevice::new());
        let ledger = ResidentRegistry::new(
            AccelCache::On,
            VramBudget::new(capacity, capacity, 0.0),
            0,
            Duration::from_secs(5),
        );
        let reg = AcceleratorRegistry::with_backend(fake.clone(), ledger)
            .with_heat_pump(pump_blocks);
        (fake, reg)
    }

    #[test]
    fn search_with_hints_pages_only_while_the_block_is_not_resident() {
        let (_fake, reg) = fake_registry(1 << 20, 0);
        let c = songs();
        c.search_with("songs", Some(&reg), &[0.1, 0.1], 3, None);
        assert_eq!(reg.telemetry().prefetch_hints, 1, "a cold corpus is warmed");
        assert!(reg.is_resident(ResidentKind::Corpus, "songs"));
        c.search_with("songs", Some(&reg), &[0.1, 0.1], 3, None);
        assert_eq!(
            reg.telemetry().prefetch_hints,
            1,
            "a resident corpus has no host read to warm"
        );
        // An IVF collection warms under its own resident kind.
        let ids: Vec<u64> = (0..12).collect();
        let vectors: Vec<Vec<f32>> = (0..12)
            .map(|i| vec![i as f32 * 0.5, (11 - i) as f32 * 0.25])
            .collect();
        let ivf = KnnCollection::with_ivf(ids, vectors, None, Metric::L2, 4, 4).unwrap();
        assert_eq!(ivf.resident_kind(), ResidentKind::IvfCorpus);
        ivf.search_with("ivf", Some(&reg), &[2.0, 1.0], 3, None);
        assert!(reg.is_resident(ResidentKind::IvfCorpus, "ivf"));
    }

    #[test]
    fn accel_prefetch_param_drives_the_pump_budget() {
        let (p, _w) = gtv_params::GtvParams::from_reader(
            "ACCEL_PREFETCH=ON\nACCEL_PREFETCH_BLOCKS=3\n".as_bytes(),
        )
        .unwrap();
        let reg = AcceleratorRegistry::from_resolved(&p.resolve(None).unwrap());
        assert_eq!(reg.heat_pump_blocks(), 3, "ACCEL_PREFETCH=ON arms the pump");

        let (p, _w) = gtv_params::GtvParams::from_reader("".as_bytes()).unwrap();
        let reg = AcceleratorRegistry::from_resolved(&p.resolve(None).unwrap());
        assert_eq!(reg.heat_pump_blocks(), 0, "the pump is off by default");
    }

    #[test]
    fn pump_pulls_in_the_hottest_refused_collection() {
        // Room for two 160-byte corpora; "hot" and "keep" take both slots.
        let (fake, reg) = fake_registry(400, 1);
        let mut map = HashMap::new();
        map.insert("hot".to_string(), songs());
        map.insert("keep".to_string(), songs());
        map.insert("cold".to_string(), songs());

        for name in ["hot", "keep"] {
            let c = map.get(name).unwrap();
            for _ in 0..5 {
                c.search_with(name, Some(&reg), &[0.1, 0.1], 3, None);
            }
        }
        let cold = map.get("cold").unwrap();
        for _ in 0..5 {
            // Refused every time: 80 B are free where 160 B are wanted, and both
            // resident corpora are protected by their heat.
            cold.search_with("cold", Some(&reg), &[0.1, 0.1], 3, None);
        }
        assert_eq!(reg.telemetry().cache_declined, 5);
        assert!(reg
            .prefetch_candidates(PREFETCH_HEAT_FLOOR, 4)
            .iter()
            .any(|k| k.name == "cold"));

        // A saturated cache gives the pump nothing: it declines rather than
        // evict a corpus a query is using.
        assert_eq!(pump_heat_prefetch(&map, &reg), 0);
        assert_eq!(fake.prefetches(), 0);
        assert!(!reg.is_resident(ResidentKind::Corpus, "cold"));

        // Drop "keep": its device copy can never be a hit again, so the pass
        // reclaims those bytes first and fills them in the same breath.
        map.remove("keep");
        assert_eq!(pump_heat_prefetch(&map, &reg), 1);
        assert_eq!(fake.prefetches(), 1, "the backend performed the upload");
        assert!(!reg.is_resident(ResidentKind::Corpus, "keep"));
        assert!(reg.is_resident(ResidentKind::Corpus, "cold"));

        // The next query is a device hit, not a supply-path miss.
        let cold = map.get("cold").unwrap();
        let before = fake.calls();
        cold.search_with("cold", Some(&reg), &[0.1, 0.1], 3, None);
        assert_eq!(fake.calls(), before + 1);
    }

    #[test]
    fn the_pump_is_a_no_op_when_disabled() {
        let (fake, reg) = fake_registry(200, 0);
        let mut map = HashMap::new();
        map.insert("hot".to_string(), songs());
        map.insert("cold".to_string(), songs());
        let hot = map.get("hot").unwrap();
        for _ in 0..5 {
            hot.search_with("hot", Some(&reg), &[0.1, 0.1], 3, None);
        }
        let cold = map.get("cold").unwrap();
        for _ in 0..5 {
            cold.search_with("cold", Some(&reg), &[0.1, 0.1], 3, None);
        }
        assert_eq!(pump_heat_prefetch(&map, &reg), 0);
        assert_eq!(fake.prefetches(), 0);
        assert!(!reg.is_resident(ResidentKind::Corpus, "cold"));
    }

    #[test]
    fn a_candidate_whose_collection_is_gone_is_forgotten() {
        let (fake, reg) = fake_registry(200, 2);
        let mut map = HashMap::new();
        map.insert("hot".to_string(), songs());
        map.insert("cold".to_string(), songs());
        let hot = map.get("hot").unwrap();
        for _ in 0..5 {
            hot.search_with("hot", Some(&reg), &[0.1, 0.1], 3, None);
        }
        let cold = map.get("cold").unwrap();
        for _ in 0..5 {
            cold.search_with("cold", Some(&reg), &[0.1, 0.1], 3, None);
        }
        // Drop the collection: the pump must not retry it forever.
        map.remove("cold");
        assert_eq!(pump_heat_prefetch(&map, &reg), 0);
        assert_eq!(fake.prefetches(), 0);
        assert!(reg.prefetch_candidates(PREFETCH_HEAT_FLOOR, 4).is_empty());
    }
}
