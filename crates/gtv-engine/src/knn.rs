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

use arrow::array::{ArrayRef, BooleanArray, Float64Array, RecordBatch, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::catalog::{TableFunctionArgs, TableFunctionImpl};
use datafusion::datasource::{MemTable, TableProvider};
use datafusion::error::{DataFusionError, Result};
use gtv_accelerator::{AcceleratorRegistry, IvfRequest, KnnRequest, RawHit};
use gtv_core::{GtvError, Metric};
use gtv_index::{CorpusView, FlatIndex, IvfIndex, ProbePlan};

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
        })
    }

    /// Whether this collection uses an IVF probe rather than a flat scan.
    pub fn is_ivf(&self) -> bool {
        matches!(self.index, Repr::Ivf(_))
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
            .map(|h| (h.id, report_distance(metric, h.distance)))
            .collect()
    }
}

/// Convert the canonical metric distance to the value surfaced to SQL/CLI.
/// L2 keeps its historical Euclidean (sqrt) report; cosine/IP report as-is.
#[inline]
fn report_distance(metric: Metric, raw: f32) -> f64 {
    match metric {
        Metric::L2 => (raw.max(0.0) as f64).sqrt(),
        _ => raw as f64,
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

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("distance", DataType::Float64, false),
        ]))
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

        let query: Vec<f32> = query
            .split(',')
            .map(|t| {
                t.trim().parse::<f32>().map_err(|_| {
                    DataFusionError::Execution(format!("knn: invalid query component `{t}`"))
                })
            })
            .collect::<Result<_>>()?;

        let collections = self.collections.read().map_err(|_| {
            DataFusionError::Execution("knn: collection registry poisoned".into())
        })?;
        let collection = collections.get(&name).ok_or_else(|| {
            DataFusionError::Execution(format!("knn: unknown collection `{name}`"))
        })?;

        if let Some(m) = requested_metric.as_deref() {
            let metric = Metric::parse(m).ok_or_else(|| {
                DataFusionError::Execution(format!(
                    "knn: unknown metric `{m}` (expected l2|cosine|dot)"
                ))
            })?;
            collection
                .check_metric(metric)
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;
        }

        if query.len() != collection.dim() {
            return Err(DataFusionError::Execution(format!(
                "knn: query dim {} != collection dim {}",
                query.len(),
                collection.dim()
            )));
        }

        let rows = collection.search_with(
            &name,
            Some(self.accelerators.as_ref()),
            &query,
            k,
            label.as_deref(),
        );
        let ids: Vec<u64> = rows.iter().map(|(id, _)| *id).collect();
        let dists: Vec<f64> = rows.iter().map(|(_, d)| *d).collect();

        let batch = RecordBatch::try_new(
            Self::schema(),
            vec![
                Arc::new(UInt64Array::from(ids)) as ArrayRef,
                Arc::new(Float64Array::from(dists)) as ArrayRef,
            ],
        )?;

        Ok(Arc::new(MemTable::try_new(
            Self::schema(),
            vec![vec![batch]],
        )?))
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
}
