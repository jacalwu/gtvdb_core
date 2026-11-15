//! A DataFusion [`SessionContext`] wrapper for the gtv engine.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::error::Result;
use datafusion::execution::disk_manager::{DiskManagerBuilder, DiskManagerMode};
use datafusion::execution::memory_pool::GreedyMemoryPool;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::logical_expr::LogicalPlan;
use datafusion::prelude::{SessionConfig, SessionContext};
use gtv_accelerator::AcceleratorRegistry;
use gtv_catalog::{ExecutionRecord, IndexRef, ModelRef, TableRef};
use gtv_core::TemporalCSR;
use gtv_index::AnyIndex;
use gtv_params::{CudaMode, GtvParams, QueryMode, ResolvedParams};

use crate::ext::{ExtensionInfo, ExtensionRegistry};
use crate::registry::{AsofResource, ResourceRegistry};
use crate::knn::{KnnCollection, KnnTableFunction};
use crate::embedding::{EmbeddingCollection, EmbeddingRegistry, EmbeddingSearchTableFunction};
use crate::motif::MotifSpec;

/// Options attached to a lineage-enabled execution.
#[derive(Debug, Clone, Default)]
pub struct ExecutionOptions {
    pub model_versions: Vec<ModelRef>,
    pub index_snapshots: Vec<IndexRef>,
    pub scenario_version: Option<String>,
    pub business_cutoff: Option<i64>,
    pub runtime_params: serde_json::Value,
}

/// Directory DataFusion uses for sort/join spill files (B3-6 §7.5).
///
/// `GTV_SPILL_DIR` overrides the default of `<system temp>/gtv-spill`.
/// Production deployments should point this at the catalog volume so spill
/// bytes are accounted for next to the data they came from.
pub fn spill_dir() -> PathBuf {
    std::env::var_os("GTV_SPILL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("gtv-spill"))
}

/// Parse a positive `usize` from an environment variable, or `None` when unset /
/// empty / invalid. Used for DataFusion execution knobs.
fn env_usize(name: &str) -> Option<usize> {
    let v = std::env::var(name).ok()?;
    let v = v.trim();
    if v.is_empty() {
        return None;
    }
    v.parse::<usize>().ok().filter(|&n| n > 0)
}

/// Apply `GTV_TARGET_PARTITIONS` / `GTV_BATCH_SIZE` (and fall back to the
/// parameter file's `WORKER_THREADS`, then to the CPU count) so query
/// parallelism is tunable and observable. Cap partitions against available
/// parallelism to reduce oversubscription with rayon-backed ANN / pattern
/// work on the same process.
fn apply_execution_env(config: SessionConfig, threads: Option<usize>) -> SessionConfig {
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let partitions = env_usize("GTV_TARGET_PARTITIONS")
        .or(threads)
        .unwrap_or(cpus)
        .max(1);
    let mut config = config.with_target_partitions(partitions);
    if let Some(batch) = env_usize("GTV_BATCH_SIZE") {
        config = config.with_batch_size(batch);
    }
    config
}

/// Run a DataFusion future from a sync call site without nesting runtimes
/// unsafely: prefer the current Tokio handle + `block_in_place`, else a
/// private current-thread runtime.
///
/// `block_in_place` is only legal on a multi-thread runtime, so a single-thread
/// runtime (the default for `#[tokio::test]`) gets an actionable error instead
/// of a panic.
fn block_on_df<F, T>(fut: F) -> Result<T>
where
    F: std::future::Future<Output = Result<T>>,
{
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::CurrentThread {
                return Err(datafusion::error::DataFusionError::Execution(
                    "gtv: blocking catalog I/O cannot run on a single-threaded Tokio runtime; \
                     use #[tokio::main(flavor = \"multi_thread\")] or #[tokio::test(flavor = \"multi_thread\")]"
                        .into(),
                ));
            }
            tokio::task::block_in_place(|| handle.block_on(fut))
        }
        Err(_) => {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| {
                    datafusion::error::DataFusionError::Execution(format!(
                        "tokio runtime for parquet register: {e}"
                    ))
                })?;
            rt.block_on(fut)
        }
    }
}

/// blake3 (hex) of the Arrow IPC encoding of the output batches.
pub(crate) fn batches_checksum(batches: &[RecordBatch]) -> String {
    let Some(first) = batches.first() else {
        return blake3::hash(b"").to_hex().to_string();
    };
    let mut buf = Vec::new();
    if let Ok(mut w) = arrow::ipc::writer::StreamWriter::try_new(&mut buf, &first.schema()) {
        for b in batches {
            let _ = w.write(b);
        }
        let _ = w.finish();
    }
    blake3::hash(&buf).to_hex().to_string()
}

/// A DataFusion `SessionContext` that gtv tables and UDFs are registered into.
#[derive(Clone)]
pub struct GtvContext {
    ctx: SessionContext,
    knn_collections: Arc<RwLock<HashMap<String, KnnCollection>>>,
    any_indexes: crate::ann::IndexRegistry,
    /// Governed embedding collections (B2-4) for `embedding_search`.
    embedding_collections: EmbeddingRegistry,
    /// Bitemporal system-time versions for `as_of` (B3-2).
    bitemporal: crate::bitemporal::BitemporalRegistry,
    /// Multimodal cost-based optimizer state (B3-5).
    cbo: crate::cbo::CboRegistry,
    /// Workload admission / isolation manager (B3-6).
    workload: Arc<crate::workload::WorkloadManager>,
    /// Registered tables mapped to the catalog snapshot they were loaded from.
    table_sources: Arc<RwLock<HashMap<String, TableRef>>>,
    /// Named resources (pit / asof / wash / tables) that kernel plans bind to.
    ///
    /// Domain-neutral on purpose: the HFT extension, the data-quality gate and
    /// the health diagnostics all read it, so it belongs to the core.
    resources: Arc<RwLock<ResourceRegistry>>,
    /// Installed extensions, in installation order.
    extensions: ExtensionRegistry,
    /// Snapshot of the above, shared with the `gtv_extensions()` table function.
    extension_info: Arc<Vec<ExtensionInfo>>,
    /// Opened GTVSeg segments served through mmap, keyed by registered name.
    segments: crate::segment::SegmentRegistry,
    /// L2 time boxes for the segments above, when they came from a manifest.
    segment_metas: crate::segment::SegmentMetaRegistry,
    /// Attached `FsCatalog`, the single authority behind `gtv_tables()` /
    /// `gtv_snapshots()` / `gtv_files()` (L1).
    catalog: crate::catalog_meta::CatalogSlot,
    /// Root of the `SegmentStore` whose published deltas back
    /// `gtv_invalidations()` (M1). A path, not a cached event list: every call
    /// re-reads the manifest, so the projection cannot go stale.
    invalidation_root: crate::invalidation::InvalidationSlot,
    /// M2: folded, de-duplicated snapshot behind
    /// `gtv_active_edge_count()` / `gtv_range_agg()`. Explicitly refreshed by
    /// `register_segment_store()` — never an implicit cache.
    aggregate: crate::aggregate::AggregateSlot,
    /// The parameter file that configured this process (resolved defaults when no
    /// file was found). Exposed so operators can inspect the effective config.
    params: GtvParams,
    /// Resolved, deployment-ready values (concrete byte budgets, chosen modes).
    resolved: ResolvedParams,
    /// Backend registry built from the parameter file. Not yet on the query path
    /// (C1/C2/C3 add operators without routing SQL to them), but CUDA_MODE /
    /// CUDA_VRAM already gate device init and its residency budget here. Behind an
    /// `Arc` so `GtvContext` can stay `Clone` (the registry itself is not `Clone`).
    accelerators: Arc<AcceleratorRegistry>,
}

impl GtvContext {
    /// Core-only context: temporal graph, vector search, lineage, DQ, quantum
    /// state metadata. No domain extension is loaded.
    ///
    /// Loads the parameter file (or the documented defaults) and builds the
    /// engine around it.
    pub fn new() -> Self {
        let (params, _warnings) = GtvParams::load();
        Self::build(&params)
    }

    /// Build a context from an explicit parameter set (tests, embedded usage).
    pub fn with_params(params: GtvParams) -> Self {
        Self::build(&params)
    }

    /// Build a context with `extensions` installed (banking, HFT, quantum
    /// compute, …). See [`crate::ext`] for the contract an extension implements.
    pub fn with_extensions(extensions: ExtensionRegistry) -> Result<Self> {
        let (params, _warnings) = GtvParams::load();
        let mut this = Self::build(&params);
        this.install_registry(extensions)?;
        Ok(this)
    }

    /// Install (or re-install) a set of extensions on a live context.
    ///
    /// DataFusion allows re-registering a name, so an extension may intentionally
    /// shadow a core function; installation order is the precedence order.
    pub fn install_registry(&mut self, extensions: ExtensionRegistry) -> Result<()> {
        extensions.install(self)?;
        let info = Arc::new(extensions.info());
        self.ctx.register_udtf(
            "gtv_extensions",
            Arc::new(crate::ext::ExtensionsTableFunction::new(info.clone())),
        );
        self.extension_info = info;
        self.extensions = extensions;
        Ok(())
    }

    /// Installed extensions, in installation order.
    pub fn extensions(&self) -> &[ExtensionInfo] {
        &self.extension_info
    }

    /// Recover a concrete extension, to configure it or read its state.
    pub fn extension<T: crate::ext::GtvExtension>(&self) -> Option<&T> {
        self.extensions.get::<T>()
    }

    /// Resolve params, downgrading `CUDA_MODE=ON` to `OFF` when no CUDA device
    /// is present (we cannot probe VRAM without a device), so the engine still
    /// boots on GPU-less machines. Other errors (e.g. `obj_area > target`) are
    /// surfaced to the caller.
    fn resolve_params(
        params: &GtvParams,
    ) -> std::result::Result<ResolvedParams, gtv_params::ParamError> {
        // Probe the device only when CUDA is actually requested, so a GPU-less
        // host never pays for driver initialisation. With the `cuda` feature the
        // total VRAM turns `CUDA_VRAM=DEVICE:0.8` into a real byte count; without
        // it the probe reports `None`, which resolves CUDA to off.
        let device_total = if params.cuda_mode == CudaMode::Off {
            None
        } else {
            gtv_accelerator::device_total_bytes(params.cuda_device)
        };
        match params.resolve(device_total) {
            Ok(r) => Ok(r),
            Err(gtv_params::ParamError::CudaNoDevice) => {
                let mut p = params.clone();
                p.cuda_mode = CudaMode::Off;
                p.resolve(None)
            }
            Err(e) => Err(e),
        }
    }

    /// Build the engine around a parameter set: resolve it, then configure the
    /// DataFusion runtime (memory pool, partitions, spill) and the backend
    /// registry from the resolved values.
    fn build(params: &GtvParams) -> Self {
        let resolved = Self::resolve_params(params).unwrap_or_else(|e| {
            eprintln!("gtv: parameter resolution warning: {e}; continuing with CUDA disabled");
            let mut safe = params.clone();
            safe.cuda_mode = CudaMode::Off;
            safe.resolve(None).expect("gtv: default resolution cannot fail")
        });

        // B3-5: EXPLAIN should surface the estimated statistics that the
        // `StatsTable` provider exposes (num_rows / min / max / null_count).
        let mut config = SessionConfig::new();
        config.options_mut().explain.show_statistics = true;
        // Explicit target_partitions / batch_size (GTV_TARGET_PARTITIONS,
        // GTV_BATCH_SIZE) override the parameter file's WORKER_THREADS, which is
        // itself a fallback for the logical CPU count.
        let config = apply_execution_env(config, resolved.worker_threads);

        // B3-6 spill-to-disk: large sorts / joins land under the configured
        // spill directory. `GTV_SPILL_DIR` still wins for container overrides.
        let spill = std::env::var_os("GTV_SPILL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| resolved.spill_dir.clone());
        let _ = std::fs::create_dir_all(&spill);
        let disk = DiskManagerBuilder::default()
            .with_mode(DiskManagerMode::Directories(vec![spill.clone()]));
        let mut runtime_builder = RuntimeEnvBuilder::new().with_disk_manager_builder(disk);
        // MEMORY_TARGET bounds DataFusion's memory pool. Only enforce it when the
        // operator set it explicitly: the auto default (half of system RAM) stays
        // unenforced so small CI boxes and ad-hoc runs behave as before.
        if resolved.memory_target_explicit {
            runtime_builder = runtime_builder.with_memory_pool(Arc::new(GreedyMemoryPool::new(
                resolved.memory_target_bytes as usize,
            )));
        }
        let runtime = runtime_builder
            .build()
            .expect("gtv: build DataFusion runtime env");
        let ctx = SessionContext::new_with_config_rt(config, Arc::new(runtime));

        // Device acceleration: budget carved from CUDA_VRAM, cache policy from
        // ACCEL_CACHE / ACCEL_SYNC_* / ACCEL_HEAT_HALF_LIFE. Falls back to the
        // host silently (the reason is kept in telemetry) so boot never depends
        // on a GPU.
        let accelerators = Arc::new(AcceleratorRegistry::from_resolved(&resolved));

        // -- core: temporal / columnar primitives --------------------------
        for udwf in crate::udf::window_udfs() {
            ctx.register_udwf(udwf);
        }
        ctx.register_udf(crate::datetime::trunc_udf());

        // -- core: graph + vector ------------------------------------------
        let knn_collections = Arc::new(RwLock::new(HashMap::new()));
        ctx.register_udtf(
            "knn",
            Arc::new(KnnTableFunction::new(
                knn_collections.clone(),
                accelerators.clone(),
            )),
        );
        ctx.register_udtf(
            "vector_search",
            Arc::new(KnnTableFunction::new(
                knn_collections.clone(),
                accelerators.clone(),
            )),
        );
        let any_indexes: crate::ann::IndexRegistry = Arc::new(RwLock::new(HashMap::new()));
        ctx.register_udtf("ann", Arc::new(crate::ann::AnnTableFunction::new(any_indexes.clone())));
        ctx.register_udtf(
            "ann_search",
            Arc::new(crate::ann::AnnTableFunction::new(any_indexes.clone())),
        );
        ctx.register_udtf(
            "ann_explain",
            Arc::new(crate::ann::AnnExplainTableFunction::new(any_indexes.clone())),
        );

        // -- core: governed embeddings + bitemporal ------------------------
        let embedding_collections: EmbeddingRegistry = Arc::new(RwLock::new(HashMap::new()));
        ctx.register_udtf(
            "embedding_search",
            Arc::new(EmbeddingSearchTableFunction::new(embedding_collections.clone())),
        );
        let bitemporal: crate::bitemporal::BitemporalRegistry =
            Arc::new(RwLock::new(gtv_storage::BitemporalStore::new()));
        let as_of = Arc::new(crate::bitemporal::BitemporalAsOfTableFunction::new(
            bitemporal.clone(),
        ));
        ctx.register_udtf("as_of", as_of.clone());
        ctx.register_udtf("bitemporal_as_of", as_of);
        ctx.register_udtf(
            "bitemporal_overlaps",
            Arc::new(crate::bitemporal::BitemporalOverlapsTableFunction::new(
                bitemporal.clone(),
            )),
        );

        // -- core: planning / admission ------------------------------------
        let cbo: crate::cbo::CboRegistry = Arc::new(RwLock::new(crate::cbo::CboState::default()));
        ctx.register_udtf(
            "cbo_explain",
            Arc::new(crate::cbo::CboExplainTableFunction::new(
                any_indexes.clone(),
                cbo.clone(),
            )),
        );
        let workload = Arc::new(crate::workload::WorkloadManager::default());
        ctx.register_udtf(
            "workload_status",
            Arc::new(crate::workload::WorkloadStatusTableFunction::new(
                workload.clone(),
            )),
        );

        // -- core: table I/O -----------------------------------------------
        ctx.register_udtf("read_csv", Arc::new(crate::csv::ReadCsvTableFunction::new()));
        ctx.register_udtf(
            "read_parquet",
            Arc::new(crate::csv::ReadParquetTableFunction::new()),
        );

        // -- core: quantum state metadata (storage, not simulation) --------
        // The compute side (`quantum_knn`, `quantum_walk`, …) is an extension.
        ctx.register_udtf(
            "quantum_state_summary",
            Arc::new(crate::quantum_state::QuantumStateSummaryTableFunction),
        );
        ctx.register_udtf(
            "quantum_state_frontier",
            Arc::new(crate::quantum_state::QuantumStateFrontierTableFunction),
        );
        ctx.register_udtf(
            "quantum_state_reconstruct",
            Arc::new(crate::quantum_state::QuantumStateReconstructTableFunction),
        );

        // -- core: data quality + observability over registered resources --
        let resources = Arc::new(RwLock::new(ResourceRegistry::default()));
        ctx.register_udtf(
            "dq_report",
            Arc::new(crate::monitor::DqReportTableFunction::new(resources.clone())),
        );
        ctx.register_udtf(
            "dq_check",
            Arc::new(crate::monitor::DqCheckTableFunction::new(resources.clone())),
        );
        ctx.register_udtf(
            "dq_gate",
            Arc::new(crate::dq::DqGateTableFunction::new(resources.clone())),
        );
        ctx.register_udtf(
            "health_check",
            Arc::new(crate::monitor::HealthCheckTableFunction::new(resources.clone())),
        );
        ctx.register_udtf(
            "strategy_stats",
            Arc::new(crate::monitor::StrategyStatsTableFunction::new(resources.clone())),
        );
        ctx.register_udtf(
            "gtv_extensions",
            Arc::new(crate::ext::ExtensionsTableFunction::new(Arc::new(Vec::new()))),
        );

        // -- core: immutable GTVSeg segments (mmap-backed temporal CSR) --
        let segments = crate::segment::empty_registry();
        let segment_metas = crate::segment::empty_meta_registry();
        ctx.register_udtf(
            "gtv_segments",
            Arc::new(crate::segment::SegmentsTableFunction::with_metas(
                segments.clone(),
                segment_metas.clone(),
            )),
        );
        ctx.register_udtf(
            "gtv_segment_edges",
            Arc::new(crate::segment::SegmentEdgesTableFunction::new(
                segments.clone(),
            )),
        );

        // -- core: L1 metadata surface (one authority: FsCatalog) -----------
        let catalog = crate::catalog_meta::empty_slot();
        ctx.register_udtf(
            "gtv_tables",
            Arc::new(crate::catalog_meta::TablesTableFunction::new(catalog.clone())),
        );
        ctx.register_udtf(
            "gtv_snapshots",
            Arc::new(crate::catalog_meta::SnapshotsTableFunction::new(
                catalog.clone(),
            )),
        );
        ctx.register_udtf(
            "gtv_files",
            Arc::new(crate::catalog_meta::FilesTableFunction::new(catalog.clone())),
        );

        // -- core: M1 invalidation projection (manifest commit log) ---------
        let invalidation_root = crate::invalidation::empty_invalidation_slot();
        ctx.register_udtf(
            "gtv_invalidations",
            Arc::new(crate::invalidation::InvalidationsTableFunction::new(
                invalidation_root.clone(),
            )),
        );
        // M3: named snapshot tags are another projection of the same manifest.
        ctx.register_udtf(
            "gtv_tags",
            Arc::new(crate::invalidation::TagsTableFunction::new(
                invalidation_root.clone(),
            )),
        );

        // -- core: M2 L2 aggregates over a folded snapshot -----------------
        let aggregate = crate::aggregate::empty_aggregate_slot();
        ctx.register_udtf(
            "gtv_active_edge_count",
            Arc::new(crate::aggregate::ActiveEdgeCountTableFunction::new(
                aggregate.clone(),
            )),
        );
        ctx.register_udtf(
            "gtv_range_agg",
            Arc::new(crate::aggregate::RangeAggTableFunction::new(
                aggregate.clone(),
            )),
        );

        Self {
            ctx,
            knn_collections,
            any_indexes,
            embedding_collections,
            bitemporal,
            cbo,
            workload,
            resources,
            extensions: ExtensionRegistry::new(),
            extension_info: Arc::new(Vec::new()),
            segments,
            segment_metas,
            catalog,
            invalidation_root,
            aggregate,
            table_sources: Arc::new(RwLock::new(HashMap::new())),
            params: params.clone(),
            resolved,
            accelerators,
        }
    }

    /// Shared KNN collections, so an extension can register vectors into the
    /// same registry the core `knn` / `vector_search` functions read.
    pub fn knn_collections(&self) -> Arc<RwLock<HashMap<String, KnnCollection>>> {
        self.knn_collections.clone()
    }

    /// Shared resource registry (pit / asof / wash / tables).
    pub fn resources(&self) -> Arc<RwLock<ResourceRegistry>> {
        self.resources.clone()
    }

    pub fn session(&self) -> &SessionContext {
        &self.ctx
    }

    /// The parameter file (or resolved defaults) this context was built from.
    pub fn params(&self) -> &GtvParams {
        &self.params
    }

    /// Concrete, deployment-ready values resolved from the parameter file.
    pub fn resolved(&self) -> &ResolvedParams {
        &self.resolved
    }

    /// The device acceleration registry (KNN / as-of offload + telemetry).
    pub fn accelerators(&self) -> &Arc<AcceleratorRegistry> {
        &self.accelerators
    }

    /// The configured default query dialect (`DEFAULT_MODE`).
    pub fn default_mode(&self) -> QueryMode {
        self.resolved.default_mode
    }

    /// Register an in-memory table from a schema and its record batches.
    /// Replaces any existing table of the same name (idempotent re-register).
    pub fn register_batches(
        &self,
        name: &str,
        schema: SchemaRef,
        batches: Vec<RecordBatch>,
    ) -> Result<()> {
        let _ = self.ctx.deregister_table(name);
        // Wrap with the B3-5 statistics provider so EXPLAIN / DataFusion's own
        // optimizers see catalog stats (num_rows / min / max / null_count).
        let provider = crate::cbo::StatsTable::new(
            name,
            schema.clone(),
            vec![batches.clone()],
            self.cbo.clone(),
        );
        self.ctx.register_table(name, Arc::new(provider))?;
        // Seed session statistics from the actual batches so EXPLAIN shows
        // min/max/null bounds even before a catalog refresh (B3-5).
        if let Ok(mut cbo) = self.cbo.write() {
            cbo.tables.insert(
                name.to_string(),
                gtv_catalog::TableStats::from_batches(name, &batches),
            );
        }
        if let Ok(mut reg) = self.resources.write() {
            reg.tables.insert(name.to_string(), Arc::new(batches));
        }
        Ok(())
    }

    /// Remove a registered table (no-op when absent).
    ///
    /// Clears both the DataFusion catalog entry and the compiled-kernel table
    /// registry (`resources`), so a later `drop table` / `DROP TABLE` fully
    /// releases the underlying record batches.
    pub fn deregister_table(&self, name: &str) {
        let _ = self.ctx.deregister_table(name);
        if let Ok(mut reg) = self.resources.write() {
            reg.tables.remove(name);
        }
    }

    /// Whether `name` resolves to a registered table (kernel registry or the
    /// DataFusion catalog). Querying with `LIMIT 0` resolves the name during
    /// planning, so `Err` means the table does not exist.
    pub async fn has_table(&self, name: &str) -> bool {
        if self
            .resources
            .read()
            .map(|r| r.tables.contains_key(name))
            .unwrap_or(false)
        {
            return true;
        }
        self.ctx.sql(&format!("SELECT * FROM {name} LIMIT 0")).await.is_ok()
    }

    /// Row count of a registered table without materializing a query, or
    /// `None` when the table is not in the kernel registry (e.g. created via
    /// native DataFusion CTAS). Cheap — used by background flush tasks to
    /// decide whether a hot table changed since the last checkpoint.
    pub fn table_rows(&self, name: &str) -> Option<usize> {
        let reg = self.resources.read().ok()?;
        reg.tables
            .get(name)
            .map(|batches| batches.iter().map(|b| b.num_rows()).sum())
    }

    /// Cloned record batches for a registered session table, or `None` when the
    /// table is not in the kernel registry. Used by the DQ gate.
    pub fn table_batches(&self, name: &str) -> Option<Vec<RecordBatch>> {
        let reg = self.resources.read().ok()?;
        reg.tables.get(name).map(|b| b.as_ref().clone())
    }

    /// Evaluate data-quality `rules` over session table `table`, resolving
    /// referential parents from other session tables. Returns the gate decision
    /// plus the per-rule evidence. Errors only when the target table is missing.
    pub fn evaluate_dq(
        &self,
        table: &str,
        rules: &[gtv_catalog::DqRule],
    ) -> Result<(gtv_catalog::GateDecision, Vec<crate::dq::RuleOutcome>)> {
        let batches = self.table_batches(table).ok_or_else(|| {
            datafusion::error::DataFusionError::Execution(format!("unknown table `{table}`"))
        })?;
        let reg = self.resources.clone();
        let lookup = move |parent: &str, col: &str| crate::dq::parent_lookup(&reg, parent, col);
        Ok(crate::dq::evaluate(&batches, rules, &lookup))
    }

    /// Load a CSV file from disk and register it as `name` (method 1:
    /// traditional disk load).
    pub fn register_csv(&self, path: &str, name: &str) -> Result<()> {
        let batches = gtv_storage::read_csv(path)
            .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;
        let first = batches
            .first()
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Execution(format!("empty csv `{path}`"))
            })?;
        self.register_batches(name, first.schema(), batches)
    }

    /// Register a Parquet file (or directory of Parquet files) as `name` via
    /// DataFusion's [`ListingTable`] / `ParquetExec` — **without** loading the
    /// whole file into a [`MemTable`]. Prefer this for large CRM / tick tables.
    ///
    /// Unlike [`Self::register_batches`], the table is **not** seeded into the
    /// HFT kernel registry (no full materialization). Use `CREATE TABLE … AS
    /// SELECT * FROM name` (or `register_batches`) when an in-memory copy is
    /// required for HFT / DQ kernels.
    ///
    /// Bridge: DataFusion's register API is async; callers are sync (CLI /
    /// catalog). We `block_on` on the current Tokio runtime when present,
    /// otherwise spin a short-lived current-thread runtime.
    pub fn register_parquet(&self, path: &str, name: &str) -> Result<()> {
        let _ = self.ctx.deregister_table(name);
        if let Ok(mut reg) = self.resources.write() {
            reg.tables.remove(name);
        }
        let fut = self.ctx.register_parquet(
            name,
            path,
            datafusion::prelude::ParquetReadOptions::default(),
        );
        block_on_df(fut)?;
        Ok(())
    }

    /// Open an immutable GTVSeg file and register it under `name`.
    ///
    /// This is O(1) in the number of edges: the file is mmap'd and its columns
    /// reinterpreted in place — no decode, no copy, no `O(E log E)` rebuild.
    /// Unlike [`Self::register_parquet`], nothing is routed through DataFusion's
    /// file-scan machinery; the CSR is available immediately to Rust-side graph
    /// operators via [`Self::segment`], and observable in SQL through
    /// `gtv_segments()` / `gtv_segment_edges(name)`.
    pub fn register_segment(&self, path: &str, name: &str) -> Result<()> {
        let seg = gtv_storage::open_segment(path).map_err(|e| {
            datafusion::error::DataFusionError::Execution(format!(
                "cannot open GTVSeg `{path}`: {e}"
            ))
        })?;
        self.segments
            .write()
            .map_err(|_| {
                datafusion::error::DataFusionError::Execution(
                    "segment registry lock poisoned".into(),
                )
            })?
            .insert(name.to_string(), Arc::new(seg));
        // A re-registered name must not keep the previous file's time box.
        if let Ok(mut metas) = self.segment_metas.write() {
            metas.remove(name);
        }
        Ok(())
    }

    /// Attach an L2 time box to an already-registered segment.
    ///
    /// `register_segment` cannot derive the union box without an `O(E)` scan, so
    /// callers that know it (manifest readers, offline tooling) supply it here.
    pub fn register_segment_meta(&self, name: &str, meta: gtv_storage::SegmentMeta) -> Result<()> {
        let mut metas = self.segment_metas.write().map_err(|_| {
            datafusion::error::DataFusionError::Execution("segment metadata lock poisoned".into())
        })?;
        metas.insert(name.to_string(), meta);
        Ok(())
    }

    /// Shared L2 metadata registry (for extensions / tooling).
    pub fn segment_metas(&self) -> crate::segment::SegmentMetaRegistry {
        self.segment_metas.clone()
    }

    /// Attach (or refresh) the folded aggregate projection behind
    /// `gtv_active_edge_count()` / `gtv_range_agg()`.
    ///
    /// The view covers exactly one system snapshot. Refreshing it is explicit,
    /// like `sync_catalog()` for L1: a stale metadata answer is worse than a
    /// slow one, and SQL never silently re-reads a store it does not own.
    pub fn attach_aggregate_view(&self, view: gtv_storage::AggregateView) -> Result<()> {
        let mut slot = self.aggregate.write().map_err(|_| {
            datafusion::error::DataFusionError::Execution("aggregate slot lock poisoned".into())
        })?;
        *slot = Some(view);
        Ok(())
    }

    /// The currently attached aggregate view, if any.
    pub fn aggregate_view(&self) -> Option<gtv_storage::AggregateView> {
        self.aggregate.read().ok()?.clone()
    }

    /// Register every segment published by a P1 [`SegmentStore`] under
    /// `<prefix>_<segment id>`, making the ingested data visible to
    /// `gtv_segments()` / `gtv_segment_edges(name)` without a Parquet round
    /// trip or a CSV import.
    ///
    /// Segment registration itself stays O(1) per segment: each file is
    /// mmap'd and its header validated, with no edge column copied. Since M2
    /// the same call is also the explicit synchronisation point for the
    /// aggregate projection, which folds the head row set once (O(E)); use
    /// `register_segment()` directly when only the segment listing is needed.
    pub fn register_segment_store(
        &self,
        store: &gtv_storage::store::SegmentStore,
        prefix: &str,
    ) -> Result<()> {
        // The same store also backs the invalidation projection, so a normal
        // P1 registration makes `gtv_invalidations()` work with no extra call.
        self.attach_delta_log(store.root())?;

        // M2: one explicit synchronisation point for both new read-only
        // projections. Observation versions make `as_of('cpi', ...)` survive a
        // restart; the aggregate view is the de-duplicated L2 snapshot.
        if let Ok(mut bt) = self.bitemporal.write() {
            store.load_observations_into(&mut bt).map_err(|e| {
                datafusion::error::DataFusionError::Execution(format!(
                    "cannot replay observation versions: {e}"
                ))
            })?;
        }
        let view = store.aggregate_view(store.system_ts()).map_err(|e| {
            datafusion::error::DataFusionError::Execution(format!(
                "cannot build aggregate view: {e}"
            ))
        })?;
        self.attach_aggregate_view(view)?;

        // Retired segments still exist on disk for `AS KNOWN AT` queries, but
        // they are not part of the head snapshot and must not appear twice.
        for entry in store.segments().iter().filter(|e| e.superseded_at.is_none()) {
            let path = store.segment_path(entry).to_string_lossy().into_owned();
            let name = format!("{prefix}_{}", entry.id);
            self.register_segment(&path, &name)?;
            // L2 box comes from the manifest, so it is free here — no O(E) scan.
            if let Some(meta) = entry.meta() {
                if let Ok(mut metas) = self.segment_metas.write() {
                    metas.insert(name, meta);
                }
            }
        }
        Ok(())
    }

    /// Look up an opened segment by its registration name.
    pub fn segment(&self, name: &str) -> Option<Arc<gtv_storage::segment::Segment>> {
        self.segments.read().ok()?.get(name).cloned()
    }

    /// Shared registry of opened segments (for extensions / tooling).
    pub fn segments(&self) -> crate::segment::SegmentRegistry {
        self.segments.clone()
    }

    // -- L1: catalog metadata surface --------------------------------------

    /// Attach a filesystem catalog. After this, `gtv_tables()` /
    /// `gtv_snapshots()` / `gtv_files()` answer from it, and
    /// [`Self::sync_catalog`] can publish its tables into SQL.
    ///
    /// Re-attaching replaces the previous catalog (the slot is shared with the
    /// table functions, so SQL sees the change immediately).
    pub fn attach_catalog(&self, root: impl Into<PathBuf>) -> Result<gtv_catalog::FsCatalog> {
        let cat = gtv_catalog::FsCatalog::open(root.into())
            .map_err(|e| datafusion::error::DataFusionError::Execution(format!("attach_catalog: {e}")))?;
        let mut slot = self.catalog.write().map_err(|_| {
            datafusion::error::DataFusionError::Execution("catalog slot lock poisoned".into())
        })?;
        *slot = Some(cat.clone());
        Ok(cat)
    }

    /// The attached catalog, if any.
    pub fn catalog(&self) -> Option<gtv_catalog::FsCatalog> {
        self.catalog.read().ok()?.clone()
    }

    // -- M1: invalidation projection ---------------------------------------

    /// Point `gtv_invalidations()` at a `SegmentStore` root. The slot holds a
    /// path only: every call re-reads the manifest and delta files, so the view
    /// tracks the single atomic publish point without an event bus.
    ///
    /// [`Self::register_segment_store`] calls this automatically.
    pub fn attach_delta_log(&self, root: impl Into<PathBuf>) -> Result<()> {
        let mut slot = self.invalidation_root.write().map_err(|_| {
            datafusion::error::DataFusionError::Execution(
                "invalidation slot lock poisoned".into(),
            )
        })?;
        *slot = Some(root.into());
        Ok(())
    }

    /// The attached delta-log root, if any.
    pub fn delta_log_root(&self) -> Option<PathBuf> {
        self.invalidation_root.read().ok()?.clone()
    }

    /// Publish every committed catalog table into the SQL session, reading
    /// **exactly** the file list of its latest snapshot.
    ///
    /// This is the "single source of truth" step: a table is visible in SQL
    /// because the catalog says so, not because someone called `register_csv`.
    /// Tables are registered as DataFusion views over their snapshot files, so
    /// the scan is still a lazy Parquet scan (row-group pruning included) —
    /// nothing is materialised into memory.
    ///
    /// Returns the number of tables registered. Tables without a committed
    /// snapshot are skipped (they have no data to expose).
    pub fn sync_catalog(&self) -> Result<usize> {
        use gtv_catalog::FileFormat;

        let cat = self.catalog().ok_or_else(|| {
            datafusion::error::DataFusionError::Execution(
                "sync_catalog: no catalog attached (call attach_catalog first)".into(),
            )
        })?;
        let map = |e: gtv_catalog::CatalogError| {
            datafusion::error::DataFusionError::Execution(e.to_string())
        };

        let mut registered = 0usize;
        for meta in cat.list_tables().map_err(map)? {
            let Some(snap) = cat.latest(meta.table_id).map_err(map)? else {
                continue;
            };
            let files = cat.files(meta.table_id, snap).map_err(map)?;
            if files.is_empty() {
                continue;
            }
            let all_parquet = files.iter().all(|f| f.format == FileFormat::Parquet);
            let all_csv = files.iter().all(|f| f.format == FileFormat::Csv);
            let paths: Vec<String> = files.iter().map(|f| f.path.clone()).collect();

            let df = if all_parquet {
                block_on_df(self.ctx.read_parquet(
                    paths,
                    datafusion::prelude::ParquetReadOptions::default(),
                ))?
            } else if all_csv {
                block_on_df(self.ctx.read_csv(
                    paths,
                    datafusion::prelude::CsvReadOptions::default(),
                ))?
            } else {
                return Err(datafusion::error::DataFusionError::Execution(format!(
                    "sync_catalog: table `{}` mixes CSV and Parquet files",
                    meta.name
                )));
            };

            let _ = self.ctx.deregister_table(&meta.name);
            self.ctx.register_table(&meta.name, df.into_view())?;
            // Seed relational stats from the catalog so EXPLAIN is not blind.
            if let Ok(stats) = cat.table_stats(meta.table_id) {
                let _ = self.set_table_stats(&meta.name, stats);
            }
            registered += 1;
        }
        Ok(registered)
    }

    /// Run a SQL query and collect all result batches.
    pub async fn sql(&self, query: &str) -> Result<Vec<RecordBatch>> {
        let df = self.ctx.sql(query).await?;
        df.collect().await
    }

    /// Register the `neighbors(src, valid_at)` graph table function.
    ///
    /// M6: both UDTFs now share a single `Arc<TemporalCSR>`. The previous code
    /// cloned the entire graph twice at registration time.
    pub fn register_neighbors(&self, csr: &TemporalCSR) {
        let shared = Arc::new(csr.clone());
        self.ctx.register_udtf(
            "neighbors",
            Arc::new(crate::graph::NeighborsTableFunction::from_arc(shared.clone())),
        );
        self.ctx.register_udtf(
            "khop",
            Arc::new(crate::graph::KhopTableFunction::from_arc(shared)),
        );
    }

    /// Register batched graph tables for a set of seeds (M6 perf line).
    ///
    /// Unlike `neighbors(src,t)` / `khop(src,k,t)`, which accept literals only,
    /// these are ordinary tables and can be joined with any SQL relation:
    ///
    /// ```sql
    /// SELECT c.case_id, b.dst FROM cases c
    /// JOIN gtv_batch_neighbors b ON b.src = c.account_id;
    /// SELECT * FROM gtv_batch_khop WHERE hop <= 2;
    /// ```
    ///
    /// One graph image, one planning cycle per SQL statement, one batch. Re-
    /// registering replaces the previous tables, so a new alert list is cheap.
    pub fn register_batch_graph(
        &self,
        csr: &TemporalCSR,
        seeds: &[u64],
        valid_at: i64,
        k: usize,
    ) -> Result<()> {
        self.register_batch_graph_shared(&Arc::new(csr.clone()), seeds, valid_at, k)
    }

    /// Same as [`Self::register_batch_graph`] but reuses an existing graph
    /// image. Seed sets change far more often than the graph does, so a caller
    /// that keeps an `Arc<TemporalCSR>` avoids copying the whole CSR on every
    /// alert batch — `bench_m6_gate` measures this as a fixed ~1.4 ms cost.
    pub fn register_batch_graph_shared(
        &self,
        csr: &Arc<TemporalCSR>,
        seeds: &[u64],
        valid_at: i64,
        k: usize,
    ) -> Result<()> {
        let neighbors =
            crate::batch_graph::BatchNeighborsTable::new(csr, seeds.iter().copied(), valid_at)?;
        let khop =
            crate::batch_graph::BatchKhopTable::new(csr, seeds.iter().copied(), k, valid_at)?;
        let _ = self.ctx.deregister_table("gtv_batch_neighbors");
        let _ = self.ctx.deregister_table("gtv_batch_khop");
        self.ctx.register_table("gtv_batch_neighbors", Arc::new(neighbors))?;
        self.ctx.register_table("gtv_batch_khop", Arc::new(khop))?;
        Ok(())
    }

    /// Register the `gtv-pattern` SQL surface (ROADMAP §3.3 / P1-4).
    ///
    /// Both UDTFs share one `Arc<TemporalCSR>`:
    ///
    /// ```sql
    /// SELECT * FROM motif('ring3', 100, 500);              -- all starts
    /// SELECT * FROM motif_from('ring3', 7, 100, 500, 30);  -- one seed + window
    /// ```
    ///
    /// They are literal-only, exactly like `neighbors()` / `khop()`, and serve
    /// as the pointwise baseline for the batch tables below.
    pub fn register_motif(&self, csr: &TemporalCSR) {
        let shared = Arc::new(csr.clone());
        self.ctx.register_udtf(
            "motif",
            Arc::new(crate::motif::MotifTableFunction::from_arc(shared.clone())),
        );
        self.ctx.register_udtf(
            "motif_from",
            Arc::new(crate::motif::MotifFromTableFunction::from_arc(shared)),
        );
    }

    /// Register joinable motif tables for a batch of seeds. One kernel run
    /// materializes both projections, one planning cycle per SQL statement:
    ///
    /// ```sql
    /// SELECT c.case_id, m.n0, m.n1, m.n2
    ///   FROM cases c JOIN gtv_motif_matches m ON m.n0 = c.account_id;
    /// SELECT * FROM gtv_motif_edges WHERE edge_type = 1;
    /// ```
    ///
    /// `MotifSpec` carries shape / valid_at / limit / rolling window / edge-type
    /// sequence. Re-registering replaces the previous tables, so a new alert
    /// list does not re-copy the graph when using the `_shared` variant.
    pub fn register_motif_batch(
        &self,
        csr: &TemporalCSR,
        spec: &MotifSpec,
        seeds: &[u64],
    ) -> Result<()> {
        self.register_motif_batch_shared(&Arc::new(csr.clone()), spec, seeds)
    }

    /// See [`Self::register_batch_graph_shared`]: seed sets change far more
    /// often than the graph image does.
    pub fn register_motif_batch_shared(
        &self,
        csr: &Arc<TemporalCSR>,
        spec: &MotifSpec,
        seeds: &[u64],
    ) -> Result<()> {
        let pair = crate::motif::build_motif_tables(csr, spec.clone(), seeds)?;
        let _ = self.ctx.deregister_table("gtv_motif_matches");
        let _ = self.ctx.deregister_table("gtv_motif_edges");
        self.ctx.register_table("gtv_motif_matches", Arc::new(pair.matches))?;
        self.ctx.register_table("gtv_motif_edges", Arc::new(pair.edges))?;
        Ok(())
    }

    /// Register the `asof_join(t0, t1, ...)` table function against a
    /// right-side time series (times + values, equally long).
    pub fn register_asof_join(&self, right_times: Vec<i64>, right_values: Vec<f64>) {
        self.ctx.register_udtf(
            "asof_join",
            Arc::new(crate::asof::AsofJoinTableFunction::new(
                right_times,
                right_values,
                self.accelerators.clone(),
            )),
        );
    }

    /// Register a multi-column + tolerance `asof_join`/`aj` against a right-side
    /// series (times + price + spread).
    pub fn register_asof_join_multi(
        &self,
        right_times: Vec<i64>,
        right_price: Vec<f64>,
        right_spread: Vec<f64>,
        tolerance_ns: i64,
    ) {
        let tf = Arc::new(crate::asof::AsofJoinTableFunction::new_multi(
            right_times.clone(),
            right_price.clone(),
            right_spread.clone(),
            tolerance_ns,
            self.accelerators.clone(),
        ));
        self.ctx.register_udtf("asof_join", tf.clone());
        self.ctx.register_udtf("aj", tf);
        let res = Arc::new(AsofResource {
            times: Arc::new(right_times),
            price: Arc::new(right_price),
            spread: Arc::new(right_spread),
            tolerance: tolerance_ns,
        });
        if let Ok(mut reg) = self.resources.write() {
            reg.asof.insert("aj".to_string(), res);
        }
    }

    /// Register a named vector collection for `knn(name, query, k [, label])`.
    pub fn register_knn(
        &self,
        name: &str,
        ids: Vec<u64>,
        vectors: Vec<Vec<f32>>,
        labels: Option<Vec<String>>,
    ) -> Result<()> {
        self.register_knn_metric(name, ids, vectors, labels, gtv_core::Metric::L2)
    }

    /// Register a named vector collection with an explicit distance metric.
    pub fn register_knn_metric(
        &self,
        name: &str,
        ids: Vec<u64>,
        vectors: Vec<Vec<f32>>,
        labels: Option<Vec<String>>,
        metric: gtv_core::Metric,
    ) -> Result<()> {
        let mut collection = KnnCollection::with_metric(ids, vectors, labels, metric)
            .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;
        // Fresh content version: registering under an existing name replaces the
        // collection, and the device copy must follow.
        collection.assign_version(self.accelerators.next_version());
        self.knn_collections
            .write()
            .map_err(|_| {
                datafusion::error::DataFusionError::Execution(
                    "knn collection registry poisoned".into(),
                )
            })?
            .insert(name.to_string(), collection);
        Ok(())
    }

    /// Register an **IVF-backed** `knn` collection.
    ///
    /// `nlist` cells are trained on the corpus; each query probes `nprobe` of
    /// them. This is the SQL entry point for the device IVF operator: the
    /// cell-ordered corpus is uploaded once, and only the probe's position list
    /// crosses PCIe per query. Distances remain exact `f32`.
    pub fn register_knn_ivf(
        &self,
        name: &str,
        ids: Vec<u64>,
        vectors: Vec<Vec<f32>>,
        labels: Option<Vec<String>>,
        options: crate::knn::IvfOptions,
    ) -> Result<()> {
        let mut collection = KnnCollection::with_ivf(
            ids,
            vectors,
            labels,
            options.metric,
            options.nlist,
            options.nprobe,
        )
        .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;
        collection.assign_version(self.accelerators.next_version());
        self.knn_collections
            .write()
            .map_err(|_| {
                datafusion::error::DataFusionError::Execution(
                    "knn collection registry poisoned".into(),
                )
            })?
            .insert(name.to_string(), collection);
        Ok(())
    }

    /// Register a `knn` collection from a (possibly mmap-backed) persisted index.
    ///
    /// This is the missing link that lets the `knn()` surface serve a corpus
    /// loaded cold from the index store (`gtv_index_store::IndexStore::load` →
    /// `AnyIndex`) instead of re-decoding owned `Vec<Vec<f32>>`. For a `Flat`
    /// index `labels` may be supplied (insertion order); for an `Ivf` index
    /// labels are rejected until the orig→cell mapping is plumbed (see
    /// [`KnnCollection::from_any_index`]); `Hnsw` is rejected (use `ann()`).
    pub fn register_knn_from_index(
        &self,
        name: &str,
        index: AnyIndex,
        labels: Option<Vec<String>>,
    ) -> Result<()> {
        let mut collection = KnnCollection::from_any_index(index, labels)
            .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;
        // Fresh content version: registering under an existing name replaces
        // the collection, and the device copy must follow.
        collection.assign_version(self.accelerators.next_version());
        self.knn_collections
            .write()
            .map_err(|_| {
                datafusion::error::DataFusionError::Execution(
                    "knn collection registry poisoned".into(),
                )
            })?
            .insert(name.to_string(), collection);
        Ok(())
    }

    /// Register a persisted (loaded) index for `ann(name, query, k [, metric])`.
    pub fn register_any_index(&self, name: &str, index: AnyIndex) {
        if let Ok(mut map) = self.any_indexes.write() {
            map.insert(name.to_string(), index);
        }
    }

    /// Snapshot a registered `knn` collection as an [`AnyIndex`] plus its build
    /// parameters, so a caller (e.g. the REPL `.knn-save` command) can persist
    /// it with `gtv_index_store::IndexStore::save` and later cold-open it via
    /// [`Self::register_knn_from_index`]. Returns `None` if the collection is
    /// unknown. This is the save-direction mirror of the wrapper above.
    pub fn knn_to_any_index(
        &self,
        name: &str,
    ) -> Option<(AnyIndex, gtv_index::BuildOptions)> {
        let guard = self.knn_collections.read().ok()?;
        let collection = guard.get(name)?;
        Some((collection.to_any_index(), collection.build_options()))
    }

    /// Register a governed embedding batch (catalog standard schema) for
    /// `embedding_search(name, query, k [, tenant [, as_of]])`. The batch is
    /// validated (dimension / single model-metric / provenance) first.
    pub fn register_embedding(&self, name: &str, batch: &RecordBatch) -> Result<()> {
        let collection = EmbeddingCollection::from_batch(batch)?;
        self.register_embedding_collection(name, collection);
        Ok(())
    }

    /// Register an already-built [`EmbeddingCollection`].
    pub fn register_embedding_collection(&self, name: &str, collection: EmbeddingCollection) {
        if let Ok(mut map) = self.embedding_collections.write() {
            map.insert(name.to_string(), collection);
        }
    }

    /// Append a system-time version of `table` for the bitemporal `as_of`
    /// surface. Corrections append a new `system_from`; older versions stay
    /// queryable.
    pub fn register_bitemporal_version(
        &self,
        table: &str,
        system_from: i64,
        batches: Vec<RecordBatch>,
    ) -> Result<()> {
        self.bitemporal
            .write()
            .map_err(|_| {
                datafusion::error::DataFusionError::Execution(
                    "bitemporal registry poisoned".into(),
                )
            })?
            .record(table, system_from, batches)
            .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))
    }

    /// Shared bitemporal store (for the CLI / catalog wiring).
    pub fn bitemporal_store(&self) -> crate::bitemporal::BitemporalRegistry {
        self.bitemporal.clone()
    }

    /// Shared multimodal cost-based optimizer state (B3-5).
    pub fn cbo_state(&self) -> crate::cbo::CboRegistry {
        self.cbo.clone()
    }

    /// Register relational statistics for `name` (used by `cbo_explain`).
    pub fn set_table_stats(&self, name: &str, stats: gtv_catalog::TableStats) -> Result<()> {
        self.cbo
            .write()
            .map_err(|_| {
                datafusion::error::DataFusionError::Execution("cbo registry poisoned".into())
            })?
            .tables
            .insert(name.to_string(), stats);
        Ok(())
    }

    /// Refresh relational statistics from the catalog (never stale: reads the
    /// current committed version).
    pub fn refresh_table_stats(
        &self,
        name: &str,
        catalog: &gtv_catalog::FsCatalog,
        table: gtv_catalog::TableId,
    ) -> Result<()> {
        let stats = catalog
            .table_stats(table)
            .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;
        self.set_table_stats(name, stats)
    }

    /// Refresh statistics for `name` from the catalog by table name (B3-5).
    ///
    /// Convenience wrapper that resolves the [`gtv_catalog::TableId`] first,
    /// used by the CLI after `publish`.
    pub fn refresh_table_stats_by_name(
        &self,
        name: &str,
        catalog: &gtv_catalog::FsCatalog,
    ) -> Result<()> {
        let stats = catalog
            .table_stats_by_name(name)
            .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;
        self.set_table_stats(name, stats)
    }

    /// Register graph shape statistics for `name`.
    pub fn set_graph_stats(&self, name: &str, stats: crate::cbo::GraphStats) -> Result<()> {
        self.cbo
            .write()
            .map_err(|_| {
                datafusion::error::DataFusionError::Execution("cbo registry poisoned".into())
            })?
            .graphs
            .insert(name.to_string(), stats);
        Ok(())
    }

    /// Replace the cost model (weights / enable flag).
    pub fn set_cost_model(&self, model: crate::cbo::CostModel) -> Result<()> {
        self.cbo
            .write()
            .map_err(|_| {
                datafusion::error::DataFusionError::Execution("cbo registry poisoned".into())
            })?
            .model = model;
        Ok(())
    }

    /// Shared workload manager (B3-6).
    pub fn workload(&self) -> Arc<crate::workload::WorkloadManager> {
        self.workload.clone()
    }

    /// Execute `query` under a workload class, with admission control.
    ///
    /// This is the B3-6 integration point: the class's resource group decides
    /// whether the query is admitted, queued or rejected; a preemption signal
    /// arriving before collection aborts the query. Operators that observe the
    /// returned [`gtv_core::CancelToken`] (graph traversal, index build) can be
    /// cancelled cooperatively mid-flight via
    /// [`crate::workload::WorkloadManager::preempt`].
    pub async fn sql_as(
        &self,
        class: crate::workload::WorkloadClass,
        query: &str,
        timeout: std::time::Duration,
    ) -> std::result::Result<Vec<RecordBatch>, crate::workload::WorkloadError> {
        use crate::workload::{Admission, WorkloadError};
        let (id, token) = match self.workload.wait_admit(class, timeout)? {
            Admission::Admit { id, token } => (id, token),
            Admission::Queue { .. } => {
                return Err(WorkloadError::Rejected("unexpected queued admission".into()))
            }
            Admission::Reject { reason } => return Err(WorkloadError::Rejected(reason)),
        };
        if token.load(std::sync::atomic::Ordering::Relaxed) {
            self.workload.release(id);
            return Err(WorkloadError::Preempted);
        }
        let out = self
            .sql(query)
            .await
            .map_err(|e| WorkloadError::Rejected(e.to_string()));
        self.workload.release(id);
        out
    }

    /// Bytes currently used by DataFusion spill files (B3-6 §7.5).
    pub fn spill_bytes(&self) -> u64 {
        self.ctx.runtime_env().disk_manager.used_disk_space()
    }

    /// Prometheus text: process query metrics, per-class workload telemetry and
    /// spill usage.
    pub fn prometheus(&self) -> String {
        let mut s = crate::monitor::prometheus_text();
        s.push_str(&self.workload.prometheus());
        s.push_str("# TYPE gtv_spill_bytes gauge\n");
        s.push_str(&format!("gtv_spill_bytes {}\n", self.spill_bytes()));
        let progress = self.ctx.runtime_env().disk_manager.spilling_progress();
        s.push_str("# TYPE gtv_spill_active_files gauge\n");
        s.push_str(&format!(
            "gtv_spill_active_files {}\n",
            progress.active_files_count
        ));
        s
    }

    /// Record which catalog snapshot a registered table was loaded from, so a
    /// lineage record can pin it.
    pub fn set_table_source(&self, name: &str, source: TableRef) {
        if let Ok(mut map) = self.table_sources.write() {
            map.insert(name.to_string(), source);
        }
    }

    /// The catalog snapshot a registered table came from, if known.
    pub fn table_source(&self, name: &str) -> Option<TableRef> {
        self.table_sources.read().ok()?.get(name).cloned()
    }

    /// Execute `sql`, returning the output batches and a complete
    /// [`ExecutionRecord`] (source snapshots, model/index versions, output
    /// checksum). The caller persists the record (e.g. via the catalog).
    pub async fn execute_with_lineage(
        &self,
        sql: &str,
        opts: ExecutionOptions,
    ) -> Result<(Vec<RecordBatch>, ExecutionRecord)> {
        let mut record = ExecutionRecord::begin(sql, env!("CARGO_PKG_VERSION"));

        let df = self.ctx.sql(sql).await?;
        let plan = df.logical_plan().clone();
        let mut names: Vec<String> = Vec::new();
        let _ = plan.apply(&mut |node: &LogicalPlan| {
            if let LogicalPlan::TableScan(scan) = node {
                names.push(scan.table_name.table().to_string());
            }
            Ok(TreeNodeRecursion::Continue)
        })?;

        let batches = df.collect().await?;
        let rows: u64 = batches.iter().map(|b| b.num_rows() as u64).sum();
        let checksum = batches_checksum(&batches);

        names.sort();
        names.dedup();
        if let Ok(sources) = self.table_sources.read() {
            record.source_tables = names
                .iter()
                .filter_map(|n| sources.get(n).cloned())
                .collect();
        }
        record.model_versions = opts.model_versions;
        record.index_snapshots = opts.index_snapshots;
        record.udf_versions = crate::lineage::extract_udfs(&plan);
        record.scenario_version = opts.scenario_version;
        record.business_cutoff = opts.business_cutoff;
        record.runtime_params = opts.runtime_params;
        record.finish(checksum, rows);
        Ok((batches, record))
    }
}

impl Default for GtvContext {
    fn default() -> Self {
        Self::new()
    }
}
