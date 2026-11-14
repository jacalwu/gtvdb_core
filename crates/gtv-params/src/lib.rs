//! Oracle-style parameters file for gtVdB.
//!
//! A flat `KEY=VALUE` file (all-uppercase keys, `#`/`--` comments, blank lines
//! ignored) used to control gtVdB deployment across machine profiles. Binary
//! size suffixes (`K`/`M`/`G`/`T`, optional trailing `B`) and fractional
//! references (`SYSTEM:0.5`, `TARGET:0.5`, `DEVICE:0.8`) are supported.
//!
//! Parsing is three-stage:
//!
//! 1. [`GtvParams`] — raw file values with their defaults applied.
//! 2. [`GtvParams::resolve`] — concrete bytes / selected mode, given an optional
//!    CUDA device VRAM total (injected by the caller, so this crate stays free
//!    of any GPU dependency).
//! 3. [`ResolvedParams`] — what the engine actually consumes.
//!
//! Environment overrides (`GTV_PARAM_<KEY>`, highest priority) and file
//! discovery (`GTV_PARAMS_FILE` → `./gtvdb.params` → exe dir → system dir) are
//! handled by [`GtvParams::load`].

use std::collections::HashSet;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

mod error;
pub use error::{ParamError, Warning, Warnings};

// ---------------------------------------------------------------------------
// Specs — how a value is expressed before it is resolved to a concrete number.
// ---------------------------------------------------------------------------

/// A memory amount expressed either as a concrete byte count or as a fraction
/// of the system's physical memory.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemorySpec {
    /// Concrete byte count (e.g. `16000M`).
    Bytes(u64),
    /// Fraction of total system physical memory (e.g. `SYSTEM:0.5`).
    SystemFraction(f64),
}

impl Default for MemorySpec {
    fn default() -> Self {
        // Half of system memory.
        Self::SystemFraction(0.5)
    }
}

impl MemorySpec {
    /// `true` when an explicit byte count was given (vs. an auto fraction).
    pub fn is_explicit(&self) -> bool {
        matches!(self, Self::Bytes(_))
    }

    /// Resolve against the system's total physical memory.
    pub fn resolve_against(&self, total: u64) -> u64 {
        match self {
            Self::Bytes(b) => *b,
            Self::SystemFraction(f) => (total as f64 * f) as u64,
        }
    }
}

/// The resident-in-memory object area, expressed as a concrete byte count or a
/// fraction of [`MemorySpec`] (the memory target).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemAreaSpec {
    /// Concrete byte count.
    Bytes(u64),
    /// Fraction of `MEMORY_TARGET` (e.g. `TARGET:0.5`).
    TargetFraction(f64),
}

impl Default for MemAreaSpec {
    fn default() -> Self {
        // Half of the memory target.
        Self::TargetFraction(0.5)
    }
}

impl MemAreaSpec {
    /// `true` when an explicit byte count was given.
    pub fn is_explicit(&self) -> bool {
        matches!(self, Self::Bytes(_))
    }
}

/// How CUDA is engaged.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum CudaMode {
    /// Never open a device.
    Off,
    /// Require a device; boot fails if none is present.
    On,
    /// Open a device only when one is detectable (the default).
    #[default]
    Auto,
}

/// How the accelerator's device-resident cache is engaged.
///
/// The counterpart of PG-Strom's `pg_strom.enable_gpucache`, with an extra
/// calibration setting so a deployment can measure what a GPU cache *would*
/// cost and save before hardware is bought.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AccelCache {
    /// Never keep a device copy; every operation runs on the host.
    Off,
    /// Run on the host, but keep the residency ledger and hit/miss accounting
    /// as if the cache were live. Reports the hit ratio and the upload bytes
    /// that a device would have saved, without touching a GPU.
    Shadow,
    /// Keep the device copy and execute on the device when it fits.
    #[default]
    On,
}

impl AccelCache {
    /// Upper-case spelling used by `ACCEL_CACHE` / `summary_lines`.
    pub fn as_str(self) -> &'static str {
        match self {
            AccelCache::Off => "OFF",
            AccelCache::Shadow => "SHADOW",
            AccelCache::On => "ON",
        }
    }

    /// `true` when a device copy may actually be allocated.
    pub fn uses_device(self) -> bool {
        matches!(self, AccelCache::On)
    }

    /// `true` when the residency ledger is maintained (everything but `Off`).
    pub fn tracks_residency(self) -> bool {
        !matches!(self, AccelCache::Off)
    }
}

/// The default query dialect.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum QueryMode {
    /// Standard SQL (DataFusion).
    #[default]
    Sql,
    /// The native q dialect (requires the quantum/q extension at runtime).
    Q,
}

/// Log verbosity; maps onto the `RUST_LOG` filter vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Level {
    Error,
    Warn,
    #[default]
    Info,
    Debug,
    Trace,
}

impl Level {
    /// The `RUST_LOG` token for this level.
    pub fn as_rust_log(&self) -> &'static str {
        match self {
            Level::Error => "error",
            Level::Warn => "warn",
            Level::Info => "info",
            Level::Debug => "debug",
            Level::Trace => "trace",
        }
    }
}

/// CUDA VRAM budget, expressed as a concrete byte count or a fraction of the
/// device's total VRAM.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CudaVramSpec {
    /// Concrete byte count.
    Bytes(u64),
    /// Fraction of the device's total VRAM (e.g. `DEVICE:0.8`).
    DeviceFraction(f64),
}

impl Default for CudaVramSpec {
    fn default() -> Self {
        // 80% of the device VRAM.
        Self::DeviceFraction(0.8)
    }
}

// ---------------------------------------------------------------------------
// The raw, pre-resolved parameter set.
// ---------------------------------------------------------------------------

/// Parsed (and defaulted) gtVdB parameters, before resolution.
///
/// Every field has a sensible default so a missing or empty file still yields a
/// usable [`GtvParams`]. Keys actually present in a file or env override are
/// tracked in [`GtvParams::is_explicit`] for consumers that care about
/// explicit-vs-default (e.g. whether to mount a DataFusion memory pool).
#[derive(Debug, Clone)]
pub struct GtvParams {
    /// Total memory gtVdB may allocate. Default: half of system RAM.
    pub memory_target: MemorySpec,
    /// Resident-in-memory object area. Default: half of `memory_target`.
    pub memory_obj_area: MemAreaSpec,
    /// Whether to use CUDA. Default: `Auto`.
    pub cuda_mode: CudaMode,
    /// CUDA VRAM budget. Default: 80% of device VRAM.
    pub cuda_vram: CudaVramSpec,
    /// CUDA device ordinal (0-based). Default: 0.
    pub cuda_device: usize,
    /// How the device-resident cache is engaged. Default: `On`.
    pub accel_cache: AccelCache,
    /// Bytes of pending host→device delta that trigger a batched upload.
    /// `None` => 25% of the VRAM budget (PG-Strom's `gpu_sync_threshold`
    /// default is likewise a quarter of its redo buffer).
    pub accel_sync_threshold: Option<u64>,
    /// Seconds since the last upload after which pending delta is applied even
    /// if the byte threshold was not reached. Counterpart of PG-Strom's
    /// `gpu_sync_interval`. Default: 5.
    pub accel_sync_interval_secs: u64,
    /// Half-life (seconds) of a resident block's decayed access heat. Heat is
    /// the admission/eviction score; `0` disables the policy and restores pure
    /// LRU. Default: 30.
    pub accel_heat_half_life_secs: u64,
    /// Default query dialect. Default: `Sql`.
    pub default_mode: QueryMode,
    /// Log level. Default: `Info`.
    pub log_level: Level,
    /// Spill directory. `None` => resolved from `GTV_SPILL_DIR` or `<tmp>/gtv-spill`.
    pub spill_dir: Option<PathBuf>,
    /// Worker thread count override. `None` => env / CPU count.
    pub worker_threads: Option<usize>,
    /// Data directory for deployment. Default: `./data`.
    pub data_dir: PathBuf,
    /// Catalog directory for deployment. Default: `./catalog`.
    pub catalog_dir: PathBuf,
    /// Telemetry toggle. Default: false.
    pub telemetry: bool,
    /// Keys explicitly present in a file or env override.
    explicit: HashSet<String>,
}

impl Default for GtvParams {
    fn default() -> Self {
        Self {
            memory_target: MemorySpec::default(),
            memory_obj_area: MemAreaSpec::default(),
            cuda_mode: CudaMode::default(),
            cuda_vram: CudaVramSpec::default(),
            cuda_device: 0,
            accel_cache: AccelCache::default(),
            accel_sync_threshold: None,
            accel_sync_interval_secs: 5,
            accel_heat_half_life_secs: 30,
            default_mode: QueryMode::default(),
            log_level: Level::default(),
            spill_dir: None,
            worker_threads: None,
            data_dir: PathBuf::from("./data"),
            catalog_dir: PathBuf::from("./catalog"),
            telemetry: false,
            explicit: HashSet::new(),
        }
    }
}

impl GtvParams {
    /// `true` if `key` was explicitly set (in a file or via env).
    pub fn is_explicit(&self, key: &str) -> bool {
        self.explicit.contains(&key.to_uppercase())
    }

    /// Parse parameters from any reader (`KEY=VALUE` / `KEY VALUE` lines).
    ///
    /// Returns the parsed set plus any non-fatal warnings (unknown keys). A
    /// malformed line aborts with [`ParamError::Parse`] / [`ParamError::InvalidValue`].
    pub fn from_reader<R: Read>(mut reader: R) -> Result<(GtvParams, Warnings), ParamError> {
        let mut content = String::new();
        reader.read_to_string(&mut content).map_err(|e| ParamError::Io {
            path: PathBuf::from("<reader>"),
            source: e,
        })?;
        let mut params = GtvParams::default();
        let mut warnings = Vec::new();

        for (idx, raw) in content.lines().enumerate() {
            let line_no = idx + 1;
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with("--") {
                continue;
            }
            let (k, v) = parse_key_value(line, line_no)?;
            match params.set_key(&k, v) {
                Ok(true) => {}
                Ok(false) => warnings.push(Warning {
                    line: Some(line_no),
                    key: Some(k.clone()),
                    message: format!("unknown parameter `{k}`, ignored"),
                }),
                Err(e) => return Err(e),
            }
        }
        Ok((params, warnings))
    }

    /// Parse parameters from a file path.
    pub fn from_path(path: &Path) -> Result<(GtvParams, Warnings), ParamError> {
        let file = std::fs::File::open(path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                ParamError::FileNotFound(path.to_path_buf())
            } else {
                ParamError::Io {
                    path: path.to_path_buf(),
                    source: e,
                }
            }
        })?;
        Self::from_reader(std::io::BufReader::new(file))
    }

    /// Locate, parse and overlay environment overrides for the parameters file.
    ///
    /// Never errors: a missing file simply yields defaults (plus a warning if a
    /// path was named but unreadable). `GTV_PARAM_<KEY>` env vars win over the
    /// file. Returns the final set and any accumulated warnings.
    pub fn load() -> (GtvParams, Warnings) {
        let mut params = GtvParams::default();
        let mut warnings = Vec::new();

        if let Some(path) = locate_file() {
            match Self::from_path(&path) {
                Ok((p, mut w)) => {
                    params = p;
                    warnings.append(&mut w);
                }
                Err(e) => warnings.push(Warning {
                    line: None,
                    key: None,
                    message: format!("failed to read {}: {e}", path.display()),
                }),
            }
        }

        apply_env_overrides(&mut params, &mut warnings);
        (params, warnings)
    }

    /// Resolve concrete values, given the optional CUDA device VRAM total in
    /// bytes (when known). `None` means "no device".
    ///
    /// `CUDA_MODE=ON` with no device is a hard error ([`ParamError::CudaNoDevice`]);
    /// `AUTO` simply stays disabled. Memory validation (`obj_area <= target`)
    /// is enforced here.
    pub fn resolve(&self, device_total: Option<u64>) -> Result<ResolvedParams, ParamError> {
        let sys_total = system_memory_bytes();

        let (memory_target_bytes, memory_target_explicit) = match self.memory_target {
            MemorySpec::Bytes(b) => (b, true),
            MemorySpec::SystemFraction(f) => ((sys_total as f64 * f) as u64, false),
        };

        let memory_obj_area_bytes = match self.memory_obj_area {
            MemAreaSpec::Bytes(b) => b,
            MemAreaSpec::TargetFraction(f) => (memory_target_bytes as f64 * f) as u64,
        };
        if memory_obj_area_bytes > memory_target_bytes {
            return Err(ParamError::ObjAreaExceedsTarget {
                area: memory_obj_area_bytes,
                target: memory_target_bytes,
            });
        }

        let (cuda_enabled, cuda_vram_bytes, cuda_device) = match self.cuda_mode {
            CudaMode::Off => (false, 0u64, self.cuda_device),
            CudaMode::On => {
                let total = device_total.ok_or(ParamError::CudaNoDevice)?;
                let vram = match self.cuda_vram {
                    CudaVramSpec::Bytes(b) => b,
                    CudaVramSpec::DeviceFraction(f) => (total as f64 * f) as u64,
                };
                (true, vram, self.cuda_device)
            }
            CudaMode::Auto => match device_total {
                Some(total) => {
                    let vram = match self.cuda_vram {
                        CudaVramSpec::Bytes(b) => b,
                        CudaVramSpec::DeviceFraction(f) => (total as f64 * f) as u64,
                    };
                    (true, vram, self.cuda_device)
                }
                None => (false, 0, self.cuda_device),
            },
        };

        // Env GTV_SPILL_DIR wins over the file/default spill directory.
        let spill_dir = std::env::var_os("GTV_SPILL_DIR")
            .map(PathBuf::from)
            .or_else(|| self.spill_dir.clone())
            .unwrap_or_else(|| std::env::temp_dir().join("gtv-spill"));

        // A GPU cache is only meaningful with a device copy; `SHADOW` keeps the
        // ledger but never allocates, and `OFF` short-circuits the registry.
        let accel_cache = if cuda_enabled {
            self.accel_cache
        } else if self.accel_cache == AccelCache::Shadow {
            AccelCache::Shadow
        } else {
            AccelCache::Off
        };
        // PG-Strom applies pending redo once 25% of the redo buffer is dirty;
        // the equivalent here is a quarter of the VRAM budget.
        let accel_sync_threshold_bytes = self
            .accel_sync_threshold
            .unwrap_or((cuda_vram_bytes as f64 * 0.25) as u64);

        Ok(ResolvedParams {
            memory_target_bytes,
            memory_target_explicit,
            memory_obj_area_bytes,
            cuda_enabled,
            cuda_vram_bytes,
            cuda_device,
            accel_cache,
            accel_sync_threshold_bytes,
            accel_sync_interval: Duration::from_secs(self.accel_sync_interval_secs),
            accel_heat_half_life: Duration::from_secs(self.accel_heat_half_life_secs),
            default_mode: self.default_mode,
            log_level: self.log_level,
            spill_dir,
            worker_threads: self.worker_threads,
            data_dir: self.data_dir.clone(),
            catalog_dir: self.catalog_dir.clone(),
            telemetry: self.telemetry,
        })
    }

    /// Dispatch a single `KEY=VALUE`. Returns `Ok(true)` when handled,
    /// `Ok(false)` for an unrecognized key, `Err` for a bad value.
    fn set_key(&mut self, key: &str, value: &str) -> Result<bool, ParamError> {
        let handled = match key {
            "MEMORY_TARGET" => {
                self.memory_target = parse_memory(value).map_err(|m| invalid("MEMORY_TARGET", value, &m))?;
                true
            }
            "MEMORY_OBJ_AREA" => {
                self.memory_obj_area = parse_memarea(value).map_err(|m| invalid("MEMORY_OBJ_AREA", value, &m))?;
                true
            }
            "CUDA_MODE" => {
                self.cuda_mode = parse_cuda_mode(value).map_err(|m| invalid("CUDA_MODE", value, &m))?;
                true
            }
            "CUDA_VRAM" => {
                self.cuda_vram = parse_cuda_vram(value).map_err(|m| invalid("CUDA_VRAM", value, &m))?;
                true
            }
            "CUDA_DEVICE" => {
                self.cuda_device = value.trim().parse().map_err(|_| {
                    invalid("CUDA_DEVICE", value, "not a non-negative integer")
                })?;
                true
            }
            "ACCEL_CACHE" => {
                self.accel_cache = parse_accel_cache(value).map_err(|m| invalid("ACCEL_CACHE", value, &m))?;
                true
            }
            "ACCEL_SYNC_THRESHOLD" => {
                self.accel_sync_threshold = if value.trim().eq_ignore_ascii_case("AUTO") {
                    None
                } else {
                    Some(parse_size(value).map_err(|m| invalid("ACCEL_SYNC_THRESHOLD", value, &m))?)
                };
                true
            }
            "ACCEL_SYNC_INTERVAL" => {
                self.accel_sync_interval_secs = value.trim().parse().map_err(|_| {
                    invalid("ACCEL_SYNC_INTERVAL", value, "not a non-negative number of seconds")
                })?;
                true
            }
            "ACCEL_HEAT_HALF_LIFE" => {
                self.accel_heat_half_life_secs = value.trim().parse().map_err(|_| {
                    invalid("ACCEL_HEAT_HALF_LIFE", value, "not a non-negative number of seconds")
                })?;
                true
            }
            "DEFAULT_MODE" => {
                self.default_mode = parse_default_mode(value).map_err(|m| invalid("DEFAULT_MODE", value, &m))?;
                true
            }
            "LOG_LEVEL" => {
                self.log_level = parse_level(value).map_err(|m| invalid("LOG_LEVEL", value, &m))?;
                true
            }
            "SPILL_DIR" => {
                self.spill_dir = Some(PathBuf::from(value.trim()));
                true
            }
            "WORKER_THREADS" => {
                self.worker_threads = if value.trim().eq_ignore_ascii_case("AUTO") {
                    None
                } else {
                    Some(value.trim().parse().map_err(|_| {
                        invalid("WORKER_THREADS", value, "not a non-negative integer")
                    })?)
                };
                true
            }
            "DATA_DIR" => {
                self.data_dir = PathBuf::from(value.trim());
                true
            }
            "CATALOG_DIR" => {
                self.catalog_dir = PathBuf::from(value.trim());
                true
            }
            "TELEMETRY" => {
                self.telemetry = parse_bool(value).map_err(|m| invalid("TELEMETRY", value, &m))?;
                true
            }
            _ => false,
        };
        if handled {
            self.explicit.insert(key.to_string());
        }
        Ok(handled)
    }
}

fn invalid(key: &str, value: &str, reason: &str) -> ParamError {
    ParamError::InvalidValue {
        key: key.to_string(),
        value: value.to_string(),
        reason: reason.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Resolved view.
// ---------------------------------------------------------------------------

/// Concrete, engine-ready parameters produced by [`GtvParams::resolve`].
#[derive(Debug, Clone)]
pub struct ResolvedParams {
    /// Total memory budget in bytes (always concrete).
    pub memory_target_bytes: u64,
    /// `true` only when `MEMORY_TARGET` was explicitly set (controls whether a
    /// DataFusion memory pool is mounted).
    pub memory_target_explicit: bool,
    /// Resident-in-memory object area in bytes.
    pub memory_obj_area_bytes: u64,
    /// Whether CUDA is actually engaged.
    pub cuda_enabled: bool,
    /// CUDA VRAM budget in bytes (0 when CUDA is off).
    pub cuda_vram_bytes: u64,
    /// Selected CUDA device ordinal.
    pub cuda_device: usize,
    /// How the device-resident cache is engaged.
    pub accel_cache: AccelCache,
    /// Pending-delta bytes that trigger a batched upload (concrete).
    pub accel_sync_threshold_bytes: u64,
    /// Wall-clock interval after which pending delta is applied regardless.
    pub accel_sync_interval: Duration,
    /// Half-life of resident-block heat. `Duration::ZERO` disables heat and
    /// restores pure LRU eviction.
    pub accel_heat_half_life: Duration,
    /// Selected default query dialect.
    pub default_mode: QueryMode,
    /// Selected log level.
    pub log_level: Level,
    /// Effective spill directory.
    pub spill_dir: PathBuf,
    /// Effective worker thread count (`None` => env / CPU count).
    pub worker_threads: Option<usize>,
    /// Data directory.
    pub data_dir: PathBuf,
    /// Catalog directory.
    pub catalog_dir: PathBuf,
    /// Telemetry toggle.
    pub telemetry: bool,
}

impl ResolvedParams {
    /// Stable, human-readable summary lines (e.g. for the `.params` CLI command).
    pub fn summary_lines(&self) -> Vec<String> {
        let mib = 1024 * 1024;
        let mut v = Vec::new();
        let mt = format!("{} MiB", self.memory_target_bytes / mib);
        let mt_note = if self.memory_target_explicit {
            "explicit"
        } else {
            "auto: system/2"
        };
        v.push(format!("MEMORY_TARGET    = {:<12} ({})", mt, mt_note));
        v.push(format!(
            "MEMORY_OBJ_AREA  = {:<12} (auto: target/2)",
            format!("{} MiB", self.memory_obj_area_bytes / mib)
        ));
        v.push(format!(
            "CUDA_MODE        = {}",
            if self.cuda_enabled { "ON" } else { "OFF" }
        ));
        let vram_note = if self.cuda_enabled {
            "device fraction"
        } else {
            "no device"
        };
        v.push(format!(
            "CUDA_VRAM        = {:<12} ({})",
            format!("{} MiB", self.cuda_vram_bytes / mib),
            vram_note
        ));
        let threshold = format!("{} MiB", self.accel_sync_threshold_bytes / mib);
        v.push(format!(
            "ACCEL_CACHE      = {:<12} (sync: {} / {}s)",
            self.accel_cache.as_str(),
            threshold,
            self.accel_sync_interval.as_secs()
        ));
        let heat = if self.accel_heat_half_life.is_zero() {
            "LRU (heat off)".to_string()
        } else {
            format!("{}s half-life", self.accel_heat_half_life.as_secs())
        };
        v.push(format!("ACCEL_HEAT      = {heat}"));
        v.push(format!(
            "DEFAULT_MODE     = {}",
            match self.default_mode {
                QueryMode::Sql => "SQL",
                QueryMode::Q => "Q",
            }
        ));
        v.push(format!("LOG_LEVEL        = {}", self.log_level.as_rust_log().to_uppercase()));
        v.push(format!("SPILL_DIR        = {}", self.spill_dir.display()));
        v.push(format!(
            "WORKER_THREADS   = {}",
            self.worker_threads
                .map(|n| n.to_string())
                .unwrap_or_else(|| "auto".to_string())
        ));
        v.push(format!("DATA_DIR         = {}", self.data_dir.display()));
        v.push(format!("CATALOG_DIR      = {}", self.catalog_dir.display()));
        v.push(format!("TELEMETRY        = {}", self.telemetry));
        v
    }
}

// ---------------------------------------------------------------------------
// File discovery & env overrides.
// ---------------------------------------------------------------------------

/// Search for the parameters file in the standard locations.
///
/// Order: `GTV_PARAMS_FILE` → `./gtvdb.params` → the executable's directory →
/// `/etc/gtvdb/gtvdb.params` (Unix) or `C:\ProgramData\gtvdb\gtvdb.params`
/// (Windows). Returns `None` when none exist (caller then uses defaults).
fn locate_file() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("GTV_PARAMS_FILE") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    let cwd = PathBuf::from("./gtvdb.params");
    if cwd.exists() {
        return Some(cwd);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let p = dir.join("gtvdb.params");
            if p.exists() {
                return Some(p);
            }
        }
    }
    #[cfg(target_os = "windows")]
    {
        let p = PathBuf::from(r"C:\ProgramData\gtvdb\gtvdb.params");
        if p.exists() {
            return Some(p);
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        let p = PathBuf::from("/etc/gtvdb/gtvdb.params");
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn apply_env_overrides(params: &mut GtvParams, warnings: &mut Warnings) {
    const KEYS: &[&str] = &[
        "MEMORY_TARGET",
        "MEMORY_OBJ_AREA",
        "CUDA_MODE",
        "CUDA_VRAM",
        "CUDA_DEVICE",
        "DEFAULT_MODE",
        "LOG_LEVEL",
        "SPILL_DIR",
        "WORKER_THREADS",
        "DATA_DIR",
        "CATALOG_DIR",
        "TELEMETRY",
    ];
    for k in KEYS {
        let env = format!("GTV_PARAM_{k}");
        if let Ok(v) = std::env::var(&env) {
            let key = (*k).to_string();
            match params.set_key(&key, v.trim()) {
                Ok(true) => {}
                Ok(false) => warnings.push(Warning {
                    line: None,
                    key: Some(key),
                    message: format!("unknown env override `{env}`, ignored"),
                }),
                Err(e) => warnings.push(Warning {
                    line: None,
                    key: Some(key),
                    message: format!("invalid env override `{env}`: {e}"),
                }),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar parsers.
// ---------------------------------------------------------------------------

/// Split a line into `(KEY, VALUE)`, supporting both `=` and whitespace.
fn parse_key_value(line: &str, line_no: usize) -> Result<(String, &str), ParamError> {
    let (k, v) = if let Some((k, v)) = line.split_once('=') {
        (k.trim().to_uppercase(), v.trim())
    } else {
        let mut it = line.splitn(2, char::is_whitespace);
        let k = it.next().unwrap_or("").trim().to_uppercase();
        let v = it.next().unwrap_or("").trim();
        (k, v)
    };
    if k.is_empty() || v.is_empty() {
        return Err(ParamError::Parse {
            line: line_no,
            message: "expected KEY=VALUE or `KEY VALUE`".to_string(),
        });
    }
    Ok((k, v))
}

/// Parse a byte count with binary unit suffixes. `16000M` => 16000 * 2^20.
fn parse_size(s: &str) -> Result<u64, String> {
    let s = s.trim();
    if s.is_empty() {
        return Err("empty value".to_string());
    }
    let end = s.find(|c: char| c.is_alphabetic()).unwrap_or(s.len());
    let num = s[..end].trim();
    let unit = s[end..].trim().to_uppercase();
    let value: f64 = num
        .parse()
        .map_err(|_| format!("`{s}` is not a number"))?;
    let mult: u64 = match unit.as_str() {
        "" => 1,
        "K" | "KB" | "KIB" => 1024,
        "M" | "MB" | "MIB" => 1024 * 1024,
        "G" | "GB" | "GIB" => 1024 * 1024 * 1024,
        "T" | "TB" | "TIB" => 1024 * 1024 * 1024 * 1024,
        _ => return Err(format!("unknown unit `{unit}`")),
    };
    Ok((value * mult as f64) as u64)
}

fn parse_memory(s: &str) -> Result<MemorySpec, String> {
    let s = s.trim();
    if s.eq_ignore_ascii_case("AUTO") || s.eq_ignore_ascii_case("SYSTEM") {
        return Ok(MemorySpec::SystemFraction(0.5));
    }
    if let Some(frac) = s
        .strip_prefix("SYSTEM:")
        .or_else(|| s.strip_prefix("system:"))
    {
        let f: f64 = frac
            .trim()
            .parse()
            .map_err(|_| format!("bad fraction `{frac}`"))?;
        return Ok(MemorySpec::SystemFraction(f));
    }
    Ok(MemorySpec::Bytes(parse_size(s)?))
}

fn parse_memarea(s: &str) -> Result<MemAreaSpec, String> {
    let s = s.trim();
    if let Some(frac) = s
        .strip_prefix("TARGET:")
        .or_else(|| s.strip_prefix("target:"))
        .or_else(|| s.strip_prefix("SYSTEM:"))
        .or_else(|| s.strip_prefix("system:"))
    {
        let f: f64 = frac
            .trim()
            .parse()
            .map_err(|_| format!("bad fraction `{frac}`"))?;
        return Ok(MemAreaSpec::TargetFraction(f));
    }
    Ok(MemAreaSpec::Bytes(parse_size(s)?))
}

fn parse_cuda_mode(s: &str) -> Result<CudaMode, String> {
    match s.trim().to_uppercase().as_str() {
        "OFF" | "FALSE" | "NO" | "0" => Ok(CudaMode::Off),
        "ON" | "TRUE" | "YES" | "1" => Ok(CudaMode::On),
        "AUTO" => Ok(CudaMode::Auto),
        _ => Err(format!("expected OFF/ON/AUTO (got `{s}`)")),
    }
}

fn parse_accel_cache(s: &str) -> Result<AccelCache, String> {
    match s.trim().to_uppercase().as_str() {
        "OFF" | "FALSE" | "NO" | "0" => Ok(AccelCache::Off),
        "SHADOW" | "CALIBRATE" => Ok(AccelCache::Shadow),
        "ON" | "TRUE" | "YES" | "1" => Ok(AccelCache::On),
        _ => Err(format!("expected OFF/SHADOW/ON (got `{s}`)")),
    }
}

fn parse_cuda_vram(s: &str) -> Result<CudaVramSpec, String> {
    let s = s.trim();
    if let Some(frac) = s
        .strip_prefix("DEVICE:")
        .or_else(|| s.strip_prefix("device:"))
        .or_else(|| s.strip_prefix("SYSTEM:"))
        .or_else(|| s.strip_prefix("system:"))
    {
        let f: f64 = frac
            .trim()
            .parse()
            .map_err(|_| format!("bad fraction `{frac}`"))?;
        return Ok(CudaVramSpec::DeviceFraction(f));
    }
    Ok(CudaVramSpec::Bytes(parse_size(s)?))
}

fn parse_bool(s: &str) -> Result<bool, String> {
    match s.trim().to_uppercase().as_str() {
        "TRUE" | "YES" | "ON" | "1" => Ok(true),
        "FALSE" | "NO" | "OFF" | "0" => Ok(false),
        _ => Err(format!("expected TRUE/FALSE (got `{s}`)")),
    }
}

fn parse_level(s: &str) -> Result<Level, String> {
    match s.trim().to_uppercase().as_str() {
        "ERROR" => Ok(Level::Error),
        "WARN" | "WARNING" => Ok(Level::Warn),
        "INFO" => Ok(Level::Info),
        "DEBUG" => Ok(Level::Debug),
        "TRACE" => Ok(Level::Trace),
        _ => Err(format!("expected ERROR/WARN/INFO/DEBUG/TRACE (got `{s}`)")),
    }
}

fn parse_default_mode(s: &str) -> Result<QueryMode, String> {
    match s.trim().to_uppercase().as_str() {
        "SQL" | "SQLMODE" => Ok(QueryMode::Sql),
        "Q" => Ok(QueryMode::Q),
        _ => Err(format!("expected SQL or Q (got `{s}`)")),
    }
}

/// Total physical system memory in bytes, with an 8 GiB fallback when probing
/// fails (e.g. an unusual platform).
pub fn system_memory_bytes() -> u64 {
    use sysinfo::System;
    let sys = System::new_all();
    let total = sys.total_memory();
    if total > 0 {
        total
    } else {
        8 * 1024 * 1024 * 1024
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_system_fraction_is_half() {
        assert_eq!(MemorySpec::default(), MemorySpec::SystemFraction(0.5));
        assert!(!MemorySpec::default().is_explicit());
        assert!(MemorySpec::Bytes(1).is_explicit());
    }

    #[test]
    fn size_suffixes() {
        assert_eq!(parse_size("1024K").unwrap(), 1024 * 1024);
        assert_eq!(parse_size("1M").unwrap(), 1024 * 1024);
        assert_eq!(parse_size("2G").unwrap(), 2 * 1024 * 1024 * 1024);
        assert_eq!(parse_size("1T").unwrap(), 1024u64 * 1024 * 1024 * 1024);
        assert_eq!(parse_size("4096").unwrap(), 4096);
        assert!(parse_size("16X").is_err());
    }

    #[test]
    fn defaults_resolve_auto_half_system() {
        let (p, _w) = GtvParams::from_reader("".as_bytes()).unwrap();
        let r = p.resolve(None).unwrap();
        assert!(!r.memory_target_explicit);
        // Half of system memory, which on any real machine is > 0.
        assert!(r.memory_target_bytes > 0);
        assert_eq!(r.memory_obj_area_bytes, r.memory_target_bytes / 2);
        // No CUDA device injected => CUDA stays OFF in AUTO mode.
        assert!(!r.cuda_enabled);
        assert_eq!(r.default_mode, QueryMode::Sql);
        assert_eq!(r.log_level, Level::Info);
    }

    #[test]
    fn explicit_obj_area_over_target_is_rejected() {
        let (p, _) = GtvParams::from_reader("MEMORY_TARGET=100M\nMEMORY_OBJ_AREA=200M\n".as_bytes()).unwrap();
        assert!(p.resolve(None).is_err());
    }
}
