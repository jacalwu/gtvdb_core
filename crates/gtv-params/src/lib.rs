//! An Oracle-style parameter file for gtVdB.
//!
//! The file is a flat list of `KEY=VALUE` lines, all-uppercase keys, `#` or `--`
//! comment lines, and blank lines ignored. Values may carry binary size suffixes
//! (`K`/`M`/`G`/`T`, an optional trailing `B` is ignored) and a few symbolic
//! forms (`AUTO`, `SYSTEM`, fractions). Example:
//!
//! ```text
//! # how much host RAM gtVdB may use
//! MEMORY_TARGET=16000M
//! MEMORY_OBJ_AREA=8000M
//! CUDA_MODE=TRUE
//! CUDA_VRAM=6000M
//! DEFAULT_MODE=SQL
//! ```
//!
//! Every parameter has a default, so a missing file is not an error: gtVdB
//! starts with the documented defaults and the resolved values are always
//! inspectable (see [`GtvParams::resolve`] and the `.params` REPL command).
//!
//! # Why a file and not just env vars
//!
//! A single file is reviewable, version-controllable, and the same shape on
//! every deployment target (laptop, rack server, container). Environment
//! overrides are still supported for the container case: `GTV_PARAM_<KEY>` wins
//! over the file, and `GTV_PARAMS_FILE` points at a non-default path.

mod error;
pub use error::{ParamError, Warning, Warnings};

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Value types
// ---------------------------------------------------------------------------

/// How a byte budget is expressed before resolution.
///
/// `SystemFraction` means "a fraction of the relevant pool": for `MEMORY_TARGET`
/// that pool is host RAM, for `CUDA_VRAM` it is device VRAM. Resolution needs the
/// pool size, which is why [`MemorySpec::resolve_against`] takes it as an argument.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemorySpec {
    /// An explicit byte count (with optional `K`/`M`/`G`/`T` suffix).
    Bytes(u64),
    /// A fraction of the relevant pool (host RAM or device VRAM).
    SystemFraction(f64),
}

impl Default for MemorySpec {
    fn default() -> Self {
        MemorySpec::SystemFraction(0.5)
    }
}

impl MemorySpec {
    /// Resolve to a concrete byte count against `total` (host RAM or device VRAM).
    pub fn resolve_against(&self, total: u64) -> u64 {
        match self {
            MemorySpec::Bytes(b) => *b,
            MemorySpec::SystemFraction(f) => ((total as f64) * f) as u64,
        }
    }

    /// True when the operator wrote an explicit size (vs. leaving the default).
    pub fn is_explicit(&self) -> bool {
        matches!(self, MemorySpec::Bytes(_))
    }
}

/// The resident-object memory region, expressed either explicitly or as a
/// fraction of the resolved [`GtvParams::memory_target`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemAreaSpec {
    Bytes(u64),
    /// Fraction of the resolved `MEMORY_TARGET`.
    TargetFraction(f64),
}

impl Default for MemAreaSpec {
    fn default() -> Self {
        MemAreaSpec::TargetFraction(0.5)
    }
}

impl MemAreaSpec {
    fn resolve(&self, target: u64) -> u64 {
        match self {
            MemAreaSpec::Bytes(b) => *b,
            MemAreaSpec::TargetFraction(f) => ((target as f64) * f) as u64,
        }
    }
}

/// Whether the CUDA backend is engaged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CudaMode {
    /// Never register the device backend.
    Off,
    /// Always register it; error at startup if no device is present.
    On,
    /// Register it only when a CUDA device is detected (the safe default: a GPU
    /// box works out of the box, a CPU-only box stays CPU-only, and an explicit
    /// `FALSE` still forces it off).
    #[default]
    Auto,
}

/// The dialect the REPL/query entrypoint defaults to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QueryMode {
    /// DataFusion SQL (the core build's only available dialect).
    #[default]
    Sql,
    /// The domain "Q" dialect (quantum / extension dialect). Requires the
    /// matching extension to be installed; the core build falls back to SQL and
    /// logs the mismatch when `Q` is requested without it.
    Q,
}

/// Logging verbosity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Level {
    Error,
    Warn,
    #[default]
    Info,
    Debug,
    Trace,
}

impl Level {
    /// The `RUST_LOG` directive string this level maps to.
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

// ---------------------------------------------------------------------------
// The parameter set
// ---------------------------------------------------------------------------

/// All tunable knobs for a gtVdB process, as written in the parameter file
/// (defaults where nothing was written).
///
/// Two of these are "dynamic": [`memory_target`](GtvParams::memory_target)
/// defaults to half of host RAM and [`cuda_vram`](GtvParams::cuda_vram) to 80%
/// of device VRAM. Those fractions are resolved by [`GtvParams::resolve`], which
/// needs the actual pool sizes.
#[derive(Debug, Clone)]
pub struct GtvParams {
    /// Total host memory gtVdB may allocate. Default: half of system RAM.
    pub memory_target: MemorySpec,
    /// Region reserved for resident-in-memory objects. Default: half of `MEMORY_TARGET`.
    pub memory_obj_area: MemAreaSpec,
    /// Whether the CUDA backend is engaged.
    pub cuda_mode: CudaMode,
    /// Device memory budget. Default: 80% of device VRAM.
    pub cuda_vram: MemorySpec,
    /// Which CUDA device ordinal to use. Default 0.
    pub cuda_device: usize,
    /// Default query dialect.
    pub default_mode: QueryMode,
    /// Logging verbosity. Default `INFO`.
    pub log_level: Level,
    /// Spill-to-disk directory for DataFusion. `None` => engine default.
    pub spill_dir: Option<PathBuf>,
    /// Query parallelism. `None` => logical CPU count.
    pub worker_threads: Option<usize>,
    /// Root data directory. `None` => `./.gtvdb`.
    pub data_dir: Option<PathBuf>,
    /// Catalog directory. `None` => `data_dir/catalog`.
    pub catalog_dir: Option<PathBuf>,
    /// Telemetry emission. Default off.
    pub telemetry: bool,
}

impl Default for GtvParams {
    fn default() -> Self {
        GtvParams {
            memory_target: MemorySpec::default(),
            memory_obj_area: MemAreaSpec::default(),
            cuda_mode: CudaMode::default(),
            // 80% of device VRAM is the deployment default; the CUDA crate's own
            // direct-open() fallback stays at 60% for headroom, but the
            // parameter-driven path realizes the operator's 80% intent.
            cuda_vram: MemorySpec::SystemFraction(0.8),
            cuda_device: 0,
            default_mode: QueryMode::default(),
            log_level: Level::default(),
            spill_dir: None,
            worker_threads: None,
            data_dir: None,
            catalog_dir: None,
            telemetry: false,
        }
    }
}

/// Concrete, deployment-ready values after resolution.
#[derive(Debug, Clone)]
pub struct ResolvedParams {
    pub memory_target_bytes: u64,
    pub memory_obj_area_bytes: u64,
    pub cuda_enabled: bool,
    pub cuda_vram_bytes: u64,
    pub cuda_device: usize,
    pub default_mode: QueryMode,
    pub log_level: Level,
    pub spill_dir: PathBuf,
    pub worker_threads: Option<usize>,
    pub data_dir: PathBuf,
    pub catalog_dir: PathBuf,
    pub telemetry: bool,
}

impl ResolvedParams {
    /// A human-readable, stable-ordered dump for startup banners and the
    /// `.params` REPL command.
    pub fn summary_lines(&self) -> Vec<String> {
        let mut out = Vec::new();
        out.push(format!(
            "MEMORY_TARGET    = {} ({} MiB)",
            human(self.memory_target_bytes),
            self.memory_target_bytes / (1 << 20)
        ));
        out.push(format!(
            "MEMORY_OBJ_AREA  = {} ({} MiB)",
            human(self.memory_obj_area_bytes),
            self.memory_obj_area_bytes / (1 << 20)
        ));
        out.push(format!(
            "CUDA_MODE        = {}",
            if self.cuda_enabled { "TRUE" } else { "FALSE" }
        ));
        out.push(format!(
            "CUDA_VRAM       = {} ({} MiB)",
            human(self.cuda_vram_bytes),
            self.cuda_vram_bytes / (1 << 20)
        ));
        out.push(format!("CUDA_DEVICE      = {}", self.cuda_device));
        out.push(format!(
            "DEFAULT_MODE     = {}",
            match self.default_mode {
                QueryMode::Sql => "SQL",
                QueryMode::Q => "Q",
            }
        ));
        out.push(format!("LOG_LEVEL        = {}", self.log_level.as_rust_log().to_uppercase()));
        out.push(format!("SPILL_DIR        = {}", self.spill_dir.display()));
        out.push(format!(
            "WORKER_THREADS   = {}",
            self.worker_threads
                .map(|n| n.to_string())
                .unwrap_or_else(|| "AUTO".into())
        ));
        out.push(format!("DATA_DIR         = {}", self.data_dir.display()));
        out.push(format!("CATALOG_DIR      = {}", self.catalog_dir.display()));
        out.push(format!("TELEMETRY        = {}", if self.telemetry { "ON" } else { "OFF" }));
        out
    }
}

fn human(bytes: u64) -> String {
    const UNIT: u64 = 1 << 20; // MiB, the natural unit for these knobs
    format!("{} MiB", bytes / UNIT)
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

impl GtvParams {
    /// Parse parameters from a reader, returning the parsed set and any
    /// non-fatal warnings (unknown keys, etc.).
    pub fn from_reader<R: std::io::Read>(
        mut reader: R,
    ) -> Result<(GtvParams, Warnings), ParamError> {
        let mut params = GtvParams::default();
        let mut warnings = Vec::new();
        let mut raw: BTreeMap<String, (usize, String)> = BTreeMap::new();

        let mut content = String::new();
        reader
            .read_to_string(&mut content)
            .map_err(|e| ParamError::Io {
                path: PathBuf::from("<reader>"),
                source: e,
            })?;

        for (idx, line) in content.lines().enumerate() {
            let line_no = idx + 1;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("--") {
                continue;
            }
            let (key, value) = match trimmed.split_once('=') {
                Some((k, v)) => (k.trim(), v.trim()),
                None => {
                    // Allow `key value` (space separated) as well as `key=value`.
                    let mut it = trimmed.splitn(2, char::is_whitespace);
                    let k = it.next().unwrap_or("").trim();
                    let v = it.next().unwrap_or("").trim();
                    if k.is_empty() || v.is_empty() {
                        return Err(ParamError::Parse {
                            line: line_no,
                            message: format!("expected `KEY=VALUE`, got `{trimmed}`"),
                        });
                    }
                    (k, v)
                }
            };
            if key.is_empty() {
                return Err(ParamError::Parse {
                    line: line_no,
                    message: "empty key".into(),
                });
            }
            raw.insert(key.to_ascii_uppercase(), (line_no, value.to_string()));
        }

        for (key, (line_no, value)) in raw {
            set_key(&mut params, &key, &value, line_no, &mut warnings)?;
        }
        Ok((params, warnings))
    }

    /// Load and parse a parameter file by path.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<(GtvParams, Warnings), ParamError> {
        let path = path.as_ref();
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
        Self::from_reader(file)
    }

    /// Locate and load the parameter file using the standard search order, then
    /// apply `GTV_PARAM_<KEY>` environment overrides.
    ///
    /// A missing file is not an error: the documented defaults are used and an
    /// (empty) warning list returned. This is what makes gtVdB runnable with zero
    /// configuration on every target.
    pub fn load() -> (GtvParams, Warnings) {
        let mut warnings = Vec::new();
        let mut params = match locate_file() {
            Some(path) => match Self::from_path(&path) {
                Ok((p, mut w)) => {
                    warnings.append(&mut w);
                    p
                }
                Err(e) => {
                    warnings.push(Warning {
                        line: 0,
                        key: String::new(),
                        message: format!("parameter file {} unreadable: {e}; using defaults", path.display()),
                    });
                    GtvParams::default()
                }
            },
            None => GtvParams::default(),
        };
        apply_env_overrides(&mut params, &mut warnings);
        (params, warnings)
    }

    /// Resolve symbolic values against the actual machine.
    ///
    /// `device_total` is the detected device VRAM in bytes, or `None` when no
    /// CUDA device is available (or the `cuda` feature is off). `CUDA_MODE=AUTO`
    /// resolves to off in that case; `CUDA_MODE=ON` is an error.
    pub fn resolve(&self, device_total: Option<u64>) -> Result<ResolvedParams, ParamError> {
        let sys = system_memory_bytes().unwrap_or(8 << 30);
        let memory_target_bytes = self.memory_target.resolve_against(sys);
        let memory_obj_area_bytes = self.memory_obj_area.resolve(memory_target_bytes);
        if memory_obj_area_bytes > memory_target_bytes {
            return Err(ParamError::ObjAreaExceedsTarget {
                area: memory_obj_area_bytes,
                target: memory_target_bytes,
            });
        }

        let (cuda_enabled, cuda_vram_bytes) = match self.cuda_mode {
            CudaMode::Off => (false, 0),
            CudaMode::On => match device_total {
                Some(t) => (true, self.cuda_vram.resolve_against(t).min(t)),
                None => return Err(ParamError::CudaNoDevice),
            },
            CudaMode::Auto => match device_total {
                Some(t) => (true, self.cuda_vram.resolve_against(t).min(t)),
                None => (false, 0),
            },
        };

        let data_dir = self
            .data_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from("./.gtvdb"));
        let catalog_dir = self
            .catalog_dir
            .clone()
            .unwrap_or_else(|| data_dir.join("catalog"));
        let spill_dir = self
            .spill_dir
            .clone()
            .unwrap_or_else(|| std::env::temp_dir().join("gtv-spill"));

        Ok(ResolvedParams {
            memory_target_bytes,
            memory_obj_area_bytes,
            cuda_enabled,
            cuda_vram_bytes,
            cuda_device: self.cuda_device,
            default_mode: self.default_mode,
            log_level: self.log_level,
            spill_dir,
            worker_threads: self.worker_threads,
            data_dir,
            catalog_dir,
            telemetry: self.telemetry,
        })
    }
}

/// Dispatch one `KEY=VALUE` assignment. Returns a fatal [`ParamError`] on a bad
/// value; unknown keys are recorded as warnings and ignored.
fn set_key(
    params: &mut GtvParams,
    key: &str,
    value: &str,
    line: usize,
    warnings: &mut Warnings,
) -> Result<(), ParamError> {
    let v = value.trim();
    let err = |reason: &str| ParamError::InvalidValue {
        key: key.to_string(),
        value: value.to_string(),
        reason: reason.to_string(),
    };
    match key {
        "MEMORY_TARGET" => {
            params.memory_target = parse_memory(v).map_err(|r| err(&r))?;
        }
        "MEMORY_OBJ_AREA" => {
            params.memory_obj_area = parse_memarea(v).map_err(|r| err(&r))?;
        }
        "CUDA_MODE" => {
            params.cuda_mode = parse_cuda_mode(v).map_err(|r| err(&r))?;
        }
        "CUDA_VRAM" => {
            params.cuda_vram = parse_memory(v).map_err(|r| err(&r))?;
        }
        "CUDA_DEVICE" => {
            params.cuda_device = v
                .parse::<usize>()
                .map_err(|_| err("expected a non-negative integer"))?;
        }
        "DEFAULT_MODE" => {
            params.default_mode = match v.to_ascii_uppercase().as_str() {
                "SQL" => QueryMode::Sql,
                "Q" => QueryMode::Q,
                _ => return Err(err("expected SQL or Q")),
            };
        }
        "LOG_LEVEL" => {
            params.log_level = match v.to_ascii_uppercase().as_str() {
                "ERROR" => Level::Error,
                "WARN" | "WARNING" => Level::Warn,
                "INFO" => Level::Info,
                "DEBUG" => Level::Debug,
                "TRACE" => Level::Trace,
                _ => return Err(err("expected ERROR/WARN/INFO/DEBUG/TRACE")),
            };
        }
        "SPILL_DIR" => {
            if v.is_empty() {
                return Err(err("expected a path"));
            }
            params.spill_dir = Some(PathBuf::from(v));
        }
        "WORKER_THREADS" => {
            if matches!(v.to_ascii_uppercase().as_str(), "AUTO" | "0" | "") {
                params.worker_threads = None;
            } else {
                let n = v.parse::<usize>().map_err(|_| err("expected a positive integer or AUTO"))?;
                params.worker_threads = Some(n.max(1));
            }
        }
        "DATA_DIR" => {
            if v.is_empty() {
                return Err(err("expected a path"));
            }
            params.data_dir = Some(PathBuf::from(v));
        }
        "CATALOG_DIR" => {
            if v.is_empty() {
                return Err(err("expected a path"));
            }
            params.catalog_dir = Some(PathBuf::from(v));
        }
        "TELEMETRY" => {
            params.telemetry = parse_bool(v).map_err(|r| err(&r))?;
        }
        _ => {
            warnings.push(Warning {
                line,
                key: key.to_string(),
                message: "unknown parameter; ignored".into(),
            });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Scalar parsers
// ---------------------------------------------------------------------------

/// Parse a [`MemorySpec`]: explicit size, `AUTO`/`SYSTEM` (=> 0.5) or
/// `SYSTEM:0.7` (=> fraction).
fn parse_memory(s: &str) -> Result<MemorySpec, String> {
    let s = s.trim();
    let up = s.to_ascii_uppercase();
    if up == "AUTO" || up == "SYSTEM" {
        return Ok(MemorySpec::SystemFraction(0.5));
    }
    if let Some(frac) = up.strip_prefix("SYSTEM:") {
        let f = frac.parse::<f64>().map_err(|_| "bad SYSTEM fraction".to_string())?;
        return Ok(MemorySpec::SystemFraction(clamp_frac(f)?));
    }
    Ok(MemorySpec::Bytes(parse_size(s)?))
}

/// Parse a [`MemAreaSpec`]: explicit size, `AUTO`/`TARGET` (=> 0.5 of target)
/// or `TARGET:0.5`.
fn parse_memarea(s: &str) -> Result<MemAreaSpec, String> {
    let s = s.trim();
    let up = s.to_ascii_uppercase();
    if up == "AUTO" || up == "TARGET" {
        return Ok(MemAreaSpec::TargetFraction(0.5));
    }
    if let Some(frac) = up.strip_prefix("TARGET:") {
        let f = frac.parse::<f64>().map_err(|_| "bad TARGET fraction".to_string())?;
        return Ok(MemAreaSpec::TargetFraction(clamp_frac(f)?));
    }
    Ok(MemAreaSpec::Bytes(parse_size(s)?))
}

fn clamp_frac(f: f64) -> Result<f64, String> {
    if !(0.0..=1.0).contains(&f) {
        return Err("fraction must be between 0 and 1".into());
    }
    Ok(f)
}

/// Parse a size with binary suffixes: `K`/`M`/`G`/`T` (case-insensitive), an
/// optional trailing `B` is ignored. `16000M` => 16000 * 2^20.
fn parse_size(s: &str) -> Result<u64, String> {
    let s = s.trim();
    if s.is_empty() {
        return Err("empty size".into());
    }
    // Split leading number (allowing one decimal point) from the unit.
    let num_end = s
        .char_indices()
        .find(|(_, c)| !c.is_ascii_digit() && *c != '.')
        .map(|(i, _)| i)
        .unwrap_or(s.len());
    let (num_part, unit_part) = s.split_at(num_end);
    let value: f64 = num_part
        .parse()
        .map_err(|_| format!("`{s}` is not a number"))?;
    if value < 0.0 {
        return Err("size must be non-negative".into());
    }
    let unit = unit_part.trim().to_ascii_uppercase();
    let mult: u64 = match unit.as_str() {
        "" | "B" => 1,
        "K" | "KB" => 1 << 10,
        "M" | "MB" => 1 << 20,
        "G" | "GB" => 1 << 30,
        "T" | "TB" => 1 << 40,
        "P" | "PB" => 1 << 50,
        other => return Err(format!("unknown unit `{other}`")),
    };
    Ok((value * mult as f64) as u64)
}

fn parse_bool(s: &str) -> Result<bool, String> {
    match s.trim().to_ascii_uppercase().as_str() {
        "TRUE" | "YES" | "ON" | "1" => Ok(true),
        "FALSE" | "NO" | "OFF" | "0" => Ok(false),
        _ => Err("expected TRUE/FALSE".into()),
    }
}

fn parse_cuda_mode(s: &str) -> Result<CudaMode, String> {
    match s.trim().to_ascii_uppercase().as_str() {
        "OFF" | "FALSE" | "NO" | "0" => Ok(CudaMode::Off),
        "ON" | "TRUE" | "YES" | "1" => Ok(CudaMode::On),
        "AUTO" => Ok(CudaMode::Auto),
        _ => Err("expected OFF/ON/AUTO".into()),
    }
}

// ---------------------------------------------------------------------------
// Discovery & env overrides
// ---------------------------------------------------------------------------

/// Standard search order for the parameter file.
///
/// 1. `GTV_PARAMS_FILE` env var (explicit override, must exist).
/// 2. `./gtvdb.params` (current directory).
/// 3. `<exe directory>/gtvdb.params`.
/// 4. `/etc/gtvdb/gtvdb.params` (Unix) or
///    `C:\ProgramData\gtvdb\gtvdb.params` (Windows).
fn locate_file() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("GTV_PARAMS_FILE") {
        let p = PathBuf::from(p);
        return if p.exists() { Some(p) } else { None };
    }
    let candidates: Vec<PathBuf> = {
        let mut v = vec![PathBuf::from("gtvdb.params")];
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                v.push(dir.join("gtvdb.params"));
            }
        }
        #[cfg(target_os = "windows")]
        v.push(PathBuf::from(r"C:\ProgramData\gtvdb\gtvdb.params"));
        #[cfg(not(target_os = "windows"))]
        v.push(PathBuf::from("/etc/gtvdb/gtvdb.params"));
        v
    };
    candidates.into_iter().find(|p| p.exists())
}

/// Apply `GTV_PARAM_<KEY>` environment variables as overrides, highest
/// precedence. Unknown keys are ignored (a typo in an env var should not crash
/// startup, but it is not worth a warning either since env is for automation).
fn apply_env_overrides(params: &mut GtvParams, warnings: &mut Warnings) {
    let keys = [
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
    for key in keys {
        let env_key = format!("GTV_PARAM_{key}");
        if let Some(val) = std::env::var_os(&env_key) {
            let val = val.to_string_lossy().to_string();
            // Env overrides are loud: a wrong override should fail fast.
            if let Err(e) = set_key(params, key, &val, 0, warnings) {
                // Promote to a warning rather than crashing the process; the
                // file-based value stays in effect.
                warnings.push(Warning {
                    line: 0,
                    key: env_key,
                    message: format!("override ignored: {e}"),
                });
            }
        }
    }
}

/// Physical host RAM in bytes, or `None` if it cannot be determined.
///
/// Used only to resolve the `MEMORY_TARGET` default (half of it). Detection
/// failure is not fatal: the caller supplies an 8 GiB fallback.
fn system_memory_bytes() -> Option<u64> {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let total = sys.total_memory();
    if total > 0 {
        Some(total)
    } else {
        None
    }
}
