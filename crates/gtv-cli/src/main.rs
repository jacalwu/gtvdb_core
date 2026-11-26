//! `gtv` — the minimal, zero-extension REPL shipped with **gtvdb_core**.
//!
//! This binary deliberately depends only on platform crates. Domain workloads
//! (banking / HFT / quantum compute / crypto / q) are NOT linked in: they live
//! in the companion gtvdb_ext repository (jacalwu/gtvdb_ext) and are installed
//! through `gtv_engine::GtvExtension` by their own composition root. Keeping this
//! binary dependency-free is the proof that the core really is standalone.

use anyhow::{anyhow, Result};
use arrow::util::pretty::print_batches;
use gtv_engine::GtvContext;
use gtv_index_store::load_index;
use gtv_params::{GtvParams, QueryMode};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

const BANNER: &str = "\
gtvdb_core — temporal graph analytics platform (core build)
Zero extensions linked. SQL, GTVSeg mmap segments and state-metadata functions are ready.
Type .help for commands; any other input is executed as SQL.";

const HELP: &str = "\
Core commands
  .help                     show this help
  .quit | .exit             leave the shell
  .tables                   list registered DataFusion tables
  .extensions               SELECT name, version, objects FROM gtv_extensions()
                            (empty in this build — extensions are external)
  .segment   <name> <file>  mmap a GTVSeg v1 file as <name> (O(1), no decode)
  .store     <prefix> <dir> <node_count>
                            register every segment published by a P1 SegmentStore
  .csv       <name> <file>  register a CSV as table <name>
  .parquet   <name> <path>  register a Parquet file / directory as <name>
  .knn-index <name> <file> [labels.txt]
                            cold-open a vector index from <file> for knn(name,query,k[,label])
                            (Flat/IVF; mmap-backed, no re-decode). Hnsw → use .ann-index.
                            labels.txt: one label per line, in the corpus's original input
                            order, to restore metadata filtering without a rebuild
  .ann-index <name> <file> cold-open a vector index from <file> for ann(name,query,k[,metric])
                            (Flat/IVF/Hnsw; mmap-backed, no re-decode)
  .knn-save  <name> <store-dir>
                            persist the in-memory knn collection <name> into an
                            IndexStore at <store-dir> (prints the .gtvidx path to
                            feed back into .knn-index)
  .params                 show the effective parameter file (MEMORY_TARGET, CUDA_*,
                          DEFAULT_MODE, …)
  .accel                  show device acceleration: VRAM budget, resident cache,
                          hit ratio, fallbacks

Everything else is executed as SQL, for example:
  SELECT * FROM gtv_segments();
  SELECT * FROM gtv_segment_edges('my_segment');
  SELECT * FROM quantum_state_frontier('examples/quantum-state/ghz3.jsonl', 6, 0.05);

Extensions are not bundled here. Implement gtv_engine::GtvExtension in the
original full workspace and pass it to GtvContext::with_extensions(...).";

/// A parsed REPL line. Kept as a pure value so command handling is testable
/// without starting a DataFusion session.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Command {
    Empty,
    Help,
    Quit,
    Tables,
    Extensions,
    Segment { name: String, path: String },
    Store { prefix: String, path: String, node_count: usize },
    Csv { name: String, path: String },
    Parquet { name: String, path: String },
    KnnIndex { name: String, path: String, labels: Option<String> },
    AnnIndex { name: String, path: String },
    KnnSave { name: String, path: String },
    Params,
    Accel,
    Sql(String),
}

fn args(head: &str, rest: &[&str], n: usize, usage: &str) -> Result<Vec<String>> {
    if rest.len() != n {
        return Err(anyhow!("usage: .{head} {usage}"));
    }
    Ok(rest.iter().map(|s| (*s).to_string()).collect())
}

/// Read a per-vector label sidecar for `.knn-index`.
///
/// Format: one label per line, in the **original input order** the corpus was
/// built with (for an IVF that is *not* cell order — the index maps masks back
/// through `orig_pos`). Trailing blank lines are ignored; inner blanks are
/// legal labels (an empty label simply never matches a filter).
fn read_labels(path: &str) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow!("labels: cannot read `{path}`: {e}"))?;
    let mut labels: Vec<String> = text.lines().map(|l| l.trim().to_string()).collect();
    while labels.last().map(|l| l.is_empty()).unwrap_or(false) {
        labels.pop();
    }
    if labels.is_empty() {
        return Err(anyhow!("labels: `{path}` contains no labels"));
    }
    Ok(labels)
}

/// Split a line into a dot-command or a SQL statement.
fn parse(line: &str) -> Result<Command> {
    let line = line.trim();
    if line.is_empty() {
        return Ok(Command::Empty);
    }
    match line.to_ascii_lowercase().as_str() {
        "help" | ".help" | "?" | ".?" => return Ok(Command::Help),
        "quit" | "exit" | ".quit" | ".exit" => return Ok(Command::Quit),
        _ => {}
    }
    let Some(rest) = line.strip_prefix('.') else {
        return Ok(Command::Sql(line.to_string()));
    };
    // `.params` / `.accel` take no arguments; handle before splitting.
    if rest.trim().eq_ignore_ascii_case("params") {
        return Ok(Command::Params);
    }
    if rest.trim().eq_ignore_ascii_case("accel") {
        return Ok(Command::Accel);
    }
    let mut words = rest.split_whitespace();
    let head = words.next().unwrap_or_default().to_ascii_lowercase();
    let tail: Vec<&str> = words.collect();
    match head.as_str() {
        "tables" => Ok(Command::Tables),
        "extensions" => Ok(Command::Extensions),
        "segment" => {
            let a = args(&head, &tail, 2, "<name> <file.seg>")?;
            Ok(Command::Segment { name: a[0].clone(), path: a[1].clone() })
        }
        "csv" => {
            let a = args(&head, &tail, 2, "<name> <file.csv>")?;
            Ok(Command::Csv { name: a[0].clone(), path: a[1].clone() })
        }
        "parquet" => {
            let a = args(&head, &tail, 2, "<name> <path>")?;
            Ok(Command::Parquet { name: a[0].clone(), path: a[1].clone() })
        }
        "knn-index" => {
            if !(2..=3).contains(&tail.len()) {
                return Err(anyhow!(
                    "usage: .knn-index <name> <file.gtvidx> [labels.txt]"
                ));
            }
            Ok(Command::KnnIndex {
                name: tail[0].to_string(),
                path: tail[1].to_string(),
                labels: tail.get(2).map(|s| (*s).to_string()),
            })
        }
        "ann-index" => {
            let a = args(&head, &tail, 2, "<name> <file.gtvidx>")?;
            Ok(Command::AnnIndex { name: a[0].clone(), path: a[1].clone() })
        }
        "knn-save" => {
            let a = args(&head, &tail, 2, "<name> <store-dir>")?;
            Ok(Command::KnnSave { name: a[0].clone(), path: a[1].clone() })
        }
        "store" => {
            let a = args(&head, &tail, 3, "<prefix> <dir> <node_count>")?;
            let node_count = a[2].parse::<usize>().map_err(|_| {
                anyhow!("node_count must be a positive integer, got `{}`", a[2])
            })?;
            Ok(Command::Store { prefix: a[0].clone(), path: a[1].clone(), node_count })
        }
        _ => Err(anyhow!("unknown command `.{head}` — type .help")),
    }
}

/// List registered tables through the catalog API.
///
/// `SHOW TABLES` is a SQL-level command that requires `information_schema` to
/// be enabled in `SessionConfig`; the core context does not enable it. Walking
/// the catalogs/schemas works on every configuration and needs no SQL planner.
fn registered_tables(ctx: &GtvContext) -> Vec<String> {
    let session = ctx.session();
    let mut names = Vec::new();
    for catalog_name in session.catalog_names() {
        let Some(catalog) = session.catalog(&catalog_name) else {
            continue;
        };
        for schema_name in catalog.schema_names() {
            let Some(schema) = catalog.schema(&schema_name) else {
                continue;
            };
            names.extend(schema.table_names());
        }
    }
    names.sort();
    names.dedup();
    names
}

/// Execute a SQL statement and print every result batch.
async fn show(ctx: &GtvContext, query: &str) -> Result<()> {
    let batches = ctx.sql(query).await?;
    if batches.is_empty() {
        println!("(0 rows)");
    } else {
        print_batches(&batches)?;
    }
    Ok(())
}

/// Print the accelerator ledger: device, budget, resident blocks, fallbacks.
///
/// `ACCEL_CACHE=SHADOW` is the useful case here — the same counters are
/// maintained without a device, so the hit ratio and the upload bytes a GPU
/// would have avoided are visible before any hardware is bought.
fn print_accel(ctx: &GtvContext) {
    let t = ctx.accelerators().telemetry();
    let mib = |b: u64| format!("{} MiB", b / (1024 * 1024));
    let used_pct = if t.resident.capacity_bytes == 0 {
        0.0
    } else {
        t.resident.resident_bytes as f64 / t.resident.capacity_bytes as f64 * 100.0
    };
    println!("  ACCEL_CACHE      = {}", t.mode.as_str());
    println!(
        "  DEVICE           = {} ({})",
        t.device.name,
        t.device.kind.as_str()
    );
    println!("  VRAM_TOTAL       = {}", mib(t.device.total_bytes));
    println!(
        "  BUDGET           = {} / {} ({used_pct:.1}% used)",
        mib(t.resident.resident_bytes),
        mib(t.resident.capacity_bytes)
    );
    println!(
        "  REQUESTS         = {} ({} device, {} host, {} fallback, {} cache-declined)",
        t.requests,
        t.executed_on_device,
        t.executed_on_host,
        t.fallbacks,
        t.cache_declined
    );
    println!(
        "  RESIDENT         = {} blocks, hit ratio {:.1}% ({} hits / {} misses)",
        t.resident.blocks,
        t.resident.hit_ratio() * 100.0,
        t.resident.hits,
        t.resident.misses
    );
    println!(
        "  UPLOADS          = {} ({} transferred)",
        t.resident.uploads,
        mib(t.resident.upload_bytes)
    );
    println!(
        "  EVICTIONS        = {} ({})",
        t.resident.evictions,
        mib(t.resident.evicted_bytes)
    );
    println!(
        "  DEVICE_FREE      = {} (live probe; clamped {} times)",
        if t.resident.device_free_bytes == 0 {
            "n/a".to_string()
        } else {
            mib(t.resident.device_free_bytes)
        },
        t.resident.elastic_clamps
    );
    println!(
        "  HEAT             = hottest {:.2} | {} heat-deferred | {} promoted | {} cold-tier candidates",
        t.resident.hottest_milli as f64 / 1000.0,
        t.resident.heat_deferred,
        t.resident.promotions,
        t.resident.shadow_blocks
    );
    println!(
        "  PENDING          = {} rows / {}",
        t.resident.pending_rows,
        mib(t.resident.pending_bytes)
    );
    println!(
        "  PREFETCH         = {} hint(s) | pump {} ok / {} refused ({} block(s)/pass)",
        t.prefetch_hints,
        t.prefetch_uploads,
        t.prefetch_failures,
        ctx.accelerators().heat_pump_blocks()
    );
    println!(
        "  GPUDIRECT        = {}",
        match t.gpudirect {
            Some((direct, staged)) => format!("{direct} direct / {staged} staged"),
            None => "n/a (host backend)".to_string(),
        }
    );
    if let Some(err) = &t.last_error {
        println!("  LAST_ERROR       = {err}");
    }
    let blocks = ctx.accelerators().blocks();
    if blocks.is_empty() {
        println!("  blocks           : (none resident)");
    } else {
        println!("  blocks:");
        for b in blocks {
            println!(
                "    {:<7} {:<18} {:>10}  rows {}/{}  hits {}  misses {}  heat {:.2}",
                b.kind.as_str(),
                b.name,
                mib(b.bytes),
                b.resident_rows,
                b.rows,
                b.hits,
                b.misses,
                b.heat_milli as f64 / 1000.0
            );
        }
    }
}

/// Run one command. Returns `false` when the shell should exit.
async fn run(ctx: &GtvContext, cmd: Command) -> Result<bool> {
    match cmd {
        Command::Empty => {}
        Command::Help => println!("{HELP}"),
        Command::Quit => return Ok(false),
        Command::Tables => {
            let names = registered_tables(ctx);
            if names.is_empty() {
                println!("(no tables registered)");
            } else {
                for name in names {
                    println!("{name}");
                }
            }
        }
        Command::Extensions => {
            show(ctx, "SELECT name, version, objects FROM gtv_extensions()").await?
        }
        Command::Params => {
            for line in ctx.resolved().summary_lines() {
                println!("  {line}");
            }
        }
        Command::Accel => print_accel(ctx),
        Command::Segment { name, path } => {
            ctx.register_segment(&path, &name)?;
            println!("mmap'd GTVSeg `{path}` as `{name}` — no decode, no copy");
        }
        Command::Csv { name, path } => {
            ctx.register_csv(&path, &name)?;
            println!("registered CSV `{path}` as table `{name}`");
        }
        Command::Parquet { name, path } => {
            ctx.register_parquet(&path, &name)?;
            println!("registered Parquet `{path}` as table `{name}`");
        }
        Command::Store { prefix, path, node_count } => {
            let store = gtv_storage::store::SegmentStore::open(&path, node_count)?;
            let count = store.segments().len();
            ctx.register_segment_store(&store, &prefix)?;
            println!("registered {count} published segment(s) from `{path}` as `{prefix}_*`");
        }
        Command::KnnIndex { name, path, labels } => {
            // Carry the manifest's vector-space id into the collection so a
            // later query naming a different embedding space is refused rather
            // than answered from the wrong model.
            let (index, manifest) = gtv_index_store::load_index_with_manifest(&path)?;
            let binding = manifest
                .and_then(|m| m.vector_space_id)
                .map(|id| gtv_core::VectorSpaceBinding::new(None, id));
            let labels = labels.as_deref().map(read_labels).transpose()?;
            let with_labels = labels.is_some();
            let bound = binding.is_some();
            ctx.register_knn_bound(&name, index, labels, binding)?;
            println!(
                "cold-opened `{path}` as knn collection `{name}` (mmap-backed, no re-decode{}{})",
                if with_labels {
                    "; labels restored from sidecar"
                } else {
                    ""
                },
                if bound { "; bound to vector space" } else { "" }
            );
        }
        Command::AnnIndex { name, path } => {
            let index = load_index(&path)?;
            ctx.register_any_index(&name, index);
            println!(
                "cold-opened `{path}` as ann index `{name}` (mmap-backed, no re-decode)"
            );
        }
        Command::KnnSave { name, path } => {
            let (index, options) = ctx
                .knn_to_any_index(&name)
                .ok_or_else(|| anyhow!("knn-save: unknown collection `{name}`"))?;
            let store = gtv_index_store::IndexStore::open(&path)?;
            // Preserve the collection's vector-space id across the save so a
            // later cold-open re-binds the same space instead of re-deriving a
            // different one from the (possibly empty) default meta.
            let meta = gtv_index_store::IndexMeta {
                vector_space_id: ctx.knn_binding(&name).map(|b| b.vector_space_id),
                ..Default::default()
            };
            let v = store.save(&name, &index, &options, &meta)?;
            let container = store
                .root()
                .join(v.index_id.to_string())
                .join(format!("v{}", v.version))
                .join("index.gtvidx");
            println!(
                "saved knn collection `{name}` → `{}` (cold-open with .knn-index <name> <file>)",
                container.display()
            );
        }
        Command::Sql(query) => show(ctx, &query).await?,
    }
    Ok(true)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load the parameter file up front so LOG_LEVEL can shape logging from the
    // first line. `GTV_PARAMS_FILE` / defaults apply; a missing file is fine.
    let (params, warnings) = GtvParams::load();
    for w in &warnings {
        eprintln!("gtv: {w}");
    }
    // Honour LOG_LEVEL unless the caller already set RUST_LOG (explicit wins).
    if std::env::var_os("RUST_LOG").is_none() {
        std::env::set_var("RUST_LOG", params.log_level.as_rust_log());
    }

    // No extension registry is installed: this is the core-only build.
    let ctx = GtvContext::with_params(params.clone());

    // DEFAULT_MODE selects the REPL dialect. The core build only links SQL; a `Q`
    // request without the domain extension falls back to SQL and says so once.
    let mode = ctx.default_mode();
    if mode == QueryMode::Q {
        println!("note: DEFAULT_MODE=Q but no domain extension is linked in this build; using SQL.");
    }

    println!("{BANNER}");

    let mut rl = DefaultEditor::new()?;
    loop {
        match rl.readline("gtv> ") {
            Ok(line) => {
                let _ = rl.add_history_entry(line.as_str());
                let cmd = match parse(&line) {
                    Ok(cmd) => cmd,
                    Err(e) => {
                        eprintln!("error: {e}");
                        continue;
                    }
                };
                match run(&ctx, cmd).await {
                    Ok(true) => {}
                    Ok(false) => break,
                    Err(e) => eprintln!("error: {e}"),
                }
            }
            Err(ReadlineError::Interrupted) => continue,
            Err(ReadlineError::Eof) => break,
            Err(e) => return Err(anyhow!("readline failed: {e}")),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_text_is_sql() {
        assert_eq!(
            parse("  SELECT * FROM gtv_segments()  ").unwrap(),
            Command::Sql("SELECT * FROM gtv_segments()".into())
        );
    }

    #[test]
    fn dot_commands_parse() {
        assert_eq!(parse(".help").unwrap(), Command::Help);
        assert_eq!(parse("quit").unwrap(), Command::Quit);
        assert_eq!(parse(".tables").unwrap(), Command::Tables);
        assert_eq!(parse(".extensions").unwrap(), Command::Extensions);
        assert_eq!(parse(".params").unwrap(), Command::Params);
        assert_eq!(parse(".PARAMS").unwrap(), Command::Params);
        assert_eq!(
            parse(".segment g data/g.seg").unwrap(),
            Command::Segment { name: "g".into(), path: "data/g.seg".into() }
        );
        assert_eq!(
            parse(".store batch /tmp/seg 128").unwrap(),
            Command::Store { prefix: "batch".into(), path: "/tmp/seg".into(), node_count: 128 }
        );
        assert_eq!(
            parse(".knn-index songs data/songs.gtvidx").unwrap(),
            Command::KnnIndex {
                name: "songs".into(),
                path: "data/songs.gtvidx".into(),
                labels: None,
            }
        );
        assert_eq!(
            parse(".knn-index songs data/songs.gtvidx data/genres.txt").unwrap(),
            Command::KnnIndex {
                name: "songs".into(),
                path: "data/songs.gtvidx".into(),
                labels: Some("data/genres.txt".into()),
            }
        );
        assert!(parse(".knn-index songs").is_err());
        assert!(parse(".knn-index a b c d").is_err());
        assert_eq!(
            parse(".ann-index idx data/idx.gtvidx").unwrap(),
            Command::AnnIndex { name: "idx".into(), path: "data/idx.gtvidx".into() }
        );
        assert_eq!(
            parse(".knn-save songs data/store").unwrap(),
            Command::KnnSave { name: "songs".into(), path: "data/store".into() }
        );
    }

    #[test]
    fn table_listing_works_without_information_schema() {
        // Regression: `.tables` must not route through SHOW TABLES, which the
        // core SessionConfig rejects (information_schema is disabled).
        let ctx = GtvContext::new();
        let _ = registered_tables(&ctx);
    }

    #[test]
    fn arity_and_unknown_commands_are_errors() {
        assert!(parse(".segment only_one").is_err());
        assert!(parse(".store a b not_a_number").is_err());
        assert!(parse(".wat").is_err());
        assert_eq!(parse("   ").unwrap(), Command::Empty);
    }
}
