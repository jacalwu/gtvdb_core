//! `gtv` — the minimal, zero-extension REPL shipped with **gtvdb_core**.
//!
//! This binary deliberately depends only on platform crates. Domain workloads
//! (banking / HFT / quantum compute / crypto / q) are NOT linked in: they live
//! in the original full workspace (jacalwu/gtvdb, gtv_ext_lib/) and are installed through
//! `gtv_engine::GtvExtension` by their own composition root. Keeping this
//! binary dependency-free is the proof that the core really is standalone.

use anyhow::{anyhow, Result};
use arrow::util::pretty::print_batches;
use gtv_engine::GtvContext;
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
    Sql(String),
}

fn args(head: &str, rest: &[&str], n: usize, usage: &str) -> Result<Vec<String>> {
    if rest.len() != n {
        return Err(anyhow!("usage: .{head} {usage}"));
    }
    Ok(rest.iter().map(|s| (*s).to_string()).collect())
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
        Command::Sql(query) => show(ctx, &query).await?,
    }
    Ok(true)
}

#[tokio::main]
async fn main() -> Result<()> {
    // No extension registry is installed: this is the core-only build.
    let ctx = GtvContext::new();
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
        assert_eq!(
            parse(".segment g data/g.seg").unwrap(),
            Command::Segment { name: "g".into(), path: "data/g.seg".into() }
        );
        assert_eq!(
            parse(".store batch /tmp/seg 128").unwrap(),
            Command::Store { prefix: "batch".into(), path: "/tmp/seg".into(), node_count: 128 }
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
