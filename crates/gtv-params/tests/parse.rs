//! Tests for the gtVdB parameter file: parsing, defaults, resolution, and the
//! deployment-specific edge cases (fractions, AUTO semantics, validation).

use gtv_params::*;
use std::io::Cursor;

fn parse(s: &str) -> (GtvParams, Warnings) {
    GtvParams::from_reader(Cursor::new(s)).expect("parse")
}

#[test]
fn the_sample_file_parses_and_defaults_are_reasonable() {
    let src = "\
# how much host RAM gtVdB may use
MEMORY_TARGET=16000M
MEMORY_OBJ_AREA=8000M
CUDA_MODE=TRUE
CUDA_VRAM=6000M
DEFAULT_MODE=SQL
";
    let (p, w) = parse(src);
    assert!(w.is_empty(), "unexpected warnings: {w:?}");
    assert_eq!(p.memory_target, MemorySpec::Bytes(16000 << 20));
    assert_eq!(p.memory_obj_area, MemAreaSpec::Bytes(8000 << 20));
    assert_eq!(p.cuda_mode, CudaMode::On);
    assert_eq!(p.cuda_vram, MemorySpec::Bytes(6000 << 20));
    assert_eq!(p.default_mode, QueryMode::Sql);
}

#[test]
fn comments_blanks_and_space_separators_are_accepted() {
    let src = "\
# a comment
   # indented comment

MEMORY_TARGET 4096M
-- sql style comment
DEFAULT_MODE Q
";
    let (p, w) = parse(src);
    assert!(w.is_empty());
    assert_eq!(p.memory_target, MemorySpec::Bytes(4096 << 20));
    assert_eq!(p.default_mode, QueryMode::Q);
}

#[test]
fn keys_are_case_insensitive_but_stored_upper() {
    let (p, w) = parse("memory_target=2048M\ncuda_mode=auto\n");
    assert!(w.is_empty());
    assert_eq!(p.memory_target, MemorySpec::Bytes(2048 << 20));
    assert_eq!(p.cuda_mode, CudaMode::Auto);
}

#[test]
fn unknown_keys_warn_and_do_not_crash() {
    let (p, w) = parse("MEMORY_TARGET=1024M\nTOTALLY_MADE_UP=yes\n");
    assert_eq!(w.len(), 1);
    assert_eq!(w[0].key, "TOTALLY_MADE_UP");
    // The known key still parsed.
    assert_eq!(p.memory_target, MemorySpec::Bytes(1024 << 20));
}

#[test]
fn size_units_are_binary_and_the_trailing_b_is_optional() {
    // Exercise the size parser through the public MEMORY_TARGET path.
    let (p, _) = parse("MEMORY_TARGET=1K\n");
    assert_eq!(p.memory_target, MemorySpec::Bytes(1024));
    let (p, _) = parse("MEMORY_TARGET=1KB\n");
    assert_eq!(p.memory_target, MemorySpec::Bytes(1024));
    let (p, _) = parse("MEMORY_TARGET=16M\n");
    assert_eq!(p.memory_target, MemorySpec::Bytes(16 << 20));
    let (p, _) = parse("MEMORY_TARGET=8G\n");
    assert_eq!(p.memory_target, MemorySpec::Bytes(8 << 30));
    let (p, _) = parse("MEMORY_TARGET=2T\n");
    assert_eq!(p.memory_target, MemorySpec::Bytes(2u64 << 40));
    assert!(matches!(
        GtvParams::from_reader(Cursor::new("MEMORY_TARGET=16X\n")),
        Err(ParamError::InvalidValue { .. })
    ));
    assert!(matches!(
        GtvParams::from_reader(Cursor::new("MEMORY_TARGET=abc\n")),
        Err(ParamError::InvalidValue { .. })
    ));
}

#[test]
fn fractions_are_accepted_for_target_and_area() {
    let (p, w) = parse("MEMORY_TARGET=SYSTEM:0.7\nMEMORY_OBJ_AREA=TARGET:0.25\n");
    assert!(w.is_empty());
    assert_eq!(p.memory_target, MemorySpec::SystemFraction(0.7));
    assert_eq!(p.memory_obj_area, MemAreaSpec::TargetFraction(0.25));
}

#[test]
fn defaults_with_no_file_are_coherent() {
    let p = GtvParams::default();
    assert_eq!(p.memory_target, MemorySpec::SystemFraction(0.5));
    assert_eq!(p.memory_obj_area, MemAreaSpec::TargetFraction(0.5));
    assert_eq!(p.cuda_mode, CudaMode::Auto);
    assert_eq!(p.cuda_vram, MemorySpec::SystemFraction(0.8));
    assert_eq!(p.default_mode, QueryMode::Sql);
    assert_eq!(p.log_level, Level::Info);
    assert!(!p.telemetry);
}

#[test]
fn resolve_applies_system_fraction_and_obj_area() {
    // device_total irrelevant here; memory_target is a system fraction.
    let p = GtvParams {
        memory_target: MemorySpec::SystemFraction(0.5),
        memory_obj_area: MemAreaSpec::TargetFraction(0.5),
        ..GtvParams::default()
    };
    let r = p.resolve(None).unwrap();
    // system_memory_bytes is real on any CI box; assert the relationship instead
    // of an exact number.
    assert!(r.memory_target_bytes > 0);
    assert_eq!(r.memory_obj_area_bytes, r.memory_target_bytes / 2);
}

#[test]
fn obj_area_cannot_exceed_target() {
    let p = GtvParams {
        memory_target: MemorySpec::Bytes(1000),
        memory_obj_area: MemAreaSpec::Bytes(2000),
        ..GtvParams::default()
    };
    let err = p.resolve(None).unwrap_err();
    assert!(matches!(err, ParamError::ObjAreaExceedsTarget { .. }));
}

#[test]
fn cuda_auto_off_without_device() {
    let p = GtvParams {
        cuda_mode: CudaMode::Auto,
        ..GtvParams::default()
    };
    let r = p.resolve(None).unwrap();
    assert!(!r.cuda_enabled);
    assert_eq!(r.cuda_vram_bytes, 0);
}

#[test]
fn cuda_on_without_device_is_an_error() {
    let p = GtvParams {
        cuda_mode: CudaMode::On,
        ..GtvParams::default()
    };
    assert!(matches!(p.resolve(None), Err(ParamError::CudaNoDevice)));
}

#[test]
fn cuda_fraction_resolves_against_device_total() {
    let p = GtvParams {
        cuda_mode: CudaMode::On,
        cuda_vram: MemorySpec::SystemFraction(0.8),
        ..GtvParams::default()
    };
    // 10 GiB device -> 8 GiB budget.
    let r = p.resolve(Some(10 << 30)).unwrap();
    assert!(r.cuda_enabled);
    assert_eq!(r.cuda_vram_bytes, 8 << 30);
}

#[test]
fn cuda_vram_is_capped_to_device_total() {
    let p = GtvParams {
        cuda_mode: CudaMode::On,
        cuda_vram: MemorySpec::Bytes(100 << 30), // asks for 100 GiB
        ..GtvParams::default()
    };
    let r = p.resolve(Some(8 << 30)).unwrap();
    assert_eq!(r.cuda_vram_bytes, 8 << 30);
}

#[test]
fn telemetry_and_paths_round_trip() {
    let (p, w) = parse(
        "TELEMETRY=ON\nDATA_DIR=/var/gtvdb\nCATALOG_DIR=/var/gtvdb/cat\nSPILL_DIR=/tmp/spill\nWORKER_THREADS=4\nLOG_LEVEL=DEBUG\n",
    );
    assert!(w.is_empty());
    assert!(p.telemetry);
    assert_eq!(p.data_dir, Some(std::path::PathBuf::from("/var/gtvdb")));
    assert_eq!(p.catalog_dir, Some(std::path::PathBuf::from("/var/gtvdb/cat")));
    assert_eq!(p.spill_dir, Some(std::path::PathBuf::from("/tmp/spill")));
    assert_eq!(p.worker_threads, Some(4));
    assert_eq!(p.log_level, Level::Debug);
}

#[test]
fn worker_threads_auto_and_zero_mean_auto() {
    let (p, _) = parse("WORKER_THREADS=AUTO\n");
    assert_eq!(p.worker_threads, None);
    let (p, _) = parse("WORKER_THREADS=0\n");
    assert_eq!(p.worker_threads, None);
}

#[test]
fn summary_lines_are_stable_and_complete() {
    let p = GtvParams {
        cuda_mode: CudaMode::On,
        cuda_vram: MemorySpec::Bytes(6000 << 20),
        ..GtvParams::default()
    };
    let r = p.resolve(Some(8 << 30)).unwrap();
    let lines = r.summary_lines();
    let joined = lines.join("\n");
    assert!(joined.contains("MEMORY_TARGET"));
    assert!(joined.contains("CUDA_VRAM"));
    assert!(joined.contains("DEFAULT_MODE"));
    assert!(joined.contains("TELEMETRY"));
    // 6000 MiB resolved, device 8 GiB.
    assert!(joined.contains("6000 MiB"));
}
