//! Integration tests for the gtVdB parameters file.

use gtv_params::{
    AccelCache, CudaMode, GtvParams, MemorySpec, MemAreaSpec, QueryMode, CudaVramSpec,
};

const SAMPLE: &str = "\
# gtVdB parameters (Oracle-style, all-caps keys)
MEMORY_TARGET=16000M
MEMORY_OBJ_AREA=8000M
CUDA_MODE=TRUE
CUDA_VRAM=6000M
DEFAULT_MODE=SQL
";

#[test]
fn sample_file_parses() {
    let (p, w) = GtvParams::from_reader(SAMPLE.as_bytes()).unwrap();
    assert!(w.is_empty(), "unexpected warnings: {w:?}");
    assert_eq!(p.memory_target, MemorySpec::Bytes(16000 * 1024 * 1024));
    assert_eq!(p.memory_obj_area, MemAreaSpec::Bytes(8000 * 1024 * 1024));
    assert_eq!(p.cuda_mode, CudaMode::On);
    assert_eq!(p.cuda_vram, CudaVramSpec::Bytes(6000 * 1024 * 1024));
    assert_eq!(p.default_mode, QueryMode::Sql);
    assert!(p.is_explicit("MEMORY_TARGET"));
}

#[test]
fn comments_and_blanks_are_ignored() {
    let src = "\
# a comment
-- also a comment

MEMORY_TARGET=4096M
   # indented comment
";
    let (p, w) = GtvParams::from_reader(src.as_bytes()).unwrap();
    assert!(w.is_empty());
    assert_eq!(p.memory_target, MemorySpec::Bytes(4096 * 1024 * 1024));
}

#[test]
fn spaces_around_equals_are_tolerated() {
    let (p, _) = GtvParams::from_reader("MEMORY_TARGET = 4096M\n".as_bytes()).unwrap();
    assert_eq!(p.memory_target, MemorySpec::Bytes(4096 * 1024 * 1024));
}

#[test]
fn whitespace_separated_form_is_supported() {
    let (p, _) = GtvParams::from_reader("MEMORY_TARGET 4096M\n".as_bytes()).unwrap();
    assert_eq!(p.memory_target, MemorySpec::Bytes(4096 * 1024 * 1024));
}

#[test]
fn keys_are_case_insensitive() {
    let (p, _) = GtvParams::from_reader("memory_target=1024M\n".as_bytes()).unwrap();
    assert_eq!(p.memory_target, MemorySpec::Bytes(1024 * 1024 * 1024));
}

#[test]
fn unknown_keys_warn_but_do_not_fail() {
    let (p, w) = GtvParams::from_reader("MEMORY_TARGET=1024M\nBOGUS_KEY=1\n".as_bytes()).unwrap();
    assert_eq!(p.memory_target, MemorySpec::Bytes(1024 * 1024 * 1024));
    assert_eq!(w.len(), 1);
    assert_eq!(w[0].key.as_deref(), Some("BOGUS_KEY"));
}

#[test]
fn binary_unit_suffixes() {
    let (p, _) = GtvParams::from_reader("MEMORY_TARGET=1G\n".as_bytes()).unwrap();
    assert_eq!(p.memory_target, MemorySpec::Bytes(1024 * 1024 * 1024));
    let (p, _) = GtvParams::from_reader("MEMORY_TARGET=2T\n".as_bytes()).unwrap();
    assert_eq!(p.memory_target, MemorySpec::Bytes(2 * 1024 * 1024 * 1024 * 1024));
}

#[test]
fn fractional_system_reference() {
    let (p, _) = GtvParams::from_reader("MEMORY_TARGET=SYSTEM:0.5\n".as_bytes()).unwrap();
    assert_eq!(p.memory_target, MemorySpec::SystemFraction(0.5));
    // Resolves to ~half of system memory at resolve() time.
    let r = p.resolve(None).unwrap();
    assert!(!r.memory_target_explicit);
    assert!(r.memory_target_bytes > 0);
}

#[test]
fn fractional_target_reference_for_obj_area() {
    let (p, _) = GtvParams::from_reader(
        "MEMORY_TARGET=1000M\nMEMORY_OBJ_AREA=TARGET:0.25\n".as_bytes(),
    )
    .unwrap();
    let r = p.resolve(None).unwrap();
    assert_eq!(r.memory_obj_area_bytes, 250 * 1024 * 1024);
}

#[test]
fn defaults_with_no_file() {
    let (p, w) = GtvParams::from_reader("".as_bytes()).unwrap();
    assert!(w.is_empty());
    assert_eq!(p.memory_target, MemorySpec::SystemFraction(0.5));
    assert_eq!(p.memory_obj_area, MemAreaSpec::TargetFraction(0.5));
    assert_eq!(p.cuda_mode, CudaMode::Auto);
    assert_eq!(p.cuda_vram, CudaVramSpec::DeviceFraction(0.8));
    assert_eq!(p.default_mode, QueryMode::Sql);
}

#[test]
fn resolve_half_system_memory_when_auto() {
    let (p, _) = GtvParams::from_reader("".as_bytes()).unwrap();
    let r = p.resolve(None).unwrap();
    assert!(!r.memory_target_explicit);
    assert!(r.memory_target_bytes > 0);
}

#[test]
fn obj_area_exceeding_target_is_an_error() {
    let (p, _) = GtvParams::from_reader("MEMORY_TARGET=100M\nMEMORY_OBJ_AREA=200M\n".as_bytes()).unwrap();
    assert!(matches!(
        p.resolve(None),
        Err(gtv_params::ParamError::ObjAreaExceedsTarget { .. })
    ));
}

#[test]
fn cuda_auto_without_device_stays_off() {
    let (p, _) = GtvParams::from_reader("CUDA_MODE=AUTO\n".as_bytes()).unwrap();
    let r = p.resolve(None).unwrap();
    assert!(!r.cuda_enabled);
    assert_eq!(r.cuda_vram_bytes, 0);
}

#[test]
fn cuda_on_without_device_is_an_error() {
    let (p, _) = GtvParams::from_reader("CUDA_MODE=ON\n".as_bytes()).unwrap();
    assert!(matches!(
        p.resolve(None),
        Err(gtv_params::ParamError::CudaNoDevice)
    ));
}

#[test]
fn cuda_on_with_device_total_resolves_vram_fraction() {
    // 10 GiB device; default 80% => 8 GiB.
    let device = 10u64 * 1024 * 1024 * 1024;
    let (p, _) = GtvParams::from_reader("CUDA_MODE=ON\n".as_bytes()).unwrap();
    let r = p.resolve(Some(device)).unwrap();
    assert!(r.cuda_enabled);
    assert_eq!(r.cuda_vram_bytes, 8 * 1024 * 1024 * 1024);
}

#[test]
fn cuda_vram_fraction_capped_at_device_total() {
    let device = 4u64 * 1024 * 1024 * 1024; // 4 GiB
    let (p, _) = GtvParams::from_reader("CUDA_MODE=AUTO\nCUDA_VRAM=DEVICE:0.8\n".as_bytes()).unwrap();
    let r = p.resolve(Some(device)).unwrap();
    assert!(r.cuda_enabled);
    assert_eq!(r.cuda_vram_bytes, 4 * 1024 * 1024 * 1024 * 8 / 10);
}

#[test]
fn telemetry_and_paths_and_worker_threads() {
    let (p, _) = GtvParams::from_reader(
        "TELEMETRY=TRUE\nSPILL_DIR=/tmp/x\nWORKER_THREADS=AUTO\nDATA_DIR=/data\nCATALOG_DIR=/cat\n"
            .as_bytes(),
    )
    .unwrap();
    assert!(p.telemetry);
    assert_eq!(p.spill_dir, Some(std::path::PathBuf::from("/tmp/x")));
    assert_eq!(p.worker_threads, None);
    assert_eq!(p.data_dir, std::path::PathBuf::from("/data"));
    assert_eq!(p.catalog_dir, std::path::PathBuf::from("/cat"));
}

#[test]
fn worker_threads_explicit_number() {
    let (p, _) = GtvParams::from_reader("WORKER_THREADS=8\n".as_bytes()).unwrap();
    assert_eq!(p.worker_threads, Some(8));
}

#[test]
fn summary_lines_are_produced() {
    let (p, _) = GtvParams::from_reader(SAMPLE.as_bytes()).unwrap();
    // SAMPLE sets CUDA_MODE=TRUE, so a device total must be supplied.
    let r = p.resolve(Some(8 * 1024 * 1024 * 1024)).unwrap();
    let lines = r.summary_lines();
    assert!(lines.iter().any(|l| l.starts_with("MEMORY_TARGET")));
    assert!(lines.iter().any(|l| l.starts_with("CUDA_MODE")));
    assert!(lines.iter().any(|l| l.starts_with("DEFAULT_MODE")));
    assert!(lines.iter().any(|l| l.starts_with("ACCEL_CACHE")));
}

#[test]
fn accel_cache_defaults_to_on_with_a_quarter_budget_threshold() {
    let (p, _) = GtvParams::from_reader("CUDA_VRAM=8000M\n".as_bytes()).unwrap();
    assert_eq!(p.accel_cache, AccelCache::On);
    let r = p.resolve(Some(8 * 1024 * 1024 * 1024)).unwrap();
    assert_eq!(r.accel_cache, AccelCache::On);
    // 25% of the configured 8000 MiB budget, matching PG-Strom's sync-threshold
    // default of a quarter of the buffer.
    assert_eq!(
        r.accel_sync_threshold_bytes,
        (8000u64 * 1024 * 1024) / 4
    );
    assert_eq!(r.accel_sync_interval.as_secs(), 5);
}

#[test]
fn accel_keys_are_parsed_and_explicit() {
    let (p, _) = GtvParams::from_reader(
        "CUDA_MODE=OFF\nACCEL_CACHE=SHADOW\nACCEL_SYNC_THRESHOLD=512M\nACCEL_SYNC_INTERVAL=0\nACCEL_HEAT_HALF_LIFE=0\n"
            .as_bytes(),
    )
    .unwrap();
    assert_eq!(p.accel_cache, AccelCache::Shadow);
    assert_eq!(p.accel_sync_threshold, Some(512 * 1024 * 1024));
    assert_eq!(p.accel_sync_interval_secs, 0);
    assert_eq!(p.accel_heat_half_life_secs, 0);
    assert!(p.is_explicit("ACCEL_CACHE"));
    assert!(p.is_explicit("ACCEL_HEAT_HALF_LIFE"));
    let r = p.resolve(None).unwrap();
    assert!(r.accel_heat_half_life.is_zero(), "0 restores plain LRU");
    assert!(r
        .summary_lines()
        .iter()
        .any(|l| l.contains("ACCEL_HEAT") && l.contains("LRU")));
}

#[test]
fn heat_half_life_defaults_to_30_seconds() {
    let (p, _) = GtvParams::from_reader("CUDA_MODE=OFF\n".as_bytes()).unwrap();
    let r = p.resolve(None).unwrap();
    assert_eq!(r.accel_heat_half_life.as_secs(), 30);
    assert!(!p.is_explicit("ACCEL_HEAT_HALF_LIFE"));
}

#[test]
fn prefetch_and_gpudirect_keys_are_parsed_and_explicit() {
    let (p, _) = GtvParams::from_reader(
        "CUDA_MODE=OFF\nACCEL_PREFETCH=ON\nACCEL_PREFETCH_BLOCKS=2\nACCEL_GPUDIRECT=TRUE\nACCEL_GPUDIRECT_THRESHOLD=256M\n"
            .as_bytes(),
    )
    .unwrap();
    assert!(p.accel_prefetch);
    assert_eq!(p.accel_prefetch_blocks, 2);
    assert!(p.accel_gpudirect);
    assert_eq!(p.accel_gpudirect_threshold, 256 * 1024 * 1024);
    assert!(p.is_explicit("ACCEL_PREFETCH"));
    assert!(p.is_explicit("ACCEL_GPUDIRECT_THRESHOLD"));

    // CUDA_MODE=OFF: GPUDirect is clamped off, prefetch survives (the pump also
    // works against the host-side shadow ledger).
    let r = p.resolve(None).unwrap();
    assert!(!r.accel_gpudirect);
    assert!(r.accel_prefetch);
    assert_eq!(r.accel_prefetch_blocks, 2);
    let lines = r.summary_lines();
    assert!(lines.iter().any(|l| l.starts_with("ACCEL_PREFETCH")));
    assert!(lines.iter().any(|l| l.starts_with("ACCEL_GPUDIRECT")));
}

#[test]
fn accel_cache_is_off_when_cuda_is_off_unless_shadow_is_asked_for() {
    let (p, _) = GtvParams::from_reader("CUDA_MODE=OFF\n".as_bytes()).unwrap();
    let r = p.resolve(None).unwrap();
    assert_eq!(r.accel_cache, AccelCache::Off);

    // SHADOW is the calibration mode: it survives CUDA_MODE=OFF so a deployment
    // can measure what a device cache would save before buying hardware.
    let (p, _) = GtvParams::from_reader("CUDA_MODE=OFF\nACCEL_CACHE=SHADOW\n".as_bytes()).unwrap();
    let r = p.resolve(None).unwrap();
    assert_eq!(r.accel_cache, AccelCache::Shadow);
    assert!(!r.accel_cache.uses_device());
    assert!(r.accel_cache.tracks_residency());
}

#[test]
fn accel_cache_rejects_an_unknown_mode() {
    assert!(GtvParams::from_reader("ACCEL_CACHE=SOMETIMES\n".as_bytes()).is_err());
}
