//! Parameter-file integration: the resolved values actually reach the context.

use gtv_engine::GtvContext;
use gtv_params::{CudaMode, GtvParams, MemorySpec, QueryMode};

#[test]
fn explicit_memory_target_is_carried_into_the_context() {
    let params = GtvParams {
        memory_target: MemorySpec::Bytes(4 << 30),
        cuda_mode: CudaMode::Off,
        default_mode: QueryMode::Q,
        ..GtvParams::default()
    };
    let ctx = GtvContext::with_params(params);
    // No device on the test machine is assumed; resolution still succeeds because
    // CUDA_MODE is Off, so MEMORY_TARGET is the only thing that must be exact.
    assert_eq!(ctx.resolved().memory_target_bytes, 4 << 30);
    assert_eq!(ctx.default_mode(), QueryMode::Q);
    assert!(!ctx.resolved().cuda_enabled);
}

#[test]
fn the_context_is_cloneable_with_its_backend_registry() {
    // Guards the Arc<AcceleratorRegistry> choice: GtvContext must stay Clone.
    let ctx = GtvContext::new();
    let _clone = ctx.clone();
    // And the resolved defaults are present (half system RAM > 0 on any box).
    assert!(ctx.resolved().memory_target_bytes > 0);
}
