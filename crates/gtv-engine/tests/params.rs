//! Integration tests for the gtVdB parameters file wired into `GtvContext`.

use gtv_engine::GtvContext;
use gtv_params::{GtvParams, QueryMode};

#[test]
fn explicit_memory_target_is_carried_into_the_context() -> Result<(), Box<dyn std::error::Error>> {
    let (params, _w) = GtvParams::from_reader(
        "MEMORY_TARGET=4G\nDEFAULT_MODE=Q\nCUDA_MODE=OFF\n".as_bytes(),
    )?;
    let ctx = GtvContext::with_params(params);
    let r = ctx.resolved();
    assert!(r.memory_target_explicit, "explicit MEMORY_TARGET must be flagged");
    assert_eq!(r.memory_target_bytes, 4 * 1024 * 1024 * 1024);
    assert_eq!(r.default_mode, QueryMode::Q);
    // No GPU on this runner, but CUDA_MODE=OFF must keep the pool mounted.
    assert!(!r.cuda_enabled);
    Ok(())
}

#[test]
fn the_context_is_cloneable_with_its_params() -> Result<(), Box<dyn std::error::Error>> {
    let (params, _) = GtvParams::from_reader("CUDA_MODE=OFF\n".as_bytes())?;
    let ctx = GtvContext::with_params(params);
    let clone = ctx.clone();
    assert_eq!(
        clone.resolved().memory_target_bytes,
        ctx.resolved().memory_target_bytes
    );
    assert_eq!(clone.params().cuda_mode, ctx.params().cuda_mode);
    Ok(())
}
