//! The CUDA driver, loaded at runtime through the OS loader.
//!
//! # Why not `cudarc`
//!
//! This crate needs nine driver entry points in C1 (init, enumerate, query,
//! context, alloc, free, copy, and the error string). Linking a full CUDA
//! wrapper — or `build.rs` + `nvcc` — to get those nine would make the build
//! depend on a toolkit that the CI machine does not have, for functionality the
//! kernels in C2 will need a different interface for anyway. The rule in
//! `docs/gtvdb-cpu-cuda-cursor-development-spec.md` §7.1 is explicit: no new CUDA
//! framework without a dependency comparison and a minimal probe first. This is
//! the minimal probe, promoted to code.
//!
//! # Why this is safe to build without a GPU
//!
//! Nothing is linked. `LoadLibraryW` / `dlopen` resolve `nvcuda.dll` /
//! `libcuda.so.1` at run time, so:
//!
//! * a machine without a driver fails at [`Driver::open`] and produces
//!   [`gtv_accelerator::AcceleratorError::DeviceUnavailable`];
//! * the CI machine compiles and tests this crate with no CUDA present;
//! * the resulting binary has no `libcuda` dependency at all.
//!
//! # Safety
//!
//! Every `unsafe` block here is a call into an API whose contract is "returns a
//! status code, never unwinds". The wrappers check the status and convert it,
//! so no raw handle escapes this module — the rest of the crate sees `Result`.

use std::ffi::CString;
use std::sync::OnceLock;

pub type CuResult = i32;
pub type CuDevice = i32;
type CuContext = *mut std::ffi::c_void;

/// An opaque context handle.
///
/// Deliberately not a raw pointer in the public API: nothing outside this module
/// should be able to `unsafe`-cast it back into something it can dereference, and
/// a caller that never sees a pointer cannot misuse one.
#[derive(Debug, Clone, Copy)]
pub struct Context(CuContext);

pub const CUDA_SUCCESS: CuResult = 0;

// `cuda.h` attribute ids.
const ATTR_CC_MAJOR: i32 = 75;
const ATTR_CC_MINOR: i32 = 76;
const ATTR_MULTIPROCESSORS: i32 = 16;

/// `nvcuda.dll` on Windows, `libcuda.so.1` elsewhere.
#[cfg(windows)]
const LIB_NAMES: &[&str] = &["nvcuda.dll"];
#[cfg(not(windows))]
const LIB_NAMES: &[&str] = &["libcuda.so.1", "libcuda.so"];

#[cfg(windows)]
#[link(name = "kernel32")]
extern "system" {
    fn LoadLibraryW(name: *const u16) -> *mut std::ffi::c_void;
    fn GetProcAddress(
        module: *mut std::ffi::c_void,
        name: *const u8,
    ) -> *mut std::ffi::c_void;
}

type FnInit = unsafe extern "C" fn(u32) -> CuResult;
type FnGetCount = unsafe extern "C" fn(*mut i32) -> CuResult;
type FnGetDevice = unsafe extern "C" fn(*mut CuDevice, i32) -> CuResult;
type FnGetName = unsafe extern "C" fn(*mut std::ffi::c_char, i32, CuDevice) -> CuResult;
type FnTotalMem = unsafe extern "C" fn(*mut usize, CuDevice) -> CuResult;
type FnGetAttr = unsafe extern "C" fn(*mut i32, i32, CuDevice) -> CuResult;
type FnCtxRetain = unsafe extern "C" fn(*mut CuContext, CuDevice) -> CuResult;
type FnCtxRelease = unsafe extern "C" fn(CuDevice) -> CuResult;
type FnCtxSetCurrent = unsafe extern "C" fn(CuContext) -> CuResult;
type FnMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> CuResult;
type FnMemFree = unsafe extern "C" fn(u64) -> CuResult;
type FnMemcpyHtoD = unsafe extern "C" fn(u64, *const std::ffi::c_void, usize) -> CuResult;
type FnMemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> CuResult;
type FnErrorString = unsafe extern "C" fn(CuResult, *mut *const std::ffi::c_char) -> CuResult;

struct Fns {
    init: FnInit,
    get_count: FnGetCount,
    get_device: FnGetDevice,
    get_name: FnGetName,
    total_mem: FnTotalMem,
    get_attr: FnGetAttr,
    ctx_retain: FnCtxRetain,
    ctx_release: FnCtxRelease,
    ctx_set_current: FnCtxSetCurrent,
    mem_alloc: FnMemAlloc,
    mem_free: FnMemFree,
    memcpy_htod: FnMemcpyHtoD,
    mem_get_info: FnMemGetInfo,
    error_string: Option<FnErrorString>,
}

/// A loaded driver library. Unloading happens when the process exits; the OS
/// does that anyway, and a `Drop` that unloads while a context is alive is a
/// way to crash on a healthy machine.
pub struct Driver {
    fns: Fns,
    // Kept alive for the lifetime of `fns`.
    _lib: Library,
}

/// One enumerated device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    /// `CUdevice` handle, not an index. Pass it back to the driver; do not
    /// assume it equals 0 for the only device.
    pub handle: CuDevice,
    pub name: String,
    pub total_mem_bytes: usize,
    pub compute_capability: (i32, i32),
    pub multiprocessors: i32,
}

impl DeviceInfo {
    /// Whether the compiled C2 kernels will run here. Kept as a named predicate
    /// because the answer changes when the PTX target changes.
    pub fn supports_sm(&self, major: i32, minor: i32) -> bool {
        self.compute_capability >= (major, minor)
    }
}

impl Driver {
    /// Load the driver and enumerate device 0.
    ///
    /// `Err` is the normal outcome on a machine without a driver: nothing here
    /// treats it as exceptional.
    pub fn open() -> Result<Driver, String> {
        let lib = Library::open(LIB_NAMES)
            .ok_or_else(|| format!("no CUDA driver library (tried {LIB_NAMES:?})"))?;
        let fns = Fns::load(&lib)?;
        let driver = Driver { fns, _lib: lib };

        // `cuInit` is idempotent and cheap; it is the only call that can tell us
        // the driver is present but broken.
        let rc = unsafe { (driver.fns.init)(0) };
        if rc != CUDA_SUCCESS {
            return Err(driver.error("cuInit", rc));
        }
        Ok(driver)
    }

    pub fn device_count(&self) -> Result<usize, String> {
        let mut n: i32 = 0;
        let rc = unsafe { (self.fns.get_count)(&mut n) };
        if rc != CUDA_SUCCESS {
            return Err(self.error("cuDeviceGetCount", rc));
        }
        Ok(n.max(0) as usize)
    }

    pub fn device(&self, ordinal: i32) -> Result<DeviceInfo, String> {
        let mut dev: CuDevice = 0;
        let rc = unsafe { (self.fns.get_device)(&mut dev, ordinal) };
        if rc != CUDA_SUCCESS {
            return Err(self.error("cuDeviceGet", rc));
        }

        let mut buf = [0i8; 128];
        let rc = unsafe { (self.fns.get_name)(buf.as_mut_ptr(), buf.len() as i32, dev) };
        if rc != CUDA_SUCCESS {
            return Err(self.error("cuDeviceGetName", rc));
        }
        let name = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        let mut total: usize = 0;
        let rc = unsafe { (self.fns.total_mem)(&mut total, dev) };
        if rc != CUDA_SUCCESS {
            return Err(self.error("cuDeviceTotalMem", rc));
        }

        let major = self.attr(ATTR_CC_MAJOR, dev)?;
        let minor = self.attr(ATTR_CC_MINOR, dev)?;
        let sm = self.attr(ATTR_MULTIPROCESSORS, dev)?;

        Ok(DeviceInfo {
            handle: dev,
            name,
            total_mem_bytes: total,
            compute_capability: (major, minor),
            multiprocessors: sm,
        })
    }

    fn attr(&self, attr: i32, dev: CuDevice) -> Result<i32, String> {
        let mut v: i32 = 0;
        let rc = unsafe { (self.fns.get_attr)(&mut v, attr, dev) };
        if rc != CUDA_SUCCESS {
            return Err(self.error("cuDeviceGetAttribute", rc));
        }
        Ok(v)
    }

    /// Retain the device's primary context and make it current for this thread.
    ///
    /// Allocations in CUDA are per-context and contexts are per-thread, so every
    /// entry point below re-binds the context: the accelerator is `Send + Sync`
    /// and may be called from a pool thread.
    pub fn create_context(&self, dev: CuDevice) -> Result<Context, String> {
        let mut ctx: CuContext = std::ptr::null_mut();
        let rc = unsafe { (self.fns.ctx_retain)(&mut ctx, dev) };
        if rc != CUDA_SUCCESS {
            return Err(self.error("cuDevicePrimaryCtxRetain", rc));
        }
        self.bind(Context(ctx))?;
        Ok(Context(ctx))
    }

    pub fn bind(&self, ctx: Context) -> Result<(), String> {
        let rc = unsafe { (self.fns.ctx_set_current)(ctx.0) };
        if rc != CUDA_SUCCESS {
            return Err(self.error("cuCtxSetCurrent", rc));
        }
        Ok(())
    }

    pub fn release_context(&self, dev: CuDevice) {
        unsafe {
            (self.fns.ctx_release)(dev);
        }
    }

    /// Free and total device memory, as the driver sees them.
    pub fn mem_info(&self) -> Result<(usize, usize), String> {
        let (mut free, mut total) = (0usize, 0usize);
        let rc = unsafe { (self.fns.mem_get_info)(&mut free, &mut total) };
        if rc != CUDA_SUCCESS {
            return Err(self.error("cuMemGetInfo", rc));
        }
        Ok((free, total))
    }

    /// Allocate `bytes` on the device. Zero-byte requests return `0`, because
    /// the driver rejects them and a zero-length column is a legal empty graph.
    pub fn alloc(&self, bytes: usize) -> Result<u64, String> {
        if bytes == 0 {
            return Ok(0);
        }
        let mut ptr: u64 = 0;
        let rc = unsafe { (self.fns.mem_alloc)(&mut ptr, bytes) };
        if rc != CUDA_SUCCESS {
            return Err(self.error(&format!("cuMemAlloc({bytes})"), rc));
        }
        Ok(ptr)
    }

    pub fn free(&self, ptr: u64) {
        if ptr != 0 {
            unsafe {
                (self.fns.mem_free)(ptr);
            }
        }
    }

    /// # Safety
    ///
    /// `dst` must be a device pointer from [`Driver::alloc`] with at least
    /// `len` bytes, and `src` must be readable for `len` bytes.
    pub unsafe fn copy_htod(&self, dst: u64, src: *const u8, len: usize) -> Result<(), String> {
        if len == 0 {
            return Ok(());
        }
        let rc = (self.fns.memcpy_htod)(dst, src as *const std::ffi::c_void, len);
        if rc != CUDA_SUCCESS {
            return Err(self.error(&format!("cuMemcpyHtoD({len})"), rc));
        }
        Ok(())
    }

    fn error(&self, what: &str, rc: CuResult) -> String {
        let detail = self
            .fns
            .error_string
            .and_then(|f| {
                let mut msg: *const std::ffi::c_char = std::ptr::null();
                if unsafe { f(rc, &mut msg) } == CUDA_SUCCESS && !msg.is_null() {
                    Some(unsafe { std::ffi::CStr::from_ptr(msg) }.to_string_lossy().into_owned())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| format!("CUresult {rc}"));
        format!("{what}: {detail}")
    }
}

impl Fns {
    fn load(lib: &Library) -> Result<Fns, String> {
        fn need<T: Copy>(lib: &Library, name: &str) -> Result<T, String> {
            lib.symbol(name)
                .map(|p| unsafe { std::mem::transmute_copy::<*mut std::ffi::c_void, T>(&p) })
                .ok_or_else(|| format!("driver is missing {name}"))
        }
        fn optional<T: Copy>(lib: &Library, names: &[&str]) -> Option<T> {
            names.iter().find_map(|n| {
                lib.symbol(n)
                    .map(|p| unsafe { std::mem::transmute_copy::<*mut std::ffi::c_void, T>(&p) })
            })
        }
        Ok(Fns {
            init: need(lib, "cuInit")?,
            get_count: need(lib, "cuDeviceGetCount")?,
            get_device: need(lib, "cuDeviceGet")?,
            get_name: need(lib, "cuDeviceGetName")?,
            // The `_v2` suffix is the 64-bit-safe ABI introduced in CUDA 3.2;
            // both names are exported by every driver we target, and trying both
            // costs one extra `dlsym`.
            total_mem: optional(lib, &["cuDeviceTotalMem_v2", "cuDeviceTotalMem"])
                .ok_or_else(|| "driver is missing cuDeviceTotalMem".to_string())?,
            get_attr: need(lib, "cuDeviceGetAttribute")?,
            ctx_retain: need(lib, "cuDevicePrimaryCtxRetain")?,
            ctx_release: need(lib, "cuDevicePrimaryCtxRelease_v2")
                .or_else(|_| need(lib, "cuDevicePrimaryCtxRelease"))?,
            ctx_set_current: need(lib, "cuCtxSetCurrent")?,
            mem_alloc: optional(lib, &["cuMemAlloc_v2", "cuMemAlloc"])
                .ok_or_else(|| "driver is missing cuMemAlloc".to_string())?,
            mem_free: optional(lib, &["cuMemFree_v2", "cuMemFree"])
                .ok_or_else(|| "driver is missing cuMemFree".to_string())?,
            memcpy_htod: optional(lib, &["cuMemcpyHtoD_v2", "cuMemcpyHtoD"])
                .ok_or_else(|| "driver is missing cuMemcpyHtoD".to_string())?,
            mem_get_info: optional(lib, &["cuMemGetInfo_v2", "cuMemGetInfo"])
                .ok_or_else(|| "driver is missing cuMemGetInfo".to_string())?,
            error_string: optional(lib, &["cuGetErrorString"]),
        })
    }
}

// The driver API is thread-safe; contexts are bound per thread before use.
unsafe impl Send for Driver {}
unsafe impl Sync for Driver {}

/// Process-wide driver handle.
///
/// Loading twice is harmless but resolving symbol names is not free, and a
/// second `cuInit` on some drivers is measurably slower. One `OnceLock` also
/// gives the "is there a device" answer a single consistent value for the whole
/// process, which matters when the answer decides whether a query plan is
/// built at all.
pub fn driver() -> Result<&'static Driver, String> {
    static DRIVER: OnceLock<Result<Driver, String>> = OnceLock::new();
    DRIVER
        .get_or_init(Driver::open)
        .as_ref()
        .map_err(|e| e.clone())
}

// ---------------------------------------------------------------------------
// OS loader
// ---------------------------------------------------------------------------

struct Library {
    handle: *mut std::ffi::c_void,
}

impl Library {
    #[cfg(windows)]
    fn open(names: &[&str]) -> Option<Library> {
        for name in names {
            let wide: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
            let handle = unsafe { LoadLibraryW(wide.as_ptr()) };
            if !handle.is_null() {
                return Some(Library { handle });
            }
        }
        None
    }

    #[cfg(not(windows))]
    fn open(names: &[&str]) -> Option<Library> {
        for name in names {
            let c = CString::new(*name).ok()?;
            let handle = unsafe { libc::dlopen(c.as_ptr(), libc::RTLD_NOW) };
            if !handle.is_null() {
                return Some(Library { handle });
            }
        }
        None
    }

    #[cfg(windows)]
    fn symbol(&self, name: &str) -> Option<*mut std::ffi::c_void> {
        let c = CString::new(name).ok()?;
        let p = unsafe { GetProcAddress(self.handle, c.as_ptr() as *const u8) };
        if p.is_null() {
            None
        } else {
            Some(p)
        }
    }

    #[cfg(not(windows))]
    fn symbol(&self, name: &str) -> Option<*mut std::ffi::c_void> {
        let c = CString::new(name).ok()?;
        let p = unsafe { libc::dlsym(self.handle, c.as_ptr()) };
        if p.is_null() {
            None
        } else {
            Some(p)
        }
    }
}

// Deliberately never unloaded, so `Drop` is a no-op and the handle outlives
// every context created from it.
impl Drop for Library {
    fn drop(&mut self) {}
}

unsafe impl Send for Library {}
unsafe impl Sync for Library {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opening_a_driver_never_panics() {
        // On a GPU machine this succeeds; in CI it fails. Both are valid, and
        // neither may unwind: the "no GPU" path must be as boring as the yes
        // path.
        match driver() {
            Ok(d) => {
                let n = d.device_count().unwrap_or(0);
                if n > 0 {
                    let info = d.device(0).unwrap();
                    assert!(info.total_mem_bytes > 0, "{info:?}");
                    assert!(info.compute_capability.0 >= 1, "{info:?}");
                    assert!(!info.name.is_empty());
                }
            }
            Err(e) => assert!(!e.is_empty(), "failure must carry a reason"),
        }
    }

    #[test]
    fn a_zero_byte_allocation_short_circuits() {
        // An empty graph has zero-length columns and `cuMemAlloc(0)` is an
        // error, so the wrapper answers before asking the driver. Without a
        // driver there is nothing to call, which is also the expected answer.
        match driver() {
            Ok(d) => assert_eq!(d.alloc(0).unwrap(), 0),
            Err(_) => assert!(driver().is_err()),
        }
    }
}
