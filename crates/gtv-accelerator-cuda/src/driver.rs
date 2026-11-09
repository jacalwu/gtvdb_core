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

// `CUjit_option` values for the JIT error log: the buffer, and its size in
// bytes. Both are passed the same way — the size is a `size_t` riding in a
// `void*` slot, which is what the header's `void** optionValues` means.
const CU_JIT_ERROR_LOG_BUFFER: i32 = 5;
const CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: i32 = 6;
/// Enough for a line number and a sentence; the JIT truncates rather than
/// failing if a message does not fit.
const JIT_LOG_BYTES: usize = 4096;

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
type FnCtxSynchronize = unsafe extern "C" fn() -> CuResult;
type FnMemHostAlloc = unsafe extern "C" fn(*mut *mut std::ffi::c_void, usize, u32) -> CuResult;
type FnMemFreeHost = unsafe extern "C" fn(*mut std::ffi::c_void) -> CuResult;
type FnMemcpyHtoDAsync =
    unsafe extern "C" fn(u64, *const std::ffi::c_void, usize, *mut std::ffi::c_void) -> CuResult;
type FnErrorString = unsafe extern "C" fn(CuResult, *mut *const std::ffi::c_char) -> CuResult;
type FnMemcpyDtoH = unsafe extern "C" fn(*mut std::ffi::c_void, u64, usize) -> CuResult;
type FnModuleLoad = unsafe extern "C" fn(
    *mut *mut std::ffi::c_void,
    *const std::ffi::c_void,
    u32,
    *mut i32,
    *mut *mut std::ffi::c_void,
) -> CuResult;
type FnModuleFunction = unsafe extern "C" fn(
    *mut *mut std::ffi::c_void,
    *mut std::ffi::c_void,
    *const std::ffi::c_char,
) -> CuResult;
type FnModuleUnload = unsafe extern "C" fn(*mut std::ffi::c_void) -> CuResult;
type FnLaunch = unsafe extern "C" fn(
    *mut std::ffi::c_void,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    *mut std::ffi::c_void,
    *mut *mut std::ffi::c_void,
    *mut *mut std::ffi::c_void,
) -> CuResult;
type FnEventCreate = unsafe extern "C" fn(*mut *mut std::ffi::c_void, u32) -> CuResult;
type FnEventRecord = unsafe extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_void) -> CuResult;
type FnEventSync = unsafe extern "C" fn(*mut std::ffi::c_void) -> CuResult;
type FnEventElapsed = unsafe extern "C" fn(*mut f32, *mut std::ffi::c_void, *mut std::ffi::c_void) -> CuResult;
type FnEventDestroy = unsafe extern "C" fn(*mut std::ffi::c_void) -> CuResult;

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
    ctx_synchronize: FnCtxSynchronize,
    mem_host_alloc: FnMemHostAlloc,
    mem_free_host: FnMemFreeHost,
    memcpy_htod_async: FnMemcpyHtoDAsync,
    memcpy_dtoh: FnMemcpyDtoH,
    module_load_ex: FnModuleLoad,
    module_function: FnModuleFunction,
    module_unload: FnModuleUnload,
    launch: FnLaunch,
    event_create: FnEventCreate,
    event_record: FnEventRecord,
    event_sync: FnEventSync,
    event_elapsed: FnEventElapsed,
    event_destroy: FnEventDestroy,
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

    /// Allocate a page-locked host buffer.
    ///
    /// Pinned memory is what lets an H2D copy be a DMA instead of a staged
    /// bounce through a driver-internal buffer, at the cost of a host copy into
    /// it and of pages the OS can never swap. Whether that trade wins is a
    /// measurement, not an assumption: see `examples/bench_residency.rs`, which
    /// prices `fill + async copy` against the pageable `cuMemcpyHtoD` rather
    /// than comparing the copy calls alone.
    ///
    /// Takes `&'static self` because the buffer outlives the call and has to be
    /// able to free itself; every driver handle in this crate comes from
    /// [`driver`], which is a process-wide `OnceLock`.
    pub fn host_alloc(&'static self, bytes: usize) -> Result<PinnedBuffer, String> {
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        // Flags 0 = default (pinned, not portable, not write-combined). Portable
        // only matters across contexts and write-combined trades read speed for
        // a much slower CPU write, which is the wrong side of this trade.
        let rc = unsafe { (self.fns.mem_host_alloc)(&mut ptr, bytes, 0) };
        if rc != CUDA_SUCCESS || ptr.is_null() {
            return Err(self.error(&format!("cuMemHostAlloc({bytes})"), rc));
        }
        Ok(PinnedBuffer {
            ptr,
            len: bytes,
            driver: self,
        })
    }

    /// Host-to-device copy from page-locked memory. Completion is signalled
    /// through the context, so the caller must [`Driver::sync`] before reading
    /// anything that depends on it.
    ///
    /// # Safety
    ///
    /// `dst` must be a live device allocation of at least `len` bytes and `buf`
    /// must be pinned memory of at least `len` bytes from [`Driver::host_alloc`].
    pub unsafe fn copy_htod_async(&self, dst: u64, buf: &PinnedBuffer) -> Result<(), String> {
        if buf.len == 0 {
            return Ok(());
        }
        // The NULL stream is the legacy default stream: it orders against the
        // context's synchronising calls, which `cuCtxSynchronize` waits on.
        let rc = (self.fns.memcpy_htod_async)(
            dst,
            buf.ptr as *const std::ffi::c_void,
            buf.len,
            std::ptr::null_mut(),
        );
        if rc != CUDA_SUCCESS {
            return Err(self.error(&format!("cuMemcpyHtoDAsync({})", buf.len), rc));
        }
        Ok(())
    }

    /// Block until every prior call on this context has completed.
    ///
    /// `cuMemcpyHtoD` already blocks, so in the current path this measures
    /// nothing — which is exactly the point: it is the control that proves the
    /// per-column timings are transfer time and not queued work. If a copy is
    /// ever made asynchronous, this is where its unaccounted remainder appears.
    pub fn sync(&self) -> Result<(), String> {
        let rc = unsafe { (self.fns.ctx_synchronize)() };
        if rc != CUDA_SUCCESS {
            return Err(self.error("cuCtxSynchronize", rc));
        }
        Ok(())
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

    /// Device-to-host copy. Blocking, like its H2D counterpart: it returns after
    /// the bytes are in host memory, which is also what makes it a barrier for a
    /// kernel queued before it on the same stream.
    ///
    /// # Safety
    ///
    /// `src` must be a device pointer with at least `len` readable bytes, and
    /// `dst` must be writable for `len` bytes.
    pub unsafe fn copy_dtoh(&self, dst: *mut u8, src: u64, len: usize) -> Result<(), String> {
        if len == 0 {
            return Ok(());
        }
        let rc = (self.fns.memcpy_dtoh)(dst as *mut std::ffi::c_void, src, len);
        if rc != CUDA_SUCCESS {
            return Err(self.error(&format!("cuMemcpyDtoH({len})"), rc));
        }
        Ok(())
    }

    // -- modules and kernels ------------------------------------------------

    /// Load PTX text into the current context.
    ///
    /// The PTX arrives as text because that is what the driver JITs: one
    /// `compute_80` image covers every card from Ampere onward, which a
    /// machine-code cubin would not. There is no toolkit dependency at either
    /// build or run time — the image is a file in this repository.
    pub fn load_ptx(&'static self, ptx: &str) -> Result<Module, String> {
        let image = CString::new(ptx)
            .map_err(|_| "PTX contains a NUL byte".to_string())?;
        let mut handle: *mut std::ffi::c_void = std::ptr::null_mut();
        // `cuModuleLoadDataEx` rather than `cuModuleLoadData` for one reason:
        // the JIT's error log. A plain load reports `CUDA_ERROR_INVALID_PTX` and
        // nothing else — "compilation failed" with no line number. The log has
        // the line and the reason, which is the difference between a five-minute
        // fix and a bisect.
        let mut log = vec![0u8; JIT_LOG_BYTES];
        let mut options: [i32; 2] = [CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES];
        let mut values: [*mut std::ffi::c_void; 2] = [
            log.as_mut_ptr() as *mut std::ffi::c_void,
            JIT_LOG_BYTES as *mut std::ffi::c_void,
        ];
        let rc = unsafe {
            (self.fns.module_load_ex)(
                &mut handle,
                image.as_ptr() as *const std::ffi::c_void,
                options.len() as u32,
                options.as_mut_ptr(),
                values.as_mut_ptr(),
            )
        };
        if rc != CUDA_SUCCESS || handle.is_null() {
            let detail = unsafe { std::ffi::CStr::from_ptr(log.as_ptr() as *const std::ffi::c_char) }
                .to_string_lossy()
                .trim()
                .to_string();
            let mut msg = self.error("cuModuleLoadDataEx", rc);
            if !detail.is_empty() {
                msg.push_str(": ");
                msg.push_str(&detail);
            }
            return Err(msg);
        }
        Ok(Module { handle, driver: self })
    }

    // -- timing -------------------------------------------------------------

    /// A recorded instant on the current stream.
    pub fn event(&'static self) -> Result<Event, String> {
        let mut handle: *mut std::ffi::c_void = std::ptr::null_mut();
        let rc = unsafe { (self.fns.event_create)(&mut handle, 0) };
        if rc != CUDA_SUCCESS || handle.is_null() {
            return Err(self.error("cuEventCreate", rc));
        }
        Ok(Event {
            handle,
            driver: self,
        })
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

/// A page-locked host buffer owned by the driver.
///
/// Held by value so the freeing cannot be forgotten, and never handed out as a
/// bare pointer: the only ways to touch its bytes are [`PinnedBuffer::fill`] and
/// [`Driver::copy_htod_async`], both of which bound-check against `len`.
pub struct PinnedBuffer {
    ptr: *mut std::ffi::c_void,
    len: usize,
    driver: &'static Driver,
}

impl PinnedBuffer {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Stage `len` bytes into the pinned buffer.
    ///
    /// This is the cost an asynchronous upload pays instead of the driver's
    /// hidden bounce copy — the reason a pinned path has to be measured
    /// end-to-end and not by timing `cuMemcpyHtoDAsync` alone.
    ///
    /// # Safety
    ///
    /// `src` must be readable for `len` bytes, and `len` must be at most
    /// [`PinnedBuffer::len`].
    pub unsafe fn fill(&mut self, src: *const u8, len: usize) -> Result<(), String> {
        if len > self.len {
            return Err(format!("fill {len} into a {} byte buffer", self.len));
        }
        std::ptr::copy_nonoverlapping(src, self.ptr as *mut u8, len);
        Ok(())
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                (self.driver.fns.mem_free_host)(self.ptr);
            }
        }
        self.ptr = std::ptr::null_mut();
    }
}

// SAFETY: the buffer owns its allocation for its own lifetime and is not shared
// between threads by anything in this module.
unsafe impl Send for PinnedBuffer {}

/// A loaded PTX module. Owns the handle, so the unload cannot be forgotten.
pub struct Module {
    handle: *mut std::ffi::c_void,
    driver: &'static Driver,
}

impl Module {
    /// Look up a `__global__` entry point. The returned [`Function`] borrows the
    /// module: unloading the module invalidates every function in it, and the
    /// borrow is what prevents that from being expressible.
    pub fn function(&self, name: &str) -> Result<Function<'_>, String> {
        let c = CString::new(name).map_err(|_| format!("{name} contains a NUL"))?;
        let mut handle: *mut std::ffi::c_void = std::ptr::null_mut();
        let rc = unsafe { (self.driver.fns.module_function)(&mut handle, self.handle, c.as_ptr()) };
        if rc != CUDA_SUCCESS || handle.is_null() {
            return Err(self.driver.error(&format!("cuModuleGetFunction({name})"), rc));
        }
        Ok(Function {
            handle,
            module: self,
        })
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // `cuModuleUnload` needs the context the module was loaded in to be
            // current. If it is not, the call fails and the module leaks with
            // the context — which is why the kernel module is a process-wide
            // `OnceLock` and never dropped in practice.
            unsafe {
                (self.driver.fns.module_unload)(self.handle);
            }
        }
        self.handle = std::ptr::null_mut();
    }
}

// SAFETY: a CUmodule is a context-scoped handle, not a pointer this code ever
// dereferences; the driver API is thread-safe and every launch below re-binds
// the context on the calling thread first.
unsafe impl Send for Module {}
unsafe impl Sync for Module {}

/// A kernel entry point, borrowed from its module.
pub struct Function<'a> {
    handle: *mut std::ffi::c_void,
    module: &'a Module,
}

// SAFETY: a CUfunction is an opaque handle scoped to a context, not a pointer
// this code ever dereferences. Launching from a thread whose current context is
// not the module's returns a driver error rather than misbehaving, and the
// driver API is thread-safe.
unsafe impl Send for Function<'_> {}
unsafe impl Sync for Function<'_> {}

impl Function<'_> {
    /// Launch on the legacy default stream.
    ///
    /// `args` is the kernel's parameter list, one `*mut c_void` per parameter
    /// pointing at the value. It is `&mut` because the driver's signature wants
    /// a mutable array; nothing writes through it.
    ///
    /// # Safety
    ///
    /// Every element of `args` must point to a live value of exactly the type
    /// the kernel declares at that position, and `grid * block` must not index
    /// outside the buffers those pointers name. The type system cannot check
    /// either, which is why exactly one call site in this crate does it.
    pub unsafe fn launch(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_bytes: usize,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<(), String> {
        let rc = (self.module.driver.fns.launch)(
            self.handle,
            grid.0,
            grid.1,
            grid.2,
            block.0,
            block.1,
            block.2,
            shared_bytes as u32,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        if rc != CUDA_SUCCESS {
            return Err(self
                .module
                .driver
                .error(&format!("cuLaunchKernel(shared={shared_bytes})"), rc));
        }
        Ok(())
    }
}

/// A recorded instant in the current stream, used to split a call into stages.
///
/// Events rather than `Instant::now()` around a blocking copy: the kernel runs
/// asynchronously, so wall time around the launch measures the launch, and wall
/// time around the copy measures copy + kernel together. Recording on the stream
/// is the only way to attribute the milliseconds to the right stage.
pub struct Event {
    handle: *mut std::ffi::c_void,
    driver: &'static Driver,
}

impl Event {
    /// Mark this point in the stream.
    pub fn record(&self) -> Result<(), String> {
        let rc = unsafe { (self.driver.fns.event_record)(self.handle, std::ptr::null_mut()) };
        if rc != CUDA_SUCCESS {
            return Err(self.driver.error("cuEventRecord", rc));
        }
        Ok(())
    }

    /// Block until the stream reaches this event.
    pub fn synchronize(&self) -> Result<(), String> {
        let rc = unsafe { (self.driver.fns.event_sync)(self.handle) };
        if rc != CUDA_SUCCESS {
            return Err(self.driver.error("cuEventSynchronize", rc));
        }
        Ok(())
    }

    /// Milliseconds from `self` to `later`, or an error if either never recorded.
    pub fn elapsed_ms(&self, later: &Event) -> Result<f32, String> {
        let mut ms = 0.0f32;
        let rc = unsafe { (self.driver.fns.event_elapsed)(&mut ms, self.handle, later.handle) };
        if rc != CUDA_SUCCESS {
            return Err(self.driver.error("cuEventElapsedTime", rc));
        }
        Ok(ms)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                (self.driver.fns.event_destroy)(self.handle);
            }
        }
        self.handle = std::ptr::null_mut();
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
            ctx_synchronize: need(lib, "cuCtxSynchronize")?,
            mem_host_alloc: need(lib, "cuMemHostAlloc")?,
            mem_free_host: need(lib, "cuMemFreeHost")?,
            memcpy_htod_async: optional(lib, &["cuMemcpyHtoDAsync_v2", "cuMemcpyHtoDAsync"])
                .ok_or_else(|| "driver is missing cuMemcpyHtoDAsync".to_string())?,
            memcpy_dtoh: optional(lib, &["cuMemcpyDtoH_v2", "cuMemcpyDtoH"])
                .ok_or_else(|| "driver is missing cuMemcpyDtoH".to_string())?,
            module_load_ex: need(lib, "cuModuleLoadDataEx")?,
            module_function: need(lib, "cuModuleGetFunction")?,
            module_unload: need(lib, "cuModuleUnload")?,
            launch: need(lib, "cuLaunchKernel")?,
            event_create: need(lib, "cuEventCreate")?,
            event_record: need(lib, "cuEventRecord")?,
            event_sync: need(lib, "cuEventSynchronize")?,
            event_elapsed: need(lib, "cuEventElapsedTime")?,
            event_destroy: optional(lib, &["cuEventDestroy_v2", "cuEventDestroy"])
                .ok_or_else(|| "driver is missing cuEventDestroy".to_string())?,
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
