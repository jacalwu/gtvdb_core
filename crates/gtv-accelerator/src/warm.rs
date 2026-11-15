//! Host-page warming for the RAM tier — P1c's "预告" half
//! (`docs/storage-tiering-ssd-ram-vram.md` §4.2, row 1).
//!
//! A cold corpus that is about to be uploaded to the device is usually still
//! cold *host* memory: an mmap view whose pages have not been faulted in yet.
//! Reading it through the pinned staging copy then pays one page fault per
//! page inside the critical path of the first query. Warming the range first
//! moves those faults (and the kernel's readahead) ahead of the upload loop:
//!
//! * Linux: `madvise(MADV_WILLNEED)` — the kernel starts reading the range
//!   into the page cache immediately, with sequential readahead.
//! * Windows: `PrefetchVirtualMemory` — the same hint through the working-set
//!   manager (Win8+; the process pseudo-handle is accepted).
//! * Anything else (and any failure): a plain volatile touch per page, which
//!   is strictly better than not faulting the pages in early and is the only
//!   portable fallback.
//!
//! All three are hints: they may do nothing and must never fail a query.
//!
//! ## SAFETY contract
//!
//! [`warm_pages`] takes a raw range. The caller guarantees the range is valid
//! readable memory for the duration of the call (the page tables may be
//! touched, but no byte is read into user code on the hint paths — the touch
//! fallback reads exactly one byte per page). The safe entry point is
//! [`warm_slice`].

use std::sync::atomic::{AtomicU64, Ordering};

/// Page stride used by the touch fallback and for `madvise` alignment.
/// 4 KiB is the smallest page size any supported target uses; warming a
/// superset of the range is harmless.
const PAGE: usize = 4096;

/// Warming requests served (telemetry; hints that did real work or fell back
/// both count — the counter answers "did anything ask for warming", not "did
/// the OS honor it").
static WARM_CALLS: AtomicU64 = AtomicU64::new(0);

/// Number of [`warm_pages`] calls served since process start.
pub fn warm_calls() -> u64 {
    WARM_CALLS.load(Ordering::Relaxed)
}

/// Ask the OS to page in `[ptr, ptr+len)` before it is first read.
///
/// SAFETY: `ptr` must be valid for reads of `len` bytes (the range may be
/// touched by the fallback path). The call is synchronous only in the sense
/// that it returns; the hint itself is fire-and-forget.
pub unsafe fn warm_pages(ptr: *const u8, len: usize) {
    if len == 0 || ptr.is_null() {
        return;
    }
    WARM_CALLS.fetch_add(1, Ordering::Relaxed);

    #[cfg(target_os = "linux")]
    {
        // `madvise` wants a page-aligned range; widen conservatively.
        let start = (ptr as usize) & !(PAGE - 1);
        let end = (ptr as usize + len).div_ceil(PAGE) * PAGE;
        const MADV_WILLNEED: i32 = 3;
        extern "C" {
            fn madvise(addr: *mut u8, len: usize, advice: i32) -> i32;
        }
        let rc = unsafe { madvise(start as *mut u8, end - start, MADV_WILLNEED) };
        if rc == 0 {
            return;
        }
        // EINVAL (alignment) or any other refusal: fall through to the touch.
    }

    #[cfg(windows)]
    {
        #[repr(C)]
        struct Win32MemoryRangeEntry {
            virtual_address: *mut core::ffi::c_void,
            number_of_bytes: usize,
        }
        extern "system" {
            fn PrefetchVirtualMemory(
                process: *mut core::ffi::c_void,
                number_of_entries: usize,
                virtual_addresses: *const Win32MemoryRangeEntry,
                flags: u32,
            ) -> i32;
        }
        // The process pseudo-handle, `(HANDLE)-1`; documented as accepted by
        // `PrefetchVirtualMemory` and avoids a `GetCurrentProcess` round-trip.
        let pseudo_process = (-1isize) as *mut core::ffi::c_void;
        let entry = Win32MemoryRangeEntry {
            virtual_address: ptr as *mut core::ffi::c_void,
            number_of_bytes: len,
        };
        let ok = unsafe {
            PrefetchVirtualMemory(pseudo_process, 1, &entry, 0)
        };
        if ok != 0 {
            return;
        }
        // Fall through to the touch on any refusal (e.g. exotic targets).
    }

    // Portable fallback: fault every page in with one volatile read each.
    let mut p = ptr;
    let end = ptr.add(len);
    while p < end {
        let _byte = std::ptr::read_volatile(p);
        p = p.add(PAGE);
    }
}

/// Warm a typed slice's backing memory. Safe wrapper around [`warm_pages`].
pub fn warm_slice<T>(data: &[T]) {
    if data.is_empty() {
        return;
    }
    let bytes = std::mem::size_of_val(data);
    // SAFETY: `data` is a live slice, so its byte range is valid for reads.
    unsafe { warm_pages(data.as_ptr() as *const u8, bytes) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warm_empty_slice_is_a_no_op() {
        let empty: [u8; 0] = [];
        warm_slice(&empty);
        let before = warm_calls();
        warm_slice(&empty);
        assert_eq!(warm_calls(), before, "empty ranges must not count");
    }

    #[test]
    fn warm_pages_touches_the_fallback_path_safely() {
        // A heap buffer big enough to span several pages; contents must be
        // intact afterwards (the touch reads, never writes).
        let buf = vec![0xABu8; 4 * PAGE + 17];
        warm_slice(&buf);
        assert!(buf.iter().all(|&b| b == 0xAB));
        assert!(warm_calls() >= 1);
    }

    #[test]
    fn warm_pages_survives_unaligned_ranges() {
        let buf = vec![7u32; PAGE]; // 16 KiB
        let start = buf.as_ptr() as usize + 4; // deliberately unaligned
        let len = 2 * PAGE;
        // SAFETY: the offset range lies inside `buf`.
        unsafe { warm_pages(start as *const u8, len) };
    }
}
