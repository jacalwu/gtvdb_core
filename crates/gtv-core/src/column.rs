//! Pluggable column storage for the temporal CSR.
//!
//! `TemporalCSR` used to own plain `Vec<T>` columns. That made it impossible to
//! serve an index straight off a memory-mapped file: the vectors live on the
//! heap and the file bytes have a different shape.
//!
//! [`Col<T>`] is a two-variant backend over which both a `Vec<T>` and an mmap
//! byte range implement `Deref<Target = [T]>`. Every operator in `csr.rs`
//! reaches columns through indexing / `.len()` / slicing, so swapping `Vec<T>`
//! for `Col<T>` leaves those operators untouched and adds **no lifetime
//! parameter** to `TemporalCSR` — a lifetime there would leak into the
//! `TemporalGraphIndex` trait, every UDF and the CLI.
//!
//! # Safety model
//!
//! The only `unsafe` in this module is `MappedCol::as_slice`, justified once at
//! construction time:
//!
//! * [`MappedCol::new`] checks the byte range lies inside the region, that the
//!   pointer is aligned for `T`, and that `len * size_of::<T>()` cannot
//!   overflow.
//! * [`Pod`] is sealed to plain integers, so every bit pattern is valid and
//!   there is no padding left uninitialized.
//! * [`MappedRegion`] is `Send + Sync` and implementors promise a stable,
//!   *immutable* address for the object's lifetime; the `Arc` held by
//!   `MappedCol` keeps that object (hence the mapping) alive.
//!
//! GTVSeg files are immutable by construction (`gtv-storage/src/segment.rs`),
//! so a mapped column can never observe a torn write.
//!
//! # Why the pointer is cached
//!
//! A naive `Mapped` variant would call `region.as_bytes()` inside `Deref`.
//! That is a trait-object call on **every single `self.dst[e]`**, i.e. in the
//! hottest loop of the engine, and it defeats inlining. `MappedCol` instead
//! resolves the slice once in `new()` and stores `*const T` + `len`; `Deref`
//! then costs one branch and one `from_raw_parts`.

use std::fmt;
use std::ops::Deref;
use std::sync::Arc;

mod sealed {
    pub trait Sealed {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
    impl Sealed for i64 {}
}

/// Plain-old-data element types allowed to back a [`Col`].
///
/// Sealed on purpose: only the four fixed-width integer types used by the CSR
/// columns qualify. `bool`, `char`, references, enums with niches and structs
/// with padding all have invalid bit patterns and would make byte
/// reinterpretation undefined behaviour.
pub trait Pod: sealed::Sealed + Copy + Send + Sync + 'static {}

impl<T> Pod for T where T: sealed::Sealed + Copy + Send + Sync + 'static {}

/// Errors raised while attaching a mapped byte range to a typed column.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ColumnError {
    #[error("mapped column byte range ends at {end}, region has only {available} bytes")]
    OutOfBounds { end: usize, available: usize },

    #[error("mapped column address {addr:#x} is not aligned to {align}")]
    Misaligned { addr: usize, align: usize },

    #[error("mapped column byte length overflows usize")]
    LengthOverflow,
}

/// A read-only, immutable byte region kept alive as long as a [`Col`] uses it.
///
/// # Contract for implementors
///
/// `as_bytes()` must return a slice whose **address and length never change**
/// for the lifetime of `self`, and no writer may mutate those bytes while any
/// `Col` derived from it is alive. `memmap2::Mmap` over an immutable segment
/// file satisfies this; a growable `Vec<u8>` does **not** (reallocation would
/// move the buffer).
pub trait MappedRegion: fmt::Debug + Send + Sync {
    fn as_bytes(&self) -> &[u8];
}

/// A typed view over a byte range owned by an [`Arc<dyn MappedRegion>`].
pub struct MappedCol<T: Pod> {
    /// Keeps the mapping alive. Never read after construction — its only job
    /// is lifetime extension for `ptr`.
    _region: Arc<dyn MappedRegion>,
    ptr: *const T,
    len: usize,
}

// SAFETY: `ptr` aliases an immutable region owned by an
// `Arc<dyn MappedRegion: Send + Sync>`. The region's address is stable for the
// lifetime of the Arc (see the trait contract), and no mutable access exists.
// Sending / sharing `MappedCol` therefore cannot race or dangle.
unsafe impl<T: Pod> Send for MappedCol<T> {}
unsafe impl<T: Pod> Sync for MappedCol<T> {}

impl<T: Pod> Clone for MappedCol<T> {
    fn clone(&self) -> Self {
        Self {
            _region: Arc::clone(&self._region),
            ptr: self.ptr,
            len: self.len,
        }
    }
}

impl<T: Pod> fmt::Debug for MappedCol<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MappedCol")
            .field("len", &self.len)
            .field("elem_size", &std::mem::size_of::<T>())
            .finish()
    }
}

impl<T: Pod> MappedCol<T> {
    /// Attach `len` elements of type `T` at `byte_offset` inside `region`.
    ///
    /// Performs every alignment / bounds / overflow check up front so that
    /// [`Deref`] can be an unconditional, branch-free `from_raw_parts`.
    pub fn new(
        region: Arc<dyn MappedRegion>,
        byte_offset: usize,
        len: usize,
    ) -> Result<Col<T>, ColumnError> {
        let size = std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();

        let needed = len.checked_mul(size).ok_or(ColumnError::LengthOverflow)?;
        let end = byte_offset
            .checked_add(needed)
            .ok_or(ColumnError::LengthOverflow)?;

        let bytes = region.as_bytes();
        if end > bytes.len() {
            return Err(ColumnError::OutOfBounds {
                end,
                available: bytes.len(),
            });
        }

        // `from_raw_parts` requires a non-null, aligned pointer even when
        // `len == 0`. An empty column may legally sit at the very end of the
        // region, where `slice.as_ptr()` is not guaranteed to be aligned for
        // T, so use a dangling-but-aligned sentinel in that case.
        let ptr = if len == 0 {
            std::ptr::NonNull::<T>::dangling().as_ptr() as *const T
        } else {
            let slice = &bytes[byte_offset..end];
            let p = slice.as_ptr() as *const T;
            if (p as usize) % align != 0 {
                return Err(ColumnError::Misaligned {
                    addr: p as usize,
                    align,
                });
            }
            p
        };

        Ok(Col::Mapped(MappedCol {
            _region: region,
            ptr,
            len,
        }))
    }

    #[inline]
    fn as_slice(&self) -> &[T] {
        // SAFETY: validated by `new` (bounds, alignment, overflow, POD), and
        // `_region` keeps the immutable backing storage alive for `&self`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

/// A CSR column: either heap-owned or a zero-copy view over a mapped segment.
pub enum Col<T: Pod> {
    /// Built by `TemporalCSR::from_arrays` (ingest / in-memory graphs).
    Owned(Vec<T>),
    /// Loaded by `open_segment` (persisted, immutable, zero-copy).
    Mapped(MappedCol<T>),
}

impl<T: Pod> From<Vec<T>> for Col<T> {
    #[inline]
    fn from(v: Vec<T>) -> Self {
        Col::Owned(v)
    }
}

impl<T: Pod> Col<T> {
    /// True when this column is backed by an immutable memory mapping.
    pub fn is_mapped(&self) -> bool {
        matches!(self, Col::Mapped(_))
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.deref().len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: Pod> Deref for Col<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        match self {
            Col::Owned(v) => v.as_slice(),
            Col::Mapped(m) => m.as_slice(),
        }
    }
}

impl<T: Pod> AsRef<[T]> for Col<T> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.deref()
    }
}

impl<T: Pod> Clone for Col<T> {
    fn clone(&self) -> Self {
        match self {
            Col::Owned(v) => Col::Owned(v.clone()),
            Col::Mapped(m) => Col::Mapped(m.clone()),
        }
    }
}

impl<T: Pod> fmt::Debug for Col<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Col::Owned(v) => f
                .debug_tuple("Col::Owned")
                .field(&format_args!("len={}", v.len()))
                .finish(),
            Col::Mapped(m) => f
                .debug_tuple("Col::Mapped")
                .field(&format_args!("len={}", m.len))
                .finish(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct FakeRegion(Vec<u8>);

    impl MappedRegion for FakeRegion {
        fn as_bytes(&self) -> &[u8] {
            &self.0
        }
    }

    #[test]
    fn owned_column_behaves_like_a_slice() {
        let c: Col<u32> = vec![1u32, 2, 3].into();
        assert!(!c.is_mapped());
        assert_eq!(c.len(), 3);
        assert_eq!(&c[..], &[1, 2, 3]);
        assert_eq!(&c[1..], &[2, 3]);
    }

    #[test]
    fn mapped_column_reinterprets_little_endian_bytes() {
        // 0..=3 as u32 LE, at a 64-byte aligned offset inside the region.
        let mut bytes = vec![0u8; 128];
        for (i, chunk) in bytes[64..80].chunks_exact_mut(4).enumerate() {
            chunk.copy_from_slice(&(i as u32).to_le_bytes());
        }
        let region: Arc<dyn MappedRegion> = Arc::new(FakeRegion(bytes));
        let c = MappedCol::<u32>::new(region, 64, 4).unwrap();
        assert!(c.is_mapped());
        assert_eq!(&c[..], &[0, 1, 2, 3]);
        assert_eq!(c[2], 2);
    }

    #[test]
    fn misaligned_or_out_of_bounds_range_is_rejected() {
        let region: Arc<dyn MappedRegion> = Arc::new(FakeRegion(vec![0u8; 128]));
        // offset 65 is not 4-byte aligned
        assert!(matches!(
            MappedCol::<u32>::new(Arc::clone(&region), 65, 1),
            Err(ColumnError::Misaligned { .. })
        ));
        assert!(matches!(
            MappedCol::<u32>::new(Arc::clone(&region), 120, 4),
            Err(ColumnError::OutOfBounds { .. })
        ));
    }

    #[test]
    fn empty_column_is_safe_to_deref() {
        let region: Arc<dyn MappedRegion> = Arc::new(FakeRegion(vec![0u8; 8]));
        // Empty column at the very end of the region: as_ptr() would be
        // unaligned, so `new` must fall back to the dangling sentinel.
        let c = MappedCol::<u64>::new(region, 8, 0).unwrap();
        assert!(c.is_empty());
        assert_eq!(&c[..], &[] as &[u64]);
    }

    #[test]
    fn cloned_mapped_column_shares_the_region() {
        let mut bytes = vec![0u8; 64];
        bytes[0..8].copy_from_slice(&42u64.to_le_bytes());
        let region: Arc<dyn MappedRegion> = Arc::new(FakeRegion(bytes));
        let c = MappedCol::<u64>::new(region, 0, 1).unwrap();
        let d = c.clone();
        assert_eq!(c[0], 42);
        assert_eq!(d[0], 42);
    }
}
