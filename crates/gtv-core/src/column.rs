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
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Plain-old-data element types allowed to back a [`Col`].
///
/// Sealed on purpose: only the fixed-width types used by the CSR columns and
/// the vector corpus qualify. `bool`, `char`, references, enums with niches and
/// structs with padding all have invalid bit patterns and would make byte
/// reinterpretation undefined behaviour.
///
/// The admission criterion is «every bit pattern is a valid value and there is
/// no padding»: `f32` and `f64` qualify because NaN / infinities / ±0 are all
/// valid floating-point bit patterns (this is the same judgment `bytemuck`
/// makes), and each has `size_of == align_of`. Any future addition (`i32`, …)
/// must be reviewed against the same criterion.
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

/// A view of `region` starting at `offset` bytes, running to the end of the
/// parent region.
///
/// Used to carve a sub-format (e.g. the GTVVEC segment inside a compound
/// index payload) out of an already-mapped container without a second `mmap`
/// and without copying. The returned view keeps the parent alive.
///
/// This is a free function rather than a trait method because the trait is
/// used as a trait object (`Arc<dyn MappedRegion>`), and a method taking
/// `self: Arc<Self>` would require `Self: Sized`.
///
/// # Panics
///
/// Panics if `offset` is past the end of the region.
pub fn subrange(region: Arc<dyn MappedRegion>, offset: usize) -> Arc<dyn MappedRegion> {
    subregion(region, offset, None)
}

/// A view of `region` covering `offset .. offset + len`.
///
/// `len = None` means "to the end of the parent". A bounded view matters when
/// the parent holds trailing bytes that are *not* part of the sub-format — a
/// container's trailing checksum, for instance — because a checksum computed
/// over the sub-format must not see them.
///
/// # Panics
///
/// Panics if the requested range is past the end of the region.
pub fn subregion(
    region: Arc<dyn MappedRegion>,
    offset: usize,
    len: Option<usize>,
) -> Arc<dyn MappedRegion> {
    let total = region.as_bytes().len();
    assert!(
        offset <= total,
        "subregion offset {offset} past region end {total}"
    );
    if let Some(l) = len {
        assert!(
            offset + l <= total,
            "subregion length {l} at offset {offset} past region end {total}"
        );
    }
    Arc::new(SubRegion {
        inner: region,
        offset,
        len,
    })
}

/// A bounded byte-range view over a parent region.
#[derive(Debug)]
pub struct SubRegion {
    inner: Arc<dyn MappedRegion>,
    offset: usize,
    /// `None` = run to the end of the parent region.
    len: Option<usize>,
}

impl SubRegion {
    /// The byte offset this view starts at within its parent.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// The byte length of this view, if it is bounded.
    pub fn len(&self) -> Option<usize> {
        self.len
    }
}

impl MappedRegion for SubRegion {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        let tail = &self.inner.as_bytes()[self.offset..];
        match self.len {
            Some(l) => &tail[..l],
            None => tail,
        }
    }
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

    /// The stable host-memory byte range backing this column, when one is safe
    /// to hand to `cudaHostRegister`.
    ///
    /// Both variants share a `NonNull` pointer that stays put for the column's
    /// lifetime — `Mapped` because the [`MappedRegion`] contract pins it, and
    /// `Owned` because the `Vec` is not reallocated while this borrow lives.
    /// This is the single place the accelerator can learn «these bytes are
    /// stable, register them» without peeking at the enum's fields.
    ///
    /// Returns `None` for an empty column (a dangling sentinel has no bytes to
    /// pin). The length is the element count times `size_of::<T>()`, i.e. the
    /// exact span `cudaHostRegister` must pin.
    pub fn stable_bytes(&self) -> Option<(std::ptr::NonNull<u8>, usize)> {
        let (ptr, len) = match self {
            Col::Owned(v) => (v.as_ptr() as *const u8, v.len() * std::mem::size_of::<T>()),
            Col::Mapped(m) => (m.ptr as *const u8, m.len * std::mem::size_of::<T>()),
        };
        if len == 0 {
            return None;
        }
        // SAFETY: `as_ptr` of a non-empty `Vec`/`MappedCol` is non-null.
        Some((unsafe { std::ptr::NonNull::new_unchecked(ptr as *mut u8) }, len))
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
    fn mapped_f32_column_reinterprets_le_bytes() {
        // f32 rows at a 64-byte aligned offset, including a NaN and ±0 to prove
        // every bit pattern survives the round trip (Pod admission criterion).
        let values = [1.5f32, -2.25f32, f32::NAN, 0.0f32, -0.0f32];
        let mut bytes = vec![0u8; 128];
        for (i, v) in values.iter().enumerate() {
            bytes[64 + i * 4..68 + i * 4].copy_from_slice(&v.to_le_bytes());
        }
        let region: Arc<dyn MappedRegion> = Arc::new(FakeRegion(bytes));
        let c = MappedCol::<f32>::new(region, 64, values.len()).unwrap();
        assert!(c.is_mapped());
        assert_eq!(c.len(), values.len());
        assert_eq!(c[0], 1.5);
        assert_eq!(c[1], -2.25);
        assert!(c[2].is_nan());
        // ±0 both map to 0.0 but keep their sign bit in the raw representation.
        assert_eq!(c[3], 0.0);
        assert_eq!(c[3].to_bits(), 0.0f32.to_bits());
        assert_eq!(c[4].to_bits(), (-0.0f32).to_bits());
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
    fn stable_bytes_reports_the_backing_span() {
        let owned: Col<u32> = vec![1u32, 2, 3].into();
        let (ptr, len) = owned.stable_bytes().unwrap();
        assert_eq!(len, 3 * 4);
        // The pointer aliases the Vec's own buffer.
        assert_eq!(ptr.as_ptr() as usize, (&owned[0] as *const u32) as usize);

        let mut bytes = vec![0u8; 64];
        bytes[0..8].copy_from_slice(&42u64.to_le_bytes());
        let region: Arc<dyn MappedRegion> = Arc::new(FakeRegion(bytes));
        let mapped = MappedCol::<u64>::new(region, 0, 1).unwrap();
        let (mptr, mlen) = mapped.stable_bytes().unwrap();
        assert_eq!(mlen, 8);
        assert_eq!(mptr.as_ptr() as usize, (&mapped[0] as *const u64) as usize);
    }

    #[test]
    fn empty_column_has_no_stable_bytes() {
        let owned: Col<u32> = Vec::new().into();
        assert!(owned.stable_bytes().is_none());
        let region: Arc<dyn MappedRegion> = Arc::new(FakeRegion(vec![0u8; 8]));
        let mapped = MappedCol::<u64>::new(region, 8, 0).unwrap();
        assert!(mapped.stable_bytes().is_none());
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

    #[test]
    fn subrange_views_a_suffix_without_copying() {
        let mut bytes = vec![0u8; 64];
        bytes[16..24].copy_from_slice(&7u64.to_le_bytes());
        let region: Arc<dyn MappedRegion> = Arc::new(FakeRegion(bytes));
        let sub = subrange(region, 16);
        assert_eq!(sub.as_bytes().len(), 48);
        assert_eq!(u64::from_le_bytes(sub.as_bytes()[0..8].try_into().unwrap()), 7);
        // The sub-view keeps the parent alive and stays a valid column source.
        let col = MappedCol::<u64>::new(sub, 0, 1).unwrap();
        assert_eq!(col[0], 7);
    }

    #[test]
    fn subrange_at_end_is_empty_but_valid() {
        let region: Arc<dyn MappedRegion> = Arc::new(FakeRegion(vec![1u8; 16]));
        let sub = subrange(region, 16);
        assert!(sub.as_bytes().is_empty());
    }

    #[test]
    #[should_panic(expected = "past region end")]
    fn subrange_past_the_end_panics() {
        let region: Arc<dyn MappedRegion> = Arc::new(FakeRegion(vec![1u8; 16]));
        let _ = subrange(region, 17);
    }

    #[test]
    fn bounded_subregion_excludes_trailing_bytes() {
        let region: Arc<dyn MappedRegion> = Arc::new(FakeRegion(vec![7u8; 16]));
        let bounded = subregion(region, 4, Some(8));
        assert_eq!(bounded.as_bytes().len(), 8);
        assert_eq!(bounded.as_bytes()[0], 7);
        // An unbounded view of the same offset would include the trailing bytes.
        let unbounded = subrange(Arc::new(FakeRegion(vec![7u8; 16])), 4);
        assert_eq!(unbounded.as_bytes().len(), 12);
    }
}
