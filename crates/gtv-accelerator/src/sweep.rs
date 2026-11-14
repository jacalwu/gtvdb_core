//! As-of match sweep.
//!
//! The previous implementation ran a binary search *per left row*: `O(L log R)`,
//! with each probe a dependent load that the branch predictor cannot hide
//! (`R` is usually far larger than L2).
//!
//! Matching is the classic two-pointer merge. Because the answer for a larger
//! timestamp is never smaller than the answer for a smaller one, one monotonic
//! right cursor serves every left row:
//!
//! * left already non-decreasing → a single `O(L + R)` forward pass.
//! * left unsorted → one `O(L log L)` index sort, the same pass over the sorted
//!   order, then a scatter back to input positions. This is still cheaper than
//!   `L` independent binary searches once `R` is past a few thousand rows, and
//!   it keeps the output order identical to the input order.
//!
//! Semantics are unchanged from the binary-search version: a row matches the
//! greatest `right_times[j] <= left` and is dropped when
//! `left - right_times[j] > tolerance`.

/// For each left timestamp, the index of the matched right row.
///
/// `right_times` must be non-decreasing (the registration path guarantees a
/// sorted series). An empty series, or one too large to index with `u32`,
/// yields all `None`.
pub fn asof_match(left: &[i64], right_times: &[i64], tolerance_ns: Option<i64>) -> Vec<Option<u32>> {
    let mut out = vec![None; left.len()];
    if left.is_empty() || right_times.is_empty() || right_times.len() > u32::MAX as usize {
        return out;
    }

    if non_decreasing(left) {
        sweep(
            left.iter().copied().enumerate(),
            right_times,
            tolerance_ns,
            &mut out,
        );
        return out;
    }

    // Unsorted left: order the rows by timestamp, sweep once, scatter back.
    let mut order: Vec<u32> = (0..left.len() as u32).collect();
    order.sort_unstable_by_key(|&i| left[i as usize]);
    sweep(
        order.into_iter().map(|i| (i as usize, left[i as usize])),
        right_times,
        tolerance_ns,
        &mut out,
    );
    out
}

/// True when `s` never decreases.
#[inline]
pub fn non_decreasing(s: &[i64]) -> bool {
    s.windows(2).all(|w| w[0] <= w[1])
}

/// Walk rows in non-decreasing timestamp order with one monotonic cursor.
fn sweep(
    items: impl Iterator<Item = (usize, i64)>,
    right: &[i64],
    tolerance: Option<i64>,
    out: &mut [Option<u32>],
) {
    let mut j = 0usize;
    for (i, lt) in items {
        while j < right.len() && right[j] <= lt {
            j += 1;
        }
        out[i] = matched(j, right, lt, tolerance);
    }
}

#[inline]
fn matched(j: usize, right: &[i64], lt: i64, tolerance: Option<i64>) -> Option<u32> {
    if j == 0 {
        return None;
    }
    let at = j - 1;
    match tolerance {
        Some(tol) if lt - right[at] > tol => None,
        _ => Some(at as u32),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference: one binary search per row, exactly the old code path.
    fn reference(left: &[i64], right: &[i64], tolerance: Option<i64>) -> Vec<Option<u32>> {
        left.iter()
            .map(|&lt| {
                let j = right.partition_point(|&rt| rt <= lt);
                if j == 0 {
                    return None;
                }
                let at = j - 1;
                match tolerance {
                    Some(tol) if lt - right[at] > tol => None,
                    _ => Some(at as u32),
                }
            })
            .collect()
    }

    #[test]
    fn matches_the_binary_search_reference_on_sorted_left() {
        let right = vec![0i64, 5, 10, 15, 20, 25];
        let left = vec![-5i64, 0, 1, 5, 6, 14, 15, 19, 20, 26, 100];
        for tol in [None, Some(0), Some(4), Some(60)] {
            assert_eq!(asof_match(&left, &right, tol), reference(&left, &right, tol));
        }
    }

    #[test]
    fn matches_the_reference_on_unsorted_left() {
        let right = vec![1i64, 3, 3, 7, 9, 100];
        let left = vec![50i64, 3, 0, 7, 9, 8, 1000, -2, 3];
        for tol in [None, Some(0), Some(2), Some(500)] {
            assert_eq!(asof_match(&left, &right, tol), reference(&left, &right, tol));
        }
    }

    #[test]
    fn a_row_before_the_series_matches_nothing() {
        let right = vec![10i64, 20];
        assert_eq!(asof_match(&[9], &right, None), vec![None]);
        assert_eq!(asof_match(&[10], &right, None), vec![Some(0)]);
    }

    #[test]
    fn tolerance_drops_a_stale_match() {
        let right = vec![0i64, 100];
        // 110 - 100 = 10 ns of staleness.
        assert_eq!(asof_match(&[110], &right, Some(5)), vec![None]);
        assert_eq!(asof_match(&[110], &right, Some(10)), vec![Some(1)]);
        assert_eq!(asof_match(&[110], &right, Some(20)), vec![Some(1)]);
    }

    #[test]
    fn duplicate_right_timestamps_take_the_last_one() {
        let right = vec![5i64, 5, 5];
        assert_eq!(asof_match(&[5], &right, None), reference(&[5], &right, None));
    }

    #[test]
    fn empty_sides_are_all_none() {
        assert_eq!(asof_match(&[], &[1, 2], None), Vec::<Option<u32>>::new());
        assert_eq!(asof_match(&[1, 2], &[], None), vec![None, None]);
    }

    #[test]
    fn single_row_sides() {
        assert_eq!(asof_match(&[0], &[1], None), vec![None]);
        assert_eq!(asof_match(&[1], &[1], None), vec![Some(0)]);
        assert_eq!(asof_match(&[2], &[1], None), vec![Some(0)]);
    }

    #[test]
    fn non_decreasing_detects_unsorted_input() {
        assert!(non_decreasing(&[1, 1, 2, 3]));
        assert!(non_decreasing(&[]));
        assert!(non_decreasing(&[7]));
        assert!(!non_decreasing(&[1, 3, 2]));
    }

    /// Not part of CI: prints the sweep-vs-binary-search cost on this host.
    ///
    /// Run with
    /// `cargo test -p gtv-accelerator --release -- --ignored --nocapture bench_sweep`.
    #[test]
    #[ignore = "measurement, not a correctness gate"]
    fn bench_sweep_vs_binary_search() {
        use std::time::Instant;

        const L: usize = 1_000_000;
        const R: usize = 4_000_000;
        let mut seed = 0x2545F491_4F6CDD1Du64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        // Ascending right series, ~4 points per left row.
        let right: Vec<i64> = (0..R as i64).map(|i| i * 4).collect();
        // Ascending left timestamps (the common case for an ordered scan).
        let left: Vec<i64> = (0..L as i64).map(|i| i * 16 + (next() % 4) as i64).collect();

        let t0 = Instant::now();
        let a = asof_match(&left, &right, Some(1_000));
        let sweep = t0.elapsed();

        let t1 = Instant::now();
        let b = reference(&left, &right, Some(1_000));
        let bin = t1.elapsed();

        assert_eq!(a, b, "both paths must agree");
        println!(
            "asof L={L} R={R}: sweep {:?} vs per-row binary search {:?} ({:.2}x)",
            sweep,
            bin,
            bin.as_secs_f64() / sweep.as_secs_f64().max(1e-9)
        );
    }
}
