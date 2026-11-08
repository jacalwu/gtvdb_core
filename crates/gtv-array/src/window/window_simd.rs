//! Target-feature-gated SIMD kernels for the window primitives.
//!
//! Each kernel is compiled only when its feature is enabled at build time via
//! `-C target-feature=+avx2` / `+avx512f`. The public dispatch in
//! [`crate::window`] selects a kernel via `cfg!` at runtime and falls back to
//! the scalar baseline otherwise.
//!
//! AVX-512 intrinsics live under `std::arch::x86_64`, which exists only on
//! x86_64 targets, so every kernel here is additionally gated on
//! `target_arch = "x86_64"`.

#[cfg(all(
    target_arch = "x86_64",
    any(target_feature = "avx2", target_feature = "avx512f")
))]
use std::arch::x86_64::*;

// =========================
// AVX2
// =========================

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub fn mavg_avx2(values: &[f64], n: usize) -> Vec<f64> {
    debug_assert!(n >= 1);
    let len = values.len();
    let mut out = Vec::with_capacity(len);

    let prefix = n.min(len);
    let mut window_sum = 0.0;

    for i in 0..prefix {
        window_sum += values[i];
        out.push(window_sum / (i + 1) as f64);
    }

    let mut i = prefix;
    unsafe {
        let recip_n = _mm256_set1_pd(1.0 / n as f64);
        let mut sum_vec = _mm256_set1_pd(window_sum);

        while i + 4 <= len {
            let v = _mm256_loadu_pd(values.as_ptr().add(i));
            let sub = _mm256_loadu_pd(values.as_ptr().add(i - n));

            sum_vec = _mm256_add_pd(sum_vec, _mm256_sub_pd(v, sub));
            let avg = _mm256_mul_pd(sum_vec, recip_n);

            let mut buf = [0f64; 4];
            _mm256_storeu_pd(buf.as_mut_ptr(), avg);

            out.extend_from_slice(&buf);

            // Advance the scalar running sum to the last lane for the tail.
            window_sum = buf[3] * n as f64;
            sum_vec = _mm256_set1_pd(window_sum);

            i += 4;
        }
    }

    while i < len {
        window_sum += values[i];
        window_sum -= values[i - n];
        out.push(window_sum / n as f64);
        i += 1;
    }

    out
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub fn msum_avx2(values: &[f64], n: usize) -> Vec<f64> {
    debug_assert!(n >= 1);
    let len = values.len();
    let mut out = Vec::with_capacity(len);

    let prefix = n.min(len);
    let mut window_sum = 0.0;

    for i in 0..prefix {
        window_sum += values[i];
        out.push(window_sum);
    }

    let mut i = prefix;
    unsafe {
        let mut sum_vec = _mm256_set1_pd(window_sum);

        while i + 4 <= len {
            let v = _mm256_loadu_pd(values.as_ptr().add(i));
            let sub = _mm256_loadu_pd(values.as_ptr().add(i - n));

            sum_vec = _mm256_add_pd(sum_vec, _mm256_sub_pd(v, sub));

            let mut buf = [0f64; 4];
            _mm256_storeu_pd(buf.as_mut_ptr(), sum_vec);

            out.extend_from_slice(&buf);

            window_sum = buf[3];
            sum_vec = _mm256_set1_pd(window_sum);

            i += 4;
        }
    }

    while i < len {
        window_sum += values[i];
        window_sum -= values[i - n];
        out.push(window_sum);
        i += 1;
    }

    out
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub fn deltas_avx2(values: &[f64]) -> Vec<f64> {
    let len = values.len();
    if len == 0 {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(len);
    out.push(values[0]);

    let mut i = 1;

    unsafe {
        while i + 4 <= len {
            let curr = _mm256_loadu_pd(values.as_ptr().add(i));
            let prev = _mm256_loadu_pd(values.as_ptr().add(i - 1));

            let diff = _mm256_sub_pd(curr, prev);

            let mut buf = [0f64; 4];
            _mm256_storeu_pd(buf.as_mut_ptr(), diff);

            out.extend_from_slice(&buf);
            i += 4;
        }
    }

    while i < len {
        out.push(values[i] - values[i - 1]);
        i += 1;
    }

    out
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub fn fused_avx2(values: &[f64], n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    debug_assert!(n >= 1);
    let len = values.len();
    let mut mavg_out = Vec::with_capacity(len);
    let mut msum_out = Vec::with_capacity(len);
    let mut deltas_out = Vec::with_capacity(len);

    if len == 0 {
        return (mavg_out, msum_out, deltas_out);
    }

    let mut window_sum = 0.0;
    let prefix = n.min(len);

    deltas_out.push(values[0]);

    for i in 0..prefix {
        window_sum += values[i];
        msum_out.push(window_sum);
        mavg_out.push(window_sum / (i + 1) as f64);
        if i > 0 {
            deltas_out.push(values[i] - values[i - 1]);
        }
    }

    let mut i = prefix;
    unsafe {
        let recip_n = _mm256_set1_pd(1.0 / n as f64);
        let mut sum_vec = _mm256_set1_pd(window_sum);

        while i + 4 <= len {
            _mm_prefetch(values.as_ptr().add(i + 64) as *const i8, _MM_HINT_T0);

            let curr = _mm256_loadu_pd(values.as_ptr().add(i));
            let prev = _mm256_loadu_pd(values.as_ptr().add(i - 1));
            let sub = _mm256_loadu_pd(values.as_ptr().add(i - n));

            let diff = _mm256_sub_pd(curr, prev);
            let win_diff = _mm256_sub_pd(curr, sub);
            sum_vec = _mm256_add_pd(sum_vec, win_diff);
            let avg = _mm256_mul_pd(sum_vec, recip_n);

            let mut buf_sum = [0f64; 4];
            let mut buf_avg = [0f64; 4];
            let mut buf_diff = [0f64; 4];

            _mm256_storeu_pd(buf_sum.as_mut_ptr(), sum_vec);
            _mm256_storeu_pd(buf_avg.as_mut_ptr(), avg);
            _mm256_storeu_pd(buf_diff.as_mut_ptr(), diff);

            window_sum = buf_sum[3];
            sum_vec = _mm256_set1_pd(window_sum);

            for j in 0..4 {
                msum_out.push(buf_sum[j]);
                mavg_out.push(buf_avg[j]);
                deltas_out.push(buf_diff[j]);
            }

            i += 4;
        }
    }

    while i < len {
        window_sum += values[i];
        window_sum -= values[i - n];
        msum_out.push(window_sum);
        mavg_out.push(window_sum / n as f64);
        deltas_out.push(values[i] - values[i - 1]);
        i += 1;
    }

    (mavg_out, msum_out, deltas_out)
}

// =========================
// AVX-512
// =========================

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub fn mavg_avx512(values: &[f64], n: usize) -> Vec<f64> {
    debug_assert!(n >= 1);
    let len = values.len();
    let mut out = Vec::with_capacity(len);

    let prefix = n.min(len);
    let mut window_sum = 0.0;

    for i in 0..prefix {
        window_sum += values[i];
        out.push(window_sum / (i + 1) as f64);
    }

    let mut i = prefix;
    unsafe {
        let recip_n = _mm512_set1_pd(1.0 / n as f64);
        let mut sum_vec = _mm512_set1_pd(window_sum);

        while i + 8 <= len {
            _mm_prefetch(values.as_ptr().add(i + 64) as *const i8, _MM_HINT_T0);

            let v = _mm512_loadu_pd(values.as_ptr().add(i));
            let sub = _mm512_loadu_pd(values.as_ptr().add(i - n));

            sum_vec = _mm512_add_pd(sum_vec, _mm512_sub_pd(v, sub));
            let avg = _mm512_mul_pd(sum_vec, recip_n);

            let mut buf = [0f64; 8];
            _mm512_storeu_pd(buf.as_mut_ptr(), avg);

            out.extend_from_slice(&buf);

            window_sum = buf[7] * n as f64;
            sum_vec = _mm512_set1_pd(window_sum);

            i += 8;
        }
    }

    while i < len {
        window_sum += values[i];
        window_sum -= values[i - n];
        out.push(window_sum / n as f64);
        i += 1;
    }

    out
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub fn msum_avx512(values: &[f64], n: usize) -> Vec<f64> {
    debug_assert!(n >= 1);
    let len = values.len();
    let mut out = Vec::with_capacity(len);

    let prefix = n.min(len);
    let mut window_sum = 0.0;

    for i in 0..prefix {
        window_sum += values[i];
        out.push(window_sum);
    }

    let mut i = prefix;
    unsafe {
        let mut sum_vec = _mm512_set1_pd(window_sum);

        while i + 8 <= len {
            _mm_prefetch(values.as_ptr().add(i + 64) as *const i8, _MM_HINT_T0);

            let v = _mm512_loadu_pd(values.as_ptr().add(i));
            let sub = _mm512_loadu_pd(values.as_ptr().add(i - n));

            sum_vec = _mm512_add_pd(sum_vec, _mm512_sub_pd(v, sub));

            let mut buf = [0f64; 8];
            _mm512_storeu_pd(buf.as_mut_ptr(), sum_vec);

            out.extend_from_slice(&buf);

            window_sum = buf[7];
            sum_vec = _mm512_set1_pd(window_sum);

            i += 8;
        }
    }

    while i < len {
        window_sum += values[i];
        window_sum -= values[i - n];
        out.push(window_sum);
        i += 1;
    }

    out
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub fn deltas_avx512(values: &[f64]) -> Vec<f64> {
    let len = values.len();
    if len == 0 {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(len);
    out.push(values[0]);

    let mut i = 1;
    unsafe {
        while i + 8 <= len {
            _mm_prefetch(values.as_ptr().add(i + 64) as *const i8, _MM_HINT_T0);

            let curr = _mm512_loadu_pd(values.as_ptr().add(i));
            let prev = _mm512_loadu_pd(values.as_ptr().add(i - 1));

            let diff = _mm512_sub_pd(curr, prev);

            let mut buf = [0f64; 8];
            _mm512_storeu_pd(buf.as_mut_ptr(), diff);

            out.extend_from_slice(&buf);
            i += 8;
        }
    }

    while i < len {
        out.push(values[i] - values[i - 1]);
        i += 1;
    }

    out
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub fn fused_avx512(values: &[f64], n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    debug_assert!(n >= 1);
    let len = values.len();
    let mut mavg_out = Vec::with_capacity(len);
    let mut msum_out = Vec::with_capacity(len);
    let mut deltas_out = Vec::with_capacity(len);

    if len == 0 {
        return (mavg_out, msum_out, deltas_out);
    }

    let mut window_sum = 0.0;
    let prefix = n.min(len);

    deltas_out.push(values[0]);

    for i in 0..prefix {
        window_sum += values[i];
        msum_out.push(window_sum);
        mavg_out.push(window_sum / (i + 1) as f64);
        if i > 0 {
            deltas_out.push(values[i] - values[i - 1]);
        }
    }

    let mut i = prefix;
    unsafe {
        let recip_n = _mm512_set1_pd(1.0 / n as f64);
        let mut sum_vec = _mm512_set1_pd(window_sum);

        while i + 8 <= len {
            _mm_prefetch(values.as_ptr().add(i + 64) as *const i8, _MM_HINT_T0);

            let curr = _mm512_loadu_pd(values.as_ptr().add(i));
            let prev = _mm512_loadu_pd(values.as_ptr().add(i - 1));
            let sub = _mm512_loadu_pd(values.as_ptr().add(i - n));

            let diff = _mm512_sub_pd(curr, prev);
            let win_diff = _mm512_sub_pd(curr, sub);
            sum_vec = _mm512_add_pd(sum_vec, win_diff);
            let avg = _mm512_mul_pd(sum_vec, recip_n);

            let mut buf_sum = [0f64; 8];
            let mut buf_avg = [0f64; 8];
            let mut buf_diff = [0f64; 8];

            _mm512_storeu_pd(buf_sum.as_mut_ptr(), sum_vec);
            _mm512_storeu_pd(buf_avg.as_mut_ptr(), avg);
            _mm512_storeu_pd(buf_diff.as_mut_ptr(), diff);

            window_sum = buf_sum[7];
            sum_vec = _mm512_set1_pd(window_sum);

            for j in 0..8 {
                msum_out.push(buf_sum[j]);
                mavg_out.push(buf_avg[j]);
                deltas_out.push(buf_diff[j]);
            }

            i += 8;
        }
    }

    while i < len {
        window_sum += values[i];
        window_sum -= values[i - n];
        msum_out.push(window_sum);
        mavg_out.push(window_sum / n as f64);
        deltas_out.push(values[i] - values[i - 1]);
        i += 1;
    }

    (mavg_out, msum_out, deltas_out)
}
