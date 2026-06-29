//! Fast Lloyd k-means in f32. The hot step (assign each point to nearest centroid) is parallel;
//! update accumulates means and re-seeds empty clusters. Used for AVQ codebooks (which need
//! balanced cells) and optionally IVF pivots. Train on a sample for speed; quality is set by iters.

use rayon::prelude::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
fn l2(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0;
    for k in 0..a.len() {
        let e = a[k] - b[k];
        s += e * e;
    }
    s
}

/// AVX2+FMA f32 squared L2 — the k-means assign hot path (8 lanes/iter).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn l2_avx(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut acc = _mm256_setzero_ps();
    let mut k = 0;
    while k + 8 <= n {
        let d = _mm256_sub_ps(_mm256_loadu_ps(a.as_ptr().add(k)), _mm256_loadu_ps(b.as_ptr().add(k)));
        acc = _mm256_fmadd_ps(d, d, acc);
        k += 8;
    }
    let mut t = [0f32; 8];
    _mm256_storeu_ps(t.as_mut_ptr(), acc);
    let mut s = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
    while k < n { let e = a[k] - b[k]; s += e * e; k += 1; }
    s
}

#[inline]
fn nearest(x: &[f32], cent: &[f32], k: usize, d: usize) -> u32 {
    let mut best = f32::INFINITY;
    let mut bj = 0u32;
    #[cfg(target_arch = "x86_64")]
    let avx = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let avx = false;
    for j in 0..k {
        let c = &cent[j * d..j * d + d];
        let dd = if avx {
            #[cfg(target_arch = "x86_64")]
            unsafe { l2_avx(x, c) }
            #[cfg(not(target_arch = "x86_64"))]
            { l2(x, c) }
        } else { l2(x, c) };
        if dd < best { best = dd; bj = j as u32; }
    }
    bj
}

/// k-means over `data` (n x d, row-major f32). Returns k*d centroids.
pub fn kmeans_f32(data: &[f32], n: usize, d: usize, k: usize, iters: usize, mut seed: u64) -> Vec<f32> {
    let mut rid = |m: usize| {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 11) as usize % m
    };
    // random init: k distinct-ish rows
    let mut cent = vec![0f32; k * d];
    for j in 0..k {
        let id = rid(n);
        cent[j * d..j * d + d].copy_from_slice(&data[id * d..id * d + d]);
    }
    for _ in 0..iters {
        // assign (parallel, the hot step)
        let asg: Vec<u32> = (0..n).into_par_iter().map(|i| nearest(&data[i * d..i * d + d], &cent, k, d)).collect();
        // accumulate sums + counts
        let mut sum = vec![0f64; k * d];
        let mut cnt = vec![0u64; k];
        for i in 0..n {
            let c = asg[i] as usize;
            cnt[c] += 1;
            let row = &data[i * d..i * d + d];
            let acc = &mut sum[c * d..c * d + d];
            for kk in 0..d {
                acc[kk] += row[kk] as f64;
            }
        }
        for j in 0..k {
            if cnt[j] > 0 {
                for kk in 0..d {
                    cent[j * d + kk] = (sum[j * d + kk] / cnt[j] as f64) as f32;
                }
            } else {
                // re-seed empty cluster from a random point
                let id = rid(n);
                cent[j * d..j * d + d].copy_from_slice(&data[id * d..id * d + d]);
            }
        }
    }
    cent
}
