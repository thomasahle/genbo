//! Native Rust SIMD kernels (AVX2), replacing the C/Cython ones. Each kernel has a scalar
//! reference used for validation (debug self-test in build) so we don't reintroduce bugs while
//! porting. x86_64 intrinsics are the same `_mm256_*` ops as the C kernels.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Scalar reference: squared L2 between two i8 vectors (sum of (x-c)^2, fits i32 for d<=~5000).
#[inline]
pub fn l2_i8_scalar(x: &[i8], c: &[i8]) -> i32 {
    let mut s = 0i32;
    for k in 0..x.len() {
        let d = x[k] as i32 - c[k] as i32;
        s += d * d;
    }
    s
}

/// AVX2 squared L2: widen 16 i8 -> i16, diff, `madd_epi16(diff,diff)` -> 8 i32 lanes, accumulate.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn l2_i8_avx2(x: &[i8], c: &[i8]) -> i32 {
    let n = x.len();
    let mut acc = _mm256_setzero_si256();
    let mut k = 0usize;
    while k + 16 <= n {
        let xv = _mm_loadu_si128(x.as_ptr().add(k) as *const __m128i);
        let cv = _mm_loadu_si128(c.as_ptr().add(k) as *const __m128i);
        let x16 = _mm256_cvtepi8_epi16(xv); // 16 x i16
        let c16 = _mm256_cvtepi8_epi16(cv);
        let diff = _mm256_sub_epi16(x16, c16);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(diff, diff)); // 8 x i32
        k += 16;
    }
    // horizontal sum of the 8 i32 lanes
    let mut tmp = [0i32; 8];
    _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, acc);
    let mut s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    while k < n {
        let d = x[k] as i32 - c[k] as i32;
        s += d * d;
        k += 1;
    }
    s
}

/// AVX2 int8 dot product: widen 16 i8 -> i16, `madd_epi16(x,c)` -> 8 i32, accumulate. For MIPS rerank.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn dot_i8_avx2(x: &[i8], c: &[i8]) -> i32 {
    let n = x.len();
    let mut acc = _mm256_setzero_si256();
    let mut k = 0usize;
    while k + 16 <= n {
        let xv = _mm_loadu_si128(x.as_ptr().add(k) as *const __m128i);
        let cv = _mm_loadu_si128(c.as_ptr().add(k) as *const __m128i);
        let x16 = _mm256_cvtepi8_epi16(xv);
        let c16 = _mm256_cvtepi8_epi16(cv);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(x16, c16));
        k += 16;
    }
    let mut tmp = [0i32; 8];
    _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, acc);
    let mut s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    while k < n { s += x[k] as i32 * c[k] as i32; k += 1; }
    s
}

/// AVX-512 VNNI int8 dot: vpdpbusd does 4 int8 MACs/lane/instr. dpbusd wants u8*i8, so shift x to
/// u8 via XOR 0x80 (= x+128) and subtract 128*sum(c) to recover the exact signed dot. ~6x fewer
/// instructions than the AVX2 madd path -> faster when the rerank is compute-bound (cache-warm).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub unsafe fn dot_i8_vnni(x: &[i8], c: &[i8]) -> i32 {
    let n = x.len();
    let m80 = _mm512_set1_epi8(0x80u8 as i8);
    let ones = _mm512_set1_epi8(1);
    let mut acc = _mm512_setzero_si512();   // Σ (x+128)·c
    let mut sc = _mm512_setzero_si512();     // Σ c  (via 1·c)
    let mut k = 0usize;
    while k + 64 <= n {
        let xv = _mm512_loadu_si512(x.as_ptr().add(k) as *const __m512i);
        let cv = _mm512_loadu_si512(c.as_ptr().add(k) as *const __m512i);
        let xu = _mm512_xor_si512(xv, m80); // i8 -> u8 (x+128)
        acc = _mm512_dpbusd_epi32(acc, xu, cv);
        sc = _mm512_dpbusd_epi32(sc, ones, cv);
        k += 64;
    }
    let mut s = _mm512_reduce_add_epi32(acc) - 128 * _mm512_reduce_add_epi32(sc);
    while k < n { s += x[k] as i32 * c[k] as i32; k += 1; }
    s
}

/// Opt-in VNNI for the int8 dot (set from SBANN_VNNI). Default off until proven faster than AVX2 on
/// this HW (AVX-512 downclocking can make it slower -- cf. the vpermw scan dead-end).
pub static VNNI_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// int8 dot product (dispatch). Returned as a "distance" via NEGATION so min-heap = max inner product.
#[inline]
pub fn negdot_i8(x: &[i8], c: &[i8]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if VNNI_ON.load(std::sync::atomic::Ordering::Relaxed) {
            return -unsafe { dot_i8_vnni(x, c) };
        }
        if is_x86_feature_detected!("avx2") { return -unsafe { dot_i8_avx2(x, c) }; }
    }
    let mut s = 0i32; for k in 0..x.len() { s += x[k] as i32 * c[k] as i32; } -s
}

/// Self-test: VNNI int8 dot must match the scalar dot.
pub fn selftest_dot(d: usize) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if !(is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512bw")) { return true; }
        let mut seed = 0x1357_2468u64;
        let mut nb = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1); ((seed >> 24) as i32 % 256 - 128) as i8 };
        for _ in 0..32 {
            let x: Vec<i8> = (0..d).map(|_| nb()).collect();
            let c: Vec<i8> = (0..d).map(|_| nb()).collect();
            let scal: i32 = (0..d).map(|k| x[k] as i32 * c[k] as i32).sum();
            if unsafe { dot_i8_vnni(&x, &c) } != scal { return false; }
        }
    }
    true
}

/// Dispatch: AVX2 if available at runtime, else scalar.
#[inline]
pub fn l2_i8(x: &[i8], c: &[i8]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { l2_i8_avx2(x, c) };
        }
    }
    l2_i8_scalar(x, c)
}

/// Mean-center + unit-normalize + quantize to i8*127 (matches Python `_q127`). Routing on these
/// normalized vectors gives better cell quality than raw int8 L2 (validated in the Python stack).
#[inline]
pub fn normalize_i8(x: &[i8], mu: &[f32], out: &mut [i8]) {
    let d = x.len();
    let mut tmp = [0f32; 256];
    let mut nrm = 0f32;
    for k in 0..d {
        let v = x[k] as f32 - mu[k];
        tmp[k] = v;
        nrm += v * v;
    }
    let inv = 127.0 / nrm.sqrt().max(1e-9);
    for k in 0..d {
        out[k] = (tmp[k] * inv).round().clamp(-127.0, 127.0) as i8;
    }
}

/// Argmin of L2 from `x` over a contiguous `[c_count x d]` i8 pivot matrix. Returns (best_j, dist).
#[inline]
pub fn assign_nearest(x: &[i8], pivots: &[i8], d: usize) -> (u32, i32) {
    let cc = pivots.len() / d;
    let mut best = i32::MAX;
    let mut bj = 0u32;
    for j in 0..cc {
        let dist = l2_i8(x, &pivots[j * d..j * d + d]);
        if dist < best {
            best = dist;
            bj = j as u32;
        }
    }
    (bj, best)
}

/// Mean-center + unit-normalize + scale by 127 (f32, no rounding) — matches the i8*127 pivot
/// space so a GEMM q.pivot is consistent with the i8 routing distance.
#[inline]
pub fn normalize_i8_to_f32(x: &[i8], mu: &[f32], out: &mut [f32]) {
    let mut nrm = 0.0f32;
    for k in 0..x.len() {
        let v = x[k] as f32 - mu[k];
        out[k] = v;
        nrm += v * v;
    }
    let inv = 127.0 / nrm.sqrt().max(1e-9);
    for v in out.iter_mut() {
        *v *= inv;
    }
}

/// Mean-center + unit-normalize to f32 (no quantization) — for AVQ routing math.
#[inline]
pub fn norm_f32(x: &[i8], mu: &[f32], out: &mut [f32]) {
    let mut nrm = 0.0f32;
    for k in 0..x.len() {
        let v = x[k] as f32 - mu[k];
        out[k] = v;
        nrm += v * v;
    }
    let inv = 1.0 / nrm.sqrt().max(1e-9);
    for v in out.iter_mut() {
        *v *= inv;
    }
}

#[inline]
pub fn l2_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0;
    for k in 0..a.len() {
        let e = a[k] - b[k];
        s += e * e;
    }
    s
}

#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0;
    for k in 0..a.len() {
        s += a[k] * b[k];
    }
    s
}

/// Assign `x` to its `k` nearest pivots (ascending), writing cell ids into `out[..k]`.
#[inline]
pub fn assign_topk(x: &[i8], pivots: &[i8], d: usize, k: usize, out: &mut [u32]) {
    let cc = pivots.len() / d;
    let mut bd = [i32::MAX; 8];
    for o in out.iter_mut().take(k) {
        *o = 0;
    }
    for j in 0..cc {
        let dist = l2_i8(x, &pivots[j * d..j * d + d]);
        if dist < bd[k - 1] {
            let mut p = k - 1;
            while p > 0 && bd[p - 1] > dist {
                bd[p] = bd[p - 1];
                out[p] = out[p - 1];
                p -= 1;
            }
            bd[p] = dist;
            out[p] = j as u32;
        }
    }
}

/// Self-test: AVX2 path must match the scalar reference bit-for-bit.
pub fn selftest(d: usize) -> bool {
    let mut x = vec![0i8; d];
    let mut c = vec![0i8; d];
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut nextb = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 24) as i32 % 256 - 128) as i8
    };
    for _ in 0..64 {
        for k in 0..d {
            x[k] = nextb();
            c[k] = nextb();
        }
        if l2_i8(&x, &c) != l2_i8_scalar(&x, &c) {
            return false;
        }
    }
    true
}
