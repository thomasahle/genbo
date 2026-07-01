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

/// Batched int8 L2 over a CONTIGUOUS [ncand x d] centroid block to one query `qn`, writing ncand i32
/// distances into `out`. Beam-descent routing (gather_fine) was a SCALAR per-centroid l2_i8 loop -- which
/// re-ran the avx2 dispatch every centroid and reloaded `qn` every time. This keeps `qn` hot and runs 2
/// centroids/step with independent accumulators (ILP), the dispatch done once via target_feature. The
/// dominant 10M-routing cost (P139: 46% of msspacev QPS@90% query time was routing). Bit-identical to l2_i8.
/// `sd` = number of leading dims actually scored (centroid STRIDE stays `d`). sd<d = APPROXIMATE routing:
/// score only the first sd coords -> cheaper finest-level scoring (#3, routing is the 10M bottleneck). For
/// normalized routing vectors the leading dims carry most energy; recall@p tolerance is measured per-config.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn l2_i8_block_avx2(qn: &[i8], block: &[i8], ncand: usize, d: usize, sd: usize, out: &mut [i32]) {
    let kmax = sd & !15; // largest multiple of 16 within the SCORED prefix
    let mut j = 0usize;
    while j + 2 <= ncand {
        let c0 = block.as_ptr().add(j * d);
        let c1 = block.as_ptr().add((j + 1) * d);
        let mut a0 = _mm256_setzero_si256();
        let mut a1 = _mm256_setzero_si256();
        let mut k = 0usize;
        while k < kmax {
            let x16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(qn.as_ptr().add(k) as *const __m128i));
            let e0 = _mm256_sub_epi16(x16, _mm256_cvtepi8_epi16(_mm_loadu_si128(c0.add(k) as *const __m128i)));
            let e1 = _mm256_sub_epi16(x16, _mm256_cvtepi8_epi16(_mm_loadu_si128(c1.add(k) as *const __m128i)));
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(e0, e0));
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(e1, e1));
            k += 16;
        }
        let mut t0 = [0i32; 8];
        let mut t1 = [0i32; 8];
        _mm256_storeu_si256(t0.as_mut_ptr() as *mut __m256i, a0);
        _mm256_storeu_si256(t1.as_mut_ptr() as *mut __m256i, a1);
        let mut s0 = t0.iter().sum::<i32>();
        let mut s1 = t1.iter().sum::<i32>();
        for k in kmax..sd {
            let q = *qn.get_unchecked(k) as i32;
            let d0 = q - *c0.add(k) as i32; s0 += d0 * d0;
            let d1 = q - *c1.add(k) as i32; s1 += d1 * d1;
        }
        *out.get_unchecked_mut(j) = s0;
        *out.get_unchecked_mut(j + 1) = s1;
        j += 2;
    }
    while j < ncand {
        *out.get_unchecked_mut(j) = l2_i8_avx2(qn, std::slice::from_raw_parts(block.as_ptr().add(j * d), sd));
        j += 1;
    }
}

/// Dispatch wrapper: batched L2 of `qn` against `ncand` contiguous centroids (stride `d`), scoring the
/// first `sd` dims (sd==d => exact full L2) -> `out[..ncand]`.
#[inline]
pub fn l2_i8_block(qn: &[i8], block: &[i8], ncand: usize, d: usize, sd: usize, out: &mut [i32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { l2_i8_block_avx2(qn, block, ncand, d, sd, out) };
            return;
        }
    }
    for j in 0..ncand { out[j] = l2_i8_scalar(qn, &block[j * d..j * d + sd]); }
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

// ---- f16 (IEEE half) rerank support ----------------------------------------------------------
// The streaming rerank cache stores each ACTIVE point's float row as f16 (u16 bits) to HALVE the
// resident cache (max_pts*d*2 vs *4): at the 30M final_runbook (max_pts=10.29M, d=100) that is
// ~2.06GB vs ~4.12GB, which is what brings the peak anon under the streaming-track 8GB cap. f16 has
// a 10-bit mantissa (~3 decimal digits) -- ample for a d=100 L2 RANKING on msturing embeddings; the
// 1M f16==f32 gate MEASURES that this is recall-neutral. The QUERY stays f32; only the cached base
// rows are f16, unpacked to f32 for the L2 (unpack is exact -- every f16 is representable in f32).

/// Exact f16(bits) -> f32 (lossless). Scalar reference / fallback (ported from the `half` crate).
#[inline]
pub fn f16_to_f32(i: u16) -> f32 {
    if i & 0x7fff == 0 { return f32::from_bits((i as u32) << 16); } // signed zero
    let sign = (i & 0x8000) as u32;
    let exp = (i & 0x7c00) as u32;
    let man = (i & 0x03ff) as u32;
    if exp == 0x7c00 { // Inf / NaN
        let m = if man == 0 { 0x7f80_0000 } else { 0x7fc0_0000 | (man << 13) };
        return f32::from_bits((sign << 16) | m);
    }
    let sign = sign << 16;
    if exp == 0 { // subnormal
        let e = (man as u16).leading_zeros() - 6;
        let exp32 = (127 - 15 - e) << 23;
        let man32 = (man << (14 + e)) & 0x007f_ffff;
        return f32::from_bits(sign | exp32 | man32);
    }
    let unbiased = ((exp as i32) >> 10) - 15;
    let exp32 = ((unbiased + 127) as u32) << 23;
    f32::from_bits(sign | exp32 | (man << 13))
}

/// Round-to-nearest-even f32 -> f16(bits). Scalar PACK fallback (F16C used when present, below).
#[inline]
pub fn f32_to_f16(value: f32) -> u16 {
    let x = value.to_bits();
    let sign = x & 0x8000_0000;
    let exp = x & 0x7f80_0000;
    let man = x & 0x007f_ffff;
    if exp == 0x7f80_0000 { // Inf / NaN
        let nan = if man == 0 { 0 } else { 0x0200 | (man >> 13) };
        return ((sign >> 16) | 0x7c00 | nan) as u16;
    }
    let half_sign = sign >> 16;
    let half_exp = ((exp >> 23) as i32) - 127 + 15;
    if half_exp >= 0x1f { return (half_sign | 0x7c00) as u16; } // overflow -> Inf
    if half_exp <= 0 { // subnormal / underflow
        if 14 - half_exp > 24 { return half_sign as u16; }
        let man = man | 0x0080_0000; // hidden bit
        let mut hm = man >> (14 - half_exp);
        let round_bit = 1u32 << (13 - half_exp);
        if (man & round_bit) != 0 && (man & (3 * round_bit - 1)) != 0 { hm += 1; }
        return (half_sign | hm) as u16;
    }
    let half_exp = (half_exp as u32) << 10;
    let half_man = man >> 13;
    if (man & 0x1000) != 0 && (man & 0x2fff) != 0 {
        ((half_sign | half_exp | half_man) + 1) as u16
    } else {
        (half_sign | half_exp | half_man) as u16
    }
}

/// F16C+AVX pack: 8 f32 -> 8 f16 via `_mm256_cvtps_ph` (IEEE round-to-nearest-even, in hardware).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "f16c,avx")]
unsafe fn pack_f16_f16c(src: &[f32], dst: &mut [u16]) {
    let n = src.len();
    let mut k = 0usize;
    while k + 8 <= n {
        let v = _mm256_loadu_ps(src.as_ptr().add(k));
        let h = _mm256_cvtps_ph::<0>(v); // 0 = round to nearest even
        _mm_storeu_si128(dst.as_mut_ptr().add(k) as *mut __m128i, h);
        k += 8;
    }
    while k < n { *dst.get_unchecked_mut(k) = f32_to_f16(*src.get_unchecked(k)); k += 1; }
}

/// Pack an f32 row into f16 bits (dispatch: F16C in hardware, else scalar round-to-nearest-even).
#[inline]
pub fn pack_f16(src: &[f32], dst: &mut [u16]) {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("f16c") && is_x86_feature_detected!("avx") {
            unsafe { pack_f16_f16c(src, dst) };
            return;
        }
    }
    for k in 0..src.len() { dst[k] = f32_to_f16(src[k]); }
}

/// Scalar L2 between an f32 query and an f16-packed row (row unpacked exactly to f32).
#[inline]
pub fn l2_f16_scalar(q: &[f32], r: &[u16]) -> f32 {
    let mut s = 0.0f32;
    for k in 0..q.len() {
        let e = q[k] - f16_to_f32(r[k]);
        s += e * e;
    }
    s
}

/// F16C+AVX L2: load 8 f16 -> `_mm256_cvtph_ps` -> 8 f32, subtract from the f32 query, square, sum.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "f16c,avx")]
pub unsafe fn l2_f16_f16c(q: &[f32], r: &[u16]) -> f32 {
    let n = q.len();
    let mut acc = _mm256_setzero_ps();
    let mut k = 0usize;
    while k + 8 <= n {
        let hv = _mm_loadu_si128(r.as_ptr().add(k) as *const __m128i); // 8 x u16
        let rf = _mm256_cvtph_ps(hv);                                   // 8 x f32 (exact)
        let qf = _mm256_loadu_ps(q.as_ptr().add(k));
        let e = _mm256_sub_ps(qf, rf);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(e, e));
        k += 8;
    }
    let mut tmp = [0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
    let mut s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    while k < n { let e = q[k] - f16_to_f32(*r.get_unchecked(k)); s += e * e; k += 1; }
    s
}

/// L2 between an f32 query and an f16-packed row (dispatch: F16C in hardware, else scalar).
#[inline]
pub fn l2_f16(q: &[f32], r: &[u16]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("f16c") && is_x86_feature_detected!("avx") {
            return unsafe { l2_f16_f16c(q, r) };
        }
    }
    l2_f16_scalar(q, r)
}

/// Self-test: the F16C L2 kernel must match the scalar f16 L2 (same stored bits) to a tiny epsilon,
/// and hardware pack must agree with the scalar pack. Validates the streaming f16 rerank path.
pub fn selftest_f16(d: usize) -> bool {
    let mut seed = 0x2468_ace0_1357_bdf9u64;
    let mut nf = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 32) as f32 / u32::MAX as f32 - 0.5) * 4.0 }; // ~U(-2,2): normal f16 range
    for _ in 0..64 {
        let q: Vec<f32> = (0..d).map(|_| nf()).collect();
        let src: Vec<f32> = (0..d).map(|_| nf()).collect();
        let mut packed = vec![0u16; d];
        pack_f16(&src, &mut packed);
        // hardware/scalar pack agreement (bit-exact over the normal range)
        for k in 0..d { if packed[k] != f32_to_f16(src[k]) { return false; } }
        // SIMD vs scalar L2 over the SAME stored bits (unpack path)
        let a = l2_f16(&q, &packed);
        let b = l2_f16_scalar(&q, &packed);
        if (a - b).abs() > 1e-2 * (1.0 + b.abs()) { return false; }
    }
    true
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
