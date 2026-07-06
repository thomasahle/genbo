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
        // truncate the query slice to `sd` too, else the scalar tail of l2_i8_avx2 reads past the
        // sd-length block slice (out-of-bounds when sd < d, the ROUTE_SDIM path).
        *out.get_unchecked_mut(j) = l2_i8_avx2(
            std::slice::from_raw_parts(qn.as_ptr(), sd),
            std::slice::from_raw_parts(block.as_ptr().add(j * d), sd));
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

/// Σx² for an i8 vector (exact i32). One-off per query for the VNNI L2 decomposition.
#[inline]
pub fn sqnorm_i8(x: &[i8]) -> i32 { x.iter().map(|&v| { let v = v as i32; v * v }).sum() }

/// Per-centroid constant for the single-chain VNNI L2 (see `l2_i8_block_vnni`): `cadj = Σc² + 256·Σc`.
/// The kernel computes Σ(q+128)·c (dpbusd for the bulk + scalar for the <64 tail) = Σqc + 128·Σc; folding
/// +256·Σc into cadj cancels that offset, so L2 = Σq² + cadj − 2·Σ(q+128)c needs only ONE dpbusd chain.
#[inline]
pub fn cadj_i8(c: &[i8]) -> i32 {
    let cnorm: i32 = c.iter().map(|&v| { let v = v as i32; v * v }).sum();
    let sc: i32 = c.iter().map(|&v| v as i32).sum();
    cnorm + 256 * sc
}

/// AVX-512 VNNI batched int8 L2 over a CONTIGUOUS [ncand x d] centroid block, via the exact decomposition
/// L2 = Σq² + Σc² − 2·<q,c>. Single-chain: `_mm512_dpbusd_epi32((q+128), c)` = Σqc + 128·Σc_hi over the
/// VNNI-covered dims, and the +128 offset is cancelled by the precomputed `cadj[j] = Σc² + 256·Σc_hi`, so
/// there is ONE dpbusd chain + ONE horizontal reduce per centroid (no separate Σc accumulator). The <64 tail
/// dims (d not a mult of 64) are added exactly in scalar. `qnorm` = Σq² (full d). 4 centroids/step for ILP.
/// BIT-IDENTICAL to `l2_i8_block` -> recall-neutral. FULL dim only. P196 routing lever (~1.5x on Zen4).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub unsafe fn l2_i8_block_vnni(qn: &[i8], block: &[i8], cadj: &[i32], ncand: usize, d: usize, qnorm: i32, out: &mut [i32]) {
    let m80 = _mm512_set1_epi8(0x80u8 as i8);
    let kmax = d & !63; // largest multiple of 64
    let qb = qnorm; // Σq²
    let mut j = 0usize;
    // 4-wide ILP: four independent dpbusd accumulator chains hide the ~4-cycle dpbusd latency across the
    // short (d/64 ≈ 3-iteration) accumulation loop.
    while j + 4 <= ncand {
        let c0 = block.as_ptr().add(j * d);
        let c1 = block.as_ptr().add((j + 1) * d);
        let c2 = block.as_ptr().add((j + 2) * d);
        let c3 = block.as_ptr().add((j + 3) * d);
        let mut a0 = _mm512_setzero_si512();
        let mut a1 = _mm512_setzero_si512();
        let mut a2 = _mm512_setzero_si512();
        let mut a3 = _mm512_setzero_si512();
        let mut k = 0usize;
        while k < kmax {
            let xu = _mm512_xor_si512(_mm512_loadu_si512(qn.as_ptr().add(k) as *const __m512i), m80);
            a0 = _mm512_dpbusd_epi32(a0, xu, _mm512_loadu_si512(c0.add(k) as *const __m512i));
            a1 = _mm512_dpbusd_epi32(a1, xu, _mm512_loadu_si512(c1.add(k) as *const __m512i));
            a2 = _mm512_dpbusd_epi32(a2, xu, _mm512_loadu_si512(c2.add(k) as *const __m512i));
            a3 = _mm512_dpbusd_epi32(a3, xu, _mm512_loadu_si512(c3.add(k) as *const __m512i));
            k += 64;
        }
        let mut r0 = _mm512_reduce_add_epi32(a0);
        let mut r1 = _mm512_reduce_add_epi32(a1);
        let mut r2 = _mm512_reduce_add_epi32(a2);
        let mut r3 = _mm512_reduce_add_epi32(a3);
        for kk in kmax..d {
            let qk = *qn.get_unchecked(kk) as i32 + 128; // (q+128)·c matches dpbusd's u8·i8; cadj folds the 128·Σc back
            r0 += qk * *c0.add(kk) as i32;
            r1 += qk * *c1.add(kk) as i32;
            r2 += qk * *c2.add(kk) as i32;
            r3 += qk * *c3.add(kk) as i32;
        }
        *out.get_unchecked_mut(j) = qb + *cadj.get_unchecked(j) - 2 * r0;
        *out.get_unchecked_mut(j + 1) = qb + *cadj.get_unchecked(j + 1) - 2 * r1;
        *out.get_unchecked_mut(j + 2) = qb + *cadj.get_unchecked(j + 2) - 2 * r2;
        *out.get_unchecked_mut(j + 3) = qb + *cadj.get_unchecked(j + 3) - 2 * r3;
        j += 4;
    }
    while j < ncand {
        let cptr = block.as_ptr().add(j * d);
        let mut a0 = _mm512_setzero_si512();
        let mut k = 0usize;
        while k < kmax {
            let xu = _mm512_xor_si512(_mm512_loadu_si512(qn.as_ptr().add(k) as *const __m512i), m80);
            a0 = _mm512_dpbusd_epi32(a0, xu, _mm512_loadu_si512(cptr.add(k) as *const __m512i));
            k += 64;
        }
        let mut r0 = _mm512_reduce_add_epi32(a0);
        for kk in kmax..d { r0 += (*qn.get_unchecked(kk) as i32 + 128) * *cptr.add(kk) as i32; }
        *out.get_unchecked_mut(j) = qb + *cadj.get_unchecked(j) - 2 * r0;
        j += 1;
    }
}

/// Dispatch: VNNI norm-decomposition L2 block if avx512vnni present, else the AVX2 madd block. Bit-identical.
/// `cadj[..ncand]` = per-centroid `cadj_i8`, `qnorm` = Σq². Caller guarantees full-dim scoring (sd==d).
#[inline]
pub fn l2_i8_block_norm(qn: &[i8], block: &[i8], cadj: &[i32], ncand: usize, d: usize, qnorm: i32, out: &mut [i32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512f") {
            unsafe { l2_i8_block_vnni(qn, block, cadj, ncand, d, qnorm, out) };
            return;
        }
    }
    l2_i8_block(qn, block, ncand, d, d, out);
}

/// Self-test: VNNI norm-decomposition L2 block must match the AVX2 madd block bit-for-bit.
pub fn selftest_l2_norm(d: usize) -> bool {
    let ncand = 39usize; // exercise the 4-wide body + 3-lane scalar remainder
    let mut seed = 0x0bad_c0deu64;
    let mut nb = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1); ((seed >> 24) as i32 % 256 - 128) as i8 };
    let q: Vec<i8> = (0..d).map(|_| nb()).collect();
    let block: Vec<i8> = (0..ncand * d).map(|_| nb()).collect();
    let cadj: Vec<i32> = (0..ncand).map(|j| cadj_i8(&block[j * d..j * d + d])).collect();
    let qnorm = sqnorm_i8(&q);
    let mut a = vec![0i32; ncand];
    let mut b = vec![0i32; ncand];
    l2_i8_block(&q, &block, ncand, d, d, &mut a);
    l2_i8_block_norm(&q, &block, &cadj, ncand, d, qnorm, &mut b);
    a == b
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
    let mut tmp = [0f32; 1024];
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

/// AVX2+FMA f32 inner product (4 independent accumulators for ILP). Used by the FLOAT-RERANK path
/// (P191 lever stack) — the scalar `dot_f32` reduction doesn't vectorize (float assoc), so this is
/// ~8x on the hot float dot. Falls back to scalar off-AVX2.
#[inline]
pub fn dot_f32_fast(a: &[f32], b: &[f32]) -> f32 {
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        unsafe { dot_f32_avx2(a, b) }
    } else {
        dot_f32(a, b)
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 32 <= n {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 8)), _mm256_loadu_ps(pb.add(i + 8)), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 16)), _mm256_loadu_ps(pb.add(i + 16)), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 24)), _mm256_loadu_ps(pb.add(i + 24)), acc3);
        i += 32;
    }
    while i + 8 <= n {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
        i += 8;
    }
    let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    // horizontal sum of the 8 lanes
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    let mut sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    sum128 = _mm_add_ss(sum128, _mm_shuffle_ps(sum128, sum128, 1));
    let mut s = _mm_cvtss_f32(sum128);
    while i < n { s += *pa.add(i) * *pb.add(i); i += 1; }
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
