//! Pluggable vector-quantization engine. Two trait slots — Router (coarse: which cells to scan)
//! and Compressor (candidate scan: approx distances) — composed by `Index`. Swap PQ/AQ/OPQ/scalar
//! for either slot at runtime via `Box<dyn ..>`; dispatch is per 16-point block, so no hot-loop cost.

use crate::ibin::I8Bin;
use crate::{kmeans, pq, simd};
use rayon::prelude::*;
use std::arch::x86_64::{__m128i, __m512i, _mm_prefetch, _MM_HINT_T0};

/// Global rerank mode: false = exact L2 (default), true = exact inner product (MIPS, via -dot so a
/// min-heap keeps the MAX inner product). Set once at startup from SBANN_IP (for the OOD/cosine path).
pub static IP_MODE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// true => i8 scan (1 vpshufb/subspace) instead of int16 (2/subspace): faster, coarser. A global
/// (not per-query env) so run() can A/B scan precision on one index. Set from SBANN_NOLUT16.
pub static LUT16_OFF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// true => FAST-SCAN: int8-LUT, 1 vpshufb/subspace + int16 accumulation (~1.7x scan vs int16 LUT16) at
/// ~12-13 bit ranking resolution (vs int16's 15, the sqrt(m) i8 path's 8). Set from SBANN_FASTSCAN.
pub static FASTSCAN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Batched exact-int8 rerank: detect AVX2 once, call it directly (no per-survivor dispatch),
/// PREFETCH the scattered survivor gathers (the real cost — random reads into the base), and keep
/// a bounded top-heap instead of sorting all T survivors. Survivors = (approx_dist, orig_id);
/// returns up to k DISTINCT ids by true L2. This is the QPS-critical kernel.
fn rerank_survivors(ds: &I8Bin, q: &[i8], pool: &[(i32, u32)], k: usize) -> Vec<u32> {
    let m = (k * 4).min(pool.len());
    if m == 0 {
        return Vec::new();
    }
    let avx = std::is_x86_feature_detected!("avx2");
    let n = pool.len();
    let mut heap: std::collections::BinaryHeap<(i32, u32)> = std::collections::BinaryHeap::with_capacity(m + 1);
    for i in 0..n {
        let id = pool[i].1 as usize;
        if i + 8 < n {
            unsafe { _mm_prefetch(ds.row(pool[i + 8].1 as usize).as_ptr() as *const i8, _MM_HINT_T0) };
        }
        let row = ds.row(id);
        let dist = if avx { unsafe { simd::l2_i8_avx2(q, row) } } else { simd::l2_i8_scalar(q, row) };
        if heap.len() < m {
            heap.push((dist, pool[i].1));
        } else if dist < heap.peek().unwrap().0 {
            heap.pop();
            heap.push((dist, pool[i].1));
        }
    }
    let mut v = heap.into_vec();
    v.sort_unstable();
    let mut out = Vec::with_capacity(k);
    for &(_, id) in &v {
        if !out.contains(&id) {
            out.push(id);
            if out.len() == k {
                break;
            }
        }
    }
    out
}

/// Like rerank_survivors but survivors are SLOTS into a cell-contiguous raw i8 array (`raw`), so the
/// gathers stay inside the small probed-cell region (cache-warm) instead of scattering across the
/// full base. Pool = (approx_dist, slot); returns up to k DISTINCT orig ids by true L2.
fn rerank_contig(raw: &[i8], d: usize, slot_orig: &[u32], q: &[i8], pool: &[(i32, u32)], k: usize) -> Vec<u32> {
    let m = (k * 4).min(pool.len());
    if m == 0 {
        return Vec::new();
    }
    let avx = std::is_x86_feature_detected!("avx2");
    let ip = IP_MODE.load(std::sync::atomic::Ordering::Relaxed);
    let n = pool.len();
    let mut heap: std::collections::BinaryHeap<(i32, u32)> = std::collections::BinaryHeap::with_capacity(m + 1);
    for i in 0..n {
        let slot = pool[i].1 as usize;
        if i + 8 < n {
            unsafe { _mm_prefetch(raw.as_ptr().add(pool[i + 8].1 as usize * d) as *const i8, _MM_HINT_T0) };
        }
        let row = &raw[slot * d..slot * d + d];
        let dist = if ip { simd::negdot_i8(q, row) } else if avx { unsafe { simd::l2_i8_avx2(q, row) } } else { simd::l2_i8_scalar(q, row) };
        if heap.len() < m {
            heap.push((dist, pool[i].1));
        } else if dist < heap.peek().unwrap().0 {
            heap.pop();
            heap.push((dist, pool[i].1));
        }
    }
    let mut v = heap.into_vec();
    v.sort_unstable();
    let mut out = Vec::with_capacity(k);
    for &(_, slot) in &v {
        let id = slot_orig[slot as usize];
        if id != u32::MAX && !out.contains(&id) {
            out.push(id);
            if out.len() == k {
                break;
            }
        }
    }
    out
}

// ---------------- Router: coarse quantizer (which cells) ----------------
pub trait Router: Send + Sync {
    fn n_cells(&self) -> usize;
    fn assign(&self, row: &[i8], a0: usize, out: &mut Vec<u32>); // build: point -> a0 cells
    fn probe(&self, q: &[i8], p: usize) -> Vec<u32>; // query: top-p cells
    /// top-p cells sorted NEAREST-FIRST (for adaptive early termination). Default: unranked probe.
    fn probe_ranked(&self, q: &[i8], p: usize) -> Vec<u32> { self.probe(q, p) }
    /// Batched routing: top-p cells for all nq queries (nq*p). Default: parallel per-query probe;
    /// FlatIvf overrides with a single GEMM (Q @ pivots^T) — far faster for large C.
    fn probe_batch(&self, q_i8: &[i8], nq: usize, d: usize, p: usize) -> Vec<u32> {
        let mut out = vec![0u32; nq * p];
        out.par_chunks_mut(p).enumerate().for_each(|(i, slot)| {
            let cells = self.probe(&q_i8[i * d..i * d + d], p);
            for (s, c) in slot.iter_mut().zip(cells.iter()) { *s = *c; }
        });
        out
    }
}

/// Flat IVF: single codebook of `c` pivots (k-means or random), normalized-L2 routing.
/// `soar` > 0 enables SOAR: the SECOND (a0=2) assignment minimizes an orthogonality-amplified
/// residual loss so it covers directions the first cell's residual missed (ScaNN SOAR).
pub struct FlatIvf {
    d: usize,
    c: usize,
    pivots: Vec<i8>,
    pivots_f32: Vec<f32>,
    cnorm: Vec<f32>, // ||pivot||^2 in f32 space, for the GEMM route (||q-c||^2 = 1 + ||c||^2 - 2 q.c)
    mu: Vec<f32>,
    soar: f32,
    rair: bool, // true = RAIR inverse-residual (signed r0·rj); false = SOAR orthogonal (proj^2)
}

impl FlatIvf {
    /// Batched routing via f32 GEMM: normalize all queries, G = Qn @ pivots^T (matrixmultiply,
    /// blocked SIMD), score = cnorm - 2*G, top-p per query. Returns nq*p cell ids. Replaces the
    /// per-query per-pivot scalar loop — the QPS-critical routing path.
    fn probe_batch_gemm(&self, q_i8: &[i8], nq: usize, p: usize) -> Vec<u32> {
        let (d, c) = (self.d, self.c);
        let mut qn = vec![0f32; nq * d];
        qn.par_chunks_mut(d).enumerate().for_each(|(i, r)| simd::normalize_i8_to_f32(&q_i8[i * d..i * d + d], &self.mu, r));
        let mut g = vec![0f32; nq * c]; // G = -2 * Qn @ pivots^T
        unsafe {
            // a = Qn (nq x d, row-major); b = pivots^T (d x c) i.e. pivots (c x d) with swapped strides
            matrixmultiply::sgemm(
                nq, d, c,
                -2.0,
                qn.as_ptr(), d as isize, 1,
                self.pivots_f32.as_ptr(), 1, d as isize,
                0.0,
                g.as_mut_ptr(), c as isize, 1,
            );
        }
        let pp = p.min(c);
        let mut out = vec![0u32; nq * pp];
        out.par_chunks_mut(pp).enumerate().for_each(|(i, slot)| {
            let row = &g[i * c..i * c + c];
            let mut sc: Vec<(f32, u32)> = (0..c).map(|j| (row[j] + self.cnorm[j], j as u32)).collect();
            sc.select_nth_unstable_by(pp - 1, |a, b| a.0.total_cmp(&b.0));
            for (s, &(_, j)) in slot.iter_mut().zip(sc[..pp].iter()) { *s = j; }
        });
        out
    }
}

impl FlatIvf {
    pub fn train_soar(ds: &I8Bin, c: usize, mu: Vec<f32>, kmeans_iters: usize, soar: f32) -> Self {
        let mut s = Self::train(ds, c, mu, kmeans_iters);
        s.soar = soar;
        s
    }
    /// RAIR (arXiv 2601.07183): second assignment minimizes ‖rj‖² + λ·(r0·rj) -> prefers the
    /// second residual ANTI-parallel to the first (centroids bracket the point).
    pub fn train_rair(ds: &I8Bin, c: usize, mu: Vec<f32>, kmeans_iters: usize, lam: f32) -> Self {
        let mut s = Self::train(ds, c, mu, kmeans_iters);
        s.soar = lam;
        s.rair = true;
        s
    }
    pub fn train(ds: &I8Bin, c: usize, mu: Vec<f32>, kmeans_iters: usize) -> Self {
        let (n, d) = (ds.nb, ds.d);
        // normalize a training sample to f32, k-means, quantize centroids to i8*127.
        // sample ~256 pts/centroid (faiss rule) so k-means isn't undertrained at scale.
        let smp = n.min((c * 64).max(200_000));
        let mut xn = vec![0f32; smp * d];
        xn.par_chunks_mut(d).enumerate().for_each(|(i, r)| simd::norm_f32(ds.row(i), &mu, r));
        let cent = if kmeans_iters > 0 {
            kmeans::kmeans_f32(&xn, smp, d, c, kmeans_iters, 0xf1a7)
        } else {
            // random rows
            let mut v = vec![0f32; c * d];
            let mut s = 0x9e37u64;
            for j in 0..c { s = s.wrapping_mul(6364136223846793005).wrapping_add(1); let id = (s >> 11) as usize % smp; v[j * d..j * d + d].copy_from_slice(&xn[id * d..id * d + d]); }
            v
        };
        let pivots: Vec<i8> = cent.iter().map(|&v| (v * 127.0).round().clamp(-127.0, 127.0) as i8).collect();
        let pivots_f32: Vec<f32> = pivots.iter().map(|&v| v as f32).collect();
        let cnorm: Vec<f32> = (0..c).map(|j| pivots_f32[j * d..j * d + d].iter().map(|&v| v * v).sum()).collect();
        FlatIvf { d, c, pivots, pivots_f32, cnorm, mu, soar: 0.0, rair: false }
    }
}

impl Router for FlatIvf {
    fn n_cells(&self) -> usize { self.c }
    fn assign(&self, row: &[i8], a0: usize, out: &mut Vec<u32>) {
        let d = self.d;
        let mut qn = [0i8; 256];
        simd::normalize_i8(row, &self.mu, &mut qn[..d]);
        if self.soar > 0.0 && a0 == 2 {
            // i0 = nearest; i1 = argmin l2 + soar*<residual_j, r0hat>^2 (orthogonality-amplified)
            let mut l2v = vec![0f32; self.c];
            let (mut bi0, mut b0) = (0usize, f32::INFINITY);
            for j in 0..self.c {
                let dist = simd::l2_i8(&qn[..d], &self.pivots[j * d..j * d + d]) as f32;
                l2v[j] = dist;
                if dist < b0 { b0 = dist; bi0 = j; }
            }
            // first residual r0 = qn - pivot[i0]
            let mut r0 = [0f32; 256];
            let mut nrm = 0.0f32;
            for k in 0..d { let v = qn[k] as f32 - self.pivots_f32[bi0 * d + k]; r0[k] = v; nrm += v * v; }
            let (mut bi1, mut b1) = (bi0, f32::INFINITY);
            if self.rair {
                // RAIR: loss = ‖rj‖² + λ (r0·rj). r0·rj = r0·qn - r0·pivot[j]; r0·qn const -> drop.
                for j in 0..self.c {
                    if j == bi0 { continue; }
                    let r0pj: f32 = (0..d).map(|k| r0[k] * self.pivots_f32[j * d + k]).sum();
                    let loss = l2v[j] + self.soar * (-r0pj); // smaller when rj anti-parallel to r0
                    if loss < b1 { b1 = loss; bi1 = j; }
                }
            } else {
                // SOAR: orthogonal — penalize the squared parallel projection onto r̂0
                let inv = 1.0 / nrm.sqrt().max(1e-9);
                for k in 0..d { r0[k] *= inv; }
                let qdot: f32 = (0..d).map(|k| qn[k] as f32 * r0[k]).sum();
                for j in 0..self.c {
                    if j == bi0 { continue; }
                    let pdot: f32 = (0..d).map(|k| self.pivots_f32[j * d + k] * r0[k]).sum();
                    let proj = qdot - pdot;
                    let loss = l2v[j] + self.soar * proj * proj;
                    if loss < b1 { b1 = loss; bi1 = j; }
                }
            }
            out.clear();
            out.push(bi0 as u32);
            out.push(bi1 as u32);
        } else {
            out.resize(a0, 0);
            simd::assign_topk(&qn[..d], &self.pivots, d, a0, out);
        }
    }
    fn probe(&self, q: &[i8], p: usize) -> Vec<u32> {
        let d = self.d;
        let mut qn = [0i8; 256];
        simd::normalize_i8(q, &self.mu, &mut qn[..d]);
        let mut cd: Vec<(i32, u32)> = (0..self.c)
            .map(|j| (simd::l2_i8(&qn[..d], &self.pivots[j * d..j * d + d]), j as u32))
            .collect();
        let p = p.min(cd.len());
        cd.select_nth_unstable(p - 1);
        cd[..p].iter().map(|&(_, j)| j).collect()
    }
    fn probe_batch(&self, q_i8: &[i8], nq: usize, _d: usize, p: usize) -> Vec<u32> {
        self.probe_batch_gemm(q_i8, nq, p)
    }
    fn probe_ranked(&self, q: &[i8], p: usize) -> Vec<u32> {
        let d = self.d;
        let mut qn = [0i8; 256];
        simd::normalize_i8(q, &self.mu, &mut qn[..d]);
        let mut cd: Vec<(i32, u32)> = (0..self.c)
            .map(|j| (simd::l2_i8(&qn[..d], &self.pivots[j * d..j * d + d]), j as u32))
            .collect();
        let p = p.min(cd.len());
        cd.select_nth_unstable(p - 1);
        cd.truncate(p);
        cd.sort_unstable(); // nearest-first
        cd.iter().map(|&(_, j)| j).collect()
    }
}

/// Additive 2-codebook multi-index (AQ routing). Buckets = (i0,i1); k-means codebooks (balanced).
pub struct AvqRouter {
    d: usize,
    c0n: usize,
    c1n: usize,
    c0: Vec<f32>,
    c1: Vec<f32>,
    c0sq: Vec<f32>,
    c1sq: Vec<f32>,
    cross: Vec<f32>, // c0n x c1n
    mu: Vec<f32>,
}

impl AvqRouter {
    pub fn train(ds: &I8Bin, c0n: usize, c1n: usize, mu: Vec<f32>, iters: usize) -> Self {
        let (n, d) = (ds.nb, ds.d);
        let smp = n.min(200_000);
        let mut xn = vec![0f32; smp * d];
        xn.par_chunks_mut(d).enumerate().for_each(|(i, r)| simd::norm_f32(ds.row(i), &mu, r));
        let c0 = kmeans::kmeans_f32(&xn, smp, d, c0n, iters, 0xc0c0);
        // residuals of the sample wrt nearest c0
        let res: Vec<f32> = (0..smp * d).into_par_iter().map(|p| {
            let i = p / d;
            let x = &xn[i * d..i * d + d];
            let i0 = (0..c0n).map(|j| (simd::l2_f32(x, &c0[j * d..j * d + d]), j)).min_by(|a, b| a.0.total_cmp(&b.0)).unwrap().1;
            xn[p] - c0[i0 * d + p % d]
        }).collect();
        let c1 = kmeans::kmeans_f32(&res, smp, d, c1n, iters, 0xc1c1);
        let c0sq: Vec<f32> = (0..c0n).map(|j| simd::dot_f32(&c0[j * d..j * d + d], &c0[j * d..j * d + d])).collect();
        let c1sq: Vec<f32> = (0..c1n).map(|j| simd::dot_f32(&c1[j * d..j * d + d], &c1[j * d..j * d + d])).collect();
        let cross: Vec<f32> = (0..c0n).into_par_iter().flat_map(|a| (0..c1n).map(|b| simd::dot_f32(&c0[a * d..a * d + d], &c1[b * d..b * d + d])).collect::<Vec<_>>()).collect();
        AvqRouter { d, c0n, c1n, c0, c1, c0sq, c1sq, cross, mu }
    }
    #[inline]
    fn ab(&self, q: &[i8]) -> (Vec<f32>, Vec<f32>) {
        let d = self.d;
        let mut qn = vec![0f32; d];
        simd::norm_f32(q, &self.mu, &mut qn);
        let a: Vec<f32> = (0..self.c0n).map(|j| -2.0 * simd::dot_f32(&qn, &self.c0[j * d..j * d + d]) + self.c0sq[j]).collect();
        let b: Vec<f32> = (0..self.c1n).map(|j| -2.0 * simd::dot_f32(&qn, &self.c1[j * d..j * d + d]) + self.c1sq[j]).collect();
        (a, b)
    }
}

impl Router for AvqRouter {
    fn n_cells(&self) -> usize { self.c0n * self.c1n }
    fn assign(&self, row: &[i8], a0: usize, out: &mut Vec<u32>) {
        // cheap RQ assignment: top-a0 nearest c0, each with nearest c1-of-residual. O(c0+a0*c1)
        // per point instead of O(c0*c1) — makes the BUILD scale (vs scanning all 65536 cells).
        let d = self.d;
        let mut qn = [0f32; 256];
        simd::norm_f32(row, &self.mu, &mut qn[..d]);
        let mut c0d: Vec<(f32, u32)> = (0..self.c0n)
            .map(|j| (simd::l2_f32(&qn[..d], &self.c0[j * d..j * d + d]), j as u32))
            .collect();
        let aa = a0.min(self.c0n);
        c0d.select_nth_unstable_by(aa - 1, |x, y| x.0.total_cmp(&y.0));
        out.clear();
        for &(_, i0) in c0d[..aa].iter() {
            let off0 = i0 as usize * d;
            let mut best = f32::INFINITY;
            let mut bi1 = 0u32;
            for j in 0..self.c1n {
                let off1 = j * d;
                let mut dd = 0f32;
                for k in 0..d {
                    let r = qn[k] - self.c0[off0 + k] - self.c1[off1 + k];
                    dd += r * r;
                }
                if dd < best { best = dd; bi1 = j as u32; }
            }
            out.push(i0 * self.c1n as u32 + bi1);
        }
    }
    fn probe(&self, q: &[i8], p: usize) -> Vec<u32> {
        let (a, b) = self.ab(q);
        let nc = self.c0n * self.c1n;
        let mut dcell: Vec<(f32, u32)> = (0..nc).map(|c| (a[c / self.c1n] + b[c % self.c1n] + 2.0 * self.cross[c], c as u32)).collect();
        let p = p.min(nc);
        dcell.select_nth_unstable_by(p - 1, |x, y| x.0.total_cmp(&y.0));
        dcell[..p].iter().map(|&(_, c)| c).collect()
    }
}

/// Hierarchical 2-level router (the score-tree): Kf fine cells grouped under C0 coarse centroids.
/// Routing is O(C0 + b0·Kf/C0) ≈ O(√Kf) instead of flat O(Kf) — this is what makes the BUILD
/// scale to 100M/1B (flat assign is O(n·Kf), prohibitive). Random centroids (P32: ~k-means for
/// IVF since exact rerank fixes ranking); the finer Kf compensates for random coarseness.
pub struct HierRouter {
    d: usize,
    kf: usize,
    c0n: usize,
    cf: Vec<i8>,      // kf*d fine centroids, REORDERED so each coarse's (or mid's) fines are contiguous
    c0: Vec<i8>,      // c0n*d coarse centroids
    gstart: Vec<u32>, // c0n+1: for 2-level, fine range per coarse; for 3-level, MID range per coarse
    mu: Vec<f32>,
    b0: usize,        // coarse cells expanded per query/assign
    // --- 3-level extension (empty c1 => 2-level). routing O(Kf^1/3) instead of O(Kf^1/2). ---
    c1: Vec<i8>,      // c1n*d MID centroids, grouped by coarse (range = gstart)
    g1start: Vec<u32>,// c1n+1: contiguous fine range per MID
    b1: usize,        // mid cells expanded per query
}

impl HierRouter {
    /// Hierarchical k-means: k-means the C0 coarse centroids, then k-means `Kf/C0` fine centroids
    /// WITHIN each coarse cell's points. Gives k-means-quality routing at O(n·Kf/C0) train cost
    /// (vs flat k-means O(n·Kf)). This is the scale lever (P58) — replaces random fine centroids.
    pub fn train_hkmeans(ds: &I8Bin, kf: usize, c0n: usize, b0: usize, mu: Vec<f32>) -> Self {
        let (n, d) = (ds.nb, ds.d);
        let smp = n.min(c0n * 400 + kf * 6);
        let mut xn = vec![0f32; smp * d];
        xn.par_chunks_mut(d).enumerate().for_each(|(i, r)| simd::norm_f32(ds.row(i), &mu, r));
        // coarse k-means
        let c0f = kmeans::kmeans_f32(&xn, smp, d, c0n, 12, 0xc0a1_5eed);
        let c0: Vec<i8> = c0f.iter().map(|&v| (v * 127.0).round().clamp(-127.0, 127.0) as i8).collect();
        // assign sample to nearest coarse
        let asg: Vec<u32> = (0..smp).into_par_iter().map(|i| {
            let x = &xn[i * d..i * d + d];
            (0..c0n).map(|q| (simd::l2_f32(x, &c0f[q * d..q * d + d]), q as u32)).min_by(|a, b| a.0.total_cmp(&b.0)).unwrap().1
        }).collect();
        let mut by_coarse: Vec<Vec<u32>> = vec![Vec::new(); c0n];
        for i in 0..smp { by_coarse[asg[i] as usize].push(i as u32); }
        let fpc = (kf / c0n).max(1);
        let kf2 = fpc * c0n;
        // per-coarse fine k-means (parallel over coarse cells)
        let cells: Vec<Vec<i8>> = (0..c0n).into_par_iter().map(|q| {
            let idx = &by_coarse[q];
            let m = idx.len().max(1);
            let mut pts = vec![0f32; m * d];
            for (j, &pi) in idx.iter().enumerate() { pts[j * d..j * d + d].copy_from_slice(&xn[pi as usize * d..pi as usize * d + d]); }
            let fine = kmeans::kmeans_f32(&pts, m, d, fpc, 8, 0xf14e_0000 ^ q as u64);
            fine.iter().map(|&v| (v * 127.0).round().clamp(-127.0, 127.0) as i8).collect()
        }).collect();
        let mut cf = vec![0i8; kf2 * d];
        let mut gstart = vec![0u32; c0n + 1];
        for q in 0..c0n {
            gstart[q + 1] = gstart[q] + fpc as u32;
            cf[gstart[q] as usize * d..(gstart[q] as usize + fpc) * d].copy_from_slice(&cells[q]);
        }
        HierRouter { d, kf: kf2, c0n, cf, c0, gstart, mu, b0, c1: Vec::new(), g1start: Vec::new(), b1: 0 }
    }

    /// 3-level hierarchical k-means: coarse C0 -> MID C1 (c1n total) -> fine Kf. Routing becomes
    /// O(C0 + b0·C1/C0 + b1·Kf/C1) ≈ O(Kf^1/3) centroid-distances/query instead of O(Kf^1/2),
    /// cutting the (compute-bound) routing phase. gstart = coarse->mid range, g1start = mid->fine range.
    pub fn train_hkmeans3(ds: &I8Bin, kf: usize, c0n: usize, c1n: usize, b0: usize, b1: usize, mu: Vec<f32>) -> Self {
        let (n, d) = (ds.nb, ds.d);
        let smp = n.min(c0n * 200 + c1n * 40 + kf * 6);
        let mut xn = vec![0f32; smp * d];
        xn.par_chunks_mut(d).enumerate().for_each(|(i, r)| simd::norm_f32(ds.row(i), &mu, r));
        // coarse k-means
        let c0f = kmeans::kmeans_f32(&xn, smp, d, c0n, 12, 0xc0a1_5eed);
        let c0: Vec<i8> = c0f.iter().map(|&v| (v * 127.0).round().clamp(-127.0, 127.0) as i8).collect();
        let asg: Vec<u32> = (0..smp).into_par_iter().map(|i| {
            let x = &xn[i * d..i * d + d];
            (0..c0n).map(|q| (simd::l2_f32(x, &c0f[q * d..q * d + d]), q as u32)).min_by(|a, b| a.0.total_cmp(&b.0)).unwrap().1
        }).collect();
        let mut by_coarse: Vec<Vec<u32>> = vec![Vec::new(); c0n];
        for i in 0..smp { by_coarse[asg[i] as usize].push(i as u32); }
        let mpc = (c1n / c0n).max(1);      // mids per coarse
        let fpm = (kf / (mpc * c0n)).max(1); // fines per mid
        // per-coarse: mid k-means, assign coarse's points to mids, then per-mid fine k-means.
        // returns (mids_i8 [mpc*d], fines_i8 [mpc*fpm*d] grouped by mid).
        let cells: Vec<(Vec<i8>, Vec<i8>)> = (0..c0n).into_par_iter().map(|q| {
            let idx = &by_coarse[q];
            let m = idx.len().max(1);
            let mut pts = vec![0f32; m * d];
            for (j, &pi) in idx.iter().enumerate() { pts[j * d..j * d + d].copy_from_slice(&xn[pi as usize * d..pi as usize * d + d]); }
            // mid k-means
            let midf = kmeans::kmeans_f32(&pts, m, d, mpc, 8, 0x111d_0000 ^ q as u64);
            let mids: Vec<i8> = midf.iter().map(|&v| (v * 127.0).round().clamp(-127.0, 127.0) as i8).collect();
            // assign points to mids
            let masg: Vec<u32> = (0..m).map(|i| {
                let x = &pts[i * d..i * d + d];
                (0..mpc).map(|mm| (simd::l2_f32(x, &midf[mm * d..mm * d + d]), mm as u32)).min_by(|a, b| a.0.total_cmp(&b.0)).unwrap().1
            }).collect();
            let mut by_mid: Vec<Vec<u32>> = vec![Vec::new(); mpc];
            for i in 0..m { by_mid[masg[i] as usize].push(i as u32); }
            // per-mid fine k-means, contiguous by mid
            let mut fines = vec![0i8; mpc * fpm * d];
            for mm in 0..mpc {
                let mi = &by_mid[mm];
                let mm2 = mi.len().max(1);
                let mut fpts = vec![0f32; mm2 * d];
                for (j, &pi) in mi.iter().enumerate() { fpts[j * d..j * d + d].copy_from_slice(&pts[pi as usize * d..pi as usize * d + d]); }
                let fine = kmeans::kmeans_f32(&fpts, mm2, d, fpm, 6, 0xf14e_3000 ^ ((q as u64) << 8) ^ mm as u64);
                let base = mm * fpm * d;
                for (j, &v) in fine.iter().enumerate() { fines[base + j] = (v * 127.0).round().clamp(-127.0, 127.0) as i8; }
            }
            (mids, fines)
        }).collect();
        // assemble: mids grouped by coarse (gstart), fines grouped by mid (g1start)
        let c1n2 = mpc * c0n;
        let kf3 = fpm * c1n2;
        let mut c1 = vec![0i8; c1n2 * d];
        let mut cf = vec![0i8; kf3 * d];
        let mut gstart = vec![0u32; c0n + 1];
        let mut g1start = vec![0u32; c1n2 + 1];
        for q in 0..c0n {
            gstart[q + 1] = gstart[q] + mpc as u32;
            let mbase = gstart[q] as usize; // first mid index of this coarse
            c1[mbase * d..(mbase + mpc) * d].copy_from_slice(&cells[q].0);
            for mm in 0..mpc {
                let gmid = mbase + mm;
                g1start[gmid + 1] = g1start[gmid] + fpm as u32;
                let fbase = g1start[gmid] as usize;
                cf[fbase * d..(fbase + fpm) * d].copy_from_slice(&cells[q].1[mm * fpm * d..(mm + 1) * fpm * d]);
            }
        }
        HierRouter { d, kf: kf3, c0n, cf, c0, gstart, mu, b0, c1, g1start, b1 }
    }

    pub fn train(ds: &I8Bin, kf: usize, c0n: usize, b0: usize, mu: Vec<f32>) -> Self {
        let (n, d) = (ds.nb, ds.d);
        // random normalized fine centroids
        let mut seed = 0x77c0_ffeeu64;
        let mut rid = |m: usize| { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1); (seed >> 11) as usize % m };
        let mut cf0 = vec![0i8; kf * d];
        for j in 0..kf { let id = rid(n); simd::normalize_i8(ds.row(id), &mu, &mut cf0[j * d..j * d + d]); }
        // coarse centroids = sample of the fine centroids
        let mut c0 = vec![0i8; c0n * d];
        for j in 0..c0n { let id = rid(kf); c0[j * d..j * d + d].copy_from_slice(&cf0[id * d..id * d + d]); }
        // assign each fine centroid to nearest coarse, then reorder fines contiguous by coarse
        let f2c: Vec<u32> = (0..kf).into_par_iter()
            .map(|j| (0..c0n).map(|q| (simd::l2_i8(&cf0[j * d..j * d + d], &c0[q * d..q * d + d]), q as u32)).min_by_key(|&(dist, _)| dist).unwrap().1)
            .collect();
        let mut order: Vec<u32> = (0..kf as u32).collect();
        order.sort_by_key(|&j| f2c[j as usize]);
        let mut counts = vec![0u32; c0n];
        for &c in &f2c { counts[c as usize] += 1; }
        let mut gstart = vec![0u32; c0n + 1];
        for q in 0..c0n { gstart[q + 1] = gstart[q] + counts[q]; }
        let mut cf = vec![0i8; kf * d];
        for (newpos, &oldj) in order.iter().enumerate() {
            cf[newpos * d..newpos * d + d].copy_from_slice(&cf0[oldj as usize * d..oldj as usize * d + d]);
        }
        HierRouter { d, kf, c0n, cf, c0, gstart, mu, b0, c1: Vec::new(), g1start: Vec::new(), b1: 0 }
    }

    /// top-`k` nearest fine cells to normalized query `qn`. 2-level: top-b0 coarse -> their fines.
    /// 3-level (c1 non-empty): top-b0 coarse -> top-b1 mids -> their fines. O(Kf^1/3) routing.
    fn route_fine(&self, qn: &[i8], k: usize, out: &mut Vec<u32>) {
        let d = self.d;
        // top-b0 coarse
        let mut cd: Vec<(i32, u32)> = (0..self.c0n).map(|q| (simd::l2_i8(qn, &self.c0[q * d..q * d + d]), q as u32)).collect();
        let b0 = self.b0.min(cd.len());
        cd.select_nth_unstable(b0 - 1);
        out.clear();
        if self.c1.is_empty() {
            // 2-level: fines directly under the top-b0 coarse
            let mut fd: Vec<(i32, u32)> = Vec::with_capacity(self.kf / self.c0n * b0 + 16);
            for &(_, q) in &cd[..b0] {
                let (s, e) = (self.gstart[q as usize] as usize, self.gstart[q as usize + 1] as usize);
                for f in s..e { fd.push((simd::l2_i8(qn, &self.cf[f * d..f * d + d]), f as u32)); }
            }
            let k = k.min(fd.len());
            if k > 0 { fd.select_nth_unstable(k - 1); out.extend(fd[..k].iter().map(|&(_, f)| f)); }
        } else {
            // 3-level: top-b0 coarse -> score their MIDS -> top-b1 mids -> their fines
            let mut md: Vec<(i32, u32)> = Vec::with_capacity(64);
            for &(_, q) in &cd[..b0] {
                let (s, e) = (self.gstart[q as usize] as usize, self.gstart[q as usize + 1] as usize);
                for mi in s..e { md.push((simd::l2_i8(qn, &self.c1[mi * d..mi * d + d]), mi as u32)); }
            }
            let b1 = self.b1.min(md.len());
            if b1 > 0 { md.select_nth_unstable(b1 - 1); }
            let mut fd: Vec<(i32, u32)> = Vec::with_capacity(256);
            for &(_, mi) in &md[..b1] {
                let (s, e) = (self.g1start[mi as usize] as usize, self.g1start[mi as usize + 1] as usize);
                for f in s..e { fd.push((simd::l2_i8(qn, &self.cf[f * d..f * d + d]), f as u32)); }
            }
            let k = k.min(fd.len());
            if k > 0 { fd.select_nth_unstable(k - 1); out.extend(fd[..k].iter().map(|&(_, f)| f)); }
        }
    }
}

impl Router for HierRouter {
    fn n_cells(&self) -> usize { self.kf }
    fn assign(&self, row: &[i8], a0: usize, out: &mut Vec<u32>) {
        let mut qn = [0i8; 256];
        simd::normalize_i8(row, &self.mu, &mut qn[..self.d]);
        self.route_fine(&qn[..self.d], a0, out);
        while out.len() < a0 { out.push(0); }
    }
    fn probe(&self, q: &[i8], p: usize) -> Vec<u32> {
        let mut qn = [0i8; 256];
        simd::normalize_i8(q, &self.mu, &mut qn[..self.d]);
        let mut out = Vec::new();
        self.route_fine(&qn[..self.d], p, &mut out);
        out
    }
}

// ---------------- Compressor: candidate scan (approx distances) ----------------
pub enum QueryCtx {
    Pq { regs: Vec<__m128i> },                  // PQ/OPQ/AQ LUT registers (i8, saturating)
    // i16 LUT: lo/hi byte-tables (AVX2 single-block) + zmm tables (AVX-512 32-wide pair). Full-res ranking.
    Pq16 { lo: Vec<__m128i>, hi: Vec<__m128i>, lut_z: Vec<__m512i> },
    Pq8 { regs: Vec<__m128i> },                  // fast-scan: int8 LUT, 1 vpshufb/subspace, i16 accum
    Scalar,                                      // exact int8: scan uses the raw query
}

pub trait Compressor: Send + Sync {
    fn block_bytes(&self) -> usize;
    fn encode_block(&self, rows: &[&[i8]], n_real: usize, out: &mut Vec<u8>);
    fn prepare_query(&self, q: &[i8]) -> QueryCtx;
    /// approx dists (smaller=closer) for the 16 points of `block` into out16.
    fn scan_block(&self, block: &[u8], ctx: &QueryCtx, q: &[i8], rows16: &[&[i8]], out16: &mut [i32; 16]);
    /// Does scan_block read the raw i8 rows? PQ/ADC compressors don't (LUT-only) -> skip the 16
    /// per-block ds.row() gathers + Vec build entirely. Only ScalarI8 (exact scan) needs them.
    fn needs_raw_rows(&self) -> bool { false }
    /// Scan TWO consecutive blocks (32 points) at once -> out[0..16]=block0, out[16..32]=block1.
    /// Default = two scan_block calls; PQ16 overrides with the AVX-512 32-wide vpermw kernel.
    fn scan_block_x2(&self, b0: &[u8], b1: &[u8], ctx: &QueryCtx, out: &mut [i32; 32]) {
        let (mut o0, mut o1) = ([0i32; 16], [0i32; 16]);
        self.scan_block(b0, ctx, &[], &[], &mut o0);
        self.scan_block(b1, ctx, &[], &[], &mut o1);
        out[..16].copy_from_slice(&o0);
        out[16..].copy_from_slice(&o1);
    }
}

/// 4-bit Product Quantization (wraps pq::Pq).
pub struct Pq4 { pq: pq::Pq }

impl Pq4 {
    pub fn train(ds: &I8Bin, dpb: usize, iters: usize) -> Self {
        let n = ds.nb;
        let sample: Vec<&[i8]> = (0..n.min(40000)).map(|i| ds.row(i * (n / n.min(40000)))).collect();
        Pq4 { pq: pq::Pq::train(&sample, ds.d, dpb, iters) }
    }
}

impl Compressor for Pq4 {
    fn block_bytes(&self) -> usize { self.pq.m / 2 * 16 }
    fn encode_block(&self, rows: &[&[i8]], n_real: usize, out: &mut Vec<u8>) {
        let mut codes16 = [[0u8; 256]; 16];
        for j in 0..16 {
            if j < n_real { self.pq.encode(rows[j], &mut codes16[j][..self.pq.m]); }
            else { for k in 0..self.pq.m { codes16[j][k] = 0; } }
        }
        pq::pack_block(&codes16, self.pq.m, out);
    }
    fn prepare_query(&self, q: &[i8]) -> QueryCtx {
        let lut = self.pq.query_lut(q);
        QueryCtx::Pq { regs: pq::lut_regs(&lut, self.pq.m) }
    }
    fn scan_block(&self, block: &[u8], ctx: &QueryCtx, _q: &[i8], _rows16: &[&[i8]], out16: &mut [i32; 16]) {
        if let QueryCtx::Pq { regs } = ctx {
            let mut o = [0i8; 16];
            unsafe { pq::block_adc_sse(block, self.pq.m, regs, &mut o) };
            for i in 0..16 { out16[i] = o[i] as i32; }
        }
    }
}

/// Anisotropic 4-bit PQ (ScaNN-style): codebooks trained with parallel-error weighting `eta`.
/// No rotation (identity); isolates the anisotropic-loss effect. ADC scan identical to Pq4.
pub struct Apq4 { pq: pq::Pq, d: usize }

impl Apq4 {
    pub fn train(ds: &I8Bin, dpb: usize, iters: usize, eta: f32) -> Self {
        let (n, d) = (ds.nb, ds.d);
        let smp = n.min(40000);
        let stride = (n / smp).max(1);
        let mut x = vec![0f32; smp * d];
        x.par_chunks_mut(d).enumerate().for_each(|(i, o)| { let r = ds.row(i * stride); for k in 0..d { o[k] = r[k] as f32; } });
        Apq4 { pq: pq::Pq::train_f32_aniso(&x, d, dpb, smp, iters, eta), d }
    }
}

impl Compressor for Apq4 {
    fn block_bytes(&self) -> usize { self.pq.m / 2 * 16 }
    fn encode_block(&self, rows: &[&[i8]], n_real: usize, out: &mut Vec<u8>) {
        let mut codes16 = [[0u8; 256]; 16];
        let mut xf = vec![0f32; self.d];
        for j in 0..16 {
            if j < n_real {
                for k in 0..self.d { xf[k] = rows[j][k] as f32; }
                self.pq.encode_f32(&xf, &mut codes16[j][..self.pq.m]);
            } else { for k in 0..self.pq.m { codes16[j][k] = 0; } }
        }
        pq::pack_block(&codes16, self.pq.m, out);
    }
    fn prepare_query(&self, q: &[i8]) -> QueryCtx {
        let qf: Vec<f32> = q.iter().map(|&v| v as f32).collect();
        // fast-scan: int8 LUT, 1 vpshufb/subspace + i16 accum. ~1.7x scan at ~12-13 bit rank. L2 or IP.
        if FASTSCAN.load(std::sync::atomic::Ordering::Relaxed) {
            let l = if IP_MODE.load(std::sync::atomic::Ordering::Relaxed) { self.pq.query_lut_f32_i8s_ip(&qf) } else { self.pq.query_lut_f32_i8s(&qf) };
            return QueryCtx::Pq8 { regs: pq::lut_regs_i8(&l, self.pq.m) };
        }
        if !LUT16_OFF.load(std::sync::atomic::Ordering::Relaxed) {
            let lut = if IP_MODE.load(std::sync::atomic::Ordering::Relaxed) { self.pq.query_lut_f32_i16_ip(&qf) } else { self.pq.query_lut_f32_i16(&qf) };
            let (lo, hi) = pq::lut_regs_i16(&lut, self.pq.m);
            let lut_z = pq::lut_regs_i16_z(&lut, self.pq.m);
            QueryCtx::Pq16 { lo, hi, lut_z }
        } else {
            { let l = if IP_MODE.load(std::sync::atomic::Ordering::Relaxed) { self.pq.query_lut_f32_ip_i8(&qf) } else { self.pq.query_lut_f32(&qf) }; QueryCtx::Pq { regs: pq::lut_regs(&l, self.pq.m) } }
        }
    }
    fn scan_block(&self, block: &[u8], ctx: &QueryCtx, _q: &[i8], _r: &[&[i8]], out16: &mut [i32; 16]) {
        match ctx {
            QueryCtx::Pq { regs } => {
                let mut o = [0i8; 16];
                unsafe { pq::block_adc_sse(block, self.pq.m, regs, &mut o) };
                for i in 0..16 { out16[i] = o[i] as i32; }
            }
            QueryCtx::Pq16 { lo, hi, .. } => unsafe { pq::block_adc_i16_avx2(block, self.pq.m, lo, hi, out16) },
            QueryCtx::Pq8 { regs } => unsafe { pq::block_adc_i8_i16acc(block, self.pq.m, regs, out16) },
            _ => {}
        }
    }
    fn scan_block_x2(&self, b0: &[u8], b1: &[u8], ctx: &QueryCtx, out: &mut [i32; 32]) {
        if let QueryCtx::Pq16 { lut_z, .. } = ctx {
            if std::env::var("SBANN_USE512").is_ok() && std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
                unsafe { pq::block_adc_i16_avx512_x2(b0, b1, self.pq.m, lut_z, out) };
                return;
            }
        }
        let (mut o0, mut o1) = ([0i32; 16], [0i32; 16]);
        self.scan_block(b0, ctx, &[], &[], &mut o0);
        self.scan_block(b1, ctx, &[], &[], &mut o1);
        out[..16].copy_from_slice(&o0);
        out[16..].copy_from_slice(&o1);
    }
}

/// Random orthogonal d×d rotation (uniform entries + Gram-Schmidt rows).
fn random_orthogonal(d: usize, mut seed: u64) -> Vec<f32> {
    let mut m = vec![0f32; d * d];
    for v in m.iter_mut() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *v = ((seed >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0;
    }
    for i in 0..d {
        for j in 0..i {
            let dp: f32 = (0..d).map(|k| m[i * d + k] * m[j * d + k]).sum();
            for k in 0..d { m[i * d + k] -= dp * m[j * d + k]; }
        }
        let nrm = (0..d).map(|k| m[i * d + k] * m[i * d + k]).sum::<f32>().sqrt().max(1e-9);
        for k in 0..d { m[i * d + k] /= nrm; }
    }
    m
}

#[inline]
fn rotate(x: &[i8], r: &[f32], d: usize, out: &mut [f32]) {
    for i in 0..d {
        let mut s = 0.0f32;
        for k in 0..d { s += r[i * d + k] * x[k] as f32; }
        out[i] = s;
    }
}

/// OPQ: learned/random rotation R applied before 4-bit PQ. The rotation (a d×d matvec) is the
/// "spend spare compute" lever — it decorrelates dims so PQ subspaces carry more info per byte.
/// Same SIMD ADC scan as Pq4 (codes are identical in shape); only encode/query gain the rotation.
pub struct Opq4 {
    pq: pq::Pq,
    r: Vec<f32>,
    d: usize,
}

impl Opq4 {
    pub fn train(ds: &I8Bin, dpb: usize, iters: usize) -> Self {
        let (n, d) = (ds.nb, ds.d);
        let r = random_orthogonal(d, 0x09b);
        let smp = n.min(40000);
        let stride = (n / smp).max(1);
        let mut rotated = vec![0f32; smp * d];
        rotated.par_chunks_mut(d).enumerate().for_each(|(i, o)| rotate(ds.row(i * stride), &r, d, o));
        let pq = pq::Pq::train_f32(&rotated, d, dpb, smp, iters);
        Opq4 { pq, r, d }
    }

    /// OPQ + anisotropic (ScaNN's full recipe): learn rotation R via OPQ-NP, then anisotropic-train
    /// the final codebooks on the R-rotated data. `eta`>1 enables the parallel-error weighting.
    pub fn train_aopq(ds: &I8Bin, dpb: usize, pq_iters: usize, opq_iters: usize, eta: f32) -> Self {
        let mut s = Self::train_learned(ds, dpb, pq_iters, opq_iters);
        if eta > 1.0 {
            let (n, d) = (ds.nb, ds.d);
            let smp = n.min(40000);
            let stride = (n / smp).max(1);
            let mut xr = vec![0f32; smp * d];
            xr.par_chunks_mut(d).enumerate().for_each(|(i, o)| rotate(ds.row(i * stride), &s.r, d, o));
            s.pq = pq::Pq::train_f32_aniso(&xr, d, dpb, smp, pq_iters, eta);
        }
        s
    }

    /// Learned OPQ-NP: alternate (rotate -> train PQ -> reconstruct -> Procrustes R update).
    pub fn train_learned(ds: &I8Bin, dpb: usize, pq_iters: usize, opq_iters: usize) -> Self {
        let (n, d) = (ds.nb, ds.d);
        let smp = n.min(40000);
        let stride = (n / smp).max(1);
        let mut x = vec![0f32; smp * d]; // unrotated sample (f32)
        x.par_chunks_mut(d).enumerate().for_each(|(i, o)| {
            let r = ds.row(i * stride);
            for k in 0..d { o[k] = r[k] as f32; }
        });
        let mut rmat = random_orthogonal(d, 0x09b);
        let mut pq = pq::Pq::train_f32(&x, d, dpb, smp, 1); // placeholder; retrained in loop
        for _ in 0..opq_iters {
            // rotate sample by current R
            let mut xr = vec![0f32; smp * d];
            xr.par_chunks_mut(d).enumerate().for_each(|(i, o)| {
                for a in 0..d {
                    let mut s = 0.0f32;
                    for k in 0..d { s += rmat[a * d + k] * x[i * d + k]; }
                    o[a] = s;
                }
            });
            pq = pq::Pq::train_f32(&xr, d, dpb, smp, pq_iters);
            // reconstruct yhat in rotated space, then M = X^T Yhat
            let m_acc: Vec<f64> = (0..smp).into_par_iter().fold(|| vec![0f64; d * d], |mut acc, i| {
                let mut code = [0u8; 256];
                let mut yh = [0f32; 256];
                pq.encode_f32(&xr[i * d..i * d + d], &mut code[..pq.m]);
                pq.decode(&code[..pq.m], &mut yh[..d]);
                for a in 0..d {
                    let xa = x[i * d + a] as f64;
                    for b in 0..d { acc[a * d + b] += xa * yh[b] as f64; }
                }
                acc
            }).reduce(|| vec![0f64; d * d], |mut a, b| { for k in 0..d * d { a[k] += b[k]; } a });
            // Procrustes: R = V U^T where M = U S V^T (maximizes tr(R M))
            use nalgebra::DMatrix;
            let mm = DMatrix::from_row_slice(d, d, &m_acc);
            let svd = mm.svd(true, true);
            let u = svd.u.unwrap();
            let vt = svd.v_t.unwrap();
            let rnew = vt.transpose() * u.transpose(); // V * U^T
            for a in 0..d { for b in 0..d { rmat[a * d + b] = rnew[(a, b)] as f32; } }
        }
        Opq4 { pq, r: rmat, d }
    }
}

impl Compressor for Opq4 {
    fn block_bytes(&self) -> usize { self.pq.m / 2 * 16 }
    fn encode_block(&self, rows: &[&[i8]], n_real: usize, out: &mut Vec<u8>) {
        let mut codes16 = [[0u8; 256]; 16];
        let mut rot = vec![0f32; self.d];
        for j in 0..16 {
            if j < n_real {
                rotate(rows[j], &self.r, self.d, &mut rot);
                self.pq.encode_f32(&rot, &mut codes16[j][..self.pq.m]);
            } else {
                for k in 0..self.pq.m { codes16[j][k] = 0; }
            }
        }
        pq::pack_block(&codes16, self.pq.m, out);
    }
    fn prepare_query(&self, q: &[i8]) -> QueryCtx {
        let mut rot = vec![0f32; self.d];
        rotate(q, &self.r, self.d, &mut rot);
        if !LUT16_OFF.load(std::sync::atomic::Ordering::Relaxed) {
            let lut = if IP_MODE.load(std::sync::atomic::Ordering::Relaxed) { self.pq.query_lut_f32_i16_ip(&rot) } else { self.pq.query_lut_f32_i16(&rot) };
            let (lo, hi) = pq::lut_regs_i16(&lut, self.pq.m);
            let lut_z = pq::lut_regs_i16_z(&lut, self.pq.m);
            QueryCtx::Pq16 { lo, hi, lut_z }
        } else {
            { let l = if IP_MODE.load(std::sync::atomic::Ordering::Relaxed) { self.pq.query_lut_f32_ip_i8(&rot) } else { self.pq.query_lut_f32(&rot) }; QueryCtx::Pq { regs: pq::lut_regs(&l, self.pq.m) } }
        }
    }
    fn scan_block(&self, block: &[u8], ctx: &QueryCtx, _q: &[i8], _rows16: &[&[i8]], out16: &mut [i32; 16]) {
        match ctx {
            QueryCtx::Pq { regs } => {
                let mut o = [0i8; 16];
                unsafe { pq::block_adc_sse(block, self.pq.m, regs, &mut o) };
                for i in 0..16 { out16[i] = o[i] as i32; }
            }
            QueryCtx::Pq16 { lo, hi, .. } => unsafe { pq::block_adc_i16_avx2(block, self.pq.m, lo, hi, out16) },
            _ => {}
        }
    }
    fn scan_block_x2(&self, b0: &[u8], b1: &[u8], ctx: &QueryCtx, out: &mut [i32; 32]) {
        if let QueryCtx::Pq16 { lut_z, .. } = ctx {
            if std::env::var("SBANN_USE512").is_ok() && std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
                unsafe { pq::block_adc_i16_avx512_x2(b0, b1, self.pq.m, lut_z, out) };
                return;
            }
        }
        let (mut o0, mut o1) = ([0i32; 16], [0i32; 16]);
        self.scan_block(b0, ctx, &[], &[], &mut o0);
        self.scan_block(b1, ctx, &[], &[], &mut o1);
        out[..16].copy_from_slice(&o0);
        out[16..].copy_from_slice(&o1);
    }
}

/// Exact int8 — no compression; "scan" is the true L2 (a baseline / high-recall compressor slot).
pub struct ScalarI8 { d: usize }
impl ScalarI8 { pub fn new(d: usize) -> Self { ScalarI8 { d } } }
impl Compressor for ScalarI8 {
    fn block_bytes(&self) -> usize { 0 } // stores nothing; rescans raw vectors
    fn encode_block(&self, _rows: &[&[i8]], _n: usize, _out: &mut Vec<u8>) {}
    fn prepare_query(&self, _q: &[i8]) -> QueryCtx { QueryCtx::Scalar }
    fn scan_block(&self, _b: &[u8], _c: &QueryCtx, q: &[i8], rows16: &[&[i8]], out16: &mut [i32; 16]) {
        for i in 0..16 {
            out16[i] = if rows16[i].len() == self.d { simd::l2_i8(q, rows16[i]) } else { i32::MAX };
        }
    }
    fn needs_raw_rows(&self) -> bool { true }
}

// ---------------- Index: compose Router + Compressor ----------------
pub struct Index {
    pub router: Box<dyn Router>,
    pub comp: Box<dyn Compressor>,
    pub cell_bstart: Vec<u32>,
    pub slot_orig: Vec<u32>,
    pub blocks: Vec<u8>,
    pub bb: usize,
    pub xfn: Vec<i32>, // raw norms unused here; rerank reads ds
    pub raw: Vec<i8>,  // raw i8 vectors in SLOT order (cell-contiguous) -> cache-warm rerank gathers
    pub d: usize,
}

impl Index {
    pub fn build(router: Box<dyn Router>, comp: Box<dyn Compressor>, ds: &I8Bin, a0: usize) -> Index {
        let (n, _d) = (ds.nb, ds.d);
        let nc = router.n_cells();
        // assign all points (parallel) into a preallocated [n*a0] array — no per-point Vec
        // allocation/flatten (that spiked multi-GB and crawled at 100M).
        let mut assign = vec![0u32; n * a0];
        assign.par_chunks_mut(a0).enumerate().for_each(|(i, out)| {
            let mut buf: Vec<u32> = Vec::with_capacity(a0);
            router.assign(ds.row(i), a0, &mut buf);
            for k in 0..a0 { out[k] = buf.get(k).copied().unwrap_or(0); }
        });
        // CSR (cell -> points)
        let mut cell_start = vec![0u32; nc + 1];
        for &c in &assign { cell_start[c as usize + 1] += 1; }
        for j in 0..nc { cell_start[j + 1] += cell_start[j]; }
        let mut ids = vec![0u32; n * a0];
        let mut cur = cell_start.clone();
        for (idx, &c) in assign.iter().enumerate() {
            let pt = (idx / a0) as u32;
            ids[cur[c as usize] as usize] = pt;
            cur[c as usize] += 1;
        }
        // encode blocks per cell
        let bb = comp.block_bytes();
        let d = ds.d;
        let mut blocks: Vec<u8> = Vec::new();
        let mut slot_orig: Vec<u32> = Vec::new();
        let mut raw: Vec<i8> = Vec::new(); // raw i8 in slot order, parallel to slot_orig
        let mut cell_bstart = vec![0u32; nc + 1];
        for cell in 0..nc {
            let (s, e) = (cell_start[cell] as usize, cell_start[cell + 1] as usize);
            let pts = &ids[s..e];
            let mut i = 0;
            while i < pts.len() {
                let cnt = (pts.len() - i).min(16);
                let rows: Vec<&[i8]> = (0..16).map(|j| if j < cnt { ds.row(pts[i + j] as usize) } else { &[][..] }).collect();
                comp.encode_block(&rows, cnt, &mut blocks);
                for j in 0..16 {
                    slot_orig.push(if j < cnt { pts[i + j] } else { u32::MAX });
                    if j < cnt { raw.extend_from_slice(rows[j]); } else { raw.resize(raw.len() + d, 0); }
                }
                i += 16;
            }
            cell_bstart[cell + 1] = if bb > 0 { (blocks.len() / bb) as u32 } else { (slot_orig.len() / 16) as u32 };
        }
        Index { router, comp, cell_bstart, slot_orig, blocks, bb, xfn: Vec::new(), raw, d }
    }

    pub fn search(&self, ds: &I8Bin, q: &[i8], p: usize, t: usize, k: usize) -> Vec<u32> {
        let cells = self.router.probe(q, p);
        self.scan_rerank(ds, q, &cells, t, k)
    }

    /// OLD rerank path (gather from ds by orig id) -- kept for clean same-index A/B vs the new
    /// cell-contiguous path. Identical results; differs only in WHERE rerank reads the i8 vectors.
    pub fn search_ds(&self, ds: &I8Bin, q: &[i8], p: usize, t: usize, k: usize) -> Vec<u32> {
        let cells = self.router.probe(q, p);
        let ctx = self.comp.prepare_query(q);
        let need_rows = self.comp.needs_raw_rows();
        let mut pool: Vec<(i32, u32)> = Vec::with_capacity(8192);
        let mut out16 = [0i32; 16];
        for &cell in &cells {
            let (bs, be) = (self.cell_bstart[cell as usize] as usize, self.cell_bstart[cell as usize + 1] as usize);
            for b in bs..be {
                let block = if self.bb > 0 { &self.blocks[b * self.bb..(b + 1) * self.bb] } else { &[][..] };
                let rows16: Vec<&[i8]> = if need_rows {
                    (0..16).map(|j| {
                        let o = self.slot_orig[b * 16 + j];
                        if o != u32::MAX { ds.row(o as usize) } else { &[][..] }
                    }).collect()
                } else { Vec::new() };
                self.comp.scan_block(block, &ctx, q, &rows16, &mut out16);
                for j in 0..16 {
                    let o = self.slot_orig[b * 16 + j];
                    if o != u32::MAX { pool.push((out16[j], o)); }
                }
            }
        }
        let tt = t.min(pool.len());
        if tt > 0 { pool.select_nth_unstable(tt - 1); pool.truncate(tt); }
        rerank_survivors(ds, q, &pool, k)
    }

    /// Profiled search: returns (route_ns, scan_ns, rerank_ns) so we can see where query time goes.
    pub fn search_prof(&self, ds: &I8Bin, q: &[i8], p: usize, t: usize, k: usize) -> (u64, u64, u64) {
        use std::time::Instant;
        let t0 = Instant::now();
        let cells = self.router.probe(q, p);
        let t1 = Instant::now();
        let ctx = self.comp.prepare_query(q);
        let need_rows = self.comp.needs_raw_rows();
        let mut pool: Vec<(i32, u32)> = Vec::with_capacity(8192);
        let mut out16 = [0i32; 16];
        for &cell in &cells {
            let (bs, be) = (self.cell_bstart[cell as usize] as usize, self.cell_bstart[cell as usize + 1] as usize);
            for b in bs..be {
                let block = if self.bb > 0 { &self.blocks[b * self.bb..(b + 1) * self.bb] } else { &[][..] };
                let rows16: Vec<&[i8]> = if need_rows {
                    (0..16).map(|j| {
                        let o = self.slot_orig[b * 16 + j];
                        if o != u32::MAX { ds.row(o as usize) } else { &[][..] }
                    }).collect()
                } else { Vec::new() };
                self.comp.scan_block(block, &ctx, q, &rows16, &mut out16);
                for j in 0..16 {
                    let slot = b * 16 + j;
                    if self.slot_orig[slot] != u32::MAX { pool.push((out16[j], slot as u32)); }
                }
            }
        }
        let t2 = Instant::now();
        let tt = t.min(pool.len());
        if tt > 0 { pool.select_nth_unstable(tt - 1); pool.truncate(tt); }
        let _ = ds;
        let _ = rerank_contig(&self.raw, self.d, &self.slot_orig, q, &pool, k);
        let t3 = Instant::now();
        ((t1 - t0).as_nanos() as u64, (t2 - t1).as_nanos() as u64, (t3 - t2).as_nanos() as u64)
    }

    /// Scan the given cells with the compressor, keep top-T by approx dist, exact-rerank to top-k.
    pub fn scan_rerank(&self, ds: &I8Bin, q: &[i8], cells: &[u32], t: usize, k: usize) -> Vec<u32> {
        let ctx = self.comp.prepare_query(q);
        let need_rows = self.comp.needs_raw_rows();
        let mut pool: Vec<(i32, u32)> = Vec::with_capacity(8192);
        let mut out16 = [0i32; 16];
        let mut out32 = [0i32; 32];
        let bb = self.bb;
        for &cell in cells {
            let (bs, be) = (self.cell_bstart[cell as usize] as usize, self.cell_bstart[cell as usize + 1] as usize);
            if need_rows {
                // exact-scan path: build raw rows, per-block
                for b in bs..be {
                    let block = if bb > 0 { &self.blocks[b * bb..(b + 1) * bb] } else { &[][..] };
                    let rows16: Vec<&[i8]> = (0..16).map(|j| {
                        let o = self.slot_orig[b * 16 + j];
                        if o != u32::MAX { ds.row(o as usize) } else { &[][..] }
                    }).collect();
                    self.comp.scan_block(block, &ctx, q, &rows16, &mut out16);
                    for j in 0..16 {
                        let slot = b * 16 + j;
                        if self.slot_orig[slot] != u32::MAX { pool.push((out16[j], slot as u32)); }
                    }
                }
            } else {
                // PQ path: scan blocks in PAIRS (AVX-512 32-wide), per-block for the odd remainder.
                let mut b = bs;
                while b + 1 < be {
                    let b0 = &self.blocks[b * bb..(b + 1) * bb];
                    let b1 = &self.blocks[(b + 1) * bb..(b + 2) * bb];
                    self.comp.scan_block_x2(b0, b1, &ctx, &mut out32);
                    for j in 0..16 {
                        let slot = b * 16 + j;
                        if self.slot_orig[slot] != u32::MAX { pool.push((out32[j], slot as u32)); }
                    }
                    for j in 0..16 {
                        let slot = (b + 1) * 16 + j;
                        if self.slot_orig[slot] != u32::MAX { pool.push((out32[16 + j], slot as u32)); }
                    }
                    b += 2;
                }
                if b < be {
                    let block = &self.blocks[b * bb..(b + 1) * bb];
                    self.comp.scan_block(block, &ctx, q, &[], &mut out16);
                    for j in 0..16 {
                        let slot = b * 16 + j;
                        if self.slot_orig[slot] != u32::MAX { pool.push((out16[j], slot as u32)); }
                    }
                }
            }
        }
        let tt = t.min(pool.len());
        if tt > 0 { pool.select_nth_unstable(tt - 1); pool.truncate(tt); }
        let _ = ds;
        rerank_contig(&self.raw, self.d, &self.slot_orig, q, &pool, k)
    }

    /// Batched search: GEMM-route ALL queries at once, then per-query scan+rerank in parallel.
    pub fn search_batch(&self, ds: &I8Bin, queries: &[i8], nq: usize, p: usize, t: usize, k: usize) -> Vec<Vec<u32>> {
        let d = ds.d;
        let cells_all = self.router.probe_batch(queries, nq, d, p);
        let pp = p.min(self.router.n_cells());
        (0..nq).into_par_iter().map(|i| {
            self.scan_rerank(ds, &queries[i * d..i * d + d], &cells_all[i * pp..i * pp + pp], t, k)
        }).collect()
    }

    /// Adaptive early termination: probe cells nearest-first, stop when the k-th best APPROX dist
    /// stops improving for `patience` windows (each `win` cells). Returns (ids, cells_probed).
    /// Easy queries terminate early -> fewer random rerank accesses; hard ones probe more.
    pub fn search_adaptive(&self, ds: &I8Bin, q: &[i8], max_p: usize, t: usize, k: usize,
                           win: usize, patience: usize) -> (Vec<u32>, usize) {
        use std::collections::BinaryHeap;
        let cells = self.router.probe_ranked(q, max_p);
        let ctx = self.comp.prepare_query(q);
        let need_rows = self.comp.needs_raw_rows();
        let mut pool: Vec<(i32, u32)> = Vec::with_capacity(8192);
        let mut out16 = [0i32; 16];
        let mut prev_kth = i32::MAX;
        let mut stall = 0usize;
        let mut used = 0usize;
        let mut kheap: BinaryHeap<i32> = BinaryHeap::with_capacity(k + 1); // max-heap of the k smallest approx dists
        for (ci, &cell) in cells.iter().enumerate() {
            let (bs, be) = (self.cell_bstart[cell as usize] as usize, self.cell_bstart[cell as usize + 1] as usize);
            for b in bs..be {
                let block = if self.bb > 0 { &self.blocks[b * self.bb..(b + 1) * self.bb] } else { &[][..] };
                let rows16: Vec<&[i8]> = if need_rows {
                    (0..16).map(|j| {
                        let o = self.slot_orig[b * 16 + j];
                        if o != u32::MAX { ds.row(o as usize) } else { &[][..] }
                    }).collect()
                } else { Vec::new() };
                self.comp.scan_block(block, &ctx, q, &rows16, &mut out16);
                for j in 0..16 {
                    let o = self.slot_orig[b * 16 + j];
                    if o != u32::MAX {
                        pool.push((out16[j], o));
                        // maintain k smallest approx dists incrementally (O(log k)/point)
                        if kheap.len() < k { kheap.push(out16[j]); }
                        else if out16[j] < *kheap.peek().unwrap() { kheap.pop(); kheap.push(out16[j]); }
                    }
                }
            }
            used = ci + 1;
            // O(1) stop check: has the k-th best approx dist stopped improving?
            if (ci + 1) % win == 0 && kheap.len() == k {
                let kth = *kheap.peek().unwrap();
                if kth >= prev_kth { stall += 1; } else { stall = 0; }
                prev_kth = kth;
                if stall >= patience { break; }
            }
        }
        let tt = t.min(pool.len());
        if tt > 0 { pool.select_nth_unstable(tt - 1); pool.truncate(tt); }
        (rerank_survivors(ds, q, &pool, k), used)
    }
}
