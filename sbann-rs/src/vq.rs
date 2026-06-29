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
/// true => use the 64-wide AVX-512 fast-scan (block_adc_i8_i16acc_avx512_il) over an INTERLEAVED
/// 64-vector superblock layout built at index time. Only meaningful with FASTSCAN (i8s LUT / Pq8 ctx)
/// + avx512bw. Identical distances to the AVX2 fast-scan (so recall is unchanged). Set from SBANN_USE512FS.
pub static USE512FS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// IDEA #4: build a SECOND finer 8-bit refine code (pq::ResidPq) in slot order and use it to refine
/// the 4-bit-ADC survivor ranking before the exact raw rerank, so far fewer raw vectors are read.
/// Set from SBANN_RESID. SBANN_RESID_DPB picks the refine subspace size (default 2 => m=d/2 bytes/vec).
pub static RESID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// true => RESIDUAL QUANTIZATION (SBANN_RESIDQ): the PRIMARY scan code encodes x - cell_centroid (codebook
/// retrained on residuals), and the scan adds the exact per-cell <q,centroid> offset. +6-11pt IP
/// pool-recall (P124) -> shallower rerank pool for OOD. Distinct from RESID (8-bit refine, which failed).
pub static RESIDQ: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// SBANN_POOLDEDUP: dedup the candidate pool by ORIG id (keep min approx-dist per id) BEFORE the
/// t_surv survivor cap. With SOAR a0>1 a point lands in multiple probed cells as duplicate slots; the
/// late dedup in rerank_contig (heap size k*4) gets crowded out by those duplicates, collapsing recall
/// as a0 grows. This dedup measures the TRUE coverage of a multi-store routing (and shrinks the pool).
pub static POOLDEDUP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

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
/// L-LEVEL general hierarchy (L>=2). Level 0 = coarsest, level L-1 = finest (the actual IVF cells).
/// Routing is O(C0 + Σ beam[l]·count[l+1]/count[l]) ≈ O(L·Kf^(1/L)) per query — deeper L cuts routing
/// for 100M/1B. All depths share ONE code path (gather_fine descent); 2-/3-/4-level are just
/// different `levels`/`beam` configs. Hyperparameters: per-level cell counts (cent lengths), per-level
/// beam widths, a0 multi-assign, soar λ — all tunable (SBANN_LEVELS/SBANN_BEAMS, or the C0/C1/B0/B1
/// back-compat knobs). `child[l]` is a general prefix-sum (supports non-uniform fan-out, e.g. random
/// `train`); hierarchical k-means uses uniform fan-out.
pub struct HierRouter {
    d: usize,
    mu: Vec<f32>,
    kf: usize,              // count[L-1] = number of finest cells (= n_cells)
    levels: usize,          // L
    cent: Vec<Vec<i8>>,     // cent[l]: count[l]*d centroids; for l>=1 grouped contiguously by parent
    child: Vec<Vec<u32>>,   // child[l]: count[l]+1 prefix-sum -> each level-l cell's child range in level l+1; child[L-1] empty
    beam: Vec<usize>,       // beam[l]: # level-l cells expanded per query/assign; len L-1 (finest takes top-k)
    // SOAR (idea #3): when >0, BUILD-time multi-assignment spills the 2nd..a0-th fine cells toward the
    // FIRST cell's residual direction (ScaNN SOAR loss ‖rj‖²+λ(rj·r̂0)²) instead of next-nearest by L2 —
    // covers orthogonal directions so fewer probes are needed at a given recall. 0 = off. Set via set_soar.
    soar: f32,
}

impl HierRouter {
    /// Hierarchical k-means: k-means the C0 coarse centroids, then k-means `Kf/C0` fine centroids
    /// WITHIN each coarse cell's points. Gives k-means-quality routing at O(n·Kf/C0) train cost
    /// (vs flat k-means O(n·Kf)). This is the scale lever (P58) — replaces random fine centroids.
    /// GENERAL L-level hierarchical k-means. `counts` = per-level cell counts coarse→fine
    /// (counts[L-1]=Kf); `beams` = per-level expansion widths (len L-1). Each level's count is rounded
    /// to a multiple of the previous (uniform fan-out). Train cost O(n·Σ fan) vs flat O(n·Kf). This is
    /// the one trainer — train_hkmeans (L=2) and train_hkmeans3 (L=3) are thin wrappers; hierk4/5… just
    /// pass longer `counts`/`beams`. Deeper L cuts per-query routing for 100M/1B.
    pub fn train_hkmeans_multi(ds: &I8Bin, counts_in: &[usize], beams: &[usize], mu: Vec<f32>) -> Self {
        let (n, d) = (ds.nb, ds.d);
        let levels = counts_in.len();
        assert!(levels >= 2 && beams.len() == levels - 1, "hier: need L>=2 levels and L-1 beams");
        // actual per-level counts: each a multiple of the previous (uniform fan-out)
        let mut counts = vec![counts_in[0].max(1)];
        for l in 1..levels { let fan = (counts_in[l] / counts[l - 1]).max(1); counts.push(counts[l - 1] * fan); }
        let kf = counts[levels - 1];
        let smp = n.min(counts[0] * 300 + kf * 6 + counts.iter().sum::<usize>() * 10);
        let mut xn = vec![0f32; smp * d];
        xn.par_chunks_mut(d).enumerate().for_each(|(i, r)| simd::norm_f32(ds.row(i), &mu, r));
        let prof = std::env::var("SBANN_BUILDPROF").is_ok();
        let mut lvl_t: Vec<f64> = Vec::new();
        let t_lvl = std::time::Instant::now();
        // level 0: flat k-means over the whole sample
        let mut centf = kmeans::kmeans_f32(&xn, smp, d, counts[0], 12, 0xc0a1_5eed);
        let mut point_cell: Vec<u32> = (0..smp).into_par_iter().map(|i| {
            let x = &xn[i * d..i * d + d];
            (0..counts[0]).map(|q| (simd::l2_f32(x, &centf[q * d..q * d + d]), q as u32)).min_by(|a, b| a.0.total_cmp(&b.0)).unwrap().1
        }).collect();
        let mut centf_lv: Vec<Vec<f32>> = vec![centf.clone()]; // float centroids per level (for optional tree-EM)
        let mut child: Vec<Vec<u32>> = Vec::with_capacity(levels);
        if prof { lvl_t.push(t_lvl.elapsed().as_secs_f64()); }
        // deeper levels: per-parent k-means into `fan` children, grouped contiguously by parent id
        for l in 1..levels {
            let t_lvl = std::time::Instant::now();
            let par = counts[l - 1];
            let fan = counts[l] / par;
            let mut by: Vec<Vec<u32>> = vec![Vec::new(); par];
            for i in 0..smp { by[point_cell[i] as usize].push(i as u32); }
            let res: Vec<(Vec<f32>, Vec<u32>)> = (0..par).into_par_iter().map(|p| {
                let idx = &by[p];
                let m = idx.len().max(1);
                let mut pts = vec![0f32; m * d];
                for (j, &pi) in idx.iter().enumerate() { pts[j * d..j * d + d].copy_from_slice(&xn[pi as usize * d..pi as usize * d + d]); }
                let cf = kmeans::kmeans_f32(&pts, m, d, fan, 8, 0x111d_0000 ^ ((l as u64) << 40) ^ p as u64);
                let asg: Vec<u32> = (0..m).map(|i| {
                    let x = &pts[i * d..i * d + d];
                    (0..fan).map(|c| (simd::l2_f32(x, &cf[c * d..c * d + d]), c as u32)).min_by(|a, b| a.0.total_cmp(&b.0)).unwrap().1
                }).collect();
                (cf, asg)
            }).collect();
            let mut newcentf = vec![0f32; counts[l] * d];
            for p in 0..par { newcentf[p * fan * d..(p + 1) * fan * d].copy_from_slice(&res[p].0); }
            let mut newpc = vec![0u32; smp];
            for p in 0..par { let idx = &by[p]; for (j, &pi) in idx.iter().enumerate() { newpc[pi as usize] = (p * fan) as u32 + res[p].1[j]; } }
            point_cell = newpc;
            centf = newcentf;
            centf_lv.push(centf.clone());
            let mut cs = vec![0u32; par + 1];
            for p in 0..par { cs[p + 1] = cs[p] + fan as u32; }
            child.push(cs);
            if prof { lvl_t.push(t_lvl.elapsed().as_secs_f64()); }
        }
        child.push(Vec::new()); // finest level has no children
        if prof {
            let fans: Vec<usize> = (0..levels).map(|l| if l == 0 { counts[0] } else { counts[l] / counts[l - 1] }).collect();
            let times: Vec<f64> = lvl_t.iter().map(|x| (x * 10.0).round() / 10.0).collect();
            println!("  [buildprof L={levels} counts={counts:?} fanout={fans:?} per-level-s={times:?}]");
        }
        // OPTIONAL JOINT TREE-LLOYD EM (SBANN_TREEEM=rounds, default 0 = pure greedy). Each round:
        // E-step reassign every sample point to its nearest LEAF via the BEAM descent (so the objective
        // matches the query-time search, not a greedy-build artifact); M-step recompute EVERY level's
        // centroids jointly as means of their assigned points (leaf-id ÷ fan-product gives each ancestor).
        // Attacks the greedy boundary problem by letting upper and lower levels co-adapt.
        let em_rounds: usize = std::env::var("SBANN_TREEEM").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
        // E-step beam CAP (SBANN_TREEEM_BEAM, default 8): finding a point's single nearest LEAF needs only
        // a few probes (the nearest leaf is ~always under the nearest coarse cell), so the assignment uses
        // a MUCH smaller beam than a query — this is the structure accelerating its own EM (~10-50x cheaper
        // E-step vs using the full query beams, with negligible assignment loss the M-step averages out).
        let em_beam: usize = std::env::var("SBANN_TREEEM_BEAM").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
        for r in 0..em_rounds {
            let t_em = std::time::Instant::now();
            // E-step: nearest leaf via SMALL-beam descent over current float centroids
            let leaf: Vec<u32> = (0..smp).into_par_iter().map(|i| {
                let x = &xn[i * d..i * d + d];
                let mut cd: Vec<(f32, u32)> = (0..counts[0]).map(|c| (simd::l2_f32(x, &centf_lv[0][c * d..c * d + d]), c as u32)).collect();
                let b = beams[0].min(em_beam).min(cd.len());
                if b > 0 && b < cd.len() { cd.select_nth_unstable_by(b - 1, |a, b| a.0.total_cmp(&b.0)); cd.truncate(b); }
                let mut sel: Vec<u32> = cd.iter().map(|&(_, c)| c).collect();
                for l in 1..levels {
                    let fan = counts[l] / counts[l - 1];
                    let mut nd: Vec<(f32, u32)> = Vec::with_capacity(sel.len() * fan);
                    for &p in &sel { for c in (p as usize * fan)..((p as usize + 1) * fan) { nd.push((simd::l2_f32(x, &centf_lv[l][c * d..c * d + d]), c as u32)); } }
                    if l == levels - 1 { return nd.iter().min_by(|a, b| a.0.total_cmp(&b.0)).map(|&(_, c)| c).unwrap_or(0); }
                    let b = beams[l].min(em_beam).min(nd.len());
                    if b > 0 && b < nd.len() { nd.select_nth_unstable_by(b - 1, |a, b| a.0.total_cmp(&b.0)); nd.truncate(b); }
                    sel = nd.iter().map(|&(_, c)| c).collect();
                }
                sel.first().copied().unwrap_or(0)
            }).collect();
            // M-step: recompute every level's centroids from the new leaf assignment (ancestor = leaf / div)
            for l in 0..levels {
                let cl = counts[l];
                let div = counts[levels - 1] / cl;
                let mut sum = vec![0f64; cl * d];
                let mut cnt = vec![0u64; cl];
                for i in 0..smp {
                    let cell = (leaf[i] as usize) / div;
                    cnt[cell] += 1;
                    let x = &xn[i * d..i * d + d];
                    for k in 0..d { sum[cell * d + k] += x[k] as f64; }
                }
                for c in 0..cl { if cnt[c] > 0 { for k in 0..d { centf_lv[l][c * d + k] = (sum[c * d + k] / cnt[c] as f64) as f32; } } }
            }
            point_cell = leaf;
            if prof { println!("  [treeEM round {r} {:.1}s]", t_em.elapsed().as_secs_f64()); }
        }
        let _ = &point_cell;
        // quantize all levels to i8 (after any EM refinement)
        let cent: Vec<Vec<i8>> = centf_lv.iter().map(|cf| cf.iter().map(|&v| (v * 127.0).round().clamp(-127.0, 127.0) as i8).collect()).collect();
        HierRouter { d, mu, kf, levels, cent, child, beam: beams.to_vec(), soar: 0.0 }
    }

    /// 2-level hierarchical k-means (back-compat wrapper): C0 coarse → Kf/C0 fine per coarse.
    pub fn train_hkmeans(ds: &I8Bin, kf: usize, c0n: usize, b0: usize, mu: Vec<f32>) -> Self {
        Self::train_hkmeans_multi(ds, &[c0n, kf], &[b0], mu)
    }

    /// 3-level hierarchical k-means (back-compat wrapper): C0 coarse → C1 mid → Kf fine.
    pub fn train_hkmeans3(ds: &I8Bin, kf: usize, c0n: usize, c1n: usize, b0: usize, b1: usize, mu: Vec<f32>) -> Self {
        Self::train_hkmeans_multi(ds, &[c0n, c1n, kf], &[b0, b1], mu)
    }

    /// 2-level RANDOM-centroid router (P32: random ≈ k-means for IVF since rerank fixes ranking).
    /// Non-uniform fan-out (variable fines per coarse) — supported by the general prefix-sum `child`.
    pub fn train(ds: &I8Bin, kf: usize, c0n: usize, b0: usize, mu: Vec<f32>) -> Self {
        let (n, d) = (ds.nb, ds.d);
        let mut seed = 0x77c0_ffeeu64;
        let mut rid = |m: usize| { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1); (seed >> 11) as usize % m };
        let mut cf0 = vec![0i8; kf * d];
        for j in 0..kf { let id = rid(n); simd::normalize_i8(ds.row(id), &mu, &mut cf0[j * d..j * d + d]); }
        let mut c0 = vec![0i8; c0n * d];
        for j in 0..c0n { let id = rid(kf); c0[j * d..j * d + d].copy_from_slice(&cf0[id * d..id * d + d]); }
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
        HierRouter { d, mu, kf, levels: 2, cent: vec![c0, cf], child: vec![gstart, Vec::new()], beam: vec![b0], soar: 0.0 }
    }

    /// Enable SOAR build-time spilled assignment with penalty λ (`s`). 0 disables (default path).
    pub fn set_soar(&mut self, s: f32) { self.soar = s; }

    /// General L-level descent: top-beam[0] coarse → expand to children → top-beam[l] … → collect ALL
    /// finest-level candidates as (l2, fine_id) into `fd` (NOT truncated — route_fine does top-k,
    /// route_fine_soar does the SOAR spill). One code path for every depth L>=2.
    fn gather_fine(&self, qn: &[i8], fd: &mut Vec<(i32, u32)>) {
        let d = self.d;
        let l0 = self.cent[0].len() / d;
        let mut cd: Vec<(i32, u32)> = (0..l0).map(|q| (simd::l2_i8(qn, &self.cent[0][q * d..q * d + d]), q as u32)).collect();
        let b = self.beam[0].min(cd.len());
        if b > 0 && b < cd.len() { cd.select_nth_unstable(b - 1); cd.truncate(b); }
        let mut sel: Vec<u32> = cd.iter().map(|&(_, c)| c).collect();
        fd.clear();
        for l in 1..self.levels {
            let mut nd: Vec<(i32, u32)> = Vec::with_capacity(sel.len() * 8 + 16);
            for &p in &sel {
                let (s, e) = (self.child[l - 1][p as usize] as usize, self.child[l - 1][p as usize + 1] as usize);
                for c in s..e { nd.push((simd::l2_i8(qn, &self.cent[l][c * d..c * d + d]), c as u32)); }
            }
            if l == self.levels - 1 { *fd = nd; return; }
            let b = self.beam[l].min(nd.len());
            if b > 0 && b < nd.len() { nd.select_nth_unstable(b - 1); nd.truncate(b); }
            sel = nd.iter().map(|&(_, c)| c).collect();
        }
    }

    /// top-`k` nearest finest cells to normalized query `qn`, via the general L-level descent.
    fn route_fine(&self, qn: &[i8], k: usize, out: &mut Vec<u32>) {
        let mut fd: Vec<(i32, u32)> = Vec::new();
        self.gather_fine(qn, &mut fd);
        out.clear();
        let k = k.min(fd.len());
        if k > 0 { fd.select_nth_unstable(k - 1); out.extend(fd[..k].iter().map(|&(_, f)| f)); }
    }

    /// SOAR-aware multi-assignment (idea #3). Gathers the SAME bounded finest-candidate set route_fine
    /// would (general L-level descent), then: i0 = nearest finest cell by L2; the remaining a0-1 picks
    /// minimize the ScaNN SOAR loss  L = ‖rj‖² + λ·(rj·r̂0)²  where r̂0 is the unit FIRST residual
    /// (qn - cent_fine[i0]) and rj = qn - cent_fine[j]. Penalizing the parallel-to-r0 component pushes
    /// the spilled copies to cover ORTHOGONAL directions, so a query needs fewer probes to hit a cell
    /// that contains the point. Only used at BUILD; query routing (probe) is unchanged.
    fn route_fine_soar(&self, qn: &[i8], a0: usize, soar: f32, out: &mut Vec<u32>) {
        let d = self.d;
        let cf = &self.cent[self.levels - 1]; // finest centroids
        let mut fd: Vec<(i32, u32)> = Vec::new();
        self.gather_fine(qn, &mut fd);
        out.clear();
        if fd.is_empty() { return; }
        // PRE-FILTER to the top-K nearest by L2 before the (costly, scalar) SOAR projection. The spilled
        // cells always sit among the near candidates (the loss is L2 + λ·proj², L2-dominated), so capping
        // the proj dots from |fd| (thousands for deep beams) to K cuts the SOAR build overhead ~order of
        // magnitude with negligible recall change. K scales with a0; 256 floor. Only the projection set
        // shrinks — i0 (the L2-nearest primary assignment) is unaffected.
        let kcap = (a0 * 64).max(256);
        if fd.len() > kcap { fd.select_nth_unstable(kcap - 1); fd.truncate(kcap); }
        // i0 = nearest finest cell (primary assignment unchanged from the L2 path)
        let mut i0pos = 0usize;
        for (idx, &(dist, _)) in fd.iter().enumerate() { if dist < fd[i0pos].0 { i0pos = idx; } }
        let i0 = fd[i0pos].1;
        out.push(i0);
        if a0 <= 1 { return; }
        // r̂0 = normalize(qn - cf[i0]); qdot = qn·r̂0 (so rj·r̂0 = qdot - cf[j]·r̂0)
        let off0 = i0 as usize * d;
        let mut r0 = [0f32; 256];
        let mut nrm = 0f32;
        for k in 0..d { let v = qn[k] as f32 - cf[off0 + k] as f32; r0[k] = v; nrm += v * v; }
        let inv = 1.0 / nrm.sqrt().max(1e-9);
        for k in 0..d { r0[k] *= inv; }
        let qdot: f32 = (0..d).map(|k| qn[k] as f32 * r0[k]).sum();
        // SOAR loss for every other candidate; take the top-(a0-1) smallest
        let mut loss: Vec<(f32, u32)> = Vec::with_capacity(fd.len());
        for (idx, &(l2, f)) in fd.iter().enumerate() {
            if idx == i0pos { continue; }
            let offj = f as usize * d;
            let pdot: f32 = (0..d).map(|k| cf[offj + k] as f32 * r0[k]).sum();
            let proj = qdot - pdot;
            loss.push((l2 as f32 + soar * proj * proj, f));
        }
        let take = (a0 - 1).min(loss.len());
        if take > 0 {
            loss.select_nth_unstable_by(take - 1, |a, b| a.0.total_cmp(&b.0));
            for &(_, f) in &loss[..take] { out.push(f); }
        }
    }
}

impl Router for HierRouter {
    fn n_cells(&self) -> usize { self.kf }
    fn assign(&self, row: &[i8], a0: usize, out: &mut Vec<u32>) {
        let mut qn = [0i8; 256];
        simd::normalize_i8(row, &self.mu, &mut qn[..self.d]);
        if self.soar > 0.0 && a0 >= 2 {
            self.route_fine_soar(&qn[..self.d], a0, self.soar, out);
        } else {
            self.route_fine(&qn[..self.d], a0, out);
        }
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
    Pq16 { lo: Vec<__m128i>, hi: Vec<__m128i>, lut_z: Vec<__m512i>, scale: f32 }, // scale = i16-units/IP for RESIDQ offset
    // fast-scan: int8 LUT, 1 vpshufb/subspace, i16 accum. regs_z = same LUT broadcast to zmm lanes for
    // the 64-wide AVX-512 path (empty unless USE512FS).
    Pq8 { regs: Vec<__m128i>, regs_z: Vec<__m512i>, scale: f32 },
    Scalar,                                      // exact int8: scan uses the raw query
}

pub trait Compressor: Send + Sync {
    fn block_bytes(&self) -> usize;
    /// Encode 16 rows into a block. `cell_cent` (empty unless SBANN_RESIDQ) = the cell's raw centroid;
    /// when non-empty a residual compressor encodes `row - cell_cent` (in f32) instead of the raw row.
    fn encode_block(&self, rows: &[&[i8]], n_real: usize, cell_cent: &[i8], out: &mut Vec<u8>);
    /// Retrain the codebook on residual vectors (SBANN_RESIDQ). Default no-op; Apq4 retrains its PQ.
    fn retrain_residual(&mut self, _sample_f32: &[f32], _n: usize, _d: usize) {}
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
    fn encode_block(&self, rows: &[&[i8]], n_real: usize, _cell_cent: &[i8], out: &mut Vec<u8>) {
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
pub struct Apq4 { pq: pq::Pq, d: usize, dpb: usize, eta: f32 }

impl Apq4 {
    pub fn train(ds: &I8Bin, dpb: usize, iters: usize, eta: f32) -> Self {
        let (n, d) = (ds.nb, ds.d);
        let smp = n.min(40000);
        let stride = (n / smp).max(1);
        let mut x = vec![0f32; smp * d];
        x.par_chunks_mut(d).enumerate().for_each(|(i, o)| { let r = ds.row(i * stride); for k in 0..d { o[k] = r[k] as f32; } });
        Apq4 { pq: pq::Pq::train_f32_aniso(&x, d, dpb, smp, iters, eta), d, dpb, eta }
    }
}

impl Compressor for Apq4 {
    fn block_bytes(&self) -> usize { self.pq.m / 2 * 16 }
    fn encode_block(&self, rows: &[&[i8]], n_real: usize, cell_cent: &[i8], out: &mut Vec<u8>) {
        let mut codes16 = [[0u8; 256]; 16];
        let mut xf = vec![0f32; self.d];
        let resid = !cell_cent.is_empty(); // SBANN_RESIDQ: encode (row - cell_cent) in f32
        for j in 0..16 {
            if j < n_real {
                if resid { for k in 0..self.d { xf[k] = rows[j][k] as f32 - cell_cent[k] as f32; } }
                else { for k in 0..self.d { xf[k] = rows[j][k] as f32; } }
                self.pq.encode_f32(&xf, &mut codes16[j][..self.pq.m]);
            } else { for k in 0..self.pq.m { codes16[j][k] = 0; } }
        }
        pq::pack_block(&codes16, self.pq.m, out);
    }
    fn retrain_residual(&mut self, sample_f32: &[f32], n: usize, _d: usize) {
        // RESIDQ: re-fit the anisotropic codebook on residual vectors (smaller range -> 4 bits resolve
        // them better -> more accurate IP ranking, +6-11pt pool-recall, P124).
        self.pq = pq::Pq::train_f32_aniso(sample_f32, self.d, self.dpb, n, 6, self.eta);
    }
    fn prepare_query(&self, q: &[i8]) -> QueryCtx {
        let qf: Vec<f32> = q.iter().map(|&v| v as f32).collect();
        // fast-scan: int8 LUT, 1 vpshufb/subspace + i16 accum. ~1.7x scan at ~12-13 bit rank. L2 or IP.
        let ip = IP_MODE.load(std::sync::atomic::Ordering::Relaxed);
        let residq = RESIDQ.load(std::sync::atomic::Ordering::Relaxed);
        if FASTSCAN.load(std::sync::atomic::Ordering::Relaxed) {
            let l = if ip { self.pq.query_lut_f32_i8s_ip(&qf) } else { self.pq.query_lut_f32_i8s(&qf) };
            let regs_z = if USE512FS.load(std::sync::atomic::Ordering::Relaxed) { pq::lut_regs_i8_z512(&l, self.pq.m) } else { Vec::new() };
            let scale = if residq && ip { self.pq.ip_i8s_scale(&qf) } else { 0.0 };
            return QueryCtx::Pq8 { regs: pq::lut_regs_i8(&l, self.pq.m), regs_z, scale };
        }
        if !LUT16_OFF.load(std::sync::atomic::Ordering::Relaxed) {
            let lut = if ip { self.pq.query_lut_f32_i16_ip(&qf) } else { self.pq.query_lut_f32_i16(&qf) };
            let (lo, hi) = pq::lut_regs_i16(&lut, self.pq.m);
            let lut_z = pq::lut_regs_i16_z(&lut, self.pq.m);
            let scale = if residq && ip { self.pq.ip_i16_scale(&qf) } else { 0.0 };
            QueryCtx::Pq16 { lo, hi, lut_z, scale }
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
            QueryCtx::Pq8 { regs, .. } => unsafe { pq::block_adc_i8_i16acc(block, self.pq.m, regs, out16) },
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
    fn encode_block(&self, rows: &[&[i8]], n_real: usize, _cell_cent: &[i8], out: &mut Vec<u8>) {
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
            QueryCtx::Pq16 { lo, hi, lut_z, scale: 0.0 }
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
    fn encode_block(&self, _rows: &[&[i8]], _n: usize, _cell_cent: &[i8], _out: &mut Vec<u8>) {}
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
    // 64-wide AVX-512 fast-scan (USE512FS): per cell, the full groups-of-4 blocks re-interleaved into
    // superblocks (each (m/2)*64 bytes, group g = b0_g|b1_g|b2_g|b3_g). cell_ilstart[cell] = cumulative
    // superblock index where cell `cell`'s superblocks begin. Empty unless built with USE512FS.
    pub blocks_il: Vec<u8>,
    pub cell_ilstart: Vec<u32>,
    // IDEA #4 refine code (empty unless SBANN_RESID): 8-bit PQ codes in SLOT order, m bytes/slot.
    pub resid_pq: Option<pq::ResidPq>,
    pub resid_codes: Vec<u8>,
    // RESIDUAL QUANTIZATION (SBANN_RESIDQ): per-cell raw centroids (nc*d i8). Empty unless RESIDQ.
    // The scan adds <q, rq_cent[cell]> (scaled) so candidate scores = <q,cent>+<q,resid_hat>.
    pub rq_cent: Vec<i8>,
}

impl Index {
    pub fn build(router: Box<dyn Router>, mut comp: Box<dyn Compressor>, ds: &I8Bin, a0: usize) -> Index {
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
        // RESIDUAL QUANTIZATION (SBANN_RESIDQ): per-cell RAW centroids + retrain the codebook on residuals.
        // The cosine-routing centroids are in NORMALIZED space; the scan/rerank is in RAW space, so we
        // compute raw-space centroids here (mean of each cell's raw vectors).
        let residq_on = RESIDQ.load(std::sync::atomic::Ordering::Relaxed);
        let dd = ds.d;
        let rq_cent: Vec<i8> = if residq_on {
            let mut cent = vec![0i8; nc * dd];
            cent.par_chunks_mut(dd).enumerate().for_each(|(c, out)| {
                let (s, e) = (cell_start[c] as usize, cell_start[c + 1] as usize);
                let cnt = (e - s).max(1);
                let mut acc = vec![0f64; dd];
                for &pt in &ids[s..e] { let r = ds.row(pt as usize); for k in 0..dd { acc[k] += r[k] as f64; } }
                for k in 0..dd { out[k] = (acc[k] / cnt as f64).round().clamp(-127.0, 127.0) as i8; }
            });
            // retrain the codebook on a residual sample (raw - its cell's centroid)
            let smp = n.min(40000);
            let stride = (n / smp).max(1);
            let mut x = vec![0f32; smp * dd];
            x.par_chunks_mut(dd).enumerate().for_each(|(i, o)| {
                let oi = i * stride; let r = ds.row(oi); let cell = assign[oi * a0] as usize;
                for k in 0..dd { o[k] = r[k] as f32 - cent[cell * dd + k] as f32; }
            });
            comp.retrain_residual(&x, smp, dd);
            cent
        } else { Vec::new() };
        // encode blocks per cell
        let bb = comp.block_bytes();
        let d = ds.d;
        // IDEA #4: train the 8-bit refine PQ and encode ALL points (parallel, by orig id) up front,
        // so the sequential slot loop below just copies the precomputed code (256-way encode is 16x
        // the 4-bit cost — must be fanned out, not done in the serial append loop).
        let resid_on = RESID.load(std::sync::atomic::Ordering::Relaxed);
        let (resid_pq, resid_by_orig): (Option<pq::ResidPq>, Vec<u8>) = if resid_on {
            let dpb_r: usize = std::env::var("SBANN_RESID_DPB").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
            let iters_r: usize = std::env::var("SBANN_RESID_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(6);
            let smp = n.min(40000);
            let stride = (n / smp).max(1);
            let mut x = vec![0f32; smp * d];
            x.par_chunks_mut(d).enumerate().for_each(|(i, o)| { let r = ds.row(i * stride); for k in 0..d { o[k] = r[k] as f32; } });
            let rpq = pq::ResidPq::train_f32(&x, d, dpb_r, smp, iters_r);
            let mr = rpq.m;
            let mut v = vec![0u8; n * mr];
            v.par_chunks_mut(mr).enumerate().for_each(|(i, out)| {
                let r = ds.row(i);
                let mut xf = vec![0f32; d];
                for k in 0..d { xf[k] = r[k] as f32; }
                rpq.encode_f32(&xf, out);
            });
            (Some(rpq), v)
        } else { (None, Vec::new()) };
        let mr = resid_pq.as_ref().map(|p| p.m).unwrap_or(0);
        let mut resid_codes: Vec<u8> = Vec::new();
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
                let cc: &[i8] = if residq_on { &rq_cent[cell * dd..cell * dd + dd] } else { &[] };
                comp.encode_block(&rows, cnt, cc, &mut blocks);
                for j in 0..16 {
                    slot_orig.push(if j < cnt { pts[i + j] } else { u32::MAX });
                    if j < cnt { raw.extend_from_slice(rows[j]); } else { raw.resize(raw.len() + d, 0); }
                    if mr > 0 {
                        if j < cnt { let o = pts[i + j] as usize; resid_codes.extend_from_slice(&resid_by_orig[o * mr..o * mr + mr]); }
                        else { resid_codes.resize(resid_codes.len() + mr, 0); }
                    }
                }
                i += 16;
            }
            cell_bstart[cell + 1] = if bb > 0 { (blocks.len() / bb) as u32 } else { (slot_orig.len() / 16) as u32 };
        }
        // Optional 64-wide AVX-512 interleaved superblock layout: re-pack each cell's full groups of 4
        // blocks (group g of the superblock = the 16 code-bytes of group g from each of the 4 blocks).
        // Distances computed from this are bit-identical to the AVX2 fast-scan -> recall is unchanged.
        let mut blocks_il: Vec<u8> = Vec::new();
        let mut cell_ilstart: Vec<u32> = Vec::new();
        if bb > 0 && USE512FS.load(std::sync::atomic::Ordering::Relaxed) {
            let m = bb / 8; // bb = (m/2)*16  =>  m = bb/8
            cell_ilstart = vec![0u32; nc + 1];
            let mut nsb_total = 0u32;
            for cell in 0..nc {
                let nb = (cell_bstart[cell + 1] - cell_bstart[cell]) as usize;
                nsb_total += (nb / 4) as u32;
                cell_ilstart[cell + 1] = nsb_total;
            }
            blocks_il = vec![0u8; nsb_total as usize * bb * 4];
            for cell in 0..nc {
                let bs = cell_bstart[cell] as usize;
                let nfull = (cell_bstart[cell + 1] - cell_bstart[cell]) as usize / 4;
                for s in 0..nfull {
                    let b = bs + 4 * s;
                    let sb_idx = cell_ilstart[cell] as usize + s;
                    let (b0, b1, b2, b3) = (
                        &blocks[b * bb..(b + 1) * bb], &blocks[(b + 1) * bb..(b + 2) * bb],
                        &blocks[(b + 2) * bb..(b + 3) * bb], &blocks[(b + 3) * bb..(b + 4) * bb],
                    );
                    pq::interleave4(b0, b1, b2, b3, m, &mut blocks_il[sb_idx * bb * 4..(sb_idx + 1) * bb * 4]);
                }
            }
        }
        Index { router, comp, cell_bstart, slot_orig, blocks, bb, xfn: Vec::new(), raw, d, blocks_il, cell_ilstart, resid_pq, resid_codes, rq_cent }
    }

    pub fn search(&self, ds: &I8Bin, q: &[i8], p: usize, t: usize, k: usize) -> Vec<u32> {
        let cells = self.router.probe(q, p);
        if RESID.load(std::sync::atomic::Ordering::Relaxed) && self.resid_pq.is_some() {
            // refine pool = t survivors; exact-rerank depth from SBANN_RR_DEPTH (default = t = no
            // shallowing); SBANN_RESID_REFINE=0 disables the 8-bit refine (plain-shallow baseline).
            let rr: usize = std::env::var("SBANN_RR_DEPTH").ok().and_then(|s| s.parse().ok()).unwrap_or(t);
            let refine = std::env::var("SBANN_RESID_REFINE").ok().map(|s| s != "0").unwrap_or(true);
            self.scan_rerank_resid(ds, q, &cells, t, rr, refine, k)
        } else {
            self.scan_rerank(ds, q, &cells, t, k)
        }
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
        let use512fs = USE512FS.load(std::sync::atomic::Ordering::Relaxed)
            && !self.blocks_il.is_empty()
            && matches!(ctx, QueryCtx::Pq8 { .. });
        // RESIDQ: add the exact per-cell <q,centroid> offset (in scan i16-units) so candidate scores =
        // <q,cent>+<q,resid_hat>. scale=0 (non-residq) skips it. Applied once per cell after its blocks.
        let rq_scale: f32 = match &ctx { QueryCtx::Pq16 { scale, .. } | QueryCtx::Pq8 { scale, .. } => *scale, _ => 0.0 };
        let residq = rq_scale != 0.0 && !self.rq_cent.is_empty();
        let dd = self.d;
        for &cell in cells {
            let pool_start = pool.len();
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
            } else if use512fs {
                // 64-wide AVX-512 fast-scan over the interleaved superblocks (4 blocks/superblock).
                if let QueryCtx::Pq8 { regs, regs_z, .. } = &ctx {
                    let m = bb / 8;
                    let il0 = self.cell_ilstart[cell as usize] as usize;
                    let nfull = (be - bs) / 4;
                    let mut out64 = [0i32; 64];
                    for s in 0..nfull {
                        let sb = il0 + s;
                        unsafe {
                            pq::block_adc_i8_i16acc_avx512_il(&self.blocks_il[sb * bb * 4..(sb + 1) * bb * 4], m, regs_z, &mut out64);
                        }
                        let bbase = bs + 4 * s;
                        for sub in 0..4 {
                            let slot0 = (bbase + sub) * 16;
                            for j in 0..16 {
                                let slot = slot0 + j;
                                if self.slot_orig[slot] != u32::MAX { pool.push((out64[sub * 16 + j], slot as u32)); }
                            }
                        }
                    }
                    // remainder blocks (< 4): plain AVX2 fast-scan, per block
                    for b in (bs + 4 * nfull)..be {
                        let block = &self.blocks[b * bb..(b + 1) * bb];
                        unsafe { pq::block_adc_i8_i16acc(block, m, regs, &mut out16); }
                        for j in 0..16 {
                            let slot = b * 16 + j;
                            if self.slot_orig[slot] != u32::MAX { pool.push((out16[j], slot as u32)); }
                        }
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
            if residq {
                // dot = <q, cell_centroid>; score is smaller=larger-IP, so subtract scale*dot.
                let dot = -simd::negdot_i8(q, &self.rq_cent[cell as usize * dd..cell as usize * dd + dd]);
                let off = (rq_scale * dot as f32).round() as i32;
                for e in &mut pool[pool_start..] { e.0 -= off; }
            }
        }
        if POOLDEDUP.load(std::sync::atomic::Ordering::Relaxed) {
            // collapse SOAR duplicate slots: keep the min-approx-dist slot per orig id, so the
            // downstream k*4 survivor heap holds DISTINCT ids (true multi-store coverage).
            let mut best: std::collections::HashMap<u32, (i32, u32)> = std::collections::HashMap::with_capacity(pool.len());
            for &(dist, slot) in pool.iter() {
                let orig = self.slot_orig[slot as usize];
                best.entry(orig).and_modify(|e| { if dist < e.0 { *e = (dist, slot); } }).or_insert((dist, slot));
            }
            pool.clear();
            pool.extend(best.into_values());
        }
        let tt = t.min(pool.len());
        if tt > 0 { pool.select_nth_unstable(tt - 1); pool.truncate(tt); }
        let _ = ds;
        rerank_contig(&self.raw, self.d, &self.slot_orig, q, &pool, k)
    }

    /// IDEA #4 refine path. Scan -> select top-`t_surv` by 4-bit ADC -> (optional) REFINE that pool
    /// with the 8-bit code (reads only m bytes/survivor from resid_codes, NOT the d-byte raw) -> keep
    /// the top-`rr_depth` by the refined order -> exact raw rerank ONLY those rr_depth. `refine=false`
    /// keeps the 4-bit-ADC order (plain-shallow baseline) for a clean same-index A/B. The headline:
    /// at fixed recall, the refined order needs a far smaller rr_depth -> far fewer raw-vector reads.
    pub fn scan_rerank_resid(&self, ds: &I8Bin, q: &[i8], cells: &[u32], t_surv: usize, rr_depth: usize, refine: bool, k: usize) -> Vec<u32> {
        let ctx = self.comp.prepare_query(q);
        let need_rows = self.comp.needs_raw_rows();
        let mut pool: Vec<(i32, u32)> = Vec::with_capacity(8192);
        let mut out16 = [0i32; 16];
        let mut out32 = [0i32; 32];
        let bb = self.bb;
        for &cell in cells {
            let (bs, be) = (self.cell_bstart[cell as usize] as usize, self.cell_bstart[cell as usize + 1] as usize);
            if need_rows {
                for b in bs..be {
                    let block = if bb > 0 { &self.blocks[b * bb..(b + 1) * bb] } else { &[][..] };
                    let rows16: Vec<&[i8]> = (0..16).map(|j| {
                        let o = self.slot_orig[b * 16 + j];
                        if o != u32::MAX { ds.row(o as usize) } else { &[][..] }
                    }).collect();
                    self.comp.scan_block(block, &ctx, q, &rows16, &mut out16);
                    for j in 0..16 { let slot = b * 16 + j; if self.slot_orig[slot] != u32::MAX { pool.push((out16[j], slot as u32)); } }
                }
            } else {
                let mut b = bs;
                while b + 1 < be {
                    let b0 = &self.blocks[b * bb..(b + 1) * bb];
                    let b1 = &self.blocks[(b + 1) * bb..(b + 2) * bb];
                    self.comp.scan_block_x2(b0, b1, &ctx, &mut out32);
                    for j in 0..16 { let slot = b * 16 + j; if self.slot_orig[slot] != u32::MAX { pool.push((out32[j], slot as u32)); } }
                    for j in 0..16 { let slot = (b + 1) * 16 + j; if self.slot_orig[slot] != u32::MAX { pool.push((out32[16 + j], slot as u32)); } }
                    b += 2;
                }
                if b < be {
                    let block = &self.blocks[b * bb..(b + 1) * bb];
                    self.comp.scan_block(block, &ctx, q, &[], &mut out16);
                    for j in 0..16 { let slot = b * 16 + j; if self.slot_orig[slot] != u32::MAX { pool.push((out16[j], slot as u32)); } }
                }
            }
        }
        // select top-t_surv by the 4-bit ADC distance (the candidate pool entering refine)
        let tt = t_surv.min(pool.len());
        if tt > 0 { pool.select_nth_unstable(tt - 1); pool.truncate(tt); }
        // refine: re-key the survivors by the 8-bit refine distance (cheap m-byte reads), else keep ADC
        let rpq = self.resid_pq.as_ref().expect("resid_pq");
        let mr = rpq.m;
        let mut keyed: Vec<(f32, u32)> = if refine {
            let qf: Vec<f32> = q.iter().map(|&v| v as f32).collect();
            // IP mode (OOD/MIPS): refine by approx INNER PRODUCT (-<q,decode>), matching the IP exact
            // rerank below; L2 otherwise. Same 8-bit codes, just a different query LUT.
            let lut = if IP_MODE.load(std::sync::atomic::Ordering::Relaxed) { rpq.query_lut_f32_ip(&qf) } else { rpq.query_lut_f32(&qf) };
            pool.iter().map(|&(_, slot)| {
                let code = &self.resid_codes[slot as usize * mr..slot as usize * mr + mr];
                (rpq.adc(code, &lut), slot)
            }).collect()
        } else {
            pool.iter().map(|&(adc, slot)| (adc as f32, slot)).collect()
        };
        // keep the top-rr_depth by the (refined or ADC) order -> these are the ONLY raw reads
        let dd = rr_depth.min(keyed.len());
        if dd > 0 { keyed.select_nth_unstable_by(dd - 1, |a, b| a.0.total_cmp(&b.0)); keyed.truncate(dd); }
        let pool2: Vec<(i32, u32)> = keyed.iter().map(|&(_, slot)| (0i32, slot)).collect();
        rerank_contig(&self.raw, self.d, &self.slot_orig, q, &pool2, k)
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
