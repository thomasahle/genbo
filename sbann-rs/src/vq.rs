//! Pluggable vector-quantization engine. Two trait slots — Router (coarse: which cells to scan)
//! and Compressor (candidate scan: approx distances) — composed by `Index`. Swap PQ/AQ/OPQ/scalar
//! for either slot at runtime via `Box<dyn ..>`; dispatch is per 16-point block, so no hot-loop cost.

use crate::ibin::I8Bin;
use crate::{kmeans, pq, simd};
use rayon::prelude::*;
use std::arch::x86_64::{__m128i, __m256i, __m512i, _mm_prefetch, _MM_HINT_T0};

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
/// true => PROPER FastScan (André 2015 / Quicker-ADC): 32-wide 256-bit vpshufb + int8 SATURATING
/// accumulate with periodic int16 hoist (block_adc_i8_fastscan32_2x16). Uses a BOUNDED [0,15] LUT
/// (query_lut_f32_i8s_fs2) so hoist groups don't saturate. ~1.8x kernel vs the crude 16-wide int16
/// fast-scan when compute/L2-bound; the exact rerank restores order past the coarser LUT. Set from
/// SBANN_FASTSCAN2 (implies the Pq8 path, adds the 256-bit LUT regs). Do NOT combine with USE512FS.
pub static FASTSCAN2: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// SBANN_FUSEDTOPK: ScaNN-style fused top-t collect. Instead of materializing EVERY candidate
/// (dist,slot) into a pool and select_nth-ing over all of them, keep a running t-th-best threshold and
/// SIMD-compare each block's kernel dists against it, pushing only survivors (dist<=thr) into a bounded
/// buffer that is periodically pruned back to t. Recall-neutral: the buffer is provably a superset of
/// the true top-t (thr only tightens, and a true-top-t element can never be pruned), so the final
/// select_nth over the buffer yields the identical top-t set as select_nth over the full pool — but the
/// O(candidates) scalar push + per-candidate slot_orig branch (the measured ~85% of the scan phase,
/// FINDINGS P187) is replaced by a SIMD threshold-compare that only touches slot_orig for survivors.
/// Gated to the NON-residq, non-pool-dedup path (see scan_rerank); falls back to scan_pool otherwise.
pub static FUSEDTOPK: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// IDEA #4: build a SECOND finer 8-bit refine code (pq::ResidPq) in slot order and use it to refine
/// the 4-bit-ADC survivor ranking before the exact raw rerank, so far fewer raw vectors are read.
/// Set from SBANN_RESID. SBANN_RESID_DPB picks the refine subspace size (default 2 => m=d/2 bytes/vec).
pub static RESID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// true => RESIDUAL QUANTIZATION (SBANN_RESIDQ): the PRIMARY scan code encodes x - cell_centroid (codebook
/// retrained on residuals), and the scan adds the exact per-cell <q,centroid> offset. +6-11pt IP
/// pool-recall (P124) -> shallower rerank pool for OOD. Distinct from RESID (8-bit refine, which failed).
pub static RESIDQ: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// SBANN_RAW_DEDUP (Task B): store the exact-rerank `raw` array per DISTINCT ORIG point (n*d, indexed
/// by orig id) instead of per SLOT (n*a0*d, a full a0x-oversized 2nd dataset copy). rerank_contig reads
/// raw[orig*d] when set. Shrinks the largest index array by ~a0x with bit-identical recall (a duplicate
/// slot's orig points at the same raw bytes either way). Set once at startup from the env in main().
pub static RAW_DEDUP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// SBANN_POOLDEDUP: dedup the candidate pool by ORIG id (keep min approx-dist per id) BEFORE the
/// t_surv survivor cap. With SOAR a0>1 a point lands in multiple probed cells as duplicate slots; the
/// late dedup in rerank_contig (heap size k*4) gets crowded out by those duplicates, collapsing recall
/// as a0 grows. This dedup measures the TRUE coverage of a multi-store routing (and shrinks the pool).
pub static POOLDEDUP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// SBANN_DEDUP_A0: smallest a0 at which scan_rerank dedups the pool by orig BEFORE the t_surv cap.
/// a0 < this uses only the cheap dedup-in-rerank (after-cap) path -- negligible recall loss at low a0
/// (1M OOD a0=3 = -0.003) but avoids the per-query dedup cost on the QPS@90% / msspacev champion configs.
/// a0 >= this uses the correct before-cap dedup (needed at heavy duplication, e.g. a0>=6). Default 4.
pub static DEDUP_A0: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(4);
/// SBANN_PROFILE: accumulate per-component query time (nanos) to see where the 10M query goes
/// (route vs scan vs rerank). Load-robust (report the FRACTIONS, not absolute). main.rs prints+resets.
pub static PROFILE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub static PROF_ROUTE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PROF_SCAN_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PROF_RERANK_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// ROUTE-PRIMITIVE attribution (P196): break gather_fine/route_fine into phases. Guarded by ROUTE_PROF so
/// the hot path is untouched in normal runs. Populated only in a separate profiling pass (routebench).
pub static ROUTE_PROF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub static PROF_R_COARSE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0); // coarse l2 block + build cd
pub static PROF_R_CSEL_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);   // coarse select_nth + sel build
pub static PROF_R_FINE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);   // fine-level expand loop (l2 + gather)
pub static PROF_R_FSEL_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);   // final top-p select_nth
pub static PROF_R_NEVAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);     // int8 centroid dist-evals (coarse+fine)
/// int8-cascade stage time (P194): the mid-precision VNNI int8 rerank that prunes the apq4 survivor pool
/// down to CASCADE_K before the expensive FLOAT reorder. Separate counter so the profile can split
/// int8-cascade vs float-reorder.
pub static PROF_CASC_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// SBANN_CASCADE (P194 rerank cascade): after the apq4 scan yields the survivor pool, insert a CHEAP
/// full-precision INT8 rerank (VNNI dpbusd over the slot-contiguous raw i8 store) to prune the pool from
/// t_surv (~464) down to CASCADE_K, then FLOAT-reorder only those K. int8 ranks far better than the 4-bit
/// apq4 code, so the true float-top-10 survive the prune at small K -> cuts the ~180ns/vec float reorder
/// count ~4-7x. Recall must be verified >= the float-rerank-only baseline (the prune is not free of risk).
pub static CASCADE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// CASCADE_K: how many int8-top survivors to pass to the float reorder (SBANN_CASCADE_K). Tuned to the
/// minimum that HOLDS recall@10 >= baseline.
pub static CASCADE_K: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(128);
/// CASC_SORT (P194): sort the deduped survivor pool by SLOT before the int8 gather. `raw` is slot-
/// contiguous, so slot-ascending order makes the int8 gather read MONOTONICALLY forward -> HW prefetch
/// + TLB stream instead of a random scatter (the P189 scattered-read lever, applied to the int8 stage).
/// Recall-EXACTLY-neutral (int8 dist is order-independent). Default OFF: measured NET-NEGATIVE (the
/// per-query sort over ~250 survivors costs more than it saves; the i+8 prefetch already hides the
/// slot-clustered gather latency). SBANN_CASC_SORT re-enables for A/B.
pub static CASC_SORT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// CASC_DIM (P194 probe): cap the #dims used in the int8-cascade dot (0 = full d). A coarse single-level
/// prune reading fewer cache lines per survivor -> tests whether the int8 gather is BANDWIDTH-bound
/// (fewer lines = faster) or LATENCY-bound (first-line miss dominates, no gain). Recall may drop.
pub static CASC_DIM: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// SBANN_SCANDIAG diagnostic: run ONLY the kernel floor (block reads + LUT, NO collect) and record its
/// time as the scan phase, so a separate run gives collect = scan_full - scan_kernelonly (same per-query
/// cold-cache pattern). Isolates how much of scan the fused top-t can actually remove (only the collect
/// part; the scattered block reads + LUT are t-independent and untouchable by the fused path).
pub static SCANDIAG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// SBANN_ROUTE_SDIM: score only the first N dims of each centroid at the FINEST routing level (the 78%-of-
/// routing term, P139). 0 = full d (exact). Approximate finest routing -> cheaper routing if recall@p holds.
pub static ROUTE_SDIM: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// SBANN_ROUTE_ADC: 4-bit ADC scoring of the finest centroids (recall gate for #3). ROUTE_ADC_KEEP = how
/// many ADC-top children to exact-rescore (default 1024). Built only when the flag is set at train time.
pub static ROUTE_ADC: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub static ROUTE_ADC_KEEP: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1024);
/// SBANN_ROUTE_VNNI (P196): compute the routing centroid L2 with the VNNI norm-decomposition kernel
/// (L2 = Σq²+Σc²−2·dot, dpbusd dot) instead of the AVX2-madd Σ(q−c)². Bit-identical cell selection
/// (recall-EXACT), ~1.5x on the dominant compute (66% of route). Full-dim only; sd<d stays on madd.
pub static ROUTE_VNNI: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// SBANN_SORTCELLS (WALL-1 scattered-read lever): sort the probed cell list ASCENDING (= block/memory
/// order, since cell_bstart is monotonic in cell id) before scanning. route_fine's select_nth returns
/// cells in SCRAMBLED order, so consecutive scanned cells make random jumps across the ~168MB blocks
/// array; sorting makes the block reads MONOTONICALLY forward -> HW prefetch + TLB stream instead of
/// stall. RECALL-EXACTLY-NEUTRAL (same cells, same candidates, only the visit order changes; the top-t
/// select is order-independent). Applied in scan_rerank (covers the full + SCANDIAG paths).
pub static SORTCELLS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// SBANN_PREFETCH: software-prefetch (T0) the NEXT probed cell's block memory while scanning the current
/// cell, to hide the cross-cell random-jump latency (the scan is memory-LATENCY bound on p random jumps).
/// SBANN_PFDIST = how many cells ahead to prefetch (default 2). SBANN_PFLINES = cache lines/cell to touch
/// (default 13 = the full 800B PQ block). Recall-neutral (prefetch is a hint; results identical). Applied
/// in scan_pool + kernel_only. Tuned on 1M OOD (P189): raises the scattered kernel FLOOR +45-53% Mcand/s,
/// which nets +8-11% e2e QPS@recall>=0.90 (block-reads are only ~26% of the e2e query). pfdist 2, full block.
pub static PREFETCH: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub static PFDIST: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(2);
pub static PFLINES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(13);

thread_local! {
    // reused open-addressing table for the per-query pool dedup (a0>1). Entries: (orig_key, best_approx,
    // slot); orig_key==u32::MAX marks empty. Fibonacci-hashed + linear-probed -> far cheaper than a
    // per-query std HashMap (no SipHash, no alloc), keeping the min-approx slot per distinct orig id.
    static DEDUP_TBL: std::cell::RefCell<Vec<(u32, i32, u32)>> = std::cell::RefCell::new(Vec::new());
}

/// Software-prefetch (T0) `nlines` cache lines starting at `ptr`. Used by the per-cell scan loop to
/// pull the NEXT probed cell's PQ blocks into cache while the current cell is still being scanned,
/// hiding the ~100ns cross-cell random-jump latency. A hint only -> zero effect on results.
#[inline(always)]
fn prefetch_lines(base: *const u8, len: usize, byte_off: usize, nlines: usize) {
    if byte_off >= len { return; }
    let avail = len - byte_off;
    let n = nlines.min(avail.div_ceil(64));
    unsafe {
        let p = base.add(byte_off);
        for i in 0..n {
            _mm_prefetch(p.add(i * 64) as *const i8, _MM_HINT_T0);
        }
    }
}

/// Dedup `pool` (approx_dist, slot) IN PLACE to one entry per distinct orig id (the min-approx slot),
/// using a reused thread-local open-addressing table. Must run BEFORE the t_surv cap so the cap selects
/// distinct survivors. slot_orig maps slot->orig (valid: pushed slots are never u32::MAX).
fn dedup_pool_by_orig(pool: &mut Vec<(i32, u32)>, slot_orig: &[u32]) {
    if pool.len() < 2 { return; }
    DEDUP_TBL.with(|tbl| {
        let mut tbl = tbl.borrow_mut();
        let cap = (pool.len() * 2).next_power_of_two();
        tbl.clear();
        tbl.resize(cap, (u32::MAX, 0, 0));
        let mask = cap - 1;
        for &(dist, slot) in pool.iter() {
            let orig = slot_orig[slot as usize];
            let mut h = (orig.wrapping_mul(0x9E3779B1) as usize) & mask;
            loop {
                let e = tbl[h];
                if e.0 == u32::MAX {
                    tbl[h] = (orig, dist, slot);
                    break;
                } else if e.0 == orig {
                    if dist < e.1 { tbl[h] = (orig, dist, slot); }
                    break;
                }
                h = (h + 1) & mask;
            }
        }
        pool.clear();
        for &(k, bd, bs) in tbl.iter() {
            if k != u32::MAX { pool.push((bd, bs)); }
        }
    });
}

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
    // bounded heap of the m smallest exact dists, DEDUPED by id (SOAR multi-assignment can surface the same
    // point via several probed cells; a duplicate has identical raw -> identical dist, so it is skipped
    // rather than allowed to crowd the m slots — the bug that POOLDEDUP worked around, fixed at the source).
    let mut heap: std::collections::BinaryHeap<(i32, u32)> = std::collections::BinaryHeap::with_capacity(m + 1);
    for i in 0..n {
        let id = pool[i].1;
        if i + 8 < n {
            unsafe { _mm_prefetch(ds.row(pool[i + 8].1 as usize).as_ptr() as *const i8, _MM_HINT_T0) };
        }
        let row = ds.row(id as usize);
        let dist = if avx { unsafe { simd::l2_i8_avx2(q, row) } } else { simd::l2_i8_scalar(q, row) };
        let full = heap.len() >= m;
        if full && dist >= heap.peek().unwrap().0 { continue; }
        if heap.iter().any(|&(_, o)| o == id) { continue; }
        if full { heap.pop(); }
        heap.push((dist, id));
    }
    let mut v = heap.into_vec();
    v.sort_unstable();
    let mut out = Vec::with_capacity(k);
    for &(_, id) in &v {
        out.push(id);
        if out.len() == k {
            break;
        }
    }
    out
}

/// Like rerank_survivors but survivors are SLOTS into a cell-contiguous raw i8 array (`raw`), so the
/// gathers stay inside the small probed-cell region (cache-warm) instead of scattering across the
/// full base. Pool = (approx_dist, slot); returns up to k DISTINCT orig ids by true L2.
// `by_orig` (Task B, SBANN_RAW_DEDUP): when true `raw` is per-distinct-orig (n*d), so a slot's row lives
// at raw[orig*d]; when false `raw` is slot-indexed (n*a0*d) and the row lives at raw[slot*d]. The chosen
// rows are bit-identical either way (a duplicate slot's orig points at the same bytes), so recall is unchanged.
fn rerank_contig(raw: &[i8], d: usize, slot_orig: &[u32], q: &[i8], pool: &[(i32, u32)], k: usize, by_orig: bool) -> Vec<u32> {
    rerank_contig_pairs(raw, d, slot_orig, q, pool, k, by_orig).into_iter().take(k).map(|(_, o)| o).collect()
}

/// Like `rerank_contig` but returns the full sorted `(exact_dist, orig)` heap (up to `k*4` entries),
/// not just the top-k ids. The streaming search reranks the main index and the per-cell append buffer
/// SEPARATELY (they read different raw stores) and merges these scored lists, so it needs the dists.
fn rerank_contig_pairs(raw: &[i8], d: usize, slot_orig: &[u32], q: &[i8], pool: &[(i32, u32)], k: usize, by_orig: bool) -> Vec<(i32, u32)> {
    let m = (k * 4).min(pool.len());
    if m == 0 {
        return Vec::new();
    }
    let avx = std::is_x86_feature_detected!("avx2");
    let ip = IP_MODE.load(std::sync::atomic::Ordering::Relaxed);
    let n = pool.len();
    // bounded heap of the m smallest exact dists, DEDUPED by orig id. SOAR multi-store places a point in
    // several cells as duplicate slots; same orig => identical raw => identical dist, so the heap stores
    // (dist, orig) and skips a duplicate orig instead of letting copies crowd the m slots (the bug the
    // per-query POOLDEDUP HashMap papered over; deduping here removes the need for that O(poolsize) map).
    let mut heap: std::collections::BinaryHeap<(i32, u32)> = std::collections::BinaryHeap::with_capacity(m + 1);
    for i in 0..n {
        let slot = pool[i].1 as usize;
        let orig = slot_orig[slot];
        if i + 8 < n {
            let nslot = pool[i + 8].1 as usize;
            let ri = if by_orig { slot_orig[nslot] as usize } else { nslot };
            unsafe { _mm_prefetch(raw.as_ptr().add(ri * d) as *const i8, _MM_HINT_T0) };
        }
        if orig == u32::MAX { continue; }
        let ri = if by_orig { orig as usize } else { slot };
        let row = &raw[ri * d..ri * d + d];
        let dist = if ip { simd::negdot_i8(q, row) } else if avx { unsafe { simd::l2_i8_avx2(q, row) } } else { simd::l2_i8_scalar(q, row) };
        let full = heap.len() >= m;
        if full && dist >= heap.peek().unwrap().0 { continue; }
        if heap.iter().any(|&(_, o)| o == orig) { continue; }
        if full { heap.pop(); }
        heap.push((dist, orig));
    }
    let mut v = heap.into_vec();
    v.sort_unstable();
    v
}

/// FLOAT-RERANK (P191 lever stack): identical survivor pool to `rerank_contig`, but the exact rerank
/// reads the ORIGINAL float vectors (mmap'd `fbase`, indexed by orig id) and scores by exact float IP
/// (`-dot` so smaller = better, matching the int8 IP path). Only the survivors are paged in, so the
/// float base stays near-zero resident. Breaks the int8 quantization ceiling vs the float-computed GT.
fn rerank_contig_float(fbase: &crate::fbin::FBin, slot_orig: &[u32], qf: &[f32], pool: &[(i32, u32)], k: usize) -> Vec<u32> {
    let mut scored: Vec<(f32, u32)> = Vec::with_capacity(pool.len());
    let n = pool.len();
    for i in 0..n {
        let slot = pool[i].1 as usize;
        let orig = slot_orig[slot];
        if i + 8 < n {
            let no = slot_orig[pool[i + 8].1 as usize];
            if no != u32::MAX { unsafe { _mm_prefetch(fbase.row(no as usize).as_ptr() as *const i8, _MM_HINT_T0) }; }
        }
        if orig == u32::MAX { continue; }
        let row = fbase.row(orig as usize);
        scored.push((-simd::dot_f32_fast(qf, row), orig));
    }
    // partial-select the k*4 best, then sort that prefix; dedup origs (SOAR duplicate slots -> same orig
    // -> identical dist) while taking the top k distinct.
    let m = (k * 4).min(scored.len());
    if m > 0 { scored.select_nth_unstable_by(m - 1, |a, b| a.0.total_cmp(&b.0)); scored.truncate(m); }
    scored.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let mut out = Vec::with_capacity(k);
    for &(_, o) in &scored {
        if !out.contains(&o) { out.push(o); if out.len() == k { break; } }
    }
    out
}

/// INT8-CASCADE FLOAT-RERANK (P194): a cheap mid-precision INT8 stage between the apq4 scan and the
/// expensive float reorder. The apq4 survivor `pool` (up to t_surv slots, dist = 4-bit code score) is:
///   (1) deduped by orig (SOAR a0>1 stores a point in several probed cells as duplicate slots; identical
///       raw -> identical dist, so deduping is recall-neutral and shrinks the int8/float work);
///   (2) INT8-rescored via VNNI dpbusd over the SLOT-CONTIGUOUS raw i8 store (200-dim int8 dot, ~4-6x
///       fewer instrs than the AVX2 madd path and 4x less memory traffic than the 800B float row);
///   (3) pruned to the `kk` int8-smallest;
///   (4) FLOAT-reordered (exact IP vs the leaderboard float GT) over ONLY those `kk`.
/// int8 ranks far better than the 4-bit apq4 code (int8 recall-ceiling ~0.924 on t2i OOD, >> the 0.90
/// target), so the true float-top-k survive the prune at small kk -> the ~180ns/vec float reorder count
/// drops from ~464 to kk (~64-128). Recall MUST be verified >= the float-rerank-only baseline.
#[allow(clippy::too_many_arguments)]
fn rerank_cascade_float(fbase: &crate::fbin::FBin, raw: &[i8], d: usize, raw_orig_indexed: bool,
    slot_orig: &[u32], q: &[i8], qf: &[f32], pool: &mut Vec<(i32, u32)>, kk: usize, k: usize) -> Vec<u32> {
    let prof = PROFILE.load(std::sync::atomic::Ordering::Relaxed);
    let tc = if prof { Some(std::time::Instant::now()) } else { None };
    // (1) dedup by orig (recall-neutral): keeps one slot per distinct orig, min apq4 dist.
    dedup_pool_by_orig(pool, slot_orig);
    let n = pool.len();
    if n == 0 { return Vec::new(); }
    // (1b) sort by slot so the slot-contiguous raw gather streams forward (recall-neutral).
    if !raw_orig_indexed && CASC_SORT.load(std::sync::atomic::Ordering::Relaxed) {
        pool.sort_unstable_by_key(|&(_, s)| s);
    }
    // (2) INT8 rescore. Prefer VNNI (dpbusd) when the CPU has it; else AVX2 madd; else scalar.
    let vnni = std::is_x86_feature_detected!("avx512vnni") && std::is_x86_feature_detected!("avx512bw")
        && std::is_x86_feature_detected!("avx512f");
    let avx = std::is_x86_feature_detected!("avx2");
    let cdim = CASC_DIM.load(std::sync::atomic::Ordering::Relaxed);
    let dd = if cdim == 0 { d } else { cdim.min(d) };
    let qd = &q[..dd];
    let mut scored: Vec<(i32, u32)> = Vec::with_capacity(n);
    for i in 0..n {
        let slot = pool[i].1 as usize;
        let orig = slot_orig[slot];
        if i + 8 < n {
            let nslot = pool[i + 8].1 as usize;
            let ri = if raw_orig_indexed { slot_orig[nslot] as usize } else { nslot };
            if ri != u32::MAX as usize { unsafe { _mm_prefetch(raw.as_ptr().add(ri * d) as *const i8, _MM_HINT_T0) }; }
        }
        if orig == u32::MAX { continue; }
        let ri = if raw_orig_indexed { orig as usize } else { slot };
        let row = &raw[ri * d..ri * d + dd];
        // negdot: smaller = better (matches the IP float path's -dot).
        let dist = if vnni { -unsafe { simd::dot_i8_vnni(qd, row) } }
                   else if avx { -unsafe { simd::dot_i8_avx2(qd, row) } }
                   else { simd::negdot_i8(qd, row) };
        scored.push((dist, slot as u32));
    }
    // (3) prune to the kk int8-smallest (already distinct origs after the dedup above).
    let kk = kk.min(scored.len());
    if kk > 0 && kk < scored.len() { scored.select_nth_unstable(kk - 1); scored.truncate(kk); }
    if let Some(tc) = tc { PROF_CASC_NS.fetch_add(tc.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); }
    // (4) exact float reorder over only the kk survivors.
    let tr = if prof { Some(std::time::Instant::now()) } else { None };
    let out = rerank_contig_float(fbase, slot_orig, qf, &scored, k);
    if let Some(tr) = tr { PROF_RERANK_NS.fetch_add(tr.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); }
    out
}

// ---------------- Fused top-t collect (SBANN_FUSEDTOPK, ScaNN keep-only-survivors) ----------------

/// Running state for the fused top-t collect. `buf` holds the current survivor set; once `filled`,
/// `thr` is the current t-th-smallest dist (admission threshold) and any candidate with dist>thr is
/// dropped WITHOUT touching slot_orig. Pruned back to `t` whenever `buf` reaches `prune_cap`.
struct FusedTopT {
    buf: Vec<(i32, u32)>,
    thr: i32,
    t: usize,
    prune_cap: usize,
    filled: bool,
}

impl FusedTopT {
    #[inline]
    fn new(t: usize) -> Self {
        // prune_cap = 2t: prune to t once the buffer doubles, so each O(prune_cap) select_nth is amortized
        // over ~t admits (O(1) amortized). After the first prune thr is tight and survivors trickle in, so
        // subsequent prunes are rare. (Tighter caps like t+t/4 prune far more often -> the select_nth
        // passes dominate and net-lose; measured.) thr starts at MAX (admit everything until we have t).
        let prune_cap = (2 * t).max(t + 64);
        FusedTopT { buf: Vec::with_capacity(prune_cap + 64), thr: i32::MAX, t, prune_cap, filled: false }
    }
    #[inline]
    fn maybe_prune(&mut self) {
        if self.buf.len() >= self.prune_cap {
            // keep the t smallest seen so far; thr := their max (= t-th smallest). A true-top-t element
            // is always among the t smallest-so-far (fewer than t elements are globally smaller than it),
            // so pruning never drops one -> the final result set is identical to select_nth over the full pool.
            self.buf.select_nth_unstable(self.t - 1);
            self.thr = self.buf[self.t - 1].0;
            self.buf.truncate(self.t);
            self.filled = true;
        }
    }
    /// Emit survivors from one 16- or 32-lane block: `out[j]` is the kernel dist for slot `base+j`.
    #[inline]
    fn emit(&mut self, out: &[i32], base: usize, slot_orig: &[u32]) {
        if self.filled {
            // SIMD: mask of lanes with out[j] <= thr, then push only those (checking slot_orig per survivor).
            let mut mask = unsafe { survivor_mask_leq(out, self.thr) };
            while mask != 0 {
                let j = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                let slot = base + j;
                if slot_orig[slot] != u32::MAX {
                    self.buf.push((out[j], slot as u32));
                }
            }
        } else {
            // not yet t candidates: admit all valid lanes (thr is still MAX).
            for (j, &d) in out.iter().enumerate() {
                let slot = base + j;
                if slot_orig[slot] != u32::MAX {
                    self.buf.push((d, slot as u32));
                }
            }
        }
        self.maybe_prune();
    }
    /// Final cap: exactly the t smallest of the buffer (identical set to select_nth over the full pool).
    #[inline]
    fn finish(mut self) -> Vec<(i32, u32)> {
        let tt = self.t.min(self.buf.len());
        if tt > 0 && tt < self.buf.len() {
            self.buf.select_nth_unstable(tt - 1);
            self.buf.truncate(tt);
        }
        self.buf
    }
}

/// Return a bitmask (bit j set) of lanes where `out[j] <= thr`, for `out.len()` <= 64. Uses AVX2
/// packed 32-bit compare (out<=thr <=> !(out>thr) <=> (thr+1)>out via _mm256_cmpgt_epi32). Callers
/// only invoke this once `thr` is a real (finite, < i32::MAX) dist, so thr+1 never overflows.
#[inline]
unsafe fn survivor_mask_leq(out: &[i32], thr: i32) -> u64 {
    use std::arch::x86_64::*;
    let n = out.len();
    debug_assert!(n <= 64);
    if std::is_x86_feature_detected!("avx2") {
        let thr1 = _mm256_set1_epi32(thr.wrapping_add(1)); // out <= thr  <=>  thr+1 > out
        let mut mask: u64 = 0;
        let mut j = 0usize;
        while j + 8 <= n {
            let v = _mm256_loadu_si256(out.as_ptr().add(j) as *const __m256i);
            let cmp = _mm256_cmpgt_epi32(thr1, v);
            let m = _mm256_movemask_ps(_mm256_castsi256_ps(cmp)) as u32;
            mask |= (m as u64) << j;
            j += 8;
        }
        while j < n {
            if out[j] <= thr { mask |= 1u64 << j; }
            j += 1;
        }
        mask
    } else {
        let mut mask: u64 = 0;
        for (j, &d) in out.iter().enumerate() {
            if d <= thr { mask |= 1u64 << j; }
        }
        mask
    }
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
    /// Serialize self (1-byte concrete-type tag + POD fields) for SBANN_INDEX_SAVE. Default: error —
    /// only the scale-path router (HierRouter) implements it; load_router reads the tag back.
    fn save(&self, _w: &mut crate::persist::Sw) -> std::io::Result<()> {
        Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "router type not serializable (only HierRouter is)"))
    }
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
    // ADC routing (#3, SBANN_ROUTE_ADC): a 4-bit PQ over the FINEST centroids so the finest-level expansion
    // (the 78%-of-routing term, P139) is scored by cheap ADC instead of exact i8 L2, then only the ADC-top
    // ROUTE_ADC_KEEP are exact-rescored. radc=codebook, rcodes=kf*m codes. Empty unless built with the flag.
    radc: Option<pq::Pq>,
    rcodes: Vec<u8>,
    // finest codes re-laid into 16-cell vpshufb blocks (m/2 groups * 16 bytes each, cell order). Lets the
    // ADC finest scoring use block_adc_i8_i16acc (16 centroids/instr) instead of the scalar LUT-sum.
    rblocks: Vec<u8>,
    // per-level, per-centroid `cadj = Σc² + 256·Σc` (i32, exact). Lets the routing L2 use the single-chain
    // VNNI decomposition L2 = Σq² + cadj − 2·Σ(q+128)c (dpbusd; ~1.5x the AVX2-madd L2 on Zen4), bit-identical
    // to the direct Σ(q-c)². Derived from `cent` at build/load (NOT persisted). Empty => VNNI path disabled.
    cadj: Vec<Vec<i32>>,
}

/// `cadj = Σc² + 256·Σc` (simd::cadj_i8) for every centroid in each per-level block. Parallel to `cent`.
fn cadj_of(cent: &[Vec<i8>], d: usize) -> Vec<Vec<i32>> {
    cent.iter().map(|lvl| {
        let n = if d > 0 { lvl.len() / d } else { 0 };
        (0..n).map(|j| crate::simd::cadj_i8(&lvl[j * d..j * d + d])).collect()
    }).collect()
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
        // optional ADC routing codebook over the FINEST centroids (recall gate for #3)
        let (radc, rcodes) = if ROUTE_ADC.load(std::sync::atomic::Ordering::Relaxed) && d % 4 == 0 {
            let cf = &cent[levels - 1];
            let kfn = cf.len() / d;
            let rows: Vec<&[i8]> = (0..kfn).map(|i| &cf[i * d..i * d + d]).collect();
            let pq = pq::Pq::train(&rows, d, 2, 8); // dpb=2 -> m=d/2 (even for d%4==0)
            let m = pq.m;
            let mut codes = vec![0u8; kfn * m];
            for i in 0..kfn { pq.encode(&cf[i * d..i * d + d], &mut codes[i * m..i * m + m]); }
            println!("  [ROUTE_ADC: 4-bit PQ over {kfn} finest centroids, m={m}]");
            (Some(pq), codes)
        } else { (None, Vec::new()) };
        // re-lay the finest codes into 16-cell vpshufb blocks (only when kf%16==0; our Kf are powers of 2)
        let rblocks: Vec<u8> = if let Some(pq) = &radc {
            let m = pq.m; let nb = kf / 16; let mut rb = vec![0u8; nb * (m / 2) * 16];
            if kf % 16 == 0 {
                for b in 0..nb {
                    for g in 0..m / 2 {
                        for i in 0..16 {
                            let c = b * 16 + i;
                            let lo = rcodes[c * m + 2 * g] & 0x0f;
                            let hi = rcodes[c * m + 2 * g + 1] & 0x0f;
                            rb[(b * (m / 2) + g) * 16 + i] = lo | (hi << 4);
                        }
                    }
                }
            }
            rb
        } else { Vec::new() };
        let cadj = cadj_of(&cent, d);
        HierRouter { d, mu, kf, levels, cent, child, beam: beams.to_vec(), soar: 0.0, radc, rcodes, rblocks, cadj }
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
        let cent = vec![c0, cf];
        let cadj = cadj_of(&cent, d);
        HierRouter { d, mu, kf, levels: 2, cent, child: vec![gstart, Vec::new()], beam: vec![b0], soar: 0.0, radc: None, rcodes: Vec::new(), rblocks: Vec::new(), cadj }
    }

    /// Enable SOAR build-time spilled assignment with penalty λ (`s`). 0 disables (default path).
    pub fn set_soar(&mut self, s: f32) { self.soar = s; }

    /// General L-level descent: top-beam[0] coarse → expand to children → top-beam[l] … → collect ALL
    /// finest-level candidates as (l2, fine_id) into `fd` (NOT truncated — route_fine does top-k,
    /// route_fine_soar does the SOAR spill). One code path for every depth L>=2.
    fn gather_fine(&self, qn: &[i8], fd: &mut Vec<(i32, u32)>) {
        let d = self.d;
        let l0 = self.cent[0].len() / d;
        // batched block L2 (qn held in registers, 2-wide ILP) instead of a scalar per-centroid l2_i8 loop;
        // routing was 20-46% of the 10M query (P139). `scores` scratch is reused across all levels.
        let mut scores: Vec<i32> = vec![0; l0.max(64)];
        let sdim = ROUTE_SDIM.load(std::sync::atomic::Ordering::Relaxed);
        let rp = ROUTE_PROF.load(std::sync::atomic::Ordering::Relaxed);
        let vnni = ROUTE_VNNI.load(std::sync::atomic::Ordering::Relaxed) && !self.cadj.is_empty();
        let qnorm = if vnni { simd::sqnorm_i8(qn) } else { 0 };
        let tc = if rp { Some(std::time::Instant::now()) } else { None };
        if vnni {
            simd::l2_i8_block_norm(qn, &self.cent[0], &self.cadj[0], l0, d, qnorm, &mut scores);
        } else {
            simd::l2_i8_block(qn, &self.cent[0], l0, d, d, &mut scores);
        }
        let mut cd: Vec<(i32, u32)> = (0..l0).map(|q| (scores[q], q as u32)).collect();
        if let Some(t) = tc { PROF_R_COARSE_NS.fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            PROF_R_NEVAL.fetch_add(l0 as u64, std::sync::atomic::Ordering::Relaxed); }
        let tcs = if rp { Some(std::time::Instant::now()) } else { None };
        let b = self.beam[0].min(cd.len());
        if b > 0 && b < cd.len() { cd.select_nth_unstable(b - 1); cd.truncate(b); }
        let mut sel: Vec<u32> = cd.iter().map(|&(_, c)| c).collect();
        if let Some(t) = tcs { PROF_R_CSEL_NS.fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); }
        let tf = if rp { Some(std::time::Instant::now()) } else { None };
        fd.clear();
        // ADC routing (#3): at the finest level, score children by 4-bit ADC (LUT+codes), keep the ADC-top
        // ROUTE_ADC_KEEP, then EXACT-rescore only those -> cheap finest scoring if recall holds.
        let adc = ROUTE_ADC.load(std::sync::atomic::Ordering::Relaxed) && self.radc.is_some();
        let adc_m = self.radc.as_ref().map(|p| p.m).unwrap_or(0);
        let adc_lut: Vec<i8> = if adc { self.radc.as_ref().unwrap().query_lut(qn) } else { Vec::new() };
        // vpshufb block path when codes are blocked + avx2; else scalar LUT-sum.
        let adc_regs = if adc && !self.rblocks.is_empty() && std::is_x86_feature_detected!("avx2") {
            pq::lut_regs_i8(&adc_lut, adc_m)
        } else { Vec::new() };
        for l in 1..self.levels {
            let finest = l == self.levels - 1;
            // EXACT capacity: sum the selected cells' child fan-out (prefix-sum lookups) so `nd` never
            // reallocs mid-gather (the heuristic sel.len()*8 under-provisioned at Kf/C0≈21 -> memmove growth).
            let cap: usize = sel.iter().map(|&p| (self.child[l - 1][p as usize + 1] - self.child[l - 1][p as usize]) as usize).sum();
            let mut nd: Vec<(i32, u32)> = Vec::with_capacity(cap + 16);
            for &p in &sel {
                let (s, e) = (self.child[l - 1][p as usize] as usize, self.child[l - 1][p as usize + 1] as usize);
                let nc = e - s;
                if nc == 0 { continue; }
                if rp { PROF_R_NEVAL.fetch_add(nc as u64, std::sync::atomic::Ordering::Relaxed); }
                if finest && adc {
                    if !adc_regs.is_empty() && s % 16 == 0 && (e - s) % 16 == 0 {
                        let gb = (adc_m / 2) * 16; // bytes per 16-cell block
                        let mut out16 = [0i32; 16];
                        for jb in 0..(e - s) / 16 {
                            let blk = s / 16 + jb;
                            unsafe { pq::block_adc_i8_i16acc(&self.rblocks[blk * gb..blk * gb + gb], adc_m, &adc_regs, &mut out16); }
                            for i in 0..16 { nd.push((out16[i], (s + jb * 16 + i) as u32)); }
                        }
                    } else {
                        for c in s..e {
                            let code = &self.rcodes[c * adc_m..c * adc_m + adc_m];
                            let mut sc = 0i32;
                            for sub in 0..adc_m { sc += adc_lut[sub * 16 + code[sub] as usize] as i32; }
                            nd.push((sc, c as u32));
                        }
                    }
                } else {
                    if scores.len() < nc { scores.resize(nc, 0); }
                    // finest level (the dominant routing term) may score a reduced dim prefix (SBANN_ROUTE_SDIM).
                    let sd = if finest && sdim > 0 && sdim < d { sdim } else { d };
                    if vnni && sd == d {
                        simd::l2_i8_block_norm(qn, &self.cent[l][s * d..e * d], &self.cadj[l][s..e], nc, d, qnorm, &mut scores);
                    } else {
                        simd::l2_i8_block(qn, &self.cent[l][s * d..e * d], nc, d, sd, &mut scores);
                    }
                    for (i, c) in (s..e).enumerate() { nd.push((scores[i], c as u32)); }
                }
            }
            if finest {
                if adc {
                    let keepv = ROUTE_ADC_KEEP.load(std::sync::atomic::Ordering::Relaxed);
                    if keepv > 0 {
                        // keep ADC-top-KEEP then EXACT-rescore them (precise cell distances).
                        let keep = keepv.min(nd.len());
                        if keep < nd.len() { nd.select_nth_unstable(keep - 1); nd.truncate(keep); }
                        for ent in nd.iter_mut() {
                            let c = ent.1 as usize;
                            ent.0 = simd::l2_i8(qn, &self.cent[l][c * d..c * d + d]);
                        }
                    }
                    // KEEP==0: NO exact rerank -- route_fine picks top-p straight from the ADC scores. The
                    // router only needs the RIGHT cells (scan+final rerank rank candidates), so ADC may suffice.
                }
                if let Some(t) = tf { PROF_R_FINE_NS.fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); }
                *fd = nd;
                return;
            }
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
        let ts = if ROUTE_PROF.load(std::sync::atomic::Ordering::Relaxed) { Some(std::time::Instant::now()) } else { None };
        if k > 0 { fd.select_nth_unstable(k - 1); out.extend(fd[..k].iter().map(|&(_, f)| f)); }
        if let Some(t) = ts { PROF_R_FSEL_NS.fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); }
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
    fn save(&self, w: &mut crate::persist::Sw) -> std::io::Result<()> {
        w.u8(ROUTER_TAG_HIER)?;
        w.usize(self.d)?;
        w.usize(self.kf)?;
        w.usize(self.levels)?;
        w.f32(self.soar)?;
        w.f32s(&self.mu)?;
        w.usize(self.cent.len())?;
        for c in &self.cent { w.i8s(c)?; }   // cent: Vec<Vec<i8>> (per-level centroids)
        w.usize(self.child.len())?;
        for c in &self.child { w.u32s(c)?; } // child: Vec<Vec<u32>> (per-level prefix sums)
        w.usizes(&self.beam)?;
        save_opt_pq(&self.radc, w)?;
        w.u8s(&self.rcodes)?;
        w.u8s(&self.rblocks)?;
        Ok(())
    }
}

// ---------------- Compressor: candidate scan (approx distances) ----------------
pub enum QueryCtx {
    Pq { regs: Vec<__m128i> },                  // PQ/OPQ/AQ LUT registers (i8, saturating)
    // i16 LUT: lo/hi byte-tables (AVX2 single-block) + zmm tables (AVX-512 32-wide pair). Full-res ranking.
    Pq16 { lo: Vec<__m128i>, hi: Vec<__m128i>, lut_z: Vec<__m512i>, scale: f32 }, // scale = i16-units/IP for RESIDQ offset
    // fast-scan: int8 LUT, 1 vpshufb/subspace, i16 accum. regs_z = same LUT broadcast to zmm lanes for
    // the 64-wide AVX-512 path (empty unless USE512FS). regs_y = 256-bit LUT (both lanes) for the PROPER
    // 32-wide int8-saturating FastScan (empty unless FASTSCAN2).
    Pq8 { regs: Vec<__m128i>, regs_z: Vec<__m512i>, regs_y: Vec<__m256i>, scale: f32 },
    Scalar,                                      // exact int8: scan uses the raw query
    // RaBitQ: the rotated query qrot = P*q (global frame, c=0). `ip` selects IP vs L2 score assembly.
    RaBitQ { qrot: Vec<f32>, ip: bool },
    // NormPq (P200 norm-rescaled ADC): i16 IP LUT + per-query offset cq. Per-vector score =
    // (adc_i16 - cq) * gamma  (gamma = ||x||/||x_hat|| stored per vector) -> de-biases PQ norm shrink.
    NormPq16 { lo: Vec<__m128i>, hi: Vec<__m128i>, cq: f32 },
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
    /// Serialize self (1-byte concrete-type tag + POD fields) for SBANN_INDEX_SAVE. Default: error —
    /// only the scale-path compressors (Apq4, Pq4) implement it; load_comp reads the tag back.
    fn save(&self, _w: &mut crate::persist::Sw) -> std::io::Result<()> {
        Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "compressor type not serializable (only Apq4/Pq4 are)"))
    }
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
    fn save(&self, w: &mut crate::persist::Sw) -> std::io::Result<()> {
        w.u8(COMP_TAG_PQ4)?;
        save_pq(&self.pq, w)
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
        if FASTSCAN2.load(std::sync::atomic::Ordering::Relaxed) {
            // PROPER 32-wide int8-saturating FastScan: bounded LUT (hoist-safe), 256-bit LUT regs.
            let l = if ip { self.pq.query_lut_f32_i8s_ip_fs2(&qf) } else { self.pq.query_lut_f32_i8s_fs2(&qf) };
            return QueryCtx::Pq8 { regs: pq::lut_regs_i8(&l, self.pq.m), regs_z: Vec::new(),
                regs_y: pq::lut_regs_i8_y256(&l, self.pq.m), scale: 0.0 };
        }
        if FASTSCAN.load(std::sync::atomic::Ordering::Relaxed) {
            let l = if ip { self.pq.query_lut_f32_i8s_ip(&qf) } else { self.pq.query_lut_f32_i8s(&qf) };
            let regs_z = if USE512FS.load(std::sync::atomic::Ordering::Relaxed) { pq::lut_regs_i8_z512(&l, self.pq.m) } else { Vec::new() };
            let scale = if residq && ip { self.pq.ip_i8s_scale(&qf) } else { 0.0 };
            return QueryCtx::Pq8 { regs: pq::lut_regs_i8(&l, self.pq.m), regs_z, regs_y: Vec::new(), scale };
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
        if let QueryCtx::Pq8 { regs_y, .. } = ctx {
            if !regs_y.is_empty() { // PROPER 32-wide int8-sat FastScan over the two adjacent 16-blocks
                unsafe { pq::block_adc_i8_fastscan32_2x16(b0, b1, self.pq.m, regs_y, out) };
                return;
            }
        }
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
    fn save(&self, w: &mut crate::persist::Sw) -> std::io::Result<()> {
        w.u8(COMP_TAG_APQ4)?;
        w.usize(self.d)?;
        w.usize(self.dpb)?;
        w.f32(self.eta)?;
        save_pq(&self.pq, w)
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
        if let QueryCtx::Pq8 { regs_y, .. } = ctx {
            if !regs_y.is_empty() { // PROPER 32-wide int8-sat FastScan over the two adjacent 16-blocks
                unsafe { pq::block_adc_i8_fastscan32_2x16(b0, b1, self.pq.m, regs_y, out) };
                return;
            }
        }
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

/// RaBitQ (Gao & Long, SIGMOD'24): a rotation + per-coordinate B-bit quantization with an UNBIASED
/// inner-product estimator. For each data vector x we form the unit residual `o = (x-c)/||x-c||`
/// (c = cell centroid when SBANN_RESIDQ supplies one, else 0 — the global frame), rotate it by a fixed
/// orthonormal P into `o' = P o`, and quantize each coordinate of o' to B bits → integer codes `t_i`,
/// reconstructing `u_i = level[t_i]`. We store the codes plus two f32 scalars per vector:
///   cadd = ||x-c||^2          (the L2 self-term)
///   cmul = ||x-c|| / f,  f = <u, o'>   (the unbiased-estimator rescale; f is the cosine of the code
///                                       with the true rotated unit vector, computed EXACTLY at encode)
/// At query time we rotate q once (`qrot = P q`) and, per vector, form `s = <u, qrot>`; then
///   <q, x-c> ≈ ||x-c|| * <u, qrot> / f = cmul * s        (RaBitQ's unbiased IP estimator)
///   L2:  ||q-x||^2 ≈ ||q||^2 + cadd - 2*cmul*s   → rank score = cadd - 2*cmul*s  (||q||^2 const, dropped)
///   IP:  <q,x>     ≈ cmul*s                       → rank score = -(cmul*s)        (smaller = closer)
/// Storing f exactly makes the estimator unbiased for ANY reconstruction `u`, so the multi-bit (B>1)
/// quantizer choice only affects variance, not bias. This is a CORRECTNESS/pluggability slot (scalar
/// scan, no vpshufb fast path); the exact int8 rerank that follows only needs a reasonable ranking.
/// NOTE: operates in the GLOBAL frame (c=0). Combining with SBANN_RESIDQ (per-cell c) is not wired —
/// the per-cell `||q-c||^2` / `<q,c>` offsets would need plumbing the centroid into the scan.
pub struct RaBitQ {
    d: usize,
    bits: usize,          // B bits/coordinate
    rot: Vec<f32>,        // d×d orthonormal P (row-major)
    code_bytes: usize,    // ceil(d*bits/8) packed B-bit codes per vector
    levels: Vec<f32>,     // dequant LUT: code t -> reconstructed coordinate value (already /sqrt(d))
    l_range: f32,         // multi-bit uniform quantizer half-range on g=sqrt(d)*o'  (unused for B=1)
}

impl RaBitQ {
    pub fn new(d: usize, bits: usize, l_range: f32, seed: u64) -> Self {
        let rot = random_orthogonal(d, seed);
        let code_bytes = (d * bits + 7) / 8;
        let inv_sqrt_d = 1.0 / (d as f32).sqrt();
        let nlev = 1usize << bits;
        // Dequant levels on g = sqrt(d)*o' (≈ unit-variance), then scaled back by 1/sqrt(d) into o'-space.
        // B=1 -> {-1,+1}; B>=2 -> 2^B uniform levels over [-L, L].
        let levels: Vec<f32> = if bits == 1 {
            vec![-inv_sqrt_d, inv_sqrt_d]
        } else {
            (0..nlev).map(|t| (-l_range + 2.0 * l_range * t as f32 / (nlev - 1) as f32) * inv_sqrt_d).collect()
        };
        RaBitQ { d, bits, rot, code_bytes, levels, l_range }
    }

    /// Quantize one rotated unit vector o' (length d) -> integer codes t_i, returning f = <u, o'>.
    #[inline]
    fn quantize(&self, oprime: &[f32], codes: &mut [u8]) -> f32 {
        let nlev = 1usize << self.bits;
        let sqrt_d = (self.d as f32).sqrt();
        for b in codes.iter_mut() { *b = 0; }
        let mut f = 0.0f32;
        let mut bitpos = 0usize;
        for i in 0..self.d {
            let t = if self.bits == 1 {
                if oprime[i] >= 0.0 { 1usize } else { 0usize }
            } else {
                let g = oprime[i] * sqrt_d; // ≈ unit variance
                let q = ((g + self.l_range) / (2.0 * self.l_range) * (nlev - 1) as f32).round();
                q.clamp(0.0, (nlev - 1) as f32) as usize
            };
            f += self.levels[t] * oprime[i];
            // pack `bits` bits of t at bitpos (LSB-first, coordinate 0 lowest)
            let mut tt = t;
            for _ in 0..self.bits {
                if tt & 1 != 0 { codes[bitpos >> 3] |= 1u8 << (bitpos & 7); }
                tt >>= 1;
                bitpos += 1;
            }
        }
        f
    }

    /// s = <u, qrot> for the vector whose codes start at `codes` (length code_bytes).
    #[inline]
    fn code_dot(&self, codes: &[u8], qrot: &[f32]) -> f32 {
        let mut s = 0.0f32;
        let mut bitpos = 0usize;
        for i in 0..self.d {
            // read `bits` bits at bitpos (LSB-first)
            let mut t = 0usize;
            for b in 0..self.bits {
                if codes[bitpos >> 3] & (1u8 << (bitpos & 7)) != 0 { t |= 1 << b; }
                bitpos += 1;
            }
            s += self.levels[t] * qrot[i];
        }
        s
    }
}

impl Compressor for RaBitQ {
    fn block_bytes(&self) -> usize { 16 * (self.code_bytes + 8) } // per lane: code + cadd(f32) + cmul(f32)
    fn encode_block(&self, rows: &[&[i8]], n_real: usize, cell_cent: &[i8], out: &mut Vec<u8>) {
        let stride = self.code_bytes + 8;
        let resid = !cell_cent.is_empty(); // SBANN_RESIDQ: encode (row - cell_cent); else global frame (c=0)
        let mut r = vec![0f32; self.d];
        let mut oprime = vec![0f32; self.d];
        let mut codes = vec![0u8; self.code_bytes];
        for j in 0..16 {
            let base = out.len();
            out.resize(base + stride, 0);
            if j >= n_real { continue; } // padding lane: zeros (cadd=cmul=0); slot_orig==MAX skips it
            // residual r = x - c (c=0 in the global frame)
            for k in 0..self.d {
                r[k] = rows[j][k] as f32 - if resid { cell_cent[k] as f32 } else { 0.0 };
            }
            let rnorm2: f32 = r.iter().map(|&v| v * v).sum();
            let rnorm = rnorm2.sqrt();
            if rnorm < 1e-9 {
                // zero residual: leave code zero, cadd=cmul=0 (rerank fixes exact distance)
                continue;
            }
            // o' = P (r / ||r||)
            let inv = 1.0 / rnorm;
            for a in 0..self.d {
                let mut acc = 0.0f32;
                let row = &self.rot[a * self.d..a * self.d + self.d];
                for k in 0..self.d { acc += row[k] * r[k]; }
                oprime[a] = acc * inv;
            }
            for b in codes.iter_mut() { *b = 0; }
            let f = self.quantize(&oprime, &mut codes);
            let f = if f.abs() < 1e-9 { 1e-9 } else { f };
            let cadd = rnorm2;
            let cmul = rnorm / f;
            out[base..base + self.code_bytes].copy_from_slice(&codes);
            out[base + self.code_bytes..base + self.code_bytes + 4].copy_from_slice(&cadd.to_le_bytes());
            out[base + self.code_bytes + 4..base + self.code_bytes + 8].copy_from_slice(&cmul.to_le_bytes());
        }
    }
    fn prepare_query(&self, q: &[i8]) -> QueryCtx {
        // qrot = P q (global frame). For L2 and for IP (c=0) the rotated query is identical.
        let mut qrot = vec![0f32; self.d];
        for a in 0..self.d {
            let mut acc = 0.0f32;
            let row = &self.rot[a * self.d..a * self.d + self.d];
            for k in 0..self.d { acc += row[k] * q[k] as f32; }
            qrot[a] = acc;
        }
        let ip = IP_MODE.load(std::sync::atomic::Ordering::Relaxed);
        QueryCtx::RaBitQ { qrot, ip }
    }
    fn scan_block(&self, block: &[u8], ctx: &QueryCtx, _q: &[i8], _rows16: &[&[i8]], out16: &mut [i32; 16]) {
        if let QueryCtx::RaBitQ { qrot, ip } = ctx {
            let stride = self.code_bytes + 8;
            for j in 0..16 {
                let base = j * stride;
                let codes = &block[base..base + self.code_bytes];
                let cadd = f32::from_le_bytes(block[base + self.code_bytes..base + self.code_bytes + 4].try_into().unwrap());
                let cmul = f32::from_le_bytes(block[base + self.code_bytes + 4..base + self.code_bytes + 8].try_into().unwrap());
                let s = self.code_dot(codes, qrot);
                let est_ip = cmul * s; // ≈ <q, x-c>
                let score = if *ip { -est_ip } else { cadd - 2.0 * est_ip };
                out16[j] = score.round().clamp(i32::MIN as f32, i32::MAX as f32) as i32;
            }
        }
    }
}

/// NormPq (P200, angle-1 "norm-rescaled ADC"): anisotropic 4-bit PQ (identical codes/training to Apq4)
/// PLUS a per-vector f32 `gamma = ||x|| / ||x_hat||` (x_hat = PQ reconstruction of the raw int8 row).
/// The IP estimate <q,x_hat> is de-biased for PQ's norm-shrinkage (which systematically UNDER-ranks the
/// large-norm MIPS winners) by rescaling: corrected_ip = <q,x_hat> * gamma. This TIGHTENS the pool
/// ranking (offline: ~halves the survivor count vs apq4 at matched recall) at +4 bytes/vec. Scan uses the
/// full-res i16 ADC (the saturating fast-scan path would clip the pre-multiply magnitude gamma needs).
/// Block layout: apq4 codes (m/2*16 B) followed by 16 little-endian f32 gammas (64 B).
pub struct NormPq { pq: pq::Pq, d: usize, dpb: usize, eta: f32 }

impl NormPq {
    pub fn train(ds: &I8Bin, dpb: usize, iters: usize, eta: f32) -> Self {
        let (n, d) = (ds.nb, ds.d);
        let smp = n.min(40000);
        let stride = (n / smp).max(1);
        let mut x = vec![0f32; smp * d];
        x.par_chunks_mut(d).enumerate().for_each(|(i, o)| { let r = ds.row(i * stride); for k in 0..d { o[k] = r[k] as f32; } });
        NormPq { pq: pq::Pq::train_f32_aniso(&x, d, dpb, smp, iters, eta), d, dpb, eta }
    }
    #[inline]
    fn gamma_of(&self, xf: &[f32], code: &[u8]) -> f32 {
        let mut xhat = [0f32; 256];
        self.pq.decode(code, &mut xhat[..self.d]);
        let nx: f32 = xf.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let nh: f32 = xhat[..self.d].iter().map(|&v| v * v).sum::<f32>().sqrt();
        if nh < 1e-6 { 1.0 } else { nx / nh }
    }
}

impl Compressor for NormPq {
    fn block_bytes(&self) -> usize { self.pq.m / 2 * 16 + 16 * 4 }
    fn encode_block(&self, rows: &[&[i8]], n_real: usize, _cell_cent: &[i8], out: &mut Vec<u8>) {
        let mut codes16 = [[0u8; 256]; 16];
        let mut gammas = [0f32; 16];
        let mut xf = vec![0f32; self.d];
        for j in 0..16 {
            if j < n_real {
                for k in 0..self.d { xf[k] = rows[j][k] as f32; }
                self.pq.encode_f32(&xf, &mut codes16[j][..self.pq.m]);
                gammas[j] = self.gamma_of(&xf, &codes16[j][..self.pq.m]);
            } else { for k in 0..self.pq.m { codes16[j][k] = 0; } gammas[j] = 0.0; }
        }
        pq::pack_block(&codes16, self.pq.m, out);
        for &g in gammas.iter() { out.extend_from_slice(&g.to_le_bytes()); }
    }
    fn prepare_query(&self, q: &[i8]) -> QueryCtx {
        let qf: Vec<f32> = q.iter().map(|&v| v as f32).collect();
        let lut = self.pq.query_lut_f32_i16_ip(&qf);
        let (lo, hi) = pq::lut_regs_i16(&lut, self.pq.m);
        let cq = self.pq.ip_i16_offset(&qf);
        QueryCtx::NormPq16 { lo, hi, cq }
    }
    fn scan_block(&self, block: &[u8], ctx: &QueryCtx, _q: &[i8], _r: &[&[i8]], out16: &mut [i32; 16]) {
        if let QueryCtx::NormPq16 { lo, hi, cq } = ctx {
            let mut adc = [0i32; 16];
            unsafe { pq::block_adc_i16_avx2(block, self.pq.m, lo, hi, &mut adc) };
            let gbase = self.pq.m / 2 * 16; // gammas follow the packed codes
            for i in 0..16 {
                let g = f32::from_le_bytes(block[gbase + i * 4..gbase + i * 4 + 4].try_into().unwrap());
                // rank score = (adc - cq) * gamma  (smaller = closer; adc-cq ~ -scale*ip <= 0)
                out16[i] = ((adc[i] as f32 - cq) * g).round() as i32;
            }
        }
    }
    fn save(&self, w: &mut crate::persist::Sw) -> std::io::Result<()> {
        w.u8(COMP_TAG_NORMPQ)?;
        w.usize(self.d)?;
        w.usize(self.dpb)?;
        w.f32(self.eta)?;
        save_pq(&self.pq, w)
    }
}

/// Selftest for NormPq.scan_block: verifies the gamma byte-read + (adc_i16 - cq)*gamma arithmetic
/// against an independent scalar reference (block_adc_i16 scalar + hand-applied gamma). The AVX2 i16
/// kernel itself is covered by pq::selftest_i16; this guards the norm-rescale wrapper.
pub fn selftest_normpq() -> bool {
    let (d, dpb) = (8usize, 2usize);
    let m = d / dpb;
    // random-ish centroids + query
    let mut cent = vec![0f32; m * 16 * dpb];
    let mut s = 0x1234u64;
    for v in cent.iter_mut() { s = s.wrapping_mul(6364136223846793005).wrapping_add(1); *v = ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0; }
    let pq = pq::Pq { d, dpb, m, cent, eta: 1.0 };
    let comp = NormPq { pq, d, dpb, eta: 1.0 };
    // 16 synthetic int8 rows
    let rows_store: Vec<Vec<i8>> = (0..16).map(|j| (0..d).map(|k| ((j * 7 + k * 3) as i32 % 17 - 8) as i8).collect()).collect();
    let rows: Vec<&[i8]> = rows_store.iter().map(|r| r.as_slice()).collect();
    let mut block = Vec::new();
    comp.encode_block(&rows, 16, &[], &mut block);
    let q: Vec<i8> = (0..d).map(|k| ((k * 5) as i32 % 13 - 6) as i8).collect();
    let ctx = comp.prepare_query(&q);
    let mut got = [0i32; 16];
    comp.scan_block(&block, &ctx, &q, &[], &mut got);
    // scalar reference
    let qf: Vec<f32> = q.iter().map(|&v| v as f32).collect();
    let lut = comp.pq.query_lut_f32_i16_ip(&qf);
    let cq = comp.pq.ip_i16_offset(&qf);
    let gbase = m / 2 * 16;
    let mut ok = true;
    for i in 0..16 {
        let mut adc = 0i32;
        for g in 0..m { // scalar ADC mirroring block_adc_i16 (subspace-major nibble layout)
            let byte = block[(g / 2) * 16 + i];
            let code = if g % 2 == 0 { byte & 0x0f } else { byte >> 4 } as usize;
            adc += lut[g * 16 + code] as i32;
        }
        let gam = f32::from_le_bytes(block[gbase + i * 4..gbase + i * 4 + 4].try_into().unwrap());
        let want = ((adc as f32 - cq) * gam).round() as i32;
        if got[i] != want { eprintln!("selftest_normpq lane {i}: got {} want {}", got[i], want); ok = false; }
    }
    ok
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
    // build-time multi-assignment factor (a0). >1 => SOAR multi-store: a point can appear in several
    // probed cells as duplicate slots, so scan_rerank must dedup the pool by orig id BEFORE the t_surv cap
    // (deduping after the cap loses recall when dups crowd the survivors). a0==1 => no dedup needed.
    pub a0: usize,
    // RAW LAYOUT (SBANN_RAW_DEDUP, Task B): false => `raw` is SLOT-indexed (n*a0*d, a full 2nd copy);
    // true => `raw` is per-DISTINCT-ORIG (n*d, indexed by orig id) so rerank reads raw[orig*d] instead of
    // raw[slot*d]. rerank_contig already dedups to distinct orig ids, so the orig-indexed read is exact
    // and the index shrinks by ~a0x on the biggest array. Set at build from RAW_DEDUP; serialized.
    pub raw_orig_indexed: bool,
    // ---- STREAMING (all empty/None until insert()/delete() is used) ----
    // Per-cell APPEND BUFFER for inserted points. ins_blocks[c] = that cell's appended points packed
    // into the SAME 16-row PQ block layout as the main index, so the existing scan_block kernel runs
    // over them unchanged. ins_gidx[c][slot] = the GLOBAL appended index for that slot. ins_full_blocks[c]
    // = #complete (16-full) blocks already sealed into ins_blocks[c]; the partial tail block is re-encoded
    // on each finalize_inserts(). Every appended point also lives in the flat ins_raw/ins_orig arrays (in
    // append order) so the exact rerank reuses rerank_contig_pairs(ins_raw, ins_orig, (approx, gidx)).
    pub ins_blocks: Vec<Vec<u8>>,
    pub ins_gidx: Vec<Vec<u32>>,
    pub ins_full_blocks: Vec<usize>,
    pub ins_raw: Vec<i8>,                              // d i8 per appended point (flat, append order)
    pub ins_orig: Vec<u32>,                           // orig id per appended point (u32::MAX once deleted)
    pub ins_loc: std::collections::HashMap<u32, u32>, // inserted orig id -> global appended index (delete)
    pub ins_dirty: Vec<u32>,                          // cells whose tail block needs (re)encoding
    pub ins_count: usize,                             // live (non-deleted) appended points
    // reverse map MAIN orig -> its slots, built lazily on the first delete (CSR; main orig ids are the
    // dense point indices 0..n_main-1 produced by build). None until a main point is deleted.
    pub main_rev: Option<(Vec<u32>, Vec<u32>)>,       // (offsets[n_main+1], slots)
    pub n_main: usize,                                // #points at build time
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
        // RAW LAYOUT (Task B). Default: slot-indexed `raw` grown in the slot loop (n*a0*d, a full a0x copy).
        // SBANN_RAW_DEDUP: per-distinct-orig `raw` (n*d), prefilled here by orig id; the slot loop skips it.
        let raw_dedup = RAW_DEDUP.load(std::sync::atomic::Ordering::Relaxed);
        let mut raw: Vec<i8> = if raw_dedup {
            let mut r = vec![0i8; n * d];
            r.par_chunks_mut(d).enumerate().for_each(|(o, out)| out.copy_from_slice(ds.row(o)));
            r
        } else { Vec::new() }; // raw i8 in slot order, parallel to slot_orig
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
                    if !raw_dedup { if j < cnt { raw.extend_from_slice(rows[j]); } else { raw.resize(raw.len() + d, 0); } }
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
        Index {
            router, comp, cell_bstart, slot_orig, blocks, bb, xfn: Vec::new(), raw, d, blocks_il,
            cell_ilstart, resid_pq, resid_codes, rq_cent, a0, raw_orig_indexed: raw_dedup,
            ins_blocks: vec![Vec::new(); nc], ins_gidx: vec![Vec::new(); nc], ins_full_blocks: vec![0; nc],
            ins_raw: Vec::new(), ins_orig: Vec::new(), ins_loc: std::collections::HashMap::new(),
            ins_dirty: Vec::new(), ins_count: 0, main_rev: None, n_main: n,
        }
    }

    pub fn search(&self, ds: &I8Bin, q: &[i8], p: usize, t: usize, k: usize) -> Vec<u32> {
        let prof = PROFILE.load(std::sync::atomic::Ordering::Relaxed);
        let t0 = if prof { Some(std::time::Instant::now()) } else { None };
        let cells = self.router.probe(q, p);
        if let Some(t0) = t0 { PROF_ROUTE_NS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); }
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
        let _ = rerank_contig(&self.raw, self.d, &self.slot_orig, q, &pool, k, self.raw_orig_indexed);
        let t3 = Instant::now();
        ((t1 - t0).as_nanos() as u64, (t2 - t1).as_nanos() as u64, (t3 - t2).as_nanos() as u64)
    }

    /// Build the candidate pool (approx_dist, slot) for `cells` using the active compressor's scan
    /// kernels (exact-row / 64-wide AVX-512 / paired PQ), applying the RESIDQ per-cell offset. Factored
    /// out of scan_rerank so the streaming search reuses the IDENTICAL hot scan path. Returns the raw
    /// pool BEFORE any dedup / survivor cap. `ctx` is the prepared query (prepare_query) shared by caller.
    fn scan_pool(&self, ds: &I8Bin, q: &[i8], cells: &[u32], ctx: &QueryCtx) -> Vec<(i32, u32)> {
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
        let rq_scale: f32 = match ctx { QueryCtx::Pq16 { scale, .. } | QueryCtx::Pq8 { scale, .. } => *scale, _ => 0.0 };
        let residq = rq_scale != 0.0 && !self.rq_cent.is_empty();
        let dd = self.d;
        let prefetch = PREFETCH.load(std::sync::atomic::Ordering::Relaxed) && bb > 0 && !self.blocks.is_empty();
        let pfdist = PFDIST.load(std::sync::atomic::Ordering::Relaxed).max(1);
        let pflines = PFLINES.load(std::sync::atomic::Ordering::Relaxed);
        let blk_ptr = self.blocks.as_ptr();
        let blk_len = self.blocks.len();
        for ci in 0..cells.len() {
            let cell = cells[ci];
            // SW-prefetch the block memory of a cell `pfdist` ahead so its random-jump latency overlaps
            // the current cell's scan (the scan is memory-latency bound on p cross-cell jumps).
            if prefetch && ci + pfdist < cells.len() {
                let nc = cells[ci + pfdist] as usize;
                let nbs = self.cell_bstart[nc] as usize;
                prefetch_lines(blk_ptr, blk_len, nbs * bb, pflines);
            }
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
                    self.comp.scan_block(block, ctx, q, &rows16, &mut out16);
                    for j in 0..16 {
                        let slot = b * 16 + j;
                        if self.slot_orig[slot] != u32::MAX { pool.push((out16[j], slot as u32)); }
                    }
                }
            } else if use512fs {
                // 64-wide AVX-512 fast-scan over the interleaved superblocks (4 blocks/superblock).
                if let QueryCtx::Pq8 { regs, regs_z, .. } = ctx {
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
                    self.comp.scan_block_x2(b0, b1, ctx, &mut out32);
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
                    self.comp.scan_block(block, ctx, q, &[], &mut out16);
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
        pool
    }

    /// FUSED top-t collect (SBANN_FUSEDTOPK): identical kernel dispatch to `scan_pool` but replaces the
    /// per-candidate scalar `pool.push + slot_orig branch` (+ terminal select_nth over ALL candidates)
    /// with a running t-th-best threshold — each block's dists are SIMD-compared against it and only
    /// survivors are pushed. Returns the ALREADY-capped top-t pool (identical set to
    /// `scan_pool` -> select_nth(t)). Caller must ensure NO per-cell residq offset and NO pre-cap dedup
    /// (both incompatible with threshold-during-scan); it gates on that and falls back to scan_pool.
    fn scan_pool_fused(&self, ds: &I8Bin, q: &[i8], cells: &[u32], ctx: &QueryCtx, t: usize) -> Vec<(i32, u32)> {
        let need_rows = self.comp.needs_raw_rows();
        let mut out16 = [0i32; 16];
        let mut out32 = [0i32; 32];
        let bb = self.bb;
        let use512fs = USE512FS.load(std::sync::atomic::Ordering::Relaxed)
            && !self.blocks_il.is_empty()
            && matches!(ctx, QueryCtx::Pq8 { .. });
        let slot_orig = &self.slot_orig[..];
        let mut top = FusedTopT::new(t);
        for &cell in cells {
            let (bs, be) = (self.cell_bstart[cell as usize] as usize, self.cell_bstart[cell as usize + 1] as usize);
            if need_rows {
                for b in bs..be {
                    let block = if bb > 0 { &self.blocks[b * bb..(b + 1) * bb] } else { &[][..] };
                    let rows16: Vec<&[i8]> = (0..16).map(|j| {
                        let o = self.slot_orig[b * 16 + j];
                        if o != u32::MAX { ds.row(o as usize) } else { &[][..] }
                    }).collect();
                    self.comp.scan_block(block, ctx, q, &rows16, &mut out16);
                    top.emit(&out16, b * 16, slot_orig);
                }
            } else if use512fs {
                if let QueryCtx::Pq8 { regs, regs_z, .. } = ctx {
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
                            top.emit(&out64[sub * 16..sub * 16 + 16], (bbase + sub) * 16, slot_orig);
                        }
                    }
                    for b in (bs + 4 * nfull)..be {
                        let block = &self.blocks[b * bb..(b + 1) * bb];
                        unsafe { pq::block_adc_i8_i16acc(block, m, regs, &mut out16); }
                        top.emit(&out16, b * 16, slot_orig);
                    }
                }
            } else {
                let mut b = bs;
                while b + 1 < be {
                    let b0 = &self.blocks[b * bb..(b + 1) * bb];
                    let b1 = &self.blocks[(b + 1) * bb..(b + 2) * bb];
                    self.comp.scan_block_x2(b0, b1, ctx, &mut out32);
                    top.emit(&out32[0..16], b * 16, slot_orig);
                    top.emit(&out32[16..32], (b + 1) * 16, slot_orig);
                    b += 2;
                }
                if b < be {
                    let block = &self.blocks[b * bb..(b + 1) * bb];
                    self.comp.scan_block(block, ctx, q, &[], &mut out16);
                    top.emit(&out16, b * 16, slot_orig);
                }
            }
        }
        top.finish()
    }

    /// DIAGNOSTIC (SBANN_SCANDIAG): run the SAME kernel dispatch as scan_pool over `cells` but do NO
    /// collect (no slot_orig read, no push, no select_nth) — just consume `out` so the kernel isn't
    /// optimized away. Returns a checksum. Timed separately from the full scan; scan - kernel = collect.
    /// Candidate count (occupied slots) over a cell list — for the scanbench Mcand/s denominator.
    pub fn cell_cand_count(&self, cells: &[u32]) -> usize {
        cells.iter().map(|&c| {
            let (bs, be) = (self.cell_bstart[c as usize] as usize, self.cell_bstart[c as usize + 1] as usize);
            (be - bs) * 16
        }).sum()
    }
    /// Public wrapper so the scanbench harness can time the raw kernel floor (block reads + LUT, no
    /// collect) over an arbitrary cell ORDER (scattered probe order vs cell-id-sorted memory order).
    pub fn scan_kernel_bench(&self, ds: &I8Bin, q: &[i8], cells: &[u32], ctx: &QueryCtx) -> i64 {
        self.scan_kernel_only(ds, q, cells, ctx)
    }
    /// Expose the compressor's query LUT prep for the scanbench harness.
    pub fn prepare_query_pub(&self, q: &[i8]) -> QueryCtx { self.comp.prepare_query(q) }
    fn scan_kernel_only(&self, ds: &I8Bin, q: &[i8], cells: &[u32], ctx: &QueryCtx) -> i64 {
        let need_rows = self.comp.needs_raw_rows();
        let mut out16 = [0i32; 16];
        let mut out32 = [0i32; 32];
        let bb = self.bb;
        let use512fs = USE512FS.load(std::sync::atomic::Ordering::Relaxed)
            && !self.blocks_il.is_empty()
            && matches!(ctx, QueryCtx::Pq8 { .. });
        let mut acc: i64 = 0;
        let prefetch = PREFETCH.load(std::sync::atomic::Ordering::Relaxed) && bb > 0 && !self.blocks.is_empty();
        let pfdist = PFDIST.load(std::sync::atomic::Ordering::Relaxed).max(1);
        let pflines = PFLINES.load(std::sync::atomic::Ordering::Relaxed);
        let blk_ptr = self.blocks.as_ptr();
        let blk_len = self.blocks.len();
        for ci in 0..cells.len() {
            let cell = cells[ci];
            if prefetch && ci + pfdist < cells.len() {
                let nc = cells[ci + pfdist] as usize;
                let nbs = self.cell_bstart[nc] as usize;
                prefetch_lines(blk_ptr, blk_len, nbs * bb, pflines);
            }
            let (bs, be) = (self.cell_bstart[cell as usize] as usize, self.cell_bstart[cell as usize + 1] as usize);
            if need_rows {
                for b in bs..be {
                    let block = if bb > 0 { &self.blocks[b * bb..(b + 1) * bb] } else { &[][..] };
                    let rows16: Vec<&[i8]> = (0..16).map(|j| {
                        let o = self.slot_orig[b * 16 + j];
                        if o != u32::MAX { ds.row(o as usize) } else { &[][..] }
                    }).collect();
                    self.comp.scan_block(block, ctx, q, &rows16, &mut out16);
                    acc = acc.wrapping_add(out16[0] as i64);
                }
            } else if use512fs {
                if let QueryCtx::Pq8 { regs, regs_z, .. } = ctx {
                    let m = bb / 8;
                    let il0 = self.cell_ilstart[cell as usize] as usize;
                    let nfull = (be - bs) / 4;
                    let mut out64 = [0i32; 64];
                    for s in 0..nfull {
                        let sb = il0 + s;
                        unsafe { pq::block_adc_i8_i16acc_avx512_il(&self.blocks_il[sb * bb * 4..(sb + 1) * bb * 4], m, regs_z, &mut out64); }
                        acc = acc.wrapping_add(out64[0] as i64);
                    }
                    for b in (bs + 4 * nfull)..be {
                        let block = &self.blocks[b * bb..(b + 1) * bb];
                        unsafe { pq::block_adc_i8_i16acc(block, m, regs, &mut out16); }
                        acc = acc.wrapping_add(out16[0] as i64);
                    }
                }
            } else {
                let mut b = bs;
                while b + 1 < be {
                    let b0 = &self.blocks[b * bb..(b + 1) * bb];
                    let b1 = &self.blocks[(b + 1) * bb..(b + 2) * bb];
                    self.comp.scan_block_x2(b0, b1, ctx, &mut out32);
                    acc = acc.wrapping_add(out32[0] as i64).wrapping_add(out32[16] as i64);
                    b += 2;
                }
                if b < be {
                    let block = &self.blocks[b * bb..(b + 1) * bb];
                    self.comp.scan_block(block, ctx, q, &[], &mut out16);
                    acc = acc.wrapping_add(out16[0] as i64);
                }
            }
        }
        acc
    }

    /// Scan the given cells with the compressor, keep top-T by approx dist, exact-rerank to top-k.
    pub fn scan_rerank(&self, ds: &I8Bin, q: &[i8], cells: &[u32], t: usize, k: usize) -> Vec<u32> {
        let prof = PROFILE.load(std::sync::atomic::Ordering::Relaxed);
        let ctx = self.comp.prepare_query(q);
        // SBANN_SORTCELLS: visit the probed cells in ASCENDING cell-id (= block/memory) order so the
        // scattered PQ-block reads become monotonically forward. Recall-exactly-neutral (the top-t select
        // is order-independent). The ~p-element sort (few us) is charged to the wall-clock query below.
        let sorted_store;
        let cells: &[u32] = if SORTCELLS.load(std::sync::atomic::Ordering::Relaxed) {
            let mut v = cells.to_vec();
            v.sort_unstable();
            sorted_store = v;
            &sorted_store
        } else { cells };
        // DIAGNOSTIC (SBANN_SCANDIAG): time ONLY the kernel floor (block reads + LUT, NO collect) into
        // PROF_SCAN_NS and return early. Run this in a SEPARATE process vs the normal run: the per-query
        // cold-cache pattern is identical, so collect = scan_full - scan_kernelonly is measured unbiased.
        if SCANDIAG.load(std::sync::atomic::Ordering::Relaxed) {
            let ts = std::time::Instant::now();
            let acc = self.scan_kernel_only(ds, q, cells, &ctx);
            if prof { PROF_SCAN_NS.fetch_add(ts.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); }
            std::hint::black_box(acc);
            return Vec::new();
        }
        let ts = if prof { Some(std::time::Instant::now()) } else { None };
        // FUSED top-t collect (SBANN_FUSEDTOPK): only valid when no pre-cap dedup is needed (a0 dups must
        // be collapsed BEFORE the cap) and no per-cell residq offset is applied during scan (the threshold
        // compare runs on the same dist the cap ranks by). Otherwise fall back to the materialize-all path.
        let need_dedup = self.a0 >= DEDUP_A0.load(std::sync::atomic::Ordering::Relaxed) || POOLDEDUP.load(std::sync::atomic::Ordering::Relaxed);
        let residq_active = RESIDQ.load(std::sync::atomic::Ordering::Relaxed) && !self.rq_cent.is_empty();
        if FUSEDTOPK.load(std::sync::atomic::Ordering::Relaxed) && !need_dedup && !residq_active {
            let pool = self.scan_pool_fused(ds, q, cells, &ctx, t);
            if let Some(ts) = ts { PROF_SCAN_NS.fetch_add(ts.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); }
            let tr = if prof { Some(std::time::Instant::now()) } else { None };
            let out = rerank_contig(&self.raw, self.d, &self.slot_orig, q, &pool, k, self.raw_orig_indexed);
            if let Some(tr) = tr { PROF_RERANK_NS.fetch_add(tr.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); }
            return out;
        }
        let mut pool = self.scan_pool(ds, q, cells, &ctx);
        // SOAR multi-store (a0>1): dedup the pool by orig id BEFORE the cap so distinct survivors enter
        // rerank (deduping after the cap loses recall at high a0). a0==1 has no dups -> skip. The fast
        // reused open-addressing table replaces the per-query SipHash HashMap (the gap-widener, P134).
        if let Some(ts) = ts { PROF_SCAN_NS.fetch_add(ts.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); }
        let tr = if prof { Some(std::time::Instant::now()) } else { None };
        if need_dedup {
            dedup_pool_by_orig(&mut pool, &self.slot_orig);
        }
        let tt = t.min(pool.len());
        if tt > 0 { pool.select_nth_unstable(tt - 1); pool.truncate(tt); }
        let _ = ds;
        let out = rerank_contig(&self.raw, self.d, &self.slot_orig, q, &pool, k, self.raw_orig_indexed);
        if let Some(tr) = tr { PROF_RERANK_NS.fetch_add(tr.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); }
        out
    }

    /// FLOAT-RERANK search (P191 lever stack): route + int8 scan are BIT-IDENTICAL to `search`
    /// (same FASTSCAN2 kernel, PREFETCH, FUSEDTOPK, SOAR dedup, top-t cap) so the scan cost/QPS is the
    /// same lever stack as the int8 path; ONLY the final exact rerank of the t survivors is swapped to
    /// exact float IP over the original float vectors (`fbase`, `qf`). This breaks the int8 rerank's
    /// hard recall ceiling vs the float-computed OOD GT (reaches 0.90 at fewer probes -> higher QPS).
    pub fn search_frr(&self, ds: &I8Bin, q: &[i8], qf: &[f32], fbase: &crate::fbin::FBin, p: usize, t: usize, k: usize) -> Vec<u32> {
        let prof = PROFILE.load(std::sync::atomic::Ordering::Relaxed);
        let t0 = if prof { Some(std::time::Instant::now()) } else { None };
        let cells = self.router.probe(q, p);
        if let Some(t0) = t0 { PROF_ROUTE_NS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); }
        self.scan_rerank_frr(ds, q, qf, fbase, &cells, t, k)
    }

    /// Mirror of `scan_rerank` (same scan/dedup/cap) but reranks the survivors by exact float IP.
    pub fn scan_rerank_frr(&self, ds: &I8Bin, q: &[i8], qf: &[f32], fbase: &crate::fbin::FBin, cells: &[u32], t: usize, k: usize) -> Vec<u32> {
        let prof = PROFILE.load(std::sync::atomic::Ordering::Relaxed);
        let ctx = self.comp.prepare_query(q);
        let sorted_store;
        let cells: &[u32] = if SORTCELLS.load(std::sync::atomic::Ordering::Relaxed) {
            let mut v = cells.to_vec();
            v.sort_unstable();
            sorted_store = v;
            &sorted_store
        } else { cells };
        let ts = if prof { Some(std::time::Instant::now()) } else { None };
        let need_dedup = self.a0 >= DEDUP_A0.load(std::sync::atomic::Ordering::Relaxed) || POOLDEDUP.load(std::sync::atomic::Ordering::Relaxed);
        let residq_active = RESIDQ.load(std::sync::atomic::Ordering::Relaxed) && !self.rq_cent.is_empty();
        let cascade = CASCADE.load(std::sync::atomic::Ordering::Relaxed);
        let kk = CASCADE_K.load(std::sync::atomic::Ordering::Relaxed);
        if FUSEDTOPK.load(std::sync::atomic::Ordering::Relaxed) && !need_dedup && !residq_active {
            let mut pool = self.scan_pool_fused(ds, q, cells, &ctx, t);
            if let Some(ts) = ts { PROF_SCAN_NS.fetch_add(ts.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); }
            if cascade {
                // int8-cascade prune (PROF_CASC_NS) then float reorder (PROF_RERANK_NS) — timed inside.
                return rerank_cascade_float(fbase, &self.raw, self.d, self.raw_orig_indexed, &self.slot_orig, q, qf, &mut pool, kk, k);
            }
            let tr = if prof { Some(std::time::Instant::now()) } else { None };
            let out = rerank_contig_float(fbase, &self.slot_orig, qf, &pool, k);
            if let Some(tr) = tr { PROF_RERANK_NS.fetch_add(tr.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); }
            return out;
        }
        let mut pool = self.scan_pool(ds, q, cells, &ctx);
        if let Some(ts) = ts { PROF_SCAN_NS.fetch_add(ts.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); }
        if need_dedup {
            dedup_pool_by_orig(&mut pool, &self.slot_orig);
        }
        let tt = t.min(pool.len());
        if tt > 0 { pool.select_nth_unstable(tt - 1); pool.truncate(tt); }
        let _ = ds;
        if cascade {
            let tc = if prof { Some(std::time::Instant::now()) } else { None };
            let out = rerank_cascade_float(fbase, &self.raw, self.d, self.raw_orig_indexed, &self.slot_orig, q, qf, &mut pool, kk, k);
            if let Some(tc) = tc { PROF_CASC_NS.fetch_add(tc.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); }
            return out;
        }
        let tr = if prof { Some(std::time::Instant::now()) } else { None };
        let out = rerank_contig_float(fbase, &self.slot_orig, qf, &pool, k);
        if let Some(tr) = tr { PROF_RERANK_NS.fetch_add(tr.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); }
        out
    }

    // ======================= STREAMING (insert / delete / search) =======================
    // No-graph IVF makes streaming structurally cheap: a point lives in its routed cell(s) only, so an
    // insert is an APPEND to a per-cell buffer and a delete is a TOMBSTONE (slot_orig := u32::MAX, which
    // every scan/rerank already skips). There is no neighbor graph to repair. Costs:
    //   delete: O(a0) (tombstone the point's a0 slots) after a one-time O(n_main) reverse-map build.
    //   insert: O(d) amortized — append raw+orig (O(d)), route (router.assign), mark the cell dirty;
    //           the PQ re-encode of the touched tail block is deferred to finalize_inserts().
    //   finalize_inserts(): O(new points) — (re)encode each dirty cell's unsealed tail into PQ blocks.
    //   search: main scan + a buffer scan over the probed cells' appended blocks, merged at rerank.
    // The buffer scan re-walks a probed cell's appended points every query, so search cost grows with the
    // buffer; production amortizes this with a periodic compaction (= rebuild folding the buffer into the
    // main cell-contiguous layout). For this recall-focused validation we keep the buffer un-compacted.

    /// Tombstone `orig` so it disappears from all future results. Returns false if `orig` is unknown.
    /// Inserted points: clear their flat ins_orig entry (referenced from every cell that holds them).
    /// Main points: clear every slot the point occupies (a0 copies under SOAR multi-store).
    pub fn delete(&mut self, orig: u32) -> bool {
        if let Some(&g) = self.ins_loc.get(&orig) {
            if self.ins_orig[g as usize] != u32::MAX { self.ins_orig[g as usize] = u32::MAX; self.ins_count -= 1; }
            self.ins_loc.remove(&orig);
            return true;
        }
        if (orig as usize) >= self.n_main { return false; }
        if self.main_rev.is_none() { self.build_main_rev(); }
        let slots: Vec<u32> = {
            let (off, sl) = self.main_rev.as_ref().unwrap();
            let (s, e) = (off[orig as usize] as usize, off[orig as usize + 1] as usize);
            sl[s..e].to_vec()
        };
        let mut any = false;
        for slot in slots {
            if self.slot_orig[slot as usize] != u32::MAX { self.slot_orig[slot as usize] = u32::MAX; any = true; }
        }
        any
    }

    /// Build the lazy MAIN reverse index orig -> slots (CSR). Main orig ids are the dense point indices
    /// 0..n_main-1 produced by build(), so a flat offset table is exact and compact (~(1+a0)*n_main u32).
    fn build_main_rev(&mut self) {
        let n = self.n_main;
        let mut off = vec![0u32; n + 1];
        for &o in &self.slot_orig { if o != u32::MAX && (o as usize) < n { off[o as usize + 1] += 1; } }
        for i in 0..n { off[i + 1] += off[i]; }
        let mut slots = vec![0u32; off[n] as usize];
        let mut cur = off.clone();
        for (slot, &o) in self.slot_orig.iter().enumerate() {
            if o != u32::MAX && (o as usize) < n { slots[cur[o as usize] as usize] = slot as u32; cur[o as usize] += 1; }
        }
        self.main_rev = Some((off, slots));
    }

    /// Insert one point: store raw+orig in the flat append arrays, route it (router.assign, a0 cells),
    /// and reference its global appended index from each routed cell's buffer. The PQ encoding of the
    /// touched cells' tail blocks is deferred to finalize_inserts() (so a batch encodes once). Uses the
    /// index's own compressor (self.comp); `a0` controls multi-store like build's a0.
    pub fn insert(&mut self, row: &[i8], orig: u32, a0: usize) {
        debug_assert_eq!(row.len(), self.d);
        let g = self.ins_orig.len() as u32;
        self.ins_raw.extend_from_slice(row);
        self.ins_orig.push(orig);
        self.ins_loc.insert(orig, g);
        self.ins_count += 1;
        let mut cells: Vec<u32> = Vec::with_capacity(a0);
        self.router.assign(row, a0, &mut cells);
        if cells.is_empty() { cells.push(0); }
        cells.sort_unstable();
        cells.dedup();
        for &c in &cells {
            self.ins_gidx[c as usize].push(g);
            self.ins_dirty.push(c);
        }
    }

    /// Encode the appended points of every dirty cell into the per-cell PQ block buffer (the 16-row
    /// layout the scan kernel expects). Only the unsealed tail (past the last full block) is re-encoded,
    /// so repeated batches stay O(new points). MUST be called after an insert batch and before searching
    /// (search takes &self and cannot mutate). No-op for ScalarI8 (bb==0: the buffer is scanned exactly).
    pub fn finalize_inserts(&mut self) {
        if self.ins_dirty.is_empty() { return; }
        let bb = self.bb;
        let d = self.d;
        let dirty = std::mem::take(&mut self.ins_dirty);
        let mut seen = std::collections::HashSet::new();
        for cell in dirty {
            if !seen.insert(cell) { continue; }
            let c = cell as usize;
            if bb == 0 { continue; }
            let n = self.ins_gidx[c].len();
            let full = self.ins_full_blocks[c];
            self.ins_blocks[c].truncate(full * bb); // drop the previously-padded tail block(s)
            let mut b = full;
            while b * 16 < n {
                let s = b * 16;
                let cnt = (n - s).min(16);
                let rows: Vec<&[i8]> = (0..16).map(|j| {
                    if j < cnt { let g = self.ins_gidx[c][s + j] as usize; &self.ins_raw[g * d..g * d + d] }
                    else { &[][..] }
                }).collect();
                self.comp.encode_block(&rows, cnt, &[], &mut self.ins_blocks[c]);
                b += 1;
            }
            self.ins_full_blocks[c] = n / 16; // only complete blocks are permanently sealed
        }
    }

    /// Scan the append buffer for the probed `cells`, returning (approx_dist, global_appended_index).
    /// PQ path: the same scan_block kernels over the cell's appended blocks. ScalarI8 / bb==0: exact L2
    /// straight from ins_raw. Tombstoned points (ins_orig==u32::MAX) are skipped.
    fn scan_ins_pool(&self, q: &[i8], cells: &[u32], ctx: &QueryCtx) -> Vec<(i32, u32)> {
        let need_rows = self.comp.needs_raw_rows();
        let bb = self.bb;
        let d = self.d;
        let mut pool: Vec<(i32, u32)> = Vec::new();
        let mut out16 = [0i32; 16];
        let mut out32 = [0i32; 32];
        for &cell in cells {
            let c = cell as usize;
            let gidx = &self.ins_gidx[c];
            let n = gidx.len();
            if n == 0 { continue; }
            if need_rows || bb == 0 {
                for idx in 0..n {
                    let g = gidx[idx];
                    if self.ins_orig[g as usize] == u32::MAX { continue; }
                    let row = &self.ins_raw[g as usize * d..g as usize * d + d];
                    pool.push((simd::l2_i8(q, row), g));
                }
            } else {
                let blk = &self.ins_blocks[c];
                let nblk = blk.len() / bb; // == ceil(n/16) after finalize_inserts
                let mut b = 0;
                while b + 1 < nblk {
                    let b0 = &blk[b * bb..(b + 1) * bb];
                    let b1 = &blk[(b + 1) * bb..(b + 2) * bb];
                    self.comp.scan_block_x2(b0, b1, ctx, &mut out32);
                    for j in 0..16 { let s = b * 16 + j; if s < n { let g = gidx[s]; if self.ins_orig[g as usize] != u32::MAX { pool.push((out32[j], g)); } } }
                    for j in 0..16 { let s = (b + 1) * 16 + j; if s < n { let g = gidx[s]; if self.ins_orig[g as usize] != u32::MAX { pool.push((out32[16 + j], g)); } } }
                    b += 2;
                }
                if b < nblk {
                    let block = &blk[b * bb..(b + 1) * bb];
                    self.comp.scan_block(block, ctx, q, &[], &mut out16);
                    for j in 0..16 { let s = b * 16 + j; if s < n { let g = gidx[s]; if self.ins_orig[g as usize] != u32::MAX { pool.push((out16[j], g)); } } }
                }
            }
        }
        pool
    }

    /// Streaming search: scan the MAIN index AND the per-cell append buffer over the same probed cells,
    /// exact-rerank each store to scored (dist, orig) survivors, then merge for the global top-k. Deletes
    /// are already reflected (tombstoned slots/ins_orig are skipped). Falls back to the main path alone
    /// when nothing was inserted.
    pub fn search_stream(&self, ds: &I8Bin, q: &[i8], p: usize, t: usize, k: usize) -> Vec<u32> {
        let cells = self.router.probe(q, p);
        let ctx = self.comp.prepare_query(q);
        let mut pool = self.scan_pool(ds, q, &cells, &ctx);
        if self.a0 >= DEDUP_A0.load(std::sync::atomic::Ordering::Relaxed) || POOLDEDUP.load(std::sync::atomic::Ordering::Relaxed) {
            dedup_pool_by_orig(&mut pool, &self.slot_orig);
        }
        let tt = t.min(pool.len());
        if tt > 0 { pool.select_nth_unstable(tt - 1); pool.truncate(tt); }
        let mut cand = rerank_contig_pairs(&self.raw, self.d, &self.slot_orig, q, &pool, k, self.raw_orig_indexed);
        if !self.ins_raw.is_empty() {
            let mut bpool = self.scan_ins_pool(q, &cells, &ctx);
            let bt = t.min(bpool.len());
            if bt > 0 { bpool.select_nth_unstable(bt - 1); bpool.truncate(bt); }
            // ins_raw is per-appended-point (append order, slot==gidx), never orig-deduped -> by_orig=false.
            let bpairs = rerank_contig_pairs(&self.ins_raw, self.d, &self.ins_orig, q, &bpool, k, false);
            cand.extend_from_slice(&bpairs);
        }
        cand.sort_unstable();
        let mut out = Vec::with_capacity(k);
        let mut seen = std::collections::HashSet::new();
        for (_, orig) in cand {
            if seen.insert(orig) { out.push(orig); if out.len() == k { break; } }
        }
        out
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
        rerank_contig(&self.raw, self.d, &self.slot_orig, q, &pool2, k, self.raw_orig_indexed)
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

// ============================= Index persistence (SBANN_INDEX_SAVE/LOAD) =============================
// Trait objects are NOT serialized generically: each concrete Router/Compressor writes a 1-byte type
// tag (these constants) + its POD fields; load_router/load_comp read the tag and rebuild the type.
const ROUTER_TAG_HIER: u8 = 1;
const COMP_TAG_APQ4: u8 = 1;
const COMP_TAG_PQ4: u8 = 2;
const COMP_TAG_NORMPQ: u8 = 3;

fn save_pq(pq: &pq::Pq, w: &mut crate::persist::Sw) -> std::io::Result<()> {
    w.usize(pq.d)?;
    w.usize(pq.dpb)?;
    w.usize(pq.m)?;
    w.f32(pq.eta)?;
    w.f32s(&pq.cent)
}
fn load_pq(r: &mut crate::persist::Pr) -> pq::Pq {
    let d = r.usize();
    let dpb = r.usize();
    let m = r.usize();
    let eta = r.f32();
    let cent = r.f32_vec();
    pq::Pq { d, dpb, m, cent, eta }
}
fn save_opt_pq(o: &Option<pq::Pq>, w: &mut crate::persist::Sw) -> std::io::Result<()> {
    match o { Some(p) => { w.u8(1)?; save_pq(p, w) } None => w.u8(0) }
}
fn load_opt_pq(r: &mut crate::persist::Pr) -> Option<pq::Pq> {
    if r.u8() == 1 { Some(load_pq(r)) } else { None }
}
fn save_opt_residpq(o: &Option<pq::ResidPq>, w: &mut crate::persist::Sw) -> std::io::Result<()> {
    match o {
        Some(p) => { w.u8(1)?; w.usize(p.d)?; w.usize(p.dpb)?; w.usize(p.m)?; w.f32s(&p.cent) }
        None => w.u8(0),
    }
}
fn load_opt_residpq(r: &mut crate::persist::Pr) -> Option<pq::ResidPq> {
    if r.u8() == 1 {
        let d = r.usize();
        let dpb = r.usize();
        let m = r.usize();
        let cent = r.f32_vec();
        Some(pq::ResidPq { d, dpb, m, cent })
    } else { None }
}

/// Rebuild the concrete Router from its tag + fields (the inverse of `Router::save`).
fn load_router(r: &mut crate::persist::Pr) -> Box<dyn Router> {
    let tag = r.u8();
    match tag {
        ROUTER_TAG_HIER => {
            let d = r.usize();
            let kf = r.usize();
            let levels = r.usize();
            let soar = r.f32();
            let mu = r.f32_vec();
            let nc = r.usize();
            let cent: Vec<Vec<i8>> = (0..nc).map(|_| r.i8_vec()).collect();
            let ncc = r.usize();
            let child: Vec<Vec<u32>> = (0..ncc).map(|_| r.u32_vec()).collect();
            let beam = r.usize_vec();
            let radc = load_opt_pq(r);
            let rcodes = r.u8_vec();
            let rblocks = r.u8_vec();
            let cadj = cadj_of(&cent, d);
            Box::new(HierRouter { d, mu, kf, levels, cent, child, beam, soar, radc, rcodes, rblocks, cadj })
        }
        _ => panic!("unknown router type tag {tag} in index file (only HierRouter={ROUTER_TAG_HIER} supported)"),
    }
}

/// Rebuild the concrete Compressor from its tag + fields (the inverse of `Compressor::save`).
fn load_comp(r: &mut crate::persist::Pr) -> Box<dyn Compressor> {
    let tag = r.u8();
    match tag {
        COMP_TAG_APQ4 => {
            let d = r.usize();
            let dpb = r.usize();
            let eta = r.f32();
            let pq = load_pq(r);
            Box::new(Apq4 { pq, d, dpb, eta })
        }
        COMP_TAG_PQ4 => {
            let pq = load_pq(r);
            Box::new(Pq4 { pq })
        }
        COMP_TAG_NORMPQ => {
            let d = r.usize();
            let dpb = r.usize();
            let eta = r.f32();
            let pq = load_pq(r);
            Box::new(NormPq { pq, d, dpb, eta })
        }
        _ => panic!("unknown compressor type tag {tag} in index file (only Apq4={COMP_TAG_APQ4}, Pq4={COMP_TAG_PQ4}, NormPq={COMP_TAG_NORMPQ} supported)"),
    }
}

impl Index {
    /// Serialize the whole index to `path` (magic+version header, then POD arrays, then the tagged
    /// concrete router+compressor). Streams to a BufWriter so the n*a0*d `raw`/`blocks` arrays never
    /// get a second in-RAM copy. Errors clearly if the router/comp type is not one of the supported tags.
    pub fn save_to(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let f = std::fs::File::create(path)?;
        let mut bw = std::io::BufWriter::new(f);
        {
            let mut w = crate::persist::Sw { w: &mut bw };
            w.w.write_all(crate::persist::MAGIC)?;
            w.u32(crate::persist::VERSION)?;
            // POD scalars
            w.usize(self.d)?;
            w.usize(self.bb)?;
            w.usize(self.a0)?;
            w.u8(self.raw_orig_indexed as u8)?;
            // POD arrays (each length-prefixed)
            w.u32s(&self.cell_bstart)?;
            w.u32s(&self.slot_orig)?;
            w.u8s(&self.blocks)?;
            w.i32s(&self.xfn)?;
            w.i8s(&self.raw)?;
            w.u8s(&self.blocks_il)?;
            w.u32s(&self.cell_ilstart)?;
            w.u8s(&self.resid_codes)?;
            w.i8s(&self.rq_cent)?;
            save_opt_residpq(&self.resid_pq, &mut w)?;
            // concrete router + compressor (each writes its own 1-byte type tag)
            self.router.save(&mut w)?;
            self.comp.save(&mut w)?;
        }
        bw.flush()?;
        Ok(())
    }

    /// Reconstruct an index from a file written by `save_to`. mmaps the file and COPIES each region into
    /// owned Vecs (the Index owns its arrays), then drops the mmap. Recall is bit-identical to the in-RAM
    /// build (same arrays, same router/comp). The build/scan global flags (FASTSCAN/USE512FS/RESID/...)
    /// must match the build env, exactly as for the in-RAM path — they gate which arrays the search reads.
    pub fn load_from(path: &str) -> std::io::Result<Index> {
        let f = std::fs::File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&f)? };
        if mmap.len() < 12 || &mmap[0..8] != &crate::persist::MAGIC[..] {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "not an sbann index file (bad magic)"));
        }
        let mut r = crate::persist::Pr::new(&mmap[..]);
        r.pos = 8;
        let ver = r.u32();
        if ver != crate::persist::VERSION {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                format!("index version {ver} != supported {}", crate::persist::VERSION)));
        }
        let d = r.usize();
        let bb = r.usize();
        let a0 = r.usize();
        let raw_orig_indexed = r.u8() != 0;
        let cell_bstart = r.u32_vec();
        let slot_orig = r.u32_vec();
        let blocks = r.u8_vec();
        let xfn = r.i32_vec();
        let raw = r.i8_vec();
        let blocks_il = r.u8_vec();
        let cell_ilstart = r.u32_vec();
        let resid_codes = r.u8_vec();
        let rq_cent = r.i8_vec();
        let resid_pq = load_opt_residpq(&mut r);
        let router = load_router(&mut r);
        let comp = load_comp(&mut r);
        let nc = router.n_cells();
        let n_main = slot_orig.iter().filter(|&&o| o != u32::MAX).count();
        Ok(Index {
            router, comp, cell_bstart, slot_orig, blocks, bb, xfn, raw, d,
            blocks_il, cell_ilstart, resid_pq, resid_codes, rq_cent, a0, raw_orig_indexed,
            // a loaded index is the built main index, no pending inserts.
            ins_blocks: vec![Vec::new(); nc], ins_gidx: vec![Vec::new(); nc], ins_full_blocks: vec![0; nc],
            ins_raw: Vec::new(), ins_orig: Vec::new(), ins_loc: std::collections::HashMap::new(),
            ins_dirty: Vec::new(), ins_count: 0, main_rev: None, n_main,
        })
    }
}
