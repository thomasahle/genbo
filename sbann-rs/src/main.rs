//! sbann — streaming int8 IVF-PQ ANN for billion-scale, all-native (no FFI).
//!
//! `build`  : streaming bounded-memory build proof (mmap -> mean -> SIMD assign -> cell counts).
//! `bench`  : end-to-end IVF + exact int8 rerank, recall@10 vs QPS on a dataset+queries+gt.
//! All SIMD lives in `simd.rs` (Rust AVX2, scalar-validated). PQ-ADC bucket scan is the next
//! kernel to fold into the query path; today's rerank is exact int8 L2 over the probed pool.

mod fbin;
mod ibin;
mod kmeans;
mod persist;
mod pq;
mod simd;
mod vq;

use ibin::I8Bin;
use rayon::prelude::*;
use std::time::Instant;

/// In-memory IVF: normalized random i8 pivots + CSR inverted lists + mean (for query normalize).
struct Ivf {
    d: usize,
    c: usize,
    pivots: Vec<i8>,    // c*d, NORMALIZED (mean-centered, unit, *127)
    cell_start: Vec<u32>, // c+1
    ids: Vec<u32>,      // nb, point ids grouped by cell
    mu: Vec<f32>,       // d, dataset mean
}

fn build_ivf(ds: &I8Bin, c: usize, a0: usize) -> Ivf {
    let (nb, d) = (ds.nb, ds.d);
    // streaming mean (parallel reduce, bounded memory)
    let sum: Vec<f64> = (0..nb)
        .into_par_iter()
        .fold(|| vec![0f64; d], |mut a, i| { let r = ds.row(i); for k in 0..d { a[k] += r[k] as f64; } a })
        .reduce(|| vec![0f64; d], |mut a, b| { for k in 0..d { a[k] += b[k]; } a });
    let mu: Vec<f32> = sum.iter().map(|s| (s / nb as f64) as f32).collect();
    // normalized random pivots (deterministic LCG sample of row ids)
    let mut pivots = vec![0i8; c * d];
    let mut seed = 0x9e3779b97f4a7c15u64;
    for j in 0..c {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let id = (seed >> 11) as usize % nb;
        let (a, b) = (j * d, j * d + d);
        simd::normalize_i8(ds.row(id), &mu, &mut pivots[a..b]);
    }
    // parallel top-a0 assignment on NORMALIZED vectors (each point -> a0 nearest cells)
    let assign: Vec<u32> = (0..nb)
        .into_par_iter()
        .map_init(|| (vec![0i8; d], vec![0u32; a0]), |(buf, top), i| {
            simd::normalize_i8(ds.row(i), &mu, buf);
            simd::assign_topk(buf, &pivots, d, a0, top);
            top.clone()
        })
        .flatten()
        .collect(); // length nb*a0, a0 cells contiguous per point
    // CSR via counting sort (nb*a0 entries; cells bounded by c, O(n))
    let mut cell_start = vec![0u32; c + 1];
    for &a in &assign {
        cell_start[a as usize + 1] += 1;
    }
    for j in 0..c {
        cell_start[j + 1] += cell_start[j];
    }
    let mut ids = vec![0u32; nb * a0];
    let mut cur = cell_start.clone();
    for (idx, &a) in assign.iter().enumerate() {
        let point = (idx / a0) as u32;
        let slot = &mut cur[a as usize];
        ids[*slot as usize] = point;
        *slot += 1;
    }
    Ivf { d, c, pivots, cell_start, ids, mu }
}

/// Top-`k` distinct ids by ascending distance (a0>1 can surface a point via multiple cells).
pub(crate) fn topk_dedup(mut cand: Vec<(i32, u32)>, k: usize) -> Vec<u32> {
    let take = (k * 3).min(cand.len());
    if take > 0 {
        cand.select_nth_unstable(take - 1);
        cand.truncate(take);
    }
    cand.sort_unstable();
    let mut out: Vec<u32> = Vec::with_capacity(k);
    for &(_, id) in &cand {
        if !out.contains(&id) {
            out.push(id);
            if out.len() == k {
                break;
            }
        }
    }
    out
}

/// Route to top-`p` cells, gather their points, exact int8 L2, return top-`k` distinct ids.
fn search(ivf: &Ivf, ds: &I8Bin, q: &[i8], p: usize, k: usize) -> Vec<u32> {
    let mut cand: Vec<(i32, u32)> = Vec::with_capacity(4096);
    for cell in route(ivf, q, p) {
        let (s, e) = (ivf.cell_start[cell as usize] as usize, ivf.cell_start[cell as usize + 1] as usize);
        for &id in &ivf.ids[s..e] {
            cand.push((simd::l2_i8(q, ds.row(id as usize)), id));
        }
    }
    topk_dedup(cand, k)
}

/// Route a query to the `p` nearest cells: normalize the query, then SIMD-L2 to normalized pivots.
fn route(ivf: &Ivf, q: &[i8], p: usize) -> Vec<u32> {
    let d = ivf.d;
    let mut qn = vec![0i8; d];
    simd::normalize_i8(q, &ivf.mu, &mut qn);
    let mut cd: Vec<(i32, u32)> = (0..ivf.c)
        .map(|j| (simd::l2_i8(&qn, &ivf.pivots[j * d..j * d + d]), j as u32))
        .collect();
    let p = p.min(cd.len());
    cd.select_nth_unstable(p - 1);
    cd[..p].iter().map(|&(_, j)| j).collect()
}

/// PQ cascade index: nibble-packed blocks grouped by cell + slot->orig map.
struct PqIndex {
    blocks: Vec<u8>,
    cell_bstart: Vec<u32>, // c+1, in blocks
    slot_orig: Vec<u32>,   // total_blocks*16
    bb: usize,             // bytes per block = m/2*16
}

fn build_pq_index(ivf: &Ivf, ds: &I8Bin, pq: &pq::Pq) -> PqIndex {
    let bb = pq.m / 2 * 16;
    let mut blocks: Vec<u8> = Vec::new();
    let mut cell_bstart = vec![0u32; ivf.c + 1];
    let mut slot_orig: Vec<u32> = Vec::new();
    let mut codes16 = [[0u8; 512]; 16];
    for cell in 0..ivf.c {
        let (s, e) = (ivf.cell_start[cell] as usize, ivf.cell_start[cell + 1] as usize);
        let pts = &ivf.ids[s..e];
        let mut i = 0;
        while i < pts.len() {
            let cnt = (pts.len() - i).min(16);
            for j in 0..16 {
                if j < cnt {
                    let id = pts[i + j];
                    pq.encode(ds.row(id as usize), &mut codes16[j][..pq.m]);
                    slot_orig.push(id);
                } else {
                    for k in 0..pq.m {
                        codes16[j][k] = 0;
                    }
                    slot_orig.push(u32::MAX);
                }
            }
            pq::pack_block(&codes16, pq.m, &mut blocks);
            i += 16;
        }
        cell_bstart[cell + 1] = (blocks.len() / bb) as u32;
    }
    PqIndex { blocks, cell_bstart, slot_orig, bb }
}

/// PQ cascade search: route -> SIMD ADC scan probed cells -> top-T by approx -> exact rerank.
fn search_pq(ivf: &Ivf, idx: &PqIndex, pq: &pq::Pq, ds: &I8Bin, q: &[i8], p: usize, t: usize, k: usize) -> Vec<u32> {
    let cells = route(ivf, q, p);
    let lut = pq.query_lut(q);
    let regs = pq::lut_regs(&lut, pq.m);
    let mut pool: Vec<(i32, u32)> = Vec::with_capacity(4096);
    let mut out16 = [0i8; 16];
    for &cell in &cells {
        let (bs, be) = (idx.cell_bstart[cell as usize] as usize, idx.cell_bstart[cell as usize + 1] as usize);
        for b in bs..be {
            let blk = &idx.blocks[b * idx.bb..(b + 1) * idx.bb];
            unsafe { pq::block_adc_sse(blk, pq.m, &regs, &mut out16) };
            for i in 0..16 {
                let oid = idx.slot_orig[b * 16 + i];
                if oid != u32::MAX {
                    pool.push((out16[i] as i32, oid));
                }
            }
        }
    }
    let tt = t.min(pool.len());
    if tt > 0 {
        pool.select_nth_unstable(tt - 1);
        pool.truncate(tt);
    }
    let cand: Vec<(i32, u32)> =
        pool.iter().map(|&(_, id)| (simd::l2_i8(q, ds.row(id as usize)), id)).collect();
    topk_dedup(cand, k)
}

fn benchpq(base: &str, qpath: &str, gtpath: &str, c: usize, dpb: usize) {
    let t0 = Instant::now();
    let ds = I8Bin::open(base).expect("base");
    assert!(simd::selftest(ds.d), "L2 selftest failed");
    let m = ds.d / dpb;
    assert!(pq::selftest(m), "PQ ADC selftest failed");
    println!("base nb={} d={}  simd=ok pq-adc-selftest=ok (M={m})", ds.nb, ds.d);
    let ivf = build_ivf(&ds, c, 2);
    // train PQ on a sample of points
    let sample: Vec<&[i8]> = (0..ds.nb.min(40000)).map(|i| ds.row(i * (ds.nb / ds.nb.min(40000)))).collect();
    let pqc = pq::Pq::train(&sample, ds.d, dpb, 6);
    let idx = build_pq_index(&ivf, &ds, &pqc);
    println!(
        "built IVF C={c} + PQ(dpb={dpb}) blocks={} in {:.1}s",
        idx.slot_orig.len() / 16,
        t0.elapsed().as_secs_f64()
    );

    let qs = I8Bin::open(qpath).expect("queries");
    let (gnq, gk, gids) = read_gt(gtpath);
    let nq = qs.nb.min(gnq);
    for &(p, t) in &[(32usize, 1500usize), (64, 2500), (128, 4000)] {
        let st = Instant::now();
        let res: Vec<Vec<u32>> = (0..nq)
            .into_par_iter()
            .map(|i| search_pq(&ivf, &idx, &pqc, &ds, qs.row(i), p, t, 10))
            .collect();
        let dt = st.elapsed().as_secs_f64();
        let mut hit = 0usize;
        for i in 0..nq {
            let truth: std::collections::HashSet<u32> = gids[i * gk..i * gk + 10].iter().copied().collect();
            hit += res[i].iter().take(10).filter(|id| truth.contains(id)).count();
        }
        println!("  p={p:4} T={t:5}: recall@10={:.4}  QPS={:.0}", hit as f64 / (nq * 10) as f64, nq as f64 / dt);
    }
}

/// AVQ-routing: additive 2-codebook (residual) inverted multi-index. Measures pool-at-recall vs
/// flat IVF. x_hat = C0[i0] + C1[i1] (full-dim codebooks); cell dist to (i0,i1) =
/// A[i0] + B[i1] + 2*cross[i0,i1] where A=-2 q.C0+||C0||^2, B likewise, cross=C0.C1.
fn benchavq(base: &str, qpath: &str, gtpath: &str, c0n: usize, c1n: usize) {
    let t0 = Instant::now();
    let ds = I8Bin::open(base).expect("base");
    let (n, d) = (ds.nb, ds.d);
    // streaming mean
    let sum: Vec<f64> = (0..n).into_par_iter()
        .fold(|| vec![0f64; d], |mut a, i| { let r = ds.row(i); for k in 0..d { a[k] += r[k] as f64; } a })
        .reduce(|| vec![0f64; d], |mut a, b| { for k in 0..d { a[k] += b[k]; } a });
    let mu: Vec<f32> = if std::env::var("SBANN_NOMU").is_ok() { vec![0f32; ds.d] } else { sum.iter().map(|s| (s / n as f64) as f32).collect() };
    // normalized f32 vectors (1M*100*4 = 400MB; this bench targets 1M)
    let mut xn = vec![0f32; n * d];
    xn.par_chunks_mut(d).enumerate().for_each(|(i, row)| simd::norm_f32(ds.row(i), &mu, row));
    // codebook 0 (random normalized points), assign i0
    let smp = n.min(200_000); // k-means training sample
    let c0 = kmeans::kmeans_f32(&xn[..smp * d], smp, d, c0n, 15, 0xc0c0);
    let i0: Vec<u32> = (0..n).into_par_iter().map(|i| {
        let x = &xn[i * d..i * d + d];
        (0..c0n).map(|j| (simd::l2_f32(x, &c0[j * d..j * d + d]), j as u32)).min_by(|a, b| a.0.total_cmp(&b.0)).unwrap().1
    }).collect();
    // residuals -> codebook 1, assign i1
    let res: Vec<f32> = (0..n * d).into_par_iter().map(|p| { let i = p / d; let k = p % d; xn[p] - c0[i0[i] as usize * d + k] }).collect();
    let c1 = kmeans::kmeans_f32(&res[..smp * d], smp, d, c1n, 15, 0xc1c1);
    let i1: Vec<u32> = (0..n).into_par_iter().map(|i| {
        let r = &res[i * d..i * d + d];
        (0..c1n).map(|j| (simd::l2_f32(r, &c1[j * d..j * d + d]), j as u32)).min_by(|a, b| a.0.total_cmp(&b.0)).unwrap().1
    }).collect();
    // CSR over c0n*c1n buckets
    let nc = c0n * c1n;
    let mut cell_start = vec![0u32; nc + 1];
    let key: Vec<u32> = (0..n).map(|i| i0[i] * c1n as u32 + i1[i]).collect();
    for &k in &key { cell_start[k as usize + 1] += 1; }
    for j in 0..nc { cell_start[j + 1] += cell_start[j]; }
    let mut ids = vec![0u32; n];
    let mut cur = cell_start.clone();
    for i in 0..n { let s = &mut cur[key[i] as usize]; ids[*s as usize] = i as u32; *s += 1; }
    // precompute ||C0||^2, ||C1||^2, cross = C0.C1^T
    let c0sq: Vec<f32> = (0..c0n).map(|j| simd::dot_f32(&c0[j * d..j * d + d], &c0[j * d..j * d + d])).collect();
    let c1sq: Vec<f32> = (0..c1n).map(|j| simd::dot_f32(&c1[j * d..j * d + d], &c1[j * d..j * d + d])).collect();
    let cross: Vec<f32> = (0..c0n).into_par_iter().flat_map(|a| {
        (0..c1n).map(|b| simd::dot_f32(&c0[a * d..a * d + d], &c1[b * d..b * d + d])).collect::<Vec<_>>()
    }).collect();
    println!("AVQ build C0={c0n} C1={c1n} ({nc} cells) in {:.1}s", t0.elapsed().as_secs_f64());

    let qs = I8Bin::open(qpath).expect("q");
    let (gnq, gk, gids) = read_gt(gtpath);
    let nq = qs.nb.min(gnq);
    for &p in &[64usize, 256, 1024, 4096] {
        let st = Instant::now();
        let stats: Vec<(usize, usize)> = (0..nq).into_par_iter().map(|qi| {
            let mut qn = vec![0f32; d];
            simd::norm_f32(qs.row(qi), &mu, &mut qn);
            let a: Vec<f32> = (0..c0n).map(|j| -2.0 * simd::dot_f32(&qn, &c0[j * d..j * d + d]) + c0sq[j]).collect();
            let b: Vec<f32> = (0..c1n).map(|j| -2.0 * simd::dot_f32(&qn, &c1[j * d..j * d + d]) + c1sq[j]).collect();
            // cell distances over all buckets, take top-p
            let mut dcell: Vec<(f32, u32)> = (0..nc).map(|c| {
                let (ia, ib) = (c / c1n, c % c1n);
                (a[ia] + b[ib] + 2.0 * cross[c], c as u32)
            }).collect();
            let pp = p.min(nc);
            dcell.select_nth_unstable_by(pp - 1, |x, y| x.0.total_cmp(&y.0));
            // gather pool + exact rerank
            let q = qs.row(qi);
            let mut cand: Vec<(i32, u32)> = Vec::new();
            for &(_, c) in &dcell[..pp] {
                let (s, e) = (cell_start[c as usize] as usize, cell_start[c as usize + 1] as usize);
                for &id in &ids[s..e] { cand.push((simd::l2_i8(q, ds.row(id as usize)), id)); }
            }
            let pool = cand.len();
            let top = topk_dedup(cand, 10);
            let truth: std::collections::HashSet<u32> = gids[qi * gk..qi * gk + 10].iter().copied().collect();
            (top.iter().filter(|id| truth.contains(id)).count(), pool)
        }).collect();
        let dt = st.elapsed().as_secs_f64();
        let rec = stats.iter().map(|s| s.0).sum::<usize>() as f64 / (nq * 10) as f64;
        let pool = stats.iter().map(|s| s.1).sum::<usize>() / nq;
        println!("  cells={p:5}: recall@10={rec:.4}  avg_pool={pool}  QPS={:.0}", nq as f64 / dt);
    }
}

/// SBANN_SOAR=λ enables SOAR spilled multi-assignment for a HierRouter at build (idea #3). No-op
/// (default L2 assignment) when unset. Only affects a0>=2 builds.
fn apply_soar(r: &mut vq::HierRouter) {
    if let Ok(s) = std::env::var("SBANN_SOAR") {
        if let Ok(v) = s.parse::<f32>() { if v > 0.0 { r.set_soar(v); println!("  [SOAR spilled assignment lambda={v}]"); } }
    }
}

/// Pluggable run: `run <base> <q> <gt> <router> <compress> [a0]`. router=flat|flatrand|avq,
/// compress=pq4|i8. Sweeps p, reports recall@10 / avg pool / QPS so routers compare at matched pool.
fn run(base: &str, qpath: &str, gtpath: &str, router_s: &str, comp_s: &str, a0: usize, c: usize, tmul: usize, batched: bool) {
    let t0 = Instant::now();
    let mut ds = I8Bin::open(base).expect("base");
    let preset_env = std::env::var("SBANN_PRESET").ok();
    let target_env = std::env::var("SBANN_TARGET_RECALL").ok();
    let (search_preset, preset_source) = resolve_search_preset(
        preset_env.as_deref(),
        target_env.as_deref(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    if std::env::var("SBANN_CASCADE_K").is_err()
        && std::env::var("SBANN_KLIST").is_err()
    {
        vq::CASCADE_K.store(
            search_preset.cascade_k(ds.d),
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    // RESIDENT-I8 (P346, flag-gated): anonymous THP-backed copy of the int8 base. The scattered union
    // rescore pays a TLB miss + page walk per row on the 4KB-paged file mmap (~918 rows/q at DEEP loose
    // configs); 2MB pages cut TLB entries ~500x. Data byte-identical — recall unchanged.
    if std::env::var("SBANN_RESIDENT_I8").is_ok() {
        let tr = Instant::now();
        ds.make_resident();
        println!("  [RESIDENT-I8] {}MB anonymous (THP-eligible)  setup={:.1}s", ds.nb * ds.d / 1_000_000, tr.elapsed().as_secs_f64());
    }
    let ds = ds;
    let n = ds.nb;
    // SBANN_INDEX_LOAD: skip the (minutes-long) router/comp train + encode and instead mmap+copy a
    // prebuilt index (seconds). Everything below in the else-branch (mean, route-train open, k-means,
    // PQ encode, Index::build) is build-only, so loading bypasses it entirely.
    let idx = if let Some(lp) = std::env::var("SBANN_INDEX_LOAD").ok() {
        let tl = Instant::now();
        let i = vq::Index::load_from(&lp).expect("index load");
        println!("[loaded index from {lp}] in {:.2}s (build skipped)", tl.elapsed().as_secs_f64());
        i
    } else {
    let sum: Vec<f64> = (0..n).into_par_iter()
        .fold(|| vec![0f64; ds.d], |mut a, i| { let r = ds.row(i); for k in 0..ds.d { a[k] += r[k] as f64; } a })
        .reduce(|| vec![0f64; ds.d], |mut a, b| { for k in 0..ds.d { a[k] += b[k]; } a });
    let mu: Vec<f32> = if std::env::var("SBANN_NOMU").is_ok() { vec![0f32; ds.d] } else { sum.iter().map(|s| (s / n as f64) as f32).collect() };
    let cb = (c as f64).sqrt().round() as usize; // per-codebook size for AVQ multi-index
    // hierk routing fan-out overrides: SBANN_C0 = #coarse cells, SBANN_B0 = #coarse expanded/query.
    // Routing cost/query ~= C0 + B0*(Kf/C0); the default C0=sqrt(Kf), B0=C0/4 is one point on that curve.
    let c0 = std::env::var("SBANN_C0").ok().and_then(|s| s.parse().ok()).unwrap_or(cb);
    let b0 = std::env::var("SBANN_B0").ok().and_then(|s| s.parse().ok()).unwrap_or((cb / 4).max(8));
    // OOD-aware routing: train the cells on a SEPARATE distribution (e.g. query.learn) so OOD
    // queries route to cells holding their true neighbors. Index is still BUILT on the base `ds`.
    let route_path = std::env::var("SBANN_ROUTE_TRAIN").unwrap_or_else(|_| base.to_string());
    let dr = I8Bin::open(&route_path).expect("route-train");
    if route_path != base { println!("  [OOD routing trained on {route_path} n={}]", dr.nb); }

    let router: Box<dyn vq::Router> = match router_s {
        "flat" => Box::new(vq::FlatIvf::train(&dr, c, mu.clone(), 15)),
        "flatsoar" => Box::new(vq::FlatIvf::train_soar(&dr, c, mu.clone(), 15, 1.0)),
        "flatrair" => Box::new(vq::FlatIvf::train_rair(&dr, c, mu.clone(), 15, 1.0)),
        "flatrand" => Box::new(vq::FlatIvf::train(&dr, c, mu.clone(), 0)),
        "avq" => Box::new(vq::AvqRouter::train(&dr, cb, cb, mu.clone(), 15)),
        "hier" => { let mut r = vq::HierRouter::train(&dr, c, c0, b0, mu.clone()); apply_soar(&mut r); Box::new(r) }
        "hierk" => { let mut r = vq::HierRouter::train_hkmeans(&dr, c, c0, b0, mu.clone()); apply_soar(&mut r); Box::new(r) }
        // 3-level: SBANN_C1 = #mid cells (default sqrt(C0*Kf)), SBANN_B1 = #mids expanded/query.
        "hierk3" => {
            let c1 = std::env::var("SBANN_C1").ok().and_then(|s| s.parse().ok()).unwrap_or(((c0 as f64 * c as f64).sqrt().round() as usize).max(c0 * 2));
            let b1 = std::env::var("SBANN_B1").ok().and_then(|s| s.parse().ok()).unwrap_or((b0 * 2).max(16));
            println!("  [hierk3 C0={c0} C1={c1} b0={b0} b1={b1}]");
            let mut r = vq::HierRouter::train_hkmeans3(&dr, c, c0, c1, b0, b1, mu.clone());
            apply_soar(&mut r);
            Box::new(r)
        }
        // ARBITRARY-DEPTH (hierk4/5/… for 100M/1B): SBANN_LEVELS = per-level cell counts coarse→fine
        // (last = Kf), SBANN_BEAMS = per-level beams (len L-1). Routing ≈ O(L·Kf^(1/L)). Defaults to a
        // 4-level geometric ladder from C0→Kf if SBANN_LEVELS unset.
        "hierkn" => {
            let levels: Vec<usize> = std::env::var("SBANN_LEVELS").ok()
                .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect::<Vec<_>>())
                .filter(|v: &Vec<usize>| v.len() >= 2)
                .unwrap_or_else(|| {
                    // default L=4 geometric: C0, C0*r, C0*r^2, Kf  (r = (Kf/C0)^(1/3))
                    let r = (c as f64 / c0 as f64).powf(1.0 / 3.0);
                    vec![c0, (c0 as f64 * r).round() as usize, (c0 as f64 * r * r).round() as usize, c]
                });
            let l = levels.len();
            let beams: Vec<usize> = std::env::var("SBANN_BEAMS").ok()
                .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect::<Vec<_>>())
                .filter(|v: &Vec<usize>| v.len() == l - 1)
                .unwrap_or_else(|| (0..l - 1).map(|i| (b0 << i).max(16)).collect());
            println!("  [hierkn levels={levels:?} beams={beams:?}]");
            let mut r = vq::HierRouter::train_hkmeans_multi(&dr, &levels, &beams, mu.clone());
            apply_soar(&mut r);
            Box::new(r)
        }
        _ => { eprintln!("router? (flat|flatsoar|flatrair|flatrand|avq|hier|hierk|hierk3|hierkn)"); return; }
    };
    // SBANN_DPB: dims-per-block for PQ (default 2). dpb=1 -> finer 4-bit-per-dim quant (more codes
    // to scan but better ranking) -- tests whether finer quant cuts probes at high recall.
    let dpb: usize = std::env::var("SBANN_DPB").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
    // SBANN_ETA: anisotropic parallel-error weight for apq4/aopq. Default 4 (L2-tuned). For OOD/IP,
    // higher eta concentrates quantization accuracy on the IP-relevant (parallel) direction -> the true
    // top-k surface in a shallower scan (P-research: never swept on OOD; d=200 may want 16-64).
    let eta: f32 = std::env::var("SBANN_ETA").ok().and_then(|s| s.parse().ok()).unwrap_or(4.0);
    // RaBitQ knobs: SBANN_RABITQ_BITS = B bits/coordinate (default 1 = sign bits); SBANN_RABITQ_L =
    // multi-bit uniform quantizer half-range on the unit-variance rotated coords (default 2.5).
    let rbq_bits: usize = std::env::var("SBANN_RABITQ_BITS").ok().and_then(|s| s.parse().ok()).unwrap_or(1).clamp(1, 8);
    let rbq_l: f32 = std::env::var("SBANN_RABITQ_L").ok().and_then(|s| s.parse().ok()).unwrap_or(2.5);
    let comp: Box<dyn vq::Compressor> = match comp_s {
        "pq4" => Box::new(vq::Pq4::train(&ds, dpb, 6)),
        "opq4" => Box::new(vq::Opq4::train(&ds, dpb, 6)),
        "opql" => Box::new(vq::Opq4::train_learned(&ds, dpb, 6, 8)),
        "opql5" => Box::new(vq::Opq4::train_learned(&ds, 5, 6, 8)),
        "apq4" => Box::new(vq::Apq4::train(&ds, dpb, 6, eta)),
        "aopq" => Box::new(vq::Opq4::train_aopq(&ds, dpb, 6, 8, eta)),
        "i8" => Box::new(vq::ScalarI8::new(ds.d)),
        "rabitq" => { println!("  [rabitq B={rbq_bits} L={rbq_l}]"); Box::new(vq::RaBitQ::new(ds.d, rbq_bits, rbq_l, 0x5a17)) }
        _ => { eprintln!("compress?"); return; }
    };
    let idx = vq::Index::build(router, comp, &ds, a0);
    println!("[{router_s}+{comp_s} a0={a0}] built in {:.1}s", t0.elapsed().as_secs_f64());
    // SBANN_INDEX_SAVE: persist the freshly built index, then continue into the bench below (so the same
    // process gives the in-RAM recall to compare the reloaded run against).
    if let Ok(sp) = std::env::var("SBANN_INDEX_SAVE") {
        let tsv = Instant::now();
        idx.save_to(&sp).expect("index save");
        let sz = std::fs::metadata(&sp).map(|m| m.len()).unwrap_or(0);
        println!("[saved index to {sp}] in {:.2}s, {sz} bytes ({:.3} GB)", tsv.elapsed().as_secs_f64(), sz as f64 / 1e9);
    }
    idx
    };

    // SCAN-PRECISION POLICY (P239): FASTSCAN2's int8-sat accumulate caps each subspace LUT at ~4 bits
    // (FS2_CAP=120/HOIST); LUT quantization noise grows ~sqrt(m), so at high subspace counts it drowns
    // small (in-distribution) neighbor gaps — cohere768 m=384: in-pool ordering 0.07 (i8s) vs 0.87 (int16)
    // on the SAME codes. Default to the int16 LUT path when m>128 (d>256 at dpb=2); an explicit
    // SBANN_FASTSCAN2 always wins. d<=256 (m<=128, e.g. t2i d=200 m=100) is untouched: champion path.
    let m_scan = idx.bb / 8; // 4-bit PQ blocks: bb = m/2*16 (non-PQ comps land >128 harmlessly: FS2 only scans PQ blocks)
    if m_scan > 128 && std::env::var("SBANN_FASTSCAN2").is_err()
        && vq::FASTSCAN2.load(std::sync::atomic::Ordering::Relaxed) {
        vq::FASTSCAN2.store(false, std::sync::atomic::Ordering::Relaxed);
        println!("  [scan-precision policy: m={m_scan}>128 -> int16 LUT scan (FASTSCAN2 auto-off; SBANN_FASTSCAN2=1 forces i8s)]");
    }

    let qs = I8Bin::open(qpath).expect("q");
    let (gnq, gk, gids) = read_gt(gtpath);
    // SBANN_NQ caps the #queries (for fair same-NQ head-to-head vs the Python frontier's NQ=1000).
    let nq_cap = std::env::var("SBANN_NQ").ok().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    let nq = qs.nb.min(gnq).min(nq_cap);
    // FLOAT-RERANK (SBANN_FLOAT_RERANK, P191 lever stack): int8 scan/route STAY (FASTSCAN2+PREFETCH),
    // but the exact survivor rerank reads the ORIGINAL float vectors (SBANN_FBASE, first ds.nb rows)
    // using the float queries (SBANN_FQUERY) -> float-precision ranking vs the leaderboard's float GT.
    let float_rerank = std::env::var("SBANN_FLOAT_RERANK").is_ok();
    let fbase: Option<fbin::FBin> = if float_rerank {
        let p = std::env::var("SBANN_FBASE").expect("SBANN_FLOAT_RERANK set but SBANN_FBASE missing");
        let fb = fbin::FBin::open(&p, ds.nb).expect("fbase");
        assert_eq!(fb.d, ds.d, "fbase dim != index dim"); assert!(fb.nb >= ds.nb, "fbase has fewer rows than index");
        println!("  [FLOAT_RERANK fbase={p} nb={} d={}]", fb.nb, fb.d);
        Some(fb)
    } else { None };
    // RERANK-F16 (P345, flag-gated): convert the f32 rerank base to a RESIDENT fp16 copy (half the
    // page footprint; fixes the 35M-scale page-cache thrash where the 143GB f32 mmap can't stay warm).
    if float_rerank && std::env::var("SBANN_RERANK_F16").is_ok() {
        let fb = fbase.as_ref().unwrap();
        let t0 = Instant::now();
        let (nb, dd) = (fb.nb, fb.d);
        let mut h = vec![0u16; nb * dd];
        h.par_chunks_mut(dd).enumerate().for_each(|(i, out)| {
            let row = fb.row(i);
            let mut j = 0usize;
            unsafe {
                use std::arch::x86_64::*;
                while j + 8 <= dd {
                    let f = _mm256_loadu_ps(row.as_ptr().add(j));
                    let ph = _mm256_cvtps_ph(f, _MM_FROUND_TO_NEAREST_INT);
                    _mm_storeu_si128(out.as_mut_ptr().add(j) as *mut __m128i, ph);
                    j += 8;
                }
                while j < dd {
                    let ph = _mm256_cvtps_ph(_mm256_set1_ps(row[j]), _MM_FROUND_TO_NEAREST_INT);
                    out[j] = _mm_extract_epi16(ph, 0) as u16;
                    j += 1;
                }
            }
        });
        println!("  [RERANK-F16] resident fp16 base {}MB (f32 mmap was {}MB)  setup={:.1}s",
            nb * dd * 2 / 1_000_000, nb * dd * 4 / 1_000_000, t0.elapsed().as_secs_f64());
        let _ = vq::F16BASE.set(h);
    }
    let fqf: Vec<f32> = if float_rerank {
        let p = std::env::var("SBANN_FQUERY").expect("SBANN_FLOAT_RERANK set but SBANN_FQUERY missing");
        let fq = fbin::FBin::open(&p, nq).expect("fquery");
        assert_eq!(fq.d, ds.d, "fquery dim != index dim"); assert!(fq.nb >= nq, "fquery has fewer rows than nq");
        let mut v = vec![0f32; nq * ds.d];
        for i in 0..nq { v[i * ds.d..i * ds.d + ds.d].copy_from_slice(fq.row(i)); }
        v
    } else { Vec::new() };
    // CELL-PORTAL loose-recall path: base-only spherical subclusters inside each IVF cell. The selected
    // buckets replace the PQ scan pool and seed the unchanged best-first graph continuation.
    if let Ok(path) = std::env::var("SBANN_PORTAL_FILE") {
        use std::sync::atomic::Ordering::Relaxed;
        let portals = vq::CellPortals::load(&path).expect("load portal sidecar");
        assert_eq!(portals.n, ds.nb, "portal/base row count mismatch");
        assert_eq!(portals.d, ds.d, "portal/base dimension mismatch");
        assert!(
            portals.nc >= idx.router.n_cells(),
            "portal sidecar has fewer cells than the loaded index"
        );
        if let Ok(v) = std::env::var("SBANN_PORTAL_KEEP") {
            vq::PORTAL_KEEP.store(v.parse().expect("SBANN_PORTAL_KEEP"), Relaxed);
        }
        println!(
            "  [CELL-PORTALS] {path} cells={} P={} keep={}",
            portals.nc,
            portals.p,
            vq::PORTAL_KEEP.load(Relaxed)
        );
        let _ = vq::CELL_PORTALS.set(portals);
    }
    // GRAPH-AUGMENTED POOL EXPANSION (SBANN_GRAPH_FILE, temporary A/B sidecar): a raw little-endian u32
    // n*k IP-kNN adjacency (no header). Enables the graph union rescore on the FLOAT_RERANK cascade path
    // (batched + per-query). k is inferred from the file size; M/kedge/pfdist come from env (defaults set
    // in vq). Orthogonal to the index -> no serialization change (fold into the index once the lever lands).
    let graph: Option<vq::GraphAdj> = if let Ok(gp) = std::env::var("SBANN_GRAPH_FILE") {
        use std::sync::atomic::Ordering::Relaxed;
        let n = ds.nb;
        let flen = std::fs::metadata(&gp).expect("graph file stat").len() as usize;
        assert!(flen % (n * 4) == 0, "graph file {gp} size {flen} not divisible by n*4 ({})", n * 4);
        let k = flen / (n * 4);
        let g = vq::GraphAdj::load(&gp, n, k).expect("load graph sidecar");
        if let Ok(v) = std::env::var("SBANN_GRAPH_M") {
            vq::GRAPH_M.store(v.parse().expect("SBANN_GRAPH_M"), Relaxed);
        } else {
            vq::GRAPH_M.store(search_preset.graph_m(), Relaxed);
        }
        if let Ok(v) = std::env::var("SBANN_GRAPH_HOPS") {
            vq::GRAPH_HOPS.store(v.parse().expect("SBANN_GRAPH_HOPS"), Relaxed);
        } else {
            vq::GRAPH_HOPS.store(search_preset.graph_hops(), Relaxed);
        }
        let bestfirst = if std::env::var("SBANN_GRAPH_BESTFIRST").is_ok() {
            env_on("SBANN_GRAPH_BESTFIRST", false)
        } else {
            // The measured best-first gain is in-distribution/high-recall. L2 is a useful safe
            // default signal; IP/cosine may be either in-distribution or OOD, so it stays off there.
            search_preset != SearchPreset::Fast && !vq::IP_MODE.load(Relaxed)
        };
        vq::GRAPH_BESTFIRST.store(bestfirst, Relaxed);
        if let Ok(v) = std::env::var("SBANN_GRAPH_KEDGE") {
            vq::GRAPH_KEDGE.store(v.parse().expect("SBANN_GRAPH_KEDGE"), Relaxed);
        } else {
            vq::GRAPH_KEDGE.store(k.min(32), Relaxed);
        }
        // ADAPT-STOP (P342): per-query adaptive beam termination; value = i32 margin on the int8 bound.
        if let Ok(v) = std::env::var("SBANN_ADAPT_STOP") {
            vq::ADAPT_STOP_ON.store(true, Relaxed);
            vq::ADAPT_STOP_MARGIN.store(v.parse().unwrap_or(0), Relaxed);
        }
        // ADAPT-POOL (P342b): patience C for adaptive rescore depth over apq4-ranked pool rows.
        if let Ok(v) = std::env::var("SBANN_ADAPT_POOL") {
            vq::ADAPT_POOL.store(v.parse().expect("SBANN_ADAPT_POOL"), Relaxed);
        }
        // SQ4 int8 escalation width (P343c).
        if let Ok(v) = std::env::var("SBANN_SQ4_INT8K") {
            vq::SQ4_INT8K.store(v.parse().expect("SBANN_SQ4_INT8K"), Relaxed);
        }
        if let Ok(v) = std::env::var("SBANN_GRAPH_PFDIST") { vq::GRAPH_PFDIST.store(v.parse().expect("SBANN_GRAPH_PFDIST"), Relaxed); }
        if let Ok(v) = std::env::var("SBANN_GRAPH_SORT") { vq::GRAPH_SORT.store(v != "0", Relaxed); }
        if let Ok(v) = std::env::var("SBANN_SPLIT_RESCORE") { vq::SPLIT_RESCORE.store(v != "0", Relaxed); }
        // QSEED per-query seed table (SBANN_SEED_IDS_FILE): header <nq:u32,S:u32> then nq*S u32 seed base-ids.
        if let Ok(sf) = std::env::var("SBANN_SEED_IDS_FILE") {
            let bytes = std::fs::read(&sf).expect("seed ids file");
            let snq = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
            let s = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
            let data: Vec<u32> = bytes[8..8 + snq * s * 4].chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            println!("  [QSEED] {sf}  nq={snq} S={s}");
            let _ = vq::SEED_IDS.set((s, data));
        }
        // RBQ-TIER (SBANN_RBQ_NAV): build the resident 1-bit sign store (bit i = base coord i >= 0), by orig.
        if std::env::var("SBANN_RBQ_NAV").is_ok() {
            let dd = ds.d; let bpr = (dd + 7) / 8;
            // RBQ-TIER: random orthonormal rotation P; store signs of P*x per base row (decorrelated 1-bit codes).
            let p = vq::random_orthogonal(dd, 0x5ba4_u64);
            let mut signs = vec![0u8; n * bpr];
            signs.par_chunks_mut(bpr).enumerate().for_each(|(o, out)| {
                let row = ds.row(o);
                for j in 0..dd {
                    let prow = &p[j * dd..j * dd + dd];
                    let mut s = 0f32;
                    for k in 0..dd { s += prow[k] * row[k] as f32; }
                    if s >= 0.0 { out[j >> 3] |= 1u8 << (j & 7); }
                }
            });
            println!("  [RBQ-NAV] rotated signs nb={n} bpr={bpr}");
            let _ = vq::RBQ_ROT.set(p);
            let _ = vq::RBQ_SIGNS.set((bpr, signs));
        }
        // SQ4-RUNG (P343, flag-gated): nibble-packed 4-bit truncation of the int8 rows (d/2 B/row = half
        // the cache lines). Union nav + selection score via dot_sq4_vnni; int8 stage skipped; float
        // rerank fixes the tail. Gates: wiki 1.0000@64 (engine-exact 0.9998), webvid 0.9990@300.
        if std::env::var("SBANN_SQ4_NAV").is_ok() {
            assert!(std::is_x86_feature_detected!("avx512vnni"), "SQ4-NAV needs AVX-512 VNNI");
            assert!(ds.d % 2 == 0, "SQ4-NAV wants even d");
            let t0 = Instant::now();
            let hb = ds.d / 2;
            // PER-DIM robust affine range (p0.5..p99.5 per dim, P343b): dims are heterogeneous (WebVid:
            // a global range clips outlier dims, -3-4pt recall). Per-dim steps are folded into the QUERY
            // at query time (vq: qs[j] = q[j]*step[j] requantized i8), so the row score stays Σ nibble·qs.
            let dd = ds.d;
            let mstep = (n / 200_000).max(1);
            let mut hist = vec![0u32; dd * 256];
            let mut i = 0usize;
            let mut nsamp = 0u32;
            while i < n {
                let row = ds.row(i);
                for j in 0..dd { hist[j * 256 + (row[j] as i16 + 128) as usize] += 1; }
                nsamp += 1;
                i += mstep;
            }
            let cut = (nsamp / 200).max(1);
            let mut lo = vec![0f32; dd];
            let mut step = vec![0f32; dd];
            for j in 0..dd {
                let h = &hist[j * 256..(j + 1) * 256];
                let (mut l, mut r, mut acc) = (-128i32, 127i32, 0u32);
                for b in 0..256 { acc += h[b]; if acc >= cut { l = b as i32 - 128; break; } }
                acc = 0;
                for b in (0..256).rev() { acc += h[b]; if acc >= cut { r = b as i32 - 128; break; } }
                let r = r.max(l + 1);
                lo[j] = l as f32;
                step[j] = (r - l) as f32 / 15.0;
            }
            // SBANN_SQ4_FILE: cache the encoded sidecar (header d:u32 pad:u32 n:u64, steps d*f32, codes n*hb).
            // Steps are recomputed above (cheap, sample-only) and VALIDATED against the cached ones so a
            // stale sidecar from a different base fails loudly instead of poisoning a sweep.
            let sq4_file = std::env::var("SBANN_SQ4_FILE").ok();
            let want = 16 + dd * 4 + n * hb;
            let cached: Option<Vec<u8>> = sq4_file.as_ref().and_then(|p| std::fs::read(p).ok()).and_then(|bytes| {
                if bytes.len() != want { return None; }
                if u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize != dd { return None; }
                if u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize != n { return None; }
                for j in 0..dd {
                    let o = 16 + j * 4;
                    let s = f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
                    if (s - step[j]).abs() > 1e-4 { return None; }
                }
                Some(bytes[16 + dd * 4..].to_vec())
            });
            let codes = if let Some(c) = cached { println!("  [SQ4] sidecar cache hit"); c } else {
                let mut codes = vec![0u8; n * hb];
                codes.par_chunks_mut(hb).enumerate().for_each(|(o, out)| {
                    let row = ds.row(o);
                    for j in 0..hb {
                        let ne = ((row[2 * j] as f32 - lo[2 * j]) / step[2 * j]).round().clamp(0.0, 15.0) as u8;
                        let no = ((row[2 * j + 1] as f32 - lo[2 * j + 1]) / step[2 * j + 1]).round().clamp(0.0, 15.0) as u8;
                        out[j] = ne | (no << 4);
                    }
                });
                if let Some(p) = sq4_file.as_ref() {
                    let mut bytes = Vec::with_capacity(want);
                    bytes.extend_from_slice(&(dd as u32).to_le_bytes());
                    bytes.extend_from_slice(&0u32.to_le_bytes());
                    bytes.extend_from_slice(&(n as u64).to_le_bytes());
                    for s in step.iter() { bytes.extend_from_slice(&s.to_le_bytes()); }
                    bytes.extend_from_slice(&codes);
                    std::fs::write(p, &bytes).expect("SQ4 sidecar write");
                }
                codes
            };
            println!("  [SQ4-NAV] nibble sidecar {hb}B/row (int8 {}B) nb={n} per-dim steps  setup={:.1}s", dd, t0.elapsed().as_secs_f64());
            let _ = vq::SQ4_STEP.set(step);
            let _ = vq::SQ4.set(codes);
        }
        // SYMPACK-B (P344, flag-gated): pack each node's ke neighbors' ids + SQ4 nibble codes into one
        // contiguous block (ke*4 + ke*d/2 bytes) so a walk expansion is ONE sequential read (17ns/cand
        // measured) instead of ke scattered gathers (94ns/row). Requires SBANN_SQ4_NAV (flat codes are
        // the entry scorer + code source). SymphonyQG-style layout — cited transplant.
        if std::env::var("SBANN_SYMPACK").is_ok() {
            let sq4codes = vq::SQ4.get().expect("SBANN_SYMPACK requires SBANN_SQ4_NAV");
            let t0 = Instant::now();
            let hb = ds.d / 2;
            let ke = vq::GRAPH_KEDGE.load(Relaxed).clamp(1, k).min(64);
            let blk = ke * 4 + ke * hb;
            let mut blocks = vec![0u8; n * blk];
            blocks.par_chunks_mut(blk).enumerate().for_each(|(o, out)| {
                let nbs = &g.adj[o * k..o * k + ke];
                for (j, &nb) in nbs.iter().enumerate() {
                    out[j * 4..j * 4 + 4].copy_from_slice(&nb.to_le_bytes());
                }
                let cb = ke * 4;
                for (j, &nb) in nbs.iter().enumerate() {
                    if (nb as usize) >= n { continue; } // padded/sentinel graph ids: leave zero codes
                    let s = nb as usize * hb;
                    out[cb + j * hb..cb + (j + 1) * hb].copy_from_slice(&sq4codes[s..s + hb]);
                }
            });
            println!("  [SYMPACK] blocks ke={ke} blk={blk}B total={}MB  setup={:.1}s", n * blk / 1_000_000, t0.elapsed().as_secs_f64());
            let _ = vq::SYMPACK.set((ke, blk, blocks));
        }
        // PQ4-NAV (P341, flag-gated): plain 4-bit PQ sidecar (own codebook, independent of the index's
        // residual apq4 — this is what gate 1 validated as a LOWER bound). Nav + float-survivor selection
        // run on m=d/2 code bytes/row (~1 cache line) and the int8 rescore stage is skipped entirely.
        // SBANN_PQ4_FILE caches the trained sidecar across sweeps.
        if std::env::var("SBANN_PQ4_NAV").is_ok() {
            let t0 = Instant::now();
            let dd = ds.d;
            assert!(dd % 2 == 0, "PQ4-NAV wants even d");
            let m = dd / 2;
            let dsub = 2usize;
            let sidecar = std::env::var("SBANN_PQ4_FILE").ok();
            let want = 16 + m * 16 * dsub * 4 + n * m;
            let loaded: Option<(Vec<f32>, Vec<u8>)> = sidecar.as_ref().and_then(|p| std::fs::read(p).ok()).and_then(|bytes| {
                if bytes.len() != want { return None; }
                let hm = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
                let hn = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
                if hm != m || hn != n { return None; }
                let mut cent = vec![0f32; m * 16 * dsub];
                for (i, c) in cent.iter_mut().enumerate() {
                    let o = 16 + i * 4;
                    *c = f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
                }
                Some((cent, bytes[16 + m * 16 * dsub * 4..].to_vec()))
            });
            let (cent, codes) = if let Some(x) = loaded { println!("  [PQ4] sidecar loaded"); x } else {
                // train: 16-centroid k-means per 2-dim sub on <=200k stride-sampled rows (Lloyd x8)
                let ns = 200_000usize.min(n);
                let step = (n / ns).max(1);
                let mut cent = vec![0f32; m * 16 * dsub];
                cent.par_chunks_mut(16 * dsub).enumerate().for_each(|(mi, cm)| {
                    let mut xs = vec![0f32; ns * 2];
                    for s in 0..ns {
                        let row = ds.row(s * step);
                        xs[s * 2] = row[mi * 2] as f32;
                        xs[s * 2 + 1] = row[mi * 2 + 1] as f32;
                    }
                    for c in 0..16 { // spread init over the sample
                        let s = c * ns / 16;
                        cm[c * 2] = xs[s * 2]; cm[c * 2 + 1] = xs[s * 2 + 1];
                    }
                    let mut asg = vec![0u8; ns];
                    for _ in 0..8 {
                        for s in 0..ns {
                            let (x0, x1) = (xs[s * 2], xs[s * 2 + 1]);
                            let (mut bc, mut bd) = (0u8, f32::INFINITY);
                            for c in 0..16 {
                                let d0 = x0 - cm[c * 2]; let d1 = x1 - cm[c * 2 + 1];
                                let dv = d0 * d0 + d1 * d1;
                                if dv < bd { bd = dv; bc = c as u8; }
                            }
                            asg[s] = bc;
                        }
                        let mut sum = [[0f64; 2]; 16]; let mut cnt = [0usize; 16];
                        for s in 0..ns {
                            let c = asg[s] as usize;
                            sum[c][0] += xs[s * 2] as f64; sum[c][1] += xs[s * 2 + 1] as f64; cnt[c] += 1;
                        }
                        for c in 0..16 {
                            if cnt[c] > 0 {
                                cm[c * 2] = (sum[c][0] / cnt[c] as f64) as f32;
                                cm[c * 2 + 1] = (sum[c][1] / cnt[c] as f64) as f32;
                            }
                        }
                    }
                });
                let mut codes = vec![0u8; n * m];
                codes.par_chunks_mut(m).enumerate().for_each(|(o, out)| {
                    let row = ds.row(o);
                    for mi in 0..m {
                        let x0 = row[mi * 2] as f32; let x1 = row[mi * 2 + 1] as f32;
                        let cm = &cent[mi * 16 * 2..(mi + 1) * 16 * 2];
                        let (mut bc, mut bd) = (0u8, f32::INFINITY);
                        for c in 0..16 {
                            let d0 = x0 - cm[c * 2]; let d1 = x1 - cm[c * 2 + 1];
                            let dv = d0 * d0 + d1 * d1;
                            if dv < bd { bd = dv; bc = c as u8; }
                        }
                        out[mi] = bc;
                    }
                });
                if let Some(p) = sidecar.as_ref() {
                    let mut bytes = Vec::with_capacity(want);
                    bytes.extend_from_slice(&(m as u32).to_le_bytes());
                    bytes.extend_from_slice(&(dsub as u32).to_le_bytes());
                    bytes.extend_from_slice(&(n as u64).to_le_bytes());
                    for c in cent.iter() { bytes.extend_from_slice(&c.to_le_bytes()); }
                    bytes.extend_from_slice(&codes);
                    std::fs::write(p, &bytes).expect("PQ4 sidecar write");
                }
                (cent, codes)
            };
            println!("  [PQ4-NAV] m={m} ({}B/row vs int8 {}B) nb={n}  setup={:.1}s", m, dd, t0.elapsed().as_secs_f64());
            let _ = vq::PQ4.set(vq::Pq4Nav { m, dsub, cent, codes });
        }
        println!(
            "  [GRAPH] {gp}  n={n} k={k}  hops={} M={} kedge={} bestfirst={} pfdist={}",
            vq::GRAPH_HOPS.load(Relaxed),
            vq::GRAPH_M.load(Relaxed),
            vq::GRAPH_KEDGE.load(Relaxed).min(k),
            vq::GRAPH_BESTFIRST.load(Relaxed),
            vq::GRAPH_PFDIST.load(Relaxed)
        );
        Some(g)
    } else { None };
    let graph_ref = graph.as_ref();
    // avq cell count is cb^2 == c; keep probes well under nc
    // SBANN_PLIST="128,256,512" overrides the preset-centered default sweep.
    let plist: Vec<usize> = match std::env::var("SBANN_PLIST") {
        Ok(s) => s.split(',').filter_map(|x| x.trim().parse().ok()).filter(|&x: &usize| x >= 1).collect(),
        Err(_) => default_probe_ladder(search_preset, ds.d, idx.router.n_cells()),
    };
    let tfloor: usize = std::env::var("SBANN_TFLOOR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| search_preset.survivor_floor(ds.nb, ds.d));
    println!(
        "  [SEARCH-PRESET] {} via {}  probes={:?} tfloor={} cascade_k={}{}",
        search_preset.name(),
        preset_source,
        plist,
        tfloor,
        vq::CASCADE_K.load(std::sync::atomic::Ordering::Relaxed),
        if graph_ref.is_some() { " graph=on" } else { " graph=off" }
    );
    // contiguous query array (for batched GEMM routing)
    let mut qarr = vec![0i8; nq * ds.d];
    for i in 0..nq { qarr[i * ds.d..i * ds.d + ds.d].copy_from_slice(qs.row(i)); }
    if batched { println!("  (batched GEMM routing)"); }
    // SBANN_TMUL="10,20,30" sweeps the rerank-survivor multiplier within one build (reuse the index).
    let tlist: Vec<usize> = match std::env::var("SBANN_TMUL") {
        Ok(s) => s.split(',').filter_map(|x| x.trim().parse().ok()).filter(|&x: &usize| x >= 1).collect(),
        Err(_) => vec![tmul],
    };
    // SBANN_KLIST="64,96,128" sweeps the int8-cascade prune width K within one index (P194). Only used
    // when SBANN_CASCADE is set; otherwise a single no-op pass at the current CASCADE_K.
    let klist: Vec<usize> = match std::env::var("SBANN_KLIST") {
        Ok(s) => s.split(',').filter_map(|x| x.trim().parse().ok()).filter(|&x: &usize| x >= 1).collect(),
        Err(_) => vec![vq::CASCADE_K.load(std::sync::atomic::Ordering::Relaxed)],
    };
    // big-ann reports BEST search time over run_count -> measure best-of-REPS to filter box-load
    // spikes on this contended box. SBANN_REPS overrides (default 1; use 3-5 for clean A/B tuning).
    let reps: usize = std::env::var("SBANN_REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    // BATCHSCAN (P202, champion default ON): cell-major batched scan — route+LUT all nq queries, sweep
    // cells in storage order reusing each cell's blocks across the queries that probe it, then per-query
    // cascade+float. Recall-BIT-IDENTICAL to the per-query path (pure execution-order change). Only wired
    // for the FLOAT_RERANK path. SBANN_BATCHSCAN=0 forces the per-query loop. SBANN_BATCH_CHUNK splits the
    // nq queries into fixed-size chunks (multiplicity ~ chunk_size).
    // DEFAULT (P245/P246): adaptive clamp(nq/(4*threads), 125, 1000). The old fixed 1000 (P202 knee) had two
    // parallel pathologies: nq<=1000 -> ONE chunk -> the whole batched path runs SERIAL (cohere-10M 8t stuck at
    // 1x, P245), and nq=10000/8t -> 10 chunks -> straggler imbalance (chunk=250: +43% at 8t, recall
    // bit-identical, P246). 4 chunks/thread balances; floor 125 keeps cell-pass amortization; 1t unaffected
    // (clamp hits 1000). Env SBANN_BATCH_CHUNK still overrides.
    let batchscan = env_on("SBANN_BATCHSCAN", true);
    let batch_chunk: usize = std::env::var("SBANN_BATCH_CHUNK").ok().and_then(|s| s.parse().ok()).filter(|&c| c >= 1)
        .unwrap_or_else(|| (nq / (4 * rayon::current_num_threads()).max(1)).clamp(125, 1000));
    let batch_verify = std::env::var("SBANN_BATCH_VERIFY").is_ok();
    // SBANN_VNNI_AB: interleave VNNI off/on per (p,t) for a clean same-index rerank-kernel A/B.
    let vnni_ab = std::env::var("SBANN_VNNI_AB").is_ok();
    let modes: Vec<bool> = if vnni_ab { vec![false, true] } else { vec![crate::simd::VNNI_ON.load(std::sync::atomic::Ordering::Relaxed)] };
    // SBANN_LUT_AB: interleave int16 (false) vs i8 (true) scan precision per (p,t) on one index.
    let lut_ab = std::env::var("SBANN_LUT_AB").is_ok();
    let lmodes: Vec<bool> = if lut_ab { vec![false, true] } else { vec![vq::LUT16_OFF.load(std::sync::atomic::Ordering::Relaxed)] };
    // IDEA #4 refine sweep: with SBANN_RESID, sweep (refine off/on) x rr_depth (=raw-rerank depth)
    // at a FIXED refine pool t_surv, on ONE built index. Reports recall vs raw reads for both, so the
    // refined order's depth saving (same recall, fewer raw reads) is a clean same-index A/B.
    if vq::RESID.load(std::sync::atomic::Ordering::Relaxed) {
        let mr = idx.resid_pq.as_ref().map(|p| p.m).unwrap_or(0);
        let tmul0 = *tlist.first().unwrap_or(&tmul);
        for &p in &plist {
            let t_surv: usize = std::env::var("SBANN_TSURV").ok().and_then(|s| s.parse().ok())
                .unwrap_or((p * tmul0).max(*plist.iter().max().unwrap_or(&p) * tmul0).max(2000));
            let rrlist: Vec<usize> = match std::env::var("SBANN_RRLIST") {
                Ok(s) => s.split(',').filter_map(|x| x.trim().parse().ok()).collect(),
                Err(_) => vec![10, 25, 50, 100, 200, 400, 800, t_surv],
            };
            println!("  [RESID p={p} t_surv={t_surv} m_r={mr}B/vec raw={}B/vec]", ds.d);
            for refine in [false, true] {
                for &rr in &rrlist {
                    let rr = rr.min(t_surv);
                    let mut best_dt = f64::INFINITY;
                    let mut res: Vec<Vec<u32>> = Vec::new();
                    for _ in 0..reps.max(1) {
                        let st = Instant::now();
                        let r: Vec<Vec<u32>> = (0..nq).into_par_iter().map(|i| {
                            let cells = idx.router.probe(qs.row(i), p);
                            idx.scan_rerank_resid(&ds, qs.row(i), &cells, t_surv, rr, refine, 10)
                        }).collect();
                        best_dt = best_dt.min(st.elapsed().as_secs_f64());
                        res = r;
                    }
                    let mut hit = 0usize;
                    for i in 0..nq {
                        let truth: std::collections::HashSet<u32> = gids[i * gk..i * gk + 10].iter().copied().collect();
                        hit += res[i].iter().take(10).filter(|id| truth.contains(id)).count();
                    }
                    // raw-read-equiv bytes/query: refine path also reads t_surv*m_r refine bytes.
                    let raw_bytes = rr * ds.d + if refine { t_surv * mr } else { 0 };
                    let tag = if refine { "refine" } else { "plain " };
                    println!("    {tag} rr={rr:5}: recall@10={:.4}  raw_reads={rr:5}  bytes/q={raw_bytes:8}  QPS={:.0}",
                        hit as f64 / (nq * 10) as f64, nq as f64 / best_dt);
                }
            }
        }
        return;
    }
    // ROAR-MODE (P340, flag-gated — champion path untouched): pure best-first walk sweep. No route, no
    // scan, no union rescore; the walk's top-L is the candidate set and cost adapts per query (DiskANN
    // termination). Sim gates (2026-07-20): int8 -dot walk == float-walk quality; warm entry -25-30% evals;
    // walk ceiling ~0.96 complements the cascade's >=0.95 win regime. SBANN_ROARMODE="30,50,80,120" (L
    // sweep). Entry per query: QSEED seeds if SBANN_SEED_IDS_FILE is loaded (first SBANN_ROAR_ENTRIES of
    // them), else nearest of an SBANN_ROAR_DIR-row stride-sampled directory (contiguous int8 scan), else
    // the global medoid (argmax dot with the int8 mean — unit-norm/IP datasets; the loss regimes are such).
    if let Ok(rl) = std::env::var("SBANN_ROARMODE") {
        use std::sync::atomic::Ordering::Relaxed;
        let llist: Vec<usize> = rl.split(',').filter_map(|x| x.trim().parse().ok()).filter(|&x: &usize| x >= 10).collect();
        let graph = graph_ref.expect("SBANN_ROARMODE needs SBANN_GRAPH_FILE");
        let fb = fbase.as_ref().expect("SBANN_ROARMODE needs SBANN_FLOAT_RERANK + SBANN_FBASE/FQUERY");
        let n_entries: usize = std::env::var("SBANN_ROAR_ENTRIES").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
        let t0 = Instant::now();
        let mut mean = vec![0f32; ds.d];
        let mstep = (ds.nb / 1_000_000).max(1); // <=1M-row sample fixes the medoid plenty
        let mut msamp = 0usize;
        let mut i = 0usize;
        while i < ds.nb { let r = ds.row(i); for j in 0..ds.d { mean[j] += r[j] as f32; } msamp += 1; i += mstep; }
        for v in mean.iter_mut() { *v /= msamp as f32; }
        let medoid = (0..ds.nb).into_par_iter().map(|o| {
            let r = ds.row(o);
            let mut s = 0f32; for j in 0..ds.d { s += mean[j] * r[j] as f32; }
            (s, o as u32)
        }).reduce(|| (f32::NEG_INFINITY, 0u32), |a, b| if b.0 > a.0 { b } else { a }).1;
        // warm-entry directory: stride-sampled rows copied contiguous (sequential VNNI scan per query).
        let dirn: usize = std::env::var("SBANN_ROAR_DIR").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
        let (dirids, dirbuf): (Vec<u32>, Vec<i8>) = if dirn > 0 {
            let ids: Vec<u32> = (0..dirn).map(|j| ((j as u64 * ds.nb as u64 / dirn as u64) as u32)).collect();
            let mut buf = vec![0i8; dirn * ds.d];
            for (j, &o) in ids.iter().enumerate() { buf[j * ds.d..(j + 1) * ds.d].copy_from_slice(ds.row(o as usize)); }
            (ids, buf)
        } else { (Vec::new(), Vec::new()) };
        let seeds = vq::SEED_IDS.get();
        let etag = if seeds.is_some() { "qseed" } else if dirn > 0 { "dir" } else { "medoid" };
        println!("  [ROARMODE] graph k={} kedge={} entries={etag}(x{n_entries}) dir={dirn} medoid={medoid} setup={:.1}s",
            graph.k, vq::GRAPH_KEDGE.load(Relaxed).min(graph.k), t0.elapsed().as_secs_f64());
        let ipm = vq::IP_MODE.load(Relaxed);
        for &l in &llist {
            vq::ROAR_EVALS.store(0, Relaxed);
            vq::ROAR_HOPS.store(0, Relaxed);
            let mut best_dt = f64::INFINITY;
            let mut res: Vec<Vec<u32>> = Vec::new();
            for _ in 0..reps.max(1) {
                let st = Instant::now();
                let r: Vec<Vec<u32>> = (0..nq).into_par_iter().map(|i| {
                    let mut ebuf = [0u32; 16];
                    let ne;
                    if let Some((s, data)) = seeds.filter(|(s, data)| *s > 0 && (i + 1) * *s <= data.len()) {
                        ne = n_entries.clamp(1, 16.min(*s));
                        ebuf[..ne].copy_from_slice(&data[i * s..i * s + ne]);
                    } else if dirn > 0 {
                        // nearest directory row(s) by -dot over the contiguous buffer
                        ne = n_entries.clamp(1, 16);
                        let mut top: Vec<(i32, u32)> = (0..dirn).map(|j| {
                            (simd::negdot_i8(qs.row(i), &dirbuf[j * ds.d..(j + 1) * ds.d]), dirids[j])
                        }).collect();
                        if ne < top.len() { top.select_nth_unstable(ne - 1); }
                        for (k, &(_, o)) in top[..ne].iter().enumerate() { ebuf[k] = o; }
                    } else {
                        ne = 1; ebuf[0] = medoid;
                    }
                    let mut walk = if vq::SYMPACK.get().is_some() {
                        // SYMPACK: SQ4-guided block walk -> int8 re-rank of the top-L -> top-kk to float
                        let w = vq::sympack_walk(&ds, qs.row(i), l, &ebuf[..ne]);
                        let mut w: Vec<(i32, u32)> = w.iter().map(|&(_, o)| {
                            (simd::negdot_i8(qs.row(i), ds.row(o as usize)), o)
                        }).collect();
                        w.sort_unstable();
                        w
                    } else {
                        vq::roar_walk(&ds, graph, qs.row(i), l, &ebuf[..ne])
                    };
                    let kkw = vq::CASCADE_K.load(std::sync::atomic::Ordering::Relaxed);
                    if vq::SYMPACK.get().is_some() && kkw > 0 && kkw < walk.len() { walk.truncate(kkw); }
                    // float rerank of the walk's top-L (exact tail ordering, CASCADE-style)
                    let qv = &fqf[i * ds.d..(i + 1) * ds.d];
                    let mut scored: Vec<(f32, u32)> = walk.iter().map(|&(_, o)| {
                        let r = fb.row(o as usize);
                        let s = if ipm {
                            let mut acc = 0f32; for j in 0..ds.d { acc += qv[j] * r[j]; } -acc
                        } else {
                            let mut acc = 0f32; for j in 0..ds.d { let t = qv[j] - r[j]; acc += t * t; } acc
                        };
                        (s, o)
                    }).collect();
                    scored.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                    scored.iter().take(10).map(|&(_, o)| o).collect()
                }).collect();
                best_dt = best_dt.min(st.elapsed().as_secs_f64());
                res = r;
            }
            let mut hit = 0usize;
            for i in 0..nq {
                let truth: std::collections::HashSet<u32> = gids[i * gk..i * gk + 10].iter().copied().collect();
                hit += res[i].iter().take(10).filter(|id| truth.contains(id)).count();
            }
            let nrun = (nq * reps.max(1)) as u64;
            println!("  RW L={l:4} e={etag}: recall@10={:.4}  QPS={:.0} (best/{reps})  evals/q={}  hops/q={}",
                hit as f64 / (nq * 10) as f64, nq as f64 / best_dt,
                vq::ROAR_EVALS.load(Relaxed) / nrun, vq::ROAR_HOPS.load(Relaxed) / nrun);
        }
        return;
    }
    for &p in &plist {
      for &tm in &tlist {
       for &lm in &lmodes {
        vq::LUT16_OFF.store(lm, std::sync::atomic::Ordering::Relaxed);
       for &vm in &modes {
        crate::simd::VNNI_ON.store(vm, std::sync::atomic::Ordering::Relaxed);
       for &kk in &klist {
        vq::CASCADE_K.store(kk, std::sync::atomic::Ordering::Relaxed);
        // survivors kept for exact rerank (tmul tunes recall/speed). The rerank floor was 1000 but that
        // was a ~2x QPS@90% HANDICAP: int16 LUT ranks well enough that t_surv=p*tmul (~256-480) holds
        // recall (P111). The preset supplies the scale/dimension-aware floor; SBANN_TFLOOR overrides.
        let t_surv = (p * tm).max(tfloor);
        let mut best_dt = f64::INFINITY;
        let mut res: Vec<Vec<u32>> = Vec::new();
        let prof = std::env::var("SBANN_PROFILE").is_ok();
        if prof {
            vq::PROF_ROUTE_NS.store(0, std::sync::atomic::Ordering::Relaxed);
            vq::PROF_SCAN_NS.store(0, std::sync::atomic::Ordering::Relaxed);
            vq::PROF_RERANK_NS.store(0, std::sync::atomic::Ordering::Relaxed);
            vq::PROF_CASC_NS.store(0, std::sync::atomic::Ordering::Relaxed);
            vq::PROF_GRAPH_NS.store(0, std::sync::atomic::Ordering::Relaxed);
            vq::PROF_GRAPH_ROWS.store(0, std::sync::atomic::Ordering::Relaxed);
        }
        for _ in 0..reps.max(1) {
            let st = Instant::now();
            let r: Vec<Vec<u32>> = if let Some(fb) = fbase.as_ref() {
                if batchscan {
                    // cell-major batched driver, chunked for the multiplicity-sensitivity sweep.
                    // Chunks are independent query ranges -> run them rayon-parallel: batching (P202)
                    // and threading compose. RAYON=1 degenerates to the old serial loop (bit-identical;
                    // per-chunk results don't depend on execution order).
                    let ranges: Vec<(usize, usize)> = (0..nq).step_by(batch_chunk.max(1))
                        .map(|s| (s, (s + batch_chunk).min(nq))).collect();
                    let subs: Vec<Vec<Vec<u32>>> = ranges.into_par_iter()
                        .map(|(s, e)| idx.search_batch_frr(&ds, &qarr[s * ds.d..e * ds.d], &fqf[s * ds.d..e * ds.d], fb, e - s, p, t_surv, 10, graph_ref, s))
                        .collect();
                    subs.into_iter().flatten().collect()
                } else {
                    (0..nq).into_par_iter().map(|i| idx.search_frr(&ds, qs.row(i), &fqf[i * ds.d..i * ds.d + ds.d], fb, p, t_surv, 10, graph_ref)).collect()
                }
            } else if batched {
                idx.search_batch(&ds, &qarr, nq, p, t_surv, 10)
            } else {
                (0..nq).into_par_iter().map(|i| idx.search(&ds, qs.row(i), p, t_surv, 10)).collect()
            };
            best_dt = best_dt.min(st.elapsed().as_secs_f64());
            res = r;
        }
        let dt = best_dt;
        // CORRECTNESS GATE: the cell-major driver is a pure execution-order change -> per query it must
        // produce the same final top-10 as the per-query path (modulo equal-score tie order). Compare
        // result-id sets and recall for all nq queries.
        if batchscan && batch_verify {
            if let Some(fb) = fbase.as_ref() {
                let refr: Vec<Vec<u32>> = (0..nq).into_par_iter()
                    .map(|i| idx.search_frr(&ds, qs.row(i), &fqf[i * ds.d..i * ds.d + ds.d], fb, p, t_surv, 10, graph_ref)).collect();
                let mut set_id = 0usize;   // queries whose top-10 id SET is identical
                let mut exact = 0usize;    // queries whose top-10 id LIST is identical (order too)
                let mut ref_hit = 0usize; let mut bat_hit = 0usize;
                for i in 0..nq {
                    let a: std::collections::HashSet<u32> = res[i].iter().take(10).copied().collect();
                    let b: std::collections::HashSet<u32> = refr[i].iter().take(10).copied().collect();
                    if a == b { set_id += 1; }
                    if res[i].iter().take(10).eq(refr[i].iter().take(10)) { exact += 1; }
                    let truth: std::collections::HashSet<u32> = gids[i * gk..i * gk + 10].iter().copied().collect();
                    bat_hit += res[i].iter().take(10).filter(|id| truth.contains(id)).count();
                    ref_hit += refr[i].iter().take(10).filter(|id| truth.contains(id)).count();
                }
                println!("  [VERIFY p={p} t_surv={t_surv}] set-identical {set_id}/{nq}  order-identical {exact}/{nq}  recall batched={:.4} perquery={:.4}",
                    bat_hit as f64 / (nq * 10) as f64, ref_hit as f64 / (nq * 10) as f64);
            }
        }
        // SBANN_RESULT_DUMP=<path> (verification hook): write the nq x 10 final result ids (flat u32 LE,
        // padded with u32::MAX) so an offline oracle can cross-check the engine's top-10 per query.
        if let Ok(rp) = std::env::var("SBANN_RESULT_DUMP") {
            use std::io::Write;
            let mut w = std::io::BufWriter::new(std::fs::File::create(&rp).expect("result dump"));
            w.write_all(&(nq as u32).to_le_bytes()).unwrap();
            for i in 0..nq {
                for j in 0..10 {
                    let id = res[i].get(j).copied().unwrap_or(u32::MAX);
                    w.write_all(&id.to_le_bytes()).unwrap();
                }
            }
            println!("  [RESULT_DUMP] {rp}  ({nq} x 10 ids)");
        }
        let mut hit = 0usize;
        for i in 0..nq {
            let truth: std::collections::HashSet<u32> = gids[i * gk..i * gk + 10].iter().copied().collect();
            hit += res[i].iter().take(10).filter(|id| truth.contains(id)).count();
        }
        let vtag = if vnni_ab { if vm { " VNNI" } else { " AVX2" } } else { "" };
        let ltag = if lut_ab { if lm { " i8" } else { " i16" } } else { "" };
        let ktag = if vq::CASCADE.load(std::sync::atomic::Ordering::Relaxed) { format!(" K={kk}") } else { String::new() };
        println!("  p={p:5} t={tm:3}{ltag}{vtag}{ktag}: recall@10={:.4}  QPS={:.0} (best/{reps})", hit as f64 / (nq * 10) as f64, nq as f64 / dt);
        if vq::ADAPT_STOP_ON.load(std::sync::atomic::Ordering::Relaxed) {
            let stops = vq::PROF_ADAPT_STOPS.swap(0, std::sync::atomic::Ordering::Relaxed);
            println!("      [adapt-stop] early-stopped {}/{} query-runs", stops, nq * reps.max(1));
        }
        if vq::ADAPT_POOL.load(std::sync::atomic::Ordering::Relaxed) > 0 {
            let sk = vq::PROF_POOL_SKIPPED.swap(0, std::sync::atomic::Ordering::Relaxed);
            println!("      [adapt-pool] skipped {} pool rows/query-run", sk / (nq * reps.max(1)) as u64);
        }
        if prof {
            let r = vq::PROF_ROUTE_NS.load(std::sync::atomic::Ordering::Relaxed) as f64;
            let s = vq::PROF_SCAN_NS.load(std::sync::atomic::Ordering::Relaxed) as f64;
            let k = vq::PROF_RERANK_NS.load(std::sync::atomic::Ordering::Relaxed) as f64;
            let c = vq::PROF_CASC_NS.load(std::sync::atomic::Ordering::Relaxed) as f64;
            let g = vq::PROF_GRAPH_NS.load(std::sync::atomic::Ordering::Relaxed) as f64;
            let grows = vq::PROF_GRAPH_ROWS.load(std::sync::atomic::Ordering::Relaxed) as f64;
            let tot = (r + s + k + c + g).max(1.0);
            // cascade (PROF_CASC_NS) = the int8 union rescore; graph (PROF_GRAPH_NS) = neighbour gather +
            // union sort/dedup; rerank (PROF_RERANK_NS) = the exact float reorder of the K survivors.
            // union-rescore ns/row = PROF_CASC_NS / total union rows (the decider metric: aim ~44, not ~74).
            let nsrow = if grows > 0.0 { c / grows } else { 0.0 };
            // P342c: sco = tight row-scoring loops only (subset of casc); the casc remainder is per-hop
            // machinery (cand rebuild, exp resize, select_nth, hash inserts) — the bookkeeping suspect.
            let sco = vq::PROF_SCORE_NS.swap(0, std::sync::atomic::Ordering::Relaxed) as f64;
            println!("      [profile] route {:.1}%  scan {:.1}%  graph {:.1}%  rescore {:.1}%  float {:.1}%  (sum {:.0}ms/{reps}reps)  [scan-us/q={:.1} graph-us/q={:.2} rescore-us/q={:.1} float-us/q={:.1} union/q={:.0} rescore-ns/row={:.1} score-us/q={:.1} mach-us/q={:.1}]",
                100.0 * r / tot, 100.0 * s / tot, 100.0 * g / tot, 100.0 * c / tot, 100.0 * k / tot, (r + s + k + c + g) / 1e6,
                s / nq as f64 / reps as f64 / 1000.0, g / nq as f64 / reps as f64 / 1000.0,
                c / nq as f64 / reps as f64 / 1000.0, k / nq as f64 / reps as f64 / 1000.0,
                grows / nq as f64 / reps as f64, nsrow,
                sco / nq as f64 / reps as f64 / 1000.0, (c - sco).max(0.0) / nq as f64 / reps as f64 / 1000.0);
        }
       }
       }
       }
      }
    }
}

/// Adaptive-termination bench: fixed max_p, stop early per query. Reports recall, AVG cells
/// probed (the adaptive nprobe), and QPS — vs a fixed-p baseline at the same recall.
fn runa(base: &str, qpath: &str, gtpath: &str, router_s: &str, comp_s: &str, a0: usize, max_p: usize) {
    let ds = I8Bin::open(base).expect("base");
    let n = ds.nb;
    let sum: Vec<f64> = (0..n).into_par_iter()
        .fold(|| vec![0f64; ds.d], |mut a, i| { let r = ds.row(i); for k in 0..ds.d { a[k] += r[k] as f64; } a })
        .reduce(|| vec![0f64; ds.d], |mut a, b| { for k in 0..ds.d { a[k] += b[k]; } a });
    let mu: Vec<f32> = if std::env::var("SBANN_NOMU").is_ok() { vec![0f32; ds.d] } else { sum.iter().map(|s| (s / n as f64) as f32).collect() };
    let router: Box<dyn vq::Router> = match router_s {
        "flat" => Box::new(vq::FlatIvf::train(&ds, 4096, mu.clone(), 15)),
        "flatsoar" => Box::new(vq::FlatIvf::train_soar(&ds, 4096, mu.clone(), 15, 1.0)),
        _ => { eprintln!("router?"); return; }
    };
    let comp: Box<dyn vq::Compressor> = match comp_s {
        "pq4" => Box::new(vq::Pq4::train(&ds, 2, 6)),
        "opql" => Box::new(vq::Opq4::train_learned(&ds, 2, 6, 8)),
        "opql5" => Box::new(vq::Opq4::train_learned(&ds, 5, 6, 8)),
        _ => { eprintln!("compress?"); return; }
    };
    let idx = vq::Index::build(router, comp, &ds, a0);
    let qs = I8Bin::open(qpath).expect("q");
    let (gnq, gk, gids) = read_gt(gtpath);
    let nq = qs.nb.min(gnq);
    let rc = |res: &[Vec<u32>]| -> f64 {
        let mut hit = 0usize;
        for i in 0..nq {
            let truth: std::collections::HashSet<u32> = gids[i * gk..i * gk + 10].iter().copied().collect();
            hit += res[i].iter().take(10).filter(|id| truth.contains(id)).count();
        }
        hit as f64 / (nq * 10) as f64
    };
    println!("[{router_s}+{comp_s}] max_p={max_p}");
    // adaptive sweeps (win, patience) -> different aggressiveness
    for &(win, pat) in &[(8usize, 2usize), (16, 2), (16, 4)] {
        let st = Instant::now();
        let out: Vec<(Vec<u32>, usize)> = (0..nq).into_par_iter()
            .map(|i| idx.search_adaptive(&ds, qs.row(i), max_p, 4000, 10, win, pat)).collect();
        let dt = st.elapsed().as_secs_f64();
        let res: Vec<Vec<u32>> = out.iter().map(|(r, _)| r.clone()).collect();
        let avgp = out.iter().map(|(_, u)| u).sum::<usize>() as f64 / nq as f64;
        println!("  adapt(win={win},pat={pat}): recall@10={:.4}  avg_cells={avgp:.0}  QPS={:.0}", rc(&res), nq as f64 / dt);
    }
    // fixed-p references
    for &p in &[32usize, 64, 128] {
        let st = Instant::now();
        let res: Vec<Vec<u32>> = (0..nq).into_par_iter().map(|i| idx.search(&ds, qs.row(i), p, 4000, 10)).collect();
        let dt = st.elapsed().as_secs_f64();
        println!("  fixed(p={p}): recall@10={:.4}  QPS={:.0}", rc(&res), nq as f64 / dt);
    }
}

fn read_gt(path: &str) -> (usize, usize, Vec<u32>) {
    let bytes = std::fs::read(path).expect("read gt");
    let nq = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let k = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let ids: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&bytes[8..8 + nq * k * 4]).to_vec();
    (nq, k, ids)
}

fn bench(base: &str, qpath: &str, gtpath: &str, c: usize) {
    let t0 = Instant::now();
    let ds = I8Bin::open(base).expect("base");
    assert!(simd::selftest(ds.d), "SIMD selftest failed");
    println!("base nb={} d={}  simd=ok", ds.nb, ds.d);
    let ivf = build_ivf(&ds, c, 2);
    println!("built IVF C={c} in {:.1}s", t0.elapsed().as_secs_f64());

    let qs = I8Bin::open(qpath).expect("queries");
    let (gnq, gk, gids) = read_gt(gtpath);
    let nq = qs.nb.min(gnq);
    println!("queries nq={} gt_k={}", nq, gk);

    for &p in &[8usize, 32, 128] {
        let t = Instant::now();
        let res: Vec<Vec<u32>> = (0..nq)
            .into_par_iter()
            .map(|i| search(&ivf, &ds, qs.row(i), p, 10))
            .collect();
        let dt = t.elapsed().as_secs_f64();
        // recall@10 vs first 10 gt ids
        let mut hit = 0usize;
        for i in 0..nq {
            let truth: std::collections::HashSet<u32> =
                gids[i * gk..i * gk + 10].iter().copied().collect();
            hit += res[i].iter().take(10).filter(|id| truth.contains(id)).count();
        }
        let rec = hit as f64 / (nq * 10) as f64;
        println!("  p={p:4}: recall@10={rec:.4}  QPS={:.0}", nq as f64 / dt);
    }
}

fn build(path: &str, c: usize) {
    let t0 = Instant::now();
    let ds = I8Bin::open(path).expect("open i8bin");
    let (nb, d) = (ds.nb, ds.d);
    assert!(simd::selftest(d), "SIMD L2 kernel disagrees with scalar reference!");
    println!("base: nb={nb} d={d}  (mmap, ~0 RAM)  simd-selftest=ok");
    let _ = (t0, nb);
    let ivf = build_ivf(&ds, c, 2);
    let counts: Vec<u32> = (0..c).map(|j| ivf.cell_start[j + 1] - ivf.cell_start[j]).collect();
    let nonempty = counts.iter().filter(|&&v| v > 0).count();
    let mx = counts.iter().copied().max().unwrap_or(0);
    println!(
        "  cells: {nonempty}/{c} non-empty, max={mx} avg={:.0}  BUILD {:.1}s",
        nb as f64 / c as f64,
        t0.elapsed().as_secs_f64()
    );
}

/// Interleaved A/B: build several hierk+opql indices, then bench them ROUND-ROBIN so competing
/// configs are timed seconds apart (same box-load window) instead of ~15min apart across separate
/// builds. This is the only way to get drift-free cross-config QPS on a contended box (FINDINGS P72).
/// Configs via SBANN_CONFIGS="Kf:C0:b0,Kf:C0:b0,..."; p via SBANN_PLIST; reps via SBANN_REPS.
fn abrun(base: &str, qpath: &str, gtpath: &str) {
    let ds = I8Bin::open(base).expect("base");
    let n = ds.nb;
    let sum: Vec<f64> = (0..n).into_par_iter()
        .fold(|| vec![0f64; ds.d], |mut a, i| { let r = ds.row(i); for k in 0..ds.d { a[k] += r[k] as f64; } a })
        .reduce(|| vec![0f64; ds.d], |mut a, b| { for k in 0..ds.d { a[k] += b[k]; } a });
    let mu: Vec<f32> = if std::env::var("SBANN_NOMU").is_ok() { vec![0f32; ds.d] } else { sum.iter().map(|s| (s / n as f64) as f32).collect() };

    // config = "Kf:C0:b0[:comp[:eta]]" (comp = opql|apq4|aopq|opql5|i8; eta = anisotropy strength).
    let cfgs_s = std::env::var("SBANN_CONFIGS").unwrap_or_else(|_| "262144:1024:128,524288:2048:128".into());
    let cfgs: Vec<(usize, usize, usize, String, f32)> = cfgs_s.split(',').filter_map(|c| {
        let parts: Vec<&str> = c.split(':').collect();
        let v: Vec<usize> = parts.iter().take(3).filter_map(|x| x.trim().parse().ok()).collect();
        if v.len() == 3 {
            let comp = parts.get(3).map(|s| s.trim().to_string()).unwrap_or_else(|| "opql".into());
            let eta = parts.get(4).and_then(|s| s.trim().parse().ok()).unwrap_or(4.0f32);
            Some((v[0], v[1], v[2], comp, eta))
        } else { None }
    }).collect();
    let plist: Vec<usize> = std::env::var("SBANN_PLIST").ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![384usize, 512, 768]);
    let reps: usize = std::env::var("SBANN_REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let tmul: usize = std::env::var("SBANN_TMUL").ok().and_then(|s| s.parse().ok()).unwrap_or(30);

    // build every index up front
    let mut idxs: Vec<(String, vq::Index)> = Vec::new();
    for (kf, c0, b0, cs, eta) in &cfgs {
        let (kf, c0, b0, eta) = (*kf, *c0, *b0, *eta);
        let t = Instant::now();
        let router: Box<dyn vq::Router> = Box::new(vq::HierRouter::train_hkmeans(&ds, kf, c0, b0, mu.clone()));
        let comp: Box<dyn vq::Compressor> = match cs.as_str() {
            "apq4" => Box::new(vq::Apq4::train(&ds, 2, 6, eta)),
            "aopq" => Box::new(vq::Opq4::train_aopq(&ds, 2, 6, 8, eta)),
            "opql5" => Box::new(vq::Opq4::train_learned(&ds, 5, 6, 8)),
            "i8" => Box::new(vq::ScalarI8::new(ds.d)),
            _ => Box::new(vq::Opq4::train_learned(&ds, 2, 6, 8)),
        };
        let idx = vq::Index::build(router, comp, &ds, 2);
        let label = format!("Kf={kf} C0={c0} b0={b0} {cs}{}", if cs == "apq4" || cs == "aopq" { format!(" eta={eta}") } else { String::new() });
        println!("[built {label}] {:.1}s", t.elapsed().as_secs_f64());
        idxs.push((label, idx));
    }

    let qs = I8Bin::open(qpath).expect("q");
    let (gnq, gk, gids) = read_gt(gtpath);
    let nq_cap = std::env::var("SBANN_NQ").ok().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    let nq = qs.nb.min(gnq).min(nq_cap);

    // best[cfg][p] time, recall[cfg][p]
    let nc = idxs.len();
    let np = plist.len();
    let mut best = vec![f64::INFINITY; nc * np];
    let mut rec = vec![0f64; nc * np];
    // ROUND-ROBIN: each rep benches every (cfg,p) once -> competitors are seconds apart, same load.
    for _ in 0..reps.max(1) {
        for (ci, (_, idx)) in idxs.iter().enumerate() {
            for (pi, &p) in plist.iter().enumerate() {
                let t_surv = (p * tmul).max(1000);
                let st = Instant::now();
                let res: Vec<Vec<u32>> = (0..nq).into_par_iter()
                    .map(|i| idx.search(&ds, qs.row(i), p, t_surv, 10)).collect();
                let dt = st.elapsed().as_secs_f64();
                best[ci * np + pi] = best[ci * np + pi].min(dt);
                if rec[ci * np + pi] == 0.0 {
                    let mut hit = 0usize;
                    for i in 0..nq {
                        let truth: std::collections::HashSet<u32> = gids[i * gk..i * gk + 10].iter().copied().collect();
                        hit += res[i].iter().take(10).filter(|id| truth.contains(id)).count();
                    }
                    rec[ci * np + pi] = hit as f64 / (nq * 10) as f64;
                }
            }
        }
    }
    println!("== interleaved A/B (best/{reps}, round-robin, nq={nq}) ==");
    for (ci, (label, _)) in idxs.iter().enumerate() {
        for (pi, &p) in plist.iter().enumerate() {
            let qps = nq as f64 / best[ci * np + pi];
            println!("  [{label}] p={p:5}: recall@10={:.4}  QPS={qps:.0}", rec[ci * np + pi]);
        }
    }
}

/// Profile where query time goes (route vs scan vs rerank) for the champion config, single-threaded
/// (clean per-phase timing), at each p in SBANN_PLIST. SBANN_CONFIGS first entry = Kf:C0:b0.
fn prof(base: &str, qpath: &str, gtpath: &str) {
    let ds = I8Bin::open(base).expect("base");
    let n = ds.nb;
    let sum: Vec<f64> = (0..n).into_par_iter()
        .fold(|| vec![0f64; ds.d], |mut a, i| { let r = ds.row(i); for k in 0..ds.d { a[k] += r[k] as f64; } a })
        .reduce(|| vec![0f64; ds.d], |mut a, b| { for k in 0..ds.d { a[k] += b[k]; } a });
    let mu: Vec<f32> = if std::env::var("SBANN_NOMU").is_ok() { vec![0f32; ds.d] } else { sum.iter().map(|s| (s / n as f64) as f32).collect() };
    let cfg = std::env::var("SBANN_CONFIGS").unwrap_or_else(|_| "262144:2048:128".into());
    let v: Vec<usize> = cfg.split(',').next().unwrap().split(':').filter_map(|x| x.trim().parse().ok()).collect();
    let (kf, c0, b0) = (v[0], v[1], v[2]);
    let comp_s = std::env::var("SBANN_COMP").unwrap_or_else(|_| "opql".into());
    let router: Box<dyn vq::Router> = Box::new(vq::HierRouter::train_hkmeans(&ds, kf, c0, b0, mu.clone()));
    let comp: Box<dyn vq::Compressor> = match comp_s.as_str() {
        "apq4" => Box::new(vq::Apq4::train(&ds, 2, 6, 4.0)),
        "aopq" => Box::new(vq::Opq4::train_aopq(&ds, 2, 6, 8, 4.0)),
        _ => Box::new(vq::Opq4::train_learned(&ds, 2, 6, 8)),
    };
    let idx = vq::Index::build(router, comp, &ds, 2);
    println!("[prof Kf={kf} C0={c0} b0={b0} comp={comp_s}] built");
    let qs = I8Bin::open(qpath).expect("q");
    let (gnq, _gk, _gids) = read_gt(gtpath);
    let nq = qs.nb.min(gnq).min(std::env::var("SBANN_NQ").ok().and_then(|s| s.parse().ok()).unwrap_or(1000));
    let plist: Vec<usize> = std::env::var("SBANN_PLIST").ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![256usize, 512]);
    let tmul: usize = std::env::var("SBANN_TMUL").ok().and_then(|s| s.parse().ok()).unwrap_or(30);
    let tfloor: usize = std::env::var("SBANN_TFLOOR").ok().and_then(|s| s.parse().ok()).unwrap_or(300);
    for &p in &plist {
        let t_surv = (p * tmul).max(tfloor);
        let (mut r, mut s, mut k) = (0u64, 0u64, 0u64);
        for i in 0..nq {
            let (a, b, c) = idx.search_prof(&ds, qs.row(i), p, t_surv, 10);
            r += a; s += b; k += c;
        }
        let tot = (r + s + k) as f64;
        println!("  p={p:5}: route={:.0}us ({:.0}%)  scan={:.0}us ({:.0}%)  rerank={:.0}us ({:.0}%)  [{:.0}us/q]",
            r as f64 / nq as f64 / 1000.0, 100.0 * r as f64 / tot,
            s as f64 / nq as f64 / 1000.0, 100.0 * s as f64 / tot,
            k as f64 / nq as f64 / 1000.0, 100.0 * k as f64 / tot,
            tot / nq as f64 / 1000.0);
    }
}

/// Clean same-index A/B of the two rerank paths (contiguous raw vs ds-gather): build ONE champion
/// index, then bench search (new) vs search_ds (old) ROUND-ROBIN so they share the exact same index
/// and load window. Isolates the cell-contiguous-rerank effect with zero build/load confound.
fn rbench(base: &str, qpath: &str, gtpath: &str) {
    let ds = I8Bin::open(base).expect("base");
    let n = ds.nb;
    let sum: Vec<f64> = (0..n).into_par_iter()
        .fold(|| vec![0f64; ds.d], |mut a, i| { let r = ds.row(i); for k in 0..ds.d { a[k] += r[k] as f64; } a })
        .reduce(|| vec![0f64; ds.d], |mut a, b| { for k in 0..ds.d { a[k] += b[k]; } a });
    let mu: Vec<f32> = if std::env::var("SBANN_NOMU").is_ok() { vec![0f32; ds.d] } else { sum.iter().map(|s| (s / n as f64) as f32).collect() };
    let cfg = std::env::var("SBANN_CONFIGS").unwrap_or_else(|_| "262144:4096:128".into());
    let v: Vec<usize> = cfg.split(',').next().unwrap().split(':').filter_map(|x| x.trim().parse().ok()).collect();
    let (kf, c0, b0) = (v[0], v[1], v[2]);
    let router: Box<dyn vq::Router> = Box::new(vq::HierRouter::train_hkmeans(&ds, kf, c0, b0, mu.clone()));
    let comp: Box<dyn vq::Compressor> = Box::new(vq::Opq4::train_learned(&ds, 2, 6, 8));
    let idx = vq::Index::build(router, comp, &ds, 2);
    println!("[rbench Kf={kf} C0={c0} b0={b0}] built");
    let qs = I8Bin::open(qpath).expect("q");
    let (gnq, gk, gids) = read_gt(gtpath);
    let nq = qs.nb.min(gnq).min(std::env::var("SBANN_NQ").ok().and_then(|s| s.parse().ok()).unwrap_or(1000));
    let plist: Vec<usize> = std::env::var("SBANN_PLIST").ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect()).unwrap_or_else(|| vec![224usize, 256]);
    let reps: usize = std::env::var("SBANN_REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(6);
    let tmul: usize = std::env::var("SBANN_TMUL").ok().and_then(|s| s.parse().ok()).unwrap_or(30);
    let rec = |res: &[Vec<u32>]| -> f64 {
        let mut hit = 0usize;
        for i in 0..nq {
            let truth: std::collections::HashSet<u32> = gids[i * gk..i * gk + 10].iter().copied().collect();
            hit += res[i].iter().take(10).filter(|id| truth.contains(id)).count();
        }
        hit as f64 / (nq * 10) as f64
    };
    for &p in &plist {
        let t_surv = (p * tmul).max(1000);
        let (mut best_new, mut best_old) = (f64::INFINITY, f64::INFINITY);
        let (mut rn, mut ro) = (0.0, 0.0);
        for _ in 0..reps {
            let st = Instant::now();
            let a: Vec<Vec<u32>> = (0..nq).into_par_iter().map(|i| idx.search(&ds, qs.row(i), p, t_surv, 10)).collect();
            best_new = best_new.min(st.elapsed().as_secs_f64()); rn = rec(&a);
            let st = Instant::now();
            let b: Vec<Vec<u32>> = (0..nq).into_par_iter().map(|i| idx.search_ds(&ds, qs.row(i), p, t_surv, 10)).collect();
            best_old = best_old.min(st.elapsed().as_secs_f64()); ro = rec(&b);
        }
        println!("  p={p:5}: CONTIG rec={rn:.4} QPS={:.0} | DS-GATHER rec={ro:.4} QPS={:.0} | speedup={:.2}x",
            nq as f64 / best_new, nq as f64 / best_old, best_old / best_new);
    }
}

/// RECALL-NEUTRALITY PROOF for the fused top-t collect: load ONE index (SBANN_INDEX_LOAD), and for every
/// query run scan_rerank with FUSEDTOPK off (baseline materialize-all + select_nth) and on (fused), then
/// compare the top-10 id SETS. Reports how many queries have an identical set. Env: SBANN_IP, SBANN_FASTSCAN2,
/// SBANN_PLIST (single p), SBANN_TMUL, SBANN_TFLOOR, SBANN_NQ. Same knobs as `run`.
fn fusedab(base: &str, qpath: &str, gtpath: &str) {
    let ds = I8Bin::open(base).expect("base");
    let lp = std::env::var("SBANN_INDEX_LOAD").expect("fusedab needs SBANN_INDEX_LOAD");
    let idx = vq::Index::load_from(&lp).expect("index load");
    println!("[fusedab] loaded {lp}");
    let qs = I8Bin::open(qpath).expect("q");
    let (gnq, gk, gids) = read_gt(gtpath);
    let nq = qs.nb.min(gnq).min(std::env::var("SBANN_NQ").ok().and_then(|s| s.parse().ok()).unwrap_or(10000));
    let p: usize = std::env::var("SBANN_PLIST").ok().and_then(|s| s.split(',').next().unwrap().parse().ok()).unwrap_or(512);
    let tmul: usize = std::env::var("SBANN_TMUL").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let tfloor: usize = std::env::var("SBANN_TFLOOR").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let t = (p * tmul).max(tfloor);
    let (mut ident, mut hb, mut hf) = (0usize, 0usize, 0usize);
    for i in 0..nq {
        let cells = idx.router.probe(qs.row(i), p);
        vq::FUSEDTOPK.store(false, std::sync::atomic::Ordering::Relaxed);
        let mut b = idx.scan_rerank(&ds, qs.row(i), &cells, t, 10);
        vq::FUSEDTOPK.store(true, std::sync::atomic::Ordering::Relaxed);
        let mut f = idx.scan_rerank(&ds, qs.row(i), &cells, t, 10);
        let (bs, fs): (std::collections::HashSet<u32>, std::collections::HashSet<u32>) =
            (b.iter().copied().collect(), f.iter().copied().collect());
        if bs == fs { ident += 1; }
        let truth: std::collections::HashSet<u32> = gids[i * gk..i * gk + 10].iter().copied().collect();
        b.truncate(10); f.truncate(10);
        hb += b.iter().filter(|id| truth.contains(id)).count();
        hf += f.iter().filter(|id| truth.contains(id)).count();
    }
    vq::FUSEDTOPK.store(false, std::sync::atomic::Ordering::Relaxed);
    println!("[fusedab] p={p} t={t} nq={nq}: identical top-10 set for {ident}/{nq} queries ({:.4}%)  | recall@10 baseline={:.5} fused={:.5} (delta={:+.5})",
        100.0 * ident as f64 / nq as f64, hb as f64 / (nq * 10) as f64, hf as f64 / (nq * 10) as f64,
        (hf as f64 - hb as f64) / (nq * 10) as f64);
}

/// WALL-1 SCATTERED-READ MICROBENCH (P189, `scanbench <base> <qpath>`). Isolates the scattered PQ-block
/// read cost: times the raw kernel floor (block reads + LUT, NO collect) over the SAME candidate set in
/// two memory orders — (A) probe order (scattered: route_fine's select_nth scrambles the beam-grouped
/// cells) vs (B) cell-id-sorted (monotonic forward = the streaming floor). Single-thread, best-of-REPS.
/// Env: SBANN_INDEX_LOAD, SBANN_IP, SBANN_FASTSCAN2, SBANN_PLIST (p, first), SBANN_NQ, SBANN_REPS.
/// SBANN_PREFETCH/SBANN_PFDIST/SBANN_PFLINES compose (measures prefetch's effect on the kernel floor).
fn scanbench(base: &str, qpath: &str) {
    let ds = I8Bin::open(base).expect("base");
    let lp = std::env::var("SBANN_INDEX_LOAD").expect("scanbench needs SBANN_INDEX_LOAD");
    let idx = vq::Index::load_from(&lp).expect("index load");
    let qs = I8Bin::open(qpath).expect("q");
    let p: usize = std::env::var("SBANN_PLIST").ok().and_then(|s| s.split(',').next().unwrap().parse().ok()).unwrap_or(512);
    let nq = qs.nb.min(std::env::var("SBANN_NQ").ok().and_then(|s| s.parse().ok()).unwrap_or(2000));
    let reps: usize = std::env::var("SBANN_REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let bb = idx.bb;
    let nc = idx.cell_bstart.len() - 1;
    let total_blocks = if bb > 0 { idx.blocks.len() / bb } else { 0 };
    println!("[scanbench] loaded {lp}  nc={nc}  bb={bb}B  total_blocks={total_blocks}  blocks_array={:.1}MB  p={p} nq={nq} reps={reps}",
        idx.blocks.len() as f64 / 1e6);

    // precompute per-query probe cells (scattered order) + a cell-id-sorted copy + per-query LUT ctx.
    let mut cells_scat: Vec<Vec<u32>> = Vec::with_capacity(nq);
    let mut cells_sort: Vec<Vec<u32>> = Vec::with_capacity(nq);
    let mut ctxs: Vec<vq::QueryCtx> = Vec::with_capacity(nq);
    let mut total_cand: usize = 0;
    let (mut sum_jump_scat, mut sum_jump_sort, mut njump) = (0f64, 0f64, 0usize);
    let mut sum_blk_per_cell = 0f64;
    for i in 0..nq {
        let c = idx.router.probe(qs.row(i), p);
        let mut cs = c.clone();
        cs.sort_unstable();
        total_cand += idx.cell_cand_count(&c);
        // avg |Δ block-offset| between consecutive visited cells (scatter magnitude), scattered vs sorted.
        for w in c.windows(2) {
            let a0 = idx.cell_bstart[w[0] as usize] as i64;
            let a1 = idx.cell_bstart[w[1] as usize] as i64;
            sum_jump_scat += (a1 - a0).unsigned_abs() as f64;
            njump += 1;
        }
        for w in cs.windows(2) {
            let a0 = idx.cell_bstart[w[0] as usize] as i64;
            let a1 = idx.cell_bstart[w[1] as usize] as i64;
            sum_jump_sort += (a1 - a0).unsigned_abs() as f64;
        }
        for &cell in &c { sum_blk_per_cell += (idx.cell_bstart[cell as usize + 1] - idx.cell_bstart[cell as usize]) as f64; }
        ctxs.push(idx.prepare_query_pub(qs.row(i)));
        cells_scat.push(c);
        cells_sort.push(cs);
    }
    let cand_per_q = total_cand as f64 / nq as f64;
    println!("[scanbench] avg cand/query={:.0}  avg blocks/cell={:.2}  avg |Δblk| scattered={:.0} blk ({:.3}MB) -> sorted={:.0} blk ({:.3}MB)",
        cand_per_q, sum_blk_per_cell / (nq * p) as f64,
        sum_jump_scat / njump as f64, sum_jump_scat / njump as f64 * bb as f64 / 1e6,
        sum_jump_sort / njump as f64, sum_jump_sort / njump as f64 * bb as f64 / 1e6);

    // INTERLEAVE the two orders per rep (share the load window) -> robust ratio under residual noise.
    let orders: [(&str, &Vec<Vec<u32>>); 2] = [("SCATTERED(probe)", &cells_scat), ("SORTED(memory) ", &cells_sort)];
    let mut best = [f64::INFINITY; 2];
    let mut sink: i64 = 0;
    for _ in 0..reps {
        for (oi, (_, cellsv)) in orders.iter().enumerate() {
            let t0 = Instant::now();
            let mut acc: i64 = 0;
            for i in 0..nq {
                acc = acc.wrapping_add(idx.scan_kernel_bench(&ds, qs.row(i), &cellsv[i], &ctxs[i]));
            }
            best[oi] = best[oi].min(t0.elapsed().as_secs_f64());
            sink = sink.wrapping_add(acc);
        }
    }
    for (oi, (name, _)) in orders.iter().enumerate() {
        let mcand = total_cand as f64 / best[oi] / 1e6;
        let us_per_q = best[oi] / nq as f64 * 1e6;
        println!("  {name}: {mcand:7.1} Mcand/s   {us_per_q:6.1} us/query   (best/{reps})");
    }
    println!("[scanbench] SORTED/SCATTERED throughput ratio = {:.2}x (recall-neutral streaming headroom); sink={sink}",
        best[0] / best[1]);
}

/// Streaming-mean of a dataset (parallel reduce); SBANN_NOMU -> zero mean (router uses raw space).
fn mean_of(ds: &I8Bin) -> Vec<f32> {
    let n = ds.nb;
    let sum: Vec<f64> = (0..n).into_par_iter()
        .fold(|| vec![0f64; ds.d], |mut a, i| { let r = ds.row(i); for k in 0..ds.d { a[k] += r[k] as f64; } a })
        .reduce(|| vec![0f64; ds.d], |mut a, b| { for k in 0..ds.d { a[k] += b[k]; } a });
    if std::env::var("SBANN_NOMU").is_ok() { vec![0f32; ds.d] } else { sum.iter().map(|s| (s / n as f64) as f32).collect() }
}

fn make_router(ds: &I8Bin, router_s: &str, c: usize, mu: Vec<f32>) -> Box<dyn vq::Router> {
    match router_s {
        "flatsoar" => Box::new(vq::FlatIvf::train_soar(ds, c, mu, 15, 1.0)),
        "flatrair" => Box::new(vq::FlatIvf::train_rair(ds, c, mu, 15, 1.0)),
        _ => Box::new(vq::FlatIvf::train(ds, c, mu, 15)), // "flat"
    }
}

fn make_comp(ds: &I8Bin, comp_s: &str, dpb: usize, eta: f32) -> Box<dyn vq::Compressor> {
    match comp_s {
        "pq4" => Box::new(vq::Pq4::train(ds, dpb, 6)),
        "opq4" => Box::new(vq::Opq4::train(ds, dpb, 6)),
        "opql" => Box::new(vq::Opq4::train_learned(ds, dpb, 6, 8)),
        "i8" => Box::new(vq::ScalarI8::new(ds.d)),
        _ => Box::new(vq::Apq4::train(ds, dpb, 6, eta)), // "apq4"
    }
}

/// recall@10 of search_stream over `nq` queries, GT FILTERED to the live set by `live(id)`. The deep
/// (k=100) ground truth lets us recover the true top-10-among-live as the first 10 live ids of each
/// query's global top-100: any point outside the global top-100 is farther than all of them, so this is
/// EXACT whenever >=10 live ids survive (rare shortfalls are reported, denom = min(10, #live-in-top100)).
#[allow(clippy::too_many_arguments)]
fn eval_stream(idx: &vq::Index, ds: &I8Bin, qs: &I8Bin, gids: &[u32], gk: usize, nq: usize,
               p: usize, t: usize, label: &str, live: &dyn Fn(u32) -> bool) -> f64 {
    let st = Instant::now();
    let res: Vec<Vec<u32>> = (0..nq).into_par_iter().map(|i| idx.search_stream(ds, qs.row(i), p, t, 10)).collect();
    let dt = st.elapsed().as_secs_f64();
    let mut rec_sum = 0f64;
    let mut few = 0usize;
    for i in 0..nq {
        let truth: Vec<u32> = gids[i * gk..i * gk + gk].iter().copied().filter(|&id| live(id)).take(10).collect();
        if truth.len() < 10 { few += 1; }
        let denom = truth.len().min(10);
        if denom == 0 { continue; }
        let tset: std::collections::HashSet<u32> = truth.iter().copied().collect();
        let hit = res[i].iter().take(10).filter(|id| tset.contains(id)).count();
        rec_sum += hit as f64 / denom as f64;
    }
    let rec = rec_sum / nq as f64;
    let warn = if few > 0 { format!("  ({few}/{nq} q had <10 live GT in top-{gk})") } else { String::new() };
    println!("  [{label:>18}] recall@10={rec:.4}  QPS={:.0}{warn}", nq as f64 / dt);
    rec
}

/// STREAMING-track validation. Build on the first n_init points of `base`, then apply a workload
/// (insert the rest -> delete a fraction of the first half) and report recall@10 at each step against
/// the live-set-filtered ground truth — verifying that streaming insert/delete tracks a from-scratch
/// build. Recall (not QPS) is the metric; the box is load-noisy. p/t/n_init via SBANN_P/SBANN_TMUL/SBANN_NINIT.
fn stream(base: &str, qpath: &str, gtpath: &str, router_s: &str, comp_s: &str, a0: usize, c: usize) {
    let t0 = Instant::now();
    let full = I8Bin::open(base).expect("base");
    let (n_total, d) = (full.nb, full.d);
    let n_init = std::env::var("SBANN_NINIT").ok().and_then(|s| s.parse().ok()).unwrap_or(n_total / 2).min(n_total);
    let dpb: usize = std::env::var("SBANN_DPB").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
    let eta: f32 = std::env::var("SBANN_ETA").ok().and_then(|s| s.parse().ok()).unwrap_or(4.0);
    let init = I8Bin::open_range(base, 0, n_init).expect("init view");
    println!("[stream] base nb={n_total} d={d}  n_init={n_init}  to_insert={}  router={router_s} comp={comp_s} a0={a0} C={c} dpb={dpb}",
        n_total - n_init);

    // build the streaming index on the INITIAL subset (cold start: PQ codebook + cells trained on it)
    let mu = mean_of(&init);
    let router = make_router(&init, router_s, c, mu.clone());
    let comp = make_comp(&init, comp_s, dpb, eta);
    let mut idx = vq::Index::build(router, comp, &init, a0);
    println!("  built initial index ({n_init} pts) in {:.1}s", t0.elapsed().as_secs_f64());

    let qs = I8Bin::open(qpath).expect("q");
    let (gnq, gk, gids) = read_gt(gtpath);
    let nq_cap = std::env::var("SBANN_NQ").ok().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    let nq = qs.nb.min(gnq).min(nq_cap);
    let p: usize = std::env::var("SBANN_P").ok().and_then(|s| s.parse().ok()).unwrap_or((c / 16).max(1));
    let tmul: usize = std::env::var("SBANN_TMUL").ok().and_then(|s| s.parse().ok()).unwrap_or(30);
    let t = (p * tmul).max(1000);
    println!("  queries nq={nq} gt_k={gk}  p={p} t={t}");

    // (a) initial-build recall: GT filtered to ids < n_init
    let rec_a = eval_stream(&idx, &init, &qs, &gids, gk, nq, p, t, "initial-build", &|id| (id as usize) < n_init);

    // (b) INSERT the held-out points (orig id = global index), then finalize the buffer encoding
    let ti = Instant::now();
    for j in n_init..n_total { idx.insert(full.row(j), j as u32, a0); }
    idx.finalize_inserts();
    let dti = ti.elapsed().as_secs_f64();
    println!("  inserted {} pts in {:.1}s ({:.0} pts/s); live={}", n_total - n_init, dti, (n_total - n_init) as f64 / dti, n_init + idx.ins_count);
    let rec_b = eval_stream(&idx, &init, &qs, &gids, gk, nq, p, t, "after-insert->1M", &|_| true);

    // (c) DELETE ~20% of the first half (every 5th id) -> tombstones; GT filtered to exclude them
    let td = Instant::now();
    let mut deleted: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for id in (0..n_init).step_by(5) { if idx.delete(id as u32) { deleted.insert(id as u32); } }
    println!("  deleted {} ids in {:.3}s; live={}", deleted.len(), td.elapsed().as_secs_f64(), n_init + idx.ins_count - deleted.len());
    let rec_c = eval_stream(&idx, &init, &qs, &gids, gk, nq, p, t, "after-delete-100k", &|id| !deleted.contains(&id));

    // FRESH from-scratch build on the full live 1M (same config) — the bar (b) should reach.
    drop(idx); // free the streaming index before allocating the fresh one (RAM budget)
    let tf = Instant::now();
    let mu2 = mean_of(&full);
    let router2 = make_router(&full, router_s, c, mu2);
    let comp2 = make_comp(&full, comp_s, dpb, eta);
    let idx2 = vq::Index::build(router2, comp2, &full, a0);
    println!("  fresh full-{n_total} build in {:.1}s", tf.elapsed().as_secs_f64());
    let rec_fresh = eval_stream(&idx2, &full, &qs, &gids, gk, nq, p, t, "fresh-full-1M", &|_| true);

    println!("\n[stream SUMMARY] init({n_init})={rec_a:.4} | insert->1M stream={rec_b:.4} vs fresh={rec_fresh:.4} (gap {:+.4}) | after-delete={rec_c:.4}",
        rec_b - rec_fresh);
}

/// One WALL-1 kernel-throughput measurement over `nblk16` 16-wide blocks (== nblk16*16 candidates).
/// Compares: current AVX2 fast-scan (16-wide, i16 accum), current AVX-512-IL (64-wide, i16 accum),
/// and the PROPER 32-wide int8-saturating FastScan. `scatter`=true iterates blocks in a shuffled
/// order (mimics the cell-scattered real scan) instead of sequentially. Reports Mcand/s (best-of-5).
#[cfg(target_arch = "x86_64")]
fn scanbench2(m: usize, nblk16: usize, scatter: bool, label: &str) {
    let bb16 = (m / 2) * 16;
    let ncand = nblk16 * 16;
    let bytes = nblk16 * bb16;
    // reps so total scanned candidates ~ const (~3e8 for large, more for tiny to stay warm)
    let target: u64 = 400_000_000;
    let reps = ((target / ncand as u64).max(3)) as usize;
    let mut st = 0x1234_5678u64;
    let mut rng = || { st ^= st << 13; st ^= st >> 7; st ^= st << 17; st };
    // 16-wide blocks (used by current AVX2 + AVX-512 kernels)
    let blocks16: Vec<u8> = (0..bytes).map(|_| (rng() & 0xff) as u8).collect();
    // bounded [0,15] LUT so the 32-wide int8-sat path is exact at HOIST=8 (no saturation loss); the
    // 16-wide/int16 kernels are exact for any LUT, so this is a fair common input.
    let lut8: Vec<i8> = (0..m * 16).map(|_| (rng() % 16) as i8).collect();
    let regs8 = pq::lut_regs_i8(&lut8, m);          // 128-bit, current 16-wide kernel
    let regs_y = pq::lut_regs_i8_y256(&lut8, m);    // 256-bit broadcast, new 32-wide kernel
    // re-pack the SAME codes into 32-wide blocks (interleave block 2b and 2b+1 -> one 32-block) so the
    // new kernel scans identical bytes / identical working set.
    let nblk32 = nblk16 / 2;
    let bb32 = (m / 2) * 32;
    let mut blocks32: Vec<u8> = vec![0u8; nblk32 * bb32];
    for b in 0..nblk32 {
        let (s0, s1) = (2 * b, 2 * b + 1);
        for g in 0..m / 2 {
            let d = &mut blocks32[b * bb32 + g * 32..b * bb32 + g * 32 + 32];
            d[..16].copy_from_slice(&blocks16[s0 * bb16 + g * 16..s0 * bb16 + g * 16 + 16]);
            d[16..].copy_from_slice(&blocks16[s1 * bb16 + g * 16..s1 * bb16 + g * 16 + 16]);
        }
    }
    // access order (shared shape for all kernels so scatter is comparable). For scatter we shuffle the
    // 32-block order and derive the 16-block order as (2b, 2b+1) pairs in that shuffled order.
    let mut order32: Vec<usize> = (0..nblk32).collect();
    if scatter {
        for i in (1..nblk32).rev() { let j = (rng() as usize) % (i + 1); order32.swap(i, j); }
    }
    let has512 = std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw");
    let bestof = |f: &mut dyn FnMut() -> f64| -> f64 { (0..5).map(|_| f()).fold(f64::INFINITY, f64::min) };

    // current AVX2 fast-scan (16-wide, int16 accum)
    let mut o16 = [0i32; 16];
    let mut sink = 0i64;
    let t_cur16 = bestof(&mut || {
        let t = Instant::now();
        for &b32 in &order32 { for &b in &[2 * b32, 2 * b32 + 1] {
            unsafe { pq::block_adc_i8_i16acc(&blocks16[b * bb16..(b + 1) * bb16], m, &regs8, &mut o16); }
            sink += o16[0] as i64;
        }}
        t.elapsed().as_secs_f64()
    });
    // proper 32-wide int8-saturating FastScan
    let mut o32 = [0i32; 32];
    let t_fs32 = bestof(&mut || {
        let t = Instant::now();
        for &b in &order32 {
            unsafe { pq::block_adc_i8_fastscan32(&blocks32[b * bb32..(b + 1) * bb32], m, &regs_y, &mut o32); }
            sink += o32[0] as i64;
        }
        t.elapsed().as_secs_f64()
    });
    // current AVX-512 interleaved (64-wide, int16 accum) — build interleaved superblocks from 4 blocks
    let (t_512, has_il) = if has512 && nblk16 >= 8 {
        let regs8z = pq::lut_regs_i8_z512(&lut8, m);
        let nsb = nblk16 / 4;
        let sbb = (m / 2) * 64;
        let mut sblocks = vec![0u8; nsb * sbb];
        for s in 0..nsb {
            let b = s * 4;
            pq::interleave4(&blocks16[b * bb16..(b + 1) * bb16], &blocks16[(b + 1) * bb16..(b + 2) * bb16],
                &blocks16[(b + 2) * bb16..(b + 3) * bb16], &blocks16[(b + 3) * bb16..(b + 4) * bb16],
                m, &mut sblocks[s * sbb..(s + 1) * sbb]);
        }
        let mut order_sb: Vec<usize> = (0..nsb).collect();
        if scatter { for i in (1..nsb).rev() { let j = (rng() as usize) % (i + 1); order_sb.swap(i, j); } }
        let mut o64 = [0i32; 64];
        let t = bestof(&mut || {
            let t = Instant::now();
            for &s in &order_sb {
                unsafe { pq::block_adc_i8_i16acc_avx512_il(&sblocks[s * sbb..(s + 1) * sbb], m, &regs8z, &mut o64); }
                sink += o64[0] as i64;
            }
            t.elapsed().as_secs_f64()
        });
        (t / (nsb * 64) as f64 * ncand as f64, true) // normalize to full ncand
    } else { (0.0, false) };

    let mcs = |t: f64| ncand as f64 / t / 1e6;
    let kb = bytes as f64 / 1024.0;
    print!("{label} m={m} ws={:.0}KB reps~{reps}: cur16w {:.0} Mcand/s | fs32(new) {:.0} Mcand/s ({:.2}x)",
        kb, mcs(t_cur16), mcs(t_fs32), t_cur16 / t_fs32);
    if has_il { print!(" | avx512-64w-il {:.0} Mcand/s ({:.2}x)", mcs(t_512), t_cur16 / t_512); }
    println!("   [sink={sink}]");
}
#[cfg(not(target_arch = "x86_64"))]
fn scanbench2(_m: usize, _n: usize, _s: bool, _l: &str) {}

/// ROUTE-PRIMITIVE microbench (P196, `routebench <base> <qpath>`): isolate router.probe on the loaded 1M
/// index. Reports TRUE us/query (best-of-REPS, no instrumentation) then a SEPARATE ROUTE_PROF pass for the
/// phase split (coarse l2 / coarse-select / fine-expand / final-select) + int8 dist-evals/query.
/// Env: SBANN_INDEX_LOAD, SBANN_PLIST (p), SBANN_NQ, SBANN_REPS. IP/VNNI/router flags honored via main().
fn routebench(base: &str, qpath: &str) {
    let _ = base;
    let lp = std::env::var("SBANN_INDEX_LOAD").expect("routebench needs SBANN_INDEX_LOAD");
    let idx = vq::Index::load_from(&lp).expect("index load");
    let qs = I8Bin::open(qpath).expect("q");
    let p: usize = std::env::var("SBANN_PLIST").ok().and_then(|s| s.split(',').next().unwrap().parse().ok()).unwrap_or(58);
    let nq = qs.nb.min(std::env::var("SBANN_NQ").ok().and_then(|s| s.parse().ok()).unwrap_or(2000));
    let reps: usize = std::env::var("SBANN_REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(7);
    println!("[routebench] loaded {lp}  nc={}  p={p} nq={nq} reps={reps}", idx.router.n_cells());
    // RECALL-EXACT proof: for each query, probe with ROUTE_VNNI off then on, compare the SORTED probed-cell
    // SET. Any mismatch => the VNNI kernel changed which cells are probed (recall not neutral). Report count.
    if std::env::var("SBANN_ROUTE_VERIFY").is_ok() {
        use std::sync::atomic::Ordering::Relaxed;
        let mut mism = 0usize;
        for i in 0..nq {
            vq::ROUTE_VNNI.store(false, Relaxed);
            let mut a = idx.router.probe(qs.row(i), p); a.sort_unstable();
            vq::ROUTE_VNNI.store(true, Relaxed);
            let mut b = idx.router.probe(qs.row(i), p); b.sort_unstable();
            if a != b { mism += 1; }
        }
        vq::ROUTE_VNNI.store(std::env::var("SBANN_ROUTE_VNNI").is_ok(), Relaxed);
        println!("[routebench] RECALL-EXACT check: {mism}/{nq} queries with a DIFFERENT probed-cell set (want 0)");
    }
    let mut sink: u64 = 0;
    // warm pages/caches
    for i in 0..nq { sink = sink.wrapping_add(idx.router.probe(qs.row(i), p).len() as u64); }
    // TRUE us/query: best-of-reps, NO instrumentation. Sum cell ids into sink so probe can't be elided.
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t0 = Instant::now();
        for i in 0..nq {
            sink = sink.wrapping_add(idx.router.probe(qs.row(i), p).iter().map(|&c| c as u64).sum::<u64>());
        }
        best = best.min(t0.elapsed().as_secs_f64());
    }
    let us = best / nq as f64 * 1e6;
    println!("[routebench] ROUTE = {:.2} us/query   ({:.0} probes/s)   best/{reps}", us, nq as f64 / best);
    // PHASE SPLIT: separate ROUTE_PROF pass. Instant per-phase (~4 calls/query) => sub-us distortion, used
    // ONLY for the relative fractions, not the headline us above.
    use std::sync::atomic::Ordering::Relaxed;
    for c in [&vq::PROF_R_COARSE_NS, &vq::PROF_R_CSEL_NS, &vq::PROF_R_FINE_NS, &vq::PROF_R_FSEL_NS, &vq::PROF_R_NEVAL] { c.store(0, Relaxed); }
    vq::ROUTE_PROF.store(true, Relaxed);
    let tp = Instant::now();
    for i in 0..nq { sink = sink.wrapping_add(idx.router.probe(qs.row(i), p).len() as u64); }
    let prof_total = tp.elapsed().as_secs_f64();
    vq::ROUTE_PROF.store(false, Relaxed);
    let (c, cs, f, fs) = (vq::PROF_R_COARSE_NS.load(Relaxed) as f64, vq::PROF_R_CSEL_NS.load(Relaxed) as f64,
                          vq::PROF_R_FINE_NS.load(Relaxed) as f64, vq::PROF_R_FSEL_NS.load(Relaxed) as f64);
    let neval = vq::PROF_R_NEVAL.load(Relaxed) as f64;
    let tot = (c + cs + f + fs).max(1.0);
    let perq = |ns: f64| ns / nq as f64 / 1000.0; // us/query
    println!("[routebench] PHASE (ROUTE_PROF pass, {:.2} us/q incl instr):", prof_total / nq as f64 * 1e6);
    println!("  coarse-l2   {:5.2} us/q  {:5.1}%", perq(c), 100.0 * c / tot);
    println!("  coarse-sel  {:5.2} us/q  {:5.1}%", perq(cs), 100.0 * cs / tot);
    println!("  fine-expand {:5.2} us/q  {:5.1}%", perq(f), 100.0 * f / tot);
    println!("  final-sel   {:5.2} us/q  {:5.1}%", perq(fs), 100.0 * fs / tot);
    println!("  int8 dist-evals/query = {:.0}   sink={sink}", neval / nq as f64);
}

/// Read a boolean override flag. Unset -> `default`. Present and equal to "0" -> false; any other
/// value -> true. The winning OOD levers (FINDINGS P194-P202) default ON via `env_on(name, true)`;
/// pass `SBANN_<NAME>=0` to disable a lever for an A/B. Refuted/experimental flags keep their
/// original opt-in `.is_ok()` reads (default OFF).
fn env_on(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(v) => v != "0",
        Err(_) => default,
    }
}

/// User-facing search policy. The preset chooses a measured-safe starting region; every low-level
/// SBANN_* knob still has precedence. `SBANN_TARGET_RECALL` is a convenience selector when
/// `SBANN_PRESET` is absent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SearchPreset {
    Fast,
    Balanced,
    Accurate,
}

impl SearchPreset {
    fn name(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Balanced => "balanced",
            Self::Accurate => "accurate",
        }
    }

    fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "fast" | "loose" => Ok(Self::Fast),
            "balanced" | "default" => Ok(Self::Balanced),
            "accurate" | "high" => Ok(Self::Accurate),
            _ => Err(format!(
                "unknown SBANN_PRESET={value:?}; expected fast, balanced, or accurate"
            )),
        }
    }

    fn from_target(target: f64) -> Result<Self, String> {
        if !(0.0..=1.0).contains(&target) {
            return Err(format!(
                "SBANN_TARGET_RECALL must be in [0,1], got {target}"
            ));
        }
        Ok(if target <= 0.91 {
            Self::Fast
        } else if target <= 0.96 {
            Self::Balanced
        } else {
            Self::Accurate
        })
    }

    fn graph_hops(self) -> usize {
        match self {
            Self::Fast => 1,
            Self::Balanced => 2,
            Self::Accurate => 3,
        }
    }

    fn graph_m(self) -> usize {
        match self {
            Self::Fast => 16,
            Self::Balanced => 24,
            Self::Accurate => 48,
        }
    }

    fn cascade_k(self, d: usize) -> usize {
        match (self, d > 512) {
            (Self::Fast, false) => 32,
            (Self::Fast, true) => 64,
            (Self::Balanced, false) => 32,
            (Self::Balanced, true) => 64,
            (Self::Accurate, false) => 64,
            (Self::Accurate, true) => 128,
        }
    }

    /// Survivor floor at a 10M reference scale. Search depth grows approximately as sqrt(n)
    /// across the measured 1M/10M/35M ladder, so `survivor_floor` applies that scale below.
    fn survivor_floor_10m(self, d: usize) -> usize {
        let band = if d <= 128 {
            0
        } else if d <= 256 {
            1
        } else if d <= 512 {
            2
        } else {
            3
        };
        match self {
            Self::Fast => [250, 500, 600, 400][band],
            Self::Balanced => [450, 1000, 1200, 1000][band],
            Self::Accurate => [1800, 3000, 2500, 3000][band],
        }
    }

    fn survivor_floor(self, n: usize, d: usize) -> usize {
        let scale = (n as f64 / 10_000_000.0).sqrt().clamp(0.5, 2.5);
        ((self.survivor_floor_10m(d) as f64 * scale).round() as usize).max(128)
    }

    /// Center probe count at Kf=65536. Scaling by the actual cell count transfers the measured
    /// operating regions to coarse 1M indices (Kf=4096/16384) without exposing raw p to users.
    fn reference_probes(self, d: usize) -> usize {
        match self {
            Self::Fast if d <= 128 => 8,
            Self::Balanced if d <= 128 => 8,
            Self::Accurate if d <= 128 => 32,
            // The wider top end is needed for uncalibrated OOD IP data: on text2image-1M,
            // p=16 can stop just short of 0.90 while p=32 clears it. The ladder still starts
            // at p=4 on a 16K-cell index, so the loose/high-QPS corner remains represented.
            Self::Fast if d <= 256 => 32,
            Self::Balanced if d <= 256 => 40,
            Self::Accurate if d <= 256 => 128,
            Self::Fast if d <= 512 => 32,
            Self::Balanced if d <= 512 => 64,
            Self::Accurate if d <= 512 => 256,
            Self::Fast => 32,
            Self::Balanced => 64,
            Self::Accurate => 160,
        }
    }
}

fn resolve_search_preset(
    preset: Option<&str>,
    target_recall: Option<&str>,
) -> Result<(SearchPreset, &'static str), String> {
    if let Some(value) = preset {
        return Ok((SearchPreset::parse(value)?, "SBANN_PRESET"));
    }
    if let Some(value) = target_recall {
        let target = value
            .parse::<f64>()
            .map_err(|_| format!("invalid SBANN_TARGET_RECALL={value:?}"))?;
        return Ok((SearchPreset::from_target(target)?, "SBANN_TARGET_RECALL"));
    }
    Ok((SearchPreset::Balanced, "default"))
}

fn default_probe_ladder(preset: SearchPreset, d: usize, n_cells: usize) -> Vec<usize> {
    let reference = preset.reference_probes(d);
    let center = ((reference * n_cells + 65_535) / 65_536)
        .clamp(1, n_cells.max(1));
    let candidates = [
        (center / 2).max(1),
        center,
        center.saturating_mul(2).min(n_cells.max(1)),
        center.saturating_mul(4).min(n_cells.max(1)),
    ];
    let mut out = Vec::with_capacity(candidates.len());
    for p in candidates {
        if out.last().copied() != Some(p) {
            out.push(p);
        }
    }
    out
}

/// ── CHAMPION OOD STACK (default query path, FINDINGS P190-P202) ─────────────────────────────────
/// The 1M text2image OOD head-to-head vs ScaNN converged on this stack; every lever below is
/// recall-EXACT (verified in its P-entry) and now defaults ON. Each stays overridable with
/// `SBANN_<NAME>=0` for A/B measurement — flip the default, keep the control.
///   • FASTSCAN2  (P195) — 32-wide int8-saturating FastScan. Enabled only when AVX2 is detected
///                         (falls back to the int16 LUT scan otherwise); kernel selftest asserted.
///   • ROUTE_VNNI (P196) — VNNI norm-decomposition routing L2, bit-identical to the AVX2 madd L2.
///                         Enabled only when AVX-512 VNNI is detected (falls back to the AVX2 route
///                         L2 otherwise); norm-kernel selftest asserted.
///   • CASCADE    (P194) — int8-VNNI mid-stage that prunes the apq4 survivor pool to CASCADE_K
///                         before the exact float reorder (only active on the FLOAT_RERANK path).
///                         The int8 rescore is runtime-dispatched VNNI→AVX2→scalar.
///   • FUSEDTOPK  (P187) — ScaNN-style keep-only-survivors scan collect (recall-neutral).
///   • BATCHSCAN  (P202) — cell-major batched FRR scan driver, chunked at BATCH_CHUNK=1000
///                         (recall-bit-identical; only active on the FLOAT_RERANK path).
///   • prefetch   (P194) — the QPS-critical survivor-gather prefetch is UNCONDITIONAL inline in the
///                         rerank kernels (no flag). The per-cell scan prefetch (SBANN_PREFETCH,
///                         P189, recall-neutral) also defaults ON to match the champion recipe.
/// Dataset/mode selectors stay explicit (NOT folded): SBANN_IP (metric), SBANN_FLOAT_RERANK +
/// SBANN_FBASE/FQUERY (needs float base vectors), SBANN_TFLOOR. Refuted levers (aniso partitioning/
/// codes P182/P198, rank-preserving NormPq P200, richer codes P201, P2LAYOUT P195) live only on
/// their experiment branches and are absent here.
///
/// ── OOD CALIBRATION STACK (per-dataset, FINDINGS P206-P216) ──────────────────────────────────────
/// The 1M text2image OOD head-to-head vs ScaNN reached ~0.91x (leaderboard-top-equivalent at 1M
/// single-thread, P216) by adding three levers ON TOP of the default stack above. Unlike the P190-P202
/// levers these are NOT default-on — gamma and graph are calibration knobs fitted on t2i OOD data, so
/// they stay opt-in (in-distribution data likely wants γ≈1 / no graph, P206 caveat). The measured
/// winning config is GR18_t470 = γ=0.5 + graph M=25 + union-trim + t_surv=470, at p=18 / K=16.
///   • gamma       (P206) — SBANN_ROUTE_GAMMA=0.5. Per-finest-cell (γ−1)‖c‖² routing bias (see the
///                          GENERALIZED CASCADE KNOBS block below). Default UNSET (γ=1, bit-identical).
///                          The dominant OOD lever: halves p at fixed recall (p54→p29) at zero query
///                          cost. SEARCH-time only — never set during build/insert (skews SOAR assign).
///                          Fitted on t2i; re-fit per dataset (10M wants 0.5-0.6, P215).
///   • graph       (P207/P209) — SBANN_GRAPH_FILE=<n×k u32 LE IP-kNN adjacency, no header>. Flag-gated
///                          (needs an offline sidecar artifact). On the FLOAT_RERANK cascade path, the
///                          survivor pool's top-M nodes (SBANN_GRAPH_M=25) have their k-NN graph edges
///                          (SBANN_GRAPH_KEDGE=16) union-ed into the rescore set — recovers the deep
///                          neighbours a shallow probe missed. BUILD RECIPE: ScaNN self-search k=16 over
///                          the base vectors (scratchpad gexp_graph_build.py), dump raw u32. Composes
///                          with gamma (additive, ~+4% e2e over gamma-only, P214); overlaps partially
///                          (compound ~0.97 not multiplicative). Orthogonal to the index → no
///                          serialization change (fold the sidecar into the index once the lever lands).
///   • union-trim  (P214) — UNCONDITIONAL on the graph path (no flag): the pool-dedup + neighbour-union
///                          is one open-addressing hash pass with adjacency prefetch, bit-identical to
///                          the old two-pass (verified 2000/2000). SBANN_GRAPH_SORT=1 forces the sorted
///                          gather (default off — deep prefetch wins, +3.7% e2e).
///   • t_surv      (P214/P215) — SBANN_TFLOOR=470 is the OOD graph-mode operating point at 1M (p=18).
///                          t_surv must scale with n: the 1M value CAPS recall at 10M (pool too shallow);
///                          10M wants t_surv≈2000 (P215). Trades against p at a marginal-cost balance.
///
/// ── GENERALIZED CASCADE KNOBS (the L-level router as one tunable family, FINDINGS P210-P211) ──────
/// The router is a general L-level cascade: level i takes the level-(i-1) beam, scores its children,
/// keeps its own beam, recurses; the finest level returns `p` cells; the leaf datapoint scan is the
/// last cascade stage (ADC → t_surv exact-rescore → K prune → float reorder). The knobs, coarse→fine:
///   • DEPTH / SIZES  — `run` router arg picks depth: hierk (L=2), hierk3 (L=3), hierkn (any L via
///                      SBANN_LEVELS=c0,c1,…,Kf). Per-level cell counts: SBANN_C0/C1 (or LEVELS);
///                      finest count = Kf (positional). BUILD-time — one cached index per geometry.
///   • BEAMS  P_i     — SBANN_B0/B1 (or SBANN_BEAMS, len L-1) = cells expanded per level. GAP: beams
///                      are BUILT INTO HierRouter.beam[] (not read at search) — they are part of the
///                      OUTER geometry, NOT a query-side knob. Retune ⇒ rebuild.
///   • FINEST p       — SBANN_PLIST: #finest cells returned to the scan. The dominant search-time lever.
///   • gamma          — SBANN_ROUTE_GAMMA: per-finest-cell (γ−1)‖c‖² bias, applied at level L-1 in ALL
///                      depths. γ≈0.5 halves p at fixed OOD recall (P206); the single biggest lever.
///   • ROUTE_ADC R_i  — SBANN_ROUTE_ADC + _KEEP: 4-bit ADC scoring of the FINEST routing cells, KEEP=0
///                      pure-ADC / KEEP>0 ADC-top-then-exact-rescore. GAP: gbias is NOT added on the
///                      ADC path, so gamma×ADC do not compose (fixable via Pq::query_lut_with_scale on
///                      the adc-route branch). REFUTED at 1M/d=200 for every geometry incl. wide 3-level
///                      finest fan-in (P210-P211, ~1.8× slower) — the exact VNNI router is at the floor.
///   • LEAF R=t_surv  — SBANN_TFLOOR/TMUL: survivors exact-rescored (t_surv = max(p·TMUL, TFLOOR)).
///   • LEAF K         — SBANN_CASCADE_K: int8-prune width before the float reorder. `run` chooses
///                      32/64/128 from the search preset and dimension; an explicit value wins.
/// P211 optimum (1M t2i OOD): L=2, Kf≈16384, C0≈768, gamma≈0.5, exact routing — the champion geometry.
/// Deeper trees only match (never beat) it, and only when kept LEAN (total cells scored ≈2500-2800);
/// (p, t_surv) sits at a marginal-cost balance (one extra probe ≈ the extra gather-bound survivors it
/// saves), so trading t_surv↑ for p↓ is a wash, not equal-work-per-level.
fn main() {
    let a: Vec<String> = std::env::args().collect();
    if std::env::var("SBANN_IP").is_ok() { vq::IP_MODE.store(true, std::sync::atomic::Ordering::Relaxed); }
    if std::env::var("SBANN_POOLDEDUP").is_ok() { vq::POOLDEDUP.store(true, std::sync::atomic::Ordering::Relaxed); }
    if let Ok(s) = std::env::var("SBANN_DEDUP_A0") { if let Ok(v) = s.parse::<usize>() { vq::DEDUP_A0.store(v, std::sync::atomic::Ordering::Relaxed); } }
    if std::env::var("SBANN_PROFILE").is_ok() { vq::PROFILE.store(true, std::sync::atomic::Ordering::Relaxed); }
    if let Ok(s) = std::env::var("SBANN_ROUTE_SDIM") { if let Ok(v) = s.parse::<usize>() { vq::ROUTE_SDIM.store(v, std::sync::atomic::Ordering::Relaxed); } }
    if let Ok(s) = std::env::var("SBANN_ROUTE_SDIM0") { if let Ok(v) = s.parse::<usize>() { vq::ROUTE_SDIM0.store(v, std::sync::atomic::Ordering::Relaxed); } }
    if std::env::var("SBANN_ROUTE_FP16").is_ok() { assert!(simd::selftest_f16(1024) && simd::selftest_f16(200), "f16 kernel != scalar"); vq::ROUTE_FP16.store(true, std::sync::atomic::Ordering::Relaxed); }
    if let Ok(s) = std::env::var("SBANN_BEAM0") { if let Ok(v) = s.parse::<usize>() { vq::BEAM0.store(v, std::sync::atomic::Ordering::Relaxed); } }
    if std::env::var("SBANN_ROUTE_ADC").is_ok() { vq::ROUTE_ADC.store(true, std::sync::atomic::Ordering::Relaxed); }
    // ROUTE_VNNI (P196, champion default ON): VNNI norm-decomposition routing L2, BIT-IDENTICAL to the
    // AVX2-madd L2 (recall-exact). Enabled only when AVX-512 VNNI is detected; vq::gather_fine falls back
    // to the AVX2 block L2 otherwise. Disable with SBANN_ROUTE_VNNI=0.
    if env_on("SBANN_ROUTE_VNNI", true)
        && std::is_x86_feature_detected!("avx512vnni")
        && std::is_x86_feature_detected!("avx512bw")
        && std::is_x86_feature_detected!("avx512f")
    {
        assert!(simd::selftest_l2_norm(200) && simd::selftest_l2_norm(204) && simd::selftest_l2_norm(100),
            "VNNI route-L2 norm kernel != AVX2 madd L2!");
        vq::ROUTE_VNNI.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    if let Ok(s) = std::env::var("SBANN_ROUTE_ADC_KEEP") { if let Ok(v) = s.parse::<usize>() { vq::ROUTE_ADC_KEEP.store(v, std::sync::atomic::Ordering::Relaxed); } }
    if std::env::var("SBANN_NOLUT16").is_ok() { vq::LUT16_OFF.store(true, std::sync::atomic::Ordering::Relaxed); }
    if std::env::var("SBANN_FASTSCAN").is_ok() {
        assert!(pq::selftest_i8_fast(50) && pq::selftest_i8_fast(100), "fast-scan kernel != scalar!");
        vq::FASTSCAN.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    // FASTSCAN2 (P195, champion default ON): PROPER 32-wide int8-saturating FastScan
    // (block_adc_i8_fastscan32_2x16 over the 16-block layout). The kernel + its 256-bit LUT regs need
    // AVX2, so it is enabled only when AVX2 is detected; without AVX2 the scan falls back to the int16
    // LUT path (Apq4::prepare_query). Kernel selftest asserted. Disable with SBANN_FASTSCAN2=0.
    if env_on("SBANN_FASTSCAN2", true) && std::is_x86_feature_detected!("avx2") {
        assert!(pq::selftest_i8_fastscan32(50) && pq::selftest_i8_fastscan32(100) && pq::selftest_i8_fastscan32(20),
            "fastscan32 kernel != scalar!");
        vq::FASTSCAN2.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    // FUSEDTOPK (P187, champion default ON): ScaNN fused top-t — keep a running threshold + emit only
    // survivors (kills the O(candidates) scalar collect). Recall-neutral vs materialize-all + select_nth.
    // Disable with SBANN_FUSEDTOPK=0.
    if env_on("SBANN_FUSEDTOPK", true) {
        vq::FUSEDTOPK.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    if std::env::var("SBANN_SCANDIAG").is_ok() { vq::SCANDIAG.store(true, std::sync::atomic::Ordering::Relaxed); }
    // WALL-1 scattered-read levers (P189): SORTCELLS = monotonic scan order; PREFETCH = SW-prefetch next cell.
    if std::env::var("SBANN_SORTCELLS").is_ok() { vq::SORTCELLS.store(true, std::sync::atomic::Ordering::Relaxed); }
    // PREFETCH (P189, champion default ON): SW-prefetch the next probed cell's blocks during the scan.
    // Recall-neutral (a pure hint). The QPS-critical survivor-gather prefetch is separate & unconditional
    // inline in the rerank kernels (vq.rs rerank_cascade_float / rerank_contig_float). SBANN_PREFETCH=0 off.
    if env_on("SBANN_PREFETCH", true) { vq::PREFETCH.store(true, std::sync::atomic::Ordering::Relaxed); }
    if let Ok(s) = std::env::var("SBANN_PFDIST") { if let Ok(v) = s.parse::<usize>() { vq::PFDIST.store(v, std::sync::atomic::Ordering::Relaxed); } }
    if let Ok(s) = std::env::var("SBANN_PFLINES") { if let Ok(v) = s.parse::<usize>() { vq::PFLINES.store(v, std::sync::atomic::Ordering::Relaxed); } }
    if std::env::var("SBANN_USE512FS").is_ok() {
        // 64-wide AVX-512 interleaved fast-scan (needs FASTSCAN to produce the Pq8 / i8s LUT path).
        assert!(pq::selftest_i8_fast_avx512(50) && pq::selftest_i8_fast_avx512(100), "avx512-64w fast-scan kernel != scalar!");
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
            vq::USE512FS.store(true, std::sync::atomic::Ordering::Relaxed);
        } else {
            eprintln!("SBANN_USE512FS requested but avx512bw not detected; falling back to AVX2 fast-scan");
        }
    }
    // USE512I16 (P241, default ON where supported): 32-wide AVX-512 vpermw kernel for the Pq16
    // (int16-LUT) pair-scan — the path the m>128 scan-precision policy (P239) selects. Identical
    // distances to the 16-wide scan (selftest-asserted), so recall is unchanged; only touches
    // QueryCtx::Pq16, which the champion d=200 FASTSCAN2/Pq8 path never reaches. SBANN_USE512=0 disables.
    if env_on("SBANN_USE512", true)
        && std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
        assert!(pq::selftest_i16_avx512(50) && pq::selftest_i16_avx512(100) && pq::selftest_i16_avx512(384),
            "avx512-32w i16 pair-scan kernel != scalar!");
        vq::USE512I16.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    if std::env::var("SBANN_VNNI").is_ok() {
        assert!(simd::selftest_dot(200) && simd::selftest_dot(204), "VNNI int8 dot != scalar!");
        simd::VNNI_ON.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    if std::env::var("SBANN_RESID").is_ok() {
        assert!(pq::selftest_resid(100, 2) && pq::selftest_resid(96, 4), "resid refine ADC != reconstruct-L2!");
        vq::RESID.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    // RESIDUAL QUANTIZATION: encode x-cell_centroid as the primary 4-bit code + per-cell <q,cent> scan
    // offset (P124, +6-11pt IP pool-recall). apq4 only. Int16 IP path (don't combine with FASTSCAN yet).
    if std::env::var("SBANN_RESIDQ").is_ok() { vq::RESIDQ.store(true, std::sync::atomic::Ordering::Relaxed); }
    // SBANN_RAW_DEDUP (Task B): store the exact-rerank raw array per distinct orig (n*d) instead of per
    // slot (n*a0*d) — shrinks the biggest index array ~a0x, bit-identical recall. Read at BUILD only; the
    // layout is recorded in the index (Index.raw_orig_indexed) so a LOAD restores it without the flag.
    if std::env::var("SBANN_RAW_DEDUP").is_ok() { vq::RAW_DEDUP.store(true, std::sync::atomic::Ordering::Relaxed); }
    // CASCADE (P194, champion default ON): int8 mid-stage that prunes the apq4 survivor pool to
    // CASCADE_K before the expensive float reorder. `run` resolves K from the user-facing preset
    // unless SBANN_CASCADE_K is explicit. Int8 rescore is runtime-dispatched
    // VNNI→AVX2→scalar (vq::rerank_cascade_float). Only active on the SBANN_FLOAT_RERANK path.
    // Disable with SBANN_CASCADE=0.
    if env_on("SBANN_CASCADE", true) { vq::CASCADE.store(true, std::sync::atomic::Ordering::Relaxed); }
    if let Ok(s) = std::env::var("SBANN_CASCADE_K") { if let Ok(v) = s.parse::<usize>() { vq::CASCADE_K.store(v, std::sync::atomic::Ordering::Relaxed); } }
    if std::env::var("SBANN_CASC_SORT").is_ok() { vq::CASC_SORT.store(true, std::sync::atomic::Ordering::Relaxed); }
    if let Ok(s) = std::env::var("SBANN_CASC_DIM") { if let Ok(v) = s.parse::<usize>() { vq::CASC_DIM.store(v, std::sync::atomic::Ordering::Relaxed); } }
    match a.get(1).map(String::as_str) {
        Some("dotbench") => {
            // microbench: VNNI vs AVX2 int8 dot, dim d, REPS over a working set that fits L2 (warm).
            let d: usize = a.get(2).map(|s| s.parse().unwrap()).unwrap_or(200);
            let nv = 4096usize; // working set ~ nv*d bytes (fits L2) so we measure compute, not RAM
            let mut seed = 0x2024u64;
            let mut nb = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1); ((seed >> 24) as i32 % 256 - 128) as i8 };
            let q: Vec<i8> = (0..d).map(|_| nb()).collect();
            let xs: Vec<i8> = (0..nv * d).map(|_| nb()).collect();
            let reps = 2000usize;
            let mut acc = 0i64;
            let t = Instant::now();
            for _ in 0..reps { for j in 0..nv { acc += unsafe { simd::dot_i8_avx2(&q, &xs[j * d..j * d + d]) } as i64; } }
            let t_avx2 = t.elapsed().as_secs_f64();
            let t = Instant::now();
            for _ in 0..reps { for j in 0..nv { acc += unsafe { simd::dot_i8_vnni(&q, &xs[j * d..j * d + d]) } as i64; } }
            let t_vnni = t.elapsed().as_secs_f64();
            let ndot = (reps * nv) as f64;
            println!("dot d={d}: AVX2 {:.1} Mdot/s ({:.2}ns), VNNI {:.1} Mdot/s ({:.2}ns), speedup {:.2}x (acc={acc})",
                ndot / t_avx2 / 1e6, t_avx2 / ndot * 1e9, ndot / t_vnni / 1e6, t_vnni / ndot * 1e9, t_avx2 / t_vnni);
        }
        Some("scanbench2") => {
            // WALL-1 audit: current fast-scan kernels vs a PROPER 32-wide int8-saturating FastScan,
            // measured at TWO working sets (L2-hot vs >L3) to separate cache from kernel throughput.
            let m: usize = a.get(2).map(|s| s.parse().unwrap()).unwrap_or(50);
            for &mm in &[2usize, 16, 50, 100, m] {
                assert!(pq::selftest_i8_fast(mm), "fast-scan kernel != scalar at m={mm}");
                assert!(pq::selftest_i8_fast_avx512(mm), "fast-scan AVX-512 64-wide kernel != scalar at m={mm}");
                assert!(pq::selftest_i8_fastscan32(mm), "fastscan32 kernel != scalar at m={mm}");
            }
            println!("kernels selftest OK (m=2,16,50,100,{m})  [avx2-16w + avx512-64w + fastscan32-32w]");
            let bb16 = (m / 2) * 16;
            // L2-hot: total codes ~256KB (< 1MB L2). LARGE: ~384MB (>> 64MB L3).
            let l2_blocks = (256 * 1024 / bb16) & !3;     // multiple of 4
            let big_blocks = (384 * 1024 * 1024 / bb16) & !3;
            scanbench2(m, l2_blocks, false, "L2-hot   ");
            scanbench2(m, big_blocks, false, "LARGE-seq");
            scanbench2(m, big_blocks, true, "LARGE-scat");
        }
        Some("scanbench") => {
            // selftest + microbench: fast-scan (i8 LUT, 1 vpshufb/subspace) vs int16 (2 vpshufb).
            let m: usize = a.get(2).map(|s| s.parse().unwrap()).unwrap_or(50);
            for &mm in &[2usize, 16, 50, 100, m] {
                assert!(pq::selftest_i8_fast(mm), "fast-scan kernel != scalar at m={mm}");
                assert!(pq::selftest_i8_fast_avx512(mm), "fast-scan AVX-512 64-wide kernel != scalar at m={mm}");
            }
            println!("fast-scan selftest OK (m=2,16,50,100,{m})  [avx2 + avx512-64w]");
            #[cfg(target_arch = "x86_64")]
            {
                let nblk = 4096usize; // working set of packed blocks (fits L2)
                let mut st = 0x1234_5678u64;
                let mut rng = || { st ^= st << 13; st ^= st >> 7; st ^= st << 17; st };
                let blocks: Vec<u8> = (0..nblk * (m / 2) * 16).map(|_| (rng() & 0xff) as u8).collect();
                let lut8: Vec<i8> = (0..m * 16).map(|_| (rng() % 128) as i8).collect();
                let lut16: Vec<i16> = (0..m * 16).map(|_| (rng() % 2000) as i16).collect();
                let regs8 = pq::lut_regs_i8(&lut8, m);
                let (lo, hi) = pq::lut_regs_i16(&lut16, m);
                let bb = (m / 2) * 16;
                let reps = 3000usize;
                let mut o = [0i32; 16];
                let mut sink = 0i64;
                let t = Instant::now();
                for _ in 0..reps { for b in 0..nblk { unsafe { pq::block_adc_i8_i16acc(&blocks[b * bb..(b + 1) * bb], m, &regs8, &mut o); } sink += o[0] as i64; } }
                let t_fast = t.elapsed().as_secs_f64();
                let t = Instant::now();
                for _ in 0..reps { for b in 0..nblk { unsafe { pq::block_adc_i16_avx2(&blocks[b * bb..(b + 1) * bb], m, &lo, &hi, &mut o); } sink += o[0] as i64; } }
                let t_i16 = t.elapsed().as_secs_f64();
                let nb_scanned = (reps * nblk * 16) as f64;
                println!("scan m={m}: fast-i8 {:.0} Mvec/s ({:.2}ns/blk), int16 {:.0} Mvec/s ({:.2}ns/blk), speedup {:.2}x (sink={sink})",
                    nb_scanned / t_fast / 1e6, t_fast / (reps * nblk) as f64 * 1e9,
                    nb_scanned / t_i16 / 1e6, t_i16 / (reps * nblk) as f64 * 1e9, t_i16 / t_fast);
                // 64-wide AVX-512 fast-scan vs the AVX2 fast-scan baseline (both i8 LUT, i16 accum).
                if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
                    let regs8z = pq::lut_regs_i8_z512(&lut8, m);
                    let nq4 = nblk / 4; // process blocks 4-at-a-time
                    let mut o64 = [0i32; 64];
                    let mut sink2 = 0i64;
                    let t = Instant::now();
                    for _ in 0..reps {
                        for q in 0..nq4 {
                            let b = q * 4;
                            unsafe {
                                pq::block_adc_i8_i16acc_avx512(
                                    [&blocks[b * bb..(b + 1) * bb], &blocks[(b + 1) * bb..(b + 2) * bb],
                                     &blocks[(b + 2) * bb..(b + 3) * bb], &blocks[(b + 3) * bb..(b + 4) * bb]],
                                    m, &regs8z, &mut o64);
                            }
                            sink2 += o64[0] as i64;
                        }
                    }
                    let t_512 = t.elapsed().as_secs_f64();
                    let nb512 = (reps * nq4 * 64) as f64;
                    println!("scan m={m}: fast-i8-512(64w,gather) {:.0} Mvec/s ({:.2}ns/4blk), vs fast-i8-avx2 speedup {:.2}x (sink={sink2})",
                        nb512 / t_512 / 1e6, t_512 / (reps * nq4) as f64 * 1e9, t_fast / t_512 * (nb512 / nb_scanned));
                    // INTERLEAVED layout: 1 load/group (proper FastScan-512). Re-pack the test data once.
                    let mut sblocks: Vec<u8> = vec![0u8; nq4 * (m / 2) * 64];
                    let sbb = (m / 2) * 64;
                    for q in 0..nq4 {
                        let b = q * 4;
                        pq::interleave4(&blocks[b * bb..(b + 1) * bb], &blocks[(b + 1) * bb..(b + 2) * bb],
                            &blocks[(b + 2) * bb..(b + 3) * bb], &blocks[(b + 3) * bb..(b + 4) * bb],
                            m, &mut sblocks[q * sbb..(q + 1) * sbb]);
                    }
                    let mut sink3 = 0i64;
                    let t = Instant::now();
                    for _ in 0..reps {
                        for q in 0..nq4 {
                            unsafe { pq::block_adc_i8_i16acc_avx512_il(&sblocks[q * sbb..(q + 1) * sbb], m, &regs8z, &mut o64); }
                            sink3 += o64[0] as i64;
                        }
                    }
                    let t_il = t.elapsed().as_secs_f64();
                    println!("scan m={m}: fast-i8-512(64w,interleaved) {:.0} Mvec/s ({:.2}ns/4blk), vs fast-i8-avx2 speedup {:.2}x (sink={sink3})",
                        nb512 / t_il / 1e6, t_il / (reps * nq4) as f64 * 1e9, t_fast / t_il * (nb512 / nb_scanned));
                }
            }
        }
        Some("abrun") => abrun(&a[2], &a[3], &a[4]),
        Some("prof") => prof(&a[2], &a[3], &a[4]),
        Some("fusedab") => fusedab(&a[2], &a[3], &a[4]),
        Some("scatterbench") => scanbench(&a[2], &a[3]),
        Some("routebench") => routebench(&a[2], &a[3]),
        Some("rbench") => rbench(&a[2], &a[3], &a[4]),
        Some("build") => build(&a[2], a.get(3).map(|s| s.parse().unwrap()).unwrap_or(16384)),
        Some("bench") => bench(&a[2], &a[3], &a[4], a.get(5).map(|s| s.parse().unwrap()).unwrap_or(4096)),
        Some("benchpq") => benchpq(
            &a[2], &a[3], &a[4],
            a.get(5).map(|s| s.parse().unwrap()).unwrap_or(4096),
            a.get(6).map(|s| s.parse().unwrap()).unwrap_or(5),
        ),
        Some("benchavq") => benchavq(
            &a[2], &a[3], &a[4],
            a.get(5).map(|s| s.parse().unwrap()).unwrap_or(256),
            a.get(6).map(|s| s.parse().unwrap()).unwrap_or(256),
        ),
        // selfknn <base.i8bin> <fbase.fbin> <out.u32> <k>: base-side IP-kNN adjacency sidecar
        // (SBANN_GRAPH_FILE recipe) via engine self-query — top-(k+1) with float rerank, self
        // filtered, n x k u32 LE, no header. Needs SBANN_INDEX_LOAD (+ SBANN_IP, usual flags);
        // gamma deliberately NOT applied (base->base is in-distribution; run with it unset).
        Some("selfknn") => {
            let ds = I8Bin::open(&a[2]).expect("base");
            let fb = fbin::FBin::open(&a[3], 0).expect("fbase");
            let kk: usize = a[5].parse().unwrap_or(16);
            let lp = std::env::var("SBANN_INDEX_LOAD").expect("selfknn needs SBANN_INDEX_LOAD");
            let idx = vq::Index::load_from(&lp).expect("index load");
            let p: usize = std::env::var("SBANN_PLIST").ok().and_then(|s| s.parse().ok()).unwrap_or(40);
            let t_surv: usize = std::env::var("SBANN_TFLOOR").ok().and_then(|s| s.parse().ok()).unwrap_or(1000);
            let n = ds.nb;
            // NN-descent bootstrap (P292): optional graph-augmented self-search (SBANN_GRAPH_FILE = current graph).
            // Iterating selfknn with the previous round's graph refines a NATIVE near-exact kNN graph — no external
            // (faiss) builder, and far faster than exhaustive routing since the graph supplies the missed coverage.
            let rl = std::sync::atomic::Ordering::Relaxed;
            let graph: Option<vq::GraphAdj> = if let Ok(gp) = std::env::var("SBANN_GRAPH_FILE") {
                let flen = std::fs::metadata(&gp).expect("graph stat").len() as usize;
                let gk = flen / (n * 4);
                let g = vq::GraphAdj::load(&gp, n, gk).expect("load graph");
                if let Ok(v) = std::env::var("SBANN_GRAPH_M") { vq::GRAPH_M.store(v.parse().expect("M"), rl); }
                if let Ok(v) = std::env::var("SBANN_GRAPH_HOPS") { vq::GRAPH_HOPS.store(v.parse().expect("HOPS"), rl); }
                if let Ok(v) = std::env::var("SBANN_GRAPH_KEDGE") { vq::GRAPH_KEDGE.store(v.parse().expect("KEDGE"), rl); }
                println!("[selfknn] graph-augmented (NN-descent): {gp} k={gk} M={} hops={}", vq::GRAPH_M.load(rl), vq::GRAPH_HOPS.load(rl));
                Some(g)
            } else { None };
            let graph_ref = graph.as_ref();
            println!("[selfknn] n={n} k={kk} p={p} t={t_surv} -> {}", &a[4]);
            let t0 = Instant::now();
            let mut w = std::io::BufWriter::new(std::fs::File::create(&a[4]).expect("out"));
            use std::io::Write;
            let cb = 200_000;
            let mut s = 0usize;
            while s < n {
                let e = (s + cb).min(n);
                let rows: Vec<Vec<u32>> = (s..e).into_par_iter().map(|i| {
                    let qf: Vec<f32> = fb.row(i).to_vec();
                    let res = idx.search_frr(&ds, ds.row(i), &qf, &fb, p, t_surv, kk + 1, graph_ref);
                    let mut out: Vec<u32> = res.into_iter().filter(|&x| x as usize != i).take(kk).collect();
                    let pad = *out.last().unwrap_or(&(i as u32));
                    while out.len() < kk { out.push(pad); }
                    out
                }).collect();
                for r in &rows { for &v in r { w.write_all(&v.to_le_bytes()).unwrap(); } }
                if (s / cb) % 50 == 0 { println!("  {:.0}M / {:.0}M ({:.0}s)", s as f64 / 1e6, n as f64 / 1e6, t0.elapsed().as_secs_f64()); }
                s = e;
            }
            println!("[selfknn] done in {:.0}s", t0.elapsed().as_secs_f64());
        }
        // nndescent <base.i8bin> <out.u32> <k> [seed.u32]: TRUE NN-descent (Dong et al. local join:
        // neighbor-of-neighbor expansion incl. reverse edges, per-edge new/old gating) — self-builds a
        // base-side IP-kNN graph with NO routing in the loop, so the d=768 IVF coverage cap (P294)
        // does not apply. Needs only the raw i8 base (no index, no SBANN_INDEX_LOAD). Output is the
        // usual SBANN_GRAPH_FILE format: flat n x k u32 LE, no header. Seed optional (flat u32 rows,
        // extra cols ignored; pads/dups/self replaced with random); unseeded = random init (pure
        // self-build, no engine artifact at all). i8 dots (VNNI/AVX2/scalar), rows sorted desc by IP.
        // env: SBANN_ND_ROUNDS (12), SBANN_ND_R (reverse-edge cap/node, 16), SBANN_ND_DELTA (0.001),
        //      SBANN_ND_AGE (rounds an edge stays "new", 2 — hubs get multiple chances to propagate it
        //      through the per-round reservoir-resampled reverse cap; 1 = classic single-round aging).
        Some("nndescent") => {
            let ds = I8Bin::open(&a[2]).expect("base");
            // SBANN_ND_N: limit to the first N rows (queries-excluded protocols where the carved
            // query rows are the tail of the base file, e.g. wiki-35M clean GT). Default: all rows.
            let n = std::env::var("SBANN_ND_N").ok().and_then(|s| s.parse().ok())
                .map(|v: usize| { assert!(v <= ds.nb, "SBANN_ND_N > nb"); v }).unwrap_or(ds.nb);
            let kk: usize = a.get(4).map(|s| s.parse().expect("k")).unwrap_or(16);
            let rounds: usize = std::env::var("SBANN_ND_ROUNDS").ok().and_then(|s| s.parse().ok()).unwrap_or(12);
            let rcap: usize = std::env::var("SBANN_ND_R").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
            let delta: f64 = std::env::var("SBANN_ND_DELTA").ok().and_then(|s| s.parse().ok()).unwrap_or(0.001);
            let age0: u8 = std::env::var("SBANN_ND_AGE").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
            assert!(kk <= 64 && rcap <= 64 && n < (1usize << 31) && age0 >= 1);
            let vnni = std::is_x86_feature_detected!("avx512vnni") && std::is_x86_feature_detected!("avx512bw")
                && std::is_x86_feature_detected!("avx512f");
            let avx = std::is_x86_feature_detected!("avx2");
            // SBANN_ND_L2=1: build the graph under L2 instead of IP -- rank candidates by
            // 2*dot(x,y) - ||y||^2 (equivalent to -||x-y||^2 up to the constant ||x||^2).
            let nd_l2 = std::env::var("SBANN_ND_L2").map(|v| v == "1").unwrap_or(false);
            println!("[nnd] n={} d={} k={kk} rounds={rounds} R={rcap} delta={delta} age={age0} vnni={vnni} l2={nd_l2}", n, ds.d);
            let sqn: Vec<i32> = if nd_l2 {
                (0..n).into_par_iter().map(|i| simd::sqnorm_i8(ds.row(i))).collect()
            } else { Vec::new() };
            #[inline(always)]
            fn nd_rand(mut x: u64) -> u64 { x ^= x >> 12; x ^= x << 25; x ^= x >> 27; x.wrapping_mul(0x2545F4914F6CDD1D) }
            const HSZ: usize = 4096; // per-thread stamped dedup table; cands ≲1100 at K=R=16
            #[inline(always)]
            fn hins(ht: &mut [u64], vstamp: u64, key: u32) -> bool { // true = newly inserted
                let mut h = ((key as u64).wrapping_mul(0x9E3779B97F4A7C15) >> 52) as usize & (HSZ - 1);
                loop {
                    let e = ht[h];
                    if e & 0xFFFF_FFFF_0000_0000 != vstamp { ht[h] = vstamp | key as u64; return true; }
                    if (e & 0xFFFF_FFFF) == key as u64 { return false; }
                    h = (h + 1) & (HSZ - 1);
                }
            }
            let t0 = Instant::now();
            // --- init ids: seed file (dedup'd, self-filtered) or random ---
            let mut g: Vec<u32> = vec![0; n * kk];
            if let Some(sp) = a.get(5) {
                let bytes = std::fs::read(sp).expect("seed read");
                let ks = bytes.len() / (n * 4);
                assert!(ks >= 1, "seed too small for n");
                println!("[nnd] seed {} ks={}", sp, ks);
                g.par_chunks_mut(kk).enumerate().for_each(|(i, row)| {
                    let off = i * ks * 4;
                    let mut m = 0usize;
                    let mut seen = [u32::MAX; 64];
                    for j in 0..ks {
                        if m == kk { break; }
                        let id = u32::from_le_bytes(bytes[off + j * 4..off + j * 4 + 4].try_into().unwrap());
                        if id as usize >= n || id as usize == i || seen[..m].contains(&id) { continue; }
                        seen[m] = id; row[m] = id; m += 1;
                    }
                    let mut s = (i as u64) ^ 0x9E37_79B9_7F4A_7C15;
                    while m < kk {
                        s = nd_rand(s);
                        let id = (s % n as u64) as u32;
                        if id as usize == i || seen[..m].contains(&id) { continue; }
                        seen[m] = id; row[m] = id; m += 1;
                    }
                });
            } else {
                println!("[nnd] random init (pure self-build, no seed)");
                g.par_chunks_mut(kk).enumerate().for_each(|(i, row)| {
                    let mut s = (i as u64) ^ 0x9E37_79B9_7F4A_7C15;
                    let mut m = 0usize;
                    let mut seen = [u32::MAX; 64];
                    while m < kk {
                        s = nd_rand(s);
                        let id = (s % n as u64) as u32;
                        if id as usize == i || seen[..m].contains(&id) { continue; }
                        seen[m] = id; row[m] = id; m += 1;
                    }
                });
            }
            // --- initial dots + sort rows desc by IP ---
            let mut gd: Vec<i32> = vec![0; n * kk];
            gd.par_chunks_mut(kk).zip(g.par_chunks_mut(kk)).enumerate().for_each(|(i, (dr, ir))| {
                let q = ds.row(i);
                let mut pairs: Vec<(i32, u32)> = (0..kk).map(|j| {
                    let r = ds.row(ir[j] as usize);
                    let mut dt = if vnni { unsafe { simd::dot_i8_vnni(q, r) } }
                             else if avx { unsafe { simd::dot_i8_avx2(q, r) } }
                             else { -simd::negdot_i8(q, r) };
                    if nd_l2 { dt = 2 * dt - sqn[ir[j] as usize]; }
                    (dt, ir[j])
                }).collect();
                pairs.sort_unstable_by(|x, y| y.0.cmp(&x.0));
                for j in 0..kk { dr[j] = pairs[j].0; ir[j] = pairs[j].1; }
            });
            println!("[nnd] init done ({:.0}s)", t0.elapsed().as_secs_f64());
            // --- rounds: reverse pass, then race-free local join (each task owns row v of the next buffers) ---
            use std::sync::atomic::AtomicU32;
            let rl = std::sync::atomic::Ordering::Relaxed;
            let rev: Vec<AtomicU32> = (0..n * rcap).map(|_| AtomicU32::new(u32::MAX)).collect();
            let rev_cnt: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(0)).collect();
            let mut newf: Vec<u8> = vec![age0; n * kk]; // per-edge "new" age (>0 = new; decays per round)
            // SBANN_ND_INCR_FROM=N (incremental maintenance): rows < N are an already-converged graph
            // (edges start OLD -> no re-join among them); only rows >= N (e.g. a freshly inserted batch,
            // seeded random/routed) start NEW. Local join then activates only around the dirty set.
            if let Some(incr) = std::env::var("SBANN_ND_INCR_FROM").ok().and_then(|s| s.parse::<usize>().ok()) {
                assert!(incr <= n);
                newf[..incr * kk].fill(0);
                println!("[nnd] incremental: rows 0..{incr} start OLD, {incr}..{n} start NEW");
            }
            for round in 0..rounds {
                let tr = Instant::now();
                rev.par_iter().for_each(|x| x.store(u32::MAX, rl));
                rev_cnt.par_iter().for_each(|x| x.store(0, rl));
                { // build capped reverse adjacency; top bit of the entry carries the fwd edge's new flag.
                  // PER-ROUND RESERVOIR sampling (salted): over-cap in-neighbors replace a random slot with
                  // prob rcap/(c+1), so hub reverse lists are freshly resampled each round instead of frozen
                  // first-come — otherwise rev x rev pairs at hubs are missed permanently once flags age.
                    let g = &g; let newf = &newf; let rev = &rev; let rev_cnt = &rev_cnt;
                    let salt = nd_rand(0xD1B5_4A32 ^ (round as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                    (0..n).into_par_iter().for_each(|v| {
                        for j in 0..kk {
                            let u = g[v * kk + j] as usize;
                            let c = rev_cnt[u].fetch_add(1, rl) as usize;
                            let tag = (v as u32) | if newf[v * kk + j] > 0 { 0x8000_0000 } else { 0 };
                            if c < rcap {
                                rev[u * rcap + c].store(tag, rl);
                            } else {
                                let r = (nd_rand(salt ^ ((v as u64) << 32) ^ u as u64) % (c as u64 + 1)) as usize;
                                if r < rcap { rev[u * rcap + r].store(tag, rl); }
                            }
                        }
                    });
                }
                let mut gn = g.clone();
                let mut gdn = gd.clone();
                let mut newn: Vec<u8> = vec![0; n * kk];
                let updates: u64 = {
                    let g = &g; let newf = &newf; let rev = &rev; let ds = &ds;
                    gn.par_chunks_mut(kk).zip(gdn.par_chunks_mut(kk)).zip(newn.par_chunks_mut(kk)).enumerate()
                        .map_init(|| (vec![u64::MAX; HSZ], Vec::<u32>::with_capacity(2048)),
                                  |st, (v, ((grow, gdrow), nrow))| {
                            let (ht, cand) = st;
                            let vstamp = (v as u64) << 32;
                            cand.clear();
                            // carry surviving edges' age forward (decayed); inserts below re-stamp age0
                            for j in 0..kk { nrow[j] = newf[v * kk + j].saturating_sub(1); }
                            hins(ht, vstamp, v as u32);
                            for j in 0..kk { hins(ht, vstamp, grow[j]); }
                            // gather through bridge u: u's fwd row + u's rev list, gated on either edge being new
                            macro_rules! bridge { ($u:expr, $un:expr) => {{
                                let u = $u; let un = $un;
                                let urow = &g[u * kk..u * kk + kk];
                                let unf = &newf[u * kk..u * kk + kk];
                                for t in 0..kk {
                                    if un || unf[t] > 0 { let w = urow[t]; if hins(ht, vstamp, w) { cand.push(w); } }
                                }
                                for t in 0..rcap {
                                    let e = rev[u * rcap + t].load(rl);
                                    if e == u32::MAX { break; }
                                    let x = e & 0x7FFF_FFFF;
                                    if un || (e & 0x8000_0000 != 0) { if hins(ht, vstamp, x) { cand.push(x); } }
                                }
                            }}; }
                            for j in 0..kk {
                                if cand.len() + 2 * (kk + rcap) > HSZ / 2 { break; }
                                bridge!(grow[j] as usize, newf[v * kk + j] > 0);
                            }
                            for t0i in 0..rcap {
                                if cand.len() + 2 * (kk + rcap) > HSZ / 2 { break; }
                                let e0 = rev[v * rcap + t0i].load(rl);
                                if e0 == u32::MAX { break; }
                                let u = (e0 & 0x7FFF_FFFF) as usize;
                                let un = e0 & 0x8000_0000 != 0;
                                if hins(ht, vstamp, u as u32) { cand.push(u as u32); } // rev neighbor itself
                                bridge!(u, un);
                            }
                            // score with prefetch, maintain top-k desc
                            let q = ds.row(v);
                            let m = cand.len();
                            let mut upd = 0u64;
                            for ci in 0..m {
                                #[cfg(target_arch = "x86_64")]
                                if ci + 8 < m {
                                    unsafe { std::arch::x86_64::_mm_prefetch(ds.row(cand[ci + 8] as usize).as_ptr() as *const i8, std::arch::x86_64::_MM_HINT_T0) };
                                }
                                let c = cand[ci];
                                let r = ds.row(c as usize);
                                let mut dt = if vnni { unsafe { simd::dot_i8_vnni(q, r) } }
                                         else if avx { unsafe { simd::dot_i8_avx2(q, r) } }
                                         else { -simd::negdot_i8(q, r) };
                                if nd_l2 { dt = 2 * dt - sqn[c as usize]; }
                                if dt <= gdrow[kk - 1] { continue; }
                                let mut pos = kk - 1;
                                while pos > 0 && gdrow[pos - 1] < dt { pos -= 1; }
                                for jj in (pos..kk - 1).rev() {
                                    gdrow[jj + 1] = gdrow[jj]; grow[jj + 1] = grow[jj]; nrow[jj + 1] = nrow[jj];
                                }
                                gdrow[pos] = dt; grow[pos] = c; nrow[pos] = age0;
                                upd += 1;
                            }
                            upd
                        }).sum()
                };
                let frac = updates as f64 / (n * kk) as f64;
                g = gn; gd = gdn; newf = newn;
                println!("[nnd] round {round}: updates={updates} ({frac:.4}) {:.0}s (cum {:.0}s)",
                         tr.elapsed().as_secs_f64(), t0.elapsed().as_secs_f64());
                if frac < delta { break; }
            }
            let mut w = std::io::BufWriter::new(std::fs::File::create(&a[3]).expect("out"));
            use std::io::Write;
            for &v in &g { w.write_all(&v.to_le_bytes()).unwrap(); }
            w.flush().unwrap();
            println!("[nnd] done n={n} k={kk} -> {} in {:.0}s", &a[3], t0.elapsed().as_secs_f64());
        }
        // prune <base.i8bin> <in_graph.u32> <out_graph.u32> [kin=64] [R=32] [alpha=1.2]:
        // Vamana/DiskANN RobustPrune of a kNN graph into a degree-diversified NAVIGABLE graph. Raw kNN
        // edges are near-parallel short hops, so the best-first beam re-expands the same tiny ball; the
        // occlusion rule keeps a diverse spread of long+short escape edges so the beam converges in far
        // fewer rescored rows. env SBANN_ND_L2=1 => L2 metric (alpha on true L2, i.e. alpha^2 on the
        // squared int8 L2); else IP (rank by -dot). Drops straight into SBANN_GRAPH_FILE.
        Some("prune") => {
            let ds = I8Bin::open(&a[2]).expect("base");
            let n = ds.nb; let d = ds.d;
            let kin: usize = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(64);
            let rr: usize = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(32);
            let alpha: f64 = a.get(7).and_then(|s| s.parse().ok()).unwrap_or(1.2);
            let nd_l2 = std::env::var("SBANN_ND_L2").map(|v| v == "1").unwrap_or(false);
            assert!(rr <= kin && kin <= 64 && alpha >= 1.0);
            let vnni = std::is_x86_feature_detected!("avx512vnni") && std::is_x86_feature_detected!("avx512bw")
                && std::is_x86_feature_detected!("avx512f");
            let avx = std::is_x86_feature_detected!("avx2");
            let a2 = alpha * alpha; // squared-domain threshold for L2 (alpha*L2 <= .. <=> alpha^2*L2sq <= ..)
            let t0 = Instant::now();
            let gin: Vec<u32> = {
                let bytes = std::fs::read(&a[3]).expect("in graph");
                assert_eq!(bytes.len(), n * kin * 4, "in graph size != n*kin*4");
                bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
            };
            println!("[prune] n={n} d={d} kin={kin} R={rr} alpha={alpha} l2={nd_l2} vnni={vnni}");
            macro_rules! dst { ($x:expr, $y:expr) => {{
                if nd_l2 { (if avx { unsafe { simd::l2_i8_avx2($x, $y) } } else { simd::l2_i8_scalar($x, $y) }) as i64 }
                else { let dot = if vnni { unsafe { simd::dot_i8_vnni($x, $y) } }
                                 else if avx { unsafe { simd::dot_i8_avx2($x, $y) } }
                                 else { -simd::negdot_i8($x, $y) }; -(dot as i64) }
            }}; }
            let reverse = std::env::var("SBANN_PRUNE_REVERSE").map(|v| v == "1").unwrap_or(false);
            let incap: usize = std::env::var("SBANN_PRUNE_INCAP").ok().and_then(|s| s.parse().ok()).unwrap_or(rr * 6);
            // RobustPrune occlusion over a candidate id list (dedup'd, self-filtered) -> R selected out-edges
            // (padded). Keep nearest unpruned p, then occlude any farther q whose alpha*d(p,q) <= d(v,q)
            // (q sits "behind" p from v's view -> the p-edge already covers that direction).
            macro_rules! robust { ($v:expr, $cand_ids:expr) => {{
                let v = $v; let qv = ds.row(v);
                let mut ids: Vec<u32> = $cand_ids.iter().copied().filter(|&c| c != u32::MAX && c as usize != v).collect();
                ids.sort_unstable(); ids.dedup();
                let mut cand: Vec<(i64, u32)> = ids.iter().map(|&c| (dst!(qv, ds.row(c as usize)), c)).collect();
                cand.sort_unstable_by_key(|&(dd, _)| dd);
                let mut sel: Vec<u32> = Vec::with_capacity(rr);
                let mut occ = vec![false; cand.len()];
                for i in 0..cand.len() {
                    if occ[i] { continue; }
                    let pi = cand[i].1;
                    sel.push(pi);
                    if sel.len() >= rr { break; }
                    let prow = ds.row(pi as usize);
                    for jx in (i + 1)..cand.len() {
                        if occ[jx] { continue; }
                        let (dvj, pj) = cand[jx];
                        let dpj = dst!(prow, ds.row(pj as usize)) as f64;
                        let lhs = if nd_l2 { a2 * dpj } else { alpha * dpj };
                        if lhs <= dvj as f64 { occ[jx] = true; }
                    }
                }
                if sel.len() < rr { for &(_, c) in cand.iter() { if sel.len() >= rr { break; } if !sel.contains(&c) { sel.push(c); } } }
                while sel.len() < rr { sel.push(if sel.is_empty() { v as u32 } else { sel[sel.len() - 1] }); }
                sel
            }}; }
            // pass 1: RobustPrune each node's kin-NN pool -> R diversified out-edges.
            let mut out = vec![0u32; n * rr];
            out.par_chunks_mut(rr).enumerate().for_each(|(v, orow)| {
                let sel = robust!(v, &gin[v * kin..v * kin + kin]);
                orow.copy_from_slice(&sel[..rr]);
            });
            // pass 2 (SBANN_PRUNE_REVERSE): add reverse edges (full Vamana) then re-prune. Reverse edges give
            // long-range reachability so the beam reaches a neighbour ball from far seeds -> lifts the
            // high-recall tail. Reverse in-edges per node capped at SBANN_PRUNE_INCAP to bound hub cost.
            if reverse {
                let mut roff = vec![0u32; n + 1];
                for &p in out.iter() { if (p as usize) < n { roff[p as usize + 1] += 1; } }
                for i in 0..n { roff[i + 1] += roff[i]; }
                let total = roff[n] as usize;
                let mut rin = vec![0u32; total];
                let mut cur = roff.clone();
                for v in 0..n { for j in 0..rr { let p = out[v * rr + j] as usize; if p < n { let pos = cur[p] as usize; rin[pos] = v as u32; cur[p] += 1; } } }
                let out1 = out.clone();
                out.par_chunks_mut(rr).enumerate().for_each(|(v, orow)| {
                    let rs = roff[v] as usize; let re = roff[v + 1] as usize;
                    let rin_v = &rin[rs..re];
                    let mut ids: Vec<u32> = out1[v * rr..v * rr + rr].to_vec();
                    if rin_v.len() > incap { ids.extend_from_slice(&rin_v[..incap]); } else { ids.extend_from_slice(rin_v); }
                    let sel = robust!(v, &ids);
                    orow.copy_from_slice(&sel[..rr]);
                });
                println!("[prune] reverse pass done (incap={incap})");
            }
            let mut w = std::io::BufWriter::new(std::fs::File::create(&a[4]).expect("out"));
            use std::io::Write;
            for &vv in &out { w.write_all(&vv.to_le_bytes()).unwrap(); }
            w.flush().unwrap();
            println!("[prune] done -> {} R={rr} alpha={alpha} rev={reverse} in {:.0}s", &a[4], t0.elapsed().as_secs_f64());
        }
        // dumpassign <out.u32>: dump the loaded index's slot->orig mapping as flat (orig, finest_cell)
        // u32 LE pairs (multi-assigned points emit one pair per stored copy; padded slots skipped).
        // Enables offline oracle-coverage analysis (which cells hold each query's true neighbors).
        // Needs SBANN_INDEX_LOAD.
        Some("dumpassign") => {
            let lp = std::env::var("SBANN_INDEX_LOAD").expect("dumpassign needs SBANN_INDEX_LOAD");
            let idx = vq::Index::load_from(&lp).expect("index load");
            let nc = idx.cell_bstart.len() - 1;
            let mut w = std::io::BufWriter::new(std::fs::File::create(&a[2]).expect("out"));
            use std::io::Write;
            let mut npairs = 0u64;
            for c in 0..nc {
                let (bs, be) = (idx.cell_bstart[c] as usize, idx.cell_bstart[c + 1] as usize);
                for s in bs * 16..be * 16 {
                    let o = idx.slot_orig[s];
                    if o == u32::MAX { continue; }
                    w.write_all(&o.to_le_bytes()).unwrap();
                    w.write_all(&(c as u32).to_le_bytes()).unwrap();
                    npairs += 1;
                }
            }
            w.flush().unwrap();
            println!("[dumpassign] nc={nc} pairs={npairs} -> {}", &a[2]);
        }
        // dumproute <query.i8bin> <out.u32> <p> [nq]: dump the exact query-time router output.
        // File format: nq:u32, p:u32, followed by nq*p cell ids in query-major order.
        // This is a read-only oracle hook for cell-local portal and cohort experiments.
        Some("dumproute") => {
            let lp = std::env::var("SBANN_INDEX_LOAD").expect("dumproute needs SBANN_INDEX_LOAD");
            let idx = vq::Index::load_from(&lp).expect("index load");
            let qs = I8Bin::open(&a[2]).expect("query i8bin");
            assert_eq!(qs.d, idx.d, "query/index dimension mismatch");
            let p: usize = a[4].parse().expect("p");
            let nq = a.get(5).map(|s| s.parse().expect("nq")).unwrap_or(qs.nb).min(qs.nb);
            let routes = idx.router.probe_batch(
                unsafe { std::slice::from_raw_parts(qs.row(0).as_ptr(), nq * qs.d) },
                nq,
                qs.d,
                p,
            );
            let mut w = std::io::BufWriter::new(std::fs::File::create(&a[3]).expect("out"));
            use std::io::Write;
            w.write_all(&(nq as u32).to_le_bytes()).unwrap();
            w.write_all(&(p as u32).to_le_bytes()).unwrap();
            for c in routes {
                w.write_all(&c.to_le_bytes()).unwrap();
            }
            w.flush().unwrap();
            println!("[dumproute] nq={nq} p={p} -> {}", &a[3]);
        }
        Some("run") => run(&a[2], &a[3], &a[4], &a[5], &a[6], a.get(7).map(|s| s.parse().unwrap()).unwrap_or(2), a.get(8).map(|s| s.parse().unwrap()).unwrap_or(4096), a.get(9).map(|s| s.parse().unwrap()).unwrap_or(30), false),
        Some("runb") => run(&a[2], &a[3], &a[4], &a[5], &a[6], a.get(7).map(|s| s.parse().unwrap()).unwrap_or(2), a.get(8).map(|s| s.parse().unwrap()).unwrap_or(4096), a.get(9).map(|s| s.parse().unwrap()).unwrap_or(30), true),
        Some("runa") => runa(&a[2], &a[3], &a[4], &a[5], &a[6], a.get(7).map(|s| s.parse().unwrap()).unwrap_or(2), a.get(8).map(|s| s.parse().unwrap()).unwrap_or(256)),
        // STREAMING track: build on first SBANN_NINIT pts, insert the rest, delete 20% of the first half,
        // report recall@10 at each step vs live-filtered GT and vs a fresh full build.
        Some("stream") => stream(&a[2], &a[3], &a[4],
            a.get(5).map(|s| s.as_str()).unwrap_or("flat"),
            a.get(6).map(|s| s.as_str()).unwrap_or("apq4"),
            a.get(7).map(|s| s.parse().unwrap()).unwrap_or(1),
            a.get(8).map(|s| s.parse().unwrap()).unwrap_or(4096)),
        _ => eprintln!("usage: sbann build|bench|benchpq|benchavq|run|stream <base> <q> <gt> [router] [compress] [a0] [C]"),
    }
}

#[cfg(test)]
mod search_preset_tests {
    use super::*;

    #[test]
    fn preset_aliases_and_target_bands() {
        assert_eq!(SearchPreset::parse("loose").unwrap(), SearchPreset::Fast);
        assert_eq!(
            SearchPreset::parse("default").unwrap(),
            SearchPreset::Balanced
        );
        assert_eq!(
            SearchPreset::parse("high").unwrap(),
            SearchPreset::Accurate
        );
        assert_eq!(
            SearchPreset::from_target(0.90).unwrap(),
            SearchPreset::Fast
        );
        assert_eq!(
            SearchPreset::from_target(0.95).unwrap(),
            SearchPreset::Balanced
        );
        assert_eq!(
            SearchPreset::from_target(0.99).unwrap(),
            SearchPreset::Accurate
        );
        assert!(SearchPreset::from_target(1.01).is_err());
    }

    #[test]
    fn explicit_preset_precedes_target() {
        let (preset, source) =
            resolve_search_preset(Some("fast"), Some("0.99")).unwrap();
        assert_eq!(preset, SearchPreset::Fast);
        assert_eq!(source, "SBANN_PRESET");
    }

    #[test]
    fn balanced_is_the_default() {
        let (preset, source) = resolve_search_preset(None, None).unwrap();
        assert_eq!(preset, SearchPreset::Balanced);
        assert_eq!(source, "default");
    }

    #[test]
    fn probe_ladders_track_dimension_and_cell_count() {
        assert_eq!(
            default_probe_ladder(SearchPreset::Balanced, 96, 65_536),
            vec![4, 8, 16, 32]
        );
        assert_eq!(
            default_probe_ladder(SearchPreset::Balanced, 200, 4_096),
            vec![1, 3, 6, 12]
        );
        assert_eq!(
            default_probe_ladder(SearchPreset::Fast, 200, 16_384),
            vec![4, 8, 16, 32]
        );
        assert_eq!(
            default_probe_ladder(SearchPreset::Balanced, 1024, 65_536),
            vec![32, 64, 128, 256]
        );
    }

    #[test]
    fn dimension_aware_cascade_and_scale_aware_floor() {
        assert_eq!(SearchPreset::Balanced.cascade_k(96), 32);
        assert_eq!(SearchPreset::Balanced.cascade_k(1024), 64);
        assert_eq!(SearchPreset::Balanced.survivor_floor(10_000_000, 96), 450);
        assert!(
            SearchPreset::Balanced.survivor_floor(35_000_000, 1024)
                > SearchPreset::Balanced.survivor_floor(10_000_000, 1024)
        );
    }
}
