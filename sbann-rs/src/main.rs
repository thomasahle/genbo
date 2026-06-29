//! sbann — streaming int8 IVF-PQ ANN for billion-scale, all-native (no FFI).
//!
//! `build`  : streaming bounded-memory build proof (mmap -> mean -> SIMD assign -> cell counts).
//! `bench`  : end-to-end IVF + exact int8 rerank, recall@10 vs QPS on a dataset+queries+gt.
//! All SIMD lives in `simd.rs` (Rust AVX2, scalar-validated). PQ-ADC bucket scan is the next
//! kernel to fold into the query path; today's rerank is exact int8 L2 over the probed pool.

mod ibin;
mod kmeans;
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
    let mut codes16 = [[0u8; 256]; 16];
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
    let mut seed = 0x1234_5678u64;
    let mut rid = |m: usize| { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1); (seed >> 11) as usize % m };
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
    let ds = I8Bin::open(base).expect("base");
    let n = ds.nb;
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
    let comp: Box<dyn vq::Compressor> = match comp_s {
        "pq4" => Box::new(vq::Pq4::train(&ds, dpb, 6)),
        "opq4" => Box::new(vq::Opq4::train(&ds, dpb, 6)),
        "opql" => Box::new(vq::Opq4::train_learned(&ds, dpb, 6, 8)),
        "opql5" => Box::new(vq::Opq4::train_learned(&ds, 5, 6, 8)),
        "apq4" => Box::new(vq::Apq4::train(&ds, dpb, 6, 4.0)),
        "aopq" => Box::new(vq::Opq4::train_aopq(&ds, dpb, 6, 8, 4.0)),
        "i8" => Box::new(vq::ScalarI8::new(ds.d)),
        _ => { eprintln!("compress?"); return; }
    };
    let idx = vq::Index::build(router, comp, &ds, a0);
    println!("[{router_s}+{comp_s} a0={a0}] built in {:.1}s", t0.elapsed().as_secs_f64());

    let qs = I8Bin::open(qpath).expect("q");
    let (gnq, gk, gids) = read_gt(gtpath);
    // SBANN_NQ caps the #queries (for fair same-NQ head-to-head vs the Python frontier's NQ=1000).
    let nq_cap = std::env::var("SBANN_NQ").ok().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    let nq = qs.nb.min(gnq).min(nq_cap);
    // avq cell count is cb^2 == c; keep probes well under nc
    // SBANN_PLIST="128,256,512" overrides the default sweep (lets a built index be probed at custom p).
    let plist: Vec<usize> = match std::env::var("SBANN_PLIST") {
        Ok(s) => s.split(',').filter_map(|x| x.trim().parse().ok()).filter(|&x: &usize| x >= 1).collect(),
        Err(_) => vec![c / 256, c / 64, c / 16, c / 4].into_iter().map(|x| x.max(1)).collect(),
    };
    // contiguous query array (for batched GEMM routing)
    let mut qarr = vec![0i8; nq * ds.d];
    for i in 0..nq { qarr[i * ds.d..i * ds.d + ds.d].copy_from_slice(qs.row(i)); }
    if batched { println!("  (batched GEMM routing)"); }
    // SBANN_TMUL="10,20,30" sweeps the rerank-survivor multiplier within one build (reuse the index).
    let tlist: Vec<usize> = match std::env::var("SBANN_TMUL") {
        Ok(s) => s.split(',').filter_map(|x| x.trim().parse().ok()).filter(|&x: &usize| x >= 1).collect(),
        Err(_) => vec![tmul],
    };
    // big-ann reports BEST search time over run_count -> measure best-of-REPS to filter box-load
    // spikes on this contended box. SBANN_REPS overrides (default 1; use 3-5 for clean A/B tuning).
    let reps: usize = std::env::var("SBANN_REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    // SBANN_VNNI_AB: interleave VNNI off/on per (p,t) for a clean same-index rerank-kernel A/B.
    let vnni_ab = std::env::var("SBANN_VNNI_AB").is_ok();
    let modes: Vec<bool> = if vnni_ab { vec![false, true] } else { vec![crate::simd::VNNI_ON.load(std::sync::atomic::Ordering::Relaxed)] };
    // SBANN_LUT_AB: interleave int16 (false) vs i8 (true) scan precision per (p,t) on one index.
    let lut_ab = std::env::var("SBANN_LUT_AB").is_ok();
    let lmodes: Vec<bool> = if lut_ab { vec![false, true] } else { vec![vq::LUT16_OFF.load(std::sync::atomic::Ordering::Relaxed)] };
    for &p in &plist {
      for &tm in &tlist {
       for &lm in &lmodes {
        vq::LUT16_OFF.store(lm, std::sync::atomic::Ordering::Relaxed);
       for &vm in &modes {
        crate::simd::VNNI_ON.store(vm, std::sync::atomic::Ordering::Relaxed);
        // survivors kept for exact rerank (tmul tunes recall/speed). The rerank floor was 1000 but that
        // was a ~2x QPS@90% HANDICAP: int16 LUT ranks well enough that t_surv=p*tmul (~256-480) holds
        // recall (P111). Floor now 300 (only affects low-p/QPS@90%; high-p already exceeds it).
        // SBANN_TFLOOR overrides for sweeps.
        let tfloor: usize = std::env::var("SBANN_TFLOOR").ok().and_then(|s| s.parse().ok()).unwrap_or(300);
        let t_surv = (p * tm).max(tfloor);
        let mut best_dt = f64::INFINITY;
        let mut res: Vec<Vec<u32>> = Vec::new();
        for _ in 0..reps.max(1) {
            let st = Instant::now();
            let r: Vec<Vec<u32>> = if batched {
                idx.search_batch(&ds, &qarr, nq, p, t_surv, 10)
            } else {
                (0..nq).into_par_iter().map(|i| idx.search(&ds, qs.row(i), p, t_surv, 10)).collect()
            };
            best_dt = best_dt.min(st.elapsed().as_secs_f64());
            res = r;
        }
        let dt = best_dt;
        let mut hit = 0usize;
        for i in 0..nq {
            let truth: std::collections::HashSet<u32> = gids[i * gk..i * gk + 10].iter().copied().collect();
            hit += res[i].iter().take(10).filter(|id| truth.contains(id)).count();
        }
        let vtag = if vnni_ab { if vm { " VNNI" } else { " AVX2" } } else { "" };
        let ltag = if lut_ab { if lm { " i8" } else { " i16" } } else { "" };
        println!("  p={p:5} t={tm:3}{ltag}{vtag}: recall@10={:.4}  QPS={:.0} (best/{reps})", hit as f64 / (nq * 10) as f64, nq as f64 / dt);
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

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if std::env::var("SBANN_IP").is_ok() { vq::IP_MODE.store(true, std::sync::atomic::Ordering::Relaxed); }
    if std::env::var("SBANN_NOLUT16").is_ok() { vq::LUT16_OFF.store(true, std::sync::atomic::Ordering::Relaxed); }
    if std::env::var("SBANN_FASTSCAN").is_ok() {
        assert!(pq::selftest_i8_fast(50) && pq::selftest_i8_fast(100), "fast-scan kernel != scalar!");
        vq::FASTSCAN.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    if std::env::var("SBANN_USE512FS").is_ok() {
        // 64-wide AVX-512 interleaved fast-scan (needs FASTSCAN to produce the Pq8 / i8s LUT path).
        assert!(pq::selftest_i8_fast_avx512(50) && pq::selftest_i8_fast_avx512(100), "avx512-64w fast-scan kernel != scalar!");
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
            vq::USE512FS.store(true, std::sync::atomic::Ordering::Relaxed);
        } else {
            eprintln!("SBANN_USE512FS requested but avx512bw not detected; falling back to AVX2 fast-scan");
        }
    }
    if std::env::var("SBANN_VNNI").is_ok() {
        assert!(simd::selftest_dot(200) && simd::selftest_dot(204), "VNNI int8 dot != scalar!");
        simd::VNNI_ON.store(true, std::sync::atomic::Ordering::Relaxed);
    }
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
        Some("run") => run(&a[2], &a[3], &a[4], &a[5], &a[6], a.get(7).map(|s| s.parse().unwrap()).unwrap_or(2), a.get(8).map(|s| s.parse().unwrap()).unwrap_or(4096), a.get(9).map(|s| s.parse().unwrap()).unwrap_or(30), false),
        Some("runb") => run(&a[2], &a[3], &a[4], &a[5], &a[6], a.get(7).map(|s| s.parse().unwrap()).unwrap_or(2), a.get(8).map(|s| s.parse().unwrap()).unwrap_or(4096), a.get(9).map(|s| s.parse().unwrap()).unwrap_or(30), true),
        Some("runa") => runa(&a[2], &a[3], &a[4], &a[5], &a[6], a.get(7).map(|s| s.parse().unwrap()).unwrap_or(2), a.get(8).map(|s| s.parse().unwrap()).unwrap_or(256)),
        _ => eprintln!("usage: sbann build|bench|benchpq|benchavq|run <base> <q> <gt> [router] [compress] [a0]"),
    }
}
