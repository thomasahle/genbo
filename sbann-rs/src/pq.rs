//! 4-bit Product Quantization, all-native Rust: per-subspace 16-centroid codebooks, nibble-packed
//! 16-point blocks, SSE `_mm_shuffle_epi8` ADC scan (signed i8 LUT, saturating add), and
//! byte-bucket top-T selection. Exact int8 rerank (simd::l2_i8) finishes. A scalar reference of
//! the block-ADC is used to self-test the SIMD scan.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SBANN_ANISO_CD: use the FAITHFUL ScaNN anisotropic-VQ loss (coordinate-descent over subspaces,
/// parallel residual on the FULL vector, cross-subspace coupling) for both codebook training and
/// per-vector encoding. Default off = the crude block-diagonal per-subspace approximation (eta near-inert).
pub static ANISO_CD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
#[inline]
fn aniso_cd_on() -> bool { ANISO_CD.load(std::sync::atomic::Ordering::Relaxed) }

pub struct Pq {
    pub d: usize,
    pub dpb: usize,
    pub m: usize,       // d/dpb, must be even (nibble packing pairs subspaces)
    pub cent: Vec<f32>, // m*16*dpb
    pub eta: f32,       // anisotropic parallel-error weight (0/1 = plain L2 encode)
}

#[inline]
fn sub_l2(x: &[i8], cent: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for k in 0..x.len() {
        let e = x[k] as f32 - cent[k];
        s += e * e;
    }
    s
}

impl Pq {
    /// Train per-subspace 16-centroid codebooks with a few Lloyd iterations on `rows`.
    pub fn train(rows: &[&[i8]], d: usize, dpb: usize, iters: usize) -> Pq {
        assert!(d % dpb == 0);
        let m = d / dpb;
        assert!(m % 2 == 0, "M={m} must be even for nibble packing (pick dpb dividing d into even M)");
        let n = rows.len();
        let mut cent = vec![0.0f32; m * 16 * dpb];
        let mut seed = 0xdead_beef_1234_5678u64;
        for sub in 0..m {
            let off = sub * dpb;
            for c in 0..16 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let id = (seed >> 11) as usize % n;
                for k in 0..dpb {
                    cent[(sub * 16 + c) * dpb + k] = rows[id][off + k] as f32;
                }
            }
            for _ in 0..iters {
                let mut sum = vec![0.0f64; 16 * dpb];
                let mut cnt = vec![0u32; 16];
                for r in rows {
                    let xs = &r[off..off + dpb];
                    let mut best = f32::INFINITY;
                    let mut bc = 0usize;
                    for c in 0..16 {
                        let dd = sub_l2(xs, &cent[(sub * 16 + c) * dpb..(sub * 16 + c) * dpb + dpb]);
                        if dd < best {
                            best = dd;
                            bc = c;
                        }
                    }
                    cnt[bc] += 1;
                    for k in 0..dpb {
                        sum[bc * dpb + k] += xs[k] as f64;
                    }
                }
                for c in 0..16 {
                    if cnt[c] > 0 {
                        for k in 0..dpb {
                            cent[(sub * 16 + c) * dpb + k] = (sum[c * dpb + k] / cnt[c] as f64) as f32;
                        }
                    }
                }
            }
        }
        Pq { d, dpb, m, cent, eta: 0.0 }
    }

    /// Encode one point to `m` 4-bit codes (one byte each, low nibble used).
    pub fn encode(&self, x: &[i8], out: &mut [u8]) {
        for sub in 0..self.m {
            let off = sub * self.dpb;
            let xs = &x[off..off + self.dpb];
            let mut best = f32::INFINITY;
            let mut bc = 0u8;
            for c in 0..16 {
                let dd = sub_l2(xs, &self.cent[(sub * 16 + c) * self.dpb..(sub * 16 + c) * self.dpb + self.dpb]);
                if dd < best {
                    best = dd;
                    bc = c as u8;
                }
            }
            out[sub] = bc;
        }
    }

    /// Per-query signed-i8 LUT: m*16 bytes, centered+scaled so saturating sums preserve ranking.
    pub fn query_lut(&self, q: &[i8]) -> Vec<i8> {
        let m = self.m;
        let mut f = vec![0.0f32; m * 16];
        for sub in 0..m {
            let off = sub * self.dpb;
            let qs = &q[off..off + self.dpb];
            let mut mean = 0.0f32;
            for c in 0..16 {
                let dd = sub_l2(qs, &self.cent[(sub * 16 + c) * self.dpb..(sub * 16 + c) * self.dpb + self.dpb]);
                f[sub * 16 + c] = dd;
                mean += dd;
            }
            mean /= 16.0;
            for c in 0..16 {
                f[sub * 16 + c] -= mean; // center per subspace
            }
        }
        let maxabs = f.iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(1e-9);
        let scale = 100.0 / (maxabs * (m as f32).sqrt()); // keep saturating sum in i8 range
        let mut lut = vec![0i8; m * 16];
        for i in 0..m * 16 {
            lut[i] = (f[i] * scale).round().clamp(-127.0, 127.0) as i8;
        }
        lut
    }
}

#[inline]
fn sub_l2f(x: &[f32], cent: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for k in 0..x.len() {
        let e = x[k] - cent[k];
        s += e * e;
    }
    s
}

impl Pq {
    /// Train on rotated f32 data (n x d row-major) — for OPQ. Same codebook layout as `train`.
    pub fn train_f32(data: &[f32], d: usize, dpb: usize, n: usize, iters: usize) -> Pq {
        assert!(d % dpb == 0);
        let m = d / dpb;
        assert!(m % 2 == 0, "M={m} must be even");
        let mut cent = vec![0.0f32; m * 16 * dpb];
        let mut seed = 0x0f32_dead_beefu64;
        for sub in 0..m {
            let off = sub * dpb;
            for c in 0..16 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let id = (seed >> 11) as usize % n;
                cent[(sub * 16 + c) * dpb..(sub * 16 + c) * dpb + dpb].copy_from_slice(&data[id * d + off..id * d + off + dpb]);
            }
            for _ in 0..iters {
                let mut sum = vec![0.0f64; 16 * dpb];
                let mut cnt = vec![0u32; 16];
                for i in 0..n {
                    let xs = &data[i * d + off..i * d + off + dpb];
                    let mut best = f32::INFINITY;
                    let mut bc = 0usize;
                    for c in 0..16 {
                        let dd = sub_l2f(xs, &cent[(sub * 16 + c) * dpb..(sub * 16 + c) * dpb + dpb]);
                        if dd < best { best = dd; bc = c; }
                    }
                    cnt[bc] += 1;
                    for k in 0..dpb { sum[bc * dpb + k] += xs[k] as f64; }
                }
                for c in 0..16 {
                    if cnt[c] > 0 {
                        for k in 0..dpb { cent[(sub * 16 + c) * dpb + k] = (sum[c * dpb + k] / cnt[c] as f64) as f32; }
                    }
                }
            }
        }
        Pq { d, dpb, m, cent, eta: 0.0 }
    }

    /// Anisotropic PQ training (ScaNN): penalize residual error parallel to the unit data vector
    /// by factor `eta`. Per-subspace assignment uses the anisotropic loss; centroid update is the
    /// matrix-weighted LS  c = (ΣA)^-1 ΣA x,  A = I + (eta-1) v v^T,  v = unit-full-vector slice.
    pub fn train_f32_aniso(data: &[f32], d: usize, dpb: usize, n: usize, iters: usize, eta: f32) -> Pq {
        if aniso_cd_on() && eta > 1.0 {
            let it = std::env::var("SBANN_ANISO_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(iters.max(10));
            return Pq::train_f32_aniso_cd(data, d, dpb, n, it, eta);
        }
        use nalgebra::{DMatrix, DVector};
        assert!(d % dpb == 0);
        let m = d / dpb;
        assert!(m % 2 == 0);
        // unit-normalized full vectors (parallel direction for the anisotropic weight)
        let mut xhat = vec![0f32; n * d];
        for i in 0..n {
            let mut nrm = 0f32;
            for k in 0..d { nrm += data[i * d + k] * data[i * d + k]; }
            let inv = 1.0 / nrm.sqrt().max(1e-9);
            for k in 0..d { xhat[i * d + k] = data[i * d + k] * inv; }
        }
        let mut cent = vec![0.0f32; m * 16 * dpb];
        let mut seed = 0xa150_beef_u64;
        let em1 = eta - 1.0;
        let aniso_loss = |xs: &[f32], c: &[f32], v: &[f32]| -> f32 {
            let mut l2 = 0f32; let mut proj = 0f32;
            for k in 0..dpb { let r = xs[k] - c[k]; l2 += r * r; proj += r * v[k]; }
            l2 + em1 * proj * proj
        };
        for sub in 0..m {
            let off = sub * dpb;
            for c in 0..16 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let id = (seed >> 11) as usize % n;
                cent[(sub * 16 + c) * dpb..(sub * 16 + c) * dpb + dpb].copy_from_slice(&data[id * d + off..id * d + off + dpb]);
            }
            for _ in 0..iters {
                // accumulate ΣA (dpb*dpb) and ΣAx (dpb) per cluster
                let mut sa = vec![0f64; 16 * dpb * dpb];
                let mut sax = vec![0f64; 16 * dpb];
                for i in 0..n {
                    let xs = &data[i * d + off..i * d + off + dpb];
                    let v = &xhat[i * d + off..i * d + off + dpb];
                    let mut best = f32::INFINITY; let mut bc = 0usize;
                    for c in 0..16 {
                        let dd = aniso_loss(xs, &cent[(sub * 16 + c) * dpb..(sub * 16 + c) * dpb + dpb], v);
                        if dd < best { best = dd; bc = c; }
                    }
                    // A = I + (eta-1) v v^T ; accumulate A and A x
                    for a in 0..dpb {
                        for b in 0..dpb {
                            let aij = (if a == b { 1.0 } else { 0.0 }) + em1 * v[a] * v[b];
                            sa[(bc * dpb + a) * dpb + b] += aij as f64;
                            sax[bc * dpb + a] += (aij * xs[b]) as f64;
                        }
                    }
                }
                for c in 0..16 {
                    let am = DMatrix::<f64>::from_row_slice(dpb, dpb, &sa[c * dpb * dpb..(c + 1) * dpb * dpb]);
                    let bv = DVector::<f64>::from_row_slice(&sax[c * dpb..(c + 1) * dpb]);
                    if let Some(sol) = am.lu().solve(&bv) {
                        for k in 0..dpb { cent[(sub * 16 + c) * dpb + k] = sol[k] as f32; }
                    }
                }
            }
        }
        Pq { d, dpb, m, cent, eta }
    }

    /// FAITHFUL ScaNN anisotropic-VQ PQ training (Guo et al. 2020). Unlike `train_f32_aniso` (which
    /// decouples subspaces and weights the tiny subspace-slice of x̂, making eta near-inert), this
    /// minimizes the anisotropic loss on the FULL residual:
    ///   L_i = ‖r_i‖² + (eta-1)·⟨r_i, x̂_i⟩²   (eta = h_∥/h_⊥, h_⊥=1),  r_i = x_i - x̃_i, x̂_i = x_i/‖x_i‖.
    /// x̃_i is the PQ reconstruction (concat of chosen subspace codewords), so ⟨r_i,x̂_i⟩ = Σ_j⟨r_{i,j},x̂_{i,j}⟩
    /// COUPLES all subspaces. Optimized by coordinate descent over subspaces (ScaNN's algorithm):
    ///   - Assignment (subspace j): pick codeword minimizing ‖x_{i,j}-c‖² + (eta-1)(s_{i,-j}+⟨x_{i,j}-c,x̂_{i,j}⟩)²
    ///     where s_{i,-j} = Σ_{k≠j}⟨r_{i,k},x̂_{i,k}⟩ is the parallel residual from the OTHER subspaces.
    ///   - Codebook update (Thm 4.2, PQ form): c_j = (Σ_i A_{i,j})⁻¹ Σ_i [A_{i,j} x_{i,j} + (eta-1) s_{i,-j} x̂_{i,j}],
    ///     A_{i,j} = I + (eta-1) x̂_{i,j} x̂_{i,j}ᵀ (dpb×dpb block). The +(eta-1)s x̂ term is the cross-subspace
    ///     coupling the crude version drops.
    pub fn train_f32_aniso_cd(data: &[f32], d: usize, dpb: usize, n: usize, iters: usize, eta: f32) -> Pq {
        use nalgebra::{DMatrix, DVector};
        use rayon::prelude::*;
        assert!(d % dpb == 0);
        let m = d / dpb;
        assert!(m % 2 == 0);
        let em1 = eta - 1.0;
        // unit-normalized full vectors (the parallel direction)
        let mut xhat = vec![0f32; n * d];
        xhat.par_chunks_mut(d).enumerate().for_each(|(i, o)| {
            let mut nrm = 0f32;
            for k in 0..d { nrm += data[i * d + k] * data[i * d + k]; }
            let inv = 1.0 / nrm.sqrt().max(1e-9);
            for k in 0..d { o[k] = data[i * d + k] * inv; }
        });
        // init centroids by random sampling per subspace
        let mut cent = vec![0.0f32; m * 16 * dpb];
        let mut seed = 0xa150_beef_u64;
        for sub in 0..m {
            let off = sub * dpb;
            for c in 0..16 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let id = (seed >> 11) as usize % n;
                cent[(sub * 16 + c) * dpb..(sub * 16 + c) * dpb + dpb]
                    .copy_from_slice(&data[id * d + off..id * d + off + dpb]);
            }
        }
        // proj of subspace j residual for point i onto x̂: ⟨x_{i,j}-c, x̂_{i,j}⟩
        let proj = |i: usize, off: usize, c: &[f32]| -> f32 {
            let mut s = 0f32;
            for k in 0..dpb { s += (data[i * d + off + k] - c[k]) * xhat[i * d + off + k]; }
            s
        };
        let l2 = |i: usize, off: usize, c: &[f32]| -> f32 {
            let mut s = 0f32;
            for k in 0..dpb { let e = data[i * d + off + k] - c[k]; s += e * e; }
            s
        };
        // init assignment: plain L2 nearest per subspace
        let mut code = vec![0u8; n * m];
        code.par_chunks_mut(m).enumerate().for_each(|(i, oc)| {
            for sub in 0..m {
                let off = sub * dpb;
                let mut best = f32::INFINITY; let mut bc = 0u8;
                for c in 0..16 {
                    let cc = &cent[(sub * 16 + c) * dpb..(sub * 16 + c) * dpb + dpb];
                    let mut dd = 0f32;
                    for k in 0..dpb { let e = data[i * d + off + k] - cc[k]; dd += e * e; }
                    if dd < best { best = dd; bc = c as u8; }
                }
                oc[sub] = bc;
            }
        });
        // p[i] = full parallel residual = Σ_j ⟨r_{i,j}, x̂_{i,j}⟩
        let mut p = vec![0f32; n];
        {
            let cent_ro = &cent;
            p.par_iter_mut().enumerate().for_each(|(i, pv)| {
                let mut s = 0f32;
                for sub in 0..m {
                    let off = sub * dpb;
                    let c = code[i * m + sub] as usize;
                    let cc = &cent_ro[(sub * 16 + c) * dpb..(sub * 16 + c) * dpb + dpb];
                    for k in 0..dpb { s += (data[i * d + off + k] - cc[k]) * xhat[i * d + off + k]; }
                }
                *pv = s;
            });
        }
        let mut pj_old = vec![0f32; n];
        for _ in 0..iters {
            // guard against float drift: full recompute of p once per outer iter
            {
                let cent_ro = &cent;
                p.par_iter_mut().enumerate().for_each(|(i, pv)| {
                    let mut s = 0f32;
                    for sub in 0..m {
                        let off = sub * dpb;
                        let c = code[i * m + sub] as usize;
                        let cc = &cent_ro[(sub * 16 + c) * dpb..(sub * 16 + c) * dpb + dpb];
                        for k in 0..dpb { s += (data[i * d + off + k] - cc[k]) * xhat[i * d + off + k]; }
                    }
                    *pv = s;
                });
            }
            for sub in 0..m {
                let off = sub * dpb;
                // ---- ASSIGNMENT (parallel over points) ----
                let csub = &cent[sub * 16 * dpb..(sub + 1) * 16 * dpb];
                let updates: Vec<(u8, f32)> = (0..n).into_par_iter().map(|i| {
                    let c_old = code[i * m + sub] as usize;
                    let proj_old = proj(i, off, &csub[c_old * dpb..c_old * dpb + dpb]);
                    let s = p[i] - proj_old; // parallel from OTHER subspaces
                    let mut best = f32::INFINITY; let mut bc = 0u8; let mut bproj = proj_old;
                    for c in 0..16 {
                        let cc = &csub[c * dpb..c * dpb + dpb];
                        let pj = proj(i, off, cc);
                        let full = s + pj;
                        let loss = l2(i, off, cc) + em1 * full * full;
                        if loss < best { best = loss; bc = c as u8; bproj = pj; }
                    }
                    (bc, s + bproj)
                }).collect();
                for i in 0..n { code[i * m + sub] = updates[i].0; p[i] = updates[i].1; }
                // ---- CODEBOOK UPDATE (weighted LS with cross-subspace coupling) ----
                let csub = &cent[sub * 16 * dpb..(sub + 1) * 16 * dpb];
                let (sa, sax): (Vec<f64>, Vec<f64>) = (0..n).into_par_iter()
                    .fold(|| (vec![0f64; 16 * dpb * dpb], vec![0f64; 16 * dpb]),
                        |(mut sa, mut sax), i| {
                            let c = code[i * m + sub] as usize;
                            let cc = &csub[c * dpb..c * dpb + dpb];
                            let xs = &data[i * d + off..i * d + off + dpb];
                            let v = &xhat[i * d + off..i * d + off + dpb];
                            let pj_cur = { let mut s = 0f32; for k in 0..dpb { s += (xs[k] - cc[k]) * v[k]; } s };
                            let s_other = p[i] - pj_cur; // s_{i,-j}
                            let vx: f32 = (0..dpb).map(|k| v[k] * xs[k]).sum();
                            for a in 0..dpb {
                                for b in 0..dpb {
                                    let aij = (if a == b { 1.0 } else { 0.0 }) + em1 * v[a] * v[b];
                                    sa[(c * dpb + a) * dpb + b] += aij as f64;
                                }
                                sax[c * dpb + a] += (xs[a] + em1 * v[a] * vx + em1 * s_other * v[a]) as f64;
                            }
                            (sa, sax)
                        })
                    .reduce(|| (vec![0f64; 16 * dpb * dpb], vec![0f64; 16 * dpb]),
                        |(mut sa, mut sax), (b1, b2)| {
                            for k in 0..sa.len() { sa[k] += b1[k]; }
                            for k in 0..sax.len() { sax[k] += b2[k]; }
                            (sa, sax)
                        });
                // capture pj_old (with OLD centroid) so p can be corrected after the update
                {
                    let csub = &cent[sub * 16 * dpb..(sub + 1) * 16 * dpb];
                    pj_old.par_iter_mut().enumerate().for_each(|(i, o)| {
                        let c = code[i * m + sub] as usize;
                        *o = proj(i, off, &csub[c * dpb..c * dpb + dpb]);
                    });
                }
                // solve per codeword
                for c in 0..16 {
                    let am = DMatrix::<f64>::from_row_slice(dpb, dpb, &sa[c * dpb * dpb..(c + 1) * dpb * dpb]);
                    let bv = DVector::<f64>::from_row_slice(&sax[c * dpb..(c + 1) * dpb]);
                    if let Some(sol) = am.lu().solve(&bv) {
                        for k in 0..dpb { cent[(sub * 16 + c) * dpb + k] = sol[k] as f32; }
                    }
                }
                // correct p for the moved centroids of subspace j
                {
                    let csub = &cent[sub * 16 * dpb..(sub + 1) * 16 * dpb];
                    let dps: Vec<f32> = (0..n).into_par_iter().map(|i| {
                        let c = code[i * m + sub] as usize;
                        proj(i, off, &csub[c * dpb..c * dpb + dpb]) - pj_old[i]
                    }).collect();
                    for i in 0..n { p[i] += dps[i]; }
                }
            }
        }
        Pq { d, dpb, m, cent, eta }
    }

    /// FAITHFUL anisotropic encode: coordinate descent over subspaces minimizing the full-vector
    /// anisotropic loss (match `train_f32_aniso_cd`). Used when SBANN_ANISO_CD is on.
    fn encode_f32_cd(&self, x: &[f32], out: &mut [u8]) {
        let m = self.m; let dpb = self.dpb; let em1 = self.eta - 1.0;
        let mut vhat = [0f32; 256];
        let mut nrm = 0f32;
        for k in 0..self.d { nrm += x[k] * x[k]; }
        let inv = 1.0 / nrm.sqrt().max(1e-9);
        for k in 0..self.d { vhat[k] = x[k] * inv; }
        let projc = |off: usize, cc: &[f32]| -> f32 {
            let mut s = 0f32; for k in 0..dpb { s += (x[off + k] - cc[k]) * vhat[off + k]; } s
        };
        for sub in 0..m {
            let off = sub * dpb;
            let mut best = f32::INFINITY; let mut bc = 0u8;
            for c in 0..16 {
                let cc = &self.cent[(sub * 16 + c) * dpb..(sub * 16 + c) * dpb + dpb];
                let dd = sub_l2f(&x[off..off + dpb], cc);
                if dd < best { best = dd; bc = c as u8; }
            }
            out[sub] = bc;
        }
        let mut p = 0f32;
        for sub in 0..m {
            let off = sub * dpb; let c = out[sub] as usize;
            p += projc(off, &self.cent[(sub * 16 + c) * dpb..(sub * 16 + c) * dpb + dpb]);
        }
        for _ in 0..4 {
            for sub in 0..m {
                let off = sub * dpb;
                let c_old = out[sub] as usize;
                let proj_old = projc(off, &self.cent[(sub * 16 + c_old) * dpb..(sub * 16 + c_old) * dpb + dpb]);
                let s = p - proj_old;
                let mut best = f32::INFINITY; let mut bc = 0u8; let mut bproj = proj_old;
                for c in 0..16 {
                    let cc = &self.cent[(sub * 16 + c) * dpb..(sub * 16 + c) * dpb + dpb];
                    let pj = projc(off, cc);
                    let full = s + pj;
                    let loss = sub_l2f(&x[off..off + dpb], cc) + em1 * full * full;
                    if loss < best { best = loss; bc = c as u8; bproj = pj; }
                }
                out[sub] = bc; p = s + bproj;
            }
        }
    }

    pub fn encode_f32(&self, x: &[f32], out: &mut [u8]) {
        if aniso_cd_on() && self.eta > 1.0 { self.encode_f32_cd(x, out); return; }
        // anisotropic encode (match aniso training): assign by L2 + (eta-1)*(parallel error)^2
        let em1 = if self.eta > 0.0 { self.eta - 1.0 } else { 0.0 };
        let mut vhat = [0f32; 256];
        if em1 != 0.0 {
            let mut nrm = 0f32;
            for k in 0..self.d { nrm += x[k] * x[k]; }
            let inv = 1.0 / nrm.sqrt().max(1e-9);
            for k in 0..self.d { vhat[k] = x[k] * inv; }
        }
        for sub in 0..self.m {
            let off = sub * self.dpb;
            let xs = &x[off..off + self.dpb];
            let mut best = f32::INFINITY;
            let mut bc = 0u8;
            for c in 0..16 {
                let cc = &self.cent[(sub * 16 + c) * self.dpb..(sub * 16 + c) * self.dpb + self.dpb];
                let mut dd = sub_l2f(xs, cc);
                if em1 != 0.0 {
                    let mut proj = 0f32;
                    for k in 0..self.dpb { proj += (xs[k] - cc[k]) * vhat[off + k]; }
                    dd += em1 * proj * proj;
                }
                if dd < best { best = dd; bc = c as u8; }
            }
            out[sub] = bc;
        }
    }

    /// Reconstruct a vector from its codes (concat of assigned subspace centroids). For OPQ update.
    pub fn decode(&self, code: &[u8], out: &mut [f32]) {
        for sub in 0..self.m {
            let c = code[sub] as usize;
            out[sub * self.dpb..sub * self.dpb + self.dpb]
                .copy_from_slice(&self.cent[(sub * 16 + c) * self.dpb..(sub * 16 + c) * self.dpb + self.dpb]);
        }
    }

    pub fn query_lut_f32(&self, q: &[f32]) -> Vec<i8> {
        let m = self.m;
        let mut f = vec![0.0f32; m * 16];
        for sub in 0..m {
            let off = sub * self.dpb;
            let qs = &q[off..off + self.dpb];
            let mut mean = 0.0f32;
            for c in 0..16 {
                let dd = sub_l2f(qs, &self.cent[(sub * 16 + c) * self.dpb..(sub * 16 + c) * self.dpb + self.dpb]);
                f[sub * 16 + c] = dd;
                mean += dd;
            }
            mean /= 16.0;
            for c in 0..16 { f[sub * 16 + c] -= mean; }
        }
        let maxabs = f.iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(1e-9);
        let scale = 100.0 / (maxabs * (m as f32).sqrt());
        f.iter().map(|&v| (v * scale).round().clamp(-127.0, 127.0) as i8).collect()
    }

    /// int16 LUT (UNCENTERED positive distances, scaled so the worst-case sum just fits i16 ->
    /// i16 accumulation never saturates, full ~15-bit resolution). Much finer ranking than the i8
    /// LUT -> far fewer exact reranks (the LUT16 trick; pairs with block_adc_i16 / _avx2).
    pub fn query_lut_f32_i16(&self, q: &[f32]) -> Vec<i16> {
        let m = self.m;
        let mut f = vec![0.0f32; m * 16];
        let mut summax = 0.0f32;
        for sub in 0..m {
            let off = sub * self.dpb;
            let qs = &q[off..off + self.dpb];
            let mut smax = 0.0f32;
            for c in 0..16 {
                let dd = sub_l2f(qs, &self.cent[(sub * 16 + c) * self.dpb..(sub * 16 + c) * self.dpb + self.dpb]);
                f[sub * 16 + c] = dd;
                smax = smax.max(dd);
            }
            summax += smax;
        }
        let scale = 30000.0 / summax.max(1e-9); // worst-case full-block sum -> ~30000 < i16 max
        f.iter().map(|&v| (v * scale).round().clamp(0.0, 32767.0) as i16).collect()
    }

    /// FAST-SCAN L2 LUT (int8 per-subspace, int16-accumulating). Each subspace's 16 dists are shifted
    /// by the subspace MIN (constant offset cancels in ranking) then scaled by a SINGLE global factor so
    /// the largest per-subspace VARIATION fits int8 (<=127). Sum over m subspaces fits int16 (<= m*127).
    /// Resolution ~12-13 bit (vs the sqrt(m) i8 path's ~8) because int8 is spent on within-subspace
    /// variation, not absolute distance. Pairs with block_adc_i8_i16acc (1 vpshufb/subspace = ~2x scan).
    pub fn query_lut_f32_i8s(&self, q: &[f32]) -> Vec<i8> {
        let m = self.m;
        let mut f = vec![0.0f32; m * 16];
        let mut maxrange = 0.0f32;
        for sub in 0..m {
            let off = sub * self.dpb;
            let qs = &q[off..off + self.dpb];
            let mut smin = f32::INFINITY;
            for c in 0..16 {
                let dd = sub_l2f(qs, &self.cent[(sub * 16 + c) * self.dpb..(sub * 16 + c) * self.dpb + self.dpb]);
                f[sub * 16 + c] = dd;
                smin = smin.min(dd);
            }
            let mut smax = 0.0f32;
            for c in 0..16 { f[sub * 16 + c] -= smin; smax = smax.max(f[sub * 16 + c]); } // [0, range_sub]
            maxrange = maxrange.max(smax);
        }
        let scale = 127.0 / maxrange.max(1e-9);
        f.iter().map(|&v| (v * scale).round().clamp(0.0, 127.0) as i8).collect()
    }

    /// ASYMMETRIC MIPS LUT (int8, sqrt(m)-scaled like the L2 i8 LUT): -<q_sub,cent> centered, for the
    /// FAST i8 scan (1 vpshufb/subspace vs the i16 path's 2). Coarser ranking, but for OOD the exact-IP
    /// rerank fixes the final order -> trades scan-resolution for ~2x scan speed (the OOD bottleneck).
    pub fn query_lut_f32_ip_i8(&self, q: &[f32]) -> Vec<i8> {
        let m = self.m;
        let mut f = vec![0.0f32; m * 16];
        for sub in 0..m {
            let off = sub * self.dpb;
            let qs = &q[off..off + self.dpb];
            let mut mean = 0.0f32;
            for c in 0..16 {
                let ct = &self.cent[(sub * 16 + c) * self.dpb..(sub * 16 + c) * self.dpb + self.dpb];
                let mut dot = 0.0f32;
                for k in 0..self.dpb { dot += qs[k] * ct[k]; }
                f[sub * 16 + c] = -dot;
                mean += -dot;
            }
            mean /= 16.0;
            for c in 0..16 { f[sub * 16 + c] -= mean; }
        }
        let maxabs = f.iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(1e-9);
        let scale = 100.0 / (maxabs * (m as f32).sqrt());
        f.iter().map(|&v| (v * scale).round().clamp(-127.0, 127.0) as i8).collect()
    }

    /// FAST-SCAN ASYMMETRIC MIPS LUT (int8, int16-accumulating): like query_lut_f32_i16_ip but encoded
    /// for the 1-vpshufb fast-scan. Per subspace -<q,cent> shifted by smin (offset cancels in ranking),
    /// single global scale so the largest per-subspace range fits int8 -> ~12-13 bit IP-ranking at i8
    /// scan speed. For OOD: accurate IP candidate selection (shallow rerank) AT fast-scan throughput.
    pub fn query_lut_f32_i8s_ip(&self, q: &[f32]) -> Vec<i8> {
        let m = self.m;
        let mut f = vec![0.0f32; m * 16];
        let mut maxrange = 0.0f32;
        for sub in 0..m {
            let off = sub * self.dpb;
            let qs = &q[off..off + self.dpb];
            let mut smin = f32::INFINITY;
            for c in 0..16 {
                let ct = &self.cent[(sub * 16 + c) * self.dpb..(sub * 16 + c) * self.dpb + self.dpb];
                let mut dot = 0.0f32;
                for k in 0..self.dpb { dot += qs[k] * ct[k]; }
                let v = -dot; // smaller = larger IP
                f[sub * 16 + c] = v;
                smin = smin.min(v);
            }
            let mut smax = 0.0f32;
            for c in 0..16 { f[sub * 16 + c] -= smin; smax = smax.max(f[sub * 16 + c]); }
            maxrange = maxrange.max(smax);
        }
        let scale = 127.0 / maxrange.max(1e-9);
        f.iter().map(|&v| (v * scale).round().clamp(0.0, 127.0) as i8).collect()
    }

    /// ASYMMETRIC MIPS LUT (int16): per subspace, the table value is (-<q_sub,cent>) shifted to be
    /// positive per subspace, so the i16-accumulating scan ranks candidates by APPROX INNER PRODUCT
    /// (smallest sum = largest IP). The per-subspace shift is a constant added to every candidate ->
    /// it cancels in ranking. Lets the PQ scan SELECT MIPS candidates directly -> shallow rerank.
    pub fn query_lut_f32_i16_ip(&self, q: &[f32]) -> Vec<i16> {
        let m = self.m;
        let mut f = vec![0.0f32; m * 16];
        let mut summax = 0.0f32;
        for sub in 0..m {
            let off = sub * self.dpb;
            let qs = &q[off..off + self.dpb];
            let mut smin = f32::INFINITY;
            for c in 0..16 {
                let ct = &self.cent[(sub * 16 + c) * self.dpb..(sub * 16 + c) * self.dpb + self.dpb];
                let mut dot = 0.0f32;
                for k in 0..self.dpb { dot += qs[k] * ct[k]; }
                let v = -dot; // smaller = larger IP
                f[sub * 16 + c] = v;
                smin = smin.min(v);
            }
            let mut smax = 0.0f32;
            for c in 0..16 { f[sub * 16 + c] -= smin; smax = smax.max(f[sub * 16 + c]); } // shift -> [0, smax]
            summax += smax;
        }
        let scale = 30000.0 / summax.max(1e-9);
        f.iter().map(|&v| (v * scale).round().clamp(0.0, 32767.0) as i16).collect()
    }

    /// Scale used by query_lut_f32_i16_ip (= 30000/summax). Lets the residual-quant scan convert the
    /// float per-cell offset <q,centroid> into the same i16 units as the residual ADC scores.
    pub fn ip_i16_scale(&self, q: &[f32]) -> f32 {
        let m = self.m; let mut summax = 0.0f32;
        for sub in 0..m {
            let qs = &q[sub * self.dpb..sub * self.dpb + self.dpb];
            let mut smin = f32::INFINITY; let mut v = [0f32; 16];
            for c in 0..16 {
                let ct = &self.cent[(sub * 16 + c) * self.dpb..(sub * 16 + c) * self.dpb + self.dpb];
                let mut dot = 0.0f32; for k in 0..self.dpb { dot += qs[k] * ct[k]; }
                v[c] = -dot; smin = smin.min(-dot);
            }
            let mut smax = 0.0f32; for c in 0..16 { smax = smax.max(v[c] - smin); }
            summax += smax;
        }
        30000.0 / summax.max(1e-9)
    }
    /// Scale used by query_lut_f32_i8s_ip (= 127/maxrange), for the fast-scan residual-quant offset.
    pub fn ip_i8s_scale(&self, q: &[f32]) -> f32 {
        let m = self.m; let mut maxrange = 0.0f32;
        for sub in 0..m {
            let qs = &q[sub * self.dpb..sub * self.dpb + self.dpb];
            let mut smin = f32::INFINITY; let mut v = [0f32; 16];
            for c in 0..16 {
                let ct = &self.cent[(sub * 16 + c) * self.dpb..(sub * 16 + c) * self.dpb + self.dpb];
                let mut dot = 0.0f32; for k in 0..self.dpb { dot += qs[k] * ct[k]; }
                v[c] = -dot; smin = smin.min(-dot);
            }
            let mut smax = 0.0f32; for c in 0..16 { smax = smax.max(v[c] - smin); }
            maxrange = maxrange.max(smax);
        }
        127.0 / maxrange.max(1e-9)
    }
}

/// 8-bit-per-subspace refine PQ (256 centroids/subspace, 1 byte/code) — a SECOND, finer in-stream
/// code (IDEA #4). Stored per-vector contiguous (m bytes) parallel to Index.raw, used to REFINE the
/// 4-bit-ADC survivor ranking before the exact raw rerank, so far fewer raw vectors are read.
/// It is an INDEPENDENT finer PQ over the same subspace partition (not the literal base-residual:
/// a true base+residual reconstruction needs BOTH codes' cross term c1·c2 per subspace at refine,
/// which couples them; an independent 256-centroid code gives the full ||q-decode8(x)||^2 from one
/// code, no cross term, no base-code recovery — same memory-traffic win, cleaner + exact ADC).
pub struct ResidPq {
    pub d: usize,
    pub dpb: usize,
    pub m: usize,       // d/dpb subspaces; code = m bytes/vector
    pub cent: Vec<f32>, // m*256*dpb centroids (row-major: [(sub*256+c)*dpb + k])
}

impl ResidPq {
    /// Train 256-centroid codebooks per subspace via a few Lloyd iterations on f32 rows (n x d).
    /// Parallel over subspaces (256-way assignment is 16x the 4-bit cost, so we fan it out).
    pub fn train_f32(data: &[f32], d: usize, dpb: usize, n: usize, iters: usize) -> ResidPq {
        assert!(d % dpb == 0);
        let m = d / dpb;
        let nc = 256usize;
        use rayon::prelude::*;
        let subcents: Vec<Vec<f32>> = (0..m).into_par_iter().map(|sub| {
            let off = sub * dpb;
            let mut cent = vec![0f32; nc * dpb];
            let mut seed = 0x9e37_79b9_7f4a_7c15u64 ^ ((sub as u64) << 32);
            for c in 0..nc {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let id = (seed >> 11) as usize % n;
                cent[c * dpb..c * dpb + dpb].copy_from_slice(&data[id * d + off..id * d + off + dpb]);
            }
            for _ in 0..iters {
                let mut sum = vec![0f64; nc * dpb];
                let mut cnt = vec![0u32; nc];
                for i in 0..n {
                    let xs = &data[i * d + off..i * d + off + dpb];
                    let mut best = f32::INFINITY;
                    let mut bc = 0usize;
                    for c in 0..nc {
                        let dd = sub_l2f(xs, &cent[c * dpb..c * dpb + dpb]);
                        if dd < best { best = dd; bc = c; }
                    }
                    cnt[bc] += 1;
                    for k in 0..dpb { sum[bc * dpb + k] += xs[k] as f64; }
                }
                for c in 0..nc {
                    if cnt[c] > 0 { for k in 0..dpb { cent[c * dpb + k] = (sum[c * dpb + k] / cnt[c] as f64) as f32; } }
                }
            }
            cent
        }).collect();
        let mut cent = vec![0f32; m * nc * dpb];
        for sub in 0..m { cent[sub * nc * dpb..(sub + 1) * nc * dpb].copy_from_slice(&subcents[sub]); }
        ResidPq { d, dpb, m, cent }
    }

    /// Encode one f32 vector to `m` 8-bit codes (nearest centroid per subspace).
    pub fn encode_f32(&self, x: &[f32], out: &mut [u8]) {
        let nc = 256usize;
        for sub in 0..self.m {
            let off = sub * self.dpb;
            let xs = &x[off..off + self.dpb];
            let mut best = f32::INFINITY;
            let mut bc = 0u8;
            for c in 0..nc {
                let dd = sub_l2f(xs, &self.cent[(sub * nc + c) * self.dpb..(sub * nc + c) * self.dpb + self.dpb]);
                if dd < best { best = dd; bc = c as u8; }
            }
            out[sub] = bc;
        }
    }

    /// Per-query refine LUT: m*256 f32 squared-L2 dists q_sub -> each subspace centroid.
    pub fn query_lut_f32(&self, q: &[f32]) -> Vec<f32> {
        let nc = 256usize;
        let mut lut = vec![0f32; self.m * nc];
        for sub in 0..self.m {
            let off = sub * self.dpb;
            let qs = &q[off..off + self.dpb];
            for c in 0..nc {
                lut[sub * nc + c] = sub_l2f(qs, &self.cent[(sub * nc + c) * self.dpb..(sub * nc + c) * self.dpb + self.dpb]);
            }
        }
        lut
    }

    /// IP refine LUT: per-subspace -<q_sub, cent[c]> so adc() sums to -<q, decode8(code)> (smaller =
    /// larger inner product). For OOD/MIPS: the 8-bit refine re-ranks IP candidates by APPROX IP, so it
    /// can shrink the exact raw-IP rerank depth the same way the L2 LUT does for msspacev.
    pub fn query_lut_f32_ip(&self, q: &[f32]) -> Vec<f32> {
        let nc = 256usize;
        let mut lut = vec![0f32; self.m * nc];
        for sub in 0..self.m {
            let off = sub * self.dpb;
            let qs = &q[off..off + self.dpb];
            for c in 0..nc {
                let ct = &self.cent[(sub * nc + c) * self.dpb..(sub * nc + c) * self.dpb + self.dpb];
                let mut dot = 0f32;
                for k in 0..self.dpb { dot += qs[k] * ct[k]; }
                lut[sub * nc + c] = -dot;
            }
        }
        lut
    }

    /// Refined approx distance ||q - decode8(code)||^2 from the precomputed query LUT.
    #[inline]
    pub fn adc(&self, code: &[u8], lut: &[f32]) -> f32 {
        let nc = 256usize;
        let mut s = 0f32;
        for sub in 0..self.m { s += lut[sub * nc + code[sub] as usize]; }
        s
    }
}

/// Self-test: refine ADC (LUT path) must equal the brute-force reconstruct-then-L2 for a random
/// query + code. Pure scalar, but guards the LUT indexing / decode wiring.
pub fn selftest_resid(d: usize, dpb: usize) -> bool {
    if d % dpb != 0 { return true; }
    let m = d / dpb;
    let mut seed = 0x1234_9e37u64;
    let mut nb = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1); (((seed >> 33) as i64 % 255) - 127) as f32 };
    let n = 600usize;
    let data: Vec<f32> = (0..n * d).map(|_| nb()).collect();
    let rpq = ResidPq::train_f32(&data, d, dpb, n, 2);
    let q: Vec<f32> = (0..d).map(|_| nb()).collect();
    let lut = rpq.query_lut_f32(&q);
    let mut code = vec![0u8; m];
    rpq.encode_f32(&data[0..d], &mut code);
    let via_lut = rpq.adc(&code, &lut);
    // brute force: reconstruct and L2
    let mut brute = 0f32;
    for sub in 0..m {
        let c = code[sub] as usize;
        for k in 0..dpb {
            let e = q[sub * dpb + k] - rpq.cent[(sub * 256 + c) * dpb + k];
            brute += e * e;
        }
    }
    (via_lut - brute).abs() <= 1e-2 * (1.0 + brute.abs())
}

/// Pack the codes of 16 points (each `m` bytes) into a block: m/2 groups of 16 bytes,
/// byte i of group g = code[i][2g] | (code[i][2g+1] << 4).
pub fn pack_block(codes16: &[[u8; 256]], m: usize, out: &mut Vec<u8>) {
    for g in 0..m / 2 {
        for i in 0..16 {
            out.push(codes16[i][2 * g] | (codes16[i][2 * g + 1] << 4));
        }
    }
}

/// Scalar reference: ADC distances for the 16 points of one block, given the signed LUT.
pub fn block_adc_scalar(block: &[u8], m: usize, lut: &[i8]) -> [i32; 16] {
    let mut acc = [0i32; 16];
    for g in 0..m / 2 {
        let base = g * 16;
        for i in 0..16 {
            let byte = block[base + i];
            let lo = (byte & 0x0f) as usize;
            let hi = (byte >> 4) as usize;
            // saturating i8 add to mirror the SIMD path exactly
            acc[i] = (acc[i] + lut[(2 * g) * 16 + lo] as i32).clamp(-128, 127);
            acc[i] = (acc[i] + lut[(2 * g + 1) * 16 + hi] as i32).clamp(-128, 127);
        }
    }
    acc
}

/// int16-LUT ADC with i32 accumulation (NO saturation) -> full-resolution approx distances.
/// Scalar reference for the AVX2 kernel below.
pub fn block_adc_i16(block: &[u8], m: usize, lut: &[i16], out: &mut [i32; 16]) {
    let mut acc = [0i32; 16];
    for g in 0..m / 2 {
        let base = g * 16;
        for i in 0..16 {
            let byte = block[base + i];
            let lo = (byte & 0x0f) as usize;
            let hi = (byte >> 4) as usize;
            acc[i] += lut[(2 * g) * 16 + lo] as i32;
            acc[i] += lut[(2 * g + 1) * 16 + hi] as i32;
        }
    }
    *out = acc;
}

/// Per-subspace LUT split into low-byte and high-byte 16-entry tables (the LUT16 trick): a vpshufb
/// on each, combined to i16, lets us look up 16 i16 values per subspace with byte shuffles.
#[cfg(target_arch = "x86_64")]
pub fn lut_regs_i16(lut: &[i16], m: usize) -> (Vec<__m128i>, Vec<__m128i>) {
    let mut lo = Vec::with_capacity(m);
    let mut hi = Vec::with_capacity(m);
    for s in 0..m {
        let mut lb = [0u8; 16];
        let mut hb = [0u8; 16];
        for c in 0..16 {
            let v = lut[s * 16 + c] as u16;
            lb[c] = (v & 0xff) as u8;
            hb[c] = (v >> 8) as u8;
        }
        unsafe {
            lo.push(_mm_loadu_si128(lb.as_ptr() as *const __m128i));
            hi.push(_mm_loadu_si128(hb.as_ptr() as *const __m128i));
        }
    }
    (lo, hi)
}

/// AVX2 int16-LUT ADC: per subspace, two vpshufb (lo/hi byte tables) -> 16 i16 -> add to i16
/// accumulators. LUT is uncentered/positive and scaled so the full sum < i16 max => no saturation,
/// so this matches block_adc_i16 (scalar) exactly. ~15-bit ranking resolution vs the i8 path's ~8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn block_adc_i16_avx2(block: &[u8], m: usize, lut_lo: &[__m128i], lut_hi: &[__m128i], out: &mut [i32; 16]) {
    let mask = _mm_set1_epi8(0x0f);
    let mut acc = _mm256_setzero_si256(); // 16 x i16
    for g in 0..m / 2 {
        let codes = _mm_loadu_si128(block.as_ptr().add(g * 16) as *const __m128i);
        let lo_nib = _mm_and_si128(codes, mask);
        let hi_nib = _mm_and_si128(_mm_srli_epi16(codes, 4), mask);
        // subspace 2g (lo nibble)
        let lb = _mm_shuffle_epi8(lut_lo[2 * g], lo_nib);
        let hb = _mm_shuffle_epi8(lut_hi[2 * g], lo_nib);
        let v = _mm256_set_m128i(_mm_unpackhi_epi8(lb, hb), _mm_unpacklo_epi8(lb, hb));
        acc = _mm256_add_epi16(acc, v);
        // subspace 2g+1 (hi nibble)
        let lb2 = _mm_shuffle_epi8(lut_lo[2 * g + 1], hi_nib);
        let hb2 = _mm_shuffle_epi8(lut_hi[2 * g + 1], hi_nib);
        let v2 = _mm256_set_m128i(_mm_unpackhi_epi8(lb2, hb2), _mm_unpacklo_epi8(lb2, hb2));
        acc = _mm256_add_epi16(acc, v2);
    }
    let mut tmp = [0i16; 16];
    _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, acc);
    for i in 0..16 { out[i] = tmp[i] as i32; }
}

/// Per-subspace int8 LUT (16 entries) loaded into a __m128i for the fast-scan kernel.
#[cfg(target_arch = "x86_64")]
pub fn lut_regs_i8(lut: &[i8], m: usize) -> Vec<__m128i> {
    (0..m).map(|s| unsafe { _mm_loadu_si128(lut.as_ptr().add(s * 16) as *const __m128i) }).collect()
}

/// FAST-SCAN ADC: int8 LUT, ONE vpshufb/subspace -> 16 int8 partials -> widen to int16 -> accumulate.
/// ~2x fewer ops than block_adc_i16_avx2 (which does 2 vpshufb + 2 unpack/subspace) at ~12-13 bit
/// ranking resolution. LUT must be the i8s LUT (per-subspace shifted, <=127) so the i16 acc never
/// saturates (sum <= m*127 < 32767 for m<=258). Matches block_adc_i8_scalar exactly.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn block_adc_i8_i16acc(block: &[u8], m: usize, lut: &[__m128i], out: &mut [i32; 16]) {
    let mask = _mm_set1_epi8(0x0f);
    let mut acc = _mm256_setzero_si256(); // 16 x i16
    for g in 0..m / 2 {
        let codes = _mm_loadu_si128(block.as_ptr().add(g * 16) as *const __m128i);
        let lo = _mm_and_si128(codes, mask);
        let hi = _mm_and_si128(_mm_srli_epi16(codes, 4), mask);
        let p0 = _mm_shuffle_epi8(lut[2 * g], lo);       // 16 i8 (subspace 2g)
        acc = _mm256_add_epi16(acc, _mm256_cvtepi8_epi16(p0));
        let p1 = _mm_shuffle_epi8(lut[2 * g + 1], hi);   // 16 i8 (subspace 2g+1)
        acc = _mm256_add_epi16(acc, _mm256_cvtepi8_epi16(p1));
    }
    let mut tmp = [0i16; 16];
    _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, acc);
    for i in 0..16 { out[i] = tmp[i] as i32; }
}

/// Scalar reference for the fast-scan kernel: sum the int8 LUT entries per vector (nibble-decoded).
pub fn block_adc_i8_scalar(block: &[u8], m: usize, lut: &[i8], out: &mut [i32; 16]) {
    for i in 0..16 {
        let mut s = 0i32;
        for g in 0..m / 2 {
            let byte = block[g * 16 + i];
            s += lut[(2 * g) * 16 + (byte & 0x0f) as usize] as i32;
            s += lut[(2 * g + 1) * 16 + (byte >> 4) as usize] as i32;
        }
        out[i] = s;
    }
}

/// Self-test: fast-scan AVX2 kernel must match the scalar reference for a random LUT + block.
pub fn selftest_i8_fast(m: usize) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if !std::is_x86_feature_detected!("avx2") { return true; }
        // pseudo-random i8 LUT in [0,127] and a random packed block
        let mut st = 0x9e3779b97f4a7c15u64;
        let mut rng = || { st ^= st << 13; st ^= st >> 7; st ^= st << 17; st };
        let lut: Vec<i8> = (0..m * 16).map(|_| (rng() % 128) as i8).collect();
        let mut block = vec![0u8; (m / 2) * 16];
        for b in block.iter_mut() { *b = (rng() & 0xff) as u8; }
        let regs = lut_regs_i8(&lut, m);
        let mut a = [0i32; 16];
        let mut b = [0i32; 16];
        unsafe { block_adc_i8_i16acc(&block, m, &regs, &mut a); }
        block_adc_i8_scalar(&block, m, &lut, &mut b);
        if a != b { eprintln!("selftest_i8_fast MISMATCH m={m}: {a:?} vs {b:?}"); return false; }
    }
    true
}

/// Per-subspace int8 LUT (16 entries) broadcast to ALL FOUR 128-bit lanes of a zmm, for the 64-wide
/// fast-scan. `_mm512_shuffle_epi8` is IN-LANE, so each 128-bit lane independently indexes its own copy
/// of the 16-entry table — the broadcast gives each lane that copy. Pairs with block_adc_i8_i16acc_avx512.
#[cfg(target_arch = "x86_64")]
pub fn lut_regs_i8_z512(lut: &[i8], m: usize) -> Vec<__m512i> {
    (0..m)
        .map(|s| unsafe {
            let r = _mm_loadu_si128(lut.as_ptr().add(s * 16) as *const __m128i);
            _mm512_broadcast_i32x4(r) // same 16-byte LUT replicated into lanes 0..3
        })
        .collect()
}

/// AVX-512 64-wide FAST-SCAN ADC: process FOUR consecutive 16-vector blocks (64 vectors) at once.
/// Per subspace, ONE `_mm512_shuffle_epi8` against the lane-broadcast i8 LUT looks up 64 int8 partials
/// (each 128-bit lane uses its block's 16 codes against its own copy of the 16-entry table); the 64 i8
/// are widened (two `_mm512_cvtepi8_epi16` over the 256-bit halves) into two int16 accumulators and
/// summed. LUT must be the i8s LUT (per-subspace shifted, <=127) so the i16 accum never saturates
/// (sum <= m*127 < 32767 for m<=258). Matches block_adc_i8_scalar on all 4 sub-blocks exactly.
///   out[0..16]=block0, out[16..32]=block1, out[32..48]=block2, out[48..64]=block3.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn block_adc_i8_i16acc_avx512(blocks: [&[u8]; 4], m: usize, lut_z: &[__m512i], out: &mut [i32; 64]) {
    let mask = _mm512_set1_epi8(0x0f);
    let mut acc01 = _mm512_setzero_si512(); // 32 x i16 (lane-pair: block0 in 0..15, block1 in 16..31)
    let mut acc23 = _mm512_setzero_si512(); // 32 x i16 (block2 in 0..15, block3 in 16..31)
    for g in 0..m / 2 {
        // gather group g of each of the 4 blocks into the 4 lanes of a zmm
        let c0 = _mm_loadu_si128(blocks[0].as_ptr().add(g * 16) as *const __m128i);
        let c1 = _mm_loadu_si128(blocks[1].as_ptr().add(g * 16) as *const __m128i);
        let c2 = _mm_loadu_si128(blocks[2].as_ptr().add(g * 16) as *const __m128i);
        let c3 = _mm_loadu_si128(blocks[3].as_ptr().add(g * 16) as *const __m128i);
        let mut codes = _mm512_castsi128_si512(c0);
        codes = _mm512_inserti32x4(codes, c1, 1);
        codes = _mm512_inserti32x4(codes, c2, 2);
        codes = _mm512_inserti32x4(codes, c3, 3);
        let lo = _mm512_and_si512(codes, mask);
        let hi = _mm512_and_si512(_mm512_srli_epi16(codes, 4), mask);
        // subspace 2g (lo nibble): 64 i8 partials, one per (block,vector)
        let p0 = _mm512_shuffle_epi8(lut_z[2 * g], lo);
        acc01 = _mm512_add_epi16(acc01, _mm512_cvtepi8_epi16(_mm512_castsi512_si256(p0)));
        acc23 = _mm512_add_epi16(acc23, _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(p0, 1)));
        // subspace 2g+1 (hi nibble)
        let p1 = _mm512_shuffle_epi8(lut_z[2 * g + 1], hi);
        acc01 = _mm512_add_epi16(acc01, _mm512_cvtepi8_epi16(_mm512_castsi512_si256(p1)));
        acc23 = _mm512_add_epi16(acc23, _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(p1, 1)));
    }
    let mut t01 = [0i16; 32];
    let mut t23 = [0i16; 32];
    _mm512_storeu_si512(t01.as_mut_ptr() as *mut __m512i, acc01);
    _mm512_storeu_si512(t23.as_mut_ptr() as *mut __m512i, acc23);
    for i in 0..16 {
        out[i] = t01[i] as i32;          // block 0
        out[16 + i] = t01[16 + i] as i32; // block 1
        out[32 + i] = t23[i] as i32;      // block 2
        out[48 + i] = t23[16 + i] as i32; // block 3
    }
}

/// INTERLEAVED 64-wide fast-scan: same math as block_adc_i8_i16acc_avx512 but reads a SUPERBLOCK
/// whose group g is the 64 contiguous bytes [b0_g | b1_g | b2_g | b3_g] (16 codes from each of the 4
/// sub-blocks). This replaces the 4-load+3-insert lane-gather with ONE `_mm512_loadu_si512` per group
/// — the proper FastScan-512 layout, the best case for whether 512-bit pays on this uarch.
///   out[0..16]=sub-block0, [16..32]=1, [32..48]=2, [48..64]=3.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn block_adc_i8_i16acc_avx512_il(sblock: &[u8], m: usize, lut_z: &[__m512i], out: &mut [i32; 64]) {
    let mask = _mm512_set1_epi8(0x0f);
    let mut acc01 = _mm512_setzero_si512();
    let mut acc23 = _mm512_setzero_si512();
    for g in 0..m / 2 {
        let codes = _mm512_loadu_si512(sblock.as_ptr().add(g * 64) as *const __m512i);
        let lo = _mm512_and_si512(codes, mask);
        let hi = _mm512_and_si512(_mm512_srli_epi16(codes, 4), mask);
        let p0 = _mm512_shuffle_epi8(lut_z[2 * g], lo);
        acc01 = _mm512_add_epi16(acc01, _mm512_cvtepi8_epi16(_mm512_castsi512_si256(p0)));
        acc23 = _mm512_add_epi16(acc23, _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(p0, 1)));
        let p1 = _mm512_shuffle_epi8(lut_z[2 * g + 1], hi);
        acc01 = _mm512_add_epi16(acc01, _mm512_cvtepi8_epi16(_mm512_castsi512_si256(p1)));
        acc23 = _mm512_add_epi16(acc23, _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(p1, 1)));
    }
    let mut t01 = [0i16; 32];
    let mut t23 = [0i16; 32];
    _mm512_storeu_si512(t01.as_mut_ptr() as *mut __m512i, acc01);
    _mm512_storeu_si512(t23.as_mut_ptr() as *mut __m512i, acc23);
    for i in 0..16 {
        out[i] = t01[i] as i32;
        out[16 + i] = t01[16 + i] as i32;
        out[32 + i] = t23[i] as i32;
        out[48 + i] = t23[16 + i] as i32;
    }
}

/// Re-pack four 16-vector blocks into one interleaved 64-vector superblock (group g = b0_g|b1_g|b2_g|b3_g).
pub fn interleave4(b0: &[u8], b1: &[u8], b2: &[u8], b3: &[u8], m: usize, out: &mut [u8]) {
    for g in 0..m / 2 {
        out[g * 64..g * 64 + 16].copy_from_slice(&b0[g * 16..g * 16 + 16]);
        out[g * 64 + 16..g * 64 + 32].copy_from_slice(&b1[g * 16..g * 16 + 16]);
        out[g * 64 + 32..g * 64 + 48].copy_from_slice(&b2[g * 16..g * 16 + 16]);
        out[g * 64 + 48..g * 64 + 64].copy_from_slice(&b3[g * 16..g * 16 + 16]);
    }
}

/// Self-test: the 64-wide AVX-512 fast-scan must match block_adc_i8_scalar on all 4 sub-blocks for a
/// random i8 LUT (in [0,127], same as selftest_i8_fast) + four random packed blocks.
pub fn selftest_i8_fast_avx512(m: usize) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if !(std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw")) {
            return true;
        }
        let mut st = 0x9e3779b97f4a7c15u64;
        let mut rng = || { st ^= st << 13; st ^= st >> 7; st ^= st << 17; st };
        let lut: Vec<i8> = (0..m * 16).map(|_| (rng() % 128) as i8).collect();
        let bb = (m / 2) * 16;
        let mut blk = [vec![0u8; bb], vec![0u8; bb], vec![0u8; bb], vec![0u8; bb]];
        for b in blk.iter_mut() { for x in b.iter_mut() { *x = (rng() & 0xff) as u8; } }
        let regs = lut_regs_i8_z512(&lut, m);
        let mut a = [0i32; 64];
        unsafe { block_adc_i8_i16acc_avx512([&blk[0], &blk[1], &blk[2], &blk[3]], m, &regs, &mut a); }
        // interleaved-layout variant must give identical results
        let mut sb_buf = vec![0u8; (m / 2) * 64];
        interleave4(&blk[0], &blk[1], &blk[2], &blk[3], m, &mut sb_buf);
        let mut a2 = [0i32; 64];
        unsafe { block_adc_i8_i16acc_avx512_il(&sb_buf, m, &regs, &mut a2); }
        for sb in 0..4 {
            let mut s = [0i32; 16];
            block_adc_i8_scalar(&blk[sb], m, &lut, &mut s);
            for i in 0..16 {
                if a[sb * 16 + i] != s[i] {
                    eprintln!("selftest_i8_fast_avx512 MISMATCH m={m} block={sb} lane={i}: {} vs {}", a[sb * 16 + i], s[i]);
                    return false;
                }
                if a2[sb * 16 + i] != s[i] {
                    eprintln!("selftest_i8_fast_avx512_il MISMATCH m={m} block={sb} lane={i}: {} vs {}", a2[sb * 16 + i], s[i]);
                    return false;
                }
            }
        }
    }
    true
}

/// Per-subspace i16 LUT broadcast into a zmm (16 entries in lanes 0..15; vpermw index 0..15 picks
/// them). For the AVX-512 32-wide scan.
#[cfg(target_arch = "x86_64")]
pub fn lut_regs_i16_z(lut: &[i16], m: usize) -> Vec<__m512i> {
    (0..m).map(|s| unsafe {
        _mm512_castsi256_si512(_mm256_loadu_si256(lut.as_ptr().add(s * 16) as *const __m256i))
    }).collect()
}

/// AVX-512 int16-LUT ADC over TWO consecutive 16-vector blocks (32 vectors) at once: one vpermw
/// (_mm512_permutexvar_epi16) looks up 32 i16 values per subspace -> ~2x the AVX2 LUT16 throughput.
/// out[0..16] = block0's 16 vectors, out[16..32] = block1's. Bounded LUT -> i16 accum never saturates.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn block_adc_i16_avx512_x2(block0: &[u8], block1: &[u8], m: usize, lut_z: &[__m512i], out: &mut [i32; 32]) {
    let mask = _mm512_set1_epi16(0x0f);
    let mut acc = _mm512_setzero_si512(); // 32 x i16
    for g in 0..m / 2 {
        let c0 = _mm_loadu_si128(block0.as_ptr().add(g * 16) as *const __m128i);
        let c1 = _mm_loadu_si128(block1.as_ptr().add(g * 16) as *const __m128i);
        let codes = _mm256_inserti128_si256(_mm256_castsi128_si256(c0), c1, 1); // 32 bytes: b0 then b1
        let codes16 = _mm512_cvtepu8_epi16(codes); // 32 i16 in [0,255]
        let lo = _mm512_and_si512(codes16, mask);
        acc = _mm512_add_epi16(acc, _mm512_permutexvar_epi16(lo, lut_z[2 * g]));
        let hi = _mm512_and_si512(_mm512_srli_epi16(codes16, 4), mask);
        acc = _mm512_add_epi16(acc, _mm512_permutexvar_epi16(hi, lut_z[2 * g + 1]));
    }
    let mut tmp = [0i16; 32];
    _mm512_storeu_si512(tmp.as_mut_ptr() as *mut __m512i, acc);
    for i in 0..32 { out[i] = tmp[i] as i32; }
}

/// Self-test: AVX-512 x2 i16 ADC must match the scalar i16 reference on both blocks.
pub fn selftest_i16_avx512(m: usize) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if !(std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw")) {
            return true;
        }
        let mut seed = 0x9e37_1234u64;
        let mut nb = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1); (seed >> 33) as u8 };
        let b0: Vec<u8> = (0..(m / 2 * 16)).map(|_| nb()).collect();
        let b1: Vec<u8> = (0..(m / 2 * 16)).map(|_| nb()).collect();
        let cap = (30000 / m.max(1)) as i16;
        let lut: Vec<i16> = (0..m * 16).map(|_| (nb() as i16 % cap.max(1)).abs()).collect();
        let mut s0 = [0i32; 16];
        let mut s1 = [0i32; 16];
        block_adc_i16(&b0, m, &lut, &mut s0);
        block_adc_i16(&b1, m, &lut, &mut s1);
        let lz = lut_regs_i16_z(&lut, m);
        let mut o = [0i32; 32];
        unsafe { block_adc_i16_avx512_x2(&b0, &b1, m, &lz, &mut o) };
        for i in 0..16 { if o[i] != s0[i] || o[16 + i] != s1[i] { return false; } }
    }
    true
}

/// Self-test: AVX2 i16 ADC must match the scalar i16 reference (bounded LUT -> no saturation).
pub fn selftest_i16(m: usize) -> bool {
    let mut seed = 0x51ed_1234u64;
    let mut nb = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1); (seed >> 33) as u8 };
    let block: Vec<u8> = (0..(m / 2 * 16)).map(|_| nb()).collect();
    // positive LUT, scaled so full sum < 32767 (per-entry <= 30000/m)
    let cap = (30000 / m.max(1)) as i16;
    let lut: Vec<i16> = (0..m * 16).map(|_| (nb() as i16 % cap.max(1)).abs()).collect();
    let mut a = [0i32; 16];
    block_adc_i16(&block, m, &lut, &mut a);
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            let (lo, hi) = lut_regs_i16(&lut, m);
            let mut b = [0i32; 16];
            unsafe { block_adc_i16_avx2(&block, m, &lo, &hi, &mut b) };
            return a == b;
        }
    }
    true
}

/// SSE ADC scan: 16 points/block, `_mm_shuffle_epi8` LUT lookup, saturating i8 add. Writes 16
/// signed bytes into `out16`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn block_adc_sse(block: &[u8], m: usize, lut_regs: &[__m128i], out16: &mut [i8; 16]) {
    let mask = _mm_set1_epi8(0x0f);
    let mut acc = _mm_setzero_si128();
    for g in 0..m / 2 {
        let codes = _mm_loadu_si128(block.as_ptr().add(g * 16) as *const __m128i);
        let lo = _mm_and_si128(codes, mask);
        acc = _mm_adds_epi8(acc, _mm_shuffle_epi8(lut_regs[2 * g], lo));
        let hi = _mm_and_si128(_mm_srli_epi16(codes, 4), mask);
        acc = _mm_adds_epi8(acc, _mm_shuffle_epi8(lut_regs[2 * g + 1], hi));
    }
    _mm_storeu_si128(out16.as_mut_ptr() as *mut __m128i, acc);
}

/// Build the per-subspace LUT registers (broadcast 16-byte tables) from the flat signed LUT.
#[cfg(target_arch = "x86_64")]
pub fn lut_regs(lut: &[i8], m: usize) -> Vec<__m128i> {
    (0..m)
        .map(|sub| unsafe { _mm_loadu_si128(lut.as_ptr().add(sub * 16) as *const __m128i) })
        .collect()
}

/// Self-test: SSE block-ADC must match the scalar reference.
pub fn selftest(m: usize) -> bool {
    let mut seed = 0xabcd_1234u64;
    let mut nb = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 33) as u8
    };
    let block: Vec<u8> = (0..(m / 2 * 16)).map(|_| nb()).collect();
    let lut: Vec<i8> = (0..m * 16).map(|_| (nb() as i32 % 41 - 20) as i8).collect();
    let scal = block_adc_scalar(&block, m, &lut);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse4.1") {
            let regs = lut_regs(&lut, m);
            let mut out = [0i8; 16];
            unsafe { block_adc_sse(&block, m, &regs, &mut out) };
            for i in 0..16 {
                if out[i] as i32 != scal[i] {
                    return false;
                }
            }
            return true;
        }
    }
    true
}
