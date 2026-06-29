# sbann-rs results vs ScaNN (this box, msspacev-10M + text2image-10M)

Native Rust int8 IVF-PQ engine. All numbers measured **same-box, 8 threads, best-of-N**, against
ScaNN (the open-source #2 on the neurips23 OOD leaderboard) running `search_batched_parallel`,
`taskset -c 0-7`. The contended shared box makes absolute QPS swing ±30% with load, so comparisons
are **same-window / back-to-back**; ratios at matched recall are the meaningful figure.

## Headline

| dataset | metric | sbann-rs (stack) | ScaNN (same box) | verdict |
|---|---|---|---|---|
| msspacev-10M (int8, L2, in-distribution) | QPS @ recall@10 ≥ 0.90 | **SOAR + fast-scan + AVX-512** | baseline | **STACK BEATS ScaNN ~1.5× at QPS@90%** (conservative; bracketed both load orders, up to ~3×); ≥0.95 competitive |
| text2image-10M (float32, MIPS, OOD — the real leaderboard) | QPS @ recall@10 ≥ 0.90 | ~5k (fast-scan IP) | ~9.8k | **ScaNN ~2× faster** (stack test pending) |

**The msspacev win (P116/P117):** three stacked levers turned the corrected ~1.5–2× *deficit* into a
~1.5× *lead* at QPS@90%. (1) **Fast-scan** (`SBANN_FASTSCAN`, P113) → int16-accuracy candidate selection
at i8 op-count, 1.6–1.9× scan; brought us to parity. (2) **SOAR-spill** (`SBANN_SOAR`, P116) → build-time
multi-assignment covering each cell's residual directions → ~25–32% fewer probes at matched recall. (3)
**AVX-512 64-wide scan** (`SBANN_USE512FS`, P116) → 1.75× scan via in-lane `_mm512_shuffle_epi8` +
interleaved superblock layout (Zen 4 has no AVX-512 downclock). Confirmed beating ScaNN at QPS@90% in
*both* load orders (the conservative 1.5× is from the run where the stack was load-*disadvantaged* and
still won). This is the legitimate result — contrast the earlier load-confounded false claim (P110).

**Fast-scan kernel (P113) roughly halved the msspacev gap.** A FAISS/Quick-ADC-style scan (int8 LUT,
1 vpshufb/subspace + int16 widening accumulation; `SBANN_FASTSCAN`) is selftest-validated and **1.6–1.9×
faster than the int16 LUT16 scan at IDENTICAL recall** (per-subspace min-subtraction spends int8 on
within-subspace variation → ~12–13-bit ranking, enough since rerank fixes final order). Same-window vs
ScaNN, **bracketed both load orders**: QPS@90% is now **parity** (FASTSCAN 1.23× ahead running second,
ScaNN 1.15× ahead running first → ~1.0× ± load noise), down from ~1.5×; recall-0.95 ScaNN ~1.4× (down
from ~2.4×). So on the leaderboard metric (QPS@recall≥0.90) the engine is competitive with ScaNN on
msspacev; ScaNN's remaining edge is at recall ≥0.95 (AH scan throughput at large candidate pools).

**MAJOR CORRECTION (2026-06, supersedes the earlier "we win msspacev" headline).** Two independent
clean same-window head-to-heads (`h3_vs_scann.log`, `h3_vs_scann2.log`) against a properly-configured
parallel ScaNN (`num_leaves=4000`, `score_ah(2)`, `reorder(200)`, `search_batched_parallel`, taskset
0-7) show ScaNN **beating** our engine across the whole frontier: at recall 0.923 ScaNN ~15,570 QPS vs
hierk3 ~6.5–9k; ScaNN also 0.95@12.7k, 0.98@6.3k (we reach 0.95 only at ~1.5–2.8k). The prior
"1.26–1.45× faster" msspacev claim was **load-confounded** — exactly the error class flagged for OOD in
P108, which I failed to apply to msspacev. Tmul tuning does NOT close it: msspacev recall is
tmul-insensitive (int16 LUT ranks well; the 1000-survivor rerank floor already captures the NN), so
fewer survivors don't buy QPS. ScaNN's edge is architectural: anisotropic **2-byte** AH + in-register
scan + exact reorder of only ~200 candidates, vs our 4-bit PQ that needs ≥1000 reranked.

What still holds: **hierk3 is a real improvement over our OWN 2-level engine** (2.2× faster build,
~1.15× faster queries, validated both load orders). Combined with the rerank-floor fix and the fast-scan
kernel below, the engine reaches **parity with ScaNN at QPS@90% on msspacev** (bracketed both load
orders) — competitive on the leaderboard metric, though ScaNN still leads ~1.4× at recall ≥0.95 and ~2×
on OOD. AVX-512 `vpermw` scan was a dead end (downclock).

**Champion config (msspacev):** `hierk3` Kf=262144 C0=1024 C1=8192 b0=48 b1=160 a0=3 `apq4`, env
`SBANN_FASTSCAN=1` (fast-scan kernel) + low rerank floor (t_surv≈p·3). This is the parity-with-ScaNN config.

**Rerank-floor fix (real ~2× QPS@90% engine win, banked).** The hardcoded `t_surv = max(p·tmul, 1000)`
floor was over-conservative — the int16 LUT ranks well enough that shallow rerank (`t_surv ≈ p·3 ≈
256–480`) holds recall (p128: 0.9194 @ t_surv=256 vs 0.9227 @ 1024, −0.3%) while ~doubling QPS@90%
(~6k → ~12k). Default floor lowered 1000 → 300 (`SBANN_TFLOOR` overrides). This narrowed the msspacev
QPS@90% gap from ~2× to ~1.5× — but ScaNN still leads, and the gap **widens with recall** (scan
throughput: ScaNN's anisotropic AH scans more candidates faster). The remaining gap is the **scan
kernel**, not routing or rerank depth — that's the lever that would actually close it.

## Best configs

**msspacev-10M (QPS@90% champion):** `hierk3` 3-level router, Kf=262144, C0=1024, C1=8192, b0=48,
b1=160, a0=2, compressor `apq4` (anisotropic 4-bit PQ), int16 LUT scan (default), tmul≈3. This beats
the old 2-level `hierk` C0=4096 champion by ~1.15× QPS at matched recall **and builds 2.2× faster**
(347s vs 767s @10M) — the 3-level hierarchy routes in O(Kf^⅓) centroid-distances instead of O(Kf^½),
which both speeds queries and is the scale lever for the 100M/1B tracks. For recall ≥ 0.95 raise b0/b1/a0.

**text2image-10M (OOD):** quantize float32→int8 (no augmentation), **cosine routing + asymmetric
MIPS-PQ scan + exact inner-product rerank**. `hierk` Kf=262144 C0=4096 b0=128 a0=3 `apq4`, tmul≈8,
env `SBANN_IP=1`. (MIPS-augmentation path also works but is ~5× slower; cosine+IP is the winner.)

## The decisive ideas (in order of impact)

1. **int16 LUT scan** (`pq::block_adc_i16_avx2`, LUT16 trick) — the int8 PQ-ADC LUT had only ~8-bit
   distance resolution (scaled to avoid saturating-int8 sum), forcing reranking ~40% of the pool;
   int16 gives ~15-bit resolution → shallow rerank at higher recall. Biggest internal engine win (vs our
   own i8-LUT baseline); did NOT surpass ScaNN (see corrected headline).
2. **cell-contiguous rerank store** (`Index.raw`) — raw i8 vectors in cell order so rerank reads the
   small probed-cell region (cache-warm) instead of scattering across the base. ~1.15×.
3. **anisotropic apq4** quant — ~1.08× (recall-per-bit).
4. **asymmetric MIPS-PQ scan** (`query_lut_f32_i16_ip`) — IP-aware candidate selection for OOD →
   halved rerank depth (tmul 16→8) at zero extra scan cost.
5. **3-level hierarchical router** (`train_hkmeans3`, `hierk3`) — coarse→mid→fine routing in
   O(Kf^⅓) centroid-distances instead of O(Kf^½). At 10M: ~1.15× faster queries at matched recall
   **and 2.2× faster build** vs the 2-level champion. The faster build is the enabler for 100M/1B.
6. routing fan-out tuning (C0/b0/b1), `needs_raw_rows` scan guard.

Dead ends (documented): AVX-512 vpermw scan (downclock), VNNI int8 dot (rerank is bandwidth-bound,
not compute-bound), PCA dim-reduction (text2image near-isotropic), query-aware routing.

## Reproduce

```bash
cd sbann-rs && cargo build --release
D=../big-ann-benchmarks/data/MSSPACEV1B; B=$D/spacev1b_base.i8bin.crop_nb_10000000
# msspacev QPS@90% champion (hierk3 3-level router, best-of-3):
SBANN_C0=1024 SBANN_C1=8192 SBANN_B0=48 SBANN_B1=160 SBANN_PLIST=128,160,192 SBANN_NQ=1000 \
  SBANN_REPS=3 RAYON_NUM_THREADS=8 \
  ./target/release/sbann run $B $D/query.i8bin $D/msspacev-gt-10M hierk3 apq4 3 262144 30
# OOD text2image (after prep_ood_simple.py writes t2i10m*.i8bin):
T=../big-ann-benchmarks/data/text2image1B
SBANN_IP=1 SBANN_C0=4096 SBANN_B0=128 SBANN_PLIST=320,384 SBANN_TMUL=8 SBANN_REPS=5 SBANN_NQ=10000 \
  RAYON_NUM_THREADS=8 ./target/release/sbann run $T/t2i10m.i8bin $T/t2i10m_query.i8bin $T/text2image-10M hierk apq4 3 262144 30
```

Full experiment ledger + measurement discipline (best-of-N, same-window, drift-free interleave) in
`../FINDINGS.md` (P63–P105). The actual leaderboard #1 (hanns, ScaNN, pinecone, zilliz on Azure
D8lds_v5) requires a submission on their hardware; these are the algorithm/primitive results feeding it.
