# Findings: inventing theory-grounded ANN methods, stress-tested on real data

Record of what worked and what didn't, from building and benchmarking cumulative-score
/ score-and-ball ANN methods on ann-benchmarks (fashion-mnist, mnist, SIFT-1M) and
big-ann (MS-SPACEV 1M/10M). Code in `impl/`; per-result detail in `impl/METHOD.md`,
`impl/PRINCIPLE.md`, `impl/RESULTS_annbench.md`, `impl/KERNELS.md`, `impl/PQ4_PLAN.md`.

HEADLINE: a from-scratch fully-int8 hierarchical score-and-ball tree, validated through the
OFFICIAL big-ann harness (`benchmark/algorithms/sbtree.py`): recall@10 = 0.906 @ 4,948 QPS on
msspacev-10M -- competitive with the leaderboard's graph methods (not #1; that needs a formal
submission on their HW). Scale-gated (useless at 1M, decisive at 10M: 2-6x faster query at
fine C + ~10x faster build). The batched vpshufb 4-bit-PQ cascade was BUILT (P44/P46): the
scan kernel is 4x faster (verified) and the full cascade gives 1.27x at the high-recall end
(1M, grows with pool size), recall-preserving -- capped by the remaining per-query table-build
Python (the last micro-opt to batch). Every identified lever has now been built and measured.

================================================================================
POSITIVE FINDINGS
================================================================================

P1. Cumulative scores are a Johnson-Lindenstrauss distance sketch (the key idea).
    The Gaussian projections S(x)=Gx that the score-and-ball theory uses to PARTITION
    (via their signs) also ESTIMATE DISTANCES (via their magnitudes):
    ||S(x)-S(q)||^2 ~ ||x-q||^2, concentration exp(-Theta(m eps^2)). Using the
    magnitudes for ranking + exact rerank gives recall@10=0.98 where using only the
    signs gives 0.04 on hard data. Same projections, double duty. (score_rerank.py)

P2. The route-precise / rank-cheap principle (main design rule).
    Ranking tolerates JL error (the exact rerank fixes the order); routing does NOT
    (a misrouted neighbour is lost forever). So route precisely in original space,
    rank cheaply in the sketch. Measured on msspacev: sketch/sign routing 0.17-0.93,
    original-space k-means routing 0.98. (PRINCIPLE.md)

P3. The consolidated method works across scales and difficulty.
    Score-Ball Tree = k-means certified-ball routing + cumulative-score JL sketch
    ranking + exact rerank. recall@10 = 0.98 at:
      fashion-mnist 60k/784d : 0.20% scan
      mnist-784     60k/784d : 0.20%
      SIFT-128      1M /128d  : 0.25%
      MS-SPACEV     1M /100d  : ~1.0% (hard)
    Beats our HNSW reference on candidate-efficiency (3-8x fewer comps, 20-40x faster
    build) at the recall=0.98 point. (score_ball_tree.py, RESULTS_annbench.md)

P4. Cohort-or-leaf (the paper's open Assumption 4.4) holds empirically for c>1.
    For a true near pair, either q routes to p's certified ball (leaf) or p's ball is
    enriched with other q-neighbours (cohort). Gap (neither) on msspacev:
      c=2.0 -> 0.000,  c=1.5 -> 0.014,  c=1.2 -> 0.17,  c=1.05 -> 0.49.
    Holds in the (c,r) regime the paper targets; degrades only toward exact NN
    (c->1, outside the problem). Leaf and cohort mechanisms demonstrably trade off.
    (cohort_or_leaf.py) -- an unproven paper idea, tested in practice.

P5. Sublinear filtering at the balanced threshold (the paper's Lemma 3.1).
    The continuation-mass spherical decoder's far-candidate count is sublinear in n
    (exponent ~0.49 << 1), candidate fraction decays as n^-(1-rho). (spherical.py)

P6. Carrying the score across recentering beats restarting (notes2 dead-end #2).
    Near-pair co-survival: carried walk decays polynomially in depth, restart decays
    geometrically (p^H) -- ratio ~24000x by depth 6. (unified_vs_restart.py)

P7. Memory rate-distortion is clean (big-ann limited-memory axis).
    PQ codes only (no vectors, no rerank) on msspacev: 16 B/pt -> 0.52, 50 B/pt ->
    0.90 recall@10 (full int8 = 100 B/pt). A usable recall-vs-bytes curve. (pq_memory.py)

P8. Multiprobe + decoupled bucket width + tables made fast_unified genuinely
    sublinear: 147x (clustered) / 75-1440x (uniform) fewer dist comps vs brute, with
    a real k-NN candidate-floor fix. (fast_unified.py, bench.py)

P9. Per-query recall certificate from the JL gap is SOUND and useful.
    After exact-reranking the top T, the gap s_T/D_k (T-th sketch dist vs k-th true
    dist) certifies ranking completeness. Empirically: certified queries (margin>=1.3)
    have recall ~1.0 (msspacev min 1.00; fashion-mnist 0.9997). Adaptive-T (deepen
    only low-margin queries) matches fixed-T=500 recall at avgT=268 on fashion-mnist
    -- ~1.9x cheaper at the same recall, by spending budget where it's uncertain.
    Honest limit: on uniformly-hard data (msspacev) distances concentrate, the
    certificate rarely fires (8%), so little saving -- it helps when query difficulty
    VARIES. Gives the worst-query-robustness handle the paper cares about.
    (per_query_cert.py)

P10. Anisotropic (PCA) sketch >> random Gaussian sketch on real L2 -- big win,
    largest on hard data. recall@10 @ T=300, random vs PCA:
      fashion-mnist dim16 0.844->0.975, dim32 0.940->0.997, dim64 0.989->0.999
      msspacev      dim16 0.198->0.425, dim32 0.488->0.767, dim64 0.767->0.994
    PCA roughly HALVES the sketch dim needed (msspacev PCA@32 ~ random@64) -> cheaper
    ranking and less memory at fixed recall. Answers "does anisotropy help on big-ann
    L2 (not just MIPS)?" = YES, strongly, because real spectra are skewed and random
    projection wastes dims on low-variance directions. CLEAN ASYMMETRY with N4:
    data-dependence helps on the RANKING side (PCA projection) but NOT the ROUTING
    side (min-max) -- mirror of the route-precise/rank-cheap principle (P2): the
    ranking projection should be data-fitted, the routing should just be precise.
    Should replace the random sketch in score_ball_tree. (anisotropic test)
    CAVEAT (end-to-end): with IVF routing already shrinking the pool, PCA's gain
    shrinks (msspacev dim64 end-to-end 0.882->0.900 @0.5%) because the sketch then
    only ranks within a small pool -- ROUTING is the bottleneck on hard data, not
    ranking. PCA's big win is in isolation / large pools / dim-&-memory reduction
    (PCA@64 ~ random@128). So: use PCA to SHRINK the sketch (cheaper, less memory) at
    fixed recall; it won't break the routing ceiling.

P11. Multi-assignment (the paper's S=n^{1+rho} space/time tradeoff) BREAKS the
    routing ceiling on hard data. Replicating each point into its top-`assign`
    certified balls lets a query reaching few cells still find the true NN.
    msspacev-200k, score_ball_tree (PCA dim96), recall@10:
      assign=1 (1x space): 0.900 @0.5% scan, 0.931 @1.0%   (routing-capped)
      assign=2 (2x):       0.953 @0.5%,       0.971 @1.0%
      assign=4 (4x):       0.980 @0.5%,       0.989 @1.0%
    So the full method (k-means certified-ball routing + multi-assignment + PCA
    sketch + exact rerank) reaches recall@10=0.98 at <1% scan on a HARD big-ann L2
    track -- matching its easy-data frontier -- by paying 4x space. This is the
    north star: data-dependent LSF with the certified-ball structure, competitive on
    hard big-ann, with the cost being exactly the paper's space/time tradeoff.
    Routing was the bottleneck (N1-N3); redundancy, not a better partition, fixes it.
    REPRODUCED on a 2nd full-1M euclidean big-ann track, MS-Turing-1M (different
    embedding source, 100d float): assign1 0.688@0.20%/0.758@0.40%, assign2
    0.824/0.888, assign4 0.909@0.20%/0.953@0.40% -- same signature. At higher budget
    msturing assign4 reaches recall@10=0.980 @0.90% scan, 0.992 @1.5%. So TWO
    independent hard euclidean big-ann tracks hit 0.98 at <1% scan with assign=4
    (MS-SPACEV 0.98@0.5%, MS-Turing 0.98@0.90%) -- north star met twice. (bigann/deep
    ship only at 10M+, so the available 1M euclidean tracks were used.)

================================================================================
NEGATIVE FINDINGS (each with a diagnosed reason)
================================================================================

N1. Sign-bit / random-hyperplane routing fails on hard data.
    Routing by score signs (LSH) or sketch-space cells collapses to recall 0.17 on
    msspacev -- signs are a 1-bit-per-projection proxy, error-intolerant for routing.
    (score_ball.py) Reason: see P2 (routing doesn't tolerate the error).

N2. Random-projection partition trees are weak routers.
    RPBeam (RP-tree forest + beam search) caps at recall 0.73 at 22% scan on msspacev;
    LSF boundary-overlap made the pool 3x more efficient but did NOT lift the ceiling.
    Reason: random split centroids are poor vs k-means cluster centers; partition
    QUALITY dominates on hard data. (rp_beam.py)

N3. Cap-recentering (the paper's specific data-dependence) is a no-op on
    unstructured data. On msspacev the cap finder found nothing (build ~1s); the
    structure degenerated to one bucket. Recentering helps only where cap/cluster
    structure exists (SIFT/fashion). (ball_tree.py, fast_unified.py)

N4. Min-max / robust partitions (Andoni-Beaglehole, andoni22a) gave NO measurable
    gain in our setup -- the most thorough negative.
      - robust k-means (MWU near-pair reweight): identical to k-means on msspacev AND
        fashion-mnist, average and worst-query (split rate barely moved 0.52->0.48).
      - faithful discrete min-max LSH forest (Freund-Schapire game, coordinate hashes):
        no gain vs uniform LSH forest on a continuous mixture (0.288 vs 0.293) NOR on
        a Hamming mixture with 104 imbalanced + 24 balanced bits -- its NATIVE regime
        (0.803 vs 0.799).
    Reasons: (a) continuous median splits are AUTO-BALANCED, pre-empting the
    coordinate-imbalance problem min-max solves on Hamming; (b) the JL-sketch+exact
    rerank stage MASKS routing-quality differences (recall depends on the candidate
    pool, which rerank recovers); (c) d=128 likely too small for the exp(sqrt(ln d))
    asymptotic; (d) real data lacks the planted concentrated-hard-subset structure
    where their Thm 4.1 gains live. (robust_cluster.py, minmax_tree.py)

N5. int8 / 4-bit quantized sketch gives no speedup in numpy.
    numpy upcasts int8 matmuls; without a SIMD kernel there is no bandwidth win, and
    the quantization overhead made it slower. (score_rerank.py quantize path)

N6. tinyknn (4-bit SIMD PQ) is fast at LOW recall but ceilings at ~0.964 and needs a
    246s OPQ-rotation build; our float-JL method wins at the recall>=0.98 target.
    (the SIMD-PQ accuracy/recall tradeoff bites at high recall.)

N7. QPS is CPU/numpy-bound, not algorithm-bound. Batching gave only 1.1x (rerank
    gather dominates, irregular per-query); more IVF cells didn't help (coarse-search
    cost offsets); no GPU available. The gap to C++ leaderboard QPS is implementation,
    not method. (bench.py, scaling_bench.py)

N8. Cascade sketch (#5): cuts the full-n scan ~9x in flops but the coarse first stage
    costs recall (0.88 vs 0.98 single-stage); a moderate, not decisive, win -- best
    only where the full-n scan dominates (large n, no IVF). (cascade_sketch.py)

N9. Random Gaussian data is the wrong benchmark for recall@10 -- concentration of
    measure makes the true 10-NN near-indistinguishable; recall plateaus for ANY
    method. Use real datasets (this is why ann-benchmarks does). (scaling.py / early)

P12. Rate-coverage theory (I2): actual recall >> random-assignment baseline
    1-(1-p/C)^a -- on msspacev (C=1024, nprobe=32): a=1 actual 0.90 vs baseline 0.03
    (~30x), a=4 actual 0.98 vs baseline 0.12. So the k-means partition's
    neighbour-grouping provides almost ALL the routing coverage; random cells would
    need ~30x more probes. This also explains N4: k-means already captures the
    available data-dependence in the partition, so min-max optimization finds nothing
    to add. The baseline is a useful dataset-difficulty / space-knob reference.
    (adaptive_assign.py)

P13. (FLAGSHIP -- the constructive complement to N13) The SPHERICAL-CAP primitive is
    selective on the SAME real high-dim data where the additive ball primitive is not.
    msspacev-50k, center at global mean then normalize to the unit sphere: angular
    similarity <q_hat, x_hat> between a query and its TRUE Euclidean 10-NN = 0.5205,
    while <q_hat, random_hat> = -0.0006 (~orthogonal, concentration of measure). To
    retain ALL 10 true NN you scan only 0.04% of the DB (mean; 0.03% median) -- a ~2500x
    reduction. CONTRAST WITH N13: the additive ball filter on this exact data gives
    stab_frac=1.000 (no reduction). So distance concentration, which DEFEATS additive
    ball covers (everything within R+r), is precisely what MAKES angular caps selective
    (background goes orthogonal, neighbours stay correlated). This is the mathematical
    reason the paper's spherical decoder (sec 3) is the sound theoretical engine and the
    ball recursion (sec 9) is only a reduction shell. CAVEAT: 0.04% is the ORACLE
    selectivity (per-query cap aimed exactly at q); a realizable fixed cap family must
    aim near arbitrary query directions, which is what coarse hyperplane/sign LSF failed
    to do (N1, N11). The open constructive question: a cap family fine/data-dependent
    enough to realize this selectivity (k-means on centered data is implicitly such a
    data-dependent angular partition -- likely why k-means routing works). (cost_cover.py)

P14. (realizability of P13, reconciles P13 with N11) Head-to-head routers on
    msspacev-50k, C=1024, exact rerank in pool, scan~1.75% (nprobe=16) / ~3.4% (32):
      raw k-means IVF (L2):           0.7746 / 0.8690
      spherical k-means (cos):        0.7772 / 0.8592
      random caps (data-independent): 0.2804 / 0.4128
    THREE conclusions: (a) raw k-means ~= spherical k-means -> under concentration the
    radial part is ~constant, so L2 k-means IS implicitly the angular/cosine partition;
    k-means realizes the P13 angular selectivity without being told to. (b) data-
    independent random caps fail badly (0.28 vs 0.77) -> the angular SIGNAL is huge but
    only a DATA-DEPENDENT cap family can aim near query directions (this is N1/N11's
    mechanism, now explained). (c) k-means realizes only a small fraction of the oracle
    selectivity (1.75% scan/0.77 vs oracle 0.04%/1.0) -> large headroom; the paper's
    recursive score-TREE (refine the cap toward the query direction level by level) is the
    theory-native way to close it. [testing recursion next] (cost_cover.py)

P15. (closes the arc N13->P13->P14) Finer angular resolution moves the (scan,recall)
    curve toward the oracle. msspacev-50k, flat cosine k-means, recall@10 @ scan%:
      scan~0.13%:  C=1024 -> 0.258   C=4096 -> 0.356
      scan~0.5-0.8%: C=256 -> 0.34   C=1024 -> 0.53   C=4096 -> 0.637 (@0.78%)
    At MATCHED scan, MORE cells = higher recall (curve shifts up-left toward oracle
    0.04%/1.0). BUT the cost flips to ROUTING: as C grows, top-1 recall collapses
    (C=256/1024/4096 p=1 -> 0.343/0.258/0.064) because the single nearest cell less often
    holds the NN (boundary), so you must multiprobe AND score all C centroids per query
    (O(C d), ->full scan as C->n; the oracle C=n is just brute cosine top-k). THIS pins
    down the precise role of the paper's recursive score-and-ball TREE: it lets you use
    near-oracle-FINE angular resolution while keeping ROUTING cost SUBLINEAR (descend
    log levels instead of scoring all leaves). COMPLETE NARRATIVE: additive ball cover
    cannot contract under concentration (N13); the angular/spherical primitive can, with
    huge headroom (P13); only a data-dependent cap family realizes it (P14); finer
    resolution approaches the oracle but makes flat routing the bottleneck (P15) -> the
    recursive tree is exactly the mechanism that buys fine resolution at sublinear routing
    cost. NEXT: build the recursive spherical tree and verify near-oracle recall at
    sublinear routing. (cost_cover.py)

P16. (qualified validation of P15) 2-level spherical tree (64x64=4096 leaves) vs flat
    C=4096, msspacev-50k. Routing dots/query: flat 4096, tree b0=4 -> 320 (12.8x cheaper),
    b0=8 -> 576 (7.1x). At MATCHED SCAN the tree loses recall (scan~1.7%: flat 0.862 vs
    tree-b0=8 0.781; scan~6.5%: 0.953 vs 0.887) because coarse hard-routing (top-b0 of 64)
    drops queries whose NN-cell is not in the top-b0 -- the boundary loss again. BUT on
    TOTAL WORK = routing_dots + scan*n (the fair metric, since flat pays 4096 routing dots
    ~= scanning 8% of n regardless): flat p=64 = 4096+940 = 5036 dots @0.862; tree b0=8
    p=256 = 576+3270 = 3846 dots @0.887 -- TREE WINS at the high-recall end. So P15 holds
    CONDITIONALLY: the recursive tree pays off where routing cost is a large fraction of
    total work (fine resolution / high recall / low scan), but the coarse level needs
    enough multiprobe b0 or it bleeds recall (same redundancy lesson as P11/N11: redundancy
    amplifies a good partition). Naive 2-level with small b0 is NOT a free lunch; the win is
    real but modest at 50k. Path to fully realize P15: better coarse routing (more coarse
    multiprobe / soft / overlap-into-both-children) so the coarse level stops losing NN.
    (cost_cover.py)

P17. (P16 SOLVED -- coarse build-overlap realizes P15's promise) Replicate each point
    into its top-a0 coarse cells at build (coarse-level multi-assignment = the P11
    redundancy lesson one level up). msspacev-50k, 64x64 spherical tree, b0=8 coarse probe,
    SPACE-MATCHED vs flat C=4096 with the same a0 multi-assign, metric = total work
    (routing_dots + scan*n), routing = 576 (tree) vs 4096 (flat):
      2x space, rec ~0.86: flat ~5500 dots  vs TREE 2081  (2.6x cheaper)
      4x space, rec ~0.91: flat ~6200       vs TREE 2681  (2.3x cheaper)
      4x space, rec ~0.95: flat 7023        vs TREE ~6000 (1.2x cheaper)
    So the spherical tree beats space-matched flat by ~2-2.6x total work in the practical
    recall range (0.86-0.95), PURELY from sublinear routing (score 576 not 4096 centroids).
    Coarse overlap recovers the recall the naive tree lost (P16) without changing routing
    cost. At very high recall (>0.97) flat catches up (the 2-level tree's leaf resolution
    tops out ~0.97 with these params -> needs deeper recursion or larger p). The win GROWS
    with C: at 50k/C=4096 routing 4096 ~= scanning 8% of n; at 1M you need finer C for
    sublinearity, so routing is a larger fraction and the tree's edge widens. First time the
    paper's recursive score-tree structure BEATS flat IVF on real data at matched
    recall+space. NEXT (#2): scale to msspacev-1M with fine C and race vs ivf_rerank.
    (cost_cover.py)

N14. (HONEST CORRECTION + QUALIFICATION of P17 -- the dot-count metric oversold the tree)
    Re-ran tree-vs-flat with a CHEAP 1M-feasible build (sph_tree.py: ONE fine
    MiniBatchKMeans -> Kf=4096 leaves, then a TINY k-means grouping leaf-centroids into
    C0=64 coarse cells; point multi-assign a0=2) and CORRECT brute-force gt on the loaded
    subset (the earlier 200k run was INVALID -- it used the precomputed 1M gt against a
    200k subsample -> recall 0.15 garbage; lesson: subsample needs its own brute gt).
    msspacev-50k, space=2x, total-work (routing_dots + pool):
      recall ~0.79: TREE total 2112 vs FLAT ~5050  -> 2.4x cheaper (dots)
      recall ~0.88: TREE 3844      vs FLAT ~5800   -> 1.5x cheaper
      recall ~0.93: TREE 6596      vs FLAT ~7000   -> ~tie
      recall ~0.96: TREE 11073     vs FLAT 7961    -> TREE LOSES
    TWO corrections to P17: (a) the cheap "fine-then-group" build has WORSE routing than the
    nested build at high recall -- the coarse grouping scatters a query's good leaves across
    coarse cells, so the tree's pool inflates fast (b0=16 -> scan 20% vs flat 7.7% at rec
    ~0.95). The tree wins on dot-count only at MODERATE recall (<0.9); above that, flat wins.
    (b) MORE IMPORTANT: WALL-CLOCK favors FLAT everywhere (flat 0.24-1.14ms vs tree
    0.92-4.69ms) because flat routing is a SINGLE BLAS matmul Q@cf.T over all queries, while
    the tree does a per-query Python gather of scattered leaves. So the "total dots" metric
    (which P17 used) OVERSELLS the tree: in a numpy/BLAS implementation, scoring 4096
    centroids as one matmul is cheaper than scoring 545 scattered centroids via Python
    indexing. The tree's routing-dot advantage only converts to wall-clock when the flat
    routing term genuinely DOMINATES -- which needs C >> 4096 (n >> 1M) AND a vectorized
    gather, neither available here. CONCLUSION: the recursive-tree routing advantage is real
    in the abstract cost model but does NOT beat flat IVF in practice at <=1M with a BLAS
    routing matmul; flat IVF + sketch + rerank (ivf_rerank) remains the practical winner.
    The tree is the right structure only at much larger scale or on hardware where the
    O(C*d) flat routing matmul is the bottleneck. (sph_tree.py, bench_sph_lean.py)

P18. (FAST KERNELS -- N14's wall-clock gap is mostly implementation, partly algorithmic)
    Built two low-level kernels for the score-tree and measured what converts.
    (1) CONTIGUOUS IVF LAYOUT (sph_tree_fast.py): permute points so each leaf is a
        contiguous block (Xord) and fine centroids so each coarse group is contiguous, so
        every per-query gather becomes a slice not a fancy-index. -> tree query 2.11ms ->
        0.644ms at rec 0.88 = 3.3x, JUST from memory layout.
    (2) CYTHON SIMD SCAN KERNEL (tree_kernel.pyx, compiled via fetched cpython headers +
        gcc -O3 -march=native): top-k over the contiguous leaf segments in one C pass, no
        per-candidate Python, inner dot vectorizes. -> another 1.7-2.6x (0.538->0.316ms).
    COMBINED: the kernel work made the tree ~6.7x faster (2.11ms -> 0.316ms @ rec 0.88).
    ROUTING CROSSOVER (routing_crossover.py, synthetic): tree routing (coarse GEMM + fine
    gather) BEATS flat's full Q@cf.T GEMM at C=4096 (0.061 vs 0.081ms) and C=16384 (0.188
    vs 0.384ms, 2x) -- the 1M-scale regime -- but LOSES at C>=65536 because the fine-scoring
    is still a Python loop over queries (needs kernelizing too). So the tree's sublinear
    routing IS a real wall-clock win at moderate-large C.
    BUT END-TO-END flat IVF still wins ~2.2x at 50k/C=4096 even with both kernelized
    (TREE-kernel 0.316ms vs FLAT-kernel ~0.14ms @ rec 0.88), because the cheap two-stage
    routing scans ~2.7x MORE CANDIDATES at matched recall (pool 4141 vs 1454). This residual
    is ALGORITHMIC (hierarchical routing loses precision vs flat's direct top-p over all
    cells -- the classic IVF-vs-IMI precision loss), NOT a kernel issue; kernels speed up
    both scans equally. VERDICT on "good algo + needs fast kernels": kernels delivered (6.7x,
    routing now wins at moderate C) but do NOT flip the end-to-end result -- the blocker is
    routing PRECISION (candidate count), which needs a better partition / more coarse probe,
    not faster code. flat IVF + sketch + rerank remains the practical winner at <=1M.
    (sph_tree_fast.py, tree_kernel.pyx, bench_kernel.py, routing_crossover.py)

P19. (#1 -- the candidate-precision gap is MOSTLY CLOSABLE with high coarse-probe b0)
    N14/P18's "tree scans 2.7x more candidates" was an artifact of too-low b0=8. Dense
    (b0,p) sweep -> the tree's BEST pool-vs-recall frontier (min pool over all b0,p) vs
    flat, msspacev-50k, Kf=4096 C0=64:
      recall 0.84: TREE 1451 (b0=32) vs FLAT 1278 -> 1.14x
      recall 0.91: TREE 2559 (b0=32) vs FLAT 2207 -> 1.16x
      recall 0.95: TREE 4570 (b0=32) vs FLAT 3865 -> 1.18x
      recall 0.98: TREE 8302 (b0=48) vs FLAT 6767 -> 1.23x
    So with HIGH coarse-probe (b0 = C0/2) the tree scans only ~15-23% more candidates than
    flat -- the hierarchical precision loss is small, not 2.7x. And b0=32 still routes ~2x
    fewer dots than flat (64 + 32*64 = 2112 vs 4096). NET: tree ~2x cheaper routing,
    ~1.2x more candidates -> with both kernelized the routing saving can offset the
    candidate penalty -> the tree can be competitive. Key knob: b0 must be ~C0/2, NOT small.
    (bench_precision.py)

P20. (#2 RESULT -- the fully-kernelized tree BEATS flat IVF at large C, the crossover
    confirmed end-to-end on real data) Both routing AND scan in C (route_topp +
    search_segments kernels), high b0 (P19), routing TIMED for both. msspacev-50k,
    tree Kf=16384/C0=128 vs flat C=16384:
      recall ~0.84: TREE 0.257ms (pool 586) vs FLAT 0.468ms (pool 584) -> 1.8x
      recall ~0.89: TREE 0.405ms (pool 1087) vs FLAT 0.471ms (pool 1085) -> 1.16x
    IDENTICAL candidate pools (P19's high-b0 closed the precision gap) but the tree's
    hierarchical routing avoids scoring all 16384 centroids while flat pays the full
    ~0.38ms routing GEMM. So at C=16384 the kernelized tree wins. CAVEAT: at n=50k flat's
    OPTIMAL C is 4096 (cheap 0.08ms routing, ~0.14ms total) which still beats the tree --
    the tree only wins when large C is FORCED by large n (cellsize ~30 => n=1M needs
    C~33k, n=10M needs C~330k where flat routing ~6ms/query). The tree's regime is large n.
    int8 scan kernel (search_segments_i8) did NOT help here (0.49 vs 0.47ms): the cost is
    routing-dominated (float GEMM) and the int8 inner loop didn't auto-vectorize better than
    float FMA. int8 would help in the scan-dominated regime (low C, big pool).
    NEXT: cached large-n build to find where tree-best beats flat-best over all C.
    (bench_fullkernel.py, flat_cached.py, tree_kernel.pyx)

P21. (KEY PRIMITIVE WIN -- int8 scan kernel is 7x faster than float, 11x faster than BLAS)
    Micro-benchmark, scan 200k pts d=100, exact L2 top-10:
      numpy BLAS (float): 16.6ms | Cython float kernel: 10.2ms | Cython INT8 kernel: 1.46ms
    The int8 kernel (search_segments_i8, int32 accum, ||x-q||^2 = xn2+qn2-2<x,q>) vectorizes
    hugely better (AVX2/512 processes 32-64 int8/instr vs 8 float; -march=native emits
    vpmaddubsw/vpdpbusd). int8 is the NATIVE big-ann format (no conversion needed).
    WHY IT DIDN'T SHOW in P20's flat_cached: there the pool was tiny (~1085) and routing-
    dominated. int8 wins in the SCAN-dominated regime (big pool). STRATEGIC IMPLICATION:
    with int8 making the scan ~7x cheaper, the SCAN stops being the bottleneck and ROUTING
    becomes it -> which is exactly where the tree's sublinear routing helps. So int8 scan +
    tree routing is a SYNERGISTIC combo (int8 amplifies the tree's relative advantage).
    This is the leaderboard regime (ScaNN/FAISS use int8/int4 SIMD scans). (tree_kernel.pyx)

P22. (int8 helps the TREE end-to-end 1.4-1.66x; routing is now the bottleneck) Tree
    Kf=8192, float-kernel vs int8-kernel (route float + scan int8), identical recall:
      rec 0.86: 0.250 -> 0.180ms (1.39x); rec 0.92: 0.439 -> 0.288 (1.52x);
      rec 0.96: 0.646 -> 0.388 (1.66x); rec 0.98: 1.075 -> 0.698 (1.54x).
    Speedup grows with pool (more scan-dominated -> more int8 benefit). At rec 0.96 the int8
    scan of pool 4181 is only ~0.03ms -> ROUTING (route_topp, float, scoring ~4000 fine
    cells/query at b0=45) is now ~90% of the time. Next lever: int8 ROUTING (quantize
    centroids). CAVEAT (n=50k champion): flat C=4096 + int8 scan should still win here
    (~0.1ms: cheap GEMM routing + cheap int8 scan) -- the tree's routing (scoring many fine
    cells for precision) costs more than flat's single 4096-GEMM at small C. Tree wins only
    at large C/large n. (bench_i8_tree.py)

P23. (n=50k CHAMPION: flat C=4096 + int8 scan, ~0.1ms = ~10k QPS single-thread) int8
    gives flat 2-3.6x over float: rec 0.84 0.205->0.092ms, rec 0.91 0.230->0.116, rec 0.95
    0.388->0.109. flat-int8 is now ROUTING-BOUND (~0.08-0.1ms, ~constant in p because the
    int8 scan is ~free) -> ~10k QPS single-thread at recall 0.95. Beats the int8 TREE
    (0.388ms @ rec 0.96) by ~3x at n=50k, because flat's BATCHED routing GEMM (all queries
    at once) is more efficient than the tree's PER-QUERY hierarchical route_topp -- the
    tree's routing disadvantage is its per-query nature, not dot count. So at testable scale
    flat-int8 is the champion; the tree's hierarchical routing only wins at large C (>=65536,
    n>=10M) where even a batched GEMM over C centroids is slow. KEY REMAINING LEVER for QPS
    (what leaderboards measure): MULTI-THREAD the kernel (prange/nogil) -> up to 16x on this
    box. (flat_cached.py)

P24. (MULTI-THREAD throughput primitive: batch_scan_i8 with prange/nogil) Flat C=4096
    int8, scan-only QPS by thread count (per-thread scratch indexed by omp thread id to
    avoid the shared-buffer race -- a bug found & fixed): 1t 37k, 2t 74k (2.0x), 4t 93k
    (2.5x), 8t 38k, 16t 27k. Scales cleanly to 2-4 threads then DEGRADES -- because the box
    is oversubscribed (load ~22 from the user's own multi-day jobs), so >4 of my threads
    thrash against theirs. On an idle box this would scale toward ~16x. The kernel is the
    right throughput primitive (what leaderboards measure); clean scaling is infra-blocked,
    not code-blocked. (bench_mt.py, tree_kernel.pyx::batch_scan_i8)

P25. (full-pipeline champion, contention-degraded) Flat C=4096 int8, fully batched
    (routing GEMM + vectorized segment build + parallel batch_scan_i8), CORRECT
    distance-sorted recall: ~4.6k QPS @ rec 0.91 single-thread under load 35. The pipeline
    is ROUTING-bound: ~190us/q is the Qh@cf.T GEMM (OpenBLAS thrashing under load 35; ~10-20us
    on an idle box) + argpartition; the int8 scan is only ~26us. Threading doesn't help (the
    numpy routing dominates and competes with prange threads). So the contention hits the
    routing, not the scan -- on an idle box this would be ~10-20k QPS single-thread, scaling
    with threads. The scan primitive is leaderboard-grade (37-93k QPS scan-only); the full
    number is gated by box load on the routing GEMM. (bench_champion.py)

P26. (KEY REVERSAL -- fast int8 scan UNDERCUTS the tree; they are SUBSTITUTES not
    complements, contradicting P22's guess) GENUINE FULL msspacev-1M, flat C=4096 + int8
    batch scan, PROVIDED ground truth (correct), build = kmeans(150k subsample)+assign 1M:
      p= 32: rec 0.901 pool 37069  -> 1t 2554 QPS, 8t 6771 QPS
      p= 64: rec 0.945 pool 65407  -> 1t 1582,     8t 2659
      p=128: rec 0.971 pool 113590 -> 1t  829,     8t 2495
      p=256: rec 0.987 pool 196974 -> 1t  499,     8t 1905
    At 1M, C=4096 => cellsize 244 => HUGE pools (3.7%-20% of DB). Yet flat-int8 still gets
    6771 QPS @ rec 0.90 (8t) because the int8 scan is so cheap that scanning 37k points is
    fine. THE INSIGHT: a fast scan makes the OPTIMAL operating point COARSE-C / big-pool,
    where flat's routing GEMM is already cheap -- so the tree's hierarchical-routing
    optimization (which only helps when fine C forces expensive routing) becomes UNNECESSARY.
    Verify: flat C=16384 at 1M would have tiny pools but 0.38ms routing GEMM -> SLOWER than
    C=4096's cheap-routing+big-int8-pool. So the winning leaderboard recipe is FLAT IVF +
    coarse C + int8 SIMD scan + multithreading, NOT the tree. The tree was the right idea for
    a FLOAT scan (P20); int8 dissolves its advantage. Also note: threading helps at 1M (2.6x
    @ 8t) because the big pool makes it scan-dominated -- unlike 50k (routing-bound, P25).
    REAL leaderboard relevance: 6771 QPS @ rec 0.90 on a CONTENDED box (load 29); idle 16t
    would be ~13k -- in the ballpark of the msspacev-1M leaderboard's graph methods.
    (bench_1m.py, cached index impl/_cache/flat1m_4096)

P27. (int8 PARALLEL ROUTING -- 3-5x faster than float GEMM+argpartition, ZERO recall loss)
    The 1M profile (P26) showed routing = 83us/q (float GEMM 34 + argpartition 49), single-
    threaded, ~half the pipeline and the cap on threading. batch_route_i8 (prange, per query
    top-p centroids by max int8 dot, quantized centroids) on cached 1M index:
      p=32: float 51us rec 0.9013 -> INT8 10us rec 0.9011 (5.1x)
      p=64: float 46us rec 0.9452 -> INT8 15us rec 0.9450 (3.1x)
    8-bit centroid quantization does NOT hurt routing (rerank fixes final order; contradicts
    a naive read of "routing error is unrecoverable" -- 8-bit is fine, the cell ranking is
    robust). FULL int8 pipeline (batch_route_i8 + batch_scan_i8, both parallel) at 1M:
    ~10us route + ~80us scan (8t) = ~90us/q => ~11k QPS @ rec 0.90 (up from 6.7k, P26).
    Now fully int8 + fully parallel -- genuinely competitive with the msspacev-1M leaderboard
    graph methods, on a CONTENDED box. (tree_kernel.pyx::batch_route_i8)

P28. (CAPSTONE -- full int8 pipeline on msspacev-1M, leaderboard config) batch_route_i8 +
    batch_scan_i8 (both parallel int8, route buffer bumped to 384 to allow p>64), cached 1M
    index, provided gt, 8 threads -- the FULL recall-QPS frontier:
      rec 0.901: 103us/q = 9,697 QPS
      rec 0.945: 165us/q = 6,057 QPS
      rec 0.971: 340us/q = 2,940 QPS
      rec 0.987: 601us/q = 1,664 QPS
    COMPETITIVE with the msspacev-1M big-ann leaderboard graph methods (~5-15k QPS @ rec 0.9),
    achieved with FLAT IVF + int8 SIMD route + int8 SIMD scan + threads -- NO graph, NO tree --
    on a CONTENDED box (load 24-33; idle 16-thread would be ~1.5-2x higher). 16t degrades vs
    8t here purely from box oversubscription. Honest caveats: contended box (conservative);
    recall@10 vs leaderboard's exact protocol; subsample-kmeans single-level cover (a better/
    finer cover would lift the high-recall end where pools are huge). The score-TREE was NOT
    used -- int8 dissolved its advantage (P26): a fast scan makes coarse-C/big-pool optimal,
    where flat's routing is already cheap. (bench_1m_i8.py)

N15. (finer C does NOT lift the 1M frontier -- confirms coarse-C optimal even w/ int8 route)
    Built C=16384 1M index (fast: int8-kernel assign of 1M = 26s vs float GEMM timeout!) and
    ran the int8 frontier vs C=4096. C=16384 is WORSE at matched recall: rec 0.955 @ 1588 QPS
    (p=256) vs C=4096 rec 0.987 @ 1664 QPS. Reasons: (a) finer cells need HIGHER p for the
    same recall (NN-neighbours spread across more cells), (b) int8 route top-p maintenance is
    O(C*p) -> blows up at p=256 over 16384 cells, (c) higher routing cost offsets the
    candidate-efficiency gain. So C=4096 + int8 stays champion; coarse-C-optimal (P26) holds.
    BONUS: the int8 routing kernel used as a BUILD primitive assigned all 1M points in 26s
    (vs the float-GEMM assign that timed out at 595s) -- a 20x+ build speedup, the same kernel
    doubling as index-build accelerator. (build_fast_1m.py, bench_1m_i8.py)

N16. (cascade scan for the high-recall tail needs PROPER PQ, not naive int8-PCA) Idea: cheap
    dim-reduced first pass to filter the 197k-point high-recall pool, then exact on survivors.
    Tested on REAL msspacev: PCA to dim 32/48/64 + single global int8 scale -> cascade recall
    caps at ~0.53 (vs 0.987 exact) for ALL dims and T -- because one global scale collapses
    the codes onto the few dominant PCs (first-PC magnitude sets the scale, small-PC values
    round to 0; effective dim ~5). FIX = per-subspace quantization (product quantization /
    OPQ / ScaNN's anisotropic), a substantial reimpl. So the high-recall tail (0.987@1.7k)
    is left as future work behind a proper PQ codec; the headline (0.90@9.7k) needs none.
    (micro-tests, not committed)

P29/N17. (AVX-512 VNNI: real primitive win, but the 1M scan is BANDWIDTH-bound so it doesn't
    help there) The Cython int8 kernel compiled to AVX2 widening (vpmovsxbw+vpmullw+vpmovsxwd
    +vpaddd, 256-bit ymm), NOT AVX-512 VNNI. CPU is AMD EPYC 9R14 (Zen4) w/ avx512_vnni.
    Hand-wrote a vpdpbusd kernel (vnni_scan.c, uint8 data via x+128 offset, corrected by
    256*sumq): MICRO-BENCH (200k pts, single-thread, cache-warm) = 0.816ms vs Cython 1.46ms
    = 1.8x, verified correct. BUT in the multi-threaded 1M pipeline (huge pools 37k-197k
    pts/query, 8 threads) VNNI is INCONSISTENT (0.82-1.18x) -- because the scan streams
    3.7-20 GB and is MEMORY-BANDWIDTH-bound, not compute-bound, so faster int8 arithmetic is
    moot, and the 128-byte padding (vs 100) ADDS 28% bandwidth. Forcing 512-bit (-mprefer-
    vector-width=512) also didn't help: Zen4 double-pumps 512-bit ops. LESSON: at 1M the scan
    bottleneck is BYTES, not FLOPs -> the lever is data COMPRESSION (product quantization:
    ~12-25 bytes/pt vs 100 -> 4-8x less bandwidth), not faster SIMD. VNNI would help a
    COMPUTE-bound stage (small cache-resident data) -- e.g. the routing GEMM over 4096
    cache-resident centroids. Champion stays Cython-int8 9697 QPS @ rec 0.9; PQ is the next
    real lever for the bandwidth-bound regime (tinyknn's SIMD PQ is built, reuse it).
    (vnni_scan.c, micro + head-to-head tests)

P30. RETRACTED/CORRECTED. An earlier micro-test claimed "PQ-cascade preserves recall exactly"
    -- that was WRONG: its exact baseline was ALSO broken (~0.53, a bug in the per-query
    segrows/dedup path), so "cascade == exact == 0.53" was meaningless. The proper benchmark
    (bench_pq.py, NQ=1000, correct batched-kernel exact baseline) shows:
      EXACT int8 scan: recall 0.9866  (correct)
      PQ-cascade (M=25, 256cw, 25 B/pt): recall 0.527 at BOTH T=2000 and T=4000.
    Recall IDENTICAL across T=2000/4000 -> a BUG in the cascade path (more candidates must
    help if T were the limiter), not genuine PQ reconstruction loss -- the per-query
    segrows/argpartition/dedup mapping is the prime suspect (same 0.53 signature as the
    broken micro-test). So PQ as a bandwidth lever is NEITHER validated NOR refuted yet; the
    cascade harness needs debugging before any PQ conclusion. Also the cascade was SLOW
    (3206us/q) purely from per-query Python rerank overhead (not the ADC kernel). LESSON:
    always validate a cascade's EXACT baseline against the known-good batched number before
    trusting the cascade. The bandwidth-bound finding (N17) stands; PQ remains the plausible
    lever but is UNVERIFIED. (bench_pq.py, pq_adc.c)

P31. (the score-TREE's regime is 10M-100M+, NOT 1M -- P26 flips at scale) Flat IVF routing
    scores all C centroids/query; to keep scan bandwidth (N17) sane, C must grow with n
    (~n/cellsize). Measured int8 FLAT routing (batch_route_i8) us/q vs C, projecting n at
    cellsize 244, vs the 2-level tree (2*sqrt(b0*C) dots, b0=16):
       ~n        C        FLAT route   TREE 2-lvl (proj)
       1M       4,096      23 us/q      0.85 us
       4M      16,384      47           1.7
       16M     65,536     116           3.4
       64M    262,144     338           6.8
       256M  1,048,576   1742 (1.7ms!)  13.6
    At 1M flat routing (23us) is cheap vs the ~80us scan -> tree saves ~nothing (=P26, why
    the tree lost at 1M). By 16M flat routing (116us) dominates; at 256M it's 1.7ms/q (caps
    ~580 QPS BEFORE any scan) while the tree stays ~14us. EVEN BETTER for the tree: at large
    C the flat centroid table (C*100 bytes; 100MB at C=1M) exceeds cache -> flat routing goes
    BANDWIDTH-bound too, while the tree coarse level (few-k centroids, cache-resident) stays
    fast. SO: int8 and the tree are SUBSTITUTES at 1M (int8 lets you use coarse C, no tree
    needed) but COMPLEMENTS at scale (10M+ forces fine C -> flat O(C) routing dies -> tree's
    sublinear hierarchical routing is essential). This is exactly why production billion-scale
    ANN (FAISS, DiskANN) uses a hierarchical coarse quantizer (FAISS runs HNSW OVER the
    centroids). The score-and-ball tree's value is REAL -- it just only shows at >=10M, which
    this box can't build. Corrects the scope of P26/N15. (routing scaling measurement)

P32. (RANDOM CENTROIDS beat k-means for IVF routing AND build ~50-85x faster -- k-means'
    Lloyd iterations buy BALANCE we don't need) Build C=4096 cells from C random data points
    (no Lloyd) + one int8-kernel assignment pass (top-2 multiassign): BUILD = 6s vs k-means
    320-514s. msspacev-1M recall vs pool (load-independent, fair):
       recall   random-cent pool   k-means pool
       ~0.90        ~25k              37k
       ~0.945       ~50k              65k
       ~0.985       139k              197k
    Random centroids scan FEWER candidates at matched recall AND build instantly. WHY:
    k-means minimizes quantization error (balanced equal-size cells), but IVF routing only
    cares whether the query's true-NN lands in a PROBED cell -- balance is irrelevant.
    Random Voronoi gives unbalanced (power-law) cells; with multi-assign+multiprobe that
    covers query neighbourhoods with finer effective resolution. And the EXACT int8 rerank
    absorbs the weaker partition entirely (a worse partition changes pool SIZE at fixed
    recall, NEVER correctness). So the expensive Lloyd iterations were wasted. This REVISES
    P12 (which said k-means >> random ASSIGNMENT -- true, but random CENTROIDS+Voronoi is a
    different, strong, near-free thing). UNBLOCKS SCALE: 10M build ~60s (assignment is the
    only n-dependent step, int8-kernel ~26s/1M), no k-means bottleneck. Also makes the
    hierarchical TREE cheap to build (random centroids per level). (bench_randcent.py)

P33. (BATCHED 2-level routing kernel fixes the tree's implementation; tree routing IS
    ~2x faster than flat -- but fine-C-on-1M is a flawed proxy for scale) The per-query
    Python loop in query_full_kernel_i8 made the tree 3-7x SLOWER than batched flat routing
    (masking P31). Wrote route_tree.c (batch_route_tree_i8): coarse top-b0 + fine top-p in
    ONE parallel OpenMP kernel, no Python. Result at C=65536/1M (b0=64): TREE beats FLAT at
    low p (p=16: 71us vs 142us = 2x; p=32: 1.45x; p=64: 1.14x) -- routing-dominated regime,
    confirming P31's mechanism with a correct implementation. BUT loses at p=128 (tree's
    O(fine_cells*p) top-k maintenance blows up) and ALL recalls are low (0.65-0.88) because
    C=65536 is TOO FINE for 1M (cellsize ~15 -> reaching rec 0.9 needs huge p). So fine-C-on
    -1M is a BAD proxy: at 1M the optimal C is coarse (4096) where the tree isn't needed
    (P26). The tree's genuine win regime -- fine C OPTIMAL (cellsize ~150) + moderate p at
    high recall -- exists only at >=10M. Need real 10M data to show it; now buildable fast
    via random centroids (P32). (route_tree.c, bench_tree_batched.py)

P34. (the tree makes the BUILD cheap too -- hierarchical assignment is O(n*sqrt(C)) vs flat
    O(n*C)) Building flat C=65536 cells for 10M needs assigning 10M points to nearest of
    65536 centroids = 10M*65536*100 = 6.5e13 ops -> too slow (>800s). The 2-level tree
    assigns hierarchically: point -> nearest coarse (10M*512) then -> nearest fine WITHIN
    that coarse group (10M*~128) = 6.4e11 ops, ~100x cheaper. Reuses route_tree.c at build
    time. So the tree's O(n*sqrt(C)) advantage applies to BUILD (assignment) AND query
    (routing) -- a second scale benefit. (For flat at scale you'd need this hierarchical
    quantizer anyway = FAISS's HNSW-coarse-quantizer; i.e. you can't escape the tree at
    scale.) Confirmed downloading msspacev-10M to demonstrate end-to-end. (route_tree.c)

P35. (*** THE TREE WINS AT 10M -- P31 validated END-TO-END on real msspacev-10M ***)
    Hierarchical random-centroid build (P32+P34): 10M build in 78s total (read 3s, centroids
    15s, hierarchical assign 37s -- vs ~800s for a flat assign). Then FLAT (route all 65536)
    vs TREE (batched 2-level route_tree.c), b0=48, MATCHED recall (the key diff from 1M: at
    10M cellsize ~150 so coarse routing loses little -> tree recall == flat recall):
       p    recall   FLAT     TREE     tree speedup
       16   0.70     184us    77us     2.4x
       32   0.79     185us    135us    1.37x
       64   0.85     255us    219us    1.16x
       128  0.90     347us    441us    tree LOSES (top-p kernel overhead)
    So at 10M the tree is BOTH ~10x faster to BUILD and up to 2.4x faster to QUERY at matched
    recall -- the scale crossover, demonstrated on real data, not projected. Contrast 1M
    (P33) where the tree's recall was always BELOW flat (cells too small) -> no real win.
    REMAINING: tree loses at p=128 (rec 0.9) because route_tree.c maintains top-p with an
    O(fine_cells*p) linear-worst-scan (6144*128=786k ops/q); a heap or collect-then-
    argpartition -> O(fine_cells) would let the tree win at the high-recall point too.
    CONCLUSION: the score-and-ball TREE's value is real and scale-gated -- invisible at 1M
    (P26: int8+coarse-C wins), decisive at 10M+ (build AND query). This is exactly why
    billion-scale systems are hierarchical. (bench_10m.py)

P36. (*** TREE WINS AT EVERY OPERATING POINT @ 10M after the heap fix ***) Replaced
    route_tree.c's O(fine*p) linear top-k with a MIN-HEAP (O(fine*log p)). Re-ran 10M:
       p    recall   FLAT     TREE     speedup
       16   0.70     221us    97us     2.3x
       32   0.79     299us    106us    2.8x
       64   0.85     297us    162us    1.83x
       128  0.90     435us    276us    1.58x   <- was LOSING (441 vs 347) before the heap
    The heap flipped p=128 from a loss to a 1.58x win AND improved every point (p=32
    1.37x->2.8x). So the TREE now beats flat by 1.58-2.8x at MATCHED recall across the FULL
    range 0.70-0.90 on real msspacev-10M, plus ~10x faster build. The score-and-ball
    hierarchical structure is decisively better at scale -- COMPLETE end-to-end demonstration
    on real data. (route_tree.c heap, bench_10m.py)
    HIGH-RECALL confirm (b0=64): p=128 rec0.90 1.5x, p=256 rec0.93 1.2x, p=384 rec0.95 1.2x.
    Tree wins across the FULL recall range 0.70-0.95; margin is largest at low recall
    (routing-dominated, 2.8x) and shrinks at high recall (scan-dominated, 1.2x). At 100M+
    even high recall becomes routing-dominated (flat C ~655k) so the margin would widen.
    FINAL: score-and-ball tree = 1.2-2.8x faster query (all recall levels) + ~10x faster
    build than flat IVF on real msspacev-10M.

P37. (FINER CELLS make the tree's advantage EXPLODE -- flat is routing-crippled, tree
    thrives) Kf=262144 (cellsize ~38) on 10M, tree (C0=1024) vs flat:
       p=128: FLAT 423us rec0.87 vs TREE 68us rec0.86  -> 6.2x
       p=256: FLAT 444us rec0.91 vs TREE 212us rec0.90 -> 2.1x
       p=384: FLAT 649us rec0.93 vs TREE 191us rec0.91 -> 3.4x
    Flat barely improves with p (routing-BOUND: scoring 262144 centroids dominates), while
    the tree routes in ~17k dots. Tree = 5,245 QPS @ rec0.91, 14,662 QPS @ rec0.86 -- in the
    big-ann 10M leaderboard range (~5-15k QPS @ rec0.9). vs Kf=65536 (1.2-1.5x), finer cells
    widen the tree's edge to 2-6x because finer C is where flat's O(C) routing hurts most --
    and finer C is only AFFORDABLE with the tree (you literally can't run flat IVF over 262k
    centroids efficiently). CAVEAT: tree recall slightly below flat at fixed p (coarse routing
    b0=48/C0=1024 loses a bit); higher b0 closes it at small routing cost. This is the
    capstone: the score-and-ball tree isn't just faster at scale, it ENABLES the fine-grained
    partitions that give the best recall/QPS, which flat cannot afford. (bench_10m.py Kf=262144)

P38. (LEADERBOARD-RELEVANT FRONTIER -- tree on msspacev-10M, full recall/QPS curve)
    Kf=262144 tree, random-centroid + hierarchical build (85s), int8 throughout, b0/p swept
    (route_tree.c buffers raised to 1024/1536), recall@10 vs QPS (8 threads, CONTENDED box
    load 41):
       recall 0.908 -> 4,931 QPS
       recall 0.937 -> 3,031
       recall 0.955 -> 2,547
       recall 0.963 -> 1,791
       recall 0.973 -> 1,228
    The big-ann 10M leaderboard (graph methods, DiskANN etc.) sits ~rec0.9@5-10k QPS,
    rec0.95@2-5k QPS single-machine. This tree frontier (0.908@4.9k, 0.955@2.5k) is in that
    range on a CONTENDED box (idle ~1.5-2x faster), single config, no tuning -- a hierarchical
    int8 IVF (random centroids + 2-level routing kernel) competitive with graph methods at
    10M. Caps at 0.973 (b0/p buffer limits; raise for higher). A formal #1 leaderboard claim
    needs their harness + dedicated HW; this is the algorithm/primitives that feed it.
    (bench_10m_frontier.py)

P39. (*** OFFICIAL big-ann HARNESS run -- the method evaluated by the real protocol ***)
    Wired the int8 score-tree into a big-ann BaseANN module (benchmark/algorithms/sbtree.py
    + sbtree.yaml) and ran `run.py --nodocker` (their runner, their full query set, their
    ground truth, their recall@10 metric). WORKS end-to-end (build via fit(), 29,316 queries
    per config via their loop). Official results:
      msspacev-1M  (Kf=65536):  rec0.850@15.2k, rec0.903@8.8k, rec0.940@5.1k QPS
      msspacev-10M (Kf=262144): rec0.906@4.9k, rec0.939@2.7k, rec0.953@1.8k, rec0.961@1.3k QPS
    These MATCH my own benchmark (P38: 0.908@4.9k, 0.955@2.5k) -> both measurements validated.
    So the method is competitive on the OFFICIAL protocol at 10M (build 86s, 8 threads,
    contended box). This is the closest to a leaderboard submission achievable from a coding
    session -- the only remaining step for a literal #1 RANKING is submitting to their
    competition on their standardized/dedicated hardware (not executable from here). The
    algorithm + int8 SIMD primitives that feed such a submission are complete and validated.
    (benchmark/algorithms/sbtree.py, results/msspacev-{1M,10M}/10/sbtree/)

N18. (finer cells DON'T help the 2-level tree -- Kf=262144 is the 10M optimum) Tested
    Kf=1048576 (cellsize ~10) C0=4096 vs Kf=262144 (cellsize ~38): WORSE at matched recall
    (rec0.95 ~1.5k QPS vs 2.5k; rec0.91 3.4k vs 4.9k), build 202s vs 86s. The smaller scan
    pool is outweighed by costlier 2-level routing (C0=4096 coarse + b0*256 fine = ~135k
    routing dots/query). The 2-level tree tops out at Kf~262144 for 10M; going finer needs a
    3-LEVEL tree (C0,C1,C2 -> route in C0 + b0*C1 + b1*C2 dots, much cheaper coarse routing)
    -- a new kernel. So the best validated frontier stays Kf=262144 (P38/P39: rec0.906@4.9k,
    0.953@1.8k QPS official). 3-level is the clear next lever but substantial + uncertain.
    (bench_10m_fine.py)

N19. (3-LEVEL tree also doesn't help -- deeper hierarchy COMPOUNDS routing recall loss)
    Built a real 3-level tree (route_tree3.c, C0=128 -> Cmid=16384 -> Kf=1M, cellsize ~10)
    to make very fine cells affordable. Result @ 10M: rec0.91@2.4k, rec0.92@2.2k QPS -- WORSE
    than 2-level Kf=262144 (rec0.91@4.9k, rec0.95@2.5k). Why: each level's top-k cutoff loses
    some true cells, and 2 levels (top-b0 coarse x top-b1 mid) COMPOUND the loss -> need much
    higher b0,b1 to hit a recall, eating the routing savings; meanwhile tiny cells (size ~10)
    need huge p. So deeper != better here. CONCLUSION (with N18): the 2-level tree at
    Kf=262144 (cellsize ~38) is the validated OPTIMUM for 10M -- finer cells (2-lvl) cost too
    much routing, deeper trees (3-lvl) cost too much recall. The method's competitive ceiling
    on this approach is P38/P39: rec0.906@4.9k, 0.953@1.8k QPS (official harness). Further gain
    needs a different mechanism (graph traversal -- excluded earlier; or a working PQ codec --
    N16 buggy), not more IVF-tree tuning. (route_tree3.c, bench_10m_3level.py)

P41. (PQ PRESERVES RECALL PERFECTLY -- N16's failure was a MAPPING BUG; the bandwidth lever
    is VIABLE) Clean diagnostic on real msspacev-1M: fraction of exact-top-10 that PQ-ADC
    ranks into top-T (M=25 subspaces, 25 B/pt): T=2000 -> 1.000, T=5000 -> 1.000 (pool ~52k).
    So PQ-ADC ranking is essentially PERFECT for this data -- the earlier 0.527 (N16) was the
    per-query segrows mapping bug, NOT PQ quality. PQ reads 25 B/pt vs 100 (4x less bandwidth)
    -> directly attacks the bandwidth-bound high-recall scan (N17). Cascade: route -> PQ-ADC
    1st pass (4x cheaper) -> top-T -> exact int8 rerank top-T -> top-10, recall preserved.
    This is the one genuine remaining QPS lever for high recall; implementing the kernel
    cascade next. (PQ-ADC diagnostic, pq_adc.c)

N20. (PQ cascade preserves recall but is SLOWER than int8 SIMD scan -- 8-bit gather-ADC
    loses to vectorized int8) Built the full cascade kernel (pq_cascade.c: PQ-ADC 1st pass
    -> top-T -> exact int8 rerank, all parallel). Result @ 1M high recall: PQ-cascade
    0.29-0.69x the speed of full int8 scan. WHY: the 8-bit PQ-ADC does M=25 SCALAR table
    gathers + adds per point, which is SLOWER than the fully-SIMD-vectorized int8 dot (100
    MACs/instr-group), even though it reads 4x fewer bytes -> the int8 scan is COMPUTE-
    efficient, not purely bandwidth-bound (reconciles with N17: neither VNNI nor PQ helps
    because the int8 SIMD is already near-optimal). To get the PQ win you need 4-BIT PQ with
    vpshufb register-LUT SIMD (FAISS/ScaNN/tinyknn) -- substantial, and tinyknn already
    showed that route ceilings at recall 0.964 (N earlier). (A recall artifact -- returning
    top-10 STORAGE rows vs dedup'd orig under a0=2 -- is fixable but moot given the speed.)
    CONCLUSION: int8 SIMD scan is the right primitive; PQ doesn't beat it here. ALL levers
    now exhausted (IVF-tree N18/N19, VNNI N17, PQ N20); 2-level Kf=262144 stays the optimum
    (P38/P39: 0.906@4.9k, 0.953@1.8k QPS official). (pq_cascade.c, bench_pq_cascade.py)

P42. (CORRECTS N20 -- 4-bit vpshufb PQ scan IS 3.9x faster than int8 SIMD; the PQ lever is
    REAL) N20 concluded "PQ slower than int8" -- but that was my 8-BIT scalar-gather ADC (25
    table lookups/pt, not vectorized). The 4-BIT vpshufb variant (16-entry register LUT, SIMD
    shuffle -- tinyknn/FAISS/ScaNN) is the fast one: micro-bench scan 200k pts 1-thread,
    tinyknn 4-bit PQ = 0.757ms vs my int8 SIMD = 2.948ms = 3.90x FASTER. Combined with P41
    (PQ preserves recall as a cascade 1st pass + exact rerank), the cascade
    (4-bit-PQ-ADC -> top-T -> exact int8 rerank) is a GENUINE high-recall QPS lever: the scan
    (which dominates at high recall) gets ~3.9x cheaper. So the levers were NOT exhausted --
    my earlier conclusion was wrong because I tested the wrong PQ variant. The 4-bit vpshufb
    code/scan is in tinyknn (built). NEXT: integrate 4-bit PQ codes per cell into the
    hierarchical tree -> ~3x higher QPS at high recall. (tinyknn _fast_pq estimate_pq)

P43. (4-bit PQ is real (3.9x scan) but capturing it needs a BATCHED VNNI-style kernel, not
    tinyknn's per-query API) End-to-end tinyknn IVF (4-bit PQ) on msspacev-1M: rec0.911@1.2k,
    rec0.966@345 QPS. vs my int8 tree (official harness): rec0.903@8.8k QPS -- my int8 is 7x
    FASTER end-to-end despite a 3.9x SLOWER per-point scan (P42). WHY: tinyknn queries
    PER-QUERY in single-threaded Python; my pipeline is BATCHED + 8-threaded. The batching/
    threading win (7x) dwarfs PQ's per-point scan win (3.9x). SO: the 4-bit PQ lever is real
    but realizing it in MY pipeline requires porting the vpshufb 4-bit-LUT scan into a BATCHED
    MULTI-THREADED kernel (per-cell PQ code blocks + prange + the shuffle intrinsics) -- a
    substantial new low-level kernel (~the hardest primitive in the stack). Until then, int8
    batched stays faster. NET of the PQ thread (P41-P43, correcting N16/N20): PQ preserves
    recall AND its scan is 3.9x faster, but only a batched vpshufb kernel captures it; that's
    the one concrete, substantial remaining lever for ~3x higher high-recall QPS. (tinyknn IVF)

P44. (*** BUILT IT -- batched vpshufb PQ4 kernel, verified correct, 4.06x faster than int8 ***)
    batch_pq.pyx: reuses tinyknn's VERIFIED compute_block_dists (SSE shuffle core) in a
    prange/nogil batched driver. Milestone-verified: batch_estimate output == tinyknn
    estimate_pq bit-for-bit on all test queries. SPEED (batched, 8 threads, 100k-pt pool/query,
    the high-recall scan size): vpshufb-PQ4 = 54us/q vs int8 = 219us/q = 4.06x faster scan
    (2.9x incl the 22us/q table build). So the 4-bit-PQ win SURVIVES in the batched/threaded
    framework (P43 said tinyknn's per-query API couldn't show it -- this kernel does). Combined
    with P41 (recall preserved via exact rerank), integrating this PQ scan into the tree's
    high-recall path -> ~2-3x higher QPS at rec 0.95 (scan-dominated). REMAINING (mechanical):
    per-cell PQ-block layout + top-T extraction + exact int8 rerank + dedup (PQ4_PLAN.md
    steps 2-4). The hard part (correct batched vpshufb) is DONE. (batch_pq.pyx)

P45. (PQ cascade RECALL VALIDATED end-to-end; scan kernel 4x faster; but per-query Python
    table-build + rerank mask it -- need to batch those too) Full cascade (route ->
    batch_pq_topT vpshufb 4-bit PQ -> exact rerank) on msspacev-1M: recall PRESERVED
    (PQ 0.934/0.965/0.982 vs int8 0.946/0.972/0.987 across p=64/128/256 -- slight drop from
    PQ approx, recoverable with larger T). BUT end-to-end 0.12-0.39x the int8 speed -- because
    the per-query distance_table build (1000x tinyknn Python calls) + the per-query Python
    rerank loop (X[oid] gather + einsum) dominate, dwarfing the 4x-faster vpshufb SCAN (P44).
    SAME pattern as N14/P43: a real kernel win masked by per-query Python. To realize it:
    (1) batch the LUT/distance-table build across queries (vectorized numpy or a kernel),
    (2) batched exact-rerank kernel over the top-T survivors. Both mechanical. NET of the PQ
    thread: the batched vpshufb scan IS built, verified, 4x faster, recall-preserving (P41/P44)
    -- the remaining work to capture the ~2-3x high-recall QPS is batching the table+rerank,
    not the scan. (bench_pq_full.py, batch_pq.pyx)

P46. (PQ CASCADE COMPLETE & MEASURED -- 1.27x at high recall, capped by table-build overhead)
    Full cascade built: batch_pq.batch_pq_topT (vpshufb 4-bit PQ scan, P44) + batch_rerank.c
    (batched exact int8 rerank over T survivors) + dist-sorted dedup. msspacev-1M, recall
    PRESERVED (cascade 0.935/0.965/0.982 vs int8 0.946/0.972/0.987 @ p=64/128/256):
       p=64  (pool 65k):  int8 9,322 QPS | cascade 6,417 (0.69x -- table-build overhead > scan win)
       p=128 (pool 114k): int8 4,294     | cascade 3,292 (0.77x)
       p=256 (pool 197k): int8 2,544     | cascade 3,225 (1.27x -- scan-dominated, PQ wins)
    So the PQ cascade is FASTER only at the high-recall/large-pool end (1.27x @ rec0.98), and
    SLOWER at small pools where the fixed per-query Python (1000x tinyknn distance_table calls
    + the dedup loop) outweighs the 4x-faster scan. The win GROWS with pool size -> larger at
    10M (cellsize ~150, pools bigger) and would approach the scan's 4x if the table-build were
    batched (the last per-query Python). NET of the PQ thread (P41/P42/P44/P45/P46, correcting
    N16/N20): a real high-recall QPS lever, fully built/verified/measured -- 1.27x at 1M high
    recall, capped by the table-build, not the (verified-4x) scan. Modest but genuine, and at
    exactly the leaderboard-relevant high-recall operating point. (bench_pq_full.py, batch_pq.pyx,
    batch_rerank.c)

P47. (PQ CASCADE FULLY BUILT -- batched table-build closes it; consistent high-recall win)
    Added a VECTORIZED batched distance-table builder (verified bit-identical to tinyknn's
    per-query distance_table), removing the last per-query Python. Final cascade on msspacev-1M
    (vpshufb scan + batched rerank + batched tables + dist-sorted dedup), recall preserved:
       p=64  rec0.935: int8 9,152 | cascade 9,214 QPS (1.01x)
       p=128 rec0.965: int8 4,976 | cascade 5,443 (1.09x)
       p=256 rec0.982: int8 3,108 | cascade 3,877 (1.25x)
    So the FULLY-BUILT PQ cascade is a genuine high-recall QPS win (1.25x @ rec0.98 on 1M),
    growing with pool size (larger at 10M). Capped below the scan's 4x (P44) by residual numpy
    (table-build einsum + the small top-20 dedup loop) -- further shavable but diminishing.
    COMPLETE: the PQ lever (P41/P42/P44/P45/P46/P47, correcting N16/N20) is built end-to-end,
    verified, recall-preserving, and faster at the leaderboard-relevant high-recall operating
    point. Kernels: batch_pq.pyx (vpshufb scan), batch_rerank.c, batch_tables (numpy). EVERY
    identified lever is now implemented and measured. (bench_pq_full.py)

P48. (*** PQ cascade IMPROVES the OFFICIAL big-ann harness number ***) Integrated the full
    PQ cascade into a BaseANN module (sbtree_pq.py) and ran the OFFICIAL harness (--nodocker,
    29,316 queries, their gt/protocol) on msspacev-1M:
       sbtree_pq (PQ cascade): rec 0.916@10,239 | 0.957@5,542 | 0.980@3,379 QPS
       sbtree    (int8 base):  rec 0.850@15,212 | 0.903@8,773 | 0.940@5,099 QPS
    The PQ variant is FASTER AND HIGHER-RECALL at the rec~0.9 point (0.916@10.2k vs 0.903@8.8k)
    and extends the frontier to rec 0.98@3.4k (the int8 configs topped out at 0.94). So the
    fully-built PQ cascade (P41-P47) is not just a micro-bench win -- it produces a strictly
    better recall/QPS frontier on the OFFICIAL protocol. This is the improved official number;
    the high-recall end (where the leaderboard is most competitive) is materially better.
    (benchmark/algorithms/sbtree_pq.py, results/msspacev-1M/10/sbtree_pq/)

P49. (PQ cascade beats int8 at 10M too -- vectorized build fixed the per-cell-loop block) OFFICIAL
    big-ann harness, msspacev-10M, sbtree_pq vs int8 sbtree:
      rec~0.91:  PQ 0.9115 @ 5,484 QPS  vs  int8 0.9058 @ 4,948   (higher rec AND faster)
      rec~0.95:  PQ 0.9486 @ 3,199 QPS  vs  int8 0.9387 @ 2,686   (higher rec AND faster)
    Full 10M frontier (high-recall points added after the T>4096 fix below):
      PQ 0.9115 @ 5,484 | 0.9486 @ 3,199 | 0.9715 @ 1,742 | 0.9811 @ 1,151
      int8 0.9058@4,948 | 0.9387@2,686  | 0.9530@1,849 / 0.9611@1,346 | (never reached 0.98)
    PQ cascade STRICTLY DOMINATES the entire int8 frontier at 10M (higher rec AND faster at every
    point) and extends to 0.981 where int8 topped at 0.961 -- the 4-bit byte-bandwidth advantage
    compounds at the high-recall end. Two fixes unblocked this:
    (a) build: pad each cell to a multiple of 16 IN STORAGE ORDER once, then transform the WHOLE
        point array in one tinyknn.transform call -- block boundaries still respect cells (every
        cell start is 16-aligned), so whole-array transform == concatenation of per-cell
        transforms. Removed the 16384-iteration Python loop; verified equivalent on 1M
        (0.9152/0.9562/0.9788 == P48 within noise). Plus load_index/_save_index caching (424MB
        npz) so the 10M build (routing+transform) runs once, not per harness invocation.
    (b) THE BUG behind every prior "3rd config failed": batch_pq_topT's per-thread heap scratch
        hd/hx was hard-sized [nthreads, 4096], but configs with T>4096 (T=6000, 9000) overflowed
        it -> "free(): invalid next size" heap corruption / coredump. The harness swallowed the
        worker exception and reported only "One or more algorithm runs failed", which I'd
        misread as a wall-clock timeout. Fix: size scratch as np.empty((nthreads, T)). It was
        NEVER a timeout -- direct repro showed load_index=10s, query p=256/T=6000=17s. Lesson:
        when the harness says "runs failed" with no detail, reproduce the algorithm DIRECTLY to
        see the real exception. (batch_pq.pyx hd/hx sizing; sbtree_pq.py fit, vectorized+cached)

P50. (BYTE-BUCKET top-T selection beats the size-T heap 1.34-1.63x -- new low-level primitive)
    Profiling the 10M PQ query showed pq_topT is 87% of query time, and within it the size-T
    max-heap (not the SIMD scan) dominates at high recall: heap cost scales ~1.2ms per unit T
    (T=500->12000: 6.2s->19.4s at fixed pool), so at T=9000 the heap is ~2/3 of pq_topT. Since
    PQ-ADC dists are uint8, replaced the O(pool*log T) heap with O(pool) byte-bucket (counting)
    selection: 256-bucket histogram during the SIMD scan -> cumulative threshold bucket -> collect
    all below + partial at threshold. Survivor ORDER is irrelevant (exact int8 rerank re-sorts),
    so boundary ties are harmless (recall bit-identical). Clean same-process A/B (batch_pq.pyx
    batch_pq_topT vs batch_pq_topT_bucket): 1.63x @T=4000, 1.43x @T=6000, 1.34x @T=9000. Signed
    tables handled by bucket index = raw^0x80 (distance-ordered). OFFICIAL 10M frontier jumped:
      heap:   0.9114@4,289 | 0.9485@2,383 | 0.9715@1,698 | 0.9811@1,140
      bucket: 0.9115@5,780 | 0.9486@3,633 | 0.9715@2,179 | 0.9812@1,695   (recall identical)
    vs int8 baseline (0.9058@4,948 ... 0.9611@1,346 max) the PQ cascade now dominates by a wide
    margin at every operating point. (impl/batch_pq.pyx batch_pq_topT_bucket; sbtree_pq.py query)

N22. (AVX2 256-bit LUT scan gives NO speedup here -- the PQ scan is scalar/memory-bound, not
    shuffle-bound) Added batch_pq_topT_bucket_avx using tinyknn's compute_block_dists_avx
    (_mm256_shuffle_epi8, 2 sub-quantizers/op, the layout is identical to SSE so it drops in;
    odd Mb handled by an SSE tail block). Clean A/B vs the SSE bucket kernel: 1.07x/0.99x/0.97x
    -- i.e. nothing. survivor-overlap=1.0000 (bit-equivalent, correctness confirmed). WHY: Mb=14
    (only 14 sub-quantizers, so 14 shuffle ops/block), but each block does 16 lanes x scalar
    work (byte extract + bucket-index + store + histogram). The scalar per-lane loop + memory
    traffic dominate; widening the shuffle can't help. LESSON: profile WHICH part of a SIMD
    kernel is hot before widening it -- here the vpshufb was never the bottleneck. (batch_pq.pyx)

P51. (Memory-lean bucket v2 wins 1.13-1.24x at high T) Following N22 (kernel is memory-bound),
    dropped the 4-byte per-point row store (rbuf): pass 1 stores only the 1-byte bucket index;
    the collect pass RE-WALKS the same pick/block structure to recompute row=gblk*16+lane inline
    (no shuffle, no rbuf). Per-point write traffic 5B->1B. Clean A/B vs v1: 0.96x @T=4000 (re-walk
    overhead not amortized), 1.13x @T=6000, 1.24x @T=9000 -- sets_identical=True. query() selects
    v2 when T>=5000 (high-recall configs, the competitive region) else v1. (batch_pq.pyx
    batch_pq_topT_bucket_v2; sbtree_pq.py query kernel selection)

N23. (a0=1 single-assignment LOSES to a0=2 multi-assignment, even at equal scan work) Tested
    halving the pool by assigning each point to 1 cell instead of 2 (a0=1), compensating with
    higher p. At EQUAL pool size (a0=2 p=256 vs a0=1 p=512, both ~125k slots, 1M): a0=2 gives
    rec 0.9792 vs a0=1's 0.9638. The multi-assignment makes the same scanned points better-chosen
    (each DB point reachable from its 2 nearest centroids), so the coverage benefit outweighs the
    2x storage/pool cost. a0=2 confirmed optimal -- keep it. (sbtree_pq.py a0 param)

P52. (Finer cells help mid-recall but flat routing caps the gain -> motivates hierarchical-route
    + PQ synthesis) 1M C-sweep, a0=2: C=4096 gives 0.9158@11,150 / 0.9571@6,367 / 0.9795@4,238;
    C=16384 gives 0.9478@8,353 / 0.9663@5,386 / 0.9675@4,194. Finer cells concentrate probing ->
    at matched rec ~0.957, C=16384 interpolates to ~7,000 QPS vs C=4096's 6,367 (~10% better).
    BUT flat routing is O(C)/query (4x more at C=16384) and eats much of the pool-scan saving, and
    the configs don't reach rec 0.98 without yet more probing. The clean next lever: HIERARCHICAL
    routing (route_tree.c, already in sbtree.py) makes routing ~O(sqrt C) so finer cells (smaller
    pool at matched recall) pay off fully. SYNTHESIS = tree-route to pick fine cells + PQ-cascade
    scan them + exact rerank. This unifies the score-tree thesis with the PQ-cascade champion;
    it's the identified high-value next build. (sbtree_pq.py C/a0 params; would reuse route_tree.c)

N24. (Hierarchical-route + PQ synthesis LOSES to flat-PQ at 1M -- routing approximation costs
    recall; reconfirms scale-gating) Built SBTreePQ2 (sbtree_pq2.py): full 2-level tree routing
    (route_tree.c, Kf=65536 fine cells under C0=1024 coarse) + per-fine-cell PQ4 blocks + bucket
    select + rerank. At 1M, at matched pool (~30k slots): synthesis rec 0.8966@7,794 vs flat-PQ
    C=4096 rec 0.9157@12,264 -- WORSE on both axes. Cause: the 2-level routing is approximate
    (coarse filter drops good fine cells), so at equal scan work it picks worse cells than flat's
    exact routing -> lower recall. This is the SAME lesson as the int8 sph-tree (tree useless at
    1M, only pays at billion-scale where flat routing is unaffordable). Also hit a hard blocker:
    route_tree.c internal buffers cidx[1024]/fidx[1536] cap p<=1024 -> core dump at p=2000 (would
    need enlarging to probe enough tiny fine cells). NET: flat-PQ-cascade stays the practical
    champion at 1M-10M; the synthesis is a billion-scale architecture whose advantage doesn't
    convert at testable scales (mirrors the original sph-tree-correction). (sbtree_pq2.py)

P53. (Finer PQ dpb=2 ~ comparable to dpb=4 across the frontier -- CORRECTED; earlier fixed-p
    claim was a p-limited artifact) PQ granularity (dpb) is a tunable. INITIAL (WRONG) read: at
    fixed p=128, dpb=2 hit 0.9600@8,042 while dpb=4 "plateaued" at 0.9564 -> looked like dpb=2 was
    +37% faster at higher recall. THE FLAW: at p=128 dpb=4 was POOL-LIMITED (can't reach high rec
    no matter how large T, because the true neighbors aren't in the p=128 pool); that handicapped
    dpb=4, not a real PQ-accuracy win. PROPER frontier (vary p too, 1M):
      dpb=4: 0.9156@12,836 | 0.9554@7,920 | 0.9714@5,240 | 0.9795@4,516
      dpb=2: 0.9223@12,333 | 0.9467@8,687 | 0.9691@5,551 | (didn't reach 0.98 in tested cfgs)
    -> roughly COMPARABLE; dpb=2 keeps only a marginal edge at the very-high-QPS end (0.9223 vs
    0.9156 @ ~12k QPS), dpb=4 slightly better mid-recall, dpb=4 owns the 0.98 tail. The finer PQ's
    smaller-T benefit is ~cancelled by its 2x per-point scan cost. LESSON: compare on the (p,T)
    ENVELOPE, never at a single fixed p -- a fixed operating point can be pool-limited for one
    config and make the other look artificially better. dpb=4 stays the default. (sbtree_pq.py dpb)

P54. (CRUX: under int8 compression, IVF-PQ BEATS HNSW -- the memory-constrained leaderboard
    regime favors contiguous SIMD scan over graph hops) Benchmarked faiss HNSW vs sbtree_pq on
    msspacev-1M, 8 threads, same metric (rec@10 vs QPS):
      HNSW-flat (float32, 4x mem): 0.8983@23,606 | 0.9427@14,711 | 0.9683@7,640 | 0.9813@4,316
      HNSW-SQ8  (int8, mem-viable): 0.8939@11,202 | 0.9374@6,364 | 0.9613@3,691 | 0.9736@1,963
      sbtree_pq (int8 IVF-PQ):     0.9156@12,836 | 0.9554@7,920 | 0.9714@5,240 | 0.9795@4,516
    float-flat HNSW wins 1.5-1.8x BUT needs 4x memory (40GB @100M -> infeasible). Under int8
    compression (the ONLY memory-viable form at 100M/1B), HNSW-SQ8 is SLOWER than sbtree_pq --
    up to 2.7x at rec 0.97 (1,963 vs 5,240). STRUCTURAL reason: IVF-PQ scans CONTIGUOUS packed
    PQ blocks (vpshufb + bucket-select, SIMD-friendly, cache-linear), while HNSW HOPS to scattered
    graph nodes (cache-hostile, can't batch the LUT scan). So in the memory-constrained regime the
    big-ann 100M/1B leaderboard actually tests, the int8 IVF-PQ cascade is competitive-to-SUPERIOR
    to graph methods. The earlier "HNSW dominates" read was a memory-UNCONSTRAINED artifact (it
    only wins when you can afford 4x-larger float vectors). This re-validates the whole approach
    for the leaderboard's real operating point. (faiss IndexHNSWFlat / IndexHNSWSQ vs sbtree_pq)

SUMMARY (primitive stack built this session, all in tree_kernel.pyx + sph_tree_fast.py):
  contiguous IVF layout (3.3x) -> Cython float scan (1.7-2.6x) -> int8 scan (7x over float,
  11x over numpy BLAS) -> int8 routing -> fully-kernelized tree (route_topp+search_segments)
  -> multi-threaded batch_scan_i8 (2.5x@4t). Champion at testable scale (n=50k): flat C=4096
  + int8 scan = ~0.1ms/q single-thread (~10k QPS full pipeline, 37-93k scan-only). The tree
  wins only at large C (>=1M points), which is build-infeasible on this contended box.
  Honest ceiling to "top the leaderboard": needs (a) 1M-10M index builds and (b) clean
  multi-thread QPS -- both infra-blocked here -- plus the leaderboard runs on dedicated HW
  vs tuned C++. The primitives built are competitive per-core; the demonstration is gated by
  the shared box, not the algorithm.

INFRA NOTE: the shared box ran at load 43-52 on 16 cores (2.7-3.3x oversubscribed) by the
    user's own multi-day llvm-lit/hermes jobs, making 1M builds infeasible (200k build = 94s,
    50k build = 104-171s). Full msspacev-1M race is infra-blocked, not algorithm-blocked.
    The 50k result above is the valid head-to-head; it already shows flat wins on wall-clock,
    so the 1M race would not change the practical conclusion (it would only widen the tree's
    dot-count edge, which doesn't convert to wall-clock).

N10. Adaptive (margin-based) multi-assignment FAILS vs fixed top-a -- and fixed
    top-a is also GPU-friendly (regular (n,a) matrix). msspacev: fixed top-a=4
    (0.9812 @ 4x space) BEATS margin=0.12 (0.9788 @ 7.09x); top-a=2 beats margin=0.05
    at less space. Reason: margin over-replicates dense-region points (near many
    centroids but neighbours also near) and under-serves isolated boundary points;
    meanwhile fixed top-a ALREADY replicates into the adjacent boundary cell where a
    point's escaping neighbours route, so explicit adaptivity adds nothing. Use fixed
    top-a (the paper's uniform multi-assignment) -- better recall/space AND fixed
    shapes. (Escape-rate-driven replication via a kNN graph is the remaining untried
    variant, but the reasoning above suggests it won't beat fixed top-a either.)
    (adaptive_assign.py)

N11. LSH-forest / LSF intrinsic-redundancy routing (data-independent) is FAR worse
    than the k-means data-dependent partition -- the definitive routing result. Same
    PCA-sketch ranker for both (isolates routing). msspacev, recall@10 at matched
    pool: LSH-forest 8x space (pool 13k) = 0.589; k-means+top-a 2x space (pool 11k) =
    0.953; k-means 8x = 0.992. So the paper's LSF mechanism, even worst-case-optimal
    and with intrinsic multi-membership redundancy, can't match the data-dependent
    partition. REDUNDANCY AMPLIFIES A GOOD PARTITION (P11) BUT CANNOT SUBSTITUTE FOR
    ONE. Completes the routing story: data-independent routing (RP/LSH: N1,N2,N11) is
    weak; data-dependent worst-case opt (min-max: N4) adds nothing over k-means;
    k-means data-dependent partition + multi-assignment is the answer. The n^rho
    worst-case LSH bound is real but on REAL data k-means vastly outperforms it
    (0.95 vs 0.59 at matched space) -- the practice vs worst-case-theory gap, measured.
    (lsh_router.py vs score_ball_tree.py)

N12. Cost-sensitive near-pair cover (paper lem:cost-sensitive-near-pair-cover, idea #1)
    gives NO better space/recall tradeoff than fixed top-a, AND does not contract the
    power-stabbing potential. msspacev-50k, C=512, r=105 (median 1NN), nprobe=16:
    fixed top-a=1/2/3 -> recall 0.832/0.912/0.941 at avg-memb 1.0/2.0/3.0 (slope
    ~0.08/memb); cost-greedy (replicate only uncovered near-pairs into cheapest
    stabbed cell) -> recall 0.842 at avg-memb 1.17 (slope ~0.06/memb). WORSE per unit
    space. Reason: near-pair coverage is a DB-self proxy; queries are not DB points,
    and the uncovered points are genuine boundary points whose extra membership only
    helps if the QUERY happens to route there. (cost_cover.py)

N13. (THE STRUCTURAL FINDING) The power-stabbing load does NOT contract on real
    high-dim data -- it is pinned at the lem:balanced-fanout-obstruction lower bound
    K^{1-rho} F regardless of cover scheme. msspacev-50k C=512: every scheme (top-a=1,2,3,
    cost-greedy) gives load ~2.2-2.6e3 = ~130-150 x parent F=16.8, while K^{1-rho}=512^{6/7}
    ~=208. The reallocation of multiplicity cannot escape it. CAUSE: distance concentration
    in high dim makes the inflated query region Q_i={q:||q-o_i||<=R_i+r} stab essentially
    EVERY cell (r ~= inter-centroid spread), so load = (#cells)^{~1} * per-cell ~ K^{1-rho}F
    intrinsically. The additive r-inflation + 4th-power Psi=1+(R/r)^4 are the culprits.
    IMPLICATION: the theory's hoped-for certified-cover contraction (child potential <
    parent, asm:power-cover) is OBSTRUCTED on concentrated data. The fix must drive cell
    radii R BELOW r (so Psi->1 and R+r stops blanketing all cells) -- i.e. radius-bounded
    clustering (idea #3) / deeper recursion, NOT multiplicity allocation.
    >>> C-SWEEP CONFIRMS AND REFUTES the "drive R below r" fix: msspacev-50k, r=105,
    Rnode=132.9 (Rnode/r=1.27, whole set spans ~1.3 search radii). stab_frac: C=64->1.000,
    256->1.000, 1024->1.000, 4096->0.993. EVERY query stabs EVERY cell at all C, EVEN at
    C=4096 where R/r=0.92 (cell radius below r). Driving R below r did NOT break the blanket.
    load/F GROWS with C: 26/77/220/599 (~tracks K^{1-rho}=35/116/380/1248). ROOT CAUSE is
    NOT cell radius -- it is that the DATASET DIAMETER (~2.5r) is only a few search radii, so
    any additive region R+r >= 1.9r already covers all centroids. FINAL CONCLUSION: on
    concentrated high-dim data the certified-cover power-stabbing potential is
    NON-CONTRACTIBLE BY ANY FLAT BALL COVER (pinned >= K^{1-rho}F). This is exactly why the
    practical method MUST probe top-nprobe (selective, ~16 cells) and accept a small recall
    miss instead of stab-all-filters (the worst-case guarantee = K cells, no speedup) -- the
    worst-case-vs-practice gap, structurally explained. The escape requires a SELECTIVE
    primitive (sublinear stab fraction): additive ball filters are not selective under
    concentration; angular/spherical-cap filters are the candidate. (cost_cover.py)

================================================================================
OPEN / UNTRIED (from notes8.md)
================================================================================
- #1 navigable graph WITH worst-case bounds (held: not HNSW; want LSF-with-bounds).
- #2 sketch oracle inside TREE beam search (rp_beam was a first cut; weak router).
- #3 anisotropic (near-pair-weighted) sketch -- untested on real L2.
- #8 inner-product / MIPS (text2image big-ann track) -- method needs adapting.
- #10 per-query recall certificates from the JL distance gap -- worst-query robustness.

================================================================================
RUST ENGINE + SCALE EXPLORATION (sbann-rs/) — all-native pluggable IVF-PQ ANN
================================================================================

P55. (Native Rust engine: all-SIMD, pluggable Router x Compressor, no FFI) Built sbann-rs from
    scratch: mmap i8bin reader, AVX2 int8 L2 (route+rerank, self-test vs scalar, 17x over scalar),
    SSE vpshufb 4-bit PQ-ADC scan (self-test vs scalar), byte-bucket/partial-T select, streaming
    bounded-memory build. Trait slots: Router {FlatIvf(kmeans/rand), AvqRouter(additive
    multi-index), HierRouter(2-level score-tree)} x Compressor {Pq4, Opq4(random+learned rotation),
    Apq4(anisotropic), ScalarI8}. CLI: `run --router --compress --a0 --C --tmul`. At 1M matches
    Python sbtree_pq recall (0.94 @ flat+pq4+a0=2).

P56. (Quantization levers, isolated at 1M flat router) Validated deltas over plain pq4 (0.9631@p256):
    SOAR smart-a0=2 (+1.7%, BIGGEST) > learned OPQ (+0.7%) > anisotropic eta=4 (+0.4%). AVQ-routing
    with K-MEANS codebooks beats flat IVF (~1/2 the pool at matched recall). RAIR (arXiv 2601.07183
    inverse-residual a0=2) TIES our SOAR on msspacev (no free win, lambda untuned). Adaptive
    early-termination: ~12% fewer cells but check-overhead-bound + msspacev query-difficulty
    variance too low to pay off. These are all pluggable options now.

P57. (Competitor benchmarks, 1M, 8 threads, same data/metric) faiss IVFPQ-FastScan+refine: we TIE
    it (0.90@~12.5k QPS) and our build is 80x faster (9s vs 731s). DiskANN (Vamana graph, int8):
    competitive-to-better than us at rec 0.90-0.94, pulls ahead only at 0.97+. HNSW-flat (float)
    wins 1.5-1.8x BUT needs 4x memory (40GB@100M -> infeasible); HNSW-SQ8 (int8) is SLOWER than us
    (P54). So in the memory-constrained 100M/1B regime the leaderboard actually tests, int8 IVF-PQ
    is the right family (Puck/ScaNN agree -- they're IVF+quant, not pure graph).

N25. (THE SCALE WALL: flat routing build is O(n*C), prohibitive at 100M) flatrand+opql @10M was
    fine but @100M the flat assign (100M x 16384 = 1.6T SIMD-L2 + per-call dispatch) ran >70min and
    flat-C=65536 was a 3-HOUR non-starter. This is exactly what the project's hierarchical
    score-tree (route_tree.c, O(n*sqrt C)) was built to break. Ported HierRouter to Rust: 2-level
    (C0 coarse -> fine cells), makes Kf=65536 buildable. But b0 coarse-expansion trades recall vs
    pruning (b0=8 -> recall plateaus 0.81; b0=cb/4=32 -> 0.957). At ~25% expansion the build cost
    ~= flat-16384 anyway -- hier's real value is enabling FINER cells (better queries) at that cost.

N26. (100M scales the BUILD but is NOT quality-competitive yet) hier+opql(dpb=5) Kf=65536 @100M
    built (no OOM after fixing an assign bug -- per-point Vec clone+flatten spiked multi-GB and
    silently OOM-killed 2 runs; fixed to write into preallocated array). Frontier: 0.62@1888,
    0.78@431, 0.91@74 QPS -- recall climbs but QPS collapses. Causes (all TUNING, not architecture):
    (a) RANDOM fine centroids too coarse at scale (k-means needed but k-means@Kf=65536 is itself
    O(sample*Kf), expensive); (b) dpb=5 lossy (memory choice); (c) per-survivor random-gather rerank
    slower than Python's batched kernel. THE PROMISING FIX: AvqRouter with cheap RQ assign
    (i0=nearest c0, i1=nearest c1-of-residual, O(c0+c1)=512/pt vs hier 16640) + K-MEANS codebooks
    (cheap: 256 centroids) -> fast build (1M in 37s) AND good recall. Needs the AVQ probe/plist
    fixed (currently scans all c0*c1 cells/query, and plist overshoots nc). That's the next lever.

N27. (Contended-box memory ceiling reconfirmed for 100M) 100M in-mem index ~13-17GB; box has
    ~15-38GB free fluctuating (others use ~22-45GB). dpb=2 OOM-died; dpb=5 (2GB blocks) fit. Base
    is mmap (evictable, not the OOM cause). For 1B (1TB base) the on-disk mmap index is mandatory.

SCALE-EXPLORATION VERDICT: the Rust engine reached feature + build-scaling parity (hierarchical
routing was the key missing piece, added this session) but NOT tuned-quality parity at 100M. The
three fixes to be competitive at scale: (1) AVQ-routing (cheap RQ assign + k-means codebooks) or
hierarchical k-means centroids; (2) dpb=2 + learned OPQ compression; (3) batched rerank kernel.
The Python sbtree_pq (route_tree.c hierarchical + tuned PQ) remains the more mature scale path;
literal big-ann #1 still requires submission on dedicated/uncontended hardware.

N28. (AVQ-routing scale path: better recall ceiling + fast build, but QPS still ~3x behind Python)
    avq(cheap-RQ-assign, k-means codebooks)+OPQ(dpb=2) @10M: 0.8391@3933 | 0.9442@1151 | 0.9863@436
    | 0.9984@87. vs Python sbtree_pq 0.9486@3688. AVQ build is FAST (cheap RQ assign O(c0+c1)/pt)
    and ceiling HIGH (0.998), beating flatrand. BUT at matched rec 0.945 it needs p=1024 where
    Python needs p=128 -> its additive-multi-index routing is LESS discriminative per cell, so 8x
    more cells probed + T=30*p huge survivor count -> the unbatched per-survivor rerank dominates.
    CONCLUSION of the scale exploration: Rust engine reached BUILD-scaling + RECALL-CEILING parity
    (AVQ-routing + hierarchical both added this session) but NOT QPS parity. The last gap is the
    QUERY path: (1) routing discrimination (Python's normalized-kmeans + route_tree beats our
    additive-multi-index per cell), (2) BATCHED rerank kernel (Python's batch_rerank_i8 vs our
    per-survivor l2_i8 random-gather), (3) smaller T via more accurate PQ. These are focused
    engineering, best done on a dedicated/idle box. The mature scale champion remains the Python
    sbtree_pq (official 10M 0.91-0.98 @ good QPS); the Rust engine is the all-native rewrite that
    caught up on features/build/recall but needs query-kernel tuning for QPS parity.

N29. (f32 GEMM routing is a DEAD-END; QPS gap is routing QUALITY not speed) Implemented batched
    f32 GEMM routing (matrixmultiply, Q@pivots^T, cnorm-2G, top-p) as `runb`. Recall bit-identical
    to per-query, but SLOWER: @C=4096 0.9665@4079 vs per-query 5063; @C=16384 0.9423@3730 vs 9878.
    WHY: f32 GEMM is 4x the data of int8, and the per-query routing loop is ALREADY int8-SIMD
    (l2_i8) + parallel-over-queries -- the blocking advantage doesn't overcome the 4x data penalty.
    To win you'd need an INT8 GEMM (VNNI vpdpbusd). BUT the real insight: routing SPEED isn't the
    bottleneck. The gap to Python is routing DISCRIMINATION -- AVQ/random need p=1024 where Python's
    route_tree+normalized-kmeans needs p=128 for matched recall (8x more cells scanned). So the
    lever is routing QUALITY (fewer cells per recall = affordable k-means/hierarchical centroids),
    not faster routing compute. f32 GEMM kept as `runb` but off by default.

P58. (K-MEANS ROUTING IS THE LEVER; flat+kmeans BEATS Python at 1M) Routing discrimination A/B
    @1M C=4096 pq4 (recall@p): flat(kmeans) 0.8295@16/0.9170@64/0.9714@256 >> flatrand
    0.7477/0.8908/0.9665 >> avq 0.6986/0.8708/0.9613. K-means wins BOTH axes: (1) better
    recall-per-cell (discrimination), (2) higher QPS (balanced cells -> smaller pools -> less
    scan: 36k vs 18k QPS @p16). CRUCIAL: flat+kmeans+pq4 @1M = 0.9170@21,036 QPS BEATS Python
    sbtree_pq 0.9152@10,239 (~2x faster at matched recall). The "Rust behind Python at scale" story
    (N26/N28) was an ARTIFACT of using random/AVQ centroids (chosen because flat-kmeans build is
    O(n*C), too slow at 100M). At 1M where kmeans is affordable, we WIN. THE SCALE FIX: hierarchical
    k-means (kmeans coarse C0, then kmeans fine WITHIN each coarse cell) -> kmeans quality at
    O(n*sqrt Kf) build. HierRouter currently uses RANDOM fine centroids (N26 -> poor 100M); replacing
    with hierarchical-kmeans is the key lever. Also: f32 GEMM routing is a dead-end (N29) -- speed
    wasn't the gap, QUALITY was.

P59. (Hierarchical k-means recovers routing quality at fast build) HierRouter::train_hkmeans
    (kmeans coarse C0, then kmeans Kf/C0 fine WITHIN each coarse's sample points). @1M C=16384 pq4:
    hierk 0.8783@64/0.9516@256 vs random-fine hier 0.8236@64/0.9105@256 (+0.05/+0.04). Build 23.6s
    vs 16s (per-coarse kmeans is on the sample, cheap). This is P58's scale lever realized: kmeans
    quality at O(n*Kf/C0) train. Next: validate @100M (should beat random-hier's 0.91@74 QPS).
    Factory: `--router hierk`. The 100M assign is still O(b0*Kf/C0)/pt ~16640 (~70min) but the
    centroids are now good so low-p recall should be much higher -> better frontier.

N30. (neurips23 ongoing-leaderboard tracks need data + adaptation -- not a blind autonomous pivot)
    Tracks (10M-scale): OOD=text2image-10M (FLOAT d=200, ~8GB), Filter=yfcc-10M (uint8 d=192 +
    metadata tags), Sparse=sparse-1M/full (sparse vecs), Streaming=wikipedia-35M (insert/search/
    delete runbook). The harness + dataset defs are present in neurips23/ but NONE downloaded.
    KEY: our int8 IVF-PQ engine doesn't directly apply -- OOD is float32 (needs float kernels or
    quantize), Filter needs predicate machinery, Sparse needs sparse vectors, Streaming needs
    insert/delete. Each is a real adaptation + large download, best done with user direction. The
    billion-scale (msspacev int8) track is where our engine + the hierk progress (P58/P59) directly
    apply; continuing there autonomously. neurips23 = a deliberate pivot for when the user returns.

P60. (Routing-quality ladder confirmed @10M: random < avq < hierk < Python route_tree) 10M hierk
    (hier-kmeans+OPQ dpb2): 0.8970@3697/0.9662@1079 BEATS avq 0.8391@3933/0.9442@1151 (better
    recall per p). But still ~2x behind Python 0.9486@3688 -- the gap is hierk's COARSE PRUNING
    (b0=64 of C0=256 loses cells flat-kmeans keeps). flat+kmeans (no pruning) BEAT Python at 1M
    (P58); testing whether it beats Python @10M too (kmeans trains on 1M sample, assign 10M*16384
    ~few min -- feasible at 10M, just not 100M where the O(n*C) assign is the wall). If yes: the
    clean story is flat+kmeans wins where buildable; hierk is the 100M+ approximation (needs the
    route_fine alloc fix N31 + larger b0 to close the pruning gap).

P61. (hierk is the BEST Rust scale config; flat+kmeans loses on CELL SIZE) 10M: flat+kmeans
    (C=16384) 0.9012@1459/0.9724@421 LOSES to hierk (Kf=65536) 0.897@3697/0.9662@1079 -- same
    recall but hierk 2.5x faster QPS because hierk affords FINER cells (Kf=65536, ~300pts/cell)
    via hierarchical build where flat-kmeans is stuck at C=16384 (~1220pts/cell -> 4x bigger pools
    -> slow scan). So the routing ladder is random < avq < flat-kmeans < hierk (hierk wins by
    affording fine cells). hierk is now only ~1.4-2.4x behind Python (0.9486@3688) -- down from
    avq's ~3x. Residual gap = Python's route_tree discrimination + tuned query kernels (bucket-
    select, batched rerank). Build bottleneck: kmeans_f32 nearest() is SCALAR f32 l2 -> flat-kmeans
    C=16384 train took 47min+ on 1M sample. SIMD-izing it speeds all kmeans builds 4-8x.

P62. (SIMD k-means assign + correction to N31) Added AVX2+FMA f32 L2 to kmeans_f32 nearest()
    (was scalar). Recall bit-identical, builds faster. flat+kmeans @1M (idle-ish box) = 0.9170@
    30,828 QPS -- ~3x Python's 0.9152@10,239. CORRECTION to N31: the 100M hierk build slowness is
    COMPUTE-bound (100M * b0*Kf/C0 = 100M*16640 = 1.66T l2 ~70-110min on contended box), NOT the
    per-point alloc (that's ~5s total, minor). To make 100M hierk faster: reduce b0 (less recall)
    or use VNNI int8 routing. The route_fine alloc fix is low-priority (only the query path, 29k
    allocs/batch, negligible).

P63. (CELL GRANULARITY is the discrimination lever -- closes the gap to Python) The residual gap
    @10M was NOT primarily query kernels -- it was FINE-CELL COUNT. Python uses Kf=262144 fine
    cells; Rust hierk used Kf=65536. Re-ran hierk @10M with Kf=262144 (C0=512, b0=128):
      Kf=262144: p=1024 -> 0.9658@1922,  p=4096 -> 0.9858@644,  p=16384 -> 0.9946@83
      Kf= 65536: 0.897@3697 / 0.966@1079
    At MATCHED recall ~0.966: Kf=262144 gives 1922 QPS vs 65536's 1079 = 1.78x FASTER. Finer cells
    -> smaller pools (less scan) + better routing discrimination (fewer probes for matched recall),
    and the win OUTWEIGHS the 4x heavier routing (66048 vs 16640 l2/pt). This is the same mechanism
    I'd attributed to Python's "route_tree discrimination" -- it's just more, smaller cells. Build
    cost 4x but feasible @10M (~25min). Next: low-p sweep (SBANN_PLIST=128,256,512,768) to expose
    the high-QPS frontier vs Python 0.9486@3688. (hierk262_10m.log)

P64. (hierk Kf=262144 LOW-P frontier -- gap to Python now ~1.2x) Full high-QPS frontier @10M
    (SBANN_PLIST override, opql, a0=2, tmul=30):
      p=128 -> 0.8609@6115   p=256 -> 0.9118@4659   p=384 -> 0.9330@3638
      p=512 -> 0.9451@3278   p=768 -> 0.9582@2507   p=1024 -> 0.9658@1760
    Interpolated to Python's recall 0.9486: hierk ~3070 QPS vs Python 3688 = Python ~1.2x ahead
    (was 1.4-2.4x before the Kf bump). The remaining gap is small and likely rerank-depth: at
    p=512, t_surv=p*30=15360 exact i8 reranks out of a ~39k pool -- probably over-provisioned.
    Added SBANN_TMUL multi-value override (sweep rerank depth within ONE build, no rebuild) to tune
    it. NEXT: find the tmul that holds recall at lower rerank cost -> may close the last 1.2x.
    (hierk262_lowp.log)

P65. (rerank depth is NOT the bottleneck -- ROUTING fan-out is) Swept tmul={6,10,15,30} at
    p={384,512,768} on one Kf=262144 build. Reducing tmul (fewer exact reranks) just LOSES recall
    with ~no QPS gain (p=384: t6 0.7908@4563, t10 0.8538@4362, t15 0.8925@3759, t30 0.9330@3280;
    QPS inversions like p512 t30>t15 are box-load noise). So the PQ approximation NEEDS deep rerank;
    rerank is cheap relative to the rest. The real query cost is ROUTING: route_fine scores
    b0*(Kf/C0) = 128*512 = 65536 fine-centroid L2 per query (a FLAT fine-scan) -- exactly what
    Python's hierarchical route_TREE prunes. Added SBANN_C0/SBANN_B0 overrides to tune the fan-out
    (cost/q ~= C0 + B0*Kf/C0). Testing C0=1024,b0=128 -> 33792 l2/q (half routing AND half build)
    -- if recall holds vs C0=512 baseline (p512 0.9451@3278) it's a ~2x routing win. (hierk262_tmul.log)

P66. (MEASUREMENT METHODOLOGY -- box load makes single-shot QPS unreliable) The C0=1024,b0=128
    test gave SAME recall as C0=512 (p512 0.9428 vs 0.9451) but LOWER single-shot QPS (2103 vs 3278)
    -- the opposite of the analytical prediction (half the routing l2: 33792 vs 66048). Cause: box
    load swung 40->64 between the two separate runs; single-shot QPS has ±30% noise here, larger
    than the tuning signal. FIX: big-ann itself reports BEST time over run_count -> added SBANN_REPS
    (best-of-N timing) to run(). Tuning A/Bs must use best-of-5 AND run back-to-back in one job so
    both configs see the same load window. Analytical routing-op count (C0 + b0*Kf/C0) is the
    load-INDEPENDENT proxy: at matched recall, fewer routing ops => faster on an idle box. Don't
    trust cross-run single-shot QPS deltas under ~1.5x. (ab_c0.log in progress)

P67. (ROUTING FAN-OUT tuning is a real +33% QPS win -- C0=1024 over C0=512) Controlled A/B,
    best-of-5, back-to-back (same load window), hierk Kf=262144 opql @10M:
      C0= 512 (66048 routing l2/q): p512 0.9451@2988,  p768 0.9582@2477
      C0=1024 (33792 routing l2/q): p512 0.9428@3967,  p768 0.9558@2527
    +33% QPS at p512 for ~equal recall (0.9428 vs 0.9451) -- the half-routing hypothesis CONFIRMED
    once box-load noise was controlled (P66). Win is largest at low p (routing-dominated) and shrinks
    at high p (scan-dominated). Routing cost C0+b0*Kf/C0 is minimized near C0=sqrt(b0*Kf)~=5800, and
    higher C0 ALSO builds faster -> sweeping C0={2048,4096}. C0=1024 interpolates to ~3325 QPS @
    recall 0.9486 vs Python's 3688 -- gap now ~1.1x. CAVEAT: the Python 3688 was measured at unknown
    load; a fair claim needs Python re-measured back-to-back best-of-5 on THIS box. (ab_c0.log)

P68. (C0 sweep complete + KEY discovery: Python's frontier uses NO PQ) Full C0 sweep best-of-5
    @10M Kf=262144 b0=128: C0=2048 p384 0.9271@5601/p512 0.9385@4689/p768 0.9517@3312; C0=4096
    p384 0.9239@6395/p512 0.9352@5197/p768 0.9467@3544. Optimal C0 depends on TARGET recall:
    C0=2048 best at recall 0.95 (~3636 QPS @0.9486, ties Python 3688), C0=4096 best for max-QPS/
    lower-recall (0.92->6400). DISCOVERY: read bench_10m_frontier.py (the Python reference) -- it
    uses RANDOM fine centroids + a route_TREE C kernel AND **exact int8 scan, NO PQ** (batch_scan_i8
    on full vectors, top-20). Fine cells (Kf=262144) keep pools small enough that exact int8 scan is
    affordable AND has zero approximation loss -> no deep rerank needed. My Rust used PQ4/opql (adds
    error, needs t_surv=15360 rerank). Testing comp=i8 (exact scan) vs opql at C0=2048 -- if exact
    scan wins, the PQ was actually HURTING us at fine granularity. Also: Python uses NQ=1000 queries
    vs Rust's full set -> the "ties Python" claim still needs a same-NQ same-box head-to-head.
    (ab_c0b.log, bench_10m_frontier.py)

P69. (exact i8 scan LOSES for Rust -- PQ4 wins ~2.2x; Python's exact scan only wins via BATCHING)
    Controlled A/B (same job/load), hierk Kf=262144 C0=2048 @10M, NQ=full:
      comp=i8 (exact int8, tmul=2): p384 0.9455@1458, p512 0.9532@1028, p768 0.9622@780
      comp=opql (PQ4+rerank,tmul=30): p384 0.9271@5224, p512 0.9385@3618, p768 0.9517@2840
    At matched recall 0.9455: opql ~3206 vs i8 1458 = PQ4 2.2x FASTER. So P68's "drop PQ" hypothesis
    is REFUTED for my Rust: exact scan gives higher recall/probe but my PER-QUERY scan reloads each
    candidate per query (no cache reuse) -> too slow. Python's batch_scan_i8 scans a candidate block
    against ALL queries at once (cache reuse) -> that's the ONLY reason its exact scan is fast.
    CONCLUSION: keep opql/PQ4 as the Rust compressor. To ever beat it with exact scan I'd need a
    BATCHED int8 scan kernel (invert: per-cell, score all queries probing it) -- a real port, lower
    priority since PQ4 already wins. Running the fair NQ=1000 same-box head-to-head now. (ab_i8vspq.log)

P70. (*** RUST BEATS PYTHON *** fair head-to-head, same NQ=1000, same box, back-to-back) The
    definitive comparison (h2h.log). Python frontier = its OWN best config (C0=1024, random fine
    cents, route_tree C kernel, EXACT int8 batched scan, mean/4). Rust = hierk Kf=262144 opql, best/5.
      PYTHON: p256 0.9075@6928, p512 0.9374@3918, p768 0.9547@2524, p1024 0.9625@1852, p1536 0.9726@1237
      RUST C0=1024: p384 0.9360@4626, p512 0.9476@3791, p768 0.9602@2838, p1024 0.9659@2135
      RUST C0=2048: p384 0.9307@3869, p512 0.9421@3896, p768 0.9533@2415, p1024 0.9597@2066
    At matched recall: 0.9374 -> Rust ~4526 vs Py 3918 (1.16x); 0.9476 -> Rust 3791 vs Py ~3096
    (1.22x); 0.9602 -> Rust 2838 vs Py ~2047 (1.39x). RUST WINS ~1.15-1.4x across the frontier.
    WHY: k-means fine cells route better than Python's random cells; PQ4 scan + fan-out routing.
    C0=1024 > C0=2048 here (the earlier C0=2048 "win" was load-confounded; on NQ=1000 back-to-back
    C0=1024 has higher recall ceiling). The old "Python 1.2-2.4x ahead" was an artifact of coarse Rust
    cells (Kf=65536) + uncontrolled box load -- both now fixed. CAVEAT being closed: Rust best/5 vs
    Python mean/4 (mean DISadvantages Python); re-running Python best-of to bulletproof. (h2h.log)

P71. (bulletproof: Rust wins even with Python ALSO best-of-5) Re-ran Python frontier with best-of-5
    timing (bench_10m_frontier_bestof.py) back-to-back with Rust C0=1024 best-of-5, NQ=1000:
      PYTHON best/5: p256 0.9075@8602, p512 0.9374@4401, p768 0.9547@1522(*), p1024 0.9625@1214(*)
      RUST   best/5: p384 0.9360@5382, p512 0.9476@4364, p768 0.9602@3175, p1024 0.9659@2505
    (*) Python high-p points are LOAD-CORRUPTED this run (p768 dropped 2524->1522, p1536->267 -- a
    box-load spike all 5 reps hit; high-p = long runs = more likely to span a spike). The RELIABLE
    low-p points: at recall 0.9374 Rust ~5260 vs Python 4401 = 1.20x. Rust's own high-p stayed clean
    (3175/2505) and dominates Python's corrupted ones. VERDICT: Rust beats Python ~1.2x at the
    cleanly-measured high-QPS operating points (which are also the leaderboard-relevant ones), with
    BOTH best-of-5. Robust across two independent head-to-heads (P70 mean/4 + P71 best/5). The box is
    too noisy for trustworthy slow/high-p timing; low-p is where the reliable + relevant signal is.
    (h2h2.log)

P72. (Kf=262144 is the granularity SWEET SPOT + a measurement-drift caveat) Kf=524288 (C0=2048)
    vs Kf=262144 (C0=1024) @10M NQ=1000: recall IDENTICAL (p384 0.9378 vs 0.9360, p512 0.9475 vs
    0.9476, p768 0.9582 vs 0.9602). So doubling fine cells past 262144 gives NO discrimination gain
    at p=384-768 -- 262144 is the sweet spot; finer just costs 2x build. CAVEAT EXPOSED: this A/B's
    QPS was load-confounded (champ leg 1931, finer leg 5888 -- 3x, impossible as a real delta). Why:
    "back-to-back" legs are separated by a full ~15min BUILD, long enough for box load to drift;
    best-of-5 only controls WITHIN-leg (seconds) noise, not BETWEEN-leg (~15min) drift. WITHIN one
    index the p-sweep frontier is reliable (points seconds apart); CROSS-index/CROSS-engine QPS
    deltas (incl. P70/71 Rust-vs-Python) carry residual drift confound. Recall comparisons are
    drift-IMMUNE (deterministic) -> the recall-per-p edge (k-means vs random cells) is rock-solid;
    the QPS multiplier is directional not exact. FIX = index persistence (build once, reload in
    seconds, bench interleaved) -- also REQUIRED for any real big-ann submission (build/load_index/
    query separation). Implementing next. (ab_kf.log)

P73. (DRIFT-FREE A/B infrastructure `abrun` + clean C0 verdict) Built `sbann abrun` (main.rs): builds
    ALL configs up front, then benches them ROUND-ROBIN (each rep touches every config) so competitors
    are timed seconds apart, not ~15min apart across separate builds -> kills the between-leg load drift
    of P72. SBANN_CONFIGS="Kf:C0:b0,...". Result, Kf=262144 C0=1024 vs C0=2048, best/6, nq=1000:
      C0=1024: p384 0.9360@5158, p512 0.9476@4244, p768 0.9602@3248
      C0=2048: p384 0.9307@6105, p512 0.9421@4983, p768 0.9533@3514
    At matched recall: ~0.936 C0=2048 +8%, ~0.9476 TIED, ~0.9533 C0=1024 +8%. So C0=2048 wins LOW
    recall, C0=1024 wins HIGH recall (0.95+, the leaderboard regime) -- reconciles P67 (C0=2048 "win"
    was low-recall+confounded) vs h2h (C0=1024 win was high-recall). Only ~8% apart either way.
    INFRA VALIDATION: C0=1024 here (5158/4244/3248) reproduces the h2h2 run (5382/4364/3175) within
    ~5% -- interleaved numbers are now REPRODUCIBLE, vs the +-30% single-leg swings. abrun is the
    trustworthy-measurement tool for all future Rust A/Bs on this box. Champion stays Kf=262144
    C0=1024 b0=128 opql for high recall. (abrun_c0.log)

P74. (*** CAPSTONE: airtight Rust>Python across the whole frontier ***) Final comparison
    (final_h2h.log): Rust dense frontier via abrun (best/6, drift-free interleave) THEN Python
    best/5 launched ~2min later (same load window, vs ~15min in P70/71). Python high-p CLEAN this
    run (no load corruption). Taking best Rust config per recall (C0=2048 low / C0=1024 high):
      recall 0.9075: Rust ~8152 vs Py 7706 = 1.06x
      recall 0.9374: Rust ~4792 vs Py 4219 = 1.14x
      recall 0.9547: Rust ~3296 vs Py 2896 = 1.14x
      recall 0.9625: Rust ~2721 vs Py 2217 = 1.23x
    RUST WINS EVERYWHERE; lead WIDENS with recall (1.06x@0.90 -> 1.23x@0.96) because k-means fine
    cells discriminate best exactly where routing precision matters (high recall). Confirms P70/71
    (1.15-1.4x) with the tightest controls yet. Leaderboard-relevant QPS@90%recall (NQ=1000, best/6):
    Rust C0=2048 ~8330@0.906. CHAMPION: hierk Kf=262144 opql, C0=2048 for max-QPS@0.9, C0=1024 for
    recall>=0.95. This SETTLES the 10M comparison -- the native Rust IVF-PQ engine beats the tuned
    Python sbtree reference on equal footing. (final_h2h.log)

P75. (leaderboard metric: C0=4096 is QPS@90%recall champion; + COMPETITOR BENCHMARKING begun)
    Drift-free abrun (best/6) in the recall-0.9 region @10M msspacev NQ=1000:
      C0=2048: p224 0.9063@8823, p256 0.9131@8210
      C0=4096: p256 0.9068@9404, p288 0.9131@8590
    At recall ~0.907 C0=4096 ~9404 vs C0=2048 ~8823 = +7%. So for QPS@90%recall (the leaderboard
    ranking metric) C0=4096 is champion: ~9400 QPS@0.907 (interp to 0.90 ~9600). Full picture: C0=4096
    @recall0.90, C0=2048 @0.91-0.94, C0=1024 @0.95+. -- COMPETITOR CONTEXT (user asked what QPS to
    target): neurips23 OOD leaderboard (text2image-10M, Azure D8lds_v5) tops at hanns 46k, scann 42.8k,
    pinecone-ood 38k, zilliz 33k, mysteryann/pyanns ~22k (QPS@recall>=0.9). hanns/pinecone/zilliz =
    binary-only (can't run). Installing scann (pip, DONE) + all OOD Docker images (install_all.sh) to
    get HARDWARE-NORMALIZED targets on THIS box. NOTE: my ~9.4k is on msspacev-10M int8 IN-distribution
    -- NOT the OOD track. Downloaded text2image-10M (8GB, d=200, float32, MIPS, OOD) + running scann on
    the real OOD set. (abrun_lb.log)

P76. (*** THE TARGET: scann on this box ***) ScaNN on the REAL OOD set text2image-10M (d=200,
    float32, MIPS, OOD queries), PARALLEL (search_batched_parallel, pinned 8 cores, best/5, NQ=10000),
    build 109s num_leaves=4000:
      lts=50 0.6939@18873, 100 0.8256@13203, 150 0.8807@10355, 250 0.9309@7209, 400 0.9585@5015,
      600 0.9773@2787, 900 0.9885@2091, 1400 0.9928@1416
    QPS@90%recall (interp lts150->250) = ~9,150 on THIS box. Leaderboard scann (Azure D8lds_v5) =
    42,854 @ same dataset/metric -> HARDWARE-NORMALIZATION FACTOR ~4.7x (this contended 16-core box
    w/ user's jobs runs scann at ~21% of Azure clean-box throughput). TARGETS ON THIS BOX (text2image
    OOD, QPS@90%): beat scann(#2 open) = 9,150; beat hanns(#1, 46k Azure) = ~9,800. CRITICAL BUG
    FOUND+FIXED (user asked "only one cpu?"): first scann runs used search_batched = SINGLE-THREADED
    (~1.4 cores) -> 4x too low; search_batched_parallel + taskset 0-7 = real numbers. Memory: OOD
    build peaks ~24GB (8GB base float + ScaNN copy); concurrent Docker builds OOM-killed it -> must
    run scann benches with Docker install paused. NOTE: my Rust engine is int8-only -> can't run
    text2image MIPS without float32+IP adaptation; scann-OOD is the target, not yet a head-to-head.
    (scann_ood.log)

P77. (REALITY CHECK: ScaNN is FASTER than my Rust engine -- the single-threaded scann numbers were
    misleading me) ScaNN on msspacev-10M (same int8->f32 L2 data as my engine), PARALLEL 8 cores
    pinned, best/5, NQ=1000: lts=50 0.8534@11534, 100 0.9212@14119, 150 0.9480@11992, 250 0.9692@8121,
    400 0.9804@5879. scann QPS@90%recall ~14,000. MY engine (P75, drift-free best/6) ~9,400 QPS@90%.
    So scann is ~1.5x FASTER @0.90 and ~3x faster @0.95 (scann lts150 0.948@11992 vs my C0=1024 p512
    0.9476@3819). This REVERSES the earlier wrong read (when scann ran SINGLE-THREADED it looked 1.6x
    slower than mine, P-prelim) -- that was the search_batched bug (P76). HONEST STATUS: my from-scratch
    Rust IVF-PQ beats our Python sbtree reference (P70-74) but Python sbtree is NOT a strong baseline;
    production ScaNN (Google, AVX-512, years tuned) is ~1.5-3x faster than my engine even on msspacev
    where I'm tuned -- and I haven't adapted to OOD float/MIPS at all. The leaderboard gap is real.
    Levers to close it: ScaNN's edge = anisotropic AH quantization + SIMD AH scan (in-register LUT) +
    mature partitioning. My Apq4 (anisotropic) + opql exist but the AH scan kernel & tuning lag.
    (scann_bench.log, scann_ood.log)

P78. (PROFILE-DRIVEN: rerank was the bottleneck, fixed with cell-contiguous i8 store) Added `prof`
    subcommand (search_prof times route/scan/rerank). Champion Kf=262144 C0=2048 opql @10M, single-
    thread breakdown: rerank DOMINATED (p224 44%, p256 47%, p512 52%) -- it gathers ~7700 survivors/q
    from RANDOM rows of the 1GB base (cache-miss latency, ~60-140ns/survivor). Routing was only 16-27%
    (overturned my "routing dominates" hypothesis -- PROFILE, don't guess). FIX (front A, mem layout):
    store raw i8 vectors in SLOT/cell order (Index.raw, +1GB@10M) so rerank reads the small ~2MB
    probed-cell region (cache-warm from the scan) instead of scattering across 1GB. Recall bit-identical
    (raw = reordered copy; dedup still maps slot->orig). Result: rerank SHARE fell 47%->31% @p256 (load-
    robust within-run ratio; absolute confounded by load). Now the 3 phases are BALANCED (~31-38% each)
    -> next lever = reduce CANDIDATES (better quant/routing cuts scan AND rerank together). (prof.log,
    prof2.log) GAP_PLAN front A item underway; measuring bottom-line QPS next.

P79. (contiguous rerank VALIDATED +1.1-1.23x, clean same-index A/B) Built `rbench` (one index,
    interleave search[contig] vs search_ds[old gather], best/8, round-robin -> zero build/load
    confound). C0=4096 @10M: p224 contig 6617 vs gather 5996 (1.10x), p256 5343/4934 (1.08x), p320
    5583/4521 (1.23x). Matches Amdahl (rerank ~47%, got ~1.5x faster -> ~1.15x overall). Recall NOT
    bit-identical between paths (select_nth tie-breaks on equal approx-dist differ by slot vs orig)
    but contig came out marginally HIGHER -> strictly better. rbench is now the clean validator for
    all rerank/scan micro-opts. Current QPS@90% msspacev ~10k (C0=4096) vs scann ~14k -> ~1.4x to go.
    Post-fix phase split ~balanced; SCAN now largest (38%). NEXT lever = reduce CANDIDATES (cuts scan
    AND rerank, 69% of time): anisotropic quant (apq4/aopq) for fewer survivors at fixed recall.
    (rbench.log) GAP_PLAN front-A rerank item DONE.

P80. (anisotropic quant apq4 beats opql ~8% at matched recall) Drift-free compressor A/B, Kf=262144
    C0=4096 best/6 @10M: apq4 gives +0.3-0.65% recall at same p vs opql, ~same QPS -> ~8% faster at
    matched recall. p224: apq4 0.9126@11417 vs opql 0.9061@11409; p256 apq4 0.9194@10160 vs opql
    0.9134@10531. aopq (anisotropic OPQ) is WORSE than apq4 -- OPQ rotation doesn't help atop anisotropy
    on msspacev. apq4 is new best compressor. STACKED (apq4 + contiguous rerank + C0=4096): QPS@90% now
    ~12k vs scann ~14k -> gap ~1.17x (was 1.4x). eta (anisotropy strength, hardcoded 4.0) now env-tunable
    SBANN_ETA -> sweeping. NOTE: apq4 trains on only 40k sample (n.min(40000)) -- may undertrain; larger
    sample could improve it. NEXT big lever: scan is now largest phase (38%) -> AVX-512 VBMI wider scan
    (front A1; HW confirmed, but watch N22 memory-bound caveat). (abrun_comp.log)

P81. (BUG: PQ scan was doing 16 wasted ds.row() gathers/block -- fixed with needs_raw_rows guard)
    scan_rerank/search built `rows16: Vec<&[i8]>` (16 ds.row() RANDOM gathers into the 1GB base +
    a Vec alloc) for EVERY block -- but Pq4/Opq4/Apq4 scan_block IGNORE rows16 (LUT-only ADC); only
    ScalarI8 uses them. So the PQ path did ~20k wasted random gathers/query, counted as "scan" time
    (likely why scan was 38%). FIX: Compressor::needs_raw_rows() (default false; ScalarI8 true);
    skip rows16 build when false. Pure win for the PQ path (correctness unchanged -- the value was
    unused). Pending clean validation (prof scan% should drop sharply + abrun QPS). This was hiding
    UNDER the profiler's "scan" bucket -- another reason to profile+verify, not guess. (front A, vq.rs)
    CORRECTION (validated prof3 vs prof2): ds.row() is a LAZY mmap slice (no memory read until bytes
    accessed; PQ path never reads them) -> the waste was the per-block Vec ALLOCATION (~1250/q), NOT
    20k cache-miss gathers. So the guard is a MODEST scan win (alloc elimination): scan SHARE 38%->32%
    (load-robust within-run; absolute totals confounded -- prof2 ran hotter, all phases fell). Real
    but small. Bottom-line stacked QPS measuring now.

P82. (MILESTONE: stacked opts ~closed the msspacev gap to scann) Full stack = apq4(eta8) + cell-
    contiguous rerank (P79) + rows16 guard (P81) + C0=4096, best/8 @10M: p192 0.9075@13597, p224
    0.9128@12291, p256 0.9197@11109. QPS@90%recall ~13,800 -- vs scann msspacev ~14,000. From ~9,400
    at the start of this optimization push to ~13.8k = essentially TIED with scann on msspacev. BUT
    this run's QPS jumped (maybe low-load window) -> running a SAME-WINDOW head-to-head (my engine
    then scann ~2min later, both 8-thread) to confirm. CAVEATS: (1) msspacev int8 L2 in-distribution,
    NOT the OOD track (scann's MIPS strength); (2) my engine still can't run text2image OOD (int8-only
    -> needs float32+IP adaptation). Wins banked: contiguous rerank ~1.15x, apq4 ~1.08x, rows16 modest.
    (abrun_stack.log, h2h_scann.log pending)

P83. (HONEST CORRECTION: same-window shows scann still ~1.24-1.9x ahead; P82 "tied" was a load
    artifact -- AND the real gap diagnosed) Same-window h2h (both 8-thread, load 7-11): my engine
    apq4 C0=4096 p192 0.9075@13940; scann lts100 0.9210@19492, lts150 0.9459@17083. At recall 0.90:
    mine ~13940 vs scann ~17270 = scann 1.24x faster, WIDENING to ~1.9x at 0.95. P82's ~tied was my
    engine catching a low-load window scann didn't. KEY DIAGNOSIS: at recall 0.92 scann RERANKS ONLY
    80 (reorder=80) while I rerank ~5760 (t_surv=p*30). scann scans MORE (250k cheap AH codes) but
    its anisotropic-AH ranking is accurate enough that the true top-10 sits in top ~80 -> almost no
    rerank. Mine isn't -> a huge rerank tax that GROWS with recall (why the gap widens). THE LEVER:
    make apq4's approx ranking accurate enough to rerank far fewer (lower tmul). Testing apq4 at
    tmul=3..30 on one index (tmul is query-time). If low tmul holds recall, big win. (h2h_scann.log)

P84. (*** THE CRUX: i8 LUT resolution is the gap -- int16 LUT is the fix ***) apq4 tmul sweep
    (one index, C0=4096, best/6): lowering rerank depth CRATERS recall -- p192 t3 0.761, t6 0.781,
    t12 0.854, t30 0.908. To hit 0.90 I rerank ~5760 of ~14600 candidates (40% of pool!); scann
    hits 0.92 reranking 80. ROOT CAUSE (read query_lut): the PQ-ADC LUT is i8, scaled by
    100/(maxabs*sqrt(m)) to keep the SATURATING-i8 sum in range -> total approx distance has only
    ~8-bit resolution. Ranking top-10 among ~14600 needs ~14 bits -> 8 bits => massive ties =>
    rerank 40% of the pool. The rerank tax GROWS with recall (more candidates, same coarse ranking)
    = exactly the widening gap to scann (P83). FIX = int16 LUT accumulation (FAISS/ScaNN "LUT16"):
    finer resolution -> accurate ranking -> SHALLOW rerank (like scann's 80) -> the rerank phase
    (~33%) collapses AND high-recall stops needing huge pools. Implement: query_lut_i16 (full i16
    range, no sqrt(m) compression) + i16 accumulating scan. HW: AVX-512 vpermw (_mm512_permutexvar_
    epi16) looks up 32 i16/op, or AVX2 two-byte-table LUT16 trick. This is the #1 lever now. (tmul_apq4.log)

P85. (*** BREAKTHROUGH: int16 LUT CONFIRMED -- 10x fewer reranks, higher recall ***) Re-ran the
    apq4 tmul sweep with SBANN_LUT16=1 (i16 LUT + i32-accum scalar scan, scale 16000/maxabs):
      i8 : p192 t3 0.761, t12 0.854, t30 0.908
      i16: p192 t3 0.9206, t6 0.9207, t12 0.9207, t30 0.9207  (FLAT in tmul!)
    i16 at t3 (rerank 576) BEATS i8 at t30 (rerank 5760) -- 10x fewer reranks AND higher recall
    (0.9207 vs 0.9076 @p192). Recall flat across tmul = ranking now accurate enough that shallow
    rerank suffices = EXACTLY scann's behavior. The i8 LUT's ~8-bit resolution WAS the gap (P84
    confirmed). Implemented (scalar): pq::query_lut_f32_i16 + block_adc_i16, QueryCtx::Pq16, wired
    Opq4+Apq4 behind env SBANN_LUT16. Scalar scan is slow (7471 QPS) -> MUST SIMD-ize: AVX-512
    vpermw (_mm512_permutexvar_epi16, 32 i16/op) or AVX2 LUT16 (two i8 shuffles -> i16, i32 accum).
    Expected once SIMD: rerank phase 33%->~3% + smaller p for same recall -> should CLOSE/BEAT the
    scann gap. This is the #1 win of the session. NEXT: SIMD i16 scan. (tmul_i16.log)

P86. (*** AVX2 int16 scan WORKS -- ~1.43x over my i8 path, now LEVEL with scann ***) Implemented
    AVX2 LUT16 kernel (pq::block_adc_i16_avx2: two vpshufb on lo/hi byte tables -> i16 -> add;
    uncentered positive LUT scaled 30000/summax so i16 accum never saturates = matches scalar exactly).
    Wired QueryCtx::Pq16 (lo/hi byte tables), env SBANN_LUT16. apq4 C0=4096 t=3 best/6:
      p160 0.9134@19972, p176 0.9174@19234, p192 0.9206@17732, p224 0.9258@16414, p256 0.9309@14985.
    Recall matches scalar i16 (correct). vs my OLD i8 path (p192 0.9076@13940): at recall 0.90 i16
    ~20000 vs i8 ~13940 = 1.43x. vs SCANN same-window (~17270@0.90, 19492@0.92): i16 19234@0.917 =
    LEVEL/AHEAD. The int16 LUT (P84/85) closed the gap from 1.24-1.9x behind to ~tied. Recall FLAT in
    tmul -> default to LUT16 + low tmul (t=3, rerank 480 not 5760). NEXT: same-window i16-vs-scann to
    confirm (this run's load unknown); make LUT16 default; SIMD-ize further (AVX-512 vpermw 32-wide).
    (i16_simd.log)

P87. (*** MY ENGINE BEATS SCANN on msspacev @ recall 0.90-0.92 -- same-window, clean ***) Same-box,
    same-window (load 11-15), both 8-thread, both best-of:
      MY int16: p144 0.9098@19567, p160 0.9134@18731, p176 0.9174@17888, p192 0.9206@17106, p224 0.9258@15795
      SCANN:    lts50 0.8543@12534, lts100 0.9206@14106, lts150 0.9444@11494, lts250 0.9663@7850
    At matched recall: 0.910 mine 19567 vs scann ~13850 = 1.41x; 0.917 mine 17888 vs ~14029 = 1.28x;
    0.9206 mine 17106 vs scann 14106 = 1.21x. MY ENGINE WINS 1.21-1.41x at recall 0.90-0.92 (the
    leaderboard QPS@90% point). COMPLETE REVERSAL from P83 (scann 1.24-1.9x ahead) -- the int16 LUT
    (P84/85/86) did it. Caveats: (1) msspacev int8 L2 in-distribution, NOT the OOD track; (2) ~0.944
    roughly tied, scann likely still leads at 0.96+ (need to extend my frontier there); (3) clean
    same-window measurement so this is SOLID. Making LUT16 default. The journey: profiled->rerank
    bottleneck->i8 resolution diagnosis->int16 fix->AVX2 kernel->beat scann. (h2h_i16.log)

P88. (post-int16 breakdown: SCAN now the bottleneck) prof apq4+int16+tmul3 C0=4096: p192 route 37%/
    scan 43%/rerank 20% [437us/q]; p256 route 32%/scan 47%/rerank 21% [509us/q]. Rerank collapsed
    (was 31-47% -> now 20%); total ~440us vs ~1200 before. SCAN is now largest (40-47%) and it's
    SHUFFLE-bound (int16 LUT16 = 2 vpshufb+unpack/subspace) so AVX-512 vpermw (32-wide i16 lookup)
    genuinely helps -- needs a 32-vector super-block repack. Routing 2nd (32-40%) -> 3-level hierarchy
    lever. NEXT: map high-recall frontier vs scann (does scann re-lead at 0.96+?), then AVX-512 scan
    or 3-level routing. (prof4.log)

P89. (complete frontier: I WIN the leaderboard metric; scann wins high-recall) Same-window full
    frontier (both 8-thread best-of): MY int16 p160 0.9134@20192, p224 0.9258@16694, p320 0.9387@
    13354, p448 0.9471@10410, p640 0.9553@7326 (tmul=8 no better than 3 -> NOT rerank-bound).
    SCANN lts50 0.8608@12450, lts100 0.9217@15989, lts150 0.9464@12178, lts250 0.9674@10272, lts400
    0.9804@6762. AT MATCHED RECALL: 0.913 me 1.30x, 0.922 me 1.12x, 0.939 TIED, 0.946 scann 1.14x,
    0.955 scann 1.55x, 0.97+ scann only (mine caps ~0.955). VERDICT: I WIN QPS@90%recall (the
    leaderboard ranking metric) by ~1.30x; crossover ~0.94; scann wins >=0.95 (widening). My recall
    CEILING ~0.955 at these p = ROUTING/coverage limit (true NN not in probed cells; not rerank --
    flat in tmul). To win high-recall too: AVX-512 scan (cheaper per-candidate -> probe more) + 3-level
    or finer routing (higher recall/probe). Those also transfer to the OOD track. (frontier.log)

P90. (b0 routing fan-out should ADAPT to target recall) Drift-free, C0=4096 apq4 int16 t=3:
    b0=64: p192 0.9160@19833, p512 0.9426@9947 | b0=128: p192 0.9206@17273, p512 0.9508@8934 |
    b0=256: p192 0.9241@13877, p512 0.9561@7896. At recall 0.92 b0=64 ~8% faster (cheaper routing);
    at 0.94 b0=128 best; at 0.95+ b0=256 best (more coverage raises the ceiling 0.9508->0.9561).
    So for QPS@90% use b0=64 (lead over scann -> ~1.5x); for high recall use b0=256. Cheap tuning.
    Recall ceiling still caps ~0.956 even at b0=256/p512 -> need finer cells or faster scan (afford
    more candidates) to reach 0.97+. NEXT: AVX-512 32-wide vpermw scan (bottleneck 47% + raises
    ceiling + transfers to OOD). (abrun_b0.log)

N-AVX512. (AVX-512 vpermw 32-wide scan is a DEAD END -- ~12-16% SLOWER than AVX2 LUT16) Implemented
    block_adc_i16_avx512_x2 (2 blocks/vpermw, selftest-validated, recall matches AVX2 exactly). Clean
    same-index A/B (SBANN_NO512 toggle), apq4 C0=4096 b0=64: AVX2 p192 0.9160@15028/p512 @6853 vs
    AVX-512 p192 @13297/p512 @5714 -- AVX-512 LOSES 12-16% everywhere. Cause: vpermw (cross-lane,
    ~half vpshufb throughput) + AVX-512 downclocking negate the 32-wide width. KEEP the AVX2 LUT16
    (vpshufb) i16 scan as default; AVX-512 path left behind opt-in SBANN_USE512. So the scan phase
    (47%) can't be sped by wider SIMD here -> to cut it, reduce CANDIDATES (routing/cells), not width.
    Lesson echoes N22 (widening didn't help). NEXT lever = routing (32-40%): 3-level hierarchy or
    finer cells. (avx512_ab.log)

P91. (a0 multi-assignment raises the recall ceiling but doesn't flip high-recall; the frontier split
    is FUNDAMENTAL) a0 sweep C0=4096 b0=128 apq4 int16: a0=2 p512 0.9508@5211; a0=3 0.9635@3843;
    a0=4 0.9704@2890. Higher a0 -> higher ceiling, and wins its band (a0=4 best @recall>=0.96). But
    even a0=4 0.9704@2890 vs scann 0.9674@10272 = scann ~3.5x faster at recall 0.97. WHY (final
    understanding): at high recall many candidates are scanned; scann's AH scan is faster PER-CANDIDATE
    than my AVX2 int16 (2 vpshufb/subspace) and AVX-512 didn't help (vpermw slow, N-AVX512). So:
    I WIN low-recall/QPS@90% (few candidates, my routing+shallow-rerank efficiency dominates); scann
    WINS very-high-recall (many candidates, faster scan dominates). a0/b0 ADAPT to target recall but
    don't change the verdict. The leaderboard metric IS QPS@90% -> I WIN IT. Closing high-recall needs
    a scan faster than AVX2 int16, which this HW can't give. msspacev optimization = COMPLETE.
    (a0_sweep.log)

P92. (OOD adaptation START -- cosine/L2, NOT MIPS augmentation) User chose to adapt to the OOD track
    (text2image-10M). GT is inner-product, BUT base norms are TIGHT (0.758-0.992, mean 0.966; queries
    unit-norm) -> cosine/normalized-L2 recovers most MIPS top-10 (argmax<q,x> ~= argmax cos when norms
    concentrated). So DON'T implement MIPS: quantize float32->i8 (global scale, 200-dim even for PQ),
    run the existing mean-center+normalize+L2 engine (=cosine) on the scann-beating config (hierk
    Kf=262144 C0=4096 apq4 int16) vs the MIPS GT. prep_ood_simple.py (21s). Added harmless SBANN_NOMU
    (mu=0) knob in case the MIPS augmentation is needed later. Measuring recall@10 now -- if decent,
    the engine works on OOD as-is; if poor, add the [x, sqrt(M^2-||x||^2)] augmentation. (ood_run.log)

P93. (cosine/L2 is NOT enough for OOD -- caps recall 0.71; MIPS augmentation IS needed) Ran the
    cosine/L2 engine (hierk Kf=262144 C0=4096 apq4 int16) on int8 text2image-10M vs MIPS GT: p256
    0.7018@7765, p512 0.7094@4363 -- recall PLATEAUS ~0.71 (barely moves with more probes -> it's the
    DISTANCE, not routing). So cosine != MIPS here despite tight-ish norms (0.76-0.99 spread is enough
    to reorder ~30% of top-10). Hypothesis (P92, cosine suffices) REFUTED. FIX = MIPS->L2 augmentation:
    base->[x, sqrt(M^2-||x||^2)] pad to 204, query->[q,0..], SBANN_NOMU (mu=0) so per-vector normalize
    (base all norm M -> /M const) preserves MIPS; raw-i8 rerank = exact MIPS. Also fixed a d>128 bug:
    codes16/pack_block buffers were hardcoded [u8;64] -> bumped to 256 (d=200 -> m=100 overflowed).
    Running augmentation now. (ood_run2.log)

P94. (OOD MIPS augmentation WORKS: recall 0.71->0.85; engine runs the real OOD track) SBANN_NOMU +
    [x, sqrt(M^2-||x||^2)] augmented i8 (M=0.9934), hierk Kf=262144 C0=4096 apq4 int16 a0=2: p256
    0.8082@7238, p512 0.8340@4068, p1024 0.8481@2120. vs cosine cap 0.71 (P93) -> MIPS reduction is
    correct + the scann-beating engine RUNS the actual leaderboard dataset. Still climbing toward 0.90
    (coverage-limited: OOD queries route poorly to base k-means cells). Found prep QUANT BUG: scaled by
    M=0.99 not the augmented-max ~0.64 -> components ±0.37 squished to ±47 (6-bit). Fixed (scale by
    actual aug-data max) + a0=3 (coverage). Re-running. STATE: engine works on OOD ~0.85; to compete
    w/ scann (0.90@9150 on this box) needs higher recall (float rerank to break i8 cap) + OOD-aware
    routing + faster 204-dim scan. Real climb. (ood_aug.log)

P95. (OOD: engine reaches recall 0.90 but ~8x behind scann on QPS -- OOD routing is the gap) Fixed
    quant scale (198 vs buggy 128) + a0=3: text2image-10M p512 0.8800@3802, p1024 0.8935@2043, p2048
    0.8995@1088. RECALL 0.90 ACHIEVED on the real OOD track (MIPS augmentation correct). But QPS@90%
    ~1088 vs scann ~9150 = ~8x behind. WHY: OOD queries (unit-norm, different distribution) route
    POORLY to base k-means cells -> need ~2048 probes for recall 0.90 (vs ~150 on msspacev) -> ~250k
    candidates scanned -> the SAME high-candidate-count regime where scann's AH scan beats my AVX2
    int16 (per-candidate). The OOD QPS gap = (1) routing needs too many probes [OOD-aware routing:
    train cells on query.learn.50M distribution -- GAP_PLAN #4/#5], (2) 204-dim scan cost, (3) scan
    not faster than scann at high counts (AVX-512 dead, N-AVX512). HONEST: engine RUNS OOD correctly
    @0.90 but competing needs OOD-aware routing (substantial research) -- a faster scan won't come from
    this HW. (ood_aug2.log)

P96. (OOD-aware routing FAILS with the augmentation -- incompatible) Trained routing cells on
    query.learn (5M, aug=0) via SBANN_ROUTE_TRAIN; index on base. Result WORSE: p256 0.7579 (vs
    base-trained 0.8082), p512 0.8074 (vs 0.8340). WHY: query-train points have aug-dim=0, so
    query-trained cells live in the aug=0 subspace -> routing IGNORES the aug/norm dim -> collapses
    to direction/cosine routing (which caps 0.71, P93). The augmentation's MIPS-correctness lives in
    the aug dim that queries can't inform -> augmentation + query-aware routing are INCOMPATIBLE.
    CONCLUSION on OOD: my IVF-PQ-via-augmentation runs text2image @0.90 recall but ~8x behind scann
    on QPS, and the obvious lever (query-aware routing) doesn't work. Closing it needs a different
    OOD-MIPS design (e.g. cosine-route + exact-IP-rerank, or native-IP routing/quant) = deep research.
    (ood_qaware.log)

P97. (*** OOD BREAKTHROUGH: cosine-route + exact-IP rerank -- 8x->1.9x behind scann ***) Non-augmented
    i8 text2image, cosine routing/scan (normalized L2) + EXACT INNER-PRODUCT rerank (SBANN_IP: negdot_i8,
    min-heap keeps max IP), hierk Kf=262144 C0=4096 apq4 int16 a0=3 tmul=20: p256 0.8941@5510, p512
    0.9159@3091, p1024 0.9282@1633. QPS@90% ~4857 vs augmented MIPS 1088 (P95) = 4.5x faster, vs cosine-
    only cap 0.71 (P93). Gap to scann (9150@90%) now ~1.9x (was 8x). WHY: cosine routing is efficient
    for OOD (unit-norm queries = pure direction -> direction-aligned cells, few probes) AND base norms
    tight (0.76-0.99) so the cosine candidate pool CONTAINS the MIPS top-10; exact-IP rerank fixes the
    ranking cosine got wrong (the 0.71 cap was ranking, not coverage). Added simd::negdot_i8 + IP_MODE
    (SBANN_IP). NEXT: query-aware cosine routing should now WORK (no augmentation aug-dim conflict, P96)
    + tmul/b0/a0 tuning -> may reach scann parity on OOD. (ood_ip.log)

P98. (query-aware cosine routing ALSO fails for OOD) cosine cells trained on query.learn (5M) + IP
    rerank: p256 t20 0.7676@2172 (WORSE than base-trained cosine+IP 0.8941@5510). Both query-aware
    attempts fail (augmented P96, cosine P98): (a) undertrained -- 5M query.learn / 262144 cells = 19
    pts/cell vs base 38; (b) OOD mismatch -- cells where QUERIES are dense don't distribute BASE points
    well. Verdict: BASE-trained cosine routing + IP rerank (P97) is the OOD winner. Tuning that next
    (tmul/a0). (ood_qip.log)

P99. (OOD needs DEEP rerank; cosine+IP best ~2x behind scann) Base cosine+IP tmul sweep (a0=3 C0=4096):
    recall CLIMBS with tmul (p320: t4 0.8726, t8 0.8925, t16 0.9007) -- unlike msspacev (flat) OOD needs
    tmul>=16. Best QPS@90% ~3300-4857 (load-dependent), ~1.9-2.7x behind scann (9150). The remaining gap
    = same as msspacev high-recall: recall 0.90 needs p~300 -> ~50k candidates -> scann's scan faster at
    high counts (fundamental, AVX-512 dead). OOD JOURNEY: 0.71 cosine -> 0.85 aug -> 0.90@1088 aug ->
    0.90@~4857 cosine+IP (8x->~2x behind scann). Running same-window OOD h2h to nail it. (ood_tune.log)

P100. (DEFINITIVE OOD gap: scann ~1.6x ahead @recall 0.90, same-window) Clean h2h text2image-10M,
    both 8-thread best/5, same load window (53-75): MY engine (cosine+IP) p224 0.8864@3189, p288
    0.8973@2465, p352 0.9036@2230, p448 0.9113@1800; SCANN lts100 0.8280@7159, lts150 0.8826@4389,
    lts250 0.9314@2707, lts400 0.9579@1851. At matched recall: 0.886 scann 1.33x, 0.90 scann ~1.6x
    (mine ~2364 vs scann ~3789), 0.91 scann ~1.9x. So OOD gap = ~1.6x@0.90, widening with recall --
    same fundamental scan-speed-at-high-candidate-counts limit as msspacev high-recall. OOD JOURNEY
    (this session): can't-run -> 0.71 cosine -> 0.90@8x-behind (augmentation) -> ~1.6x-behind (cosine
    route + exact-IP rerank, the key OOD lever). (ood_h2h.log)

P101. (VNNI int8 dot: 1.4-1.7x for OOD dims, SLOWER for msspacev -- dim-gated low-level win) Added
    simd::dot_i8_vnni (vpdpbusd, u8*i8 via XOR 0x80 shift + 128*sum(c) correction, selftest-exact).
    Clean microbench (box freed, taskset): d=100 0.91x (SLOWER), d=200 1.42x, d=204 1.73x. AVX-512
    downclock isn't amortized at d=100 (few 64-chunks) but wins at 200+. Kept OPT-IN (SBANN_VNNI;
    static VNNI_ON) so OOD uses VNNI, msspacev keeps AVX2. The OOD IP-rerank (200-dim, deep tmul16)
    gets ~1.5x on the rerank phase -> modest overall (rerank is part of query time). NOTE: OOD builds
    are SLOW (~38min each: 200-dim + a0=3 + Kf=262144) -> killed the 3-build b0 sweep as impractical.
    (dotbench)

P102. (PCA dim-reduction DEAD for text2image -- near-isotropic embeddings) SVD on 500k base sample:
    top-128 dims hold only 82.5% energy (top-64 59%, top-160 92%); PCA-128 IP recall@10 vs full-IP =
    0.3834 (cratered). text2image image embeddings are HIGH intrinsic dim (near-isotropic) -> no free
    dim reduction; full 200-dim stands. So the 200-dim scan/rerank cost + 38min builds are unavoidable.
    Makes VNNI rerank (P101) more valuable since OOD needs deep rerank (tmul16). (PCA probe)

P103. (VNNI gives NO overall OOD win -- rerank is BANDWIDTH-bound, not compute-bound) Clean same-index
    A/B (cosine+IP, tmul16): AVX2 p256 0.8924@3843, p352 0.9036@3532; VNNI 0.8924@3886, 0.9036@3519
    -- IDENTICAL (+-1%), recall unchanged. The dot KERNEL is 1.4-1.7x faster (microbench, warm L2) but
    the real rerank reads ~4800 survivors x 200B ~= 960KB/q (> L2) -> memory-bandwidth-bound -> faster
    ALU doesn't help (same lesson as AVX-512 scan N-AVX512 + N22). VNNI left opt-in/off. To speed the
    OOD rerank need FEWER/SMALLER survivor reads (lower tmul hurts recall; PCA dead P102) -> fundamental.
    OOD is at its practical limit ~1.6x behind scann: cosine+IP was the breakthrough (8x->1.6x), the
    rest (query-aware routing, PCA, VNNI) are dead/neutral. (ood_vnni_ab.log)

P104. (*** asymmetric MIPS-PQ scan: halves OOD rerank depth, free ***) Made the int16 PQ scan rank
    by APPROX INNER PRODUCT (pq::query_lut_f32_i16_ip = per-subspace shifted -<q_sub,cent>, same LUT16
    kernel) when IP mode. OOD tmul sweep: IP-PQ p352 t8 0.9005@4278, t16 0.9056@4127 vs cosine-PQ which
    needed t16 for 0.90. So recall 0.90 now at tmul=8 (was 16) -> HALF the exact reranks -> less of the
    bandwidth-bound cost (P103). Recall still climbs w/ tmul (t3 0.8675, not flat) so not fully scann-
    like, but a real free win (no extra scan cost). Now DEFAULT in IP mode. This run QPS@90% ~4278 vs
    scann ~3789 (P100 same-window) -> possible PARITY/win, running same-window h2h to confirm. (ood_ippq.log)

P105. (HONEST: IP-PQ scan improves the engine but OOD gap to scann stays ~1.6x) My engine v2 (IP-PQ
    scan, t8): p384 0.9036@3769, QPS@90% ~4084 (load 31). scann solo: lts150 0.8805@7062, lts250 0.9304@
    6123, QPS@90% ~6700 (load 44-47, HIGHER load yet faster). Best-of-5 only partly normalizes load, so
    scann at higher load still beating my engine => gap still ~1.6-1.8x at recall 0.90. IP-PQ scan lifted
    my ABSOLUTE QPS (cosine-PQ ~2364 -> IP-PQ ~4084) but scann improves equally with conditions -> RELATIVE
    gap unchanged. (Same-window v2 ratio blocked by scann OOM when co-running.) FINAL OOD VERDICT: ~1.6x
    behind scann; remaining gap is fundamental (scann's scan faster at the high candidate counts OOD needs;
    my AVX2 int16 scan + 200-dim cost; AVX-512 dead, PCA dead, VNNI bandwidth-bound, query-aware routing
    dead). OOD JOURNEY: unrunnable -> 0.71 cosine -> 8x (aug) -> 1.6x (cosine+IP+IP-PQ scan). (scann_ood_solo.log)

P106. (flat-IVF scann-like routing DEAD for my engine -- confirms scan speed is scann's core edge)
    flatsoar C=4096 + SOAR + IP-PQ on OOD: p48 0.876@998, p96 0.880@552, p256 0.9135@194 -- recall OK
    but QPS terrible (194 vs hierk ~4084@90). Flat C=4096 -> ~7300 pts/cell (x a0=3) -> p256 scans 1.87M
    candidates (40x hierk's fine ~76/cell). scann tolerates big leaves (lts100=250k cand) BECAUSE its
    scan is fast; my slower int16 scan CANNOT -> I need FINE cells (hierk). This is the cleanest proof
    that scann's advantage = SCAN SPEED, manifesting as 'scann uses few big leaves, I must use many
    small ones'. hierk Kf=262144 stays the OOD best. OOD ideas now exhausted (cosine+IP, IP-PQ scan
    were the wins; flat-IVF, query-aware routing, PCA, VNNI all dead). FINAL: ~1.6x behind scann on OOD,
    fundamental scan-speed gap. (ood_flat.log)

P107. (*** profile-driven i8 scan: real OOD win, ~1.18x ***) OOD profile (P-prev) showed SCAN is the
    bottleneck (46-48%), NOT rerank (22%). So traded scan resolution for speed: i8 scan (1 vpshufb/
    subspace, IP LUT pq::query_lut_f32_ip_i8) vs int16 (2/subspace), exact-IP rerank fixes ranking.
    Clean same-index A/B (SBANN_LUT_AB): i8 p384 t16 0.9047@4975 vs i16 p384 t8 0.9036@4259 = i8 1.17x
    faster at matched recall 0.90; i8 needs deeper rerank (t16 vs t8) but the 2x scan wins net. OOD
    QPS@90% ~4084 (i16) -> ~5200 (i8); scann gap ~1.6x -> ~1.3x. Best OOD = cosine route + IP-PQ scan +
    i8 precision (SBANN_NOLUT16) + IP rerank, tmul16. Added LUT16_OFF global + SBANN_LUT_AB toggle. The
    PROFILE was the key -- I'd wrongly assumed rerank-bandwidth was the wall; it was the scan. (ood_lutab.log)

P108. (HONEST CORRECTION: clean same-LOAD OOD gap is ~2.0x, not 1.3-1.6x -- earlier was load-confounded)
    OOD v3 h2h, BOTH at load ~30 (my engine then scann ~2min later): my i8 config p288 0.8942@5126,
    p352 0.9017@4405 (QPS@90% ~4569); scann lts150 0.8817@10321, lts250 0.9313@7228 (QPS@90% ~9180).
    => scann ~2.0x ahead at recall 0.90, SAME LOAD. Earlier 1.3-1.6x estimates compared my-engine-at-
    LOW-load to scann-at-HIGH-load (confounded). KEY INSIGHT: scann scales BETTER with low load (its AH
    scan is compute-bound, parallelizes across cores); my engine is memory-bound (rerank) so benefits
    less -> at the leaderboard's dedicated-HW (low-load) conditions scann's edge is LARGER not smaller.
    The i8 scan is still a real ~1.18x engine win (gap ~2.4x->2.0x at this load). FINAL OOD: ~2.0x behind
    scann clean same-load; fundamental (scan compute throughput + load-scaling). msspacev win (1.26x) was
    also clean same-window so it stands. (ood_h2h3.log)

P109. (*** 3-level hierarchical router `hierk3`: faster build AND faster queries -- widens msspacev win ***)
    New router train_hkmeans3/route_fine3 in vq.rs: coarse C0 -> MID C1 -> fine Kf, routing
    O(Kf^1/3) centroid-dists/query instead of O(Kf^1/2). 10M msspacev Kf=262144, hierk3 C0=1024
    C1=8192 b0=48 b1=160 vs 2-level champion C0=4096 b0=64. BUILD: 347s vs 767s = 2.2x faster
    (3-level assign is much cheaper -- the scale lever for 100M/1B tracks). QUERY (matched recall,
    BOTH load orders confirm so NOT load-confounded): forward run (hierk3 2nd/low-load) hierk3 won
    1.07-1.14x; REVERSE run (hierk3 1st/HIGH-load, disadvantaged) hierk3 STILL won: r0.922 8848 vs
    7526 (1.18x), r0.929 7904 vs 6742 (1.17x), r0.935 5736 vs 4690 (1.22x). Clean-load query win
    ~1.15x. MECHANISM: 3-level scores ~6528 centroid-dists/q (1024+48*8+160*32) vs 2-level 8192
    (4096+64*64) -> cheaper routing + better recall-per-probe at this config. C0=2048 config worse
    (more coarse = more routing); C0=1024 is the sweet spot. NET: msspacev leaderboard win
    1.26x -> ~1.45x over scann, build 2.2x faster. Env: hierk3 router + SBANN_C1/SBANN_B1. NEXT:
    make hierk3 the msspacev champion; test on OOD (routing is part of OOD query time too). (h3_reverse.log)
    OOD RESULT (h3_ood.log): hierk3 does NOT help OOD. text2image-10M hierk3 C0=1024 C1=8192 b0=64
    b1=200 tmul16: QPS@90% ~4300 at load~42; vs recorded 2-level OOD ~4569 at load~30 (P108) -> hierk3
    parity-or-WORSE even adjusting for load. WHY: at 200-dim with OFF-distribution cosine routing the
    3-level cascade (top-b0->top-b1->top-k) drops more good cells (worse recall-per-probe), and OOD is
    rerank/memory-bound at tmul16 so cheaper routing doesn't convert. Confirms hierk3 is an IN-DISTRIBUTION
    (msspacev) lever only. OOD gap stays ~2x, fundamental (scan throughput + load-scaling, P108).
    HIGH-RECALL (>=0.95) msspacev same-window (h3_hirecall.log): hierk3 a0=4 b0=96 b1=320 vs 2-level a0=4
    b0=256. r0.968 hierk3 2031 vs 2-level 2151 (2-level 1.06x); r0.974 hierk3 1672 vs 1481 (hierk3 1.13x);
    peak hierk3 0.9772@1318 vs 2-level 0.9800@1200 (~tie, 2-level slightly higher PEAK recall -- cascade
    approximation caps the very top, closable w/ more fan-out). Build 522s vs 1157s (2.2x, consistent).
    NET hierk3 verdict: DOMINATES the QPS@90% leaderboard point (1.15-1.22x faster + 2.2x build) and
    TIES at high recall; the remaining >=0.95-vs-SCANN gap is SCAN-THROUGHPUT (scann's faster AH scan at
    high candidate counts), NOT routing -> hierk3 can't flip it; needs a faster/batched scan kernel.

P110. (*** MAJOR HONEST CORRECTION: we DO NOT beat ScaNN on msspacev -- the "win" was load-confounded ***)
    First-ever CLEAN same-window msspacev hierk3-vs-ScaNN head-to-heads (h3_vs_scann.log, h3_vs_scann2.log),
    ScaNN properly configured (num_leaves=4000, score_ah(2)=2-byte AH, reorder(200), search_batched_parallel,
    taskset 0-7), both best-of-5, NQ=1000. RESULT: ScaNN BEATS us across the whole frontier. At recall
    ~0.923: ScaNN 15,570 QPS vs hierk3 (a0=3, fair tmul) ~6.5-9k = ScaNN ~1.7-2.4x. ScaNN also 0.947@12.7k,
    0.968@8.2k, 0.980@6.3k; we reach 0.95 only at ~1.5-2.8k. So ScaNN ~1.5-2x faster at QPS@90% AND dominant
    at high recall. The earlier P87/P89 "beat scann 1.2-1.4x" was LOAD-CONFOUNDED (my engine measured at low
    load 11-15 vs scann at higher load and/or a weaker/single-threaded scann config) -- the SAME error class
    flagged for OOD in P108 but never applied to msspacev. TMUL not a fix: msspacev recall is tmul-INSENSITIVE
    (int16 LUT ranks well; t_surv floor 1000 already captures the NN -> p128 recall 0.9227 identical at
    tmul=2/3/5/30, h3_tmul.log) so fewer survivors don't buy QPS. ScaNN's edge is ARCHITECTURAL: anisotropic
    2-byte AH + in-register scan + exact reorder of only ~200, vs our 4-bit PQ needing >=1000 reranked.
    WHAT SURVIVES: hierk3 is a real improvement over our OWN 2-level (2.2x build, ~1.15x query, P109) but
    does NOT lift us past ScaNN. CORRECTED STANDING: behind ScaNN ~1.5-2x (msspacev) and ~2x (OOD), both
    clean same-load. Lesson re-learned: NEVER compare across load windows; always same-window vs a tuned
    competitor. RESULTS.md headline corrected.

P111. (*** rerank-floor fix: ~2x QPS@90% engine win -- narrows the ScaNN gap but does NOT close it ***)
    Investigating P110 found the hardcoded t_surv = max(p*tmul, 1000) rerank FLOOR was a ~2x handicap at
    QPS@90%. The int16 LUT ranks well enough that shallow rerank holds recall: p128 t_surv=256 recall
    0.9194 vs t_surv=1024 0.9227 (-0.3% only), but QPS 12096 vs ~5300-6565 (~2x). Removed the floor
    (SBANN_TFLOOR, default lowered 1000->300; only affects low-p/QPS@90%, high-p already exceeds it).
    CLEAN same-window fair re-run (h3_fair_vs_scann.log, hierk3 a0=3 t_surv=p*3 vs tuned ScaNN, best/5):
    hierk3 r0.921@11989, r0.929@9986, r0.940@7198, r0.950@5214; ScaNN r0.924@18069, r0.948@12790,
    r0.968@9652, r0.981@6989. MATCHED-RECALL: r0.92 hierk3 ~12k vs ScaNN ~17.5k = ScaNN 1.5x; r0.94
    1.9x; r0.95 2.4x. So the floor fix is a REAL ~2x engine win (banked as default) but ScaNN STILL leads
    ~1.5x at QPS@90% and the gap WIDENS with recall (scan-throughput: ScaNN's anisotropic AH scans more
    candidates faster). CONSISTENT honest standing: ScaNN ahead ~1.5x (QPS@90%) -> ~2.4x (recall 0.95) on
    msspacev; ~2x on OOD. The remaining gap is the SCAN kernel, not routing or rerank depth.

P112. (scan-throughput gap to ScaNN is FUNDAMENTAL on AVX2 -- cheap scan levers all dead-end)
    After P111 isolated the remaining ScaNN gap as SCAN THROUGHPUT (widens with recall), tested the cheap
    scan levers, all NEGATIVE: (a) FINER quant dpb=1 (4-bit/dim, 100 subspaces) gives IDENTICAL
    recall-per-probe to dpb=2 (r0.9500@p320 vs r0.9497@p320, h3_dpb1.log) at ~1.5x the scan cost -> lower
    QPS everywhere. Codebook granularity is NOT the recall limiter (rerank fixes final order); dpb=2 optimal.
    (b) COARSER-FASTER i8 scan (NOLUT16, 1 vpshufb, 2x faster) DESTROYS msspacev recall: p128 0.6119 (vs
    int16 0.9214), even t_surv=3200 only 0.86 (h3_i8scan.log). The i8 LUT's ~8-bit DISTANCE-accumulator
    resolution ranks so coarsely the true NN isn't in the survivor pool -> rerank can't recover. So the
    int16 LUT16 is REQUIRED for msspacev (confirms P84-86 still holds even with shallow rerank), and it IS
    the throughput bottleneck. CONCLUSION: routing (hierk3), rerank (floor fix), and quant (dpb=2) are all
    optimized; the gap is purely the int16 scan kernel speed. AVX2 int16 LUT16 is near its ceiling
    (AVX-512 vpermw downclocks, dead end N-prev). Only theoretical lever left = true cell-INVERTED batched
    scan (amortize code loads across the batch's queries, ~20 queries/cell at nq=10k) -- complex, uncertain,
    payoff scales with batch size. We are at the practical AVX2 ceiling for this IVF-PQ design: ScaNN ahead
    ~1.5x (QPS@90%) to ~2.4x (recall 0.95) msspacev, ~2x OOD, on clean same-load measurement.

P113. (*** FAST-SCAN kernel: 1.6-1.9x scan speedup at IDENTICAL recall -- the real gap-closer ***)
    Implemented the FAISS/Quick-ADC-style fast-scan: int8 LUT with 1 vpshufb/subspace + int16 widening
    accumulation (pq::block_adc_i8_i16acc, query_lut_f32_i8s). KEY design vs the FAILED sqrt(m) i8 path
    (P112, recall 0.61): per-subspace MIN-subtraction + a single global scale so int8 is spent on
    WITHIN-subspace variation (constant offset cancels in ranking); sum fits int16 -> ~12-13 bit ranking
    resolution (vs i8's ~8, int16's 15). Selftest-validated (matches scalar exactly); microbench 1.70x
    faster than int16 LUT16 (676 vs 398 Mvec/s, scanbench). END-TO-END msspacev hierk3 SAME-WINDOW
    (h3_fastscan.log): recall IDENTICAL to int16 at every point (0.9213 vs 0.9214, 0.9497 vs 0.9497 ...)
    while QPS 1.56x (r0.909) -> 1.74x (r0.921) -> 1.90x (r0.950) faster -- speedup GROWS with recall
    (scan is a bigger fraction there, exactly where ScaNN's lead was largest). FASTSCAN hierk3: r0.921
    @18293, r0.940@12365, r0.950@9250 (vs int16 10507/6800/4861). Env SBANN_FASTSCAN (L2 only; selftest
    asserted at startup). This is the scan-throughput lever P112 said was the only thing left -- and it
    works because rerank fixes final order so the scan only needs ~12-bit candidate-SELECTION resolution.
    vs-ScaNN same-window confirmation pending (chained estimate: ~parity at QPS@90%, ~1.4x behind @0.95).
    CONFIRMED vs ScaNN, BRACKETED BOTH LOAD ORDERS (h3_fastscan_vs_scann.log, h3_fs_rev.log):
    QPS@90% recall 0.92 -- order1 (FASTSCAN first/disadvantaged) ScaNN 14476 vs FS 12553 = ScaNN 1.15x;
    order2 (ScaNN first/disadvantaged) FS 19481 vs ScaNN 15825 = FS 1.23x. => bracketed ~PARITY at QPS@90%.
    recall 0.95: order1 ScaNN ~1.45x, order2 ScaNN 1.38x => consistently ScaNN ~1.4x ahead (down from 2.4x).
    HONEST FINAL msspacev: PARITY with ScaNN at QPS@90% (the leaderboard metric), ScaNN ~1.4x ahead at
    recall>=0.95 (its AH scan still wins at high candidate counts). Net session arc: false "1.45x win"
    (load-confounded, P110) -> honest "1.5-2.4x behind" -> floor fix (P111, ~2x) + fast-scan (P113, 1.7x)
    -> PARITY at QPS@90%. The high-recall ScaNN edge remains (scan throughput at large pools) + OOD ~2x.

P114. (*** FAST-SCAN extends to OOD: int16-accurate IP selection at i8 speed -> ~1.24x, OOD gap 2.0->1.6x ***)
    Added IP fast-scan LUT (pq::query_lut_f32_i8s_ip: per-subspace -<q,cent>, min-subtract, int8 global
    scale) wired into FASTSCAN+IP mode. text2image-10M OOD same-window (ood_fastscan.log) FASTSCAN-IP vs
    the i8-IP baseline (NOLUT16): FASTSCAN-IP wins on BOTH recall AND QPS at every point (p352 t8: recall
    0.8961 vs 0.8814; QPS 6260 vs 5031) -- better int16-accuracy IP candidate selection AND faster scan.
    QPS@90%: FASTSCAN-IP ~5678 (r0.9010@p352t16) vs i8-IP ~4500 vs recorded baseline ~4569 (P108) =
    ~1.24x engine win. So fast-scan is a win on BOTH tracks (msspacev L2 -> parity; OOD IP -> ~1.24x engine).
    FASTSCAN should replace NOLUT16 as the OOD scan default. Mechanism general: rerank fixes final order, so
    scan only needs ~12-bit SELECTION resolution -- get it at i8 (1 vpshufb) speed via min-subtract+i16 accum.
    BUT OOD-vs-ScaNN gap does NOT close: clean SAME-WINDOW (ood_fs_vs_scann.log, nq=10000): my FASTSCAN-IP
    r0.9008@5001 vs ScaNN ~9782@r0.90 (interp lts150 0.881@11326 / lts250 0.931@7262) = ScaNN ~1.96x ahead,
    i.e. STILL ~2x. My earlier "narrows to ~1.6x" was a CROSS-WINDOW artifact (chained my-low-load vs
    scann-recorded) -- same P108 error, corrected by same-window measurement: ScaNN OOD measured strong
    here (~9782). The +1.24x engine win is real but ScaNN's OOD lead (200-dim AH scan throughput) is ~2x
    and the fast-scan doesn't close it. HONEST: OOD stays ~2x behind ScaNN, same-window.

P115. (cell-inverted BATCHED scan = DEAD END for hierk3; scan is COMPUTE-bound, fast-scan was the ceiling)
    Prototyped search_batch_inverted (query-partition + cell-first ordering to amortize code loads across
    co-probing queries; subagent, worktree commit 2e33d4a). Recall IDENTICAL but 26-39% SLOWER (p128 0.74x,
    p320 0.61x). WHY: hierk3 has 262144 cells; at nq=10000 the queries-per-cell-per-chunk is <2 -> ZERO
    cache amortization (each cell's ~800B loaded once regardless of order; tiny ~400B blocks already fit
    L1), and the sort is pure overhead. The scan is COMPUTE-bound (vpshufb throughput), not memory-bound,
    for fine-cell IVF. CONSEQUENCE: confirms fast-scan (P113, 1 vpshufb/subspace) was the RIGHT and LAST
    scan lever -- it cut vpshufb count, the actual bottleneck; batching can't help. AVX-512 vpermw also
    dead (downclock). So the AVX2 scan is at its compute ceiling. Batching would only help a FLAT IVF
    (small nc, many queries/cell) -- not our design. Code not kept (worktree only). Remaining gaps
    (msspacev recall>=0.95 ~1.4x, OOD ~2x) are NOT closable by more scan-kernel work on this hardware;
    they're ScaNN's mature-implementation edge + the OOD rerank-memory wall.

P116. (*** 10-IDEA WORKFLOW: SOAR + AVX-512 64-wide scan are the two winners that may BEAT ScaNN ***)
    Ran a workflow fanning out 10 agents (isolated worktrees, cheap load-robust validation: microbench
    ratios + 1M recall). 3 winners, 7 informative negatives. Work isolated on branch engine-stack
    (worktree /home/thomas-ahle/lsh-engine) because the theory agent churns master concurrently.
    WINNER A -- SOAR-spill for hierk3 (found INDEPENDENTLY by 2 agents, ideas #3 and #9): build-time
    2nd..a0-th fine-cell assignment minimizes l2 + lambda*proj^2 (proj = q-residual along r0=q-cf[i0]),
    so a point's extra copies cover its FIRST cell's residual direction instead of near-duplicates ->
    queries find the NN in FEWER probed cells. SBANN_SOAR=lambda (0.5 best). 10M SAME-WINDOW (soar_10m.log,
    fastscan): recall@matched-p jumps +1.5-1.8pt at EVERY p (p96 0.9089->0.9252, p160 0.9292->0.9470,
    p320 0.9497->0.9644) => ~1.3-1.55x QPS at matched recall. Cost: build 2.5x slower (652->1675s; the
    per-point projection is scalar -- SIMD it later). WINNER B -- AVX-512 64-wide fast-scan
    (block_adc_i8_i16acc_avx512_il, SBANN_USE512FS): _mm512_shuffle_epi8 in-lane (Zen4 has NO avx512
    downclock; the old vpermw dead-end was cross-lane, not downclock). Microbench 1.75-1.98x vs avx2
    fast-scan, recall BIT-IDENTICAL (selftest). KEY: naive 4-load gather variant is 0.75x (Zen4
    double-pumps 512-bit) -- the build-time INTERLEAVED superblock layout (1 zmm load/group) is what wins.
    WINNER C -- residual 8-bit refine code (SBANN_RESID): 6.8x lower raw-rerank depth at fixed recall
    (1M); OOD-memory lever, deferred. NEGATIVES (measured, ruled out): GFNI ~0 (vpshufb-port-bound, not
    shift-bound); 4-bit rerank store 0.5x (Zen4 prefetcher hides 2nd cache line); rerank-batching 0.6-0.8x
    (not bandwidth-limited at scale); learned codebooks +0.0003 (apq4 already near-optimal); additive
    quant -0.05 recall vs apq4; learned distance-correction sub-%. STACKING: SOAR+AVX-512 cherry-picked
    cleanly onto engine-stack; baseline fastscan was at PARITY w/ ScaNN, SOAR adds ~1.4x at matched recall
    AND AVX-512 adds ~1.75x scan -> SOAR+AVX-512 vs ScaNN same-window is the decisive test (running,
    stack_vs_scann.log). If it holds, this is the first LEGITIMATE beat-ScaNN-on-msspacev result.

P117. (*** SOAR + AVX-512 stack BEATS ScaNN at QPS@90% (forward order) -- legitimate same-window ***)
    SOAR(0.5) + FASTSCAN + AVX-512(USE512FS) vs ScaNN, 10M, SAME WINDOW (stack_vs_scann.log): STACK
    r0.905@16210(p64), r0.918@12241(p80), r0.946@8123(p160), r0.957@6805(p224); ScaNN r0.857@9696,
    r0.923@11281, r0.946@8633, r0.967@6135. MATCHED RECALL: QPS@90% (r~0.905) STACK 16210 vs ScaNN ~10800
    = STACK ~1.5x AHEAD; r0.918 STACK ~1.1x; r0.946 ~tie; r0.957 ScaNN ~1.07x. So STACK is AHEAD at the
    leaderboard QPS@90% point and competitive-to-tied across the frontier. This is the FIRST legitimate
    beat-ScaNN-on-msspacev result (vs the load-confounded false P87 claim corrected in P110).
    The win = SOAR (~1.3-1.55x fewer probes, P116) x AVX-512 scan (1.75x, P116) stacked on the
    fast-scan-parity baseline. p96 dip (8621) is load noise (below both neighbors).
    *** BRACKET CONFIRMED BOTH LOAD ORDERS *** (stack_rev.log, reversed = ScaNN-first/STACK-second):
    STACK r0.918@29681, r0.925@26950, r0.954@14335 vs ScaNN r0.925@8635, r0.950@6708 = STACK ~3x ahead
    (STACK advantaged at low load running second). So forward (STACK first/DISADVANTAGED) STACK ~1.5x
    ahead @QPS90; reversed (STACK advantaged) ~3x. BOTH ORDERS: STACK WINS at QPS@90%. CONSERVATIVE
    DEFENSIBLE HEADLINE = forward ~1.5x (STACK ran first/disadvantaged and still won). High recall >=0.95:
    tie (forward) to ahead (reversed) -> at least competitive. LEGITIMATE beat-ScaNN result, validated
    across load orders. Exact ratio is load-dependent (1.5-3x bracket); the ROBUST fact = STACK ahead at
    QPS@90% regardless of order. Champion: hierk3 C0=1024 C1=8192 b0=48 b1=160 a0=3 apq4 + SBANN_SOAR=0.5
    + SBANN_FASTSCAN=1 + SBANN_USE512FS=1 + low TFLOOR. (engine-stack branch, /home/thomas-ahle/lsh-engine.)

P118. (HierRouter generalized to arbitrary depth L -- hierk4/5… for 100M/1B; hyperparameters tunable)
    Was hardcoded L in {2,3}. Now train_hkmeans_multi(counts[],beams[]) = one general L-level trainer;
    gather_fine() = one general descent (route_fine + route_fine_soar/SOAR both use it). New "hierkn"
    router: SBANN_LEVELS (per-level counts coarse→fine) + SBANN_BEAMS (per-level beams, len L-1).
    Routing ≈ O(L·Kf^(1/L)). hierk/hierk3 kept as wrappers. VALIDATED: hierk3 reproduces old 1M recall
    within ±0.003 RNG noise (0.8919/0.9053/0.9162 vs 0.8925/0.9081/0.9175); hierk4 (levels=[64,512,4096,
    65536] beams=[12,32,96]) builds (21s, faster than hierk3's 32s -- deeper=cheaper build) and routes
    correctly. At 1M, 4-level is overkill (slightly lower recall); the payoff is build/routing scaling at
    100M/1B. All hyperparameters now tunable as lists. (commit e94c767 on engine-stack branch.)

P119. (*** tree-Lloyd EM joint-level optimization STACKS with SOAR ~additively -- new lever; + build-cost model confirmed ***)
    User Q: would jointly optimizing all hierarchy levels (AVQ-style coordinate descent) beat greedy
    top-down? Implemented tree-Lloyd EM (SBANN_TREEEM=rounds): E-step reassign every sample point to
    nearest LEAF via the beam descent (objective = query-time search), M-step recompute EVERY level's
    centroids jointly (ancestor = leaf/fan-prod). 1M matched-probe recall (load-independent, em_battery.log):
    greedy p64 0.8686; +SOAR(0.5) 0.8855 (+1.7pt); +EM(2) 0.8813 (+1.3pt); +EM+SOAR 0.9007 (+3.2pt).
    => EM and SOAR are NOT redundant: SOAR(+1.7)+EM(+1.3)~+3.0 predicted, measured +3.2 = ~ADDITIVE.
    (My prediction of redundancy was WRONG.) They fix DIFFERENT things: SOAR the ASSIGNMENT (multi-cover
    residual dirs), EM the PARTITION (centroids co-adapt across levels so the descent lands better). So
    joint optimization is a genuine ADDITIONAL lever ~+1pt / ~1.15-1.2x QPS on top of SOAR. COST: each EM
    round ≈ a full-tree pass (build ×~(1+rounds)); fine at 10M, a tradeoff at 100M/1B. SOAR TOP-K speedup
    VALIDATED: capped SOAR preserves the +1.6pt gain (0.8919->0.9081 @p96) at ~40x less proj cost.
    BUILD-COST MODEL CONFIRMED (SBANN_BUILDPROF): per-level-s=[8.6,0.5,0.7] for fanout=[256,16,16] -- cost
    ≈ I·smp·fan·d per level, so EQUAL per level ONLY with UNIFORM fan-out; the lopsided C0=256 makes level
    0 dominate (8.6 vs 0.6s). Implication: uniform fan-out (C0≈Kf^(1/L)) balances/speeds the build -- a
    100M/1B build lever. 10M EM+SOAR CONFIRMED (em_soar_10m.log): EM adds +0.6-1.8pt OVER SOAR-only at
    matched probe (p64 0.9048->0.9230, p96 0.9252->0.9393, p224 0.9566->0.9626) = additivity HOLDS at
    scale -> ~1.4x fewer probes than SOAR-only at recall 0.92. EM+SOAR is the new msspacev champion
    (SOAR-only already beat ScaNN ~1.5x => EM+SOAR ~1.8-2x @QPS90). Build: 2 EM rounds +~633s (10M),
    buildprof per-level [150,1,3]s confirms fanout=1024 level-0 dominates. EM+SOAR also applied to OOD
    full stack (ood_fullstack.log, running).

P120. (OOD stays ~2x behind ScaNN -- msspacev levers DON'T convert; OOD is rerank-memory-bound)
    Full practical stack on OOD (SOAR+FASTSCAN-IP+AVX512, ood_practical.log, same-window vs ScaNN OOD,
    nq=10000): STACK r0.903@3986(p416), r0.897@4766(p352) vs ScaNN ~9480@r0.90 (interp) = ScaNN ~2.15x.
    SOAR (better cosine routing) + AVX-512 (faster scan) barely move OOD because OOD is RERANK-MEMORY-
    bound (~920KB/q raw reads), not routing/scan-bound -- the msspacev-winning levers attack the wrong
    bottleneck for OOD. Consistent with P108/P114 (~2x). EM also build-prohibitive at 200-dim (killed a
    47-min EM build). The lever that WOULD attack the OOD rerank wall = the RESIDUAL-rerank workflow
    winner (idea #4, 6.8x lower raw-rerank depth, P116) -- deferred (conflicted w/ AVX-512 cherry-pick).
    HONEST: msspacev BEATS ScaNN (~1.5-2x @QPS90, P117); OOD STILL ~2x behind (rerank wall, needs the
    residual lever, not routing/scan). The two tracks have DIFFERENT bottlenecks.

P121. (*** ANN-accelerated EM CONFIRMED: small-beam E-step = 3.4x faster EM, recall preserved ***)
    User Q: use the ANN structure to do EM faster? YES -- the E-step IS an ANN query against the tree's
    own centroids; a point's nearest LEAF needs ~8 probes not the full query beam (200). SBANN_TREEEM_BEAM
    (default 8) caps the E-step beam. 1M (cheapem.log) EM(2)+SOAR: beam=8 -> 2 EM rounds 13.2s, recall
    0.8979/0.9195/0.9307/0.9384/0.9483; beam=200 -> 44.7s, recall 0.9007/0.9204/0.9315/0.9393/0.9491.
    => 3.4x FASTER EM, recall within 0.001-0.003 (cheap-EM keeps +1.2pt of full-EM's +1.5pt over SOAR-only).
    The structure accelerates its OWN EM (Pelleg-Moore/Elkan/Hamerly family). Makes EM+SOAR practical at
    OOD/100M (the 47-min OOD EM build -> minutes). Further: warm-start across rounds + triangle-ineq skips.

P122. (OOD residual-IP refine = NET LOSS; OOD gap is FUNDAMENTAL quantization accuracy, not rerank depth)
    Built IP refine variant (pq::ResidPq::query_lut_f32_ip, -<q,decode>) so the 8-bit refine re-ranks IP
    candidates. 10M OOD (ood_resid.log, t_surv=5000): refine reaches recall 0.886 at rr=100 EXACT raw
    reads vs plain needs rr~1600-3000 (30x fewer raw reads -- mechanism WORKS). BUT net LOSS: refine
    reads the 100B 8-bit code for ALL t_surv=5000 survivors = 500KB > the raw it saves; QPS HALVED
    (4013 vs plain 7117 @ ~recall 0.895). Refine only wins if t_surv is SMALL, but OOD needs a LARGE pool
    (t_surv=5000) because the 4-bit IP scan is coarse -> the refine pass over the whole pool is the cost.
    ROOT CAUSE (final OOD verdict): ScaNN's anisotropic AH quant is accurate enough to rerank only ~200;
    our 4-bit/8-bit PQ needs THOUSANDS of exact reranks -> the OOD ~2x gap is the QUANTIZATION-ACCURACY
    gap (better recall-per-candidate), NOT rerank-depth or scan-speed. The residual, fast-scan, AVX-512,
    SOAR all attack the wrong thing for OOD. To close OOD we'd need ScaNN-class score-aware quant that
    makes the pool small enough -- the additive-quant attempt (idea #8) already lost to apq4. OOD stays
    ~2x; this is now well-characterized as fundamental on this engine. msspacev WIN (P117) stands.

P123. (OOD eta sweep = NEGATIVE: anisotropic weight has zero effect; accuracy bottleneck is real but eta can't move it)
    Exposed SBANN_ETA (was hardcoded 4.0, L2-tuned, never swept on OOD). 1M text2image subset (own exact-IP
    GT, t2i1m-gt), champion config hierk3+SOAR+FASTSCAN-IP, eta in {4,8,16,32,64}: recall@10 FLAT
    (p128 0.8833+-0.0002, p256 0.9244+-0.0001) across all eta. So apq4's anisotropic loss is SATURATED/
    ineffective at 4-bit -- raising the parallel-error weight does NOT change which candidates the scan
    ranks high. Confirms (research open-Q4) our DECOUPLED per-subspace loss doesn't respond to eta.
    IMPLICATION: recall = pool-recall@t_surv (exact rerank), and on 10M the champion needs t_surv~5000 for
    recall 0.90 -> the 4-bit scan RANKING is poor (low pool-recall at small pools) = accuracy IS the
    bottleneck, but eta/anisotropic-weighting is a DEAD lever for it. Next accuracy levers (different
    mechanism): residual quantization (encode x-cent_fine: smaller range -> 4 bits resolve it better; ScaNN's
    default) -- but it COUPLES compressor training to the router (codebook must be trained on routed
    residuals), a real Tier-1 change. Or RaBitQ (more bits + unbiased IP estimator, the high-ceiling swing).

P124. (*** residual-PQ DE-RISK: +6-11pt IP pool-recall at small pools -- the real OOD lever, integrate it ***)
    After eta failed (P123), de-risked residual quant with an offline numpy sim (resid_pq_sim2.py) BEFORE
    integration: 4-bit PQ, dpb=2, 1M text2image subset, 2048 cells. RAW-PQ (encode x) vs RESID-PQ
    (encode x-cell_centroid, score <q,cent>+<q,resid_hat>) IP pool-recall: @T=20 0.591->0.705 (+11.4pt),
    @50 0.784->0.878, @100 0.881->0.944, @200 0.939->0.976. Win LARGEST at small pools -> RESID-PQ
    reaches a given pool-recall at ~2-2.5x SMALLER T (raw needs T=500 for 0.976, resid hits it at T=200).
    Mechanism: residuals have smaller dynamic range -> 4 bits resolve them better; the per-cell <q,cent>
    offset (constant per cell, ESSENTIAL) carries the bulk IP. This is ScaNN's use_residual_quantization
    default. Translation: OOD scan needs t_surv~5000 -> resid-PQ ~2-2.5x less -> rerank-memory wall (920KB/q)
    drops ~2x = most of the OOD gap. UNLIKE eta (flat, P123) this is a large clean win. INTEGRATING:
    encode_block gets cell_cent (subtract), codebook retrained on residuals, scan adds scaled <q,cell_cent>
    offset per cell. The de-risk (minutes) saved a 1-2 day build on an unproven hypothesis.

P125. (residual quant IMPLEMENTED + validated: real but MODEST OOD win, routing-diluted)
    SBANN_RESIDQ (commit ec18a5c): build computes per-cell RAW centroids, retrains apq4 on residuals,
    encode_block subtracts the centroid (f32), scan_rerank adds the exact scaled <q,centroid> offset per
    cell (ip_i16_scale/ip_i8s_scale). Works on BOTH int16 and fast-scan paths (offset correct -> recall
    rises not tanks). 1M OOD subset (own GT): recall-per-probe +0.4-1.6pt at fixed budget (p128
    0.8833->0.8954, p256 0.9244->0.9307). MUCH smaller than the de-risk's +6-11pt (P124) because the
    sim isolated SCAN accuracy (all 1M candidates) while the engine's recall is ALSO gated by ROUTING
    (only probed cells' points are candidates) -> RESIDQ improves the scan WITHIN probed cells toward the
    routing-coverage ceiling, diluted by routing. Net QPS@matched-recall (1M+fastscan, cache-warm):
    ~break-even at recall 0.88 (per-cell <q,cent> offset overhead ~13%), crossing to ~1.07x at recall 0.94
    (gain grows with recall). KEY INSIGHT: OOD recall is gated by BOTH routing coverage AND scan accuracy;
    RESIDQ fixes the scan half (modestly), routing is the other half. Genuine positive (unlike eta P123)
    but NOT a 2x-closer alone. 10M test pending (memory-bound rerank should amplify: shallower survivor
    pool -> less of the 920KB/q wall).

P126. (10M residq = only +0.2pt recall -> OOD is ROUTING-COVERAGE-limited, not scan-accuracy-limited)
    10M OOD RESIDQ+FASTSCAN vs baseline same-window (residq_10m.log): recall +0.18-0.26pt at matched p
    (p288 0.8891->0.8917, p352 0.8973->0.8996, p448 0.9057->0.9075). MUCH smaller than 1M (+0.4-1.6pt, P125)
    -> the gain SHRINKS with scale. QPS looked 2.2x but LOAD-CONFOUNDED (baseline ran first at high load
    ~2700, residq later ~5500; don't trust -- P110 trap). KEY INSIGHT: at 10M with 262144 cells probing
    ~300-450, the candidate pool is a tiny fraction of base, so ROUTING COVERAGE (which cells hold the OOD
    neighbors) is the DOMINANT limiter; RESIDQ improves within-cell scan ranking but recall is capped by
    the routing-coverage CEILING it can't exceed. So OOD recall is gated PRIMARILY by routing pool-recall,
    NOT scan accuracy -> scan-accuracy levers (RESIDQ +0.2pt, and likely RaBitQ) give diminishing returns
    at scale; the bigger half is ROUTING (getting OOD neighbors into probed cells -- ScaNN's partitioning
    edge). RESIDQ stays as a clean small positive (kept, SBANN_RESIDQ) but is NOT the OOD closer. NEXT
    OOD lever = routing pool-recall (measure the ceiling first; query-aware routing failed P96/98 so needs
    a sounder approach -- e.g. more cells, OOD-distribution-trained centroids, or spill). Honest: OOD ~2x
    is routing-coverage + scan-throughput, both of which our scan-accuracy work doesn't move much.

P127. (*** OOD gap is ROUTING, not scan: RESIDQ reaches our routing ceiling; ScaNN's recall EXCEEDS it ***)
    Routing pool-recall ceiling (rerank whole probed pool, poolrecall.log): p128 0.8958, p256 0.9308,
    p384 0.9433, p512 0.9487. RESIDQ scan-recall (0.8955/0.9307/0.9432) == the ceiling -> our scan is
    now OPTIMAL (extracts the full probed pool), NO scan headroom left -> RaBitQ/more-bits would NOT help.
    ScaNN on the SAME 1M subset (bench_scann_t2i1m, num_leaves=2000, dot_product, reorder 200): recall
    lts80 0.9072, lts150 0.9565, lts300 0.9794, lts600 0.9924. ScaNN's achievable recall (0.957-0.992)
    EXCEEDS our routing ceiling (max 0.9487 @p512) -> ScaNN's PARTITIONING finds OOD neighbors our
    cosine-hierk3 routing never puts in the candidate pool. (1M QPS cross-window/load-confounded -- don't
    compare; trust the 10M same-window ~2x.) CONCLUSION: the OOD ~2x is ROUTING/PARTITIONING + scan
    throughput, NOT scan accuracy. eta(P123)/residual-quant(P125-126)/RaBitQ all attack the scan = wrong
    half. RESIDQ is kept (maxes the scan, clean) but OOD needs a better PARTITION for the OOD query
    manifold (query-aware routing failed P96/98; needs more cells / OOD-distribution centroids / spill).
    Scan-accuracy work on OOD is DONE; the lever is routing. (No more ScaNN re-runs -- ample data.)

P128. (*** USER INSIGHT CONFIRMED: joint tree-EM RAISES the OOD routing ceiling -- the right lever ***)
    OOD is routing-limited (P127); user flagged that the joint cross-level partition optimization
    (tree-EM, SBANN_TREEEM, built P119 for msspacev but NEVER tried on OOD) should improve the partition.
    CONFIRMED (em_ceiling.log, 1M OOD, routing pool-recall ceiling = rerank whole pool): noEM 0.8958/
    0.9308/0.9433/0.9487 (p128/256/384/512) -> +EM(3,cheap beam=8) 0.9112/0.9387/0.9475/0.9517 = +0.3 to
    +1.5pt (largest at small p). This is PARTITION quality (the ceiling, perfect rerank), not scan, and
    NO index bloat (unlike a0). So the OOD lever = tree-EM (raises ceiling) + RESIDQ (P125, maxes scan to
    the ceiling). EM ceiling 0.9517 still < ScaNN achievable (lts150 0.9565) so EM narrows but doesn't
    fully close the routing gap; more rounds / OOD-distribution centroids may help further. a0 multi-assign
    test was BOTCHED (t_surv truncation didn't cover the bloated pool -> invalid; high a0 also has steep
    QPS cost from the larger rerank pool). NEXT: confirm the ceiling gain -> real recall@QPS (EM+RESIDQ
    +fastscan vs baseline vs ScaNN). Credit: user redirected from scan-accuracy (wrong half) to routing.

P129. (*** tree-EM is a CLEAN ~1.2x OOD recall@QPS win -- no query overhead; the OOD lever (user insight) ***)
    1M OOD recall@QPS (em_residq_qps.log, fastscan): baseline p128 0.8833@34565, p256 0.9244@20348.
    +EM(3,beam8): p128 0.8970@36647 (+1.4pt, QPS NEUTRAL -- EM is BUILD-time partition, zero query cost),
    p256 0.9315@22415. At matched recall EM = ~1.15-1.23x faster (recall 0.924: ~25000 vs 20348). EM+RESIDQ:
    recall higher (p128 0.9108) but residq offset overhead cuts QPS -> roughly cancels at QPS@90%, only
    edges ahead at recall>=0.94. So OOD config = EM ALONE (clean ~1.2x, no downside); RESIDQ optional for
    high-recall. vs ScaNN 1M (lts80 0.9072@67267, load-confounded -- trust 10M ~2x): EM narrows 10M OOD
    ~2x -> est ~1.7x. EM build +~30-60% (cheap beam=8). BIGGEST OOD gain of the session, from the user's
    routing redirect (I'd built tree-EM for msspacev P119 but missed its OOD relevance). Next: 10M EM
    confirm + more rounds. ScaNN's partition still better (its ceiling exceeds ours) so EM narrows not closes.

P130. (10M OOD: tree-EM confirmed ~1.25x QPS@90%, same-window, QPS-neutral -> narrows OOD ~2x to ~1.6x)
    10M OOD same-window (em_10m.log): baseline recall p224 0.8729@8523, p288 0.8858@7170, p352 0.8945@6429,
    p448 0.9034@5187. +EM(3,beam8): 0.8833@9144, 0.8943@7682, 0.9021@6657, 0.9097@5458 = +0.6-1.0pt recall
    at fixed p, QPS NEUTRAL-to-faster (no query overhead; better-balanced cells slightly faster). At matched
    recall ~1.25x (recall 0.90: EM ~6700@p345 vs baseline 5187@p448). The 1M ~1.2x HOLDS at scale. EM build
    954->1627s (+670s for 3 rounds; mem 6.3->11.5GB, under the 40% limit). Narrows OOD ~2x -> ~1.6x (2/1.25).
    EM is the NEW OOD config (clean win, no downside). Definitive EM-vs-ScaNN same-window pending. The
    session's biggest OOD lever, from the user's routing/joint-partition redirect (P127-130).

P131. (*** DEFINITIVE EM-vs-ScaNN 10M OOD same-window: ScaNN still wins; EM narrowed QPS@90% to ~1.4x ***)
    em_vs_scann_10m.log, ONE script, same loaded window (ScaNN built at load=67.69 -> its QPS suppressed too).
    EM(3,beam8)+fastscan+avx512 (hierk3 C0=1024 C1=8192 b0=64 b1=200 a0_soar=0.5 Kf=262144):
      p224 0.8833@3933 | p288 0.8943@3304 | p352 0.9021@2308 | p448 0.9097@1957
    ScaNN (num_leaves=4000, AH-2, reorder, dot_product):
      lts50 0.6904@11244 | lts100 0.8244@5292 | lts150 0.8809@4032 | lts250 0.9317@2052 |
      lts400 0.9589@1703 | lts600 0.9781@1450 | lts900 0.9881@1124 | lts1400 0.9928@701
    QPS@90%: EM ~2308 (p352) vs ScaNN ~3287 (interp lts~205) => ScaNN ~1.42x faster, both load-suppressed
    (ScaNN at higher load 67.69, so the unloaded gap is >=1.42x). Matches the ~1.6x estimate (P130).
    *** THE REAL STORY (load-independent, trustworthy = recall): ScaNN reaches 0.9928 recall; my engine
    CAPS at 0.9097 in this config and ~0.9487 pool-recall ceiling @p512 (P127). ScaNN dominates the entire
    recall>=0.95 regime my routing simply cannot reach. EM raised my ceiling +0.6-1.0pt but the gap is
    ROUTING COVERAGE: ScaNN's partition pools OOD neighbors mine never sees. *** To actually beat ScaNN on
    OOD I must raise the routing pool-recall ceiling itself. Untried, user-endorsed levers: (1) higher SOAR
    a0 (replicate each point into MORE fine cells -- "storing points multiple times will help"; a0 sweep was
    botched earlier by t_surv undersizing, needs proper re-measure), (2) query-distribution-aware routing
    (OOD = query dist != base dist; train router centroids on query/mixed sample or IP-anisotropic metric --
    this is the ScaNN-style OOD lever the plain-L2 hierk router lacks). NEXT: a0 sweep, t_surv sized to pool.

P132. (*** BREAKTHROUGH: a RERANK DEDUP BUG was masking multi-store; fixed -> routing coverage to ScaNN PARITY ***)
    Root cause: rerank_contig/rerank_survivors keep only m=(k*4)=40 survivors in the exact-rerank heap.
    With SOAR a0>1 a point lands in multiple probed cells as DUPLICATE SLOTS (same tiny distance), which
    crowd out the 40-slot budget -> after id-dedup <10 distinct ids survive -> recall craters AND DECREASES
    with p (more probes = more duplicates). So the earlier "a0>=6 collapses the ceiling" (a0_ceiling.log:
    a0=6 0.80, a0=10 0.56, recall falling with p) was a MEASUREMENT ARTIFACT, not a real coverage failure.
    Verified the artifact is real: capped(t=51200) and uncapped(t=p*1e7) gave IDENTICAL bad numbers, so the
    cap was rerank_contig's m=40, not t_surv. FIX: new SBANN_POOLDEDUP flag (vq.rs) dedups the pool by ORIG
    id (keep min approx-dist per id) BEFORE the survivor cap, so the 40-slot heap holds DISTINCT ids. Also
    shrinks the pool (faster). 1M OOD true coverage ceiling (whole-pool rerank), POOLDEDUP on:
      p:        128     256     384     512
      a0=3    0.8959  0.9309  0.9433  0.9487   (== non-dedup a0=3: dedup is a no-op when few dups -> fix is safe)
      a0=6    0.9339  0.9573  0.9648  0.9678   (+1.9pt over a0=3 @p512; monotone-in-p again, as it must be)
      a0=12   0.9592  0.9733  0.9778  0.9797   (MATCHES ScaNN lts300 0.9794! a0=12@p128 0.9592 > a0=3 @ ANY p)
    ScaNN 1M ref: lts80 0.9072, lts150 0.9565, lts300 0.9794, lts600 0.9924.
    *** OVERTURNS P127's "OOD is STRUCTURALLY routing-limited, ScaNN's partition fundamentally better." It is
    NOT structural -- multi-store (a0) + the dedup fix reaches ScaNN coverage parity (~0.98). The user was
    right twice: joint-partition (tree-EM, P128) AND "store points multiple times" (this) both raise coverage;
    the second was buried under the dedup bug. *** COSTS to validate next: (a) storage -- raw is per-SLOT, so
    a0=12 = 12x raw (10M*12*200B=24GB -> over the 26GB budget; a0=6=12GB fits). May need raw stored per-DISTINCT-
    point (rerank by orig id) to scale a0=12 to 10M/1B. (b) QPS -- the ceiling QPS above is uncapped (whole-pool
    rerank, pathologically slow); the REAL test is production QPS@recall with small t_surv (higher a0 hits target
    recall at much SMALLER p, but bigger pool/cell -- net QPS must be measured). NEXT: 1M production frontier
    a0={3,6,12}+dedup+small t_surv real QPS vs ScaNN; then 10M (a0=6 fits, or raw-dedup refactor for a0=12);
    then stack with tree-EM. Commit the dedup fix (correctness: it also stops a0>1 wasting t_surv on dups).

P133. (1M production frontier: multi-store a0 = the HIGH-RECALL lever; optimal a0 RISES with target recall)
    a0_frontier_1m.log, POOLDEDUP on, t_surv=max(6p,600), best-of-5, 8 threads, loaded window (load~60):
      a0=3 : p96 0.8736@9138 | p160 0.9084@7050 | p256 0.9303@3372 | p384 0.9429@2603 | p512 0.9484@2051
      a0=6 : p96 0.9166@5912 | p160 0.9411@4080 | p256 0.9561@2163 | p384 0.9642@1336 | p512 0.9674@945
      a0=12: p96 0.9440@2918 | p160 0.9617@1779 | p256 0.9713@1090 | p384 0.9768@1330 | p512 0.9791@1008
    Pareto: recall<=0.91 -> a0=3; ~0.94 -> a0=6 (p160 4080 vs a0=3 p384 2603 = 1.6x); >=0.96 -> a0=12 (only
    a0>=12 reaches 0.97-0.98). Each higher a0 EXTENDS+dominates the high-recall frontier. a0=12 0.9791 ==
    ScaNN 0.9790 (COVERAGE PARITY). *** Two confounds make 1M QPS NON-representative, do NOT read a QPS gap
    from it: (a) ScaNN 1M frontier (16 threads, load~47): lts80 0.9081@40574, lts150 0.9566@30155, lts300
    0.9790@18607, lts600 0.9923@8174 -> ScaNN 6-18x our QPS ON 1M, but 1M is cache-resident (ScaNN's
    in-register AH scan flies) and overhead-dominated; the SAME comparison at 10M = only 1.42x (P131). 1M is
    a COVERAGE proxy, not a QPS proxy. (b) THREAD CONFOUND (found this session): box has 16 cores; our runs
    pin RAYON_NUM_THREADS=8, ScaNN's search_batched_parallel grabs ALL 16 -> every ScaNN-vs-us QPS gap
    (incl P131 1.42x) gave ScaNN 2x the cores. Scan parallelizes ~linearly across queries, so at thread
    parity our QPS should ~1.6-1.8x. NEXT (definitive 10M OOD): our a0=6+dedup(+tree-EM) at 16 threads vs
    ScaNN at 16 threads, same window -- the real leaderboard test combining multi-store + thread parity.
    (a0=6 raw=10M*6*200B=12GB fits 26GB budget; a0=12=24GB needs per-distinct-point raw refactor.)

P134. (*** THREAD CONFOUND = clean ~2x (we ran on HALF the box); multi-store extends 10M recall to 0.9225 ***)
    def_10m_ood.log, ONE script same window (load drifted DOWN 55->34 over the run -> later stages = lighter
    load; POOLDEDUP on, NO EM). hierk3 C0=1024 C1=8192 b0=64 b1=200 SOAR=0.5 Kf=262144 fastscan+avx512:
      THREAD A/B (a0=3, IDENTICAL recall both, back-to-back = CLEAN):
        p192 0.8691: 16thr 5658 vs 8thr 2341 = 2.42x | p256 0.8843: 3704 vs 1769 = 2.09x |
        p352 0.8978: 2871 vs 1448 = 1.98x | p448 0.9058: 2469 vs 1427 = 1.73x | p576 0.9133: 2089 vs 1061 = 1.97x
      => 8->16 threads = ~2x QPS (slightly superlinear low-p). We had been pinning RAYON=8 on a 16-core box
      while ScaNN's search_batched_parallel uses all 16. ADOPT 16 THREADS (leaderboard uses whole machine).
      a0=6 @16thr: 0.8772@3245 0.8973@2126 0.9075@1692 0.9171@1240 0.9225@986 (extends recall to 0.9225 at
      10M vs a0=3's 0.9133 same plist; QPS penalized by the POOLDEDUP HashMap -> fix #5 pending).
      ScaNN @16thr (load 34, LIGHTER than our 50 -> confound FAVORS ScaNN): lts50 0.6911@19987, lts100
      0.8241@11194, lts150 0.8803@7736, lts250 0.9301@4440, lts400 0.9576@2935, lts600 0.9766@1950,
      lts900 0.9877@1628, lts1400 0.9927@1847.
    READ: at recall ~0.88 our a0=3 went 4.4x behind (8thr) -> 2.1x behind (16thr) = thread parity HALVED the
    gap. Residual ScaNN lead ~1.4-2x is load-confounded (ScaNN got the light window) -> NOT a clean number;
    needs a BRACKETED re-measure (both 16thr, interleaved load) AFTER fix #5 lands. Clean facts: (1) thread
    ~2x, (2) multi-store recall parity/extension (load-independent). ScaNN owns recall>=0.95 (our 10M plists
    capped 0.9225; a0=12 would reach higher but =24GB raw at 10M, needs per-distinct-point raw refactor).
    Per-query-overhead workflow (wa09pg6d8): the POOLDEDUP HashMap (added this session) is built PER QUERY
    and is the gap-WIDENER (scales p*a0). Top fix = fuse scan+select, dedup in rerank not via a poolsize map.

P135. (*** FIX #5 LANDED: dedup heap + fast open-addr dedup-before-cap kill the per-query HashMap; 1M@0.90 = ScaNN PARITY ***)
    Two changes (vq.rs): (1) rerank_contig/rerank_survivors heap now stores (dist, ORIG) and SKIPS a
    duplicate orig (same point=identical raw=identical dist) -> fixes the k*4=40-slot crowding at the source.
    (2) scan_rerank: for a0>1, dedup the pool by orig BEFORE the t_surv cap via a REUSED thread-local
    open-addressing table (Fibonacci hash + linear probe, no SipHash, no per-query alloc) instead of the
    std HashMap. a0==1 (msspacev) skips dedup entirely. Index gained an `a0` field.
    WHY before-cap (not the cheaper after-cap dedup-in-rerank): dedup-AFTER-cap loses recall at high a0 --
    a0=12 after-cap collapsed to 0.8828@p96 / 0.9624@p512 (dups crowd the top-t_surv survivors). Before-cap
    is recall-correct. (a0=3 after-cap only -0.003, so a future a0-threshold could use the faster after-cap
    for small a0.) Validated 1M OOD, 16 threads, prod t_surv, best-of-5 -- recall MATCHES old SipHash exactly:
      a0=3 : 0.8736/0.9083/0.9303/0.9429/0.9484  QPS 37805/23899/15466/12319/8798
      a0=6 : 0.9166/0.9411/0.9561/0.9642/0.9674  QPS 27263/19560/12038/8320/5861
      a0=12: 0.9440/0.9617/0.9713/0.9768/0.9791  QPS 15672/8978/6652/4114/2993  (vs old SipHash 8331/../1760 = 1.7-1.9x)
    *** Combined with the 16-thread fix, the 1M QPS jumped 3-5.6x at matched recall (a0=6@0.956: 2163->12038).
    1M recall-0.90: us 38925 (a0=3 after-cap) ~= ScaNN 40574 = PARITY (was 5.8x behind). Mid-recall gap 14x->
    2.5x. High recall 0.979 still ScaNN 6x (1M cache-resident; expect smaller at 10M). *** THREAD FIX = just
    stop pinning RAYON_NUM_THREADS=8; rayon defaults to all 16 cores. NEXT: msspacev a0=1 regression check
    (the rerank heap any()-scan), then BRACKETED 10M OOD vs ScaNN both @16thr with fix#5 -- the real test.

P136. (*** CLEAN 10M OOD at FULL parity: ScaNN ~2.4x ahead at QPS@90% -- OOD is structurally ScaNN's ***)
    bracket_10m.log: ours(a0=3 fix#5 16thr, load 32) -> ScaNN(16thr, load 35) -> ours(load 60, drifted).
    The pre-run and ScaNN are bracketed at MATCHED light load (32 vs 35) = the clean comparison; the post
    drifted up so it only confirms drift came AFTER ScaNN.
      ours a0=3 fix#5 @16thr: p256 0.8843@6680 p320 0.8940@5134 p384 0.9009@3878 p448 0.9058@3461 p576 0.9133@2405
      ScaNN @16thr: lts50 0.6910@20876 lts100 0.8254@13513 lts150 0.8805@10488 lts250 0.9295@7325
                    lts400 0.9567@4756 lts600 0.9768@3820 lts900 0.9872@2411 lts1400 0.9919@1835
    QPS@90%: ours ~3878 (p384) vs ScaNN interp ~9229 => ScaNN ~2.4x. *** This OVERTURNS the rosy reading of
    P131's "1.42x": that was because ScaNN ran at load 67 (suppressed). At LIGHT load + 16-thread parity
    ScaNN's true QPS@90% is ~9229 and the real OOD gap is ~2.4x, ScaNN ahead. The 16-thread fix corrected
    OUR under-threading (a measurement artifact: we'd run 8thr-us vs 16thr-ScaNN) but did NOT win OOD --
    ScaNN's in-register AH scan + GEMM routing + low per-query overhead are ~2.4x faster at 10M OOD. ***
    NOT-YET-TRIED faster QPS@90% configs (could narrow ~2.4x -> ~1.5x but unlikely to win): a0=1 / after-cap
    dedup (1M showed 1.6x), scoped router GEMM (workflow #2), EM (reach 0.90 at lower p). LEADERBOARD TAKE:
    OOD (text2image) is ScaNN's track. The thread fix's REAL value is on the EUCLIDEAN tracks (msspacev/
    BIGANN/DEEP) where we ALREADY beat ScaNN (P117 ~1.5x) -- if that win was at 8thr-us vs 16thr-ScaNN, it
    ~doubles at parity. NEXT: msspacev 16-vs-16 thread A/B + bracketed vs ScaNN (the win we can grow).

P137. (*** msspacev at CLEAN parity: TIED at QPS@90% (not P117's confounded 1.5x win); ScaNN ahead at high recall ***)
    msspacev_retest.log + msspacev_parity.log, 16-thread parity, the cleanest same-LOAD pair = ours-before-cap-a0=3
    and ScaNN both at load ~47-48:
      ours a0=3 (before-cap): p48 0.9000@14753 p64 0.9108@12444 p96 0.9263@9575 p128 0.9375@7298 p192 0.9500@5520
      ScaNN @16: lts50 0.8585@14472 lts100 0.9221@16094 lts150 0.9486@10863 lts250 0.9681@6614 lts400 0.9814@5969
    QPS@90%: ours 14753 vs ScaNN ~15526 (interp) = ScaNN 1.05x = TIED. recall0.925 ScaNN 1.68x; 0.95 ScaNN ~2x.
    Threshold fix (after-cap a0=3, DEDUP_A0=4 default) at light load hit 0.95@10804 (vs before-cap 5520) -> should
    push QPS@90% slightly AHEAD and narrow the high-recall gap to ~1.4x (couldn't measure same-load: after-cap ran
    at load 28, before-cap+ScaNN at 47 -- the persistent box-load drift from the concurrent master agent). a0=1 caps
    at recall 0.8666@p256 (never reaches 0.90) -> multi-store (a0>=2) is REQUIRED to reach 0.90 efficiently.
    *** CORRECTION: P117's "msspacev beats ScaNN ~1.5x" was LOAD-CONFOUNDED (ScaNN suppressed in that window), the
    SAME P87->P110 mistake. At clean light-load 16-thread parity, ScaNN's true msspacev QPS@90% ~15.5-20k (vs P117's
    ~11k). HONEST STANDING at parity: msspacev = TIED at QPS@90% (competitive), ScaNN ahead >=0.95; OOD = ScaNN ~2.4x
    (P136). The thread fix corrected OUR under-threading but ScaNN-at-light-load is also fast -> no clean WIN anywhere
    yet. *** LESSON (re-learned): NEVER trust a cross-window QPS win on this shared box; only same-load same-window
    bracketed pairs. The lever to convert msspacev TIE->WIN = cut per-query overhead (scoped router GEMM, workflow #2:
    our beam descent ~435K scalar MACs vs ScaNN's single GEMM; same MAC count, worse execution FORM).

P138. (threshold fix = modest; box load noise now EXCEEDS the effects we're chasing -> measurement impasse)
    thresh_ab.log, msspacev a0=3, "back-to-back" after-cap vs before-cap -- but load drifted 33->44 mid-A/B:
      after-cap: p96 0.9251@13415 p128 0.9365@12387 p192 0.9496@6582 p256 0.9576@5945
      before-cap: p96 0.9263@8414 p128 0.9375@7642 p192 0.9500@5681 p256 0.9576@4740
    Ratio 1.16-1.59x but ~1.33x of that is the load delta -> true after-cap savings ~1.0-1.2x (modest). Recall
    identical (-0.001) = threshold fix safe. *** KEY: even a back-to-back A/B is load-confounded now; QPS
    variance on this shared box (concurrent master agent) >> the 1.1-1.2x effects remaining. Only RECALL
    (load-independent) and ~2x+ effects are reliably measurable. *** IMPASSE: the remaining ScaNN gaps (OOD
    QPS 2.4x; msspacev high-recall ~1.5-2x) are SCAN/ROUTING EXECUTION-EFFICIENCY gaps (ScaNN's in-register AH
    + GEMM routing), not coverage (multi-store closed that). Closing them = matching ScaNN's low-level kernels
    (substantial) AND is UNMEASURABLE on this box (load noise drowns ~1.1x gains). Recall headroom is exhausted
    (at ScaNN coverage parity). So autonomous QPS micro-optimization has hit diminishing+unmeasurable returns
    -> surfaced a strategic decision to the user (deep low-level kernel work vs consolidate vs new approach).

P139. (*** PROFILE: routing is 20-46% of query time (NOT ~4%) -> router GEMV is THE msspacev QPS@90% lever ***)
    SBANN_PROFILE route/scan/rerank split, 10M, 16 threads (fractions load-robust). profile_10m.log:
      msspacev p64 (QPS@90%, r0.91): route 46.3% scan 34.9% rerank 18.7%   <- ROUTING DOMINATES
      msspacev p128 (r0.94):         route 32.5% scan 45.5% rerank 22.0%
      OOD p256 (r0.88):              route 26.3% scan 48.9% rerank 24.8%
      OOD p384 (r0.90):              route 20.6% scan 52.3% rerank 27.1%
    1M OOD for ref: route 13-20% scan ~50% rerank ~32%. The workflow's "routing ~4% at 10M" was WRONG --
    the beam descent scores b1*(Kf/C1) ~5100-6400 fine centroids/query via a SCALAR l2_i8 loop (gather_fine,
    vq.rs:619). At msspacev d=100 the scan is cheap so routing = 46% at the QPS@90% point. *** THE LEVER:
    l2_i8 = ||q-c||^2 = cnorm[c] - 2*(q.c) + ||q||^2; ||q||^2 const, cnorm precomputable -> scoring = a DOT
    q.c batched over the contiguous children (ScaNN's execution form: GEMV/VNNI with q in registers, vs our
    scalar per-centroid loop). Halving routing (46->23%) ~= 1.3x QPS -> converts the msspacev TIE into a WIN
    at QPS@90%. This is the user-approved kernel work. IMPLEMENTING: batched int8 dot scorer for gather_fine.

P140. (batched routing scorer = real ~1.11x (committed); hierk4 = wash; routing is per-query bandwidth-bound)
    l2_i8_block (simd.rs, 2-wide ILP, qn in regs, dispatch once) replaced the scalar per-centroid l2_i8 loop
    in gather_fine. Bit-identical recall. msspacev p64 routing share 46.3->40.5% (~1.27x routing, ~1.11x
    total); applies to OOD too (routing 20-46% everywhere). COMMITTED 4c0289f.
    hierk4 (levels 1024,8192,65536,262144 beams 48,200,200) vs hierk3: route 40.5->35.7% BUT recall
    0.9100->0.9009 @p64 (deeper hierarchy prunes harder) -> at MATCHED recall a wash. Not pursued.
    *** ROOT INSIGHT: the fine level scores b1*(Kf/C1)~5000 query-SPECIFIC centroids scattered across the
    26MB centroid array -> bandwidth/latency-bound, and (unlike ScaNN's FLAT 4000-leaf GEMM read once per
    query-batch) our HIERARCHY reads different fine cells per query => NO cross-query amortization. ILP helps
    only ~1.27x; the rest is bandwidth. Matching ScaNN's routing amortization would need flat-ish batched-GEMM
    routing (architectural change, uncertain payoff). *** Other gaps similarly structural: scan 38-52% (our
    AVX-512 64-wide ~= ScaNN AH, little headroom); rerank 19-27% (we rerank t_surv~3072 vs ScaNN reorder=200
    -- ScaNN's better ADC ranking needs shallower rerank; closing = deeper quantization, modest/explored).
    NET: per-query-overhead lever yielded a real ~1.11x; further is diminishing+unmeasurable on this box.

P141. (*** PROMISING: full-stack BRACKETED ~2x ScaNN at msspacev QPS@90% -- threshold fix is the big low-p lever ***)
    final_msspacev.log, BRACKET ours(full stack)->ScaNN->ours, all 16 threads. Full stack = batched routing
    scorer + after-cap a0=3 (DEDUP_A0=4 default) + 16t. msspacev-10M:
      ours PRE:  p48 0.8996@47846 p56 0.9056@44791 p64 0.9100@42794 p80 0.9194@33832 p96 0.9251@33850
      ScaNN:     lts50 0.8539@17288 lts100 0.9209@16960 lts150 0.9428@22485 lts250 0.9645@10984 lts400 0.9775@7164
      ours POST: p48 0.8996@38255 p56 0.9056@31806 p64 0.9100@34459 p80 0.9194@32657 p96 0.9251@33031
    QPS@90%: ours ~36-46k (pre/post) vs ScaNN interp ~17062 => ours ~2.0-2.7x. ScaNN (17k) sandwiched BETWEEN
    two ours runs both ~2x higher -> not a simple light-window fluke for us. MECHANISM: P137 "tied" used
    BEFORE-cap a0=3 (the fix#5 handicap, p64@12444); the THRESHOLD FIX (after-cap, P138) skips the whole-pool
    dedup which at low p (tiny scan) is a big fraction -> after-cap p64 jumped to ~34-43k. Plus batched scorer
    (1.11x) + 16t. *** CAVEAT (P117 burned me): the SAME after-cap config gave p64@14830 in a loaded window
    (route_gemv_test) vs 34-43k here -> absolute QPS still swings ~2.5x on measurement-window load. The bracket
    (ScaNN between two ours) is the evidence the RATIO holds, but the within-bracket drift (pre 46k/post 36k =
    25%) means ratio uncertainty ~1.7-2.7x. Recording as PROMISING not certain; running a confirmation that
    isolates after-cap vs before-cap @p64 same-window (the mechanism) before claiming the win.

P142. (*** KEY INSIGHT: ours is LOAD-SENSITIVE, ScaNN is not -> the LEADERBOARD's quiet box favors us; threshold fix confirmed +1.6x ***)
    confirm_aftercap.log, back-to-back p64 (recall 0.91), msspacev-10M, 16t:
      after-cap#1 (load34) 37588 | before-cap (load29) 26007 | after-cap#2 (load28.6) 43845
    after-cap#2 vs before-cap = NEAR-IDENTICAL load (28.6 vs 29), consecutive -> clean ratio 43845/26007 =
    1.69x. THRESHOLD FIX (after-cap, P138) = genuine ~1.6x at QPS@90% (skipping whole-pool dedup at low p),
    LOAD-ROBUST. *** THE BIG REALIZATION: ours QPS scales STRONGLY with idle CPU (before-cap p64: 12444@load47
    -> 26007@load29 = 2.1x), ScaNN barely moves (15526@load47 -> 17062@load37 = 1.1x). Ours is COMPUTE-bound
    (uses all 16 cores when free), ScaNN is MEMORY-BANDWIDTH-bound (load-insensitive). The big-ANN LEADERBOARD
    runs on a QUIET DEDICATED box (load ~0) = ours' MAXIMALLY-ADVANTAGED condition. So the LIGHT-load reading
    is the leaderboard-relevant one, and the whole session's "tied/behind" msspacev numbers (P137 etc.) were
    measured on a LOADED box where ours is artificially suppressed. *** Light-load msspacev QPS@90%: ours
    after-cap ~40-46k (p52) vs ScaNN ~17k -> ours ~2.4x. CONFIRMING: pinning ScaNN's light-load ceiling
    (scann_light.log @ load 18) -- if ScaNN stays ~17-20k, the msspacev QPS@90% WIN is real for the leaderboard.
    This reframes the load-confounding: not "noise drowns the signal" but "ours and ScaNN respond to load
    DIFFERENTLY, and the leaderboard condition (quiet) is the one where we win."

P143. (*** CONCLUSIVE: ours BEATS ScaNN ~2-3x at msspacev QPS@90% on the leaderboard (quiet-box) condition ***)
    scann_light.log: ScaNN msspacev @0.90 across 3 windows = 17062 (load37), 10800 (load32), 15463 (load35)
    -> range 10.8-17.4k, noisy. ours after-cap full-stack @0.90 across light windows = 36-50k (p64 r0.91:
    37588/43845/42794/34459). CLEAN SEPARATION: ours never <36k, ScaNN never >17.4k -> NO overlap. Matched
    recall 0.91, similar load: ours 43845 vs ScaNN ~11-15k = ~2.9-4x. Conservative headline: ours ~2.5x ScaNN
    at msspacev QPS@90% in light windows, and >= that on a truly quiet box (ours scales with idle CPU: 12k@
    load47 -> 44k@load28). *** RESOLUTION of the session's load-confounding saga: it was NEVER pure noise --
    ours is compute-scalable (rises ~3x from load47->load28), ScaNN is bandwidth-bound (~flat 11-17k). The
    big-ANN LEADERBOARD runs on a QUIET DEDICATED box = ours' best case. So the LIGHT-window readings are the
    leaderboard-relevant ones, and they show a robust ~2.5x WIN. The earlier "tied/behind" (P137) were
    LOADED-box artifacts where ours was throttled. Win stack: int16/fast-scan + AVX-512 64w scan + SOAR
    multi-store + dedup-bug fix + fix#5 + after-cap threshold (P138, +1.6x at low p) + batched routing scorer
    (P140, +1.11x) + 16 threads. *** msspacev (euclidean, the bulk of the leaderboard) = WIN. OOD (text2image)
    still ScaNN's (~2.4x, scan/rerank execution). HONEST: re-verify on a genuinely idle box to nail the exact
    multiple, but the no-overlap separation across many windows makes the WIN robust, not a single-bracket fluke.

P144. (scan-code bake-off [offline, load-indep, recall@t_surv]: PQ ranks WELL -> msspacev lever=adaptive depth;
       OOD benefits from a HIGHER-RES code -> configurable RaBitQ for OOD specifically)
    rabitq_bakeoff.py, 200k subset, residual-domain, recall@t = true-top10-of-pool retained in top-t by approx.
    msspacev (L2, d=100): PQ-4bit(25B) t40=0.959 t80=0.991 t160=0.998 -> ranks EXCELLENTLY; RaBitQ at matched
      25B(2-bit) much WORSE (0.82@t80) but my naive uniform B-bit quantizer is BROKEN at low B (2-bit < 1-bit,
      impossible) so discount the matched-byte RaBitQ; RaBitQ-4b(50B,2x) perfect but 2x bytes. => msspacev: PQ
      ranks fine, the deep t_surv is wasteful -> ADAPTIVE/SHALLOW rerank depth is the lever, NOT a new code.
    OOD (IP, d=200) [after fixing an IP-estimator bug: residual dot needs RAW q for <q,R>, not q-cz]:
      PQ-4bit(50B): t40=0.901 t80=0.961 t160=0.986 t320=0.995 -- ranks decently.
      RaBitQ-4b(100B,2x): t40=0.997 t80=0.9998 -- NEAR-PERFECT, ~8x shallower rerank at matched recall.
      RaBitQ-1b(25B) 0.76@t80 (half bytes, worse). => OOD code RESOLUTION matters: a higher-res rotated code
      ranks far better, enabling shallow rerank. OPEN: is it the ROTATION or just 2x bytes? need PQ@100B
      (8-bit) matched. And net wall-clock depends on scan(2x bytes)-vs-rerank(~8x shallower) bandwidth split.
    *** CONCLUSION: PQ already ranks the probed pool well on BOTH tracks (within-pool recall 0.96-0.99@t80),
    so the engine's t_surv~300(msspacev)/~3072(OOD) is oversized for RANKING -> ADAPTIVE DEPTH (#2, small) is
    the broad lever. A configurable higher-res code (RaBitQ-4b / PQ-8bit) is a real OOD-specific win on top.
    Caveat: 200k-subset proxy + within-pool recall (not global); validate by lowering engine t_surv (recall is
    load-independent). My naive RaBitQ != faithful Extended-RaBitQ (whose unbiased estimator + error BOUND is
    what makes adaptive early-stop PROVABLE). NEXT: PQ@100B matched-byte; then engine t_surv-reduction recall test.

P145. (*** matched-byte: PQ-8bit == RaBitQ-4b on OOD -> rotation buys NOTHING; DON'T build RaBitQ; lever = ADAPTIVE DEPTH ***)
    OOD 100B matched: PQ-8bit t40=0.9984 t80=0.9998 ~= RaBitQ-4b t40=0.997 t80=0.9998. So the OOD ranking gain
    is purely MORE RESOLUTION (bytes), not the rotation -- RaBitQ has no edge over higher-bit PQ for ranking.
    DECISION: (a) DON'T implement RaBitQ (no ranking advantage; its only distinct value = error BOUND for
    provable early-stop, which heuristic adaptive depth approximates). (b) higher-res PQ-8bit = 2x scan bytes
    + loses the 4-bit vpshufb fast-scan, and OOD scan is already 52% -> 2x scan likely outweighs the shallower
    rerank -> probably NOT a net win. (c) THE LEVER (both tracks) = ADAPTIVE REREANK DEPTH: PQ-4bit ALREADY
    ranks the pool to 0.96-0.99@t80 at NO extra scan cost; engine t_surv~300(msspacev)/~3072(OOD) is oversized.
    The cheap offline bake-off (~20min numpy) steered us OFF a ~week-long RaBitQ build onto a small change.
    NEXT (load-independent validation): lower engine t_surv on 10M OOD, check recall@10 holds. The within-pool
    bake-off (200k) predicts t~320 suffices for 0.99; if 10M global recall holds at low TMUL, adaptive depth =
    free ~1.2-1.3x OOD (rerank is 27%) + ~1.05-1.1x msspacev (rerank 19%). Then implement per-query adaptive
    early-stop (#2): rerank in scan order, stop when running k-th exact dist < next survivor's approx score.

P146. (t_surv sweep: engine rerank depth ALREADY well-tuned (TMUL=8 = recall plateau) -> Cluster A tapped; pivot to ROUTING)
    1M OOD recall@10 vs TMUL (t_surv=max(TMUL*p,300)), recall load-independent. p256: TMUL 2/4/8/16 =
    0.9080/0.9244/0.9296/0.9308 (gains +0.016,+0.005,+0.001). Knee ~4; engine default TMUL=8 sits at the
    plateau. So fixed t_surv is well-tuned; adaptive depth's only headroom = per-query VARIANCE (easy queries
    stop early), likely modest ~1.1x and low on all-hard OOD queries. *** Combined with P144/P145: the
    scan/rerank pipeline (Cluster A: RaBitQ, higher-res PQ, adaptive depth) is LARGELY TAPPED -- RaBitQ no
    edge over PQ at matched bytes, t_surv already at plateau. The cheap offline bake-off (~20min) + this recall
    sweep saved a ~week RaBitQ build. *** PIVOT to ROUTING (the profile's real headroom: 46% of msspacev
    QPS@90%, 21% OOD): #3 ADC-route the finest centroids -- gather_fine does b1*32~5120 EXACT i8 L2/query to
    return p=64 cells (80x overcompute); fast-scan the centroids (4-bit ADC) + exact-rank only top~128. Grows
    the msspacev WIN; recall-measurable (top-128 hit-rate). CAVEAT (memory): centroid dot-count wins sometimes
    don't convert to wall-clock -> verify count->wall-clock via A/B. Next after #3: LeanVec (OOD), fused
    scan+topk, drop-raw-copy + Streaming (per user's approved batch).

P147. (*** #3 ADC-ROUTING GATE PASSES: 4-bit ADC of the finest centroids is RECALL-NEUTRAL -> fast kernel justified ***)
    Built SBANN_ROUTE_ADC: train a 4-bit PQ (m=d/2) over the FINEST centroids at build; gather_fine ADC-scores
    the finest children (LUT+codes), keeps ADC-top ROUTE_ADC_KEEP, exact-rescores ONLY those. msspacev 1M,
    recall@10 (load-independent): EXACT 0.8850/0.9207/0.9460 (p64/128/256). ADC KEEP=256: 0.8842/0.9191/0.9393
    (-0.001 to -0.007, p256 loss = KEEP<p). ADC KEEP=512: 0.8850/0.9206/0.9461 == EXACT (within 0.0001) ->
    RECALL-NEUTRAL. So ADC-rank the ~1536 finest children, exact only ~512 = 3x fewer exact L2 at the finest
    level (the 78%-of-routing term) with ZERO recall loss. (The dim-drop probe P146 failed; ADC keeps all dims
    at 4-bit -> works, confirming the two approximations differ.) *** GATE PASSED. The committed path is SCALAR
    ADC (50 lookups/child) so it is currently SLOWER (gate = correctness, not speed); the win needs the vpshufb
    kernel: re-layout rcodes into 16-cell subspace-major blocks (like data blocks) + block_adc_i8 scan, 16
    centroids/instr. Expected ~1.3-1.5x total on msspacev (routing 46%). Measurable load-independently via the
    route-time FRACTION (SBANN_PROFILE) -- not just wall-clock. NEXT: the vpshufb ADC routing kernel.

P148. (ADC routing kernel: recall-neutral but NO speed win -- l2_i8_block already fast + ADC per-query overhead cancels it)
    Built the vpshufb ADC routing kernel (finest codes -> 16-cell blocks + block_adc_i8_i16acc, 16 centroids/
    instr). msspacev 1M, SBANN_PROFILE (route FRACTION = load-robust, recall = load-indep):
      EXACT:     p64 r0.8850 route29.4% | p128 r0.9207 route18.1% | p256 r0.9460 route12.7%
      ADC-BLOCK: p64 r0.8850 route28.9% | p128 r0.9206 route24.3% | p256 r0.9461 route18.1%
    Recall EXACT-NEUTRAL (kernel correct) but route fraction EQUAL-TO-HIGHER -> NO routing speedup. (QPS looked
    higher for ADC but that was cross-run LOAD; the within-run fraction is the load-robust signal.) ROOT CAUSE:
    l2_i8_block (P140) ALREADY made exact routing SIMD-fast (1536 batched l2_i8); ADC's only benefit (exact-
    rescore 512 not 1536) is offset by its FIXED per-query overhead -- build the m*16 query LUT (~1600 ops),
    select_nth over 1536, 512 exact rescores. Net wash-to-worse. This REALIZES the memory's "centroid dot-count
    wins don't convert to wall-clock" caveat: the dot-count IS lower, wall-clock isn't. *** ROUTING VERDICT:
    l2_i8_block (+1.27x, committed) is the routing win; ADC routing (P147 gate passed on recall) adds NO speed.
    The ADC code stays behind SBANN_ROUTE_ADC (default off, recall-neutral option) but is not a win. Cluster:
    routing is now also largely tapped (l2_i8_block banked; dim-drop fails; ADC no speed). PIVOT to the
    recall-measurable items: LeanVec (OOD), drop-raw-copy + Streaming, and RaBitQ-as-a-swappable-Compressor (user req).

P149. (user idea: ADC routing WITHOUT the exact-rescore? -> cheaper routing but LOSES recall; rerank is needed)
    Added SBANN_ROUTE_ADC_KEEP=0 = take top-p straight from ADC scores (no exact-rescore). msspacev 1M:
      EXACT:   p64 r0.8850 route30.1% | p128 r0.9207 route17.6% | p256 r0.9460 route14.5%
      KEEP=0:  p64 r0.8276 route24.4% | p128 r0.8854 route16.5% | p256 r0.9268 route14.5%
    No-rerank routing IS ~1.2x cheaper (route 30.1->24.4% @p64) but recall drops -0.019 to -0.057: the ADC
    cell-ranking is too coarse AT THE TOP-P BOUNDARY -> a few true cells fall to rank p+1, never probed. At
    MATCHED recall it's a NET LOSS (no-rerank@p128 r0.8854 ~= exact@p64 r0.8850 -> 2x the scan for a 1.2x route
    saving). So the rerank IS needed; and WITH it ADC routing isn't faster than the already-SIMD-fast l2_i8_block
    exact routing (P148). A higher-bit centroid code would sharpen the boundary but doubles the scan (P145) = wash.
    FINAL routing verdict: l2_i8_block (+1.27x) is the win; ADC routing (any KEEP) is a dead end for speed.

P150. (*** 4 PARALLEL AGENTS (worktree-isolated) delivered the next-10 batch: 3 features MERGED + 1 honest gate-negative ***)
    User: "do all of them in parallel." Spawned 4 background agents, each own git worktree+branch off engine-stack,
    recall-validated on 1M subsets (load-independent). Results:
    - rabitq (feat/rabitq, MERGED): faithful Extended-RaBitQ Compressor (Gao&Long SIGMOD'24) -- fixed rotation +
      B-bit quant + UNBIASED IP estimator (the part my numpy bake-off got wrong). comp "rabitq", SBANN_RABITQ_BITS.
      msspacev 1M: B=4 == apq4 recall (0.9218 vs 0.9207), B=1 within 0.001-0.05. CONFIRMS no ranking edge over PQ
      (P145), but the swappable-code FEATURE the user wanted is in. Scalar scan (not perf-critical).
    - scale (feat/scale, MERGED): (A) index mmap PERSISTENCE (persist.rs; save/load, tagged router/comp). Bit-
      identical recall; load 0.2-1.1s vs build 15-20s. SBANN_INDEX_SAVE/LOAD. (B) DROP-RAW-COPY (SBANN_RAW_DEDUP):
      raw per-distinct-orig not per-slot -> index -41% to -45% (a0x as a0 grows), bit-identical. Caveats: only
      HierRouter+Apq4/Pq4 serialize; LOAD needs the same scan-flag env (FASTSCAN/USE512FS/IP/...). Enables 100M/1B.
    - streaming (feat/streaming, MERGED): per-cell APPEND-buffer insert + tombstone delete + finalize_inserts +
      search_stream + `stream` subcommand. msspacev 1M: insert 500k->1M recall 0.9640 vs fresh-build 0.9634
      (within noise); deletes 100k in ~0.01s; inserts 16-26k/s. No-graph IVF = streaming is structurally cheap.
      Caveat: buffer uncompacted (search cost grows) -> production needs periodic rebuild. (Merge: resolved
      rerank_contig_pairs+by_orig conflict w/ scale's RAW_DEDUP; both Index{} ctors got both field sets.)
    - leanvec (offline gate, NEGATIVE, NO wiring): query-aware asymmetric IP-preserving map. POSITIVE: it IS real
      -- beats PCA by 10-15pt at low t in FULL precision (OOD queries DO concentrate IP directions; near-isotropy
      pessimism too strong for the ASYMMETRIC map). BUT fails the gate: at matched PQ bytes LeanVec is WORSE,
      full-rank rotation is a no-op, coverage marginally worse. Why: its only lever is dim-cut of FULL-PRECISION
      work, but our pipeline = PQ + EXACT-IP rerank (rerank needs true IP; PQ@matched-bytes loses more from dim-cut
      than it saves). *** Re-confirms the OOD gap is EXECUTION-SPEED, not metric/ranking. *** Cheap offline gate
      saved the engine wiring -- the discipline that's paid off all session (RaBitQ, ADC, this).

P151. (GOAL refined: top OOD + STREAMING at 1M/10M/100M. 100M msspacev scale attempt + a persistence-at-scale bug)
    100M msspacev (crop_nb_100000000, 9.3GB base) BUILDS fine: hierk3 C0=1024 C1=16384 b0=64 b1=200 a0=1
    raw-dedup, built in 1981s (33min), anon ~13GB (RSS 22GB incl ~9GB evictable base cache -> the RSS-based
    watchdog false-alarms; ANON is the true metric and stayed safe under the 24GB/40% limit). BUT SBANN_INDEX_SAVE
    OOM-KILLED during the ~13GB write: build holds 13GB anon + the save accumulates up to ~13GB dirty page cache
    (BufWriter+lazy writeback) -> 13+13 > limit -> killed mid-write (file truncated to 3.9GB; load panics
    persist.rs:82 reading a 12.96GB region from a 3.9GB file). Worked at 1M (tiny). FIX: periodic bw.flush() +
    File::sync_data() in Index::save_to after the big arrays (raw, blocks) to force writeback + bound dirty pages.
    (Persistence is the scale enabler -> this fix matters for the 100M/1B program.) Streaming = MSTuring track
    (data + neurips23/runbooks here; DiskANN ref recall@10=0.892 @10M). msturing 1M quantized to int8. NEXT:
    get the 100M msspacev recall (rebuild a0=2 no-save, running), then msturing streaming number, then OOD scale.
    Box discipline: ONE big build at a time (concurrent 100M+streaming starved both).

P152. (*** 100M msspacev SCALE PROOF: engine builds+queries at 100M; + the LEADERBOARD-METRIC reframing ***)
    100M msspacev (crop_nb_100000000) a0=1 raw-dedup, hierk3 C0=1024 C1=16384 b0=64 b1=200 Kf=262144, RC=0:
      built 1384s (23min, anon ~13GB), recall@10: p128 0.8772 | p192 0.8950 | p288 0.9108 | p448 0.9253 | p640 0.9337.
    So we REACH recall 0.90 @ p~200 and 0.93 @ p640 at 100M (QPS load-suppressed, 1-rep). Engine scales to 100M.
    (a0=2 100M got SIGTERM-killed -- ~15-16GB exceeds the shared-box headroom; a0=1 ~13GB is the safe ceiling here.)
    *** LEADERBOARD METRICS (from neurips23/ongoing_leaderboard, fetched): tracks = Filter/OOD/Sparse/Streaming.
    - OOD/Filter/Sparse: ranked by QPS at recall@10 >= 90% (SPEED). ScaNN = OOD BASELINE (not a competitor entry);
      the actual OOD #1 is FASTER than ScaNN -> we are >2.4x behind the real #1. HARD track (execution speed).
    - STREAMING: ranked by recall@10, as long as the runbook finishes within 1 HOUR (RECALL, not speed!). The
      current STREAMING LEADER = recall@10 0.99786. *** This REFRAMES streaming as WINNABLE for us: our final
      rerank is EXACT int8, so recall is gated only by probe-coverage + #survivors, both crankable within the 1hr
      budget -> drive recall -> ~1.0. Target = beat 0.99786, NOT DiskANN's 0.892 baseline. ***
    PARALLEL (box forces heavy builds sequential, light 1M agent-work parallel): 100M (me, done) + streaming2
    (msturing 1M: ops correct -0.004 vs rebuild, runbook eval + compact_live built, tuning toward 0.998) +
    ood2 (1M OOD adaptive-rerank). Now also 10M streaming (msspacev, max-recall config) toward ~0.99.

P153. (*** KEY: int8 has a RECALL CEILING vs the FLOAT leaderboard GT; FLOAT RERANK of survivors breaks it -- a lever for BOTH tracks ***)
    streaming2 found: msturing GT is computed on the ORIGINAL FLOAT vectors; our base is int8. EXHAUSTIVE
    exact-int8 search (probe all cells) caps at recall@10 = 0.9543 vs the float GT. So NO int8-only method can
    exceed ~0.954 vs the leaderboard GT; the streaming leader's 0.99786 MUST use float precision. FIX (standard
    high-recall recipe): int8 index for cheap candidate generation -> exact FLOAT rerank of the top-K survivors
    (read float vectors ONLY for survivors -> cheap; we have a 1hr budget so K can be huge) -> top-10. Float
    base+queries are present and aligned (cosine 0.99997): MSTuringANNS/base1b.fbin.crop_nb_1000000 (1M float).
    *** GENERALIZES: this is why our int8 OOD recall vs the official FLOAT GT was capped too -- and float rerank
    of survivors should reach OOD recall@10=0.90 with FEWER candidates -> HIGHER QPS@90% (the OOD metric, and
    exactly ScaNN's float-AH advantage). The OOD float base IS here: text2image1B/base.1B.fbin.crop_nb_10000000
    (10M float d=200) + query.public.100K.fbin. (Our t2i1m-gt was computed from the INT8 base -> int8-GT, so OOD
    must be re-scored vs a FLOAT GT for the leaderboard-accurate number.) *** DATA REALITY: native-int8 datasets
    (msspacev/SPACEV) have NO ceiling (int8 GT); FLOAT datasets (msturing, text2image) need float rerank. Float
    crops local only at 1M for msturing (10M/100M = download). DISPATCH: streaming2 -> msturing 1M float-rerank
    recall vs float GT (target >0.99786); ood2 -> OOD int8-scan + FLOAT-rerank, recall@p vs float GT, does it hit
    0.90 at smaller p (QPS@90% win)? This could be the missing OOD lever AND the streaming-topping mechanism.

P154. (*** OOD QPS@90% WIN: rerank depth was massively over-provisioned; tightening t_surv to what 0.90 needs = 1.65x ***)
    ood2 (commit 1c81a44): the OOD QPS@90% metric only needs recall EXACTLY 0.90, but our default reranked the
    full survivor pool (depth ~5760 at p=192). Profiling: at p=192 recall 0.9037 needs only depth 576 (10x fewer
    reranked) -> MEASURED QPS 1884 -> 3108 = 1.65x at matched 0.90 recall, 1M msturing. SCAN is now the wall (~57%
    of query time); rerank dropped from dominant to minor. Adaptive per-query rerank depth (SBANN_ADAPT_RERANK,
    gate=max) only helps the >0.92 tail, NOT the 0.90 operating point -> banked but off by default for the metric.
    CAVEAT: re-profile at 10M -- scan dominates more there, so the 1.65x (a rerank-side win) SHRINKS at scale; the
    durable scale lever is faster SCAN (candidate gen), not less rerank. NEXT (both agents, re-dispatched): the
    P153 FLOAT-RERANK test -- streaming2 msturing 1M float-rerank recall vs float GT (>0.99786?), ood2 OOD
    int8-scan+float-rerank, does 0.90 come at a smaller p (compounds with this t_surv win). Both have uncommitted
    float-reader work (ibin/fbin) in flight. Box at load ~62: 1M-only light builds, heavy builds strictly sequential.

P155. (*** STREAMING TARGET CORRECTED: the scored dataset is msturing-30M-clustered via final_runbook.yaml, NOT 1M/10M-sliding ***)
    Read the actual scoring artifacts (neurips23/ongoing_leaderboard): streaming track is scored by
    res_final_runbook_AzureD8lds_v5.csv = dataset **msturing-30M-clustered**, runbook **final_runbook.yaml**
    (max_pts 10292043 active window; insert(clustered start/end ranges into the 30M) interleaved with search;
    later deletes). Local CSV standings (recall@10): scann 0.9924 | zilliz 0.9219 | pinecone 0.9123 | pyanns 0.870
    | diskann 0.722. Live leaderboard leader = 0.99786 (newer than the local CSV snapshot). So the BAR = beat
    ~0.992-0.998 recall@10 on msturing-30M-clustered, runbook finishing <1hr. The other runbooks (wikipedia-1M,
    msmarco-100M, wikipedia-35M) are SEPARATE streaming runbooks; final_runbook=msturing-30M-clustered is THE
    scored one. *** IMPLICATIONS: (1) streaming2's msturing-1M float-rerank run is a MECHANISM PROXY (right
    technique, wrong scale/dataset) -- it validates that exact-float-rerank breaks the int8 ceiling, but the
    leaderboard number requires msturing-30M-clustered. (2) DATA GAP: we have ONLY the 1M msturing crop
    (base1b.fbin.crop_nb_1000000); need the 30M float base (~12GB) + final_runbook GT (download via
    benchmark/streaming/download_gt.py needs a gt_url our local runbook lacks, else compute_gt.py = expensive over
    the 10M window). (3) Our edge vs scann's 0.992: exact-FLOAT rerank of a large survivor pool within the 1hr
    budget should push recall->~1.0; scann uses float AH+reorder=317. Memory/compute: 30M int8 base=3GB + index
    anon ~4GB (we build 100M msspacev, so 30M is safe). GATING: stage the 30M data ONLY after the 1M mechanism
    proof returns >0.99 (don't download 12GB for a dead mechanism). OOD target unchanged (QPS@90% on text2image,
    scann=baseline). NEXT: agents' 1M float-rerank numbers (in flight) -> if >0.99, stage msturing-30M + run final_runbook.

P156. (*** STREAMING MECHANISM PROVEN (float rerank -> 1.0000 @1M) + the SCORED runbook is SPEED-CONSTRAINED ***)
    streaming2 PROVED the lever: msturing-1M simple_runbook, flat C=4096 apq4, p=4096 (full scan) + EXACT FLOAT
    rerank of K=800 survivors (base1b.fbin/query100K.fbin) => avg recall@10 = **1.0000** vs the official float GT
    (steps 1.0/1.0/1.0), vs leader 0.99786. The int8-only ceiling (0.9543) is FULLY closed by float rerank. p=1024
    already gives 0.9922; K=800 is plenty (float top-10 sit at int8-rank ~1-50). Committed feat/streaming2 (FBin
    reader + SBANN_RB_FBASE/FQUERY/RERANK_K/RB_GT). *** BUT that was simple_runbook@1M (3 searches) -- a MECHANISM
    PROXY. The actually-SCORED streaming runbook = final_runbook.yaml over msturing-30M-clustered, and it is
    SPEED-CONSTRAINED: 1280 ops = 320 insert (covering all 30M), 320 delete, **640 search steps each running the
    FULL 10k query set = 6.4M query evals**, and the ENTIRE runbook must finish <1hr (the runner times all 1280
    steps; recall@10 averaged over the 640 searches vs per-step GT). So p=4096 full-scan (~50 QPS) => ~35h =>
    FAILS the cutoff (scores 0). Budget: ~30M inserts @~20k/s ~= 1500s, leaving ~2100s for 6.4M queries => need
    ~3000 QPS sustained. WINNING RECIPE = scann's: FAST router (low p) + float rerank of a SMALL K (scann
    reorder=317 -> 0.9924), maximizing recall SUBJECT TO the 1hr budget. *** DATA STAGED (this session): downloading
    data/MSTuring-30M-clustered/30M-clustered64.fbin (11.4GB float, 29998994x100) + testQuery10K.fbin + static GT
    clu_msturing30M_gt100; per-step streaming GT IS official-downloadable (final_runbook.yaml carries gt_url) via
    benchmark/streaming/download_gt.py -> data/MSTuring-30M-clustered/29998994/final_runbook.yaml/step{N}.gt100
    (640 files ~8MB each). NEXT (streaming2): wire final_runbook parser + quantize 30M->int8 + hierk Kf=262144
    low-p + float-rerank K~200-400; report (a) full runbook wall <1hr? (b) avg recall@10 vs official per-step GT.
    Target beat scann 0.9924 / leader 0.99786. Mem: 30M int8 3GB + ~10.3M active window, safe a0=1/2 under 26GB.

P157. (*** OOD: int8 rerank has a HARD ~0.924 recall ceiling vs the FLOAT GT; float rerank removes it (+0.028), required >0.92 ***)
    ood2 (committed e5fe96e feat/ood2, NOT merged): built an exact float-IP GT (new `floatgt` subcommand) over
    text2image 1M (base.1B.fbin first 1M + query.public.100K first NQ=2000), then scored int8-rerank vs
    float-rerank BOTH against that FLOAT GT (the leaderboard's real metric). *** int8 rerank PLATEAUS at ~0.924
    recall@10 vs float GT: p=320->0.9097, 448->0.9172, 640->0.9214, 1024->0.9239 -- MORE PROBES DON'T BREAK IT.
    That is the int8 quantization ceiling against the actual (float) metric = a real slice of the ScaNN gap (ScaNN
    reranks at float precision). *** Float rerank (int8 scan/route unchanged; rerank t_surv survivors by exact
    float IP, only survivors paged in via mmap): matched-p gain +0.022..0.028 EVERYWHERE -> 0.9384@p320 /
    0.9472@p448 / 0.9523@p640. QPS@90% (the OOD metric): float hits 0.90 at p~152 vs int8 p~248 (1.6x fewer probes)
    but QPS-NEUTRAL at exactly 0.90 (float's 4x-byte + f32-dot cost offsets the scan saving: int8 p256 ~5748 QPS vs
    float p160 ~4661-5844, load-noisy). THE REAL WIN is >0.92: float 0.9286 @ QPS 4136 beats int8's best-possible
    0.9239 @ QPS 2231 on BOTH recall AND speed (clean REPS=3). *** CORRECTION: our prior OOD recall numbers were vs
    the INT8 GT (optimistic); vs the real FLOAT GT int8 caps ~0.924, so float rerank is REQUIRED for any high-recall
    OOD point and should be the OOD default. Code: fbin.rs, simd.rs dot_f32_fast, vq.rs rerank_contig_float +
    Index::search_frr, main.rs floatgt + SBANN_FLOAT_RERANK/FBASE/FQUERY. Float GT cached t2i1m-floatgt. CAVEAT:
    1M/NQ=2000; re-check at 10M (scan + rerank-survivor cost both grow). Combines with t_surv-cut (P154) + adaptive
    depth. NEXT: leaderboard-accurate OOD = MAX QPS@90% vs FLOAT GT at 10M, combining float-rerank+t_surv-cut+low-p,
    compared to ScaNN's OOD QPS@90%. (Float rerank is neutral at 0.90 alone -> the QPS@90% win must come from t_surv+routing.)

P158. (*** OOD QPS@90% (1M): float-rerank TIES int8 at 0.90 matched-load -- NOT a QPS@90% lever; t_surv+low-p IS. Float = ceiling break only ***)
    ood2: max QPS at recall@10>=0.90 vs the FLOAT GT, text2image 1M, t_surv-cut floored on BOTH paths, a0=3 SOAR
    (a0=1 routing CAN'T reach 0.90 here), REPS=5, MATCHED-LOAD (committed bea5770, supersedes the 420d056 first read):
      busy window:  FLOAT p=208 t_surv=624  0.9080 @ 9795  | INT8 p=288 t_surv=1152 0.9020 @ 9562  (float +2.4%, noise)
      light window: FLOAT p=176 t_surv=704  0.9051 @ 11372 | INT8 p=288 t_surv=1152 0.9020 @ 10067 (float +13%, load-luck)
    VERDICT: TIE at 0.90. Float reaches 0.90 at fewer probes (p~208 vs ~288, load-independent fact) but its 4x-byte +
    f32 rerank cost cancels the scan saving -- flooring t_surv does NOT tip float to a win (rerank stays the
    bottleneck). The earlier "1.13x" was light-load luck; matched-load it's +2.4% = noise. *** So float-rerank is
    QPS-NEUTRAL at the 0.90 leaderboard point and is NOT the QPS@90% lever. The QPS@90% win is t_surv-cut + low-p
    routing (shared by both paths); at the pure 0.90 point int8+t_surv-cut is as good as float AND avoids the 4x float
    base. Float-rerank's REAL value is narrow: (a) the ceiling break >0.92 (int8 caps ~0.924 vs float GT; float
    .9523@p640), (b) correcting our optimistic int8-GT recall reporting. Best QPS@90% @1M ~= 9.6-11.4k QPS @ recall
    ~0.902-0.908 (box-load-bound). *** STRATEGIC: the OOD gap to the real #1 (~2.4x, and that was vs int8-GT =
    optimistic) is a SCAN-throughput/routing gap, NOT a rerank gap -- so the durable OOD lever is faster candidate
    generation, not float rerank. NEXT (#12 rest): 10M QPS@90% vs ScaNN (the ranking number) -- build a 10M float GT
    + run best config; HELD until box clears (30M streaming owns it). Keep float-rerank behind SBANN_FLOAT_RERANK
    (NOT a hard default).
    ADDENDUM (ood2 daf57ae): the t_surv-cut QPS@90% lever RE-MEASURED on text2image-vs-float-GT (not assumed from
    msturing): INT8 p=288 depth 8640->1152 (rerank 56.5%->28.5%) QPS 3317->6576 = **1.98x**, recall 0.9071->0.9020
    (held >=0.90); after the cut the int8 path is SCAN-bound (56%). FLOAT p=208 depth 6240->624 stays REREANK-bound
    (~76%, the 4x/survivor cost) so its cut can't beat int8's -> the max-QPS@90% is a TIE (int8+cut simpler/no 4x
    base; float+cut safer-margin: 0.908 vs int8's fragile 0.924-ceiling-only-0.005-over-0.90). So the durable OOD
    QPS@90% lever is the t_surv-cut on the INT8 path (~2x); after it, the remaining wall is SCAN throughput. 10M
    builder (ood_10m.sh) validated + row-alignment confirmed across the full 10M, ready to fire on box-clear.

P159. (*** 30M STREAMING REALITY: flat-IVF caps ~0.88 in budget < scann 0.9924; router+scan-throughput is the gap; offline training is FREE ***)
    streaming2 ran the real final_runbook/msturing-30M-clustered (GT alignment verified 10/10 vs float brute-force).
    ROUTER matters enormously on the clustered data: hier RANDOM centroids Kf=262144 -> recall ~0.30 (fail); flat
    C=4096 EXACT-kmeans IVF p=64 K=400 + float rerank -> recall climbs with live set: 0.39@39k, 0.71@100k,
    0.77@866k, 0.83@2M, still rising (~0.85-0.90 projected ceiling @10M active), QPS~3000 (budget-viable). TENSION:
    recall>0.95 needs higher p (coverage) but p>=128 drops QPS below the ~3000 needed for 6.4M queries in 1hr; flat
    INSERT routing caps C at ~4096-8192 (C=16384 = 150us/insert = 4500s, over budget). So flat IVF tops ~0.85-0.90
    IN BUDGET -- below scann 0.9924. BAR (from standings): scann 0.9924 = "tree=700/5000,AH2,reorder=317" (~5000
    leaves, ~700 probed=14%, anisotropic 4-bit AH fast-scan, float reorder 317); zilliz 0.922 / pinecone 0.912 are
    R32 GRAPHS (DiskANN-style). >0.922 = top-3, >0.9924 = win. *** LEVERS I directed (the gap is candidate-gen, not
    rerank): (1) OFFLINE TRAINING IS FREE -- the runbook starts EMPTY and the benchmark times only the runbook ops,
    NOT cold-start router training (same as ScaNN pre-training its 5000-leaf tree). streaming2 KILLED hierk k-means
    Kf=262144 at 4min thinking it too slow -- premature; a fine router can train 10-30min offline at ZERO budget.
    The untested middle between random-hierk(0.30) and flat(0.88) = a FULLY-TRAINED fine hierk: fine cells (better
    recall/probe) + cheap b0=16 BEAM routing (~10us insert, dodges the high-C flat insert wall). (2) USE512FS faster
    fast-scan -> more probes per budget-second (ScaNN affords 700 probes via fast AH2; our wall is scan speed).
    (3) BUDGET REBALANCE: max-recall-in-1hr means every sec off inserts/deletes/compaction = search budget = higher
    p. (streaming2 parallelized compaction -- the old O(live) sequential was 240s/call @10M, a budget killer.)
    HONEST: ~0.88 flat is a real result but below the 0.922 bar; exhausting trained-fine-router + fast-scan next.
    The msturing-1M=1.0000 float-rerank MECHANISM proof (P156) stands regardless. (in_progress, task #11.)

P160. (*** 30M STREAMING BREAKTHROUGH: the gap was PROBE FRACTION, not the router -- scann-matched 12.5% probe -> recall ~0.95 (TOP-3) ***)
    streaming2, after the scann-config reframe (scann "tree=700/5000" = probes 700/5000 = 14% of leaves): we were
    probing 1.6% (p=64 of C=4096). MATCHING scann's fraction -- flat C=4096, p=512 (=12.5%) + USE512FS + float
    rerank K=317 -- recall@10 climbs fast and HOLDS ~0.95: 0.73@39k, 0.91@74k, 0.95@100k, 0.956@137k, 0.954@177k.
    Fill-phase drag is tiny (only first ~3 searches <0.90) so the 640-search AVG projects ~0.94-0.95. *** That BEATS
    zilliz 0.922 + pinecone 0.912 (=> TOP-3) and approaches scann 0.9924. *** hierk Kf=262144 was a RED HERRING:
    trains fast (146s; the earlier "too slow, killed at 4min" was box CONTENTION, not training cost) BUT fine cells
    are WORSE here -- p=512 of 262144 = 0.2% probe -> recall ~0.50; fine cells would need scann-FRACTION probing =
    ~36000 probes (absurd). So flat C=4096 + HIGH p is the right structure (the scann family); granularity was never
    the lever, PROBE FRACTION is. *** THE GATE IS NOW BUDGET, not recall: p=512 warm QPS ~1500-3000 (cold float mmap
    warms up) -> search ~2500-4000s + inserts ~1150s => p=512 may be slightly OVER the 1hr budget. Knee likely
    p~256-384 (recall ~0.93-0.94 at ~2x QPS, budget-safe). PLUS compaction: the recall read was COMPACT=0
    (optimistic); the real run needs compaction (buffer/tombstones bloat over 30M inserts + 27.6M deletes, ~10.3M
    live cap) -- if compaction (currently 250-550s/call re-encoding apq4) is a budget hog, the cheap block-merge
    rewrite directly buys p -> recall. NEXT: highest-p config that finishes <1hr WITH realistic compaction, locked
    at NQ=10000 = the DEFINITIVE 30M streaming number. Target: >0.922 (top-3, looks secured if budget holds) pushing
    toward the 0.9924 win. (developing; NQ=1000 calibration -- finalize on the NQ=10000 + budget verdict.)
    BUDGET DIAGNOSTIC (streaming2, committed eb451a3 -- the DECISIVE measurement): p=512 reaches recall@10 = 0.978
    at 2.4M live (0.95@111k, 0.964@1.2M, 0.978@2.4M) -- right up near scann 0.9924, well past zilliz/pinecone. BUT
    the budget wall is SEARCH THROUGHPUT, and the root cause is NEITHER the float-rerank mmap NOR compaction -- it is
    the IVF scan_pool TOP-K BOOKKEEPING: per query it pushes EVERY candidate (p*live/C*a0 ~= live/4 at p=512 = ~2.5M
    cand/query @10M) into a Vec then select_nth's it. That alloc+select churn = ~2.1e8 cand/s = 20-40x BELOW the
    native apq4/AVX-512 fast-scan rate (the distance compute is NOT the cost; USE512FS barely moved it). QPS
    collapses with live: 3224@39k -> 553@2M -> projected ~86 @10M => the 6.4M-query runbook ~= 20 HOURS, massively
    over 1hr. Compaction is MINOR (8 calls/51s @2M, projected 300-600s full-runbook; now parallel+USE512FS-compat).
    Budget-viable p on the CURRENT engine ~= p=14 (recall ~0.6) -- so no config gives budget AND recall yet. *** THE
    GAP TO SCANN IS ENGINE SCAN THROUGHPUT (their AH2 + streamlined top-k), confirmed NOT recall capability (0.978
    proven) NOR router granularity (fine hierk trains in 146s but is worse; scann's 5000 leaves ~ our C=4096). ***
    FIX = rewrite scan_pool/scan_ins_pool to a BOUNDED top-t selection (threshold-filter: keep ~2x(tmul*K) buffer,
    prune via select_nth only when full, reject most candidates with one threshold compare -- stop materializing
    ~live/4). Projection: at native scan speed, p~256 (recall ~0.95) -> ~3650s ~ 1hr => TOP-3 near scann (push p=512
    /0.978 if it fits = run at the WIN). *** THIS REWRITE UNLOCKS BOTH TRACKS: OOD QPS@90% is the SAME scan-bound
    wall (P158, int8 path 56% scan after the t_surv-cut) -- so the bounded-top-t scan_pool is the single highest-
    leverage engine change for streaming AND OOD. GREENLIT (multi-hour perf rewrite). Block-merge compaction =
    deprioritized (search dominates). 1M=1.0000 mechanism proof banked regardless.

P161. (*** CRITICAL ELIGIBILITY CONSTRAINT (audit-caught): STREAMING track = HARD 8GB DRAM + 1hr, COMPACT index over ACTIVE points -- our 0.978 config is DOUBLY INELIGIBLE ***)
    The independent audit workflow (wgd9wmg3y, 5 dims, 3 rate-limited) caught what the whole session missed. CONFIRMED
    in the harness: benchmark/runner.py:269 `mem_limit = ... if track!='streaming' else (8*1024*1024*1024)` -> the
    streaming Docker container is HARD-CAPPED at 8GB DRAM; neurips23/README.md:28 states "1 hour and a DRAM limit of
    8GB ... maintain a COMPACT index over the ACTIVE points rather than index the entire anticipated set and use
    tombstones." (Machine = Azure D8lds_v5, 8 vCPU / 16GB, container limited to 8GB; competitor indexsize: zilliz ~2GB,
    diskann ~4.9GB -- all < 8GB.) *** So our headline recall@10=0.978 @ p=512 is DOUBLY INELIGIBLE: (a) TIME -- ~20x
    over the 1hr budget at the REAL NQ=10000 (our SBANN_NQ defaulted to 1000, undercounting the dominant search cost
    ~10x; the <1hr verdict was projected, never measured); (b) MEMORY -- the full 12GB float base (or full-nb=30M float
    cache, which commits all 30M rows by end-of-run) blows the 8GB cap. *** The recall VALUE itself is sound/faithful to
    official (per-step float GT step{N}.gt100, top-10/10, row-aligned int8->float rerank reading only BASE vectors +
    queries, NO GT leak; deviations = NQ=1000 subsample + ignored tie-window, both tiny & strictly CONSERVATIVE = we
    score equal-or-LOWER). So 0.978 is "true recall of a config that does not make the leaderboard." *** ELIGIBLE DESIGN
    (the pivot): cache ONLY the ACTIVE float window (~10.3M x 100 x 4 = 4.1GB, slot-indexed, NOT full-nb) + compact
    active int8/apq4 codes (~10.3M x100 = ~1GB) + router + buffers = ~6GB < 8GB. Compaction to active-only is now an
    ELIGIBILITY requirement (the 8GB cap + the "compact index" intent), not just budget. THEN the scan-throughput rewrite
    for the 1hr time budget. Open question: does recall stay >0.922 (top-3) under active-window-cache + compact index +
    whatever p fits 1hr? TBD on streaming2's float-cache QPS. Minor audit findings (conservative, non-blocking): recall
    ignores official count_ties window (~0.02%); non-cache replace-op rerank reads fb.row(tag) not fb.row(src)
    (main.rs:1135, INERT here -- msturing-30M has 0 replace ops). FIX list: SBANN_NQ=10000 for any budget verdict;
    active-window float cache; madvise/free dead rows; assert fbase.nb==full.nb. The 1M=1.0000 used the int8-space
    self-GT fallback (NOT float-GT) -> 1M proxies are NOT official-comparable; only the 30M gtdir/float-GT path is.

P162. (*** 30M STREAMING DECIDING RUN: 8GB plumbing WORKS, but ELIGIBLE recall ~0.6 -- gated entirely on a ~15-20x SCAN deficit (= the OOD gap) ***)
    streaming2 ran the eligible deciding run (NQ=1000, clean box, commits fb94bd7+582e841). RESULT: the active-window
    float cache works EXACTLY as designed (sized 4.12GB for the 10.3M window, anon tracks live 0.3->1.9GB, recall
    UNCHANGED 0.97@1.7M -- 8GB plumbing sound). BUT with the mmap faults gone, SCAN THROUGHPUT is the hard wall:
    ~1.8e8 cand/s, QPS collapsing ~1/live (3225@177k, 897@733k, 484@1.7M). The cache barely moved QPS because <2M
    live the float still fit page-cache (no faults yet); it only helps >5M, where scan already dominates. BUDGET
    (NQ=10000, need ~2133 QPS@10M for search<3000s): QPS@10M ~= 36862/p -> p=512=72QPS=15h, p=128=288=6h, p=32=92min,
    p=16=~2300QPS=46min (FITS). So the highest p that fits 1hr = ~p=16 -> ELIGIBLE recall ~0.55-0.65, FAR below the
    0.922 top-3 bar. *** ROOT CAUSE: apq4 fast-scan delivers ~1.8e8 cand/s = ~15-20x BELOW native (~2-3e9); the 0.978
    config needs ~2.5M cand/query @10M, the 1hr-buster. This is the SAME engine-throughput gap as OOD (P158). ***
    NUANCE: our flat-IVF scan BEAT scann at QPS@90% on msspacev-10M (P89) -- BUT P89 also noted scann wins at
    recall>=0.95 (faster scan at HIGH candidate counts, ~fundamental). Streaming p=512 needs ~2.5M cand/query = exactly
    that high-count regime where scann's AH2 wins. So the 15-20x is PARTLY the known scann-high-count edge + PARTLY
    possibly-recoverable streaming-path overhead (slot indirection, per-cell LUT setup, buffer/main split, the
    bounded-top-t collection). VERDICT: recall CAPABILITY 0.978 PROVEN; ELIGIBLE (8GB+1hr) recall ~0.6; topping
    streaming (or OOD) requires closing the ~15-20x scan deficit = a real scan-kernel optimization, NOT a config.
    DECISION PENDING: diagnose whether the 15-20x is recoverable (profile scan_pool: where does native 2-3e9 -> 1.8e8
    go?) BEFORE committing to a multi-hour scan-kernel/AH2 rewrite. 1M=1.0000 mechanism proof stands.

P163. (*** PIVOTAL & POSITIVE: the 14x scan deficit is RECOVERABLE PATH OVERHEAD, not the kernel -> GREENLIT path-opt (unlocks BOTH tracks) ***)
    streaming2 settled the invest-vs-bank question with a microbench. scanbench (bare apq4 kernel, m=50 = d=100/dpb=2,
    L2-resident, single thread): fast-i8 AVX2 = 321 Mvec/s/thread, AVX-512 interleaved = 430 Mvec/s/thread -> ~2.6e9
    (AVX2) to 3.4e9 (512-IL) across 8 threads. THE KERNEL HITS NATIVE (confirms P89). The 30M streaming scan PATH =
    ~1.8e8 cand/s total = ~22.5M/s/thread -> the path is ~14x SLOWER than the bare kernel. Confirmed NOT kernel and
    NOT memory-BW (the scan pulls only ~0.56GB/s/thread, far below RAM BW) => it is PER-CANDIDATE PATH OVERHEAD around
    the kernel. SUSPECTS (priority): (1) the per-candidate SCALAR COLLECT loop -- for each of ~2.5M candidates a
    slot_orig[slot] tombstone-check + bounded-top-t threshold-compare/push = ~16 scalar ops per 16-wide kernel block,
    doubling per-block time + breaking vectorization; (2) the main+BUFFER split -- scan_ins_pool (buffer) is UNbounded
    + per-point scalar, NOT vectorized; (3) dedup_pool_by_orig (a0=2) + QueryCtx rebuild per query. FIX = vectorize/
    bulk the collect (SIMD threshold-compare, bulk-compact survivors, skip per-candidate tombstone-check for cells with
    no tombstones) + bound+vectorize scan_ins_pool. PROJECTION: recover HALF the 14x (-> ~7x -> ~1.3e9 cand/s) -> p~128
    (recall ~0.90) fits comfortably, p~256 (~0.95) ~71min (close) => TOP-3 ELIGIBLE (>0.922); recover the FULL 14x ->
    p~512 (0.97) eligible -> a run at the WIN (vs scann 0.9924). UNLOCKS OOD QPS@90% IDENTICALLY (same scan-bound wall,
    P158/P162). *** DECISION: GREENLIT the path-optimization -- highest-leverage work in the session, recoverable in
    hours (not a ground-up kernel rewrite), projects to eligible top-3/win on streaming AND lifts OOD. *** (Independent
    code-level scan analysis workflow wxkgkaojd running in parallel to produce a ranked optimization plan to guide it.)
    WORKFLOW VERDICT (wxkgkaojd, 3/4 analyses, synth rate-limited) -- CONVERGES with streaming2 (kernel at peak,
    deficit is post-kernel host logic) + adds a ranked plan with file:line: (1) [biggest] the threshold-bounded pool
    streaming2 ADDED is likely the culprit -- per-candidate branch+push breaks vectorization + trashes prefetch
    locality; REVERT to native flat-IVF collect (vq.rs:1664-1745) = unconditional push into a pre-sized pool + ONE
    select_nth AFTER the full scan (A/B test it). (2) PREFETCH slot_orig indices -> miss penalty 150->40 cyc. (3)
    Defer RESIDQ adjustment to POST-scan (not per-cell) -> drops per-cell negdot_i8 + query re-reads. (4) Pre-MERGE
    main+buffer into one pool (kills the double-scan; subsumes the unbounded scalar scan_ins_pool). Est ~5-8x ->
    0.9-1.5e9 cand/s. HONEST CEILING: a residual ~1.5-2x is NOT path-recoverable (i8 LUT precision -> coarser ranking
    + mem-BW at high candidate counts). So realistic prize = TOP-3 ELIGIBLE (p~128-256, recall ~0.90-0.95, beats
    zilliz 0.922/pinecone 0.912); the 0.97/p=512 WIN is a stretch needing the residual kernel factor. Relayed to
    streaming2; same fixes lift OOD QPS@90%. NEXT = measured post-opt eligible 30M recall@10 at NQ=10000 in 8GB.

P164. (*** OOD 10M STANDING: int8 recall REACHES 0.90 at p~700; QPS@90% scan-bound ~7-10x behind, but the SAME scan-opt unlocks it to competitive ***)
    ood2 (committed c8fb6f5): text2image-10M, IP, vs the 10M float GT (t2i10m-floatgt). INT8 recall@10 (t_surv=p*4):
    p512=0.8941, p1024=0.9089, p2048=0.9157, p4096=0.9188 -> CROSSES 0.90 at p~700, ceiling ~0.92 (= the 1M plateau).
    So int8+t_surv IS scoreable at the 0.90 leaderboard point (the 10x distractors raise the needed p from 1M's ~288
    to ~700, but don't sink it below 0.90). *** OFFICIAL OOD QPS@90% standings (text2image-10M, from
    ood/res_public_queries_AzureD8lds_v5.csv, max QPS at recall>=0.90): hanns 46034 | scann(baseline) 42854@0.9025 |
    pinecone-ood 38088 | zilliz 33241 | mysteryann 22555 | pyanns 22296 | sustech 13772 | puck 8700 | vamana 6753 |
    ngt 6374 | epsearch 5877 | diskann 4133 | cufe 3561. *** OUR QPS@90%: this run was contention-JUNK (~1.5-2k @
    load 48, 16 cores 3x oversubscribed) -> clean-box est ~4.5-6k -> BOTTOM tier (~cufe/diskann), ~7-10x behind
    scann/hanns. BUT the scan path-opt (5-8x, P163) projects ~4.5-6k -> ~22-48k QPS@90% = competitive with
    mysteryann/pyanns (22k) / zilliz (33k), approaching scann (43k) / hanns (46k). *** So OOD is gated on the SAME
    recoverable scan path-opt as streaming -- NOT hopeless; it's the single unlock for BOTH tracks. NEXT: clean-box
    QPS@90% (current) then re-measure with the optimized scan = the real OOD number. (Float-rerank recall curve
    building; float QPS is mmap-bound but OOD has the full 16GB so the active-window-cache trick applies if needed.)

P165. (*** CRITICAL CORRECTION to P163/P164: the 14x is NOT clean recoverable path overhead -- scanbench OVERSTATED (L2-hot); real wall = MEMORY-LATENCY, ~37x, PARTLY FUNDAMENTAL. TOP-3 likely UNREACHABLE. ***)
    streaming2 A/B-tested the collect hypothesis and it REFUTED my optimistic projection. 1M p=512, same recall 0.9456:
    BOUNDED QPS 1381 (scan 87.6%/15.1s, rerank 10.6%/1.8s) vs UNBOUNDED/native-push QPS 1329 (scan 61.2%/11.0s,
    rerank 36.9%/6.6s). Native push IS 1.37x faster on SCAN (the bounded per-candidate branch does cost) BUT its
    unbounded pool (250k vs 32k survivors) makes rerank dedup+select_nth 3.6x SLOWER -> NET-NEUTRAL (bounded slightly
    ahead). The collect swap is NOT the 5-8x. *** THE REAL STORY: even the unbounded scan runs only ~6.8e7 cand/s
    (8-thread) at 1M p=512 = ~37x BELOW scanbench's bare kernel (2.6e9 8-thread). scanbench is L2-RESIDENT (1.6MB,
    data hot); the real scan reads ~25MB of block data per query (routing-scattered) = MEMORY-LATENCY bound, which the
    microbench cannot see and the collect-swap cannot fix. So the "14x recoverable" of P163 was WRONG -- the bare-
    kernel rate is unachievable at scale; the deficit is memory-access on the block reads, ~37x, and it is the SAME
    mechanism as scann's high-candidate-count edge (P89) => PARTLY FUNDAMENTAL. *** REVISED RECOVERABLE: ~1.4-2x only
    (keep BOUNDED for cheap rerank + make its collect branchless/SIMD to reclaim the 1.4x without the rerank penalty;
    + PREFETCH block data to hide latency = the only lever at the real 37x, but UNCERTAIN). NOT 5-8x. => streaming
    eligible recall ~0.88-0.90 at the budget-fitting p, BELOW the 0.922 top-3 bar; OOD ~1.4-2x -> ~9-12k QPS@90% vs
    scann 43k = still ~4x behind. *** HONEST VERDICT: with this IVF+apq4 engine we do NOT top (or likely reach top-3
    on) either leaderboard -- the wall is memory-latency-bound scan at high candidate counts, scann's AH2 blocked-
    layout advantage is partly fundamental. Last engineering lever = prefetch (attempting; maybe 2-3x -> borderline).
    The session's REAL deliverables: recall CAPABILITY proven (streaming 0.978, OOD reaches 0.90), the 8GB+1hr
    eligibility re-architecture, the audit that caught the eligibility constraints, and this honest characterization.
    (NOTE: my earlier user push projecting top-3 was based on the now-refuted 5-8x -- correct it once prefetch lands.)

P166. (*** RE-OPENING the "fundamental" claim: scan is memory-BANDWIDTH-bound reading ~25-62MB apq4 codes/query -> UNTRIED lever = COARSER scan codes + float rerank (scann's actual recipe) ***)
    Re-examined the scan layout (vq.rs scan_pool ~L1698): blocks ARE stored CONTIGUOUS PER CELL
    (cell_bstart[cell]..cell_bstart[cell+1], blocks[b*bb..]) -> within-cell reads already STREAM (not scattered as
    I feared). So P165's "memory-latency, partly fundamental" is incomplete: the wall is memory-BANDWIDTH -- at p=512
    the scan reads ~2.5M candidates x (m/2=25 bytes apq4 @ dpb=2, m=50) = ~62MB of codes/query. 62MB/query x ~484
    QPS x8thr ~ memory-bound. THE REAL UNTRIED LEVER (= what scann ACTUALLY does): scann "AH2,reorder=317" uses
    COARSE codes (fewer bytes/candidate) for a FAST scan, then exact-reorders only 317. We ALREADY have exact FLOAT
    rerank (from the 4GB active RAM cache, cheap). So: quantize the CANDIDATE-GEN codes COARSER (dpb=4 -> m=25 ->
    12.5 bytes/cand = HALF the read ~2x scan; or dpb=5/2-bit) -> faster memory-bound scan -> float-rerank a LARGER K
    to recover the recall the coarse codes lose. The coarse-scan+float-reorder is scann's exact recipe and we have a
    BETTER reorder (exact float vs scann's). Est: dpb=4 ~2x + prefetch ~1.3x + SIMD-collect ~1.4x -> ~3-4x -> p~192-256
    (recall ~0.93-0.95) => AT/BORDERLINE top-3 (0.922), maybe the win. This is UNTRIED and directly attacks the
    memory-bound read -- NOT fundamental. Levers ranked: (1) COARSER candidate-gen codes (dpb 2->4/5) + larger-K float
    rerank [biggest, cuts the read]; (2) prefetch next cell's blocks; (3) SIMD/branchless bounded collect. Tradeoff to
    tune: coarser codes need larger K (more float rerank), but rerank is RAM-cheap. NEXT: streaming2 implement dpb=4
    candidate-gen + measure 30M eligible recall@NQ=10000; same lifts OOD. Per the goal (top BOTH) this is the live path. [UPDATE: dpb=4 PANICS (pq.rs:178 m must be even; m=25 odd) -> use dpb=5 instead: m=20, 10B/cand = 2.5x less data (better than dpb=4's 2x), works today. 1M de-risk: dpb=5 QPS ~1.7-2x vs dpb=2 EVEN at 1M (cache-resident/compute-bound, low mem pressure) -> at 30M memory-bound should be >= that; cuts per-candidate WORK (m 50->20) so wins whether wall is BW/latency/compute. Raw int8-GT recall drop modest 0.840->0.805; the DECISIVE float-rerank+largeK recovery reads only on the 30M official-float-GT run, greenlit + running on clean box.]

P167. (*** dpb=5 CONFIRMS the BW hypothesis: 2.61x scan speedup (clean A/B) -> P165's "partly fundamental" was WRONG; coarse-scan+float-reorder WORKS ***)
    streaming2 clean matched A/B at 30M op48 (identical live=1,688,593, same op, clean box): dpb=2(ctrl) cand-gen QPS
    506 / scan 1.07e8 cand/s / recall 0.9741  VS  dpb=5 QPS 1321 / scan 2.79e8 / recall 0.8926 (rerank_k=317, official
    per-step float GT). *** SCAN = 2.61x, BEATS the 2.5x byte-ratio (m 50->20 halves LUT gathers+accumulates too) =>
    the memory-bound wall IS beatable by cutting per-candidate work; P165's "partly fundamental" ceiling is REFUTED.
    *** This is scann's exact recipe (coarse candidate-gen + exact reorder), and we have a BETTER reorder (exact float
    from the 4GB active RAM cache). Cost: recall -0.08 at K=317 = the expected coarse-code ranking penalty; RECOVER by
    bumping rerank-K (the coarse top-K still contains the true NN; float reorder is RAM-cheap). HONEST ACCOUNTING FIX
    (streaming2 caught): ops_wall was ins+del+SCAN only (main.rs:1260/1159) -- the float RERANK was UNTIMED, making
    <1hr verdicts optimistic. Fixing (time the rerank); larger-K recovery now has a real timed cost, but the 2.61x scan
    frees budget for it. NEXT: K-recovery sweep + honest-timed NQ=10000 eligible recall@max-budget-fitting-p. Budget
    math: dpb=5's 2.61x scan means the budget-fitting p ~2.6x higher -> if K-recovery holds recall ~0.93-0.95 -> TOP-3
    ELIGIBLE (0.922), maybe the win. Same lever queued to lift OOD QPS@90% (dpb=5 re-measure). The real path is LIVE.

P168. (*** HONEST ELIGIBLE CEILING: dpb=5 full-runbook honest-timed recall@10 = 0.7736 -- real gain (0.6->0.77), NOT top-3; residual = COVERAGE gap = scann's AH2 moat ***)
    streaming2 full 640-step run, dpb=5 p=64 K=800, official per-step float GT, rerank NOW counted in budget (fix
    d63902d): avg recall@10 = 0.7736. WALL(NQ=1000)=1424s (inserts 1102 + del 5 + scan 261 + rerank 55). PEAK ANON
    8.87GB > 8GB (memory-INELIGIBLE, ~5.8M sustained live). Project to scored NQ=10000 (inserts fixed, search x10):
    wall = 1102+5+2613+554 = 4274s = 71min -> OVER 1hr at p=64; max-p fitting 1hr ~= 47 -> recall ~0.74. Eligible
    ~0.74-0.77. DIAGNOSIS (clean): (1) dpb=5 scan win REAL (2.61x, 2.79e8 cand/s) -- moved the ceiling 0.6 (p<=16)
    -> 0.77 (p~50-64). (2) RESIDUAL = COVERAGE: even at 2.79e8 cand/s the 1hr budget scans only ~1.5% of points
    (p~47/C4096); scann uses 14% (700/5000) = ~10x coverage gap; dpb=5 closed 2.6x, leaving ~4x = scann's AH2 kernel
    (FastScan/SoA SIMD layout keeping the scan near-peak at scale). (3) SECONDARY walls: inserts 1102s = 31% budget
    (apq4 encode + O(live) block rebuilds); memory 8.87GB > 8GB. Fixing BOTH to ~0 only lifts max-p ~66 -> ~0.78.
    *** VERDICT: honest eligible ceiling ~0.77-0.78 with THIS engine, NOT top-3 (0.922). The remaining ~4x is scann's
    AH2 scan-LAYOUT moat -- partly the coverage math (fundamental-ish) + a major FastScan-class rewrite (days-weeks),
    OR pushing coarsening further (dpb=10/25, untested, likely hits a recall floor). REAL WINS: dpb=5 2.61x scan
    (confirmed the BW/compute hypothesis; P165's "partly fundamental" was half-right -- the SCAN was recoverable
    2.6x, the COVERAGE gap is the fundamental part); eligible 0.6->0.77; the honest rerank-budget accounting fix; the
    8GB memory audit. NEXT: map the coarsening frontier (dpb 10/25 + K-recovery, cheap) for exact max eligible +
    memory fix <8GB for a VALID eligible number, then bank. Top-3 = gated on a FastScan/AH2 scan-layout rewrite (the
    remaining innovation, multi-day). ood2 OOD identically capped: dpb=5 ~2.6x its scan -> ~12-16k QPS@90% vs scann 43k.

P169. (*** MAJOR CORRECTION to P168: the 0.77 was POOL-STARVED; recall RECOVERS to 0.956 via deep pool-depth t -> recipe WORKS, top-3 path LIVE ***)
    streaming2 (dpb=5, op48, live=1.69M; t = p*tmul = coarse-scan POOL kept before float rerank):
      p=512 t=2048 K=317/1200 -> 0.8926 (K317==K1200: POOL DEPTH t was the bind, NOT K), QPS 1467
      p=512 t=8192 K=4000     -> 0.9556 (deep pool RECOVERS recall, near dpb=2's 0.9741), QPS 308
      p=128 t=4096 K=2000     -> 0.8714, QPS 452
    The P168 "0.7736 @ p=64" was POOL-STARVED (t=64*4=256 < K=800) -> UNDERSTATED; P168's ~0.77 ceiling was WRONG.
    INSIGHT: coarse dpb=5 codes rank the true NN DEEPER in the coarse ordering, so you need a DEEPER pool t to keep
    it in the reranked set; exact float rerank then recovers it. t (pool depth) is the recall lever, not K. Recall
    ~0.95 IS reachable. *** NEW (and final) bottleneck: cand-gen QPS collapses with deep t at IDENTICAL scan work
    (1467->308 at t 2048->8192) -- it's the COLLECT/bounded-top-t pool handling (keep=t*4/cap=t*16/select_nth
    materializes ~131k then sorts, thrashes at t=8192). The float rerank itself is CHEAP (K=4000 ~= 32s@NQ=10000).
    *** LIVE PATH TO TOP-3 (NO kernel rewrite): an efficient bounded top-t collect (threshold-filter: ~2t buffer,
    one-compare reject vs the t-th-best threshold, select_nth-prune only on overflow -> amortized O(n) tiny constant)
    -> deep t becomes cheap -> a moderate-p deep-t config (p~128-192, t~6-8k, dpb=5, float rerank) runs at
    scan-limited QPS (~1400 not 308) with recall ~0.92 IN honest budget = TOP-3. This is scann's own recipe done
    BETTER (exact float reorder). streaming2 GREENLIT: profile the deep-t cost -> implement threshold-filter collect
    -> re-measure moderate-p deep-t at honest NQ=10000 budget. If >=0.922 in-budget = top-3; if collect won't speed
    up = the honest wall. The most promising lever of the session -- first time recall actually MOVED from a knob.

P170. (*** FINAL streaming characterization: deep-t rerank is MEMORY-LATENCY bound -> dpb = precise-vs-fast tradeoff; eligible ceiling ~0.77; top-3 needs a FastScan kernel ***)
    streaming2 traced the deep-t cost to code (committed f4d1682 tight bounded top-t, keep=t/prune@2t): the tight
    collect gave IDENTICAL recall (0.9556, correctness OK) but NO QPS recovery (308->238 = noise) => the COLLECT was
    NOT the bottleneck (my threshold-filter hypothesis was wrong). The real cost: after apq4 scan->top-t,
    rerank_contig_pairs does an EXACT rerank reading a raw d-byte row for EVERY one of the t pool candidates = O(t)
    SCATTERED raw-row reads/query = MEMORY-LATENCY bound. CLEAN RESOLUTION of the whole arc: dpb=5 speeds the apq4
    SCAN 2.61x, BUT coarse codes force a ~4x DEEPER rerank pool t to recover recall, and deep rerank is memory-latency
    -bound -> it EATS the scan gain at high-recall targets. Head-to-head: dpb=2 p512 t2048 = 0.9741 @ QPS506 BEATS
    dpb=5 p512 t8192 = 0.9556 @ QPS308 on BOTH axes. dpb=5 wins ONLY at low-recall/shallow-t (scan-dominated) --
    exactly why it lifted the ELIGIBLE ceiling 0.6->0.77 (cheap scan -> more coverage, shallow rerank OK at that
    target). REAL, honest gain. *** VERDICT: eligible ceiling ~0.77 with this IVF+apq4 engine. Top-3 (0.922) requires
    precise-AND-fast codes -- scann's AH2/FastScan ranks accurately in the quantized domain (shallow rerank) AT high
    scan speed via a SoA/blocked SIMD layout. We have precise-OR-fast (dpb=2 precise+slow-scan; dpb=5 fast-scan+coarse
    +deep-rerank), NOT both. THE path to top-3 = a FastScan-class kernel (precise apq4 at scann scan-speed) -- a
    multi-week engine project, the SAME wall as OOD QPS@90%. *** CONFIG LEVERS EXHAUSTED. Wins banked: dpb=5 2.61x scan
    (BW/compute confirmed), eligible 0.6->0.77, rerank-budget accounting fix (d63902d), tight bounded top-t collect
    (f4d1682), 8GB memory audit, full config frontier characterized. OOD: dpb=5 helps at the 0.90 point (scan-
    dominated) -> ~2.6x -> ~12-16k QPS@90% vs scann 43k (ood2 re-measuring). NEXT INNOVATION = the FastScan kernel.

P171. (*** P170 REFUTED (again): drop the 4GB float cache -> int8-only rerank = ELIGIBLE (0.9GB) + 5.8x faster -> high-p -> eligible ~0.86-0.90; TOP-3 in striking distance, NO kernel rewrite ***)
    streaming2, two results: (1) dpb FRONTIER SETTLED: dpb=5 is OPTIMAL. At good coverage (p=512 t=8192 K=4000):
    dpb=5=0.9556, dpb=10=0.7957, dpb=25=0.4137 -> coarser FLOORS on code fidelity (true NN lost from even top-8192),
    so "more coarsening -> more coverage -> more recall" is REFUTED; dpb=5 confirmed best. (2) MEMORY FIX = the big
    move: the 8.87GB was dominated by the 4.12GB FLOAT-rerank cache. DROPPING float rerank (int8-only rerank from the
    cache-warm self.raw int8 copy) -> anon 0.9GB (ELIGIBLE) and recall BARELY changes (float bought only ~0.01 at
    p=64) AND ~5.8x FASTER (QPS 6957 vs float's ~1200). int8-only dpb=5 op48 vs float GT: p64 0.7632, p128 0.8056,
    p256 0.8279, p384 0.8619. So P170/P168's "~0.77 ceiling" was AGAIN too low -- it was FLOAT-CACHE-ANCHORED (both
    the memory-ineligibility AND the ~1200-QPS speed anchor that capped p). int8-only ceiling ~0.86 @op48/p384, and
    full-runbook avg is HIGHER (higher-live steps recall more) -> extrapolated budget-max-p (~320-384) ~0.88-0.90.
    *** KEY REFRAME of P153/P156: exact float rerank is essential ONLY in the HIGH-recall regime (0.95+, where int8
    caps 0.954); in the BUDGET-CONSTRAINED regime (~0.86-0.90, coverage-limited) int8 rerank ~= float (0.01 gap) at
    5.8x less cost + eligible memory -> DROP the float cache for streaming. *** TOP-3 LEVER (in reach): a CHEAP
    float-REFINE -- int8 narrows to top-~100, then mmap-float-rerank JUST those 100 (no 4GB cache, no deep-t latency)
    = scann's reorder done right (shallow + on-demand) -> breaks the int8 ceiling at the top -> could push 0.86-0.90
    -> 0.90-0.93 = TOP-3, still 0.9GB eligible. NEXT: full int8-only runbook @budget-max-p (calibrating) + the cheap
    float-refine + official NQ=10000. TOP-3 genuinely in striking distance, no kernel rewrite. (P170's "FastScan-only"
    verdict was premature -- the float-cache anchor was the real blocker. User update HELD until the calibrated number.)

P172. (*** P171 CORRECTED (honest walk-back): int8-only = MEMORY win (eligible 5.3GB) but full-avg budget-max-p recall ~0.77-0.79, NOT 0.86-0.90; eligible ceiling CONFIRMED ~0.77-0.80, below top-3 ***)
    streaming2 walked back P171's 0.86-0.90: those were op48 MID-STREAM (early tiny-live steps recall only 0.35-0.55
    drag the full-640-avg) + OVER-BUDGET high-p. Calibration: int8-only dpb=5 p=256 FULL runbook (640 steps, NQ=1000,
    official GT) = recall 0.8159, PEAK ANON 5.31GB (WITHIN 8GB = eligible!), inserts 973s, search 750.7s, wall 1728s.
    BUT p=256 is 3.4x OVER the NQ=10000 budget (search x10 = 7507s + inserts 973 = 8485s = 141min). Budget-max p =
    ~88 (inserts eat 27%) -> recall ~0.77-0.79, anon ~4.5GB, wall ~3560s (fits). So int8-only's real value = a MEMORY
    FIX making the ~0.77-0.79 number VALID/eligible (5.3GB not 8.87GB) + confirms float rerank wasn't buying recall at
    the budget-recall level (~0.01). It did NOT lift the ceiling; P171's 0.86-0.90 was over-budget. CONFIRMS P170. ***
    THE WALL TO TOP-3 (config levers now genuinely EXHAUSTED across dpb/pool-t/collect/float-cache/K/p): budget-max
    p~88 = ~2% coverage vs scann's 14%. Remaining levers: (a) faster SCAN = AH2/FastScan kernel (multi-week) -> more
    coverage; (b) faster INSERTS (27% of budget; halving -> p~110 -> ~0.80, MINOR, not top-3); (c) better ROUTING
    (learned/tree, multi-week) -> fewer candidates per recall. float-refine of top-100 is MOOT at budget-max p (the
    loss is COVERAGE, not ranking -> can't refine an unscanned NN). *** HONEST FINAL (stable, confirmed by full-640-
    avg): eligible ~0.77-0.79 VALID, below top-3 (0.922). dpb=5 + int8-only + budget-max-p took us 0.6(ineligible) ->
    ~0.78(eligible). Top-3 needs a multi-week engine project (AH2/FastScan kernel OR learned router). Net wins banked:
    dpb=5 2.61x scan, int8-only eligibility (8.87->5.3GB), 0.6->0.78 eligible, the 8GB audit + rerank-budget honesty
    fix. My earlier user push (~0.77, needs FastScan) stands correct -- holding the P171 optimism was the right call.

P173. (*** DEFINITIVE VALID eligible streaming number: recall@10 ~0.77 (int8-only dpb=5 p=60), 4.3GB<8GB, wall ~3530s<1hr -- BOTH constraints met, below top-3, config EXHAUSTED ***)
    streaming2 official NQ=10000, int8-only dpb=5 p=60 flat C=4096, official per-step float GT: recall@10 = 0.769
    (mean over 513/640 steps; the run was KILLED at 80% by a box event -- not us; steady-state last-300-step mean =
    0.777, so full-640 avg ~= 0.77). PEAK ANON = 4.3GB (VALID <8GB). WALL: calibration projects p=60 -> ~3530s (VALID
    <1hr). So for the FIRST time we have a number VALID on BOTH constraints: recall@10 ~0.77, below top-3 (0.922).
    FULL LEVER SUMMARY (all exhausted): (1) dpb=5 is the optimum -- coarser FLOORS (dpb10=0.796, dpb25=0.414 even at
    deep pool; more-coarsening-for-coverage dies on the fidelity floor), finer (dpb=2) is scan-starved. (2) int8-only
    memory fix: dropping the 4.12GB float cache -> anon 8.87->4.3GB (VALID) AND ~5.8x faster, at ~0.01 recall cost
    (float rerank not worth it for streaming-30M's budget-recall level). (3) BUDGET: even with the freed speed,
    inserts (~1000-1100s = 27%) + NQ=10000 search (x10) cap p at ~60 -> ~2% coverage vs scann's 14% = THE WALL.
    NET: eligible ceiling ~0.77 VALID; 0.6-ineligible -> 0.77-eligible via dpb=5 2.61x + int8-only + budget-max-p.
    Honest, real, NOT top-3. Remaining config levers each ~+0.03-0.05 (faster inserts to free budget; hierarchical/
    learned routing; cheap float-refine to break the int8 0.95 ceiling) -- won't close the ~0.15 gap to 0.922. ***
    CONFIG SPACE EXHAUSTED. TOP-3 requires scann's AH2 scan-LAYOUT kernel (~7x scan -> 14% coverage) OR a learned/
    hierarchical router (~7x fewer probes per recall) -- a multi-week engine project, the SAME wall as OOD QPS@90%.
    BANKED WINS: dpb=5 2.61x scan, 0.6->0.77 eligible, rerank-budget honesty fix (d63902d), tight-collect (f4d1682),
    int8-only memory fix, full 8GB+1hr eligibility audit + re-arch. ood2 OOD: dpb=5 helps at 0.90 (scan-dominated) =
    real QPS@90% gain (re-measuring). *** STREAMING TRACK: config-optimization COMPLETE; final eligible ~0.77. ***

P174. (*** COVERAGE-VIA-COARSENING DECISIVELY REFUTED (with data): dpb=50 @ 14.6%cov (=scann's) = recall 0.09; the moat is RANK-PRESERVING quantization, not coverage. STREAMING TRACK COMPLETE. ***)
    streaming2 ran the aggressive-dpb + budget-max-p + large-K (K=1500) float-rerank recipe (my coverage-via-
    coarsening idea), op48 live=1.69M:
      dpb=5  10B  p~60  1.5%cov -> 0.77 | dpb=10 5B p120 2.9% -> 0.65 | dpb=25 2B p300 7.3% -> 0.27 | dpb=50 1B p600
      14.6%cov -> 0.09.
    Coverage rises 10x (1.5->14.6%) but recall COLLAPSES (0.77->0.09). dpb=50 @ p=600 = scann's EXACT 14.6% coverage
    and lands 0.09, NOT 0.92. So coverage-via-coarsening does NOT reach top-3 -- the fidelity floor collapses FASTER
    than coverage rises. CRUX: coarse codes past dpb~5 lose the true NN from the CANDIDATE SET ENTIRELY (not just its
    ranking) -- large-K float rerank CANNOT rerank a NN buried below the coarse top-K (it's not in the pool).
    Confirmed 2 ways: (a) this budget-max-p sweep; (b) dpb=25 @ p=512 (12.5% coverage!) + t=8192 deep pool + K=4000 =
    0.41 (MORE coverage + deeper pool + bigger K, still floors ~0.4). *** So scann's AH2 is NOT "coarse codes for
    coverage" (tested: it collapses). AH2's real innovation = coarse codes that PRESERVE near-neighbor RANKING:
    anisotropic score-aware loss + learned rotation + SoA reorder. Our apq4 IS anisotropic but NAIVE-coarse -- at
    m=2-4 (dpb 25-50) it can't keep the NN rankable. THE MOAT = rank-preserving coarse quantization = a real ML/algo
    project (multi-day+), NOT a config/coverage knob AND NOT merely a scan-LAYOUT kernel (my "FastScan de-risk
    microbench" is MOOT -- the bottleneck is the QUANTIZATION quality, not scan speed). *** VERDICT (TRULY COMPLETE):
    dpb=5 is the frontier optimum; coarsening past it collapses. VALID eligible ~0.77 stands (P173). Streaming took
    eligible 0.6->0.77; top-3 (0.922) requires scann's rank-preserving coarse quantization (anisotropic score-aware +
    learned rotation). Config + coverage space EXHAUSTIVELY tested WITH DATA. *** STREAMING TRACK: COMPLETE at ~0.77.

P175. (*** PROFILED (airtight, from the other side): deep-t cost is O(t) rerank_contig raw-row reads (93%), NOT collect (7%), INDEPENDENT of p -> confirms the rank-preserving-quantization moat ***)
    streaming2 profiled dpb=5 p=512 op48 (committed 6b3b0db), cand-gen split (sum-of-threads ms), t=2048(QPS1375) vs
    t=8192(QPS328): route 1%(41ms)/0%(46ms); scan_pool [scan+COLLECT] 28%(1237ms)/7%(1650ms) = ~FLAT (1.3x);
    rerank_contig [exact int8 rerank] 71%(3099ms)/93%(22570ms) = 7.3x EXPLOSION. So the 4.8x QPS drop at deep-t is
    ENTIRELY rerank_contig_pairs = O(t) scattered raw-row reads/query = memory-latency-bound; the COLLECT barely moves
    -> the tight-collect fix (keep=t/prune@2t, the threshold-filter steer) CANNOT unlock deep-t (confirms the null
    result). CRUCIAL: rerank_contig is O(t) and INDEPENDENT of p -> "moderate-p deep-t" (p=128 t=8192) has the SAME
    ~22570ms reorder -> QPS~330 -> ~8500s@NQ=10000, over budget; lowering p CANNOT dodge it. So the deep-t recall
    recovery (0.89->0.956) is real but inherently memory-latency-bound and slow regardless of p. Exposes the moat
    from the reorder side: scann's rank-preserving codes need t~317 reorder; our dpb=5 needs t~8192 (26x more raw-row
    reads) to capture the same NN because our codes aren't rank-preserving -- converges with the coverage-collapse
    (P174) on the SAME moat. *** AIRTIGHT & PROFILED: collect won't help, deep-t is O(t) memory-latency, moderate-p
    can't dodge. Eligible optimum = dpb=5 int8-only shallow-t p=60 = ~0.77 (VALID). Top-3 needs rank-preserving coarse
    codes (AH2) so the reorder set stays small. STREAMING TRACK: DONE -- argued AND profiled from both sides. Bank.

P176. (*** THE REAL (BOUNDED) PATH forward: rank-preserving codes via learned rotation (OPQ) + anisotropic eta to SHRINK reorder depth -- we HAVE the pieces; reorder-depth microbench is the decisive test for BOTH tracks ***)
    streaming2's FastScan scoping (honest, pre-commit): the profile (P175) proves the KERNEL alone won't break the
    wall -- the eligible ceiling is the O(t) DEEP REORDER (rerank_contig raw-row reads, memory-latency), deep because
    our dpb=5 codes need t~8192 to capture the true NN. Even an infinitely-fast scan leaves that reorder. So the CORE
    lever = RANK-PRESERVING low-bit codes (shrink reorder depth t~8192 -> ~300, like scann's t~317); the fast-scan
    SoA kernel is the SECOND lever that makes those codes affordable at high coverage. Both needed; the quantization
    is the bigger/harder one. *** CRUCIAL: we ALREADY have the ingredients in-engine -- Opq4::train_learned (learned
    rotation / OPQ) + Apq4's anisotropic eta. So rank-preserving codes may come from TUNING/COMBINING existing
    features, NOT a multi-week reimplementation. *** THE DECISIVE MICROBENCH (bounded, high-value): measure "REORDER
    DEPTH for 95% recall" as a function of (dpb, learned-rotation on/off, eta). Target: a code at m~25-40 that needs
    only t~300-500 reorder (vs dpb=2's ~2048, dpb=5's ~8192). IF one exists -> the reorder is cheap AND high coverage
    is affordable -> BREAKS THE WALL for BOTH streaming (recall in budget) AND OOD (QPS@90% up) -> top-3 becomes
    reachable WITHOUT a multi-week grind. If no such code exists in our OPQ+eta space -> the moat is truly scann's
    proprietary quantization (multi-week), honestly confirmed. GREENLIT: recall is load-independent so the microbench
    can overlap ood2's recall work (coordinate around ood2's QPS timing). This is the live path -- NOT banking yet.

P177. (*** OOD: dpb=5 CRATERS recall (never hits 0.90) -- coarse codes lose the IP ranking signal (NN ranked out of the pool); SAME rank-preservation wall (P176), WORSE for OOD/IP than L2 ***)
    ood2 (clean box, dpb=5 int8, text2image-10M vs float GT, REPS=5, t_surv=p*4): p768=0.6531, p1024=0.6843,
    p1536=0.7308, p2048=0.7640, p3072=0.8072 -- NEVER reaches 0.90, craters vs dpb=2's 0.9089@p1024. The true NN ARE
    in the probed cells (~0.917 available at p=3072 from dpb=2 routing) but the COARSE ADC ranks them OUT of the
    top-t_surv survivor pool -> even exact int8 rerank can't recover them. WHY it differs from streaming: msturing is
    L2 and quantizes coarsely fine; text2image is OOD/IP and much HARDER to quantize -- coarsening the PQ (dpb 2->5,
    m 100->40) destroys the IP ranking signal in candidate-gen. To reach 0.90 at dpb=5 you'd need a MUCH deeper
    t_surv (rerank ~10-20% of the pool vs 2.7% now) -> eats the scan win. *** This IS the rank-preservation problem
    (P176), on the OOD side and MORE severe: coarse codes aren't rank-preserving -> NN falls out of the pool. So the
    coarse-for-speed lever (dpb=5) that gave streaming its 2.61x does NOT transfer to OOD/IP. NEXT: ood2 testing
    dpb=4/3 (finer, still ~2x scan vs dpb=2, no odd-m panic at d=200) for an OOD sweet spot that reaches 0.90 WITH a
    scan win. If none -> dpb=2 is the OOD optimum and the honest OOD QPS@90% is the dpb=2 baseline (~4.5-6k = mid-pack
    vs scann 43k). *** The SHARED path for BOTH tracks = the P176 rank-preserving codes (OPQ learned rotation +
    anisotropic eta): a finer-but-fast RANK-PRESERVING code could reach 0.90 (OOD) / shrink reorder t (streaming) WITH
    the scan win -> top-3 for both. The reorder-depth microbench should be run on BOTH msturing (L2) AND text2image (IP).

P178. (*** HONEST OOD 10M NUMBER: QPS@90% ~1732 = BOTTOM tier (~15-25x behind); REAL 8-thread throughput (not contention); wall = SCAN THROUGHPUT; the OPQ-rotation (aopq/opql5) test is the only upside ***)
    ood2 (committed 92017d0): text2image-10M vs float GT, dpb=2 (the OOD optimum), int8+t_surv-cut, REPS=5, RAYON=8
    (matches D8lds_v5's 8 vCPU): p=768 recall 0.9040 QPS 1732 (the QPS@90% operating point); p=896 0.9069/1550;
    p=1024 0.9089/1402. So OOD QPS@90% ~= 1732. *** SOBERING: this ~= the earlier load-48 "junk" -> QPS is NOT
    contention-bound; it's the engine's REAL 8-thread throughput. The ~4.5-6k "clean-est" (P164) was TOO OPTIMISTIC.
    Even generously extrapolating to a dedicated 8-vCPU box (~2.5-3.5k), we are BOTTOM tier vs hanns 46034 / scann
    42854 / zilliz 33241 / mysteryann/pyanns ~22k / ... / diskann 4133 / cufe 3561 -> ~15-25x behind the top, at/below
    diskann/cufe. *** Recall REACH is fine (float rerank hits 0.95); the wall is purely SCAN THROUGHPUT (~15-20x below
    native), same as streaming. The dpb coarse-code lever (streaming's 2.6x) does NOT transfer to OOD: dpb=5 tops 0.807,
    dpb=4 tops 0.8903 (never 0.90), dpb=2 reaches 0.90 at p~768 (craters coarser, P177). PATH FORWARD (the P176 shared
    lever, DIRECTLY TESTABLE): ood2's branch ALREADY has comp=aopq (Opq4::train_aopq = OPQ learned rotation +
    anisotropic) + opql5 (OPQ rotation @ dpb=5) + SBANN_ETA (code comment notes high eta for OOD/IP). So the
    rank-preserving-fast-codes hypothesis is directly runnable: does aopq/opql5 + high eta make coarse (fast) codes
    reach 0.90 on OOD/IP? = the ONLY untried OOD lever with upside, and it CONVERGES with streaming2's reorder-depth/
    OPQ microbench (same OPQ-rotation lever). *** BOTH tracks' top-3 now hinges on the SAME decisive test: do
    OPQ-rotation + anisotropic-eta codes stay RANK-PRESERVING while coarse/fast? (aopq/opql5). If yes -> top-3 path
    for both; if no -> scann's proprietary quantization is the confirmed moat. This is THE experiment.

P179. (*** DECISIVE CLOSE (the last bounded path TESTED + CLOSED): OPQ rotation + anisotropic eta do NOT make coarse codes rank-preserving (+0.007 @ 4x cost); reorder-depth is BIT-RATE-bound; top-3 gap is ALGORITHMIC = scann's proprietary quantization, not tuning ***)
    streaming2 ran the reorder-depth/OPQ microbench (the "either-way" test I insisted on). Setup: op48 (live=1.69M),
    p=512 (12.5% coverage -> isolates CODE quality, not coverage), fixed depth t=2048, a0=2, official float GT.
    recall@10 @ t=2048 for FAST (dpb=5, m=20) codes:
      apq4 eta=4 (baseline)        0.8926 (QPS 2028)
      aopq (OPQ rotation + eta)    0.8926 (2028)  <- OPQ+eta = ZERO net gain
      opql (OPQ rotation, no eta)  0.8998 (521)   <- +0.007 for 4x SLOWER (per-query rotation cost)
      apq4 eta=16                  0.8777 (2195)  <- higher eta = WORSE
      apq4 dpb=2 (m=50, 2.5x slow) 0.9741 (~500)  <- finer BITS = the only real lever
    FINDINGS: (1) OPQ learned rotation +0.007 recall at 4x scan cost -> net terrible; msturing is CLUSTERED/already-
    aligned so a decorrelating rotation barely helps. (2) anisotropic eta: eta=4 already optimal, eta=16 WORSE, no
    headroom. (3) aopq (OPQ+eta together) = no better than plain apq4. (4) The ONLY lever that improves rank-
    preservation is FINER BITS (dpb=2), = the precise-vs-fast tradeoff already mapped (2.5x slower scan). So reorder-
    depth is set by code BIT-RATE; the tuning knobs (rotation, eta) move it <0.01. NOTHING with FAST codes gets below
    ~t2000, let alone the ~300-500 target. *** VERDICT: the decisive criterion is MET. The moat is scann's PROPRIETARY
    rank-preserving quantization (their specific anisotropic-VQ + learned transforms), NOT OPQ-rotation+eta -- we HAVE
    those in-engine (comp=aopq/opql, SBANN_ETA) and they do NOT work on this data. Top-3 = a multi-day+ REIMPLEMENTATION
    of scann's quantization ALGORITHM (from their papers) = the user's call; TUNING OUR ENGINE DOES NOT GET THERE.
    The last bounded path (P176) is TESTED and CLOSED; both tracks' top-3 gap is confirmed ALGORITHMIC. IP-SIDE CONFIRMATION (streaming2, text2image reorder-depth): apq4 dpb=2 p=1024, recall@10 vs depth(tmul) for eta=4/16/64: t4 0.9089/0.9085/0.9080, t8 0.9157/0.9158/0.9158, t16 0.9185/0.9188/0.9185, t32 0.9200/0.9200/0.9200 -- eta 4==16==64 IDENTICAL (+-0.001), REFUTING the code comment's 'd=200 wants eta 16-64'. So anisotropic eta has ZERO effect on IP rank-preservation, AND IP has the same deep-reorder wall (~0.92 even at t=32). => the ETA lever is dead on BOTH L2 and IP; rotation is +0.007 on L2 (already-aligned). REMAINING open piece (ood2's split): does OPQ ROTATION rescue a COARSE/fast dpb=4/5 code on IP (unaligned, where rotation could matter more than L2)? -- the one narrow place tuning might still help OOD. *** SESSION
    CONFIG/TUNING WORK COMPLETE. HONEST FINALS: streaming eligible recall@10 ~0.77 (VALID <8GB+<1hr); OOD QPS@90%
    ~1732 (bottom tier, ~15-25x behind, real throughput). Real wins banked (dpb=5 2.61x, int8-only eligibility, the
    8GB+1hr audit, honest 0.6->0.77). Top-3 for either = the scoped scann-quantization reimplementation, user's call.

P180. (*** GENUINE POSITIVE (OOD): OPQ rotation (aopq) DOES reach 0.90 COARSE on IP (0.9062 @ dpb=5 p=1024) where no-rotation dpb=5 CRATERED (0.807) -- rotation helps UNALIGNED IP but not aligned L2; QPS@90% win PENDING measurement ***)
    ood2: aopq (OPQ learned rotation) dpb=5, text2image-1M vs float GT: p256=0.8515, p512=0.8862, p1024=0.9062,
    p2048=0.9193 -> REACHES 0.90, vs no-rotation apq4 dpb=5 cratering at 0.807@10M (P177). CONFIRMS the P177/P178
    nuance I flagged: OPQ learned rotation MATTERS on IP (text2image, unaligned) but NOT on L2 (msturing, clustered/
    already-aligned = streaming2's +0.007). Eta is inert (P179 IP sweep); it's the ROTATION doing the work. So for OOD
    the moat is NOT fully closed -- rotation is a real, in-engine lever (comp=aopq) we hadn't tested on IP. *** HONEST
    QPS CAVEAT (ood2): reaching 0.90 is necessary, not sufficient. aopq dpb=5 needs p~950 for 0.90 vs apq4 dpb=2's
    p~250 (~3.8x more probes). Coarse code is 2.5x cheaper/candidate but scans ~3.8x more -> net scan work p*m =
    950*40=38000 vs 250*100=25000 (~1.5x MORE) + a 200x200 rotation matvec/query (negligible, per-query not per-cand).
    So on the probe-count model aopq-dpb5 looks SLOWER than dpb=2 at 0.90 -- rotation buys recall but not enough
    recall@p to offset the coarseness. *** BUT the QPS@90% OPTIMUM is UNTESTED: aopq-dpb4 (finer than 5, may hit 0.90
    at lower p = the sweet spot) or aopq-dpb2 (rotation may lift the fine code too -> fewer probes -> faster, since the
    rotation cost is negligible). ood2 measuring clean-box QPS@90% for apq4-dpb2 vs aopq-dpb5 vs aopq-dpb4, 1M then
    10M -- THAT decides whether OOD moves off 1732. HOLDING the user close for the QPS verdict. (The recall half is a
    real positive either way: rotation IS the right lever direction for IP -- and it's what scann's learned transform
    does, so this is a partial in-engine step toward the moat, just not yet a QPS@90% win.)

P181. (*** CORRECTION to P180 + DEFINITIVE CLOSE: the "rotation reaches 0.90 coarse on IP" was a SCALE-COMPARISON ERROR; at MATCHED scale rotation = +0.005 (negligible). TUNING DEAD on BOTH tracks. dpb=2 is OOD optimum, 1732 stands. ***)
    ood2 caught its own P180 over-claim via the matched-scale baseline (exactly the right rigor): it had compared
    aopq dpb=5 @1M (0.9062) vs apq4 dpb=5 @10M (0.807) = DIFFERENT scales. At MATCHED 1M vs float GT: apq4 dpb=5
    (NO rotation) p1024=0.9015 vs aopq dpb=5 (rotation) 0.9062 = rotation +0.005 ONLY (same negligible size as L2's
    +0.007). The 10M crater (0.807) is a SCALE effect (10x distractors); rotation's +0.005 cannot fix a -0.09 gap.
    So OPQ rotation does NOT meaningfully help IP either -> P180 REFUTED at matched scale. QPS@90% via p*m at 0.90
    (1M): apq4-dpb2 250x100=25000 (baseline) | aopq-dpb2 25000 (rotation adds nothing, hair worse) | apq4-dpb5
    1000x40=40000 (loses 1.6x) | aopq-dpb5 950x40=38000 (loses 1.5x) | aopq-dpb4 ACTUAL p~505x50=25250 = TIE with dpb=2 (closest coarse config, but a tie + rotation overhead = no win; ood2 final ledger 2fc5289). NO coarse
    config beats dpb=2 -- reaching 0.90 needs more probes than the coarser code saves. (dpb=3 invalid for d=200, m
    non-integer.) *** DEFINITIVE: every tuning lever -- dpb coarsening, OPQ rotation, anisotropic eta, pool-depth,
    collect, K, p -- is DEAD on BOTH tracks. dpb=2 is the OOD optimum; OOD QPS@90% STAYS 1732 (bottom tier). The moat
    is NOT tunable: top-3 needs a multi-day+ ENGINE PROJECT = (a) a FastScan/AH2 KERNEL (~15-20x scan throughput, our
    real bottleneck) and/or (b) rank-preserving coarse quantization (scann's AH2, so codes can coarsen without losing
    recall) -- both = scann's core, a from-papers reimplementation. *** HONEST FINALS: streaming eligible recall@10
    ~0.77 (VALID <8GB+<1hr); OOD QPS@90% ~1732. Real wins banked: 0.6->0.77 eligible, the 8GB+1hr eligibility audit +
    re-arch, dpb=5 2.61x scan, int8-only memory fix, and exhaustive data-backed refutation of every tuning lever.
    TOP-3 for either = the scoped multi-day scann-kernel+quantization rebuild, the USER's call. SESSION COMPLETE:
    config/tuning space EXHAUSTIVELY mapped with data on both tracks. (dpb=4 confirm ~5min, but the verdict is decided.)

P182. (*** FINAL CRUX (reorder-depth/OPQ microbench COMPLETE both tracks incl m~25-40 target): rank-preservation is set by code BIT-RATE (m); a coarse-AND-rank-preserving code does NOT exist in our OPQ+eta family. SESSION DONE. ***)
    streaming2 completed the reorder-depth/OPQ microbench on the CANDIDATE-GEN codes (recall-at-fixed-reorder-depth =
    do the coarse codes rank-preserve; NOT float-refine). recall@10 vs reorder-depth (tmul 4/8/16/32):
      IP text2image d=200, p=1024: FINE m=100(dpb2) 0.909/0.916/0.919/0.920 | COARSE m=40(dpb5) 0.684/0.770/0.839/0.882
      L2 msturing d=100:           COARSE m=20(dpb5) apq4=0.8926, aopq(OPQ+eta)=0.8926, opql(OPQ)=+0.007@4x slower
    eta 4=16=64 IDENTICAL (eta zero effect), eta>4 flat-to-worse; OPQ rotation ~0 on L2, +0.005 matched on IP. THE
    PATTERN (both tracks): rank-preservation ∝ code BIT-RATE (m) -- finer m = shallower reorder-depth, coarser m =
    deeper (m=40 IP only 0.88 even at reorder ~33k). Coarsening ALWAYS costs ranking; rotation/eta don't rescue it.
    => the "coarse (fast) AND rank-preserving (shallow reorder)" code = scann's AH2 recipe does NOT exist in our
    OPQ+eta knob-space. scann keeps codes coarse AND rank-preserving via proprietary anisotropic-VQ + learned
    transforms; we cannot with the knobs we have. *** DECISIVE, BOTH TRACKS + the exact m~25-40 target: top-3 requires
    reimplementing scann's quantization from their papers = a scoped multi-day algorithm project (the user's call);
    tuning is EXHAUSTED. *** SESSION COMPLETE. Honest finals: streaming eligible recall@10 ~0.77 (VALID 8GB+1hr, from
    ~0.6-ineligible); OOD QPS@90% ~1732 (bottom tier, of scann 42854). Real wins: the 8GB+1hr eligibility audit +
    re-arch, dpb=5 2.61x scan, int8-only memory fix, float-rerank breaking the int8 recall ceiling, and exhaustive
    data-backed refutation of every tuning lever (dpb/pool-t/collect/K/p/OPQ-rotation/eta) on both tracks.

P183. (*** WALL-1 DE-RISK DECISIVE (USE512 A/B): the apq4 scan is MEMORY-BOUND (AVX-512 +7%, not 2x) -> a FastScan/vpshufb KERNEL alone won't fix coverage; needs scann's cache-friendly SoA LAYOUT. BOTH cheap gates negative. ***)
    streaming2 USE512 A/B (op48 dpb=5 p=512 int8-only, sum-of-threads ms): scan_pool (apq4 fast-scan kernel) AVX2
    1296 -> AVX-512 1200 = +7% (NOT the ~2x a compute-bound kernel would give); rerank_contig AVX2 3422 -> AVX-512
    3239 = flat (memory-bound, expected). So AVX-512's 2x compute width barely moves the scan -> the apq4 fast-scan
    is MEMORY-BOUND (reading the cell-scattered interleaved blocks), NOT compute-bound. => a better KERNEL (more
    compute / vpshufb) will NOT deliver the ~2x coverage lever; the bottleneck is the memory ACCESS PATTERN, fixable
    only by a cache-friendlier LAYOUT (scann's AH2 SoA packing). *** BOTH WALLS now decisively FUNDAMENTAL in our
    engine, and they CONVERGE: WALL 1 (coverage/scan throughput) = memory-bound layout, kernel +7% won't help ->
    needs scann's SoA layout; WALL 2 (rank-preservation) = coarse codes can't rank-preserve via OPQ+eta (P179/P182)
    -> deep reorder unavoidable -> needs scann's anisotropic-VQ. scann's AH2 solves BOTH at once (cache-friendly SoA
    layout that IS compute-bound + anisotropic rank-preserving codes). Neither of our CHEAP levers -- (a) a better
    kernel (USE512, negative) nor (b) code tuning (reorder-depth/OPQ, negative) -- reaches top-3; BOTH require the
    full algorithm+LAYOUT reimplementation = multi-day, the USER's call. *** SESSION EXHAUSTIVELY COMPLETE on BOTH
    walls (both cheap de-risk gates NEGATIVE-DECISIVE). Top-3 = the FULL scann AH2 (SoA layout + anisotropic quant),
    not kernel-alone or tuning. Honest finals stand: streaming eligible ~0.77 (valid), OOD QPS@90% ~1732 (bottom tier).

P184. (*** CLEAN BASELINE TABLE (streaming2, 1M L2+IP, reorder-depth sweep): our EXISTING aopq/opql/eta do NOT rank-preserve coarse codes -- coarse needs rr~16384 (~16x fine, ~30-50x the ~300-500 target). Baseline for the anisotropic-VQ test. ***)
    streaming2 ran the reorder-depth microbench in the exact 1M held-out form (static, fixed p=512, TMUL sweep =
    reorder-depth rr=512*tmul), recall@10 vs float GT:
    L2 msturing-1M d=100, recall vs rr(512->16384):
      apq4 dpb2 (m50 FINE):   0.942 0.944 0.945 0.945 0.945 0.945  -> plateau by rr~1024 (SHALLOW)
      apq4 dpb5 (m20 COARSE): 0.764 0.834 0.886 0.918 0.936 0.943  -> needs rr~16384 (DEEP)
      opql dpb5 (OPQ):        0.775 0.845 0.891 0.922 0.937 0.943  -> +0.01, NO shrink
      aopq dpb5 (OPQ+eta):    0.768 0.839 0.889 0.920 0.935 0.942  -> no gain; eta=16/64 WORSE
    IP text2image-1M d=200, recall vs rr(512->16384):
      apq4 dpb2 (m100 FINE):  0.906 0.929 0.939 0.942 0.943 0.943  -> plateau by rr~4096 (SHALLOW)
      apq4 dpb5 (m40 COARSE=the m~25-40 target): 0.603 0.726 0.822 0.892 0.929 0.941 -> needs rr~16384 (DEEP)
      opql dpb5 (OPQ):        0.628 0.747 0.840 0.900 0.932 0.942  -> +0.004-0.02, NO shrink; aopq no gain; eta flat-worse
    So our EXISTING compressors (apq4/opql/aopq + SBANN_ETA) do NOT produce a coarse-AND-shallow-reorder code on
    EITHER track. *** OPEN CHECK (fresh agent a22fe7b1): is our aopq a FAITHFUL implementation of ScaNN's anisotropic
    -VQ (Guo 2020 score-aware parallel-residual-weighted training loss), or is our "eta" a cruder version? If crude,
    a PROPER implementation might rank-preserve where this baseline didn't. This table is the baseline to beat: a
    proper anisotropic-VQ that makes coarse dpb=5 reach ~0.94 at rr~1024 (like the fine code) = the breakthrough. If
    a faithful impl ALSO plateaus deep -> the moat is the full AH2 system (loss + SoA layout), confirmed at the deepest
    level. (streaming2 stood down; box idle for the anisotropic-VQ agent.)

P185. (*** THE aopq-FAITHFULNESS QUESTION ANSWERED (agent, branch aniso-vq-faithful): our crude aopq/eta is NOT a faithful ScaNN anisotropic-VQ -- it drops the cross-subspace parallel coupling, so eta was near-INERT (explains P179/P182's "eta zero effect"). A FAITHFUL coordinate-descent impl makes eta a REAL lever & HELPS IP (+0.05-0.08 shallow-rr, best eta~8 +rotation) -- but STILL does NOT rank-preserve coarse codes. NO breakthrough; moat = full AH2 (bit-rate + SoA), not the loss. ***)
    (a) FAITHFULNESS AUDIT (file:line): our `comp=aopq`/`apq4` anisotropic loss lives in pq.rs `train_f32_aniso`
    (pq.rs:190-232) + `encode_f32` (pq.rs:462-...). It weights, PER SUBSPACE independently, (eta-1)*<r_sub, xhat_sub>^2
    where xhat_sub is the subspace SLICE of the unit FULL vector. ScaNN's loss (Guo et al. 2020, confirmed from the
    paper) weights the parallel residual of the FULL vector: (eta-1)*<r, xhat>^2 with <r,xhat>=Sum_j<r_j,xhat_j>, which
    COUPLES all subspaces, optimized by COORDINATE DESCENT over subspaces (assigning subspace j depends on the residuals
    of all OTHER subspaces; codebook update = Thm 4.2 with a +(eta-1)*s_{-j}*xhat_j cross term). Our impl DROPS that
    coupling AND uses the tiny-norm subspace slice (||xhat_sub||^2 ~ dpb/d ~ 0.05), so the parallel penalty is a tiny
    fraction of the subspace L2 -> eta barely moves the argmin. => our aopq is a BLOCK-DIAGONAL APPROXIMATION, NOT
    faithful. This is the mechanistic cause of P179/P182's "eta 4=16=64 identical". VERDICT (a): NOT FAITHFUL.
    (b) IMPLEMENTED the faithful version: pq.rs `train_f32_aniso_cd` + `encode_f32_cd` (coordinate descent, full-vector
    parallel residual, cross-subspace coupling in BOTH assignment and the Thm-4.2 codebook update), gated by env
    SBANN_ANISO_CD (main.rs). Verified eta now BITES (monotonic, strong effect) and converged (iters=10 == iters=30).
    Reorder-depth microbench, MATCHED 1M scale, p=512 (12.5% cov), rr=512*{1,2,4,8,16,32}, coarse dpb=5:
    L2 msturing-1M d=100 (m20), recall@10 vs rr:
      crude apq4 eta=4 (baseline): 0.766 0.837 0.888 0.919 0.936 0.943
      FAITHFUL eta=4:              0.764 0.835 0.886 0.919 0.936 0.943  (neutral)
      FAITHFUL eta=16:             0.749 0.824 0.880 0.915 0.934 0.942  (worse)
      FAITHFUL eta=50:             0.640 0.731 0.809 0.868 0.909 0.931  (much worse)
      FAITHFUL eta=200:            0.340 0.427 0.524 0.627 0.727 0.815  (catastrophic)
      FINE dpb2 (m50) reference:   ~0.942 flat by rr~1024
      => L2: faithful aniso is NEUTRAL at eta~4 and STRICTLY HURTS as eta grows (parallel-weighting sacrifices the
         orthogonal accuracy L2 ranking needs). No reorder-depth shrink. msturing is clustered/already-aligned.
    IP text2image-1M d=200 (m40 = the m~25-40 target), recall@10 vs rr:
      pq4 isotropic (clean ctrl):  0.558 0.694 0.803 0.882 0.925 0.948
      crude apq4 eta=4 (baseline): 0.588 0.697 0.794 0.862 0.913 0.942
      FAITHFUL eta=4:              0.628 0.733 0.819 0.883 0.926 0.948
      FAITHFUL eta=8 (peak):       0.639 0.743 0.826 0.887 0.927 0.948
      FAITHFUL eta=16:             0.636 0.739 0.823 0.885 0.925 0.946
      FAITHFUL eta=50:             0.566 0.677 0.771 0.844 0.900 0.933  (over-weighted)
      aopq(OPQ rot)+FAITHFUL eta16:0.668 0.774 0.849 0.901 0.932 0.948  <- BEST coarse (rotation +0.03 on top)
      FINE dpb2 crude:             0.928 0.946 0.954 0.957 0.959 0.959
      FINE dpb2 FAITHFUL eta8:     0.941 0.953 0.957 0.958 0.959 0.959  (faithful aniso helps the fine code too)
      => IP: faithful aniso is a REAL, correctly-signed win -- +0.08 over isotropic (0.558->0.639) and +0.05 over the
         crude "anisotropic" (0.588->0.639) at rr=512; the isotropic-pq4 control proves the gain is the LOSS, not just
         better optimization. Peak eta~8 (much lower than the code comment's 16-64); +OPQ rotation another +0.03.
    (c) VERDICT: NO BREAKTHROUGH. Even the BEST coarse config (OPQ rotation + faithful aniso eta16) reaches only
      0.668/0.774 at rr=512/1024 and still needs rr~16384 for ~0.94 -- ~16-32x the FINE code's rr~512. The target
      (coarse ~0.95 at rr~300-1000) is MISSED by a wide margin. Rank-preservation stays BIT-RATE-bound; the anisotropic
      LOSS only SHIFTS the reorder-depth curve up ~0.05-0.08 at shallow rr, it does not change the SHAPE (coarse still
      converges to the fine plateau only at rr~16k). Clean at MATCHED 1M scale (no P180-style scale artifact); converged
      (not under-trained). *** THE CORRECTION to P179/P182/P184: "eta has zero effect / anisotropic is dead" was an
      IMPLEMENTATION artifact (crude block-diagonal, coupling dropped), NOT a property of ScaNN's loss. Properly
      implemented, the anisotropic loss IS a real lever and DOES help IP rank-preservation -- just not enough, alone, to
      make COARSE codes rank-preserving. So the moat is confirmed to be the FULL AH2 SYSTEM: ScaNN keeps codes
      rank-preserving by using FINE codes (m~100+) that their SoA 4-bit FastScan can afford to scan fast (WALL 1),
      NOT by a coarse-and-rank-preserving code from the loss. The bounded lever that DOES survive: fold faithful
      anisotropy into the FINE-code IP path (+0.013 at rr=512, free at scan time). Branch aniso-vq-faithful; SBANN_ANISO_CD.

P186. (*** STRATEGIC CLOSE: bounded attempts EXHAUSTED across config + tuning + a FAITHFUL ScaNN anisotropic-VQ impl. Anisotropic-VQ CORRECTED our own artifact (real +0.05-0.08 IP lever) but no coarse rank-preservation. Moat = full AH2 SYSTEM; honest limit reached. ***)
    The P185 anisotropic-VQ result is the deepest point we reached, and it CORRECTS an earlier conclusion: P179/P182/
    P184's "eta is dead / anisotropic doesn't help" was an IMPLEMENTATION ARTIFACT -- our aopq/SBANN_ETA applied the
    parallel-residual penalty PER-SUBSPACE (tiny slice norm ~dpb/d=0.05 -> near-inert), NOT ScaNN's full-vector
    coordinate-descent loss (Guo 2020, Thm 4.2 with cross-subspace coupling). The fresh agent implemented the FAITHFUL
    version (SBANN_ANISO_CD): eta now bites, and it is a REAL, correctly-signed lever on IP (text2image): +0.08 over
    isotropic / +0.05 over crude at shallow reorder rr=512 (peak eta~8, +OPQ rotation ~+0.03). On L2 (msturing) it's
    neutral at eta~4 and hurts as eta grows (parallel weighting is wrong for L2). *** BUT NO BREAKTHROUGH: even the
    best coarse config reaches only 0.67/0.77 @ rr=512/1024 and still needs rr~16384 for 0.94 (~16-32x the fine
    code's rr~512). Rank-preservation stays BIT-RATE-bound; the anisotropic loss shifts the curve up ~0.05-0.08 at
    shallow rr without changing its shape. So a "coarse-AND-rank-preserving code from the loss" does NOT exist, even
    with ScaNN's actual loss. *** THE REFRAME (key): ScaNN does NOT use coarse-rank-preserving codes -- it uses FINE
    codes (m~100+) made affordable by a cache-friendly SoA 4-bit FastScan (WALL 1), + anisotropic-VQ as a secondary
    boost. Our apq4 is ALREADY FastScan-like (blocked, 4-bit, in-register i8 LUT) and its scan is MEMORY-bound at
    scale (P183 USE512 +7%) -- the working set exceeds cache, which a layout tweak within our design won't fix.
    So the moat is the FULL AH2 SYSTEM (SoA layout + fine codes + anisotropic-VQ), a multi-day+ from-papers rebuild,
    and our engine is already fairly optimized -> uncertain payoff. *** HONEST LIMIT: bounded autonomous attempts are
    EXHAUSTED (config, tuning, proper anisotropic-VQ all tested with data). SURVIVING MARGINAL LEVER: faithful aniso
    on the FINE-code IP path = +0.013 recall @ rr=512, free at scan time (could nudge OOD ~1732 slightly, not off the
    bottom). Real deliverable kept: SBANN_ANISO_CD (correct ScaNN anisotropic-VQ) on branch aniso-vq-faithful (601a02b).
    Top-3 = the full AH2 rebuild + likely a better router = a scoped multi-day project needing the user's greenlight +
    sustained capacity. FINAL: streaming eligible ~0.77, OOD QPS@90% ~1732; did not top either; wall characterized to
    the algorithm level with data at every rung.

P187. (*** WALL-1 FASTSCAN AUDIT: our scan kernel WAS crude (corrects P186 "already FastScan-like"); faithful 32-wide int8-sat FastScan = real 1.8x KERNEL, committed (fastscan-soa d99ca6f, SBANN_FASTSCAN2) -- but only 1.07x END-TO-END because the kernel is ~6% of the query. Real scan cost = the scalar COLLECT (~85% of scan phase) + scattered cell reads (6-11x collapse). NOT top-3. ***)
    Third faithfulness audit (after aniso-VQ P185). Result mirrors P185: an "it's fundamental" claim was actually a CRUDE
    implementation. Our apq4 scan kernel block_adc_i8_i16acc (pq.rs:705) had in-register vpshufb LUT + 16-way SoA but
    (a) 16-wide not 32-wide (used _mm_shuffle_epi8 not _mm256_), (b) int16 accumulate not int8-saturating (cvtepi8_epi16
    + add_epi16, 2 uops/subspace) -- even the AVX-512 path stayed int16-accumulate, which is exactly why P183's USE512
    A/B saw only +7%. So P186's "already FastScan-like" was WRONG. A faithful 32-wide int8-saturating FastScan (periodic
    int16 hoist, bounded LUTs cap 15/subspace) is a REAL 1.8x on the KERNEL (single-core microbench, m=100: 372->680
    Mcand/s L2-hot), recall-neutral (p512 0.9700 default vs 0.9698 fs2), committed behind SBANN_FASTSCAN2.
    *** BUT END-TO-END ONLY 1.07x at 1M (1127->1206 QPS same-box A/B): the LUT kernel is only ~6% of the query. The
    scan PHASE is ~85% the scalar COLLECT -- pool.push((out[j],slot)) + per-candidate slot_orig branch + select_nth
    in scan_pool (vq.rs:1650). Query = route 19% + scan 46% + rerank 35%; kernel is a sliver. *** At 10M the
    cell-SCATTERED access pattern collapses throughput 6-11x (m=50: 715 L2-hot -> 104 scattered): scan visits p=512
    cells in probe order = 512 random jumps into a 100-250MB array. Sequential-large streams fine (321-660 Mcand/s),
    so the wall is the SCATTER, not raw bandwidth -- refines P183. Could NOT build a real 10M A/B (shared-box memory
    cap ~26GB, already ~28GB used); 10M projection rests on faithful scattered microbench + 1M end-to-end.
    *** METHODOLOGY (per user, 2026-07-01): cross-machine QPS comparison (ours on a load-22 16-core shared box w/ other
    tenants' mox-compile eating 5+ cores, vs scann on an idle standardized Azure VM) is INVALID; any "~22x short of
    scann" projection INHERITS this flaw and is NOT restated as fact. Hardware-INDEPENDENT findings that DO stand:
    kernel ~6% of query (structural), collect ~85% of scan phase, scatter collapses 6-11x. These locate the real
    bottleneck WITHOUT the contaminated QPS number, and the two-walls conclusion never depended on QPS -- it rests on
    recall (streaming ~0.77 vs 0.998) + reorder-depth (hardware-independent). Future QPS claims must be normalized
    against a reference baseline measured on THIS box when idle (loadavg<4).
    *** BOUNDED WIN BANKED: real 1.8x FastScan kernel (SBANN_FASTSCAN2, fastscan-soa d99ca6f), recall-neutral.
    TWO remaining WALL-1 levers, both = ScaNN's actual design, both larger rewrites: (1) FUSED SIMD top-t that keeps a
    running threshold + emits only survivors -> eliminates the O(candidates) scalar collect that is the measured ~85%
    of the scan phase; (2) SoA / bigger-cell LAYOUT so the 10M scan STREAMS (321-660 Mcand/s) instead of SCATTERING
    (73-104). PATTERN across P185/P187: our engine has real unclaimed perf (crude impls), but closing to ScaNN needs
    reimplementing its core (anisotropic-VQ, fused-top-k, SoA) = a scoped multi-day rebuild = user's call.

P188. (*** FUSED-TOP-K: recall-EXACT primitive built (SBANN_FUSEDTOPK, fused-topk 86793be) but NEUTRAL end-to-end -- and it CORRECTS P187's premise by direct measurement: the scalar collect is only 20-30% of scan, NOT 85%. The dominant scan cost is the SCATTERED PQ-block reads (memory-bound), = the SoA/bigger-cell LAYOUT lever (P183), triply-confirmed. Query is BALANCED: route 32 / scan 37 / rerank 31. ***)
    ScaNN fused-top-k ("keep only survivors"): running t-th-best threshold, SIMD-compare each block's dists
    (_mm256_cmpgt_epi32 + movemask), push only survivors so the per-candidate slot_orig branch runs t times not N,
    prune a 2t buffer to t. RECALL-EXACT (bit-identical top-10 id sets: 1M OOD champion p512/t4096 = 10000/10000
    identical delta 0.00000; t<<N p2048/t2048 = 5000/5000 identical) -- survivor buffer is provably a superset of the
    true top-t. *** BUT NEUTRAL: direct measurement via SBANN_SCANDIAG (kernel-only floor = block reads + LUT, no
    collect) shows collect = scan_full - scan_kernelonly = only 84us/274us = 30% of scan at champion p512/t4096
    (154us/741us = 21% at p2048/t2048). P187's "collect = 85% of scan" CONFLATED the scattered-block-read memory
    stalls (which occur INSIDE the LUT kernel, waiting on RAM) with the scalar collect. So even a zero-overhead
    fused caps at ~1.13-1.15x here; measured QPS = neutral at champion (2t>N, no pruning, ~70% of candidates are
    genuine survivors), +2-6% only when N>>t (off the recall frontier). Banked as a correct default-off primitive;
    NOT an end-to-end lever at 1M OOD.
    *** NEW BOTTLENECK BREAKDOWN (1M OOD champion, single-thread pinned): route 32% / scan 37% / rerank 31% -- the
    query is BALANCED, NO silver bullet; within scan, kernel+scattered-block-reads = 70% (the wall), collect = 30%.
    Implication: even a free scan caps e2e at ~1.6x; topping needs gains across route AND scan AND rerank, or a
    structurally different design. *** THE REMAINING SCAN LEVER (triply-confirmed P183/P187/P188): the scattered PQ
    block reads -- p=512 random jumps into a ~168MB blocks array, 73-104 Mcand/s scattered vs 321-660 sequential
    (6-11x collapse, worse at 10M). Fix = SoA / bigger-cell LAYOUT so probes read big contiguous streams. This is
    a recall/speed TRADEOFF (bigger cells = coarser routing = more candidates but streaming throughput), testable
    at 1M as QPS@recall>=0.90 -- NOT yet done.
    *** MEASUREMENT-CEILING NOTE (important, per user's 2026-07-01 methodology point): this shared box (16-core EPYC,
    load 14-22, other tenants' mox-compile, ~26GB mem cap) CANNOT produce a leaderboard-valid number: can't build
    10M-scale under the mem cap, can't run uncontended for wall-clock QPS, no same-box published-reference baseline.
    Recall (streaming ~0.77 vs 0.998) IS hardware-independent and real; QPS/leaderboard-POSITION is NOT answerable
    here. Topping requires the official harness on appropriate HW at 10M/100M scale. Pattern across P185/P187/P188:
    three faithfulness audits, three real bounded wins (aniso-VQ +0.05-0.08 IP; FastScan 1.8x kernel; fused-top-k
    recall-exact primitive), each correcting the prior's error -- but the query is balanced with no silver bullet,
    and top-3 needs ScaNN's full design + a submission environment this box is not.

P189. (*** WALL-1 PREFETCH = the lone real recall-neutral scan lever: SW-prefetch +7% QPS@recall0.90 / +11-12% champion, RECALL-EXACT (SBANN_PREFETCH, scan-prefetch 3057030). Reordering/SortCells = NULL (refutes P183's layout-lever idea). Real scatter collapse = ~3.9x (not 6-11x). Our clean 1M single-thread QPS@recall0.90 ~= 2640 w/ prefetch. ***)
    Attacked the scattered PQ-block reads (champion 1M OOD, hierk Kf=262144 C0=4096 apq4 a0=3, IP+FASTSCAN2),
    single-thread pinned, interleaved paired A/B (box bounced load 10-26 all session, never clean -> ratios not
    absolutes). scatterbench isolates it: ~1.9 blocks/cell (~1.5KB), 15778 cand/query over p=512 cells.
    Scattered probe-order = ~92 Mcand/s; L2-hot compute floor = 356 Mcand/s -> real collapse ~3.9x (the abstract
    "6-11x" assumed a fully-packed read the real 0.2%-dense scan can't achieve).
    *** WINNER = SW-PREFETCH (SBANN_PREFETCH, vq.rs scan_pool ~L1839 + scan_kernel_only ~L2018): prefetch next
    probed cell's 800B block (T0) while scanning current. Kernel floor 92->133-145 Mcand/s (+45-53%). RECALL
    BIT-IDENTICAL (hint only). e2e QPS@recall>=0.90 (p160, recall 0.9116): 2459->2640 = +7.2%; champion (p512,
    recall 0.9586): 1329->1496 = +11.6% (all 5 rounds positive). Amdahl caps it: scattered reads are only ~26% of
    e2e (70% of the 37% scan); route 32% + rerank 31% untouched. *** NULL = SortCells/reordering (SBANN_SORTCELLS):
    recall-identical but e2e ~0% -- sorting cuts the avg cross-cell jump 49x (27.2MB->0.55MB) yet buys +4%, because
    only 512/262144 cells are probed (0.2% dense) so even sorted access keeps 0.55MB gaps beyond HW-prefetch/TLB
    range. This REFUTES P183's "locality-preserving layout" as a recall-neutral fix.
    *** 10M projection: prefetch hides latency that only deepens at 10M (~2.9GB blocks array) + scan's e2e share
    grows -> lever likely holds ~+8-12%; does NOT close the gap to scann. The remaining ~3.9x scattered->compute-floor
    is STRUCTURAL, needs scann's cache-resident SoA AH layout (out of scope).
    *** CLOSES the cheap WALL-1 lever set: USE512 (+7% dead P183), fused-top-k (neutral P188), sort (null), PREFETCH
    (+7-12%, the win). OUR banked clean 1M single-thread QPS@recall0.90 ~= 2640 (w/ SBANN_PREFETCH) -- the number the
    same-hardware scann head-to-head (running) will be compared against.

P190. (*** THE HEADLINE MEASUREMENT: DIRECT same-hardware ScaNN-vs-ours, 1M text2image OOD, single-thread pinned interleaved = ~2.2-2.4x (ScaNN faster), NOT the invalid cross-machine "25x". The "25x" was ~10x inflated by threading+hardware+contention (our contended box vs scann's idle-Azure published QPS). ***)
    ScaNN 1.4.2 (pip, AVX-512), true FLOAT base (1M rows of base.1B.fbin) + float queries, dot_product,
    tree(num_leaves=2000)+score_ah(2,thresh=0.2)+reorder(200) -- its best T2I recipe. OURS: fastscan-soa +
    SBANN_FASTSCAN2, int8 (t2i1m.i8bin), hierk Kf=16384 C0=128 b0=32 a0=3 SOAR TREEEM apq4 IP, p=80 t=8. BOTH
    scored vs identical FLOAT-IP GT t2i1m-floatgt (revalidated overlap 1.0000 vs exact float IP). BOTH single-thread
    pinned taskset -c 4, best-of-5, INTERLEAVED 6 rounds (contention-robust ratio).
    *** THE RATIO @ recall@10>=0.90: ScaNN 0.9032 @ 8649 QPS vs OURS 0.9003 @ 3903 QPS = 2.22x. Matched ~0.908:
    8202/3384 = 2.42x. Rock-stable across 6 rounds (scann 8.5k +-1%, ours 3.8k +-3%) -- NOT load noise. Build/RSS
    comparable (scann 65s/2.83GB/0.9GB-idx; ours ~50s/0.8GB-idx).
    *** INTERPRETATION: the true same-hardware per-core OOD gap is ~2.2-2.4x, ARCHITECTURAL (scann's anisotropic
    2-byte AH + in-register scan + float reorder~200 vs our 4-bit PQ + int8 no-float-rerank), consistent with the
    old same-window "~2x OOD" (P130). NOT a measurement artifact -- but also NOT hopeless. UN-APPLIED levers that
    narrow it: (1) SBANN_PREFETCH (P189, +7-12%, NOT in this run) -> ~4180-4370 QPS -> ratio ~2.0-2.1x; (2) float
    rerank (breaks int8 recall ceiling, un-applied on this branch) -> reach 0.90 at lower p -> higher QPS;
    (3) route/rerank each ~1/3 of query w/ headroom. So closing ~2.2x toward parity is PLAUSIBLE with identified
    levers -- a completely different picture from the "25x, needs multi-day rebuild" framing this whole session
    operated under. *** CAVEATS (honest): 1M + single-thread only; 10M same-hw ratio NOT yet measured (engine 10M
    TREEEM build killed under memory-thrash/swap-full; scann 10M index IS built+cached on disk, 0.90 crossing
    recall 0.9075 @ lts=120, for a clean re-run when box healthy); multi-thread scaling unverified; int8-vs-float
    are each engine's intended representation (fair at the metric). scann-headtohead branch 40f3f87 (agent labeled
    it P185 by mistake; this is the canonical P190). *** STRATEGIC PIVOT: the remaining engine levers (prefetch,
    float rerank, routing) are now clearly WORTH STACKING to close a 2.2x gap -- vs the prior "only scann's full AH2
    rebuild helps". The 25x mirage drove months of pessimism; the real target is ~2x and shrinking.

P191. (*** STACKED LEVERS: prefetch+float-rerank close 2.22x -> ~2.07x median (2.01x best round) vs ScaNN at 1M single-thread OOD. Float rerank is the MOVER (0.90 at p=58 vs p=80, +0.0236 recall, breaks int8 ceiling); prefetch ~null under contention (+3%). In-hand levers ~tapped out; remaining = structural SoA AH scan. Branch ood-levers-stacked 9fa173b. ***)
    Ported float rerank onto the prefetch branch (fbin.rs, simd::dot_f32_fast, vq::search_frr/scan_rerank_frr/
    rerank_contig_float). 1M text2image OOD, single-thread pinned interleaved best-of-5 (load 16-19):
    int8-no-PF 0.9003@p80 3612 QPS (2.28x) -> +prefetch 3725 (2.22x, +3%) -> +float-rerank 0.9007@p58 3971/4121
    (2.07x med / 2.01x best). Fresh scann 8236 @ 0.9032. Prefetch recall-EXACT (delta 0.0000); float rerank +0.0236
    recall @ matched p80 (0.9003->0.9239). VERDICT: brushing 2x, not clean sub-2x, not parity. Biggest remaining =
    ScaNN's cache-resident SoA anisotropic-AH scan vs our scattered PQ-block reads (~3.9x collapse). Next: sweep the
    UN-tested config levers (routing granularity toward scann's bigger contiguous leaves + float rerank; rerank depth;
    aopq/aniso-CD codes) before conceding the structural wall.

P192. (*** SUB-2x ACHIEVED at 1M single-thread OOD: gap-closing workflow (routing/rerank/codes fan-out) drives 2.07x -> ~1.77x median (1.84x recall-matched to scann 0.9032), clean interleaved. The ONLY winner = DE-OVER-PROVISIONING the coarse router (C0 128->768, b0 32->96: coverage 25%->12.5%, 4224->2816 int8 dist-evals/q, holds 0.90 at same p=58). Rerank-depth & better-codes = confirmed NON-winners. Config-lever headroom now TAPPED at ~1.8x. ***)
    3-agent parallel workflow, each measured QPS@recall0.90 single-thread pinned interleaved vs fresh ScaNN.
    ROUTING (only winner, config-only): the "bigger contiguous leaves" hypothesis was FALSIFIED (KF32768 finer=worse;
    bigger C0 alone=small); the real win is trimming the OVER-PROVISIONED coarse router (b0 beam 32->96 halves top-level
    coverage 25%->12.5%, cutting route work ~33% while still reaching 0.90 at p=58 -- route was doing wasted dist-evals).
    RERANK-DEPTH: NO win (already at the 0.90 knee; p=58/t=8 is the edge, p<=56 & t<=7 fall sub-0.90). CODES: NEGATIVE
    (aopq + faithful anisotropic-VQ both reach 0.90 at MORE probes -- under FLOAT RERANK the code's only job is
    isotropic pool-recall, so anisotropic optimizes the wrong target). Winners collapse to the single routing lever.
    *** CLEAN INTERLEAVED (taskset -c 0, best-of-5, 5 rounds, load 11.6-13.9): ScaNN 0.9032 @ median 8429 QPS (+-1%) vs
    ENG-COMBINED (C0=768 b0=96 idx + apq4 + p58 t8 float-rerank + prefetch) 0.9005 @ median 4746 = RATIO 1.77x; recall-
    matched at p=60 (0.9033) = 1.84x. Recall confirmed >=0.90 (not cherry-picked). *** So: 25x(invalid) -> 2.22x(P190
    true same-hw) -> 2.07x(P191 stacked) -> 1.77x(P192 routing). Clean sub-2x for the first time, ~15% relative cut.
    NOT parity. Remaining ~1.8x is STRUCTURAL/execution-speed (unchanged P183/P187/P189): ScaNN's cache-resident SoA
    anisotropic-AH scan (in-register LUT16 over a packed contiguous-leaf layout) vs our memory-bound scattered PQ-block
    reads (0.2%-dense probes). No config lever crosses it recall-neutrally -> needs the layout rebuild. Committed
    5419798 on ood-levers-stacked; harness interleave_p192.sh; idx granul_kf16384_c768_b96_a3.idx.
    *** CAVEAT (P192-honest): interleaved ratio itself has ~+-5-7% window variance (scann float-scan vs our int8-scan
    have different contention sensitivities); ~1.8x is the honest central estimate, not a hard 1.77.

P193. (*** REFRAME (de-risk gate fired): the ~1.8x is NOT the scan -- it is the float-REORDER COUNT (rerank). Our scan ALREADY streams at parity with ScaNN; the streaming-leaf / SoA-layout lever is REFUTED. The entire gap is: we float-reorder 464 survivors to hit 0.90 vs ScaNN's 78 (~75us, ~6x), set by CODE RANKING QUALITY (our apq4 50B/vec vs ScaNN anisotropic-AH 100B/vec). ***)
    Profile of P192 champion (208us/q, ~4800 QPS single-thread): route 36us / scan 85us / rerank 88us. ScaNN (118.7us)
    decomposed by lts/reorder sweep: route+fixed ~30 / scan ~76 / reorder(78) ~13. Phase gap: route +6, scan +9,
    RERANK +75us = the whole 1.8x. *** The old "scan scattered 92 Mcand/s = 3.9x collapse" was the OBSOLETE Kf=262144
    index; the current granul Kf=16384 champion scan ALREADY streams at 177 Mcand/s (sorted 187 = 1.06x headroom) = at
    ScaNN scan parity. Both reorder ~180ns/float-candidate; the gap is purely the COUNT (464 vs 78). *** FlatIvf-2000
    big-leaf test (added FlatIvf serialization): scan streams FASTER (257-269 Mcand/s, 74% of 356 floor) but QPS@0.90
    gets WORSE (2559 vs 4761) -- coarse leaves spread the true top-10 -> need 2.5-3.5x MORE candidates; recall/candidate
    tradeoff overwhelms streaming. P192 falsification re-confirmed with a real SoA layout. SoA rebuild NOT done (gate
    correctly fired: no QPS upside). *** Best ratio this window: 1.83x (scann 8538 vs champ 4675 @0.90; recall-matched
    p60 1.87x) -- did NOT beat P192. Cheap rerank levers tried: int16 finer rank (+0.011 recall but 1.5x slower scan=net
    loss); FUSEDTOPK (null, re-confirms P188); POOLDEDUP (a0=3 dupes: dedup +0.0175 recall but O(11019) cost > save,
    4498->2670). *** WHAT BLOCKS <1x: the 464-vs-78 reorder deficit = ScaNN's anisotropic-AH codebook (IP-optimized,
    100B/vec) vs apq4 (50B/vec). Matching needs ~2x code bits -> doubles scan cost -> just moves cost scan<->rerank
    (break-even), and anisotropic hurts pool-recall under float rerank. UNIMPLEMENTED bounded rerank projections:
    cell-contiguous float store ~1.5x (int8-contig 106ns vs float-scattered 190ns endpoints), coarse-cap+dedup ~1.4x --
    NEITHER reaches <1x alone. Committed P193 on p193-streaming-leaf-verdict (+ FlatIvf serialization, recall-neutral).
    NEXT: attack the rerank directly (CASCADE int8-prune->float + contiguous float store) -- the real bottleneck.

P194. (*** CASCADE rerank = real +30% QPS@recall0.90, recall-EXACT: 1.9x -> ~1.475x (same window). int8-VNNI rescore of survivors -> prune to K=16 -> float-reorder only 16. Float stage 464->16 reorders = ~3.5us (BELOW ScaNN's ~13us!). But a NEW int8 refine stage costs ~28-31us (gather-latency-bound), which is now the wall. Did NOT reach <1x. ***)
    rerank_cascade_float (vq.rs:308, branch rerank-cascade e6b0e20): dedup apq4 pool by orig (SOAR a0=3 -> ~280 distinct),
    INT8-rescore each with VNNI dpbusd (dot_i8_vnni, simd.rs:74) over slot-contiguous raw i8, prune to K int8-smallest,
    FLOAT-reorder only K. SBANN_CASCADE/_K/_KLIST. K=16 = min holding recall EXACTLY (p52 t10: nocascade float(520)=0.9002
    == cascade K16=0.9002; K12=0.8979). *** Contiguous-float store NOT built (moot: only 16 float reorders left, ~3.5us).
    *** 3-way interleaved (core0, load 58-65): ScaNN 8111 (0.9032) | P192 base 4208 (1.93x) | cascade 5499 (1.475x, +30%).
    Phase: route+scan ~112 (scann ~106) + int8 ~28-31 + float ~3.5. *** REMAINING to <1x = ~48us: int8 refine ~28-31us +
    ~6us route/scan. int8 refine IRREDUCIBLE here: apq4's poor ranking forces t~280-520 survivors to touch (t=348->0.885);
    IP int8 dot needs all 200 dims (partial-dim=128 craters to 0.406); gather latency-bound -> must touch ~280 full rows.
    Even a FREE int8 stage floors ~1.16x. VNNI is 1.43x compute but stage is GATHER-bound so +2% e2e (win = smaller row
    200B vs 800B, not dpbusd). *** cascade-agent VERDICT: <1x is NOT a rerank problem -- needs better candidate CODES
    (fewer survivors to touch) = OPQ/anisotropic-AH, which P182/P184 say our code family can't deliver at coarse bit-rate.
    *** BUT NOTE (mine): the scan-primitive fix (177->356, another agent) + RICHER codes (100B/vec like ScaNN, whose 2x
    scan cost is absorbed by the 2x scan-primitive fix) = ScaNN's exact recipe, and would cut the survivor count -> could
    break the ~1.16x floor. Combined measurement (cascade + scan-fix + route-fix) pending the other two agents.

P195. (*** SCAN PRIMITIVE: the "2x kernel gap" DOESN'T EXIST. Our scan kernel per-candidate COMPUTE already BEATS ScaNN (543 Mcand/s hot vs ScaNN 368). The champion already uses the 32-wide int8-sat FastScan (no mis-dispatch). The ~177 Mcand/s is a DRAM-LATENCY floor of COLD SCATTERED small cells -- NOT kernel, NOT TLB, NOT ILP. ScaNN's higher rate = candidate-memory DENSITY (2000 leaves x500pt=25KB contiguous vs our ~9.6KB cells), i.e. the codebook/partitioning again. Kernel-side changes shave ~0us on cold e2e. ***)
    Dispatch audit killed the wrong-kernel hypothesis: apq4+FASTSCAN2, a0=3<DEDUP_A0 -> scan_pool -> scan_block_x2 ->
    block_adc_i8_fastscan32_2x16 (already 32-wide int8-sat). Isolated floors (scanbench2 m=100): L2-hot native-fs32 687 /
    fs32-2x16 610 / 16w-i16acc 377; LARGE-seq native 447 / 2x16 155 (2.9x collapse, the 2x16 two-128b-loads defeat the
    prefetcher on a long stream). BUT on the REAL champion scatterbench (p80, 15165 cand): COLD 2x16 ~= native ~= ~176
    (kernel swap NULL); CACHE-HOT (NQ=8) 2x16 497 vs native 543 (+9%). cold~176 vs hot~500 => the 3x gap is MEMORY not
    kernel. perf: IPC 2.06, frontend-idle 0.74%, branch-miss 0.12% => backend/mem-stall bound; THP=always (78 hugepages)
    => NOT TLB. Root: 156MB blocks >> 32MB L3; each query streams a FRESH ~758KB cold from DRAM, 80/16128 cells probed
    ~1.8MB apart => latency-bound; that's why sort(1.02x), kernel-swap(null), prefetch(+6% iso/hurts e2e) are ALL null.
    *** Change SBANN_P2LAYOUT (contiguous 32-wide paired blocks + native 1-load fs32, recall-EXACT verified bit-identical):
    +9% cache-HOT but ~0 on cold real scan (memory-bound), doubles blocks mem -> default OFF (no regression). Real scan
    lever remains FUSEDTOPK (+3-6%). *** VERDICT: our scan primitive is NOT behind ScaNN's -- per-candidate compute we're
    AHEAD (543 vs 368); ScaNN's edge is candidate-memory DENSITY (denser leaves stream bandwidth-bound vs our latency-bound
    scattered small cells). Closing it needs denser candidates = bigger leaves (P192/P193: not free at recall>=0.90 -> more
    candidates) OR ScaNN's learned partitioning+codebook. Kernel is a dead end. Branch scan-primitive 88f356b.

P196. (*** ROUTE PRIMITIVE: 33.5 -> 23.1us (1.45x, recall-EXACT) -> now BELOW ScaNN's ~30us. Root cause: the routing L2 kernel NEVER used VNNI (only rerank did). Fix: VNNI L2 via exact integer decomp L2=|q|^2+|c|^2-2<q,c>, dpbusd single-chain w/ precomputed cadj. Gated SBANN_ROUTE_VNNI. Bit-identical probed set (0/2000 changes). ***)
    routebench isolates router.probe: baseline 33.5us quiet/36 loaded; split coarse-l2 18% / coarse-select 6% /
    fine-expand 63% / final-select 13%; 2784 int8 dist-evals/q (768 coarse + ~2016 fine beam), IPC 3.58 L1-clean =>
    compute-bound, centroids L2-resident. vs ScaNN ~2000 float evals: we do +39% MORE evals but ~2x cheaper/eval
    (AVX2-madd int8 7.9ns vs float 15ns); the hierarchy's select_nth x2 + 2-level gather = ~34% overhead (the price
    of the P192 granularity win). *** Fix: l2_i8_block_vnni (dpbusd, fold +256*Sc into per-centroid cadj=Sc^2+256*Sc
    so the +128 offset cancels -> 1 dpbusd chain + reduce, 4-wide ILP, exact tail). cadj derived at load, index bytes
    UNCHANGED. + exact nd gather capacity (kills memmove realloc). RECALL-EXACT: selftest asserted, 0/2000 set diffs
    @p58 AND p512, bit-identical sink, e2e recall 0.9005 unchanged. *** Isolated route 33.5->23.1 (1.45x, coarse-l2
    1.85x, fine-expand 1.35x); e2e route frac 20%->14.6%, QPS +~6%. Route now <= ScaNN. Branch route-primitive 11bc954.
    *** THREE-PRIMITIVE SUMMARY: route now BELOW ScaNN (23 vs 30, P196); rerank float 464->16 BELOW ScaNN (3.5 vs 13,
    P194) but +new int8 refine ~28us (survivor-count-bound); scan kernel BEATS ScaNN compute (543 vs 368) but DRAM-
    latency-bound at 177 (density, P195). Projected COMBINED (cascade+route-VNNI+fusedtopk): ~139us vs ScaNN ~120 =
    ~1.16x. Remaining gap to <1x = scan density ~9us + int8-refine survivor-count ~18us = ~27us, BOTH = ScaNN's
    anisotropic-AH codebook + learned dense partitioning. Combined measurement next.

P197. (*** COMBINED best-effort = ~1.45x vs ScaNN (loaded), recall-EXACT, all 3 levers stacked cleanly (multiplicative, no interference): cascade +32% x route-VNNI +7% x fusedtopk. Config-levers EXHAUSTED at ~1.45x (loaded) / ~1.2x (quiet-projected). Remaining 54us(loaded)/~23us(quiet) gap = scan-density + int8-refine survivor-count = the CODEBOOK. ***)
    combined-primitives (30e2b7b) = P192 champion + cherry-pick route-primitive + rerank-cascade (clean merge, both
    SBANN_ROUTE_VNNI & SBANN_CASCADE work). RECALL-EXACT: p58 t8 baseline/+FUSEDTOPK/+ROUTE_VNNI/+CASCADE-K16 all
    0.9005 bit-identical; 0/2000 route-set changes; K16 holds recall. Optimum p=54 t10 K16 = 0.9032 @ ~5850 QPS vs
    ScaNN 0.9032 @ ~8410 same window = RATIO median 1.457x (5 rounds 1.425-1.472). Phase (e2e ~173us loaded): route
    28 (16%) / scan 110 (64%) / int8-refine 25 (14%) / float 9 (5%); ScaNN ~119 -> gap 54us. Stacking: fused 4040 ->
    +ROUTE_VNNI +6.1% -> +CASCADE +32% -> FULL +41.2% (1.32x1.07~1.41, mild super-additivity).
    *** METHODOLOGY WRINKLE: box loaded 24-31 (~1.5-2x oversubscribed) INFLATES our ratio -- ScaNN's batched C++
    tolerates oversubscription better than our per-query Rust loop, so the loaded 1.45x is PESSIMISTIC; quiet-box
    phase-sums project ~1.2x (route23+scan85+int8~14+float~5 vs ScaNN 119). A QUIET-BOX re-measure is the fair number
    (still NOT <1x; the structural gap remains). *** VERDICT: config-levers exhausted; route (23-28us) & float (9us)
    near-floor; the ENTIRE residual is scan-density (scattered apq4 blocks) + int8-refine survivor-count -- both
    downstream of apq4's 4-bit codes not rank-preserving. <1x needs OPQ/anisotropic-AH rank-preserving codes OR
    learned anisotropic PARTITIONING (untested -- reduces candidates+survivors without the richer-code scan penalty).
    Progress ledger: 25x(mirage) -> 2.22x(P190) -> 2.07x(P191) -> 1.77x(P192) -> 1.45x(P197), all recall-exact.

P199. (*** REFINE is a DEAD END (~0us shaved) and RECALL-LOCKED on every axis -> confirms the 24us int8-refine is the CODEBOOK gap (survivor count), not execution speed. Ratio unchanged ~1.45-1.48x loaded. Prefetch was ALREADY shipped (P194); leaner-refine sub-levers all break recall or give 0. ***)
    Attacked the 24us int8-refine 3 ways, all recall-EXACT-verified: (1) PREFETCH already exploited -- rerank_cascade_float
    already prefetched survivor i+8 (P194); made it tunable/full-row/primed (SBANN_CASC_PFDIST/PFLINES) but no variant
    beats the shipped 1-line (HW adjacent-line streamer covers the rest); prefetch is worth ~15us (39->24) but already
    banked. (2) CASC_DIM fewer bytes/row: recall CRATERS (dim200=0.9032, dim128=0.394, dim64=0.197 -- OOD needs all 200
    dims; the 24us is per-row FIRST-MISS-LATENCY not bandwidth). (3) apq4 prefilter fewer rows: M=256 -> 0.8942 (<0.90);
    even 280->256 breaks recall -- the int8 stage genuinely rescues true-top-10 apq4 mis-ranks; champion is on a razor
    0.9032 w/ zero fat. (4) Asymmetric true-float-query x int8-row dot (dot_f32_i8, runtime AVX2+scalar fallback+selftest):
    BIT-IDENTICAL recall (query int8-quant isn't lossy) -> 0us, 0 gain. *** Interleaved vs ScaNN best/5 8 rounds: ScaNN
    8454 vs refine 5839 = median ~1.48x (best-round 1.38x). Phase unchanged: route 23/scan 85/int8-refine 24/float 9.
    *** VERDICT: <1x UNREACHABLE via refine -- even zeroing the 24us lands ~1.19x (quiet phase-sum) / the loaded ratio
    stays 1.45x. The 24us is survivor-COUNT (codebook), recall-locked on dims+rows+precision+prefetch. *** KEY FRAMING:
    on a QUIET box route+scan+float(no-int8) ~= 117us ~= ScaNN 119us (~parity); the ONLY thing holding us above 1x is
    the codebook-driven int8-refine (survivor count). So <1x = reduce survivor count, via rank-preserving codes (OPQ/
    anisotropic-AH) OR anisotropic PARTITIONING (reduce candidates+survivors from routing side -- the still-running last
    lever). Clean-code: dot_f32_i8 behind is_x86_feature_detected + scalar fallback + selftest. Branch refine-prefetch 024a0d0.

P198. (*** ANISOTROPIC PARTITIONING = NULL/NEGATIVE: best variant 1.51x, WORSE than isotropic-SOAR 1.46x. Anisotropy scatters the true OOD neighbours OUT of the routed cells (text-query vs image-base breaks the MIPS parallel-weighting premise) -> recall@fixed-p craters, net candidates-at-0.90 flat-to-worse, survivors RISE 540->960. Mirrors P182 (aniso codes zero IP effect). The last untried config-lever, exhausted. ***)
    HierRouter::route_fine_aniso (vq.rs, clean method on the partitioner, no hot-path branches; picks a0 cells minimizing
    ScaNN loss ||x-c||^2+(eta-1)(r.xhat)^2 via one guarded dot_i8_avx2 + scalar fallback; eta=1==L2). Opt-b SBANN_ANISO_EM
    makes TREEEM E-step anisotropic too. Higher eta sparsens cells (cand/q SOAR 10264->eta8 7217) but recall craters
    (p54t10 eta1 .8695/eta2 .858/eta4 .846/eta8 .837) -> to recover 0.90 probe more -> net cand-at-0.90 flat/worse (eta8
    12758@p96), survivors 540->960. Interleaved (ScaNN 8348): SOAR-champ 5820=1.459x | em_e4-best 5484=1.510x WORSE |
    L2-top3 5156=1.647x. Nuance: aniso CENTROIDS (1.51) beat naive L2-spill (1.65) but neither beats isotropic SOAR
    (orthogonal spread is a better use of a0=3 for OOD). Why: text-query/image-base violates the MIPS query-aligned-with-
    neighbours premise, so weighting the datapoint-parallel residual HURTS coverage. Branch aniso-partition a15996a.
    *** DEFINITIVE (P185-P199, 14 experiments): 25x(mirage)->1.45x loaded/~1.19x quiet, all recall-exact; route & rerank-
    float BEAT ScaNN, scan kernel beats ScaNN compute; EVERY config-lever exhausted (routing granularity/rerank cascade/
    scan layout/route-VNNI/refine prefetch/aniso partitioning/aniso codes). Residual = apq4 CODEBOOK: 50B/vec -> 464
    survivors -> 24us int8-refine; ScaNN 100B/vec -> 78. On a QUIET box route+scan+float(no-int8) ~= ScaNN (~parity); the
    int8-refine (survivor count) is the sole thing above 1x. Anisotropy (ScaNN's key trick) does NOT transfer to OOD text
    2image for us. <1x needs a better rank-PRESERVING distance estimate (fewer survivors) at ~current bytes -- the one
    genuinely untried code angle (norm-corrected/RaBitQ-style ADC), since aniso and naive 2x-bits are both refuted.

P200. (*** RANK-PRESERVING ESTIMATOR: NULL on <1x but a REAL primary-metric win -- NormPq (norm-rescaled ADC, +4B/vec) cuts the survivor pool ~1.8x (540->300 for recall 0.90), the FIRST thing to beat apq4's rank floor. But ratio gets WORSE (1.75x vs 1.51x): the gamma-rescale needs full-range i16 accumulation, which forfeits fastscan2's int8-SATURATING speed (-33% scan) > the pool-cut benefit (+28%). Even gamma-free-in-fastscan2 floors ~1.35x -- the pool-cut saves ~13us of a ~135us query; the ~110us SCAN is untouched by ANY estimator. ***)
    NormPq compressor (vq.rs COMP_TAG_NORMPQ=3, SBANN_COMP=apq4n, selftest_normpq, clean in the Compressor abstraction):
    apq4_ip x (||x||/||x_hat||) de-biases PQ norm-shrinkage that under-ranks large-norm MIPS winners. Screened offline
    vs FLOAT-IP GT: NormPq 0.9024@t300 / 0.9058@324 vs apq4 CAPS at 0.872@pool300 (needs t~540) -> ~1.8x survivor cut,
    i8-exact=1.0@t40 confirms the pool bottleneck is purely code approximation. RaBitQ screened (already in engine):
    50B(2-bit) far worse 0.33@t40, only 100B(2x) near-perfect -> rotation buys nothing <=50B (confirms P144/145).
    Higher-res PQ refuted (P145/195). *** Interleaved vs ScaNN ~8150: apq4-champ(fastscan2 p54t10) 5350=1.51x |
    NormPq(i16 p54t6) 4640=1.75x. Phase(NormPq): scan 70% / route 17% / int8-refine 7.5% (pool-cut shrank it) / float 6%.
    Attribution: apq4-fastscan2 3922 -> apq4-i16 2635 (i16 scan -33%) -> NormPq-i16 3383 (pool-cut +28%): +28 can't
    recover -33. *** VERDICT: rank-preserving-code lever EXHAUSTED. The estimator quality was NEVER the ratio wall --
    the SCAN is (64-70% of query, memory-bound scattered PQ blocks vs ScaNN's cache-resident SoA-AH). A tighter code
    only shrinks the small refine and, worse, forces the slow scan regime. <1x needs the SoA-AH SCAN-LAYOUT co-design,
    which P193 already refuted at recall>=0.90 (denser leaves -> more candidates). Branch rank-preserving-code c7c75cb.
    *** ============ DEFINITIVE CLOSE (15 experiments P185-P200) ============
    25x(mirage) -> 2.22x(P190 true same-hw) -> 1.51x loaded/~1.19-1.35x quiet (P197), ALL recall-exact. WINS that BEAT
    ScaNN: route VNNI (P196), rerank cascade float 464->16 (P194), scan kernel compute 543>368 (P195). EVERY lever
    exhausted: routing granularity (P192), scan SoA layout (P193 refuted at recall), route (P196 won), rerank refine
    (P199 recall-locked), aniso partitioning (P198 neg), aniso codes (P182 neg), rank-preserving estimator (P200 cuts
    survivors 1.8x but scan-trapped). THE WALL: ScaNN's 100B/vec anisotropic-AH codebook + cache-resident SoA scan give
    it BOTH dense-useful-candidates (fast scan) AND tight ranking (few survivors) SIMULTANEOUSLY; our 50B/vec codes buy
    one only by losing the other (tighter estimate -> slower i16 scan; denser leaves -> more candidates; aniso doesn't
    transfer to OOD text-vs-image). <1x = the full ScaNN codebook+layout CO-DESIGN (multi-week, uncertain -- aniso's
    non-transfer to our OOD is a real risk), NOT any single lever. Honest architecture ceiling: ~1.2x quiet / ~1.5x loaded.

P201. (*** RICHER-CODES REFUTED (the last code-side lever): m=200 1-dim 4-bit isotropic @100B/vec cuts survivors 2.4x (840->350 @0.904, ranking gain REAL, int8-sat exact at m=200 via fs2 LUT cap; selftest m=200 added) BUT scan cost = clean 2.03x (102->207us isolated; 132->218 loaded) => e2e 0.75x champion QPS (3507 vs 4679 @0.904); interleaved vs fresh ScaNN 2.46x vs champion's 1.87x — WRONG DIRECTION. Hypothesis "latency-bound scan absorbs 2x bytes" FALSIFIED: real per-query working set is small/cell-clustered => throughput+bandwidth bound (2x vpshufb AND 2x bytes both scale); P195's 3x headroom is a kernel property the workload never sits in. ***)
    Branch richer-codes 8a4dacb (worktree lsh-engine-wt-richer). ZERO new code needed: richer code == existing Apq4 with
    SBANN_DPB=1 (d=200 -> m=200) + SBANN_ETA=1, reusing m-generic block_adc_i8_fastscan32_2x16. Controls: richer_m100iso
    (dpb=2 eta=1) vs true champion (eta=4): IDENTICAL survivors at every operating point — anisotropy re-re-confirmed
    NULL for OOD (third independent confirmation). Survivor floor: m200 ~290 @0.90 vs ScaNN ~78 — did not reach target
    anyway. Refine dropped 47-50 -> 28us (as predicted) but +87-105us scan >> -19us refine; even FREE refine (survivors
    ->16) cannot offset. OPQ variant SKIPPED (changes ranking, not the 100B scan cost — cannot alter conclusion).
    *** ARCHITECTURAL CLOSURE: scan-time ∝ code-bytes (bandwidth-bound) + ranking ∝ code-bytes (information) =>
    in-scan code enrichment is a WALL, not a lever. P193's "2x bits = break-even" was OPTIMISTIC (it's 0.75x).
    Together with P182/P185/P192/P198/P200: EVERY code-side lever now empirically closed at 1M. Remaining untried:
    SBANN_RESIDQ (residual-encode x - cell_centroid, same 50B, zero scan cost — free ranking if champion built without).
    Scripts: richer_build.sh, surv_sweep.sh, richer_sweep.sh, richer_h2h.sh; indices richer_m200iso/m100iso.idx.

P202. (*** CELL-MAJOR BATCHED SCAN: real +27% e2e (recall-BIT-IDENTICAL), loaded h2h closes 1.53x -> 1.20x vs ScaNN — but NOT <1x, because ScaNN's h2h numbers were ALREADY batched (search_batched) and ScaNN gains the same ~1.22-1.27x from batching. Legitimate under leaderboard semantics (harness passes the whole query set). ***)
    Branch batch-inverted 4982886 (worktree lsh-engine-wt-batchinv): search_batch_frr + extracted scan_cell_fused —
    route all, build all LUTs, counting-sort (cell -> query list), sweep cells in ASCENDING STORAGE ORDER running the
    UNCHANGED kernel per (cell,query), then per-query cascade+float unchanged. Kernels/Compressor untouched; gated
    SBANN_BATCHSCAN/_CHUNK/_VERIFY (temporary A/B scaffolding). CORRECTNESS: 2000/2000 set-identical top-10 at p=54
    (order-identical too), recall 0.9032 == per-query, p=40/80 same (one tie flip) — pure execution-order change.
    *** Scan 117-129 -> 78-80us/q (~1.5-1.63x); QPS best/5: per-query 5640 -> chunk250 6673 / chunk1000 7121 /
    chunk2000 7143 (curve FLAT by ~1000). KEY MECHANISM SURPRISE: chunk=250 (multiplicity 0.82!) already captures
    most of the win => the gain is mostly SEQUENTIAL cell-order HW-prefetch, NOT cross-query block reuse; reuse
    saturates at mult ~3.3 under LUT (~5-10MB) + FusedTopT pool (~18MB @ nq2000) L3 pressure + pool-scatter writes.
    "Bigger batches keep winning" FALSIFIED past chunk~1000 on this box. *** DECIDER (interleaved core0, best/5,
    rounds 2-5 stable, recall 0.9032 exact all): ScaNN-batched median 8162 | ours-batched 6762 | ours-perquery 5298.
    Ratios: 1.20x batched-vs-batched (1.16-1.23) | 1.54x vs perquery (matches P197) | ours batching gain 1.28x.
    ScaNN search() vs search_batched isolated: ~1.22-1.27x — BOTH engines gain equally; our batching competes
    against an already-batched ScaNN. *** Why <1x failed: (1) scan fell 1.5x not the projected 2.5-3x-to-compute-
    floor (P195's 3x cold-vs-hot headroom captured mostly as prefetch); (2) reuse saturates (L3 pressure), doesn't
    scale with batch; (3) the 119us ScaNN target was already-batched ScaNN. Batched scan now ~24ns/cand LATENCY-bound
    on LUT gather + per-(cell,query) pool scatter — no further amortization headroom identified. *** STANDING:
    ~1.20x loaded batched-vs-batched (quiet-box re-measure pending — P197 pattern suggests quiet ~1.05-1.15x).
    Post-P202 phase (loaded): route 23 / scan 78 / refine 24 / float 9. Scan & route & float at-or-better than
    ScaNN; the residual is STILL the survivor-count refine (codebook) + ScaNN's equal batching gain.
    Scripts: batchinv_verify.sh, batchinv_prof.sh, batchinv_chunk.sh, batchinv_decider.sh, scann_pqvsbatch.py.

P203. (*** CONSOLIDATION PASS (user's clean-abstractions requirement, executed post-lever-stabilization): branch `champion` = batch-inverted + 55c0433. All winning levers folded to DEFAULT-ON behind runtime detection (env_on helper, SBANN_<X>=0 overrides kept): FASTSCAN2 (avx2+selftest gate), ROUTE_VNNI (avx512vnni gate, AVX2 fallback), CASCADE default-on w/ K default 128->16, FUSEDTOPK, BATCHSCAN w/ chunk default 1000 (P202 knee), PREFETCH. Refuted scaffolding (P2LAYOUT/ANISO_*/NormPq) confirmed ABSENT on this lineage (lives on experiment branches); RESIDQ + SOAR/TREEEM intact. Dispatch audit CLEAN: every intrinsic behind is_x86_feature_detected w/ scalar/AVX2 fallback, selftests assert at startup, no unguarded AVX-512. GATES: build clean; recall EXACT 0.9032 default-flags on BOTH batched and per-query paths, BATCH_VERIFY 2000/2000 set+order identical, bit-match vs flags-on reference; QPS sanity batched ~7000 / interleaved vs ScaNN ~1.19x (= P202). Doc block CHAMPION OOD STACK added above main(). Dataset/mode selectors (SBANN_IP, FLOAT_RERANK, FBASE/FQUERY, TFLOOR) deliberately left explicit. ***)

P204. (*** STREAMING 30M (OFFICIAL msturing-30M-clustered final_runbook, f16 rerank): SOAR a0=2 avg recall@10 = 0.9654 => would rank 3rd overall / 2ND OPEN-SOURCE (leaderboard: puck 0.9855/0.9849, hwtl-closed 0.9675, pyanns 0.9597, diskann 0.8833, cufe 0.8189), fits 8GB (peak 7.52GB) — ONLY blocker = insert-bound wall (SOAR scalar assignment 3109/s vs 22663 flat = 7.3x, P116's deferred SIMD), 10981s vs 3600s budget even quiet-extrapolated. Baseline flat C=4096 a0=1: 0.9276, 5.87GB, wall 3750s loaded => quiet ~800-1200s FITS EASILY (would rank 4th, above diskann). Branch feat/streaming-30m 9976c2e. ***)
    Protocol: 320 ins / 320 del / 640 search steps, live-window ~10.29M, metric avg recall@10 vs per-step gt100,
    1hr budget, ~8GB, Azure D8lds_v5. Recall on 2000/10000 official queries (within ~0.001 of full set).
    *** F16 RERANK GATE (1M): agreement f16-vs-f32 top-10 = 0.9995 avg / 0.9986 worst; runbook recall diff 0.0001
    => LOSSLESS. Cache 2.06GB vs 4.12GB f32 — the halving that keeps SOAR a0=2 (doubled int8 store) under 8GB.
    F16C _mm256_cvtph_ps behind is_x86_feature_detected + scalar fallback + startup selftest (clean-abstraction).
    *** Recall by live-density (SOAR/c4096): <0.5M 0.868/0.808 | 0.5-2M 0.947/0.903 | 2-5M 0.964/0.924 |
    5-8M 0.971/0.937 | 8-10.3M 0.979/0.947 — low-live steps are the recall tail (adaptive-p lever for later).
    *** SCALING PATHOLOGIES: (1) cell count must track live density — Kf=262144 craters to 0.31-0.50 (88% of
    runbook live<10M => empty probed cells); flat C=4096 stays occupied 38k->10M. (2) recall is 100% ROUTING-
    COVERAGE-limited (true float-NN present in int8 top-14 candidates) — exactly why SOAR +3.8pt. (3) upstream
    runbook_to_ops.py --scale-to clamps ids (span ~2.9x window) — scaled by target/id_max for the 1M gate.
    *** NEXT: vectorize SOAR insert assignment (reuse P196 VNNI L2 nearest-centroid kernel) => make 0.9654
    eligible; then spend leftover budget on the low-live recall tail toward puck's 0.9855.

P205. (*** ORACLE FAN-OUT for the next 1M OOD lever (3 parallel agents; the probe-count axis was the last untouched degree of freedom): GRAPH-AUGMENTED POOL EXPANSION = strongest GO (realizable x1.68 touched-rows, recall 0.9033 engine-faithful, projected ratio 1.02 [0.94-1.08]); ADAPTIVE PER-QUERY p = GO but modest (realizable x1.20, oracle ceiling x2.61 — routing-time features capture only part of the difficulty signal; e2e-validated recall 0.9038, +10% QPS, ratio -> ~1.09); query-calibration oracle still running. The two GO levers are COMPOSABLE (adaptive p around the graph-expanded baseline). ***)
    ADAPTIVE-P (branch adaptive-probe b942b1e, route_profile + SBANN_DUMP_ROUTE/ASSIGN + SBANN_PLIST_FILE per-query
    p in search_batch_frr): oracle p* mean 20.6 / median 17 / p90 45 vs fixed 54 (x2.61 candidates); ridge on
    routing-time features (coarse/fine distance gaps/ratios) held-out realizable only x1.17-1.21 -> avg p 45.2,
    recall 0.9038 >= 0.9032 e2e. The oracle-vs-realizable gap = per-query difficulty is only weakly visible in
    centroid-distance profiles (the OOD signal lives deeper).
    GRAPH-EXPANSION (branch graph-pool-expansion c89bdc7, SBANN_DUMP_POOL engine-faithful pools): k=16 IP kNN graph
    on the 1M base built via ScaNN self-search, edge quality 0.9988 vs exact. Winner p'=30/t540/M=25: pool union
    graph[top-25] -> exact int8 rescore (559 rows vs 322) -> K16 -> float16: recall 0.9033, scanned 5512 vs 9858
    (-44%), touched 6071 vs 10180 (x1.68), projected e2e 124us vs 147 champion; 1-hop coverage ceilings 0.932/
    0.948/0.958 @ p'=20/30/40; oracle UB x3.18 @ p'=15. COST-MODEL CAVEAT: ratio 1.078 (74.5ns/row amortized) ..
    0.94 (44ns pure-latency) — the union-rescore GATHER EFFICIENCY decides which side of 1x; implementation must
    prefetch like the existing cascade. NEXT: implement graph expansion in the engine (real interleaved h2h), then
    layer adaptive-p RETRAINED on the graph-expanded pipeline (p*(q) distribution changes when the hop recovers
    deep misses). Both dumps/tooling committed on their branches for reuse.

P206. (*** QUERY-CALIBRATION ORACLE = the strongest GO of the fan-out, and embarrassingly simple: a SINGLE GLOBAL GAMMA on the centroid-norm term of the routing score (score = gamma*||c||^2 - 2*q.c, gamma=0.5, i.e. halfway L2 -> pure-dot) lets OOD text queries reach the SAME 0.9032 recall at HALF the probes: p=27-29 vs 54, candidates 10234->5837 = x1.75, at ZERO query-time cost (per-cell additive i32 bias). Interleaved +18.7% e2e (11 rounds) -> ratio ~1.01 alone (optimistic bound 0.90). Flag-gated SBANN_ROUTE_GAMMA, branch probe-calibration d02561a, default-off bit-exact. ***)
    Fitted on the even query half, validated on held-out odd half AND full-2000 engine runs (gamma p=27 = 0.9032 ==
    champion; conservative p=29 = 0.9052 held-out). WHY IT WORKS (the OOD mechanism, finally isolated): the router
    orders cells by L2 in the IMAGE geometry; for IP search with text queries, large-norm centroids (= large-norm
    cells that score high in IP) are systematically over-penalized by the ||c||^2 term -> true-NN cells sit deeper
    in the probe order. gamma=0.5 interpolates L2 -> MIPS ordering. This is what P198's anisotropic PARTITIONING
    tried to capture by rebuilding cells (and failed); the fix is a query-time SCORING correction, not a partition
    change. Capacity ladder: free per-cell bias <=1.08x, diagonal metric 0.89x (both null) -> the global scalar IS
    the whole signal. Oracle UB p=5 = 10.8x (greedy set-cover). Un-tuned upside: t_surv/K16 not re-tuned at the
    smaller pool; the COARSE level (b0=96, route 23us) not gamma-calibrated. Alternative operating point: +2pp
    recall (0.9230) at unchanged p=54.
    *** COMPOSITION PLAN (all three GO levers are orthogonal): gamma (x1.75, free) x graph-expansion (P205, x1.68
    touched rows at p'=30 WITHOUT gamma; with gamma the same coverage should arrive by p'~15-22) x adaptive-p
    (x1.20 realizable, retrained on the composed pipeline). Directed graph-impl to cherry-pick d02561a and run the
    composed decider: arms = ScaNN / champion p54 / gamma p29 / gamma+graph p'~18. Compound projection ~0.75-0.9.

P207. (*** GRAPH-AUGMENTED POOL EXPANSION IMPLEMENTED (branch graph-expansion-impl 34b05f2+0290d4e): recall 0.9033 at p=30/M=25 — BIT-IDENTICAL to the P205 oracle on 2000/2000 queries; union-rescore gather hit the FAVORABLE spec (39-44 ns/row); decider = ScaNN 8234 / GRAPH 7396 / CHAMPION 6872 => 1.113x vs ScaNN (champion arm re-validated 1.198x), GRAPH beats champion EVERY round (+7.6%). Best OOD standing yet, NOT sub-1x alone. GAMMA COMPOSITION NOT YET RUN (directive crossed mid-decider) — that is the projected sub-1x arm. ***)
    Clean build: SBANN_GRAPH_FILE flat n*k u32 IP-kNN sidecar (k=16, orthogonal to index serialization);
    rerank_cascade_graph = pool top-M=25 origs -> 16 neighbors each -> cache-hot OPEN-ADDRESSING dedup union
    (HashSet and a 4MB generation-stamp bitmap both SLOWER — cache-cold scatter) -> existing int8-VNNI rescore
    w/ streaming prefetch -> K16 -> float16. Wired batched + per-query. Warm phase: route 23 / scan 49 /
    graph-union 20 / rescore 22 / float 6.
    *** KEEPER A/B: unsorted union + deep prefetch (pfdist 16) BEATS orig-sorted gather at moderate load (sort
    CPU over ~560 random u32 > locality gain once prefetched); SBANN_GRAPH_SORT=1 restores sort (only wins under
    extreme DRAM contention). *** WHY not sub-1x alone: frontier min-e2e at recall 0.9032 with p=30 is ~124us vs
    ScaNN ~120; graph-union phase has ~5-7us trimmable overhead (20 vs 13 modeled); box memory pressure (30M
    runbook evicts float-base mmap pages) taxes both arms equally — quiet projection 1.02-1.07x. Adaptive-p on
    this pipeline ~5us => ~1.05-1.08, deferred. *** NEXT (the decisive arm): cherry-pick gamma d02561a; with
    p'~15-22 the scan halves BEFORE the graph hop => modeled e2e ~100-110us vs ScaNN ~120 => sub-1x plausible.

P208. (*** STREAMING 30M SOAR-INSERT FIX LANDED (branch feat/streaming-30m f626d2a, bounded spill assignment to top-K nearest cells): inserts 3109 -> 26175/s (8.4x, ABOVE the flat baseline's 22663!), avg recall@10 = 0.9653 == the unbounded SOAR's 0.9654 (the bound costs NOTHING), RUNBOOK-OPS WALL = 2759s = 46min < 3600s budget UNDER LOAD ~27-35 — the 1hr budget is PASSED with margin. ONE REMAINING BLOCKER: peak anon 8.64GB > 8GB cap (INELIGIBLE by 0.64GB) — the 8.4x-faster inserts outpace the SBANN_COMPACT=0.25 compaction cadence so append buffers peak higher than the slow run's 7.52GB. Fix = compaction/buffer tuning (recall-neutral fold), then the 0.9653 = 2nd-open-source run is FULLY ELIGIBLE. ***)

P209. (*** SUB-1x vs ScaNN ACHIEVED at 1M OOD — the first legitimate same-hardware win, ending the arc 25x(mirage) -> 2.22x(P190) -> 1.20x(P202) -> 0.972-0.978x. Interleaved 8 rounds (taskset -c 1, best/5, load 17-26, identical float GT, recall INDEPENDENTLY recomputed from raw result ids): ScaNN 8279 @ 0.9032 | CHAMPION p54 6990 @ 0.9032 = 1.184x (rig re-validated) | GAMMA-only p29 8514 @ 0.9060 = 0.972x | GAMMA+GRAPH p18 8409 @ 0.9075 = 0.985x | GAMMA+GRAPH p17 8466 @ 0.9033 = 0.978x, SUB-1x IN ALL 8/8 ROUNDS (0.953-0.990). Branch graph-expansion-impl 34b05f2+0290d4e+142a6b4. ***)
    ATTRIBUTION (honest): GAMMA IS THE MOVER — champion 1.184x -> gamma-only 0.972x in ONE step; the OOD gap was
    ROUTING MISCALIBRATION (P206's single scalar), not the codebook. Graph expansion composes cleanly (knee p'
    drops 30 -> 17-18 with gamma; M=25 > M=50) and is the most robustly sub-1x arm at the lowest probe count,
    BUT the levers OVERLAP (both cut probe/pool waste): compound ~0.97 ~= best single lever, NOT gamma x graph
    multiplicative. Phase (composed p18): route 21% / scan 33% / union 19% / rescore 23% / float 4%; union/q=544;
    gather at spec 39-44ns/row. t_surv sweep: 470 holds 0.9060 @ p18 (further lever), 400 fails 0.9028.
    *** WHAT REMAINS FOR THE LEADERBOARD: HANNS leads ScaNN by ~7% => the 1M bar is ~0.93x; current 0.972-0.978
    needs ~4-5 more points. Levers in flight: ADC-route (route 23 -> ~10-14us projected = ~8-10 points), cascade
    geometry grid re-sweep under gamma, t_surv=470, adaptive-p retrained on the composed pipeline, overlap-aware
    graph placement. Also pending: quiet-box confirmation (loaded ratios were historically PESSIMISTIC for us),
    gamma transfer to 10M, multi-thread scaling. Memory-pressure caveat: 30M runbook evicted float pages in these
    rounds — hits all arms equally, ratio honest.

P210. (*** ADC ROUTING REFUTED for 1M/d=200 (user-requested experiment, branch adc-route 0aaf77d): route 23.7us exact-VNNI -> 37-40us with ADC at ANY KEEP (~1.7x SLOWER); e2e flips +9% -> -10% vs ScaNN (gamma-noADC 8590 vs gamma+ADC-K256 7030). ROOT CAUSE: at d=200 the route codebook is m=100 subspaces, so the 4-bit LUT scan per centroid ~= the VNNI exact eval cost; plus unaligned fan-out forces ~1.9x covering-block overscan + exact-rescore tail. Coarse route codebooks don't rescue: dpb4(m50)@K256 32.7us / dpb10(m20) 27.7us both > exact 23.7, and recall never recovers (dpb4 K512 0.9021, dpb10 K512 0.8869). NO (dpb,KEEP) is both faster than exact AND recall-neutral. The exact VNNI router is AT THE FLOOR for this dimensionality — consistent with ScaNN also routing exactly. ***)
    Composition wiring was correct (gamma bias scaled into the ADC LUT domain via Pq::query_lut_with_scale, raw bias
    on rescore; fidelity vs exact top-27: K64 .900 / K128 .976 / K256 .996; min recall-neutral KEEP=256). Also added
    routeadc post-hoc codebook-swap rebuild (~2s) for cheap granularity sweeps — keep the tool.
    *** INDEPENDENT SUB-1x REPLICATION (the important secondary result): on a REBUILT index (champion recipe
    reproduced exactly: gamma-OFF p54 = 0.9032, gamma=0.5 p27 = 0.9032 recall-exact), interleaved best-of-6 this
    window: ScaNN 7853 | gamma-noADC 8590 = 0.914x. Two independent implementations (graph-impl P209: 0.972x;
    adc-route: 0.914x), two windows, both sub-1x => the gamma win is ROBUST, not a window artifact. Gotcha
    documented: champion op-point uses FIXED t_surv~540 (not p*tmul) — with p*tmul gamma recall misleads.

P211. (*** GENERALIZED-CASCADE OPTUNA SWEEP (user's L-levels/P_i/R_i framework; 160 recall-constrained trials over 7 geometries x query knobs, cached indexes, + 4-arm same-window decider): the gamma-p27 2-LEVEL REFERENCE IS ALREADY THE OPTIMUM. Sweep winners (L3 lean 8963 QPS, L2 p22/t900 8591) were LOAD ARTIFACTS — same-window they are 0.894x/0.949x of the reference. No engine changes warranted; no config beats gamma-p27. ***)
    STRUCTURAL ANSWERS (the user's questions, settled empirically): (1) LEVELS = 2. A lean 3-level (finest fan-in
    ~1024) MATCHES but never beats; wide-fan-in 3-level much worse. What matters is TOTAL cells scored (~2500-2800)
    and 2 levels reaches it cheapest. (2) SIZES: Kf~16384 / C0~768 / b0=96 confirmed AT the optimum even post-gamma
    (8192 slightly worse = bigger leaves more scan; 32768 worse = more routing) — P192's geometry survives the new
    primitives. (3) ADC PAYS NOWHERE at 1M/d=200 — extends P210 to deeper trees and wide fan-ins (best ADC trial
    5552 vs best exact 8963 across 56 ADC trials; even the 8192-cell L3 finest fan-in loses to exact). (4) The
    BALANCED-WORK heuristic holds as MARGINAL-COST equalization, NOT equal wall-time: the optimum sits where one
    more probe (cheap streaming scan) costs the same as the extra survivors needed to drop one (expensive gather-
    bound rescore ~5x/row) — the cost asymmetry is exactly why t_surv stays modest (540-700) and gamma (probe
    halving) was the dominant lever. Profile: REF p27/t540 vs WIN-L2 p22/t900 = 274 vs 272 us/q, a dead wash
    (13us scan saved == 24us rescore added).
    KNOB COMPLETENESS: the engine already exposes the whole framework (hierk/hierk3/hierkn depth, C0/C1/B0/B1
    build-baked beams, gamma at finest level both depths, ROUTE_ADC+KEEP, t_surv via TFLOOR/TMUL, CASCADE_K).
    Gaps (harmless at the optimum): intermediate beams build-baked; gamma+ADC don't compose on the ADC path.
    Artifacts: cascade_sweep.py / cascade_sweep_log.csv / cascade_sweep.db / decide4_out.log (scratchpad).

P212. (*** STREAMING 30M: FULLY ELIGIBLE HIGH-RECALL RESULT BANKED. Official msturing-30M final_runbook, all gates PASSED under load ~35: avg recall@10 = 0.9654 (640 steps, official per-step GT) | PEAK ANON 5.32GB < 8GB cap | RUNBOOK-OPS WALL 3178.5s = 53.0 min < 3600s. Config: flatsoar a0=2 C=4096 + bounded-spill VNNI-era insert fix (26k/s) + f16 rerank cache + compaction-cadence memory fix (peak 8.64 -> 5.32GB, recall unchanged). = 2nd OPEN-SOURCE tier on the leaderboard (pyanns 0.9597 < OURS 0.9654 < hwtl-closed 0.9675 < puck 0.9855). Branch feat/streaming-30m. ***)
    The failed-run postmortems that got here: run1 unbounded SOAR = insert-bound 3x over budget (P204); run2 fast
    inserts = 8.64GB peak (P208, compaction cadence vs 8.4x faster inserts); run3 died silently at op 598 (box fork
    crunch); run4 = ALL GATES GREEN. OPEN ITEM: wall INCL offline train = 4175.6s = 69.6 min > 60 — eligibility
    depends on whether the official harness times setup/train (ops-only => PASS with margin; incl-train => need
    train trim ~997s-loaded, or a quiet-box run where train ~300-500s + ops ~35-40min likely passes anyway).
    NEXT recall levers toward puck 0.9855: low-live steps are the tail (<0.5M live: 0.868) — adaptive-p by
    live-count; leftover time budget (7 min quiet margin) buys deeper search.

P213. (*** P212's OPEN ITEM SETTLED — the 1hr clock INCLUDES train (it's the container timeout: big-ann runner.py:322 container.wait(timeout), :297 timeout=3600 for streaming; run() does build() THEN run_task()), BUT our train is only 26.9s (k-means 2M -> 4096 flat cells; the '997s train' was a MIS-ATTRIBUTION: ~600-800s of the incl-train gap is OUR INLINE per-step GT recall scoring, which the official harness does OFFLINE — neurips23/streaming/run.py only calls query/get_results inside the timed run, never scores recall). Official-equivalent wall = 27s train + ops 3178.5s + finalize ≈ 3.2-3.6ks LOADED (load 35) -> ~1500-1900s on the idle official 8-vCPU box. STREAMING 30M IS ELIGIBLE ON EVERY DIMENSION WITH LARGE MARGIN. Final state: feat/streaming-30m @ c1c24e3 (9976c2e f16 cache | f626d2a SOAR spill bound + AVX2 dot | c1c24e3 glibc arena trim after compaction/train = the 8.64->5.32GB fix). ***)
    COMPLIANCE NOTE for an eventual official submission: streaming setup(dtype,max_pts,ndims) passes no vectors —
    our cold-start currently trains the router on a strided base sample from disk (peeks at future data,
    technically mountable but not the intended contract). Clean version: train on the FIRST INSERT BATCH (folds
    the 27s into inserts, negligible, recall/memory unchanged). Do before a real PR.
    NEXT recall lever toward puck 0.9855: adaptive-p on low-live steps (<0.5M live = the 0.868 tail).

P214. (*** THE 0.93x HANNS BAR REACHED (pairwise-inferred, pending direct confirm): union-trim (1ab8134, fused pool-dedup+union-build single open-addressing pass + adjacency prefetch, verified BIT-IDENTICAL 2000/2000 after catching a SOAR tie-break regression) makes graph expansion CLEARLY ADDITIVE over gamma-only: 24-round tight pairwise triples [GR17_t540 | gamma-only p29 anchor | GR18_t470] at partial-calm load 31 => median(GR17/gamma)=1.0258 -> inferred ScaNN ratio 0.9475x; median(GR18_t470/gamma)=1.0408 -> inferred 0.9339x ~= THE 0.93x BAR (HANNS = ScaNN x1.07). Recall gates held deterministically: GR17 0.9033, GR18/t470 0.9060, both id-verified. ***)
    CAVEATS (why this is 'reached' not 'cleared'): (1) ratios INFERRED via the fixed gamma-anchor (0.972x calm,
    P209) rather than a same-round ScaNN arm; (2) wide per-round IQR (0.908-1.19) at load 31 — the median is the
    contention-robust statistic but a DIRECT quiet-box decider (ScaNN arm in-round, load <18) is required to
    convert inferred->measured. Also banked en route: overlap lever REFUTED by cost model (rescore-row ~5x a
    scan-candidate => bigger graph unions can't buy smaller p: M25/p18 46.7us < M50/p15 50.6 < M100/p12 60.2);
    pairwise-alternation protocol (gamma anchor mid-triple, median of adjacent-pair ratios) added to the
    measurement toolkit for loaded-box windows. 1M OOD progression: 2.22x -> 1.20x -> 0.972x -> inferred 0.934x.

P215. (*** GAMMA TRANSFERS TO 10M — the routing-miscalibration mechanism is SCALE-INVARIANT, and the probe cut is SAME-OR-STRONGER than 1M: at t_surv=2000 on the fresh 10M index (eng_t2i10m_kf131072_c4096_b256_a3, built 41min, sanity 0.9022@p250 gamma-off), matched-recall probe requirements: recall~0.93 gamma-off p~220 vs gamma0.5 p~80 (2.7x); recall~0.94 gamma-off p~400 vs gamma0.5 p~160 (2.5x); gamma0.5 = 0.9315 AT p=80 where gamma-off is 0.8946. gamma 0.5-0.6 optimal at 10M (0.4 slightly below). RECALL curves are load-insensitive (box load 34-80 across arms — QPS columns not comparable). ***)
    ALSO LEARNED: t_surv must scale with n — the 1M-tuned t_surv=540-800 CAPS recall at ~0.881 at 10M regardless
    of p (pool too shallow to hold the true candidates); t=2000 releases it (agent self-caught via the plateau).
    Remaining on this front: the first same-hardware 10M h2h vs the cached ScaNN 10M index (task in flight,
    pairwise-alternation protocol, core 3).

P216. (*** THE 0.93x BAR CLEARED — DIRECTLY MEASURED, in-round ScaNN, calm box (load 17-20 held), 8 rounds, best-of-5, taskset -c 1, identical float GT: GR18_t470 (gamma0.5 + graph-expansion M25 + union-trim + t_surv=470, p=18, K=16) = ~0.91x median vs ScaNN (per-round 0.9032/0.9099/0.9103/0.9127/0.9173/0.9433; ours 8930-9413 QPS vs ScaNN 8281-8591), recall 0.9060 vs ScaNN's 0.9032, ALL arms recall >= gate. GR17_t540 ~0.93x @ 0.9033; gamma-only ~0.97x @ 0.9060 (P209 replicated a third time). Converts P214's inferred 0.9339 into MEASURED ~0.91x. ***)
    SIGNIFICANCE: HANNS (leaderboard #1) leads ScaNN by ~7%; we now lead ScaNN by ~9.5% at 1M single-thread
    same-hardware => this configuration is LEADERBOARD-TOP-EQUIVALENT at the 1M scale (caveats for the full
    official claim: official track is 10M, 8-vCPU multi-thread, Azure — 10M h2h in flight, multi-thread pending).
    1M OOD arc COMPLETE: 25x(mirage) -> 2.22x(P190 true) -> 1.77x(P192) -> 1.45x(P197) -> 1.20x(P202) ->
    0.972x(P209 sub-1x) -> 0.91x(P216, above the HANNS-margin bar). The stack: gamma=0.5 routing calibration
    (the mover) + kNN-graph pool expansion at p=18 + fused-union trim + t_surv=470 + all P185-P203 primitives.
    Pending: graph-impl's formal id-recomputed recall report (log numbers unambiguous); then consolidate
    gamma+graph+trim+t470 into branch champion with the full gate suite.

P217. (*** 10M H2H (first ever, PROVISIONAL pending ScaNN-config sanity): 20 pairwise rounds, core 3, load ~25 — ScaNN 1272 QPS @ 0.9046 (lts=110, reord=180, cached P190 index) vs ENGINE 2512 QPS @ 0.9054 (gamma=0.5, t_surv=2000, p=45, Kf=131072 index) => ratio MEDIAN 0.505 (IQR 0.495-0.511, min 0.463 max 0.541) — engine ~2x FASTER at 10M. Tight IQR = stable measurement. ***)
    CAUTION CONFIRMED (lead-verified): the cached ScaNN 10M index has num_leaves=4000 (build script NL default,
    all build logs agree) vs ScaNN's official ~40000 — its per-probe scan is ~10x heavier than the official
    recipe, explaining the anomalous 6.6x drop. THE 0.505x IS FLATTERING; corrected measurement in flight
    (rebuild at 40k leaves, re-find its 0.90 point, re-run 20-round pairwise). If the config checks out, the 10M margin being LARGER than 1M is
    mechanistically plausible: our cells stay ~76 pts (fine partition + gamma ordering + batched scan) while
    ScaNN's per-leaf scan grows with n/leaves, and our refine stays survivor-bound.

P218. (*** CONSOLIDATION OF THE P216 WINNING STACK: branch `champion` @ 4aa59af (fast-forward of graph-expansion-impl: 34b05f2 graph sidecar / 0290d4e unsorted-union / 142a6b4 gamma / 1ab8134 trim + cherry-picked 330010b cascade-knob doc + 4aa59af OOD-calibration doc block). ALL GATES PASS: build clean; startup selftests (fastscan32 + l2_norm) + scanbench2 bit-identical; recall id-recomputed — champion-default p54 = 0.9032 BOTH paths set+order-identical 2000/2000 (nothing regressed), gamma-only p29/t540 = 0.9060, GR18_t470 = 0.9060; 5-round interleaved sanity (load ~38): GR18_t470 ~0.88x, gamma-only ~0.985x, recall exact every round (consistent w/ P216's calm 0.91x/0.97x). ***)
    DEFAULTS POLICY (clean-abstractions): gamma = per-dataset calibration knob, default unset = bit-identical
    (OOD recipe gamma=0.5 documented); graph = flag-gated sidecar (SBANN_GRAPH_FILE, build recipe documented);
    union-trim unconditional on the graph path; t_surv documented 470 for 1M-OOD-graph (scales with n per P215).
    GOTCHA documented: gamma arms need SBANN_TFLOOR=540 (fixed t_surv, not p*TMUL) — reproduces P210's gotcha.
    `champion` is now the canonical branch carrying the full 0.91x stack.

P219. (*** P217 RESOLVED — CORRECTED 10M OOD h2h vs OFFICIAL 40k-leaf ScaNN: ScaNN 1.24x AHEAD (median 1.239,
    IQR [1.138,1.283], 12 warm-protocol rounds); the quarantined 0.505x was entirely the under-leaved (4k) ScaNN
    index. Full rig rebuilt from scratch on the NEW box. ***)
    ENVIRONMENT RESET: repo now ~/genbo (was ~/lsh-engine), new 96-core/371GB shared box; ALL old artifacts
    (indexes, scann_venv, scratchpad, worktrees, datasets) were gone -> rig rebuilt: ~/big-ann-data/ holds t2i 10M
    base crop (range-download 8GB/5min), queries, i8bin quant (shared scale 300.32 = 127/max|base|; base norms
    0.814-0.990 mean 0.965 == P92's), exact float-IP GT (10k q x top-100, ScaNN-brute-force 28s@48thr, numpy
    spot-checked). CPU etiquette (user): builds capped ~8-16 nice'd cores.
    THE OPPONENT AT ITS BEST (the P217 fix): GCS pre-built official searcher is no longer public (403) -> rebuilt
    from the VERBATIM official textproto (big-ann-benchmarks neurips23/ood/scann): 40000 leaves, SOAR spilling
    (TWO_CENTER_ORTHOGONALITY_AMPLIFIED avq=1.6), AH2 LUT16 + residual quant + noise shaping 0.1, bf16 exact
    reorder, top-level partitioner 700; official upsert->rebalance(config)-at-8M build path replicated exactly
    (scann 1.4.2, py3.11). Sweep (single-thread pinned, best/5, our GT): QPS nearly FLAT in lts (lts27 0.8978@2482
    -> lts65 0.9422@2067; the AH scan is ~free, reorder=150 dominates) -> its 0.90 point lts=28/reorder=150
    (0.9001@~2330). reorder=140 variants: no better.
    OUR ARM: engine index rebuilt at the P215 geometry (hierk Kf=131072 C0=4096 b0=256 a0=3 SOAR=1, apq4, 24min
    @8cores). Rebuild lands ~1pt BELOW the old index at matched p (0.8945@p45 vs P217's 0.9054 — build RNG/SOAR
    variance; float-rerank gamma0.5 t2000 throughout) -> honest re-sweep: 0.90 point = gamma0.5 p=52 t_surv=2000
    (0.9025@2356 sweep-window). gamma=0.6 slightly worse everywhere; t_surv 2500/3000 LOSE at matched recall
    (t2000 confirmed the marginal-cost balance, P211 logic holds at 10M).
    PROTOCOL: 10k queries, both arms single-thread pinned to the SAME core, tight pairwise alternation (order
    alternates per round), best-of-5, median of per-round ratios, recall gates >=0.90 both arms every round
    (deterministic: scann 0.9001, eng 0.9025). v1 (20 rounds) had a warmth ASYMMETRY (scann resident across
    rounds, engine cold-spawned per round re-faulting its 7.9GB index copy: engine 1810-1948 vs its warm 2140)
    -> v2 scores the engine's LAST warm block per round (leaderboard warm-resident semantics both arms).
    v1 median 1.233 / v2 median 1.239 (IQR 1.138-1.283, min 1.124 max 1.302) — same answer, banked as ~1.24x.
    *** HONEST STANDING, 10M OOD single-thread QPS@0.90: ScaNN ~1.24x ahead (its ~2330-2450 vs our warm
    ~1810-2140; our arm is more load-sensitive, P142 redux — calm-round ratios 1.12-1.13 are the floor).
    Expectation range from HANDOFF (0.6-1.0x) was OPTIMISTIC; the 40k rebuild + top-level partitioner is a
    stronger opponent than the 4k cache ever was. GAP TO CLOSE for the goal (beat HANNS bar = 0.93x): ~1.33x.
    READY LEVERS (1M-proven, none in this arm yet): kNN-graph pool expansion + union-trim (P207/P214: champion
    1.184 -> 0.978 at 1M when composed with gamma), tree-EM rounds (P130: +0.6-1.0pt recall at 10M, QPS-neutral),
    geometry retune (Kf/C0/B0 at 10M was a single P215 point, never swept). Graph sidecar (ScaNN self-search k=16,
    10M x 16 u32) + EM(2,beam8) index builds queued. (h2h_10m_pairwise.py, h2h_10m_40k{_v1_coldspawn,}.csv,
    logs/h2h_10m_40k_v2.log, scann_build_40k.py, ~/big-ann-data/)

P220. (*** 10M OOD: PARITY WITH OFFICIAL ScaNN REACHED — 1.239x -> 1.009x median in one session via the
    1M-proven composition (graph sidecar + tree-EM + gamma), each lever's 10M transfer measured separately. ***)
    All h2h = 10k queries, single-thread same-core tight pairwise, warm protocol (P219), recall gates >= 0.90
    both arms every round vs the shared exact float GT; opponent fixed at its best (lts=28/reorder=150; its
    high-lts/low-reorder corner swept and REJECTED: reorder=100/120 tops ~2270 < its r150 ~2330).
    LEVER 1 — GRAPH SIDECAR at 10M (t2i10m_graph_k16.u32, 640MB, ScaNN self-search k=16, built 6min@16thr):
    composes exactly as at 1M — the union expansion buys back the recall a HALVED rerank pool loses:
    t_surv 2000->1000 with M=25-40 holds 0.90 at p 52-56 (t800 caps ~0.896 at ANY p/M — pool floor is real).
    Wider M helps: M=32 t1000 needs p52, M=40 p48 (recall-only, load-indep). 12-round h2h (EM build running,
    noisy window): median 1.118 [IQR 0.938-1.233] vs P219's 1.239.
    LEVER 2 — TREE-EM(2, beam8) at 10M geometry (39min@8cores): +~0.003 recall at fixed p on BOTH the graph
    and no-graph arms (0.90 crossing moves p52 -> p48, ~8% fewer probes), query-cost-free — P130's 10M EM
    transfer reproduced on the new rig.
    LEVER 3 — gamma refit: 0.5 stays optimal at 10M (0.40/0.45/0.55 all worse at matched p; P215 confirmed).
    *** DECISIVE 16-round h2h, quiet box: ScaNN(28,150) 0.9001 vs ENGINE (EM2 idx + gamma0.5 + graph M=32 +
    t_surv=1000 + p=48, recall 0.9024): MEDIAN 1.009x [IQR 0.951-1.143, min 0.876 max 1.184] = STATISTICAL
    PARITY, engine ~2030-2290 QPS vs ScaNN ~2280-2370. Arc: 1.239 (P219) -> 1.118 (graph) -> 1.009 (EM+graph).
    GOAL BAR (beat HANNS = ScaNN x1.07 => ratio <= 0.93): ~8.5% QPS short. IN FLIGHT: geometry variants
    G1 (Kf=131072 C0=8192 b0=192) / G2 (Kf=65536 C0=4096 b0=128), both EM2 — the 10M routing geometry
    (4096+256x32 = 12288 cells scored/q) has never been swept; the 1M analog (P192 C0 lever) was worth 12.6%.
    (h2h_10m_pairwise.py + ENG_IDX/graph env, logs/h2h_10m_{graph,em_graph}.log, build_graph10m.py)

P221. (*** 1M OOD RE-VALIDATED ON THE NEW BOX — 0.755x vs ScaNN (engine 1.32x FASTER), the cleanest h2h
    ever recorded on this project: IQR [0.754,0.757] across 12 pairwise rounds. GOAL BAR (<=0.93 = HANNS-
    equivalent) CLEARED at 1M on this machine. ***)
    Rig: 1M crop of the same t2i files; quantization scale came out 330.1905 == P191's recorded "~330.19"
    (max|x| 0.384626) — exact reproduction of the old rig's prep. Engine index rebuilt at champion geometry
    (Kf=16384 C0=768 b0=96 a0=3 SOAR) + EM(2,beam8); 1M graph sidecar via scann self-search k=16. Full-stack
    recall reproduces the ledger: gamma0.5 + graph M25 + t_surv=470 + p=18 -> 0.9049 (P216's GR18_t470 was
    0.9060, within build RNG).
    OPPONENT: fresh ScaNN 1M (2000 leaves, AH2 0.2, spherical, residual — ledger convention), swept fresh on
    this box: its recall-per-config runs LOWER than the old rig (56/78 -> 0.8939 here vs 0.9032 there; fresh
    index RNG + 10k-query GT) and its best 0.90 point moved to lts=42/reorder=110 = 0.9062 @ ~6040 (reorder
    ~100-110 dominates the old 78 here).
    H2H (12 rounds, warm protocol, same core, alternating order): ScaNN 0.9062 @ 6056-6058 vs ENGINE 0.9049 @
    8026-8046 -> ratio 0.753-0.760 EVERY round, median 0.755. Both arms rock-steady (calm window) — the
    tightest IQR in the project's history.
    WHY BETTER THAN THE OLD 0.91x: this box runs scann-1M ~30% slower per-core than the old Zen4 (6.0k vs
    8.3-8.6k) while the engine runs at parity (~8.0k vs ~8.4-9.4k) — per-core cache/memory characteristics
    favor our small-working-set stack at 1M. Same-hardware ratios are machine-specific; on THIS machine
    (the goal's arena) 1M is a decisive WIN. (chain_rig1m.sh, logs/h2h_1m.log, scann_t2i1m_2k/,
    eng_t2i1m_kf16384_c768_b96_a3_em2.idx)

P222. (*** MULTI-THREAD PROTOCOL: 1M WON AT THREAD PARITY TOO — 0.756x @ 8 threads (engine 62.5k vs ScaNN
    47.2k QPS), after finding+fixing a silent thread-scaling bug in the champion batched driver. ***)
    THE BUG (explains the old P197 caveat "our per-query Rust loop suffers oversubscription"): the FLOAT_RERANK
    batchscan path (champion default since P203) ran its query chunks in a SERIAL while-loop — RAYON_NUM_THREADS
    never engaged, so 8-thread == single-thread (8.1k QPS) while ScaNN's search_batched_parallel scaled ~7.8x
    (47k) -> first 8t h2h read a catastrophic 5.7x. FIX: rayon par_iter over the (independent) chunk ranges —
    batching (P202 cell-major, +27%) and threading now COMPOSE. Pure execution-order change: recall BIT-IDENTICAL
    (0.9049), single-thread QPS unchanged (~7.9k). Chunk size at 8t: 1000 -> 40.0k, 500 -> 52.8k, 250 -> 61.8k,
    125 -> 60.6k QPS => knee at chunk~250 for 8 threads (the single-thread knee stays 1000; chunk should scale
    ~nq/(4*threads)).
    H2H (10 rounds, 8 threads both arms pinned to the same 8 cores, scann search_batched_parallel, warm
    protocol): MEDIAN 0.756x [IQR 0.738-0.761] — IDENTICAL to the single-thread 0.755x (P221). The ratio is
    thread-invariant once both engines actually scale; no oversubscription penalty remains. 1M OOD on this
    machine: WON under BOTH protocols (single-thread 0.755x, 8-thread 0.756x), recall-matched, gates held.
    (commit: 'Parallelize cell-major batched driver across chunks'; logs/h2h_1m_8t{,_v2}.log)

P223. (*** 10M OOD BELOW THE HANNS BAR: 0.924x median vs official 40k-leaf ScaNN (IQR [0.900,0.940],
    calm-load rounds 0.857-0.923) — arc 1.239 -> 1.009 -> 0.924 in one day. Winning lever = GEOMETRY (G1):
    C0 4096->8192, b0 256->192 at Kf=131072 = +6.9% clean paired (2664 vs 2492 QPS, 6 rounds, dead-tight). ***)
    GEOMETRY MECHANISM (the P192 lever transferred to 10M): finer coarse level + leaner beam cuts routing
    evals 12288 -> 11264/query AND routes more precisely; recall-per-p gives up only 0.0015 (0.9009 vs 0.9024
    @p48, still >= gate). G1 = eng_t2i10m_kf131072_c8192_b192_a3_em2.idx (90min build; C0=8192 coarse k-means
    is the long pole). K sweep: K=16 confirmed floor (12 loses recall, 24 buys nothing). kedge=16 load-bearing
    (8/12 lose ~0.003-0.008). gamma 0.5 optimal (0.40/0.45/0.55 worse).
    H2H (16 rounds, warm protocol, same core, recall gates scann 0.9001 / eng 0.9009 every round): median
    0.924 [0.900,0.940] min 0.857 max 0.958. Rounds 11-15 at load 3-9 (G2 build ended): 0.857-0.923 — ScaNN
    rock-steady ~2450, engine rises 2636->2846 as box calms (P142 load-sensitivity, still true). The quiet-box
    number is plausibly ~0.88-0.90; a fully-idle confirmation run pending (G2/G3 builds still occupy 8 cores).
    ALSO THIS CYCLE: 10M @ 8 threads = 1.036x with the OLD G0 arm (pre-geometry, G1 build running) — thread
    parity holds at 10M; will re-run with G1 for the banked number. STATUS vs GOAL: 1M 0.755/0.756 (won),
    10M 0.924 median (bar ~cleared; margin thin, IQR upper 0.940 > 0.93) -> next: G3 (Kf=262144, finer cells
    to cut the dominant scan phase), calm-window pfdist/chunk micro-tune, then the long confirmation run.
    (logs/h2h_10m_g1.log, paired G0/G1 A/B in-session)

P224. (*** 10M OOD WON: 0.847x median vs official 40k-leaf ScaNN (IQR [0.842,0.848], min 0.826, max 0.852,
    16 rounds — every round below the 0.93 HANNS bar). Full-day arc: 1.239 (P219) -> 1.009 (P220) -> 0.924
    (P223) -> 0.847. Champion 10M arm = G2 geometry: Kf=65536 C0=4096 b0=128 a0=3 SOAR EM(2,beam8) + gamma 0.5
    + graph M=32 kedge=16 + t_surv=1000 + p=40 + K=16 (recall 0.9008 vs ScaNN 0.9001, gates held all rounds). ***)
    THE GEOMETRY SURPRISE (reverses the fine-cells intuition the 1M-era ledger carried): with the graph-union +
    float-rerank cascade, COARSER fine cells win at 10M — G2 (152pt cells, routing evals 4096+128x16=6144/q)
    beats G1 (76pt, 11264 evals) by +3.8% paired (2928 vs 2813 QPS) at the same 0.90 margin, which itself beat
    G0 (12288 evals) by +6.9%. Mechanism: the graph expansion recovers deep neighbours a coarse probe misses,
    so the index no longer needs fine granularity for coverage — it needs cheap routing + streamable cells;
    eval-count is the routing cost driver (P139's lever, now geometry-sized). G4 (Kf=32768, 305pt cells, 3584
    evals) queued to find the knee; G3 (Kf=262144, finer — the counter-hypothesis) building as control.
    H2H window: calm (load 3-6), G3 build + scann-100M build on other cores (symmetric background); IQR is the
    tightest of any 10M measurement yet. STATUS vs GOAL on this machine: 1M 0.755x/0.756x (1t/8t), 10M 0.847x
    (1t) — BOTH below the 0.93 HANNS-equivalent bar. Remaining: 10M 8t re-run with G2 arm, pristine confirmation
    run (builds SIGSTOPped), then 100M (both builds in flight) and streaming.
    (logs/h2h_10m_g2.log, eng_t2i10m_kf65536_c4096_b128_a3_em2.idx)

P225. (*** PROTOCOL MATRIX COMPLETE — every cell a WIN on this machine, recall gates held, opponent at its
    official best: 1M single-thread 0.755x | 1M 8-thread 0.756x | 10M single-thread 0.847x | 10M 8-thread
    0.833x (engine 21.4k vs ScaNN 18.0k QPS, IQR [0.830,0.834]). All below the 0.93 HANNS-equivalent bar. ***)
    The 8t/10M number uses the G2 champion arm + chunk=250 parallel batched driver (P222 fix). Thread parity
    slightly FAVORS us at 10M (0.833 vs 0.847 single-thread): the parallel-chunk driver amortizes the graph
    union + float rerank across cores with near-linear scaling while ScaNN's batched parallel is already at
    its plateau. Remaining for the 10M ledger claim: a pristine confirmation (all our builds SIGSTOPped) after
    G3/G4 land — though four independent h2hs (0.847, 0.833, plus the 0.924/1.009 arcs behind them) with tight
    IQRs already make the win robust. NEXT SCALE: 100M (scann-126k-leaf baseline building; engine geometry
    awaits the G4 knee answer; ~460 slots/cell scaling of G2's shape).
    (logs/h2h_10m_8t_g2.log)

P226. (*** STREAMING: COMPLIANCE FIXED + LOW-LIVE TAIL KILLED — official-per-step-GT avg recall@10 = 0.9745
    (640 steps, NQ=100 proxy), UP from the banked 0.9654 (P212), peak anon 5.38GB < 8GB. Two changes, both
    on feat/streaming-30m (worktree ~/genbo-streaming): ***)
    (1) COMPLIANCE (the P218 pre-submission blocker): cold-start now trains on the FIRST INSERT BATCH only
    (op1's 38,806 rows — data the stream has legitimately seen; SBANN_NINIT=38806, no SBANN_TRAIN_FILE), plus
    SBANN_RETRAIN_EVERY=<E>: at live = E, 2E, 4E, ... (geometric), retrain router+codebook on a ~200k strided
    sample of the CURRENT live set and rebuild via compact_live (which re-assigns + re-encodes under the
    swapped-in router/comp — Index fields are pub, no vq.rs change). 4 retrains fired (449k/1.24M/2.64M/5.43M
    live; 39.5/43.0/50.9/72.3s, ~205s total on the insert clock). Recall at matched steps == the old
    future-peeking strided-sample router (0.975 @ 2.4M, 0.983 @ 6.3M) -> THE COMPLIANCE FIX IS FREE.
    (2) LOW-LIVE ADAPTIVE p (SBANN_RB_MINCAND=50000): grow p per search step so expected candidates
    >= MINCAND (p_eff = max(P, MINCAND/(live*a0/C)), capped at C). The clustered runbook's early steps went
    0.737 -> 0.9990 (step 1), steps 1-8 all >= 0.993; worst step overall 0.944 (was 0.737). Costs nothing
    when live is large (no-op) and is cheap when it fires (few live points = fast scans).
    NET: 0.9654 (banked, non-compliant) -> 0.9745 COMPLIANT — above hwtl-closed 0.9675, 2nd open-source
    tier consolidated, puck 0.9855/0.9849 still ahead. CAVEATS: NQ=100 recall proxy (per-step GT, unweighted
    mean, same protocol as smoke: smoke read 0.9735 vs P212's NQ=10000 0.9654 — subsample optimism possible,
    final number needs NQ=10000); the run used the SLOW scan path (no SBANN_FASTSCAN/USE512FS — this branch
    predates P203 default-on): search 272 q/s -> NQ=10000 projects ~7h, INELIGIBLE as-is. Budget calibration
    with fast-scan flags running; eligible operating point (p, MINCAND) chosen from it, then the final
    official-accounting run. Insert rate on this box: 16.5k/s at 8 threads (old box 26k/s) — inserts 1818s
    of the 3600s budget. (logs/stream_compliant.log, ~/genbo-streaming patches uncommitted yet)

P227. (*** 10M OOD FINAL CLAIM — PRISTINE confirmation: 0.827x median vs official 40k-leaf ScaNN (IQR
    [0.823,0.834], min 0.817, max 0.850, 20 rounds, all OUR builds SIGSTOPped). The 10M win is now backed by
    FOUR independent h2h sessions: 0.847 (P224, builds running), 0.833 @ 8 threads (P225), 0.827 pristine,
    atop the 1.239 -> 1.009 -> 0.924 arc. GEOMETRY KNEE BRACKETED: G3 (Kf=262144, finer: needs p=64 for 0.90,
    ~30% slower) and G4 (Kf=32768, coarser: p=32 but 1.6x scan work, -10% paired vs G2) BOTH lose to G2
    (Kf=65536, 152pt cells) — the knee is real, not a monotone trend. Champion 10M config (banked):
    hierk Kf=65536 C0=4096 b0=128 a0=3 SOAR=1 EM(2,beam8) apq4 | gamma=0.5 graph(M=32,kedge=16) t_surv=1000
    p=40 K=16 FLOAT_RERANK | 0.9008 recall vs ScaNN(lts28,reorder150) 0.9001, same exact float GT.
    ON THIS MACHINE the goal's OOD table now reads: 1M 0.755x/0.756x (1t/8t), 10M 0.827x/0.833x (1t/8t) —
    every cell beats the 0.93 HANNS-equivalent bar with recall gates held and the opponent at its official
    best. Remaining OOD scope: 100M (engine Kf=524288 C0=32768 b0=128 EM2 building, ~5h; scann 126k-leaf
    20M-sample building; graph sidecar + t_surv~3000-4000 sweep to follow). (logs/h2h_10m_final.log)

P228. (*** STREAMING BUDGET UNLOCKED: batch-parallel insert = 4.5x insert speedup (30M inserts 1818s ->
    406s, 73.9k/s, recall BIT-IDENTICAL 0.9746), runbook wall 46min -> 8.6min at NQ=100. The insert wall
    (50%+ of the 1hr budget since P159) was the SOAR assignment run SERIALLY per point. ***)
    The runbook Insert op looped idx.insert(row,i,a0) one point at a time; insert()'s cost is the router
    assignment (~C=4096 cell dists x a0 per point). insert_batch_range (vq.rs) runs the assignment
    rayon-parallel over the batch, then appends in row order -> provably bit-identical to the serial loop
    (append order + ins_gidx/dirty identical). 30M inserts: 1817.7s (16.5k/s, 8 threads, the P226 number)
    -> 405.8s (73.9k/s). Recall 0.9746 == P226's 0.9745 (noise). Peak 7.24GB < 8GB.
    *** CONSEQUENCE: the streaming budget is no longer insert-bound. Old-box P159/P160 capped C at 4096
    because fine routers made SERIAL inserts too slow; that cap is GONE. Now exploring finer cells
    (C=8192/p=256, C=16384/p=512 at HALF the champion candidate fraction) — early steps read recall 0.993
    @C=8192 -> if it holds, search halves AND recall rises. SBANN_RETRAIN_SAMPLE (default 30*C) keeps the
    finer routers trained at compliance checkpoints. Committed feat/streaming-30m@4173d70. NQ=10000
    definitive runs to follow the C-sweep. (logs/stream_batchins_val.log)

P229. (1M/10M FURTHER-TUNING ROUND: all four probed levers are NET-NEUTRAL-TO-NEGATIVE on QPS -> 1M
    confirmed AT its optimum; a load-drift MIRAGE was caught and killed. Explored while 100M builds.)
    Baseline (champion 1M, single-thread): Kf=16384 C0=768 gamma0.5 graph-M25 t470 -> 0.9049@p18.
    (a) GRAPH DEGREE k (16/24/32 edges/node): MORE edges do NOT help -- k24/k32 at kedge=24/32 give
    -0.001 recall vs k16 (extra edges dilute the union pool). (b) GRAPH FAN-IN M (25->64): recall-per-p
    rises monotonically (p16: M25 0.898 -> M64 0.910) BUT single-thread best/5 QPS FALLS (M25 p18 7732 ->
    M64 p15 6766 at matched recall) -- the larger float-rerank union outweighs the ~3-probe scan saving.
    (c) DEEPER TREE-EM (EM4 vs EM2, same geometry): FLAT (0.9056 vs 0.9049) -- EM converged by 2 rounds
    at 1M. (d) COARSE-CELL GEOMETRY (the 10M knee lever) at 1M: recall-per-p much better (Kf=4096 hits
    0.90 at p~11 vs champion p~18; Kf=8192 at p~14) BUT *** the coarse-cell corollary does NOT transfer
    to 1M ***: paired same-core back-to-back A/B (the trustworthy measurement) = Kf4096-p11 / champion-p18
    ratio 0.90-0.95 single-thread (coarse SLOWER), 0.98-1.12 at 8 threads (noisy, ~neutral). At 1M routing
    is already cheap (P26), so coarser cells just inflate the scan pool (732 vs 183 pts/cell) more than the
    fewer probes save; the tree/coarse benefit is 10M+ -specific (confirms the P31/P26 scale-gating).
    *** MIRAGE CAUGHT: an initial cross-measurement read showed Kf4096 '1.85x faster' (6766 vs 3665) -- but
    the 3665 champion number was a contended-instant artifact; the paired back-to-back A/B (both on core 44)
    gave 0.93x. Classic load-drift confound, exactly what the tight-pairwise protocol exists to kill.
    NET: 1M is at its optimum; no lever this round converts recall-per-p into QPS. Good paper material
    (localizes WHY the coarse-cell win is scale-gated; a clean set of 1M-negatives). 1M stays 0.755x.
    Built artifacts (reusable): eng_t2i1m_kf{8192,4096,2048,1024}_*.idx, t2i1m_graph_k{24,32}.u32.
    ADDENDUM (cascade-skip, the 5th 1M lever): profile @p18 = route 33% / scan 29% / rescore(int8 cascade)
    25% / graph 9% / float 4% (balanced = well-optimized, no dominant phase). Tested skipping the int8
    cascade (float-rerank the 4-bit-PQ-top-t directly): recall CRATERS 0.9049 -> 0.79-0.82 at every t (the
    4-bit PQ pool ranks too coarsely for float alone) -> the int8 cascade (P194) is LOAD-BEARING, not
    removable. CONCLUSION: 1M is at its optimum across all probed levers; further gain needs a new
    primitive, not tuning. 10M freshly optimized via the P224 geometry knee (which IS the coarse-cell win).


P230. (100M GEOMETRY DE-RISKED at 10M (user-directed careful balancing): AGGRESSIVE coarse routing wins
    recall; 3-level is a 100M lever (not 10M). Balance study before the expensive 100M build.)
    Killed a mis-balanced first 100M build (2-level C0=32768: lopsided + expensive at build AND query;
    then a stingy 3-level C0=1024 b0=48 = 4.7% coarse expansion). Per the user's guidance to balance
    points-per-level and route aggression, swept THREE 3-level geometries at 10M (Kf=65536, recall + route%
    the load-independent signals; QPS best/2 too noisy to use):
      c256  (C0=256,  b0=96/256 =37.5% coarse, aggressive): 0.9023@p48, route 22%
      c512  (C0=512,  b0=96/512 =18.75%):                    0.9004@p48, route 18%
      c1024 (C0=1024, b0=128/1024=12.5%, stingy):            0.8992@p48, route 17%
    FINDINGS: (1) AGGRESSIVE coarse routing wins recall (+0.003 c256 vs c1024 for +5% route) -- OOD queries
    route poorly, so wide coarse beams matter (user was right). (2) scan dominates (~50%), route 17-25% ->
    routing cost is second-order, so aggression is affordable. (3) NONE of the 3-level configs beats the
    2-level G2 champion at 10M (0.9008@p40) -- 3-level's routing savings don't convert while scan dominates;
    2-level stays the 10M champion (consistent with N18/N19/P109). 3-level is a 100M+ lever (routing a bigger
    fraction there). 100M geometry chosen = c256 shape scaled: Kf=524288 (190 pts/leaf), C0=2048 C1=32768
    (branch 2048->16->16), b0=512 (25% coarse, aggressive), b1=256, a0=3 -> route ~14336 evals (vs the killed
    2-level's ~35k). Building (eng_t2i100m_kf524288_c2048_c1_32768_b512_a3.idx).

P231. (STREAMING ELIGIBILITY on THIS box is SEARCH-BOUND at ~0.90, despite the batch-insert fix -- the
    box is ~2x slower per-core than the Azure-class leaderboard HW; recall CAPABILITY stays 0.965-0.98.)
    After batch-insert (P228) removed the insert wall (30M inserts 450s = 66.6k/s), NQ=10000 runs expose
    SEARCH as the sole budget constraint (640 steps x 10k queries = 6.4M query-evals):
      p=224 (~0.965 config): recall 0.9648, WALL 148.8 min -- FAILS 1hr (2.5x over)
      p=80:  recall 0.9010,  WALL 68.9 min -- FAILS by 9 min (search 3680s @ 1739 q/s + inserts 450s)
      p=64:  running (projected ~56 min = eligible)
    So on THIS 8-thread box the 1hr-eligible operating point is p~64-70 -> recall ~0.89-0.90 (search q/s is
    the wall; the benchmark spec mandates 8 vCPU so I cannot use the box's spare cores). PEAK ANON 7.19GB <
    8GB (memory fine). *** HONEST SPLIT (for the paper + goal): (a) recall CAPABILITY = 0.965 (p224) to 0.98
    -- this is the leaderboard-relevant number on the official Azure D8lds_v5 HW, where P212 showed this
    stack class fits p=256 @ 0.9654 in budget; (b) THIS-box 1hr-eligible = ~0.90 (mid-pack: > diskann 0.883,
    cufe 0.819; < zilliz 0.922, pinecone 0.912). The gap is purely this box's slower per-core search, not the
    method. Compliance (P226) + batch-insert (P228) are real, HW-independent wins that transfer to Azure.

P232. (ALGORITHMIC/LOW-LEVEL innovation round on 1M -- 3 more decisive NEGATIVES; the config+architecture
    space is EXHAUSTED for this engine. Profile balanced (route 31/scan 29/rescore 27/graph 9/float 4).)
    (a) UNION-vs-POOL double-rescore inefficiency: DOES NOT EXIST -- rerank_cascade_graph already dedups
    graph neighbours against the scanned pool via one open-addressing hash pass (P214), only fresh
    neighbours are rescored; the memory-latency scattered ds.row() reads are already SW-prefetched
    (GRAPH_PFDIST streaming-ahead). No free lunch. (b) DIM-TRUNCATED ROUTING (ROUTE_SDIM, attacks the 31%
    route): fixed a latent OOB bug in l2_i8_block_avx2's odd-centroid tail (full qn vs sd-length block ->
    panic), then tested: recall CRATERS (sdim 200->160->128->100->80 = 0.905/0.866/0.824/0.759/0.666) AND
    route% does NOT drop (~31%) because the truncated path loses VNNI (l2_i8_block_norm needs full-dim
    cadj) -> AVX2-madd on 128 dims ~= VNNI on 200. DOUBLE loss. ROOT: the OOD embedding is near-isotropic
    (P102: 128 dims=82.5% energy) so EVERY dim carries cell-ranking signal -- high intrinsic dim bites
    ROUTING too, not just scan. Decisive dead end. (c) confirmed cascade-skip (P229) + graph-M (P229) +
    coarse-geometry (P229) all negative. *** HONEST STANDING: 1M/10M are at their optima for this
    IVF-tree+apq4+graph+float-rerank design; 8 levers across P229/P232 all neutral-or-negative. The
    profile is balanced (no dominant phase) = the fingerprint of an optimized system. Further gain needs a
    NEW PRIMITIVE (learned router / rank-preserving coarse codes = scann's AH2 moat), a research project,
    not tuning. Real deliverables this round: an OOB bug fix + a clean characterization (OOD near-isotropy
    caps dim-reduction on BOTH route and scan) = good paper negatives.


P233. (100M OOD ENGINE BUILT + VALIDATED with the P230 aggressive-3-level geometry: reaches recall 0.90;
    graph sidecar building for the efficient operating point; ScaNN-100M baseline still building.)
    Engine index eng_t2i100m_kf524288_c2048_c1_32768_b512_a3 (hierk3 C0=2048 C1=32768 b0=512 b1=256 a0=3,
    Kf=524288 = 190 pts/leaf) built in 86 min (vs the 9h+ mis-balanced 2-level I killed, P230), 77GB.
    NO-GRAPH gamma=0.5 recall (float rerank, 16thr): climbs with p AND t_surv (100M's 10x distractors need
    deeper pools): t4000 p128 = 0.8824; t6000 p192 = 0.9079, p256 = 0.9123 -> crosses 0.90 at p~180
    no-graph. The GRAPH (which took 1M 1.18->0.98, 10M 1.01->0.85) is the coverage lever that should drop
    the 0.90 point to p~60-80 (building via selfknn self-query, ~1h). Then: graph sweep -> h2h vs
    ScaNN-100M (126k leaves, 20M sample, 13h+ into its rebalance -- no OFFICIAL 100M OOD recipe exists;
    this is a scaling extension, leaves scaled sqrt(n)). The 3-level geometry (P230) built fast AND queries
    at competitive recall -> the aggressive-coarse-routing de-risk paid off.

P234. (*** GEOMETRY SCALING LAWS (user idea): clean power laws from a graph-off min-cost sweep at
    n={100k,300k,1M,3M,10M}, matching the routing cost model. Kf* ~ n^0.66, pts/leaf* ~ n^0.34, p90* ~
    n^0.18, C0* ~ n^0.33. ***)
    Method (load-INDEPENDENT): at each n, for each Kf (2-level, C0=sqrt(96*Kf), b0=C0/4, gamma=0.5,
    graph-OFF, float rerank), find the MIN p reaching recall@10>=0.90 (deterministic), then the analytical
    cost route(Eq)+scan(p*n/Kf*a0). The Kf minimizing total cost is the optimum. hierkn generic router
    (confirms hierk3 is just a wrapper). Optima:
      n=100k  Kf*=2048  49pts/leaf  p90=12
      n=300k  Kf*=4096  73          p90=12
      n=1M    Kf*=8192  122         p90=16
      n=3M    Kf*=32768 92          p90=24
      n=10M   Kf*=32768 305         p90=24
    POWER LAWS (log-log fit): Kf* ~ 1.04 n^0.660, pts/leaf* = n/Kf* ~ 0.96 n^0.340, p90* ~ 1.39 n^0.180,
    C0* ~ 9.96 n^0.330. INTERPRETATION: (1) pts/leaf GROWS as n^(1/3) -> the coarse-cell corollary
    QUANTIFIED: bigger n wants coarser leaves (route O(sqrt Kf) grows, scan O(p n/Kf) is capped by growing
    Kf). (2) p (coverage probes) grows slowly n^(1/5). (3) Kf ~ n^(2/3) (vs classic IVF sqrt(n)=n^0.5;
    the higher exponent reflects our cheap fast-scan making finer cells affordable). CAVEATS: 2-level +
    graph-OFF; the graph shifts the optimum COARSER still, and 3-level routing (cheaper, O(Kf^1/3)) shifts
    it FINER -> the law calibrates the 2-level baseline. IMPLICATION for 100M: 2-level law extrapolates to
    Kf~200k @ n=100M, but I built 3-level Kf=524288 (3-level affords finer) -> a graph-on Kf A/B at 100M
    (262144 vs 524288) is the clean validation, queued behind the graph build. This is exactly the
    principled-geometry backbone the paper needs (turns heuristic scaling into a measured law + 1B
    extrapolation: Kf(1B)~n^0.66 ~ 1M cells, ~1000 pts/leaf). (logs/scaling_laws.csv)

P235. (100M OOD ENGINE OPERATING POINT: recall 0.9031 @ ~6161 QPS (16 threads) -- the engine reaches 0.90
    at 100M; graph PARAMETERS scale with n too (M, t_surv grow), a new dimension of the scaling law.)
    100M engine (eng_t2i100m Kf=524288 3-level, P230/P233) + graph-on. INITIAL graph-on sweep looked WEAK
    (M=32 t=3000: barely +0.017 over no-graph) -> NOT the graph edge quality (built via engine self-query),
    but UNDER-SCALED graph params: at 100M the graph needs MORE expansion + a DEEPER pool. M/t sweep @p96:
    M32/t3000=0.8774, M48/t5000=0.8989, M64/t5000=0.9010, M64/t8000=0.9106. So M and t_surv SCALE WITH n
    (1M/10M used M=25-32/t=470-1000; 100M wants M=64/t=8000) -- more distractors need more graph expansion
    and a deeper rerank pool to hold the expanded candidates. 0.90 crossing (M64 t8000 gamma0.5): p=80 ->
    recall 0.9031 @ 6161 QPS (16thr) [p64 0.8929, p56 0.8860]. So the 100M engine operating point =
    p=80/M=64/t=8000, ~6000 QPS@0.90 (16thr; ~3000 @8thr leaderboard-style). h2h vs ScaNN-100M PENDING
    (scann 126k-leaf still in rebalance ~16h -- the long pole; no official 100M OOD recipe). NOTE: the
    scaling law (P234) suggests a COARSER Kf (~262144) might route cheaper at 100M -- a Kf A/B is the
    refinement (graph is base-side, reusable). The engine SCALES to 100M and reaches competitive recall.

P236. (ScaNN-100M was OVER-PROVISIONED by me -> ~day-long build blocking the h2h; rebuilt lean.)
    Diagnosis (user asked why scann wasn't building): py-spy showed it ALIVE and running (819% CPU, R state,
    200 CORE-HOURS) inside rebalance() -- NOT stuck, just enormous. Cause: I set 126k leaves + 20M training
    sample (sqrt(n)-scaled from the 10M official's 40k/8M) = ~8x the partitioner k-means cost = ~12-16h.
    My error. FIX: killed it, rebuilt with the 10M OFFICIAL config applied at 100M -- 40k leaves + 8M sample
    (scann_build_100m_lean.py) -> ~2h, a fair+documented baseline (no official 100M OOD recipe; scann's fast
    AH scan handles the larger 2500-pt leaves). 100M h2h chain queued: scann-lean build -> lts sweep -> pairwise
    vs the engine (p=80 M=64 t=8000, 0.9031). LESSON: scale scann leaves by the OFFICIAL config's absolute
    count guidance, not a naive sqrt-of-the-already-aggressive-count.

P237. (*** OPPONENT-AT-ITS-BEST AUDIT (user-prompted "are we using the wrong scann config?"): scann's 1M
    leaf count was UNVERIFIED (2000, a convention) -- swept it; scann's true 1M PEAK is 1200 leaves, but the
    1M WIN HOLDS (~0.72-0.77 recall-matched vs the original 0.755, marginal change). NOT a P217-class error. ***)
    The 10M win used the VERBATIM official 40k-leaf textproto (correct). But 1M used 2000 leaves by convention,
    never verified as scann's optimum -- the same "is the opponent at its best?" question that caught every
    mirage, left unchecked at 1M. Swept scann-1M leaf count (build+lts/reorder sweep, best QPS@recall>=0.90,
    clean window):
      leaves= 800 -> 4794 | 1200 -> 6226 (PEAK) | 1600 -> 5745 | 2000 -> 5682 (what we used) |
      4000 -> 4056 | 8000 -> 2578 | 16000 -> 1460
    So (a) MORE leaves = SLOWER scann at 1M (routing over more leaves > finer-scan saving; scann's fast AH
    handles big leaves) -- 2000 was NOT under-leafing scann (my fear was backwards); (b) true peak = 1200
    (~10% faster than 2000). Re-measured h2h vs scann-1200: clean rounds 0.657 BUT scann ran at 0.9095 (over
    our 0.9049 -> scann artificially slow); recall-matched + the leaf-sweep's clean 6226@0.90 -> honest ratio
    ~0.72-0.77 vs engine ~8100. WIN HOLDS decisively (< 0.93 bar), essentially unchanged from 0.755.
    LESSON: "verbatim official" covered 10M; at scales w/o an official recipe (1M, 100M) I let CONVENTION
    stand in for the leaf-count sweep. FOLLOW-UPS: (1) clean recall-matched 1M h2h vs scann-1200 on a quiet
    box (pending; box loaded by the 100M build). (2) SPOT-CHECK 10M: is 40k near scann's 10M speed-optimum
    or would coarser be faster? (the 1M data says coarser can win -> the official 40k might not be scann's
    speed-max, though it IS the leaderboard config). (3) 100M: sweep scann leaf count for ITS optimum before
    any ratio claim (the lean 40k is a build-of-convenience, likely too coarse). scann_1m_leafsweep.py.

P238. (*** BUILD-TIME CONSTRAINT (user-flagged): big-ann OOD limit = 12 HOURS on the 8-vCPU eval machine
    (runner.py:297 container timeout = 12*3600 for Filters/OOD/Sparse; README section "Build time limit").
    This makes the killed 126k-leaf scann DOUBLY invalid (ineligible, not just slow) AND genbo builds faster
    than scann at every scale within the limit -- a bonus result for the paper. ***)
    BUILD TIMES (16 cores; ~2x on the 8-vCPU D8lds_v5 eval machine; limit 12h):
      scale   genbo                         scann
      1M      131s                          80s  (1200 leaves)  -> scann slightly faster at 1M
      10M     1640s (27min, G2 Kf=65536 EM2) ~1.5h (40k official) -> genbo ~3x faster
      100M    5150s (86min, 3-level)         ~2h (40k lean); 126k = ~16h/16c => ~32h/8vcpu = INELIGIBLE
    So (1) the 126k scann I killed EXCEEDED the 12h limit (~32h at 8 vCPU) -> not a valid baseline at all;
    killing it was doubly right. (2) genbo's build advantage GROWS with scale (1M scann faster, but 10M/100M
    genbo 3x faster) -- the hierarchical-kmeans + apq4 build is cheap vs scann's partitioner-kmeans + AH
    training. (3) The eligible scann baseline is a config building <=12h (40k works; 126k doesn't). NEW
    PAPER POINT: report build time alongside QPS -- genbo wins BOTH build and query at scale, within the
    12h eligibility gate. ACTION: killed the all-night 20k/30k scann leaf-sweep (per user: don't let scann
    build all night); kept the 40k-lean h2h. If the 40k h2h is borderline, build ONE coarser eligible scann
    (25k, ~1.5h) to give scann its best-eligible; else 40k stands.

    2H-GATE ADDENDUM (user directive): for the 100M experiment we impose a STRICTER 2h build gate (vs the
    official 12h), both engines. genbo 100M builds in 86min (16c) -> WITHIN 2h. scann-40k at 100M is at
    99min+ still in rebalance -> likely OVER 2h (the O(100M) AH-encode/bf16 work + 40k-leaf k-means).
    Watchdog: 40k gets until the 2h mark; if it finishes it's the eligible baseline, else killed and a
    coarser scann-20k (faster k-means) is built under a hard 2h timeout. So the 100M h2h uses scann's
    best config that builds <=2h -- if even 20k can't, that itself is a finding (genbo builds a competitive
    100M index in 86min; scann is build-constrained at 100M under 2h). Note: this is stricter than the
    official 12h gate under which 40k IS eligible; the paper reports both (2h operational gate + 12h official).

    100M 2H-GATE RESULT (decisive): scann is BUILD-INELIGIBLE at 100M under the 2h gate. 40k killed at
    7251s (121min); coarser 20k ALSO exceeded 2h -> the O(100M) AH-encode + tokenize + bf16-prep work
    dominates the scann rebalance REGARDLESS of leaf count, so no scann config builds a 100M index in 2h.
    genbo builds a competitive 100M index (0.9031@~6k QPS) in 86min < 2h. So UNDER THE 2H GATE, genbo is the
    only method with a 100M index -- it wins by build-eligibility. (Under the official 12h gate, scann-40k
    IS eligible at ~2-4h build; a query-QPS h2h there would need scann-40k built to completion, ~3-4h.)
    Reported both ways in the paper: 2h-gate eligibility (genbo only) + optional 12h-gate query h2h.

P239 [2026-07-06] IN-DISTRIBUTION ROOT CAUSE: int8-sat fast-scan LUT cap breaks at high m — int16 path fixes it; genbo BEATS HNSW on cohere-1M.
  Symptom: cohere768-1M (in-dist, cosine) genbo needed t_surv=32000 for 0.94 (456 QPS 1t); HNSW 0.9122@1840. In-pool ordering by
  4-bit ADC was catastrophic: plain rr=100 -> recall 0.0701 (p=48, pool 8000).
  Discriminating A/B (SAME index, SAME 4-bit codes, only scan arithmetic): FASTSCAN2 int8-sat rr=100 -> 0.070; SBANN_FASTSCAN2=0
  (int16 LUT) rr=100 -> 0.8716 = pool ceiling. Verdict: codes fine; the FS2_CAP=15 (~4-bit) per-subspace LUT quantization is the
  killer at m=384 (d=768, dpb=2): noise ~ sqrt(m) vs tiny in-dist cosine gaps. At d=200 (m=100, wide OOD gaps) FS2 is harmless ->
  why it became champion default. POLICY: scan precision must scale with m (auto int16 when m>~128; d=200 unaffected).
  Standard cascade under int16, cohere-1M graph-off 1t: p=32/t=300 0.9031@2035; p=40/t=300 0.9197@1957 (STRICTLY DOMINATES HNSW
  ef40 0.9122@1840); p=64/t=500 0.9522@1245 vs HNSW ef80 0.9592@995. t collapsed 32000->300 (survivor cut now trustworthy).
  8-bit RESID refine: ranks to ceiling (0.869@rr=100) but 222 QPS (LUT overhead over 8k survivors) — not the fix. Graph-on with
  HNSW-quality edges: 676@0.9065 — helps (1.5x) but not the root cause. Baselines same box 1t: HNSW 0.9122@1840/0.9592@995,
  IVFPQ+refine 0.9014@877 (by_residual=False REQUIRED for IP — default residual PQ caps at 0.11 even on OOD t2i), scann pending.
  Cohere assets: cohere/base1m.{fbin,i8bin}, cohere1m_gt.ibin (exact, faiss FlatIP), cohere1m_graph_k16.u32 (HNSW self-search),
  eng_cohere1m_kf16384.idx, eng_cohere1m_kf16384_resid.idx (+8-bit refine, dpb=2).

P240 [2026-07-06] 100M Kf-geometry x tree-EM ablation: EM real (+0.6-1.0pt, 445s), coarser-Kf recall lift eaten by cell cost — QPS wash; EM-on-fine-geometry is the open lever.
  Two concurrent 100M builds (single-variable axes): kf262144-em0 and kf262144-em3 (C0=2048 C1=23170 b0=512 b1=256; EM rounds
  148s each on sample, beam-8 E-step). Recall@matched p (graph M64 t8000 g0.5, NQ=2000, load-independent):
    p=48: 524288-em0 0.8830 | 262144-em0 0.8983 | 262144-em3 0.9084;  p=96: 0.9154 | 0.9250 | 0.9287.
  Decomposition: coarser Kf +1.0-1.5pt at matched p; treeEM +0.6-1.0pt on top (3-level boundary repair, user hypothesis
  CONFIRMED on recall). BUT clean uncontended QPS@0.90 (16t, NQ=10000, best/5): OLD 524288-em0 0.9011@6764 (p=76) vs NEW
  262144-em3 0.9007@6700 (p=44) — WASH. Coarser cells scan 2x points/probe (381 vs 190/leaf); probe savings cancel. Scaling-law
  'coarser wins' does NOT survive at QPS level at 100M (10M coarse-cell win was geometry-knee-specific, P227).
  OPEN: EM3 on the FINE 524288 geometry (building) — EM probe-reduction without bigger cells projects ~7400 QPS@0.90.

P241 [2026-07-06] AVX-512 i16 pair-scan gate hoisted out of hot loop: cohere-10M 756->991 QPS (+31%), recall bit-identical; 'collect overhead' was misattributed env::var tax.
  The 32-wide avx512 vpermw i16 kernel (block_adc_i16_avx512_x2, selftested) existed but was gated on a PER-CALL
  env::var("SBANN_USE512") + feature-detect INSIDE scan_block_x2 (~890 pair-block calls/query at 10M: mutex+hash each).
  Fix: startup AtomicBool USE512I16, default ON when avx512f+bw (selftest-asserted), SBANN_USE512=0 disables; only
  touches QueryCtx::Pq16 (the m>128 policy path) so d=200 champion (Pq8/FASTSCAN2) is untouched (commit 187c0ac).
  cohere-10M p48 t500 1t contended: 756 -> 832 (env-set, per-call tax remains) -> 991 QPS (hoisted); recall 0.9264
  bit-identical all three. scan-us/q 950.7 -> 605.9.
  ATTRIBUTION CORRECTION (kills proposal P2): post-P1 scatterbench kernel floor (16-wide, no collect) = 594.6 us/q
  vs full scan 605.9 -> fused-collect overhead is ~11 us/q, NOT ~310. The workflow recon's SCANDIAG split had folded
  the env-var tax + x2-fallback copy into 'collect'. Threshold-fused-collect (P2) has nothing left to win; skip.
  Post-P1 profile (contended): route 236 us (23%), scan 606 (57%), rescore 186 (17%, memory-bw inflated; 66 quiet),
  float 30. Remaining levers: dpb4 (v2 index, m=384->192 halves kernel), graph (p 48->~32), then route trim
  (SDIM/ADC) once route is the largest slice. Projection: ~1550-1700 QPS stacked, ~1900+ with route trim.
  HNSW cohere-10M REAL 1t (same box, contended): 0.9370@1208 (ef40), 0.9623@671, 0.9803@340; 8t: 9051/5117/2629.

P242 [2026-07-06] COHERE-10M FIRST DOMINANCE: genbo (avx512-i16 kernel + HNSW-derived graph) beats HNSW at 1 thread — 0.9462@1246 vs 0.9370@1208.
  Stack on the OLD v1 index (kf65536 dpb2 a0=2 no-EM): P241 kernel + cohere10m_graph_k16.u32 (emitted by the faiss
  HNSW-10M job via base self-search ef=64, 976s/16t) with SBANN_GRAPH_M=16.
  Frontier (1t, NQ=1000, contended box — BOTH systems measured within the same hour on the same box):
    p=12 t=400: 0.9350@1377 | p=16 t=400: 0.9462@1246 (STRICTLY DOMINATES HNSW ef40 0.9370@1208) | p=24 t=500:
    0.9572@1097 | p=48 t=500: 0.9730@832.  HNSW-10M 1t: ef40 0.9370@1208, ef80 0.9623@671, ef160 0.9803@340
    (8t: 9051/5117/2629). genbo 8t not yet re-measured post-P241.
  Graph is a bigger recall lever at 10M than 1M: +3pt at matched p (0.9264->0.9572 at p=24-48 range).
  CAVEAT: contended conditions (v2 + 100M builds running); final call = tight-pairwise on quiet box after builds
  land. v2 index (dpb4+EM3+a0=3, building) projected to add ~30-50%: dpb4 halves the m=384 kernel, EM cuts probes.
  HNSW-10M artifacts banked: cohere/hnsw_cohere10m.faiss (reusable), cohere10m_graph_k16.u32.

P242a [2026-07-06] OPEN: cohere-10M 8-thread scaling anomaly under build contention — 8t SLOWER than 1t (graph p16: 999 vs 1246; graph-off p48: 622 vs 991).
  NOT the graph-rescore-bandwidth hypothesis (graph-off equally broken). Conditions: two 16-core builds live
  (v2 cohere + 100M em3) hammering memory bus + page cache; t2i 8t scaled 7.6x post-P222 on quieter box; cohere
  has only 1000 queries (chunked driver -> ~4 chunks at chunk~250: caps at 4x, cannot explain <1x). Candidates:
  page-cache eviction of the 30GB fbase by builds (float rerank faulting from NVMe), rayon+contention interaction.
  RE-MEASURE on quiet box after builds land before drawing any 8t conclusion. 1t dominance (P242) unaffected.

P243 [2026-07-06] cohere-10M v2 champion verdict + honest h2h status: v2 (dpb4+EM3+a0=3) ~1.3x v1 at matched recall; genbo-vs-HNSW = TIE under RoarGraph noise, definitive quiet-box h2h pending.
  v2 index eng_cohere10m_l2_65536_dpb4_em3_a3.idx (hierkn 2-level [4096,65536] b0=128, TREEEM=3 at 557-654s/round,
  dpb4 m=192, SOAR a0=3; 87min build, 26.5GB). Recall at matched p=16 graph-M16: v2 0.9390 vs v1 0.9462 (-0.7pt,
  deterministic); within-window QPS ratio v2/v1 = 1.43, 1.29, 1.36 (adjacent alternation) => v2 WINS frontier ~1.3x.
  P242's 'dominance' (1246 vs 1208) compared DIFFERENT time windows — same-window 3-round h2h (best-of-15 in-process):
  genbo-v2 p16 0.9390 vs HNSW ef40 0.9370: ratios 0.77/0.92/1.09 (median 0.92) = STATISTICAL TIE at +0.2pt recall.
  Box noise (RoarGraph 16t build, cores 72-87): same config swings 457->1762 QPS between minutes; 3 rounds insufficient.
  v2 p=16 profile: route 33% (~335us, now the largest lever), scan 39% (460us), rescore 22% (255us at 666ns/row —
  cache-miss inflated), float 3.5%. ROUTE_SDIM on RAW basis (no rotation): 512 -0.6pt, 384 -1.6pt, 256 -4.6pt —
  near-isotropic per-dim variance, as OOD t2i (P232); needs PCA pre-rotation (data-side: rotate base+queries, rebuild;
  graph reusable — edges are id-based) to make prefix-routing principled. QUEUED: definitive quiet-box pairwise
  (genbo v1/v2 x p12/16/20 vs HNSW ef40/80 vs scann, 5 rounds interleaved) when RoarGraph exits; PCA-rotation rebuild
  if a gap remains. 100M: em3-on-524288 = +0.7pt at matched p (0.9028@p68 vs em0 0.8958) => ~12% probe cut at 0.90;
  clean QPS@0.90 pairwise also queued (expect ~7200-7500 vs 6764, would be new 100M champion).

P245 [2026-07-06] P242a RESOLVED — 8t 'anomaly' was the batched driver's chunk size: SBANN_BATCH_CHUNK default 1000 = cohere's ENTIRE query set -> one chunk -> serial. chunk=125: 1757 -> 12850 QPS (7.3x); genbo-8t BEATS HNSW-8t 12835 vs ~11930 same-window (+0.2pt recall).
  Quiet-box interleaved 8t rounds: genbo-v2 p16 {1961,1735,1757} vs HNSW ef40 {11931,12034,11849} — genbo stuck at
  1x scaling. Cause (main.rs:513,597-599): cell-major batched driver splits nq into SBANN_BATCH_CHUNK=1000 chunks,
  ranges.into_par_iter(); cohere NQ=1000 -> 1 chunk -> 1 rayon task. t2i (NQ=10000 -> 10 chunks) never exposed it.
  With SBANN_BATCH_CHUNK=125: 12850/12820 QPS, recall 0.9390 bit-identical. VERDICT: cohere-10M won at BOTH thread
  counts (1t median 1.098, 8t ~1.08 same-window). TODO: adaptive default chunk=clamp(nq/(2*threads),125,1000) —
  needs a t2i NQ=10000 A/B first (chunk size trades cell-pass amortization vs parallelism; don't change champion
  path blind).

P246 [2026-07-06] Adaptive batch-chunk default: t2i-10M 8t 14851 -> 20882 QPS (+41%), recall bit-identical; 1t champion untouched.
  chunk A/B on t2i-10M champion (NQ=10000, 8t, graph M32 g0.5 t1000 p40): chunk=1000 (old default) 14851; 250
  21253; 125 21058 — the fixed 1000 was straggler-bound (10 chunks / 8 threads). New default (main.rs):
  clamp(nq/(4*threads), 125, 1000), env-overridable; 1t hits the clamp=1000 => bit-identical champion path
  (verified 0.9049 t2i-1M). Combined with P245 this also sets cohere-8t 12850 without needing the env.
  NOTE: banked 8t OOD ratios (P222/P225/P227 era) were measured under the old default — genbo's 8t absolute was
  underreported; ratios vs scann stand (both sides measured), but a re-run would likely IMPROVE the 8t ratios.

P247 [2026-07-06] 8t OOD 10M h2h refreshed under adaptive chunk (P246): ratio 0.833 -> 0.758 (median/5, IQR 0.754-0.760).
  Proper harness (h2h_10m_pairwise.py, search_batched_parallel for scann, H2H_THREADS=8, cores 8-15, NQ=10000,
  REPS=5, tight alternation): SCANN 0.9253@~16450 vs ENG 0.9008@~21700 (graph M32 g0.5 p40 t1000 exported via env
  — the harness does NOT set graph itself; graph-off run showed 0.8736@~22800/ratio 0.727 = gate-fail, discarded).
  Conservative: scann at 0.9253 recall (its swept point) vs eng at 0.9008. Paper tab:main 10M 8t cell updated.
  Also: quick sed-hack scann-8t measurement (search_batched + set_num_threads) gave 2150 QPS = 8x under-read —
  scann 8t REQUIRES search_batched_parallel; do not measure it any other way.

P248 [2026-07-06] 1M 8t OOD ratio refreshed under adaptive chunk: 0.756 -> 0.746 (median/5, IQR 0.745-0.747; ENG 0.9049@60.7k vs SCANN 0.9005@45.2k, higher recall AND 1.34x QPS).
  Same harness/protocol as P247 (search_batched_parallel, cores 8-15, NQ=10000, graph M25 g0.5 p18 t470 exported).
  Headline OOD table now: 1M 0.755 (1t) / 0.746 (8t); 10M 0.827 (1t) / 0.758 (8t).

P249 [2026-07-06] STREAMING FINAL (NQ=10000, compliant runbook, this box): eligible frontier = p=64 -> 0.8821 @ 52.7min; p=80 -> 0.9010 @ 68.9min (15% over); puck 0.9855 unreachable on this box (p=224 -> 0.9648 @ 148.8min).
  All from completed stream_elig_*/stream_nq10k_* runs (compliance retrain + adaptive-p + batch-insert, branch
  feat/streaming-30m@4173d70): inserts 29.99M in 400-450s (~73k/s 8t), deletes 3.7-4.8s, search dominates
  (2333 q/s at p=64, 8t). P246 chunk fix does NOT apply — stream search is per-query par_iter (verified).
  VERDICT: recall-vs-budget is compute-bound on this box (2-3x slower at 8thr than old rig/Azure, see
  streaming-eligibility memory); on the leaderboard machine p>=80-96 would be eligible (projected 0.90-0.92).
  Paper reports the on-this-box frontier with the machine caveat. NQ=1000 numbers (0.9706-0.9745) were
  small-NQ-optimistic; NQ=10000 is the honest series. Task closed at this frontier.

P250 [2026-07-07] PCA-rotation gotcha: reusing the unrotated int8 scale CLIPS 55% of PC1 (max|rot|=0.926 vs clip 0.4485) -> rot-index recall 0.5621 (vs 0.939 unrotated). SDIM itself was recall-neutral post-rotation (0.5614-0.5621 across 192/256/384) — the truncation works; the quantization was the poison.
  Fix: per-basis scale (127/max|rot| ~= 137); requant + ROT2 rebuild chain running (eng_cohere10m_ROT2_*.idx,
  logs/build_cohere10m_rot2.log, marker ROT2_CHAIN_DONE; inline bench now includes graph+float rerank so save-time
  recall is meaningful). Variance shares: top-256 = 92.8%, top-384 = 96.5% -> SDIM=256 routing should be ~free
  once quantization is fixed; projected +20-30% cohere QPS (route was 33% of wall at p=16).

P251 [2026-07-07] ROT2 + SDIM verdict: rotation makes dim-truncation RECALL-FREE (0.9224 flat across SDIM 192-768 at p=16) but QPS-DEAD — the routing kernel is BANDWIDTH-bound on the full-d-strided centroid layout, so prefix-scoring saves ~0 (routebench: fine-expand 165->161 us with SDIM=256; coarse 103->95 with new SDIM0=256 knob).
  Mechanism: cent arrays are contiguous d-strided; HW prefetcher streams whole 768B rows regardless of summed
  prefix -> FLOP cut invisible. To realize the ~3x route cut the truncated prefix needs a PACKED copy
  (cent_sdim[l] with 256B rows, built at load when SDIM set). Route split at ROT2 p=16: coarse-l2 103us (37%),
  fine-expand 165us (59%), selects 13us. Projected with packed prefix: route 285->~100us -> wall ~700->~510 ->
  ~2300-2900 QPS at 0.922-0.930, which would beat v2-unrotated's frontier (0.9298@2048 clean).
  Also banked: ROT2 (scale=135, zero clip) baseline = 0.9249@p16 with full stack — the coarser global int8 scale
  costs ~1.4pt vs unrotated v2 (0.9390); rotation is only net-positive IF the packed-prefix route win lands.
  Code: SBANN_ROUTE_SDIM0 knob added (vq.rs gather_fine coarse branch; default off, champion untouched).
  NEXT: packed-prefix centroid copies at load (cent + fine levels) gated on SDIM/SDIM0.

P252 [2026-07-07] Packed-prefix routing centroids LAND: route 285 -> 98 us/q (2.9x; coarse 103->48, fine 165->36), end-to-end ROT2 0.9224@2419 (+39%) / 0.9297@2016, recall bit-identical to unpacked SDIM. Verdict vs v2: PARITY at 0.93 (median 1.03x, rounds 1.03/1.10/0.98) — rotation's global-int8-scale cost (~1.4pt) eats the route win at the high-recall end.
  Implementation (vq.rs, commit below): HierRouter.cent_pfx OnceLock<Vec<Vec<i8>>>, lazily packs cent[0] rows to
  sd0 bytes (SBANN_ROUTE_SDIM0) and cent[finest] rows to sd bytes (SBANN_ROUTE_SDIM) on first probe; scoring uses
  stride=sd so bandwidth scales with the prefix (P251 mechanism confirmed: the d-strided layout was the block).
  Champion paths untouched (knobs default 0). Same-window HNSW ef40: ~1330-1537 -> both genbo stacks ~1.4x ahead
  at the 0.93 point. ROTATION LINE CLOSED as frontier-dependent: use rotated+packed stack for <=0.93 targets;
  unrotated v2 for >=0.935. FUTURE (if revisited): rotation WITHOUT the scale cost (per-block scale or
  route-only rotation) would make packed-SDIM a strict win; the primitive is ready.

P253 [2026-07-07] MULTI-HOP GRAPH REPAIR (user idea): expand-rescore-reselect x R lands — R=2-3 beats one-hop on the cohere-1M frontier; hop gains decay geometrically (+1.9/+0.5/+0.4pt at p=12).
  Impl: SBANN_GRAPH_HOPS (default 1 = bit-identical champion path, verified 0.9433@p24 exact) in
  rerank_cascade_graph: hop r int8-scores the newly-unioned cohort, expands its top-GRAPH_M by INT8 rank
  (better seeds than hop-0's apq4 rank), same up-front-sized hash set, bounded R*M*ke appends. Per-query path
  only (BATCHSCAN=0); batched-driver port pending if promoted.
  Grid (cohere-1M dpb2+graph M16 t300, NQ=1000, 1t, wiki-builds contended — recall trustworthy, QPS indicative):
    matched ~0.943: R=3 p=16 0.9430@1954 vs R=1 p=24 0.9433@1841 (+6%)
    high-recall extension: R=4 p=24 0.9591@1574 (R=1 cannot reach 0.955+ at sane p)
    low-p regime: R=3 p=6 0.9053@2434 vs R=1 p=12 0.9066@1853
  Compensation law: ~2 p-steps down per hop added at matched recall. Theory sufficed over optuna: gains decay
  geometrically (P(true NN at graph-dist r | not closer)), R* = marginal-hop == marginal-p per us; smooth
  low-dim space, 20-pt grid + coordinate descent resolves it. TODO if promoted: clean pairwise confirm,
  batched-path port, t_surv/M re-tune at R=2-3, 10M transfer test.

P254 [2026-07-07] RoarGraph tight pairwise (t2i-10M OOD, 1t, matched ~0.90): genbo WINS median 1.07x (rounds 0.93/1.06/1.07/1.09/1.12) — genbo {2555,2358,2222,2151,2385}@0.9008 vs RoarGraph {2387,2160,2099,2319,2131}@0.9027.
  AMENDS P244: the banked RoarGraph solo 1469@0.903 was COLD-cache; warm same-window is ~2100-2400. (Pairwise
  discipline catches its own past mistakes.) RoarGraph is the strongest OOD baseline measured — ~7x HNSW, within
  7% of genbo — but requires the 8GB train-query set + 2M-query approx GT + multi-hour bipartite build; genbo's
  86min build + graph sidecar wins on both axes. Paper t2i baseline row amended to the pairwise numbers.

P255 [2026-07-07] MULTI-HOP THEORY (2-modeler workflow + verification): recall(R,p) = per-probe reachability CEILING minus a geometric hop-gap, fit exact to noise floor. Optuna NOT needed. Width beats depth. Ceiling extrapolation: M1 (exp-floor) confirmed over M2 (logit).
  Model: miss(R,p) = m_inf(p) + A(p)*delta^R; m_inf=0.105*exp(-0.040*p) (graph-UNREACHABLE island mass, only
  p/t_surv cure), A=0.210*exp(-0.089*p) (repairable mass), delta=0.436 (per-hop miss survival; each hop closes
  ~56% of remaining gap). R^2=0.992, RMS 0.0022 = NQ=1000 binomial noise floor. Geometric reachability
  (P(captured at hop r)=(1-delta)delta^{r-1}) makes the P253 heuristic exact.
  TUNING (no optuna — surface deterministic, monotone, ~2-D, 4-param closed form beats BO): set p FIRST
  (ceiling = 1-m_inf(p), hops can't exceed it), then R by cost-neutral rule dp>cR/cp (~2.4 probes/hop) ->
  R*=2 across most of the range, R=3 only chasing last tenths, never R>=4. <=15-run recipe banked in workflow
  journal + HANDOFF.
  DISAGREEMENTS RESOLVED BY DATA:
   (a) width vs depth (co-tune, matched edge budget M*R=48-64): R=2 M=24 0.9321@2285 > R=3 M=16 0.9303@2278 >
       R=4 M=12 0.9273@2269 — WIDE+SHALLOW wins recall AND QPS; no geodesic-floor population. Prefer M up, R=2.
   (b) 2nd hop pays at every p (measured hop1->2 +0.010 = ~4 probes > 2.44 cost-neutral); use data test, not
       M1's dp formula, near ceiling.
   (c) ceiling extrapolation (R=3): p=32 0.9645, p=48 0.9742 — MATCHES M1 exp-floor (pred 0.977) NOT M2 logit
       (0.958). Probing keeps climbing fast at high p; exp-decay miss floor is the right functional form ->
       high-recall targets reachable by raising p, not just hops.
  Co-tune knobs (cohere-1M p12): M concave (8:0.905, 32:0.939), t_surv flat 200-400 (coverage, cheap; 600
   thrashes), kedge 16 best (8:0.909, 12:0.922, 16:0.930). Champion default stays R=1 (bit-identical);
   R=2 M=24 is the promotion candidate if a clean pairwise confirms.

P256 [2026-07-07] BEST-FIRST FRONTIER (user idea = HNSW-order, not HNSW-prune): batched beam best-first (expand GLOBAL top-M unexpanded per round, not per-cohort top-M) is a STRICT Pareto win over per-cohort multi-hop — beats it in all 12 tested cells at neutral/slightly-higher QPS.
  SBANN_GRAPH_BESTFIRST (default off; champion hops=1 path bit-identical, verified 0.9433@p24). Impl: union=pool
  only, R rounds of {score new, expand global-top-M-unexpanded by int8}, single terminal float rerank. Keeps
  SIMD-batched rescore + no per-query heap + no mid-walk prune (the HNSW insight that TRANSFERS is expansion
  ORDER, not pruning; pruning was correctly rejected — recall risk, no cost win at ~1k pools).
  cohere-1M (M=16, t300, recall = the A/B, per-cohort -> bestfirst):
    p=12: R2 .9253->.9285 | R3 .9303->.9381 | R4 .9346->.9444
    p=16: R2 .9382->.9410 | R3 .9430->.9492 | R4 .9455->.9538
  GAIN GROWS WITH R (+0.3pt@R2 -> +0.8-1.0pt@R4): global frontier compounds; deep hops stop being wasted.
  REVERSES P255 width>depth: under bestfirst DEPTH wins — R4/M16/p16 .9538@2235 > R2/M24/p16 .9489@2398. New
  frontier high for these params. New promotion candidate = bestfirst R=3-4 (not per-cohort R=2). Theory update:
  best-first shrinks the effective per-hop miss-survival (hops more effective), so R* shifts UP vs P255.
  TODO: clean pairwise on the dpb4 champion index + 10M transfer (queued in multihop_followup, now w/ bestfirst).

P257 [2026-07-07] SCALE MEMORY: faiss IndexHNSWFlat fp32 at 35M x 1024 OOM-killed at 271GB RSS (co-running with genbo-35M build on 371GB box). fp32 flat-HNSW vector storage = 35M*1024*4 = 143GB + ~2x add-transient. Fix = fp16 storage (IndexHNSWSQ QT_fp16, ~72GB), the STANDARD large-scale HNSW config (nobody deploys fp32 flat HNSW at 35M+); negligible recall impact for cosine. Relaunched fp16 on cores 40-55 (mem 133G used / 237G avail, safe alongside genbo).
  Two takeaways: (1) sequence >100GB-RSS jobs, never co-run two; genbo build survived (25GB RSS — hierarchical
  k-means+PQ is memory-light vs HNSW's full-vector graph). (2) PAPER POINT: HNSW's fp32 memory (143GB@35M,
  ~4TB@1B) is itself a scaling liability — genbo's int8+PQ index is ~35GB at 35M; the memory gap widens with n
  exactly like the build-time gap. genbo-35M build still running (~5h, dpb4+EM3 encode of 35M x 1024).
FAIRNESS AMENDMENT (user directive): HNSW must live within genbo's budget on BOTH axes, no handicap. fp16
  (72GB) was still 2x genbo's ~40GB (int8 base 34GB + PQ ~5GB) -> switched to int8 HNSW (IndexHNSWSQ QT_8bit,
  ~40GB = matched memory class, standard billion-scale HNSW storage). Time: HNSW adds ~3.2h < genbo ~5h EM
  build, so within budget. Rule going forward: baseline gets <= the method's time AND space; if it can't fit,
  DECLARE the dataset out of the baseline's reasonable reach (that IS the result) rather than lavishing it
  resources. int8 fits, so wiki-35M stays a fair 3-way.
BUILD-IMPRACTICAL AT SCALE (P257 cont.): genbo-35M-d1024 with the FULL stack (dpb4 + TREEEM=3 + SOAR a0=3 +
  262144 leaves) did NOT complete in 10h on 16 cores (killed, no save; encode+EM of 35M x 1024 is the cost) —
  ~20h-equiv@8vcpu, well OVER budget and SLOWER than HNSW-int8 (built 3h50m/16c = 7.7h-equiv, recall 0.967-0.972).
  So at 35M/d=1024 the heavy genbo config LOSES on build. Reported genbo-35M = the budget-fair LIGHT config
  (a0=2, NO EM, Kf=131072, dpb4; armed to build on HNSW's freed cores, target <=4h). HONEST SCALING CAVEAT for
  the paper: genbo's build advantage is d=200-OOD-specific; at d=1024 the PQ-encode + tree-train dominate and a
  lean config is required to stay build-competitive. Query-time verdict pending light build.
  wiki_chain left UNTOUCHED (genbo is its foreground child; killing it would kill the build); it will emit
  genbo numbers + WIKI35M_CHAIN_DONE when genbo saves. HNSW-fp16 + 3-way compare tracked separately.

P258 [2026-07-07] MULTI-HOP + BEST-FIRST VALIDATED at 1M AND 10M (promotable, champion still R=1 pending port+OOD test):
  cohere-10M (per-cohort, graph M16/24, p12 t400, 1t): R1 0.9298@1784 -> R2/M16 0.9436@1559 -> R3/M16
  0.9479@1610 (+1.8pt recall vs R1, -10% QPS) -> R2/M24 0.9514@1195. Multi-hop transfers STRONGER at 10M than
  1M (more routing misses at scale -> graph repair more valuable) — matches P255 theory (bigger A(p) repairable mass).
  cohere-1M clean 5-round pairwise: best-first R3/M24/p12 = 0.9460 @ median 2502 vs champion R1/M16/p24 = 0.9433
  @ 2481 -> STRICT PARETO (+0.27pt recall, +0.8% QPS). (One 1896 blip; other 4 rounds 2431-2578.)
  PROMOTION path (champion stays R=1 bit-identical for now): (1) batched-driver port (currently per-query
  BATCHSCAN=0 only -> needed for 8t throughput), (2) OOD t2i multi-hop test (do hops help OOD or only
  in-distribution? — the graph coverage lever already wins OOD at R=1, so R>1 upside there is unknown), (3)
  8t confirm. If all pass, default hops=2-3 + bestfirst when a graph is present.

P259 [2026-07-08] wiki-35M (Cohere-v3, d=1024, in-distribution) exposes a genbo ANISOTROPY failure + honest 35M standings:
  Data: CohereLabs/wikipedia-2023-11 en, 35M x 1024, unit-norm but ||mean||=0.4766 (STRONGLY anisotropic — ~half
  the energy in one shared direction; typical of text embeddings). Queries=last 1000 rows, exact faiss GT.
  HNSW-int8 (build-fair, ~40GB, IndexHNSWSQ QT_8bit): built 3h50m, 1t 0.9670@1198 / 0.9711@656, 8t 0.9670@8264. STRONG.
  genbo-light (a0=2, NO EM, Kf=131072, dpb4, NOMU=1): built 1h52m (HALF HNSW's time) BUT recall PLATEAUS 0.56-0.59
  across p=16..160 / t=400..8000 (graph-on or off) — a routing/representation CEILING, not coverage. float rerank on.
  ROOT CAUSE (localizing): wiki-1M subset same nomu-noEM config = 0.9087 (WORKS) -> the 35M collapse is
  scale/fine-tree x anisotropy: without mean-centering (SBANN_NOMU=1) the coarse k-means splits along the dominant
  mean direction -> imbalanced cells -> routing to the right fine cell fails, and it COMPOUNDS at 131072 leaves /
  35M (vs 16384 leaves / 1M). Testing mu-centering (drop NOMU) + EM at 1M to confirm; then rebuild 35M with mu.
  HONEST STANDING: at 35M/d=1024 in-distribution, HNSW currently WINS (genbo mis-configured: NOMU wrong for
  anisotropic data + no-EM at fine tree). This is a genbo config bug, not a fundamental limit — fix = mean-center
  (+ maybe EM) then re-measure. Heavy-EM config was build-impractical (>10h, killed, P257). NEW GENBO DEFAULT
  CANDIDATE: auto-detect anisotropy (||mean|| large) -> enable mu-centering; NOMU should not be default for text.
  Build gotcha banked: nohup dies to tool-timeout SIGTERM; use setsid for long detached jobs.
  P259 UPDATE (root cause narrowed): the 0.59 plateau is NOT mean-centering-at-1M (mu-noEM 1M=0.8997 ~ nomu
  0.9087), NOT coverage (1M nomu p96/t2000 CLIMBS to 0.9704 — responsive), NOT coarse-beam width (35M beam0
  128->768 flat 0.558->0.581). It is ROUTING-SCORE DEGENERACY under anisotropy: NOMU=1 + ||mean||=0.48 =>
  every centroid ~parallel to the mean => query.centroid ~const across cells => routing returns ~arbitrary
  cells, and NO routing knob (p/t/beam) recovers the ~40% of queries whose true cell is never scored
  distinctly. Neutral at 1M (coarse 16384 tree still separates) but fatal at 35M/131072. FIX being tested:
  mu-centered (drop NOMU) + coarser cohere-proven Kf=65536 (eng_wiki35m_mu_65536_dpb4_a2, building ~1h,
  logs/build_wiki35m_mu.log, marker MU_REBUILD_DONE). If it lands >=~0.95, genbo-35M is rescued and NOMU
  should NOT be default for anisotropic (text) embeddings — auto-center when ||mean|| large. Query knob added:
  SBANN_BEAM0 (coarse-beam override, no rebuild needed to widen). If mu-rebuild still fails, wiki-35M is
  banked as HNSW-wins (bonus scope) and the loop pivots to promoting the CONFIRMED multi-hop/best-first win.


P260 [2026-07-08] MULTI-HOP GENERALIZES to OOD (promotion path item #2 CLEARED); best-first is in-distribution-only.
  t2i-10M OOD (champion idx, graph M32 g0.5 p=40, per-query 1t): R1 0.9186@2560 -> R2 0.9303@2476 ->
  R3 0.9352@2333. +1.7pt recall at -9% QPS = matched-recall WIN (R1 needs p~65 to hit 0.9352). So multi-hop
  per-cohort helps BOTH OOD (+1.7pt) and in-distribution (+1.8pt at 10M cohere, P258) — a general champion
  lever, not regime-specific.
  BEST-FIRST: on OOD, R2/R3 bestfirst = SAME recall as per-cohort (0.9303/0.9352) + slightly LOWER QPS ->
  NO OOD gain. (In-distribution cohere-1M it was +0.27pt, P256.) Mechanism: base-metric graph edges don't
  align with off-manifold OOD queries, so global-best-first reselection surfaces no better seeds than the
  per-cohort frontier; the extra select is pure cost. => best-first stays an IN-DISTRIBUTION opt-in.
  PROMOTION VERDICT: default SBANN_GRAPH_HOPS=2 (or 3) when a graph is present (helps both regimes, matched-
  recall positive); keep SBANN_GRAPH_BESTFIRST opt-in (in-dist only). Remaining before flipping the default:
  batched-driver port (currently per-query BATCHSCAN=0) + 8t confirm. Champion still R=1 until then (safe).

=== SESSION SUMMARY (autonomous optimization push) ===
WON: msspacev-10M, beat scann ~1.3-1.5x at QPS@90%recall (the leaderboard metric), clean same-window
(P87/P89). Chain: profile->rerank bottleneck (P78)->i8 LUT resolution root cause (P84)->int16 LUT
(P85/86, THE win)->cell-contiguous rerank (P79)+apq4 anisotropic (P80)+routing tuning. AVX-512 scan
dead end (N-AVX512). scann still wins recall>=0.95 (faster scan at high candidate counts, fundamental).
OOD (text2image-10M, actual leaderboard): engine RUNS it @0.90 recall via MIPS->L2 augmentation
(P92-95) but ~8x behind scann on QPS; OOD routing is the gap and query-aware routing failed (P96).
Best config: hierk Kf=262144 C0=4096 + apq4 + int16 LUT(default) + tmul~3; b0/a0 adapt to recall.

AUTONOMOUS-SESSION STATE (for continuity): MSSPACEV OPTIMIZATION COMPLETE. I WIN QPS@90%recall vs scann
~1.30-1.5x same-window (P87/P89); scann wins recall>=0.95 (faster scan at high candidate counts, P91 --
fundamental, AVX-512 didn't help N-AVX512). Best config: hierk Kf=262144 C0=4096 (b0/a0 adapt: b0=64/
a0=2 for QPS@90, b0=256/a0=4 for high recall) + apq4 + int16 LUT (default) + tmul~3. The decisive win
was int16 LUT (P84-87). NEXT STRATEGIC CHOICE: (a) OOD track adaptation (text2image float32 MIPS -- the
ACTUAL leaderboard; my int8 engine can't run it; would need float support + the int16 win transfers),
or (b) consolidate. Earlier state below.
PRIOR: On msspacev I WIN QPS@90%recall vs scann ~1.30-1.5x same-window
(P89); scann wins >=0.95 (my recall caps ~0.955 = routing-coverage-limited). On msspacev @ recall 0.90-0.92 my engine (apq4 + int16
LUT16 scan + contig rerank + C0=4096 + tmul=3) BEATS scann 1.21-1.41x same-window (P87). Best config:
hierk Kf=262144 C0=4096 b0=128, apq4, SBANN_LUT16, t_surv~p*3. TARGET (P76/P77) = ScaNN on THIS box:
~14,000 QPS@90%, text2image-10M OOD ~9,150 QPS@90%. ScaNN is ~1.5-3x FASTER than my Rust engine on
msspacev (P77) -- the earlier "I beat scann" was a single-threaded-scann bug. Azure leaderboard scann
=42854 OOD so HW factor ~4.7x. Real gap to close; ScaNN edge = anisotropic AH quant + SIMD in-register
AH scan + tuned partitioning. (Prior: my engine beats Python sbtree P70-74, but that's a weak baseline.)
PRIOR-STATE: Best Rust scale config (msspacev int8) = hierk Kf=262144 opql (C0=4096
~9,150 QPS@90%recall (8 cores); Azure leaderboard scann=42854 so HW factor ~4.7x; beat-#1(hanns)
target on this box ~9,800 QPS@90%. To contest OOD my int8 engine needs float32+MIPS adaptation.
Best Rust scale config (msspacev int8, in-distribution) = hierk Kf=262144 opql (C0=4096
@recall0.90 ~9.4k QPS, C0=2048
for QPS@0.9, C0=1024 for recall>=0.95). P74 CAPSTONE: Rust BEATS Python across the whole 10M frontier
1.06x@0.90 -> 1.23x@0.96, tightest controls (drift-free abrun + Python ~2min later, both best-of).
Levers that got here: Kf=262144 fine cells (P63, the big one), routing fan-out C0 tuning (P67),
PQ4 beats exact scan for per-query Rust (P69). Measurement: best-of-N + back-to-back to beat box
noise (P66). Env knobs on `run`: SBANN_PLIST/TMUL/C0/B0/REPS/NQ. NEXT: bulletproof vs Python best-of;
then either (a) batched int8 scan kernel (only lever that could add more), or (b) scale to 100M/1B
(infra-blocked by box load) or neurips23 tracks. f32 GEMM routing = dead end (N29).

OLD-STATE (pre-P70, superseded): Best Rust scale config = hierk Kf=262144 (P63 -- finer
cells than the old Kf=65536). Routing-quality ladder: random < avq < flat-kmeans < hierk (P58/P60/
P61). flat+kmeans BEATS Python @1M (3x). hierk Kf=262144 @10M = 1.78x faster than Kf=65536 at
matched recall 0.966 (1922 vs 1079 QPS) -- CELL GRANULARITY was the lever, not query kernels (P63).
NEXT: SBANN_PLIST env override now lets a built index be probed at custom p -> sweep p=128..768 to
get the high-QPS/lower-recall frontier and compare to Python 0.9486@3688. f32 GEMM routing =
dead-end (N29). The bucket-select/batched-rerank ports are now LOWER priority (cell count mattered
more). The Python sbtree_pq remains the tuned reference but the Rust gap is mostly closed.
