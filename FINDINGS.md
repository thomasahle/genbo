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
