# DEEP low-recall: measured status and next experiments

## Current boundary

The physical-layout sweep evaluated 2,216 screening configurations and confirmed
the selected policies on all 2,000 queries. Strict two-round ABBA/BAAB comparisons
against RoarGraph give:

| Recall (genbo/Roar) | genbo QPS | Roar QPS | Result |
|---|---:|---:|---|
| 0.9039 / 0.9031 | 4,630 | 5,952 | Roar 1.29x |
| 0.9265 / 0.9255 | 4,449 | 4,920 | Roar 1.11x |
| 0.9426 / 0.9430 | 3,625 | 4,049 | Roar 1.12x |
| 0.9603 / 0.9606 | 3,037 | 3,147 | Roar 1.04x |
| 0.9718 / 0.9718 | 2,587 | 2,554 | genbo 1.01x |

Thus ordinary policy tuning removes most of the old deficit and moves the
crossover to about 0.97, but it does not close the recall-0.90 corner.

The tuned 0.9039 profile is route 30.1%, PQ scan 19.5%, graph bookkeeping 4.3%,
int8 union rescore 20.1%, and float rerank 26.0%. The union is only 235 rows/query.
The old proposal to optimize a 1,600-row, three-hop union is no longer aimed at
the live bottleneck.

## Ranked next experiments

1. **Resident fp16 rerank on the tuned path (cheap, highest immediate EV).**
   `SBANN_RERANK_F16` directly attacks the 26% float stage and has already passed
   a DEEP-1M parity check. Gate it on DEEP-10M at the five confirmed policies;
   require recall loss <=0.0005 and at least 8% paired QPS gain.

2. **Coarse-beam override sweep (zero new engine code).**
   Routing is now the largest stage. Sweep `SBANN_BEAM0` around the built value
   while holding each tuned policy fixed. A useful result preserves recall within
   0.0005 and cuts route time by at least 15% (about 5% end-to-end).

3. **Separate graph seeds from pool eligibility (small engine change).**
   The scan keeps 192 rows, but only `M=8` of them seed the graph at recall 0.904.
   Add `SBANN_GRAPH_POOL_KEEP`: use the full APQ pool to choose seeds, but admit
   only its best 64/96/128 rows to exact union scoring and float eligibility.
   First gate containment from result/union dumps; proceed only if 96 rows retain
   at least 99.5% of the current final top-10.

4. **Confidence-dispatched effort (small/medium, query-adaptive).**
   Start every query at the M8/floor192 policy, and promote only ambiguous queries
   to M16 or M24 using router-margin, scan-margin, or frontier-margin features.
   Unlike the killed generic hop-stop rule, this dispatch chooses among measured
   Pareto policies before paying their work. Gate offline with per-query results:
   match fixed-policy recall while sending fewer than 30% of queries upward.

5. **Batched routing as GEMM or tiled VNNI (medium, throughput-oriented).**
   Batchscan still routes queries independently. Pack coarse/fine centroids once
   and score a query tile against them, reusing centroid cache lines across the
   batch. The target is route 100us -> <=55us and >=12% end-to-end at unchanged
   cell ids. This will not help single-query latency but directly helps QPS.

6. **Pipeline routing and cell scan across query tiles (medium).**
   While one tile scans its selected cells, route the next tile and prefetch its
   first cell blocks. This attacks the combined 50% route+scan wall without
   changing candidate sets. Require bit-identical results and >=8% QPS.

7. **Supervised cell reranking (novel, higher risk).**
   The structural goal is to reach recall 0.904 with roughly 10 probes instead of
   15. Train a small held-out-query correction over router score, parent score,
   cell occupancy, and query-centroid margin; rerank only the existing fine-cell
   shortlist. Gate offline before Rust: same true-neighbour cell coverage at
   <=2/3 of the probes. This is the first idea that changes candidate efficiency,
   rather than only executing the same policy faster.

8. **Physical page-cohort incremental walk (novel, currently lower priority).**
   Revisit only if the stages above stall. In the tuned path graph bookkeeping is
   4% and the graph suffix is small, so a full walk rewrite has less headroom than
   route/rerank work. A trace oracle must first show >=15% total latency headroom.

## Do not repeat without new evidence

The following have already failed on DEEP in the relevant regime: RBQ/PQ4/SQ4
navigation, partial-dot bounds, generic adaptive hop/pool stopping, graph-primary,
cell portals, QSEED/HUBSEED, route dimension truncation, coarse-cell rebuilds,
Vamana diversification/reverse edges, pure int8 walk, and SYMPACK packed walk.
