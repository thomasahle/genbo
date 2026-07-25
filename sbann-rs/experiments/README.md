# Post-boundary experiment gates

These scripts reproduce the P346 tests without changing the champion path.
They intentionally use absolute paths to the local big-ANN corpus so the
recorded JSON configurations are directly executable on the measurement host.

## Cheap gates

Run the projection, exact-bound, and query-cohort gates on DEEP-1M:

```sh
/home/thomas-ahle/scann_venv/bin/python gate_next_rungs.py \
  --out /home/thomas-ahle/big-ann-data/deep10m/next_rungs_gate_nq500.json
```

The script requires NumPy and FAISS. Its exact top-2100 candidate set is a
conservative adversarial union: all rows are genuine near neighbours.

## Cell-local portals

Dump the loaded index's assignment and exact router output:

```sh
SBANN_INDEX_LOAD=/home/thomas-ahle/big-ann-data/deep10m/eng_deep10m_kf65536.idx \
  ../target/release/sbann dumpassign \
  /home/thomas-ahle/big-ann-data/deep10m/deep10m_assign.u32

SBANN_INDEX_LOAD=/home/thomas-ahle/big-ann-data/deep10m/eng_deep10m_kf65536.idx \
  ../target/release/sbann dumproute \
  /home/thomas-ahle/big-ann-data/deep10m/query2k.i8bin \
  /home/thomas-ahle/big-ann-data/deep10m/routes_p8_nq500.u32 8 500
```

Then run the oracle and build the full sidecar:

```sh
python3 gate_cell_portals.py --nq 500 --portals 16 --keeps 1,2 \
  --out /home/thomas-ahle/big-ann-data/deep10m/cell_portal_gate_i8.json
python3 build_cell_portals.py
```

The engine path is opt-in through `SBANN_PORTAL_FILE`; `SBANN_PORTAL_KEEP`
defaults to one. It also requires the existing graph cascade.

### Adaptive portal walk

P353 reuses the portal partition only to choose graph-walk entries, bypassing
the PQ scan and union. Build its portal-order, 64-byte-padded SQ4 tier:

```sh
python3 build_portal_sq4.py
```

With `SBANN_PORTAL_FILE`, `SBANN_PORTAL_SQ4_FILE`, the fine-centroid graph, and
the jointly relabeled graph/base loaded, `SBANN_PRESET=fast` selects the
measured `L=25..88` loose ladder automatically. The full configuration and
strict RoarGraph comparison are executable from:

```sh
python3 abba_bench.py abba_deep_portal_walk.json --rounds 2 \
  --out abba_deep_portal_walk_results.json
```

`gate_deep_portal_representatives.py` records the fixed-representative rejects;
`gate_deep_portal_sq4.py` isolates entry quality before online routing cost.

### Graph-free loose-recall alternatives

The P354 gates test whether the adaptive point walk can be replaced by routed
block streaming:

```sh
python3 gate_deep_graphfree_coverage.py --nq 2000
python3 gate_deep_graphfree_scan.py --nq 2000
python3 gate_deep_support_tiles.py --nq 2000
python3 gate_deep_tile_diffusion.py --nq 500
python3 gate_deep_bucket_diffusion.py --nq 500
python3 gate_deep_learned_tiles.py
```

The first script also gates residual product enumeration and sparse component
voting; the support-tile script includes exact portal-local sphere bounds. The
consolidated verdict and exact budgets are in
`deep_graphfree_alternatives_results.json`. The strongest direct path is
executable in the engine with `SBANN_PORTALSCAN=144`,
`SBANN_PORTAL_SCAN_CELLS=128`, and `SBANN_PORTAL_SURVIVORS=128`;
`SBANN_PORTAL_BATCH=1` enables its cell-major scheduling A/B. Both are
diagnostic flags—the measured adaptive walk remains the `fast` default.
The strict same-window comparison is:

```sh
python3 abba_bench.py abba_deep_graphfree_vs_walk.json --rounds 2 \
  --out abba_deep_graphfree_vs_walk_results.json
```

## Paired measurements

`abba_bench.py` selects and pins the least-busy physical core (including its
SMT sibling in the activity score), warms both arms, alternates ABBA/BAAB, and
records every raw result plus load, faults, and context switches:

```sh
python3 abba_bench.py abba_deep_portals.json --rounds 2 \
  --out /home/thomas-ahle/big-ann-data/deep10m/abba_portals_gate.json
```

The prefetch and resident JSON files are positive-control and storage-backend
controls for the same runner.

## Graph-local physical layout

First dump an exact engine union trace with `SBANN_DUMP_UNIONS`, then derive a
query-independent cell-pair permutation and materialize the int8 base and graph
under that same id relabeling:

```sh
python3 gate_graph_layout.py \
  --materialize-layout cellpair --materialize-graph \
  --sort-graph-neighbors --base-offset 64
```

At search time, supply all three matching artifacts:

```sh
SBANN_GRAPH_BASE=/home/thomas-ahle/big-ann-data/deep10m/graph_layout_gate.cellpair.aligned64.i8bin \
SBANN_GRAPH_BASE_OFFSET=64 \
SBANN_GRAPH_RANK=/home/thomas-ahle/big-ann-data/deep10m/graph_layout_gate.cellpair.u32 \
SBANN_GRAPH_FILE=/home/thomas-ahle/big-ann-data/deep10m/graph_layout_gate.cellpair.graphsorted.u32 \
SBANN_RESIDENT_I8=1 \
  ../target/release/sbann run ...
```

`unionbench` replays the graph suffix of a `GUN1` trace without routing or graph
bookkeeping. `abba_deep_graph_layout_aligned.json` measures the complete engine
against the unchanged id order. `abba_deep_layout_vs_roar_090.json` additionally
contains named `--variant` points from recall 0.90 through 0.991 for direct
RoarGraph ABBA comparisons.

## Low-recall policy sweep

`sweep_deep_low_recall.py` pins one physical core, checkpoints after every loaded
index, and sweeps probes and cascade widths inside each process. The broad DEEP
screen is:

```sh
python3 sweep_deep_low_recall.py \
  --hops 1,2,3 --beams 4,8,12,16,24 --edges 8,16,24,32 \
  --floors 320 --cascade-widths 32 \
  --out /home/thomas-ahle/big-ann-data/deep10m/deep_low_recall_sweep_coarse.json
```

Use the unsorted relabeled graph when `SBANN_GRAPH_KEDGE` is below 32: sorting a
row preserves its complete edge set but would change which prefix is selected.
The named `tuned_r0903` through `tuned_r0972` variants in the Roar configuration
are the full-query ABBA confirmations selected from the sweep.

## Post-retune experiments 1--6

`sweep_deep_next.py` reproduces the retained fp16 correction-depth, coarse-beam,
and batch-chunk screens. It can also dump final result ids and the diagnostic
confidence CSV needed by `gate_deep_dispatch.py`:

```sh
python3 sweep_deep_next.py fp16-refine --values 10,12,16,24,48 \
  --nq 2000 --reps 2 \
  --out /home/thomas-ahle/big-ann-data/deep10m/deep_next_fp16_refine.json

python3 sweep_deep_next.py beam0 --values 0,32,48,64,128 \
  --nq 2000 --reps 2 \
  --out /home/thomas-ahle/big-ann-data/deep10m/deep_next_beam0_confirm.json
```

`abba_deep_next.json` defines the five tuned policies for strict engine-internal
ABBA. `abba_bench.py` accepts repeatable `--a-env`, `--b-env`, `--a-unset`, and
`--b-unset` overrides, so the same configuration can compare correction depths
without copying JSON. The complete verdict and artifact names are in
`DEEP_LOW_RECALL_NEXT.md`. Pool trimming, tiled routing, and route/scan prefetch
were removed after failing their gates; they are documented rather than left as
inactive production branches.

## Supervised cell-reranking gate

`gate_deep_cell_rerank.py` prepares a raw top-128 route-feature dump, trains on
50,000 disjoint RoarGraph DEEP training queries, chooses model capacity and epoch
on a separate 10,000-query validation tail, and holds the public 2,000 queries
out of all model and policy selection:

```sh
python3 gate_deep_cell_rerank.py --prepare \
  --ranks 0,4,8,16 --geometry-models diag,full --epochs 12 \
  --out /home/thomas-ahle/big-ann-data/deep10m/deep_cell_rerank_gate.json
```

The diagnostic `dumproutefeat` format contains the normalized query and, for
each candidate, its fine/parent ids, distances, norms, and occupancy.
`dumproutermeta` records the finest centroids for shared geometric models.
Neither command is called by normal search. The gate failed, so no learned
reranker was added to the Rust hot path; exact results are recorded in
`DEEP_LOW_RECALL_NEXT.md`.

## Ten structural low-recall gates

`deep_big_ideas_results.json` is the compact ledger for the subsequent
centroid-router, query-memory, evidence-jump, conditional-selector, trace-edge,
edge-sketch, page-slab, block-bound, query-hypergraph, and inverted-multi-index
gates. The only engine-internal survivor is the opt-in finest-centroid graph
router. Build its graph and landmarks from the loaded index's metadata, then
reproduce its strict hierarchy comparison:

```sh
SBANN_INDEX_LOAD=/home/thomas-ahle/big-ann-data/deep10m/eng_deep10m_kf65536.idx \
  ../target/release/sbann dumproutermeta \
  /home/thomas-ahle/big-ann-data/deep10m/deep10m_router.rcm
python3 build_centroid_route_graph.py \
  /home/thomas-ahle/big-ann-data/deep10m/deep10m_router.rcm \
  /home/thomas-ahle/big-ann-data/deep10m/deep10m_centroid
python3 abba_bench.py abba_deep_centroid_router.json --rounds 3 \
  --out /home/thomas-ahle/big-ann-data/deep10m/abba_centroid_router_vs_hier.json
```

The signed-4-bit displacement artifact used to reject the edge-sketch hot path
can be regenerated without changing the engine:

```sh
python3 build_edge_sketch.py \
  /home/thomas-ahle/big-ann-data/deep10m/graph_layout_gate.cellpair.aligned64.i8bin \
  /home/thomas-ahle/big-ann-data/deep10m/graph_layout_gate.cellpair.graph.u32 \
  /home/thomas-ahle/big-ann-data/deep10m/deep10m_edge_i4.cellpair.eds \
  --base-offset 64 --n 10000000 --d 96 --k 32
```

The complete runtime branch was removed after the 10M test showed that
decode/select overhead exceeded the avoided destination-row gathers.
