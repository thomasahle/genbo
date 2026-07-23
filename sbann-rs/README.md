# sbann search defaults

`sbann run` uses the `balanced` search preset when no search-effort variables are
set. The preset derives a probe ladder and survivor floor from the loaded index's
cell count and the base collection's size and dimension, then chooses graph and
rerank effort for the same recall band.

```bash
# Default: balanced
sbann run BASE.i8bin QUERY.i8bin GT.ibin hierkn apq4 3 65536 8

# Prefer throughput or high recall
SBANN_PRESET=fast     sbann run ...
SBANN_PRESET=accurate sbann run ...

# Convenience selection by desired recall region
SBANN_TARGET_RECALL=0.95 sbann run ...
```

The presets mean:

| Preset | Intended region | Graph effort | Rerank effort |
|---|---|---:|---:|
| `fast` | loose recall / maximum QPS | 1 hop, beam 16 | smallest scaled survivor floor |
| `balanced` | default frontier | 2 hops, beam 24 | medium scaled survivor floor |
| `accurate` | high-recall tail | 3 hops, beam 48 | largest scaled survivor floor |

`SBANN_TARGET_RECALL` selects one of those measured search regions (`<=0.91`:
fast, `<=0.96`: balanced, otherwise accurate). It is not a recall guarantee:
recall still depends on the index, graph, metric, and query distribution. The run
prints the resolved policy as `[SEARCH-PRESET]` and evaluates a four-point probe
ladder around that region.

Expert settings always win over a preset. These overrides remain supported:

- `SBANN_PLIST`: exact comma-separated probe counts.
- `SBANN_TFLOOR`: minimum survivors passed to exact scoring.
- `SBANN_CASCADE_K`: width passed from the int8 stage to float reranking.
- `SBANN_GRAPH_HOPS`, `SBANN_GRAPH_M`, `SBANN_GRAPH_KEDGE`, and
  `SBANN_GRAPH_BESTFIRST`: graph traversal policy.
- `SBANN_GRAPH_BASE` and `SBANN_GRAPH_RANK`: a jointly relabeled physical
  int8 base and original-id-to-physical-id permutation. The supplied
  `SBANN_GRAPH_FILE` must be relabeled by the same permutation.
- `SBANN_GRAPH_BASE_OFFSET`: byte offset of the relabeled base payload
  (normally `64` for cache-line alignment).
- `SBANN_RESIDENT_I8`: copy the active int8 scoring base into aligned,
  transparent-hugepage-eligible memory. With a graph layout active this
  applies to the relabeled base, without retaining a redundant original copy.

Dataset semantics remain explicit: metric selection (`SBANN_IP`), float reranking
and its files, resident-memory flags, and `SBANN_ROUTE_GAMMA` are not guessed.
In particular, the OOD routing gamma that helps text-to-image queries can hurt an
in-distribution workload, so it is intentionally not part of a generic preset.
