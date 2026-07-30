# paperdata/ — single source of truth for paper_ood.tex plot data

One CSV per (dataset, system) curve, header `recall,qps`, recall@10 and single-thread QPS.
The tikz/pgfplots figures read these directly:

- linear-axis panels: `\addplot table[col sep=comma, x=recall, y=qps] {paperdata/<name>.csv};`
- log-tail panels (Wikipedia, DEEP, WebVid): same file, `x expr=1-\thisrow{recall}` — the
  transform lives in the plot spec, never in the data.

Update rule: measurements land HERE first (from FINDINGS.md / the campaign ledger), then the
PDF is recompiled and the rendered pages are visually inspected (pdftoppm -> read the PNG)
before committing. Do not put coordinates inline in the .tex.

Provenance: every `*_ours.csv` is materialized by
`sbann-rs/experiments/materialize_uniform_frontiers.py` from
`uniform_frontier_results.json`. The 2026-07-29 campaign uses each full official
query set, one thread, best of five repetitions, one highest-work warmup, and the
same exact plan in forward/reverse order in one loaded process. A family is accepted
only when mirrored recall is identical and every mirrored QPS pair differs by at
most 5%; the latest accepted run is used intact, never stitched pointwise across
windows. Exact settings and accepted-run metadata are in
`uniform_frontier_manifest.csv`.

DEEP's fixed-round walk and cascade are intentionally separate files and plot
series; there is no interpolating segment across their unmeasured join. Its legacy
isolated 0.9987 row is quarantined because the current corrected-fp16/layout stack
reproduces a 0.9980 plateau for probes 768--1024. Baseline curves remain the audited
P335--P342 data: wiki/MSTuring/DEEP Roar curves are fair re-measures (d=104 pad fix,
clean GT, quiet windows); `wiki35m_roar_full` is the build-ineligible 6.5h build,
and `wiki35m_roar_budget` the gate-eligible L=50/250k-training-query build.
