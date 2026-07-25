# paperdata/ — single source of truth for paper_ood.tex plot data

One CSV per (dataset, system) curve, header `recall,qps`, recall@10 and single-thread QPS.
The tikz/pgfplots figures read these directly:

- linear-axis panels: `\addplot table[col sep=comma, x=recall, y=qps] {paperdata/<name>.csv};`
- log-tail panels (Wikipedia, DEEP, WebVid): same file, `x expr=1-\thisrow{recall}` — the
  transform lives in the plot spec, never in the data.

Update rule: measurements land HERE first (from FINDINGS.md / the campaign ledger), then the
PDF is recompiled and the rendered pages are visually inspected (pdftoppm -> read the PNG)
before committing. Do not put coordinates inline in the .tex.

Provenance: values digit-verified against FINDINGS P335–P342 (5-agent cross-check, 2026-07-21);
the DEEP curve was rebuilt from P348–P353 artifacts (sbann-rs/experiments/*.json,
deep_low_confirm_*.json), audit-corrected in P355, and refreshed by the full-2k
fixed-round strict pairs in P356 (`deep_round_walk_results.json`; the 0.9032 row
is the direct-Roar best 8333); wiki/msturing/deep Roar curves are our fair re-measures
(d=104 pad fix, clean GT, quiet windows); `wiki35m_roar_full` is the build-ineligible 6.5h
build, `wiki35m_roar_budget` the gate-eligible L=50/250k-train-query build.
