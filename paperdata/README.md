# paperdata/ — single source of truth for paper_ood.tex plot data

One CSV per (dataset, system) curve, header `recall,qps`, recall@10 and single-thread QPS.
The tikz/pgfplots figures read these directly:

- linear-axis panels: `\addplot table[col sep=comma, x=recall, y=qps] {paperdata/<name>.csv};`
- log-tail panels (Wikipedia, DEEP, WebVid): same file, `x expr=1-\thisrow{recall}` — the
  transform lives in the plot spec, never in the data.

Update rule: measurements land HERE first (from FINDINGS.md / the campaign ledger), then the
PDF is recompiled and the rendered pages are visually inspected (pdftoppm -> read the PNG)
before committing. Do not put coordinates inline in the .tex.

Provenance (2026-07-21): all values digit-verified against FINDINGS P335–P342 and the raw
sweep logs by a 5-agent cross-check; wiki/msturing/deep Roar curves are our fair re-measures
(d=104 pad fix, clean GT, quiet windows); `wiki35m_roar_full` is the build-ineligible 6.5h
build, `wiki35m_roar_budget` the gate-eligible L=50/250k-train-query build.
