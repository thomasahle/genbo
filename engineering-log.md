## 2026-06-23

- Reviewed `paper.tex` for proof bookkeeping. The main `7/9` optimization is algebraically consistent once the restart reduction is accepted as a black-box AR15-style theorem.
- The proof needed explicit amplification across `O_c(eta^{-2})` restarted phases. A per-phase constant success probability would otherwise multiply along a route; using failure probability `O(1/D)` only costs an extra `O(log D)` factor.
- The spherical theorem is stated for fixed `(a,b)`, while the reduction uses `(a,b)` varying with `eta`. The constants are uniform because these correlations stay in a compact neighborhood depending only on fixed `c`.
- The reduction theorem should expose polynomial factors in `eta^{-1}` as well as `log n`; for the chosen `eta=L^{-2/9}` these are still lower-order polylogarithmic factors.
- Local TeX install is missing `cleveref.sty`, so `paper.tex` now has a minimal fallback for local compilation.
