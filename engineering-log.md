## 2026-06-23

- Reviewed `paper.tex` for proof bookkeeping. The main `7/9` optimization is algebraically consistent once the certified route-potential accounting is in place.
- The proof needed explicit amplification across `O_c(eta^{-2})` restarted phases. A per-phase constant success probability would otherwise multiply along a route; using failure probability `O(1/D)` only costs an extra `O(log D)` factor.
- The spherical theorem is stated for fixed `(a,b)`, while the reduction uses `(a,b)` varying with `eta`. The constants are uniform because these correlations stay in a compact neighborhood depending only on fixed `c`.
- The reduction theorem should expose polynomial factors in `eta^{-1}` as well as `log n`; for the chosen `eta=L^{-2/9}` these are still lower-order polylogarithmic factors.
- Local TeX install is missing `cleveref.sty`, so `paper.tex` now has a minimal fallback for local compilation.
- Refactored the Euclidean part from an external "certified ball recursion" into a certified score-and-ball tree. The important state is now explicit: score edges, recentering edges, certified radii, query activation by expanded balls, depth, and route-potential invariants.
- The remaining mathematical bottleneck is no longer hidden as a black-box theorem, but the route-potential invariant is still the key thing a fully explicit cap-selection implementation must maintain.
