## 2026-06-23

- Reviewed `paper.tex` for proof bookkeeping. The main `7/9` optimization is algebraically consistent once the certified route-potential accounting is in place.
- The proof needed explicit amplification across `O_c(eta^{-2})` restarted phases. A per-phase constant success probability would otherwise multiply along a route; using failure probability `O(1/D)` only costs an extra `O(log D)` factor.
- The spherical theorem is stated for fixed `(a,b)`, while the reduction uses `(a,b)` varying with `eta`. The constants are uniform because these correlations stay in a compact neighborhood depending only on fixed `c`.
- The reduction theorem should expose polynomial factors in `eta^{-1}` as well as `log n`; for the chosen `eta=L^{-2/9}` these are still lower-order polylogarithmic factors.
- Local TeX install is missing `cleveref.sty`, so `paper.tex` now has a minimal fallback for local compilation.
- Refactored the Euclidean part from an external "certified ball recursion" into a certified score-and-ball tree. The important state is now explicit: score edges, recentering edges, certified radii, query activation by expanded balls, depth, and route-potential invariants.
- The remaining mathematical bottleneck is no longer hidden as a black-box theorem, but the route-potential invariant is still the key thing a fully explicit cap-selection implementation must maintain.

## 2026-06-24

- Reoriented `paper.tex` away from the restart-loss baseline and toward the polylogarithmic target in `notes2.md`.
- Replaced the old spherical primitive with continuation-mass pruning. The expected-work and far-candidate bounds are proved exactly; near-pair survival is now an explicit one-sided spherical survival assumption.
- Replaced route-depth restart accounting with posterior first-hit and radius Bellman cancellation. The exact threshold is `tau=lambda^2`, which turns the recursive child charge into `alpha F(parent)`.
- Added the cohort-or-leaf assumption as the central remaining coupling theorem between ordinary paths and detector cohorts.
- Split the spherical survival gap into a proved good-leaf first moment and a remaining one-sided LCA second-moment assumption. The first moment uses the tilted measure, the endpoint unit square, and a Gaussian-bridge union bound for the continuation barrier.
- Refined the spherical witness set to "tame good" leaves with an analytical upper barrier on the common score coordinate. This upper barrier is not enforced by the algorithm, but it is needed to make the LCA second-moment target plausible by preventing prolific high shared prefixes.
- Promoted the tame-good LCA second moment from an assumption to a lemma. The key cancellation is in the common coordinate: after multiplying the split-depth probability by the ordered pair count `B^{2k-j}`, the main exponential term cancels because the common-coordinate drift has `mu^2=2 log B`; the remaining endpoint-difference density is summable over split depths.
- Added an exponent-improvement ladder to the paper: continuation-mass spherical filtering alone moves the restart-style target from `(log n)^{7/9}` overhead to roughly `(log n)^{2/3}`, one geometric improvement gives `(log n)^{1/2}`, two geometric improvements give `(log n)^{1/3}`, and the Bellman cohort proof removes the independent phase term entirely.
- The draft now has one named probabilistic bottleneck: cohort-or-leaf. This is the right place to focus next, because the exact continuation-work, spherical survival, posterior first-hit, and Bellman cancellation pieces are all written as self-contained lemmas.
